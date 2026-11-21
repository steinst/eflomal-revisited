//! Priors: eflomal's priors file, makepriors, and the prior
//! tables of one direction that training adds to its Dirichlet hyperparameters.
//!
//! Three layers:
//!
//! * [`PriorSet`]: words of the two corpus sides plus a [`DirTable`] of raw prior counts per
//!   direction, keyed by those words' ids. A priors file ([`PriorSet::parse`]), makepriors
//!   ([`make_priors`]) and a saved model ([`crate::io::Model`]) all produce one.
//! * [`PriorSet::remap`]: the table of one direction in the ids of a corpus to be aligned
//!   (words normalised with the corpus's options, words the corpus does not have dropped).
//! * [`DirPriors`]: the `f32`/`f64` constants the weights use (α + λ, A_e, β + η, B, γ + κ).
//!   Without priors none of this exists and the weights are the plain ones bit for bit; with
//!   all-zero priors they are the same values.

use std::collections::BTreeMap;
use std::io::{self, Write};

use rustc_hash::FxHashMap;

use crate::bitext::{normalize, Bitext, View, NULL_WORD};
use crate::corpus::Vocab;
use crate::counts::{JumpCounts, FERT_CATS, F_MAX, JUMP_BUCKETS, NULL};
use crate::io::Direction;
use crate::model::Hyper;

const UNSET: u32 = u32::MAX;

/// Raw prior counts of one direction, keyed by the ids of a (source words, target words) pair
/// of vocabularies in that direction's orientation; source id 0 is NULL.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirTable {
    /// λ(e, f): `(e, f, count)`, e may be NULL (0), f ≥ 1; sorted by (e, f), keys unique.
    pub lex: Vec<(u32, u32, f64)>,
    /// η(m) by jump m (not clamped).
    pub jump: BTreeMap<i32, f64>,
    /// κ(e, n): `(e ≥ 1, n, count)` (n not clamped); sorted by (e, n), keys unique.
    pub fert: Vec<(u32, u32, f64)>,
}

impl DirTable {
    pub fn is_empty(&self) -> bool {
        self.lex.is_empty() && self.jump.is_empty() && self.fert.is_empty()
    }

    /// Build from unsorted records; duplicate keys add up, zero counts are dropped.
    fn from_records(
        lex: impl IntoIterator<Item = (u32, u32, f64)>,
        jump: impl IntoIterator<Item = (i32, f64)>,
        fert: impl IntoIterator<Item = (u32, u32, f64)>,
    ) -> Self {
        DirTable {
            lex: sum_sorted(lex),
            jump: {
                let mut m = BTreeMap::new();
                for (k, c) in jump {
                    *m.entry(k).or_insert(0.0) += c;
                }
                m.retain(|_, c| *c != 0.0);
                m
            },
            fert: sum_sorted(fert),
        }
    }

    /// The sum of two tables in the same id space.
    pub fn plus(&self, other: &DirTable) -> DirTable {
        DirTable::from_records(
            self.lex.iter().chain(&other.lex).copied(),
            self.jump.iter().chain(&other.jump).map(|(&m, &c)| (m, c)),
            self.fert.iter().chain(&other.fert).copied(),
        )
    }

    /// The lexical records with the two sides swapped (fwd ↔ inv); NULL rows are dropped and
    /// NULL never appears as a target word, so nothing else changes.
    fn flipped_lex(&self) -> Vec<(u32, u32, f64)> {
        sum_sorted(
            self.lex
                .iter()
                .filter(|r| r.0 != NULL)
                .map(|&(e, f, c)| (f, e, c)),
        )
    }

    /// Check ids and counts against vocabulary sizes (including NULL).
    pub fn validate(&self, n_src: usize, n_tgt: usize) -> Result<(), String> {
        let ok = |c: f64| (0.0..=MAX_TABLE_COUNT).contains(&c);
        for &(e, f, c) in &self.lex {
            if e as usize >= n_src || f == NULL || f as usize >= n_tgt || !ok(c) {
                return Err(format!("invalid lexical prior ({e}, {f}, {c})"));
            }
        }
        for (&m, &c) in &self.jump {
            if !ok(c) {
                return Err(format!("invalid jump prior ({m}, {c})"));
            }
        }
        for &(e, n, c) in &self.fert {
            if e == NULL || e as usize >= n_src || !ok(c) {
                return Err(format!("invalid fertility prior ({e}, {n}, {c})"));
            }
        }
        Ok(())
    }
}

fn sum_sorted(v: impl IntoIterator<Item = (u32, u32, f64)>) -> Vec<(u32, u32, f64)> {
    let mut v: Vec<(u32, u32, f64)> = v.into_iter().collect();
    v.sort_by_key(|r| (r.0, r.1));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(v.len());
    for (a, b, c) in v {
        match out.last_mut() {
            Some(l) if l.0 == a && l.1 == b => l.2 += c,
            _ => out.push((a, b, c)),
        }
    }
    out.retain(|r| r.2 != 0.0);
    out
}

/// Words of both sides (id 0 = `<NULL>`) and the prior tables of both directions: `fwd` in
/// (source, target) ids, `inv` in (target, source) ids.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorSet {
    pub source: Vocab,
    pub target: Vocab,
    pub fwd: DirTable,
    pub inv: DirTable,
}

/// Interns words, id 0 = NULL.
struct Interner {
    ids: FxHashMap<Vec<u8>, u32>,
    words: Vec<Vec<u8>>,
}

impl Interner {
    fn new() -> Self {
        Interner {
            ids: FxHashMap::default(),
            words: vec![NULL_WORD.to_vec()],
        }
    }
    fn id(&mut self, w: &[u8]) -> u32 {
        if w == NULL_WORD {
            return NULL;
        }
        if let Some(&id) = self.ids.get(w) {
            return id;
        }
        let id = self.words.len() as u32;
        self.words.push(w.to_vec());
        self.ids.insert(w.to_vec(), id);
        id
    }
    fn vocab(&self) -> Vocab {
        Vocab::from_words(self.words.iter().map(|w| w.as_slice()))
    }
}

/// The largest count of one priors record: the weights are `f32` (max ≈ 3.4e38), and a
/// count far above any real corpus's must not overflow them into NaN.
pub const MAX_PRIOR_COUNT: f64 = 1e25;
/// The largest count a (loaded) table may hold: summed records, with room to spare.
const MAX_TABLE_COUNT: f64 = 1e30;

fn parse_count(s: &[u8]) -> Option<f64> {
    let x: f64 = std::str::from_utf8(s).ok()?.trim().parse().ok()?;
    (x.is_finite() && (0.0..=MAX_PRIOR_COUNT).contains(&x)).then_some(x)
}

/// `c` as parts of at most [`MAX_PRIOR_COUNT`] whose sum is `c` (one part when it fits).
fn split_count(c: f64) -> Vec<f64> {
    if c <= MAX_PRIOR_COUNT {
        return vec![c];
    }
    let n = (c / MAX_PRIOR_COUNT).ceil();
    let mut parts = vec![MAX_PRIOR_COUNT; n as usize - 1];
    parts.push(c - MAX_PRIOR_COUNT * (n - 1.0));
    parts
}

fn parse_int<T: std::str::FromStr>(s: &[u8]) -> Option<T> {
    std::str::from_utf8(s).ok()?.trim().parse().ok()
}

impl PriorSet {
    /// An empty set over the given vocabularies.
    pub fn empty(source: Vocab, target: Vocab) -> Self {
        PriorSet {
            source,
            target,
            fwd: DirTable::default(),
            inv: DirTable::default(),
        }
    }

    /// Parse an eflomal priors file: tab-separated `LEX e f c`, `HMMF m c`,
    /// `HMMR m c`, `FERF e n c`, `FERR f n c`. Empty lines are skipped; a trailing CR is
    /// ignored. `LEX` records go to both directions (fwd as λ(e, f), inv as λ(f, e)); `<NULL>`
    /// as e is fwd's NULL row, as f inv's NULL row. Errors name the 1-based line.
    pub fn parse(text: &[u8]) -> Result<Self, String> {
        let mut src = Interner::new();
        let mut tgt = Interner::new();
        let (mut fl, mut il) = (Vec::new(), Vec::new());
        let (mut fj, mut ij) = (Vec::new(), Vec::new());
        let (mut ff, mut iff) = (Vec::new(), Vec::new());
        for (n, line) in text.split(|&b| b == b'\n').enumerate() {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.is_empty() {
                continue;
            }
            let err = |what: &str| {
                format!(
                    "priors line {}: {what}: {:?}",
                    n + 1,
                    String::from_utf8_lossy(line)
                )
            };
            let f: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            let count = |s: &[u8]| parse_count(s).ok_or_else(|| err("bad count"));
            match f[0] {
                b"LEX" if f.len() == 4 => {
                    let c = count(f[3])?;
                    let (e, t) = (src.id(f[1]), tgt.id(f[2]));
                    if t != NULL {
                        fl.push((e, t, c));
                    }
                    if e != NULL {
                        il.push((t, e, c));
                    }
                }
                b"HMMF" | b"HMMR" if f.len() == 3 => {
                    let m: i64 = parse_int(f[1]).ok_or_else(|| err("bad jump"))?;
                    let m = m.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                    let c = count(f[2])?;
                    if f[0] == b"HMMF" { &mut fj } else { &mut ij }.push((m, c));
                }
                b"FERF" | b"FERR" if f.len() == 4 => {
                    let k: u64 = parse_int(f[2]).ok_or_else(|| err("bad fertility"))?;
                    let k = k.min(u64::from(u32::MAX)) as u32;
                    let c = count(f[3])?;
                    if f[0] == b"FERF" {
                        let e = src.id(f[1]);
                        if e != NULL {
                            ff.push((e, k, c));
                        }
                    } else {
                        let t = tgt.id(f[1]);
                        if t != NULL {
                            iff.push((t, k, c));
                        }
                    }
                }
                b"LEX" | b"HMMF" | b"HMMR" | b"FERF" | b"FERR" => {
                    return Err(err("wrong number of fields"))
                }
                _ => return Err(err("unknown record type")),
            }
        }
        Ok(PriorSet {
            source: src.vocab(),
            target: tgt.vocab(),
            fwd: DirTable::from_records(fl, fj, ff),
            inv: DirTable::from_records(il, ij, iff),
        })
    }

    /// Write as an eflomal priors file: `LEX` from the forward table (the file has one lexical
    /// table for both directions), `FERF`, `FERR`, `HMMF`, `HMMR`; each sorted by key (words by
    /// bytes, numbers numerically), counts in Rust's shortest form (`3`, `2.5`).
    ///
    /// The inverse table's NULL row (`LEX e <NULL> c` records, which have no forward
    /// counterpart) is written too, and a count above [`MAX_PRIOR_COUNT`] (records summed when
    /// parsed) is written as several records of at most that count, which the parser sums back:
    /// reading the output gives the same counts. The inverse table's other
    /// lexical counts cannot be represented (the format has one lexical table).
    pub fn write(&self, w: &mut dyn Write) -> io::Result<()> {
        let (s, t) = (&self.source, &self.target);
        let mut lex: Vec<(&[u8], &[u8], f64)> = self
            .fwd
            .lex
            .iter()
            .map(|&(e, f, c)| (s.word(e), t.word(f), c))
            .chain(
                self.inv
                    .lex
                    .iter()
                    .filter(|r| r.0 == NULL)
                    .map(|&(_, e, c)| (s.word(e), t.word(NULL), c)),
            )
            .collect();
        lex.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        for (e, f, c) in lex {
            for c in split_count(c) {
                w.write_all(b"LEX\t")?;
                w.write_all(e)?;
                w.write_all(b"\t")?;
                w.write_all(f)?;
                writeln!(w, "\t{c}")?;
            }
        }
        for (tag, table, v) in [(&b"FERF"[..], &self.fwd, s), (b"FERR", &self.inv, t)] {
            let mut fert: Vec<(&[u8], u32, f64)> = table
                .fert
                .iter()
                .map(|&(e, n, c)| (v.word(e), n, c))
                .collect();
            fert.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            for (e, n, c) in fert {
                for c in split_count(c) {
                    w.write_all(tag)?;
                    w.write_all(b"\t")?;
                    w.write_all(e)?;
                    writeln!(w, "\t{n}\t{c}")?;
                }
            }
        }
        for (tag, table) in [("HMMF", &self.fwd), ("HMMR", &self.inv)] {
            for (m, c) in &table.jump {
                for c in split_count(*c) {
                    writeln!(w, "{tag}\t{m}\t{c}")?;
                }
            }
        }
        Ok(())
    }

    /// Every count keyed by words: `(table, key, count)` with table `LEX`/`LEXR`/`HMMF`/`HMMR`/
    /// `FERF`/`FERR` (for comparing sets with different ids, e.g. before and after a round trip).
    pub fn word_counts(&self) -> std::collections::BTreeMap<(String, Vec<u8>), f64> {
        let mut m = std::collections::BTreeMap::new();
        let key = |a: &[u8], b: &[u8]| [a, b"\t", b].concat();
        for (name, t, ev, fv) in [
            ("LEX", &self.fwd, &self.source, &self.target),
            ("LEXR", &self.inv, &self.target, &self.source),
        ] {
            for &(e, f, c) in &t.lex {
                m.insert((name.to_owned(), key(ev.word(e), fv.word(f))), c);
            }
        }
        for (name, t, v) in [
            ("FERF", &self.fwd, &self.source),
            ("FERR", &self.inv, &self.target),
        ] {
            for &(e, n, c) in &t.fert {
                m.insert(
                    (name.to_owned(), key(v.word(e), n.to_string().as_bytes())),
                    c,
                );
            }
        }
        for (name, t) in [("HMMF", &self.fwd), ("HMMR", &self.inv)] {
            for (&j, &c) in &t.jump {
                m.insert((name.to_owned(), j.to_string().into_bytes()), c);
            }
        }
        m
    }

    /// The table of direction `d` in the ids of `bitext`: every word is normalised
    /// with the bitext's options for its side (lowercasing, stemming; idempotent for words
    /// that are normalised already) and looked up; records with a word the corpus does not
    /// have are dropped, also from the denominators. Several words may normalise to one.
    pub fn remap(&self, bitext: &Bitext, d: Direction) -> DirTable {
        let o = bitext.options();
        let src_map = map_words(&self.source, &bitext.source.vocab, |w| {
            normalize(w, o.lowercase, o.source_prefix, o.source_suffix).into_owned()
        });
        let tgt_map = map_words(&self.target, &bitext.target.vocab, |w| {
            normalize(w, o.lowercase, o.target_prefix, o.target_suffix).into_owned()
        });
        let (table, ms, mt) = match d {
            Direction::Forward => (&self.fwd, &src_map, &tgt_map),
            Direction::Inverse => (&self.inv, &tgt_map, &src_map),
        };
        let get = |m: &[u32], id: u32| m.get(id as usize).copied().unwrap_or(UNSET);
        DirTable::from_records(
            table.lex.iter().filter_map(|&(e, f, c)| {
                let (e, f) = (get(ms, e), get(mt, f));
                (e != UNSET && f != UNSET && f != NULL).then_some((e, f, c))
            }),
            table.jump.iter().map(|(&m, &c)| (m, c)),
            table.fert.iter().filter_map(|&(e, n, c)| {
                let e = get(ms, e);
                (e != UNSET && e != NULL).then_some((e, n, c))
            }),
        )
    }
}

/// For every id of `from`, the id of the same (normalised) word in `to`, or `UNSET`; id 0
/// (NULL) maps to 0.
fn map_words(from: &Vocab, to: &Vocab, norm: impl Fn(&[u8]) -> Vec<u8>) -> Vec<u32> {
    let mut index: FxHashMap<&[u8], u32> = FxHashMap::default();
    for id in 1..to.len() as u32 {
        index.insert(to.word(id), id);
    }
    (0..from.len() as u32)
        .map(|id| {
            if id == NULL {
                NULL
            } else {
                index
                    .get(norm(from.word(id)).as_slice())
                    .copied()
                    .unwrap_or(UNSET)
            }
        })
        .collect()
}

// ------------------------------------------------------------------------------------------
// makepriors

/// Links `x-y` separated by any ASCII whitespace, in file order, duplicates kept (they count
/// twice in eflomal-makepriors, measured).
fn parse_links(line: &[u8]) -> Option<Vec<(u32, u32)>> {
    let num = |s: &[u8]| -> Option<u32> {
        if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(s).ok()?.parse().ok()
    };
    line.split(|b| b.is_ascii_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            let d = t.iter().position(|&b| b == b'-')?;
            Some((num(&t[..d])?, num(&t[d + 1..])?))
        })
        .collect()
}

/// The prior counts of one direction derived from links, in the view's ids:
/// one lexical count per link, the fertility of every source position with ≥ 1 link, and the
/// jumps of every sentence with a non-empty target side: for each target position with links,
/// in target order, one jump from the previous position to the source position of its first
/// link in file order, after which the position is that of its last link in file order; then
/// one jump to the end position I, also for a sentence with an empty side (measured on
/// eflomal-makepriors, which this reproduces).
/// `text` is Pharaoh in the corpus's source-target order (a ` ||| score` suffix is ignored).
pub fn link_counts(view: &View<'_>, text: &[u8]) -> Result<DirTable, String> {
    link_counts_except(view, text, &[])
}

/// [`link_counts`] without the lines `skip` (sorted 0-based line indices, e.g.
/// [`crate::bitext::Bitext::skipped`]): they contribute nothing, not even the end jump of an
/// empty pair. Training uses this for its saved model, so a pair skipped for length (whose
/// sides are emptied) is not counted as an observation; a genuinely empty input pair still is,
/// as in eflomal-makepriors. A skipped line must have no links.
pub fn link_counts_except(
    view: &View<'_>,
    text: &[u8],
    skip: &[usize],
) -> Result<DirTable, String> {
    let lines: Vec<&[u8]> = {
        let mut v: Vec<&[u8]> = text.split(|&b| b == b'\n').collect();
        if v.last().is_some_and(|l| l.is_empty()) {
            v.pop();
        }
        v
    };
    if lines.len() != view.len() {
        return Err(format!(
            "the alignment has {} lines but the corpus {}",
            lines.len(),
            view.len()
        ));
    }
    let mut lex: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    let mut jump: FxHashMap<i32, f64> = FxHashMap::default();
    let mut fert: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    let mut links: Vec<(u32, u32)> = Vec::new();
    let mut phi: Vec<u32> = Vec::new();
    let mut skip = skip.iter().copied().peekable();
    for (k, line) in lines.iter().enumerate() {
        let (src, tgt) = view.pair(k);
        let body = match line.windows(5).position(|w| w == b" ||| ") {
            Some(cut) => &line[..cut],
            None => line,
        };
        let parsed = parse_links(body)
            .ok_or_else(|| format!("alignment line {}: not Pharaoh links", k + 1))?;
        if skip.next_if_eq(&k).is_some() {
            if !parsed.is_empty() {
                return Err(format!("alignment line {}: links on a skipped pair", k + 1));
            }
            continue;
        }
        links.clear();
        for l in parsed {
            // (view source position, view target position)
            let (vs, vt) = match view.direction {
                Direction::Forward => (l.0, l.1),
                Direction::Inverse => (l.1, l.0),
            };
            if vs as usize >= src.len() || vt as usize >= tgt.len() {
                return Err(format!(
                    "alignment line {}: link {}-{} is outside the sentence pair",
                    k + 1,
                    l.0,
                    l.1
                ));
            }
            links.push((vt, vs));
        }
        phi.clear();
        phi.resize(src.len(), 0);
        for &(vt, vs) in &links {
            *lex.entry((src[vs as usize], tgt[vt as usize]))
                .or_insert(0.0) += 1.0;
            phi[vs as usize] += 1;
        }
        for (i, &n) in phi.iter().enumerate() {
            if n > 0 {
                *fert.entry((src[i], n)).or_insert(0.0) += 1.0;
            }
        }
        // group by target position, keeping the file order within a group
        links.sort_by_key(|&(vt, _)| vt);
        let mut prev = -1i32;
        for g in links.chunk_by(|a, b| a.0 == b.0) {
            *jump.entry(g[0].1 as i32 - prev).or_insert(0.0) += 1.0;
            prev = g[g.len() - 1].1 as i32;
        }
        *jump.entry(src.len() as i32 - prev).or_insert(0.0) += 1.0;
    }
    Ok(DirTable::from_records(
        lex.into_iter().map(|((e, f), c)| (e, f, c)),
        jump,
        fert.into_iter().map(|((e, n), c)| (e, n, c)),
    ))
}

/// eflomal-makepriors: `LEX` from the forward links, `FERF`/`HMMF` from the
/// forward links, `FERR`/`HMMR` from the reverse links (both files in source-target order;
/// they may be the same symmetrised file). Words are the bitext's normalised words.
pub fn make_priors(bitext: &Bitext, fwd: &[u8], rev: &[u8]) -> Result<PriorSet, String> {
    let f = link_counts(&bitext.view(Direction::Forward), fwd).map_err(|e| format!("-f: {e}"))?;
    let r = link_counts(&bitext.view(Direction::Inverse), rev).map_err(|e| format!("-r: {e}"))?;
    let inv = DirTable {
        lex: f.flipped_lex(),
        jump: r.jump,
        fert: r.fert,
    };
    Ok(PriorSet {
        source: bitext.source.vocab.clone(),
        target: bitext.target.vocab.clone(),
        fwd: f,
        inv,
    })
}

// ------------------------------------------------------------------------------------------
// The constants of the weights

/// The prior parts of the weights of one direction, in the view's ids.
#[derive(Debug)]
pub struct DirPriors {
    /// (e, f) → α + λ(e, f) as `f32` (summed in `f64`), only where λ > 0.
    lex: FxHashMap<u64, f32>,
    /// A_e = α·|V_T| + Σ_f λ(e, f) per source id, as `f32` (summed in `f64`).
    den: Vec<f32>,
    /// max_e A_e.
    max_den: f32,
    /// β + η(m) per jump bucket.
    jump_alpha: [f32; JUMP_BUCKETS],
    /// B = Σ_m (β + η(m)).
    jump_den: f32,
    /// γ + κ(e, n), `FERT_CATS` per source id; empty if there are no fertility priors.
    fert: Vec<f64>,
    /// γ.
    fert_alpha: f64,
}

#[inline]
fn lex_key(e: u32, f: u32) -> u64 {
    (u64::from(e) << 32) | u64::from(f)
}

impl DirPriors {
    /// `table` in the ids of a view with `n_src` source ids (NULL included) and
    /// `n_target_words` target words (NULL excluded).
    pub fn new(table: &DirTable, h: &Hyper, n_src: usize, n_target_words: usize) -> Self {
        let mut sums = vec![0.0f64; n_src];
        let mut lex = FxHashMap::default();
        lex.reserve(table.lex.len());
        for &(e, f, c) in &table.lex {
            sums[e as usize] += c;
            lex.insert(lex_key(e, f), (h.lex_alpha + c) as f32);
        }
        let base = h.lex_alpha * n_target_words as f64;
        let den: Vec<f32> = sums.iter().map(|&s| (base + s) as f32).collect();
        let max_den = den.iter().copied().fold(base as f32, f32::max);
        let mut eta = [0.0f64; JUMP_BUCKETS];
        for (&m, &c) in &table.jump {
            eta[JumpCounts::bucket(m)] += c;
        }
        let mut jump_alpha = [0.0f32; JUMP_BUCKETS];
        for (a, &x) in jump_alpha.iter_mut().zip(&eta) {
            *a = (h.jump_alpha + x) as f32;
        }
        let jump_den = (h.jump_alpha * JUMP_BUCKETS as f64 + eta.iter().sum::<f64>()) as f32;
        let fert = if table.fert.is_empty() {
            Vec::new()
        } else {
            let mut v = vec![h.fert_alpha; n_src * FERT_CATS];
            for &(e, n, c) in &table.fert {
                v[e as usize * FERT_CATS + (n as usize).min(F_MAX)] += c;
            }
            v
        };
        DirPriors {
            lex,
            den,
            max_den,
            jump_alpha,
            jump_den,
            fert,
            fert_alpha: h.fert_alpha,
        }
    }

    /// α + λ(e, f), or `alpha` (= α as `f32`) without a record.
    #[inline]
    pub fn lex_alpha(&self, e: u32, f: u32, alpha: f32) -> f32 {
        self.lex.get(&lex_key(e, f)).copied().unwrap_or(alpha)
    }

    #[inline]
    pub fn lex_den(&self, e: u32) -> f32 {
        self.den[e as usize]
    }

    #[inline]
    pub fn jump_alpha(&self, m: i32) -> f32 {
        self.jump_alpha[JumpCounts::bucket(m)]
    }

    pub fn jump_den(&self) -> f32 {
        self.jump_den
    }

    /// Replace the denominators A_e (one per source id).
    pub fn with_den(mut self, den: Vec<f32>) -> Self {
        assert_eq!(den.len(), self.den.len());
        self.max_den = den.iter().copied().fold(0.0, f32::max);
        self.den = den;
        self
    }

    /// max_e A_e.
    pub fn max_den(&self) -> f32 {
        self.max_den
    }

    /// γ + κ(e, n), n clamped.
    #[inline]
    pub fn fert_alpha(&self, e: u32, n: usize) -> f64 {
        if self.fert.is_empty() {
            self.fert_alpha
        } else {
            self.fert[e as usize * FERT_CATS + n.min(F_MAX)]
        }
    }

    pub fn n_lex(&self) -> usize {
        self.lex.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.lex.capacity() * 16 + self.den.capacity() * 4 + self.fert.capacity() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitext::TextOptions;

    fn bitext(src: &str, tgt: &str) -> Bitext {
        Bitext::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), &TextOptions::default()).unwrap()
    }

    fn text(ps: &PriorSet) -> String {
        let mut v = Vec::new();
        ps.write(&mut v).unwrap();
        String::from_utf8(v).unwrap()
    }

    /// A 3-line probe: eflomal-makepriors' output for it,
    /// with the corpus lowercased as eflomal-align reads it.
    #[test]
    fn makepriors_reproduces_eflomal() {
        let b = bitext("a b c d\nb a\nx\n", "A B C\nB2 A2 Z\n\n");
        let ps = make_priors(&b, b"0-0 1-1 1-2 3-1\n0-1 1-0\n\n", b"0-0 2-2\n1-0 0-2\n\n").unwrap();
        let want = "LEX\ta\ta\t1\nLEX\ta\tb2\t1\nLEX\tb\ta2\t1\nLEX\tb\tb\t1\nLEX\tb\tc\t1\n\
                    LEX\td\tb\t1\nFERF\ta\t1\t2\nFERF\tb\t1\t1\nFERF\tb\t2\t1\nFERF\td\t1\t1\n\
                    FERR\ta\t1\t1\nFERR\tb2\t1\t1\nFERR\tc\t1\t1\nFERR\tz\t1\t1\n\
                    HMMF\t-2\t1\nHMMF\t-1\t1\nHMMF\t1\t2\nHMMF\t2\t3\nHMMF\t3\t1\n\
                    HMMR\t-2\t1\nHMMR\t1\t3\nHMMR\t2\t1\nHMMR\t3\t2\n";
        assert_eq!(text(&ps), want);
    }

    /// Jumps of many-to-many links, measured on eflomal-makepriors: per target word the
    /// jump goes to its first link in file order and continues from its last.
    #[test]
    fn jumps_of_many_to_many_links() {
        let b = bitext("a b c d e\na b c d e\na b c d e\n", "x y\nx y z\nx y\n");
        let jumps = |links: &str| {
            let t = format!("{links}\n\n\n");
            let f = link_counts(&b.view(Direction::Forward), t.as_bytes()).unwrap();
            f.jump
                .into_iter()
                .map(|(m, c)| (m, c as u32))
                .collect::<Vec<_>>()
        };
        // + the two lines without links: one jump 5 - (-1) = 6 each
        assert_eq!(jumps("0-0 2-0 1-1"), [(-1, 1), (1, 1), (4, 1), (6, 2)]);
        assert_eq!(jumps("2-0 0-0 1-1"), [(1, 1), (3, 1), (4, 1), (6, 2)]);
        assert_eq!(jumps("0-0 3-0"), [(1, 1), (2, 1), (6, 2)]);
        assert_eq!(jumps("3-0 0-0"), [(4, 1), (5, 1), (6, 2)]);
        assert_eq!(jumps("1-0 1-1"), [(0, 1), (2, 1), (4, 1), (6, 2)]);
        let rev = |links: &str| {
            let t = format!("{links}\n\n\n");
            let r = link_counts(&b.view(Direction::Inverse), t.as_bytes()).unwrap();
            r.jump
                .into_iter()
                .map(|(m, c)| (m, c as u32))
                .collect::<Vec<_>>()
        };
        // + lines 2 and 3 without links: jumps 4 and 3 to their end positions J = 3, 2
        assert_eq!(
            rev("0-0 2-0 1-1"),
            [(-1, 1), (1, 2), (2, 1), (3, 1), (4, 1)]
        );
        // the second line has 3 target words
        let t = "\n4-0 1-1 2-1 0-2\n\n";
        let f = link_counts(&b.view(Direction::Forward), t.as_bytes()).unwrap();
        let j: Vec<_> = f.jump.into_iter().map(|(m, c)| (m, c as u32)).collect();
        // two sentences without links: jump 5 - (-1) = 6 each
        assert_eq!(j, [(-3, 1), (-2, 1), (5, 2), (6, 2)]);
        // empty sides (measured): every sentence ends with a jump to I, even I = J = 0
        let e = bitext(
            "a b c


", "
x y

",
        );
        let t = link_counts(
            &e.view(Direction::Forward),
            b"


",
        )
        .unwrap();
        assert_eq!(t.jump.into_iter().collect::<Vec<_>>(), [(1, 2.0), (4, 1.0)]);
        let t = link_counts(
            &e.view(Direction::Inverse),
            b"


",
        )
        .unwrap();
        assert_eq!(t.jump.into_iter().collect::<Vec<_>>(), [(1, 2.0), (3, 1.0)]);
    }

    /// The writer kept neither the inverse NULL row nor merged counts above
    /// the per-record limit; reading its output now gives the same counts.
    #[test]
    fn write_then_parse_keeps_every_count() {
        for text in [
            &b"LEX\tword\t<NULL>\t7\n"[..],
            b"LEX\ta\tb\t1e25\nLEX\ta\tb\t1e25\nLEX\ta\tb\t3\n",
            b"HMMF\t2\t1e25\nHMMF\t2\t1e25\nFERR\tx\t1\t1e25\nFERR\tx\t1\t5e24\n",
            b"LEX\t<NULL>\tb\t2\nLEX\ta\t<NULL>\t1\nLEX\ta\tb\t4\nHMMR\t-3\t1\n",
        ] {
            let ps = PriorSet::parse(text).unwrap();
            let mut out = Vec::new();
            ps.write(&mut out).unwrap();
            assert!(!out.is_empty());
            let back = PriorSet::parse(&out).expect("the writer's output parses");
            let (a, b) = (ps.word_counts(), back.word_counts());
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (k, x) in &a {
                assert!((x - b[k]).abs() <= 1e-12 * x, "{k:?}");
            }
        }
        assert_eq!(split_count(2e25), [1e25, 1e25]);
        assert_eq!(split_count(3.0), [3.0]);
    }

    #[test]
    fn skipped_lines_are_not_counted() {
        let b = bitext("a\n\na\n", "x\n\nx\n");
        let v = b.view(Direction::Forward);
        let jumps = |t: DirTable| t.jump.into_iter().collect::<Vec<_>>();
        // the genuine empty pair (line 1) adds its end jump +1, as eflomal-makepriors does
        assert_eq!(jumps(link_counts(&v, b"0-0\n\n0-0\n").unwrap()), [(1, 5.0)]);
        // unless it is listed as skipped
        let t = link_counts_except(&v, b"0-0\n\n0-0\n", &[1]).unwrap();
        assert_eq!(jumps(t), [(1, 4.0)]);
        let t = link_counts_except(&v, b"\n\n0-0\n", &[0, 1]).unwrap();
        assert_eq!(t.lex, [(1, 1, 1.0)]);
        assert_eq!(jumps(t), [(1, 2.0)]);
        assert!(link_counts_except(&v, b"0-0\n\n0-0\n", &[2])
            .unwrap_err()
            .contains("skipped"));
    }

    #[test]
    fn link_errors_are_reported() {
        let b = bitext("a b\n", "x\n");
        let v = b.view(Direction::Forward);
        assert!(link_counts(&v, b"0-0\n0-0\n")
            .unwrap_err()
            .contains("lines"));
        assert!(link_counts(&v, b"2-0\n").unwrap_err().contains("outside"));
        assert!(link_counts(&v, b"0-1\n").unwrap_err().contains("outside"));
        assert!(link_counts(&v, b"zz\n").unwrap_err().contains("line 1"));
        // scores and no trailing newline are fine; duplicates count twice
        assert_eq!(link_counts(&v, b"1-0 ||| 3.5").unwrap().lex, [(2, 1, 1.0)]);
        assert_eq!(link_counts(&v, b"1-0\t1-0").unwrap().lex, [(2, 1, 2.0)]);
    }

    #[test]
    fn parse_write_and_errors() {
        let ps = PriorSet::parse(
            b"LEX\ta\tx\t2\r\nLEX\t<NULL>\tx\t1.5\nLEX\ta\t<NULL>\t3\n\nHMMF\t-2\t1.25488e+06\n\
              HMMR\t500\t1\nFERF\ta\t2\t4\nFERR\tx\t40\t1\nLEX\ta\tx\t1\n",
        )
        .unwrap();
        // fwd: (a, x) 2 + 1, (NULL, x) 1.5; inv: (x, a) 3, (NULL, a) 3
        assert_eq!(ps.fwd.lex, [(0, 1, 1.5), (1, 1, 3.0)]);
        assert_eq!(ps.inv.lex, [(0, 1, 3.0), (1, 1, 3.0)]);
        assert_eq!(ps.fwd.jump.get(&-2), Some(&1254880.0));
        assert_eq!(ps.inv.jump.get(&500), Some(&1.0));
        assert_eq!(ps.fwd.fert, [(1, 2, 4.0)]);
        assert_eq!(ps.inv.fert, [(1, 40, 1.0)]);
        // writing and reading again gives the same tables (LEX from the forward table only)
        let again = PriorSet::parse(text(&ps).as_bytes()).unwrap();
        assert_eq!(again.fwd, ps.fwd);
        assert_eq!(again.inv.jump, ps.inv.jump);
        assert_eq!(again.inv.fert, ps.inv.fert);
        for (bad, what) in [
            ("LEX\ta\tb\n", "line 1: wrong number"),
            ("LEX\ta\tb\t-1\n", "bad count"),
            ("LEX\ta\tb\tnan\n", "bad count"),
            ("LEX\ta\tb\t2e25\n", "bad count"),
            ("HMMF\t1\tinf\n", "bad count"),
            ("\nHMMF\tx\t1\n", "line 2: bad jump"),
            ("FERF\ta\t-1\t1\n", "bad fertility"),
            ("FOO\t1\n", "unknown record type"),
        ] {
            let e = PriorSet::parse(bad.as_bytes()).unwrap_err();
            assert!(e.contains(what), "{bad:?}: {e}");
        }
    }

    #[test]
    fn remap_normalises_and_drops_unknown_words() {
        let ps = PriorSet::parse(
            b"LEX\tHouse\tMAISON\t5\nLEX\tzzz\tmaison\t7\nLEX\thouse\tqqq\t9\n\
                                   FERF\tHOUSE\t1\t2\nFERR\tMaison\t1\t3\nHMMF\t1\t4\n",
        )
        .unwrap();
        let b = bitext("the house\n", "la maison\n");
        let f = ps.remap(&b, Direction::Forward);
        let (house, maison) = (2, 2);
        assert_eq!(f.lex, [(house, maison, 5.0)]);
        assert_eq!(f.fert, [(house, 1, 2.0)]);
        assert_eq!(f.jump.get(&1), Some(&4.0));
        let i = ps.remap(&b, Direction::Inverse);
        assert_eq!(i.lex, [(maison, house, 5.0)]);
        assert_eq!(i.fert, [(maison, 1, 3.0)]);
        // stemming applies to the priors' words as to the corpus
        let o = TextOptions {
            source_prefix: Some(3),
            ..TextOptions::default()
        };
        let b = Bitext::from_pair_bytes(b"the houses\n", b"la maison\n", &o).unwrap();
        assert_eq!(
            ps.remap(&b, Direction::Forward).lex,
            [(2, 2, 5.0), (2, 2, 9.0)]
                .into_iter()
                .take(1)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn dir_priors_without_records_are_the_plain_constants() {
        use crate::model::Params;
        let h = Hyper::default();
        let pr = DirPriors::new(&DirTable::default(), &h, 10, 7);
        let p = Params::new(&h, 7);
        for e in 0..10 {
            assert_eq!(pr.lex_den(e).to_bits(), p.lex_den.to_bits());
            assert_eq!(
                pr.lex_alpha(e, 3, p.lex_alpha).to_bits(),
                p.lex_alpha.to_bits()
            );
            assert_eq!(pr.fert_alpha(e, 4), p.fert_alpha);
        }
        for m in -70..70 {
            assert_eq!(pr.jump_alpha(m).to_bits(), p.jump_alpha.to_bits());
        }
        assert_eq!(pr.jump_den().to_bits(), p.jump_den.to_bits());
        // with records: sums in f64, clamped buckets
        let t = DirTable::from_records(
            [(1, 2, 3.0), (1, 4, 1.0)],
            [(100, 2.0), (64, 1.0), (-3, 1.0)],
            [(1, 40, 2.0)],
        );
        let pr = DirPriors::new(&t, &h, 10, 7);
        assert_eq!(pr.lex_alpha(1, 2, 0.0), 3.001f32);
        assert_eq!(pr.lex_den(1), (0.007 + 4.0) as f32);
        assert_eq!(pr.jump_alpha(64), 3.5);
        assert_eq!(pr.jump_alpha(-3), 1.5);
        assert_eq!(pr.jump_den(), (0.5 * 129.0 + 4.0) as f32);
        assert_eq!(pr.fert_alpha(1, 31), 3.0);
        assert_eq!(pr.fert_alpha(2, 31), 1.0);
    }
}
