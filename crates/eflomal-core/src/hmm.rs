//! The deterministic decode of `align --fast`: no sampling, the
//! exact posterior marginals of the Model-2 HMM (eflomal's model without fertility) with fixed parameters taken from a
//! saved model alone ([`FastModel`]), in the model's own vocabulary:
//!
//! * θ̂_e(f) = (α + λ(e, f)) / A_e with A_e = α + Σ_f λ(e, f) · c_f / T, the row's counts
//!   weighted by how often each target word is linked (c_f = Σ_e λ(e, f), T = Σ_f c_f), over
//!   all of the model's words ([`frequency_weighted_den`]);
//! * a target word the model does not know has θ̂_e(f) = α / A_e (a word never linked to e),
//!   and a source word it does not know the empty row, θ̂_e(f) = 1 for every f (the same for
//!   every candidate f, so it ranks candidates by NULL and jumps only);
//! * ψ̂(m) = (β + η(m)) / B;
//!
//! so the links of a sentence pair depend on that pair and the model only, never on the other
//! pairs of the call, their order or how the text is split into calls.
//!
//! NULL as in the generative story: a NULL-aligned token does not move the position, and
//! after the last token one jump goes to the end position I. The states of target position j
//! are "aligned to i" (i = 0 … I−1) and "NULL, last position p" (p = −1 … I−1); the future of
//! both depends only on the last position, so the backward pass has I + 1 values per j.
//! O(I² J) per sentence, `f64`, each forward and backward vector rescaled to sum 1 (the
//! per-position posteriors are normalised, so the scale factors cancel).
//!
//! The link of a target token is the argmax over {0 … I−1, ε} of its posterior, ties to the
//! lowest position, ε counting as position I (as the sampler's decode).

use std::sync::OnceLock;

use rustc_hash::FxHashMap;

use rayon::prelude::*;

use crate::bitext::{Bitext, View};
use crate::corpus::Vocab;
use crate::counts::{MAX_JUMP, NULL};
use crate::io::{Direction, Model};
use crate::model::Params;
use crate::priors::{DirPriors, DirTable};

/// The fixed parameters the HMM reads: θ̂ by (source id, target id) of the view, ψ̂ by jump.
pub trait Emission: Sync {
    fn theta(&self, e: u32, f: u32) -> f64;
    /// ψ̂(m) (normalised).
    fn psi(&self, m: i32) -> f64;
    fn p_null(&self) -> f64;
    fn p_word(&self) -> f64;
}

/// θ̂ and ψ̂ from a direction's [`Params`] with priors (the ids are those of the priors).
impl Emission for Params {
    fn theta(&self, e: u32, f: u32) -> f64 {
        f64::from(self.lex_alpha_of(e, f)) / f64::from(self.lex_den_of(e))
    }
    fn psi(&self, m: i32) -> f64 {
        f64::from(self.jump_alpha_of(m)) / f64::from(self.jump_den)
    }
    fn p_null(&self) -> f64 {
        f64::from(self.p_null)
    }
    fn p_word(&self) -> f64 {
        f64::from(self.p_word)
    }
}

/// A saved model indexed once for fast decoding: the word → id maps of its vocabularies and,
/// per direction (built on first use), [`Params`] over the model's own ids with one extra
/// source id for unknown words. Immutable after construction, so one instance serves
/// concurrent calls.
#[derive(Default)]
pub struct FastModel {
    index: OnceLock<[FxHashMap<Box<[u8]>, u32>; 2]>,
    dirs: [OnceLock<Option<Params>>; 2],
}

/// The row normalisers of fast decoding: A_e = α + Σ_f λ(e, f) · c_f / T, where c_f = Σ_e λ(e, f)
/// is how often target word f is linked and T = Σ_f c_f; one per source id (`n_src`, the
/// unknown-word row included).
///
/// Selected on held-out data: a row's mass is
/// weighted by how likely each target word is to occur at all. It is the parameter-free limit
/// (B → 0, a common factor cancels in the posterior) of normalising each row over the expected
/// vocabulary of a virtual batch of B training sentences. Measured, grow-diag-final-and F1 against
/// the training-total normalisers: en-fr +1.54 / +1.94 (auto / words), ro-en +1.62 / +1.78,
/// and on the English–Icelandic pair held out from the selection +3.82 / +3.81 (95 % bootstrap
/// intervals [+3.25, +4.39] / [+3.23, +4.40]); B from 0.1 to 4 gave the same results.
fn frequency_weighted_den(table: &DirTable, alpha: f64, n_src: usize, n_f: usize) -> Vec<f32> {
    let mut col = vec![0.0f64; n_f];
    for &(_, f, c) in &table.lex {
        col[f as usize] += c;
    }
    let t = col.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let mut den = vec![alpha; n_src];
    for &(e, f, c) in &table.lex {
        den[e as usize] += c * col[f as usize] / t;
    }
    den.into_iter().map(|x| x as f32).collect()
}

/// Marks a target word the model does not know (no lexical record has it).
const UNKNOWN: u32 = u32::MAX;

fn word_index(v: &Vocab) -> FxHashMap<Box<[u8]>, u32> {
    (1..v.len() as u32)
        .map(|id| (v.word(id).into(), id))
        .collect()
}

impl FastModel {
    fn index(&self, model: &Model) -> &[FxHashMap<Box<[u8]>, u32>; 2] {
        self.index
            .get_or_init(|| [word_index(&model.source), word_index(&model.target)])
    }

    /// The parameters of direction `d` (`None` if the model does not have it).
    fn params(&self, model: &Model, d: Direction) -> Option<&Params> {
        let k = usize::from(d == Direction::Inverse);
        self.dirs[k]
            .get_or_init(|| {
                let (table, ev, fv) = match d {
                    Direction::Forward => (model.fwd.as_ref()?, &model.source, &model.target),
                    Direction::Inverse => (model.inv.as_ref()?, &model.target, &model.source),
                };
                let h = &model.options.hyper;
                let n_tw = fv.len() - 1;
                // one more source id: the empty row of the words the model does not know
                let pr = DirPriors::new(table, h, ev.len() + 1, n_tw);
                let pr = pr.with_den(frequency_weighted_den(
                    table,
                    h.lex_alpha,
                    ev.len() + 1,
                    fv.len(),
                ));
                Some(Params::with_priors(h, n_tw, Some(std::sync::Arc::new(pr))))
            })
            .as_ref()
    }

    /// Decode direction `d` of `bitext` (read with the model's text options): Pharaoh text in
    /// the corpus's source-target order, one line per pair. `None` if the model lacks `d`.
    pub fn decode(&self, model: &Model, bitext: &Bitext, d: Direction) -> Option<Vec<u8>> {
        let p = self.params(model, d)?;
        let [si, ti] = self.index(model);
        let view = bitext.view(d);
        let (ei, fi, n_e) = match d {
            Direction::Forward => (si, ti, model.source.len()),
            Direction::Inverse => (ti, si, model.target.len()),
        };
        // the view's ids → the model's (unknown source words → the extra empty row)
        let map = |v: &Vocab, index: &FxHashMap<Box<[u8]>, u32>, unknown: u32| -> Vec<u32> {
            (0..v.len() as u32)
                .map(|id| {
                    if id == NULL {
                        NULL
                    } else {
                        index.get(v.word(id)).copied().unwrap_or(unknown)
                    }
                })
                .collect()
        };
        let em = map(&view.src.vocab, ei, n_e as u32);
        let fm = map(&view.tgt.vocab, fi, UNKNOWN);
        Some(decode(
            &view,
            &Mapped {
                p,
                em: &em,
                fm: &fm,
            },
        ))
    }
}

/// [`Params`] over the model's ids, read through the view → model id maps.
struct Mapped<'a> {
    p: &'a Params,
    em: &'a [u32],
    fm: &'a [u32],
}

impl Emission for Mapped<'_> {
    #[inline]
    fn theta(&self, e: u32, f: u32) -> f64 {
        self.p.theta(self.em[e as usize], self.fm[f as usize])
    }
    fn psi(&self, m: i32) -> f64 {
        self.p.psi(m)
    }
    fn p_null(&self) -> f64 {
        self.p.p_null()
    }
    fn p_word(&self) -> f64 {
        self.p.p_word()
    }
}

/// Sentences per parallel work item (fixed; the output does not depend on the thread count).
const CHUNK: usize = 32;

/// Decode one direction; Pharaoh text in the corpus's source-target order, one line per pair.
pub fn decode<E: Emission>(view: &View<'_>, p: &E) -> Vec<u8> {
    let k_n = view.len();
    let parts: Vec<Vec<u8>> = (0..k_n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let mut out = Vec::new();
            let mut s = Scratch::default();
            for k in c * CHUNK..((c + 1) * CHUNK).min(k_n) {
                crate::cancel::check();
                let (src, tgt) = view.pair(k);
                let links = sentence(src, tgt, p, &mut s);
                let mut v: Vec<(u32, u32)> = links
                    .iter()
                    .enumerate()
                    .filter(|&(_, &a)| a < src.len())
                    .map(|(j, &a)| match view.direction {
                        Direction::Forward => (a as u32, j as u32),
                        Direction::Inverse => (j as u32, a as u32),
                    })
                    .collect();
                v.sort_unstable();
                crate::symmetrize::format_links(&v, &mut out);
                out.push(b'\n');
            }
            out
        })
        .collect();
    parts.concat()
}

#[derive(Default)]
struct Scratch {
    /// θ per (j, i) for i in 0 … I (I = NULL), row-major.
    theta: Vec<f64>,
    /// ψ(m) for m = −I … I+1 at index m + I.
    psi: Vec<f64>,
    /// Forward: per j, A(i) for i < I then N(p) for p = −1 … I−1 (index I + p + 1).
    fwd: Vec<f64>,
    /// Backward: per j, B(p) for p = −1 … I−1 (index p + 1).
    bwd: Vec<f64>,
    g: Vec<f64>,
    /// Prefix sums for the clamped-jump tails.
    pre: Vec<f64>,
    /// Posterior per (j, i), i = I for NULL, each row normalised.
    post: Vec<f64>,
    best: Vec<usize>,
}

fn normalize(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 0.0 && s.is_finite() {
        let inv = 1.0 / s;
        for x in v {
            *x *= inv;
        }
    }
}

/// The argmax position of every target token (I = NULL).
fn sentence<'s, E: Emission>(src: &[u32], tgt: &[u32], p: &E, s: &'s mut Scratch) -> &'s [usize] {
    let (i_len, j_len) = (src.len(), tgt.len());
    s.best.clear();
    if j_len == 0 {
        return &s.best;
    }
    if i_len == 0 {
        s.best.resize(j_len, 0);
        return &s.best;
    }
    let w = i_len + 1;
    let p_null = p.p_null();
    let p_word = p.p_word();
    // parameters
    s.theta.clear();
    for &f in tgt {
        for e in src.iter().copied().chain(std::iter::once(NULL)) {
            s.theta.push(p.theta(e, f));
        }
    }
    s.psi.clear();
    for m in -(i_len as i32)..=(i_len as i32 + 1) {
        s.psi.push(p.psi(m));
    }
    let psi = |m: i32| s.psi[(m + i_len as i32) as usize];
    // forward: row j has I aligned states then I + 1 NULL states
    let fw = 2 * i_len + 1;
    s.fwd.clear();
    s.fwd.resize(j_len * fw, 0.0);
    s.g.clear();
    s.g.resize(i_len + 1, 0.0); // G(p), index p + 1
    s.g[0] = 1.0; // the start: last position −1
                  // ψ is constant for jumps m ≥ MAX_JUMP and m ≤ −MAX_JUMP (clamped buckets): those terms are
                  // summed from prefix sums (the "tails"), the |m| < MAX_JUMP window term by term. Without
                  // tails (I < MAX_JUMP) the sums are the plain loops, in the same order.
    let (psi_hi, psi_lo) = (p.psi(MAX_JUMP), p.psi(-MAX_JUMP));
    let reach = MAX_JUMP as usize;
    s.pre.clear();
    s.pre.resize(i_len + 2, 0.0);
    for j in 0..j_len {
        crate::cancel::check(); // bounded polling in long sentences
        let th = &s.theta[j * w..(j + 1) * w];
        let row = &mut s.fwd[j * fw..(j + 1) * fw];
        // pre[k] = Σ_{pi < k} g[pi]
        for (k, &g) in s.g.iter().enumerate() {
            s.pre[k + 1] = s.pre[k] + g;
        }
        for i in 0..i_len {
            // window: pi − 1 = p with |i − p| < MAX_JUMP  ⇔  pi in [i + 2 − MAX_JUMP, i + MAX_JUMP]
            let lo = (i + 2).saturating_sub(reach);
            let hi = (i + reach).min(i_len);
            let mut acc = 0.0;
            for (pi, &g) in s.g.iter().enumerate().take(hi + 1).skip(lo) {
                if g != 0.0 {
                    acc += g * psi(i as i32 - (pi as i32 - 1));
                }
            }
            if lo > 0 {
                acc += s.pre[lo] * psi_hi; // m ≥ MAX_JUMP
            }
            if hi < i_len {
                acc += (s.pre[i_len + 1] - s.pre[hi + 1]) * psi_lo; // m ≤ −MAX_JUMP
            }
            row[i] = p_word * th[i] * acc;
        }
        for (pi, &g) in s.g.iter().enumerate() {
            row[i_len + pi] = p_null * th[i_len] * g;
        }
        normalize(row);
        // G for the next step: aligned to p or NULL with last p
        s.g[0] = row[i_len];
        for i in 0..i_len {
            s.g[i + 1] = row[i] + row[i_len + i + 1];
        }
    }
    // backward
    s.bwd.clear();
    s.bwd.resize(j_len * w, 0.0);
    {
        let last = &mut s.bwd[(j_len - 1) * w..j_len * w];
        for (pi, x) in last.iter_mut().enumerate() {
            *x = psi(i_len as i32 - (pi as i32 - 1));
        }
        normalize(last);
    }
    for j in (0..j_len - 1).rev() {
        crate::cancel::check();
        let th = &s.theta[(j + 1) * w..(j + 2) * w];
        let (head, tail) = s.bwd.split_at_mut((j + 1) * w);
        let next = &tail[..w];
        let row = &mut head[j * w..];
        // pre[k] = Σ_{i < k} θ_i · next[i + 1]
        for i in 0..i_len {
            s.pre[i + 1] = s.pre[i] + th[i] * next[i + 1];
        }
        for (pi, x) in row.iter_mut().enumerate() {
            let pp = pi as i32 - 1;
            // window: |i − pp| < MAX_JUMP  ⇔  i in [pi − MAX_JUMP, pi + MAX_JUMP − 2]
            let lo = pi.saturating_sub(reach);
            let hi = (pi + reach - 2).min(i_len.saturating_sub(1));
            let mut acc = p_null * th[i_len] * next[pi];
            for i in lo..=hi {
                acc += p_word * psi(i as i32 - pp) * th[i] * next[i + 1];
            }
            if hi + 1 < i_len {
                acc += p_word * psi_hi * (s.pre[i_len] - s.pre[hi + 1]); // m ≥ MAX_JUMP
            }
            if lo > 0 {
                acc += p_word * psi_lo * s.pre[lo]; // m ≤ −MAX_JUMP
            }
            *x = acc;
        }
        normalize(row);
    }
    // posteriors and argmax
    s.post.clear();
    for j in 0..j_len {
        let a = &s.fwd[j * fw..(j + 1) * fw];
        let b = &s.bwd[j * w..(j + 1) * w];
        let at = s.post.len();
        s.post.extend((0..i_len).map(|i| a[i] * b[i + 1]));
        s.post
            .push((0..=i_len).map(|pi| a[i_len + pi] * b[pi]).sum::<f64>());
        let row = &mut s.post[at..];
        normalize(row);
        let mut best = 0;
        for i in 1..=i_len {
            if row[i] > row[best] {
                best = i;
            }
        }
        s.best.push(best);
    }
    &s.best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Hyper;

    /// Brute force: enumerate every alignment of a tiny sentence and sum the joint; the
    /// normalised posterior per (j, i), i = I for NULL.
    fn brute(src: &[u32], tgt: &[u32], p: &Params) -> Vec<f64> {
        let i_len = src.len();
        let j_len = tgt.len();
        let th = |e: u32, f: u32| f64::from(p.lex_alpha_of(e, f)) / f64::from(p.lex_den_of(e));
        let psi = |m: i32| f64::from(p.jump_alpha_of(m)) / f64::from(p.jump_den);
        let mut post = vec![vec![0.0f64; i_len + 1]; j_len];
        let n = (i_len + 1).pow(j_len as u32);
        for code in 0..n {
            let mut a = Vec::with_capacity(j_len);
            let mut c = code;
            for _ in 0..j_len {
                a.push(c % (i_len + 1));
                c /= i_len + 1;
            }
            let mut prob = 1.0;
            let mut last = -1i32;
            for (j, &x) in a.iter().enumerate() {
                if x == i_len {
                    prob *= f64::from(p.p_null) * th(NULL, tgt[j]);
                } else {
                    prob *= f64::from(p.p_word) * psi(x as i32 - last) * th(src[x], tgt[j]);
                    last = x as i32;
                }
            }
            prob *= psi(i_len as i32 - last);
            for (j, &x) in a.iter().enumerate() {
                post[j][x] += prob;
            }
        }
        post.into_iter()
            .flat_map(|r| {
                let s: f64 = r.iter().sum();
                r.into_iter().map(move |x| x / s)
            })
            .collect()
    }

    #[test]
    fn forward_backward_equals_brute_force() {
        use crate::priors::{DirPriors, DirTable};
        use std::sync::Arc;
        let mut x = 7u64;
        let mut rnd = move |m: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % m
        };
        for _ in 0..200 {
            let i_len = 1 + rnd(4) as usize;
            let j_len = 1 + rnd(4) as usize;
            let src: Vec<u32> = (0..i_len).map(|_| 1 + rnd(4) as u32).collect();
            let tgt: Vec<u32> = (0..j_len).map(|_| 1 + rnd(4) as u32).collect();
            let mut t = DirTable::default();
            for e in 0..5u32 {
                for f in 1..5u32 {
                    if rnd(2) == 0 {
                        t.lex.push((e, f, rnd(20) as f64));
                    }
                }
            }
            for m in -4..=5 {
                t.jump.insert(m, rnd(30) as f64);
            }
            let h = Hyper::default();
            let pr = DirPriors::new(&t, &h, 5, 4);
            let p = Params::with_priors(&h, 4, Some(Arc::new(pr)));
            let mut s = Scratch::default();
            sentence(&src, &tgt, &p, &mut s);
            let want = brute(&src, &tgt, &p);
            assert_eq!(s.post.len(), want.len());
            for (a, b) in s.post.iter().zip(&want) {
                assert!(
                    (a - b).abs() <= 1e-9 * b.max(1e-300) + 1e-15,
                    "{src:?} {tgt:?}: {a} vs {b}"
                );
            }
        }
    }
    /// The O(I² J) recursion before the clamped-jump tails were aggregated (reference).
    fn naive_sentence<'s, E: Emission>(
        src: &[u32],
        tgt: &[u32],
        p: &E,
        s: &'s mut Scratch,
    ) -> &'s [usize] {
        let (i_len, j_len) = (src.len(), tgt.len());
        s.best.clear();
        if j_len == 0 {
            return &s.best;
        }
        if i_len == 0 {
            s.best.resize(j_len, 0);
            return &s.best;
        }
        let w = i_len + 1;
        let p_null = p.p_null();
        let p_word = p.p_word();
        // parameters
        s.theta.clear();
        for &f in tgt {
            for e in src.iter().copied().chain(std::iter::once(NULL)) {
                s.theta.push(p.theta(e, f));
            }
        }
        s.psi.clear();
        for m in -(i_len as i32)..=(i_len as i32 + 1) {
            s.psi.push(p.psi(m));
        }
        let psi = |m: i32| s.psi[(m + i_len as i32) as usize];
        // forward: row j has I aligned states then I + 1 NULL states
        let fw = 2 * i_len + 1;
        s.fwd.clear();
        s.fwd.resize(j_len * fw, 0.0);
        s.g.clear();
        s.g.resize(i_len + 1, 0.0); // G(p), index p + 1
        s.g[0] = 1.0; // the start: last position −1
        for j in 0..j_len {
            crate::cancel::check(); // O(I²) work per position: bounded polling in long sentences
            let th = &s.theta[j * w..(j + 1) * w];
            let row = &mut s.fwd[j * fw..(j + 1) * fw];
            for i in 0..i_len {
                let mut acc = 0.0;
                for (pi, &g) in s.g.iter().enumerate() {
                    if g != 0.0 {
                        acc += g * psi(i as i32 - (pi as i32 - 1));
                    }
                }
                row[i] = p_word * th[i] * acc;
            }
            for (pi, &g) in s.g.iter().enumerate() {
                row[i_len + pi] = p_null * th[i_len] * g;
            }
            normalize(row);
            // G for the next step: aligned to p or NULL with last p
            s.g[0] = row[i_len];
            for i in 0..i_len {
                s.g[i + 1] = row[i] + row[i_len + i + 1];
            }
        }
        // backward
        s.bwd.clear();
        s.bwd.resize(j_len * w, 0.0);
        {
            let last = &mut s.bwd[(j_len - 1) * w..j_len * w];
            for (pi, x) in last.iter_mut().enumerate() {
                *x = psi(i_len as i32 - (pi as i32 - 1));
            }
            normalize(last);
        }
        for j in (0..j_len - 1).rev() {
            crate::cancel::check();
            let th = &s.theta[(j + 1) * w..(j + 2) * w];
            let (head, tail) = s.bwd.split_at_mut((j + 1) * w);
            let next = &tail[..w];
            let row = &mut head[j * w..];
            for (pi, x) in row.iter_mut().enumerate() {
                let pp = pi as i32 - 1;
                let mut acc = p_null * th[i_len] * next[pi];
                for i in 0..i_len {
                    acc += p_word * psi(i as i32 - pp) * th[i] * next[i + 1];
                }
                *x = acc;
            }
            normalize(row);
        }
        // posteriors and argmax
        s.post.clear();
        for j in 0..j_len {
            let a = &s.fwd[j * fw..(j + 1) * fw];
            let b = &s.bwd[j * w..(j + 1) * w];
            let at = s.post.len();
            s.post.extend((0..i_len).map(|i| a[i] * b[i + 1]));
            s.post
                .push((0..=i_len).map(|pi| a[i_len + pi] * b[pi]).sum::<f64>());
            let row = &mut s.post[at..];
            normalize(row);
            let mut best = 0;
            for i in 1..=i_len {
                if row[i] > row[best] {
                    best = i;
                }
            }
            s.best.push(best);
        }
        &s.best
    }

    /// The tail aggregation gives the plain recursion's posteriors (to rounding) and links on
    /// long sentences with arbitrary jump priors, and the same bits below MAX_JUMP tokens.
    #[test]
    fn clamped_tails_equal_the_plain_recursion() {
        use crate::priors::{DirPriors, DirTable};
        use std::sync::Arc;
        let mut x = 5u64;
        let mut rnd = move |m: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % m
        };
        for case in 0..60 {
            let i_len = [1, 5, 40, 63, 64, 65, 66, 129, 200, 300][case % 10] as usize;
            let j_len = 1 + rnd(i_len as u64 + 20) as usize;
            let src: Vec<u32> = (0..i_len).map(|_| 1 + rnd(30) as u32).collect();
            let tgt: Vec<u32> = (0..j_len).map(|_| 1 + rnd(30) as u32).collect();
            let mut t = DirTable::default();
            for _ in 0..200 {
                t.lex
                    .push((rnd(31) as u32, 1 + rnd(30) as u32, rnd(50) as f64));
            }
            t.lex.sort_by_key(|r| (r.0, r.1));
            t.lex.dedup_by_key(|r| (r.0, r.1));
            for m in -100..=100 {
                if rnd(3) == 0 {
                    t.jump.insert(m, rnd(1000) as f64);
                }
            }
            let h = Hyper::default();
            let pr = DirPriors::new(&t, &h, 31, 30);
            let p = Params::with_priors(&h, 30, Some(Arc::new(pr)));
            let (mut a, mut b) = (Scratch::default(), Scratch::default());
            let fast = sentence(&src, &tgt, &p, &mut a).to_vec();
            let plain = naive_sentence(&src, &tgt, &p, &mut b).to_vec();
            if i_len < MAX_JUMP as usize {
                assert_eq!(a.post, b.post, "I {i_len}: same bits");
            }
            for (x, y) in a.post.iter().zip(&b.post) {
                assert!(
                    (x - y).abs() <= 1e-9 * y.max(1e-300) + 1e-15,
                    "I {i_len}: {x} vs {y}"
                );
            }
            assert_eq!(fast, plain, "I {i_len}");
        }
    }
}
