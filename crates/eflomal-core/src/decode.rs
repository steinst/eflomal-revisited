//! The final decode pass: for every target token, the
//! normalised conditional of each sampler at its final state (computed without changing the
//! state), summed over the samplers; the link is the argmax over the source positions and ε
//! (ties to the lowest position, ε counting as position I; an ε argmax gives no link).
//!
//! The pass is read-only, so sentences are decoded in parallel; each sentence's line is a
//! pure function of the chains, and lines are concatenated in sentence order, so the output
//! does not depend on the thread count.

use rayon::prelude::*;

use crate::bitext::View;
use crate::counts::{JumpCounts, LexCounts, SentenceGather, JUMP_BUCKETS, NULL};
use crate::io::Direction;
use crate::model::{
    fill_weights, ratio_of, rescue_weights, total_ok, Counts, Gathered, JumpTable, Params, Token,
    EPS,
};
use crate::oracle::{acc_offsets, sentence_fertility, Chain};

/// The counts of a chain with one target token's contribution removed on the fly: the
/// lexical count of (e_old, f) and n(e_old) less one, and the one or two jumps through the
/// token less one each. The same numbers as removing it for real.
struct Without<'a> {
    lex: &'a LexCounts,
    jump: &'a JumpCounts,
    /// Source type the token is aligned to (NULL for ε).
    e_old: u32,
    f: u32,
    /// Buckets of the jumps through the token (a second `usize::MAX` if only one).
    jb: [usize; 2],
    n_jumps: u32,
    /// β(m) + d(m) of the chain without the token's own jumps (unused below Model 2).
    jt: &'a [f32; JUMP_BUCKETS],
}

impl Counts for Without<'_> {
    #[inline]
    fn lex(&self, e: u32, f: u32) -> u32 {
        self.lex.get(e, f) - u32::from(e == self.e_old && f == self.f)
    }
    #[inline]
    fn total(&self, e: u32) -> u32 {
        self.lex.total(e) - u32::from(e == self.e_old)
    }
    #[inline]
    fn jump(&self, m: i32) -> u32 {
        let b = JumpCounts::bucket(m);
        self.jump.buckets()[b] - u32::from(b == self.jb[0]) - u32::from(b == self.jb[1])
    }
    #[inline]
    fn jump_total(&self) -> u32 {
        self.jump.total() - self.n_jumps
    }
    #[inline(always)]
    fn jump_term(&self, m: i32, _alpha: f32) -> f32 {
        self.jt[JumpCounts::bucket(m)]
    }
}

/// Sentences per parallel work item of the decode pass (fixed, so that work items do not
/// depend on the thread count; the output would not either way).
const CHUNK: usize = 2048;

/// Decode one direction from its N chains and write its Pharaoh text, links in the corpus's
/// source-target order, one line per pair (empty for pairs without links). With `scores`,
/// each line gets ` ||| score`. `model` is the model whose conditional gives the
/// links, `score_model` the one whose conditional gives the scores (the chains must have the
/// jump counts and fertility tables it needs). Chains with accumulated conditionals
/// ([`Chain::acc`]) add them to the final pass's distribution of each token.
pub fn decode(
    view: &View<'_>,
    chains: &[Chain],
    p: &Params,
    model: u8,
    score_model: u8,
    scores: bool,
) -> Vec<u8> {
    join_parts(decode_parts(view, chains, p, model, score_model, scores))
}

/// Join the parts of [`decode_parts`] into one text, freeing each part as soon as it is
/// copied (the text is never held twice).
pub fn join_parts(parts: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for part in parts {
        out.extend_from_slice(&part);
    }
    out
}

/// [`decode`] without the final join: the text as consecutive parts (one per work item), so
/// that the caller can free the chains before joining them ([`join_parts`]).
pub fn decode_parts(
    view: &View<'_>,
    chains: &[Chain],
    p: &Params,
    model: u8,
    score_model: u8,
    scores: bool,
) -> Vec<Vec<u8>> {
    assert!(!chains.is_empty());
    let k_n = view.len();
    // with accumulated conditionals (`--accumulate`)
    let acc_off = (!chains[0].acc.is_empty()).then(|| acc_offsets(view));
    let acc_off = acc_off.as_deref();
    (0..k_n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let mut out = Vec::new();
            let mut sc = SentenceScratch::default();
            if model.max(score_model) >= 2 {
                sc.jts = chains
                    .iter()
                    .map(|ch| {
                        let mut t = JumpTable::default();
                        t.fill(p, |b| ch.jump.buckets()[b]);
                        t
                    })
                    .collect();
            } else {
                sc.jts = vec![JumpTable::default(); chains.len()];
            }
            for k in c * CHUNK..((c + 1) * CHUNK).min(k_n) {
                crate::cancel::check();
                let models = [model, if scores { score_model } else { model }];
                decode_sentence(
                    view, chains, p, models, acc_off, k, scores, &mut sc, &mut out,
                );
            }
            out
        })
        .collect()
}

#[derive(Default)]
struct SentenceScratch {
    acc: Vec<f32>,
    w: Vec<f32>,
    phi: Vec<Vec<u16>>,
    links: Vec<(u32, u32)>,
    /// The sentence's distinct words and each chain's gathered lexical counts.
    g: SentenceGather,
    mats: Vec<Vec<u32>>,
    /// Each chain's 1 / (A + n(e)) per distinct source word of the sentence.
    rds: Vec<Vec<f32>>,
    /// Each chain's R(e_i, min(φ_i, F_MAX)) per position (Model 3).
    rrs: Vec<Vec<f32>>,
    /// Each chain's jump terms (Model 2 and up), patched per token.
    jts: Vec<JumpTable>,
}

/// A token's column, the position → distinct-word map, the reciprocals and the fertility
/// ratios of a gathered sentence (see [`Gathered`]).
type GatheredParts<'a> = (&'a [u32], &'a [u32], &'a [f32], &'a [f32]);

/// Fill `w` from the counts `c` of a chain, through the sentence's gathered counts if any.
#[inline]
fn weights<C: Counts>(
    c: C,
    gat: Option<GatheredParts<'_>>,
    p: &Params,
    model: u8,
    tok: &Token<'_>,
    w: &mut [f32],
) {
    match gat {
        Some((col, sidx, rinv, rr)) => {
            let g = Gathered {
                c,
                col,
                sidx,
                rinv,
                rr,
            };
            fill_weights(&g, p, model, tok, w)
        }
        None => fill_weights(&c, p, model, tok, w),
    }
}

/// [`weights`] followed by their total: `(ln total, 1 / total)`, from the `f32` sum, or, if
/// that is not usable ([`total_ok`]), from the conditional recomputed in log space
/// ([`rescue_weights`], which also replaces `w` with normalised weights).
#[inline]
fn weights_total<C: Counts>(
    c: C,
    gat: Option<GatheredParts<'_>>,
    p: &Params,
    model: u8,
    tok: &Token<'_>,
    w: &mut [f32],
) -> (f64, f32) {
    let total: f32 = w.iter().sum();
    if !p.checks || total_ok(total) {
        return (f64::from(total).ln(), 1.0 / total);
    }
    let ln_total = match gat {
        Some((col, sidx, rinv, rr)) => {
            let g = Gathered {
                c,
                col,
                sidx,
                rinv,
                rr,
            };
            rescue_weights(&g, p, model, tok, w)
        }
        None => rescue_weights(&c, p, model, tok, w),
    };
    (ln_total, 1.0 / w.iter().sum::<f32>())
}

#[allow(clippy::too_many_arguments)]
fn decode_sentence(
    view: &View<'_>,
    chains: &[Chain],
    p: &Params,
    [model, score_model]: [u8; 2],
    acc_off: Option<&[usize]>,
    k: usize,
    scores: bool,
    sc: &mut SentenceScratch,
    out: &mut Vec<u8>,
) {
    let off = view.tgt.offsets();
    let (src, tgt) = view.pair(k);
    let i_len = src.len();
    sc.links.clear();
    let mut score = 0.0f64;
    if !tgt.is_empty() {
        sc.acc.resize(i_len + 1, 0.0);
        sc.w.resize(i_len + 1, 0.0);
        sc.phi.resize(chains.len(), Vec::new());
        let top = model.max(score_model);
        if top >= 3 {
            for (ch, phi) in chains.iter().zip(sc.phi.iter_mut()) {
                sentence_fertility(i_len, &ch.align[off[k]..off[k + 1]], phi);
            }
        }
        let gathered = sc.g.index(src, tgt);
        let rows = sc.g.rows();
        sc.mats.resize(chains.len(), Vec::new());
        sc.rds.resize(chains.len(), Vec::new());
        sc.rrs.resize(chains.len(), Vec::new());
        if gathered {
            for ((ch, phi), rr) in chains.iter().zip(&sc.phi).zip(&mut sc.rrs) {
                rr.clear();
                if top >= 3 {
                    rr.extend(
                        src.iter()
                            .zip(phi)
                            .map(|(&e, &n)| ratio_of(&ch.ratio, e, n)),
                    );
                }
            }
            for ((ch, mat), rd) in chains.iter().zip(sc.mats.iter_mut()).zip(&mut sc.rds) {
                sc.g.fill(&ch.lex, mat);
                rd.clear();
                rd.extend(
                    sc.g.src_ids
                        .iter()
                        .map(|&e| 1.0 / (p.lex_den_of(e) + ch.lex.total(e) as f32)),
                );
            }
        }
        for (j, &f) in tgt.iter().enumerate() {
            sc.acc.fill(0.0);
            for (((((ch, phi), mat), rd), rr), jt) in chains
                .iter()
                .zip(sc.phi.iter_mut())
                .zip(sc.mats.iter_mut())
                .zip(sc.rds.iter_mut())
                .zip(sc.rrs.iter_mut())
                .zip(sc.jts.iter_mut())
            {
                let a = &ch.align[off[k]..off[k + 1]];
                let old = a[j];
                let (prev, next) = if top >= 2 {
                    (
                        a[..j]
                            .iter()
                            .rev()
                            .find(|&&x| x != EPS)
                            .map_or(-1, |&x| i32::from(x)),
                        a[j + 1..]
                            .iter()
                            .find(|&&x| x != EPS)
                            .map_or(i_len as i32, |&x| i32::from(x)),
                    )
                } else {
                    (-1, i_len as i32)
                };
                let (e_old, jb, n_jumps) = if old == EPS {
                    (NULL, [JumpCounts::bucket(next - prev), usize::MAX], 1)
                } else {
                    let o = i32::from(old);
                    (
                        src[old as usize],
                        [JumpCounts::bucket(o - prev), JumpCounts::bucket(next - o)],
                        2,
                    )
                };

                let mut rr_with = 0.0;
                if top >= 3 && old != EPS {
                    phi[old as usize] -= 1;
                    if gathered {
                        let r = ratio_of(&ch.ratio, e_old, phi[old as usize]);
                        rr_with = std::mem::replace(&mut rr[old as usize], r);
                    }
                }
                if top >= 2 {
                    jt.patch(p, jb, |b| ch.jump.buckets()[b]);
                }
                let jt_t = &jt.t;
                let without = |m: u8| Without {
                    lex: &ch.lex,
                    jump: &ch.jump,
                    e_old,
                    f,
                    jb,
                    n_jumps: if m >= 2 { n_jumps } else { 0 },
                    jt: jt_t,
                };
                let tok = Token {
                    src,
                    f,
                    prev,
                    next,
                    phi,
                    ratio: &ch.ratio,
                };
                // the token's own count and reciprocal, removed and restored below (the
                // values `Without` gives)
                let (at, di_old) = if gathered {
                    let old_pos = if old == EPS { i_len } else { old as usize };
                    (sc.g.tidx[j] as usize * rows, sc.g.sidx[old_pos] as usize)
                } else {
                    (0, 0)
                };
                let r_with = if gathered {
                    mat[at + di_old] -= 1;
                    let n = ch.lex.total(e_old) - 1;
                    std::mem::replace(&mut rd[di_old], 1.0 / (p.lex_den_of(e_old) + n as f32))
                } else {
                    0.0
                };
                let gat = gathered.then(|| (&mat[at..at + rows], &sc.g.sidx[..], &rd[..], &rr[..]));
                weights(without(model), gat, p, model, &tok, &mut sc.w);
                let (ln_total, inv) = weights_total(without(model), gat, p, model, &tok, &mut sc.w);
                for (acc, &w) in sc.acc.iter_mut().zip(&sc.w) {
                    *acc += w * inv;
                }
                if let (Some(ao), true) = (acc_off, i_len > 0) {
                    let at = ao[k] + j * (i_len + 1);
                    for (acc, &x) in sc.acc.iter_mut().zip(&ch.acc[at..at + i_len + 1]) {
                        *acc += x;
                    }
                }
                if score_model == model {
                    score += ln_total;
                } else {
                    weights(without(score_model), gat, p, score_model, &tok, &mut sc.w);
                    score +=
                        weights_total(without(score_model), gat, p, score_model, &tok, &mut sc.w).0;
                }
                if top >= 3 && old != EPS {
                    phi[old as usize] += 1;
                    if gathered {
                        rr[old as usize] = rr_with;
                    }
                }
                if top >= 2 {
                    jt.restore();
                }
                if gathered {
                    mat[at + di_old] += 1;
                    rd[di_old] = r_with;
                }
            }
            let mut best = 0;
            for i in 1..=i_len {
                if sc.acc[i] > sc.acc[best] {
                    best = i;
                }
            }
            if best < i_len {
                let (i, j) = (best as u32, j as u32);
                sc.links.push(match view.direction {
                    Direction::Forward => (i, j),
                    Direction::Inverse => (j, i),
                });
            }
        }
        score = -score / (tgt.len() as f64 * chains.len() as f64);
    }
    sc.links.sort_unstable();
    crate::symmetrize::format_links(&sc.links, out);
    if scores {
        use std::io::Write;
        let _ = write!(out, " ||| {score}");
    }
    out.push(b'\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitext::{Bitext, TextOptions};
    use crate::model::{Hyper, Plain};
    use crate::oracle::{chain_of, run_chain, ChainConfig};
    use crate::rng::{Rng, Stream};

    /// The on-the-fly removal gives exactly the weights of a real removal.
    #[test]
    fn without_equals_real_removal() {
        let b = Bitext::from_pair_bytes(
            b"a b c\nb c a b\nc\n",
            b"x y z\ny z x x w\nz z\n",
            &TextOptions::default(),
        )
        .unwrap();
        for dir in [Direction::Forward, Direction::Inverse] {
            let v = b.view(dir);
            for model in 1..=3u8 {
                let mut iters = [0; 3];
                iters[usize::from(model) - 1] = 2;
                let cfg = ChainConfig {
                    seed: 3,
                    hyper: Hyper::default(),
                    iters,
                    sampler: 0,
                    priors: None,
                    accumulate: false,
                };
                let ch = run_chain(&v, &cfg, &|_| {});
                let p = Params::new(&cfg.hyper, v.tgt.n_types() - 1);
                let off = v.tgt.offsets();
                for k in 0..v.len() {
                    let (src, tgt) = v.pair(k);
                    let a = &ch.align[off[k]..off[k + 1]];
                    let i_len = src.len();
                    let mut phi = Vec::new();
                    sentence_fertility(i_len, a, &mut phi);
                    for j in 0..tgt.len() {
                        let f = tgt[j];
                        let old = a[j];
                        let prev = a[..j]
                            .iter()
                            .rev()
                            .find(|&&x| x != EPS)
                            .map_or(-1, |&x| i32::from(x));
                        let next = a[j + 1..]
                            .iter()
                            .find(|&&x| x != EPS)
                            .map_or(i_len as i32, |&x| i32::from(x));
                        // real removal on copies
                        let (mut lex, mut jump) = (ch.lex.clone(), ch.jump.clone());
                        let mut phi2 = phi.clone();
                        let (e_old, jb, nj) = if old == EPS {
                            lex.dec(NULL, f);
                            if model >= 2 {
                                jump.dec(next - prev);
                            }
                            (NULL, [JumpCounts::bucket(next - prev), usize::MAX], 1)
                        } else {
                            let o = i32::from(old);
                            lex.dec(src[old as usize], f);
                            if model >= 2 {
                                jump.dec(o - prev);
                                jump.dec(next - o);
                            }
                            phi2[old as usize] -= 1;
                            (
                                src[old as usize],
                                [JumpCounts::bucket(o - prev), JumpCounts::bucket(next - o)],
                                2,
                            )
                        };
                        let tok = Token {
                            src,
                            f,
                            prev,
                            next,
                            phi: &phi2,
                            ratio: &ch.ratio,
                        };
                        let mut w1 = vec![0.0; i_len + 1];
                        let mut w2 = vec![0.0; i_len + 1];
                        fill_weights(
                            &Plain {
                                lex: &lex,
                                jump: &jump,
                            },
                            &p,
                            model,
                            &tok,
                            &mut w1,
                        );
                        let mut jt = JumpTable::default();
                        jt.fill(&p, |b| ch.jump.buckets()[b]);
                        if model >= 2 {
                            jt.patch(&p, jb, |b| ch.jump.buckets()[b]);
                        }
                        let wo = Without {
                            lex: &ch.lex,
                            jump: &ch.jump,
                            e_old,
                            f,
                            jb,
                            n_jumps: nj,
                            jt: &jt.t,
                        };
                        fill_weights(&wo, &p, model, &tok, &mut w2);
                        assert_eq!(w1, w2, "{dir:?} model {model} k {k} j {j}");
                    }
                }
            }
        }
    }

    #[test]
    fn decode_writes_one_line_per_pair_in_corpus_order() {
        let b = Bitext::from_pair_bytes(
            b"a b\n\nb a\nc\n",
            b"x y\nz\ny x\n\n",
            &TextOptions::default(),
        )
        .unwrap();
        for dir in [Direction::Forward, Direction::Inverse] {
            let v = b.view(dir);
            let cfg = ChainConfig {
                seed: 1,
                hyper: Hyper::default(),
                iters: [3, 3, 3],
                sampler: 0,
                priors: None,
                accumulate: false,
            };
            let chains: Vec<_> = (0..2)
                .map(|s| {
                    run_chain(
                        &v,
                        &ChainConfig {
                            sampler: s,
                            ..cfg.clone()
                        },
                        &|_| {},
                    )
                })
                .collect();
            let p = Params::new(&cfg.hyper, v.tgt.n_types() - 1);
            let text = decode(&v, &chains, &p, 3, 3, true);
            let lines: Vec<&[u8]> = text.split(|&c| c == b'\n').collect();
            assert_eq!(lines.len(), 5, "{}", String::from_utf8_lossy(&text));
            // the pair with an empty target side has score 0; the one with an empty source
            // side has no links but a score
            let (empty_tgt, empty_src) = match dir {
                Direction::Forward => (3, 1),
                Direction::Inverse => (1, 3),
            };
            assert_eq!(lines[empty_tgt], b" ||| 0");
            assert!(lines[empty_src].starts_with(b" ||| ") && lines[empty_src].len() > 6);
            // links are within bounds and in source-target order
            for (k, l) in lines[..4].iter().enumerate() {
                let body = &l[..l.windows(5).position(|w| w == b" ||| ").unwrap()];
                for link in crate::symmetrize::parse_line(body).unwrap() {
                    assert!((link.0 as usize) < b.source.line_len(k));
                    assert!((link.1 as usize) < b.target.line_len(k));
                }
            }
        }
        let _ = (chain_of, Rng::new(1, Stream::Align));
    }
}
