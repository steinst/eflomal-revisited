//! The sequential collapsed Gibbs sampler: a direct transcription of the model
//! (Östling and Tiedemann 2016) for one chain (one direction, one sampler). Clarity over speed; this is the reference
//! the parallel sampler is checked against.
//!
//! A [`Chain`] is the state of one Markov chain: one alignment per target token, the lexical
//! counts, the jump counts (Models 2 and 3) and the fertility ratio table (Model 3). Chains are
//! independent; [`crate::pipeline::train`] runs them on separate threads, which keeps the
//! result exact and deterministic.
//!
//! Random numbers: [`Stream::Align`] at counter `(chain, t, sentence)` for the sweep t (t = 0 is
//! the initialisation; sweeps are numbered 1, 2, … across models), one `u32` per target token
//! of a sentence with a non-empty source side; [`Stream::Fertility`] at `(chain, t, word type)`
//! for the fertility table used in sweep t (t = total + 1 for the decode pass).

use std::sync::Arc;
use std::time::Instant;

use crate::bitext::View;
use crate::cancel;
use crate::counts::{FertCounts, JumpCounts, LexCounts, FERT_CATS, F_MAX, NULL};
use crate::model::{
    cumulate, fill_weights, rescue_weights, total_ok, Hyper, Params, Plain, Token, EPS,
};
use crate::pipeline::Progress;
use crate::priors::DirPriors;
use crate::rng::{chain_id, draw_cumulative, Rng, Stream};

/// Fertility ratios are clamped to this range so that an extreme draw cannot overflow `f32`.
const RATIO_MIN: f64 = 1e-30;
const RATIO_MAX: f64 = 1e30;
/// Gamma variates are floored here before they are divided.
const GAMMA_MIN: f64 = 1e-300;

/// What one chain needs besides the corpus view.
#[derive(Clone, Debug)]
pub struct ChainConfig {
    pub seed: u64,
    pub hyper: Hyper,
    /// Iterations of Model 1, 2, 3.
    pub iters: [usize; 3],
    /// 0-based sampler index within the direction.
    pub sampler: u32,
    /// Prior counts of the chain's direction, in the view's ids.
    pub priors: Option<Arc<DirPriors>>,
    /// Add up the normalised conditionals of every sweep of the last model ([`Chain::acc`]).
    pub accumulate: bool,
}

impl ChainConfig {
    /// The weight constants of the chain's direction.
    pub fn params(&self, view: &View<'_>) -> Params {
        Params::with_priors(&self.hyper, view.tgt.n_types() - 1, self.priors.clone())
            .for_corpus(view.tgt.n_tokens(), view.len())
    }
}

/// The state of one chain.
#[derive(Clone, Debug)]
pub struct Chain {
    /// a_j per target token of the view (same offsets as the view's target side); [`EPS`] =
    /// NULL.
    pub align: Vec<u16>,
    pub lex: LexCounts,
    /// Jump counts; consistent with `align` once a Model 2 or 3 sweep has run.
    pub jump: JumpCounts,
    /// R(e, n), `FERT_CATS` per source id; empty unless Model 3 is active.
    pub ratio: Vec<f32>,
    /// The last model that ran (1 if none did: the random start is scored with Model 1).
    pub model: u8,
    /// With [`ChainConfig::accumulate`]: per sampled target token the sum of its normalised
    /// conditionals (I + 1 values, ε last) over the sweeps of the last model, at the offsets
    /// of [`acc_offsets`]; otherwise empty.
    pub acc: Vec<f32>,
}

/// Start of each sentence's accumulator block: J · (I + 1) values for a sentence with both
/// sides non-empty, none otherwise (K + 1 offsets).
pub fn acc_offsets(view: &View<'_>) -> Vec<usize> {
    let mut off = Vec::with_capacity(view.len() + 1);
    let mut at = 0usize;
    off.push(0);
    for k in 0..view.len() {
        let (i, j) = (view.src.line_len(k), view.tgt.line_len(k));
        if i > 0 && j > 0 {
            at += j * (i + 1);
        }
        off.push(at);
    }
    off
}

/// The last model of a schedule with at least one sweep (0 if none).
pub fn last_model(iters: &[usize; 3]) -> u8 {
    (1..=3u8)
        .rev()
        .find(|&m| iters[usize::from(m) - 1] > 0)
        .unwrap_or(0)
}

/// Turn the weights `w` of one token into cumulative sums for [`draw_cumulative`], first adding
/// the normalised weights to `acc` if given. If the total is not usable ([`total_ok`]),
/// `rescue` replaces `w` with the conditional recomputed in log space ([`rescue_weights`]; it
/// recomputes from the counts, so it may run after `w` was cumulated). On ordinary input this
/// is exactly the previous `accumulate_into` + `cumulate`, bit for bit.
#[inline(always)]
pub fn finish_weights(w: &mut [f32], acc: Option<&mut [f32]>, rescue: impl FnOnce(&mut [f32])) {
    match acc {
        None => {
            if !total_ok(cumulate(w)) {
                rescue(w);
                cumulate(w);
            }
        }
        Some(acc) => {
            if !total_ok(w.iter().sum()) {
                rescue(w);
            }
            accumulate_into(acc, w);
            cumulate(w);
        }
    }
}

/// Add w · (1 / Σw) to `acc` (Σw summed left to right, as the decode pass).
#[inline]
pub fn accumulate_into(acc: &mut [f32], w: &[f32]) {
    let total: f32 = w.iter().sum();
    let inv = 1.0 / total;
    for (a, &x) in acc.iter_mut().zip(w) {
        *a += x * inv;
    }
}

/// Counter word of the RNG for a view's direction and a sampler.
pub fn chain_of(view: &View<'_>, sampler: u32) -> u32 {
    chain_id(u32::from(view.direction.reverse()), sampler)
}

/// Random initialisation: for each target token one `f32` u; ε if I = 0 or
/// u < p_ε, else `below(I)`.
pub fn init_alignment(view: &View<'_>, rng: &Rng, chain: u32, p_null: f32) -> Vec<u16> {
    let off = view.tgt.offsets();
    let mut a = vec![EPS; view.tgt.n_tokens()];
    for k in 0..view.len() {
        let i_len = view.src.line_len(k) as u32;
        let mut d = rng.draws(chain, 0, k as u32);
        for x in &mut a[off[k]..off[k + 1]] {
            let u = d.next_f32();
            *x = if i_len == 0 || u < p_null {
                EPS
            } else {
                d.below(i_len) as u16
            };
        }
    }
    a
}

/// Lexical counts of an alignment.
pub fn lex_counts(view: &View<'_>, align: &[u16]) -> LexCounts {
    let off = view.tgt.offsets();
    let mut lex = LexCounts::new(view.src.n_types(), view.tgt.n_types());
    for k in 0..view.len() {
        let (src, tgt) = view.pair(k);
        for (&f, &a) in tgt.iter().zip(&align[off[k]..off[k + 1]]) {
            let e = if a == EPS { NULL } else { src[a as usize] };
            lex.inc(e, f);
        }
    }
    lex
}

/// The jumps of one sentence's alignment: from −1 through the non-NULL positions
/// to I; nothing if J = 0.
pub fn sentence_jumps(i_len: usize, a: &[u16], mut f: impl FnMut(i32)) {
    if a.is_empty() {
        return;
    }
    let mut prev = -1i32;
    for &x in a {
        if x != EPS {
            f(i32::from(x) - prev);
            prev = i32::from(x);
        }
    }
    f(i_len as i32 - prev);
}

/// Jump counts of an alignment.
pub fn jump_counts(view: &View<'_>, align: &[u16]) -> JumpCounts {
    let off = view.tgt.offsets();
    let mut jump = JumpCounts::default();
    for k in 0..view.len() {
        sentence_jumps(view.src.line_len(k), &align[off[k]..off[k + 1]], |m| {
            jump.inc(m)
        });
    }
    jump
}

/// Fertility counts c″(e, n) of an alignment (every source token, fertility 0 included).
pub fn fert_counts(view: &View<'_>, align: &[u16]) -> FertCounts {
    let off = view.tgt.offsets();
    let mut fc = FertCounts::new(view.src.n_types());
    let mut phi = Vec::new();
    for k in 0..view.len() {
        let src = view.src.line(k);
        sentence_fertility(src.len(), &align[off[k]..off[k + 1]], &mut phi);
        for (&e, &n) in src.iter().zip(&phi) {
            fc.inc(e, usize::from(n));
        }
    }
    fc
}

/// φ_i of every source position of one sentence into `phi` (resized to I).
pub fn sentence_fertility(i_len: usize, a: &[u16], phi: &mut Vec<u16>) {
    phi.clear();
    phi.resize(i_len, 0);
    for &x in a {
        if x != EPS {
            phi[x as usize] += 1;
        }
    }
}

/// Draw π_e ~ Dir(γ + c″(e, ·)) for every source word type and return the ratio table
/// R(e, n) = π_e(n+1) / π_e(n), R(e, F_MAX) = 1. Row 0 (NULL) is all 1.
pub fn fertility_ratios(
    fc: &FertCounts,
    n_types: usize,
    p: &Params,
    rng: &Rng,
    chain: u32,
    t: u32,
) -> Vec<f32> {
    let mut ratio = vec![1.0f32; n_types * FERT_CATS];
    let mut g = [0.0f64; FERT_CATS];
    for e in 1..n_types {
        let mut d = rng.draws(chain, t, e as u32);
        for (n, x) in g.iter_mut().enumerate() {
            *x = d
                .gamma(p.fert_alpha_of(e as u32, n) + f64::from(fc.get(e as u32, n)))
                .max(GAMMA_MIN);
        }
        let row = &mut ratio[e * FERT_CATS..(e + 1) * FERT_CATS];
        for n in 0..F_MAX {
            row[n] = (g[n + 1] / g[n]).clamp(RATIO_MIN, RATIO_MAX) as f32;
        }
        row[F_MAX] = 1.0;
    }
    ratio
}

/// Reusable per-chain buffers of a sweep.
#[derive(Default)]
struct Scratch {
    w: Vec<f32>,
    phi: Vec<u16>,
}

/// One Gibbs sweep of Model `model` over every sentence in order, with the
/// alignment draws of sweep `t`.
#[allow(clippy::too_many_arguments)]
fn sweep(
    view: &View<'_>,
    ch: &mut Chain,
    p: &Params,
    model: u8,
    rng: &Rng,
    chain: u32,
    t: u32,
    s: &mut Scratch,
    acc_off: Option<&[usize]>,
) {
    let off = view.tgt.offsets();
    for k in 0..view.len() {
        let (src, tgt) = view.pair(k);
        let i_len = src.len();
        if tgt.is_empty() || i_len == 0 {
            continue; // nothing to sample: every token of an empty-source pair is ε
        }
        if k % 4096 == 0 {
            cancel::check();
        }
        let a = &mut ch.align[off[k]..off[k + 1]];
        let mut d = rng.draws(chain, t, k as u32);
        if model >= 3 {
            sentence_fertility(i_len, a, &mut s.phi);
        }
        s.w.resize(i_len + 1, 0.0);
        let mut prev = -1i32;
        for j in 0..a.len() {
            let f = tgt[j];
            let old = a[j];
            let next = if model >= 2 {
                a[j + 1..]
                    .iter()
                    .find(|&&x| x != EPS)
                    .map_or(i_len as i32, |&x| i32::from(x))
            } else {
                i_len as i32
            };
            // remove a_j
            if old == EPS {
                ch.lex.dec(NULL, f);
                if model >= 2 {
                    ch.jump.dec(next - prev);
                }
            } else {
                let o = i32::from(old);
                ch.lex.dec(src[old as usize], f);
                if model >= 2 {
                    ch.jump.dec(o - prev);
                    ch.jump.dec(next - o);
                }
                if model >= 3 {
                    s.phi[old as usize] -= 1;
                }
            }
            // weights and draw
            let tok = Token {
                src,
                f,
                prev,
                next,
                phi: &s.phi,
                ratio: &ch.ratio,
            };
            let c = Plain {
                lex: &ch.lex,
                jump: &ch.jump,
            };
            fill_weights(&c, p, model, &tok, &mut s.w);
            let acc = acc_off.map(|ao| {
                let at = ao[k] + j * (i_len + 1);
                &mut ch.acc[at..at + i_len + 1]
            });
            finish_weights(&mut s.w, acc, |w| {
                rescue_weights(&c, p, model, &tok, w);
            });
            let u = d.next_f32();
            let pick = draw_cumulative(&s.w, u);
            // add back
            if pick == i_len {
                a[j] = EPS;
                ch.lex.inc(NULL, f);
                if model >= 2 {
                    ch.jump.inc(next - prev);
                }
            } else {
                a[j] = pick as u16;
                let n = pick as i32;
                ch.lex.inc(src[pick], f);
                if model >= 2 {
                    ch.jump.inc(n - prev);
                    ch.jump.inc(next - n);
                }
                if model >= 3 {
                    s.phi[pick] += 1;
                }
                prev = n;
            }
        }
    }
}

/// Fraction of target tokens currently aligned to NULL.
pub fn null_fraction(view: &View<'_>, ch: &Chain) -> f64 {
    let n = view.tgt.n_tokens();
    if n == 0 {
        0.0
    } else {
        f64::from(ch.lex.total(NULL)) / n as f64
    }
}

/// Run one chain through the whole schedule and prepare it for the decode pass: the state is
/// the last sample, and for Model 3 `ratio` holds the table drawn for iteration total + 1.
pub fn run_chain(
    view: &View<'_>,
    cfg: &ChainConfig,
    progress: &(dyn Fn(&Progress) + Sync),
) -> Chain {
    let started = Instant::now();
    let align_rng = Rng::new(cfg.seed, Stream::Align);
    let fert_rng = Rng::new(cfg.seed, Stream::Fertility);
    let chain = chain_of(view, cfg.sampler);
    let p = cfg.params(view);
    let align = init_alignment(view, &align_rng, chain, p.p_null);
    let lex = lex_counts(view, &align);
    let mut ch = Chain {
        align,
        lex,
        jump: JumpCounts::default(),
        ratio: Vec::new(),
        model: 1,
        acc: Vec::new(),
    };
    let last = last_model(&cfg.iters);
    let acc_off = cfg.accumulate.then(|| acc_offsets(view));
    if let Some(ao) = &acc_off {
        ch.acc = vec![0.0; ao[view.len()]];
    }
    let mut s = Scratch::default();
    let mut t = 0u32;
    let mut jumps_built = false;
    for m in 1..=3u8 {
        let n = cfg.iters[usize::from(m) - 1];
        if n == 0 {
            continue;
        }
        if m >= 2 && !jumps_built {
            ch.jump = jump_counts(view, &ch.align);
            jumps_built = true;
        }
        for it in 1..=n {
            cancel::check();
            t += 1;
            if m >= 3 {
                let fc = fert_counts(view, &ch.align);
                ch.ratio = fertility_ratios(&fc, view.src.n_types(), &p, &fert_rng, chain, t);
            }
            let ao = acc_off.as_deref().filter(|_| m == last);
            sweep(view, &mut ch, &p, m, &align_rng, chain, t, &mut s, ao);
            ch.model = m;
            progress(&Progress {
                direction: view.direction,
                sampler: cfg.sampler as usize,
                model: m,
                iteration: it,
                iterations: n,
                elapsed: started.elapsed(),
                null_fraction: null_fraction(view, &ch),
            });
        }
    }
    if ch.model >= 3 {
        let fc = fert_counts(view, &ch.align);
        ch.ratio = fertility_ratios(&fc, view.src.n_types(), &p, &fert_rng, chain, t + 1);
    }
    ch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitext::{Bitext, TextOptions};
    use crate::io::Direction;

    pub(crate) fn toy() -> Bitext {
        let src = b"a b c\nb c\na c d e\n\nc a\nd\nx y z\n";
        let tgt = b"A B C\nB C\nA C D E\nq\nC A\n\nZ Y X W\n";
        Bitext::from_pair_bytes(src, tgt, &TextOptions::default()).unwrap()
    }

    fn cfg(iters: [usize; 3]) -> ChainConfig {
        ChainConfig {
            seed: 5,
            hyper: Hyper::default(),
            iters,
            sampler: 0,
            priors: None,
            accumulate: false,
        }
    }

    #[test]
    fn counts_stay_consistent_with_the_alignment() {
        let b = toy();
        for dir in [Direction::Forward, Direction::Inverse] {
            let v = b.view(dir);
            for iters in [[3, 0, 0], [1, 3, 0], [1, 1, 3], [0, 2, 0], [0, 0, 2]] {
                let ch = run_chain(&v, &cfg(iters), &|_| {});
                assert!(ch.lex.same_counts(&lex_counts(&v, &ch.align)), "{iters:?}");
                if iters[1] + iters[2] > 0 {
                    assert_eq!(ch.jump, jump_counts(&v, &ch.align), "{iters:?}");
                }
                // every alignment points into its sentence; empty sources are NULL
                let off = v.tgt.offsets();
                for k in 0..v.len() {
                    let i_len = v.src.line_len(k);
                    for &a in &ch.align[off[k]..off[k + 1]] {
                        assert!(a == EPS || (a as usize) < i_len);
                    }
                }
            }
        }
    }

    #[test]
    fn chains_are_reproducible_and_seeded() {
        let b = toy();
        let v = b.view(Direction::Forward);
        let a = run_chain(&v, &cfg([2, 2, 2]), &|_| {});
        let c = run_chain(&v, &cfg([2, 2, 2]), &|_| {});
        assert_eq!(a.align, c.align);
        assert_eq!(a.ratio, c.ratio);
        let mut other = cfg([2, 2, 2]);
        other.sampler = 1;
        let mut differ = false;
        for seed in 1..20 {
            other.seed = seed;
            differ |= run_chain(&v, &other, &|_| {}).align != a.align;
        }
        assert!(differ);
    }

    #[test]
    fn jumps_of_a_sentence() {
        let mut got = Vec::new();
        sentence_jumps(3, &[1, EPS, 0, 2], |m| got.push(m));
        assert_eq!(got, [2, -1, 2, 1]);
        got.clear();
        sentence_jumps(0, &[EPS, EPS], |m| got.push(m));
        assert_eq!(got, [1]);
        got.clear();
        sentence_jumps(4, &[], |m| got.push(m));
        assert!(got.is_empty());
    }

    #[test]
    fn progress_reports_every_iteration() {
        let b = toy();
        let v = b.view(Direction::Inverse);
        let n = std::sync::Mutex::new(Vec::new());
        run_chain(&v, &cfg([2, 1, 3]), &|p| {
            n.lock().unwrap().push((p.model, p.iteration, p.iterations))
        });
        assert_eq!(
            n.into_inner().unwrap(),
            [
                (1, 1, 2),
                (1, 2, 2),
                (2, 1, 1),
                (3, 1, 3),
                (3, 2, 3),
                (3, 3, 3)
            ]
        );
    }

    /// Model 1 collapsed Gibbs is exact: the sampler's long-run state frequencies must match
    /// the posterior over all alignments, enumerated.
    #[test]
    fn model1_sampler_matches_the_exact_posterior() {
        let b =
            Bitext::from_pair_bytes(b"a b\nb\n", b"x y\ny x\n", &TextOptions::default()).unwrap();
        let v = b.view(Direction::Forward);
        let h = Hyper {
            lex_alpha: 0.5,
            ..Hyper::default()
        };
        let p = Params::new(&h, v.tgt.n_types() - 1);
        let (vs, vt) = (v.src.n_types(), v.tgt.n_types() - 1);
        // states: sentence 0 has 2 tokens with 3 choices, sentence 1 has 2 tokens with 2
        let decode = |mut s: usize| -> Vec<u16> {
            let mut a = Vec::new();
            for &c in &[3usize, 3, 2, 2] {
                let x = s % c;
                s /= c;
                a.push(if x == c - 1 { EPS } else { x as u16 });
            }
            a
        };
        let n_states = 36;
        let lgamma = |x: f64| ln_gamma(x);
        let mut post = vec![0.0f64; n_states];
        for (st, w) in post.iter_mut().enumerate() {
            let a = decode(st);
            let lex = lex_counts(&v, &a);
            let mut lp = 0.0;
            // alignment prior
            for (k, &x) in a.iter().enumerate() {
                let i_len = if k < 2 { 2.0 } else { 1.0 };
                lp += if x == EPS {
                    h.null_prior.ln()
                } else {
                    ((1.0 - h.null_prior) / i_len).ln()
                };
            }
            for e in 0..vs as u32 {
                lp += lgamma(h.lex_alpha * vt as f64)
                    - lgamma(h.lex_alpha * vt as f64 + f64::from(lex.total(e)));
                for f in 1..=vt as u32 {
                    lp += lgamma(h.lex_alpha + f64::from(lex.get(e, f))) - lgamma(h.lex_alpha);
                }
            }
            *w = lp.exp();
        }
        let z: f64 = post.iter().sum();
        post.iter_mut().for_each(|x| *x /= z);
        let encode = |a: &[u16]| -> usize {
            let mut s = 0;
            let mut mul = 1;
            for (&x, &c) in a.iter().zip(&[3usize, 3, 2, 2]) {
                let d = if x == EPS { c - 1 } else { x as usize };
                s += d * mul;
                mul *= c;
            }
            s
        };
        let rng = Rng::new(9, Stream::Align);
        let chain = chain_of(&v, 0);
        let align = init_alignment(&v, &rng, chain, p.p_null);
        let lex = lex_counts(&v, &align);
        let mut ch = Chain {
            align,
            lex,
            jump: JumpCounts::default(),
            ratio: Vec::new(),
            model: 1,
            acc: Vec::new(),
        };
        let mut s = Scratch::default();
        let n = 200_000u32;
        let mut freq = vec![0u64; n_states];
        for t in 1..=n {
            sweep(&v, &mut ch, &p, 1, &rng, chain, t, &mut s, None);
            freq[encode(&ch.align)] += 1;
        }
        for (st, (&f, &q)) in freq.iter().zip(&post).enumerate() {
            let obs = f as f64 / f64::from(n);
            // autocorrelated samples: allow 5 binomial sd plus a small absolute slack
            let sd = (q * (1.0 - q) / f64::from(n)).sqrt();
            assert!(
                (obs - q).abs() < 5.0 * sd * 3.0 + 2e-3,
                "state {st}: {obs} vs {q}"
            );
        }
    }

    /// Lanczos approximation of ln Γ(x), x > 0 (tests only).
    fn ln_gamma(x: f64) -> f64 {
        const G: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        if x < 0.5 {
            return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln()
                - ln_gamma(1.0 - x);
        }
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, &g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}
