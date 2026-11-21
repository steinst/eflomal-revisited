//! The model: hyperparameters, the iteration schedule and the unnormalised conditional weights
//! of one alignment variable.
//!
//! The weights are computed in `f32` in a fixed order of operations, written out in
//! [`fill_weights`], so that every sampler that calls it (the sequential oracle, the
//! parallel sampler, the decode pass) gets the same bits from the same counts.

use std::sync::Arc;

use crate::counts::{JumpCounts, LexCounts, FERT_CATS, F_MAX, JUMP_BUCKETS, MAX_JUMP, NULL};
use crate::priors::DirPriors;

/// Alignment value of a NULL-aligned target token (ε).
pub const EPS: u16 = u16::MAX;

/// Dirichlet hyperparameters and the NULL prior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyper {
    /// α, symmetric lexical prior.
    pub lex_alpha: f64,
    /// β, symmetric jump prior.
    pub jump_alpha: f64,
    /// γ, symmetric fertility prior.
    pub fert_alpha: f64,
    /// p_ε, prior probability of a NULL alignment.
    pub null_prior: f64,
}

/// Accepted range of the Dirichlet parameters α (see [`Hyper::validate`]).
pub const ALPHA_RANGE: (f64, f64) = (1e-30, 1e25);
/// Accepted range of the NULL prior.
pub const NULL_PRIOR_RANGE: (f64, f64) = (1e-6, 1.0 - 1e-6);

impl Default for Hyper {
    fn default() -> Self {
        Hyper {
            lex_alpha: 0.001,
            jump_alpha: 0.5,
            fert_alpha: 1.0,
            null_prior: 0.2,
        }
    }
}

impl Hyper {
    /// The ranges keep the `f32` weights finite: an α below 1e-30 or above
    /// 1e25 (or a NULL prior within 1e-6 of 0 or 1) rounds to 0 or overflows in `f32`, and
    /// the weights become NaN.
    pub fn validate(&self) -> Result<(), String> {
        let alpha = |x: f64, n: &str| {
            if (ALPHA_RANGE.0..=ALPHA_RANGE.1).contains(&x) {
                Ok(())
            } else {
                Err(format!(
                    "{n} must be between {:e} and {:e}, got {x:?}",
                    ALPHA_RANGE.0, ALPHA_RANGE.1
                ))
            }
        };
        alpha(self.lex_alpha, "lex-alpha")?;
        alpha(self.jump_alpha, "jump-alpha")?;
        alpha(self.fert_alpha, "fert-alpha")?;
        if !(NULL_PRIOR_RANGE.0..=NULL_PRIOR_RANGE.1).contains(&self.null_prior) {
            return Err(format!(
                "null-prior must be between {:e} and 1 - {:e}, got {:?}",
                NULL_PRIOR_RANGE.0, NULL_PRIOR_RANGE.0, self.null_prior
            ));
        }
        Ok(())
    }
}

/// The `f32` constants of one direction, folded from [`Hyper`] and the target vocabulary size,
/// and the direction's priors if any. Without priors every accessor returns the
/// plain constant; with priors the values that have no record are the same constants, so a
/// run with an empty priors file gives the same bits as one without.
#[derive(Clone, Debug)]
pub struct Params {
    /// α.
    pub lex_alpha: f32,
    /// A = α · |V_T| (|V_T| without NULL).
    pub lex_den: f32,
    /// β.
    pub jump_alpha: f32,
    /// B = β · number of jump buckets.
    pub jump_den: f32,
    /// p_ε.
    pub p_null: f32,
    /// 1 − p_ε.
    pub p_word: f32,
    /// γ.
    pub fert_alpha: f64,
    /// Prior counts of this direction, in the view's ids.
    pub priors: Option<Arc<DirPriors>>,
    /// Whether a Model 3 weight's product can underflow before its fertility ratio (see
    /// [`MIN_TOTAL`]); `true` (checked) unless [`Params::for_corpus`] proved it cannot.
    pub fert_guard: bool,
    /// Whether the total of a token's weights must be checked ([`total_ok`]) and a Model 3
    /// underflow can occur; `true` unless [`Params::for_corpus`] proved neither can happen.
    pub checks: bool,
}

impl Params {
    /// `n_target_words`: target vocabulary size without NULL.
    pub fn new(h: &Hyper, n_target_words: usize) -> Self {
        Params {
            lex_alpha: h.lex_alpha as f32,
            lex_den: (h.lex_alpha * n_target_words as f64) as f32,
            jump_alpha: h.jump_alpha as f32,
            jump_den: (h.jump_alpha * crate::counts::JUMP_BUCKETS as f64) as f32,
            p_null: h.null_prior as f32,
            p_word: (1.0 - h.null_prior) as f32,
            fert_alpha: h.fert_alpha,
            priors: None,
            fert_guard: true,
            checks: true,
        }
    }

    /// [`Params::new`] with the priors of the direction; B becomes Σ_m (β + η(m)).
    pub fn with_priors(h: &Hyper, n_target_words: usize, priors: Option<Arc<DirPriors>>) -> Self {
        let mut p = Params::new(h, n_target_words);
        if let Some(pr) = &priors {
            p.jump_den = pr.jump_den();
        }
        p.priors = priors;
        p
    }

    /// Clear [`Params::fert_guard`] when no Model 3 weight of a corpus with `n_tokens` target
    /// tokens and `n_lines` lines can underflow before its fertility ratio: every factor before
    /// it is at least α / (max A_e + T), 1 − p_ε, and twice β / (B + T + K) (a count n(e) or the
    /// jump total D never exceeds T, resp. T + K), and their product is at least 4 ×
    /// `f32::MIN_POSITIVE`. True for the defaults on any realistic corpus (≈ 1e-28 at 10⁸
    /// tokens), so ordinary runs skip the per-candidate check; extreme priors keep it. Clears
    /// [`Params::checks`] too when, in addition, every token's total is at least 2 ×
    /// [`MIN_TOTAL`] (its NULL candidate alone, ≈ 1e-21 for the defaults at 10⁸ tokens): the
    /// samplers then run without the per-token total check.
    pub fn for_corpus(mut self, n_tokens: usize, n_lines: usize) -> Self {
        let t = n_tokens as f64;
        let a_max = f64::from(self.priors.as_ref().map_or(self.lex_den, |p| p.max_den()));
        let jp = f64::from(self.jump_alpha) / (f64::from(self.jump_den) + t + n_lines as f64);
        let bound = f64::from(self.lex_alpha) / (a_max + t) * f64::from(self.p_word) * jp * jp;
        let safe = bound >= 4.0 * f64::from(f32::MIN_POSITIVE); // false for NaN
        self.fert_guard = !safe;
        // every total includes the NULL candidate: p_ε · L(ε, f) [· Jp], each factor bounded
        // below as above (L(ε, f) ≥ α / (max A_e + T))
        let null = f64::from(self.p_null) * f64::from(self.lex_alpha) / (a_max + t) * jp;
        let totals_safe = null >= 2.0 * f64::from(MIN_TOTAL);
        self.checks = self.fert_guard || !totals_safe;
        self
    }

    /// α + λ(e, f).
    #[inline]
    pub fn lex_alpha_of(&self, e: u32, f: u32) -> f32 {
        match &self.priors {
            None => self.lex_alpha,
            Some(pr) => pr.lex_alpha(e, f, self.lex_alpha),
        }
    }

    /// A_e = α·|V_T| + Σ_f λ(e, f).
    #[inline]
    pub fn lex_den_of(&self, e: u32) -> f32 {
        match &self.priors {
            None => self.lex_den,
            Some(pr) => pr.lex_den(e),
        }
    }

    /// β + η(m), m clamped.
    #[inline]
    pub fn jump_alpha_of(&self, m: i32) -> f32 {
        match &self.priors {
            None => self.jump_alpha,
            Some(pr) => pr.jump_alpha(m),
        }
    }

    /// γ + κ(e, n), n clamped (`f64`, the precision of the fertility draw).
    #[inline]
    pub fn fert_alpha_of(&self, e: u32, n: usize) -> f64 {
        match &self.priors {
            None => self.fert_alpha,
            Some(pr) => pr.fert_alpha(e, n),
        }
    }
}

// ------------------------------------------------------------------------------------------
// Schedule

/// eflomal's automatic base iteration count x = max(2, round_half_even(l · 5000 / √K))
/// (measured).
pub fn auto_iterations(k: usize, length: f64) -> usize {
    let x = (length * 5000.0 / (k.max(1) as f64).sqrt()).round_ties_even();
    if x.is_finite() && x > 2.0 {
        x as usize
    } else {
        2
    }
}

/// Iterations of Model 1, 2 and 3 for `model` ∈ {1, 2, 3}, K lines and the `--length` factor;
/// `overrides[m]` replaces the automatic count of model m + 1 (our choice: each
/// override acts on its own stage). Stages above `model` get 0.
pub fn schedule(model: u8, k: usize, length: f64, overrides: [Option<usize>; 3]) -> [usize; 3] {
    let x = auto_iterations(k, length);
    let auto = match model {
        1 => [x, 0, 0],
        2 => [(x / 4).max(2), x, 0],
        _ => [(x / 4).max(2), (x / 4).max(1), x],
    };
    let mut it = [0; 3];
    for m in 0..usize::from(model.clamp(1, 3)) {
        it[m] = overrides[m].unwrap_or(auto[m]);
    }
    it
}

// ------------------------------------------------------------------------------------------
// Conditional weights

/// Read access to the counts of one chain as the conditional sees them (the counts without
/// the variable being sampled). The sampler passes its real counts after removing the
/// variable; the decode pass passes a view that subtracts it on the fly.
pub trait Counts {
    /// c(e, f).
    fn lex(&self, e: u32, f: u32) -> u32;
    /// c(e, f) for the word e at position `pos` of the sentence (`pos` = I for NULL) and the
    /// target word f of the token being sampled. An implementation may read it from a
    /// per-sentence gather, but it must be exactly [`Counts::lex`].
    #[inline]
    fn lex_at(&self, pos: usize, e: u32, f: u32) -> u32 {
        let _ = pos;
        self.lex(e, f)
    }
    /// n(e).
    fn total(&self, e: u32) -> u32;
    /// d(m), m clamped.
    fn jump(&self, m: i32) -> u32;
    /// D.
    fn jump_total(&self) -> u32;
    /// R(e, min(φ_i, F_MAX)) of the source word e at position `pos` (Model 3). An
    /// implementation may return it from a per-sentence array, but it must have exactly
    /// these bits.
    #[inline]
    fn fert_ratio(&self, t: &Token<'_>, pos: usize, e: u32) -> f32 {
        t.ratio[e as usize * FERT_CATS + usize::from(t.phi[pos]).min(F_MAX)]
    }
    /// β(m) + d(m) in `f32`, where `alpha` = β(m) (β + η(m) with priors). An implementation may
    /// return it from a table, but it must have exactly these bits.
    #[inline]
    fn jump_term(&self, m: i32, alpha: f32) -> f32 {
        alpha + self.jump(m) as f32
    }
    /// 1 / (A + n(e)) for the source word e at position `pos` of the sentence (`pos` = I for
    /// NULL). An implementation may return a cached value, but it must have exactly these
    /// bits.
    #[inline]
    fn recip(&self, p: &Params, pos: usize, e: u32) -> f32 {
        let _ = pos;
        1.0 / (p.lex_den_of(e) + self.total(e) as f32)
    }
}

/// The plain counts of a chain.
pub struct Plain<'a> {
    pub lex: &'a LexCounts,
    pub jump: &'a JumpCounts,
}

impl Counts for Plain<'_> {
    #[inline]
    fn lex(&self, e: u32, f: u32) -> u32 {
        self.lex.get(e, f)
    }
    #[inline]
    fn total(&self, e: u32) -> u32 {
        self.lex.total(e)
    }
    #[inline]
    fn jump(&self, m: i32) -> u32 {
        self.jump.get(m)
    }
    #[inline]
    fn jump_total(&self) -> u32 {
        self.jump.total()
    }
}

/// Counts `c` with the lexical counts of the sentence gathered once
/// ([`crate::counts::SentenceGather`]): `col` is the token's target-word column of the
/// sentence's matrix and `rinv` holds 1 / (A + n(e)) per distinct source word, both indexed
/// through `sidx` (position → distinct word) and both already *without* the token (the caller
/// patches the token's own entry before the weights and restores it after). So
/// [`Counts::lex_at`] and [`Counts::recip`] return exactly `c`'s values; everything else is
/// `c`'s.
pub struct Gathered<'a, C> {
    pub c: C,
    pub col: &'a [u32],
    pub sidx: &'a [u32],
    pub rinv: &'a [f32],
    /// R(e_i, min(φ_i, F_MAX)) per position, with the token removed (Model 3; else empty).
    pub rr: &'a [f32],
}

impl<C: Counts> Counts for Gathered<'_, C> {
    #[inline]
    fn lex(&self, e: u32, f: u32) -> u32 {
        self.c.lex(e, f)
    }
    #[inline(always)]
    fn lex_at(&self, pos: usize, _e: u32, _f: u32) -> u32 {
        self.col[self.sidx[pos] as usize]
    }
    #[inline]
    fn total(&self, e: u32) -> u32 {
        self.c.total(e)
    }
    #[inline(always)]
    fn recip(&self, _p: &Params, pos: usize, _e: u32) -> f32 {
        self.rinv[self.sidx[pos] as usize]
    }
    #[inline]
    fn jump(&self, m: i32) -> u32 {
        self.c.jump(m)
    }
    #[inline]
    fn jump_total(&self) -> u32 {
        self.c.jump_total()
    }
    #[inline(always)]
    fn jump_term(&self, m: i32, alpha: f32) -> f32 {
        self.c.jump_term(m, alpha)
    }
    #[inline(always)]
    fn fert_ratio(&self, _t: &Token<'_>, pos: usize, _e: u32) -> f32 {
        self.rr[pos]
    }
}

/// R(e, min(φ, F_MAX)) from a fertility ratio table.
#[inline(always)]
pub fn ratio_of(ratio: &[f32], e: u32, phi: u16) -> f32 {
    ratio[e as usize * FERT_CATS + usize::from(phi).min(F_MAX)]
}

/// β(m) + d(m) of every jump bucket, in `f32`, for [`Counts::jump_term`] from a table.
/// [`JumpTable::patch`] subtracts a token's own jumps (as the on-the-fly removal does) and
/// [`JumpTable::restore`] undoes it.
#[derive(Clone, Debug)]
pub struct JumpTable {
    pub t: [f32; JUMP_BUCKETS],
    saved: [(usize, f32); 2],
}

impl Default for JumpTable {
    fn default() -> Self {
        JumpTable {
            t: [0.0; JUMP_BUCKETS],
            saved: [(usize::MAX, 0.0); 2],
        }
    }
}

impl JumpTable {
    /// Entry of bucket `b` from its integer count.
    #[inline]
    pub fn set(&mut self, p: &Params, b: usize, count: u32) {
        self.t[b] = p.jump_alpha_of(b as i32 - MAX_JUMP) + count as f32;
    }

    /// Every bucket from `count(b)`.
    pub fn fill(&mut self, p: &Params, count: impl Fn(usize) -> u32) {
        for b in 0..JUMP_BUCKETS {
            self.set(p, b, count(b));
        }
    }

    /// Replace the entries of the token's jump buckets `jb` (`usize::MAX` = none; they may be
    /// equal) by their counts without the token, `count(b)` being the count with it.
    #[inline]
    pub fn patch(&mut self, p: &Params, jb: [usize; 2], count: impl Fn(usize) -> u32) {
        for (k, &b) in jb.iter().enumerate() {
            if b == usize::MAX {
                self.saved[k].0 = usize::MAX;
                continue;
            }
            self.saved[k] = (b, self.t[b]);
            let own = u32::from(b == jb[0]) + u32::from(b == jb[1]);
            self.set(p, b, count(b) - own);
        }
    }

    /// Undo [`JumpTable::patch`].
    #[inline]
    pub fn restore(&mut self) {
        for k in [1, 0] {
            let (b, x) = self.saved[k];
            if b != usize::MAX {
                self.t[b] = x;
            }
        }
    }
}

/// Everything the weights of one target token depend on besides the counts.
pub struct Token<'a> {
    /// Source sentence (word ids; NULL is not in it).
    pub src: &'a [u32],
    /// The target word.
    pub f: u32,
    /// Position of the nearest non-NULL alignment before this token, or −1.
    pub prev: i32,
    /// Position of the nearest non-NULL alignment after this token, or I.
    pub next: i32,
    /// Fertility of every source position without this token (Model 3 only).
    pub phi: &'a [u16],
    /// Fertility ratio table R(e, n), `FERT_CATS` per source id (Model 3 only).
    pub ratio: &'a [f32],
}

/// The prior parts of the weights as [`fill_weights`] reads them: monomorphised for "no
/// priors" (plain constants) and "priors", so the inner loop has no per-candidate branch.
trait Prior {
    /// α + λ(e, f).
    fn alpha(&self, e: u32, f: u32) -> f32;
    /// β + η(m).
    fn jump_alpha(&self, m: i32) -> f32;
}

/// No priors: α and β.
struct NoPrior {
    alpha: f32,
    jump_alpha: f32,
}

impl Prior for NoPrior {
    #[inline(always)]
    fn alpha(&self, _e: u32, _f: u32) -> f32 {
        self.alpha
    }
    #[inline(always)]
    fn jump_alpha(&self, _m: i32) -> f32 {
        self.jump_alpha
    }
}

/// Priors of a direction (α as the fallback of pairs without a record).
struct WithPrior<'a> {
    pr: &'a DirPriors,
    alpha: f32,
}

impl Prior for WithPrior<'_> {
    #[inline(always)]
    fn alpha(&self, e: u32, f: u32) -> f32 {
        self.pr.lex_alpha(e, f, self.alpha)
    }
    #[inline(always)]
    fn jump_alpha(&self, m: i32) -> f32 {
        self.pr.jump_alpha(m)
    }
}

/// L(e, f) = ((α + λ(e, f)) + c(e, f)) · (1 / (A_e + n(e))), in this order, in `f32`; e is the
/// word at position `pos` (I for NULL), see [`Counts::recip`].
#[inline]
pub fn lex_weight<C: Counts>(c: &C, p: &Params, pos: usize, e: u32, f: u32) -> f32 {
    (p.lex_alpha_of(e, f) + c.lex(e, f) as f32) * c.recip(p, pos, e)
}

#[inline(always)]
fn lex_weight_with<C: Counts, P: Prior>(
    c: &C,
    p: &Params,
    pr: &P,
    pos: usize,
    e: u32,
    f: u32,
) -> f32 {
    (pr.alpha(e, f) + c.lex_at(pos, e, f) as f32) * c.recip(p, pos, e)
}

/// Fill `w[0 ..= I]` with the unnormalised weights of the candidates 0 … I−1, ε.
/// For each candidate, in `f32`, multiplied left to right:
///
/// * Model 1: i: `L(e_i, f) · ((1 − p_ε) / I)`; ε: `p_ε · L(ε, f)`.
/// * Model 2: i: `L(e_i, f) · (1 − p_ε) · Jp(i − prev) · Jp(next − i)`;
///   ε: `p_ε · L(ε, f) · Jp(next − prev)`, with `Jp(m) = (β + d(m)) · (1 / (B + D))`.
/// * Model 3: Model 2's weight of i times `R(e_i, min(φ_i, F_MAX))`; ε as Model 2.
///
/// With priors, α, β, A and B are those of [`Params`]'s accessors. Totals and
/// cumulative sums add the weights in the order 0 … I−1, ε ([`cumulate`]).
#[inline(always)]
pub fn fill_weights<C: Counts>(c: &C, p: &Params, model: u8, t: &Token<'_>, w: &mut [f32]) {
    match &p.priors {
        None => {
            let pr = NoPrior {
                alpha: p.lex_alpha,
                jump_alpha: p.jump_alpha,
            };
            if model >= 3 && p.fert_guard {
                fill_weights_with::<C, _, true>(c, p, &pr, model, t, w)
            } else {
                fill_weights_with::<C, _, false>(c, p, &pr, model, t, w)
            }
        }
        Some(d) => {
            let pr = WithPrior {
                pr: d,
                alpha: p.lex_alpha,
            };
            if model >= 3 && p.fert_guard {
                fill_weights_with::<C, _, true>(c, p, &pr, model, t, w)
            } else {
                fill_weights_with::<C, _, false>(c, p, &pr, model, t, w)
            }
        }
    }
}

/// `GUARD`: mark Model 3 products that underflowed before a ratio > 1 ([`Params::fert_guard`]);
/// without it the loop is the plain fast path.
#[inline(always)]
fn fill_weights_with<C: Counts, P: Prior, const GUARD: bool>(
    c: &C,
    p: &Params,
    pr: &P,
    model: u8,
    t: &Token<'_>,
    w: &mut [f32],
) {
    let i_len = t.src.len();
    debug_assert!(w.len() > i_len);
    if model <= 1 {
        let u = p.p_word / i_len as f32;
        for (i, &e) in t.src.iter().enumerate() {
            w[i] = lex_weight_with(c, p, pr, i, e, t.f) * u;
        }
        w[i_len] = p.p_null * lex_weight_with(c, p, pr, i_len, NULL, t.f);
        return;
    }
    let jinv = 1.0 / (p.jump_den + c.jump_total() as f32);
    let jp = |m: i32| c.jump_term(m, pr.jump_alpha(m)) * jinv;
    for (i, &e) in t.src.iter().enumerate() {
        let ii = i as i32;
        let mut x =
            lex_weight_with(c, p, pr, i, e, t.f) * p.p_word * jp(ii - t.prev) * jp(t.next - ii);
        if model >= 3 {
            let r = c.fert_ratio(t, i, e);
            // an underflowed product would be scaled up by a ratio > 1: mark it for the rescue
            // (only where Params::for_corpus could not rule underflow out)
            x = if GUARD && x < f32::MIN_POSITIVE && r > 1.0 {
                f32::NAN
            } else {
                x * r
            };
        }
        w[i] = x;
    }
    w[i_len] = p.p_null * lex_weight_with(c, p, pr, i_len, NULL, t.f) * jp(t.next - t.prev);
}

/// The smallest total of candidate weights the `f32` fast path accepts: 2⁻⁸⁰ ≈ 8.3e-25.
///
/// Every factor of a weight is a positive normal `f32` for the accepted ranges of the
/// hyperparameters and prior counts, but their product can underflow: e.g. a jump prior of 1e25
/// makes every other jump probability ≈ 5e-26, and two of them times a lexical weight fall
/// below `f32`'s range, so all weights and the total become 0. All factors
/// but the fertility ratio are ≤ 1, so a weight lost to underflow is below `f32::MIN_POSITIVE`
/// (1.2e-38); with a total of at least 2⁻⁸⁰ and at most 1024 candidates the lost mass is below
/// 1.5e-11 of the total, under `f32` rounding. A weight whose product underflowed before a
/// fertility ratio > 1 is marked NaN by [`fill_weights`], which fails this check too. Ordinary
/// totals are many orders of magnitude larger (the NULL candidate alone is ≳ 1e-20 with the
/// default hyperparameters on a 10⁸-token corpus), so the fast path is unchanged there.
pub const MIN_TOTAL: f32 = f32::from_bits(0x1780_0000);

/// Whether an `f32` total of candidate weights can be used as is ([`MIN_TOTAL`]; NaN fails); otherwise [`rescue_weights`] recomputes the conditional.
#[inline(always)]
pub fn total_ok(total: f32) -> bool {
    // NaN fails the comparison; +∞ cannot occur (each weight ≤ 1e30, ≤ 1024 candidates)
    total >= MIN_TOTAL
}

/// Recompute the weights of [`fill_weights`] from their factors in `f64` log space and replace
/// `w[0 ..= I]` with the normalised conditional (in `f32`, summing to 1 within rounding);
/// returns the natural log of the true total (the log normaliser, for scores). Used only when
/// [`total_ok`] fails: the factors are the same `f32` values `fill_weights` multiplies, so the
/// result is the conditional the fast path would give with unlimited range.
#[cold]
#[inline(never)]
pub fn rescue_weights<C: Counts>(
    c: &C,
    p: &Params,
    model: u8,
    t: &Token<'_>,
    w: &mut [f32],
) -> f64 {
    let mut lw = Vec::with_capacity(t.src.len() + 1);
    match &p.priors {
        None => {
            let pr = NoPrior {
                alpha: p.lex_alpha,
                jump_alpha: p.jump_alpha,
            };
            log_weights_with(c, p, &pr, model, t, &mut lw)
        }
        Some(d) => {
            let pr = WithPrior {
                pr: d,
                alpha: p.lex_alpha,
            };
            log_weights_with(c, p, &pr, model, t, &mut lw)
        }
    }
    normalize_log(&lw, w)
}

/// `w[k] = exp(lw[k] − log Σ exp(lw))` in `f32`; returns log Σ exp(lw). If every log weight is
/// −∞ (a factor of 0, impossible for accepted input), the candidates are taken as equally
/// likely and the log normaliser is −∞.
pub fn normalize_log(lw: &[f64], w: &mut [f32]) -> f64 {
    let m = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY || m.is_nan() {
        let u = 1.0 / lw.len() as f32;
        w[..lw.len()].iter_mut().for_each(|x| *x = u);
        return f64::NEG_INFINITY;
    }
    let s: f64 = lw.iter().map(|&x| (x - m).exp()).sum();
    for (x, &l) in w.iter_mut().zip(lw) {
        *x = ((l - m).exp() / s) as f32;
    }
    m + s.ln()
}

/// The natural logs of [`fill_weights_with`]'s weights, each the sum of the logs of the same
/// factors (the lexical numerator α + c and the counts are summed in `f64`).
fn log_weights_with<C: Counts, P: Prior>(
    c: &C,
    p: &Params,
    pr: &P,
    model: u8,
    t: &Token<'_>,
    out: &mut Vec<f64>,
) {
    let ln = |x: f32| f64::from(x).ln();
    let lex = |pos: usize, e: u32| {
        (f64::from(pr.alpha(e, t.f)) + f64::from(c.lex_at(pos, e, t.f))).ln()
            + ln(c.recip(p, pos, e))
    };
    let i_len = t.src.len();
    out.clear();
    if model <= 1 {
        let u = ln(p.p_word) - (i_len as f64).ln();
        out.extend(t.src.iter().enumerate().map(|(i, &e)| lex(i, e) + u));
        out.push(ln(p.p_null) + lex(i_len, NULL));
        return;
    }
    let ljinv = -(f64::from(p.jump_den) + f64::from(c.jump_total())).ln();
    let jp = |m: i32| ln(c.jump_term(m, pr.jump_alpha(m))) + ljinv;
    for (i, &e) in t.src.iter().enumerate() {
        let ii = i as i32;
        let mut x = lex(i, e) + ln(p.p_word) + jp(ii - t.prev) + jp(t.next - ii);
        if model >= 3 {
            x += ln(c.fert_ratio(t, i, e));
        }
        out.push(x);
    }
    out.push(ln(p.p_null) + lex(i_len, NULL) + jp(t.next - t.prev));
}

/// Turn weights into cumulative sums in place (left to right) and return the total.
#[inline]
pub fn cumulate(w: &mut [f32]) -> f32 {
    let mut s = 0.0f32;
    for x in w.iter_mut() {
        s += *x;
        *x = s;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_eflomal() {
        // iteration counts measured on eflomal 2.0.0
        assert_eq!(schedule(3, 1447, 1.0, [None; 3]), [32, 32, 131]);
        assert_eq!(schedule(3, 100_447, 1.0, [None; 3]), [4, 4, 16]);
        assert_eq!(schedule(3, 1_130_551, 1.0, [None; 3]), [2, 1, 5]);
        assert_eq!(schedule(3, 447, 1.0, [None; 3]), [59, 59, 236]);
        assert_eq!(auto_iterations(10_000, 0.25), 12);
        assert_eq!(auto_iterations(10_000, 0.27), 14);
        assert_eq!(auto_iterations(10_000, 0.29), 14);
        assert_eq!(auto_iterations(10_000, 0.0001), 2);
        assert_eq!(schedule(1, 100_447, 1.0, [None; 3]), [16, 0, 0]);
        assert_eq!(schedule(2, 100_447, 1.0, [None; 3]), [4, 16, 0]);
        assert_eq!(
            schedule(3, 100_447, 1.0, [Some(7), None, Some(0)]),
            [7, 4, 0]
        );
        assert_eq!(
            schedule(1, 100_447, 1.0, [Some(3), Some(5), Some(6)]),
            [3, 0, 0]
        );
        assert_eq!(auto_iterations(0, 1.0), 5000);
    }

    /// An independent enumerator of the conditional in `f64` log space, straight
    /// from the hyperparameters, prior records and counts (no `Params`, no `f32` factors):
    /// `(probabilities of 0 … I−1, ε; log normaliser)`.
    #[allow(clippy::too_many_arguments)]
    fn enumerate(
        h: &Hyper,
        prior: &crate::priors::DirTable,
        n_target_words: usize,
        lex: &LexCounts,
        jump: &JumpCounts,
        model: u8,
        t: &Token<'_>,
        ratio: impl Fn(usize) -> f64,
    ) -> (Vec<f64>, f64) {
        let lam = |e: u32, f: u32| {
            prior
                .lex
                .iter()
                .filter(|r| r.0 == e && r.1 == f)
                .map(|r| r.2)
                .sum::<f64>()
        };
        let lam_row = |e: u32| {
            prior
                .lex
                .iter()
                .filter(|r| r.0 == e)
                .map(|r| r.2)
                .sum::<f64>()
        };
        let eta = |b: usize| {
            prior
                .jump
                .iter()
                .filter(|(&m, _)| JumpCounts::bucket(m) == b)
                .map(|(_, &c)| c)
                .sum::<f64>()
        };
        let l = |e: u32| {
            ((h.lex_alpha + lam(e, t.f) + f64::from(lex.get(e, t.f))).ln())
                - (h.lex_alpha * n_target_words as f64 + lam_row(e) + f64::from(lex.total(e))).ln()
        };
        let b_den: f64 = (0..JUMP_BUCKETS)
            .map(|b| h.jump_alpha + eta(b))
            .sum::<f64>()
            + f64::from(jump.total());
        let jp = |m: i32| {
            let b = JumpCounts::bucket(m);
            (h.jump_alpha + eta(b) + f64::from(jump.buckets()[b])).ln() - b_den.ln()
        };
        let i_len = t.src.len();
        let mut lw = Vec::new();
        for (i, &e) in t.src.iter().enumerate() {
            let ii = i as i32;
            lw.push(if model <= 1 {
                l(e) + (1.0 - h.null_prior).ln() - (i_len as f64).ln()
            } else {
                let mut x = l(e) + (1.0 - h.null_prior).ln() + jp(ii - t.prev) + jp(t.next - ii);
                if model >= 3 {
                    x += ratio(i).ln();
                }
                x
            });
        }
        lw.push(if model <= 1 {
            h.null_prior.ln() + l(NULL)
        } else {
            h.null_prior.ln() + l(NULL) + jp(t.next - t.prev)
        });
        let m = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z = m + lw.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        (lw.iter().map(|x| (x - z).exp()).collect(), z)
    }

    /// Accepted extreme priors underflow the `f32` product; the rescue gives
    /// the enumerator's conditional and a finite log normaliser. Ordinary input keeps the fast
    /// path and agrees with the rescue.
    #[test]
    fn underflowing_weights_are_rescued() {
        use crate::priors::{DirPriors, DirTable};
        // source ids: 0 NULL, 1 a, 2 b, 3 c; target ids: 0 NULL, 1 x, 2 y
        let n_src = 4;
        let n_tw = 2;
        let fixture = DirTable {
            lex: vec![(NULL, 2, 1e25)],
            jump: [(-64, 1e25)].into_iter().collect(),
            fert: vec![],
        };
        let none = DirTable::default();
        let mut lex = LexCounts::new(n_src, n_tw + 1);
        let mut jump = JumpCounts::default();
        lex.inc(1, 1);
        lex.inc(2, 2);
        lex.inc(NULL, 1);
        jump.inc(1);
        jump.inc(2);
        let c = Plain {
            lex: &lex,
            jump: &jump,
        };
        let extreme = Hyper {
            lex_alpha: ALPHA_RANGE.0,
            jump_alpha: ALPHA_RANGE.0,
            fert_alpha: ALPHA_RANGE.1,
            null_prior: NULL_PRIOR_RANGE.0,
        };
        let long: Vec<u32> = (0..150).map(|i| 1 + i % 3).collect();
        // (case, hyperparameters, priors, source, prev, next)
        type Case<'a> = (&'a str, Hyper, &'a DirTable, Vec<u32>, i32, i32);
        let cases: Vec<Case> = vec![
            ("fixture", Hyper::default(), &fixture, vec![1, 2], -1, 2),
            ("reversed", Hyper::default(), &fixture, vec![2, 1], -1, 2),
            // i = 1: jumps 1 and 1 share a bucket; long: ±64 clamped buckets shared
            (
                "same buckets",
                Hyper::default(),
                &fixture,
                vec![1, 2, 3],
                0,
                2,
            ),
            (
                "clamped buckets",
                Hyper::default(),
                &fixture,
                long.clone(),
                -1,
                150,
            ),
            ("extreme alpha", extreme, &fixture, vec![1, 2, 2], -1, 3),
            ("extreme, no priors", extreme, &none, long, -1, 150),
            (
                "null prior high",
                Hyper {
                    null_prior: NULL_PRIOR_RANGE.1,
                    ..Hyper::default()
                },
                &fixture,
                vec![3, 1],
                -1,
                2,
            ),
            ("ordinary", Hyper::default(), &none, vec![1, 2, 3], -1, 3),
        ];
        let mut rescued = 0;
        for (name, h, prior, src, prev, next) in cases {
            let pr = Arc::new(DirPriors::new(prior, &h, n_src, n_tw));
            let p = Params::with_priors(&h, n_tw, Some(pr));
            let i_len = src.len();
            let mut ratio = vec![1.0f32; n_src * FERT_CATS];
            ratio[FERT_CATS] = 1e-30; // R(a, 0): the clamp's lower end
            ratio[2 * FERT_CATS] = 1e30;
            let phi = vec![0u16; i_len];
            for f in [1u32, 2] {
                let t = Token {
                    src: &src,
                    f,
                    prev,
                    next,
                    phi: &phi,
                    ratio: &ratio,
                };
                for model in 1..=3u8 {
                    let ctx = format!("{name} f {f} model {model}");
                    let (want, z) = enumerate(&h, prior, n_tw, &lex, &jump, model, &t, |i| {
                        f64::from(ratio[src[i] as usize * FERT_CATS])
                    });
                    let mut w = vec![0.0f32; i_len + 1];
                    fill_weights(&c, &p, model, &t, &mut w);
                    let fast_total: f32 = w.iter().sum();
                    let ok = total_ok(fast_total);
                    let mut r = vec![0.0f32; i_len + 1];
                    let lz = rescue_weights(&c, &p, model, &t, &mut r);
                    assert!(lz.is_finite(), "{ctx}");
                    assert!(
                        (lz - z).abs() <= 1e-5 * z.abs().max(1.0),
                        "{ctx}: {lz} vs {z}"
                    );
                    for (k, (&got, &exp)) in r.iter().zip(&want).enumerate() {
                        assert!(
                            (f64::from(got) - exp).abs() < 1e-6,
                            "{ctx} [{k}]: {got} {exp}"
                        );
                    }
                    if ok {
                        // the fast path's normalised weights and log total agree
                        for (k, (&x, &exp)) in w.iter().zip(&want).enumerate() {
                            let got = f64::from(x) / f64::from(fast_total);
                            assert!((got - exp).abs() < 1e-5, "{ctx} fast [{k}]: {got} {exp}");
                        }
                        let lt = f64::from(fast_total).ln();
                        assert!(
                            (lt - z).abs() < 1e-4 * z.abs().max(1.0),
                            "{ctx}: {lt} vs {z}"
                        );
                    } else {
                        rescued += 1;
                    }
                }
            }
        }
        // the fixture's Model 2 and 3 weights do underflow
        assert!(rescued >= 4, "{rescued}");
    }

    #[test]
    fn min_total_is_two_to_minus_80() {
        assert_eq!(MIN_TOTAL, 2f32.powi(-80));
        assert!(!total_ok(0.0) && !total_ok(f32::NAN) && !total_ok(-1.0));
        assert!(!total_ok(1e-30) && total_ok(1e-24) && total_ok(1.0));
    }

    #[test]
    fn fert_guard_only_where_underflow_is_possible() {
        use crate::priors::{DirPriors, DirTable};
        let h = Hyper::default();
        // defaults on a 10⁸-token corpus: no Model 3 product can underflow, the fast loop runs
        let p = Params::new(&h, 50_000).for_corpus(100_000_000, 5_000_000);
        assert!(!p.fert_guard && !p.checks);
        // totals: a tiny NULL prior and α on a huge corpus need the per-token check
        let small = Hyper {
            null_prior: NULL_PRIOR_RANGE.0,
            lex_alpha: 1e-20,
            ..h
        };
        assert!(
            Params::new(&small, 10)
                .for_corpus(100_000_000, 1_000_000)
                .checks
        );
        assert!(Params::new(&h, 10).checks); // without a corpus bound: checked
                                             // extreme accepted priors (a 1e25 jump record): the guard stays on
        let t = DirTable {
            jump: [(-64, 1e25)].into_iter().collect(),
            ..DirTable::default()
        };
        let pr = Arc::new(DirPriors::new(&t, &h, 3, 2));
        assert!(
            Params::with_priors(&h, 2, Some(pr))
                .for_corpus(4, 2)
                .fert_guard
        );
        // and the smallest accepted α
        let tiny = Hyper {
            lex_alpha: ALPHA_RANGE.0,
            ..h
        };
        assert!(
            Params::new(&tiny, 10)
                .for_corpus(100_000_000, 1_000_000)
                .fert_guard
        );
        assert!(!Params::new(&tiny, 10).for_corpus(10, 2).fert_guard); // 3e-35: still safe
    }

    #[test]
    fn normalize_log_edges() {
        let mut w = [0.0f32; 3];
        let z = normalize_log(&[-1000.0, -1000.0, f64::NEG_INFINITY], &mut w);
        assert_eq!(w, [0.5, 0.5, 0.0]);
        assert!((z - (-1000.0 + 2f64.ln())).abs() < 1e-9);
        let z = normalize_log(&[f64::NEG_INFINITY; 3], &mut w);
        assert_eq!(z, f64::NEG_INFINITY);
        assert!(w.iter().all(|&x| (x - 1.0 / 3.0).abs() < 1e-7));
    }

    #[test]
    fn weights_by_hand() {
        let mut lex = LexCounts::new(3, 4);
        let mut jump = JumpCounts::default();
        lex.inc(1, 2);
        lex.inc(1, 2);
        lex.inc(2, 3);
        lex.inc(NULL, 2);
        jump.inc(1);
        jump.inc(1);
        jump.inc(-1);
        let h = Hyper::default();
        let p = Params::new(&h, 3);
        let c = Plain {
            lex: &lex,
            jump: &jump,
        };
        let src = [1u32, 2];
        let ratio = vec![1.0f32; 3 * FERT_CATS];
        let phi = [0u16, 0];
        let t = Token {
            src: &src,
            f: 2,
            prev: -1,
            next: 2,
            phi: &phi,
            ratio: &ratio,
        };
        let mut cum = [0.0f32; 3];
        fill_weights(&c, &p, 1, &t, &mut cum);
        let tot = cumulate(&mut cum);
        let l = |a: f64, n: f64| (0.001 + a) / (0.003 + n);
        let w = [l(2.0, 2.0) * 0.4, l(0.0, 1.0) * 0.4, 0.2 * l(1.0, 1.0)];
        let mut s = 0.0;
        for k in 0..3 {
            s += w[k];
            assert!((f64::from(cum[k]) - s).abs() < 1e-5 * s, "{k}");
        }
        assert_eq!(tot, cum[2]);
        // Model 2: jumps from -1 and into 2
        fill_weights(&c, &p, 2, &t, &mut cum);
        cumulate(&mut cum);
        let jp = |d: f64| (0.5 + d) / (0.5 * 129.0 + 3.0);
        let w = [
            l(2.0, 2.0) * 0.8 * jp(2.0) * jp(0.0),
            l(0.0, 1.0) * 0.8 * jp(0.0) * jp(2.0),
            0.2 * l(1.0, 1.0) * jp(0.0),
        ];
        let mut s = 0.0;
        for k in 0..3 {
            s += w[k];
            assert!((f64::from(cum[k]) - s).abs() < 1e-5 * s, "{k}");
        }
    }
}
