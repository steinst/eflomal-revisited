//! Counter-based deterministic random numbers.
//!
//! The generator is **Philox4x32-10** (Salmon, Moraes, Dror and Shaw, *Parallel random numbers:
//! as easy as 1, 2, 3*, SC 2011): a bijection of a 128-bit counter under a 64-bit key, so any
//! draw is a pure function of its key and counter. Every draw of the sampler is addressed by
//!
//! * the key: SplitMix64(seed ⊕ (stream · 2⁵⁶)), split into two `u32` halves, where the
//!   [`Stream`] separates the families of draws (alignment sampling, fertility tables);
//! * the counter: `(chain, iteration, item, block)` with `chain = direction · 2¹⁶ + sampler`
//!   ([`chain_id`]), `iteration` 0 for the random initialisation and t ≥ 1 for sweep t
//!   (numbered across models), `item` the sentence (or, for fertility tables, the word type),
//!   and `block` the index of the 4-word Philox output within that item.
//!
//! Hence the numbers a sentence sees do not depend on which thread samples it, or when.
//! [`Draws`] turns one `(chain, iteration, item)` address into a sequential stream of `u32`,
//! four per Philox call.

/// Philox4x32 multipliers and Weyl key increments (Salmon et al. 2011, Random123).
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// Philox4x32 with 10 rounds: the 128-bit output for `ctr` under `key`.
#[inline]
pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        if round > 0 {
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
        let p0 = u64::from(M0) * u64::from(ctr[0]);
        let p1 = u64::from(M1) * u64::from(ctr[2]);
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
    }
    ctr
}

/// SplitMix64 finaliser (Steele, Lea and Flood 2014), used to spread the seed into a key.
#[inline]
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Families of random draws; each has its own Philox key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Stream {
    /// Alignment variables: random initialisation (iteration 0) and Gibbs sweeps (t ≥ 1).
    Align = 1,
    /// Model-3 fertility tables: item = source word type.
    Fertility = 2,
}

/// The counter's first word for a direction (0 = fwd, 1 = inv) and sampler.
#[inline]
pub fn chain_id(direction: u32, sampler: u32) -> u32 {
    debug_assert!(sampler < 1 << 16 && direction < 1 << 16);
    (direction << 16) | sampler
}

/// A keyed Philox generator for one stream of one seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    key: [u32; 2],
}

impl Rng {
    pub fn new(seed: u64, stream: Stream) -> Self {
        let k = splitmix64(seed ^ ((stream as u64) << 56));
        Rng {
            key: [k as u32, (k >> 32) as u32],
        }
    }

    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    /// The four `u32` at counter `ctr`.
    #[inline]
    pub fn block(&self, ctr: [u32; 4]) -> [u32; 4] {
        philox4x32_10(ctr, self.key)
    }

    /// The sequential stream of item `item` in `iteration` of `chain`.
    #[inline]
    pub fn draws(&self, chain: u32, iteration: u32, item: u32) -> Draws {
        Draws {
            key: self.key,
            ctr: [chain, iteration, item, 0],
            buf: [0; 4],
            pos: 4,
        }
    }
}

/// A sequential stream of `u32` for one `(chain, iteration, item)`: Philox at blocks 0, 1, 2, …
/// of the counter's last word, four words per block. The k-th `u32` taken is word k mod 4 of
/// block k div 4.
#[derive(Clone, Debug)]
pub struct Draws {
    key: [u32; 2],
    ctr: [u32; 4],
    buf: [u32; 4],
    pos: u8,
}

impl Draws {
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == 4 {
            self.buf = philox4x32_10(self.ctr, self.key);
            self.ctr[3] = self.ctr[3].wrapping_add(1);
            self.pos = 0;
        }
        let v = self.buf[self.pos as usize];
        self.pos += 1;
        v
    }

    /// Uniform in [0, 1) with 24 random bits (exactly representable in `f32`).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        u32_to_f32(self.next_u32())
    }

    /// Uniform in [0, 1) with 53 random bits (two `u32`).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let hi = u64::from(self.next_u32() >> 5); // 27 bits
        let lo = u64::from(self.next_u32() >> 6); // 26 bits
        ((hi << 26) | lo) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in (0, 1]: never 0, for logarithms.
    #[inline]
    pub fn next_f64_open0(&mut self) -> f64 {
        1.0 - self.next_f64()
    }

    /// Uniform integer in 0 … n−1 (n ≥ 1) by multiply-shift of one `u32`; the bias is below
    /// n / 2³², negligible for sentence lengths.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        ((u64::from(self.next_u32()) * u64::from(n)) >> 32) as u32
    }

    /// Standard normal by the Box–Muller transform (one value per two uniforms; the second
    /// value of the pair is discarded so that the stream position depends only on the number
    /// of calls).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64_open0();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Gamma(shape, 1) for shape > 0: Marsaglia and Tsang (2000) for shape ≥ 1; for shape < 1
    /// Gamma(shape + 1) · U^(1/shape).
    pub fn gamma(&mut self, shape: f64) -> f64 {
        debug_assert!(shape > 0.0);
        if shape < 1.0 {
            let g = self.gamma(shape + 1.0);
            return g * self.next_f64_open0().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.next_f64_open0();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

/// The top 24 bits of `x` as a uniform in [0, 1).
#[inline]
pub fn u32_to_f32(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

/// Categorical draw from cumulative weights `cum` (non-decreasing, `cum[k]` = sum of the
/// weights of candidates 0 … k) with a uniform `u` in [0, 1): the first k with
/// `cum[k] > u · cum[last]`. If rounding leaves none, the last candidate with non-zero weight
///. `cum` must be non-empty with a positive total.
#[inline]
pub fn draw_cumulative(cum: &[f32], u: f32) -> usize {
    let n = cum.len();
    debug_assert!(n > 0);
    let total = cum[n - 1];
    let r = u * total;
    // Linear scan: sentences are short and this is what the sampler does per token.
    for (k, &s) in cum.iter().enumerate() {
        if s > r {
            return k;
        }
    }
    last_nonzero(cum)
}

/// The last candidate whose weight (difference of consecutive cumulative sums) is non-zero.
fn last_nonzero(cum: &[f32]) -> usize {
    for k in (0..cum.len()).rev() {
        let prev = if k == 0 { 0.0 } else { cum[k - 1] };
        if cum[k] > prev {
            return k;
        }
    }
    cum.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known-answer vectors of Random123 (kat_vectors, philox4x32_10).
    #[test]
    fn philox_known_answers() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn splitmix_known_answer() {
        // First output of SplitMix64 seeded with 0 (Vigna's reference implementation).
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn streams_are_reproducible_and_distinct() {
        let a = Rng::new(7, Stream::Align);
        let b = Rng::new(7, Stream::Align);
        let f = Rng::new(7, Stream::Fertility);
        let other = Rng::new(8, Stream::Align);
        assert_eq!(a, b);
        assert_ne!(a.key(), f.key());
        assert_ne!(a.key(), other.key());
        let take = |r: &Rng, it, item| {
            let mut d = r.draws(chain_id(1, 2), it, item);
            (0..10).map(|_| d.next_u32()).collect::<Vec<_>>()
        };
        assert_eq!(take(&a, 3, 5), take(&b, 3, 5));
        assert_ne!(take(&a, 3, 5), take(&a, 3, 6));
        assert_ne!(take(&a, 3, 5), take(&a, 4, 5));
        assert_ne!(take(&a, 3, 5), take(&f, 3, 5));
        // Stream layout: word k is word k % 4 of block k / 4.
        let s = take(&a, 3, 5);
        for k in 0..10 {
            let blk = a.block([chain_id(1, 2), 3, 5, (k / 4) as u32]);
            assert_eq!(s[k], blk[k % 4]);
        }
    }

    #[test]
    fn chain_ids_do_not_collide() {
        let mut seen = std::collections::HashSet::new();
        for d in 0..2 {
            for s in 0..300 {
                assert!(seen.insert(chain_id(d, s)));
            }
        }
    }

    /// Pearson χ² statistic of observed counts against equal expected counts.
    fn chi2_uniform(counts: &[u64]) -> f64 {
        let n: u64 = counts.iter().sum();
        let e = n as f64 / counts.len() as f64;
        counts.iter().map(|&c| (c as f64 - e).powi(2) / e).sum()
    }

    #[test]
    fn uniform_f32_is_uniform() {
        // 64 bins, 640 000 draws over many items: χ² with 63 dof has mean 63, sd ≈ 11.2;
        // 130 is beyond 6 sd.
        let r = Rng::new(42, Stream::Align);
        let mut counts = [0u64; 64];
        for item in 0..10_000 {
            let mut d = r.draws(0, 1, item);
            for _ in 0..64 {
                let u = d.next_f32();
                assert!((0.0..1.0).contains(&u));
                counts[(u * 64.0) as usize] += 1;
            }
        }
        let x2 = chi2_uniform(&counts);
        assert!(x2 < 130.0, "chi2 = {x2}");
        // f64 and below()
        let mut d = r.draws(0, 2, 0);
        let mut c2 = [0u64; 7];
        let mut sum = 0.0;
        for _ in 0..70_000 {
            let x = d.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
            c2[d.below(7) as usize] += 1;
        }
        assert!((sum / 70_000.0 - 0.5).abs() < 0.01);
        assert!(chi2_uniform(&c2) < 40.0, "{c2:?}");
    }

    #[test]
    fn categorical_draw_matches_weights() {
        let w = [0.5f32, 0.0, 2.0, 1.0, 0.25, 0.0, 4.25];
        let mut cum = Vec::new();
        let mut s = 0.0;
        for &x in &w {
            s += x;
            cum.push(s);
        }
        let total: f32 = w.iter().sum();
        let r = Rng::new(3, Stream::Align);
        let n = 400_000u64;
        let mut counts = vec![0u64; w.len()];
        let mut d = r.draws(0, 1, 0);
        for _ in 0..n {
            counts[draw_cumulative(&cum, d.next_f32())] += 1;
        }
        assert_eq!(counts[1], 0);
        assert_eq!(counts[5], 0);
        // χ² over the candidates with non-zero weight (4 dof; 30 is far in the tail).
        let mut x2 = 0.0;
        for (k, &x) in w.iter().enumerate() {
            if x > 0.0 {
                let e = n as f64 * f64::from(x / total);
                x2 += (counts[k] as f64 - e).powi(2) / e;
            }
        }
        assert!(x2 < 30.0, "chi2 = {x2}, counts {counts:?}");
    }

    #[test]
    fn categorical_draw_edges() {
        // u = 0 picks the first candidate with non-zero weight.
        assert_eq!(draw_cumulative(&[0.0, 0.0, 1.0, 2.0], 0.0), 2);
        // u just below 1 picks the last non-zero candidate.
        let u_max = u32_to_f32(u32::MAX);
        assert_eq!(draw_cumulative(&[1.0, 2.0, 3.0, 3.0], u_max), 2);
        // Rounding that leaves no cum[k] > r falls back to the last non-zero weight.
        assert_eq!(last_nonzero(&[1.0, 1.0, 2.0, 2.0]), 2);
        assert_eq!(draw_cumulative(&[5.0], 0.999), 0);
    }

    #[test]
    fn gamma_moments() {
        // Gamma(k, 1): mean k, variance k. Check for shapes below and above 1.
        let r = Rng::new(11, Stream::Fertility);
        for &shape in &[0.05, 0.5, 1.0, 2.5, 30.0] {
            let n = 200_000;
            let mut d = r.draws(0, 1, (shape * 100.0) as u32);
            let (mut s, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let g = d.gamma(shape);
                assert!(g >= 0.0 && g.is_finite());
                s += g;
                s2 += g * g;
            }
            let mean = s / n as f64;
            let var = s2 / n as f64 - mean * mean;
            // Standard error of the mean is sqrt(k/n); allow 6 of them, and 5 % on the variance
            // (10 % for the very skewed small shapes).
            let se = (shape / n as f64).sqrt();
            assert!(
                (mean - shape).abs() < 6.0 * se,
                "shape {shape}: mean {mean}"
            );
            let tol = if shape < 1.0 { 0.1 } else { 0.05 };
            assert!(
                (var - shape).abs() < tol * shape,
                "shape {shape}: var {var}"
            );
        }
    }

    #[test]
    fn normal_moments() {
        let r = Rng::new(5, Stream::Fertility);
        let mut d = r.draws(1, 1, 1);
        let n = 200_000;
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let x = d.normal();
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        assert!(mean.abs() < 0.015, "{mean}");
        assert!((s2 / n as f64 - 1.0).abs() < 0.02);
    }
}
