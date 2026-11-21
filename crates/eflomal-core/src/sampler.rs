//! The parallel deterministic sampler: block-parallel sweeps
//! within one chain, approximate distributed Gibbs sampling in the style of AD-LDA.
//!
//! The corpus is cut into a fixed number of blocks of contiguous sentences ([`Layout`],
//! independent of the thread count). A sweep runs in rounds: in round r every block samples its
//! next `round` sentences against the chain's shared counts as they were at the end of round
//! r − 1 plus its own changes in this round, which it keeps in a private overlay of count deltas
//! ([`Overlay`]). After the round the overlays are added to the shared counts in block order
//! (integer additions, so the counts after a round do not depend on the order either) and
//! cleared. With one block every sentence sees the exact counts, so the chain equals the
//! sequential oracle ([`crate::oracle::run_chain`]) bit for bit; the tests check it.
//!
//! Random numbers are those of the oracle: [`Stream::Align`] at `(chain, t, sentence)`, so a
//! sentence draws the same numbers whichever thread samples it. Fertility tables are drawn per
//! word type from [`Stream::Fertility`] in parallel chunks, with the same numbers as the oracle.
//!
//! Work items: the chains of a run share one rayon pool (one task per chain); within a chain
//! each round is one task per block. Each block owns its scratch buffers, which are reused
//! across sentences, rounds and sweeps; the per-sentence loop does not allocate (the overlay
//! grows only when a round touches more keys than any earlier one).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::bitext::View;
use crate::cancel;
use crate::counts::{
    JumpCounts, LexCounts, SentenceGather, FERT_CATS, F_MAX, JUMP_BUCKETS, MAX_JUMP, NULL,
};
use crate::model::{
    cumulate, fill_weights, ratio_of, rescue_weights, total_ok, Counts, Gathered, JumpTable,
    Params, Token, EPS,
};
use crate::oracle::{
    acc_offsets, accumulate_into, chain_of, fert_counts, jump_counts, last_model, lex_counts,
    null_fraction, sentence_fertility, Chain, ChainConfig,
};
use crate::pipeline::Progress;
use crate::rng::{draw_cumulative, Rng, Stream};
use crate::timing::{self, Counter, Phase};

/// Blocks per chain (B).
pub const BLOCKS: usize = 16;
/// Sentences per block between two merges (R).
pub const ROUND: usize = 64;

/// How a chain's sweep is divided: `blocks` contiguous blocks, merged every `round` sentences
/// per block. The result for a seed depends on the layout, never on the thread count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub blocks: usize,
    pub round: usize,
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            blocks: BLOCKS,
            round: ROUND,
        }
    }
}

/// Word types per parallel work item when the fertility tables are drawn.
const FERT_CHUNK: usize = 4096;
/// Sentences per parallel work item of the random initialisation.
const INIT_CHUNK: usize = 4096;

// ------------------------------------------------------------------------------------------
// Overlay of count deltas

const NO_KEY: u64 = u64::MAX;
/// Target id used for the key of a row total n(e) (never a real target id).
const TOTAL: u32 = u32::MAX;

#[inline]
fn key(e: u32, f: u32) -> u64 {
    (u64::from(e) << 32) | u64::from(f)
}

/// A block's changes in the current round: lexical deltas δc(e, f), row-total deltas δn(e)
/// (stored under the key (e, [`TOTAL`])), jump deltas per bucket and their total. The lexical
/// map is open addressing with linear probing, kept at most a quarter full; `used` lists the
/// occupied slots in insertion order, for clearing and for the merge. `dirty` has one bit per
/// source id with lexical deltas, so that reads of untouched rows skip the map.
#[derive(Clone, Debug)]
pub struct Overlay {
    keys: Vec<u64>,
    vals: Vec<i32>,
    used: Vec<u32>,
    shift: u32,
    dirty: Vec<u64>,
    jump: [i32; JUMP_BUCKETS],
    jump_total: i32,
}

impl Default for Overlay {
    fn default() -> Self {
        Overlay::with_capacity(1024)
    }
}

impl Overlay {
    fn with_capacity(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(16);
        Overlay {
            keys: vec![NO_KEY; cap],
            vals: vec![0; cap],
            used: Vec::new(),
            shift: 64 - cap.trailing_zeros(),
            dirty: Vec::new(),
            jump: [0; JUMP_BUCKETS],
            jump_total: 0,
        }
    }

    #[inline]
    fn home(&self, k: u64) -> usize {
        (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn get(&self, k: u64) -> i32 {
        let mask = self.keys.len() - 1;
        let mut i = self.home(k);
        // the first slot decides almost always (load ≤ 1/4); select without a branch
        let x = self.keys[i];
        let v = if x == k { self.vals[i] } else { 0 };
        if x == k || x == NO_KEY {
            return v;
        }
        loop {
            i = (i + 1) & mask;
            let x = self.keys[i];
            if x == k {
                return self.vals[i];
            }
            if x == NO_KEY {
                return 0;
            }
        }
    }

    #[inline]
    fn add(&mut self, k: u64, d: i32) {
        let mask = self.keys.len() - 1;
        let mut i = self.home(k);
        loop {
            let x = self.keys[i];
            if x == k {
                self.vals[i] += d;
                return;
            }
            if x == NO_KEY {
                break;
            }
            i = (i + 1) & mask;
        }
        if (self.used.len() + 1) * 4 > self.keys.len() {
            self.grow();
            return self.add(k, d);
        }
        self.keys[i] = k;
        self.vals[i] = d;
        self.used.push(i as u32);
    }

    fn grow(&mut self) {
        let mut g = Overlay::with_capacity(self.keys.len() * 2);
        for &s in &self.used {
            g.add(self.keys[s as usize], self.vals[s as usize]);
        }
        self.keys = g.keys;
        self.vals = g.vals;
        self.used = g.used;
        self.shift = g.shift;
    }

    /// Size the dirty-row bitset for `n_source` source ids (no-op if already large enough).
    fn ensure_rows(&mut self, n_source: usize) {
        let words = n_source.div_ceil(64);
        if self.dirty.len() < words {
            self.dirty.resize(words, 0);
        }
    }

    #[inline]
    fn is_dirty(&self, e: u32) -> bool {
        self.dirty[(e >> 6) as usize] & (1u64 << (e & 63)) != 0
    }

    /// Record c(e, f) += d and n(e) += d.
    #[inline]
    fn lex_add(&mut self, e: u32, f: u32, d: i32) {
        self.dirty[(e >> 6) as usize] |= 1u64 << (e & 63);
        self.add(key(e, f), d);
        self.add(key(e, TOTAL), d);
    }

    /// δn(e).
    #[inline]
    fn total(&self, e: u32) -> i32 {
        if self.is_dirty(e) {
            self.get(key(e, TOTAL))
        } else {
            0
        }
    }

    #[inline]
    fn jump_add(&mut self, m: i32, d: i32) {
        self.jump[JumpCounts::bucket(m)] += d;
        self.jump_total += d;
    }

    /// Add the deltas to the shared counts (in insertion order) and clear the overlay.
    fn merge_into(&mut self, lex: &mut LexCounts, jump: &mut JumpCounts) {
        for &s in &self.used {
            let s = s as usize;
            let (k, d) = (self.keys[s], self.vals[s]);
            self.keys[s] = NO_KEY;
            self.vals[s] = 0;
            self.dirty[(k >> 38) as usize] = 0;
            let f = k as u32;
            if f != TOTAL && d != 0 {
                lex.add((k >> 32) as u32, f, d);
            }
        }
        self.used.clear();
        if self.jump_total != 0 || self.jump.iter().any(|&x| x != 0) {
            for (b, d) in self.jump.iter_mut().enumerate() {
                if *d != 0 {
                    jump.add(b as i32 - MAX_JUMP, *d);
                    *d = 0;
                }
            }
            self.jump_total = 0;
        }
    }

    /// Heap bytes held.
    pub fn memory_bytes(&self) -> usize {
        self.keys.capacity() * 8 + self.vals.capacity() * 4 + self.used.capacity() * 4
    }
}

/// The counts one target token's conditional sees: the chain's shared counts plus the block's
/// overlay, minus the token's own contribution, which is subtracted on the fly as in
/// [`crate::decode`] (the lexical count of (e_old, f) and n(e_old) less one, the one or two
/// jumps through the token less one each). These are exactly the counts after a real removal,
/// so the weights have the oracle's bits, but a token whose alignment does not change never
/// touches the overlay. `rinv` caches 1 / (A + n(e)) of the sentence's positions (index I =
/// NULL) with the token included; `r_old` is the value of e_old with the token removed.
#[derive(Clone, Copy)]
struct Overlaid<'a> {
    lex: &'a LexCounts,
    jump: &'a JumpCounts,
    ov: &'a Overlay,
    rinv: &'a [f32],
    e_old: u32,
    f: u32,
    r_old: f32,
    /// Buckets of the jumps through the token (`usize::MAX` if unused).
    jb: [usize; 2],
    n_jumps: u32,
    /// β(m) + d(m) of the shared counts plus the overlay, without the token's own jumps.
    jt: &'a [f32; JUMP_BUCKETS],
}

impl Counts for Overlaid<'_> {
    #[inline]
    fn lex(&self, e: u32, f: u32) -> u32 {
        self.lex
            .get(e, f)
            .wrapping_add_signed(self.ov.get(key(e, f)))
            .wrapping_sub(u32::from(e == self.e_old && f == self.f))
    }
    #[inline]
    fn total(&self, e: u32) -> u32 {
        self.lex.total(e).wrapping_add_signed(self.ov.total(e)) - u32::from(e == self.e_old)
    }
    #[inline]
    fn recip(&self, _p: &Params, pos: usize, e: u32) -> f32 {
        if e == self.e_old {
            self.r_old
        } else {
            self.rinv[pos]
        }
    }
    #[inline]
    fn jump(&self, m: i32) -> u32 {
        let b = JumpCounts::bucket(m);
        self.jump.buckets()[b].wrapping_add_signed(self.ov.jump[b])
            - u32::from(b == self.jb[0])
            - u32::from(b == self.jb[1])
    }
    #[inline]
    fn jump_total(&self) -> u32 {
        self.jump.total().wrapping_add_signed(self.ov.jump_total) - self.n_jumps
    }
    #[inline(always)]
    fn jump_term(&self, m: i32, _alpha: f32) -> f32 {
        self.jt[JumpCounts::bucket(m)]
    }
}

/// d(bucket b) of the shared counts plus the overlay.
#[inline]
fn jump_count(jump: &JumpCounts, ov: &Overlay, b: usize) -> u32 {
    jump.buckets()[b].wrapping_add_signed(ov.jump[b])
}

// ------------------------------------------------------------------------------------------
// Blocks

/// One block of a chain: its sentences, overlay and scratch buffers.
struct Block {
    /// Sentences `lo .. hi`.
    lo: usize,
    hi: usize,
    ov: Overlay,
    w: Vec<f32>,
    phi: Vec<u16>,
    rinv: Vec<f32>,
    /// The sentence's distinct words and its gathered lexical counts (shared + overlay; see
    /// [`Gathered`]), kept up to date on every move of the sentence.
    g: SentenceGather,
    mat: Vec<u32>,
    /// 1 / (A + n(e)) per distinct source word of the sentence (with `g`).
    rd: Vec<f32>,
    /// R(e_i, min(φ_i, F_MAX)) per position of the sentence (Model 3, with `g`).
    rr: Vec<f32>,
    /// The jump terms of the shared counts plus the overlay, rebuilt every round and kept up
    /// to date on every move.
    jt: JumpTable,
    /// Measurement only ([`crate::timing`]): tokens sampled, alignment changes, and the time
    /// of the block's last round in ns.
    tokens: u64,
    moves: u64,
    round_ns: u64,
    /// Measurement only: ns from the round's fork to this block's start.
    start_ns: u64,
}

/// Sentence range of block `b` of `n` over K sentences: ⌊bK/n⌋ … ⌊(b+1)K/n⌋ − 1.
pub fn block_bounds(k: usize, n: usize, b: usize) -> (usize, usize) {
    let at = |x: usize| ((x as u128 * k as u128) / n as u128) as usize;
    (at(b), at(b + 1))
}

/// The shared, read-only state of a round.
struct Shared<'a> {
    view: &'a View<'a>,
    lex: &'a LexCounts,
    jump: &'a JumpCounts,
    ratio: &'a [f32],
    p: &'a Params,
    model: u8,
    rng: &'a Rng,
    chain: u32,
    t: u32,
}

/// 1 / (A + n(e)) from the shared counts plus the overlay, with the formula of
/// [`Counts::recip`]'s default.
#[inline]
fn recip_of(sh: &Shared<'_>, ov: &Overlay, e: u32) -> f32 {
    1.0 / (sh.p.lex_den_of(e) + sh.lex.total(e).wrapping_add_signed(ov.total(e)) as f32)
}

/// Update the cached reciprocal of every position of `src` holding word `e` (of the NULL slot
/// `src.len()` if e is NULL).
#[inline]
fn refresh(sh: &Shared<'_>, ov: &Overlay, src: &[u32], rinv: &mut [f32], e: u32) {
    let r = recip_of(sh, ov, e);
    if e == NULL {
        rinv[src.len()] = r;
    } else {
        for (x, &w) in rinv.iter_mut().zip(src) {
            if w == e {
                *x = r;
            }
        }
    }
}

/// Sample the alignment `a` of sentence k against the shared counts plus the block's overlay
/// (the same steps and numbers as the oracle's sweep; the token's removal is
/// virtual, see [`Overlaid`]). A non-empty `acc` (the sentence's J · (I + 1) accumulator
/// values) gets every token's normalised conditional added.
fn sample_sentence(sh: &Shared<'_>, k: usize, a: &mut [u16], blk: &mut Block, acc: &mut [f32]) {
    let (src, tgt) = sh.view.pair(k);
    let i_len = src.len();
    if tgt.is_empty() || i_len == 0 {
        return;
    }
    let model = sh.model;
    let Block {
        ov,
        w,
        phi,
        rinv,
        g,
        mat,
        rd,
        rr,
        jt,
        tokens,
        moves,
        ..
    } = blk;
    *tokens += a.len() as u64;
    let mut d = sh.rng.draws(sh.chain, sh.t, k as u32);
    if model >= 3 {
        sentence_fertility(i_len, a, phi);
    }
    w.resize(i_len + 1, 0.0);
    // gather the sentence's lexical counts (shared + overlay) and reciprocals once
    let gathered = g.index(src, tgt);
    let rows = g.rows();
    rinv.clear();
    if gathered {
        rd.clear();
        rd.extend(g.src_ids.iter().map(|&e| recip_of(sh, ov, e)));
        rr.clear();
        if model >= 3 {
            rr.extend(
                src.iter()
                    .zip(&phi[..])
                    .map(|(&e, &n)| ratio_of(sh.ratio, e, n)),
            );
        }
    } else {
        rinv.extend(src.iter().map(|&e| recip_of(sh, ov, e)));
        rinv.push(recip_of(sh, ov, NULL));
    }
    if gathered {
        g.fill(sh.lex, mat);
        for (di, &e) in g.src_ids.iter().enumerate() {
            if ov.is_dirty(e) {
                for (dj, &f) in g.tgt_ids.iter().enumerate() {
                    let x = &mut mat[dj * rows + di];
                    *x = x.wrapping_add_signed(ov.get(key(e, f)));
                }
            }
        }
    }
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
        // virtual removal of a_j
        let (e_old, old_pos) = if old == EPS {
            (NULL, i_len)
        } else {
            (src[old as usize], old as usize)
        };
        let (jb, n_jumps) = if model < 2 {
            ([usize::MAX; 2], 0)
        } else if old == EPS {
            ([JumpCounts::bucket(next - prev), usize::MAX], 1)
        } else {
            let o = i32::from(old);
            (
                [JumpCounts::bucket(o - prev), JumpCounts::bucket(next - o)],
                2,
            )
        };
        let n_old = sh.lex.total(e_old).wrapping_add_signed(ov.total(e_old)) - 1;
        let r_old = 1.0 / (sh.p.lex_den_of(e_old) + n_old as f32);
        if model >= 3 && old != EPS {
            phi[old_pos] -= 1;
            if gathered {
                rr[old_pos] = ratio_of(sh.ratio, e_old, phi[old_pos]);
            }
        }
        if model >= 2 {
            jt.patch(sh.p, jb, |b| jump_count(sh.jump, ov, b));
        }
        // weights and draw
        let tok = Token {
            src,
            f,
            prev,
            next,
            phi,
            ratio: sh.ratio,
        };
        let c = Overlaid {
            lex: sh.lex,
            jump: sh.jump,
            ov,
            rinv,
            e_old,
            f,
            r_old,
            jb,
            n_jumps,
            jt: &jt.t,
        };
        let col_at = g.tidx.get(j).map_or(0, |&dj| dj as usize * rows);
        // the weights, with the token's own count and reciprocal removed from the gathered
        // counts (and restored) as before; on ordinary input exactly the old sequence
        let di_old = if gathered {
            g.sidx[old_pos] as usize
        } else {
            0
        };
        if gathered {
            mat[col_at + di_old] -= 1;
            let r_with = std::mem::replace(&mut rd[di_old], r_old);
            let gc = Gathered {
                c,
                col: &mat[col_at..col_at + rows],
                sidx: &g.sidx,
                rinv: rd,
                rr,
            };
            fill_weights(&gc, sh.p, model, &tok, w);
            mat[col_at + di_old] += 1;
            rd[di_old] = r_with;
        } else {
            fill_weights(&c, sh.p, model, &tok, w);
        }
        // (Params::for_corpus proved the check unnecessary for ordinary input: no compare)
        let ok = if acc.is_empty() {
            let total = cumulate(w);
            !sh.p.checks || total_ok(total)
        } else {
            !sh.p.checks || total_ok(w.iter().sum())
        };
        if !ok {
            // rare: recompute in log space from the same state
            if gathered {
                mat[col_at + di_old] -= 1;
                let r_with = std::mem::replace(&mut rd[di_old], r_old);
                let gc = Gathered {
                    c,
                    col: &mat[col_at..col_at + rows],
                    sidx: &g.sidx,
                    rinv: rd,
                    rr,
                };
                rescue_weights(&gc, sh.p, model, &tok, w);
                mat[col_at + di_old] += 1;
                rd[di_old] = r_with;
            } else {
                rescue_weights(&c, sh.p, model, &tok, w);
            }
            if acc.is_empty() {
                cumulate(w);
            }
        }
        if !acc.is_empty() {
            accumulate_into(&mut acc[j * (i_len + 1)..(j + 1) * (i_len + 1)], w);
            cumulate(w);
        }
        if model >= 2 {
            jt.restore();
        }
        let pick = draw_cumulative(w, d.next_f32());
        let new = if pick == i_len { EPS } else { pick as u16 };
        if model >= 3 && new != EPS {
            phi[pick] += 1;
            if gathered {
                rr[pick] = ratio_of(sh.ratio, src[pick], phi[pick]);
            }
        }
        if new != old {
            // record the move in the overlay
            *moves += 1;
            a[j] = new;
            let e_new = if new == EPS { NULL } else { src[pick] };
            if gathered {
                mat[col_at + g.sidx[old_pos] as usize] -= 1;
                mat[col_at + g.sidx[pick] as usize] += 1;
            }
            ov.lex_add(e_old, f, -1);
            ov.lex_add(e_new, f, 1);
            if model >= 2 {
                if old == EPS {
                    ov.jump_add(next - prev, -1);
                } else {
                    let o = i32::from(old);
                    ov.jump_add(o - prev, -1);
                    ov.jump_add(next - o, -1);
                }
                if new == EPS {
                    ov.jump_add(next - prev, 1);
                } else {
                    let n = pick as i32;
                    ov.jump_add(n - prev, 1);
                    ov.jump_add(next - n, 1);
                }
                let nb = if new == EPS {
                    [JumpCounts::bucket(next - prev), usize::MAX]
                } else {
                    let n = pick as i32;
                    [JumpCounts::bucket(n - prev), JumpCounts::bucket(next - n)]
                };
                for b in jb.into_iter().chain(nb) {
                    if b != usize::MAX {
                        jt.set(sh.p, b, jump_count(sh.jump, ov, b));
                    }
                }
            }
            if e_new != e_old {
                if gathered {
                    rd[g.sidx[old_pos] as usize] = recip_of(sh, ov, e_old);
                    rd[g.sidx[pick] as usize] = recip_of(sh, ov, e_new);
                } else {
                    refresh(sh, ov, src, rinv, e_old);
                    refresh(sh, ov, src, rinv, e_new);
                }
            }
        }
        if new != EPS {
            prev = pick as i32;
        }
    }
}

/// Block `blk`'s part of round `r0` (its sentences `lo + r0 … lo + r0 + round − 1`): `a` and
/// `acc` are the block's alignment and accumulator slices (from its first sentence on).
/// `fork` is when the round was started (measurement only).
#[allow(clippy::too_many_arguments)]
fn block_round(
    sh: &Shared<'_>,
    blk: &mut Block,
    a: &mut [u16],
    acc: &mut [f32],
    acc_off: Option<&[usize]>,
    r0: usize,
    round: usize,
    fork: Option<Instant>,
) {
    let off = sh.view.tgt.offsets();
    let t0 = timing::start();
    if let (Some(f), Some(t)) = (fork, t0) {
        blk.start_ns = t.duration_since(f).as_nanos() as u64;
    }
    if sh.model >= 2 {
        let ov = &blk.ov;
        blk.jt.fill(sh.p, |b| jump_count(sh.jump, ov, b));
    }
    let base = off[blk.lo];
    let lo = (blk.lo + r0).min(blk.hi);
    let hi = (blk.lo + r0 + round).min(blk.hi);
    for k in lo..hi {
        let acc_k: &mut [f32] = match acc_off {
            Some(ao) => {
                let b = ao[blk.lo];
                &mut acc[ao[k] - b..ao[k + 1] - b]
            }
            None => &mut [],
        };
        sample_sentence(sh, k, &mut a[off[k] - base..off[k + 1] - base], blk, acc_k);
    }
    if let Some(t0) = t0 {
        blk.round_ns = t0.elapsed().as_nanos() as u64;
    }
}

/// One block-parallel sweep of Model `model` with the alignment draws of sweep `t`.
#[allow(clippy::too_many_arguments)]
fn sweep(
    view: &View<'_>,
    ch: &mut Chain,
    blocks: &mut [Block],
    round: usize,
    p: &Params,
    model: u8,
    rng: &Rng,
    chain: u32,
    t: u32,
    acc_off: Option<&[usize]>,
) {
    let off = view.tgt.offsets();
    let Chain {
        align,
        lex,
        jump,
        ratio,
        acc,
        ..
    } = ch;
    // the blocks' alignment (and accumulator) slices; blocks are contiguous and in order
    type Part<'b> = (&'b mut Block, &'b mut [u16], &'b mut [f32]);
    let mut parts: Vec<Part<'_>> = Vec::with_capacity(blocks.len());
    let mut rest: &mut [u16] = align;
    let mut rest_acc: &mut [f32] = if acc_off.is_some() { acc } else { &mut [] };
    let mut at = 0;
    for blk in blocks.iter_mut() {
        let (lo, hi) = (off[blk.lo], off[blk.hi]);
        debug_assert_eq!(lo, at);
        let (mine, tail) = rest.split_at_mut(hi - lo);
        rest = tail;
        at = hi;
        let n_acc = acc_off.map_or(0, |ao| ao[blk.hi] - ao[blk.lo]);
        let (mine_acc, tail_acc) = std::mem::take(&mut rest_acc).split_at_mut(n_acc);
        rest_acc = tail_acc;
        parts.push((blk, mine, mine_acc));
    }
    let n_source = lex.n_source();
    for (blk, _, _) in parts.iter_mut() {
        blk.ov.ensure_rows(n_source);
    }
    let longest = parts.iter().map(|(b, _, _)| b.hi - b.lo).max().unwrap_or(0);
    let mut r0 = 0;
    while r0 < longest {
        cancel::check();
        {
            let sh = Shared {
                view,
                lex,
                jump,
                ratio,
                p,
                model,
                rng,
                chain,
                t,
            };
            let fork = timing::start();
            parts
                .par_iter_mut()
                .with_max_len(1)
                .for_each(|(blk, a, acc)| block_round(&sh, blk, a, acc, acc_off, r0, round, fork));
            if let Some(f) = fork {
                timing::add(Phase::RoundWall, f.elapsed().as_nanos() as u64);
                let last = parts.iter().map(|(b, _, _)| b.start_ns).max();
                timing::add(Phase::RoundLastStart, last.unwrap_or(0));
            }
        }
        let t0 = timing::start();
        for (blk, _, _) in parts.iter_mut() {
            blk.ov.merge_into(lex, jump);
        }
        if t0.is_some() {
            timing::stop(Phase::Merge, t0);
            let ns = parts.iter().map(|(b, _, _)| b.round_ns);
            timing::add(Phase::RoundCritical, ns.clone().max().unwrap_or(0));
            timing::add(Phase::BlockWork, ns.sum());
            timing::count(Counter::Rounds, 1);
        }
        r0 += round;
    }
    if timing::enabled() {
        for (blk, _, _) in parts.iter_mut() {
            timing::count(Counter::Tokens, std::mem::take(&mut blk.tokens));
            timing::count(Counter::Moves, std::mem::take(&mut blk.moves));
        }
    }
}

// ------------------------------------------------------------------------------------------
// Chain

/// [`crate::oracle::init_alignment`], computed in parallel chunks of sentences (same result).
pub fn init_alignment(view: &View<'_>, rng: &Rng, chain: u32, p_null: f32) -> Vec<u16> {
    let off = view.tgt.offsets();
    let k_n = view.len();
    let mut a = vec![EPS; view.tgt.n_tokens()];
    let mut parts: Vec<(usize, &mut [u16])> = Vec::new();
    let mut rest: &mut [u16] = &mut a;
    for c in 0..k_n.div_ceil(INIT_CHUNK) {
        let (lo, hi) = (c * INIT_CHUNK, ((c + 1) * INIT_CHUNK).min(k_n));
        let (mine, tail) = rest.split_at_mut(off[hi] - off[lo]);
        rest = tail;
        parts.push((lo, mine));
    }
    parts.into_par_iter().for_each(|(lo, part)| {
        let base = off[lo];
        let hi = (lo + INIT_CHUNK).min(k_n);
        for k in lo..hi {
            let i_len = view.src.line_len(k) as u32;
            let mut d = rng.draws(chain, 0, k as u32);
            for x in &mut part[off[k] - base..off[k + 1] - base] {
                let u = d.next_f32();
                *x = if i_len == 0 || u < p_null {
                    EPS
                } else {
                    d.below(i_len) as u16
                };
            }
        }
    });
    a
}

/// [`crate::oracle::fertility_ratios`] computed in parallel chunks of word types (same result:
/// every type has its own random stream).
pub fn fertility_ratios(
    fc: &crate::counts::FertCounts,
    n_types: usize,
    p: &Params,
    rng: &Rng,
    chain: u32,
    t: u32,
) -> Vec<f32> {
    const RATIO_MIN: f64 = 1e-30;
    const RATIO_MAX: f64 = 1e30;
    const GAMMA_MIN: f64 = 1e-300;
    let mut ratio = vec![1.0f32; n_types * FERT_CATS];
    ratio
        .par_chunks_mut(FERT_CHUNK * FERT_CATS)
        .enumerate()
        .for_each(|(c, part)| {
            let mut g = [0.0f64; FERT_CATS];
            for (r, row) in part.chunks_mut(FERT_CATS).enumerate() {
                let e = c * FERT_CHUNK + r;
                if e == 0 {
                    continue; // NULL: all 1
                }
                let mut d = rng.draws(chain, t, e as u32);
                for (n, x) in g.iter_mut().enumerate() {
                    *x = d
                        .gamma(p.fert_alpha_of(e as u32, n) + f64::from(fc.get(e as u32, n)))
                        .max(GAMMA_MIN);
                }
                for n in 0..F_MAX {
                    row[n] = (g[n + 1] / g[n]).clamp(RATIO_MIN, RATIO_MAX) as f32;
                }
                row[F_MAX] = 1.0;
            }
        });
    ratio
}

/// The blocks of a chain over `k_n` sentences.
fn new_blocks(k_n: usize, layout: Layout) -> Vec<Block> {
    (0..layout.blocks)
        .map(|b| {
            let (lo, hi) = block_bounds(k_n, layout.blocks, b);
            Block {
                lo,
                hi,
                ov: Overlay::default(),
                w: Vec::new(),
                phi: Vec::new(),
                rinv: Vec::new(),
                g: SentenceGather::default(),
                mat: Vec::new(),
                rd: Vec::new(),
                rr: Vec::new(),
                jt: JumpTable::default(),
                tokens: 0,
                moves: 0,
                round_ns: 0,
                start_ns: 0,
            }
        })
        .collect()
}

/// Run one chain through the whole schedule with the block-parallel sweep and prepare it for
/// the decode pass, as [`crate::oracle::run_chain`].
pub fn run_chain(
    view: &View<'_>,
    cfg: &ChainConfig,
    layout: Layout,
    progress: &(dyn Fn(&Progress) + Sync),
) -> Chain {
    run_chain_with(view, cfg, layout, progress, &mut |_| {})
}

/// [`run_chain`] with a hook called after every sweep (tests check the counts there).
fn run_chain_with(
    view: &View<'_>,
    cfg: &ChainConfig,
    layout: Layout,
    progress: &(dyn Fn(&Progress) + Sync),
    after_sweep: &mut dyn FnMut(&Chain),
) -> Chain {
    assert!(layout.blocks >= 1 && layout.round >= 1, "invalid layout");
    let started = Instant::now();
    let align_rng = Rng::new(cfg.seed, Stream::Align);
    let fert_rng = Rng::new(cfg.seed, Stream::Fertility);
    let chain = chain_of(view, cfg.sampler);
    let p = cfg.params(view);
    let t_init = timing::start();
    let align = init_alignment(view, &align_rng, chain, p.p_null);
    let lex = lex_counts(view, &align);
    timing::stop(Phase::Init, t_init);
    timing::count(Counter::LexBytesInit, lex.memory_bytes() as u64);
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
    let k_n = view.len();
    let mut blocks = new_blocks(k_n, layout);
    let mut t = 0u32;
    let mut jumps_built = false;
    for m in 1..=3u8 {
        let n = cfg.iters[usize::from(m) - 1];
        if n == 0 {
            continue;
        }
        if m >= 2 && !jumps_built {
            ch.jump = timing::time(Phase::Jumps, || jump_counts(view, &ch.align));
            jumps_built = true;
        }
        let sweep_phase = [Phase::Sweep1, Phase::Sweep2, Phase::Sweep3][usize::from(m) - 1];
        for it in 1..=n {
            cancel::check();
            t += 1;
            if m >= 3 {
                ch.ratio = timing::time(Phase::Fertility, || {
                    let fc = fert_counts(view, &ch.align);
                    fertility_ratios(&fc, view.src.n_types(), &p, &fert_rng, chain, t)
                });
            }
            let t_sweep = timing::start();
            sweep(
                view,
                &mut ch,
                &mut blocks,
                layout.round,
                &p,
                m,
                &align_rng,
                chain,
                t,
                acc_off.as_deref().filter(|_| m == last),
            );
            timing::stop(sweep_phase, t_sweep);
            ch.model = m;
            if m == 1 {
                // the random initialisation's large tables shrink in the first sweeps; give
                // the freed memory back (all chains' initial tables exist at once)
                crate::pipeline::release_free_memory();
            }
            after_sweep(&ch);
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
        ch.ratio = timing::time(Phase::Fertility, || {
            let fc = fert_counts(view, &ch.align);
            fertility_ratios(&fc, view.src.n_types(), &p, &fert_rng, chain, t + 1)
        });
    }
    if timing::enabled() {
        timing::count(Counter::LexBytesFinal, ch.lex.memory_bytes() as u64);
        timing::count(Counter::AlignBytes, (ch.align.capacity() * 2) as u64);
        timing::count(Counter::RatioBytes, (ch.ratio.capacity() * 4) as u64);
        let ov: usize = blocks.iter().map(|b| b.ov.memory_bytes()).sum();
        timing::count(Counter::OverlayBytes, ov as u64);
    }
    ch
}

// ------------------------------------------------------------------------------------------
// Several chains without blocking waits

/// The state of one chain in [`run_chains`]. Only the task that finishes a round (or starts
/// the chain) touches `st` mutably, when none of the chain's block tasks is running; a block
/// task touches only its own `blocks[b]` and its own part of the alignment and accumulator,
/// and reads the chain's counts, which do not change during a round.
struct Run<'a> {
    view: View<'a>,
    cfg: ChainConfig,
    layout: Layout,
    p: Params,
    align_rng: Rng,
    fert_rng: Rng,
    chain: u32,
    last: u8,
    acc_off: Option<Vec<usize>>,
    started: Instant,
    progress: &'a (dyn Fn(&Progress) + Sync),
    /// Block tasks of the current round that have not finished.
    pending: AtomicUsize,
    st: UnsafeCell<RunState>,
    blocks: Vec<UnsafeCell<Block>>,
}

struct RunState {
    ch: Chain,
    /// Model, iteration within it, sweep number t and round start r0 of the current sweep.
    m: u8,
    it: usize,
    t: u32,
    r0: usize,
    longest: usize,
    jumps_built: bool,
    /// The alignment and accumulator buffers of `ch` (fixed while the chain samples).
    align: *mut u16,
    acc: *mut f32,
    sweep_t0: Option<Instant>,
    fork: Option<Instant>,
}

// SAFETY: see `Run`: mutable access to `st` and to `blocks[b]` is exclusive by the round
// protocol (the `pending` counter hands the chain from its block tasks to the one task that
// finishes the round, with acquire/release ordering).
unsafe impl Sync for Run<'_> {}

/// Run the chains `jobs` (view and configuration each) to the end of their schedules, all
/// concurrently on the current rayon pool, and return them in order. Each chain gives exactly
/// what [`run_chain`] gives (same blocks, rounds, merges in block order and random numbers).
/// Instead of one waiting thread per chain, the block task that finishes a round merges it
/// and starts the next one, so no thread waits for a round while work is queued.
pub fn run_chains<'a>(
    jobs: Vec<(View<'a>, ChainConfig)>,
    layout: Layout,
    progress: &'a (dyn Fn(&Progress) + Sync),
) -> Vec<Chain> {
    assert!(layout.blocks >= 1 && layout.round >= 1, "invalid layout");
    let runs: Vec<Run<'a>> = jobs
        .into_iter()
        .map(|(view, cfg)| {
            let p = cfg.params(&view);
            let chain = chain_of(&view, cfg.sampler);
            let last = last_model(&cfg.iters);
            let acc_off = cfg.accumulate.then(|| acc_offsets(&view));
            let blocks = new_blocks(view.len(), layout)
                .into_iter()
                .map(UnsafeCell::new)
                .collect();
            Run {
                align_rng: Rng::new(cfg.seed, Stream::Align),
                fert_rng: Rng::new(cfg.seed, Stream::Fertility),
                view,
                p,
                chain,
                last,
                acc_off,
                layout,
                started: Instant::now(),
                progress,
                pending: AtomicUsize::new(0),
                st: UnsafeCell::new(RunState {
                    ch: Chain {
                        align: Vec::new(),
                        lex: LexCounts::new(1, 0),
                        jump: JumpCounts::default(),
                        ratio: Vec::new(),
                        model: 1,
                        acc: Vec::new(),
                    },
                    m: 0,
                    it: 0,
                    t: 0,
                    r0: 0,
                    longest: 0,
                    jumps_built: false,
                    align: std::ptr::null_mut(),
                    acc: std::ptr::null_mut(),
                    sweep_t0: None,
                    fork: None,
                }),
                blocks,
                cfg,
            }
        })
        .collect();
    rayon::scope(|s| {
        for run in &runs {
            s.spawn(move |s| start_chain(run, s));
        }
    });
    runs.into_iter().map(|r| r.st.into_inner().ch).collect()
}

/// Initialise a chain and start its first sweep.
fn start_chain<'s>(run: &'s Run<'_>, s: &rayon::Scope<'s>) {
    // SAFETY: no block task of this chain exists yet.
    let st = unsafe { &mut *run.st.get() };
    let v = &run.view;
    let t_init = timing::start();
    let align = init_alignment(v, &run.align_rng, run.chain, run.p.p_null);
    let lex = lex_counts(v, &align);
    timing::stop(Phase::Init, t_init);
    timing::count(Counter::LexBytesInit, lex.memory_bytes() as u64);
    st.ch.align = align;
    st.ch.lex = lex;
    if let Some(ao) = &run.acc_off {
        st.ch.acc = vec![0.0; ao[v.len()]];
    }
    st.align = st.ch.align.as_mut_ptr();
    st.acc = st.ch.acc.as_mut_ptr();
    st.m = 1;
    st.it = 0;
    next_sweep(run, st, s);
}

/// Advance to the next sweep of the schedule (or finish the chain) and start it.
fn next_sweep<'s>(run: &'s Run<'_>, st: &mut RunState, s: &rayon::Scope<'s>) {
    loop {
        while st.m <= 3 && st.it >= run.cfg.iters[usize::from(st.m) - 1] {
            st.m += 1;
            st.it = 0;
        }
        if st.m > 3 {
            return finish_chain(run, st);
        }
        cancel::check();
        let v = &run.view;
        let m = st.m;
        if m >= 2 && !st.jumps_built {
            st.ch.jump = timing::time(Phase::Jumps, || jump_counts(v, &st.ch.align));
            st.jumps_built = true;
        }
        st.it += 1;
        st.t += 1;
        if m >= 3 {
            let t = st.t;
            st.ch.ratio = timing::time(Phase::Fertility, || {
                let fc = fert_counts(v, &st.ch.align);
                fertility_ratios(&fc, v.src.n_types(), &run.p, &run.fert_rng, run.chain, t)
            });
        }
        st.sweep_t0 = timing::start();
        let n_source = st.ch.lex.n_source();
        let mut longest = 0;
        for b in &run.blocks {
            // SAFETY: no block task of this chain is running between rounds.
            let blk = unsafe { &mut *b.get() };
            blk.ov.ensure_rows(n_source);
            longest = longest.max(blk.hi - blk.lo);
        }
        st.longest = longest;
        st.r0 = 0;
        if longest > 0 {
            return start_round(run, st, s);
        }
        end_sweep(run, st);
    }
}

/// Spawn the block tasks of the round at `st.r0`.
fn start_round<'s>(run: &'s Run<'_>, st: &mut RunState, s: &rayon::Scope<'s>) {
    cancel::check();
    st.fork = timing::start();
    run.pending.store(run.blocks.len(), Ordering::Release);
    for b in 0..run.blocks.len() {
        s.spawn(move |s| block_task(run, b, s));
    }
}

/// Sample block `b`'s part of the current round; the last block to finish ends the round.
fn block_task<'s>(run: &'s Run<'_>, b: usize, s: &rayon::Scope<'s>) {
    {
        // SAFETY: during a round the chain state is only read (its counts do not change),
        // block b is touched by this task alone, and the alignment / accumulator parts of
        // the blocks are disjoint.
        let st = unsafe { &*run.st.get() };
        let blk = unsafe { &mut *run.blocks[b].get() };
        let v = &run.view;
        let off = v.tgt.offsets();
        let a = unsafe {
            std::slice::from_raw_parts_mut(st.align.add(off[blk.lo]), off[blk.hi] - off[blk.lo])
        };
        let acc_off = run.acc_off.as_deref().filter(|_| st.m == run.last);
        let acc: &mut [f32] = match acc_off {
            Some(ao) => unsafe {
                std::slice::from_raw_parts_mut(st.acc.add(ao[blk.lo]), ao[blk.hi] - ao[blk.lo])
            },
            None => &mut [],
        };
        let sh = Shared {
            view: v,
            lex: &st.ch.lex,
            jump: &st.ch.jump,
            ratio: &st.ch.ratio,
            p: &run.p,
            model: st.m,
            rng: &run.align_rng,
            chain: run.chain,
            t: st.t,
        };
        block_round(&sh, blk, a, acc, acc_off, st.r0, run.layout.round, st.fork);
    }
    if run.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
        // SAFETY: every block task of the round has finished (acquire above).
        let st = unsafe { &mut *run.st.get() };
        end_round(run, st, s);
    }
}

/// Merge the round's overlays in block order, then start the next round or sweep.
fn end_round<'s>(run: &'s Run<'_>, st: &mut RunState, s: &rayon::Scope<'s>) {
    if let Some(f) = st.fork {
        timing::add(Phase::RoundWall, f.elapsed().as_nanos() as u64);
    }
    let t0 = timing::start();
    // SAFETY: no block task of this chain is running.
    let blocks: Vec<&mut Block> = run
        .blocks
        .iter()
        .map(|b| unsafe { &mut *b.get() })
        .collect();
    let mut ns = Vec::new();
    for blk in blocks {
        blk.ov.merge_into(&mut st.ch.lex, &mut st.ch.jump);
        ns.push((blk.round_ns, blk.start_ns));
    }
    if t0.is_some() {
        timing::stop(Phase::Merge, t0);
        timing::add(
            Phase::RoundCritical,
            ns.iter().map(|x| x.0).max().unwrap_or(0),
        );
        timing::add(Phase::BlockWork, ns.iter().map(|x| x.0).sum());
        timing::add(
            Phase::RoundLastStart,
            ns.iter().map(|x| x.1).max().unwrap_or(0),
        );
        timing::count(Counter::Rounds, 1);
    }
    st.r0 += run.layout.round;
    if st.r0 < st.longest {
        return start_round(run, st, s);
    }
    end_sweep(run, st);
    next_sweep(run, st, s);
}

/// Bookkeeping after the last round of a sweep (as [`run_chain`] after each sweep).
fn end_sweep(run: &Run<'_>, st: &mut RunState) {
    let m = st.m;
    timing::stop(
        [Phase::Sweep1, Phase::Sweep2, Phase::Sweep3][usize::from(m) - 1],
        st.sweep_t0.take(),
    );
    st.ch.model = m;
    // the random initialisation's large tables shrink in the first sweeps, and the merges
    // and fertility tables of this chain allocate and free on whichever thread ends a round:
    // give the freed memory back after every sweep
    crate::pipeline::release_free_memory();
    (run.progress)(&Progress {
        direction: run.view.direction,
        sampler: run.cfg.sampler as usize,
        model: m,
        iteration: st.it,
        iterations: run.cfg.iters[usize::from(m) - 1],
        elapsed: run.started.elapsed(),
        null_fraction: null_fraction(&run.view, &st.ch),
    });
}

/// The end of the schedule: the final fertility tables and the measurement counters.
fn finish_chain(run: &Run<'_>, st: &mut RunState) {
    let v = &run.view;
    if st.ch.model >= 3 {
        let t = st.t + 1;
        st.ch.ratio = timing::time(Phase::Fertility, || {
            let fc = fert_counts(v, &st.ch.align);
            fertility_ratios(&fc, v.src.n_types(), &run.p, &run.fert_rng, run.chain, t)
        });
    }
    if timing::enabled() {
        timing::count(Counter::LexBytesFinal, st.ch.lex.memory_bytes() as u64);
        timing::count(Counter::AlignBytes, (st.ch.align.capacity() * 2) as u64);
        timing::count(Counter::RatioBytes, (st.ch.ratio.capacity() * 4) as u64);
        let mut ov = 0;
        for b in &run.blocks {
            // SAFETY: the chain has no block task left.
            let blk = unsafe { &mut *b.get() };
            ov += blk.ov.memory_bytes();
            timing::count(Counter::Tokens, std::mem::take(&mut blk.tokens));
            timing::count(Counter::Moves, std::mem::take(&mut blk.moves));
        }
        timing::count(Counter::OverlayBytes, ov as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitext::{Bitext, TextOptions};
    use crate::io::Direction;
    use crate::model::Hyper;
    use crate::oracle;
    use crate::pipeline::with_threads;

    /// A synthetic corpus of `n` lines with a noisy word-for-word dictionary, empty sides and
    /// a few long lines.
    fn synthetic(n: u32) -> Bitext {
        let mut src = Vec::new();
        let mut tgt = Vec::new();
        for k in 0..n {
            let len = if k % 97 == 0 { 40 } else { 1 + k % 9 };
            if k % 53 != 7 {
                for i in 0..len {
                    src.extend_from_slice(format!("s{} ", (k * 7 + i * 13) % 60).as_bytes());
                }
            }
            if k % 61 != 3 {
                for i in 0..len {
                    tgt.extend_from_slice(format!("t{} ", (k * 7 + i * 13) % 60).as_bytes());
                }
                if k % 3 == 0 {
                    tgt.extend_from_slice(b"extra ");
                }
            }
            src.push(b'\n');
            tgt.push(b'\n');
        }
        Bitext::from_pair_bytes(&src, &tgt, &TextOptions::default()).unwrap()
    }

    fn cfg(iters: [usize; 3], sampler: u32) -> ChainConfig {
        ChainConfig {
            seed: 11,
            hyper: Hyper::default(),
            iters,
            sampler,
            priors: None,
            accumulate: false,
        }
    }

    fn same_chain(a: &Chain, b: &Chain) -> bool {
        a.align == b.align
            && a.lex.same_counts(&b.lex)
            && a.jump == b.jump
            && a.ratio == b.ratio
            && a.model == b.model
            && a.acc == b.acc
    }

    #[test]
    fn one_block_equals_the_oracle() {
        let b = synthetic(700);
        for dir in [Direction::Forward, Direction::Inverse] {
            let v = b.view(dir);
            for iters in [[3, 0, 0], [1, 2, 0], [1, 1, 3], [0, 0, 2]] {
                let want = oracle::run_chain(&v, &cfg(iters, 1), &|_| {});
                for round in [1, 7, 64, 100_000] {
                    let l = Layout { blocks: 1, round };
                    let got = run_chain(&v, &cfg(iters, 1), l, &|_| {});
                    assert!(same_chain(&got, &want), "{dir:?} {iters:?} R={round}");
                }
            }
        }
    }

    #[test]
    fn accumulation_is_exact_and_thread_independent() {
        let b = synthetic(900);
        for dir in [Direction::Forward, Direction::Inverse] {
            let v = b.view(dir);
            for iters in [[3, 0, 0], [1, 2, 0], [1, 1, 3]] {
                let c = ChainConfig {
                    accumulate: true,
                    ..cfg(iters, 2)
                };
                let want = oracle::run_chain(&v, &c, &|_| {});
                let l1 = Layout {
                    blocks: 1,
                    round: 5,
                };
                assert!(same_chain(&run_chain(&v, &c, l1, &|_| {}), &want));
                let l = Layout::default();
                let one = with_threads(1, || run_chain(&v, &c, l, &|_| {}));
                let four = with_threads(4, || run_chain(&v, &c, l, &|_| {}));
                assert!(same_chain(&one, &four), "{dir:?} {iters:?}");
                // every token's sums add up to the number of sweeps of the last model
                let sweeps = iters[usize::from(oracle::last_model(&iters)) - 1] as f32;
                let ao = acc_offsets(&v);
                assert_eq!(one.acc.len(), ao[v.len()]);
                for k in 0..v.len() {
                    let w = v.src.line_len(k) + 1;
                    for tok in one.acc[ao[k]..ao[k + 1]].chunks(w) {
                        let s: f32 = tok.iter().sum();
                        assert!((s - sweeps).abs() < 1e-4 * sweeps, "{s} vs {sweeps}");
                    }
                }
                // accumulation does not change the chain itself
                let plain = run_chain(&v, &cfg(iters, 2), l, &|_| {});
                assert_eq!(plain.align, one.align);
                assert!(plain.acc.is_empty());
            }
        }
    }

    #[test]
    fn many_blocks_are_identical_at_every_thread_count() {
        let b = synthetic(1500);
        let v = b.view(Direction::Forward);
        let l = Layout {
            blocks: 16,
            round: 8,
        };
        let one = with_threads(1, || run_chain(&v, &cfg([2, 2, 2], 0), l, &|_| {}));
        for t in [2, 3, 8] {
            let o = with_threads(t, || run_chain(&v, &cfg([2, 2, 2], 0), l, &|_| {}));
            assert!(same_chain(&o, &one), "{t} threads");
        }
        // and it is a different chain from the exact one
        let exact = oracle::run_chain(&v, &cfg([2, 2, 2], 0), &|_| {});
        assert_ne!(exact.align, one.align);
    }

    #[test]
    fn counts_stay_consistent_after_every_sweep() {
        let b = synthetic(400);
        for dir in [Direction::Forward, Direction::Inverse] {
            let v = b.view(dir);
            for l in [
                Layout {
                    blocks: 5,
                    round: 3,
                },
                Layout {
                    blocks: 16,
                    round: 64,
                },
                Layout {
                    blocks: 1000,
                    round: 1,
                },
            ] {
                let mut sweeps = 0;
                run_chain_with(&v, &cfg([2, 2, 2], 2), l, &|_| {}, &mut |ch| {
                    sweeps += 1;
                    assert!(ch.lex.same_counts(&lex_counts(&v, &ch.align)));
                    if ch.model >= 2 {
                        assert_eq!(ch.jump, jump_counts(&v, &ch.align));
                    }
                    let off = v.tgt.offsets();
                    for k in 0..v.len() {
                        let i_len = v.src.line_len(k);
                        for &a in &ch.align[off[k]..off[k + 1]] {
                            assert!(a == EPS || (a as usize) < i_len);
                        }
                    }
                });
                assert_eq!(sweeps, 6);
            }
        }
    }

    #[test]
    fn parallel_helpers_equal_the_oracle_ones() {
        let b = synthetic(9000);
        let v = b.view(Direction::Inverse);
        let rng = Rng::new(3, Stream::Align);
        let a = init_alignment(&v, &rng, 7, 0.2);
        assert_eq!(a, oracle::init_alignment(&v, &rng, 7, 0.2));
        let fc = fert_counts(&v, &a);
        let frng = Rng::new(3, Stream::Fertility);
        let n = v.src.n_types();
        let p = crate::model::Params::new(&Hyper::default(), v.tgt.n_types() - 1);
        assert_eq!(
            with_threads(4, || fertility_ratios(&fc, n, &p, &frng, 7, 5)),
            oracle::fertility_ratios(&fc, n, &p, &frng, 7, 5)
        );
    }

    #[test]
    fn run_chains_equals_run_chain() {
        let b = synthetic(1200);
        let l = Layout {
            blocks: 16,
            round: 8,
        };
        for (iters, acc) in [([2, 2, 2], false), ([1, 1, 3], true), ([3, 0, 0], false)] {
            let jobs: Vec<(View<'_>, ChainConfig)> = [Direction::Forward, Direction::Inverse]
                .into_iter()
                .flat_map(|d| {
                    (0..3).map(move |s| {
                        let c = ChainConfig {
                            accumulate: acc,
                            ..cfg(iters, s)
                        };
                        (d, c)
                    })
                })
                .map(|(d, c)| (b.view(d), c))
                .collect();
            let want: Vec<Chain> = jobs
                .iter()
                .map(|(v, c)| run_chain(v, c, l, &|_| {}))
                .collect();
            for t in [1, 3, 8] {
                let got = with_threads(t, || run_chains(jobs.clone(), l, &|_| {}));
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    assert!(same_chain(g, w), "{iters:?} {t} threads");
                }
            }
        }
        // an empty corpus and a schedule without sweeps
        let e = Bitext::from_pair_bytes(b"", b"", &TextOptions::default()).unwrap();
        let v = e.view(Direction::Forward);
        let got = run_chains(vec![(v, cfg([2, 1, 1], 0))], l, &|_| {});
        assert!(same_chain(
            &got[0],
            &run_chain(&v, &cfg([2, 1, 1], 0), l, &|_| {})
        ));
        let v = b.view(Direction::Forward);
        let got = run_chains(vec![(v, cfg([0, 0, 0], 0))], l, &|_| {});
        assert!(same_chain(
            &got[0],
            &run_chain(&v, &cfg([0, 0, 0], 0), l, &|_| {})
        ));
    }

    #[test]
    fn block_bounds_cover_the_corpus() {
        for (k, n) in [(0, 16), (5, 16), (1447, 16), (1_130_551, 16), (10, 1)] {
            let mut at = 0;
            for b in 0..n {
                let (lo, hi) = block_bounds(k, n, b);
                assert_eq!(lo, at);
                assert!(hi >= lo);
                at = hi;
            }
            assert_eq!(at, k);
        }
    }

    #[test]
    fn overlay_grows_and_merges() {
        let mut ov = Overlay::with_capacity(16);
        ov.ensure_rows(50);
        let mut lex = LexCounts::new(50, 50);
        let mut jump = JumpCounts::default();
        for e in 1..50u32 {
            for f in 1..50u32 {
                lex.add(e, f, 2);
            }
        }
        jump.add(3, 5);
        for e in 1..50u32 {
            for f in 1..50u32 {
                ov.lex_add(e, f, if (e + f) % 3 == 0 { -1 } else { 1 });
            }
        }
        ov.lex_add(4, 4, -1);
        ov.lex_add(4, 4, 1);
        ov.jump_add(3, -2);
        ov.jump_add(200, 1);
        let c = Overlaid {
            lex: &lex,
            jump: &jump,
            ov: &ov,
            rinv: &[],
            e_old: u32::MAX,
            f: u32::MAX,
            r_old: 0.0,
            jb: [usize::MAX; 2],
            n_jumps: 0,
            jt: &[0.0; JUMP_BUCKETS],
        };
        assert_eq!(c.lex(1, 2), 1);
        assert_eq!(c.lex(1, 1), 3);
        assert_eq!(c.jump(3), 3);
        assert_eq!(c.jump(64), 1);
        assert_eq!(c.jump_total(), 4);
        let tot1 = c.total(1);
        ov.merge_into(&mut lex, &mut jump);
        assert_eq!(lex.get(1, 2), 1);
        assert_eq!(lex.get(4, 4), 3);
        assert_eq!(lex.total(1), tot1);
        assert_eq!(jump.get(3), 3);
        assert_eq!(jump.total(), 4);
        assert!(ov.used.is_empty() && ov.keys.iter().all(|&k| k == NO_KEY));
        assert!(ov.dirty.iter().all(|&w| w == 0));
        assert_eq!(ov.get(key(1, 2)), 0);
    }
}
