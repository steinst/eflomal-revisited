//! Optional per-phase timing for performance measurement. Off unless the environment variable
//! `EFLOMALR_TIMING` is set to a non-empty value other than `0`; it never changes any output
//! except the report the native binary then prints to stderr.
//!
//! Phases inside a chain (`init`, `sweep m1..m3`, `merge`, `fertility`, `jumps`) are summed over
//! all chains: they are **chain-seconds**, not wall time, because the chains of a run overlap.
//! `round critical` is the sum over rounds of the slowest block's time and `block work` the sum
//! of all blocks' times; at one thread `block work` is the sampling time. Phases outside the
//! chains (`read`, `chains`, `decode`, `model`, `symmetrize`, `write`) are wall time of the
//! calling thread.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Reading and building the corpus (wall).
    Read,
    /// All chains, from the first start to the last finish (wall).
    Chains,
    /// Random initial alignment and the first lexical counts (per chain).
    Init,
    /// Building the jump counts before the first Model-2/3 sweep (per chain).
    Jumps,
    /// Fertility counts and the gamma draws of the ratio table (per chain).
    Fertility,
    /// Whole sweeps of Model 1, 2, 3, merges included (per chain).
    Sweep1,
    Sweep2,
    Sweep3,
    /// The serial merges of the block overlays (per chain; part of the sweeps).
    Merge,
    /// Sum over rounds of the slowest block (per chain; part of the sweeps).
    RoundCritical,
    /// Sum of every block's sampling time (per chain; part of the sweeps).
    BlockWork,
    /// Sum over rounds of the time from the fork of the round's blocks to their join (per
    /// chain): `round critical` plus the time the blocks waited to start.
    RoundWall,
    /// Sum over rounds of the time from the fork to the start of the last block to start.
    RoundLastStart,
    /// Decoding both directions (wall).
    Decode,
    /// Deriving the model's tables for `--save` (wall).
    Model,
    /// Symmetrisation (wall).
    Symmetrize,
    /// Writing the output files and the model (wall).
    Write,
}

const PHASES: [(Phase, &str); 17] = [
    (Phase::Read, "read"),
    (Phase::Chains, "chains (wall)"),
    (Phase::Init, "init"),
    (Phase::Jumps, "jumps"),
    (Phase::Fertility, "fertility"),
    (Phase::Sweep1, "sweep m1"),
    (Phase::Sweep2, "sweep m2"),
    (Phase::Sweep3, "sweep m3"),
    (Phase::Merge, "  merge"),
    (Phase::RoundCritical, "  round critical"),
    (Phase::BlockWork, "  block work"),
    (Phase::RoundWall, "  round wall"),
    (Phase::RoundLastStart, "  round last start"),
    (Phase::Decode, "decode"),
    (Phase::Model, "model"),
    (Phase::Symmetrize, "symmetrize"),
    (Phase::Write, "write"),
];

/// Counters reported with the phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    /// Target tokens sampled (all chains, all sweeps).
    Tokens,
    /// Sampled tokens whose alignment changed.
    Moves,
    /// Rounds (all chains).
    Rounds,
    /// Bytes of the lexical count tables after the initialisation (all chains).
    LexBytesInit,
    /// Bytes of the lexical count tables at the end of the chains (all chains).
    LexBytesFinal,
    /// Bytes of the alignment state (all chains).
    AlignBytes,
    /// Bytes of the fertility ratio tables (all chains).
    RatioBytes,
    /// Bytes of the block overlays at the end of the chains (all chains).
    OverlayBytes,
}

const COUNTERS: [(Counter, &str); 8] = [
    (Counter::Tokens, "tokens sampled"),
    (Counter::Moves, "alignment changes"),
    (Counter::Rounds, "rounds"),
    (Counter::LexBytesInit, "lex bytes init"),
    (Counter::LexBytesFinal, "lex bytes final"),
    (Counter::AlignBytes, "align bytes"),
    (Counter::RatioBytes, "ratio bytes"),
    (Counter::OverlayBytes, "overlay bytes"),
];

const N: usize = PHASES.len();
static NANOS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static COUNTS: [AtomicU64; COUNTERS.len()] = [const { AtomicU64::new(0) }; COUNTERS.len()];

/// Whether timing is on (read once from `EFLOMALR_TIMING`).
#[inline]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("EFLOMALR_TIMING").is_ok_and(|v| !v.is_empty() && v != "0"))
}

fn index(p: Phase) -> usize {
    PHASES.iter().position(|(q, _)| *q == p).expect("listed")
}

/// Add `nanos` to phase `p`.
#[inline]
pub fn add(p: Phase, nanos: u64) {
    if enabled() {
        let i = index(p);
        NANOS[i].fetch_add(nanos, Ordering::Relaxed);
        CALLS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Add `n` to counter `c`.
#[inline]
pub fn count(c: Counter, n: u64) {
    if enabled() {
        let i = COUNTERS.iter().position(|(q, _)| *q == c).expect("listed");
        COUNTS[i].fetch_add(n, Ordering::Relaxed);
    }
}

/// A start time if timing is on.
#[inline]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Add the time since `t0` (from [`start`]) to phase `p`.
#[inline]
pub fn stop(p: Phase, t0: Option<Instant>) {
    if let Some(t0) = t0 {
        add(p, t0.elapsed().as_nanos() as u64);
    }
}

/// Run `f` and add its wall time to phase `p`.
#[inline]
pub fn time<T>(p: Phase, f: impl FnOnce() -> T) -> T {
    let t0 = start();
    let out = f();
    stop(p, t0);
    out
}

/// The phase table (seconds and number of timed spans) and the counters, one per line.
pub fn report() -> String {
    let mut s = String::from("timing (phase, seconds, spans):\n");
    for (i, (_, name)) in PHASES.iter().enumerate() {
        let ns = NANOS[i].load(Ordering::Relaxed);
        let calls = CALLS[i].load(Ordering::Relaxed);
        if calls > 0 {
            s += &format!("  {name:<18} {:>10.3} {calls:>9}\n", ns as f64 / 1e9);
        }
    }
    for (i, (_, name)) in COUNTERS.iter().enumerate() {
        s += &format!("  {name:<18} {:>10}\n", COUNTS[i].load(Ordering::Relaxed));
    }
    s
}
