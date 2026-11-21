//! The training pipeline: one or both directions of a loaded corpus, with per-iteration
//! progress events, cooperative cancellation (see [`crate::cancel`]) and the results packaged as
//! a model plus Pharaoh alignments.
//!
//! [`train`] runs every chain (direction × sampler) as one task of the current rayon pool, with
//! the block-parallel sampler ([`crate::sampler`]) or, with `exact`, the sequential oracle
//! ([`crate::oracle`]); then the decode pass of each direction. With priors (a
//! priors file or a saved model) their counts are added to the hyperparameters;
//! [`align`] is training on new text with a saved model as priors, [`align_fast`] the
//! deterministic decode of a saved model's HMM.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;

use crate::bitext::{Bitext, TextOptions};
use crate::decode::{decode_parts, join_parts};
use crate::io::{strip_scores, Direction, Model, ModelOptions};
use crate::model::{schedule, Hyper, Params};
use crate::oracle::{chain_of, fert_counts, jump_counts, Chain, ChainConfig};
use crate::priors::{link_counts_except, DirPriors, DirTable, PriorSet};
use crate::rng::{Rng, Stream};
use crate::sampler::Layout;
use crate::symmetrize::{symmetrize_text, Method};

/// Which directions to train.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Directions {
    Both,
    Forward,
    Inverse,
}

impl Directions {
    pub fn list(self) -> Vec<Direction> {
        match self {
            Directions::Both => vec![Direction::Forward, Direction::Inverse],
            Directions::Forward => vec![Direction::Forward],
            Directions::Inverse => vec![Direction::Inverse],
        }
    }
}

/// All options of a training run (the text options are given when the corpus is read).
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub directions: Directions,
    /// Seed of the deterministic random number generator; the output depends only on it, never
    /// on the thread count.
    pub seed: u64,
    /// 1 = IBM1, 2 = + HMM jumps, 3 = + fertility.
    pub model: u8,
    /// Independent samplers per direction.
    pub samplers: usize,
    pub hyper: Hyper,
    /// Scale of the automatic iteration counts (eflomal `--length`).
    pub length: f64,
    /// Explicit iteration counts of Model 1, 2, 3 (each overrides its own stage).
    pub iters: [Option<usize>; 3],
    /// Append ` ||| score` to the directional lines.
    pub scores: bool,
    /// Run each chain with the exact sequential sampler instead of the block-parallel one.
    pub exact: bool,
    /// Model whose conditional gives the scores (default: `model`).
    pub score_model: Option<u8>,
    /// Prior counts added to the hyperparameters (a priors file or a saved model).
    pub priors: Option<Arc<PriorSet>>,
    /// Build the [`Model`] of the run (for saving).
    pub keep_model: bool,
    /// Decode from the conditionals of every sweep of the last model plus the final pass
    /// instead of the final pass alone.
    pub accumulate: bool,
    /// Refuse `accumulate` if its buffers would need more bytes than this.
    pub accumulate_limit: u64,
}

/// Default [`TrainConfig::accumulate_limit`]: 4 GiB.
pub const ACCUMULATE_LIMIT: u64 = 4 << 30;

/// Bytes the `accumulate` buffers of `cfg` need on `bitext`: Σ J · (I + 1) `f32` per chain.
pub fn accumulate_bytes(bitext: &Bitext, cfg: &TrainConfig) -> u64 {
    let n: u64 = (0..bitext.len())
        .map(|k| {
            let (i, j) = (
                bitext.source.line_len(k) as u64,
                bitext.target.line_len(k) as u64,
            );
            if i > 0 && j > 0 {
                let d = cfg.directions.list();
                d.iter()
                    .map(|&d| {
                        if d == Direction::Forward {
                            j * (i + 1)
                        } else {
                            i * (j + 1)
                        }
                    })
                    .sum::<u64>()
            } else {
                0
            }
        })
        .sum();
    4 * n * cfg.samplers as u64
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            directions: Directions::Both,
            seed: 1,
            model: 3,
            samplers: 3,
            hyper: Hyper::default(),
            length: 1.0,
            iters: [None; 3],
            scores: false,
            exact: false,
            score_model: None,
            priors: None,
            keep_model: false,
            accumulate: false,
            accumulate_limit: ACCUMULATE_LIMIT,
        }
    }
}

impl TrainConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=3).contains(&self.model) {
            return Err(format!("model must be 1, 2 or 3, got {}", self.model));
        }
        if !(1..=255).contains(&self.samplers) {
            return Err(format!(
                "samplers must be in 1 ..= 255, got {}",
                self.samplers
            ));
        }
        if let Some(m) = self.score_model {
            if !(1..=3).contains(&m) {
                return Err(format!("score model must be 1, 2 or 3, got {m}"));
            }
        }
        // 0 is eflomal's minimum schedule (2 iterations of the last model, `-l 0`)
        if !(self.length.is_finite() && self.length >= 0.0) {
            return Err(format!("length must be a number >= 0, got {}", self.length));
        }
        self.hyper.validate()
    }
}

/// Result of [`train`]: the Pharaoh text of each trained direction.
#[derive(Clone, Debug, Default)]
pub struct Trained {
    pub fwd: Option<Vec<u8>>,
    pub inv: Option<Vec<u8>>,
    /// Iterations of Model 1, 2, 3 that ran.
    pub iterations: [usize; 3],
    /// The model of the run, with [`TrainConfig::keep_model`].
    pub model: Option<Model>,
}

impl Trained {
    pub fn output(&self, d: Direction) -> Option<&[u8]> {
        match d {
            Direction::Forward => self.fwd.as_deref(),
            Direction::Inverse => self.inv.as_deref(),
        }
    }
}

/// Warnings about a corpus that gives nothing to align, for the front ends (`train` and
/// `align`; printed even with `--quiet`, never part of the outputs or logs). Such a corpus is
/// not an error: training and aligning succeed and write one empty line per input line
/// (none for an empty input).
pub fn corpus_warnings(bitext: &Bitext) -> Vec<String> {
    let n = bitext.len();
    if n == 0 {
        return vec!["the input has no line pairs: nothing was aligned".to_owned()];
    }
    if nothing_to_align(bitext) {
        let what = if n == 1 {
            "the only input line pair does not have".to_owned()
        } else {
            format!("none of the {n} input line pairs has")
        };
        return vec![format!(
            "{what} words on both sides within --max-len: nothing was aligned \
             (every output line is empty)"
        )];
    }
    Vec::new()
}

/// Whether no pair has words on both sides (also true for an empty corpus).
pub fn nothing_to_align(bitext: &Bitext) -> bool {
    let v = bitext.view(Direction::Forward);
    (0..bitext.len()).all(|k| {
        let (s, t) = v.pair(k);
        s.is_empty() || t.is_empty()
    })
}

/// Train the chosen directions on `bitext` and decode them. Chains run concurrently on the
/// current rayon pool; the result depends only on the corpus, the configuration and the seed.
pub fn train(
    bitext: &Bitext,
    cfg: &TrainConfig,
    progress: &(dyn Fn(&Progress) + Sync),
) -> Result<Trained, String> {
    cfg.validate()?;
    if cfg.accumulate {
        let need = accumulate_bytes(bitext, cfg);
        if need > cfg.accumulate_limit {
            return Err(format!(
                "--accumulate needs {:.0} MB for this corpus, more than the limit of {:.0} MB \
                 (raise it with --accumulate-limit)",
                need as f64 / 1e6,
                cfg.accumulate_limit as f64 / 1e6
            ));
        }
    }
    // nothing to align: no sweeps (and so no progress events) - the schedule of an empty
    // corpus would be 1,250/1,250/5,000 sweeps of nothing; the output (empty lines), the
    // warnings and the model are the same
    let iters = if nothing_to_align(bitext) {
        [0; 3]
    } else {
        schedule(cfg.model, bitext.len(), cfg.length, cfg.iters)
    };
    let dirs = cfg.directions.list();
    // the priors of each direction in this corpus's ids
    let tables: Vec<Option<DirTable>> = dirs
        .par_iter()
        .map(|&d| cfg.priors.as_ref().map(|ps| ps.remap(bitext, d)))
        .collect();
    let dir_priors: Vec<Option<Arc<DirPriors>>> = dirs
        .iter()
        .zip(&tables)
        .map(|(&d, t)| {
            let v = bitext.view(d);
            t.as_ref().map(|t| {
                Arc::new(DirPriors::new(
                    t,
                    &cfg.hyper,
                    v.src.n_types(),
                    v.tgt.n_types() - 1,
                ))
            })
        })
        .collect();
    let jobs: Vec<(usize, Direction, u32)> = dirs
        .iter()
        .enumerate()
        .flat_map(|(n, &d)| (0..cfg.samplers as u32).map(move |s| (n, d, s)))
        .collect();
    let t_chains = crate::timing::start();
    let chain_cfg = |n: usize, s: u32| ChainConfig {
        seed: cfg.seed,
        hyper: cfg.hyper,
        iters,
        sampler: s,
        priors: dir_priors[n].clone(),
        accumulate: cfg.accumulate,
    };
    let mut chains: Vec<Chain> = if cfg.exact {
        jobs.par_iter()
            .with_max_len(1)
            .map(|&(n, d, s)| crate::oracle::run_chain(&bitext.view(d), &chain_cfg(n, s), progress))
            .collect()
    } else {
        let runs = jobs
            .iter()
            .map(|&(n, d, s)| (bitext.view(d), chain_cfg(n, s)))
            .collect();
        crate::sampler::run_chains(runs, Layout::default(), progress)
    };
    crate::timing::stop(crate::timing::Phase::Chains, t_chains);
    let link_model = chains.first().map_or(1, |c| c.model);
    let score_model = cfg.score_model.unwrap_or(cfg.model);
    if cfg.scores && score_model != link_model {
        // give every chain the counts the score model needs
        let total = iters.iter().sum::<usize>() as u32;
        let fert_rng = Rng::new(cfg.seed, Stream::Fertility);
        chains
            .par_iter_mut()
            .zip(&jobs)
            .for_each(|(ch, &(n, d, s))| {
                let v = bitext.view(d);
                if score_model >= 2 && ch.model < 2 {
                    ch.jump = jump_counts(&v, &ch.align);
                }
                if score_model >= 3 && ch.ratio.is_empty() {
                    let p =
                        Params::with_priors(&cfg.hyper, v.tgt.n_types() - 1, dir_priors[n].clone());
                    ch.ratio = crate::sampler::fertility_ratios(
                        &fert_counts(&v, &ch.align),
                        v.src.n_types(),
                        &p,
                        &fert_rng,
                        chain_of(&v, s),
                        total + 1,
                    );
                }
            });
    }
    release_free_memory();
    let t_decode = crate::timing::start();
    // each direction owns its chains and frees them as soon as it is decoded, before its
    // text is joined (the end of the decode is the run's memory peak)
    let mut per_dir: Vec<Vec<Chain>> = Vec::with_capacity(dirs.len());
    for _ in &dirs {
        let rest = chains.split_off(cfg.samplers.min(chains.len()));
        per_dir.push(std::mem::replace(&mut chains, rest));
    }
    let texts: Vec<Vec<u8>> = dirs
        .par_iter()
        .zip(per_dir)
        .enumerate()
        .map(|(n, (&d, ch))| {
            let v = bitext.view(d);
            let p = Params::with_priors(&cfg.hyper, v.tgt.n_types() - 1, dir_priors[n].clone())
                .for_corpus(v.tgt.n_tokens(), v.len());
            let parts = decode_parts(&v, &ch, &p, link_model, score_model, cfg.scores);
            drop(ch);
            join_parts(parts)
        })
        .collect();
    crate::timing::stop(crate::timing::Phase::Decode, t_decode);
    let mut out = Trained {
        iterations: iters,
        ..Trained::default()
    };
    let t_model = crate::timing::start();
    if cfg.keep_model {
        let derived: Vec<DirTable> = dirs
            .par_iter()
            .zip(&texts)
            .zip(&tables)
            .map(|((&d, text), prior)| {
                // pairs skipped for length are not observations
                let own =
                    link_counts_except(&bitext.view(d), &strip_scores(text), bitext.skipped())
                        .expect("decoded links fit the corpus");
                match prior {
                    Some(p) => own.plus(p),
                    None => own,
                }
            })
            .collect();
        let mut model = Model {
            options: ModelOptions {
                model: cfg.model,
                samplers: cfg.samplers,
                hyper: cfg.hyper,
                seed: cfg.seed,
                text: TextOptions {
                    chunks: None,
                    ..bitext.options().clone()
                },
            },
            source: bitext.source.vocab.clone(),
            target: bitext.target.vocab.clone(),
            fwd: None,
            inv: None,
            segmentation: bitext.segmentation().copied(),
        };
        for (d, t) in dirs.iter().zip(derived) {
            match d {
                Direction::Forward => model.fwd = Some(t),
                Direction::Inverse => model.inv = Some(t),
            }
        }
        out.model = Some(model);
        crate::timing::stop(crate::timing::Phase::Model, t_model);
    }
    for (d, t) in dirs.into_iter().zip(texts) {
        match d {
            Direction::Forward => out.fwd = Some(t),
            Direction::Inverse => out.inv = Some(t),
        }
    }
    Ok(out)
}

/// What [`align`] may change relative to the saved model's options.
#[derive(Clone, Debug, Default)]
pub struct AlignConfig {
    /// Directions to align (default: the model's).
    pub directions: Option<Directions>,
    pub seed: Option<u64>,
    pub samplers: Option<usize>,
    pub length: Option<f64>,
    pub iters: [Option<usize>; 3],
    pub scores: bool,
    pub exact: bool,
    pub accumulate: bool,
    pub accumulate_limit: Option<u64>,
}

/// The training configuration that aligns new text with `model`: the model's options
/// and its tables as priors; the iteration schedule follows the new corpus's size.
pub fn align_config(model: &Model, a: &AlignConfig) -> Result<TrainConfig, String> {
    let have = model.directions().ok_or("the model has no direction")?;
    let directions = a.directions.unwrap_or(have);
    if have != Directions::Both && directions != have {
        return Err(format!(
            "the model has only the {} direction",
            have.list()[0].name()
        ));
    }
    let o = &model.options;
    Ok(TrainConfig {
        directions,
        seed: a.seed.unwrap_or(o.seed),
        model: o.model,
        samplers: a.samplers.unwrap_or(o.samplers),
        hyper: o.hyper,
        length: a.length.unwrap_or(1.0),
        iters: a.iters,
        scores: a.scores,
        exact: a.exact,
        score_model: None,
        priors: Some(Arc::new(model.prior_set())),
        keep_model: false,
        accumulate: a.accumulate,
        accumulate_limit: a.accumulate_limit.unwrap_or(ACCUMULATE_LIMIT),
    })
}

/// Align new text (read with the model's text options, [`Model::options`]) by sampling with the
/// saved model as priors (eflomal's method).
pub fn align(
    model: &Model,
    bitext: &Bitext,
    a: &AlignConfig,
    progress: &(dyn Fn(&Progress) + Sync),
) -> Result<Trained, String> {
    train(bitext, &align_config(model, a)?, progress)
}

/// The deterministic decode (`align --fast`): per sentence the exact posterior of
/// the Model-2 HMM with the model's own tables as fixed parameters ([`crate::hmm::FastModel`]);
/// no sampling, no randomness, and each pair's links independent of the rest of the call.
pub fn align_fast(
    model: &Model,
    bitext: &Bitext,
    directions: Option<Directions>,
) -> Result<Trained, String> {
    align_fast_with(&crate::hmm::FastModel::default(), model, bitext, directions)
}

/// [`align_fast`] with a [`crate::hmm::FastModel`] that indexes `model` once for many calls
/// (it must only ever be used with this model).
pub fn align_fast_with(
    fast: &crate::hmm::FastModel,
    model: &Model,
    bitext: &Bitext,
    directions: Option<Directions>,
) -> Result<Trained, String> {
    let have = model.directions().ok_or("the model has no direction")?;
    let directions = directions.unwrap_or(have);
    if have != Directions::Both && directions != have {
        return Err(format!(
            "the model has only the {} direction",
            have.list()[0].name()
        ));
    }
    let dirs = directions.list();
    let texts: Vec<Vec<u8>> = dirs
        .iter()
        .map(|&d| {
            fast.decode(model, bitext, d)
                .expect("the model has the direction")
        })
        .collect();
    let mut out = Trained::default();
    for (d, t) in dirs.into_iter().zip(texts) {
        match d {
            Direction::Forward => out.fwd = Some(t),
            Direction::Inverse => out.inv = Some(t),
        }
    }
    Ok(out)
}

/// One progress event: an iteration of one model of one direction has finished.
#[derive(Clone, Debug)]
pub struct Progress {
    pub direction: Direction,
    /// 0-based sampler index.
    pub sampler: usize,
    /// Model 1, 2 or 3.
    pub model: u8,
    /// 1-based, within this model.
    pub iteration: usize,
    pub iterations: usize,
    /// Time since this direction started training.
    pub elapsed: Duration,
    /// Fraction of target tokens aligned to NULL after this iteration.
    pub null_fraction: f64,
}

impl fmt::Display for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} sampler {}] model {} iteration {}/{}: NULL {:.1} %, {:.1} s",
            self.direction.name(),
            self.sampler + 1,
            self.model,
            self.iteration,
            self.iterations,
            100.0 * self.null_fraction,
            self.elapsed.as_secs_f64()
        )
    }
}

/// Symmetrise two directional Pharaoh texts, ignoring ` ||| score` suffixes.
pub fn symmetrize_pharaoh(fwd: &[u8], inv: &[u8], method: Method) -> Result<Vec<u8>, String> {
    let (f, i) = (crate::io::strip_scores(fwd), crate::io::strip_scores(inv));
    let mut out = Vec::with_capacity(f.len());
    symmetrize_text(&f, Some(&i), method, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Return free heap memory to the operating system (glibc keeps freed memory of every thread's
/// arena otherwise). Allocation only: no value changes.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: glibc's malloc_trim only releases free memory; it is thread-safe.
        unsafe {
            malloc_trim(0);
        }
    }
}

/// Run `f` on a rayon pool of `threads` threads (0 = all cores: the global pool, or inside a
/// [`crate::cancel::Scope`] a new pool of that size whose workers see the call's token).
pub fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    let token = crate::cancel::current();
    if threads == 0 && token.is_none() {
        return f();
    }
    // a call with a cancellation scope gets its own pool (0 = rayon's default size), whose
    // workers all see the call's token (crate::cancel)
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |_| {
            crate::cancel::set_current(token.clone());
        })
        .build();
    match built {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetrize_ignores_scores() {
        let out = symmetrize_pharaoh(
            b"0-0 1-1 ||| -1.5\n0-0\n",
            b"0-0\n0-0 0-1\n",
            Method::Intersect,
        )
        .unwrap();
        assert_eq!(out, b"0-0\n0-0\n");
    }

    #[test]
    fn with_threads_sets_the_pool_size() {
        assert_eq!(with_threads(3, rayon::current_num_threads), 3);
    }

    #[test]
    fn train_is_identical_at_every_thread_count() {
        use crate::bitext::TextOptions;
        // a few thousand lines so that the decode pass has several chunks
        let mut src = Vec::new();
        let mut tgt = Vec::new();
        for k in 0..5000u32 {
            let n = 1 + k % 7;
            for i in 0..n {
                src.extend_from_slice(format!("s{} ", (k * 7 + i * 13) % 50).as_bytes());
                tgt.extend_from_slice(format!("t{} ", (k * 7 + i * 13) % 50).as_bytes());
            }
            if k % 3 == 0 {
                tgt.extend_from_slice(b"extra");
            }
            src.push(b'\n');
            tgt.push(b'\n');
        }
        let b = Bitext::from_pair_bytes(&src, &tgt, &TextOptions::default()).unwrap();
        for exact in [false, true] {
            let cfg = TrainConfig {
                iters: [Some(2), Some(2), Some(2)],
                scores: true,
                exact,
                ..TrainConfig::default()
            };
            let run = |t| with_threads(t, || train(&b, &cfg, &|_| {}).unwrap());
            let one = run(1);
            assert_eq!(one.iterations, [2, 2, 2]);
            for t in [2, 3, 8] {
                let o = run(t);
                assert_eq!(o.fwd, one.fwd, "fwd at {t} threads, exact {exact}");
                assert_eq!(o.inv, one.inv, "inv at {t} threads, exact {exact}");
            }
        }
        let one = with_threads(1, || {
            train(
                &b,
                &TrainConfig {
                    iters: [Some(2), Some(2), Some(2)],
                    ..TrainConfig::default()
                },
                &|_| {},
            )
            .unwrap()
        });
        // the trivial dictionary is learnt: most fwd links of line 1 are on the diagonal
        let fwd = one.fwd.unwrap();
        let line1 = fwd.split(|&c| c == b'\n').nth(1).unwrap();
        assert!(
            line1.starts_with(b"0-0 1-1"),
            "{}",
            String::from_utf8_lossy(line1)
        );
    }

    fn synthetic(lines: u32) -> Bitext {
        use crate::bitext::TextOptions;
        let mut src = Vec::new();
        let mut tgt = Vec::new();
        for k in 0..lines {
            let n = 1 + k % 6;
            for i in 0..n {
                src.extend_from_slice(format!("s{} ", (k * 7 + i * 13) % 40).as_bytes());
                tgt.extend_from_slice(format!("t{} ", (k * 7 + i * 13) % 40).as_bytes());
            }
            if k % 5 == 0 {
                tgt.extend_from_slice(b"extra");
            }
            src.push(b'\n');
            tgt.push(b'\n');
        }
        Bitext::from_pair_bytes(&src, &tgt, &TextOptions::default()).unwrap()
    }

    fn small_cfg() -> TrainConfig {
        TrainConfig {
            iters: [Some(2), Some(2), Some(3)],
            scores: true,
            ..TrainConfig::default()
        }
    }

    #[test]
    fn neutral_priors_change_nothing() {
        let b = synthetic(600);
        let plain = with_threads(2, || train(&b, &small_cfg(), &|_| {}).unwrap());
        // records of words the corpus does not have, and zero counts
        let ps = PriorSet::parse(
            b"LEX\tnope\tt1\t5\nLEX\ts1\tnope\t5\nLEX\ts1\tt1\t0\n\
                                   FERF\tnope\t1\t3\nFERR\tnope\t2\t3\n",
        )
        .unwrap();
        for exact in [false, true] {
            let cfg = TrainConfig {
                priors: Some(Arc::new(ps.clone())),
                exact,
                ..small_cfg()
            };
            let with = with_threads(2, || train(&b, &cfg, &|_| {}).unwrap());
            let base = if exact {
                with_threads(2, || {
                    train(
                        &b,
                        &TrainConfig {
                            exact,
                            ..small_cfg()
                        },
                        &|_| {},
                    )
                    .unwrap()
                })
            } else {
                plain.clone()
            };
            assert_eq!(with.fwd, base.fwd, "exact {exact}");
            assert_eq!(with.inv, base.inv, "exact {exact}");
        }
    }

    #[test]
    fn a_strong_prior_decides_the_link() {
        let b = Bitext::from_pair_bytes(b"a b\nb a\n", b"x y\ny x\n", &Default::default()).unwrap();
        let ps = PriorSet::parse(b"LEX\ta\ty\t1000\nLEX\tb\tx\t1000\n").unwrap();
        let cfg = TrainConfig {
            priors: Some(Arc::new(ps)),
            ..TrainConfig::default()
        };
        // without priors every pairing is equally likely; with them only the prior's pairs a–y,
        // b–x are linked (on two lines a word may stay on NULL)
        let out = train(&b, &cfg, &|_| {}).unwrap();
        for text in [out.fwd.unwrap(), out.inv.unwrap()] {
            let text = String::from_utf8(text).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(lines.len(), 2);
            for l in lines {
                assert!(!l.is_empty(), "{text}");
                for link in l.split(' ') {
                    assert!(link == "0-1" || link == "1-0", "{text}");
                }
            }
        }
    }

    /// End to end: the accepted priors `HMMF -64 1e25` and `LEX <NULL> y 1e25`
    /// gave `0-0 ||| inf` (all weights 0 in `f32`). Every sampling mode and model now gives
    /// finite scores, in both word orders.
    #[test]
    fn extreme_priors_give_finite_scores() {
        let ps = PriorSet::parse(b"HMMF\t-64\t1e25\nLEX\t<NULL>\ty\t1e25\n").unwrap();
        for src in [&b"a b\na b\n"[..], b"b a\nb a\n"] {
            let b = Bitext::from_pair_bytes(src, b"x\ny\n", &Default::default()).unwrap();
            for model in [2u8, 3] {
                for (exact, accumulate) in [(false, false), (true, false), (false, true)] {
                    let cfg = TrainConfig {
                        directions: Directions::Forward,
                        model,
                        samplers: 1,
                        iters: [Some(1), Some(1), Some(1)],
                        scores: true,
                        exact,
                        accumulate,
                        priors: Some(Arc::new(ps.clone())),
                        ..TrainConfig::default()
                    };
                    let out = with_threads(1, || train(&b, &cfg, &|_| {}).unwrap());
                    let text = String::from_utf8(out.fwd.unwrap()).unwrap();
                    let ctx = format!("model {model} exact {exact} acc {accumulate}: {text}");
                    assert_eq!(text.lines().count(), 2, "{ctx}");
                    for line in text.lines() {
                        let (links, score) = line.split_once(" ||| ").expect("a score");
                        let score: f64 = score.parse().unwrap();
                        assert!(score.is_finite() && score > 0.0, "{ctx}");
                        assert!(
                            links.split(' ').all(|l| l.is_empty() || l.len() == 3),
                            "{ctx}"
                        );
                    }
                    // the source words outweigh NULL (whose row has a 1e25 prior on y alone)
                    assert!(!text.lines().next().unwrap().starts_with(" |||"), "{ctx}");
                }
            }
        }
    }

    /// The lines of a fast decode of `pairs` (source, target) in one call.
    fn fast_lines(model: &Model, pairs: &[(&str, &str)], d: Directions) -> Vec<String> {
        let src: String = pairs.iter().map(|p| format!("{}\n", p.0)).collect();
        let tgt: String = pairs.iter().map(|p| format!("{}\n", p.1)).collect();
        let b =
            Bitext::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), &model.options.text).unwrap();
        let out = align_fast(model, &b, Some(d)).unwrap();
        let text = out.fwd.or(out.inv).unwrap();
        String::from_utf8(text)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// Fast decoding uses the trained lexical distribution, and a pair's links
    /// do not depend on the other pairs of the call (batch size, order, duplicates, words).
    #[test]
    fn fast_decoding_is_independent_of_the_batch() {
        let src = "cat\ndog\n".repeat(100);
        let tgt = "chat\nchien\n".repeat(100);
        let b = Bitext::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), &TextOptions::default())
            .unwrap();
        let cfg = TrainConfig {
            iters: [Some(2), Some(2), Some(2)],
            keep_model: true,
            ..TrainConfig::default()
        };
        let m = with_threads(1, || train(&b, &cfg, &|_| {}).unwrap().model.unwrap());
        for d in [Directions::Forward, Directions::Inverse] {
            // the dictionary: chien is dog's (position 0), chat is cat's (position 1)
            let one = |s: &str, t: &str| fast_lines(&m, &[(s, t)], d)[0].clone();
            assert_eq!(one("dog cat", "chien"), "0-0", "{d:?}");
            assert_eq!(one("dog cat", "chat"), "1-0", "{d:?}");
            // unknown words on either side do not take the known word's link
            assert_eq!(one("dog zebra", "chien"), "0-0", "{d:?}");
            assert_eq!(one("dog", "chien zèbre"), "0-0", "{d:?}");
        }
        // any split, order or duplication of a set of pairs gives each pair the same links
        let words = ["cat", "dog", "zebra", "cat", "dog"];
        let pairs: Vec<(String, String)> = (0..447)
            .map(|k: usize| {
                let n = 1 + k % 4;
                let s: Vec<&str> = (0..n).map(|i| words[(k * 3 + i * 7) % 5]).collect();
                let t: Vec<&str> = s
                    .iter()
                    .rev()
                    .map(|w| match *w {
                        "cat" => "chat",
                        "dog" => "chien",
                        _ => "zèbre",
                    })
                    .chain((k % 3 == 0).then_some("lion"))
                    .collect();
                (s.join(" "), t.join(" "))
            })
            .collect();
        let refs: Vec<(&str, &str)> = pairs.iter().map(|(s, t)| (&s[..], &t[..])).collect();
        for d in [Directions::Forward, Directions::Inverse] {
            let all = fast_lines(&m, &refs, d);
            for size in [1, 32, 447] {
                let split: Vec<String> = refs
                    .chunks(size)
                    .flat_map(|c| fast_lines(&m, c, d))
                    .collect();
                assert_eq!(split, all, "{d:?} chunks of {size}");
            }
            let mut rev = refs.clone();
            rev.reverse();
            let mut back = fast_lines(&m, &rev, d);
            back.reverse();
            assert_eq!(back, all, "{d:?} reversed");
            let dup: Vec<(&str, &str)> = refs.iter().flat_map(|&p| [p, p]).collect();
            let twice = fast_lines(&m, &dup, d);
            assert!(twice
                .chunks(2)
                .zip(&all)
                .all(|(x, y)| x[0] == *y && x[1] == *y));
            // a both-direction call equals the single-direction ones
            let both = {
                let src: String = refs.iter().map(|p| format!("{}\n", p.0)).collect();
                let tgt: String = refs.iter().map(|p| format!("{}\n", p.1)).collect();
                let b = Bitext::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), &m.options.text)
                    .unwrap();
                align_fast(&m, &b, None).unwrap()
            };
            let text = match d {
                Directions::Forward => both.fwd.unwrap(),
                _ => both.inv.unwrap(),
            };
            assert_eq!(
                String::from_utf8(text).unwrap().lines().collect::<Vec<_>>(),
                all
            );
        }
    }

    type WordTable = (
        Vec<(Vec<u8>, Vec<u8>, f64)>,
        Vec<(i32, f64)>,
        Vec<(Vec<u8>, u32, f64)>,
    );

    /// A direction's table keyed by words, so models with different vocabularies compare.
    fn word_table(m: &Model, d: Direction) -> WordTable {
        let (t, ev, fv) = match d {
            Direction::Forward => (m.fwd.as_ref().unwrap(), &m.source, &m.target),
            Direction::Inverse => (m.inv.as_ref().unwrap(), &m.target, &m.source),
        };
        let mut lex: Vec<_> = t
            .lex
            .iter()
            .map(|&(e, f, c)| (ev.word(e).to_vec(), fv.word(f).to_vec(), c))
            .collect();
        lex.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        let mut fert: Vec<_> = t
            .fert
            .iter()
            .map(|&(e, n, c)| (ev.word(e).to_vec(), n, c))
            .collect();
        fert.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        (lex, t.jump.iter().map(|(&m, &c)| (m, c)).collect(), fert)
    }

    /// Pairs skipped for length add nothing to the saved model's counts (they
    /// used to add one +1 jump each); a genuine empty pair keeps makepriors' measured behaviour.
    #[test]
    fn skipped_pairs_add_no_counts() {
        let opts = TextOptions {
            max_len: 1,
            ..TextOptions::default()
        };
        let cfg = TrainConfig {
            model: 1,
            samplers: 1,
            iters: [Some(0), Some(0), Some(0)],
            hyper: Hyper {
                null_prior: 1e-6,
                ..Hyper::default()
            },
            keep_model: true,
            ..TrainConfig::default()
        };
        // a -> x, a genuine empty pair, and 100 pairs `a b -> y` too long for max_len 1
        let (mut s, mut t) = (
            b"a

"
            .to_vec(),
            b"x

"
            .to_vec(),
        );
        let (mut s2, mut t2) = (s.clone(), t.clone());
        for k in 0..100 {
            s2.extend_from_slice(
                b"a b
",
            );
            t2.extend_from_slice(
                b"y
",
            );
            if k == 50 {
                s2.extend_from_slice(
                    b"a
",
                );
                t2.extend_from_slice(
                    b"x
",
                );
                s.extend_from_slice(
                    b"a
",
                );
                t.extend_from_slice(
                    b"x
",
                );
            }
        }
        let plain = Bitext::from_pair_bytes(&s, &t, &opts).unwrap();
        let with = Bitext::from_pair_bytes(&s2, &t2, &opts).unwrap();
        assert_eq!(with.skipped().len(), 100);
        let m0 = train(&plain, &cfg, &|_| {}).unwrap().model.unwrap();
        let m1 = train(&with, &cfg, &|_| {}).unwrap().model.unwrap();
        for d in [Direction::Forward, Direction::Inverse] {
            let (lex, jump, fert) = word_table(&m1, d);
            assert_eq!(
                (lex.clone(), jump.clone(), fert.clone()),
                word_table(&m0, d),
                "{d:?}"
            );
            // two links a-x, each one jump +1 in and +1 out; the empty pair adds one +1 jump
            assert_eq!(jump, [(1, 5.0)], "{d:?}");
            assert_eq!(lex.len(), 1, "{d:?}");
        }
    }

    #[test]
    fn save_load_align() {
        let b = synthetic(800);
        let cfg = TrainConfig {
            keep_model: true,
            ..small_cfg()
        };
        let trained = with_threads(3, || train(&b, &cfg, &|_| {}).unwrap());
        let model = trained.model.clone().unwrap();
        // the model's tables are the link counts of the training output
        let fwd_links = strip_scores(trained.fwd.as_ref().unwrap());
        assert_eq!(
            model.fwd.as_ref().unwrap(),
            &crate::priors::link_counts(&b.view(Direction::Forward), &fwd_links).unwrap()
        );
        let mut bytes = Vec::new();
        model.write(&mut bytes).unwrap();
        let loaded = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(loaded, model);
        // new text: a slice of the corpus plus unknown words
        let new = Bitext::from_pair_bytes(
            b"s1 s14 s27\nunknown s3\n\ns5\n",
            b"t1 t14 t27\nt3 never\nt9\n\n",
            &model.options.text,
        )
        .unwrap();
        let a = AlignConfig {
            iters: [Some(2), Some(2), Some(4)],
            ..AlignConfig::default()
        };
        let one = with_threads(1, || align(&loaded, &new, &a, &|_| {}).unwrap());
        for t in [2, 5] {
            let o = with_threads(t, || align(&loaded, &new, &a, &|_| {}).unwrap());
            assert_eq!(o.fwd, one.fwd, "{t} threads");
            assert_eq!(o.inv, one.inv, "{t} threads");
        }
        // a single direction equals that direction of a two-direction run
        let f = align(
            &loaded,
            &new,
            &AlignConfig {
                directions: Some(Directions::Forward),
                ..a.clone()
            },
            &|_| {},
        )
        .unwrap();
        assert_eq!(f.fwd, one.fwd);
        assert!(f.inv.is_none());
        let fwd = String::from_utf8(one.fwd.unwrap()).unwrap();
        let lines: Vec<&str> = fwd.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "0-0 1-1 2-2", "{fwd}");
        assert_eq!(lines[2], "");
        assert_eq!(lines[3], "");
        // the deterministic decode: thread-independent, one line per pair
        let fast = with_threads(1, || align_fast(&loaded, &new, None).unwrap());
        let fast4 = with_threads(4, || align_fast(&loaded, &new, None).unwrap());
        assert_eq!(fast.fwd, fast4.fwd);
        assert_eq!(fast.inv, fast4.inv);
        let ff = String::from_utf8(fast.fwd.unwrap()).unwrap();
        assert_eq!(ff.lines().next(), Some("0-0 1-1 2-2"), "{ff}");
        assert_eq!(ff.lines().count(), 4);
        // a model of one direction aligns only that one
        let only = Model {
            inv: None,
            ..loaded.clone()
        };
        assert!(align(
            &only,
            &new,
            &AlignConfig {
                directions: Some(Directions::Both),
                ..a.clone()
            },
            &|_| {}
        )
        .is_err());
        assert!(align(&only, &new, &a, &|_| {}).unwrap().inv.is_none());
    }

    #[test]
    fn accumulate_decodes_and_respects_its_limit() {
        let b = synthetic(700);
        let cfg = TrainConfig {
            accumulate: true,
            ..small_cfg()
        };
        let one = with_threads(1, || train(&b, &cfg, &|_| {}).unwrap());
        let many = with_threads(4, || train(&b, &cfg, &|_| {}).unwrap());
        assert_eq!(one.fwd, many.fwd);
        assert_eq!(one.inv, many.inv);
        // on this easy corpus both decodes agree; on an ambiguous one they do not always
        let plain = train(&b, &small_cfg(), &|_| {}).unwrap();
        assert_eq!(
            strip_scores(one.fwd.as_ref().unwrap()),
            strip_scores(plain.fwd.as_ref().unwrap())
        );
        let amb = Bitext::from_pair_bytes(
            b"a b c\nb a\nc a b\n",
            b"x y z\ny x\nz y x\n",
            &Default::default(),
        )
        .unwrap();
        let differs = (1..=20).any(|seed| {
            let c = TrainConfig {
                seed,
                ..small_cfg()
            };
            let p = train(&amb, &c, &|_| {}).unwrap();
            let a = train(
                &amb,
                &TrainConfig {
                    accumulate: true,
                    ..c
                },
                &|_| {},
            )
            .unwrap();
            strip_scores(p.fwd.as_ref().unwrap()) != strip_scores(a.fwd.as_ref().unwrap())
        });
        assert!(differs);
        let need = accumulate_bytes(&b, &cfg);
        assert!(need > 0);
        let tight = TrainConfig {
            accumulate_limit: need - 1,
            ..cfg.clone()
        };
        assert!(train(&b, &tight, &|_| {})
            .unwrap_err()
            .contains("--accumulate"));
        let exact = TrainConfig {
            accumulate_limit: need,
            ..cfg
        };
        assert!(train(&b, &exact, &|_| {}).is_ok());
    }

    #[test]
    fn score_model_changes_only_scores() {
        let b = synthetic(300);
        let links = |t: &Option<Vec<u8>>| strip_scores(t.as_ref().unwrap());
        for model in 1..=3u8 {
            let base = TrainConfig {
                model,
                ..small_cfg()
            };
            let a = train(&b, &base, &|_| {}).unwrap();
            for sm in 1..=3u8 {
                let cfg = TrainConfig {
                    score_model: Some(sm),
                    ..base.clone()
                };
                let o = train(&b, &cfg, &|_| {}).unwrap();
                assert_eq!(
                    links(&o.fwd),
                    links(&a.fwd),
                    "model {model} score model {sm}"
                );
                assert_eq!(links(&o.inv), links(&a.inv));
                assert_eq!(
                    o.fwd == a.fwd,
                    sm == model,
                    "model {model} score model {sm}"
                );
            }
        }
    }

    #[test]
    fn invalid_config_is_rejected() {
        let b = Bitext::from_pair_bytes(b"a\n", b"b\n", &Default::default()).unwrap();
        for cfg in [
            TrainConfig {
                model: 4,
                ..TrainConfig::default()
            },
            TrainConfig {
                samplers: 0,
                ..TrainConfig::default()
            },
            TrainConfig {
                length: -0.5,
                ..TrainConfig::default()
            },
            TrainConfig {
                length: f64::NAN,
                ..TrainConfig::default()
            },
        ] {
            assert!(train(&b, &cfg, &|_| {}).is_err());
        }
    }

    #[test]
    fn directions_list() {
        assert_eq!(Directions::Both.list().len(), 2);
        assert_eq!(Directions::Inverse.list(), [Direction::Inverse]);
    }
}
