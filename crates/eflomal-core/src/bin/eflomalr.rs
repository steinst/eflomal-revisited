//! `eflomalr`: native command line for training eflomal's models, saving them, aligning new
//! text with a saved model, building and using eflomal priors files, and symmetrisation.
//!
//! eflomal's options keep their long names; eflomal's short flags are accepted where they do
//! not clash with the common ones (`-s`, `-t`, `-o`, `-q`): see `eflomalr train --help`.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use eflomal_core::bitext::{Bitext, Parsed, TextOptions, DEFAULT_MAX_LEN, MAX_MAX_LEN, NO_MAX_LEN};
use eflomal_core::io::{Direction, Model};
use eflomal_core::model::Hyper;
use eflomal_core::pipeline::{
    align, align_fast, corpus_warnings, symmetrize_pharaoh, train, with_threads, AlignConfig,
    Directions, Progress, TrainConfig, Trained,
};
use eflomal_core::priors::{make_priors, PriorSet};
use eflomal_core::segment::{
    align_check, choice_line, estimate, estimate_model, Estimate, Segment,
};
use eflomal_core::symmetrize::{symmetrize_text, Method};
use eflomal_core::timing::{self, Phase};

#[derive(Parser)]
#[command(
    name = "eflomalr",
    version = eflomal_core::version(),
    about = "Bayesian word alignment by collapsed Gibbs sampling (eflomal's models)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train on a parallel corpus and write alignments (eflomal-align).
    Train(TrainArgs),
    /// Align new sentence pairs with a saved model (sampling with the model as priors).
    Align(AlignArgs),
    /// Symmetrise two directional alignment files (as `atools -c METHOD`).
    Symmetrize(SymArgs),
    /// Build an eflomal priors file from an aligned corpus (eflomal-makepriors).
    Makepriors(PriorsArgs),
    /// Report a corpus's statistics, the segmentation `train` would choose and why, the
    /// measured evidence, and the sampler parameters (changes nothing).
    Estimate(EstimateArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum SegmentArg {
    Auto,
    Words,
}

impl SegmentArg {
    fn segment(self) -> Segment {
        match self {
            SegmentArg::Auto => Segment::Auto,
            SegmentArg::Words => Segment::Words,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Dirs {
    Both,
    Fwd,
    Inv,
}

/// One direction (`align --direction`; two directions are `-s METHOD`, as in the Python CLI).
#[derive(Clone, Copy, ValueEnum)]
enum OneDir {
    Fwd,
    Inv,
}

impl Dirs {
    fn directions(self) -> Directions {
        match self {
            Dirs::Both => Directions::Both,
            Dirs::Fwd => Directions::Forward,
            Dirs::Inv => Directions::Inverse,
        }
    }
}

/// How tokens are normalised before alignment (also applied to the words of a priors file).
#[derive(Args, Clone)]
struct TextArgs {
    /// Keep the case of tokens (eflomal always lowercases).
    #[arg(long)]
    no_lowercase: bool,
    /// Stem source tokens to their first N characters (eflomal `--source-prefix`; 0 = do not
    /// stem, as eflomal).
    #[arg(long, value_name = "N")]
    source_prefix: Option<usize>,
    /// Stem source tokens to their last N characters, after the prefix.
    #[arg(long, value_name = "N")]
    source_suffix: Option<usize>,
    /// Stem target tokens to their first N characters.
    #[arg(long, value_name = "N")]
    target_prefix: Option<usize>,
    /// Stem target tokens to their last N characters, after the prefix.
    #[arg(long, value_name = "N")]
    target_suffix: Option<usize>,
    /// Pairs with more tokens than this on either side are not aligned (empty line); eflomal
    /// skips pairs of 1024 tokens or more.
    #[arg(long, default_value_t = DEFAULT_MAX_LEN)]
    max_len: usize,
}

impl TextArgs {
    fn options(&self) -> R<TextOptions> {
        if !(1..=MAX_MAX_LEN).contains(&self.max_len) {
            return Err(format!("--max-len must be in 1 ..= {MAX_MAX_LEN}").into());
        }
        Ok(TextOptions {
            lowercase: !self.no_lowercase,
            source_prefix: self.source_prefix,
            source_suffix: self.source_suffix,
            target_prefix: self.target_prefix,
            target_suffix: self.target_suffix,
            max_len: self.max_len,
            chunks: None,
        })
    }
}

/// Sampling options shared by `train` and `align`.
#[derive(Args, Clone)]
struct SamplingArgs {
    /// Seed of the random number generator; the same seed gives the same output at any
    /// thread count (train: default 1; align: default the model's).
    #[arg(long)]
    seed: Option<u64>,
    /// Independent samplers per direction (eflomal `--n-samplers`; train: default 3; align:
    /// default the model's).
    #[arg(short = 'n', long, visible_alias = "n-samplers")]
    samplers: Option<usize>,
    /// Relative number of iterations (eflomal `-l`); the counts follow the corpus size (0 = the
    /// minimum, as eflomal).
    #[arg(short, long, default_value_t = 1.0)]
    length: f64,
    /// Model 1 iterations (default from the corpus size; eflomal `-1`).
    #[arg(short = '1', long, value_name = "N")]
    ibm1_iters: Option<usize>,
    /// Model 2 (HMM) iterations (eflomal `-2`).
    #[arg(short = '2', long, value_name = "N")]
    hmm_iters: Option<usize>,
    /// Model 3 (fertility) iterations (eflomal `-3`).
    #[arg(short = '3', long, value_name = "N")]
    fert_iters: Option<usize>,
    /// Use the exact sequential sampler for every chain (slower; the default samples each
    /// chain in parallel blocks, with a different but equally deterministic result).
    #[arg(long)]
    exact: bool,
    /// Decode from the conditionals of every sweep of the last model plus the final pass
    /// instead of the final pass alone (often a little better; needs Σ J·(I+1) floats per
    /// sampler and direction, refused above --accumulate-limit).
    #[arg(long)]
    accumulate: bool,
    /// Memory limit of --accumulate in MB.
    #[arg(
        long,
        value_name = "MB",
        default_value_t = 4096,
        requires = "accumulate"
    )]
    accumulate_limit: u64,
}

#[derive(Args)]
struct TrainArgs {
    /// Corpus: one file of `source ||| target` lines, or the source text (then give TGT).
    input: PathBuf,
    /// Target text, line-aligned with INPUT.
    tgt: Option<PathBuf>,
    /// Output prefix: writes PREFIX.fwd.pharaoh, PREFIX.inv.pharaoh and
    /// PREFIX.<method>.pharaoh for every --symmetrize. Without it the single direction chosen
    /// by --directions is written to standard output.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    /// Directions to train: both, or only forward (source → target) or inverse (target →
    /// source). Links are written in source-target order either way.
    #[arg(long, value_enum, default_value = "both")]
    directions: Dirs,
    /// 1 = IBM1, 2 = IBM1 + HMM, 3 = IBM1 + HMM + fertility (eflomal `-m`).
    #[arg(short, long, default_value_t = 3)]
    model: u8,
    #[command(flatten)]
    sampling: SamplingArgs,
    /// Prior probability of a NULL alignment (eflomal `--null-prior`; 1e-6 ..= 1 - 1e-6).
    #[arg(long, default_value_t = 0.2)]
    null_prior: f64,
    /// Dirichlet prior of the lexical distributions (every α: 1e-30 ..= 1e25).
    #[arg(long, default_value_t = 0.001)]
    lex_alpha: f64,
    /// Dirichlet prior of the jump distribution.
    #[arg(long, default_value_t = 0.5)]
    jump_alpha: f64,
    /// Dirichlet prior of the fertility distributions.
    #[arg(long, default_value_t = 1.0)]
    fert_alpha: f64,
    /// eflomal priors file (from `eflomalr makepriors` or eflomal-makepriors); its counts are
    /// added to the Dirichlet priors (eflomal `-p`).
    #[arg(short = 'p', long)]
    priors: Option<PathBuf>,
    #[command(flatten)]
    text: TextArgs,
    /// Stemming: `auto` chooses a prefix length on both sides from the corpus's line count
    /// (4 below 30,000 lines, 5 below 450,000, else 6); `words` does not stem. Explicit
    /// --source-prefix/--target-prefix/--source-suffix/--target-suffix options override both.
    /// `eflomalr estimate` shows the choice and the measurements behind it.
    #[arg(long, value_enum, default_value = "auto")]
    segment: SegmentArg,
    /// Symmetrisation heuristics to write (repeatable): intersect, union, grow-diag,
    /// grow-diag-final, grow-diag-final-and. Needs both directions and --output.
    #[arg(short = 's', long)]
    symmetrize: Vec<Method>,
    /// Append ` ||| score` (negative mean log probability per target token) to the
    /// directional lines (eflomal `-F`/`-R` write separate files).
    #[arg(long)]
    scores: bool,
    /// Model used for the scores (eflomal `-M`; default --model).
    #[arg(short = 'M', long, value_name = "N")]
    score_model: Option<u8>,
    /// Save the trained model here (for `eflomalr align`).
    #[arg(long)]
    save: Option<PathBuf>,
    /// Also write each direction's progress lines to PREFIX.<dir>.log.
    #[arg(long, requires = "output")]
    log: bool,
    /// No progress output.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Args)]
struct AlignArgs {
    /// MODEL SRC [TGT]; SRC alone is a `source ||| target` file. The text is lowercased,
    /// stemmed and length-filtered as the model's training corpus was.
    files: Vec<PathBuf>,
    /// Output file; standard output if omitted.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Symmetrisation heuristic (intersect, union, grow-diag, grow-diag-final,
    /// grow-diag-final-and; needs a model of both directions); otherwise one direction.
    #[arg(short, long)]
    symmetrize: Option<Method>,
    /// Which direction to write without --symmetrize.
    #[arg(long, value_enum, default_value = "fwd")]
    direction: OneDir,
    /// Append ` ||| score` to each line (not with --fast).
    #[arg(long)]
    scores: bool,
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    #[command(flatten)]
    sampling: SamplingArgs,
    /// Deterministic decode without sampling: the posterior of the Model-2 HMM with the
    /// model's counts as parameters (fast; works on any number of sentences).
    #[arg(long, conflicts_with_all = ["scores", "exact", "accumulate"])]
    fast: bool,
    /// No progress output.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Args)]
struct SymArgs {
    /// Forward alignments (Pharaoh), one line per sentence pair.
    fwd: PathBuf,
    /// Inverse alignments (Pharaoh, source-target orientation); not needed for `invert`.
    inv: Option<PathBuf>,
    /// intersect, union, grow-diag, grow-diag-final, grow-diag-final-and, or invert (swap
    /// the sides of FWD).
    #[arg(short, long, default_value = "grow-diag-final-and")]
    method: Method,
    /// Output file; standard output if omitted.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct PriorsArgs {
    /// Corpus: one file of `source ||| target` lines, or the source text (then give TGT).
    input: PathBuf,
    /// Target text, line-aligned with INPUT.
    tgt: Option<PathBuf>,
    /// Forward links (Pharaoh, source-target order): lexical, forward jump and fertility
    /// priors.
    #[arg(short = 'f', long)]
    forward_links: PathBuf,
    /// Reverse links (Pharaoh, source-target order; may be the same symmetrised file):
    /// reverse jump and fertility priors.
    #[arg(short = 'r', long)]
    reverse_links: PathBuf,
    /// Priors file to write (eflomal-makepriors `-p`); standard output if omitted.
    #[arg(short, long, short_alias = 'p')]
    output: Option<PathBuf>,
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    #[command(flatten)]
    text: TextArgs,
}

#[derive(Args)]
struct EstimateArgs {
    /// Training corpus: one file of `source ||| target` lines, or the source text (then give
    /// TGT). With --model: the text to align instead.
    input: PathBuf,
    /// Target text, line-aligned with INPUT.
    tgt: Option<PathBuf>,
    /// Text to be aligned later with a model trained on INPUT (`source ||| target` file, or
    /// the source side with --align-tgt): compared with the training corpus.
    #[arg(long, value_name = "FILE", conflicts_with = "model")]
    align_src: Option<PathBuf>,
    /// Target side of --align-src.
    #[arg(long, value_name = "FILE", requires = "align_src")]
    align_tgt: Option<PathBuf>,
    /// A saved model: report its segmentation (its training corpus fixed it) and compare
    /// INPUT [TGT], the text to align, with its training corpus.
    #[arg(long, value_name = "MODEL")]
    model: Option<PathBuf>,
    /// As `train --segment`.
    #[arg(long, value_enum, default_value = "auto", conflicts_with = "model")]
    segment: SegmentArg,
    #[command(flatten)]
    text: TextArgs,
    /// Machine-readable output (JSON).
    #[arg(long)]
    json: bool,
    /// Output file; standard output if omitted.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
}

type R<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match cli.command {
        Command::Train(a) => cmd_train(a),
        Command::Align(a) => cmd_align(a),
        Command::Symmetrize(a) => cmd_symmetrize(a),
        Command::Makepriors(a) => cmd_makepriors(a),
        Command::Estimate(a) => cmd_estimate(a),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("eflomalr: error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn out_file(p: &Path) -> R<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        1 << 20,
        File::create(p).map_err(|e| format!("{}: {e}", p.display()))?,
    ))
}

fn write_file(path: &Path, bytes: &[u8]) -> R<()> {
    let mut w = out_file(path)?;
    w.write_all(bytes)?;
    w.flush()?;
    Ok(())
}

fn write_out(path: &Option<PathBuf>, bytes: &[u8]) -> R<()> {
    match path {
        Some(p) => write_file(p, bytes),
        None => {
            let mut w = io::stdout().lock();
            w.write_all(bytes)?;
            w.flush()?;
            Ok(())
        }
    }
}

fn read_file(p: &Path) -> R<Vec<u8>> {
    let mut v = Vec::new();
    if p == Path::new("-") {
        io::stdin().read_to_end(&mut v)?;
    } else {
        v = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(v)
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn read_bitext(input: &Path, tgt: &Option<PathBuf>, opts: &TextOptions) -> R<Bitext> {
    Ok(read_parsed(input, tgt, opts.chunks)?.build(opts))
}

/// Parse a corpus without normalising it yet (the first phase of reading).
fn read_parsed(input: &Path, tgt: &Option<PathBuf>, chunks: Option<usize>) -> R<Parsed> {
    Ok(match tgt {
        Some(t) => Parsed::read_pair(input, t, chunks)?,
        None if input == Path::new("-") => Parsed::from_joined_bytes(&read_file(input)?, chunks),
        None => Parsed::read_joined(input, chunks)?,
    })
}

fn report_read(b: &Bitext, t0: Instant) {
    eprintln!(
        "read {} lines ({} skipped for length), vocabularies {} / {}, {:.1} s",
        b.len(),
        b.skipped().len(),
        b.source.n_types() - 1,
        b.target.n_types() - 1,
        t0.elapsed().as_secs_f64()
    );
}

fn read_priors(path: &Path) -> R<PriorSet> {
    let bytes = read_file(path)?;
    Ok(PriorSet::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?)
}

fn iters(s: &SamplingArgs) -> [Option<usize>; 3] {
    [s.ibm1_iters, s.hmm_iters, s.fert_iters]
}

/// Progress lines on stderr (unless quiet) and, with `logs`, into one file per direction.
struct Reporter {
    quiet: bool,
    /// fwd, inv: a log file for each requested direction only
    logs: [Option<Mutex<BufWriter<File>>>; 2],
}

impl Reporter {
    fn report(&self, p: &Progress) {
        if !self.quiet {
            eprintln!("{p}");
        }
        let k = usize::from(p.direction == Direction::Inverse);
        if let Some(log) = &self.logs[k] {
            if let Ok(mut w) = log.lock() {
                let _ = writeln!(w, "{p}");
            }
        }
    }

    fn finish(self) -> R<()> {
        for l in self.logs.into_iter().flatten() {
            l.into_inner().map_err(|_| "log writer poisoned")?.flush()?;
        }
        Ok(())
    }
}

fn cmd_train(a: TrainArgs) -> R<()> {
    let dirs = a.directions.directions();
    if !a.symmetrize.is_empty() {
        if a.output.is_none() || dirs != Directions::Both {
            return Err("--symmetrize needs --output and both directions".into());
        }
        if a.symmetrize.contains(&Method::Invert) {
            return Err("`invert` is not a symmetrisation of two directions".into());
        }
    }
    if a.output.is_none() && dirs == Directions::Both {
        return Err("without --output choose one direction with --directions fwd|inv".into());
    }
    let opts = a.text.options()?;
    let segment = a.segment.segment();
    let priors = a.priors.as_deref().map(read_priors).transpose()?;
    let cfg = TrainConfig {
        directions: dirs,
        seed: a.sampling.seed.unwrap_or(1),
        model: a.model,
        samplers: a.sampling.samplers.unwrap_or(3),
        hyper: Hyper {
            lex_alpha: a.lex_alpha,
            jump_alpha: a.jump_alpha,
            fert_alpha: a.fert_alpha,
            null_prior: a.null_prior,
        },
        length: a.sampling.length,
        iters: iters(&a.sampling),
        scores: a.scores,
        exact: a.sampling.exact,
        score_model: a.score_model,
        priors: priors.map(Arc::new),
        keep_model: a.save.is_some(),
        accumulate: a.sampling.accumulate,
        accumulate_limit: a.sampling.accumulate_limit << 20,
    };
    cfg.validate()?;
    // a log file is created only for a direction that is trained: a one-direction run never
    // touches (or removes) the other direction's log of an earlier run
    let mut logs = [None, None];
    if let (Some(prefix), true) = (&a.output, a.log) {
        for d in dirs.list() {
            let path = with_suffix(prefix, &format!(".{}.log", d.name()));
            logs[usize::from(d == Direction::Inverse)] = Some(Mutex::new(out_file(&path)?));
        }
    }
    let reporter = Reporter {
        quiet: a.quiet,
        logs,
    };
    let quiet = a.quiet;
    let out = with_threads(a.threads, || -> R<Trained> {
        let t0 = Instant::now();
        let bitext = read_parsed(&a.input, &a.tgt, opts.chunks)?.build_segmented(&opts, segment);
        timing::stop(Phase::Read, timing::enabled().then_some(t0));
        if !quiet {
            report_read(&bitext, t0);
            let s = bitext.segmentation().expect("training path");
            eprintln!("{}", choice_line(bitext.options(), s.origin, s.stats.lines));
        }
        warn(&bitext);
        let out = train(&bitext, &cfg, &|p| reporter.report(p))?;
        if !quiet {
            let it = out.iterations;
            eprintln!(
                "trained (iterations {}/{}/{}), {:.1} s",
                it[0],
                it[1],
                it[2],
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(out)
    })?;
    reporter.finish()?;
    match &a.output {
        None => write_out(
            &None,
            out.fwd
                .as_ref()
                .or(out.inv.as_ref())
                .expect("one direction"),
        )?,
        Some(prefix) => {
            for d in [Direction::Forward, Direction::Inverse] {
                if let Some(t) = out.output(d) {
                    timing::time(Phase::Write, || {
                        write_file(&with_suffix(prefix, &format!(".{}.pharaoh", d.name())), t)
                    })?;
                }
            }
            if let (Some(f), Some(i)) = (&out.fwd, &out.inv) {
                for m in &a.symmetrize {
                    let s = timing::time(Phase::Symmetrize, || {
                        with_threads(a.threads, || symmetrize_pharaoh(f, i, *m))
                    })?;
                    timing::time(Phase::Write, || {
                        write_file(&with_suffix(prefix, &format!(".{}.pharaoh", m.name())), &s)
                    })?;
                }
            }
        }
    }
    if let (Some(path), Some(model)) = (&a.save, &out.model) {
        timing::time(Phase::Write, || model.save(path))?;
        if !a.quiet {
            eprintln!("saved the model to {}", path.display());
        }
    }
    if timing::enabled() {
        eprint!("{}", timing::report());
    }
    Ok(())
}

/// Corpus warnings (nothing to align): printed even with --quiet, never in --log files.
fn warn(bitext: &Bitext) {
    for w in corpus_warnings(bitext) {
        eprintln!("eflomalr: warning: {w}");
    }
}

fn cmd_align(a: AlignArgs) -> R<()> {
    let (model_path, input, tgt) = match a.files.as_slice() {
        [m, s] => (m, s, None),
        [m, s, t] => (m, s, Some(t.clone())),
        _ => return Err("usage: eflomalr align MODEL SRC [TGT]".into()),
    };
    if a.symmetrize == Some(Method::Invert) {
        return Err("`invert` is not a symmetrisation of two directions".into());
    }
    let model = Model::load(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?;
    let directions = match (a.symmetrize, a.direction) {
        (Some(_), _) => Directions::Both,
        (None, OneDir::Inv) => Directions::Inverse,
        (None, OneDir::Fwd) => Directions::Forward,
    };
    if a.symmetrize.is_some() && model.directions() != Some(Directions::Both) {
        return Err("--symmetrize needs a model of both directions".into());
    }
    let cfg = AlignConfig {
        directions: Some(directions),
        seed: a.sampling.seed,
        samplers: a.sampling.samplers,
        length: Some(a.sampling.length),
        iters: iters(&a.sampling),
        scores: a.scores,
        exact: a.sampling.exact,
        accumulate: a.sampling.accumulate,
        accumulate_limit: Some(a.sampling.accumulate_limit << 20),
    };
    let quiet = a.quiet;
    let text = with_threads(a.threads, || -> R<Vec<u8>> {
        let t0 = Instant::now();
        let bitext = read_bitext(input, &tgt, &model.options.text)?;
        if !quiet {
            report_read(&bitext, t0);
        }
        warn(&bitext);
        let out = if a.fast {
            align_fast(&model, &bitext, Some(directions))?
        } else {
            let progress = |p: &Progress| {
                if !quiet {
                    eprintln!("{p}");
                }
            };
            align(&model, &bitext, &cfg, &progress)?
        };
        if !quiet {
            eprintln!("aligned, {:.1} s", t0.elapsed().as_secs_f64());
        }
        Ok(match a.symmetrize {
            Some(m) => symmetrize_pharaoh(
                out.fwd.as_ref().expect("fwd"),
                out.inv.as_ref().expect("inv"),
                m,
            )?,
            None => out.fwd.or(out.inv).expect("one direction"),
        })
    })?;
    write_out(&a.output, &text)
}

fn cmd_symmetrize(a: SymArgs) -> R<()> {
    let fwd = read_file(&a.fwd)?;
    let inv = match (&a.inv, a.method) {
        (Some(p), m) if m != Method::Invert => Some(read_file(p)?),
        (None, m) if m != Method::Invert => {
            return Err(format!("{m} needs two alignment files").into())
        }
        _ => None,
    };
    let mut out = Vec::new();
    symmetrize_text(&fwd, inv.as_deref(), a.method, &mut out).map_err(|e| e.to_string())?;
    write_out(&a.output, &out)
}

fn cmd_makepriors(a: PriorsArgs) -> R<()> {
    // every pair counts, whatever its length (as eflomal-makepriors)
    let opts = TextOptions {
        max_len: NO_MAX_LEN, // every length counts (no sampling)
        ..a.text.options()?
    };
    let fwd = read_file(&a.forward_links)?;
    // `-f - -r -`: standard input is read once and used for both (as the Python CLI)
    let rev = if a.forward_links == a.reverse_links && a.forward_links == Path::new("-") {
        fwd.clone()
    } else {
        read_file(&a.reverse_links)?
    };
    let ps = with_threads(a.threads, || -> R<PriorSet> {
        let bitext = read_bitext(&a.input, &a.tgt, &opts)?;
        Ok(make_priors(&bitext, &fwd, &rev)?)
    })?;
    let mut buf = Vec::new();
    ps.write(&mut buf)?;
    write_out(&a.output, &buf)
}

fn cmd_estimate(a: EstimateArgs) -> R<()> {
    let opts = a.text.options()?;
    let est = with_threads(a.threads, || -> R<Estimate> {
        if let Some(mp) = &a.model {
            let model = Model::load(mp).map_err(|e| format!("{}: {e}", mp.display()))?;
            let mut est = estimate_model(&model.options.text, model.segmentation.as_ref());
            let parsed = read_parsed(&a.input, &a.tgt, None)?;
            let new = parsed.build(&model.options.text);
            est.align = Some(align_check(
                "model",
                model.segmentation.as_ref().map(|s| &s.stats),
                &model.source,
                &model.target,
                &new,
                parsed.stats(model.options.text.lowercase),
            ));
            return Ok(est);
        }
        let parsed = read_parsed(&a.input, &a.tgt, None)?;
        let stats = parsed.stats(opts.lowercase);
        let mut est = estimate(&stats, &opts, a.segment.segment());
        if let Some(src) = &a.align_src {
            let train = parsed.build(&est.text);
            drop(parsed);
            let new_parsed = read_parsed(src, &a.align_tgt, None)?;
            let new = new_parsed.build(&est.text);
            est.align = Some(align_check(
                "training corpus",
                Some(&stats),
                &train.source.vocab,
                &train.target.vocab,
                &new,
                new_parsed.stats(opts.lowercase),
            ));
        }
        Ok(est)
    })?;
    let text = if a.json { est.json() } else { est.text() };
    write_out(&a.output, text.as_bytes())
}
