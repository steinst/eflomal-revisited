//! PyO3 layer over eflomal-core. Exposed to Python as `eflomalr._core`.
//!
//! Deliberately thin: the documented API lives in the Python package `eflomalr` (`train`,
//! `Model`, ...), which validates and normalises its inputs and calls the classes and functions
//! here. All alignment output is the Pharaoh *text* that the core produces (the same bytes as
//! the native `eflomalr` binary); [`PyAlignments`] only indexes into it. Every long computation
//! runs without the GIL on a helper thread, while the calling thread polls for Ctrl-C (see
//! [`run_interruptible`]).

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eflomal_core::bitext::{Bitext, Parsed, TextOptions, MAX_MAX_LEN, NO_MAX_LEN};
use eflomal_core::cancel;
use eflomal_core::corpus::CorpusError;
use eflomal_core::hmm::FastModel;
use eflomal_core::io::{Direction, Model, ModelError};
use eflomal_core::model::Hyper;
use eflomal_core::pipeline::{
    align as core_align, align_fast_with, corpus_warnings, symmetrize_pharaoh, train as core_train,
    with_threads, AlignConfig, Directions, Progress, TrainConfig, Trained,
};
use eflomal_core::priors::{make_priors as core_make_priors, PriorSet};
use eflomal_core::segment::{
    align_check, choice_line, estimate as core_estimate, estimate_model, Estimate, Segment,
};
use eflomal_core::symmetrize::{self, format_links, Link, Method};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyKeyboardInterrupt, PyOSError, PyRuntimeError,
    PyRuntimeWarning, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pymodule;
use pyo3::types::{PyBytes, PySlice};

type Links = Vec<Link>;

// ---------------------------------------------------------------------------------------------
// Errors

fn io_err(e: &std::io::Error, what: String) -> PyErr {
    if e.kind() == std::io::ErrorKind::NotFound {
        PyFileNotFoundError::new_err(what)
    } else {
        PyOSError::new_err(what)
    }
}

fn corpus_err(e: CorpusError) -> PyErr {
    match &e {
        CorpusError::Io(_, source) => io_err(source, e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn model_err(e: ModelError, path: Option<&Path>) -> PyErr {
    let at = |s: String| match path {
        Some(p) => format!("{}: {s}", p.display()),
        None => s,
    };
    match &e {
        ModelError::Io(source) => io_err(source, at(source.to_string())),
        ModelError::Format(s) => PyValueError::new_err(at(s.clone())),
        ModelError::Unsupported(s) => PyValueError::new_err(s.clone()),
    }
}

// ---------------------------------------------------------------------------------------------
// Interruptible execution

/// Run `f` on a helper thread with the GIL released, under its own cancellation scope
/// ([`cancel::Scope`]). The calling thread wakes every 50 ms to check for pending signals; on
/// Ctrl-C (or when `slot` receives an error from this call's progress callback) it cancels
/// every eflomalr call active at that moment, and returns that error once the helper has
/// unwound. A call cancelled by another call's interrupt or callback error raises
/// `KeyboardInterrupt`. A call started later is not affected.
fn run_interruptible<T: Send>(
    py: Python<'_>,
    slot: &Mutex<Option<PyErr>>,
    f: impl FnOnce() -> T + Send,
) -> PyResult<T> {
    let scope = cancel::Scope::new();
    let (tx, rx) = mpsc::channel();
    let rx = Mutex::new(rx);
    let result = std::thread::scope(|s| {
        let scope = &scope;
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn_scoped(s, move || {
                let _entered = scope.enter();
                let r = catch_unwind(AssertUnwindSafe(f));
                let _ = tx.send(r);
            })
            .map_err(|e| PyRuntimeError::new_err(format!("cannot start a thread: {e}")))?;
        loop {
            match py.detach(|| rx.lock().unwrap().recv_timeout(Duration::from_millis(50))) {
                Ok(r) => return Ok(r),
                Err(RecvTimeoutError::Timeout) => {
                    if !scope.is_cancelled() {
                        if let Err(e) = py.check_signals() {
                            slot.lock().unwrap().get_or_insert(e);
                            cancel::cancel_active();
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(PyRuntimeError::new_err("worker thread died"))
                }
            }
        }
    });
    drop(scope);
    let r = result?;
    let pending = slot.lock().unwrap().take();
    match r {
        Ok(v) => match pending {
            Some(e) => Err(e),
            None => Ok(v),
        },
        Err(p) if cancel::is_cancelled(&*p) => Err(pending.unwrap_or_else(|| {
            PyKeyboardInterrupt::new_err(
                "interrupted: another eflomalr call running at the same time was interrupted \
                 or its progress callback raised",
            )
        })),
        Err(p) => resume_unwind(p),
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs

/// A text argument: a file path or in-memory sentences (one whitespace-separated string each).
enum Text {
    File(PathBuf),
    Lines(Vec<String>),
}

fn text_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Text> {
    if let Ok(p) = obj.extract::<PathBuf>() {
        return Ok(Text::File(p));
    }
    obj.extract::<Vec<String>>().map(Text::Lines).map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a file path or a list of sentences"))
    })
}

impl Text {
    /// The bytes of the text, one line per sentence.
    fn into_bytes(self) -> PyResult<Vec<u8>> {
        match self {
            Text::File(p) => {
                std::fs::read(&p).map_err(|e| io_err(&e, format!("{}: {e}", p.display())))
            }
            Text::Lines(l) => {
                let mut v = Vec::with_capacity(l.iter().map(|s| s.len() + 1).sum());
                for s in &l {
                    v.extend_from_slice(s.as_bytes());
                    v.push(b'\n');
                }
                Ok(v)
            }
        }
    }
}

/// Read a corpus: one `source ||| target` file (`tgt` is `None`), or two line-aligned texts,
/// with the given normalisation (as the native binary's `read_bitext`).
fn load_bitext(src: Text, tgt: Option<Text>, opts: &TextOptions) -> PyResult<Bitext> {
    Ok(load_parsed(src, tgt)?.build(opts))
}

/// Parse a corpus without normalising it (the first phase of reading; see [`Parsed`]).
fn load_parsed(src: Text, tgt: Option<Text>) -> PyResult<Parsed> {
    match (src, tgt) {
        (Text::File(a), None) => Parsed::read_joined(&a, None).map_err(corpus_err),
        (Text::Lines(_), None) => Err(PyTypeError::new_err(
            "a single text must be the path of a file of `source ||| target` lines",
        )),
        (Text::File(a), Some(Text::File(b))) => Parsed::read_pair(&a, &b, None).map_err(corpus_err),
        (a, Some(b)) => {
            let (a, b) = (a.into_bytes()?, b.into_bytes()?);
            Parsed::from_pair_bytes(&a, &b, None).map_err(corpus_err)
        }
    }
}

/// The text options `(lowercase, source_prefix, source_suffix, target_prefix, target_suffix,
/// max_len)` of the Python layer.
type TextTuple = (
    bool,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    usize,
);

fn text_options(t: TextTuple) -> PyResult<TextOptions> {
    let (lowercase, sp, ss, tp, ts, max_len) = t;
    if !(1..=MAX_MAX_LEN).contains(&max_len) {
        return Err(PyValueError::new_err(format!(
            "max_len must be in 1 ..= {MAX_MAX_LEN}, got {max_len}"
        )));
    }
    Ok(TextOptions {
        lowercase,
        source_prefix: sp,
        source_suffix: ss,
        target_prefix: tp,
        target_suffix: ts,
        max_len,
        chunks: None,
    })
}

fn parse_directions(s: &str) -> PyResult<Directions> {
    match s {
        "both" => Ok(Directions::Both),
        "fwd" => Ok(Directions::Forward),
        "inv" => Ok(Directions::Inverse),
        _ => Err(PyValueError::new_err(format!(
            "directions must be 'both', 'fwd' or 'inv', not '{s}'"
        ))),
    }
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s {
        "fwd" => Ok(Direction::Forward),
        "inv" => Ok(Direction::Inverse),
        _ => Err(PyValueError::new_err(format!(
            "direction must be 'fwd' or 'inv', not '{s}'"
        ))),
    }
}

/// A symmetrisation method of two directions (`invert` is a different operation).
fn parse_method(s: &str) -> PyResult<Method> {
    let m: Method = s.parse().map_err(PyValueError::new_err)?;
    if m == Method::Invert {
        return Err(PyValueError::new_err(
            "'invert' is not a symmetrisation of two directions",
        ));
    }
    Ok(m)
}

// ---------------------------------------------------------------------------------------------
// Progress

/// Forwards progress events to an optional Python callable and keeps each direction's events
/// (formatted as the native binary's `--log` lines only when `training_log` asks for them).
struct Reporter<'a> {
    callback: Option<&'a Py<PyAny>>,
    slot: &'a Mutex<Option<PyErr>>,
    logs: [Mutex<Vec<Progress>>; 2],
}

impl<'a> Reporter<'a> {
    fn new(callback: Option<&'a Py<PyAny>>, slot: &'a Mutex<Option<PyErr>>) -> Self {
        Reporter {
            callback,
            slot,
            logs: [Mutex::new(Vec::new()), Mutex::new(Vec::new())],
        }
    }

    fn report(&self, p: &Progress) {
        let k = usize::from(p.direction == Direction::Inverse);
        if let Ok(mut l) = self.logs[k].lock() {
            l.push(p.clone());
        }
        let Some(cb) = self.callback else { return };
        if cancel::is_requested() {
            return; // this call is being cancelled (the event runs on one of its threads)
        }
        Python::attach(|py| {
            let args = (
                p.direction.name(),
                p.sampler,
                p.model,
                p.iteration,
                p.iterations,
                p.elapsed.as_secs_f64(),
                p.null_fraction,
            );
            if let Err(e) = cb.call1(py, args) {
                self.slot.lock().unwrap().get_or_insert(e);
                cancel::cancel_active();
            }
        });
    }

    fn into_logs(self) -> [Vec<Progress>; 2] {
        self.logs.map(|l| l.into_inner().unwrap_or_default())
    }
}

// ---------------------------------------------------------------------------------------------
// Alignment results

/// Parse the links of one Pharaoh line, ignoring a ` ||| score` suffix.
fn parse_links(line: &[u8]) -> Links {
    let cut = line
        .windows(5)
        .position(|w| w == b" ||| ")
        .unwrap_or(line.len());
    line[..cut]
        .split(|&b| b == b' ')
        .filter(|t| !t.is_empty())
        .filter_map(|t| {
            let dash = t.iter().position(|&b| b == b'-')?;
            let num = |s: &[u8]| std::str::from_utf8(s).ok()?.parse::<u32>().ok();
            Some((num(&t[..dash])?, num(&t[dash + 1..])?))
        })
        .collect()
}

/// A read-only sequence of alignments, one list of `(source, target)` links per line, backed by
/// the Pharaoh text itself (so that writing it out gives exactly the core's bytes).
#[pyclass(name = "Alignments", module = "eflomalr._core", frozen)]
struct PyAlignments {
    text: Vec<u8>,
    /// Start offset of every line, plus the end of the text.
    bounds: Vec<usize>,
}

impl PyAlignments {
    fn from_text(text: Vec<u8>) -> Self {
        let mut bounds = vec![0];
        bounds.extend(
            text.iter()
                .enumerate()
                .filter(|(_, &b)| b == b'\n')
                .map(|(k, _)| k + 1),
        );
        if *bounds.last().unwrap() != text.len() {
            bounds.push(text.len());
        }
        PyAlignments { text, bounds }
    }

    fn n(&self) -> usize {
        self.bounds.len() - 1
    }

    fn line(&self, i: usize) -> &[u8] {
        let (a, b) = (self.bounds[i], self.bounds[i + 1]);
        let l = &self.text[a..b];
        l.strip_suffix(b"\n").unwrap_or(l)
    }

    fn index(&self, i: isize) -> PyResult<usize> {
        let n = self.n() as isize;
        let j = if i < 0 { i + n } else { i };
        if (0..n).contains(&j) {
            Ok(j as usize)
        } else {
            Err(PyIndexError::new_err("alignment index out of range"))
        }
    }
}

#[pymethods]
impl PyAlignments {
    fn __len__(&self) -> usize {
        self.n()
    }

    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(slice) = index.cast::<PySlice>() {
            let ix = slice.indices(self.n() as isize)?;
            let mut out = Vec::with_capacity(ix.slicelength);
            let mut i = ix.start;
            for _ in 0..ix.slicelength {
                out.push(parse_links(self.line(i as usize)));
                i += ix.step;
            }
            return Ok(out.into_pyobject(py)?.into_any().unbind());
        }
        let i: isize = index
            .extract()
            .map_err(|_| PyTypeError::new_err("indices must be integers or slices"))?;
        let links = parse_links(self.line(self.index(i)?));
        Ok(links.into_pyobject(py)?.into_any().unbind())
    }

    fn __iter__(slf: Py<Self>) -> AlignmentsIter {
        AlignmentsIter { src: slf, pos: 0 }
    }

    /// Pharaoh-format line `i` (`"0-0 1-2"`), without any score.
    fn pharaoh(&self, i: isize) -> PyResult<String> {
        let mut out = Vec::new();
        format_links(&parse_links(self.line(self.index(i)?)), &mut out);
        Ok(String::from_utf8(out).expect("ascii"))
    }

    /// The score of line `i` (the number after ` ||| `), or `None` without scores.
    fn score(&self, i: isize) -> PyResult<Option<f64>> {
        let line = self.line(self.index(i)?);
        Ok(line.windows(5).rposition(|w| w == b" ||| ").and_then(|k| {
            std::str::from_utf8(&line[k + 5..])
                .ok()?
                .trim()
                .parse()
                .ok()
        }))
    }

    /// All alignments as a list of lists of `(source, target)` pairs.
    fn to_list(&self) -> Vec<Links> {
        (0..self.n()).map(|i| parse_links(self.line(i))).collect()
    }

    /// The Pharaoh text, exactly as written by [`write`](Self::write).
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.text)
    }

    /// Write the alignments in Pharaoh format, one line per sentence pair.
    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| write_file(&path, &self.text))
    }
}

#[pyclass(module = "eflomalr._core")]
struct AlignmentsIter {
    src: Py<PyAlignments>,
    pos: usize,
}

#[pymethods]
impl AlignmentsIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Links> {
        let a = self.src.get();
        if self.pos >= a.n() {
            return None;
        }
        let l = parse_links(a.line(self.pos));
        self.pos += 1;
        Some(l)
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> PyResult<()> {
    std::fs::write(path, bytes).map_err(|e| io_err(&e, format!("{}: {e}", path.display())))
}

// ---------------------------------------------------------------------------------------------
// Model

/// What training produced besides the model.
struct TrainData {
    fwd: Option<Vec<u8>>,
    inv: Option<Vec<u8>>,
    /// Progress events of each direction (`fwd`, `inv`), formatted on demand.
    logs: [Vec<Progress>; 2],
    /// Number of lines of the input (alignment texts have one line per input line).
    n_lines: usize,
    /// 0-based indices of the pairs skipped for length.
    skipped: Vec<usize>,
    /// Iterations of Model 1, 2, 3 that ran.
    iterations: [usize; 3],
    /// `pipeline::corpus_warnings` of the training corpus.
    warnings: Vec<String>,
}

struct Inner {
    model: Model,
    train: Option<TrainData>,
    /// The model indexed for fast decoding, built on first use and reused by every call.
    fast: FastModel,
}

enum Sel {
    Dir(Direction),
    Sym(Method),
}

impl Inner {
    fn has(&self, d: Direction) -> bool {
        self.model.table(d).is_some()
    }

    /// Resolve the `symmetrize` / `direction` arguments (see `eflomalr.Model.alignments`).
    fn select(&self, symmetrize: Option<&str>, direction: Option<&str>) -> PyResult<Sel> {
        let both = self.has(Direction::Forward) && self.has(Direction::Inverse);
        match (symmetrize, direction) {
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "give either symmetrize or direction, not both",
            )),
            (Some(m), None) => {
                if !both {
                    return Err(PyValueError::new_err(
                        "symmetrisation needs a model trained in both directions",
                    ));
                }
                Ok(Sel::Sym(parse_method(m)?))
            }
            (None, Some(d)) => {
                let d = parse_direction(d)?;
                if !self.has(d) {
                    return Err(PyValueError::new_err(format!(
                        "the model has no '{}' direction",
                        d.name()
                    )));
                }
                Ok(Sel::Dir(d))
            }
            (None, None) => {
                if both {
                    Ok(Sel::Sym(Method::GrowDiagFinalAnd))
                } else if self.has(Direction::Forward) {
                    Ok(Sel::Dir(Direction::Forward))
                } else {
                    Ok(Sel::Dir(Direction::Inverse))
                }
            }
        }
    }

    fn train_data(&self) -> PyResult<&TrainData> {
        self.train.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "this model was loaded from a file and holds no training alignments; \
                 use align() or align_many() to align sentences with it",
            )
        })
    }

    /// The Pharaoh text of the training alignments for a selection.
    fn training_text(&self, sel: &Sel) -> PyResult<Vec<u8>> {
        let td = self.train_data()?;
        let dir = |d: Direction| match d {
            Direction::Forward => td.fwd.as_ref(),
            Direction::Inverse => td.inv.as_ref(),
        };
        match sel {
            Sel::Dir(d) => Ok(dir(*d).expect("direction checked").clone()),
            Sel::Sym(m) => symmetrize_pharaoh(
                dir(Direction::Forward).expect("both directions checked"),
                dir(Direction::Inverse).expect("both directions checked"),
                *m,
            )
            .map_err(PyValueError::new_err),
        }
    }
}

/// The options of a saved model: `(model, samplers, seed, lex_alpha, jump_alpha, fert_alpha,
/// null_prior, text options)`.
type OptionsTuple = (u8, usize, u64, f64, f64, f64, f64, TextTuple);

/// The sampling arguments of `align_many`: `(seed, samplers, length, iterations, exact,
/// accumulate, accumulate_limit in MB)`.
type SamplingTuple = (
    Option<u64>,
    Option<usize>,
    f64,
    (Option<usize>, Option<usize>, Option<usize>),
    bool,
    bool,
    u64,
);

type SegmentationTuple = (&'static str, bool, u64, (u64, u64, u64), (u64, u64, u64));

/// A trained model (both directions, or one) and, right after training, the alignments of the
/// training corpus.
#[pyclass(name = "Model", module = "eflomalr._core", frozen)]
struct PyModel {
    inner: Arc<Inner>,
}

#[pymethods]
impl PyModel {
    /// Directions the model holds: a subset of `("fwd", "inv")`.
    #[getter]
    fn directions(&self) -> Vec<&'static str> {
        [Direction::Forward, Direction::Inverse]
            .into_iter()
            .filter(|&d| self.inner.has(d))
            .map(Direction::name)
            .collect()
    }

    /// Number of distinct (normalised) words of the source and of the target side.
    #[getter]
    fn vocab_size(&self) -> (usize, usize) {
        let m = &self.inner.model;
        (
            m.source.len().saturating_sub(1),
            m.target.len().saturating_sub(1),
        )
    }

    #[getter]
    fn has_training_alignments(&self) -> bool {
        self.inner.train.is_some()
    }

    /// Number of input lines of the training corpus, or `None` for a loaded model.
    #[getter]
    fn n_lines(&self) -> Option<usize> {
        self.inner.train.as_ref().map(|t| t.n_lines)
    }

    /// Pairs skipped for length in training (0-based), or `None` for a loaded model.
    #[getter]
    fn skipped(&self) -> Option<Vec<usize>> {
        self.inner.train.as_ref().map(|t| t.skipped.clone())
    }

    /// What training warned about (nothing to align); empty for a loaded model.
    #[getter]
    fn training_warnings(&self) -> Vec<String> {
        self.inner
            .train
            .as_ref()
            .map_or_else(Vec::new, |t| t.warnings.clone())
    }

    /// Iterations of Model 1, 2, 3 of the training run, or `None` for a loaded model.
    #[getter]
    fn iterations(&self) -> Option<(usize, usize, usize)> {
        self.inner
            .train
            .as_ref()
            .map(|t| (t.iterations[0], t.iterations[1], t.iterations[2]))
    }

    /// The options the model was trained with.
    #[getter]
    fn options(&self) -> OptionsTuple {
        let o = &self.inner.model.options;
        let t = &o.text;
        (
            o.model,
            o.samplers,
            o.seed,
            o.hyper.lex_alpha,
            o.hyper.jump_alpha,
            o.hyper.fert_alpha,
            o.hyper.null_prior,
            (
                t.lowercase,
                t.source_prefix,
                t.source_suffix,
                t.target_prefix,
                t.target_suffix,
                t.max_len,
            ),
        )
    }

    /// How the stemming was chosen and the training corpus's statistics: `(origin, lowercase,
    /// lines, (source tokens, types, singletons), (target ...))`, or `None` for a model saved
    /// without them.
    #[getter]
    fn segmentation(&self) -> Option<SegmentationTuple> {
        self.inner.model.segmentation.map(|s| {
            let side = |x: &eflomal_core::segment::SideStats| (x.tokens, x.types, x.singletons);
            (
                s.origin.name(),
                s.stats.lowercase,
                s.stats.lines,
                side(&s.stats.source),
                side(&s.stats.target),
            )
        })
    }

    /// The native binary's `segmentation: ...` line for this model, or `None` without the
    /// record.
    #[getter]
    fn segmentation_line(&self) -> Option<String> {
        let m = &self.inner.model;
        m.segmentation
            .map(|s| choice_line(&m.options.text, s.origin, s.stats.lines))
    }

    /// The progress lines of one direction of the training run (the native binary's `--log`).
    fn training_log<'py>(&self, py: Python<'py>, direction: &str) -> PyResult<Bound<'py, PyBytes>> {
        let d = parse_direction(direction)?;
        let td = self.inner.train_data()?;
        if !self.inner.has(d) {
            return Err(PyValueError::new_err(format!(
                "the model has no '{}' direction",
                d.name()
            )));
        }
        let mut text = Vec::new();
        for p in &td.logs[usize::from(d == Direction::Inverse)] {
            use std::io::Write;
            let _ = writeln!(text, "{p}");
        }
        Ok(PyBytes::new(py, &text))
    }

    /// Write the model to `path` through a uniquely named, exclusively created temporary file
    /// in the same directory, renamed over `path` when complete (`Model::save`).
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = &self.inner;
        // the I/O error already names the path
        py.detach(|| inner.model.save(&path))
            .map_err(|e| model_err(e, None))
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<PyModel> {
        let model = py
            .detach(|| Model::load(&path))
            .map_err(|e| model_err(e, Some(&path)))?;
        if model.directions().is_none() {
            return Err(PyValueError::new_err(format!(
                "{}: the model has no direction",
                path.display()
            )));
        }
        Ok(PyModel {
            inner: Arc::new(Inner {
                model,
                train: None,
                fast: FastModel::default(),
            }),
        })
    }

    /// Alignments of the training corpus, one entry per input line.
    #[pyo3(signature = (symmetrize=None, direction=None))]
    fn training_alignments(
        &self,
        py: Python<'_>,
        symmetrize: Option<&str>,
        direction: Option<&str>,
    ) -> PyResult<PyAlignments> {
        let inner = &self.inner;
        let sel = inner.select(symmetrize, direction)?;
        inner.train_data()?;
        let text = py.detach(|| inner.training_text(&sel))?;
        Ok(PyAlignments::from_text(text))
    }

    /// Write the training alignments to `path` in Pharaoh format.
    #[pyo3(signature = (path, symmetrize=None, direction=None))]
    fn write_training_alignments(
        &self,
        py: Python<'_>,
        path: PathBuf,
        symmetrize: Option<&str>,
        direction: Option<&str>,
    ) -> PyResult<()> {
        let inner = &self.inner;
        let sel = inner.select(symmetrize, direction)?;
        inner.train_data()?;
        py.detach(|| write_file(&path, &inner.training_text(&sel)?))
    }

    /// Align new sentence pairs with the model (file paths or lists of sentences; with `tgt`
    /// omitted, `src` is a `source ||| target` file), read with the model's text options.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, tgt, symmetrize, direction, threads, scores, fast, sampling,
                        progress=None))]
    fn align_many(
        &self,
        py: Python<'_>,
        src: &Bound<'_, PyAny>,
        tgt: Option<&Bound<'_, PyAny>>,
        symmetrize: Option<&str>,
        direction: Option<&str>,
        threads: usize,
        scores: bool,
        fast: bool,
        sampling: SamplingTuple,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<PyAlignments> {
        let inner = &self.inner;
        let sel = inner.select(symmetrize, direction)?;
        if scores && matches!(sel, Sel::Sym(_)) {
            return Err(PyValueError::new_err(
                "scores belong to one direction: give direction= instead of symmetrize=",
            ));
        }
        if fast && scores {
            return Err(PyValueError::new_err("fast decoding has no scores"));
        }
        let (seed, samplers, length, (i1, i2, i3), exact, accumulate, limit_mb) = sampling;
        if fast && (exact || accumulate) {
            return Err(PyValueError::new_err(
                "fast decoding does not sample: exact and accumulate do not apply",
            ));
        }
        let directions = match sel {
            Sel::Sym(_) => Directions::Both,
            Sel::Dir(Direction::Forward) => Directions::Forward,
            Sel::Dir(Direction::Inverse) => Directions::Inverse,
        };
        let cfg = AlignConfig {
            directions: Some(directions),
            seed,
            samplers,
            length: Some(length),
            iters: [i1, i2, i3],
            scores,
            exact,
            accumulate,
            accumulate_limit: Some(limit_mb << 20),
        };
        let src = text_arg(src, "src")?;
        let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
        let slot = Mutex::new(None);
        let reporter = Reporter::new(progress.as_ref(), &slot);
        let (text, warnings) =
            run_interruptible(py, &slot, || -> PyResult<(Vec<u8>, Vec<String>)> {
                with_threads(threads, || {
                    let bitext = load_bitext(src, tgt, &inner.model.options.text)?;
                    let warnings = corpus_warnings(&bitext);
                    let out = if fast {
                        align_fast_with(&inner.fast, &inner.model, &bitext, Some(directions))
                    } else {
                        core_align(&inner.model, &bitext, &cfg, &|p| reporter.report(p))
                    }
                    .map_err(PyValueError::new_err)?;
                    let text = match sel {
                        Sel::Sym(m) => symmetrize_pharaoh(
                            out.fwd.as_ref().expect("fwd"),
                            out.inv.as_ref().expect("inv"),
                            m,
                        )
                        .map_err(PyValueError::new_err)?,
                        Sel::Dir(_) => out.fwd.or(out.inv).expect("one direction"),
                    };
                    Ok((text, warnings))
                })
            })??;
        for w in warnings {
            // stack level 3: the caller of Model.align_many (or of Model.align)
            let msg = std::ffi::CString::new(w).expect("no NUL in a warning");
            PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), &msg, 3)?;
        }
        Ok(PyAlignments::from_text(text))
    }
}

// ---------------------------------------------------------------------------------------------
// Training

fn read_priors(path: &Path) -> PyResult<PriorSet> {
    let bytes = std::fs::read(path).map_err(|e| io_err(&e, format!("{}: {e}", path.display())))?;
    PriorSet::parse(&bytes).map_err(|e| PyValueError::new_err(format!("{}: {e}", path.display())))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (src, tgt, threads, directions, seed, model, samplers, length, iterations,
                    hyper, priors, text, scores, score_model, exact, accumulate,
                    accumulate_limit, segment="auto", progress=None))]
fn train(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    tgt: Option<&Bound<'_, PyAny>>,
    threads: usize,
    directions: &str,
    seed: u64,
    model: u8,
    samplers: usize,
    length: f64,
    iterations: (Option<usize>, Option<usize>, Option<usize>),
    hyper: (f64, f64, f64, f64),
    priors: Option<PathBuf>,
    text: TextTuple,
    scores: bool,
    score_model: Option<u8>,
    exact: bool,
    accumulate: bool,
    accumulate_limit: u64,
    segment: &str,
    progress: Option<Py<PyAny>>,
) -> PyResult<PyModel> {
    let (lex_alpha, jump_alpha, fert_alpha, null_prior) = hyper;
    let opts = text_options(text)?;
    let segment = Segment::parse(segment).map_err(PyValueError::new_err)?;
    let mut cfg = TrainConfig {
        directions: parse_directions(directions)?,
        seed,
        model,
        samplers,
        hyper: Hyper {
            lex_alpha,
            jump_alpha,
            fert_alpha,
            null_prior,
        },
        length,
        iters: [iterations.0, iterations.1, iterations.2],
        scores,
        exact,
        score_model,
        priors: None,
        keep_model: true,
        accumulate,
        accumulate_limit: accumulate_limit << 20,
    };
    cfg.validate().map_err(PyValueError::new_err)?;
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;

    let slot: Mutex<Option<PyErr>> = Mutex::new(None);
    let reporter = Reporter::new(progress.as_ref(), &slot);
    let out = run_interruptible(
        py,
        &slot,
        || -> PyResult<(Trained, usize, Vec<usize>, Vec<String>)> {
            with_threads(threads, || {
                if let Some(p) = &priors {
                    cfg.priors = Some(Arc::new(read_priors(p)?));
                }
                let bitext = load_parsed(src, tgt)?.build_segmented(&opts, segment);
                let warnings = corpus_warnings(&bitext);
                let trained = core_train(&bitext, &cfg, &|p| reporter.report(p))
                    .map_err(PyValueError::new_err)?;
                Ok((trained, bitext.len(), bitext.skipped().to_vec(), warnings))
            })
        },
    )??;
    let (trained, n_lines, skipped, warnings) = out;
    let Trained {
        fwd,
        inv,
        iterations,
        model,
    } = trained;
    let data = TrainData {
        fwd,
        inv,
        logs: reporter.into_logs(),
        n_lines,
        skipped,
        iterations,
        warnings,
    };
    Ok(PyModel {
        inner: Arc::new(Inner {
            model: model.expect("keep_model"),
            train: Some(data),
            fast: FastModel::default(),
        }),
    })
}

/// eflomal-makepriors: the priors file text for a corpus and its forward and reverse links
/// (Pharaoh text, source-target order). Every pair counts, whatever its length.
#[pyfunction]
fn make_priors<'py>(
    py: Python<'py>,
    src: &Bound<'_, PyAny>,
    tgt: Option<&Bound<'_, PyAny>>,
    fwd: Vec<u8>,
    rev: Vec<u8>,
    text: TextTuple,
    threads: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let opts = TextOptions {
        max_len: NO_MAX_LEN, // every length counts (no sampling)
        ..text_options(text)?
    };
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
    let slot = Mutex::new(None);
    let buf = run_interruptible(py, &slot, || -> PyResult<Vec<u8>> {
        with_threads(threads, || {
            let bitext = load_bitext(src, tgt, &opts)?;
            let ps = core_make_priors(&bitext, &fwd, &rev).map_err(PyValueError::new_err)?;
            let mut buf = Vec::new();
            ps.write(&mut buf)
                .map_err(|e| PyOSError::new_err(e.to_string()))?;
            Ok(buf)
        })
    })??;
    Ok(PyBytes::new(py, &buf))
}

// ---------------------------------------------------------------------------------------------
// estimate

/// The `estimate` report as `(json, text)`, the same strings as `eflomalr estimate [--json]`.
/// With `model`, `src`/`tgt` are text to align with it; otherwise they are the training corpus
/// and `align_src`/`align_tgt` optional text to align later.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (src, tgt, align_src, align_tgt, model, segment, text, threads))]
fn estimate(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    tgt: Option<&Bound<'_, PyAny>>,
    align_src: Option<&Bound<'_, PyAny>>,
    align_tgt: Option<&Bound<'_, PyAny>>,
    model: Option<PyRef<'_, PyModel>>,
    segment: &str,
    text: TextTuple,
    threads: usize,
) -> PyResult<(String, String)> {
    let opts = text_options(text)?;
    let segment = Segment::parse(segment).map_err(PyValueError::new_err)?;
    let src = text_arg(src, "src")?;
    let tgt = tgt.map(|t| text_arg(t, "tgt")).transpose()?;
    let align_src = align_src.map(|t| text_arg(t, "align_src")).transpose()?;
    let align_tgt = align_tgt.map(|t| text_arg(t, "align_tgt")).transpose()?;
    let model = model.map(|m| m.inner.clone());
    let slot = Mutex::new(None);
    let est = run_interruptible(py, &slot, || -> PyResult<Estimate> {
        with_threads(threads, || {
            if let Some(inner) = &model {
                let m = &inner.model;
                let mut est = estimate_model(&m.options.text, m.segmentation.as_ref());
                let parsed = load_parsed(src, tgt)?;
                let new = parsed.build(&m.options.text);
                est.align = Some(align_check(
                    "model",
                    m.segmentation.as_ref().map(|s| &s.stats),
                    &m.source,
                    &m.target,
                    &new,
                    parsed.stats(m.options.text.lowercase),
                ));
                return Ok(est);
            }
            let parsed = load_parsed(src, tgt)?;
            let stats = parsed.stats(opts.lowercase);
            let mut est = core_estimate(&stats, &opts, segment);
            if let Some(a) = align_src {
                let train = parsed.build(&est.text);
                drop(parsed);
                let np = load_parsed(a, align_tgt)?;
                let new = np.build(&est.text);
                est.align = Some(align_check(
                    "training corpus",
                    Some(&stats),
                    &train.source.vocab,
                    &train.target.vocab,
                    &new,
                    np.stats(opts.lowercase),
                ));
            }
            Ok(est)
        })
    })??;
    Ok((est.json(), est.text()))
}

// ---------------------------------------------------------------------------------------------
// Symmetrisation

/// Symmetrise directional alignments pair by pair, as `atools -c METHOD`. `fwd` and `inv` hold
/// links in `(source, target)` orientation.
#[pyfunction]
fn symmetrize_many(
    py: Python<'_>,
    fwd: Vec<Links>,
    inv: Vec<Links>,
    method: &str,
) -> PyResult<Vec<Links>> {
    let method = parse_method(method)?;
    if fwd.len() != inv.len() {
        return Err(PyValueError::new_err(format!(
            "fwd and inv must have the same size ({} and {})",
            fwd.len(),
            inv.len()
        )));
    }
    py.detach(|| {
        fwd.into_iter()
            .zip(inv)
            .enumerate()
            .map(|(k, (mut a, mut b))| {
                a.sort_unstable();
                a.dedup();
                b.sort_unstable();
                b.dedup();
                symmetrize::symmetrize(&a, &b, method).ok_or_else(|| {
                    PyValueError::new_err(format!("pair {k}: the alignment grid is too large"))
                })
            })
            .collect()
    })
}

/// Symmetrise two Pharaoh texts line by line, exactly as `atools -c METHOD -i FWD -j INV` (only
/// `fwd` is used for `invert`).
#[pyfunction]
#[pyo3(signature = (fwd, inv, method))]
fn symmetrize_text<'py>(
    py: Python<'py>,
    fwd: &[u8],
    inv: Option<&[u8]>,
    method: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let method: Method = method.parse().map_err(PyValueError::new_err)?;
    let mut out = Vec::new();
    py.detach(|| {
        symmetrize::symmetrize_text(fwd, inv, method, &mut out).map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Swap the two sides of alignments (atools `invert`).
#[pyfunction]
fn invert_many(fwd: Vec<Links>) -> Vec<Links> {
    fwd.iter().map(|a| symmetrize::invert(a)).collect()
}

/// Parse a Pharaoh alignment line (`"0-0 1-2"`); a ` ||| score` suffix is ignored.
#[pyfunction]
fn parse_pharaoh(line: &str) -> PyResult<Links> {
    let bytes = line.as_bytes();
    let cut = bytes
        .windows(5)
        .position(|w| w == b" ||| ")
        .unwrap_or(bytes.len());
    let mut links = Vec::new();
    for tok in line[..cut].split_whitespace() {
        let bad = || PyValueError::new_err(format!("bad alignment link {tok:?} (expected 'i-j')"));
        let (a, b) = tok.split_once('-').ok_or_else(bad)?;
        links.push((
            a.parse::<u32>().map_err(|_| bad())?,
            b.parse::<u32>().map_err(|_| bad())?,
        ));
    }
    Ok(links)
}

#[pymodule]
mod _core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::estimate;
    #[pymodule_export]
    use super::invert_many;
    #[pymodule_export]
    use super::make_priors;
    #[pymodule_export]
    use super::parse_pharaoh;
    #[pymodule_export]
    use super::symmetrize_many;
    #[pymodule_export]
    use super::symmetrize_text;
    #[pymodule_export]
    use super::train;
    #[pymodule_export]
    use super::AlignmentsIter;
    #[pymodule_export]
    use super::PyAlignments;
    #[pymodule_export]
    use super::PyModel;

    /// Version of the Rust core, from Cargo metadata.
    #[pyfunction]
    fn version() -> &'static str {
        eflomal_core::version()
    }
}
