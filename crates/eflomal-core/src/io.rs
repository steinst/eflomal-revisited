//! Model files and Pharaoh helpers.
//!
//! # Model file format (version 1, little endian)
//!
//! ```text
//! magic    "EFLOMALR"                 8 bytes
//! version  u32                        FORMAT_VERSION
//! payload  u32 n, then n sections     each: tag [u8; 4], u64 length, `length` bytes
//! end      "ENDEFLMR"                 8 bytes (detects truncation)
//! ```
//!
//! A section whose tag starts with an upper-case letter is required: a reader that does not
//! know it refuses the file (it would align wrongly without it). A tag starting with a
//! lower-case letter is optional and skipped by readers that do not know it. New sections can
//! be added this way without a new format version (a tokeniser section, if one is ever added,
//! would be a required `TOKN`). Each section must be read to its exact end.
//!
//! Sections of version 1, in this order:
//!
//! * `OPTS`: `u8` model, `u32` samplers, `f64` α, β, γ, p_ε, `u64` seed, `u8` lowercase, four
//!   times `u8` present + `u64` value (source prefix, source suffix, target prefix, target
//!   suffix), `u64` max_len, `u8` directions (bit 0 fwd, bit 1 inv).
//! * `VOCS`, `VOCT`: the source and target words, `u64` n, then n times `u32` length + bytes;
//!   id 0 is `<NULL>`. Words are normalised (lowercased and stemmed as in training).
//! * `DFWD`, `DINV` (present for each trained direction): the prior counts that the direction
//!   contributes when new text is aligned: `u64` n, n × (`u32` e, `u32` f, `f64` c)
//!   lexical; `u64` n, n × (`i32` m, `f64` c) jumps; `u64` n, n × (`u32` e, `u32` n, `f64` c)
//!   fertilities. Ids are those of `VOCS`/`VOCT` in the direction's orientation (inv: e is a
//!   target word).
//! * `segm` (optional, last; written by training since the `auto` segmentation): how the
//!   stemming in `OPTS` was chosen and the training corpus's statistics
//!   ([`crate::segment::Segmentation`]): `u8` origin (0 words, 1 auto, 2 explicit), `u8`
//!   lowercase, `u64` lines, then source and target `u64` tokens, types, singletons. Only for
//!   reporting (`estimate`); alignment reads the stemming from `OPTS` alone.
//!
//! Shared with fastalign revisited: [`ModelError`], [`Direction`], the bounded reader [`Rd`]
//! (counts from a corrupt header can never make it allocate more than the file holds), the
//! chunked array writers and [`strip_scores`]. The payload is eflomal revisited's own.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::bitext::{TextOptions, MAX_MAX_LEN};
use crate::corpus::Vocab;
use crate::model::Hyper;
use crate::pipeline::Directions;
use crate::priors::{DirTable, PriorSet};
use crate::segment::{CorpusStats, Origin, Segmentation, SideStats};

const MAGIC: &[u8; 8] = b"EFLOMALR";
const END: &[u8; 8] = b"ENDEFLMR";
/// Current model file format version.
pub const FORMAT_VERSION: u32 = 1;

/// Errors reading or writing model files.
#[derive(Debug)]
pub enum ModelError {
    Io(io::Error),
    /// The file is not a (complete, supported) model.
    Format(String),
    /// The model cannot be used as asked.
    Unsupported(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "{e}"),
            ModelError::Format(s) | ModelError::Unsupported(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<io::Error> for ModelError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            ModelError::Format("the model file is truncated".into())
        } else {
            ModelError::Io(e)
        }
    }
}

fn bad<T>(msg: &str) -> Result<T, ModelError> {
    Err(ModelError::Format(msg.into()))
}

/// Which direction of the corpus a model conditions on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// First corpus side generates the second (eflomal's forward links).
    Forward,
    /// Second side generates the first (eflomal's reverse links).
    Inverse,
}

impl Direction {
    pub fn name(self) -> &'static str {
        match self {
            Direction::Forward => "fwd",
            Direction::Inverse => "inv",
        }
    }

    pub fn reverse(self) -> bool {
        self == Direction::Inverse
    }
}

/// Write `v` as little-endian `u32`s, in chunks.
pub fn w_u32s(w: &mut dyn Write, v: &[u32]) -> io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 18);
    for chunk in v.chunks(1 << 16) {
        buf.clear();
        for x in chunk {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        w.write_all(&buf)?;
    }
    Ok(())
}

/// Write the items as little-endian `u64`s, in chunks.
pub fn w_u64s(w: &mut dyn Write, v: impl Iterator<Item = u64>) -> io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 19);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
        if buf.len() >= 1 << 19 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)
}

/// Write `v` as the bit patterns of the `f64`s (exact round trip).
pub fn w_f64s(w: &mut dyn Write, v: &[f64]) -> io::Result<()> {
    w_u64s(w, v.iter().map(|x| x.to_bits()))
}

/// Reader that knows how many bytes are left, so counts from a corrupt header cannot make us
/// allocate more than the file can hold.
pub struct Rd<R: Read> {
    r: R,
    left: u64,
}

impl<R: Read> Rd<R> {
    /// A reader of `len` bytes from `r`.
    pub fn new(r: R, len: u64) -> Self {
        Rd { r, left: len }
    }

    /// Bytes not yet read.
    pub fn left(&self) -> u64 {
        self.left
    }

    pub fn bytes(&mut self, n: usize) -> Result<Vec<u8>, ModelError> {
        if n as u64 > self.left {
            return bad("the model file is truncated or corrupt");
        }
        let mut v = vec![0u8; n];
        self.r.read_exact(&mut v)?;
        self.left -= n as u64;
        Ok(v)
    }

    pub fn arr<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let v = self.bytes(N)?;
        Ok(v.try_into().unwrap())
    }

    pub fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.arr::<1>()?[0])
    }

    pub fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    pub fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    pub fn f64(&mut self) -> Result<f64, ModelError> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// `n` elements of `size` bytes each, read in chunks and converted by `f`.
    pub fn vec<T>(
        &mut self,
        n: u64,
        size: u64,
        f: impl Fn(&[u8]) -> T,
    ) -> Result<Vec<T>, ModelError> {
        if n.checked_mul(size).is_none_or(|b| b > self.left) {
            return bad("the model file is truncated or corrupt");
        }
        let n = n as usize;
        let size = size as usize;
        let mut out = Vec::with_capacity(n);
        let mut buf = vec![0u8; size << 16];
        let mut todo = n;
        while todo > 0 {
            let k = todo.min(1 << 16);
            self.r.read_exact(&mut buf[..k * size])?;
            out.extend(buf[..k * size].chunks_exact(size).map(&f));
            todo -= k;
        }
        self.left -= (n * size) as u64;
        Ok(out)
    }
}

/// Write a framed model file: magic, version, `payload`, end marker.
pub fn write_framed(
    w: &mut dyn Write,
    payload: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    payload(w)?;
    w.write_all(END)
}

/// Read a framed model of `len` bytes from `r`: checks magic and version, lets `payload` read
/// the body, then checks the end marker and that nothing follows it. Corrupt or truncated input
/// gives [`ModelError::Format`], never a panic.
pub fn read_framed<R: Read, T>(
    r: R,
    len: u64,
    payload: impl FnOnce(&mut Rd<R>) -> Result<T, ModelError>,
) -> Result<T, ModelError> {
    let mut r = Rd::new(r, len);
    if r.arr::<8>()
        .map_err(|_| ModelError::Format("not an eflomalr model file".into()))?
        != *MAGIC
    {
        return bad("not an eflomalr model file");
    }
    let version = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(ModelError::Unsupported(format!(
            "model file version {version} is not supported (this build reads version {FORMAT_VERSION})"
        )));
    }
    let out = payload(&mut r)?;
    if r.arr::<8>()? != *END || r.left != 0 {
        return bad("the model file is corrupt (end marker)");
    }
    Ok(out)
}

/// Save a framed model to a file (atomically, see [`write_atomic`]).
pub fn save_framed(
    path: &Path,
    payload: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> Result<(), ModelError> {
    write_atomic(path, |w| write_framed(w, payload).map_err(ModelError::from))
}

/// Write a file through a temporary file in the same directory, renamed over `path` only after
/// the whole content is written, flushed and synced.
///
/// The temporary file has a name no other writer uses (process id, a per-process counter and the
/// clock) and is created exclusively (`create_new`), so an unrelated file can never be truncated
/// or removed, and two simultaneous saves to the same path never share it: the destination ends
/// up as one complete file. If writing fails, the temporary file is removed and an existing
/// destination keeps its bytes.
pub fn write_atomic<E: From<io::Error>>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<(), E>,
) -> Result<(), E> {
    let (tmp, file) = create_temp_sibling(path).map_err(|e| E::from(with_path_io(e, path)))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let done = write(&mut w).and_then(|()| {
        let f = w.into_inner().map_err(|e| E::from(e.into_error()))?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path).map_err(|e| E::from(with_path_io(e, path)))
    });
    if done.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    done
}

/// A new, exclusively created file next to `path` whose name no other writer uses.
fn create_temp_sibling(path: &Path) -> io::Result<(std::path::PathBuf, File)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path has no file name"))?;
    loop {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let mut tmp_name = std::ffi::OsString::from(".");
        tmp_name.push(name);
        tmp_name.push(format!(
            ".{}-{}-{nanos:08x}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let tmp = dir.join(tmp_name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(f) => return Ok((tmp, f)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

fn with_path_io(e: io::Error, path: &Path) -> io::Error {
    io::Error::new(e.kind(), format!("{}: {e}", path.display()))
}

/// Load a framed model from a file.
pub fn load_framed<T>(
    path: &Path,
    payload: impl FnOnce(&mut Rd<BufReader<File>>) -> Result<T, ModelError>,
) -> Result<T, ModelError> {
    let f = File::open(path)?;
    let len = f.metadata()?.len();
    read_framed(BufReader::with_capacity(1 << 20, f), len, payload)
}

// ------------------------------------------------------------------------------------------
// The saved model

/// The options a saved model was trained with (and that `align` reuses).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOptions {
    pub model: u8,
    pub samplers: usize,
    pub hyper: Hyper,
    pub seed: u64,
    /// Lowercasing, stemming and max_len of the training corpus (`chunks` is not stored).
    pub text: TextOptions,
}

/// A trained model: options, both sides' words and, per trained direction, the
/// prior counts that aligning new text with it uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub options: ModelOptions,
    pub source: Vocab,
    pub target: Vocab,
    pub fwd: Option<DirTable>,
    pub inv: Option<DirTable>,
    /// How the stemming was chosen, with the training corpus's statistics (`None` for a model
    /// saved before this was recorded).
    pub segmentation: Option<Segmentation>,
}

fn w_u8(w: &mut dyn Write, x: u8) -> io::Result<()> {
    w.write_all(&[x])
}
fn w_u32(w: &mut dyn Write, x: u32) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_u64(w: &mut dyn Write, x: u64) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_f64(w: &mut dyn Write, x: f64) -> io::Result<()> {
    w.write_all(&x.to_bits().to_le_bytes())
}

fn w_vocab(w: &mut dyn Write, v: &Vocab) -> io::Result<()> {
    w_u64(w, v.len() as u64)?;
    for id in 0..v.len() as u32 {
        let word = v.word(id);
        w_u32(w, word.len() as u32)?;
        w.write_all(word)?;
    }
    Ok(())
}

fn w_table(w: &mut dyn Write, t: &DirTable) -> io::Result<()> {
    let mut buf = Vec::with_capacity(16 * t.lex.len() + 32);
    buf.extend_from_slice(&(t.lex.len() as u64).to_le_bytes());
    for &(e, f, c) in &t.lex {
        buf.extend_from_slice(&e.to_le_bytes());
        buf.extend_from_slice(&f.to_le_bytes());
        buf.extend_from_slice(&c.to_bits().to_le_bytes());
    }
    buf.extend_from_slice(&(t.jump.len() as u64).to_le_bytes());
    for (&m, &c) in &t.jump {
        buf.extend_from_slice(&m.to_le_bytes());
        buf.extend_from_slice(&c.to_bits().to_le_bytes());
    }
    buf.extend_from_slice(&(t.fert.len() as u64).to_le_bytes());
    for &(e, n, c) in &t.fert {
        buf.extend_from_slice(&e.to_le_bytes());
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&c.to_bits().to_le_bytes());
    }
    w.write_all(&buf)
}

fn r_vocab<R: Read>(r: &mut Rd<R>) -> Result<Vocab, ModelError> {
    let n = r.u64()?;
    // every word takes at least 4 bytes
    if n.checked_mul(4).is_none_or(|b| b > r.left()) || n == 0 {
        return bad("the model file is corrupt (vocabulary)");
    }
    let mut words = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let len = r.u32()?;
        words.push(r.bytes(len as usize)?);
    }
    Ok(Vocab::from_words(words.iter().map(|w| w.as_slice())))
}

fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().unwrap())
}
fn le_f64(b: &[u8]) -> f64 {
    f64::from_bits(u64::from_le_bytes(b.try_into().unwrap()))
}

fn r_table<R: Read>(r: &mut Rd<R>) -> Result<DirTable, ModelError> {
    let n = r.u64()?;
    let lex = r.vec(n, 16, |b| {
        (le_u32(&b[..4]), le_u32(&b[4..8]), le_f64(&b[8..]))
    })?;
    let n = r.u64()?;
    let jump = r.vec(n, 12, |b| (le_u32(&b[..4]) as i32, le_f64(&b[4..])))?;
    let n = r.u64()?;
    let fert = r.vec(n, 16, |b| {
        (le_u32(&b[..4]), le_u32(&b[4..8]), le_f64(&b[8..]))
    })?;
    let sorted = |v: &[(u32, u32, f64)]| v.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1));
    if !sorted(&lex) || !sorted(&fert) || !jump.windows(2).all(|w| w[0].0 < w[1].0) {
        return bad("the model file is corrupt (prior table order)");
    }
    Ok(DirTable {
        lex,
        jump: jump.into_iter().collect(),
        fert,
    })
}

/// Write one section: tag, length, body.
fn w_section(
    w: &mut dyn Write,
    tag: &[u8; 4],
    body: impl FnOnce(&mut Vec<u8>) -> io::Result<()>,
) -> io::Result<()> {
    let mut buf = Vec::new();
    body(&mut buf)?;
    w.write_all(tag)?;
    w_u64(w, buf.len() as u64)?;
    w.write_all(&buf)
}

impl Model {
    /// Directions this model has.
    pub fn directions(&self) -> Option<Directions> {
        match (self.fwd.is_some(), self.inv.is_some()) {
            (true, true) => Some(Directions::Both),
            (true, false) => Some(Directions::Forward),
            (false, true) => Some(Directions::Inverse),
            (false, false) => None,
        }
    }

    pub fn table(&self, d: Direction) -> Option<&DirTable> {
        match d {
            Direction::Forward => self.fwd.as_ref(),
            Direction::Inverse => self.inv.as_ref(),
        }
    }

    /// The model's tables as a prior set (a missing direction has an empty table).
    pub fn prior_set(&self) -> PriorSet {
        PriorSet {
            source: self.source.clone(),
            target: self.target.clone(),
            fwd: self.fwd.clone().unwrap_or_default(),
            inv: self.inv.clone().unwrap_or_default(),
        }
    }

    pub fn write(&self, w: &mut dyn Write) -> io::Result<()> {
        write_framed(w, |w| {
            let n = 3
                + u32::from(self.fwd.is_some())
                + u32::from(self.inv.is_some())
                + u32::from(self.segmentation.is_some());
            w_u32(w, n)?;
            w_section(w, b"OPTS", |b| {
                let o = &self.options;
                w_u8(b, o.model)?;
                w_u32(b, o.samplers as u32)?;
                for x in [
                    o.hyper.lex_alpha,
                    o.hyper.jump_alpha,
                    o.hyper.fert_alpha,
                    o.hyper.null_prior,
                ] {
                    w_f64(b, x)?;
                }
                w_u64(b, o.seed)?;
                w_u8(b, u8::from(o.text.lowercase))?;
                for x in [
                    o.text.source_prefix,
                    o.text.source_suffix,
                    o.text.target_prefix,
                    o.text.target_suffix,
                ] {
                    w_u8(b, u8::from(x.is_some()))?;
                    w_u64(b, x.unwrap_or(0) as u64)?;
                }
                w_u64(b, o.text.max_len as u64)?;
                w_u8(
                    b,
                    u8::from(self.fwd.is_some()) | (u8::from(self.inv.is_some()) << 1),
                )
            })?;
            w_section(w, b"VOCS", |b| w_vocab(b, &self.source))?;
            w_section(w, b"VOCT", |b| w_vocab(b, &self.target))?;
            if let Some(t) = &self.fwd {
                w_section(w, b"DFWD", |b| w_table(b, t))?;
            }
            if let Some(t) = &self.inv {
                w_section(w, b"DINV", |b| w_table(b, t))?;
            }
            if let Some(s) = &self.segmentation {
                w_section(w, b"segm", |b| {
                    w_u8(b, s.origin.code())?;
                    w_u8(b, u8::from(s.stats.lowercase))?;
                    w_u64(b, s.stats.lines)?;
                    for side in [&s.stats.source, &s.stats.target] {
                        w_u64(b, side.tokens)?;
                        w_u64(b, side.types)?;
                        w_u64(b, side.singletons)?;
                    }
                    Ok(())
                })?;
            }
            Ok(())
        })
    }

    /// Read a model of `len` bytes. Corrupt or truncated input is an error, never a panic.
    pub fn read<R: Read>(r: R, len: u64) -> Result<Self, ModelError> {
        read_framed(r, len, |r| {
            let n = r.u32()?;
            let mut opts: Option<(ModelOptions, u8)> = None;
            let (mut src, mut tgt, mut fwd, mut inv) = (None, None, None, None);
            let mut segm = None;
            for _ in 0..n {
                let tag = r.arr::<4>()?;
                let size = r.u64()?;
                if size > r.left() {
                    return bad("the model file is truncated or corrupt");
                }
                let body = r.bytes(size as usize)?;
                let mut b = Rd::new(&body[..], size);
                let seen = match &tag {
                    b"OPTS" => opts.is_some(),
                    b"VOCS" => src.is_some(),
                    b"VOCT" => tgt.is_some(),
                    b"DFWD" => fwd.is_some(),
                    b"DINV" => inv.is_some(),
                    b"segm" => segm.is_some(),
                    _ => false,
                };
                if seen {
                    return bad("the model file is corrupt (repeated section)");
                }
                match &tag {
                    b"OPTS" => opts = Some(r_options(&mut b)?),
                    b"VOCS" => src = Some(r_vocab(&mut b)?),
                    b"VOCT" => tgt = Some(r_vocab(&mut b)?),
                    b"DFWD" => fwd = Some(r_table(&mut b)?),
                    b"DINV" => inv = Some(r_table(&mut b)?),
                    b"segm" => segm = Some(r_segmentation(&mut b)?),
                    t if t[0].is_ascii_lowercase() => continue,
                    t => {
                        return Err(ModelError::Unsupported(format!(
                            "the model file has a section `{}` that this build does not know \
                             (made by a newer version?)",
                            String::from_utf8_lossy(t)
                        )))
                    }
                }
                if b.left() != 0 {
                    return bad("the model file is corrupt (section length)");
                }
            }
            let (Some((options, dirs)), Some(source), Some(target)) = (opts, src, tgt) else {
                return bad("the model file is corrupt (missing section)");
            };
            if dirs & 1 != u8::from(fwd.is_some()) || (dirs >> 1) != u8::from(inv.is_some()) {
                return bad("the model file is corrupt (directions)");
            }
            if fwd.is_none() && inv.is_none() {
                return bad("the model file has no direction");
            }
            for (t, a, b) in [(&fwd, &source, &target), (&inv, &target, &source)] {
                if let Some(t) = t {
                    t.validate(a.len(), b.len()).map_err(|e| {
                        ModelError::Format(format!("the model file is corrupt: {e}"))
                    })?;
                }
            }
            Ok(Model {
                options,
                source,
                target,
                fwd,
                inv,
                segmentation: segm,
            })
        })
    }

    /// Write the model to `path` atomically ([`write_atomic`]): the destination is replaced
    /// only by a complete file, and no other file is touched.
    pub fn save(&self, path: &Path) -> Result<(), ModelError> {
        write_atomic(path, |w| self.write(w).map_err(ModelError::from))
    }

    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let f = File::open(path).map_err(|e| with_path(e, path))?;
        let len = f.metadata()?.len();
        Self::read(BufReader::with_capacity(1 << 20, f), len)
    }

    /// Approximate size in memory.
    pub fn memory_bytes(&self) -> usize {
        let t = |t: &Option<DirTable>| {
            t.as_ref()
                .map_or(0, |t| 16 * (t.lex.len() + t.fert.len()) + 40 * t.jump.len())
        };
        self.source.memory_bytes() + self.target.memory_bytes() + t(&self.fwd) + t(&self.inv)
    }
}

fn with_path(e: io::Error, path: &Path) -> ModelError {
    ModelError::Io(with_path_io(e, path))
}

fn r_options<R: Read>(r: &mut Rd<R>) -> Result<(ModelOptions, u8), ModelError> {
    let model = r.u8()?;
    let samplers = r.u32()? as usize;
    let hyper = Hyper {
        lex_alpha: r.f64()?,
        jump_alpha: r.f64()?,
        fert_alpha: r.f64()?,
        null_prior: r.f64()?,
    };
    let seed = r.u64()?;
    let lowercase = match r.u8()? {
        0 => false,
        1 => true,
        _ => return bad("the model file is corrupt (options)"),
    };
    let mut stems = [None; 4];
    for s in &mut stems {
        let present = r.u8()?;
        let v = r.u64()?;
        *s = match present {
            0 => None,
            1 => Some(usize::try_from(v).map_err(|_| ModelError::Format("bad stem".into()))?),
            _ => return bad("the model file is corrupt (options)"),
        };
    }
    let max_len = r.u64()?;
    let dirs = r.u8()?;
    if !(1..=3).contains(&model)
        || !(1..=255).contains(&samplers)
        || hyper.validate().is_err()
        || !(1..=MAX_MAX_LEN as u64).contains(&max_len)
        || dirs > 3
    {
        return bad("the model file is corrupt (options)");
    }
    Ok((
        ModelOptions {
            model,
            samplers,
            hyper,
            seed,
            text: TextOptions {
                lowercase,
                source_prefix: stems[0],
                source_suffix: stems[1],
                target_prefix: stems[2],
                target_suffix: stems[3],
                max_len: max_len as usize,
                chunks: None,
            },
        },
        dirs,
    ))
}

fn r_segmentation<R: Read>(r: &mut Rd<R>) -> Result<Segmentation, ModelError> {
    let origin = Origin::from_code(r.u8()?);
    let lowercase = r.u8()?;
    let lines = r.u64()?;
    let mut sides = [SideStats::default(); 2];
    for s in &mut sides {
        *s = SideStats {
            tokens: r.u64()?,
            types: r.u64()?,
            singletons: r.u64()?,
        };
    }
    let (Some(origin), true) = (origin, lowercase <= 1) else {
        return bad("the model file is corrupt (segmentation)");
    };
    if sides
        .iter()
        .any(|s| s.types > s.tokens || s.singletons > s.types)
    {
        return bad("the model file is corrupt (segmentation)");
    }
    Ok(Segmentation {
        origin,
        stats: CorpusStats {
            lines,
            lowercase: lowercase == 1,
            source: sides[0],
            target: sides[1],
        },
    })
}

/// Remove ` ||| score` suffixes from Pharaoh lines (output made with scores), leaving links.
pub fn strip_scores(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for line in text.split_inclusive(|&b| b == b'\n') {
        let body = line.strip_suffix(b"\n").unwrap_or(line);
        let cut = body
            .windows(5)
            .position(|w| w == b" ||| ")
            .unwrap_or(body.len());
        out.extend_from_slice(&body[..cut]);
        if line.ends_with(b"\n") {
            out.push(b'\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy payload: u64 count, then that many u32s and f64s.
    fn write_toy(w: &mut dyn Write, xs: &[u32], ys: &[f64]) -> io::Result<()> {
        w.write_all(&(xs.len() as u64).to_le_bytes())?;
        w_u32s(w, xs)?;
        w_f64s(w, ys)
    }

    fn read_toy<R: Read>(r: &mut Rd<R>) -> Result<(Vec<u32>, Vec<f64>), ModelError> {
        let n = r.u64()?;
        let xs = r.vec(n, 4, |b| u32::from_le_bytes(b.try_into().unwrap()))?;
        let ys = r.vec(n, 8, |b| {
            f64::from_bits(u64::from_le_bytes(b.try_into().unwrap()))
        })?;
        Ok((xs, ys))
    }

    #[test]
    fn framed_roundtrip_and_corruption() {
        let xs = [1u32, 7, u32::MAX];
        let ys = [0.5f64, -1e-300, f64::INFINITY];
        let mut bytes = Vec::new();
        write_framed(&mut bytes, |w| write_toy(w, &xs, &ys)).unwrap();
        let (a, b) = read_framed(&bytes[..], bytes.len() as u64, read_toy).unwrap();
        assert_eq!(a, xs);
        assert_eq!(b, ys);
        // every truncation is an error; every single-byte corruption an error or a value, never a panic
        for cut in 0..bytes.len() {
            assert!(read_framed(&bytes[..cut], cut as u64, read_toy).is_err());
        }
        for k in 0..bytes.len() {
            let mut c = bytes.clone();
            c[k] ^= 0xff;
            let _ = read_framed(&c[..], c.len() as u64, read_toy);
        }
        let mut c = bytes.clone();
        c.extend_from_slice(b"x");
        assert!(read_framed(&c[..], c.len() as u64, read_toy).is_err());
        let mut c = bytes.clone();
        c[8] = 99;
        assert!(matches!(
            read_framed(&c[..], c.len() as u64, read_toy),
            Err(ModelError::Unsupported(_))
        ));
    }

    #[test]
    fn save_load_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.eflomalr");
        save_framed(&p, |w| write_toy(w, &[3, 4], &[1.0, 2.0])).unwrap();
        let (a, b) = load_framed(&p, read_toy).unwrap();
        assert_eq!((a, b), (vec![3, 4], vec![1.0, 2.0]));
    }

    fn toy_model() -> Model {
        use crate::priors::DirTable;
        let v = |w: &[&str]| Vocab::from_words(w.iter().map(|x| x.as_bytes()));
        Model {
            options: ModelOptions {
                model: 2,
                samplers: 4,
                hyper: Hyper {
                    lex_alpha: 0.002,
                    ..Hyper::default()
                },
                seed: 99,
                text: TextOptions {
                    lowercase: false,
                    source_prefix: Some(4),
                    target_suffix: Some(2),
                    max_len: 300,
                    ..TextOptions::default()
                },
            },
            source: v(&["<NULL>", "a", "b"]),
            target: v(&["<NULL>", "x", "y", "z"]),
            fwd: Some(DirTable {
                lex: vec![(0, 3, 1.0), (1, 1, 2.5), (2, 2, 7.0)],
                jump: [(-200, 1.0), (1, 9.0)].into_iter().collect(),
                fert: vec![(1, 1, 3.0), (2, 40, 1.0)],
            }),
            inv: Some(DirTable {
                lex: vec![(1, 1, 2.5)],
                ..DirTable::default()
            }),
            segmentation: Some(Segmentation {
                origin: Origin::Explicit,
                stats: CorpusStats {
                    lines: 12,
                    lowercase: false,
                    source: SideStats {
                        tokens: 40,
                        types: 9,
                        singletons: 3,
                    },
                    target: SideStats {
                        tokens: 41,
                        types: 10,
                        singletons: 4,
                    },
                },
            }),
        }
    }

    #[test]
    fn model_roundtrip_and_corruption() {
        let m = toy_model();
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        let back = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(back, m);
        for cut in 0..bytes.len() {
            assert!(Model::read(&bytes[..cut], cut as u64).is_err(), "cut {cut}");
        }
        for k in 0..bytes.len() {
            for x in [0x01u8, 0x80, 0xff] {
                let mut c = bytes.clone();
                c[k] ^= x;
                let _ = Model::read(&c[..], c.len() as u64); // never panics
            }
        }
        // one direction only
        let one = Model {
            inv: None,
            ..m.clone()
        };
        let mut b1 = Vec::new();
        one.write(&mut b1).unwrap();
        let back = Model::read(&b1[..], b1.len() as u64).unwrap();
        assert_eq!(back.directions(), Some(Directions::Forward));
        // a model without the (optional) segmentation record, as saved before it existed
        let old = Model {
            segmentation: None,
            ..m.clone()
        };
        let mut b2 = Vec::new();
        old.write(&mut b2).unwrap();
        assert_eq!(Model::read(&b2[..], b2.len() as u64).unwrap(), old);
        assert_eq!(b2.len() + 4 + 8 + 2 + 7 * 8, bytes.len());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.eflomalr");
        m.save(&p).unwrap();
        assert_eq!(Model::load(&p).unwrap(), m);
    }

    /// Splice an extra section before the end marker (and count it).
    fn with_section(bytes: &[u8], tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut c = bytes[..bytes.len() - 8].to_vec();
        let n = u32::from_le_bytes(c[12..16].try_into().unwrap()) + 1;
        c[12..16].copy_from_slice(&n.to_le_bytes());
        c.extend_from_slice(tag);
        c.extend_from_slice(&(body.len() as u64).to_le_bytes());
        c.extend_from_slice(body);
        c.extend_from_slice(END);
        c
    }

    #[test]
    fn unknown_sections() {
        let m = toy_model();
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        // an optional (lower-case) section is skipped
        let c = with_section(&bytes, b"note", b"anything");
        assert_eq!(Model::read(&c[..], c.len() as u64).unwrap(), m);
        // a required one is refused cleanly
        let c = with_section(&bytes, b"TOKN", b"x");
        assert!(matches!(
            Model::read(&c[..], c.len() as u64),
            Err(ModelError::Unsupported(_))
        ));
        // a repeated known section is corrupt
        let c = with_section(&bytes, b"VOCS", &[0; 8]);
        assert!(matches!(
            Model::read(&c[..], c.len() as u64),
            Err(ModelError::Format(_))
        ));
    }

    fn dir_listing(d: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(d)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    /// Saving never touches a file it does not own, a failed write keeps the
    /// existing destination, and simultaneous saves leave one complete model.
    #[test]
    fn save_is_atomic_and_owns_only_its_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m");
        // unrelated siblings, including the name the Python binding used to write through
        for (name, body) in [("m.tmp", "unrelated"), (".m.tmp", "also unrelated")] {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        let m = toy_model();
        m.save(&p).unwrap();
        assert_eq!(Model::load(&p).unwrap(), m);
        assert_eq!(
            std::fs::read(dir.path().join("m.tmp")).unwrap(),
            b"unrelated"
        );
        assert_eq!(
            std::fs::read(dir.path().join(".m.tmp")).unwrap(),
            b"also unrelated"
        );
        assert_eq!(dir_listing(dir.path()), [".m.tmp", "m", "m.tmp"]);

        // an injected failure after some bytes: the old destination is intact, no temp file
        let before = std::fs::read(&p).unwrap();
        let r = write_atomic(&p, |w| -> Result<(), ModelError> {
            w.write_all(b"EFLOMALR partial")?;
            w.flush()?;
            Err(ModelError::Io(io::Error::other("injected")))
        });
        assert!(r.is_err());
        assert_eq!(std::fs::read(&p).unwrap(), before);
        assert_eq!(dir_listing(dir.path()), [".m.tmp", "m", "m.tmp"]);

        // two distinguishable models saved at the same time to the same path
        let other = Model {
            inv: None,
            ..m.clone()
        };
        for _ in 0..20 {
            std::thread::scope(|s| {
                for model in [&m, &other] {
                    let p = &p;
                    s.spawn(move || model.save(p).unwrap());
                }
            });
            let got = Model::load(&p).unwrap();
            assert!(got == m || got == other);
        }
        assert_eq!(dir_listing(dir.path()), [".m.tmp", "m", "m.tmp"]);

        // a destination that cannot be replaced (a directory): error, nothing left behind
        let d = dir.path().join("sub");
        std::fs::create_dir(&d).unwrap();
        assert!(m.save(&d).is_err());
        assert_eq!(dir_listing(dir.path()), [".m.tmp", "m", "m.tmp", "sub"]);
        // a relative path without a directory part (the temporary file goes in ".")
        let rel = Path::new("eflomalr-io-test-relative.model");
        m.save(rel).unwrap();
        assert_eq!(Model::load(rel).unwrap(), m);
        std::fs::remove_file(rel).unwrap();
    }

    #[test]
    fn strip_scores_keeps_links() {
        assert_eq!(
            strip_scores(b"0-0 1-1 ||| -3.5\n ||| -2\n0-1\n"),
            b"0-0 1-1\n\n0-1\n"
        );
    }
}
