//! The aligner's view of a corpus: separate source and target vocabularies, normalised tokens,
//! flat token arrays, and a per-direction view without copying.
//!
//! Built on top of the reader in [`crate::corpus`], which parses lines, splits tokens and interns them into one shared dictionary in
//! parallel. This module then
//!
//! 1. sanitises the input bytes where eflomal's tokenisation differs from fast_align's (only if
//!    such bytes occur, so normally without a copy): CR, VT and FF separate tokens; with two
//!    files a tab separates tokens too, and a `|||` token in the source file is an ordinary
//!    word (in the shared reader both would split the line);
//! 2. normalises every word of the shared dictionary (lowercasing, prefix/suffix stemming);
//! 3. drops the tokens of pairs with more than `max_len` tokens on either side (the pair stays
//!    as an empty line, so line numbers and K are unchanged, and is listed in
//!    [`Bitext::skipped`]);
//! 4. interns the normalised words of each side separately: id 0 is NULL (`<NULL>`, the
//!    spelling of eflomal's priors files), words get ids 1, 2, … in order of first occurrence on
//!    that side (line order, left to right).
//!
//! Every step is a function of the input bytes and options only, so ids and token arrays are
//! identical at any thread count and chunking.

use std::borrow::Cow;
use std::fs::File;
use std::path::Path;

use memchr::memmem;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::corpus::{Corpus, CorpusError, LoadOptions, Vocab};
use crate::io::Direction;
use crate::segment::{choose, CorpusStats, Segment, Segmentation, SideStats};

/// Spelling of the NULL word (id 0 of each side's vocabulary).
pub const NULL_WORD: &[u8] = b"<NULL>";
/// Default `max_len`: eflomal skips pairs with 1024 or more tokens on a side.
pub const DEFAULT_MAX_LEN: usize = 1023;
/// Largest supported `max_len`: alignments are `u16` with `0xFFFF` = NULL.
pub const MAX_MAX_LEN: usize = 65_534;
/// `max_len` of a bitext that is never sampled (makepriors): no pair is skipped, at any length
/// (links are counted in `u32` positions, independently of the sampler's `u16` alignments).
pub const NO_MAX_LEN: usize = usize::MAX;
const UNSET: u32 = u32::MAX;
/// Stand-in for a `|||` token in a source file read as one of two files; mapped back to `|||`
/// when the corpus's words are read ([`unescape`]). A token that itself starts with NUL is
/// escaped with one more NUL, so the stand-in can never collide with a real token.
const PIPES_STANDIN: &[u8] = b"\x00|||";

/// A word of the sanitised text as it was in the input: one leading NUL removed (the escape of
/// [`sanitize`]; the stand-in becomes `|||`).
fn unescape(w: &[u8]) -> &[u8] {
    if w.first() == Some(&0) {
        &w[1..]
    } else {
        w
    }
}

/// Normalisation and filtering options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextOptions {
    /// Lowercase tokens (Unicode for valid UTF-8 tokens, ASCII otherwise). Default true.
    pub lowercase: bool,
    /// Keep only the first N characters of each source token (`--source-prefix`).
    pub source_prefix: Option<usize>,
    /// Keep only the last N characters of each source token (`--source-suffix`), applied
    /// after the prefix.
    pub source_suffix: Option<usize>,
    pub target_prefix: Option<usize>,
    pub target_suffix: Option<usize>,
    /// Pairs with more tokens than this on either side are skipped (empty output line).
    pub max_len: usize,
    /// Chunks for the parallel reader (default: from the thread count); results do not
    /// depend on it.
    pub chunks: Option<usize>,
}

impl TextOptions {
    /// The options with stem lengths of 0 removed: `--source-prefix 0` (and the other three)
    /// means "do not stem that side", as in eflomal. Such an option still counts as explicit
    /// for `--segment auto` ([`crate::segment::choose`]); the vocabularies and a saved model
    /// only ever see the normalised options.
    pub fn without_zero_stems(&self) -> TextOptions {
        let nz = |v: Option<usize>| v.filter(|&n| n > 0);
        TextOptions {
            source_prefix: nz(self.source_prefix),
            source_suffix: nz(self.source_suffix),
            target_prefix: nz(self.target_prefix),
            target_suffix: nz(self.target_suffix),
            ..self.clone()
        }
    }
}

impl Default for TextOptions {
    fn default() -> Self {
        TextOptions {
            lowercase: true,
            source_prefix: None,
            source_suffix: None,
            target_prefix: None,
            target_suffix: None,
            max_len: DEFAULT_MAX_LEN,
            chunks: None,
        }
    }
}

/// One side of the corpus with its own vocabulary.
#[derive(Clone, Debug)]
pub struct SideText {
    /// Id → normalised word; id 0 is [`NULL_WORD`].
    pub vocab: Vocab,
    tokens: Vec<u32>,
    offsets: Vec<usize>,
}

impl SideText {
    /// Token ids of line `k` (empty for a skipped line).
    #[inline]
    pub fn line(&self, k: usize) -> &[u32] {
        &self.tokens[self.offsets[k]..self.offsets[k + 1]]
    }

    #[inline]
    pub fn line_len(&self, k: usize) -> usize {
        self.offsets[k + 1] - self.offsets[k]
    }

    /// All tokens, line after line.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Line offsets into [`SideText::tokens`] (number of lines + 1).
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Vocabulary size including NULL.
    pub fn n_types(&self) -> usize {
        self.vocab.len()
    }

    /// Longest line in tokens.
    pub fn max_line_len(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    pub fn memory_bytes(&self) -> usize {
        self.vocab.memory_bytes()
            + self.tokens.capacity() * 4
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

/// A corpus ready for alignment.
#[derive(Clone, Debug)]
pub struct Bitext {
    /// The first side (source files / before `|||`).
    pub source: SideText,
    /// The second side.
    pub target: SideText,
    /// 0-based indices of the pairs skipped for length (their lines are empty).
    skipped: Vec<usize>,
    options: TextOptions,
    /// How the stemming was chosen and the statistics it was chosen from (set by
    /// [`Parsed::build_segmented`], the training path; `None` otherwise).
    segmentation: Option<Segmentation>,
}

/// A corpus parsed by the shared reader but not yet normalised: the first phase of reading,
/// so that the corpus statistics ([`Parsed::stats`]) can choose the stemming
/// ([`crate::segment`]) before the vocabularies are built.
pub struct Parsed {
    corpus: Corpus,
}

impl Parsed {
    /// Parse two line-aligned files (`chunks` as [`TextOptions::chunks`]).
    pub fn read_pair(src: &Path, tgt: &Path, chunks: Option<usize>) -> Result<Self, CorpusError> {
        let a = Mapped::open(src)?;
        let b = Mapped::open(tgt)?;
        Self::from_pair_bytes(a.bytes(), b.bytes(), chunks)
    }

    /// Parse one `SRC ||| TGT` file.
    pub fn read_joined(path: &Path, chunks: Option<usize>) -> Result<Self, CorpusError> {
        let a = Mapped::open(path)?;
        Ok(Self::from_joined_bytes(a.bytes(), chunks))
    }

    /// Parse two in-memory line-aligned texts.
    pub fn from_pair_bytes(
        src: &[u8],
        tgt: &[u8],
        chunks: Option<usize>,
    ) -> Result<Self, CorpusError> {
        let (s, t) = rayon::join(|| sanitize(src, true, true), || sanitize(tgt, true, false));
        let corpus = Corpus::from_pair_bytes(&s, &t, &load_options(chunks))?;
        Ok(Parsed { corpus })
    }

    /// Parse an in-memory `SRC ||| TGT` text (shared reader rules: the first `|||` or tab
    /// separates the sides).
    pub fn from_joined_bytes(text: &[u8], chunks: Option<usize>) -> Self {
        let s = sanitize(text, false, false);
        Parsed {
            corpus: Corpus::from_joined_bytes(&s, &load_options(chunks)),
        }
    }

    /// Number of line pairs K (empty lines and lines later skipped for length included).
    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corpus.is_empty()
    }

    /// Per-side statistics of the whole corpus (every line, also those a `max_len` would
    /// skip), over lowercased tokens if `lowercase`, as they were measured for the `auto` rule.
    pub fn stats(&self, lowercase: bool) -> CorpusStats {
        let c = &self.corpus;
        let (source, target) = rayon::join(
            || side_stats(&c.vocab, c.first.tokens(), lowercase),
            || side_stats(&c.vocab, c.second.tokens(), lowercase),
        );
        CorpusStats {
            lines: c.len() as u64,
            lowercase,
            source,
            target,
        }
    }

    /// Normalise with `opts` exactly as given (no `auto`).
    pub fn build(&self, opts: &TextOptions) -> Bitext {
        Bitext::from_corpus(&self.corpus, opts)
    }

    /// The training path: compute the statistics, let `segment` choose the stemming (explicit
    /// stemming options in `opts` win, see [`crate::segment::choose`]) and normalise with the
    /// result. The returned bitext remembers the choice ([`Bitext::segmentation`]).
    pub fn build_segmented(&self, opts: &TextOptions, segment: Segment) -> Bitext {
        let stats = self.stats(opts.lowercase);
        let (text, origin) = choose(segment, opts, stats.lines);
        let mut b = Bitext::from_corpus(&self.corpus, &text);
        b.segmentation = Some(Segmentation { origin, stats });
        b
    }
}

/// Token, type and singleton counts of one side (types after lowercasing, if on).
fn side_stats(shared: &Vocab, tokens: &[u32], lowercase: bool) -> SideStats {
    // occurrences per shared id, then merged per (lowercased) word
    let hist = tokens
        .par_chunks(1 << 20)
        .map(|c| {
            let mut h = vec![0u64; shared.len()];
            for &t in c {
                h[t as usize] += 1;
            }
            h
        })
        .reduce(
            || vec![0u64; shared.len()],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let mut per_word: FxHashMap<Cow<[u8]>, u64> = FxHashMap::default();
    for (id, &n) in hist.iter().enumerate() {
        if n > 0 {
            *per_word
                .entry(normalize(
                    unescape(shared.word(id as u32)),
                    lowercase,
                    None,
                    None,
                ))
                .or_insert(0) += n;
        }
    }
    SideStats {
        tokens: tokens.len() as u64,
        types: per_word.len() as u64,
        singletons: per_word.values().filter(|&&n| n == 1).count() as u64,
    }
}

/// One direction's view: `src` generates `tgt`. For [`Direction::Inverse`] the sides are
/// swapped; nothing is copied.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    pub direction: Direction,
    pub src: &'a SideText,
    pub tgt: &'a SideText,
}

impl Bitext {
    /// Read two line-aligned files, normalised with `opts` exactly as given.
    pub fn read_pair(src: &Path, tgt: &Path, opts: &TextOptions) -> Result<Self, CorpusError> {
        Ok(Parsed::read_pair(src, tgt, opts.chunks)?.build(opts))
    }

    /// Read one `SRC ||| TGT` file.
    pub fn read_joined(path: &Path, opts: &TextOptions) -> Result<Self, CorpusError> {
        Ok(Parsed::read_joined(path, opts.chunks)?.build(opts))
    }

    /// Parse two in-memory line-aligned texts.
    pub fn from_pair_bytes(
        src: &[u8],
        tgt: &[u8],
        opts: &TextOptions,
    ) -> Result<Self, CorpusError> {
        Ok(Parsed::from_pair_bytes(src, tgt, opts.chunks)?.build(opts))
    }

    /// Parse an in-memory `SRC ||| TGT` text (shared reader rules: the first `|||` or tab
    /// separates the sides).
    pub fn from_joined_bytes(text: &[u8], opts: &TextOptions) -> Self {
        Parsed::from_joined_bytes(text, opts.chunks).build(opts)
    }

    /// Build from a corpus parsed by the shared reader (without its `max_len`).
    pub fn from_corpus(corpus: &Corpus, opts: &TextOptions) -> Self {
        let opts = &opts.without_zero_stems();
        assert!(
            (1..=MAX_MAX_LEN).contains(&opts.max_len) || opts.max_len == NO_MAX_LEN,
            "max_len must be in 1 ..= {MAX_MAX_LEN}"
        );
        let n = corpus.len();
        let (first, second) = (&corpus.first, &corpus.second);
        let skipped: Vec<usize> = (0..n)
            .into_par_iter()
            .filter(|&k| first.line_len(k) > opts.max_len || second.line_len(k) > opts.max_len)
            .collect();
        let (source, target) = rayon::join(
            || {
                build_side(
                    corpus,
                    first.tokens(),
                    first.offsets(),
                    &skipped,
                    opts.lowercase,
                    opts.source_prefix,
                    opts.source_suffix,
                )
            },
            || {
                build_side(
                    corpus,
                    second.tokens(),
                    second.offsets(),
                    &skipped,
                    opts.lowercase,
                    opts.target_prefix,
                    opts.target_suffix,
                )
            },
        );
        Bitext {
            source,
            target,
            skipped,
            options: opts.clone(),
            segmentation: None,
        }
    }

    /// Number of line pairs K (skipped and empty lines included).
    pub fn len(&self) -> usize {
        self.source.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 0-based indices of the pairs skipped for length.
    pub fn skipped(&self) -> &[usize] {
        &self.skipped
    }

    pub fn options(&self) -> &TextOptions {
        &self.options
    }

    /// How the stemming was chosen (training path only, see [`Parsed::build_segmented`]).
    pub fn segmentation(&self) -> Option<&Segmentation> {
        self.segmentation.as_ref()
    }

    pub fn view(&self, direction: Direction) -> View<'_> {
        let (src, tgt) = match direction {
            Direction::Forward => (&self.source, &self.target),
            Direction::Inverse => (&self.target, &self.source),
        };
        View {
            direction,
            src,
            tgt,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.source.memory_bytes() + self.target.memory_bytes() + self.skipped.capacity() * 8
    }
}

impl View<'_> {
    /// Number of line pairs.
    pub fn len(&self) -> usize {
        self.src.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// (source tokens, target tokens) of pair k in this direction.
    #[inline]
    pub fn pair(&self, k: usize) -> (&[u32], &[u32]) {
        (self.src.line(k), self.tgt.line(k))
    }
}

fn load_options(chunks: Option<usize>) -> LoadOptions {
    LoadOptions {
        max_len: None,
        chunks,
    }
}

// ------------------------------------------------------------------------------------------
// Input bytes

/// A memory-mapped file (or nothing, for an empty file).
/// A whole input file: memory-mapped when it is a regular file, read to its end otherwise
/// (pipes, FIFOs, `/dev/stdin`, process substitution report a length of 0 but are not empty:
/// they used to be read as empty input).
enum Mapped {
    Map(memmap2::Mmap),
    Owned(Vec<u8>),
    Empty,
}

impl Mapped {
    fn open(path: &Path) -> Result<Self, CorpusError> {
        let err = |e| CorpusError::Io(path.to_path_buf(), e);
        let mut f = File::open(path).map_err(err)?;
        let meta = f.metadata().map_err(err)?;
        if !meta.is_file() {
            let mut v = Vec::new();
            std::io::Read::read_to_end(&mut f, &mut v).map_err(err)?;
            return Ok(Mapped::Owned(v));
        }
        if meta.len() == 0 {
            return Ok(Mapped::Empty);
        }
        // SAFETY: the file is only read; concurrent modification by another process is the
        // usual mmap caveat, as in the shared reader.
        let m = unsafe { memmap2::Mmap::map(&f) }.map_err(err)?;
        Ok(Mapped::Map(m))
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Mapped::Map(m) => m,
            Mapped::Owned(v) => v,
            Mapped::Empty => &[],
        }
    }
}

#[inline]
fn is_extra_space(c: u8, tab: bool) -> bool {
    c == b'\r' || c == 0x0b || c == 0x0c || (tab && c == b'\t')
}

/// Rewrite bytes the shared reader would treat differently from eflomal: CR, VT, FF (and tab
/// with `tab`) become spaces; with `pipes`, a `|||` token becomes [`PIPES_STANDIN`]; a token
/// starting with NUL gets one more NUL ([`unescape`] reverses both). Borrows the input when
/// nothing needs rewriting.
fn sanitize(text: &[u8], tab: bool, pipes: bool) -> Cow<'_, [u8]> {
    let has_space = text
        .par_chunks(1 << 20)
        .any(|c| c.iter().any(|&b| is_extra_space(b, tab)));
    let has_pipes = pipes && memmem::find(text, b"|||").is_some();
    let has_nul = memchr::memchr(0, text).is_some();
    if !has_space && !has_pipes && !has_nul {
        return Cow::Borrowed(text);
    }
    let mut out = Vec::with_capacity(text.len() + 16);
    let is_sep = |b: u8| b == b' ' || b == b'\n' || is_extra_space(b, tab) || b == b'\t';
    let mut k = 0;
    while k < text.len() {
        let c = text[k];
        if is_extra_space(c, tab) {
            out.push(b' ');
            k += 1;
        } else if c == 0 && (k == 0 || is_sep(text[k - 1])) {
            // a token starting with NUL: escape it (see PIPES_STANDIN)
            out.extend_from_slice(b"\x00\x00");
            k += 1;
        } else if has_pipes
            && text[k..].starts_with(b"|||")
            && (k == 0 || is_sep(text[k - 1]))
            && (k + 3 == text.len() || is_sep(text[k + 3]))
        {
            out.extend_from_slice(PIPES_STANDIN);
            k += 3;
        } else {
            out.push(c);
            k += 1;
        }
    }
    Cow::Owned(out)
}

// ------------------------------------------------------------------------------------------
// Normalisation

/// Lowercase: Unicode lowercasing (`str::to_lowercase`, the same full mapping as
/// Python's `str.lower`) for valid UTF-8, ASCII lowercasing otherwise.
pub fn lowercase(w: &[u8]) -> Cow<'_, [u8]> {
    if w.is_ascii() {
        if w.iter().any(u8::is_ascii_uppercase) {
            Cow::Owned(w.to_ascii_lowercase())
        } else {
            Cow::Borrowed(w)
        }
    } else {
        match std::str::from_utf8(w) {
            Ok(s) => {
                let l = s.to_lowercase();
                if l.as_bytes() == w {
                    Cow::Borrowed(w)
                } else {
                    Cow::Owned(l.into_bytes())
                }
            }
            Err(_) => Cow::Owned(w.to_ascii_lowercase()),
        }
    }
}

/// The first `n` characters of `w` (Unicode scalar values if `w` is valid UTF-8, else bytes).
pub fn prefix(w: &[u8], n: usize) -> &[u8] {
    match std::str::from_utf8(w) {
        Ok(s) => match s.char_indices().nth(n) {
            Some((b, _)) => &w[..b],
            None => w,
        },
        Err(_) => &w[..n.min(w.len())],
    }
}

/// The last `n` characters of `w` (as [`prefix`]).
pub fn suffix(w: &[u8], n: usize) -> &[u8] {
    match std::str::from_utf8(w) {
        Ok(s) => {
            let count = s.chars().count();
            if count <= n {
                w
            } else {
                let (b, _) = s.char_indices().nth(count - n).unwrap();
                &w[b..]
            }
        }
        Err(_) => &w[w.len() - n.min(w.len())..],
    }
}

/// Normalise one token: lowercase (optional), then prefix, then suffix.
pub fn normalize(w: &[u8], lower: bool, pre: Option<usize>, suf: Option<usize>) -> Cow<'_, [u8]> {
    let w: Cow<[u8]> = if lower {
        lowercase(w)
    } else {
        Cow::Borrowed(w)
    };
    if pre.is_none() && suf.is_none() {
        return w;
    }
    let mut s: &[u8] = &w;
    if let Some(n) = pre {
        s = prefix(s, n);
    }
    if let Some(n) = suf {
        s = suffix(s, n);
    }
    Cow::Owned(s.to_vec())
}

/// Build one side: normalise the shared dictionary, assign side ids by first occurrence over
/// the kept lines, and rewrite the tokens.
fn build_side(
    corpus: &Corpus,
    tokens: &[u32],
    offsets: &[usize],
    skipped: &[usize],
    lower: bool,
    pre: Option<usize>,
    suf: Option<usize>,
) -> SideText {
    let shared = &corpus.vocab;
    // Normalised word of every shared id (in parallel), then a canonical index per distinct
    // normalised word (sequential, in shared-id order, so deterministic).
    let norm: Vec<Cow<[u8]>> = (0..shared.len() as u32)
        .into_par_iter()
        .map(|id| normalize(unescape(shared.word(id)), lower, pre, suf))
        .collect();
    let mut canon_of: FxHashMap<&[u8], u32> = FxHashMap::default();
    let canon: Vec<u32> = norm
        .iter()
        .map(|w| {
            let next = canon_of.len() as u32;
            *canon_of.entry(w.as_ref()).or_insert(next)
        })
        .collect();
    let n_canon = canon_of.len();
    drop(canon_of);

    // Side ids by first occurrence.
    let mut side_of_canon = vec![UNSET; n_canon];
    let mut words: Vec<&[u8]> = vec![NULL_WORD];
    let mut visit = |toks: &[u32]| {
        for &t in toks {
            let c = canon[t as usize] as usize;
            if side_of_canon[c] == UNSET {
                side_of_canon[c] = words.len() as u32;
                words.push(&norm[t as usize]);
            }
        }
    };
    if skipped.is_empty() {
        visit(tokens);
    } else {
        let mut next_skip = skipped.iter().peekable();
        for k in 0..offsets.len() - 1 {
            if next_skip.peek() == Some(&&k) {
                next_skip.next();
                continue;
            }
            visit(&tokens[offsets[k]..offsets[k + 1]]);
        }
    }
    let vocab = Vocab::from_words(words.iter().copied());
    let map: Vec<u32> = canon.iter().map(|&c| side_of_canon[c as usize]).collect();

    let (tokens, offsets) = if skipped.is_empty() {
        let t: Vec<u32> = tokens.par_iter().map(|&t| map[t as usize]).collect();
        (t, offsets.to_vec())
    } else {
        let mut t = Vec::with_capacity(tokens.len());
        let mut o = Vec::with_capacity(offsets.len());
        o.push(0);
        let mut next_skip = skipped.iter().peekable();
        for k in 0..offsets.len() - 1 {
            if next_skip.peek() == Some(&&k) {
                next_skip.next();
            } else {
                t.extend(
                    tokens[offsets[k]..offsets[k + 1]]
                        .iter()
                        .map(|&x| map[x as usize]),
                );
            }
            o.push(t.len());
        }
        (t, o)
    };
    debug_assert!(tokens.iter().all(|&t| t != UNSET && t != 0));
    SideText {
        vocab,
        tokens,
        offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &SideText, k: usize) -> Vec<String> {
        s.line(k)
            .iter()
            .map(|&i| String::from_utf8_lossy(s.vocab.word(i)).into_owned())
            .collect()
    }

    fn opts() -> TextOptions {
        TextOptions::default()
    }

    #[test]
    fn separate_vocabularies_first_occurrence() {
        let b = Bitext::from_pair_bytes(b"a b c\nb d\n", b"x a\ny y z\n", &opts()).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b.source.vocab.word(0), NULL_WORD);
        assert_eq!(b.source.tokens(), &[1, 2, 3, 2, 4]);
        assert_eq!(b.target.tokens(), &[1, 2, 3, 3, 4]);
        assert_eq!(b.source.n_types(), 5);
        assert_eq!(b.target.n_types(), 5);
        assert_eq!(words(&b.target, 0), ["x", "a"]);
        // `a` has different ids on the two sides.
        assert_eq!(b.source.line(0)[0], 1);
        assert_eq!(b.target.line(0)[1], 2);
    }

    #[test]
    fn views_swap_without_copying() {
        let b = Bitext::from_pair_bytes(b"a b\n", b"x\n", &opts()).unwrap();
        let f = b.view(Direction::Forward);
        let i = b.view(Direction::Inverse);
        assert_eq!(f.pair(0), (&[1u32, 2][..], &[1u32][..]));
        assert_eq!(i.pair(0), (&[1u32][..], &[1u32, 2][..]));
        assert!(std::ptr::eq(f.src, i.tgt));
        assert!(std::ptr::eq(f.tgt, i.src));
        assert_eq!(i.len(), 1);
    }

    #[test]
    fn lowercasing_and_option() {
        let src = "The THE the Été ÉTÉ ΟΔΟΣ\n".as_bytes();
        let b = Bitext::from_pair_bytes(src, b"x\n", &opts()).unwrap();
        assert_eq!(
            words(&b.source, 0),
            ["the", "the", "the", "été", "été", "οδος"]
        );
        assert_eq!(b.source.n_types(), 4);
        let o = TextOptions {
            lowercase: false,
            ..opts()
        };
        let b = Bitext::from_pair_bytes(src, b"x\n", &o).unwrap();
        assert_eq!(b.source.n_types(), 7);
        // Invalid UTF-8: ASCII lowercasing of the bytes.
        let b = Bitext::from_pair_bytes(b"AB\xe9C ab\xe9c\n", b"x\n", &opts()).unwrap();
        assert_eq!(b.source.tokens(), &[1, 1]);
        assert_eq!(b.source.vocab.word(1), b"ab\xe9c");
    }

    #[test]
    fn final_sigma_like_python() {
        // Python: "ΟΔΟΣ".lower() == "οδος" with a final sigma.
        assert_eq!(&*lowercase("ΟΔΟΣ".as_bytes()), "οδος".as_bytes());
        assert_eq!(&*lowercase("ΣΑ".as_bytes()), "σα".as_bytes());
    }

    #[test]
    fn stemming() {
        assert_eq!(prefix("héllo".as_bytes(), 2), "hé".as_bytes());
        assert_eq!(suffix("héllo".as_bytes(), 4), "éllo".as_bytes());
        assert_eq!(prefix(b"ab", 5), b"ab");
        assert_eq!(suffix(b"ab", 5), b"ab");
        assert_eq!(prefix(b"a\xffcd", 2), b"a\xff");
        assert_eq!(suffix(b"a\xffcd", 3), b"\xffcd");
        let o = TextOptions {
            source_prefix: Some(4),
            target_suffix: Some(2),
            ..opts()
        };
        let b = Bitext::from_pair_bytes(b"Walking walked\n", b"maison maisons\n", &o).unwrap();
        assert_eq!(words(&b.source, 0), ["walk", "walk"]);
        assert_eq!(words(&b.target, 0), ["on", "ns"]);
        // Prefix first, then suffix of that.
        assert_eq!(&*normalize(b"abcdef", true, Some(4), Some(2)), b"cd");
    }

    #[test]
    fn whitespace_rules_two_files() {
        // Tab, CR, VT, FF separate tokens; `|||` in the source file is a word.
        let src = b"a\tb\r\nc ||| d\x0be\x0cf\n|||\n";
        let tgt = b"x\ty\r\nz ||| w\n|||\n";
        let b = Bitext::from_pair_bytes(src, tgt, &opts()).unwrap();
        assert_eq!(words(&b.source, 0), ["a", "b"]);
        assert_eq!(words(&b.target, 0), ["x", "y"]);
        assert_eq!(words(&b.source, 1), ["c", "|||", "d", "e", "f"]);
        assert_eq!(words(&b.target, 1), ["z", "|||", "w"]);
        assert_eq!(words(&b.source, 2), ["|||"]);
        assert_eq!(words(&b.target, 2), ["|||"]);
        assert_eq!(b.source.line(1)[1], b.source.line(2)[0]);
        // Tokens that merely contain `|||` are unchanged.
        let b = Bitext::from_pair_bytes(b"a|||b ||||\n", b"x\n", &opts()).unwrap();
        assert_eq!(words(&b.source, 0), ["a|||b", "||||"]);
    }

    #[test]
    fn whitespace_rules_joined() {
        // Joined format: the shared reader's rules (first `|||` or tab splits); CR, VT, FF are
        // token separators.
        let b = Bitext::from_joined_bytes(b"a b ||| x ||| y\r\nc\td\x0ce\n", &opts());
        assert_eq!(words(&b.source, 0), ["a", "b"]);
        assert_eq!(words(&b.target, 0), ["x", "|||", "y"]);
        assert_eq!(words(&b.source, 1), ["c"]);
        assert_eq!(words(&b.target, 1), ["d", "e"]);
    }

    #[test]
    fn long_pairs_are_skipped_and_kept_as_empty_lines() {
        let o = TextOptions {
            max_len: 3,
            ..opts()
        };
        let src = b"a b c\nd e f g\nh\ni\n";
        let tgt = b"x\ny\nonly here too\nz\n";
        let b = Bitext::from_pair_bytes(src, tgt, &o).unwrap();
        assert_eq!(b.len(), 4);
        assert_eq!(b.skipped(), &[1]);
        assert_eq!(b.source.line_len(1), 0);
        assert_eq!(b.target.line_len(1), 0);
        // Words only in skipped lines are not in the vocabulary; 3 tokens are allowed.
        assert_eq!(b.source.n_types(), 1 + 5); // a b c h i
        assert_eq!(b.target.n_types(), 1 + 5); // x only here too z (not y)
        assert_eq!(words(&b.target, 2), ["only", "here", "too"]);
        assert_eq!(words(&b.target, 3), ["z"]);
    }

    #[test]
    fn empty_lines_and_sides() {
        let b = Bitext::from_pair_bytes(b"a\n\n\nb", b"\nx\n\ny", &opts()).unwrap();
        assert_eq!(b.len(), 4);
        assert_eq!(b.source.line_len(0), 1);
        assert_eq!(b.target.line_len(0), 0);
        assert_eq!(b.source.line_len(1), 0);
        assert_eq!(b.target.line_len(1), 1);
        assert_eq!(b.source.line_len(2) + b.target.line_len(2), 0);
        assert_eq!(words(&b.source, 3), ["b"]);
        assert!(b.skipped().is_empty());
        let e = Bitext::from_pair_bytes(b"", b"", &opts()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.source.n_types(), 1);
    }

    #[test]
    fn line_count_mismatch() {
        let e = Bitext::from_pair_bytes(b"a\nb\n", b"x\n", &opts()).unwrap_err();
        assert!(matches!(
            e,
            CorpusError::LineCountMismatch { src: 2, tgt: 1 }
        ));
    }

    #[test]
    fn ids_independent_of_chunking() {
        // A corpus with case variants, CR and repeated words, read with 1 … 13 chunks.
        let mut src = Vec::new();
        let mut tgt = Vec::new();
        for k in 0..400u32 {
            let w = |m: u32| format!("w{}", (k * m) % 97);
            src.extend_from_slice(
                format!("{} {} {}\n", w(3), w(5).to_uppercase(), w(7)).as_bytes(),
            );
            tgt.extend_from_slice(format!("{} {}\r\n", w(11), w(13)).as_bytes());
        }
        let base = Bitext::from_pair_bytes(
            &src,
            &tgt,
            &TextOptions {
                chunks: Some(1),
                ..opts()
            },
        )
        .unwrap();
        for chunks in [2, 3, 7, 13] {
            let b = Bitext::from_pair_bytes(
                &src,
                &tgt,
                &TextOptions {
                    chunks: Some(chunks),
                    ..opts()
                },
            )
            .unwrap();
            assert_eq!(b.source.tokens(), base.source.tokens());
            assert_eq!(b.target.tokens(), base.target.tokens());
            assert_eq!(b.source.vocab, base.source.vocab);
            assert_eq!(b.target.vocab, base.target.vocab);
            assert_eq!(b.source.offsets(), base.source.offsets());
        }
        // Joined input gives the same result as two files.
        let mut joined = Vec::new();
        for (a, b) in src
            .split(|&c| c == b'\n')
            .zip(tgt.split(|&c| c == b'\n'))
            .take(400)
        {
            joined.extend_from_slice(a);
            joined.extend_from_slice(b" ||| ");
            joined.extend_from_slice(b);
            joined.push(b'\n');
        }
        let j = Bitext::from_joined_bytes(&joined, &opts());
        assert_eq!(j.source.tokens(), base.source.tokens());
        assert_eq!(j.target.tokens(), base.target.tokens());
        assert_eq!(j.target.vocab, base.target.vocab);
    }

    #[test]
    fn files() {
        let dir = tempfile::tempdir().unwrap();
        let s = dir.path().join("s");
        let t = dir.path().join("t");
        let empty = dir.path().join("e");
        std::fs::write(&s, b"A b\n").unwrap();
        std::fs::write(&t, b"x\n").unwrap();
        std::fs::write(&empty, b"").unwrap();
        let b = Bitext::read_pair(&s, &t, &opts()).unwrap();
        assert_eq!(words(&b.source, 0), ["a", "b"]);
        let e = Bitext::read_pair(&empty, &empty, &opts()).unwrap();
        assert!(e.is_empty());
        assert!(Bitext::read_pair(&dir.path().join("missing"), &t, &opts()).is_err());
    }
}
