//! Segmentation: the `auto` stemming rule, corpus statistics and `estimate`.
//!
//! eflomal's prefix stemming (`--source-prefix/--target-prefix N`) keeps one token per word
//! and lowered AER on every corpus measured. The `auto` rule, fitted
//! on 27 corpora of three language pairs, picks the prefix length from the number of
//! input lines K alone, the same on both sides:
//!
//! | K | prefix |
//! |---|---|
//! | < 30 000 | 4 |
//! | < 450 000 | 5 |
//! | otherwise | 6 |
//!
//! [`choose`] applies it: `auto` is the default, `words` turns it off, and any explicit
//! stemming option (`--source-prefix`, `--source-suffix`, `--target-prefix`,
//! `--target-suffix`) switches it off as well and is used exactly as given. Training
//! ([`crate::bitext::Parsed::build_segmented`]) and [`estimate`] both call [`choose`], so they
//! cannot disagree. The choice and the statistics it came from are saved with the model
//! ([`Segmentation`]).
//!
//! [`estimate`] reports the per-side statistics, the choice and the rule that fired, the
//! measured evidence of the nearest measured corpora ([`crate::segment_data`]) and the sampler
//! parameters (reported only: no default is changed by it). For text to be aligned with an
//! existing model, [`align_check`] compares the new text with the training corpus and warns
//! when it is far outside it.

use std::fmt::Write as _;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::bitext::{Bitext, SideText, TextOptions};
use crate::corpus::Vocab;
pub use crate::segment_data::{AUTO_PREFIXES, AUTO_THRESHOLDS, EVIDENCE, PARAMETERS};

/// How the stemming of a training run is chosen (`--segment`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Segment {
    /// The line-count rule (default), unless stemming options are given explicitly.
    #[default]
    Auto,
    /// No stemming, unless stemming options are given explicitly.
    Words,
}

impl Segment {
    pub fn name(self) -> &'static str {
        match self {
            Segment::Auto => "auto",
            Segment::Words => "words",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Segment::Auto),
            "words" => Ok(Segment::Words),
            _ => Err(format!("segment must be 'auto' or 'words', not '{s}'")),
        }
    }
}

/// Where the stemming of a run came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// No stemming (`--segment words`, no stemming options).
    Words,
    /// Chosen by the `auto` rule.
    Auto,
    /// Explicit stemming options.
    Explicit,
}

impl Origin {
    pub fn name(self) -> &'static str {
        match self {
            Origin::Words => "words",
            Origin::Auto => "auto",
            Origin::Explicit => "explicit",
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Origin::Words => 0,
            Origin::Auto => 1,
            Origin::Explicit => 2,
        }
    }

    pub fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(Origin::Words),
            1 => Some(Origin::Auto),
            2 => Some(Origin::Explicit),
            _ => None,
        }
    }
}

/// Counts of one side of a corpus (types after lowercasing, if it is on).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SideStats {
    pub tokens: u64,
    pub types: u64,
    /// Types that occur exactly once (= tokens of such types).
    pub singletons: u64,
}

impl SideStats {
    pub fn tokens_per_type(&self) -> f64 {
        if self.types == 0 {
            0.0
        } else {
            self.tokens as f64 / self.types as f64
        }
    }

    /// Share of tokens whose type occurs once, in percent.
    pub fn singleton_share(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            100.0 * self.singletons as f64 / self.tokens as f64
        }
    }
}

/// Statistics of a corpus, as the reader computes them in one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CorpusStats {
    /// Input lines K (empty and later skipped lines included): the rule's statistic.
    pub lines: u64,
    /// Whether the types were counted after lowercasing.
    pub lowercase: bool,
    pub source: SideStats,
    pub target: SideStats,
}

impl CorpusStats {
    /// Mean tokens per line of a side.
    pub fn mean_length(&self, s: &SideStats) -> f64 {
        if self.lines == 0 {
            0.0
        } else {
            s.tokens as f64 / self.lines as f64
        }
    }
}

/// How a training run's stemming was chosen, and the statistics of its corpus (saved with the
/// model).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segmentation {
    pub origin: Origin,
    pub stats: CorpusStats,
}

/// The rule: prefix length on both sides for a corpus of `lines` input lines.
pub fn auto_prefix(lines: u64) -> usize {
    AUTO_THRESHOLDS
        .iter()
        .position(|&t| lines < t)
        .map_or(AUTO_PREFIXES[AUTO_PREFIXES.len() - 1], |i| AUTO_PREFIXES[i])
}

fn has_stems(o: &TextOptions) -> bool {
    o.source_prefix.is_some()
        || o.source_suffix.is_some()
        || o.target_prefix.is_some()
        || o.target_suffix.is_some()
}

/// The text options a training run uses: `opts` with the stemming chosen by `segment` for a
/// corpus of `lines` lines. Explicit stemming options in `opts` always win (and switch `auto`
/// off for both sides).
pub fn choose(segment: Segment, opts: &TextOptions, lines: u64) -> (TextOptions, Origin) {
    if has_stems(opts) {
        return (opts.without_zero_stems(), Origin::Explicit);
    }
    match segment {
        Segment::Words => (opts.clone(), Origin::Words),
        Segment::Auto => {
            let n = auto_prefix(lines);
            (
                TextOptions {
                    source_prefix: Some(n),
                    target_prefix: Some(n),
                    ..opts.clone()
                },
                Origin::Auto,
            )
        }
    }
}

/// `1234567` → `1,234,567`.
pub fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The rule's bin for `lines`, e.g. `30,000 ≤ 48,689 lines < 450,000 → prefix 5`.
pub fn rule_fired(lines: u64) -> String {
    let i = AUTO_THRESHOLDS
        .iter()
        .position(|&t| lines < t)
        .unwrap_or(AUTO_THRESHOLDS.len());
    let mut s = String::new();
    if i > 0 {
        let _ = write!(s, "{} ≤ ", thousands(AUTO_THRESHOLDS[i - 1]));
    }
    let _ = write!(s, "{} lines", thousands(lines));
    if i < AUTO_THRESHOLDS.len() {
        let _ = write!(s, " < {}", thousands(AUTO_THRESHOLDS[i]));
    }
    let _ = write!(s, " → prefix {}", AUTO_PREFIXES[i]);
    s
}

/// The whole rule in one line.
pub fn rule_description() -> String {
    let mut s = String::new();
    for (t, p) in AUTO_THRESHOLDS.iter().zip(AUTO_PREFIXES) {
        let _ = write!(s, "below {} lines prefix {p}, ", thousands(*t));
    }
    let _ = write!(
        s,
        "otherwise prefix {} (both sides)",
        AUTO_PREFIXES[AUTO_PREFIXES.len() - 1]
    );
    s
}

fn stems_text(o: &TextOptions) -> String {
    let side = |p: Option<usize>, s: Option<usize>| match (p, s) {
        (None, None) => "words".to_string(),
        (Some(p), None) => format!("prefix {p}"),
        (None, Some(s)) => format!("suffix {s}"),
        (Some(p), Some(s)) => format!("prefix {p} then suffix {s}"),
    };
    let (a, b) = (
        side(o.source_prefix, o.source_suffix),
        side(o.target_prefix, o.target_suffix),
    );
    if a == b {
        if a == "words" {
            "no stemming (words)".into()
        } else {
            format!("{a} on both sides")
        }
    } else {
        format!("source {a}, target {b}")
    }
}

/// One line for the training log, e.g. `segmentation: prefix 5 on both sides (auto: 30,000 ≤
/// 48,689 lines < 450,000 → prefix 5)`.
pub fn choice_line(text: &TextOptions, origin: Origin, lines: u64) -> String {
    let why = match origin {
        Origin::Auto => format!("auto: {}", rule_fired(lines)),
        Origin::Words => "--segment words".into(),
        Origin::Explicit => "explicit stemming options".into(),
    };
    format!("segmentation: {} ({why})", stems_text(text))
}

// ------------------------------------------------------------------------------------------
// Evidence and parameter data (generated into `segment_data.rs`)

/// The rule's pick against plain words on one measured corpus.
#[derive(Debug)]
pub struct Evidence {
    pub corpus: &'static str,
    /// Language pair (`enfr`, `enis`, `roen`).
    pub pair: &'static str,
    pub lines: u64,
    /// The rule's pick, e.g. `prefix:5`.
    pub setting: &'static str,
    /// gdfa AER with words and with the pick.
    pub words_gdfa: f64,
    pub gdfa: f64,
    /// AER(pick) − AER(words) and its 95 % bootstrap interval, per output.
    pub gdfa_delta: f64,
    pub gdfa_ci: (f64, f64),
    pub fwd_delta: f64,
    pub fwd_ci: (f64, f64),
    pub inv_delta: f64,
    pub inv_ci: (f64, f64),
    /// The best measured setting on this corpus.
    pub best: &'static str,
    /// Median 95 % half-width of setting-vs-words differences on this corpus.
    pub noise_halfwidth: f64,
}

/// One measured alternative of a parameter.
#[derive(Debug)]
pub struct Tradeoff {
    pub value: &'static str,
    pub corpora: u32,
    pub mean_gdfa_delta: f64,
    pub real_better_gdfa: u32,
    pub real_worse_gdfa: u32,
    /// Sampling time relative to the default (rough).
    pub time_ratio: f64,
}

/// A sampler parameter and what the measurements say about it.
#[derive(Debug)]
pub struct ParamInfo {
    pub option: &'static str,
    pub python: &'static str,
    pub default: &'static str,
    pub decision: &'static str,
    pub tradeoff: &'static [Tradeoff],
}

/// The measured corpus nearest to `lines` (in log line count) of each language pair.
pub fn nearest_evidence(lines: u64) -> Vec<&'static Evidence> {
    let x = (lines.max(1) as f64).ln();
    let mut best: Vec<&'static Evidence> = Vec::new();
    for e in EVIDENCE {
        let d = ((e.lines as f64).ln() - x).abs();
        match best.iter_mut().find(|b| b.pair == e.pair) {
            Some(b) => {
                if d < ((b.lines as f64).ln() - x).abs() {
                    *b = e;
                }
            }
            None => best.push(e),
        }
    }
    best
}

// ------------------------------------------------------------------------------------------
// estimate

/// The new text against the corpus that fixed the segmentation.
#[derive(Clone, Debug)]
pub struct AlignCheck {
    /// `"model"` or `"training corpus"`.
    pub reference: &'static str,
    pub stats: CorpusStats,
    /// Percent of the new text's tokens (normalised as for alignment) whose word the
    /// reference does not have, source and target.
    pub oov: [f64; 2],
    pub warnings: Vec<String>,
}

/// What [`estimate`] reports.
#[derive(Clone, Debug)]
pub struct Estimate {
    /// Statistics of the training corpus (`None` for a model saved without them).
    pub stats: Option<CorpusStats>,
    /// The text options training uses (or used).
    pub text: TextOptions,
    /// The requested `--segment` (`None` when a model fixes the segmentation).
    pub segment: Option<Segment>,
    /// How the stemming was chosen (`None`: a model saved without the record).
    pub origin: Option<Origin>,
    /// Whether the segmentation is a saved model's.
    pub from_model: bool,
    /// Measured evidence near the training corpus's size.
    pub evidence: Vec<&'static Evidence>,
    pub align: Option<AlignCheck>,
}

/// Estimate for training on a corpus with these statistics: the same choice as training.
pub fn estimate(stats: &CorpusStats, opts: &TextOptions, segment: Segment) -> Estimate {
    let (text, origin) = choose(segment, opts, stats.lines);
    Estimate {
        stats: Some(*stats),
        text: TextOptions {
            chunks: None,
            ..text
        },
        segment: Some(segment),
        origin: Some(origin),
        from_model: false,
        evidence: nearest_evidence(stats.lines),
        align: None,
    }
}

/// Estimate for a saved model: its own segmentation (the training corpus fixed it).
pub fn estimate_model(text: &TextOptions, seg: Option<&Segmentation>) -> Estimate {
    Estimate {
        stats: seg.map(|s| s.stats),
        text: TextOptions {
            chunks: None,
            ..text.clone()
        },
        segment: None,
        origin: seg.map(|s| s.origin),
        from_model: true,
        evidence: seg.map_or_else(Vec::new, |s| nearest_evidence(s.stats.lines)),
        align: None,
    }
}

/// Share of tokens of `side` whose word is not in `vocab`, in percent.
fn oov_share(side: &SideText, vocab: &Vocab) -> f64 {
    let known: FxHashSet<&[u8]> = (1..vocab.len() as u32).map(|i| vocab.word(i)).collect();
    let mut count: FxHashMap<u32, u64> = FxHashMap::default();
    for &t in side.tokens() {
        *count.entry(t).or_insert(0) += 1;
    }
    let unknown: u64 = count
        .iter()
        .filter(|(&id, _)| !known.contains(side.vocab.word(id)))
        .map(|(_, &n)| n)
        .sum();
    if side.n_tokens() == 0 {
        0.0
    } else {
        100.0 * unknown as f64 / side.n_tokens() as f64
    }
}

/// Above this share of unknown tokens (percent) on a side, `estimate` warns.
pub const OOV_WARN: f64 = 20.0;
/// Outside this ratio of mean sentence lengths (new / training) on a side, `estimate` warns.
pub const LENGTH_WARN: (f64, f64) = (0.5, 2.0);

/// Compare text to be aligned (`new`, read with the reference's text options, and its
/// statistics) with the reference that fixes the segmentation: the training corpus's statistics
/// (if known) and its normalised vocabularies (`source`, `target`).
pub fn align_check(
    reference: &'static str,
    train: Option<&CorpusStats>,
    source: &Vocab,
    target: &Vocab,
    new: &Bitext,
    stats: CorpusStats,
) -> AlignCheck {
    let oov = [
        oov_share(&new.source, source),
        oov_share(&new.target, target),
    ];
    let mut warnings = Vec::new();
    for (k, name) in ["source", "target"].into_iter().enumerate() {
        if oov[k] > OOV_WARN {
            warnings.push(format!(
                "{:.1} % of the {name} tokens are unknown to the {reference} (warning above \
                 {OOV_WARN:.0} %): the text may be another domain or language, or be \
                 tokenised differently",
                oov[k]
            ));
        }
        if let Some(t) = train {
            let side = |s: &CorpusStats| if k == 0 { s.source } else { s.target };
            let (a, b) = (stats.mean_length(&side(&stats)), t.mean_length(&side(t)));
            if b > 0.0 && a > 0.0 && !(LENGTH_WARN.0..=LENGTH_WARN.1).contains(&(a / b)) {
                warnings.push(format!(
                    "the {name} sentences average {a:.1} tokens against {b:.1} in the training \
                     corpus (warning outside ×{} … ×{})",
                    LENGTH_WARN.0, LENGTH_WARN.1
                ));
            }
        }
    }
    if train.is_none() {
        warnings.push(
            "the model was saved without its training statistics; only unknown words were \
             compared"
                .into(),
        );
    }
    AlignCheck {
        reference,
        stats,
        oov,
        warnings,
    }
}

// ------------------------------------------------------------------------------------------
// Output

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// A finite f64 as JSON (rounded to 4 decimals for derived ratios).
fn json_f(x: f64) -> String {
    if x.is_finite() {
        let r = (x * 1e4).round() / 1e4;
        format!("{r}")
    } else {
        "null".into()
    }
}

fn json_opt(x: Option<usize>) -> String {
    x.map_or("null".into(), |v| v.to_string())
}

fn json_side(stats: &CorpusStats, s: &SideStats) -> String {
    format!(
        "{{\"tokens\": {}, \"types\": {}, \"singletons\": {}, \"tokens_per_type\": {}, \
         \"singleton_share\": {}, \"mean_length\": {}}}",
        s.tokens,
        s.types,
        s.singletons,
        json_f(s.tokens_per_type()),
        json_f(s.singleton_share()),
        json_f(stats.mean_length(s))
    )
}

fn json_stats(s: &CorpusStats) -> String {
    format!(
        "{{\"lines\": {}, \"lowercase\": {}, \"source\": {}, \"target\": {}}}",
        s.lines,
        s.lowercase,
        json_side(s, &s.source),
        json_side(s, &s.target)
    )
}

fn json_pair(p: (f64, f64)) -> String {
    format!("[{}, {}]", json_f(p.0), json_f(p.1))
}

impl Estimate {
    /// Warnings about the training corpus: no line pairs, or a side without words (training
    /// on it succeeds but aligns nothing; see `pipeline::corpus_warnings`).
    pub fn warnings(&self) -> Vec<String> {
        match &self.stats {
            Some(s) if s.lines == 0 => {
                vec!["the corpus has no line pairs: training would align nothing".to_owned()]
            }
            Some(s) if s.source.tokens == 0 || s.target.tokens == 0 => vec![format!(
                "the {} side has no words: training would align nothing",
                if s.source.tokens == 0 {
                    "source"
                } else {
                    "target"
                }
            )],
            _ => Vec::new(),
        }
    }

    /// The rule's bin for the training corpus (`None` without statistics).
    pub fn rule(&self) -> Option<String> {
        self.stats.map(|s| rule_fired(s.lines))
    }

    /// What the rule would pick for the training corpus.
    pub fn auto_prefix(&self) -> Option<usize> {
        self.stats.map(|s| auto_prefix(s.lines))
    }

    /// Machine-readable output (one JSON object).
    pub fn json(&self) -> String {
        let mut o = String::from("{\n");
        let _ = writeln!(
            o,
            "  \"statistics\": {},",
            self.stats.as_ref().map_or("null".into(), json_stats)
        );
        let _ = writeln!(
            o,
            "  \"warnings\": [{}],",
            self.warnings()
                .iter()
                .map(|w| json_str(w))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let t = &self.text;
        let _ = writeln!(
            o,
            "  \"segmentation\": {{\"segment\": {}, \"origin\": {}, \"from_model\": {}, \
             \"source_prefix\": {}, \"source_suffix\": {}, \"target_prefix\": {}, \
             \"target_suffix\": {}, \"lowercase\": {}, \"auto_prefix\": {}, \"rule_fired\": {}, \
             \"rule\": {}, \"thresholds\": [{}], \"prefixes\": [{}]}},",
            self.segment.map_or("null".into(), |s| json_str(s.name())),
            self.origin.map_or("null".into(), |s| json_str(s.name())),
            self.from_model,
            json_opt(t.source_prefix),
            json_opt(t.source_suffix),
            json_opt(t.target_prefix),
            json_opt(t.target_suffix),
            t.lowercase,
            json_opt(self.auto_prefix()),
            self.rule().map_or("null".into(), |r| json_str(&r)),
            json_str(&rule_description()),
            AUTO_THRESHOLDS
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            AUTO_PREFIXES
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        );
        o.push_str("  \"evidence\": [");
        for (k, e) in self.evidence.iter().enumerate() {
            let _ = write!(
                o,
                "{}\n    {{\"corpus\": {}, \"pair\": {}, \"lines\": {}, \"setting\": {}, \
                 \"words_gdfa\": {}, \"gdfa\": {}, \"gdfa_delta\": {}, \"gdfa_ci\": {}, \
                 \"fwd_delta\": {}, \"fwd_ci\": {}, \"inv_delta\": {}, \"inv_ci\": {}, \
                 \"best\": {}, \"noise_halfwidth\": {}}}",
                if k == 0 { "" } else { "," },
                json_str(e.corpus),
                json_str(e.pair),
                e.lines,
                json_str(e.setting),
                json_f(e.words_gdfa),
                json_f(e.gdfa),
                json_f(e.gdfa_delta),
                json_pair(e.gdfa_ci),
                json_f(e.fwd_delta),
                json_pair(e.fwd_ci),
                json_f(e.inv_delta),
                json_pair(e.inv_ci),
                json_str(e.best),
                json_f(e.noise_halfwidth),
            );
        }
        o.push_str(if self.evidence.is_empty() {
            "],\n"
        } else {
            "\n  ],\n"
        });
        o.push_str("  \"parameters\": [");
        for (k, p) in PARAMETERS.iter().enumerate() {
            let tr: Vec<String> = p
                .tradeoff
                .iter()
                .map(|t| {
                    format!(
                        "{{\"value\": {}, \"corpora\": {}, \"mean_gdfa_delta\": {}, \
                         \"real_better_gdfa\": {}, \"real_worse_gdfa\": {}, \"time_ratio\": {}}}",
                        json_str(t.value),
                        t.corpora,
                        json_f(t.mean_gdfa_delta),
                        t.real_better_gdfa,
                        t.real_worse_gdfa,
                        json_f(t.time_ratio)
                    )
                })
                .collect();
            let _ = write!(
                o,
                "{}\n    {{\"option\": {}, \"python\": {}, \"default\": {}, \"decision\": {}, \
                 \"tradeoff\": [{}]}}",
                if k == 0 { "" } else { "," },
                json_str(p.option),
                json_str(p.python),
                json_str(p.default),
                json_str(p.decision),
                tr.join(", ")
            );
        }
        o.push_str("\n  ],\n");
        match &self.align {
            None => o.push_str("  \"align\": null\n"),
            Some(a) => {
                let _ = writeln!(
                    o,
                    "  \"align\": {{\"reference\": {}, \"statistics\": {}, \"oov_share\": \
                     {{\"source\": {}, \"target\": {}}}, \"warnings\": [{}]}}",
                    json_str(a.reference),
                    json_stats(&a.stats),
                    json_f(a.oov[0]),
                    json_f(a.oov[1]),
                    a.warnings
                        .iter()
                        .map(|w| json_str(w))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        o.push_str("}\n");
        o
    }

    /// Human-readable report.
    pub fn text(&self) -> String {
        let mut o = String::new();
        let stats_table = |o: &mut String, s: &CorpusStats| {
            let _ = writeln!(
                o,
                "  {:<8} {:>12} {:>10} {:>12} {:>11} {:>12}",
                "", "tokens", "types", "tokens/type", "singletons", "mean length"
            );
            for (name, side) in [("source", &s.source), ("target", &s.target)] {
                let _ = writeln!(
                    o,
                    "  {:<8} {:>12} {:>10} {:>12.1} {:>9.2} % {:>12.1}",
                    name,
                    thousands(side.tokens),
                    thousands(side.types),
                    side.tokens_per_type(),
                    side.singleton_share(),
                    s.mean_length(side)
                );
            }
        };
        match &self.stats {
            Some(s) => {
                let _ = writeln!(
                    o,
                    "{}: {} lines{}",
                    if self.from_model {
                        "Training corpus of the model"
                    } else {
                        "Corpus"
                    },
                    thousands(s.lines),
                    if s.lowercase {
                        ", types counted after lowercasing"
                    } else {
                        ""
                    }
                );
                stats_table(&mut o, s);
                for w in self.warnings() {
                    let _ = writeln!(o, "  warning: {w}");
                }
            }
            None => o.push_str(
                "Training corpus of the model: no statistics (saved by an older version)\n",
            ),
        }
        o.push('\n');
        let _ = writeln!(o, "Segmentation: {}", stems_text(&self.text));
        match (self.origin, &self.stats) {
            (Some(Origin::Auto), Some(s)) => {
                let _ = writeln!(o, "  chosen by the auto rule: {}", rule_fired(s.lines));
            }
            (Some(Origin::Words), Some(s)) => {
                let _ = writeln!(
                    o,
                    "  --segment words (the auto rule would choose prefix {})",
                    auto_prefix(s.lines)
                );
            }
            (Some(Origin::Explicit), Some(s)) => {
                let _ = writeln!(
                    o,
                    "  explicit stemming options (the auto rule would choose prefix {})",
                    auto_prefix(s.lines)
                );
            }
            _ => o.push_str("  set by the model (how it was chosen is not recorded)\n"),
        }
        let _ = writeln!(o, "  rule: {}", rule_description());
        if self.from_model {
            o.push_str(
                "  The training corpus fixes the segmentation; `align` reads new text with it.\n",
            );
        } else {
            o.push_str(
                "  Override with --segment words or explicit --source-prefix/--target-prefix \
                 (and suffix) options.\n",
            );
        }
        if !self.evidence.is_empty() {
            o.push_str(
                "\nMeasured evidence (rule's pick vs words, gdfa AER, seeds 1-3, 95 % bootstrap \
                 CI over gold sentences;\nnearest measured corpus of each language pair):\n",
            );
            let _ = writeln!(
                o,
                "  {:<12} {:>10}  {:<9} {:>15}  {:<24} best measured",
                "corpus", "lines", "setting", "words → pick", "Δ gdfa [95 % CI]"
            );
            for e in &self.evidence {
                let _ = writeln!(
                    o,
                    "  {:<12} {:>10}  {:<9} {:>6.2} → {:>5.2}  {:+.2} [{:+.2}, {:+.2}]{:<5} {}",
                    e.corpus,
                    thousands(e.lines),
                    e.setting,
                    e.words_gdfa,
                    e.gdfa,
                    e.gdfa_delta,
                    e.gdfa_ci.0,
                    e.gdfa_ci.1,
                    "",
                    e.best
                );
            }
        }
        o.push_str("\nSampler parameters (reported only; estimate changes none of them):\n");
        for p in PARAMETERS {
            let _ = writeln!(o, "  {} {}: {}", p.option, p.default, p.decision);
            if !p.tradeoff.is_empty() {
                let parts: Vec<String> = p
                    .tradeoff
                    .iter()
                    .map(|t| {
                        format!(
                            "{} → {:+.2} mean gdfa at ×{:.2} sampling time ({} of {} corpora \
                             real gain)",
                            t.value, t.mean_gdfa_delta, t.time_ratio, t.real_better_gdfa, t.corpora
                        )
                    })
                    .collect();
                let _ = writeln!(o, "    measured: {}", parts.join("; "));
            }
        }
        if let Some(a) = &self.align {
            let _ = writeln!(
                o,
                "\nText to align: {} lines (segmentation fixed by the {})",
                thousands(a.stats.lines),
                a.reference
            );
            stats_table(&mut o, &a.stats);
            let _ = writeln!(
                o,
                "  unknown to the {}: source {:.1} %, target {:.1} % of tokens",
                a.reference, a.oov[0], a.oov[1]
            );
            if a.warnings.is_empty() {
                o.push_str("  no warnings\n");
            }
            for w in &a.warnings {
                let _ = writeln!(o, "  warning: {w}");
            }
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitext::Parsed;

    #[test]
    fn rule_at_and_around_thresholds() {
        for (k, want) in [
            (0, 4),
            (1, 4),
            (1_447, 4),
            (29_999, 4),
            (30_000, 5),
            (30_001, 5),
            (48_689, 5),
            (449_999, 5),
            (450_000, 6),
            (450_001, 6),
            (1_130_551, 6),
            (u64::MAX, 6),
        ] {
            assert_eq!(auto_prefix(k), want, "K = {k}");
        }
        assert_eq!(AUTO_THRESHOLDS, [30_000, 450_000]);
        assert_eq!(AUTO_PREFIXES, [4, 5, 6]);
        assert_eq!(rule_fired(29_999), "29,999 lines < 30,000 → prefix 4");
        assert_eq!(
            rule_fired(30_000),
            "30,000 ≤ 30,000 lines < 450,000 → prefix 5"
        );
        assert_eq!(rule_fired(450_000), "450,000 ≤ 450,000 lines → prefix 6");
    }

    #[test]
    fn embedded_evidence_is_the_rule() {
        assert_eq!(EVIDENCE.len(), 27);
        for e in EVIDENCE {
            assert_eq!(
                e.setting,
                format!("prefix:{}", auto_prefix(e.lines)),
                "{}",
                e.corpus
            );
            assert!(e.gdfa_ci.0 <= e.gdfa_delta && e.gdfa_delta <= e.gdfa_ci.1);
            assert!(
                (e.gdfa - e.words_gdfa - e.gdfa_delta).abs() < 0.01,
                "{}",
                e.corpus
            );
            // never worse than words beyond the noise (the rule's admissibility)
            assert!(e.gdfa_ci.0 < 0.0, "{}", e.corpus);
        }
        let names: Vec<_> = PARAMETERS.iter().map(|p| p.option).collect();
        assert_eq!(
            names,
            ["--null-prior", "--lex-alpha", "--length", "--samplers"]
        );
        let defaults: Vec<_> = PARAMETERS.iter().map(|p| p.default).collect();
        assert_eq!(defaults, ["0.2", "0.001", "1", "3"]);
    }

    #[test]
    fn explicit_stems_override_auto_and_words() {
        let o = TextOptions::default();
        let (t, w) = choose(Segment::Auto, &o, 100);
        assert_eq!(
            (t.source_prefix, t.target_prefix, w),
            (Some(4), Some(4), Origin::Auto)
        );
        let (t, w) = choose(Segment::Auto, &o, 1_000_000);
        assert_eq!(
            (t.source_prefix, t.target_prefix, w),
            (Some(6), Some(6), Origin::Auto)
        );
        let (t, w) = choose(Segment::Words, &o, 100);
        assert_eq!((t, w), (o.clone(), Origin::Words));
        for set in [
            |o: &mut TextOptions| o.source_prefix = Some(3),
            |o: &mut TextOptions| o.target_prefix = Some(7),
            |o: &mut TextOptions| o.source_suffix = Some(2),
            |o: &mut TextOptions| o.target_suffix = Some(2),
        ] {
            let mut e = TextOptions::default();
            set(&mut e);
            for seg in [Segment::Auto, Segment::Words] {
                let (t, w) = choose(seg, &e, 100);
                assert_eq!((&t, w), (&e, Origin::Explicit));
            }
        }
    }

    /// Statistics as `corpus_stats.py` computes them: Python's `bytes.split()` and
    /// `str.lower()`, naive counting.
    fn naive(text: &str, lowercase: bool) -> SideStats {
        let mut c: std::collections::HashMap<String, u64> = Default::default();
        let mut tokens = 0;
        for line in text.lines() {
            for w in line.split_ascii_whitespace() {
                tokens += 1;
                let w = if lowercase {
                    w.to_lowercase()
                } else {
                    w.to_string()
                };
                *c.entry(w).or_default() += 1;
            }
        }
        SideStats {
            tokens,
            types: c.len() as u64,
            singletons: c.values().filter(|&&n| n == 1).count() as u64,
        }
    }

    #[test]
    fn statistics_against_naive_count() {
        let src = "The cat sat\nthe CAT  ran away\n\nÉcole école ÉCOLE x\n|||\tA b\n";
        let tgt = "le chat\nLe chat a fui\nrien\nécole\nb B b\n";
        for lowercase in [true, false] {
            for chunks in [None, Some(1), Some(3)] {
                let p = Parsed::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), chunks).unwrap();
                let s = p.stats(lowercase);
                assert_eq!(s.lines, 5);
                assert_eq!(s.lowercase, lowercase);
                assert_eq!(s.source, naive(src, lowercase), "lower {lowercase}");
                assert_eq!(s.target, naive(tgt, lowercase), "lower {lowercase}");
            }
        }
        // random corpora, joined input too
        let mut x: u64 = 7;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let words = ["a", "B", "b", "Cc", "cc", "d", "Ää", "ää", "e", "f"];
        for _ in 0..20 {
            let mut s = String::new();
            let mut t = String::new();
            let mut j = String::new();
            for _ in 0..(next() % 40) {
                let mut line = |out: &mut String| {
                    let n = next() % 6;
                    let ws: Vec<&str> = (0..n).map(|_| words[(next() % 10) as usize]).collect();
                    out.push_str(&ws.join(" "));
                    ws.join(" ")
                };
                let a = line(&mut s);
                let b = line(&mut t);
                s.push('\n');
                t.push('\n');
                j.push_str(&format!("{a} ||| {b}\n"));
            }
            let p = Parsed::from_pair_bytes(s.as_bytes(), t.as_bytes(), Some(2)).unwrap();
            let st = p.stats(true);
            assert_eq!((st.source, st.target), (naive(&s, true), naive(&t, true)));
            let q = Parsed::from_joined_bytes(j.as_bytes(), Some(3)).stats(true);
            assert_eq!(q, st);
        }
    }

    #[test]
    fn training_and_estimate_agree() {
        let src = "a b c\nb d\n";
        let tgt = "x a\ny y z\n";
        let p = Parsed::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), None).unwrap();
        for seg in [Segment::Auto, Segment::Words] {
            for o in [
                TextOptions::default(),
                TextOptions {
                    target_suffix: Some(2),
                    ..TextOptions::default()
                },
            ] {
                let b = p.build_segmented(&o, seg);
                let e = estimate(&p.stats(o.lowercase), &o, seg);
                let mut bo = b.options().clone();
                bo.chunks = None;
                assert_eq!(bo, e.text);
                assert_eq!(b.segmentation().unwrap().origin, e.origin.unwrap());
                assert_eq!(b.segmentation().unwrap().stats, e.stats.unwrap());
            }
        }
    }

    #[test]
    fn nearest_evidence_one_per_pair() {
        let ev = nearest_evidence(48_689);
        let names: Vec<_> = ev.iter().map(|e| e.corpus).collect();
        assert_eq!(names, ["enfr-s50k", "enis-s50k", "roen-medium"]);
        let ev = nearest_evidence(5_000_000);
        let names: Vec<_> = ev.iter().map(|e| e.corpus).collect();
        assert_eq!(names, ["enfr-large", "enis-large", "roen-large"]);
        assert_eq!(nearest_evidence(0).len(), 3);
    }

    #[test]
    fn json_and_text_render() {
        let p = Parsed::from_pair_bytes(b"a b\n\"q\\\n", b"x\ny\n", None).unwrap();
        let mut e = estimate(&p.stats(true), &TextOptions::default(), Segment::Auto);
        let b = p.build(&e.text);
        e.align = Some(align_check(
            "training corpus",
            e.stats.as_ref(),
            &b.source.vocab,
            &b.target.vocab,
            &b,
            p.stats(true),
        ));
        let j = e.json();
        assert!(j.contains("\"source_prefix\": 4"));
        assert!(j.contains("\"oov_share\": {\"source\": 0, \"target\": 0}"));
        let t = e.text();
        assert!(t.contains("prefix 4 on both sides"), "{t}");
        assert!(t.contains("no warnings"), "{t}");
        assert_eq!(json_str("a\"b\\c\n\u{1}"), "\"a\\\"b\\\\c\\n\\u0001\"");
    }

    #[test]
    fn model_without_segmentation_record() {
        let o = TextOptions {
            source_prefix: Some(5),
            target_prefix: Some(5),
            ..TextOptions::default()
        };
        let e = estimate_model(&o, None);
        assert!(e.stats.is_none() && e.origin.is_none() && e.evidence.is_empty());
        let t = e.text();
        assert!(t.contains("no statistics"), "{t}");
        assert!(t.contains("prefix 5 on both sides"), "{t}");
        let j = e.json();
        assert!(j.contains("\"statistics\": null"), "{j}");
        assert!(j.contains("\"origin\": null"), "{j}");
        assert!(j.contains("\"evidence\": [],"), "{j}");
    }

    #[test]
    fn align_check_warns_far_outside() {
        let train = Parsed::from_pair_bytes(b"a b\nb c\n", b"x y\ny z\n", None).unwrap();
        let tb = train.build(&TextOptions::default());
        let new = Parsed::from_pair_bytes(b"q r s t u v w\n", b"x y\n", None).unwrap();
        let nb = new.build(&TextOptions::default());
        let ts = train.stats(true);
        let a = align_check(
            "model",
            Some(&ts),
            &tb.source.vocab,
            &tb.target.vocab,
            &nb,
            new.stats(true),
        );
        assert_eq!(a.oov, [100.0, 0.0]);
        assert_eq!(a.warnings.len(), 2, "{:?}", a.warnings); // unknown words, length
        let a = align_check("model", None, &tb.source.vocab, &tb.target.vocab, &tb, ts);
        assert_eq!(a.oov, [0.0, 0.0]);
        assert_eq!(a.warnings.len(), 1); // no statistics
    }
}
