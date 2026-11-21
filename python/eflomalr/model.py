"""Training, the :class:`Model` class, priors and symmetrisation."""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eflomalr import _core
from eflomalr._input import sentence_to_line, text_source

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from eflomalr._input import Sentence, TextSource

__all__ = [
    "METHODS",
    "SEGMENTS",
    "Alignments",
    "Estimate",
    "Link",
    "Model",
    "Progress",
    "estimate_parameters",
    "invert",
    "make_priors",
    "parse_pharaoh",
    "symmetrize",
    "symmetrize_many",
    "train",
]

logger = logging.getLogger("eflomalr")

#: An alignment link ``(source_position, target_position)``, both 0-based.
Link = tuple[int, int]

#: Sequence of alignments (one list of links per sentence pair); see :meth:`Model.alignments`.
Alignments = _core.Alignments

#: Symmetrisation heuristics, named as in ``atools -c``. ``"intersection"`` is accepted for
#: ``"intersect"``.
METHODS: tuple[str, ...] = (
    "intersect",
    "union",
    "grow-diag",
    "grow-diag-final",
    "grow-diag-final-and",
)

#: Values of ``segment``: ``"auto"`` chooses prefix stemming from the corpus's line count,
#: ``"words"`` does not stem.
SEGMENTS: tuple[str, ...] = ("auto", "words")

#: Default ``max_len``: eflomal skips pairs with 1024 or more tokens on a side.
DEFAULT_MAX_LEN = 1023
#: Largest supported ``max_len``.
MAX_MAX_LEN = 65534


@dataclass(frozen=True)
class Progress:
    """One progress event of :func:`train` or :meth:`Model.align_many`: one sampling iteration of
    one sampler of one direction has finished.

    Attributes:
        direction: ``"fwd"`` (source generates target) or ``"inv"``.
        sampler: 0-based index of the sampler (independent chain) within its direction.
        model: The model this iteration sampled: 1 (IBM1), 2 (+ HMM jumps), 3 (+ fertility).
        iteration: Iteration just finished (1-based, within this model).
        iterations: Iterations of this model planned.
        seconds: Time since this direction started training.
        null_fraction: Fraction of target tokens aligned to NULL after this iteration.
    """

    direction: str
    sampler: int
    model: int
    iteration: int
    iterations: int
    seconds: float
    null_fraction: float

    def __str__(self) -> str:
        # The same line as the native binary prints.
        return (
            f"[{self.direction} sampler {self.sampler + 1}] model {self.model} iteration "
            f"{self.iteration}/{self.iterations}: NULL {100.0 * self.null_fraction:.1f} %, "
            f"{self.seconds:.1f} s"
        )


# -- argument checks -------------------------------------------------------------------------


def _check_int(name: str, v: Any, lo: int, hi: int | None = None, none: bool = False) -> Any:
    if v is None and none:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        rng = f">= {lo}" if hi is None else f"in {lo} ..= {hi}"
        extra = " or None" if none else ""
        raise ValueError(f"{name} must be an integer {rng}{extra}, got {v!r}")
    return v


def _check_threads(threads: int) -> int:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise ValueError(f"threads must be a non-negative integer (0 = all cores), got {threads!r}")
    return threads


def _opt_float(name: str, v: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def _method(name: str | None) -> str | None:
    if name == "intersection":
        return "intersect"
    return name


def _check_direction(direction: str) -> str:
    if direction not in ("fwd", "inv"):
        raise ValueError(f"direction must be 'fwd' or 'inv', not {direction!r}")
    return direction


def _text_options(
    lowercase: bool,
    source_prefix: int | None,
    source_suffix: int | None,
    target_prefix: int | None,
    target_suffix: int | None,
    max_len: int,
) -> tuple:
    for name, v in (
        ("source_prefix", source_prefix),
        ("source_suffix", source_suffix),
        ("target_prefix", target_prefix),
        ("target_suffix", target_suffix),
    ):
        _check_int(name, v, 0, none=True)  # 0 = do not stem that side (as eflomal)
    _check_int("max_len", max_len, 1, MAX_MAX_LEN)
    return (bool(lowercase), source_prefix, source_suffix, target_prefix, target_suffix, max_len)


def _segment(segment: str) -> str:
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be 'auto' or 'words', not {segment!r}")
    return segment


def _iterations(ibm1: int | None, hmm: int | None, fert: int | None) -> tuple:
    return (
        _check_int("ibm1_iters", ibm1, 0, none=True),
        _check_int("hmm_iters", hmm, 0, none=True),
        _check_int("fert_iters", fert, 0, none=True),
    )


def _length(length: float) -> float:
    length = _opt_float("length", length)
    if length < 0:  # 0 = the minimum schedule, as eflomal's -l 0
        raise ValueError(f"length must be >= 0, got {length!r}")
    return length


def _accumulate_limit(mb: int) -> int:
    return _check_int("accumulate_limit", mb, 0, (1 << 44) - 1)


def _progress_hook(callback: Callable[[Progress], None] | None):
    """The function the core calls per iteration, or ``None`` when nobody listens (then the
    core never takes the GIL for progress)."""
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if callback is None and not logger.isEnabledFor(logging.INFO):
        return None

    def on_progress(*args) -> None:
        event = Progress(*args)
        logger.info("%s", event)
        if callback is not None:
            callback(event)

    return on_progress


def _path_arg(v: str | os.PathLike[str] | None, name: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, (str, os.PathLike)):
        raise TypeError(f"{name} must be a file path, not {type(v).__name__}")
    return os.fspath(v)


# -- the model --------------------------------------------------------------------------------


class Model:
    """A trained word alignment model.

    Create one with :func:`train` or :meth:`Model.load`. Per direction (*forward*: source words
    generate target words; *inverse*: the other way round) a model holds the final count
    statistics of the training run (lexical, jump and fertility counts, from the decoded links
    plus any priors the run used), both sides' vocabularies, and the options it was trained
    with, including lowercasing, stemming and ``max_len``.

    A model that has just been trained also holds the alignments of its training corpus
    (:meth:`alignments`, :meth:`write_alignments`). A model read back with :meth:`load` does
    not; use :meth:`align` and :meth:`align_many` to align new sentences with it: they sample
    the new text with the model's counts as priors (eflomal's own method), or, with
    ``fast=True``, decode it without sampling.

    Instances are immutable and may be shared between threads.
    """

    def __init__(self, _inner: _core.Model) -> None:
        self._inner = _inner

    # -- introspection ------------------------------------------------------------------

    @property
    def directions(self) -> tuple[str, ...]:
        """The directions this model holds: ``("fwd", "inv")``, ``("fwd",)`` or ``("inv",)``."""
        return tuple(self._inner.directions)

    @property
    def vocab_size(self) -> tuple[int, int]:
        """Number of distinct words of the source and of the target side, after lowercasing
        and stemming (each side has its own vocabulary)."""
        return self._inner.vocab_size

    @property
    def has_training_alignments(self) -> bool:
        """Whether :meth:`alignments` is available (true right after :func:`train`)."""
        return self._inner.has_training_alignments

    @property
    def n_lines(self) -> int | None:
        """Number of lines of the training corpus, or ``None`` for a loaded model."""
        return self._inner.n_lines

    @property
    def skipped(self) -> list[int] | None:
        """0-based indices of the training pairs skipped for length (they have empty
        alignments), or ``None`` for a loaded model."""
        return self._inner.skipped

    @property
    def training_warnings(self) -> list[str]:
        """What :func:`train` warned about (also issued as :class:`RuntimeWarning`): nothing
        to align (no line pairs, or none with words on both sides). Empty for a loaded model
        or an ordinary run."""
        return self._inner.training_warnings

    @property
    def iterations(self) -> tuple[int, int, int] | None:
        """Iterations of Model 1, 2 and 3 that the training run sampled, or ``None`` for a
        loaded model."""
        return self._inner.iterations

    @property
    def options(self) -> dict[str, Any]:
        """The options the model was trained with: ``model``, ``samplers``, ``seed``,
        ``lex_alpha``, ``jump_alpha``, ``fert_alpha``, ``null_prior``, ``lowercase``,
        ``source_prefix``, ``source_suffix``, ``target_prefix``, ``target_suffix`` and
        ``max_len``. :meth:`align_many` reads new text with the same lowercasing, stemming and
        ``max_len``."""
        model, samplers, seed, la, ja, fa, npr, text = self._inner.options
        lowercase, sp, ss, tp, ts, max_len = text
        return {
            "model": model,
            "samplers": samplers,
            "seed": seed,
            "lex_alpha": la,
            "jump_alpha": ja,
            "fert_alpha": fa,
            "null_prior": npr,
            "lowercase": lowercase,
            "source_prefix": sp,
            "source_suffix": ss,
            "target_prefix": tp,
            "target_suffix": ts,
            "max_len": max_len,
        }

    @property
    def segmentation(self) -> dict[str, Any] | None:
        """How the stemming in :attr:`options` was chosen, and the statistics of the training
        corpus it was chosen from, or ``None`` for a model saved without them.

        ``origin`` is ``"auto"`` (the line-count rule), ``"words"`` (``segment="words"``) or
        ``"explicit"`` (stemming options given); ``lines`` is the number of input lines;
        ``source`` and ``target`` hold ``tokens``, ``types`` and ``singletons`` (types after
        lowercasing if ``lowercase``)."""
        seg = self._inner.segmentation
        if seg is None:
            return None
        origin, lowercase, lines, src, tgt = seg

        def side(t: tuple[int, int, int]) -> dict[str, int]:
            return {"tokens": t[0], "types": t[1], "singletons": t[2]}

        return {
            "origin": origin,
            "lowercase": lowercase,
            "lines": lines,
            "source": side(src),
            "target": side(tgt),
        }

    def training_log(self, direction: str = "fwd") -> str:
        """The progress lines of one direction of the training run (one per iteration and
        sampler, as the native binary's ``--log`` writes them)."""
        raw = self._inner.training_log(_check_direction(direction))
        return raw.decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return (
            f"<eflomalr.Model directions={self.directions!r} model={self.options['model']} "
            f"vocab={self.vocab_size}>"
        )

    # -- persistence --------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the model to ``path`` through a uniquely named temporary file in the same
        directory, renamed over ``path`` only when complete: an existing file is replaced by a
        complete model or left as it was, and no other file is touched. Releases the GIL while
        writing. The file is the native binary's ``--save`` format.

        Raises:
            OSError: the file cannot be written.
        """
        self._inner.save(os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Model:
        """Read a model written by :meth:`save` or ``eflomalr train --save``.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: the file is not an eflomalr model, is truncated or corrupt, or has an
                unsupported format version.
        """
        return cls(_core.Model.load(os.fspath(path)))

    # -- alignments ---------------------------------------------------------------------

    def alignments(self, symmetrize: str | None = None, direction: str | None = None) -> Alignments:
        """Alignments of the training corpus, one entry per input line.

        The result behaves like a read-only list: ``len()``, indexing, slicing and iteration
        give lists of ``(source, target)`` links (0-based positions); ``.pharaoh(i)`` gives the
        text of line ``i``, ``.score(i)`` its score (with ``scores=True``). Lines without
        alignment (an empty side, or skipped by ``max_len``) give empty lists. Links of one
        direction come in target order, symmetrised links sorted.

        Args:
            symmetrize: One of :data:`METHODS`; needs a model of both directions.
            direction: ``"fwd"`` or ``"inv"``: the links of one direction without
                symmetrisation.

        With neither argument, a model of both directions is symmetrised with
        ``grow-diag-final-and`` and a model of one direction returns that direction.

        Raises:
            ValueError: the model was loaded from a file (no training alignments), both
                arguments were given, or the requested direction is not in the model.
        """
        return self._inner.training_alignments(_method(symmetrize), direction)

    def write_alignments(
        self,
        path: str | os.PathLike[str],
        format: str = "pharaoh",
        symmetrize: str | None = None,
        direction: str | None = None,
    ) -> None:
        """Write the training alignments to ``path``: one line of ``i-j`` links per input line
        (source-target orientation), the format ``atools`` reads and writes. If the model was
        trained with ``scores=True``, the lines of a single direction end with ``||| score``.

        Args:
            path: Output file.
            format: ``"pharaoh"`` (the only format).
            symmetrize, direction: As for :meth:`alignments`.
        """
        if format != "pharaoh":
            raise ValueError(f"format must be 'pharaoh', not {format!r}")
        self._inner.write_training_alignments(os.fspath(path), _method(symmetrize), direction)

    def align(
        self,
        src: Sentence,
        tgt: Sentence,
        symmetrize: str | None = None,
        direction: str | None = None,
        *,
        fast: bool = False,
        seed: int | None = None,
    ) -> list[Link]:
        """Align one sentence pair with the model and return its links.

        Args:
            src, tgt: The sentences, as lists of tokens or as strings split on whitespace.
            symmetrize, direction: As for :meth:`alignments`.
            fast: Decode without sampling (see :meth:`align_many`).
            seed: Seed of the sampler (default: the model's).

        This is :meth:`align_many` on a one-line corpus, so the result can differ from that
        pair's line in a larger :meth:`align_many` call: with sampling the sentences aligned
        together share the counts and the iteration schedule (which follows the number of
        lines: a single pair gets thousands of cheap iterations, about 0.3 s), and in both
        modes the size of the new text's vocabulary enters the priors. To align many
        sentences, use :meth:`align_many`. A pair with an empty side gives ``[]`` (no warning).
        """
        with warnings.catch_warnings():
            # "nothing to align" is the answer here, not something to warn about
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.align_many(
                [sentence_to_line(src, "src")],
                [sentence_to_line(tgt, "tgt")],
                symmetrize,
                direction,
                fast=fast,
                seed=seed,
            )[0]

    def align_many(
        self,
        src: TextSource,
        tgt: TextSource | None = None,
        symmetrize: str | None = None,
        direction: str | None = None,
        threads: int = 0,
        scores: bool = False,
        *,
        fast: bool = False,
        seed: int | None = None,
        samplers: int | None = None,
        length: float = 1.0,
        ibm1_iters: int | None = None,
        hmm_iters: int | None = None,
        fert_iters: int | None = None,
        exact: bool = False,
        accumulate: bool = False,
        accumulate_limit: int = 4096,
        callback: Callable[[Progress], None] | None = None,
    ) -> Alignments:
        """Align new sentence pairs with the model.

        By default the new text is *sampled* as in training, with the model's counts added to
        the priors (eflomal's way of aligning new text with knowledge from a previous corpus;
        the same as ``eflomalr align``). The text is lowercased, stemmed and length-filtered as
        the training corpus was, and the model's options (model, hyperparameters, samplers,
        seed) are used unless given here. The iteration schedule follows the size of the new
        text.

        Args:
            src, tgt: Line-aligned text files, or lists of sentences (strings or token lists)
                of equal length. With ``tgt`` omitted, ``src`` is one file of
                ``source ||| target`` lines.
            symmetrize, direction: As for :meth:`alignments`.
            threads: Worker threads, 0 for all cores. The output never depends on it.
            scores: Append ``||| score`` to every line of the written text; needs
                ``direction``. The links you read from the result are the same.
            fast: Decode deterministically without sampling: per sentence the exact posterior
                of the Model-2 HMM with the model's counts as parameters (``align --fast``).
                Much faster; no seed.
            seed, samplers: Override the model's.
            length, ibm1_iters, hmm_iters, fert_iters, exact, accumulate, accumulate_limit:
                As for :func:`train`.
            callback: As for :func:`train`.

        A model trained on a large corpus has large counts: they can dominate the counts of a
        large new text and make its alignments worse than training on that text. Use this for
        small or medium amounts of new text, and train afresh on large ones.

        The GIL is released while aligning and Ctrl-C interrupts the call. Lines with an empty
        side give empty alignments; a text with nothing to align (no line pairs, or none with
        words on both sides) gives a :class:`RuntimeWarning`, not an error.
        """
        if fast and scores:
            raise ValueError("fast decoding has no scores")
        if seed is not None:
            _check_int("seed", seed, 0, (1 << 64) - 1)
        if samplers is not None:
            _check_int("samplers", samplers, 1, 255)
        sampling = (
            seed,
            samplers,
            _length(length),
            _iterations(ibm1_iters, hmm_iters, fert_iters),
            bool(exact),
            bool(accumulate),
            _accumulate_limit(accumulate_limit),
        )
        return self._inner.align_many(
            text_source(src, "src"),
            None if tgt is None else text_source(tgt, "tgt"),
            _method(symmetrize),
            direction,
            _check_threads(threads),
            bool(scores),
            bool(fast),
            sampling,
            None if fast else _progress_hook(callback),
        )


# -- training ---------------------------------------------------------------------------------


def train(
    src: TextSource,
    tgt: TextSource | None = None,
    threads: int = 0,
    *,
    directions: str = "both",
    seed: int = 1,
    model: int = 3,
    samplers: int = 3,
    length: float = 1.0,
    ibm1_iters: int | None = None,
    hmm_iters: int | None = None,
    fert_iters: int | None = None,
    null_prior: float = 0.2,
    lex_alpha: float = 0.001,
    jump_alpha: float = 0.5,
    fert_alpha: float = 1.0,
    priors: str | os.PathLike[str] | None = None,
    lowercase: bool = True,
    source_prefix: int | None = None,
    source_suffix: int | None = None,
    target_prefix: int | None = None,
    target_suffix: int | None = None,
    segment: str = "auto",
    max_len: int = DEFAULT_MAX_LEN,
    scores: bool = False,
    score_model: int | None = None,
    exact: bool = False,
    accumulate: bool = False,
    accumulate_limit: int = 4096,
    callback: Callable[[Progress], None] | None = None,
) -> Model:
    """Train eflomal's Bayesian alignment model by collapsed Gibbs sampling.

    The defaults are eflomal's: Model 3 (IBM1, then the HMM jump model, then fertility), three
    independent samplers per direction, NULL prior 0.2, and iteration counts derived from the
    corpus size, plus prefix stemming chosen from the corpus size (``segment="auto"``; see
    :func:`estimate_parameters`). Both directions are trained concurrently from one corpus load. The output
    depends only on the corpus, the options and ``seed``, never on ``threads``; it is
    byte-identical to ``eflomalr train`` with the same options.

    Args:
        src, tgt: The two sides of the corpus: paths of line-aligned text files, or lists of
            sentences (each a string split on whitespace, or a list of tokens) of equal length.
            With ``tgt`` omitted, ``src`` is the path of one file of ``source ||| target``
            lines. A plain ``str`` is always a file path.
        threads: Worker threads; 0 uses all cores.
        directions: ``"both"``, ``"fwd"`` (source generates target) or ``"inv"``. Symmetrising
            needs both.
        seed: Seed of the random number generator.
        model: 1 = IBM1, 2 = IBM1 + HMM, 3 = IBM1 + HMM + fertility (eflomal ``-m``).
        samplers: Independent samplers per direction (eflomal ``--n-samplers``); their link
            marginals are averaged.
        length: Relative number of iterations (eflomal ``-l``); the counts follow the corpus
            size.
        ibm1_iters, hmm_iters, fert_iters: Iterations of each model (eflomal ``-1 -2 -3``);
            each overrides its own stage of the automatic schedule.
        null_prior: Prior probability of a NULL alignment (eflomal ``--null-prior``).
        lex_alpha, jump_alpha, fert_alpha: Dirichlet priors of the lexical, jump and fertility
            distributions.
        priors: An eflomal priors file (:func:`make_priors`, ``eflomalr makepriors`` or
            eflomal-makepriors); its counts are added to the Dirichlet priors (eflomal ``-p``).
        lowercase: Lowercase tokens first (eflomal always does). ``False`` keeps case.
        source_prefix, source_suffix, target_prefix, target_suffix: Stem each token to its
            first / last N characters (eflomal ``--source-prefix`` etc.). Any of them switches
            ``segment="auto"`` off; only the given ones apply.
        segment: ``"auto"`` (default) stems both sides to their first 4 characters below
            30,000 input lines, 5 below 450,000 and 6 above, the rule measured best on 27
            corpora (never worse than plain words); ``"words"`` does not stem. The choice is
            saved with the model (:attr:`Model.options`, :attr:`Model.segmentation`), so
            :meth:`Model.align_many` reads new text the same way.
        max_len: Pairs with more tokens than this on either side are not aligned (empty line;
            listed in :attr:`Model.skipped`). eflomal skips pairs of 1024 tokens or more.
        scores: Append ``||| score`` (negative mean log probability per target token) to the
            lines of each direction's alignments.
        score_model: Model used for the scores (eflomal ``-M``; default ``model``).
        exact: Sample every chain with the exact sequential sampler (slower). The default
            samples each chain in parallel blocks, with a different but equally deterministic
            result.
        accumulate: Decode from the conditionals of every sweep of the last model plus the
            final pass instead of the final pass alone. Often a little better, but needs
            ``Σ J·(I+1)`` floats per sampler and direction.
        accumulate_limit: Refuse ``accumulate`` above this many MB.
        callback: Called with a :class:`Progress` after every iteration of every sampler (from
            a worker thread; samplers and directions run concurrently). An exception raised by
            it stops training and propagates. Progress is also logged to the ``eflomalr``
            logger at INFO level.

    Returns:
        The trained :class:`Model`, which also holds the alignments of the training corpus.

    Raises:
        ValueError: bad arguments (mismatched corpus sizes, a token with whitespace, a bad
            priors file, ...).
        FileNotFoundError, OSError: unreadable input files.
        KeyboardInterrupt: Ctrl-C. Training stops within a fraction of a second.

    Warns:
        RuntimeWarning: nothing to align: the corpus has no line pairs, or no pair has words
            on both sides within ``max_len``. Training still succeeds, with one empty alignment
            per input line (none for an empty corpus). :attr:`Model.training_warnings` keeps
            the messages.

    The GIL is released for the whole run, so other Python threads keep running.
    """
    if directions not in ("both", "fwd", "inv"):
        raise ValueError(f"directions must be 'both', 'fwd' or 'inv', not {directions!r}")
    _check_int("seed", seed, 0, (1 << 64) - 1)
    _check_int("model", model, 1, 3)
    _check_int("samplers", samplers, 1, 255)
    _check_int("score_model", score_model, 1, 3, none=True)
    hyper = (
        _opt_float("lex_alpha", lex_alpha),
        _opt_float("jump_alpha", jump_alpha),
        _opt_float("fert_alpha", fert_alpha),
        _opt_float("null_prior", null_prior),
    )
    text = _text_options(
        lowercase, source_prefix, source_suffix, target_prefix, target_suffix, max_len
    )
    inner = _core.train(
        text_source(src, "src"),
        None if tgt is None else text_source(tgt, "tgt"),
        _check_threads(threads),
        directions,
        seed,
        model,
        samplers,
        _length(length),
        _iterations(ibm1_iters, hmm_iters, fert_iters),
        hyper,
        _path_arg(priors, "priors"),
        text,
        bool(scores),
        score_model,
        bool(exact),
        bool(accumulate),
        _accumulate_limit(accumulate_limit),
        _segment(segment),
        _progress_hook(callback),
    )
    for message in inner.training_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return Model(inner)


# -- estimate ---------------------------------------------------------------------------------


class Estimate(dict):
    """The report of :func:`estimate_parameters`: a ``dict`` (the native ``eflomalr estimate
    --json`` object) whose ``str()`` is the human-readable report of ``eflomalr estimate``.

    Keys: ``statistics`` (``lines``, ``lowercase`` and per side ``tokens``, ``types``,
    ``singletons``, ``tokens_per_type``, ``singleton_share`` in percent, ``mean_length``),
    ``segmentation`` (``segment``, ``origin``, ``source_prefix`` ... ``target_suffix``,
    ``auto_prefix``, ``rule_fired``, ``rule``, ``thresholds``, ``prefixes``), ``evidence``
    (the nearest measured corpus of each language pair: the rule's pick against words with
    its gdfa/fwd/inv AER difference and 95 % bootstrap interval), ``parameters`` (each sampler
    parameter's default, the measured verdict and, for ``samplers``, the trade-off) and
    ``align`` (``None``, or the text to align against the reference corpus: its statistics,
    the share of unknown tokens and warnings).
    """

    def __init__(self, data: str, text: str) -> None:
        super().__init__(json.loads(data))
        self._json = data
        self._text = text

    def __str__(self) -> str:
        return self._text

    def to_json(self) -> str:
        """The JSON text, exactly as ``eflomalr estimate --json`` writes it."""
        return self._json


def estimate_parameters(
    src: TextSource,
    tgt: TextSource | None = None,
    *,
    align_src: TextSource | None = None,
    align_tgt: TextSource | None = None,
    model: Model | str | os.PathLike[str] | None = None,
    segment: str = "auto",
    lowercase: bool = True,
    source_prefix: int | None = None,
    source_suffix: int | None = None,
    target_prefix: int | None = None,
    target_suffix: int | None = None,
    threads: int = 0,
) -> Estimate:
    """Report what :func:`train` would choose for a corpus, and why. Changes nothing.

    Returns the corpus's per-side statistics, the segmentation :func:`train` uses with the same
    ``segment`` and stemming options (it calls the same function, so the two always agree) and
    the rule that fired, the measured evidence of the nearest measured corpora, and the sampler
    parameters (``null_prior``, ``lex_alpha``, ``length``, ``samplers``): their defaults and
    what the measurements say. No parameter default is changed by this.

    Args:
        src, tgt: The training corpus, as for :func:`train`. With ``model``, the text to be
            aligned with that model instead.
        align_src, align_tgt: Text to be aligned later with a model trained on ``src``/``tgt``.
            The training corpus fixes the segmentation; the report compares the new text with
            it (unknown tokens, sentence lengths) and warns when it is far outside it.
        model: A :class:`Model` or the path of a saved one: report its segmentation and compare
            ``src``/``tgt`` with its training corpus.
        segment, lowercase, source_prefix, source_suffix, target_prefix, target_suffix: As for
            :func:`train` (ignored with ``model``).
        threads: Worker threads; 0 uses all cores.

    Returns:
        An :class:`Estimate`: a dict (the ``--json`` output) that prints as the text report.
    """
    if model is not None and (align_src is not None or align_tgt is not None):
        raise ValueError("give either model= or align_src=, not both")
    if align_tgt is not None and align_src is None:
        raise ValueError("align_tgt needs align_src")
    text = _text_options(
        lowercase, source_prefix, source_suffix, target_prefix, target_suffix, DEFAULT_MAX_LEN
    )
    inner = None
    if model is not None:
        inner = (model if isinstance(model, Model) else Model.load(model))._inner

    def opt(v: TextSource | None, name: str) -> Any:
        return None if v is None else text_source(v, name)

    js, report = _core.estimate(
        text_source(src, "src"),
        opt(tgt, "tgt"),
        opt(align_src, "align_src"),
        opt(align_tgt, "align_tgt"),
        inner,
        _segment(segment),
        text,
        _check_threads(threads),
    )
    return Estimate(js, report)


# -- priors -----------------------------------------------------------------------------------


def _links_text(links: Any, name: str) -> bytes:
    """Pharaoh text from a file path, an :class:`Alignments` or a list of link lists."""
    if isinstance(links, (str, os.PathLike)):
        with open(links, "rb") as f:
            return f.read()
    if isinstance(links, _core.Alignments):
        return links.to_bytes()
    if isinstance(links, (bytes, bytearray)):
        raise TypeError(f"{name} must be a file path, an Alignments or a list of link lists")
    try:
        return "".join(" ".join(f"{i}-{j}" for i, j in ls) + "\n" for ls in links).encode()
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a file path, an Alignments or a list of link lists"
        ) from None


def make_priors(
    src: TextSource,
    tgt: TextSource | None = None,
    *,
    forward: Any,
    reverse: Any,
    output: str | os.PathLike[str] | None = None,
    lowercase: bool = True,
    source_prefix: int | None = None,
    source_suffix: int | None = None,
    target_prefix: int | None = None,
    target_suffix: int | None = None,
    threads: int = 0,
) -> bytes:
    """Build an eflomal priors file from an aligned corpus (eflomal-makepriors).

    Args:
        src, tgt: The corpus, as for :func:`train`.
        forward: Forward links (lexical, forward jump and fertility priors): a Pharaoh file,
            an :class:`Alignments` (e.g. ``model.alignments(direction="fwd")``) or a list of
            link lists, in source-target order.
        reverse: Reverse links (reverse jump and fertility priors), likewise; may be the same
            symmetrised alignments as ``forward``.
        output: Also write the file here.
        lowercase, source_prefix, ...: How the words are normalised (as for :func:`train`;
            training with the priors normalises them again with its own options).
        threads: Worker threads, 0 for all cores.

    Returns:
        The priors file's bytes (``LEX``, ``HMMF``, ``HMMR``, ``FERF``, ``FERR`` records),
        readable by :func:`train` (``priors=``) and by eflomal. Every pair counts, whatever its
        length.
    """
    text = _text_options(
        lowercase, source_prefix, source_suffix, target_prefix, target_suffix, DEFAULT_MAX_LEN
    )
    data = _core.make_priors(
        text_source(src, "src"),
        None if tgt is None else text_source(tgt, "tgt"),
        _links_text(forward, "forward"),
        _links_text(reverse, "reverse"),
        text,
        _check_threads(threads),
    )
    if output is not None:
        with open(output, "wb") as f:
            f.write(data)
    return data


# -- symmetrisation -------------------------------------------------------------------------


def symmetrize(
    fwd: Sequence[Link],
    inv: Sequence[Link],
    method: str = "grow-diag-final-and",
) -> list[Link]:
    """Symmetrise the two directional alignments of one sentence pair.

    Both ``fwd`` and ``inv`` are lists of ``(source, target)`` links. The result is sorted and
    identical to what ``atools -c METHOD`` gives for the two lines (atools needs no sentence
    lengths).
    """
    return symmetrize_many([fwd], [inv], method)[0]


def symmetrize_many(
    fwd: Sequence[Sequence[Link]],
    inv: Sequence[Sequence[Link]],
    method: str = "grow-diag-final-and",
) -> list[list[Link]]:
    """Symmetrise many sentence pairs; see :func:`symmetrize`."""
    if len(fwd) != len(inv):
        raise ValueError(f"fwd and inv must have the same size ({len(fwd)} and {len(inv)})")
    return _core.symmetrize_many(
        [list(map(tuple, f)) for f in fwd],
        [list(map(tuple, i)) for i in inv],
        _method(method),
    )


def invert(links: Sequence[Link]) -> list[Link]:
    """Swap the two sides of an alignment (``atools -c invert``); the result is sorted."""
    return _core.invert_many([list(map(tuple, links))])[0]


def parse_pharaoh(line: str) -> list[Link]:
    """Parse a Pharaoh alignment line such as ``"0-0 1-2"`` into links (a ``||| score``
    suffix is ignored)."""
    return _core.parse_pharaoh(line)
