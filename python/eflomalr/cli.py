"""Command-line interface: ``eflomalr train | align | symmetrize | makepriors | estimate``.

Same subcommands and options as the native ``eflomalr`` binary built from eflomal-core, and the
same output bytes for the same options and seed.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings

import eflomalr
from eflomalr import _core
from eflomalr.model import DEFAULT_MAX_LEN, METHODS, SEGMENTS, _method

#: ``intersection`` is accepted for ``intersect`` (the name GIZA++-style tools use).
_METHOD_CHOICES = (*METHODS, "intersection")

logger = logging.getLogger("eflomalr")


def _add_threads(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="worker threads, 0 = all cores (the output never depends on it)",
    )


def _add_text_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group(
        "text", "How tokens are normalised before alignment (also applied to priors words)."
    )
    g.add_argument(
        "--no-lowercase", action="store_true", help="keep the case of tokens (eflomal lowercases)"
    )
    for side in ("source", "target"):
        g.add_argument(
            f"--{side}-prefix",
            type=int,
            metavar="N",
            help=f"stem {side} tokens to their first N characters (0 = do not stem)",
        )
        g.add_argument(
            f"--{side}-suffix",
            type=int,
            metavar="N",
            help=f"stem {side} tokens to their last N characters, after the prefix",
        )
    g.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="pairs with more tokens than this on either side are not aligned (empty line); "
        "default %(default)s, as eflomal",
    )


def _add_segment(p: argparse.ArgumentParser, help_extra: str = "") -> None:
    p.add_argument(
        "--segment",
        choices=SEGMENTS,
        default="auto",
        help="stemming: `auto` chooses a prefix length on both sides from the corpus's line "
        "count (4 below 30,000 lines, 5 below 450,000, else 6); `words` does not stem. Explicit "
        "--source-prefix/--target-prefix/--source-suffix/--target-suffix options override "
        "both" + help_extra,
    )


def _text_kwargs(args: argparse.Namespace) -> dict:
    return {
        "lowercase": not args.no_lowercase,
        "source_prefix": args.source_prefix,
        "source_suffix": args.source_suffix,
        "target_prefix": args.target_prefix,
        "target_suffix": args.target_suffix,
    }


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("sampling")
    g.add_argument(
        "--seed",
        type=int,
        help="seed of the random number generator; the same seed gives the same output at any "
        "thread count (train: default 1; align: default the model's)",
    )
    g.add_argument(
        "-n",
        "--samplers",
        "--n-samplers",
        dest="samplers",
        type=int,
        help="independent samplers per direction (train: default 3; align: default the model's)",
    )
    g.add_argument(
        "-l",
        "--length",
        type=float,
        default=1.0,
        help="relative number of iterations (eflomal -l); the counts follow the corpus size "
        "(0 = the minimum)",
    )
    g.add_argument(
        "-1",
        "--ibm1-iters",
        dest="ibm1_iters",
        type=int,
        metavar="N",
        help="Model 1 iterations (default from the corpus size; eflomal -1)",
    )
    g.add_argument(
        "-2",
        "--hmm-iters",
        dest="hmm_iters",
        type=int,
        metavar="N",
        help="Model 2 (HMM) iterations (eflomal -2)",
    )
    g.add_argument(
        "-3",
        "--fert-iters",
        dest="fert_iters",
        type=int,
        metavar="N",
        help="Model 3 (fertility) iterations (eflomal -3)",
    )
    g.add_argument(
        "--exact",
        action="store_true",
        help="use the exact sequential sampler for every chain (slower)",
    )
    g.add_argument(
        "--accumulate",
        action="store_true",
        help="decode from the conditionals of every sweep of the last model plus the final pass",
    )
    g.add_argument(
        "--accumulate-limit",
        type=int,
        default=None,
        metavar="MB",
        help="memory limit of --accumulate in MB (default 4096)",
    )


def _sampling_kwargs(args: argparse.Namespace) -> dict:
    if args.accumulate_limit is not None and not args.accumulate:
        raise ValueError("--accumulate-limit needs --accumulate")
    return {
        "samplers": args.samplers,
        "length": args.length,
        "ibm1_iters": args.ibm1_iters,
        "hmm_iters": args.hmm_iters,
        "fert_iters": args.fert_iters,
        "exact": args.exact,
        "accumulate": args.accumulate,
        "accumulate_limit": 4096 if args.accumulate_limit is None else args.accumulate_limit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eflomalr",
        description="Bayesian word alignment by collapsed Gibbs sampling (eflomal's models).",
    )
    parser.add_argument("-V", "--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="train on a parallel corpus and write alignments")
    t.add_argument(
        "input",
        help="corpus: one file of `source ||| target` lines, or the source text (then give TGT)",
    )
    t.add_argument("tgt", nargs="?", help="target text, line-aligned with INPUT")
    t.add_argument(
        "-o",
        "--output",
        help="output prefix: writes PREFIX.fwd.pharaoh, PREFIX.inv.pharaoh and "
        "PREFIX.<method>.pharaoh for every --symmetrize; without it the single direction "
        "chosen by --directions goes to standard output",
    )
    _add_threads(t)
    t.add_argument(
        "--directions",
        choices=["both", "fwd", "inv"],
        default="both",
        help="directions to train: both, or only forward or inverse; links are in source-target "
        "order either way (default %(default)s)",
    )
    t.add_argument(
        "-m",
        "--model",
        type=int,
        default=3,
        help="1 = IBM1, 2 = IBM1 + HMM, 3 = IBM1 + HMM + fertility (default %(default)s)",
    )
    _add_sampling_args(t)
    t.add_argument(
        "--null-prior",
        type=float,
        default=0.2,
        help="prior probability of a NULL alignment, 1e-6 ..= 1 - 1e-6 (default %(default)s)",
    )
    t.add_argument(
        "--lex-alpha",
        type=float,
        default=0.001,
        help="lexical Dirichlet prior (every alpha: 1e-30 ..= 1e25)",
    )
    t.add_argument("--jump-alpha", type=float, default=0.5, help="jump Dirichlet prior")
    t.add_argument("--fert-alpha", type=float, default=1.0, help="fertility Dirichlet prior")
    t.add_argument(
        "-p", "--priors", help="eflomal priors file; its counts are added to the Dirichlet priors"
    )
    _add_text_args(t)
    _add_segment(t, ". `eflomalr estimate` shows the choice and the measurements behind it")
    t.add_argument(
        "-s",
        "--symmetrize",
        action="append",
        default=[],
        choices=_METHOD_CHOICES,
        metavar="METHOD",
        help="symmetrisation to write (repeatable): " + ", ".join(METHODS),
    )
    t.add_argument(
        "--scores",
        action="store_true",
        help="append ` ||| score` (negative mean log probability per target token) to the "
        "directional lines",
    )
    t.add_argument("-M", "--score-model", type=int, metavar="N", help="model used for the scores")
    t.add_argument("--save", help="save the trained model here (for `eflomalr align`)")
    t.add_argument(
        "--log",
        action="store_true",
        help="also write each direction's progress lines to PREFIX.<dir>.log; needs --output",
    )
    t.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    a = sub.add_parser(
        "align", help="align new sentence pairs with a saved model (sampling with it as priors)"
    )
    a.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="MODEL SRC [TGT]; SRC alone is a `source ||| target` file. The text is lowercased, "
        "stemmed and length-filtered as the model's training corpus was",
    )
    a.add_argument("-o", "--output", help="output file (default: standard output)")
    a.add_argument(
        "-s",
        "--symmetrize",
        choices=_METHOD_CHOICES,
        metavar="METHOD",
        help="symmetrisation (needs a model of both directions); otherwise one direction",
    )
    a.add_argument(
        "--direction",
        choices=["fwd", "inv"],
        default="fwd",
        help="direction written without --symmetrize (default %(default)s)",
    )
    a.add_argument("--scores", action="store_true", help="append ` ||| score` to each line")
    _add_threads(a)
    _add_sampling_args(a)
    a.add_argument(
        "--fast",
        action="store_true",
        help="deterministic decode without sampling: the posterior of the Model-2 HMM with the "
        "model's counts as parameters",
    )
    a.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    s = sub.add_parser(
        "symmetrize", help="symmetrise two directional alignment files (as atools -c METHOD)"
    )
    s.add_argument("fwd", help="forward alignments (Pharaoh), one line per sentence pair")
    s.add_argument(
        "inv", nargs="?", help="inverse alignments (source-target orientation); not for invert"
    )
    s.add_argument(
        "-m",
        "--method",
        choices=[*_METHOD_CHOICES, "invert"],
        default="grow-diag-final-and",
        help="symmetrisation heuristic, or invert (swap the sides of FWD); default %(default)s",
    )
    s.add_argument("-o", "--output", help="output file (default: standard output)")

    mp = sub.add_parser("makepriors", help="build an eflomal priors file from an aligned corpus")
    mp.add_argument(
        "input",
        help="corpus: one file of `source ||| target` lines, or the source text (then give TGT)",
    )
    mp.add_argument("tgt", nargs="?", help="target text, line-aligned with INPUT")
    mp.add_argument(
        "-f",
        "--forward-links",
        required=True,
        help="forward links (Pharaoh, source-target order): lexical, forward jump and "
        "fertility priors",
    )
    mp.add_argument(
        "-r",
        "--reverse-links",
        required=True,
        help="reverse links (may be the same symmetrised file): reverse jump and fertility priors",
    )
    mp.add_argument("-o", "-p", "--output", help="priors file to write (default: standard output)")
    _add_threads(mp)
    _add_text_args(mp)

    e = sub.add_parser(
        "estimate",
        help="report a corpus's statistics, the segmentation `train` would choose and why, the "
        "measured evidence, and the sampler parameters (changes nothing)",
    )
    e.add_argument(
        "input",
        help="training corpus: one file of `source ||| target` lines, or the source text (then "
        "give TGT); with --model, the text to align instead",
    )
    e.add_argument("tgt", nargs="?", help="target text, line-aligned with INPUT")
    e.add_argument(
        "--align-src",
        metavar="FILE",
        help="text to be aligned later with a model trained on INPUT (`source ||| target` file, "
        "or the source side with --align-tgt): compared with the training corpus",
    )
    e.add_argument("--align-tgt", metavar="FILE", help="target side of --align-src")
    e.add_argument(
        "--model",
        metavar="MODEL",
        help="a saved model: report its segmentation (its training corpus fixed it) and compare "
        "INPUT [TGT], the text to align, with its training corpus",
    )
    _add_segment(e, " (as `train --segment`)")
    _add_text_args(e)
    e.add_argument("--json", action="store_true", help="machine-readable output (JSON)")
    e.add_argument("-o", "--output", help="output file (default: standard output)")
    _add_threads(e)
    return parser


def _write_bytes(path: str | None, data: bytes) -> None:
    if path:
        with open(path, "wb") as f:
            f.write(data)
    else:
        sys.stdout.flush()
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def _read_bytes(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    with open(path, "rb") as f:
        return f.read()


@contextlib.contextmanager
def _input_path(path: str, tgt: str | None):
    """A joined corpus read from standard input (``-``) is spooled to a temporary file, so that
    the core reads exactly the same bytes as the native binary."""
    if path != "-" or tgt is not None:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="eflomalr-") as d:
        p = os.path.join(d, "stdin")
        with open(p, "wb") as f:
            shutil.copyfileobj(sys.stdin.buffer, f)
        yield p


def _cmd_train(args: argparse.Namespace) -> None:
    both = args.directions == "both"
    if args.symmetrize and (args.output is None or not both):
        raise ValueError("--symmetrize needs --output and both directions")
    if args.output is None and both:
        raise ValueError("without --output choose one direction with --directions fwd|inv")
    if args.log and args.output is None:
        raise ValueError("--log needs --output PREFIX")
    t0 = time.perf_counter()
    with _input_path(args.input, args.tgt) as src:
        model = eflomalr.train(
            src,
            args.tgt,
            threads=args.threads,
            directions=args.directions,
            seed=1 if args.seed is None else args.seed,
            model=args.model,
            null_prior=args.null_prior,
            lex_alpha=args.lex_alpha,
            jump_alpha=args.jump_alpha,
            fert_alpha=args.fert_alpha,
            priors=args.priors,
            max_len=args.max_len,
            scores=args.scores,
            score_model=args.score_model,
            segment=args.segment,
            **{**_sampling_kwargs(args), "samplers": 3 if args.samplers is None else args.samplers},
            **_text_kwargs(args),
        )
    v = model.vocab_size
    logger.info(
        "read %d lines (%d skipped for length), vocabularies %d / %d",
        model.n_lines,
        len(model.skipped),
        v[0],
        v[1],
    )
    logger.info("%s", model._inner.segmentation_line)
    it = model.iterations
    logger.info(
        "trained (iterations %d/%d/%d), %.1f s", it[0], it[1], it[2], time.perf_counter() - t0
    )
    prefix = args.output
    for d in model.directions:
        if prefix is None:
            _write_bytes(None, model.alignments(direction=d).to_bytes())
            continue
        model.write_alignments(f"{prefix}.{d}.pharaoh", direction=d)
        if args.log:
            _write_bytes(f"{prefix}.{d}.log", model._inner.training_log(d))
    for m in args.symmetrize:
        model.write_alignments(f"{prefix}.{_method(m)}.pharaoh", symmetrize=m)
    if args.save:
        model.save(args.save)
        logger.info("saved the model to %s", args.save)


def _cmd_align(args: argparse.Namespace) -> None:
    files = args.files
    if len(files) not in (2, 3):
        raise ValueError("usage: eflomalr align MODEL SRC [TGT]")
    if args.fast and (args.scores or args.exact or args.accumulate):
        raise ValueError("--fast cannot be combined with --scores, --exact or --accumulate")
    model = eflomalr.Model.load(files[0])
    if args.symmetrize and len(model.directions) < 2:
        raise ValueError("--symmetrize needs a model of both directions")
    tgt = files[2] if len(files) == 3 else None
    t0 = time.perf_counter()
    with _input_path(files[1], tgt) as src:
        result = model.align_many(
            src,
            tgt,
            symmetrize=args.symmetrize,
            direction=None if args.symmetrize else args.direction,
            threads=args.threads,
            # as the native binary: scores belong to one direction and are dropped by
            # symmetrisation
            scores=args.scores and not args.symmetrize,
            fast=args.fast,
            seed=args.seed,
            **_sampling_kwargs(args),
        )
    logger.info("aligned, %.1f s", time.perf_counter() - t0)
    _write_bytes(args.output, result.to_bytes())


def _cmd_symmetrize(args: argparse.Namespace) -> None:
    method = _method(args.method)
    if method != "invert" and args.inv is None:
        raise ValueError(f"{method} needs two alignment files")
    fwd = _read_bytes(args.fwd)
    inv = _read_bytes(args.inv) if (args.inv is not None and method != "invert") else None
    _write_bytes(args.output, _core.symmetrize_text(fwd, inv, method))


def _cmd_makepriors(args: argparse.Namespace) -> None:
    fwd = _read_bytes(args.forward_links)
    rev = (
        fwd if args.reverse_links == args.forward_links == "-" else _read_bytes(args.reverse_links)
    )
    if not 1 <= args.max_len <= eflomalr.model.MAX_MAX_LEN:
        raise ValueError(f"--max-len must be in 1 ..= {eflomalr.model.MAX_MAX_LEN}")
    with _input_path(args.input, args.tgt) as src:
        data = _core.make_priors(
            src,
            args.tgt,
            fwd,
            rev,
            eflomalr.model._text_options(
                not args.no_lowercase,
                args.source_prefix,
                args.source_suffix,
                args.target_prefix,
                args.target_suffix,
                DEFAULT_MAX_LEN,
            ),
            eflomalr.model._check_threads(args.threads),
        )
    _write_bytes(args.output, data)


def _cmd_estimate(args: argparse.Namespace) -> None:
    if args.model is not None and args.align_src is not None:
        raise ValueError("--model and --align-src exclude each other")
    if args.align_tgt is not None and args.align_src is None:
        raise ValueError("--align-tgt needs --align-src")
    if not 1 <= args.max_len <= eflomalr.model.MAX_MAX_LEN:
        raise ValueError(f"--max-len must be in 1 ..= {eflomalr.model.MAX_MAX_LEN}")
    with _input_path(args.input, args.tgt) as src:
        est = eflomalr.estimate_parameters(
            src,
            args.tgt,
            align_src=args.align_src,
            align_tgt=args.align_tgt,
            model=args.model,
            segment=args.segment,
            threads=args.threads,
            **_text_kwargs(args),
        )
    out = est.to_json() if args.json else str(est)
    _write_bytes(args.output, out.encode("utf-8"))


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    print(f"eflomalr: warning: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        print(f"eflomalr {eflomalr.__version__}")
        return 0
    if args.command is None:
        parser.print_usage(sys.stderr)
        print(
            "eflomalr: error: a command is required (train, align, symmetrize, makepriors, "
            "estimate)",
            file=sys.stderr,
        )
        return 2
    quiet = getattr(args, "quiet", False)
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    handlers = {
        "train": _cmd_train,
        "align": _cmd_align,
        "symmetrize": _cmd_symmetrize,
        "makepriors": _cmd_makepriors,
        "estimate": _cmd_estimate,
    }
    try:
        with warnings.catch_warnings():
            # the library's RuntimeWarnings (nothing to align) look like the native binary's
            warnings.simplefilter("always", RuntimeWarning)
            warnings.showwarning = _show_warning
            return handlers[args.command](args) or 0
    except KeyboardInterrupt:
        print("eflomalr: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Output closed early (`eflomalr align ... | head`): not an error worth a message.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    except (ValueError, TypeError, OSError) as e:
        print(f"eflomalr: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
