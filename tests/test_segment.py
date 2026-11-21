"""`segment` (the `auto` stemming rule) and `estimate`: the Python API, the CLI, the native
binary, and the model file."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter

import eflomalr
import pytest
from conftest import FAST, make_inflected_corpus
from eflomalr.cli import main

#: sha256 of the output of the binary *before* `segment` existed (commit 04a726c, whose default
#: was plain words), on `make_inflected_corpus()` with `-l 0.2 -s grow-diag-final-and` and
#: seed 1: once with its defaults (words) and once with `--source-prefix 4 --target-prefix 4`.
PRE_8B = {
    "words": {
        "fwd": "2084c99d840afef3ffd8b6f93d435da62466d2069d93c91ff9e256da101c5c9f",
        "inv": "ba2e26fd49377a7f9e5c508a04ea9ad0e1537a457fa5466380ce7f3fe3b8599a",
        "grow-diag-final-and": "4eac601b3084b387739e107157340a9aedfdf9c91d8d3137c44cc3bce7732ca1",
    },
    "prefix4": {
        "fwd": "11a5a87ab02ae71b9770cb897bbc48b10d4774b733d531578cdf621079e9744d",
        "inv": "11a5a87ab02ae71b9770cb897bbc48b10d4774b733d531578cdf621079e9744d",
        "grow-diag-final-and": "11a5a87ab02ae71b9770cb897bbc48b10d4774b733d531578cdf621079e9744d",
    },
}
OUTPUTS = ("fwd", "inv", "grow-diag-final-and")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def py_outputs(model: eflomalr.Model) -> dict[str, str]:
    return {
        "fwd": sha(model.alignments(direction="fwd").to_bytes()),
        "inv": sha(model.alignments(direction="inv").to_bytes()),
        "grow-diag-final-and": sha(model.alignments(symmetrize="grow-diag-final-and").to_bytes()),
    }


def native_outputs(binary, a, b, prefix, *flags) -> dict[str, str]:
    subprocess.run(
        [str(binary), "train", str(a), str(b), "-o", str(prefix), "-s", "grow-diag-final-and"]
        + ["-l", "0.2", "-q", *flags],
        check=True,
    )
    return {
        d: sha(prefix.parent.joinpath(f"{prefix.name}.{d}.pharaoh").read_bytes()) for d in OUTPUTS
    }


# -- the rule and its choice --------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines, prefix",
    [(1, 4), (29_999, 4), (30_000, 5), (449_999, 5), (450_000, 6), (2_000_000, 6)],
)
def test_rule_through_estimate(lines, prefix):
    """The rule at and around its thresholds, seen through the public API (only the line count
    matters, so a corpus of `lines` one-word pairs is enough)."""
    est = eflomalr.estimate_parameters(["a"] * lines, ["b"] * lines)
    seg = est["segmentation"]
    assert est["statistics"]["lines"] == lines
    assert (seg["source_prefix"], seg["target_prefix"]) == (prefix, prefix)
    assert seg["auto_prefix"] == prefix and seg["origin"] == "auto"
    assert seg["rule_fired"].endswith(f"→ prefix {prefix}")
    assert seg["thresholds"] == [30000, 450000] and seg["prefixes"] == [4, 5, 6]


def test_statistics_against_naive_count(inflected_files):
    a, b = inflected_files
    est = eflomalr.estimate_parameters(a, b)
    for path, side in ((a, "source"), (b, "target")):
        lines = path.read_bytes().split(b"\n")[:-1]
        c = Counter(w.decode().lower() for line in lines for w in line.split())
        s = est["statistics"][side]
        tokens = sum(c.values())
        assert (s["tokens"], s["types"]) == (tokens, len(c))
        assert s["singletons"] == sum(1 for n in c.values() if n == 1)
        assert s["tokens_per_type"] == pytest.approx(tokens / len(c), abs=1e-4)
        assert s["mean_length"] == pytest.approx(tokens / len(lines), abs=1e-4)


@pytest.mark.parametrize(
    "kwargs, want",
    [
        ({}, ("auto", 4, None, 4, None)),
        ({"segment": "words"}, ("words", None, None, None, None)),
        ({"source_prefix": 3}, ("explicit", 3, None, None, None)),
        ({"segment": "words", "target_suffix": 2}, ("explicit", None, None, None, 2)),
        ({"source_prefix": 6, "target_prefix": 5}, ("explicit", 6, None, 5, None)),
        ({"lowercase": False}, ("auto", 4, None, 4, None)),
    ],
)
def test_estimate_and_train_agree(inflected_files, kwargs, want):
    a, b = inflected_files
    est = eflomalr.estimate_parameters(a, b, **kwargs)
    m = eflomalr.train(a, b, **FAST, **kwargs)
    seg = est["segmentation"]
    got = (seg["origin"], seg["source_prefix"], seg["source_suffix"], seg["target_prefix"])
    got = (*got, seg["target_suffix"])
    assert got == want
    o = m.options
    keys = ("source_prefix", "source_suffix", "target_prefix", "target_suffix")
    assert tuple(o[k] for k in keys) == want[1:]
    assert m.segmentation["origin"] == want[0]
    assert m.segmentation["lines"] == est["statistics"]["lines"] == 300
    for side in ("source", "target"):
        for k in ("tokens", "types", "singletons"):
            assert m.segmentation[side][k] == est["statistics"][side][k]
    assert m.segmentation["lowercase"] == kwargs.get("lowercase", True)


# -- outputs ----------------------------------------------------------------------------------


def test_words_reproduces_pre_8b_output(inflected_files, native_binary, tmp_path):
    a, b = inflected_files
    assert (
        native_outputs(native_binary, a, b, tmp_path / "w", "--segment", "words")
        == (PRE_8B["words"])
    )
    assert py_outputs(eflomalr.train(a, b, segment="words", **FAST)) == PRE_8B["words"]
    # the default (auto, prefix 4 here) is the old explicit prefix 4, and differs from words
    assert native_outputs(native_binary, a, b, tmp_path / "d") == PRE_8B["prefix4"]
    assert py_outputs(eflomalr.train(a, b, **FAST)) == PRE_8B["prefix4"]
    assert PRE_8B["words"] != PRE_8B["prefix4"]


def test_auto_is_thread_independent(inflected_files):
    a, b = inflected_files
    ref = py_outputs(eflomalr.train(a, b, threads=1, **FAST))
    for t in (2, 4, 8):
        assert py_outputs(eflomalr.train(a, b, threads=t, **FAST)) == ref


def test_save_load_align_uses_the_stored_prefixes(inflected_files, tmp_path):
    a, b = inflected_files
    m = eflomalr.train(a, b, **FAST)
    m.save(tmp_path / "m")
    loaded = eflomalr.Model.load(tmp_path / "m")
    assert loaded.options == m.options and loaded.options["source_prefix"] == 4
    assert loaded.segmentation == m.segmentation
    # a word form never seen in training is read as its stem
    src, tgt = make_inflected_corpus()
    w = src[0][0][:5] + "xyz"
    assert w not in {x for s in src for x in s}
    assert loaded.align([w], [tgt[0][0]], direction="fwd", fast=True) == [(0, 0)]
    # the loaded model aligns new text exactly as the trained one
    new_src, new_tgt = src[:40], tgt[:40]
    al = loaded.align_many(new_src, new_tgt, direction="fwd").to_bytes()
    assert al == m.align_many(new_src, new_tgt, direction="fwd").to_bytes()
    # the stems are what the model holds: 30 lemmas, against 150 word forms without stemming
    words = eflomalr.train(a, b, segment="words", **FAST)
    assert words.options["source_prefix"] is None
    assert (loaded.vocab_size, words.vocab_size) == ((30, 30), (150, 150))


# -- estimate ---------------------------------------------------------------------------------


def test_estimate_report_contents(inflected_files):
    a, b = inflected_files
    est = eflomalr.estimate_parameters(a, b)
    assert isinstance(est, dict) and isinstance(est, eflomalr.Estimate)
    assert json.loads(est.to_json()) == dict(est)
    ev = est["evidence"]
    assert [e["pair"] for e in ev] == ["enfr", "enis", "roen"]
    assert all(e["setting"] == "prefix:4" and e["gdfa_ci"][1] < 0 for e in ev)
    params = {p["python"]: p for p in est["parameters"]}
    assert set(params) == {"null_prior", "lex_alpha", "length", "samplers"}
    assert params["null_prior"]["default"] == "0.2"
    assert "no reliable gain" in params["lex_alpha"]["decision"]
    trade = {t["value"]: t for t in params["samplers"]["tradeoff"]}
    assert trade["6"]["mean_gdfa_delta"] == pytest.approx(-0.25, abs=0.01)
    assert trade["6"]["time_ratio"] == pytest.approx(1.9, abs=0.05)
    assert trade["8"]["mean_gdfa_delta"] == pytest.approx(-0.31, abs=0.01)
    assert est["align"] is None
    text = str(est)
    assert "prefix 4 on both sides" in text and "--samplers 3" in text
    assert "changes none of them" in text


def test_estimate_text_to_align(inflected_files, tmp_path):
    a, b = inflected_files
    src, tgt = make_inflected_corpus()
    # in-domain text: no warnings
    est = eflomalr.estimate_parameters(a, b, align_src=src[:50], align_tgt=tgt[:50])
    al = est["align"]
    assert al["reference"] == "training corpus" and al["statistics"]["lines"] == 50
    assert al["oov_share"] == {"source": 0, "target": 0} and al["warnings"] == []
    # another "language": unknown words and much longer sentences are flagged
    other = [" ".join(f"zz{i}{k}" for k in range(30)) for i in range(20)]
    est = eflomalr.estimate_parameters(a, b, align_src=other, align_tgt=tgt[:20])
    w = est["align"]["warnings"]
    assert est["align"]["oov_share"]["source"] == 100
    assert any("unknown" in x for x in w) and any("average" in x for x in w)
    # with a model: its segmentation and its training statistics
    m = eflomalr.train(a, b, segment="words", **FAST)
    m.save(tmp_path / "m")
    for model in (m, tmp_path / "m"):
        est = eflomalr.estimate_parameters(src[:50], tgt[:50], model=model)
        seg = est["segmentation"]
        assert seg["from_model"] and seg["origin"] == "words" and seg["segment"] is None
        assert seg["source_prefix"] is None
        assert est["statistics"]["lines"] == 300
        assert est["align"]["reference"] == "model" and est["align"]["warnings"] == []
        assert "fixes the segmentation" in str(est)


def test_estimate_errors(inflected_files, tmp_path):
    a, b = inflected_files
    with pytest.raises(ValueError, match="segment"):
        eflomalr.estimate_parameters(a, b, segment="subword")
    with pytest.raises(ValueError, match="segment"):
        eflomalr.train(a, b, segment="bpe")
    with pytest.raises(ValueError, match="align_tgt needs"):
        eflomalr.estimate_parameters(a, b, align_tgt=b)
    with pytest.raises(ValueError, match="either"):
        eflomalr.estimate_parameters(a, b, align_src=a, model=tmp_path / "x")
    with pytest.raises(ValueError, match="source_prefix"):
        eflomalr.estimate_parameters(a, b, source_prefix=-1)
    with pytest.raises(FileNotFoundError):
        eflomalr.estimate_parameters(tmp_path / "missing", b)


# -- CLI and native binary --------------------------------------------------------------------


def test_cli_estimate_equals_native(inflected_files, native_binary, tmp_path):
    a, b = inflected_files
    joined = tmp_path / "j"
    joined.write_bytes(
        b"".join(
            x + b" ||| " + y + b"\n"
            for x, y in zip(a.read_bytes().splitlines(), b.read_bytes().splitlines())
        )
    )
    model = tmp_path / "m"
    subprocess.run(
        [str(native_binary), "train", str(a), str(b), "--directions", "fwd", "-q", "-l", "0.2"]
        + ["--save", str(model)],
        check=True,
        capture_output=True,
    )
    for args in (
        [str(a), str(b)],
        [str(joined)],
        [str(a), str(b), "--segment", "words", "--no-lowercase"],
        [str(a), str(b), "--target-prefix", "3"],
        [str(a), str(b), "--align-src", str(joined)],
        [str(a), str(b), "--align-src", str(b), "--align-tgt", str(a)],
        [str(joined), "--model", str(model)],
    ):
        for fmt in ([], ["--json"]):
            n = subprocess.run(
                [str(native_binary), "estimate", *args, *fmt], check=True, capture_output=True
            ).stdout
            assert main(["estimate", *args, *fmt, "-o", str(tmp_path / "p")]) == 0
            assert (tmp_path / "p").read_bytes() == n, (args, fmt)
    # the CLI's JSON is the API's
    assert main(["estimate", str(a), str(b), "--json", "-o", str(tmp_path / "p")]) == 0
    assert json.loads((tmp_path / "p").read_text()) == eflomalr.estimate_parameters(a, b)


def test_cli_segment_equals_native(inflected_files, native_binary, tmp_path, caplog):
    a, b = inflected_files
    for flags in ([], ["--segment", "words"], ["--segment", "auto", "--source-suffix", "3"]):
        args = ["train", str(a), str(b), "-l", "0.2", "-s", "union", "-q", *flags]
        subprocess.run(
            [str(native_binary), *args, "-o", str(tmp_path / "n"), "--save", str(tmp_path / "n.m")],
            check=True,
        )
        assert main([*args, "-o", str(tmp_path / "p"), "--save", str(tmp_path / "p.m")]) == 0
        for ext in ("fwd.pharaoh", "inv.pharaoh", "union.pharaoh", "m"):
            assert (tmp_path / f"n.{ext}").read_bytes() == (tmp_path / f"p.{ext}").read_bytes(), (
                flags,
                ext,
            )
        # align with the saved model, native and Python
        al = ["align", str(tmp_path / "n.m"), str(a), str(b), "-q", "-s", "union"]
        subprocess.run([str(native_binary), *al, "-o", str(tmp_path / "na")], check=True)
        assert main([*al, "-o", str(tmp_path / "pa")]) == 0
        assert (tmp_path / "na").read_bytes() == (tmp_path / "pa").read_bytes(), flags
    # the progress output names the choice, as the native binary's
    n = subprocess.run(
        [str(native_binary), "train", str(a), str(b), "-l", "0.2", "--directions", "fwd"],
        check=True,
        capture_output=True,
        text=True,
    )
    line = "segmentation: prefix 4 on both sides (auto: 300 lines < 30,000 → prefix 4)"
    assert line in n.stderr.splitlines()
    caplog.set_level("INFO", logger="eflomalr")
    assert (
        main(
            ["train", str(a), str(b), "-l", "0.2", "--directions", "fwd", "-o", str(tmp_path / "x")]
        )
        == 0
    )
    assert line in caplog.messages


def test_cli_estimate_errors(inflected_files, capsys):
    a, b = inflected_files
    assert main(["estimate", str(a), str(b), "--align-tgt", str(b)]) == 1
    assert "--align-tgt needs --align-src" in capsys.readouterr().err
    assert main(["estimate", str(a), "--model", "m", "--align-src", str(a)]) == 1
    assert "exclude" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["estimate", str(a), str(b), "--segment", "bpe"])
