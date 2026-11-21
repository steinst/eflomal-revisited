"""Edge cases: degenerate input through the Python API, the Python CLI and the native
binary."""

from __future__ import annotations

import os
import subprocess
import warnings

import eflomalr
import pytest
from eflomalr.cli import main

FEW = {"ibm1_iters": 2, "hmm_iters": 2, "fert_iters": 3}
FEW_CLI = ["-1", "2", "-2", "2", "-3", "3"]

TRICKY = (
    b"the house ||| la maison\n"
    b"\n"
    b" ||| x\n"
    b"c d |||\n"
    b"no separator here\n"
    b"a ||| b ||| c\n"
    b"caf\xe9 \xe9t\xe9 ||| \xc3\xa9 z\r\n"
    b"<NULL> a ||| x <NULL>\n"
    b"\xd0\xb4\xd0\xbe\xd0\xbc \xce\xa3\xce\x9f\xce\xa6\xce\x9f\xce\xa3 ||| \xe6\x88\x91 \xd8\xa8\n"
    b"e f\tg ||| x y z"  # a tab separates the sides; no trailing newline
)


@pytest.fixture()
def tricky(tmp_path):
    p = tmp_path / "tricky.txt"
    p.write_bytes(TRICKY)
    return p


def _lines(b: bytes) -> list[bytes]:
    return b.split(b"\n")[:-1]


def test_tricky_input_python_equals_native(tmp_path, tricky, native_binary):
    subprocess.run(
        [
            native_binary,
            "train",
            tricky,
            "-o",
            tmp_path / "n",
            "-s",
            "grow-diag-final-and",
            "-t",
            "2",
            "-q",
            *FEW_CLI,
        ],
        check=True,
    )
    for threads in (1, 3):
        m = eflomalr.train(str(tricky), threads=threads, **FEW)
        assert m.n_lines == 10
        for d in ("fwd", "inv"):
            out = m.alignments(direction=d).to_bytes()
            assert out == (tmp_path / f"n.{d}.pharaoh").read_bytes()
            lines = _lines(out)
            assert len(lines) == 10
            assert lines[1] == lines[2] == lines[3] == b""  # empty sides
        gdfa = m.alignments(symmetrize="grow-diag-final-and").to_bytes()
        assert gdfa == (tmp_path / "n.grow-diag-final-and.pharaoh").read_bytes()


def test_empty_corpus_warns_in_every_front_end(tmp_path, native_binary, capsys):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.warns(RuntimeWarning, match="no line pairs"):
        m = eflomalr.train(str(empty))
    assert m.n_lines == 0 and len(m.alignments()) == 0
    assert m.training_warnings == ["the input has no line pairs: nothing was aligned"]
    # the model still saves, loads and aligns
    m.save(tmp_path / "e.eflomalr")
    m2 = eflomalr.Model.load(tmp_path / "e.eflomalr")
    assert m2.training_warnings == []
    with pytest.warns(RuntimeWarning, match="no line pairs"):
        assert len(m2.align_many(str(empty), symmetrize="union")) == 0
    # Python CLI: exit 0, empty files, the native binary's warning line
    assert main(["train", str(empty), "-o", str(tmp_path / "p"), "-q"]) == 0
    err = capsys.readouterr().err
    assert err == "eflomalr: warning: the input has no line pairs: nothing was aligned\n"
    r = subprocess.run(
        [native_binary, "train", empty, "-o", tmp_path / "n", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and r.stderr == err
    for d in ("fwd", "inv"):
        assert (tmp_path / f"p.{d}.pharaoh").read_bytes() == b""
        assert (tmp_path / f"n.{d}.pharaoh").read_bytes() == b""


def test_nothing_to_align_gives_empty_lines(tmp_path):
    with pytest.warns(RuntimeWarning, match="none of the 3 input line pairs"):
        m = eflomalr.train(["", "a b", ""], ["x", "", ""])
    assert m.alignments().to_bytes() == b"\n\n\n"
    with pytest.warns(RuntimeWarning, match="within --max-len"):
        eflomalr.train(["a b c"], ["x"], max_len=2)
    # an ordinary corpus does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = eflomalr.train(["a b"], ["x y"], **FEW)
    assert m.training_warnings == []


def test_estimate_on_an_empty_corpus_warns(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    est = eflomalr.estimate_parameters(str(empty))
    assert est["warnings"] == ["the corpus has no line pairs: training would align nothing"]
    assert "warning: the corpus has no line pairs" in str(est)
    one_sided = tmp_path / "one.txt"
    one_sided.write_bytes(b"a b |||\nc |||\n")
    assert "target side has no words" in eflomalr.estimate_parameters(str(one_sided))["warnings"][0]


def test_line_ends_do_not_matter(tmp_path):
    pairs = [("the house", "la maison"), ("a blue house", "une maison bleue"), ("a b", "x y")]
    lf = "".join(f"{s} ||| {t}\n" for s, t in pairs).encode()
    variants = [lf, lf.replace(b"\n", b"\r\n"), lf[:-1]]
    outs = []
    for k, text in enumerate(variants):
        p = tmp_path / f"v{k}.txt"
        p.write_bytes(text)
        outs.append(eflomalr.train(str(p), **FEW).alignments(direction="fwd").to_bytes())
    assert outs[0] == outs[1] == outs[2]


def test_zero_stem_length_and_zero_length():
    src, tgt = ["houses housing", "books booking"], ["maisons logement", "livres reserve"]
    words = eflomalr.train(src, tgt, segment="words", **FEW)
    zero = eflomalr.train(src, tgt, source_prefix=0, target_suffix=0, **FEW)
    assert zero.alignments().to_bytes() == words.alignments().to_bytes()
    assert zero.options["source_prefix"] is None and zero.segmentation["origin"] == "explicit"
    assert eflomalr.train(src, tgt, length=0).iterations == (2, 1, 2)  # eflomal's -l 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lex_alpha": 1e-50}, "lex-alpha must be between"),
        ({"jump_alpha": 1e30}, "jump-alpha must be between"),
        ({"fert_alpha": 0.0}, "fert-alpha"),
        ({"null_prior": 1e-9}, "null-prior must be between"),
        ({"null_prior": 1 - 1e-9}, "null-prior"),
        ({"length": -1.0}, "length"),
    ],
)
def test_out_of_range_numbers_are_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        eflomalr.train(["a"], ["x"], **kwargs)


def test_more_samplers_than_threads():
    src = ["a b c"] * 300 + ["a c", "b c"]
    tgt = ["x y z"] * 300 + ["x z", "y z"]
    one = eflomalr.train(src, tgt, threads=1, samplers=8, **FEW).alignments().to_bytes()
    two = eflomalr.train(src, tgt, threads=2, samplers=8, **FEW).alignments().to_bytes()
    assert one == two
    assert len(set(_lines(one)[:300])) == 1  # identical pairs, identical links


def test_corrupt_and_truncated_model_files(tmp_path):
    m = eflomalr.train(["a b", "c"], ["x y", "z"], **FEW)
    p = tmp_path / "m.eflomalr"
    m.save(p)
    good = p.read_bytes()
    for name, data, match in [
        ("trunc", good[: len(good) // 2], "truncated"),
        ("tail", good[:-3], "truncated|corrupt"),
        ("magic", b"XXXXXXXX" + good[8:], "not an eflomalr model"),
        ("text", b"LEX\ta\tx\t1\n", "not an eflomalr model"),
        ("empty", b"", "not an eflomalr model"),
        ("extra", good + b"\0", "corrupt"),
    ]:
        q = tmp_path / name
        q.write_bytes(data)
        with pytest.raises(ValueError, match=match):
            eflomalr.Model.load(q)
    with pytest.raises(FileNotFoundError):
        eflomalr.Model.load(tmp_path / "missing")


def test_corrupt_priors_files(tmp_path, capsys):
    for name, text, match in [
        ("count", b"LEX\ta\tx\tlots\n", "line 1: bad count"),
        ("huge", b"LEX\ta\tx\t1\nHMMF\t1\t1e30\n", "line 2: bad count"),
        ("fields", b"FERF\ta\t2\n", "wrong number of fields"),
        ("type", b"XYZ\t1\t2\n", "unknown record type"),
    ]:
        q = tmp_path / name
        q.write_bytes(text)
        with pytest.raises(ValueError, match=match):
            eflomalr.train(["a"], ["x"], priors=q)
        assert (
            main(["train", "-q", "-o", str(tmp_path / "o"), "-p", str(q), str(_joined(tmp_path))])
            == 1
        )
        err = capsys.readouterr().err
        assert err.startswith("eflomalr: error: ") and match.split(": ")[-1] in err


def _joined(tmp_path):
    p = tmp_path / "j.txt"
    p.write_bytes(b"a ||| x\n")
    return p


def test_long_pairs(tmp_path):
    long = " ".join(f"w{i % 200}" for i in range(1500))
    src, tgt = ["a b", long, "b"], ["x y", long, "y"]
    m = eflomalr.train(src, tgt, samplers=1, **FEW)
    assert m.skipped == [1] and _lines(m.alignments(direction="fwd").to_bytes())[1] == b""
    m = eflomalr.train(src, tgt, samplers=1, max_len=2000, **FEW)
    assert m.skipped == [] and _lines(m.alignments(direction="fwd").to_bytes())[1] != b""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes")
def test_pipes_are_read_to_the_end(tmp_path, native_binary):
    """A pipe reports length 0; it used to be read as an empty corpus."""
    import subprocess
    import threading

    src = "".join(f"a{k % 5} b\n" for k in range(30))
    tgt = "".join(f"x{k % 5} y\n" for k in range(30))
    joined = "".join(f"{s} ||| {t}\n" for s, t in zip(src.splitlines(), tgt.splitlines()))
    r = subprocess.run(
        [str(native_binary), "train", "/dev/stdin", "--directions", "fwd", "-q", "-t", "1"],
        input=joined.encode(),
        capture_output=True,
        check=True,
    )
    assert r.stdout.count(b"\n") == 30

    def feed(path, text):
        with open(path, "w") as f:
            f.write(text)

    for tgt_text, ok in ((tgt, True), ("".join(tgt.splitlines(True)[:29]), False)):
        fs, ft = tmp_path / "fs", tmp_path / "ft"
        for p in (fs, ft):
            if p.exists():
                p.unlink()
            os.mkfifo(p)
        ts = [threading.Thread(target=feed, args=a) for a in ((fs, src), (ft, tgt_text))]
        for t in ts:
            t.start()
        r = subprocess.run(
            [str(native_binary), "train", str(fs), str(ft), "--directions", "fwd", "-q", "-t", "1"],
            capture_output=True,
            check=False,
        )
        for t in ts:
            t.join(10)
        if ok:
            assert r.returncode == 0 and r.stdout.count(b"\n") == 30
        else:
            assert r.returncode != 0 and b"different numbers of lines" in r.stderr
    # the Python API reads a pipe the same way
    fs = tmp_path / "py"
    os.mkfifo(fs)
    t = threading.Thread(target=feed, args=(fs, joined))
    t.start()
    m = eflomalr.train(str(fs), threads=1, length=0.2, directions="fwd")
    t.join(10)
    assert m.n_lines == 30


@pytest.mark.parametrize("n", [65534, 65535, 70000])
def test_make_priors_counts_links_of_any_length(tmp_path, native_binary, n):
    """The sampler's u16 length cap rejected an in-bounds end link."""
    want = [b"HMMF\t1\t1", f"HMMF\t{n}\t1".encode(), b"HMMR\t1\t2"]
    p = eflomalr.make_priors([["a"] * n], [["b"]], forward=[[(n - 1, 0)]], reverse=[[(n - 1, 0)]])
    assert [x for x in p.splitlines() if x.startswith(b"HMM")] == want
    (tmp_path / "s").write_text(" ".join(["a"] * n) + "\n")
    (tmp_path / "t").write_text("b\n")
    (tmp_path / "l").write_text(f"{n - 1}-0\n")
    out = tmp_path / "p"
    import subprocess

    subprocess.run(
        [str(native_binary), "makepriors", str(tmp_path / "s"), str(tmp_path / "t")]
        + ["-f", str(tmp_path / "l"), "-r", str(tmp_path / "l"), "-o", str(out)],
        check=True,
    )
    assert [x for x in out.read_bytes().splitlines() if x.startswith(b"HMM")] == want


@pytest.mark.parametrize("order", [0, 1])
def test_nul_escape_does_not_merge_words(tmp_path, order):
    """`\\x00|||` (the reader's stand-in for a literal `|||`) and `|||` were one
    source word; every distinct byte string must stay a distinct word, on both sides."""
    words = [b"\x00|||", b"|||"] if order == 0 else [b"|||", b"\x00|||"]
    plain = [b"x", b"y"]
    for side in ("src", "tgt"):
        a, b = (words, plain) if side == "src" else (plain, words)
        (tmp_path / "s").write_bytes(b"\n".join(a) + b"\n")
        (tmp_path / "t").write_bytes(b"\n".join(b) + b"\n")
        m = eflomalr.train(
            str(tmp_path / "s"),
            str(tmp_path / "t"),
            threads=1,
            length=0.2,
            segment="words",
            lowercase=False,
        )
        assert m.vocab_size == (2, 2), (side, m.vocab_size)  # two words each side
        st = m.segmentation["source" if side == "src" else "target"]
        assert st["types"] == 2
    # a joined file (`|||` there separates the sides) keeps a NUL-led word too
    (tmp_path / "j").write_bytes(b"\x00a ||| x\na ||| y\n")
    m = eflomalr.train(str(tmp_path / "j"), threads=1, length=0.2, segment="words")
    assert m.vocab_size == (2, 2)


def test_nothing_to_align_runs_no_sweeps(tmp_path):
    """An input with nothing to align ran 1,250/1,250/5,000 sweeps and wrote
    1.4 MB of progress log. Now: no sweeps, no progress events; same output, warning, model."""
    for src, tgt in (([], []), (["a", ""], ["", "x"])):
        events = []
        with pytest.warns(RuntimeWarning, match="nothing was aligned"):
            m = eflomalr.train(src, tgt, threads=1, callback=events.append)
        assert m.iterations == (0, 0, 0) and events == []
        assert m.training_log("fwd") == "" and len(m.alignments()) == len(src)
        m.save(tmp_path / "m")
        assert eflomalr.Model.load(tmp_path / "m").directions == ("fwd", "inv")
