from __future__ import annotations

import subprocess
import sys

import eflomalr
import pytest
from eflomalr.cli import main

FAST = ["-l", "0.2"]


def test_no_command(capsys):
    assert main([]) == 2
    assert "command is required" in capsys.readouterr().err


def test_version(capsys):
    assert main(["-V"]) == 0
    assert eflomalr.__version__ in capsys.readouterr().out


def test_train_align_symmetrize_makepriors(tmp_path, corpus_files, corpus, capsys):
    a, b = corpus_files
    src = corpus[0]
    prefix = tmp_path / "out"
    model_path = tmp_path / "m.eflomalr"
    rc = main(
        ["train", str(a), str(b), "-o", str(prefix), "-t", "2", "-s", "grow-diag-final-and"]
        + ["-s", "intersection", "--save", str(model_path), "--log", "-q", *FAST]
    )
    assert rc == 0
    for name in ("fwd.pharaoh", "inv.pharaoh", "grow-diag-final-and.pharaoh", "intersect.pharaoh"):
        assert (tmp_path / f"out.{name}").exists(), name
    assert len((tmp_path / "out.fwd.pharaoh").read_text().split("\n")) == len(src) + 1
    log = (tmp_path / "out.fwd.log").read_text()
    assert "[fwd sampler 1] model 1 iteration 1/" in log and "[inv" not in log

    m = eflomalr.Model.load(model_path)
    assert m.directions == ("fwd", "inv")

    # align: stdout, file, direction, scores, fast
    capsys.readouterr()
    assert main(["align", str(model_path), str(a), str(b), "-s", "grow-diag-final-and"]) == 0
    sym = capsys.readouterr().out
    assert sym.encode() == m.align_many(a, b, symmetrize="grow-diag-final-and").to_bytes()
    dest = tmp_path / "al.out"
    assert (
        main(
            ["align", str(model_path), str(a), str(b), "--direction", "inv", "-o", str(dest), "-q"]
        )
        == 0
    )
    assert dest.read_bytes() == m.align_many(a, b, direction="inv").to_bytes()
    assert main(["align", str(model_path), str(a), str(b), "--scores", "-o", str(dest)]) == 0
    assert all(" ||| " in ln for ln in dest.read_text().split("\n")[:-1])
    assert main(["align", str(model_path), str(a), str(b), "--fast", "-o", str(dest)]) == 0
    assert dest.read_bytes() == m.align_many(a, b, direction="fwd", fast=True).to_bytes()

    # symmetrize reproduces the training-time symmetrisation; invert needs one file
    capsys.readouterr()
    fwd, inv = tmp_path / "out.fwd.pharaoh", tmp_path / "out.inv.pharaoh"
    assert main(["symmetrize", str(fwd), str(inv), "-m", "grow-diag-final-and"]) == 0
    assert capsys.readouterr().out == (tmp_path / "out.grow-diag-final-and.pharaoh").read_text()
    assert main(["symmetrize", str(fwd), "-m", "invert"]) == 0
    inverted = capsys.readouterr().out.split("\n")[:-1]
    first = eflomalr.parse_pharaoh(fwd.read_text().split("\n")[0])
    assert inverted[0] == " ".join(f"{i}-{j}" for i, j in eflomalr.invert(first))

    # makepriors, then train with the priors
    pri = tmp_path / "p.priors"
    assert main(["makepriors", str(a), str(b), "-f", str(fwd), "-r", str(inv), "-o", str(pri)]) == 0
    assert pri.read_bytes() == eflomalr.make_priors(a, b, forward=fwd, reverse=inv)
    assert (
        main(["train", str(a), str(b), "-o", str(tmp_path / "pp"), "-p", str(pri), "-q", *FAST])
        == 0
    )
    assert (tmp_path / "pp.fwd.pharaoh").exists()


def test_train_single_direction_to_stdout(corpus_files, joined_file, capfdbinary):
    a, b = corpus_files
    assert main(["train", str(a), str(b), "--directions", "inv", "-q", *FAST]) == 0
    out = capfdbinary.readouterr().out
    assert out == eflomalr.train(joined_file, directions="inv", length=0.2).alignments().to_bytes()


def test_train_reads_joined_file(tmp_path, joined_file):
    assert main(["train", str(joined_file), "-o", str(tmp_path / "j"), "-q", *FAST]) == 0
    assert (tmp_path / "j.fwd.pharaoh").exists()


def test_progress_is_logged(tmp_path, joined_file, caplog):
    with caplog.at_level("INFO", logger="eflomalr"):
        assert (
            main(
                [
                    "train",
                    str(joined_file),
                    "-o",
                    str(tmp_path / "j"),
                    "-1",
                    "1",
                    "-2",
                    "1",
                    "-3",
                    "2",
                ]
            )
            == 0
        )
    text = caplog.text
    assert "[fwd sampler 1] model 3 iteration 2/2" in text and "trained (iterations 1/1/2)" in text
    assert "read 400 lines (0 skipped for length), vocabularies 40 / 40" in text


@pytest.mark.parametrize(
    "argv, message",
    [
        (["train", "{j}"], "without --output"),
        (["train", "{j}", "-o", "{p}", "--directions", "fwd", "-s", "union"], "both directions"),
        (["train", "{j}", "-s", "union"], "needs --output"),
        (["train", "{j}", "--directions", "fwd", "--log"], "needs --output"),
        (["train", "{j}", "-o", "{p}", "-m", "4"], "model"),
        (["train", "{j}", "-o", "{p}", "-n", "0"], "samplers"),
        (["train", "{j}", "-o", "{p}", "--accumulate-limit", "5"], "--accumulate"),
        (["train", "/nonexistent/file", "-o", "{p}"], "/nonexistent/file"),
        (["train", "{j}", "-o", "{p}", "-p", "/nonexistent/priors"], "/nonexistent/priors"),
        (["align", "/nonexistent/model", "{j}"], "/nonexistent/model"),
        (["align", "{j}"], "usage"),
        (["symmetrize", "{j}"], "needs two"),
    ],
)
def test_errors_exit_1(tmp_path, joined_file, capsys, argv, message):
    argv = [x.format(j=joined_file, p=tmp_path / "o") for x in argv]
    assert main(argv) == 1
    assert message in capsys.readouterr().err


def test_bad_option_exits_2():
    with pytest.raises(SystemExit) as e:
        main(["train", "x", "-s", "bogus"])
    assert e.value.code == 2


def test_module_entry_point():
    r = subprocess.run(
        [sys.executable, "-m", "eflomalr.cli", "-V"], check=False, capture_output=True, text=True
    )
    assert r.returncode == 0 and eflomalr.__version__ in r.stdout


def test_cli_equals_native_binary(tmp_path, joined_file, native_binary):
    args = [
        "train",
        str(joined_file),
        "--scores",
        "-s",
        "union",
        "-s",
        "grow-diag",
        "-q",
        "--seed",
        "3",
        "-n",
        "2",
        *FAST,
    ]
    subprocess.run(
        [str(native_binary), *args, "-o", str(tmp_path / "n"), "--save", str(tmp_path / "n.m")],
        check=True,
    )
    assert main([*args, "-o", str(tmp_path / "p"), "--save", str(tmp_path / "p.m")]) == 0
    for ext in ("fwd.pharaoh", "inv.pharaoh", "union.pharaoh", "grow-diag.pharaoh", "m"):
        assert (tmp_path / f"n.{ext}").read_bytes() == (tmp_path / f"p.{ext}").read_bytes(), ext
    for extra in (
        ["-s", "union"],
        ["--direction", "inv", "--scores"],
        ["--fast"],
        ["--seed", "9", "-n", "1"],
        ["--exact", "-1", "2", "-2", "2", "-3", "2"],
    ):
        al = ["align", str(tmp_path / "n.m"), str(joined_file), "-q", *extra]
        subprocess.run([str(native_binary), *al, "-o", str(tmp_path / "na")], check=True)
        assert main([*al, "-o", str(tmp_path / "pa")]) == 0
        assert (tmp_path / "na").read_bytes() == (tmp_path / "pa").read_bytes(), extra


def test_stdin_input(tmp_path, joined_file, native_binary):
    """A joined corpus on standard input (`-`) gives the same output as the file."""
    data = joined_file.read_bytes()
    r = subprocess.run(
        [sys.executable, "-m", "eflomalr.cli", "train", "-", "--directions", "fwd", "-q", *FAST],
        input=data,
        capture_output=True,
        check=True,
    )
    n = subprocess.run(
        [str(native_binary), "train", "-", "--directions", "fwd", "-q", *FAST],
        input=data,
        capture_output=True,
        check=True,
    )
    assert r.stdout == n.stdout
    assert (
        r.stdout
        == eflomalr.train(joined_file, directions="fwd", length=0.2).alignments().to_bytes()
    )


def test_cli_contracts_match_native(tmp_path, corpus_files, native_binary):
    """`align --direction both` (native: silently fwd) and `makepriors -f -
    -r -` (native: stdin read twice) behave the same in both command lines."""
    a, b = corpus_files
    model = tmp_path / "m"
    assert (
        main(
            ["train", str(a), str(b), "--save", str(model), "-o", str(tmp_path / "o")]
            + ["-q", "-t", "2", *FAST]
        )
        == 0
    )
    py = [sys.executable, "-m", "eflomalr.cli"]
    for cli in ([str(native_binary)], py):
        r = subprocess.run(
            cli + ["align", str(model), str(a), str(b), "--direction", "both"],
            capture_output=True,
            check=False,
        )
        assert r.returncode == 2 and b"both" in r.stderr, (cli, r.stderr)
    links = (tmp_path / "o.fwd.pharaoh").read_bytes()
    outs = []
    for cli in ([str(native_binary)], py):
        out = tmp_path / f"p{len(outs)}"
        subprocess.run(
            cli + ["makepriors", str(a), str(b), "-f", "-", "-r", "-", "-o", str(out)],
            input=links,
            check=True,
        )
        outs.append(out.read_bytes())
    assert outs[0] == outs[1] and outs[0].count(b"\n") > 10
