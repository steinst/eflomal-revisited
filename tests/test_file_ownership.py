"""Saving a model and writing logs must not remove or truncate other files."""

from __future__ import annotations

import subprocess
import threading

import eflomalr
import pytest
from eflomalr.cli import main

FAST = ["-l", "0.2"]


@pytest.mark.parametrize("direction,other", [("fwd", "inv"), ("inv", "fwd")])
def test_native_one_direction_log_keeps_other_log(
    tmp_path, corpus_files, native_binary, direction, other
):
    a, b = corpus_files
    old = tmp_path / f"run.{other}.log"
    old.write_bytes(b"a log of an earlier run\n")
    subprocess.run(
        [str(native_binary), "train", str(a), str(b), "--directions", direction, "--log"]
        + ["-o", str(tmp_path / "run"), "-q", "-t", "2", *FAST],
        check=True,
    )
    assert old.read_bytes() == b"a log of an earlier run\n"
    log = (tmp_path / f"run.{direction}.log").read_text()
    assert f"[{direction} sampler 1]" in log


@pytest.mark.parametrize("direction,other", [("fwd", "inv"), ("inv", "fwd")])
def test_python_one_direction_log_keeps_other_log(tmp_path, corpus_files, direction, other):
    a, b = corpus_files
    old = tmp_path / f"run.{other}.log"
    old.write_bytes(b"earlier\n")
    rc = main(
        ["train", str(a), str(b), "--directions", direction, "--log"]
        + ["-o", str(tmp_path / "run"), "-q", "-t", "2", *FAST]
    )
    assert rc == 0
    assert old.read_bytes() == b"earlier\n"
    assert (tmp_path / f"run.{direction}.log").exists()


def _listing(d):
    return sorted(p.name for p in d.iterdir())


def test_save_keeps_unrelated_sibling(tmp_path, model):
    sibling = tmp_path / "m.tmp"
    sibling.write_bytes(b"not ours")
    model.save(tmp_path / "m")
    assert sibling.read_bytes() == b"not ours"
    assert _listing(tmp_path) == ["m", "m.tmp"]
    assert eflomalr.Model.load(tmp_path / "m").directions == model.directions


def test_simultaneous_saves_leave_one_complete_model(tmp_path, corpus):
    src, tgt, _ = corpus
    both = eflomalr.train(src, tgt, threads=1, length=0.2)
    fwd = eflomalr.train(src, tgt, threads=1, length=0.2, directions="fwd")
    path = tmp_path / "m"
    for _ in range(10):
        go = threading.Barrier(2)
        errors = []

        def save(m, go, errors):
            try:
                go.wait()
                m.save(path)
            except (OSError, ValueError) as e:  # reported below
                errors.append(e)

        ts = [threading.Thread(target=save, args=(m, go, errors)) for m in (both, fwd)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errors
        assert eflomalr.Model.load(path).directions in (("fwd", "inv"), ("fwd",))
    assert _listing(tmp_path) == ["m"]


def test_failed_save_keeps_destination(tmp_path, model):
    path = tmp_path / "m"
    model.save(path)
    before = path.read_bytes()
    ro = tmp_path / "ro"
    ro.mkdir()
    (ro / "m").write_bytes(before)
    ro.chmod(0o500)  # the temporary file cannot be created: the save fails before any write
    try:
        with pytest.raises(OSError):
            model.save(ro / "m")
    finally:
        ro.chmod(0o700)
    assert (ro / "m").read_bytes() == before
    assert _listing(ro) == ["m"]
