"""The README quick start runs as written (on a synthetic corpus)."""

from __future__ import annotations

import re
from pathlib import Path

import eflomalr
from conftest import ROOT, make_corpus


def test_quick_start_runs_as_written(tmp_path, monkeypatch):
    readme = (ROOT / "README.md").read_text().replace("\r\n", "\n")
    block = re.search(r"```python\n(.*?)```", readme, re.DOTALL).group(1)
    src, tgt, _ = make_corpus(n=300)
    for stem, n in (("corpus", 300), ("new", 10)):
        Path(tmp_path / f"{stem}.en").write_text("".join(" ".join(s) + "\n" for s in src[:n]))
        Path(tmp_path / f"{stem}.fr").write_text("".join(" ".join(t) + "\n" for t in tgt[:n]))
    monkeypatch.chdir(tmp_path)
    # The README's own example sentence pair must be in the vocabulary for its comment to hold.
    (tmp_path / "corpus.en").write_text("the house\n" * 20 + (tmp_path / "corpus.en").read_text())
    (tmp_path / "corpus.fr").write_text("la maison\n" * 20 + (tmp_path / "corpus.fr").read_text())
    namespace: dict = {}
    exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
    assert (tmp_path / "en-fr.eflomalr").exists() and (tmp_path / "out.align").exists()
    assert len((tmp_path / "out.align").read_text().splitlines()) == 320
    assert len((tmp_path / "new.align").read_text().splitlines()) == 10
    assert namespace["model"].align(["the", "house"], ["la", "maison"]) == [(0, 0), (1, 1)]
    assert eflomalr.Model.load(tmp_path / "en-fr.eflomalr").directions == ("fwd", "inv")
