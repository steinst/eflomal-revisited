"""Shared fixtures: a small synthetic parallel corpus (no real text is committed)."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "target" / "release" / "eflomalr"


# Tests write and read text files; make that independent of the platform: UTF-8 and "\n" line
# ends.
def _write_text(self, data, encoding=None, errors=None, newline=None):
    return self.write_bytes(data.encode(encoding or "utf-8", errors or "strict"))


def _read_text(self, encoding=None, errors=None):
    return self.read_bytes().decode(encoding or "utf-8", errors or "strict")


Path.write_text = _write_text  # type: ignore[method-assign]
Path.read_text = _read_text  # type: ignore[method-assign]


def make_corpus(
    n: int = 400, vocab: int = 40, seed: int = 7, min_len: int = 3, max_len: int = 9
) -> tuple[list[list[str]], list[list[str]], list[list[int]]]:
    """Source sentences, target sentences and the gold permutation of each pair.

    Target word ``t{k}`` translates source word ``s{k}``; each target sentence is the source
    sentence with a few adjacent words swapped.
    """
    rng = random.Random(seed)
    src, tgt, perms = [], [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        words = [rng.randrange(vocab) for _ in range(length)]
        order = list(range(length))
        for _ in range(rng.randint(0, 2)):
            i = rng.randrange(length - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
        src.append([f"s{w}" for w in words])
        tgt.append([f"t{words[o]}" for o in order])
        perms.append(order)
    return src, tgt, perms


def make_inflected_corpus(
    n: int = 300, lemmas: int = 30, seed: int = 11
) -> tuple[list[list[str]], list[list[str]]]:
    """A corpus whose words are a stem plus an ending, so that prefix stemming changes the
    vocabulary (and the alignment): source ``<stem><ending>`` with stems of 5-7 letters,
    target ``<stem'><ending'>`` likewise; endings are chosen independently per side."""
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"

    def stem() -> str:
        return "".join(rng.choice(letters) for _ in range(rng.randint(5, 7)))

    s_stems = [stem() for _ in range(lemmas)]
    t_stems = [stem() for _ in range(lemmas)]
    s_end, t_end = ["", "s", "ing", "ed", "er"], ["", "e", "es", "ent", "ons"]
    src, tgt = [], []
    for _ in range(n):
        words = [rng.randrange(lemmas) for _ in range(rng.randint(3, 8))]
        src.append([s_stems[w] + rng.choice(s_end) for w in words])
        tgt.append([t_stems[w] + rng.choice(t_end) for w in words])
    return src, tgt


#: Few iterations keep the tests fast; the corpus is easy.
FAST = {"length": 0.2}


@pytest.fixture(scope="session")
def corpus():
    return make_corpus()


@pytest.fixture()
def corpus_files(tmp_path: Path, corpus):
    src, tgt, _ = corpus
    a, b = tmp_path / "c.src", tmp_path / "c.tgt"
    a.write_text("".join(" ".join(s) + "\n" for s in src), encoding="utf-8")
    b.write_text("".join(" ".join(t) + "\n" for t in tgt), encoding="utf-8")
    return a, b


@pytest.fixture()
def joined_file(tmp_path: Path, corpus):
    src, tgt, _ = corpus
    p = tmp_path / "c.joined"
    p.write_text(
        "".join(f"{' '.join(s)} ||| {' '.join(t)}\n" for s, t in zip(src, tgt)), encoding="utf-8"
    )
    return p


@pytest.fixture(scope="session")
def model(corpus):
    import eflomalr

    src, tgt, _ = corpus
    return eflomalr.train(src, tgt, threads=2, **FAST)


@pytest.fixture(scope="session")
def native_binary():
    """The native `eflomalr` binary, if it has been built (`cargo build --release`)."""
    if not NATIVE.exists():
        if os.environ.get("EFLOMALR_REQUIRE_NATIVE"):  # CI: a missing binary is a failure
            pytest.fail(f"native binary missing: {NATIVE}")
        pytest.skip("native binary not built (cargo build --release)")
    return NATIVE


@pytest.fixture(scope="session")
def atools_binary():
    """fast_align's `atools` on PATH, for the symmetrisation parity test."""
    found = shutil.which("atools")
    if not found:
        pytest.skip("atools (fast_align) is not on PATH")
    return Path(found)


@pytest.fixture()
def inflected_files(tmp_path: Path):
    src, tgt = make_inflected_corpus()
    a, b = tmp_path / "i.src", tmp_path / "i.tgt"
    a.write_text("".join(" ".join(x) + "\n" for x in src), encoding="utf-8")
    b.write_text("".join(" ".join(x) + "\n" for x in tgt), encoding="utf-8")
    return a, b
