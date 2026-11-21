"""Type stubs for the native module (internal)."""

import os
from collections.abc import Iterator
from typing import Callable, overload

from typing_extensions import TypeAlias

_Link: TypeAlias = tuple[int, int]
_Path: TypeAlias = str | os.PathLike[str]
_Text: TypeAlias = tuple[bool, int | None, int | None, int | None, int | None, int]
_Iters: TypeAlias = tuple[int | None, int | None, int | None]
_Progress: TypeAlias = Callable[[str, int, int, int, int, float, float], None]

def version() -> str: ...

class Alignments:
    """A read-only sequence of alignments: one list of ``(source, target)`` links per line."""

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> list[_Link]: ...
    @overload
    def __getitem__(self, index: slice) -> list[list[_Link]]: ...
    def __iter__(self) -> Iterator[list[_Link]]: ...
    def pharaoh(self, i: int) -> str:
        """Pharaoh-format line of alignment ``i`` (no score)."""
    def score(self, i: int) -> float | None:
        """The ``||| score`` of line ``i``, or ``None`` without scores."""
    def to_list(self) -> list[list[_Link]]:
        """All alignments as a list of lists."""
    def to_bytes(self) -> bytes:
        """The Pharaoh text, as :meth:`write` writes it."""
    def write(self, path: _Path) -> None:
        """Write the alignments in Pharaoh format."""

class Model:
    @property
    def directions(self) -> list[str]: ...
    @property
    def vocab_size(self) -> tuple[int, int]: ...
    @property
    def has_training_alignments(self) -> bool: ...
    @property
    def n_lines(self) -> int | None: ...
    @property
    def skipped(self) -> list[int] | None: ...
    @property
    def iterations(self) -> tuple[int, int, int] | None: ...
    @property
    def training_warnings(self) -> list[str]: ...
    @property
    def options(self) -> tuple[int, int, int, float, float, float, float, _Text]: ...
    @property
    def segmentation(
        self,
    ) -> tuple[str, bool, int, tuple[int, int, int], tuple[int, int, int]] | None: ...
    @property
    def segmentation_line(self) -> str | None: ...
    def training_log(self, direction: str) -> bytes: ...
    def save(self, path: _Path) -> None: ...
    @staticmethod
    def load(path: _Path) -> Model: ...
    def training_alignments(
        self, symmetrize: str | None = None, direction: str | None = None
    ) -> Alignments: ...
    def write_training_alignments(
        self, path: _Path, symmetrize: str | None = None, direction: str | None = None
    ) -> None: ...
    def align_many(
        self,
        src: str | list[str],
        tgt: str | list[str] | None,
        symmetrize: str | None,
        direction: str | None,
        threads: int,
        scores: bool,
        fast: bool,
        sampling: tuple[int | None, int | None, float, _Iters, bool, bool, int],
        progress: _Progress | None = None,
    ) -> Alignments: ...

def train(
    src: str | list[str],
    tgt: str | list[str] | None,
    threads: int,
    directions: str,
    seed: int,
    model: int,
    samplers: int,
    length: float,
    iterations: _Iters,
    hyper: tuple[float, float, float, float],
    priors: str | None,
    text: _Text,
    scores: bool,
    score_model: int | None,
    exact: bool,
    accumulate: bool,
    accumulate_limit: int,
    segment: str = "auto",
    progress: _Progress | None = None,
) -> Model: ...
def estimate(
    src: str | list[str],
    tgt: str | list[str] | None,
    align_src: str | list[str] | None,
    align_tgt: str | list[str] | None,
    model: Model | None,
    segment: str,
    text: _Text,
    threads: int,
) -> tuple[str, str]: ...
def make_priors(
    src: str | list[str],
    tgt: str | list[str] | None,
    fwd: bytes,
    rev: bytes,
    text: _Text,
    threads: int,
) -> bytes: ...
def symmetrize_many(
    fwd: list[list[_Link]], inv: list[list[_Link]], method: str
) -> list[list[_Link]]: ...
def symmetrize_text(fwd: bytes, inv: bytes | None, method: str) -> bytes: ...
def invert_many(fwd: list[list[_Link]]) -> list[list[_Link]]: ...
def parse_pharaoh(line: str) -> list[_Link]: ...
