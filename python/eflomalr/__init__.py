"""eflomal revisited: fast, low-memory Bayesian word alignment by collapsed Gibbs sampling.

A clean-room reimplementation of eflomal's models (IBM Model 1 with Dirichlet priors, plus an
HMM-style jump distribution and fertility), with a deterministic seed.

Quick start::

    import eflomalr

    model = eflomalr.train("corpus.en", "corpus.fr", seed=1)  # model 3, both directions
    model.save("en-fr.eflomalr")

    for links in model.alignments(symmetrize="grow-diag-final-and"):
        ...                                                    # list of (source, target) links

    model = eflomalr.Model.load("en-fr.eflomalr")
    model.align(["the", "house"], ["la", "maison"])            # [(0, 0), (1, 1)]

See :func:`train`, :class:`Model`, :func:`estimate_parameters`, :func:`make_priors` and
:func:`symmetrize`. Progress is
reported through the ``eflomalr`` logger and an optional callback.
"""

from __future__ import annotations

import logging as _logging

from eflomalr._core import version as _version
from eflomalr.model import (
    METHODS,
    SEGMENTS,
    Alignments,
    Estimate,
    Link,
    Model,
    Progress,
    estimate_parameters,
    invert,
    make_priors,
    parse_pharaoh,
    symmetrize,
    symmetrize_many,
    train,
)

__version__: str = _version()

_logging.getLogger("eflomalr").addHandler(_logging.NullHandler())

__all__ = [
    "METHODS",
    "SEGMENTS",
    "Alignments",
    "Estimate",
    "Link",
    "Model",
    "Progress",
    "__version__",
    "estimate_parameters",
    "invert",
    "make_priors",
    "parse_pharaoh",
    "symmetrize",
    "symmetrize_many",
    "train",
]
