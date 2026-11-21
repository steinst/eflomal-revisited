# Changelog

## 0.1.0 (2026-11-21)

First public release.

- eflomal's models (IBM Model 1, HMM-like jumps, fertility) trained by collapsed Gibbs sampling,
  with eflomal's defaults, iteration schedule, NULL prior, priors files and makepriors.
- Deterministic for a seed at any thread count; parallel within each sampler as well as across
  samplers and directions; `--exact` for the sequential sampler.
- Symmetrisation byte-identical to fast_align's `atools`.
- Saved models: `align` (sampling with the model as priors) and `align --fast` (exact HMM decode,
  independent of the batch).
- `--segment auto` prefix stemming chosen from the corpus size (default); `estimate` reports the
  choice and the measurements behind it.
- Python package `eflomalr` (abi3, Python ≥ 3.9) with the same command line as the native binary.
