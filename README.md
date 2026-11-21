# eflomal revisited

A fast, low-memory Rust reimplementation of [eflomal](https://github.com/robertostling/eflomal)
(Östling and Tiedemann 2016): Bayesian IBM-style word alignment by collapsed Gibbs sampling, with a
command-line tool and Python bindings. Same models and defaults as eflomal, deterministic for a given
seed at any thread count, about 3× faster than eflomal at 20 threads, and with saved models that align
new sentences.

## Installation

From source; needs a Rust toolchain ≥ 1.85 (<https://rustup.rs>) and Python ≥ 3.9:

```bash
git clone https://github.com/steinst/eflomal-revisited && cd eflomal-revisited
python -m venv .venv && source .venv/bin/activate
pip install .                      # builds the Rust extension; installs `import eflomalr` and the `eflomalr` command
```

The native binary alone: `cargo install --path crates/eflomal-core` (or `cargo build --release`, then
`target/release/eflomalr`).

## Quick start (Python)

```python
import eflomalr

model = eflomalr.train("corpus.en", "corpus.fr", threads=0, seed=1)    # model 3, both directions
model.save("en-fr.eflomalr")
for links in model.alignments(symmetrize="grow-diag-final-and"):        # list of (src, tgt)
    pass
model.write_alignments("out.align")                                     # grow-diag-final-and

model = eflomalr.Model.load("en-fr.eflomalr")
model.align(["the", "house"], ["la", "maison"])                         # [(0, 0), (1, 1)]
model.align_many("new.en", "new.fr", symmetrize="union").write("new.align")
```

Input is tokenised text, one sentence per line; `train` also takes one file of `source ||| target`
lines, or in-memory lists of sentences (strings or token lists). Options follow eflomal's names:
`model`, `samplers`, `length`, `ibm1_iters`/`hmm_iters`/`fert_iters`, `null_prior`, `priors`,
`source_prefix` … `target_suffix`, `scores`, `score_model`; additions are `seed`, `threads` (never
changes the output), `directions`, `exact`, `accumulate`, `lowercase`, `max_len`, `segment` and the
Dirichlet priors `lex_alpha`, `jump_alpha`, `fert_alpha`. Every function has a docstring. Long calls
release the GIL and stop on Ctrl-C.

## Command line

```bash
eflomalr train corpus.en corpus.fr -o out -s grow-diag-final-and --save en-fr.eflomalr
#   writes out.fwd.pharaoh, out.inv.pharaoh, out.grow-diag-final-and.pharaoh
eflomalr align en-fr.eflomalr new.en new.fr -s grow-diag-final-and -o new.align   # --fast: no sampling
eflomalr symmetrize out.fwd.pharaoh out.inv.pharaoh -m union -o out.union
eflomalr makepriors corpus.en corpus.fr -f out.fwd.pharaoh -r out.inv.pharaoh -o en-fr.priors
eflomalr estimate corpus.en corpus.fr            # corpus statistics and the stemming `train` would use
```

Output is Pharaoh format (`i-j`, 0-based, source-target, one line per input line; an unaligned pair
gives an empty line). `eflomalr <command> --help` lists every option. The Python command and the native
binary take the same options and write the same bytes for the same seed.

**Stemming.** By default (`--segment auto`) tokens are cut to their first 4 characters below 30,000
lines, 5 below 450,000 and 6 above (eflomal's own prefix stemming, with the length chosen from the
corpus size). `--segment words` aligns plain words, as eflomal does by default; explicit
`--source-prefix`/`--target-prefix` options override both. For already segmented text such as
Japanese, `--segment words` can be slightly better (see the results).

**New text.** `align` samples new sentences with a saved model's counts as priors (eflomal's method
for priors files, packaged); `align --fast` decodes each sentence exactly with the model's HMM, gives
the same links for a sentence whatever else is in the batch, and takes milliseconds.

### From eflomal

| eflomal | eflomal revisited |
|---|---|
| `eflomal-align -s SRC -t TGT -f FWD -r REV` | `eflomalr train SRC TGT -o PREFIX` (here `-s` is `--symmetrize`, `-t` is `--threads`) |
| `atools -c METHOD` | `train … -s METHOD` or `eflomalr symmetrize FWD INV -m METHOD` (byte-identical to `atools`) |
| `-m`, `--null-prior`, `-l`, `-1 -2 -3`, `--n-samplers`, `-p`, `-M`, `--source-prefix` … | the same |
| `-F FILE -R FILE` | `--scores` (` ||| score` appended to each line) |
| `eflomal-makepriors` | `eflomalr makepriors` (the same records; priors files work in both tools) |
| no seed, no saved model | `--seed N` (default 1); `--save MODEL`, then `eflomalr align MODEL …` |

## Results

Grow-diag-final-and on gold-annotated sentences that are appended, unlabelled, to the training text.
eflomal revisited: mean of seeds 1–3; eflomal 2.0.0 (+ `atools`): mean of 3–5 runs (it has no seed).
Brackets: 95 % bootstrap intervals of the difference to eflomal's defaults.

**Development sets** (English–French Hansards, 447 sentences; Romanian–English, 248; the default
stemming rule was fitted on these and on English–Icelandic data), AER (lower is better):

| corpus | eflomal revisited (default) | eflomal revisited `--segment words` | eflomal 2.0.0 |
|---|---:|---:|---:|
| en-fr, 1.13 M lines | **7.04** | 7.57 | 8.02 ± 0.15 |
| en-fr, 100 k lines | **7.11** | 7.59 | 8.18 ± 0.19 |
| ro-en, 49 k lines | **24.84** | 25.84 | 26.17 ± 0.31 |

**Independent test sets** (not used for any choice in this tool), F1 (higher is better):

| corpus | eflomal revisited (default) | `--segment words` | eflomal 2.0.0 | eflomal with the same stemming |
|---|---:|---:|---:|---:|
| English–Hindi, WPT 2005 (3.6 k lines, 90 gold) | **54.16** (+3.06 [+0.84, +5.13]) | 52.28 | 51.11 | 54.32 |
| English–Inuktitut, WPT 2005 (341 k lines, 75 gold) | **91.44** (+9.05 [+6.45, +11.88]) | 83.79 | 82.38 | 90.09 |
| Japanese–English, KFTT test (330 k lines, 582 gold) | 52.95 (+1.30 [+0.78, +1.84]) | **53.37** | 51.65 | 51.77 |

With the same preprocessing, eflomal revisited is level with eflomal on English–Hindi (−0.16 F1
[−1.78, +1.25]) and better on English–Inuktitut (+1.34 [+0.28, +2.40]) and Japanese–English
(+1.18 [+0.69, +1.64]). Grow-diag-final-and is a good default output; on some pairs the intersection
or one direction is better, so choose on held-out data of your own pair if it matters.

## Speed and memory

Defaults (Model 3, 3 samplers, both directions + grow-diag-final-and), wall time and peak memory, on a
20-core desktop (8 performance + 12 efficiency cores), against eflomal 2.0.0 + `atools`:

| corpus | threads | eflomal revisited | eflomal 2.0.0 |
|---|---:|---:|---:|
| en-fr, 1.13 M lines | 20 | **28.2 s**, 871 MB | 93.6 s, 977 MB |
| en-fr, 1.13 M lines | 1 | **300 s**, 644 MB | 332 s, 812 MB |
| Japanese–English, 330 k lines | 20 | **11.1 s**, 629 MB | 26.7 s, 432 MB |
| English–Inuktitut, 341 k lines | 20 | **3.6 s**, 252 MB | 11.5 s, 364 MB |

The output never depends on the thread count. Memory at 20 threads can exceed eflomal's on corpora
with a large vocabulary; use fewer threads if memory is tight.

## Citing

If you use eflomal revisited, please cite the model it implements, and this software:

```bibtex
@article{ostling-tiedemann-2016,
  author  = {Östling, Robert and Tiedemann, Jörg},
  title   = {Efficient Word Alignment with {M}arkov {C}hain {M}onte {C}arlo},
  journal = {The Prague Bulletin of Mathematical Linguistics},
  volume  = {106},
  pages   = {125--146},
  year    = {2016}
}

@software{steingrimsson-eflomal-revisited,
  author  = {Steingrímsson, Steinþór},
  title   = {eflomal revisited: fast Bayesian word alignment in Rust},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/steinst/eflomal-revisited}
}
```

GitHub's "Cite this repository" uses [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0 (`LICENSE`, `NOTICE`). eflomal revisited is an independent implementation written from the
published description of the models; it is not affiliated with eflomal or its authors.
