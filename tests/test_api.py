from __future__ import annotations

import logging
import subprocess

import eflomalr
import pytest
from conftest import FAST


def lines(path):
    return path.read_text().split("\n")[:-1]


# -- training: inputs -----------------------------------------------------------------------


def test_train_sources_agree(model, corpus, corpus_files, joined_file):
    """Token lists, strings, two files and one `|||` file give the same alignments."""
    src, tgt, _ = corpus
    ref = model.alignments().to_bytes()
    strs = eflomalr.train(
        [" ".join(s) for s in src], [" ".join(t) + "\n" for t in tgt], threads=1, **FAST
    )
    assert strs.alignments().to_bytes() == ref
    a, b = corpus_files
    assert eflomalr.train(a, b, threads=3, **FAST).alignments().to_bytes() == ref
    assert eflomalr.train(str(a), str(b), **FAST).alignments().to_bytes() == ref
    assert eflomalr.train(joined_file, **FAST).alignments().to_bytes() == ref
    # a file and a list mix
    assert eflomalr.train(a, tgt, **FAST).alignments().to_bytes() == ref


@pytest.mark.parametrize("exact", [False, True])
def test_thread_count_does_not_change_output(corpus, exact):
    src, tgt, _ = corpus
    outs = {
        eflomalr.train(src, tgt, threads=t, exact=exact, scores=True, **FAST)
        .alignments(direction="fwd")
        .to_bytes()
        for t in (1, 2, 5)
    }
    assert len(outs) == 1


def test_shapes(model, corpus):
    src, tgt, _ = corpus
    assert model.directions == ("fwd", "inv")
    assert model.has_training_alignments and model.n_lines == len(src)
    assert model.vocab_size == (40, 40)
    assert model.skipped == []
    it = model.iterations
    assert len(it) == 3 and all(x > 0 for x in it)
    al = model.alignments()
    assert len(al) == len(src)
    for k in (0, 5, -1):
        links = al[k]
        assert all(0 <= i < len(src[k]) and 0 <= j < len(tgt[k]) for i, j in links)
    assert al[3:6] == [al[3], al[4], al[5]]
    assert list(al)[:3] == al[:3]
    assert al.to_list()[:3] == al[:3]
    assert al.pharaoh(0) == " ".join(f"{i}-{j}" for i, j in al[0])
    assert al.score(0) is None
    with pytest.raises(IndexError):
        al[len(al)]
    o = model.options
    assert (o["model"], o["samplers"], o["seed"], o["null_prior"]) == (3, 3, 1, 0.2)
    # the synthetic tokens are at most 3 characters, so auto's prefix 4 does not change them
    assert o["lowercase"] and o["max_len"] == 1023 and o["source_prefix"] == 4
    log = model.training_log("inv")
    assert "[inv sampler 3] model 3 iteration" in log and "[fwd" not in log
    assert "eflomalr.Model" in repr(model)


def test_alignment_quality(model, corpus):
    """The synthetic corpus is nearly monotone: most links are the gold ones."""
    perms = corpus[2]
    hits = total = 0
    for k, links in enumerate(model.alignments(direction="fwd")):
        gold = {(perms[k][j], j) for j in range(len(perms[k]))}
        hits += len(gold & set(links))
        total += len(gold)
    assert hits / total > 0.9


def test_directions_and_symmetrize(corpus, model):
    src, tgt, _ = corpus
    fwd = eflomalr.train(src, tgt, directions="fwd", threads=2, **FAST)
    inv = eflomalr.train(src, tgt, directions="inv", threads=2, **FAST)
    assert fwd.directions == ("fwd",) and inv.directions == ("inv",)
    # a direction trained alone equals that direction of a run of both
    assert fwd.alignments().to_bytes() == model.alignments(direction="fwd").to_bytes()
    assert inv.alignments().to_bytes() == model.alignments(direction="inv").to_bytes()
    with pytest.raises(ValueError, match="both directions"):
        fwd.alignments(symmetrize="union")
    with pytest.raises(ValueError, match="no 'inv'"):
        fwd.alignments(direction="inv")
    with pytest.raises(ValueError, match="not both"):
        model.alignments(symmetrize="union", direction="fwd")
    with pytest.raises(ValueError, match="unknown"):
        model.alignments(symmetrize="nope")
    f, i = model.alignments(direction="fwd"), model.alignments(direction="inv")
    for m in eflomalr.METHODS:
        s = model.alignments(symmetrize=m)
        for k in (0, 1, 17):
            assert s[k] == eflomalr.symmetrize(f[k], i[k], m)
    assert model.alignments(symmetrize="intersection").to_bytes() == (
        model.alignments(symmetrize="intersect").to_bytes()
    )
    assert (
        model.alignments().to_bytes()
        == model.alignments(symmetrize="grow-diag-final-and").to_bytes()
    )


def test_options_change_the_result(corpus):
    """Every sampling option reaches the core (seen through the scores, which follow the
    counts even where the links of this easy corpus coincide)."""
    src, tgt, _ = corpus

    def run(**kw):
        args = {**FAST, **kw}
        return eflomalr.train(src, tgt, directions="fwd", scores=True, threads=2, **args)

    base = run().alignments().to_bytes()
    for kw in (
        {"seed": 2},
        {"model": 1},
        {"model": 2},
        {"samplers": 1},
        {"length": 0.3},
        {"ibm1_iters": 3, "hmm_iters": 2, "fert_iters": 1},
        {"null_prior": 0.4},
        {"lex_alpha": 0.01},
        {"jump_alpha": 1.0},
        {"fert_alpha": 2.0},
        {"exact": True},
        {"accumulate": True},
        {"score_model": 1},
    ):
        assert run(**kw).alignments().to_bytes() != base, kw
    m = run(ibm1_iters=3, hmm_iters=2, fert_iters=1)
    assert m.iterations == (3, 2, 1)
    assert run(model=1).iterations[1:] == (0, 0)


def test_text_options():
    src = [["Aa", "BB"], ["aa", "bb"]] * 10
    tgt = [["xy", "Zz"], ["XY", "zz"]] * 10
    assert eflomalr.train(src, tgt, **FAST).vocab_size == (2, 2)
    assert eflomalr.train(src, tgt, lowercase=False, **FAST).vocab_size == (4, 4)
    m = eflomalr.train(src, tgt, source_prefix=1, target_suffix=1, **FAST)
    assert m.vocab_size == (2, 2) and m.options["source_prefix"] == 1
    assert eflomalr.train([["abc", "abd"]] * 5, [["x", "y"]] * 5, source_prefix=2).vocab_size == (
        1,
        2,
    )


def test_scores(corpus, tmp_path):
    src, tgt, _ = corpus
    m = eflomalr.train(src, tgt, directions="fwd", scores=True, **FAST)
    m.write_alignments(tmp_path / "s.out", direction="fwd")
    text = lines(tmp_path / "s.out")
    assert len(text) == len(src) and all(" ||| " in t for t in text)
    assert isinstance(m.alignments().score(0), float)
    plain = eflomalr.train(src, tgt, directions="fwd", **FAST)
    assert m.alignments().to_list() == plain.alignments().to_list()


def test_max_len_keeps_line_correspondence(corpus):
    src, tgt, _ = corpus
    m = eflomalr.train(src, tgt, directions="fwd", max_len=5, **FAST)
    al = m.alignments()
    assert len(al) == len(src)
    long = [k for k, (s, t) in enumerate(zip(src, tgt)) if len(s) > 5 or len(t) > 5]
    assert m.skipped == long and long
    for k in long:
        assert al[k] == []
    assert any(links for links in al)


def test_empty_sides_give_empty_alignments(corpus):
    src, tgt, _ = corpus
    s2, t2 = [list(x) for x in src[:50]], [list(x) for x in tgt[:50]]
    s2[3] = []
    t2[7] = []
    m = eflomalr.train(s2, t2, **FAST)
    for d in ("fwd", "inv"):
        al = m.alignments(direction=d)
        assert len(al) == 50 and al[3] == [] and al[7] == []


def test_progress_callback_and_logging(corpus, caplog):
    src, tgt, _ = corpus
    events = []
    with caplog.at_level(logging.INFO, logger="eflomalr"):
        m = eflomalr.train(
            src, tgt, threads=2, callback=events.append, ibm1_iters=2, hmm_iters=1, fert_iters=3
        )
    assert m.iterations == (2, 1, 3)
    got = sorted((e.direction, e.sampler, e.model, e.iteration) for e in events)
    want = sorted(
        (d, s, mo, i)
        for d in ("fwd", "inv")
        for s in range(3)
        for mo, n in ((1, 2), (2, 1), (3, 3))
        for i in range(1, n + 1)
    )
    assert got == want
    e = next(e for e in events if e.model == 3 and e.iteration == 2)
    assert e.iterations == 3 and 0 <= e.null_fraction <= 1 and e.seconds >= 0
    assert "model 3 iteration 2/3: NULL" in str(e)
    assert sum("iteration" in r.getMessage() for r in caplog.records) == len(want)


def test_callback_exception_stops_training(corpus):
    src, tgt, _ = corpus

    class Boom(Exception):
        pass

    def cb(_):
        raise Boom

    with pytest.raises(Boom):
        eflomalr.train(src, tgt, callback=cb)
    # the library is not poisoned
    assert len(eflomalr.train(src[:30], tgt[:30], **FAST).alignments()) == 30


# -- training: bad input --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"directions": "x"}, ValueError, "directions"),
        ({"model": 4}, ValueError, "model"),
        ({"model": True}, ValueError, "model"),
        ({"samplers": 0}, ValueError, "samplers"),
        ({"seed": -1}, ValueError, "seed"),
        ({"threads": -1}, ValueError, "threads"),
        ({"length": -0.5}, ValueError, "length"),
        ({"ibm1_iters": -1}, ValueError, "ibm1_iters"),
        ({"max_len": 0}, ValueError, "max_len"),
        ({"max_len": 70000}, ValueError, "max_len"),
        ({"source_prefix": -1}, ValueError, "source_prefix"),
        ({"null_prior": float("nan")}, ValueError, "null_prior"),
        ({"null_prior": 1.0}, ValueError, "null"),
        ({"lex_alpha": 0.0}, ValueError, "alpha"),
        ({"score_model": 0}, ValueError, "score_model"),
        ({"callback": 3}, TypeError, "callback"),
        ({"priors": 3}, TypeError, "priors"),
        ({"accumulate": True, "accumulate_limit": 0}, ValueError, "accumulate"),
    ],
)
def test_train_validation(corpus, kwargs, exc, match):
    src, tgt, _ = corpus
    with pytest.raises(exc, match=match):
        eflomalr.train(src[:10], tgt[:10], **kwargs)


def test_train_input_errors(corpus, tmp_path):
    src, tgt, _ = corpus
    with pytest.raises(ValueError, match="different numbers"):
        eflomalr.train(src[:10], tgt[:9])
    with pytest.warns(RuntimeWarning, match="no line pairs"):
        m = eflomalr.train([], [])  # an empty corpus aligns nothing, as the native binary
    assert m.n_lines == 0 and len(m.alignments()) == 0 and m.training_warnings
    with pytest.raises(TypeError):
        eflomalr.train(src[:3])  # a single list is not a joined file
    with pytest.raises(FileNotFoundError):
        eflomalr.train(tmp_path / "nope.src", tmp_path / "nope.tgt")
    with pytest.raises(FileNotFoundError):
        eflomalr.train(tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError):
        eflomalr.train(src[:3], tgt[:3], priors=tmp_path / "nope.priors")
    with pytest.raises(TypeError, match="bytes"):
        eflomalr.train([b"a b"], [b"x"])
    with pytest.raises(TypeError, match="tokens must be str"):
        eflomalr.train([["a", 1]], [["x"]])
    with pytest.raises(ValueError, match="whitespace"):
        eflomalr.train([["a b"]], [["x"]])
    with pytest.raises(ValueError, match="whitespace"):
        eflomalr.train([[""]], [["x"]])
    with pytest.raises(ValueError, match=r"\|\|\|"):
        eflomalr.train([["a", "|||"]], [["x"]])
    with pytest.raises(TypeError):
        eflomalr.train(5, 6)


def test_string_sentences_are_split_on_ascii_whitespace():
    m = eflomalr.train(["a  b\tc\n", "a b"], ["x y z", "x\ny"], directions="fwd", **FAST)
    assert m.n_lines == 2
    ref = eflomalr.train(
        [["a", "b", "c"], ["a", "b"]], [["x", "y", "z"], ["x", "y"]], directions="fwd", **FAST
    )
    assert m.alignments().to_bytes() == ref.alignments().to_bytes()
    assert m.vocab_size == (3, 3)


def test_non_ascii_tokens(tmp_path):
    src = [["Maison", "grüne", "日本語"], ["ÇA", "va"]] * 20
    tgt = [["house", "green", "Japanese"], ["it", "goes"]] * 20
    m = eflomalr.train(src, tgt, threads=1, **FAST)
    assert m.vocab_size == (5, 5)
    path = tmp_path / "m.eflomalr"
    m.save(path)
    loaded = eflomalr.Model.load(path)
    assert loaded.align(["日本語"], ["Japanese"]) == [(0, 0)]
    assert loaded.align(["ça", "VA"], ["it", "goes"], fast=True) == [(0, 0), (1, 1)]


# -- save / load / aligning new text --------------------------------------------------------


def test_save_load_roundtrip(model, corpus, tmp_path):
    src, tgt, _ = corpus
    path = tmp_path / "m.eflomalr"
    model.save(path)
    assert not (tmp_path / "m.eflomalr.tmp").exists()
    loaded = eflomalr.Model.load(path)
    assert loaded.directions == model.directions and loaded.vocab_size == model.vocab_size
    assert loaded.options == model.options
    assert not loaded.has_training_alignments and loaded.n_lines is None
    assert loaded.skipped is None and loaded.iterations is None
    with pytest.raises(ValueError, match="loaded from a file"):
        loaded.alignments()
    with pytest.raises(ValueError, match="loaded from a file"):
        loaded.write_alignments(tmp_path / "x")
    with pytest.raises(ValueError, match="loaded from a file"):
        loaded.training_log()
    # a loaded and the trained model align new text identically
    for kw in ({"direction": "fwd"}, {"direction": "inv"}, {"symmetrize": "grow-diag-final-and"}):
        for fast in (False, True):
            a = loaded.align_many(src[:60], tgt[:60], fast=fast, **kw).to_bytes()
            assert a == model.align_many(src[:60], tgt[:60], fast=fast, **kw).to_bytes()


def test_align_with_model(model, corpus):
    src, tgt, perms = corpus
    new = model.align_many(src[:100], tgt[:100], direction="fwd", threads=2)
    assert len(new) == 100
    hits = sum(
        len({(perms[k][j], j) for j in range(len(perms[k]))} & set(new[k])) for k in range(100)
    )
    assert hits / sum(len(p) for p in perms[:100]) > 0.9
    # deterministic, at any thread count, and the model's seed is the default
    for t in (1, 3):
        assert model.align_many(src[:100], tgt[:100], direction="fwd", threads=t).to_bytes() == (
            new.to_bytes()
        )
    assert model.align_many(src[:100], tgt[:100], direction="fwd", seed=1).to_bytes() == (
        new.to_bytes()
    )
    scored = model.align_many(src[:100], tgt[:100], direction="fwd", seed=2, scores=True)
    assert scored.to_bytes() != new.to_bytes() and scored.score(0) is not None
    # a direction alone equals that direction of the symmetrised call's input
    fwd = model.align_many(src[:100], tgt[:100], direction="fwd")
    inv = model.align_many(src[:100], tgt[:100], direction="inv")
    sym = model.align_many(src[:100], tgt[:100], symmetrize="union")
    assert sym.to_list() == eflomalr.symmetrize_many(fwd.to_list(), inv.to_list(), "union")


def test_align_one_pair(model, corpus):
    src, tgt, _ = corpus
    assert model.align(src[0], tgt[0], fast=True) == model.align(
        " ".join(src[0]), " ".join(tgt[0]), fast=True
    )
    assert model.align(["s1"], ["t1"], direction="fwd") == [(0, 0)]
    assert model.align(["s1", "s2"], ["t1", "t2"]) == [(0, 0), (1, 1)]
    assert model.align([], ["t1"]) == []
    # unseen words are handled
    links = model.align(["zzz", "s1"], ["t1", "qqq"], direction="fwd")
    assert all(0 <= i < 2 and 0 <= j < 2 for i, j in links)


def test_align_many_inputs(model, corpus, corpus_files, joined_file):
    src, tgt, _ = corpus
    a, b = corpus_files
    ref = model.align_many(src, tgt, direction="fwd").to_bytes()
    assert model.align_many(a, b, direction="fwd").to_bytes() == ref
    assert model.align_many(joined_file, direction="fwd").to_bytes() == ref
    with pytest.raises(ValueError, match="different numbers"):
        model.align_many(src[:5], tgt[:4])
    with pytest.raises(ValueError, match="scores"):
        model.align_many(src, tgt, symmetrize="union", scores=True)
    with pytest.raises(ValueError, match="fast"):
        model.align_many(src, tgt, direction="fwd", fast=True, scores=True)
    with pytest.raises(ValueError, match="fast"):
        model.align_many(src, tgt, fast=True, exact=True)
    with pytest.raises(ValueError, match="samplers"):
        model.align_many(src, tgt, samplers=0)


def test_align_uses_the_models_text_options(tmp_path):
    """A model trained with stemming reads new text with the same stemming."""
    src = [["walking", "dogs"], ["walked", "cats"]] * 20
    tgt = [["x1", "x2"], ["x1", "x3"]] * 20
    m = eflomalr.train(src, tgt, source_prefix=4, **FAST)
    m.save(tmp_path / "m")
    loaded = eflomalr.Model.load(tmp_path / "m")
    assert loaded.options["source_prefix"] == 4
    assert loaded.align(["walkers"], ["x1"], fast=True) == [(0, 0)]


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        eflomalr.Model.load(tmp_path / "missing")
    bad = tmp_path / "bad"
    bad.write_bytes(b"not a model at all")
    with pytest.raises(ValueError, match="bad"):
        eflomalr.Model.load(bad)


def test_truncated_model_is_a_clean_error(model, tmp_path):
    path = tmp_path / "m.eflomalr"
    model.save(path)
    data = path.read_bytes()
    for cut in (0, 7, 100, len(data) // 2, len(data) - 1):
        p = tmp_path / f"t{cut}"
        p.write_bytes(data[:cut])
        with pytest.raises(ValueError):
            eflomalr.Model.load(p)


# -- priors ----------------------------------------------------------------------------------


def test_make_priors_and_train_with_them(model, corpus, corpus_files, tmp_path):
    src, tgt, _ = corpus
    fwd, inv = model.alignments(direction="fwd"), model.alignments(direction="inv")
    data = eflomalr.make_priors(src, tgt, forward=fwd, reverse=inv)
    kinds = {ln.split(b"\t")[0] for ln in data.splitlines()}
    assert kinds == {b"LEX", b"HMMF", b"HMMR", b"FERF", b"FERR"}
    # every way of giving the links agrees, and output= writes the same bytes
    a, b = corpus_files
    model.write_alignments(tmp_path / "f", direction="fwd")
    out = tmp_path / "p.priors"
    assert eflomalr.make_priors(a, b, forward=fwd.to_list(), reverse=inv.to_list()) == data
    assert eflomalr.make_priors(a, b, forward=tmp_path / "f", reverse=inv, output=out) == data
    assert out.read_bytes() == data
    # training on a small part with the priors helps on this corpus
    with_p = eflomalr.train(src[:20], tgt[:20], priors=out, directions="fwd", **FAST)
    assert len(with_p.alignments()) == 20
    with pytest.raises(TypeError, match="forward"):
        eflomalr.make_priors(src, tgt, forward=b"0-0", reverse=inv)
    bad = tmp_path / "bad.priors"
    bad.write_text("LEX\tonly two\n")
    with pytest.raises(ValueError, match="bad.priors"):
        eflomalr.train(src[:5], tgt[:5], priors=bad)


def test_make_priors_equals_native(model, corpus_files, tmp_path, native_binary):
    a, b = corpus_files
    model.write_alignments(tmp_path / "f", direction="fwd")
    model.write_alignments(tmp_path / "r", direction="inv")
    native = subprocess.run(
        [
            str(native_binary),
            "makepriors",
            str(a),
            str(b),
            "-f",
            str(tmp_path / "f"),
            "-r",
            str(tmp_path / "r"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    assert eflomalr.make_priors(a, b, forward=tmp_path / "f", reverse=tmp_path / "r") == native


# -- symmetrisation --------------------------------------------------------------------------


def test_symmetrize_functions():
    fwd = [(0, 0), (1, 1), (2, 1)]
    inv = [(0, 0), (1, 1), (1, 2)]
    assert eflomalr.symmetrize(fwd, inv, "intersect") == [(0, 0), (1, 1)]
    assert eflomalr.symmetrize(fwd, inv, "intersection") == [(0, 0), (1, 1)]
    assert eflomalr.symmetrize(fwd, inv, "union") == [(0, 0), (1, 1), (1, 2), (2, 1)]
    assert eflomalr.symmetrize_many([fwd, []], [inv, []], "grow-diag")[1] == []
    assert eflomalr.invert(fwd) == [(0, 0), (1, 1), (1, 2)]
    assert eflomalr.parse_pharaoh("0-0 1-2") == [(0, 0), (1, 2)]
    assert eflomalr.parse_pharaoh("0-0 ||| -3.5") == [(0, 0)]
    assert eflomalr.parse_pharaoh("") == []
    with pytest.raises(ValueError):
        eflomalr.parse_pharaoh("0:1")
    with pytest.raises(ValueError, match="unknown"):
        eflomalr.symmetrize(fwd, inv, "nope")
    with pytest.raises(ValueError, match="invert"):
        eflomalr.symmetrize(fwd, inv, "invert")
    with pytest.raises(ValueError, match="same size"):
        eflomalr.symmetrize_many([fwd], [], "union")


def test_symmetrize_matches_atools(model, tmp_path, atools_binary):
    """Every method, on the model's own directional alignments, equals `atools -c METHOD`."""
    model.write_alignments(tmp_path / "f", direction="fwd")
    model.write_alignments(tmp_path / "i", direction="inv")
    for m in eflomalr.METHODS:
        want = subprocess.run(
            [str(atools_binary), "-c", m, "-i", str(tmp_path / "f"), "-j", str(tmp_path / "i")],
            capture_output=True,
            check=True,
        ).stdout
        assert model.alignments(symmetrize=m).to_bytes() == want, m


# -- identical to the native binary ----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, flags",
    [
        ({}, []),
        ({"seed": 7, "samplers": 2, "model": 2}, ["--seed", "7", "-n", "2", "-m", "2"]),
        (
            {"ibm1_iters": 4, "hmm_iters": 3, "fert_iters": 5, "null_prior": 0.3},
            ["-1", "4", "-2", "3", "-3", "5", "--null-prior", "0.3"],
        ),
        ({"exact": True, "scores": True, "score_model": 1}, ["--exact", "--scores", "-M", "1"]),
        ({"accumulate": True, "max_len": 6}, ["--accumulate", "--max-len", "6"]),
        (
            {"lowercase": False, "source_prefix": 2, "target_suffix": 1},
            ["--no-lowercase", "--source-prefix", "2", "--target-suffix", "1"],
        ),
    ],
)
def test_identical_to_native_binary(joined_file, native_binary, tmp_path, kwargs, flags):
    prefix = tmp_path / "n"
    kwargs = {**FAST, **kwargs}
    flags = ["-l", "0.2", *flags]
    subprocess.run(
        [
            str(native_binary),
            "train",
            str(joined_file),
            "-o",
            str(prefix),
            "-s",
            "union",
            "-q",
            "--save",
            str(tmp_path / "n.model"),
            *flags,
        ],
        check=True,
    )
    py = eflomalr.train(joined_file, threads=3, **kwargs)
    for d in ("fwd", "inv"):
        assert (tmp_path / f"n.{d}.pharaoh").read_bytes() == py.alignments(direction=d).to_bytes()
    assert (tmp_path / "n.union.pharaoh").read_bytes() == py.alignments(
        symmetrize="union"
    ).to_bytes()
    py.save(tmp_path / "p.model")
    assert (tmp_path / "n.model").read_bytes() == (tmp_path / "p.model").read_bytes()


def test_fast_decoding_does_not_depend_on_the_batch():
    """The cat/dog dictionary, through the cached fast path."""
    m = eflomalr.train(
        ["cat", "dog"] * 100,
        ["chat", "chien"] * 100,
        threads=1,
        seed=1,
        segment="words",
        ibm1_iters=2,
        hmm_iters=2,
        fert_iters=2,
    )
    for d in ("fwd", "inv"):
        one = m.align_many(["dog cat"], ["chien"], fast=True, direction=d, threads=1)
        two = m.align_many(["dog cat", "dog cat"], ["chien", "chat"], fast=True, direction=d)
        assert one.to_bytes() == b"0-0\n" and two.to_bytes() == b"0-0\n1-0\n", d
