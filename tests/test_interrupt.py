"""Ctrl-C handling: training stops promptly, cleanly, and the library stays usable."""

from __future__ import annotations

import _thread
import os
import signal
import sys
import threading
import time

import eflomalr
import pytest
from conftest import make_corpus

pytestmark = pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(), reason="needs the main thread"
)


def _interrupt_after(delay: float) -> threading.Timer:
    send = (
        _thread.interrupt_main
        if sys.platform == "win32"
        else (lambda: os.kill(os.getpid(), signal.SIGINT))
    )
    t = threading.Timer(delay, send)
    t.start()
    return t


def test_sigint_interrupts_train_quickly():
    src, tgt, _ = make_corpus(n=20000, vocab=2000, max_len=30, min_len=20, seed=5)
    timer = _interrupt_after(0.5)
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            eflomalr.train(src, tgt, threads=2, length=50)
    finally:
        timer.cancel()
    assert time.perf_counter() - t0 < 10.0
    # Not poisoned: a following call works and is not cancelled.
    m = eflomalr.train(src[:50], tgt[:50], threads=1, length=0.2)
    assert len(m.alignments()) == 50


def test_sigint_interrupts_align_many():
    src, tgt, _ = make_corpus(n=200, vocab=40, seed=9)
    m = eflomalr.train(src, tgt, threads=2, length=0.2)
    big_s, big_t, _ = make_corpus(n=50_000, vocab=40, max_len=40, min_len=30, seed=11)
    timer = _interrupt_after(0.3)
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            m.align_many(big_s, big_t, threads=1, length=20)
    finally:
        timer.cancel()
    assert time.perf_counter() - t0 < 10.0
    assert len(m.align_many(src[:5], tgt[:5])) == 5


def test_gil_is_released_during_train():
    """Another Python thread keeps running while training."""
    src, tgt, _ = make_corpus(n=5000, vocab=1000, max_len=30, min_len=20, seed=3)
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.005)

    th = threading.Thread(target=tick)
    th.start()
    try:
        t0 = time.perf_counter()
        eflomalr.train(src, tgt, threads=1)
        elapsed = time.perf_counter() - t0
    finally:
        stop.set()
        th.join()
    assert elapsed > 0.3, "corpus too small to prove anything"
    assert len(ticks) > elapsed / 0.05


def _run(results, name, fn):
    try:
        results[name] = fn()
    except BaseException as e:  # noqa: BLE001 - the test inspects what was raised
        results[name] = e


def test_callback_error_cancels_other_active_calls_only():
    """Call A's callback raises while call B is held in its first progress
    callback; A returns, then B is released. B must stop (it used to run to completion because
    A cleared the shared flag on exit), and a call started after the cancellation must run."""
    src, tgt, _ = make_corpus(n=400, vocab=60, seed=4)
    b_started, b_release = threading.Event(), threading.Event()
    b_events = []

    def cb_b(p):
        b_events.append(p)
        if len(b_events) == 1:
            b_started.set()
            b_release.wait(30)

    def cb_a(p):
        raise ValueError("stop A")

    results = {}
    kw = {"threads": 1, "ibm1_iters": 4, "hmm_iters": 0, "fert_iters": 0, "samplers": 1}
    tb = threading.Thread(
        target=_run,
        args=(results, "B", lambda: eflomalr.train(src, tgt, callback=cb_b, model=1, **kw)),
    )
    tb.start()
    assert b_started.wait(30)
    ta = threading.Thread(
        target=_run,
        args=(results, "A", lambda: eflomalr.train(src, tgt, callback=cb_a, model=1, **kw)),
    )
    ta.start()
    ta.join(30)
    assert isinstance(results["A"], ValueError) and str(results["A"]) == "stop A"
    # a third call started after the cancellation (while B is still held) is not affected
    c_events = []
    m = eflomalr.train(src, tgt, callback=c_events.append, model=1, **kw)
    assert len(m.alignments()) == len(src) and len(c_events) >= 4
    b_release.set()
    tb.join(30)
    assert isinstance(results["B"], KeyboardInterrupt), results["B"]
    assert len(b_events) == 1, "B kept running after the cancellation"


def test_sigint_interrupts_a_long_fast_decode():
    """The fast HMM polls inside long sentences (it used to check every 256 sentences)."""
    src, tgt, _ = make_corpus(n=200, vocab=40, seed=9)
    m = eflomalr.train(src, tgt, threads=1, length=0.2)
    words = [w for s in src for w in s]
    long_s = [" ".join(words[k : k + 1000]) for k in range(300)]
    long_t = [" ".join(reversed(s.split())) for s in long_s]
    timer = _interrupt_after(0.3)
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            m.align_many(long_s, long_t, threads=1, fast=True, direction="fwd")
    finally:
        timer.cancel()
    assert time.perf_counter() - t0 < 3.0
    assert len(m.align_many(src[:5], tgt[:5], fast=True)) == 5
