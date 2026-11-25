"""Pure-function tests for ai_trader.research.intraday_geometry (block GEO)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from ai_trader.research import intraday_geometry as G


def _machine_one(closes, highs, lows, level, s1h=0.01):
    lc, lh, ll = np.log(closes), np.log(highs), np.log(lows)
    sig = np.full(len(lc), s1h)
    W = len(lc) - 1
    L = np.full((1, W), math.log(level))
    return G.machine(lh, ll, lc, sig, np.array([1]), W, L, L, L[:, 0], L[:, 0])


def test_machine_from_below_touch_accept_retest():
    # level 100; window bars 1..; start well below (arm), touch, accept, pull back to the level and hold
    c = np.array([97, 98, 99, 99.8, 100.5, 101, 100.2, 101.5, 102, 103.0])
    h = np.array([97.2, 98.2, 99.3, 100.05, 100.7, 101.2, 101.1, 101.6, 102.2, 103.1])
    lo = np.array([96.8, 97.8, 98.8, 99.5, 99.9, 100.6, 99.95, 100.9, 101.8, 102.8])
    r = _machine_one(c, h, lo, 100.0)
    assert r["s"][0] == 1
    # window offsets (bar index - 1): touch at bar 3 (high 100.1), accept at bar 4 (close 100.5), retest at bar 6
    assert r["touch"][0] == 2
    assert r["accept"][0] == 3
    assert r["retest"][0] == 5
    assert r["sweep"][0] == r["W"]  # no sweep before acceptance


def test_machine_sweep_and_mirror_from_above():
    # from above: price falls, wicks below the level by more than 0.1 sigma and closes back above -> sweep
    c = np.array([104, 103, 102, 100.6, 101.5, 102.0])
    h = np.array([104.2, 103.5, 102.3, 101.0, 101.8, 102.2])
    lo = np.array([103.8, 102.8, 101.6, 99.5, 100.9, 101.7])
    r = _machine_one(c, h, lo, 100.0)
    assert r["s"][0] == -1
    assert r["touch"][0] == 2
    assert r["sweep"][0] == 2
    assert r["accept"][0] == r["W"]


def test_machine_never_armed():
    c = np.full(6, 100.0)
    r = _machine_one(c, c * 1.0001, c * 0.9999, 100.0)
    assert r["s"][0] == 0
    assert r["touch"][0] == r["W"]


def test_diff_test_and_bh():
    rng = np.random.default_rng(0)
    D = 400
    nr = np.full(D, 5.0)
    npl = np.full(D, 10.0)
    sr = rng.normal(0.2, 1, D) * np.sqrt(5) + 0.0
    sp = rng.normal(0.0, 1, D) * np.sqrt(10)
    out = G.diff_test(np.arange(D), sr, nr, sp, npl)
    assert out["n"] == 2000 and out["n_pl"] == 4000
    assert abs(out["diff"] - (sr.sum() / 2000 - sp.sum() / 4000)) < 1e-12
    rej = G.bh(np.array([0.001, 0.02, 0.5, 0.9, np.nan]), 0.1)
    assert rej.tolist() == [True, True, False, False, False]


def test_round_step_and_grid_fill():
    assert G.round_step(91234.0) == 1000.0
    assert G.round_step(3123.0) == 100.0
    assert G.round_step(0.55) == 0.01
    ts = np.array([0, 1, 3], dtype=np.int64) * G.BAR_NS
    t0, g = G.grid_fill(ts, *(np.array([1.0, 2.0, 4.0]),) * 4, np.ones(3), np.ones(3))
    assert t0 == 0 and len(g["c"]) == 4
    assert g["c"].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert g["real"].tolist() == [True, True, False, True]
    assert g["v"].tolist() == [1.0, 1.0, 0.0, 1.0]


def test_sigma_slow_is_causal():
    r = np.r_[np.full(300, 0.01), np.full(10, 1.0)]
    s = G.sigma_slow(r, win=50, min_n=10)
    assert abs(s[300] - 0.01) < 1e-12  # bar 300 does not see its own return
    assert s[301] > 0.01


def test_pivot_zones_match_levels_script():
    spec = importlib.util.spec_from_file_location("levels_script", Path(__file__).resolve().parents[1] / "scripts" / "levels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(3)
    n = 400
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    ts = list(range(n))

    class _T:  # levels.zones formats timestamps with strftime
        def __init__(self, i):
            self.i = i

        def strftime(self, _f):
            return str(self.i)

    ref, _ = mod.zones([_T(i) for i in ts], o, h, lo, c, k=3, top=8)
    got = G.pivot_zones(o, h, lo, c, k=3, top=8)
    assert len(ref) == len(got)
    for a, b in zip(ref, got, strict=True):
        assert math.isclose(a["lo"], b["lo"], rel_tol=1e-5) and math.isclose(a["hi"], b["hi"], rel_tol=1e-5)
        assert a["pivots"] == b["pivots"] and a["rejections"] == b["rejections"]


def test_simulate_stop_target_time():
    n = 60
    c = np.full(n, 100.0)
    b = G.Bars("X", 0, c.copy(), c * 1.0005, c * 0.9995, c.copy(), np.ones(n), np.ones(n), np.ones(n, bool), np.zeros(n), np.full(n, 0.001))
    sig1h = 0.001 * math.sqrt(12)
    # bar 11 trades 3 sigma_1h higher: a long with a 2R target at 1 sigma stop hits the target
    b.h[11] = 100 * math.exp(3 * sig1h)
    fc = (np.array([0], np.int64), np.array([0.0, 0.0]))
    r = G.simulate(b, np.array([10]), np.array([100.0]), np.array([1], np.int8), np.array([2.0]), np.array([0.001]), 1.0, 2.0, 12, np.ones(1), fc)
    exp_t = (2 * sig1h - 4e-4) / sig1h
    assert abs(r[0] - exp_t) < 1e-9
    # a bar 2 sigma lower stops it out first (worst case) even if the target is also touched in the same bar
    b.lo[11] = 100 * math.exp(-2 * sig1h)
    r = G.simulate(b, np.array([10]), np.array([100.0]), np.array([1], np.int8), np.array([2.0]), np.array([0.001]), 1.0, 2.0, 12, np.ones(1), fc)
    exp_s = (-sig1h - 2e-4 - 7.5e-4) / sig1h
    assert abs(r[0] - exp_s) < 1e-9
    # no stop, no target: time exit at the close, taker + slippage
    b.h[11], b.lo[11] = 100.05, 99.95
    r = G.simulate(b, np.array([10]), np.array([100.0]), np.array([1], np.int8), np.array([2.0]), np.array([0.001]), 1.0, 0.0, 12, np.ones(1), fc)
    assert abs(r[0] - (-2e-4 - 7.5e-4) / sig1h) < 1e-9


def test_how_cells_monday_zero():
    from datetime import UTC, datetime

    t = np.array([int(datetime(2026, 11, 23, 0, tzinfo=UTC).timestamp() * 1e9), int(datetime(2026, 11, 29, 23, tzinfo=UTC).timestamp() * 1e9)])
    assert G.how_cells(t).tolist() == [0, 167]
