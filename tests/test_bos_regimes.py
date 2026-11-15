"""Mechanics of the BOS study engine (src/ai_trader/research/bos_regimes_study.py)."""

import numpy as np

from ai_trader.features import vol_bos as vb
from ai_trader.research import bos_regimes_study as B


def _path(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, n))
    return o, h, lo, c


def test_fractal_structure_matches_vol_bos():
    o, h, lo, c = _path()
    for k in (2, 3, 5):
        b1, ch1 = vb.structure(c, h, lo, k)
        r = B.fractal_structure(c, h, lo, k)
        assert (b1 == r[0]).all() and (ch1 == r[1]).all()
        assert np.isfinite(r[2][r[0] != 0]).all()


def test_structure_is_causal():
    o, h, lo, c = _path()
    full = B.fractal_structure(c, h, lo, 3)
    cut = B.fractal_structure(c[:300], h[:300], lo[:300], 3)
    assert (full[0][:300] == cut[0]).all()
    dfull = B.donchian_structure(c, h, lo, 20)
    dcut = B.donchian_structure(c[:300], h[:300], lo[:300], 20)
    assert (dfull[0][:300] == dcut[0]).all()


def _sim(o, h, lo, c, side, stop, su, entry_bar, px, intrabar=False, H=5, slip=0.0, fee=0.0, sof=True, opp=None):
    n = len(c)
    P = {"o": np.asarray(o, float), "h": np.asarray(h, float), "l": np.asarray(lo, float), "c": np.asarray(c, float)}
    ts = np.arange(n, dtype=np.int64) * 3_600_000_000
    fund = (np.array([0], dtype=np.int64), np.array([0.0]))
    opp_lo = np.full(n, -1e9) if opp is None else opp
    opp_hi = np.full(n, 1e9)
    sd = np.array([side], np.int8)
    R = np.array([abs(c[0] - stop * side)])
    return B.simulate(P, np.array([0]), sd, np.array([stop]), np.array([su]), opp_lo, opp_hi, np.array([entry_bar]),
                      np.array([px * side]), np.array([intrabar]), H, ts, fund, R, slip, fee, sof)


def test_long_stop_and_time_exit():
    old = (B.EXIT_SLIP, B.EXIT_FEE)
    B.EXIT_SLIP, B.EXIT_FEE = 0.0, 0.0
    try:
        # signal at bar 0 close 100, stop 98 (R = 2); enter at bar 1 open 100; bar 3 low 97 hits the stop
        o = [100, 100, 101, 100, 99, 99, 99, 99]
        h = [100, 101, 102, 100, 99, 99, 99, 99]
        lo = [99, 99.5, 100.5, 97, 98.5, 98.5, 98.5, 98.5]
        c = [100, 101, 101, 98.5, 99, 99, 99, 99]
        r = _sim(o, h, lo, c, 1, 98.0, 100.0, 1, 100.0)
        assert np.isclose(r["time"][0][0], -1.0)          # stopped at 98: (98 - 100) / 2
        # no stop: time exit after H = 3 bars at the open of bar 4
        r = _sim(o, h, [99, 99.5, 100.5, 99.0, 98.5, 98.5, 98.5, 98.5], c, 1, 98.0, 100.0, 1, 100.0, H=3)
        assert np.isclose(r["time"][0][0], (99 - 100) / 2)
    finally:
        B.EXIT_SLIP, B.EXIT_FEE = old


def test_short_is_mirrored_long():
    old = (B.EXIT_SLIP, B.EXIT_FEE)
    B.EXIT_SLIP, B.EXIT_FEE = 0.0, 0.0
    try:
        o = [100, 100, 99, 100, 101, 101, 101, 101]
        h = [101, 100.5, 99.5, 103, 101.5, 101.5, 101.5, 101.5]
        lo = [100, 99, 98, 100, 101, 101, 101, 101]
        c = [100, 99, 99, 101.5, 101, 101, 101, 101]
        r = _sim(o, h, lo, c, -1, -102.0, 100.0, 1, 100.0)  # mirrored stop = -102
        assert np.isclose(r["time"][0][0], -1.0)
    finally:
        B.EXIT_SLIP, B.EXIT_FEE = old


def test_retest_stop_first_vs_fill_only():
    old = (B.EXIT_SLIP, B.EXIT_FEE)
    B.EXIT_SLIP, B.EXIT_FEE = 0.0, 0.0
    try:
        # limit filled intrabar at 99 on bar 1, the same bar also trades the stop at 98
        o = [100, 100, 100, 101, 102, 102, 102, 102]
        h = [100, 100.5, 101, 102, 102, 102, 102, 102]
        lo = [99, 97.5, 99.5, 100.5, 101.5, 101.5, 101.5, 101.5]
        c = [100, 100, 101, 102, 102, 102, 102, 102]
        worst = _sim(o, h, lo, c, 1, 98.0, 100.0, 1, 99.0, intrabar=True, H=3, sof=True)
        best = _sim(o, h, lo, c, 1, 98.0, 100.0, 1, 99.0, intrabar=True, H=3, sof=False)
        assert np.isclose(worst["time"][0][0], (98 - 99) / 2)
        assert np.isclose(best["time"][0][0], (102 - 99) / 2)
    finally:
        B.EXIT_SLIP, B.EXIT_FEE = old
