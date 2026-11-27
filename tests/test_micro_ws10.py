import math

import numpy as np

from ai_trader.research.micro_ws10 import _peg, walk_premium
from ai_trader.research.micro_ws10_fit import dm_test, vecm_is


def test_walk_premium_inside_l1_is_half_spread():
    hs = np.array([0.5])
    p = walk_premium(100.0, np.array([1000.0]), np.array([5000.0]), np.array([9000.0]), np.array([20000.0]), hs)
    assert abs(p[0] - 0.5) < 1e-12


def test_walk_premium_grows_beyond_l1():
    hs = np.array([0.5])
    a = walk_premium(2000.0, np.array([1000.0]), np.array([5000.0]), np.array([9000.0]), np.array([20000.0]), hs)[0]
    b = walk_premium(4000.0, np.array([1000.0]), np.array([5000.0]), np.array([9000.0]), np.array([20000.0]), hs)[0]
    assert 0.5 < a < b < 5.0
    # first 1000 at 0.5, next 1000 linear from 0.5 towards 5 over 4000 dollars: mean of that slice 0.5 + 4.5/4000*500
    assert abs(a - (1000 * 0.5 + 1000 * (0.5 + 4.5 / 4000 * 500)) / 2000) < 1e-9


def test_dm_detects_better_model():
    rng = np.random.default_rng(0)
    d = 0.2 + rng.standard_normal(2000)
    r = dm_test(d, lag=5)
    assert r["t"] > 3


def test_vecm_is_leader_gets_share():
    rng = np.random.default_rng(1)
    n = 20000
    e = rng.standard_normal(n) * 1e-4
    p1 = np.cumsum(e)  # leader
    p2 = np.r_[0.0, p1[:-1]] + rng.standard_normal(n) * 2e-5  # follows with one lag
    r = vecm_is(np.column_stack([p1, p2]), pmax=3)
    assert r["is_lower"] > 0.8 and r["gg"] > 0.8


def _toy_book(n=200, bid=100.0, ask=100.1, size=10.0):
    ts = np.arange(1, n + 1) * 1000
    b = np.full(n, bid)
    a = np.full(n, ask)
    m = (b + a) / 2
    return ts, b, a, m, b * size, a * size


def test_peg_fills_after_queue_ahead_consumed():
    ts, bid, ask, mid, l1b, l1a = _toy_book()
    # one sell print of 2000 USD at the bid in second 3 (queue ahead 1000 + our 100): filled at level
    ms = np.array([2500])
    osd, no, fp, lp = np.array([-1]), np.array([2000.0]), np.array([100.0]), np.array([100.0])
    oi = np.searchsorted(ts, ms, "left")
    ss = np.searchsorted(oi, np.arange(len(ts) + 1), "left")
    how, t, px, rep, *_ = _peg(0, 1, False, bid, ask, mid, l1b, l1a, ts, ss, ms, osd, no, fp, lp, 0.025, len(ts))
    assert how == 2 and px == 100.0 and math.isclose(t, 1.5)


def test_peg_unfilled_falls_back_to_taker():
    ts, bid, ask, mid, l1b, l1a = _toy_book()
    ms = np.array([], dtype=np.int64)
    e = np.array([], dtype=float)
    oi = np.searchsorted(ts, ms, "left")
    ss = np.searchsorted(oi, np.arange(len(ts) + 1), "left")
    how, t, px, *_ = _peg(0, 1, False, bid, ask, mid, l1b, l1a, ts, ss, ms, e.astype(np.int64), e, e, e, 0.025, len(ts))
    assert how == 0 and px == ask[61]
