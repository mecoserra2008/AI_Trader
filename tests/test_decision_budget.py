from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.research.decision_budget_study import (
    EMBARGO,
    HORIZON,
    block_bootstrap_idx,
    brier_skill,
    half_life,
    holm,
    lag_onto,
    purged_train_mask,
)
from ai_trader.research.decision_budget_study import test_mask as window_test_mask
from ai_trader.research.evaluate import strategy_pnl

US = 1_000_000


def _daily_decisions(n: int = 400):
    t0 = np.datetime64("2024-01-01T00:00:00", "us")
    T = t0 + np.arange(n) * np.timedelta64(86400 * US, "us")
    return T, T + np.timedelta64(int(HORIZON.total_seconds() * US), "us")


def test_purge_embargo_leaves_no_overlap():
    T, le = _daily_decisions()
    for start in (np.datetime64("2024-03-01T00:00:00", "us"), np.datetime64("2024-07-01T00:00:00", "us")):
        end = start + np.timedelta64(31 * 86400 * US, "us")
        for w in (None, 90):
            tr = purged_train_mask(le, start, window_days=w)
            te = window_test_mask(T, start, end)
            assert tr.any() and te.any()
            assert not (tr & te).any()
            # every training label interval [T, T+24h] ends at least the embargo before the first test decision
            gap = T[te].min() - le[tr].max()
            assert gap >= np.timedelta64(int(EMBARGO.total_seconds() * US), "us")
            # no training label interval intersects any test label interval
            assert le[tr].max() < T[te].min()
            if w is not None:
                assert (le[tr] > start - np.timedelta64((w + 1) * 86400 * US, "us")).all()
                assert tr.sum() == min(w, purged_train_mask(le, start).sum())
    # the label that ends exactly at test_start - 1 day is kept, the next one is purged
    start = np.datetime64("2024-03-01T00:00:00", "us")
    tr = purged_train_mask(le, start)
    assert le[tr].max() == start - np.timedelta64(86400 * US, "us")


def test_lag_onto_same_grid_and_4h_to_1h():
    t = np.arange(20, dtype=np.int64) * 3600 * US
    c = t + 3600 * US
    s = np.arange(20, dtype=float) + 1
    assert np.array_equal(lag_onto(c, c, s, 0), s)
    lagged = lag_onto(c, c, s, 2)
    assert np.array_equal(lagged[2:], s[:-2]) and np.all(lagged[:2] == 0)
    # 4h source: signal of the 4h bar starting at 0h is available at its close, 4h
    t4 = np.arange(5, dtype=np.int64) * 4 * 3600 * US
    a4 = t4 + 4 * 3600 * US
    s4 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    on1 = lag_onto(c, a4, s4, 0)
    # 1h rows whose close is 4h..7h (rows 3..6) hold the first 4h signal; the row closing at 3h holds nothing
    assert on1[2] == 0 and np.all(on1[3:7] == 10.0) and on1[7] == 20.0
    on1_d1 = lag_onto(c, a4, s4, 1)
    assert on1_d1[3] == 0 and np.all(on1_d1[4:8] == 10.0) and on1_d1[8] == 20.0


def test_lagging_signal_shifts_pnl():
    rng = np.random.default_rng(3)
    n = 200
    ts = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
    fwd = rng.normal(0, 0.01, n)
    frame = pl.DataFrame({"event_ts": ts, "fwd_ret": fwd, "sigma_next": np.ones(n)})
    sig = np.sign(rng.normal(size=n))
    c = (np.arange(n, dtype=np.int64) + 1) * 3600 * US
    for d in (0, 1, 2, 4, 8):
        lagged = lag_onto(c, c, sig, d)
        p = strategy_pnl(frame, pl.Series(lagged), cost_bps=0.0)["pnl"].to_numpy()
        expect = np.r_[np.zeros(d), sig[: n - d]] * fwd
        assert np.allclose(p, expect)
    # with costs, a delay never changes the number of trades (only when they happen)
    p0 = strategy_pnl(frame, pl.Series(lag_onto(c, c, sig, 0)), cost_bps=10.0)["cost"].sum()
    p4 = strategy_pnl(frame, pl.Series(lag_onto(c, c, sig, 4)), cost_bps=10.0)["cost"].sum()
    assert p4 <= p0 + 1e-12


def test_block_bootstrap_and_skill_and_holm():
    idx = block_bootstrap_idx(95, block=10, n_boot=50, seed=1)
    assert idx.shape == (50, 95) and idx.min() >= 0 and idx.max() < 95
    assert np.all(np.diff(idx[:, :10], axis=1) == 1)  # first block contiguous
    y = np.array([0, 1, 1, 0, 1], dtype=float)
    base = np.full(5, 0.6)
    assert brier_skill(y, y, base) == 1.0
    assert abs(brier_skill(base, y, base)) < 1e-12
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)
    assert half_life((0, 1, 2, 4, 8), [1.0, 0.8, 0.6, 0.4, 0.2]) == 3.0
    assert half_life((0, 1, 2, 4, 8), [1.0, 0.9, 0.9, 0.8, 0.7]) == ">8"
