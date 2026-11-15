"""sim/book.py (ob200 reconstruction, spread / depth) and the Hayashi-Yoshida / VAR tools of book_spillover_study."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

from ai_trader.research.book_spillover_study import LAGS_MS, connectedness, gfevd, granger, hy_ccf, lead_stats, var_fit
from ai_trader.sim.book import Book, book_metrics, build_from_messages, iter_messages, validate_trades


def _msg(kind, ts, b, a, u):
    return {"topic": "orderbook.200.TEST", "type": kind, "ts": ts, "cts": ts - 5,
            "data": {"s": "TEST", "b": [[str(p), str(s)] for p, s in b], "a": [[str(p), str(s)] for p, s in a], "u": u, "seq": u}}


def test_book_snapshot_delta_removal_and_reset():
    bk = Book()
    assert not bk.apply("delta", [["100", "1"]], [])  # before any snapshot: ignored
    assert bk.bids == {}
    bk.apply("snapshot", [["100", "2"], ["99", "3"]], [["101", "1"], ["102", "4"]])
    bk.apply("delta", [["100", "0"], ["99.5", "5"]], [["101", "2.5"]])  # remove 100, add 99.5, replace 101
    assert bk.bids == {99.5: 5.0, 99.0: 3.0}
    assert bk.asks == {101.0: 2.5, 102.0: 4.0}
    m = bk.metrics()
    assert m[0] == 99.5 and m[1] == 101.0 and m[2] == 5.0 and m[3] == 2.5
    bk.apply("snapshot", [["50", "1"]], [["51", "1"]])  # snapshot resets everything
    assert bk.bids == {50.0: 1.0} and bk.asks == {51.0: 1.0}
    assert bk.n_snapshots == 2


def test_book_metrics_spread_depth_imbalance():
    bp = np.array([99.99, 99.95, 99.80, 99.0])
    bs = np.array([1.0, 2.0, 3.0, 100.0])
    ap = np.array([100.01, 100.08, 100.30])
    asz = np.array([2.0, 1.0, 5.0])
    m = book_metrics(bp, bs, ap, asz, bands=(5, 10, 25), imb=(5, 10))
    b1, a1, b1s, a1s, mid, sp = m[:6]
    assert (b1, a1, b1s, a1s) == (99.99, 100.01, 1.0, 2.0)
    assert mid == pytest.approx(100.0)
    assert sp == pytest.approx(0.02 / 100.0 * 1e4)  # 2 bp
    db5, da5, db10, da10, db25, da25 = m[6:12]
    assert db5 == pytest.approx(99.99 * 1 + 99.95 * 2)  # 99.95 is exactly 5 bp below 100: inclusive
    assert da5 == pytest.approx(100.01 * 2)
    assert db10 == pytest.approx(db5)
    assert da10 == pytest.approx(100.01 * 2 + 100.08 * 1)
    assert db25 == pytest.approx(db5 + 99.80 * 3)
    assert da25 == pytest.approx(da10)  # 100.30 is 30 bp away
    assert m[12] == pytest.approx((db5 - da5) / (db5 + da5))
    assert m[13] == pytest.approx((db10 - da10) / (db10 + da10))
    assert m[14] == pytest.approx(100.0) and m[15] == pytest.approx(30.0)  # book span (deepest level) in bp
    assert all(math.isnan(x) for x in book_metrics(np.array([]), np.array([]), ap, asz))


def test_build_series_grid_and_file(tmp_path):
    msgs = [
        _msg("delta", 950, [(1, 1)], [], 1),  # before snapshot
        _msg("snapshot", 1010, [(100, 1), (99, 1)], [(101, 1)], 10),
        _msg("delta", 1050, [], [(101, 3)], 11),  # same 100 ms bin: last state wins, 2 updates
        _msg("delta", 1320, [(100, 0)], [], 12),  # bins 1100-1300 carry state forward; 100 removed
        _msg("snapshot", 1450, [(200, 2)], [(202, 2)], 1),  # reset
    ]
    p = tmp_path / "x.data"
    p.write_text("\n".join(json.dumps(m) for m in msgs) + "\n")
    df, qa = build_from_messages(iter_messages(p))
    t = df["ts"].dt.epoch("ms").to_list()
    assert t == [1100, 1200, 1300, 1400, 1500]
    assert df["n_updates"].to_list() == [2, 0, 0, 1, 1]
    assert df["ask1_size"].to_list()[:3] == [3.0, 3.0, 3.0]
    assert df["bid1"].to_list() == [100.0, 100.0, 100.0, 99.0, 200.0]
    assert df["rel_spread_bp"][-1] == pytest.approx(2 / 201 * 1e4)
    assert qa["ignored_before_snapshot"] == 1 and qa["snapshots"] == 2
    tr = pl.DataFrame({"event_ts": [1080, 1350, 1360], "side": ["Buy", "Sell", "Buy"], "price": [101.0, 99.0, 105.0]}).with_columns(
        pl.col("event_ts").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"))
    v = validate_trades(df, tr)
    assert v["n_trades"] == 3 and v["strict_violation_rate"] == pytest.approx(1 / 3)


def _lagged_pair(lag_ms: int, n: int = 40_000, seed: int = 0):
    """Leader: random walk observed at Poisson times; follower = leader's price lag_ms later + own noise, observed at
    its own (asynchronous) Poisson times."""
    rng = np.random.default_rng(seed)
    T = 4_000_000  # ms
    grid = np.arange(0, T + 1, 10)
    x = np.cumsum(rng.normal(0, 1e-4, len(grid)))
    tx = np.unique(rng.integers(0, T, n))
    ty = np.unique(rng.integers(lag_ms, T, n))
    px = x[tx // 10]
    py = x[(ty - lag_ms) // 10] + np.cumsum(rng.normal(0, 3e-5, len(ty)))
    return tx.astype(np.int64), px, ty.astype(np.int64), py


@pytest.mark.parametrize("lag_ms", [500, 2000, 0])
def test_hayashi_yoshida_recovers_lag(lag_ms):
    tx, px, ty, py = _lagged_pair(lag_ms)
    rho = hy_ccf(tx, px, ty, py, LAGS_MS)
    st = lead_stats(rho)
    assert st["peak_lag_s"] == pytest.approx(lag_ms / 1000)
    assert st["peak_rho"] > 0.5
    if lag_ms:
        assert st["llr"] > 1 and st["positive_peak"]
    # swapping the series mirrors the lag (base-series switch inside hy_blocks)
    rho2 = hy_ccf(ty, py, tx, px, LAGS_MS)
    assert lead_stats(rho2)["peak_lag_s"] == pytest.approx(-lag_ms / 1000)
    np.testing.assert_allclose(rho2[::-1], rho, atol=1e-12)


def test_hy_equals_realised_correlation_on_synchronous_grid():
    rng = np.random.default_rng(3)
    t = np.arange(0, 100_000, 1000, dtype=np.int64)
    a = np.cumsum(rng.normal(size=len(t)))
    b = 0.6 * a + np.cumsum(rng.normal(size=len(t)))
    rho = hy_ccf(t, a, t, b, np.array([0], dtype=np.int64))
    ra, rb = np.diff(a), np.diff(b)
    assert rho[0] == pytest.approx(ra @ rb / math.sqrt((ra @ ra) * (rb @ rb)))


def test_granger_detects_direction():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(size=n)
    y = np.r_[0.0, 0.4 * x[:-1]] + rng.normal(size=n)
    g = granger(y, x, pmax=5)
    assert g["x_to_y"]["p_robust"] < 1e-6 and g["y_to_x"]["p_robust"] > 1e-3


def test_gfevd_connectedness_direction():
    rng = np.random.default_rng(2)
    n = 5000
    Y = np.zeros((n, 3))
    for t in range(1, n):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + rng.normal()
        Y[t, 1] = 0.3 * Y[t - 1, 1] + 0.6 * Y[t - 1, 0] + rng.normal()
        Y[t, 2] = 0.3 * Y[t - 1, 2] + rng.normal()
    A, S = var_fit(Y, 1)
    th = gfevd(A, S, 10)
    np.testing.assert_allclose(th.sum(1), 1.0)
    c = connectedness(th)
    assert c["net"][0] > 5 and c["net"][1] < -5 and abs(c["net"][2]) < 2
    assert c["net"].sum() == pytest.approx(0.0, abs=1e-9)
