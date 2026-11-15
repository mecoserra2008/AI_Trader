from datetime import timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")

from ai_trader.ml.hourly import HOURLY_FEATURES, complete_hourly, qlike_diff, symbol_hourly, train_subsample  # noqa: E402
from ai_trader.research.frame import add_robust_vol, enrich_bars  # noqa: E402
from tests.conftest import T0, make_bars  # noqa: E402


def _frame(n_hours: int, drop=()):
    b = make_bars(n_hours, freq_h=1, seed=3)
    if drop:
        b = b.with_row_index("i").filter(~pl.col("i").is_in(list(drop))).drop("i")
    return add_robust_vol(enrich_bars(b))


def test_hourly_features_are_causal():
    n = 24 * 90
    full = _frame(n, drop=range(1000, 1003))
    t_cut = T0 + timedelta(hours=1700)
    part = add_robust_vol(enrich_bars(full.filter(pl.col("event_ts") < t_cut).select(
        "event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")))
    a = symbol_hourly(part)
    b = symbol_hourly(full).join(a.select("event_ts"), on="event_ts", how="semi")
    assert a.height == b.height and a.height > 1500
    for c in (*HOURLY_FEATURES, "s2", "fac"):
        x, y = a[c].cast(pl.Float64).fill_null(np.nan).to_numpy(), b[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=1e-9, atol=1e-12), c
    assert complete_hourly(symbol_hourly(full)).height > 200


def test_gap_returns_are_dropped_and_lags_align():
    f = _frame(24 * 70, drop=(500,))
    h = symbol_hourly(f)
    ts = h["event_ts"].to_list()
    i = ts.index(T0 + timedelta(hours=501))
    assert h["r2"][i] is None  # return across the missing bar is not used
    j = i + 30
    s2 = h["s2"][j]
    assert np.isclose(np.exp(h["lu_1"][j]), max(h["r2"][j - 1] / s2, 1e-4))


def test_qlike_diff_and_subsample():
    r2 = np.array([0.0, 1e-4, 4e-4])
    a, b = np.array([1e-4, 2e-4, 3e-4]), np.array([2e-4, 2e-4, 2e-4])
    q = lambda f: r2 / f - np.log(np.where(r2 > 0, r2, 1) / f) - 1  # noqa: E731
    d = qlike_diff(r2, a, b)
    assert np.allclose(d[1:], (q(a) - q(b))[1:])
    assert np.isfinite(d[0])
    m1, m2 = train_subsample(3000), train_subsample(3000)
    assert np.array_equal(m1, m2) and 0.28 < m1.mean() < 0.39
