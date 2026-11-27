"""Tests for ai_trader.research.vol_spec (WS04): no look-ahead in HAR components, clock factor, losses and tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.research import vol_spec as vs


def test_trailing_mean_uses_only_past_bars():
    x = np.arange(10, dtype=float)
    m = vs.trailing_mean(x, 3, min_frac=1.0)
    assert np.isnan(m[:3]).all()
    assert m[3] == 1.0 and m[9] == 7.0          # mean of x[i-3..i-1]
    x2 = x.copy()
    x2[5] = 1e9                                  # a future shock must not move earlier components
    m2 = vs.trailing_mean(x2, 3, min_frac=1.0)
    assert np.array_equal(m2[:6], m[:6], equal_nan=True)


def test_clock_factor_mean_one_and_recovers_shape():
    rng = np.random.default_rng(0)
    n = 168 * 60
    cell = np.arange(n) % 168
    shape = 1 + 0.8 * np.sin(2 * np.pi * np.arange(168) / 168)
    y = shape[cell] * rng.chisquare(5, n) / 5
    f = vs.clock_factor(y, np.ones(n), cell, 168, np.ones(n, bool), smooth=True)
    assert abs(f.mean() - 1) < 1e-12
    assert np.corrcoef(f, shape / shape.mean())[0, 1] > 0.95
    # training mask: bars outside it are ignored
    tr = np.zeros(n, bool)
    tr[: n // 2] = True
    y2 = y.copy()
    y2[n // 2:] *= 100
    assert np.allclose(vs.clock_factor(y2, np.ones(n), cell, 168, tr, True), vs.clock_factor(y, np.ones(n), cell, 168, tr, True))


def test_qlike_minimised_at_truth_and_dm_sign():
    rng = np.random.default_rng(1)
    s2 = np.exp(rng.normal(0, 0.5, 20000))
    rv = s2 * rng.chisquare(1, 20000)
    good, bad = vs.qlike(rv, s2), vs.qlike(rv, 1.5 * s2)
    d = vs.dm_test(good - bad)
    assert d["mean"] < 0 and d["t_hln"] < -5 and d["p"] < 1e-6


def test_kupiec_and_christoffersen_calibrated_hits():
    rng = np.random.default_rng(2)
    h = rng.random(5000) < 0.01
    assert vs.kupiec(h, 0.01)["p"] > 0.01
    assert vs.christoffersen(h, 0.01)["p_ind"] > 0.01
    assert vs.kupiec(rng.random(5000) < 0.03, 0.01)["p"] < 1e-6


def test_t_fit_recovers_nu_and_gpd_quantile_monotone():
    from scipy import stats

    rng = np.random.default_rng(3)
    nu = 4.0
    z = stats.t.rvs(nu, size=20000, random_state=rng) / np.sqrt(nu / (nu - 2))
    th = vs.fit_dist("t", z)
    p = vs._params("t", th)
    assert 3.3 < p["nu"] < 4.8 and abs(p["scale"] - 1) < 0.05
    t = vs.gpd_tail(z, "lo")
    assert vs.gpd_quantile(t, 0.001) > vs.gpd_quantile(t, 0.01) > t["u"]


def test_release_flags_respect_schedule_knowledge(monkeypatch, tmp_path):
    ts = pl.Series("event_ts", [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(5)]).cast(pl.Datetime("us", "UTC"))
    ev = pl.DataFrame({"kind": ["cpi", "cpi"], "name": ["a", "b"],
                       "scheduled_ts": [datetime(2026, 1, 1, 1, 30, tzinfo=UTC), datetime(2026, 1, 1, 3, 30, tzinfo=UTC)],
                       "schedule_known_ts": [datetime(2025, 12, 1, tzinfo=UTC), datetime(2026, 1, 1, 3, 45, tzinfo=UTC)],
                       "source": ["x", "x"]})
    d = tmp_path / "data" / "research" / "news" / "events"
    d.mkdir(parents=True)
    ev.write_parquet(d / "events.parquet")

    class C:
        root = tmp_path

    monkeypatch.setattr(vs, "load", lambda: C)
    f = vs.release_flags(ts, 1)
    assert f.tolist() == [False, True, False, False, False]   # the second schedule became known after the bar opened


def test_mz_unbiased_forecast():
    rng = np.random.default_rng(4)
    s2 = np.exp(rng.normal(0, 0.7, 5000))
    rv = s2 * np.exp(rng.normal(-0.02, 0.2, 5000))
    m = vs.mz_log(rv, s2)
    assert abs(m["b"] - 1) < 0.05
