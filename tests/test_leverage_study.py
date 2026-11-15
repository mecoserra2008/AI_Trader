from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import leverage_study as ls


def _daily(n=60, start=date(2024, 1, 1), seed=0, drop=()):
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(n)]
    rv = rng.uniform(1e-4, 4e-4, n)
    share = rng.uniform(0.2, 0.8, n)
    ret = rng.normal(0, 0.02, n)
    d = pl.DataFrame({"date": dates, "rv": rv, "rs_neg": rv * share, "rs_pos": rv * (1 - share), "ret": ret,
                      "close": 100 * np.exp(np.cumsum(ret))})
    return d.filter(~pl.col("date").is_in(list(drop)))


def test_calendar_reindex_and_features_respect_gaps():
    gap = date(2024, 1, 20)
    d = ls.add_features(ls.calendar_reindex(_daily(drop=[gap])))
    assert d.height == 60
    row = lambda dt: d.filter(pl.col("date") == dt).row(0, named=True)  # noqa: E731
    # the day before the gap has no next-day target; windows covering the gap are null
    assert row(gap - timedelta(days=1))["rv_next"] is None
    for k in range(5):
        assert row(gap + timedelta(days=k))["rv_w"] is None
    assert row(gap + timedelta(days=5))["rv_w"] is not None
    assert row(gap + timedelta(days=21))["rv_m"] is None
    assert row(gap + timedelta(days=22))["rv_m"] is not None
    r = row(date(2024, 1, 10))
    assert r["origin"] == datetime(2024, 1, 11, tzinfo=UTC)
    assert r["rv_w"] == pytest.approx(d.filter((pl.col("date") >= date(2024, 1, 6)) & (pl.col("date") <= date(2024, 1, 10)))["rv"].mean())


def test_semivariance_floor():
    d = _daily(30).with_columns(pl.lit(0.0).alias("rs_neg"))
    f = ls.add_features(ls.calendar_reindex(d))
    assert np.allclose(f["rs_neg_f"].to_numpy(), 0.01 * f["rv"].to_numpy())


def test_drawdown_flag():
    n = 40
    ret = np.r_[np.full(30, 0.001), np.full(10, -0.05)]
    d = pl.DataFrame({"date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)], "rv": np.full(n, 1e-4),
                      "rs_neg": np.full(n, 5e-5), "rs_pos": np.full(n, 5e-5), "ret": ret, "close": 100 * np.exp(np.cumsum(ret))})
    f = ls.add_features(d)
    dd = f["dd"].to_numpy()
    assert np.isnan(dd[:29]).all() or all(x is None for x in f["dd"].to_list()[:29])
    # manual check of the last day
    c = f["close"].to_numpy()
    lhs = np.log(c[-1] / c[-30:].max())
    rhs = -ls.DD_K * np.std(ret[-30:], ddof=1) * np.sqrt(30)
    assert dd[-1] == float(lhs < rhs)
    assert dd[-1] == 1.0 and dd[29] == 0.0


def test_funding_z_point_in_time():
    t0 = np.datetime64("2024-01-01T00:00:00", "us")
    ft = t0 + np.arange(400) * np.timedelta64(8, "h")
    fr = np.sin(np.arange(400) / 7.0) * 1e-4
    o = np.array([ft[300] + np.timedelta64(1, "h")])
    z = ls.funding_z(ft, fr, o, window=timedelta(days=90), min_prints=90)
    lo = np.searchsorted(ft, o[0] - np.timedelta64(90, "D"), side="right")
    w = fr[lo:301]
    assert z[0] == pytest.approx((fr[300] - w.mean()) / w.std(ddof=1))
    # a print known after the origin must not change the value
    fr2 = fr.copy()
    fr2[301:] = 1.0
    assert ls.funding_z(ft, fr2, o)[0] == pytest.approx(z[0])
    # too few prints -> NaN
    assert np.isnan(ls.funding_z(ft, fr, np.array([ft[10]]))[0])


def test_fe_ols_recovers_beta_and_cluster_cov_reduces_to_hc1():
    rng = np.random.default_rng(1)
    n_sym, n_t = 5, 400
    rows = []
    for i in range(n_sym):
        x1, x2 = rng.normal(size=n_t), rng.normal(size=n_t)
        y = 3.0 * i + 0.5 * x1 - 0.2 * x2 + rng.normal(scale=0.5, size=n_t)
        rows.append(pl.DataFrame({"symbol": f"S{i}", "origin": np.arange(n_t), "y": y, "x1": x1, "x2": x2}))
    df = pl.concat(rows)
    r = ls.fe_ols(df, "y", ["x1", "x2"])
    assert r["beta"] == pytest.approx([0.5, -0.2], abs=0.05)
    d = ls.lincomb(r, {"x1": 1, "x2": -1})
    assert d["est"] == pytest.approx(0.7, abs=0.06) and d["t"] > 10
    # singleton clusters, no absorbed FE -> HC1
    X = rng.normal(size=(200, 2))
    u = rng.normal(size=200)
    V = ls.cluster_cov(X, u, np.arange(200))
    bread = np.linalg.inv(X.T @ X)
    hc1 = 200 / 198 * bread @ (X.T * u**2) @ X @ bread
    assert np.allclose(V, hc1)


def test_demean():
    a = np.array([[1.0, 2], [3, 4], [10, 20]])
    g = np.array([0, 0, 1])
    assert np.allclose(ls.demean(a, g), [[-1, -1], [1, 1], [0, 0]])


def test_holm():
    adj = ls.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert np.isnan(adj["d"])


def _panel(n_sym=3, n=400, seed=2):
    out = []
    for i in range(n_sym):
        d = ls.add_features(ls.calendar_reindex(_daily(n, seed=seed + i))).with_columns(pl.lit(f"S{i}").alias("symbol"))
        out.append(d)
    return pl.concat(out)


def test_oos_panel_is_causal():
    p = _panel()
    base = ls.oos_panel(p, "shar", "log")
    assert base["f"].is_finite().sum() > 100
    cut = datetime(2024, 9, 1, tzinfo=UTC)
    # perturb every target known after `cut`: forecasts with origin before cut must be unchanged
    p2 = p.with_columns(pl.when(pl.col("origin") >= cut - timedelta(days=1)).then(pl.col("rv_next") * 50).otherwise(pl.col("rv_next")).alias("rv_next"))
    alt = ls.oos_panel(p2, "shar", "log")
    j = base.join(alt, on=["symbol", "origin"], suffix="_2").filter((pl.col("origin") < cut) & pl.col("f").is_finite())
    assert j.height > 50
    assert np.allclose(j["f"].to_numpy(), j["f_2"].to_numpy())


def test_paired_qlike_and_date_avg():
    p = _panel()
    q = ls.paired_qlike(ls.oos_panel(p, "shar", "log"), ls.oos_panel(p, "har", "log"))
    assert q.height > 0 and np.allclose(q["dq"].to_numpy(), q["qa"].to_numpy() - q["qb"].to_numpy())
    t = ls.date_avg_test(q)
    assert t["n_dates"] == q["origin"].n_unique()


def test_blocks_4h():
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(48)]
    close = 100 * np.exp(np.cumsum(np.r_[0.0, np.full(47, 0.01)] * np.where(np.arange(48) % 2, 1, -1)))
    bars = pl.DataFrame({"event_ts": ts, "open": close, "high": close * 1.001, "low": close * 0.999, "close": close})
    b = ls.blocks_4h(bars.filter(pl.col("event_ts") != ts[10]))
    # block 0 has bars 1..3 only (bar 0 has no previous close): incomplete, so the calendar starts at block 1
    assert b["blk"][0] == ts[4]
    blk1 = b.filter(pl.col("blk") == ts[4]).row(0, named=True)
    assert blk1["rv"] == pytest.approx(4 * 0.01**2) and blk1["rs_neg"] == pytest.approx(2 * 0.01**2)
    assert b.filter(pl.col("blk") == ts[8])["rv"][0] is None  # bar 10 missing


def test_garch_asym_detects_positive_gamma():
    rng = np.random.default_rng(3)
    n, om, a, g, b = 3000, 0.05, 0.03, 0.15, 0.88
    r = np.zeros(n)
    s2 = om / (1 - a - g / 2 - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_t(8) / np.sqrt(8 / 6)
        s2 = om + (a + g * (r[t] < 0)) * r[t] ** 2 + b * s2
    out = ls.garch_asym(r / 100)
    assert out["gjr"]["gamma"] > 0.05 and out["gjr"]["t"] > 2
    assert out["egarch"]["gamma"] < 0


def test_shar_4h_runs_on_synthetic_panel():
    rng = np.random.default_rng(5)
    frames = []
    for i in range(2):
        ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(24 * 60)]
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ts))))
        bars = pl.DataFrame({"event_ts": ts, "open": close, "high": close, "low": close, "close": close})
        frames.append(ls.blocks_4h(bars).with_columns(pl.lit(f"S{i}").alias("symbol")))
    out = ls.shar_4h(pl.concat(frames), "pooled")
    assert out["n"] > 400 and np.isfinite(out["diff"]["t"])
