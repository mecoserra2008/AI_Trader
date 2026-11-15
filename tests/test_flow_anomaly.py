from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features import flow_anomaly as fa

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _minutes(n: int, start: datetime = T0) -> list[datetime]:
    return [start + timedelta(minutes=i) for i in range(n)]


def test_robust_z_is_causal_and_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.standard_t(3, 1200)
    w, mv = 200, 100
    z = fa.robust_z(x, w, mv)
    z_more = fa.robust_z(np.concatenate([x, rng.standard_t(3, 300) * 50]), w, mv)
    assert np.allclose(z, z_more[: len(x)], equal_nan=True)  # appending the future leaves the past unchanged
    assert np.isnan(z[: mv]).all()
    t = 700
    past = x[t - w: t]
    med = np.median(past)
    assert np.isclose(z[t], (x[t] - med) / (1.4826 * np.median(np.abs(past - med))))
    # changing the current value changes only the current score's numerator (window excludes t)
    y = x.copy()
    y[t] = 1e6
    zy = fa.robust_z(y, w, mv)
    assert np.allclose(zy[:t], z[:t], equal_nan=True) and zy[t] > 1e3


def test_panel_z_causal_when_rows_are_appended():
    rng = np.random.default_rng(2)
    n = 400

    def frame(k):
        return pl.DataFrame({"symbol": "X", "event_ts": _minutes(k), "f": rng.normal(size=k), "ret": rng.normal(size=k) * 1e-3})

    a = frame(n)
    b = pl.concat([a, frame(n + 100).tail(100).with_columns(pl.Series("event_ts", _minutes(100, T0 + timedelta(minutes=n))))])
    za = fa.add_robust_z(a, ["f"], window=120, min_valid=60)
    zb = fa.add_robust_z(b, ["f"], window=120, min_valid=60).head(n)
    assert np.allclose(za["z_f"].fill_null(np.nan).to_numpy(), zb["z_f"].fill_null(np.nan).to_numpy(), equal_nan=True)
    assert za["z_f"].head(120).null_count() == 120  # warm-up dropped
    assert np.allclose(za["sigma_1m"].fill_null(np.nan).to_numpy(), zb["sigma_1m"].fill_null(np.nan).to_numpy(), equal_nan=True)


def test_robust_z_zero_mad_fallback():
    x = np.zeros(300)
    x[::3] = 1.0  # MAD = 0 (2/3 zeros) -> mean-absolute-deviation fallback
    x[250] = 5.0
    z = fa.robust_z(x, 100, 50)
    assert np.isfinite(z[250]) and z[250] > 4
    flat = np.zeros(200)
    flat[150] = 1.0
    zf = fa.robust_z(flat, 100, 50)
    assert zf[149] == 0.0 and np.isinf(zf[150])


def _liq_grid(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    notional = np.where(rng.random(n) < 0.05, rng.exponential(1000, n), 0.0)
    count = (notional > 0).astype(int)
    return pl.DataFrame({"symbol": "X", "event_ts": _minutes(n), "liq_notional": notional, "liq_n": count,
                         "liq_long": notional, "liq_short": 0.0})


def test_liquidation_cluster_detector():
    g = _liq_grid(600)
    big = g["liq_notional"].to_numpy().copy()
    n = g["liq_n"].to_numpy().copy()
    lng, sht = g["liq_long"].to_numpy().copy(), g["liq_short"].to_numpy().copy()
    for i, k in ((300, 3), (400, 2), (50, 5)):  # flagged / too few prints / inside warm-up
        big[i], n[i] = 1e6, k
    lng[300], sht[300] = 1e5, 9e5
    g = g.with_columns(pl.Series("liq_notional", big), pl.Series("liq_n", n), pl.Series("liq_long", lng), pl.Series("liq_short", sht))
    out = fa.liquidation_clusters(g, window=200, min_valid=100)
    flags = out["liq_cluster"].to_numpy()
    assert flags[300] and not flags[400] and not flags[50]
    assert out["liq_side"][300] == "short"
    assert (out["liq_n"].to_numpy()[flags] >= 3).all()
    # liquidations not yet collected before liq_start: unknown, not zero, so no threshold there
    out2 = fa.liquidation_clusters(g, window=200, min_valid=100, liq_start=T0 + timedelta(minutes=250))
    assert not out2["liq_cluster"][300] and out2["liq_q99"][300] is None
    # causal threshold: appending future rows does not move past thresholds
    out3 = fa.liquidation_clusters(pl.concat([g, g.tail(100).with_columns(pl.Series("event_ts", _minutes(100, T0 + timedelta(minutes=600))))]), window=200, min_valid=100)
    assert np.allclose(out["liq_q99"].fill_null(np.nan).to_numpy(), out3["liq_q99"].head(600).fill_null(np.nan).to_numpy(), equal_nan=True)


def _auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def test_mahalanobis_flags_synthetic_outliers_and_resists_contamination():
    rng = np.random.default_rng(4)
    p = 7
    a = rng.normal(size=(p, p)) * 0.4
    cov = a @ a.T + np.eye(p)
    x = rng.multivariate_normal(np.zeros(p), cov, 6000)
    x[:300] = rng.normal(8, 1, (300, p))  # 5% contamination in the training set
    m = fa.RobustMahalanobis().fit(x)
    assert np.all(np.abs(m.location) < 0.2)  # location not dragged toward the contamination
    assert 0.0 <= m.shrinkage <= 1.0
    test_in = rng.multivariate_normal(np.zeros(p), cov, 2000)
    out = test_in[:50].copy()
    # outliers that break the correlation structure but stay small per coordinate
    v = np.linalg.eigh(cov)[1][:, 0]
    out = rng.normal(size=(50, 1)) * 0 + 6 * v[None, :] * np.sign(rng.normal(size=(50, 1)))
    xt = np.vstack([test_in, out])
    y = np.r_[np.zeros(2000), np.ones(50)]
    d = m.score(xt)
    assert _auc(d, y) > 0.98
    # chi2 consistency: median squared distance of clean data ~ chi2_7 median (6.35)
    assert 4.5 < np.median(d[:2000] ** 2) < 8.5
    # infinite z is clipped, NaN rows score NaN; JSON round trip
    s = m.score(np.array([[np.inf] + [0.0] * (p - 1), [np.nan] * p]))
    assert np.isfinite(s[0]) and np.isnan(s[1])
    m2 = fa.RobustMahalanobis.from_json(m.to_json())
    assert np.allclose(m2.score(xt), d)


def test_ledoit_wolf_shrinks_small_samples_more():
    rng = np.random.default_rng(5)
    cov = np.diag(np.arange(1.0, 8.0)) + 0.5
    small = fa.ledoit_wolf(rng.multivariate_normal(np.zeros(7), cov, 20))[1]
    large, = [fa.ledoit_wolf(rng.multivariate_normal(np.zeros(7), cov, 20000))[1]]
    assert small > large and large < 0.05


def _flow3s(n_min: int, symbol: str = "XUSDT", gap: tuple[int, int] | None = None, seed: int = 0) -> pl.DataFrame:
    """One 3 s candle every 20 s (3 per minute), with a burst of buying at the end."""
    rng = np.random.default_rng(seed)
    ts = [T0 + timedelta(seconds=20 * i) for i in range(3 * n_min)]
    k = len(ts)
    bn, sn = rng.exponential(100, k), rng.exponential(100, k)
    bn[-6:] = 1e6  # last two minutes: a buy-imbalance anomaly
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, k)))
    df = pl.DataFrame({
        "venue": "bybit", "category": "linear", "symbol": symbol, "event_ts": ts,
        "buy_vol": (bn / mid).astype(np.float32), "sell_vol": (sn / mid).astype(np.float32),
        "buy_notional": bn.astype(np.float32), "sell_notional": sn.astype(np.float32),
        "n_buy": rng.integers(1, 5, k).astype(np.int32), "n_sell": rng.integers(1, 5, k).astype(np.int32),
        "max_trade": np.float32(1.0), "mid": mid, "close": mid, "spread_bps": rng.uniform(1, 2, k).astype(np.float32),
        "depth_bid_25": rng.uniform(1e5, 2e5, k).astype(np.float32), "depth_ask_25": rng.uniform(1e5, 2e5, k).astype(np.float32),
        "book_cov": np.float32(1.0), "book_updates": np.int32(10),
        "known_ts": [t + timedelta(seconds=4) for t in ts],
    }).with_columns(pl.col(["event_ts", "known_ts"]).dt.replace_time_zone("UTC").dt.cast_time_unit("ms"))
    if gap:
        lo, hi = T0 + timedelta(minutes=gap[0]), T0 + timedelta(minutes=gap[1])
        df = df.filter((pl.col("event_ts") < lo) | (pl.col("event_ts") >= hi))
    return df


def test_rollup_1m_sums_and_keeps_gaps_null():
    df3 = _flow3s(30, gap=(10, 13))
    m = fa.rollup_1m(df3.select(fa.FLOW_COLUMNS))
    assert m.height == 30  # complete grid
    assert m.filter(pl.col("event_ts") == T0 + timedelta(minutes=11))["buy_notional"][0] is None
    first = df3.filter(pl.col("event_ts") < T0 + timedelta(minutes=1))
    assert np.isclose(m["buy_notional"][0], first["buy_notional"].cast(pl.Float64).sum(), rtol=1e-6)
    assert m["trades"][0] == (first["n_buy"] + first["n_sell"]).sum()
    f = fa.add_features(m)
    assert f.filter(pl.col("event_ts") == T0 + timedelta(minutes=13))["ret"][0] is None  # no return across a gap


def test_latest_anomalies_point_in_time(lake):
    n_min = fa.WINDOW + 120
    lake.write("flow_3s", _flow3s(n_min))
    as_of = T0 + timedelta(minutes=n_min) + timedelta(seconds=30)
    res = fa.latest_anomalies(["XUSDT"], as_of, lookback_h=1, lake=lake, use_model=False)
    r = res["XUSDT"]
    last_minute = (T0 + timedelta(minutes=n_min - 1)).isoformat()
    assert r["latest"]["time"] == last_minute and abs(r["latest"]["z"]) > 4
    imb = r["latest_by_type"]["imbalance"]
    assert imb["side"] == "buy" and imb["z"] > 4 and imb["time"] == last_minute
    assert r["counts"]["imbalance"] >= 1
    # as of two minutes earlier the burst is not yet known
    early = fa.latest_anomalies(["XUSDT"], as_of - timedelta(minutes=2), lookback_h=1, lake=lake, use_model=False)["XUSDT"]
    assert early["counts"].get("imbalance", 0) == 0


def test_relative_scale_floor_tames_one_tick_spreads():
    rng = np.random.default_rng(6)
    tick = 0.0132 * (1 + rng.normal(0, 1e-6, 600))  # one tick in bps: varies only with the price level
    tick[300:] *= 0.999  # the price rises 0.1%: same one-tick book, a slightly smaller spread in bps
    tick[500] = 0.0264  # two ticks: a genuine widening
    z_raw = fa.robust_z(tick, 200, 100, rel_floor=0.0)
    z = fa.robust_z(tick, 200, 100)
    assert np.mean(np.abs(z_raw[200:]) > 4) > 0.2  # degenerate without the floor
    assert np.mean(np.abs(np.delete(z, 500)[200:]) > 4) == 0 and z[500] > 4


def test_rollup_drops_late_finalised_candles():
    df3 = _flow3s(5).select(fa.FLOW_COLUMNS)
    late = df3.head(1).with_columns(pl.col("known_ts") + timedelta(hours=3))
    df = pl.concat([late, df3.slice(1)])
    m = fa.rollup_1m(df)
    assert m["n3s"][0] == 2 and m["n3s"][1] == 3
    assert fa.rollup_1m(df, max_lag_s=None)["n3s"][0] == 3
