from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.realized import daily_measures, har_lags


def _bars(days: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * days + 1
    r = rng.standard_t(5, n) * 0.004
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    t0 = datetime(2026, 1, 1, tzinfo=UTC) - timedelta(hours=1)
    return pl.DataFrame({"event_ts": [t0 + timedelta(hours=i) for i in range(n)], "open": o, "high": h, "low": lo, "close": c})


def test_identities_and_point_in_time():
    b = _bars()
    d = daily_measures(b)
    assert d.height == 3 and (d["n_bars"] == 24).all()
    assert np.allclose(d["rv"], d["rs_neg"] + d["rs_pos"])
    assert (d["jv"] >= 0).all() and (d["pk"] > 0).all()
    # day 1's return equals ln(last close of day 1 / last close of the previous day)
    closes = b["close"].to_numpy()
    assert np.isclose(d["ret"][0], np.log(closes[24] / closes[0]))
    assert (d["known_ts"] == d["date"].cast(pl.Datetime("us", "UTC")) + timedelta(days=1)).all()


def test_incomplete_days_are_dropped_and_har_targets_are_future_only():
    d = daily_measures(_bars(5).head(24 * 4 + 1 - 5))
    assert d.height == 3  # the partial 4th day is dropped
    p = har_lags(d.with_columns(pl.lit("X").alias("symbol")))
    assert np.isclose(p["rv_next"][0], p["rv"][1]) and p["rv_next"][-1] is None
