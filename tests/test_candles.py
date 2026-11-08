from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.candles import decluster, vector_candles, vector_events
from ai_trader.research.frame import enrich_bars


def _bars(n: int = 600, seed: int = 3, sweep_at: int | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    r = 0.005 * rng.standard_normal(n)
    if sweep_at is not None:
        r[sweep_at] = 0.06  # ~12 sigma up bar
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    wick = 0.001 * np.ones(n)
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    turnover = rng.lognormal(0, 0.2, n) * 1e6
    if sweep_at is not None:
        turnover[sweep_at] *= 5
    ev = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
    bars = pl.DataFrame({
        "event_ts": ev, "close_ts": [t + timedelta(hours=1) for t in ev], "known_ts": [t + timedelta(hours=1, seconds=2) for t in ev],
        "open": open_, "high": high, "low": low, "close": close, "volume": turnover / close, "turnover": turnover,
    }).with_columns(pl.col(["event_ts", "close_ts", "known_ts"]).dt.cast_time_unit("ms"))
    return enrich_bars(bars)


def test_sweep_bar_is_flagged_with_zone():
    f = vector_candles(_bars(sweep_at=400))
    row = f.row(400, named=True)
    assert row["is_vector"] and row["direction"] == 1
    assert row["range_sig"] >= 3 and row["vol_rel"] >= 2 and row["body"] >= 0.6
    assert row["zone_lo"] == row["open"] and row["zone_hi"] == row["close"]
    assert abs(row["zone_mid"] - (row["open"] + row["close"]) / 2) < 1e-12
    # warm-up: no vol_rel (and no vectors) before 168 prior bars exist
    assert f["vol_rel"][:168].null_count() == 168 and not f["is_vector"][:168].any()
    # the ordinary bars are (almost) never vectors
    assert f["is_vector"].sum() <= 3


def test_vol_rel_excludes_current_bar():
    f = vector_candles(_bars(sweep_at=400))
    t = f["turnover"].to_numpy()
    assert abs(f["vol_rel"][400] - t[400] / np.median(t[400 - 168:400])) < 1e-9


def test_thin_uses_prior_distribution():
    f = vector_candles(_bars())
    ratio = (f["vol_rel"] / f["range_sig"]).to_numpy()
    i = 500
    q = np.quantile(ratio[i - 168:i], 0.2)
    assert f["thin"][i] == (ratio[i] <= q)
    assert 0.1 < f["thin"].drop_nulls().mean() < 0.3


def test_causal_truncation_invariance():
    full = vector_candles(_bars(sweep_at=400))
    cols = ["range_sig", "vol_rel", "body", "is_vector", "direction", "thin", "zone_mid"]
    for cut in (300, 401, 450):
        part = vector_candles(_bars(sweep_at=400)[:cut])
        a, b = full[:cut].select(cols), part.select(cols)
        assert a.equals(b), cut


def test_decluster_rule():
    flags = np.zeros(100, dtype=bool)
    flags[[10, 20, 34, 35, 60, 84, 85]] = True
    keep = decluster(flags, 24)
    # 20 is within 24 of kept 10; 34 is 24 after 10 -> dropped (needs > 24); 35 kept; 60 kept (25 after 35);
    # 84 is 24 after 60 -> dropped; 85 kept.
    assert np.flatnonzero(keep).tolist() == [10, 35, 60, 85]
    # causal: truncating the flags doesn't change earlier decisions
    assert np.array_equal(decluster(flags[:61], 24), keep[:61])


def test_vector_events_indices():
    f = vector_candles(_bars(sweep_at=400))
    ev = vector_events(f)
    assert 400 in ev["i"].to_list()
