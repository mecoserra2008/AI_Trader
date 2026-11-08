from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.regime_features import ALL_COLUMNS, regime_features, variance_ratio
from ai_trader.research.frame import add_robust_vol, enrich_bars

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _frame(close: np.ndarray, start: datetime = T0) -> pl.DataFrame:
    n = len(close)
    ev = [start + timedelta(hours=4 * i) for i in range(n)]
    open_ = np.r_[close[0], close[:-1]]
    b = pl.DataFrame({
        "event_ts": ev, "close_ts": [t + timedelta(hours=4) for t in ev], "known_ts": [t + timedelta(hours=4, seconds=2) for t in ev],
        "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close, "volume": 1.0, "turnover": close,
    }).with_columns(pl.col(["event_ts", "close_ts", "known_ts"]).dt.cast_time_unit("ms"))
    return add_robust_vol(enrich_bars(b))


def _rw(n: int, seed: int = 0, phi: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_t(5, n) * 0.01
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    return 100 * np.exp(np.cumsum(r))


def test_truncation_invariance():
    close = _rw(6 * 420, seed=1)
    full = regime_features(_frame(close))
    assert full["vol_pct_1y"].drop_nans().len() > 100
    for k in (6 * 200 + 3, 6 * 300, 6 * 400 + 5):
        part = regime_features(_frame(close[:k]))
        m = full.filter(pl.col("date") <= part["date"].max())
        assert m.height == part.height
        for c in ALL_COLUMNS:
            assert np.allclose(m[c].to_numpy(), part[c].to_numpy(), equal_nan=True, atol=1e-12), c


def test_efficiency_ratio_is_one_on_a_straight_line():
    close = 100 * np.exp(0.002 * np.arange(6 * 40))
    f = regime_features(_frame(close)).drop_nans(["er_30d"])
    assert f.height > 5
    for c in ("er_7d", "er_14d", "er_30d", "dvr_7d", "dvr_30d"):
        assert np.allclose(f[c].to_numpy(), 1.0)
    down = regime_features(_frame(close[::-1].copy())).drop_nans(["er_30d"])
    assert np.allclose(down["er_30d"].to_numpy(), 1.0) and np.allclose(down["dvr_30d"].to_numpy(), -1.0)


def test_dvr_sign_on_asymmetric_returns():
    # four +1% bars then one -4% bar: zero drift, but all the squared variation is on the downside
    pat = np.array([0.01, 0.01, 0.01, 0.01, -0.04])
    r = np.tile(pat, 6 * 40 // 5 + 1)[: 6 * 40]
    f = regime_features(_frame(100 * np.exp(np.cumsum(r)))).drop_nans(["dvr_30d"])
    assert (f["dvr_7d"] < -0.4).all() and (f["dvr_30d"] < -0.5).all()
    assert (f["er_30d"] < 0.05).all()
    g = regime_features(_frame(100 * np.exp(np.cumsum(-r)))).drop_nans(["dvr_30d"])
    assert (g["dvr_7d"] > 0.4).all() and (g["dvr_30d"] > 0.5).all()


def test_autocorrelation_and_variance_ratio_respond_to_persistence():
    rng = np.random.default_rng(3)
    assert abs(np.mean([variance_ratio(rng.standard_normal(180), 6) for _ in range(300)]) - 1.0) < 0.05
    iid = regime_features(_frame(_rw(6 * 200, seed=4)))
    ar = regime_features(_frame(_rw(6 * 200, seed=4, phi=0.4)))
    assert ar["ac1_4h_30d"].drop_nans().mean() > iid["ac1_4h_30d"].drop_nans().mean() + 0.25
    assert ar["vr_30d"].drop_nans().mean() > 1.5 > iid["vr_30d"].drop_nans().mean()


def test_missing_bar_never_spans_a_gap():
    close = 100 * np.exp(0.002 * np.arange(6 * 40))
    fr = _frame(close)
    fr = fr.filter(pl.col("event_ts") != T0 + timedelta(days=35, hours=8))  # drop one bar mid-window
    f = regime_features(fr)
    # 1 missing of 180 is within MIN_COVER: still a straight line (no double-length return)
    last = f.filter(pl.col("date") == (T0 + timedelta(days=36)).date())
    assert np.isclose(last["er_30d"][0], 1.0)
    # the day whose last bar is missing has no row
    fr2 = _frame(close).filter(pl.col("event_ts") != T0 + timedelta(days=35, hours=20))
    assert (T0 + timedelta(days=35)).date() not in regime_features(fr2)["date"].to_list()
