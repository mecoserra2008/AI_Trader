from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research.frame import enrich_bars
from ai_trader.research.regimes import label
from ai_trader.strategies import REGISTRY, assert_causal

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SHOCK = 3000  # row of the engineered shock (regime labeller needs >= 2160 rows of warm-up)


def _pre_rets(vol_a: float, vol_b: float, seed: int = 0) -> np.ndarray:
    """SHOCK bars: vol_a, then a slow geometric ramp to vol_b. Noise clipped at 3 sd so no accidental |z| >= 4."""
    rng = np.random.default_rng(seed)
    vol = np.full(SHOCK, vol_a)
    ramp = np.arange(2300, 2600)
    vol[ramp] = vol_a * (vol_b / vol_a) ** ((ramp - 2300) / 300)
    vol[2600:] = vol_b
    return np.clip(rng.standard_normal(SHOCK), -3, 3) * vol


def _frame(pre: np.ndarray, shock_ret: float | None, post: list[float]) -> pl.DataFrame:
    rets = np.r_[pre, [] if shock_ret is None else [shock_ret], post]
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[100.0, close[:-1]]
    ev = [T0 + timedelta(hours=i) for i in range(len(close))]
    bars = pl.DataFrame({
        "event_ts": ev,
        "close_ts": [t + timedelta(hours=1) for t in ev],
        "known_ts": [t + timedelta(hours=1, seconds=2) for t in ev],
        "open": open_, "high": np.maximum(open_, close) * 1.0002, "low": np.minimum(open_, close) * 0.9998,
        "close": close, "volume": 1.0, "turnover": close,
    })
    return enrich_bars(bars)


def _strat(**kw):
    return REGISTRY["shock_regime"](**kw)


@pytest.fixture(scope="module")
def low_pre() -> np.ndarray:
    return _pre_rets(0.01, 0.003)


@pytest.fixture(scope="module")
def high_pre() -> np.ndarray:
    return _pre_rets(0.005, 0.02, seed=1)


def test_preconditions(low_pre, high_pre):
    lo = _frame(low_pre, 6 * 0.003, [0.001] * 5)
    hi = _frame(high_pre, 5 * 0.02, [0.001] * 5)
    assert label(lo)["vol_state"][SHOCK] == "low"
    assert label(hi)["vol_state"][SHOCK] == "high"
    for f in (lo, hi):  # only the engineered bar is a shock
        assert np.flatnonzero(np.abs(np.nan_to_num(f["z"].to_numpy())) >= 4).tolist() == [SHOCK]


def test_low_vol_shock_continues_and_trails(low_pre):
    post = [0.002] * 5 + [-0.004] * 10  # extend, then roll over
    f = _frame(low_pre, 6 * 0.003, post)
    s = _strat()(f).to_numpy()
    assert s[:SHOCK].max() == 0 and s[:SHOCK].min() == 0
    assert s[SHOCK] > 0  # continuation: long an up-shock
    assert s[SHOCK] == pytest.approx(min(1.0, f["z"][SHOCK] / 8.0))
    c, sig = f["close"].to_numpy(), f["sigma_next"][SHOCK]
    best = np.maximum.accumulate(c[SHOCK:])
    hit = SHOCK + int(np.argmax(c[SHOCK:] < best - 2.0 * sig * c[SHOCK]))
    assert hit > SHOCK + 5 and c[hit] > f["open"][SHOCK]  # the trail fires before the invalidation level
    assert (s[SHOCK:hit] > 0).all() and (s[hit:] == 0).all()


def test_continuation_invalidation_goes_flat(low_pre):
    post = [0.001, -0.004, -0.004, -0.004, -0.004, -0.004, 0.003]
    f = _frame(low_pre, 6 * 0.003, post)
    s = _strat(trail_sigma=100.0)(f).to_numpy()  # trail out of the way: test the shock-open invalidation
    c, o = f["close"].to_numpy(), f["open"][SHOCK]
    hit = SHOCK + int(np.argmax(c[SHOCK:] < o))
    assert hit > SHOCK + 1
    assert (s[SHOCK:hit] > 0).all() and (s[hit:] == 0).all()


def test_high_vol_shock_reverses_and_exits_at_target(high_pre):
    post = [-0.02] * 8 + [0.01] * 3
    f = _frame(high_pre, 5 * 0.02, post)
    s = _strat()(f).to_numpy()
    assert s[:SHOCK].max() == 0 and s[:SHOCK].min() == 0
    assert s[SHOCK] < 0  # reversal: short an up-shock
    c, o = f["close"].to_numpy(), f["open"][SHOCK]
    hit = SHOCK + int(np.argmax(c[SHOCK:] <= o))
    assert hit > SHOCK + 1
    assert (s[SHOCK:hit] < 0).all() and (s[hit:] == 0).all()


def test_reversal_invalidation_goes_flat(high_pre):
    post = [0.01] * 6 + [-0.03] * 6
    f = _frame(high_pre, 5 * 0.02, post)
    s = _strat()(f).to_numpy()
    c = f["close"].to_numpy()
    lvl = f["high"][SHOCK] + 0.5 * f["sigma_next"][SHOCK] * c[SHOCK]
    hit = SHOCK + int(np.argmax(c[SHOCK:] > lvl))
    assert hit > SHOCK + 1
    assert (s[SHOCK:hit] < 0).all() and (s[hit:] == 0).all()  # flat even though price later falls to the target


def test_new_shock_replaces_position(high_pre):
    post = [0.0] * 30 + [-5 * 0.02] + [0.0] * 3  # second (down) shock 31 bars later: past the decluster window
    f = _frame(high_pre, 5 * 0.02, post)
    s = _strat(invalid_sigma=100.0)(f).to_numpy()
    assert s[SHOCK] < 0
    second = SHOCK + 31
    assert abs(f["z"][second]) >= 4
    assert s[second] > 0  # re-evaluated on the new shock (fade the down-shock)


def test_no_shock_no_trade(low_pre, high_pre):
    for pre in (low_pre, high_pre):
        f = _frame(pre, None, [0.001] * 50)
        assert (_strat()(f).to_numpy() == 0).all()


def test_causal(low_pre, high_pre):
    for pre, shock, post in ((low_pre, 0.018, [0.002] * 5 + [-0.004] * 10), (high_pre, 0.1, [-0.02] * 8)):
        f = _frame(pre, shock, post)
        assert_causal(_strat(), f, cut_points=6)
        assert_causal(_strat(), f.head(SHOCK + 6), cut_points=4, seed=3)


def test_spec():
    sp = REGISTRY["shock_regime"].spec
    assert sp.family == "shock" and sp.version == "0.1.0" and sp.requires == ("bars",)
    assert "H-S2" in sp.hypothesis and "H-S3" in sp.hypothesis and "forward" in sp.hypothesis
    assert _strat().params["z_min"] == 4.0
