from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.core.config import load
from ai_trader.features.transforms import ewma_sigma_ahead
from ai_trader.strategies import REGISTRY, assert_causal
from ai_trader.strategies.base import CELLS

T0 = datetime(2026, 1, 1, tzinfo=UTC)
MODULES = {f"ai_trader.strategies.{m}" for m in ("trend", "mean_reversion", "volatility", "positioning", "flow")}
MINE = sorted(n for n, cls in REGISTRY.items() if cls.__module__ in MODULES)
LAM = 0.94


def build_frame(close: np.ndarray, rng: np.random.Generator | None = None, **extra: np.ndarray) -> pl.DataFrame:
    """Same derived columns as research.frame.market_frame, from a close path."""
    rng = rng or np.random.default_rng(0)
    n = len(close)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.standard_normal(n)) * 0.001
    ret = np.r_[np.nan, np.diff(np.log(close))]
    sig_prev = ewma_sigma_ahead(np.nan_to_num(ret, nan=0.0), lam=LAM, warmup=min(30, n - 1))
    sig_next = np.sqrt(LAM * sig_prev**2 + (1 - LAM) * np.nan_to_num(ret) ** 2)
    ev = [T0 + timedelta(hours=i) for i in range(n)]
    df = pl.DataFrame({
        "event_ts": ev,
        "close_ts": [t + timedelta(hours=1) for t in ev],
        "known_ts": [t + timedelta(hours=1, seconds=2) for t in ev],
        "open": open_, "high": np.maximum(open_, close) * (1 + wick), "low": np.minimum(open_, close) * (1 - wick),
        "close": close, "volume": 1.0, "turnover": close,
        "ret": ret, "sigma_prev": sig_prev, "sigma_next": sig_next,
    }).with_columns((pl.col("ret") / pl.col("sigma_prev")).alias("z"))
    return df.with_columns([pl.Series(k, v, dtype=pl.Float64) for k, v in extra.items()])


def add_flow(df: pl.DataFrame, buy: np.ndarray, sell: np.ndarray) -> pl.DataFrame:
    return df.with_columns(pl.Series("buy_volume", buy), pl.Series("sell_volume", sell)).with_columns(
        ((pl.col("buy_volume") - pl.col("sell_volume")) / (pl.col("buy_volume") + pl.col("sell_volume"))).alias("flow_imbalance"),
        (pl.col("buy_volume") - pl.col("sell_volume")).fill_null(0).cum_sum().alias("cvd"),
    )


def add_oi(df: pl.DataFrame, oi: np.ndarray) -> pl.DataFrame:
    return df.with_columns(pl.Series("open_interest", oi, dtype=pl.Float64)).with_columns(
        (pl.col("open_interest").log() - pl.col("open_interest").log().shift(1)).alias("oi_chg_1")
    )


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    """~3000 hourly bars of a regime-switching walk with t(4) shocks plus synthetic positioning/flow."""
    rng = np.random.default_rng(42)
    n = 3000
    regimes = [(0.0, 0.004), (0.0008, 0.006), (-0.001, 0.012), (0.0, 0.002), (0.002, 0.02), (-0.0005, 0.008)]
    mu, vol = np.empty(n), np.empty(n)
    t = 0
    while t < n:
        m, v = regimes[rng.integers(len(regimes))]
        k = int(rng.integers(150, 500))
        mu[t : t + k], vol[t : t + k] = m, v
        t += k
    shocks = rng.standard_t(4, n) / np.sqrt(2.0)  # unit variance
    r = mu + vol * shocks
    close = 100 * np.exp(np.cumsum(r))
    # Funding: 8h prints, AR(1) around a positive mean plus a response to recent returns.
    f = np.empty(n)
    x = 0.0
    for i in range(n):
        if i % 8 == 0:
            x = 0.9 * x + 0.00003 * rng.standard_normal() + 0.002 * r[max(0, i - 8) : i + 1].sum()
        f[i] = 0.0001 + x
    # Open interest: 4h prints, log random walk partially driven by |returns|.
    loi = np.cumsum(np.where(np.arange(n) % 4 == 0, 0.01 * rng.standard_normal(n) + 0.5 * np.abs(r), 0.0))
    oi = 1e6 * np.exp(loi)
    imb = np.clip(0.3 * np.tanh(r / vol) + 0.2 * rng.standard_normal(n), -0.9, 0.9)
    tot = np.exp(rng.normal(5, 0.5, n))
    buy, sell = tot * (1 + imb) / 2, tot * (1 - imb) / 2
    buy[:500] = sell[:500] = np.nan  # flow history starts late
    df = build_frame(close, rng, funding_rate=f)
    return add_flow(add_oi(df, oi), buy, sell)


def test_all_registered():
    assert set(MINE) == {
        "tsmom", "donchian_breakout", "ema_stack", "zscore_reversion", "rsi_reversion", "compression_breakout",
        "funding_contrarian", "oi_price_divergence", "flow_momentum", "cvd_divergence",
    }


@pytest.mark.parametrize("name", MINE)
def test_causal_and_bounded(name, frame):
    strat = REGISTRY[name]()
    assert_causal(strat, frame)
    s = strat(frame)
    assert len(s) == frame.height
    v = s.to_numpy()
    assert np.isfinite(v).all() and (np.abs(v) <= 1).all()
    assert (v != 0).any(), f"{name} never produces a signal on the synthetic frame"


@pytest.mark.parametrize("name", MINE)
def test_spec_and_config(name):
    spec = REGISTRY[name].spec
    cfg = load().strategies["strategies"]
    assert name in cfg and isinstance(cfg[name].get("params"), dict)
    assert set(cfg[name]["params"]) == set(spec.params)
    assert spec.expected_regimes and set(spec.expected_regimes) <= set(CELLS)
    assert spec.version == "0.1.0" and spec.hypothesis and spec.description
    assert set(spec.requires) <= {"bars", "funding", "oi", "flow"}


# ---------------------------------------------------------------- family sanity properties
def _walk(n: int, sigma: float = 0.005, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sigma * rng.standard_normal(n)


def _close(r: np.ndarray) -> np.ndarray:
    return 100 * np.exp(np.cumsum(r))


def _trend_frame(direction: int) -> pl.DataFrame:
    r = _walk(1500)
    r[1000:] += direction * 0.003  # 0.6σ per bar for 500 bars
    return build_frame(_close(r))


@pytest.mark.parametrize("name", ["tsmom", "donchian_breakout", "ema_stack"])
@pytest.mark.parametrize("direction", [1, -1])
def test_trend_follows_sustained_trend(name, direction):
    s = REGISTRY[name]()(_trend_frame(direction)).to_numpy()
    assert direction * s[-1] > 0.3


def test_zscore_reversion_fades_spike():
    r = _walk(1000)
    r[-6:] = 0.02  # ~+24σ cumulative over 6 bars
    s = REGISTRY["zscore_reversion"]()(build_frame(_close(r))).to_numpy()
    assert s[-1] < -0.5
    r[-6:] = -0.02
    assert REGISTRY["zscore_reversion"]()(build_frame(_close(r))).to_numpy()[-1] > 0.5


def test_rsi_reversion_fades_run():
    r = _walk(500)
    r[-20:] = 0.004
    assert REGISTRY["rsi_reversion"]()(build_frame(_close(r))).to_numpy()[-1] < -0.5
    r[-20:] = -0.004
    assert REGISTRY["rsi_reversion"]()(build_frame(_close(r))).to_numpy()[-1] > 0.5


def test_compression_breakout_enters_and_exits_on_price():
    r = _walk(1200, sigma=0.01)
    r[1000:1100] = _walk(100, sigma=0.001, seed=3)  # compression
    r[1100] = 0.02  # breakout up
    r[1101:1110] = 0.0005
    c = _close(r[:1110])
    s = REGISTRY["compression_breakout"]()(build_frame(c)).to_numpy()
    assert s[1100] > 0.5 and (s[1100:] > 0).all()
    # Falling back through the box middle exits; staying above it does not.
    df = build_frame(c)
    mid = 0.5 * (df["high"][1076:1100].max() + df["low"][1076:1100].min())  # box remembered at entry
    floor = df["low"][1086:1110].min()  # current box low: staying above it avoids a short breakout
    target = mid - 0.2 * (mid - floor)
    r[1110] = np.log(target / c[-1])
    s2 = REGISTRY["compression_breakout"]()(build_frame(_close(r[:1111]))).to_numpy()
    assert s2[1109] > 0 and s2[1110] == 0


@pytest.mark.parametrize("name,cols", [
    ("funding_contrarian", ["funding_rate"]),
    ("oi_price_divergence", ["open_interest", "oi_chg_1"]),
    ("flow_momentum", ["buy_volume", "sell_volume", "flow_imbalance", "cvd"]),
    ("cvd_divergence", ["buy_volume", "sell_volume", "flow_imbalance", "cvd"]),
])
def test_aux_strategies_zero_without_data(name, cols, frame):
    nulled = frame.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in cols])
    assert (REGISTRY[name]()(nulled).to_numpy() == 0).all()
    assert (REGISTRY[name]()(frame.drop(cols)).to_numpy() == 0).all()


def test_flow_zero_before_flow_history(frame):
    for name in ("flow_momentum", "cvd_divergence"):
        assert (REGISTRY[name]()(frame).to_numpy()[:500] == 0).all()


def test_funding_contrarian_fades_crowded_longs():
    n = 1200
    r = _walk(n)
    r[-72:] += 0.002  # price extended up
    f = 0.0001 + 0.00001 * np.random.default_rng(5).standard_normal(n)
    f[-8:] = 0.001  # extreme positive funding
    s = REGISTRY["funding_contrarian"]()(build_frame(_close(r), funding_rate=f)).to_numpy()
    assert s[-1] < -0.3
    # Same funding extreme without a crowded price move: no view.
    r[-72:] -= 0.004
    s2 = REGISTRY["funding_contrarian"]()(build_frame(_close(r), funding_rate=f)).to_numpy()
    assert s2[-1] == 0


def test_oi_price_divergence_mapping():
    n = 1200
    rng = np.random.default_rng(6)
    loi = np.cumsum(0.002 * rng.standard_normal(n))
    loi[-24:] += np.linspace(0.005, 0.12, 24)  # sharp OI build-up
    r = _walk(n)
    up, down = r.copy(), r.copy()
    up[-24:] += 0.003
    down[-24:] -= 0.003
    sig = REGISTRY["oi_price_divergence"]()
    s_up = sig(add_oi(build_frame(_close(up)), 1e6 * np.exp(loi))).to_numpy()[-1]
    s_dn = sig(add_oi(build_frame(_close(down)), 1e6 * np.exp(loi))).to_numpy()[-1]
    assert s_up > 0.3 and 0 < s_dn < s_up
    # Deleveraging: falling OI -> flat.
    s_dl = sig(add_oi(build_frame(_close(up)), 1e6 * np.exp(-loi))).to_numpy()[-1]
    assert s_dl == 0


def test_flow_momentum_follows_persistent_buying():
    n = 1200
    rng = np.random.default_rng(7)
    imb = 0.1 * rng.standard_normal(n)
    imb[-24:] = 0.4
    tot = np.full(n, 100.0)
    df = add_flow(build_frame(_close(_walk(n))), tot * (1 + imb) / 2, tot * (1 - imb) / 2)
    assert REGISTRY["flow_momentum"]()(df).to_numpy()[-1] > 0.5


def test_cvd_divergence_fades_unconfirmed_high():
    n = 600
    r = _walk(n)
    r[-48:] *= 0.1
    r[-1] = 0.05  # new 24-bar high
    tot = np.full(n, 100.0)
    imb = np.zeros(n)
    imb[-24:] = -0.3  # while net aggressor flow is selling
    df = add_flow(build_frame(_close(r)), tot * (1 + imb) / 2, tot * (1 - imb) / 2)
    assert REGISTRY["cvd_divergence"]()(df).to_numpy()[-1] < -0.5
    imb[-24:] = 0.3  # confirmed by buying: no fade
    df = add_flow(build_frame(_close(r)), tot * (1 + imb) / 2, tot * (1 - imb) / 2)
    assert REGISTRY["cvd_divergence"]()(df).to_numpy()[-1] == 0
