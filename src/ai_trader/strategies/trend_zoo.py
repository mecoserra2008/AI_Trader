"""Fixed zoo of simple trend-following models (research/hypotheses.yaml, block "Trend vs mean-reversion
regimes + trend model zoo"; H-TR1..H-TR4).

The zoo is defined in **days** and built for a bar size via `zoo(bars_per_day)`; `ZOO` is the 4h build
(1 day = 6 bars). Every variant is a causal function `frame -> pl.Series` in [-1, 1] (row t uses rows
<= t only; `fwd_ret` is dropped before computing). The variants are deliberately NOT registered in the
global strategy REGISTRY: they are a research panel, not production strategies.

Variants (21, fixed before testing; no parameter search beyond this list):
  tsmom_{1,3,7,14,30}d        tanh(z), z = log(c_t / c_{t-k}) / (σ̂ √k)
  mac_{f}_{s}                 (1,7) (3,14) (7,30) (14,60) days: SMA_f - SMA_s of log price, as a z-score
                              under a random-walk null (exact null std from the filter's return weights)
  donch_{7,14,30,60}          Donchian breakout (strategies.trend.DonchianBreakout), exit channel = half
                              the entry length, conviction scale 3.0 as configured
  ema_1_4_16, ema_4_16_64     EMA stack (strategies.trend.EMAStack), spans in days, k = √slow / 1.4 (the
                              rule documented in trend.py for the configured stack)
  slope_{7,14,30}d            OLS slope of log price on time over the window, as a t-stat whose standard
                              error is the random-walk-null one (the textbook OLS s.e. is spurious on
                              integrated prices: residuals are autocorrelated, t grows like √n)
  cfg_tsmom, cfg_ema_stack, cfg_donchian_breakout
                              the registered strategies exactly as configured (config/strategies.yaml),
                              parameters in bars (so on 4h bars they are 4x slower than on 1h)
The linear-filter variants (tsmom, mac, slope) all use z = Σ c_m r_{t-m} / (σ̂ √Σ c_m²): a vol-scaled
t-stat that is N(0, 1) under a driftless random walk, mapped through tanh.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from .base import REGISTRY
from .trend import DonchianBreakout, EMAStack, arr

ZooFn = Callable[[pl.DataFrame], pl.Series]

TSMOM_DAYS = (1, 3, 7, 14, 30)
MAC_DAYS = ((1, 7), (3, 14), (7, 30), (14, 60))
DONCH_DAYS = (7, 14, 30, 60)
EMA_DAYS = ((1, 4, 16), (4, 16, 64))
SLOPE_DAYS = (7, 14, 30)
CONFIGURED = ("tsmom", "ema_stack", "donchian_breakout")


# ---------------------------------------------------------------- linear filters on log price
def _filter_z(close: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """z_t = Σ_j w_j log c_{t-n+1+j} / (σ̂_t · ||c||), with w (oldest -> newest) summing to 0.

    Rewritten on returns, the filter is Σ_m c_m r_{t-m} with c_m = -cumsum(w)[n-2-m]; under a random
    walk with per-bar vol σ its standard deviation is σ·sqrt(Σ c_m²), so z ~ N(0, 1) under the null."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    x = np.log(np.asarray(close, dtype=float))
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        # Direct (non-FFT) sliding dot product: each output uses only its own window -> truncation invariant.
        win = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = win @ w
    norm = float(np.sqrt(np.sum(np.cumsum(w)[:-1] ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = out / (np.asarray(sigma, dtype=float) * norm)
    return np.where(np.isfinite(z), z, np.nan)


def tsmom_weights(k: int) -> np.ndarray:
    w = np.zeros(k + 1)
    w[0], w[-1] = -1.0, 1.0
    return w


def mac_weights(fast: int, slow: int) -> np.ndarray:
    w = np.full(slow, -1.0 / slow)
    w[-fast:] += 1.0 / fast
    return w


def slope_weights(n: int) -> np.ndarray:
    j = np.arange(n, dtype=float)
    d = j - j.mean()
    return d / np.sum(d**2)


def _clean(x: np.ndarray | pl.Series, name: str) -> pl.Series:
    v = x.cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy() if isinstance(x, pl.Series) else np.asarray(x, dtype=float)
    v = np.clip(np.where(np.isfinite(v), v, 0.0), -1.0, 1.0)
    return pl.Series(name, v)


def _strip(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop("fwd_ret") if "fwd_ret" in frame.columns else frame


def _linear(name: str, w: np.ndarray) -> ZooFn:
    def fn(frame: pl.DataFrame) -> pl.Series:
        frame = _strip(frame)
        return _clean(np.tanh(_filter_z(arr(frame, "close"), arr(frame, "sigma_next"), w)), name)

    fn.__name__ = name
    return fn


def _wrap(name: str, strat) -> ZooFn:
    def fn(frame: pl.DataFrame) -> pl.Series:
        return _clean(strat(frame), name)  # Strategy.__call__ drops labels itself

    fn.__name__ = name
    return fn


# ---------------------------------------------------------------- the zoo
def zoo(bars_per_day: int = 6) -> dict[str, ZooFn]:
    b = int(bars_per_day)
    z: dict[str, ZooFn] = {}
    for d in TSMOM_DAYS:
        z[f"tsmom_{d}d"] = _linear(f"tsmom_{d}d", tsmom_weights(d * b))
    for f, s in MAC_DAYS:
        z[f"mac_{f}_{s}"] = _linear(f"mac_{f}_{s}", mac_weights(f * b, s * b))
    for d in DONCH_DAYS:
        n = d * b
        z[f"donch_{d}"] = _wrap(f"donch_{d}", DonchianBreakout(entry_window=n, exit_window=max(1, n // 2), scale=3.0))
    for f, m, s in EMA_DAYS:
        z[f"ema_{f}_{m}_{s}"] = _wrap(f"ema_{f}_{m}_{s}", EMAStack(fast=f * b, mid=m * b, slow=s * b, k=float(np.sqrt(s * b) / 1.4)))
    for d in SLOPE_DAYS:
        z[f"slope_{d}d"] = _linear(f"slope_{d}d", slope_weights(d * b))
    for name in CONFIGURED:
        z[f"cfg_{name}"] = _wrap(f"cfg_{name}", REGISTRY[name]())
    return z


def lookback_bars(bars_per_day: int = 6) -> int:
    """Bars needed before every variant is warm (longest window + 1, plus the σ̂ warm-up)."""
    b = int(bars_per_day)
    cfg = [max(REGISTRY["tsmom"]().params["horizons"]) + 1, int(REGISTRY["ema_stack"]().params["slow"]),
           int(REGISTRY["donchian_breakout"]().params["entry_window"]) + 1]
    own = [max(TSMOM_DAYS) * b + 1, max(s for _, s in MAC_DAYS) * b, max(DONCH_DAYS) * b + 1,
           max(s for *_, s in EMA_DAYS) * b, max(SLOPE_DAYS) * b]
    return max(own + cfg) + 30


ZOO: dict[str, ZooFn] = zoo(6)


def ensemble(frame: pl.DataFrame, bars_per_day: int = 6, variants: dict[str, ZooFn] | None = None) -> pl.Series:
    """Equal-weight mean of all zoo variant signals (H-TR1)."""
    v = variants if variants is not None else (ZOO if bars_per_day == 6 else zoo(bars_per_day))
    sigs = np.vstack([fn(frame).to_numpy() for fn in v.values()])
    return _clean(sigs.mean(axis=0), "ensemble")
