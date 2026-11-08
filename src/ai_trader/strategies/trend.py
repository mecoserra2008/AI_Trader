"""Trend strategies: price persistence measured in volatility units.

All moves are standardised by σ̂ (sigma_next, known at the close of t) times √horizon, so the same
parameters mean the same thing across assets and vol states. Logic runs in numpy because polars
orders NaN above every number in comparisons; polars is used only for causal rolling/ewm kernels.
The small helpers here are shared by the other strategy modules.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .base import Strategy, StrategySpec, register


# ---------------------------------------------------------------- shared helpers (causal only)
def arr(frame: pl.DataFrame, name: str) -> np.ndarray | None:
    """Column as float64 numpy with nulls -> NaN; None if the column is absent."""
    if name not in frame.columns:
        return None
    return frame[name].cast(pl.Float64).fill_null(float("nan")).to_numpy().copy()


def _ser(x: np.ndarray) -> pl.Series:
    return pl.Series(np.asarray(x, dtype=float)).fill_nan(None)


def _out(s: pl.Series) -> np.ndarray:
    return s.cast(pl.Float64).fill_null(float("nan")).to_numpy().copy()


def rmax(x: np.ndarray, w: int) -> np.ndarray:
    return _out(_ser(x).rolling_max(w, min_samples=w))


def rmin(x: np.ndarray, w: int) -> np.ndarray:
    return _out(_ser(x).rolling_min(w, min_samples=w))


def rmean(x: np.ndarray, w: int, min_samples: int | None = None) -> np.ndarray:
    return _out(_ser(x).rolling_mean(w, min_samples=min_samples or w))


def rsum(x: np.ndarray, w: int) -> np.ndarray:
    return _out(_ser(x).rolling_sum(w, min_samples=w))


def rstd(x: np.ndarray, w: int, min_samples: int | None = None) -> np.ndarray:
    return _out(_ser(x).rolling_std(w, min_samples=min_samples or w))


def rmedian(x: np.ndarray, w: int, min_samples: int | None = None) -> np.ndarray:
    return _out(_ser(x).rolling_median(w, min_samples=min_samples or w))


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Recursive EMA (adjust=False), null until `span` observations exist."""
    return _out(_ser(x).ewm_mean(span=span, adjust=False, min_samples=span))


def lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(np.asarray(x, dtype=float), np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Cast first: columns that are entirely null arrive as object arrays.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(np.isfinite(out), out, np.nan)


def vol_move(close: np.ndarray, sigma: np.ndarray, k: int) -> np.ndarray:
    """log(c_t / c_{t-k}) / (σ̂ √k): a k-bar move in standard deviations."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return safe_div(np.log(close) - np.log(lag(close, k)), sigma * np.sqrt(k))


def series(x: np.ndarray, name: str) -> pl.Series:
    return pl.Series(name, np.where(np.isfinite(x), x, 0.0))


# ---------------------------------------------------------------- strategies
@register
class TSMom(Strategy):
    spec = StrategySpec(
        name="tsmom",
        version="0.1.0",
        family="trend",
        description=(
            "Multi-horizon time-series momentum. For each horizon k the trailing log return is "
            "expressed in standard deviations, z_k = log(c_t/c_{t-k}) / (σ̂_{t+1|t}·√k); the signal is "
            "tanh of the average z_k across horizons divided by a scale, so it is continuous and "
            "saturates on strong, multi-horizon-consistent trends."
        ),
        hypothesis=(
            "Vol-standardised returns over the past 1-10 days are positively autocorrelated with the "
            "next bar's return (trends persist more than a random walk implies)."
        ),
        requires=("bars",),
        expected_regimes=("R1", "R2", "R7", "R8"),
        # 1, 3 and 10 days on 1h bars; scale 1.0 = a 1σ average move gives ~0.76 conviction.
        params={"horizons": [24, 72, 240], "scale": 1.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        c, s = arr(frame, "close"), arr(frame, "sigma_next")
        zs = np.vstack([vol_move(c, s, int(k)) for k in self.params["horizons"]])
        with np.errstate(invalid="ignore"):
            m = zs.mean(axis=0)  # NaN until the longest horizon is available
        return series(np.tanh(m / self.params["scale"]), self.spec.name)


@register
class DonchianBreakout(Strategy):
    spec = StrategySpec(
        name="donchian_breakout",
        version="0.1.0",
        family="trend",
        description=(
            "Channel breakout. Go long when the close exceeds the prior N-bar high (current bar "
            "excluded), short below the prior N-bar low; hold until the close breaks the opposite, "
            "shorter M-bar channel. Conviction is 0.5 + 0.5·tanh(d/scale), where d is the distance of "
            "the close beyond the channel level that was broken at entry, in one-bar σ units."
        ),
        hypothesis=(
            "Closes outside a multi-day range are followed by continuation in the breakout direction "
            "more often than by reversal."
        ),
        requires=("bars",),
        expected_regimes=("R2", "R3", "R8", "R4"),
        # N = 5 days, M = 2 days on 1h bars (classic turtle 20/10 ratio, shortened for crypto).
        params={"entry_window": 120, "exit_window": 48, "scale": 3.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        n, m = int(p["entry_window"]), int(p["exit_window"])
        c, h, lo, s = arr(frame, "close"), arr(frame, "high"), arr(frame, "low"), arr(frame, "sigma_next")
        hi_n, lo_n = lag(rmax(h, n), 1), lag(rmin(lo, n), 1)
        hi_m, lo_m = lag(rmax(h, m), 1), lag(rmin(lo, m), 1)
        out = np.zeros(len(c))
        pos, level = 0, np.nan
        for t in range(len(c)):
            if not (np.isfinite(hi_n[t]) and np.isfinite(s[t]) and s[t] > 0):
                continue
            if c[t] > hi_n[t]:
                if pos != 1:
                    level = hi_n[t]
                pos = 1
            elif c[t] < lo_n[t]:
                if pos != -1:
                    level = lo_n[t]
                pos = -1
            elif (pos == 1 and c[t] < lo_m[t]) or (pos == -1 and c[t] > hi_m[t]):
                pos = 0
            if pos:
                d = pos * np.log(c[t] / level) / s[t]
                out[t] = pos * (0.5 + 0.5 * np.tanh(d / p["scale"]))
        return series(out, self.spec.name)


@register
class EMAStack(Strategy):
    spec = StrategySpec(
        name="ema_stack",
        version="0.1.0",
        family="trend",
        description=(
            "EMA alignment. Long only when fast > mid > slow, short only when fast < mid < slow, flat "
            "otherwise. Conviction is tanh(|fast - slow| / (close·σ̂·k)): the fast-slow gap in one-bar "
            "σ units divided by k, which roughly absorbs the √span growth of the gap under a trend."
        ),
        hypothesis=(
            "When short, medium and long smoothed prices are ordered in the same direction, the next "
            "bar's return has the sign of that ordering."
        ),
        requires=("bars",),
        expected_regimes=("R1", "R2", "R7", "R8"),
        # 12h / 2d / 8d spans on 1h bars (~1:4:16 spacing); k ≈ √192 / 1.4.
        params={"fast": 12, "mid": 48, "slow": 192, "k": 10.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        c, s = arr(frame, "close"), arr(frame, "sigma_next")
        f, md, sl = ema(c, int(p["fast"])), ema(c, int(p["mid"])), ema(c, int(p["slow"]))
        with np.errstate(invalid="ignore"):
            align = np.where((f > md) & (md > sl), 1.0, np.where((f < md) & (md < sl), -1.0, 0.0))
        gap = safe_div(np.abs(f - sl), c * s * p["k"])
        return series(align * np.tanh(gap), self.spec.name)
