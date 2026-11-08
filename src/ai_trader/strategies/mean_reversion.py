"""Mean-reversion strategies: fade stretched prices back toward a trailing anchor.

Stretch is measured in σ units so thresholds are comparable across assets and vol states. Entry
and exit use hysteresis on price state (enter beyond a threshold, exit near the anchor), never a
holding-period limit.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .base import Strategy, StrategySpec, register
from .trend import arr, ema, rstd, safe_div, series


@register
class ZScoreReversion(Strategy):
    spec = StrategySpec(
        name="zscore_reversion",
        version="0.1.0",
        family="mean_reversion",
        description=(
            "Distance of the close from its EMA in σ units: dev = log(c/EMA) divided by the trailing "
            "std of that log deviation over `std_window` bars. A slow trailing std is used instead of "
            "σ̂_{t+1|t}·√(span/2) because the EWMA σ̂ reacts within a few bars and would standardise "
            "the very spike being faded back below the threshold. Enter a fade when |dev| exceeds the entry threshold, keep it while the deviation keeps "
            "its sign and |dev| stays above the exit threshold; while active the signal is "
            "-tanh(dev/scale)."
        ),
        hypothesis=(
            "Price deviations from a 2-day EMA larger than 2 standard deviations shrink over the "
            "following bars more than a random walk implies."
        ),
        requires=("bars",),
        expected_regimes=("R5", "R6"),
        # 2-day anchor, 10-day std window on 1h bars; 2σ entry / 0.5σ exit are conventional, not fitted.
        params={"span": 48, "std_window": 240, "entry": 2.0, "exit": 0.5, "scale": 2.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        c = arr(frame, "close")
        raw = np.log(c) - np.log(ema(c, int(p["span"])))
        w = int(p["std_window"])
        sd = rstd(raw, w, w // 2)
        dev = safe_div(raw, np.where(sd > 0, sd, np.nan))
        out = np.zeros(len(c))
        side = 0  # +1 = fading an up-stretch (short), -1 = fading a down-stretch (long)
        for t in range(len(c)):
            d = dev[t]
            if not np.isfinite(d):
                side = 0
                continue
            if side and (np.sign(d) != side or abs(d) < p["exit"]):
                side = 0
            if not side and abs(d) > p["entry"]:
                side = int(np.sign(d))
            if side:
                out[t] = -np.tanh(d / p["scale"])
        return series(out, self.spec.name)


def wilder_rsi(close: np.ndarray, n: int) -> np.ndarray:
    """Wilder RSI: SMA seed over the first n changes, then recursive (1/n) smoothing. Causal."""
    out = np.full(len(close), np.nan)
    d = np.diff(close)
    if len(d) < n:
        return out
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:n].mean(), l[:n].mean()
    for i in range(n, len(d) + 1):
        if i > n:
            ag = (ag * (n - 1) + g[i - 1]) / n
            al = (al * (n - 1) + l[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


@register
class RSIReversion(Strategy):
    spec = StrategySpec(
        name="rsi_reversion",
        version="0.1.0",
        family="mean_reversion",
        description=(
            "Wilder RSI fade. Long when RSI < low, short when RSI > high, flat in between; "
            "conviction ramps linearly from 0 at the threshold to 1 at `full` points beyond it."
        ),
        hypothesis=(
            "After a run of one-sided bar changes that pushes 14-bar RSI outside 30-70, the next "
            "bar's return tends to have the opposite sign."
        ),
        requires=("bars",),
        expected_regimes=("R4", "R5"),
        # Wilder's original 14 / 30 / 70; full conviction at RSI 10 / 90.
        params={"period": 14, "low": 30.0, "high": 70.0, "full": 20.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        rsi = wilder_rsi(arr(frame, "close"), int(p["period"]))
        with np.errstate(invalid="ignore"):
            long_ = np.clip((p["low"] - rsi) / p["full"], 0, 1)
            short = np.clip((rsi - p["high"]) / p["full"], 0, 1)
        return series(long_ - short, self.spec.name)
