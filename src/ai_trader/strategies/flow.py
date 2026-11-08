"""Order-flow strategies: aggressor (taker) buy/sell volume.

Flow history is short and patchy, so every window requires complete flow data and the signal is 0
wherever buy_volume / sell_volume are absent or null. The frame's `cvd` is a cumulative sum with
gaps filled by 0, so CVD changes are recomputed here as rolling sums of buy - sell, which are null
over any window that touches a gap.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .base import Strategy, StrategySpec, register
from .trend import arr, lag, rmax, rmin, rstd, rsum, safe_div, series


def _flow(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    b, s = arr(frame, "buy_volume"), arr(frame, "sell_volume")
    if b is None or s is None or not (np.isfinite(b) & np.isfinite(s)).any():
        return None
    return b, s


@register
class FlowMomentum(Strategy):
    spec = StrategySpec(
        name="flow_momentum",
        version="0.1.0",
        family="flow",
        description=(
            "Sum of the per-bar aggressor imbalance (buy - sell)/(buy + sell) over N bars, divided by "
            "the trailing std of that sum (not demeaned: zero imbalance is the natural neutral). "
            "Signal = tanh(z/scale): follow persistent one-sided aggression."
        ),
        hypothesis=(
            "Persistent one-sided taker aggression over the past day predicts next-bar returns in the "
            "same direction (informed or forced flow is split over time)."
        ),
        requires=("bars", "flow"),
        expected_regimes=("R2", "R3", "R8"),
        # 1-day accumulation; 7-day minimum history for the std because flow history is short.
        params={"window": 24, "std_window": 720, "min_samples": 168, "scale": 1.5},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        fl = _flow(frame)
        if fl is None:
            return pl.Series(self.spec.name, np.zeros(frame.height))
        b, s = fl
        imb = safe_div(b - s, b + s)
        acc = rsum(imb, int(p["window"]))
        sd = rstd(acc, int(p["std_window"]), int(p["min_samples"]))
        z = safe_div(acc, np.where(sd > 0, sd, np.nan))
        return series(np.tanh(z / p["scale"]), self.spec.name)


@register
class CVDDivergence(Strategy):
    spec = StrategySpec(
        name="cvd_divergence",
        version="0.1.0",
        family="flow",
        description=(
            "Close makes a new N-bar high (above the prior N-bar high) while the N-bar CVD change, "
            "normalised by N-bar total volume, is negative -> short; new N-bar low with positive CVD "
            "change -> long. Conviction tanh(|cvd_norm|/scale) is fixed at entry. Exit when the close "
            "crosses the midpoint of the N-bar channel (reversion achieved) or when price makes a new "
            "extreme in the trade's adverse direction with confirming CVD (divergence gone)."
        ),
        hypothesis=(
            "New price extremes not confirmed by net aggressor volume are driven by thin liquidity "
            "rather than demand, and tend to revert toward the range."
        ),
        requires=("bars", "flow"),
        expected_regimes=("R5", "R6"),
        # 1-day extremes; scale 0.1 = a 10% net-volume disagreement gives ~0.76 conviction.
        params={"window": 24, "scale": 0.1},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        fl = _flow(frame)
        if fl is None:
            return pl.Series(self.spec.name, np.zeros(frame.height))
        b, s = fl
        n = int(p["window"])
        cvd_norm = safe_div(rsum(b - s, n), rsum(b + s, n))
        c, h, lo = arr(frame, "close"), arr(frame, "high"), arr(frame, "low")
        hi_n, lo_n = lag(rmax(h, n), 1), lag(rmin(lo, n), 1)
        out = np.zeros(len(c))
        pos, mag = 0, 0.0
        for t in range(len(c)):
            cn = cvd_norm[t]
            if not (np.isfinite(cn) and np.isfinite(hi_n[t])):
                pos = 0  # no flow view through a data gap
                continue
            new_hi, new_lo = c[t] > hi_n[t], c[t] < lo_n[t]
            mid = 0.5 * (hi_n[t] + lo_n[t])
            if pos == -1 and (c[t] < mid or (new_hi and cn >= 0)):
                pos = 0
            elif pos == 1 and (c[t] > mid or (new_lo and cn <= 0)):
                pos = 0
            if new_hi and cn < 0:
                pos, mag = -1, np.tanh(abs(cn) / p["scale"])
            elif new_lo and cn > 0:
                pos, mag = 1, np.tanh(abs(cn) / p["scale"])
            out[t] = pos * mag
        return series(out, self.spec.name)
