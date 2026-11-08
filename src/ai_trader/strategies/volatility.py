"""Volatility strategies: trade transitions between vol states.

Volatility clusters, and low-vol stretches tend to end in directional expansion. Compression is
judged relative to the asset's own trailing history (rolling median), never a full-sample level.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .base import Strategy, StrategySpec, register
from .trend import arr, lag, rmax, rmedian, rmin, rmean, safe_div, series


@register
class CompressionBreakout(Strategy):
    spec = StrategySpec(
        name="compression_breakout",
        version="0.1.0",
        family="volatility",
        description=(
            "Compression = short-window realised vol (RMS of log returns over W bars) below a "
            "fraction of its own trailing L-bar median. If the bar before t was compressed and the "
            "close of t breaks the W-bar box (high/low of the W bars before t), take the breakout "
            "direction and remember the box. Exit when the close re-enters the box's middle (crosses "
            "the box midpoint). Conviction is 0.5 + 0.5·tanh(d/scale), d = distance beyond the "
            "broken box edge in one-bar σ units."
        ),
        hypothesis=(
            "When realised vol is unusually low relative to its recent history, a close outside the "
            "compressed range is followed by continuation in that direction."
        ),
        requires=("bars",),
        expected_regimes=("R4", "R5"),
        # 1-day box vs 30-day history on 1h bars; "compressed" = below 60% of the median rv.
        params={"window": 24, "history": 720, "ratio": 0.6, "scale": 3.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        w, L = int(p["window"]), int(p["history"])
        c, h, lo, s, r = (arr(frame, k) for k in ("close", "high", "low", "sigma_next", "ret"))
        rv = np.sqrt(rmean(r**2, w))
        rel = safe_div(rv, rmedian(rv, L, min_samples=L // 2))
        # State of the bar *before* t: the breakout bar itself expands vol.
        compressed_prev = lag(rel, 1) < p["ratio"]
        box_hi, box_lo = lag(rmax(h, w), 1), lag(rmin(lo, w), 1)
        out = np.zeros(len(c))
        pos, edge, mid = 0, np.nan, np.nan
        for t in range(len(c)):
            if pos == 1 and c[t] < mid or pos == -1 and c[t] > mid:
                pos = 0
            if not pos and compressed_prev[t] and np.isfinite(box_hi[t]):
                if c[t] > box_hi[t]:
                    pos, edge = 1, box_hi[t]
                elif c[t] < box_lo[t]:
                    pos, edge = -1, box_lo[t]
                if pos:
                    mid = 0.5 * (box_hi[t] + box_lo[t])
            if pos and np.isfinite(s[t]) and s[t] > 0:
                d = pos * np.log(c[t] / edge) / s[t]
                out[t] = pos * (0.5 + 0.5 * np.tanh(d / p["scale"]))
        return series(out, self.spec.name)
