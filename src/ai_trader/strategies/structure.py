"""Swing-structure trend strategies: port of the TradingView "Pivot geometry trend (DC pivots)" indicator.

Directional-change pivots (features.dc_pivots) confirm a swing once the close reverses by K·σ from the
run's extreme, σ = EWMA std of close changes (half-life 60 bars). Each model is the mean of five K votes
(K = 3, 4, 6, 8, 12), so conviction is in {0, ±0.2, ..., ±1}. Pivots are known only at their
confirmation bar, so the signals are causal by construction.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ai_trader.features.dc_pivots import geometry

from .base import Strategy, StrategySpec, register
from .trend import series

_PARAMS = {"ks": [3, 4, 6, 8, 12], "vol_hl": 60, "warm": 60, "max_retr": 0.618}
_REGIMES = ("R1", "R2", "R3", "R7", "R8", "R9")  # all trending cells, any vol: the models only trade structure
_PRIOR = (
    " Pre-registered as H-GT1; the prior project found pivot-geometry trend models added nothing beyond "
    "slow trend following (H-GT2 tests that directly)."
)


class _Geo(Strategy):
    column: str = ""

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        if frame.height == 0:
            return pl.Series(self.spec.name, [], dtype=pl.Float64)
        g = geometry(frame, ks=tuple(int(k) for k in p["ks"]), vol_hl=float(p["vol_hl"]), warm=int(p["warm"]), max_retr=float(p["max_retr"]))
        return series(np.asarray(g[self.column].to_numpy(), dtype=float), self.spec.name)


@register
class GeoStruct(_Geo):
    column = "geo_struct"
    spec = StrategySpec(
        name="geo_struct",
        version="0.1.0",
        family="trend",
        description=(
            "Dow structure on directional-change pivots: per K, +1 when the last two confirmed swing highs "
            "and lows are both rising, -1 when both falling, else 0; signal = mean over K."
        ),
        hypothesis="Higher highs with higher lows (in vol-scaled swings) are followed by positive next-bar returns, and vice versa." + _PRIOR,
        requires=("bars",),
        expected_regimes=_REGIMES,
        params=dict(_PARAMS),
    )


@register
class GeoPivotBO(_Geo):
    column = "geo_pivot_bo"
    spec = StrategySpec(
        name="geo_pivot_bo",
        version="0.1.0",
        family="trend",
        description=(
            "Swing breakout: per K, long once the close exceeds the last confirmed swing high, short once it "
            "falls below the last confirmed swing low, held until the opposite break; signal = mean over K."
        ),
        hypothesis="Closes beyond the last confirmed swing extreme are followed by continuation in the breakout direction." + _PRIOR,
        requires=("bars",),
        expected_regimes=_REGIMES,
        params=dict(_PARAMS),
    )


@register
class GeoFlagBO(_Geo):
    column = "geo_flag_bo"
    spec = StrategySpec(
        name="geo_flag_bo",
        version="0.1.0",
        family="trend",
        description=(
            "Flag continuation: per K, after a higher low with retracement (H0-L0)/(H0-L1) <= 0.618, long on a "
            "close above H0 with a trailing stop at the latest swing low (ratchets only up); short symmetric; "
            "flat on a close through the stop. Signal = mean over K."
        ),
        hypothesis="Shallow pullbacks inside an established swing trend resolve in the trend direction." + _PRIOR,
        requires=("bars",),
        expected_regimes=_REGIMES,
        params=dict(_PARAMS),
    )
