"""Positioning strategies: derivatives positioning (funding, open interest) as crowding signals.

Funding and OI are as-of joined on known_ts, so they are step functions on the bar grid. They are
standardised against their own trailing windows only. Where a column is absent or null the signal
is 0 (no view), never an imputed value.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .base import Strategy, StrategySpec, register
from .trend import arr, lag, rmean, rstd, safe_div, series, vol_move


def _zeros(frame: pl.DataFrame, name: str) -> pl.Series:
    return pl.Series(name, np.zeros(frame.height))


@register
class FundingContrarian(Strategy):
    spec = StrategySpec(
        name="funding_contrarian",
        version="0.1.0",
        family="positioning",
        description=(
            "Trailing z-score of the funding rate over W bars. When |z| exceeds the threshold AND the "
            "k-bar vol-standardised return has the same sign as z (the crowd is paying up for a move "
            "that already happened), fade it: signal = -tanh(z/scale). Otherwise flat."
        ),
        hypothesis=(
            "When perpetual funding is extreme relative to its 30-day history and price has already "
            "moved in the direction the crowd is paying for, subsequent returns are against the crowd."
        ),
        requires=("bars", "funding"),
        expected_regimes=("R3", "R9", "R6"),
        # 30 days of hourly rows (≈90 funding prints); 2σ extreme; 3-day crowding look-back.
        params={"window": 720, "min_samples": 240, "threshold": 2.0, "scale": 3.0, "extension": 72},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        f = arr(frame, "funding_rate")
        if f is None or not np.isfinite(f).any():
            return _zeros(frame, self.spec.name)
        w, ms = int(p["window"]), int(p["min_samples"])
        sd = rstd(f, w, ms)
        z = safe_div(f - rmean(f, w, ms), np.where(sd > 0, sd, np.nan))
        ext = vol_move(arr(frame, "close"), arr(frame, "sigma_next"), int(p["extension"]))
        with np.errstate(invalid="ignore"):
            gate = (np.abs(z) > p["threshold"]) & (np.sign(ext) == np.sign(z))
        return series(np.where(gate, -np.tanh(z / p["scale"]), 0.0), self.spec.name)


@register
class OIPriceDivergence(Strategy):
    spec = StrategySpec(
        name="oi_price_divergence",
        version="0.1.0",
        family="positioning",
        description=(
            "k-bar change in log open interest, standardised by its trailing std (oi_z), against the "
            "k-bar vol-standardised price move (pz). Mapping: oi_z > threshold and pz > 0 (leverage "
            "funding an up-move) -> follow, +tanh(pz/scale); oi_z > threshold and pz < 0 (shorts "
            "building into a decline) -> squeeze risk, +squeeze_weight·tanh(|pz|/scale); OI flat or "
            "falling (deleveraging / no new positioning) -> flat. Rising OI is therefore always a "
            "long tilt; the price move only sets its strength and its weight."
        ),
        hypothesis=(
            "Fresh leverage (rising open interest) predicts upward pressure: up-moves funded by new "
            "longs continue and declines funded by new shorts get squeezed."
        ),
        requires=("bars", "oi"),
        expected_regimes=("R3", "R6", "R9"),
        # 1-day OI change vs 30-day history; the squeeze leg is half-weighted because the
        # contrarian reading of falling-price OI build-ups is the weaker prior.
        params={"horizon": 24, "window": 720, "min_samples": 240, "threshold": 1.0, "scale": 1.5, "squeeze_weight": 0.5},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params
        oi = arr(frame, "open_interest")
        if oi is None or not np.isfinite(oi).any():
            return _zeros(frame, self.spec.name)
        k = int(p["horizon"])
        with np.errstate(divide="ignore", invalid="ignore"):
            loi = np.log(np.where(oi > 0, oi, np.nan))
        d = loi - lag(loi, k)
        sd = rstd(d, int(p["window"]), int(p["min_samples"]))
        oi_z = safe_div(d, np.where(sd > 0, sd, np.nan))
        pz = vol_move(arr(frame, "close"), arr(frame, "sigma_next"), k)
        mag = np.tanh(np.abs(pz) / p["scale"])
        with np.errstate(invalid="ignore"):
            rising = oi_z > p["threshold"]
            sig = np.where(rising & (pz > 0), mag, np.where(rising & (pz < 0), p["squeeze_weight"] * mag, 0.0))
        return series(sig, self.spec.name)
