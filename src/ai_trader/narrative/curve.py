"""Yield-curve regimes, exactly as registered in the MAC block of research/hypotheses.yaml.

Per FRED observation day: d2, d10 = 5-observation changes of DGS2 / DGS10 (in bp, on the days both exist).
  no_regime        |d2| < 5 bp and |d10| < 5 bp
  otherwise the dominant leg (larger |change|; a tie goes to the 2y) decides, which is the only reading under which
  the four registered phrases are mutually exclusive:
  bear_flattening  2y dominant, up      ("2y up > 10y up")
  bull_steepening  2y dominant, down    ("2y down > 10y down")
  bear_steepening  10y dominant, up     ("10y up > 2y up")
  bull_flattening  10y dominant, down   ("10y down > 2y down")
known_ts = obs_date + 1 day 00:00 UTC (registered).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

REGIMES = ("bear_flattening", "bear_steepening", "bull_flattening", "bull_steepening", "no_regime")
BAND_BP = 5.0
LAG = 5


def classify(d2_bp: float, d10_bp: float, band: float = BAND_BP) -> str:
    if d2_bp != d2_bp or d10_bp != d10_bp:   # NaN
        return "no_regime"
    if abs(d2_bp) < band and abs(d10_bp) < band:
        return "no_regime"
    if abs(d2_bp) >= abs(d10_bp):
        return "bear_flattening" if d2_bp > 0 else "bull_steepening"
    return "bear_steepening" if d10_bp > 0 else "bull_flattening"


def regimes(y2: pl.DataFrame, y10: pl.DataFrame, lag: int = LAG) -> pl.DataFrame:
    """y2, y10: obs_date, value (percent). Returns obs_date, known_ts, y2, y10, spread_bp, d2_bp, d10_bp, regime."""
    j = (y2.select("obs_date", pl.col("value").alias("y2")).join(y10.select("obs_date", pl.col("value").alias("y10")), on="obs_date", how="inner")
         .sort("obs_date"))
    j = j.with_columns(((pl.col("y2") - pl.col("y2").shift(lag)) * 100).alias("d2_bp"),
                       ((pl.col("y10") - pl.col("y10").shift(lag)) * 100).alias("d10_bp"),
                       ((pl.col("y10") - pl.col("y2")) * 100).alias("spread_bp"))
    reg = [classify(a, b) if a is not None and b is not None else None for a, b in zip(j["d2_bp"].to_list(), j["d10_bp"].to_list(), strict=True)]
    j = j.with_columns(pl.Series("regime", reg, dtype=pl.Utf8),
                       (pl.col("obs_date").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias("known_ts"))
    return j.drop_nulls("regime").select("obs_date", "known_ts", "y2", "y10", "spread_bp", "d2_bp", "d10_bp", "regime")


def regimes_asof(as_of: datetime) -> pl.DataFrame:
    from ai_trader.narrative.macro import read_asof
    out = regimes(read_asof("DGS2", as_of), read_asof("DGS10", as_of))
    return out.filter(pl.col("known_ts") <= as_of)
