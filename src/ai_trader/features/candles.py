"""Vector candles (research/hypotheses.yaml, H-VC*): causal per-bar flags, exactly as registered.

    range_sig = ln(high/low) / sigma_prev                        (sigma_prev: forecast made before bar t)
    vol_rel   = turnover / median(turnover over the PRIOR 168 bars, current bar excluded)
    body      = |close - open| / (high - low)
    is_vector = range_sig >= 3 AND vol_rel >= 2 AND body >= 0.6
    direction = sign(close - open)
    zone      = body [min(o, c), max(o, c)]; zone_mid = its midpoint
    thin      = vol_rel / range_sig <= the 20% quantile of its own PRIOR 168-bar values

Everything is a backward-looking rolling statistic, so row t depends only on rows <= t.
`decluster` keeps an event only if no *kept* event occurred in the previous 24 bars (same rule as
research.evaluate.shock_events). No parameter here was tuned: they are the registry's constants.
"""

from __future__ import annotations

import numpy as np
import polars as pl

WINDOW = 168
RANGE_MIN = 3.0
VOL_REL_MIN = 2.0
BODY_MIN = 0.6
THIN_Q = 0.2
DECLUSTER = 24


def vector_candles(frame: pl.DataFrame) -> pl.DataFrame:
    """Add range_sig, vol_rel, body, is_vector, direction, thin, zone_lo, zone_hi, zone_mid.

    Requires open/high/low/close/turnover and sigma_prev (research.frame.market_frame / enrich_bars).
    Warm-up rows (fewer than 168 prior bars) have null vol_rel / thin and is_vector = False."""
    f = frame.sort("event_ts")
    rng = pl.col("high") - pl.col("low")
    f = f.with_columns(
        ((pl.col("high") / pl.col("low")).log() / pl.col("sigma_prev")).fill_nan(None).alias("range_sig"),
        (pl.col("turnover") / pl.col("turnover").shift(1).rolling_median(window_size=WINDOW, min_samples=WINDOW)).fill_nan(None).alias("vol_rel"),
        pl.when(rng > 0).then((pl.col("close") - pl.col("open")).abs() / rng).otherwise(None).alias("body"),
        (pl.col("close") - pl.col("open")).sign().cast(pl.Int8).alias("direction"),
        pl.min_horizontal("open", "close").alias("zone_lo"),
        pl.max_horizontal("open", "close").alias("zone_hi"),
    ).with_columns(
        ((pl.col("zone_lo") + pl.col("zone_hi")) / 2).alias("zone_mid"),
        (pl.col("vol_rel") / pl.col("range_sig")).fill_nan(None).alias("_vpr"),
    )
    q = pl.col("_vpr").shift(1).rolling_quantile(THIN_Q, interpolation="linear", window_size=WINDOW, min_samples=WINDOW)
    f = f.with_columns(
        pl.when(pl.col("_vpr").is_null() | q.is_null()).then(None).otherwise(pl.col("_vpr") <= q).alias("thin"),
        ((pl.col("range_sig") >= RANGE_MIN) & (pl.col("vol_rel") >= VOL_REL_MIN) & (pl.col("body") >= BODY_MIN)).fill_null(False).alias("is_vector"),
    )
    return f.drop("_vpr")


def decluster(flags: np.ndarray, gap: int = DECLUSTER) -> np.ndarray:
    """Causal declustering: keep flagged bar i only if i - (last kept index) > gap."""
    flags = np.asarray(flags, dtype=bool)
    keep = np.zeros(len(flags), dtype=bool)
    last = -(10**9)
    for i in np.flatnonzero(flags):
        if i - last > gap:
            keep[i] = True
            last = i
    return keep


def vector_events(frame: pl.DataFrame, gap: int = DECLUSTER) -> pl.DataFrame:
    """Declustered vector-candle rows of a frame (already passed through vector_candles), with row index `i`."""
    keep = decluster(frame["is_vector"].to_numpy(), gap)
    idx = np.flatnonzero(keep)
    return frame[idx.tolist()].with_columns(pl.Series("i", idx, dtype=pl.Int64))
