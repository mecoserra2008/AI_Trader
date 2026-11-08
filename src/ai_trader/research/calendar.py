"""Market-hours context for a 24/7 asset (ARCHITECTURE §3.5).

Crypto trades through the weekend, but the venues that absorb macro/institutional information
(CME futures, US cash equities, spot ETFs) do not. Prior (research/hypotheses.yaml, H-W*):
weekend price discovery is thin, so a weekend move *without* information tends to revert once
TradFi reopens, while a move driven by information TradFi couldn't trade on tends to persist
and get absorbed when it reopens.

All hours are computed in America/New_York, so US daylight-saving shifts are handled.
Known simplification: exchange holidays are not modelled yet (a Monday holiday looks like a
normal Monday). The weekend window is the CME Globex crypto closure:
Friday 17:00 ET -> Sunday 18:00 ET.
"""

from __future__ import annotations

import numpy as np
import polars as pl

NY = "America/New_York"
FRI, SAT, SUN = 5, 6, 7  # polars: Monday=1 ... Sunday=7


def add_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-bar context from the bar's open time (event_ts). Pure function of the timestamp."""
    ny = pl.col("event_ts").dt.convert_time_zone(NY)
    # Cast before multiplying: polars returns hour/minute as Int8 (hour * 60 overflows silently).
    wd, minute = ny.dt.weekday(), ny.dt.hour().cast(pl.Int32) * 60 + ny.dt.minute().cast(pl.Int32)
    weekend = ((wd == FRI) & (minute >= 17 * 60)) | (wd == SAT) | ((wd == SUN) & (minute < 18 * 60))
    cme_break = (wd <= 4) & (minute >= 17 * 60) & (minute < 18 * 60)
    nyse = (wd <= FRI) & (minute >= 9 * 60 + 30) & (minute < 16 * 60)
    utc_h = pl.col("event_ts").dt.hour()
    return frame.with_columns(
        weekend.alias("tradfi_weekend"),
        (~weekend & ~cme_break).alias("cme_open"),
        nyse.alias("nyse_open"),
        (wd == 1).alias("is_monday"),
        pl.when(weekend).then(pl.lit("weekend"))
        .when(nyse).then(pl.lit("us_cash"))
        .when(utc_h < 7).then(pl.lit("asia"))
        .when(utc_h < 13).then(pl.lit("europe"))
        .otherwise(pl.lit("us_off_hours")).alias("session"),
        # Weekend id = NY date of the Friday that opened it (null outside weekends).
        pl.when(weekend).then((ny.dt.date() - pl.duration(days=(wd - FRI) % 7)).cast(pl.Utf8)).otherwise(None).alias("weekend_id"),
    )


def weekend_table(frame: pl.DataFrame, baseline_bars: int = 120) -> pl.DataFrame:
    """One row per completed weekend, computed only from bars up to the weekend's end.

    fri_close      close of the last bar before the weekend (≈ CME Friday close)
    sigma_fri      σ̂ per bar known at that close (sigma_next)
    wk_ret, wk_z   weekend log return and its size in σ units: wk_ret / (sigma_fri · √bars)
    rv_ratio       realised weekend vol / expected (sigma_fri · √bars)
    vol_ratio      weekend turnover per bar / trailing weekday turnover per bar (activity surprise)
    wk_high/low    weekend extremes; end_known_ts = when the whole weekend became known
    """
    f = add_calendar(frame) if "tradfi_weekend" not in frame.columns else frame
    # Weekday-only activity baseline: weekend bars are masked out of the trailing mean.
    f = f.with_row_index("_i").with_columns(
        pl.when(pl.col("tradfi_weekend")).then(None).otherwise(pl.col("turnover"))
        .rolling_mean(baseline_bars, min_samples=baseline_bars // 2).alias("_turn_base")
    )
    wk = f.filter(pl.col("tradfi_weekend"))
    if wk.height == 0:
        return pl.DataFrame()
    agg = wk.group_by("weekend_id").agg(
        pl.col("_i").min().alias("first_i"),
        pl.col("_i").max().alias("last_i"),
        pl.len().alias("bars"),
        pl.col("ret").sum().alias("wk_ret"),
        (pl.col("ret") ** 2).sum().sqrt().alias("wk_rv"),
        pl.col("high").max().alias("wk_high"),
        pl.col("low").min().alias("wk_low"),
        pl.col("turnover").mean().alias("wk_turn"),
        pl.col("close").last().alias("sun_close"),
        pl.col("known_ts").max().alias("end_known_ts"),
        pl.col("close_ts").max().alias("end_ts"),
    )
    pre = f.select(
        (pl.col("_i") + 1).alias("first_i"),
        pl.col("close").alias("fri_close"),
        pl.col("sigma_next").alias("sigma_fri"),
        pl.col("_turn_base").alias("turn_base"),
    )
    out = agg.join(pre, on="first_i", how="inner").with_columns(
        (pl.col("wk_ret") / (pl.col("sigma_fri") * pl.col("bars").sqrt())).alias("wk_z"),
        (pl.col("wk_rv") / (pl.col("sigma_fri") * pl.col("bars").sqrt())).alias("rv_ratio"),
        (pl.col("wk_turn") / pl.col("turn_base")).alias("vol_ratio"),
    )
    # Only weekends that have ended inside the frame (causal: needs a non-weekend bar after them).
    return out.filter(pl.col("last_i") < f.height - 1).sort("end_known_ts")


def weekend_context(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the most recent *completed* weekend's stats to every non-weekend bar (as-of join
    on known_ts), plus `wk_first_bar` (first bar after that weekend) and `wk_absorbed`
    (True from Monday's NYSE close onwards, the end of the TradFi absorption window)."""
    f = add_calendar(frame) if "tradfi_weekend" not in frame.columns else frame
    cols = ["fri_close", "sigma_fri", "wk_ret", "wk_z", "rv_ratio", "vol_ratio", "wk_high", "wk_low", "end_ts", "weekend_id"]
    wt = weekend_table(f)
    if wt.height == 0:
        return f.with_columns([pl.lit(None).alias(f"ctx_{c}") for c in cols] + [pl.lit(False).alias("wk_first_bar"), pl.lit(True).alias("wk_absorbed")])
    ctx = wt.select(pl.col("end_known_ts").alias("_k"), *[pl.col(c).alias(f"ctx_{c}") for c in cols]).sort("_k")
    out = f.sort("known_ts").join_asof(ctx, left_on="known_ts", right_on="_k", strategy="backward").drop("_k").sort("event_ts")
    out = out.with_columns(
        pl.when(pl.col("tradfi_weekend")).then(None).otherwise(pl.col(f"ctx_{c}")).alias(f"ctx_{c}") for c in cols
    )
    first = (pl.col("ctx_weekend_id") != pl.col("ctx_weekend_id").shift(1)).fill_null(pl.col("ctx_weekend_id").is_not_null())
    ny = pl.col("event_ts").dt.convert_time_zone(NY)
    # Absorbed = Tue..Sat, or Monday from 16:00 ET. Sunday evening after the CME reopen is NOT absorbed.
    wd = ny.dt.weekday()
    after_mon_close = wd.is_between(2, 6) | ((wd == 1) & (ny.dt.hour() >= 16))
    return out.with_columns(
        (first & pl.col("ctx_weekend_id").is_not_null()).alias("wk_first_bar"),
        (after_mon_close | pl.col("tradfi_weekend")).alias("wk_absorbed"),
    )


def gap_fill_baseline(distance_sigma: np.ndarray, bars: np.ndarray) -> np.ndarray:
    """Driftless random-walk probability of revisiting a level `distance_sigma` (in per-bar σ)
    away within `bars` bars (reflection principle): 2·(1 − Φ(d/√bars)). The null for CME-gap fills."""
    from scipy.stats import norm

    d = np.abs(np.asarray(distance_sigma, dtype=float))
    return 2.0 * (1.0 - norm.cdf(d / np.sqrt(np.asarray(bars, dtype=float))))
