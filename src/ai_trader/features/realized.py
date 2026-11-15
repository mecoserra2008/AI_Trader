"""Daily realised measures from intraday bars: one shared, point-in-time definition for every volatility study.

For a UTC day d with intraday log returns r_1..r_n (r_i = ln C_i / C_{i-1}, C_0 = the previous day's last close):

  rv      sum r_i^2                                   realised variance
  rs_neg  sum r_i^2 1{r_i < 0}, rs_pos (r_i > 0)       semivariances (Barndorff-Nielsen, Kinnebrock & Shephard 2010)
  rq      n/3 * sum r_i^4                             realised quarticity (HAR-Q, Bollerslev, Patton & Quaedvlieg 2016)
  bv      pi/2 * sum_{i>=2} |r_i||r_{i-1}|             bipower variation (jump-robust)
  jv      max(rv - bv, 0)                             jump variation
  pk      sum (ln H_i/L_i)^2 / (4 ln 2)               Parkinson range variance, summed over bars
  gk      sum 0.5 (ln H_i/L_i)^2 - (2 ln 2 - 1)(ln C_i/O_i)^2   Garman-Klass, summed over bars
  ret     ln C_n / C_0                                daily close-to-close log return

Only complete days (n == bars per day) are kept. `known_ts` is the end of day d (d + 1 00:00 UTC): a day's measures
may be used for decisions at or after that time only. Bars come from research.frame.market_frame (price sanity and
listing burn-in applied).

    from ai_trader.features.realized import daily_panel
    panel = daily_panel(["BTCUSDT", "ETHUSDT"], as_of=now())
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

BARS_PER_DAY = {"1h": 24, "4h": 6, "15m": 96, "5m": 288}
LN2 = float(np.log(2.0))


def daily_measures(bars: pl.DataFrame, freq: str = "1h") -> pl.DataFrame:
    """bars: event_ts, open, high, low, close (one symbol, sorted). Returns one row per complete UTC day."""
    n_need = BARS_PER_DAY[freq]
    b = bars.sort("event_ts").with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("r"),
        (pl.col("high").log() - pl.col("low").log()).alias("hl"),
        (pl.col("close").log() - pl.col("open").log()).alias("co"),
        pl.col("event_ts").dt.date().alias("date"),
    ).drop_nulls("r")
    b = b.with_columns((pl.col("r").abs() * pl.col("r").abs().shift(1).over("date")).alias("bp"))
    d = b.group_by("date").agg(
        pl.len().alias("n_bars"),
        pl.col("r").sum().alias("ret"),
        (pl.col("r") ** 2).sum().alias("rv"),
        (pl.when(pl.col("r") < 0).then(pl.col("r") ** 2).otherwise(0.0)).sum().alias("rs_neg"),
        (pl.when(pl.col("r") > 0).then(pl.col("r") ** 2).otherwise(0.0)).sum().alias("rs_pos"),
        (pl.col("r") ** 4).sum().alias("r4"),
        pl.col("bp").sum().alias("bp_sum"),
        ((pl.col("hl") ** 2) / (4 * LN2)).sum().alias("pk"),
        (0.5 * pl.col("hl") ** 2 - (2 * LN2 - 1) * pl.col("co") ** 2).sum().alias("gk"),
        pl.col("close").last().alias("close"),
    )
    d = d.filter(pl.col("n_bars") == n_need).with_columns(
        (pl.col("n_bars") / 3.0 * pl.col("r4")).alias("rq"),
        (np.pi / 2 * pl.col("bp_sum")).alias("bv"),
    ).with_columns(
        (pl.col("rv") - pl.col("bv")).clip(lower_bound=0.0).alias("jv"),
        (pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias("known_ts"),
    )
    return d.select("date", "known_ts", "n_bars", "ret", "rv", "rs_neg", "rs_pos", "rq", "bv", "jv", "pk", "gk", "close").sort("date")


def daily_panel(symbols: list[str], as_of: datetime, freq: str = "1h", start: datetime | None = None) -> pl.DataFrame:
    """Long panel (symbol, date, ...) of daily realised measures, every row known at or before `as_of`."""
    from ai_trader.research.frame import market_frame

    out = []
    for s in symbols:
        f = market_frame(s, freq, as_of, start=start, funding=False, oi=False)
        if f.height == 0:
            continue
        d = daily_measures(f.select("event_ts", "open", "high", "low", "close"), freq)
        d = d.filter(pl.col("known_ts") <= as_of)
        if d.height:
            out.append(d.with_columns(pl.lit(s).alias("symbol")))
    if not out:
        return pl.DataFrame()
    return pl.concat(out).select("symbol", pl.all().exclude("symbol")).sort(["symbol", "date"])


def har_lags(panel: pl.DataFrame, col: str = "rv") -> pl.DataFrame:
    """Adds the HAR regressors (Corsi 2009) per symbol: col_d (day t), col_w (mean t-4..t), col_m (mean t-21..t), all
    known at the end of day t, plus targets col_next (day t+1) and col_next5 (mean of t+1..t+5)."""
    g = pl.col(col)
    return panel.sort(["symbol", "date"]).with_columns(
        g.alias(f"{col}_d"),
        g.rolling_mean(5).over("symbol").alias(f"{col}_w"),
        g.rolling_mean(22).over("symbol").alias(f"{col}_m"),
        g.shift(-1).over("symbol").alias(f"{col}_next"),
        g.shift(-1).rolling_mean(5).shift(-4).over("symbol").alias(f"{col}_next5"),
    )


def known_before(panel: pl.DataFrame, t: datetime) -> pl.DataFrame:
    """Rows usable for a decision at time t (point-in-time guard)."""
    return panel.filter(pl.col("known_ts") <= t)


__all__ = ["BARS_PER_DAY", "daily_measures", "daily_panel", "har_lags", "known_before", "timedelta"]
