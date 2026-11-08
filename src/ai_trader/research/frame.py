"""Point-in-time market frame: one row per bar, with every auxiliary series as-of joined on
**known_ts** (what was knowable when the bar closed), never on event_ts.

Row t is "the world at the close of bar t". Columns:
  event_ts, close_ts, known_ts, open, high, low, close, volume, turnover
  ret         log return of bar t (close_{t-1} -> close_t)
  sigma_prev  σ̂_{t|t-1}: EWMA vol forecast for bar t made before it (look-ahead safe)
  z           ret / sigma_prev (vol-standardised return)
  sigma_next  σ̂_{t+1|t}: forecast for the next bar, known at the close of t
  fwd_ret     close_t -> close_{t+1} log return. **Label only**: never use it as a feature
  sigma_rb_next    bias-corrected robust EWMA σ for bar t+1, weekend multiplier applied if t+1 is in the
                   TradFi weekend: for CENTRAL estimates (first passage, TP probabilities). H-V1b, H-W1, H-W7
  sigma_risk_next  the same model WITHOUT the weekend shrink: for STOPS and SIZING (weekends are jump-prone;
                   shrinking σ worsens tail loss, diary 2026-11-07)
  wk_m             weekend vol multiplier known at the close of t
  funding_rate, open_interest (+ oi_chg_1)   latest values known at known_ts (if requested)
  buy_volume, sell_volume, flow_imbalance, cvd   same-bar order flow (if requested and available)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.features.transforms import ewma_sigma_ahead
from ai_trader.storage.lake import Lake

LABEL_COLUMNS = ("fwd_ret",)


def _asof_join(left: pl.DataFrame, right: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    if right.height == 0:
        return left.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in cols])
    r = right.select(pl.col("known_ts").alias("_k"), *cols).sort("_k")
    return left.sort("known_ts").join_asof(r, left_on="known_ts", right_on="_k", strategy="backward").drop("_k")


def enrich_bars(bars: pl.DataFrame, ewma_lambda: float = 0.94) -> pl.DataFrame:
    """Add ret, sigma_prev, sigma_next, z and the fwd_ret label to a sorted bar frame."""
    bars = bars.sort("event_ts")
    close = bars["close"].to_numpy()
    ret = np.r_[np.nan, np.diff(np.log(close))]
    sig_prev = ewma_sigma_ahead(np.nan_to_num(ret, nan=0.0), lam=ewma_lambda, warmup=max(1, min(30, len(ret) - 1)))
    # σ̂_{t+1|t}: one more EWMA step with r_t (known at the close of t).
    sig_next = np.sqrt(ewma_lambda * sig_prev**2 + (1 - ewma_lambda) * np.nan_to_num(ret) ** 2)
    return bars.with_columns(
        pl.Series("ret", ret),
        pl.Series("sigma_prev", sig_prev),
        pl.Series("sigma_next", sig_next),
    ).with_columns(
        (pl.col("ret") / pl.col("sigma_prev")).alias("z"),
        pl.col("ret").shift(-1).alias("fwd_ret"),
    )


def add_robust_vol(df: pl.DataFrame) -> pl.DataFrame:
    """Append sigma_rb_next / sigma_risk_next / wk_m (features.vol_models, validated H-V1b)."""
    from ai_trader.features.vol_models import VolModel, run
    from ai_trader.research.calendar import add_calendar

    if df.height < 40:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in ("sigma_rb_next", "sigma_risk_next", "wk_m"))
    wk = add_calendar(df.select("event_ts"))["tradfi_weekend"].to_numpy()
    nxt_wk = add_calendar(df.select(pl.col("close_ts").alias("event_ts")))["tradfi_weekend"].to_numpy()
    r = np.nan_to_num(df["ret"].to_numpy())
    seasonal = run(r, wk, VolModel(bias_correct=True))                 # central: weekend-aware
    risk = run(r, wk, VolModel(bias_correct=True, use_weekend=False))  # exactly the H-V1b-validated model
    return df.with_columns(
        pl.Series("sigma_rb_next", np.sqrt(seasonal["v_next"]) * np.where(nxt_wk, seasonal["m"], 1.0)),
        pl.Series("sigma_risk_next", np.sqrt(risk["v_next"])),
        pl.Series("wk_m", seasonal["m"]),
    )


def market_frame(
    symbol: str,
    freq: str,
    as_of: datetime,
    *,
    start: datetime | None = None,
    lake: Lake | None = None,
    funding: bool = True,
    oi: bool = True,
    oi_interval: str = "4h",
    flow: bool = False,
    ewma_lambda: float = 0.94,
    venue: str = "bybit",
    burn_in: bool = True,
) -> pl.DataFrame:
    lake = lake or Lake()
    bars = lake.bars(symbol, freq, as_of, start=start, venue=venue).select(
        "event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover"
    )
    # Price sanity (data audit 2026-11-07): impossible bars exist in venue history.
    bars = bars.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close")) & (pl.col("low") <= pl.min_horizontal("open", "close")))
    if burn_in and bars.height:
        # Listing burn-in: the first days after listing are illiquid and extreme. First bar, not launch_ts, marks the listing.
        rs = load().settings["research"]
        first = bars["event_ts"].min()
        days = rs["burn_in_days_new"] if first.year >= rs["burn_in_new_listing_year"] else rs["burn_in_days"]
        bars = bars.filter(pl.col("event_ts") >= first + timedelta(days=days))
    if bars.height == 0:
        return bars
    df = enrich_bars(bars, ewma_lambda)
    df = add_robust_vol(df)

    if funding:
        f = lake.read_asof("funding", as_of, symbols=[symbol], start=start, filters={"venue": venue})
        df = _asof_join(df, f.rename({"rate": "funding_rate"}) if f.height else f, ["funding_rate"])
    if oi:
        o = lake.read_asof("open_interest", as_of, symbols=[symbol], start=start, filters={"venue": venue, "interval": oi_interval})
        df = _asof_join(df, o, ["open_interest"]).with_columns(
            (pl.col("open_interest").log() - pl.col("open_interest").log().shift(1)).alias("oi_chg_1")
        )
    if flow:
        fl = lake.read_asof("flow_bars", as_of, symbols=[symbol], start=start, filters={"venue": venue, "freq": freq})
        if fl.height:
            fl = fl.select("event_ts", "buy_volume", "sell_volume")
            df = df.join(fl, on="event_ts", how="left").with_columns(
                ((pl.col("buy_volume") - pl.col("sell_volume")) / (pl.col("buy_volume") + pl.col("sell_volume"))).alias("flow_imbalance"),
                (pl.col("buy_volume") - pl.col("sell_volume")).fill_null(0).cum_sum().alias("cvd"),
            )
        else:
            df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in ("buy_volume", "sell_volume", "flow_imbalance", "cvd")])
    return df.sort("event_ts")
