"""Point-in-time next-hour volatility panel on 1h bars, universe L (H-NN8, program 2026-11-12 round 2).

Row = 1h bar t of one symbol, on a regular hourly grid (missing bars are NaN, so no return spans a gap). The forecast
for bar t is made at its open (= the close of bar t-1). Everything below uses bars t-1 and earlier only.

  s2      s_pre_t^2 = sigma_risk_next of bar t-1 squared (robust_bc, H-V1b; the H-NS4 base F0)
  fac     features.seasonality.seasonal_factor on u = r^2 / s_pre^2 (expanding, kappa 20, 1 h kernel, cap 50, floor
          0.1, >= 8 weeks), causal: fac[t] uses u[j], j < t (the H-NS4 factor)
  champion F1 = s2 * fac (the grade-A intraday model, R-VOL-26)

Features (HOURLY_FEATURES, 19, fixed before any model was fitted):
  lsig2                 log s2
  lfac                  log fac
  fund_after, fund_before   bar t opens at 00/08/16 UTC (first hour after a Bybit funding settlement) / at 23/07/15
                        (the hour before one). A calendar flag: the settlement schedule of universe-L perps is 8-hourly
  lu_1 lu_2 lu_3 lu_6 lu_12 lu_24   log(max(r_{t-k}^2 / s2_t, 1e-4)): recent |returns| at 1-24 h lags, in units of
                        the current base variance
  lu_m6, lu_m24         log(mean_{k=1..6 / 1..24} r_{t-k}^2 / s2_t) (>= 5 / >= 20 finite)
  jump24                1{ max_{k=1..24} |r_{t-k}| / s_pre_{t-k} > 4 } (recent jump flag, H-S4 shock threshold)
  since_jump            min(hours since the last bar with |r| / s_pre > 4, 168) / 168
  har_d, har_w, har_m   daily HAR level relative to the hourly base: log RV_{D-1} - log(24 s2_t), mean log RV over
                        D-7..D-1 (>= 5) and D-30..D-1 (>= 20), minus log(24 s2_t); D = UTC date of bar t, RV from the
                        24 hourly r^2 of complete days
  dow_sin, dow_cos      day of week of bar t
Target: y = log(max(r_t^2 / s2_t, 1e-6)) (next-hour log squared return, in base units); forecast F = s2 exp(f + c),
c = the validation-block QLIKE-optimal constant (models.bias_const).
Segment: llm_in_sample if bar t opens before the knowledge cutoff, else clean_holdout (the H-NS4 convention).
"""

from __future__ import annotations

import numpy as np
import polars as pl

HOURLY_FEATURES = (
    "lsig2", "lfac", "fund_after", "fund_before",
    "lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24", "lu_m6", "lu_m24",
    "jump24", "since_jump", "har_d", "har_w", "har_m", "dow_sin", "dow_cos",
)
assert len(HOURLY_FEATURES) == 19
HOURLY_GROUPS = {
    "base_sigma": ("lsig2",), "seasonal": ("lfac",), "funding_hour": ("fund_after", "fund_before"),
    "recent_returns": ("lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24", "lu_m6", "lu_m24"),
    "jump": ("jump24", "since_jump"), "daily_har": ("har_d", "har_w", "har_m"), "weekday": ("dow_sin", "dow_cos"),
}
assert sorted(sum(HOURLY_GROUPS.values(), ())) == sorted(HOURLY_FEATURES)
JUMP_Z = 4.0


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _trail_mean_lagged(x: np.ndarray, w: int, min_n: int) -> np.ndarray:
    """mean of x[t-w..t-1] (finite values, >= min_n of them)."""
    s = pl.Series(_shift(np.asarray(x, dtype=float), 1)).fill_nan(None)
    return s.rolling_mean(w, min_samples=min_n).to_numpy().astype(float)


def symbol_hourly(f: pl.DataFrame) -> pl.DataFrame:
    """f: market_frame-style 1h frame (event_ts, ret, sigma_risk_next), one symbol. Returns the feature rows."""
    from ai_trader.features import seasonality as sz

    f = f.sort("event_ts").unique("event_ts", keep="last", maintain_order=True).select("event_ts", "ret", "sigma_risk_next")
    grid = pl.DataFrame({"event_ts": pl.datetime_range(f["event_ts"].min(), f["event_ts"].max(), "1h", eager=True)}).with_columns(
        pl.col("event_ts").cast(f["event_ts"].dtype))
    g = grid.join(f, on="event_ts", how="left")
    # a return is only kept if the previous bar exists (market_frame returns span gaps)
    prev_ok = g["ret"].is_not_null().to_numpy() & np.r_[False, g["sigma_risk_next"].is_not_null().to_numpy()[:-1]]
    r = np.where(prev_ok, g["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(), np.nan)
    sig_next = g["sigma_risk_next"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    s_pre = _shift(sig_next, 1)
    s2 = s_pre * s_pre
    with np.errstate(all="ignore"):
        u = r * r / s2
    how = sz.hour_of_week(g["event_ts"])
    fac = sz.seasonal_factor(how, u)
    with np.errstate(all="ignore"):
        z_abs = np.abs(r) / s_pre
        r2 = r * r
        feats: dict[str, np.ndarray] = {"lsig2": np.log(s2), "lfac": np.log(fac)}
        hour = g["event_ts"].dt.hour().to_numpy()
        feats["fund_after"] = np.isin(hour, (0, 8, 16)).astype(float)
        feats["fund_before"] = np.isin(hour, (23, 7, 15)).astype(float)
        for k in (1, 2, 3, 6, 12, 24):
            feats[f"lu_{k}"] = np.log(np.maximum(_shift(r2, k) / s2, 1e-4))
        feats["lu_m6"] = np.log(_trail_mean_lagged(r2, 6, 5) / s2)
        feats["lu_m24"] = np.log(_trail_mean_lagged(r2, 24, 20) / s2)
        jump = np.where(np.isfinite(z_abs), (z_abs > JUMP_Z).astype(float), np.nan)
        j24 = pl.Series(_shift(jump, 1)).fill_nan(None).rolling_max(24, min_samples=20).to_numpy().astype(float)
        feats["jump24"] = j24
        # hours since the last jump bar (bars t-1 and earlier); 168 cap; NaN before the first 168 bars
        last = -10**9
        since = np.full(len(r), np.nan)
        for t in range(len(r)):
            if t > 0 and jump[t - 1] == 1.0:
                last = t - 1
            since[t] = min(t - last, 168) / 168.0 if t >= 168 else np.nan
        feats["since_jump"] = since
    # daily HAR level from complete days strictly before the bar's day
    dates = g["event_ts"].dt.date()
    dd = pl.DataFrame({"date": dates, "r2": r2}).group_by("date", maintain_order=True).agg(
        pl.col("r2").sum().alias("rv"), pl.col("r2").is_finite().sum().alias("n"))
    dd = dd.with_columns(pl.when(pl.col("n") == 24).then(pl.col("rv").log()).otherwise(None).alias("lrv")).sort("date")
    cal = pl.DataFrame({"date": pl.date_range(dd["date"].min(), dd["date"].max(), "1d", eager=True)}).join(dd.select("date", "lrv"), on="date", how="left")
    lrv = cal["lrv"].fill_nan(None)
    cal = cal.with_columns(lrv.shift(1).alias("ld"), lrv.shift(1).rolling_mean(7, min_samples=5).alias("lw"),
                           lrv.shift(1).rolling_mean(30, min_samples=20).alias("lm"))
    m = pl.DataFrame({"date": dates}).join(cal.select("date", "ld", "lw", "lm"), on="date", how="left")
    with np.errstate(all="ignore"):
        base24 = np.log(24 * s2)
        for c_, k in (("ld", "har_d"), ("lw", "har_w"), ("lm", "har_m")):
            feats[k] = m[c_].cast(pl.Float64).fill_null(np.nan).to_numpy() - base24
    wd = g["event_ts"].dt.weekday().to_numpy().astype(float)
    feats["dow_sin"] = np.sin(2 * np.pi * (wd - 1) / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * (wd - 1) / 7)
    with np.errstate(all="ignore"):
        y = np.log(np.maximum(r2 / s2, 1e-6))
    out = g.select("event_ts").with_columns(
        *[pl.Series(k, np.where(np.isfinite(v), v, np.nan)) for k, v in feats.items()],
        pl.Series("r2", r2), pl.Series("s2", s2), pl.Series("fac", fac), pl.Series("y", np.where(np.isfinite(y), y, np.nan)))
    return out.with_columns(pl.col(pl.Float64).fill_nan(None))


def complete_hourly(p: pl.DataFrame) -> pl.DataFrame:
    cols = [*HOURLY_FEATURES, "y", "r2", "s2", "fac"]
    return p.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cols]) & (pl.col("s2") > 0))


def qlike_diff(r2: np.ndarray, Fa: np.ndarray, Fb: np.ndarray) -> np.ndarray:
    """QLIKE(Fa) - QLIKE(Fb) = r^2 (1/Fa - 1/Fb) + log(Fa/Fb) (the log r^2 term cancels, so r = 0 is fine)."""
    with np.errstate(all="ignore"):
        d = r2 * (1 / Fa - 1 / Fb) + np.log(Fa / Fb)
    return np.where(np.isfinite(d), d, np.nan)


def train_subsample(n: int, frac: float = 1 / 3, seed: int = 20261112) -> np.ndarray:
    """Deterministic random subsample of training rows (a fixed every-k-th-hour rule would alias with the 24 h cycle)."""
    return np.random.default_rng(seed).random(n) < frac


__all__ = ["HOURLY_FEATURES", "HOURLY_GROUPS", "complete_hourly", "qlike_diff", "symbol_hourly", "train_subsample"]
