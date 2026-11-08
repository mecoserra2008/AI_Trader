"""Causal daily trend-vs-mean-reversion regime features from 4h bars (research/hypotheses.yaml, block
"Trend vs mean-reversion regimes + trend model zoo"; used by H-TR3 / H-TR4 and the regime.trend_vs_mr tool).

One row per UTC day D. Every feature of day D uses only 4h bars whose event_ts falls on or before D, so the row is
known at the close of the 20:00 bar of D (00:00 UTC of D+1, `known_ts`). Bars are placed on a regular 4h grid;
a missing bar gives a missing return (never a return that spans the gap). A rolling feature is NaN when fewer than
MIN_COVER of its window's returns are present, or when the last bar of D is missing.

Definitions (n = window in 4h returns: 7d = 42, 14d = 84, 30d = 180):
  er_{7,14,30}d     efficiency ratio |sum r| / sum |r|               (1 on a straight line, ~0 in chop)
  dvr_{7,30}d       directional-vol ratio (RS+ - RS-) / (RS+ + RS-), RS+- = sum r^2 over up / down 4h bars
  tstat_{7,30}d     trend t-stat mean(r) / sd(r) * sqrt(n)
  vr_30d            Lo-MacKinlay VR(q = 6) (1d vs 4h) on the 180 4h returns, unbiased variances, overlapping sums
  ac1_4h_30d        lag-1 autocorrelation of the 180 4h returns
  ac1_1d_30d        lag-1 autocorrelation of the last 30 daily returns (close of D vs close of D-1)
  vol_pct_1y        percentile of today's robust vol (sigma_risk_next at the last bar of D, H-V1b model) among the
                    trailing 365 daily values incl. today (ties count half; needs >= MIN_PCT_DAYS values)
  vol_of_vol_30d    sd of log daily realised vol (sqrt of the day's summed squared 4h returns) over 30 days
(er_14d is not one of the H-TR3 model inputs; it is the H-TR4 single feature.)
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

BARS_PER_DAY = 6
WINDOWS = {"7d": 7 * BARS_PER_DAY, "14d": 14 * BARS_PER_DAY, "30d": 30 * BARS_PER_DAY}
MIN_COVER = 0.9
PCT_DAYS = 365
MIN_PCT_DAYS = 180
VOV_DAYS = 30
AC1D_DAYS = 30

FEATURES = (
    "er_7d", "er_30d", "dvr_7d", "dvr_30d", "tstat_7d", "tstat_30d", "vr_30d",
    "ac1_4h_30d", "ac1_1d_30d", "vol_pct_1y", "vol_of_vol_30d",
)
ALL_COLUMNS = ("er_14d", *FEATURES)


def _grid(frame: pl.DataFrame) -> pl.DataFrame:
    """Reindex onto a regular 4h event_ts grid (missing bars -> null close) and recompute returns on the grid."""
    f = frame.sort("event_ts").unique("event_ts", keep="last", maintain_order=True)
    cols = ["event_ts", "close"] + [c for c in ("sigma_risk_next", "known_ts") if c in f.columns]
    f = f.select(cols)
    t0, t1 = f["event_ts"].min(), f["event_ts"].max()
    g = pl.DataFrame({"event_ts": pl.datetime_range(t0, t1, interval="4h", eager=True)}).with_columns(
        pl.col("event_ts").cast(f["event_ts"].dtype)
    )
    g = g.join(f, on="event_ts", how="left")
    return g.with_columns((pl.col("close").log() - pl.col("close").log().shift(1)).alias("r"))


def variance_ratio(r: np.ndarray, q: int) -> float:
    """Lo-MacKinlay VR(q), unbiased variances, overlapping q-sums (same estimator as the corrected H-RG1 one)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n <= q + 1:
        return float("nan")
    mu = r.mean()
    v1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    m = q * (n - q + 1) * (1 - q / n)
    vq = ((rq - q * mu) ** 2).sum() / m
    return float(vq / v1) if v1 > 0 else float("nan")


def _ac1(x: np.ndarray) -> float:
    a, b = x[1:], x[:-1]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    a, b = a[ok], b[ok]
    sa, sb = a.std(), b.std()
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def _window_feats(w: np.ndarray, out: dict[str, float], tag: str, full: bool) -> None:
    ok = np.isfinite(w)
    if ok.mean() < MIN_COVER:
        return
    x = w[ok]
    s_abs = np.abs(x).sum()
    out[f"er_{tag}"] = float(abs(x.sum()) / s_abs) if s_abs > 0 else float("nan")
    if not full:
        return
    up, dn = (x[x > 0] ** 2).sum(), (x[x < 0] ** 2).sum()
    out[f"dvr_{tag}"] = float((up - dn) / (up + dn)) if up + dn > 0 else float("nan")
    sd = x.std(ddof=1)
    out[f"tstat_{tag}"] = float(x.mean() / sd * np.sqrt(len(x))) if sd > 0 else float("nan")


def regime_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Daily features from a 4h bar frame (needs event_ts, close; sigma_risk_next for vol_pct_1y; known_ts optional).

    Returns date, known_ts, close, sigma_risk and ALL_COLUMNS; one row per UTC day with a bar at 20:00."""
    g = _grid(frame)
    ts = g["event_ts"]
    r = g["r"].to_numpy().astype(float)
    close = g["close"].to_numpy().astype(float)
    sig = g["sigma_risk_next"].to_numpy().astype(float) if "sigma_risk_next" in g.columns else np.full(len(r), np.nan)
    hour = ts.dt.hour().to_numpy()
    day = ts.dt.date().to_list()
    last = np.flatnonzero((hour == 20) & np.isfinite(close))  # the last bar of each day, present

    # daily series indexed by calendar day (for percentile, vol-of-vol, daily autocorrelation)
    di = (ts.dt.date().to_numpy() - ts.dt.date().to_numpy()[0]).astype("timedelta64[D]").astype(np.int64)
    nd = int(di[-1]) + 1
    d_close = np.full(nd, np.nan)
    d_close[di[last]] = close[last]
    d_sig = np.full(nd, np.nan)
    d_sig[di[last]] = sig[last]
    r2 = np.where(np.isfinite(r), r * r, 0.0)
    cnt = np.zeros(nd)
    np.add.at(cnt, di, np.isfinite(r).astype(float))
    ssq = np.zeros(nd)
    np.add.at(ssq, di, r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_lrv = np.where(cnt >= BARS_PER_DAY - 1, 0.5 * np.log(ssq * BARS_PER_DAY / np.maximum(cnt, 1)), np.nan)
        d_lrv[~np.isfinite(d_lrv)] = np.nan
        d_ret = np.r_[np.nan, np.diff(np.log(d_close))]

    rows = []
    kts = g["known_ts"].to_list() if "known_ts" in g.columns else [None] * len(r)
    for e in last:
        k = int(di[e])
        out: dict[str, float] = {c: float("nan") for c in ALL_COLUMNS}
        for tag, n in WINDOWS.items():
            if e + 1 >= n:
                _window_feats(r[e - n + 1:e + 1], out, tag, full=tag != "14d")
        n30 = WINDOWS["30d"]
        if e + 1 >= n30:
            w = r[e - n30 + 1:e + 1]
            if np.isfinite(w).mean() >= MIN_COVER:
                out["ac1_4h_30d"] = _ac1(w)
                out["vr_30d"] = variance_ratio(w[np.isfinite(w)], BARS_PER_DAY)
        if k + 1 >= AC1D_DAYS + 1:
            wd = d_ret[k - AC1D_DAYS + 1:k + 1]
            if np.isfinite(wd).mean() >= MIN_COVER:
                out["ac1_1d_30d"] = _ac1(wd)
        if k + 1 >= VOV_DAYS:
            wv = d_lrv[k - VOV_DAYS + 1:k + 1]
            wv = wv[np.isfinite(wv)]
            if len(wv) >= MIN_COVER * VOV_DAYS:
                out["vol_of_vol_30d"] = float(wv.std(ddof=1))
        if np.isfinite(d_sig[k]):
            h = d_sig[max(0, k - PCT_DAYS + 1):k + 1]
            h = h[np.isfinite(h)]
            if len(h) >= MIN_PCT_DAYS:
                out["vol_pct_1y"] = float(((h < d_sig[k]).sum() + 0.5 * (h == d_sig[k]).sum()) / len(h))
        known = kts[e] if kts[e] is not None else ts[int(e)] + timedelta(hours=4)
        rows.append({"date": day[e], "known_ts": known, "close": close[e], "sigma_risk": sig[e], **out})
    schema = {"date": pl.Date, "known_ts": ts.dtype, "close": pl.Float64, "sigma_risk": pl.Float64, **{c: pl.Float64 for c in ALL_COLUMNS}}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
