"""Volatility break of structure (vol BoS), directional semivariance breaks and fractal price structure.

Research features for H-VBOS1-4 (research/hypotheses.yaml, block "Volatility break of structure"; plan
research/plans/2026-11-13_vol_bos_trend.md; design fixed in the diary 2026-11-13 before testing).

Every function is causal: the value at bar t uses bars <= t only (thresholds use bars < t), so appending
future bars never changes a past flag (tests/test_vol_bos.py). Windows are given in DAYS and converted with
`bpd` (bars per day: 1 for 1d, 6 for 4h). Thresholds are trailing QUANTILES of the series' own history, so
the same hyperparameter means the same thing on every asset.

Inputs per bar (one symbol, sorted, contiguous-ish):
  rv, rs_pos, rs_neg   intrabar realised variance and semivariances from 1h log returns (see `bar_realized`)
  high, low, close     for the fractal structure

Detector families (fixed grids, `detector_grid`):
  range   L_t = log mean rv_adj over the last w days > trailing q-quantile of L over the previous N days
          (q = 1.0 is the literal range break: above the trailing max)
  cusum   one-sided CUSUM on standardised y = log rv_adj (trailing mean / sd over N days), drift kappa,
          alarm at S > h, reset at an alarm
  har     rv_adj above k x the causal log-HAR level forecast for m consecutive bars
  vov     sd of 1-day changes in log 1-day RV over wv days > its trailing q-quantile over N days
Each returns ONSET flags (first bar of an episode / alarm). `decluster` keeps onsets >= gap bars apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

EPS = 1e-12


# ------------------------------------------------------------------ realised measures per bar
def bar_realized(h1: pl.DataFrame, bar_ts: np.ndarray, bar_hours: int) -> dict[str, np.ndarray]:
    """rv, rs_pos, rs_neg, n per target bar from 1h bars (columns event_ts, close; sorted).

    The 1h return r_i = ln C_i / C_{i-1} belongs to the target bar containing the 1h bar i (floor of its
    event_ts to the target bar size), so the first return of a bar spans the previous bar's last close,
    exactly as features.realized.daily_measures does. Sums are rescaled by expected / observed count."""
    h = h1.sort("event_ts").select(
        pl.col("event_ts").dt.truncate(f"{bar_hours}h").alias("bt"),
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("r"),
    ).drop_nulls("r")
    g = h.group_by("bt").agg(
        pl.len().alias("n"),
        (pl.col("r") ** 2).sum().alias("rv"),
        (pl.when(pl.col("r") > 0).then(pl.col("r") ** 2).otherwise(0.0)).sum().alias("rs_pos"),
        (pl.when(pl.col("r") < 0).then(pl.col("r") ** 2).otherwise(0.0)).sum().alias("rs_neg"),
    )
    tgt = pl.DataFrame({"bt": pl.Series(bar_ts).cast(pl.Datetime("us", "UTC"))})
    j = tgt.join(g.with_columns(pl.col("bt").cast(pl.Datetime("us", "UTC"))), on="bt", how="left")
    n = j["n"].fill_null(0).to_numpy().astype(float)
    scale = np.where(n > 0, bar_hours / np.maximum(n, 1), np.nan)
    return {k: j[k].to_numpy().astype(float) * scale for k in ("rv", "rs_pos", "rs_neg")} | {"n": n}


# ------------------------------------------------------------------ causal rolling helpers
def _s(x: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float))


def rolling_mean(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    """Mean of x[t-n+1..t] (includes t)."""
    return _s(x).rolling(n, min_periods=min_periods or n).mean().to_numpy()


def rolling_std(x: np.ndarray, n: int, min_periods: int | None = None) -> np.ndarray:
    return _s(x).rolling(n, min_periods=min_periods or n).std(ddof=1).to_numpy()


def trailing_quantile(x: np.ndarray, n: int, q: float) -> np.ndarray:
    """q-quantile of x[t-n..t-1] (t EXCLUDED), full window required; q = 1 is the trailing max."""
    s = _s(x)
    r = s.rolling(n, min_periods=n).max() if q >= 1.0 else s.rolling(n, min_periods=n).quantile(q, interpolation="linear")
    return r.shift(1).to_numpy()


def trailing_mean_sd(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of x[t-n..t-1] (t excluded); half a window required."""
    s = _s(x)
    mp = max(2, n // 2)
    return s.rolling(n, min_periods=mp).mean().shift(1).to_numpy(), s.rolling(n, min_periods=mp).std(ddof=1).shift(1).to_numpy()


def onsets(flag: np.ndarray) -> np.ndarray:
    """First bar of each run of True."""
    f = np.asarray(flag, dtype=bool)
    return f & ~np.r_[False, f[:-1]]


def decluster(onset: np.ndarray, gap: int) -> np.ndarray:
    """Keep an onset only if >= gap bars after the previously KEPT onset."""
    idx = np.flatnonzero(onset)
    out = np.zeros(len(onset), dtype=bool)
    last = -10**12
    for i in idx:
        if i - last >= gap:
            out[i] = True
            last = i
    return out


def bars_since(onset: np.ndarray) -> np.ndarray:
    """Bars since the most recent onset at or before t (inf if none)."""
    n = len(onset)
    idx = np.where(onset, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, np.arange(n) - last, np.inf)


# ------------------------------------------------------------------ seasonality (4h)
def seasonal_factor(rv: np.ndarray, slot: np.ndarray, n_slots: int, days: int = 60) -> np.ndarray:
    """Causal intraday seasonal factor: trailing mean of rv in the bar's slot over the last `days` days
    divided by the trailing mean over all bars of those days (both exclude bar t). 1 for daily bars."""
    if n_slots <= 1:
        return np.ones(len(rv))
    x = _s(rv)
    allm = x.rolling(days * n_slots, min_periods=days * n_slots // 2).mean().shift(1).to_numpy()
    f = np.full(len(rv), np.nan)
    for s in range(n_slots):
        ix = np.flatnonzero(slot == s)
        if len(ix):
            f[ix] = _s(rv[ix]).rolling(days, min_periods=days // 2).mean().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = f / allm
    return np.where(np.isfinite(fac) & (fac > 0), fac, 1.0)


def log_rv(rv: np.ndarray) -> np.ndarray:
    v = np.asarray(rv, dtype=float)
    return np.where(np.isfinite(v), np.log(np.maximum(v, EPS)), np.nan)


# ------------------------------------------------------------------ detector families
def range_break(rv_adj: np.ndarray, bpd: int, w: int, n_days: int, q: float) -> np.ndarray:
    L = log_rv(rolling_mean(rv_adj, w * bpd))
    thr = trailing_quantile(L, n_days * bpd, q)
    return onsets(np.isfinite(thr) & (L > thr))


def cusum(y: np.ndarray, bpd: int, n_days: int, kappa: float, h: float) -> np.ndarray:
    m, sd = trailing_mean_sd(y, n_days * bpd)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (y - m) / sd
    out = np.zeros(len(y), dtype=bool)
    s = 0.0
    for t, xt in enumerate(x):
        if not np.isfinite(xt):
            continue
        s = max(0.0, s + xt - kappa)
        if s > h:
            out[t] = True
            s = 0.0
    return out


def har_level_forecast(y: np.ndarray, bpd: int, refit_days: int = 30, min_days: int = 180) -> np.ndarray:
    """F_t: level forecast of rv_adj for bar t made at the close of t-1 (log-HAR, expanding OLS, causal).

    Regressors at origin s: y_s, mean y over 5 days, mean y over 22 days; target y_{s+1}. Fitted with
    features.vol_panel.per_asset_forecast (rows whose target is known at the refit time only)."""
    from ai_trader.features.vol_panel import per_asset_forecast

    yy = np.asarray(y, dtype=float)
    X = np.column_stack([yy, rolling_mean(yy, 5 * bpd), rolling_mean(yy, 22 * bpd)])
    tgt = np.r_[yy[1:], np.nan]
    f = per_asset_forecast(X, tgt, 1, log=True, refit_every=refit_days * bpd, min_days=min_days * bpd, min_obs=60)
    return np.r_[np.nan, f[:-1]]


def har_surprise(rv_adj: np.ndarray, F: np.ndarray, k: float, m: int) -> np.ndarray:
    cond = np.isfinite(F) & (rv_adj > k * F)
    run = np.zeros(len(cond), dtype=int)
    c = 0
    for t, v in enumerate(cond):
        c = c + 1 if v else 0
        run[t] = c
    return run == m


def vov_break(y: np.ndarray, bpd: int, wv: int, n_days: int, q: float, rv_adj: np.ndarray | None = None) -> np.ndarray:
    """y unused when rv_adj given (log 1-day RV is rebuilt from rv_adj)."""
    l1 = log_rv(rolling_mean(rv_adj, bpd)) if rv_adj is not None else y
    d1 = l1 - np.r_[np.full(bpd, np.nan), l1[:-bpd]]
    vov = rolling_std(d1, wv * bpd)
    thr = trailing_quantile(vov, n_days * bpd, q)
    return onsets(np.isfinite(thr) & (vov > thr))


@dataclass(frozen=True)
class Detector:
    family: str
    params: tuple

    @property
    def name(self) -> str:
        return self.family + "_" + "_".join(str(p) for p in self.params)


RANGE_W, RANGE_N, RANGE_Q = (5, 10, 20), (60, 120, 180), (0.90, 0.95, 1.00)
CUSUM_N, CUSUM_K, CUSUM_H = (60, 180), (0.25, 0.5, 1.0), (2.0, 4.0, 8.0)
HAR_K = (1.25, 1.5, 2.0)
HAR_M = {1: (2, 3, 5), 6: (3, 6, 12)}
VOV_W, VOV_Q, VOV_N = (10, 20), (0.90, 0.95), (60, 180)
SEMI_W, SEMI_N, SEMI_Q = (5, 10, 20), (60, 120, 180), (0.80, 0.90, 0.95)
FAMILIES = ("range", "cusum", "har", "vov")


def detector_grid(bpd: int) -> list[Detector]:
    g = [Detector("range", (w, n, q)) for w in RANGE_W for n in RANGE_N for q in RANGE_Q]
    g += [Detector("cusum", (n, k, h)) for n in CUSUM_N for k in CUSUM_K for h in CUSUM_H]
    g += [Detector("har", (k, m)) for k in HAR_K for m in HAR_M[bpd]]
    g += [Detector("vov", (w, q, n)) for w in VOV_W for q in VOV_Q for n in VOV_N]
    return g


def semi_grid() -> list[tuple[int, int, float]]:
    return [(w, n, q) for w in SEMI_W for n in SEMI_N for q in SEMI_Q]


def detector_onsets(rv_adj: np.ndarray, bpd: int, grid: list[Detector] | None = None) -> np.ndarray:
    """(len(grid), n) bool matrix of onset flags (not declustered)."""
    grid = grid or detector_grid(bpd)
    y = log_rv(rv_adj)
    F = har_level_forecast(y, bpd) if any(d.family == "har" for d in grid) else None
    out = np.zeros((len(grid), len(rv_adj)), dtype=bool)
    for i, d in enumerate(grid):
        if d.family == "range":
            out[i] = range_break(rv_adj, bpd, *d.params)
        elif d.family == "cusum":
            out[i] = cusum(y, bpd, *d.params)
        elif d.family == "har":
            out[i] = har_surprise(rv_adj, F, *d.params)
        elif d.family == "vov":
            w, q, n = d.params
            out[i] = vov_break(y, bpd, w, n, q, rv_adj=rv_adj)
    return out


# ------------------------------------------------------------------ semivariance breaks
def semi_breaks(rs_pos: np.ndarray, rs_neg: np.ndarray, bpd: int, w: int, n_days: int, q: float) -> np.ndarray:
    """Signed onset: +1 when upside semivariance breaks above its own trailing q-quantile while downside does
    not (first bar of the episode), -1 symmetric, 0 otherwise."""
    lp = log_rv(rolling_mean(rs_pos, w * bpd))
    ln = log_rv(rolling_mean(rs_neg, w * bpd))
    tp = trailing_quantile(lp, n_days * bpd, q)
    tn = trailing_quantile(ln, n_days * bpd, q)
    ok = np.isfinite(tp) & np.isfinite(tn)
    up = ok & (lp > tp) & ~(ln > tn)
    dn = ok & (ln > tn) & ~(lp > tp)
    return onsets(up).astype(np.int8) - onsets(dn).astype(np.int8)


def semi_dominance(rs_pos: np.ndarray, rs_neg: np.ndarray, bpd: int, days: int = 10) -> np.ndarray:
    """sign(sum rs_pos - sum rs_neg) over the last `days` days (0 while not warm)."""
    d = rolling_mean(rs_pos, days * bpd) - rolling_mean(rs_neg, days * bpd)
    return np.where(np.isfinite(d), np.sign(d), 0.0).astype(np.int8)


# ------------------------------------------------------------------ fractal structure
def fractal_pivots(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Swing levels by CONFIRMATION bar: hi_conf[t] = high_i if i = t - k is a swing high (high_i > max of the
    k previous highs and >= max of the k next), else nan; lows symmetric. Known at the close of t."""
    h, lo = _s(high), _s(low)
    prev_max = h.rolling(k, min_periods=k).max().shift(1).to_numpy()
    next_max = h[::-1].rolling(k, min_periods=k).max()[::-1].shift(-1).to_numpy()
    prev_min = lo.rolling(k, min_periods=k).min().shift(1).to_numpy()
    next_min = lo[::-1].rolling(k, min_periods=k).min()[::-1].shift(-1).to_numpy()
    H, L = np.asarray(high, float), np.asarray(low, float)
    is_hi = (H > prev_max) & (H >= next_max)
    is_lo = (L < prev_min) & (L <= next_min)
    n = len(H)
    hi_conf, lo_conf = np.full(n, np.nan), np.full(n, np.nan)
    ih, il = np.flatnonzero(is_hi), np.flatnonzero(is_lo)
    hi_conf[ih[ih + k < n] + k] = H[ih[ih + k < n]]
    lo_conf[il[il + k < n] + k] = L[il[il + k < n]]
    return hi_conf, lo_conf


def structure(close: np.ndarray, high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Break of structure and change of character.

    bos[t] = +1 if close_t is above the most recent confirmed swing high that has not been broken yet (each
    level is consumed once), -1 for the swing low, 0 otherwise. choch[t] = True if bos[t] is against the
    direction of the previous BoS."""
    hi_conf, lo_conf = fractal_pivots(high, low, k)
    c = np.asarray(close, float)
    n = len(c)
    bos = np.zeros(n, dtype=np.int8)
    choch = np.zeros(n, dtype=bool)
    cur_hi = cur_lo = np.nan
    prev = 0
    for t in range(n):
        if np.isfinite(hi_conf[t]):
            cur_hi = hi_conf[t]
        if np.isfinite(lo_conf[t]):
            cur_lo = lo_conf[t]
        d = 0
        if np.isfinite(cur_hi) and c[t] > cur_hi:
            d, cur_hi = 1, np.nan
        elif np.isfinite(cur_lo) and c[t] < cur_lo:
            d, cur_lo = -1, np.nan
        if d:
            bos[t] = d
            choch[t] = prev != 0 and d != prev
            prev = d
    return bos, choch


__all__ = ["Detector", "FAMILIES", "bar_realized", "bars_since", "cusum", "decluster", "detector_grid", "detector_onsets",
           "fractal_pivots", "har_level_forecast", "har_surprise", "onsets", "range_break", "seasonal_factor", "semi_breaks",
           "semi_dominance", "semi_grid", "structure", "trailing_quantile", "vov_break"]
