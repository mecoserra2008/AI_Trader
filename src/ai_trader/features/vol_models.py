"""Seasonal, jump-robust EWMA volatility (H-S4, H-W1/H-W7; validated by H-V1/H-V2).

One causal recursion per symbol:

    s_t      = m if bar t is in the TradFi weekend window else 1        (seasonal scale)
    σ̂_t|t-1  = sqrt(v) · s_t                                            (forecast for bar t)
    x_t      = clip(r_t / s_t, ±c·sqrt(v))                              (deseasonalise, then winsorise jumps)
    v        = λ v + (1 − λ) x_t²                                       (weekday-equivalent base variance)

m is re-estimated each time a weekend completes, from that weekend's realised variance relative to
the base variance at its start, over a trailing window of weekends, and shrunk toward a prior:
    m² = (Σ_k ratio_k² + κ·m0²) / (n + κ),   ratio_k² = Σ r² over weekend k / (v_fri · bars_k)

With clip = inf and use_weekend = False this is exactly the plain EWMA (ewma_sigma_ahead).
Forecasts over a horizon use the known calendar of future bars: F_t(H) = v_t · Σ_{j=1..H} s_{t+j}².
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolModel:
    lam: float = 0.94
    clip: float = 4.0            # winsorise at ±clip · σ̂ (np.inf = no clipping)
    use_weekend: bool = True
    m_prior: float = 0.6         # weekend vol multiplier prior (H-W1: ~0.77, H-W7 vs implied: lower)
    kappa: float = 8.0           # pseudo-weekends of prior weight
    window: int = 26             # trailing weekends used for m
    warmup: int = 30
    bias_correct: bool = False   # scale by slow EWMA(r²)/EWMA(x_clipped²): restores the level clipping removes (H-V1b)
    lam_bias: float = 0.999


def run(r: np.ndarray, weekend: np.ndarray, cfg: VolModel = VolModel()) -> dict[str, np.ndarray]:
    """Returns per bar t:
    sigma_prev  forecast for bar t made at t-1 (includes the seasonal scale of bar t)
    v_next      base (weekday-equivalent) variance after observing bar t (bias-corrected if enabled)
    m           weekend multiplier known at the close of bar t
    """
    r = np.nan_to_num(np.asarray(r, dtype=float))
    wk = np.asarray(weekend, dtype=bool)
    n = len(r)
    sigma_prev = np.full(n, np.nan)
    v_next = np.full(n, np.nan)
    m_out = np.full(n, np.nan)
    if n <= cfg.warmup:
        return {"sigma_prev": sigma_prev, "v_next": v_next, "m": m_out}
    v = float(np.mean(r[: cfg.warmup] ** 2)) or 1e-12
    m = cfg.m_prior if cfg.use_weekend else 1.0
    ratios: deque[float] = deque(maxlen=cfg.window)
    raw_e, clip_e = v, v  # slow EWMAs of raw and clipped squared (deseasonalised) returns
    in_wk, v_fri, acc, nb = False, 0.0, 0.0, 0
    for t in range(cfg.warmup, n):
        is_wk = bool(wk[t]) and cfg.use_weekend
        # weekend bookkeeping (entering / leaving the window)
        if is_wk and not in_wk:
            in_wk, v_fri, acc, nb = True, v, 0.0, 0
        elif not is_wk and in_wk:
            in_wk = False
            if nb >= 12 and v_fri > 0:
                ratios.append(acc / (v_fri * nb))
                m = float(np.sqrt((sum(ratios) + cfg.kappa * cfg.m_prior**2) / (len(ratios) + cfg.kappa)))
        s = m if is_wk else 1.0
        b = raw_e / clip_e if cfg.bias_correct and clip_e > 0 else 1.0
        sigma_prev[t] = np.sqrt(v * b) * s
        x = r[t] / s
        lim = cfg.clip * np.sqrt(v)
        if np.isfinite(lim):
            x = min(max(x, -lim), lim)
        x_raw = r[t] / s
        v = cfg.lam * v + (1 - cfg.lam) * x * x
        raw_e = cfg.lam_bias * raw_e + (1 - cfg.lam_bias) * x_raw * x_raw
        clip_e = cfg.lam_bias * clip_e + (1 - cfg.lam_bias) * x * x
        if in_wk:
            acc += r[t] ** 2
            nb += 1
        v_next[t] = v * (raw_e / clip_e if cfg.bias_correct and clip_e > 0 else 1.0)
        m_out[t] = m
    return {"sigma_prev": sigma_prev, "v_next": v_next, "m": m_out}


def horizon_variance(v_next: np.ndarray, m: np.ndarray, weekend: np.ndarray, h: int, use_weekend: bool = True) -> np.ndarray:
    """F_t(h) = v_t · Σ_{j=1..h} s_{t+j}², with s from the known future calendar and m known at t."""
    n = len(v_next)
    wk = np.asarray(weekend, dtype=float)
    out = np.full(n, np.nan)
    if n <= h:
        return out
    # count of weekend bars in (t, t+h]
    c = np.concatenate([[0.0], np.cumsum(wk)])
    idx = np.arange(n)
    n_wk = c[np.minimum(idx + h + 1, n)] - c[idx + 1]
    valid = idx + h < n
    mm = m if use_weekend else np.ones(n)
    out[valid] = v_next[valid] * ((h - n_wk[valid]) + n_wk[valid] * mm[valid] ** 2)
    return out


def har_forecast(rv_daily: np.ndarray, refit_every: int = 30, min_days: int = 180) -> np.ndarray:
    """Causal HAR-RV (Corsi 2009) on log realised variance: forecast of day d+1's RV made at the end of day d.
    log RV_{d+1} = b0 + b1 log RV_d + b2 mean_7(log RV) + b3 mean_30(log RV) + e; expanding-window OLS,
    refit every `refit_every` days on pairs fully known at d; mean = exp(fit + s²/2). Validated: H-VCL1."""
    import polars as pl

    rv = np.asarray(rv_daily, dtype=float)
    lrv = np.where(np.isfinite(rv) & (rv > 0), np.log(np.maximum(rv, 1e-12)), np.nan)
    n = len(lrv)
    # NaN-safe trailing means: a missing day must not poison every later window (a cumsum would).
    s_l = pl.Series(lrv).fill_nan(None)
    w = s_l.rolling_mean(7, min_samples=5).to_numpy().astype(float)
    m = s_l.rolling_mean(30, min_samples=20).to_numpy().astype(float)
    X = np.column_stack([np.ones(n), lrv, w, m])
    out = np.full(n, np.nan)
    coef, s2, last = None, 0.0, -10**9
    for d in range(30, n):
        if d >= min_days and (coef is None or d - last >= refit_every):
            Xj, yj = X[30:d], lrv[31 : d + 1]
            ok = np.isfinite(Xj).all(axis=1) & np.isfinite(yj)
            coef, *_ = np.linalg.lstsq(Xj[ok], yj[ok], rcond=None)
            s2, last = float(np.var(yj[ok] - Xj[ok] @ coef)), d
        if coef is not None and np.isfinite(X[d]).all():
            out[d] = float(np.exp(X[d] @ coef + 0.5 * s2))
    return out
