"""Extended HAR-RV variants (H-VCL2..H-VCL5, H-FU2). The validated base model is
`ai_trader.features.vol_models.har_forecast` (H-VCL1); it is left untouched.

har_forecast_x generalises it in two ways:
  * extra regressors (already causal at the end of day d, e.g. log RS-_d, BTC log RV_d, log IV²_d);
  * a direct multi-day horizon h: the target is log Σ_{k=1..h} RV_{d+k} (forecast of the next h days' variance).

Fit at day d (expanding window, refit every `refit_every` days once d >= min_days) on the pairs (X_j, y_j) with
j + h <= d, i.e. only targets fully observed by the end of day d. Forecast = exp(X_d·b + s²/2) (lognormal mean).
Rows with any non-finite regressor are dropped from the fit and get no forecast. With extra=None and h=1 the
output equals har_forecast on gap-free input.

The weekly / monthly components use sliding-window means, so a NaN day only blanks the windows containing it
(har_forecast's cumulative sum would propagate it forever).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of x[d-w+1..d]; NaN for d < w-1 or if the window holds a NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = sliding_window_view(x, w).mean(axis=1)
    return out


def forward_sum(x: np.ndarray, h: int) -> np.ndarray:
    """Σ_{k=1..h} x[d+k]; NaN where the window runs past the end or holds a NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) > h:
        out[: len(x) - h] = sliding_window_view(x[1:], h).sum(axis=1)
    return out


def har_design(rv_daily: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """[1, log RV_d, mean_7 log RV, mean_30 log RV, extra...] known at the end of day d."""
    rv = np.asarray(rv_daily, dtype=float)
    lrv = np.where(np.isfinite(rv), np.log(np.maximum(rv, 1e-12)), np.nan)
    cols = [np.ones(len(lrv)), lrv, trailing_mean(lrv, 7), trailing_mean(lrv, 30)]
    if extra is not None:
        e = np.asarray(extra, dtype=float)
        e = e.reshape(len(lrv), -1)
        cols += [e[:, k] for k in range(e.shape[1])]
    return np.column_stack(cols)


def har_forecast_x(
    rv_daily: np.ndarray,
    extra_regressors: np.ndarray | None = None,
    horizon: int = 1,
    refit_every: int = 30,
    min_days: int = 180,
    min_obs: int = 0,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Causal (direct) HAR-X forecast of Σ_{k=1..horizon} RV_{d+k}, made at the end of day d.

    extra_regressors: (n,) or (n, k) array aligned with rv_daily, each row known at the end of day d.
    min_obs: minimum number of complete fitting pairs before the first fit (0 = any).
    target_mask: optional bool (n,) — False excludes y_j from fitting (e.g. windows spanning a data gap).
    """
    rv = np.asarray(rv_daily, dtype=float)
    n = len(rv)
    X = har_design(rv, extra_regressors)
    tgt = forward_sum(np.maximum(rv, 1e-12), horizon)
    y = np.where(np.isfinite(tgt), np.log(np.maximum(tgt, 1e-12)), np.nan)
    if target_mask is not None:
        y = np.where(np.asarray(target_mask, dtype=bool), y, np.nan)
    rowok = np.isfinite(X).all(axis=1)
    out = np.full(n, np.nan)
    coef, s2, last = None, 0.0, -10**9
    for d in range(30, n):
        if d >= min_days and (coef is None or d - last >= refit_every):
            hi = d - horizon + 1  # pairs j in [30, d - h]: target window ends at j + h <= d
            if hi > 30:
                ok = rowok[30:hi] & np.isfinite(y[30:hi])
                if ok.sum() >= max(min_obs, X.shape[1] + 1):
                    Xj, yj = X[30:hi][ok], y[30:hi][ok]
                    coef, *_ = np.linalg.lstsq(Xj, yj, rcond=None)
                    s2, last = float(np.var(yj - Xj @ coef)), d
        if coef is not None and rowok[d]:
            out[d] = float(np.exp(X[d] @ coef + 0.5 * s2))
    return out


def qlike(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss V/F - log(V/F) - 1 (0 at a perfect forecast; robust to noisy variance proxies)."""
    r = np.asarray(realised, dtype=float) / np.asarray(forecast, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = r - np.log(r) - 1
    return np.where(np.isfinite(q), q, np.nan)


def iv_to_daily_var(iv: np.ndarray, vol_points: bool) -> np.ndarray:
    """Annualised implied vol -> daily variance of log returns: (IV)^2 / 365 (IV/100 when in vol points)."""
    v = np.asarray(iv, dtype=float) / (100.0 if vol_points else 1.0)
    return v * v / 365.0
