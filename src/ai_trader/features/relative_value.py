"""Relative value / fair value features (research/hypotheses.yaml H-RV1..H-RV4). Pure, causal functions.

Conventions
  * Inputs are aligned arrays on one synchronous clock (row t = the bar closing at t). NaN marks a missing return.
  * `rolling_ols(y, X, window)` row t = OLS fitted on rows t-window+1..t (data known at the close of t).
    Everything *applied* at row t uses the fit through t-1 (row t-1 of rolling_ols), so a value at row t never
    depends on rows > t (truncation invariance, tested in tests/test_relative_value.py).
  * Spreads are in log units; z-scores are vs the trailing window ending at t-1.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

LN2 = float(np.log(2.0))


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum over rows t-window+1..t along axis 0 (rows < window-1 get the partial sum)."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return out


def rolling_ols(y: np.ndarray, X: np.ndarray, window: int, *, intercept: bool = True, min_obs: int | None = None) -> dict[str, np.ndarray]:
    """Trailing-window OLS of y on X (n x k), fitted at every row on rows t-window+1..t with finite data.

    Returns coef (n x k[+1], intercept first when intercept=True), resid_std (in-window residual std, dof-corrected),
    r2, n_obs. Rows with fewer than `min_obs` finite observations (default 0.8*window) are NaN."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if intercept:
        X = np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    min_obs = int(0.8 * window) if min_obs is None else min_obs
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xz = np.where(ok[:, None], X, 0.0)
    yz = np.where(ok, y, 0.0)
    xx = _rolling_sum(Xz[:, :, None] * Xz[:, None, :], window)
    xy = _rolling_sum(Xz * yz[:, None], window)
    yy = _rolling_sum(yz * yz, window)
    sy = _rolling_sum(yz, window)
    cnt = _rolling_sum(ok.astype(float), window)
    coef = np.full((n, k), np.nan)
    sd = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    good = cnt >= max(min_obs, k + 2)
    idx = np.flatnonzero(good)
    if len(idx):
        A, v = xx[idx], xy[idx]
        okA = np.abs(np.linalg.det(A)) > 1e-300
        idx, A, v = idx[okA], A[okA], v[okA]
        b = np.linalg.solve(A, v[:, :, None])[:, :, 0]
        ssr = np.maximum(yy[idx] - np.einsum("ij,ij->i", b, v), 0.0)
        coef[idx] = b
        sd[idx] = np.sqrt(ssr / (cnt[idx] - k))
        sst = yy[idx] - sy[idx] ** 2 / cnt[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[idx] = np.where(sst > 0, 1 - ssr / sst, np.nan)
    return {"coef": coef, "resid_std": sd, "r2": r2, "n_obs": cnt}


def _lag(a: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def factor_residuals(alt_ret: np.ndarray, factor_rets: np.ndarray, window: int = 720, lookback: int = 72,
                     fair_lookback: int | None = None, alt_price: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Causal factor model of an alt's log returns on factor log returns (n x m; m=2 for BTC, ETH).

    At row t the coefficients fitted on rows t-window..t-1 are applied to row t:
        resid_t  = r_t - a_{t-1} - b_{t-1}·f_t        (OLS residual, used for the signal)
        hedged_t = r_t - b_{t-1}·f_t                  (tradeable factor-hedged return; no intercept)
        cum_resid_t = Σ_{s=t-lookback+1..t} resid_s,  z_t = cum_resid_t / (resid_std_{t-1} √lookback)
    Fair value: the price the factor exposure alone implies since `fair_lookback` bars ago,
        fair_t = P_{t-L} · exp(Σ_{s=t-L+1..t} b_{s-1}·f_s),  gap_t = P_t / fair_t - 1 = exp(Σ hedged) - 1.
    `beta_next` = coefficients fitted through t (what applies to row t+1, e.g. a hedge put on at the close of t)."""
    r = np.asarray(alt_ret, dtype=float)
    F = np.asarray(factor_rets, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    fit = rolling_ols(r, F, window)
    coef_next = fit["coef"]
    coef = np.vstack([np.full((1, coef_next.shape[1]), np.nan), coef_next[:-1]])
    sd = _lag(fit["resid_std"])
    a, b = coef[:, 0], coef[:, 1:]
    expl = np.einsum("ij,ij->i", b, F)
    resid = r - a - expl
    hedged = r - expl
    cum = _rolling_nansum_strict(resid, lookback)
    z = cum / (sd * np.sqrt(lookback))
    L = fair_lookback or window
    gap_log = _rolling_nansum_strict(hedged, L)
    out = {"alpha": a, "beta": b, "beta_next": coef_next[:, 1:], "alpha_next": coef_next[:, 0], "resid": resid,
           "hedged": hedged, "resid_std": sd, "r2": _lag(fit["r2"]), "cum_resid": cum, "z": z,
           "gap_log": gap_log, "gap_pct": np.expm1(gap_log)}
    if alt_price is not None:
        p = np.asarray(alt_price, dtype=float)
        out["fair_price"] = p * np.exp(-gap_log)
    return out


def _rolling_nansum_strict(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over the last `window` rows; NaN unless all `window` values are finite."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    s = _rolling_sum(np.where(ok, x, 0.0), window)
    c = _rolling_sum(ok.astype(float), window)
    out = np.where(c >= window, s, np.nan)
    out[: window - 1] = np.nan
    return out


def _rolling_mean_std(x: np.ndarray, window: int, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x)
    xz = np.where(ok, x, 0.0)
    s = _rolling_sum(xz, window)
    s2 = _rolling_sum(xz * xz, window)
    c = _rolling_sum(ok.astype(float), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
        v = (s2 - c * m * m) / (c - 1)
    m = np.where(c >= min_obs, m, np.nan)
    sd = np.where(c >= min_obs, np.sqrt(np.maximum(v, 0.0)), np.nan)
    return m, sd


def spread_z(log_a: np.ndarray, log_b: np.ndarray, window: int = 720, *, hedge: bool = False, min_obs: int | None = None) -> dict[str, np.ndarray]:
    """Log spread and its causal z-score.

    hedge=False: s_t = log_a - log_b; mean/std over rows t-window..t-1; z_t = (s_t - mean)/std.
    hedge=True:  rolling OLS of log_a on log_b (levels) through t-1; s_t = log_a - alpha - beta·log_b (the
                 out-of-window residual), mean 0 by construction, z_t = s_t / in-window residual std."""
    a = np.asarray(log_a, dtype=float)
    b = np.asarray(log_b, dtype=float)
    min_obs = window if min_obs is None else min_obs
    if not hedge:
        s = a - b
        m, sd = _rolling_mean_std(s, window, min_obs)
        m, sd = _lag(m), _lag(sd)
        return {"spread": s, "mean": m, "std": sd, "z": (s - m) / sd, "beta": np.ones_like(s)}
    fit = rolling_ols(a, b, window, min_obs=min_obs)
    coef = np.vstack([np.full((1, 2), np.nan), fit["coef"][:-1]])
    sd = _lag(fit["resid_std"])
    s = a - coef[:, 0] - coef[:, 1] * b
    return {"spread": s, "mean": np.zeros_like(s), "std": sd, "z": s / sd, "beta": coef[:, 1], "alpha": coef[:, 0]}


def ar1_half_life(x: np.ndarray) -> dict[str, float]:
    """OLS fit x_t = c + phi x_{t-1} + e on the finite pairs; half-life = -ln2 / ln(phi) in rows (inf if phi >= 1)."""
    x = np.asarray(x, dtype=float)
    x0, x1 = x[:-1], x[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    x0, x1 = x0[ok], x1[ok]
    if len(x0) < 10:
        return {"phi": float("nan"), "half_life": float("nan"), "n": int(len(x0))}
    xc = x0 - x0.mean()
    phi = float(xc @ (x1 - x1.mean()) / (xc @ xc))
    hl = -LN2 / np.log(phi) if 0 < phi < 1 else (float("inf") if phi >= 1 else float("nan"))
    e = x1 - x1.mean() - phi * xc
    se = float(np.sqrt(e @ e / (len(e) - 2) / (xc @ xc)))
    return {"phi": phi, "phi_se": se, "half_life": float(hl), "n": int(len(x0))}


# ------------------------------------------------------------------ perp-spot basis (H-RV4, forward)

def basis(perp_mid: np.ndarray, spot_mid: np.ndarray) -> np.ndarray:
    """Basis in bps: 1e4 · (ln perp_mid - ln spot_mid)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1e4 * (np.log(np.asarray(perp_mid, dtype=float)) - np.log(np.asarray(spot_mid, dtype=float)))


def basis_frame(perp: pl.DataFrame, spot: pl.DataFrame, window: timedelta = timedelta(days=1), min_obs: int = 1000) -> pl.DataFrame:
    """Join perp and spot flow rows (event_ts, mid) on identical timestamps (synchronous snapshots only) and add
    basis_bps plus its z vs the trailing `window` (rows strictly before t: closed='left')."""
    j = (perp.select("event_ts", pl.col("mid").alias("perp_mid"))
         .join(spot.select("event_ts", pl.col("mid").alias("spot_mid")), on="event_ts", how="inner")
         .filter((pl.col("perp_mid") > 0) & (pl.col("spot_mid") > 0)).sort("event_ts"))
    j = j.with_columns(pl.Series("basis_bps", basis(j["perp_mid"].to_numpy(), j["spot_mid"].to_numpy())))
    w = f"{int(window.total_seconds())}s"
    j = j.with_columns(
        pl.col("basis_bps").rolling_mean_by("event_ts", w, closed="left").alias("b_mean"),
        pl.col("basis_bps").rolling_std_by("event_ts", w, closed="left").alias("b_std"),
        pl.lit(1, dtype=pl.Int64).alias("_one"),
    ).with_columns(pl.col("_one").rolling_sum_by("event_ts", w, closed="left").fill_null(0).alias("b_n")).drop("_one")
    return j.with_columns(pl.when(pl.col("b_n") >= min_obs).then((pl.col("basis_bps") - pl.col("b_mean")) / pl.col("b_std")).otherwise(None).alias("b_z"))


def basis_excursions(bf: pl.DataFrame, z_thresh: float = 3.0, horizon: timedelta = timedelta(minutes=60), decluster: timedelta | None = None) -> pl.DataFrame:
    """H-RV4 measurement: events where |b_z| >= z_thresh (one per `decluster`, default = horizon); reversion =
    1 - (b_{t+h} - mean_t) / (b_t - mean_t)  (1 = fully back to the trailing mean, 0 = no reversion)."""
    decluster = decluster or horizon
    d = bf.filter(pl.col("b_z").is_not_null()).sort("event_ts")
    ts = d["event_ts"].to_list()
    z = d["b_z"].to_numpy()
    keep, last = [], None
    for i in np.flatnonzero(np.abs(z) >= z_thresh):
        if last is None or ts[i] - last >= decluster:
            keep.append(i)
            last = ts[i]
    ev = d[keep].select("event_ts", "basis_bps", "b_mean", "b_std", "b_z").with_columns((pl.col("event_ts") + horizon).alias("t_h"))
    fut = bf.select(pl.col("event_ts").alias("t_h"), pl.col("basis_bps").alias("b_h")).sort("t_h")
    ev = ev.sort("t_h").join_asof(fut, on="t_h", strategy="backward", tolerance=timedelta(seconds=30))
    return ev.with_columns((1 - (pl.col("b_h") - pl.col("b_mean")) / (pl.col("basis_bps") - pl.col("b_mean"))).alias("reversion"))
