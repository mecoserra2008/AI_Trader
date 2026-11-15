"""H-NEWS1 (news / event / on-chain block vs log-HAR x weekday for next-day BTC variance) and H-NEWS2 (direction null).

Design (fixed in the diary before the evaluation, 2026-11-13):
  data      BTCUSDT 1h bars -> daily realised measures (features.realized), on a gap-aware calendar; origin = end of day
            t (asof = t + 1 00:00 UTC), target = day t + 1. Segment by the target day vs research.knowledge_cutoff.
  baseline  log-HAR (ln RV_t, mean ln RV over 7 d, over 30 d; vol_panel 'loghar') + 6 dummies for the forecast weekday
            (day t + 1), OLS on ln RV_{t+1}, lognormal back-transform exp(fit + s^2 / 2).
  augmented baseline + the 13-column block of news.features.BLOCK (missing block values set to 0 = 'no abnormal news').
  fitting   expanding window, refit every 30 days, first fit after 365 days (vol_panel.per_asset_forecast).
  H-NEWS1   paired QLIKE difference (augmented - baseline) per target day; Newey-West t; supports if mean < 0 and
            t < -2, contradicts if mean >= 0, else inconclusive; per segment.
  H-NEWS2   walk-forward L2 logistic (C = 1) of 1{ret_{t+1} > 0} on the standardised block, same refit schedule; base =
            the training up-day share; Brier skill = 1 - Brier_model / Brier_base; 95% circular 10-day block bootstrap
            CI (2,000 draws); supports if the upper bound <= 0.005, contradicts if the lower bound > 0.005, else
            inconclusive; per segment.
  robustness (reported, not recorded): news lagged one more day (*_l1), each sub-block alone, coefficients of the
            last fit.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.vol_panel import design, per_asset_forecast, qlike, target
from ai_trader.news.features import BLOCK, EVENT_COLS, NEWS_COLS, ONCHAIN_COLS
from ai_trader.research.evaluate import newey_west_t

REFIT, MIN_DAYS = 30, 365


def btc_daily(as_of: datetime, symbol: str = "BTCUSDT") -> pl.DataFrame:
    from ai_trader.features.realized import daily_panel

    p = daily_panel([symbol], as_of=as_of)
    grid = pl.DataFrame({"date": pl.date_range(p["date"].min(), p["date"].max(), "1d", eager=True)})
    return grid.join(p.select("date", "rv", "ret", "known_ts"), on="date", how="left").sort("date")


def frame(rv_df: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """Join daily RV (row = day t) with the feature row whose asof = t + 1 00:00."""
    f = feats.with_columns((pl.col("asof").dt.date() - timedelta(days=1)).alias("date"))
    return rv_df.join(f, on="date", how="left").sort("date")


def weekday_dummies(dates: pl.Series) -> np.ndarray:
    """6 dummies for the weekday of day t + 1 (Monday omitted)."""
    wd = np.array([(d + timedelta(days=1)).weekday() for d in dates.to_list()])
    return np.column_stack([(wd == k).astype(float) for k in range(1, 7)])


def block_matrix(df: pl.DataFrame, cols) -> np.ndarray:
    X = np.column_stack([df[c].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy() if c in df.columns else np.zeros(df.height) for c in cols])
    return X


def segments(dates: pl.Series, cutoff: date) -> np.ndarray:
    tgt = np.array([d + timedelta(days=1) for d in dates.to_list()])
    return np.where(tgt >= cutoff, "clean_holdout", "llm_in_sample")


def har_forecasts(df: pl.DataFrame, extra: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    rv = df["rv"].to_numpy().astype(float)
    Xh, _ = design("loghar", {"rv": rv})
    X = np.column_stack([Xh, weekday_dummies(df["date"])] + ([extra] if extra is not None else []))
    y = target(rv, 1, log=True)
    f = per_asset_forecast(X, y, 1, log=True, refit_every=REFIT, min_days=MIN_DAYS)
    realised = np.r_[rv[1:], np.nan]
    return f, realised


def last_fit_coefs(df: pl.DataFrame, cols, upto: int | None = None) -> dict[str, float]:
    rv = df["rv"].to_numpy().astype(float)
    Xh, _ = design("loghar", {"rv": rv})
    X = np.column_stack([np.ones(len(rv)), Xh, weekday_dummies(df["date"]), block_matrix(df, cols)])
    y = target(rv, 1, log=True)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if upto is not None:
        ok[upto:] = False
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    e = y[ok] - X[ok] @ b
    s2 = e @ e / (ok.sum() - X.shape[1])
    cov = s2 * np.linalg.pinv(X[ok].T @ X[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = b / np.sqrt(np.diag(cov))
    names = ["const", "lrv_d", "lrv_w", "lrv_m", "tue", "wed", "thu", "fri", "sat", "sun", *cols]
    return {n: {"coef": float(b[i]), "t_ols": float(tt[i])} for i, n in enumerate(names)}


def paired(loss_a: np.ndarray, loss_b: np.ndarray, seg: np.ndarray) -> dict:
    out = {}
    for s in ("llm_in_sample", "clean_holdout"):
        m = (seg == s) & np.isfinite(loss_a) & np.isfinite(loss_b)
        d = loss_a[m] - loss_b[m]
        mean, t, n = newey_west_t(d)
        out[s] = {"n": int(n), "qlike_aug": float(np.mean(loss_a[m])) if n else None, "qlike_base": float(np.mean(loss_b[m])) if n else None,
                  "diff": mean, "nw_t": t, "rel": float(mean / np.mean(loss_b[m])) if n else None}
    return out


def h1_verdict(r: dict) -> str:
    if r["n"] < 30 or not np.isfinite(r["nw_t"]):
        return "inconclusive"
    if r["diff"] < 0 and r["nw_t"] < -2:
        return "supports"
    if r["diff"] >= 0:
        return "contradicts"
    return "inconclusive"


def run_h1(df: pl.DataFrame, cutoff: date, block=BLOCK, variants: dict | None = None) -> dict:
    """`block` is the primary (recorded) block; `variants` (name -> columns) are robustness checks. Defaults = H-NEWS1;
    H-NEWS3 passes news.alt.ALT_COLS + EVENT_COLS + ONCHAIN_COLS and its own sub-blocks (scripts/news_alt_study.py)."""
    seg = segments(df["date"], cutoff)
    fb, real = har_forecasts(df, None)
    qb = qlike(real, fb)
    block = list(block)
    res = {"baseline": "loghar + forecast-weekday dummies", "block": block, "variants": {}}
    if variants is None:
        variants = {"news_l1": [f"{c}_l1" for c in NEWS_COLS] + list(EVENT_COLS + ONCHAIN_COLS),
                    "news_only": list(NEWS_COLS), "events_only": list(EVENT_COLS), "onchain_only": list(ONCHAIN_COLS)}
    variants = {"primary": block, **{k: list(v) for k, v in variants.items() if k != "primary"}}
    fc = {"baseline": fb}
    for name, cols in variants.items():
        fa, _ = har_forecasts(df, block_matrix(df, cols))
        fc[name] = fa
        res["variants"][name] = paired(qlike(real, fa), qb, seg)
    for s in ("llm_in_sample", "clean_holdout"):
        res["variants"]["primary"][s]["verdict"] = h1_verdict(res["variants"]["primary"][s])
    cut_i = int(np.searchsorted(np.array([d + timedelta(days=1) for d in df["date"].to_list()]), cutoff))
    res["coefs_at_cutoff"] = last_fit_coefs(df, block, upto=cut_i)
    res["coefs_full"] = last_fit_coefs(df, block)
    res["_forecasts"] = fc
    res["_realised"] = real
    return res


def _block_boot_ci(a: np.ndarray, b: np.ndarray, block: int = 10, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of 1 - mean(a)/mean(b) by circular block bootstrap of paired days."""
    n = len(a)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    st = []
    for _ in range(B):
        starts = rng.integers(0, n, size=k)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        st.append(1 - a[idx].mean() / b[idx].mean())
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def logistic_l2(X: np.ndarray, y: np.ndarray, C: float = 1.0, iters: int = 50) -> tuple[float, np.ndarray]:
    """L2 logistic regression (sklearn's objective with penalty 1/(2C)||w||^2, unpenalised intercept) by Newton steps."""
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    w = np.zeros(k + 1)
    P = np.eye(k + 1) / C
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + P @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + P + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(w[0]), w[1:]


def run_h2(df: pl.DataFrame, cutoff: date, cols=BLOCK) -> dict:
    seg = segments(df["date"], cutoff)
    ret = df["ret"].to_numpy().astype(float)
    y = np.r_[ret[1:], np.nan]
    X = block_matrix(df, cols)
    n = len(y)
    p_model = np.full(n, np.nan)
    p_base = np.full(n, np.nan)
    ok = np.isfinite(y)
    first = int(np.flatnonzero(ok)[0])
    refits = list(range(first + MIN_DAYS, n, REFIT))
    for i, d in enumerate(refits):
        top = d  # pairs j with j + 1 <= d (target known by d)
        tr = ok[:top].copy()
        if tr.sum() < 200:
            continue
        Xt, yt = X[:top][tr], (y[:top][tr] > 0).astype(int)
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        sd[sd == 0] = 1.0
        b0, w = logistic_l2((Xt - mu) / sd, yt.astype(float))
        hi = refits[i + 1] if i + 1 < len(refits) else n
        rows = np.arange(d, hi)
        p_model[rows] = 1 / (1 + np.exp(-(b0 + ((X[rows] - mu) / sd) @ w)))
        p_base[rows] = yt.mean()
    up = (y > 0).astype(float)
    bm, bb = (p_model - up) ** 2, (p_base - up) ** 2
    out = {"model": "L2 logistic (C=1) on the standardised block", "segments": {}}
    for s in ("llm_in_sample", "clean_holdout"):
        mk = (seg == s) & np.isfinite(bm) & np.isfinite(bb) & np.isfinite(y)
        a, b = bm[mk], bb[mk]
        if len(a) < 30:
            out["segments"][s] = {"n": int(len(a)), "verdict": "inconclusive"}
            continue
        skill = 1 - a.mean() / b.mean()
        lo, hi = _block_boot_ci(a, b)
        verdict = "supports" if hi <= 0.005 else "contradicts" if lo > 0.005 else "inconclusive"
        out["segments"][s] = {"n": int(len(a)), "brier_model": float(a.mean()), "brier_base": float(b.mean()), "skill": float(skill),
                              "ci95": [lo, hi], "hit_rate": float(np.mean((p_model[mk] > 0.5) == (up[mk] > 0.5))), "verdict": verdict}
    return out


__all__ = ["btc_daily", "frame", "run_h1", "run_h2"]
