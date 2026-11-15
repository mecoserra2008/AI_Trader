"""Volatility-based signals (block VB, program 2026-11-10): H-VB1..H-VB4.

H-VB1  variance-managed exposure (Moreira & Muir 2017). Base series per universe: the equal-weight (EW) long-only
       portfolio and the EW trend-zoo ensemble (4h, same signals and execution policy as trend_zoo_study, checked
       against data/priors/trend_zoo_daily_4h.parquet). Managed_D = w_D * base_D, w_D = c / sigma2_hat_D, sigma2_hat_D
       = HAR forecast of day D's variance of the EW index (index from EW 1h simple returns; RV via
       features.realized.daily_measures), made at the end of D-1. c matches the unmanaged vol on the full sample (as in
       the paper; the spanning-regression alpha t-stat is invariant to a constant c) or in real time (expanding).
       Costs 7.5 bps per side on |delta weight| (long-only) and on |delta (w * leverage)| per 4h bar (ensemble).
       Spanning regression managed = a + b base, HAC (Newey-West) t on a.
H-VB2  daily vol surprise: RV_t > 1.5 F_t and |ret_t| > sqrt(F_t) (F_t made at the end of t-1); signed next-1d / 3d
       returns in sigma units (sqrt(h F_{t+1|t})), net of 2 x 7.5 bps; declustered 3 days per symbol; t from NW over
       the date-mean series; split by the v0 vol state.
H-VB3  4h squeeze-release breakouts (BB(20,2) inside KC(EMA20, 1.5 ATR20) for >= 6 bars, first close outside BB at or
       after release) vs fresh BB breakouts with no recent squeeze vs a count-matched random-entry baseline.
H-VB4  1h exceedances |r_t| > 2 sigma_t (sigma_t = sigma_risk_next of bar t-1): per-symbol Hawkes (exponential kernel,
       the continuous-time likelihood of jump_diffusion.fit_hawkes, evaluated on a bar grid), expanding window, refit
       monthly; next-24h Poisson log score vs a Poisson with rate 24 * P(|r| > 2 sigma_0) under a unit-variance
       Student-t with per-bar variance F/24 (HAR daily forecast; df fitted on universe L llm_in_sample).

Daily variance forecast: universe L per symbol `vol_models_x.har_forecast_x` (min 180 days, refit 30); universe T
(115-555 days of history) a pooled log-HAR (`pooled_har_forecast`) with target-day weekend / Monday dummies.
Segments: decision timestamp vs research.knowledge_cutoff. Holm across the four VB primaries per segment.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.vol_signal_study [--cache DIR] [--record]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
from scipy import optimize, signal, special, stats

UNIVERSE_T = ["XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "TSLAUSDT", "INTCUSDT", "GOOGLUSDT", "AAPLUSDT", "MSTRUSDT", "NVDAUSDT", "MUUSDT", "QQQUSDT", "SOXLUSDT", "SPCXUSDT", "PAXGUSDT", "XAUTUSDT"]
COST_BPS = 7.5
SEGS = ("llm_in_sample", "clean_holdout")
MIN_DAYS = {"L": 180, "T": 60}
SURPRISE_RATIO, SURPRISE_K, DECLUSTER_DAYS = 1.5, 1.0, 3
SQ_MIN, SQ_WINDOW, SQ_H = 6, 20, 6
EXC_K, HORIZON_H = 2.0, 24
N_RANDOM = 200
_ND = NormalDist()


def universe_l() -> list[str]:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    return list(UNIVERSE)


# ====================================================================== statistics helpers
def nw_t(x: np.ndarray) -> tuple[float, float, int]:
    from ai_trader.research.evaluate import newey_west_t

    return newey_west_t(np.asarray(x, float))


def p_one_sided(t: float) -> float:
    """P(Z > t) under the null; NaN t -> 1."""
    return 1.0 - _ND.cdf(t) if np.isfinite(t) else 1.0


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def date_mean_t(values: np.ndarray, dates: np.ndarray) -> dict:
    """Average events per date (cross-sectional clustering), then NW t over the date series in time order."""
    v, d = np.asarray(values, float), np.asarray(dates)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"mean": float("nan"), "t": float("nan"), "n_events": 0, "n_dates": 0}
    dd = d[ok]
    dser = pl.Series("d", dd.tolist() if dd.dtype == object else dd)
    df = pl.DataFrame({"d": dser, "v": v[ok]}).group_by("d").agg(pl.col("v").mean()).sort("d")
    m, t, nd = nw_t(df["v"].to_numpy())
    return {"mean": float(np.mean(v[ok])), "date_mean": m, "t": t, "n_events": int(ok.sum()), "n_dates": nd,
            "p_pos": float((v[ok] > 0).mean())}


def cluster_diff(y: np.ndarray, is_a: np.ndarray, dates: np.ndarray) -> dict:
    """OLS y = c + d * 1{a}: d with SE clustered by date (CR1)."""
    import statsmodels.api as sm

    y, a, d = np.asarray(y, float), np.asarray(is_a, float), np.asarray(dates)
    ok = np.isfinite(y)
    y, a, d = y[ok], a[ok], d[ok]
    if len(y) < 20 or a.sum() < 5 or (1 - a).sum() < 5:
        return {"diff": float("nan"), "t": float("nan"), "n": len(y)}
    codes = pl.Series(d).cast(pl.Utf8).rank("dense").to_numpy()
    res = sm.OLS(y, sm.add_constant(a)).fit(cov_type="cluster", cov_kwds={"groups": codes})
    return {"diff": float(res.params[1]), "t": float(res.tvalues[1]), "n": len(y), "n_clusters": int(codes.max())}


def nw_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def spanning(managed: np.ndarray, base: np.ndarray) -> dict:
    """managed = a + b base + e, HAC t (Bartlett, NW automatic lag). Daily units; alpha_ann = 365 a."""
    import statsmodels.api as sm

    m, b = np.asarray(managed, float), np.asarray(base, float)
    ok = np.isfinite(m) & np.isfinite(b)
    m, b = m[ok], b[ok]
    n = len(m)
    if n < 60:
        return {"n_days": n, "alpha": float("nan"), "t_alpha": float("nan")}
    res = sm.OLS(m, sm.add_constant(b)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag(n)})
    resid_sd = float(np.std(res.resid, ddof=2))
    sr = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(365)) if x.std() > 0 else float("nan")
    return {"n_days": n, "alpha": float(res.params[0]), "alpha_ann": float(res.params[0] * 365), "t_alpha": float(res.tvalues[0]),
            "beta": float(res.params[1]), "appraisal_ann": float(res.params[0] / resid_sd * np.sqrt(365)) if resid_sd > 0 else float("nan"),
            "sharpe_managed": sr(m), "sharpe_base": sr(b), "vol_ratio": float(m.std() / b.std()) if b.std() > 0 else float("nan")}


# ====================================================================== indicators / event detectors (causal)
def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).mean(axis=1)
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Population sd over x[t-n+1..t] (Bollinger's convention)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).std(axis=1)
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """EMA (alpha = 2/(n+1)) seeded with the SMA of the first n values; NaN before."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for t in range(n, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def squeeze_bands(close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int = 20, k_bb: float = 2.0, k_kc: float = 1.5) -> dict:
    """Bollinger (SMA n, k_bb population sd) and Keltner (EMA n, k_kc x SMA n of true range); row t uses bars <= t."""
    c, h, lo = (np.asarray(v, float) for v in (close, high, low))
    mid = rolling_mean(c, n)
    sd = rolling_std(c, n)
    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - lo, np.abs(h - prev), np.abs(lo - prev)]), axis=0)
    atr = rolling_mean(tr, n)
    kmid = ema(c, n)
    bb_up, bb_lo = mid + k_bb * sd, mid - k_bb * sd
    kc_up, kc_lo = kmid + k_kc * atr, kmid - k_kc * atr
    with np.errstate(invalid="ignore"):
        sq = (bb_up < kc_up) & (bb_lo > kc_lo)
    return {"mid": mid, "bb_up": bb_up, "bb_lo": bb_lo, "kc_up": kc_up, "kc_lo": kc_lo, "squeeze": sq & np.isfinite(bb_up) & np.isfinite(kc_up)}


def squeeze_release_breakouts(close: np.ndarray, bands: dict, min_squeeze: int = SQ_MIN, window: int = SQ_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Indices and signs of the first close outside BB at or after a release (first non-squeeze bar after >= min_squeeze
    squeeze bars), before the next squeeze starts and within `window` bars of the release. Causal (online loop)."""
    c, sq, up, lo = np.asarray(close, float), bands["squeeze"], bands["bb_up"], bands["bb_lo"]
    idx, sgn = [], []
    run, armed, until = 0, False, -1
    for t in range(len(c)):
        if sq[t]:
            run += 1
            armed = False
            continue
        if run >= min_squeeze:
            armed, until = True, t + window - 1
        run = 0
        if armed:
            if t > until:
                armed = False
            elif c[t] > up[t]:
                idx.append(t), sgn.append(1)
                armed = False
            elif c[t] < lo[t]:
                idx.append(t), sgn.append(-1)
                armed = False
    return np.array(idx, int), np.array(sgn, int)


def plain_bb_breakouts(close: np.ndarray, bands: dict, min_squeeze: int = SQ_MIN, window: int = SQ_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Fresh closes outside BB (previous close inside), not in a squeeze, with no >= min_squeeze squeeze ended
    (released) within the last `window` bars."""
    c, sq, up, lo = np.asarray(close, float), bands["squeeze"], bands["bb_up"], bands["bb_lo"]
    idx, sgn = [], []
    run, last_rel = 0, -10**9
    for t in range(len(c)):
        if sq[t]:
            run += 1
            continue
        if run >= min_squeeze:
            last_rel = t
        run = 0
        if t == 0 or not np.isfinite(up[t]) or not np.isfinite(up[t - 1]) or t - last_rel < window:
            continue
        if c[t] > up[t] and c[t - 1] <= up[t - 1]:
            idx.append(t), sgn.append(1)
        elif c[t] < lo[t] and c[t - 1] >= lo[t - 1]:
            idx.append(t), sgn.append(-1)
    return np.array(idx, int), np.array(sgn, int)


def decluster(idx: np.ndarray, gap: int) -> np.ndarray:
    """Boolean keep-mask: an index within `gap` of the previous kept one is dropped."""
    keep = np.zeros(len(idx), bool)
    last = -10**18
    for i, v in enumerate(idx):
        if v - last > gap:
            keep[i] = True
            last = v
    return keep


def vol_surprise_events(rv: np.ndarray, ret: np.ndarray, f_cur: np.ndarray, ratio: float = SURPRISE_RATIO, k: float = SURPRISE_K,
                        gap: int = DECLUSTER_DAYS) -> np.ndarray:
    """Day indices t (contiguous daily grid) with RV_t > ratio F_t and |ret_t| > k sqrt(F_t), declustered `gap` days."""
    rv, ret, f = (np.asarray(v, float) for v in (rv, ret, f_cur))
    with np.errstate(invalid="ignore"):
        hit = np.isfinite(rv) & np.isfinite(ret) & np.isfinite(f) & (f > 0) & (rv > ratio * f) & (np.abs(ret) > k * np.sqrt(np.maximum(f, 0)))
    idx = np.where(hit)[0]
    return idx[decluster(idx, gap)]


# ====================================================================== daily variance forecasts
def har_design_pooled(rv: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    from ai_trader.features.vol_models_x import har_design

    return har_design(rv, extra)


def pooled_har_forecast(series: dict[str, tuple[np.ndarray, np.ndarray]], extra: dict[str, np.ndarray] | None = None,
                        refit_every: int = 30, min_pairs: int = 300) -> dict[str, np.ndarray]:
    """Pooled log-HAR across symbols (common slopes and intercept). series[sym] = (dates on a contiguous daily grid,
    rv). Forecast for day t+1 made at the end of t: at refit day d (every `refit_every` days on a global calendar) fit
    on all pairs (X_j, log RV_{j+1}) with j + 1 <= d, i.e. targets observed by the end of d; forecast rows j in
    [d, d + refit_every) as exp(X b + s2/2). Returns sym -> forecast array aligned with its grid."""
    rows = []
    for s, (dts, rv) in series.items():
        X = har_design_pooled(rv, None if extra is None else extra[s])
        lrv = np.where(np.isfinite(rv) & (rv > 0), np.log(np.maximum(rv, 1e-12)), np.nan)
        y = np.r_[lrv[1:], np.nan]
        rows.append((s, np.asarray(dts, "datetime64[D]"), X, y))
    all_d = np.concatenate([r[1] for r in rows])
    d0, d1 = all_d.min(), all_d.max()
    out = {s: np.full(len(d), np.nan) for s, d, _, _ in rows}
    coef, s2 = None, 0.0
    d = d0
    while d <= d1:
        Xs, ys = [], []
        for _s, dts, X, y in rows:
            m = (dts <= d - np.timedelta64(1, "D")) & np.isfinite(X).all(axis=1) & np.isfinite(y)
            if m.any():
                Xs.append(X[m]), ys.append(y[m])
        if Xs and sum(len(v) for v in ys) >= max(min_pairs, Xs[0].shape[1] + 1):
            Xa, ya = np.vstack(Xs), np.concatenate(ys)
            coef, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
            s2 = float(np.var(ya - Xa @ coef))
        if coef is not None:
            for s, dts, X, _y in rows:
                m = (dts >= d) & (dts < d + np.timedelta64(refit_every, "D")) & np.isfinite(X).all(axis=1)
                out[s][m] = np.exp(X[m] @ coef + 0.5 * s2)
        d = d + np.timedelta64(refit_every, "D")
    return out


def calendar_extra(dts: np.ndarray) -> np.ndarray:
    """[target day (t+1) is Sat/Sun, target day is Monday], known at the end of t."""
    nxt = np.asarray(dts, "datetime64[D]") + np.timedelta64(1, "D")
    wd = (nxt.astype("datetime64[D]").view("int64") - 4) % 7  # 0 = Monday (1970-01-01 was a Thursday)
    return np.column_stack([(wd >= 5).astype(float), (wd == 0).astype(float)])


def contiguous(daily: pl.DataFrame) -> pl.DataFrame:
    """Reindex a one-symbol daily frame to a gap-free calendar grid (missing days -> nulls)."""
    grid = pl.DataFrame({"date": pl.date_range(daily["date"].min(), daily["date"].max(), "1d", eager=True)})
    return grid.join(daily, on="date", how="left").sort("date")


# ====================================================================== trend-zoo ensemble execution (replicates strategy_pnl)
def held_leverage(sig: np.ndarray, sigma_next: np.ndarray, band: float = 0.2) -> np.ndarray:
    """The evaluate.strategy_pnl execution policy, returning the leverage path (s / sigma at entry, held)."""
    s = np.nan_to_num(np.asarray(sig, float))
    lev = np.zeros(len(s))
    held_s, held_lev = 0.0, 0.0
    for t in range(len(s)):
        st = s[t]
        if st == 0.0:
            held_s, held_lev = 0.0, 0.0
        elif held_s == 0.0 or np.sign(st) != np.sign(held_s) or abs(st - held_s) > band:
            held_s = st
            held_lev = st / sigma_next[t] if np.isfinite(sigma_next[t]) and sigma_next[t] > 0 else 0.0
        lev[t] = held_lev
    return lev


def scaled_pnl(lev: np.ndarray, fwd_ret: np.ndarray, w: np.ndarray, cost_bps: float = COST_BPS) -> np.ndarray:
    """Per-bar net P&L of leverage w_b * lev_b (risk units; cost on |delta leverage|)."""
    L = np.nan_to_num(np.asarray(w, float)) * lev
    turn = np.abs(np.diff(np.r_[0.0, L]))
    return L * np.nan_to_num(fwd_ret) - turn * cost_bps / 1e4


# ====================================================================== Hawkes on a bar grid
def hawkes_nll_grid(th: np.ndarray, flags: np.ndarray, ev: np.ndarray) -> float:
    """Same negative log-likelihood as jump_diffusion._hawkes_nll for event times = indices of `flags` on [0, T],
    T = len(flags); the excitation sum is computed by an IIR filter (A_t = e^-b (A_{t-1} + flag_{t-1}))."""
    mu, a, b = math.exp(th[0]), math.exp(th[1]), math.exp(th[2])
    if a >= b:
        return 1e12
    d = math.exp(-b)
    A = signal.lfilter([0.0, d], [1.0, -d], flags)
    T = len(flags)
    ll = float(np.sum(np.log(mu + a * A[ev.astype(np.int64)])))
    comp = mu * T + (a / b) * float(np.sum(1 - np.exp(-b * (T - ev))))
    return -(ll - comp)


def fit_hawkes_grid(flags: np.ndarray, x0: tuple[float, float, float] | None = None) -> tuple[float, float, float, float]:
    """(mu, alpha, beta, LR vs constant) per bar; starts as fit_hawkes plus an optional warm start."""
    flags = np.asarray(flags, float)
    ev = np.where(flags > 0)[0].astype(float)
    T, n = float(len(flags)), len(ev)
    if n < 10:
        return n / T if T else float("nan"), 0.0, 1.0, 0.0
    mu0 = n / T
    ll0 = n * math.log(mu0) - mu0 * T
    starts = [[math.log(mu0 * 0.7), math.log(b0 * 0.3), math.log(b0)] for b0 in (0.05, 0.2, 1.0)]
    if x0 is not None and x0[1] > 0:
        starts = [[math.log(x0[0]), math.log(x0[1]), math.log(x0[2])], starts[1]]
    best = None
    for s in starts:
        res = optimize.minimize(hawkes_nll_grid, s, args=(flags, ev), method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    mu, a, b = (math.exp(v) for v in best.x)
    return mu, a, b, max(0.0, 2 * (-best.fun - ll0))


def hawkes_expected_count(lam0: float, mu: float, alpha: float, beta: float, H: float) -> float:
    """E[N(t, t+H)] for an exponential Hawkes with current intensity lam0 (events before t only):
    E lambda(t+s) = m + (lam0 - m) e^{-(beta-alpha) s}, m = mu beta / (beta - alpha)."""
    k = beta - alpha
    if k <= 0:
        return float("nan")
    m = mu * beta / k
    return float(m * H + (lam0 - m) * (1 - math.exp(-k * H)) / k)


def t_exceed_prob(q: np.ndarray, nu: float) -> np.ndarray:
    """P(|X| > q) for a unit-variance Student-t with nu > 2 df."""
    s = math.sqrt(nu / (nu - 2))
    return 2 * stats.t.sf(np.asarray(q, float) * s, nu)


def fit_t_df(z: np.ndarray) -> float:
    """MLE of nu for a unit-variance Student-t (scale fixed by the variance forecast)."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]

    def nll(nu):
        s = math.sqrt((nu - 2) / nu)
        return -float(np.sum(stats.t.logpdf(z / s, nu) - math.log(s)))

    return float(optimize.minimize_scalar(nll, bounds=(2.05, 60.0), method="bounded").x)


def poisson_logpmf(n: np.ndarray, lam: np.ndarray) -> np.ndarray:
    n, lam = np.asarray(n, float), np.maximum(np.asarray(lam, float), 1e-12)
    return n * np.log(lam) - lam - special.gammaln(n + 1)


# ====================================================================== per-symbol loading (worker)
def _seg_of_ts(ts: pl.Expr) -> pl.Expr:
    from ai_trader.research.evaluate import knowledge_cutoff

    return pl.when(ts < knowledge_cutoff()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout"))


def load_symbol(sym: str, univ: str, as_of: datetime) -> dict | None:
    """Everything the study needs from one symbol (point in time: bars known at or before as_of)."""
    from ai_trader.features.realized import daily_measures
    from ai_trader.research import regimes
    from ai_trader.research.frame import market_frame
    from ai_trader.strategies.trend_zoo import lookback_bars, zoo

    f1 = market_frame(sym, "1h", as_of, funding=False, oi=False)
    if f1.height < 24 * 60:
        return None
    lab = regimes.label(f1, bar_hours=1.0)
    daily = daily_measures(f1.select("event_ts", "open", "high", "low", "close"), "1h").filter(pl.col("known_ts") <= as_of)
    vs = (lab.with_columns(pl.col("event_ts").dt.date().alias("date")).sort("event_ts").group_by("date")
          .agg(pl.col("vol_state").last().alias("vs_end")))
    daily = contiguous(daily.join(vs, on="date", how="left")).with_columns(pl.col("vs_end").shift(1).alias("vs_start"))
    h1 = f1.select("event_ts", "ret", pl.col("sigma_risk_next").shift(1).alias("sig_thr"))
    out = {"symbol": sym, "univ": univ, "daily": daily, "h1": h1}

    # ---- 4h: squeeze study and the trend-zoo ensemble
    f4 = market_frame(sym, "4h", as_of, funding=False, oi=False)
    if f4.height < 200:
        return out
    c, hi, lo = f4["close"].to_numpy(), f4["high"].to_numpy(), f4["low"].to_numpy()
    sn, ts, cts = f4["sigma_next"].to_numpy(), f4["event_ts"], f4["close_ts"]
    bands = squeeze_bands(c, hi, lo)
    lc = np.log(c)
    fwd = np.full(len(c), np.nan)
    fwd[:-SQ_H] = lc[SQ_H:] - lc[:-SQ_H]
    gap_ok = np.zeros(len(c), bool)
    tsn = ts.dt.epoch("s").to_numpy()
    gap_ok[:-SQ_H] = (tsn[SQ_H:] - tsn[:-SQ_H]) == SQ_H * 4 * 3600
    fwd = np.where(gap_ok, fwd, np.nan)
    den = sn * np.sqrt(SQ_H)
    cost_z = 2 * COST_BPS / 1e4 / den
    ev = []
    for cls, (ix, sg) in (("squeeze", squeeze_release_breakouts(c, bands)), ("plain", plain_bb_breakouts(c, bands))):
        keep = decluster(ix, SQ_H)
        ix, sg = ix[keep], sg[keep]
        for i, s_ in zip(ix, sg, strict=True):
            if np.isfinite(fwd[i]) and np.isfinite(den[i]) and den[i] > 0:
                g = s_ * fwd[i] / den[i]
                ev.append({"cls": cls, "decision_ts": cts[int(i)], "sign": int(s_), "z_gross": float(g), "z_net": float(g - cost_z[i])})
    out["vb3"] = pl.DataFrame(ev) if ev else None
    side = np.sign(c - bands["mid"])
    ok = np.isfinite(fwd) & np.isfinite(den) & (den > 0) & (side != 0) & np.isfinite(bands["mid"])
    out["vb3_pool"] = pl.DataFrame({"decision_ts": cts.filter(pl.Series(ok)),
                                    "z_net": (side * fwd / den - cost_z)[ok]})

    lb = lookback_bars(6)
    if f4.height >= lb + 90 * 6:
        variants = zoo(6)
        sig = np.mean(np.vstack([fn(f4).to_numpy() for fn in variants.values()]), axis=0)
        lev = held_leverage(sig, sn)
        start = ts[lb].date() + timedelta(days=1)
        out["ens"] = pl.DataFrame({"event_ts": ts, "lev": lev, "fwd_ret": f4["fwd_ret"].to_numpy()}).with_columns(
            pl.col("event_ts").dt.date().alias("date")).filter(pl.col("date") >= start)
    return out


def _load_job(args):
    return load_symbol(*args)


def load_all(as_of: datetime, cache: Path | None = None, workers: int = 4) -> list[dict]:
    if cache is not None and (cache / "symbols.pkl").is_file():
        return pickle.loads((cache / "symbols.pkl").read_bytes())
    jobs = [(s, "L", as_of) for s in universe_l()] + [(s, "T", as_of) for s in UNIVERSE_T]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = [r for r in ex.map(_load_job, jobs) if r is not None]
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "symbols.pkl").write_bytes(pickle.dumps(res))
    return res


# ====================================================================== forecasts for every symbol
def add_forecasts(syms: list[dict]) -> None:
    """Adds daily columns f_next (forecast of RV_{t+1} made at the end of t) and f_cur (of RV_t made at end of t-1)."""
    from ai_trader.features.vol_models_x import har_forecast_x

    t_series, t_extra = {}, {}
    for s in syms:
        d = s["daily"]
        rv = d["rv"].fill_null(float("nan")).to_numpy()
        if s["univ"] == "L":
            fc = har_forecast_x(rv, horizon=1, refit_every=30, min_days=MIN_DAYS["L"])
            s["daily"] = d.with_columns(pl.Series("f_next", fc)).with_columns(pl.col("f_next").shift(1).alias("f_cur"))
        else:
            dts = d["date"].to_numpy().astype("datetime64[D]")
            t_series[s["symbol"]] = (dts, rv)
            t_extra[s["symbol"]] = calendar_extra(dts)
    if t_series:
        fc = pooled_har_forecast(t_series, t_extra)
        for s in syms:
            if s["univ"] == "T":
                s["daily"] = s["daily"].with_columns(pl.Series("f_next", fc[s["symbol"]])).with_columns(pl.col("f_next").shift(1).alias("f_cur"))
    for s in syms:
        s["daily"] = s["daily"].with_columns(pl.col("f_next").fill_nan(None), pl.col("f_cur").fill_nan(None))


# ====================================================================== H-VB1
def ew_index_daily(syms: list[dict], univ: str, min_members: int = 5) -> pl.DataFrame:
    """EW index of 1h simple returns (members with a bar that hour) -> daily measures; days with < min_members dropped."""
    from ai_trader.features.realized import daily_measures

    h = pl.concat([s["h1"].select("event_ts", (pl.col("ret").exp() - 1).alias("sr"), pl.lit(s["symbol"]).alias("sym"))
                   for s in syms if s["univ"] == univ]).drop_nulls("sr").filter(pl.col("sr").is_finite())
    g = h.group_by("event_ts").agg(pl.col("sr").mean(), pl.len().alias("k")).sort("event_ts")
    g = g.with_columns((1 + pl.col("sr")).log().cum_sum().exp().alias("close"))
    bars = g.select("event_ts", pl.col("close").alias("open"), pl.col("close").alias("high"), pl.col("close").alias("low"), "close")
    d = daily_measures(bars, "1h")
    k = g.with_columns(pl.col("event_ts").dt.date().alias("date")).group_by("date").agg(pl.col("k").min().alias("members"))
    d = d.join(k, on="date").filter(pl.col("members") >= min_members)
    return contiguous(d.select("date", "ret", "rv", "members"))


def _realtime_c(base: np.ndarray, raw: np.ndarray, min_n: int) -> np.ndarray:
    """c_D = sd(base_<D) / sd(raw_<D) on days before D (expanding), NaN before min_n paired days."""
    out = np.full(len(base), np.nan)
    ok = np.isfinite(base) & np.isfinite(raw)
    b, r = np.where(ok, base, 0.0), np.where(ok, raw, 0.0)
    n = np.cumsum(ok)
    sb, sb2, sr, sr2 = np.cumsum(b), np.cumsum(b * b), np.cumsum(r), np.cumsum(r * r)
    for t in range(1, len(base)):
        k = n[t - 1]
        if k >= min_n:
            vb = sb2[t - 1] / k - (sb[t - 1] / k) ** 2
            vr = sr2[t - 1] / k - (sr[t - 1] / k) ** 2
            out[t] = math.sqrt(vb / vr) if vr > 0 else np.nan
    return out


def _managed_longonly(base: np.ndarray, s2: np.ndarray, c_mode: str, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (managed net daily return, weight). Turnover cost 7.5 bps x |delta w| per day."""
    inv = np.where(np.isfinite(s2) & (s2 > 0), 1.0 / s2, np.nan)
    raw = inv * base
    ok = np.isfinite(raw) & np.isfinite(base)
    if c_mode == "full":
        c = np.full(len(base), np.std(base[ok]) / np.std(raw[ok]))
    else:
        c = _realtime_c(base, raw, min_n)
    w = c * inv
    wz = np.nan_to_num(w)
    cost = np.abs(np.diff(np.r_[0.0, wz])) * COST_BPS / 1e4
    return np.where(np.isfinite(w), w * base - cost, np.nan), w


def _ens_portfolio(syms: list[dict], univ: str, w_by_date: dict[str, dict[date, float]] | None, per_symbol_w: dict[str, dict[date, float]] | None = None,
                   c_scale: float = 1.0) -> pl.DataFrame:
    """EW over symbols of daily ensemble net P&L. w_by_date['__all__'] (common) or per_symbol_w[sym] scales leverage."""
    rows = []
    for s in syms:
        if s["univ"] != univ or s.get("ens") is None:
            continue
        e = s["ens"]
        dts = e["date"].to_list()
        if per_symbol_w is not None:
            wmap = per_symbol_w.get(s["symbol"], {})
        elif w_by_date is not None:
            wmap = w_by_date["__all__"]
        else:
            wmap = None
        w = np.ones(len(dts)) if wmap is None else np.array([wmap.get(d, np.nan) for d in dts]) * c_scale
        valid = np.isfinite(w)
        p = scaled_pnl(e["lev"].to_numpy(), e["fwd_ret"].to_numpy(), np.where(valid, w, 0.0))
        rows.append(pl.DataFrame({"date": dts, "pnl": p, "valid": valid, "sym": s["symbol"]}))
    if not rows:
        return pl.DataFrame({"date": [], "pnl": []})
    d = pl.concat(rows).group_by("sym", "date").agg(pl.col("pnl").sum(), pl.col("valid").all())
    return d.filter(pl.col("valid")).group_by("date").agg(pl.col("pnl").mean()).sort("date")


def vb1(syms: list[dict]) -> dict:
    from ai_trader.features.vol_models_x import har_forecast_x
    from ai_trader.research.evaluate import knowledge_cutoff
    from ai_trader.research.trend_zoo_study import deflated_sharpe

    cut = knowledge_cutoff().date()
    out: dict = {"universes": {}}
    all_series = {}
    for univ in ("L", "T"):
        idx = ew_index_daily(syms, univ)
        rv = idx["rv"].fill_null(float("nan")).to_numpy()
        dts = idx["date"].to_numpy().astype("datetime64[D]")
        extra = calendar_extra(dts) if univ == "T" else None
        f_next = har_forecast_x(rv, extra_regressors=extra, horizon=1, refit_every=30, min_days=MIN_DAYS[univ])
        f_cur = np.r_[np.nan, f_next[:-1]]  # forecast of day D made at the end of D-1
        base_lo = idx["ret"].fill_null(float("nan")).to_numpy()
        dates = idx["date"].to_list()
        # MM-native sigma2: sum of the previous 22 daily squared base returns (all present)
        sq = base_lo**2
        mm_lo = np.full(len(sq), np.nan)
        for t in range(22, len(sq)):
            win = sq[t - 22:t]
            if np.isfinite(win).all():
                mm_lo[t] = win.sum()
        res_u: dict = {"index_days": int(np.isfinite(base_lo).sum()), "first_forecast": str(dates[int(np.argmax(np.isfinite(f_cur)))]) if np.isfinite(f_cur).any() else None,
                       "bases": {}}
        # ---------------- long-only
        variants = {}
        for vname, s2, cm in (("har_fullc", f_cur, "full"), ("har_realtime_c", f_cur, "rt"), ("mm22_fullc", mm_lo, "full")):
            m, w = _managed_longonly(base_lo, s2, cm, MIN_DAYS[univ])
            variants[vname] = (m, base_lo.copy(), w)
        # per-symbol HAR scaling, then EW (base = EW of the same symbol-days)
        per = []
        for s in syms:
            if s["univ"] != univ:
                continue
            d = s["daily"]
            r, fc = d["ret"].fill_null(float("nan")).to_numpy(), d["f_cur"].fill_null(float("nan")).to_numpy()
            m, _w = _managed_longonly(r, fc, "full", MIN_DAYS[univ])
            ok = np.isfinite(m)
            per.append(pl.DataFrame({"date": d["date"].filter(pl.Series(ok)), "m": m[ok], "b": r[ok]}))
        if per:
            pp = pl.concat(per).group_by("date").agg(pl.col("m").mean(), pl.col("b").mean())
            pj = pl.DataFrame({"date": dates}).join(pp, on="date", how="left")
            variants["per_symbol_har_fullc"] = (pj["m"].fill_null(float("nan")).to_numpy(), pj["b"].fill_null(float("nan")).to_numpy(), None)
        res_u["bases"]["long_only"] = _vb1_tables(variants, dates, cut)
        for k, v in variants.items():
            all_series[(univ, "long_only", k)] = (v[0], v[1], dates)

        # ---------------- trend-zoo ensemble (4h, risk units)
        ens_base = _ens_portfolio(syms, univ, None)
        if ens_base.height > 100:
            ej = pl.DataFrame({"date": dates}).join(ens_base, on="date", how="left")
            eb = ej["pnl"].fill_null(float("nan")).to_numpy()
            evar = {}
            # common weight from the index HAR (raw 1/F), c on full sample, then the managed P&L recomputed with costs
            inv = np.where(np.isfinite(f_cur) & (f_cur > 0), 1 / f_cur, np.nan)
            sqe = eb**2
            mm_e = np.full(len(sqe), np.nan)
            for t in range(22, len(sqe)):
                win = sqe[t - 22:t]
                if np.isfinite(win).all():
                    mm_e[t] = win.sum()
            inv_mm = np.where(np.isfinite(mm_e) & (mm_e > 0), 1 / mm_e, np.nan)
            for vname, iv, cm in (("har_fullc", inv, "full"), ("har_realtime_c", inv, "rt"), ("mm22_fullc", inv_mm, "full")):
                raw = iv * eb
                ok = np.isfinite(raw) & np.isfinite(eb)
                c = np.full(len(eb), np.std(eb[ok]) / np.std(raw[ok])) if cm == "full" else _realtime_c(eb, raw, MIN_DAYS[univ])
                w = c * iv
                wmap = {d: float(x) for d, x in zip(dates, w, strict=True) if np.isfinite(x)}
                mp = _ens_portfolio(syms, univ, {"__all__": wmap})
                mj = pl.DataFrame({"date": dates}).join(mp, on="date", how="left")["pnl"].fill_null(float("nan")).to_numpy()
                evar[vname] = (np.where(np.isfinite(w), mj, np.nan), eb.copy(), w)
            # per-symbol: each symbol's own HAR forecast, c_i on the symbol's full sample
            psw = {}
            for s in syms:
                if s["univ"] != univ or s.get("ens") is None:
                    continue
                d = s["daily"]
                fc = d["f_cur"].fill_null(float("nan")).to_numpy()
                own = _ens_portfolio([s], univ, None)
                oj = d.select("date").join(own, on="date", how="left")["pnl"].fill_null(float("nan")).to_numpy()
                ivs = np.where(np.isfinite(fc) & (fc > 0), 1 / fc, np.nan)
                ok = np.isfinite(ivs * oj)
                if ok.sum() < 60:
                    continue
                ci = np.std(oj[ok]) / np.std((ivs * oj)[ok])
                psw[s["symbol"]] = {dd: float(ci * x) for dd, x in zip(d["date"].to_list(), ivs, strict=True) if np.isfinite(x)}
            if psw:
                mp = _ens_portfolio(syms, univ, None, per_symbol_w=psw)
                # base on the same symbol-days
                bs = _ens_portfolio([s for s in syms if s["symbol"] in psw], univ, None, per_symbol_w={k: dict.fromkeys(v, 1.0) for k, v in psw.items()})
                mj = pl.DataFrame({"date": dates}).join(mp, on="date", how="left")["pnl"].fill_null(float("nan")).to_numpy()
                bj = pl.DataFrame({"date": dates}).join(bs, on="date", how="left")["pnl"].fill_null(float("nan")).to_numpy()
                evar["per_symbol_har_fullc"] = (mj, bj, None)
            res_u["bases"]["trend_ensemble"] = _vb1_tables(evar, dates, cut)
            for k, v in evar.items():
                all_series[(univ, "trend_ensemble", k)] = (v[0], v[1], dates)
        out["universes"][univ] = res_u

    # deflated Sharpe of the managed variant series (llm_in_sample), all variants as trials
    srs, ins = {}, {}
    for key, (m, _b, dts) in all_series.items():
        mask = np.array([d < cut for d in dts]) & np.isfinite(m)
        if mask.sum() > 60:
            x = m[mask]
            srs[key] = x.mean() / x.std(ddof=1)
            ins[key] = x
    if len(srs) >= 2:
        tr = np.array(list(srs.values()))
        best = max(srs, key=lambda k: srs[k])
        out["dsr"] = {"n_trials": len(srs), "best": "/".join(best), **deflated_sharpe(ins[best], tr, len(srs))}
        prim = ("L", "long_only", "har_fullc")
        if prim in ins:
            out["dsr_primary_L_long_only"] = deflated_sharpe(ins[prim], tr, len(srs))
    return out


def _vb1_tables(variants: dict, dates: list[date], cut: date) -> dict:
    res = {}
    seg = np.array(["llm_in_sample" if d < cut else "clean_holdout" for d in dates])
    for k, (m, b, w) in variants.items():
        r = {}
        for sname in (*SEGS, "pooled"):
            mask = np.ones(len(dates), bool) if sname == "pooled" else seg == sname
            r[sname] = spanning(m[mask], b[mask])
            if w is not None:
                ww = w[mask & np.isfinite(w)]
                if len(ww):
                    r[sname]["w_median"] = float(np.median(ww))
                    r[sname]["w_p95"] = float(np.quantile(ww, 0.95))
                    r[sname]["turnover_per_day"] = float(np.mean(np.abs(np.diff(ww)))) if len(ww) > 1 else float("nan")
        res[k] = r
    return res


# ====================================================================== H-VB2
def vb2(syms: list[dict]) -> dict:
    from ai_trader.research.evaluate import knowledge_cutoff

    cut = knowledge_cutoff()
    ev_rows, base_rows = [], []
    for s in syms:
        d = s["daily"]
        rv, ret = d["rv"].fill_null(float("nan")).to_numpy(), d["ret"].fill_null(float("nan")).to_numpy()
        fcur, fnext = d["f_cur"].fill_null(float("nan")).to_numpy(), d["f_next"].fill_null(float("nan")).to_numpy()
        n = len(ret)
        r1 = np.r_[ret[1:], np.nan]
        r3 = np.full(n, np.nan)
        if n > 3:
            r3[:-3] = ret[1:-2] + ret[2:-1] + ret[3:]
        dts = d["date"].to_list()
        dec = [datetime(x.year, x.month, x.day, tzinfo=UTC) + timedelta(days=1) for x in dts]
        idx = vol_surprise_events(rv, ret, fcur)
        is_ev = np.zeros(n, bool)
        is_ev[idx] = True
        sg = np.sign(ret)
        with np.errstate(invalid="ignore", divide="ignore"):
            z1 = sg * r1 / np.sqrt(fnext)
            z3 = sg * r3 / np.sqrt(3 * fnext)
            c1 = 2 * COST_BPS / 1e4 / np.sqrt(fnext)
            c3 = 2 * COST_BPS / 1e4 / np.sqrt(3 * fnext)
        vs0, vs1 = d["vs_start"].to_list(), d["vs_end"].to_list()
        for i in range(n):
            if not (np.isfinite(fcur[i]) and np.isfinite(fnext[i]) and np.isfinite(ret[i]) and ret[i] != 0):
                continue
            row = {"sym": s["symbol"], "univ": s["univ"], "date": dts[i], "seg": "llm_in_sample" if dec[i] < cut else "clean_holdout",
                   "z1": z1[i], "z3": z3[i], "z1n": z1[i] - c1[i], "z3n": z3[i] - c3[i], "vs_start": vs0[i], "vs_end": vs1[i],
                   "surprise": rv[i] / fcur[i]}
            (ev_rows if is_ev[i] else base_rows).append(row)
    ev, base = pl.DataFrame(ev_rows), pl.DataFrame(base_rows)
    out = {"n_events_total": ev.height, "segments": {}}
    for sname in (*SEGS, "pooled"):
        e = ev if sname == "pooled" else ev.filter(pl.col("seg") == sname)
        b = base if sname == "pooled" else base.filter(pl.col("seg") == sname)
        so = {}
        for col in ("z1n", "z3n", "z1", "z3"):
            so[col] = date_mean_t(e[col].to_numpy(), e["date"].to_numpy()) if e.height else {}
        so["baseline_all_days_signed"] = {col: date_mean_t(b[col].to_numpy(), b["date"].to_numpy()) for col in ("z1n", "z3n")} if b.height else {}
        so["by_universe"] = {u: {col: date_mean_t(e.filter(pl.col("univ") == u)[col].to_numpy(), e.filter(pl.col("univ") == u)["date"].to_numpy())
                                 for col in ("z1n", "z3n")} for u in ("L", "T") if e.filter(pl.col("univ") == u).height}
        for vcol in ("vs_start", "vs_end"):
            so[f"by_{vcol}"] = {}
            for v in ("low", "normal", "high"):
                g = e.filter(pl.col(vcol) == v)
                if g.height:
                    so[f"by_{vcol}"][v] = {col: date_mean_t(g[col].to_numpy(), g["date"].to_numpy()) for col in ("z1n", "z3n")}
        so["median_surprise_ratio"] = float(e["surprise"].median()) if e.height else float("nan")
        out["segments"][sname] = so
    return out


# ====================================================================== H-VB3
def vb3(syms: list[dict], seed: int = 0) -> dict:
    ev = [s["vb3"].with_columns(pl.lit(s["symbol"]).alias("sym"), pl.lit(s["univ"]).alias("univ")) for s in syms if s.get("vb3") is not None]
    ev = pl.concat(ev).with_columns(_seg_of_ts(pl.col("decision_ts")).alias("seg"), pl.col("decision_ts").dt.date().alias("date"))
    pools = {s["symbol"]: s["vb3_pool"].with_columns(_seg_of_ts(pl.col("decision_ts")).alias("seg")) for s in syms if s.get("vb3_pool") is not None}
    rng = np.random.default_rng(seed)
    out = {"segments": {}}
    for sname in (*SEGS, "pooled"):
        e = ev if sname == "pooled" else ev.filter(pl.col("seg") == sname)
        so = {}
        for u in ("all", "L", "T"):
            g = e if u == "all" else e.filter(pl.col("univ") == u)
            sq, pl_ = g.filter(pl.col("cls") == "squeeze"), g.filter(pl.col("cls") == "plain")
            if sq.height == 0:
                continue
            r = {"squeeze_net": date_mean_t(sq["z_net"].to_numpy(), sq["date"].to_numpy()),
                 "squeeze_gross": date_mean_t(sq["z_gross"].to_numpy(), sq["date"].to_numpy()),
                 "plain_net": date_mean_t(pl_["z_net"].to_numpy(), pl_["date"].to_numpy()),
                 "diff_net": cluster_diff(g["z_net"].to_numpy(), (g["cls"] == "squeeze").to_numpy(), g["date"].to_numpy())}
            # random-entry baseline matched in count per symbol (and segment)
            counts = sq.group_by("sym").len()
            draws = []
            for _ in range(N_RANDOM):
                vals = []
                for sym, k in counts.iter_rows():
                    p = pools.get(sym)
                    if p is None:
                        continue
                    p = p if sname == "pooled" else p.filter(pl.col("seg") == sname)
                    if p.height:
                        vals.append(p["z_net"].to_numpy()[rng.integers(0, p.height, k)])
                if vals:
                    draws.append(float(np.concatenate(vals).mean()))
            if draws:
                dr = np.array(draws)
                r["random_entry_net"] = {"mean": float(dr.mean()), "sd": float(dr.std()), "p5": float(np.quantile(dr, 0.05)), "p95": float(np.quantile(dr, 0.95)),
                                         "frac_ge_squeeze": float((dr >= r["squeeze_net"]["mean"]).mean())}
            so[u] = r
        out["segments"][sname] = so
    return out


# ====================================================================== H-VB4
def fit_nu(syms: list[dict], cut: datetime) -> float:
    zs = []
    for s in syms:
        if s["univ"] != "L":
            continue
        d = s["daily"].select("date", "f_cur").drop_nulls()
        h = s["h1"].with_columns(pl.col("event_ts").dt.date().alias("date")).filter(pl.col("event_ts") < cut).join(d, on="date")
        z = (h["ret"] / (h["f_cur"] / 24).sqrt()).to_numpy()
        zs.append(z[np.isfinite(z)])
    z = np.concatenate(zs)
    rng = np.random.default_rng(1)
    if len(z) > 300_000:
        z = z[rng.choice(len(z), 300_000, replace=False)]
    return fit_t_df(z)


def vb4_symbol(sym: str, univ: str, hours: np.ndarray, ret: np.ndarray, sig: np.ndarray, day_hours: np.ndarray, f_next: np.ndarray,
               nu: float, min_days: int) -> list[dict]:
    """hours: int hour index (since epoch) of each 1h bar; day_hours: hour index of each decision (00:00 after day D);
    f_next: HAR forecast of day D+1 made at the end of D."""
    from ai_trader.features.jump_diffusion import hawkes_intensity

    ok = np.isfinite(ret) & np.isfinite(sig) & (sig > 0)
    if ok.sum() < 24 * min_days:
        return []
    h0 = int(hours[ok].min())
    n = int(hours.max()) - h0 + 1
    valid, flag, thr = np.zeros(n, bool), np.zeros(n), np.full(n, np.nan)
    ix = (hours - h0).astype(int)
    m = ix >= 0
    valid[ix[m]] = ok[m]
    flag[ix[m]] = (ok & (np.abs(np.nan_to_num(ret)) > EXC_K * np.nan_to_num(sig)))[m]
    thr[ix[m]] = np.where(ok, sig, np.nan)[m]
    # month starts (hour index of 00:00 on the 1st)
    t0 = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(hours=h0)
    mstarts = []
    y, mo = t0.year, t0.month
    while True:
        mo += 1
        if mo > 12:
            y, mo = y + 1, 1
        hi = int((datetime(y, mo, 1, tzinfo=UTC) - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() // 3600) - h0
        if hi >= n:
            break
        mstarts.append(hi)
    rows, prev = [], None
    cum_flag, cum_valid = np.cumsum(flag), np.cumsum(valid)
    for j, ms in enumerate(mstarts):
        nxt = mstarts[j + 1] if j + 1 < len(mstarts) else n
        if ms < 24 * min_days or flag[:ms].sum() < 20:
            continue
        mu, a, b, lr = fit_hawkes_grid(flag[:ms], x0=prev)
        prev = (mu, a, b) if a > 0 else None
        lam = hawkes_intensity(flag[:nxt].astype(bool), mu, a, b)
        for dh, fn in zip(day_hours, f_next, strict=True):
            i0 = int(dh) - h0
            if i0 < ms or i0 >= nxt or i0 + HORIZON_H > n:
                continue
            if not valid[i0:i0 + HORIZON_H].all() or not np.isfinite(fn) or fn <= 0 or not np.isfinite(thr[i0]):
                continue
            N = float(flag[i0:i0 + HORIZON_H].sum())
            lh = hawkes_expected_count(float(lam[i0]), mu, a, b, HORIZON_H)
            q = EXC_K * thr[i0] / math.sqrt(fn / 24)
            lhar = HORIZON_H * float(t_exceed_prob(q, nu))
            lc = HORIZON_H * cum_flag[i0 - 1] / max(cum_valid[i0 - 1], 1)
            rows.append({"sym": sym, "univ": univ, "decision_hour": int(dh), "N": N, "lam_hawkes": lh, "lam_har_t": lhar, "lam_const": float(lc),
                         "ls_hawkes": float(poisson_logpmf(N, lh)), "ls_har_t": float(poisson_logpmf(N, lhar)), "ls_const": float(poisson_logpmf(N, lc)),
                         "hk_mu": mu, "hk_alpha": a, "hk_beta": b, "hk_lr": lr})
    return rows


def _vb4_job(args):
    return vb4_symbol(*args)


def vb4(syms: list[dict], workers: int = 4) -> dict:
    from ai_trader.research.evaluate import knowledge_cutoff

    cut = knowledge_cutoff()
    nu = fit_nu(syms, cut)
    jobs = []
    for s in syms:
        h = s["h1"]
        hours = (h["event_ts"].dt.epoch("s") // 3600).to_numpy()
        d = s["daily"].select("date", "f_next").drop_nulls()
        day_hours = ((d["date"].cast(pl.Datetime("us", "UTC")).dt.epoch("s") // 3600) + 24).to_numpy()
        jobs.append((s["symbol"], s["univ"], hours, h["ret"].fill_null(float("nan")).to_numpy(), h["sig_thr"].fill_null(float("nan")).to_numpy(),
                     day_hours, d["f_next"].to_numpy(), nu, MIN_DAYS[s["univ"]]))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for res in ex.map(_vb4_job, jobs) for r in res]
    df = pl.DataFrame(rows).with_columns(
        (pl.lit(datetime(1970, 1, 1, tzinfo=UTC)) + pl.duration(hours=pl.col("decision_hour"))).alias("decision_ts")
    ).with_columns(_seg_of_ts(pl.col("decision_ts")).alias("seg"), pl.col("decision_ts").dt.date().alias("date"),
                   (pl.col("ls_hawkes") - pl.col("ls_har_t")).alias("d_har"), (pl.col("ls_hawkes") - pl.col("ls_const")).alias("d_const"),
                   (pl.col("ls_har_t") - pl.col("ls_const")).alias("d_har_vs_const"))
    out = {"nu": nu, "segments": {}}
    for sname in (*SEGS, "pooled"):
        g = df if sname == "pooled" else df.filter(pl.col("seg") == sname)
        so = {}
        for u in ("all", "L", "T"):
            x = g if u == "all" else g.filter(pl.col("univ") == u)
            if x.height == 0:
                continue
            dd = x["date"].to_numpy()
            so[u] = {"n_symbol_days": x.height, "n_symbols": x["sym"].n_unique(),
                     "hawkes_minus_har_t": date_mean_t(x["d_har"].to_numpy(), dd),
                     "hawkes_minus_const": date_mean_t(x["d_const"].to_numpy(), dd),
                     "har_t_minus_const": date_mean_t(x["d_har_vs_const"].to_numpy(), dd),
                     "mean_N": float(x["N"].mean()), "mean_lam_hawkes": float(x["lam_hawkes"].mean()), "mean_lam_har_t": float(x["lam_har_t"].mean()),
                     "mean_lam_const": float(x["lam_const"].mean()),
                     "corr_N_hawkes": float(np.corrcoef(x["N"].to_numpy(), x["lam_hawkes"].to_numpy())[0, 1]),
                     "corr_N_har_t": float(np.corrcoef(x["N"].to_numpy(), x["lam_har_t"].to_numpy())[0, 1])}
        out["segments"][sname] = so
    last = df.sort("decision_hour").group_by("sym").agg(pl.col("hk_mu").last(), pl.col("hk_alpha").last(), pl.col("hk_beta").last(), pl.col("hk_lr").last())
    out["last_fit_median"] = {"branching_ratio": float((last["hk_alpha"] / last["hk_beta"]).median()), "half_life_h": float((np.log(2) / last["hk_beta"]).median()),
                              "lr": float(last["hk_lr"].median())}
    return out


# ====================================================================== verdicts
def _ok(x: float) -> bool:
    return x is not None and np.isfinite(x)


def primaries(out: dict) -> dict:
    """Per segment the primary statistic, t and one-sided p of each hypothesis. Event-study estimates are the mean of the
    per-date means (the estimator whose NW t is reported); the plain event mean is kept alongside."""
    pr = {}
    for sname in (*SEGS, "pooled"):
        p = {}
        # VB1: best base (by t) per universe; L carries the in-sample test; Bonferroni x2 across the two bases
        v1 = {}
        for u, ru in out["vb1"]["universes"].items():
            cands = [(b, ru["bases"][b]["har_fullc"][sname]) for b in ru["bases"] if "har_fullc" in ru["bases"][b]]
            cands = [(b, r) for b, r in cands if _ok(r.get("t_alpha"))]
            if cands:
                b, r = max(cands, key=lambda c: c[1]["t_alpha"])
                v1[u] = {"base": b, "alpha_ann": r["alpha_ann"], "t": r["t_alpha"], "p": min(1.0, 2 * p_one_sided(r["t_alpha"])),
                         "all": {bb: {"alpha_ann": rr.get("alpha_ann"), "t": rr.get("t_alpha")} for bb, rr in cands}}
        p["H-VB1"] = v1
        s2 = out["vb2"]["segments"][sname].get("z3n", {})
        p["H-VB2"] = {"mean": s2.get("date_mean"), "event_mean": s2.get("mean"), "t": s2.get("t"), "p": p_one_sided(s2.get("t", float("nan"))), "n": s2.get("n_events")}
        s3 = out["vb3"]["segments"][sname].get("all", {})
        if s3:
            tm, td = s3["squeeze_net"].get("t", float("nan")), s3["diff_net"].get("t", float("nan"))
            p["H-VB3"] = {"squeeze_mean": s3["squeeze_net"]["date_mean"], "squeeze_event_mean": s3["squeeze_net"]["mean"], "t_mean": tm, "diff": s3["diff_net"]["diff"], "t_diff": td,
                          "p": max(p_one_sided(tm), p_one_sided(td)), "n": s3["squeeze_net"]["n_events"]}
        s4 = out["vb4"]["segments"][sname].get("all", {}).get("hawkes_minus_har_t", {})
        p["H-VB4"] = {"mean": s4.get("date_mean"), "event_mean": s4.get("mean"), "t": s4.get("t"), "p": p_one_sided(s4.get("t", float("nan"))), "n_dates": s4.get("n_dates")}
        pv = {"H-VB1": p["H-VB1"].get("L", {}).get("p", 1.0) if sname != "clean_holdout" else min([v["p"] for v in v1.values()] or [1.0]),
              "H-VB2": p["H-VB2"]["p"], "H-VB3": p.get("H-VB3", {}).get("p", 1.0), "H-VB4": p["H-VB4"]["p"]}
        p["holm"] = holm(pv)
        pr[sname] = p
    return pr


def verdicts(pr: dict) -> dict[str, dict[str, str]]:
    v: dict[str, dict[str, str]] = {h: {} for h in ("H-VB1", "H-VB2", "H-VB3", "H-VB4")}
    ins, ho = pr["llm_in_sample"], pr["clean_holdout"]
    hi, hh = ins["holm"], ho["holm"]

    def seg_ins(est_pos: bool, t_ok: bool, holm_p: float) -> str:
        if est_pos and t_ok and holm_p < 0.05:
            return "supports"
        if not est_pos:
            return "contradicts"
        return "inconclusive"

    # VB1: in-sample: L, at least one base with alpha > 0 and t > 2
    L = ins["H-VB1"].get("L")
    v["H-VB1"]["llm_in_sample"] = seg_ins(bool(L and L["alpha_ann"] > 0), bool(L and L["t"] > 2), hi["H-VB1"]) if L else "inconclusive"
    hv = ho["H-VB1"]
    if not hv:
        v["H-VB1"]["clean_holdout"] = "inconclusive"
    else:
        pos = all(any((x["alpha_ann"] or -1) > 0 for x in hv[u]["all"].values()) for u in hv)
        neg = all(all((x["alpha_ann"] or 0) <= 0 for x in hv[u]["all"].values()) for u in hv)
        v["H-VB1"]["clean_holdout"] = "supports" if pos else ("contradicts" if neg else "inconclusive")
    # VB2
    a, b = ins["H-VB2"], ho["H-VB2"]
    v["H-VB2"]["llm_in_sample"] = seg_ins(_ok(a["mean"]) and a["mean"] > 0, _ok(a["t"]) and a["t"] > 2, hi["H-VB2"])
    v["H-VB2"]["clean_holdout"] = ("inconclusive" if not _ok(b["mean"]) else "supports" if b["mean"] > 0 else "contradicts")
    # VB3
    a, b = ins.get("H-VB3"), ho.get("H-VB3")
    if a:
        both_pos = a["squeeze_mean"] > 0 and a["diff"] > 0
        v["H-VB3"]["llm_in_sample"] = ("supports" if both_pos and a["t_mean"] > 2 and a["t_diff"] > 2 and hi["H-VB3"] < 0.05
                                       else "contradicts" if (a["squeeze_mean"] <= 0 or a["diff"] <= 0) else "inconclusive")
    else:
        v["H-VB3"]["llm_in_sample"] = "inconclusive"
    if b and _ok(b["squeeze_mean"]) and _ok(b["diff"]):
        v["H-VB3"]["clean_holdout"] = "supports" if (b["squeeze_mean"] > 0 and b["diff"] > 0) else (
            "contradicts" if (b["squeeze_mean"] <= 0 and b["diff"] <= 0) else "inconclusive")
    else:
        v["H-VB3"]["clean_holdout"] = "inconclusive"
    # VB4: difference > 0 with t > 2 in both segments
    for sname, src, hp in (("llm_in_sample", ins, hi), ("clean_holdout", ho, hh)):
        x = src["H-VB4"]
        if not _ok(x["mean"]):
            v["H-VB4"][sname] = "inconclusive"
        elif x["mean"] > 0 and _ok(x["t"]) and x["t"] > 2 and hp["H-VB4"] < 0.05:
            v["H-VB4"][sname] = "supports"
        elif x["mean"] < 0 and _ok(x["t"]) and x["t"] < -2:
            v["H-VB4"][sname] = "contradicts"
        else:
            v["H-VB4"][sname] = "inconclusive" if x["mean"] > 0 else "contradicts"
    for vh in v.values():
        a, b = vh["llm_in_sample"], vh["clean_holdout"]
        vh["pooled"] = "supports" if (a == "supports" and b == "supports") else ("contradicts" if "contradicts" in (a, b) else "inconclusive")
    return v


# ====================================================================== run / record / report
def ensemble_check(syms: list[dict]) -> dict:
    """Max |difference| between the recomputed ensemble daily P&L and the trend-zoo parquet (overlapping dates)."""
    from ai_trader.research.trend_zoo_study import panel_path

    p = panel_path("4h")
    if not p.is_file():
        return {"parquet": "missing"}
    pq = pl.read_parquet(p).filter(pl.col("variant") == "ensemble")
    diffs, n = [], 0
    for s in syms:
        if s["univ"] != "L" or s.get("ens") is None:
            continue
        own = _ens_portfolio([s], "L", None).rename({"pnl": "mine"})
        j = own.join(pq.filter(pl.col("symbol") == s["symbol"]).select("date", "pnl"), on="date")
        j = j.filter(pl.col("date") < pq["date"].max())  # the parquet's last day may be partial
        if j.height:
            diffs.append(float((j["mine"] - j["pnl"]).abs().max()))
            n += j.height
    return {"parquet": str(p), "symbol_days_compared": n, "max_abs_diff": max(diffs) if diffs else None}


def run(as_of: datetime | None = None, cache: Path | None = None, workers: int = 4) -> dict:
    from ai_trader.core.time import now

    as_of = as_of or now()
    syms = load_all(as_of, cache, workers)
    add_forecasts(syms)
    out = {"as_of": as_of.isoformat(), "symbols": {"L": [s["symbol"] for s in syms if s["univ"] == "L"], "T": [s["symbol"] for s in syms if s["univ"] == "T"]},
           "ensemble_check": ensemble_check(syms)}
    out["vb1"] = vb1(syms)
    out["vb2"] = vb2(syms)
    out["vb3"] = vb3(syms)
    out["vb4"] = vb4(syms, workers)
    out["primaries"] = primaries(out)
    out["verdicts"] = verdicts(out["primaries"])
    return out


DATA_NOTE = ("bybit 1h/4h market_frame (funding/oi off), daily realised measures = features.realized.daily_measures; universe L = "
             "trend_zoo_study.UNIVERSE (23 crypto), T = 16 TradFi perps (holdout-dominated, histories 115-555 days); module "
             "research/vol_signal_study.py; report research/reports/vol_signals_2026-11-10.{json,md}")


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record

    pr, v = out["primaries"], out["verdicts"]
    notes = {
        "H-VB1": "factor-level EW long-only and EW trend-zoo ensemble, w = c / HAR forecast of the EW index RV; c full-sample (alpha t invariant to c); 7.5 bps per side on weight changes; HAC spanning alpha. Variants (real-time c, MM 22d RV, per-symbol) in the report. T is holdout-only.",
        "H-VB2": "RV_t > 1.5 HAR forecast and |ret_t| > 1 sigma_f; signed next-3d return / sqrt(3 F_{t+1|t}) net of 2 x 7.5 bps; declustered 3 days; NW t on date means; pooled L+T.",
        "H-VB3": "4h BB(20,2) inside KC(EMA20,1.5 ATR20) >= 6 bars, first close outside BB within 20 bars of release; signed 6-bar return net of costs in sigma units; diff vs fresh non-squeeze BB breakouts, date-clustered OLS; random-entry baseline in report.",
        "H-VB4": "1h |r| > 2 sigma_risk (t-1) exceedances; per-symbol Hawkes MLE (expanding, monthly refit) vs Poisson rate from HAR daily variance /24 under unit-variance Student-t (df fitted on L in-sample); paired next-24h log-score diff, NW t on date means.",
    }
    for h in ("H-VB1", "H-VB2", "H-VB3", "H-VB4"):
        for sname in (*SEGS, "pooled"):
            m = {"primary": pr[sname].get(h), "holm_p": pr[sname]["holm"].get(h)}
            if sname == "pooled":
                m["segment_verdicts"] = {k: v[h][k] for k in SEGS}
            extra = " Holdout includes all TradFi-perp evidence (holdout-only data)." if sname != "llm_in_sample" else ""
            record(h, sname, v[h][sname], json.loads(json.dumps(m, default=float)), note=notes[h] + extra, data=DATA_NOTE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    out = run(cache=Path(a.cache) if a.cache else None, workers=min(4, a.workers))
    s = json.dumps(out, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(s, encoding="utf-8")
    print(json.dumps({"verdicts": out["verdicts"], "primaries": out["primaries"]}, indent=1, default=str))
    if a.record:
        record_all(out)


if __name__ == "__main__":
    main()
