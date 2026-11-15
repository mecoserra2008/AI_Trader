"""Daily variance forecasters for the volatility tournament (H-VM1..H-VM5): per-asset and pooled-panel HAR family,
GARCH family filters, a stochastic-volatility state-space model, and the Model Confidence Set.

Conventions (all forecasts are made at the END of day t, i.e. known at t+1 00:00 UTC):
  * inputs are the daily realised measures of `features.realized` (rv, rs_pos, rs_neg, rq, jv, gk, pk, ret) on a gap-aware
    calendar (missing days are NaN);
  * HAR windows are calendar days 1 / 7 / 30 (as `vol_models_x`, crypto and Bybit TradFi perps quote 7 days a week); the
    weekly / monthly means tolerate missing days (>= 5 of 7, >= 20 of 30 present);
  * target for horizon h: mean RV_{t+1..t+h} (NaN if any day is missing). Level models regress it directly (direct
    multi-step); log models regress log of it and back-transform with exp(fit + s^2/2) (lognormal mean, as H-VCL1);
  * expanding window, refit every `refit_every` days; a fit at day d uses only pairs (x_j, y_j) with j + h <= d;
  * insanity filter for level models (Bollerslev, Patton & Quaedvlieg 2016): a forecast outside the [min, max] of the
    training targets is replaced by the training mean (OLS in levels can go negative or explode).

Panel models pool slopes across symbols with symbol fixed effects (within estimator). Level panels first divide each
symbol's target and regressors by c_s^p, c_s = the symbol's training-mean target (p = 1 for variance-like columns, 2 for
the HAR-Q interaction RV_d * sqrt(RQ_d)), so slopes are pooled in scale-free units and high-vol assets don't dominate.
With one symbol the panel forecast equals the per-asset forecast (OLS is invariant to column scaling).

GARCH family (arch fits on 100 x returns, Student-t): filtered here in raw units from the series start with the latest
parameters (as arch's fixed-parameter filter would); GARCH/GJR h-step means are analytic, EGARCH is Monte Carlo.

SV: y_t = log(r_t^2 + c s^2) - c s^2 / (r_t^2 + c s^2) (Fuller's inlier-robust transform, c = 0.02, s^2 the expanding
mean of r^2), y_t = h_t + LOGCHI2_MEAN + eps_t, eps ~ (0, pi^2/2) treated as Gaussian (QML), h_{t+1} = mu + phi (h_t - mu)
+ eta_t, eta ~ N(0, s_eta^2). Kalman filter, QML by L-BFGS-B. Variance forecast E[sigma^2_{t+k}] = exp(m_k + P_k / 2).
The 'rv' variant uses y_t = log RV_t with free measurement variance H and mean 0: E[RV_{t+k}] = exp(m_k + P_k/2 + H/2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import polars as pl

WIN_W, MIN_W = 7, 5
WIN_M, MIN_M = 30, 20
LOGCHI2_MEAN = -1.2703628454614782  # E log chi2_1 = digamma(1/2) + log 2
LOGCHI2_VAR = math.pi**2 / 2
EPS = 1e-12

# ------------------------------------------------------------------ helpers


def trailing_nanmean(x: np.ndarray, w: int, min_n: int) -> np.ndarray:
    """Mean of the finite values in x[t-w+1..t]; NaN when fewer than min_n are finite."""
    s = pl.Series(np.asarray(x, dtype=float)).fill_nan(None)
    return s.rolling_mean(w, min_samples=min_n).to_numpy().astype(float)


def forward_mean(x: np.ndarray, h: int) -> np.ndarray:
    """mean(x[t+1..t+h]); NaN if the window runs past the end or holds a NaN."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n > h:
        c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
        bad = np.concatenate([[0], np.cumsum(~np.isfinite(x))])
        t = np.arange(n - h)
        s = c[t + h + 1] - c[t + 1]
        nb = bad[t + h + 1] - bad[t + 1]
        out[: n - h] = np.where(nb == 0, s / h, np.nan)
    return out


def safe_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(x), np.log(np.maximum(x, EPS)), np.nan)


def qlike(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE V/F - log(V/F) - 1 (same definition as vol_models_x.qlike)."""
    from ai_trader.features.vol_models_x import qlike as _q

    return _q(realised, forecast)


def mse_log(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    d = safe_log(realised) - safe_log(forecast)
    return np.where(np.isfinite(d), d * d, np.nan)


# ------------------------------------------------------------------ HAR designs

LEVEL_MODELS = ("har", "harq", "shar", "harj", "har_gk", "har_pk", "har_rvgk")
LOG_MODELS = ("loghar",)
PANEL_LEVEL = ("har", "harq", "shar", "harj", "har_gk", "har_rvgk")
PANEL_LOG = ("loghar", "loghar_f")


def _wm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return trailing_nanmean(x, WIN_W, MIN_W), trailing_nanmean(x, WIN_M, MIN_M)


def design(name: str, a: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Regressor matrix (T, k) without intercept, and per-column scale powers p (level panels divide by c^p).

    a: rv, rs_pos, rs_neg, rq, jv, gk, pk (daily arrays); for 'loghar_f' also f (factor, log units) and is_crypto (bool).
    """
    rv = a["rv"]
    if name in ("har", "harq", "shar", "harj", "har_rvgk"):
        rw, rm = _wm(rv)
    if name == "har":
        cols, p = [rv, rw, rm], [1, 1, 1]
    elif name == "harq":
        cols, p = [rv, rv * np.sqrt(a["rq"]), rw, rm], [1, 2, 1, 1]
    elif name == "shar":
        cols, p = [a["rs_pos"], a["rs_neg"], rw, rm], [1, 1, 1, 1]
    elif name == "harj":
        cols, p = [rv, rw, rm, a["jv"]], [1, 1, 1, 1]
    elif name in ("har_gk", "har_pk"):
        g = a["gk" if name == "har_gk" else "pk"]
        gw, gm = _wm(g)
        cols, p = [g, gw, gm], [1, 1, 1]
    elif name == "har_rvgk":
        g = a["gk"]
        gw, gm = _wm(g)
        cols, p = [rv, rw, rm, g, gw, gm], [1] * 6
    elif name in ("loghar", "loghar_f"):
        lrv = safe_log(rv)
        lw, lm = _wm(lrv)
        cols, p = [lrv, lw, lm], [0, 0, 0]
        if name == "loghar_f":
            f = a["f"]
            fw, fm = _wm(f)
            cr = np.asarray(a["is_crypto"], dtype=float) * np.ones(len(rv))
            tr = 1.0 - cr
            cols += [f * cr, fw * cr, fm * cr, f * tr, fw * tr, fm * tr]
            p += [0] * 6
    else:
        raise KeyError(name)
    return np.column_stack(cols).astype(float), np.asarray(p, dtype=float)


def is_log(name: str) -> bool:
    return name.startswith("loghar")


def target(rv: np.ndarray, h: int, log: bool) -> np.ndarray:
    y = forward_mean(np.maximum(np.asarray(rv, dtype=float), EPS), h)
    return safe_log(y) if log else y


def _refit_days(first: int, n: int, min_days: int, every: int) -> list[int]:
    return list(range(first + min_days, n, every))


# ------------------------------------------------------------------ per-asset expanding-window OLS


def per_asset_forecast(X: np.ndarray, y: np.ndarray, h: int, *, log: bool, refit_every: int = 30, min_days: int = 180,
                       min_obs: int = 30, first: int | None = None) -> np.ndarray:
    """Causal forecast at each origin t of the h-day target (level: mean RV; log: log of it, returned as a variance)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    rowok = np.isfinite(X).all(axis=1)
    if first is None:
        fin = np.flatnonzero(rowok | np.isfinite(y))
        if not len(fin):
            return np.full(n, np.nan)
        first = int(fin[0])
    Z = np.column_stack([np.ones(n), X])
    out = np.full(n, np.nan)
    coef = None
    s2 = lo = hi = ybar = 0.0
    refits = set(_refit_days(first, n, min_days, refit_every))
    pending = False
    for d in range(first, n):
        if d in refits or (pending and coef is None):
            pending = True
            top = d - h + 1  # rows j <= d - h
            ok = rowok[first:top] & np.isfinite(y[first:top]) if top > first else np.zeros(0, bool)
            if ok.sum() >= max(min_obs, Z.shape[1] + 1):
                Zj, yj = Z[first:top][ok], y[first:top][ok]
                coef, *_ = np.linalg.lstsq(Zj, yj, rcond=None)
                e = yj - Zj @ coef
                s2, lo, hi, ybar = float(np.mean(e * e)), float(yj.min()), float(yj.max()), float(yj.mean())
                pending = False
        if coef is not None and rowok[d]:
            f = float(Z[d] @ coef)
            out[d] = math.exp(f + 0.5 * s2) if log else (f if lo <= f <= hi else ybar)
    return out


# ------------------------------------------------------------------ panel (pooled slopes, symbol fixed effects)


def panel_forecast(Xs: Sequence[np.ndarray], ys: Sequence[np.ndarray], h: int, *, log: bool, powers: np.ndarray | None = None,
                   refit_every: int = 30, start: int = 60, min_pairs: int = 30, refit_days: Sequence[int] | None = None) -> np.ndarray:
    """Xs[s]: (T, k) regressors and ys[s]: (T,) targets of symbol s on a common day grid. Returns (S, T) forecasts.

    At each refit day d (start, start + refit_every, ...) symbols with >= min_pairs complete pairs j <= d - h enter the
    within (fixed-effects) regression. A symbol is forecast from the latest refit that included it."""
    S = len(Xs)
    T = len(ys[0])
    k = Xs[0].shape[1]
    pw = np.zeros(k) if powers is None or log else np.asarray(powers, dtype=float)
    X3 = np.stack([np.asarray(x, dtype=float) for x in Xs])  # (S, T, k)
    Y = np.stack([np.asarray(v, dtype=float) for v in ys])  # (S, T)
    rowok = np.isfinite(X3).all(axis=2)
    pairok = rowok & np.isfinite(Y)
    out = np.full((S, T), np.nan)
    days = list(refit_days) if refit_days is not None else list(range(start, T, refit_every))
    state: dict[int, tuple] = {}
    b = None
    s2 = 0.0
    bounds = days + [T]
    for i, d in enumerate(days):
        top = d - h + 1
        if top <= 0:
            continue
        xs, yss, meta = [], [], {}
        for s in range(S):
            ok = pairok[s, :top]
            m = int(ok.sum())
            if m < max(min_pairs, 2):
                continue
            yj = Y[s, :top][ok]
            Xj = X3[s, :top][ok]
            c = 1.0 if log else float(np.mean(yj))
            if not (np.isfinite(c) and c > 0):
                continue
            sc = c ** pw
            Xn, yn = Xj / sc, yj / c
            xm, ym = Xn.mean(axis=0), yn.mean()
            xs.append(Xn - xm)
            yss.append(yn - ym)
            meta[s] = (c, sc, xm, ym, float(yj.min()), float(yj.max()))
        if not meta:
            continue
        Xw, yw = np.vstack(xs), np.concatenate(yss)
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        e = yw - Xw @ b
        s2 = float(np.mean(e * e))
        for s, (c, sc, xm, ym, lo, hi) in meta.items():
            state[s] = (c, sc, ym - xm @ b, lo, hi, b, s2)
        lo_t, hi_t = d, bounds[i + 1]
        for s, (c, sc, a_s, lo, hi, bb, ss2) in state.items():
            rows = np.arange(lo_t, hi_t)
            rows = rows[rowok[s, lo_t:hi_t]]
            if not len(rows):
                continue
            f = a_s + (X3[s, rows] / sc) @ bb
            if log:
                out[s, rows] = np.exp(f + 0.5 * ss2)
            else:
                fl = c * f
                out[s, rows] = np.where((fl >= lo) & (fl <= hi), fl, c)  # c = the symbol's training-mean target
    return out


# ------------------------------------------------------------------ EWMA and robust_bc aggregation


def ewma_daily(r: np.ndarray, lam: float = 0.94, warmup: int = 30) -> np.ndarray:
    """Variance forecast for day t+1 made at the end of day t: v <- lam v + (1 - lam) r_t^2 (missing r: v unchanged)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    fin = np.flatnonzero(np.isfinite(r))
    if len(fin) <= warmup:
        return out
    v = float(np.mean(r[fin[:warmup]] ** 2))
    for t in range(fin[0], n):
        if np.isfinite(r[t]):
            v = lam * v + (1 - lam) * r[t] * r[t]
        if t >= fin[warmup - 1]:
            out[t] = v
    return out


# ------------------------------------------------------------------ GARCH family

_SCALE = 100.0
ARCH_SPECS = {"garch": dict(vol="GARCH", p=1, o=0, q=1), "gjr": dict(vol="GARCH", p=1, o=1, q=1), "egarch": dict(vol="EGARCH", p=1, o=1, q=1)}


def fit_arch(r: np.ndarray, kind: str, start: dict | None = None) -> dict | None:
    """Student-t GARCH / GJR / EGARCH(1,1) with constant mean on 100 x r; parameters back in raw units."""
    from arch import arch_model

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    am = arch_model(r * _SCALE, mean="Constant", dist="t", rescale=False, **ARCH_SPECS[kind])
    sv = None
    if start is not None:
        sv = np.array([start["mu"] * _SCALE, start["omega_s"], start["alpha"]] + ([start["gamma"]] if kind != "garch" else [])
                      + [start["beta"], start["nu"]])
    try:
        res = am.fit(disp="off", starting_values=sv, options={"maxiter": 300})
    except Exception:  # noqa: BLE001 - optimiser failure: keep previous parameters
        return None
    p = res.params
    if not np.all(np.isfinite(p.values)):
        return None
    om = float(p["omega"])
    return {"loglik": float(res.loglikelihood), "mu": float(p["mu"]) / _SCALE, "omega_s": om,
            "omega": om / _SCALE**2 if kind != "egarch" else om - 2 * math.log(_SCALE) * (1 - float(p["beta[1]"])),
            "alpha": float(p["alpha[1]"]), "gamma": float(p["gamma[1]"]) if kind != "garch" else 0.0,
            "beta": float(p["beta[1]"]), "nu": float(p["nu"]), "converged": bool(res.convergence_flag == 0)}


def arch_sane(pr: dict | None, kind: str) -> bool:
    """Reject degenerate optima (seen on short samples): non-stationary GARCH/GJR, |beta| >= 1 or exploding ARCH terms in
    EGARCH, nu <= 2.05. A rejected refit keeps the previous parameters."""
    if pr is None or not all(np.isfinite(v) for k, v in pr.items() if isinstance(v, float)) or pr["nu"] <= 2.05:
        return False
    if kind == "egarch":
        return abs(pr["beta"]) < 0.9999 and abs(pr["alpha"]) < 2.0 and abs(pr["gamma"]) < 2.0
    return pr["alpha"] >= 0 and pr["beta"] >= 0 and pr["alpha"] + 0.5 * pr["gamma"] + pr["beta"] < 0.9999


def arch_path_sane(r: np.ndarray, pr: dict, kind: str, band: float = 1e3) -> bool:
    """The in-sample filtered variance must stay within [1/band, band] x the sample variance (catches EGARCH optima
    that pass the parameter checks but collapse the variance path to ~0)."""
    s = arch_filter(r, pr, kind)
    v = float(np.mean((r - np.mean(r)) ** 2))
    return bool(np.all(np.isfinite(s)) and s.min() > v / band and s.max() < v * band)


def arch_filter(r: np.ndarray, pr: dict, kind: str, backcast_n: int = 30) -> np.ndarray:
    """sig2_next[t] = conditional variance of r_{t+1} given r_..t (r gap-free). EGARCH: also used for MC."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    e = r - pr["mu"]
    bc = float(np.mean(e[: min(backcast_n, n)] ** 2))
    out = np.empty(n)
    if kind == "egarch":
        c = math.sqrt(2 / math.pi)
        lv = math.log(bc)
        om, al, ga, be = pr["omega"], pr["alpha"], pr["gamma"], pr["beta"]
        for t in range(n):
            z = e[t] / math.exp(0.5 * lv)
            lv = om + al * (abs(z) - c) + ga * z + be * lv
            lv = min(max(lv, -50.0), 5.0)
            out[t] = math.exp(lv)
        return out
    from scipy.signal import lfilter

    neg = (e < 0).astype(float)
    x = pr["omega"] + pr["alpha"] * e * e + pr["gamma"] * e * e * neg  # drives sig2_{t+1}
    # sig2_{t+1} = x_t + beta sig2_t, sig2_0 = bc
    zi = np.array([pr["beta"] * bc])
    out, _ = lfilter([1.0], [1.0, -pr["beta"]], x, zi=zi)
    return out


def arch_multi(sig2_1: np.ndarray, pr: dict, kind: str, h: int, n_mc: int = 1000, seed: int = 0) -> np.ndarray:
    """Mean over k = 1..h of E[sigma^2_{t+k}] given sigma^2_{t+1} = sig2_1."""
    if h == 1:
        return sig2_1
    if kind in ("garch", "gjr"):
        rho = pr["alpha"] + 0.5 * pr["gamma"] + pr["beta"]
        acc = np.zeros_like(sig2_1)
        s = sig2_1.copy()
        for _ in range(h):
            acc += s
            s = pr["omega"] + rho * s
        return acc / h
    rng = np.random.default_rng(seed)
    nu = pr["nu"]
    z = rng.standard_t(nu, size=(h - 1, n_mc)) * math.sqrt((nu - 2) / nu)
    c = math.sqrt(2 / math.pi)
    lv = np.log(np.maximum(sig2_1, 1e-300))[:, None] * np.ones((1, n_mc))
    acc = sig2_1.copy()
    for k in range(h - 1):
        lv = pr["omega"] + pr["alpha"] * (np.abs(z[k]) - c) + pr["gamma"] * z[k] + pr["beta"] * lv
        lv = np.clip(lv, -50.0, 5.0)
        acc = acc + np.exp(lv).mean(axis=1)
    return acc / h


def arch_forecast(r: np.ndarray, kind: str, horizons: Sequence[int] = (1, 5), refit_every: int = 30, min_days: int = 180,
                  n_mc: int = 1000, seed: int = 0) -> dict[int, np.ndarray]:
    """Expanding-window refit every `refit_every` observations from `min_days`; r is gap-free (observed days only)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    out = {h: np.full(n, np.nan) for h in horizons}
    prev = None
    days = list(range(min_days, n, refit_every))
    for i, d in enumerate(days):
        cands = [c for c in (fit_arch(r[: d + 1], kind, start=prev) if prev else None, fit_arch(r[: d + 1], kind)) if arch_sane(c, kind)]
        cands = [c for c in sorted(cands, key=lambda c: -c["loglik"]) if arch_path_sane(r[: d + 1], c, kind)]
        pr = cands[0] if cands else prev
        if pr is None:
            continue
        prev = pr
        hi = days[i + 1] if i + 1 < len(days) else n
        s1 = arch_filter(r[:hi], pr, kind)[d:hi]
        for h in horizons:
            out[h][d:hi] = arch_multi(s1, pr, kind, h, n_mc=n_mc, seed=seed + i)
    return out


# ------------------------------------------------------------------ stochastic volatility (QML Kalman)


def fuller_log_r2(r: np.ndarray, c: float = 0.02) -> np.ndarray:
    """y_t = log(r^2 + c s^2) - c s^2 / (r^2 + c s^2), s^2 = expanding mean of r^2 up to t (causal)."""
    r = np.asarray(r, dtype=float)
    r2 = r * r
    s2 = np.cumsum(r2) / np.arange(1, len(r) + 1)
    k = c * s2
    return np.log(r2 + k) - k / (r2 + k)


def sv_filter(y: np.ndarray, mu: float, phi: float, s_eta: float, c: float, H: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Kalman filter for y_t = h_t + c + eps (var H), h_{t+1} = mu + phi (h_t - mu) + eta (var s_eta^2).
    Returns m_next[t], P_next[t] = mean / variance of h_{t+1} given y_..t, and the Gaussian log likelihood."""
    n = len(y)
    q = s_eta * s_eta
    m = mu
    P = q / max(1 - phi * phi, 1e-6)
    m_next = np.empty(n)
    P_next = np.empty(n)
    ll = 0.0
    for t in range(n):
        yt = y[t]
        if yt == yt:  # finite
            F = P + H
            v = yt - m - c
            K = P / F
            ll -= 0.5 * (math.log(F) + v * v / F)
            m = m + K * v
            P = P * (1 - K)
        m = mu + phi * (m - mu)
        P = phi * phi * P + q
        m_next[t] = m
        P_next[t] = P
    return m_next, P_next, ll - 0.5 * n * math.log(2 * math.pi)


def sv_fit(y: np.ndarray, measurement: str = "r2", start: np.ndarray | None = None) -> dict:
    """QML estimates. measurement 'r2': c = LOGCHI2_MEAN, H = pi^2/2 fixed; 'rv': c = 0, H free."""
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    yf = y[np.isfinite(y)]
    c = LOGCHI2_MEAN if measurement == "r2" else 0.0

    def unpack(th):
        mu = th[0]
        phi = math.tanh(th[1])
        s_eta = math.exp(th[2])
        H = LOGCHI2_VAR if measurement == "r2" else math.exp(th[3])
        return mu, phi, s_eta, H

    def nll(th):
        mu, phi, s_eta, H = unpack(th)
        return -sv_filter(y, mu, phi, s_eta, c, H)[2] / len(yf)

    if start is None:
        start = np.array([float(np.mean(yf)) - c, math.atanh(0.95), math.log(0.2)] + ([math.log(max(np.var(yf) * 0.3, 1e-3))] if measurement == "rv" else []))
    bnds = [(None, None), (math.atanh(-0.5), math.atanh(0.9995)), (math.log(1e-3), math.log(3.0))] + ([(math.log(1e-4), math.log(20.0))] if measurement == "rv" else [])
    res = minimize(nll, start, method="L-BFGS-B", bounds=bnds)
    mu, phi, s_eta, H = unpack(res.x)
    return {"mu": mu, "phi": phi, "s_eta": s_eta, "H": H, "c": c, "theta": res.x, "converged": bool(res.success), "measurement": measurement}


def sv_multi(m1: np.ndarray, P1: np.ndarray, pr: dict, h: int) -> np.ndarray:
    """Mean over k=1..h of E[exp(h_{t+k})] (times exp(H/2) for the log-RV measurement)."""
    mu, phi, q = pr["mu"], pr["phi"], pr["s_eta"] ** 2
    extra = 0.5 * pr["H"] if pr["measurement"] == "rv" else 0.0
    acc = np.zeros_like(m1)
    m, P = m1.copy(), P1.copy()
    for _ in range(h):
        acc += np.exp(m + 0.5 * P + extra)
        m = mu + phi * (m - mu)
        P = phi * phi * P + q
    return acc / h


def sv_forecast(y: np.ndarray, measurement: str = "r2", horizons: Sequence[int] = (1, 5), refit_every: int = 30, min_days: int = 180,
                calibrate_target: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Expanding-window QML SV forecasts on a gap-free series y. If calibrate_target (e.g. r^2) is given, also returns
    level-calibrated forecasts under keys -h: the forecast times mean(target_j) / mean(forecast_{j-1}) over the training window."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out: dict[int, np.ndarray] = {h: np.full(n, np.nan) for h in horizons}
    if calibrate_target is not None:
        out.update({-h: np.full(n, np.nan) for h in horizons})
    days = list(range(min_days, n, refit_every))
    th = None
    for i, d in enumerate(days):
        pr = sv_fit(y[: d + 1], measurement, start=th)
        th = pr["theta"]
        hi = days[i + 1] if i + 1 < len(days) else n
        m1, P1, _ = sv_filter(y[:hi], pr["mu"], pr["phi"], pr["s_eta"], pr["c"], pr["H"])
        k = 1.0
        if calibrate_target is not None:
            f_in = sv_multi(m1[:d], P1[:d], pr, 1)  # forecasts of days 1..d made at 0..d-1
            tg = np.asarray(calibrate_target, dtype=float)[1 : d + 1]
            ok = np.isfinite(f_in) & np.isfinite(tg)
            k = float(np.mean(tg[ok]) / np.mean(f_in[ok])) if ok.sum() > 30 else 1.0
        for h in horizons:
            f = sv_multi(m1[d:hi], P1[d:hi], pr, h)
            out[h][d:hi] = f
            if calibrate_target is not None:
                out[-h][d:hi] = f * k
    return out


def simulate_sv(n: int, mu: float = -8.0, phi: float = 0.95, s_eta: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (r, h): r_t = exp(h_t / 2) z_t, h an AR(1) around mu."""
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = mu + s_eta / math.sqrt(1 - phi * phi) * rng.standard_normal()
    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + s_eta * rng.standard_normal()
    return np.exp(h / 2) * rng.standard_normal(n), h


# ------------------------------------------------------------------ Model Confidence Set (Hansen, Lunde & Nason 2011)


def _block_means(L: np.ndarray, block: int, B: int, seed: int) -> np.ndarray:
    """Circular moving-block bootstrap means, (B, M)."""
    T, M = L.shape
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    ext = np.vstack([L, L[:block]])
    C = np.vstack([np.zeros((1, M)), np.cumsum(ext, axis=0)])
    starts = rng.integers(0, T, size=(B, nb))
    sums = (C[starts + block] - C[starts]).sum(axis=1)
    return sums / (nb * block)


def mcs(losses: np.ndarray, alpha: float = 0.10, block: int = 10, B: int = 1000, stat: str = "Tmax", seed: int = 0) -> dict:
    """losses: (T, M), rows = dates (common sample), columns = models. Returns included mask, MCS p-values and the
    elimination order. stat 'Tmax' (t of each model vs the set average) or 'TR' (max pairwise |t|)."""
    L = np.asarray(losses, dtype=float)
    T, M = L.shape
    mbar = L.mean(axis=0)
    boot = _block_means(L, block, B, seed)  # (B, M)
    alive = list(range(M))
    pvals = np.ones(M)
    order: list[int] = []
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        if stat == "Tmax":
            d = mbar[a] - mbar[a].mean()
            db = boot[:, a] - boot[:, a].mean(axis=1, keepdims=True)
            dev = db - d
            se = np.sqrt(np.mean(dev * dev, axis=0))
            se = np.where(se > 0, se, np.inf)
            t = d / se
            Tstat = t.max()
            Tb = (dev / se).max(axis=1)
            elim = int(a[int(np.argmax(t))])
        else:
            d = mbar[a][:, None] - mbar[a][None, :]
            db = boot[:, a][:, :, None] - boot[:, a][:, None, :]
            dev = db - d[None]
            se = np.sqrt(np.mean(dev * dev, axis=0))
            np.fill_diagonal(se, np.inf)
            se = np.where(se > 0, se, np.inf)
            t = d / se
            Tstat = np.abs(t).max()
            Tb = np.abs(dev / se[None]).reshape(B, -1).max(axis=1)
            elim = int(a[int(np.argmax(t.max(axis=1)))])
        p = float(np.mean(Tb >= Tstat))
        running = max(running, p)
        if p >= alpha:
            break
        pvals[elim] = running
        order.append(elim)
        alive.remove(elim)
    for i in alive:
        pvals[i] = running if len(alive) > 1 else 1.0
    inc = np.zeros(M, dtype=bool)
    inc[alive] = True
    return {"included": inc, "pvalues": pvals, "eliminated": order, "mean_loss": mbar}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone)."""
    items = sorted(((p if np.isfinite(p) else 1.0, k) for k, p in pvals.items()))
    m = len(items)
    out, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


__all__ = [
    "ARCH_SPECS", "LEVEL_MODELS", "LOG_MODELS", "LOGCHI2_MEAN", "LOGCHI2_VAR", "PANEL_LEVEL", "PANEL_LOG",
    "arch_filter", "arch_forecast", "arch_multi", "arch_path_sane", "arch_sane", "design", "ewma_daily", "fit_arch", "forward_mean", "fuller_log_r2", "holm",
    "is_log", "mcs", "mse_log", "panel_forecast", "per_asset_forecast", "qlike", "safe_log", "simulate_sv", "sv_filter", "sv_fit",
    "sv_forecast", "sv_multi", "target", "trailing_nanmean",
]
