"""Local-linear-trend (LLT) state-space tools: Kalman filter with time-varying variances, steady-state gain,
and a concentrated diffuse maximum-likelihood fit (H-KF1, H-KF2).

Model (Harvey 1989, Durbin & Koopman 2012 ch. 3):
    y_t      = mu_t + eps_t,                 eps_t  ~ N(0, R_t)
    mu_{t+1} = mu_t + beta_t + eta_t,        eta_t  ~ N(0, Q_eta_t)
    beta_{t+1} = beta_t + zeta_t,            zeta_t ~ N(0, Q_zeta_t)
With Q_eta = 0 this is the "smooth trend" (integrated random walk) model whose steady-state smoother is the
Hodrick-Prescott filter with lambda = R / Q_zeta.

Everything here is causal: the filtered state at t uses y_0..y_t only, and the recursion starts from the first
observation of whatever array it is given, so truncating the input never changes earlier outputs.

The inner loops are plain-float Python (2x2 algebra written out): ~1-2 us per step, fast enough for the research
panel (no numba in the environment).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterOut:
    level: np.ndarray      # filtered mu_{t|t}
    slope: np.ndarray      # filtered beta_{t|t}
    slope_sd: np.ndarray   # sqrt(P22_{t|t})
    innov: np.ndarray      # v_t = y_t - mu_{t|t-1} (nan where y_t is missing)
    innov_var: np.ndarray  # F_t


def _as_arr(x, n: int) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.full(n, float(a)) if a.ndim == 0 else a


def llt_filter(y, r, q_zeta, q_eta=0.0, p0_level: float | None = None, p0_slope: float = 1.0) -> FilterOut:
    """Kalman filter for the LLT model. `r`, `q_zeta`, `q_eta` are scalars or arrays aligned with y
    (variance of the noise entering at step t: R_t for y_t, Q_t for the transition t-1 -> t).

    Initialisation (causal, no look-ahead): at the first finite y, mu = y, P11 = R (or p0_level), beta = 0 with a
    large slope variance p0_slope (relative to slope magnitudes ~1e-3 per bar for log prices this is diffuse).
    Missing y (nan) -> predict only."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    r, qz, qe = _as_arr(r, n), _as_arr(q_zeta, n), _as_arr(q_eta, n)
    lev = np.full(n, np.nan)
    slo = np.full(n, np.nan)
    ssd = np.full(n, np.nan)
    inn = np.full(n, np.nan)
    fv = np.full(n, np.nan)
    yl, rl, qzl, qel = y.tolist(), r.tolist(), qz.tolist(), qe.tolist()
    started = False
    m = b = p11 = p12 = p22 = 0.0
    for t in range(n):
        yt = yl[t]
        if not started:
            if yt == yt and rl[t] == rl[t] and rl[t] > 0:  # finite
                m, b = yt, 0.0
                p11 = p0_level if p0_level is not None else rl[t]
                p12, p22 = 0.0, p0_slope
                started = True
                lev[t], slo[t], ssd[t] = m, b, math.sqrt(p22)
            continue
        # predict
        qzt = qzl[t] if qzl[t] == qzl[t] else 0.0
        qet = qel[t] if qel[t] == qel[t] else 0.0
        m = m + b
        a11 = p11 + 2.0 * p12 + p22 + qet
        a12 = p12 + p22
        a22 = p22 + qzt
        rt = rl[t]
        if yt == yt and rt == rt and rt > 0:
            f = a11 + rt
            v = yt - m
            k1 = a11 / f
            k2 = a12 / f
            m = m + k1 * v
            b = b + k2 * v
            p11 = a11 * rt / f          # = a11 - a11^2/f, cancellation-free
            p12 = a12 * rt / f          # = a12 - a11*a12/f
            p22 = a22 - a12 * a12 / f
            if p22 < 0.0:
                p22 = 0.0
            inn[t], fv[t] = v, f
        else:
            p11, p12, p22 = a11, a12, a22
        lev[t], slo[t], ssd[t] = m, b, math.sqrt(p22)
    return FilterOut(lev, slo, ssd, inn, fv)


def steady_state_gain(q_zeta: float, r: float = 1.0, q_eta: float = 0.0, iters: int = 20000, tol: float = 1e-14) -> tuple[float, float]:
    """Steady-state Kalman gain (K_level, K_slope) of the time-invariant LLT, by iterating the Riccati recursion.
    Depends only on the ratios q_zeta / r and q_eta / r."""
    p11, p12, p22 = r, 0.0, 1.0
    k1 = k2 = float("nan")
    for _ in range(iters):
        a11 = p11 + 2 * p12 + p22 + q_eta
        a12 = p12 + p22
        a22 = p22 + q_zeta
        f = a11 + r
        n1, n2 = a11 / f, a12 / f
        p11, p12, p22 = a11 * r / f, a12 * r / f, a22 - a12 * a12 / f
        if abs(n1 - k1) < tol and abs(n2 - k2) < tol:
            k1, k2 = n1, n2
            break
        k1, k2 = n1, n2
    return float(k1), float(k2)


# ------------------------------------------------------------------ maximum likelihood (concentrated, diffuse)
def _conc_loglik(y: list[float], q_eps: float, q_zeta: float, n_diffuse: int = 2, kappa: float = 1e4, scale: bool = False):
    """Concentrated log likelihood with sigma2_eta scaled out (Q_eta = 1, R = q_eps, Q_zeta = q_zeta).
    Big-kappa diffuse prior; the first `n_diffuse` innovations are dropped (Harvey 1989 §3.4).
    With scale=True returns (loglik, sigma2_eta_hat)."""
    n = len(y)
    m, b = y[0], 0.0
    p11, p12, p22 = kappa, 0.0, kappa
    # condition on y_0
    f = p11 + q_eps
    p11, p12, p22 = p11 * q_eps / f, 0.0, p22
    ss = 0.0
    slf = 0.0
    k = 0
    for t in range(1, n):
        yt = y[t]
        m = m + b
        a11 = p11 + 2.0 * p12 + p22 + 1.0
        a12 = p12 + p22
        a22 = p22 + q_zeta
        f = a11 + q_eps
        v = yt - m
        m = m + a11 / f * v
        b = b + a12 / f * v
        p11 = a11 * q_eps / f
        p12 = a12 * q_eps / f
        p22 = a22 - a12 * a12 / f
        if p22 < 0.0:
            p22 = 0.0
        if t >= n_diffuse:
            ss += v * v / f
            slf += math.log(f)
            k += 1
    if k == 0 or ss <= 0:
        return (-math.inf, float("nan")) if scale else -math.inf
    s2 = ss / k
    ll = -0.5 * (k * math.log(2 * math.pi * s2) + k + slf)
    return (ll, s2) if scale else ll


@dataclass(frozen=True)
class LLTFit:
    q_eps: float      # sigma2_eps / sigma2_eta
    q_zeta: float     # sigma2_zeta / sigma2_eta  (the registry's q_hat for H-KF2)
    sigma2_eta: float
    loglik: float
    n: int


def llt_mle(y, bounds: tuple[float, float] = (-16.0, 4.0)) -> LLTFit:
    """MLE of the full LLT (eps, eta, zeta) on y by the concentrated diffuse likelihood: coarse grid over
    (log q_eps, log q_zeta), then bounded Nelder-Mead from the best grid point. Returns the variance ratios.
    Feed y on a sensible scale (e.g. 100 x log price, demeaned): the ratios are scale-free."""
    from scipy.optimize import minimize

    ya = np.asarray(y, dtype=float)
    ya = ya[np.isfinite(ya)]
    yl = (ya - ya[0]).tolist()
    lo, hi = bounds

    def nll(p):
        pe, pz = float(np.clip(p[0], lo, hi)), float(np.clip(p[1], lo, hi))
        ll = _conc_loglik(yl, math.exp(pe), math.exp(pz))
        return 1e300 if not np.isfinite(ll) else -ll

    grid = [(a, c) for a in (-12.0, -8.0, -4.0, -1.0) for c in (-14.0, -11.0, -8.0, -5.0, -2.0)]
    best = min(grid, key=nll)
    res = minimize(nll, np.array(best), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
    pe, pz = (float(np.clip(v, lo, hi)) for v in res.x)
    qe, qz = math.exp(pe), math.exp(pz)
    ll, s2 = _conc_loglik(yl, qe, qz, scale=True)  # sigma2_eta from the concentrated scale
    return LLTFit(q_eps=qe, q_zeta=qz, sigma2_eta=s2, loglik=float(ll), n=len(yl))
