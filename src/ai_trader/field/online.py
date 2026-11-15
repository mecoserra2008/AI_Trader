"""Online Bayesian updating of the field coefficients (H-FLD3): recursive least squares with exponential forgetting on
the IRLS working problem of each component (a Kalman filter with a random-walk state whose process noise is set by the
forgetting factor), starting from the in-sample fit as the prior (mean = fitted coefficients, covariance = the fit's
model-based covariance phi (X'WX)^-1).

Per hour, after r_{t+1} is known (close of bar t+1), with X_t the origin state and w the posterior jump weight of
r_{t+1} under the CURRENT coefficients:
    P <- P / lam_f
    working residual u = (y - m) / m (log link) or y - eta (identity); working weight W (Gamma: w, Poisson: w m)
    k = P x / (phi / W + x' P x);  b <- b + k u;  P <- P - k x' P
Components: logD (Gamma, y = (r - mu)^2, offset ln vf, weight 1 - w), mu (Gaussian, y = r / sqrt(D), weight 1 - w),
loglam (Poisson, y = w). With lam_f = 1 and a flat prior this reproduces batch least squares (tested).
The coefficients used for the forecast at origin i are those after the updates with outcomes known at i (pairs
(X_j, r_{j+1}) with j + 1 <= i).
"""

from __future__ import annotations

import numpy as np
from scipy import special

from ai_trader.field.kramers_moyal import P_CAP, _core_logpdf, jump_logpdf
from ai_trader.field.model import Field
from ai_trader.field.poly import SparsePoly

LOG2PI = float(np.log(2 * np.pi))


class OnlineGLM:
    def __init__(self, poly: SparsePoly, forget: float = 1.0, prior_scale: float = 1.0, P0: np.ndarray | None = None):
        self.b = poly.coef.astype(float).copy()
        p = len(self.b)
        P = poly.cov if P0 is None else P0
        self.P = (np.eye(p) * 1e6 if P is None else np.asarray(P, float).copy()) * prior_scale
        self.family = poly.family
        self.link = poly.link
        self.phi = float(poly.phi) if poly.phi and np.isfinite(poly.phi) else 1.0
        self.forget = float(forget)

    def update(self, x: np.ndarray, y: float, offset: float = 0.0, w: float = 1.0) -> None:
        P = self.P / self.forget
        eta = offset + float(x @ self.b)
        if self.link == "log":
            m = float(np.exp(np.clip(eta, -30, 30)))
            W = w if self.family == "gamma" else w * m
            u = (y - m) / max(m, 1e-300)
        else:
            W = w
            u = y - eta
        if W <= 1e-12:
            self.P = P
            return
        Px = P @ x
        k = Px / (self.phi / W + float(x @ Px))
        self.b = self.b + k * u
        self.P = P - np.outer(k, Px)


def run_online(fld: Field, X: np.ndarray, lnvf: np.ndarray, r_next: np.ndarray, forget: float = 1 - 1 / 1440, prior_scale: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficient histories (n, p_mu), (n, p_D), (n, p_lam): row i = coefficients usable at origin i."""
    Tm, Td, Tl = fld.designs(fld.Xs(X))
    om, od, ol = (OnlineGLM(fld.mu, forget, prior_scale), OnlineGLM(fld.logD, forget, prior_scale), OnlineGLM(fld.loglam, forget, prior_scale))
    n = len(r_next)
    Cm, Cd, Cl = np.empty((n, Tm.shape[1])), np.empty((n, Td.shape[1])), np.empty((n, Tl.shape[1]))
    kou = fld.kou
    nu = fld.nu()
    for i in range(n):
        Cm[i], Cd[i], Cl[i] = om.b, od.b, ol.b
        # r_{i+1} becomes known at the close of bar i+1: the update below is used from origin i+1 on
        r = r_next[i]
        if not np.isfinite(r) or not np.isfinite(lnvf[i]):
            continue
        D = float(np.exp(np.clip(lnvf[i] + Td[i] @ od.b, -40, 5)))
        sd = np.sqrt(D)
        mu = sd * float(Tm[i] @ om.b)
        p = min(float(np.exp(np.clip(Tl[i] @ ol.b, -30, 0))), P_CAP)
        z = (r - mu) / sd
        lj = np.log(p) + float(jump_logpdf(np.array([z]), np.ones(1), kou.p_up, kou.eta_up, kou.eta_down)[0])
        ln = np.log1p(-p) + float(_core_logpdf(np.array([z]), nu)[0])
        w = float(special.expit(lj - ln))
        od.update(Td[i], (r - mu) ** 2, offset=float(lnvf[i]), w=1 - w)
        om.update(Tm[i], r / sd, w=1 - w)
        ol.update(Tl[i], w)
    return Cm, Cd, Cl


def params_varying(fld: Field, X: np.ndarray, lnvf: np.ndarray, Cm: np.ndarray, Cd: np.ndarray, Cl: np.ndarray, drift: bool = True):
    Tm, Td, Tl = fld.designs(fld.Xs(X))
    D = np.exp(np.clip(lnvf + np.sum(Td * Cd, 1), -40, 5))
    mu = np.sqrt(D) * np.sum(Tm * Cm, 1) if drift else np.zeros(len(D))
    p = np.minimum(np.exp(np.clip(np.sum(Tl * Cl, 1), -30, 0)), P_CAP)
    return mu, D, p


__all__ = ["OnlineGLM", "params_varying", "run_online"]
