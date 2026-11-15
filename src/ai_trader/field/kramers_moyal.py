"""Kramers-Moyal estimation of a jump-diffusion field from discretely sampled increments.

For a Markov process with generator  L f = mu f' + D/2 f'' + lambda E[f(x + J) - f(x)]  the conditional moments of an
increment dY over a step dt are, to first order in dt (Kramers-Moyal coefficients; Friedrich et al. 2011, Anvari et al.
2016 for jumps):
    M1(x) = E[dY | x] / dt                   -> mu(x) + lambda E J
    M2(x) = E[dY^2 | x] / dt                 -> D(x) + lambda E J^2
    Mn(x) = E[dY^n | x] / dt,  n >= 3        -> lambda E J^n
Estimators here:
  * `binned_moments` (histogram cells, 1-D or 2-D state) and `kernel_moments` (Nadaraya-Watson, Gaussian product
    kernel) for M1, conditional variance, M2, M4, M6 and, for a 2-vector increment, the conditional covariance
    (return x log-vol increment);
  * finite-dt bias corrections: (a) the conditional variance (M2 - M1^2 dt) instead of the raw second moment removes
    the O(dt) drift term; (b) `ou_correct`: exact OU inversion of a linear M1 regression (theta = -ln(1 + b dt) / dt,
    D = Var / dt * 2 theta dt / (1 - e^{-2 theta dt})); (c) `jump_moments` removes the Gaussian contributions to M4 / M6
    (3 D^2 dt, 6 D lambda E J^2 dt, ...) by fixed-point iteration before solving for (D, lambda, eta);
  * `fit_field_em`: the production estimator. Each 1h log return is a two-branch mixture
        r = mu(X) + sqrt(D(X)) e            with prob 1 - p(X)
        r = mu(X) + sqrt(D(X)) (e + K)      with prob p(X),  K ~ Kou(p_up, eta_up, eta_down) in diffusion-sd units
    D(X) = vf * exp(poly_D(X~)), mu(X) = sqrt(D(X)) * poly_mu(X~), p(X) = exp(poly_lam(X~)) (capped at 0.5): the KM
    regression (Boninsegna et al. 2018) of the second moment (Gamma-QML, log link), the first moment (least squares)
    and the jump indicator (Poisson-QML, log link) on polynomial libraries, with EM posterior jump weights separating
    the diffusion moments from the jump moments. Kou parameters by weighted MLE on the standardised residuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from ai_trader.field.poly import SparsePoly, glm_fit, stability_support

LOG2PI = math.log(2 * math.pi)
SQRT2 = math.sqrt(2.0)


def emg_logpdf(x: np.ndarray, sigma, eta: float) -> np.ndarray:
    """log density of N(0, sigma^2) + Exp(rate eta) at x, stable for all x (erfc branch for b < 0, erfcx for b >= 0;
    features.jump_diffusion's version overflows when b << 0)."""
    x = np.asarray(x, float)
    sigma = np.asarray(sigma, float)
    b = (eta * sigma**2 - x) / (SQRT2 * sigma)
    a = eta * (eta * sigma**2 / 2.0 - x)
    with np.errstate(all="ignore"):
        neg = math.log(eta / 2.0) + np.log(special.erfc(np.minimum(b, 0.0))) + a
        pos = math.log(eta / 2.0) + np.log(special.erfcx(np.maximum(b, 0.0))) + a - np.maximum(b, 0.0) ** 2
    return np.where(b < 0, neg, pos)


def jump_logpdf(x, sigma, p_up: float, eta_up: float, eta_down: float) -> np.ndarray:
    """log density of r given a jump: Gaussian diffusion plus a Kou jump (stable)."""
    return np.logaddexp(math.log(p_up) + emg_logpdf(x, sigma, eta_up), math.log(1 - p_up) + emg_logpdf(-np.asarray(x, float), sigma, eta_down))


# ------------------------------------------------------------------ local moment estimators


def binned_moments(x: np.ndarray, dy: np.ndarray, edges, dt: float = 1.0, min_n: int = 30) -> dict:
    """Conditional moments of increments dy (n,) or (n, 2) per state cell. x: (n,) with edges (array) or (n, 2) with a
    pair of edge arrays. Returns per cell: center, n, m1, var (conditional variance per dt: the finite-dt corrected
    second moment), m2, m4, m6 (raw moments per dt) and, for 2-D dy, cov / corr of the two components."""
    x = np.asarray(x, dtype=float)
    dy = np.asarray(dy, dtype=float)
    one = x.ndim == 1
    X = x[:, None] if one else x
    E = [np.asarray(edges, float)] if one else [np.asarray(e, float) for e in edges]
    idx = [np.clip(np.searchsorted(e, X[:, j], side="right") - 1, -1, len(e) - 1) for j, e in enumerate(E)]
    ok = np.ones(len(X), bool)
    for j, e in enumerate(E):
        ok &= (idx[j] >= 0) & (idx[j] < len(e) - 1)
    ok &= np.isfinite(dy).all(axis=1) if dy.ndim == 2 else np.isfinite(dy)
    shape = tuple(len(e) - 1 for e in E)
    flat = np.ravel_multi_index(tuple(i[ok] for i in idx), shape)
    D = dy[ok]
    K = int(np.prod(shape))
    n = np.bincount(flat, minlength=K).astype(float)

    def s(v):
        return np.bincount(flat, weights=v, minlength=K)

    out: dict = {"n": n.reshape(shape)}
    centers = [0.5 * (e[1:] + e[:-1]) for e in E]
    out["center"] = centers[0] if one else np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        comps = [D] if D.ndim == 1 else [D[:, k] for k in range(D.shape[1])]
        for k, d in enumerate(comps):
            sfx = "" if k == 0 else f"_{k}"
            m1 = s(d) / n
            m2 = s(d * d) / n
            out["m1" + sfx] = np.where(n >= min_n, m1 / dt, np.nan).reshape(shape)
            out["m2" + sfx] = np.where(n >= min_n, m2 / dt, np.nan).reshape(shape)
            out["var" + sfx] = np.where(n >= min_n, (m2 - m1 * m1) / dt, np.nan).reshape(shape)
            out["m4" + sfx] = np.where(n >= min_n, s(d**4) / n / dt, np.nan).reshape(shape)
            out["m6" + sfx] = np.where(n >= min_n, s(d**6) / n / dt, np.nan).reshape(shape)
        if len(comps) == 2:
            a, b = comps
            ma, mb = s(a) / n, s(b) / n
            cov = s(a * b) / n - ma * mb
            va, vb = s(a * a) / n - ma * ma, s(b * b) / n - mb * mb
            out["cov"] = np.where(n >= min_n, cov / dt, np.nan).reshape(shape)
            out["corr"] = np.where(n >= min_n, cov / np.sqrt(va * vb), np.nan).reshape(shape)
    return out


def kernel_moments(x: np.ndarray, dy: np.ndarray, grid: np.ndarray, bw, dt: float = 1.0, chunk: int = 16) -> dict:
    """Nadaraya-Watson conditional moments on `grid` (g, d) with a Gaussian product kernel of bandwidths bw (d,).
    Returns m1, var (conditional variance per dt), m2, m4 and the effective sample size per grid point."""
    X = np.asarray(x, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    G = np.asarray(grid, dtype=float)
    G = G[:, None] if G.ndim == 1 else G
    d = np.asarray(dy, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(d)
    X, d = X[ok], d[ok]
    h = np.broadcast_to(np.asarray(bw, dtype=float), (X.shape[1],))
    res = {k: np.empty(len(G)) for k in ("m1", "var", "m2", "m4", "neff")}
    for a in range(0, len(G), chunk):
        g = G[a: a + chunk]
        u = (X[None, :, :] - g[:, None, :]) / h
        W = np.exp(-0.5 * np.sum(u * u, axis=2))  # (c, n)
        sw = W.sum(1)
        m1 = W @ d / sw
        m2 = W @ (d * d) / sw
        res["m1"][a: a + chunk] = m1 / dt
        res["m2"][a: a + chunk] = m2 / dt
        res["var"][a: a + chunk] = (m2 - m1 * m1) / dt
        res["m4"][a: a + chunk] = W @ d**4 / sw / dt
        res["neff"][a: a + chunk] = sw**2 / np.sum(W * W, axis=1)
    return res


# ------------------------------------------------------------------ finite-dt corrections


@dataclass
class OUFit:
    theta: float    # mean-reversion rate per unit time
    mean: float
    D: float        # diffusion coefficient (sigma^2 per unit time)
    theta_naive: float
    D_naive: float

    @property
    def half_life(self) -> float:
        return math.log(2) / self.theta if self.theta > 0 else float("inf")


def ou_correct(x: np.ndarray, dx: np.ndarray, dt: float = 1.0) -> OUFit:
    """Exact-discretisation inversion of the linear KM drift: dx = a + b x + e over dt."""
    x = np.asarray(x, float)
    dx = np.asarray(dx, float)
    ok = np.isfinite(x) & np.isfinite(dx)
    x, dx = x[ok], dx[ok]
    A = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(A, dx, rcond=None)
    e = dx - A @ np.array([a, b])
    var = float(np.mean(e * e))
    theta_n, D_n = -b / dt, var / dt
    if -1 < b < 0:
        theta = -math.log(1 + b) / dt
        D = var / dt * (2 * theta * dt) / (1 - math.exp(-2 * theta * dt))
    else:
        theta, D = theta_n, D_n
    m = -a / b if b != 0 else float("nan")
    return OUFit(theta, m, D, theta_n, D_n)


def jump_moments(m2: float, m4: float, m6: float, dt: float = 1.0, law: str = "laplace", iters: int = 50) -> dict:
    """(D, lambda, jump scale) from KM moments per dt of a diffusion + compound-Poisson jump increment, removing the
    Gaussian / cross contributions at finite dt by fixed-point iteration. law 'laplace': symmetric double exponential
    (Kou with p_up = 1/2, eta_up = eta_down = eta; E J^2 = 2/eta^2, E J^4 = 24/eta^4, E J^6 = 720/eta^6);
    'gauss': N(0, s2) jumps (E J^2 = s2, E J^4 = 3 s2^2, E J^6 = 15 s2^3)."""
    D = m2
    lam = eta = s2 = float("nan")
    for _ in range(iters):
        if law == "laplace":
            ej2 = 2 / eta**2 if np.isfinite(eta) else 0.0
            ej4 = 24 / eta**4 if np.isfinite(eta) else 0.0
        else:
            ej2 = s2 if np.isfinite(s2) else 0.0
            ej4 = 3 * s2 * s2 if np.isfinite(s2) else 0.0
        lm = lam if np.isfinite(lam) else 0.0
        a4 = m4 - 3 * D * D * dt - 6 * D * lm * ej2 * dt
        a6 = m6 - 15 * D**3 * dt * dt - 45 * D * D * lm * ej2 * dt * dt - 15 * D * lm * ej4 * dt
        if a4 <= 0 or a6 <= 0:
            return {"D": m2, "lam": 0.0, "eta": float("inf"), "s2": 0.0, "ok": False}
        if law == "laplace":
            eta = math.sqrt(30 * a4 / a6)
            lam = a4 * eta**4 / 24
            Dn = m2 - lam * 2 / eta**2
        else:
            s2 = a6 / (5 * a4)
            lam = a4 / (3 * s2 * s2)
            Dn = m2 - lam * s2
        if abs(Dn - D) < 1e-12 * max(abs(D), 1e-300):
            D = Dn
            break
        D = max(Dn, 1e-12 * m2)
    return {"D": D, "lam": lam, "eta": eta, "s2": s2, "ok": True}


# ------------------------------------------------------------------ Kou law (standardised units)


@dataclass
class Kou:
    p_up: float = 0.5
    eta_up: float = 0.4     # rate in diffusion-sd units: mean up jump = 1 / eta_up sds
    eta_down: float = 0.4

    def to_dict(self) -> dict:
        return {"p_up": self.p_up, "eta_up": self.eta_up, "eta_down": self.eta_down, "mean_up_sd": 1 / self.eta_up, "mean_down_sd": 1 / self.eta_down}

    @property
    def mean(self) -> float:
        return self.p_up / self.eta_up - (1 - self.p_up) / self.eta_down

    @property
    def second(self) -> float:
        return 2 * (self.p_up / self.eta_up**2 + (1 - self.p_up) / self.eta_down**2)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        up = rng.random(n) < self.p_up
        return np.where(up, rng.exponential(1 / self.eta_up, n), -rng.exponential(1 / self.eta_down, n))


def fit_kou(z: np.ndarray, w: np.ndarray, start: Kou | None = None, eta_max: float = np.inf) -> Kou:
    """Weighted MLE of the Kou law of jump branch residuals z = N(0, 1) + K. eta_max caps the rates (mean jump size
    >= 1 / eta_max sds), which keeps the jump branch from turning into a second diffusion scale."""
    z = np.asarray(z, float)
    w = np.asarray(w, float)
    k0 = start or Kou(0.5, min(0.4, 0.9 * eta_max), min(0.4, 0.9 * eta_max))
    if w.sum() < 3:
        return k0
    keep = w > 1e-4  # rows with negligible jump weight do not move the estimate
    z, w = z[keep], w[keep]
    capped = bool(np.isfinite(eta_max))

    def rate(t):
        return eta_max * special.expit(t) if capped else math.exp(t)

    def inv(e):
        if capped:
            return special.logit(min(e, 0.999 * eta_max) / eta_max)
        return math.log(e)

    def nll(th):
        if np.max(np.abs(th)) > 15:
            return 1e300
        pu, eu, ed = special.expit(th[0]), rate(th[1]), rate(th[2])
        with np.errstate(all="ignore"):
            v = -float(np.sum(w * jump_logpdf(z, np.ones_like(z), pu, eu, ed)))
        return v if np.isfinite(v) else 1e300

    th0 = [special.logit(min(max(k0.p_up, 0.02), 0.98)), inv(k0.eta_up), inv(k0.eta_down)]
    r = optimize.minimize(nll, th0, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4 * max(1.0, float(w.sum()))})
    return Kou(float(special.expit(r.x[0])), float(rate(r.x[1])), float(rate(r.x[2])))


# ------------------------------------------------------------------ EM fit of the polynomial field

P_CAP = 0.5
ETA_MAX = 1.0 / 3.0   # jumps average >= 3 diffusion sds (identification of the jump branch; pilot 2026-11-13)
NU_BOUNDS = (2.2, 60.0)


def t_logpdf_unit(z: np.ndarray, nu: float) -> np.ndarray:
    """log density of a unit-VARIANCE Student-t at z."""
    sc2 = (nu - 2) / nu
    return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(nu * math.pi * sc2)
            - (nu + 1) / 2 * np.log1p(z * z / (nu * sc2)))


def fit_nu(z: np.ndarray, w: np.ndarray) -> float:
    def f(nu):
        return -float(np.sum(w * t_logpdf_unit(z, nu)))

    return float(optimize.minimize_scalar(f, bounds=NU_BOUNDS, method="bounded", options={"xatol": 1e-3}).x)


@dataclass
class EMFit:
    mu: SparsePoly
    logD: SparsePoly
    loglam: SparsePoly
    kou: Kou
    w: np.ndarray            # posterior jump weights of the training rows
    loglik: float
    n: int
    iters: int
    t_dof: float = float("nan")   # df of the unit-variance Student-t non-jump branch (core 't'); nan for core 'gauss'
    core: str = "t"


def components(Ts: tuple[np.ndarray, np.ndarray, np.ndarray], coefs: tuple[np.ndarray, np.ndarray, np.ndarray], lnvf: np.ndarray):
    """(mu_r, D, p) from the three design matrices and coefficient vectors."""
    Tm, Td, Tl = Ts
    cm, cd, cl = coefs
    D = np.exp(np.clip(lnvf + Td @ cd, -40, 5))
    mu = np.sqrt(D) * (Tm @ cm)
    p = np.minimum(np.exp(np.clip(Tl @ cl, -30, 0)), P_CAP)
    return mu, D, p


def _core_logpdf(z, t_dof):
    if t_dof is not None and np.isfinite(t_dof):
        return t_logpdf_unit(z, t_dof)
    return -0.5 * LOG2PI - 0.5 * z * z


def mixture_logpdf(r: np.ndarray, mu: np.ndarray, D: np.ndarray, p: np.ndarray, kou: Kou, t_dof: float | None = None) -> np.ndarray:
    """log density of the 1-step field law at r. With t_dof, the non-jump branch is a unit-variance Student-t."""
    sd = np.sqrt(D)
    z = (r - mu) / sd
    lj = jump_logpdf(z, np.ones_like(z), kou.p_up, kou.eta_up, kou.eta_down)
    return np.logaddexp(np.log1p(-p) + _core_logpdf(z, t_dof), np.log(p) + lj) - np.log(sd)


def _emg_cdf(x: np.ndarray, eta: float) -> np.ndarray:
    """CDF of N(0,1) + Exp(eta)."""
    # F(x) = Phi(x) - exp(-eta x + eta^2/2) Phi(x - eta), computed stably via log
    a = -eta * x + 0.5 * eta * eta + special.log_ndtr(x - eta)
    return special.ndtr(x) - np.exp(np.minimum(a, 50))


def mixture_cdf(r: np.ndarray, mu: np.ndarray, D: np.ndarray, p: np.ndarray, kou: Kou, t_dof: float | None = None) -> np.ndarray:
    z = (r - mu) / np.sqrt(D)
    if t_dof is not None and np.isfinite(t_dof):
        sc = math.sqrt((t_dof - 2) / t_dof)
        cn = special.stdtr(t_dof, z / sc)
    else:
        cn = special.ndtr(z)
    cu = _emg_cdf(z, kou.eta_up)
    cd = 1 - _emg_cdf(-z, kou.eta_down)
    cj = kou.p_up * cu + (1 - kou.p_up) * cd
    return np.clip((1 - p) * cn + p * cj, 0.0, 1.0)


def fit_field_em(r: np.ndarray, lnvf: np.ndarray, Xs: np.ndarray, names, monos_mu, monos_D, monos_lam, *, groups: np.ndarray | None = None,
                 select: bool = False, n_sub: int = 40, pi_thr: float = 0.7, ev: float = 1.0, n_em: int = 8, init_thr: float = 3.5,
                 start: EMFit | None = None, seed: int = 0, progress=None, core: str = "t", eta_max: float = ETA_MAX) -> tuple[EMFit, dict]:
    """EM fit of the jump-diffusion field on training rows (r = next-bar log return, lnvf = ln vf at the origin,
    Xs = standardised coordinates). core 't': the non-jump branch is a unit-variance Student-t (df by EM), 'gauss':
    Gaussian. With select=True the supports of the three polynomials are chosen by stability selection (on the
    libraries given) after an EM round with linear-only libraries; otherwise the libraries are the supports."""
    from ai_trader.field.poly import design

    r = np.asarray(r, float)
    lnvf = np.asarray(lnvf, float)
    n = len(r)
    diag: dict = {}
    lib = {"mu": list(monos_mu), "D": list(monos_D), "lam": list(monos_lam)}
    if select:
        # stage 1: linear libraries to get stable jump weights
        lin = [m for m in monos_D if sum(m) <= 1]
        f0, _ = fit_field_em(r, lnvf, Xs, names, [lin[0]], lin, [lin[0]], n_em=6, init_thr=init_thr, core=core, eta_max=eta_max)
        w = f0.w
        diag["stage1"] = {"jump_share": float(w.mean()), "t_dof": f0.t_dof, "kou": f0.kou.to_dict()}
        sd0 = np.sqrt(np.exp(lnvf + design(Xs, lin) @ f0.logD.coef))
        sup = {}
        for key, fam in (("D", "gamma"), ("mu", "gaussian"), ("lam", "poisson")):
            T = design(Xs, lib[key])
            if key == "D":
                y, ww, off = r * r, 1 - w, lnvf
            elif key == "mu":
                y, ww, off = r / sd0, 1 - w, None
            else:
                y, ww, off = w, None, None
            s_, mp = stability_support(T, y, groups if groups is not None else np.arange(n) // 24, fam, w=ww, offset=off, n_sub=n_sub,
                                       pi_thr=pi_thr, ev=ev, seed=seed)
            sup[key] = [lib[key][i] for i in s_]
            diag[f"sel_prob_{key}"] = {_nm(m, names): float(p_) for m, p_ in zip(lib[key], mp, strict=True) if p_ >= 0.3}
            if progress:
                progress(key)
        lib = sup
        diag["support"] = {k: [_nm(m, names) for m in v] for k, v in lib.items()}
    Tm, Td, Tl = design(Xs, lib["mu"]), design(Xs, lib["D"]), design(Xs, lib["lam"])
    nu = 5.0 if core == "t" else None
    same = start is not None and [tuple(m) for m in start.logD.monos] == lib["D"] and [tuple(m) for m in start.mu.monos] == lib["mu"] \
        and [tuple(m) for m in start.loglam.monos] == lib["lam"]
    if same:
        cm, cd, cl, kou = start.mu.coef.copy(), start.logD.coef.copy(), start.loglam.coef.copy(), start.kou
        if core == "t" and np.isfinite(start.t_dof):
            nu = float(start.t_dof)
        mu, D, p = components((Tm, Td, Tl), (cm, cd, cl), lnvf)
        w = _posterior(r, mu, D, p, kou, nu)
    else:
        z0 = r / np.exp(0.5 * lnvf)
        w = (np.abs(z0) > init_thr).astype(float) * 0.9
        kou = Kou(0.5, 0.9 * min(0.4, eta_max), 0.9 * min(0.4, eta_max))
        cm = cd = cl = None
    res_m = res_d = res_l = None
    ll_old = -np.inf
    it = 0
    for it in range(1, n_em + 1):
        mu_prev = 0.0 if (cm is None or cd is None) else np.sqrt(np.exp(np.clip(lnvf + Td @ cd, -40, 5))) * (Tm @ cm)
        res_d = glm_fit(Td, (r - mu_prev) ** 2, "gamma", w=1 - w, offset=lnvf, start=cd)
        cd = res_d.coef
        sd = np.sqrt(np.exp(np.clip(lnvf + Td @ cd, -40, 5)))
        res_m = glm_fit(Tm, r / sd, "gaussian", w=1 - w)
        cm = res_m.coef
        res_l = glm_fit(Tl, w, "poisson", start=cl)
        cl = res_l.coef
        mu, D, p = components((Tm, Td, Tl), (cm, cd, cl), lnvf)
        z = (r - mu) / np.sqrt(D)
        kou = fit_kou(z, w, kou, eta_max)
        if core == "t":
            nu = fit_nu(z, 1 - w)
        w = _posterior(r, mu, D, p, kou, nu)
        ll = float(np.sum(mixture_logpdf(r, mu, D, p, kou, nu)))
        if abs(ll - ll_old) < 1e-3 * max(1.0, n / 1e4):
            break
        ll_old = ll
    mu, D, p = components((Tm, Td, Tl), (cm, cd, cl), lnvf)
    fit = EMFit(SparsePoly.from_fit(names, lib["mu"], res_m, "identity", "gaussian", "z = r / sqrt(D)"),
                SparsePoly.from_fit(names, lib["D"], res_d, "log", "gamma", "ln vf"),
                SparsePoly.from_fit(names, lib["lam"], res_l, "log", "poisson", ""),
                kou, w, float(np.sum(mixture_logpdf(r, mu, D, p, kou, nu))), n, it, float(nu) if nu is not None else float("nan"), core)
    return fit, diag


def _nm(m, names):
    from ai_trader.field.poly import mono_name

    return mono_name(tuple(m), names)


def _posterior(r, mu, D, p, kou: Kou, t_dof: float | None = None) -> np.ndarray:
    z = (r - mu) / np.sqrt(D)
    lj = np.log(p) + jump_logpdf(z, np.ones_like(z), kou.p_up, kou.eta_up, kou.eta_down)
    ln = np.log1p(-p) + _core_logpdf(z, t_dof)
    return special.expit(lj - ln)


__all__ = [
    "ETA_MAX",
    "P_CAP",
    "EMFit",
    "Kou",
    "OUFit",
    "binned_moments",
    "components",
    "fit_field_em",
    "fit_kou",
    "fit_nu",
    "jump_moments",
    "kernel_moments",
    "mixture_cdf",
    "mixture_logpdf",
    "ou_correct",
    "t_logpdf_unit",
]
