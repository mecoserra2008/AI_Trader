"""Sparse polynomial fields (SINDy-style discovery for stochastic dynamics; Brunton, Proctor and Kutz 2016; Boninsegna,
Nueske and Clementi 2018 for drift / diffusion of SDEs).

A field component is g(X) = link^{-1}( offset + sum_k c_k * phi_k(X~) ), phi_k a monomial of degree <= 3 in the
standardised coordinates X~ = (X - mean) / sd (means / sds from the training rows). Links:
  identity  (drift: E[z | X] with z the standardised increment)            Gaussian least squares
  log       (diffusion multiplier E[r^2 / vf | X], jump probability)        Gamma-QML (variance ~ mean^2) or
                                                                            Poisson / Bernoulli-QML (variance ~ mean)
Both are fitted by IRLS (`glm_fit`), with optional observation weights (EM posterior weights) and a small ridge.

Selection:
  * `stlsq`: sequentially thresholded least squares on |t| (scale-free version of SINDy's coefficient threshold):
    refit on the support, drop terms with |t| < t_min, repeat until stable. Works on the IRLS working problem.
  * `stability_support`: randomised-lasso stability selection (Meinshausen and Buhlmann 2010; ml.sparse) on the IRLS
    working response at a full-library fit, subsamples of half the DATES, PFER bound E[V] <= ev. Intercept always kept.
Readable equations: `SparsePoly.equation()`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

Mono = tuple[int, ...]


# ------------------------------------------------------------------ library


def monomials(n_vars: int, degree: int = 3, include_const: bool = True, max_deg: dict[int, int] | None = None) -> list[Mono]:
    """All exponent tuples with total degree <= degree (graded order). max_deg[i] caps the power of variable i."""
    out: list[Mono] = []
    for d in range(0 if include_const else 1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_vars), d):
            e = [0] * n_vars
            for i in combo:
                e[i] += 1
            if max_deg and any(e[i] > m for i, m in max_deg.items()):
                continue
            out.append(tuple(e))
    return out


def design(Xs: np.ndarray, monos: list[Mono]) -> np.ndarray:
    """Theta (n, len(monos)) of monomials of the standardised coordinates."""
    Xs = np.asarray(Xs, dtype=float)
    n, p = Xs.shape
    maxp = max((max(m) if m else 0) for m in monos) if monos else 0
    pw = [np.ones((n, p))]
    for _ in range(maxp):
        pw.append(pw[-1] * Xs)
    T = np.empty((n, len(monos)))
    for j, m in enumerate(monos):
        col = np.ones(n)
        for i, e in enumerate(m):
            if e:
                col = col * pw[e][:, i]
        T[:, j] = col
    return T


def mono_name(m: Mono, names: tuple[str, ...] | list[str]) -> str:
    parts = []
    for i, e in enumerate(m):
        if e == 1:
            parts.append(names[i])
        elif e > 1:
            parts.append(f"{names[i]}^{e}")
    return "*".join(parts) if parts else "1"


@dataclass
class Standardizer:
    names: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, names) -> Standardizer:
        X = np.asarray(X, dtype=float)
        sd = X.std(0)
        return cls(tuple(names), X.mean(0), np.where(sd > 1e-12, sd, 1.0))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.sd

    def to_dict(self) -> dict:
        return {"names": list(self.names), "mean": self.mean.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Standardizer:
        return cls(tuple(d["names"]), np.asarray(d["mean"], float), np.asarray(d["sd"], float))


# ------------------------------------------------------------------ GLM by IRLS

FAMILIES = {"gaussian": ("identity", lambda m: np.ones_like(m)), "gamma": ("log", lambda m: m * m), "poisson": ("log", lambda m: m)}


@dataclass
class GLMResult:
    coef: np.ndarray
    cov: np.ndarray          # robust (sandwich) covariance
    cov_model: np.ndarray    # model-based phi * (X'WX)^-1 (the online prior)
    phi: float               # Pearson dispersion
    n: int
    converged: bool
    eta: np.ndarray | None = None


def _mean(eta, link):
    return np.exp(np.clip(eta, -30, 30)) if link == "log" else eta


def glm_fit(T: np.ndarray, y: np.ndarray, family: str = "gaussian", *, w: np.ndarray | None = None, offset: np.ndarray | None = None,
            ridge: float = 1e-6, max_iter: int = 50, tol: float = 1e-8, start: np.ndarray | None = None, keep_eta: bool = False) -> GLMResult:
    """IRLS for E[y|x] = link^{-1}(offset + T b) with prior weights w. Gaussian / Gamma / Poisson (QML) families."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = T.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    link, vfun = FAMILIES[family]
    R = ridge * np.eye(p)
    if p and np.allclose(T[:, 0], 1.0):
        R[0, 0] = 0.0
    if start is not None:
        b = np.asarray(start, dtype=float).copy()
    elif link == "log":
        b = np.zeros(p)
        if p and np.allclose(T[:, 0], 1.0):
            b[0] = math.log(max(float(np.sum(w * y) / max(np.sum(w * np.exp(off)), 1e-300)), 1e-12))
    else:
        b = np.zeros(p)
    conv = False
    for _ in range(max_iter if link == "log" else 1):
        eta = off + T @ b
        m = _mean(eta, link)
        if link == "log":
            # canonical working weights for log link: dmu/deta = mu, so W = w * mu^2 / V(mu)
            W = w * m * m / np.maximum(vfun(m), 1e-300)
            z = eta - off + (y - m) / np.maximum(m, 1e-300)
        else:
            W = w
            z = y - off
        A = T.T @ (T * W[:, None]) + R
        bn = np.linalg.solve(A, T.T @ (W * z))
        if link != "log":
            b = bn
            conv = True
            break
        step = np.max(np.abs(bn - b))
        # damping against overshoot on rare-event Poisson fits
        if not np.isfinite(step):
            break
        b = bn if step < 5 else b + (bn - b) * (5 / step)
        if step < tol:
            conv = True
            break
    eta = off + T @ b
    m = _mean(eta, link)
    if link == "log":
        W = w * m * m / np.maximum(vfun(m), 1e-300)
        resid = (y - m) / np.maximum(m, 1e-300)  # working residual
    else:
        W = w
        resid = y - off - T @ b
    A = T.T @ (T * W[:, None]) + R
    Ai = np.linalg.pinv(A)
    pear = w * (y - m) ** 2 / np.maximum(vfun(m), 1e-300)
    sw = float(np.sum(w))
    phi = float(np.sum(pear) / max(sw - p, 1.0))
    S = T * (W * resid)[:, None]
    cov = Ai @ (S.T @ S) @ Ai
    return GLMResult(b, cov, phi * Ai, phi, n, conv, eta if keep_eta else None)


def glm_tstats(res: GLMResult) -> np.ndarray:
    se = np.sqrt(np.maximum(np.diag(res.cov), 1e-300))
    return res.coef / se


# ------------------------------------------------------------------ STLSQ


def stlsq(T: np.ndarray, y: np.ndarray, family: str = "gaussian", *, w=None, offset=None, t_min: float = 2.0, keep: tuple[int, ...] = (0,),
          max_rounds: int = 20, start_support: np.ndarray | None = None) -> tuple[np.ndarray, GLMResult]:
    """Sequentially thresholded (on robust |t|) GLM. Returns (support indices, final fit on the support)."""
    p = T.shape[1]
    sup = np.arange(p) if start_support is None else np.asarray(sorted(set(start_support) | set(keep)))
    res = glm_fit(T[:, sup], y, family, w=w, offset=offset)
    for _ in range(max_rounds):
        t = np.abs(glm_tstats(res))
        drop = [i for i, j in enumerate(sup) if t[i] < t_min and j not in keep]
        if not drop:
            break
        # drop the weakest half of the failing terms per round (stable against collinear groups)
        drop = sorted(drop, key=lambda i: t[i])[: max(1, len(drop) // 2)]
        sup = np.array([j for i, j in enumerate(sup) if i not in drop])
        res = glm_fit(T[:, sup], y, family, w=w, offset=offset)
    return sup, res


# ------------------------------------------------------------------ stability selection on the working problem


def stability_support(T: np.ndarray, y: np.ndarray, groups: np.ndarray, family: str = "gaussian", *, w=None, offset=None, n_sub: int = 50,
                      pi_thr: float = 0.7, ev: float = 1.0, seed: int = 0, max_rows: int = 40000) -> tuple[np.ndarray, np.ndarray]:
    """Columns of T (column 0 = intercept, always kept) selected by randomised-lasso stability selection on the IRLS
    working response of a ridge full-library fit. Returns (support, max selection probability per column)."""
    from ai_trader.ml.sparse import q_for_bound, stability_selection

    T = np.asarray(T, dtype=float)
    n, p = T.shape
    rng = np.random.default_rng(seed)
    if n > max_rows:  # subsample whole groups to bound memory / time
        ug = np.unique(groups)
        pick = rng.choice(ug, size=max(2, int(len(ug) * max_rows / n)), replace=False)
        rows = np.isin(groups, pick)
        T, y, groups = T[rows], np.asarray(y)[rows], np.asarray(groups)[rows]
        w = None if w is None else np.asarray(w)[rows]
        offset = None if offset is None else np.asarray(offset)[rows]
        n = len(y)
    res = glm_fit(T, y, family, w=w, offset=offset, ridge=1e-3, keep_eta=True)
    link, vfun = FAMILIES[family]
    ww = np.ones(n) if w is None else np.asarray(w, float)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    if link == "log":
        m = np.exp(np.clip(res.eta, -30, 30))
        W = ww * m * m / np.maximum(vfun(m), 1e-300)
        z = res.eta - off + (np.asarray(y) - m) / np.maximum(m, 1e-300)
    else:
        W = ww
        z = np.asarray(y) - off
    sw = np.sqrt(W / W.mean())
    X = T[:, 1:]
    xm = (X * W[:, None]).sum(0) / W.sum()
    zm = float((z * W).sum() / W.sum())
    Xt = (X - xm) * sw[:, None]
    zt = (z - zm) * sw
    q = q_for_bound(X.shape[1], pi_thr, ev)
    sr = stability_selection(Xt, zt, groups, n_sub=n_sub, seed=seed, stop_union=int(math.ceil(3 * q)) + 2)
    mp = sr.max_prob(q)
    sel = np.flatnonzero(mp >= pi_thr) + 1
    return np.r_[0, sel].astype(int), np.r_[1.0, mp]


# ------------------------------------------------------------------ a fitted sparse polynomial


@dataclass
class SparsePoly:
    names: tuple[str, ...]
    monos: list[Mono]
    coef: np.ndarray
    se: np.ndarray
    link: str = "identity"          # identity | log
    family: str = "gaussian"
    offset: str = ""                # description of the offset (e.g. 'ln vf')
    phi: float = 1.0
    cov: np.ndarray | None = None   # model-based covariance (online prior)
    sel_prob: list[float] = field(default_factory=list)

    def theta(self, Xs: np.ndarray) -> np.ndarray:
        return design(Xs, self.monos)

    def eta(self, Xs: np.ndarray, coef: np.ndarray | None = None) -> np.ndarray:
        return self.theta(Xs) @ (self.coef if coef is None else coef)

    def __call__(self, Xs: np.ndarray, offset: np.ndarray | float = 0.0) -> np.ndarray:
        e = self.eta(Xs) + offset
        return np.exp(np.clip(e, -30, 30)) if self.link == "log" else e

    def terms(self) -> list[dict]:
        return [{"term": mono_name(m, self.names), "exponents": {self.names[i]: e for i, e in enumerate(m) if e}, "coef": float(c), "se": float(s),
                 "t": float(c / s) if s > 0 else float("nan")} for m, c, s in zip(self.monos, self.coef, self.se, strict=True)]

    def equation(self, lhs: str = "g", prec: int = 3) -> str:
        parts = []
        for m, c in zip(self.monos, self.coef, strict=True):
            nm = mono_name(m, [f"{n}~" for n in self.names])
            parts.append(f"{c:+.{prec}f}" + ("" if nm == "1" else f" {nm}"))
        body = " ".join(parts) if parts else "0"
        if self.link == "log":
            off = f"{self.offset} + " if self.offset else ""
            return f"ln {lhs} = {off}{body}"
        return f"{lhs} = {body}"

    def to_dict(self) -> dict:
        return {"names": list(self.names), "link": self.link, "family": self.family, "offset": self.offset, "phi": self.phi,
                "terms": [{"exponents": list(m), "coef": float(c), "se": float(s)} for m, c, s in zip(self.monos, self.coef, self.se, strict=True)],
                "cov": None if self.cov is None else np.asarray(self.cov).tolist(), "sel_prob": list(map(float, self.sel_prob)),
                "equation": self.equation()}

    @classmethod
    def from_dict(cls, d: dict) -> SparsePoly:
        monos = [tuple(t["exponents"]) for t in d["terms"]]
        return cls(tuple(d["names"]), monos, np.array([t["coef"] for t in d["terms"]], float), np.array([t["se"] for t in d["terms"]], float),
                   d["link"], d["family"], d.get("offset", ""), d.get("phi", 1.0), None if d.get("cov") is None else np.asarray(d["cov"], float),
                   d.get("sel_prob", []))

    @classmethod
    def from_fit(cls, names, monos, res: GLMResult, link: str, family: str, offset: str = "", sel_prob=()) -> SparsePoly:
        se = np.sqrt(np.maximum(np.diag(res.cov), 0.0))
        return cls(tuple(names), list(monos), res.coef.copy(), se, link, family, offset, res.phi, res.cov_model.copy(), list(sel_prob))


__all__ = ["FAMILIES", "GLMResult", "Mono", "SparsePoly", "Standardizer", "design", "glm_fit", "glm_tstats", "mono_name", "monomials",
           "stability_support", "stlsq"]
