"""Forward propagation of the field: predictive densities, quantiles and first-passage probabilities at 1h / 4h / 1d.

Monte-Carlo of the jump-diffusion (vectorised over origins x paths, origin-major). Step j moves every path from the
close of bar t+j to the close of bar t+j+1:
    X_j  -> (mu, D, p) from the field;   r = mu + sqrt(D) (e + 1[U < p] K)
    state update with r through the SAME causal recursion as the real state (state.step): the vol EWMAs, the jump
    excitation and the 24h momentum move inside each path; funding z, OI change and the regime direction score are
    exogenous and stay at their origin values; the hour-of-week factor follows the calendar.
Density at the horizon (Rao-Blackwellised): the last step is integrated analytically,
    p(R) = mean_i p_1(R - x_{h-1}^{(i)} | X_{h-1}^{(i)}),   F(R) likewise for the PIT;
a smooth, unbiased estimate of the predictive density that needs far fewer paths than a histogram / KDE.
First passage: per path and step, Brownian-bridge touch probability exp(-2 (b - x0)(b - x1) / D) for the diffusion leg
(a close beyond the level counts as a touch, which covers jumps), accumulated as survival; P(hit) = 1 - mean survival.
Memory: n_origins * n_paths * (~15 + library size) doubles; `simulate` refuses > 400k paths per call (caller batches).

Fokker-Planck checks (finite volume, explicit Euler, zero-flux walls): `fp1d` for a scalar SDE, `fp2d_logprice_vol`
for (log price, log vol) with dX = sigma(v) dW1, dv = mu_v(v) dt + sqrt(D_v(v)) dW2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import special

from ai_trader.field.kramers_moyal import components, mixture_cdf, mixture_logpdf
from ai_trader.field.model import Field
from ai_trader.field.state import (
    COORDS,
    EXOGENOUS,
    coords_from,
    future_factors,
    last_returns,
    step,
)

MAX_PATHS = 400_000


@dataclass
class Origins:
    vf: np.ndarray
    vm: np.ndarray
    vs: np.ndarray
    jx: np.ndarray
    last24: np.ndarray   # (n_o, 24) real returns of bars t-23..t
    exo: np.ndarray      # (n_o, 3) fz, oi, dir
    fut_f: np.ndarray    # (n_o, h) hour-of-week factors of bars t+1..t+h

    def __len__(self) -> int:
        return len(self.vf)


def origins_from_state(state: pl.DataFrame, idx, profile: np.ndarray, h: int) -> Origins:
    idx = np.asarray(idx)
    g = lambda c: state[c].to_numpy().astype(float)[idx]
    return Origins(g("vf"), g("vm"), g("vs"), g("jx"), last_returns(state, idx, 24),
                   np.column_stack([g(c) for c in EXOGENOUS]), future_factors(state, idx, profile, h)[:, :h])


def _logmeanexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def simulate(fld: Field, org: Origins, h: int, n_paths: int = 1000, *, seed: int = 0, drift: bool = True, coefs=None,
             target: np.ndarray | None = None, levels: tuple[float, ...] = (), keep_terminal: bool = True) -> dict:
    """Propagate `org` h bars ahead. target (n_o,) = realised h-bar log returns for RB log density and PIT.
    levels: log-return levels (signed) for first-passage probabilities within h."""
    n_o = len(org)
    N = n_o * n_paths
    if N > MAX_PATHS:  # batch the origins to bound memory
        per = max(1, MAX_PATHS // n_paths)
        parts = []
        for k, a in enumerate(range(0, n_o, per)):
            sl = slice(a, a + per)
            sub = Origins(org.vf[sl], org.vm[sl], org.vs[sl], org.jx[sl], org.last24[sl], org.exo[sl], org.fut_f[sl])
            parts.append(simulate(fld, sub, h, n_paths, seed=seed + 7919 * k, drift=drift, coefs=coefs,
                                  target=None if target is None else np.asarray(target)[sl], levels=levels, keep_terminal=keep_terminal))
        out = {}
        for key in parts[0]:
            if key == "p_hit":
                out[key] = {b: np.concatenate([q[key][b] for q in parts]) for b in parts[0][key]}
            else:
                out[key] = np.concatenate([q[key] for q in parts])
        return out
    rng = np.random.default_rng(seed)
    rep = lambda a: np.repeat(np.asarray(a, float), n_paths, axis=0)
    vf, vm, vs, jx = rep(org.vf), rep(org.vm), rep(org.vs), rep(org.jx)
    sum24 = rep(org.last24.sum(1))
    exo = rep(org.exo)
    x = np.zeros(N)
    sim_r = np.zeros((N, h)) if h > 24 else None
    surv = {b: np.ones(N) for b in levels}
    no_jump = np.ones(N)  # P(no jump so far) per path (RB over the jump draws)
    out: dict = {}
    c = coefs or (fld.mu.coef, fld.logD.coef, fld.loglam.coef)
    nu = fld.nu()
    for j in range(h):
        f_bar = rep(org.fut_f[:, j])
        co = coords_from(vf, vm, vs, jx, sum24, f_bar, fld.cfg)
        X = np.column_stack([co[k] for k in COORDS[:6]] + [exo])
        Ts = fld.designs(fld.Xs(X))
        mu, D, p = components(Ts, c, np.log(vf))
        if not drift:
            mu = np.zeros_like(mu)
        no_jump *= 1 - p
        if j == h - 1 and target is not None:
            R = np.repeat(np.asarray(target, float), n_paths) - x
            lp = mixture_logpdf(R, mu, D, p, fld.kou, nu).reshape(n_o, n_paths)
            cd = mixture_cdf(R, mu, D, p, fld.kou, nu).reshape(n_o, n_paths)
            out["logpdf"] = _logmeanexp(lp, 1)
            out["pit"] = cd.mean(1)
            cd0 = mixture_cdf(-x, mu, D, p, fld.kou, nu).reshape(n_o, n_paths)
            out["p_up"] = 1 - cd0.mean(1)
            out["mean"] = (x + mu + np.sqrt(D) * p * fld.kou.mean).reshape(n_o, n_paths).mean(1)
        sd = np.sqrt(D)
        jump = rng.random(N) < p
        K = np.where(jump, fld.kou.sample(rng, N), 0.0)  # always N draws: common random numbers across variants
        g = rng.standard_normal(N)
        tt = rng.standard_t(nu, N) * math.sqrt((nu - 2) / nu) if nu is not None else g
        r = mu + sd * (np.where(jump, g, tt) + K)  # t core off the jump branch, N(0,1) + K on it
        x1 = x + r
        for b in levels:
            d0, d1 = b - x, b - x1
            crossed = d0 * d1 <= 0
            pt = np.where(crossed, 1.0, np.exp(-2.0 * np.maximum(d0 * d1, 0.0) / D))
            surv[b] *= 1 - pt
        vf, vm, vs, jx, _ = step(vf, vm, vs, jx, r, f_bar, fld.cfg)
        drop = rep(org.last24[:, j]) if j < 24 else sim_r[:, j - 24]
        sum24 = sum24 + r - drop
        if sim_r is not None:
            sim_r[:, j] = r
        x = x1
    term = x.reshape(n_o, n_paths)
    if keep_terminal:
        out["terminal"] = term
    out["p_jump"] = 1 - no_jump.reshape(n_o, n_paths).mean(1)
    out["p_hit"] = {b: 1 - s.reshape(n_o, n_paths).mean(1) for b, s in surv.items()}
    out["vf_end"] = vf.reshape(n_o, n_paths).mean(1)
    return out


def quantiles(terminal: np.ndarray, qs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)) -> np.ndarray:
    return np.quantile(terminal, qs, axis=-1).T


# ------------------------------------------------------------------ Fokker-Planck checks


def _face(pl_, pr_, mu, central):
    """Face value of p for the drift flux: central (2nd order) where the cell Peclet number is < 2, upwind elsewhere
    (upwinding alone adds numerical diffusion |mu| dx / 2)."""
    return np.where(central, 0.5 * (pl_ + pr_), np.where(mu > 0, pl_, pr_))


def fp1d(mu, D, grid: np.ndarray, p0: np.ndarray, T: float, cfl: float = 0.4) -> np.ndarray:
    """Density at time T of dX = mu(X) dt + sqrt(D(X)) dW (callables), finite volume on a uniform grid, upwind drift
    flux, central diffusion flux on D p (Ito), zero-flux walls. Mass is conserved to rounding."""
    x = np.asarray(grid, float)
    dx = x[1] - x[0]
    xf = 0.5 * (x[1:] + x[:-1])
    muf = np.asarray(mu(xf), float)
    Dc = np.asarray(D(x), float)
    dt = cfl * min(dx * dx / max(Dc.max(), 1e-300), dx / max(np.abs(muf).max(), 1e-300))
    central = np.abs(muf) * dx < 0.5 * np.minimum(Dc[1:], Dc[:-1]) * 2.0  # cell Peclet < 2: central flux is stable
    n = max(1, int(math.ceil(T / dt)))
    dt = T / n
    p = np.asarray(p0, float).copy()
    p /= p.sum() * dx
    for _ in range(n):
        F = muf * _face(p[:-1], p[1:], muf, central) - 0.5 * (Dc[1:] * p[1:] - Dc[:-1] * p[:-1]) / dx
        p[:-1] -= dt / dx * F
        p[1:] += dt / dx * F
    return p


def fp2d_logprice_vol(sigma, mu_v, D_v, xg: np.ndarray, vg: np.ndarray, p0: np.ndarray, T: float, cfl: float = 0.35) -> np.ndarray:
    """Density p(x, v) at T for dX = sigma(v) dW1, dv = mu_v(v) dt + sqrt(D_v(v)) dW2 (independent), p0 on (len(xg),
    len(vg)). Dimension-split finite volume, explicit Euler."""
    dx, dv = xg[1] - xg[0], vg[1] - vg[0]
    s2 = np.asarray(sigma(vg), float) ** 2          # (nv,)
    vf_ = 0.5 * (vg[1:] + vg[:-1])
    muf = np.asarray(mu_v(vf_), float)
    Dv = np.asarray(D_v(vg), float)
    dt = cfl * min(dx * dx / s2.max(), dv * dv / max(Dv.max(), 1e-300), dv / max(np.abs(muf).max(), 1e-300))
    central = np.abs(muf) * dv < np.minimum(Dv[1:], Dv[:-1])
    n = max(1, int(math.ceil(T / dt)))
    dt = T / n
    p = np.asarray(p0, float).copy()
    p /= p.sum() * dx * dv
    for _ in range(n):
        # x: pure diffusion with v-dependent coefficient (D does not depend on x)
        Fx = -0.5 * s2[None, :] * (p[1:, :] - p[:-1, :]) / dx
        p[:-1, :] -= dt / dx * Fx
        p[1:, :] += dt / dx * Fx
        Fv = muf[None, :] * _face(p[:, :-1], p[:, 1:], muf[None, :], central[None, :]) - 0.5 * (Dv[None, 1:] * p[:, 1:] - Dv[None, :-1] * p[:, :-1]) / dv
        p[:, :-1] -= dt / dv * Fv
        p[:, 1:] += dt / dv * Fv
    return p


def mc_logprice_vol(sigma, mu_v, D_v, v0: float, T: float, n_paths: int = 20000, substeps: int = 20, seed: int = 0) -> np.ndarray:
    """Euler MC of the same 2-D SDE (terminal log price), for the FP cross-check."""
    rng = np.random.default_rng(seed)
    n = int(round(T * substeps))
    dt = T / n
    x = np.zeros(n_paths)
    v = np.full(n_paths, float(v0))
    for _ in range(n):
        x += np.asarray(sigma(v)) * math.sqrt(dt) * rng.standard_normal(n_paths)
        v += np.asarray(mu_v(v)) * dt + np.sqrt(np.maximum(np.asarray(D_v(v)), 0.0) * dt) * rng.standard_normal(n_paths)
    return x


def cdf_quantiles(grid: np.ndarray, dens: np.ndarray, qs) -> np.ndarray:
    d = np.asarray(dens, float)
    c = np.cumsum(d) - 0.5 * d  # CDF at the cell centres (a plain cumsum sits at the right edges: half-cell bias)
    c /= d.sum()
    return np.interp(qs, c, grid)


def norm_ppf(q):
    return special.ndtri(q)


__all__ = ["MAX_PATHS", "Origins", "cdf_quantiles", "fp1d", "fp2d_logprice_vol", "mc_logprice_vol", "origins_from_state", "quantiles", "simulate"]
