"""Tests for ai_trader.field (H-FLD1-3 machinery): Kramers-Moyal recovery, sparse fits, propagation, online updates,
and the point-in-time state."""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.field.kramers_moyal import (
    Kou,
    binned_moments,
    fit_field_em,
    jump_moments,
    kernel_moments,
    mixture_cdf,
    mixture_logpdf,
    ou_correct,
)
from ai_trader.field.model import Field, coords_matrix
from ai_trader.field.online import OnlineGLM, run_online
from ai_trader.field.poly import (
    SparsePoly,
    Standardizer,
    design,
    glm_fit,
    monomials,
    stability_support,
    stlsq,
)
from ai_trader.field.propagate import (
    Origins,
    cdf_quantiles,
    fp1d,
    fp2d_logprice_vol,
    mc_logprice_vol,
    simulate,
)
from ai_trader.field.state import (
    COORDS,
    StateConfig,
    build_state,
    coords_from,
    hour_profile,
    load_frame,
    step,
)

from .conftest import make_bars


def _ou(n, theta, m, D, dt, seed=0):
    rng = np.random.default_rng(seed)
    a = math.exp(-theta * dt)
    s = math.sqrt(D / (2 * theta) * (1 - a * a))
    x = np.empty(n)
    x[0] = m
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = m + a * (x[t - 1] - m) + s * e[t]
    return x


# ------------------------------------------------------------------ Kramers-Moyal


def test_ou_drift_diffusion_recovered_with_finite_dt_correction():
    theta, m, D, dt = 0.4, 0.5, 0.04, 1.0
    x = _ou(200_000, theta, m, D, dt)
    fit = ou_correct(x[:-1], np.diff(x), dt)
    assert abs(fit.theta - theta) < 0.02
    assert abs(fit.mean - m) < 0.01
    assert abs(fit.D - D) / D < 0.03
    # the naive (first-order) KM estimate is biased at this step size
    assert fit.theta_naive < theta - 0.05 and fit.D_naive < D * 0.85
    # binned M1 follows the exact discrete drift -(1 - e^{-theta dt})(x - m)
    sd = math.sqrt(D / (2 * theta))
    b = binned_moments(x[:-1], np.diff(x), np.linspace(m - 2 * sd, m + 2 * sd, 11), dt)
    exact = -(1 - math.exp(-theta * dt)) * (b["center"] - m)
    assert np.nanmax(np.abs(b["m1"] - exact)) < 0.01
    k = kernel_moments(x[:-1], np.diff(x), np.array([m]), bw=sd / 5, dt=dt)
    assert abs(k["m1"][0]) < 0.005


def test_jump_diffusion_moments_recover_rate_and_size():
    rng = np.random.default_rng(1)
    n, D, lam, eta = 1_000_000, 1e-4, 0.02, 1 / 0.01
    dy = math.sqrt(D) * rng.standard_normal(n)
    jumps = rng.random(n) < lam
    sz = rng.exponential(1 / eta, jumps.sum()) * np.where(rng.random(jumps.sum()) < 0.5, 1, -1)
    dy[jumps] += sz
    r = jump_moments(float(np.mean(dy**2)), float(np.mean(dy**4)), float(np.mean(dy**6)), 1.0, "laplace")
    assert r["ok"]
    assert abs(r["lam"] - lam) / lam < 0.15
    assert abs(r["eta"] - eta) / eta < 0.08
    assert abs(r["D"] - D) / D < 0.05


def test_em_recovers_state_dependent_jump_diffusion():
    rng = np.random.default_rng(2)
    n = 80_000
    x = rng.standard_normal(n)
    lnvf = np.log(1e-4) + 0.3 * rng.standard_normal(n)
    D = np.exp(lnvf + 0.1 + 0.4 * x)
    p = np.exp(-4.0 + 0.6 * x)
    kou = Kou(0.45, 0.25, 0.2)
    j = rng.random(n) < p
    r = np.sqrt(D) * (rng.standard_normal(n) + np.where(j, kou.sample(rng, n), 0.0))
    Xs = x[:, None]
    lin = monomials(1, 1)
    fit, _ = fit_field_em(r, lnvf, Xs, ("x",), [lin[0]], lin, lin, n_em=8, core="gauss")
    assert np.allclose(fit.logD.coef, [0.1, 0.4], atol=0.05)
    assert np.allclose(fit.loglam.coef, [-4.0, 0.6], atol=0.35)
    assert abs(fit.mu.coef[0]) < 0.02
    assert abs(1 / fit.kou.eta_up - 4.0) < 1.0 and abs(1 / fit.kou.eta_down - 5.0) < 1.2


def test_em_t_core_recovers_degrees_of_freedom():
    rng = np.random.default_rng(21)
    n = 60_000
    lnvf = np.full(n, np.log(1e-4))
    nu = 4.5
    r = 0.01 * rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    one = [(0,)]
    fit, _ = fit_field_em(r, lnvf, np.zeros((n, 1)), ("x",), one, one, one, n_em=10, core="t")
    assert abs(fit.t_dof - nu) < 0.8
    assert fit.w.mean() < 0.02  # a pure t sample is not read as jumps
    assert abs(fit.logD.coef[0]) < 0.12  # sample variance of a t(4.5) is noisy


# ------------------------------------------------------------------ sparse polynomial discovery


def _planted(n=20_000, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    monos = monomials(3, 3)
    truth = {(0, 0, 0): 1.0, (1, 0, 0): 2.0, (1, 2, 0): -0.5, (0, 0, 3): 0.3}
    return X, monos, truth


def test_stlsq_recovers_planted_polynomial():
    X, monos, truth = _planted()
    T = design(X, monos)
    y = sum(c * T[:, monos.index(m)] for m, c in truth.items()) + 0.5 * np.random.default_rng(4).standard_normal(len(X))
    sup, res = stlsq(T, y, t_min=4.0)
    assert {monos[i] for i in sup} == set(truth)
    est = dict(zip([monos[i] for i in sup], res.coef, strict=True))
    for m, c in truth.items():
        assert abs(est[m] - c) < 0.05


def test_stability_selection_recovers_planted_log_link_polynomial():
    X, monos, _ = _planted(30_000, 5)
    truth = {(0, 0, 0): 0.2, (1, 0, 0): 0.5, (0, 1, 1): -0.3}
    T = design(X, monos)
    eta = sum(c * T[:, monos.index(m)] for m, c in truth.items())
    y = np.exp(eta) * np.random.default_rng(6).standard_normal(len(X)) ** 2
    sup, prob = stability_support(T, y, np.arange(len(y)) // 24, "gamma", n_sub=30, pi_thr=0.7, ev=1.0)
    assert {monos[i] for i in sup} == set(truth)
    res = glm_fit(T[:, sup], y, "gamma")
    est = dict(zip([monos[i] for i in sup], res.coef, strict=True))
    for m, c in truth.items():
        assert abs(est[m] - c) < 0.06


# ------------------------------------------------------------------ propagation


def _const_field(c_D=0.0, log_p=-30.0, mu=0.0, kou=Kou(), t_dof=float("nan")):
    names = tuple(COORDS)
    one = [tuple([0] * len(names))]
    mk = lambda c, link, fam: SparsePoly(names, one, np.array([c]), np.array([0.0]), link, fam)
    std = Standardizer(names, np.zeros(len(names)), np.ones(len(names)))
    return Field(std, mk(mu, "identity", "gaussian"), mk(c_D, "log", "gamma"), mk(log_p, "log", "poisson"), kou, np.ones(168),
                 StateConfig(clip=50.0), t_dof=t_dof)


def _org(n_o, h, vf=1e-4):
    return Origins(np.full(n_o, vf), np.full(n_o, vf), np.full(n_o, vf), np.zeros(n_o), np.zeros((n_o, 24)), np.zeros((n_o, 3)), np.ones((n_o, h)))


def test_propagation_reproduces_known_moments():
    fld = _const_field()
    for h in (1, 24):
        sim = simulate(fld, _org(2, h), h, 50_000, seed=7)
        var = sim["terminal"].var(1)
        assert np.all(np.abs(var / (h * 1e-4) - 1) < 0.04)  # E vf is a martingale -> Var R_h = h vf0
        assert np.all(np.abs(sim["terminal"].mean(1)) < 3 * math.sqrt(h * 1e-4 / 50_000) + 1e-5)
    # one step with jumps: Var = D (1 + p E K^2 - (p E K)^2)
    kou = Kou(0.4, 0.5, 0.35)
    fj = _const_field(log_p=math.log(0.05), kou=kou)
    sim = simulate(fj, _org(1, 1), 1, 200_000, seed=8)
    want = 1e-4 * (1 + 0.05 * kou.second - (0.05 * kou.mean) ** 2)
    assert abs(sim["terminal"].var() / want - 1) < 0.03
    # the t core keeps unit variance off the jump branch
    s1 = simulate(_const_field(log_p=math.log(0.05), kou=kou, t_dof=4.0), _org(1, 1), 1, 200_000, seed=18)
    assert abs(s1["terminal"].var() / want - 1) < 0.06
    # Rao-Blackwellised density integrates to one and PITs of model draws are uniform (t core)
    fj = _const_field(log_p=math.log(0.05), kou=kou, t_dof=4.0)
    grid = np.linspace(-0.15, 0.15, 3001)
    o = _org(len(grid), 4)
    s4 = simulate(fj, o, 4, 200, seed=9, target=grid)
    assert abs(np.sum(np.exp(s4["logpdf"])) * (grid[1] - grid[0]) - 1) < 0.01
    draws = simulate(fj, _org(1, 4), 4, 4000, seed=10)["terminal"][0]
    pit = simulate(fj, _org(len(draws), 4), 4, 300, seed=11, target=draws)["pit"]
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    assert np.max(np.abs(hist - 0.1)) < 0.025


def test_hit_probability_matches_reflection_principle():
    fld = _const_field()
    sd = 0.01
    sim = simulate(fld, _org(1, 24, vf=sd**2 / 24), 24, 40_000, seed=12, levels=(0.01, -0.01))
    # driftless Brownian motion: P(max_T >= b) = 2 (1 - Phi(b / (sd sqrt T)))
    from scipy.stats import norm
    want = 2 * (1 - norm.cdf(0.01 / sd))
    assert abs(sim["p_hit"][0.01][0] - want) < 0.02 and abs(sim["p_hit"][-0.01][0] - want) < 0.02


def test_fokker_planck_ou_and_2d_against_mc():
    theta, m, D = 0.5, 0.0, 0.2
    g = np.linspace(-3, 3, 301)
    p0 = np.exp(-0.5 * ((g - 1.0) / 0.05) ** 2)
    p = fp1d(lambda x: -theta * (x - m), lambda x: np.full_like(x, D), g, p0, 2.0)
    dx = g[1] - g[0]
    mean = np.sum(g * p) * dx
    var = np.sum((g - mean) ** 2 * p) * dx
    assert abs(np.sum(p) * dx - 1) < 1e-9
    assert abs(mean - math.exp(-theta * 2.0)) < 0.01
    want_var = D / (2 * theta) * (1 - math.exp(-2 * theta * 2.0)) + 0.05**2 * math.exp(-2 * theta * 2.0)
    assert abs(var / want_var - 1) < 0.03
    sigma = lambda v: np.exp(v)
    mu_v = lambda v: -0.1 * (v + 5.0)
    D_v = lambda v: np.full_like(np.asarray(v, float), 0.01)
    xg, vg = np.linspace(-0.12, 0.12, 121), np.linspace(-5.8, -4.2, 41)
    p0 = np.zeros((len(xg), len(vg)))
    p0[60, np.argmin(np.abs(vg + 4.8))] = 1.0
    P = fp2d_logprice_vol(sigma, mu_v, D_v, xg, vg, p0, 24.0)
    px = P.sum(1)
    q_fp = cdf_quantiles(xg, px, [0.05, 0.25, 0.5, 0.75, 0.95])
    q_mc = np.quantile(mc_logprice_vol(sigma, mu_v, D_v, vg[np.argmin(np.abs(vg + 4.8))], 24.0, 40_000, 10), [0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.max(np.abs(q_fp - q_mc)) < 0.003


def test_mixture_cdf_matches_density():
    kou = Kou(0.4, 0.6, 0.45)
    r = np.linspace(-0.2, 0.2, 40001)
    mu, D, p = np.full_like(r, 0.001), np.full_like(r, 1e-4), np.full_like(r, 0.08)
    dens = np.exp(mixture_logpdf(r, mu, D, p, kou))
    c = np.cumsum(dens) * (r[1] - r[0])
    F = mixture_cdf(r, mu, D, p, kou)
    assert np.max(np.abs(c - F)) < 2e-3


# ------------------------------------------------------------------ online


def test_online_rls_matches_batch_and_tracks_drift():
    rng = np.random.default_rng(13)
    n = 4000
    x = rng.standard_normal(n)
    T = np.column_stack([np.ones(n), x])
    y = 0.5 + 1.5 * x + 0.3 * rng.standard_normal(n)
    poly = SparsePoly(("x",), [(0,), (1,)], np.zeros(2), np.ones(2), "identity", "gaussian", phi=0.09)
    o = OnlineGLM(poly, forget=1.0, P0=np.eye(2) * 1e6)
    for i in range(n):
        o.update(T[i], y[i])
    b = np.linalg.lstsq(T, y, rcond=None)[0]
    assert np.allclose(o.b, b, atol=1e-3)
    # a slope that changes halfway: forgetting tracks, the frozen batch fit does not
    y2 = 0.5 + np.where(np.arange(n) < n // 2, 1.5, -0.5) * x + 0.3 * rng.standard_normal(n)
    fz = glm_fit(T[: n // 2], y2[: n // 2], "gaussian")
    poly2 = SparsePoly(("x",), [(0,), (1,)], fz.coef, np.sqrt(np.diag(fz.cov)), "identity", "gaussian", phi=fz.phi, cov=fz.cov_model)
    o2 = OnlineGLM(poly2, forget=1 - 1 / 300)
    err_on = []
    for i in range(n // 2, n):
        err_on.append((y2[i] - T[i] @ o2.b) ** 2)
        o2.update(T[i], y2[i])
    err_fz = (y2[n // 2:] - T[n // 2:] @ fz.coef) ** 2
    assert np.mean(err_on) < 0.5 * np.mean(err_fz)


def test_run_online_uses_only_past_outcomes():
    fld = _const_field(c_D=0.0, log_p=math.log(0.01))
    fld.logD.cov = np.eye(1) * 0.01
    fld.mu.cov = np.eye(1) * 0.01
    fld.loglam.cov = np.eye(1) * 0.01
    n = 50
    X = np.zeros((n, len(COORDS)))
    lnvf = np.full(n, math.log(1e-4))
    r = np.full(n, 0.01)  # every outcome is a 1-sd move... except we poison the last one
    r[-1] = 10.0
    Cm, Cd, Cl = run_online(fld, X, lnvf, r)
    Cm2, Cd2, Cl2 = run_online(fld, X, lnvf, np.r_[r[:-1], 0.0])
    assert np.array_equal(Cd[-1], Cd2[-1]) and np.array_equal(Cl[-1], Cl2[-1])  # origin n-1 cannot see r_{n}


# ------------------------------------------------------------------ point-in-time state


def _write(lake, n=2600):
    lake.write("bars", make_bars(n, symbol="BTCUSDT"))
    return make_bars(n, symbol="BTCUSDT")["known_ts"]


def test_state_is_point_in_time(lake):
    kts = _write(lake)
    t1, t2 = kts[1900], kts[2500]
    f1, f2 = load_frame(t1, lake=lake), load_frame(t2, lake=lake)
    assert f1["known_ts"].max() <= t1
    prof = hour_profile(f1, t1)
    # the profile known at t1 is the same whether or not later bars exist in the frame
    assert np.allclose(prof, hour_profile(f2, t1))
    s1, s2 = build_state(f1, prof), build_state(f2, prof)
    s2 = s2.filter(pl.col("known_ts") <= t1)
    cols = [c for c in COORDS] + ["vf", "vm", "vs", "jx", "flag"]
    assert s1.height == s2.height and s1.height > 100
    for c in cols:
        assert np.allclose(s1[c].to_numpy().astype(float), s2[c].to_numpy().astype(float), equal_nan=True), c
    # r_next is the only label: in s1 the last row's label is unknown at t1
    assert s1["r_next"][-1] is None


def test_vectorised_step_matches_state_recursion(lake):
    kts = _write(lake, 2200)
    f = load_frame(kts[-1], lake=lake)
    prof = np.linspace(0.6, 1.6, 168)
    st = build_state(f, prof)
    cfg = StateConfig()
    i0 = 50
    vf, vm, vs, jx = (np.array([st[c][i0]]) for c in ("vf", "vm", "vs", "jx"))
    r = st["ret"].to_numpy()
    fb = st["f"].to_numpy()
    for i in range(i0 + 1, i0 + 200):
        vf, vm, vs, jx, _ = step(vf, vm, vs, jx, r[i], fb[i], cfg)
    j = i0 + 199
    assert np.allclose([vf[0], vm[0], vs[0], jx[0]], [st["vf"][j], st["vm"][j], st["vs"][j], st["jx"][j]], rtol=1e-10)
    co = coords_from(vf, vm, vs, jx, np.array([st["sum24"][j]]), np.array([st["f_next"][j]]), cfg)
    X = coords_matrix(st.slice(j, 1))[0]
    assert np.allclose([co[c][0] for c in COORDS[:6]], X[:6])


def test_field_json_roundtrip(tmp_path):
    fld = _const_field(c_D=0.2, log_p=-5.0)
    fld.meta = {"fit_as_of": "2026-07-01T00:00:00+00:00"}
    p = fld.save(tmp_path / "f.json")
    g = Field.load(p)
    X = np.zeros((3, len(COORDS)))
    a = fld.logpdf(np.array([0.0, 0.01, -0.02]), X, np.full(3, math.log(1e-4)))
    b = g.logpdf(np.array([0.0, 0.01, -0.02]), X, np.full(3, math.log(1e-4)))
    assert np.allclose(a, b)
    assert g.meta["fit_as_of"].startswith("2026-07-01")


_ = (pytest, timedelta)
