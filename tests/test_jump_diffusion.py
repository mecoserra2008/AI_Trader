import math

import numpy as np

from ai_trader.features.jump_diffusion import (
    JumpParams,
    bipower_sigma,
    fit_hawkes,
    fit_mixture,
    hawkes_intensity,
    lee_mykland,
    posterior,
    simulate_bracket,
)


def _sim(n=30000, sigma=0.005, p=0.01, p_up=0.4, mu_up=0.03, mu_dn=0.04, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * sigma
    j = rng.random(n) < p
    up = rng.random(n) < p_up
    size = np.where(up, rng.exponential(mu_up, n), -rng.exponential(mu_dn, n))
    return e + j * size, j


def test_bipower_ignores_jumps():
    r, _ = _sim()
    s = bipower_sigma(r, 72)
    assert abs(np.nanmedian(s) / 0.005 - 1) < 0.12
    # plain std is inflated by jumps: var = s^2 + p * E[J^2] = 2.5e-5 + 0.01 * (0.4*2*0.03^2 + 0.6*2*0.04^2) -> std 0.00717
    assert abs(np.std(r) / math.sqrt(2.5e-5 + 0.01 * (0.4 * 2 * 0.03**2 + 0.6 * 2 * 0.04**2)) - 1) < 0.05


def test_mixture_recovers_parameters_and_posterior_flags_jumps():
    r, j = _sim()
    sig = np.full(len(r), 0.005)
    prm = fit_mixture(r, sig)
    assert abs(prm.p / 0.01 - 1) < 0.35
    assert abs(prm.mean_up / 0.03 - 1) < 0.35 and abs(prm.mean_down / 0.04 - 1) < 0.35
    assert 0.25 < prm.p_up < 0.55
    post = posterior(r, sig, prm)
    big = j & (np.abs(r) > 6 * 0.005)
    assert post[big].mean() > 0.9
    assert post[~j].mean() < 0.02


def test_lee_mykland_threshold():
    _, thr = lee_mykland(np.zeros(3), np.ones(3), n_per_day=24, alpha=0.01)
    assert 4.5 < thr < 5.2


def _sim_hawkes(mu, a, b, T, seed=2):
    rng = np.random.default_rng(seed)
    t, ev, lam_bar = 0.0, [], mu
    while t < T:
        lam_bar = mu + a * sum(math.exp(-b * (t - s)) for s in ev[-50:])
        t += rng.exponential(1 / (lam_bar + a))
        lam_t = mu + a * sum(math.exp(-b * (t - s)) for s in ev[-50:])
        if rng.random() < lam_t / (lam_bar + a) and t < T:
            ev.append(t)
    return np.array(ev)


def test_hawkes_detects_clustering_but_not_poisson():
    T = 40000.0
    ev = _sim_hawkes(0.005, 0.4, 0.5, T)
    mu, a, b, lr = fit_hawkes(ev, T)
    assert a > 0.1 and lr > 20
    pois = np.sort(np.random.default_rng(3).uniform(0, T, 200))
    _, a2, _, lr2 = fit_hawkes(pois, T)
    assert lr2 < 10
    lam = hawkes_intensity(np.r_[1, np.zeros(9)].astype(bool), 0.01, 0.4, 0.5)
    assert lam[0] == 0.01 and lam[1] > lam[5] > 0.01


def test_bracket_matches_driftless_geometry_without_jumps():
    prm = JumpParams(p=0.0, p_up=0.5, eta_up=30, eta_down=30)
    sym = simulate_bracket(100.0, 98.0, 102.0, 0.004, prm, bars=3000, n_paths=3000, use_jumps=False)
    assert abs(sym["p_tp"] - 0.5) < 0.04 and sym["p_open"] < 0.02
    three_r = simulate_bracket(100.0, 99.0, 100.0 * math.exp(3 * math.log(100 / 99)), 0.004, prm, bars=4000, n_paths=3000, use_jumps=False)
    assert abs(three_r["p_tp"] - 0.25) < 0.04


def test_negative_jump_skew_hurts_longs_more_than_shorts():
    prm = JumpParams(p=0.02, p_up=0.2, eta_up=1 / 0.02, eta_down=1 / 0.02)
    lo = simulate_bracket(100.0, 98.0, 102.0, 0.003, prm, bars=500, n_paths=4000, seed=5)
    sh = simulate_bracket(100.0, 102.0, 98.0, 0.003, prm, bars=500, n_paths=4000, seed=5)
    assert lo["p_tp"] < sh["p_tp"] - 0.05
    assert lo["p_stop_worse_1p5R"] > 0.0
