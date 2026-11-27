"""WS05 estimators on synthetic data: LM size, BNS / ASJ size and power, Hawkes recovery, time-rescaling, MCS, DM."""

import math

import numpy as np

from ai_trader.research import ws05_tails as w


def _diffusion(n_days=200, n=288, seed=0, jump_every=None):
    rng = np.random.default_rng(seed)
    T = n_days * n
    ts = np.arange(T, dtype=np.int64) * (86_400_000 // n)
    hod = (ts // 3_600_000) % 24
    season = 1 + 0.5 * np.sin(2 * np.pi * hod / 24)
    r = 0.001 * season * rng.standard_normal(T)
    if jump_every:
        idx = rng.choice(T, T // jump_every, replace=False)
        r[idx] += 0.012 * rng.choice([-1, 1], len(idx))
    return w.Series("syn", ts, r, n)


def test_lm_threshold_values():
    assert abs(w.lm_threshold(288) - 4.305) < 0.01
    assert abs(w.lm_threshold(1440) - 4.609) < 0.01


def test_lm_size_under_diffusion():
    s = w.lm_detect(_diffusion())
    dt = w.day_tests(s)
    # daily family-wise false-positive rate near 1% (periodicity filtered); allow sampling slack
    assert dt["lm"].mean() < 0.05
    assert dt["bns"].mean() < 0.05
    assert dt["asj"].mean() < 0.06


def test_tests_detect_jumps():
    s = w.lm_detect(_diffusion(jump_every=288))
    dt = w.day_tests(s)
    assert dt["lm"].mean() > 0.5
    assert dt["bns"].mean() > 0.3
    assert dt["asj"].mean() > 0.2  # ASJ(4,2) has low power at n = 288: critical S ~ 1.0


def _sim_hawkes(T=200_000, mu=0.002, alpha=0.2, kappa=0.8, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    E = 0.0
    for t in range(T):
        lam = mu + alpha * E
        y[t] = rng.random() < 1 - math.exp(-lam)
        E = kappa * E + y[t]
    return y


def test_hawkes_recovers_parameters_and_passes_gof():
    y = _sim_hawkes()
    T = len(y)
    m = w.HModel(y[:, None], np.ones((T, 1)), np.zeros((T, 1, 0)), np.zeros((T, 1)), np.array([0]), [(y, 0, np.array([0]))], 1, 1, 1)
    r = m.fit(np.array([math.log(0.001), math.log(0.1), math.log(0.1)]))
    a, b, la, lb = m.split(r.x)
    assert abs(math.exp(a[0]) - 0.002) < 0.0006
    assert abs(math.exp(la[0]) - 0.2) < 0.06
    assert abs(math.exp(-math.exp(lb[0])) - 0.8) < 0.06
    g = w.gof(m.rescaled(r.x))
    assert g["ks_p"] > 0.01
    # analytic gradient matches finite differences
    f0, g0 = m.nll(r.x + 0.01)
    e = np.zeros(3)
    e[2] = 1e-5
    fd = (m.nll(r.x + 0.01 + e)[0] - m.nll(r.x + 0.01 - e)[0]) / 2e-5
    assert abs(fd - g0[2]) < 1e-2 * max(1, abs(fd))


def test_poisson_fails_gof_on_hawkes_data():
    y = _sim_hawkes(alpha=0.5)
    T = len(y)
    m = w.HModel(y[:, None], np.ones((T, 1)), np.zeros((T, 1, 0)), np.zeros((T, 1)), np.array([0]), [], 1, 0, 0)
    r = m.fit(np.array([math.log(0.003)]))
    assert w.gof(m.rescaled(r.x))["ks_p"] < 0.01


def test_dm_and_mcs():
    rng = np.random.default_rng(0)
    d = 0.1 + rng.standard_normal(2000)
    assert w.dm_hln(d, 10)["t"] > 2
    L = rng.standard_normal((3000, 3)) + np.array([0.0, 0.0, 0.5])
    out = w.mcs(L, ["a", "b", "c"], B=300)
    assert "c" not in out["mcs"]


def test_kappa():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    assert abs(w.kappa(a, a)["kappa"] - 1) < 1e-12
