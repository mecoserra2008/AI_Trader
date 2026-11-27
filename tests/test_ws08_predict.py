import numpy as np

from ai_trader.research import ws08_predict as w


def _sim(T, rho, delta, beta, rng):
    u = rng.standard_normal(T + 1)
    e = delta * u + np.sqrt(1 - delta**2) * rng.standard_normal(T + 1)
    x = np.zeros(T + 1)
    for t in range(1, T + 1):
        x[t] = rho * x[t - 1] + u[t]
    # z1[t] = return over (t, t+1] = beta x[t] + e[t+1]
    z1 = beta * x[:-1] + e[1:]
    return z1, x[:-1]


def test_ivx_size_persistent_regressor():
    rng = np.random.default_rng(0)
    rej_ivx, rej_ivx5, rej_ols = 0, 0, 0
    R = 200
    for _ in range(R):
        z1, x = _sim(800, 0.995, -0.95, 0.0, rng)
        rej_ivx += w.ivx_wald(z1, x, k=1)["p"] < 0.05
        rej_ivx5 += w.ivx_wald(z1, x, k=5)["p"] < 0.05
        rej_ols += abs(w.ols_nw(z1, x, 0)["t"]) > 1.96
    assert 0.0 < rej_ivx / R < 0.11
    assert rej_ivx5 / R < 0.20      # long-horizon variant over-rejects mildly with rho 0.995, T 800 (documented)
    assert rej_ols >= rej_ivx       # OLS-t over-rejects with a persistent, endogenous regressor


def test_ivx_power():
    rng = np.random.default_rng(1)
    z1, x = _sim(2000, 0.95, -0.5, 0.08, rng)
    assert w.ivx_wald(z1, x, k=1)["p"] < 0.05


def test_hodrick_h1_is_white():
    rng = np.random.default_rng(2)
    z1, x = _sim(500, 0.5, 0.0, 0.0, rng)
    m = np.ones(len(x), bool)
    assert abs(w.hodrick_t(z1, x, 1, m) - w.ols_nw(z1, x, 0)["t"]) < 1e-9


def test_sums_and_slopes():
    r = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    f = w.forward_sum_valid(r, 2)
    assert np.isnan(f[0]) and np.isnan(f[1]) and f[2] == 9.0 and f[3] == 11.0 and np.isnan(f[4])
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal(100), rng.standard_normal(100)
    S = np.array([100, x.sum(), y.sum(), (x * y).sum(), (x * x).sum()])
    assert abs(w.slopes_from_stats(S) - np.polyfit(x, y, 1)[0]) < 1e-10


def test_romano_wolf_dominates_single():
    rng = np.random.default_rng(4)
    tb = np.abs(rng.standard_normal((999, 20)))
    t = np.abs(rng.standard_normal(20)) + np.linspace(0, 3, 20)
    padj = w.romano_wolf(t, tb)
    pind = np.mean(tb >= t, axis=0)
    assert np.all(padj >= pind - 1e-12)
