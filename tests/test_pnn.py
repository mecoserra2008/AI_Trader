import numpy as np
import pytest

pytest.importorskip("torch")

from ai_trader.ml import pnn  # noqa: E402


def test_kan_edge_fits_planted_polynomial():
    """A single KAN edge (Chebyshev in tanh(x/2) + SiLU) learns a planted cubic on its well-resolved domain and the
    symbolic extraction recovers the coefficients."""
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, size=(3000, 1))
    f = 0.4 * x[:, 0] ** 3 + 0.2 * x[:, 0] ** 2 - 0.8 * x[:, 0] + 0.3
    KAN = pnn.classes()[0]
    layer = KAN(1, 1, degree=4)
    opt = torch.optim.Adam(layer.parameters(), lr=0.03)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    X, Y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(f, dtype=torch.float32)
    for _ in range(3000):
        opt.zero_grad()
        loss = ((layer(X)[:, 0] - Y) ** 2).mean()
        loss.backward()
        opt.step()
        sch.step()
    assert float(loss.detach()) / f.var() < 0.01
    tab = pnn.edge_table(layer, x, ["x"], ["y"], max_deg=3)
    cubic = next(p for p in tab[0]["fits"] if p["deg"] == 3)
    assert cubic["r2"] > 0.99
    c = np.array(cubic["coef"])
    c[0] += float(layer.bias.detach())  # the node bias holds the constant
    assert np.allclose(c, [0.3, -0.8, 0.2, 0.4], atol=0.03)


def test_quantile_heads_never_cross():
    import torch

    torch.manual_seed(1)
    for n_in in (5, 19):
        m = pnn.make_pnn(n_in)
        mm = pnn.make_mlp(n_in, 10)
        with torch.no_grad():
            for p in list(m.parameters()) + list(mm.parameters()):
                p.mul_(20.0)
            x = torch.randn(4000, n_in) * 5
            for q in (m(x), mm(x)):
                assert torch.all(q[:, 1:] - q[:, :-1] >= 0)


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(2)
    n_cal, n_te = 2000, 20000
    y_cal, y_te = rng.standard_t(4, n_cal), rng.standard_t(4, n_te)
    lo_c, hi_c = np.full(n_cal, -0.5), np.full(n_cal, 0.5)  # deliberately too narrow base band
    lo_t, hi_t = np.full(n_te, -0.5), np.full(n_te, 0.5)
    for a in (0.2, 0.1, 0.05):
        lo, hi, _ = pnn.split_conformal(lo_t, hi_t, pnn.cqr_scores(lo_c, hi_c, y_cal), a)
        cov = np.mean((y_te >= lo) & (y_te <= hi))
        assert abs(cov - (1 - a)) < 0.015
        r = pnn.aci(lo_t, hi_t, y_te, a, gamma=0.005, window=1000, warm=pnn.cqr_scores(lo_c, hi_c, y_cal))
        assert abs(1 - r["err"].mean() - (1 - a)) < 0.01
    assert pnn.conformal_quantile(np.arange(5.0), 0.01) == float("inf")


def test_latent_dynamics_recovers_planted_A():
    rng = np.random.default_rng(3)
    A = np.array([[0.9, 0.1, 0.0], [-0.2, 0.7, 0.05], [0.0, 0.0, 0.5]])
    B = np.array([[0.3, 0.0], [0.0, -0.4], [0.2, 0.1]])
    n = 20000
    u = rng.normal(size=(n, 2))
    z = np.zeros((n, 3))
    for t in range(n - 1):
        z[t + 1] = A @ z[t] + B @ u[t] + 0.1 * rng.normal(size=3)
    d = pnn.fit_latent_dynamics(z[:-1], z[1:], u[:-1], ridge=1e-6)
    # standardised coordinates: A_std = D^-1 A D with D = diag(zsd)
    Ahat = np.diag(d.zsd) @ d.A @ np.diag(1 / d.zsd)
    Bhat = np.diag(d.zsd) @ d.B
    assert np.allclose(Ahat, A, atol=0.02)
    assert np.allclose(Bhat, B, atol=0.02)
    g = pnn.latent_r2_gain(z[:-1], z[1:], d.predict(z[:-1], u[:-1]), d.zmu, d.zsd)
    assert g["gain"] > 0.05 and abs(np.mean(g["g"]) - g["gain"]) < 1e-9


def test_parameter_counts_under_cap_and_matched():
    for n_in in (14, 19):
        m = pnn.make_pnn(n_in)
        k = pnn.n_params(m)
        assert k == pnn.pnn_params(n_in) and k <= pnn.PARAM_CAP
        h = pnn.mlp_hidden_for(n_in, k)
        mm = pnn.make_mlp(n_in, h)
        assert pnn.n_params(mm) == pnn.mlp_params(n_in, h)
        assert abs(pnn.n_params(mm) - k) / k < 0.05
        fb = pnn.make_pnn(n_in, fb=True, u_dim=4)
        assert pnn.n_params(fb) == pnn.pnn_params(n_in, fb=True) <= pnn.PARAM_CAP


def _toy(seed=4, n=3000, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.standard_t(5, n) * np.exp(0.5 * X[:, 0]) + 0.2 * X[:, 1]
    return X[: n * 4 // 5], y[: n * 4 // 5], X[n * 4 // 5:], y[n * 4 // 5:]


def test_determinism_with_seed_and_learning():
    Xf, yf, Xv, yv = _toy()
    cfg = {"batch": 256, "max_epochs": 15, "lr": 3e-3}
    a = pnn.fit_pnn(Xf, yf, Xv, yv, seed=7, cfg=cfg)
    b = pnn.fit_pnn(Xf, yf, Xv, yv, seed=7, cfg=cfg)
    assert np.array_equal(a.predict(Xv), b.predict(Xv))
    m1 = pnn.fit_mlp(Xf, yf, Xv, yv, seed=7, hidden=8, cfg=cfg)
    m2 = pnn.fit_mlp(Xf, yf, Xv, yv, seed=7, hidden=8, cfg=cfg)
    assert np.array_equal(m1.predict(Xv), m2.predict(Xv))
    # learned scale dependence beats the unconditional quantiles
    unc = np.tile(np.quantile(yf, pnn.TAUS), (len(yv), 1))
    assert pnn.pinball_np(yv, a.predict(Xv)).mean() < pnn.pinball_np(yv, unc).mean()


def test_feedback_head_starts_from_base():
    Xf, yf, Xv, yv = _toy(seed=5)
    cfg = {"batch": 256, "max_epochs": 5}
    base = pnn.fit_pnn(Xf, yf, Xv, yv, seed=0, cfg=cfg)
    zf = base.latent(Xf)
    uf = np.random.default_rng(0).normal(size=(len(Xf), 2))
    dyn = pnn.fit_latent_dynamics(zf[:-1], zf[1:], uf[:-1])
    fb = pnn.fit_head_variant(base, Xf, yf, Xv, yv, seed=0, cfg={**cfg, "max_epochs": 0}, dyn=dyn, Uf=uf,
                              Uv=np.zeros((len(Xv), 2)))
    assert np.allclose(fb.predict(Xv, np.zeros((len(Xv), 2))), base.predict(Xv), atol=1e-5)
    assert np.allclose(fb.latent(Xv), base.latent(Xv))
