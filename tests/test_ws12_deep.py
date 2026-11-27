import math

import numpy as np
import pytest

from ai_trader.research import ws12_deep as wd


def test_factor_ahead_is_causal():
    rng = np.random.default_rng(0)
    n = 168 * 10
    how = np.arange(n) % 168
    u = rng.chisquare(1, n)
    a = wd.factor_ahead(how, u)
    u2 = u.copy()
    u2[1500:] *= 50.0  # future shock must not change earlier values
    b = wd.factor_ahead(how, u2)
    for h in (4, 24):
        assert np.allclose(a[h][:1501], b[h][:1501], equal_nan=True)
        assert np.isnan(a[h][: 8 * 168]).all() and np.isfinite(a[h][8 * 168:]).all()


def test_pinball_and_t_quantiles():
    y = np.array([0.0, 1.0])
    q = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(wd.pinball(q, y, (0.5, 0.5)), [0.0, 0.5])
    qq = wd.t_quantiles(np.zeros(1), np.ones(1), np.full(1, 1e6), (0.5, 0.975))
    assert abs(qq[0, 0]) < 1e-9 and abs(qq[0, 1] - 1.959964) < 1e-3


def test_t_nll_matches_scipy_and_net_shapes():
    torch = pytest.importorskip("torch")
    from scipy import stats

    y, mu, sc, df = torch.tensor([0.3, -2.0]), torch.tensor([0.1, 0.0]), torch.tensor([1.2, 0.7]), torch.tensor([3.5, 8.0])
    ref = -stats.t.logpdf(y.numpy(), df.numpy(), loc=mu.numpy(), scale=sc.numpy())
    assert np.allclose(wd.t_nll_torch(y, mu, sc, df).numpy(), ref, atol=1e-5)
    for arch in ("gru", "tcn", "mlp"):
        net = wd.make_net(arch, len(wd.STEP_FEATURES), len(wd.STATIC_FEATURES), 19)
        x = torch.zeros(5, wd.L, len(wd.STEP_FEATURES)) if arch != "mlp" else torch.zeros(5, 19)
        mu_, sc_, df_, lf = net(x, torch.zeros(5, len(wd.STATIC_FEATURES)))
        assert mu_.shape == (5, 3) and (sc_ > 0).all() and (df_ > 2).all() and lf.shape == (5, 3)


def test_tcn_is_causal():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = wd.make_net("tcn", 3, 1, 1).eval()
    x = torch.randn(1, wd.L, 3)
    h1 = net.enc(x)
    x2 = x.clone()
    x2[:, :5] += 10.0  # beyond the 61-step receptive field: the last-step output must not change
    assert torch.allclose(h1, net.enc(x2), atol=1e-6)


def test_cluster_t_iid():
    from ai_trader.research.ws12_meta import cluster_t

    rng = np.random.default_rng(1)
    R = rng.normal(0.2, 1.0, 2000)
    day = np.arange(2000).astype("datetime64[D]")
    m, t, n = cluster_t(R, day)
    assert n == 2000 and abs(t - m / (R.std() / math.sqrt(2000))) < 1.5
