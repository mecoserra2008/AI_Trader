import numpy as np

from ai_trader.ml.hourly import HOURLY_FEATURES
from ai_trader.research import ws14_interp as W


def _quantiles(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q50 = rng.normal(0, 0.05, n)
    w = np.exp(rng.normal(0, 0.2, n))
    base = np.array([-0.95, -0.5, -0.25, 0.0, 0.25, 0.5, 0.95])
    return q50[:, None] + w[:, None] * base[None, :]


def test_rebuild_is_exact_for_a_fixed_shape():
    q = _quantiles()
    s = W.summaries(q)
    shape = W.fit_shape(q)
    qr = W.rebuild_quantiles(s["q50"], s["w"], np.exp(s["ltr"]), shape)
    assert np.allclose(qr, q, atol=1e-12)
    assert np.all(np.diff(qr, axis=1) >= 0)


def test_r2_and_ols():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = 1 + X @ np.array([0.5, -1.0, 2.0])
    b = W._ols(X, y)
    assert np.allclose(b, [1, 0.5, -1.0, 2.0])
    assert abs(W.r2(y, W._pred(b, X)) - 1) < 1e-12
    assert abs(W.r2(y, np.full_like(y, y.mean()))) < 1e-12


def test_perturb_last_return_is_consistent():
    feats = list(HOURLY_FEATURES)
    X = np.zeros((2, len(feats)))
    X[:, feats.index("lu_1")] = np.log(0.5)
    X[:, feats.index("lu_m6")] = np.log(1.0)
    X[:, feats.index("lu_m24")] = np.log(1.0)
    X[:, feats.index("since_jump")] = 0.5
    Xn = W.perturb_last_return(X, 2.0, feats)
    assert np.allclose(Xn[:, feats.index("lu_1")], np.log(4.0))
    assert np.allclose(np.exp(Xn[:, feats.index("lu_m6")]), 1 + (4.0 - 0.5) / 6)
    assert np.allclose(np.exp(Xn[:, feats.index("lu_m24")]), 1 + (4.0 - 0.5) / 24)
    assert np.all(Xn[:, feats.index("jump24")] == 0) and np.all(Xn[:, feats.index("since_jump")] == 0.5)
    Xj = W.perturb_last_return(X, 4.5, feats)
    assert np.all(Xj[:, feats.index("jump24")] == 1) and np.allclose(Xj[:, feats.index("since_jump")], 1 / 168)
    others = [i for i, c in enumerate(feats) if c not in ("lu_1", "lu_m6", "lu_m24", "jump24", "since_jump")]
    assert np.array_equal(Xj[:, others], X[:, others])


def test_rowwise_spearman():
    a = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    r = W.rowwise_spearman(a, np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    assert np.allclose(r, [1.0, -1.0])
