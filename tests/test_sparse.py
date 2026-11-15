"""Tests for ai_trader.ml.sparse (H-SP1..H-SP3): pruning masks, rewinding, stability selection, additive splines."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai_trader.ml import models as md
from ai_trader.ml import sparse as sp


def _toy(n=600, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 0.8 * X[:, 0] - 0.5 * X[:, 1] * (X[:, 2] > 0) + 0.1 * rng.standard_normal(n)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


CFG = {"hidden": (8, 4), "max_epochs": 15, "patience": 5, "batch": 64}


def test_dense_train_matches_models_fit_mlp():
    Xf, yf, Xv, yv = _toy()
    a = md.fit_mlp(Xf, yf, Xv, yv, "vol", 3, CFG)
    b = sp.train_masked(Xf, yf, Xv, yv, "vol", 3, CFG).fit
    assert np.array_equal(a.predict(Xv), b.predict(Xv))
    assert a.epochs == b.epochs


def test_masks_applied_and_kept_after_rewinding():
    Xf, yf, Xv, yv = _toy()
    ch = sp.imp_chain(Xf, yf, Xv, yv, "vol", 0, CFG, rounds=4)
    sched = sp.sparsity_schedule(6, (8, 4), 4)
    for r, (fit, mk) in enumerate(zip(ch.fits, ch.masks, strict=True)):
        assert sp.check_mask(fit, mk), f"masked weights nonzero at round {r}"
        assert sum(int(m.sum()) for m in mk.values()) == sched[r]["n_alive"]
        if r:
            prev = ch.masks[r - 1]
            assert all(not np.any(mk[k] & ~prev[k]) for k in mk), "a pruned weight came back"
            # surviving weights were trained after rewinding (not just copies of the rewind point)
            k = "layers.0.weight"
            assert np.any(fit.state[k].numpy()[mk[k]] != 0)
    # masked zeros survive a reload through MLPFit.model()
    fit, mk = ch.fits[-1], ch.masks[-1]
    w = fit.model().layers[0].weight.detach().numpy()
    assert np.all(w[~mk["layers.0.weight"]] == 0)


def test_rewind_restores_epoch1_weights_under_mask():
    Xf, yf, Xv, yv = _toy()
    dense = sp.train_masked(Xf, yf, Xv, yv, "vol", 1, CFG)
    masks = sp.prune(dense.fit.state, sp.dense_masks(6, (8, 4)))
    m = md.make_mlp(6, (8, 4))
    m.load_state_dict(dense.rewind_state)
    sp.apply_masks(m, masks)
    w0 = dense.rewind_state["layers.0.weight"].numpy()
    w1 = m.layers[0].weight.detach().numpy()
    km = masks["layers.0.weight"]
    assert np.array_equal(w1[km], w0[km]) and np.all(w1[~km] == 0)


def test_prune_removes_smallest_and_schedule():
    st = {"layers.0.weight": torch.tensor([[0.1, -2.0], [0.5, -0.05], [1.0, 3.0]]), "head.weight": torch.tensor([[1.0, 2.0, 0.01]])}
    m = {"layers.0.weight": np.ones((3, 2), bool), "head.weight": np.ones((1, 3), bool)}
    nm = sp.prune(st, m, hidden_rate=0.5, head_rate=0.34)
    assert nm["layers.0.weight"].sum() == 3
    assert not nm["layers.0.weight"][1, 1] and not nm["layers.0.weight"][0, 0] and not nm["layers.0.weight"][1, 0]
    assert nm["head.weight"].tolist() == [[True, True, False]]
    s = sp.sparsity_schedule(52, (32, 16), 22)
    assert sp.round_at(s, 0.95) == 14 and sp.round_at(s, 0.99) == 22


def test_surviving_units_and_input_paths():
    m = {"layers.0.weight": np.zeros((3, 4), bool), "layers.1.weight": np.zeros((2, 3), bool), "head.weight": np.zeros((1, 2), bool)}
    m["layers.0.weight"][0, 1] = True  # input 1 -> h1_0
    m["layers.0.weight"][2, 3] = True  # input 3 -> h1_2 (dead end: no outgoing)
    m["layers.1.weight"][1, 0] = True  # h1_0 -> h2_1
    m["head.weight"][0, 1] = True
    fit = md.MLPFit({}, (3, 2), 4, 0.0, 1.0, "vol", 0.0, 0)
    assert sp.surviving_units(fit, m) == [(0, 0), (1, 1)]
    assert sp.input_paths(m).tolist() == [False, True, False, False]


def test_stability_selection_recovers_planted_sparse_coefficients():
    rng = np.random.default_rng(1)
    n_dates, per, p = 300, 4, 60
    X = rng.standard_normal((n_dates * per, p))
    X[:, 10] = X[:, 3] + 0.3 * rng.standard_normal(len(X))  # a correlated decoy
    beta = np.zeros(p)
    beta[[0, 3, 7]] = [1.0, -0.8, 0.6]
    y = X @ beta + 1.0 * rng.standard_normal(len(X))
    groups = np.repeat(np.arange(n_dates), per)
    res = sp.stability_selection(X, y, groups, n_sub=60, alpha=0.5, seed=0)
    q = sp.q_for_bound(p, 0.8, 1.0)
    sel = set(res.selected(q, 0.8).tolist())
    assert sel == {0, 3, 7}
    assert sp.pfer_bound(res.q_hat(q), p, 0.8) <= 1.0 + 1e-9


def test_lasso_matches_soft_threshold_orthogonal():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((200, 5)))
    X = Q * np.sqrt(200)
    y = X @ np.array([2.0, -1.0, 0.3, 0.0, 0.0]) + 0.01 * rng.standard_normal(200)
    G, c = X.T @ X / 200, X.T @ y / 200
    b = sp.lasso_path_gram(G, c, np.array([0.5]))[0]
    expect = np.sign(c) * np.maximum(np.abs(c) - 0.5, 0)
    assert np.allclose(b, expect, atol=1e-6)


def test_gam_recovers_planted_two_input_function():
    rng = np.random.default_rng(2)
    n, p = 4000, 6
    X = rng.standard_normal((n, p))
    X[:, 5] = rng.integers(0, 2, n)  # a binary input
    y = np.sin(1.5 * X[:, 1]) + 0.5 * X[:, 4] ** 2 + 0.05 * rng.standard_normal(n)
    g = sp.gam_search(X[:3000], y[:3000], X[3000:], y[3000:])
    assert set(g["best2"]) == {1, 4}
    assert g["r2_ev_2"] > 0.97
    assert g["r2_ev_1"] < 0.8
    # binary one-hot path
    yb = 2.0 * X[:, 5] + np.maximum(X[:, 0], 0) + 0.05 * rng.standard_normal(n)
    gb = sp.gam_search(X[:3000], yb[:3000], X[3000:], yb[3000:])
    assert set(gb["best2"]) == {0, 5} and gb["r2_ev_2"] > 0.97


def test_forward_np_matches_torch_and_patch():
    Xf, yf, Xv, yv = _toy()
    fit = sp.train_masked(Xf, yf, Xv, yv, "vol", 0, CFG).fit
    out, _hs = sp.forward_np(fit, Xv)
    assert np.allclose(out, fit.predict(Xv), atol=1e-5)
    _out2, hs2 = sp.forward_np(fit, Xv, {(0, 0): 0.0})
    assert np.all(hs2[0][:, 0] == 0)
    att = sp.unit_attribution(fit, Xv)
    assert len(att) == 12 and all(v >= 0 for v in att.values())
