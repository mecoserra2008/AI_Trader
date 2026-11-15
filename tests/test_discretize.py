"""Discretised state spaces (ai_trader.ml.discretize, H-DS1..H-DS3): training-only edges, EB shrinkage, honest trees."""

from __future__ import annotations

import numpy as np

from ai_trader.ml.discretize import (CategoricalDim, HonestTree, KMeansCodebook, Lattice, QuantileDim, cell_effects,
                                     date_cell_sums, eb_shrink, honest_halves)


def _lattice() -> Lattice:
    return Lattice([QuantileDim("x", 0, 3), CategoricalDim("wd", 1, ((1.0,), (2.0, 3.0, 4.0, 5.0), (6.0,), (7.0,)), ("Mon", "Tue-Fri", "Sat", "Sun"))])


def test_cell_assignment_uses_only_training_edges():
    rng = np.random.default_rng(0)
    Xtr = np.c_[rng.normal(size=3000), rng.integers(1, 8, 3000)]
    lat = _lattice().fit(Xtr)
    e0 = lat.dims[0].edges.copy()
    np.testing.assert_allclose(e0, np.quantile(Xtr[:, 0], [1 / 3, 2 / 3]))
    # test rows far outside the training range: edges don't move, rows land in the extreme training bins
    Xte = np.c_[np.r_[np.full(50, 100.0), np.full(50, -100.0)], np.full(100, 6.0)]
    cells = lat.assign(Xte)
    np.testing.assert_array_equal(lat.dims[0].edges, e0)
    assert set(cells[:50]) == {2 * 4 + 2} and set(cells[50:]) == {0 * 4 + 2}
    # assigning the test set together with or without other rows gives the same cells
    both = lat.assign(np.r_[Xte, Xtr])
    np.testing.assert_array_equal(both[:100], cells)
    assert lat.n_cells == 12 and "Sat" in lat.describe(int(cells[0]))
    # k-means: standardisation and centroids from training rows only
    km = KMeansCodebook(4, (0,), n_init=2, max_iter=50).fit(Xtr)
    mu, c = km.mu.copy(), km.centroids.copy()
    km.assign(Xte)
    np.testing.assert_allclose(km.mu, mu)
    np.testing.assert_allclose(km.centroids, c)
    assert np.isclose(km.mu[0], Xtr[:, 0].mean())


def test_shrinkage_toward_zero_with_few_observations():
    rng = np.random.default_rng(1)
    n_dates = 400
    # three big cells with real effects -0.3 / 0 / +0.3, one tiny cell (3 rows) with a huge raw mean
    vals, cells, dates = [], [], []
    for c, eff in enumerate((-0.3, 0.0, 0.3)):
        d = rng.integers(0, n_dates, 2000)
        vals.append(eff + rng.normal(0, 1, 2000))
        cells.append(np.full(2000, c))
        dates.append(d)
    vals.append(np.array([3.0, 2.5, 3.5]))
    cells.append(np.full(3, 3))
    dates.append(np.array([1, 2, 3]))
    v, c, d = np.concatenate(vals), np.concatenate(cells), np.concatenate(dates)
    ce = cell_effects(v, c, d, 4)
    assert np.isfinite(ce.kappa) and ce.kappa > 0
    assert ce.w[3] < 0.2 and abs(ce.shrunk[3]) < 0.25 * abs(ce.mean[3])
    assert ce.w[0] > 0.9 and abs(ce.shrunk[0] - ce.mean[0]) < 0.05
    assert abs(ce.shrunk[3]) <= abs(ce.mean[3])
    # pure noise: tau2 = 0 -> everything shrinks to 0
    v0 = rng.normal(0, 1, 4000)
    ce0 = cell_effects(v0, rng.integers(0, 8, 4000), rng.integers(0, 300, 4000), 8)
    if ce0.tau2 == 0:
        assert np.allclose(ce0.shrunk, 0.0, atol=1e-9)
    else:
        assert np.all(np.abs(ce0.shrunk) <= np.abs(np.nan_to_num(ce0.mean)) + 1e-12)
    # empty cells get 0, and value() maps unknown ids to 0
    ce2 = cell_effects(v, c, d, 6)
    assert ce2.shrunk[4] == 0.0 and ce2.shrunk[5] == 0.0 and ce2.value(np.array([5, 99]))[1] == 0.0
    # date clustering: perfectly correlated rows within a date carry one date's information
    S, N = date_cell_sums(np.repeat(rng.normal(0, 1, 50), 20), np.zeros(1000, int), np.repeat(np.arange(50), 20), 1)
    e1 = eb_shrink(S, N)
    assert e1.var[0] > 10 * (1.0 / 1000) * 0.5  # far above the iid variance of the mean


def test_honest_tree_leaf_values_come_from_estimation_half():
    rng = np.random.default_rng(2)
    n = 8000
    dates = np.datetime64("2024-01-01") + rng.integers(0, 700, n).astype("timedelta64[D]")
    X = rng.normal(size=(n, 3))
    split = honest_halves(dates)
    assert 0.4 < split.mean() < 0.6
    # same dates always in the same half
    for dd in np.unique(dates)[:20]:
        assert len(set(split[dates == dd])) == 1
    # split half: strong signal on x0; estimation half: the opposite sign
    y = np.where(split, np.where(X[:, 0] > 0, 1.0, -1.0), np.where(X[:, 0] > 0, -2.0, 2.0)) + rng.normal(0, 0.1, n)
    tree = HonestTree(max_depth=1, min_leaf_abs=100).fit(X[split], y[split])
    assert tree.n_cells == 2 and tree.nodes[0].feature == 0
    leaf = tree.assign(X)
    est = ~split
    ce = cell_effects(y[est], leaf[est], dates[est], tree.n_cells)
    hi_leaf = int(leaf[np.argmax(X[:, 0])])
    # the value attached to the high-x0 leaf is the estimation half's (about -2 relative to its pooled mean), not +1
    assert ce.shrunk[hi_leaf] < -1.5
    exp_mean = y[est & (leaf == hi_leaf)].mean() - y[est].mean()
    assert abs(ce.mean[hi_leaf] - exp_mean) < 1e-9
    # splits are a function of the split half only: changing estimation-half rows doesn't change the tree
    y2 = y.copy()
    y2[est] = rng.normal(size=est.sum())
    tree2 = HonestTree(max_depth=1, min_leaf_abs=100).fit(X[split], y2[split])
    assert tree2.nodes[0].threshold == tree.nodes[0].threshold and tree2.nodes[0].feature == tree.nodes[0].feature
    # depth cap and leaf count
    deep = HonestTree(max_depth=6, min_leaf_abs=20).fit(X[split], y[split])
    assert deep.n_cells <= 64 and deep.assign(X).max() < deep.n_cells
