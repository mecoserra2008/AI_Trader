from datetime import date, timedelta

import numpy as np
import polars as pl

from ai_trader.research.pnl_decomposition_study import decompose, effective_bets, factor_panel, ipca_fit, managed, meucci_enb, pca_stats


def _proj(G: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(G)
    return Q @ Q.T


def test_effective_bets_formula():
    assert np.isclose(effective_bets(np.ones(7)), 7.0)
    assert np.isclose(effective_bets(np.array([3.0, 0.0, 0.0])), 1.0)
    lam = np.array([2.0, 1.0, 1.0])
    p = lam / lam.sum()
    assert np.isclose(effective_bets(lam), np.exp(-(p * np.log(p)).sum()))


def test_pca_stats_one_factor_book():
    rng = np.random.default_rng(0)
    f = rng.standard_normal(2000)
    X = np.column_stack([f + 0.1 * rng.standard_normal(2000) for _ in range(8)])
    s = pca_stats(X)
    assert s["top1_share"] > 0.95 and s["enb"] < 1.3
    Y = rng.standard_normal((2000, 8))
    s2 = pca_stats(Y)
    assert s2["enb"] > 7.5


def test_ipca_recovers_simulated_loadings():
    rng = np.random.default_rng(3)
    N, T, L, K = 300, 250, 5, 2
    G_true = np.linalg.qr(rng.standard_normal((L, K)))[0]
    Z, R = [], []
    F_true = rng.standard_normal((T, K)) * np.array([2.0, 1.0]) + np.array([0.3, 0.1])
    for t in range(T):
        z = np.column_stack([np.ones(N), rng.standard_normal((N, L - 1))])
        r = z @ G_true @ F_true[t] + 0.5 * rng.standard_normal(N)
        Z.append(z)
        R.append(r)
    Ws, xs = managed(Z, R)
    G, F, it = ipca_fit(Ws, xs, K)
    assert np.allclose(G.T @ G, np.eye(K), atol=1e-8)
    assert np.linalg.norm(_proj(G) - _proj(G_true)) < 0.05
    # fitted factor paths span the true ones
    corr = np.corrcoef(np.column_stack([F, F_true]), rowvar=False)[:K, K:]
    assert np.all(np.max(np.abs(corr), axis=0) > 0.98)
    # warm start converges immediately to the same solution
    G2, _, it2 = ipca_fit(Ws, xs, K, G0=G)
    assert it2 <= 3 and np.linalg.norm(_proj(G2) - _proj(G)) < 1e-6


def test_trade_decomposition_identity():
    rng = np.random.default_rng(5)
    g = rng.standard_normal(500)
    mfe = np.maximum(g, 0) + np.abs(rng.standard_normal(500))
    c = np.abs(rng.standard_normal(500)) * 0.1
    t = pl.DataFrame({"gross_d": g, "cost_d": c, "mfe_d": mfe})
    d = decompose(t)
    assert np.isclose(d["net"], d["D_mfe"] - d["exit_leak"] - d["dir_loss"] - d["costs"])
    assert np.isclose(d["share"], (d["costs"] + d["exit_leak"]) / d["D_mfe"])
    assert 0 <= d["capture"] <= 1


def test_factor_panel_is_point_in_time():
    """Factors on dates <= D must not change when data after D is removed."""
    rng = np.random.default_rng(9)
    rows = []
    d0 = date(2024, 1, 1)
    syms = [f"S{i}USDT" for i in range(8)] + ["QQQUSDT", "PAXGUSDT", "CLUSDT"]
    for s in syms:
        for k in range(120):
            rows.append({"symbol": s, "date": d0 + timedelta(days=k), "ret": 0.02 * rng.standard_normal(), "rv": 0.0004 * (1 + rng.random()),
                         "turnover_day": 1e6 * (1 + rng.random()), "funding_sod": 1e-4 * rng.standard_normal()})
    ctx = pl.DataFrame(rows)
    univ = syms[:8]
    full = factor_panel(ctx, univ)
    cut = d0 + timedelta(days=90)
    part = factor_panel(ctx.filter(pl.col("date") <= cut), univ)
    a = full.filter(pl.col("date") <= cut).sort("date")
    b = part.sort("date")
    for c in ("MKT", "TSMOM", "VOL", "CARRY", "LIQ", "QQQ", "GOLD", "OIL"):
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        assert np.allclose(np.nan_to_num(x, nan=-9), np.nan_to_num(y, nan=-9)), c
    assert full["MKT"].drop_nulls().len() > 60


def test_meucci_enb():
    rng = np.random.default_rng(1)
    v = np.array([1.0, 2.0, 4.0, 8.0])
    X = rng.standard_normal((20000, 4)) * np.sqrt(v)  # independent, distinct variances: principal axes = the assets
    assert abs(meucci_enb(X, 1 / np.sqrt(v)) - 4.0) < 0.1  # equal risk contributions -> 4 bets
    assert abs(meucci_enb(X, np.array([1.0, 0, 0, 0])) - 1.0) < 0.05
    f = rng.standard_normal(20000)
    Y = np.column_stack([f + 0.05 * rng.standard_normal(20000) for _ in range(4)])
    assert meucci_enb(Y, np.full(4, 0.25)) < 1.1  # one common bet
