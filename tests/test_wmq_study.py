"""Tests for the WMQ study helpers (pure parts: grid, GLFT / AS algebra, Poisson fit, edges, selection statistics)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ai_trader.research import wmq_report as R
from ai_trader.research import wmq_study as S


def fake_cal() -> dict:
    p = {"A": 0.01, "kappa": 2.0, "xstar": 1.5, "eps": {f"{r}_s{s:g}": 0.1 for r in ("glft", "as") for s in S.S_CAP},
         "A_deep": 0.001, "kappa_deep": 2.5, "eps_deep": {"glft": 0.12, "as": 0.12}}
    return {"symbols": {s: {"params": p, "fit": None} for s in S.SYMBOLS}}


def test_universe_rule_and_weekends():
    assert S.extra_names() == S.EXTRA
    assert len(set(S.SYMBOLS)) == 12 and "SPYUSDT" not in S.SYMBOLS
    assert len(S.phase_days("cal")) == 12 and len(S.phase_days("sel")) == 22 and len(S.phase_days("hold")) == 42
    assert max(S.phase_days("sel")) < S.KNOWLEDGE_CUTOFF <= min(S.phase_days("hold"))


def test_grid_is_complete_and_unique():
    g = S.grid(fake_cal(), "XLMUSDT", "sel")
    names = [v["name"] for v in g]
    assert len(names) == 42 == len(set(names))  # 38 registered + 4 A2 (amendment 1)
    assert sum(n.startswith("A2_") for n in names) == 4
    assert [v["name"] for v in S.grid(None, "XLMUSDT", "cal")] == ["PROBE"]
    b = {v["name"]: v for v in g}
    assert b["B_m1_lin"]["lots"] == 3 and b["B_m1_lin"]["cap_usd"] == 75.0 and b["B_m1_lin"]["overrides"]["symbol_loss_limit_usd"] == 4.5
    assert b["D_xstar"]["depth_sigma"] == 1.5
    assert b["A_as_s0.5_adj"]["add_sigma"] == 0.1 and b["A_as_s0.5_adj"]["add_bps"] == 2.0 and "add_sigma" not in b["A_as_s0.5_p"]
    assert b["F_D_m1_btc"]["btc_k"] == 2.0 and b["T_A_glft_s0.5_p_trend"]["trend_k"] == 1.5


def test_glft_gamma_solves_the_skew_and_as_algebra():
    for s in (0.05, 0.1, 0.2):
        g = S.solve_glft_gamma(s, 0.01, 2.0)
        assert abs(S.glft_c2(g, 0.01, 2.0) - s) < 1e-9
        b, a = S.glft_hs(g, 0.01, 2.0, 1.0)
        b0, a0 = S.glft_hs(g, 0.01, 2.0, 0.0)
        assert abs((b - b0) - s) < 1e-9 and abs((a0 - a) - s) < 1e-9 and abs(b0 - a0) < 1e-12
    b, a = S.as_hs(0.1, 2.0, 2.0)
    h = math.log(1.05) / 0.1 + 0.05
    assert abs(b - (h + 0.2)) < 1e-12 and abs(a - (h - 0.2)) < 1e-12


def test_poisson_fit_recovers_an_exponential_intensity():
    rng = np.random.default_rng(1)
    x = np.repeat(np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0]), 40)
    e = rng.uniform(5000, 20000, len(x))
    n = rng.poisson(e * 0.02 * np.exp(-1.7 * x))
    f = S.poisson_fit(n, e, x)
    assert abs(-f["b1"] - 1.7) < 4 * f["se1"] and abs(math.exp(f["b0"]) - 0.02) < 0.004


def test_edges_in_bp():
    f = pl.DataFrame({"s": [1, -1], "price": [100.0, 100.0], "qty": [1.0, 1.0], "fee": [0.02, 0.02], "mid": [100.1, 99.9],
                      "mid_1s": [100.0, 100.0], "mid_10s": [100.0, 100.0], "mid_60s": [99.9, 100.2], "mid_300s": [100.0, 100.0], "place_sigma": [0.004, 0.004]})
    e = S.add_edges(f)
    assert np.allclose(e["spread_bp"].to_numpy(), [10.0, 10.0])
    assert np.allclose(e["fee_bp"].to_numpy(), [2.0, 2.0])
    assert np.allclose(e["e_60s_bp"].to_numpy(), [10.0 - 20.0 - 2.0, 10.0 - 30.0 - 2.0])
    assert np.allclose(e["adv_cost_60_sigma"].to_numpy(), [20.0 / 40.0, 30.0 / 40.0])


def test_bootstrap_pbo_and_dsr_behave():
    rng = np.random.default_rng(3)
    v = rng.normal(1.0, 1.0, 400)
    days = np.repeat(np.arange(40), 10)
    b = R.boot(v, days)
    assert b["ci"][0] < 1.0 < b["ci"][1] and b["ci"][0] > 0.5
    # PBO is conditional on the sample (a lucky noise column can win both halves): average over noise draws ~ 0.5
    ps = [R.pbo_mc(rng.normal(0, 1, (22, 8)), n_splits=200)["pbo"] for _ in range(20)]
    assert 0.3 < float(np.mean(ps)) < 0.7
    m = rng.normal(0, 1, (22, 8))
    m[:, 3] += 3.0
    assert R.pbo_mc(m, n_splits=300)["pbo"] < 0.05
    d = R.deflated_sharpe(rng.normal(0.5, 1, 30), 37)
    assert 0 <= d["dsr"] <= d["psr0"] <= 1
