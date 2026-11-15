"""Execution simulator v0 (src/ai_trader/sim, H-SIM1..H-SIM4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ai_trader.sim.fill_model import FillModel, fill_prob_static, fit, quote_exposure
from ai_trader.sim.flow_data import Grid
from ai_trader.sim.flow_sim import FlowSim, facts_of_grid
from ai_trader.sim.reaction import ols_cluster, stacked_wald
from ai_trader.sim.replay import Account, Inputs, MmCfg, Rule, grid_params, round_to_step, run_mm, sigma_1h, target_quotes

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------ helpers

def toy_model(symbol: str = "TOYUSDT") -> FlowSim:
    """Hand-set generator (3 s bars): ~3 trades per candle, ~1 bp sigma, 1-tick spread around 1 bp."""
    rng = np.random.default_rng(1)
    return FlowSim(symbol=symbol, sigma=1.0, depth=2e5, trade_size=50.0, tick_bps=1.0,
                   act=[0.3, 0.25, 0.5, 0.008, 0.98], nb_alpha=0.8, branching=0.9, cop=[0.3, 0.9, 0.3, 0.999],
                   n_quant=np.quantile(np.random.default_rng(2).negative_binomial(1.5, 1.5 / 4.5, 5000), np.linspace(0, 1, 2001)), obi=[0.0, 0.9], obi_sd=0.05, obi_df=10.0,
                   dirn=[0.0, 0.25, 0.25, 0.05, 0.01, 0.1], dir_sd=[0.9, 0.8, 0.6, 0.5, 0.4], dir_df=6.0,
                   vol=[np.log(50.0), 1.0], vol_sd=0.8, vol_df=8.0, spr=[0.0, 0.0], spr_sd=0.05, spr_df=10.0, spr_floor=1.0,
                   ret=[20.0, 0.5, -0.05], ret_h=[-1.0, 0.5, 0.2, 0.2], har_c=0.05, vol_rho=0.3, a_max=7.0, ret_zsd=1.0, ret_df=5.0, pi0=[0.7, 0.3],
                   clips={"obi": 8.0, "dir": 8.0, "vol": 8.0, "spr": 8.0, "ret": 12.0}, fit_stats={"mean_trades": 3.0},
                   exc={0: rng.normal(0.2, 0.3, (500, 2)), 1: rng.normal(0.0, 0.2, (500, 2)), 2: rng.normal(0.2, 0.3, (500, 2))})


def sol_rule() -> Rule:
    return Rule(tick_size=0.01, qty_step=0.1, min_qty=0.1, min_notional=5.0)


def mm_yaml() -> MmCfg:
    return MmCfg.from_yaml(ROOT / "config" / "mm.yaml")


def inp(inv: float, avg: float, **kw) -> Inputs:
    return Inputs(bid1=89.99, ask1=90.00, sigma=0.004, inv_qty=inv, avg_entry=avg, **kw)


# ---------------------------------------------------------------------------------------- fill model

@pytest.mark.parametrize("variant", ["touch", "strict"])
def test_fill_probability_decreases_with_distance(variant):
    rng = np.random.default_rng(0)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, 400)))
    highs, lows = mid * (1 + np.abs(rng.normal(0, 1e-4, 400))), mid * (1 - np.abs(rng.normal(0, 1e-4, 400)))
    m = FillModel(variant, theta0=0.2, b=0.3, eps=1e-4)
    probs = [fill_prob_static(m, +1, 100.0 * (1 - d / 1e4), highs, lows, 0.01) for d in range(0, 60, 5)]
    assert all(a >= b - 1e-12 for a, b in zip(probs, probs[1:], strict=False))
    assert probs[0] > probs[-1]
    asks = [fill_prob_static(m, -1, 100.0 * (1 + d / 1e4), highs, lows, 0.01) for d in range(0, 60, 5)]
    assert all(a >= b - 1e-12 for a, b in zip(asks, asks[1:], strict=False))


def test_fill_model_mle_recovers_queue_factor():
    """Synthetic quotes filled with theta0 = 0.3 on touch trade-throughs: the MLE recovers it."""
    rng = np.random.default_rng(3)
    n = 20_000
    t0 = 1_800_000_000_000
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    lows, highs = mid * (1 - np.abs(rng.normal(0, 2e-4, n))), mid * (1 + np.abs(rng.normal(0, 2e-4, n)))
    g = Grid("X", t0 + np.arange(n) * 3000, np.ones(n, bool), np.ones(n, bool), mid, np.full(n, 1.0), highs, lows,
             np.ones(n), np.ones(n), np.ones(n), np.ones(n), np.zeros(n), np.zeros(n), np.full(n, 1e6))
    true = FillModel("touch", 0.3, 0.0, 0.0)
    quotes = []
    for k in range(600):
        a = int(rng.integers(0, n - 100))
        price = float(mid[a] * (1 - 3e-4))
        fill_ms = None
        for j in range(a + 1, a + 20):
            if rng.random() < true.p_candle(+1, price, highs[j], lows[j], 0.0):
                fill_ms = int(g.t_ms[j]) + 1000
                break
        quotes.append({"symbol": "X", "side": 1, "price": price, "created_ms": int(g.t_ms[a + 1]), "end_ms": int(g.t_ms[a + 20]), "fill_ms": fill_ms})
    ex, info = quote_exposure(quotes, {"X": g}, {"X": 0.0})
    m, _ = fit(ex, "touch")
    assert abs(m.theta0 - 0.3) < 0.08
    assert m.eps < 0.01


# -------------------------------------------------------------------------------------- replay / port

def test_round_to_step_exact():
    assert round_to_step(0.3, 0.1, "down") == 0.3
    assert round_to_step(90.120993, 0.01, "up") == 90.13
    assert round_to_step(89.707016, 0.01, "down") == 89.70


def test_quote_rule_flat_book_hand_checked():
    """mm.rs flat_book_is_skewed_bearish_and_minimum_size, plus the hand-computed prices for config/mm.yaml:
    r = mid = 89.995; ask = r (1 + 0.35 * 0.004) = 90.12099 -> 90.13; bid = r (1 - 0.80 * 0.004) = 89.70702 -> 89.70."""
    b, a = target_quotes(mm_yaml(), sol_rule(), inp(0.0, 0.0))
    mid = 89.995
    assert a.price - mid < mid - b.price
    assert (a.qty, b.qty) == (0.1, 0.1)
    assert not a.reduce_only and not b.reduce_only
    assert (b.price, a.price) == (89.70, 90.13)


def test_quote_rule_caps_and_reduce_scaling():
    c = mm_yaml()
    b, a = target_quotes(c, sol_rule(), inp(-0.3, 91.0))
    assert a is None and b.reduce_only and b.qty >= 0.1
    b0, _ = target_quotes(c, sol_rule(), inp(0.0, 0.0))
    assert b.price > b0.price
    assert target_quotes(c, sol_rule(), inp(0.0, 0.0, gross_others_usd=55.0)) == (None, None)
    c3 = replace(c, reduce_lots_max=3)
    assert abs(target_quotes(c3, sol_rule(), inp(-0.3, 91.0))[0].qty - 0.3) < 1e-9
    assert abs(target_quotes(c3, sol_rule(), inp(-0.1, 91.0))[0].qty - 0.1) < 1e-9
    assert target_quotes(replace(c3, max_long_usd=2.0), sol_rule(), inp(0.0, 0.0))[0] is not None


def test_quote_rule_take_profit_exit_flow_tox_and_crossing():
    c = mm_yaml()
    b, a = target_quotes(c, sol_rule(), inp(-0.1, 91.0))
    assert b.price == 89.99 and a is not None
    assert target_quotes(c, sol_rule(), inp(-0.1, 89.5))[0].price < 89.99
    b, a = target_quotes(c, sol_rule(), inp(-0.1, 89.5, exit_only=True))
    assert a is None and b.price == 89.99
    assert target_quotes(c, sol_rule(), inp(0.0, 0.0, exit_only=True)) == (None, None)
    b0, a0 = target_quotes(c, sol_rule(), inp(0.0, 0.0))
    b1, a1 = target_quotes(c, sol_rule(), inp(0.0, 0.0, flow=1.0))
    assert a1.price > a0.price and b1.price > b0.price
    b2, a2 = target_quotes(c, sol_rule(), inp(0.0, 0.0, tox=-0.5))
    assert a2.price > a0.price and b2.price < b0.price
    assert target_quotes(c, sol_rule(), inp(0.0, 0.0, tox=-1.0)) == (None, None)
    b, a = target_quotes(c, sol_rule(), Inputs(bid1=89.99, ask1=90.0, sigma=1e-6))
    assert b.price < 90.0 and a.price > 89.99
    assert target_quotes(c, sol_rule(), Inputs(bid1=90.0, ask1=90.0, sigma=0.004)) == (None, None)
    s = sigma_1h([1.0] * 10, 15.0)
    assert s == 0.0015


def test_account_conserves_cash_and_inventory_and_charges_fees():
    rng = np.random.default_rng(5)
    acc = Account()
    flows = []
    for _ in range(300):
        side = 1 if rng.random() < 0.5 else -1
        qty = float(rng.integers(1, 5)) / 10
        px = float(100 + rng.normal(0, 1))
        if rng.random() < 0.7:
            acc.limit_fill(side, qty, px)
            fee = qty * px * 2.0 / 1e4
            flows.append((side, qty, px, fee))
        else:
            mid, spr, depth = px, 2.0, 1e5
            exec_px = acc.market(side, qty, mid, spr, depth)
            assert (exec_px - mid) * side > 0
            flows.append((side, qty, exec_px, qty * exec_px * 5.5 / 1e4))
    inv = sum(s * q for s, q, _, _ in flows)
    cash = -sum(s * q * p + f for s, q, p, f in flows)
    assert abs(acc.inv - inv) < 1e-9
    assert abs(acc.cash - cash) < 1e-9
    assert abs(acc.fees - sum(f for *_, f in flows)) < 1e-9
    assert abs(acc.turnover - sum(q * p for _, q, p, _ in flows)) < 1e-9
    assert abs(acc.equity(101.0) - (cash + inv * 101.0)) < 1e-9


def test_replay_ends_flat_and_pays_fees():
    m = toy_model()
    g = m.generate(12_000, np.random.default_rng(2), mid0=90.0, t0_ms=1_800_000_000_000)
    fm = FillModel("touch", 0.3, 0.1, 1e-4)
    cfg = replace(MmCfg(), sigma_floor_bps=5.0, ask_offset_sigma=0.05, bid_offset_sigma=0.05)
    out = run_mm(g, sol_rule(), cfg, fm, np.random.default_rng(0), start=6000)
    assert out["maker_fills"] > 0
    assert out["fees"] >= out["turnover"] * 2.0 / 1e4 - 1e-9
    assert out["fees"] <= out["turnover"] * 5.5 / 1e4 + 1e-9
    assert np.isfinite(out["pnl"])
    assert len(grid_params()) >= 24


# ------------------------------------------------------------------------------------------ generator

def test_generator_reproduces_its_own_fitted_moments():
    m0 = toy_model()
    g0 = m0.generate(60_000, np.random.default_rng(11), mid0=100.0, t0_ms=1_800_000_000_000)
    m1 = FlowSim.fit(g0)
    g1 = m1.generate(60_000, np.random.default_rng(12), mid0=100.0, t0_ms=1_800_000_000_000)
    f0, _ = facts_of_grid(g0)
    f1, _ = facts_of_grid(g1)
    s0, s1 = np.std(np.diff(np.log(g0.mid))), np.std(np.diff(np.log(g1.mid)))
    assert abs(s1 / s0 - 1) < 0.15
    n0, n1 = (g0.n_buy + g0.n_sell).mean(), (g1.n_buy + g1.n_sell).mean()
    assert abs(n1 / n0 - 1) < 0.15
    assert abs(np.median(g1.spread_bps) / np.median(g0.spread_bps) - 1) < 0.1
    assert abs(f1["flow_acf"] - f0["flow_acf"]) < 0.05
    assert abs(f1["impact"] - f0["impact"]) < 0.1
    assert abs(m1.pi0[0] - m0.pi0[0]) < 0.03 and abs(m1.pi0[1] - m0.pi0[1]) < 0.03


def test_rescale_changes_sigma_as_requested():
    m = toy_model()
    g = m.generate(30_000, np.random.default_rng(4))
    base = float(np.std(np.diff(np.log(g.mid))) * 1e4)
    m2 = m.rescale(target_sigma=2.5 * base, target_depth=5e4, target_trade_size=20.0, iters=3)
    g2 = m2.generate(40_000, np.random.default_rng(9))
    got = float(np.std(np.diff(np.log(g2.mid))) * 1e4)
    assert abs(got / (2.5 * base) - 1) < 0.1
    tr = g2.n_buy + g2.n_sell > 0
    per_trade = (g2.buy + g2.sell)[tr] / (g2.n_buy + g2.n_sell)[tr]
    assert np.median(per_trade) < 0.6 * 50.0  # trade notional moved toward the target size


# ------------------------------------------------------------------------------------------ reaction

def test_ols_cluster_and_wald_equal_coefficients():
    rng = np.random.default_rng(8)
    blocks = []
    for _ in range(3):
        n = 6000
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        cl = np.arange(n) // 50
        f = ols_cluster(y, np.column_stack([np.ones(n), x]), cl)
        assert abs(f.beta[1] - 0.5) < 0.1
        blocks.append((f.beta[1:2], f.clusters, f.infl[:, 1:2]))
    w = stacked_wald(blocks)
    assert w["df"] == 2 and w["p"] > 0.001
    est, cl, inf = blocks[2]
    blocks[2] = (est + 0.5, cl, inf)
    assert stacked_wald(blocks)["p"] < 1e-6
