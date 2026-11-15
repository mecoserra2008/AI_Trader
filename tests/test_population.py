"""Population model (sim/population.py, H-MS7): cross-sectional fits, leave-one-out R^2, universality test, and
params_for monotone in W for the order rate and the spread."""

import math
from pathlib import Path

import numpy as np
import pytest

from ai_trader.sim import flow_sim_v1 as V
from ai_trader.sim import population as POP

MODEL = Path(__file__).resolve().parents[1] / "data" / "models" / "population" / "population_2026-11-10.json"


def _rows(n: int = 45, seed: int = 1) -> list[dict]:
    """Synthetic symbols obeying invariance-like laws (rate ~ W^2/3, order size ~ W^1/3, spread falling with W for the
    free-spread ones, tick-bound when the tick is large relative to the natural spread)."""
    rng = np.random.default_rng(seed)
    rows = []
    cls = ["major"] * 5 + ["alt"] * (n - 15) + ["tradfi"] * 10
    for i, c in enumerate(cls):
        lw = rng.uniform(math.log(1e4), math.log(1e9))
        tick_bps = math.exp(rng.uniform(math.log(0.02), math.log(8.0)))
        nat = math.exp(6.0 - 0.45 * lw + rng.normal(0, 0.2))           # natural spread (bp)
        tb = nat < 1.2 * tick_bps
        sp = tick_bps * 1.01 if tb else nat
        rows.append({
            "symbol": f"S{i}", "class": c, "W": math.exp(lw), "tick_bps": tick_bps, "tick_bound": tb, "price": 10.0,
            "spr_mu": 0.0 if tb else math.log(sp / tick_bps), "spr_sd": 0.02 if tb else 0.3,
            "log_rate": -9.0 + 2 / 3 * lw + rng.normal(0, 0.2), "log_v_ref": -1.0 + 1 / 3 * lw + rng.normal(0, 0.2),
            "log_sigma_day": math.log(0.03) + rng.normal(0, 0.2), "log_depth25": 2.0 + 0.6 * lw + rng.normal(0, 0.3),
            "log_spread_bps": math.log(sp), "log_lambda": 4.0 - 0.6 * lw + rng.normal(0, 0.3), "sign_gamma": 0.6 + rng.normal(0, 0.1),
            "sign_acf1": 0.35 + rng.normal(0, 0.05), "zeta_order": 1.3 + rng.normal(0, 0.2), "size_mu": -1.0 + 1 / 3 * lw,
            "log_k_imp": -2.0 + rng.normal(0, 0.3), "impact_corr": 0.35 + rng.normal(0, 0.03), "flow_acf_3s": 0.01 + rng.normal(0, 0.003),
            "absr_acf_3s": 0.08 + rng.normal(0, 0.02), "ppo_mean": 3.0, "size_sigma": 1.8, "log_xmin_over_vref": 3.0, "size_p_tail": 0.02,
            "log_cap_over_vref": 9.0, "mix_w": [0.25, 0.25, 0.5], "mix_mean": [0.01, 0.3, 1.8], "int_v": [0.1, 0.1], "int_phi": [0.8, 0.98],
            "sign_c": 0.3, "sign_short": {"1": 0.35, "2": 0.2, "5": 0.1, "10": 0.06, "20": 0.04}, "ppo_pmf": {k: 1.0 for k in V.PMF_KEYS},
            "ppo_tail_zeta": 1.0, "rho_kv": 0.5, "g_inf": 0.6, "tau_s": 30.0, "noise_beta": 1.0, "size_xi_share": 0.05, "size_g_share": 0.1,
            "log_sigma0_over_sigma1m": -2.0, "jump_rate": 0.0, "jump_z": [7.0, -8.0], "spr_phi": 0.9, "vr_3s_1m": 1.0, "vr5": 1.0,
            "size_ns_acf10": 0.03, "exc_rate": 1e-4, "rob_over_sigma1m": 0.8, "impact_delta": 0.5})
    return rows


def test_loo_r2_and_slopes_recover_the_law():
    rows = _rows()
    lw = np.log([r["W"] for r in rows])
    X = POP.design(lw, [r["class"] for r in rows])
    y = np.array([r["log_rate"] for r in rows])
    b, _ = POP.ols(y, X)
    assert b[1] == pytest.approx(2 / 3, abs=0.03)
    assert POP.loo_r2(y, X) > 0.9
    noise = np.random.default_rng(0).normal(size=len(y))
    assert POP.loo_r2(noise, X) < 0.1
    bs = POP.loo_r2_boot(y, X, 100, np.random.default_rng(1))
    assert np.nanpercentile(bs, 2.5) > 0.8


def test_universality_separates_noise_from_symbol_effects():
    rng = np.random.default_rng(3)
    n = 50
    F, T = 0.6 + rng.normal(0, 0.1, n), 0.6 + rng.normal(0, 0.1, n)          # universal: dispersion = noise
    u = POP.universality(F, T, 500, rng)
    assert u["D"] == pytest.approx(1.0, abs=0.6) and u["p_disp"] > 0.05
    a = rng.normal(0, 0.3, n)                                                 # persistent symbol effects
    u2 = POP.universality(0.6 + a + rng.normal(0, 0.1, n), 0.6 + a + rng.normal(0, 0.1, n), 500, rng)
    assert u2["D"] > 5 and u2["p_disp"] < 0.01


def _check_monotone(pop: POP.Population, cls: str, tick_rel: float):
    Ws = np.geomspace(1e4, 1e10, 25)
    rates = [pop.params_for(W, cls, tick_rel).rate for W in Ws]
    spreads = [pop.spread_bps(W, cls, tick_rel * 1e4)["spread_bps"] for W in Ws]
    assert np.all(np.diff(rates) > 0)
    assert np.all(np.diff(spreads) <= 1e-12)


def test_params_for_monotone_in_W_synthetic():
    pop = POP.Population.fit(_rows())
    assert pop.lin["log_rate"]["coef"][1] > 0
    for cls in ("major", "alt", "tradfi"):
        for tick_rel in (2e-6, 1e-4, 5e-4):
            _check_monotone(pop, cls, tick_rel)


def test_params_for_builds_a_runnable_generator_and_roundtrips(tmp_path):
    pop = POP.Population.fit(_rows(), exclude="S3")
    assert "S3" not in pop.symbols and pop.excluded == "S3"
    p = pop.params_for(5e7, "alt", 1e-4, price=2.0)
    assert isinstance(p, V.V1Params) and p.rate > 0 and p.size_alpha > 1
    o = V.generate(p, 1800, 1)
    assert len(o.orders["ms"]) > 0 and np.isfinite(o.grid.mid).all()
    pop.save(tmp_path / "pop.json")
    q = POP.Population.load(tmp_path / "pop.json").params_for(5e7, "alt", 1e-4, price=2.0)
    assert q.rate == pytest.approx(p.rate) and q.spr_mu == pytest.approx(p.spr_mu)


@pytest.mark.skipif(not MODEL.exists(), reason="fitted population model not built (research.population_simulator_study)")
def test_fitted_population_monotone_in_W():
    pop = POP.Population.load(MODEL)
    for cls in ("major", "alt", "tradfi"):
        for tick_rel in (1e-6, 1e-4, 4e-4):
            _check_monotone(pop, cls, tick_rel)
