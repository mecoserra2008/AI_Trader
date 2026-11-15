"""Event-level simulator v1 (sim/flow_sim_v1.py, H-MS8): determinism, candle validity, reproduction of its own sign
ACF and size tail, estimators, calibration and the replay engine on generated paths."""

import math

import numpy as np
import pytest

from ai_trader.sim import flow_sim_v1 as V
from ai_trader.sim.laws import hill


def _p(**kw) -> V.V1Params:
    base = dict(rate=1.0, price0=50.0, tick_bps=1.0, k_imp=0.2, sigma0=0.5, spr_mu=-0.3, spr_sd=0.2, depth25=2e5, v_ref=200.0,
                size_mu=5.0, size_sigma=1.5, size_xmin=2000.0, size_alpha=2.4, size_p_tail=0.03, size_cap=1e9)
    base.update(kw)
    return V.V1Params(**base)


def test_deterministic_with_seed():
    p = _p()
    a, b = V.generate(p, 1800, 42), V.generate(p, 1800, 42)
    np.testing.assert_array_equal(a.grid.mid, b.grid.mid)
    np.testing.assert_array_equal(a.trades["price"], b.trades["price"])
    np.testing.assert_array_equal(a.orders["side"], b.orders["side"])
    c = V.generate(p, 1800, 43)
    assert not np.array_equal(a.grid.mid, c.grid.mid)


def test_candles_are_valid_ohlc():
    p = _p(jump_rate=0.002, jump_pool=[30.0, -40.0])
    o = V.generate(p, 3 * 3600, 7, t0_ms=1_000_000_000_000 // 3000 * 3000)
    g = o.grid
    n = len(g)
    assert n == 3600
    assert np.all(np.diff(g.t_ms) == 3000)
    assert g.ok.all() and g.present.all()
    has = np.isfinite(g.high)
    assert np.array_equal(has, np.isfinite(g.low))
    assert np.array_equal(has, (g.n_buy + g.n_sell) > 0)
    assert np.all(g.high[has] >= g.low[has])
    assert np.all((g.buy >= 0) & (g.sell >= 0))
    assert np.all(g.mid > 0) and np.all(np.isfinite(g.mid))
    assert np.all(g.spread_bps >= 0.99 * p.tick_bps * p.price0 / g.mid.max())
    # prints sit on the tick grid at or beyond the quoted touch: buys at or above the mid, sells at or below
    tick = p.tick_bps * p.price0 / 1e4
    k = np.round(o.trades["price"] / tick)
    assert np.allclose(o.trades["price"], k * tick, rtol=0, atol=1e-6 * tick + 1e-9)
    # notional is conserved between orders and prints
    assert math.isclose(o.trades["notional"].sum(), o.orders["notional"].sum(), rel_tol=1e-9)
    assert (o.orders["n_prints"] >= 1).all()


def test_reproduces_its_own_sign_acf():
    p = _p(rate=5.0, sign_short={"1": 0.35, "2": 0.25, "3": 0.2, "5": 0.15, "10": 0.1, "20": 0.07}, sign_c=0.3, sign_gamma=0.6)
    st = V.order_stream(p, 20_000, 3)
    s = st["s"].astype(float)
    assert len(s) > 50_000
    target = p.sign_model().target_acf(np.array([1, 10, 100]))

    def acf(x, k):
        x = x - x.mean()
        return float(x[k:] @ x[:-k] / (x @ x))

    assert acf(s, 1) == pytest.approx(target[0], abs=0.04)
    assert acf(s, 10) == pytest.approx(target[1], abs=0.03)
    assert acf(s, 100) == pytest.approx(target[2], abs=0.02)
    assert abs(s.mean()) < 0.1


@pytest.mark.parametrize("coupled", [False, True])
def test_reproduces_its_own_size_tail(coupled):
    kw = {"size_xi_share": 0.2, "size_g_share": 0.3} if coupled else {}
    p = _p(rate=5.0, size_alpha=2.5, size_p_tail=0.05, **kw)
    st = V.order_stream(p, 20_000, 5)
    v = st["v"]
    tail = v[v >= p.size_xmin]
    assert len(tail) / len(v) == pytest.approx(p.size_p_tail, abs=0.01)
    z = hill(v, len(tail))
    assert z == pytest.approx(p.size_alpha - 1.0, rel=0.12)
    if coupled:
        # the coupling reorders sizes (exact marginal) and makes them persistent in order time
        ns = V._nscore(v)
        assert np.corrcoef(ns[10:], ns[:-10])[0, 1] > 0.05


def test_arrival_rate_and_burstiness():
    p = _p(rate=2.0, int_v=[0.05, 0.05], int_phi=[0.5, 0.9])
    st = V.order_stream(p, 40_000, 9)
    assert len(st["t"]) / 40_000 == pytest.approx(2.0, rel=0.15)
    d = np.diff(st["t"])
    assert np.std(d) / np.mean(d) > 1.3          # bursts: far from Poisson (CV 1)
    assert st["m"].mean() == pytest.approx(1.0, rel=0.15)


def test_quote_path_hysteresis():
    c = np.array([0.0, 0.6, 0.7, 0.2, 0.9, 1.1, 0.4, -0.8])
    q = V.quote_path(c, 0.25)
    assert q.tolist() == [0, 0, 0, 0, 1, 1, 1, -1]
    np.testing.assert_array_equal(V.quote_path(c, 0.0), np.rint(c).astype(int))


def test_trade_stats_and_grid_stats_finite():
    p = _p(rate=2.0)
    o = V.generate(p, 4 * 3600, 2)
    st = V.sim_trade_stats(o, 4 * 3600)
    for k in ("lambda_mid", "sigma_1m", "flow_acf_3s", "absr_acf_3s", "corr_F_r3s"):
        assert np.isfinite(st[k]), k
    assert st["lambda_mid"] > 0                       # buys push the price up
    gs = V.grid_stats(o.grid)
    for k in ("kurtosis", "absr_acf", "flow_acf", "impact", "spread_med", "sigma_1m", "vr_3s_1m"):
        assert np.isfinite(gs[k]), k
    assert gs["impact"] > 0


def test_calibrate_moves_toward_targets():
    p = _p(rate=1.5, sigma0=0.2, k_imp=0.05)
    tgt = {"sigma_1m": 8.0, "impact": 0.35, "vr_3s_1m": 1.0, "vr5": 1.0}
    q, fin = V.calibrate(p, tgt, seconds=2 * 3600, iters=4, n_pilots=1, fit_stream=False)
    assert abs(fin["sigma_1m"] - 8.0) < abs(V.grid_stats(V.generate(p, 2 * 3600, 11).grid)["sigma_1m"] - 8.0)
    assert fin["impact"] == pytest.approx(0.35, abs=0.12)
    assert V.V1Params.from_dict(q.to_dict()).k_imp == q.k_imp


def test_replay_engine_runs_on_v1_paths():
    from ai_trader.sim import fill_model as fmod
    from ai_trader.sim import replay as rp

    p = _p(rate=1.0, price0=2.0, tick_bps=5.0)
    g = V.generate(p, 3 * 9000, 4, t0_ms=1_794_312_000_000 - 6000 * 3000).grid
    rule = rp.Rule(tick_size=p.tick, qty_step=1.0, min_qty=1.0, min_notional=5.0)
    out = rp.run_mm(g, rule, rp.MmCfg(), fmod.FillModel(theta0=0.45, b=20.0, eps=0.0), np.random.default_rng(0), start=6000, warmup=6000)
    assert np.isfinite(out["pnl"]) and out["decisions"] > 0
