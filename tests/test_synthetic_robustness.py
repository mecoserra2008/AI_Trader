"""The vectorised zoo engine of research.synthetic_robustness_study must reproduce strategies.trend_zoo +
evaluate.strategy_pnl exactly (no lake access: a synthetic 4h frame)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.research import synthetic_robustness_study as srs
from ai_trader.research.evaluate import strategy_pnl
from ai_trader.research.frame import enrich_bars
from ai_trader.strategies.trend_zoo import zoo


def _frame(n: int = 1500, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    r = 0.01 * rng.standard_t(4, n) + 0.002 * np.sin(np.arange(n) / 80)  # some trend so the signals trade
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    ev = [t0 + timedelta(hours=4 * i) for i in range(n)]
    return enrich_bars(pl.DataFrame({"event_ts": ev, "close_ts": [e + timedelta(hours=4) for e in ev], "open": o, "high": h, "low": lo, "close": c}))


def test_enrich_vec_matches_enrich_bars():
    f = _frame()
    sn, fwd = srs.enrich_vec(f["close"].to_numpy()[None, :])
    assert np.allclose(sn[0], f["sigma_next"].to_numpy(), equal_nan=True, rtol=1e-12)
    assert np.allclose(fwd[0], f["fwd_ret"].to_numpy(), equal_nan=True)


def test_vectorised_zoo_matches_production():
    f = _frame()
    specs = srs.variant_specs()
    c, h, lo = (f[k].to_numpy()[None, :] for k in ("close", "high", "low"))
    s, fwd = f["sigma_next"].to_numpy()[None, :], f["fwd_ret"].to_numpy()[None, :]
    cost = srs._cost_bps()
    for name, fn in zoo(6).items():
        ref = fn(f)
        mine = srs.signal(specs[name], c, h, lo, s)[0]
        assert np.max(np.abs(ref.to_numpy() - mine)) < 1e-9, name
        rp = strategy_pnl(f, ref)["pnl"].to_numpy()
        mp = srs.pnl_vec(mine[None, :], s, fwd, cost)[0]
        assert np.max(np.abs(rp - mp)) < 1e-9, name
    # several paths at once give the same answer as one at a time
    c2 = np.vstack([c[0], c[0] * 1.5])
    sn2, fwd2 = srs.enrich_vec(c2)
    h2, lo2 = np.vstack([h[0], h[0] * 1.5]), np.vstack([lo[0], lo[0] * 1.5])
    two = srs.signal(specs["donch_14"], c2, h2, lo2, sn2)
    assert np.allclose(two[0], two[1])


def test_breakout_grid_is_fixed():
    bo = [k for k, v in srs.variant_specs().items() if v[0] == "bo"]
    assert bo == ["bo_5d_x0.25", "bo_5d_x0.5", "bo_10d_x0.25", "bo_10d_x0.5", "bo_20d_x0.25", "bo_20d_x0.5", "bo_40d_x0.25", "bo_40d_x0.5"]
    assert len([k for k, v in srs.variant_specs().items() if v[0] == "zoo"]) == 21


def test_verdict_map():
    assert srs._verdict(-0.2, -0.3, -0.1, "neg") == "supports"
    assert srs._verdict(-0.2, -0.3, 0.1, "neg") == "inconclusive"
    assert srs._verdict(0.1, -0.3, 0.4, "neg") == "contradicts"
    assert srs._verdict(0.2, 0.1, 0.3, "pos") == "supports"
    assert srs._verdict(-0.1, -0.3, 0.1, "pos") == "contradicts"


def test_pbo_cscv_noise_vs_dominant():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(1600, 20))
    r = srs.pbo_cscv(noise, 16)
    assert r["n_splits"] == 12870
    assert 0.3 < r["pbo"] < 0.75  # no true skill: the IS winner is a coin flip OOS
    dom = noise.copy()
    dom[:, 3] += 0.5  # one configuration with a large true edge
    r2 = srs.pbo_cscv(dom, 16)
    assert r2["pbo"] < 0.01 and r2["is_winner_counts"][3] == r2["n_splits"]


def test_cpcv_select_picks_true_edge():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1200, 8))
    x[:, 5] += 0.3
    assert srs.cpcv_select(x)["choice"] == 5
