"""Tests for the H-MINE2 daily-panel mining protocol (ai_trader.research.daily_mining)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ai_trader.research import daily_mining as dm


def _frame(n_dates: int = 200, n_sym: int = 5, seed: int = 0, seg: str = dm.DISC) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_dates * n_sym
    d = np.repeat(np.arange(n_dates), n_sym)
    cols = {c: rng.normal(size=n) for c in ("log_sigma", "lrv_w", "lrv_q", "dow_sin", "lrv_m", "lrv_d", "rq_ratio_d", "v0_dir_score", "funding_z_90d", "dd_90d_z")}
    cols["next_is_weekend"] = (rng.random(n) < 0.28).astype(float)
    cols["next_wk_hours"] = np.where(cols["next_is_weekend"] > 0, 24.0, 0.0)
    return pl.DataFrame({"symbol": np.tile([f"S{k}USDT" for k in range(n_sym - 1)] + ["BTCUSDT"], n_dates), "date_i": d, "block": d // 10,
                         "segment": [seg] * n, **cols, "wd": (d % 7 + 1).astype(float), "v0cell": (d % 9).astype(float),
                         "btc_v0cell": (d % 9).astype(float)})


def test_bh_synthetic_pvalues():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # BH at q = 0.05 on this classic list: thresholds 0.005, 0.010, 0.015, ... -> first two rejected
    rej = dm.bh(p, 0.05)
    assert rej.tolist() == [True, True] + [False] * 8
    # q = 0.10: k-th threshold 0.01 k -> largest k with p(k) <= 0.01 k is k = 6 (0.06 <= 0.06)
    assert dm.bh(p, 0.10).sum() == 6
    # order invariance and adjusted p monotone
    perm = np.random.default_rng(1).permutation(len(p))
    assert (dm.bh(p[perm], 0.10) == dm.bh(p, 0.10)[perm]).all()
    adj = dm.bh_adjusted(p)
    assert (np.diff(adj[np.argsort(p)]) >= -1e-15).all() and (adj >= p).all()
    # all nulls: uniform p-values rarely produce rejections at q = 0.1
    rng = np.random.default_rng(0)
    n_rej = [dm.bh(rng.random(500), 0.10).sum() for _ in range(50)]
    assert np.mean([r > 0 for r in n_rej]) <= 0.2


def test_tercile_edges_fitted_on_discovery_only():
    disc = _frame(seed=0, seg=dm.DISC)
    ho = _frame(n_dates=50, seed=1, seg=dm.CONF).with_columns(pl.col("log_sigma") + 100.0)  # wildly shifted holdout
    edges = dm.fit_all_edges(disc)
    e_ls = edges["log_sigma:2"]
    assert np.allclose(e_ls, np.quantile(disc["log_sigma"].to_numpy(), [1 / 3, 2 / 3]))
    # mixing holdout rows into the fit is refused
    with pytest.raises(ValueError):
        dm.fit_all_edges(pl.concat([disc, ho]))
    # frozen edges assign the shifted holdout to the top tercile instead of re-centring
    s = next(x for x in dm.states() if x.id == "T3:log_sigma")
    codes = dm.state_codes(ho, s, edges)
    assert (codes == 2).all()
    # binary feature: ties fall low, the middle tercile is empty (structurally absent cell)
    b = dm.assign_bins(np.array([0.0, 1.0, 0.0]), dm.fit_edges(disc["next_is_weekend"].to_numpy(), dm.TERC_P))
    assert b.tolist() == [0, 2, 0]


def test_state_codes_population_and_pairs():
    f = _frame()
    edges = dm.fit_all_edges(f)
    sts = {s.id: s for s in dm.states()}
    btc = dm.state_codes(f, sts["btc_v0cell"], edges)
    assert (btc[(f["symbol"] == "BTCUSDT").to_numpy()] == -1).all() and (btc[(f["symbol"] != "BTCUSDT").to_numpy()] >= 0).all()
    pair = dm.state_codes(f, sts["T3xT3:log_sigmaxlrv_w"], edges)
    assert pair.min() >= 0 and pair.max() <= 8 and len(np.unique(pair)) == 9
    fam = dm.effect_family(f.with_columns(pl.lit(0.0).alias("vr1"), pl.lit(0.0).alias("r1"), pl.lit(0.0).alias("r5"),
                                          pl.lit(dm.DISC).alias("seg5")), edges)
    ids = [e.id for e in fam]
    assert len(ids) == len(set(ids))
    assert "r1|T3:next_is_weekend|c1" not in ids  # empty middle tercile is not an effect
    assert "vr1|Q5:lrv_q|hi-lo" in ids


def test_effect_stat_detects_planted_effect_and_not_noise():
    rng = np.random.default_rng(0)
    n_dates, n_sym = 400, 10
    d = np.repeat(np.arange(n_dates), n_sym)
    codes = rng.integers(0, 3, size=len(d))
    day = np.repeat(rng.normal(size=n_dates), n_sym)  # common date shock
    v = day + rng.normal(size=len(d)) + 0.3 * (codes == 2)
    r = dm.effect_stat(v, codes, d // 10, d, cell=2)
    assert 0.08 < r["d"] < 0.32 and r["t"] > 3 and r["p"] < 1e-2
    c = dm.effect_stat(v, codes, d // 10, d, hi=2, lo=0)
    assert abs(c["d"] - 0.3) < 0.08
    one = dm.effect_stat(v, codes, d // 10, d, cell=2, sided=-1)
    assert one["p"] > 0.99
    null = dm.effect_stat(day + rng.normal(size=len(d)), codes, d // 10, d, cell=1)
    assert abs(null["t"]) < 3.5
    # a pure date effect assigned by date must not look significant with clustered SEs
    cd = (d % 2).astype(int)
    v2 = np.repeat(rng.normal(size=n_dates) * 3, n_sym) + rng.normal(size=len(d)) * 0.1
    r2 = dm.effect_stat(v2, cd, d // 10, d, cell=1)
    naive_t = r2["d"] / (v2.std() / np.sqrt((cd == 1).sum()))
    assert abs(r2["t_date"]) < abs(naive_t)


def test_confirmation_refuses_without_frozen_file(tmp_path):
    missing = tmp_path / "daily_discoveries_frozen.json"
    with pytest.raises(dm.FrozenMissing):
        dm.require_frozen(missing)
    with pytest.raises(dm.FrozenMissing):
        dm.run_confirm(record=False, frozen_path=missing)
    f = _frame(seg=dm.CONF).with_columns(pl.lit(0.0).alias("r1"))
    with pytest.raises(dm.FrozenMissing):
        dm.target_rows(f, "r1", dm.CONF, frozen=missing)
    # a file that is not a frozen H-MINE2 list is refused too
    bad = tmp_path / "bad.json"
    bad.write_text('{"hypothesis": "H-MINE1", "frozen_at": "x", "discoveries": []}', encoding="utf-8")
    with pytest.raises(dm.FrozenMissing):
        dm.require_frozen(bad)
    # discovery rows need no frozen file
    assert dm.target_rows(_frame().with_columns(pl.lit(0.0).alias("r1")), "r1", dm.DISC, frozen=missing).height > 0


def test_verdict_rules():
    def summ(v, r):
        tot = (v[0] + r[0], v[1] + r[1])
        mk = lambda k, n: {"n": n, "replicated": k, "rate": (k / n if n else None)}  # noqa: E731
        return {"overall": mk(tot[1], tot[0]), "by_kind": {"vol": mk(v[1], v[0]), "return": mk(r[1], r[0])}}

    assert dm.verdict(summ((10, 8), (4, 1))) == "supports"
    assert dm.verdict(summ((10, 3), (4, 1))) == "contradicts"
    assert dm.verdict(summ((10, 6), (0, 0))) == "inconclusive"
    assert dm.verdict(summ((4, 2), (4, 3))) == "contradicts"
