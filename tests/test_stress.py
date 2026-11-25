"""Tests for the STRESS harness (src/ai_trader/research/stress.py): trade layer, fills, costs, engine, verdicts."""

from __future__ import annotations

import itertools
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import stress as S

H = S.H_MS


def _path(o, h, lo, c, t0=0, bar_min=60.0):
    n = len(c)
    return S.Path(np.arange(n, dtype=np.int64) * int(bar_min * 60_000) + t0, np.asarray(o, float), np.asarray(h, float),
                  np.asarray(lo, float), np.asarray(c, float), bar_min=bar_min)


def _trades(rows, kind="peg", tier="liquid"):
    base = datetime(2025, 1, 1, tzinfo=UTC)
    out = []
    for i, r in enumerate(rows):
        d = {"unit": "u", "variant": "base", "symbol": r.get("symbol", "BTCUSDT"), "side": r.get("side", 1),
             "entry_ts": base + timedelta(hours=r.get("h", i)), "exit_ts": base + timedelta(hours=r.get("h", i) + r.get("dur", 1)),
             "entry": r.get("entry", 100.0), "stop": r.get("stop", 99.0), "exit": r.get("exit", 101.0), "why": r.get("why", "signal"),
             "gap": False, "kind": kind, "ev": r.get("ev", float("nan")), "cell": "R5", "seg": "llm_in_sample", "weekend": False,
             "holiday": False, "funding_win": False, "tier": r.get("tier", tier)}
        out.append(d)
    return pl.DataFrame(out)


def test_walk_signal_entry_exit_and_stop_modes():
    # strategy = path (1h); bar t close known at open of bar t+1
    c = [100, 100, 101, 102, 103, 104, 100, 100, 100]
    o = [100, 100, 100, 101, 102, 103, 95, 100, 100]     # bar 6 gaps down to 95
    h = [x + 0.5 for x in np.maximum(o, c)]
    lo = [x - 0.5 for x in np.minimum(o, c)]
    p = _path(o, h, lo, c)
    close_ms = p.ts + H
    sig = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1], float)
    sigma = np.full(9, 0.01)
    tr = S.walk_signal(sig, close_ms, np.array(c, float), sigma, p, stop_k=4.0)
    assert len(tr) == 1 and tr[0]["why"] == "stop"
    assert tr[0]["entry"] == 100 and math.isclose(tr[0]["stop"], 100 * math.exp(-0.04))
    assert math.isclose(tr[0]["exit"], tr[0]["stop"]) and tr[0]["gap"]
    g = S.walk_signal(sig, close_ms, np.array(c, float), sigma, p, stop_mode="gap")
    assert g[0]["exit"] == 95.0
    nx = S.walk_signal(sig, close_ms, np.array(c, float), sigma, p, stop_mode="next_open")
    assert nx[0]["exit"] == 100.0          # open of bar 7
    ns = S.walk_signal(sig, close_ms, np.array(c, float), sigma, p, stop_mode="none")
    assert len(ns) == 1 and ns[0]["why"] == "end" and ns[0]["exit"] == 100.0   # never flat: marked to the last close


def test_walk_signal_no_reentry_until_flat_and_signal_exit():
    c = [100, 100, 100, 90, 90, 90, 90, 90, 95, 96]
    o = [100] * 3 + [90] * 7
    h = [x + 0.1 for x in np.maximum(o, c)]
    lo = [x - 0.1 for x in np.minimum(o, c)]
    p = _path(o, h, lo, c)
    sig = np.array([1, 1, 1, 1, 0, 0, 1, 1, -1, -1], float)
    tr = S.walk_signal(sig, p.ts + H, np.array(c, float), np.full(10, 0.01), p)
    # long at bar 0 close, stopped in bar 3; not re-armed while s stays +1 (bar 3); flat at 4 re-arms; long at 6; flip at 8
    assert [t["why"] for t in tr] == ["stop", "signal", "end"]   # the short opened on the flip is marked at the end
    assert tr[1]["entry"] == 90 and tr[1]["exit"] == 95 and tr[1]["side"] == 1


def test_theta_threshold():
    c = [100.0] * 6
    p = _path(c, [101] * 6, [99.9] * 6, c)
    sig = np.array([0.2, 0.25, 0.29, 0.31, 0.0, 0.0])
    tr = S.walk_signal(sig, p.ts + H, np.array(c), np.full(6, 0.01), p, theta=0.3)
    assert len(tr) == 1 and tr[0]["entry_ms"] == int(p.ts[3] + H)


def test_bridge_fill_is_log_linear_and_exact_override():
    p = _path([100, 100], [130, 130], [90, 90], [100, 121])
    assert p.fill(1, 100.0, 0) == 100.0
    assert math.isclose(p.fill(1, 100.0, 30), 110.0, rel_tol=1e-9)    # half-way in log: sqrt(100 * 121)
    assert math.isclose(p.fill(1, 100.0, 60), 121.0)
    p.lat = {30.0: np.array([np.nan, 105.0])}
    assert p.fill(1, 100.0, 30.0) == 105.0


def test_walk_bracket_stop_before_target_and_time_stop():
    o = [100, 100, 100, 100]
    h = [100, 103, 100.5, 100.5]
    lo = [100, 97, 99.5, 99.5]
    c = [100, 100, 100, 100.2]
    p = _path(o, h, lo, c)
    it = [S.Intent(int(p.ts[0] + H), 1, 100.0, 98.0, 102.0)]
    tr = S.walk_bracket(it, p)
    assert tr[0]["why"] == "stop" and tr[0]["exit"] == 98.0           # both hit in bar 1 -> stop first
    it2 = [S.Intent(int(p.ts[1] + H), 1, 100.0, 98.0, 102.0, time_stop_h=2)]
    tr2 = S.walk_bracket(it2, p)
    assert tr2[0]["why"] == "time" and tr2[0]["exit"] == 100.2
    it3 = [S.Intent(int(p.ts[1] + H), 1, 100.0, 98.0, 102.0, time_stop_h=5)]
    assert S.walk_bracket(it3, p) == []                               # time stop beyond the data: unresolved


def test_walk_bracket_one_position_at_a_time():
    c = [100] * 6
    p = _path(c, [100.5, 100.5, 100.5, 103, 100.5, 100.5], [99.5] * 6, c)
    it = [S.Intent(int(p.ts[0] + H), 1, 100.0, 98.0, 102.0), S.Intent(int(p.ts[1] + H), 1, 100.0, 98.0, 102.0)]
    tr = S.walk_bracket(it, p)
    assert len(tr) == 1 and tr[0]["why"] == "target"


def test_r_net_denominator_fixed_under_cost_shocks():
    t = _trades([{"entry": 100.0, "stop": 98.0, "exit": 98.0, "why": "stop"}])
    base = S.r_net(t)[0]
    c_in, c_out = (2.0 + 1.3) / 1e4, (5.5 + 2.0) / 1e4
    den = 2.0 + 100 * (c_in + c_out)
    assert math.isclose(base, (-2.0 - c_in * 100 - c_out * 98) / den)
    assert math.isclose(base, -1.0, abs_tol=0.01)                      # a stop at the stop price is ~ -1R
    hi = S.r_net(t, S.Costs(fee_mult=2.0))[0]
    den2 = 2.0 + 100 * (c_in + c_out)
    assert math.isclose(hi, (-2.0 - (2 * 2.0 + 1.3) / 1e4 * 100 - (2 * 5.5 + 2.0) / 1e4 * 98) / den2)


def test_drought_only_hits_thin_tiers():
    t = pl.concat([_trades([{"why": "stop", "exit": 98.0, "stop": 98.0}], tier="liquid"),
                   _trades([{"why": "stop", "exit": 98.0, "stop": 98.0}], tier="mid")])
    b = S.r_net(t)
    d = S.r_net(t, S.Costs(drought=True))
    assert math.isclose(b[0], d[0]) and d[1] < b[1]


def test_live_engine_limits():
    rows = [{"h": 0, "dur": 10, "side": 1, "symbol": "BTCUSDT"}, {"h": 0, "dur": 10, "side": 1, "symbol": "ETHUSDT"},
            {"h": 0, "dur": 10, "side": 1, "symbol": "SOLUSDT"},   # third long: same-side limit
            {"h": 1, "dur": 10, "side": -1, "symbol": "XRPUSDT"},  # short: allowed (3 open)
            {"h": 2, "dur": 10, "side": -1, "symbol": "ADAUSDT"}]  # 4th open: refused
    t = _trades(rows)
    acc, info = S.live_engine(t, S.r_net(t), S.LiveLimits())
    assert acc.height == 3 and info["refused"] == {"same_side": 1, "max_open": 1}
    assert np.allclose(acc["risk_usd"].to_numpy(), 1.30)


def test_live_engine_daily_budget_and_ev_gate():
    # 2% stops, sequential full losers on one day -> refusals once realised - worst case would breach -10 USD
    rows = [{"h": i, "dur": 0.5, "exit": 98.0, "stop": 98.0, "why": "stop"} for i in range(12)]
    t = _trades(rows)
    acc, info = S.live_engine(t, S.r_net(t), S.LiveLimits())
    assert 6 <= acc.height <= 8 and info["refused"].get("daily_budget", 0) >= 1
    assert acc["usd"].sum() > -10.0 - 1e-9
    t2 = _trades([{"ev": 0.05}, {"h": 3, "ev": 0.2}])
    acc2, info2 = S.live_engine(t2, S.r_net(t2), S.LiveLimits())
    assert acc2.height == 1 and info2["refused"] == {"ev_gate": 1}
    acc3, _ = S.live_engine(t2, S.r_net(t2), S.LiveLimits(ev_gate=False))
    assert acc3.height == 2


def test_live_engine_worst_case_cap_scales_risk():
    t = _trades([{"entry": 100.0, "stop": 99.9, "exit": 100.5}], tier="illiquid")  # 10 bp stop, 60 bp stop slip budget
    acc, _ = S.live_engine(t, S.r_net(t), S.LiveLimits())
    assert acc.height == 1 and acc["risk_usd"][0] < 1.30


def test_run_prob_loss_streak_matches_brute_force():
    p, n, k = 0.6, 9, 5
    brute = 0.0
    for seq in itertools.product([0, 1], repeat=n):
        run = best = 0
        for x in seq:
            run = run + 1 if x else 0
            best = max(best, run)
        brute += (best >= k) * np.prod([p if x else 1 - p for x in seq])
    assert math.isclose(S.run_prob_loss_streak(n, p, k), brute, rel_tol=1e-9)


def test_mean_ci_and_metrics():
    rng = np.random.default_rng(0)
    rows = [{"h": 24 * (i // 3) + i % 3, "exit": 100 + float(x)} for i, x in enumerate(rng.normal(0.5, 1, 300))]
    t = _trades(rows)
    r = S.r_net(t)
    m = S.metrics(t, r)
    assert m["n"] == 300 and m["lo"] < m["mean"] < m["hi"] and m["lo"] > 0
    assert m["max_dd"] <= 0 and m["cvar5_trade"] < m["mean"]


def test_verdict_rules():
    good = {"n": 100, "mean": 0.1, "lo": 0.02, "hi": 0.2, "total_ex_top5": 5.0}
    assert S.verdict({"base": good, "fees15": {"n": 100, "mean": 0.05}})[0] == "SURVIVES"
    assert S.verdict({"base": {**good, "mean": -0.01}})[0] == "FAILS"
    assert S.verdict({"base": good, "fees15": {"n": 100, "mean": -0.01}})[0] == "FAILS"
    v, why = S.verdict({"base": good, "fees15": {"n": 1, "mean": 0.05}, "cliff": True, "crisis_engine_min": -25.0})
    assert v == "FRAGILE" and any("G1" in w for w in why) and any("G3" in w for w in why)
    assert S.verdict({"base": {**good, "lo": -0.01}})[0] == "FRAGILE"
    assert S.verdict({"base": good, "holdout": {"n": 25, "mean": -0.1}})[0] == "FRAGILE"
    assert S.verdict({"base": good, "holdout": {"n": 10, "mean": -0.1}})[0] == "SURVIVES"


def test_perturb_rounding_and_lists():
    p = {"window": 24, "ks": [3, 4, 6], "ratio": 0.6, "low": 30.0, "high": 70.0}
    assert S.perturb(p, "window", 0.5)["window"] == 12 and S.perturb(p, "window", 0.01)["window"] == 1
    assert S.perturb(p, "ks", 1.5)["ks"] == [4, 6, 9]
    assert math.isclose(S.perturb(p, "ratio", 1.2)["ratio"], 0.72)
    b = S.perturb(p, "band", 0.5)
    assert b["low"] == 40.0 and b["high"] == 60.0
    v = S.variants_for("tsmom")
    assert len(v) == 4 * len(S.units()["tsmom"].keys) + S.N_JOINT


def test_co_stop_closes_concurrent_same_side_in_down_state():
    t = _trades([{"h": 0, "dur": 5, "why": "stop", "exit": 99.0}, {"h": 1, "dur": 10, "exit": 105.0, "symbol": "ETHUSDT"},
                 {"h": 1, "dur": 10, "side": -1, "exit": 95.0, "stop": 101.0, "symbol": "SOLUSDT"}])
    stop_ms = int(t["exit_ts"][0].timestamp() * 1000)
    cs = S.co_stop(t, np.array([stop_ms - 1]), np.array([True])).sort("symbol")
    eth = cs.filter(pl.col("symbol") == "ETHUSDT")
    assert eth["why"][0] == "co_stop" and eth["exit"][0] == 99.0
    assert cs.filter(pl.col("symbol") == "SOLUSDT")["why"][0] == "signal"
    calm = S.co_stop(t, np.array([stop_ms - 1]), np.array([False]))
    assert (calm["why"] == "co_stop").sum() == 0


def test_cap_stop_and_net_er():
    stop, tp = S.cap_stop(1, 100.0, 90.0, 120.0, 0.01, 2.0)
    assert math.isclose(math.log(100 / stop), 0.0396, rel_tol=1e-6) and math.isclose(math.log(tp / 100), 2 * 0.0396, rel_tol=1e-6)
    assert S.cap_stop(1, 100.0, 99.999, 101.0, 0.01, None) is None   # inside min 0.5 sigma
    assert S.net_er(100, 99, 101, 0.5, "peg") < 0                  # driftless 1:1 bracket loses the costs


@pytest.mark.parametrize("turnover,tier", [(2e8, "liquid"), (5e7, "mid"), (5e6, "illiquid"), (1e6, "below"), (float("nan"), "unknown")])
def test_tier_of(turnover, tier):
    assert S.tier_of(turnover) == tier
