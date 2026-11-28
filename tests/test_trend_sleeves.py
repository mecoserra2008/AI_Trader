"""Paper trend sleeves (D4: WS16 spec A zoo_ew + spec B tsmom_30d), the H-WS16-15 kill monitor, and the D5 retirement
of the shock sleeve / setup C (agent.trend_sleeves, agent.session, agent.decision, agent.scorecard). Synthetic data only."""

from __future__ import annotations

import inspect
import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.agent import decision as dm
from ai_trader.agent import session as se
from ai_trader.agent import trend_sleeves as ts
from ai_trader.research import stress as S

H = 3_600_000
BASE = datetime(2026, 12, 1, tzinfo=UTC)
B_MS = int(BASE.timestamp() * 1000)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(7)
    n = 24 * 420
    r = rng.standard_t(4, n) * 0.006 + 0.0003 * np.sin(np.arange(n) / 700)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.002, n)))
    return S.synth_frames(o, h, lo, c, datetime(2024, 1, 1, tzinfo=UTC))


def grid(n4: int, extra_h: int = 4):
    """n4 4h closes after BASE and a flat 1h path at 100 covering them (+ extra hours)."""
    close_ms = np.array([B_MS + (t + 1) * 4 * H for t in range(n4)], dtype=np.int64)
    k = 4 * n4 + extra_h
    path = {"ts": np.array([B_MS + i * H for i in range(k)], dtype=np.int64), "o": np.full(k, 100.0), "h": np.full(k, 100.0),
            "l": np.full(k, 100.0), "c": np.full(k, 100.0)}
    return close_ms, np.full(n4, 100.0), np.full(n4, 0.01), path


def fake_sg(sym: str, s_zoo, s_tsm=None, *, sigma4: float = 0.02, start: datetime = BASE, warm: bool = True) -> ts.SymbolSignals:
    s_zoo = np.asarray(s_zoo, dtype=float)
    s_tsm = np.asarray(s_tsm if s_tsm is not None else np.zeros(len(s_zoo)), dtype=float)
    ms = np.array([int(start.timestamp() * 1000) + (t + 1) * 4 * H for t in range(len(s_zoo))], dtype=np.int64)
    return ts.SymbolSignals(sym, start, ms, np.full(len(ms), 3000.0), np.full(len(ms), sigma4), {"zoo_ew": s_zoo, "tsmom_30d": s_tsm}, warm)


# ---------------------------------------------------------------- the signals (spec A / spec B)

def test_signals_are_the_registered_zoo_ensemble_and_tsmom_30d(frames):
    from ai_trader.strategies.trend_zoo import ensemble

    _, f4 = frames
    s = ts.compute_signals(f4)
    assert np.allclose(s["zoo_ew"], ensemble(f4, 6).to_numpy())
    c, sig = f4["close"].to_numpy(), f4["sigma_next"].to_numpy()
    z = (np.log(c[180:]) - np.log(c[:-180])) / (sig[180:] * math.sqrt(180))
    ok = np.isfinite(z)
    assert ok.sum() > 1000
    assert np.allclose(s["tsmom_30d"][180:][ok], np.clip(np.tanh(z[ok]), -1, 1))
    assert np.all(np.abs(s["zoo_ew"]) <= 1) and np.all(np.abs(s["tsmom_30d"]) <= 1)


def test_signals_are_causal(frames):
    _, f4 = frames
    full = ts.signals_from_frame("X", f4, BASE)
    k = f4.height - 37
    part = ts.signals_from_frame("X", f4.head(k), BASE)
    for name in ("zoo_ew", "tsmom_30d"):
        assert np.allclose(part.s[name], full.s[name][:k]), name
    assert full.warm and not ts.signals_from_frame("X", f4.head(100), BASE).warm


# ---------------------------------------------------------------- the trade layer

def test_walk_reproduces_stress_walk_signal(frames):
    f1, f4 = frames
    sig = ts.compute_signals(f4)["zoo_ew"]
    path = S.Path.from_frame(f1, 60.0)
    cms, cl, sg4 = S._ms(f4["close_ts"]), f4["close"].to_numpy().astype(float), f4["sigma_risk_next"].to_numpy().astype(float)
    ref = [x for x in S.walk_signal(sig, cms, cl, sg4, path) if x["why"] != "end"]
    p = {"ts": path.ts, "o": path.o, "h": path.h, "l": path.lo, "c": path.c}
    got = ts.walk(cms, cl, sig, sg4, p, gap_fills=False)["trades"]
    cut = int(cms[-1])
    ref = [x for x in ref if x["exit_ms"] < cut]
    got = [x for x in got if x["exit_ms"] < cut]
    assert len(ref) > 10 and len(got) == len(ref)
    for a, g in zip(ref, got, strict=True):
        for k in ("side", "entry_ms", "exit_ms", "why"):
            assert a[k] == g[k]
        for k in ("entry", "stop", "exit"):
            assert a[k] == pytest.approx(g[k])


def test_entry_stop_exit_and_no_reentry_after_a_stop():
    close_ms, close, sigma, path = grid(12)
    path["l"][25] = 95.0                       # inside (close 5, close 6]: hits the long stop 100 exp(-0.04)
    s = [0.1, 0.29, 0.31, 0.5, 0.0, 0.4, 0.4, 0.6, 0.2, -0.1, 0.35, -0.35]
    w = ts.walk(close_ms, close, s, sigma, path)
    tr = w["trades"]
    assert [(x["side"], x["why"]) for x in tr] == [(1, "signal"), (1, "stop"), (1, "signal")]
    # entry only at |s| >= 0.30 (0.29 is not enough), at the close price
    assert tr[0]["entry_ms"] == close_ms[2] and tr[0]["entry"] == 100.0
    # stop = entry x exp(-side x 4 x sigma), filled at the stop price (no gap)
    assert tr[1]["stop"] == pytest.approx(100 * math.exp(-0.04)) and tr[1]["exit"] == pytest.approx(100 * math.exp(-0.04))
    assert tr[1]["exit_ms"] == path["ts"][25] + H
    # exit at the first close with s x side <= 0 (s = 0 counts), at the close
    assert tr[0]["exit_ms"] == close_ms[4] and tr[0]["exit"] == 100.0
    # after the stop: s 0.4 / 0.6 / 0.2 do not re-enter; s -0.1 re-arms; 0.35 enters again
    assert tr[2]["entry_ms"] == close_ms[10]
    # a flip at the close exits and reverses at the same close
    assert tr[2]["exit_ms"] == close_ms[11] and w["open"]["side"] == -1 and w["open"]["entry_ms"] == close_ms[11]


def test_a_flip_after_a_stop_re_arms_at_once_and_gaps_fill_at_the_open():
    close_ms, close, sigma, path = grid(6)
    path["o"][9] = path["l"][9] = 94.0         # gap below the long stop inside (close 1, close 2]
    s = [0.5, 0.5, 0.6, -0.5, -0.5, -0.5]
    tr = ts.walk(close_ms, close, s, sigma, path)["trades"]
    assert tr[0]["why"] == "stop" and tr[0]["gap"] and tr[0]["exit"] == 94.0
    w = ts.walk(close_ms, close, s, sigma, path)
    assert w["open"]["side"] == -1 and w["open"]["entry_ms"] == close_ms[3], "s -0.5 x long <= 0 re-arms and enters short"


def test_unfilled_peg_stays_flat_and_decides_again_at_the_next_close():
    close_ms, close, sigma, path = grid(6)
    start = int(close_ms[0])
    calls = []

    def peg(ms, side, px):
        calls.append(ms)
        return False if ms == close_ms[1] else (True if ms == close_ms[2] else None)

    w = ts.walk(close_ms, close, [0.0, 0.5, 0.5, 0.5, 0.5, 0.5], sigma, path, start_ms=start, peg=peg)
    assert w["open"]["entry_ms"] == close_ms[2] and w["open"]["peg"] == "filled" and w["open"]["forward"]
    assert [a["result"] for a in w["attempts"]] == ["unfilled", "filled"]
    calls.clear()
    h = ts.walk(close_ms, close, [0.5] * 6, sigma, path, start_ms=start, peg=peg)
    assert h["open"]["entry_ms"] == close_ms[0] and h["open"]["peg"] == "history" and not h["open"]["forward"]
    assert calls == [] and h["attempts"] == [], "closes at or before the forward start are history (always filled, not checked)"


def test_net_r_matches_stress_r_net():
    t = pl.DataFrame({"kind": ["peg", "peg"], "why": ["signal", "stop"], "tier": ["liquid", "liquid"], "entry": [100.0, 100.0],
                      "stop": [96.0, 104.0], "exit": [110.0, 104.0], "side": [1, -1]})
    ref = S.r_net(t, S.Costs.live())
    got = [ts.net_r(1, 100.0, 96.0, 110.0), ts.net_r(-1, 100.0, 104.0, 104.0)]
    assert got == pytest.approx(list(ref))


def test_stream_counts_only_forward_entries(frames):
    f1, f4 = frames
    as_of = f4["close_ts"][-1]
    start = f4["close_ts"][f4.height - 900]
    sg = ts.signals_from_frame("X", f4, as_of)
    p = {"ts": f1["event_ts"].dt.epoch("ms").to_numpy(), "o": f1["open"].to_numpy(), "h": f1["high"].to_numpy(), "l": f1["low"].to_numpy(), "c": f1["close"].to_numpy()}
    st = ts.stream(as_of, symbols=["X"], signals_fn=lambda s, t: sg, path_fn=lambda s, t: p, start=start, walk_start=None)
    full = ts.walk(sg.close_ms, sg.close, sg.s["zoo_ew"], sg.sigma4, p, start_ms=int(start.timestamp() * 1000))
    fwd = [x for x in full["trades"] if x["forward"]]
    v = st["trend_zoo_ew"]
    assert len(v["closed"]) == len(fwd) > 3
    assert all(x["entry_ms"] > int(start.timestamp() * 1000) for x in v["closed"])
    assert [x["exit_ms"] for x in v["closed"]] == sorted(x["exit_ms"] for x in v["closed"])
    assert all(np.isfinite(x["R"]) and x["hold_days"] >= 0 for x in v["closed"])
    assert set(st) == set(ts.SLEEVES)


# ---------------------------------------------------------------- paper-book rules: re-entry and rule exits

def test_reentry_block_after_a_stop_until_flat_or_flip():
    sg = fake_sg("ETHUSDT", [0.6, 0.5, 0.4, -0.1, 0.5])
    stop_ts = int(sg.close_ms[0]) + H
    tr = [{"strategy": "trend_zoo_ew", "symbol": "ETHUSDT", "side": "long", "exit_reason": "stop", "closed_ts": stop_ts}]
    assert ts.reentry_block("trend_zoo_ew", "ETHUSDT", "long", tr, fake_sg("ETHUSDT", [0.6, 0.5, 0.4])) is not None
    assert ts.reentry_block("trend_zoo_ew", "ETHUSDT", "long", tr, sg) is None, "s -0.1 after the stop re-arms"
    assert ts.reentry_block("trend_zoo_ew", "ETHUSDT", "short", tr, fake_sg("ETHUSDT", [0.6, 0.5, -0.4])) is None, "a flip is allowed"
    assert ts.reentry_block("trend_tsmom30", "ETHUSDT", "long", tr, fake_sg("ETHUSDT", [0.6, 0.5, 0.4])) is None, "per sleeve"
    tr2 = [{**tr[0], "exit_reason": "exit_signal"}]
    assert ts.reentry_block("trend_zoo_ew", "ETHUSDT", "long", tr2, fake_sg("ETHUSDT", [0.6, 0.5, 0.4])) is None


def test_exits_due_first_close_with_s_times_side_le_zero():
    sg = fake_sg("ETHUSDT", [0.6, 0.5, -0.1, 0.4, 0.5])
    opened = int(sg.close_ms[0]) + H
    pos = [{"id": "p1", "symbol": "ETHUSDT", "strategy": "trend_zoo_ew", "family": "trend", "side": "long", "opened_ts": opened, "mark": 3001.0},
           {"id": "p2", "symbol": "ETHUSDT", "strategy": "trend_4h", "family": "trend", "side": "long", "opened_ts": opened},
           {"id": "p3", "symbol": "ETHUSDT", "strategy": "trend_tsmom30", "family": "trend", "side": "long", "opened_ts": opened, "exit_requested": True}]
    ex = ts.exits_due(pos, BASE, lambda s, t: sg)
    assert [x["position_id"] for x in ex] == ["p1"], "legacy brackets and already-requested exits are skipped"
    assert ex[0]["s"] == pytest.approx(-0.1) and ex[0]["late"] and ex[0]["close_ts"].startswith((BASE + timedelta(hours=12)).isoformat()[:16])
    short = [{**pos[0], "side": "short"}]
    assert ts.exits_due(short, BASE, lambda s, t: fake_sg("ETHUSDT", [-0.6, -0.5, -0.4]))  == []
    assert not ts.exits_due(short, BASE, lambda s, t: fake_sg("ETHUSDT", [-0.6, -0.5, 0.0]))[0]["late"]


# ---------------------------------------------------------------- the kill monitor (H-WS16-15)

def trades(rs, per_day: int = 3, hold_days: float = 1.0):
    out = []
    for i, r in enumerate(rs):
        e = B_MS + (i // per_day) * ts.DAY_MS + (i % per_day) * H
        out.append({"R": float(r), "entry_ms": e, "exit_ms": e + int(hold_days * ts.DAY_MS), "side": 1 if i % 2 else -1, "hold_days": hold_days})
    return out


def test_kill_rules_mean_at_100_and_upper_bound_at_300():
    assert not ts.kill_rules(trades([-1.0] * 99))["killed"], "nothing before 100 closed trades"
    k = ts.kill_rules(trades([-0.35] * 100))
    assert k["killed"] and k["at_n"] == 100 and "first 100" in k["rule"]
    assert not ts.kill_rules(trades([-0.25] * 100))["killed"]
    rng = np.random.default_rng(3)
    bad = ts.kill_rules(trades(-0.2 + rng.normal(0, 0.3, 300)))
    assert bad["killed"] and bad["at_n"] == 300 and bad["ub95"] < 0
    good = ts.kill_rules(trades(0.3 + rng.normal(0, 3.0, 320)))
    assert not good["killed"] and good["ub95"] > 0 and [c["n"] for c in good["checks"] if c["rule"] == "ub95"] == [300, 320]
    # a bad first 100 kills even if later trades recover
    rec = ts.kill_rules(trades([-0.5] * 100 + [2.0] * 250))
    assert rec["killed"] and rec["at_n"] == 100


def test_upper_bound_uses_a_holding_matched_block():
    u = ts.upper_bound(trades([0.1] * 50, hold_days=20.0))
    assert u["block_days"] == pytest.approx(20.0)
    assert ts.upper_bound(trades([0.1] * 50, hold_days=1.0))["block_days"] == ts.MIN_BLOCK_DAYS


def test_peg_alert_needs_enough_checked_entries():
    a = [{"result": "filled"}] * 15 + [{"result": "unfilled"}] * 10 + [{"result": "unchecked"}] * 7
    st = ts.peg_stats(a)
    assert st["rate"] == pytest.approx(0.6) and st["alert"] and st["unchecked"] == 7
    assert not ts.peg_stats(a[:5] + a[15:20])["alert"], "10 checked: too few for the alert"
    assert not ts.peg_stats([{"result": "filled"}] * 30)["alert"]


def fake_stream(closed_a, closed_b=()):
    def f(t, peg):
        return {"trend_zoo_ew": {"closed": list(closed_a), "open": [], "attempts": [{"result": "filled"}], "errors": {}},
                "trend_tsmom30": {"closed": list(closed_b), "open": [], "attempts": [], "errors": {}}}
    return f


def test_monitor_latches_kills_point_in_time(tmp_path):
    t0, t1, t2 = BASE, BASE + timedelta(days=30), BASE + timedelta(days=60)
    m0 = ts.monitor(t0, root=tmp_path, write=True, stream_fn=fake_stream(trades([0.1] * 50)))
    assert not m0["sleeves"]["trend_zoo_ew"]["killed"] and ts.killed(m0, "trend_zoo_ew") is None
    m1 = ts.monitor(t1, root=tmp_path, write=True, stream_fn=fake_stream(trades([-0.4] * 100)))
    assert m1["sleeves"]["trend_zoo_ew"]["killed"] and not m1["sleeves"]["trend_tsmom30"]["killed"]
    doc = json.loads((tmp_path / "monitor.json").read_text(encoding="utf-8"))
    assert doc["kills"]["trend_zoo_ew"]["at"] == t1.isoformat()
    # latched: a later stream that no longer fires the rule keeps the kill
    m2 = ts.monitor(t2, root=tmp_path, write=False, stream_fn=fake_stream(trades([0.1] * 50)))
    assert ts.killed(m2, "trend_zoo_ew") and "first 100" in ts.killed(m2, "trend_zoo_ew")
    # point in time: before the latch time the kill is unknown
    assert ts.killed(ts.monitor(t0, root=tmp_path, write=False, stream_fn=fake_stream(trades([0.1] * 50))), "trend_zoo_ew") is None
    # a broken stream still applies the latch
    def broken(t, peg):
        raise RuntimeError("lake down")
    m3 = ts.monitor(t2, root=tmp_path, stream_fn=broken)
    assert m3["error"] and ts.killed(m3, "trend_zoo_ew")
    lines = ts.describe(m2)
    assert any("KILLED" in x for x in lines) and any("spec B trend_tsmom30" in x for x in lines)


class FakeLake:
    def __init__(self, n: int, low: float, high: float):
        self.n, self.low, self.high, self.calls = n, low, high, 0

    def read_asof(self, table, as_of, **kw):
        self.calls += 1
        assert table == "flow_3s" and kw["filters"]["date"] == "2026-12-01"
        t0 = kw["start"]
        return pl.DataFrame({"event_ts": [t0 + timedelta(seconds=3 * i) for i in range(self.n)], "high": [self.high] * self.n, "low": [self.low] * self.n})


def test_peg_checker_on_flow_3s_is_point_in_time_and_cached(tmp_path):
    ms = B_MS + 8 * H                         # the 08:00 close
    t_end = BASE + timedelta(hours=8, minutes=15)
    f = tmp_path / "peg.json"
    assert ts.PegChecker(t_end - timedelta(minutes=1), f, lake=FakeLake(300, 99.0, 101.0)).check("ETHUSDT", ms, 1, 100.0) is None, "window not over"
    pc = ts.PegChecker(t_end + timedelta(minutes=5), f, lake=FakeLake(300, 99.0, 101.0), write=True)
    assert pc.check("ETHUSDT", ms, 1, 100.0) is True and pc.check("ETHUSDT", ms, -1, 100.0) is True
    assert pc.check("ETHUSDT", ms, 1, 98.0) is False, "no trade 1 bp below a buy at 98"
    pc.save()
    lk = FakeLake(300, 0.0, 0.0)
    again = ts.PegChecker(t_end + timedelta(hours=1), f, lake=lk)
    assert again.check("ETHUSDT", ms, 1, 98.0) is False and lk.calls == 0, "cached"
    early = ts.PegChecker(t_end + timedelta(minutes=1), f, lake=FakeLake(300, 99.0, 101.0))
    assert early.check("ETHUSDT", ms, 1, 98.0) is None, "a cached result is not used before it was known"
    thin = ts.PegChecker(t_end + timedelta(minutes=5), None, lake=FakeLake(100, 99.0, 101.0))
    assert thin.check("ETHUSDT", ms, 1, 100.0) is None, "coverage below 80%: unchecked for now"
    late = ts.PegChecker(t_end + timedelta(hours=2), None, lake=FakeLake(100, 99.0, 101.0))
    assert late.check("ETHUSDT", ms, 1, 100.0) is None and next(iter(late.cache.values()))["result"] == "no_data"


# ---------------------------------------------------------------- candidates (agent.session) and the D5 retirement

def test_candidates_have_no_target_stop_at_the_cap_and_the_weekend_block():
    sg = fake_sg("ETHUSDT", [0.1, 0.5], [0.1, 0.2], sigma4=0.02)
    cs = se.trend_candidates("ETHUSDT", 3000.0, sg, weekend=False, cap=4.0, min_sig=0.5, cell="R2")
    assert [c.sleeve for c in cs] == ["trend_zoo_ew"], "spec B |s| 0.2 < 0.30"
    c = cs[0]
    assert c.take_profit is None and c.p_quant is None and c.rr is None and c.exit_rule == "signal_flat" and c.side == "long"
    assert c.stop == pytest.approx(3000 * math.exp(-3.96 * 0.02)) and c.stop_sigma == pytest.approx(3.96) and c.freq == "4h"
    assert se.net_er(c) == pytest.approx(ts.PRIOR_NET_R) and c.id == "zoo-ETHU"
    sh = se.trend_candidates("ETHUSDT", 3000.0, fake_sg("ETHUSDT", [-0.4], [-0.31]), weekend=True, cap=3.0, min_sig=0.5, cell="R8")
    assert [x.sleeve for x in sh] == ["trend_zoo_ew", "trend_tsmom30"] and all(x.side == "short" for x in sh)
    assert all("TradFi weekend" in x.blocked for x in sh)
    assert sh[0].stop_sigma == pytest.approx(2.97) and any("stop_capped" in f for f in sh[0].flags)
    assert se.trend_candidates("ETHUSDT", 3000.0, fake_sg("ETHUSDT", [0.9], warm=False), weekend=False, cap=4.0, min_sig=0.5, cell="R2") == []


def test_shock_sleeve_and_setup_c_are_retired():
    assert "shock_regime" not in dm.SLEEVE_FAMILY and "shock_regime" not in dm.GATE_STRATEGY
    assert set(ts.SLEEVES) == {"trend_zoo_ew", "trend_tsmom30"}
    assert {"shock_regime", "shock_desk", "setup_c"} <= set(ts.RETIRED) and "trend_4h" in ts.LEGACY
    assert "shock_regime" not in se.PRIOR_EDGE
    src = inspect.getsource(se.analyse) + inspect.getsource(se.trend_candidates)
    assert "shock_regime" not in src and "ema_stack" not in src
    card = dm.compact_card({"symbol": "ETHUSDT", "price": 3000.0, "cell": "R2", "zoo_ew": 0.5, "tsmom_30d": 0.4, "sigma_risk_4h": 0.02,
                            "sigma_risk_h": 0.01, "har_day": 0.03, "signal_close": "2026-12-01T08:00:00+00:00"}, None)
    assert not any("shock" in x for x in card) and any("zoo_ew (spec A) +0.50" in x for x in card)


def test_own_exit_decisions_and_intents():
    meta = {"session_id": "S20261201-0805", "as_of": "2026-12-01T08:05:00Z", "model_id": "claude-opus-5-5", "prompt_version": "v0.1",
            "pipeline_test": False}
    c = se.trend_candidates("ETHUSDT", 3000.0, fake_sg("ETHUSDT", [0.5]), weekend=False, cap=4.0, min_sig=0.5, cell="R2")[0]
    from dataclasses import asdict

    row = {**asdict(c), "entry_type": "pegged_limit", "net_er": se.net_er(c), "blocked": None, "fired": [], "signal": 0.5, "asset_class": "crypto",
           "gate": {"decision": "pass", "rules_applied": [], "reasons": [], "blocked_by": []}}
    errs, norm = dm.validate_decisions({"decisions": [{"candidate_id": c.id, "action": "take", "reason": "trend agrees",
                                                       "tilt": {"view_logodds": 0.5, "confidence": "high"}}]}, meta, [row], {"positions": [], "pending_orders": []}, {})
    assert errs == [] and norm["decisions"][0]["p_final"] is None and norm["decisions"][0]["p_used"] is None
    it = dm.build_intents(meta, [row], norm)[0]
    assert it["targets"] == [] and it["net_er_r"] == pytest.approx(ts.PRIOR_NET_R) and it["p_tp"] == ts.REF_P_TP
    assert it["strategy"] == "trend_zoo_ew" and it["family"] == "trend" and it["entry"] == {"type": "limit", "price": 3000.0, "peg": "touch"}
    assert it["context"]["exit_rule"] == "signal_flat" and it["context"]["p_null"] is None and it["sigma"]["freq"] == "4h"
    ex = dm.rule_exit_intents("S1", BASE, [{"position_id": "p1", "symbol": "ETHUSDT", "strategy": "trend_zoo_ew", "family": "trend", "side": "long",
                                             "close_ts": BASE.isoformat(), "s": -0.1, "late": False, "mark": 3000.0}], False, "m")[0]
    assert ex["side"] == "flat" and ex["stop"] is None and ex["context"]["decision"] == "rule_exit" and "R-EXIT-1" in ex["gate"]["rules_applied"]


def test_scorecard_resolves_own_exit_candidates_on_stop_or_signal():
    from ai_trader.agent.scorecard import resolve_signal_exit

    t0 = BASE + timedelta(minutes=5)
    ev = [BASE + timedelta(hours=k) for k in range(12)]
    bars = pl.DataFrame({"event_ts": ev, "close_ts": [x + timedelta(hours=1) for x in ev], "high": [101.0] * 12, "low": [99.0] * 12, "close": [100.5] * 12})
    c = {"side": "long", "entry": 100.0, "stop": 96.0, "entry_type": "pegged_limit"}
    ms = np.array([B_MS + 4 * H, B_MS + 8 * H], dtype=np.int64)
    r = resolve_signal_exit(c, t0, bars, ms, np.array([0.4, -0.2]))
    assert r["status"] == "signal" and r["closed"].startswith(str(BASE + timedelta(hours=8))[:16]) and r["R"] == pytest.approx(0.5 / 4 - 100 * 10.8e-4 / 4)
    low = bars.with_columns(pl.when(pl.col("event_ts") == BASE + timedelta(hours=2)).then(95.0).otherwise(pl.col("low")).alias("low"))
    assert resolve_signal_exit(c, t0, low, ms, np.array([0.4, -0.2]))["status"] == "stop"
    assert resolve_signal_exit(c, t0, bars, ms, np.array([0.4, 0.3]))["status"] == "open"


# ---------------------------------------------------------------- end to end on the Rust paper book

from tests.test_decision_session import SYM, T, env, needs_atx, pass_gate, take_all  # noqa: E402,F401


def sleeve_world(s_by_close):
    """analyse / signals for SYM: s_by_close(close_dt) -> zoo_ew value; closes every 4h for 40 days before each as_of."""
    def signals(sym, t):
        last = t.replace(minute=0, second=0, microsecond=0)
        last = last - timedelta(hours=last.hour % 4)
        closes = [last - timedelta(hours=4 * k) for k in range(240)][::-1]
        z = np.array([s_by_close(x) for x in closes])
        return ts.SymbolSignals(sym, t, np.array([int(x.timestamp() * 1000) for x in closes], dtype=np.int64), np.full(len(closes), 3000.0),
                                np.full(len(closes), 0.02), {"zoo_ew": z, "tsmom_30d": np.zeros(len(closes))}, True)

    def analyse(sym, t):
        sg = signals(sym, t)
        cands = se.trend_candidates(sym, 3000.0, sg, weekend=False, cap=4.0, min_sig=0.5, cell="R2")
        st = {"symbol": sym, "price": 3000.0, "cell": "R2", "label": "test", "weekend": False, "har_day": 0.03, "sigma_risk_h": 0.01, "sigma_rb_h": 0.01,
              "wk_m": 1.0, "sigma_risk_4h": 0.02, "zoo_ew": sg.last("zoo_ew"), "tsmom_30d": 0.0, "signal_close": sg.last_close().isoformat(),
              "bar_ts": str(t - timedelta(hours=1)), "data_max_ts": str(t - timedelta(minutes=4)), "data_max_ts_4h": None}
        return f"{sym} card", cands, st
    return analyse, signals


def tprep(env, as_of, analyse, signals, monitor=None):
    p, bars = env
    book = dm.Book(p, dm.BOOK_RUN, bars_dir=bars, journal=False)
    mon = monitor or (lambda t: ts.monitor(t, root=p.trend, write=False, stream_fn=fake_stream([])))
    out = dm.prepare(as_of, paths=p, symbols=[SYM], analyse_fn=analyse, gate_fn=pass_gate, book=book, wall=as_of + timedelta(minutes=1),
                     monitor_fn=mon, signals_fn=signals)
    return out, book


@needs_atx
def test_sleeve_entry_rule_exit_and_packet_end_to_end(env):
    p, _ = env
    flip = T.replace(minute=0) + timedelta(hours=4)                        # s turns <= 0 at the 12:00 close
    analyse, signals = sleeve_world(lambda x: 0.6 if x < flip else -0.1)
    out, book = tprep(env, T, analyse, signals)
    rows = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == ["zoo-ETHU"] and rows[0]["blocked"] is None and rows[0]["take_profit"] is None
    pk = (out / "PACKET.md").read_text(encoding="utf-8")
    assert "- (signal exit)" in pk and "H-WS16-15" in pk
    assert all("etired" in ln for ln in pk.splitlines() if "shock" in ln.lower()), "shock appears only as retired"
    assert len(pk.splitlines()) <= 120 and "compact-3" in pk
    sub = dm.decide(out.name, take_all(out.name, ["zoo-ETHU"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    assert sub["accepted"] == ["zoo-ETHU"], sub["rejected"]
    it = json.loads((out / "intents.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert it["targets"] == [] and it["net_er_r"] == pytest.approx(ts.PRIOR_NET_R)
    # next session: the 12:00 close has s x side <= 0 -> the harness sends the rule exit; no new entry (flat signal)
    out2, book2 = tprep(env, T + timedelta(hours=4), analyse, signals)
    meta = json.loads((out2 / "session.json").read_text(encoding="utf-8"))
    assert [x["decision"] for x in meta["rule_exits"]["sent"]] == ["accept"]
    pk2 = (out2 / "PACKET.md").read_text(encoding="utf-8")
    assert "rule exits this pass (deterministic, not yours): ETHUSDT trend_zoo_ew" in pk2 and "rule exit sent" in pk2
    book2.advance(T + timedelta(hours=6))
    tr = book2.trades()
    assert len(tr) == 1 and tr[0]["exit_reason"] == "exit_signal" and tr[0]["strategy"] == "trend_zoo_ew"


@needs_atx
def test_killed_sleeve_family_cap_and_reentry_block_candidates(env):
    p, _ = env
    analyse, signals = sleeve_world(lambda x: 0.6)
    kill = {"trend_zoo_ew": {"at": (T - timedelta(days=1)).isoformat(), "rule": "mean -0.40 R < -0.3 over the first 100 closed trades", "n": 100}}
    p.trend.mkdir(parents=True, exist_ok=True)
    (p.trend / "monitor.json").write_text(json.dumps({"kills": kill}), encoding="utf-8")
    out, book = tprep(env, T, analyse, signals)
    r = json.loads((out / "candidates.json").read_text(encoding="utf-8"))[0]
    assert "KILLED" in r["blocked"]
    assert "KILLED" in (out / "PACKET.md").read_text(encoding="utf-8")
    (p.trend / "monitor.json").unlink()
    # a stop of this sleeve in this symbol blocks a same-side re-entry until s x side <= 0
    stop_ms = int((T - timedelta(hours=2)).timestamp() * 1000)
    trades_f = book.dir / "trades.jsonl"
    trades_f.parent.mkdir(parents=True, exist_ok=True)
    out2, _ = tprep(env, T + timedelta(hours=4), analyse, signals)
    assert json.loads((out2 / "candidates.json").read_text(encoding="utf-8"))[0]["blocked"] is None
    with trades_f.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x", "symbol": SYM, "strategy": "trend_zoo_ew", "family": "trend", "side": "long", "qty": 1, "entry_price": 3000,
                            "exit_price": 2770, "opened_ts": stop_ms - 10 * H, "closed_ts": stop_ms, "exit_reason": "stop", "gross_pnl": -1, "fees": 0,
                            "pnl": -1, "risk_usd": 1, "r_multiple": -1.0}) + "\n")
    out3, _ = tprep(env, T + timedelta(hours=8), analyse, signals)
    assert "no re-entry after the long stop" in json.loads((out3 / "candidates.json").read_text(encoding="utf-8"))[0]["blocked"]


def test_packet_gate_line_is_point_in_time_and_the_edge_line_drops_retired_sleeves(monkeypatch):
    pytest.importorskip("ai_trader.live.gates")
    import ai_trader.live.gates as lg

    t = BASE
    monkeypatch.setattr(lg, "load", lambda path=None: {"written_ms": B_MS + 60_000})
    assert "no gate state known at as_of" in dm.gates_line(t)
    seen = {}

    def fake_line(symbols=None, *, gates=None, now=None, path=None):
        seen["now"] = now
        return "Live gates: test line"

    monkeypatch.setattr(lg, "packet_line", fake_line)
    monkeypatch.setattr(lg, "load", lambda path=None: {"written_ms": B_MS - 60_000})
    assert dm.gates_line(t) == "Live gates: test line" and seen["now"] == B_MS
    doc = {"sleeves": {"shock_regime": {"x": 1}, "trend_4h": {"x": 2}}}
    assert set(dm._legacy_edges(doc)["sleeves"]) == {"trend_4h"}
    assert dm._finite({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}
