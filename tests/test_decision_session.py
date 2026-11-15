"""Phase-4 decision harness (agent.decision, scripts/decision_session.py) and the H-STOP1 stop posterior.

Synthetic data only: candidates come from an injected analyse function, the gate from an injected function, and the
paper book is the real Rust engine (`atx exec-intents --mode paper --as-of`) on CSV bars in a temp dir, so the
persistence, kill-switch and no-lookahead checks exercise the actual fill logic.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.agent import decision as dm
from ai_trader.agent.session import Candidate, p_first
from ai_trader.core.time import LookAheadError
from ai_trader.risk import stop_posterior as sp

T = datetime(2026, 11, 10, 8, 5, tzinfo=UTC)        # a Tuesday, slot 08:00
SYM = "ETHUSDT"
SIG = 0.01

needs_atx = pytest.mark.skipif(dm.atx_exe() is None, reason="target/debug/atx with exec-intents --as-of not built")


# ---------------------------------------------------------------- fixtures

def write_bars(d: Path, path_fn, start: datetime, n: int) -> None:
    """1h CSV bars for the Rust book: open/high/low/close from path_fn(k) -> (o, h, l, c)."""
    d.mkdir(parents=True, exist_ok=True)
    rows = ["ts_ms,open,high,low,close"]
    for k in range(n):
        o, h, lo, c = path_fn(k)
        rows.append(f"{int((start + timedelta(hours=k)).timestamp() * 1000)},{o},{h},{lo},{c}")
    (d / f"{SYM}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (d / "instruments.json").write_text(json.dumps([{"symbol": SYM, "tick_size": 0.01, "qty_step": 0.01, "min_qty": 0.01, "max_qty": 1e5, "max_mkt_qty": 1e4,
                                                     "min_notional": 5.0, "max_leverage": 50.0}]), encoding="utf-8")


def flat_path(px: float = 3000.0):
    return lambda k: (px, px * 1.001, px * 0.999, px)


@pytest.fixture
def env(tmp_path):
    p = dm.Paths(tmp_path / "sessions", tmp_path / "book", tmp_path / "KILL", tmp_path / "live_nn", tmp_path / "stop_cap.json", tmp_path / "journal")
    bars = tmp_path / "bars"
    write_bars(bars, flat_path(), T.replace(minute=0) - timedelta(hours=48), 120)      # bars well past every as_of (future bars must be ignored)
    return p, bars


def fake_analyse(n_cands: int = 1, px: float = 3000.0, future_data: bool = False):
    def f(sym, t):
        cands = []
        for k in range(n_cands):
            stop = px * math.exp(-3 * SIG)
            tp = px * math.exp(6 * SIG)
            c = Candidate(f"shk-{k}", SYM, "shock_regime", "long", "market", px, stop, tp, p_first(px, stop, tp), 0.45, 2.0, None, "test shock",
                          "majors", str(t - timedelta(hours=1)), sigma_per_bar=SIG, freq="1h", stop_sigma=3.0)
            cands.append(c)
        known = t + timedelta(minutes=5) if future_data else t - timedelta(minutes=4)
        st = {"symbol": SYM, "price": px, "cell": "R4", "label": "test", "weekend": False, "har_day": 0.03, "sigma_risk_h": SIG, "sigma_rb_h": SIG, "wk_m": 1.0,
              "trend_4h": 0.0, "shock_signal": 1.0, "shock": "test", "bar_ts": str(t - timedelta(hours=1)), "data_max_ts": str(known), "data_max_ts_4h": None}
        return f"{SYM} card", cands, st
    return f


def pass_gate(cands, states, as_of):
    g = {c.id: {"decision": "pass", "rules_applied": ["R-PX-2"], "reasons": [], "blocked_by": [], "warnings": [], "asset_class": "crypto", "signal": 1.0,
                "gate_strategy": c.sleeve, "rules_hash": "test"} for c in cands}
    return g, {}


def prep(env, as_of=T, *, n=1, test=False, analyse=None, wall=None):
    p, bars = env
    book = dm.Book(p, dm.TEST_RUN if test else dm.BOOK_RUN, bars_dir=bars, journal=False)
    out = dm.prepare(as_of, paths=p, symbols=[SYM], pipeline_test=test, analyse_fn=analyse or fake_analyse(n), gate_fn=pass_gate, book=book,
                     wall=wall or as_of + timedelta(minutes=1))
    return out, book


def take_all(sid: str, ids: list[str], **extra) -> dict:
    return {"session_id": sid, "model_id": "claude-opus-5-5", "decisions": [{"candidate_id": i, "action": "take", "reason": "test take"} for i in ids], **extra}


# ---------------------------------------------------------------- stop posterior (H-STOP1)

def test_prior_predictive_q90_is_four_sigma_with_20_pseudo_trades():
    p = sp.prior()
    assert p.kappa == 20 and p.a == 10 and math.isclose(p.b, 10 * 0.5**2)
    assert math.isclose(p.quantile(0.9), 4.0, rel_tol=1e-12)
    # other prior log-sds keep the same pinned quantile
    assert math.isclose(sp.prior(0.3).quantile(0.9), 4.0, rel_tol=1e-12)


def test_conjugate_update_matches_the_closed_form():
    p = sp.prior()
    x = np.log(np.array([1.2, 2.0, 0.8, 3.1, 1.6]))
    q = sp.update(p, x)
    n, xb = len(x), x.mean()
    assert math.isclose(q.kappa, p.kappa + n) and math.isclose(q.a, p.a + n / 2)
    assert math.isclose(q.m, (p.kappa * p.m + n * xb) / (p.kappa + n))
    assert math.isclose(q.b, p.b + 0.5 * ((x - xb) ** 2).sum() + p.kappa * n * (xb - p.m) ** 2 / (2 * (p.kappa + n)))
    # sequential == batch (conjugacy)
    s = sp.update(sp.update(p, x[:2]), x[2:])
    assert all(math.isclose(a, b) for a, b in zip((s.m, s.kappa, s.a, s.b), (q.m, q.kappa, q.a, q.b), strict=True))
    assert sp.update(p, []) == p
    # winners with small MAE pull the cap down, large MAE push it up
    assert sp.update(p, np.log(np.full(40, 1.0))).quantile() < 4.0 < sp.update(p, np.log(np.full(40, 5.0))).quantile()


def test_applied_cap_is_clipped_to_the_hard_bounds(tmp_path):
    lo, hi = sp.bounds()
    assert (lo, hi) == (2.5, 6.0)
    tiny = [{"id": f"t{i}", "symbol": SYM, "closed": "x", "mae_sigma": 0.2, "stop_sigma": 3.0} for i in range(200)]
    big = [{"id": f"b{i}", "symbol": SYM, "closed": "x", "mae_sigma": 12.0, "stop_sigma": 13.0} for i in range(200)]
    d1 = sp.compute(tiny, sp.week_start(T))
    d2 = sp.compute(big, sp.week_start(T))
    assert d1["q90_predictive"] < lo and d1["applied_cap_sigma"] == lo and d1["clipped"]
    assert d2["q90_predictive"] > hi and d2["applied_cap_sigma"] == hi and d2["clipped"]
    # a file outside the bounds is refused (fail closed), like crates/at-risk
    f = tmp_path / "cap.json"
    f.write_text(json.dumps({"applied_cap_sigma": 7.0, "history": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        sp.applied_cap(path=f)
    assert sp.applied_cap(path=tmp_path / "absent.json")[0] == 4.0


def test_weekly_update_has_no_midweek_drift_and_keeps_history(tmp_path):
    f = tmp_path / "cap.json"
    w = [{"id": "a", "symbol": SYM, "closed": "x", "mae_sigma": 1.0, "stop_sigma": 3.0}]
    d = sp.run_update(T, path=f, winners=w)
    assert d["effective_from"] == "2026-11-09T00:00:00+00:00" and d["n"] == 1 and len(d["history"]) == 1
    same = sp.run_update(T + timedelta(days=3), path=f, winners=w * 30)          # Friday, same week: unchanged
    assert same["n"] == 1 and len(same["history"]) == 1
    nxt = sp.run_update(T + timedelta(days=6), path=f, winners=w * 30)           # Monday 2026-11-16
    assert nxt["n"] == 30 and len(nxt["history"]) == 2 and nxt["effective_from"].startswith("2026-11-16")
    # point in time: the cap in force before the second update is the first one
    assert sp.applied_cap(T + timedelta(days=2), f)[0] == pytest.approx(d["applied_cap_sigma"])
    assert sp.applied_cap(datetime(2026, 11, 1, tzinfo=UTC), f)[0] == 4.0
    for k in ("version", "n", "prior", "posterior", "history", "applied_cap_sigma"):
        assert k in nxt


def test_mae_in_risk_sigma_units():
    assert sp.mae_sigma("long", 100.0, np.array([99.0, 98.0]), np.array([101.0, 102.0]), 0.01) == pytest.approx(math.log(100 / 98) / 0.01)
    assert sp.mae_sigma("short", 100.0, np.array([99.0]), np.array([101.0]), 0.01) == pytest.approx(math.log(1.01) / 0.01)
    assert sp.mae_sigma("long", 100.0, np.array([100.5]), np.array([101.0]), 0.01) == 0.0


# ---------------------------------------------------------------- validation (no book needed)

def _meta_rows(blocked_second: bool = False):
    meta = {"session_id": "S1", "as_of": dm.iso(T), "deadline": dm.iso(T + dm.DEADLINE), "pipeline_test": False, "model_id": "claude-opus-5-5",
            "prompt_version": "v0.1", "packet_version": dm.PACKET_VERSION, "book_run": "ai"}
    rows = [{"id": "a", "p_quant": 0.45, "blocked": None}, {"id": "b", "p_quant": 0.45, "blocked": "gate: R-PX-2" if blocked_second else None}]
    return meta, rows


EMPTY = {"positions": [], "pending_orders": []}


def test_every_candidate_must_be_answered_once():
    meta, rows = _meta_rows()
    errs, _ = dm.validate_decisions({"decisions": [{"candidate_id": "a", "action": "take", "reason": "ok ok"}]}, meta, rows, EMPTY, {})
    assert any("unanswered" in e and "'b'" in e for e in errs)
    errs, _ = dm.validate_decisions({"decisions": [{"candidate_id": "a", "action": "skip", "reason": "no"}, {"candidate_id": "a", "action": "skip", "reason": "no"},
                                                   {"candidate_id": "b", "action": "skip", "reason": "nah"}]}, meta, rows, EMPTY, {})
    assert any("answered twice" in e for e in errs)
    errs, _ = dm.validate_decisions({"decisions": [{"candidate_id": "a", "action": "maybe", "reason": "hmm"}, {"candidate_id": "b", "action": "skip", "reason": ""}]},
                                    meta, rows, EMPTY, {})
    assert any("take or skip" in e for e in errs) and any("reason is required" in e for e in errs)
    meta, rows = _meta_rows(blocked_second=True)
    errs, _ = dm.validate_decisions({"decisions": [{"candidate_id": "a", "action": "skip", "reason": "no"}, {"candidate_id": "b", "action": "take", "reason": "yes"}]},
                                    meta, rows, EMPTY, {})
    assert any("can only be skipped" in e for e in errs)
    ok, norm = dm.validate_decisions({"decisions": [{"candidate_id": "a", "action": "take", "reason": "fine", "tilt": {"view_logodds": 0.5, "confidence": "high"}},
                                                    {"candidate_id": "b", "action": "skip", "reason": "blocked"}]}, meta, rows, EMPTY, {})
    assert ok == []
    a = norm["decisions"][0]
    assert a["p_final"] == pytest.approx(1 / (1 + math.exp(-(math.log(0.45 / 0.55) + 0.5))))
    assert a["p_used"] == 0.45, "a positive tilt never loosens the engine's E[R] gate"


@pytest.mark.parametrize("bad", [
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5, "qty": 3}]},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5, "size": "double"}]},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5, "stop": 2900.0}]},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5, "tilt": {"view_logodds": 0.2, "risk_multiplier": 2}}]},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5}], "risk_per_trade_pct": 2.0},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5}], "max_stop_distance_sigma": 8},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5, "tilt": {"view_logodds": 3.0, "confidence": "high"}}]},
    {"decisions": [{"candidate_id": "a", "action": "take", "reason": "x" * 5}], "model_id": "some-other-model"},
])
def test_size_stop_cap_and_limit_overrides_are_refused(bad):
    meta, rows = _meta_rows()
    bad = json.loads(json.dumps(bad))
    bad["decisions"].append({"candidate_id": "b", "action": "skip", "reason": "skip"})
    errs, _ = dm.validate_decisions(bad, meta, rows, EMPTY, {})
    assert errs, bad


# ---------------------------------------------------------------- end to end on the Rust paper book

@needs_atx
def test_size_override_is_refused_end_to_end_and_nothing_reaches_the_book(env):
    out, book = prep(env)
    sid = out.name
    doc = take_all(sid, ["shk-0"])
    doc["decisions"][0]["qty"] = 10
    with pytest.raises(dm.RefusedError, match="risk engine"):
        dm.decide(sid, doc, paths=env[0], wall=T + timedelta(minutes=5), book=book)
    assert not (out / "submitted.json").exists()
    assert (out / "refusals.jsonl").is_file()
    assert book.positions()["pending_orders"] == []


@needs_atx
def test_deadline_refusal(env):
    out, book = prep(env)
    sid = out.name
    with pytest.raises(dm.RefusedError, match="deadline"):
        dm.decide(sid, take_all(sid, ["shk-0"]), paths=env[0], wall=T + timedelta(minutes=31), book=book)
    with pytest.raises(dm.RefusedError, match="pipeline_test"):
        dm.decide(sid, take_all(sid, ["shk-0"]), paths=env[0], wall=T + timedelta(minutes=5), simulate_at=T + timedelta(minutes=5), book=book)
    assert not (out / "submitted.json").exists()
    # a real session cannot be prepared for a stale as_of either
    with pytest.raises(dm.RefusedError):
        prep(env, as_of=T + timedelta(hours=4), wall=T + timedelta(hours=5))


@needs_atx
def test_paper_book_persists_across_sessions_and_size_comes_from_the_engine(env):
    p, _ = env
    out, book = prep(env)
    sub = dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    assert sub["ids"] == ["shk-0"] and sub["accepted"] == ["shk-0"] and sub["tests"] == []
    it = json.loads((out / "intents.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "qty" not in json.dumps(it) and it["targets"] and it["sigma"]["per_bar"] == SIG
    pend = book.positions()["pending_orders"]
    assert len(pend) == 1 and pend[0]["qty"] > 0          # sized by the Rust risk engine
    risk = pend[0]["risk_usd"]
    assert risk == pytest.approx(dm.BOOK_EQUITY * 0.005, rel=0.02)
    # second session 4h later: the order filled at the first open after the decision, the position is carried over
    out2, book2 = prep(env, as_of=T + timedelta(hours=4))
    pos = book2.positions()
    assert len(pos["positions"]) == 1 and pos["pending_orders"] == []
    fill_ts = json.loads((book2.dir / "fills.jsonl").read_text(encoding="utf-8").splitlines()[0])["ts"]
    assert fill_ts == int((T.replace(minute=0) + timedelta(hours=1)).timestamp() * 1000), "fills at the 09:00 open, after the 08:15 decision"
    pk = (out2 / "PACKET.md").read_text(encoding="utf-8")
    assert "open 1" in pk and f"{out.name}/shk-0" in pk
    assert len(pk.splitlines()) <= 120
    # the new candidate on the held symbol is blocked (one position per symbol)
    rows = json.loads((out2 / "candidates.json").read_text(encoding="utf-8"))
    assert "already in the paper book" in rows[0]["blocked"]


@needs_atx
def test_no_lookahead(env, tmp_path):
    p, bars = env
    # data newer than as_of in a state card is refused
    with pytest.raises(LookAheadError):
        prep(env, analyse=fake_analyse(future_data=True))
    # the book never processes bars that close after as_of, though the CSV holds 2 more days
    out, book = prep(env)
    dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    t2 = T + timedelta(hours=3, minutes=30)
    book.advance(t2)
    st = json.loads((book.dir / "state.json").read_text(encoding="utf-8"))
    last = max(st["last_bar_ts"].values())
    assert last + 3_600_000 <= int(t2.timestamp() * 1000)
    fills = dm._jsonl(book.dir / "fills.jsonl")
    assert fills and all(f["ts"] + 3_600_000 <= int(t2.timestamp() * 1000) for f in fills)
    # and never moves backwards
    with pytest.raises(dm.RefusedError, match="backwards"):
        book.advance(T + timedelta(hours=1))
    # the live-network line is the newest one known at as_of
    p.live_nn.mkdir(parents=True, exist_ok=True)
    lines = [{"id": f"nn-{h}", "as_of": dm.iso(T.replace(minute=3) - timedelta(hours=1) + timedelta(hours=h)), "close": 1.0, "net": {}} for h in range(3)]
    (p.live_nn / f"{T.date()}.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    assert "nn-1 " in dm.live_line(T, p.live_nn) and "nn-2" not in dm.live_line(T, p.live_nn)


@needs_atx
def test_kill_switch_flattens_and_blocks(env):
    p, _ = env
    out, book = prep(env)
    dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    book.advance(T + timedelta(hours=2))
    assert len(book.positions()["positions"]) == 1
    p.kill.write_text("stop", encoding="utf-8")
    out2, book2 = prep(env, as_of=T + timedelta(hours=4))
    assert book2.positions()["positions"] == []
    assert book2.trades()[-1]["exit_reason"] == "kill"
    rows = json.loads((out2 / "candidates.json").read_text(encoding="utf-8"))
    assert all("KILL" in r["blocked"] for r in rows)
    with pytest.raises(dm.RefusedError, match="only be skipped"):
        dm.decide(out2.name, take_all(out2.name, [r["id"] for r in rows]), paths=p, wall=T + timedelta(hours=4, minutes=5), book=book2)


@needs_atx
def test_pipeline_test_sessions_are_excluded_from_the_scorecard(env):
    from ai_trader.agent.scorecard import ai_stats, run, summary

    p, _ = env
    out, book = prep(env, test=True)
    sub = dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(hours=9), simulate_at=T + timedelta(minutes=10), book=book)
    assert sub["pipeline_test"] and sub["tests"] == ["shk-0"]
    assert book.run == dm.TEST_RUN

    class StubLake:
        def bars(self, sym, freq, t, start=None):
            ts = [T.replace(minute=0) + timedelta(hours=k) for k in range(6)]
            return pl.DataFrame({"event_ts": ts, "close_ts": [x + timedelta(hours=1) for x in ts], "high": [3001.0] * 6, "low": [2999.0] * 6, "close": [3000.0] * 6})

    df = run(p.sessions, T + timedelta(hours=6), StubLake())
    assert df["test"].all()
    s = {r["book"]: r for r in summary(df)}
    assert s["AI (submitted)"]["n"] == 0 and s["RULE (sleeves, no Claude)"]["n"] == 0
    assert ai_stats(df)["H-AI1"]["closed_ai"] == 0


# ---------------------------------------------------------------- per-sleeve edge posterior + paper EV gate (2026-11-14)

from ai_trader.risk import edge_posterior as ep  # noqa: E402


def test_edge_prior_is_the_documented_edge_worth_30_pseudo_trades():
    assert ep.PRIOR_PSEUDO_TRADES == 30
    assert ep.prior_edges() == {"shock_regime": 0.05, "trend_4h": 0.03}
    p = ep.posterior(0.05, [])
    assert p["mean"] == 0.05 and p["sd"] == pytest.approx(0.5 / math.sqrt(30))
    ys = [0.6, -0.4, -0.4, 0.6, -0.4]
    q = ep.posterior(0.05, ys)
    assert q["mean"] == pytest.approx((30 * 0.05 + sum(ys)) / 35) and q["sd"] == pytest.approx(0.5 / math.sqrt(35)) and q["n"] == 5
    # sequential == batch
    assert ep.posterior(ep.posterior(0.05, ys[:2])["mean"], ys[2:], n0=32)["mean"] == pytest.approx(q["mean"])


def test_trade_observation_is_the_bracket_outcome_minus_the_null():
    it = {"context": {"p_null": 0.35, "g_net": 1.9}}
    assert ep.trade_y({"r_multiple": 1.9}, it) == pytest.approx(1 - 0.35)
    assert ep.trade_y({"r_multiple": -1.0}, it) == pytest.approx(-0.35)
    assert ep.trade_y({"r_multiple": -1.0}, {"context": {}}) is None


def test_edge_update_is_weekly_point_in_time_and_fail_closed(tmp_path):
    f = tmp_path / "edge.json"
    assert ep.applied_edges(T, f)[0] == {"shock_regime": 0.05, "trend_4h": 0.03}
    losers = {"shock_regime": [{"id": f"x{i}", "symbol": SYM, "closed_ts": 0, "r": -1.0, "exit_reason": "stop", "y": -0.35} for i in range(20)]}
    d = ep.run_update(T, path=f, trades=losers)
    assert d["sleeves"]["shock_regime"]["n"] == 20 and d["edges"]["shock_regime"] == pytest.approx((1.5 - 7.0) / 50)
    assert d["edges"]["trend_4h"] == pytest.approx(0.03)
    assert ep.run_update(T + timedelta(days=2), path=f, trades={})["n"] == 20, "no mid-week drift"
    assert ep.applied_edges(T, f)[0]["shock_regime"] < 0
    assert ep.applied_edges(datetime(2026, 11, 1, tzinfo=UTC), f)[0]["shock_regime"] == 0.05, "point in time: the prior before the first update"
    nxt = ep.run_update(T + timedelta(days=6), path=f, trades={})
    assert len(nxt["history"]) == 2 and nxt["edges"]["shock_regime"] == 0.05
    bad = json.loads(f.read_text())
    bad["history"][-1]["edges"]["shock_regime"] = 3.0
    f.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        ep.applied_edges(T + timedelta(days=7), f)


def test_session_p_quant_uses_the_edge():
    from ai_trader.agent.session import cap_stop, net_er, p_with_edge

    assert p_with_edge(0.33, 0.05) == pytest.approx(0.38) and p_with_edge(0.02, -0.1) == 0.0 and p_with_edge(0.93, 0.1) == 0.95
    px = 3000.0
    c = Candidate("c", SYM, "shock_regime", "long", "market", px, px * math.exp(-0.05), px * math.exp(0.1), 0.0, 0.0, 2.0, None, "", "majors",
                  sigma_per_bar=0.01)
    c.p_null = p_first(c.entry, c.stop, c.take_profit)
    cap_stop(c, 4.0, 0.5, 2.0, edge=-0.2)          # 5 sigma stop is capped, p_quant recomputed with the posterior edge
    assert c.p_quant == pytest.approx(max(0.0, c.p_null - 0.2)) and net_er(c) < 0


def edge_analyse(edge_file):
    """Like fake_analyse, but p_quant = p_null + the sleeve edge in force (as agent.session.analyse does)."""
    base = fake_analyse()

    def f(sym, t):
        card, cands, st = base(sym, t)
        e = ep.applied_edges(t, edge_file)[0]["shock_regime"]
        for c in cands:
            c.take_profit = c.entry * math.exp(4 * SIG)        # 4 sigma target: prior net E[R] about +0.08 (between 0 and 0.10)
            c.p_null = p_first(c.entry, c.stop, c.take_profit)
            c.rr = 4 / 3
            c.p_quant = min(0.95, max(0.0, c.p_null + e))
        return card, cands, st
    return f


@needs_atx
def test_paper_gate_accepts_small_positive_edges_and_a_bad_posterior_blocks_again(env):
    p, _ = env
    # prior edge +0.05: net E[R] about +0.06 -> blocked under 0.10, accepted under the paper gate (> 0), by Python and Rust
    out, book = prep(env, analyse=edge_analyse(p.edge))
    r = json.loads((out / "candidates.json").read_text(encoding="utf-8"))[0]
    assert 0 < r["net_er"] < 0.10 and r["blocked"] is None
    meta = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert meta["paper_gate"] == {"min_net_er": 0.0, "exclusive": True} and meta["edges"]["shock_regime"] == 0.05
    assert "edge" in (out / "PACKET.md").read_text(encoding="utf-8")
    sub = dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    assert sub["accepted"] == ["shk-0"], sub["rejected"]
    assert book.summary()["risk"]["min_net_expected_r"] == 0.0
    ctx = json.loads((out / "intents.jsonl").read_text(encoding="utf-8").splitlines()[0])["context"]
    assert {"p_null", "er_null", "g_net"} <= set(ctx)
    # a posterior below zero (e.g. after a losing streak) blocks the sleeve again
    losers = {"shock_regime": [{"id": f"x{i}", "symbol": SYM, "closed_ts": 0, "r": -1.0, "exit_reason": "stop", "y": -0.35} for i in range(20)]}
    p.edge.unlink()                                    # replace this week's (n = 0) update by one after a losing streak
    ep.run_update(T + timedelta(hours=4), path=p.edge, trades=losers)
    out2, _ = prep(env, as_of=T + timedelta(hours=4), analyse=edge_analyse(p.edge))
    r2 = json.loads((out2 / "candidates.json").read_text(encoding="utf-8"))[0]
    assert r2["net_er"] <= 0 and "paper gate" in r2["blocked"]


@needs_atx
def test_closed_paper_trades_feed_the_edge_posterior(tmp_path):
    p = dm.Paths(tmp_path / "sessions", tmp_path / "book", tmp_path / "KILL", tmp_path / "live_nn", tmp_path / "stop_cap.json", tmp_path / "journal")
    bars = tmp_path / "bars"
    t0 = T.replace(minute=0) - timedelta(hours=48)
    up = 3000.0 * math.exp(0.08)
    write_bars(bars, lambda k: (3000.0, 3003.0, 2997.0, 3000.0) if k < 51 else (up, up * 1.001, up * 0.999, up), t0, 120)   # jump through the target at 11:00
    env2 = (p, bars)
    out, book = prep(env2)
    dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    book.advance(T + timedelta(hours=6))
    tr = book.trades()
    assert len(tr) == 1 and tr[0]["exit_reason"] == "target"
    d = ep.run_update(T + timedelta(days=6), path=p.edge, root=p.book)      # Monday 2026-11-16
    s = d["sleeves"]["shock_regime"]
    p0 = json.loads((out / "candidates.json").read_text(encoding="utf-8"))[0]["p_null"]
    assert s["n"] == 1 and s["trades"][0]["y"] == pytest.approx(1 - p0, abs=0.05)
    assert s["posterior"]["mean"] == pytest.approx((30 * 0.05 + s["trades"][0]["y"]) / 31)


# ---------------------------------------------------------------- pegged limit entries at the touch (H-EXE1, 2026-11-15)

def test_pegged_entry_cost_model():
    from ai_trader.agent.scorecard import cost_bps
    from ai_trader.agent.session import entry_cost_bps, net_er

    assert entry_cost_bps("pegged_limit") == pytest.approx(2.0 + 1.3)
    assert entry_cost_bps("market") == pytest.approx(7.5) and entry_cost_bps("limit") == pytest.approx(2.0)
    assert cost_bps("pegged_limit") == pytest.approx(10.8) and cost_bps(None) == pytest.approx(15.0)
    px = 3000.0
    c = Candidate("c", SYM, "shock_regime", "long", "market", px, px * math.exp(-0.03), px * math.exp(0.06), 0.33, 0.38, 2.0, None, "", "majors")
    m = net_er(c)
    c.entry_type = "pegged_limit"
    assert net_er(c) > m, "maker + 1.3 bp is cheaper than taker + 2 bp"


@needs_atx
def test_paper_entries_are_limits_pegged_to_the_touch(env):
    p, _ = env
    out, book = prep(env)
    meta = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert meta["packet_version"] == "compact-2" and meta["exec_version"] == "pegged-limit-v1" and meta["tool_versions"]["exec"] == "pegged-limit-v1"
    assert meta["paper_entry"]["mode"] == "pegged_limit" and meta["paper_entry"]["slip_bps"] == 1.3
    head = (out / "PACKET.md").read_text(encoding="utf-8").splitlines()[:4]
    assert any("pegged to the touch" in x for x in head)
    row = json.loads((out / "candidates.json").read_text(encoding="utf-8"))[0]
    assert row["entry_type"] == "pegged_limit"
    dm.decide(out.name, take_all(out.name, ["shk-0"]), paths=p, wall=T + timedelta(minutes=10), book=book)
    it = json.loads((out / "intents.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert it["entry"] == {"type": "limit", "price": 3000.0, "peg": "touch"}
    dec = dm._jsonl(p.book / "decisions" / "ai.jsonl")[-1]
    assert dec["peg_slip_bps"] == 1.3 and dec["entry_type"] == "limit"
    book.advance(T + timedelta(hours=2))
    f = dm._jsonl(book.dir / "fills.jsonl")[0]
    assert f["price"] == pytest.approx(3000.0 * (1 + 1.3e-4)) and f["liquidity"] == "maker"
    assert f["ts"] == int((T.replace(minute=0) + timedelta(hours=1)).timestamp() * 1000)
    assert f["fee"] == pytest.approx(f["qty"] * f["price"] * 2.0e-4)
