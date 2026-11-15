"""Strategy runtime + bias gate (src/ai_trader/execution, agent K): rule gating, the OrderIntent contract, point in time,
placement bounds and the research-only safety checks. Synthetic bars only (no lake reads)."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.core.config import load
from ai_trader.execution import gate as gate_mod
from ai_trader.execution.gate import BiasGate, SymbolState
from ai_trader.execution.intents import TOP_KEYS, build_intent, validate_intent
from ai_trader.execution.runner import SafetyError, safety_check
from ai_trader.execution.signals import (
    SignalEvent,
    SymbolFrames,
    strategy_events,
    strategy_refs,
)

T0 = datetime(2025, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------- fixtures

def synthetic_bars(n: int = 24 * 260, seed: int = 3) -> pl.DataFrame:
    """1h bars with vol clustering and trending / ranging stretches (so trend, reversion and shock strategies fire)."""
    rng = np.random.default_rng(seed)
    sig = np.empty(n)
    v = 0.006**2
    for t in range(n):
        v = 0.006**2 * 0.02 + 0.93 * v + 0.05 * (sig[t - 1] ** 2 if t else v) * rng.standard_t(5) ** 2 / 1.67
        sig[t] = math.sqrt(v)
    drift = 0.0006 * np.sign(np.sin(np.arange(n) / (24 * 9)))
    r = drift + sig * rng.standard_t(4, n) / math.sqrt(2)
    r[rng.integers(500, n, 12)] *= 8.0            # a few shocks
    c = 100.0 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.002, n)))
    ets = [T0 + timedelta(hours=k) for k in range(n)]
    return pl.DataFrame({"event_ts": ets, "close_ts": [t + timedelta(hours=1) for t in ets], "known_ts": [t + timedelta(hours=1, seconds=2) for t in ets],
                         "open": o, "high": hi, "low": lo, "close": c, "volume": np.full(n, 1e3), "turnover": c * 1e3}).with_columns(
        pl.col("event_ts").dt.cast_time_unit("us"), pl.col("close_ts").dt.cast_time_unit("us"), pl.col("known_ts").dt.cast_time_unit("us"))


def frames_asof(bars: pl.DataFrame, as_of: datetime) -> SymbolFrames:
    """What a lake read at `as_of` would give: bars known by then, enriched from scratch (as research.frame.market_frame)."""
    from ai_trader.research.frame import add_robust_vol, enrich_bars

    b = bars.filter(pl.col("known_ts") <= as_of)
    f = add_robust_vol(enrich_bars(b))
    return SymbolFrames("SYNUSDT", "crypto", as_of, f)


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return synthetic_bars()


def make_event(strategy: str, family: str, side: str = "long", kind: str = "entry", ts: datetime | None = None, price: float = 100.0) -> SignalEvent:
    ts = ts or datetime(2025, 6, 4, 13, 0, 2, tzinfo=UTC)   # a Wednesday
    return SignalEvent(strategy, family, family, "1h", "SYNUSDT", "crypto", kind, side if kind == "entry" else "flat", "flat" if kind == "entry" else side,
                       0.6 if side == "long" else -0.6, 0.6 if side == "long" else -0.6, ts - timedelta(hours=1, seconds=2), ts - timedelta(seconds=2), ts, price, 0)


def ctx(cell: str, dev: float = 0.3, weekend: bool = False, vol_state: str = "normal") -> dict:
    return {"i": 0, "regime_cell": cell, "cell_now": cell, "vol_state": vol_state, "dir_state": "range", "weekday": "Wed", "forecast_day": "2025-06-04",
            "vol_pctile": 0.5, "asset_class": "crypto", "next_bar_tradfi_weekend": weekend, "dev_1h": dev}


# ---------------------------------------------------------------- rules

def test_gate_blocks_1h_mean_reversion_in_R4_and_passes_it_in_allowed_cell():
    g = BiasGate(jump=False)
    assert g.book.active("R-MAP-3")
    for strat in ("rsi_reversion", "zscore_reversion"):
        ev = make_event(strat, "mean_reversion")
        r4 = g.check(ev, ctx("R4"), None)
        assert r4.decision == "block" and "R-MAP-3" in r4.blocked_by, r4
        for allowed in ("R5", "R6"):
            ok = g.check(ev, ctx(allowed), None)
            assert ok.decision == "pass" and "R-MAP-3" not in ok.blocked_by, ok
    # the cells come from the rule text, not from code: R1 R2 R4 R7 are avoided
    cr = next(c for c in g.book.cell_rules if c["id"] == "R-MAP-3")
    assert cr["cells"] == {"R1", "R2", "R4", "R7"}


def test_gate_other_machine_rules():
    g = BiasGate(jump=False)
    # R-PX-2: a long entry against a -2.5 sigma deviation is a fade; with the deviation it is not
    assert "R-PX-2" in g.check(make_event("rsi_reversion", "mean_reversion", "long"), ctx("R5", dev=-2.5), None).blocked_by
    assert g.check(make_event("tsmom", "trend", "long"), ctx("R2", dev=2.5), None).decision == "pass"
    # R-MAP-4: 1h trend in a range cell
    assert "R-MAP-4" in g.check(make_event("tsmom", "trend"), ctx("R6"), None).blocked_by
    # R-AVOID-1: failed families are always blocked
    assert "R-AVOID-1" in g.check(make_event("funding_contrarian", "positioning"), ctx("R2"), None).blocked_by
    assert "R-AVOID-1" in g.check(make_event("geo_struct", "trend"), ctx("R2"), None).blocked_by
    # R-PX-9: no entries into the TradFi weekend window (shock_regime exempt) and no weekend-move strategies
    assert "R-PX-9" in g.check(make_event("tsmom", "trend"), ctx("R2", weekend=True), None).blocked_by
    assert "R-PX-9" in g.check(make_event("cme_gap_fill", "calendar"), ctx("R5"), None).blocked_by
    # exits are never blocked, whatever the state
    ex = g.check(make_event("zscore_reversion", "mean_reversion", "long", kind="exit"), ctx("R4", dev=-3.0, weekend=True), None)
    assert ex.decision == "pass" and "R-EXIT-1" in ex.rules_applied
    # side comes from the strategy only (no cell / calendar tilt)
    assert "R-PX-6" in g.check(make_event("tsmom", "trend"), ctx("R2"), None).rules_applied


def test_retired_rule_stops_gating():
    from ai_trader.research.rules import load_rules

    rules = [dict(r) for r in load_rules()]
    for r in rules:
        if r["id"] == "R-MAP-3":
            r["status"] = "retired"
    g = BiasGate(rules, jump=False)
    assert g.check(make_event("rsi_reversion", "mean_reversion"), ctx("R4"), None).decision == "pass"


# ---------------------------------------------------------------- end-to-end on synthetic bars

def run_intents(sf: SymbolFrames, strategies: list[str], start: datetime, end: datetime, cache) -> list[dict]:
    st = SymbolState.build(sf)
    g = BiasGate(jump=True, jump_cache_dir=cache, n_paths=400)
    out = []
    for ref in strategy_refs(strategies):
        evs, _ = strategy_events(ref, sf, start, end)
        for ev in evs:
            gr, plc, c = g.evaluate(ev, st)
            out.append(build_intent(ev, gr, plc, c, "replay", rules_hash=g.book.hash))
    return out


@pytest.fixture(scope="module")
def full_run(bars, tmp_path_factory):
    end = bars["known_ts"].max()
    sf = frames_asof(bars, end)
    start = end - timedelta(days=40)
    return run_intents(sf, ["tsmom", "zscore_reversion", "shock_regime"], start, end, tmp_path_factory.mktemp("jump_full")), start, end


def test_intents_follow_the_contract(full_run):
    intents, _, _ = full_run
    assert len(intents) >= 5
    doc = (load().root / "docs" / "EXECUTION_PIPELINE.md").read_text(encoding="utf-8")
    block = re.search(r"## OrderIntent.*?```json\n(.*?)```", doc, re.DOTALL).group(1)
    example = json.loads(block.replace('"replay|paper"', '"replay"'))
    assert set(example) == set(TOP_KEYS)
    for d in intents:
        validate_intent(d)
        assert set(d) == set(example)
        for k in ("entry", "sigma", "gate", "context"):
            assert set(example[k]) <= set(d[k]), (k, set(example[k]) - set(d[k]))
        json.dumps(d)  # serialisable
    assert {d["side"] for d in intents} >= {"flat"} and any(d["side"] != "flat" for d in intents)


def test_no_intent_carries_a_size(full_run):
    intents, _, _ = full_run
    txt = json.dumps(intents).lower()
    for k in ('"size"', '"qty"', '"quantity"', '"notional_usd"', '"risk_usd"', '"leverage"'):
        assert k not in txt
    bad = json.loads(json.dumps(intents[0]))
    bad["context"]["qty"] = 1.0
    with pytest.raises(ValueError, match="size-like"):
        validate_intent(bad)
    bad = json.loads(json.dumps(intents[0]))
    bad["size"] = 1.0
    with pytest.raises(ValueError):
        validate_intent(bad)


def test_point_in_time_appending_future_bars_does_not_change_past_intents(bars, full_run, tmp_path):
    full, start, end = full_run
    cut = start + (end - start) * 0.55
    past_full = [d for d in full if datetime.fromisoformat(d["as_of"]) <= cut]
    assert past_full, "no intents before the cut: pick another seed"
    sf_cut = frames_asof(bars, cut)          # the world as a pass at `cut` sees it (re-enriched from the truncated bars)
    past_cut = run_intents(sf_cut, ["tsmom", "zscore_reversion", "shock_regime"], start, cut, tmp_path / "jump_cut")
    assert json.dumps(past_cut, sort_keys=True) == json.dumps(past_full, sort_keys=True)


def test_stop_distance_respects_min_max_sigma(full_run, monkeypatch, bars):
    intents, _, end = full_run
    rk = load().risk
    lo, hi = rk["placement"]["min_stop_distance_sigma"], rk["placement"]["max_stop_distance_sigma"]
    ents = [d for d in intents if d["side"] != "flat"]
    assert ents
    for d in ents:
        s = d["gate"]["stop_sigma"]
        assert lo < s < hi      # strictly inside: the engine rounds the stop away from entry to the tick
        assert abs(math.log(d["stop"] / d["entry"]["price"])) <= rk["max_stop_distance_pct"] / 100 + 1e-12
        assert abs(abs(math.log(d["stop"] / d["entry"]["price"])) / d["sigma"]["per_bar"] - s) < 1e-6   # the contract's unit
        assert d["targets"] == [] and d["net_er_r"] is not None   # own exits (R-EXIT-1): the engine checks net_er_r
        # the risk sigma is the unshrunk one (R-VOL-2) and p_tp is a probability
        assert "R-VOL-2" in d["gate"]["rules_applied"] and 0 <= d["p_tp"] <= 1
    # extreme placement constants are clamped into the bounds
    sf = frames_asof(bars, end)
    st = SymbolState.build(sf)
    i = st.row_at(end)
    ev = make_event("tsmom", "trend", "long", ts=end, price=float(st.close[i]))
    g = BiasGate(jump=False)
    for k, want in ((40.0, hi), (0.01, lo)):
        monkeypatch.setattr(gate_mod, "STOP_K", k)
        p = g.placement(ev, g.context(ev, st), st)
        assert lo < p.stop_sigma < hi and (abs(p.stop_sigma - want) < 0.011 * want or "stop_capped_by_max_stop_distance_pct" in p.details["flags"])


# ---------------------------------------------------------------- safety

def test_runner_refuses_live_modes_and_mainnet(monkeypatch):
    monkeypatch.delenv("AI_TRADER_ALLOW_MAINNET", raising=False)
    monkeypatch.setenv("AI_TRADER_ENVIRONMENT", "demo")
    ex = load().settings["execution"]
    for flag in (False, True):
        # execution.allow_mainnet is the live desk's key (2026-11-15): on its own it neither stops nor enables paper / replay
        monkeypatch.setitem(ex, "allow_mainnet", flag)
        for m in ("live", "mainnet", "demo", ""):
            with pytest.raises(SafetyError):
                safety_check(m)
        safety_check("replay")
        safety_check("paper")
        # this process set up for mainnet: refused whatever the flag says
        monkeypatch.setenv("AI_TRADER_ENVIRONMENT", "mainnet")
        for m in ("paper", "replay"):
            with pytest.raises(SafetyError, match="mainnet"):
                safety_check(m)
        monkeypatch.setenv("AI_TRADER_ENVIRONMENT", "demo")
        for v in ("1", "true", "yes"):
            monkeypatch.setenv("AI_TRADER_ALLOW_MAINNET", v)
            for m in ("paper", "replay"):
                with pytest.raises(SafetyError, match="AI_TRADER_ALLOW_MAINNET"):
                    safety_check(m)
        monkeypatch.setenv("AI_TRADER_ALLOW_MAINNET", "0")
        safety_check("paper")
        monkeypatch.delenv("AI_TRADER_ALLOW_MAINNET")
        # settings.environment mainnet with no override: refused
        monkeypatch.delenv("AI_TRADER_ENVIRONMENT")
        monkeypatch.setitem(load().settings, "environment", "mainnet")
        with pytest.raises(SafetyError):
            safety_check("paper")
        monkeypatch.setitem(load().settings, "environment", "demo")
        monkeypatch.setenv("AI_TRADER_ENVIRONMENT", "demo")
