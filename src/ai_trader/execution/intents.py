"""OrderIntent JSONL writer and validator (docs/EXECUTION_PIPELINE.md, written by agent K, read by agent L).

One JSON object per line in data/exec/intents/<run_id>.jsonl. An intent NEVER carries a size: `validate_intent`
rejects any key anywhere in the object that names a quantity, notional, leverage or risk amount (CLAUDE.md principle 4:
sizing belongs to the risk engine). Exits (side 'flat') carry no stop, no targets, no p_tp and no net_er_r (null / []). Entries
carry targets = [] by default: every registered strategy exits on its own signal (R-EXIT-1) and the risk engine then
checks the intent's net_er_r (computed against the reference 3R bracket in context.reference_target).

Extra keys beyond the contract live only inside the nested `sigma`, `gate` and `context` objects (see the dated
Changes note in the contract); the top-level key set is exactly the contract's.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .gate import GateResult, Placement, exec_dir
from .signals import SignalEvent

NS = uuid.UUID("5b3f7f0e-2f51-4c1e-9a77-6a1d1f0e0c4b")   # namespace of deterministic intent ids
TOP_KEYS = ("intent_id", "ts_ms", "as_of", "mode", "strategy", "family", "symbol", "asset_class", "side", "signal", "entry", "stop",
            "targets", "sigma", "p_tp", "net_er_r", "gate", "context")
MODES = ("replay", "paper")
FAMILIES = ("trend", "mean_reversion", "breakout", "positioning", "calendar", "shock", "carry", "flow")
SIDES = ("long", "short", "flat")
DECISIONS = ("pass", "block", "reduce")
SIGMA_SOURCES = ("vol.har", "vol_ml")
ASSET_CLASSES = ("crypto", "tradfi")
FORBIDDEN = ("size", "qty", "quantity", "notional", "leverage", "risk_usd", "risk_r", "contracts", "lots")


def intent_id(mode: str, ev: SignalEvent) -> str:
    """Deterministic: the same strategy event in the same mode always gets the same id (idempotent re-runs)."""
    return str(uuid.uuid5(NS, f"{mode}|{ev.strategy}|{ev.symbol}|{ev.decision_ts.isoformat()}|{ev.kind}|{ev.side}"))


def _r(x: float | None, nd: int = 10) -> float | None:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(f"{x:.{nd}g}")


def _clean(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, float):
        return _r(d)
    if isinstance(d, datetime):
        return d.isoformat()
    return d


def build_intent(ev: SignalEvent, gate: GateResult, plc: Placement | None, ctx: dict, mode: str, *, rules_hash: str = "",
                 extra_context: dict | None = None) -> dict:
    entry = ev.kind == "entry"
    c = {k: ctx.get(k) for k in ("regime_cell", "weekday", "vol_pctile")}
    c["card"] = f"strategy_choice / {ev.asset_class} {'perp' if ev.asset_class == 'crypto' else 'TradFi perp'}"
    c.update({"kind": ev.kind, "prev_side": ev.prev_side, "freq": ev.freq, "cluster": ev.cluster, "cell_now": ctx.get("cell_now"),
              "vol_state": ctx.get("vol_state"), "dir_state": ctx.get("dir_state"), "forecast_day": ctx.get("forecast_day"),
              "next_bar_tradfi_weekend": ctx.get("next_bar_tradfi_weekend"), "dev_1h": ctx.get("dev_1h"),
              "bar_event_ts": ev.bar_event_ts.isoformat(), "bar_close_ts": ev.bar_close_ts.isoformat(), "held_signal": ev.held_signal,
              "llm_rules": ctx.get("llm_rules", []), "rules_hash": rules_hash})
    if plc is not None:
        c.update(plc.details)
    if extra_context:
        c.update(extra_context)
    d = {
        "intent_id": intent_id(mode, ev),
        "ts_ms": int(ev.decision_ts.timestamp() * 1000),
        "as_of": ev.decision_ts.isoformat(),
        "mode": mode,
        "strategy": ev.strategy,
        "family": ev.family,
        "symbol": ev.symbol,
        "asset_class": ev.asset_class,
        "side": ev.side,
        "signal": ev.signal,
        "entry": {"type": "market", "price": ev.price},
        "stop": plc.stop if (entry and plc) else None,
        "targets": list(plc.targets) if (entry and plc) else [],
        "sigma": dict(plc.sigma) if plc else {"per_bar": None, "daily_logHAR": None, "weekday_factor": None, "source": "vol.har"},
        "p_tp": plc.p_tp if (entry and plc) else None,
        "net_er_r": plc.net_er_r if (entry and plc) else None,
        "gate": {"decision": gate.decision, "rules_applied": list(gate.rules_applied), "reasons": list(gate.reasons),
                 "blocked_by": list(gate.blocked_by), "stop_sigma": plc.stop_sigma if plc else None},
        "context": c,
    }
    d = _clean(d)
    validate_intent(d)
    return d


def _forbidden_keys(o: Any, path: str = "") -> list[str]:
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            if any(kl == f or kl.startswith(f + "_") or kl.endswith("_" + f) for f in FORBIDDEN):
                bad.append(f"{path}{k}")
            bad += _forbidden_keys(v, f"{path}{k}.")
    elif isinstance(o, list):
        for j, v in enumerate(o):
            bad += _forbidden_keys(v, f"{path}[{j}].")
    return bad


def validate_intent(d: dict) -> None:
    """Raise ValueError if `d` does not follow the OrderIntent contract."""
    errs = []
    keys = tuple(d)
    if set(keys) != set(TOP_KEYS):
        errs.append(f"top-level keys {sorted(set(keys) ^ set(TOP_KEYS))} differ from the contract")
    bad = _forbidden_keys(d)
    if bad:
        errs.append(f"size-like keys are forbidden in an intent: {bad}")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    try:
        uuid.UUID(str(d.get("intent_id")))
    except ValueError:
        errs.append("intent_id is not a uuid")
    if not isinstance(d.get("ts_ms"), int) or d["ts_ms"] <= 0:
        errs.append("ts_ms must be a positive int")
    try:
        t = datetime.fromisoformat(str(d.get("as_of")))
        if t.tzinfo is None:
            errs.append("as_of must carry a timezone")
        elif abs(int(t.timestamp() * 1000) - int(d.get("ts_ms") or 0)) > 1:
            errs.append("as_of and ts_ms disagree")
    except ValueError:
        errs.append("as_of is not ISO8601")
    for k, allowed in (("mode", MODES), ("family", FAMILIES), ("side", SIDES), ("asset_class", ASSET_CLASSES)):
        if d.get(k) not in allowed:
            errs.append(f"{k}={d.get(k)!r} not in {allowed}")
    for k in ("strategy", "symbol"):
        if not isinstance(d.get(k), str) or not d[k]:
            errs.append(f"{k} must be a non-empty string")
    if not num(d.get("signal")) or not -1.0 <= d["signal"] <= 1.0:
        errs.append("signal must be a number in [-1, 1]")
    e = d.get("entry") or {}
    if e.get("type") not in ("market", "limit") or not num(e.get("price")) or e["price"] <= 0:
        errs.append("entry must be {type: market|limit, price > 0}")
    g = d.get("gate") or {}
    if g.get("decision") not in DECISIONS or not isinstance(g.get("rules_applied"), list) or not isinstance(g.get("reasons"), list):
        errs.append("gate must be {decision: pass|block|reduce, rules_applied: [...], reasons: [...]}")
    s = d.get("sigma") or {}
    if s.get("source") not in SIGMA_SOURCES:
        errs.append(f"sigma.source not in {SIGMA_SOURCES}")
    for k in ("per_bar", "daily_logHAR", "weekday_factor"):
        if k not in s:
            errs.append(f"sigma.{k} missing")
    if not isinstance(d.get("context"), dict):
        errs.append("context must be an object")
    if d.get("side") == "flat":
        if d.get("stop") is not None or d.get("targets") or d.get("p_tp") is not None or d.get("net_er_r") is not None:
            errs.append("an exit (side flat) carries no stop / targets / p_tp / net_er_r")
    elif d.get("side") in ("long", "short") and g.get("decision") != "block":
        px = e.get("price")
        st, tg = d.get("stop"), d.get("targets") or []
        if not num(st) or not all(num(x) for x in tg):
            errs.append("a passed entry needs a numeric stop (and numeric targets, if any)")
        elif num(px):
            sgn = 1 if d["side"] == "long" else -1
            if not (sgn * (px - st) > 0 and all(sgn * (x - px) > 0 for x in tg)):
                errs.append("stop / targets are on the wrong side of the entry")
        if not num(d.get("p_tp")) or not 0.0 <= d["p_tp"] <= 1.0:
            errs.append("p_tp must be in [0, 1]")
        if not num(d.get("net_er_r")):
            errs.append("net_er_r must be a number (own-exit strategies send no target, so the engine checks this value)")
        if not num(s.get("per_bar")) or s["per_bar"] <= 0:
            errs.append("sigma.per_bar must be > 0 (one bar of the strategy's own frequency)")
    if errs:
        raise ValueError(f"invalid OrderIntent {d.get('intent_id')}: " + "; ".join(errs))


def intents_path(run_id: str) -> Path:
    return exec_dir() / "intents" / f"{run_id}.jsonl"


def write_intents(run_id: str, intents: Iterable[dict], path: Path | None = None) -> Path:
    """Validate every intent and append it to data/exec/intents/<run_id>.jsonl (one line each)."""
    p = path or intents_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(intents)
    for d in rows:
        validate_intent(d)
    with p.open("a", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")
    return p


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
