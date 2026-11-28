"""Desk report contract (research/desk/CONTRACT.md): schemas and a validator for analyst reports and trader proposals.

    uv run python -m ai_trader.desk.contract validate FILE [--snapshot research/desk/<date>/<HHMM>]

FILE is an analyst report (JSON, "contract": "desk_report_v1"), its markdown twin (.md), or a trader proposal file
(JSON with "proposals"). Errors make the file unusable; warnings are shown to the orchestrator. The validator also
derives numbers the trader must not invent: the driftless first-passage P(target before stop) and the reward / risk
ratio in log distance, and it checks each cited evidence number against the snapshot when it can resolve the path.

Proposal files are a strict superset of `atx propose` (docs/EXECUTION.md: top level session_id / proposals / clusters /
meta; each proposal id, symbol, side, entry_type, entry, stop, take_profit, p_tp, cluster, channel). The desk fields
(entry_mode, peg, targets, horizon, thesis, equilibrium, supports, rules, invalidation, evidence) ride inside each
proposal and are ignored by the current serde parser (TradeProposal does not deny unknown fields). A size is never
part of a proposal: any size-like key is an error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

CONTRACT = "desk_report_v1"
AGENTS = ("xs", "geometry", "news", "sentiment", "trader")
IDEA_KINDS = ("relative_value", "carry", "positioning", "level", "event", "sentiment", "flow", "volatility", "watch", "no_trade")
DIRECTIONS = ("long", "short", "pair", "none")
SIZE_KEYS = frozenset({"qty", "size", "quantity", "notional", "leverage", "risk_usd", "risk_amount", "contracts", "margin", "position_size"})
HORIZONS = ("15m", "1h", "4h", "12h", "24h", "3d", "7d")
EQUILIBRIUM_KEYS = ("positioning", "trapped", "stops", "forced_flows", "payoff_asymmetry", "next_moves", "pain_thresholds")
MAX_PROPOSALS = 3
MAX_MD_LINES = 80
P_TP_TOL = 0.10
SNAP_FILES = ("xs", "geometry", "news", "sentiment", "tape", "account", "gates")   # gates.json: D2 / D3 entry gates (live.gates)
MD_SECTIONS = ("Regime", "Ideas", "No trade / watch", "Data gaps", "Honesty")


def rule_ids(root: Path | None = None) -> set[str]:
    import yaml

    from ai_trader.core.config import find_root

    p = (root or find_root()) / "research" / "rules.yaml"
    return {r["id"] for r in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("rules", [])}


def p_tp_driftless(side: str, entry: float, stop: float, target: float) -> float | None:
    """P(target before stop) for a driftless log-price random walk started at entry (gambler's ruin in log space)."""
    try:
        e, s, t = math.log(entry), math.log(stop), math.log(target)
    except (TypeError, ValueError):
        return None
    if side == "long" and s < e < t:
        return (e - s) / (t - s)
    if side == "short" and t < e < s:
        return (s - e) / (s - t)
    return None


def reward_risk(side: str, entry: float, stop: float, target: float) -> float | None:
    try:
        r = abs(math.log(target / entry)) / abs(math.log(entry / stop))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return r if math.isfinite(r) else None


# ---------------------------------------------------------------- snapshot lookup

def load_snapshot(d: Path | None) -> dict:
    if d is None:
        return {}
    out = {}
    for k in SNAP_FILES:
        p = Path(d) / f"{k}.json"
        if p.is_file():
            out[k] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _token(snap: dict, source: str, symbol: str | None) -> dict | None:
    dom = snap.get(source.replace(".json", ""))
    if dom is None:
        return None
    if symbol is None:
        return dom
    for t in dom.get("tokens", []):
        if t.get("symbol") == symbol or t.get("ticker") == symbol:
            return t
    return None


def resolve(snap: dict, source: str, symbol: str | None, path: str):
    node = _token(snap, source, symbol)
    for part in path.split("."):
        if node is None:
            return None
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def matches(actual: float, reported) -> bool:
    """Within 1% or within the rounding of the reported value's printed precision (0.0 matches 0.002; -2.3 matches -2.31)."""
    txt = repr(float(reported)) if not isinstance(reported, int) else str(reported)
    dec = len(txt.split(".")[1]) if "." in txt and "e" not in txt else 0
    return math.isclose(actual, float(reported), rel_tol=0.01, abs_tol=1e-9) or abs(actual - float(reported)) <= 0.5 * 10 ** (-dec) + 1e-12


def check_evidence(ev: list, snap: dict, where: str, errors: list, warnings: list) -> None:
    if not isinstance(ev, list) or not ev:
        errors.append(f"{where}: evidence must be a non-empty list")
        return
    nums = 0
    for i, e in enumerate(ev):
        if not isinstance(e, dict) or "metric" not in e or "value" not in e or "source" not in e:
            errors.append(f"{where}.evidence[{i}]: needs metric, value, source")
            continue
        v = e["value"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums += 1
            if snap and e["source"].replace(".json", "") in snap:
                got = resolve(snap, e["source"], e.get("symbol"), e["metric"])
                if isinstance(got, (int, float)) and not isinstance(got, bool):
                    if not matches(float(got), v):
                        errors.append(f"{where}.evidence[{i}]: {e['source']} {e.get('symbol')} {e['metric']} = {got}, report says {v}")
                elif got is None:
                    warnings.append(f"{where}.evidence[{i}]: could not resolve {e['source']}:{e.get('symbol')}:{e['metric']} in the snapshot")
    if nums == 0:
        errors.append(f"{where}: no numeric evidence (numbers come from the snapshot tools)")


# ---------------------------------------------------------------- analyst reports

def validate_report(d: dict, snap: dict | None = None, rules: set[str] | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    snap = snap or {}
    if d.get("contract") != CONTRACT:
        errors.append(f"contract must be {CONTRACT!r}")
    if d.get("agent") not in AGENTS:
        errors.append(f"agent must be one of {AGENTS}")
    for k in ("cycle_id", "as_of", "snapshot", "model", "prompt_version"):
        if not d.get(k):
            errors.append(f"missing {k}")
    keys = list(d.keys())
    if "regime" not in d or not isinstance(d["regime"], dict) or not d["regime"].get("summary"):
        errors.append("regime.summary is required and must come first")
    elif "ideas" in keys and keys.index("regime") > keys.index("ideas"):
        errors.append("regime must be stated before ideas")
    ideas = d.get("ideas")
    if not isinstance(ideas, list):
        errors.append("ideas must be a list (it may be empty with a no_trade reason)")
        ideas = []
    if not ideas and not d.get("no_trade"):
        errors.append("no ideas and no no_trade reason")
    prev_conv, ranks = 2.0, []
    for i, it in enumerate(ideas):
        w = f"ideas[{i}]"
        ranks.append(it.get("rank"))
        if it.get("kind") not in IDEA_KINDS:
            errors.append(f"{w}.kind must be one of {IDEA_KINDS}")
        if it.get("direction") not in DIRECTIONS:
            errors.append(f"{w}.direction must be one of {DIRECTIONS}")
        th = it.get("thesis") or ""
        if not th:
            errors.append(f"{w}.thesis required")
        elif len(th) > 400:
            warnings.append(f"{w}.thesis longer than 400 chars")
        if not it.get("symbols"):
            errors.append(f"{w}.symbols required")
        if not it.get("reasoning"):
            errors.append(f"{w}.reasoning (why this rank) required")
        inv = it.get("invalidation")
        if not isinstance(inv, dict) or not (inv.get("price") is not None or inv.get("condition")):
            errors.append(f"{w}.invalidation needs a price level or a condition")
        c = it.get("conviction")
        if not isinstance(c, (int, float)) or not 0 <= c <= 1:
            errors.append(f"{w}.conviction must be a number in [0, 1]")
        else:
            if c > prev_conv + 1e-9:
                errors.append(f"{w}: ideas must be ranked by conviction (descending)")
            prev_conv = c
        check_evidence(it.get("evidence"), snap, w, errors, warnings)
        cited = it.get("rules") or []
        if rules is not None:
            for r in cited:
                if r not in rules:
                    errors.append(f"{w}: unknown rule {r}")
        if it.get("direction") in ("long", "short", "pair") and not cited:
            warnings.append(f"{w}: a directional idea should cite the rules it respects (e.g. R-BUDGET-1, R-PX-4)")
        for k in SIZE_KEYS & set(it):
            errors.append(f"{w}: size key {k!r} is not allowed (the risk engine sizes)")
    if ranks and ranks != list(range(1, len(ranks) + 1)):
        errors.append("ideas[].rank must be 1..n in order")
    h = d.get("honesty")
    if not isinstance(h, dict) or "direction_claim" not in h or not h.get("what_would_change_my_mind"):
        errors.append("honesty needs direction_claim (bool) and what_would_change_my_mind")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def validate_markdown(text: str) -> dict:
    errors, warnings = [], []
    lines = text.splitlines()
    if len(lines) > MAX_MD_LINES:
        errors.append(f"markdown has {len(lines)} lines (max {MAX_MD_LINES})")
    heads = [ln[3:].strip() for ln in lines if ln.startswith("## ")]
    if not heads or heads[0] != "Regime":
        errors.append("the first section must be '## Regime'")
    missing = [s for s in MD_SECTIONS if s not in heads]
    if missing:
        errors.append(f"missing sections {missing}")
    idea_heads = [ln for ln in lines if ln.startswith("### ")]
    for ln in idea_heads:
        if not re.search(r"conviction (0(\.\d+)?|1(\.0+)?)\)", ln):
            errors.append(f"idea heading without '(conviction x.xx)': {ln[:60]}")
    for label in ("Thesis:", "Evidence:", "Invalidation:"):
        n = sum(1 for ln in lines if ln.strip().startswith(f"- {label}"))
        if idea_heads and n < len(idea_heads):
            errors.append(f"every idea needs a '- {label}' line")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------- trader proposals

def _walk_keys(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield f"{path}.{k}".lstrip("."), k
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _walk_keys(v, f"{path}[{i}]")


def validate_proposals(d: dict, snap: dict | None = None, rules: set[str] | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    derived: list[dict] = []
    snap = snap or {}
    extra = set(d) - {"session_id", "proposals", "clusters", "meta"}
    if extra:
        errors.append(f"unknown top-level keys {sorted(extra)} (atx propose denies them)")
    if not d.get("session_id"):
        errors.append("session_id required")
    meta = d.get("meta") or {}
    for k in ("model", "prompt_version", "as_of", "snapshot", "cycle_id"):
        if not meta.get(k):
            errors.append(f"meta.{k} required (pin and log)")
    props = d.get("proposals")
    if not isinstance(props, list) or not props:
        errors.append("proposals must be a non-empty list (write no file when there is no trade)")
        props = []
    if len(props) > MAX_PROPOSALS:
        errors.append(f"at most {MAX_PROPOSALS} proposals per cycle (max 3 open)")
    for p, k in _walk_keys(d):
        if k.lower() in SIZE_KEYS:
            errors.append(f"size-like key {p!r}: proposals never carry a size")
    universe = {t["symbol"]: t for t in snap.get("xs", {}).get("tokens", [])}
    geo = {t["symbol"]: t for t in snap.get("geometry", {}).get("tokens", []) if "symbol" in t}
    tape = {t["symbol"]: t for t in snap.get("tape", {}).get("tokens", [])}
    gates = snap.get("gates")
    ids = set()
    for i, p in enumerate(props):
        w = f"proposals[{i}]"
        pid, sym, side = p.get("id"), (p.get("symbol") or "").upper(), p.get("side")
        if not pid or pid in ids:
            errors.append(f"{w}: id missing or duplicate")
        ids.add(pid)
        if not sym:
            errors.append(f"{w}: symbol required")
        if sym == "SPYUSDT":
            errors.append(f"{w}: SPYUSDT is never traded")
        if universe and sym not in universe:
            errors.append(f"{w}: {sym} is not in the snapshot universe")
        if side not in ("long", "short"):
            errors.append(f"{w}: side must be long | short")
        if p.get("entry_type") != "limit" or p.get("entry_mode") != "pegged_limit":
            errors.append(f"{w}: entry_type 'limit' with entry_mode 'pegged_limit' required (R-EXE-1)")
        if p.get("peg", "touch") != "touch":
            errors.append(f"{w}: peg must be 'touch'")
        for k in ("cluster", "channel"):
            if not isinstance(p.get(k), str):
                errors.append(f"{w}: {k} must be a string (may be empty)")
        nums = {k: p.get(k) for k in ("entry", "stop", "take_profit", "p_tp")}
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for k, v in nums.items() if k != "p_tp"):
            errors.append(f"{w}: entry, stop, take_profit must be positive numbers")
            continue
        e, s, tp = float(p["entry"]), float(p["stop"]), float(p["take_profit"])
        tg = p.get("targets") or []
        if not tg or not all(isinstance(x, (int, float)) for x in tg):
            errors.append(f"{w}: targets must be a non-empty list of prices")
            tg = [tp]
        if not any(math.isclose(tp, float(x), rel_tol=1e-9) for x in tg):
            errors.append(f"{w}: take_profit must be one of targets (it is the bracket target)")
        if side == "long" and not (s < e < min(tg)):
            errors.append(f"{w}: long needs stop < entry < every target")
        if side == "short" and not (s > e > max(tg)):
            errors.append(f"{w}: short needs stop > entry > every target")
        if p.get("horizon") not in HORIZONS:
            errors.append(f"{w}: horizon must be one of {HORIZONS}")
        for k in ("thesis", "invalidation"):
            if not p.get(k):
                errors.append(f"{w}: {k} required")
        eq = p.get("equilibrium") or {}
        miss = [k for k in EQUILIBRIUM_KEYS if not eq.get(k)]
        if miss:
            errors.append(f"{w}: equilibrium needs {miss}")
        sup = p.get("supports") or []
        if not sup or not all(isinstance(x, dict) and x.get("agent") in AGENTS[:4] and x.get("idea") for x in sup):
            errors.append(f"{w}: supports must list analyst inputs as {{agent, idea}}")
        cited = p.get("rules") or []
        if not cited:
            errors.append(f"{w}: rules required (ids from research/rules.yaml)")
        if rules is not None:
            for r in cited:
                if r not in rules:
                    errors.append(f"{w}: unknown rule {r}")
        check_evidence(p.get("evidence"), snap, w, errors, warnings)
        pt = p.get("p_tp")
        p0 = p_tp_driftless(side, e, s, tp) if side in ("long", "short") else None
        rr = reward_risk(side, e, s, tp)
        if not isinstance(pt, (int, float)) or not 0 <= pt <= 1:
            errors.append(f"{w}: p_tp must be in [0, 1]")
        elif p0 is not None and abs(pt - p0) > P_TP_TOL and not p.get("p_tp_source"):
            warnings.append(f"{w}: p_tp {pt:.2f} vs driftless {p0:.2f}: cite p_tp_source (a tool) or use the driftless value (R-BUDGET-1, R-FLD-3)")
        row = {"id": pid, "symbol": sym, "p_tp_driftless": None if p0 is None else round(p0, 4), "reward_risk": None if rr is None else round(rr, 3),
               "stop_dist_pct": round(100 * abs(math.log(e / s)), 3)}
        g = geo.get(sym)
        if g and g.get("atr14"):
            a1h, a1d = g["atr14"].get("1h"), g["atr14"].get("1d")
            if a1h:
                row["stop_dist_atr_1h"] = round(abs(e - s) / a1h, 2)
                if abs(e - s) < 0.5 * a1h:
                    warnings.append(f"{w}: stop is {row['stop_dist_atr_1h']} x 1h ATR from entry: inside the noise")
            if a1d and abs(e - s) > 3 * a1d:
                warnings.append(f"{w}: stop is more than 3 x 1d ATR from entry")
        last = (universe.get(sym) or {}).get("last") or ((tape.get(sym) or {}).get("price") or {}).get("last")
        if last:
            row["entry_vs_last_pct"] = round(100 * math.log(e / last), 3)
            if abs(row["entry_vs_last_pct"]) > 0.5:
                warnings.append(f"{w}: entry {row['entry_vs_last_pct']:+.2f}% from last: a touch peg fills near the touch, not at a distant price")
        cell = ((universe.get(sym) or {}).get("regime") or {}).get("cell")
        if side == "long" and cell == "R9" and "R-RB-1" not in cited:
            warnings.append(f"{w}: {sym} is in R9 (liquidation): a long must address R-RB-1 (the flush low usually comes later)")
        warnings += _gate_warnings(w, sym, cited, gates, tape)
        derived.append(row)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "derived": derived}


def _gate_warnings(w: str, sym: str, cited, gates: dict | None, tape: dict) -> list[str]:
    """R-FLOW-2 on the live jump-intensity gate (D3) and R-WS05-2 (D2), from the gates state (data/live/gates.json, in the
    snapshot or fresh live). Without a gates state the old 60-min tape cluster flag is the fallback."""
    out = []
    if not gates or not gates.get("jump"):
        cl = (((tape.get(sym) or {}).get("liq") or {}).get("cluster_60m") or {})
        if cl.get("fired") and "R-FLOW-2" not in cited:
            out.append(f"{w}: a liquidation cluster fired on {sym} in the last 60 min (no live gates state: 60-min fallback): address R-FLOW-2 (do not fade it)")
        return out
    jm = gates["jump"]
    fb = (gates.get("jump_model") or {}).get("fallback_symbol", "BTCUSDT")
    j = jm.get(sym) or jm.get(fb) or {"elevated": True, "reason": "no jump state"}
    if j.get("elevated") and "R-FLOW-2" not in cited:
        comp = j.get("components") or {}
        why = (f"x{j['ratio']:.2f}" if j.get("ratio") is not None else (j.get("reason") or "?")) + (f" until {j['until_utc'][11:16]}Z" if j.get("until_utc") else "")
        liq = (comp.get("liq") or 0) > 0 or (j.get("liq_cluster_minutes_6h") or 0) > 0
        out.append(f"{w}: {sym} jump gate elevated ({why}{', liquidation cascade' if liq else ''}{'' if jm.get(sym) else ', ' + fb + ' fallback'}): "
                   "address R-FLOW-2 (do not fade a cascade; atx live refuses stops < 2 sigma while elevated)")
    if "R-WS05-2" not in cited:
        act = [x for x in gates.get("release_windows", []) if x.get("active")]
        if act:
            out.append(f"{w}: inside the {act[0]['event']} release window: address R-WS05-2 (atx live refuses stops < 2 sigma)")
    return out


def _live_gates(max_age_ms: int = 300_000) -> dict | None:
    """The live gates file when it is fresh (validation runs at decision time), else None."""
    try:
        import time

        from ai_trader.live.gates import load
        g = load()
        return g if g and g.get("written_ms") and time.time() * 1000 - g["written_ms"] <= max_age_ms else None
    except Exception:  # noqa: BLE001
        return None


def validate_file(path: Path, snapshot: Path | None = None) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".md":
        return validate_markdown(text)
    d = json.loads(text)
    if snapshot is None:
        ref = d.get("snapshot") or (d.get("meta") or {}).get("snapshot")
        snapshot = Path(ref) if ref and Path(ref).is_dir() else None
    snap = load_snapshot(snapshot)
    if "proposals" in d and "gates" not in snap and (g := _live_gates()) is not None:
        snap["gates"] = g
    try:
        rules = rule_ids()
    except Exception:  # noqa: BLE001
        rules = None
    if "proposals" in d:
        return validate_proposals(d, snap, rules)
    return validate_report(d, snap, rules)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ai_trader.desk.contract")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.add_argument("--snapshot", default=None)
    a = ap.parse_args(argv)
    if a.snapshot and not Path(a.snapshot).is_dir():
        print(json.dumps({"ok": False, "errors": [f"snapshot dir {a.snapshot} not found"]}))
        return 2
    r = validate_file(Path(a.file), Path(a.snapshot) if a.snapshot else None)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
