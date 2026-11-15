"""Confluence log: the trader's signed per-layer reading for each candidate asset, every desk cycle.

    uv run python -m ai_trader.narrative.confluence validate <snapshot>/reports/confluence.json [--snapshot DIR]
    uv run python -m ai_trader.narrative.confluence log <snapshot>/reports/confluence.json [--snapshot DIR]
    uv run python -m ai_trader.narrative.confluence status

Spec: research/desk/CONFLUENCE.md. One record per cycle (`confluence_v1`); `log` validates it and appends it to
`data/journal/confluence/<YYYY-MM>.jsonl` (append-only; a cycle id is logged once). The point is to turn the user's
discretionary process (levels, flow, book, structure, momentum, vol / liquidity, positioning, narrative, on-chain)
into numbers logged *before* outcomes, so that `scripts/confluence_fit.py` can later measure, walk-forward, which
layers carry information. Readings never size anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from ai_trader.core.config import load

CONTRACT = "confluence_v1"
LAYERS = ("levels", "flow", "book", "structure", "momentum", "vol_liq", "positioning", "narrative", "onchain")
LAYER_DOC = {
    "levels": "levels / liquidity: unfilled levels, untouched leg opens, unswept swing points, stop pools (geometry.json)",
    "flow": "taker delta and flow imbalance (tape.json flow)",
    "book": "order-book depth and asymmetry (tape.json book)",
    "structure": "break of structure / change of character (geometry.json swings; BoS research)",
    "momentum": "returns and relative strength across 1h / 4h / 24h / 7d (xs.json)",
    "vol_liq": "volatility / liquidity state: thin books with rising vol (tape.json, players.dvol, v0 cell)",
    "positioning": "funding, OI change, liquidations (xs.json funding / oi, tape.json liq)",
    "narrative": "active theses in the ledger (theses.json: status, direction, net_support)",
    "onchain": "placeholder until a source exists (whale / exchange-flow alerts); log null with reason 'no source'",
}
SOURCES = ("xs.json", "geometry.json", "tape.json", "news.json", "sentiment.json", "theses.json", "account.json")
SIZE_KEYS = frozenset({"qty", "size", "quantity", "notional", "leverage", "risk_usd", "risk_amount", "contracts", "margin", "position_size"})
MIN_CYCLES_TO_FIT = 300


def journal_dir() -> Path:
    p = load().root / "data" / "journal" / "confluence"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _keys(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [str(k) for k in x] + [k for v in x.values() for k in _keys(v)]
    if isinstance(x, list):
        return [k for v in x for k in _keys(v)]
    return []


def _resolve_theses(snap_dir: Path | None, metric: str) -> Any:
    if snap_dir is None or not (snap_dir / "theses.json").is_file():
        return None
    d = json.loads((snap_dir / "theses.json").read_text(encoding="utf-8"))
    tid, _, path = metric.partition(".")
    node: Any = next((t for t in d.get("theses", []) if t.get("id") == tid), None)
    for part in path.split(".") if path else []:
        node = node.get(part) if isinstance(node, dict) else None
    return node


def validate(rec: dict[str, Any], snap_dir: Path | None = None) -> dict[str, list[str]]:
    """errors (the record is rejected) and warnings (logged, but the evidence could not be checked)."""
    from ai_trader.desk.contract import load_snapshot, matches, resolve

    errors: list[str] = []
    warnings: list[str] = []
    for k in ("contract", "cycle_id", "as_of", "snapshot", "model", "prompt_version", "assets"):
        if k not in rec:
            errors.append(f"missing {k}")
    if errors:
        return {"errors": errors, "warnings": warnings}
    if rec["contract"] != CONTRACT:
        errors.append(f"contract must be {CONTRACT}")
    bad = SIZE_KEYS & set(_keys(rec))
    if bad:
        errors.append(f"size keys {sorted(bad)}: readings never size (the risk engine sizes)")
    snap = load_snapshot(snap_dir) if snap_dir else {}
    if not isinstance(rec["assets"], list) or not rec["assets"]:
        errors.append("assets must be a non-empty list (one entry per candidate asset)")
        return {"errors": errors, "warnings": warnings}
    seen = set()
    for i, a in enumerate(rec["assets"]):
        sym = a.get("symbol")
        where = f"assets[{i}] {sym}"
        if not sym or sym == "SPYUSDT":
            errors.append(f"{where}: symbol required, never SPYUSDT")
        if sym in seen:
            errors.append(f"{where}: duplicate symbol")
        seen.add(sym)
        rd = a.get("readings", {})
        missing = [L for L in LAYERS if L not in rd]
        if missing:
            errors.append(f"{where}: missing layers {missing} (log null with a reason when a layer has no reading)")
        for L, r in rd.items():
            if L not in LAYERS:
                errors.append(f"{where}: unknown layer {L}")
                continue
            if not isinstance(r, dict) or "value" not in r:
                errors.append(f"{where}.{L}: needs {{value, evidence | reason}}")
                continue
            v = r["value"]
            if v is None:
                if not r.get("reason"):
                    errors.append(f"{where}.{L}: a null reading needs a reason")
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not -1 <= v <= 1:
                errors.append(f"{where}.{L}: value must be a number in [-1, 1] or null")
                continue
            ev = r.get("evidence")
            if not ev:
                errors.append(f"{where}.{L}: a reading needs at least one evidence pointer")
                continue
            for j, e in enumerate(ev):
                if not isinstance(e, dict) or e.get("source") not in SOURCES or "metric" not in e:
                    errors.append(f"{where}.{L}.evidence[{j}]: needs source in {SOURCES} and metric")
                    continue
                if "value" not in e:
                    continue
                if e["source"] == "theses.json":
                    got = _resolve_theses(snap_dir, e["metric"])
                elif snap:
                    got = resolve(snap, e["source"], e.get("symbol", sym), e["metric"])
                else:
                    got = None
                ev_v = e["value"]
                if isinstance(got, (int, float)) and not isinstance(got, bool) and isinstance(ev_v, (int, float)) and not isinstance(ev_v, bool):
                    if not matches(float(got), ev_v):
                        errors.append(f"{where}.{L}.evidence[{j}]: {e['source']} {e['metric']} = {got}, reading cites {ev_v}")
                elif got is None and snap_dir is not None:
                    warnings.append(f"{where}.{L}.evidence[{j}]: could not resolve {e['source']}:{e['metric']}")
                elif got is not None and (isinstance(got, bool) or not isinstance(got, (int, float))) and got != ev_v:
                    errors.append(f"{where}.{L}.evidence[{j}]: {e['source']} {e['metric']} = {got!r}, reading cites {ev_v!r}")
    return {"errors": errors, "warnings": warnings}


def logged_cycles() -> set[str]:
    out: set[str] = set()
    for p in sorted(journal_dir().glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.add(json.loads(line)["cycle_id"])
    return out


def log(rec: dict[str, Any], snap_dir: Path | None = None) -> Path:
    res = validate(rec, snap_dir)
    if res["errors"]:
        raise ValueError("confluence record invalid:\n  " + "\n  ".join(res["errors"]))
    if rec["cycle_id"] in logged_cycles():
        raise ValueError(f"cycle {rec['cycle_id']} already logged (append-only)")
    p = journal_dir() / f"{str(rec['as_of'])[:7]}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec | {"warnings": res["warnings"]}, default=str) + "\n")
    return p


def rows() -> list[dict[str, Any]]:
    """Long format: one row per (cycle, symbol, layer)."""
    out = []
    for p in sorted(journal_dir().glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for a in r["assets"]:
                for L in LAYERS:
                    v = a["readings"].get(L, {}).get("value")
                    out.append({"cycle_id": r["cycle_id"], "as_of": r["as_of"], "symbol": a["symbol"], "layer": L, "value": v,
                                "model": r["model"], "prompt_version": r["prompt_version"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("validate", "log", "status"))
    ap.add_argument("file", nargs="?")
    ap.add_argument("--snapshot", default="")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        n = len(logged_cycles())
        print(f"{n} cycles logged in {journal_dir()}; the fit needs >= {MIN_CYCLES_TO_FIT}")
        return 0
    rec = json.loads(Path(a.file).read_text(encoding="utf-8"))
    snap = Path(a.snapshot) if a.snapshot else None
    if a.cmd == "validate":
        res = validate(rec, snap)
        print(json.dumps(res, indent=1))
        return 1 if res["errors"] else 0
    p = log(rec, snap)
    print(f"logged {rec['cycle_id']} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
