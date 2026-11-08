"""Hypothesis registry access. Definitions live in research/hypotheses.yaml (pre-registered);
results are appended to research/hypotheses_results.jsonl and never edited."""

from __future__ import annotations

import json
from typing import Any

from ai_trader.core.config import load
from ai_trader.core.time import now

SEGMENTS = ("llm_in_sample", "clean_holdout", "pooled")
VERDICTS = ("supports", "contradicts", "inconclusive")


def registry() -> dict[str, dict[str, Any]]:
    return load().hypotheses.get("hypotheses", {})


def record(hid: str, segment: str, verdict: str, metrics: dict[str, Any], note: str = "", data: str = "") -> dict[str, Any]:
    if hid not in registry():
        raise KeyError(f"{hid} is not pre-registered in research/hypotheses.yaml")
    if segment not in SEGMENTS or verdict not in VERDICTS:
        raise ValueError(f"segment must be one of {SEGMENTS}, verdict one of {VERDICTS}")
    row = {"ts": now().isoformat(), "id": hid, "segment": segment, "verdict": verdict, "metrics": metrics, "data": data, "note": note}
    path = load().root / "research" / "hypotheses_results.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")
    return row


def results(hid: str | None = None) -> list[dict[str, Any]]:
    path = load().root / "research" / "hypotheses_results.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if hid is None or r["id"] == hid]
