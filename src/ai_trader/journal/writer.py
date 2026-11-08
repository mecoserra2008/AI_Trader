"""Append-only decision journal: everything needed to audit and evaluate a session forward.
One JSONL file per (UTC day, event type) under paths.journal. Never edited, only appended."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_trader.core.config import load
from ai_trader.core.time import now

EVENT_TYPES = frozenset({
    "session_start",      # model_id, prompt_version, tool_versions, config hashes
    "snapshot",           # state cards shown to the LLM
    "llm_message",        # raw request / response
    "tool_call",          # name, args, ToolResult
    "view",               # LLM view: symbol, Δ log-odds, confidence, reason codes, evidence
    "decision",           # ranked candidates, takes
    "risk_decision",      # accepted / rejected with reasons
    "order",              # submitted orders (broker, link id, venue id)
    "fill",               # executions
    "position_update",    # stop moves, exits, re-scores
    "baseline",           # RANDOM_N / RULE_N actions on the same candidate list
    "shadow_outcome",     # resolved outcome of an untaken candidate
    "manual_action",      # any operator intervention (contaminates the record; must be logged)
    "session_end",
})


class Journal:
    def __init__(self, session_id: str, root: Path | None = None):
        self.session_id = session_id
        self.root = Path(root) if root else load().path("journal")

    def write(self, event_type: str, payload: dict[str, Any], ts: datetime | None = None) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown journal event type {event_type!r}")
        ts = ts or now()
        d = self.root / ts.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": ts.isoformat(), "session_id": self.session_id, "type": event_type, **payload}, default=str)
        with (d / f"{event_type}.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
