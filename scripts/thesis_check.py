"""Desk cycle step: evaluate the thesis ledger at the snapshot's as_of (run by the orchestrator after desk_snapshot).

    PYTHONIOENCODING=utf-8 uv run python scripts/thesis_check.py [--snapshot research/desk/2026-11-15/1723] [--no-refresh] [--dry-run]

Reads research/desk/theses.yaml, refreshes the FRED store if older than 6 h (data/research/macro), evaluates every
proxy / trigger / invalidation point in time (known_ts <= as_of), updates the running state in
data/journal/theses/state.json (+ history.jsonl), and writes <snapshot>/theses.json and <snapshot>/theses.md (<= 30
lines). Never touches the desk snapshot code or files other than these two. Exit code 1 if the ledger is invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ai_trader.core.config import load
from ai_trader.narrative import macro, theses


def resolve(snapshot: str) -> tuple[Path, datetime]:
    root = load().root
    if snapshot:
        snap = (root / snapshot) if not Path(snapshot).is_absolute() else Path(snapshot)
    else:
        latest = json.loads((root / "research" / "desk" / "LATEST.json").read_text(encoding="utf-8"))
        snap = root / latest["snapshot"]
    idx = snap / "index.json"
    as_of = None
    if idx.is_file():
        as_of = json.loads(idx.read_text(encoding="utf-8")).get("as_of")
    if not as_of:
        latest = json.loads((root / "research" / "desk" / "LATEST.json").read_text(encoding="utf-8"))
        as_of = latest["as_of"]
    return snap, datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).astimezone(UTC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the digest, write nothing")
    ap.add_argument("--as-of", default="", help="override as_of (dry runs only)")
    a = ap.parse_args()
    if a.as_of and not a.dry_run:
        print("--as-of is for --dry-run only: a written check must use the snapshot's as_of", file=sys.stderr)
        return 2
    errs = theses.validate(theses.load_ledger())
    if errs:
        print("theses.yaml invalid:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 1
    snap, as_of = resolve(a.snapshot)
    if a.as_of:
        as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC)
    if not a.no_refresh:
        try:
            macro.refresh(max_age_h=6.0, log=lambda m: None)
        except RuntimeError as e:   # stale data is a data gap, not a crash
            print(f"FRED refresh failed (using stored data): {e}", file=sys.stderr)
    res = theses.check(as_of, state={} if a.dry_run else theses.load_state())
    res["snapshot"] = str(snap.relative_to(load().root)).replace("\\", "/")
    md = theses.digest(res)
    if a.dry_run:
        print(md)
        return 0
    (snap / "theses.json").write_text(json.dumps(theses.public(res), indent=1, default=str), encoding="utf-8")
    (snap / "theses.md").write_text(md, encoding="utf-8")
    theses.save_state(res)
    print(md)
    print(f"wrote {snap / 'theses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
