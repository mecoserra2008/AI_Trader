"""Live entry gates (D2 release-hour rule, D3 jump-intensity gate) -> data/live/gates.json for `atx live`.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/live_gates.py --once          one write, print the summary
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/live_gates.py --loop          every 60 s on the wall clock
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/live_gates.py --packet-line   the packet line from the file

atx live refuses every new entry while the file is missing or older than 5 minutes ("gates stale"), so keep --loop
running (scripts/ops/desk_start.sh starts it). Model, threshold and fail-safe rules: src/ai_trader/live/gates.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "2")


def main() -> int:
    from ai_trader.live import gates

    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--packet-line", action="store_true")
    ap.add_argument("--symbols", nargs="*", help="override the modelled symbols (default: WS05's 23-symbol pool)")
    a = ap.parse_args()
    if a.packet_line:
        print(gates.packet_line())
        return 0
    if a.loop:
        gates.loop(a.interval, symbols=a.symbols, log=lambda s: print(s, flush=True))
        return 0
    doc = gates.run_once(symbols=a.symbols)
    short = {k: doc[k] for k in ("written_utc", "version", "calendar", "errors")}
    short["release_windows"] = [f"{w['event']} {w['release_utc']}{' ACTIVE' if w['active'] else ''}" for w in doc["release_windows"]]
    short["jump"] = {s: {k: j.get(k) for k in ("ratio", "elevated", "until_utc", "p_jump_1h", "source", "reason")} for s, j in doc["jump"].items()}
    print(json.dumps(short, indent=1))
    print(gates.packet_line(gates=doc))
    return 0 if not doc["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
