"""Headless brain recorder (research only; never trades): snapshots + H-BR1 checkpoints without the dashboard.

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/brain_recorder.py [--once] [--export]

Once a minute (a few seconds after the boundary) it pulls Bybit public REST 1m klines (BTCUSDT linear), rebuilds the
intra-hour nowcast of the newest v1 line (ai_trader.livedash.brain, recipe brain_recipe.v1) and appends a snapshot to
data/journal/brain/<date>.jsonl; at minutes 15 / 30 / 45 it logs the H-BR1 checkpoint (refused when late). The dashboard
server runs the same loop in a thread once /api/brain/* is first requested; a heartbeat lock
(data/journal/brain/recorder.lock) makes sure only one of them records. One thread, no torch at run time
(--export uses torch once to write data/models/pnn/brain_1h.json). Stop with Ctrl+C.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.livedash import brain as B
from ai_trader.livedash.server import bybit_fetch
from ai_trader.livedash.store import Store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Headless brain recorder (snapshots + H-BR1 checkpoints).")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--once", action="store_true", help="one step now (if the lock is free) and exit")
    ap.add_argument("--export", action="store_true", help="(re)write data/models/pnn/brain_1h.json and exit")
    a = ap.parse_args(argv)
    store = Store(a.root)
    if a.export:
        print(B.write_net(store.pnn_dir))
        return
    rec = B.Recorder(a.root, store, store.pnn_dir, bybit_fetch)
    owner = f"script-{os.getpid()}"
    if a.once:
        print(json.dumps(rec.step() if B.acquire_lock(a.root, owner) else {"status": "lock held by another recorder"}, default=str))
        return
    print(f"brain recorder {owner}: data/journal/brain (Ctrl+C to stop)", flush=True)
    stop = threading.Event()
    try:
        B.run_loop(rec, owner, stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
