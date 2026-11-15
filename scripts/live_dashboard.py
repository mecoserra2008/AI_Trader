"""Local live dashboard of the frozen BTC networks (research only; read-only, 127.0.0.1 only, never trades).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_dashboard.py [--port 8765] [--no-preload]

Then open http://127.0.0.1:8765/ . The page streams BTCUSDT from Bybit's public websocket in the browser and polls
/api/state every 30 s. The server reads data/journal/live_nn (and live_nn_v2 if present), data/journal/live_claude,
research/live/LATEST.md and the frozen models data/models/pnn; it writes nothing. Threads: torch 1, one request at a
time. Stop with Ctrl+C.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.livedash.server import HOST, PORT, App, make_server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Local read-only live dashboard (127.0.0.1).")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--no-preload", action="store_true", help="load the networks on the first activation request")
    a = ap.parse_args(argv)
    app = App(a.root)
    if not a.no_preload:
        v1 = app.store.v1()
        if v1:
            app.activations(v1[-1]["id"])
            app.edges("1h")
            app.edges("24h")
    srv = make_server(app, HOST, a.port)
    print(f"live dashboard on http://{HOST}:{a.port}/  (read-only; Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
