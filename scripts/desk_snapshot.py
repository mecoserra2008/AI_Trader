"""Desk snapshot: deterministic, point-in-time inputs for the LLM analyst agents (research/desk/CONTRACT.md).

    PYTHONIOENCODING=utf-8 uv run python scripts/desk_snapshot.py [--watch SOLUSDT ZECUSDT] [--n-xs 60] [--n-geo 20] [--n-tape 10]
        [--out-dir DIR] [--print xs,geometry,tape]

Writes research/desk/<UTC date>/<HHMM>/{xs,geometry,news,sentiment,tape,account}.json, a markdown digest per domain
(<= 60 lines each), index.json (timings, errors, data cut) and research/desk/LATEST.json. Public REST (Bybit
/v5/market, Deribit /public) and local data only; no trading endpoint is reachable from this code. Budget: < 90 s,
<= 2 threads, <= 1 GB. The sentiment / news domains read the stores filled by scripts/desk_collect.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.desk import snapshot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", nargs="*", default=[])
    ap.add_argument("--n-xs", type=int, default=snapshot.N_XS)
    ap.add_argument("--n-geo", type=int, default=snapshot.N_GEO)
    ap.add_argument("--n-tape", type=int, default=snapshot.N_TAPE)
    ap.add_argument("--out-dir", default=None, help="test runs: write here instead of research/desk")
    ap.add_argument("--print", default="", help="comma-separated digests to print, e.g. xs,geometry,tape")
    a = ap.parse_args()
    r = snapshot.run(out_root=Path(a.out_dir) if a.out_dir else None, watch=a.watch, n_xs=a.n_xs, n_geo=a.n_geo, n_tape=a.n_tape)
    for k in [x for x in a.print.split(",") if x]:
        print("\n".join(r["digests"][k]) + "\n")
    if r["index"]["errors"]:
        print(f"{len(r['index']['errors'])} errors (see index.json):", *r["index"]["errors"][:5], sep="\n  ")


if __name__ == "__main__":
    main()
