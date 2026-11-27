"""Hourly live line of the frozen BTC networks (research only; see src/ai_trader/live/nn.py for the full contract).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_nn.py [--as-of 2026-11-13T22:03Z] [--dry-run] [--force] [--no-v2]
    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_nn.py --v2-replay-from 2026-11-13T20:00Z   (one-off v2 replay)

Run at hh:03 UTC. v1 (live_nn.v1, unchanged): appends one line to data/journal/live_nn/<date>.jsonl (skipped if the
forecast bar already has a line, unless --force), scores closed windows once into data/journal/live_nn/scores.jsonl and
updates the online ACI state data/models/pnn/live_aci_state.json. v1 is computed and written first; then the v2 shadow
(live_nn.v2, src/ai_trader/live/nn_v2.py, config/live_nn_v2.yaml) writes data/journal/live_nn_v2/ and
data/models/pnn/live_aci_state_v2.json, and the regime-and-players block is computed (information only). Last, the
cross-sectional v3 context line (xs.v3, src/ai_trader/live/xs_v3.py; context only, not a signal) writes
data/journal/live_xs_v3/ and data/models/xs_v3/live/aci_state.json (--no-v3 skips it; v1 / v2.1 are identical either way).
research/live/LATEST.md is overwritten with all four. --dry-run writes nothing and prints. Threads: torch 1, polars 2.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:  # already set
    pass

from ai_trader.live import hourly, nn, nn_v2  # noqa: E402
from ai_trader.risk.sigma import pin_process_base  # noqa: E402

# The frozen networks (v1, v2, xs_v3) were trained on robust_bc units: frames in this process never follow the D1
# stop-sigma switch (risk.sigma; their modules are code-hash guarded, so the pin lives here). No effect before a switch.
pin_process_base()


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="append a rerun line even if the bar already has one")
    ap.add_argument("--no-field", action="store_true")
    ap.add_argument("--no-v2", action="store_true", help="v1 only (v1 output is identical either way)")
    ap.add_argument("--no-v3", action="store_true", help="skip the cross-sectional v3 context line (v1 / v2.1 identical either way)")
    ap.add_argument("--v2-replay-from", default="", help="one-off: replay v2 from this bar to the newest closed hour, then exit")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.v2_replay_from:
        lines = hourly.replay_v2(_ts(a.v2_replay_from), paths2=nn_v2.PathsV2.default())
        sys.stdout.write(json.dumps({"replayed": len(lines), "runtime_s": round(time.time() - t0, 1)}) + "\n")
        return {"replayed": lines}
    as_of = _ts(a.as_of) if a.as_of else datetime.now(UTC)
    res = hourly.run_hour(as_of, paths1=nn.Paths.default(), dry_run=a.dry_run, force=a.force, with_field=not a.no_field, v2=not a.no_v2,
                          v2_require_state=True, v3=not a.no_v3, t0=t0)
    line = res["line"]
    sys.stdout.write(res["latest"])
    info = {"id": line["id"], "flags": line["flags"], "runtime_s_v1": line["runtime_s"], "runtime_s": res["runtime_s"], "v1": res.get("v1"),
            "v2": res.get("v2"), "v2_id": res.get("line_v2", {}).get("id"), "v2_error": res.get("v2_error"),
            "v3": res.get("v3"), "v3_id": res.get("line_v3", {}).get("id"), "v3_runtime_s": res.get("line_v3", {}).get("runtime_s"),
            "v3_error": res.get("v3_error")}
    sys.stdout.write(json.dumps(info, default=str) + "\n")
    return line


if __name__ == "__main__":
    main()
