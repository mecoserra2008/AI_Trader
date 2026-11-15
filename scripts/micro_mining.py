"""H-MINE1 runner (agent E). Stages:

    PYTHONIOENCODING=utf-8 uv run python scripts/micro_mining.py discover      # 16 discovery days -> discovery.json (+ freeze)
    PYTHONIOENCODING=utf-8 uv run python scripts/micro_mining.py status        # which confirmation days are ready
    PYTHONIOENCODING=utf-8 uv run python scripts/micro_mining.py confirm [--min-days 10]
    PYTHONIOENCODING=utf-8 uv run python scripts/micro_mining.py report

Day tables are cached (scratchpad agentE/cache) so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.research import micro_mining as mm  # noqa: E402

OUT = mm.MINING_DIR


def _dump(path: Path, obj) -> None:
    from ai_trader.research.micro_days import jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def discover(freeze: bool) -> None:
    t1, t5, infos = mm.load_days(mm.DISCOVERY_DAYS, workers=4)
    print(f"T1 rows {t1.height:,}  T5 rows {t5.height:,}")
    disc = mm.discovery(t1, t5)
    disc["inclusion"] = infos
    _dump(OUT / "discovery.json", disc)
    rej = [r for r in disc["effects"] if r["bh_reject"]]
    print(f"family {disc['family']}; testable {sum(r['testable'] for r in disc['effects'])}; BH discoveries {len(rej)} "
          f"(vol {sum(r['kind'] == 'vol' for r in rej)}, direction {sum(r['kind'] == 'direction' for r in rej)})")
    if freeze:
        fz = mm.freeze(disc)
        print(f"frozen {fz['n_discoveries']} discoveries at {fz['frozen_at']} -> {mm.FROZEN}")


def status() -> list[str]:
    ready = [d for d in mm.CONFIRMATION_DAYS if mm.day_ready(d)]
    print(f"confirmation days ready {len(ready)}/15: {ready}")
    return ready


def confirm(min_days: int) -> None:
    mm.require_frozen()
    ready = status()
    if len(ready) < min_days:
        print(f"only {len(ready)} < {min_days} confirmation days ready; not running")
        return
    t1, t5, infos = mm.load_days(ready, workers=4)
    conf = mm.confirmation(t1, t5, ready)
    conf["inclusion"] = infos
    _dump(OUT / "confirmation.json", conf)
    print(json.dumps(conf["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["discover", "status", "confirm"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--min-days", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "discover":
        discover(a.freeze)
    elif a.stage == "status":
        status()
    else:
        confirm(a.min_days)
