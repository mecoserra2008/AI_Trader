"""Run every forward-only hypothesis evaluator; print progress; write status.json and the Markdown summary.

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/forward_tests.py [--record] [--only H-LEV6,H-OF6] [--as-of ISO] [--workers 1]

Without --record nothing is written to research/hypotheses_results.jsonl. With --record an evaluator records only once
its registered sample size is met (and never twice); below it the run says 'refused'.
Outputs: data/research/forward_tests/status.json, research/reports/forward_tests_status.md.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from ai_trader.research import forward_tests as ft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--as-of", default="")
    ap.add_argument("--workers", type=int, default=ft.WORKERS)
    a = ap.parse_args()
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC) if a.as_of else None
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None
    status = ft.run(as_of, record=a.record, only=only, workers=a.workers)
    if only:  # partial run: merge into the last full status so the summary stays complete
        import json

        p = ft.out_dir() / "status.json"
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8"))
            keep = [r for r in old.get("tests", []) if r["hypothesis"] not in only]
            status["tests"] = sorted(keep + status["tests"], key=lambda r: list(ft.EVALUATORS).index(r["hypothesis"]))
    j, md = ft.write_outputs(status)
    print(f"-> {j}\n-> {md}")


if __name__ == "__main__":
    main()
