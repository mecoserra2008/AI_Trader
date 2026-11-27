"""Forward scorer of the D1 stop-sigma shadow (ai_trader.risk.sigma_shadow): QLIKE of every spec against the bars' 5m
realised variance once they are known, pooled DM-HLN tests, Student-t 1% VaR exceedances (all bars and weekend bars).

    PYTHONIOENCODING=utf-8 uv run python scripts/sigma_shadow_score.py            # print the table, write score_summary.json
    ... --gate      also take the one-time switch decision when risk_sigma.switch_on has passed (H-SIGMA-1 rule; latched,
                    appended to data/journal/sigma_shadow/gate.jsonl; the shadow loop does this every hour)
    ... --record    record H-SIGMA-1..3 verdicts once their registered minimum samples are met; run it under the results
                    lock: .venv/Scripts/python.exe scripts/ops/governor.py lock results -- uv run python scripts/sigma_shadow_score.py --record
    ... --json      print the summary JSON instead of the table
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")

from ai_trader.risk import sigma_shadow as sh  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--as-of", default=None, help="score as of this time (ISO, UTC); default now")
    a = ap.parse_args(argv)
    s = sh.score_all(now=a.as_of, gate=a.gate)
    print(json.dumps(s, indent=1, default=str) if a.json else sh.table(s))
    if a.record:
        rows = sh.record(s)
        print(f"recorded {len(rows)} rows: {[r['id'] + ' ' + r['verdict'] for r in rows]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
