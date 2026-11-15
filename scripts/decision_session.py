"""Phase-4 decision session CLI (paper only): prepare a packet, record Claude's decisions, report the scorecard.

    PYTHONIOENCODING=utf-8 uv run python scripts/decision_session.py prepare [--as-of ISO] [--symbols ...] [--pipeline-test]
    PYTHONIOENCODING=utf-8 uv run python scripts/decision_session.py decide --session S20261114-1200 --file decisions.json
    PYTHONIOENCODING=utf-8 uv run python scripts/decision_session.py decide --session S... --take shk-SOLU="reason" --skip trd-BTCU="reason" [--close ID="reason"]
    PYTHONIOENCODING=utf-8 uv run python scripts/decision_session.py status [--record]

Sessions run at 00/04/08/12/16/20 UTC (the main session schedules them; nothing here loops). Decisions are refused after
as_of + 30 min. `--pipeline-test` (prepare) marks a replay session that uses the separate `pipeline_test` paper book and is
excluded from the scorecard; only such sessions may be decided with `--simulate-at`. See src/ai_trader/agent/decision.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_trader.agent.decision import RefusedError, decide, load_decisions, prepare, status
from ai_trader.core.time import utc


def _pairs(xs: list[str] | None, flag: str) -> list[tuple[str, str]]:
    out = []
    for x in xs or []:
        if "=" not in x:
            raise SystemExit(f"{flag} expects ID=\"reason\", got {x!r}")
        k, v = x.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build data/sessions/<id>/PACKET.md as of now (or --as-of)")
    p.add_argument("--as-of", default=None, help="ISO time (default now); a real session must be decidable within 30 min")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--pipeline-test", action="store_true", help="replay / test session: separate book, excluded from the scorecard")
    d = sub.add_parser("decide", help="validate and execute Claude's decisions for one session")
    d.add_argument("--session", required=True)
    d.add_argument("--file", type=Path, default=None, help="decisions.json")
    d.add_argument("--take", nargs="*", default=None, metavar='ID="reason"')
    d.add_argument("--skip", nargs="*", default=None, metavar='ID="reason"')
    d.add_argument("--close", nargs="*", default=None, metavar='POSITION="reason"')
    d.add_argument("--simulate-at", default=None, help="pipeline_test sessions only: the simulated decision time")
    s = sub.add_parser("status", help="book, scorecard, H-AI1/H-AI2 running stats, stop posterior")
    s.add_argument("--record", action="store_true", help="record verdicts once the registered minimum samples are reached")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            out = prepare(utc(a.as_of) if a.as_of else None, symbols=a.symbols, pipeline_test=a.pipeline_test)
            print((out / "PACKET.md").read_text(encoding="utf-8"))
            print(f"-> {out}")
        elif a.cmd == "decide":
            if a.file:
                doc, raw = load_decisions(a.file)
            else:
                doc = {"decisions": [{"candidate_id": k, "action": "take", "reason": v} for k, v in _pairs(a.take, "--take")]
                       + [{"candidate_id": k, "action": "skip", "reason": v} for k, v in _pairs(a.skip, "--skip")],
                       "close": [{"position_id": k, "reason": v} for k, v in _pairs(a.close, "--close")]}
                raw = json.dumps(doc)
            r = decide(a.session, doc, simulate_at=utc(a.simulate_at) if a.simulate_at else None, raw=raw)
            print(json.dumps(r, indent=1))
        else:
            print(status(record=a.record))
    except RefusedError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
