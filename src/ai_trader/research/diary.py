"""Research & trading diary: Claude's own log of observations, hypotheses, tests, decisions,
trades and reflections. Human-readable Markdown per day (research/diary/YYYY-MM-DD.md) plus a
machine-readable index (research/diary/index.jsonl). Append-only.

From the shell:
    uv run python -m ai_trader.research.diary add --kind observation --title "..." --body "..." \
        [--tags weekend,btc] [--hyp H-W2] [--link reports/x.md]
    uv run python -m ai_trader.research.diary tail [-n 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ai_trader.core.config import load
from ai_trader.core.time import now

KINDS = ("observation", "hypothesis", "test", "result", "decision", "trade", "reflection", "mistake", "question")


def _dir():
    d = load().root / "research" / "diary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def entry(kind: str, title: str, body: str, *, tags: Sequence[str] = (), hypotheses: Sequence[str] = (), links: Sequence[str] = ()) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ts = now()
    row = {"ts": ts.isoformat(timespec="seconds"), "kind": kind, "title": title, "tags": list(tags), "hypotheses": list(hypotheses), "links": list(links), "body": body}
    d = _dir()
    md = d / f"{ts:%Y-%m-%d}.md"
    new = not md.exists()
    with md.open("a", encoding="utf-8") as f:
        if new:
            f.write(f"# Diary {ts:%Y-%m-%d}\n\n")
        meta = " · ".join(x for x in (", ".join(tags), ", ".join(hypotheses)) if x)
        f.write(f"## {ts:%H:%M} UTC [{kind}] {title}\n")
        if meta:
            f.write(f"_{meta}_\n")
        f.write(f"\n{body.strip()}\n")
        for link in links:
            f.write(f"\n- {link}")
        f.write("\n\n")
    with (d / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def tail(n: int = 10) -> list[dict]:
    p = _dir() / "index.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:]]


def _cli(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="diary")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--kind", required=True, choices=KINDS)
    a.add_argument("--title", required=True)
    a.add_argument("--body", required=True)
    a.add_argument("--tags", default="")
    a.add_argument("--hyp", default="")
    a.add_argument("--link", action="append", default=[])
    t = sub.add_parser("tail")
    t.add_argument("-n", type=int, default=10)
    args = ap.parse_args(argv)
    if args.cmd == "add":
        split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
        entry(args.kind, args.title, args.body, tags=split(args.tags), hypotheses=split(args.hyp), links=args.link)
    else:
        for r in tail(args.n):
            sys.stdout.write(f"{r['ts']} [{r['kind']}] {r['title']}\n")


if __name__ == "__main__":
    _cli()
