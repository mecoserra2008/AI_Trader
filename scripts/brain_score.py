"""Score the H-BR1 checkpoints (forward only; research only).

    PYTHONIOENCODING=utf-8 uv run python scripts/brain_score.py [--record]

Reads data/journal/brain/checkpoints.jsonl (nowcast and frozen remaining-hour quantiles at minutes 15 / 30 / 45, logged
live by the brain recorder) and the v1 journal (data/journal/live_nn: the line for hour S+1h stores the close at S+1h).
Appends newly resolved checkpoints to data/journal/brain/scores.jsonl (once per id) and prints the running summary:
paired pinball difference nowcast - frozen per (hour, checkpoint), Newey-West t on the daily means.

A verdict exists only once >= 300 distinct hours are resolved (brain.SCORE_MIN_HOURS); with --record it is then written
to research/hypotheses_results.jsonl via ai_trader.research.hypotheses.record (segment clean_holdout: every checkpoint is
logged after the model's knowledge cutoff), at most once per day. Below 300 hours nothing is recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.livedash import brain as B


def v1_lines(root: Path) -> list[dict]:
    rows = []
    for f in sorted((root / "data" / "journal" / "live_nn").glob("20*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted(rows, key=lambda r: (r["forecast_hour"], r["logged_at"]))


def run(root: Path, record: bool = False, min_hours: int = B.SCORE_MIN_HOURS) -> dict:
    cps = B.read_checkpoints(root)
    rows = B.score_checkpoints(cps, B.hour_closes(v1_lines(root)))
    sp = B.brain_dir(root) / "scores.jsonl"
    done = {json.loads(x)["id"] for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()} if sp.is_file() else set()
    new = [r for r in rows if r["id"] not in done]
    if new:
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("a", encoding="utf-8") as fh:
            for r in new:
                fh.write(json.dumps({"scored_at": datetime.now(UTC).isoformat(), **r}) + "\n")
    summ = B.hbr1_summary(rows, min_hours)
    summ.update({"n_checkpoints_logged": len(cps), "new_scored": len(new)})
    if record and summ["verdict"] is not None:
        from ai_trader.research.hypotheses import record as rec
        from ai_trader.research.hypotheses import results

        today = datetime.now(UTC).date().isoformat()
        if not any(r["ts"][:10] == today for r in results("H-BR1")):
            metrics = {k: summ[k] for k in ("n_hours", "n_rows", "n_days", "mean_d", "nw_t_daily", "mean_pinball_nowcast", "mean_pinball_frozen", "by_minute")}
            rec("H-BR1", "clean_holdout", summ["verdict"], metrics,
                note="paired pinball (nowcast - frozen x sqrt(1-f)) per (hour, checkpoint); NW t on daily means; recipe brain_recipe.v1",
                data="data/journal/brain/checkpoints.jsonl + data/journal/live_nn closes; scripts/brain_score.py")
            summ["recorded"] = True
    return summ


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score the H-BR1 brain checkpoints (verdict only at >= 300 hours).")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--record", action="store_true", help="record the verdict in research/hypotheses_results.jsonl once >= 300 hours")
    a = ap.parse_args(argv)
    print(json.dumps(run(Path(a.root), a.record), indent=2, default=str))


if __name__ == "__main__":
    main()
