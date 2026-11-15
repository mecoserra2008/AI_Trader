"""Multi-day microstructure batch (overnight program 2026-11-12, agent A).

    PYTHONIOENCODING=utf-8 uv run python scripts/micro_batch.py --batch 2 --days 2025-08-31 2025-11-19 2026-07-22 2026-08-17 2026-10-25

Downloads the batch's public Bybit trade and ob200 archives to data/raw/batch_<n>/ (never the lake), aggregates every
symbol-day into data/research/micro_days/<date>/ (ai_trader.research.micro_days), verifies, deletes the raw files
(unless --keep-raw), logs storage to logs/micro_batch_<n>.log, then reruns the multi-day H-MD study on every verified
day so far (ai_trader.research.multiday_micro_study; --no-study to skip). The 2026-11-10 pilot day is built first if
it is not yet aggregated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--symbols", default="", help="comma list (default: config/mm_multi.yaml + BTCUSDT)")
    ap.add_argument("--no-pilot", action="store_true")
    ap.add_argument("--no-study", action="store_true")
    ap.add_argument("--no-diary", action="store_true")
    ap.add_argument("--no-books", action="store_true", help="trades only (no ob200); backfill later with --books-only (default off)")
    ap.add_argument("--books-only", action="store_true", help="backfill ob200 books of days processed with --no-books, then rerun the study")
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="RAM guard: no new task below this free memory")
    ap.add_argument("--max-heavy", type=int, default=1, help="max concurrent heavy tasks (ob200 parse, BTC / ETH, archives > 80 MB)")
    ap.add_argument("--no-resume", action="store_true", help="recompute symbol-days whose outputs already exist (default: resume)")
    a = ap.parse_args(argv)
    from ai_trader.research import diary, micro_days

    syms = [s for s in a.symbols.split(",") if s] or None
    if not a.no_diary and not a.books_only:
        diary.entry("test", f"Multi-day microstructure batch {a.batch} started",
                    f"Days {' '.join(a.days)} (random draw, seed 20261112). Trades: config/mm_multi.yaml + BTCUSDT from public.bybit.com; "
                    f"ob200 for the 12 pilot L2 symbols. Raw to data/raw/batch_{a.batch}/, aggregates to data/research/micro_days/<date>/, "
                    "raw deleted after verification. Laws refitted per symbol-day with the pilot code; H-MD1..6 rerun afterwards.",
                    tags=["microstructure", "multi-day", f"batch{a.batch}"], hypotheses=["H-MD1", "H-MD2", "H-MD3", "H-MD4", "H-MD5", "H-MD6"],
                    links=["research/plans/2026-11-12_overnight_loop.md", f"logs/micro_batch_{a.batch}.log"])
    if a.books_only:
        summ = micro_days.run_books_only(a.batch, a.days, keep_raw=a.keep_raw, workers=a.workers)
        if not a.no_diary:
            diary.entry("test", f"Multi-day microstructure batch {a.batch}: ob200 books backfilled",
                        json.dumps({d: {"verified": (v.get("verified") or {}).get("ok"), "books": v.get("books")} for d, v in summ.get("days", {}).items()})[:1500],
                        tags=["microstructure", "multi-day", f"batch{a.batch}"], hypotheses=["H-MD4", "H-MD6"], links=[f"logs/micro_batch_{a.batch}.log"])
        if not a.no_study:
            from ai_trader.research import multiday_micro_study

            multiday_micro_study.main(["--batch", str(a.batch), "--workers", str(a.workers)])
        return 0
    summ = micro_days.run_batch(a.batch, a.days, keep_raw=a.keep_raw, workers=a.workers, symbols=syms, include_pilot=not a.no_pilot, resume=not a.no_resume, no_books=a.no_books,
                                 min_free_gb=a.min_free_gb, max_heavy=a.max_heavy)
    study = None
    if not a.no_study and "error" not in summ:
        from ai_trader.research import multiday_micro_study

        study = multiday_micro_study.main(["--batch", str(a.batch), "--workers", str(a.workers)])
    if not a.no_diary:
        days = summ.get("days", {})
        ok = [d for d, v in days.items() if (v.get("verified") or {}).get("ok")]
        bad = {d: (v.get("verified") or {}).get("problems") for d, v in days.items() if not (v.get("verified") or {}).get("ok")}
        st0, st1 = summ.get("storage_before", {}), summ.get("storage_after", {})
        body = (f"Verified days: {ok}. Problems: {json.dumps(bad)[:800]}. Missing (404) per day: "
                f"{ {d: (v.get('missing_trades'), v.get('missing_books')) for d, v in days.items()} }. "
                f"Storage lake+research {st0.get('lake_plus_research_gb', 0):.2f} -> {st1.get('lake_plus_research_gb', 0):.2f} GB "
                f"(micro_days {st1.get('micro_days_gb', 0):.2f} GB, raw now {st1.get('raw_gb', 0):.2f} GB).")
        if study:
            body += f" Study: {study.get('summary_line', '')}"
        diary.entry("result", f"Multi-day microstructure batch {a.batch} done", body, tags=["microstructure", "multi-day", f"batch{a.batch}"],
                    hypotheses=["H-MD1", "H-MD2", "H-MD3", "H-MD4", "H-MD5", "H-MD6"],
                    links=[f"logs/micro_batch_{a.batch}.log", f"research/reports/multiday_micro_batch{a.batch}.md"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
