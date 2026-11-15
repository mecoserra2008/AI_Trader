"""Phase 2 command line: ``python -m vecopen phase2`` (the reproduce command of ``reports/phase_2.md``).

Steps: (1) smoke run of ``LevelRegistry`` over one asset's research bars (default BTCUSDT): base 5m bars plus the
closed bars of every primary timeframe from phase 1's ``resample`` with the asset's session, merged into the as-of
feed order by ``levels.bar_stream`` and fed one bar at a time through ``LevelRegistry.update``; (2) the vecopen tests;
(3) ``reports/phase_2_tables.md``; (4) one registry line (phase 2, kind ``build``). It reads ``data/research`` only.

Runtime is wall-clock (``time.perf_counter``) of the data preparation and of the streaming loop, measured in this
process before the tests run; peak RAM is the process's peak working set (Windows) or max RSS (POSIX).
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from vecopen import registry
from vecopen.config import PROJECT_ROOT, load_config

TABLES_PATH = PROJECT_ROOT / "reports" / "phase_2_tables.md"


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float("nan")
        return pmc.PeakWorkingSetSize / 2**20
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, UTC).year


def smoke(asset: str) -> dict[str, Any]:
    """Stream ``asset``'s research bars through the registry; return counts, ages, timings and memory."""
    from vecopen.bars import load_sessions, resample
    from vecopen.data import load_bars
    from vecopen.levels import LevelRegistry, bar_stream

    cfg = load_config()
    t_start = time.perf_counter()
    df = load_bars(asset)          # validated full load, then only the columns the level layer reads
    df = df.select([c for c in ("ts_close", "o", "h", "l", "c", "v") if c in df.columns])
    session = load_sessions().for_asset(asset)
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    higher = {tf: resample(df, tf, session) for tf in tfs}
    t_prep = time.perf_counter()

    reg = LevelRegistry(cfg, session=session)
    births: dict[str, Counter] = defaultdict(Counter)
    fills: Counter = Counter()
    flips: Counter = Counter()
    expired: Counter = Counter()
    expired_untouched: Counter = Counter()
    ages: dict[str, list[int]] = defaultdict(list)
    wall_h: dict[str, list[float]] = defaultdict(list)
    confl: dict[str, list[float]] = defaultdict(list)
    leg_k: dict[str, list[int]] = defaultdict(list)
    digest = hashlib.sha256()
    n_bars: Counter = Counter()
    for b in bar_stream(df, higher, reg.base_tf):
        n_bars[b.tf] += 1
        for e in reg.update(b):
            k = e.kind
            if k == "birth":
                births[e.tf][_year(e.ts)] += 1
                leg_k[e.tf].append(e.k_star)
                digest.update(f"B{e.level_id},{e.tf},{e.ts},{e.price!r},{e.zone_lo!r},{e.zone_hi!r}\n".encode())
            elif k == "first_fill":
                fills[e.tf] += 1
                ages[e.tf].append(e.age)
                wall_h[e.tf].append((e.t0 - e.birth_ts) / 3_600_000)
                confl[e.tf].append(e.confluence)
                digest.update(f"F{e.level_id},{e.t0},{e.confluence!r}\n".encode())
            elif k == "flip":
                flips[e.tf] += 1
            elif k == "expire":
                expired[e.tf] += 1
                if e.state_before == "untouched":
                    expired_untouched[e.tf] += 1
    t_end = time.perf_counter()

    ts = df["ts_close"]
    per_tf = {}
    for tf in tfs:
        nb = sum(births[tf].values())
        live = [x for x in reg.active.values() if x.tf == tf]
        per_tf[tf] = {
            "tau_bars": n_bars[tf],
            "no_trade_bars": reg.n_no_trade[tf],
            "births": nb,
            "births_by_year": {str(y): births[tf][y] for y in sorted(births[tf])},
            "k_star_median": statistics.median(leg_k[tf]) if leg_k[tf] else None,
            "first_fills": fills[tf],
            "fill_fraction": round(fills[tf] / nb, 4) if nb else None,
            "median_age_tau": statistics.median(ages[tf]) if ages[tf] else None,
            "median_age_hours": round(statistics.median(wall_h[tf]), 2) if wall_h[tf] else None,
            "median_confluence": statistics.median(confl[tf]) if confl[tf] else None,
            "flips": flips[tf],
            "expired": expired[tf],
            "expired_untouched": expired_untouched[tf],
            "live_untouched_at_end": sum(1 for x in live if x.state == "untouched"),
            "live_reference_at_end": sum(1 for x in live if x.state == "reference"),
        }
    return {
        "asset": asset,
        "session": session.name,
        "base_bars": n_bars[reg.base_tf],
        "base_no_trade_bars": reg.n_no_trade[reg.base_tf],
        "bars_fed": sum(n_bars.values()),
        "first_bar_close": ts.min().isoformat(),
        "last_bar_close": ts.max().isoformat(),
        "timeframes": per_tf,
        "prep_seconds": round(t_prep - t_start, 2),
        "loop_seconds": round(t_end - t_prep, 2),
        "total_seconds": round(t_end - t_start, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "events_sha256": digest.hexdigest(),
    }


def tables(s: dict[str, Any]) -> str:
    tfs = list(s["timeframes"])
    years = sorted({y for tf in tfs for y in s["timeframes"][tf]["births_by_year"]})
    L = [f"# Phase 2 smoke-run tables ({s['asset']}, research sample)", "",
         "Generated by `uv run --extra vecopen python -m vecopen phase2`. Do not edit by hand.", "",
         f"Asset {s['asset']}, session `{s['session']}`, base bars {s['base_bars']:,} (5m, closes "
         f"{s['first_bar_close']} .. {s['last_bar_close']}), bars fed {s['bars_fed']:,}. "
         f"No-trade 5m bars skipped: {s['base_no_trade_bars']}. Events digest `{s['events_sha256'][:16]}…`.", "",
         "## Levels born per timeframe per year (by birth close, UTC)", "",
         "| tf | " + " | ".join(str(y) for y in years) + " | total |", "|---|" + "---|" * (len(years) + 1)]
    for tf in tfs:
        by = s["timeframes"][tf]["births_by_year"]
        L.append(f"| {tf} | " + " | ".join(f"{by.get(str(y), 0):,}" for y in years)
                 + f" | {s['timeframes'][tf]['births']:,} |")
    L += ["", f"{years[-1]} is partial: the research sample ends at {s['last_bar_close'][:10]}." if years else "", "",
          "## First fills, age at first fill and lifecycle per timeframe", "",
          "| tf | tau-bars | no-trade tau-bars | births | median k* | first fills | filled / born | median age at first fill "
          "(tau-bars) | median age (hours) | median confluence | flips | expired (of which untouched) | live at end "
          "(untouched / reference) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tf in tfs:
        r = s["timeframes"][tf]
        L.append(f"| {tf} | {r['tau_bars']:,} | {r['no_trade_bars']} | {r['births']:,} | {r['k_star_median']} | "
                 f"{r['first_fills']:,} | {r['fill_fraction']} | {r['median_age_tau']} | {r['median_age_hours']} | "
                 f"{r['median_confluence']} | {r['flips']:,} | {r['expired']:,} ({r['expired_untouched']:,}) | "
                 f"{r['live_untouched_at_end']} / {r['live_reference_at_end']} |")
    L += ["", "## Runtime and memory", "",
          "| step | value |", "|---|---|",
          f"| load_bars + resample to {len(tfs)} timeframes | {s['prep_seconds']} s |",
          f"| streaming loop ({s['bars_fed']:,} update calls) | {s['loop_seconds']} s |",
          f"| total | {s['total_seconds']} s |",
          f"| peak RAM (process peak working set) | {s['peak_rss_mb']} MB |",
          "", "Threads: POLARS_MAX_THREADS=2, OMP_NUM_THREADS=2 (set by `vecopen/__init__`); the loop is single-threaded.", ""]
    return "\n".join(L)


def cmd_phase2(args: argparse.Namespace) -> int:
    from vecopen.__main__ import run_tests

    cfg = load_config()
    s = smoke(args.asset)
    md = tables(s)
    TABLES_PATH.write_text(md, encoding="utf-8", newline="\n")
    print(md)
    counts = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "pytest_exit_code": 0}
    if not args.skip_tests:
        counts = run_tests(PROJECT_ROOT / "tests")
    print(f"tests: {counts}")
    green = counts["pytest_exit_code"] == 0 and counts["failed"] == 0 and counts["errors"] == 0
    tables_sha = hashlib.sha256(md.encode("utf-8")).hexdigest()
    metrics = {"tests": counts, "smoke": s, "tables_sha256": tables_sha}
    if args.dry_run:
        print("dry run: registry not written")
    else:
        rec = registry.append(2, metrics, kind="build", config=cfg,
                              note=f"phase 2 levels: LegOriginDetector + LevelRegistry (streaming); smoke run {args.asset}")
        print(f"registry line {rec['seq'] + 1} appended (kind build): line_sha256 {rec['line_sha256']} "
              f"git_commit {rec['git_commit']} diff_sha256 {rec['diff_sha256']}")
    return 0 if green else 1


def add_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("phase2", help="phase 2 reproduce command: smoke run, tests, tables, registry line")
    p.add_argument("--asset", default="BTCUSDT")
    p.add_argument("--dry-run", action="store_true", help="do not append a registry line")
    p.add_argument("--skip-tests", action="store_true", help="smoke run and tables only (implies a dry run is wise)")
    p.set_defaults(fn=cmd_phase2)
