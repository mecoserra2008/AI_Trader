"""Phase 1 command line: ``python -m vecopen data <step>`` and ``python -m vecopen phase1``.

Every data run appends one registry line (kind ``data``) whose metrics carry the sha256 of
``config/sessions.yaml``. ``phase1`` is the single reproduce command of ``reports/phase_1.md``; it never reads
``data/holdout`` (holdout numbers come from ``split.json``, written when the holdout was created).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vecopen import data as D
from vecopen import gaps as G
from vecopen import registry
from vecopen.bars import load_sessions
from vecopen.config import PROJECT_ROOT, load_config

TABLES_PATH = PROJECT_ROOT / "reports" / "phase_1_tables.md"


def _crypto_symbols() -> list[str]:
    return list(load_config().data["assets"]["crypto"]["symbols"])


def _cfd_assets() -> list[str]:
    return list(load_config().data["assets"]["cfd"]["candidates"])


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _append(note: str, metrics: dict[str, Any], dry_run: bool) -> None:
    sess = load_sessions()
    m = {"sessions_sha256": sess.sha256, **metrics}
    if dry_run:
        print(f"dry run: registry not written ({note})")
        return
    rec = registry.append(1, m, kind="data", note=note, config=load_config())
    print(f"registry line {rec['seq'] + 1} appended (kind data): line_sha256 {rec['line_sha256']}")


# ---------------------------------------------------------------- steps

def cmd_crypto_download(args: argparse.Namespace) -> int:
    from vecopen.ingest import Manifest, ingest_binance

    res = ingest_binance(args.symbols or _crypto_symbols(), Manifest())
    _append("phase 1 crypto download (Binance USDT-M 5m archive, sha256-verified, staged)",
            {"step": "crypto-download", **res}, args.dry_run)
    return 0


def cmd_fill_gaps(args: argparse.Namespace) -> int:
    from vecopen.ingest import Manifest, fill_from_daily

    res = fill_from_daily(args.symbols or _crypto_symbols(), Manifest())
    _append("phase 1 archive-gap repair from Binance daily zips (before the split)",
            {"step": "fill-gaps", "per_symbol": res}, args.dry_run)
    return 0


def cmd_build_split(args: argparse.Namespace) -> int:
    from vecopen.split import build_split

    source = args.source
    assets = args.assets or (_crypto_symbols() if source == "binance" else _cfd_assets())
    res = build_split(assets, source=source)
    if res["split"]:
        _append(f"phase 1 build + split ({source}): research/holdout written, holdout locked",
                {"step": "build-split", "source": source, **res, "split_json_sha256": _sha(D.SPLIT_PATH)}, args.dry_run)
    return 0


def cmd_cfd_download(args: argparse.Namespace) -> int:
    from vecopen.ingest import Manifest, ctrader_download, ctrader_keys_present, stage_ctrader

    if not ctrader_keys_present():
        print("cTrader keys are missing in .env (see `target/debug/atx ctrader check`); nothing downloaded")
        return 2
    man = Manifest()
    assets = args.assets or _cfd_assets()
    got = ctrader_download(man, assets)
    staged = {a: stage_ctrader(a, man) for a in got}
    _append("phase 1 CFD download (cTrader m5 trend bars, BlackBull)",
            {"step": "cfd-download", "assets": sorted(got), "months_staged": staged,
             "history": {a: [s.get("earliest"), s.get("latest")] for a, s in got.items()}}, args.dry_run)
    return 0


def research_gap_check() -> dict[str, Any]:
    """Recompute research gap runs AND classes from data/research only and compare with gaps.json."""
    from vecopen.ingest import Manifest
    from vecopen.split import classify_windows, research_window

    if not D.GAPS_PATH.exists():
        return {"assets": 0, "match": True}
    doc = json.loads(D.GAPS_PATH.read_text(encoding="utf-8"))
    sessions, man = load_sessions(), Manifest()
    mismatches = []
    for source in ("binance", "ctrader"):
        assets = [a for a, r in doc["assets"].items() if r.get("source", "binance") == source]
        if not assets:
            continue
        cls = classify_windows({a: research_window(a) for a in assets}, source, sessions, man)
        flat = G.classify_flat({a: research_window(a, flat=True) for a in assets})
        for a in assets:
            for got_df, key in ((cls[a], "runs"), (flat[a], "flat_runs")):
                got = [(r["gap_first"].isoformat(), r["n_missing"], r["class"]) for r in got_df.iter_rows(named=True)]
                want = [(r["gap_first"], r["n_missing"], r["class"]) for r in doc["assets"][a].get(key, [])]
                if got != want:
                    mismatches.append(f"{a}:{key}")
    return {"assets": len(doc["assets"]), "match": not mismatches, "mismatches": mismatches}


def cmd_gaps(_: argparse.Namespace) -> int:
    res = research_gap_check()
    print(res)
    return 0 if res["match"] else 1


def cmd_storage(_: argparse.Namespace) -> int:
    from vecopen.ingest import Manifest, storage_check

    print(storage_check(Manifest(), 0))
    return 0


# ---------------------------------------------------------------- tables

def _d(s: str | None) -> str:
    return "" if not s else datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M")


def build_tables() -> tuple[str, dict[str, Any]]:
    """Markdown tables (history, research gaps, holdout gap counts, notable runs) and summary metrics."""
    split = json.loads(D.SPLIT_PATH.read_text(encoding="utf-8")) if D.SPLIT_PATH.exists() else {"assets": {}}
    gaps = json.loads(D.GAPS_PATH.read_text(encoding="utf-8")) if D.GAPS_PATH.exists() else {"assets": {}}
    man = json.loads(D.MANIFEST_PATH.read_text(encoding="utf-8")) if D.MANIFEST_PATH.exists() else {"files": {}}
    q_by: dict[str, dict[str, int]] = {}
    for f in man.get("files", {}).values():
        q = q_by.setdefault(f["asset"], {"files": 0})
        q["files"] += 1
        for k, v in (f.get("quality") or {}).items():
            q[k] = q.get(k, 0) + int(v)
    L: list[str] = ["<!-- generated by `python -m vecopen phase1`; do not edit -->", "",
                    "### Per-asset history (5m base bars)", "",
                    "| asset | session | first bar open (UTC) | history start | last bar close (UTC) | years | bars | "
                    "coverage | split boundary (UTC) | research bars | holdout bars | dups identical / conflicting | "
                    "archive files |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    classes = ["exchange_outage", "archive_gap", "weekend", "daily_break", "holiday", "feed_gap"]
    hcols = [*classes, "venue_flat", "symbol_flat"]
    metrics: dict[str, Any] = {"assets": {}}
    for a, r in sorted(split["assets"].items()):
        f0, f1 = datetime.fromisoformat(r["first_ts_open"]), datetime.fromisoformat(r["last_ts_close"])
        expected = int((f1 - f0).total_seconds() // 300)
        hold_rows = sum(p["rows"] for p in r["holdout"]["parts"])
        cov = r["rows_total"] / expected
        dd = r["dedup"]
        first_month = min((f["period"] for f in man.get("files", {}).values()
                           if f["asset"] == a and f.get("kind") == "monthly"), default=None)
        start = ("archive start" if first_month and f0.strftime("%Y-%m") == first_month and f0.day == 1
                 and f0.hour == 0 and f0.minute == 0 else "listing")
        L.append(f"| {a} | {r['session']} | {_d(r['first_ts_open'])} | {start} | {_d(r['last_ts_close'])} | "
                 f"{(f1 - f0).days / 365.25:.2f} | {r['rows_total']:,} | {cov:.5%} | {_d(r['boundary_utc'])} | "
                 f"{r['research']['rows']:,} | {hold_rows:,} | {dd['identical']} / {dd['conflicting']} | "
                 f"{q_by.get(a, {}).get('files', 0)} |")
        metrics["assets"][a] = {"boundary": r["boundary_utc"], "rows": r["rows_total"],
                                "research_rows": r["research"]["rows"], "holdout_rows": hold_rows,
                                "research_gaps": gaps["assets"].get(a, {}).get("summary", {}),
                                "holdout_gaps": r["holdout"].get("gap_summary", {})}
    L += ["", "### Research-sample gaps by class (runs / missing bars)", "",
          "| asset | " + " | ".join(classes) + " | total missing | zero-volume bars | OHLC violations | buy_v > v (clipped) |",
          "|---|" + "---|" * (len(classes) + 4)]
    for a, g in sorted(gaps["assets"].items()):
        s = g["summary"]
        tot = sum(v["bars"] for v in s.values())
        q = q_by.get(a, {})
        L.append(f"| {a} | " + " | ".join(f"{s[c]['runs']} / {s[c]['bars']}" if c in s else "" for c in classes)
                 + f" | {tot} | {q.get('zero_volume', 0)} | {q.get('ohlc_violations', 0)} | {q.get('buy_gt_volume', 0)} |")
    L += ["", "### Research-sample flat bars (present, zero volume) by class (runs / bars)", "",
          "| asset | venue_flat | symbol_flat |", "|---|---|---|"]
    for a, g in sorted(gaps["assets"].items()):
        s = g.get("flat_summary", {})
        L.append(f"| {a} | " + " | ".join(f"{s[c]['runs']} / {s[c]['bars']}" if c in s else ""
                                         for c in ("venue_flat", "symbol_flat")) + " |")
    L += ["", "Venue-wide flat runs (research sample; first and last flat bar close, UTC):", "",
          "| first | last | bars | symbols flat |", "|---|---|---|---|"]
    vf: dict[tuple[str, str], list[int]] = {}
    for a, g in gaps["assets"].items():
        for x in g.get("flat_runs", []):
            if x["class"] == "venue_flat":
                vf.setdefault((x["gap_first"], x["gap_last"]), []).append(x["n_missing"])
    for (f0, f1), ns in sorted(vf.items()):
        L.append(f"| {_d(f0)} | {_d(f1)} | {max(ns)} | {len(ns)} |")
    L += ["", "Zero-volume, OHLC and buy_v counters are over all archive files of the asset (research and holdout "
          "periods), counted at conversion, before the split.", "",
          "### Holdout-period gaps and flat bars by class (counts only; the run lists are sealed in "
          "`holdout/<ASSET>/gaps.json`)", "",
          "| asset | " + " | ".join(hcols) + " |", "|---|" + "---|" * len(hcols)]
    for a, r in sorted(split["assets"].items()):
        s = {**r["holdout"].get("gap_summary", {}), **r["holdout"].get("flat_summary", {})}
        L.append(f"| {a} | " + " | ".join(f"{s[c]['runs']} / {s[c]['bars']}" if c in s else "" for c in hcols) + " |")
    L += ["", "### Every research-sample run (UTC; first and last missing bar close)", "",
          "| asset | first missing | last missing | bars | class | peers listed | peers also missing |",
          "|---|---|---|---|---|---|---|"]
    allruns = [(a, x) for a, g in sorted(gaps["assets"].items()) for x in g["runs"]]
    for a, x in sorted(allruns, key=lambda t: (t[1]["gap_first"], t[0])):
        L.append(f"| {a} | {_d(x['gap_first'])} | {_d(x['gap_last'])} | {x['n_missing']} | {x.get('class', '')} | "
                 f"{x.get('peers', '')} | {x.get('peers_missing', '')} |")
    if not allruns:
        L.append("| (none) | | | | | | |")
    # archive-gap repairs: research-period days listed, holdout-period days only counted
    L += ["", "### Archive-gap repairs (daily zips fetched before the split because the monthly zip lacked those days)", "",
          "| asset | research-period days repaired (UTC) | bars added | holdout-period days repaired (count) |",
          "|---|---|---|---|"]
    rep_by: dict[str, list[tuple[str, int]]] = {}
    for f in man.get("files", {}).values():
        if f.get("purpose") == "archive-gap repair" and f.get("status") == "staged":
            rep_by.setdefault(f["asset"], []).append((f["period"], int(f["quality"]["rows"])))
    for a in sorted(rep_by):
        b = split["assets"].get(a, {}).get("boundary_utc", "9999")[:10]
        rs = sorted(x for x in rep_by[a] if x[0] < b)
        n_hold = sum(1 for x in rep_by[a] if x[0] >= b)
        L.append(f"| {a} | {', '.join(d for d, _ in rs)} | {sum(n for _, n in rs)} | {n_hold} |")
    if not rep_by:
        L.append("| (none) | | | |")
    return "\n".join(L) + "\n", metrics


# ---------------------------------------------------------------- phase1

def cmd_phase1(args: argparse.Namespace) -> int:
    from vecopen.__main__ import run_tests
    from vecopen.ingest import Manifest, fill_from_daily, ingest_binance, storage_check
    from vecopen.split import build_split

    man = Manifest()
    dl = ingest_binance(_crypto_symbols(), man)
    fill = fill_from_daily(_crypto_symbols(), Manifest())
    bs = build_split(_crypto_symbols(), source="binance")
    cfd: dict[str, Any] = {"run": False}
    if args.with_cfd:
        rc = cmd_cfd_download(argparse.Namespace(assets=None, dry_run=args.dry_run))
        if rc == 0:
            cfd = {"run": True, **build_split(_cfd_assets(), source="ctrader")}
    gc = research_gap_check()
    print(f"research gaps recomputed from data/research: {gc}")
    tables, tm = build_tables()
    TABLES_PATH.write_text(tables, encoding="utf-8")
    print(f"tables -> {TABLES_PATH.relative_to(PROJECT_ROOT.parent).as_posix()}")
    st = storage_check(Manifest(), 0)
    counts = run_tests(PROJECT_ROOT / "tests")
    green = counts["pytest_exit_code"] == 0 and counts["failed"] == 0 and counts["errors"] == 0
    print(f"tests: {counts}")
    metrics = {"step": "phase1", "tests": counts, "downloaded_files": dl["files_downloaded"],
               "gap_repair_daily_fetched": sum(v["daily_fetched"] for v in fill.values()),
               "newly_split": bs.get("split", []), "cfd": cfd, "research_gaps_match": gc["match"],
               "split_json_sha256": _sha(D.SPLIT_PATH), "gaps_json_sha256": _sha(D.GAPS_PATH),
               "tables_sha256": _sha(TABLES_PATH), "storage_gb": st["repo_data_gb"] + st["vecopen_data_gb"],
               "boundaries": {a: v["boundary"] for a, v in tm["assets"].items()},
               "rows": {a: [v["research_rows"], v["holdout_rows"]] for a, v in tm["assets"].items()}}
    _append("phase 1 data: load_bars, resample, roll adjustment, split; reproduce run", metrics, args.dry_run)
    rc = registry.verify()
    print(f"registry {'OK' if rc.ok else 'FAILED'}: {rc.n_lines} lines")
    return 0 if green and gc["match"] and rc.ok else 1


def add_commands(sub: argparse._SubParsersAction) -> None:
    d = sub.add_parser("data", help="phase 1 data steps")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    c = dsub.add_parser("crypto-download", help="Binance USDT-M 5m archive -> data/staging (manifest)")
    c.add_argument("--symbols", nargs="*")
    c.add_argument("--dry-run", action="store_true", help="do not append a registry line")
    c.set_defaults(fn=cmd_crypto_download)
    fg = dsub.add_parser("fill-gaps", help="fetch Binance daily zips for days with missing 5m bars (before the split)")
    fg.add_argument("--symbols", nargs="*")
    fg.add_argument("--dry-run", action="store_true")
    fg.set_defaults(fn=cmd_fill_gaps)
    b = dsub.add_parser("build-split", help="staging -> research + locked holdout, gap classes, split.json")
    b.add_argument("--source", choices=["binance", "ctrader"], default="binance")
    b.add_argument("--assets", nargs="*")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(fn=cmd_build_split)
    f = dsub.add_parser("cfd-download", help="atx ctrader symbols --resolve + trendbars M5 -> staging (needs keys)")
    f.add_argument("--assets", nargs="*")
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(fn=cmd_cfd_download)
    g = dsub.add_parser("gaps", help="recompute research gap runs and compare with gaps.json")
    g.set_defaults(fn=cmd_gaps)
    s = dsub.add_parser("storage", help="measure data/ and vecopen/data/ against the 30 GB budget")
    s.set_defaults(fn=cmd_storage)
    p = sub.add_parser("phase1", help="phase 1 reproduce command")
    p.add_argument("--with-cfd", action="store_true", help="also download, build and split the CFDs (needs keys)")
    p.add_argument("--dry-run", action="store_true", help="do not append registry lines")
    p.set_defaults(fn=cmd_phase1)
