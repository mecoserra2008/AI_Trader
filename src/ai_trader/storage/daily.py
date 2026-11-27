"""Daily data jobs that do not belong in the always-on updater (D8, user-approved 2026-11-27).

  1. maintain   flow_3s: compact + 1-minute rollup of every COLD day (reversible, storage.flow / storage.repair), then
                prune to storage.budget_gb;
  2. flow_bars  the public trade archive of the last `--days` UTC days (default 2) -> flow_bars 1h + 15m for every
                symbol already in flow_bars, only for days that symbol does not have yet (`atx trades`);
  3. svi        the 08:00 UTC Deribit surface (storage.deribit_svi) for BTC and ETH, one process per currency.
Heavy downloads (a BTC trade archive is ~100-160 MB a day) are why flow_bars is not an updater stream: the updater
must never stall the hourly bars. Closing an older flow_bars hole is a data decision: run it explicitly with
`--flow-bars-since YYYY-MM-DD`.

    uv run python -m ai_trader.storage.daily [--only maintain,flow_bars,svi] [--days 2] [--flow-bars-since D] [--dry-run]

`scripts/ops/desk_start.sh` runs it as the `daily_ops` loop at 08:20 UTC (wall clock, suspend-safe) through the
governor. There is no OS scheduler entry: after a reboot run scripts/ops/desk_start.sh.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb

from ai_trader.core.config import load
from ai_trader.storage import flow
from ai_trader.storage.lake import Lake


def atx_binary(root: Path) -> Path:
    for p in (root / "bin" / "atx-ops.exe", root / "target" / "release" / "atx.exe"):
        if p.is_file():
            return p
    raise FileNotFoundError("no atx binary: build with `cargo build --release -p atx` and copy it to bin/atx-ops.exe")


def flow_bars_newest(lake: Lake) -> dict[str, date]:
    files = list((lake.root / "flow_bars").rglob("part-*.parquet"))
    if not files:
        return {}
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'; SET memory_limit = '500MB'")
    flist = "[" + ", ".join("'" + f.as_posix() + "'" for f in files) + "]"
    rows = con.sql(f"SELECT symbol, max(epoch_ms(event_ts)) FROM read_parquet({flist}, hive_partitioning = true) WHERE freq = '1h' GROUP BY 1").fetchall()
    con.close()
    return {s: datetime.fromtimestamp(ms / 1000, UTC).date() for s, ms in rows}


def flow_bars_plan(newest: dict[str, date], today: date, days: int, since: date | None) -> list[tuple[str, date]]:
    """(symbol, day) pairs to fetch: days after the symbol's newest day, inside [since or today - days, today - 1]."""
    lo = since or (today - timedelta(days=days))
    out = []
    for sym in sorted(newest):
        d = max(lo, newest[sym] + timedelta(days=1))
        while d < today:
            out.append((sym, d))
            d += timedelta(days=1)
    return out


def run_flow_bars(root: Path, lake: Lake, days: int, since: date | None, dry_run: bool) -> list[str]:
    today = datetime.now(UTC).date()
    plan = flow_bars_plan(flow_bars_newest(lake), today, days, since)
    log = [f"flow_bars: {len(plan)} symbol-days to fetch"]
    if dry_run or not plan:
        return log + [f"  would fetch {s} {d}" for s, d in plan[:20]]
    exe = atx_binary(root)
    for sym, d in plan:
        p = subprocess.run([str(exe), "trades", "--symbol", sym, "--from", d.isoformat(), "--to", d.isoformat(), "--freqs", "1h,15m"],
                           cwd=root, capture_output=True, text=True, env={**os.environ, "RUST_LOG": "warn"})
        last = (p.stdout.strip().splitlines() or [""])[-1]
        log.append(f"  {sym} {d}: rc {p.returncode} {last[:120]}" + (f" | {p.stderr.strip()[-200:]}" if p.returncode else ""))
        if "no archive file" in p.stdout:
            log.append(f"  {sym}: archive for {d} not published yet; stopping this symbol")
    return log


def run_svi(dry_run: bool) -> list[str]:
    log = []
    for ccy in ("BTC", "ETH"):
        args = [sys.executable, "-m", "ai_trader.storage.deribit_svi", "--ccy", ccy] + (["--dry-run"] if dry_run else [])
        p = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "OPT_CCY": ccy, "PYTHONIOENCODING": "utf-8"})
        log += [f"svi {ccy}: {x}" for x in p.stdout.strip().splitlines()[-6:]]
        if p.returncode:
            log.append(f"svi {ccy}: FAILED rc {p.returncode}: {p.stderr.strip()[-400:]}")
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="daily")
    ap.add_argument("--only", default="maintain,flow_bars,svi")
    ap.add_argument("--days", type=int, default=2, help="flow_bars: look back this many days for missing ones")
    ap.add_argument("--flow-bars-since", type=date.fromisoformat, help="flow_bars: close a hole from this day (explicit backfill)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    cfg = load()
    lake = Lake()
    log = [f"daily jobs {datetime.now(UTC):%Y-%m-%dT%H:%M}Z: {sorted(only)}"]
    rc = 0
    for name, job in (("maintain", lambda: flow.maintain(lake, dry_run=a.dry_run)),
                      ("flow_bars", lambda: run_flow_bars(cfg.root, lake, a.days, a.flow_bars_since, a.dry_run)),
                      ("svi", lambda: run_svi(a.dry_run))):
        if name not in only:
            continue
        try:
            log += job()
        except Exception as e:  # noqa: BLE001 - one failing job must not stop the others
            log.append(f"{name}: FAILED {type(e).__name__}: {e}")
            rc = 1
    sys.stdout.write("\n".join(log) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
