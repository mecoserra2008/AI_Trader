"""Reversible lake repairs on cold partitions (D8, user-approved 2026-11-27).

Every repair rewrites ONE leaf partition (a directory of part-*.parquet files):
  1. snapshot its part files: only those are touched (a file a writer adds meanwhile stays where it is);
  2. write the repaired rows next to them as `part-<newest written_ms>-<tag>-<run>.parquet.tmp`, typed by
     config/schema.yaml (ms timestamps, f32 columns stay f32);
  3. verify the written file against the intended result as multisets (both EXCEPT ALL directions empty; above
     300k rows an equivalent streamed checksum: count, sum and xor of 64-bit row hashes) and that it holds no exact
     duplicates; report rows in / distinct in / dropped / added / out;
  4. swap: rename the tmp file into place, then move the old files to data/lake_backup/<run>/<lake path>/ (a reader
     can see exact duplicates for a moment, never a hole);
  5. append one line to data/lake_backup/<run>/manifest.jsonl (files moved, counts, sha256 of the new file, dropped
     and added rows when there are few).
`undo --run R` restores a run, `purge --run R` deletes its backup (the manifest is kept under data/lake_backup/manifests/).

Only cold partitions: refused when a file in it is younger than `min_age_min` or when the live updater / collector
writes it (`live_partitions`), unless `allow_hot` (a deploy window with the updater stopped). `--predeploy` means the
D8 updater streams (instruments, stablecoins, extra streams) are not running yet, so their partitions are cold.

    uv run python -m ai_trader.storage.repair plan [--predeploy]
    uv run python -m ai_trader.storage.repair dedup [--table T] [--predeploy] [--allow-hot] [--dry-run]
    uv run python -m ai_trader.storage.repair d8-fixes [--only bch,binance_eth,oi,lsr] [--predeploy] [--allow-hot] [--dry-run]
    uv run python -m ai_trader.storage.repair undo --run R | purge --run R
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ai_trader.core.config import load
from ai_trader.storage.lake import _ARROW

EXACT_VERIFY_ROWS = 300_000  # above this, verification uses the streamed hash checksum (memory bound)
PERIOD_MS = {"5min": 300_000, "15min": 900_000, "30min": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BYBIT = "https://api.bybit.com"


# ---------------------------------------------------------------- run, partitions, hotness

@dataclass
class Run:
    root: Path  # data/lake_backup
    run_id: str
    dry_run: bool = False
    lines: list[dict[str, Any]] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return self.root / self.run_id

    def record(self, line: dict[str, Any]) -> None:
        self.lines.append(line)
        if self.dry_run:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")


def new_run(lake_root: Path, name: str, dry_run: bool = False) -> Run:
    rid = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{name}"
    return Run(lake_root.parent / "lake_backup", rid, dry_run)


def leaf_dirs(lake_root: Path, table: str) -> list[Path]:
    return sorted({p.parent for p in (lake_root / table).rglob("part-*.parquet")})


def part_files(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("part-*.parquet") if p.is_file())


def written_ms(p: Path) -> int:
    try:
        return int(p.name.split("-")[1])
    except (IndexError, ValueError):
        return int(p.stat().st_mtime * 1000)


def live_partitions(settings: dict[str, Any], today: date, *, predeploy: bool = False) -> list[str]:
    """Lake path prefixes (posix, relative to the lake) that the live updater / collector write today."""
    live, y = settings.get("live", {}) or {}, today.year
    out = [f"flow_3s/venue=bybit/category={c}/date={today.isoformat()}" for c in ("linear", "spot", "inverse")]
    out.append(f"liquidations/venue=bybit/year={y}/month={today.month:02d}")
    for pt in live.get("price_types") or ["last", "mark"]:
        for f in live.get("bar_freqs") or ["1h", "4h", "1d"]:
            out.append(f"bars/venue=bybit/price_type={pt}/freq={f}/year={y}")
    if live.get("funding", True):
        out.append(f"funding/venue=bybit/year={y}")
    oi = live.get("open_interest", {"interval": "4h"})
    if oi is not False:
        out.append(f"open_interest/venue=bybit/interval={oi.get('interval', '4h') if isinstance(oi, dict) else oi}/year={y}")
    out.append(f"alt_series/source=deribit/year={y}")  # DVOL (and Deribit funding after D8)
    out.append(f"alt_series/source=alternative_me/year={y}")
    if not predeploy:
        if live.get("instruments"):
            out.append("instruments/venue=bybit")
        if live.get("stablecoins"):
            out.append(f"alt_series/source=defillama/year={y}")
        for x in live.get("extra") or []:
            k = x.get("kind")
            if k == "open_interest":
                out.append(f"open_interest/venue=bybit/interval={x['interval']}/year={y}")
            elif k == "ls_ratio":
                out.append(f"ls_ratio/venue=bybit/period={x['period']}/year={y}")
            elif k == "bars":
                out.append(f"bars/venue=bybit/price_type={x.get('price_type', 'last')}/freq={x['freq']}/year={y}")
    return out


def rel(lake_root: Path, d: Path) -> str:
    return d.relative_to(lake_root).as_posix()


def hot_reason(lake_root: Path, d: Path, hot: list[str], min_age_min: float, now_s: float | None = None) -> str | None:
    r = rel(lake_root, d)
    if any(r == h or r.startswith(h + "/") for h in hot):
        return "written by the live updater / collector"
    files = part_files(d)
    if files:
        age_min = ((now_s or time.time()) - max(f.stat().st_mtime for f in files)) / 60
        if age_min < min_age_min:
            return f"a file is only {age_min:.0f} min old"
    return None


# ---------------------------------------------------------------- the rewrite

def file_columns(spec: dict[str, Any]) -> list[dict[str, Any]]:
    parts = set(spec["partition_by"])
    return [c for c in spec["columns"] if c["name"] not in parts]


def arrow_schema(spec: dict[str, Any]) -> pa.Schema:
    return pa.schema([pa.field(c["name"], _ARROW[c["type"]], nullable=bool(c.get("nullable"))) for c in file_columns(spec)])


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _connect(tmp: Path | None, mem: str = "1500MB", threads: int = 2) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit = '{mem}'; SET threads = {threads}; SET TimeZone = 'UTC'; SET preserve_insertion_order = false")
    if tmp is not None:
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    return con


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _move(src: Path, dst: Path, tries: int = 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.5)


def rewrite_partition(lake_root: Path, d: Path, table: str, spec: dict[str, Any], run: Run, *, tag: str, reason: str,
                      drop_sql: str | None = None, add: pa.Table | None = None, order_by: list[str] | None = None,
                      always: bool = False, compression_level: int = 3, con: duckdb.DuckDBPyConnection | None = None,
                      keep_rows: int = 200) -> dict[str, Any] | None:
    """Rewrite one partition as DISTINCT rows, minus `drop_sql` (a predicate on file columns), plus `add`.
    Returns the manifest line, or None when there is nothing to do (no duplicates, nothing dropped or added and not `always`)."""
    files = part_files(d)
    if not files:
        return None
    own = con is None
    con = con or _connect(run.dir / "duckdb_tmp")
    cols = [c["name"] for c in file_columns(spec)]
    sel = ", ".join(_q(c) for c in cols)
    flist = "[" + ", ".join("'" + f.as_posix().replace("'", "''") + "'" for f in files) + "]"
    con.execute(f"CREATE OR REPLACE TEMP VIEW src AS SELECT {sel} FROM read_parquet({flist}, hive_partitioning = false, union_by_name = true)")
    # Materialised once: every count, set difference and the write read this table, not the (many) part files.
    con.execute("DROP TABLE IF EXISTS dist")
    con.execute("CREATE TEMP TABLE dist AS SELECT DISTINCT * FROM src")
    if add is not None and add.num_rows:
        con.register("add_rows", add.select(cols))
        add_sql = f"SELECT {sel} FROM add_rows"
    else:
        add_sql = None
    keep = f"SELECT {sel} FROM dist" + (f" WHERE NOT ({drop_sql})" if drop_sql else "")
    target = keep + (f" UNION ALL SELECT DISTINCT * FROM ({add_sql})" if add_sql else "")
    con.execute(f"CREATE OR REPLACE TEMP VIEW target AS {target}")
    n_in = con.sql("SELECT count(*) FROM src").fetchall()[0][0]
    n_dist = con.sql("SELECT count(*) FROM dist").fetchall()[0][0]
    n_drop = con.sql(f"SELECT count(*) FROM dist WHERE {drop_sql}").fetchall()[0][0] if drop_sql else 0
    n_add = con.sql(f"SELECT count(*) FROM (SELECT DISTINCT * FROM ({add_sql}))").fetchall()[0][0] if add_sql else 0
    changed = drop_sql is not None or add_sql is not None
    removed = con.sql("SELECT * FROM (SELECT * FROM dist EXCEPT SELECT * FROM target) LIMIT ?", params=[keep_rows]).to_arrow_table().to_pylist() if changed else []
    added = con.sql("SELECT * FROM (SELECT * FROM target EXCEPT SELECT * FROM dist) LIMIT ?", params=[keep_rows]).to_arrow_table().to_pylist() if changed else []
    line: dict[str, Any] = {
        "run": run.run_id, "ts": datetime.now(UTC).isoformat(), "table": table, "partition": rel(lake_root, d), "tag": tag,
        "reason": reason, "files_in": [f.name for f in files], "rows_in": n_in, "rows_distinct_in": n_dist,
        "exact_duplicates": n_in - n_dist, "rows_dropped": n_drop, "rows_added": n_add, "rows_out": n_dist - n_drop + n_add,
        "removed_rows": removed, "added_rows": added,
    }
    if n_in == n_dist and n_drop == 0 and n_add == 0 and not (always and len(files) > 1):
        if own:
            con.close()
        return None
    if run.dry_run:
        line["dry_run"] = True
        run.record(line)
        if own:
            con.close()
        return line
    schema = arrow_schema(spec)
    # The name carries the write time (part-<written_ms>-...). Kept rows were all on disk by the newest input file;
    # re-fetched rows were written now, so a file holding them is stamped now (else they would look written earlier).
    stamp = int(time.time() * 1000) if add_sql else max(written_ms(f) for f in files)
    final = d / f"part-{stamp}-{tag}-{run.run_id}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    order = " ORDER BY " + ", ".join(_q(c) for c in order_by) if order_by else ""
    reader = con.sql(f"SELECT * FROM target{order}").to_arrow_reader(200_000)
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=compression_level) as w:
        for b in reader:
            w.write_table(pa.Table.from_batches([b]).cast(schema))
    # Verify the file as written (types round-tripped through the schema cast).
    con.execute(f"CREATE OR REPLACE TEMP VIEW written AS SELECT {sel} FROM read_parquet('{tmp.as_posix()}')")
    n_out = con.sql("SELECT count(*) FROM written").fetchall()[0][0]
    n_out_dist = con.sql("SELECT count(*) FROM (SELECT DISTINCT * FROM written)").fetchall()[0][0]
    if line["rows_out"] <= EXACT_VERIFY_ROWS:
        extra = con.sql("SELECT count(*) FROM (SELECT * FROM written EXCEPT ALL SELECT * FROM target)").fetchall()[0][0]
        missing = con.sql("SELECT count(*) FROM (SELECT * FROM target EXCEPT ALL SELECT * FROM written)").fetchall()[0][0]
        method = "except_all"
    else:
        # Large partitions (a flow_3s day is 1-3M rows): an order-independent multiset checksum, streamed, instead of
        # two set differences that need the whole partition in memory. Equal counts + equal sums and xors of 64-bit
        # row hashes; a mismatch aborts the swap.
        hrow = "hash(" + ", ".join(_q(c) for c in cols) + ")"
        agg = f"SELECT count(*), sum({hrow}::HUGEINT), bit_xor({hrow}) FROM "
        a, b = con.sql(agg + "written").fetchall()[0], con.sql(agg + "target").fetchall()[0]
        extra = missing = 0 if a == b else -1
        method = "hash_checksum"
    ok = n_out == line["rows_out"] and n_out_dist == n_out and extra == 0 and missing == 0
    line.update({"verify": {"method": method, "rows_out": n_out, "distinct_out": n_out_dist, "extra_vs_target": extra, "missing_vs_target": missing, "ok": ok}})
    if not ok:
        tmp.unlink(missing_ok=True)
        line["status"] = "aborted: verification failed"
        run.record(line)
        if own:
            con.close()
        raise RuntimeError(f"{line['partition']}: verification failed {line['verify']}")
    line["new_file"] = final.name
    line["new_sha256"] = sha256(tmp)
    line["bytes_in"] = sum(f.stat().st_size for f in files)
    line["bytes_out"] = tmp.stat().st_size
    os.replace(tmp, final)
    bdir = run.dir / rel(lake_root, d)
    for f in files:
        _move(f, bdir / f.name)
    line["backup_dir"] = bdir.as_posix()
    line["status"] = "done"
    run.record(line)
    con.execute("DROP TABLE IF EXISTS dist")
    if own:
        con.close()
    return line


def undo(lake_root: Path, run_dir: Path) -> list[str]:
    log = []
    lines = [json.loads(x) for x in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for ln in reversed(lines):
        if ln.get("status") != "done":
            continue
        d = lake_root / ln["partition"]
        for name in ln["files_in"]:
            src = Path(ln["backup_dir"]) / name
            if src.exists():
                _move(src, d / name)
        (d / ln["new_file"]).unlink(missing_ok=True)
        log.append(f"restored {ln['partition']} ({len(ln['files_in'])} files)")
    with (run_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run_dir.name, "ts": datetime.now(UTC).isoformat(), "status": "undone"}) + "\n")
    return log


def purge(run_dir: Path) -> str:
    keep = run_dir.parent / "manifests"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "manifest.jsonl", keep / f"{run_dir.name}.jsonl")
    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
    shutil.rmtree(run_dir)
    return f"purged {run_dir.name}: {size / 1e6:.1f} MB freed; manifest kept in {keep / (run_dir.name + '.jsonl')}"


# ---------------------------------------------------------------- the D8 repairs

def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-trader/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def refetch_open_bucket(con: duckdb.DuckDBPyConnection, d: Path, table: str, period: str) -> tuple[str | None, pa.Table | None, list[dict]]:
    """Rows written while their period was still open (file write time < event_ts + period): re-fetch the final
    value from Bybit. Returns (drop predicate, replacement rows, log). known_ts keeps the table rule (end + 2 s)."""
    pm = PERIOD_MS[period]
    files = part_files(d)
    flist = "[" + ", ".join("'" + f.as_posix() + "'" for f in files) + "]"
    rows = con.sql(f"""SELECT DISTINCT symbol, epoch_ms(event_ts) AS ev FROM read_parquet({flist}, filename = true)
                       WHERE CAST(regexp_extract(filename, 'part-([0-9]+)-', 1) AS BIGINT) < epoch_ms(event_ts) + {pm}
                       ORDER BY 1, 2""").fetchall()
    if not rows:
        return None, None, []
    lat = int(load().settings["data"]["known_ts_latency_ms"].get(table, 2000))
    out, log = [], []
    for sym, ev in rows:
        if table == "open_interest":
            j = _get_json(f"{BYBIT}/v5/market/open-interest?category=linear&symbol={sym}&intervalTime={period}&startTime={ev}&endTime={ev}&limit=5")
            hit = [x for x in j["result"]["list"] if int(x["timestamp"]) == ev]
            vals = {"open_interest": float(hit[0]["openInterest"])} if hit else None
        else:
            j = _get_json(f"{BYBIT}/v5/market/account-ratio?category=linear&symbol={sym}&period={period}&startTime={ev}&endTime={ev}&limit=5")
            hit = [x for x in j["result"]["list"] if int(x["timestamp"]) == ev]
            vals = {"buy_ratio": float(hit[0]["buyRatio"]), "sell_ratio": float(hit[0]["sellRatio"])} if hit else None
        time.sleep(0.12)
        if vals is None:
            log.append({"symbol": sym, "event_ts": ev, "refetch": "not served"})
            continue
        out.append({"symbol": sym, "event_ts": ev, **vals, "known_ts": ev + pm + lat})
        log.append({"symbol": sym, "event_ts": ev, **vals})
    if not out:
        return None, None, log
    keys = " OR ".join(f"(symbol = '{r['symbol']}' AND epoch_ms(event_ts) = {r['event_ts']})" for r in out)
    t = pa.Table.from_pylist(out)
    t = t.set_column(t.schema.get_field_index("event_ts"), "event_ts", pa.array(t["event_ts"].to_pylist(), pa.timestamp("ms", tz="UTC")))
    t = t.set_column(t.schema.get_field_index("known_ts"), "known_ts", pa.array(t["known_ts"].to_pylist(), pa.timestamp("ms", tz="UTC")))
    return keys, t, log


def d8_fixes(lake_root: Path, run: Run, only: set[str], hot: list[str], allow_hot: bool, min_age_min: float) -> list[str]:
    cfg = load()
    tables = cfg.schema["tables"]
    con = _connect(run.dir / "duckdb_tmp")
    log: list[str] = []

    def go(d: Path, table: str, **kw: Any) -> None:
        if not d.is_dir():
            log.append(f"skip {rel(lake_root, d)}: missing")
            return
        why = hot_reason(lake_root, d, hot, min_age_min)
        if why and not allow_hot:
            log.append(f"REFUSED {rel(lake_root, d)}: hot ({why}); run in the deploy window with --allow-hot")
            return
        ln = rewrite_partition(lake_root, d, table, tables[table], run, con=con, **kw)
        log.append(f"{kw['tag']} {rel(lake_root, d)}: " + ("nothing to do" if ln is None else f"in {ln['rows_in']} distinct {ln['rows_distinct_in']} dropped {ln['rows_dropped']} added {ln['rows_added']} out {ln['rows_out']} ({ln.get('status', 'dry run')})"))

    if "bch" in only:
        go(lake_root / "bars/venue=bybit/price_type=last/freq=1d/year=2020", "bars", tag="d8bch",
           reason="BCHUSDT 1d 2020-12-14 has open = 0 and low = 0 (launch-day bar, still served so by Bybit): dropped",
           drop_sql="symbol = 'BCHUSDT' AND event_ts = TIMESTAMPTZ '2020-12-14 00:00:00+00' AND (open <= 0 OR low <= 0)")
    if "binance_eth" in only:
        go(lake_root / "bars/venue=binance/price_type=last/freq=1h/year=2018", "bars", tag="d8grid",
           reason="43 Binance ETHUSDT 1h bars on a :28:14.8 grid after the 2018-02-08/09 outage (no on-grid bar for those hours): "
                  "dropped rather than relabelled 28 min early",
           drop_sql="epoch_ms(event_ts) % 3600000 <> 0")
    for kind, table, pcol in (("oi", "open_interest", "interval"), ("lsr", "ls_ratio", "period")):
        if kind not in only:
            continue
        for d in sorted((lake_root / table / "venue=bybit").glob(f"{pcol}=*/year=*")):
            period = d.parent.name.split("=", 1)[1]
            why = hot_reason(lake_root, d, hot, min_age_min)
            if why and not allow_hot:
                # Only refuse when there is something to fix.
                drop, _, _ = refetch_open_bucket_dry(con, d, period)
                if drop:
                    log.append(f"REFUSED {rel(lake_root, d)}: hot ({why}); run in the deploy window with --allow-hot")
                continue
            drop, add, rlog = refetch_open_bucket(con, d, table, period)
            if drop is None:
                continue
            go(d, table, tag=f"d8{kind}", reason=f"{add.num_rows} rows had been fetched while their {period} period was open: "
               f"replaced by the final venue value (known_ts unchanged: end + latency); refetch log {json.dumps(rlog)}",
               drop_sql=drop, add=add)
    con.close()
    return log


def refetch_open_bucket_dry(con: duckdb.DuckDBPyConnection, d: Path, period: str) -> tuple[str | None, None, list]:
    pm = PERIOD_MS[period]
    files = part_files(d)
    if not files:
        return None, None, []
    flist = "[" + ", ".join("'" + f.as_posix() + "'" for f in files) + "]"
    n = con.sql(f"""SELECT count(*) FROM read_parquet({flist}, filename = true)
                    WHERE CAST(regexp_extract(filename, 'part-([0-9]+)-', 1) AS BIGINT) < epoch_ms(event_ts) + {pm}""").fetchall()[0][0]
    return ("x" if n else None), None, []


ORDER = {"bars": ["symbol", "event_ts"], "funding": ["symbol", "event_ts"], "open_interest": ["symbol", "event_ts"],
         "ls_ratio": ["symbol", "event_ts"], "flow_bars": ["symbol", "event_ts"], "alt_series": ["series_id", "entity", "event_ts"],
         "instruments": ["event_ts", "symbol"]}


def dedup(lake_root: Path, run: Run, tables: list[str], hot: list[str], allow_hot: bool, min_age_min: float) -> list[str]:
    cfg = load()
    con = _connect(run.dir / "duckdb_tmp")
    log: list[str] = []
    for t in tables:
        for d in leaf_dirs(lake_root, t):
            why = hot_reason(lake_root, d, hot, min_age_min)
            if why and not allow_hot:
                n, nd = con.sql(f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT * FROM read_parquet('{d.as_posix()}/part-*.parquet'))) "
                                f"FROM read_parquet('{d.as_posix()}/part-*.parquet')").fetchall()[0]
                if n != nd:
                    log.append(f"REFUSED {rel(lake_root, d)}: {n - nd} exact duplicates but hot ({why})")
                continue
            ln = rewrite_partition(lake_root, d, t, cfg.schema["tables"][t], run, tag="dedup", reason="exact duplicate rows (same values and known_ts)",
                                   order_by=ORDER.get(t), con=con)
            if ln:
                log.append(f"dedup {rel(lake_root, d)}: {ln['exact_duplicates']} exact duplicates removed, {len(ln['files_in'])} files -> 1 ({ln.get('status', 'dry run')})")
    con.close()
    return log


def _cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="repair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "dedup", "d8-fixes"):
        p = sub.add_parser(name)
        p.add_argument("--predeploy", action="store_true", help="the D8 updater streams are not running yet")
        p.add_argument("--allow-hot", action="store_true", help="deploy window only: the updater is stopped")
        p.add_argument("--min-age-min", type=float, default=120.0)
        p.add_argument("--dry-run", action="store_true")
        if name == "dedup":
            p.add_argument("--table", action="append")
        if name == "d8-fixes":
            p.add_argument("--only", default="bch,binance_eth,oi,lsr")
    for name in ("undo", "purge"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = load()
    lake_root = cfg.lake_dir
    if a.cmd in ("undo", "purge"):
        run_dir = lake_root.parent / "lake_backup" / a.run
        out = undo(lake_root, run_dir) if a.cmd == "undo" else [purge(run_dir)]
        sys.stdout.write("\n".join(out) + "\n")
        return
    hot = live_partitions(cfg.settings, datetime.now(UTC).date(), predeploy=a.predeploy)
    dry = a.dry_run or a.cmd == "plan"
    run = new_run(lake_root, a.cmd.replace("-", ""), dry_run=dry)
    tables = (a.table if a.cmd == "dedup" and a.table else ["bars", "funding", "open_interest", "ls_ratio", "flow_bars", "alt_series", "instruments"])
    log: list[str] = []
    if a.cmd in ("plan", "dedup"):
        log += dedup(lake_root, run, tables, hot, a.allow_hot, a.min_age_min)
    if a.cmd in ("plan", "d8-fixes"):
        only = set(getattr(a, "only", "bch,binance_eth,oi,lsr").split(","))
        log += d8_fixes(lake_root, run, only, hot, a.allow_hot, a.min_age_min)
    sys.stdout.write("\n".join(log) + "\n")
    if not dry and run.lines:
        sys.stdout.write(f"manifest: {run.dir / 'manifest.jsonl'}\n")
    shutil.rmtree(run.dir / "duckdb_tmp", ignore_errors=True)
    if run.dir.is_dir() and not any(run.dir.iterdir()):
        run.dir.rmdir()


if __name__ == "__main__":
    _cli()
