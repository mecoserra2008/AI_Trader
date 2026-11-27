"""WS02 data quality and point-in-time audit of the lake (report: research/reports/ws02_2026-11-27.md).

Outputs (data/research/ws02/, never the lake):
    machine_windows.json   machine-down intervals (Windows sleep/wake, shutdown/boot) + reconnect margin
    gaps.parquet           one row per hole: dataset, stream, symbol, gap_start, gap_end, n_missing, kind, cause
    coverage.json          per dataset / stream / symbol / day: present, expected, coverage, PIT and duplicate counts

    python -m ai_trader.research.data_quality machine
    python -m ai_trader.research.data_quality flow3s [--days 2026-11-20 ...] [--category linear]
    python -m ai_trader.research.data_quality tables
    python -m ai_trader.research.data_quality assemble

Joining the gap map (other workstreams): drop rows whose [event_ts, event_ts + bar) overlaps any gaps.parquet row of the
same dataset/symbol (or symbol == '*' for machine-wide windows). Times are UTC epoch milliseconds.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from ai_trader.core.config import load

CANDLE_MS = 3_000
DAY_MS = 86_400_000
RECONNECT_MARGIN_MS = 10 * 60_000
EMPTY_RUN_MIN = 20  # >= 60 s of candles with no trades AND no valid book = collector had no data
WRITTEN_RE = r"part-(\d{13})-"
FREQ_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
           "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000, "1d": DAY_MS}

GAP_SCHEMA = {"dataset": pl.Utf8, "stream": pl.Utf8, "symbol": pl.Utf8, "gap_start": pl.Int64, "gap_end": pl.Int64,
              "n_missing": pl.Int64, "kind": pl.Utf8, "cause": pl.Utf8, "fillable_from": pl.Utf8, "filled": pl.Utf8}


def out_dir() -> Path:
    d = load().root / "data" / "research" / "ws02"
    (d / "tmp").mkdir(parents=True, exist_ok=True)
    return d


def ms(t: datetime | date) -> int:
    if not isinstance(t, datetime):
        t = datetime(t.year, t.month, t.day, tzinfo=UTC)
    return int(t.timestamp() * 1000)


def iso(t_ms: int) -> str:
    return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=int(t_ms))).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- pure helpers (tested)

def missing_runs(ts: np.ndarray, start: int, end: int, step: int) -> list[tuple[int, int, int]]:
    """Holes on the grid start, start+step, ... < end given sorted unique present stamps `ts`.
    Returns (gap_start, gap_end_exclusive, n_missing)."""
    ts = np.asarray(ts, dtype=np.int64)
    ts = ts[(ts >= start) & (ts < end)]
    edges = np.concatenate([[start - step], ts, [end]])
    d = np.diff(edges)
    out = []
    for i in np.nonzero(d > step)[0]:
        a, b = int(edges[i] + step), int(edges[i + 1])
        out.append((a, b, (b - a) // step))
    return out


def true_runs(ts: np.ndarray, flag: np.ndarray, step: int, min_len: int) -> list[tuple[int, int, int]]:
    """Runs of consecutive grid stamps (ts sorted, spacing == step) where flag is True, of length >= min_len."""
    ts = np.asarray(ts, dtype=np.int64)
    flag = np.asarray(flag, dtype=bool)
    out, i, n = [], 0, len(ts)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1] and ts[j + 1] - ts[j] == step:
            j += 1
        if j - i + 1 >= min_len:
            out.append((int(ts[i]), int(ts[j] + step), j - i + 1))
        i = j + 1
    return out


def merge(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def overlap_ms(a: int, b: int, iv: list[tuple[int, int]]) -> int:
    return sum(max(0, min(b, y) - max(a, x)) for x, y in iv)


def cause_of(a: int, b: int, down: list[tuple[int, int]]) -> str:
    """machine_down if >= 50% of the hole lies in a machine-down window (incl. reconnect margin)."""
    return "machine_down" if b > a and overlap_ms(a, b, down) >= 0.5 * (b - a) else "unknown"


KIND_LABEL = {"sleep": "pc_suspend", "reboot": "reboot", "modern_standby": "modern_standby"}


def cause_kind(a: int, b: int, wins: list[dict]) -> str | None:
    """Label of the machine-down kind covering >= 50% of [a, b) (windows incl. reconnect margin), else None."""
    if b <= a:
        return None
    tot: dict[str, int] = {}
    for k in KIND_LABEL:
        iv = merge([(w["down_start"], w["margin_end"]) for w in wins if w["kind"] == k])
        tot[k] = overlap_ms(a, b, iv)
    allv = overlap_ms(a, b, merge([(w["down_start"], w["margin_end"]) for w in wins]))
    if allv < 0.5 * (b - a):
        return None
    return KIND_LABEL[max(tot, key=tot.get)]


def _complement(a: int, b: int, iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, t = [], a
    for x, y in merge(iv):
        if y <= t or x >= b:
            continue
        if x > t:
            out.append((t, x))
        t = max(t, y)
    if t < b:
        out.append((t, b))
    return out


def mask_in(ts: np.ndarray, iv: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(len(ts), dtype=bool)
    for a, b in iv:
        m |= (ts >= a) & (ts < b)
    return m


# ---------------------------------------------------------------- machine-down windows

PS_SLEEP = ("Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Power-Troubleshooter'; Id=1; "
            "StartTime=(Get-Date '2026-11-01')} -ErrorAction SilentlyContinue | ForEach-Object { $x=[xml]$_.ToXml(); $d=@{}; "
            "$x.Event.EventData.Data | ForEach-Object { $d[$_.Name]=$_.'#text' }; '{0}|{1}' -f $d['SleepTime'],$d['WakeTime'] }")
PS_BOOT = ("Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-General'; Id=@(12,13); "
           "StartTime=(Get-Date '2026-11-01')} -ErrorAction SilentlyContinue | ForEach-Object { '{0}|{1}' -f $_.Id, "
           "$_.TimeCreated.ToUniversalTime().ToString('o') }")


PS_STANDBY = ("Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=@(506,507); "
              "StartTime=(Get-Date '2026-11-01')} -ErrorAction SilentlyContinue | ForEach-Object { '{0}|{1}' -f $_.Id, "
              "$_.TimeCreated.ToUniversalTime().ToString('o') }")


def _ps(cmd: str) -> list[str]:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
    return [x.strip() for x in r.stdout.splitlines() if "|" in x]


def _parse_iso(s: str) -> int:
    s = re.sub(r"(\.\d{6})\d*", r"\1", s.strip().replace("Z", "+00:00"))
    return ms(datetime.fromisoformat(s))


def machine_windows(refresh: bool = False) -> list[dict]:
    p = out_dir() / "machine_windows.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())["windows"]
    wins = []
    for line in _ps(PS_SLEEP):
        a, b = line.split("|")[:2]
        wins.append({"kind": "sleep", "down_start": _parse_iso(a), "down_end": _parse_iso(b)})
    ev = sorted((_parse_iso(t), int(i)) for i, t in (x.split("|") for x in _ps(PS_BOOT)))
    for (t0, i0), (t1, i1) in zip(ev, ev[1:], strict=False):
        if i0 == 13 and i1 == 12:  # shutdown -> boot
            wins.append({"kind": "reboot", "down_start": t0, "down_end": t1})
    # Modern Standby (Kernel-Power 506 enter -> 507 exit): the network is off, websockets and pollers stall
    ev = sorted((_parse_iso(t), int(i)) for i, t in (x.split("|") for x in _ps(PS_STANDBY)))
    start = None
    for t, i in ev:
        if i == 506 and start is None:
            start = t
        elif i == 507 and start is not None:
            if t - start > 60_000:
                wins.append({"kind": "modern_standby", "down_start": start, "down_end": t})
            start = None
    wins.sort(key=lambda w: w["down_start"])
    for w in wins:
        w["margin_end"] = w["down_end"] + RECONNECT_MARGIN_MS
        w["start_utc"], w["end_utc"] = iso(w["down_start"]), iso(w["down_end"])
        w["hours"] = round((w["down_end"] - w["down_start"]) / 3.6e6, 2)
    p.write_text(json.dumps({"source": "Windows System log: Power-Troubleshooter 1 (sleep), Kernel-General 12/13 (reboot), Kernel-Power 506/507 (Modern Standby)",
                             "reconnect_margin_ms": RECONNECT_MARGIN_MS, "windows": wins}, indent=1))
    return wins


def down_intervals(wins: list[dict], kinds: tuple[str, ...] = ("sleep", "reboot", "modern_standby")) -> list[tuple[int, int]]:
    return merge([(w["down_start"], w["margin_end"]) for w in wins if w["kind"] in kinds])


# ---------------------------------------------------------------- flow_3s per day

def _con(mem: str = "1200MB", threads: int = 2) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute(f"SET threads={threads}")
    return con


def scan_flow3s_day(category: str, day: str, down: list[tuple[int, int]], now_ms: int | None = None) -> dict:
    """One flow_3s day partition -> PIT/duplicate stats, coverage per symbol, holes, hourly rollups."""
    root = load().lake_dir / "flow_3s" / "venue=bybit" / f"category={category}" / f"date={day}"
    files = sorted(root.glob("*.parquet"))
    d0 = ms(date.fromisoformat(day))
    d1 = d0 + DAY_MS
    now_ms = now_ms or ms(datetime.now(UTC))
    end = min(d1, now_ms - 10 * 60_000)  # the current day: only up to 10 min ago
    con = _con()
    df = pl.from_arrow(con.sql(f"""
        SELECT symbol, epoch_ms(event_ts) AS ev, epoch_ms(known_ts) AS kn, (n_buy + n_sell) AS n,
               CAST(buy_vol AS DOUBLE) + CAST(sell_vol AS DOUBLE) AS vol, close, book_cov,
               TRY_CAST(regexp_extract(filename, '{WRITTEN_RE}', 1) AS BIGINT) AS wms
        FROM read_parquet({[f.as_posix() for f in files]}, filename = true)
    """).to_arrow_table())
    stats = {"category": category, "day": day, "files": len(files), "rows": df.height,
             "pit_grid": int((df["ev"] % CANDLE_MS != 0).sum()),
             "pit_known_before_end": int((df["kn"] < df["ev"] + CANDLE_MS).sum()),
             "pit_written_before_known": int((df["wms"].is_not_null() & (df["wms"] < df["kn"])).sum()),
             "written_before_known_max_ms": int((df["kn"] - df["wms"]).max() or 0),
             "no_write_ms": int(df["wms"].is_null().sum()),
             "outside_partition": int(((df["ev"] < d0) | (df["ev"] >= d1)).sum())}
    lag = (df["kn"] - df["ev"] - CANDLE_MS)
    stats["lag_ms_p50"], stats["lag_ms_p99"], stats["lag_ms_max"] = (float(lag.quantile(0.5)), float(lag.quantile(0.99)), int(lag.max()))
    stats["late_gt_60s"] = int((lag > 60_000).sum())
    g = df.group_by("symbol", "ev").agg(pl.len().alias("k"), pl.col("kn").n_unique().alias("nk"))
    dups = g.filter(pl.col("k") > 1)
    stats["dup_keys"] = dups.height
    conf = (df.join(dups.select("symbol", "ev"), on=["symbol", "ev"]).group_by("symbol", "ev", "kn")
            .agg(pl.struct("n", "vol", "close", "book_cov").n_unique().alias("nv")).filter(pl.col("nv") > 1))
    stats["dup_conflicting"] = conf.height
    # as-of dedupe: latest known_ts per key
    df = df.sort("symbol", "ev", "kn").unique(["symbol", "ev"], keep="last").sort("symbol", "ev")
    df = df.filter((pl.col("ev") >= d0) & (pl.col("ev") < end))
    df = df.with_columns(((pl.col("n") == 0) & (pl.col("book_cov") <= 0)).alias("empty"))
    gaps, cov = [], []
    awake_mask_grid = None
    for (sym,), s in df.group_by("symbol", maintain_order=True):
        ts = s["ev"].to_numpy()
        for a, b, k in missing_runs(ts, d0, end, CANDLE_MS):
            gaps.append((sym, a, b, k, "missing_rows"))
        for a, b, k in true_runs(ts, s["empty"].to_numpy(), CANDLE_MS, EMPTY_RUN_MIN):
            gaps.append((sym, a, b, k, "empty_fill"))
        exp_n = (end - d0) // CANDLE_MS
        if awake_mask_grid is None:
            grid = np.arange(d0, end, CANDLE_MS, dtype=np.int64)
            awake_mask_grid = ~mask_in(grid, down)
        awake_present = int((~mask_in(ts, down)).sum())
        cov.append({"symbol": sym, "present": len(ts), "expected": int(exp_n), "empty": int(s["empty"].sum()),
                    "awake_expected": int(awake_mask_grid.sum()), "awake_present": awake_present})
    hourly = (df.with_columns((pl.col("ev") // 3_600_000 * 3_600_000).alias("hour"))
              .group_by("symbol", "hour").agg(pl.len().alias("rows"), pl.col("vol").sum(), pl.col("n").sum().alias("trades"),
                                              pl.col("empty").sum().alias("empty_rows"),
                                              pl.col("close").drop_nulls().last().alias("last_close"),
                                              pl.col("ev").max().alias("last_ev")))
    gdf = pl.DataFrame(gaps, schema={"symbol": pl.Utf8, "gap_start": pl.Int64, "gap_end": pl.Int64, "n_missing": pl.Int64, "kind": pl.Utf8}, orient="row")
    return {"stats": stats, "gaps": gdf, "coverage": pl.DataFrame(cov), "hourly": hourly}


def run_flow3s(days: list[str] | None, categories: list[str]) -> None:
    base = load().lake_dir / "flow_3s" / "venue=bybit"
    down = down_intervals(machine_windows())
    tmp = out_dir() / "tmp" / "flow3s"
    tmp.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        all_days = sorted(p.name.split("=")[1] for p in (base / f"category={cat}").glob("date=*"))
        for day in (days or all_days):
            if day not in all_days:
                continue
            t0 = datetime.now(UTC)
            r = scan_flow3s_day(cat, day, down)
            r["stats"]["secs"] = (datetime.now(UTC) - t0).total_seconds()
            for k in ("gaps", "coverage", "hourly"):
                r[k].with_columns(pl.lit(cat).alias("category"), pl.lit(day).alias("day")).write_parquet(tmp / f"{cat}_{day}_{k}.parquet")
            (tmp / f"{cat}_{day}_stats.json").write_text(json.dumps(r["stats"]))
            print(json.dumps(r["stats"]), flush=True)


# ---------------------------------------------------------------- REST / WS tables

def _read(con, glob: str, select: str, where: str = "true") -> pl.DataFrame:
    return pl.from_arrow(con.sql(f"""SELECT {select},
        TRY_CAST(regexp_extract(filename, '{WRITTEN_RE}', 1) AS BIGINT) AS wms
        FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true, filename = true) WHERE {where}""").to_arrow_table())


def dup_stats(df: pl.DataFrame, key: list[str], value_cols: list[str]) -> dict:
    """exact = same key + known_ts + values; conflicting = same key + known_ts, different values;
    revisions = same key, several known_ts with different values."""
    g = df.group_by(key).agg(pl.len().alias("k"))
    dk = g.filter(pl.col("k") > 1)
    if dk.height == 0:
        return {"dup_keys": 0, "dup_rows_extra": 0, "dup_conflicting": 0, "revised_keys": 0}
    sub = df.join(dk.select(key), on=key)
    byk = sub.group_by([*key, "kn"]).agg(pl.struct(value_cols).n_unique().alias("nv"), pl.len().alias("c"))
    rev = sub.group_by(key).agg(pl.col("kn").n_unique().alias("nk"), pl.struct(value_cols).n_unique().alias("nv"))
    return {"dup_keys": dk.height, "dup_rows_extra": int((dk["k"] - 1).sum()),
            "dup_conflicting": int((byk["nv"] > 1).sum()),
            "revised_keys": int(((rev["nk"] > 1) & (rev["nv"] > 1)).sum())}


def grid_gaps(df: pl.DataFrame, by: list[str], step: int, start: int | None = None, end: int | None = None) -> list[tuple]:
    out = []
    for keys, s in df.group_by(by, maintain_order=True):
        ts = np.unique(s["ev"].to_numpy())
        a = max(int(ts[0]), start) if start else int(ts[0])
        b = min(int(ts[-1]) + step, end) if end else int(ts[-1]) + step
        for x, y, k in missing_runs(ts, a, b, step):
            out.append((*keys, x, y, k))
    return out


def scan_bars(con, down) -> tuple[list[dict], list[dict]]:
    """Per (venue, price_type, freq): stats per year partition (bounded memory), holes from DISTINCT (symbol, event_ts)."""
    stats, gaps = [], []
    root = load().lake_dir / "bars"
    for d in sorted(root.glob("venue=*/price_type=*/freq=*")):
        venue, pt, freq = (p.name.split("=")[1] for p in (d.parent.parent, d.parent, d))
        step = FREQ_MS[freq]
        st = {"table": "bars", "stream": f"{venue}/{pt}/{freq}", "rows": 0, "pit_known_before_close": 0, "bad_duration": 0, "off_grid": 0,
              "written_before_close": 0, "no_write_ms": 0, "backfilled_rows": 0, "ohlc_bad": 0, "known_minus_close_ms": [None, None],
              "dup_keys": 0, "dup_rows_extra": 0, "dup_conflicting": 0, "revised_keys": 0, "written_before_close_examples": []}
        for y in sorted(d.glob("year=*")):
            df = _read(con, y.as_posix() + "/*.parquet",
                       "symbol, epoch_ms(event_ts) ev, epoch_ms(close_ts) ct, epoch_ms(known_ts) kn, open o, high h, low l, close c, volume v")
            if df.height == 0:
                continue
            km = df["kn"] - df["ct"]
            lo, hi = st["known_minus_close_ms"]
            st["known_minus_close_ms"] = [int(km.min()) if lo is None else min(lo, int(km.min())), int(km.max()) if hi is None else max(hi, int(km.max()))]
            wb = df["wms"].is_not_null() & (df["wms"] < df["ct"])
            for k, v in {"rows": df.height, "pit_known_before_close": int((df["kn"] < df["ct"]).sum()),
                         "bad_duration": int((df["ct"] - df["ev"] != step).sum()), "off_grid": int((df["ev"] % step != 0).sum()),
                         "written_before_close": int(wb.sum()), "no_write_ms": int(df["wms"].is_null().sum()),
                         "backfilled_rows": int(((df["wms"] - df["kn"]) > 60_000).sum()),
                         "ohlc_bad": int(df.select(((pl.col("l") > pl.min_horizontal("o", "c")) | (pl.col("h") < pl.max_horizontal("o", "c"))
                                                    | (pl.col("l") <= 0) | pl.col("o").is_nan() | pl.col("c").is_nan()).sum()).item())}.items():
                st[k] += v
            for k, v in dup_stats(df, ["symbol", "ev"], ["o", "h", "l", "c", "v"]).items():
                st[k] += v
            og = df["ev"] % step != 0
            if og.any() and len(st.setdefault("off_grid_examples", [])) < 5:
                st["off_grid_examples"] += [(r[0], iso(r[1]), iso(r[2]), iso(r[3])) for r in df.filter(og).select("symbol", "ev", "ct", "kn").head(5).rows()]
            if wb.any() and len(st["written_before_close_examples"]) < 5:
                st["written_before_close_examples"] += [(r[0], iso(r[1]), iso(r[2])) for r in df.filter(wb).select("symbol", "ev", "wms").head(5).rows()]
            del df
        ev = pl.from_arrow(con.sql(f"""SELECT DISTINCT symbol, epoch_ms(event_ts) ev FROM read_parquet('{d.as_posix()}/**/*.parquet')""").to_arrow_table())
        st["symbols"] = ev["symbol"].n_unique()
        st["first"], st["last"] = iso(int(ev["ev"].min())), iso(int(ev["ev"].max()))
        st["backfilled_share"] = st["backfilled_rows"] / max(st["rows"], 1)
        stats.append(st)
        for sym, a, b, k in grid_gaps(ev, ["symbol"], step):
            gaps.append({"dataset": "bars", "stream": st["stream"], "symbol": sym, "gap_start": a, "gap_end": b, "n_missing": k,
                         "kind": "missing_rows", "cause": cause_of(a, b, down), "fillable_from": "bybit REST /v5/market/kline" if venue == "bybit" else venue,
                         "filled": ""})
        print(json.dumps({k: v for k, v in st.items() if k != "written_before_close_examples"}), flush=True)
    return stats, gaps


def scan_trades_raw(con) -> tuple[dict, list]:
    """Per symbol/date partition (bounded memory): PIT, duplicates by trade_id, day coverage."""
    st = {"table": "trades_raw", "stream": "venue=bybit", "rows": 0, "pit_violations": 0, "known_minus_event_ms": [None, None, None],
          "symbols": 0, "no_write_ms": None, "dup_keys": 0, "dup_rows_extra": 0, "dup_conflicting": 0, "revised_keys": 0}
    days = []
    root = load().lake_dir / "trades_raw" / "venue=bybit"
    for sd in sorted(root.glob("symbol=*")):
        st["symbols"] += 1
        for dd in sorted(sd.glob("date=*")):
            src = f"read_parquet('{dd.as_posix()}/*.parquet')"
            r = con.sql(f"""SELECT count(*), sum(CASE WHEN known_ts < event_ts THEN 1 ELSE 0 END), min(epoch_ms(known_ts) - epoch_ms(event_ts)),
                            max(epoch_ms(known_ts) - epoch_ms(event_ts)), count(DISTINCT trade_id), min(event_ts)::VARCHAR, max(event_ts)::VARCHAR,
                            count(DISTINCT (trade_id, price, size, side)) FROM {src}""").fetchall()[0]
            st["rows"] += r[0]
            st["pit_violations"] += int(r[1] or 0)
            lo, _, hi = st["known_minus_event_ms"]
            st["known_minus_event_ms"] = [r[2] if lo is None else min(lo, r[2]), None, r[3] if hi is None else max(hi, r[3])]
            st["dup_rows_extra"] += r[0] - r[4]
            st["dup_conflicting"] += r[7] - r[4]
            days.append((sd.name.split("=")[1], dd.name.split("=")[1], r[0], r[5], r[6]))
    print(json.dumps(st), flush=True)
    return st, days


def scan_simple(con, down, table: str, sub: str, period_ms: int | None, value_cols: list[str], select: str,
                pit_expr: str, written_expr: str | None, key: list[str], fill: str) -> tuple[dict, list[dict], pl.DataFrame]:
    glob = (load().lake_dir / table / sub).as_posix() + "/**/*.parquet"
    df = _read(con, glob, select)
    st = {"table": table, "stream": sub, "rows": df.height, "symbols": df["symbol"].n_unique() if "symbol" in df.columns else None,
          "pit_violations": int(df.select(pl.sql_expr(pit_expr).sum()).item() or 0),
          "no_write_ms": int(df["wms"].is_null().sum()),
          "known_minus_event_ms": [int((df["kn"] - df["ev"]).min()), int((df["kn"] - df["ev"]).median()), int((df["kn"] - df["ev"]).max())],
          "first": iso(int(df["ev"].min())), "last": iso(int(df["ev"].max()))}
    if written_expr:
        st["written_before_complete"] = int(df.select(pl.sql_expr(written_expr).sum()).item() or 0)
    st.update(dup_stats(df, key, value_cols))
    gaps = []
    if period_ms:
        st["off_grid"] = int((df["ev"] % period_ms != 0).sum())
        for sym, a, b, k in grid_gaps(df, ["symbol"], period_ms):
            gaps.append({"dataset": table, "stream": sub, "symbol": sym, "gap_start": a, "gap_end": b, "n_missing": k,
                         "kind": "missing_rows", "cause": cause_of(a, b, down), "fillable_from": fill, "filled": ""})
    print(json.dumps(st), flush=True)
    return st, gaps, df


def _mode(a: np.ndarray) -> int:
    v, c = np.unique(a, return_counts=True)
    return int(v[c.argmax()])


def funding_gaps(df: pl.DataFrame, down) -> tuple[dict, list[dict]]:
    """Missing settlements allowing interval changes: a step is a hole iff it exceeds the larger of the modal steps of the
    6 steps before and the 6 after (an interval change leaves one of the two modes equal to the step)."""
    gaps, mis, misl = [], 0, []
    allowed = {h * 3_600_000 for h in (1, 2, 4, 8)}
    for (sym,), s in df.group_by("symbol", maintain_order=True):
        ts = np.unique(s["ev"].to_numpy())
        mis += int((ts % 3_600_000 != 0).sum())
        d = np.diff(ts)
        for i, x in enumerate(d):
            mp = _mode(d[max(0, i - 6):i]) if i > 0 else int(x)
            mn = _mode(d[i + 1:i + 7]) if i + 1 < len(d) else int(x)
            ref = max(mp, mn)
            loc = min(mp, mn)
            if loc in allowed and ts[i + 1] % loc != 0:
                misl.append((sym, iso(int(ts[i + 1])), loc // 3_600_000))
            if ref in allowed and x > ref:
                step = ref if x % ref == 0 else min(mp, mn)
                a, b = int(ts[i] + step), int(ts[i + 1])
                gaps.append({"dataset": "funding", "stream": "bybit", "symbol": sym, "gap_start": a, "gap_end": b, "n_missing": int(x // step - 1),
                             "kind": "missing_settlement", "cause": cause_of(a, b, down), "fillable_from": "bybit REST /v5/market/funding/history", "filled": ""})
    return {"misaligned_hour": mis, "misaligned_local_interval": len(misl), "misaligned_local_examples": misl[:10],
            "missing_settlement_runs": len(gaps)}, gaps


def scan_alt(con) -> tuple[list[dict], dict]:
    """Per (source, series_id), all in DuckDB (bounded memory)."""
    glob = (load().lake_dir / "alt_series").as_posix() + "/**/*.parquet"
    out = []
    pairs = con.sql(f"SELECT DISTINCT source, series_id FROM read_parquet('{glob}', hive_partitioning = true) ORDER BY 1, 2").fetchall()
    for src, sid in pairs:
        g = (load().lake_dir / "alt_series" / f"source={src}").as_posix() + "/**/*.parquet"
        t = f"""(SELECT entity, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, value, vintage, pit_quality,
                 epoch_ms(TRY_CAST(TRY_STRPTIME(vintage, '%Y-%m-%d') AS TIMESTAMP)) vms FROM read_parquet('{g}') WHERE series_id = '{sid}')"""
        r = con.sql(f"""SELECT count(*), count(DISTINCT entity), sum(CASE WHEN kn < ev THEN 1 ELSE 0 END), string_agg(DISTINCT pit_quality, ','),
                        sum(CASE WHEN vintage IS NULL THEN 1 ELSE 0 END), count(DISTINCT vintage), min(kn - ev), min(ev), max(ev),
                        sum(CASE WHEN ev > kn + {365 * DAY_MS} THEN 1 ELSE 0 END),
                        sum(CASE WHEN kn < greatest(coalesce(vms, ev), ev) + {DAY_MS} THEN 1 ELSE 0 END),
                        sum(CASE WHEN kn < ev + 3600000 THEN 1 ELSE 0 END) FROM {t}""").fetchall()[0]
        st = {"table": "alt_series", "stream": f"{src}/{sid}", "rows": r[0], "entities": r[1], "pit_known_before_event": int(r[2] or 0),
              "pit_quality": r[3], "vintage_null": int(r[4] or 0), "vintages": r[5], "known_minus_event_ms_min": r[6],
              "first": iso(r[7]), "last": iso(r[8]), "future_event_rows": int(r[9] or 0)}
        if src == "fred":
            st["pit_known_before_vintage_plus_1d"] = int(r[10] or 0)
            o = con.sql(f"""SELECT count(*), sum(CASE WHEN nv > 1 THEN 1 ELSE 0 END), sum(CASE WHEN nval > 1 THEN 1 ELSE 0 END) FROM
                           (SELECT entity, ev, count(DISTINCT vintage) nv, count(DISTINCT value) nval FROM {t} GROUP BY 1, 2)""").fetchall()[0]
            st["obs"], st["obs_with_2plus_vintages"], st["obs_revised_value"] = o[0], int(o[1] or 0), int(o[2] or 0)
        if src == "deribit" and sid == "dvol_close":
            st["pit_known_before_close_1h"] = int(r[11] or 0)
        key = "entity, ev, vintage" if src == "fred" else "entity, ev"
        dk = con.sql(f"SELECT count(*), coalesce(sum(c - 1), 0) FROM (SELECT {key}, count(*) c FROM {t} GROUP BY ALL HAVING count(*) > 1)").fetchall()[0]
        cf = con.sql(f"SELECT count(*) FROM (SELECT {key}, kn FROM {t} GROUP BY ALL HAVING count(DISTINCT value) > 1)").fetchall()[0][0]
        rv = con.sql(f"SELECT count(*) FROM (SELECT {key} FROM {t} GROUP BY ALL HAVING count(DISTINCT kn) > 1 AND count(DISTINCT value) > 1)").fetchall()[0][0]
        st.update({"dup_keys": dk[0], "dup_rows_extra": int(dk[1]), "dup_conflicting": cf, "revised_keys": rv})
        out.append(st)
    return out, {}


def fred_asof_check(n: int = 200, seed: int = 7) -> dict:
    """H-WS02-11: for revised FRED observations, Lake.read_asof one day after the first vintage returns the first release."""
    from ai_trader.storage.lake import Lake
    con = _con()
    glob = (load().lake_dir / "alt_series" / "source=fred").as_posix() + "/**/*.parquet"
    sids = [r[0] for r in con.sql(f"SELECT DISTINCT series_id FROM read_parquet('{glob}')").fetchall()]
    parts = []
    for sid in sids:  # per series: bounded memory (NFCI / ANFCI hold ~1M rows each)
        parts.append(pl.from_arrow(con.sql(f"""
            WITH t AS (SELECT series_id, entity, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, value FROM read_parquet('{glob}') WHERE series_id = '{sid}'),
            f AS (SELECT series_id, entity, ev, arg_min(value, kn) first_val, min(kn) first_kn FROM t GROUP BY ALL HAVING count(DISTINCT value) > 1),
            s AS (SELECT t.series_id, t.entity, t.ev, min(t.kn) second_kn FROM t JOIN f USING (series_id, entity, ev) WHERE t.kn > f.first_kn GROUP BY ALL)
            SELECT f.*, s.second_kn FROM f JOIN s USING (series_id, entity, ev)""").to_arrow_table()))
    rev = pl.concat([x for x in parts if x.height], how="vertical_relaxed")
    smp = rev.sample(min(n, rev.height), seed=seed) if rev.height else rev
    lake = Lake()
    ok = bad = 0
    bad_rows = []
    for r in smp.iter_rows(named=True):
        as_of = min(r["first_kn"] + DAY_MS, r["second_kn"] - 1)
        t = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=int(as_of))
        ev = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=int(r["ev"]))
        got = lake.read_asof("alt_series", t, start=ev, end=ev + timedelta(milliseconds=1),
                             filters={"source": "fred", "series_id": r["series_id"]})
        v = got.filter(pl.col("entity") == r["entity"])["value"]
        if v.len() == 1 and v[0] == r["first_val"]:
            ok += 1
        else:
            bad += 1
            bad_rows.append((r["series_id"], iso(r["ev"]), r["first_val"], v.to_list(), iso(as_of)))
    return {"revised_obs": rev.height, "checked": smp.height, "first_release_returned": ok, "wrong": bad, "wrong_examples": bad_rows[:10]}


def survivorship(con) -> dict:
    live = json.loads((out_dir() / "instruments_live_2026-11-26.json").read_text())
    rows = [r for r in live["rows"] if r["category"] == "linear"]
    trading_now = {r["symbol"] for r in rows if r["query_status"] == "default" and r["status"] == "Trading"}
    closed_now = {r["symbol"] for r in rows if r["status"] == "Closed" and r["contractType"] == "LinearPerpetual" and r["quoteCoin"] == "USDT"}
    glob = (load().lake_dir / "instruments").as_posix() + "/**/*.parquet"
    ins = pl.from_arrow(con.sql(f"""SELECT symbol, status, contract_type, quote, symbol_type, category, epoch_ms(event_ts) ev
        FROM read_parquet('{glob}', hive_partitioning = true)""").to_arrow_table())
    snaps = sorted(ins["ev"].unique().to_list())
    first = ins.filter((pl.col("ev") == snaps[0]) & (pl.col("category") == "linear") & (pl.col("status") == "Trading")
                       & (pl.col("contract_type") == "LinearPerpetual") & (pl.col("quote") == "USDT"))
    gone = sorted(set(first["symbol"]) - trading_now)
    bars_syms = set(pl.from_arrow(con.sql(f"""SELECT DISTINCT symbol FROM read_parquet('{(load().lake_dir / 'bars' / 'venue=bybit').as_posix()}/**/*.parquet',
        hive_partitioning = true)""").to_arrow_table())["symbol"].to_list())
    return {"snapshots": [iso(s) for s in snaps], "first_snapshot_trading_usdt_perps": first.height,
            "trading_now": len(trading_now), "closed_usdt_perps_listed_by_bybit": len(closed_now),
            "delisted_since_first_snapshot": gone, "bars_symbols": len(bars_syms),
            "bars_symbols_delisted_since_first_snapshot": sorted(bars_syms & set(gone)),
            "bars_symbols_closed_ever": sorted(bars_syms & closed_now),
            "bars_symbols_not_trading_now": sorted(bars_syms - trading_now)}


def run_tables() -> None:
    con = _con("450MB", 2)
    down = down_intervals(machine_windows())
    stats: list[dict] = []
    gaps: list[dict] = []
    s, g = scan_bars(con, down)
    stats += s
    gaps += g
    simple = [
        ("funding", "venue=bybit", None, ["rate"], "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, rate", "kn < ev", "wms < ev", ["symbol", "ev"], "bybit REST"),
        ("open_interest", "venue=bybit/interval=1h", 3_600_000, ["oi"], "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, open_interest oi", "kn < ev + 3600000", "wms < ev + 3600000", ["symbol", "ev"], "bybit REST /v5/market/open-interest"),
        ("open_interest", "venue=bybit/interval=4h", 14_400_000, ["oi"], "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, open_interest oi", "kn < ev + 14400000", "wms < ev + 14400000", ["symbol", "ev"], "bybit REST /v5/market/open-interest"),
        ("ls_ratio", "venue=bybit/period=1h", 3_600_000, ["b"], "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, buy_ratio b", "kn < ev + 3600000", "wms < ev + 3600000", ["symbol", "ev"], "bybit REST /v5/market/account-ratio"),
        ("ls_ratio", "venue=bybit/period=4h", 14_400_000, ["b"], "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, buy_ratio b", "kn < ev + 14400000", "wms < ev + 14400000", ["symbol", "ev"], "bybit REST /v5/market/account-ratio"),
        ("liquidations", "venue=bybit", None, ["p"], "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, side, price p, size z", "kn < ev", None, ["symbol", "ev", "side", "p", "z"], "none (WS only)"),
        ("instruments", "venue=bybit", None, ["status"], "symbol, category, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, status", "kn < ev", None, ["category", "symbol", "ev"], "bybit REST instruments-info (current only)"),
    ]
    for d in sorted((load().lake_dir / "flow_bars" / "venue=bybit").glob("freq=*")):
        f = d.name.split("=")[1]
        simple.append(("flow_bars", f"venue=bybit/freq={f}", FREQ_MS[f], ["bv", "sv"],
                       "symbol, epoch_ms(event_ts) ev, epoch_ms(known_ts) kn, epoch_ms(close_ts) ct, buy_volume bv, sell_volume sv",
                       "kn < ct", "wms < ct", ["symbol", "ev"], "public.bybit.com/trading"))
    extra: dict = {}
    for spec in simple:
        st, g, df = scan_simple(con, down, *spec)
        if spec[0] == "funding":
            fs, fg = funding_gaps(df, down)
            st.update(fs)
            g += fg
        if spec[0] == "liquidations":
            lag = df["kn"] - df["ev"]
            st["lag_ms_p50_p99_max"] = [float(lag.quantile(0.5)), float(lag.quantile(0.99)), int(lag.max())]
            extra["liq_per_day"] = (df.with_columns((pl.col("ev") // DAY_MS).alias("d")).group_by("d").len().sort("d")
                                    .with_columns(pl.col("d").map_elements(lambda x: iso(x * DAY_MS)[:10], return_dtype=pl.Utf8)).rows())
        stats.append(st)
        gaps += g
        del df
    st, extra["trades_raw_days"] = scan_trades_raw(con)
    stats.append(st)
    alt, _ = scan_alt(con)
    stats += alt
    tmp = out_dir() / "tmp"
    (tmp / "tables_stats.json").write_text(json.dumps({"stats": stats, "extra": extra}, default=str, indent=1))
    pl.DataFrame(gaps, schema=GAP_SCHEMA).write_parquet(tmp / "tables_gaps.parquet")
    del con
    extras(stats, extra)


def extras(stats: list | None = None, extra: dict | None = None) -> None:
    tmp = out_dir() / "tmp"
    if stats is None:
        ts = json.loads((tmp / "tables_stats.json").read_text())
        stats, extra = ts["stats"], ts["extra"]
    extra["survivorship"] = survivorship(_con("400MB", 2))
    extra["fred_asof"] = fred_asof_check()
    (tmp / "tables_stats.json").write_text(json.dumps({"stats": stats, "extra": extra}, default=str, indent=1))
    print(json.dumps(extra["fred_asof"]), json.dumps(extra["survivorship"])[:2000], flush=True)


# ---------------------------------------------------------------- fills from public.bybit.com trade archives

ARCHIVE = {"linear": "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz", "spot": "https://public.bybit.com/spot/{s}/{s}_{d}.csv.gz"}


def archive_candles(gz: Path, day: str) -> pl.DataFrame:
    """3 s candles (trade fields only, flow_3s semantics) from one archive day. Archive timestamps are seconds with 0.1 ms
    resolution; they are floored to ms like websocket `T`. Close is carried forward through trade-less candles; the grid
    covers the whole UTC day. known_ts = candle end + 500 ms (trades_raw backfill rule); pit_quality = backfill_archive."""
    con = _con("500MB", 2)
    d0 = ms(date.fromisoformat(day))
    c = pl.from_arrow(con.sql(f"""
        WITH t AS (SELECT CAST(round(timestamp * 10000) AS BIGINT) // 10 AS tms, row_number() OVER () AS seq, side, size, price
                   FROM read_csv('{gz.as_posix()}', header = true, columns = {{'timestamp': 'DOUBLE', 'symbol': 'VARCHAR', 'side': 'VARCHAR',
                   'size': 'DOUBLE', 'price': 'DOUBLE', 'tickDirection': 'VARCHAR', 'trdMatchID': 'VARCHAR', 'grossValue': 'DOUBLE',
                   'homeNotional': 'DOUBLE', 'foreignNotional': 'DOUBLE', 'RPI': 'VARCHAR'}}))
        SELECT (tms // 3000) * 3000 AS ev, arg_min(price, seq) AS open, max(price) AS high, min(price) AS low, arg_max(price, seq) AS close,
               sum(price * size) / sum(size) AS vwap,
               sum(CASE WHEN side = 'Buy' THEN size ELSE 0 END) AS buy_vol, sum(CASE WHEN side = 'Sell' THEN size ELSE 0 END) AS sell_vol,
               sum(CASE WHEN side = 'Buy' THEN size * price ELSE 0 END) AS buy_notional, sum(CASE WHEN side = 'Sell' THEN size * price ELSE 0 END) AS sell_notional,
               CAST(sum(CASE WHEN side = 'Buy' THEN 1 ELSE 0 END) AS INTEGER) AS n_buy, CAST(sum(CASE WHEN side = 'Sell' THEN 1 ELSE 0 END) AS INTEGER) AS n_sell,
               max(size) AS max_trade, min(seq) AS s0
        FROM t GROUP BY 1 ORDER BY 1""").to_arrow_table())
    grid = pl.DataFrame({"ev": np.arange(d0, d0 + DAY_MS, CANDLE_MS, dtype=np.int64)})
    out = grid.join(c, on="ev", how="left").sort("ev").with_columns(
        pl.col("close").forward_fill(),
        *[pl.col(k).fill_null(0) for k in ("buy_vol", "sell_vol", "buy_notional", "sell_notional", "n_buy", "n_sell", "max_trade")])
    return out.drop("s0").with_columns((pl.col("ev") + CANDLE_MS + 500).alias("kn"))


def fill_trades(symbols: list[str], days: list[str], category: str = "linear") -> list[dict]:
    """Download -> 3 s candles -> data/research/ws02/fills/flow_3s_trades/category=<c>/date=<d>/<symbol>.parquet; raw deleted."""
    import urllib.request
    raw = load().root / "data" / "raw" / "WS02"
    raw.mkdir(parents=True, exist_ok=True)
    base = out_dir() / "fills" / "flow_3s_trades" / f"category={category}"
    done = []
    for day in days:
        for sym in symbols:
            dst = base / f"date={day}" / f"{sym}.parquet"
            if dst.exists():
                continue
            url = ARCHIVE[category].format(s=sym, d=day)
            gz = raw / url.rsplit("/", 1)[1]
            try:
                urllib.request.urlretrieve(url, gz)
            except Exception as e:  # noqa: BLE001
                print(f"skip {sym} {day}: {e}", flush=True)
                continue
            mb = gz.stat().st_size / 1e6
            df = archive_candles(gz, day).with_columns(pl.lit(sym).alias("symbol"), pl.lit("backfill_archive").alias("pit_quality"))
            gz.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(dst, compression="zstd", compression_level=6)
            rec = {"dataset": "flow_3s", "stream": f"bybit/{category}", "symbol": sym, "day": day, "start": ms(date.fromisoformat(day)),
                   "end": ms(date.fromisoformat(day)) + DAY_MS, "path": dst.relative_to(load().root).as_posix(), "source": url,
                   "raw_mb": round(mb, 1), "out_mb": round(dst.stat().st_size / 1e6, 2), "trades": int(df["n_buy"].sum() + df["n_sell"].sum())}
            done.append(rec)
            print(json.dumps(rec), flush=True)
    man = out_dir() / "fills" / "manifest.json"
    old = json.loads(man.read_text())["fills"] if man.exists() else []
    keep = {(r["symbol"], r["day"], r["stream"]): r for r in old + done}
    man.write_text(json.dumps({"doc": "trade-field 3 s candles rebuilt from public.bybit.com archives; book fields absent; "
                                      "known_ts = candle end + 500 ms (backfill rule); use only inside gaps.parquet flow_3s holes",
                               "fills": sorted(keep.values(), key=lambda r: (r["day"], r["symbol"]))}, indent=1))
    return done


def compare_fill(symbol: str, day: str, windows: list[tuple[int, int]], category: str = "linear") -> dict:
    """H-WS02-9: archive candles vs flow_3s on covered windows (candles present in flow_3s, not empty-fill)."""
    f = pl.read_parquet(out_dir() / "fills" / "flow_3s_trades" / f"category={category}" / f"date={day}" / f"{symbol}.parquet")
    con = _con("600MB", 2)
    root = load().lake_dir / "flow_3s" / "venue=bybit" / f"category={category}" / f"date={day}"
    lake = pl.from_arrow(con.sql(f"""SELECT epoch_ms(event_ts) ev, n_buy + n_sell AS n, CAST(buy_vol AS DOUBLE) bv, CAST(sell_vol AS DOUBLE) sv, book_cov, epoch_ms(known_ts) kn
        FROM read_parquet('{root.as_posix()}/*.parquet') WHERE symbol = '{symbol}'""").to_arrow_table())
    lake = lake.sort("ev", "kn").unique("ev", keep="last")
    j = lake.join(f.select("ev", (pl.col("n_buy") + pl.col("n_sell")).alias("n_a"), pl.col("buy_vol").alias("bv_a"), pl.col("sell_vol").alias("sv_a")), on="ev")
    j = j.filter(pl.any_horizontal([(pl.col("ev") >= a) & (pl.col("ev") < b) for a, b in windows]) & ~((pl.col("n") == 0) & (pl.col("book_cov") <= 0)))
    rel = lambda x, y: (pl.col(x) - pl.col(y)).abs() <= 1e-6 * pl.col(y).abs() + 1e-9  # noqa: E731
    r = j.select(pl.len().alias("candles"), (pl.col("n") == pl.col("n_a")).mean().alias("count_match"),
                 (rel("bv", "bv_a") & rel("sv", "sv_a")).mean().alias("vol_match_1e6"),
                 (((pl.col("bv") - pl.col("bv_a")).abs() + (pl.col("sv") - pl.col("sv_a")).abs()) <= 1e-4 * (pl.col("bv_a") + pl.col("sv_a")) + 1e-9).mean().alias("vol_match_1e4"),
                 pl.col("n").sum().alias("trades_lake"), pl.col("n_a").sum().alias("trades_archive"),
                 (pl.col("bv") + pl.col("sv")).sum().alias("vol_lake"), (pl.col("bv_a") + pl.col("sv_a")).sum().alias("vol_archive")).row(0, named=True)
    return {"symbol": symbol, "day": day, **r}


def _kline_hours(con) -> pl.DataFrame:
    return pl.from_arrow(con.sql(f"""SELECT symbol, epoch_ms(event_ts) AS "hour", arg_max(volume, known_ts) kv, arg_max(close, known_ts) kc
        FROM read_parquet('{(load().lake_dir / 'bars' / 'venue=bybit' / 'price_type=last' / 'freq=1h').as_posix()}/year=2026/*.parquet')
        WHERE event_ts >= TIMESTAMP '2026-11-07' GROUP BY 1, 2""").to_arrow_table())


def trade_deficit_hours(types: dict[str, str], wins: list[dict], thresh: float = 0.9) -> pl.DataFrame:
    """Silent partial feed: linear crypto symbol-hours with >= 600 candles present whose flow_3s taker volume is below
    `thresh` x the kline volume scaled by the share of candles present (trades trickle while the book stream is alive)."""
    h = pl.concat([pl.read_parquet(p) for p in sorted((out_dir() / "tmp" / "flow3s").glob("linear_*_hourly.parquet"))], how="diagonal")
    h = h.filter((pl.col("rows") >= 600) & pl.col("symbol").map_elements(lambda s: types.get(s, "x") == "", return_dtype=pl.Boolean))
    j = h.join(_kline_hours(_con("500MB", 2)), on=["symbol", "hour"]).filter(pl.col("kv") > 0)
    j = j.filter(pl.col("vol") < thresh * pl.col("kv") * pl.col("rows") / 1200)
    return j.select(pl.lit("flow_3s").alias("dataset"), pl.lit("bybit/linear").alias("stream"), "symbol", pl.col("hour").alias("gap_start"),
                    (pl.col("hour") + 3_600_000).alias("gap_end"), pl.lit(0, dtype=pl.Int64).alias("n_missing"), pl.lit("trade_deficit").alias("kind"),
                    pl.struct("hour").map_elements(lambda r: cause_kind(r["hour"], r["hour"] + 3_600_000, wins) or "collector_or_ws", return_dtype=pl.Utf8).alias("cause"),
                    pl.lit("public.bybit.com/trading (trade fields)").alias("fillable_from"), pl.lit("").alias("filled"))


def flow_vs_klines() -> dict:
    """H-WS02-8: fully covered flow_3s symbol-hours (1200 candles, no empty-fill run) vs bybit last 1h klines."""
    d = out_dir()
    types = symbol_types()
    h = pl.concat([pl.read_parquet(p) for p in sorted((d / "tmp" / "flow3s").glob("linear_*_hourly.parquet"))], how="diagonal")
    h = h.filter((pl.col("rows") == 1200) & pl.col("symbol").map_elements(lambda s: types.get(s, "x") == "", return_dtype=pl.Boolean))
    g = pl.read_parquet(d / "gaps.parquet").filter((pl.col("dataset") == "flow_3s") & (pl.col("stream") == "bybit/linear") & (pl.col("kind") == "empty_fill"))
    bad = set()
    for sym, a, b in g.select("symbol", "gap_start", "gap_end").rows():
        for hr in range(a // 3_600_000 * 3_600_000, b, 3_600_000):
            bad.add((sym, hr))
    h = h.filter(~pl.struct("symbol", "hour").map_elements(lambda r: (r["symbol"], r["hour"]) in bad, return_dtype=pl.Boolean))
    j = h.join(_kline_hours(_con("500MB", 2)), on=["symbol", "hour"], how="inner")
    j = j.with_columns(((pl.col("vol") - pl.col("kv")).abs() <= 0.005 * pl.col("kv")).alias("vok"),
                       ((pl.col("last_close") - pl.col("kc")).abs() <= 1e-9 * pl.col("kc")).alias("cok"),
                       (pl.col("vol") / pl.col("kv") - 1).alias("vrel"))
    worst = j.filter(~pl.col("vok")).sort((pl.col("vrel")).abs(), descending=True).select("symbol", "hour", "vol", "kv", "vrel").head(8)
    r = {"symbol_hours": j.height, "symbols": j["symbol"].n_unique(), "vol_share_within_0.5pct": float(j["vok"].mean()),
         "close_share_equal": float(j["cok"].mean()), "median_abs_vol_rel": float(j["vrel"].abs().median()),
         "vol_rel_p01_p99": [float(j["vrel"].quantile(0.01)), float(j["vrel"].quantile(0.99))],
         "worst": [(a, iso(b), c, e, f) for a, b, c, e, f in worst.rows()]}
    r["verdict"] = ("supports" if r["vol_share_within_0.5pct"] >= 0.99 and r["close_share_equal"] >= 0.99
                    else ("contradicts" if r["vol_share_within_0.5pct"] < 0.95 else "inconclusive"))
    return r


def symbol_types() -> dict[str, str]:
    """'' = crypto, else Bybit symbolType (stock / commodity / ETF / forex ...), from the fresh public listing."""
    p = out_dir() / "instruments_live_2026-11-26.json"
    rows = json.loads(p.read_text())["rows"] if p.exists() else []
    return {r["symbol"]: r.get("symbolType") or "" for r in rows if r["category"] == "linear"}


def merge_gap_rows(g: pl.DataFrame) -> pl.DataFrame:
    """Join holes that touch (e.g. across day partitions) within dataset/stream/symbol/kind."""
    if g.height == 0:
        return g
    g = g.sort("dataset", "stream", "symbol", "kind", "gap_start")
    new_run = ((pl.col("gap_start") > pl.col("gap_end").shift(1).over("dataset", "stream", "symbol", "kind"))
               | pl.col("gap_end").shift(1).over("dataset", "stream", "symbol", "kind").is_null())
    g = g.with_columns(new_run.cum_sum().over("dataset", "stream", "symbol", "kind").alias("_r"))
    return (g.group_by("dataset", "stream", "symbol", "kind", "_r", maintain_order=True)
            .agg(pl.col("gap_start").min(), pl.col("gap_end").max(), pl.col("n_missing").sum(),
                 pl.col("cause").first(), pl.col("fillable_from").first(), pl.col("filled").first())
            .drop("_r"))


def flow3s_gaps(down, types: dict[str, str]) -> tuple[pl.DataFrame, dict]:
    tmp = out_dir() / "tmp" / "flow3s"
    cov = pl.concat([pl.read_parquet(p) for p in sorted(tmp.glob("*_coverage.parquet"))], how="diagonal")
    gp = [pl.read_parquet(p) for p in sorted(tmp.glob("*_gaps.parquet"))]
    gaps = pl.concat([g for g in gp if g.height], how="diagonal")
    days = sorted(cov["day"].unique().to_list())
    all_days = [str(date.fromisoformat(days[0]) + timedelta(days=i)) for i in range((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days + 1)]
    rows = []
    first_last = cov.group_by("category", "symbol").agg(pl.col("day").min().alias("d0"), pl.col("day").max().alias("d1"))
    have = set(zip(cov["category"], cov["symbol"], cov["day"], strict=True))
    for cat, sym, d0, d1 in first_last.rows():
        for d in all_days:
            if d0 <= d <= d1 and (cat, sym, d) not in have:   # whole day partition absent
                a = ms(date.fromisoformat(d))
                rows.append({"category": cat, "symbol": sym, "gap_start": a, "gap_end": a + DAY_MS, "n_missing": DAY_MS // CANDLE_MS,
                             "kind": "missing_rows", "day": d})
    if rows:
        gaps = pl.concat([gaps, pl.DataFrame(rows)], how="diagonal")
    gaps = gaps.with_columns(pl.concat_str([pl.lit("bybit/"), pl.col("category")]).alias("stream"), pl.lit("flow_3s").alias("dataset"))
    # trim: nothing before a symbol's first candle ever (stream added later) or after its last (stream removed)
    span = (cov.join(pl.DataFrame({"day": days}), on="day").group_by("category", "symbol").agg(pl.col("day").min().alias("d0"), pl.col("day").max().alias("d1")))
    firsts = {}
    for cat, sym, d0, d1 in span.rows():
        g = gaps.filter((pl.col("category") == cat) & (pl.col("symbol") == sym) & (pl.col("day") == d0) & (pl.col("kind") == "missing_rows")
                        & (pl.col("gap_start") == ms(date.fromisoformat(d0))))
        firsts[(cat, sym)] = int(g["gap_end"].max()) if g.height else ms(date.fromisoformat(d0))
    gaps = gaps.with_columns(pl.struct("category", "symbol").map_elements(lambda s: firsts.get((s["category"], s["symbol"]), 0), return_dtype=pl.Int64).alias("_first"))
    gaps = gaps.filter(pl.col("gap_end") > pl.col("_first")).with_columns(pl.max_horizontal("gap_start", "_first").alias("gap_start"))
    gaps = gaps.with_columns(((pl.col("gap_end") - pl.col("gap_start")) // CANDLE_MS).alias("n_missing"))

    wins = machine_windows()

    def cause(r):
        c = cause_kind(r["gap_start"], r["gap_end"], wins)
        if c:
            return c
        return "venue_session" if types.get(r["symbol"], "") else "collector_or_ws"
    gaps = gaps.with_columns(pl.struct("gap_start", "gap_end", "symbol").map_elements(cause, return_dtype=pl.Utf8).alias("cause"),
                             pl.when(pl.col("category") == "linear").then(pl.lit("public.bybit.com/trading (trade fields); quote-saver ob200 (book)"))
                             .otherwise(pl.lit("public.bybit.com/spot (trade fields); quote-saver ob200 spot (book)")).alias("fillable_from"),
                             pl.lit("").alias("filled"))
    out = merge_gap_rows(gaps.select(list(GAP_SCHEMA)))
    return out, {"first_ms": firsts, "coverage": cov, "days": all_days}


def assemble() -> None:
    d = out_dir()
    wins = machine_windows()
    down = down_intervals(wins)
    types = symbol_types()
    f3, info = flow3s_gaps(down, types)
    tgp = d / "tmp" / "tables_gaps.parquet"
    tg = pl.read_parquet(tgp) if tgp.exists() else pl.DataFrame(schema=GAP_SCHEMA)
    mach = pl.DataFrame([{"dataset": "machine", "stream": w["kind"], "symbol": "*", "gap_start": w["down_start"], "gap_end": w["margin_end"],
                          "n_missing": 0, "kind": "machine_down", "cause": {"sleep": "pc_suspend", "reboot": "reboot"}.get(w["kind"], "modern_standby"),
                          "fillable_from": "", "filled": ""} for w in wins], schema=GAP_SCHEMA)
    # collector down = BTCUSDT linear holes (always collected, never quiet)
    btc = f3.filter((pl.col("symbol") == "BTCUSDT") & (pl.col("stream") == "bybit/linear"))
    coll = btc.with_columns(pl.lit("collector").alias("dataset"), pl.lit("bybit/ws").alias("stream"), pl.lit("*").alias("symbol"))
    liq = btc.with_columns(pl.lit("liquidations").alias("dataset"), pl.lit("bybit/allLiquidation").alias("stream"), pl.lit("*").alias("symbol"),
                           pl.lit("none (WS only; no REST history)").alias("fillable_from"))
    td = trade_deficit_hours(types, wins)
    fills = d / "fills" / "manifest.json"
    tg = tg.with_columns(pl.struct("gap_start", "gap_end", "cause").map_elements(
        lambda r: cause_kind(r["gap_start"], r["gap_end"], wins) or ("unknown" if r["cause"] == "machine_down" else r["cause"]), return_dtype=pl.Utf8).alias("cause"))
    gaps = pl.concat([x.select(list(GAP_SCHEMA)) for x in (mach, coll, liq, f3, td, tg)], how="vertical_relaxed")
    if fills.exists():
        man = json.loads(fills.read_text())
        for f in man.get("fills", []):
            hit = ((pl.col("dataset") == f["dataset"]) & (pl.col("symbol") == f["symbol"]) & (pl.col("stream") == f["stream"])
                   & (pl.col("gap_start") < f["end"]) & (pl.col("gap_end") > f["start"]))
            gaps = gaps.with_columns(pl.when(hit).then(pl.lit(f["path"])).otherwise(pl.col("filled")).alias("filled"))
    gaps = gaps.with_columns(((pl.col("gap_end") - pl.col("gap_start")) / 1000).alias("duration_s"),
                             pl.from_epoch("gap_start", time_unit="ms").dt.replace_time_zone("UTC").alias("start_utc"),
                             pl.from_epoch("gap_end", time_unit="ms").dt.replace_time_zone("UTC").alias("end_utc"))
    gaps = gaps.sort("dataset", "stream", "symbol", "gap_start")
    gaps.write_parquet(d / "gaps.parquet")
    cov = info["coverage"]
    covj: dict = {"generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                  "doc": "research/reports/ws02_2026-11-27.md; times UTC; flow_3s day rows = [present, expected, empty_fill, awake_present, awake_expected]",
                  "machine_windows": wins, "flow_3s": {}, "tables": {}}
    for (cat, sym), s in cov.group_by("category", "symbol", maintain_order=True):
        s = s.sort("day")
        covj["flow_3s"].setdefault(cat, {})[sym] = {
            "symbol_type": types.get(sym, "") if cat == "linear" else "spot", "first_candle_utc": iso(info["first_ms"].get((cat, sym), 0)),
            "days": {r["day"]: [r["present"], r["expected"], r["empty"], r["awake_present"], r["awake_expected"]] for r in s.iter_rows(named=True)}}
    tsp = d / "tmp" / "tables_stats.json"
    ts = json.loads(tsp.read_text()) if tsp.exists() else {"stats": [], "extra": {}}
    covj["tables"] = ts["stats"]
    covj["extra"] = {k: v for k, v in ts["extra"].items() if k in ("fred_asof", "survivorship")}
    covj["flow_3s_scan"] = [json.loads(p.read_text()) for p in sorted((d / "tmp" / "flow3s").glob("*_stats.json"))]
    (d / "coverage.json").write_text(json.dumps(covj, default=str))
    print(gaps.group_by("dataset", "cause").agg(pl.len(), (pl.col("duration_s").sum() / 3600).alias("hours")).sort("dataset"))


def verdicts() -> dict:
    """Evaluate H-WS02-1..12 by the registered rules from the scan outputs."""
    d = out_dir()
    ts = json.loads((d / "tmp" / "tables_stats.json").read_text())
    st, ex = ts["stats"], ts["extra"]
    f3 = [json.loads(p.read_text()) for p in sorted((d / "tmp" / "flow3s").glob("*_stats.json"))]
    gaps = pl.read_parquet(d / "gaps.parquet")
    types = symbol_types()
    live = load().settings["live"]["symbols"]
    crypto_live = [s for s in live if types.get(s, "") == ""]
    down = down_intervals(machine_windows())
    out: dict = {}
    bars = [s for s in st if s["table"] == "bars"]
    v1 = sum(s["pit_known_before_close"] + s["bad_duration"] + s["off_grid"] for s in bars)
    out["H-WS02-1"] = {"violations": v1, "per_stream": {s["stream"]: [s["pit_known_before_close"], s["bad_duration"], s["off_grid"]] for s in bars},
                       "verdict": "supports" if v1 == 0 else "contradicts"}
    v2 = sum(s["written_before_close"] for s in bars)
    out["H-WS02-2"] = {"rows_written_before_close": v2, "no_write_ms": sum(s["no_write_ms"] for s in bars),
                       "per_stream": {s["stream"]: s["written_before_close"] for s in bars if s["written_before_close"]},
                       "verdict": "supports" if v2 == 0 else "contradicts"}
    v3 = {k: sum(x[k] for x in f3) for k in ("pit_grid", "pit_known_before_end", "pit_written_before_known", "rows")}
    v3["written_before_known_max_ms"] = max(x.get("written_before_known_max_ms", 0) for x in f3)
    out["H-WS02-3"] = {**v3, "verdict": "supports" if v3["pit_grid"] + v3["pit_known_before_end"] + v3["pit_written_before_known"] == 0 else "contradicts"}
    v4 = {}
    for s in st:
        if s["table"] in ("funding", "open_interest", "ls_ratio", "liquidations", "trades_raw", "instruments", "flow_bars"):
            v4[f"{s['table']}/{s['stream']}"] = s["pit_violations"]
        if s["table"] == "alt_series":
            v4[f"alt/{s['stream']}"] = s["pit_known_before_event"] + s.get("pit_known_before_vintage_plus_1d", 0) + s.get("pit_known_before_close_1h", 0)
    out["H-WS02-4"] = {"per_table": {k: v for k, v in v4.items() if v}, "total": sum(v4.values()), "tables_checked": len(v4),
                       "verdict": "supports" if sum(v4.values()) == 0 else "contradicts"}
    conf = {f"{s['table']}/{s['stream']}": s["dup_conflicting"] for s in st if s.get("dup_conflicting")}
    conf_f3 = sum(x["dup_conflicting"] for x in f3)
    exact = {f"{s['table']}/{s['stream']}": s["dup_rows_extra"] for s in st if s.get("dup_rows_extra")}
    out["H-WS02-5"] = {"conflicting": conf, "flow_3s_conflicting": conf_f3, "exact_extra_rows": exact,
                       "revised_keys": {f"{s['table']}/{s['stream']}": s["revised_keys"] for s in st if s.get("revised_keys")},
                       "verdict": "supports" if not conf and conf_f3 == 0 else "contradicts"}
    a, b = ms(date(2026, 11, 7)), ms(date(2026, 11, 26))
    bg = gaps.filter((pl.col("dataset") == "bars") & pl.col("stream").is_in(["bybit/last/1h", "bybit/last/4h", "bybit/last/1d", "bybit/mark/1h"])
                     & (pl.col("gap_end") > a) & (pl.col("gap_start") < b))
    bgc = bg.filter(pl.col("symbol").is_in(crypto_live))
    out["H-WS02-6"] = {"missing_crypto_bars": int(bgc["n_missing"].sum() or 0), "crypto_runs": bgc.height,
                       "examples": [(r[0], r[1], str(r[2]), r[3]) for r in bgc.select("symbol", "stream", "start_utc", "n_missing").head(10).rows()],
                       "missing_tradfi_bars": int(bg.filter(~pl.col("symbol").is_in(crypto_live))["n_missing"].sum() or 0),
                       "verdict": "supports" if bgc.height == 0 else "contradicts"}
    # H-WS02-7 (registered windows: sleep + reboot; post hoc variant adds Modern Standby)
    wins = machine_windows()
    f3g = gaps.filter((pl.col("dataset") == "flow_3s") & (pl.col("stream") == "bybit/linear") & (pl.col("kind") == "missing_rows"))
    cj = json.loads((d / "coverage.json").read_text())["flow_3s"]["linear"]
    firsts = {s: ms(datetime.fromisoformat(v["first_candle_utc"]).replace(tzinfo=UTC)) for s, v in cj.items()}
    by_sym = {sym: list(zip(g["gap_start"].to_list(), g["gap_end"].to_list(), strict=True)) for (sym,), g in f3g.group_by("symbol")}
    for label, kinds in (("registered", ("sleep", "reboot")), ("post_hoc_with_standby", ("sleep", "reboot", "modern_standby"))):
        dn = down_intervals(wins, kinds)
        rows = []
        for sym, f in firsts.items():
            if types.get(sym, "") != "":
                continue
            for i in range(18):
                d0 = ms(date(2026, 11, 8) + timedelta(days=i))
                a0 = max(d0, f)
                if a0 >= d0 + DAY_MS:
                    continue
                awake = [(x, y) for x, y in _complement(a0, d0 + DAY_MS, dn)]
                aw = sum(y - x for x, y in awake)
                if aw < 6 * 3_600_000:
                    continue
                miss = sum(overlap_ms(x, y, awake) for x, y in by_sym.get(sym, []))
                rows.append((sym, str(date(2026, 11, 8) + timedelta(days=i)), round(1 - miss / aw, 5), round(aw / 3.6e6, 2)))
        share = float(np.mean([x[2] >= 0.99 for x in rows])) if rows else float("nan")
        out.setdefault("H-WS02-7", {})[label] = {
            "stream_days": len(rows), "share_ge_99pct": share, "median_awake_coverage": float(np.median([x[2] for x in rows])) if rows else None,
            "worst": sorted(rows, key=lambda x: x[2])[:8],
            "verdict": "supports" if share >= 0.95 else ("contradicts" if share < 0.90 else "inconclusive")}
    out["H-WS02-7"]["verdict"] = out["H-WS02-7"]["registered"]["verdict"]
    out["H-WS02-8"] = flow_vs_klines()
    # H-WS02-9: BTC / ETH on one overlap day; windows = day minus (flow_3s holes of the symbol +- 60 s)
    day = "2026-11-24"
    d0 = ms(date.fromisoformat(day))
    res = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        hs = gaps.filter((pl.col("dataset") == "flow_3s") & (pl.col("stream") == "bybit/linear") & (pl.col("symbol") == sym))
        holes = [(a - 60_000, b + 60_000) for a, b in hs.select("gap_start", "gap_end").rows()]
        if (out_dir() / "fills" / "flow_3s_trades" / "category=linear" / f"date={day}" / f"{sym}.parquet").exists():
            res.append(compare_fill(sym, day, _complement(d0, d0 + DAY_MS, holes)))
    ok = res and all(r["count_match"] >= 0.98 and r["vol_match_1e6"] >= 0.98 for r in res)
    badv = any(r["count_match"] < 0.95 or r["vol_match_1e6"] < 0.95 for r in res)
    out["H-WS02-9"] = {"per_symbol": res, "verdict": "supports" if ok else ("contradicts" if badv else "inconclusive")}
    # H-WS02-10 survivorship
    sv = ex.get("survivorship", {})
    gone = sv.get("delisted_since_first_snapshot", [])
    out["H-WS02-10"] = {"delisted_since_first_snapshot": len(gone), "examples": gone[:15],
                        "bars_symbols_delisted": sv.get("bars_symbols_delisted_since_first_snapshot"),
                        "closed_usdt_perps_listed_by_bybit": sv.get("closed_usdt_perps_listed_by_bybit"), "snapshots": sv.get("snapshots"),
                        "verdict": "supports" if gone and not sv.get("bars_symbols_delisted_since_first_snapshot") else "contradicts"}
    # H-WS02-11 FRED vintages
    fr = [s for s in st if s["table"] == "alt_series" and s["stream"].startswith("fred/")]
    fa = ex.get("fred_asof", {})
    c1 = all(s["vintage_null"] == 0 and s["pit_quality"] == "vintage" for s in fr)
    c2 = any(s.get("obs_with_2plus_vintages", 0) > 0 for s in fr)
    c3 = fa.get("checked", 0) > 0 and fa.get("wrong", 1) == 0
    out["H-WS02-11"] = {"series": len(fr), "all_vintage_nonnull": c1, "any_2plus_vintages": c2, "asof_first_release": fa,
                        "pit_known_before_vintage_plus_1d": sum(s.get("pit_known_before_vintage_plus_1d", 0) for s in fr),
                        "verdict": "supports" if c1 and c2 and c3 else "contradicts"}
    # H-WS02-12 funding
    fs = next(s for s in st if s["table"] == "funding")
    fg = gaps.filter((pl.col("dataset") == "funding") & pl.col("symbol").is_in(crypto_live) & (pl.col("gap_start") < ms(date(2026, 11, 26))))
    out["H-WS02-12"] = {"misaligned_hour": fs.get("misaligned_hour"), "misaligned_local_interval": fs.get("misaligned_local_interval"),
                        "misaligned_examples": fs.get("misaligned_local_examples"), "missing_runs_crypto_live": fg.height,
                        "missing_settlements": int(fg["n_missing"].sum() or 0),
                        "examples": [(a, str(b), c) for a, b, c in fg.select("symbol", "start_utc", "n_missing").head(10).rows()],
                        "verdict": "supports" if fg.height == 0 and not fs.get("misaligned_local_interval") else ("contradicts" if fg.height else "inconclusive")}
    return out


# ---------------------------------------------------------------- CLI

def _cli(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="data_quality")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("machine")
    f = sub.add_parser("flow3s")
    f.add_argument("--days", nargs="*")
    f.add_argument("--category", nargs="*", default=["linear", "spot"])
    sub.add_parser("tables")
    sub.add_parser("assemble")
    fl = sub.add_parser("fill")
    fl.add_argument("--symbols", nargs="+", required=True)
    fl.add_argument("--days", nargs="+", required=True)
    fl.add_argument("--category", default="linear")
    sub.add_parser("verdicts")
    sub.add_parser("extras")
    a = ap.parse_args(argv)
    if a.cmd == "machine":
        for w in machine_windows(refresh=True):
            sys.stdout.write(f"{w['kind']:7s} {w['start_utc']} -> {w['end_utc']}  {w['hours']} h\n")
    elif a.cmd == "flow3s":
        run_flow3s(a.days, a.category)
    elif a.cmd == "tables":
        run_tables()
    elif a.cmd == "assemble":
        assemble()
    elif a.cmd == "fill":
        fill_trades(a.symbols, a.days, a.category)
    elif a.cmd == "extras":
        extras()
    elif a.cmd == "verdicts":
        v = verdicts()
        (out_dir() / "verdicts.json").write_text(json.dumps(v, default=str, indent=1))
        sys.stdout.write(json.dumps({k: x.get("verdict") for k, x in v.items()}) + chr(10))


if __name__ == "__main__":
    _cli()
