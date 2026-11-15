"""Phase 1: downloads into ``data/staging`` with a manifest (SPEC sections 8 and 9).

Crypto: Binance USDT-M futures 5m klines from the public archive (data.binance.vision), monthly zips plus the
daily zips of every month after the last published monthly zip. One archive file at a time: download the zip,
verify it against the archive's ``.CHECKSUM`` (sha256), convert it to a staging parquet, delete the zip, and
record it in ``data/manifest.json``. A file already in the manifest is never downloaded again.

Kline CSV columns: open_time, open, high, low, close, volume, close_time, quote_volume, count,
taker_buy_volume, taker_buy_quote_volume, ignore (newer files have a header row, older ones do not).
Mapping: ``ts_close = open_time + 5m``; ``buy_v`` = taker_buy_volume (taker-buy base volume, the true aggressor
volume); ``sell_v = volume - buy_v``, clipped at 0 when float rounding makes it negative (counted).

CFDs: ``atx ctrader symbols --resolve`` (names) and ``atx ctrader trendbars`` (m5 history) write into
``data/staging/ctrader_raw``; ``stage_ctrader`` converts those monthly files to the staging schema
(``v`` = tick volume, no ``buy_v``/``sell_v``) and records the symbol mapping in the manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from vecopen.config import PROJECT_ROOT
from vecopen.data import DATA_DIR, MANIFEST_PATH, RAW_DIR, STAGING_DIR, TS

REPO_ROOT = PROJECT_ROOT.parent
ARCHIVE_URL = "https://data.binance.vision"
LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
BUDGET_GB = 30.0
PHASE1_ADD_CAP_GB = 1.5
FIVE_MIN_MS = 300_000
USER_AGENT = "vecopen-phase1/1 (research; one file at a time)"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- manifest

class Manifest:
    """``data/manifest.json``: every archive file, conversion and split, written atomically after each change."""

    def __init__(self, path: Path | None = None):
        path = path if path is not None else MANIFEST_PATH
        self.path = path
        if path.exists():
            self.d: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.d = {"schema": 1, "created_utc": utc_now(), "files": {}, "assets": {}, "storage_checks": [],
                      "ctrader": {}}

    @property
    def files(self) -> dict[str, Any]:
        return self.d["files"]

    @property
    def assets(self) -> dict[str, Any]:
        return self.d["assets"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- storage

def dir_bytes(path: Path) -> int:
    """Total size of the regular files under ``path`` (unreadable folders, i.e. the locked holdout, are skipped
    and the holdout's size is taken from the manifest instead)."""
    total = 0
    stack = [path]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def storage_check(manifest: Manifest, planned_bytes: int) -> dict[str, Any]:
    """Measure the repo's ``data/`` and ``vecopen/data/`` and refuse if the plan would break the budget."""
    t0 = time.time()
    repo = dir_bytes(REPO_ROOT / "data")
    vec = dir_bytes(DATA_DIR)
    hold = sum(a.get("holdout_bytes", 0) for a in manifest.assets.values())
    used_gb = (repo + vec + hold) / 1e9
    peak_add_gb = 2 * planned_bytes / 1e9  # staging plus research/holdout before staging is deleted
    rec = {"utc": utc_now(), "repo_data_gb": round(repo / 1e9, 3), "vecopen_data_gb": round((vec + hold) / 1e9, 3),
           "planned_download_gb": round(planned_bytes / 1e9, 3), "peak_add_gb": round(peak_add_gb, 3),
           "budget_gb": BUDGET_GB, "secs": round(time.time() - t0, 1)}
    ok = used_gb + peak_add_gb < BUDGET_GB and planned_bytes / 1e9 < PHASE1_ADD_CAP_GB
    rec["ok"] = ok
    manifest.d["storage_checks"].append(rec)
    manifest.save()
    if not ok:
        raise RuntimeError(f"storage check failed: {rec}")
    return rec


# ---------------------------------------------------------------- HTTP

def http_get(url: str, *, timeout: float = 60.0, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def list_keys(prefix: str, get: Callable[[str], bytes] = http_get) -> list[tuple[str, int]]:
    """All (key, size) under ``prefix`` in the archive bucket (S3 ListObjects v1, paged by marker)."""
    out: list[tuple[str, int]] = []
    marker = ""
    while True:
        q = {"delimiter": "/", "prefix": prefix}
        if marker:
            q["marker"] = marker
        root = ET.fromstring(get(f"{LISTING_URL}?{urllib.parse.urlencode(q)}"))
        for c in root.findall(f"{S3_NS}Contents"):
            out.append((c.findtext(f"{S3_NS}Key") or "", int(c.findtext(f"{S3_NS}Size") or 0)))
        if (root.findtext(f"{S3_NS}IsTruncated") or "false").lower() != "true" or not out:
            return out
        marker = out[-1][0]


# ---------------------------------------------------------------- Binance plan / convert

def _month_iter(first: str, last: str) -> Iterable[str]:
    y, m = int(first[:4]), int(first[5:7])
    ly, lm = int(last[:4]), int(last[5:7])
    while (y, m) <= (ly, lm):
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def binance_plan(symbol: str, today: date, get: Callable[[str], bytes] = http_get) -> list[dict[str, Any]]:
    """Archive files for ``symbol``: every monthly zip, then the daily zips of months after the last monthly one
    (up to yesterday; today's file does not exist yet)."""
    base = f"data/futures/um/monthly/klines/{symbol}/5m/"
    monthly = [(k, s) for k, s in list_keys(base, get) if k.endswith(".zip")]
    plan = [{"key": k, "size": s, "kind": "monthly", "period": k[-11:-4]} for k, s in sorted(monthly)]
    last_month = plan[-1]["period"] if plan else None
    start = last_month if last_month else f"{today.year:04d}-{today.month:02d}"
    months_after = [m for m in _month_iter(start, f"{today.year:04d}-{today.month:02d}") if last_month is None or m > last_month]
    for m in months_after:
        dprefix = f"data/futures/um/daily/klines/{symbol}/5m/{symbol}-5m-{m}"
        daily = [(k, s) for k, s in list_keys(dprefix, get) if k.endswith(".zip")]
        plan += [{"key": k, "size": s, "kind": "daily", "period": k[-14:-4]} for k, s in sorted(daily)
                 if k[-14:-4] < today.isoformat()]
    return plan


def parse_klines(csv_bytes: bytes) -> tuple[pl.DataFrame, dict[str, int]]:
    """Kline CSV -> staging frame (ts_close, o, h, l, c, v, buy_v, sell_v) plus quality counters."""
    has_header = csv_bytes[:9].lower() == b"open_time"
    raw = pl.read_csv(io.BytesIO(csv_bytes), has_header=has_header, new_columns=KLINE_COLS,
                      schema_overrides={"open_time": pl.Int64, "close_time": pl.Int64, "open": pl.Float64,
                                        "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                                        "volume": pl.Float64, "taker_buy_volume": pl.Float64})
    ot = pl.col("open_time")
    # millisecond timestamps (futures archive); microseconds would be > 1e14 in this era
    if raw.height and raw["open_time"].max() > 10**14:
        raw = raw.with_columns(open_time=ot // 1000, close_time=pl.col("close_time") // 1000)
    q = {
        "rows": raw.height,
        "off_grid": int(raw.select((ot % FIVE_MIN_MS != 0).sum()).item() or 0),
        "close_time_mismatch": int(raw.select((pl.col("close_time") != ot + FIVE_MIN_MS - 1).sum()).item() or 0),
        "ohlc_violations": int(raw.select(((pl.col("high") < pl.max_horizontal("open", "close")) |
                                           (pl.col("low") > pl.min_horizontal("open", "close")) |
                                           (pl.col("low") > pl.col("high"))).sum()).item() or 0),
        "buy_gt_volume": int(raw.select((pl.col("taker_buy_volume") > pl.col("volume")).sum()).item() or 0),
        "zero_volume": int(raw.select((pl.col("volume") == 0).sum()).item() or 0),
    }
    df = raw.filter(ot % FIVE_MIN_MS == 0).select(
        ts_close=pl.from_epoch(ot + FIVE_MIN_MS, time_unit="ms").dt.replace_time_zone("UTC").cast(TS),
        o=pl.col("open"), h=pl.col("high"), l=pl.col("low"), c=pl.col("close"), v=pl.col("volume"),
        buy_v=pl.col("taker_buy_volume"),
        sell_v=(pl.col("volume") - pl.col("taker_buy_volume")).clip(lower_bound=0.0),
    )
    return df, q


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True)
    os.replace(tmp, path)


def staging_path(source: str, asset: str, period: str) -> Path:
    return STAGING_DIR / source / asset / f"{period}.parquet"


def ingest_one(item: dict[str, Any], symbol: str, manifest: Manifest, get: Callable[[str], bytes] = http_get) -> dict[str, Any]:
    """Download, verify, convert and delete one archive zip; returns its manifest entry."""
    key = item["key"]
    url = f"{ARCHIVE_URL}/{key}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / Path(key).name
    try:
        expected = get(url + ".CHECKSUM").decode().split()[0].lower()
        zpath.write_bytes(get(url))
        got = sha256_file(zpath)
        if got != expected:
            raise RuntimeError(f"{key}: sha256 {got} does not match CHECKSUM {expected}")
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise RuntimeError(f"{key}: expected one CSV, found {names}")
            df, q = parse_klines(zf.read(names[0]))
        out = staging_path("binance", symbol, item["period"])
        write_parquet_atomic(df, out)
        entry = {
            "source": "binance_usdm_archive", "asset": symbol, "kind": item["kind"], "period": item["period"],
            "url": url, "zip_bytes": zpath.stat().st_size, "zip_sha256": got, "checksum_verified": True,
            "staged": out.relative_to(DATA_DIR).as_posix(), "parquet_bytes": out.stat().st_size,
            "first_ts_close": df["ts_close"].min().isoformat() if df.height else None,
            "last_ts_close": df["ts_close"].max().isoformat() if df.height else None,
            "quality": q, "status": "staged", "utc": utc_now(),
        }
    finally:
        if zpath.exists():
            zpath.unlink()
    manifest.files[key] = entry
    manifest.save()
    return entry


def ingest_binance(symbols: Iterable[str], manifest: Manifest, *, today: date | None = None,
                   check_storage: bool = True, get: Callable[[str], bytes] = http_get,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Plan all symbols, check storage, then fetch every archive file not yet in the manifest."""
    today = today or datetime.now(UTC).date()
    symbols = list(symbols)
    plans: dict[str, list[dict[str, Any]]] = {}
    for s in symbols:
        if manifest.assets.get(s, {}).get("status") == "split":
            log(f"{s}: already split; nothing to download")
            plans[s] = []
            continue
        plans[s] = [it for it in binance_plan(s, today, get) if it["key"] not in manifest.files]
    planned = sum(it["size"] for p in plans.values() for it in p)
    log(f"plan: {sum(len(p) for p in plans.values())} archive files, {planned / 1e6:.1f} MB")
    if check_storage and planned:
        rec = storage_check(manifest, planned)
        log(f"storage: repo data {rec['repo_data_gb']} GB, vecopen data {rec['vecopen_data_gb']} GB, "
            f"peak add {rec['peak_add_gb']} GB of {rec['budget_gb']} GB budget: OK")
    n = 0
    for s, items in plans.items():
        for it in items:
            e = ingest_one(it, s, manifest, get)
            n += 1
            if n % 25 == 0:
                log(f"  {n} files; last {s} {it['period']} rows {e['quality']['rows']}")
        if items or s not in manifest.assets:
            a = manifest.assets.setdefault(s, {"source": "binance_usdm_archive"})
            a["status"] = a.get("status") if a.get("status") == "split" else "staged"
            a["staged_files"] = sorted(k for k, v in manifest.files.items() if v["asset"] == s)
            manifest.save()
    return {"files_downloaded": n, "planned_bytes": planned, "symbols": symbols}


# ---------------------------------------------------------------- cTrader (run when credentials exist)

ATX = REPO_ROOT / "target" / "debug" / ("atx.exe" if os.name == "nt" else "atx")
CTRADER_RAW = STAGING_DIR / "ctrader_raw"
CTRADER_MAP = DATA_DIR / "ctrader_map.json"
CANDIDATES = PROJECT_ROOT / "config" / "ctrader_candidates.yaml"


def ctrader_keys_present(atx: Path = ATX) -> bool:
    """``atx ctrader check`` prints present/missing per key (never the values); True only if none is missing."""
    r = subprocess.run([str(atx), "ctrader", "check"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("CTRADER_")]
    return bool(lines) and r.returncode == 0 and not any(" missing" in ln and "optional" not in ln for ln in lines)


def ctrader_download(manifest: Manifest, assets: Iterable[str], *, atx: Path = ATX, from_date: str = "2000-01-01",
                     log: Callable[[str], None] = print) -> dict[str, Any]:
    """Resolve names, then pull m5 trend bars for each resolved asset as far back as the server allows."""
    run = lambda *a: subprocess.run([str(atx), "ctrader", *a], cwd=REPO_ROOT, check=True, timeout=6 * 3600)
    run("symbols", "--resolve", "--config", str(CANDIDATES), "--map-out", str(CTRADER_MAP))
    cmap = json.loads(CTRADER_MAP.read_text(encoding="utf-8"))
    manifest.d["ctrader"]["map"] = cmap
    sym_list = REPO_ROOT / "data" / "research" / "xvenue" / "ctrader_symbols.json"
    desc = {}
    if sym_list.exists():
        desc = {r.get("name"): r.get("description") for r in json.loads(sym_list.read_text(encoding="utf-8"))["symbols"]}
    from vecopen.bars import load_sessions
    sess = load_sessions()
    for a in assets:
        name = cmap["map"].get(a, {}).get("ctrader_name")
        if name and sess.roll(a) == "panama_if_futures_based" and "roll_decision" not in manifest.assets.get(a, {}):
            manifest.assets.setdefault(a, {})["roll_decision"] = roll_decision(name, desc.get(name))
    manifest.save()
    done = {}
    for a in assets:
        if manifest.assets.get(a, {}).get("status") == "split":
            log(f"{a}: already split; not downloaded again")
            continue
        row = cmap["map"].get(a, {})
        name = row.get("ctrader_name")
        if not name:
            log(f"{a}: unresolved in {CTRADER_MAP.name} (candidates {row.get('ambiguous')}); skipped")
            continue
        run("trendbars", "--symbol", name, "--period", "M5", "--from", from_date, "--out", str(CTRADER_RAW))
        summ = json.loads((CTRADER_RAW / _safe(name) / "M5" / "_summary.json").read_text(encoding="utf-8"))
        manifest.assets.setdefault(a, {}).update({"source": "ctrader_trendbars_blackbull", "ctrader_name": name,
                                                  "ctrader_id": row.get("ctrader_id"), "trendbars": summ,
                                                  "status": "downloaded"})
        manifest.save()
        done[a] = summ
    return done


FUTURES_HINT = re.compile(r"(?<![a-z])(futures?|fut|expir[a-z]*|contract|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
                          r"[a-z]*[ ./_-]?[0-9]{2,4}|[fghjkmnquvxz][0-9]{1,2})(?![a-z])", re.IGNORECASE)


def roll_decision(name: str, description: str | None) -> dict[str, Any]:
    """`panama_if_futures_based` rule of sessions.yaml, from symbol METADATA only (name and description): futures
    based if either names a futures contract, a contract month or an expiry; otherwise a cash / spot CFD."""
    text = f"{name} {description or ''}"
    m = FUTURES_HINT.search(text)
    return {"roll": "panama" if m else "none", "evidence": text.strip(), "matched": m.group(0) if m else None,
            "rule": "sessions.yaml panama_if_futures_based (metadata only)"}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)


def stage_ctrader(asset: str, manifest: Manifest) -> int:
    """Convert ``ctrader_raw/<name>/M5/<YYYY-MM>.parquet`` (final months only) to staging files."""
    name = manifest.assets[asset]["ctrader_name"]
    src = CTRADER_RAW / _safe(name) / "M5"
    n = 0
    for f in sorted(src.glob("????-??.parquet")):
        raw = pl.read_parquet(f)
        df = raw.select(ts_close=(pl.col("ts_open") + pl.duration(minutes=5)).cast(TS), o="o", h="h", l="l", c="c",
                        v=pl.col("tick_volume").cast(pl.Float64))
        out = staging_path("ctrader", asset, f.stem)
        write_parquet_atomic(df, out)
        manifest.files[f"ctrader/{name}/M5/{f.name}"] = {
            "source": "ctrader_trendbars_blackbull", "asset": asset, "kind": "monthly", "period": f.stem,
            "raw": f.relative_to(DATA_DIR).as_posix(), "raw_sha256": sha256_file(f),
            "staged": out.relative_to(DATA_DIR).as_posix(), "rows": df.height, "status": "staged", "utc": utc_now()}
        n += 1
    manifest.assets[asset]["status"] = "staged"
    manifest.save()
    return n


# ---------------------------------------------------------------- archive-gap repair (before the split)

def fill_from_daily(symbols: Iterable[str], manifest: Manifest, *, max_days: int = 400,
                    get: Callable[[str], bytes] = http_get, log: Callable[[str], None] = print) -> dict[str, Any]:
    """For every missing 5m run of a staged symbol, fetch the archive's DAILY zip of each UTC day the run touches
    (when it exists and is not in the manifest yet). Monthly zips of the archive are sometimes truncated at month
    edges while the daily zips are complete. Runs before the split are repaired for the whole history; only counts
    are logged, never dates, so no holdout-period detail is printed. Monthly rows keep priority over daily rows
    (duplicate rule in vecopen.data)."""
    from vecopen.gaps import missing_runs

    out: dict[str, Any] = {}
    for s in symbols:
        if manifest.assets.get(s, {}).get("status") != "staged":
            continue
        d = STAGING_DIR / "binance" / s
        ts = pl.concat([pl.read_parquet(f, columns=["ts_close"]) for f in sorted(d.glob("*.parquet"))])["ts_close"]
        ts = ts.unique().sort()
        runs = missing_runs(ts)
        days: set[str] = set()
        for r in runs.iter_rows(named=True):
            t = r["gap_first"] - timedelta(minutes=5)  # bar open
            while t < r["gap_last"]:
                days.add(t.date().isoformat())
                t += timedelta(days=1)
            days.add((r["gap_last"] - timedelta(minutes=5)).date().isoformat())
        days = set(sorted(days)[:max_days])
        fetched = missing_in_archive = 0
        for day in sorted(days):
            key = f"data/futures/um/daily/klines/{s}/5m/{s}-5m-{day}.zip"
            if key in manifest.files or (d / f"{day}.parquet").exists():
                continue
            try:
                ingest_one({"key": key, "kind": "daily", "period": day}, s, manifest, get)
                manifest.files[key]["purpose"] = "archive-gap repair"
                fetched += 1
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                missing_in_archive += 1
                manifest.files[key] = {"source": "binance_usdm_archive", "asset": s, "kind": "daily", "period": day,
                                       "status": "absent_in_archive", "purpose": "archive-gap repair", "utc": utc_now()}
        manifest.save()
        out[s] = {"runs": runs.height, "days_checked": len(days), "daily_fetched": fetched,
                  "daily_absent": missing_in_archive}
        log(f"{s}: {runs.height} missing runs, {len(days)} days checked, {fetched} daily zips fetched, "
            f"{missing_in_archive} absent in the archive")
    return out
