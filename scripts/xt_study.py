"""XT study steps (agent XT, 2026-11-28). Library: ai_trader.research.xt. Data only: no orders anywhere.

    .venv/Scripts/python.exe scripts/xt_study.py map      # cTrader symbol map + request latency + clock offset
    .venv/Scripts/python.exe scripts/xt_study.py probe    # get_spot_prices rate ladder (latency, cTrader CPU)
    .venv/Scripts/python.exe scripts/xt_study.py hist     # Bybit archive: trades + ob200 top of book -> bybit_hist/
    .venv/Scripts/python.exe scripts/xt_study.py bars     # cTrader m1 trendbars for the same days (BARS, NOT TICKS)
    .venv/Scripts/python.exe scripts/xt_study.py pilot --since ... --until ...   # data-quality section of the pilot

cTrader calls go only through ai_trader.ctrader.mcp_ro (READ_ONLY_TOOLS). Bybit: public archives / REST only.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import io
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.ctrader.mcp_ro import CTraderRO  # noqa: E402
from ai_trader.research import xt as X  # noqa: E402

UA = "Mozilla/5.0 (AI_Trader research; public archives)"
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
DETAIL_KEYS = ("name", "description", "category", "assetClass", "digits", "pipSize", "lotSize", "minVolume", "maxVolume", "volumeStep")


def pct(a: list[float] | np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(a, float), q)) if len(a) else float("nan")


def lat_stats(a: list[float]) -> dict[str, float]:
    return {"n": len(a), "p50": round(pct(a, 50), 1), "p90": round(pct(a, 90), 1), "p99": round(pct(a, 99), 1),
            "max": round(max(a), 1) if a else None}


# ---------------------------------------------------------------------------------------------------------------- map


def utc_schedule(sess: dict) -> list[str]:
    """Human-readable weekly sessions in UTC at today's offset of the symbol's time zone."""
    if sess.get("marketIsAlwaysOpen"):
        return ["24/7"]
    tz = X.tz_of(sess.get("timeZone", "UTC"))
    off = datetime.now(UTC).astimezone(tz).utcoffset().total_seconds()
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    out = []
    for s in sess.get("sessions", []):
        a, b = int(s["startSecond"]) - off, int(s["endSecond"]) + 1 - off
        f = lambda x: f"{days[int(x // 86400) % 7]} {int(x % 86400 // 3600):02d}:{int(x % 3600 // 60):02d}"  # noqa: E731
        out.append(f"{f(a)} - {f(b)}")
    return out


def cmd_map(_: argparse.Namespace) -> None:
    c = CTraderRO()
    syms = []
    for p in X.PAIRS:
        for name in p["ctrader"]:
            d = c.call("get_symbol_details", symbolName=name)
            s = c.call("get_symbol_sessions", symbolName=name)
            try:
                c.call("get_spot_prices", symbolName=name)
                live = True
            except RuntimeError as e:
                live = "No live quote" not in str(e) and None
            syms.append({**{k: d.get(k) for k in DETAIL_KEYS}, "pair": p["key"], "class": p["class"], "bybit": p["bybit"],
                         "role": p["role"], "note": p["note"], "live_quote_2026_11_28": bool(live),
                         "commission": "not exposed by the MCP symbol details (see report: assumed per class)",
                         "sessions": {k: s.get(k) for k in ("timeZone", "marketIsAlwaysOpen", "sessions")},
                         "sessions_utc_now": utc_schedule(s)})
    # latency of the two calls the recorder uses, and the clock offset (server - local midpoint)
    live = [s["name"] for s in syms if s["live_quote_2026_11_28"]]
    spot, srv, offs = [], [], []
    for i in range(60):
        t = time.perf_counter()
        c.call("get_spot_prices", symbolName=live[i % len(live)])
        spot.append((time.perf_counter() - t) * 1000)
        time.sleep(0.1)
    for _ in range(30):
        s0 = time.time() * 1000
        t = time.perf_counter()
        r = c.call("get_server_time")
        rtt = (time.perf_counter() - t) * 1000
        srv.append(rtt)
        offs.append((int(r["unixMs"]) - (s0 + rtt / 2), rtt))
        time.sleep(0.1)
    best = sorted(offs, key=lambda x: x[1])[:5]
    out = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "source": "cTrader Desktop MCP (read-only: get_symbols, "
           "get_symbol_details, get_symbol_sessions, get_spot_prices, get_server_time)", "broker": "BlackBull Markets (cTrader)",
           "symbols": syms,
           "latency_ms": {"get_spot_prices": lat_stats(spot), "get_server_time": lat_stats(srv)},
           "clock_offset_ms": {"definition": "cTrader server unixMs - local midpoint of the request", "median_all": round(float(np.median([o for o, _ in offs])), 1),
                               "min_rtt_5_mean": round(float(np.mean([o for o, _ in best])), 1),
                               "range": [round(min(o for o, _ in offs), 1), round(max(o for o, _ in offs), 1)]},
           "unsubscribed": [s["name"] for s in syms if not s["live_quote_2026_11_28"]],
           "note": "get_spot_prices only answers for symbols the app is subscribed to (open in a chart / watchlist / market "
                   "watch); the others say 'No live quote ... unsubscribed'. Trendbars come from the server for every symbol."}
    X.write_json(X.MAP_PATH, out)
    print(json.dumps({k: v for k, v in out.items() if k != "symbols"}, indent=1))
    for s in syms:
        print(f"{s['name']:10s} {s['pair']:7s} digits {s['digits']} lot {s['lotSize']} live {s['live_quote_2026_11_28']} "
              f"{s['sessions']['timeZone']} {s['sessions_utc_now'][:1]}")


# ---------------------------------------------------------------------------------------------------------------- probe


def _pid(image: str = "cTrader.exe") -> int | None:
    r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        parts = [x.strip('"') for x in ln.split('","')]
        if len(parts) > 1 and parts[0].lower() == image.lower():
            return int(parts[1])
    return None


def _cpu_s(pid: int) -> float:
    """Kernel + user CPU seconds of a process (read-only query, PROCESS_QUERY_LIMITED_INFORMATION)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return float("nan")
    try:
        c, e, k, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
        k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
        f = lambda t: (t.dwHighDateTime << 32 | t.dwLowDateTime) / 1e7  # noqa: E731
        return f(k) + f(u)
    finally:
        k32.CloseHandle(h)


def cmd_probe(a: argparse.Namespace) -> None:
    c = CTraderRO(timeout=5)
    m = json.loads(X.MAP_PATH.read_text(encoding="utf-8"))
    live = [s["name"] for s in m["symbols"] if s["live_quote_2026_11_28"]]
    pid = _pid()
    res = []
    cpu0, t0 = _cpu_s(pid), time.time()
    time.sleep(a.secs)
    idle = (_cpu_s(pid) - cpu0) / (time.time() - t0) * 100
    res.append({"rate_target": 0, "cpu_pct_one_core": round(idle, 1)})
    print(f"idle cTrader CPU {idle:.1f}% of one core")
    for rate in a.rates:
        lat, err = [], 0
        cpu0, t0 = _cpu_s(pid), time.time()
        nxt = time.perf_counter()
        i = 0
        while time.time() - t0 < a.secs:
            t = time.perf_counter()
            try:
                c.call("get_spot_prices", symbolName=live[i % len(live)])
            except RuntimeError:
                err += 1
            lat.append((time.perf_counter() - t) * 1000)
            i += 1
            if rate > 0:
                nxt += 1.0 / rate
                d = nxt - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                else:
                    nxt = time.perf_counter()
        wall = time.time() - t0
        cpu = (_cpu_s(pid) - cpu0) / wall * 100
        r = {"rate_target": rate if rate > 0 else "max", "achieved_per_s": round(len(lat) / wall, 1), **lat_stats(lat), "errors": err,
             "cpu_pct_one_core": round(cpu, 1), "cpu_added_pct_one_core": round(cpu - idle, 1)}
        res.append(r)
        print(r)
        time.sleep(3)
    out = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "symbols": live, "secs_per_step": a.secs,
           "cpu_note": "cTrader.exe process CPU (kernel+user) per wall second, % of one core; 20 logical cores on this machine",
           "steps": res}
    X.write_json(X.XT_DIR / "ctrader_rate_probe.json", out)


# ---------------------------------------------------------------------------------------------------------------- hist


def download(url: str, dest: Path, tries: int = 4) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r, tmp.open("wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            tmp.replace(dest)
            return dest.stat().st_size
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0
        except Exception:  # noqa: BLE001 - retried
            time.sleep(5 * (k + 1))
    tmp.unlink(missing_ok=True)
    return -1


def ob200_top(path: Path, day: str) -> tuple[Any, dict]:
    """Stream an ob200 archive (snapshot + deltas, 200 levels) and keep one row per change of the top of book:
    ts (publication ms), cts (matching-engine ms), bid, bid_sz, ask, ask_sz. Only rows inside the UTC day are kept."""
    import polars as pl

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    ready = False
    rows_ts, rows_cts, rb, rbs, ra, ras = [], [], [], [], [], []
    last = (None, None, None, None)
    best_b, best_a = None, None
    qa = {"messages": 0, "snapshots": 0, "u_gaps": 0, "crossed": 0}
    last_u = None
    d0 = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    loads = json.loads
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            if not line.strip():
                continue
            msg = loads(line)
            qa["messages"] += 1
            d = msg["data"]
            if msg["type"] == "snapshot":
                bids.clear()
                asks.clear()
                ready = True
                qa["snapshots"] += 1
                best_b = best_a = None
            elif not ready:
                continue
            u = d.get("u")
            if msg["type"] == "delta" and last_u is not None and u is not None and u != last_u + 1:
                qa["u_gaps"] += 1
            last_u = u
            rb_dirty = ra_dirty = False
            for p, s in d.get("b", ()):
                px, sz = float(p), float(s)
                if sz == 0.0:
                    bids.pop(px, None)
                    if px == best_b:
                        rb_dirty = True
                else:
                    bids[px] = sz
                    if best_b is None or px > best_b:
                        best_b = px
            for p, s in d.get("a", ()):
                px, sz = float(p), float(s)
                if sz == 0.0:
                    asks.pop(px, None)
                    if px == best_a:
                        ra_dirty = True
                else:
                    asks[px] = sz
                    if best_a is None or px < best_a:
                        best_a = px
            if rb_dirty or best_b is None or best_b not in bids:
                best_b = max(bids) if bids else None
            if ra_dirty or best_a is None or best_a not in asks:
                best_a = min(asks) if asks else None
            if best_b is None or best_a is None:
                continue
            cur = (best_b, bids[best_b], best_a, asks[best_a])
            if cur != last:
                ts = int(msg["ts"])
                if d0 <= ts < d0 + 86_400_000:
                    rows_ts.append(ts)
                    rows_cts.append(int(msg.get("cts", ts)))
                    rb.append(cur[0]); rbs.append(cur[1]); ra.append(cur[2]); ras.append(cur[3])  # noqa: E702
                    if cur[2] <= cur[0]:
                        qa["crossed"] += 1
                last = cur
    df = pl.DataFrame({"ts": rows_ts, "cts": rows_cts, "bid": rb, "bid_sz": rbs, "ask": ra, "ask_sz": ras},
                      schema={"ts": pl.Int64, "cts": pl.Int64, "bid": pl.Float64, "bid_sz": pl.Float64, "ask": pl.Float64, "ask_sz": pl.Float64})
    qa["rows"] = df.height
    return df, qa


def trades_compact(path: Path) -> Any:
    import polars as pl

    t = pl.read_csv(path, columns=["timestamp", "side", "size", "price"])
    return t.select((pl.col("timestamp") * 1000).round().cast(pl.Int64).alias("T"), pl.col("side").str.slice(0, 1),
                    pl.col("price").cast(pl.Float64), pl.col("size").cast(pl.Float64)).sort("T", maintain_order=True)


def hist_days(a: argparse.Namespace) -> list[str]:
    d0, d1 = datetime.fromisoformat(a.start).date(), datetime.fromisoformat(a.end).date()
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def cmd_hist(a: argparse.Namespace) -> None:
    out = X.XT_DIR / "bybit_hist"
    qa_log = out / "_qa.jsonl"
    syms = X.refuse_spy(a.symbols or X.bybit_symbols())
    for d in hist_days(a):
        for s in syms:
            for kind in ("trades", "book"):
                dest = out / kind / s / f"{d}.parquet"
                if dest.exists():
                    continue
                t0 = time.time()
                url = (TRADES_URL if kind == "trades" else OB_URL).format(s=s, d=d)
                raw = X.RAW_DIR / (f"{s}{d}.csv.gz" if kind == "trades" else f"{d}_{s}_ob200.data.zip")
                n = download(url, raw)
                rec: dict[str, Any] = {"sym": s, "date": d, "kind": kind, "bytes": n}
                if n > 0:
                    try:
                        if kind == "trades":
                            df = trades_compact(raw)
                            qa = {"rows": df.height}
                        else:
                            df, qa = ob200_top(raw, d)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        df.write_parquet(dest, compression="zstd")
                        rec.update(qa, out_mb=round(dest.stat().st_size / 1e6, 2))
                    finally:
                        raw.unlink(missing_ok=True)
                rec["secs"] = round(time.time() - t0, 1)
                with qa_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------------------------------------------- bars


def cmd_bars(a: argparse.Namespace) -> None:
    import polars as pl

    c = CTraderRO()
    out = X.XT_DIR / "ctrader_m1_BARS"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text("cTrader m1 trendbars via the read-only MCP (get_trendbars). BARS, NOT TICKS: one OHLC per "
                                    "minute of the BID price (cTrader trendbars are bid-based), volume = tick count. Minute "
                                    "stamp = bar open (UTC).\n", encoding="utf-8")
    syms = a.symbols or X.ctrader_symbols()
    for s in X.refuse_spy(syms):
        rows = []
        for d in hist_days(a):
            for h0 in range(0, 24, 12):        # 720 bars per request (limit 1000)
                st = f"{d}T{h0:02d}:00:00Z"
                en = f"{d}T{h0 + 11:02d}:59:00Z"
                bars = c.bars(s, "m1", st, en, limit=1000)
                rows += bars
                time.sleep(0.2)
        if not rows:
            print(s, "no bars")
            continue
        df = (pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC").dt.epoch("ms").alias("ts"))
              .select("ts", *[pl.col(k).cast(pl.Float64) for k in ("open", "high", "low", "close")], pl.col("volume").cast(pl.Int64).alias("ticks"))
              .unique("ts").sort("ts"))
        df.write_parquet(out / f"{s}.parquet", compression="zstd")
        print(s, df.height, "bars", datetime.fromtimestamp(df["ts"][0] / 1000, UTC), "..", datetime.fromtimestamp(df["ts"][-1] / 1000, UTC))


# ---------------------------------------------------------------------------------------------------------------- pilot


def ms_day(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp() * 1000)


def usdcusdt_1m(t0: int, t1: int) -> Any:
    """Bybit SPOT USDCUSDT 1 m kline closes (public REST), cached in data/research/xt/usdcusdt_1m.parquet."""
    import polars as pl

    path = X.XT_DIR / "usdcusdt_1m.parquet"
    old = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema={"ts": pl.Int64, "usdc": pl.Float64})
    if old.height and old["ts"].min() <= t0 and old["ts"].max() >= t1 - 120_000:
        return old
    rows, start = [], t0
    while start < t1:
        end = min(t1, start + 999 * 60_000)
        url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol=USDCUSDT&interval=1&start={start}&end={end}&limit=1000"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
            lst = json.loads(r.read())["result"]["list"]
        rows += [(int(k[0]), float(k[4])) for k in lst]
        start = end + 60_000
        time.sleep(0.2)
    df = pl.concat([old, pl.DataFrame(rows, schema={"ts": pl.Int64, "usdc": pl.Float64}, orient="row")]).unique("ts").sort("ts")
    df.write_parquet(path)
    return df


def _q(s: Any) -> dict[str, float]:
    a = np.asarray(s.to_numpy() if hasattr(s, "to_numpy") else s, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "median": round(float(np.median(a)), 2), "p25": round(float(np.percentile(a, 25)), 2),
            "p75": round(float(np.percentile(a, 75)), 2), "p1": round(float(np.percentile(a, 1)), 2), "p99": round(float(np.percentile(a, 99)), 2)}


def minute_bid_gap(bybit: str, ctr: str, book: Any, bars: Any, usdc: Any, sessions: dict | None) -> Any:
    """Minute-end bid gap: 1e4 ln(Bybit best bid at the bar's close time / cTrader m1 bar close bid). BARS, NOT TICKS."""
    import polars as pl

    b = bars.with_columns((pl.col("ts") + 60_000 - 1).alias("te")).sort("te")
    if sessions:
        keep = [X.session_open(int(t) + 30_000, sessions) for t in b["ts"].to_list()]
        b = b.filter(pl.Series(keep))
    j = b.join_asof(book.select(pl.col("ts").alias("te"), pl.col("bid").alias("bb"), pl.col("ask").alias("ba")).sort("te"), on="te",
                    strategy="backward", tolerance=5_000)
    j = j.join_asof(usdc.select(pl.col("ts").alias("te"), "usdc").sort("te"), on="te", strategy="backward", tolerance=180_000)
    j = j.with_columns((1e4 * (pl.col("bb") / pl.col("close")).log()).alias("gap_raw"),
                       (1e4 * pl.col("usdc").log()).alias("usdt_basis_bp"),
                       (1e4 * (pl.col("ba") / pl.col("bb")).log()).alias("bybit_spread_bp"))
    corr = bybit in X.USDT_PRICED
    return j.with_columns((pl.col("gap_raw") - (pl.col("usdt_basis_bp") if corr else pl.lit(0.0))).alias("gap_corr"),
                          pl.lit(bybit).alias("bybit"), pl.lit(ctr).alias("ctrader"))


def _read_stream(d: Path, t0: int, t1: int) -> Any:
    import polars as pl

    fs = sorted(d.glob("date=*/*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed").filter(pl.col("recv_ts").is_between(t0, t1)) if fs else None


def cmd_pilot(a: argparse.Namespace) -> None:
    import polars as pl

    live = X.XT_DIR / "live"
    t0, t1 = (int(datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp() * 1000) for x in (a.since, a.until))
    out: dict[str, Any] = {"window_utc": [a.since, a.until], "note": "data quality and a first look only; no verdicts (H-XT-1..7 need the registered sample)"}
    # A. Bybit recorder
    bk, tr, ck = (_read_stream(live / n, t0, t1) for n in ("bybit_book", "bybit_trades", "bybit_clock"))
    off = float(ck.sort("rtt_ms")["offset_ms"][0]) if ck is not None and ck.height else float("nan")
    span = (min(t1, int(bk["recv_ts"].max())) - max(t0, int(bk["recv_ts"].min()))) / 1000
    rows = []
    for (sym,), g in bk.sort("recv_ts").group_by(["symbol"], maintain_order=True):
        dt = np.diff(g["recv_ts"].to_numpy()) / 1000
        lag = (g.filter(pl.col("kind") != "r")["recv_ts"] - g.filter(pl.col("kind") != "r")["cts"]).to_numpy()   # r = keep-alive re-send
        n_tr = tr.filter(pl.col("symbol") == sym).height
        rows.append({"symbol": sym, "book_rows": g.height, "book_per_s": round(g.height / span, 2), "trades": n_tr,
                     "trades_per_s": round(n_tr / span, 2), "max_quiet_s": round(float(dt.max()), 1) if len(dt) else None,
                     "recv_minus_cts_p50_ms": round(float(np.median(lag)), 1), "recv_minus_cts_p90_ms": round(float(np.percentile(lag, 90)), 1),
                     "one_way_latency_p50_ms_offset_corrected": round(float(np.median(lag)) + off, 1),
                     "median_spread_bp": round(float(np.median(1e4 * np.log(g["ask"].to_numpy() / g["bid"].to_numpy()))), 2)})
    out["bybit"] = {"span_s": span, "clock": {"samples": ck.height if ck is not None else 0, "offset_server_minus_local_ms_at_min_rtt": round(off, 1),
                                               "min_rtt_ms": round(float(ck["rtt_ms"].min()), 1) if ck is not None else None,
                                               "offset_range_ms": [round(float(ck["offset_ms"].min()), 1), round(float(ck["offset_ms"].max()), 1)] if ck is not None else None},
                    "symbols": sorted(rows, key=lambda r: r["symbol"])}
    # B. cTrader recorder
    cq, cc, cp = (_read_stream(live / n, t0, t1) for n in ("ctrader_quotes_SAMPLED", "ctrader_clock", "ctrader_polls"))
    out["ctrader"] = {
        "quote_changes": cq.group_by("symbol").len().sort("symbol").rows() if cq is not None else [],
        "polls_by_state": [(st, int(n), int(e), sorted(set(sy))) for st, n, e, sy in
                           cp.group_by("state").agg(pl.col("n_polls").sum(), pl.col("n_errors").sum(), pl.col("symbol")).rows()] if cp is not None else [],
        "rtt_p50_ms_median": round(float(cp["rtt_p50"].drop_nans().median()), 1) if cp is not None else None,
        "clock": {"samples": cc.height if cc is not None else 0,
                  "offset_server_minus_local_ms_at_min_rtt": round(float(cc.sort("rtt_ms")["offset_ms"][0]), 1) if cc is not None and cc.height else None}}
    # C. history QA
    qa = [json.loads(ln) for ln in (X.XT_DIR / "bybit_hist" / "_qa.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    qd = pl.DataFrame(qa, infer_schema_length=None)
    out["bybit_hist"] = {"by_kind": [{"kind": k, "symbol_days": int(n), "ok": int(ok), "out_mb": round(float(mb or 0), 1)} for k, n, ok, mb in
                                     qd.group_by("kind").agg(pl.len(), (pl.col("bytes") > 0).sum(), pl.col("out_mb").sum()).rows()],
                         "u_gaps_total": int(qd["u_gaps"].fill_null(0).sum()) if "u_gaps" in qd.columns else None,
                         "crossed_rows_total": int(qd["crossed"].fill_null(0).sum()) if "crossed" in qd.columns else None,
                         "download_gb": round(float(qd["bytes"].clip(0).sum()) / 1e9, 2)}
    # D/E. minute bid gaps (BARS, NOT TICKS)
    m = json.loads(X.MAP_PATH.read_text(encoding="utf-8"))
    sess = {s["name"]: s["sessions"] for s in m["symbols"]}
    bars_dir = X.XT_DIR / "ctrader_m1_BARS"
    hist_dir = X.XT_DIR / "bybit_hist" / "book"
    usdc = usdcusdt_1m(ms_day("2026-11-14"), t1)
    gaps, live_gaps = [], []
    for by, ct in X.ARB_LEGS:
        bp, hp = bars_dir / f"{ct}.parquet", sorted((hist_dir / by).glob("*.parquet"))
        if not bp.exists() or not hp:
            continue
        bars = pl.read_parquet(bp)
        book = pl.concat([pl.read_parquet(f, columns=["ts", "bid", "ask"]) for f in hp])
        g = minute_bid_gap(by, ct, book, bars, usdc, sess.get(ct))
        gaps.append(g.select("bybit", "ctrader", "ts", "ticks", "gap_raw", "gap_corr", "usdt_basis_bp", "bybit_spread_bp").drop_nulls("gap_raw"))
        if ct in ("BTCUSD", "ETHUSD"):
            lb = bk.filter(pl.col("symbol") == by).select("ts", "bid", "ask")
            lg = minute_bid_gap(by, ct, lb, bars.filter(pl.col("ts").is_between(t0, t1 - 60_000)), usdc, None).drop_nulls("gap_raw")
            live_gaps.append(lg)
    tab = []
    for g0 in gaps:
        by, ct = g0["bybit"][0], g0["ctrader"][0]
        g = g0.sort("ts")
        day = g.with_columns(pl.from_epoch("ts", time_unit="ms").dt.date().alias("d")).group_by("d").agg(pl.col("gap_corr").median()).sort("d")
        dg = np.diff(g["gap_corr"].to_numpy())
        dg = dg[np.isfinite(dg)]
        tab.append({"pair": f"{by}/{ct}", "minutes": g.height, "gap_raw": _q(g["gap_raw"]), "gap_corr": _q(g["gap_corr"]),
                    "daily_median_corr": [round(float(x), 1) for x in day["gap_corr"].to_list() if x is not None],
                    "sd_1min_change_bp": round(float(np.std(dg)), 2), "ticks_per_min_median": float(g["ticks"].median()),
                    "bybit_spread_bp_median": round(float(g["bybit_spread_bp"].median()), 2)})
    out["minute_bid_gap_hist"] = {"window": "2026-11-16 .. 2026-11-27 (Bybit ob200 top of book vs cTrader m1 bar close; CFD session minutes only)",
                                  "label": "BARS, NOT TICKS; bid vs bid (cTrader trendbars are bid-based), so it carries the half-spread difference",
                                  "pairs": tab}
    out["minute_bid_gap_pilot_window"] = [{"pair": f"{g['bybit'][0]}/{g['ctrader'][0]}", "minutes": g.height, "gap_raw": _q(g["gap_raw"]),
                                           "gap_corr": _q(g["gap_corr"]), "usdt_basis_bp": _q(g["usdt_basis_bp"])} for g in live_gaps if g.height]
    # F. cTrader tick rates (bar volume) and the poller's modelled coverage at 20 req/s
    cov = []
    for s in m["symbols"]:
        bp = bars_dir / f"{s['name']}.parquet"
        if not bp.exists():
            continue
        v = pl.read_parquet(bp)["ticks"].to_numpy() / 60.0
        lam = float(np.median(v))
        row = {"symbol": s["name"], "ticks_per_s_median": round(lam, 2), "ticks_per_s_p90": round(float(np.percentile(v, 90)), 2)}
        for n in (2, 6, 16):
            d = n / 20.0
            row[f"seen_share_n{n}"] = round(float((1 - np.exp(-lam * d)) / (lam * d)), 2) if lam > 0 else None
        cov.append(row)
    out["ctrader_tick_rates"] = {"source": "m1 trendbar volume (tick count) 2026-11-14 .. 11-28", "model": "Poisson ticks at rate lambda, "
                                 "poll interval d = n_live / 20 s: share of quote changes seen = (1 - exp(-lambda d)) / (lambda d)", "rows": cov}
    X.write_json(X.XT_DIR / "pilot_2026-11-28.json", out)
    print(json.dumps(out, indent=1, default=str))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="XT study steps")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("map")
    p = sub.add_parser("probe")
    p.add_argument("--rates", type=float, nargs="+", default=[2, 5, 10, 20, 40, 0])
    p.add_argument("--secs", type=float, default=20)
    for name in ("hist", "bars"):
        h = sub.add_parser(name)
        h.add_argument("--start", default="2026-11-16")
        h.add_argument("--end", default="2026-11-27")
        h.add_argument("--symbols", nargs="*", default=None)
    pi = sub.add_parser("pilot")
    pi.add_argument("--since", required=True)
    pi.add_argument("--until", required=True)
    a = ap.parse_args(argv)
    {"map": cmd_map, "probe": cmd_probe, "hist": cmd_hist, "bars": cmd_bars, "pilot": cmd_pilot}[a.cmd](a)


if __name__ == "__main__":
    main()
