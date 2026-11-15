"""Multi-day microstructure pipeline (overnight program 2026-11-12, agent A; plan research/plans/2026-11-12_overnight_loop.md).

For each UTC day of a batch: download the Bybit public trade archives (50 symbols of config/mm_multi.yaml + BTCUSDT) and
the ob200 L2 archives (12 pilot symbols) into data/raw/batch_<n>/<date>/ (never the lake), reduce every symbol-day to
light aggregates under data/research/micro_days/<date>/, verify them, then delete the raw files.

Per symbol-day (trades):
  trades_1s/<SYM>.parquet   1 s bars (only seconds with prints): ts = second open (epoch ms), OHLC, vwap, buy/sell base
                            volume and USD notional, n_trades (prints), n_buy / n_sell prints, n_orders / n_buy_orders /
                            n_sell_orders (order = maximal run of consecutive prints with the same ms and taker side,
                            counted in the second of its first print)
  orders/<SYM>.parquet      the merged order stream: ms, side (+1 / -1), size, notional, n_prints, first_price, last_price
  laws/<SYM>.json           laws fitted on the RAW prints with the pilot code (research.microstructure_laws_study blocks
                            and sim.laws), in the schema of data/models/microstructure_laws/params_2026-11-10.json
                            ("halves" {fit, test}) plus "tests" (held-out comparisons used by H-MD1/H-MD3) and "activity"
                            (W, rate, Kyle lambda... per half, research.population_simulator_study._half_measures)
Per symbol-day (ob200):
  book_1s/<SYM>.parquet     sim.book state at the END of each 1 s bin of the push ts (bid1, ask1, sizes, mid,
                            rel_spread_bp, depth at 5/10/25 bp per side, imbalance at 5/10 bp, span_bid_bp/span_ask_bp,
                            n_updates). A band wider than the span is truncated by the 200-level feed (BTC ~2.7 bp).
Per day:
  day_stats.json            Hayashi-Yoshida lead-lag of ETH / SOL / BTC vs every symbol (per half), Diebold-Yilmaz
                            connectedness of log 1 m RV for the H-MS6 token set (full day + halves), H-MS5-style
                            spread -> future-vol gains per L2 symbol (H-MD4), book descriptives, activity rows (H-MD2)
  qa.json                   row counts, coverage, missing (404) symbols, download sizes, timings, errors

FIT = 00:00-12:00 UTC and TEST = 12:00-24:00 UTC within each day.

The 2026-11-10 pilot day is built from the lake (trades_raw, read-only) and data/raw/pilot_2026-11-10/book_100ms
(downsampled to 1 s); its laws come from the pilot's params_2026-11-10.json and the pilot per-symbol caches.

    PYTHONIOENCODING=utf-8 uv run python scripts/micro_batch.py --batch 1 --days 2025-09-05 2025-12-18 ... [--keep-raw]
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import threading
import time
import traceback
import urllib.error
import urllib.request
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw"
OUT_ROOT = ROOT / "data" / "research" / "micro_days"
LOG_DIR = ROOT / "logs"
PILOT_DATE = "2026-11-10"
PILOT_RAW = RAW_ROOT / "pilot_2026-11-10"
PILOT_PARAMS = ROOT / "data" / "models" / "microstructure_laws" / "params_2026-11-10.json"
PILOT_CACHE = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\c--Users-Asus-Desktop-AI-Trader\31faef80-e320-4ca6-ab63-04fa013eeb03\scratchpad\c1\sym")
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
UA = "Mozilla/5.0 (AI_Trader research; public archives)"
L2_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "HYPEUSDT", "CARVUSDT", "KASUSDT", "ATOMUSDT", "NVDAUSDT",
              "QQQUSDT", "XAUUSDT")
LEADERS = ("ETHUSDT", "SOLUSDT", "BTCUSDT")
# H-MS6 Diebold-Yilmaz set of the pilot (research/reports/book_spillover_2026-11-11.json, dy.set), fixed for every day
DY_SET = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "XRPUSDT", "HYPEUSDT", "1000PEPEUSDT", "ENAUSDT", "ARBUSDT", "NEARUSDT",
          "AVAXUSDT", "DOGEUSDT", "INJUSDT", "UNIUSDT", "AAVEUSDT")
DY_MIN_COVER = 0.98
DY_BOOT = 200
DAY_MS = 86_400_000
HALF_MS = 43_200_000
B_CSN_TEST_PRINT = 50      # pilot used 200 on TEST prints, 50 elsewhere; p_boot is not used by any H-MD criterion
B_CSN_OTHER = 0            # no bootstrap on FIT prints and orders (saves most of the CSN time)
WORKERS = 4
MIN_BOOK_COVER = 0.90
DOWNLOAD_WORKERS = 4
SEED = 20261112

METHOD = {
    "source": "public.bybit.com/trading (every linear print; timestamp s -> ms = round(ts * 1000) as at-bybit archive.rs; stable sort by ms "
              "keeps the archive order within a millisecond) and quote-saver.bycsi.com ob200 (snapshot + delta JSON lines, sim.book)",
    "orders": "order = maximal run of consecutive prints with the same ms and taker side (pilot C1 definition)",
    "laws": "research.microstructure_laws_study blocks (rates, durations event/print, ACD, size print/order, signs, volume) with the "
            "module's day constants set to this day (FIT 00-12, TEST 12-24 UTC; no CPI exclusion window: test_ex == test); CSN "
            f"bootstrap B = {B_CSN_TEST_PRINT} on TEST prints only (pilot 200 / 50); no AFT",
    "activity": "research.population_simulator_study._half_measures on the half's prints (W = sigma_day x P V_day, orders/s, Kyle "
                "lambda of the trade mid proxy, 1-min sigma, ...)",
    "book_1s": "sim.book.build_series(bin_ms = 1000): state at the end of each 1 s bin of the push ts; pilot: last 100 ms state of each "
               "second from book_100ms (identical by construction), n_updates summed",
    "hy": "book_spillover_study.hy_blocks / lead_stats on per-ms VWAP log prices, lags 0, +-0.1..60 s, per half, no bootstrap",
    "dy": "VAR (BIC lag <= 4) on log 1 m RV (12 x 5 s last-trade returns from the 1 s bars), generalized FEVD H = 10 "
          f"(book_spillover_study.dy_table); fixed pilot token set, a symbol is dropped on a day if missing or its nonzero-RV "
          f"minute coverage < {DY_MIN_COVER}; full day (primary, 30-minute block bootstrap B = {DY_BOOT}) and halves",
    "spread_vol": "per L2 symbol-day as book_spillover_study.ms5_per_symbol: y = log RV over (T, T+h], h = 5, 15 min (1 s mid returns), "
                  "x = log TWA spread over minute m (1 s states), controls log RV last 5 / 60 min, log depth10, intercept (no hour-of-week "
                  "factor); OLS on FIT rows (T >= 01:00, T + h <= 12:00), scored on TEST rows; gain = OOS R^2 full - restricted; "
                  "qualifying (H-MD4) = share of FIT 1 s states with spread > 1.5 ticks >= 0.25 (tick = minimum price increment seen "
                  "in the day's prints)",
}

# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------


def day0_ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def trade_symbols() -> list[str]:
    import yaml

    syms = yaml.safe_load((ROOT / "config" / "mm_multi.yaml").read_text(encoding="utf-8"))["symbols"]
    return list(dict.fromkeys([*syms, "BTCUSDT"]))


def asset_class(sym: str) -> str:
    from ai_trader.research.microstructure_laws_study import asset_class as ac

    return ac(sym)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, np.bool_ | bool):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj)), encoding="utf-8")
    tmp.replace(path)


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def storage() -> dict[str, float]:
    gb = 1024 ** 3
    lake, res = du_bytes(ROOT / "data" / "lake"), du_bytes(ROOT / "data" / "research")
    return {"lake_gb": lake / gb, "research_gb": res / gb, "micro_days_gb": du_bytes(OUT_ROOT) / gb,
            "raw_gb": du_bytes(RAW_ROOT) / gb, "lake_plus_research_gb": (lake + res) / gb,
            "trades_raw_gb": du_bytes(ROOT / "data" / "lake" / "trades_raw") / gb}


def budget_gb() -> float:
    import yaml

    s = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return float(s.get("storage", {}).get("budget_gb", 30))


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------------------------------------------------


def download(url: str, dest: Path, retries: int = 5, timeout: float = 120.0) -> dict[str, Any]:
    """GET url -> dest (atomic via .part). 404 -> status 'missing' (symbol not listed / no archive that day)."""
    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "ok", "bytes": dest.stat().st_size, "cached": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
                expected = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f, 1 << 20)
            size = tmp.stat().st_size
            if expected is not None and int(expected) != size:
                raise OSError(f"short read {size} of {expected}")
            tmp.replace(dest)
            return {"status": "ok", "bytes": size, "attempts": attempt + 1}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"status": "missing", "bytes": 0}
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - retried, reported
            last = repr(e)
        time.sleep(5 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return {"status": "failed", "bytes": 0, "error": last}


# ----------------------------------------------------------------------------------------------------------------------
# trades: load, orders, 1 s bars
# ----------------------------------------------------------------------------------------------------------------------


def load_csv_trades(path: Path, d0: int) -> dict[str, np.ndarray]:
    """One day of public.bybit.com prints: ms (int64), side (+1 buy / -1 sell taker), price, size; archive order kept
    within a millisecond, rows outside [d0, d0 + 1 day) dropped."""
    cols = ["timestamp", "side", "size", "price"]
    sch = {"timestamp": pl.Float64, "side": pl.Utf8, "size": pl.Float64, "price": pl.Float64}
    if Path(path).stat().st_size > 20e6:
        # large archive: stream-decompress to a temp CSV next to it (deleted below), then a lazy scan of 4 columns only,
        # so the decompressed text never sits in memory
        tmp_csv = Path(str(path) + ".tmp.csv")
        with gzip.open(path, "rb") as fi, tmp_csv.open("wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
        try:
            df = pl.scan_csv(tmp_csv, schema_overrides=sch).select(cols).collect()
        finally:
            tmp_csv.unlink(missing_ok=True)
    else:
        with gzip.open(path, "rb") as f:
            df = pl.read_csv(f.read(), columns=cols, schema_overrides=sch)
    ts = df["timestamp"].to_numpy()
    ms = np.round(ts * 1000.0).astype(np.int64)
    order = np.argsort(ms, kind="stable")
    ms = ms[order]
    side = np.where(df["side"].to_numpy()[order] == "Buy", 1, -1).astype(np.int8)
    out = {"ms": ms, "side": side, "price": df["price"].to_numpy()[order].astype(float), "size": df["size"].to_numpy()[order].astype(float)}
    keep = (ms >= d0) & (ms < d0 + DAY_MS)
    return {k: v[keep] for k, v in out.items()}


def load_lake_trades(sym: str, date: str) -> dict[str, np.ndarray]:
    """Read-only DuckDB read of lake trades_raw for one symbol-day (pilot day), archive order kept within a ms."""
    import duckdb

    d0 = day0_ms(date)
    glob = (ROOT / "data" / "lake" / "trades_raw" / "venue=bybit" / f"symbol={sym}" / f"date={date}" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    df = con.sql(f"SELECT epoch_ms(event_ts) AS ms, side, price, size, filename, file_row_number FROM read_parquet('{glob}', "
                 f"file_row_number=true, filename=true) ORDER BY ms, filename, file_row_number").pl()
    ms = df["ms"].to_numpy().astype(np.int64)
    keep = (ms >= d0) & (ms < d0 + DAY_MS)
    side = np.where(df["side"].to_numpy() == "Buy", 1, -1).astype(np.int8)
    return {"ms": ms[keep], "side": side[keep], "price": df["price"].to_numpy()[keep].astype(float), "size": df["size"].to_numpy()[keep].astype(float)}


def merge_orders(tr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Same-ms same-side consecutive prints -> one order (pilot definition)."""
    ms, side, px, sz = tr["ms"], tr["side"], tr["price"], tr["size"]
    n = len(ms)
    if n == 0:
        return {"ms": np.zeros(0, np.int64), "side": np.zeros(0, np.int8), "size": np.zeros(0), "notional": np.zeros(0),
                "n_prints": np.zeros(0, np.int32), "first_price": np.zeros(0), "last_price": np.zeros(0)}
    new = np.ones(n, dtype=bool)
    new[1:] = (ms[1:] != ms[:-1]) | (side[1:] != side[:-1])
    gid = np.cumsum(new) - 1
    st = np.flatnonzero(new)
    en = np.r_[st[1:] - 1, n - 1]
    return {"ms": ms[st], "side": side[st].astype(np.int8), "size": np.bincount(gid, sz), "notional": np.bincount(gid, px * sz),
            "n_prints": np.bincount(gid).astype(np.int32), "first_price": px[st], "last_price": px[en]}


def bars_1s(tr: dict[str, np.ndarray], od: dict[str, np.ndarray], d0: int) -> pl.DataFrame:
    """1 s trade bars for the seconds with at least one print (ts = second open, epoch ms)."""
    ms, side, px, sz = tr["ms"], tr["side"], tr["price"], tr["size"]
    sec = (ms - d0) // 1000
    buy = side > 0
    df = pl.DataFrame({"sec": sec, "price": px, "size": sz, "notional": px * sz, "buy": buy})
    g = (df.group_by("sec", maintain_order=True).agg(
        pl.col("price").first().alias("open"), pl.col("price").max().alias("high"), pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        (pl.col("notional").sum() / pl.col("size").sum()).alias("vwap"),
        pl.col("size").filter(pl.col("buy")).sum().alias("buy_vol"), pl.col("size").filter(~pl.col("buy")).sum().alias("sell_vol"),
        pl.col("notional").filter(pl.col("buy")).sum().alias("buy_notional"),
        pl.col("notional").filter(~pl.col("buy")).sum().alias("sell_notional"),
        pl.len().cast(pl.Int32).alias("n_trades"), pl.col("buy").sum().cast(pl.Int32).alias("n_buy"),
        (~pl.col("buy")).sum().cast(pl.Int32).alias("n_sell")))
    osec = (od["ms"] - d0) // 1000
    o = pl.DataFrame({"sec": osec, "buy": od["side"] > 0}).group_by("sec").agg(
        pl.len().cast(pl.Int32).alias("n_orders"), pl.col("buy").sum().cast(pl.Int32).alias("n_buy_orders"),
        (~pl.col("buy")).sum().cast(pl.Int32).alias("n_sell_orders"))
    g = g.join(o, on="sec", how="left").with_columns(pl.col("n_orders", "n_buy_orders", "n_sell_orders").fill_null(0))
    return (g.sort("sec").with_columns((pl.col("sec") * 1000 + d0).cast(pl.Int64).alias("ts")).drop("sec")
            .select("ts", "open", "high", "low", "close", "vwap", "buy_vol", "sell_vol", "buy_notional", "sell_notional",
                    "n_trades", "n_buy", "n_sell", "n_orders", "n_buy_orders", "n_sell_orders"))


def ms_vwap(tr: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Unique-ms trade times and log VWAP per ms (Hayashi-Yoshida input, as book_spillover_study.ms_prices)."""
    ms = tr["ms"]
    if len(ms) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    new = np.ones(len(ms), bool)
    new[1:] = ms[1:] != ms[:-1]
    gid = np.cumsum(new) - 1
    v = np.bincount(gid, tr["price"] * tr["size"]) / np.bincount(gid, tr["size"])
    return ms[new], np.log(v)


def tick_estimate(price: np.ndarray) -> float | None:
    """Minimum positive increment between distinct print prices of the day (= the tick for any active symbol)."""
    u = np.unique(np.round(price, 10))
    if len(u) < 2:
        return None
    d = np.diff(u)
    d = d[d > 0]
    return float(f"{d.min():.4g}") if len(d) else None


# ----------------------------------------------------------------------------------------------------------------------
# laws (pilot code, day constants patched in the worker)
# ----------------------------------------------------------------------------------------------------------------------


def _patch_laws_module(d0: int):
    from ai_trader.research import microstructure_laws_study as M

    M.DAY0, M.HALF, M.DAY1 = d0, d0 + HALF_MS, d0 + DAY_MS
    M.CPI = (d0 + DAY_MS, d0 + DAY_MS)   # no exclusion window on a random day: test_ex == test
    return M


def fit_laws(sym: str, tr: dict[str, np.ndarray], d0: int) -> dict[str, Any]:
    """Pilot per-symbol result dict (same keys as microstructure_laws_study.analyse_symbol); each block isolated."""
    M = _patch_laws_module(d0)
    rng = np.random.default_rng(SEED + sum(map(ord, sym)))
    lv = M.build_levels(tr)
    res: dict[str, Any] = {"symbol": sym, "class": M.asset_class(sym), "n_prints": int(len(tr["ms"])),
                           "n_orders": int(len(lv["orders"]["ms"])), "n_events": int(len(lv["events"]["ms"])), "errors": {}}
    blocks = {
        "rates": lambda: M.rates_block(lv),
        "durations": lambda: {"event": M.duration_block(lv["events"]["ms"], None), "print": M.duration_block(lv["prints"]["ms"], 0.0005)},
        "acd": lambda: M.acd_block(lv["events"]["ms"])[0],
        "size": lambda: {"print": M.size_block(lv["prints"]["notional"], lv["prints"]["ms"], B_CSN_TEST_PRINT, B_CSN_OTHER, rng),
                         "order": M.size_block(lv["orders"]["notional"], lv["orders"]["ms"], B_CSN_OTHER, B_CSN_OTHER, rng)},
        "signs": lambda: M.sign_block(lv),
        "volume": lambda: M.volume_block(lv),
    }
    for k, fn in blocks.items():
        t = time.time()
        try:
            res[k] = fn()
        except Exception:  # noqa: BLE001 - one block's failure is recorded, the others continue
            res["errors"][k] = traceback.format_exc(limit=3)
        res.setdefault("block_seconds", {})[k] = time.time() - t
    return M._jsonable(res)


def half_params(res: dict[str, Any], d0: int) -> dict[str, Any]:
    """Params-file 'halves' block (microstructure_laws_study._half_params) for whatever blocks succeeded."""
    M = _patch_laws_module(d0)
    out = {}
    for h in ("fit", "test"):
        try:
            out[h] = M._half_params(res, None, h)
        except Exception:  # noqa: BLE001
            out[h] = {"error": traceback.format_exc(limit=2)}
    return out


def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def tests_from_res(r: dict[str, Any]) -> dict[str, Any]:
    """Held-out comparisons of the pilot criteria (H-MS1/2/3/4 style) per symbol-day, for the multi-day study."""
    t: dict[str, Any] = {"n_prints": r.get("n_prints"), "n_orders": r.get("n_orders"), "n_events": r.get("n_events")}
    ev = _g(r, "durations", "event") or {}
    t["dur_ks_exp_test"] = _g(ev, "ks_exp_test", "stat")
    t["dur_cv_test"] = _g(ev, "ks_exp_test", "cv")
    t["dur_heldout_winner"] = _g(ev, "heldout", "test", "winner")
    t["dur_heldout_ranking"] = _g(ev, "heldout", "test", "ranking")
    for h in ("fit", "test"):
        mix = _g(ev, "laws", h, "expmix3")
        if isinstance(mix, dict):
            p = mix["params"]
            w = np.array([p[f"w{i}"] for i in (1, 2, 3)], float)
            rate = np.array([p[f"rate{i}"] for i in (1, 2, 3)], float)
            o = np.argsort(-rate)   # fastest (burst) first
            t[f"mix3_{h}"] = {"w": w[o].tolist(), "mean_s": (1.0 / rate[o]).tolist()}
    t["acd_ks_ratio_test"] = _g(r, "acd", "ks_ratio_test")
    t["acd_best_by_fit"] = _g(r, "acd", "best_by_fit")
    for lvl in ("print", "order"):
        for h in ("fit", "test"):
            z = _g(r, "size", lvl, h) or {}
            t[f"size_{lvl}_{h}"] = {"zeta": _g(z, "powerlaw", "zeta"), "xmin": _g(z, "powerlaw", "xmin"), "n_tail": _g(z, "powerlaw", "n_tail"),
                                    "hill_zeta": z.get("hill_zeta"), "p_boot": z.get("p_boot"), "vuong_R": _g(z, "vuong", "R"),
                                    "vuong_p": _g(z, "vuong", "p")}
    for lvl in ("order", "print"):
        s = _g(r, "signs", lvl) or {}
        t[f"sign_{lvl}"] = {"power_preferred_test": _g(s, "heldout_test", "power_preferred"),
                            "mse_power_test": _g(s, "heldout_test", "mse_power"), "mse_exp_test": _g(s, "heldout_test", "mse_exp"),
                            "n_bins_test": _g(s, "heldout_test", "n_bins"),
                            **{f"gamma_{h}": _g(s, h, "power", "gamma") for h in ("fit", "test")},
                            **{f"acf1_{h}": _g(s, h, "acf", "1") for h in ("fit", "test")},
                            **{f"pacf_1_10_{h}": (_g(s, h, "pacf") or [None] * 10)[:10] for h in ("fit", "test")}}
    c1 = _g(r, "signs", "clock_1s") or {}
    t["hurst_1s_signed"] = {h: _g(c1, h, "hurst_dfa") for h in ("fit", "test")}
    t["hurst_1s_count"] = {h: _g(c1, h, "hurst_dfa_count") for h in ("fit", "test")}
    for b in ("10s", "60s"):
        V = _g(r, "volume", b) or {}
        t[f"volume_{b}"] = {"heavy_beats_gamma": _g(V, "heldout_test", "heavy_beats_gamma"),
                            "acf_power_preferred": _g(V, "acf", "heldout_test", "power_preferred"),
                            "spearman_absret_test": _g(V, "spearman_absret_test", "rho")}
    t["errors"] = list((r.get("errors") or {}).keys())
    return t


def activity(tr: dict[str, np.ndarray], d0: int) -> dict[str, Any]:
    """Population-model numbers per half (research.population_simulator_study._half_measures)."""
    from ai_trader.research import population_simulator_study as PS

    trd = {"ms": tr["ms"], "side": tr["side"].astype(np.int64), "price": tr["price"], "notional": tr["price"] * tr["size"]}
    od = PS.orders_of(trd)
    out = {}
    for h, a in (("fit", d0), ("test", d0 + HALF_MS)):
        try:
            st = PS._half_measures(PS._sel(trd, a, a + HALF_MS), PS._sel(od, a, a + HALF_MS), a)
            st["exc_n"] = len(st.pop("exc_values", []) or [])
            st.pop("impact_moves", None)
            out[h] = st
        except Exception:  # noqa: BLE001
            out[h] = {"error": traceback.format_exc(limit=2)}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------------------------------------------------------


def _worker_init() -> None:
    warnings.filterwarnings("ignore")


def process_trades(date: str, sym: str, src: str, out_dir: str, tmp_dir: str, laws_from_pilot: bool = False) -> dict[str, Any]:
    """One symbol-day: bars, orders, ms-VWAP (temp), laws, activity. src = csv.gz path or 'lake'."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d0 = day0_ms(date)
    out, tmp = Path(out_dir), Path(tmp_dir)
    tr = load_lake_trades(sym, date) if src == "lake" else load_csv_trades(Path(src), d0)
    qa: dict[str, Any] = {"symbol": sym, "kind": "trades", "n_prints": int(len(tr["ms"]))}
    if len(tr["ms"]) == 0:
        qa["error"] = "no prints in the day"
        return qa
    od = merge_orders(tr)
    b = bars_1s(tr, od, d0)
    (out / "trades_1s").mkdir(parents=True, exist_ok=True)
    b.write_parquet(out / "trades_1s" / f"{sym}.parquet", compression="zstd")
    (out / "orders").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(od).write_parquet(out / "orders" / f"{sym}.parquet", compression="zstd")
    t, lp = ms_vwap(tr)
    tmp.mkdir(parents=True, exist_ok=True)
    np.savez(tmp / f"{sym}_msvwap.npz", t=t, lp=lp)
    tick = tick_estimate(tr["price"])
    qa.update({"n_orders": int(len(od["ms"])), "bars_1s": b.height, "first_ms": int(tr["ms"][0]), "last_ms": int(tr["ms"][-1]),
               "notional_usd": float((tr["price"] * tr["size"]).sum()), "tick": tick, "prints_per_order": float(len(tr["ms"]) / max(len(od["ms"]), 1)),
               "secs_bars": time.time() - t0})
    t1 = time.time()
    if laws_from_pilot:
        laws = pilot_laws(sym)
    else:
        res = fit_laws(sym, tr, d0)
        laws = {"symbol": sym, "class": res["class"], "date": date, "n_prints": res["n_prints"], "n_orders": res["n_orders"],
                "n_events": res["n_events"], "halves": half_params(res, d0), "tests": tests_from_res(res), "errors": res.get("errors", {}),
                "block_seconds": res.get("block_seconds"), "source": "micro_days.fit_laws"}
    laws["tick"] = tick
    laws["activity"] = activity(tr, d0)
    write_json(out / "laws" / f"{sym}.json", laws)
    qa["secs_laws"] = time.time() - t1
    qa["law_errors"] = list((laws.get("errors") or {}).keys())
    qa["secs"] = time.time() - t0
    return qa


def pilot_laws(sym: str) -> dict[str, Any]:
    P = json.loads(PILOT_PARAMS.read_text(encoding="utf-8"))
    s = P["symbols"][sym]
    cache = PILOT_CACHE / f"{sym}.json"
    tests = tests_from_res(json.loads(cache.read_text(encoding="utf-8"))) if cache.exists() else {"error": "pilot cache missing"}
    return {"symbol": sym, "class": s["class"], "date": PILOT_DATE, "n_prints": s["n_prints"], "n_orders": s["n_orders"], "n_events": s["n_events"],
            "halves": s["halves"], "tests": tests, "errors": {}, "source": "pilot params_2026-11-10.json + C1 per-symbol cache (B_CSN 200/50)"}


def process_book(date: str, sym: str, src: str, out_dir: str) -> dict[str, Any]:
    """One L2 symbol-day -> book_1s. src = ob200 zip path, or 'pilot' (downsample book_100ms)."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d0 = day0_ms(date)
    if src == "pilot":
        df = downsample_book(pl.read_parquet(PILOT_RAW / "book_100ms" / f"{sym}.parquet"))
        qa_b = json.loads((PILOT_RAW / "book_100ms" / "_qa.json").read_text()).get(sym, {})
    else:
        from ai_trader.sim.book import build_series

        df, qa_b = build_series(src, bin_ms=1000)
    df = df.with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("_t")).filter((pl.col("_t") > d0) & (pl.col("_t") <= d0 + DAY_MS)).drop("_t")
    out = Path(out_dir) / "book_1s"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{sym}.parquet", compression="zstd")
    mid = df["mid"].to_numpy() if df.height else np.zeros(0)
    return {"symbol": sym, "kind": "book", "rows": df.height, "coverage": df.height / 86400, "mid_finite_share": float(np.isfinite(mid).mean()) if len(mid) else 0.0,
            "book_qa": qa_b, "secs": time.time() - t0}


def downsample_book(b100: pl.DataFrame) -> pl.DataFrame:
    """100 ms book series -> 1 s: the state at the end of each second (the 100 ms row whose ts is the second boundary,
    i.e. the last state of (T - 1 s, T]) with n_updates summed over the 10 bins. Same as build_series(bin_ms=1000)."""
    t = b100["ts"].dt.epoch("ms").cast(pl.Int64) if b100["ts"].dtype != pl.Int64 else b100["ts"]
    b = b100.with_columns(t.alias("_t")).with_columns(((pl.col("_t") + 999) // 1000 * 1000).alias("_T"))
    upd = b.group_by("_T").agg(pl.col("n_updates").sum().alias("_nu"))
    last = b.filter(pl.col("_t") == pl.col("_T")).join(upd, on="_T", how="left")
    last = last.with_columns(pl.col("_nu").cast(pl.Int32).alias("n_updates")).drop("_nu", "_t", "_T").sort("ts")
    return last


# ----------------------------------------------------------------------------------------------------------------------
# day stage
# ----------------------------------------------------------------------------------------------------------------------


def _last_price_series(out: Path, sym: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = out / "trades_1s" / f"{sym}.parquet"
    if not p.exists():
        return None
    b = pl.read_parquet(p, columns=["ts", "close"])
    return b["ts"].to_numpy() + 999, np.log(b["close"].to_numpy())


def rv_minutes(t: np.ndarray, lp: np.ndarray, d0: int, step_ms: int = 5000) -> np.ndarray:
    from ai_trader.research.book_spillover_study import grid_last

    g = grid_last(t, lp, d0, d0 + DAY_MS, step_ms)
    r = np.diff(g)
    per = 60_000 // step_ms
    rr = r.reshape(-1, per)
    return np.nansum(rr ** 2, axis=1) * np.where(np.isnan(rr).all(1), np.nan, 1.0)


def dy_day(out: Path, d0: int, rng: np.random.Generator) -> dict[str, Any]:
    from ai_trader.research import book_spillover_study as BS

    rvm, cover = {}, {}
    for s in DY_SET:
        lp = _last_price_series(out, s)
        if lp is None:
            continue
        rvm[s] = rv_minutes(*lp, d0)
        cover[s] = float(np.mean(np.isfinite(rvm[s]) & (rvm[s] > 0)))
    used = [s for s in DY_SET if s in rvm and cover[s] >= DY_MIN_COVER]
    res: dict[str, Any] = {"set_fixed": list(DY_SET), "used": used, "coverage": cover, "dropped": [s for s in DY_SET if s not in used]}
    if len(used) < 4 or not all(s in used for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT")):
        res["error"] = "too few symbols or a leader missing"
        return res

    def mat(sl):
        return 0.5 * np.column_stack([BS._floor_log(np.where(np.isfinite(rvm[s][sl]) & (rvm[s][sl] > 0), rvm[s][sl], 0.0)) for s in used])

    for name, sl, boot in (("day", slice(0, 1440), DY_BOOT), ("fit", slice(0, 720), 0), ("test", slice(720, 1440), 0)):
        try:
            res[name] = BS.dy_table(mat(sl), used, boot=boot, rng=rng)
            res[name].pop("theta", None)
        except Exception:  # noqa: BLE001
            res[name] = {"error": traceback.format_exc(limit=2)}
    return res


def hy_day(tmp: Path, syms: list[str], d0: int) -> dict[str, Any]:
    from ai_trader.research import book_spillover_study as BS

    ser = {}
    for s in syms:
        f = tmp / f"{s}_msvwap.npz"
        if f.exists():
            z = np.load(f)
            ser[s] = (z["t"].astype(np.int64), z["lp"].astype(float))
    out: dict[str, Any] = {"lags_ms": BS.LAGS_MS.tolist(), "pairs": {}}
    for L in LEADERS:
        if L not in ser:
            continue
        for tg in syms:
            if tg == L or tg not in ser:
                continue
            row = {}
            for half, (lo, hi) in (("fit", (d0, d0 + HALF_MS)), ("test", (d0 + HALF_MS, d0 + DAY_MS))):
                try:
                    C, v1, v2 = BS.hy_blocks(*ser[L], *ser[tg], BS.LAGS_MS, (lo, hi))
                    rho = BS.hy_rho(C, v1, v2)
                    st = BS.lead_stats(rho)
                    st["rho"] = np.round(rho, 5).tolist()
                    row[half] = st
                except Exception:  # noqa: BLE001
                    row[half] = {"error": traceback.format_exc(limit=2)}
            out["pairs"][f"{L}->{tg}"] = row
    return out


def minute_book(b: pl.DataFrame, d0: int) -> pl.DataFrame:
    """Per-minute book aggregates from the 1 s series (minute m holds states with ts in (m, m+1] min)."""
    b = b.with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("t")).filter((pl.col("t") > d0) & (pl.col("t") <= d0 + DAY_MS))
    b = b.sort("t").with_columns(((pl.col("t") - d0 - 1) // 60_000).alias("m"), (pl.col("depth_bid_10") + pl.col("depth_ask_10")).alias("depth10"),
                                 pl.col("mid").log().diff().alias("r"))
    mins = b.group_by("m").agg(pl.col("rel_spread_bp").mean().alias("spread"), pl.col("depth10").mean().alias("depth10"),
                               (pl.col("r") ** 2).sum().alias("rv1m"))
    return pl.DataFrame({"m": np.arange(1440, dtype=np.int64)}).join(mins, on="m", how="left").sort("m")


def spread_vol_gain(b: pl.DataFrame, d0: int, horizons=(5, 15)) -> dict[str, Any]:
    """H-MS5 regression per symbol-day (ms5_per_symbol design + OOS R^2 gain of the spread term), see METHOD['spread_vol']."""
    from ai_trader.research import book_spillover_study as BS

    mins = minute_book(b, d0)
    rv = mins["rv1m"].to_numpy().astype(float)
    cs = np.concatenate([[0.0], np.nancumsum(rv)])
    n = 1440
    m = np.arange(n)

    def window(lo, hi):
        ok = (lo >= 0) & (hi < n)
        return np.where(ok, cs[np.clip(hi + 1, 0, n)] - cs[np.clip(lo, 0, n)], np.nan)

    T = d0 + (m + 1) * 60_000
    with np.errstate(divide="ignore", invalid="ignore"):
        ls = np.log(mins["spread"].to_numpy().astype(float))
        ld = np.log(mins["depth10"].to_numpy().astype(float))
    l5, l60 = BS._floor_log(window(m - 4, m)), BS._floor_log(window(m - 59, m))
    out: dict[str, Any] = {}
    for h in horizons:
        y = BS._floor_log(window(m + 1, m + h))
        ok = np.isfinite(y) & np.isfinite(ls) & np.isfinite(ld) & np.isfinite(l5) & np.isfinite(l60)
        fit = ok & (T >= d0 + 3_600_000) & (T + h * 60_000 <= d0 + HALF_MS)
        test = ok & (T >= d0 + HALF_MS) & (T + h * 60_000 <= d0 + DAY_MS)
        if fit.sum() < 60 or test.sum() < 60:
            out[str(h)] = {"error": "too few rows", "n_fit": int(fit.sum()), "n_test": int(test.sum())}
            continue
        Xf = np.column_stack([ls, l5, l60, ld, np.ones(n)])
        Xr = Xf[:, 1:]
        bf = BS._ols(y[fit], Xf[fit])[0]
        br = BS._ols(y[fit], Xr[fit])[0]
        yt = y[test]
        ef, er = yt - Xf[test] @ bf, yt - Xr[test] @ br
        sst = float(((yt - yt.mean()) ** 2).sum())
        g = (T[test] - 1) // 300_000
        bt, set_ = BS.cluster_ols(yt, Xf[test], g)
        gf = (T[fit] - 1) // 300_000
        bff, sef = BS.cluster_ols(y[fit], Xf[fit], gf)
        out[str(h)] = {"n_fit": int(fit.sum()), "n_test": int(test.sum()), "r2_oos_full": 1 - float(ef @ ef) / sst,
                       "r2_oos_restricted": 1 - float(er @ er) / sst, "r2_oos_gain": (float(er @ er) - float(ef @ ef)) / sst,
                       "coef_fit": float(bff[0]), "t_fit": float(bff[0] / sef[0]), "coef_test": float(bt[0]), "t_test": float(bt[0] / set_[0])}
    return out


def book_stats(b: pl.DataFrame, tick: float | None, d0: int) -> dict[str, Any]:
    t = b["ts"].dt.epoch("ms").to_numpy()
    sp = b["rel_spread_bp"].to_numpy()
    out: dict[str, Any] = {"rows": b.height, "tick": tick, "updates_per_s": float(b["n_updates"].mean()) if b.height else None}
    for h, (lo, hi) in (("fit", (d0, d0 + HALF_MS)), ("test", (d0 + HALF_MS, d0 + DAY_MS)), ("day", (d0, d0 + DAY_MS))):
        k = (t > lo) & (t <= hi)
        if not k.any():
            continue
        r: dict[str, Any] = {"n": int(k.sum()), "spread_bp_q": dict(zip(("0.05", "0.25", "0.5", "0.75", "0.95"),
                                                                          np.nanquantile(sp[k], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()))}
        for bnd in (5, 10, 25):
            d = (b[f"depth_bid_{bnd}"] + b[f"depth_ask_{bnd}"]).to_numpy()[k]
            r[f"depth{bnd}_usd_med"] = float(np.nanmedian(d))
        span = np.minimum(b["span_bid_bp"].to_numpy(), b["span_ask_bp"].to_numpy())[k]
        r["span_bp_med"] = float(np.nanmedian(span))
        r["band_truncated_share"] = {str(bnd): float(np.nanmean(span < bnd)) for bnd in (5, 10, 25)}
        r["abs_imb10_mean"] = float(np.nanmean(np.abs(b["imb_10"].to_numpy()[k])))
        r["mid_med"] = float(np.nanmedian(b["mid"].to_numpy()[k]))
        if tick:
            ticks = np.round((b["ask1"].to_numpy()[k] - b["bid1"].to_numpy()[k]) / tick)
            r["tick_bp"] = float(tick / r["mid_med"] * 1e4)
            r["share_gt_1tick"] = float(np.mean(ticks > 1.5))
            r["spread_ticks_med"] = float(np.median(ticks))
            r["tick_bound"] = bool(np.mean(ticks < 1.5) >= 0.9)
        out[h] = r
    return out


def day_stage(date: str, out_dir: str, tmp_dir: str, syms: list[str], l2: list[str]) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d0 = day0_ms(date)
    out, tmp = Path(out_dir), Path(tmp_dir)
    rng = np.random.default_rng(SEED + d0 // DAY_MS)
    res: dict[str, Any] = {"date": date, "segment": segment_of(date), "method": METHOD, "generated": datetime.now(UTC).isoformat(timespec="seconds")}
    laws = {}
    for s in syms:
        p = out / "laws" / f"{s}.json"
        if p.exists():
            laws[s] = json.loads(p.read_text(encoding="utf-8"))
    try:
        res["hy"] = hy_day(tmp, [s for s in syms if s in laws], d0)
    except Exception:  # noqa: BLE001
        res["hy"] = {"error": traceback.format_exc(limit=3)}
    try:
        res["dy"] = dy_day(out, d0, rng)
    except Exception:  # noqa: BLE001
        res["dy"] = {"error": traceback.format_exc(limit=3)}
    res["book"], res["spread_vol"] = {}, {}
    for s in l2:
        p = out / "book_1s" / f"{s}.parquet"
        if not p.exists():
            continue
        b = pl.read_parquet(p)
        tick = (laws.get(s) or {}).get("tick")
        try:
            res["book"][s] = book_stats(b, tick, d0)
            res["spread_vol"][s] = spread_vol_gain(b, d0)
        except Exception:  # noqa: BLE001
            res["spread_vol"][s] = {"error": traceback.format_exc(limit=3)}
    rows = []
    for s, lw in laws.items():
        a = lw.get("activity") or {}
        for h in ("fit", "test"):
            m = a.get(h) or {}
            if "W" not in m:
                continue
            bk = (res["book"].get(s) or {}).get(h) or {}
            rows.append({"symbol": s, "class": lw.get("class"), "half": h, "W": m.get("W"), "sigma_day": m.get("sigma_day"), "pv_day": m.get("pv_day"),
                         "orders_per_s": m.get("orders_per_s"), "prints_per_s": m.get("prints_per_s"), "lambda_mid": m.get("lambda_mid"),
                         "lambda_trade": m.get("lambda_trade"), "sigma_1m": m.get("sigma_1m"), "illiq": m.get("illiq"), "price": m.get("price"),
                         "order_notional_med": m.get("order_notional_med"), "depth10_med": bk.get("depth10_usd_med"),
                         "depth25_med": bk.get("depth25_usd_med"), "spread_med_bp": (bk.get("spread_bp_q") or {}).get("0.5"),
                         "tick_bound": bk.get("tick_bound"), "tick": lw.get("tick")})
    res["activity_rows"] = rows
    res["secs"] = time.time() - t0
    write_json(out / "day_stats.json", res)
    return {"kind": "day", "date": date, "secs": res["secs"], "n_activity_rows": len(rows)}


def segment_of(date: str) -> str:
    from ai_trader.research.evaluate import knowledge_cutoff

    cut = knowledge_cutoff().isoformat()[:10]
    return "llm_in_sample" if date < cut else "clean_holdout"


# ----------------------------------------------------------------------------------------------------------------------
# verification and cleanup
# ----------------------------------------------------------------------------------------------------------------------


def verify_day(out: Path, expect_trades: list[str], expect_books: list[str]) -> dict[str, Any]:
    """Checks before raw deletion: every expected symbol has non-empty, finite aggregates; day_stats and qa exist."""
    problems: list[str] = []
    for s in expect_trades:
        p = out / "trades_1s" / f"{s}.parquet"
        if not p.exists():
            problems.append(f"{s}: trades_1s missing")
            continue
        b = pl.read_parquet(p, columns=["close", "n_trades", "n_orders"])
        if b.height == 0:
            problems.append(f"{s}: trades_1s empty")
        elif not np.isfinite(b["close"].to_numpy()).all() or (b["close"].to_numpy() <= 0).any():
            problems.append(f"{s}: non-finite or non-positive close")
        elif int(b["n_orders"].sum()) <= 0 or int(b["n_orders"].sum()) > int(b["n_trades"].sum()):
            problems.append(f"{s}: order count inconsistent")
        if not (out / "orders" / f"{s}.parquet").exists():
            problems.append(f"{s}: orders missing")
        lp = out / "laws" / f"{s}.json"
        if not lp.exists():
            problems.append(f"{s}: laws missing")
        else:
            lw = json.loads(lp.read_text(encoding="utf-8"))
            if "error" in (lw.get("halves") or {}).get("fit", {}) or not lw.get("halves"):
                problems.append(f"{s}: laws halves failed")
    for s in expect_books:
        p = out / "book_1s" / f"{s}.parquet"
        if not p.exists():
            problems.append(f"{s}: book_1s missing")
            continue
        b = pl.read_parquet(p, columns=["mid"])
        if b.height < MIN_BOOK_COVER * 86400:
            problems.append(f"{s}: book_1s coverage {b.height / 86400:.3f}")
        elif np.isfinite(b["mid"].to_numpy()).mean() < 0.99:
            problems.append(f"{s}: book mid not finite")
    for f in ("day_stats.json", "qa.json"):
        if not (out / f).exists():
            problems.append(f"{f} missing")
    return {"ok": not problems, "problems": problems}


def cleanup_raw(raw_dir: Path, verified: dict[str, Any], keep_raw: bool = False) -> bool:
    """Delete a raw folder only when verification passed and --keep-raw is off. Returns True if deleted."""
    if keep_raw or not verified.get("ok") or not raw_dir.exists():
        return False
    shutil.rmtree(raw_dir)
    return True


# ----------------------------------------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------------------------------------


MIN_FREE_RAM_GB = 6.0      # do not start a new task below this much free physical memory (user on a call, 2026-11-12)
HEAVY_SYMBOLS = {"BTCUSDT", "ETHUSDT"}
HEAVY_TRADES_BYTES = 80e6


def free_ram_gb() -> float:
    """Available physical memory in GB (Windows GlobalMemoryStatusEx; /proc/meminfo elsewhere; inf if unknown)."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullAvailPhys / 1024 ** 3
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:  # noqa: BLE001
        pass
    return float("inf")


def is_heavy(kind: str, sym: str, nbytes: float) -> bool:
    """At most one heavy task in flight: every ob200 parse, BTC / ETH symbol-days and any trades archive > 80 MB."""
    return kind == "book" or sym in HEAVY_SYMBOLS or nbytes > HEAVY_TRADES_BYTES


def _set_thread_env() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "2")


def resumed_task(d: str, kind: str, sym: str, out: Path, tmp: Path) -> dict[str, Any] | None:
    """Resume mode (added after batch 3 was interrupted by a machine sleep): a symbol-day whose outputs are complete on
    disk is not recomputed. Trades: laws/<SYM>.json (written atomically, last) and the temp ms-VWAP file exist and
    trades_1s / orders are readable. Book: book_1s/<SYM>.parquet is readable and non-empty. Returns a qa row or None."""
    try:
        if kind == "trades":
            lp = out / "laws" / f"{sym}.json"
            if not (lp.exists() and (tmp / f"{sym}_msvwap.npz").exists()):
                return None
            lw = json.loads(lp.read_text(encoding="utf-8"))
            n = pl.read_parquet(out / "trades_1s" / f"{sym}.parquet", columns=["n_trades"]).height
            pl.read_parquet(out / "orders" / f"{sym}.parquet", columns=["ms"])
            np.load(tmp / f"{sym}_msvwap.npz")["t"]
            return {"symbol": sym, "kind": "trades", "resumed": True, "n_prints": lw.get("n_prints"), "n_orders": lw.get("n_orders"), "bars_1s": n,
                    "tick": lw.get("tick"), "law_errors": list((lw.get("errors") or {}).keys())}
        p = out / "book_1s" / f"{sym}.parquet"
        if not p.exists():
            return None
        b = pl.read_parquet(p, columns=["mid"])
        if b.height == 0:
            return None
        mid = b["mid"].to_numpy()
        return {"symbol": sym, "kind": "book", "resumed": True, "rows": b.height, "coverage": b.height / 86400, "mid_finite_share": float(np.isfinite(mid).mean())}
    except Exception:  # noqa: BLE001 - unreadable / partial output: recompute
        return None


def run_batch(batch: int, days: list[str], keep_raw: bool = False, workers: int = WORKERS, symbols: list[str] | None = None,
              l2: list[str] | None = None, include_pilot: bool = True, resume: bool = True, no_books: bool = False,
              min_free_gb: float = MIN_FREE_RAM_GB, max_heavy: int = 1) -> dict[str, Any]:
    _set_thread_env()
    log = Log(LOG_DIR / f"micro_batch_{batch}.log")
    syms = symbols or trade_symbols()
    l2 = [] if no_books else list(l2 or L2_SYMBOLS)   # --no-books: trades only; books later with run_books_only
    raw = RAW_ROOT / f"batch_{batch}"
    st0 = storage()
    budget = budget_gb()
    log(f"### batch {batch} start days={days} symbols={len(syms)} l2={len(l2)} workers={workers} keep_raw={keep_raw}")
    log(f"storage before: {json.dumps({k: round(v, 3) for k, v in st0.items()})} budget {budget} GB")
    if st0["lake_plus_research_gb"] > budget - 2.0:
        log("ABORT: lake + research within 2 GB of the budget")
        return {"error": "storage"}
    summary: dict[str, Any] = {"batch": batch, "days": {}, "storage_before": st0}
    if include_pilot and not (OUT_ROOT / PILOT_DATE / "qa.json").exists():
        summary["pilot"] = run_pilot(log, syms, l2, workers)
    dl_pool = ThreadPoolExecutor(DOWNLOAD_WORKERS)
    pool = ProcessPoolExecutor(workers, initializer=_worker_init)
    dl: dict[Future, tuple] = {}
    dl_left: dict[str, int] = {}
    files: dict[str, dict[str, dict]] = {d: {} for d in days}
    for d in days:
        if (OUT_ROOT / d / "qa.json").exists() and json.loads((OUT_ROOT / d / "qa.json").read_text()).get("verified", {}).get("ok"):
            log(f"{d}: already processed and verified, skipped")
            continue
        dl_left[d] = 0
        for s in syms:
            dest = raw / d / "trades" / f"{s}{d}.csv.gz"
            dl[dl_pool.submit(download, TRADES_URL.format(s=s, d=d), dest)] = (d, "trades", s, dest)
            dl_left[d] += 1
        for s in l2:
            dest = raw / d / "ob200" / f"{d}_{s}_ob200.data.zip"
            dl[dl_pool.submit(download, OB_URL.format(s=s, d=d), dest)] = (d, "book", s, dest)
            dl_left[d] += 1
    comp: dict[Future, tuple] = {}
    queue: list[tuple] = []
    heavy_of: dict[Future, bool] = {}
    ram_pause = [False]
    left: dict[str, int] = {}
    qa: dict[str, dict] = {d: {"date": d, "segment": segment_of(d), "downloads": {}, "symbols": {}, "books": {}, "missing_trades": [],
                               "missing_books": [], "failed_downloads": [], "errors": {}, "books_skipped": bool(no_books)} for d in dl_left}
    t_start = time.time()

    def submit_day(d: str) -> None:
        out, tmp = OUT_ROOT / d, raw / d / "tmp"
        out.mkdir(parents=True, exist_ok=True)
        jobs = sorted(files[d].items(), key=lambda kv: -kv[1]["bytes"])
        left[d] = 0
        n_res = 0
        for key, info in jobs:
            kind, s = key.split(":")
            if resume:
                r = resumed_task(d, kind, s, out, tmp)
                if r is not None:
                    (qa[d]["symbols"] if kind == "trades" else qa[d]["books"])[s] = r
                    n_res += 1
                    continue
            if kind == "trades":
                queue.append(((process_trades, d, s, str(info["path"]), str(out), str(tmp)), (d, kind, s), is_heavy(kind, s, info["bytes"])))
            else:
                queue.append(((process_book, d, s, str(info["path"]), str(out)), (d, kind, s), is_heavy(kind, s, info["bytes"])))
            left[d] += 1
        log(f"{d}: queued {left[d]} symbol tasks" + (f" ({n_res} finished symbol-days resumed from disk)" if n_res else ""))
        if left[d] == 0:
            ok_t = [x for x, v in qa[d]["symbols"].items() if "error" not in v]
            ok_b = [x for x, v in qa[d]["books"].items() if "error" not in v]
            queue.insert(0, ((day_stage, d, str(out), str(tmp), ok_t, ok_b), (d, "day", ""), False))
        pump()

    def pump() -> None:
        # at most `workers` tasks in flight; day stages are queued at the front so a finished day is verified (and its
        # raw files deleted) before the next day's symbol tasks start (a FIFO pool would run them last)
        # RAM guard: no new task while free memory < MIN_FREE_RAM_GB; at most one heavy task (ob200 parse, BTC / ETH) in flight
        while queue and len(comp) < workers:
            ram = free_ram_gb()
            if ram < min_free_gb:
                if not ram_pause[0]:
                    log(f"RAM guard: {ram:.1f} GB free < {min_free_gb} GB, pausing new tasks")
                    ram_pause[0] = True
                return
            if ram_pause[0]:
                log(f"RAM guard: {ram:.1f} GB free, resuming")
                ram_pause[0] = False
            n_heavy = sum(heavy_of.get(f, False) for f in comp)
            j = next((i for i, q in enumerate(queue) if not (q[2] and n_heavy >= max_heavy)), None)
            if j is None:
                return
            (fn, *args), tag, hv = queue.pop(j)
            f = pool.submit(fn, *args)
            comp[f] = tag
            heavy_of[f] = hv

    while dl or comp or queue:
        pump()
        if not dl and not comp:
            time.sleep(15)   # RAM guard pause with nothing in flight
            continue
        done, _ = wait(list(dl) + list(comp), timeout=30, return_when=FIRST_COMPLETED)
        for f in done:
            if f in dl:
                d, kind, s, dest = dl.pop(f)
                r = f.result()
                qa[d]["downloads"][f"{kind}:{s}"] = r
                if r["status"] == "ok":
                    files[d][f"{kind}:{s}"] = {"path": dest, "bytes": r["bytes"]}
                elif r["status"] == "missing":
                    qa[d]["missing_trades" if kind == "trades" else "missing_books"].append(s)
                else:
                    qa[d]["failed_downloads"].append(f"{kind}:{s}")
                    log(f"{d}: download FAILED {kind} {s}: {r.get('error')}")
                dl_left[d] -= 1
                if dl_left[d] == 0:
                    mb = sum(v["bytes"] for v in files[d].values()) / 1e6
                    log(f"{d}: downloads done ({len(files[d])} files, {mb:.0f} MB; missing trades {qa[d]['missing_trades']} "
                        f"books {qa[d]['missing_books']}; failed {qa[d]['failed_downloads']}) at {time.time() - t_start:.0f}s")
                    submit_day(d)
                continue
            d, kind, s = comp.pop(f)
            heavy_of.pop(f, None)
            try:
                r = f.result()
            except Exception:  # noqa: BLE001 - recorded per symbol
                r = {"symbol": s, "kind": kind, "error": traceback.format_exc(limit=4)}
            if kind == "day":
                qa[d]["day_stage"] = r
                finish_day(d, qa[d], raw, log, keep_raw)
                summary["days"][d] = {k: qa[d].get(k) for k in ("verified", "raw_deleted", "missing_trades", "missing_books", "failed_downloads")}
                continue
            (qa[d]["symbols"] if kind == "trades" else qa[d]["books"])[s] = r
            if "error" in r:
                qa[d]["errors"][f"{kind}:{s}"] = r["error"]
                log(f"{d}: ERROR {kind} {s}: {str(r['error'])[-300:]}")
            else:
                log(f"{d}: {kind} {s} done in {r.get('secs', 0):.0f}s")
            left[d] -= 1
            if left[d] == 0:
                ok_t = [x for x, v in qa[d]["symbols"].items() if "error" not in v]
                ok_b = [x for x, v in qa[d]["books"].items() if "error" not in v]
                queue.insert(0, ((day_stage, d, str(OUT_ROOT / d), str(raw / d / "tmp"), ok_t, ok_b), (d, "day", ""), False))
                log(f"{d}: symbol tasks done, day stage queued first")
        pump()
    pool.shutdown()
    dl_pool.shutdown()
    if not keep_raw and raw.exists() and not any(raw.iterdir()):
        raw.rmdir()
    st1 = storage()
    summary["storage_after"] = st1
    log(f"storage after: {json.dumps({k: round(v, 3) for k, v in st1.items()})}")
    log(f"### batch {batch} done in {time.time() - t_start:.0f}s")
    write_json(OUT_ROOT / f"_batch_{batch}.json", summary)
    return summary


def finish_day(d: str, qa: dict, raw: Path, log: Log, keep_raw: bool) -> None:
    out = OUT_ROOT / d
    exp_t = [s for s, v in qa["symbols"].items() if "error" not in v]
    exp_b = [s for s, v in qa["books"].items() if "error" not in v]
    write_json(out / "qa.json", qa)
    ver = verify_day(out, exp_t, exp_b)
    if qa["errors"] or qa["failed_downloads"]:
        ver = {"ok": False, "problems": ver["problems"] + [f"task errors: {list(qa['errors'])}", f"failed downloads: {qa['failed_downloads']}"]}
    qa["verified"] = ver
    qa["raw_bytes"] = du_bytes(raw / d)
    qa["raw_deleted"] = cleanup_raw(raw / d, ver, keep_raw)
    qa["aggregate_bytes"] = du_bytes(out)
    write_json(out / "qa.json", qa)
    log(f"{d}: verified={ver['ok']} problems={ver['problems'][:5]} raw {qa['raw_bytes'] / 1e9:.2f} GB deleted={qa['raw_deleted']} "
        f"aggregates {qa['aggregate_bytes'] / 1e6:.0f} MB")


def run_pilot(log: Log, syms: list[str], l2: list[str], workers: int) -> dict[str, Any]:
    """2026-11-10 from the lake (trades_raw, read-only) + book_100ms; then delete the pilot ob200 zips once verified."""
    d = PILOT_DATE
    out, tmp = OUT_ROOT / d, ROOT / "data" / "raw" / "pilot_2026-11-10" / "tmp_micro"
    out.mkdir(parents=True, exist_ok=True)
    log(f"pilot {d}: building from lake trades_raw + book_100ms")
    have = [s for s in syms if (ROOT / "data" / "lake" / "trades_raw" / "venue=bybit" / f"symbol={s}" / f"date={d}").exists()]
    qa: dict[str, Any] = {"date": d, "segment": segment_of(d), "source": "lake trades_raw + pilot book_100ms", "symbols": {}, "books": {},
                          "missing_trades": [s for s in syms if s not in have], "missing_books": [], "failed_downloads": [], "errors": {}}
    with ProcessPoolExecutor(workers, initializer=_worker_init) as pool:
        futs = {pool.submit(process_trades, d, s, "lake", str(out), str(tmp), True): ("trades", s) for s in have}
        futs.update({pool.submit(process_book, d, s, "pilot", str(out)): ("book", s) for s in l2
                     if (PILOT_RAW / "book_100ms" / f"{s}.parquet").exists()})
        for f, (kind, s) in futs.items():
            try:
                r = f.result()
            except Exception:  # noqa: BLE001
                r = {"error": traceback.format_exc(limit=4)}
            (qa["symbols"] if kind == "trades" else qa["books"])[s] = r
            if "error" in r:
                qa["errors"][f"{kind}:{s}"] = r["error"]
                log(f"pilot ERROR {kind} {s}: {str(r['error'])[-300:]}")
        ok_t = [x for x, v in qa["symbols"].items() if "error" not in v]
        ok_b = [x for x, v in qa["books"].items() if "error" not in v]
        qa["day_stage"] = pool.submit(day_stage, d, str(out), str(tmp), ok_t, ok_b).result()
    write_json(out / "qa.json", qa)
    ver = verify_day(out, ok_t, ok_b)
    if qa["errors"]:
        ver = {"ok": False, "problems": ver["problems"] + [f"task errors: {list(qa['errors'])}"]}
    qa["verified"] = ver
    shutil.rmtree(tmp, ignore_errors=True)
    zips = sorted((PILOT_RAW / "ob200").glob("*.zip"))
    qa["ob200_zip_bytes"] = sum(z.stat().st_size for z in zips)
    qa["ob200_deleted"] = False
    if ver["ok"] and len(ok_b) == len(L2_SYMBOLS):
        for z in zips:
            z.unlink()
        qa["ob200_deleted"] = True
    qa["aggregate_bytes"] = du_bytes(out)
    write_json(out / "qa.json", qa)
    log(f"pilot {d}: verified={ver['ok']} problems={ver['problems'][:5]} ob200 zips {qa['ob200_zip_bytes'] / 1e6:.0f} MB deleted={qa['ob200_deleted']}")
    return {k: qa[k] for k in ("verified", "ob200_deleted", "ob200_zip_bytes", "aggregate_bytes")}


def processed_days() -> list[str]:
    """Days with verified aggregates."""
    out = []
    for p in sorted(OUT_ROOT.glob("20*-*-*")):
        q = p / "qa.json"
        if q.exists() and json.loads(q.read_text(encoding="utf-8")).get("verified", {}).get("ok"):
            out.append(p.name)
    return out


def run_books_only(batch: int, days: list[str], keep_raw: bool = False, workers: int = WORKERS, l2: list[str] | None = None) -> dict[str, Any]:
    """Backfill the ob200 books of days processed with --no-books: download, build book_1s, then update ONLY the book-
    dependent parts of day_stats.json (book, spread_vol, the depth / spread / tick fields of activity_rows) and qa.json
    (books, books_skipped = False, verified). Trades aggregates, laws, HY and DY of those days are not touched."""
    _set_thread_env()
    log = Log(LOG_DIR / f"micro_batch_{batch}.log")
    l2 = list(l2 or L2_SYMBOLS)
    raw = RAW_ROOT / f"batch_{batch}_books"
    log(f"### books-only backfill batch {batch} days={days}; storage {json.dumps({k: round(v, 3) for k, v in storage().items()})}")
    summary: dict[str, Any] = {"batch": batch, "books_only": True, "days": {}}
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as dlp, ProcessPoolExecutor(workers, initializer=_worker_init) as pool:
        for d in days:
            out = OUT_ROOT / d
            qp = out / "qa.json"
            if not qp.exists():
                log(f"{d}: no qa.json (trades not processed), skipped")
                continue
            qa = json.loads(qp.read_text(encoding="utf-8"))
            todo = [s for s in l2 if not (out / "book_1s" / f"{s}.parquet").exists()]
            futs = {s: dlp.submit(download, OB_URL.format(s=s, d=d), raw / d / f"{d}_{s}_ob200.data.zip") for s in todo}
            got, missing = {}, []
            for s, f in futs.items():
                r = f.result()
                if r["status"] == "ok":
                    got[s] = raw / d / f"{d}_{s}_ob200.data.zip"
                elif r["status"] == "missing":
                    missing.append(s)
                else:
                    qa.setdefault("errors", {})[f"book:{s}"] = f"download failed: {r.get('error')}"
            res = {s: pool.submit(process_book, d, s, str(p), str(out)) for s, p in got.items()}
            for s, f in res.items():
                try:
                    qa.setdefault("books", {})[s] = f.result()
                except Exception:  # noqa: BLE001
                    qa.setdefault("errors", {})[f"book:{s}"] = traceback.format_exc(limit=3)
            qa["missing_books"] = sorted(set(qa.get("missing_books", [])) | set(missing))
            d0 = day0_ms(d)
            ds = json.loads((out / "day_stats.json").read_text(encoding="utf-8"))
            ds.setdefault("book", {})
            ds.setdefault("spread_vol", {})
            for s in l2:
                p = out / "book_1s" / f"{s}.parquet"
                if not p.exists():
                    continue
                b = pl.read_parquet(p)
                lw = out / "laws" / f"{s}.json"
                tick = json.loads(lw.read_text(encoding="utf-8")).get("tick") if lw.exists() else None
                try:
                    ds["book"][s] = book_stats(b, tick, d0)
                    ds["spread_vol"][s] = spread_vol_gain(b, d0)
                except Exception:  # noqa: BLE001
                    ds["spread_vol"][s] = {"error": traceback.format_exc(limit=3)}
            for r in ds.get("activity_rows", []):
                bk = (ds["book"].get(r["symbol"]) or {}).get(r["half"]) or {}
                r.update({"depth10_med": bk.get("depth10_usd_med"), "depth25_med": bk.get("depth25_usd_med"),
                          "spread_med_bp": (bk.get("spread_bp_q") or {}).get("0.5"), "tick_bound": bk.get("tick_bound")})
            ds["books_backfilled"] = datetime.now(UTC).isoformat(timespec="seconds")
            write_json(out / "day_stats.json", ds)
            exp_t = [s for s, v in qa.get("symbols", {}).items() if "error" not in v]
            exp_b = [s for s, v in qa.get("books", {}).items() if "error" not in v]
            ver = verify_day(out, exp_t, exp_b)
            book_errs = [k for k in qa.get("errors", {}) if k.startswith("book:")]
            if book_errs:
                ver = {"ok": False, "problems": ver["problems"] + [f"book errors: {book_errs}"]}
            qa["books_skipped"] = bool(book_errs)
            qa["verified"] = ver
            qa["raw_books_deleted"] = cleanup_raw(raw / d, ver, keep_raw)
            write_json(qp, qa)
            summary["days"][d] = {"verified": ver, "books": sorted(exp_b), "missing_books": qa["missing_books"]}
            log(f"{d}: books backfilled {sorted(exp_b)} missing {qa['missing_books']} verified={ver['ok']} raw deleted={qa['raw_books_deleted']}")
    if raw.exists() and not any(raw.iterdir()):
        raw.rmdir()
    return summary
