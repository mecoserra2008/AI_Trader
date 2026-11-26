"""PART: market participants with machine learning (agent PART, 2026-11-25/26).

Question: who trades on Bybit USDT perps, how, and does knowing it predict anything tradable?

Design (frozen in research/diary 2026-11-25 "PART design frozen before results", hypotheses H-PART1..H-PART8 in
research/hypotheses.yaml, registered before any fit):

* Universe: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT (linear perps).
* Days: the 12 pre-cutoff dates of the micro_days pipeline (llm_in_sample) + every UTC day 2026-10-06 .. 2026-11-24
  (clean_holdout). FIT = the 12 pre-cutoff days + 2026-10-06 .. 2026-10-29; HELD = 2026-10-30 .. 2026-11-24 (read once).
* Raw: public.bybit.com trades (every print with taker side) and quote-saver.bycsi.com ob200 (snapshot + delta),
  downloaded to data/raw/participants/<date>/, reduced, then deleted.
* Per symbol-day reductions (data/research/participants/days/<date>/):
    book_1s/<SYM>.parquet   book state at the END of each 1 s bin of the push ts (same columns as sim.book) plus the
                            flows inside the bin: near-touch (<= 5 bp from the pre-update mid) added / removed quoted
                            notional and update counts per side, refills at the best level (size added back at the same
                            best price after a decrease in the same bin), L1 Cont-Kukanov-Stoikov OFI (notional), best
                            price changes.
    orders/<SYM>.parquet    taker orders (same-ms same-side runs of prints, micro_days definition) with point-in-time
                            features (book = the 1 s state at or before the order ms) and descriptors (markouts; not used
                            as features).
* Taxonomy (H-PART1): GMM (full covariance) on robust-standardised order-trait features, pooled over symbols, fitted on
  a seeded sample of FIT orders; k chosen by bootstrap ARI + BIC on FIT only (rule `choose_k`). k-means as a robustness
  cross-check. Flow states: Gaussian HMM (diagonal) on 1-minute flow + maker-side features, K by the same rule; only the
  forward filter (point-in-time) is used for features.
* Supervised (H-PART3..5): 1-minute grid per symbol, OLS (cluster-by-day SEs), baseline vs baseline + participant
  features; FIT purged blocked CV by day (6 blocks, 1-day embargo) and one HELD evaluation; day-block bootstrap CIs.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "participants"
RAW = ROOT / "data" / "raw" / "participants"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
CUTOFF = "2026-07-01"
PRE_DAYS = ("2025-08-31", "2025-09-02", "2025-09-05", "2025-09-10", "2025-11-19", "2025-11-21", "2025-12-18", "2026-01-25",
            "2026-05-11", "2026-05-20", "2026-06-14", "2026-06-21")
POST_FIRST, POST_LAST, HELD_FIRST = "2026-10-06", "2026-11-24", "2026-10-30"
SEED = 20261125
DAY_MS = 86_400_000
NEAR_BP = 5.0
BANDS_BP = (5, 10, 25)
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"

# order-trait features clustered by the GMM (fixed before any fit)
ORDER_FEATURES = ("lognot", "rel_l1", "rel_d10", "ticks", "nprints", "dt_any", "dt_same", "n1s", "n10s", "clip20", "sigdig")
# 1-minute flow-state features for the HMM (fixed before any fit)
HMM_FEATURES = ("h_lognot", "h_imb", "h_absr", "h_ofi", "h_upd", "h_cxl", "h_refill", "h_spread")
K_GRID = tuple(range(2, 11))
HMM_GRID = tuple(range(2, 7))
ARI_MIN = 0.80
HORIZONS_MIN = (1, 5, 60)


def post_days() -> list[str]:
    d0, d1 = date.fromisoformat(POST_FIRST), date.fromisoformat(POST_LAST)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def all_days() -> list[str]:
    return list(PRE_DAYS) + post_days()


def split_of(d: str) -> str:
    """FIT / HELD by the frozen design."""
    return "HELD" if d >= HELD_FIRST else "FIT"


def segment_of(d: str) -> str:
    return "llm_in_sample" if d < CUTOFF else "clean_holdout"


def day0_ms(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp() * 1000)


# ----------------------------------------------------------------------------------------------------------------------
# book reducer (ob200 -> 1 s state + in-bin maker-side flows)
# ----------------------------------------------------------------------------------------------------------------------

BOOK_COLS = (["ts", "bid1", "ask1", "bid1_size", "ask1_size", "mid", "rel_spread_bp"]
             + [f"depth_{s}_{b}" for b in BANDS_BP for s in ("bid", "ask")] + ["imb_5", "imb_10", "span_bid_bp", "span_ask_bp",
                                                                               "n_updates"]
             + ["add_bid", "add_ask", "rem_bid", "rem_ask", "n_add", "n_rem", "refill_bid", "refill_ask", "ofi", "n_best_moves"])
FLOW_COLS = ("n_updates", "add_bid", "add_ask", "rem_bid", "rem_ask", "n_add", "n_rem", "refill_bid", "refill_ask", "ofi",
             "n_best_moves")


def cks_ofi(pb: float, qb: float, pa: float, qa: float, pb0: float, qb0: float, pa0: float, qa0: float) -> float:
    """One-update L1 order-flow imbalance (Cont, Kukanov, Stoikov 2014) in base units: bid-side contribution minus
    ask-side contribution. Positive = net buying pressure at the touch."""
    e = 0.0
    if pb >= pb0:
        e += qb
    if pb <= pb0:
        e -= qb0
    if pa <= pa0:
        e -= qa
    if pa >= pa0:
        e += qa0
    return e


def depth_metrics(bp: np.ndarray, bs: np.ndarray, ap: np.ndarray, as_: np.ndarray) -> tuple[float, ...]:
    """(bid1, ask1, bid1_size, ask1_size, mid, rel_spread_bp, depth per side per band (notional), imb_5, imb_10, span_bid,
    span_ask). NaN tuple if a side is empty."""
    n = 6 + 2 * len(BANDS_BP) + 4
    if len(bp) == 0 or len(ap) == 0:
        return (np.nan,) * n
    ib, ia = int(np.argmax(bp)), int(np.argmin(ap))
    b1, a1 = float(bp[ib]), float(ap[ia])
    mid = 0.5 * (b1 + a1)
    out = [b1, a1, float(bs[ib]), float(as_[ia]), mid, (a1 - b1) / mid * 1e4]
    bn, an = bp * bs, ap * as_
    dep = {}
    for b in BANDS_BP:
        lo, hi = mid * (1.0 - b * 1e-4), mid * (1.0 + b * 1e-4)
        db, da = float(bn[bp >= lo].sum()), float(an[ap <= hi].sum())
        dep[b] = (db, da)
        out += [db, da]
    for b in (5, 10):
        db, da = dep[b]
        out.append((db - da) / (db + da) if db + da > 0 else np.nan)
    out += [(mid - float(bp.min())) / mid * 1e4, (float(ap.max()) - mid) / mid * 1e4]
    return tuple(out)


class FlowBook:
    """L2 book (price -> size dicts) with cached best prices and per-bin maker-side flow accumulators."""

    def __init__(self, near_bp: float = NEAR_BP) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.bb = math.nan
        self.ba = math.nan
        self.ready = False
        self.near = near_bp * 1e-4
        self.reset_bin()

    def reset_bin(self) -> None:
        self.acc = dict.fromkeys(FLOW_COLS, 0.0)
        self.dec_b: set[float] = set()
        self.dec_a: set[float] = set()

    def _l1(self) -> tuple[float, float, float, float]:
        bb, ba = self.bb, self.ba
        return bb, self.bids.get(bb, 0.0), ba, self.asks.get(ba, 0.0)

    def apply(self, kind: str, bids: Sequence, asks: Sequence) -> bool:
        if kind == "snapshot":
            self.bids = {float(p): float(s) for p, s in bids if float(s) > 0}
            self.asks = {float(p): float(s) for p, s in asks if float(s) > 0}
            self.bb = max(self.bids) if self.bids else math.nan
            self.ba = min(self.asks) if self.asks else math.nan
            self.ready = True
            return True
        if not self.ready:
            return False
        acc = self.acc
        pb0, qb0, pa0, qa0 = self._l1()
        mid = 0.5 * (pb0 + pa0)
        thr = mid * self.near if mid == mid else 0.0
        # bids
        book, best, dec = self.bids, self.bb, self.dec_b
        for p, s in bids:
            px, sz = float(p), float(s)
            old = book.get(px, 0.0)
            dv = sz - old
            if dv != 0.0 and abs(px - mid) <= thr:
                if dv > 0:
                    acc["add_bid"] += dv * px
                    acc["n_add"] += 1
                    if px == best and px in dec:
                        acc["refill_bid"] += dv * px
                else:
                    acc["rem_bid"] -= dv * px
                    acc["n_rem"] += 1
                    if px == best:
                        dec.add(px)
            if sz == 0.0:
                book.pop(px, None)
                if px == best:
                    best = max(book) if book else math.nan
            else:
                book[px] = sz
                if not (px <= best):  # also true when best is NaN
                    best = px
        self.bb = best
        book, best, dec = self.asks, self.ba, self.dec_a
        for p, s in asks:
            px, sz = float(p), float(s)
            old = book.get(px, 0.0)
            dv = sz - old
            if dv != 0.0 and abs(px - mid) <= thr:
                if dv > 0:
                    acc["add_ask"] += dv * px
                    acc["n_add"] += 1
                    if px == best and px in dec:
                        acc["refill_ask"] += dv * px
                else:
                    acc["rem_ask"] -= dv * px
                    acc["n_rem"] += 1
                    if px == best:
                        dec.add(px)
            if sz == 0.0:
                book.pop(px, None)
                if px == best:
                    best = min(book) if book else math.nan
            else:
                book[px] = sz
                if not (px >= best):
                    best = px
        self.ba = best
        pb, qb, pa, qa = self._l1()
        if pb0 == pb0 and pa0 == pa0 and pb == pb and pa == pa:
            acc["ofi"] += cks_ofi(pb, qb, pa, qa, pb0, qb0, pa0, qa0) * 0.5 * (pb + pa)
            if pb != pb0 or pa != pa0:
                acc["n_best_moves"] += 1
        acc["n_updates"] += 1
        return True

    def row(self, ts: int) -> tuple:
        b, a = self.bids, self.asks
        m = depth_metrics(np.fromiter(b.keys(), float, len(b)), np.fromiter(b.values(), float, len(b)),
                          np.fromiter(a.keys(), float, len(a)), np.fromiter(a.values(), float, len(a)))
        return (ts, *m, *(self.acc[c] for c in FLOW_COLS))


PART_BOOK_EXE = ROOT / "target" / "agent_part" / "bin" / "part-book.exe"


def reduce_book_rust(path: Path, d0: int, exe: Path = PART_BOOK_EXE) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fast path: crates/at-part `part-book` (same semantics as FlowBook, parity-checked) -> the same frame."""
    import subprocess

    out = Path(str(path) + ".f64")
    r = subprocess.run([str(exe), str(path), str(out)], capture_output=True, text=True, check=True)
    qa = json.loads(r.stdout.strip().splitlines()[-1])
    a = np.fromfile(out, dtype="<f8").reshape(-1, len(BOOK_COLS))
    out.unlink()
    df = pl.DataFrame({c: a[:, i] for i, c in enumerate(BOOK_COLS)}).with_columns(pl.col("ts").cast(pl.Int64))
    return grid_frame(df, d0, qa)


def iter_ob(path: Path) -> Iterator[dict]:
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def reduce_book(messages: Iterator[dict], d0: int, bin_ms: int = 1000) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Messages -> one row per bin (state at bin end + flows inside the bin) on the complete grid (d0, d0 + 1 day];
    bins without messages carry the state forward with zero flows. Snapshots reset the book (their levels are not
    counted as adds)."""
    fb = FlowBook()
    rows: list[tuple] = []
    cur = None
    qa = {"messages": 0, "snapshots": 0, "before_snapshot": 0, "u_gaps": 0}
    last_u = None
    for msg in messages:
        qa["messages"] += 1
        b = int(msg["ts"]) // bin_ms
        if cur is not None and b != cur and fb.ready:
            rows.append(fb.row((cur + 1) * bin_ms))
            fb.reset_bin()
        cur = b
        d = msg["data"]
        if msg["type"] == "snapshot":
            qa["snapshots"] += 1
        if not fb.apply(msg["type"], d.get("b", ()), d.get("a", ())):
            qa["before_snapshot"] += 1
            continue
        u = d.get("u")
        if msg["type"] == "delta" and last_u is not None and u is not None and u != last_u + 1:
            qa["u_gaps"] += 1
        last_u = u
    if cur is not None and fb.ready:
        rows.append(fb.row((cur + 1) * bin_ms))
    if not rows:
        return pl.DataFrame(schema=dict.fromkeys(BOOK_COLS, pl.Float64)), qa
    df = pl.DataFrame(rows, schema=list(BOOK_COLS), orient="row").with_columns(pl.col("ts").cast(pl.Int64))
    return grid_frame(df, d0, qa, bin_ms)


def grid_frame(df: pl.DataFrame, d0: int, qa: dict[str, Any], bin_ms: int = 1000) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Emitted bin rows -> the complete grid (d0, d0 + 1 day]: state carried forward, flows 0 in silent bins."""
    grid = pl.DataFrame({"ts": np.arange(d0 + bin_ms, d0 + DAY_MS + bin_ms, bin_ms, dtype=np.int64)})
    full = grid.join(df.filter((pl.col("ts") > d0) & (pl.col("ts") <= d0 + DAY_MS)), on="ts", how="left")
    state = [c for c in BOOK_COLS if c != "ts" and c not in FLOW_COLS]
    full = full.with_columns([pl.col(c).forward_fill() for c in state] + [pl.col(c).fill_null(0.0) for c in FLOW_COLS])
    qa["bins_with_update"] = int(df.filter((pl.col("ts") > d0) & (pl.col("ts") <= d0 + DAY_MS)).height)
    qa["coverage"] = float(full["mid"].is_not_null().mean())
    cast = {c: pl.Float32 for c in BOOK_COLS if c not in ("ts", "bid1", "ask1", "mid")}
    return full.with_columns([pl.col(c).cast(t) for c, t in cast.items()]), qa


# ----------------------------------------------------------------------------------------------------------------------
# taker orders and their point-in-time features
# ----------------------------------------------------------------------------------------------------------------------


def sig_digits(x: np.ndarray, max_d: int = 6) -> np.ndarray:
    """Number of significant digits of positive sizes (1 for 0.5, 2 for 0.25 or 12, ...), capped at max_d."""
    x = np.asarray(x, float)
    out = np.full(len(x), max_d, dtype=np.int8)
    pos = x > 0
    e = np.floor(np.log10(np.where(pos, x, 1.0)))
    m = np.where(pos, x / 10.0 ** e, 1.0)  # mantissa in [1, 10)
    done = ~pos
    for d in range(1, max_d):
        r = np.round(m, d - 1)
        hit = ~done & (np.abs(r - m) < 1e-6 * m)
        out[hit] = d
        done |= hit
    out[~pos] = 1
    return out


def trailing_count(ms: np.ndarray, window_ms: int) -> np.ndarray:
    """Orders strictly before index i with time in [ms_i - window, ms_i] (ms sorted)."""
    lo = np.searchsorted(ms, ms - window_ms, side="left")
    return np.arange(len(ms)) - lo


def gap_prev(ms: np.ndarray, cap_ms: float = 60_000.0) -> np.ndarray:
    """ms since the previous event (first event: cap)."""
    g = np.full(len(ms), cap_ms)
    if len(ms) > 1:
        g[1:] = np.minimum(np.diff(ms).astype(float), cap_ms)
    return g


def clip_count(size: np.ndarray, lags: int = 20) -> np.ndarray:
    """How many of the previous `lags` sizes are exactly equal to this one (repeated child-order size)."""
    n = len(size)
    c = np.zeros(n, dtype=np.int16)
    for j in range(1, min(lags, n - 1) + 1):
        c[j:] += (size[j:] == size[:-j])
    return c


def by_side(values_fn, side: np.ndarray, *arrays: np.ndarray) -> np.ndarray:
    """Apply values_fn to the subsequence of each side and scatter back."""
    out = None
    for s in (1, -1):
        idx = np.flatnonzero(side == s)
        r = values_fn(*(a[idx] for a in arrays))
        if out is None:
            out = np.zeros(len(side), dtype=r.dtype)
        out[idx] = r
    return out if out is not None else np.zeros(0)


def asof_index(book_ts: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """Index of the last book row with ts <= ms (-1 if none)."""
    return np.searchsorted(book_ts, ms, side="right") - 1


def order_features(od: dict[str, np.ndarray], book: pl.DataFrame, tick: float) -> pl.DataFrame:
    """Point-in-time features of each taker order. Book = state at or before the order ms (the 1 s row ts <= ms)."""
    ms, side, size, notional = od["ms"], od["side"], od["size"], od["notional"]
    fp, lp, npr = od["first_price"], od["last_price"], od["n_prints"]
    bts = book["ts"].to_numpy()
    j = np.clip(asof_index(bts, ms), 0, len(bts) - 1)
    valid = asof_index(bts, ms) >= 0
    g = {c: book[c].to_numpy().astype(float)[j] for c in ("bid1", "ask1", "bid1_size", "ask1_size", "mid", "depth_bid_10",
                                                           "depth_ask_10", "rel_spread_bp")}
    buy = side > 0
    opp_l1 = np.where(buy, g["ask1"] * g["ask1_size"], g["bid1"] * g["bid1_size"])
    opp_d10 = np.where(buy, g["depth_ask_10"], g["depth_bid_10"])
    mid = g["mid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_l1 = np.log(notional / opp_l1)
        rel_d10 = np.log(notional / opp_d10)
        ticks = np.round(np.abs(lp - fp) / tick)
        sweep_bp = np.abs(lp - fp) / mid * 1e4
    df = pl.DataFrame({
        "ms": ms, "side": side.astype(np.int8), "size": size, "notional": notional, "vwap": notional / size,
        "first_price": fp, "last_price": lp, "n_prints": npr.astype(np.int32), "book_ok": valid & np.isfinite(mid),
        "mid": mid, "spread_bp": g["rel_spread_bp"], "sweep_bp": sweep_bp,
        "lognot": np.log10(notional), "rel_l1": rel_l1, "rel_d10": rel_d10, "ticks": np.log1p(ticks),
        "nprints": np.log1p(npr.astype(float)), "dt_any": np.log1p(gap_prev(ms)),
        "dt_same": np.log1p(by_side(lambda m: gap_prev(m), side, ms)),
        "n1s": np.log1p(trailing_count(ms, 1000).astype(float)), "n10s": np.log1p(trailing_count(ms, 10_000).astype(float)),
        "clip20": np.log1p(by_side(lambda s: clip_count(s), side, size).astype(float)),
        "sigdig": sig_digits(size).astype(float),
    })
    return df


def markouts(ms: np.ndarray, side: np.ndarray, price: np.ndarray, book_ts: np.ndarray, mid: np.ndarray,
             horizons_ms: Sequence[int] = (1000, 10_000, 60_000)) -> dict[str, np.ndarray]:
    """Taker markout in bp: side * (mid(t + h) - price) / price * 1e4, mid at the first 1 s state ts >= t + h
    (descriptor of informedness; never a feature)."""
    out = {}
    for h in horizons_ms:
        k = np.searchsorted(book_ts, ms + h, side="left")
        ok = k < len(book_ts)
        m = np.where(ok, mid[np.clip(k, 0, len(mid) - 1)], np.nan)
        out[f"mk_{h // 1000}s"] = side * (m - price) / price * 1e4
    return out


# ----------------------------------------------------------------------------------------------------------------------
# day pipeline: download -> reduce -> verify -> delete raw
# ----------------------------------------------------------------------------------------------------------------------

STORAGE_CAP_GB = 28.0   # stop before the 30 GB budget (lake + research + raw)
RAW_CAP_GB = 2.0


def storage_gb() -> dict[str, float]:
    from ai_trader.research.micro_days import du_bytes

    gb = 1024 ** 3
    lake, res, raw = (du_bytes(ROOT / "data" / x) / gb for x in ("lake", "research", "raw"))
    return {"lake_gb": lake, "research_gb": res, "raw_gb": raw, "participants_gb": du_bytes(OUT) / gb,
            "part_raw_gb": du_bytes(RAW) / gb, "total_gb": lake + res + raw}


def _log(msg: str) -> None:
    line = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "reduce_log.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def raw_paths(d: str, sym: str) -> tuple[Path, Path]:
    return RAW / d / f"{sym}.csv.gz", RAW / d / f"{sym}_ob200.zip"


def download(url: str, dest: Path, tries: int = 6) -> dict[str, Any]:
    """curl with a stall guard (abort below 300 kB/s for 30 s, then retry); 404 -> 'missing'. urllib could hang on a
    trickling connection for many minutes (seen 2026-11-25 23:40)."""
    import subprocess
    import time

    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "ok", "bytes": dest.stat().st_size, "cached": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = ""
    for attempt in range(tries):
        r = subprocess.run(["curl", "-sS", "-L", "-A", "Mozilla/5.0 (AI_Trader research; public archives)", "--connect-timeout", "20",
                            "--speed-limit", "300000", "--speed-time", "30", "-o", str(tmp), "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        code = r.stdout.strip()[-3:]
        if code == "404":
            tmp.unlink(missing_ok=True)
            return {"status": "missing", "bytes": 0}
        if r.returncode == 0 and code == "200" and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return {"status": "ok", "bytes": dest.stat().st_size, "attempts": attempt + 1}
        last = f"curl rc={r.returncode} http={code} {r.stderr.strip()[:200]}"
        tmp.unlink(missing_ok=True)
        time.sleep(5 * (attempt + 1))
    return {"status": "failed", "bytes": 0, "error": last}


def download_day(d: str, guard: bool = False, urgent: Any = None) -> dict[str, Any]:
    """Download the day's raw files; with guard=True (prefetch) wait before each file while raw >= RAW_CAP_GB - 0.35,
    until `urgent` (a threading.Event) is set: the day became the current one, so it must not wait any more."""
    import time

    from ai_trader.research.micro_days import du_bytes

    out = {}
    for sym in SYMBOLS:
        tp, bp = raw_paths(d, sym)
        for key, url, dest, done in ((f"trades:{sym}", TRADES_URL.format(s=sym, d=d), tp, OUT / "days" / d / "orders" / f"{sym}.parquet"),
                                     (f"book:{sym}", OB_URL.format(s=sym, d=d), bp, OUT / "days" / d / "book_1s" / f"{sym}.parquet")):
            if done.exists():
                continue
            while guard and not (urgent is not None and urgent.is_set()) and not dest.exists() and du_bytes(RAW) / 1024 ** 3 >= RAW_CAP_GB - 0.35:
                time.sleep(5)
            out[key] = download(url, dest)
    return out


def reduce_symbol(d: str, sym: str) -> dict[str, Any]:
    """Worker: one symbol-day raw -> orders + book_1s parquet; deletes its raw files after writing."""
    import os
    import time

    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    from ai_trader.research.micro_days import load_csv_trades, merge_orders, tick_estimate

    t0 = time.time()
    d0 = day0_ms(d)
    od_dir, bk_dir = OUT / "days" / d / "orders", OUT / "days" / d / "book_1s"
    od_dir.mkdir(parents=True, exist_ok=True)
    bk_dir.mkdir(parents=True, exist_ok=True)
    tp, bp = raw_paths(d, sym)
    qa: dict[str, Any] = {"symbol": sym, "date": d}
    if not (od_dir / f"{sym}.parquet").exists() and tp.exists():
        tr = load_csv_trades(tp, d0)
        od = merge_orders(tr)
        qa.update(n_prints=len(tr["ms"]), n_orders=len(od["ms"]), tick=tick_estimate(tr["price"]),
                  notional_usd=float(od["notional"].sum()))
        pl.DataFrame(od).write_parquet(od_dir / f"{sym}.tmp", compression="zstd")
        (od_dir / f"{sym}.tmp").replace(od_dir / f"{sym}.parquet")
        tp.unlink()
    if not (bk_dir / f"{sym}.parquet").exists() and bp.exists():
        df, bqa = reduce_book_rust(bp, d0) if PART_BOOK_EXE.exists() else reduce_book(iter_ob(bp), d0)
        qa["book"] = bqa
        df.write_parquet(bk_dir / f"{sym}.tmp", compression="zstd")
        (bk_dir / f"{sym}.tmp").replace(bk_dir / f"{sym}.parquet")
        bp.unlink()
    qa["secs"] = time.time() - t0
    return qa


def run_reduce(days: Sequence[str] | None = None, workers: int = 2) -> None:
    import shutil
    import threading
    from concurrent.futures import ProcessPoolExecutor

    days = list(days or all_days())
    todo = [d for d in days if not (OUT / "days" / d / "qa.json").exists()]
    _log(f"reduce start: {len(todo)} days to do, storage {storage_gb()}")
    pre: dict[str, threading.Thread] = {}
    res: dict[str, dict] = {}

    urgent: dict[str, threading.Event] = {}

    def fetch(d, guard=False):
        res[d] = download_day(d, guard, urgent.setdefault(d, threading.Event()))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, d in enumerate(todo):
            st = storage_gb()
            if st["total_gb"] > STORAGE_CAP_GB:
                _log(f"STOP: storage {st['total_gb']:.2f} GB > cap {STORAGE_CAP_GB}")
                return
            if d not in pre:
                pre[d] = threading.Thread(target=fetch, args=(d,))
                pre[d].start()
            urgent.setdefault(d, threading.Event()).set()
            pre[d].join()
            dl = res.pop(d, {})
            bad = {k: v for k, v in dl.items() if v.get("status") != "ok"}
            # prefetch the next day while this one is reduced (raw stays < RAW_CAP_GB: files are deleted as reduced)
            if i + 1 < len(todo):
                nxt = todo[i + 1]
                pre[nxt] = threading.Thread(target=fetch, args=(nxt, True))
                pre[nxt].start()
            futs = [ex.submit(reduce_symbol, d, s) for s in SYMBOLS]
            qas = [f.result() for f in futs]
            ok = all((OUT / "days" / d / k / f"{s}.parquet").exists() for s in SYMBOLS for k in ("orders", "book_1s"))
            qa = {"date": d, "split": split_of(d), "segment": segment_of(d), "downloads": dl, "failed": bad, "symbols": qas,
                  "complete": ok}
            (OUT / "days" / d / "qa.json").write_text(json.dumps(qa, indent=1, default=str), encoding="utf-8")
            if ok:
                shutil.rmtree(RAW / d, ignore_errors=True)
            _log(f"{d} done complete={ok} secs={max(q['secs'] for q in qas):.0f} failed={list(bad)} raw={storage_gb()['part_raw_gb']:.2f} GB")
    _log(f"reduce end: storage {storage_gb()}")


# ----------------------------------------------------------------------------------------------------------------------
# taxonomy: sample, standardise, choose k, fit, assign
# ----------------------------------------------------------------------------------------------------------------------

TAX = OUT / "taxonomy"
N_SAMPLE_PER_SYMDAY = 3000
N_BOOT, N_BOOT_SAMPLE, N_EVAL, N_BIC = 10, 30_000, 20_000, 100_000


def day_qa(d: str) -> dict[str, Any]:
    return json.loads((OUT / "days" / d / "qa.json").read_text(encoding="utf-8"))


def complete_days() -> list[str]:
    out = []
    for d in all_days():
        p = OUT / "days" / d / "qa.json"
        if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("complete"):
            out.append(d)
    return out


def tick_of(d: str, sym: str) -> float:
    for q in day_qa(d)["symbols"]:
        if q["symbol"] == sym and q.get("tick"):
            return float(q["tick"])
    # orders reduced by an interrupted run have no QA row: same estimator on the order first / last prices
    from ai_trader.research.micro_days import tick_estimate

    o = pl.read_parquet(OUT / "days" / d / "orders" / f"{sym}.parquet", columns=["first_price", "last_price"])
    t = tick_estimate(np.r_[o["first_price"].to_numpy(), o["last_price"].to_numpy()])
    if t is None:
        raise KeyError((d, sym))
    return t


def load_book(d: str, sym: str) -> pl.DataFrame:
    return pl.read_parquet(OUT / "days" / d / "book_1s" / f"{sym}.parquet")


def load_orders(d: str, sym: str, book: pl.DataFrame | None = None) -> pl.DataFrame:
    """Orders of a symbol-day with point-in-time features and (descriptor-only) markouts."""
    book = load_book(d, sym) if book is None else book
    o = pl.read_parquet(OUT / "days" / d / "orders" / f"{sym}.parquet")
    od = {c: o[c].to_numpy() for c in o.columns}
    df = order_features(od, book, tick_of(d, sym))
    bts, mid = book["ts"].to_numpy(), book["mid"].to_numpy().astype(float)
    side = df["side"].to_numpy().astype(float)
    m0 = df["mid"].to_numpy()
    mk = {}
    for h in (10_000, 60_000):
        k = np.searchsorted(bts, df["ms"].to_numpy() + h, side="left")
        ok = k < len(bts)
        mh = np.where(ok, mid[np.clip(k, 0, len(mid) - 1)], np.nan)
        mk[f"mm_{h // 1000}s"] = side * np.log(mh / m0) * 1e4                       # mid-to-mid (informedness)
        fp = df["first_price"].to_numpy()
        mk[f"as_{h // 1000}s"] = side * (mh - fp) / fp * 1e4                        # touch maker's loss
    return df.with_columns([pl.Series(k, v) for k, v in mk.items()])


def robust_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and IQR / 1.349 per column (scale floored at 1e-6; a constant column gets scale 1)."""
    med = np.nanmedian(x, axis=0)
    q75, q25 = np.nanpercentile(x, 75, axis=0), np.nanpercentile(x, 25, axis=0)
    sc = (q75 - q25) / 1.349
    sd = np.nanstd(x, axis=0)
    sc = np.where(sc > 1e-6, sc, np.where(sd > 1e-6, sd, 1.0))
    return med, sc


def standardise(x: np.ndarray, med: np.ndarray, sc: np.ndarray, clip: float = 6.0) -> np.ndarray:
    return np.clip((x - med) / sc, -clip, clip)


def choose_k(stats: dict[int, dict[str, float]], ari_min: float = ARI_MIN) -> int:
    """Frozen rule: among k with mean bootstrap ARI >= ari_min pick the minimum BIC; if none, the max-ARI k
    (ties -> smaller k)."""
    cand = [k for k, s in stats.items() if s["ari"] >= ari_min]
    if cand:
        return min(cand, key=lambda k: (stats[k]["bic"], k))
    return max(sorted(stats), key=lambda k: stats[k]["ari"])


def mean_pairwise_ari(labels: Sequence[np.ndarray]) -> float:
    from sklearn.metrics import adjusted_rand_score

    v = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return float(np.mean(v)) if v else float("nan")


def build_fit_sample() -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for d in complete_days():
        if split_of(d) != "FIT":
            continue
        for sym in SYMBOLS:
            df = load_orders(d, sym)
            ok = df["book_ok"].to_numpy() & np.all(np.isfinite(df.select(ORDER_FEATURES).to_numpy()), axis=1)
            idx = np.flatnonzero(ok)
            take = np.sort(rng.choice(idx, size=min(N_SAMPLE_PER_SYMDAY, len(idx)), replace=False))
            parts.append(df[take].with_columns(pl.lit(d).alias("date"), pl.lit(sym).alias("symbol")))
    out = pl.concat(parts, how="vertical_relaxed")
    TAX.mkdir(parents=True, exist_ok=True)
    out.write_parquet(TAX / "fit_sample.parquet")
    return out


def scalers_from(sample: pl.DataFrame) -> dict[str, dict[str, list[float]]]:
    sc = {}
    for sym in SYMBOLS:
        x = sample.filter(pl.col("symbol") == sym).select(ORDER_FEATURES).to_numpy()
        med, s = robust_scaler(x)
        sc[sym] = {"median": med.tolist(), "scale": s.tolist()}
    return sc


def std_matrix(df: pl.DataFrame, sym_col: str | None, scalers: dict, sym: str | None = None) -> np.ndarray:
    x = df.select(ORDER_FEATURES).to_numpy().astype(float)
    if sym is not None:
        s = scalers[sym]
        return standardise(x, np.array(s["median"]), np.array(s["scale"]))
    out = np.empty_like(x)
    syms = df[sym_col].to_numpy()
    for s_name, s in scalers.items():
        m = syms == s_name
        out[m] = standardise(x[m], np.array(s["median"]), np.array(s["scale"]))
    return out


def run_taxonomy() -> dict[str, Any]:
    """H-PART1 fitting on FIT only: sample, scalers, k grid (bootstrap ARI + BIC), final GMM + k-means; saved as JSON."""
    import pickle
    import time

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    t0 = time.time()
    sample = pl.read_parquet(TAX / "fit_sample.parquet") if (TAX / "fit_sample.parquet").exists() else build_fit_sample()
    scalers = scalers_from(sample)
    X = std_matrix(sample, "symbol", scalers)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    ev, pool = X[perm[:N_EVAL]], X[perm[N_EVAL:]]
    bic_x = pool[rng.choice(len(pool), size=min(N_BIC, len(pool)), replace=False)]
    stats: dict[int, dict[str, float]] = {}
    for k in K_GRID:
        labs = []
        for b in range(N_BOOT):
            xb = pool[rng.integers(0, len(pool), N_BOOT_SAMPLE)]
            g = GaussianMixture(k, covariance_type="full", random_state=SEED + 100 * k + b, max_iter=300, reg_covar=1e-4).fit(xb)
            labs.append(g.predict(ev))
        gb = GaussianMixture(k, covariance_type="full", random_state=SEED + k, n_init=2, max_iter=300, reg_covar=1e-4).fit(bic_x)
        stats[k] = {"ari": mean_pairwise_ari(labs), "bic": float(gb.bic(bic_x))}
        _log(f"taxonomy k={k} ari={stats[k]['ari']:.3f} bic={stats[k]['bic']:.0f}")
    k = choose_k(stats)
    gmm = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=3, max_iter=500, reg_covar=1e-4).fit(bic_x)
    km = KMeans(k, n_init=5, random_state=SEED).fit(bic_x)
    ari_km = float(adjusted_rand_score(gmm.predict(ev), km.predict(ev)))
    # canonical type order: by mean log notional of the component (type 0 = smallest)
    order = np.argsort(gmm.means_[:, ORDER_FEATURES.index("lognot")])
    res = {"k_stats": {str(a): b for a, b in stats.items()}, "k": k, "ari_kmeans": ari_km, "order": order.tolist(),
           "scalers": scalers, "n_sample": len(X), "features": ORDER_FEATURES, "secs": time.time() - t0,
           "rule": f"candidates ARI >= {ARI_MIN} -> min BIC, else max ARI"}
    (TAX / "taxonomy.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with (TAX / "gmm.pkl").open("wb") as f:
        pickle.dump({"gmm": gmm, "kmeans": km, "order": order}, f)
    return res


def load_gmm() -> tuple[Any, np.ndarray, dict]:
    import pickle

    with (TAX / "gmm.pkl").open("rb") as f:
        m = pickle.load(f)
    tx = json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))
    return m["gmm"], np.asarray(m["order"]), tx


def assign_types(df: pl.DataFrame, sym: str, gmm, order: np.ndarray, scalers: dict) -> np.ndarray:
    """Type id per order (canonical order; -1 where features are not finite / no book)."""
    x = df.select(ORDER_FEATURES).to_numpy().astype(float)
    ok = df["book_ok"].to_numpy() & np.all(np.isfinite(x), axis=1)
    lab = np.full(len(df), -1, dtype=np.int8)
    if ok.any():
        s = scalers[sym]
        raw = gmm.predict(standardise(x[ok], np.array(s["median"]), np.array(s["scale"])))
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        lab[ok] = inv[raw]
    return lab


# ----------------------------------------------------------------------------------------------------------------------
# minute panel (per symbol-day): base flows, per-type flows, targets
# ----------------------------------------------------------------------------------------------------------------------

N_MIN = 1440


def minute_of_ms(ms: np.ndarray, d0: int) -> np.ndarray:
    """Minute index i of an event at ms: minute i covers [d0 + 60 s i, d0 + 60 s (i + 1)); its features are known at
    T_i = d0 + 60 s (i + 1)."""
    return ((ms - d0) // 60_000).astype(np.int64)


def minute_of_book_ts(ts: np.ndarray, d0: int) -> np.ndarray:
    """Book row ts = end of its 1 s bin, the bin (ts - 1 s, ts] lies in minute (ts - d0 - 1) // 60 s."""
    return ((ts - d0 - 1) // 60_000).astype(np.int64)


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of x[i-w+1 .. i] over the finite values available (point in time)."""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(float)
    cs, cc = np.r_[0.0, np.cumsum(v)], np.r_[0.0, np.cumsum(c)]
    i = np.arange(len(x))
    lo = np.maximum(i - w + 1, 0)
    n = cc[i + 1] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[i + 1] - cs[lo]) / n, np.nan)


def trailing_sum(x: np.ndarray, w: int) -> np.ndarray:
    v = np.where(np.isfinite(x), x, 0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    i = np.arange(len(x))
    return cs[i + 1] - cs[np.maximum(i - w + 1, 0)]


def forward_sum(x: np.ndarray, h: int) -> np.ndarray:
    """sum x[i+1 .. i+h]; NaN where the window passes the end of the day."""
    v = np.where(np.isfinite(x), x, 0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    n = len(x)
    i = np.arange(n)
    out = np.full(n, np.nan)
    ok = i + h < n
    out[ok] = cs[i[ok] + h + 1] - cs[i[ok] + 1]
    return out


def ewm_var(r: np.ndarray, half_life: float = 30.0) -> np.ndarray:
    """Adjusted EWMA of r^2 over the available past (point in time, NaN-skipping)."""
    lam = 0.5 ** (1.0 / half_life)
    num = den = 0.0
    out = np.full(len(r), np.nan)
    for i, x in enumerate(r):
        num *= lam
        den *= lam
        if np.isfinite(x):
            num += x * x
            den += 1.0
        out[i] = num / den if den > 0 else np.nan
    return out


def minute_as(ms: np.ndarray, side: np.ndarray, as_h: np.ndarray, d0: int, min_orders: int = 5) -> np.ndarray:
    """Equal-weighted mean touch-maker loss per minute (NaN if < min_orders valid orders)."""
    i = minute_of_ms(ms, d0)
    ok = np.isfinite(as_h) & (i >= 0) & (i < N_MIN)
    s = np.bincount(i[ok], as_h[ok], minlength=N_MIN)
    c = np.bincount(i[ok], minlength=N_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= min_orders, s / np.maximum(c, 1), np.nan)


def minute_panel(d: str, sym: str, orders: pl.DataFrame, types: np.ndarray, book: pl.DataFrame, k: int) -> pl.DataFrame:
    """One row per minute i (known at T_i = d0 + 60 s (i + 1)); features from minutes <= i, targets after T_i."""
    d0 = day0_ms(d)
    bts = book["ts"].to_numpy()
    bm = minute_of_book_ts(bts, d0)
    okb = (bm >= 0) & (bm < N_MIN)
    bm = bm[okb]

    def bsum(col):
        return np.bincount(bm, book[col].to_numpy().astype(float)[okb], minlength=N_MIN)

    def bmean(arr):
        a = arr[okb]
        f = np.isfinite(a)
        s = np.bincount(bm[f], a[f], minlength=N_MIN)
        c = np.bincount(bm[f], minlength=N_MIN)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, np.nan)

    # mid at T_i: the book row with ts = T_i
    T = d0 + 60_000 * (np.arange(N_MIN) + 1)
    j = np.searchsorted(bts, T, side="right") - 1
    mid_all = book["mid"].to_numpy().astype(float)
    mid = np.where(j >= 0, mid_all[np.clip(j, 0, len(bts) - 1)], np.nan)
    l1 = 0.5 * (book["bid1"].to_numpy() * book["bid1_size"].to_numpy() + book["ask1"].to_numpy() * book["ask1_size"].to_numpy())
    # 10 s mid log returns -> squared, summed per minute
    t10 = d0 + 10_000 * np.arange(N_MIN * 6 + 1)
    j10 = np.searchsorted(bts, t10, side="right") - 1
    m10 = np.where(j10 >= 0, mid_all[np.clip(j10, 0, len(bts) - 1)], np.nan)
    r10 = np.diff(np.log(m10))
    rv = np.nansum(r10.reshape(N_MIN, 6) ** 2, axis=1)
    rv[np.isnan(r10.reshape(N_MIN, 6)).all(axis=1)] = np.nan

    ms = orders["ms"].to_numpy()
    om = minute_of_ms(ms, d0)
    side = orders["side"].to_numpy().astype(float)
    notl = orders["notional"].to_numpy()
    oko = (om >= 0) & (om < N_MIN)

    def osum(w, mask=None):
        m = oko if mask is None else (oko & mask)
        return np.bincount(om[m], w[m], minlength=N_MIN)

    buy, sell = osum(notl, side > 0), osum(notl, side < 0)
    n_all = osum(np.ones(len(ms)))
    cols: dict[str, np.ndarray] = {
        "minute": np.arange(N_MIN), "ts": T, "mid": mid, "buy": buy, "sell": sell, "n_orders": n_all,
        "ofi": bsum("ofi"), "l1": bmean(l1), "spread": bmean(book["rel_spread_bp"].to_numpy().astype(float)),
        "n_updates": bsum("n_updates"), "add": bsum("add_bid") + bsum("add_ask"), "rem": bsum("rem_bid") + bsum("rem_ask"),
        "refill": bsum("refill_bid") + bsum("refill_ask"), "rv": rv,
    }
    for t in range(k):
        m = types == t
        cols[f"b{t}"] = osum(notl, m & (side > 0))
        cols[f"s{t}"] = osum(notl, m & (side < 0))
        cols[f"n{t}"] = osum(np.ones(len(ms)), m)
    as10, as60 = orders["as_10s"].to_numpy(), orders["as_60s"].to_numpy()
    cols["as10"] = minute_as(ms, side, as10, d0)
    cols["as60"] = minute_as(ms, side, as60, d0)
    df = pl.DataFrame(cols).with_columns(pl.lit(d).alias("date"), pl.lit(sym).alias("symbol"))
    return add_features(df, k)


def add_features(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Point-in-time features and forward targets of one symbol-day minute panel (sorted by minute)."""
    g = {c: df[c].to_numpy().astype(float) for c in df.columns if c not in ("date", "symbol")}
    lm = np.log(g["mid"])
    r1 = np.r_[np.nan, np.diff(lm)]
    sig = np.sqrt(ewm_var(r1))
    sig = np.where(sig > 1e-6, sig, np.nan)
    tot = g["buy"] + g["sell"]
    l1bar = trailing_mean(g["l1"], 60)
    tot5 = trailing_sum(tot, 5)
    f: dict[str, np.ndarray] = {"sigma": sig}
    with np.errstate(invalid="ignore", divide="ignore"):
        f["ofi1"] = g["ofi"] / l1bar
        f["ofi5"] = trailing_sum(g["ofi"], 5) / l1bar
        f["imb1"] = np.where(tot > 0, (g["buy"] - g["sell"]) / tot, 0.0)
        f["imb5"] = np.where(tot5 > 0, (trailing_sum(g["buy"], 5) - trailing_sum(g["sell"], 5)) / tot5, 0.0)
        f["rl1"] = r1 / sig
        f["rl5"] = (lm - np.r_[np.full(5, np.nan), lm[:-5]]) / (sig * np.sqrt(5))
        f["rl60"] = (lm - np.r_[np.full(60, np.nan), lm[:-60]]) / (sig * np.sqrt(60))
        n5 = trailing_sum(g["n_orders"], 5)
        for t in range(k):
            tt = g[f"b{t}"] + g[f"s{t}"]
            f[f"sg{t}_1"] = np.where(tot > 0, (g[f"b{t}"] - g[f"s{t}"]) / tot, 0.0)
            f[f"sg{t}_5"] = np.where(tot5 > 0, (trailing_sum(g[f"b{t}"], 5) - trailing_sum(g[f"s{t}"], 5)) / tot5, 0.0)
            f[f"sh{t}_5"] = np.where(tot5 > 0, trailing_sum(tt, 5) / tot5, 0.0)
            f[f"cs{t}_5"] = np.where(n5 > 0, trailing_sum(g[f"n{t}"], 5) / n5, 0.0)
            f[f"ab{t}_1"] = np.abs(f[f"sg{t}_1"])
        # vol / AS baselines
        lrv5, lrv60 = np.log(trailing_sum(g["rv"], 5) + 1e-12), np.log(trailing_sum(g["rv"], 60) + 1e-12)
        f["lrv5"], f["lrv60"] = lrv5, lrv60
        f["lnot5"] = np.log1p(tot5 / 5.0) - trailing_mean(np.log1p(tot), 60)
        f["absr1"] = np.abs(f["rl1"])
        f["lspread"] = np.log(np.maximum(g["spread"], 1e-3))
        f["as10_l"] = np.r_[np.nan, g["as10"][:-1]]
        f["as60_l"] = np.r_[np.nan, g["as60"][:-1]]
        f["absimb1"], f["absofi1"] = np.abs(f["imb1"]), np.abs(f["ofi1"])
        # HMM inputs (unstandardised)
        f["h_lognot"] = np.log1p(tot) - trailing_mean(np.log1p(tot), 60)
        f["h_imb"] = f["imb1"]
        f["h_absr"] = f["absr1"]
        f["h_ofi"] = f["ofi1"]
        f["h_upd"] = np.log1p(g["n_updates"]) - trailing_mean(np.log1p(g["n_updates"]), 60)
        cxl = np.maximum(g["rem"] - tot, 0.0)
        f["h_cxl"] = np.log((1.0 + cxl) / (1.0 + g["add"]))
        f["h_refill"] = g["refill"] / (1.0 + g["add"])
        f["h_spread"] = f["lspread"]
        # targets
        for h in HORIZONS_MIN:
            fut = np.r_[lm[h:], np.full(h, np.nan)]
            f[f"y_r{h}"] = np.clip((fut - lm) / (sig * np.sqrt(h)), -10, 10)
        for h in (5, 60):
            f[f"y_rv{h}"] = np.log(forward_sum(g["rv"], h) + 1e-12)
        f["y_as10"] = np.r_[g["as10"][1:], np.nan]
        f["y_as60"] = np.r_[g["as60"][1:], np.nan]
    return df.with_columns([pl.Series(c, v) for c, v in f.items()])


def run_assign() -> dict[str, Any]:
    """Assign types to every order of every complete day; per (date, symbol, type) descriptors; minute panels."""
    gmm, order, tx = load_gmm()
    k, scalers = tx["k"], tx["scalers"]
    (OUT / "minute_raw").mkdir(parents=True, exist_ok=True)
    pdir = TAX / "per_day"
    pdir.mkdir(parents=True, exist_ok=True)
    for d in complete_days():
        if (OUT / "minute_raw" / f"{d}.parquet").exists() and (pdir / f"desc_{d}.parquet").exists():
            continue
        desc_rows, trait_rows = [], []
        rng = np.random.default_rng(SEED + 1 + all_days().index(d))
        panels = []
        for sym in SYMBOLS:
            book = load_book(d, sym)
            o = load_orders(d, sym, book)
            lab = assign_types(o, sym, gmm, order, scalers)
            tdir = OUT / "days" / d / "types"
            tdir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"ms": o["ms"], "type": lab}).write_parquet(tdir / f"{sym}.parquet", compression="zstd")
            hour = ((o["ms"].to_numpy() - day0_ms(d)) // 3_600_000).astype(int)
            o2 = o.with_columns(pl.Series("type", lab), pl.Series("hour", hour))
            for t in range(-1, k):
                s = o2.filter(pl.col("type") == t)
                if s.height == 0:
                    continue
                desc_rows.append({"date": d, "symbol": sym, "type": t, "n": s.height, "notional": float(s["notional"].sum()),
                                  **{f"sum_{c}": float(s[c].fill_nan(None).drop_nulls().sum()) for c in ("mm_10s", "mm_60s", "as_10s", "as_60s")},
                                  **{f"cnt_{c}": int(s[c].fill_nan(None).drop_nulls().len()) for c in ("mm_60s", "as_60s")},
                                  **{f"h{h:02d}": int((s["hour"] == h).sum()) for h in range(24)},
                                  "buy_share": float((s["side"] > 0).mean())})
            # trait sample for descriptions (all days, 400 per symbol-day)
            ok = np.flatnonzero(lab >= 0)
            take = np.sort(rng.choice(ok, size=min(400, len(ok)), replace=False)) if len(ok) else ok
            trait_rows.append(o2[take].select(["ms", "side", "size", "notional", "sweep_bp", "n_prints", "spread_bp", "type",
                                               *ORDER_FEATURES, "mm_60s", "as_60s"]).with_columns(pl.lit(d).alias("date"), pl.lit(sym).alias("symbol")))
            panels.append(minute_panel(d, sym, o, lab, book, k))
        pl.concat(panels, how="vertical_relaxed").write_parquet(OUT / "minute_raw" / f"{d}.parquet", compression="zstd")
        pl.DataFrame(desc_rows).write_parquet(pdir / f"desc_{d}.parquet")
        pl.concat(trait_rows, how="vertical_relaxed").write_parquet(pdir / f"trait_{d}.parquet")
        _log(f"assign {d} done")
    pl.concat([pl.read_parquet(f) for f in sorted(pdir.glob("desc_*.parquet"))], how="vertical_relaxed").write_parquet(TAX / "type_day_stats.parquet")
    pl.concat([pl.read_parquet(f) for f in sorted(pdir.glob("trait_*.parquet"))], how="vertical_relaxed").write_parquet(TAX / "trait_sample.parquet")
    return {"k": k, "days": len(complete_days())}


# ----------------------------------------------------------------------------------------------------------------------
# flow-state HMM (fit on FIT, forward filter only)
# ----------------------------------------------------------------------------------------------------------------------

HMM_DIR = OUT / "hmm"
HMM_FIRST_MIN = 15
N_HMM_BOOT, N_HMM_BOOT_SEQ, N_HMM_EVAL_SEQ = 10, 60, 30


def logsumexp_rows(a: np.ndarray) -> np.ndarray:
    m = np.max(a, axis=1, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=1, keepdims=True)))[:, 0]


def forward_filter(log_b: np.ndarray, start: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Filtered state probabilities p(s_t | x_1..t) (no smoothing: point in time). log_b: (T, K) log emissions."""
    T, K = log_b.shape
    out = np.empty((T, K))
    alpha = np.log(np.maximum(start, 1e-300)) + log_b[0]
    alpha -= logsumexp_rows(alpha[None, :])[0]
    out[0] = np.exp(alpha)
    for t in range(1, T):
        alpha = np.log(np.maximum(np.exp(alpha) @ trans, 1e-300)) + log_b[t]   # alpha is normalised: exp is safe
        alpha -= logsumexp_rows(alpha[None, :])[0]
        out[t] = np.exp(alpha)
    return out


def diag_gauss_logpdf(x: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    """(T, K) log N(x_t | mean_k, diag var_k)."""
    d = x[:, None, :] - means[None, :, :]
    return -0.5 * (np.sum(d * d / var[None], axis=2) + np.sum(np.log(2 * np.pi * var), axis=1)[None, :])


def hmm_sequences(days: Sequence[str], scal: dict | None = None) -> tuple[list[np.ndarray], list[tuple[str, str]], dict]:
    frames = {d: pl.read_parquet(OUT / "minute_raw" / f"{d}.parquet") for d in days}
    if scal is None:
        allx = pl.concat([f.filter(pl.col("minute") >= HMM_FIRST_MIN) for f in frames.values()])
        scal = {}
        for sym in SYMBOLS:
            x = allx.filter(pl.col("symbol") == sym).select(HMM_FEATURES).to_numpy().astype(float)
            x = np.where(np.isfinite(x), x, np.nan)
            med, sc = robust_scaler(x)
            scal[sym] = {"median": med.tolist(), "scale": sc.tolist()}
    seqs, keys = [], []
    for d, f in frames.items():
        for sym in SYMBOLS:
            x = f.filter((pl.col("symbol") == sym) & (pl.col("minute") >= HMM_FIRST_MIN)).sort("minute").select(HMM_FEATURES).to_numpy().astype(float)
            z = standardise(x, np.array(scal[sym]["median"]), np.array(scal[sym]["scale"]))
            seqs.append(np.where(np.isfinite(z), z, 0.0))
            keys.append((d, sym))
    return seqs, keys, scal


def run_hmm() -> dict[str, Any]:
    import pickle
    import time

    from hmmlearn.hmm import GaussianHMM

    t0 = time.time()
    HMM_DIR.mkdir(parents=True, exist_ok=True)
    fit_days = [d for d in complete_days() if split_of(d) == "FIT"]
    seqs, keys, scal = hmm_sequences(fit_days)
    rng = np.random.default_rng(SEED + 7)
    ev_idx = rng.choice(len(seqs), size=min(N_HMM_EVAL_SEQ, len(seqs)), replace=False)
    ev = np.concatenate([seqs[i] for i in ev_idx])
    ev_len = [len(seqs[i]) for i in ev_idx]
    X_all, L_all = np.concatenate(seqs), [len(s) for s in seqs]
    D = X_all.shape[1]
    stats: dict[int, dict[str, float]] = {}
    for K in HMM_GRID:
        labs = []
        for b in range(N_HMM_BOOT):
            idx = rng.integers(0, len(seqs), N_HMM_BOOT_SEQ)
            m = GaussianHMM(K, covariance_type="diag", n_iter=50, tol=1e-3, random_state=SEED + 10 * K + b)
            m.fit(np.concatenate([seqs[i] for i in idx]), [len(seqs[i]) for i in idx])
            labs.append(m.predict(ev, ev_len))
        mf = GaussianHMM(K, covariance_type="diag", n_iter=100, tol=1e-3, random_state=SEED + K).fit(X_all, L_all)
        ll = mf.score(X_all, L_all)
        p = K * (K - 1) + (K - 1) + 2 * K * D
        stats[K] = {"ari": mean_pairwise_ari(labs), "bic": float(-2 * ll + p * np.log(len(X_all))), "ll": float(ll)}
        _log(f"hmm K={K} ari={stats[K]['ari']:.3f} bic={stats[K]['bic']:.0f}")
        with (HMM_DIR / f"hmm_K{K}.pkl").open("wb") as f:
            pickle.dump(mf, f)
    K = choose_k(stats)
    with (HMM_DIR / f"hmm_K{K}.pkl").open("rb") as f:
        model = pickle.load(f)
    order = np.argsort(model.means_[:, HMM_FEATURES.index("h_absr")])   # state 0 = calmest
    var = np.array([np.diag(c) for c in model.covars_])
    res = {"K_stats": {str(a): b for a, b in stats.items()}, "K": int(K), "order": order.tolist(), "scalers": scal,
           "features": HMM_FEATURES, "means": model.means_[order].tolist(), "var": var[order].tolist(),
           "startprob": model.startprob_[order].tolist(), "transmat": model.transmat_[np.ix_(order, order)].tolist(),
           "n_fit_rows": int(len(X_all)), "secs": time.time() - t0}
    (HMM_DIR / "hmm.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def run_state_features() -> None:
    """Forward-filter every complete day with the frozen HMM and write the participant-state feature set."""
    hm = json.loads((HMM_DIR / "hmm.json").read_text(encoding="utf-8"))
    K = hm["K"]
    means, var = np.array(hm["means"]), np.array(hm["var"])
    start, trans = np.array(hm["startprob"]), np.array(hm["transmat"])
    sdir = OUT / "state_1m"
    sdir.mkdir(parents=True, exist_ok=True)
    for d in complete_days():
        f = pl.read_parquet(OUT / "minute_raw" / f"{d}.parquet")
        seqs, keys, _ = hmm_sequences([d], hm["scalers"])
        parts = []
        for x, (_, sym) in zip(seqs, keys):
            prob = forward_filter(diag_gauss_logpdf(x, means, var), start, trans)
            st = prob.argmax(axis=1)
            chg = np.r_[0, (st[1:] != st[:-1]).astype(int)]
            g = f.filter(pl.col("symbol") == sym).sort("minute")
            pad = g.height - len(x)
            cols = {f"hp{s}": np.r_[np.full(pad, np.nan), prob[:, s]] for s in range(K)}
            cols["hstate"] = np.r_[np.full(pad, -1), st]
            cols["hchg"] = np.r_[np.zeros(pad), chg]
            parts.append(g.with_columns([pl.Series(c, v) for c, v in cols.items()]))
        out = pl.concat(parts).with_columns(pl.col("hstate").cast(pl.Int8))
        out = out.with_columns(pl.lit(split_of(d)).alias("split"), pl.lit(segment_of(d)).alias("segment"))
        out.write_parquet(sdir / f"{d}.parquet", compression="zstd")
    _log("state features written")


# ----------------------------------------------------------------------------------------------------------------------
# supervised: OOS dR2, purged blocked CV on FIT, one HELD evaluation, day-block bootstrap
# ----------------------------------------------------------------------------------------------------------------------

SUP = OUT / "supervised"
N_CV_BLOCKS, N_BOOT_DAYS = 6, 2000
FIRST_ROW_MIN = 60


def cv_blocks(days: Sequence[str], n_blocks: int = N_CV_BLOCKS, embargo_days: int = 1) -> list[tuple[list[str], list[str]]]:
    """Chronological blocks of days; train = other days farther than `embargo_days` calendar days from every test day."""
    ds = sorted(days)
    blocks = [list(b) for b in np.array_split(np.array(ds), n_blocks) if len(b)]
    out = []
    for b in blocks:
        tb = [date.fromisoformat(x) for x in b]
        tr = [x for x in ds if x not in b and min(abs((date.fromisoformat(x) - t).days) for t in tb) > embargo_days]
        out.append((tr, b))
    return out


def day_boot(per_day: dict[str, tuple[float, float, float]], n_boot: int = N_BOOT_DAYS, seed: int = SEED) -> dict[str, float]:
    """per_day: day -> (SSE_B, SSE_P, SST). dR2 = (sum SSE_B - sum SSE_P) / sum SST; percentile CI and one-sided p
    (share of resamples with dR2 <= 0) by resampling days with replacement."""
    a = np.array(list(per_day.values()), float)
    if len(a) == 0:
        return {"dr2": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n_days": 0}
    point = (a[:, 0].sum() - a[:, 1].sum()) / a[:, 2].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    s = a[idx].sum(axis=1)
    bs = (s[:, 0] - s[:, 1]) / s[:, 2]
    return {"dr2": float(point), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(np.mean(bs <= 0)), "n_days": int(len(a))}


def holm(p: Sequence[float]) -> list[float]:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def design(df: pl.DataFrame, cols: Sequence[str], lo: np.ndarray | None = None, hi: np.ndarray | None = None):
    """X with intercept + symbol dummies; features clipped at the train 0.1 / 99.9 percentiles."""
    x = df.select(cols).to_numpy().astype(float)
    if lo is None:
        lo, hi = np.nanpercentile(x, 0.1, axis=0), np.nanpercentile(x, 99.9, axis=0)
    x = np.clip(x, lo, hi)
    sym = df["symbol"].to_numpy()
    dummies = np.column_stack([(sym == s).astype(float) for s in SYMBOLS[1:]])
    return np.column_stack([np.ones(len(x)), dummies, x]), lo, hi


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def feature_sets(k: int, K: int) -> dict[str, dict[str, list[str]]]:
    hm = [f"hp{s}" for s in range(1, K)] + ["hchg"]
    sg = [f"sg{t}_{w}" for t in range(k) for w in (1, 5)]
    sh = [f"sh{t}_5" for t in range(k - 1)]
    cs = [f"cs{t}_5" for t in range(k - 1)]
    ab = [f"ab{t}_1" for t in range(k)]
    b_ret = ["ofi1", "ofi5", "imb1", "imb5", "rl1", "rl5", "rl60"]
    b_vol = ["lrv5", "lrv60", "lnot5", "absr1", "lspread"]
    b_as = ["as10_l", "as60_l", "lspread", "lrv5", "absimb1", "absofi1"]
    return {"ret": {"B": b_ret, "P": b_ret + sg + sh + hm}, "vol": {"B": b_vol, "P": b_vol + sh + cs + hm},
            "as": {"B": b_as, "P": b_as + cs + sh + ab + hm}}


TARGETS = {"y_r1": "ret", "y_r5": "ret", "y_r60": "ret", "y_rv5": "vol", "y_rv60": "vol", "y_as10": "as", "y_as60": "as"}


def load_panel(cols: Sequence[str] | None = None) -> pl.DataFrame:
    """Rows from 01:00 UTC of every state_1m day; only `cols` (+ ids) to keep memory low (float32 features)."""
    ids = ["date", "symbol", "minute", "ts", "split", "segment"]
    fs = sorted((OUT / "state_1m").glob("*.parquet"))
    parts = []
    for f in fs:
        g = pl.read_parquet(f, columns=None if cols is None else list(dict.fromkeys(ids + list(cols))))
        g = g.filter(pl.col("minute") >= FIRST_ROW_MIN)
        parts.append(g.with_columns([pl.col(c).cast(pl.Float64) for c in g.columns if c not in ids]))
    return pl.concat(parts, how="vertical_relaxed")


def per_day_sse(df: pl.DataFrame, y: np.ndarray, pb: np.ndarray, pp: np.ndarray, center: float) -> dict[str, tuple[float, float, float]]:
    days = df["date"].to_numpy()
    out = {}
    for d in np.unique(days):
        m = days == d
        out[str(d)] = (float(np.sum((y[m] - pb[m]) ** 2)), float(np.sum((y[m] - pp[m]) ** 2)), float(np.sum((y[m] - center) ** 2)))
    return out


def eval_target(panel: pl.DataFrame, target: str, B: list[str], P: list[str], zero_center: bool) -> dict[str, Any]:
    cols = list(dict.fromkeys(P + [target]))
    d = panel.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in cols]))
    fit = d.filter(pl.col("split") == "FIT")
    held = d.filter(pl.col("split") == "HELD")
    fit_days = sorted(set(fit["date"].to_list()))
    res: dict[str, Any] = {"target": target, "n_fit": fit.height, "n_held": held.height, "B": B, "P": P}
    # FIT CV
    cv_days: dict[str, tuple[float, float, float]] = {}
    oos_parts = []
    for tr_days, te_days in cv_blocks(fit_days):
        tr = fit.filter(pl.col("date").is_in(tr_days))
        te = fit.filter(pl.col("date").is_in(te_days))
        y_tr, y_te = tr[target].to_numpy(), te[target].to_numpy()
        center = 0.0 if zero_center else float(y_tr.mean())
        Xb, lo, hi = design(tr, B)
        Xp, lo2, hi2 = design(tr, P)
        pb = design(te, B, lo, hi)[0] @ ols_fit(Xb, y_tr)
        pp = design(te, P, lo2, hi2)[0] @ ols_fit(Xp, y_tr)
        cv_days.update(per_day_sse(te, y_te, pb, pp, center))
        oos_parts.append(te.select(["date", "symbol", "minute"]).with_columns(pl.Series("pb", pb), pl.Series("pp", pp), pl.Series("y", y_te)))
    seg = {}
    for s_name in ("llm_in_sample", "clean_holdout"):
        sub = {k: v for k, v in cv_days.items() if segment_of(k) == s_name}
        seg[s_name] = day_boot(sub)
    res["fit_cv"] = {"all": day_boot(cv_days), **seg}
    # HELD once
    y_f, y_h = fit[target].to_numpy(), held[target].to_numpy()
    center = 0.0 if zero_center else float(y_f.mean())
    Xb, lo, hi = design(fit, B)
    Xp, lo2, hi2 = design(fit, P)
    bb, bp = ols_fit(Xb, y_f), ols_fit(Xp, y_f)
    pb = design(held, B, lo, hi)[0] @ bb
    pp = design(held, P, lo2, hi2)[0] @ bp
    hd = per_day_sse(held, y_h, pb, pp, center)
    res["held"] = day_boot(hd)
    a = np.array(list(hd.values()))
    res["held"]["r2_B"] = float(1 - a[:, 0].sum() / a[:, 2].sum())
    res["held"]["r2_P"] = float(1 - a[:, 1].sum() / a[:, 2].sum())
    res["held_by_symbol"] = {}
    for sym in SYMBOLS:
        m = held["symbol"].to_numpy() == sym
        if m.sum():
            sst = float(np.sum((y_h[m] - center) ** 2))
            res["held_by_symbol"][sym] = float((np.sum((y_h[m] - pb[m]) ** 2) - np.sum((y_h[m] - pp[m]) ** 2)) / sst)
    # coefficients of P on FIT with day-clustered SEs
    import statsmodels.api as sm

    names = ["const"] + [f"sym_{s}" for s in SYMBOLS[1:]] + P
    ols = sm.OLS(y_f, Xp).fit(cov_type="cluster", cov_kwds={"groups": fit["date"].cast(pl.Categorical).to_physical().to_numpy()})
    res["coef_P_fit"] = {n: {"b": float(b), "t": float(t)} for n, b, t in zip(names, ols.params, ols.tvalues)}
    # joint Wald test of the participant block
    extra = [i for i, n in enumerate(names) if n not in B and not n.startswith("sym_") and n != "const"]
    if extra:
        R = np.zeros((len(extra), len(names)))
        for r, i in enumerate(extra):
            R[r, i] = 1.0
        w = ols.wald_test(R, scalar=True)
        res["wald_P_block_fit"] = {"stat": float(w.statistic), "p": float(w.pvalue), "df": len(extra)}
    res["_pred"] = {"held": held.select(["date", "symbol", "minute", "ts"]).with_columns(pl.Series("pb", pb), pl.Series("pp", pp), pl.Series("y", y_h)),
                    "fit_oos": pl.concat(oos_parts), "beta_P": bp, "clip": (lo2, hi2), "fit_insample_pp": Xp @ bp}
    return res


def logistic_eval(panel: pl.DataFrame, target: str, B: list[str], P: list[str], fit_oos: pl.DataFrame) -> dict[str, Any]:
    """Secondary: P(y > 0) by logistic regression on B and on P; Platt map fitted on FIT-CV OOS decision values (from
    models refit per CV block), applied to the HELD decision values of the full-FIT model; Brier / log-loss on HELD."""
    from sklearn.linear_model import LogisticRegression

    cols = list(dict.fromkeys(P + [target]))
    d = panel.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in cols]))
    fit, held = d.filter(pl.col("split") == "FIT"), d.filter(pl.col("split") == "HELD")
    out = {}
    fit_days = sorted(set(fit["date"].to_list()))
    for name, cs in (("B", B), ("P", P)):
        oos_s, oos_y = [], []
        for tr_days, te_days in cv_blocks(fit_days):
            tr, te = fit.filter(pl.col("date").is_in(tr_days)), fit.filter(pl.col("date").is_in(te_days))
            Xtr, lo, hi = design(tr, cs)
            mu, sd = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0) + 1e-12
            m = LogisticRegression(C=1.0, max_iter=500).fit((Xtr[:, 1:] - mu) / sd, tr[target].to_numpy() > 0)
            Xte = design(te, cs, lo, hi)[0]
            oos_s.append(m.decision_function((Xte[:, 1:] - mu) / sd))
            oos_y.append(te[target].to_numpy() > 0)
        s_oos, y_oos = np.concatenate(oos_s), np.concatenate(oos_y)
        platt = LogisticRegression(C=1e6, max_iter=500).fit(s_oos[:, None], y_oos)
        Xf, lo, hi = design(fit, cs)
        mu, sd = Xf[:, 1:].mean(0), Xf[:, 1:].std(0) + 1e-12
        m = LogisticRegression(C=1.0, max_iter=500).fit((Xf[:, 1:] - mu) / sd, fit[target].to_numpy() > 0)
        Xh = design(held, cs, lo, hi)[0]
        ph = platt.predict_proba(m.decision_function((Xh[:, 1:] - mu) / sd)[:, None])[:, 1]
        yh = held[target].to_numpy() > 0
        ph = np.clip(ph, 1e-6, 1 - 1e-6)
        bins = np.clip((ph * 10).astype(int), 0, 9)
        out[name] = {"brier": float(np.mean((ph - yh) ** 2)), "logloss": float(-np.mean(yh * np.log(ph) + (~yh) * np.log(1 - ph))),
                     "reliability": [{"bin": int(b), "n": int((bins == b).sum()), "p_mean": float(ph[bins == b].mean()),
                                      "y_rate": float(yh[bins == b].mean())} for b in range(10) if (bins == b).sum() >= 200]}
    base = float((fit[target].to_numpy() > 0).mean())
    yh = held[target].to_numpy() > 0
    out["climatology_brier"] = float(np.mean((base - yh) ** 2))
    out["brier_skill_P_vs_B"] = 1 - out["P"]["brier"] / out["B"]["brier"]
    return out


def gbm_eval(panel: pl.DataFrame, target: str, P: list[str], zero_center: bool, lin_pp_held: np.ndarray) -> dict[str, Any]:
    """Registered challenger: LightGBM depth <= 3 on P features vs the linear P model on HELD."""
    import lightgbm as lgb

    cols = list(dict.fromkeys(P + [target]))
    d = panel.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in cols]))
    fit, held = d.filter(pl.col("split") == "FIT"), d.filter(pl.col("split") == "HELD")
    Xf, lo, hi = design(fit, P)
    Xh = design(held, P, lo, hi)[0]
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, num_leaves=7, min_child_samples=500,
                          subsample=0.8, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1)
    m.fit(Xf, fit[target].to_numpy())
    pg = m.predict(Xh)
    y = held[target].to_numpy()
    center = 0.0 if zero_center else float(fit[target].to_numpy().mean())
    pdays = per_day_sse(held, y, lin_pp_held, pg, center)
    return {"gbm_vs_linearP": day_boot(pdays)}


def run_supervised() -> dict[str, Any]:
    import pickle

    SUP.mkdir(parents=True, exist_ok=True)
    tx = json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))
    hm = json.loads((HMM_DIR / "hmm.json").read_text(encoding="utf-8"))
    fs = feature_sets(tx["k"], hm["K"])
    need = sorted({c for fam in fs.values() for c in fam["P"]} | set(TARGETS))
    panel = load_panel(need)
    out: dict[str, Any] = {"k": tx["k"], "K": hm["K"], "feature_sets": fs, "n_rows": panel.height}
    preds = {}
    for tgt, fam in TARGETS.items():
        r = eval_target(panel, tgt, fs[fam]["B"], fs[fam]["P"], zero_center=(fam == "ret"))
        preds[tgt] = r.pop("_pred")
        gate = r["fit_cv"]["llm_in_sample"]["dr2"] > 0 and r["fit_cv"]["clean_holdout"]["dr2"] > 0
        r["gate_fit_cv_both_segments"] = bool(gate)
        if gate:
            r["gbm"] = gbm_eval(panel, tgt, fs[fam]["P"], fam == "ret", preds[tgt]["held"]["pp"].to_numpy())
        if tgt in ("y_r5", "y_r60"):
            r["logistic"] = logistic_eval(panel, tgt, fs[fam]["B"], fs[fam]["P"], preds[tgt]["fit_oos"])
        out[tgt] = r
        _log(f"supervised {tgt}: held dR2 {r['held']['dr2']:.5f} [{r['held']['lo']:.5f}, {r['held']['hi']:.5f}] "
             f"cv llm {r['fit_cv']['llm_in_sample']['dr2']:.5f} clean {r['fit_cv']['clean_holdout']['dr2']:.5f}")
    for fam, tg in (("ret", ["y_r1", "y_r5", "y_r60"]), ("vol", ["y_rv5", "y_rv60"]), ("as", ["y_as10", "y_as60"])):
        adj = holm([out[t]["held"]["p"] for t in tg])
        for t, a in zip(tg, adj):
            out[t]["held"]["p_holm"] = a
    (SUP / "supervised.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    with (SUP / "preds.pkl").open("wb") as f:
        pickle.dump(preds, f)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# H-PART2 informedness, H-PART6 executable, H-PART7 xvenue, H-PART8 tape sanity
# ----------------------------------------------------------------------------------------------------------------------


def day_t(values_by_day: dict[str, float]) -> dict[str, float]:
    v = np.array([x for x in values_by_day.values() if np.isfinite(x)])
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else float("nan"), "t": float("nan"), "n_days": int(len(v))}
    return {"mean": float(v.mean()), "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), "n_days": int(len(v))}


def run_informedness() -> dict[str, Any]:
    """H-PART2: per type, symbol-day mean markout minus the all-order mean, averaged per day, t over days."""
    st = pl.read_parquet(TAX / "type_day_stats.parquet")
    k = json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))["k"]
    allm = st.group_by(["date", "symbol"]).agg((pl.col("sum_mm_60s").sum() / pl.col("cnt_mm_60s").sum()).alias("all60"),
                                              (pl.col("sum_mm_10s").sum() / pl.col("cnt_mm_60s").sum()).alias("all10"),
                                              (pl.col("sum_as_60s").sum() / pl.col("cnt_as_60s").sum()).alias("allas60"))
    s = st.join(allm, on=["date", "symbol"]).with_columns(
        (pl.col("sum_mm_60s") / pl.col("cnt_mm_60s") - pl.col("all60")).alias("d60"),
        (pl.col("sum_mm_10s") / pl.col("cnt_mm_60s") - pl.col("all10")).alias("d10"),
        (pl.col("sum_mm_60s") / pl.col("cnt_mm_60s")).alias("m60"),
        (pl.col("sum_as_60s") / pl.col("cnt_as_60s")).alias("as60"))
    out: dict[str, Any] = {}
    for t in range(k):
        g = s.filter(pl.col("type") == t)
        r = {}
        for name, filt in (("FIT", pl.col("date") < HELD_FIRST), ("HELD", pl.col("date") >= HELD_FIRST),
                           ("FIT_llm_in_sample", pl.col("date") < CUTOFF),
                           ("FIT_clean", (pl.col("date") >= CUTOFF) & (pl.col("date") < HELD_FIRST))):
            gg = g.filter(filt).group_by("date").agg(pl.col("d60").mean(), pl.col("d10").mean(), pl.col("m60").mean(), pl.col("as60").mean())
            r[name] = {"d60": day_t(dict(zip(gg["date"], gg["d60"]))), "d10": day_t(dict(zip(gg["date"], gg["d10"]))),
                       "m60": day_t(dict(zip(gg["date"], gg["m60"]))), "taker_vs_touch_60": day_t(dict(zip(gg["date"], gg["as60"])))}
        out[str(t)] = r
    passing = [t for t, r in out.items() if abs(r["FIT"]["d60"]["t"]) > 3 and abs(r["FIT"]["d60"]["mean"]) > 0.5
               and np.sign(r["HELD"]["d60"]["mean"]) == np.sign(r["FIT"]["d60"]["mean"]) and abs(r["HELD"]["d60"]["t"]) > 2]
    any_fit2 = any(abs(r["FIT"]["d60"]["t"]) > 2 for r in out.values())
    out["_verdict"] = {"passing_types": passing, "any_fit_t_gt_2": any_fit2,
                       "verdict": "supports" if passing else ("contradicts" if not any_fit2 else "inconclusive")}
    (TAX / "informedness.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def boot_mean_by_day(sums: dict[str, tuple[float, int]], n_boot: int = N_BOOT_DAYS, seed: int = SEED) -> dict[str, float]:
    """Pooled mean (sum / count) with a day-block bootstrap CI and one-sided p (share of resamples with mean <= 0)."""
    a = np.array([v for v in sums.values() if v[1] > 0], float)
    if len(a) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0, "n_days": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    s = a[idx].sum(axis=1)
    bs = s[:, 0] / np.maximum(s[:, 1], 1)
    return {"mean": float(a[:, 0].sum() / a[:, 1].sum()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(np.mean(bs <= 0)), "n": int(a[:, 1].sum()), "n_days": int(len(a))}


def taker_trades(pred: pl.DataFrame, thr: float, h: int, book: pl.DataFrame, orders: pl.DataFrame | None,
                 maker: bool) -> list[dict[str, float]]:
    """Non-overlapping trades of one symbol-day. Entry state = 1 s book row at T + 1 s; exit at the touch at T + h min.
    maker=True: post at the touch at T + 1 s, filled only if a taker order prints strictly through it within h/2."""
    bts = book["ts"].to_numpy()
    bid, ask = book["bid1"].to_numpy(), book["ask1"].to_numpy()
    if orders is not None:
        oms, oside, olast = orders["ms"].to_numpy(), orders["side"].to_numpy(), orders["last_price"].to_numpy()
    out, free = [], -1
    for T, p in zip(pred["ts"].to_numpy(), pred["pp"].to_numpy()):
        if T < free or abs(p) <= thr:
            continue
        side = 1 if p > 0 else -1
        j0 = np.searchsorted(bts, T + 1000, side="left")
        j1 = np.searchsorted(bts, T + h * 60_000, side="left")
        if j1 >= len(bts) or j0 >= len(bts):
            continue
        if not maker:
            entry = ask[j0] if side > 0 else bid[j0]
        else:
            entry = bid[j0] if side > 0 else ask[j0]
            lo_i = np.searchsorted(oms, T + 1000, side="right")
            hi_i = np.searchsorted(oms, T + h * 30_000, side="right")
            s_, lp = oside[lo_i:hi_i], olast[lo_i:hi_i]
            hit = np.any((s_ < 0) & (lp < entry)) if side > 0 else np.any((s_ > 0) & (lp > entry))
            if not hit:
                continue
        exitp = bid[j1] if side > 0 else ask[j1]
        out.append({"T": float(T), "side": side, "gross": float(side * np.log(exitp / entry) * 1e4)})
        free = T + h * 60_000
    return out


def run_exec() -> dict[str, Any]:
    """H-PART6 on HELD: variants a5, a60 (taker), b5, b60 (maker entry), c (AS quote filter)."""
    import pickle

    with (SUP / "preds.pkl").open("rb") as f:
        preds = pickle.load(f)
    res: dict[str, Any] = {}
    costs = {"taker": 11.0, "maker_in_taker_out": 7.5, "pegged": 6.6}
    held_days = sorted(set(preds["y_r5"]["held"]["date"].to_list()))
    trades: dict[str, dict[str, list]] = {v: {} for v in ("a5", "a60", "b5", "b60")}
    thr = {h: float(np.percentile(np.abs(preds[f"y_r{h}"]["fit_insample_pp"]), 95)) for h in (5, 60)}
    as_med = float(np.median(preds["y_as60"]["fit_insample_pp"]))
    c_keep: dict[str, tuple[float, int]] = {}
    c_all: dict[str, tuple[float, int]] = {}
    l1_med, cap = {}, {}
    for d in held_days:
        for sym in SYMBOLS:
            book = load_book(d, sym)
            orders = load_orders(d, sym, book)
            for h in (5, 60):
                pr = preds[f"y_r{h}"]["held"].filter((pl.col("date") == d) & (pl.col("symbol") == sym)).sort("ts")
                trades[f"a{h}"].setdefault(d, []).extend(taker_trades(pr, thr[h], h, book, None, False))
                trades[f"b{h}"].setdefault(d, []).extend(taker_trades(pr, thr[h], h, book, orders, True))
            pa = preds["y_as60"]["held"].filter((pl.col("date") == d) & (pl.col("symbol") == sym))
            keep_min = set((pa.filter(pl.col("pp") < as_med)["minute"].to_numpy() + 1).tolist())   # pred at i -> fills in i + 1
            all_min = set((pa["minute"].to_numpy() + 1).tolist())
            om = minute_of_ms(orders["ms"].to_numpy(), day0_ms(d))
            a60 = orders["as_60s"].to_numpy()
            fin = np.isfinite(a60)
            mk = fin & np.isin(om, list(keep_min))
            ma = fin & np.isin(om, list(all_min))
            pk, pa_ = c_keep.get(d, (0.0, 0)), c_all.get(d, (0.0, 0))
            c_keep[d] = (pk[0] + float(np.sum(-a60[mk] - 2.0)), pk[1] + int(mk.sum()))
            c_all[d] = (pa_[0] + float(np.sum(-a60[ma] - 2.0)), pa_[1] + int(ma.sum()))
            l1 = 0.5 * (book["bid1"].to_numpy() * book["bid1_size"].to_numpy() + book["ask1"].to_numpy() * book["ask1_size"].to_numpy())
            l1_med.setdefault(sym, []).append(float(np.nanmedian(l1)))
            cap.setdefault(sym, []).append(float(orders["notional"].sum()) / 1440.0)
    pv = {}
    for v in ("a5", "a60", "b5", "b60"):
        cost = costs["taker"] if v.startswith("a") else costs["maker_in_taker_out"]
        sums = {d: (sum(t["gross"] - cost for t in ts), len(ts)) for d, ts in trades[v].items()}
        gsum = {d: (sum(t["gross"] for t in ts), len(ts)) for d, ts in trades[v].items()}
        res[v] = {"net": boot_mean_by_day(sums), "gross": boot_mean_by_day(gsum), "cost_bp": cost}
        if v.startswith("b"):
            res[v]["net_pegged"] = boot_mean_by_day({d: (sum(t["gross"] - costs["pegged"] for t in ts), len(ts)) for d, ts in trades[v].items()})
        pv[v] = res[v]["net"]["p"]
    res["c"] = {"net_kept": boot_mean_by_day(c_keep), "net_all": boot_mean_by_day(c_all), "as_median_threshold": as_med}
    pv["c"] = res["c"]["net_kept"]["p"]
    adj = holm([pv[v] for v in pv])
    res["holm"] = dict(zip(pv, adj))
    res["thresholds"] = thr
    res["capacity"] = {s: {"median_L1_usd": float(np.median(l1_med[s])), "taker_usd_per_min": float(np.median(cap[s]))} for s in SYMBOLS}
    sup = any(res[v]["net"]["mean"] > 0 and res["holm"][v] < 0.05 for v in ("a5", "a60", "b5", "b60")) or \
        (res["c"]["net_kept"]["mean"] > 0 and res["holm"]["c"] < 0.05)
    allneg = all(res[v]["net"]["mean"] <= 0 for v in ("a5", "a60", "b5", "b60") if res[v]["net"]["n"]) and res["c"]["net_kept"]["mean"] <= 0
    res["verdict"] = "supports" if sup else ("contradicts" if allneg else "inconclusive")
    (SUP / "exec.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def describe_types() -> dict[str, Any]:
    """Descriptive table per type (FIT and HELD): shares, traits in natural units, timing, markouts."""
    tr = pl.read_parquet(TAX / "trait_sample.parquet")
    st = pl.read_parquet(TAX / "type_day_stats.parquet")
    k = int(json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))["k"])
    out: dict[str, Any] = {}
    for split, filt in (("FIT", pl.col("date") < HELD_FIRST), ("HELD", pl.col("date") >= HELD_FIRST)):
        s = st.filter(filt)
        tot_n, tot_v = float(s["n"].sum()), float(s["notional"].sum())
        t_ = tr.filter(filt)
        # notional relative to the symbol-day median order notional (scale-free across symbols)
        t_ = t_.with_columns((pl.col("notional") / pl.col("notional").median().over(["date", "symbol"])).alias("rel_notional"))
        rows = {}
        for t in range(-1, k):
            g, gt = s.filter(pl.col("type") == t), t_.filter(pl.col("type") == t)
            if g.height == 0:
                continue
            hours = np.array([float(g[f"h{h:02d}"].sum()) for h in range(24)])
            row = {"share_orders": float(g["n"].sum()) / tot_n, "share_notional": float(g["notional"].sum()) / tot_v,
                   "buy_share": float((g["buy_share"] * g["n"]).sum() / g["n"].sum()),
                   "mm60_bp": float(g["sum_mm_60s"].sum() / max(g["cnt_mm_60s"].sum(), 1)),
                   "mm10_bp": float(g["sum_mm_10s"].sum() / max(g["cnt_mm_60s"].sum(), 1)),
                   "as60_bp": float(g["sum_as_60s"].sum() / max(g["cnt_as_60s"].sum(), 1)),
                   "hour_share_peak": int(np.argmax(hours)), "share_08_16utc": float(hours[8:16].sum() / max(hours.sum(), 1)),
                   "share_by_symbol": {sym: float(g.filter(pl.col("symbol") == sym)["n"].sum() / max(s.filter(pl.col("symbol") == sym)["n"].sum(), 1)) for sym in SYMBOLS}}
            if gt.height:
                row.update({
                    "median_usd": {sym: float(gt.filter(pl.col("symbol") == sym)["notional"].median() or np.nan) for sym in SYMBOLS},
                    "median_rel_notional": float(gt["rel_notional"].median()),
                    "p90_rel_notional": float(gt["rel_notional"].quantile(0.9)),
                    "median_ticks": float(np.expm1(gt["ticks"].median())), "share_multi_level": float((gt["ticks"] > 0).mean()),
                    "median_prints": float(gt["n_prints"].median()), "median_dt_any_ms": float(np.expm1(gt["dt_any"].median())),
                    "median_dt_same_ms": float(np.expm1(gt["dt_same"].median())),
                    "median_n1s": float(np.expm1(gt["n1s"].median())), "share_clipped": float((gt["clip20"] > 0).mean()),
                    "median_sigdig": float(gt["sigdig"].median()), "median_rel_l1": float(np.exp(gt["rel_l1"].median())),
                    "median_sweep_bp": float(gt["sweep_bp"].median()), "n_sample": gt.height})
            rows[str(t)] = row
        out[split] = rows
    (TAX / "type_description.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def run_export() -> dict[str, Any]:
    """Reusable point-in-time participant-state feature set: data/research/participants/features_1m/<date>.parquet,
    one row per (symbol, minute), `known_ts` = minute end (epoch ms UTC). No forward-looking columns."""
    tx = json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))
    hm = json.loads((HMM_DIR / "hmm.json").read_text(encoding="utf-8"))
    k, K = tx["k"], hm["K"]
    keep = (["known_ts", "symbol", "minute", "split", "segment", "mid", "sigma", "buy", "sell", "n_orders", "ofi", "l1", "spread",
             "n_updates", "add", "rem", "refill", "rv", "ofi1", "ofi5", "imb1", "imb5", "rl1", "rl5", "rl60", "lrv5", "lrv60", "lnot5",
             "lspread", "as10_l", "as60_l"]
            + [f"{p}{t}" for t in range(k) for p in ("b", "s", "n")]
            + [f"{p}{t}_{w}" for t in range(k) for p, w in (("sg", 1), ("sg", 5), ("sh", 5), ("cs", 5))]
            + list(HMM_FEATURES) + [f"hp{s_}" for s_ in range(K)] + ["hstate", "hchg"])
    odir = OUT / "features_1m"
    odir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((OUT / "state_1m").glob("*.parquet")):
        g = pl.read_parquet(f).rename({"ts": "known_ts"})
        assert not [c for c in keep if c.startswith("y_")]
        g.select(keep).write_parquet(odir / f.name, compression="zstd")
        n += 1
    return {"days": n, "columns": keep}


def run_liq_check(tol_ms: int = 100) -> dict[str, Any]:
    """Descriptive (not a hypothesis): type mix of taker orders matched to lake liquidation events (same symbol, same
    taker side: long liquidated -> sell, |dt| <= tol_ms), 2026-11-08 .. 2026-11-24 (lake coverage)."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET threads TO 1")
    glob = (ROOT / "data" / "lake" / "liquidations" / "venue=bybit" / "**" / "*.parquet").as_posix()
    liq = con.sql(f"SELECT symbol, epoch_ms(event_ts) AS ms, side FROM read_parquet('{glob}', hive_partitioning=true) "
                  f"WHERE symbol IN {tuple(SYMBOLS)}").pl()
    k = int(json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))["k"])
    base, hit, n_liq, n_match = np.zeros(k), np.zeros(k), 0, 0
    for d in [x for x in post_days() if "2026-11-08" <= x <= "2026-11-24"]:
        d0 = day0_ms(d)
        for sym in SYMBOLS:
            ty = pl.read_parquet(OUT / "days" / d / "types" / f"{sym}.parquet")
            side = pl.read_parquet(OUT / "days" / d / "orders" / f"{sym}.parquet")["side"].to_numpy()
            ms, lab = ty["ms"].to_numpy(), ty["type"].to_numpy().astype(int)
            ok = lab >= 0
            base += np.bincount(lab[ok], minlength=k)
            lq = liq.filter((pl.col("symbol") == sym) & (pl.col("ms") >= d0) & (pl.col("ms") < d0 + DAY_MS))
            for lm, ls in zip(lq["ms"].to_numpy(), lq["side"].to_numpy()):
                n_liq += 1
                want = -1 if ls == "long" else 1
                lo_i, hi_i = np.searchsorted(ms, lm - tol_ms), np.searchsorted(ms, lm + tol_ms, side="right")
                cand = [i for i in range(lo_i, hi_i) if side[i] == want and lab[i] >= 0]
                if cand:
                    i = min(cand, key=lambda i: abs(ms[i] - lm))
                    hit[lab[i]] += 1
                    n_match += 1
    res = {"n_liquidations": n_liq, "n_matched": n_match, "tol_ms": tol_ms, "share_matched": (hit / max(hit.sum(), 1)).tolist(),
           "share_base": (base / base.sum()).tolist(), "ratio": (hit / max(hit.sum(), 1) / (base / base.sum())).tolist()}
    (TAX / "liquidation_mix.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


XV_FIT = ("2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19")
XV_TEST = ("2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23", "2026-11-24")
XV_VENUES = ("bn", "ok", "cb")


def xvenue_rows(d: str) -> pl.DataFrame:
    p = ROOT / "data" / "journal" / "xvenue" / f"{d}.jsonl"
    rows = []
    if not p.exists():
        return pl.DataFrame()
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = f"{r.get('sym')}USDT"
            if sym not in SYMBOLS:
                continue
            ts = int(datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp() * 1000)
            row: dict[str, Any] = {"symbol": sym, "xts": ts}
            for v in XV_VENUES:
                ven = (r.get("venues") or {}).get(v) or {}
                for w in ("1m", "5m"):
                    x = ven.get(w) or {}
                    vol = x.get("vol")
                    row[f"x_{v}_{w}"] = float(x["delta"]) / vol if vol else np.nan
            rows.append(row)
    return pl.DataFrame(rows)


def run_xvenue() -> dict[str, Any]:
    """H-PART7 on the xvenue journal days."""
    tx = json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))
    hm = json.loads((HMM_DIR / "hmm.json").read_text(encoding="utf-8"))
    Pc = feature_sets(tx["k"], hm["K"])["ret"]["P"]
    X = Pc + [f"x_{v}_{w}" for v in XV_VENUES for w in ("1m", "5m")]
    parts = []
    for d in XV_FIT + XV_TEST:
        if not (OUT / "state_1m" / f"{d}.parquet").exists():
            continue
        xv = xvenue_rows(d)
        if xv.height == 0:
            continue
        st = pl.read_parquet(OUT / "state_1m" / f"{d}.parquet")
        d0 = day0_ms(d)
        for sym in SYMBOLS:
            xs = xv.filter(pl.col("symbol") == sym).sort("xts")
            if xs.height == 0:
                continue
            book = load_book(d, sym)
            bts, mid = book["ts"].to_numpy(), book["mid"].to_numpy().astype(float)
            t = xs["xts"].to_numpy()
            j0 = np.searchsorted(bts, t, side="right") - 1
            j1 = np.searchsorted(bts, t + 300_000, side="right") - 1
            ok = (j0 >= 0) & (t + 300_000 <= d0 + DAY_MS)
            minute = (t - d0) // 60_000 - 1
            xs = xs.with_columns(pl.Series("minute", minute), pl.Series("m0", np.where(ok, mid[np.clip(j0, 0, None)], np.nan)),
                                 pl.Series("m1", np.where(ok, mid[np.clip(j1, 0, None)], np.nan)))
            g = st.filter(pl.col("symbol") == sym).drop(["symbol", "date"])
            j = xs.join(g, on="minute", how="inner").with_columns(
                ((pl.col("m1") / pl.col("m0")).log() / (pl.col("sigma") * np.sqrt(5.0))).clip(-10, 10).alias("y_x5"),
                pl.lit(sym).alias("symbol"), pl.lit(d).alias("date"))
            parts.append(j)
    if not parts:
        return {"verdict": "not run", "reason": "no xvenue days"}
    df = pl.concat(parts, how="diagonal_relaxed")
    cols = list(dict.fromkeys(X + ["y_x5"]))
    df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in cols]))
    fit, te = df.filter(pl.col("date").is_in(XV_FIT)), df.filter(pl.col("date").is_in(XV_TEST))
    Xp, lo, hi = design(fit, Pc)
    Xx, lo2, hi2 = design(fit, X)
    y_f, y_t = fit["y_x5"].to_numpy(), te["y_x5"].to_numpy()
    pp = design(te, Pc, lo, hi)[0] @ ols_fit(Xp, y_f)
    px = design(te, X, lo2, hi2)[0] @ ols_fit(Xx, y_f)
    r = day_boot(per_day_sse(te, y_t, pp, px, 0.0))
    res = {"n_fit": fit.height, "n_test": te.height, "fit_days": sorted(set(fit["date"].to_list())),
           "test_days": sorted(set(te["date"].to_list())), "dr2_X_vs_P": r,
           "verdict": "supports" if r["dr2"] > 0 and r["lo"] > 0 else ("contradicts" if r["dr2"] <= 0 else "inconclusive")}
    (SUP / "xvenue.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def run_tape() -> dict[str, Any]:
    """H-PART8: over-representation of the high-ticks type near SWEEP alerts and of the high-clip type in ALGO windows."""
    tr = pl.read_parquet(TAX / "trait_sample.parquet").filter(pl.col("date") < HELD_FIRST)
    med = tr.group_by("type").agg(pl.col("ticks").median(), pl.col("clip20").median(), pl.len()).sort("type")
    t_sweep = int(med.sort(["ticks", "len"], descending=True)["type"][0])
    t_clip = int(med.sort(["clip20", "len"], descending=True)["type"][0])
    tot = {"base": np.zeros(64), "sweep": np.zeros(64), "algo": np.zeros(64)}
    days = [d for d in post_days() if "2026-11-15" <= d <= "2026-11-24"]
    n_alerts = {"SWEEP": 0, "ALGO": 0}
    for d in days:
        p = ROOT / "data" / "journal" / "tape" / f"{d}.jsonl"
        if not p.exists():
            continue
        sweeps: dict[str, list[int]] = {s: [] for s in SYMBOLS}
        algos: dict[str, list[tuple[int, int, int]]] = {s: [] for s in SYMBOLS}
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sym, kind = r.get("symbol"), r.get("kind")
                if sym not in sweeps:
                    continue
                if kind == "SWEEP":
                    sweeps[sym].append(int(r["data"]["sweep"]["ms"]))
                elif kind in ("ALGO_BUY", "ALGO_SELL"):
                    a = r["data"]["algo"]
                    algos[sym].append((int(a["first_ms"]), int(a["last_ms"]), 1 if a["buy"] else -1))
        for sym in SYMBOLS:
            tp = OUT / "days" / d / "types" / f"{sym}.parquet"
            if not tp.exists():
                continue
            ty = pl.read_parquet(tp)
            ms, lab = ty["ms"].to_numpy(), ty["type"].to_numpy().astype(np.int64)
            side = pl.read_parquet(OUT / "days" / d / "orders" / f"{sym}.parquet")["side"].to_numpy()
            ok = lab >= 0
            tot["base"] += np.bincount(lab[ok], minlength=64)
            sw = np.zeros(len(ms), bool)
            for s_ms in sweeps[sym]:
                lo_i, hi_i = np.searchsorted(ms, s_ms - 1000), np.searchsorted(ms, s_ms + 1000, side="right")
                sw[lo_i:hi_i] = True
            al = np.zeros(len(ms), bool)
            for a0, a1, sd in algos[sym]:
                lo_i, hi_i = np.searchsorted(ms, a0), np.searchsorted(ms, a1, side="right")
                al[lo_i:hi_i] |= side[lo_i:hi_i] == sd
            n_alerts["SWEEP"] += len(sweeps[sym])
            n_alerts["ALGO"] += len(algos[sym])
            tot["sweep"] += np.bincount(lab[ok & sw], minlength=64)
            tot["algo"] += np.bincount(lab[ok & al], minlength=64)
    sh = {k_: v / max(v.sum(), 1) for k_, v in tot.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        r_sweep = float(sh["sweep"][t_sweep] / sh["base"][t_sweep])
        r_algo = float(sh["algo"][t_clip] / sh["base"][t_clip])
    k = int(json.loads((TAX / "taxonomy.json").read_text(encoding="utf-8"))["k"])
    res = {"type_high_ticks": t_sweep, "type_high_clip": t_clip, "ratio_sweep": r_sweep, "ratio_algo": r_algo,
           "n_alerts": n_alerts, "n_orders": {k_: int(v.sum()) for k_, v in tot.items()},
           "share_table": {k_: v[:k].tolist() for k_, v in sh.items()},
           "verdict": "supports" if (r_sweep >= 2 and r_algo >= 2) else ("contradicts" if (r_sweep < 1.2 and r_algo < 1.2) else "inconclusive")}
    (TAX / "tape_sanity.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def _cli(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="participants")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce")
    r.add_argument("--days", nargs="*")
    r.add_argument("--workers", type=int, default=2)
    for name in ("taxonomy", "assign", "describe", "hmm", "states", "supervised", "informedness", "exec", "xvenue", "tape", "export", "liq"):
        sub.add_parser(name)
    a = ap.parse_args(argv)
    fns = {"taxonomy": run_taxonomy, "assign": run_assign, "describe": describe_types, "hmm": run_hmm, "states": run_state_features,
           "supervised": run_supervised, "informedness": run_informedness, "exec": run_exec, "xvenue": run_xvenue, "tape": run_tape, "export": run_export, "liq": run_liq_check}
    if a.cmd == "reduce":
        run_reduce(a.days, a.workers)
    else:
        r = fns[a.cmd]()
        if isinstance(r, dict):
            print(json.dumps({k: v for k, v in r.items() if not str(k).startswith("_")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    _cli()
