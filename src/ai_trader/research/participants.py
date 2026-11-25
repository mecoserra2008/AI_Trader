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
