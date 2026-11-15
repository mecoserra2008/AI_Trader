"""Streaming reconstruction of Bybit `orderbook.200` archives into a compact 100 ms book series (agent C2, H-MS5).

Archive format (data/raw/pilot_2026-11-10/ob200/<date>_<SYMBOL>_ob200.data.zip): one JSON message per line,
    {"topic", "type": "snapshot" | "delta", "ts": ms (publication time), "data": {"s", "b": [[price, size], ...],
     "a": [[price, size], ...], "u", "seq"}, "cts": ms (matching-engine time)}
A snapshot replaces the whole book; in a delta a level with size "0" is removed and any other size replaces the level.
Deltas before the first snapshot are ignored. Files are read line by line from the zip (never loaded whole).

The 100 ms series holds the book state at the END of each 100 ms bin of the publication time `ts`: row `ts` = bin end,
so the state is known at `ts` (point in time). Bins without a message carry the previous state forward with
n_updates = 0. Depth is quoted notional (price x size, USDT) within x bp of the mid on each side; imbalance at x bp is
(bid depth - ask depth) / (bid depth + ask depth). span_bid_bp / span_ask_bp = distance of the deepest level held from
the mid: a depth band wider than the span is truncated by the 200-level feed (BTC: 200 ticks of 0.1 = ~2.7 bp).

    from ai_trader.sim.book import build_series, validate_trades
    df = build_series("data/raw/.../2026-11-10_BTCUSDT_ob200.data.zip")      # polars frame, one row per 100 ms
    stats = validate_trades(df, trades)                                         # trades: event_ts, side, price, size

`python -m ai_trader.sim.book [SYMBOL ...]` writes data/raw/pilot_2026-11-10/book_100ms/<SYMBOL>.parquet (<= 4 workers).
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

BIN_MS = 100
BANDS_BP = (5, 10, 25)
IMB_BP = (5, 10)
PILOT = Path("data/raw/pilot_2026-11-10")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "HYPEUSDT", "CARVUSDT", "KASUSDT", "ATOMUSDT", "NVDAUSDT",
           "QQQUSDT", "XAUUSDT")
COLUMNS = (["ts", "cts", "bid1", "ask1", "bid1_size", "ask1_size", "mid", "rel_spread_bp"]
           + [f"depth_{side}_{b}" for b in BANDS_BP for side in ("bid", "ask")] + [f"imb_{b}" for b in IMB_BP] + ["span_bid_bp", "span_ask_bp", "n_updates"])


# ------------------------------------------------------------------ book


class Book:
    """L2 book as two dicts price -> size (floats). `apply` handles snapshot reset, level replacement and removal."""

    __slots__ = ("bids", "asks", "ready", "n_snapshots", "n_crossed")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False
        self.n_snapshots = 0
        self.n_crossed = 0

    def apply(self, kind: str, bids: Sequence, asks: Sequence) -> bool:
        """Apply one message; returns False for a delta that arrives before any snapshot (ignored)."""
        if kind == "snapshot":
            self.bids.clear()
            self.asks.clear()
            self.ready = True
            self.n_snapshots += 1
        elif not self.ready:
            return False
        for side, levels in ((self.bids, bids), (self.asks, asks)):
            for p, s in levels:
                px, sz = float(p), float(s)
                if sz == 0.0:
                    side.pop(px, None)
                else:
                    side[px] = sz
        return True

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        b, a = self.bids, self.asks
        return (np.fromiter(b.keys(), float, len(b)), np.fromiter(b.values(), float, len(b)),
                np.fromiter(a.keys(), float, len(a)), np.fromiter(a.values(), float, len(a)))

    def metrics(self) -> tuple[float, ...]:
        return book_metrics(*self.arrays())


def book_metrics(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray,
                 bands: Sequence[int] = BANDS_BP, imb: Sequence[int] = IMB_BP) -> tuple[float, ...]:
    """(bid1, ask1, bid1_size, ask1_size, mid, rel_spread_bp, depth_bid_b, depth_ask_b for b in bands, imb_b for b in imb,
    span_bid_bp, span_ask_bp). Depth = sum price x size of levels within b bp of the mid (inclusive). span = distance of the
    deepest level from the mid in bp: a 200-level book cannot show depth beyond it (BTC: ~2.7 bp), so a depth band wider
    than the span is truncated. NaN where a side is empty."""
    nb = 8 + 2 * len(bands) + len(imb)
    if len(bid_px) == 0 or len(ask_px) == 0:
        return (np.nan,) * nb
    ib, ia = int(np.argmax(bid_px)), int(np.argmin(ask_px))
    b1, a1 = float(bid_px[ib]), float(ask_px[ia])
    mid = 0.5 * (b1 + a1)
    out = [b1, a1, float(bid_sz[ib]), float(ask_sz[ia]), mid, (a1 - b1) / mid * 1e4]
    bn, an = bid_px * bid_sz, ask_px * ask_sz
    dep = {}
    for b in bands:
        lo, hi = mid * (1.0 - b * 1e-4), mid * (1.0 + b * 1e-4)
        db, da = float(bn[bid_px >= lo].sum()), float(an[ask_px <= hi].sum())
        dep[b] = (db, da)
        out += [db, da]
    for b in imb:
        if b in dep:
            db, da = dep[b]
        else:
            lo, hi = mid * (1.0 - b * 1e-4), mid * (1.0 + b * 1e-4)
            db, da = float(bn[bid_px >= lo].sum()), float(an[ask_px <= hi].sum())
        out.append((db - da) / (db + da) if db + da > 0 else np.nan)
    out += [(mid - float(bid_px.min())) / mid * 1e4, (float(ask_px.max()) - mid) / mid * 1e4]
    return tuple(out)


# ------------------------------------------------------------------ streaming


def iter_messages(path: str | Path) -> Iterator[dict]:
    """Yield parsed messages from an ob200 .zip (first member) or a plain JSON-lines file, one line at a time."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def build_from_messages(messages: Iterator[dict], bin_ms: int = BIN_MS) -> tuple[pl.DataFrame, dict]:
    """Rebuild the book from a message stream and return (100 ms series on a complete grid, QA counters)."""
    book = Book()
    rows: list[tuple] = []
    cur_bin, n_in_bin, last_cts, last_u = None, 0, None, None
    qa = {"messages": 0, "ignored_before_snapshot": 0, "u_gaps": 0, "crossed_bins": 0, "empty_side_bins": 0}

    def flush() -> None:
        m = book.metrics()
        if not np.isfinite(m[0]):
            qa["empty_side_bins"] += 1
        elif m[1] <= m[0]:
            qa["crossed_bins"] += 1
        rows.append(((cur_bin + 1) * bin_ms, last_cts, *m, n_in_bin))

    for msg in messages:
        qa["messages"] += 1
        ts = int(msg["ts"])
        d = msg["data"]
        b = ts // bin_ms
        if cur_bin is not None and b != cur_bin and book.ready and n_in_bin:
            flush()
            n_in_bin = 0
        if cur_bin is None or b != cur_bin:
            cur_bin = b
        if not book.apply(msg["type"], d.get("b", ()), d.get("a", ())):
            qa["ignored_before_snapshot"] += 1
            continue
        u = d.get("u")
        if msg["type"] == "delta" and last_u is not None and u is not None and u != last_u + 1:
            qa["u_gaps"] += 1
        last_u = u
        last_cts = int(msg.get("cts", ts))
        n_in_bin += 1
    if cur_bin is not None and book.ready and n_in_bin:
        flush()
    qa["snapshots"] = book.n_snapshots
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in COLUMNS}), qa
    df = pl.DataFrame(rows, schema=COLUMNS, orient="row").with_columns(pl.col("ts").cast(pl.Int64), pl.col("cts").cast(pl.Int64),
                                                                        pl.col("n_updates").cast(pl.Int32))
    grid = pl.DataFrame({"ts": np.arange(int(df["ts"][0]), int(df["ts"][-1]) + bin_ms, bin_ms, dtype=np.int64)})
    full = grid.join(df, on="ts", how="left")
    full = full.with_columns(pl.col("n_updates").fill_null(0)).with_columns(
        [pl.col(c).forward_fill() for c in COLUMNS if c not in ("ts", "n_updates")])
    qa["bins"] = full.height
    qa["bins_with_update"] = df.height
    return full.with_columns(pl.col("ts").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")), qa


def build_series(path: str | Path, bin_ms: int = BIN_MS) -> tuple[pl.DataFrame, dict]:
    return build_from_messages(iter_messages(path), bin_ms)


# ------------------------------------------------------------------ validation against trades


def validate_trades(book: pl.DataFrame, trades: pl.DataFrame, tol_rel: float = 1e-9) -> dict:
    """Share of trades whose price lies outside the book's [bid1, ask1].

    strict: the book state with the latest matching-engine time cts <= trade time (state before the trade);
    loose:  inside [min(bid1 before, bid1 after), max(ask1 before, ask1 after)], `after` = first state with cts > trade
            time (covers the up-to-100 ms publication lag);
    sweeps: trades in a same-ms, same-side group with more than one distinct price (aggressive orders walking levels).
    trades: event_ts (datetime ms UTC), side ('Buy' | 'Sell'), price."""
    bk = (book.select("cts", "bid1", "ask1").drop_nulls().unique("cts", keep="last").sort("cts")
          .with_columns(pl.col("cts").cast(pl.Int64)))
    tr = (trades.select("event_ts", "side", "price").with_columns(pl.col("event_ts").dt.epoch("ms").cast(pl.Int64).alias("t"))
          .sort("t"))
    tr = tr.with_columns(pl.col("price").n_unique().over(["t", "side"]).alias("_np"))
    before = tr.join_asof(bk.rename({"cts": "t_b"}), left_on="t", right_on="t_b", strategy="backward")
    after = tr.select("t").join_asof(bk.rename({"cts": "t_a", "bid1": "bid1_a", "ask1": "ask1_a"}).with_columns(pl.col("t_a") - 1),
                                     left_on="t", right_on="t_a", strategy="forward")
    j = before.with_columns(after["bid1_a"], after["ask1_a"]).drop_nulls(["bid1"])
    p = j["price"].to_numpy()
    b1, a1 = j["bid1"].to_numpy(), j["ask1"].to_numpy()
    tol = np.abs(p) * tol_rel
    strict = (p < b1 - tol) | (p > a1 + tol)
    ba = j["bid1_a"].fill_null(j["bid1"]).to_numpy()
    aa = j["ask1_a"].fill_null(j["ask1"]).to_numpy()
    loose = (p < np.minimum(b1, ba) - tol) | (p > np.maximum(a1, aa) + tol)
    sweep = j["_np"].to_numpy() > 1
    buy = j["side"].to_numpy() == "Buy"
    n = len(p)
    if n == 0:
        return {"n_trades": 0}
    wrong_side = (buy & (p < b1 - tol)) | (~buy & (p > a1 + tol))
    return {
        "n_trades": int(n),
        "strict_violation_rate": float(strict.mean()),
        "loose_violation_rate": float(loose.mean()),
        "sweep_share": float(sweep.mean()),
        "strict_violation_rate_ex_sweeps": float(strict[~sweep].mean()) if (~sweep).any() else float("nan"),
        "loose_violation_rate_ex_sweeps": float(loose[~sweep].mean()) if (~sweep).any() else float("nan"),
        "violations_in_sweeps_share": float(sweep[loose].mean()) if loose.any() else float("nan"),
        "wrong_side_rate": float(wrong_side.mean()),
        "median_lag_ms_to_book": float(np.median(j["t"].to_numpy() - j["t_b"].to_numpy())),
    }


# ------------------------------------------------------------------ CLI


def _one(symbol: str) -> tuple[str, dict]:
    src = PILOT / "ob200" / f"2026-11-10_{symbol}_ob200.data.zip"
    out = PILOT / "book_100ms" / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df, qa = build_series(src)
    df.with_columns(pl.lit(symbol).alias("symbol")).write_parquet(out, compression="zstd")
    return symbol, qa


def main(argv: Sequence[str] | None = None) -> None:
    syms = list(argv if argv is not None else sys.argv[1:]) or list(SYMBOLS)
    qa_all = {}
    with ProcessPoolExecutor(max_workers=min(4, len(syms))) as ex:
        for sym, qa in ex.map(_one, syms):
            qa_all[sym] = qa
            print(sym, qa, flush=True)
    p = PILOT / "book_100ms" / "_qa.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(qa_all)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
