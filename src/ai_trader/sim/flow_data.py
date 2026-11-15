"""Data access for the execution simulator: point-in-time flow_3s per symbol, complete 3 s grids, live mm records.

- `read_symbol` reads ONE symbol of flow_3s as of a fixed timestamp with column projection (the table is large).
- `to_grid` puts it on a complete 3 s grid from the first to the last candle; candles missing from the lake
  (collector gaps) or without a valid book (mid / spread null) are flagged `ok = False` and never zero-filled.
- `load_live` reads our own mainnet market-making records (data/mm/analysis/mainnet-*/), deduplicated across the
  overlapping pulls by orderId / execId. Only quotes whose orderLinkId starts with "mm" are the quoter's.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

STEP_MS = 3000
COLUMNS = ["symbol", "event_ts", "high", "low", "close", "buy_notional", "sell_notional", "buy_vol", "sell_vol", "n_buy", "n_sell",
           "max_trade", "mid", "spread_bps", "obi_5", "obi_25", "depth_bid_25", "depth_ask_25", "book_cov", "known_ts"]
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"}
TRADFI_TYPES = {"ETF", "stock", "commodity", "Stock", "Commodity", "forex", "FX", "index"}
# Fallback for TradFi perps whose instrument row has no symbol_type (same list as research.flow_anomaly_study.TRADFI)
TRADFI_FALLBACK = {
    "SOXLUSDT", "QQQUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT", "MUUSDT", "GOOGLUSDT", "AAPLUSDT", "INTCUSDT", "SPCXUSDT",
    "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "USDJPYUSDT", "EURUSDUSDT", "SPYUSDT", "IWMUSDT", "TLTUSDT", "SMHUSDT",
    "XLEUSDT", "GDXUSDT", "UVXYUSDT", "IBITUSDT", "CRWVUSDT", "IRENUSDT", "CORZUSDT", "COINUSDT", "HOODUSDT", "MARAUSDT",
    "CRCLUSDT", "AMDSTOCKUSDT", "AVGOUSDT", "TSMUSDT", "METAUSDT", "MSFTUSDT", "ORCLUSDT", "PAXGUSDT", "XAUTUSDT",
}


def asset_class(symbol: str, symbol_type: str | None = None) -> str:
    """major | alt | tradfi. Majors: the six largest crypto perps; TradFi: instrument symbol_type or the fallback list."""
    if symbol in MAJORS:
        return "major"
    if (symbol_type or "") in TRADFI_TYPES or symbol in TRADFI_FALLBACK:
        return "tradfi"
    return "alt"


def read_symbol(lake, symbol: str, as_of: datetime, start: datetime | None = None, end: datetime | None = None,
                category: str = "linear") -> pl.DataFrame:
    """flow_3s rows of one symbol knowable at as_of (Lake.read_asof: known_ts <= as_of, latest revision)."""
    return lake.read_asof("flow_3s", as_of, symbols=[symbol], start=start, end=end,
                          filters={"category": category, "venue": "bybit"}, columns=COLUMNS).sort("event_ts")


@dataclass
class Grid:
    """Complete 3 s grid of one symbol. All arrays have the same length; `ok` marks usable candles."""
    symbol: str
    t_ms: np.ndarray        # candle open, epoch ms
    ok: np.ndarray          # row present and book valid (mid and spread known)
    present: np.ndarray     # row present in the lake (trades are known even if the book was invalid)
    mid: np.ndarray         # mid at candle close (NaN where unknown)
    spread_bps: np.ndarray
    high: np.ndarray        # NaN when no trade in the candle
    low: np.ndarray
    buy: np.ndarray         # taker-buy notional (quote ccy)
    sell: np.ndarray
    n_buy: np.ndarray
    n_sell: np.ndarray
    obi_5: np.ndarray
    obi_25: np.ndarray
    depth: np.ndarray       # bid + ask notional within 25 bps

    def __len__(self) -> int:
        return len(self.t_ms)

    def day_index(self) -> np.ndarray:
        """UTC calendar day of each candle as integer days since epoch."""
        return self.t_ms // 86_400_000

    def slice(self, mask: np.ndarray) -> Grid:
        return Grid(self.symbol, *(getattr(self, f)[mask] for f in GRID_FIELDS))

    def segments(self, min_len: int = 20) -> list[tuple[int, int]]:
        """Maximal runs [a, b) of consecutive ok candles (on the 3 s grid) with at least min_len candles."""
        ok = self.ok.astype(np.int8)
        d = np.diff(np.r_[0, ok, 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a >= min_len]


GRID_FIELDS = ("t_ms", "ok", "present", "mid", "spread_bps", "high", "low", "buy", "sell", "n_buy", "n_sell", "obi_5", "obi_25", "depth")


def to_grid(df: pl.DataFrame, symbol: str | None = None) -> Grid:
    """flow_3s rows of one symbol -> complete 3 s grid (gaps explicit)."""
    sym = symbol or (df["symbol"][0] if df.height else "")
    if df.height == 0:
        z = np.zeros(0)
        return Grid(sym, z.astype(np.int64), z.astype(bool), z.astype(bool), *(z for _ in range(11)))
    t = df["event_ts"].dt.epoch("ms").to_numpy().astype(np.int64)
    t0 = int(t.min())
    idx = ((t - t0) // STEP_MS).astype(np.int64)
    n = int(idx.max()) + 1

    def col(name: str, fill: float = np.nan) -> np.ndarray:
        out = np.full(n, fill, dtype=float)
        v = df[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
        out[idx] = v
        return out

    present = np.zeros(n, dtype=bool)
    present[idx] = True
    mid, spr = col("mid"), col("spread_bps")
    depth = col("depth_bid_25") + col("depth_ask_25")
    ok = present & np.isfinite(mid) & (mid > 0) & np.isfinite(spr) & (spr > 0)
    return Grid(sym, t0 + np.arange(n, dtype=np.int64) * STEP_MS, ok, present, mid, spr, col("high"), col("low"),
                col("buy_notional", 0.0), col("sell_notional", 0.0), col("n_buy", 0.0), col("n_sell", 0.0),
                col("obi_5"), col("obi_25"), depth)


def save_grid(g: Grid, path: Path) -> None:
    pl.DataFrame({f: getattr(g, f) for f in GRID_FIELDS}).with_columns(pl.lit(g.symbol).alias("symbol")).write_parquet(path)


def load_grid(path: Path) -> Grid:
    d = pl.read_parquet(path)
    sym = d["symbol"][0] if d.height else Path(path).stem
    return Grid(sym, *(d[f].to_numpy() for f in GRID_FIELDS))


# --------------------------------------------------------------------------------------------- live records

def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_live(root: Path, env: str = "mainnet") -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """(quotes, fills, info) from data/mm/analysis/<env>-*/ (read only).

    quotes: one row per mm post-only quote: symbol, side (+1 buy / -1 sell), price, qty, created_ms, end_ms (fill or
            cancel time = updatedTime), filled (cumExecQty > 0), reduce_only, link, order_id, reject.
    fills:  one row per maker execution of an mm quote: symbol, side, price, qty, exec_ms, order_id, maker."""
    orders, execs = {}, {}
    dirs = sorted(glob.glob(str(root / "data" / "mm" / "analysis" / f"{env}-*")))
    for d in dirs:
        for x in json.loads(Path(d, "orders.json").read_text(encoding="utf-8")):
            orders[x["orderId"]] = x  # later pulls carry the final status
        for x in json.loads(Path(d, "executions.json").read_text(encoding="utf-8")):
            execs[x["execId"]] = x
    q = [{"symbol": x["symbol"], "side": 1 if x["side"] == "Buy" else -1, "price": _f(x.get("price")), "qty": _f(x.get("qty")),
          "cum": _f(x.get("cumExecQty")), "created_ms": int(x["createdTime"]), "end_ms": int(x["updatedTime"]),
          "status": x["orderStatus"], "reduce_only": bool(x.get("reduceOnly")), "link": x.get("orderLinkId") or "",
          "order_id": x["orderId"], "reject": x.get("rejectReason") or "", "tif": x.get("timeInForce") or ""}
         for x in orders.values() if (x.get("orderLinkId") or "").startswith("mm")]
    quotes = pl.DataFrame(q).with_columns((pl.col("cum") > 0).alias("filled")).sort("created_ms")
    e = [{"symbol": x["symbol"], "side": 1 if x["side"] == "Buy" else -1, "price": _f(x["execPrice"]), "qty": _f(x["execQty"]),
          "exec_ms": int(x["execTime"]), "order_id": x["orderId"], "maker": bool(x["isMaker"]), "fee": _f(x.get("execFee")),
          "link": x.get("orderLinkId") or ""}
         for x in execs.values() if x.get("execType") == "Trade" and (x.get("orderLinkId") or "").startswith("mm")]
    fills = pl.DataFrame(e).sort("exec_ms")
    info = {"dirs": [Path(d).name for d in dirs], "orders_all": len(orders), "execs_all": len(execs),
            "mm_quotes": quotes.height, "mm_filled": int(quotes["filled"].sum()), "mm_fills": fills.height}
    return quotes, fills, info
