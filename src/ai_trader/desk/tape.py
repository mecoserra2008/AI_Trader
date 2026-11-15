"""Tape domain (tape.json): what is printing now for the tokens in play. Descriptive only: R-FLOW-1 (|imbalance z| > 4 is
neither a vol nor a direction signal), R-FLOW-2 (don't fade a liquidation cluster within 60 min), R-FLOW-3, R-SIM-2 / R-SIM-3.

Per token, everything known at as_of:
  price      last / mark / index from the ticker snapshot (known_ts <= as_of)
  moves      100 ln(last / close of the 1m bar that closed at or before as_of - m) for m = 1, 5, 15, 60 minutes, and the
             last-60-minute range
  book       Bybit public REST order book (200 levels): spread 1e4 ln(ask / bid), USD depth within 10 / 25 / 100 bp of the
             mid per side (`span_bp` = how far the received levels reach; a band beyond it is marked truncated), imbalance
             (bid - ask) / (bid + ask) within 25 bp
  liq        lake `liquidations` (allLiquidation stream): USD of liquidated longs / shorts over 15m / 1h / 4h / 24h, the
             largest print, and the R-FLOW-2 cluster test on the last 60 minutes (a minute above the trailing-24h 99th
             percentile of 1m liquidation notional with >= 3 prints)
  flow       lake `flow_3s` (atx collect): taker buy / sell notional and imbalance (b - s) / (b + s) over 5 / 15 / 60 min, the
             share of 3 s candles present (coverage), mean spread and 25 bp depth over the last 5 minutes
  players    the live.players read (funding percentile, OI 24h / 72h, liquidations 4h / 24h, DVOL, pattern words) with
             the v0 cell computed by desk.xs.regime_cells
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.desk.public import closed

MOVE_MIN = (1, 5, 15, 60)
BOOK_BANDS_BP = (10, 25, 100)
FLOW_WIN_MIN = (5, 15, 60)
LIQ_WIN = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1), "4h": timedelta(hours=4), "24h": timedelta(hours=24)}


def _r(x, nd: int = 3):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def moves(k1m: pl.DataFrame | None, last: float | None, as_of: datetime, minutes=MOVE_MIN) -> dict:
    if k1m is None or k1m.height == 0 or not last:
        return {"status": "none"}
    b = closed(k1m, as_of)
    if b.height == 0:
        return {"status": "none"}
    ct, c = b["close_ts"].to_list(), b["close"].to_numpy()
    out: dict = {"status": "ok", "pct": {}}
    for m in minutes:
        cut = as_of - timedelta(minutes=m)
        idx = [i for i, t in enumerate(ct) if t <= cut]
        if idx:
            out["pct"][f"{m}m"] = _r(100 * math.log(last / c[idx[-1]]), 3)
    w = b.filter(pl.col("event_ts") >= as_of - timedelta(minutes=60))
    if w.height:
        out["range_60m_pct"] = _r(100 * math.log(float(w["high"].max()) / float(w["low"].min())), 3)
        out["high_60m"], out["low_60m"] = float(w["high"].max()), float(w["low"].min())
    return out


def book_stats(ob: dict | None, as_of: datetime, bands=BOOK_BANDS_BP) -> dict:
    if not ob or not ob.get("bids") or not ob.get("asks"):
        return {"status": "none"}
    if ob.get("known_ts") and ob["known_ts"] > as_of:
        return {"status": "after as_of"}
    bids, asks = sorted(ob["bids"], key=lambda x: -x[0]), sorted(ob["asks"], key=lambda x: x[0])
    bb, ba = bids[0][0], asks[0][0]
    mid = 0.5 * (bb + ba)
    span = min(1e4 * math.log(mid / bids[-1][0]), 1e4 * math.log(asks[-1][0] / mid))
    out = {"status": "ok", "ts": ob["ts"].isoformat() if ob.get("ts") else None, "bid": bb, "ask": ba, "mid": mid,
           "spread_bp": _r(1e4 * math.log(ba / bb), 2), "l1_usd": {"bid": round(bids[0][0] * bids[0][1]), "ask": round(asks[0][0] * asks[0][1])},
           "span_bp": _r(span, 1), "depth_usd": {}}
    for bp in bands:
        lo, hi = mid * math.exp(-bp / 1e4), mid * math.exp(bp / 1e4)
        bd = sum(p * q for p, q in bids if p >= lo)
        ad = sum(p * q for p, q in asks if p <= hi)
        out["depth_usd"][f"{bp}bp"] = {"bid": round(bd), "ask": round(ad), "truncated": bool(span < bp),
                                       "imbalance": _r((bd - ad) / (bd + ad), 3) if bd + ad > 0 else None}
    return out


def liq_summary(liq: pl.DataFrame | None, as_of: datetime) -> dict:
    """Liquidation prints of one symbol known at as_of (side = side of the liquidated position)."""
    if liq is None or liq.height == 0:
        return {"status": "no prints in 24h", **{k: {"long_usd": 0, "short_usd": 0, "n": 0} for k in LIQ_WIN}}
    d = liq.filter((pl.col("known_ts") <= as_of) & (pl.col("event_ts") > as_of - timedelta(hours=24))).with_columns(
        (pl.col("price") * pl.col("size")).alias("usd"))
    out: dict = {"status": "ok"}
    for k, w in LIQ_WIN.items():
        x = d.filter(pl.col("event_ts") > as_of - w)
        out[k] = {"long_usd": round(float(x.filter(pl.col("side") == "long")["usd"].sum())),
                  "short_usd": round(float(x.filter(pl.col("side") == "short")["usd"].sum())), "n": x.height}
    if d.height:
        big = d.sort("usd").tail(1).row(0, named=True)
        out["largest_24h"] = {"usd": round(big["usd"]), "side": big["side"], "price": big["price"], "ts": big["event_ts"].isoformat()}
        m = d.with_columns(pl.col("event_ts").dt.truncate("1m").alias("m")).group_by("m").agg(pl.col("usd").sum(), pl.len().alias("n"))
        start = (as_of - timedelta(hours=24)).replace(second=0, microsecond=0)
        vals = np.zeros(1440)
        mins = {r["m"]: (r["usd"], r["n"]) for r in m.iter_rows(named=True)}
        for t, (u, _n) in mins.items():
            i = int((t - start).total_seconds() // 60)
            if 0 <= i < 1440:
                vals[i] += u
        thr = float(np.quantile(vals, 0.99))
        hits = [(t, u, n) for t, (u, n) in mins.items() if t >= as_of - timedelta(minutes=60) and u > thr and n >= 3 and thr > 0]
        out["cluster_60m"] = {"fired": bool(hits), "threshold_usd": round(thr), "minutes": [{"ts": t.isoformat(), "usd": round(u), "n": n} for t, u, n in sorted(hits)][:5],
                              "rule": "R-FLOW-2: do not fade a liquidation cluster expecting a reversal within 60 min"}
    return out


def flow_summary(f3: pl.DataFrame | None, as_of: datetime, windows=FLOW_WIN_MIN) -> dict:
    if f3 is None or f3.height == 0:
        return {"status": "not collected"}
    d = f3.filter((pl.col("known_ts") <= as_of) & (pl.col("event_ts") + pl.duration(seconds=3) <= as_of))
    if d.height == 0:
        return {"status": "not collected"}
    out: dict = {"status": "ok", "last_candle": d["event_ts"].max().isoformat(), "windows": {}}
    for m in windows:
        x = d.filter(pl.col("event_ts") >= as_of - timedelta(minutes=m))
        b, s = float(x["buy_notional"].sum()), float(x["sell_notional"].sum())
        out["windows"][f"{m}m"] = {"buy_usd": round(b), "sell_usd": round(s), "imbalance": _r((b - s) / (b + s), 3) if b + s > 0 else None,
                                   "prints": int(x["n_buy"].sum() + x["n_sell"].sum()), "coverage": _r(x.height / (m * 20), 2)}
    x = d.filter(pl.col("event_ts") >= as_of - timedelta(minutes=5))
    if x.height:
        def mean(c):
            v = x[c].drop_nulls() if c in x.columns else None
            return _r(float(v.mean()), 2) if v is not None and len(v) else None
        out["book_5m"] = {"spread_bps": mean("spread_bps"), "depth_bid_25_usd": mean("depth_bid_25"), "depth_ask_25_usd": mean("depth_ask_25"),
                          "obi_5": mean("obi_5")}
    out["rule"] = "R-FLOW-1: imbalance extremes are not a vol or direction signal"
    return out


def player_read(sym: str, as_of: datetime, lake, regime: dict | None) -> dict:
    from ai_trader.live import players as P

    p: dict = {"regime": regime or {"status": "none"}}
    for k, fn in (("funding", P.funding), ("oi", P.open_interest), ("liq", P.liquidations), ("dvol", P.dvol)):
        try:
            p[k] = fn(sym, as_of, lake)
        except Exception as e:  # noqa: BLE001
            p[k] = {"status": f"error: {type(e).__name__}"}
    if "long_24h_usd" not in p["liq"]:
        p["liq"] = {**p["liq"], "long_24h_usd": 0.0, "short_24h_usd": 0.0, "long_4h_usd": 0.0, "short_4h_usd": 0.0}
    try:
        p["pattern"] = P.pattern(p)
    except Exception as e:  # noqa: BLE001
        p["pattern"] = f"error: {type(e).__name__}"
    return {"pattern": p["pattern"], "funding": p["funding"], "oi": p["oi"], "dvol": p["dvol"],
            "liq_4h_24h_usd": {k: round(v) for k, v in p["liq"].items() if k.endswith("_usd")}}


def build_tape(as_of: datetime, tokens: dict[str, dict], *, reasons: dict[str, list[str]] | None = None) -> dict:
    """tokens: sym -> {"ticker": dict, "ticker_known": dt, "k1m": frame, "book": dict, "liq": frame, "flow": frame, "players": dict}."""
    rows = []
    for s, t in tokens.items():
        tk = t.get("ticker") or {}
        ok = t.get("ticker_known") is None or t["ticker_known"] <= as_of
        last = float(tk["lastPrice"]) if ok and tk.get("lastPrice") else None
        rows.append({
            "symbol": s, "why": (reasons or {}).get(s, []),
            "price": {"last": last, "mark": float(tk["markPrice"]) if ok and tk.get("markPrice") else None,
                      "index": float(tk["indexPrice"]) if ok and tk.get("indexPrice") else None,
                      "chg_24h_pct": _r(100 * math.log(1 + float(tk["price24hPcnt"])), 2) if ok and tk.get("price24hPcnt") else None},
            "moves": moves(t.get("k1m"), last, as_of),
            "book": book_stats(t.get("book"), as_of),
            "liq": liq_summary(t.get("liq"), as_of),
            "flow": flow_summary(t.get("flow"), as_of),
            "players": t.get("players"),
        })
    return {"domain": "tape", "as_of": as_of.isoformat(),
            "note": "descriptive: R-FLOW-1/2/3, R-SIM-2/3; moves are log %, spreads / bands in bp, notional in USD", "tokens": rows}


__all__ = ["book_stats", "build_tape", "flow_summary", "liq_summary", "moves", "player_read"]
