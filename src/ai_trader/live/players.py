"""Regime and players: what the market's participants are doing, point in time (information only; no forecast uses it).

For BTC, ETH and SOL at as_of (every read is Lake.read_asof(as_of), so nothing known after as_of can enter):
  regime        the v0 cell of research.regimes.label on the 1h market frame (the same call as agent.session.analyse)
  funding       the last Bybit funding settlement and its percentile in the trailing 90 days of settlements
  OI            open interest (4h snapshots, the only interval the live updater keeps) change over 24h and 72h
  liquidations  notional (price x size, USD) of liquidated longs / shorts over the last 4h and 24h (allLiquidation stream,
                collected since 2026-11-08 00:43 UTC; windows reaching before that are marked partial)
  DVOL          Deribit DVOL (BTC, ETH; no SOL index) and its percentile in the trailing 90 days of hourly closes
  ls_ratio      shown only if the lake has a row known within 12 h of as_of (the live updater has no ls_ratio stream)
  pattern       one line of plain words from fixed rules (PATTERN_RULES), descriptive only.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DVOL_ENTITY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
LIQ_START = datetime(2026, 11, 8, 0, 43, tzinfo=UTC)
LS_FRESH = timedelta(hours=12)
PATTERN_RULES = {
    "liq_dominance": 2.0,      # one side's 24h (or 4h) liquidations >= 2x the other's
    "oi_flat_pct": 1.0,        # |OI change| below 1% = flat
    "oi_rebuild_pct": 2.0,     # OI fell >= 2% below its level 72h ago, then rose >= 2% off that low (and up over 24h)
    "funding_rich_pct": 0.90,  # funding percentile >= 0.90 = rich
    "dvol_high_pct": 0.90, "dvol_low_pct": 0.10,
}


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


def _pct(x: np.ndarray, v: float) -> float | None:
    x = x[np.isfinite(x)]
    if not len(x) or not math.isfinite(v):
        return None
    return float((np.sum(x < v) + 0.5 * np.sum(x == v)) / len(x))


def regime(sym: str, as_of: datetime, lake) -> dict:
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import CELL_LABELS, label

    f = market_frame(sym, "1h", as_of, funding=False, oi=False, lake=lake)
    if f.height < 50:
        return {"status": "no bars"}
    f = label(f, 1.0)
    last = f.tail(1).row(0, named=True)
    out = {"status": "ok", "bar": _utc(last["event_ts"]).isoformat(), "cell": last["cell"], "label": CELL_LABELS.get(last["cell"] or "", ""),
           "dir_state": last["dir_state"], "vol_state": last["vol_state"], "close": float(last["close"])}
    cl = dict(zip([_utc(t) for t in f["close_ts"].to_list()], f["close"].to_list(), strict=True))
    tc = _utc(last["close_ts"])
    for k in (24, 72):
        p = cl.get(tc - timedelta(hours=k))
        out[f"ret_{k}h_pct"] = 100 * math.log(last["close"] / p) if p else None
    return out


def funding(sym: str, as_of: datetime, lake) -> dict:
    d = lake.read_asof("funding", as_of, symbols=[sym], start=as_of - timedelta(days=90), filters={"venue": "bybit"})
    if d.height == 0:
        return {"status": "none"}
    d = d.sort("event_ts")
    r = d["rate"].cast(pl.Float64).to_numpy()
    ev = d["event_ts"].to_list()
    iv = (ev[-1] - ev[-2]).total_seconds() / 3600 if len(ev) > 1 else None
    return {"status": "ok", "ts": _utc(ev[-1]).isoformat(), "rate_pct": 100 * float(r[-1]), "pct_90d": _pct(r, r[-1]), "n_90d": len(r), "interval_h": iv}


def open_interest(sym: str, as_of: datetime, lake) -> dict:
    d = lake.read_asof("open_interest", as_of, symbols=[sym], start=as_of - timedelta(days=5), filters={"venue": "bybit", "interval": "4h"})
    if d.height == 0:
        return {"status": "none"}
    d = d.sort("event_ts")
    t = [_utc(x) for x in d["event_ts"].to_list()]
    v = d["open_interest"].cast(pl.Float64).to_list()
    out = {"status": "ok", "ts": t[-1].isoformat(), "oi": v[-1]}
    for k in (24, 72):
        prev = [x for x, tt in zip(v, t, strict=True) if tt <= t[-1] - timedelta(hours=k)]
        out[f"chg_{k}h_pct"] = 100 * (v[-1] / prev[-1] - 1) if prev and prev[-1] > 0 else None
    w = [x for x, tt in zip(v, t, strict=True) if tt >= t[-1] - timedelta(hours=72)]
    prev72 = [x for x, tt in zip(v, t, strict=True) if tt <= t[-1] - timedelta(hours=72)]
    if w and prev72 and prev72[-1] > 0 and min(w) > 0:
        out["low_72h_pct"] = 100 * (min(w) / prev72[-1] - 1)       # the 72h low vs the level 72h ago
        out["from_low_pct"] = 100 * (v[-1] / min(w) - 1)           # now vs that low
    return out


def liquidations(sym: str, as_of: datetime, lake) -> dict:
    d = lake.read_asof("liquidations", as_of, symbols=[sym], start=as_of - timedelta(hours=24))
    out: dict = {"status": "ok", "partial_24h": as_of - timedelta(hours=24) < LIQ_START, "partial_4h": as_of - timedelta(hours=4) < LIQ_START}
    if d.height:
        d = d.with_columns((pl.col("price") * pl.col("size")).alias("usd"))
    for lab, h in (("4h", 4), ("24h", 24)):
        w = d.filter(pl.col("event_ts") > as_of - timedelta(hours=h)) if d.height else d
        for side in ("long", "short"):
            out[f"{side}_{lab}_usd"] = float(w.filter(pl.col("side") == side)["usd"].sum()) if w.height else 0.0
    return out


def dvol(sym: str, as_of: datetime, lake) -> dict:
    ent = DVOL_ENTITY.get(sym)
    if ent is None:
        return {"status": "no index"}
    d = lake.read_asof("alt_series", as_of, symbols=[ent], start=as_of - timedelta(days=90), filters={"source": "deribit", "series_id": "dvol_close"})
    if d.height == 0:
        return {"status": "none"}
    d = d.sort("event_ts").drop_nulls("value")
    x = d["value"].cast(pl.Float64).to_numpy()
    return {"status": "ok", "ts": _utc(d["event_ts"][-1]).isoformat(), "level": float(x[-1]), "pct_90d": _pct(x, x[-1]), "median_90d": float(np.median(x))}


def ls_ratio(sym: str, as_of: datetime, lake) -> dict:
    best = None
    for period in ("1h", "4h"):
        d = lake.read_asof("ls_ratio", as_of, symbols=[sym], start=as_of - timedelta(days=30), filters={"venue": "bybit", "period": period})
        if d.height:
            row = d.sort("known_ts").tail(1).row(0, named=True)
            if best is None or row["known_ts"] > best["known_ts"]:
                best = {**row, "period": period}
    if best is None:
        return {"status": "none in last 30 days"}
    kt = _utc(best["known_ts"])
    if as_of - kt > LS_FRESH:
        return {"status": "stale", "last_known": kt.isoformat()}
    return {"status": "ok", "ts": _utc(best["event_ts"]).isoformat(), "period": best["period"], "buy_ratio": float(best["buy_ratio"])}


def pattern(p: dict) -> str:
    """One line of plain words from fixed rules (descriptive only)."""
    R = PATTERN_RULES
    words = []
    lq, fu, oi, dv, rg = p["liq"], p["funding"], p["oi"], p["dvol"], p["regime"]
    l24, s24, l4, s4 = lq["long_24h_usd"], lq["short_24h_usd"], lq["long_4h_usd"], lq["short_4h_usd"]
    if l24 > 0 and l24 >= R["liq_dominance"] * s24:
        words.append("long flush")
    elif s24 > 0 and s24 >= R["liq_dominance"] * l24:
        words.append("short squeeze")
    elif l24 + s24 > 0:
        words.append("two-sided liquidations")
    if fu.get("status") == "ok":
        if fu["rate_pct"] < 0:
            words.append("funding negative")
        elif (fu["pct_90d"] or 0) >= R["funding_rich_pct"]:
            words.append("funding rich")
        else:
            words.append("funding normal")
    if oi.get("status") == "ok" and oi.get("chg_24h_pct") is not None:
        c24, c72, r24 = oi["chg_24h_pct"], oi.get("chg_72h_pct"), rg.get("ret_24h_pct")
        lo, up = oi.get("low_72h_pct"), oi.get("from_low_pct")
        if lo is not None and up is not None and lo <= -R["oi_rebuild_pct"] and up >= R["oi_rebuild_pct"] and c24 > 0:
            words.append("OI rebuilding after a drop" + (" while price fell (new shorts)" if r24 is not None and r24 < 0 else ""))
        elif abs(c24) < R["oi_flat_pct"]:
            words.append("OI flat")
        elif c24 > 0 and c72 is not None and c72 < c24:
            words.append("OI rebuilding" + (" while price fell (new shorts)" if r24 is not None and r24 < 0 else ""))
        elif c24 > 0:
            words.append("OI rising" + (" with price (new longs)" if r24 is not None and r24 >= 0 else " into the drop (new shorts)"))
        else:
            words.append("OI falling" + (" with price (deleveraging)" if r24 is not None and r24 < 0 else " into the rise (short covering)"))
    if s4 > 0 and s4 >= R["liq_dominance"] * l4:
        words.append("short-liquidation burst (4h)")
    elif l4 > 0 and l4 >= R["liq_dominance"] * s4:
        words.append("long-liquidation burst (4h)")
    if dv.get("status") == "ok" and dv["pct_90d"] is not None:
        if dv["pct_90d"] >= R["dvol_high_pct"]:
            words.append("implied vol high")
        elif dv["pct_90d"] <= R["dvol_low_pct"]:
            words.append("implied vol low")
    return ", ".join(words) or "no clear pattern"


def players(as_of: datetime, lake=None, symbols=SYMBOLS) -> dict:
    from ai_trader.storage.lake import Lake

    lake = lake or Lake()
    as_of = _utc(as_of)
    out = {"as_of": as_of.isoformat(), "symbols": {}}
    for s in symbols:
        p = {}
        for k, fn in (("regime", regime), ("funding", funding), ("oi", open_interest), ("liq", liquidations), ("dvol", dvol), ("ls", ls_ratio)):
            try:
                p[k] = fn(s, as_of, lake)
            except Exception as e:  # noqa: BLE001 - information only, never stops the run
                p[k] = {"status": f"error: {type(e).__name__}: {e}"[:160]}
        if "long_24h_usd" not in p["liq"]:
            p["liq"] = {**p["liq"], "long_24h_usd": 0.0, "short_24h_usd": 0.0, "long_4h_usd": 0.0, "short_4h_usd": 0.0}
        p["pattern"] = pattern(p)
        out["symbols"][s] = p
    return out


def _m(x: float) -> str:
    return f"{x / 1e6:.1f}M" if x >= 1e5 else f"{x / 1e3:.0f}k"


def _p(x) -> str:
    return "n/a" if x is None else f"p{100 * x:.0f}"


def render(P: dict) -> list[str]:
    L = [f"\n## Regime and players (information only, changes no forecast; as_of {P['as_of'][:16]}Z)",
         "| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |",
         "|---|---|---|---|---|---|---|---|"]
    ls_notes = set()
    for s, p in P["symbols"].items():
        rg, fu, oi, lq, dv = p["regime"], p["funding"], p["oi"], p["liq"], p["dvol"]
        cell = f"{rg['cell']} {rg['label']}" if rg.get("status") == "ok" else rg.get("status")
        px = f"{rg['ret_24h_pct']:+.1f}%" if rg.get("ret_24h_pct") is not None else "n/a"
        fs = f"{fu['rate_pct']:+.4f}% ({_p(fu['pct_90d'])})" if fu.get("status") == "ok" else fu.get("status")
        os_ = (f"{oi['chg_24h_pct']:+.1f}% / {oi['chg_72h_pct']:+.1f}%" if oi.get("status") == "ok" and oi.get("chg_72h_pct") is not None
               else oi.get("status", "n/a"))
        part = " (partial)" if lq.get("partial_24h") else ""
        ds = f"{dv['level']:.1f} ({_p(dv['pct_90d'])})" if dv.get("status") == "ok" else dv.get("status")
        L.append(f"| {s[:-4]} | {cell} | {px} | {fs} | {os_} | {_m(lq['long_4h_usd'])} / {_m(lq['short_4h_usd'])} | "
                 f"{_m(lq['long_24h_usd'])} / {_m(lq['short_24h_usd'])}{part} | {ds} |")
        if p["ls"].get("status") == "ok":
            ls_notes.add(f"{s[:-4]} {p['ls']['buy_ratio']:.2f} long ({p['ls']['period']})")
        else:
            ls_notes.add("stale:" + str(p["ls"].get("last_known", p["ls"].get("status")))[:16])
    L.append("- Pattern: " + "; ".join(f"{s[:-4]} {p['pattern']}" for s, p in P["symbols"].items()))
    st = sorted(x for x in ls_notes if x.startswith("stale:"))
    ok = sorted(x for x in ls_notes if not x.startswith("stale:"))
    if ok:
        L.append("- Long/short accounts: " + ", ".join(ok))
    if st:
        L.append(f"- Long/short ratio omitted: lake ls_ratio last known {st[0][6:]} (one-off backfill of 11-07; the live updater has no ls_ratio stream)")
    return L


__all__ = ["PATTERN_RULES", "SYMBOLS", "pattern", "players", "render"]
