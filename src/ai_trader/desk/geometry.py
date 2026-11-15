"""Market geometry domain (geometry.json): where the levels are, never what price will do at them (R-PX-4: levels break
slightly more often than they hold; R-PX-10: floor pivots are not support / resistance; R-AVOID-1: pivot-geometry trend
and vector-candle continuation failed).

Per token, from bars fully closed at as_of (the open of the forming bar is the only forming-bar value used):
  swings        fractal pivots (k bars each side, strict on the left) on 15m / 1h / 4h / 1d; a pivot is known at the close
                of bar i + k; `swept` = a later bar traded beyond it (resting stops there have been taken)
  periods       current day / week (ISO, Monday 00:00 UTC) / month open, prior day / week / month O H L C (UTC)
  legs          vecopen LegOriginDetector (frozen vecopen/config/baseline.yaml, read-only import) on 1h / 4h / 1d; a leg's
                level is the origin open with zone +- kappa_atr x ATR; `untouched` = no later bar of the same timeframe
                traded into the zone (a coarser test than vecopen's 5m first fill), expired after a_max_bars tau-bars
  profile       volume profile (features.profile.session_profile: volume spread uniformly over each bar's range, 0.05%
                geometric bins, 70% value area) of the last 24h from 1m bars and the last 7d from 15m bars
  liq_map       ESTIMATED liquidation clusters: each 1h OI increase over 7d opens positions at the bar's typical price,
                split 50 / 50 long / short, at 10x / 25x / 50x (equal weights), maintenance margin 0.5%; positions whose
                liquidation price was already traded through are removed; OI decreases are not attributed (so it
                overstates). A model with stated assumptions, not data.
  round         round numbers around the price (major / minor steps)
  distances     every level: dist_pct = 100 ln(level / price), dist_atr_1h / dist_atr_1d = (level - price) / ATR14 (Wilder)
BTC also gets the field card (touch probabilities) of the latest live line known at as_of (data/journal/live_nn).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.desk.public import closed, forming_open

SWING_K = {"15m": 3, "1h": 3, "4h": 3, "1d": 2}
SWING_KEEP = 4
LEG_TFS = ("1h", "4h", "1d")
LEG_KEEP = 3
LIQ_LEVERAGE = (10.0, 25.0, 50.0)
LIQ_MMR = 0.005
LIQ_BIN_FRAC = 0.0025


def _r(x, nd: int = 6):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def wilder_atr(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> float | None:
    if len(c) < n + 1:
        return None
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(lo[1:], c[:-1])
    a = float(np.mean(tr[:n]))
    for x in tr[n:]:
        a = (a * (n - 1) + float(x)) / n
    return a


def dist(level: float, price: float, atr1h: float | None, atr1d: float | None) -> dict:
    return {"dist_pct": _r(100 * math.log(level / price), 3) if level > 0 and price > 0 else None,
            "dist_atr_1h": _r((level - price) / atr1h, 2) if atr1h else None,
            "dist_atr_1d": _r((level - price) / atr1d, 2) if atr1d else None}


# ---------------------------------------------------------------- swings

def swings(bars: pl.DataFrame, as_of: datetime, k: int = 3, keep: int = SWING_KEEP) -> dict:
    """Fractal swing highs / lows known at as_of. A high at i: high_i > max(high[i-k:i]) and high_i >= max(high[i+1:i+k+1])."""
    b = closed(bars, as_of)
    n = b.height
    out = {"highs": [], "lows": []}
    if n < 2 * k + 1:
        return out
    h, lo = b["high"].to_numpy(), b["low"].to_numpy()
    ev, ct = b["event_ts"].to_list(), b["close_ts"].to_list()
    for i in range(k, n - k):
        if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:i + k + 1].max():
            later = h[i + 1:]
            out["highs"].append({"price": float(h[i]), "bar_ts": ev[i].isoformat(), "known_ts": ct[i + k].isoformat(),
                                 "swept": bool((later > h[i]).any()), "i": i})
        if lo[i] < lo[i - k:i].min() and lo[i] <= lo[i + 1:i + k + 1].min():
            later = lo[i + 1:]
            out["lows"].append({"price": float(lo[i]), "bar_ts": ev[i].isoformat(), "known_ts": ct[i + k].isoformat(),
                                "swept": bool((later < lo[i]).any()), "i": i})
    for side in ("highs", "lows"):
        allp = out[side]
        recent = allp[-keep:]
        unswept = [p for p in allp if not p["swept"]][-keep:]
        seen, merged = set(), []
        for p in recent + unswept:
            if p["i"] not in seen:
                seen.add(p["i"])
                merged.append({x: y for x, y in p.items() if x != "i"})
        out[side] = sorted(merged, key=lambda p: p["bar_ts"])
    return out


# ---------------------------------------------------------------- prior periods

def period_levels(daily: pl.DataFrame, as_of: datetime) -> dict:
    """Current day / week / month open and the prior completed period's O H L C from UTC daily bars."""
    if daily.height == 0:
        return {}
    d_open = forming_open(daily, as_of)
    c = closed(daily, as_of)
    day0 = datetime(as_of.year, as_of.month, as_of.day, tzinfo=as_of.tzinfo)
    week0 = day0 - timedelta(days=day0.weekday())
    month0 = day0.replace(day=1)
    prev_week0 = week0 - timedelta(days=7)
    prev_month0 = (month0 - timedelta(days=1)).replace(day=1)
    allb = daily.filter(pl.col("event_ts") <= as_of).sort("event_ts")

    def agg(f: pl.DataFrame) -> dict | None:
        if f.height == 0:
            return None
        return {"open": float(f["open"][0]), "high": float(f["high"].max()), "low": float(f["low"].min()), "close": float(f["close"][-1]),
                "from": f["event_ts"][0].isoformat(), "bars": f.height}

    def first_open(start: datetime) -> float | None:
        f = allb.filter(pl.col("event_ts") >= start)
        return float(f["open"][0]) if f.height else None

    prev_day = c.filter(pl.col("event_ts") == day0 - timedelta(days=1))
    return {
        "day": {"open": d_open, "prior": agg(prev_day)},
        "week": {"open": first_open(week0), "start": week0.isoformat(), "prior": agg(c.filter((pl.col("event_ts") >= prev_week0) & (pl.col("event_ts") < week0)))},
        "month": {"open": first_open(month0), "start": month0.isoformat(), "prior": agg(c.filter((pl.col("event_ts") >= prev_month0) & (pl.col("event_ts") < month0)))},
    }


# ---------------------------------------------------------------- vecopen legs

def _vecopen_cfg():
    from vecopen.config import load_config

    return load_config()


def leg_levels(bars: pl.DataFrame, tf: str, as_of: datetime, keep: int = LEG_KEEP, cfg=None) -> dict:
    """Leg-origin opens from vecopen's streaming LegOriginDetector (read-only import; nothing in vecopen is modified)."""
    try:
        from vecopen.legs import Bar, LegOriginDetector

        cfg = cfg or _vecopen_cfg()
    except Exception as e:  # noqa: BLE001
        return {"status": f"vecopen unavailable: {type(e).__name__}: {e}"[:160]}
    lv = cfg.data["levels"]
    kappa, amax = float(lv["kappa_atr"]), int(lv["a_max_bars"])
    b = closed(bars, as_of)
    if b.height < 30:
        return {"status": "too few bars"}
    det = LegOriginDetector(tf, cfg)
    o, h, lo, c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
    v = b["volume"].to_numpy() if "volume" in b.columns else np.full(b.height, np.nan)
    ev = [int(t.timestamp() * 1000) for t in b["event_ts"].to_list()]
    ct = [int(t.timestamp() * 1000) for t in b["close_ts"].to_list()]
    legs = []
    for i in range(b.height):
        bar = Bar(tf, ct[i], float(o[i]), float(h[i]), float(lo[i]), float(c[i]), float(v[i]) if math.isfinite(v[i]) else None, ev[i])
        for leg in det.update(bar):
            legs.append((i, leg))
    out = []
    for i, leg in legs:
        zlo, zhi = leg.price - kappa * leg.atr, leg.price + kappa * leg.atr
        age = b.height - 1 - i
        if age >= amax:
            continue
        after_h, after_l = h[i + 1:], lo[i + 1:]
        touched = bool(((after_l <= zhi) & (after_h >= zlo)).any())
        out.append({"price": float(leg.price), "direction": "down-leg (level above)" if leg.direction < 0 else "up-leg (level below)",
                    "zone": [_r(zlo), _r(zhi)], "origin_ts": datetime.fromtimestamp(leg.origin_ts_close / 1000, as_of.tzinfo).isoformat(),
                    "birth_ts": datetime.fromtimestamp(leg.birth_ts / 1000, as_of.tzinfo).isoformat(),
                    "leg_size_atr": _r(leg.leg_size_atr, 2), "leg_vol": _r(leg.leg_vol, 2) if leg.leg_vol is not None else None,
                    "untouched": not touched, "age_bars": age})
    unt = [x for x in out if x["untouched"]]
    return {"status": "ok", "kappa_atr": kappa, "a_max_bars": amax, "n_live": len(out), "n_untouched": len(unt),
            "levels": unt[-keep:] + [x for x in out if not x["untouched"]][-1:]}


# ---------------------------------------------------------------- volume profile

def volume_profile(bars: pl.DataFrame, as_of: datetime, window: timedelta) -> dict | None:
    from ai_trader.features.profile import session_profile

    b = closed(bars, as_of).filter(pl.col("event_ts") >= as_of - window)
    if b.height < 10:
        return None
    p = session_profile(b["low"].to_numpy(), b["high"].to_numpy(), b["volume"].to_numpy())
    if p is None:
        return None
    return {"poc": p.poc, "val": p.val, "vah": p.vah, "va_share": _r(p.va_share, 3), "bars": b.height, "from": b["event_ts"][0].isoformat()}


# ---------------------------------------------------------------- estimated liquidation map

def liq_map(bars1h: pl.DataFrame, oi: pl.DataFrame | None, as_of: datetime, price: float, *, days: int = 7,
            leverage=LIQ_LEVERAGE, mmr: float = LIQ_MMR, top: int = 3) -> dict:
    """Estimated resting liquidation clusters (assumptions in the module docstring). USD notional per 0.25% bin."""
    if oi is None or oi.height == 0 or not price:
        return {"status": "no OI"}
    b = closed(bars1h, as_of).filter(pl.col("event_ts") >= as_of - timedelta(days=days))
    o = oi.filter((pl.col("event_ts") <= as_of) & (pl.col("known_ts") <= as_of)).sort("event_ts")
    if b.height < 24 or o.height < 24:
        return {"status": "too little history"}
    om = dict(zip(o["event_ts"].to_list(), o["oi"].to_list(), strict=True))
    ev = b["event_ts"].to_list()
    h, lo, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    tp = (h + lo + c) / 3
    # suffix extremes after each bar: the path a position opened in bar i has lived through since
    hi_after = np.r_[np.maximum.accumulate(h[::-1])[::-1][1:], -np.inf]
    lo_after = np.r_[np.minimum.accumulate(lo[::-1])[::-1][1:], np.inf]
    w = 1.0 / len(leverage)
    longs: dict[int, float] = {}
    shorts: dict[int, float] = {}
    step = math.log1p(LIQ_BIN_FRAC)
    for i, t in enumerate(ev):
        a, z = om.get(t), om.get(t + timedelta(hours=1))
        if a is None or z is None or z <= a:
            continue
        usd = (z - a) * tp[i]
        for L in leverage:
            pl_ = tp[i] * (1 - 1 / L + mmr)
            ps_ = tp[i] * (1 + 1 / L - mmr)
            if lo_after[i] > pl_ and pl_ < price:
                k = math.floor(math.log(pl_ / price) / step)
                longs[k] = longs.get(k, 0.0) + 0.5 * w * usd
            if hi_after[i] < ps_ and ps_ > price:
                k = math.floor(math.log(ps_ / price) / step)
                shorts[k] = shorts.get(k, 0.0) + 0.5 * w * usd

    def fmt(d: dict[int, float]) -> list[dict]:
        rows = sorted(d.items(), key=lambda kv: -kv[1])[:top]
        return [{"price": _r(price * math.exp((k + 0.5) * step)), "est_usd": round(v), "dist_pct": _r(100 * (k + 0.5) * step, 2)} for k, v in rows]

    return {"status": "ok", "model": f"OI increases over {days}d, 50/50 sides, leverage {list(leverage)} equal weights, mmr {mmr}; estimate",
            "long_liq_below": fmt(longs), "short_liq_above": fmt(shorts),
            "total_est_usd": {"long_below": round(sum(longs.values())), "short_above": round(sum(shorts.values()))}}


# ---------------------------------------------------------------- round numbers

def round_levels(price: float) -> dict:
    if not price or price <= 0:
        return {}
    e = math.floor(math.log10(price))
    major, minor = 10 ** e / 2, 10 ** (e - 1)
    out = {}
    for name, st in (("major", major), ("minor", minor)):
        lo = math.floor(price / st) * st
        hi = lo + st if lo < price else lo
        lo = lo if lo < price else lo - st
        out[name] = {"step": st, "below": _r(lo, 8), "above": _r(hi, 8)}
    return out


# ---------------------------------------------------------------- BTC field

def field_card(live_nn_dir: Path, as_of: datetime, days: int = 1) -> dict | None:
    lines = []
    for k in range(days, -1, -1):
        p = live_nn_dir / f"{(as_of - timedelta(days=k)).date().isoformat()}.jsonl"
        if p.is_file():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.startswith("{\"id\": \"nn-"):
                    try:
                        lines.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    ok = [x for x in lines if x.get("field") and datetime.fromisoformat(x["as_of"]) <= as_of]
    if not ok:
        return None
    ln = max(ok, key=lambda x: x["as_of"])
    f = ln["field"]
    close = f.get("close") or ln.get("close")
    hz = {}
    for h, v in f.get("horizons", {}).items():
        hz[h] = {"p_touch": v.get("p_touch"), "touch_prices": {k: _r(close * (1 + float(k.rstrip("%")) / 100), 1) for k in v.get("p_touch", {})},
                 "q05_q95_pct": [v["quantiles_pct"].get("q05"), v["quantiles_pct"].get("q95")], "p_jump": v.get("p_jump")}
    return {"line": ln["id"], "line_as_of": ln["as_of"], "close": close, "cell": f.get("cell"), "horizons": hz,
            "note": "touch = P(price reaches close x (1 + x%) within the horizon); not a direction call (R-FLD-3)"}


# ---------------------------------------------------------------- assembly

def token_geometry(sym: str, price: float, bars: dict[str, pl.DataFrame], as_of: datetime, *, oi: pl.DataFrame | None = None,
                   vecopen_cfg=None) -> dict:
    """bars: tf -> frame for '1m', '15m', '1h', '4h', '1d' (any may be missing)."""
    def atr(tf):
        b = closed(bars.get(tf, pl.DataFrame()), as_of) if bars.get(tf) is not None else None
        return wilder_atr(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()) if b is not None and b.height > 15 else None
    a1h, a1d = atr("1h"), atr("1d")
    out: dict = {"symbol": sym, "price": price, "atr14": {"1h": _r(a1h), "4h": _r(atr("4h")), "1d": _r(a1d)}}
    sw = {}
    for tf, k in SWING_K.items():
        if bars.get(tf) is not None:
            s = swings(bars[tf], as_of, k)
            for side in ("highs", "lows"):
                for p in s[side]:
                    p.update(dist(p["price"], price, a1h, a1d))
            sw[tf] = s
    out["swings"] = sw
    per = period_levels(bars["1d"], as_of) if bars.get("1d") is not None else {}
    for v in per.values():
        if v.get("open"):
            v["open_dist"] = dist(v["open"], price, a1h, a1d)
        if v.get("prior"):
            for x in ("high", "low"):
                v["prior"][f"{x}_dist"] = dist(v["prior"][x], price, a1h, a1d)
    out["periods"] = per
    legs = {}
    for tf in LEG_TFS:
        if bars.get(tf) is not None:
            lg = leg_levels(bars[tf], tf, as_of, cfg=vecopen_cfg)
            for x in lg.get("levels", []):
                x.update(dist(x["price"], price, a1h, a1d))
            legs[tf] = lg
    out["legs"] = legs
    prof = {}
    if bars.get("1m") is not None:
        prof["24h"] = volume_profile(bars["1m"], as_of, timedelta(hours=24))
    if bars.get("15m") is not None:
        prof["7d"] = volume_profile(bars["15m"], as_of, timedelta(days=7))
    for v in prof.values():
        if v:
            for x in ("poc", "val", "vah"):
                v[f"{x}_dist"] = dist(v[x], price, a1h, a1d)
    out["profile"] = {**prof, "source": {"24h": "1m bars", "7d": "15m bars"}}
    if bars.get("1h") is not None:
        out["liq_map"] = liq_map(bars["1h"], oi, as_of, price)
    rl = round_levels(price)
    for v in rl.values():
        for x in ("below", "above"):
            v[f"{x}_dist"] = dist(v[x], price, a1h, a1d)
    out["round"] = rl
    out["nearest"] = nearest_levels(out, price)
    return out


def flat_levels(g: dict) -> list[dict]:
    rows = []
    for tf, s in g.get("swings", {}).items():
        for side, kind in (("highs", "swing high"), ("lows", "swing low")):
            for p in s.get(side, []):
                rows.append({"kind": f"{tf} {kind}" + (" (swept)" if p["swept"] else ""), "price": p["price"]})
    for name, v in g.get("periods", {}).items():
        if v.get("open"):
            rows.append({"kind": f"{name} open", "price": v["open"]})
        if v.get("prior"):
            rows += [{"kind": f"prior {name} high", "price": v["prior"]["high"]}, {"kind": f"prior {name} low", "price": v["prior"]["low"]}]
    for tf, lg in g.get("legs", {}).items():
        for x in lg.get("levels", []):
            rows.append({"kind": f"{tf} leg origin" + ("" if x["untouched"] else " (touched)"), "price": x["price"]})
    for w, p in g.get("profile", {}).items():
        if isinstance(p, dict) and "poc" in p:
            rows += [{"kind": f"{w} POC", "price": p["poc"]}, {"kind": f"{w} VAH", "price": p["vah"]}, {"kind": f"{w} VAL", "price": p["val"]}]
    for name, v in g.get("round", {}).items():
        rows += [{"kind": f"round {name}", "price": v["below"]}, {"kind": f"round {name}", "price": v["above"]}]
    return rows


def nearest_levels(g: dict, price: float, n: int = 4) -> dict:
    a1h, a1d = g["atr14"]["1h"], g["atr14"]["1d"]
    rows = [{**r, **dist(r["price"], price, a1h, a1d)} for r in flat_levels(g) if r["price"]]
    above = sorted((r for r in rows if r["price"] > price), key=lambda r: r["price"])[:n]
    below = sorted((r for r in rows if r["price"] < price), key=lambda r: -r["price"])[:n]
    return {"above": above, "below": below}


def build_geometry(as_of: datetime, tokens: dict[str, dict], *, field: dict | None = None, reasons: dict[str, list[str]] | None = None) -> dict:
    """tokens: sym -> {"price", "bars": {tf: frame}, "oi": frame}."""
    try:
        cfg = _vecopen_cfg()
        vstat = {"status": "ok", "config_sha256": getattr(cfg, "sha256", None) or getattr(cfg, "hash", None)}
    except Exception as e:  # noqa: BLE001
        cfg, vstat = None, {"status": f"unavailable: {type(e).__name__}"}
    out = {"domain": "geometry", "as_of": as_of.isoformat(), "vecopen": vstat,
           "note": "levels are places, not predictions: R-PX-4 (breaks slightly more often than holds), R-PX-10, R-AVOID-1",
           "conventions": "dist_pct = 100 ln(level/price); dist_atr = (level - price) / Wilder ATR14 of 1h or 1d closed bars",
           "tokens": []}
    for s, t in tokens.items():
        try:
            g = token_geometry(s, t["price"], t["bars"], as_of, oi=t.get("oi"), vecopen_cfg=cfg)
        except Exception as e:  # noqa: BLE001
            g = {"symbol": s, "error": f"{type(e).__name__}: {e}"[:200]}
        g["why"] = (reasons or {}).get(s, [])
        out["tokens"].append(g)
    if field:
        out["btc_field"] = field
    return out


__all__ = ["build_geometry", "dist", "field_card", "leg_levels", "liq_map", "nearest_levels", "period_levels", "round_levels",
           "swings", "token_geometry", "volume_profile", "wilder_atr"]
