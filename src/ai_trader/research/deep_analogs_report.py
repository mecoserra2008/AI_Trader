"""Deep analogs report builder: runs the fingerprints (deep_analogs_features.run), the level-probability sources, the
figures, and writes research/reports/deep_analogs_2026-11-15.{md,json}. Narrative text lives in NARRATIVE (written after
reading the computed tables; descriptive, chosen after seeing today).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m ai_trader.research.deep_analogs_report [P0]
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai_trader.research import deep_analogs_features as F
from ai_trader.research.deep_analogs_figures import make_all
from ai_trader.research.deep_analogs_study import EPISODES, OUT

ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "research" / "reports" / "deep_analogs_2026-11-15"
FIELD_CARD = {1: 0.87, -1: 0.82, 2: 0.73, -2: 0.66, 3: 0.60, -3: 0.53}  # research/live/LATEST.md, line nn-20261115T19, 24h


def live_price() -> tuple[float, str]:
    with urllib.request.urlopen("https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT", timeout=20) as r:
        js = json.loads(r.read())
    return float(js["result"]["list"][0]["lastPrice"]), datetime.now(UTC).isoformat(timespec="seconds")


def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def analog_level_stats(res: dict, tag: str) -> dict:
    eps = [e for e in res["fingerprints"] if e != "NOW"]
    rows = [res["fingerprints"][e]["outcome"][f"levels_{tag}"] for e in eps]
    n = len(rows)
    dn = sum(1 for r in rows if r["first"] == "down") + 0.5 * sum(1 for r in rows if r["first"] == "both")
    up = sum(1 for r in rows if r["first"] == "up") + 0.5 * sum(1 for r in rows if r["first"] == "both")
    ne = sum(1 for r in rows if r["first"] == "neither")
    tdn = [r for r in rows if r["touch_dn"]]
    tup = [r for r in rows if r["touch_up"]]
    return {"n": n, "p_dn_first": dn / n, "p_up_first": up / n, "p_neither": ne / n,
            "n_touch_dn": len(tdn), "p58_given_dn": (sum(r["touch_58k"] for r in tdn) / len(tdn)) if tdn else None,
            "n_touch_up": len(tup), "p68_given_up": (sum(r["touch_68k"] for r in tup) / len(tup)) if tup else None,
            "p_flush_equiv_undercut": sum(r["undercut_flush_equiv"] for r in rows) / n,
            "first": {e: r["first"] for e, r in zip(eps, rows, strict=True)}}


def top20_stats(res: dict) -> dict:
    rows = res["top20"]["rows"]
    n = len(rows)
    dn = sum(r["first"] == "down" for r in rows) + 0.5 * sum(r["first"] == "both" for r in rows)
    up = sum(r["first"] == "up" for r in rows) + 0.5 * sum(r["first"] == "both" for r in rows)
    tdn = [r for r in rows if r["touch_dn"]]
    tup = [r for r in rows if r["touch_up"]]
    return {"n": n, "p_dn_first": dn / n, "p_up_first": up / n, "p_neither": sum(r["first"] == "neither" for r in rows) / n,
            "n_touch_dn": len(tdn), "p58_given_dn": sum(r["touch_58k"] for r in tdn) / len(tdn) if tdn else None,
            "n_touch_up": len(tup), "p68_given_up": sum(r["touch_68k"] for r in tup) / len(tup) if tup else None,
            "p_lower_low14": sum(r["lower_low14"] for r in rows) / n}


def main() -> None:
    if len(sys.argv) > 1:
        P0, p_ts = float(sys.argv[1]), "given"
    else:
        P0, p_ts = live_price()
    res = F.run(P0, FIELD_CARD)
    A_raw, A_vs, T20 = analog_level_stats(res, "raw"), analog_level_stats(res, "volscaled"), top20_stats(res)
    b, fld = res["baseline"], res["field"]

    def mc_row(name, m):
        return (name, m["dn_first"], m["up_first"], m["neither"], m["p58_given_dn"], m["p68_given_up"])

    lvl = [("10 deep analogs, same % (14 d)", A_raw["p_dn_first"], A_raw["p_up_first"], A_raw["p_neither"], A_raw["p58_given_dn"], A_raw["p68_given_up"]),
           ("10 deep analogs, vol-scaled (14 d)", A_vs["p_dn_first"], A_vs["p_up_first"], A_vs["p_neither"], A_vs["p58_given_dn"], A_vs["p68_given_up"]),
           ("top-20 analog list, 1h bars (14 d)", T20["p_dn_first"], T20["p_up_first"], T20["p_neither"], T20["p58_given_dn"], T20["p68_given_up"]),
           mc_row("field card, 24 h (BM fit)", fld["d1"]),
           mc_row("field BM extrapolated to 14 d", fld["d14_extrapolated"]),
           mc_row(f"driftless, 7d RV {b['rv7d']['sig_ann']:.0%} (14 d)", b["rv7d"]["d14"]),
           mc_row(f"driftless, DVOL {b['dvol']['sig_ann']:.0%} (14 d)", b["dvol"]["d14"])]
    make_all(lvl)
    extra = {"P0_ts": p_ts, "analog_levels_raw": A_raw, "analog_levels_volscaled": A_vs, "top20_levels": T20, "level_table": lvl}
    extra["markers"] = marker_table(res)
    res.update(extra)
    REP.parent.mkdir(parents=True, exist_ok=True)
    (REP.with_suffix(".json")).write_text(json.dumps(res, indent=1, default=str))
    (OUT / "report_inputs.json").write_text(json.dumps(extra, indent=1, default=str))
    print("P0", P0, p_ts)
    print(json.dumps({"A_raw": A_raw, "A_vs": A_vs, "T20": T20}, indent=1, default=str)[:4000])




MARKERS = [("climax", "volume climax (max 1h vol / 30d median, 72h)"), ("vol_at_low", "volume in the low hour (x median)"),
           ("forced_sell_h", "forced-selling proxy notional (hours of median volume, 72h)"), ("forced_sell_share", "forced selling share of forced flow"),
           ("oi_flush", "OI flush (log, into the low)"), ("oi_rebuild", "OI change low -> T (log)"),
           ("f_last_bp", "funding, last print (bp / 8h)"), ("f_min72_bp", "funding, 72h min (bp / 8h)"), ("n_neg72", "negative funding prints in 72h"),
           ("top_pos_chg72", "top-trader position L/S, 72h log change"), ("top_acc_chg72", "top-trader account L/S, 72h log change"),
           ("glob_acc_chg72", "global account L/S, 72h log change"), ("taker_ratio24", "taker buy/sell vol ratio, 24h"),
           ("delta72", "taker delta share, 72h"), ("delta_post_low", "taker delta share, low -> T"), ("retrace_gap", "price retrace minus CVD retrace"),
           ("cvd_lead_h", "hours the CVD low led the price low"), ("basis6h", "spot-perp basis, last 6h (bp)"), ("basis_min72", "basis 72h min (bp)"),
           ("depth_min_ratio", "book depth +-1% min vs pre-cascade"), ("depth_last_ratio", "book depth at T vs pre-cascade"),
           ("rv_ratio", "24h RV / 30d median"), ("dvol_vs30", "DVOL vs 30d median (log)"), ("dvol_chg72", "DVOL 72h change (log)"),
           ("fng", "Fear & Greed"), ("dd30", "drawdown from 30d high"), ("rebound", "rebound off the flush low at T"), ("h_since_low", "hours since the low")]


def marker_table(res: dict) -> list[dict]:
    fps = res["fingerprints"]
    sd = res["similarity"]["sd"]
    bot = [e for e in fps if e != "NOW" and not fps[e]["outcome"].get("undercut14d")]
    low = [e for e in fps if e != "NOW" and fps[e]["outcome"].get("undercut14d")]
    out = []
    for f, lab in MARKERS:
        now = fps["NOW"]["features"].get(f)
        vb = [fps[e]["features"].get(f) for e in bot if fps[e]["features"].get(f) is not None]
        vl = [fps[e]["features"].get(f) for e in low if fps[e]["features"].get(f) is not None]
        mb = float(np.median(vb)) if vb else None
        ml = float(np.median(vl)) if vl else None
        closer = None
        if now is not None and mb is not None and ml is not None and sd.get(f):
            db, dl = abs(now - mb) / sd[f], abs(now - ml) / sd[f]
            closer = "bottomed" if db < dl - 0.15 else ("lower low" if dl < db - 0.15 else "neither")
        out.append({"feature": f, "label": lab, "now": now, "bottomed_median": mb, "n_b": len(vb), "lower_low_median": ml, "n_l": len(vl),
                    "now_closer_to": closer})
    return out


if __name__ == "__main__":
    main()
