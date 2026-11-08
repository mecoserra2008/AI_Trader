"""Pivot-geometry (DC pivots) study for H-GT1, H-GT2, H-GT3 (research/hypotheses.yaml).

Design (RESEARCH_PROTOCOL §3b): signals, pivots and race outcomes are computed on each symbol's FULL 4h
history; P&L / events are split afterwards at the knowledge cutoff (by the UTC date of the P&L, or the
event_ts of the touch bar).

H-GT1/H-GT2 (H-X2b method): daily net P&L per symbol (sum of evaluate.strategy_pnl by UTC date),
annualised Sharpe = mean/std*sqrt(365) per segment (> 30 days), averaged over symbols.
H-GT3: K = 6. After each confirmed swing high H0 (known at its confirmation bar c), the first bar j > c
whose high >= H0, searched until the next swing high is confirmed (the level is then no longer the last
one); race on closes from j over 200 bars: rejection = close <= H0 - sigma_j before close >= H0 + sigma_j,
sigma_j = indicator sigma (price points) at the touch. Lows mirror. Unresolved / truncated races dropped.

    uv run python -m ai_trader.research.geo_study --segment llm_in_sample [--record]
    uv run python -m ai_trader.research.geo_study --segment clean_holdout [--record]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import polars as pl

from ai_trader.core.time import now
from ai_trader.features.dc_pivots import KS, run_engines
from ai_trader.research.evaluate import knowledge_cutoff, strategy_pnl
from ai_trader.research.frame import market_frame
from ai_trader.strategies import REGISTRY

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
GEO = ("geo_struct", "geo_pivot_bo", "geo_flag_bo")
BASE = ("ema_stack", "donchian_breakout", "tsmom")
MODELS = (*GEO, "geo_avg", *BASE)
SEGS = ("llm_in_sample", "clean_holdout")
RACE_BARS = 200
LEVEL_K = 6
MIN_DAYS = 30


def daily_sharpe(pnl: pl.DataFrame) -> dict[str, float]:
    d = pnl.group_by(pl.col("event_ts").dt.date().alias("day")).agg(pl.col("pnl").sum()).sort("day")
    cut = knowledge_cutoff().date()
    out = {}
    for seg, sub in (("llm_in_sample", d.filter(pl.col("day") < cut)), ("clean_holdout", d.filter(pl.col("day") >= cut))):
        x = sub["pnl"].to_numpy()
        out[seg] = float(x.mean() / x.std(ddof=1) * np.sqrt(365)) if len(x) > MIN_DAYS and x.std(ddof=1) > 0 else float("nan")
        out[f"{seg}_days"] = len(x)
    return out


def touch_race(close: np.ndarray, hi: np.ndarray, lo: np.ndarray, sig: np.ndarray, ready: np.ndarray, events: pl.DataFrame) -> list[dict]:
    n = len(close)
    rows = []
    for ty in (1, -1):
        ev = events.filter(pl.col("type") == ty).sort("confirm_bar")
        prices, cbars = ev["price"].to_numpy(), ev["confirm_bar"].to_numpy()
        for i, (lvl, c) in enumerate(zip(prices, cbars)):
            if not ready[c]:
                continue
            stop = int(cbars[i + 1]) if i + 1 < len(cbars) else n
            seg = hi[c + 1:stop] >= lvl if ty == 1 else lo[c + 1:stop] <= lvl
            hits = np.flatnonzero(seg)
            if not len(hits):
                continue
            c = int(c)
            j = c + 1 + int(hits[0])
            s = sig[j]
            if not np.isfinite(s) or s <= 0:
                continue
            path = close[j:min(n, j + RACE_BARS)]
            away = np.flatnonzero(path <= lvl - s) if ty == 1 else np.flatnonzero(path >= lvl + s)
            thru = np.flatnonzero(path >= lvl + s) if ty == 1 else np.flatnonzero(path <= lvl - s)
            a = int(away[0]) if len(away) else None
            b = int(thru[0]) if len(thru) else None
            if a is None and b is None:
                continue  # unresolved within 200 bars (or truncated by the end of data): dropped
            # Diagnostic null: the race starts at close_j, usually back on the near side of the level (only the
            # wick touched). x = start offset in sigma toward rejection; a driftless walk rejects w.p. (1+x)/2.
            x = float(np.clip(ty * (lvl - close[j]) / s, -1.0, 1.0))
            rows.append({"touch_bar": j, "type": ty, "reject": b is None or (a is not None and a < b), "rw_null": (1 + x) / 2})
    return rows


def symbol_study(sym: str, as_of: datetime) -> dict | None:
    f = market_frame(sym, "4h", as_of, funding=False, oi=False)
    if f.height < 400:
        return None
    sigs = {m: REGISTRY[m]()(f) for m in (*GEO, *BASE)}
    sigs["geo_avg"] = ((sigs["geo_struct"] + sigs["geo_pivot_bo"] + sigs["geo_flag_bo"]) / 3).alias("geo_avg")
    sharpe = {m: daily_sharpe(strategy_pnl(f, sigs[m])) for m in MODELS}
    exposure = {m: float((sigs[m].to_numpy() != 0).mean()) for m in MODELS}
    corr = {m: float(np.corrcoef(sigs[m].to_numpy(), sigs["ema_stack"].to_numpy())[0, 1]) for m in ("geo_avg", *GEO)}

    close = f["close"].to_numpy().astype(float)
    sig, ready, res = run_engines(close, KS)
    sn = f["sigma_next"].to_numpy()
    ts = f["event_ts"]
    stops = []
    for k, r in res.items():
        for t in np.flatnonzero((r.flag_entry != 0) & ready):
            d = r.flag_entry[t] * (close[t] - r.fstop[t]) / (sn[t] * close[t])
            stops.append({"k": k, "event_ts": ts[int(t)], "dist_sigma": float(d), "dist_k": float(abs(close[t] - r.fstop[t]) / sig[t - 1] / k)})
    races = touch_race(close, f["high"].to_numpy().astype(float), f["low"].to_numpy().astype(float), sig, ready, res[LEVEL_K].events)
    for row in races:
        row["event_ts"] = ts[row.pop("touch_bar")]
        row["symbol"] = sym
    return {"symbol": sym, "sharpe": sharpe, "exposure": exposure, "corr_ema": corr, "stops": stops, "races": races}


def binom_z(k: int, n: int, p0: float = 0.5) -> float:
    return float((k - n * p0) / np.sqrt(n * p0 * (1 - p0))) if n else float("nan")


def run(as_of: datetime | None = None) -> dict:
    as_of = as_of or now()
    per = [r for r in (symbol_study(s, as_of) for s in UNIVERSE) if r is not None]
    table = {}
    for m in MODELS:
        table[m] = {}
        for seg in SEGS:
            v = np.array([r["sharpe"][m][seg] for r in per])
            table[m][seg] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            table[m][f"{seg}_n_sym"] = int(np.isfinite(v).sum())
            table[m][f"{seg}_frac_pos"] = float(np.nanmean(v > 0)) if np.isfinite(v).any() else float("nan")
        table[m]["exposure"] = float(np.mean([r["exposure"][m] for r in per]))
    corr = {m: float(np.nanmean([r["corr_ema"][m] for r in per])) for m in ("geo_avg", *GEO)}
    st = pl.DataFrame([s for r in per for s in r["stops"]])
    stops = {}
    for name, sub in (("all_k", st), ("k6", st.filter(pl.col("k") == 6))):
        x = sub["dist_sigma"].to_numpy()
        stops[name] = {"n": len(x), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
                       "frac_gt4": float((x > 4).mean()), "median_in_K_units": float(np.median(sub["dist_k"].to_numpy()))}
    rc = pl.DataFrame([x for r in per for x in r["races"]])
    k_cut = knowledge_cutoff()
    gt3 = {}
    for seg, sub in (("llm_in_sample", rc.filter(pl.col("event_ts") < k_cut)), ("clean_holdout", rc.filter(pl.col("event_ts") >= k_cut))):
        n, k = sub.height, int(sub["reject"].sum())
        gt3[seg] = {"n_resolved": n, "n_reject": k, "reject_rate": k / n if n else float("nan"), "binom_z": binom_z(k, n),
                    "rate_highs": float(sub.filter(pl.col("type") == 1)["reject"].mean() or float("nan")),
                    "rate_lows": float(sub.filter(pl.col("type") == -1)["reject"].mean() or float("nan")),
                    "rw_null": float(sub["rw_null"].mean()) if n else float("nan"),
                    "z_vs_rw_null": float((k - sub["rw_null"].sum()) / np.sqrt((sub["rw_null"] * (1 - sub["rw_null"])).sum())) if n else float("nan"),
                    "per_symbol": dict(sub.group_by("symbol").agg(pl.len()).sort("symbol").iter_rows())}
    return {"n_symbols": len(per), "table": table, "corr_with_ema_stack": corr, "flag_stop_dist": stops, "gt3": gt3,
            "per_symbol_sharpe": {r["symbol"]: {m: r["sharpe"][m] for m in MODELS} for r in per}}


# ------------------------------------------------------------------ verdicts (mechanical, from the registry)

def verdict_gt1(t: dict, seg: str) -> str:
    v = [t[m][seg] for m in GEO]
    if not all(np.isfinite(v)):
        return "inconclusive"
    return "supports" if all(x > 0 for x in v) else "contradicts"


def verdict_gt2(t: dict, seg: str) -> str:
    a, b = t["geo_avg"][seg], t["ema_stack"][seg]
    if not (np.isfinite(a) and np.isfinite(b)):
        return "inconclusive"
    return "supports" if a > b else "contradicts"


def verdict_gt3(r: dict, seg: str) -> str:
    if seg == "clean_holdout":
        return "inconclusive" if r["n_resolved"] < 8 else ("supports" if r["reject_rate"] > 0.5 else "contradicts")
    return "supports" if r["reject_rate"] > 0.55 and r["binom_z"] > 2 else "contradicts"


def record_verdicts(out: dict, seg: str) -> None:
    from ai_trader.research.hypotheses import record

    t = out["table"]
    data = "bybit 4h, 23-symbol universe, market_frame(funding=False, oi=False), signals on full history"
    m1 = {m: {"sharpe": t[m][seg], "n_sym": t[m][f"{seg}_n_sym"], "frac_pos": t[m][f"{seg}_frac_pos"]} for m in (*GEO, "geo_avg", *BASE)}
    record("H-GT1", seg, verdict_gt1(t, seg), m1, note="mean over symbols of annualised daily net Sharpe", data=data)
    record("H-GT2", seg, verdict_gt2(t, seg), {"geo_avg": t["geo_avg"][seg], "ema_stack": t["ema_stack"][seg],
           "corr_geo_avg_ema": out["corr_with_ema_stack"]["geo_avg"]}, note="geo_avg = mean of the three geo signals", data=data)
    g = {k: v for k, v in out["gt3"][seg].items() if k != "per_symbol"}
    record("H-GT3", seg, verdict_gt3(out["gt3"][seg], seg), g, note="K=6 first touch after confirmation, 1-sigma race on closes, 200-bar cap", data=data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", choices=SEGS, required=True)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    out = run()
    seg = a.segment
    view = {
        "segment": seg,
        "n_symbols": out["n_symbols"],
        "sharpe": {m: {k: v for k, v in out["table"][m].items() if k.startswith(seg) or k == "exposure"} for m in MODELS},
        "gt3": out["gt3"][seg],
        "verdicts": {"H-GT1": verdict_gt1(out["table"], seg), "H-GT2": verdict_gt2(out["table"], seg), "H-GT3": verdict_gt3(out["gt3"][seg], seg)},
    }
    if seg == "llm_in_sample":
        view["corr_with_ema_stack"] = out["corr_with_ema_stack"]
        view["flag_stop_dist"] = out["flag_stop_dist"]
    print(json.dumps(view, indent=1, default=str))
    if a.record:
        record_verdicts(out, seg)


if __name__ == "__main__":
    main()
