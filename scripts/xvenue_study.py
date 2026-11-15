"""Cross-venue study, Bybit perps vs BlackBull cTrader CFDs (H-ARB1, H-ARB2). Library: ai_trader.research.xvenue.

Steps (each idempotent; data under data/research/xvenue/, never the lake):

    uv run python scripts/xvenue_study.py split      # seeded day draw -> research/plans/xvenue_split_2026-11-14.json (once)
    uv run python scripts/xvenue_study.py bybit      # Bybit 1 s books (ob200 archives, <= 2 workers), USDCUSDT 1 m, funding
    target/debug/atx ctrader check                   # needs the CTRADER_* keys in .env
    target/debug/atx ctrader symbols --resolve       # -> data/research/xvenue/ctrader_map.json
    uv run python scripts/xvenue_study.py ctrader    # cTrader ticks for the split days (calls atx ctrader ticks)
    uv run python scripts/xvenue_study.py run [--no-record]   # study -> research/reports/xvenue_2026-11-14.{md,json}

Nothing here sends orders. SPYUSDT is excluded by config (and refused by the loader).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.research import xvenue as X  # noqa: E402

TAG = "2026-11-14"
SPLIT = ROOT / "research" / "plans" / f"xvenue_split_{TAG}.json"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
REST = "https://api.bybit.com"
UA = "Mozilla/5.0 (AI_Trader research; public archives)"
WINDOW = ("2026-08-03", "2026-11-13")       # all after research.knowledge_cutoff (2026-07-01): clean_holdout
# Latency, measured before any cTrader data (no orders were sent for this study):
#   Bybit: demo order/create p50 0.66 s, p90 10.6 s (scripts/conn_probe.py, 2026-11-09); mainnet REST p50 0.47-0.82 s
#          (2026-11-14, data/research/xvenue/latency_bybit_2026-11-14.txt)
#   cTrader: ProtoOAVersionReq round trip p50 0.12 s, p90 0.29 s (atx ctrader ping, 2026-11-14); order round trip not
#            measurable without orders -> assumed equal to Bybit's (the slower venue bounds a two-leg trade)
LATENCY = {"rtt_p50_s": 0.66, "rtt_p90_s": 10.6, "source": "Bybit demo order/create 2026-11-09 (slower venue); cTrader request RTT p50 0.12 s"}


def _get_json(url: str, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001 - retried
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"GET failed: {url}")


def launch_dates() -> dict[str, str]:
    d = pl.read_parquet(ROOT / "data/lake/instruments/venue=bybit/*.parquet").sort("known_ts").unique(["symbol", "category"], keep="last")
    d = d.filter(pl.col("category") == "linear").select("symbol", pl.col("launch_ts").dt.strftime("%Y-%m-%d"))
    return {s: v for s, v in d.iter_rows() if v}


def load_split() -> dict[str, Any]:
    if not SPLIT.exists():
        sys.exit("no split yet: run `xvenue_study.py split` first")
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def all_days(split: dict[str, Any], sym: str) -> list[str]:
    v = split["instruments"][sym]
    return sorted(set(v["days"]) | set(v["reopen_sundays"]))


# ---------------------------------------------------------------------------------------------------------------- split


def cmd_split(_: argparse.Namespace) -> None:
    c = X.cfg()
    insts = X.study_instruments(c)
    cached = {i["bybit"]: X.micro_days_cached(i["bybit"]) for i in insts}
    cached = {k: v for k, v in cached.items() if v}
    split = X.draw_split(insts, *WINDOW, launch_dates(), seed=int(c["seed"]), cached=cached)
    meta = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "hypotheses": ["H-ARB1", "H-ARB2"], "seed": c["seed"], "window": WINDOW,
            "segment": "clean_holdout (every day is after research.knowledge_cutoff 2026-07-01)",
            "rules": "5 days per study instrument from day_pool(): weekdays (any day for crypto), listed >= 7 days, no US holiday for stocks/ETFs; "
                     "where the Bybit 1 s book already exists in data/research/micro_days the pool is restricted to those days (no duplicate "
                     "ob200 downloads; availability only). Sundays before sampled Mondays are added for FX / metals / energy (weekend "
                     "reopen, descriptive only, never in H-ARB1/2 counts).",
            "latency": LATENCY,
            "pre_specified": {"arb1_min_episode_s": f"> 2 x rtt_p50 = {2 * LATENCY['rtt_p50_s']:.2f} s", "fill_delay_s": math.ceil(LATENCY["rtt_p50_s"]),
                              "sensitivity": f"rtt_p90: episodes > {2 * LATENCY['rtt_p90_s']:.1f} s, fill delay {math.ceil(LATENCY['rtt_p90_s'])} s",
                              "fees": "round trip = 2 x (Bybit taker from config/fees.yaml + cTrader commission per side from symbol details)",
                              "exit": "gap halves, 60 min, or CFD session end", "arb2_ci": "day-clustered bootstrap, 4000 draws"},
            "written_before_ctrader_data": not any((X.xdir() / "ctrader_ticks").glob("*/*.parquet"))}
    X.write_split(SPLIT, split, meta)
    print(json.dumps(split, indent=1))
    print("->", SPLIT)


# ---------------------------------------------------------------------------------------------------------------- bybit

BOOK_COLS = ["ts", "bid1", "ask1", "bid1_size", "ask1_size", "n_updates"]


def _book_job(sym: str, d: str) -> dict[str, Any]:
    from ai_trader.research.micro_days import download
    from ai_trader.sim.book import build_series

    t0 = time.time()
    out = X.xdir() / "bybit_book_1s" / sym / f"{d}.parquet"
    raw = ROOT / "data" / "raw" / "xvenue" / f"{d}_{sym}_ob200.data.zip"
    r = download(OB_URL.format(s=sym, d=d), raw)
    if r["status"] != "ok":
        return {"sym": sym, "date": d, **r}
    try:
        df, qa = build_series(raw, bin_ms=1000)
        d0 = int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp() * 1000)
        df = df.with_columns(pl.col("ts").dt.epoch("ms").alias("_t")).filter((pl.col("_t") > d0) & (pl.col("_t") <= d0 + 86_400_000)).select(BOOK_COLS)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out, compression="zstd")
    finally:
        raw.unlink(missing_ok=True)
    return {"sym": sym, "date": d, "status": "ok", "rows": df.height, "zip_mb": round(r["bytes"] / 1e6, 1), "secs": round(time.time() - t0), "qa": qa}


def cmd_bybit(a: argparse.Namespace) -> None:
    split = load_split()
    jobs = [(s, d) for s in split["instruments"] for d in all_days(split, s) if X.bybit_book_path(s, d) is None]
    print(f"{len(jobs)} Bybit symbol-days to build (others reuse micro_days / earlier runs)")
    log = X.xdir() / "bybit_book_1s" / "_qa.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(2, a.workers)) as ex:
        futs = {ex.submit(_book_job, s, d): (s, d) for s, d in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - logged
                r = {"sym": futs[f][0], "date": futs[f][1], "status": "error", "error": repr(e)}
            print(json.dumps({k: v for k, v in r.items() if k != "qa"}), flush=True)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, default=str) + "\n")
    # USDT basis: Bybit spot USDCUSDT 1 m (each split day and the day before)
    days = sorted({x for s in split["instruments"] for d in all_days(split, s) for x in (d, (date.fromisoformat(d) - timedelta(days=1)).isoformat())})
    p = X.xdir() / "usdcusdt_1m.parquet"
    have = set(pl.read_parquet(p)["day"].unique().to_list()) if p.exists() else set()
    rows = []
    for d in days:
        if d in have:
            continue
        s0 = int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp() * 1000)
        for half in (0, 1):
            st, en = s0 + half * 43_200_000, s0 + (half + 1) * 43_200_000 - 1
            j = _get_json(f"{REST}/v5/market/kline?category=spot&symbol=USDCUSDT&interval=1&start={st}&end={en}&limit=1000")
            rows += [{"day": d, "open_s": int(k[0]) // 1000, "close": float(k[4])} for k in j["result"]["list"]]
    if rows:
        new = pl.DataFrame(rows)
        (pl.concat([pl.read_parquet(p), new]) if p.exists() else new).unique(["open_s"]).sort("open_s").write_parquet(p)
    print("USDCUSDT 1m:", pl.read_parquet(p).height if p.exists() else 0, "rows")
    # funding history (public REST; the lake has gaps for recently added TradFi perps)
    fp = X.xdir() / "funding.parquet"
    st = int(datetime.fromisoformat(WINDOW[0]).replace(tzinfo=UTC).timestamp() * 1000) - 86_400_000
    en = int(datetime.fromisoformat(WINDOW[1]).replace(tzinfo=UTC).timestamp() * 1000) + 2 * 86_400_000
    frows = []
    for s in split["instruments"]:
        end = en
        while end > st:
            j = _get_json(f"{REST}/v5/market/funding/history?category=linear&symbol={s}&startTime={st}&endTime={end}&limit=200")
            lst = j["result"]["list"]
            if not lst:
                break
            frows += [{"symbol": s, "ts_s": int(x["fundingRateTimestamp"]) // 1000, "rate": float(x["fundingRate"])} for x in lst]
            end = min(int(x["fundingRateTimestamp"]) for x in lst) - 1
            if len(lst) < 200:
                break
    if frows:
        pl.DataFrame(frows).unique(["symbol", "ts_s"]).sort("symbol", "ts_s").write_parquet(fp)
        print("funding:", len(frows), "rows ->", fp)


# ---------------------------------------------------------------------------------------------------------------- ctrader


def ctrader_map() -> dict[str, Any]:
    p = X.xdir() / "ctrader_map.json"
    if not p.exists():
        sys.exit("no ctrader_map.json: run `target/debug/atx ctrader check` and `target/debug/atx ctrader symbols --resolve` first")
    return json.loads(p.read_text(encoding="utf-8"))["map"]


def cmd_ctrader(_: argparse.Namespace) -> None:
    split, cmap = load_split(), ctrader_map()
    atx = ROOT / "target" / "debug" / ("atx.exe" if os.name == "nt" else "atx")
    for s in split["instruments"]:
        name = (cmap.get(s) or {}).get("ctrader_name")
        if not name:
            print(f"{s}: not resolved on this account, skipped")
            continue
        for d in all_days(split, s):
            nd = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            r = subprocess.run([str(atx), "ctrader", "ticks", "--symbol", name, "--from", d, "--to", nd, "--side", "both"], cwd=ROOT, capture_output=True, text=True)
            print((r.stdout or r.stderr).strip().splitlines()[-1:] if (r.stdout or r.stderr) else f"{s} {d}: no output")
            if r.returncode != 0:
                print(r.stderr[-2000:])


# ---------------------------------------------------------------------------------------------------------------- run


def instrument_costs(inst: dict[str, Any], det: dict[str, Any] | None, price: float) -> X.Costs:
    cb = X.ctrader_commission_bp(det, price)
    return X.Costs(X.bybit_taker_bp(inst), cb if cb is not None and math.isfinite(cb) else 0.0, cb is not None)


def funding_events(sym: str) -> list[tuple[int, float]]:
    p = X.xdir() / "funding.parquet"
    if not p.exists():
        return []
    f = pl.read_parquet(p).filter(pl.col("symbol") == sym)
    return list(zip(f["ts_s"].to_list(), f["rate"].to_list(), strict=True))


def analyse(insts: list[dict[str, Any]], split: dict[str, Any], cmap: dict[str, Any], c: dict[str, Any], rtt_s: float, delay_s: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for inst in insts:
        s = inst["bybit"]
        m = cmap.get(s) or {}
        name, det = m.get("ctrader_name"), m.get("details")
        res: dict[str, Any] = {"ctrader_name": name, "days": {}, "trades": [], "hour_profile": [], "events": {}, "reopen": []}
        if not name:
            res["skipped"] = "not resolved on this account"
            out[s] = res
            continue
        corr_on = inst["class"] in c.get("usdt_correct_classes", [])
        hp = []
        for d in all_days(split, s):
            b = X.load_bybit(s, d)
            if b is None:
                res["days"][d] = {"missing": "bybit book"}
                continue
            t = b["t"].to_numpy()
            ct = X.load_ctrader(name, d, t)
            if ct is None:
                res["days"][d] = {"missing": "ctrader ticks"}
                continue
            sched = X.in_schedule(t, (det or {}).get("schedule"), (det or {}).get("schedule_time_zone"), (det or {}).get("holidays"))
            corr = X.usdt_corr(t) if corr_on else 0.0
            g = X.build_grid(b, ct, corr, float(inst.get("multiplier", 1.0)), float(c["live_age_s"]), sched)
            g_raw = X.build_grid(b, ct, 0.0, float(inst.get("multiplier", 1.0)), float(c["live_age_s"]), sched) if corr_on else g
            if not g["common"].any():
                res["days"][d] = {"common_s": 0}
                continue
            px = float(g.filter(pl.col("common"))["bid_c"].median())
            cost = instrument_costs(inst, det, px)
            is_reopen = d in split["instruments"][s]["reopen_sundays"]
            lo, sh = X.swap_bp_per_night(det, px)
            carry = X.Carry(funding=funding_events(s), swap_long_bp=lo, swap_short_bp=sh,
                            swap_time_s=int(((det or {}).get("swap_time_min_utc") or 22 * 60) * 60), triple_weekday=X.triple_weekday(det))
            day = {"reopen_only": is_reopen, "fees_rt_bp": cost.round_trip_bp, "ctrader_comm_known": cost.ctrader_comm_known,
                   "gap_median_bp": float(g.filter(pl.col("common"))["gap_bp"].median()),
                   "gap_raw_median_bp": float(g_raw.filter(pl.col("common"))["gap_bp"].median()),
                   "spr_b_median_bp": float(g.filter(pl.col("common"))["spr_b_bp"].median()), "spr_c_median_bp": float(g.filter(pl.col("common"))["spr_c_bp"].median())}
            day["arb1"] = X.arb1_day(g, cost.round_trip_bp, 2 * rtt_s)
            day["arb1_p90"] = X.arb1_day(g, cost.round_trip_bp, 2 * LATENCY["rtt_p90_s"])
            day["arb1_raw"] = X.arb1_day(g_raw, cost.round_trip_bp, 2 * rtt_s) if corr_on else None
            if not is_reopen:
                tr = X.arb2_day(g, cost.round_trip_bp, delay_s, carry)
                for x in tr:
                    x["date"] = d
                res["trades"] += tr
                tr90 = X.arb2_day(g, cost.round_trip_bp, math.ceil(LATENCY["rtt_p90_s"]), carry)
                day["arb2_p90_net"] = [x["net_bp"] for x in tr90]
                hp.append(X.hour_profile(g).with_columns(pl.lit(d).alias("date")))
                o, cl = X.us_open_close_utc(d)
                res["events"][d] = {"us_open": X.window_stats(g, o, 600, 1800, cost.round_trip_bp), "us_close": X.window_stats(g, cl, 1800, 600, cost.round_trip_bp)}
            dt = date.fromisoformat(d)
            if is_reopen or (dt.weekday() == 0 and inst["class"] in ("stock", "etf")):
                r = X.reopen_stats(g)
                if r:
                    res["reopen"].append({"date": d, **r})
            res["days"][d] = day
        if hp:
            h = pl.concat(hp).group_by("hour").agg(pl.col("gap_median_bp").median(), pl.col("spr_c_bp").median(), pl.col("spr_b_bp").median(), pl.col("n_s").sum()).sort("hour")
            res["hour_profile"] = h.to_dicts()
        sampled = [d for d in split["instruments"][s]["days"] if isinstance(res["days"].get(d), dict) and "arb1" in res["days"][d]]
        res["n_days"] = len(sampled)
        res["arb1_days_pass"] = sum(res["days"][d]["arb1"]["pass"] for d in sampled)
        res["arb1_days_pass_p90"] = sum(res["days"][d]["arb1_p90"]["pass"] for d in sampled)
        res["arb1_pass"] = res["arb1_days_pass"] >= 3
        nets = np.array([x["net_bp"] for x in res["trades"]])
        res["arb2"] = X.cluster_boot_mean(nets, np.array([x["date"] for x in res["trades"]])) if len(nets) else {"n": 0}
        out[s] = res
    return out


def verdicts(out: dict[str, Any]) -> dict[str, Any]:
    tested = {s: r for s, r in out.items() if r.get("n_days", 0) >= 3}
    passing = [s for s, r in tested.items() if r["arb1_pass"]]
    if not tested:
        v1 = "inconclusive"
    else:
        v1 = "supports" if passing else ("contradicts" if all(r["n_days"] >= 5 or r["arb1_days_pass"] + (5 - r["n_days"]) < 3 for r in tested.values()) else "inconclusive")
    trades = [(f"{s}|{x['date']}", x["net_bp"]) for s, r in tested.items() for x in r["trades"]]
    pooled = X.cluster_boot_mean(np.array([n for _, n in trades]), np.array([k for k, _ in trades])) if trades else {"n": 0}
    if pooled.get("n", 0) < 30:
        v2 = "inconclusive"
    elif pooled["lo"] > 0:
        v2 = "supports"
    elif pooled["mean"] <= 0 or pooled["hi"] < 0:
        v2 = "contradicts"
    else:
        v2 = "inconclusive"
    return {"H-ARB1": {"verdict": v1, "instruments_tested": len(tested), "instruments_passing": passing},
            "H-ARB2": {"verdict": v2, "pooled": pooled, "per_instrument": {s: r["arb2"] for s, r in tested.items()}}}


def fmt(x: Any, nd: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def rule_candidates(out: dict[str, Any], v: dict[str, Any]) -> list[dict[str, Any]]:
    rep = f"research/reports/xvenue_{TAG}.md"
    rules = []
    v1, v2 = v["H-ARB1"]["verdict"], v["H-ARB2"]["verdict"]
    pooled = v["H-ARB2"]["pooled"]
    if v1 == "contradicts" or v2 == "contradicts":
        rules.append({"id": "R-ARB-1", "kind": "avoid",
                      "rule": "WHEN a Bybit perp and the BlackBull CFD on the same underlying show a price difference THEN do not treat it as an arbitrage or a signal",
                      "when": "cross-venue gap between a Bybit linear perp and a BlackBull cTrader CFD",
                      "then": "no convergence trades; read the gap as structural basis plus quote noise",
                      "scope": {"assets": sorted(out), "horizon": "seconds to 60 minutes", "sides": "both"},
                      "effect": f"H-ARB1 {v1} ({len(v['H-ARB1']['instruments_passing'])} of {v['H-ARB1']['instruments_tested']} instruments pass); "
                                f"H-ARB2 {v2}: mean net {fmt(pooled.get('mean'))} bp per trade, 95% CI {fmt(pooled.get('lo'))}..{fmt(pooled.get('hi'))}, n {pooled.get('n')}",
                      "confidence": "C", "evidence": ["H-ARB1", "H-ARB2", rep], "review": "a forward month of live spots (atx ctrader stream) with episodes that beat costs", "status": "provisional"})
    if v1 == "supports":
        rules.append({"id": "R-ARB-2", "kind": "expect",
                      "rule": "WHEN both venues trade and the touch-to-touch gap between a Bybit perp and its BlackBull CFD exceeds the round-trip fees THEN expect it to last longer than our order round trip",
                      "when": "executable cross-venue gap > 0 after fees", "then": "the gap is reachable at our latency; whether it converges is H-ARB2",
                      "scope": {"assets": v["H-ARB1"]["instruments_passing"], "horizon": "seconds", "sides": "both"},
                      "effect": "; ".join(f"{s}: {out[s]['arb1_days_pass']}/{out[s]['n_days']} days" for s in v["H-ARB1"]["instruments_passing"]),
                      "confidence": "C", "evidence": ["H-ARB1", rep], "review": "fails on 5 fresh days or under the p90 latency", "status": "provisional"})
    basis = {s: float(np.median([d["gap_median_bp"] for d in r["days"].values() if isinstance(d, dict) and "gap_median_bp" in d])) for s, r in out.items()
             if any(isinstance(d, dict) and "gap_median_bp" in d for d in r["days"].values())}
    if basis:
        rules.append({"id": "R-ARB-3", "kind": "expect",
                      "rule": "WHEN comparing a Bybit perp price with the BlackBull CFD quote THEN expect a persistent instrument-specific offset (structural basis), not a mispricing",
                      "when": "reading a Bybit perp against a CFD reference price", "then": "subtract the instrument's median basis before calling a level rich or cheap",
                      "scope": {"assets": sorted(basis), "horizon": "intraday", "sides": "both"},
                      "effect": ", ".join(f"{s} {b:+.1f} bp" for s, b in sorted(basis.items())),
                      "confidence": "C", "evidence": ["H-ARB1", rep], "review": "a basis that changes sign across months", "status": "provisional"})
    return rules


def write_report(out: dict[str, Any], v: dict[str, Any], split: dict[str, Any], rules: list[dict[str, Any]], errs: list[str]) -> Path:
    js = ROOT / "research" / "reports" / f"xvenue_{TAG}.json"
    md = ROOT / "research" / "reports" / f"xvenue_{TAG}.md"
    slim = {s: {k: r[k] for k in r if k != "trades"} | {"n_trades": len(r["trades"])} for s, r in out.items()}
    js.write_text(json.dumps({"split": split["meta"], "verdicts": v, "instruments": slim, "rules_validation": errs}, indent=1, default=float), encoding="utf-8")
    L = [f"# Bybit perps vs BlackBull CFDs: are there tradeable price gaps? ({TAG})", "",
         "Hypotheses H-ARB1 (executable gaps outlast our latency) and H-ARB2 (a convergence trade beats costs). Segment: clean_holdout "
         "(all days after the 2026-07-01 cutoff). Days were drawn with a fixed seed before any cTrader data was read "
         f"({SPLIT.relative_to(ROOT).as_posix()}). No orders were sent on either venue.", "", "## Answer in plain words", ""]
    v1, v2 = v["H-ARB1"], v["H-ARB2"]
    L.append(f"- H-ARB1: **{v1['verdict']}**. {len(v1['instruments_passing'])} of {v1['instruments_tested']} instruments had executable gaps (after both "
             f"half-spreads and round-trip fees) on >= 0.5% of common seconds in episodes longer than 2 x {LATENCY['rtt_p50_s']} s on >= 3 of 5 days"
             + (f": {', '.join(v1['instruments_passing'])}." if v1["instruments_passing"] else "."))
    p = v2["pooled"]
    L.append(f"- H-ARB2: **{v2['verdict']}**. Pooled convergence trades: mean net {fmt(p.get('mean'))} bp per trade (95% day-clustered CI "
             f"{fmt(p.get('lo'))} to {fmt(p.get('hi'))}), {p.get('n', 0)} trades.")
    L += ["", "## Per instrument", "", "| Bybit | cTrader | days | H-ARB1 days pass (p50 / p90 latency) | median basis bp | spread Bybit / CFD bp | fees rt bp | trades | mean net bp [CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, r in out.items():
        ds = [d for d in r["days"].values() if isinstance(d, dict) and "arb1" in d and not d.get("reopen_only")]
        med = lambda k: float(np.median([d[k] for d in ds])) if ds else None  # noqa: E731
        a2 = r.get("arb2", {})
        L.append(f"| {s} | {r.get('ctrader_name') or '-'} | {r.get('n_days', 0)} | {r.get('arb1_days_pass', 0)} / {r.get('arb1_days_pass_p90', 0)} | {fmt(med('gap_median_bp'))} | "
                 f"{fmt(med('spr_b_median_bp'))} / {fmt(med('spr_c_median_bp'))} | {fmt(med('fees_rt_bp'))} | {a2.get('n', 0)} | {fmt(a2.get('mean'))} [{fmt(a2.get('lo'))}, {fmt(a2.get('hi'))}] |")
    L += ["", "## Structural basis by hour of day (UTC, median mid gap in bp; descriptive)", ""]
    for s, r in out.items():
        if r["hour_profile"]:
            L.append(f"- {s}: " + ", ".join(f"{h['hour']:02d}h {h['gap_median_bp']:+.1f}" for h in r["hour_profile"]))
    L += ["", "## US open and close; weekend reopen (descriptive)", ""]
    for s, r in out.items():
        ev = [(d, e) for d, e in r["events"].items() if e and (e.get("us_open") or e.get("us_close"))]
        if ev:
            uo = [e["us_open"]["exec_share"] for _, e in ev if e.get("us_open")]
            uc = [e["us_close"]["exec_share"] for _, e in ev if e.get("us_close")]
            L.append(f"- {s}: executable share around the US open {fmt(float(np.mean(uo)) * 100 if uo else None, 2)}%, close {fmt(float(np.mean(uc)) * 100 if uc else None, 2)}%")
        for x in r["reopen"]:
            L.append(f"  - reopen {x['date']} ({x['reopen_utc']} UTC): gap at reopen {fmt(x['gap_at_reopen_bp'])} bp, first 5 min {fmt(x['gap_first5m_median_bp'])}, 1-2 h later {fmt(x['gap_1to2h_median_bp'])}")
    L += ["", "## Method", "", "```text", "Grid." + X.__doc__.split("Grid.")[1].rstrip() if X.__doc__ else "", "```", "",
          f"Latency: {LATENCY['source']}; RTT p50 {LATENCY['rtt_p50_s']} s (fill delay {math.ceil(LATENCY['rtt_p50_s'])} s), sensitivity p90 {LATENCY['rtt_p90_s']} s.",
          "Context: Bybit TradFi perps trade 24/7 and track their index within ~5 bp (BELIEFS T1); weekend direction is uninformative (P5). "
          "A CFD that is closed while the perp trades is not an arbitrage: there is no second leg until the CFD reopens.", "",
          "## Rule candidates", "", "```yaml", yaml.safe_dump({"rules": rules}, sort_keys=False, allow_unicode=True, width=200).strip(), "```", ""]
    md.write_text("\n".join(L), encoding="utf-8")
    return md


def cmd_run(a: argparse.Namespace) -> None:
    from ai_trader.research import diary, hypotheses, rules as R

    c, split, cmap = X.cfg(), load_split(), ctrader_map()
    insts = [i for i in X.study_instruments(c) if i["bybit"] in split["instruments"]]
    out = analyse(insts, split, cmap, c, LATENCY["rtt_p50_s"], math.ceil(LATENCY["rtt_p50_s"]))
    v = verdicts(out)
    print(json.dumps(v, indent=1, default=float))
    if not a.no_record and v["H-ARB1"]["instruments_tested"]:
        for hid in ("H-ARB1", "H-ARB2"):
            hypotheses.record(hid, "clean_holdout", v[hid]["verdict"], {k: x for k, x in v[hid].items() if k != "verdict"}, data=f"research/reports/xvenue_{TAG}.json",
                              note="Bybit ob200 1 s books vs BlackBull cTrader ticks; 5 seeded days per instrument; fees + p50 latency")
    rules = rule_candidates(out, v)
    errs = R.validate(rules) if not a.no_record else ["not validated (--no-record: verdicts not recorded)"]
    md = write_report(out, v, split, rules, errs)
    print("->", md, "| rule validation:", errs or "ok")
    if not a.no_record and v["H-ARB1"]["instruments_tested"]:
        diary.entry("result", "Cross-venue Bybit vs BlackBull study", f"H-ARB1 {v['H-ARB1']['verdict']}, H-ARB2 {v['H-ARB2']['verdict']} "
                    f"(pooled mean net {fmt(v['H-ARB2']['pooled'].get('mean'))} bp, n {v['H-ARB2']['pooled'].get('n', 0)}). See the report.",
                    tags=["xvenue", "result"], hypotheses=["H-ARB1", "H-ARB2"], links=[md.relative_to(ROOT).as_posix()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("split")
    b = sp.add_parser("bybit")
    b.add_argument("--workers", type=int, default=2)
    sp.add_parser("ctrader")
    r = sp.add_parser("run")
    r.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    {"split": cmd_split, "bybit": cmd_bybit, "ctrader": cmd_ctrader, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":

    main()
