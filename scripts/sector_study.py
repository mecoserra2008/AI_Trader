"""Crypto sectors: taxonomy, sector board with within-sector outliers, and the registered panel tests H-SEC1-3, H-PAN1-2.

    PYTHONIOENCODING=utf-8 uv run python scripts/sector_study.py [--record] [--refresh-sectors]

Universe: Bybit linear crypto perps (symbolType ''), Trading, listed >= 180 days, 24h turnover >= 1M USD (survivorship-
biased: today's listings only). Daily closes / turnover from Bybit REST (up to 1000 days). Sectors: CoinGecko public
categories, one primary sector per token by priority (meme > ai > gaming > rwa > privacy > oracle > infra > exchange > l2 >
defi > l1 > other), cached in config/sectors.yaml. Weekly formations on Mondays (closes through Sunday); costs 30 bp per leg
round trip per rebalance. Fama-MacBeth t with Newey-West lags. Segments by formation date vs the knowledge cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ai_trader.core.config import load

ROOT = load().root
SECTORS_FILE = ROOT / "config" / "sectors.yaml"
CUTOFF = date(2026, 7, 1)
COST_LEG = 0.0030  # round trip per leg per rebalance (taker 5.5 + slippage ~2 bp per side, rounded up)
PRIORITY = [("meme", "meme-token"), ("ai", "artificial-intelligence"), ("gaming", "gaming"), ("rwa", "real-world-assets-rwa"),
            ("privacy", "privacy-coins"), ("oracle", "oracle"), ("infra", "depin"), ("infra", "storage"), ("exchange", "exchange-based-tokens"),
            ("l2", "layer-2"), ("defi", "decentralized-finance-defi"), ("l1", "layer-1"), ("l1", "smart-contract-platform")]


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def base_of(sym: str) -> str:
    b = sym.removesuffix("USDT")
    for p in ("1000000", "100000", "10000", "1000"):
        if b.startswith(p):
            b = b[len(p):]
    if b.endswith("1000"):
        b = b[:-4]
    return b.lower()


def universe() -> list[str]:
    inst = get("https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000")["result"]["list"]
    tick = {x["symbol"]: float(x["turnover24h"]) for x in get("https://api.bybit.com/v5/market/tickers?category=linear")["result"]["list"]}
    now_ms = time.time() * 1000
    return sorted(x["symbol"] for x in inst if x.get("symbolType", "") == "" and x["status"] == "Trading" and x["quoteCoin"] == "USDT"
                  and now_ms - int(x["launchTime"]) > 180 * 86400_000 and tick.get(x["symbol"], 0) >= 1e6)


def sectors(symbols: list[str], refresh: bool) -> dict[str, str]:
    cache = yaml.safe_load(SECTORS_FILE.read_text(encoding="utf-8")) if SECTORS_FILE.exists() else {}
    m = dict(cache.get("sectors", {}))
    missing = [s for s in symbols if s not in m]
    if missing or refresh:
        cat_syms: dict[str, list[tuple[str, float]]] = {}
        for _, cid in PRIORITY:
            if cid in cat_syms:
                continue
            rows = []
            for page in (1, 2):
                try:
                    rows += get(f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category={cid}&per_page=250&page={page}")
                except Exception as e:  # noqa: BLE001
                    print("coingecko", cid, page, e)
                time.sleep(7)
            cat_syms[cid] = [(r["symbol"].lower(), float(r.get("market_cap") or 0)) for r in rows]
        for s in symbols if refresh else missing:
            b = base_of(s)
            sec = "other"
            for name, cid in PRIORITY:
                if any(sym == b for sym, _ in cat_syms.get(cid, [])):
                    sec = name
                    break
            m[s] = sec
        keep = {k: cache[k] for k in ("overrides_note", "overrides") if k in cache}
        SECTORS_FILE.write_text(yaml.safe_dump({"source": "CoinGecko public categories, primary sector by priority (scripts/sector_study.py)",
                                                "priority": [p[0] for p in PRIORITY], "updated": date.today().isoformat(), "sectors": dict(sorted(m.items())), **keep},
                                               sort_keys=False), encoding="utf-8")
    ov = (yaml.safe_load(SECTORS_FILE.read_text(encoding="utf-8")) or {}).get("overrides", {})
    return {s: ov.get(s, m.get(s, "other")) for s in symbols}


def daily(sym: str) -> pl.DataFrame:
    r = get(f"https://api.bybit.com/v5/market/kline?category=linear&symbol={sym}&interval=D&limit=1000")["result"]["list"]
    return pl.DataFrame({"d": [datetime.fromtimestamp(int(x[0]) / 1000, UTC).date() for x in r], "close": [float(x[4]) for x in r], "turnover": [float(x[6]) for x in r]}).sort("d")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return np.nan
    ra, rb = pl.Series(a).rank().to_numpy(), pl.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def fm(ics: list[tuple[date, float]], lags: int) -> dict:
    v = np.array([x for _, x in ics if np.isfinite(x)])
    if len(v) < 8:
        return {"n": len(v)}
    n, mu = len(v), v.mean()
    e = v - mu
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return {"n": n, "mean_ic": float(mu), "t": float(mu / math.sqrt(s / n)), "share_pos": float((v > 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--refresh-sectors", action="store_true")
    a = ap.parse_args()
    syms = universe()
    print(f"universe: {len(syms)} perps")
    sec = sectors(syms, a.refresh_sectors)
    frames = []
    for s in syms + (["BTCUSDT"] if "BTCUSDT" not in syms else []):
        try:
            frames.append(daily(s).with_columns(pl.lit(s).alias("sym")))
        except Exception as e:  # noqa: BLE001
            print("kline", s, e)
        time.sleep(0.12)
    P = pl.concat(frames).filter(pl.col("close") > 0).sort(["sym", "d"])
    P = P.with_columns(pl.col("close").log().diff().over("sym").alias("r"))
    # wide matrices
    R = P.pivot(on="sym", index="d", values="r").sort("d")
    C = P.pivot(on="sym", index="d", values="close").sort("d")
    T = P.pivot(on="sym", index="d", values="turnover").sort("d")
    days = R["d"].to_list()
    idx = {d: i for i, d in enumerate(days)}
    cols = [c for c in R.columns if c != "d"]
    r = R.select(cols).to_numpy()
    lc = np.log(C.select(cols).to_numpy())
    tv = T.select(cols).to_numpy()
    btc = cols.index("BTCUSDT")
    sec_of = np.array(["btc" if c == "BTCUSDT" else sec.get(c, "other") for c in cols])
    EXCL = {"btc", "gold", "other"}

    def ret(i0, i1):  # log return between day indexes (close to close)
        return lc[i1] - lc[i0]

    mondays = [i for i, d in enumerate(days) if d.isoweekday() == 1 and i >= 120 and i + 7 < len(days)]
    res = {"universe_n": len(syms), "sector_counts": {k: int((sec_of == k).sum()) for k in sorted(set(sec_of))}}
    ic = {"SEC1": [], "SEC2": [], "PAN1": [], "PAN2": []}
    ls = {"SEC2": [], "PAN1": []}
    sec3 = []
    for i in mondays:
        f0, f7 = i - 1, i + 6  # formation uses closes through Sunday (index i-1); forward = next 7 closes
        past7, fwd7 = ret(f0 - 7, f0), ret(f0, f7)
        ok = np.isfinite(past7) & np.isfinite(fwd7) & ~np.isin(sec_of, list(EXCL))
        # sector EW returns
        sp, sf = [], []
        for k in sorted(set(sec_of[ok])):
            mk = ok & (sec_of == k)
            if k not in EXCL and mk.sum() >= 3:
                sp.append(np.nanmean(past7[mk]))
                sf.append(np.nanmean(fwd7[mk]))
        if len(sp) >= 6:
            ic["SEC1"].append((days[i], spearman(np.array(sp), np.array(sf))))
        # within-sector relative returns
        relp, relf, vol_spike = np.full(len(cols), np.nan), np.full(len(cols), np.nan), np.full(len(cols), np.nan)
        for k in set(sec_of[ok]):
            mk = ok & (sec_of == k)
            if k not in EXCL and mk.sum() >= 3:
                relp[mk] = past7[mk] - np.nanmean(past7[mk])
                relf[mk] = fwd7[mk] - np.nanmean(fwd7[mk])
        t7 = np.nanmean(tv[f0 - 6:f0 + 1], axis=0)
        t30 = np.nanmean(tv[f0 - 29:f0 + 1], axis=0)
        vol_spike = t7 / t30
        m2 = np.isfinite(relp) & np.isfinite(relf)
        if m2.sum() >= 15:
            ic["SEC2"].append((days[i], spearman(relp[m2], relf[m2])))
            q1, q3 = np.nanpercentile(relp[m2], [33.3, 66.7])
            lo, hi = m2 & (relp <= q1), m2 & (relp >= q3)
            ls["SEC2"].append((days[i], float(np.nanmean(relf[lo]) - np.nanmean(relf[hi]) - 2 * COST_LEG)))  # reversal: long losers, short winners
            thr = np.nanpercentile(np.abs(relp[m2]), 90)
            out = m2 & (np.abs(relp) >= thr)
            for j in np.where(out)[0]:
                sec3.append({"d": days[i], "signed_fwd": float(np.sign(relp[j]) * relf[j]), "spike": bool(vol_spike[j] >= 2)})
        # monthly residual momentum / ivol (every 4th Monday)
        if mondays.index(i) % 4 == 0 and i + 29 < len(days) and i >= 150:
            g0, g30 = f0, i + 29
            rb = r[g0 - 89:g0 + 1, btc]
            out30 = []
            for j, c in enumerate(cols):
                if j == btc or sec_of[j] == "gold":
                    continue
                rj = r[g0 - 89:g0 + 1, j]
                mm = np.isfinite(rj) & np.isfinite(rb)
                if mm.sum() < 60:
                    continue
                beta = np.cov(rj[mm], rb[mm])[0, 1] / np.var(rb[mm])
                resid_d = rj[mm] - beta * rb[mm]
                past = ret(g0 - 30, g0)[j] - beta * ret(g0 - 30, g0)[btc]
                fwd = ret(g0, g30)[j] - beta * ret(g0, g30)[btc]
                ivol = np.std(resid_d[-60:])
                if np.isfinite(past) and np.isfinite(fwd) and ivol > 0:
                    out30.append((past, fwd, ivol))
            if len(out30) >= 20:
                A = np.array(out30)
                ic["PAN1"].append((days[i], spearman(A[:, 0], A[:, 1])))
                q1, q3 = np.percentile(A[:, 0], [33.3, 66.7])
                ls["PAN1"].append((days[i], float(A[A[:, 0] >= q3, 1].mean() - A[A[:, 0] <= q1, 1].mean() - 2 * COST_LEG)))
                ic["PAN2"].append((days[i], spearman(A[:, 2], A[:, 1] / A[:, 2])))
    segs = {"llm_in_sample": lambda d: d < CUTOFF - timedelta(days=30), "clean_holdout": lambda d: d >= CUTOFF}
    lags = {"SEC1": 4, "SEC2": 4, "PAN1": 2, "PAN2": 2}
    res["tests"] = {}
    for k, v in ic.items():
        for sname, f in segs.items():
            res["tests"].setdefault(k, {})[sname] = fm([x for x in v if f(x[0])], lags[k])
    for k, v in ls.items():
        for sname, f in segs.items():
            vv = np.array([x for d, x in v if f(d)])
            res["tests"][k][sname]["net_ls_mean"] = float(vv.mean()) if len(vv) else None
            res["tests"][k][sname]["net_ls_t"] = float(vv.mean() / (vv.std() / math.sqrt(len(vv)))) if len(vv) > 3 and vv.std() > 0 else None
    S3 = pl.DataFrame(sec3) if sec3 else pl.DataFrame()
    res["tests"]["SEC3"] = {}
    for sname, f in segs.items():
        q = (S3.filter(pl.col("d") < CUTOFF - timedelta(days=30)) if sname == "llm_in_sample" else S3.filter(pl.col("d") >= CUTOFF)) if S3.height else S3
        if q.height > 20:
            a1, a0 = q.filter(pl.col("spike"))["signed_fwd"].to_numpy(), q.filter(~pl.col("spike"))["signed_fwd"].to_numpy()
            # cluster by week: weekly means of the difference
            wk = q.group_by(["d", "spike"]).agg(pl.col("signed_fwd").mean()).pivot(on="spike", index="d", values="signed_fwd").drop_nulls()
            dif = (wk["true"] - wk["false"]).to_numpy() if {"true", "false"} <= set(wk.columns) else np.array([])
            res["tests"]["SEC3"][sname] = {"n_spike": len(a1), "n_nospike": len(a0), "mean_spike": float(a1.mean()) if len(a1) else None, "mean_nospike": float(a0.mean()) if len(a0) else None,
                                           "diff_weekly_mean": float(dif.mean()) if len(dif) else None, "t": float(dif.mean() / (dif.std() / math.sqrt(len(dif)))) if len(dif) > 3 and dif.std() > 0 else None, "weeks": len(dif)}
    # ---- current board (as of the last full day)
    L = len(days) - 1
    while L > 0 and np.isnan(lc[L, btc]):
        L -= 1
    board = []
    r1, r7, r30 = ret(L - 1, L), ret(L - 7, L), ret(L - 30, L)
    sp7 = np.nanmean(tv[L - 6:L + 1], axis=0) / np.nanmean(tv[L - 29:L + 1], axis=0)
    for k in sorted(set(sec_of)):
        if k in ("btc", "gold"):
            continue
        mk = (sec_of == k) & np.isfinite(r30)
        if mk.sum() == 0:
            continue
        board.append({"sector": k, "n": int(mk.sum()), "ret_1d_%": 100 * float(np.nanmean(np.expm1(r1[mk]))), "ret_7d_%": 100 * float(np.nanmean(np.expm1(r7[mk]))),
                      "ret_30d_%": 100 * float(np.nanmean(np.expm1(r30[mk]))), "disp_30d_%": 100 * float(np.nanstd(np.expm1(r30[mk])))})
    res["board_as_of"] = str(days[L])
    res["btc_7d_%"], res["btc_30d_%"] = 100 * float(np.expm1(r7[btc])), 100 * float(np.expm1(r30[btc]))
    res["board"] = sorted(board, key=lambda x: x["ret_7d_%"])
    outl = []
    for k in set(sec_of):
        mk = (sec_of == k) & np.isfinite(r7) & np.isfinite(r30)
        if k in EXCL or mk.sum() < 4:
            continue
        for w, rr in (("7d", r7), ("30d", r30)):
            rel = rr[mk] - np.nanmean(rr[mk])
            sd = np.nanstd(rel)
            for j, z in zip(np.where(mk)[0], rel / sd if sd > 0 else rel * 0):
                if abs(z) >= 1.8:
                    outl.append({"sym": cols[j], "sector": k, "window": w, "rel_z": float(z), "rel_ret_%": 100 * float(rr[j] - np.nanmean(rr[mk])), "ret_%": 100 * float(np.expm1(rr[j])), "vol_spike_7d_30d": float(sp7[j])})
    res["outliers"] = sorted(outl, key=lambda x: -abs(x["rel_z"]))
    out = ROOT / "research" / "reports" / "sector_study_2026-11-08.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    pl.Config.set_tbl_rows(60), pl.Config.set_tbl_cols(12), pl.Config.set_tbl_width_chars(200), pl.Config.set_float_precision(1)
    print("sector counts:", res["sector_counts"])
    print(f"BTC 7d {res['btc_7d_%']:+.1f}%  30d {res['btc_30d_%']:+.1f}%   board as of {res['board_as_of']}")
    print(pl.DataFrame(res["board"]))
    print(pl.DataFrame(res["outliers"]).head(30) if res["outliers"] else "no outliers")
    print(json.dumps(res["tests"], indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else str(x)))
    if a.record:
        from ai_trader.research.hypotheses import record
        T = res["tests"]
        rules = {
            "H-SEC1": ("SEC1", lambda s, h: s.get("mean_ic", 0) > 0 and s.get("t", 0) > 2, lambda s, h: s.get("mean_ic", -1) > 0),
            "H-SEC2": ("SEC2", lambda s, h: s.get("mean_ic", 0) < 0 and s.get("t", 0) < -2 and (s.get("net_ls_mean") or -1) > 0, lambda s, h: s.get("mean_ic", 1) < 0),
            "H-PAN1": ("PAN1", lambda s, h: s.get("mean_ic", 0) > 0 and s.get("t", 0) > 2 and (s.get("net_ls_mean") or -1) > 0, lambda s, h: s.get("mean_ic", -1) > 0),
            "H-PAN2": ("PAN2", lambda s, h: s.get("mean_ic", 0) < 0 and s.get("t", 0) < -2, lambda s, h: s.get("mean_ic", 1) < 0),
        }
        for hid, (k, rin, rho) in rules.items():
            for seg, rule in (("llm_in_sample", rin), ("clean_holdout", rho)):
                s = T[k].get(seg, {})
                verdict = "inconclusive" if s.get("n", 0) < 8 else ("supports" if rule(s, None) else "contradicts")
                record(hid, seg, verdict, s, data="Bybit daily perps (survivorship-biased universe), CoinGecko sectors; scripts/sector_study.py")
        for seg in ("llm_in_sample", "clean_holdout"):
            s = T["SEC3"].get(seg, {})
            ok = (s.get("diff_weekly_mean") or -1) > 0 and ((s.get("t") or 0) > 2 if seg == "llm_in_sample" else True)
            record("H-SEC3", seg, "inconclusive" if not s else ("supports" if ok else "contradicts"), s, data="scripts/sector_study.py")
        print("recorded")


if __name__ == "__main__":
    main()
