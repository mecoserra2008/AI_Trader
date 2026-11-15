"""Multi-day microstructure tests H-MD1..H-MD6 (research/hypotheses.yaml, block "Overnight program 2026-11-12"; agent A).

Input: every verified day under data/research/micro_days/<date>/ (ai_trader.research.micro_days): laws/<SYM>.json
(per symbol-day laws + held-out tests + activity), day_stats.json (HY, DY, spread -> vol gains, book stats, activity
rows), book_1s / trades_1s / orders (H-MD6 rebuilds 3 s grids and calibrates the simulators from them).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.multiday_micro_study --batch 1 [--no-md6] [--no-record]

Segments: a day before research.knowledge_cutoff (2026-07-01) is llm_in_sample, else clean_holdout; pooled = all days.
Choices fixed before the multi-day data were looked at (the registry leaves them open) are in METHOD.

Recording rule (one row per hypothesis x segment across all batches): a verdict is recorded only when it can no longer
change with the remaining pre-drawn days of research/plans/2026-11-12_overnight_loop.md (a share criterion that is
already lost even if every remaining unit passes -> 'contradicts'), or after the last batch (all planned days
processed: every verdict, Holm over the six). Early 'supports' are never recorded (Holm needs the whole block).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.research import micro_days as MD

ROOT = MD.ROOT
OUT_ROOT = MD.OUT_ROOT
REPORT_DIR = ROOT / "research" / "reports"
SEED = 20261112
B_ICC = 2000
B_LOO = 300
B_DAY = 2000
B_ENV = 300
N_PATHS = 10
WORKERS = 4
MIN_PRINTS_MD3 = 20_000
HIDS = ("H-MD1", "H-MD2", "H-MD3", "H-MD4", "H-MD5", "H-MD6")
PLANNED_DAYS = ("2025-09-05", "2025-12-18", "2026-07-12", "2026-07-13", "2026-09-23",
                "2025-08-31", "2025-11-19", "2026-07-22", "2026-08-17", "2026-10-25",
                "2026-05-11", "2026-06-21", "2026-07-03", "2026-08-16", "2026-10-02",
                "2026-05-20", "2026-06-14", "2026-07-07", "2026-07-18", "2026-10-09",
                "2025-11-21", "2026-01-25", "2026-07-17", "2026-08-04", "2026-08-18",
                "2025-09-02", "2025-09-10", "2026-10-16", "2026-11-02", "2026-11-08", MD.PILOT_DATE)
SIM_SYMBOLS = MD.L2_SYMBOLS
FACT_KEYS = ("kurtosis", "absr_acf", "flow_acf", "impact", "spread_med", "spread_p90", "continuation")

METHOD = {
    "segments": "day < 2026-07-01 -> llm_in_sample, else clean_holdout; pooled = all days; the 2026-11-10 pilot day is clean_holdout",
    "H-MD1": "per symbol-day TEST-half (12-24 UTC) estimates from laws/<SYM>.json tests: order-sign ACF power-law gamma, order-sign lag-1 ACF, "
             "print-size CSN tail index zeta, event-duration expmix3 weights (components ordered fastest first); ICC(1) one-way random "
             "effects with symbol as the group (unbalanced n0), symbols with >= 2 days; mixture trait = mean of the three weight ICCs; 95% CI "
             f"by a bootstrap over symbols (B = {B_ICC}); a trait passes if ICC > 0.5 and CI lower > 0.3; supports = >= 3 of 4 traits; "
             "p = 3rd smallest of the trait p's (P*(ICC <= 0.5)); robustness: FIT half, symbol-days with >= 20k prints",
    "H-MD2": "FIT-half activity rows (research.population_simulator_study._half_measures): W = sigma_day x P V_day; rate = log orders/s, impact = "
             "log Kyle lambda (mid proxy, bp per USD 1k; rows with lambda <= 0 dropped), depth = log median 25 bp book depth (L2 symbols, "
             "symbol-days whose 25 bp band is truncated in > 50% of the 1 s states dropped; descriptive); model y = 1 + log W + alt + tradfi "
             "+ day FE; LOO R^2 leaving out all days of one symbol; bootstrap over symbols (B = 300) for P*(LOO R^2 <= 0.5); interaction "
             "model adds day x log W, Wald chi2 of the interactions with CR1 covariance clustered by symbol; supports = LOO R^2 > 0.5 for "
             "rate and impact and Wald p > 0.05 for both; p = max of the two LOO bootstrap p's",
    "H-MD3": "symbol-days with >= 20k prints: order-sign ACF power law preferred on held-out TEST lags (fit on FIT, as H-MS3) and 0 < gamma_TEST < 1; "
             "share vs 0.8 (one-sided binomial); pooled Hurst = mean over the same symbol-days of the DFA-1 Hurst of 1 s signed notional "
             f"(TEST half), 95% CI by a bootstrap over days (B = {B_DAY}); supports = share >= 0.8 and CI lower > 0.5; p = max(binomial p, "
             "P*(mean H <= 0.5))",
    "H-MD4": "L2 symbol-days whose FIT-half 1 s states have spread > 1.5 ticks in >= 25% (qualifying, known before TEST); gain = OOS R^2 "
             "(TEST) of the FIT-estimated per-symbol regression with log spread minus without (micro_days.METHOD spread_vol), h = 5 and 15 "
             "min; supports = >= 10 qualifying symbol-days and gain > 0 on >= 70% of them at both horizons; p = max over h of the one-sided "
             "binomial p vs 0.7 (sign test vs 0.5 reported)",
    "H-MD5": "daily Diebold-Yilmaz net directional connectedness (full UTC day, primary; halves reported) of log 1 m RV for the fixed H-MS6 "
             "token set; supports = ETH net > 0 and SOL net > 0 on >= 80% of days; p = max of the binomial p's vs 0.8; BTC vs ETH: mean of "
             "net_BTC - net_ETH over days with a day bootstrap CI (reported, not in the success line)",
    "H-MD6": "L2 symbols; real 3 s grid from book_1s (mid at candle close, spread = mean of the 3 one-second states, 25 bp depth at close) and "
             "trades_1s (buy / sell notional, prints, high / low); facts as H-SIM2 / H-MS8 on the TEST half, envelope = [2.5, 97.5]% of the "
             f"class median under a 1 h block bootstrap (B = {B_ENV}); v1 = flow_sim_v1 built and calibrated on the FIT half exactly as "
             "population_simulator_study.stage_symbol_one (laws = that day's FIT laws, spread / tick regime / shape targets from the FIT-half "
             "grid instead of pre-TEST flow_3s days, trade statistics from the order stream); v0 = flow_sim.FlowSim.fit on the FIT-half grid; "
             f"{N_PATHS} paths each of the TEST ok-candle count; class-day = (day, class) with >= 2 symbols; count = facts inside "
             "(kurtosis, |r| ACF, flow ACF, impact, spread median AND p90); supports = v1 > v0 on >= 70% of class-days and majors v1 >= 4/5 "
             "on >= 50% of days; p = max of the binomial p's vs 0.7 and 0.5",
    "holm": "Holm over the six MD summary p's (only at the final batch); supports needs the criterion and Holm p < 0.05",
    "recording": "early record only for a share criterion already lost given the remaining planned days ('contradicts'); everything else "
                 "after the last batch",
}

AGGREGATE_SCHEMA = {
    "layout": "data/research/micro_days/<YYYY-MM-DD>/{trades_1s,book_1s,orders}/<SYMBOL>.parquet, laws/<SYMBOL>.json, sim/<SYMBOL>.json, "
              "day_stats.json, qa.json; a day is usable only if qa.json has verified.ok = true (micro_days.processed_days()); "
              "FIT = 00:00-12:00 UTC, TEST = 12:00-24:00 UTC; segment = llm_in_sample if date < 2026-07-01 else clean_holdout",
    "trades_1s": "one row per UTC second WITH at least one print (sparse: missing seconds had no trade; reindex to 86400 rows yourself). "
                 "ts Int64 = second OPEN in epoch ms (the bar covers [ts, ts + 1000)); open/high/low/close Float64 = first/max/min/last print "
                 "price; vwap Float64; buy_vol/sell_vol Float64 = taker-buy / taker-sell base quantity; buy_notional/sell_notional Float64 = USD "
                 "(price x size); n_trades/n_buy/n_sell Int32 = prints; n_orders/n_buy_orders/n_sell_orders Int32 = taker orders (consecutive "
                 "same-ms same-side prints merged, counted in the second of the order's first print). Signed flow = buy_notional - sell_notional. "
                 "Point in time: a bar is known at ts + 1000",
    "book_1s": "L2 symbols only (BTC ETH SOL XRP DOGE HYPE CARV KAS ATOM NVDA QQQ XAU; NVDA/QQQ/XAU absent before their listing); one row per "
               "second, dense from the first message (forward-filled when no update). ts Datetime(ms, UTC) = END of the 1 s bin of the push "
               "time: the book state as of ts (known at ts); cts Int64 = matching-engine ms of the last applied message; bid1, ask1, bid1_size, "
               "ask1_size, mid, rel_spread_bp = (ask1 - bid1) / mid x 1e4; depth_{bid,ask}_{5,10,25} = USD notional within 5/10/25 bp of mid "
               "per side; imb_5, imb_10 = (bid - ask) / (bid + ask) depth at 5 / 10 bp; span_bid_bp, span_ask_bp = distance of the deepest "
               "of the 200 levels from mid (a depth band wider than the span is TRUNCATED: BTC ~2.7 bp, so BTC 5/10/25 bp depths are lower "
               "bounds); n_updates Int32 = messages in the bin; symbol Utf8. Join to trades_1s with book ts = trades ts + 1000 (state at the "
               "end of the trade second)",
    "orders": "one row per taker order: ms Int64, side Int8 (+1 buy / -1 sell), size, notional (USD), n_prints Int32, first_price, last_price",
    "laws/<SYM>.json": "halves{fit,test} in the schema of data/models/microstructure_laws/params_2026-11-10.json; tests (held-out comparisons); "
                       "activity{fit,test} (W, sigma_day, orders_per_s, Kyle lambda, ...); tick (minimum price increment of the day)",
}

# ----------------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------------


def load_all() -> dict[str, dict]:
    days = {}
    for d in MD.processed_days():
        p = OUT_ROOT / d
        laws = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((p / "laws").glob("*.json"))}
        ds = json.loads((p / "day_stats.json").read_text(encoding="utf-8"))
        qa = json.loads((p / "qa.json").read_text(encoding="utf-8"))
        days[d] = {"laws": laws, "stats": ds, "qa": qa, "segment": MD.segment_of(d)}
    return days


def _g(d, *path, default=None):
    return MD._g(d, *path, default=default)


def _fin(x) -> bool:
    return x is not None and isinstance(x, int | float) and math.isfinite(x)


def seg_days(days: dict, seg: str) -> list[str]:
    return sorted(d for d in days if seg == "pooled" or days[d]["segment"] == seg)


def _binom_p(k: int, n: int, p0: float) -> float:
    return float(stats.binom.sf(k - 1, n, p0)) if n else 1.0


def holm(p: dict[str, float]) -> dict[str, float]:
    ks = sorted(p, key=lambda k: p[k])
    m, out, run = len(ks), {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------------------------------------------------
# H-MD1: ICC
# ----------------------------------------------------------------------------------------------------------------------


def icc1(y: np.ndarray, g: np.ndarray) -> float:
    """One-way random-effects ICC(1) for unbalanced groups."""
    y = np.asarray(y, float)
    u, gi = np.unique(g, return_inverse=True)
    k = len(u)
    N = len(y)
    if k < 2 or N <= k:
        return float("nan")
    n_i = np.bincount(gi)
    m_i = np.bincount(gi, y) / n_i
    gm = y.mean()
    msb = float((n_i * (m_i - gm) ** 2).sum() / (k - 1))
    msw = float(((y - m_i[gi]) ** 2).sum() / (N - k))
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den > 0 else float("nan")


def icc_boot(y: np.ndarray, g: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    u = np.unique(g)
    idx = {s: np.flatnonzero(g == s) for s in u}
    out = np.full(B, np.nan)
    for b in range(B):
        pick = rng.choice(u, len(u), replace=True)
        yy = np.concatenate([y[idx[s]] for s in pick])
        gg = np.concatenate([np.full(len(idx[s]), i) for i, s in enumerate(pick)])
        out[b] = icc1(yy, gg)
    return out


def md1_traits(days: dict, dlist: list[str], half: str = "test", min_prints: int = 0) -> dict[str, list[tuple[str, str, float]]]:
    tr: dict[str, list] = {"gamma": [], "acf1": [], "zeta": [], "w1": [], "w2": [], "w3": []}
    for d in dlist:
        for s, lw in days[d]["laws"].items():
            t = lw.get("tests") or {}
            if (t.get("n_prints") or 0) < min_prints:
                continue
            so = t.get("sign_order") or {}
            vals = {"gamma": so.get(f"gamma_{half}"), "acf1": so.get(f"acf1_{half}"), "zeta": _g(t, f"size_print_{half}", "zeta")}
            w = _g(t, f"mix3_{half}", "w")
            if w:
                vals.update({"w1": w[0], "w2": w[1], "w3": w[2]})
            for k, v in vals.items():
                if _fin(v):
                    tr[k].append((s, d, float(v)))
    return tr


def md1(days: dict, dlist: list[str], rng: np.random.Generator, half: str = "test", min_prints: int = 0, B: int = B_ICC) -> dict[str, Any]:
    tr = md1_traits(days, dlist, half, min_prints)
    res: dict[str, Any] = {"n_days": len(dlist), "traits": {}}
    boots = {}
    for k, rows in tr.items():
        syms = [r[0] for r in rows]
        cnt = {s: syms.count(s) for s in set(syms)}
        rows = [r for r in rows if cnt[r[0]] >= 2]
        if len({r[0] for r in rows}) < 3:
            res["traits"][k] = {"error": "too few symbols with >= 2 days", "n": len(rows)}
            continue
        y = np.array([r[2] for r in rows])
        g = np.array([r[0] for r in rows])
        bs = icc_boot(y, g, B, rng)
        boots[k] = bs
        between = np.var([y[g == s].mean() for s in np.unique(g)])
        res["traits"][k] = {"icc": icc1(y, g), "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)),
                            "p_le_05": float(np.nanmean(bs <= 0.5)), "n_obs": len(y), "n_symbols": int(len(np.unique(g))),
                            "mean": float(y.mean()), "sd_between_symbol_means": float(math.sqrt(between)),
                            "sd_within": float(math.sqrt(np.mean([y[g == s].var(ddof=1) for s in np.unique(g)])))}
    if all(k in boots for k in ("w1", "w2", "w3")):
        # same bootstrap seeds are not shared across weights; the mean trait uses the mean of the three ICCs and their boots
        iccm = float(np.mean([res["traits"][k]["icc"] for k in ("w1", "w2", "w3")]))
        bm = np.nanmean(np.vstack([boots[k] for k in ("w1", "w2", "w3")]), axis=0)
        res["traits"]["mix_weights"] = {"icc": iccm, "lo": float(np.nanquantile(bm, 0.025)), "hi": float(np.nanquantile(bm, 0.975)),
                                        "p_le_05": float(np.nanmean(bm <= 0.5)), "components": ["w1", "w2", "w3"]}
    four = ("gamma", "acf1", "zeta", "mix_weights")
    passed = {k: bool(_fin(_g(res, "traits", k, "icc")) and res["traits"][k]["icc"] > 0.5 and res["traits"][k]["lo"] > 0.3) for k in four}
    ps = sorted(_g(res, "traits", k, "p_le_05", default=1.0) for k in four)
    res.update({"passed": passed, "n_passed": sum(passed.values()), "criterion": sum(passed.values()) >= 3, "p": float(ps[2])})
    return res


# ----------------------------------------------------------------------------------------------------------------------
# H-MD2: activity scaling
# ----------------------------------------------------------------------------------------------------------------------


def _cluster_cov(X: np.ndarray, e: np.ndarray, g: np.ndarray) -> np.ndarray:
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    u, gi = np.unique(g, return_inverse=True)
    S = np.zeros((len(u), k))
    np.add.at(S, gi, X * e[:, None])
    G = len(u)
    return A @ (S.T @ S) @ A * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))


def md2_rows(days: dict, dlist: list[str], half: str = "fit") -> list[dict]:
    rows = []
    for d in dlist:
        bk = days[d]["stats"].get("book", {})
        for r in days[d]["stats"].get("activity_rows", []):
            if r.get("half") != half or not _fin(r.get("W")) or r["W"] <= 0:
                continue
            trunc = _g(bk, r["symbol"], half, "band_truncated_share", "25")
            rows.append({**r, "day": d, "depth_ok": bool(_fin(r.get("depth25_med")) and r["depth25_med"] > 0 and (trunc is None or trunc <= 0.5))})
    return rows


def _design2(rows: list[dict], dlist: list[str], interact: bool) -> np.ndarray:
    lw = np.log([r["W"] for r in rows])
    cols = [np.ones(len(rows)), lw, np.array([r["class"] == "alt" for r in rows], float), np.array([r["class"] == "tradfi" for r in rows], float)]
    used = sorted({r["day"] for r in rows})
    for d in used[1:]:
        cols.append(np.array([r["day"] == d for r in rows], float))
    if interact:
        for d in used[1:]:
            cols.append(lw * np.array([r["day"] == d for r in rows], float))
    X = np.column_stack(cols)
    keep = [j for j in range(X.shape[1]) if j < 2 or X[:, j].std() > 0]
    return X[:, keep]


def md2_one(rows: list[dict], name: str, ykey, rng: np.random.Generator, B: int = B_LOO) -> dict[str, Any]:
    from ai_trader.sim.population import loo_r2

    rr = [r for r in rows if ykey(r) is not None]
    if len(rr) < 10:
        return {"error": "too few rows", "n": len(rr)}
    y = np.array([ykey(r) for r in rr])
    g = np.array([r["symbol"] for r in rr])
    dl = sorted({r["day"] for r in rr})
    X = _design2(rr, dl, False)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    V = _cluster_cov(X, e, g)
    out: dict[str, Any] = {"n": len(rr), "n_symbols": int(len(np.unique(g))), "n_days": len(dl), "slope": float(b[1]),
                           "slope_se": float(math.sqrt(max(V[1, 1], 0))), "r2_in": 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum()),
                           "loo_r2": loo_r2(y, X, groups=g)}
    if len(dl) >= 2:
        Xi = _design2(rr, dl, True)
        bi, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        ei = y - Xi @ bi
        Vi = _cluster_cov(Xi, ei, g)
        k0 = X.shape[1]
        ii = np.arange(k0, Xi.shape[1])
        if len(ii):
            bI = bi[ii]
            VI = Vi[np.ix_(ii, ii)]
            try:
                w = float(bI @ np.linalg.pinv(VI) @ bI)
                out["wald"] = {"chi2": w, "df": int(len(ii)), "p": float(stats.chi2.sf(w, len(ii)))}
            except np.linalg.LinAlgError:
                out["wald"] = {"error": "singular"}
    per = {}
    for d in dl:
        k = np.array([r["day"] == d for r in rr])
        if k.sum() < 8:
            continue
        Xd = np.column_stack([np.ones(k.sum()), np.log([r["W"] for r, kk in zip(rr, k, strict=True) if kk]),
                              np.array([r["class"] == "alt" for r, kk in zip(rr, k, strict=True) if kk], float),
                              np.array([r["class"] == "tradfi" for r, kk in zip(rr, k, strict=True) if kk], float)])
        Xd = Xd[:, [j for j in range(Xd.shape[1]) if j < 2 or Xd[:, j].std() > 0]]
        bd, *_ = np.linalg.lstsq(Xd, y[k], rcond=None)
        from ai_trader.sim.population import hc1_se

        se = hc1_se(Xd, y[k] - Xd @ bd)
        per[d] = {"slope": float(bd[1]), "se": float(se[1]), "n": int(k.sum())}
    out["per_day_slope"] = per
    # bootstrap over symbols of the LOO R^2
    u = np.unique(g)
    idx = {s: np.flatnonzero(g == s) for s in u}
    bs = []
    for _ in range(B):
        pick = rng.choice(u, len(u), replace=True)
        ix = np.concatenate([idx[s] for s in pick])
        gg = np.concatenate([np.full(len(idx[s]), i) for i, s in enumerate(pick)])
        Xb = X[ix]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        bs.append(loo_r2(y[ix], Xb, groups=gg))
    bs = np.array(bs)
    out["loo_ci"] = [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))] if len(bs) else None
    out["p_loo_le_05"] = float(np.nanmean(bs <= 0.5)) if len(bs) else 1.0
    return out


def md2(days: dict, dlist: list[str], rng: np.random.Generator, half: str = "fit") -> dict[str, Any]:
    rows = md2_rows(days, dlist, half)
    res = {"n_rows": len(rows),
           "rate": md2_one(rows, "rate", lambda r: math.log(r["orders_per_s"]) if _fin(r.get("orders_per_s")) and r["orders_per_s"] > 0 else None, rng),
           "impact": md2_one(rows, "impact", lambda r: math.log(r["lambda_mid"]) if _fin(r.get("lambda_mid")) and r["lambda_mid"] > 0 else None, rng),
           "depth": md2_one([r for r in rows if r["depth_ok"]], "depth", lambda r: math.log(r["depth25_med"]), rng, B=100)}
    ok = all(_fin(_g(res, k, "loo_r2")) and res[k]["loo_r2"] > 0.5 and _fin(_g(res, k, "wald", "p")) and res[k]["wald"]["p"] > 0.05
             for k in ("rate", "impact"))
    res["criterion"] = bool(ok)
    res["p"] = float(max(_g(res, "rate", "p_loo_le_05", default=1.0), _g(res, "impact", "p_loo_le_05", default=1.0)))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# H-MD3: sign long memory
# ----------------------------------------------------------------------------------------------------------------------


def md3(days: dict, dlist: list[str], rng: np.random.Generator) -> dict[str, Any]:
    flags, H, hd = [], [], []
    per_day = {}
    for d in dlist:
        k = n = 0
        for _s, lw in days[d]["laws"].items():
            t = lw.get("tests") or {}
            if (t.get("n_prints") or 0) < MIN_PRINTS_MD3:
                continue
            so = t.get("sign_order") or {}
            gm = so.get("gamma_test")
            f = bool(so.get("power_preferred_test")) and _fin(gm) and 0 < gm < 1
            flags.append(f)
            k += f
            n += 1
            h = _g(t, "hurst_1s_signed", "test")
            if _fin(h):
                H.append(h)
                hd.append(d)
        per_day[d] = {"k": k, "n": n, "share": k / n if n else None}
    k, n = sum(flags), len(flags)
    H, hd = np.array(H), np.array(hd)
    res: dict[str, Any] = {"k": k, "n": n, "share": k / n if n else None, "p_share": _binom_p(k, n, 0.8), "per_day": per_day}
    if len(H):
        ud = np.unique(hd)
        idx = {d: np.flatnonzero(hd == d) for d in ud}
        bs = np.array([H[np.concatenate([idx[d] for d in rng.choice(ud, len(ud), replace=True)])].mean() for _ in range(B_DAY)])
        res["hurst"] = {"mean": float(H.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                        "p_le_05": float(np.mean(bs <= 0.5)), "n": int(len(H)), "n_days": int(len(ud)),
                        "per_day_mean": {d: float(H[idx[d]].mean()) for d in ud}}
    res["criterion"] = bool(n and k / n >= 0.8 and _g(res, "hurst", "lo", default=0) > 0.5)
    res["p"] = float(max(res["p_share"], _g(res, "hurst", "p_le_05", default=1.0)))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# H-MD4: spread -> future vol for free-spread symbol-days
# ----------------------------------------------------------------------------------------------------------------------


def md4(days: dict, dlist: list[str]) -> dict[str, Any]:
    rows = []
    for d in dlist:
        st = days[d]["stats"]
        for s, b in st.get("book", {}).items():
            sh = _g(b, "fit", "share_gt_1tick")
            sv = st.get("spread_vol", {}).get(s, {})
            rows.append({"day": d, "symbol": s, "share_gt_1tick_fit": sh, "qualifying": bool(_fin(sh) and sh >= 0.25),
                         **{f"gain{h}": _g(sv, str(h), "r2_oos_gain") for h in (5, 15)},
                         **{f"t_test{h}": _g(sv, str(h), "t_test") for h in (5, 15)}})
    q = [r for r in rows if r["qualifying"]]
    res: dict[str, Any] = {"n_symbol_days": len(rows), "n_qualifying": len(q), "qualifying": [(r["day"], r["symbol"]) for r in q]}
    ps = []
    for h in (5, 15):
        g = [r[f"gain{h}"] for r in q if _fin(r[f"gain{h}"])]
        kpos = sum(x > 0 for x in g)
        res[f"h{h}"] = {"n": len(g), "k_pos": kpos, "share_pos": kpos / len(g) if g else None, "mean_gain": float(np.mean(g)) if g else None,
                        "median_gain": float(np.median(g)) if g else None, "p_vs_07": _binom_p(kpos, len(g), 0.7),
                        "p_sign": _binom_p(kpos, len(g), 0.5)}
        ps.append(res[f"h{h}"]["p_vs_07"])
        nq = [r[f"gain{h}"] for r in rows if not r["qualifying"] and _fin(r[f"gain{h}"])]
        res[f"h{h}_non_qualifying"] = {"n": len(nq), "share_pos": float(np.mean([x > 0 for x in nq])) if nq else None,
                                      "median_gain": float(np.median(nq)) if nq else None}
    res["rows"] = rows
    res["enough"] = len(q) >= 10
    res["criterion"] = bool(res["enough"] and all((res[f"h{h}"]["share_pos"] or 0) >= 0.7 for h in (5, 15)))
    res["p"] = float(max(ps))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# H-MD5: DY transmitters
# ----------------------------------------------------------------------------------------------------------------------


def md5(days: dict, dlist: list[str], rng: np.random.Generator) -> dict[str, Any]:
    per = {}
    for d in dlist:
        dy = days[d]["stats"].get("dy", {})
        net = _g(dy, "day", "net") or {}
        if not net:
            continue
        per[d] = {"net_day": {s: net.get(s) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT")},
                  "net_ci_day": {s: _g(dy, "day", "net_ci", s) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT")},
                  "net_test": {s: _g(dy, "test", "net", s) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT")},
                  "net_fit": {s: _g(dy, "fit", "net", s) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT")},
                  "total_day": _g(dy, "day", "total"), "n_used": len(dy.get("used", [])), "dropped": dy.get("dropped")}
    res: dict[str, Any] = {"per_day": per, "n_days": len(per)}
    ps = []
    for s in ("ETHUSDT", "SOLUSDT", "DOGEUSDT", "BTCUSDT"):
        v = [per[d]["net_day"][s] for d in per if _fin(per[d]["net_day"].get(s))]
        k = sum(x > 0 for x in v)
        res[s] = {"k_pos": k, "n": len(v), "share_pos": k / len(v) if v else None, "mean_net": float(np.mean(v)) if v else None,
                  "p_vs_08": _binom_p(k, len(v), 0.8)}
        vt = [per[d]["net_test"][s] for d in per if _fin(per[d]["net_test"].get(s))]
        res[s]["share_pos_test_half"] = float(np.mean([x > 0 for x in vt])) if vt else None
        if s in ("ETHUSDT", "SOLUSDT"):
            ps.append(res[s]["p_vs_08"])
    diff = np.array([per[d]["net_day"]["BTCUSDT"] - per[d]["net_day"]["ETHUSDT"] for d in per
                     if _fin(per[d]["net_day"].get("BTCUSDT")) and _fin(per[d]["net_day"].get("ETHUSDT"))])
    if len(diff):
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B_DAY)])
        res["btc_minus_eth"] = {"mean": float(diff.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": int(len(diff))}
    res["criterion"] = bool(all((res[s]["share_pos"] or 0) >= 0.8 for s in ("ETHUSDT", "SOLUSDT")) and res["n_days"] > 0)
    res["p"] = float(max(ps)) if ps else 1.0
    return res


# ----------------------------------------------------------------------------------------------------------------------
# H-MD6: v1 vs v0 envelopes per class-day
# ----------------------------------------------------------------------------------------------------------------------


def grid_from_aggregates(date: str, sym: str):
    """3 s Grid of one L2 symbol-day from book_1s + trades_1s (flow_3s-like fields)."""
    from ai_trader.sim.flow_data import Grid

    d0 = MD.day0_ms(date)
    n = MD.DAY_MS // 3000
    b = pl.read_parquet(OUT_ROOT / date / "book_1s" / f"{sym}.parquet").with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("t"))
    # 1 s state at end of second (t - 1000, t]; candle k = [d0 + 3000k, d0 + 3000(k+1)) holds states t in (open, open + 3000]
    k = (b["t"].to_numpy() - d0 - 1) // 3000
    ok_k = (k >= 0) & (k < n)
    k = k[ok_k]
    mid_s = b["mid"].to_numpy()[ok_k]
    spr_s = b["rel_spread_bp"].to_numpy()[ok_k]
    dep_s = (b["depth_bid_25"] + b["depth_ask_25"]).to_numpy()[ok_k]
    i5 = b["imb_5"].to_numpy()[ok_k]
    i10 = b["imb_10"].to_numpy()[ok_k]
    cnt = np.bincount(k, minlength=n)[:n]
    mid = np.full(n, np.nan)
    depth = np.full(n, np.nan)
    o5, o25 = np.full(n, np.nan), np.full(n, np.nan)
    last = np.flatnonzero(np.r_[k[1:] != k[:-1], True])
    mid[k[last]] = mid_s[last]
    depth[k[last]] = dep_s[last]
    o5[k[last]] = i5[last]
    o25[k[last]] = i10[last]
    fs = np.isfinite(spr_s)
    sp_sum = np.bincount(k[fs], spr_s[fs], minlength=n)[:n]
    sp_n = np.bincount(k[fs], minlength=n)[:n]
    spread = np.where(sp_n > 0, sp_sum / np.maximum(sp_n, 1), np.nan)
    tr = pl.read_parquet(OUT_ROOT / date / "trades_1s" / f"{sym}.parquet")
    kt = (tr["ts"].to_numpy() - d0) // 3000
    m = (kt >= 0) & (kt < n)
    kt = kt[m]
    buy = np.bincount(kt, tr["buy_notional"].to_numpy()[m], minlength=n)[:n]
    sell = np.bincount(kt, tr["sell_notional"].to_numpy()[m], minlength=n)[:n]
    nb = np.bincount(kt, tr["n_buy"].to_numpy()[m].astype(float), minlength=n)[:n]
    ns = np.bincount(kt, tr["n_sell"].to_numpy()[m].astype(float), minlength=n)[:n]
    hi = np.full(n, -np.inf)
    lo = np.full(n, np.inf)
    np.maximum.at(hi, kt, tr["high"].to_numpy()[m])
    np.minimum.at(lo, kt, tr["low"].to_numpy()[m])
    hi[~np.isfinite(hi)] = np.nan
    lo[~np.isfinite(lo)] = np.nan
    present = cnt > 0
    ok = present & np.isfinite(mid) & (mid > 0) & np.isfinite(spread) & (spread > 0)
    return Grid(sym, d0 + np.arange(n, dtype=np.int64) * 3000, ok, present, mid, spread, hi, lo, buy, sell, nb, ns, o5, o25, depth)


def _orders_as_prints(date: str, sym: str, a: int, b: int) -> tuple[dict, dict]:
    o = pl.read_parquet(OUT_ROOT / date / "orders" / f"{sym}.parquet")
    ms = o["ms"].to_numpy()
    k = (ms >= a) & (ms < b)
    tr = {"ms": ms[k], "side": o["side"].to_numpy()[k].astype(np.int64), "price": o["last_price"].to_numpy()[k], "notional": o["notional"].to_numpy()[k]}
    od = {"ms": ms[k], "side": tr["side"], "notional": tr["notional"], "n_prints": o["n_prints"].to_numpy()[k]}
    return tr, od


def sim_symbol_day(date: str, sym: str) -> dict[str, Any]:
    """Real TEST facts + bootstrap, v1 (FIT-calibrated) and v0 (FIT-fitted) path facts for one L2 symbol-day (cached)."""
    warnings.filterwarnings("ignore")
    cache = OUT_ROOT / date / "sim" / f"{sym}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from ai_trader.research import population_simulator_study as PS
    from ai_trader.research.simulator_study import _boot_facts
    from ai_trader.sim import flow_sim as fs
    from ai_trader.sim import flow_sim_v1 as V

    t0 = time.time()
    d0 = MD.day0_ms(date)
    half = d0 + MD.HALF_MS
    lw = json.loads((OUT_ROOT / date / "laws" / f"{sym}.json").read_text(encoding="utf-8"))
    tick = lw.get("tick")
    g = grid_from_aggregates(date, sym)
    fit_m = g.t_ms < half
    test_m = g.t_ms >= half
    n_val = int(g.ok[test_m].sum())
    f, bs = fs.facts_of_grid(g, test_m)
    rng = np.random.default_rng(SEED + sum(map(ord, sym)) + d0 // MD.DAY_MS)
    boot = _boot_facts(bs, rng, B_ENV)
    # FIT-half spread info (replaces the pilot's pre-TEST flow_3s days)
    pre_ok = fit_m & g.ok
    mid_pre = float(np.nanmedian(g.mid[pre_ok]))
    tick_bps = tick / mid_pre * 1e4
    sp = g.spread_bps[pre_ok]
    cf = V.censored_spread_fit(sp / tick_bps)
    ls = np.log(np.maximum(g.spread_bps, 1e-9))
    okp = pre_ok[1:] & pre_ok[:-1]
    a_, b_ = ls[1:][okp], ls[:-1][okp]
    acf1 = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 1e-9 and b_.std() > 1e-9 else float("nan")
    spr = {"tick_bps": tick_bps, "mid_pre": mid_pre, "spread_med_bps": float(np.median(sp)), "spread_p90_bps": float(np.percentile(sp, 90)),
           "acf1_log_spread_3s": acf1, "depth25_med": float(np.nanmedian(g.depth[pre_ok])), **cf}
    spr["tick_bound"] = bool(cf["share_lt_1p5_tick"] >= PS.REGIME_SHARE)
    fit_tr, fit_od = _orders_as_prints(date, sym, d0, half)
    meas = PS._half_measures(fit_tr, fit_od, d0)
    P = {"symbols": {sym: lw}}
    p = PS.build_params(sym, P, fit_tr, fit_od, meas, spr)
    # build_params anchors the price process at the pilot DAY0; only fitted quantities matter here
    pre = V.grid_stats(g, fit_m)
    tgt = PS.calib_targets(meas, pre)
    p, fin = V.calibrate(p, tgt, jump_z=PS.jump_z(meas), seed=SEED)
    v1 = []
    for k in range(N_PATHS):
        o = V.generate(p, 3 * n_val, PS._seed(sym, k, SEED), t0_ms=half)
        v1.append(PS._facts_vec(o.grid))
    out: dict[str, Any] = {"date": date, "symbol": sym, "class": MD.asset_class(sym), "n_val": n_val, "real": [f[x] for x in FACT_KEYS],
                           "boot": boot.tolist(), "v1": v1, "calib": fin, "spread_info": spr}
    try:
        m0 = fs.FlowSim.fit(g, fit_m, tick=tick)
        mid0 = float(g.mid[test_m & g.ok][0])
        out["v0"] = [PS._facts_vec(m0.generate(n_val, np.random.default_rng(PS._seed(sym, k, SEED + 7)), mid0=mid0, t0_ms=half)) for k in range(N_PATHS)]
    except Exception:  # noqa: BLE001
        out["v0_error"] = traceback.format_exc(limit=3)
    out["secs"] = time.time() - t0
    MD.write_json(cache, out)
    return MD.jsonable(out)


def _safe_sim(args) -> dict[str, Any]:
    d, s = args
    try:
        return sim_symbol_day(d, s)
    except Exception:  # noqa: BLE001
        return {"date": d, "symbol": s, "error": traceback.format_exc(limit=4)}


def md6_stage(days: dict, workers: int = WORKERS) -> list[dict]:
    jobs = [(d, s) for d in days for s in SIM_SYMBOLS if (OUT_ROOT / d / "book_1s" / f"{s}.parquet").exists() and s in days[d]["laws"]]
    todo = [j for j in jobs if not (OUT_ROOT / j[0] / "sim" / f"{j[1]}.json").exists()]
    print(f"H-MD6 sims: {len(todo)} of {len(jobs)} symbol-days to run", flush=True)
    res = []
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(_safe_sim, todo):
                print(f"  sim {r.get('date')} {r.get('symbol')} {'ERROR' if 'error' in r else f'{r.get(chr(115) + chr(101) + chr(99) + chr(115), 0):.0f}s'}", flush=True)
    for d, s in jobs:
        p = OUT_ROOT / d / "sim" / f"{s}.json"
        res.append(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"date": d, "symbol": s, "error": "missing"})
    return res


def _class_eval(F: list[dict]) -> dict[str, Any]:
    from ai_trader.research.population_simulator_study import _inside

    med_boot = np.nanmedian(np.stack([np.asarray(f["boot"], float) for f in F]), axis=0)
    lo, hi = np.nanpercentile(med_boot, 2.5, axis=0), np.nanpercentile(med_boot, 97.5, axis=0)
    out: dict[str, Any] = {"lo": lo.tolist(), "hi": hi.tolist(), "real": np.nanmedian(np.array([f["real"] for f in F], float), axis=0).tolist(),
                           "symbols": [f["symbol"] for f in F]}
    for v in ("v1", "v0"):
        vals = [np.nanmedian(np.asarray(f[v], float), axis=0) for f in F if f.get(v)]
        if len(vals) == len(F):
            sv = np.nanmedian(np.array(vals), axis=0)
            ins = _inside(sv, lo, hi)
            out[v] = {"value": sv.tolist(), "inside": ins, "n_inside": int(sum(ins.values()))}
    return out


def md6(sims: list[dict], dlist: list[str]) -> dict[str, Any]:
    ok = [s for s in sims if "error" not in s and s.get("date") in dlist and s.get("v1") and s.get("v0")]
    cd = {}
    for d in dlist:
        for c in ("major", "alt", "tradfi"):
            F = [s for s in ok if s["date"] == d and s["class"] == c]
            if len(F) >= 2:
                e = _class_eval(F)
                if "v1" in e and "v0" in e:
                    cd[f"{d}|{c}"] = {"v1": e["v1"]["n_inside"], "v0": e["v0"]["n_inside"], "v1_inside": e["v1"]["inside"],
                                      "v0_inside": e["v0"]["inside"], "n_symbols": len(F), "real": e["real"], "lo": e["lo"], "hi": e["hi"],
                                      "v1_value": e["v1"]["value"], "v0_value": e["v0"]["value"]}
    k = sum(v["v1"] > v["v0"] for v in cd.values())
    n = len(cd)
    maj = {key.split("|")[0]: v["v1"] for key, v in cd.items() if key.endswith("|major")}
    km = sum(x >= 4 for x in maj.values())
    res = {"class_days": cd, "k_v1_gt_v0": k, "n_class_days": n, "share": k / n if n else None, "p_share": _binom_p(k, n, 0.7),
           "majors_ge4": km, "n_major_days": len(maj), "share_majors_ge4": km / len(maj) if maj else None, "p_majors": _binom_p(km, len(maj), 0.5),
           "n_symbol_days": len(ok), "errors": [(s.get("date"), s.get("symbol")) for s in sims if "error" in s or not s.get("v0")],
           "mean_diff": float(np.mean([v["v1"] - v["v0"] for v in cd.values()])) if cd else None}
    fact_share = {}
    for fk in ("kurtosis", "absr_acf", "flow_acf", "impact", "spread"):
        fact_share[fk] = {"v1": float(np.mean([v["v1_inside"][fk] for v in cd.values()])) if cd else None,
                          "v0": float(np.mean([v["v0_inside"][fk] for v in cd.values()])) if cd else None}
    res["fact_inside_share"] = fact_share
    res["criterion"] = bool(n and k / n >= 0.7 and maj and km / len(maj) >= 0.5)
    res["p"] = float(max(res["p_share"], res["p_majors"]))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# pilot comparison (descriptive)
# ----------------------------------------------------------------------------------------------------------------------


def day_profile(days: dict) -> dict[str, Any]:
    prof = {}
    for d, D in sorted(days.items()):
        laws = D["laws"]
        cls = {s: lw.get("class") for s, lw in laws.items()}

        def med(fn, c=None):
            v = [fn(lw) for s, lw in laws.items() if (c is None or cls[s] == c)]
            v = [x for x in v if _fin(x)]
            return float(np.median(v)) if v else None

        hy = _g(D["stats"], "hy", "pairs") or {}
        crypto = [s for s in laws if cls[s] != "tradfi"]
        hyp = {}
        for L in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
            pk = [_g(hy, f"{L}->{t}", "test", "peak_lag_s") for t in crypto if t != L]
            pk = [x for x in pk if _fin(x)]
            hyp[L] = {"share_peak_at_0": float(np.mean([x == 0 for x in pk])) if pk else None,
                      "share_positive_peak": float(np.mean([0.1 <= x <= 30 for x in pk])) if pk else None, "n": len(pk),
                      "median_rho0": (float(np.median([_g(hy, f"{L}->{t}", "test", "rho0") for t in crypto if t != L and _fin(_g(hy, f"{L}->{t}", "test", "rho0"))]))
                                      if pk else None)}
        rows = [r for r in D["stats"].get("activity_rows", []) if r["half"] == "fit"]
        bk = D["stats"].get("book", {})
        prof[d] = {"segment": D["segment"], "n_symbols": len(laws), "missing": D["qa"].get("missing_trades"), "missing_books": D["qa"].get("missing_books"),
                   "events_per_s_major_test": med(lambda lw: _g(lw, "halves", "test", "rates", "events_per_s"), "major"),
                   "events_per_s_alt_test": med(lambda lw: _g(lw, "halves", "test", "rates", "events_per_s"), "alt"),
                   "test_over_fit_events_major": med(lambda lw: (_g(lw, "halves", "test", "rates", "events_per_s") or np.nan) /
                                                     (_g(lw, "halves", "fit", "rates", "events_per_s") or np.nan), "major"),
                   "sigma_day_bp_median": float(np.median([r["sigma_day"] for r in rows if _fin(r.get("sigma_day"))])) if rows else None,
                   "sigma_day_bp_btc": next((r["sigma_day"] for r in rows if r["symbol"] == "BTCUSDT"), None),
                   "notional_btc_usd": next((r["pv_day"] for r in rows if r["symbol"] == "BTCUSDT"), None),
                   "gamma_order_median": med(lambda lw: _g(lw, "tests", "sign_order", "gamma_test")),
                   "acf1_order_median": med(lambda lw: _g(lw, "tests", "sign_order", "acf1_test")),
                   "zeta_print_median": med(lambda lw: _g(lw, "tests", "size_print_test", "zeta")),
                   "hurst_1s_median": med(lambda lw: _g(lw, "tests", "hurst_1s_signed", "test")),
                   "prints_per_order_major": med(lambda lw: _g(lw, "halves", "fit", "rates", "prints_per_order_mean"), "major"),
                   "mix3_w_burst_median": med(lambda lw: (_g(lw, "tests", "mix3_test", "w") or [None])[0]),
                   "dy_total_day": _g(D["stats"], "dy", "day", "total"),
                   "dy_net_day": {s: _g(D["stats"], "dy", "day", "net", s) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT")},
                   "hy": hyp,
                   "tick_bound_l2": [s for s, b in bk.items() if _g(b, "fit", "tick_bound")],
                   "free_spread_l2": [s for s, b in bk.items() if (_g(b, "fit", "share_gt_1tick") or 0) >= 0.25],
                   "spread_med_bp_l2": {s: _g(b, "day", "spread_bp_q", "0.5") for s, b in bk.items()}}
    return prof


# ----------------------------------------------------------------------------------------------------------------------
# verdicts, locking, recording
# ----------------------------------------------------------------------------------------------------------------------


def books_pending(days: dict) -> set[str]:
    """Days processed with --no-books whose ob200 books are not yet backfilled (count as remaining for H-MD4 / H-MD6)."""
    return {d for d, D in days.items() if D["qa"].get("books_skipped")}


def remaining_days(days: dict, seg: str, need_books: bool = False) -> int:
    planned = [d for d in PLANNED_DAYS if seg == "pooled" or MD.segment_of(d) == seg]
    pend = books_pending(days) if need_books else set()
    return sum(d not in days or d in pend for d in planned)


def locked_contradiction(hid: str, R: dict, rem: int) -> bool:
    """True if a share part of the criterion cannot be met even if every remaining unit passes."""
    if "error" in R:
        return False
    if hid == "H-MD3":
        k, n = R["k"], R["n"]
        r = rem * 51
        return bool(n and (k + r) / (n + r) < 0.8)
    if hid == "H-MD4":
        r = rem * len(MD.L2_SYMBOLS)
        return any(((R[f"h{h}"]["n"] - R[f"h{h}"]["k_pos"]) > 0 and (R[f"h{h}"]["k_pos"] + r) / (R[f"h{h}"]["n"] + r) < 0.7) for h in (5, 15))
    if hid == "H-MD5":
        return any(R[s]["n"] and (R[s]["k_pos"] + rem) / (R[s]["n"] + rem) < 0.8 for s in ("ETHUSDT", "SOLUSDT"))
    if hid == "H-MD6":
        r = rem * 3
        n, k = R["n_class_days"], R["k_v1_gt_v0"]
        nm, km = R["n_major_days"], R["majors_ge4"]
        return bool((n and (k + r) / (n + r) < 0.7) or (nm and (km + rem) / (nm + rem) < 0.5))
    return False


def summarize(hid: str, R: dict) -> dict[str, Any]:
    keep = {"H-MD1": lambda r: {"n_passed": r.get("n_passed"), "passed": r.get("passed"),
                                "icc": {k: {x: _g(r, "traits", k, x) for x in ("icc", "lo", "hi", "n_symbols", "n_obs")} for k in ("gamma", "acf1", "zeta", "mix_weights")}},
            "H-MD2": lambda r: {k: {x: _g(r, k, x) for x in ("slope", "slope_se", "loo_r2", "loo_ci", "n", "n_symbols", "n_days")} | {"wald_p": _g(r, k, "wald", "p")}
                                for k in ("rate", "impact", "depth")},
            "H-MD3": lambda r: {"k": r["k"], "n": r["n"], "share": r["share"], "hurst": {x: _g(r, "hurst", x) for x in ("mean", "lo", "hi", "n", "n_days")}},
            "H-MD4": lambda r: {"n_qualifying": r["n_qualifying"], **{f"h{h}": {x: r[f"h{h}"][x] for x in ("n", "k_pos", "share_pos", "median_gain", "p_sign")} for h in (5, 15)}},
            "H-MD5": lambda r: {"n_days": r["n_days"], **{s: {x: r[s][x] for x in ("k_pos", "n", "share_pos", "mean_net")} for s in ("ETHUSDT", "SOLUSDT", "DOGEUSDT", "BTCUSDT")},
                                "btc_minus_eth": r.get("btc_minus_eth")},
            "H-MD6": lambda r: {x: r.get(x) for x in ("k_v1_gt_v0", "n_class_days", "share", "majors_ge4", "n_major_days", "share_majors_ge4", "mean_diff", "n_symbol_days")}}
    try:
        return keep[hid](R)
    except Exception:  # noqa: BLE001
        return {"error": "summary failed"}


def run(batch: int, do_md6: bool = True, record: bool = True, workers: int = WORKERS) -> dict[str, Any]:
    from ai_trader.research import hypotheses

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    days = load_all()
    print(f"days: {sorted(days)}", flush=True)
    sims = md6_stage(days, workers) if do_md6 else []
    out: dict[str, Any] = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "batch": batch, "days": sorted(days),
                           "segments_days": {s: seg_days(days, s) for s in ("llm_in_sample", "clean_holdout", "pooled")}, "method": METHOD,
                           "results": {}, "robustness": {}}
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        dl = seg_days(days, seg)
        if not dl:
            continue
        R: dict[str, Any] = {}
        for hid, fn in (("H-MD1", lambda: md1(days, dl, rng)), ("H-MD2", lambda: md2(days, dl, rng)), ("H-MD3", lambda: md3(days, dl, rng)),
                        ("H-MD4", lambda: md4(days, dl)), ("H-MD5", lambda: md5(days, dl, rng)),
                        ("H-MD6", (lambda: md6(sims, dl)) if do_md6 else (lambda: {"error": "not run", "p": 1.0, "criterion": False}))):
            try:
                R[hid] = fn()
            except Exception:  # noqa: BLE001
                R[hid] = {"error": traceback.format_exc(limit=4), "p": 1.0, "criterion": False}
        out["results"][seg] = R
        if seg == "pooled":
            out["robustness"]["H-MD1_fit_half"] = summarize("H-MD1", md1(days, dl, rng, half="fit", B=500))
            out["robustness"]["H-MD1_ge20k"] = summarize("H-MD1", md1(days, dl, rng, min_prints=MIN_PRINTS_MD3, B=500))
            out["robustness"]["H-MD2_test_half"] = summarize("H-MD2", md2(days, dl, rng, half="test"))
    out["profile"] = day_profile(days)
    # decisions
    final = all(d in days for d in PLANNED_DAYS) and not books_pending(days)
    dec: dict[str, Any] = {"final": final}
    if final:
        hp = {seg: holm({h: out["results"][seg][h].get("p", 1.0) for h in HIDS}) for seg in out["results"]}
    for seg, R in out["results"].items():
        dec[seg] = {}
        for hid in HIDS:
            rem = remaining_days(days, seg, need_books=hid in ("H-MD4", "H-MD6"))
            r = R[hid]
            already = [x for x in hypotheses.results(hid) if x["segment"] == seg]
            if final:
                v = "supports" if r.get("criterion") and hp[seg][hid] < 0.05 else ("inconclusive" if r.get("criterion") else "contradicts")
                decide = True
            else:
                v = "contradicts" if locked_contradiction(hid, r, rem) else None
                decide = v is not None
            dec[seg][hid] = {"criterion_so_far": bool(r.get("criterion")), "p": r.get("p"), "remaining_days": rem, "decidable": decide,
                             "verdict": v, "already_recorded": [x["verdict"] for x in already], "p_holm": hp[seg][hid] if final else None}
            if decide and record and not already:
                hypotheses.record(hid, seg, v, {"p_summary": r.get("p"), "p_holm": hp[seg][hid] if final else None, **summarize(hid, r),
                                                "days": seg_days(days, seg)},
                                  note=f"multiday_micro_study batch {batch}: " + ("final" if final else "locked: share criterion cannot be met with the remaining planned days"),
                                  data=f"data/research/micro_days ({len(seg_days(days, seg))} days); research/reports/multiday_micro_batch{batch}.json")
                dec[seg][hid]["recorded_now"] = v
            elif decide and already and already[-1]["verdict"] != v:
                dec[seg][hid]["note"] = f"recorded earlier as {already[-1]['verdict']}, now {v}: NOT re-recorded (orchestrator to review)"
    out["decisions"] = dec
    out["storage"] = MD.storage()
    out["runtime_s"] = time.time() - t0
    out["summary_line"] = summary_line(out)
    return out


def summary_line(out: dict) -> str:
    R = out["results"].get("pooled", {})
    parts = []
    try:
        r = R["H-MD1"]
        parts.append("MD1 ICC " + ", ".join(f"{k} {_g(r, 'traits', k, 'icc', default=float('nan')):.2f}" for k in ("gamma", "acf1", "zeta", "mix_weights")))
    except Exception:  # noqa: BLE001
        pass
    try:
        r = R["H-MD2"]
        parts.append(f"MD2 LOO R2 rate {r['rate']['loo_r2']:.2f} impact {r['impact']['loo_r2']:.2f} Wald p {_g(r, 'rate', 'wald', 'p', default=float('nan')):.3f}/{_g(r, 'impact', 'wald', 'p', default=float('nan')):.3f}")
    except Exception:  # noqa: BLE001
        pass
    try:
        r = R["H-MD3"]
        parts.append(f"MD3 {r['k']}/{r['n']} power-law, Hurst {r['hurst']['mean']:.3f} [{r['hurst']['lo']:.3f}, {r['hurst']['hi']:.3f}]")
    except Exception:  # noqa: BLE001
        pass
    try:
        r = R["H-MD4"]
        parts.append(f"MD4 qualifying {r['n_qualifying']}, gain>0 {r['h5']['k_pos']}/{r['h5']['n']} (5m) {r['h15']['k_pos']}/{r['h15']['n']} (15m)")
    except Exception:  # noqa: BLE001
        pass
    try:
        r = R["H-MD5"]
        parts.append(f"MD5 net>0 ETH {r['ETHUSDT']['k_pos']}/{r['ETHUSDT']['n']} SOL {r['SOLUSDT']['k_pos']}/{r['SOLUSDT']['n']}")
    except Exception:  # noqa: BLE001
        pass
    try:
        r = R["H-MD6"]
        parts.append(f"MD6 v1>v0 {r['k_v1_gt_v0']}/{r['n_class_days']} class-days, majors>=4/5 {r['majors_ge4']}/{r['n_major_days']}")
    except Exception:  # noqa: BLE001
        pass
    return "; ".join(parts)


# ----------------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------------


def _f(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    try:
        if not math.isfinite(x):
            return "n/a"
    except TypeError:
        return str(x)
    a = abs(x)
    if a != 0 and (a < 10 ** -nd or a >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _tbl(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(out: dict, batch: int, extra: dict | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = REPORT_DIR / f"multiday_micro_batch{batch}"
    slim = json.loads(json.dumps(MD.jsonable(out)))
    for seg in slim.get("results", {}).values():
        if "H-MD4" in seg:
            seg["H-MD4"].pop("rows", None)
    slim["aggregate_schema"] = AGGREGATE_SCHEMA
    if extra:
        slim["pipeline"] = extra
    base.with_suffix(".json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    md = render_md(out, batch, extra or {})
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    return base


def render_md(out: dict, batch: int, extra: dict) -> str:
    L: list[str] = []
    days = out["days"]
    L.append(f"# Multi-day microstructure, batch {batch}: H-MD1..H-MD6 interim (overnight program 2026-11-12)")
    L.append("")
    L.append(f"_data: data/research/micro_days ({len(days)} verified days: {', '.join(days)}) | segments: llm_in_sample "
             f"{out['segments_days'].get('llm_in_sample')}, clean_holdout {out['segments_days'].get('clean_holdout')} | code: "
             "src/ai_trader/research/micro_days.py, src/ai_trader/research/multiday_micro_study.py, scripts/micro_batch.py | tests: tests/test_micro_days.py_")
    L.append("")
    dec = out["decisions"]
    L.append("## Status of each hypothesis")
    L.append("")
    L.append("Interim numbers only: a verdict is recorded when a share criterion is already lost for good given the remaining pre-drawn days, "
             "or after the last batch (with Holm over the six). " + ("This is the FINAL batch." if dec.get("final") else "Not final."))
    L.append("")
    rows = []
    for hid in HIDS:
        for seg in ("llm_in_sample", "clean_holdout", "pooled"):
            if seg not in dec:
                continue
            d = dec[seg][hid]
            rows.append([hid, seg, "met" if d["criterion_so_far"] else "not met", _f(d["p"]), str(d["remaining_days"]),
                         d.get("recorded_now") or ("; ".join(d["already_recorded"]) if d["already_recorded"] else "not recorded")])
    L.append(_tbl(["id", "segment", "criterion so far", "summary p", "planned days left", "recorded"], rows))
    L.append("")
    L.append(f"Pooled summary: {out['summary_line']}")
    L.append("")
    if extra:
        L.append("## Processing and storage")
        L.append("")
        for k, v in extra.items():
            L.append(f"- **{k}**: {v}")
        L.append("")
    R = out["results"]
    # H-MD1
    L.append("## H-MD1: persistence of trade-level traits (ICC, symbol as group)")
    L.append("")
    rows = []
    for seg in R:
        r = R[seg]["H-MD1"]
        for k in ("gamma", "acf1", "zeta", "w1", "w2", "w3", "mix_weights"):
            t = _g(r, "traits", k) or {}
            rows.append([seg, k, t.get("icc"), t.get("lo"), t.get("hi"), t.get("n_symbols"), t.get("n_obs"), t.get("mean"),
                         t.get("sd_between_symbol_means"), t.get("sd_within")])
    L.append(_tbl(["segment", "trait (TEST half)", "ICC", "CI lo", "CI hi", "symbols", "symbol-days", "mean", "sd between", "sd within"], rows))
    L.append("")
    rb = out.get("robustness", {})
    for k in ("H-MD1_fit_half", "H-MD1_ge20k"):
        if k in rb:
            L.append(f"- robustness {k}: " + ", ".join(f"{t} {_f(_g(rb[k], 'icc', t, 'icc'))} [{_f(_g(rb[k], 'icc', t, 'lo'))}, {_f(_g(rb[k], 'icc', t, 'hi'))}]"
                                                       for t in ("gamma", "acf1", "zeta", "mix_weights")))
    L.append("")
    # H-MD2
    L.append("## H-MD2: activity scaling with W (FIT half, day fixed effects)")
    L.append("")
    rows = []
    for seg in R:
        r = R[seg]["H-MD2"]
        for k in ("rate", "impact", "depth"):
            t = r.get(k) or {}
            rows.append([seg, k, t.get("slope"), t.get("slope_se"), t.get("loo_r2"), str([round(x, 3) for x in t["loo_ci"]]) if t.get("loo_ci") else "n/a",
                         _g(t, "wald", "p"), t.get("n"), t.get("n_symbols"), t.get("n_days")])
    L.append(_tbl(["segment", "y", "slope on log W", "SE (cl. symbol)", "LOO R^2", "LOO CI", "Wald p (day x log W)", "rows", "symbols", "days"], rows))
    L.append("")
    pd_ = _g(R, "pooled", "H-MD2") or {}
    rows = []
    for d in days:
        rows.append([d] + [_g(pd_, k, "per_day_slope", d, "slope") for k in ("rate", "impact", "depth")] + [_g(pd_, k, "per_day_slope", d, "se") for k in ("rate", "impact")])
    L.append("Per-day slopes (pilot 2026-11-10 one-day values: rate ~W^0.54, depth ~W^0.68, impact ~W^-0.75):")
    L.append("")
    L.append(_tbl(["day", "rate slope", "impact slope", "depth slope", "rate SE", "impact SE"], rows))
    L.append("")
    # H-MD3
    L.append("## H-MD3: order-sign long memory")
    L.append("")
    rows = []
    for seg in R:
        r = R[seg]["H-MD3"]
        h = r.get("hurst") or {}
        rows.append([seg, f"{r.get('k')}/{r.get('n')}", r.get("share"), h.get("mean"), h.get("lo"), h.get("hi"), h.get("n_days")])
    L.append(_tbl(["segment", "power law & 0<gamma<1", "share", "mean Hurst 1 s signed flow", "CI lo (days)", "CI hi", "days"], rows))
    L.append("")
    r = _g(R, "pooled", "H-MD3") or {}
    L.append(_tbl(["day", "k/n", "share", "mean Hurst"], [[d, f"{v['k']}/{v['n']}", v["share"], _g(r, "hurst", "per_day_mean", d)] for d, v in (r.get("per_day") or {}).items()]))
    L.append("")
    # H-MD4
    L.append("## H-MD4: spread -> next 5 / 15 min volatility on free-spread symbol-days")
    L.append("")
    rows = []
    for seg in R:
        r = R[seg]["H-MD4"]
        for h in (5, 15):
            t = r.get(f"h{h}") or {}
            nq = r.get(f"h{h}_non_qualifying") or {}
            rows.append([seg, str(h), str(r.get("n_qualifying")), f"{t.get('k_pos')}/{t.get('n')}", t.get("share_pos"), t.get("median_gain"), t.get("p_sign"),
                         nq.get("share_pos"), nq.get("median_gain")])
    L.append(_tbl(["segment", "h (min)", "qualifying", "gain > 0", "share", "median gain", "sign-test p", "non-qualifying share > 0", "non-qual. median gain"], rows))
    L.append("")
    r = _g(R, "pooled", "H-MD4") or {}
    L.append(_tbl(["day", "symbol", "FIT share > 1 tick", "qualifying", "gain 5m", "gain 15m", "t TEST 5m"],
                  [[x["day"], x["symbol"], x["share_gt_1tick_fit"], x["qualifying"], x["gain5"], x["gain15"], x["t_test5"]] for x in r.get("rows", [])]))
    L.append("")
    # H-MD5
    L.append("## H-MD5: Diebold-Yilmaz net transmitters (log 1 m RV, full day)")
    L.append("")
    r = _g(R, "pooled", "H-MD5") or {}
    rows = []
    for d, v in (r.get("per_day") or {}).items():
        ci = v["net_ci_day"]
        rows.append([d] + [v["net_day"].get(s) for s in ("ETHUSDT", "SOLUSDT", "DOGEUSDT", "BTCUSDT")] +
                    [str([round(x, 1) for x in ci["ETHUSDT"]]) if ci.get("ETHUSDT") else "n/a", str([round(x, 1) for x in ci["SOLUSDT"]]) if ci.get("SOLUSDT") else "n/a",
                     v["total_day"], v["net_test"].get("ETHUSDT"), v["net_test"].get("SOLUSDT"), str(v["n_used"])])
    L.append(_tbl(["day", "ETH net", "SOL net", "DOGE net", "BTC net", "ETH CI", "SOL CI", "total", "ETH net TEST half", "SOL net TEST half", "symbols"], rows))
    L.append("")
    rows = []
    for seg in R:
        r5 = R[seg]["H-MD5"]
        rows.append([seg] + [f"{_g(r5, s, 'k_pos')}/{_g(r5, s, 'n')}" for s in ("ETHUSDT", "SOLUSDT", "DOGEUSDT", "BTCUSDT")] +
                    [_g(r5, "btc_minus_eth", "mean"), str([round(_g(r5, "btc_minus_eth", "lo", default=float("nan")), 1), round(_g(r5, "btc_minus_eth", "hi", default=float("nan")), 1)])])
    L.append(_tbl(["segment", "ETH net > 0", "SOL net > 0", "DOGE net > 0", "BTC net > 0", "mean BTC - ETH", "CI (days)"], rows))
    L.append("")
    # H-MD6
    L.append("## H-MD6: simulator v1 vs v0 on held-out half-days (12 L2 symbols)")
    L.append("")
    r = _g(R, "pooled", "H-MD6") or {}
    if "error" in r:
        L.append(f"Not run: {r['error'][:300]}")
    else:
        rows = [[k.split("|")[0], k.split("|")[1], str(v["n_symbols"]), str(v["v1"]), str(v["v0"]),
                 ", ".join(f for f, x in v["v1_inside"].items() if x) or "-", ", ".join(f for f, x in v["v0_inside"].items() if x) or "-"]
                for k, v in (r.get("class_days") or {}).items()]
        L.append(_tbl(["day", "class", "symbols", "v1 inside (of 5)", "v0 inside", "v1 facts inside", "v0 facts inside"], rows))
        L.append("")
        rows = []
        for seg in R:
            r6 = R[seg]["H-MD6"]
            rows.append([seg, f"{r6.get('k_v1_gt_v0')}/{r6.get('n_class_days')}", r6.get("share"), f"{r6.get('majors_ge4')}/{r6.get('n_major_days')}", r6.get("mean_diff")])
        L.append(_tbl(["segment", "v1 > v0 class-days", "share", "majors v1 >= 4/5", "mean v1 - v0"], rows))
        L.append("")
        fsh = r.get("fact_inside_share") or {}
        L.append("Share of class-days with the fact inside the envelope: " + ", ".join(f"{k} v1 {_f(v['v1'], 2)} / v0 {_f(v['v0'], 2)}" for k, v in fsh.items()))
        if r.get("errors"):
            L.append("")
            L.append(f"Symbol-days without a usable v1/v0 run: {r['errors']}")
    L.append("")
    # profile
    L.append("## Day profiles and differences from the 2026-11-10 pilot")
    L.append("")
    P = out["profile"]
    rows = [[d, v["segment"], str(v["n_symbols"]), v["events_per_s_major_test"], v["test_over_fit_events_major"], v["sigma_day_bp_btc"],
             v["notional_btc_usd"], v["gamma_order_median"], v["acf1_order_median"], v["zeta_print_median"], v["hurst_1s_median"],
             v["prints_per_order_major"], v["mix3_w_burst_median"], v["dy_total_day"]] for d, v in P.items()]
    L.append(_tbl(["day", "segment", "symbols", "major events/s TEST", "major TEST/FIT events", "BTC sigma_day (bp)", "BTC notional (USD, FIT x2)",
                   "median gamma", "median sign ACF1", "median zeta (prints)", "median Hurst", "major prints/order", "median burst weight", "DY total"], rows))
    L.append("")
    rows = [[d] + [_g(v, "hy", L_, "share_peak_at_0") for L_ in ("ETHUSDT", "SOLUSDT", "BTCUSDT")] + [_g(v, "hy", L_, "share_positive_peak") for L_ in ("ETHUSDT", "SOLUSDT", "BTCUSDT")]
            + [_g(v, "hy", "ETHUSDT", "median_rho0")] for d, v in P.items()]
    L.append("Hayashi-Yoshida lead-lag (TEST half, crypto targets): share of peaks at lag 0 and in (0.1, 30] s:")
    L.append("")
    L.append(_tbl(["day", "ETH peak at 0", "SOL peak at 0", "BTC peak at 0", "ETH positive peak", "SOL positive peak", "BTC positive peak", "ETH median rho(0)"], rows))
    L.append("")
    rows = [[d, ", ".join(v["free_spread_l2"]) or "-", ", ".join(s for s in v["missing"] or []) or "-", ", ".join(v["missing_books"] or []) or "-"] for d, v in P.items()]
    L.append(_tbl(["day", "free-spread L2 symbols (FIT share > 1 tick >= 25%)", "missing trade archives (404)", "missing ob200 (404)"], rows))
    L.append("")
    if extra.get("pilot_differences"):
        L.append(extra["pilot_differences"])
        L.append("")
    L.append("## Aggregate schema (data/research/micro_days, read-only for other agents)")
    L.append("")
    for k, v in AGGREGATE_SCHEMA.items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## Methods")
    L.append("")
    for k, v in METHOD.items():
        L.append(f"- **{k}**: {v}")
    for k, v in MD.METHOD.items():
        L.append(f"- **pipeline {k}**: {v}")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--no-md6", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--extra", default="", help="JSON file with pipeline notes for the report")
    a = ap.parse_args(argv)
    out = run(a.batch, do_md6=not a.no_md6, record=not a.no_record, workers=a.workers)
    extra = json.loads(Path(a.extra).read_text(encoding="utf-8")) if a.extra else None
    if extra is None:
        bj = OUT_ROOT / f"_batch_{a.batch}.json"
        if bj.exists():
            b = json.loads(bj.read_text(encoding="utf-8"))
            extra = {"batch days": ", ".join(f"{d} (verified {_g(v, 'verified', 'ok')}, raw deleted {v.get('raw_deleted')})" for d, v in b.get("days", {}).items()),
                     "storage before (GB)": json.dumps({k: round(v, 3) for k, v in b.get("storage_before", {}).items()}),
                     "storage after (GB)": json.dumps({k: round(v, 3) for k, v in b.get("storage_after", {}).items()})}
    p = write_report(out, a.batch, extra)
    print(f"report: {p}.md / .json | {out['summary_line']}", flush=True)
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
