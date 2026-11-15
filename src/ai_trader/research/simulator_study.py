"""H-SIM1..H-SIM4 (research/hypotheses.yaml, block SIM): execution simulator v0 on flow_3s and our live mm records.

All data (flow_3s from 2026-11-07, live quotes 2026-11-08/09) is after the knowledge cutoff: every result is segment
clean_holdout, a short first sample (about 3 days of collector data with gaps, one market state).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.simulator_study --as-of 2026-11-10T23:00:00Z [--record]
    (stages: --stage cache | sim1 | sim3 | sim2 | sim4 | report | all)

Choices the registry leaves open are fixed in METHOD (below) before the tests are run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.sim import fill_model as fmod
from ai_trader.sim import flow_sim as fs
from ai_trader.sim import reaction as rx
from ai_trader.sim import replay as rp
from ai_trader.sim.flow_data import Grid, asset_class, load_grid, load_live, read_symbol, save_grid, to_grid

SEED = 20261111
B = 2000
WORKERS = 4
DAY_MS = 86_400_000
FIT_DAYS = ("2026-11-08", "2026-11-09")
VAL_DAY = "2026-11-10"
ENVELOPE_DAYS = ("2026-11-08", "2026-11-09", "2026-11-10")
METHOD = {
    "data": "flow_3s Bybit linear, Lake.read_asof per symbol with column projection at a fixed as_of; complete 3 s grid; candles missing "
            "(collector gaps) or without a valid book (mid / spread null) are not ok and never zero-filled; lags and ACFs never cross a gap",
    "classes": "major = BTC, ETH, SOL, XRP, BNB, DOGE; tradfi = instrument symbol_type in ETF/stock/commodity or the TradFi stream list "
               "(incl. XAUT, PAXG); alt = the rest",
    "H-SIM1": "live mm quotes (orderLinkId 'mm...', deduplicated across the 5 pulls by orderId) with at-risk candles [created, cancel) or "
              "[created, fill candle]; usable when >= 95% of at-risk weight is in the lake, the fill candle is present and the 60 s markout mid is known; "
              "train = first half of usable quotes by creation time, test = second half; MLE of (theta0, b, eps) per variant (touch / strict); "
              "primary variant = lower train AIC; predicted fill rate = compensator sum of hazards / quotes; predicted markout = hazard-weighted; "
              "observed CIs: 30-min block bootstrap of the test half (B = 2000); markout = side * (mid at the close of the candle 60 s after the fill candle "
              "- quote price) / price in bps",
    "H-SIM3": "per symbol-day OLS of y = signed taker notional / day median depth_25 on lagged returns (sigma units; lags 1, 2-5, 6-20), obi_5, "
              "flow lags and recent vol; intensity log(1+trades) on lagged log spread, vol and activity lags; momentum = b1 + 4 b2 + 15 b3; "
              "5-minute clock clusters; sign stability = share of symbol-days with the modal sign for each primary response (momentum, obi, spread); "
              "equality: per symbol pooled over days, stacked cluster influence across symbols, Wald H0 equal (momentum, obi, spread) within class; "
              "supports = all three sign shares >= 0.80 and no rejection at 5% in >= 2 of 3 classes; symbol-days with < 2000 usable 3 s rows dropped",
    "H-SIM2": "generator (sim/flow_sim.py) fitted per symbol on 2026-11-08 + 2026-11-09, 10 paths of the 2026-11-10 ok-candle count each; facts: "
              "excess kurtosis of 3 s mid returns, mean ACF of |r| lags 1-60, mean ACF of signed flow lags 1-20, standardised impact slope corr(r_t, F_t), "
              "spread median and p90 (both must pass); class value = median over symbols (sim: of each symbol's median over paths); envelope = "
              "[min over the 3 real days of the 2.5% bootstrap quantile, max of the 97.5%] of the class median, 1 h block bootstrap within each symbol-day (B = 300); "
              "supports = >= 4 of 5 facts inside for every class; contradicts = some class has <= 2 inside",
    "H-SIM4": "atx mm quote rule ported (sim/replay.py) with config/mm_multi.yaml values, one symbol at a time, decisions every 3 candles (9 s), "
              "no structural levels; grid ask {0.2, 0.45, 0.9} x bid {0.2, 0.45, 0.9} x take-profit {0.3, 0.9} x requote {0.15, 0.5} sigma = 36 sets; "
              "replay on the real 2026-11-10 paths of the mm_multi symbols (warm-up from earlier candles) with the H-SIM1 primary fill model, 3 fill seeds; "
              "simulator: the H-SIM2 model of each symbol, 3 paths of the same ok-candle count, same fill model; metric = pooled net P&L / turnover (bps) "
              "per parameter set; Spearman of the 36 rankings, bootstrap over symbols (B = 2000); supports = rho > 0.5 with the 95% CI excluding 0",
    "fees": "maker 2 bps, taker 5.5 bps; market orders pay half-spread + 10 bps * sqrt(notional / depth_25)",
}


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


def day_mask(g: Grid, day: str) -> np.ndarray:
    a = _ms(day)
    return (g.t_ms >= a) & (g.t_ms < a + DAY_MS)


# ------------------------------------------------------------------------------------------ cache

def _cache_one(sym: str, as_of_iso: str, cache_dir: str) -> tuple[str, int]:
    from ai_trader.storage.lake import Lake
    p = Path(cache_dir) / f"{sym}.parquet"
    if p.exists():
        return sym, -1
    df = read_symbol(Lake(), sym, datetime.fromisoformat(as_of_iso))
    g = to_grid(df, sym)
    save_grid(g, p)
    return sym, len(g)


def stage_cache(as_of: datetime, cache: Path) -> dict:
    from ai_trader.features import flow_anomaly as fa
    from ai_trader.storage.lake import Lake
    cache.mkdir(parents=True, exist_ok=True)
    lake = Lake()
    syms = fa.flow_symbols(lake, as_of, "linear")
    ins = lake.read_asof("instruments", as_of, filters={"venue": "bybit", "category": "linear"},
                         columns=["symbol", "symbol_type", "tick_size", "qty_step", "min_qty", "min_notional"])
    meta = {r["symbol"]: r for r in ins.iter_rows(named=True)}
    (cache / "instruments.json").write_text(json.dumps({s: meta.get(s) for s in syms}, default=str), encoding="utf-8")
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_cache_one, syms, [as_of.isoformat()] * len(syms), [str(cache)] * len(syms)))
    print(f"cached {len(res)} symbols in {time.time() - t0:.0f}s", flush=True)
    return {"symbols": syms}


def load_meta(cache: Path) -> dict:
    return json.loads((cache / "instruments.json").read_text(encoding="utf-8"))


def symbols_in(cache: Path) -> list[str]:
    """Cached symbol grids (other parquet outputs in the folder are not symbols)."""
    known = set(load_meta(cache))
    return sorted(p.stem for p in cache.glob("*.parquet") if p.stem in known)



# ------------------------------------------------------------------------------------------ H-SIM1

def _blocks(ts_ms: np.ndarray, block_ms: int = 30 * 60_000) -> tuple[np.ndarray, int]:
    u, inv = np.unique(ts_ms // block_ms, return_inverse=True)
    return inv, len(u)


def _boot_ratio(num: np.ndarray, den: np.ndarray, blk: np.ndarray, nb: int, rng, b: int = B) -> tuple[float, float]:
    """95% CI of sum(num) / sum(den) under a block bootstrap (blocks resampled with replacement)."""
    sn = np.bincount(blk, weights=num, minlength=nb)
    sd = np.bincount(blk, weights=den, minlength=nb)
    draws = rng.integers(0, nb, (b, nb))
    tn, td = sn[draws].sum(axis=1), sd[draws].sum(axis=1)
    r = np.where(td > 0, tn / np.where(td > 0, td, 1), np.nan)
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def stage_sim1(cache: Path, root: Path) -> dict:
    meta = load_meta(cache)
    quotes, fills, info = load_live(root)
    first_fill = fills.group_by("order_id").agg(pl.col("exec_ms").min().alias("fill_ms"), pl.col("qty").sum().alias("fill_qty"))
    q = quotes.join(first_fill, on="order_id", how="left")
    # post-only rejects never rested on the book; user cancels (EC_PerCancelRequest) and fills are the sample
    rej = q.filter(pl.col("reject") == "EC_PostOnlyWillTakeLiquidity")
    q = q.filter(pl.col("reject") != "EC_PostOnlyWillTakeLiquidity")
    # a quote counts as filled only through its own maker executions (cumExecQty > 0 and an execution row)
    q = q.with_columns(pl.when(pl.col("filled") & pl.col("fill_ms").is_not_null()).then(pl.col("fill_ms")).otherwise(None).alias("fill_ms"))
    rows = q.to_dicts()
    syms = sorted({r["symbol"] for r in rows})
    grids = {s: load_grid(cache / f"{s}.parquet") for s in syms if (cache / f"{s}.parquet").exists()}
    ticks = {s: float((meta.get(s) or {}).get("tick_size") or 0.0) for s in syms}
    ex, qi = fmod.quote_exposure(rows, grids, ticks)
    reasons: dict[str, int] = {}
    for r in qi:
        reasons[r["reason"] or "usable"] = reasons.get(r["reason"] or "usable", 0) + 1
    usable = np.array([r["usable"] for r in qi])
    created = np.array([r["created_ms"] for r in rows])
    filled = np.array([r["fill_ms"] is not None for r in rows])
    cut = float(np.median(created[usable]))
    train_q = usable & (created < cut)
    test_q = usable & (created >= cut)
    row_train, row_test = train_q[ex.q], test_q[ex.q]
    fits = {}
    for v in ("touch", "strict"):
        m, st = fmod.fit(ex, v, row_train)
        fits[v] = (m, st)
    primary_v = min(fits, key=lambda v: fits[v][1]["aic"])
    rng = np.random.default_rng(SEED)
    idx_test = np.flatnonzero(test_q)
    blk, nb = _blocks(created[idx_test])
    obs_rate = float(filled[idx_test].mean())
    obs_mark_arr = np.array([qi[i]["obs_mark"] for i in idx_test])
    fm_mask = filled[idx_test] & np.isfinite(obs_mark_arr)
    obs_mark = float(np.nanmean(obs_mark_arr[fm_mask])) if fm_mask.any() else float("nan")
    ci_rate = _boot_ratio(filled[idx_test].astype(float), np.ones(len(idx_test)), blk, nb, rng)
    ci_mark = _boot_ratio(np.where(fm_mask, obs_mark_arr, 0.0), fm_mask.astype(float), blk, nb, rng)
    variants = {}
    for v, (m, st) in fits.items():
        pr = fmod.predict(ex, m, row_test)
        ef, _, _ = fmod.predict_by_quote(ex, m, len(rows))
        naive = fmod.predict(ex, fmod.FillModel(v, 1.0, 0.0, 0.0), row_test)
        pr_train = fmod.predict(ex, m, row_train)
        variants[v] = {"params": m.to_json(), "fit": st, "test_ll": fmod.loglik(ex, m, row_test),
                       "pred_fill_rate": pr["exp_fills"] / len(idx_test), "pred_markout_bps": pr["pred_markout_bps"],
                       "train_pred_fill_rate": pr_train["exp_fills"] / max(train_q.sum(), 1),
                       "train_obs_fill_rate": float(filled[train_q].mean()),
                       "naive_theta1_fill_rate": naive["exp_fills"] / len(idx_test), "naive_theta1_markout_bps": naive["pred_markout_bps"],
                       "rate_inside": bool(ci_rate[0] <= pr["exp_fills"] / len(idx_test) <= ci_rate[1]),
                       "markout_inside": bool(ci_mark[0] <= pr["pred_markout_bps"] <= ci_mark[1])}
        _ = ef
    # self-print diagnostic: fill candles whose aggressive notional on that side equals our own fill
    self_only, n_f = 0, 0
    for i in np.flatnonzero(usable & filled):
        r = rows[i]
        g = grids[r["symbol"]]
        k = (int(r["fill_ms"]) - int(g.t_ms[0])) // 3000
        aggr = g.sell[k] if r["side"] > 0 else g.buy[k]
        mine = (r.get("fill_qty") or r["qty"]) * r["price"]
        n_f += 1
        self_only += int(abs(aggr - mine) <= 0.02 * mine)
    pv = variants[primary_v]
    n_test_fills = int(filled[idx_test].sum())
    if n_test_fills < 30:
        verdict = "inconclusive"
    elif pv["rate_inside"] and pv["markout_inside"]:
        verdict = "supports"
    else:
        verdict = "contradicts"
    by_sym = (pl.DataFrame({"symbol": [r["symbol"] for r in rows], "usable": usable, "filled": filled})
              .group_by("symbol").agg(pl.len().alias("quotes"), pl.col("usable").sum(), pl.col("filled").sum()).sort("quotes", descending=True))
    out = {"live": info, "rejected_post_only_or_other": rej.height, "quotes_considered": len(rows), "coverage": reasons,
           "usable": int(usable.sum()), "usable_share": float(usable.mean()), "usable_filled": int((usable & filled).sum()),
           "split_ms": cut, "split_utc": datetime.fromtimestamp(cut / 1000, UTC).isoformat(),
           "train": {"quotes": int(train_q.sum()), "fills": int((train_q & filled).sum())},
           "test": {"quotes": int(test_q.sum()), "fills": n_test_fills, "obs_fill_rate": obs_rate, "ci_fill_rate": ci_rate,
                    "obs_markout_bps": obs_mark, "ci_markout_bps": ci_mark, "blocks": nb},
           "variants": variants, "primary_variant": primary_v,
           "self_print_share": self_only / max(n_f, 1), "self_print_n": n_f,
           "by_symbol_top": by_sym.head(15).to_dicts(), "verdict": verdict}
    (cache / "sim1.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("coverage", "usable", "train", "test", "primary_variant", "verdict")}, default=float), flush=True)
    for v, d in variants.items():
        print(v, {k: d[k] for k in ("params", "pred_fill_rate", "pred_markout_bps", "naive_theta1_fill_rate", "rate_inside", "markout_inside")}, flush=True)
    return out



# ------------------------------------------------------------------------------------------ H-SIM3

ALL_DAYS = ("2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10")
MIN_ROWS_DAY = 2000


def _sim3_one(sym: str, cache_dir: str) -> dict:
    g = load_grid(Path(cache_dir) / f"{sym}.parquet")
    out = {"symbol": sym, "days": {}, "days_1m": {}}
    masks = []
    for d in ALL_DAYS:
        m = day_mask(g, d)
        if g.ok[m].sum() < MIN_ROWS_DAY:
            continue
        f3 = rx.fit_symbol_day(g, m, 1)
        if f3 is None or f3["n"] < MIN_ROWS_DAY:
            continue
        out["days"][d] = f3
        masks.append(m)
        f1 = rx.fit_symbol_day(g, m, 20)
        if f1 is not None:
            out["days_1m"][d] = f1
    if masks:
        pooled = rx.fit_pooled(g, masks, 1)
        if pooled is not None:
            cl = np.union1d(pooled["clus_dir"], pooled["clus_int"])
            inf = np.zeros((len(cl), 3))
            inf[np.searchsorted(cl, pooled["clus_dir"]), 0:2] = pooled["infl_dir"]
            inf[np.searchsorted(cl, pooled["clus_int"]), 2:3] = pooled["infl_int"]
            out["pooled"] = {"est": np.array([pooled["est"][k] for k in rx.PRIMARY]), "se": pooled["se"], "clusters": cl, "infl": inf, "n": pooled["n"]}
    return out


COMPONENTS = [("dir", "r1", "3 s momentum (lag 1)"), ("dir", "r2_5", "momentum lags 2-5"), ("dir", "r6_20", "momentum lags 6-20 (to 1 min)"),
              ("dir", "obi5", "book-imbalance takers (obi_5)"), ("dir", "y1", "flow persistence lag 1 (order splitting / herding)"),
              ("dir", "y2_5", "flow persistence lags 2-5"), ("dir", "vol", "signed flow on recent vol"),
              ("int", "log_spread", "intensity on log spread (liquidity-sensitive)"), ("int", "vol", "intensity on recent vol")]


def stage_sim3(cache: Path) -> dict:
    meta = load_meta(cache)
    syms = symbols_in(cache)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_sim3_one, syms, [str(cache)] * len(syms)))
    print(f"sim3 fits {time.time() - t0:.0f}s", flush=True)
    cls = {s: asset_class(s, (meta.get(s) or {}).get("symbol_type")) for s in syms}
    rows = []
    for r in res:
        for freq, key in (("3s", "days"), ("1m", "days_1m")):
            for d, f in r[key].items():
                row = {"symbol": r["symbol"], "class": cls[r["symbol"]], "day": d, "freq": freq, "n": f["n"], "r2_dir": f["r2_dir"], "r2_int": f["r2_int"]}
                row.update({f"p_{k}": v for k, v in f["primary"].items()})
                row.update({f"t_{k}": f["primary"][k] / f["primary_se"][k] if f["primary_se"][k] > 0 else float("nan") for k in rx.PRIMARY})
                for eq, name, _ in COMPONENTS:
                    row[f"{eq}_{name}"] = f[eq][name]
                    row[f"{eq}_{name}_t"] = f[eq + "_t"][name]
                for k, v in (f.get("var") or {}).items():
                    row[f"var_{k}"] = v
                row["mom_trade_scaled"] = f.get("momentum_trade_scaled", float("nan"))
                rows.append(row)
    T = pl.DataFrame(rows)
    T.write_parquet(cache / "sim3_symbol_days.parquet")
    t3 = T.filter(pl.col("freq") == "3s")
    sign = {k: rx.sign_stability(t3[f"p_{k}"].to_list()) for k in rx.PRIMARY}
    sign_1m = {k: rx.sign_stability(T.filter(pl.col("freq") == "1m")[f"p_{k}"].to_list()) for k in rx.PRIMARY}
    sign_by_class = {c: {k: rx.sign_stability(t3.filter(pl.col("class") == c)[f"p_{k}"].to_list()) for k in rx.PRIMARY}
                     for c in ("major", "alt", "tradfi")}
    comp = []
    for c in ("all", "major", "alt", "tradfi"):
        for freq in ("3s", "1m"):
            sub = T.filter(pl.col("freq") == freq)
            if c != "all":
                sub = sub.filter(pl.col("class") == c)
            for eq, name, label in COMPONENTS:
                v = sub[f"{eq}_{name}"].to_numpy()
                t = sub[f"{eq}_{name}_t"].to_numpy()
                ok = np.isfinite(v)
                if ok.sum() == 0:
                    continue
                comp.append({"class": c, "freq": freq, "component": f"{eq}.{name}", "label": label, "n": int(ok.sum()),
                             "median": float(np.median(v[ok])), "share_pos": float((v[ok] > 0).mean()),
                             "sig_pos": float((t[ok] > 2).mean()), "sig_neg": float((t[ok] < -2).mean())})
            for k in ("var_flow_to_ret", "var_ret_to_flow", "var_flow_persist"):
                if k in sub.columns:
                    v = sub[k].drop_nulls().to_numpy()
                    v = v[np.isfinite(v)]
                    if len(v):
                        comp.append({"class": c, "freq": freq, "component": k, "label": "VAR(5) cumulative 20-step response", "n": int(len(v)),
                                     "median": float(np.median(v)), "share_pos": float((v > 0).mean()), "sig_pos": float("nan"), "sig_neg": float("nan")})
    wald = {}
    for c in ("major", "alt", "tradfi"):
        blocks = [(r["pooled"]["est"], r["pooled"]["clusters"], r["pooled"]["infl"]) for r in res if "pooled" in r and cls[r["symbol"]] == c]
        w = {"joint": rx.stacked_wald(blocks)}
        for j, k in enumerate(rx.PRIMARY):
            w[k] = rx.stacked_wald([(b[0][j:j + 1], b[1], b[2][:, j:j + 1]) for b in blocks])
        ests = np.array([b[0] for b in blocks]) if blocks else np.zeros((0, 3))
        w["scaled_coef"] = {k: {"median": float(np.median(ests[:, j])) if len(ests) else float("nan"),
                                "iqr": [float(np.percentile(ests[:, j], 25)), float(np.percentile(ests[:, j], 75))] if len(ests) else None}
                            for j, k in enumerate(rx.PRIMARY)}
        wald[c] = w
    sign_ok = all(sign[k]["share"] >= 0.80 for k in rx.PRIMARY)
    classes_equal = sum(1 for c in wald if np.isfinite(wald[c]["joint"]["p"]) and wald[c]["joint"]["p"] >= 0.05)
    verdict = "supports" if sign_ok and classes_equal >= 2 else "contradicts"
    out = {"symbol_days_3s": t3.height, "symbols": t3["symbol"].n_unique(), "days": sorted(t3["day"].unique().to_list()),
           "sign_stability_3s": sign, "sign_stability_1m": sign_1m, "sign_by_class_3s": sign_by_class, "components": comp, "wald": wald,
           "classes": {c: sum(1 for s in syms if cls[s] == c) for c in ("major", "alt", "tradfi")},
           "r2_dir_median": float(t3["r2_dir"].median()), "r2_int_median": float(t3["r2_int"].median()),
           "sign_ok": sign_ok, "classes_not_rejected": classes_equal, "verdict": verdict}
    (cache / "sim3.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("symbol_days_3s", "sign_stability_3s", "classes_not_rejected", "verdict")}, default=float), flush=True)
    for c, w in wald.items():
        print(c, {k: (round(v["W"], 1), v["df"], v["p"]) for k, v in w.items() if k != "scaled_coef"}, flush=True)
    return out


# ------------------------------------------------------------------------------------------ H-SIM2

B_ENV = 300
N_PATHS = 10
FACT_KEYS = ("kurtosis", "absr_acf", "flow_acf", "impact", "spread_med", "spread_p90", "continuation")  # last = descriptive extra


def _boot_facts(bs: list, rng, b: int = B_ENV) -> np.ndarray:
    nb = len(bs)
    out = np.full((b, len(FACT_KEYS)), np.nan)
    if nb == 0:
        return out
    for k in range(b):
        f = fs.facts_from_blocks(bs, rng.integers(0, nb, nb))
        out[k] = [f[x] for x in FACT_KEYS]
    return out


def _sim2_one(sym: str, cache_dir: str, tick: float) -> dict:
    import pickle
    cache = Path(cache_dir)
    g = load_grid(cache / f"{sym}.parquet")
    rng = np.random.default_rng(SEED + sum(map(ord, sym)))
    out = {"symbol": sym, "real": {}, "boot": {}}
    fit_mask = day_mask(g, FIT_DAYS[0]) | day_mask(g, FIT_DAYS[1])
    vm = day_mask(g, VAL_DAY)
    n_val = int(g.ok[vm].sum())
    if g.ok[fit_mask].sum() < 4 * 1200 or n_val < 1200:
        out["skip"] = f"fit candles {int(g.ok[fit_mask].sum())}, validation candles {n_val}"
        return out
    try:
        m = fs.FlowSim.fit(g, fit_mask, tick or None)
    except Exception as e:  # noqa: BLE001
        out["skip"] = f"fit failed: {e}"
        return out
    (cache / "models").mkdir(exist_ok=True)
    (cache / "models" / f"{sym}.pkl").write_bytes(pickle.dumps(m))
    for d in ENVELOPE_DAYS:
        mk = day_mask(g, d)
        if g.ok[mk].sum() < 1200:
            continue
        f, bs = fs.facts_of_grid(g, mk)
        out["real"][d] = [f[x] for x in FACT_KEYS]
        out["boot"][d] = _boot_facts(bs, rng)
    mid0 = float(g.mid[vm & g.ok][0])
    paths = []
    for k in range(N_PATHS):
        gp = m.generate(n_val, np.random.default_rng(SEED + 1000 * k + sum(map(ord, sym))), mid0=mid0, t0_ms=int(g.t_ms[vm][0]))
        f, _ = fs.facts_of_grid(gp)
        paths.append([f[x] for x in FACT_KEYS])
    out["sim_paths"] = np.array(paths)
    out["sim"] = np.nanmedian(out["sim_paths"], axis=0)
    mj = replace(m, jumps=True)
    pj = []
    for k in range(N_PATHS):
        gp = mj.generate(n_val, np.random.default_rng(SEED + 1000 * k + sum(map(ord, sym))), mid0=mid0, t0_ms=int(g.t_ms[vm][0]))
        f, _ = fs.facts_of_grid(gp)
        pj.append([f[x] for x in FACT_KEYS])
    out["sim_jump"] = np.nanmedian(np.array(pj), axis=0)
    out["jump_lam"] = m.jump_lam
    out["n_val"] = n_val
    out["model"] = {"sigma": m.sigma, "depth": m.depth, "trade_size": m.trade_size, "act_persist": float(sum(m.act[1:4])),
                    "impact_lin": m.ret[0], "impact_sqrt": m.ret[1], "ret_ar": m.ret[2], "dir_mom1": m.dirn[3], "dir_mom2_20": m.dirn[4],
                    "dir_q1": m.dirn[1], "dir_obi": m.dirn[5], "ret_df": m.ret_df, "pi0": m.pi0, "spread_rho": m.spr[1]}
    return out


def stage_sim2(cache: Path) -> dict:
    meta = load_meta(cache)
    syms = symbols_in(cache)
    ticks = [float((meta.get(s) or {}).get("tick_size") or 0.0) for s in syms]
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_sim2_one, syms, [str(cache)] * len(syms), ticks))
    print(f"sim2 fits {time.time() - t0:.0f}s", flush=True)
    cls = {s: asset_class(s, (meta.get(s) or {}).get("symbol_type")) for s in syms}
    ok = [r for r in res if "sim" in r]
    skipped = {r["symbol"]: r["skip"] for r in res if "skip" in r}
    classes = {}
    sym_pass = []
    for r in ok:
        lo = np.full(len(FACT_KEYS), np.inf)
        hi = np.full(len(FACT_KEYS), -np.inf)
        for _d, bt in r["boot"].items():
            lo = np.fmin(lo, np.nanpercentile(bt, 2.5, axis=0))
            hi = np.fmax(hi, np.nanpercentile(bt, 97.5, axis=0))
        inside = (r["sim"] >= lo) & (r["sim"] <= hi)
        sym_pass.append({"symbol": r["symbol"], "class": cls[r["symbol"]], **{f"in_{k}": bool(v) for k, v in zip(FACT_KEYS, inside, strict=True)},
                         **{f"real3_{k}": (r["real"].get(VAL_DAY) or [np.nan] * len(FACT_KEYS))[j] for j, k in enumerate(FACT_KEYS)},
                         **{f"sim_{k}": float(r["sim"][j]) for j, k in enumerate(FACT_KEYS)}})
    for c in ("major", "alt", "tradfi"):
        rs = [r for r in ok if cls[r["symbol"]] == c]
        if not rs:
            continue
        env_lo = np.full(len(FACT_KEYS), np.inf)
        env_hi = np.full(len(FACT_KEYS), -np.inf)
        real_day = {}
        for d in ENVELOPE_DAYS:
            arr = [r["boot"][d] for r in rs if d in r["boot"]]
            if not arr:
                continue
            med = np.nanmedian(np.stack(arr), axis=0)  # (B, facts): class median per bootstrap replicate
            env_lo = np.fmin(env_lo, np.nanpercentile(med, 2.5, axis=0))
            env_hi = np.fmax(env_hi, np.nanpercentile(med, 97.5, axis=0))
            real_day[d] = np.nanmedian(np.array([r["real"][d] for r in rs if d in r["real"]]), axis=0).tolist()
        simv = np.nanmedian(np.array([r["sim"] for r in rs]), axis=0)
        simj = np.nanmedian(np.array([r["sim_jump"] for r in rs]), axis=0)

        def facts_in(v, lo=env_lo, hi=env_hi):
            ins = (v >= lo) & (v <= hi)
            return {"kurtosis": bool(ins[0]), "absr_acf": bool(ins[1]), "flow_acf": bool(ins[2]), "impact": bool(ins[3]),
                    "spread": bool(ins[4] and ins[5])}, bool(ins[6])
        facts, cont_in = facts_in(simv)
        facts_j, cont_in_j = facts_in(simj)
        classes[c] = {"symbols": len(rs), "sim": simv.tolist(), "env_lo": env_lo.tolist(), "env_hi": env_hi.tolist(), "real_by_day": real_day,
                      "facts_inside": facts, "n_inside": int(sum(facts.values())), "continuation_inside": cont_in,
                      "jump_variant": {"sim": simj.tolist(), "facts_inside": facts_j, "n_inside": int(sum(facts_j.values())), "continuation_inside": cont_in_j,
                                       "jump_lam_median": float(np.median([r["jump_lam"] for r in rs]))},
                      "model_median": {k: float(np.median([r["model"][k] for r in rs])) for k in (
                          "sigma", "act_persist", "impact_lin", "impact_sqrt", "ret_ar", "dir_mom1", "dir_mom2_20", "dir_q1", "dir_obi", "ret_df",
                          "spread_rho")}}
    SP = pl.DataFrame(sym_pass) if sym_pass else pl.DataFrame()
    sym_share = {c: {k: float(SP.filter(pl.col("class") == c)[f"in_{k}"].mean()) for k in FACT_KEYS} for c in classes} if SP.height else {}
    all_ge4 = all(v["n_inside"] >= 4 for v in classes.values()) and len(classes) == 3
    any_le2 = any(v["n_inside"] <= 2 for v in classes.values())
    verdict = "supports" if all_ge4 else "contradicts" if any_le2 else "inconclusive"
    out = {"fit_days": FIT_DAYS, "val_day": VAL_DAY, "symbols": len(ok), "skipped": skipped, "classes": classes, "symbol_pass_share": sym_share,
           "fact_keys": FACT_KEYS, "verdict": verdict}
    (cache / "sim2.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    print(json.dumps({c: (v["symbols"], v["facts_inside"]) for c, v in classes.items()}), verdict, flush=True)
    for c, v in classes.items():
        print(c, "sim", np.round(v["sim"], 3), "lo", np.round(v["env_lo"], 3), "hi", np.round(v["env_hi"], 3), flush=True)
    return out


# ------------------------------------------------------------------------------------------ H-SIM4

N_SEEDS = 3


def _sim4_one(sym: str, cache_dir: str, rule: dict, fm: dict, cfgs: list) -> dict:
    import pickle
    cache = Path(cache_dir)
    g = load_grid(cache / f"{sym}.parquet")
    mp = cache / "models" / f"{sym}.pkl"
    if not mp.exists():
        return {"symbol": sym, "skip": "no H-SIM2 model"}
    m = pickle.loads(mp.read_bytes())
    R = rp.Rule(float(rule["tick_size"]), float(rule["qty_step"]), float(rule["min_qty"]), float(rule.get("min_notional") or 5.0))
    F = fmod.FillModel(**fm)
    vm = day_mask(g, VAL_DAY)
    start = int(np.flatnonzero(vm)[0])
    n_val = int(g.ok[vm].sum())
    mid0 = float(g.mid[vm & g.ok][0])
    real = np.zeros((len(cfgs), 4))
    sim = np.zeros((len(cfgs), 4))
    paths = [m.generate(6000 + n_val, np.random.default_rng(SEED + 77 * k + sum(map(ord, sym))), mid0=mid0, t0_ms=int(g.t_ms[start]) - 6000 * 3000)
             for k in range(N_SEEDS)]
    for j, c in enumerate(cfgs):
        for k in range(N_SEEDS):
            o = rp.run_mm(g, R, c, F, np.random.default_rng(SEED + k), start=start, warmup=6000)
            real[j] += [o["pnl"], o["turnover"], o["maker_fills"], o["fees"]]
            o = rp.run_mm(paths[k], R, c, F, np.random.default_rng(SEED + k), start=6000, warmup=6000)
            sim[j] += [o["pnl"], o["turnover"], o["maker_fills"], o["fees"]]
    return {"symbol": sym, "real": real / N_SEEDS, "sim": sim / N_SEEDS, "n_val": n_val}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def stage_sim4(cache: Path) -> dict:
    import yaml
    meta = load_meta(cache)
    s1 = json.loads((cache / "sim1.json").read_text(encoding="utf-8"))
    fm = s1["variants"][s1["primary_variant"]]["params"]
    symbols = yaml.safe_load(Path("config/mm_multi.yaml").read_text(encoding="utf-8"))["symbols"]
    base = rp.MmCfg.from_yaml(Path("config/mm_multi.yaml"))
    cfgs = rp.grid_params(base) + [base]  # last = the live configuration (reference, not ranked)
    syms = [s for s in symbols if (cache / f"{s}.parquet").exists() and (cache / "models" / f"{s}.pkl").exists()]
    rules = [meta[s] for s in syms]
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_sim4_one, syms, [str(cache)] * len(syms), rules, [fm] * len(syms), [cfgs] * len(syms)))
    print(f"sim4 replays {time.time() - t0:.0f}s", flush=True)
    res = [r for r in res if "real" in r]
    cls = {s: asset_class(s, (meta.get(s) or {}).get("symbol_type")) for s in syms}
    RP = np.stack([r["real"][:, 0] for r in res])   # (S, C) pnl
    RT = np.stack([r["real"][:, 1] for r in res])   # turnover
    SPn = np.stack([r["sim"][:, 0] for r in res])
    ST = np.stack([r["sim"][:, 1] for r in res])
    nC = len(cfgs) - 1

    def pooled(P, T, idx=None):
        P, T = (P, T) if idx is None else (P[idx], T[idx])
        t = T[:, :nC].sum(axis=0)
        return np.where(t > 0, P[:, :nC].sum(axis=0) / np.where(t > 0, t, 1) * 1e4, np.nan)

    real_bps, sim_bps = pooled(RP, RT), pooled(SPn, ST)
    rho = _spearman(real_bps, sim_bps)
    rng = np.random.default_rng(SEED)
    S = len(res)
    boots = np.array([_spearman(pooled(RP, RT, ix), pooled(SPn, ST, ix)) for ix in (rng.integers(0, S, S) for _ in range(B))])
    ci = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)))
    by_class = {}
    for c in ("major", "alt", "tradfi"):
        ix = np.array([i for i, r in enumerate(res) if cls[r["symbol"]] == c])
        if len(ix) >= 2:
            by_class[c] = {"symbols": int(len(ix)), "rho": _spearman(pooled(RP, RT, ix), pooled(SPn, ST, ix))}
    keys = [rp.param_key(c) for c in cfgs]
    table = []
    for j, k in enumerate(keys):
        table.append({"params": k if j < nC else "live mm_multi (" + k + ")", "real_bps": float(RP[:, j].sum() / max(RT[:, j].sum(), 1e-12) * 1e4),
                      "sim_bps": float(SPn[:, j].sum() / max(ST[:, j].sum(), 1e-12) * 1e4), "real_pnl_usd": float(RP[:, j].sum()),
                      "sim_pnl_usd": float(SPn[:, j].sum()), "real_turnover": float(RT[:, j].sum()), "sim_turnover": float(ST[:, j].sum()),
                      "real_fills": float(sum(r["real"][j, 2] for r in res)), "sim_fills": float(sum(r["sim"][j, 2] for r in res)),
                      "real_pos_symbols": int((RP[:, j] > 0).sum())})
    order = np.argsort(-real_bps)
    verdict = "supports" if rho > 0.5 and ci[0] > 0 else "contradicts" if ci[1] < 0.5 else "inconclusive"
    out = {"symbols": S, "grid": nC, "fill_model": fm, "spearman": rho, "ci": ci, "by_class": by_class, "table": table,
           "best_real": [table[i] for i in order[:5]], "worst_real": [table[i] for i in order[-3:]],
           "rank_corr_turnover": _spearman(RT[:, :nC].sum(axis=0), ST[:, :nC].sum(axis=0)),
           "rank_corr_fills": _spearman(np.array([t["real_fills"] for t in table[:nC]]), np.array([t["sim_fills"] for t in table[:nC]])),
           "symbols_list": [r["symbol"] for r in res], "verdict": verdict}
    (cache / "sim4.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    print("spearman", rho, ci, by_class, verdict, flush=True)
    for t in out["best_real"]:
        print(t, flush=True)
    return out


# ------------------------------------------------------------- market-microstructure invariance (descriptive, H-SIM3 section)

def _loglog(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return {"n": int(len(x)), "slope": float("nan"), "se": float("nan")}
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    A = np.linalg.inv(X.T @ X)
    V = A @ (X.T * e ** 2) @ X @ A * len(x) / (len(x) - 2)  # HC1
    return {"n": int(len(x)), "slope": float(b[1]), "se": float(math.sqrt(V[1, 1])), "r2": float(1 - e @ e / ((y - y.mean()) @ (y - y.mean())))}


def stage_invariance(cache: Path) -> dict:
    """Kyle-Obizhaeva (2016) invariance: with trading activity W = sigma * P * V (daily return sigma x dollar volume), bet size
    scales as W^(1/3), the number of bets as W^(2/3), and spreads / impact in bps as (sigma^2 / (P V))^(1/3). Prints are not
    bets (one bet is split into many prints), so the trade-level exponents are a rough check only."""
    import pickle
    meta = load_meta(cache)
    rows = []
    for s in symbols_in(cache):
        g = load_grid(cache / f"{s}.parquet")
        segs = g.segments(100)
        if not segs:
            continue
        r = np.concatenate([np.diff(np.log(g.mid[a:b])) for a, b in segs])
        ok = g.ok
        n = (g.n_buy + g.n_sell)[ok]
        V = (g.buy + g.sell)[ok]
        sig_d = float(np.std(r) * math.sqrt(28_800))
        pv = float(V.mean() * 28_800)
        tr = n > 0
        row = {"symbol": s, "class": asset_class(s, (meta.get(s) or {}).get("symbol_type")), "sigma_day": sig_d, "dollar_vol_day": pv,
               "W": sig_d * pv, "illiq": (sig_d ** 2 / pv) ** (1 / 3) if pv > 0 else float("nan"),
               "trades_day": float(n.mean() * 28_800), "trade_notional_med": float(np.median(V[tr] / n[tr])) if tr.any() else float("nan"),
               "spread_bps_med": float(np.nanmedian(g.spread_bps[ok])), "depth25_med": float(np.nanmedian(g.depth[ok]))}
        mp = cache / "models" / f"{s}.pkl"
        if mp.exists():
            m = pickle.loads(mp.read_bytes())
            row["impact_10k_bps"] = float(m.ret[1] * math.sqrt(10_000 / m.depth) + m.ret[0] * 10_000 / m.depth)
        rows.append(row)
    T = pl.DataFrame(rows)
    lw, li = np.log(T["W"].to_numpy()), np.log(T["illiq"].to_numpy())
    out = {"symbols": T.height,
           "trade_size_vs_W": {**_loglog(lw, np.log(T["trade_notional_med"].to_numpy())), "invariance": 1 / 3},
           "trades_vs_W": {**_loglog(lw, np.log(T["trades_day"].to_numpy())), "invariance": 2 / 3},
           "spread_vs_illiq": {**_loglog(li, np.log(T["spread_bps_med"].to_numpy())), "invariance": 1.0},
           "depth_vs_W": {**_loglog(lw, np.log(T["depth25_med"].to_numpy())), "invariance": None}}
    if "impact_10k_bps" in T.columns:
        out["impact10k_vs_illiq"] = {**_loglog(li, np.log(T["impact_10k_bps"].to_numpy())), "invariance": 1.0}
    T.write_parquet(cache / "invariance.parquet")
    (cache / "invariance.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    print(json.dumps(out, default=float), flush=True)
    return out


# ------------------------------------------------------------------------------------------ report + record

REPORT = Path("research/reports/simulator_2026-11-11")


def _f(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(c, nd=3):
    return f"[{_f(c[0], nd)}, {_f(c[1], nd)}]"


def load_results(cache: Path) -> dict:
    out = {}
    for k in ("sim1", "sim2", "sim3", "sim4", "invariance"):
        p = cache / f"{k}.json"
        out[k] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


def stage_report(cache: Path, as_of: datetime, coverage: dict | None, storage: dict, rules_yaml: str, extra_md: str = "") -> dict:
    R = load_results(cache)
    s1, s2, s3, s4, inv = R["sim1"], R["sim2"], R["sim3"], R["sim4"], R["invariance"]
    doc = {"as_of": as_of.isoformat(), "segment": "clean_holdout", "method": METHOD, "coverage": coverage, "storage_estimate": storage, **R}
    REPORT.with_suffix(".json").write_text(json.dumps(doc, default=float, indent=1), encoding="utf-8")
    L = []
    w = L.append
    w("# Execution simulator v0: fill model, reaction functions, flow generator, replay (H-SIM1..H-SIM4)")
    w("")
    w(f"_as of {as_of:%Y-%m-%d %H:%M} UTC (flow_3s read point in time at this timestamp) | segment: clean_holdout only | "
      "code: src/ai_trader/sim/, src/ai_trader/research/simulator_study.py | tests: tests/test_sim.py_")
    w("")
    w("All data is after the knowledge cutoff (2026-07-01), so every verdict is clean_holdout. It is a short first sample: "
      "about three days of collector data with gaps (one market state: a crypto sell-off), and our own live quotes from "
      "about 15 hours on 2026-11-08/09. Treat every number as a first measurement, not a settled property.")
    w("")
    w("## Verdicts")
    w("")
    w("| id | test (as registered) | key numbers | verdict |")
    w("|---|---|---|---|")
    if s1:
        pv = s1["variants"][s1["primary_variant"]]
        w(f"| H-SIM1 | fill rate and 60 s markout of held-out live quotes inside their 95% CIs | variant {s1['primary_variant']}: predicted fill rate "
          f"{_f(pv['pred_fill_rate'] * 100, 2)}% vs observed {_f(s1['test']['obs_fill_rate'] * 100, 2)}% CI {_ci([c * 100 for c in s1['test']['ci_fill_rate']], 2)}; "
          f"markout {_f(pv['pred_markout_bps'], 2)} bp vs {_f(s1['test']['obs_markout_bps'], 2)} bp CI {_ci(s1['test']['ci_markout_bps'], 2)}; "
          f"test fills {s1['test']['fills']} | **{s1['verdict']}** |")
    if s2:
        cl = s2["classes"]
        w(f"| H-SIM2 | >= 4 of 5 stylised facts inside the day-to-day envelope, each class | facts inside: "
          + ", ".join(f"{c} {v['n_inside']}/5" for c, v in cl.items()) + f" | **{s2['verdict']}** |")
    if s3:
        ss = s3["sign_stability_3s"]
        w(f"| H-SIM3 | sign stable on >= 80% of symbol-days and equal scaled coefficients within >= 2 of 3 classes | sign shares: momentum "
          f"{_f(ss['momentum']['share'], 2)}, obi {_f(ss['obi']['share'], 2)}, spread {_f(ss['spread']['share'], 2)} ({s3['symbol_days_3s']} symbol-days); "
          f"classes not rejected {s3['classes_not_rejected']}/3 | **{s3['verdict']}** |")
    if s4:
        w(f"| H-SIM4 | Spearman > 0.5 with CI excluding 0 (36-set grid) | rho {_f(s4['spearman'], 2)}, CI {_ci(s4['ci'], 2)}, {s4['symbols']} symbols | **{s4['verdict']}** |")
    w("")
    w(extra_md)
    REPORT.with_suffix(".md").write_text("\n".join(L) + "\n\n## Rule candidates\n\n```yaml\n" + rules_yaml.strip() + "\n```\n", encoding="utf-8")
    return doc


def stage_record(cache: Path) -> list:
    from ai_trader.research import hypotheses as hyp
    R = load_results(cache)
    data = "flow_3s (Bybit linear, as of 2026-11-10T23:00Z) + data/mm/analysis/mainnet-* live records; report research/reports/simulator_2026-11-11.md"
    done = []
    s1, s2, s3, s4 = R["sim1"], R["sim2"], R["sim3"], R["sim4"]
    items = []
    if s1:
        pv = s1["variants"][s1["primary_variant"]]
        items.append(("H-SIM1", s1["verdict"], {
            "variant": s1["primary_variant"], "theta0": pv["params"]["theta0"], "b": pv["params"]["b"], "eps": pv["params"]["eps"],
            "pred_fill_rate": pv["pred_fill_rate"], "obs_fill_rate": s1["test"]["obs_fill_rate"], "ci_fill_rate": s1["test"]["ci_fill_rate"],
            "pred_markout_bps": pv["pred_markout_bps"], "obs_markout_bps": s1["test"]["obs_markout_bps"], "ci_markout_bps": s1["test"]["ci_markout_bps"],
            "test_quotes": s1["test"]["quotes"], "test_fills": s1["test"]["fills"], "train_fills": s1["train"]["fills"],
            "naive_theta1_fill_rate": pv["naive_theta1_fill_rate"], "usable_share": s1["usable_share"]},
            "Literal criterion: both predictions inside the observed CIs. The study fixed a minimum of 30 held-out fills before the run; "
            "the test half has fewer, so the record is inconclusive unless that minimum is met. Short first sample, one market state."))
    if s2:
        items.append(("H-SIM2", s2["verdict"], {c: {"n_inside": v["n_inside"], "facts": v["facts_inside"], "jump_variant_n_inside": v["jump_variant"]["n_inside"]}
                                                for c, v in s2["classes"].items()} | {"symbols": s2["symbols"]},
                      "Generator fitted on 2026-11-08/09, validated on 2026-11-10; envelope from 3 real days (1 h block bootstrap). Short first sample."))
    if s3:
        items.append(("H-SIM3", s3["verdict"], {"sign_share": {k: v["share"] for k, v in s3["sign_stability_3s"].items()},
                                                "modal_sign": {k: v["modal_sign"] for k, v in s3["sign_stability_3s"].items()},
                                                "wald_joint_p": {c: v["joint"]["p"] for c, v in s3["wald"].items()},
                                                "symbol_days": s3["symbol_days_3s"]},
                      "Per symbol-day 3 s regressions, 5-minute clustered SE; scaled by sigma and depth. Short first sample."))
    if s4:
        items.append(("H-SIM4", s4["verdict"], {"spearman": s4["spearman"], "ci": s4["ci"], "symbols": s4["symbols"], "grid": s4["grid"],
                                                "by_class": s4["by_class"]},
                      "Replay on real 2026-11-10 paths vs H-SIM2 generated paths, same fill model (H-SIM1). Short first sample."))
    for hid, verdict, metrics, note in items:
        if hyp.results(hid):
            print(hid, "already recorded; skipping", flush=True)
            continue
        hyp.record(hid, "clean_holdout", verdict, metrics, note=note, data=data)
        done.append(hid)
    print("recorded", done, flush=True)
    return done


FACT_LABELS = {"kurtosis": "excess kurtosis of 3 s returns", "absr_acf": "mean ACF of |r|, lags 1-60", "flow_acf": "mean ACF of signed flow, lags 1-20",
               "impact": "impact slope corr(r_t, F_t)", "spread_med": "spread median (bp)", "spread_p90": "spread p90 (bp)",
               "continuation": "continuation ratio (extra, not registered)"}
COMP_LABELS = {"dir.r1": "momentum 3 s (lag 1)", "dir.r2_5": "momentum lags 2-5", "dir.r6_20": "momentum lags 6-20",
               "dir.obi5": "book imbalance obi_5", "dir.y1": "flow persistence lag 1", "dir.y2_5": "flow persistence lags 2-5",
               "dir.vol": "signed flow on recent vol", "int.log_spread": "intensity on log spread", "int.vol": "intensity on recent vol",
               "var_flow_to_ret": "VAR: cum. flow response to a 1-sigma return", "var_ret_to_flow": "VAR: cum. return response to a unit flow shock",
               "var_flow_persist": "VAR: cum. flow response to a flow shock"}


def detail_md(R: dict, coverage: dict | None, storage: dict, as_of: datetime) -> str:
    s1, s2, s3, s4, inv = R["sim1"], R["sim2"], R["sim3"], R["sim4"], R["invariance"]
    L = []
    w = L.append
    w("## Data and coverage")
    w("")
    if coverage:
        w("flow_3s candles per symbol and UTC day (a full day is 28,800 three-second candles); collector gaps are never filled:")
        w("")
        w("| day | symbols | median candles | max candles | share of a full day (median / max) |")
        w("|---|---|---|---|---|")
        for d, v in coverage.items():
            w(f"| {d} | {v['symbols']} | {v['median']:.0f} | {v.get('max', v['median']):.0f} | {v['median'] / 28800:.0%} / {v.get('max', v['median']) / 28800:.0%} |")
        w("")
        w("On 2026-11-08 the 50 original symbols have about 85% of the day; the 54 symbols added to the collector at about 21:00 UTC have 3 hours.")
        w("")
    if s1:
        w(f"Live records: {len(s1['live']['dirs'])} overlapping pulls deduplicated by orderId / execId: {s1['live']['mm_quotes']} mm quotes "
          f"({s1['live']['mm_filled']} filled), {s1['live']['mm_fills']} maker executions; {s1['rejected_post_only_or_other']} post-only rejects dropped "
          f"(they never rested). Quote coverage by flow_3s: {s1['usable']} of {s1['quotes_considered']} usable ({s1['usable_share']:.0%}; reasons {s1['coverage']}). "
          "The collector covered both quoting sessions (2026-11-08 02:20-11:21 on SOL/XRP/ADA, 2026-11-08 22:59 to 2026-11-09 08:15 on up to 50 symbols).")
        w("")
    w("## Methods")
    w("")
    for k, v in METHOD.items():
        w(f"- **{k}**: {v}")
    w("")
    # ---------------------------------------------------------------- H-SIM1
    if s1:
        w("## H-SIM1: passive fill model")
        w("")
        w(f"Split at {s1['split_utc'][:19]} UTC (median creation time of usable quotes). Train: {s1['train']['quotes']} quotes, {s1['train']['fills']} fills; "
          f"test: {s1['test']['quotes']} quotes, {s1['test']['fills']} fills ({s1['test']['blocks']} half-hour blocks).")
        w("")
        w("| variant | theta0 (touch) | b per bp | eps | train AIC | test log-lik | train fill rate pred / obs | test fill rate pred | test markout pred (bp) | naive theta=1 fill rate | inside both CIs |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for v, d in s1["variants"].items():
            p = d["params"]
            w(f"| {v} | {_f(p['theta0'], 3)} | {_f(p['b'], 1)} | {p['eps']:.1e} | {_f(d['fit']['aic'], 1)} | {_f(d['test_ll'], 1)} | "
              f"{_f(d['train_pred_fill_rate'] * 100, 2)}% / {_f(d['train_obs_fill_rate'] * 100, 2)}% | {_f(d['pred_fill_rate'] * 100, 2)}% | {_f(d['pred_markout_bps'], 2)} | "
              f"{_f(d['naive_theta1_fill_rate'] * 100, 2)}% | {d['rate_inside'] and d['markout_inside']} |")
        w("")
        w(f"Observed on the test half: fill rate {_f(s1['test']['obs_fill_rate'] * 100, 2)}% (95% CI {_ci([c * 100 for c in s1['test']['ci_fill_rate']], 2)}), "
          f"60 s markout {_f(s1['test']['obs_markout_bps'], 2)} bp (CI {_ci(s1['test']['ci_markout_bps'], 2)}).")
        w("")
        w("Reading: a strict trade-through always reaches our order (b at its upper bound: any penetration of a fraction of a bp fills), and a "
          f"candle that only touches our price fills us with probability theta0 = {_f(s1['variants']['touch']['params']['theta0'], 2)} (queue position). "
          "The touch variant explains the six-fold drop in fill rate between the halves (5.4% to 0.9%) from the 3 s candles alone. "
          "The naive model (every touch fills) over-predicts the held-out fill rate, outside the CI. The strict variant is rejected by AIC: "
          "touch-only fills are common, and its leak term puts fill probability on candles far from the quote. "
          f"Self-print check: in {s1['self_print_share']:.1%} of fill candles the aggressive notional equalled our own fill, so our own prints "
          "rarely create the touch that the model then explains.")
        w("")
        w(f"Verdict: **{s1['verdict']}**. Both predictions are inside the observed CIs (the literal success criterion), but the held-out half has "
          f"only {s1['test']['fills']} fills, below the 30-fill minimum fixed in the study code before the run; the markout CI is about "
          f"{_f(s1['test']['ci_markout_bps'][1] - s1['test']['ci_markout_bps'][0], 0)} bp wide, so the markout test has little power.")
        w("")
    # ---------------------------------------------------------------- H-SIM3
    if s3:
        w("## H-SIM3: reaction functions of aggressive flow (implicit trader types)")
        w("")
        w(f"{s3['symbol_days_3s']} symbol-days (3 s regressions, {s3['symbols']} symbols, days {', '.join(s3['days'])}); classes: "
          + ", ".join(f"{c} {n}" for c, n in s3["classes"].items()) + f". Median R^2: direction {_f(s3['r2_dir_median'], 3)}, intensity {_f(s3['r2_int_median'], 3)}.")
        w("")
        w("Sign stability of the primary responses (share of symbol-days with the modal sign):")
        w("")
        w("| response | all (3 s) | modal sign | major | alt | TradFi | all (1 m) |")
        w("|---|---|---|---|---|---|---|")
        for k in ("momentum", "obi", "spread"):
            a = s3["sign_stability_3s"][k]
            row = [f"{_f(s3['sign_by_class_3s'][c][k]['share'], 2)} ({'+' if s3['sign_by_class_3s'][c][k]['modal_sign'] > 0 else '-'})" for c in ("major", "alt", "tradfi")]
            m1 = s3["sign_stability_1m"][k]
            w(f"| {k} | {_f(a['share'], 2)} | {'+' if a['modal_sign'] > 0 else '-'} | {' | '.join(row)} | {_f(m1['share'], 2)} ({'+' if m1['modal_sign'] > 0 else '-'}) |")
        w("")
        w("Reaction-function table by class (3 s; median coefficient, share positive, share with t > 2 / t < -2). Flow is in units of the day's "
          "depth within 25 bp, returns in units of the day's 3 s sigma:")
        w("")
        w("| component | class | median | share + | t>2 | t<-2 |")
        w("|---|---|---|---|---|---|")
        for c in ("major", "alt", "tradfi"):
            for row in s3["components"]:
                if row["class"] == c and row["freq"] == "3s":
                    w(f"| {COMP_LABELS.get(row['component'], row['component'])} | {c} | {row['median']:.2e} | {_f(row['share_pos'], 2)} | "
                      f"{_f(row['sig_pos'], 2)} | {_f(row['sig_neg'], 2)} |")
        w("")
        w("Equality of scaled coefficients within class (per symbol pooled over days; stacked 5-minute-cluster influence functions; Wald):")
        w("")
        w("| class | symbols | joint W (df), p | momentum p | obi p | spread p | momentum median [IQR] | obi median [IQR] | spread median [IQR] |")
        w("|---|---|---|---|---|---|---|---|---|")
        for c, v in s3["wald"].items():
            sc = v["scaled_coef"]
            w(f"| {c} | {v['joint']['symbols']} | {_f(v['joint']['W'], 0)} ({v['joint']['df']}), {v['joint']['p']:.1e} | {v['momentum']['p']:.1e} | {v['obi']['p']:.1e} | "
              f"{v['spread']['p']:.1e} | {sc['momentum']['median']:.2e} [{sc['momentum']['iqr'][0]:.1e}, {sc['momentum']['iqr'][1]:.1e}] | "
              f"{sc['obi']['median']:.2e} [{sc['obi']['iqr'][0]:.1e}, {sc['obi']['iqr'][1]:.1e}] | {_f(sc['spread']['median'], 3)} [{_f(sc['spread']['iqr'][0], 2)}, {_f(sc['spread']['iqr'][1], 2)}] |")
        w("")
        w("Trader types read from the stable signs:")
        w("")
        w("- **Short-horizon momentum takers** (flow follows the last 3 s to 1 min of returns): stable in majors and alts, weak in TradFi perps. "
          "This is the most transferable component across crypto.")
        w("- **Order splitting / herding** (signed flow persists at lag 1): positive on most symbol-days at 3 s, gone or reversed at 1 m.")
        w("- **Liquidity-seeking takers** (obi_5 has a NEGATIVE coefficient): takers sell into the deeper bid side and buy the deeper ask. "
          "They do not chase book imbalance; book imbalance is where liquidity is consumed.")
        w("- **Activity followers** (trade intensity rises with recent volatility): the most stable component of all.")
        w("- **Spread response**: intensity falls when the spread widens in alts and TradFi (liquidity-sensitive takers) but rises in majors, "
          "where a wider TWA spread is the footprint of a burst of aggressive trading (reverse causality at 3 s).")
        w("")
        w("Signs are mostly stable at the class level, but the magnitudes are not interchangeable: with about 580 five-minute clusters the "
          "equal-coefficient Wald test rejects in every class, and the within-class IQRs span a factor of 2-10. A transfer to another "
          "market therefore needs that market's own coefficients (or at least its class median with an explicit dispersion), not a "
          "universal scaled reaction function.")
        w("")
    if inv:
        w("### Market-microstructure invariance check (Kyle-Obizhaeva 2016; descriptive)")
        w("")
        w("Cross-symbol log-log slopes (HC1 SE); W = daily sigma x daily dollar volume; illiquidity = (sigma^2 / dollar volume)^(1/3). "
          "Prints are not bets (a bet is split into many prints), so the trade-level slopes are only indicative.")
        w("")
        w("| relation | slope (SE) | invariance prediction | R^2 | n |")
        w("|---|---|---|---|---|")
        for k, lab in (("trade_size_vs_W", "median trade notional vs W"), ("trades_vs_W", "trades per day vs W"),
                       ("spread_vs_illiq", "median spread (bp) vs illiquidity"), ("impact10k_vs_illiq", "model impact of a 10k USD order (bp) vs illiquidity"),
                       ("depth_vs_W", "depth within 25 bp vs W")):
            if k in inv:
                v = inv[k]
                w(f"| {lab} | {_f(v['slope'], 2)} ({_f(v['se'], 2)}) | {_f(v['invariance'], 2) if v.get('invariance') is not None else '-'} | {_f(v.get('r2'), 2)} | {v['n']} |")
        w("")
        w("Reading: the activity exponents match invariance closely (trade notional about W^(1/3), number of prints about W^(2/3)), "
          "so W is a good single scale for transferring trade size and intensity to a market we have not recorded. Spreads and "
          "model impact fall faster with liquidity than invariance predicts (slopes about 1.4-1.7 vs 1): tick-size floors keep "
          "liquid perps at one tick, and illiquid ones pay more. Use per-class intercepts and these empirical slopes, not the "
          "theoretical ones, when rescaling the generator to a new market.")
        w("")
    # ---------------------------------------------------------------- H-SIM2
    if s2:
        w("## H-SIM2: flow generator vs held-out stylised facts")
        w("")
        w(f"Fitted per symbol on {', '.join(s2['fit_days'])}, validated on {s2['val_day']} ({s2['symbols']} symbols; skipped: {len(s2['skipped'])}). "
          "Class value = median over symbols; the envelope is the range over the three real days of the 95% bootstrap interval of the class median.")
        w("")
        keys = list(FACT_LABELS)
        for c, v in s2["classes"].items():
            w(f"**{c}** ({v['symbols']} symbols): {v['n_inside']}/5 registered facts inside; jump variant {v['jump_variant']['n_inside']}/5 "
              f"(median jump probability per nonzero-return candle {v['jump_variant']['jump_lam_median']:.4f}).")
            w("")
            w("| fact | simulated | jump variant | envelope | real by day (" + ", ".join(d[5:] for d in v["real_by_day"]) + ") | inside |")
            w("|---|---|---|---|---|---|")
            for j, k in enumerate(keys):
                real = ", ".join(_f(x[j], 3) for x in v["real_by_day"].values())
                inside = (v["sim"][j] >= v["env_lo"][j]) and (v["sim"][j] <= v["env_hi"][j])
                w(f"| {FACT_LABELS[k]} | {_f(v['sim'][j], 3)} | {_f(v['jump_variant']['sim'][j], 3)} | [{_f(v['env_lo'][j], 3)}, {_f(v['env_hi'][j], 3)}] | {real} | {inside} |")
            w("")
        if s2.get("symbol_pass_share"):
            w("Share of symbols whose own simulated fact lies inside their own envelope:")
            w("")
            w("| class | " + " | ".join(FACT_LABELS[k] for k in keys) + " |")
            w("|---|" + "---|" * len(keys))
            for c, v in s2["symbol_pass_share"].items():
                w(f"| {c} | " + " | ".join(_f(v[k], 2) for k in keys) + " |")
            w("")
        w("Reading: the generator gets the level facts (spread distribution, and kurtosis in majors and TradFi) and, for majors, the "
          "volatility clustering and impact, but it fails the dependence facts everywhere: signed-flow autocorrelation comes out at "
          "about 0 (real 0.002-0.044) because the heavy-tailed notional of each candle is drawn independently of the previous "
          "candle's direction (only the buy share q is persistent, and F = q V is dominated by V); |r| autocorrelation is too low for "
          "alts and TradFi; impact is too strong for alts and TradFi (the fit on days 1-2 is not stable into day 3); and the "
          "continuation ratio of price changes is 0.50 (real 0.53-0.57 in crypto), the price-level footprint of the momentum takers "
          "of H-SIM3 that the generator does not carry into the mid. The compound-Poisson jump variant raises kurtosis and does not "
          "help any class. A v1 should model events (marked Hawkes on prints with the reaction functions as cross-excitation) "
          "rather than 3 s aggregates.")
        w("")
        w("Fitted generator parameters (class medians):")
        w("")
        mk = [k for k in next(iter(s2["classes"].values()))["model_median"] if k != "act_persist"]  # act_persist: Hawkes-only field
        w("| class | " + " | ".join(mk) + " |")
        w("|---|" + "---|" * len(mk))
        for c, v in s2["classes"].items():
            w(f"| {c} | " + " | ".join(_f(v["model_median"][k], 3) for k in mk) + " |")
        w("")
    # ---------------------------------------------------------------- H-SIM4
    if s4:
        w("## H-SIM4: quoting-parameter rankings, replay vs simulator")
        w("")
        w(f"{s4['symbols']} mm_multi symbols, {s4['grid']} parameter sets, fill model {s4['fill_model']['variant']} (theta0 {_f(s4['fill_model']['theta0'], 2)}). "
          f"Spearman of the pooled net P&L per turnover rankings: **{_f(s4['spearman'], 2)}** (95% CI over symbols {_ci(s4['ci'], 2)}); "
          "by class: " + ", ".join(f"{c} {_f(v['rho'], 2)} ({v['symbols']} symbols)" for c, v in s4["by_class"].items())
          + f". Rank agreement on turnover {_f(s4['rank_corr_turnover'], 2)}, on fill counts {_f(s4['rank_corr_fills'], 2)}.")
        w("")
        w("Best and worst parameter sets on the real 2026-11-10 replay (net of fees, flattened at the end of the path):")
        w("")
        w("| parameters (ask, bid, take-profit, requote; sigma units) | real bp of turnover | sim bp | real P&L USD | real turnover USD | real fills | symbols with P&L > 0 |")
        w("|---|---|---|---|---|---|---|")
        for t in s4["best_real"] + s4["worst_real"]:
            w(f"| {t['params']} | {_f(t['real_bps'], 1)} | {_f(t['sim_bps'], 1)} | {_f(t['real_pnl_usd'], 2)} | {_f(t['real_turnover'], 0)} | {_f(t['real_fills'], 0)} | {t['real_pos_symbols']} |")
        live = s4["table"][-1]
        w(f"| {live['params']} | {_f(live['real_bps'], 1)} | {_f(live['sim_bps'], 1)} | {_f(live['real_pnl_usd'], 2)} | {_f(live['real_turnover'], 0)} | {_f(live['real_fills'], 0)} | {live['real_pos_symbols']} |")
        w("")
        w("Reading: the simulator ranks ACTIVITY almost perfectly (turnover and fill counts, rank correlation about 0.98) but not P&L. "
          "Generated paths have no drift by construction, while 2026-11-10 was a sell-off: on the real day the sets with a tight ask "
          "and a wide bid (the bearish distance skew) earn, and wide asks lose; on generated paths every set loses a few bp and the "
          "order is noise. P&L rankings of a skewed quoter are a bet on the day's drift, which a stationary generator cannot "
          "reproduce. The generator also fills about twice as often as the real paths (flatter, more mean-reverting sim paths "
          "touch the quotes more).")
        w("")
    # ---------------------------------------------------------------- findings / caveats / data
    w("## Finding for the live quoter (crates/atx/src/mm.rs, not edited)")
    w("")
    w("The fast sigma never activates: `sigma_1h(&closes[closes.len() - sigma_fast_min..])` passes the last 30 closes (29 returns) to a "
      "function that returns the floor when it has fewer than 30 returns. With sigma_fast_min = 30 the fast estimate is always the 20 bp "
      "floor, so `max(slow, fast)` is the 4 h sigma (or the floor) and quotes do not widen on a 30-minute volatility burst as the config "
      "comment intends. The Python port reproduces this behaviour on purpose. A fix (pass 31 closes, or lower the minimum) is the user's call.")
    w("")
    w("## Caveats")
    w("")
    w("- Three days of collector data with gaps and one market state (a crypto sell-off; 2026-11-08 is a Sunday for TradFi perps). "
      "Every verdict is clean_holdout and a first sample.")
    w("- The fill model sees 3 s candle highs and lows, not the queue: theta0 is an average queue position for our minimum-size quotes; "
      "larger quotes would queue differently. Held-out fills: 23.")
    w("- The replay is one symbol at a time (portfolio net / gross caps never bind), without structural levels, with the book top "
      "rebuilt from the 3 s mid and TWA spread, and with decisions every 9 s (the live loop over 50 symbols was slower). "
      "Our own historical quotes are in the real data, but at minimum size their footprint is negligible (self-print check above).")
    w("- The generator has no cross-symbol dependence, no intraday seasonality and no feedback from returns to trade activity; "
      "activity persistence is capped at a 1 h half-life, so day-level regime shifts are not simulated within a path.")
    w("- Reaction-function regressions are reduced-form: lagged flow and returns are correlated with unobserved order splitting, "
      "so 'trader types' are statistical components, not identified agents.")
    w("- The Spearman test pools P&L over symbols; a handful of volatile symbols dominate turnover.")
    w("")
    w("## What data would make v1 much better")
    w("")
    w(storage["text"])
    w("")
    return "\n".join(L)


def coverage_table(cache: Path) -> dict:
    per = {}
    for s in symbols_in(cache):
        g = load_grid(cache / f"{s}.parquet")
        for d in ALL_DAYS:
            n = int((g.present & day_mask(g, d)).sum())
            if n:
                per.setdefault(d, []).append(n)
    return {d: {"symbols": len(v), "median": float(np.median(v)), "max": float(np.max(v))} for d, v in sorted(per.items())}


def storage_estimate(as_of: datetime) -> dict:
    """Raw trades and book-update counts per day from flow_3s (n_buy + n_sell, book_updates), bytes per trade from trades_raw."""
    import glob
    import os

    from ai_trader.storage.lake import Lake
    lake = Lake()
    q = f"""SELECT symbol, count(*) c, sum(n_buy + n_sell) trades, sum(book_updates) bu
            FROM read_parquet('{lake._glob('flow_3s')}', hive_partitioning = true, union_by_name = true)
            WHERE category = 'linear' AND known_ts <= ? GROUP BY 1"""
    d = pl.from_arrow(lake.con.sql(q, params=[as_of]).to_arrow_table()).with_columns(
        (pl.col("trades").cast(pl.Float64) / pl.col("c") * 28_800).alias("trades_day"), (pl.col("bu").cast(pl.Float64) / pl.col("c") * 28_800).alias("bu_day"))
    d = d.sort("trades_day", descending=True)
    files = glob.glob(str(lake.root / "trades_raw" / "**" / "*.parquet"), recursive=True)
    nraw = int(lake.con.sql(f"SELECT count(*) FROM read_parquet('{lake._glob('trades_raw')}', hive_partitioning = true)").fetchall()[0][0]) if files else 0
    b_trade = sum(os.path.getsize(f) for f in files) / max(nraw, 1) if nraw else 25.0
    f3 = sum(os.path.getsize(f) for f in glob.glob(str(lake.root / "flow_3s" / "**" / "*.parquet"), recursive=True))
    lake_b = sum(os.path.getsize(f) for f in glob.glob(str(lake.root / "**" / "*"), recursive=True) if os.path.isfile(f))
    mm = yaml_symbols()
    t50 = float(d.filter(pl.col("symbol").is_in(mm))["trades_day"].sum())
    t_all = float(d["trades_day"].sum())
    reps = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT", "ENAUSDT", "MAVIAUSDT", "WIFUSDT", "TSLAUSDT", "NVDAUSDT", "XAUUSDT", "CLUSDT"]
    bu12 = float(d.filter(pl.col("symbol").is_in(reps))["bu_day"].sum())
    gb = 1e9
    b_book_lo, b_book_hi = 40.0, 120.0   # bytes per compressed book-delta message (assumption to verify with a 1-day pilot)
    weeks = 4
    trades_gb = t50 * b_trade * 7 * weeks / gb
    book_lo, book_hi = bu12 * b_book_lo * 7 * weeks / gb, bu12 * b_book_hi * 7 * weeks / gb
    flow_gb_day = f3 / gb / 3.3
    out = {"bytes_per_trade": b_trade, "trades_day_all": t_all, "trades_day_mm50": t50, "book_updates_day_12": bu12, "flow3s_gb_per_day": flow_gb_day,
           "lake_gb_now": lake_b / gb, "trades_gb_4w_mm50": trades_gb, "book_gb_4w_12_lo": book_lo, "book_gb_4w_12_hi": book_hi, "rep_symbols": reps}
    total_lo = lake_b / gb + flow_gb_day * 7 * weeks + trades_gb + book_lo
    total_hi = lake_b / gb + flow_gb_day * 7 * weeks + trades_gb + book_hi
    out["text"] = (
        f"The v0 bottlenecks are (1) the fill model sees candles, not queues; (2) 23 held-out live fills; (3) three gappy days. "
        f"Proposed v1 data, for the user to decide (storage budget 30 GB, lake today {lake_b / gb:.2f} GB):\n\n"
        f"| item | scope | estimate for {weeks} weeks |\n|---|---|---|\n"
        f"| raw trades (trades_raw: ts, side, price, size, id) | the 50 mm_multi symbols, {t50 / 1e6:.1f} M prints/day at {b_trade:.0f} B/print (measured on BTC trades_raw) | {trades_gb:.1f} GB |\n"
        f"| L2 book deltas (orderbook.50 or .200 updates, levels within 25-100 bp) | 12 representative symbols (4 majors, 4 alts, 4 TradFi: {', '.join(reps)}), "
        f"{bu12 / 1e6:.1f} M update messages/day (the collector's book_updates count) at an assumed {b_book_lo:.0f}-{b_book_hi:.0f} B/message | {book_lo:.1f}-{book_hi:.1f} GB |\n"
        f"| flow_3s as today (all symbols) | ongoing, {flow_gb_day:.2f} GB/day | {flow_gb_day * 7 * weeks:.1f} GB |\n"
        f"| total incl. the current lake | | {total_lo:.0f}-{total_hi:.0f} GB |\n\n"
        "With raw trades and book deltas the fill model can track queue position (our size vs the size ahead at our price, depletion "
        "by prints at that price, cancels), the generator can be a proper marked Hawkes on events instead of 3 s aggregates, and "
        "reaction functions can be estimated on event time. The book-delta byte cost is an assumption: measure it with a one-day pilot "
        "on 2-3 symbols before committing, and keep the deltas on a rolling retention (like trades_raw) after the derived features are "
        "stored. A further cheap gain: run the live quoter with a few deliberate parameter variants (paired, same symbols) to collect "
        "hundreds of fills per variant, which is what H-SIM1 and H-SIM4 lack most.")
    return out


def yaml_symbols() -> list[str]:
    import yaml
    return yaml.safe_load(Path("config/mm_multi.yaml").read_text(encoding="utf-8"))["symbols"]


def build_rules(R: dict) -> str:
    """Rule candidates in the research/rules.yaml schema (never prescribing a position size)."""
    s1, s3, s4 = R["sim1"], R["sim3"], R["sim4"]
    th = s1["variants"]["touch"]["params"]["theta0"] if s1 else float("nan")
    ss = s3["sign_stability_3s"] if s3 else {}
    sbc = s3["sign_by_class_3s"] if s3 else {}
    rho = s4["spearman"] if s4 else float("nan")
    ci = s4["ci"] if s4 else [float("nan")] * 2
    best = s4["best_real"][0] if s4 else {}
    live = s4["table"][-1] if s4 else {}
    rules = f"""
- id: R-SIM-1
  kind: expect
  rule: "WHEN estimating whether a resting post-only quote at the venue minimum quantity gets filled THEN expect a fill on almost every 3 s candle that trades strictly through its price and on only about {th:.0%} of candles that just touch it"
  when: "passive quote at the minimum quantity on a Bybit linear perp; 3 s candle low <= bid (high >= ask)"
  then: "fill probability per candle ~1 for a strict trade-through, ~{th:.2f} for a touch (queue position); never assume every touch fills"
  scope: {{assets: crypto and TradFi perps quoted by atx mm, horizon: "3s-5m", sides: both}}
  effect: "touch model predicts {s1['variants']['touch']['pred_fill_rate'] * 100:.2f}% held-out fill rate vs {s1['test']['obs_fill_rate'] * 100:.2f}% observed (CI {s1['test']['ci_fill_rate'][0] * 100:.2f}-{s1['test']['ci_fill_rate'][1] * 100:.2f}%); every-touch-fills predicts {s1['variants']['touch']['naive_theta1_fill_rate'] * 100:.2f}%"
  confidence: C
  evidence: [H-SIM1, research/reports/simulator_2026-11-11.md]
  review: ">= 150 held-out live fills, or raw trades + book deltas allowing a queue-position model"
  status: candidate
- id: R-SIM-2
  kind: expect
  rule: "WHEN a crypto perp (major or alt) has just moved over the last 3 s to 1 min THEN expect aggressive flow to keep following that move (momentum takers), so a resting quote on the side the price moved toward is the one likely to be hit next"
  when: "sign of the last 3 s - 1 min mid return on BTC/ETH/SOL/XRP/BNB/DOGE or alts"
  then: "expect signed taker flow in the same direction over the next 3 s - 1 min; treat fills on that side as adverse-selection prone"
  scope: {{assets: crypto, horizon: "3s-1m", sides: both}}
  effect: "momentum coefficient > 0 on {sbc.get('major', {}).get('momentum', {}).get('share', float('nan')):.0%} of major and {sbc.get('alt', {}).get('momentum', {}).get('share', float('nan')):.0%} of alt symbol-days (3 s); TradFi perps only {sbc.get('tradfi', {}).get('momentum', {}).get('share', float('nan')):.0%}"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "a month of flow_3s with the sign share staying >= 80% for majors and alts"
  status: candidate
- id: R-SIM-3
  kind: expect
  rule: "WHEN the top-of-book depth within 5 bp is imbalanced THEN expect the next aggressive flow to hit the DEEPER side (sell into a deep bid, buy a deep ask), not to chase the imbalance"
  when: "obi_5 = (bid - ask) / (bid + ask) depth within 5 bp, latest 3 s candle"
  then: "do not read book imbalance as a signal of taker direction; positive obi_5 precedes net taker selling"
  scope: {{assets: all, horizon: "3s", sides: both}}
  effect: "obi coefficient < 0 on {ss.get('obi', {}).get('share', float('nan')):.0%} of {s3['symbol_days_3s'] if s3 else 'n/a'} symbol-days (all classes, 3 s); t < -2 on about a quarter"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "event-time estimates on raw book deltas; a month of flow_3s keeping the negative sign on >= 70% of symbol-days"
  status: candidate
- id: R-SIM-4
  kind: avoid
  rule: "WHEN carrying flow reaction functions or quoting calibrations from one market to another THEN avoid assuming one universal coefficient after scaling by sigma, depth and median trade notional; use the target market's own estimates or its class median with the class dispersion"
  when: "transferring simulator or quoting parameters across symbols"
  then: "re-estimate per symbol (or per class with the IQR as uncertainty); signs transfer within crypto, magnitudes do not"
  scope: {{assets: all, horizon: "3s-1m", sides: both}}
  effect: "Wald test of equal scaled coefficients rejects in every class (p < 1e-5); within-class IQRs span a factor of 2-10"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "a longer sample where the within-class equality is not rejected for >= 2 classes"
  status: candidate
- id: R-SIM-5
  kind: avoid
  rule: "WHEN choosing market-making parameters THEN avoid ranking them on paths from the v0 flow generator; rank them on replays of real 3 s paths with the calibrated fill model"
  when: "parameter search for atx mm (offsets, take-profit, requote threshold)"
  then: "use real-path replay (sim/replay.py) for rankings until a generator passes H-SIM2 and H-SIM4"
  scope: {{assets: all, horizon: "1d", sides: both}}
  effect: "generator failed H-SIM2 (alt 0/5, TradFi 2/5 facts) and its ranking agreement with replay is Spearman {rho:.2f} (CI {ci[0]:.2f} to {ci[1]:.2f})"
  confidence: C
  evidence: [H-SIM2, H-SIM4, research/reports/simulator_2026-11-11.md]
  review: "a v1 generator (event-level data) that passes H-SIM2 in all classes and reaches Spearman > 0.5 with CI excluding 0"
  status: candidate
- id: R-SIM-6
  kind: expect
  rule: "WHEN recent 1-minute volatility on a perp is high THEN expect more aggressive prints in the next seconds and therefore more fills (and more trade-throughs) on resting quotes"
  when: "mean |3 s return| over the last minute above its usual level"
  then: "expect trade intensity to rise; quotes left at the same sigma distance will be reached more often"
  scope: {{assets: all, horizon: "3s-1m", sides: both}}
  effect: "intensity on recent vol > 0 on 88% of symbol-days (t > 2 on about 60%), all classes"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "a month of flow_3s"
  status: candidate
"""
    if s4 and best:
        rules += f"""- id: R-SIM-7
  kind: expect
  rule: "WHEN replaying the atx mm quoter on real 3 s paths of a sell-off day THEN expect net results per turnover to depend mostly on the quote distances, with the best grid set ({best['params']}) at {best['real_bps']:.1f} bp of turnover vs {live['real_bps']:.1f} bp for the live mm_multi settings"
  when: "real-path replay, 2026-11-10, {s4['symbols']} mm_multi symbols, minimum order quantities"
  then: "treat the replay ranking as one day's evidence; confirm any change with paired live variants before adopting it"
  scope: {{assets: all, horizon: "1d", sides: both}}
  effect: "best set {best['real_bps']:.1f} bp ({best['real_pnl_usd']:.2f} USD on {best['real_turnover']:.0f} USD turnover); live settings {live['real_bps']:.1f} bp"
  confidence: C
  evidence: [H-SIM4, research/reports/simulator_2026-11-11.md]
  review: "replays on >= 10 more days and paired live variants"
  status: candidate
"""
    return rules


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-11-10T23:00:00Z")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC)
    cache = Path(a.cache) if a.cache else Path("data/cache/sim") / as_of.strftime("%Y%m%dT%H%M")
    root = Path(".").resolve()
    if a.stage in ("cache", "all"):
        stage_cache(as_of, cache)
    if a.stage in ("sim1", "all"):
        stage_sim1(cache, root)
    if a.stage in ("sim3", "all"):
        stage_sim3(cache)
    if a.stage in ("sim2", "all"):
        stage_sim2(cache)
    if a.stage in ("sim4", "all"):
        stage_sim4(cache)
    if a.stage in ("inv", "all"):
        stage_invariance(cache)
    if a.stage in ("report", "all"):
        cov = coverage_table(cache)
        sto = storage_estimate(as_of)
        R = load_results(cache)
        stage_report(cache, as_of, cov, sto, build_rules(R), detail_md(R, cov, sto, as_of))
        print("report written", REPORT.with_suffix(".md"), flush=True)
    if a.record:
        stage_record(cache)


if __name__ == "__main__":
    main(sys.argv[1:])
