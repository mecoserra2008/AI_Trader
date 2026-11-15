"""H-MS7 (population extrapolation) and H-MS8 (event-level simulator v1): agent C3 of the microstructure pilot.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.population_simulator_study --stage all [--record]
    stages: symbol | facts | population | loo | replay | report | record

Data (all after the knowledge cutoff, so every result is clean_holdout):
  trades_raw 2026-11-10 (51 symbols, every print), FIT = 00-12 UTC, TEST = 12-24 UTC (TEST holds the 13:30 CPI release and
  about twice the activity); C1 parameter file data/models/microstructure_laws/params_2026-11-10.json (FIT-half laws);
  flow_3s (v0 cache data/cache/sim/20261110T2300): the TEST half of 2026-11-10 gives the real stylised facts (the collector
  did not run 00-12 UTC that day, so there is no FIT-half flow_3s), and 2026-11-07..09 (before TEST) give spread and depth
  for the 39 symbols without an ob200 book; v0 generator models (H-SIM2, fitted 11-08/09) from the same cache.
Choices the registry leaves open are fixed in METHOD before the tests are run.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.sim import flow_sim as fs
from ai_trader.sim import flow_sim_v1 as V
from ai_trader.sim.flow_data import Grid, load_grid

ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-11-10"
DAY0 = 1794268800000
HALF = DAY0 + 43_200_000
DAY1 = DAY0 + 86_400_000
FIT_S = 43_200
CACHE_V0 = ROOT / "data" / "cache" / "sim" / "20261110T2300"
OUT = ROOT / "data" / "cache" / "pop_sim"
PARAMS = ROOT / "data" / "models" / "microstructure_laws" / "params_2026-11-10.json"
BOOK = ROOT / "data" / "raw" / "pilot_2026-11-10" / "book_100ms"
MODEL_OUT = ROOT / "data" / "models" / "population"
REPORT = ROOT / "research" / "reports" / "population_simulator_2026-11-11"
SEED = 20261112
WORKERS = 4
B_ENV = 300
N_PATHS = 10
B_BOOT = 2000
FACT_KEYS = ("kurtosis", "absr_acf", "flow_acf", "impact", "spread_med", "spread_p90", "continuation")
REGIME_SHARE = 0.9   # tick-bound if >= 90% of pre-TEST 3 s candles have a TWA spread below 1.5 ticks

METHOD = {
    "data": "trades_raw 2026-11-10, 51 symbols (config/mm_multi.yaml + BTCUSDT); orders = same-ms same-side prints merged (C1); FIT = 00-12 UTC, "
            "TEST = 12-24 UTC; flow_3s TEST half for the real facts (12-24 UTC; 6.3k-9.9k ok 3 s candles per symbol, gaps never filled); "
            "flow_3s 2026-11-07..09 (before TEST) for spread / depth / tick regime; C1 FIT-half laws for sizes, signs, prints per order",
    "classes": "major = BTC ETH SOL XRP DOGE; tradfi = CL BZ XAU XAG QQQ NVDA TSLA MSTR COIN CRWV GOOGL; alt = the rest (incl. BNB) (MS block)",
    "W": "W = sigma_day x P V_day from FIT trades: sigma_day = std of 1-min log returns of the trade mid proxy ((last buy + last sell) / 2) "
         "x sqrt(1440); P V_day = 2 x FIT notional (USD); illiquidity 1/L = (sigma_day^2 / P V_day)^(1/3) (Kyle-Obizhaeva 2016)",
    "parameters": "per symbol, FIT half: log orders/s (rate); deseasonalised expmix3 duration weights and log component means (mean-1 units); "
                  "order size tail zeta (C1 size.order) and log-normal body mu, sigma; order sign gamma (C1 signs.order.acf_power) and lag-1 "
                  "ACF; log median relative spread (bp, pre-TEST flow_3s TWA) of all symbols; tick regime (logistic on log tick/price and log W); "
                  "log Kyle lambda = OLS slope of 1 s trade-mid-proxy log return (bp) on 1 s signed notional (USD 1k) (trade-price slope reported)",
    "H-MS7": "OLS of each parameter on [1, log W, alt, tradfi] over the 51 symbols; leave-one-symbol-out R^2 = 1 - PRESS / TSS; 95% CI and "
             "p = P*(R2_LOO <= 0.5) by a bootstrap over symbols (B = 2000; LOO on the resample with all copies of the held-out symbol dropped); "
             "universality of zeta (order size tail) and gamma (order sign ACF): D = Var((F+T)/2) / (Var(F-T)/4) across symbols "
             "(F, T = FIT and TEST estimates; D = 1 when the between-symbol spread is all estimation noise; D > 1 iff the halves covary), "
             "p_disp = bootstrap P*(D <= 1); supports = LOO R^2 > 0.5 for rate, spread and impact AND p_disp >= 0.05 for zeta and gamma, with "
             "Holm p < 0.05; summary p = max of the three R^2 bootstrap p values; criterion met with Holm p >= 0.05 -> inconclusive; not met -> contradicts",
    "v1": "sim/flow_sim_v1.py fitted per symbol on FIT: intensity two-component AR of log order counts (bins with mean >= 10 orders), "
          "expmix3 on durations x local rate (centered 5-bin window) rescaled to mean 1, C1 size law (capped at the FIT maximum order), "
          "C1 TranslatedSignModel with p_buy = 0.5 (no drift), prints-per-order pmf rank-coupled to notional (FIT Spearman), "
          "square-root impact per order with permanent share g_inf = 0.6 and tau = 30 s (fixed, not fitted), noise variance prop. to activity, "
          "jumps from FIT 1-min exceedances (> 6 robust sd), spread latent AR(1) in log ticks censored at 1 tick (pre-TEST flow_3s quantiles, "
          "3 s lag-1 ACF), elasticity 0.07 to log activity; k_imp, sigma0 and jump rate by indirect inference (2 pilots x 4 h, 5 iterations) "
          "so that the same trade-based estimators on simulated trades match FIT lambda, 1-min sigma and exceedance rate",
    "H-MS8": "facts as H-SIM2 (excess kurtosis of 3 s mid returns, mean ACF |r| lags 1-60, mean ACF signed flow lags 1-20, corr(r_t, F_t), "
             "spread median AND p90) on the TEST half; envelope = [2.5%, 97.5%] of the class median under a 1 h block bootstrap of each symbol's "
             "TEST facts (B = 300), the v0 method on one period; sim value = class median over symbols of each symbol's median over 10 paths "
             "of the symbol's TEST ok-candle count; v0 = the H-SIM2 FlowSim models (fitted 2026-11-08/09) with the same path count; "
             "supports = v1 count > v0 count in every class and v1 >= 4/5 for majors, with Holm p < 0.05; p = share of 1000 symbol-bootstrap "
             "replicates (resampling symbols within class for envelope and sims together) where the criterion fails; level-vs-shape: v1 refitted "
             "with the TEST levels (order rate, 1-min sigma, lambda, spread location) as an oracle diagnostic",
    "holm": "Holm over H-MS7 and H-MS8 (the registered family is the whole MS block; the block-wide adjustment with the other recorded MS p values is reported)",
    "replay": "descriptive only (H-SIM4 / H-MM3 are the registered tests): sim.replay run_mm with the H-SIM1 touch fill model, decisions every 9 s, "
              "real = flow_3s TEST half of the 50 mm_multi symbols, v1 = 3 paths per symbol (5 h warm-up + TEST length), 3 fill seeds; "
              "sets: live config/mm_multi.yaml and the replay-best a0.2_b0.9_tp0.9_rq0.15 (plus the 36-set grid for a rank comparison)",
}

MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"}
TRADFI = {"CLUSDT", "BZUSDT", "XAUUSDT", "XAGUSDT", "QQQUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT", "CRWVUSDT", "GOOGLUSDT"}


def asset_class(sym: str) -> str:
    return "major" if sym in MAJORS else "tradfi" if sym in TRADFI else "alt"


def _params() -> dict:
    return json.loads(PARAMS.read_text(encoding="utf-8"))


def symbols() -> list[str]:
    return sorted(_params()["symbols"])


def _meta() -> dict:
    return json.loads((CACHE_V0 / "instruments.json").read_text(encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------------- data

def load_trades(sym: str) -> dict[str, np.ndarray]:
    import duckdb

    glob = (ROOT / "data" / "lake" / "trades_raw" / "venue=bybit" / f"symbol={sym}" / f"date={DATE}" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    df = con.sql(f"SELECT epoch_ms(event_ts) AS ms, side, price, size, filename, file_row_number FROM read_parquet('{glob}', "
                 f"file_row_number=true, filename=true) ORDER BY ms, filename, file_row_number").pl()
    ms = df["ms"].to_numpy().astype(np.int64)
    keep = (ms >= DAY0) & (ms < DAY1)
    side = np.where(df["side"].to_numpy() == "Buy", 1, -1).astype(np.int64)
    return {"ms": ms[keep], "side": side[keep], "price": df["price"].to_numpy()[keep].astype(float),
            "notional": (df["price"].to_numpy() * df["size"].to_numpy())[keep].astype(float)}


def orders_of(tr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ms, side = tr["ms"], tr["side"]
    new = np.ones(len(ms), dtype=bool)
    new[1:] = (ms[1:] != ms[:-1]) | (side[1:] != side[:-1])
    gid = np.cumsum(new) - 1
    st = np.flatnonzero(new)
    return {"ms": ms[st], "side": side[st], "notional": np.bincount(gid, tr["notional"]), "n_prints": np.bincount(gid)}


def _sel(d: dict[str, np.ndarray], a: int, b: int) -> dict[str, np.ndarray]:
    m = (d["ms"] >= a) & (d["ms"] < b)
    return {k: v[m] for k, v in d.items()}


def spread_info(sym: str, tick_size: float) -> dict[str, Any]:
    """Pre-TEST flow_3s (2026-11-07..09): tick in bp, TWA spread quantiles, censored latent fit, 3 s ACF of log spread, depth."""
    g = load_grid(CACHE_V0 / f"{sym}.parquet")
    pre = (g.t_ms < DAY0) & g.ok
    mid = float(np.nanmedian(g.mid[pre]))
    tick_bps = tick_size / mid * 1e4
    sp = g.spread_bps[pre]
    ticks = sp / tick_bps
    cf = V.censored_spread_fit(ticks)
    ls = np.log(np.maximum(g.spread_bps, 1e-9))
    okp = pre[1:] & pre[:-1]
    a, b = ls[1:][okp], ls[:-1][okp]
    acf1 = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
    out = {"tick_bps": tick_bps, "mid_pre": mid, "spread_med_bps": float(np.median(sp)), "spread_p90_bps": float(np.percentile(sp, 90)),
           "acf1_log_spread_3s": acf1, "depth25_med": float(np.nanmedian(g.depth[pre])), "n_pre": int(pre.sum()), **cf}
    out["tick_bound"] = bool(cf["share_lt_1p5_tick"] >= REGIME_SHARE)
    bp = BOOK / f"{sym}.parquet"
    if bp.exists():
        bk = pl.read_parquet(bp, columns=["ts", "rel_spread_bp", "bid1", "ask1"])
        t = bk["ts"].dt.epoch("ms").to_numpy()
        f = (t < HALF) & np.isfinite(bk["rel_spread_bp"].to_numpy())
        rs = bk["rel_spread_bp"].to_numpy()[f]
        tk = (bk["ask1"].to_numpy() - bk["bid1"].to_numpy())[f] / tick_size
        out["book_fit"] = {"rel_spread_med_bp": float(np.median(rs)), "one_tick_share": float(np.mean(tk < 1.5))}
    return out


# ------------------------------------------------------------------------------------------------------- stage symbol

def _half_measures(tr: dict[str, np.ndarray], od: dict[str, np.ndarray], a: int) -> dict[str, Any]:
    st = V.trade_stats(tr["ms"], tr["side"], tr["price"], tr["notional"], a, FIT_S)
    pv_day = 2.0 * float(tr["notional"].sum())
    st["pv_day"] = pv_day
    st["W"] = st["sigma_day"] * 1e-4 * pv_day if np.isfinite(st["sigma_day"]) else float("nan")   # sigma as a fraction
    st["illiq"] = ((st["sigma_day"] * 1e-4) ** 2 / pv_day) ** (1 / 3) if pv_day > 0 else float("nan")
    st["orders_per_s"] = len(od["ms"]) / FIT_S
    st["prints_per_s"] = len(tr["ms"]) / FIT_S
    st["price"] = float(np.median(tr["price"]))
    st["order_notional_med"] = float(np.median(od["notional"]))
    st["print_notional_med"] = float(np.median(tr["notional"]))
    st["rho_kv"] = float(stats.spearmanr(od["notional"], od["n_prints"]).statistic) if len(od["ms"]) > 50 else 0.5
    return st


def build_params(sym: str, P: dict, fit_tr: dict, fit_od: dict, meas: dict, spr: dict, *, level: dict | None = None) -> V.V1Params:
    """V1Params from the FIT half (level: optional overrides of the level quantities for the oracle diagnostic)."""
    h = P["symbols"][sym]["halves"]["fit"]
    sz = h["size"]["order"]["size_law"]
    sm = h["signs"]["order"]["sign_model"]
    rates = h["rates"]
    inten = V.fit_intensity(fit_od["ms"], DAY0, FIT_S)
    mix = V.fit_mixture_normalized(fit_od["ms"], DAY0, FIT_S, inten["bin_min"])
    depth = spr["depth25_med"] if np.isfinite(spr["depth25_med"]) else 1e5
    tick_bps = spr["tick_bps"]
    level_usd = max(depth / 2.0 * min(1.0, tick_bps / 25.0), 10.0)
    p = V.V1Params(symbol=sym, asset_class=asset_class(sym), price0=meas["price"], tick_bps=tick_bps * spr["mid_pre"] / meas["price"],
                   rate=meas["orders_per_s"], mix_w=mix["mix_w"], mix_mean=mix["mix_mean"], int_v=inten["int_v"], int_phi=inten["int_phi"],
                   size_mu=sz["mu"], size_sigma=sz["sigma"], size_xmin=sz["xmin"], size_alpha=sz["alpha"], size_p_tail=sz["p_tail"],
                   size_cap=float(fit_od["notional"].max()), sign_c=sm["c"], sign_gamma=sm["gamma"], sign_short=sm["short"], p_buy=0.5,
                   ppo_pmf=rates["prints_per_order_pmf"], ppo_tail_zeta=float(rates.get("prints_per_order_tail_zeta") or 1.0),
                   rho_kv=meas["rho_kv"], level_usd=level_usd, v_ref=meas["order_notional_med"],
                   g_inf=0.6, tau_s=30.0, k_imp=0.1, sigma0=max(meas["sigma_1m"], 0.5) / math.sqrt(60) * 0.8,
                   spr_mu=spr["spr_mu"], spr_sd=spr["spr_sd"], spr_phi=float(np.clip(spr["acf1_log_spread_3s"], 0.05, 0.995)) ** (1 / 3)
                   if np.isfinite(spr["acf1_log_spread_3s"]) else 0.9, spr_elast=0.07, depth25=depth,
                   info={"intensity": inten, "mixture": mix})
    if level:
        for k, v in level.items():
            setattr(p, k, v)
    return p


def stage_symbol_one(sym: str) -> dict[str, Any]:
    t0 = time.time()
    P = _params()
    meta = _meta()[sym]
    tr = load_trades(sym)
    od = orders_of(tr)
    fit_tr, test_tr = _sel(tr, DAY0, HALF), _sel(tr, HALF, DAY1)
    fit_od, test_od = _sel(od, DAY0, HALF), _sel(od, HALF, DAY1)
    meas = {"fit": _half_measures(fit_tr, fit_od, DAY0), "test": _half_measures(test_tr, test_od, HALF)}
    spr = spread_info(sym, float(meta["tick_size"]))
    p = build_params(sym, P, fit_tr, fit_od, meas["fit"], spr)
    pre = pre_test_grid_stats(sym)
    tgt = calib_targets(meas["fit"], pre)
    p, fin = V.calibrate(p, tgt, jump_z=jump_z(meas["fit"]), seed=SEED)
    # oracle level variant: TEST order rate, 1-min sigma, price level (tick in bp) and spread location; shape targets unchanged
    tt = meas["test"]
    # (the tick in bp at the TEST price level: a tick-bound spread in bp is tick / price)
    q = build_params(sym, P, fit_tr, fit_od, meas["fit"], spr, level={"rate": tt["orders_per_s"], "price0": tt["price"],
                                                                       "tick_bps": spr["tick_bps"] * spr["mid_pre"] / tt["price"]})
    test_spr = _test_spread(sym, spr["tick_bps"])
    if test_spr is not None:
        q.spr_mu, q.spr_sd = test_spr["spr_mu"], test_spr["spr_sd"]
    q, fin_q = V.calibrate(q, calib_targets(tt, pre), jump_z=jump_z(tt), seed=SEED)
    out = {"symbol": sym, "class": asset_class(sym), "meas": meas, "spread": spr, "params": p.to_dict(), "params_level": q.to_dict(),
           "calib": fin, "calib_level": fin_q, "test_spread": test_spr, "pre_test": {k: v for k, v in pre.items() if k != "exc_z"},
           "targets": tgt, "secs": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    for k in ("exc_values",):
        for h in ("fit", "test"):
            out["meas"][h][k + "_n"] = len(out["meas"][h][k])
    (OUT / "symbols").mkdir(exist_ok=True)
    (OUT / "symbols" / f"{sym}.json").write_text(json.dumps(_jsonable(out)), encoding="utf-8")
    return {"symbol": sym, "secs": out["secs"], "calib": fin, "target": tgt}


def pre_test_grid_stats(sym: str) -> dict[str, Any]:
    """Book-mid statistics of the pre-TEST flow_3s days (2026-11-07..09): shape targets of the price process."""
    g = load_grid(CACHE_V0 / f"{sym}.parquet")
    return V.grid_stats(g, g.t_ms < DAY0)


def calib_targets(meas: dict[str, Any], pre: dict[str, Any]) -> dict[str, Any]:
    """Level from the half's trades (1-min sigma of the trade mid proxy, its robust sd and 1-min exceedance rate), shape
    from the pre-TEST book mids (3 s flow-return correlation and the two variance ratios, which are stable across days),
    flow and volatility clustering from the half's trades (3 s signed-flow ACF, size ACF, |r| ACF of the trade mid proxy,
    matched in the same trade space). Kurtosis is NOT a target: it is left to the structure (impact exponent 0.5,
    spread-out jumps) and tested out of sample."""
    return {"sigma_1m": meas["sigma_1m"], "robust_sd_1m": meas["robust_sd_1m"], "impact": pre["impact"], "kurtosis": None,
            "absr_acf": None, "absr_acf_3s": meas["absr_acf_3s"], "vr_3s_1m": pre["vr_3s_1m"], "vr5": pre["vr5"], "exc_rate": meas["exc_rate"],
            "flow_acf_3s": meas["flow_acf_3s"], "size_ns_acf10": meas["size_ns_acf10"]}


def jump_z(meas: dict[str, Any]) -> list[float]:
    rob = meas.get("robust_sd_1m")
    return [float(x) / rob for x in meas.get("exc_values", [])] if rob and np.isfinite(rob) and rob > 0 else []


def _test_spread(sym: str, tick_bps: float) -> dict | None:
    g = load_grid(CACHE_V0 / f"{sym}.parquet")
    m = (g.t_ms >= HALF) & (g.t_ms < DAY1) & g.ok
    if m.sum() < 1000:
        return None
    return V.censored_spread_fit(g.spread_bps[m] / tick_bps)


def load_symbol(sym: str) -> dict[str, Any]:
    return json.loads((OUT / "symbols" / f"{sym}.json").read_text(encoding="utf-8"))


def _run(fn, items, workers=WORKERS):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _safe_symbol_one(sym: str) -> dict[str, Any]:
    import traceback
    try:
        return stage_symbol_one(sym)
    except Exception:  # noqa: BLE001 - reported per symbol, the stage continues
        return {"symbol": sym, "error": traceback.format_exc()}


def stage_symbol(syms: list[str], force: bool = False) -> None:
    todo = [s for s in syms if force or not (OUT / "symbols" / f"{s}.json").exists()]
    t0 = time.time()
    for r in _run(_safe_symbol_one, todo):
        if "error" in r:
            print(r["symbol"], "ERROR", r["error"], flush=True)
            continue
        print(r["symbol"], f"{r['secs']:.0f}s", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["calib"].items()},
              {k: round(v, 4) if isinstance(v, float) else v for k, v in r["target"].items()}, flush=True)
    print(f"stage symbol {len(todo)} in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------------- stage facts

def _seed(sym: str, k: int, base: int = SEED) -> int:
    return base + 1000 * k + sum(map(ord, sym))


def _facts_vec(g: Grid) -> list[float]:
    f, _ = fs.facts_of_grid(g)
    return [f[x] for x in FACT_KEYS]


def _paths_facts(p: V.V1Params, n_val: int, sym: str, n_paths: int = N_PATHS, base: int = SEED) -> tuple[np.ndarray, list[dict]]:
    out, extra = [], []
    for k in range(n_paths):
        o = V.generate(p, 3 * n_val, _seed(sym, k, base), t0_ms=HALF)
        out.append(_facts_vec(o.grid))
        if k < 2:
            gs = V.grid_stats(o.grid)
            extra.append({x: gs[x] for x in ("sigma_3s", "sigma_1m", "vr_3s_1m", "vr5", "exc_rate")} | {"orders_per_s": len(o.orders["ms"]) / (3 * n_val)})
    return np.array(out), extra


def stage_facts_one(sym: str) -> dict[str, Any]:
    from ai_trader.research.simulator_study import _boot_facts   # the H-SIM2 envelope bootstrap (1 h blocks)

    t0 = time.time()
    d = load_symbol(sym)
    g = load_grid(CACHE_V0 / f"{sym}.parquet")
    m = (g.t_ms >= HALF) & (g.t_ms < DAY1)
    n_val = int(g.ok[m].sum())
    f, bs = fs.facts_of_grid(g, m)
    rng = np.random.default_rng(SEED + sum(map(ord, sym)))
    boot = _boot_facts(bs, rng, B_ENV)
    real_gs = V.grid_stats(g, m)
    p = V.V1Params.from_dict(d["params"])
    q = V.V1Params.from_dict(d["params_level"])
    v1, v1x = _paths_facts(p, n_val, sym)
    v1l, v1lx = _paths_facts(q, n_val, sym)
    m0 = pickle.loads((CACHE_V0 / "models" / f"{sym}.pkl").read_bytes())
    mid0 = float(g.mid[m & g.ok][0])
    v0 = np.array([_facts_vec(m0.generate(n_val, np.random.default_rng(_seed(sym, k, SEED + 7)), mid0=mid0, t0_ms=HALF)) for k in range(N_PATHS)])
    out = {"symbol": sym, "class": asset_class(sym), "n_val": n_val, "real": [f[x] for x in FACT_KEYS], "boot": boot,
           "v1": v1, "v1_level": v1l, "v0": v0, "v1_extra": v1x, "v1_level_extra": v1lx,
           "real_gs": {x: real_gs[x] for x in ("sigma_3s", "sigma_1m", "vr_3s_1m", "vr5", "exc_rate", "n_3s")}, "secs": time.time() - t0}
    (OUT / "facts").mkdir(parents=True, exist_ok=True)
    (OUT / "facts" / f"{sym}.pkl").write_bytes(pickle.dumps(out))
    return {"symbol": sym, "secs": out["secs"], "real": np.round(out["real"], 4).tolist(), "v1": np.round(np.nanmedian(v1, axis=0), 4).tolist()}


def _safe(fn, sym):
    import traceback
    try:
        return fn(sym)
    except Exception:  # noqa: BLE001
        return {"symbol": sym, "error": traceback.format_exc()}


def _safe_facts(sym):
    return _safe(stage_facts_one, sym)


def stage_generic(fn, folder: str, syms: list[str], force: bool, suffix: str = ".pkl") -> None:
    todo = [s for s in syms if force or not (OUT / folder / f"{s}{suffix}").exists()]
    t0 = time.time()
    for r in _run(fn, todo):
        if "error" in r:
            print(r["symbol"], "ERROR", r["error"], flush=True)
        else:
            print({k: v for k, v in r.items()}, flush=True)
    print(f"stage {folder}: {len(todo)} symbols in {time.time() - t0:.0f}s", flush=True)


def load_facts(sym: str) -> dict[str, Any]:
    return pickle.loads((OUT / "facts" / f"{sym}.pkl").read_bytes())


# ------------------------------------------------------------------------------------------------------- stage population

PRIMARY7 = {"rate": "log_rate", "spread": "log_spread_bps", "impact": "log_lambda"}
SECONDARY7 = ("log_v_ref", "size_mu", "log_sigma_day", "log_depth25", "sign_acf1", "sign_gamma", "zeta_order", "ppo_mean", "mix_w1", "mix_w2",
              "log_mix_m1", "log_mix_m2", "log_mix_m3", "log_lambda_trade", "impact_corr", "flow_acf_3s", "absr_acf_3s", "log_k_imp")


def population_rows() -> list[dict]:
    from ai_trader.sim.population import rows_from_symbols

    P = _params()
    rows = rows_from_symbols([load_symbol(s) for s in symbols()], P)
    for r in rows:
        lt = r.get("lambda_trade")
        r["log_lambda_trade"] = math.log(lt) if lt and lt > 0 else None
    return rows


def _reg_block(rows: list[dict], name: str, rng: np.random.Generator, B: int = B_BOOT) -> dict[str, Any]:
    from ai_trader.sim import population as POP

    y = np.array([r.get(name) if r.get(name) is not None else np.nan for r in rows], dtype=float)
    ok = np.isfinite(y)
    lw = np.log(np.array([r["W"] for r in rows]))
    cl = [r["class"] for r in rows]
    X = POP.design(lw, cl)[ok]
    yy = y[ok]
    b, e = POP.ols(yy, X)
    se = POP.hc1_se(X, e)
    r2 = float(1 - (e @ e) / (((yy - yy.mean()) ** 2).sum()))
    loo = POP.loo_r2(yy, X)
    bs = POP.loo_r2_boot(yy, X, B, rng)
    Xw = X[:, :2]
    bw, ew = POP.ols(yy, Xw)
    return {"param": name, "n": int(ok.sum()), "coef": b.tolist(), "se": se.tolist(), "r2": r2, "loo_r2": loo,
            "loo_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "p_le_05": float(np.nanmean(bs <= 0.5)),
            "loo_r2_W_only": POP.loo_r2(yy, Xw), "slope_W_only": float(bw[1]), "slope_W_only_se": float(POP.hc1_se(Xw, ew)[1]),
            "class_means_resid": {c: float(np.mean([e_ for e_, cc in zip(e, np.array(cl)[ok], strict=True) if cc == c])) for c in ("major", "alt", "tradfi")}}


def _ko_block(rows: list[dict]) -> dict[str, Any]:
    """Kyle-Obizhaeva comparisons: log-log slopes on log W (HC1 SE) of the invariance-normalised quantities."""
    from ai_trader.sim import population as POP

    lw = np.log(np.array([r["W"] for r in rows]))
    sig = np.array([r["sigma_day"] for r in rows])
    out = {}

    def fit(y, x, pred, label):
        ok = np.isfinite(y) & np.isfinite(x)
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        b, e = POP.ols(y[ok], X)
        se = POP.hc1_se(X, e)
        out[label] = {"slope": float(b[1]), "se": float(se[1]), "prediction": pred, "t_vs_prediction": float((b[1] - pred) / se[1]) if se[1] > 0 else None,
                      "r2": float(1 - (e @ e) / (((y[ok] - y[ok].mean()) ** 2).sum())), "n": int(ok.sum())}

    g = lambda k: np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows], dtype=float)  # noqa: E731
    fit(g("log_rate"), lw, 2 / 3, "orders per day vs W (bets ~ W^2/3)")
    fit(np.log(np.exp(g("log_rate")) * g("ppo_mean")), lw, 2 / 3, "prints per day vs W")
    fit(g("log_v_ref"), lw, 1 / 3, "median order notional vs W (bet size ~ W^1/3)")
    fit(g("log_spread_bps") - np.log(sig * 1e4), lw, -1 / 3, "spread / sigma vs W (~ W^-1/3)")
    fit(g("log_spread_bps"), np.log(g("illiq")), 1.0, "spread vs illiquidity (sigma^2 / PV)^1/3")
    fit(g("log_lambda") - 2 * np.log(sig * 1e4), lw, -2 / 3, "lambda / sigma^2 vs W (~ W^-2/3)")
    fit(g("log_lambda_trade") - 2 * np.log(sig * 1e4), lw, -2 / 3, "lambda (trade price) / sigma^2 vs W")
    tb = np.array([not r["tick_bound"] for r in rows])
    yy, xx = g("log_spread_bps") - np.log(sig * 1e4), lw.copy()
    yy[~tb] = np.nan
    fit(yy, xx, -1 / 3, "spread / sigma vs W, free-spread symbols only")
    return out


def stage_population() -> dict[str, Any]:
    from ai_trader.sim import population as POP

    rows = population_rows()
    rng = np.random.default_rng(SEED)
    prim = {k: _reg_block(rows, v, rng) for k, v in PRIMARY7.items()}
    sec = {v: _reg_block(rows, v, rng, B=500) for v in SECONDARY7}
    uni = {}
    for name, fk, tk in (("zeta_order", "zeta_order", "zeta_order_test"), ("sign_gamma", "sign_gamma", "sign_gamma_test"),
                         ("zeta_print", "zeta_print", "zeta_print_test"), ("sign_acf1", "sign_acf1", "sign_acf1_test")):
        F = np.array([r.get(fk) if r.get(fk) is not None else np.nan for r in rows], float)
        T = np.array([r.get(tk) if r.get(tk) is not None else np.nan for r in rows], float)
        uni[name] = POP.universality(F, T, B_BOOT, rng)
        cl = np.array([r["class"] for r in rows])
        Fd, Td = F.copy(), T.copy()
        for c in ("major", "alt", "tradfi"):
            mk = (cl == c) & np.isfinite(F) & np.isfinite(T)
            if mk.any():
                mu = np.mean((F[mk] + T[mk]) / 2)
                Fd[mk] -= mu
                Td[mk] -= mu
        uni[name]["within_class"] = POP.universality(Fd, Td, B_BOOT, rng)
    # gamma universality restricted to the C1 eligibility (>= 20k prints in the day)
    P = _params()
    big = [r for r in rows if P["symbols"][r["symbol"]]["n_prints"] >= 20_000]
    Fg = np.array([r["sign_gamma"] for r in big], float)
    Tg = np.array([r["sign_gamma_test"] for r in big], float)
    uni["sign_gamma"]["ge20k"] = POP.universality(Fg, Tg, B_BOOT, rng)
    # regime model (tick-bound) and free-spread level
    lw = np.log(np.array([r["W"] for r in rows]))
    lt = np.log(np.array([r["tick_bps"] for r in rows]))
    tb = np.array([float(r["tick_bound"]) for r in rows])
    Xr = np.column_stack([np.ones(len(lw)), lt, lw])
    br = POP.logistic(tb, Xr)
    pr = 1 / (1 + np.exp(-(Xr @ br)))
    loo_pred = np.empty(len(tb))
    for i in range(len(tb)):
        mk = np.arange(len(tb)) != i
        bi = POP.logistic(tb[mk], Xr[mk])
        loo_pred[i] = 1 / (1 + np.exp(-(Xr[i] @ bi)))
    fr = tb == 0
    ys = np.array([r["log_spread_bps"] for r in rows])
    Xf = np.column_stack([np.ones(int(fr.sum())), lw[fr], lt[fr]])
    bf, ef = POP.ols(ys[fr], Xf)
    regime = {"coef": br.tolist(), "names": ["const", "log_tick_bps", "log_W"], "acc_in": float(np.mean((pr >= 0.5) == (tb == 1))),
              "acc_loo": float(np.mean((loo_pred >= 0.5) == (tb == 1))), "brier_loo": float(np.mean((loo_pred - tb) ** 2)),
              "base_rate": float(tb.mean()), "n_free": int(fr.sum()), "free_symbols": [r["symbol"] for r, f in zip(rows, fr, strict=True) if f],
              "free_spread_coef": bf.tolist(), "free_spread_se": POP.hc1_se(Xf, ef).tolist(), "free_spread_loo_r2": POP.loo_r2(ys[fr], Xf),
              "free_spread_class_loo_r2": POP.loo_r2(ys[fr], POP.design(lw[fr], [r["class"] for r, f in zip(rows, fr, strict=True) if f]))
              if fr.sum() > 6 else None}
    ko = _ko_block(rows)
    # verdict
    r2_ok = all(prim[k]["loo_r2"] > 0.5 for k in prim)
    uni_ok = all(uni[k]["p_disp"] >= 0.05 for k in ("zeta_order", "sign_gamma"))
    p_sum = max(prim[k]["p_le_05"] for k in prim)
    pop = POP.Population.fit(rows)
    pop.save(MODEL_OUT / "population_2026-11-10.json")
    table = [{k: r.get(k) for k in ("symbol", "class", "W", "sigma_day", "pv_day", "tick_bps", "tick_bound", "log_rate", "log_spread_bps", "log_lambda",
                                     "log_lambda_trade", "zeta_order", "zeta_order_test", "sign_gamma", "sign_gamma_test", "sign_acf1", "log_v_ref",
                                     "impact_corr", "flow_acf_3s")} for r in rows]
    out = {"primary": prim, "secondary": sec, "universality": uni, "regime": regime, "ko": ko, "criterion_r2": r2_ok, "criterion_universal": uni_ok,
           "criterion_met": bool(r2_ok and uni_ok), "p_summary": p_sum, "rows": table, "n": len(rows)}
    (OUT / "population.json").write_text(json.dumps(_jsonable(out), indent=1), encoding="utf-8")
    print(json.dumps({k: (round(v["loo_r2"], 3), [round(x, 3) for x in v["loo_ci"]], round(v["p_le_05"], 3)) for k, v in prim.items()}), flush=True)
    print(json.dumps({k: (round(v["D"], 2), [round(x, 2) for x in v["ci"]], round(v["p_disp"], 3)) for k, v in uni.items()}), flush=True)
    return out


# ------------------------------------------------------------------------------------------------------- stage loo (extrapolation)

def stage_loo_one(sym: str) -> dict[str, Any]:
    from ai_trader.sim.population import Population

    t0 = time.time()
    rows = population_rows()
    me = next(r for r in rows if r["symbol"] == sym)
    pop = Population.fit(rows, exclude=sym)
    p = pop.params_for(me["W"], me["class"], me["tick_bps"] / 1e4, price=me["price"], symbol=sym, calibrate=True, seed=SEED)
    fct = load_facts(sym)
    n_val = fct["n_val"]
    pf, px = _paths_facts(p, n_val, sym, n_paths=N_PATHS, base=SEED + 3)
    pred = {"log_rate": math.log(p.rate), "log_spread_bps": math.log(pop.spread_bps(me["W"], me["class"], me["tick_bps"])["spread_bps"]),
            "regime": pop.spread_bps(me["W"], me["class"], me["tick_bps"])["regime"], "log_sigma_day": math.log(p.info["population"]["sigma_day"]),
            "log_v_ref": math.log(p.v_ref), "sign_gamma": p.sign_gamma, "zeta_order": p.size_alpha - 1}
    actual = {k: me.get(k) for k in ("log_rate", "log_spread_bps", "log_sigma_day", "log_v_ref", "sign_gamma", "zeta_order")}
    actual["regime"] = "tick_bound" if me["tick_bound"] else "free"
    out = {"symbol": sym, "class": me["class"], "params": p.to_dict(), "pop": pf, "pop_extra": px, "pred": pred, "actual": actual,
           "secs": time.time() - t0}
    (OUT / "loo").mkdir(parents=True, exist_ok=True)
    (OUT / "loo" / f"{sym}.pkl").write_bytes(pickle.dumps(out))
    return {"symbol": sym, "secs": out["secs"], "pop": np.round(np.nanmedian(pf, axis=0), 4).tolist(), "real": np.round(fct["real"], 4).tolist()}


def _safe_loo(sym):
    return _safe(stage_loo_one, sym)


# ------------------------------------------------------------------------------------------------------- stage replay (descriptive)

REPLAY_SEEDS = 3
WARM = 6000


def _replay_cfgs():
    from dataclasses import replace

    from ai_trader.sim import replay as rp

    base = rp.MmCfg.from_yaml(ROOT / "config" / "mm_multi.yaml")
    best = replace(base, ask_offset_sigma=0.2, bid_offset_sigma=0.9, take_profit_sigma=0.9, requote_sigma=0.15)
    return [base, best] + rp.grid_params(base)


def stage_replay_one(sym: str) -> dict[str, Any]:
    from ai_trader.sim import fill_model as fmod
    from ai_trader.sim import replay as rp

    t0 = time.time()
    meta = _meta()[sym]
    s1 = json.loads((CACHE_V0 / "sim1.json").read_text(encoding="utf-8"))
    F = fmod.FillModel(**s1["variants"][s1["primary_variant"]]["params"])
    R = rp.Rule(float(meta["tick_size"]), float(meta["qty_step"]), float(meta["min_qty"]), float(meta.get("min_notional") or 5.0))
    cfgs = _replay_cfgs()
    g = load_grid(CACHE_V0 / f"{sym}.parquet")
    m = (g.t_ms >= HALF) & (g.t_ms < DAY1)
    start = int(np.flatnonzero(m)[0])
    n_val = int(g.ok[m].sum())
    mid0 = float(g.mid[m & g.ok][0])
    p = V.V1Params.from_dict(load_symbol(sym)["params"])
    p.tick_bps = float(meta["tick_size"]) / mid0 * 1e4     # the instrument's tick at the replay price
    paths = [V.generate(p, 3 * (WARM + n_val), _seed(sym, k, SEED + 11), t0_ms=HALF - WARM * STEP_MS_, mid0=mid0).grid for k in range(REPLAY_SEEDS)]
    real = np.zeros((len(cfgs), 4))
    sim = np.zeros((len(cfgs), 4))
    for j, c in enumerate(cfgs):
        for k in range(REPLAY_SEEDS):
            o = rp.run_mm(g, R, c, F, np.random.default_rng(SEED + k), start=start, warmup=WARM)
            real[j] += [o["pnl"], o["turnover"], o["maker_fills"], o["fees"]]
            o = rp.run_mm(paths[k], R, c, F, np.random.default_rng(SEED + k), start=WARM, warmup=WARM)
            sim[j] += [o["pnl"], o["turnover"], o["maker_fills"], o["fees"]]
    out = {"symbol": sym, "real": real / REPLAY_SEEDS, "sim": sim / REPLAY_SEEDS, "keys": [rp.param_key(c) for c in cfgs], "secs": time.time() - t0}
    (OUT / "replay").mkdir(parents=True, exist_ok=True)
    (OUT / "replay" / f"{sym}.pkl").write_bytes(pickle.dumps(out))
    return {"symbol": sym, "secs": round(out["secs"], 1)}


STEP_MS_ = 3000


def _safe_replay(sym):
    return _safe(stage_replay_one, sym)


def mm_symbols() -> list[str]:
    import yaml

    return list(yaml.safe_load((ROOT / "config" / "mm_multi.yaml").read_text(encoding="utf-8"))["symbols"])


# ------------------------------------------------------------------------------------------------------- stage report

FACTS5 = ("kurtosis", "absr_acf", "flow_acf", "impact", "spread")
FACT_LABEL = {"kurtosis": "excess kurtosis of 3 s returns", "absr_acf": "mean ACF of |r|, lags 1-60", "flow_acf": "mean ACF of signed flow, lags 1-20",
              "impact": "impact corr(r_t, F_t)", "spread_med": "spread median (bp)", "spread_p90": "spread p90 (bp)",
              "continuation": "continuation ratio (extra)"}
VARIANTS = ("v1", "v0", "v1_level", "pop")


def _inside(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, bool]:
    ins = (v >= lo) & (v <= hi)
    return {"kurtosis": bool(ins[0]), "absr_acf": bool(ins[1]), "flow_acf": bool(ins[2]), "impact": bool(ins[3]), "spread": bool(ins[4] and ins[5])}


def _sym_val(f: dict, variant: str) -> np.ndarray | None:
    a = f.get(variant)
    if a is None:
        return None
    return np.nanmedian(np.asarray(a, dtype=float), axis=0)


def _class_eval(F: list[dict], idx: np.ndarray | None = None) -> dict[str, Any]:
    rs = F if idx is None else [F[i] for i in idx]
    med_boot = np.nanmedian(np.stack([f["boot"] for f in rs]), axis=0)
    lo, hi = np.nanpercentile(med_boot, 2.5, axis=0), np.nanpercentile(med_boot, 97.5, axis=0)
    out = {"lo": lo, "hi": hi, "real": np.nanmedian(np.array([f["real"] for f in rs]), axis=0)}
    for v in VARIANTS:
        vals = [x for x in (_sym_val(f, v) for f in rs) if x is not None]
        if len(vals) == len(rs):
            sv = np.nanmedian(np.array(vals), axis=0)
            ins = _inside(sv, lo, hi)
            out[v] = {"value": sv, "inside": ins, "n_inside": int(sum(ins.values()))}
    return out


def ms8_eval(F: list[dict], B: int = 1000) -> dict[str, Any]:
    by = {c: [f for f in F if f["class"] == c] for c in ("major", "alt", "tradfi")}
    res = {c: _class_eval(v) for c, v in by.items()}
    counts = {c: {v: res[c][v]["n_inside"] for v in VARIANTS if v in res[c]} for c in res}
    crit = all(counts[c]["v1"] > counts[c]["v0"] for c in counts) and counts["major"]["v1"] >= 4
    rng = np.random.default_rng(SEED + 8)
    fails = 0
    boot_counts = {c: {"v1": [], "v0": []} for c in by}
    for _ in range(B):
        ok = True
        cc = {}
        for c, rs in by.items():
            ix = rng.integers(0, len(rs), len(rs))
            e = _class_eval(rs, ix)
            cc[c] = (e["v1"]["n_inside"], e["v0"]["n_inside"])
            boot_counts[c]["v1"].append(cc[c][0])
            boot_counts[c]["v0"].append(cc[c][1])
        ok = all(cc[c][0] > cc[c][1] for c in cc) and cc["major"][0] >= 4
        fails += not ok
    p = fails / B
    # level vs shape per class and fact
    lvl = {}
    for c, e in res.items():
        lvl[c] = {}
        for j, k in enumerate(FACTS5):
            if e["v1"]["inside"][k]:
                lvl[c][k] = "pass"
                continue
            lvl[c][k] = "level" if e["v1_level"]["inside"][k] else "shape"
    # per-symbol own envelopes (descriptive)
    own = {}
    for c, rs in by.items():
        own[c] = {}
        for v in VARIANTS:
            flags = []
            for f in rs:
                sv = _sym_val(f, v)
                if sv is None:
                    continue
                lo, hi = np.nanpercentile(f["boot"], 2.5, axis=0), np.nanpercentile(f["boot"], 97.5, axis=0)
                flags.append(_inside(sv, lo, hi))
            own[c][v] = {k: float(np.mean([fl[k] for fl in flags])) if flags else None for k in FACTS5}
    return {"classes": res, "counts": counts, "criterion_met": bool(crit), "p": p, "level_vs_shape": lvl, "own_envelope_share": own,
            "boot_counts": {c: {v: np.bincount(np.array(x), minlength=6).tolist() for v, x in d.items()} for c, d in boot_counts.items()}}


def _holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _block_ps() -> dict[str, float]:
    from ai_trader.research import hypotheses as H

    out = {}
    for hid in ("H-MS1", "H-MS2", "H-MS3", "H-MS4", "H-MS5", "H-MS6", "H-MS9"):
        rs = H.results(hid)
        if rs:
            m = rs[-1]["metrics"]
            v = m.get("p_summary", m.get("p"))
            if v is not None:
                out[hid] = float(v)
    return out


def _replay_summary() -> dict[str, Any] | None:
    files = sorted((OUT / "replay").glob("*.pkl")) if (OUT / "replay").exists() else []
    if not files:
        return None
    R = [pickle.loads(f.read_bytes()) for f in files]
    keys = R[0]["keys"]
    RP = np.stack([r["real"] for r in R])
    SP = np.stack([r["sim"] for r in R])

    def bps(A, j):
        t = A[:, j, 1].sum()
        return float(A[:, j, 0].sum() / t * 1e4) if t > 0 else float("nan")

    rows = []
    for j, k in enumerate(keys):
        rows.append({"params": k, "real_bps": bps(RP, j), "v1_bps": bps(SP, j), "real_pnl": float(RP[:, j, 0].sum()), "v1_pnl": float(SP[:, j, 0].sum()),
                     "real_turnover": float(RP[:, j, 1].sum()), "v1_turnover": float(SP[:, j, 1].sum()), "real_fills": float(RP[:, j, 2].sum()),
                     "v1_fills": float(SP[:, j, 2].sum()), "real_pos": int((RP[:, j, 0] > 0).sum()), "v1_pos": int((SP[:, j, 0] > 0).sum())})
    grid = rows[2:]
    from scipy import stats as st

    rb, sb = np.array([r["real_bps"] for r in grid]), np.array([r["v1_bps"] for r in grid])
    rt, stt = np.array([r["real_turnover"] for r in grid]), np.array([r["v1_turnover"] for r in grid])
    rf, sf = np.array([r["real_fills"] for r in grid]), np.array([r["v1_fills"] for r in grid])
    s4 = json.loads((CACHE_V0 / "sim4.json").read_text(encoding="utf-8"))
    v0 = {}
    for t in s4["table"]:
        k = t["params"]
        if k.startswith("live mm_multi"):
            v0["live"] = t
        if k == "a0.2_b0.9_tp0.9_rq0.15":
            v0["best"] = t
    return {"symbols": len(R), "rows": rows, "live": rows[0], "best": rows[1],
            "spearman_bps": float(st.spearmanr(rb, sb).statistic), "spearman_turnover": float(st.spearmanr(rt, stt).statistic),
            "spearman_fills": float(st.spearmanr(rf, sf).statistic), "v0": v0, "v0_spearman": s4.get("spearman"),
            "best_real_v1_rank": int(np.argsort(np.argsort(-sb))[int(np.argmax(rb))]) + 1}


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 10 ** -nd):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _tbl(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return out


def stage_report(record: bool = False) -> dict[str, Any]:
    pop = json.loads((OUT / "population.json").read_text(encoding="utf-8"))
    syms = symbols()
    F = []
    for s in syms:
        f = load_facts(s)
        lp = OUT / "loo" / f"{s}.pkl"
        if lp.exists():
            L = pickle.loads(lp.read_bytes())
            f["pop"] = L["pop"]
            f["loo"] = L
        F.append(f)
    m8 = ms8_eval(F)
    p7 = float(pop["p_summary"]) if pop["criterion_met"] else 1.0
    p8 = float(m8["p"]) if m8["criterion_met"] else 1.0
    holm2 = _holm({"H-MS7": p7, "H-MS8": p8})
    block = _block_ps()
    holm_block = _holm({**block, "H-MS7": p7, "H-MS8": p8})

    def verdict(met, ph):
        return "supports" if met and ph < 0.05 else "inconclusive" if met else "contradicts"

    v7 = verdict(pop["criterion_met"], holm2["H-MS7"])
    v8 = verdict(m8["criterion_met"], holm2["H-MS8"])
    rep = _replay_summary()
    # featured out-of-sample instrument: the alt with the median log W
    alts = sorted([r for r in pop["rows"] if r["class"] == "alt"], key=lambda r: r["W"])
    feat = alts[len(alts) // 2]["symbol"]
    ff = next(f for f in F if f["symbol"] == feat)
    # calibration quality (FIT targets vs simulated) per symbol
    cal = []
    for s in syms:
        d = load_symbol(s)
        t, fn = d["targets"], d["calib"]
        cal.append({"symbol": s, "class": d["class"], **{f"t_{k}": t.get(k) for k in ("sigma_1m", "impact", "flow_acf_3s", "absr_acf_3s", "vr_3s_1m", "vr5")},
                    **{f"s_{k}": fn.get(k) for k in ("sigma_1m", "impact", "flow_acf_3s", "absr_acf_3s", "vr_3s_1m", "vr5")}})
    out = {"method": METHOD, "H-MS7": {**{k: pop[k] for k in ("primary", "universality", "regime", "ko", "criterion_r2", "criterion_universal",
                                                              "criterion_met", "p_summary", "n")}, "p_holm": holm2["H-MS7"], "verdict": v7},
           "H-MS8": {"counts": m8["counts"], "criterion_met": m8["criterion_met"], "p": m8["p"], "p_holm": holm2["H-MS8"], "verdict": v8,
                     "classes": {c: {"lo": e["lo"], "hi": e["hi"], "real": e["real"], **{v: e[v] for v in VARIANTS if v in e}} for c, e in m8["classes"].items()},
                     "level_vs_shape": m8["level_vs_shape"], "own_envelope_share": m8["own_envelope_share"], "boot_counts": m8["boot_counts"]},
           "holm_block": holm_block, "block_p": block, "featured": feat, "replay": rep, "calibration": cal, "secondary": pop["secondary"]}
    REPORT.with_suffix(".json").write_text(json.dumps(_jsonable(out), indent=1), encoding="utf-8")
    write_markdown(out, pop, F, ff)
    if record:
        _record(out)
    print("verdicts", v7, v8, "holm", holm2, flush=True)
    return out


PARAM_LABEL = {"log_rate": "log orders / s", "log_spread_bps": "log median relative spread (bp)", "log_lambda": "log Kyle lambda (mid proxy, bp per $1k)",
               "log_v_ref": "log median order notional", "size_mu": "size body mu (log USD)", "log_sigma_day": "log daily sigma", "log_depth25": "log depth within 25 bp",
               "sign_acf1": "order sign ACF lag 1", "sign_gamma": "order sign gamma", "zeta_order": "order size tail zeta", "ppo_mean": "prints per order",
               "mix_w1": "burst weight (mixture)", "mix_w2": "middle weight", "log_mix_m1": "log burst mean (x mean duration)", "log_mix_m2": "log middle mean",
               "log_mix_m3": "log slow mean", "log_lambda_trade": "log Kyle lambda (trade price)", "impact_corr": "3 s corr(r, F), pre-TEST book",
               "flow_acf_3s": "3 s signed-flow ACF (FIT)", "absr_acf_3s": "3 s |r| ACF of the mid proxy (FIT)", "log_k_imp": "log calibrated impact k"}


def _narr() -> dict[str, Any]:
    p = OUT / "narrative.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_markdown(out: dict, pop: dict, F: list[dict], ff: dict) -> None:
    N = _narr()
    m7, m8 = out["H-MS7"], out["H-MS8"]
    md: list[str] = []
    w = md.append
    w("# Population extrapolation and event-level simulator v1 (H-MS7, H-MS8)")
    w("")
    w("_Agent C3, microstructure pilot, 2026-11-11. Data: trades_raw 2026-11-10 (51 Bybit perps, every print), FIT 00-12 UTC, TEST 12-24 UTC "
      "(CPI at 13:30, about twice the activity); flow_3s TEST half for the real stylised facts; flow_3s 2026-11-07..09 (before TEST) for spread, depth and "
      "the book-mid shape targets. Segment: clean_holdout. Code: `src/ai_trader/sim/population.py`, `src/ai_trader/sim/flow_sim_v1.py`, "
      "`src/ai_trader/research/population_simulator_study.py`; tests `tests/test_population.py`, `tests/test_flow_sim_v1.py`; population model "
      "`data/models/population/population_2026-11-10.json`; machine-readable results `research/reports/population_simulator_2026-11-11.json`. "
      "Generated data is for deterministic strategies and the execution engine only, never for the LLM layer._")
    w("")
    if N.get("summary"):
        w("## Summary")
        w("")
        for line in N["summary"]:
            w(f"- {line}")
        w("")
    w("## Verdicts")
    w("")
    pr = m7["primary"]
    un = m7["universality"]
    c8 = m8["counts"]
    w("| id | registered success criterion | key numbers | p / Holm (MS7, MS8) | verdict |")
    w("|---|---|---|---|---|")
    w(f"| H-MS7 | LOO R^2 > 0.5 for rate, spread, impact; zeta and gamma dispersion not significantly above FIT-vs-TEST noise | "
      f"LOO R^2 rate {_f(pr['rate']['loo_r2'])} [{_f(pr['rate']['loo_ci'][0])}, {_f(pr['rate']['loo_ci'][1])}], spread {_f(pr['spread']['loo_r2'])} "
      f"[{_f(pr['spread']['loo_ci'][0])}, {_f(pr['spread']['loo_ci'][1])}], impact {_f(pr['impact']['loo_r2'])} [{_f(pr['impact']['loo_ci'][0])}, "
      f"{_f(pr['impact']['loo_ci'][1])}]; D(zeta) {_f(un['zeta_order']['D'], 2)} (p_disp {_f(un['zeta_order']['p_disp'])}), D(gamma) {_f(un['sign_gamma']['D'], 2)} "
      f"(p_disp {_f(un['sign_gamma']['p_disp'])}) | {_f(m7['p_summary'])} / {_f(m7['p_holm'])} | **{m7['verdict']}** |")
    w(f"| H-MS8 | v1 facts inside the TEST envelopes > v0 in every class and v1 >= 4/5 for majors | "
      f"v1 / v0: major {c8['major']['v1']}/{c8['major']['v0']}, alt {c8['alt']['v1']}/{c8['alt']['v0']}, tradfi {c8['tradfi']['v1']}/{c8['tradfi']['v0']} | "
      f"{_f(m8['p'])} / {_f(m8['p_holm'])} | **{m8['verdict']}** |")
    w("")
    w("p for H-MS7 = the largest bootstrap P*(LOO R^2 <= 0.5) of the three liquidity parameters (the universality part is a pass/fail "
      "condition: p_disp >= 0.05); p for H-MS8 = share of 1000 symbol-bootstrap replicates (envelopes and simulated medians resampled "
      "together) in which the criterion fails. A criterion that is not met gives contradicts; met with Holm p >= 0.05 gives inconclusive. "
      "Holm over the whole MS block with the recorded p values of H-MS1..6, 9: " +
      ", ".join(f"{k} {_f(v)}" for k, v in sorted(out["holm_block"].items())) + ".")
    w("")
    w("## Methods (fixed before the tests)")
    w("")
    for k, v in METHOD.items():
        w(f"- **{k}**: {v}")
    w("")
    # ---------------------------------------------------------------- H-MS7
    w("## H-MS7: what scales with liquidity")
    w("")
    if N.get("ms7"):
        for para in N["ms7"]:
            w(para)
            w("")
    w("Cross-sectional OLS on [1, log W, alt, tradfi], 51 symbols (FIT half). Slopes are per unit of log W; class effects are "
      "relative to majors. LOO R^2 = 1 - PRESS / TSS with a symbol bootstrap 95% CI.")
    w("")
    rows = []
    for k, d in [*pr.items(), *out["secondary"].items()]:
        name = d["param"]
        rows.append([PARAM_LABEL.get(name, name) + (" (primary)" if k in ("rate", "spread", "impact") else ""), d["n"],
                     f"{_f(d['coef'][1])} ({_f(d['se'][1])})", f"{_f(d['coef'][2], 2)} / {_f(d['coef'][3], 2)}", d["r2"], d["loo_r2"],
                     f"[{_f(d['loo_ci'][0])}, {_f(d['loo_ci'][1])}]", d["loo_r2_W_only"]])
    md += _tbl(["parameter", "n", "slope on log W (HC1 SE)", "alt / tradfi effect", "R^2", "LOO R^2", "LOO 95% CI", "LOO R^2 (log W only)"], rows)
    w("")
    w("Universality of the shape exponents (D = Var((F+T)/2) / (Var(F-T)/4); D = 1 when the between-symbol spread equals estimation noise; "
      "p_disp = bootstrap P*(D <= 1)):")
    w("")
    rows = []
    for k, d in un.items():
        rows.append([k, d["n"], d["mean"], d["between_sd"], d["noise_sd"], d["D"], f"[{_f(d['ci'][0], 2)}, {_f(d['ci'][1], 2)}]", d["p_disp"],
                     d["spearman_fit_test"], d["within_class"]["D"], d["within_class"]["p_disp"]])
    md += _tbl(["parameter", "n", "mean", "between-symbol sd", "noise sd (FIT-TEST)", "D", "D 95% CI", "p_disp", "Spearman FIT vs TEST",
                "D within class", "p_disp within class"], rows)
    g20 = un["sign_gamma"].get("ge20k")
    if g20:
        w("")
        w(f"Gamma on the C1 eligibility set (>= 20k prints, n = {g20['n']}): D {_f(g20['D'], 2)} [{_f(g20['ci'][0], 2)}, {_f(g20['ci'][1], 2)}], p_disp {_f(g20['p_disp'])}.")
    w("")
    w("Kyle-Obizhaeva invariance (log-log slopes on log W, HC1 SE):")
    w("")
    rows = [[k, d["n"], f"{_f(d['slope'])} ({_f(d['se'])})", _f(d["prediction"]), d["t_vs_prediction"], d["r2"]] for k, d in m7["ko"].items()]
    md += _tbl(["relation", "n", "slope (SE)", "invariance", "t vs invariance", "R^2"], rows)
    w("")
    rg = m7["regime"]
    w(f"Spread regime: tick-bound = at least {REGIME_SHARE:.0%} of pre-TEST 3 s candles below 1.5 ticks; {51 - rg['n_free']} tick-bound, {rg['n_free']} free "
      f"({', '.join(rg['free_symbols'])}). Logistic on [1, log tick_bps, log W]: coefficients {', '.join(_f(x, 2) for x in rg['coef'])}; accuracy in "
      f"sample {_f(rg['acc_in'])}, leave-one-out {_f(rg['acc_loo'])} (base rate {_f(rg['base_rate'])}), LOO Brier {_f(rg['brier_loo'])}. Free-spread level "
      f"(log bp on [1, log W, log tick_bps], n = {rg['n_free']}): coefficients {', '.join(_f(x, 3) for x in rg['free_spread_coef'])}, LOO R^2 "
      f"{_f(rg['free_spread_loo_r2'])}.")
    w("")
    # ---------------------------------------------------------------- H-MS8
    w("## H-MS8: simulator v1 against the TEST stylised facts")
    w("")
    if N.get("ms8"):
        for para in N["ms8"]:
            w(para)
            w("")
    for c in ("major", "alt", "tradfi"):
        e = m8["classes"][c]
        n_sym = sum(1 for f in F if f["class"] == c)
        w(f"**{c}** ({n_sym} symbols): v1 {c8[c]['v1']}/5, v0 {c8[c]['v0']}/5, v1 with TEST levels {c8[c].get('v1_level')}/5, "
          f"population v1 (leave-one-out) {c8[c].get('pop')}/5")
        w("")
        rows = []
        for j, k in enumerate(FACT_KEYS):
            fk = {"spread_med": "spread", "spread_p90": "spread"}.get(k, k)
            def ins(v, fk=fk, j=j, k=k, e=e):
                if v not in e:
                    return "n/a"
                if k == "continuation":
                    return "yes" if e["lo"][j] <= e[v]["value"][j] <= e["hi"][j] else "no"
                return "yes" if e[v]["inside"][fk] else "no"
            rows.append([FACT_LABEL[k], e["real"][j], f"[{_f(e['lo'][j])}, {_f(e['hi'][j])}]", e["v1"]["value"][j], ins("v1"), e["v0"]["value"][j], ins("v0"),
                         e["v1_level"]["value"][j], ins("v1_level"), e["pop"]["value"][j] if "pop" in e else None, ins("pop"),
                         m8["level_vs_shape"][c].get(fk, "") if k != "continuation" else ""])
        md += _tbl(["fact", "real TEST", "envelope", "v1", "in", "v0", "in", "v1 TEST levels", "in", "population v1", "in", "v1 failure"], rows)
        w("")
    w("Share of symbols whose own simulated fact lies inside their own TEST envelope (v1 / v0):")
    w("")
    rows = []
    for c in ("major", "alt", "tradfi"):
        o = m8["own_envelope_share"][c]
        rows.append([c, *[f"{_f(o['v1'][k], 2)} / {_f(o['v0'][k], 2)}" for k in FACTS5]])
    md += _tbl(["class", *FACTS5], rows)
    w("")
    w("Calibration check (FIT / pre-TEST target vs simulated value on the calibration pilots, class medians):")
    w("")
    cal = out["calibration"]
    rows = []
    for c in ("major", "alt", "tradfi"):
        cs = [x for x in cal if x["class"] == c]
        rows.append([c, *[f"{_f(np.nanmedian([x['t_' + k] if x['t_' + k] is not None else np.nan for x in cs]))} / "
                          f"{_f(np.nanmedian([x['s_' + k] if x['s_' + k] is not None else np.nan for x in cs]))}"
                          for k in ("sigma_1m", "impact", "flow_acf_3s", "absr_acf_3s", "vr_3s_1m", "vr5")]])
    md += _tbl(["class", "sigma 1 min (bp)", "corr(r, F) 3 s", "flow ACF 3 s", "|r| ACF 3 s (proxy)", "VR 3 s -> 1 min", "VR 1 -> 5 min"], rows)
    w("")
    # ---------------------------------------------------------------- featured instrument
    fs_ = out["featured"]
    w(f"## The extrapolated instrument: {fs_} built only from params_for(W, class, tick)")
    w("")
    if N.get("featured"):
        for para in N["featured"]:
            w(para)
            w("")
    L = ff.get("loo")
    if L:
        lo, hi = np.nanpercentile(ff["boot"], 2.5, axis=0), np.nanpercentile(ff["boot"], 97.5, axis=0)
        rows = []
        for j, k in enumerate(FACT_KEYS):
            rows.append([FACT_LABEL[k], ff["real"][j], f"[{_f(lo[j])}, {_f(hi[j])}]", np.nanmedian(ff["v1"], axis=0)[j], np.nanmedian(L["pop"], axis=0)[j],
                         np.nanmedian(ff["v0"], axis=0)[j]])
        md += _tbl(["fact", "real TEST", "own envelope", "own v1 (fitted on its FIT half)", "population v1 (never saw the symbol)", "v0"], rows)
        w("")
        rows = [[k, L["actual"].get(k), L["pred"].get(k)] for k in ("log_rate", "log_spread_bps", "regime", "log_sigma_day", "log_v_ref", "sign_gamma", "zeta_order")]
        md += _tbl(["parameter", "actual (FIT)", "predicted from the other 50"], rows)
        w("")
    # all-symbol LOO prediction errors
    errs = {k: [] for k in ("log_rate", "log_spread_bps", "log_sigma_day", "log_v_ref")}
    reg_ok = []
    for f in F:
        if "loo" in f:
            for k in errs:
                a, p_ = f["loo"]["actual"].get(k), f["loo"]["pred"].get(k)
                if a is not None and p_ is not None:
                    errs[k].append(p_ - a)
            reg_ok.append(f["loo"]["actual"]["regime"] == f["loo"]["pred"]["regime"])
    if reg_ok:
        w("Leave-one-out prediction errors over all 51 symbols (predicted minus actual, log units; median absolute error and the factor it "
          f"implies): " + "; ".join(f"{k} {_f(np.median(np.abs(v)), 2)} (x{_f(math.exp(np.median(np.abs(v))), 2)})" for k, v in errs.items() if v) +
          f"; spread regime right for {sum(reg_ok)}/{len(reg_ok)}.")
        w("")
    # ---------------------------------------------------------------- replay
    rep = out["replay"]
    w("## Market-making replay on v1 paths (descriptive, not a verdict)")
    w("")
    if rep:
        if N.get("replay"):
            for para in N["replay"]:
                w(para)
                w("")
        rows = []
        for key, nm in (("live", "live mm_multi"), ("best", "replay-best a0.2_b0.9_tp0.9_rq0.15")):
            r = rep[key]
            v0 = rep["v0"].get(key, {})
            rows.append([nm, r["real_bps"], r["v1_bps"], v0.get("sim_bps"), r["real_pnl"], r["v1_pnl"], r["real_turnover"], r["v1_turnover"], r["real_fills"],
                         r["v1_fills"], r["real_pos"], r["v1_pos"]])
        md += _tbl(["parameters", "real bp of turnover", "v1 bp", "v0 bp (H-SIM4)", "real P&L USD", "v1 P&L USD", "real turnover", "v1 turnover",
                    "real fills", "v1 fills", "real symbols > 0", "v1 symbols > 0"], rows)
        w("")
        w(f"Over the 36-set grid ({rep['symbols']} symbols, 3 seeds): Spearman(real, v1) of net bp per turnover {_f(rep['spearman_bps'])} "
          f"(v0 in H-SIM4: {_f(rep['v0_spearman'])}), of turnover {_f(rep['spearman_turnover'])}, of fill counts {_f(rep['spearman_fills'])}; the real-best "
          f"set ranks {rep['best_real_v1_rank']} of 36 on v1.")
        w("")
    # ---------------------------------------------------------------- multi-day, caveats, rules
    if N.get("multiday"):
        w("## What a multi-day pilot would most improve")
        w("")
        for line in N["multiday"]:
            w(f"- {line}")
        w("")
    if N.get("caveats"):
        w("## Caveats")
        w("")
        for line in N["caveats"]:
            w(f"- {line}")
        w("")
    if N.get("rules_yaml"):
        w("## Rule candidates")
        w("")
        w("```yaml")
        w(N["rules_yaml"].rstrip())
        w("```")
    REPORT.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _record(out: dict) -> None:
    from ai_trader.research import hypotheses as H

    data = "trades_raw 2026-11-10 (51 symbols), FIT 00-12 / TEST 12-24 UTC; flow_3s TEST half + 2026-11-07..09; research/reports/population_simulator_2026-11-11.md"
    m7, m8 = out["H-MS7"], out["H-MS8"]
    if not H.results("H-MS7"):
        pr, un = m7["primary"], m7["universality"]
        H.record("H-MS7", "clean_holdout", m7["verdict"], _jsonable({
            "p_summary": m7["p_summary"], "p_holm": m7["p_holm"], "criterion_met": m7["criterion_met"], "criterion_r2": m7["criterion_r2"],
            "criterion_universal": m7["criterion_universal"], "n_symbols": m7["n"],
            "loo_r2": {k: pr[k]["loo_r2"] for k in pr}, "loo_ci": {k: pr[k]["loo_ci"] for k in pr}, "slope_logW": {k: pr[k]["coef"][1] for k in pr},
            "D": {k: un[k]["D"] for k in ("zeta_order", "sign_gamma")}, "D_ci": {k: un[k]["ci"] for k in ("zeta_order", "sign_gamma")},
            "p_disp": {k: un[k]["p_disp"] for k in ("zeta_order", "sign_gamma")},
            "ko_slopes": {k: v["slope"] for k, v in m7["ko"].items()}}),
            note="Population model: OLS on log W (sigma_day x dollar volume, FIT trades) and class; LOO R^2 with symbol bootstrap; universality "
                 "D = Var((F+T)/2) / (Var(F-T)/4). Holm over H-MS7, H-MS8. One day.", data=data)
    if not H.results("H-MS8"):
        H.record("H-MS8", "clean_holdout", m8["verdict"], _jsonable({
            "p": m8["p"], "p_holm": m8["p_holm"], "criterion_met": m8["criterion_met"], "counts": m8["counts"],
            "level_vs_shape": m8["level_vs_shape"]}),
            note="v1 event-level generator (orders, spliced sizes, translated signs, sqrt impact with decay, trade-time noise, jumps, censored "
                 "spread) calibrated on FIT trades + pre-TEST book shape; envelopes = 1 h block bootstrap of TEST class medians; v0 = H-SIM2 models. "
                 "Holm over H-MS7, H-MS8. One day.", data=data)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    syms = [s for s in a.symbols.split(",") if s] or symbols()
    if a.stage in ("symbol", "all"):
        stage_symbol(syms, a.force)
    if a.stage in ("facts", "all"):
        stage_generic(_safe_facts, "facts", syms, a.force)
    if a.stage in ("population", "all"):
        stage_population()
    if a.stage in ("loo", "all"):
        stage_generic(_safe_loo, "loo", syms, a.force)
    if a.stage in ("replay", "all"):
        stage_generic(_safe_replay, "replay", [s for s in mm_symbols() if s in syms], a.force)
    if a.stage in ("report", "all"):
        stage_report(record=a.record)


if __name__ == "__main__":
    main(sys.argv[1:])
