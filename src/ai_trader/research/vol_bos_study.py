"""Volatility break of structure as a trend-following filter (H-VBOS1-4).

Plan: research/plans/2026-11-13_vol_bos_trend.md. The full design (families, grids, objectives, nested purged
walk-forward, purge/embargo, declustering, verdict rules) was fixed in the diary (2026-11-13, kind 'hypothesis',
"VBOS design fixed before testing") before any number was computed. Features: ai_trader.features.vol_bos.

Pipeline
  1. cache   per (symbol, freq) arrays: bars, trend-ensemble signal + P&L (evaluate.strategy_pnl), intrabar RV /
             semivariances from 1h bars, 62 vol-BoS onset series, 27 semivariance-break series, 10-day dominance,
             fractal BoS / CHoCH for two k. L = trend_zoo_study.UNIVERSE (23 crypto), T = 16 TradFi perps.
  2. insample  VBOS1 / VBOS2 event studies and the VBOS3 strategy grid inside llm_in_sample with the nested purged
             walk-forward; final selections frozen to <cache>/selection.json; DSR, PBO (CSCV, 16 blocks); VBOS4
             split-half surfaces. --record writes the llm_in_sample verdicts.
  3. holdout   ONE look at clean_holdout with the frozen selections (L verdicts, T robustness, BTC look), VBOS4
             holdout surfaces, figures, report json + md. --record writes clean_holdout and pooled verdicts.

    uv run python -m ai_trader.research.vol_bos_study --stage cache  [--cache DIR]
    uv run python -m ai_trader.research.vol_bos_study --stage insample [--record]
    uv run python -m ai_trader.research.vol_bos_study --stage holdout  [--record]
    uv run python -m ai_trader.research.vol_bos_writeup [--cache DIR]          (figures + report from the saved stages)

Units: P&L in risk units per day (sum of per-bar strategy_pnl, dated by the UTC day the bar's return accrues);
VBOS2 returns in daily-sigma units. Costs 7.5 bps per side.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features import vol_bos as vb
from ai_trader.research.trend_zoo_study import UNIVERSE, deflated_sharpe

UNIV_L = tuple(UNIVERSE)
UNIV_T = tuple("XAUUSDT XAGUSDT CLUSDT BZUSDT TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT "
               "SOXLUSDT SPCXUSDT PAXGUSDT XAUTUSDT".split())
FREQS = {"1d": 1, "4h": 6}
SEGS = ("llm_in_sample", "clean_holdout")
DAY0 = date(2020, 1, 1)
DAY_US = 86_400_000_000
H = (5, 10, 20)
H_PRIMARY = 10
H_MAX = 20
EMBARGO = 5
DECLUSTER_DAYS = 20
ACTIVE_DAYS = 20
DOM_DAYS = 10
WARM_DAYS = 200
MIN_EV_FOLD = 10
MIN_FOLD_DAYS = 60
K_STRUCT = {1: (2, 5), 6: (3, 12)}
BAND = 0.2
N_PLACEBO = 200
ND = NormalDist()
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def cost_side() -> float:
    c = load().risk["costs"]
    return (float(c["taker_bps"]) + float(c["slippage_bps"])) / 1e4


def dix(d: date) -> int:
    return (d - DAY0).days


def ddate(i: int) -> str:
    return (DAY0 + timedelta(days=int(i))).isoformat()


def cutoff_day() -> int:
    from ai_trader.research.evaluate import knowledge_cutoff

    return dix(knowledge_cutoff().date())


def default_cache() -> Path:
    return load().root / "data" / "cache" / "vol_bos"


# ====================================================================== 1. per-symbol cache
def _aggregate(h1: pl.DataFrame, bar_h: int) -> pl.DataFrame:
    """TradFi: target bars from 1h bars (no native 4h/1d history on bybit for most T symbols)."""
    g = (h1.sort("event_ts").group_by(pl.col("event_ts").dt.truncate(f"{bar_h}h").alias("bt"), maintain_order=True)
         .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
              pl.col("volume").sum(), pl.col("turnover").sum()))
    return (g.rename({"bt": "event_ts"}).with_columns((pl.col("event_ts") + pl.duration(hours=bar_h)).alias("close_ts"))
            .with_columns(pl.col("close_ts").alias("known_ts")).sort("event_ts"))


def symbol_data(sym: str, freq: str, as_of: datetime, tradfi: bool = False) -> dict | None:
    from ai_trader.research.evaluate import strategy_pnl
    from ai_trader.research.frame import add_robust_vol, enrich_bars, market_frame
    from ai_trader.strategies.trend_zoo import ensemble, lookback_bars

    bpd = FREQS[freq]
    bar_h = 24 // bpd
    h1 = market_frame(sym, "1h", as_of, funding=False, oi=False)
    if h1.height < 24 * 30:
        return None
    if tradfi:
        f = add_robust_vol(enrich_bars(_aggregate(h1, bar_h)))
        lb = 30 * bpd
    else:
        f = market_frame(sym, freq, as_of, funding=False, oi=False)
        lb = max(lookback_bars(bpd), WARM_DAYS * bpd)
    if f.height < lb + 30 * bpd:
        return None
    ts = f["event_ts"].dt.epoch("us").to_numpy()
    n = len(ts)
    ens = ensemble(f, bpd).to_numpy().astype(float)
    p = strategy_pnl(f, pl.Series("ens", ens))
    rm = vb.bar_realized(h1.select("event_ts", "close"), f["event_ts"].to_numpy(), bar_h)
    slot = ((ts // 3_600_000_000) % 24) // bar_h
    rv_adj = rm["rv"] / vb.seasonal_factor(rm["rv"], slot, bpd)
    det = vb.detector_onsets(rv_adj, bpd)
    semi = np.stack([vb.semi_breaks(rm["rs_pos"], rm["rs_neg"], bpd, w, nn, q) for w, nn, q in vb.semi_grid()])
    dom = vb.semi_dominance(rm["rs_pos"], rm["rs_neg"], bpd, DOM_DAYS)
    bos, choch = zip(*[vb.structure(f["close"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy(), k) for k in K_STRUCT[bpd]])
    day0_us = _day0_us()
    bar_us = bar_h * 3_600_000_000
    acc_day = ((ts + bar_us - day0_us) // DAY_US).astype(np.int64)        # day on which bar t's fwd return accrues
    fo_day = (-((-(ts + bar_us - day0_us)) // DAY_US)).astype(np.int64)    # first full accrual day after the bar's close
    start_day = int((ts[lb] - day0_us) // DAY_US) + 1
    return {
        "sym": sym, "freq": freq, "bpd": bpd, "tradfi": tradfi, "lb": lb, "start_day": start_day,
        "ts": ts, "close": f["close"].to_numpy(), "high": f["high"].to_numpy(), "low": f["low"].to_numpy(),
        "sigma_next": f["sigma_next"].to_numpy(), "fwd_ret": np.nan_to_num(f["fwd_ret"].to_numpy()),
        "ens": ens, "ens_pnl": p["pnl"].to_numpy(), "acc_day": acc_day, "fo_day": fo_day,
        "rv": rm["rv"], "rv_adj": rv_adj, "rs_pos": rm["rs_pos"], "rs_neg": rm["rs_neg"],
        "det": det, "semi": semi, "dom": dom, "bos": np.stack(bos), "choch": np.stack(choch),
    }


def _day0_us() -> int:
    return int((datetime(DAY0.year, DAY0.month, DAY0.day) - datetime(1970, 1, 1)).total_seconds()) * 1_000_000


def cache_path(cache: Path, sym: str, freq: str) -> Path:
    return cache / f"{sym}_{freq}.npz"


def save_symbol(cache: Path, d: dict) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    meta = {k: d[k] for k in ("sym", "freq", "bpd", "tradfi", "lb", "start_day")}
    arrs = {k: v for k, v in d.items() if isinstance(v, np.ndarray)}
    tmp = cache_path(cache, d["sym"], d["freq"]).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrs)
    os.replace(tmp, cache_path(cache, d["sym"], d["freq"]))


def load_symbol(cache: Path, sym: str, freq: str) -> dict | None:
    p = cache_path(cache, sym, freq)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    d = json.loads(str(z["meta"]))
    d.update({k: z[k] for k in z.files if k != "meta"})
    return d


def build_cache(cache: Path, as_of: datetime, symbols=None) -> list[str]:
    done = []
    syms = [(s, False) for s in UNIV_L] + [(s, True) for s in UNIV_T] if symbols is None else symbols
    for s, tf in syms:
        for fr in FREQS:
            if cache_path(cache, s, fr).exists():
                done.append(f"{s}_{fr}")
                continue
            t0 = time.time()
            d = symbol_data(s, fr, as_of, tradfi=tf)
            if d is None:
                print(f"skip {s} {fr}", flush=True)
                continue
            save_symbol(cache, d)
            done.append(f"{s}_{fr}")
            print(f"cached {s} {fr} n={len(d['ts'])} start={ddate(d['start_day'])} {time.time() - t0:.1f}s", flush=True)
    return done


# ====================================================================== statistics
def nw_t(x: np.ndarray, min_lag: int = 0) -> tuple[float, float, int]:
    """Mean, Newey-West t (Bartlett, automatic lag floor(4 (n/100)^(2/9)), at least min_lag), n."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return (float(x.mean()) if n else float("nan"), float("nan"), n)
    m = x.mean()
    e = x - m
    lag = max(min_lag, int(np.floor(4 * (n / 100) ** (2 / 9))))
    lag = min(lag, n - 1)
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(v, 1e-300) / n)
    return float(m), float(m / se), n


def clustered_t(vals: np.ndarray, days: np.ndarray, bucket: int = H_PRIMARY, min_lag: int = 2) -> dict:
    """Date-clustered event mean: mean of 10-day calendar-bucket means, NW t over the bucket series (lag >= 2)."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    v, dd = v[ok], np.asarray(days)[ok]
    if len(v) == 0:
        return {"mean": float("nan"), "t": float("nan"), "n_events": 0, "n_clusters": 0, "event_mean": float("nan")}
    b = dd // bucket
    ub, inv = np.unique(b, return_inverse=True)
    bm = np.bincount(inv, weights=v) / np.bincount(inv)
    m, t, nb = nw_t(bm, min_lag)
    return {"mean": m, "t": t, "n_events": int(len(v)), "n_clusters": int(nb), "event_mean": float(v.mean())}


def p_one(t: float) -> float:
    return 1.0 - ND.cdf(t) if np.isfinite(t) else 1.0


def holm(ps: dict[str, float]) -> dict[str, float]:
    keys = sorted(ps, key=lambda k: ps[k])
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def sharpe_ann(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(365)) if sd > 0 else float("nan")


def pbo_cscv_chunked(perf: np.ndarray, n_blocks: int = 16, chunk: int = 1000) -> dict:
    """Same estimator as synthetic_robustness_study.pbo_cscv, evaluated in chunks of splits (memory)."""
    x = np.where(np.isfinite(perf), perf, 0.0).astype(float)
    t, n = x.shape
    L = t // n_blocks
    xb = x[: L * n_blocks].reshape(n_blocks, L, n)
    s1, s2 = xb.sum(axis=1), (xb**2).sum(axis=1)
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    lam_all, rank_all, isb_all, oosb_all, best_all = [], [], [], [], []

    def sharpe(w: np.ndarray) -> np.ndarray:
        cnt = w.sum(axis=1, keepdims=True) * L
        m = (w @ s1) / cnt
        v = ((w @ s2) / cnt - m**2) * cnt / (cnt - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, m / np.sqrt(np.where(v > 0, v, 1.0)), -np.inf)

    for c0 in range(0, len(combos), chunk):
        cb = combos[c0:c0 + chunk]
        ind = np.zeros((len(cb), n_blocks))
        ind[np.arange(len(cb))[:, None], cb] = 1.0
        sr_is, sr_oos = sharpe(ind), sharpe(1.0 - ind)
        idx = np.arange(len(cb))
        best = np.argmax(sr_is, axis=1)
        ob = sr_oos[idx, best]
        rank = (sr_oos < ob[:, None]).sum(axis=1) + 0.5 * ((sr_oos == ob[:, None]).sum(axis=1) - 1) + 1
        omega = rank / (n + 1)
        lam_all.append(np.log(omega / (1 - omega)))
        rank_all.append(rank)
        isb_all.append(sr_is[idx, best])
        oosb_all.append(ob)
        best_all.append(best)
    lam, rank = np.concatenate(lam_all), np.concatenate(rank_all)
    isb, oob = np.concatenate(isb_all), np.concatenate(oosb_all)
    ok = np.isfinite(isb) & np.isfinite(oob)
    slope = float(np.polyfit(isb[ok], oob[ok], 1)[0]) if ok.sum() > 2 and np.std(isb[ok]) > 0 else float("nan")
    best = np.concatenate(best_all)
    return {"pbo": float(np.mean(lam <= 0)), "logit_median": float(np.median(lam)), "n_splits": len(combos), "n_blocks": n_blocks,
            "block_len": L, "oos_rel_rank_mean_of_is_winner": float(np.mean(rank / n)), "degradation_slope": slope,
            "is_winner_counts": np.bincount(best, minlength=n).tolist(), "logits": lam.tolist()}


# ====================================================================== 2. nested purged walk-forward
def half_years(a: int, b: int) -> list[tuple[int, int]]:
    """Calendar half-years intersecting [a, b), clipped to it (day indices)."""
    out = []
    d = DAY0 + timedelta(days=a)
    y, h = d.year, 0 if d.month <= 6 else 1
    while True:
        s = dix(date(y, 1 if h == 0 else 7, 1))
        e = dix(date(y, 7, 1) if h == 0 else date(y + 1, 1, 1))
        if s >= b:
            break
        lo, hi = max(s, a), min(e, b)
        if hi > lo:
            out.append((lo, hi))
        y, h = (y, 1) if h == 0 else (y + 1, 0)
    return out


def outer_blocks() -> list[tuple[int, int]]:
    return half_years(dix(date(2022, 1, 1)), cutoff_day())


def inner_folds(first: int, train_end: int) -> list[tuple[int, int]]:
    return [f for f in half_years(first, train_end) if f[1] - f[0] >= MIN_FOLD_DAYS]


def outer_plan(first: int, gap: int) -> list[dict]:
    """Per outer block: its inner folds (all strictly before block start - gap)."""
    plan = []
    for a, b in outer_blocks():
        te = a - gap
        plan.append({"block": (a, b), "train_end": te, "folds": inner_folds(first, te)})
    return plan


def select_median(scores: np.ndarray, min_valid: int = 2) -> tuple[int, float]:
    """scores: (n_folds, n_cfg). Median over valid folds; configs with < min_valid folds are not eligible."""
    valid = np.isfinite(scores).sum(axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(np.isfinite(scores), scores, np.nan), axis=0) if scores.shape[0] else np.full(scores.shape[1], np.nan)
    med = np.where(valid >= min_valid, med, -np.inf)
    if not np.isfinite(med).any():
        return -1, float("nan")
    i = int(np.argmax(med))
    return i, float(med[i])


# ====================================================================== event tables
@dataclass
class Events:
    sym: np.ndarray
    cfg: np.ndarray
    d0: np.ndarray
    out: dict[int, np.ndarray]
    side: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def take(self, m: np.ndarray) -> Events:
        return Events(self.sym[m], self.cfg[m], self.d0[m], {h: v[m] for h, v in self.out.items()},
                      None if self.side is None else self.side[m], {k: v[m] for k, v in self.extra.items()})


def ens_daily(d: dict, end_day: int) -> np.ndarray:
    """Daily ensemble P&L of the symbol from its start_day to end_day (inclusive)."""
    m = (d["acc_day"] >= d["start_day"]) & (d["acc_day"] <= end_day)
    return np.bincount(d["acc_day"][m] - d["start_day"], weights=d["ens_pnl"][m], minlength=end_day - d["start_day"] + 1)


def vbos1_events(datas: list[dict], end_day: int) -> Events:
    rows = {"sym": [], "cfg": [], "d0": [], **{h: [] for h in H}}
    for si, d in enumerate(datas):
        bpd, lb = d["bpd"], d["lb"]
        daily = ens_daily(d, end_day)
        cs = np.r_[0.0, np.cumsum(daily)]
        for ci in range(d["det"].shape[0]):
            on = d["det"][ci].copy()
            on[:lb] = False
            for t in np.flatnonzero(vb.decluster(on, DECLUSTER_DAYS * bpd)):
                d0 = int(d["fo_day"][t])
                o = d0 - d["start_day"]
                if o < 0:
                    continue
                rows["sym"].append(si)
                rows["cfg"].append(ci)
                rows["d0"].append(d0)
                for h in H:
                    rows[h].append((cs[o + h] - cs[o]) / h if o + h <= len(daily) else np.nan)
    return Events(np.array(rows["sym"], int), np.array(rows["cfg"], int), np.array(rows["d0"], int), {h: np.array(rows[h], float) for h in H})


def vbos2_events(datas: list[dict], end_day: int) -> Events:
    c2 = 2 * cost_side()
    rows = {"sym": [], "cfg": [], "d0": [], "side": [], "mom": [], **{h: [] for h in H}}
    grid = vb.semi_grid()
    for si, d in enumerate(datas):
        bpd, lb, close, sig, n = d["bpd"], d["lb"], d["close"], d["sigma_next"], len(d["ts"])
        last_t = np.searchsorted(d["acc_day"], end_day, side="right") - 1
        for ci, (w, _, _) in enumerate(grid):
            s = d["semi"][ci]
            for side in (1, -1):
                on = s == side
                on[:lb] = False
                for t in np.flatnonzero(vb.decluster(on, DECLUSTER_DAYS * bpd)):
                    sd = sig[t] * math.sqrt(bpd)
                    if not (np.isfinite(sd) and sd > 0):
                        continue
                    rows["sym"].append(si)
                    rows["cfg"].append(ci)
                    rows["d0"].append(int(d["fo_day"][t]))
                    rows["side"].append(side)
                    j = t - w * bpd
                    rows["mom"].append(np.sign(math.log(close[t] / close[j])) if j >= 0 else 0.0)
                    for h in H:
                        e = t + h * bpd
                        rows[h].append((side * math.log(close[e] / close[t]) - c2) / (sd * math.sqrt(h)) if e <= last_t + 1 and e < n else np.nan)
    return Events(np.array(rows["sym"], int), np.array(rows["cfg"], int), np.array(rows["d0"], int), {h: np.array(rows[h], float) for h in H},
                  np.array(rows["side"], int), {"mom": np.array(rows["mom"], float)})


class Baseline:
    """Mean daily ensemble P&L per symbol over a day window (the VBOS1 unconditional comparison)."""

    def __init__(self, datas: list[dict], end_day: int):
        self.start = np.array([d["start_day"] for d in datas])
        self.cs = [np.r_[0.0, np.cumsum(ens_daily(d, end_day))] for d in datas]

    def window(self, a: int, b: int) -> np.ndarray:
        out = np.full(len(self.cs), np.nan)
        for i, cs in enumerate(self.cs):
            lo, hi = max(a - self.start[i], 0), min(b - self.start[i], len(cs) - 1)
            if hi > lo:
                out[i] = (cs[hi] - cs[lo]) / (hi - lo)
        return out


def ev_values(ev: Events, h: int, base: Baseline | None, a: int, b: int) -> np.ndarray:
    v = ev.out[h]
    return v - base.window(a, b)[ev.sym] if base is not None else v


def ev_stat_by_cfg(ev: Events, vals: np.ndarray, mask: np.ndarray, cfgs: list[int], min_ev: int) -> np.ndarray:
    out = np.full(len(cfgs), np.nan)
    for j, c in enumerate(cfgs):
        m = mask & (ev.cfg == c) & np.isfinite(vals)
        if m.sum() >= min_ev:
            out[j] = clustered_t(vals[m], ev.d0[m])["t"]
    return out


def ev_select(ev: Events, base: Baseline | None, cfgs: list[int], folds: list[tuple[int, int]], limit: int, min_ev: int) -> tuple[int, float]:
    """Inner loop: config with the best median fold t (h = 10); events need d0 + H_MAX <= limit (purge)."""
    sc = np.full((len(folds), len(cfgs)), np.nan)
    for fi, (a, b) in enumerate(folds):
        vals = ev_values(ev, H_PRIMARY, base, a, b)
        m = (ev.d0 >= a) & (ev.d0 < b) & (ev.d0 + H_MAX <= limit)
        sc[fi] = ev_stat_by_cfg(ev, vals, m, cfgs, min_ev)
    j, s = select_median(sc)
    return (cfgs[j] if j >= 0 else -1), s


def ev_walk_forward(ev: Events, base: Baseline | None, cfgs: list[int], first: int, min_ev: int = MIN_EV_FOLD) -> dict:
    cut = cutoff_day()
    plan = outer_plan(first, EMBARGO)
    picks, parts = [], {h: [] for h in H}
    days = []
    for p in plan:
        a, b = p["block"]
        c, s = ev_select(ev, base, cfgs, p["folds"], p["train_end"], min_ev)
        picks.append({"block": [ddate(a), ddate(b)], "cfg": c, "inner_score": s, "n_folds": len(p["folds"])})
        if c < 0:
            continue
        m = (ev.cfg == c) & (ev.d0 >= a) & (ev.d0 < b) & (ev.d0 + H_MAX <= cut)
        for h in H:
            parts[h].append(ev_values(ev, h, base, a, b)[m])
        days.append(ev.d0[m])
    dd = np.concatenate(days) if days else np.zeros(0, int)
    res = {"picks": picks}
    for h in H:
        vv = np.concatenate(parts[h]) if parts[h] else np.zeros(0)
        res[f"h{h}"] = clustered_t(vv, dd)
    # final selection on all of llm_in_sample (gap before the cutoff), frozen for the holdout
    folds = inner_folds(first, cut - EMBARGO)
    c, s = ev_select(ev, base, cfgs, folds, cut - EMBARGO, min_ev)
    res["final"] = {"cfg": c, "inner_score": s, "n_folds": len(folds)}
    return res


def ev_segment(ev: Events, base: Baseline | None, cfg: int, a: int, b: int, h_limit: int) -> dict:
    """Stats of one config over events with d0 in [a, b) and outcome complete before h_limit."""
    out = {}
    for h in H:
        m = (ev.cfg == cfg) & (ev.d0 >= a) & (ev.d0 < b) & (ev.d0 + h <= h_limit)
        out[f"h{h}"] = clustered_t(ev_values(ev, h, base, a, b)[m], ev.d0[m])
    return out


# ====================================================================== VBOS3 strategy machinery
def next_index_arrays(sgn: np.ndarray, bos: np.ndarray) -> dict:
    return {"not_pos": np.flatnonzero(sgn != 1), "not_neg": np.flatnonzero(sgn != -1),
            "bos_dn": np.flatnonzero(bos == -1), "bos_up": np.flatnonzero(bos == 1)}


def _next_after(idx: np.ndarray, a: int, n: int) -> int:
    j = np.searchsorted(idx, a, side="right")
    return int(idx[j]) if j < len(idx) else n


def trade_pnl(s: np.ndarray, sig: np.ndarray, fwd: np.ndarray, a: int, b: int, c: float, band: float = BAND) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar P&L of a position = s on bars a..b-1, flat from bar b (exit cost at b if b < n), with exactly the
    evaluate.strategy_pnl execution policy (sized at entry, re-sized on a > band signal move)."""
    n = len(s)
    end = min(b, n)
    bars = list(range(a, end)) + ([b] if b < n else [])
    out = np.zeros(len(bars))
    held_s, lev, prev = 0.0, 0.0, 0.0
    for j, t in enumerate(range(a, end)):
        st = s[t]
        if held_s == 0.0 or np.sign(st) != np.sign(held_s) or abs(st - held_s) > band:
            held_s = st
            lev = st / sig[t] if np.isfinite(sig[t]) and sig[t] > 0 else 0.0
        out[j] = lev * fwd[t] - c * abs(lev - prev)
        prev = lev
    if b < n:
        out[-1] = -c * abs(prev)
    return np.array(bars, int), out


@dataclass
class Trades:
    a: np.ndarray
    d: np.ndarray
    b: np.ndarray
    held: np.ndarray
    offs: np.ndarray
    days: np.ndarray
    vals: np.ndarray
    choch: np.ndarray


def build_trades(d: dict, k_idx: int, exit_struct: bool, c: float, entries: np.ndarray | None = None, dirs: np.ndarray | None = None) -> Trades:
    s, sig, fwd = d["ens"], d["sigma_next"], d["fwd_ret"]
    n = len(s)
    sgn = np.sign(s)
    bos = d["bos"][k_idx]
    if entries is None:
        ok = (bos != 0) & (sgn == bos) & (d["acc_day"] >= d["start_day"])
        entries = np.flatnonzero(ok)
        dirs = bos[entries]
    nx = next_index_arrays(sgn, bos)
    A, D, B, HELD, OFF, DAYS, VALS = [], [], [], [], [0], [], []
    for a, dd in zip(entries, dirs):
        b = _next_after(nx["not_pos"] if dd == 1 else nx["not_neg"], a, n)
        if exit_struct:
            b = min(b, _next_after(nx["bos_dn"] if dd == 1 else nx["bos_up"], a, n))
        bars, v = trade_pnl(s, sig, fwd, int(a), int(b), c)
        A.append(a)
        D.append(dd)
        B.append(b)
        HELD.append(min(b, n) - a)
        DAYS.append(d["acc_day"][bars])
        VALS.append(v)
        OFF.append(OFF[-1] + len(v))
    cat = (lambda x, dt: np.concatenate(x).astype(dt) if x else np.zeros(0, dt))
    return Trades(np.array(A, int), np.array(D, int), np.array(B, int), np.array(HELD, int), np.array(OFF, int),
                  cat(DAYS, np.int64), cat(VALS, float), d["choch"][k_idx][np.array(A, int)] if A else np.zeros(0, bool))


def take_trades(tr: Trades, permitted: np.ndarray) -> list[int]:
    out, cur = [], -1
    for j in np.flatnonzero(permitted):
        if tr.a[j] >= cur:
            out.append(int(j))
            cur = int(tr.b[j])
    return out


def trades_daily(tr: Trades, taken: list[int], start_day: int, n_days: int) -> np.ndarray:
    if not taken:
        return np.zeros(n_days)
    idx = np.concatenate([np.arange(tr.offs[j], tr.offs[j + 1]) for j in taken])
    dd = tr.days[idx] - start_day
    m = (dd >= 0) & (dd < n_days)
    return np.bincount(dd[m], weights=tr.vals[idx][m], minlength=n_days)


def strat_grid(bpd: int) -> list[tuple[int, int, int, int]]:
    """(detector index, k index, semi on/off, exit_struct) in grid order."""
    nd = len(vb.detector_grid(bpd))
    return [(i, k, s, e) for i in range(nd) for k in range(2) for s in (0, 1) for e in (0, 1)]


def cfg_label(freq: str, cfg: tuple[int, int, int, int]) -> dict:
    bpd = FREQS[freq]
    i, k, s, e = cfg
    det = vb.detector_grid(bpd)[i]
    return {"freq": freq, "detector": det.name, "family": det.family, "fractal_k_bars": K_STRUCT[bpd][k],
            "semi_filter": bool(s), "exit": "own+struct" if e else "own"}


@dataclass
class Books:
    freq: str
    grid: list
    book: np.ndarray        # (n_cfg, n_days) sum over live symbols
    live: np.ndarray        # (n_days,) live symbol count
    ens: np.ndarray         # (n_days,) ensemble sum over live symbols
    held: np.ndarray        # (n_cfg,) bars in position
    bars: int               # eval bars total
    n_trades: np.ndarray
    n_choch: np.ndarray
    act: np.ndarray         # (n_cfg, n_days) live symbols with a nonzero filtered P&L that day

    def ew(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.live > 0, self.book / np.maximum(self.live, 1), np.nan)

    def ens_ew(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.live > 0, self.ens / np.maximum(self.live, 1), np.nan)


def symbol_perms(d: dict, k_idx: int, tr: Trades, det_ids, shift: int = 0) -> dict[int, np.ndarray]:
    bpd = d["bpd"]
    out = {}
    for i in det_ids:
        since = vb.bars_since(d["det"][i])
        if shift:
            since = np.roll(since, shift)
        out[i] = since[tr.a] <= ACTIVE_DAYS * bpd
    return out


def build_books(datas: list[dict], freq: str, end_day: int, grid: list | None = None, verbose: bool = True) -> Books:
    bpd = FREQS[freq]
    grid = grid or strat_grid(bpd)
    c = cost_side()
    nd = end_day + 1
    book = np.zeros((len(grid), nd))
    live = np.zeros(nd)
    ens = np.zeros(nd)
    held = np.zeros(len(grid))
    ntr = np.zeros(len(grid))
    nch = np.zeros(len(grid))
    act = np.zeros((len(grid), nd), dtype=np.float32)
    bars = 0
    det_ids = sorted({g[0] for g in grid})
    for d in datas:
        t0 = time.time()
        sd = d["start_day"]
        nds = nd - sd
        if nds <= 0:
            continue
        live[sd:] += 1
        ens[sd:] += ens_daily(d, end_day)
        bars += int(((d["acc_day"] >= sd) & (d["acc_day"] <= end_day)).sum())
        trs = {(k, e): build_trades(d, k, bool(e), c) for k in range(2) for e in (0, 1)}
        perms = {k: symbol_perms(d, k, trs[(k, 0)], det_ids) for k in range(2)}
        for gi, (i, k, s, e) in enumerate(grid):
            tr = trs[(k, e)]
            p = perms[k][i]
            if s:
                p = p & (d["dom"][tr.a] == tr.d)
            taken = take_trades(tr, p)
            dly = trades_daily(tr, taken, sd, nds)
            book[gi, sd:] += dly
            act[gi, sd:] += dly != 0
            held[gi] += float(tr.held[taken].sum()) if taken else 0.0
            ntr[gi] += len(taken)
            nch[gi] += float(tr.choch[taken].sum()) if taken else 0.0
        if verbose:
            print(f"  books {freq} {d['sym']} {time.time() - t0:.1f}s", flush=True)
    return Books(freq, grid, book, live, ens, held, bars, ntr, nch, act)


def placebo_books(datas: list[dict], freq: str, cfg: tuple, end_day: int, n: int = N_PLACEBO, seed: int = 0) -> np.ndarray:
    """EW daily books of the cfg with its vol-BoS permission circularly shifted per symbol (timing placebo)."""
    rng = np.random.default_rng(seed)
    i, k, s, e = cfg
    c = cost_side()
    nd = end_day + 1
    out = np.zeros((n, nd))
    live = np.zeros(nd)
    for d in datas:
        sd = d["start_day"]
        if nd - sd <= 0:
            continue
        live[sd:] += 1
        tr = build_trades(d, k, bool(e), c)
        nb = len(d["ts"])
        span = nb - d["lb"]
        for r in range(n):
            sh = int(rng.integers(30 * d["bpd"], max(31 * d["bpd"], span - 30 * d["bpd"])))
            p = symbol_perms(d, k, tr, [i], shift=sh)[i]
            if s:
                p = p & (d["dom"][tr.a] == tr.d)
            out[r, sd:] += trades_daily(tr, take_trades(tr, p), sd, nd - sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(live > 0, out / np.maximum(live, 1), np.nan)


# ---------------------------------------------------------------------- VBOS3 selection
def book_fold_sharpes(ew: np.ndarray, folds: list[tuple[int, int]]) -> np.ndarray:
    sc = np.full((len(folds), ew.shape[0]), np.nan)
    for fi, (a, b) in enumerate(folds):
        x = ew[:, a:b]
        okc = np.isfinite(x[0])
        x = x[:, okc]
        if x.shape[1] < 20:
            continue
        sd = x.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sc[fi] = np.where(sd > 0, x.mean(axis=1) / sd * math.sqrt(365), np.nan)
    return sc


def combined_ew(books: dict[str, Books]) -> tuple[np.ndarray, list[tuple[str, tuple]], dict[str, np.ndarray]]:
    ews, labels, ensd = [], [], {}
    for fr, bk in books.items():
        ews.append(bk.ew())
        labels += [(fr, g) for g in bk.grid]
        ensd[fr] = bk.ens_ew()
    return np.vstack(ews), labels, ensd


def strat_walk_forward(ew: np.ndarray, labels: list, ensd: dict, first: int) -> dict:
    cut = cutoff_day()
    plan = outer_plan(first, EMBARGO)
    picks, sel, base, days = [], [], [], []
    for p in plan:
        a, b = p["block"]
        sc = book_fold_sharpes(ew, p["folds"])
        j, s = select_median(sc)
        picks.append({"block": [ddate(a), ddate(b)], "cfg": j, "label": cfg_label(*labels[j]) if j >= 0 else None, "inner_score": s,
                      "n_folds": len(p["folds"])})
        if j < 0:
            continue
        b2 = min(b, cut)
        x, y = ew[j, a:b2], ensd[labels[j][0]][a:b2]
        m = np.isfinite(x) & np.isfinite(y)
        sel.append(x[m])
        base.append(y[m])
        days.append(np.arange(a, b2)[m])
    x = np.concatenate(sel) if sel else np.zeros(0)
    y = np.concatenate(base) if base else np.zeros(0)
    dm, dt, dn = nw_t(x - y)
    folds = inner_folds(first, cut - EMBARGO)
    j, s = select_median(book_fold_sharpes(ew, folds))
    return {"picks": picks, "wf_sharpe": sharpe_ann(x), "wf_ens_sharpe": sharpe_ann(y), "wf_days": int(len(x)),
            "wf_diff_mean": dm, "wf_diff_t": dt, "wf_mean_t": nw_t(x)[1], "wf_series": x, "wf_ens_series": y,
            "wf_days_idx": np.concatenate(days) if days else np.zeros(0, int),
            "final": {"cfg": j, "label": cfg_label(*labels[j]) if j >= 0 else None, "inner_score": s, "n_folds": len(folds)}}


# ====================================================================== 3. stages
def load_universe(cache: Path, syms, freq: str) -> list[dict]:
    out = []
    for s in syms:
        d = load_symbol(cache, s, freq)
        if d is not None:
            out.append(d)
    return out


def end_day_of(as_of: datetime) -> int:
    return dix(as_of.date()) - 1


def _family_cfgs(bpd: int) -> dict[str, list[int]]:
    g = vb.detector_grid(bpd)
    return {fam: [i for i, d in enumerate(g) if d.family == fam] for fam in vb.FAMILIES}


def first_day(datas: list[dict]) -> int:
    return min(d["start_day"] for d in datas)


def run_events(datas_by_freq: dict[str, list[dict]], end_day: int, min_ev: int = MIN_EV_FOLD) -> dict:
    """VBOS1 + VBOS2 walk-forward and final selections inside llm_in_sample."""
    out = {"vbos1": {}, "vbos2": {}}
    for fr, datas in datas_by_freq.items():
        bpd = FREQS[fr]
        first = first_day(datas)
        ev1 = vbos1_events(datas, end_day)
        base = Baseline(datas, end_day)
        for fam, cfgs in _family_cfgs(bpd).items():
            r = ev_walk_forward(ev1, base, cfgs, first, min_ev)
            r["detector_names"] = {c: vb.detector_grid(bpd)[c].name for c in cfgs}
            fc = r["final"]["cfg"]
            r["final"]["name"] = vb.detector_grid(bpd)[fc].name if fc >= 0 else None
            if fc >= 0:
                r["final"]["in_sample_full"] = ev_segment(ev1, base, fc, first, cutoff_day(), cutoff_day())
            out["vbos1"][f"{fam}|{fr}"] = r
        ev2 = vbos2_events(datas, end_day)
        r = ev_walk_forward(ev2, None, list(range(len(vb.semi_grid()))), first, min_ev)
        fc = r["final"]["cfg"]
        r["final"]["params"] = list(vb.semi_grid()[fc]) if fc >= 0 else None
        if fc >= 0:
            r["final"]["in_sample_full"] = ev_segment(ev2, None, fc, first, cutoff_day(), cutoff_day())
            m = (ev2.cfg == fc) & (ev2.d0 < cutoff_day()) & (ev2.d0 >= first)
            r["final"]["momentum_agreement"] = float(np.mean(ev2.extra["mom"][m] == ev2.side[m])) if m.any() else float("nan")
            r["final"]["by_side"] = {str(sd): clustered_t(ev2.out[H_PRIMARY][m & (ev2.side == sd)], ev2.d0[m & (ev2.side == sd)]) for sd in (1, -1)}
        out["vbos2"][fr] = r
        out[f"_ev1_{fr}"], out[f"_ev2_{fr}"], out[f"_base_{fr}"] = ev1, ev2, base
    return out


def surfaces(ev: Events, base: Baseline | None, cfgs: list[int], windows: dict[str, tuple[int, int]]) -> dict[str, list[float]]:
    out = {}
    for name, (a, b) in windows.items():
        vals = ev_values(ev, H_PRIMARY, base, a, b)
        m = (ev.d0 >= a) & (ev.d0 < b) & (ev.d0 + H_PRIMARY <= b)
        out[name] = ev_stat_by_cfg(ev, vals, m, cfgs, 5).tolist()
    return out


def top_quintile_hit(surf_in: list[float], surf_out: list[float]) -> dict:
    a, b = np.array(surf_in, float), np.array(surf_out, float)
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return {"hit": None}
    thr = np.nanquantile(a, 0.8)
    region = np.flatnonzero(np.isfinite(a) & (a >= thr))
    j = int(np.nanargmax(np.where(np.isfinite(b), b, -np.inf)))
    ra, rb = np.argsort(np.argsort(np.nan_to_num(a, nan=-9))), np.argsort(np.argsort(np.nan_to_num(b, nan=-9)))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return {"hit": bool(j in set(region.tolist())), "out_argmax": j, "in_argmax": int(np.nanargmax(a)), "region": region.tolist(),
            "threshold": float(thr), "spearman": rho}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def stage_insample(cache: Path, as_of: datetime, syms=UNIV_L, tag: str = "L", min_ev: int = MIN_EV_FOLD, strategy: bool = True,
                   freqs: tuple[str, ...] = tuple(FREQS)) -> dict:
    """Everything inside llm_in_sample. No clean_holdout number is computed here."""
    end = end_day_of(as_of)
    cut = cutoff_day()
    dbf = {fr: load_universe(cache, syms, fr) for fr in freqs}
    res = {"tag": tag, "symbols": {fr: [d["sym"] for d in ds] for fr, ds in dbf.items()},
           "start_days": {fr: {d["sym"]: ddate(d["start_day"]) for d in ds} for fr, ds in dbf.items()}}
    ev = run_events(dbf, end, min_ev)
    res["vbos1"], res["vbos2"] = ev["vbos1"], ev["vbos2"]
    # Holm over the 8 family x freq walk-forward tests (one-sided)
    p1 = {k: p_one(v["h10"]["t"]) for k, v in res["vbos1"].items()}
    res["vbos1_holm"] = {"p": p1, "p_holm": holm(p1)}
    p2 = {k: p_one(v["h10"]["t"]) for k, v in res["vbos2"].items()}
    res["vbos2_holm"] = {"p": p2, "p_holm": holm(p2)}
    # VBOS4 split-half surfaces (in-sample analogue)
    split = dix(date(2024, 1, 1))
    res["vbos4_split"] = {}
    for fr, ds in dbf.items():
        bpd = FREQS[fr]
        first = first_day(ds)
        win = {"first_half": (first, split - H_MAX - EMBARGO), "second_half": (split, cut)}
        rcfg = _family_cfgs(bpd)["range"]
        s1 = surfaces(ev[f"_ev1_{fr}"], ev[f"_base_{fr}"], rcfg, win)
        s2 = surfaces(ev[f"_ev2_{fr}"], None, list(range(len(vb.semi_grid()))), win)
        res["vbos4_split"][f"range|{fr}"] = {**s1, "check": top_quintile_hit(s1["first_half"], s1["second_half"])}
        res["vbos4_split"][f"semi|{fr}"] = {**s2, "check": top_quintile_hit(s2["first_half"], s2["second_half"])}
        # full in-sample surfaces (frozen for the holdout comparison)
        res["vbos4_split"][f"range|{fr}"]["in_sample"] = surfaces(ev[f"_ev1_{fr}"], ev[f"_base_{fr}"], rcfg, {"x": (first, cut)})["x"]
        res["vbos4_split"][f"semi|{fr}"]["in_sample"] = surfaces(ev[f"_ev2_{fr}"], None, list(range(len(vb.semi_grid()))), {"x": (first, cut)})["x"]
    if strategy:
        books = {}
        for fr, ds in dbf.items():
            print(f"building books {tag} {fr}", flush=True)
            books[fr] = build_books(ds, fr, cut - 1, verbose=False)
        ew, labels, ensd = combined_ew(books)
        first = min(first_day(ds) for ds in dbf.values())
        wf = strat_walk_forward(ew, labels, ensd, first)
        # common in-sample window for DSR / PBO: days where both freqs have live symbols
        w0 = max(int(np.flatnonzero(books[fr].live > 0)[0]) for fr in books)
        W = np.arange(w0, cut)
        X = ew[:, W]
        X = np.where(np.isfinite(X), X, 0.0)
        j = wf["final"]["cfg"]
        sr_d = X.mean(axis=1) / np.where(X.std(axis=1, ddof=1) > 0, X.std(axis=1, ddof=1), np.nan)
        dsr = deflated_sharpe(X[j], sr_d[np.isfinite(sr_d)], len(labels)) if j >= 0 else {}
        dsr_wf = deflated_sharpe(wf["wf_series"], sr_d[np.isfinite(sr_d)], len(labels)) if len(wf["wf_series"]) > 30 else {}
        pbo = pbo_cscv_chunked(X.T, 16)
        fr_j = labels[j][0]
        bk = books[fr_j]
        gi = bk.grid.index(labels[j][1])
        ens_is = ensd[fr_j][W]
        res["vbos3"] = {
            "final_cfg": [labels[j][0], list(labels[j][1])], "n_configs": len(labels), "window": [ddate(W[0]), ddate(W[-1])], "walk_forward": {k: v for k, v in wf.items() if not isinstance(v, np.ndarray)},
            "final_in_sample": {"sharpe": sharpe_ann(X[j]), "ens_sharpe_same_freq": sharpe_ann(ens_is),
                                **{f"ens_sharpe_{f}": sharpe_ann(ensd[f][W]) for f in ensd},
                                "diff_vs_ens": dict(zip(("mean", "t", "n"), nw_t(X[j] - np.nan_to_num(ens_is)))),
                                "exposure_bars": float(bk.held[gi] / max(bk.bars, 1)), "trades": int(bk.n_trades[gi]),
                                "exposure_days": float(bk.act[gi, W].sum() / max(bk.live[W].sum(), 1)),
                                "choch_share": float(bk.n_choch[gi] / max(bk.n_trades[gi], 1))},
            "dsr_final": dsr, "dsr_walk_forward": dsr_wf, "pbo": pbo,
            "grid_sharpe_quantiles": np.nanquantile(sr_d * math.sqrt(365), [0, 0.1, 0.5, 0.9, 1]).tolist(),
            "share_configs_beating_ens": {f: float(np.mean(sr_d * math.sqrt(365) > sharpe_ann(ensd[f][W]))) for f in ensd},
            "ablation": {},
        }
        i, k, s, e = labels[j][1]
        for name, alt in {"semi_flipped": (i, k, 1 - s, e), "exit_flipped": (i, k, s, 1 - e), "k_flipped": (i, 1 - k, s, e)}.items():
            ga = bk.grid.index(alt)
            res["vbos3"]["ablation"][name] = {"label": cfg_label(fr_j, alt), "sharpe": sharpe_ann(np.nan_to_num(bk.ew()[ga, W]))}
        res["_wf"] = wf
        res["_books"] = books
    return res


# ====================================================================== verdicts
def verdict_vbos1_is(r: dict) -> tuple[str, str]:
    ph = r["vbos1_holm"]["p_holm"]
    best = min(ph, key=lambda k: ph[k])
    t = r["vbos1"][best]["h10"]["t"]
    means = [v["h10"]["mean"] for v in r["vbos1"].values() if v["h10"]["mean"] is not None and np.isfinite(v["h10"]["mean"])]
    if ph[best] < 0.05 and np.isfinite(t) and t > 2:
        return "supports", best
    if not means or max(means) <= 0:
        return "contradicts", best
    return "inconclusive", best


def verdict_vbos2_is(r: dict) -> tuple[str, str]:
    ph = r["vbos2_holm"]["p_holm"]
    best = min(ph, key=lambda k: ph[k])
    t = r["vbos2"][best]["h10"]["t"]
    means = [v["h10"]["mean"] for v in r["vbos2"].values() if np.isfinite(v["h10"]["mean"])]
    if ph[best] < 0.05 and np.isfinite(t) and t > 2:
        return "supports", best
    if not means or max(means) <= 0:
        return "contradicts", best
    return "inconclusive", best


def verdict_vbos3_is(r: dict) -> str:
    v = r["vbos3"]
    dsr = v["dsr_final"].get("dsr", float("nan"))
    if dsr > 0.95 and v["pbo"]["pbo"] < 0.5:
        return "supports"
    if v["final_in_sample"]["sharpe"] <= 0 or v["pbo"]["pbo"] >= 0.5:
        return "contradicts"
    return "inconclusive"


def verdict_vbos4(hits: list) -> str:
    h = [x for x in hits if x is not None]
    k = sum(bool(x) for x in h)
    if len(h) < 4:
        return "inconclusive"
    return "supports" if k >= 3 else ("contradicts" if k <= 1 else "inconclusive")


# ====================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("cache", "insample", "holdout", "pilot"), required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--as-of", default=None, help="ISO timestamp; default now (cache stage), then frozen in the cache")
    a = ap.parse_args()
    cache = Path(a.cache) if a.cache else default_cache()
    cache.mkdir(parents=True, exist_ok=True)
    asof_path = cache / "as_of.txt"
    if a.as_of:
        as_of = datetime.fromisoformat(a.as_of)
    elif asof_path.exists():
        as_of = datetime.fromisoformat(asof_path.read_text().strip())
    else:
        as_of = now()
    asof_path.write_text(as_of.isoformat())
    from ai_trader.research import vol_bos_report as rep

    if a.stage == "cache":
        build_cache(cache, as_of)
    elif a.stage == "pilot":
        rep.pilot(cache, as_of)
    elif a.stage == "insample":
        rep.run_insample(cache, as_of, record=a.record)
    else:
        rep.run_holdout(cache, as_of, record=a.record)


if __name__ == "__main__":
    main()
