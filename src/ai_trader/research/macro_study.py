"""MAC block: yield-curve regimes vs BTC (H-MAC1, H-MAC2) plus descriptive macro context for the thesis ledger.

    PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 uv run python -m ai_trader.research.macro_study [--record] [--no-refresh]

Design frozen in the diary (2026-11-15 18:53 UTC, kind decision) before any result:
- FRED daily series from ai_trader.narrative.macro (point in time; known_ts = obs_date + 1 day 00:00 UTC);
- curve regime from ai_trader.narrative.curve (5-observation changes, 5 bp band, dominant leg);
- BTC = Deribit BTC-PERPETUAL 1h until the first Bybit bar (2020-03-25), then Bybit BTCUSDT 1h; UTC-day closes.
  The registration says 2017-2026: the lake has no BTC before 2020 and the user leads data decisions, so 2017-2019 is
  missing and the test runs 2020-01 .. 2026-11;
- entry = the BTC UTC-day close of obs_date (known at known_ts); forward log return over 5 / 20 calendar days;
  vol-scaled by sd(30 daily log returns ending at entry) x sqrt(h);
- H-MAC1: week-clustered bootstrap (null imposed), one-sided, Holm over the 4 cells; H-MAC2: difference in means
  (HY OAS 5-obs change < 0 minus > 0 among bull-steepening days), week-clustered bootstrap CI.
HY / IG / BBB OAS on FRED (and ALFRED) start 2023-11-14 (ICE licence): H-MAC2 can only use 2023-11 .. 2026-11.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.narrative import macro
from ai_trader.narrative.curve import regimes
from ai_trader.research.evaluate import knowledge_cutoff

SEED = 7
NBOOT = 5000
HORIZONS = (5, 20)
CELLS = (("bear_flattening", -1), ("bull_steepening", +1))
OUT = "data/research/macro"


# ====================================================================== data

def btc_daily(as_of: datetime) -> tuple[pl.DataFrame, dict]:
    """d (UTC date), close (last 1h close of the day), n (hours); only complete-to-as_of bars."""
    from ai_trader.research.analog_study import btc_bars
    from ai_trader.storage.lake import Lake

    b, meta = btc_bars(Lake(), as_of)
    d = (b.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
         .agg(pl.col("close").last(), pl.len().alias("n"), pl.col("src").last()).sort("d"))
    # a day's close is usable only when the day is over (its last bar closes at d+1 00:00)
    last_full = (as_of - timedelta(days=1)).date() if as_of.hour < 24 else as_of.date()
    d = d.filter(pl.col("d") <= last_full)
    meta = {k: str(v) for k, v in meta.items()} | {"days": d.height, "first_day": str(d["d"].min()), "last_day": str(d["d"].max())}
    return d, meta


def lake_daily(sym: str, as_of: datetime) -> pl.DataFrame:
    from ai_trader.research.analog_study import symbol_bars
    from ai_trader.storage.lake import Lake

    b = symbol_bars(Lake(), sym, as_of)
    return (b.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d").agg(pl.col("close").last(), pl.len().alias("n"))
            .sort("d").filter(pl.col("d") < as_of.date()))


def btc_outcomes(btc: pl.DataFrame) -> pl.DataFrame:
    """Per calendar day d: close, sigma30 (sd of the 30 daily log returns ending at d), fwd_h log returns, rv_next5 / rv_prev5."""
    full = pl.DataFrame({"d": pl.date_range(btc["d"].min(), btc["d"].max(), "1d", eager=True)}).join(btc.select("d", "close"), on="d", how="left")
    c = full["close"].to_numpy().astype(float)
    r = np.full(len(c), np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    s = pl.Series(r)
    sig = s.rolling_std(30, min_samples=25).to_numpy()
    cols: dict[str, Any] = {"d": full["d"], "close": c, "ret1": r, "sigma30": sig}
    for h in (5, 20):
        f = np.full(len(c), np.nan)
        f[:-h] = np.log(c[h:] / c[:-h])
        cols[f"fwd{h}"] = f
        cols[f"z{h}"] = f / (sig * math.sqrt(h))
    sq = pl.Series(r**2)
    prev5 = np.sqrt(sq.rolling_mean(5, min_samples=5).to_numpy() * 365)
    nxt = np.full(len(c), np.nan)
    nxt[:-5] = prev5[5:]
    cols["rv_prev5"] = prev5
    cols["rv_next5"] = nxt
    out = pl.DataFrame(cols)
    return out.with_columns([pl.col(c).fill_nan(None) for c in out.columns if out[c].dtype == pl.Float64])


def frame(as_of: datetime, lag_days: int = 0) -> tuple[pl.DataFrame, dict]:
    """Curve regimes joined to BTC outcomes at entry = UTC close of obs_date (+ lag_days for the robustness)."""
    reg = regimes(macro.read_asof("DGS2", as_of), macro.read_asof("DGS10", as_of)).filter(pl.col("known_ts") <= as_of)
    hy = macro.read_asof("BAMLH0A0HYM2", as_of).select("obs_date", pl.col("value").alias("hy"))
    reg = reg.join_asof(hy.sort("obs_date"), on="obs_date", strategy="backward")
    prev_dates = reg["obs_date"].shift(5)
    hy_prev = reg.select(pl.col("obs_date").alias("pd")).with_columns(prev_dates.alias("obs_date")).drop_nulls().join_asof(
        hy.sort("obs_date"), on="obs_date", strategy="backward")
    reg = reg.with_columns(pl.Series("hy_prev", [None] * 5 + hy_prev["hy"].to_list(), dtype=pl.Float64))
    reg = reg.with_columns(((pl.col("hy") - pl.col("hy_prev")) * 100).alias("dhy_bp"))
    btc, meta = btc_daily(as_of)
    out = btc_outcomes(btc)
    reg = reg.with_columns((pl.col("obs_date") + pl.duration(days=lag_days)).alias("d"))
    j = reg.join(out, on="d", how="inner").filter(pl.col("sigma30").is_finite())
    cut = knowledge_cutoff().date()
    j = j.with_columns(pl.when(pl.col("d") < cut).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"),
                       (pl.col("d").dt.iso_year() * 100 + pl.col("d").dt.week()).alias("week"))
    return j, meta


# ====================================================================== statistics

def _week_stats(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uw, inv = np.unique(w, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(uw))
    cnts = np.bincount(inv, minlength=len(uw)).astype(float)
    return sums, cnts


def boot_means(x: np.ndarray, w: np.ndarray, rng: np.random.Generator, nboot: int = NBOOT) -> np.ndarray:
    sums, cnts = _week_stats(x, w)
    k = len(sums)
    idx = rng.integers(0, k, size=(nboot, k))
    return sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)


def one_sided_p(x: np.ndarray, w: np.ndarray, sign: int, rng: np.random.Generator) -> float:
    """Week-clustered bootstrap with the null (mean 0) imposed: P(mean* as extreme as observed in the hypothesised direction)."""
    if len(x) < 5:
        return float("nan")
    m = float(x.mean())
    bm = boot_means(x - m, w, rng)
    return float(((bm >= m) if sign > 0 else (bm <= m)).mean() + 1 / (len(bm) + 1))


def holm(ps: list[float]) -> list[float]:
    idx = [i for i, p in enumerate(ps) if p == p]
    order = sorted(idx, key=lambda i: ps[i])
    m = len(order)
    adj = [float("nan")] * len(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def nw_t(x: np.ndarray, lags: int) -> float:
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    v = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else float("nan")


def _cell(df: pl.DataFrame, regime: str, h: int, sign: int, rng: np.random.Generator) -> dict[str, Any]:
    allx = df.filter(pl.col(f"z{h}").is_finite())
    sub = allx.filter(pl.col("regime") == regime)
    x = sub[f"z{h}"].to_numpy()
    w = sub["week"].to_numpy()
    res: dict[str, Any] = {"regime": regime, "h": h, "hyp_sign": sign, "n_days": len(x), "n_weeks": int(len(np.unique(w))) if len(x) else 0}
    if len(x) < 5:
        return res | {"mean": float(x.mean()) if len(x) else float("nan"), "p_one_sided": float("nan")}
    bm = boot_means(x, w, rng)
    unc = allx[f"z{h}"].to_numpy()
    # difference vs unconditional, bootstrapped jointly by week over all days
    ww = allx["week"].to_numpy()
    flag = (allx["regime"] == regime).to_numpy()
    uw, inv = np.unique(ww, return_inverse=True)
    s_all = np.bincount(inv, weights=unc, minlength=len(uw))
    c_all = np.bincount(inv, minlength=len(uw)).astype(float)
    s_in = np.bincount(inv, weights=np.where(flag, unc, 0.0), minlength=len(uw))
    c_in = np.bincount(inv, weights=flag.astype(float), minlength=len(uw))
    idx = rng.integers(0, len(uw), size=(NBOOT, len(uw)))
    dd = s_in[idx].sum(1) / np.maximum(c_in[idx].sum(1), 1) - s_all[idx].sum(1) / c_all[idx].sum(1)
    raw = sub[f"fwd{h}"].to_numpy()
    return res | {
        "mean": float(x.mean()), "median": float(np.median(x)), "ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))],
        "p_one_sided": one_sided_p(x, w, sign, rng), "nw_t": nw_t(x, h),
        "uncond_mean": float(unc.mean()), "diff_vs_uncond": float(x.mean() - unc.mean()),
        "diff_ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))],
        "raw_mean_pct": float(100 * raw.mean()), "hit_up": float((raw > 0).mean()),
    }


def mac1(df: pl.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {}
    for seg in ("llm_in_sample", "clean_holdout", "full"):
        d = df if seg == "full" else df.filter(pl.col("segment") == seg)
        cells = [_cell(d, r, h, s, rng) for r, s in CELLS for h in HORIZONS]
        adj = holm([c["p_one_sided"] for c in cells])
        for c, a in zip(cells, adj, strict=True):
            c["p_holm"] = a
            c["right_sign"] = bool(c["mean"] == c["mean"] and np.sign(c["mean"]) == c["hyp_sign"])
            # significance in the wrong direction, also Holm-adjusted across the 4 cells (computed below)
        wrong_p = holm([one_sided_p(d.filter((pl.col("regime") == c["regime"]) & pl.col(f"z{c['h']}").is_finite())[f"z{c['h']}"].to_numpy(),
                                    d.filter((pl.col("regime") == c["regime"]) & pl.col(f"z{c['h']}").is_finite())["week"].to_numpy(),
                                    -c["hyp_sign"], rng) for c in cells])
        for c, a in zip(cells, wrong_p, strict=True):
            c["p_holm_wrong_dir"] = a
        dist = d.group_by("regime").agg(pl.len().alias("n")).sort("regime").to_dicts()
        out[seg] = {"cells": cells, "regime_counts": dist, "n_days": d.height,
                    "first": str(d["d"].min()) if d.height else None, "last": str(d["d"].max()) if d.height else None}
    ins = out["llm_in_sample"]["cells"]
    ho = out["clean_holdout"]["cells"]

    def contra(cells: list[dict]) -> bool:
        return any(c.get("p_holm_wrong_dir", 1) < 0.05 for c in cells) or sum(not c["right_sign"] for c in cells) >= 3

    v_in = "supports" if all(c["right_sign"] and c["p_holm"] < 0.05 for c in ins) else "contradicts" if contra(ins) else "inconclusive"
    if contra(ho):
        v_ho = "contradicts"
    elif v_in == "supports" and all(c["right_sign"] for c in ho):
        v_ho = "supports"
    else:
        v_ho = "inconclusive"
    v_pool = "supports" if v_in == v_ho == "supports" else "contradicts" if v_in == "contradicts" else "inconclusive"
    out["verdicts"] = {"llm_in_sample": v_in, "clean_holdout": v_ho, "pooled": v_pool}
    return out


def mac2(df: pl.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 1)
    bs = df.filter((pl.col("regime") == "bull_steepening") & pl.col("z20").is_finite() & pl.col("dhy_bp").is_not_null())
    out: dict[str, Any] = {"hy_first_obs": str(df.filter(pl.col("hy").is_not_null())["obs_date"].min())}
    for seg in ("full", "llm_in_sample", "clean_holdout"):
        d = bs if seg == "full" else bs.filter(pl.col("segment") == seg)
        a = d.filter(pl.col("dhy_bp") < 0)
        b = d.filter(pl.col("dhy_bp") > 0)
        res: dict[str, Any] = {"n_eased": a.height, "n_widened": b.height,
                               "weeks_eased": a["week"].n_unique(), "weeks_widened": b["week"].n_unique()}
        if a.height and b.height:
            xa, xb = a["z20"].to_numpy(), b["z20"].to_numpy()
            res |= {"mean_eased": float(xa.mean()), "mean_widened": float(xb.mean()), "diff": float(xa.mean() - xb.mean()),
                    "raw_mean_eased_pct": float(100 * a["fwd20"].mean()), "raw_mean_widened_pct": float(100 * b["fwd20"].mean())}
            if a.height >= 3 and b.height >= 3:
                da = boot_means(xa, a["week"].to_numpy(), rng)
                db = boot_means(xb, b["week"].to_numpy(), rng)
                diff = da - db
                res |= {"diff_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))], "p_diff_le_0": float((diff <= 0).mean())}
        out[seg] = res
    f = out["full"]
    segs = [out["llm_in_sample"].get("diff"), out["clean_holdout"].get("diff")]
    if "diff_ci95" not in f:
        v = "inconclusive"
    elif f["diff"] > 0 and f["diff_ci95"][0] > 0 and all(s is not None and s > 0 for s in segs):
        v = "supports"
    elif (f["diff"] < 0 and f["diff_ci95"][1] < 0) or all(s is not None and s < 0 for s in segs):
        v = "contradicts"
    else:
        v = "inconclusive"
    out["verdict"] = v

    def seg_v(s: dict) -> str:
        if "diff" not in s:
            return "inconclusive"
        if "diff_ci95" in s and s["diff_ci95"][0] > 0:
            return "supports"
        if "diff_ci95" in s and s["diff_ci95"][1] < 0:
            return "contradicts"
        return "inconclusive"

    out["segment_verdicts"] = {"llm_in_sample": seg_v(out["llm_in_sample"]), "clean_holdout": seg_v(out["clean_holdout"])}
    return out


# ====================================================================== descriptives

def vol_by_regime(df: pl.DataFrame) -> list[dict]:
    return (df.group_by("regime").agg(pl.len().alias("n"), (100 * pl.col("rv_prev5").median()).alias("rv_same5_med_pct"),
                                      (100 * pl.col("rv_next5").median()).alias("rv_next5_med_pct"),
                                      (100 * pl.col("rv_next5").mean()).alias("rv_next5_mean_pct"),
                                      (100 * pl.col("sigma30").median() * math.sqrt(365)).alias("sigma30_ann_med_pct"))
            .sort("regime").to_dicts())


def today(as_of: datetime) -> dict[str, Any]:
    reg = regimes(macro.read_asof("DGS2", as_of), macro.read_asof("DGS10", as_of)).filter(pl.col("known_ts") <= as_of)
    last20 = reg.tail(20)
    j = reg.with_columns(((pl.col("y2") - pl.col("y2").shift(20)) * 100).alias("d2_20"), ((pl.col("y10") - pl.col("y10").shift(20)) * 100).alias("d10_20"))
    lr = j.tail(1).to_dicts()[0]
    counts = last20.group_by("regime").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts()
    return {"last_obs": str(lr["obs_date"]), "known_ts": str(lr["known_ts"]), "regime": lr["regime"], "y2": lr["y2"], "y10": lr["y10"],
            "spread_bp": lr["spread_bp"], "d2_bp": lr["d2_bp"], "d10_bp": lr["d10_bp"], "d2_20obs_bp": lr["d2_20"], "d10_20obs_bp": lr["d10_20"],
            "regime_20d_counts": counts,
            "last20": [{k: (str(v) if isinstance(v, (date, datetime)) else v) for k, v in r.items()} for r in last20.to_dicts()]}


def _pct_1y(s: pl.DataFrame) -> tuple[float, float, str]:
    """(today's value, percentile in the last 252 obs / 12 monthly obs incl. today, obs date)."""
    v = s["value"].to_numpy()
    win = v[-252:] if len(v) > 300 else v[-12:]
    return float(v[-1]), float((win <= v[-1]).mean()), str(s["obs_date"][-1])


def proxies_today(as_of: datetime) -> list[dict]:
    rows = []
    for sid in ("BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLH0A0HYM2", "THREEFYTP10", "DFII10", "DGS2", "DGS10", "VIXCLS", "DTWEXBGS"):
        s = macro.read_asof(sid, as_of)
        v, p, d = _pct_1y(s)
        chg20 = float(v - s["value"][-21]) if s.height > 21 else float("nan")
        med = float(np.median(s["value"].to_numpy()[-252:]))
        rows.append({"series": sid, "obs_date": d, "value": v, "pct_1y": p, "chg_20obs": chg20, "median_1y": med})
    for sid in ("UNRATE", "SAHMREALTIME"):
        s = macro.read_asof(sid, as_of)
        v = s["value"].to_numpy()
        rows.append({"series": sid, "obs_date": str(s["obs_date"][-1]), "value": float(v[-1]), "pct_1y": float((v[-12:] <= v[-1]).mean()),
                     "chg_3m": float(v[-1] - v[-4]), "median_1y": float(np.median(v[-12:]))})
    for sid in ("CPILFESL", "PCEPILFE"):
        s = macro.read_asof(sid, as_of)
        yoy = yoy_pct(s)
        y = yoy["value"].to_numpy()
        rows.append({"series": f"{sid}_yoy", "obs_date": str(yoy["obs_date"][-1]), "value": float(y[-1]), "pct_1y": float((y[-12:] <= y[-1]).mean()),
                     "last4_yoy": [round(float(a), 2) for a in y[-4:]], "falling_3m": bool(np.all(np.diff(y[-4:]) < 0)),
                     "ann3m_pct": float(100 * ((s["value"][-1] / s["value"][-4]) ** 4 - 1))})
    return rows


def yoy_pct(s: pl.DataFrame) -> pl.DataFrame:
    """Year-on-year % on monthly obs, matching by calendar month (FRED skipped Oct 2025: a missing month stays missing)."""
    m = s.with_columns(pl.col("obs_date").dt.offset_by("12mo").alias("next"))
    j = s.join(m.select(pl.col("next").alias("obs_date"), pl.col("value").alias("prev")), on="obs_date", how="inner")
    return j.with_columns((100 * (pl.col("value") / pl.col("prev") - 1)).alias("value")).select("obs_date", "value").sort("obs_date")


def sp_drawdowns(as_of: datetime, btc: pl.DataFrame, thresh: float = -0.10) -> dict[str, Any]:
    sp = macro.read_asof("SP500", as_of)
    v = sp["value"].to_numpy()
    d = sp["obs_date"].to_list()
    peak = np.maximum.accumulate(v)
    dd = v / peak - 1
    bc = dict(zip(btc["d"].to_list(), btc["close"].to_list(), strict=True))

    def bclose(x: date) -> float | None:
        for k in range(4):
            c = bc.get(x - timedelta(days=k))
            if c:
                return c
        return None

    eps = []
    i = 0
    n = len(v)
    while i < n:
        if dd[i] <= thresh:
            # walk back to the peak and forward to recovery
            p = i
            while p > 0 and dd[p] < 0:
                p -= 1
            j = i
            while j < n and dd[j] < 0:
                j += 1
            seg = slice(p, j)
            t = p + int(np.argmin(dd[seg]))
            eps.append({"peak": d[p], "trough": d[t], "recovered": d[j] if j < n else None, "depth_pct": float(100 * dd[t]), "peak_level": float(v[p]),
                        "trough_level": float(v[t])})
            i = j + 1
        else:
            i += 1
    for e in eps:
        b0, b1 = bclose(e["peak"]), bclose(e["trough"])
        b2 = bclose(e["trough"] + timedelta(days=20))
        if b0 and b1:
            e["btc_peak_to_trough_pct"] = float(100 * math.log(b1 / b0))
            # BTC max drawdown inside the S&P episode
            sub = btc.filter((pl.col("d") >= e["peak"]) & (pl.col("d") <= e["trough"]))["close"].to_numpy()
            e["btc_maxdd_in_episode_pct"] = float(100 * (np.min(sub / np.maximum.accumulate(sub)) - 1)) if len(sub) else None
        if b1 and b2:
            e["btc_trough_plus20d_pct"] = float(100 * math.log(b2 / b1))
        for k in ("peak", "trough", "recovered"):
            e[k] = str(e[k]) if e[k] else None
    cur = {"date": str(d[-1]), "level": float(v[-1]), "ath": float(peak[-1]), "dd_pct": float(100 * dd[-1]),
           "user_level_7000_logdist_pct": float(100 * math.log(7000 / v[-1])),
           "user_level_7000_dd_from_ath_pct": float(100 * (7000 / peak[-1] - 1))}
    return {"threshold_pct": 100 * thresh, "episodes": [e for e in eps if e["trough"] >= "2020-01-01"], "current": cur}


def divergence(as_of: datetime, btc: pl.DataFrame) -> dict[str, Any]:
    """Tech/semis vs S&P: NDX-SPX 20-obs log relative return since 2020 (long history), and the lake's semis perps
    (SOXLUSDT / 3 as a rough daily SOX proxy, NVDAUSDT, MUUSDT) vs SPX since May 2026. Never SPYUSDT."""
    sp = macro.read_asof("SP500", as_of).select("obs_date", pl.col("value").alias("spx"))
    nd = macro.read_asof("NASDAQ100", as_of).select("obs_date", pl.col("value").alias("ndx"))
    j = sp.join(nd, on="obs_date", how="inner").sort("obs_date")
    j = j.with_columns((100 * ((pl.col("ndx") / pl.col("ndx").shift(20)).log() - (pl.col("spx") / pl.col("spx").shift(20)).log())).alias("rel20"),
                       (pl.col("ndx") / pl.col("spx")).alias("ratio"))
    j = j.with_columns(((pl.col("rel20") - pl.col("rel20").rolling_mean(252)) / pl.col("rel20").rolling_std(252)).alias("rel20_z"))
    bo = btc_outcomes(btc).select(pl.col("d").alias("obs_date"), "fwd20", "z20", "close")
    jj = j.join(bo, on="obs_date", how="left").filter(pl.col("obs_date") >= date(2020, 1, 1))
    jb = jj.with_columns((100 * (pl.col("close") / pl.col("close").shift(20)).log()).alias("btc_same20"))
    groups = {}
    for name, cond in (("ndx_lags_spx_z<-1.5", pl.col("rel20_z") < -1.5), ("ndx_leads_spx_z>1.5", pl.col("rel20_z") > 1.5),
                       ("normal", pl.col("rel20_z").abs() <= 1.5)):
        g = jb.filter(cond & pl.col("rel20_z").is_finite())
        groups[name] = {"n_days": g.height, "btc_same20_mean_pct": g["btc_same20"].mean(), "btc_fwd20_mean_pct": (100 * g["fwd20"]).mean() if g.height else None,
                        "btc_z20_mean": g["z20"].mean()}
    last = jj.tail(1).to_dicts()[0]
    ratio = j["ratio"].to_numpy()
    semis: dict[str, Any] = {}
    for sym in ("SOXLUSDT", "NVDAUSDT", "MUUSDT"):
        try:
            s = lake_daily(sym, as_of).select(pl.col("d").alias("obs_date"), pl.col("close").alias(sym))
        except Exception as e:  # noqa: BLE001 - descriptive only
            semis[sym] = {"error": str(e)}
            continue
        k = s.join(sp, on="obs_date", how="inner").sort("obs_date")
        lev = 3.0 if sym == "SOXLUSDT" else 1.0
        k = k.with_columns((100 * ((pl.col(sym) / pl.col(sym).shift(20)).log() / lev - (pl.col("spx") / pl.col("spx").shift(20)).log())).alias("rel20"))
        kk = k.drop_nulls("rel20")
        semis[sym] = {"first": str(s["obs_date"].min()), "obs": kk.height, "rel20_now_pct": float(kk["rel20"][-1]) if kk.height else None,
                      "rel20_min_pct": float(kk["rel20"].min()) if kk.height else None, "rel20_max_pct": float(kk["rel20"].max()) if kk.height else None,
                      "date_now": str(kk["obs_date"][-1]) if kk.height else None, "deleveraged": lev != 1.0,
                      "series": [{"d": str(a), "rel20": float(b)} for a, b in zip(kk["obs_date"].to_list(), kk["rel20"].to_list(), strict=True)]}
    return {"ndx_spx": {"date": str(last["obs_date"]), "rel20_pct": last["rel20"], "rel20_z": last["rel20_z"],
                        "ratio": last["ratio"], "ratio_pct_since_2016": float((ratio <= ratio[-1]).mean()),
                        "ratio_pct_1y": float((ratio[-252:] <= ratio[-1]).mean()), "groups": groups},
            "semis": semis,
            "series": [{"d": str(a), "rel20": b, "z": c} for a, b, c in zip(jj["obs_date"].to_list(), jj["rel20"].to_list(), jj["rel20_z"].to_list(), strict=True)]}


def spreads_vs_2y(df: pl.DataFrame, as_of: datetime) -> dict[str, Any]:
    """Descriptive (not registered): HY and IG OAS widening over 20 obs, split by the 2y's 20-obs change."""
    ig = macro.read_asof("BAMLC0A0CM", as_of).select("obs_date", pl.col("value").alias("ig"))
    d = df.join(ig, on="obs_date", how="left").sort("obs_date")
    d = d.with_columns(((pl.col("hy") - pl.col("hy").shift(20)) * 100).alias("dhy20"), ((pl.col("ig") - pl.col("ig").shift(20)) * 100).alias("dig20"),
                       ((pl.col("y2") - pl.col("y2").shift(20)) * 100).alias("d2_20"),
                       (100 * (pl.col("close") / pl.col("close").shift(20)).log()).alias("btc_same20"))
    d = d.filter(pl.col("dhy20").is_not_null() & pl.col("dig20").is_not_null())
    widen = d.filter((pl.col("dhy20") > 0) & (pl.col("dig20") > 0))
    out: dict[str, Any] = {"first": str(d["obs_date"].min()), "n_days": d.height, "n_widening": widen.height}
    for name, cond in (("widening_2y_up_or_flat", pl.col("d2_20") >= 0), ("widening_2y_down", pl.col("d2_20") < 0),
                       ("widening_2y_down_25bp", pl.col("d2_20") <= -25)):
        g = widen.filter(cond)
        out[name] = {"n_days": g.height, "n_weeks": g["week"].n_unique(), "btc_same20_mean_pct": g["btc_same20"].mean(),
                     "btc_fwd20_mean_pct": (100 * g["fwd20"]).mean(), "btc_z20_mean": g["z20"].mean(),
                     "btc_fwd20_hit_up": float((g["fwd20"].drop_nulls() > 0).mean()) if g["fwd20"].drop_nulls().len() else None}
    tight = d.filter((pl.col("dhy20") < 0) & (pl.col("dig20") < 0))
    out["tightening_any_2y"] = {"n_days": tight.height, "btc_same20_mean_pct": tight["btc_same20"].mean(), "btc_fwd20_mean_pct": (100 * tight["fwd20"]).mean()}
    return out


# ====================================================================== run / record

def run(as_of: datetime | None = None, refresh: bool = True, log=print) -> dict[str, Any]:
    as_of = as_of or now()
    if refresh:
        macro.refresh(max_age_h=6.0, log=log)
    df, meta = frame(as_of)
    rob, _ = frame(as_of, lag_days=1)
    res: dict[str, Any] = {
        "as_of": as_of.isoformat(), "btc": meta, "cutoff": knowledge_cutoff().date().isoformat(),
        "data_gaps": ["BTC 2017-2019 missing (lake starts 2020-01-01; registration says 2017-2026; user leads data decisions)",
                      "HY / IG / BBB OAS on FRED and ALFRED start 2023-11-14 (ICE licence): H-MAC2 uses 2023-11 .. 2026-11 only",
                      "no FRED semiconductor index; semis proxy = Bybit SOXLUSDT/3, NVDAUSDT, MUUSDT perps from 2026-05 only",
                      "ACM term premium is not on FRED; Kim-Wright THREEFYTP10 used",
                      "monthly series are latest vintage with a release-lag known_ts rule (ALFRED vintages fetched only for the live check)"],
        "manifest": {k: {kk: v[kk] for kk in ("first_obs", "last_obs", "obs", "last_download", "sha256")} for k, v in macro.manifest()["series"].items()},
    }
    res["mac1"] = mac1(df)
    res["mac1_robust_lag1"] = {k: v for k, v in mac1(rob).items() if k in ("verdicts", "full")}
    res["mac2"] = mac2(df)
    res["mac2_robust_lag1"] = {k: v for k, v in mac2(rob).items() if k in ("verdict", "full")}
    res["vol_by_regime"] = {"full": vol_by_regime(df), "llm_in_sample": vol_by_regime(df.filter(pl.col("segment") == "llm_in_sample")),
                            "clean_holdout": vol_by_regime(df.filter(pl.col("segment") == "clean_holdout"))}
    res["today"] = today(as_of)
    res["proxies_today"] = proxies_today(as_of)
    btc, _ = btc_daily(as_of)
    res["sp_drawdowns_10"] = sp_drawdowns(as_of, btc, -0.10)
    res["sp_drawdowns_5"] = sp_drawdowns(as_of, btc, -0.05)
    res["divergence"] = divergence(as_of, btc)
    res["spreads_vs_2y"] = spreads_vs_2y(df, as_of)
    res["history"] = [{"d": str(a), "regime": b, "btc": c, "y2": e, "y10": f} for a, b, c, e, f in
                      zip(df["obs_date"].to_list(), df["regime"].to_list(), df["close"].to_list(), df["y2"].to_list(), df["y10"].to_list(), strict=True)]
    path = load().root / OUT / "macro_study.json"
    path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    log(f"wrote {path}")
    return res


def record_results(res: dict[str, Any]) -> None:
    from ai_trader.research.hypotheses import record

    data = ("FRED DGS2/DGS10/BAMLH0A0HYM2 (data/research/macro, PIT known_ts obs+1d); BTC Deribit->Bybit 1h 2020-01..2026-11 UTC closes; "
            "src/ai_trader/research/macro_study.py; research/reports/macro_narrative_2026-11-15.md")
    m1 = res["mac1"]
    for seg in ("llm_in_sample", "clean_holdout"):
        cells = {f"{c['regime']}_{c['h']}d": {k: c.get(k) for k in ("n_days", "n_weeks", "mean", "ci95", "p_one_sided", "p_holm", "nw_t", "diff_vs_uncond", "raw_mean_pct")}
                 for c in m1[seg]["cells"]}
        record("H-MAC1", seg, m1["verdicts"][seg], {"cells": cells, "trials": "2 regimes x 2 horizons x 2 segments (Holm within segment)",
                                                     "period": f"{m1[seg]['first']}..{m1[seg]['last']}", "missing": "2017-2019 (no BTC in lake)"},
               note="vol-scaled fwd log returns after bear-flattening / bull-steepening days; week-clustered bootstrap, null imposed",
               data=data)
    record("H-MAC1", "pooled", m1["verdicts"]["pooled"], {"llm_in_sample": m1["verdicts"]["llm_in_sample"], "clean_holdout": m1["verdicts"]["clean_holdout"],
                                                          "robust_known_ts_plus1d": res["mac1_robust_lag1"]["verdicts"]}, data=data)
    m2 = res["mac2"]
    for seg in ("llm_in_sample", "clean_holdout"):
        record("H-MAC2", seg, m2["segment_verdicts"][seg], m2[seg] | {"hy_first_obs": m2["hy_first_obs"]},
               note="segment-level read; HY OAS only from 2023-11-14", data=data)
    record("H-MAC2", "pooled", m2["verdict"], m2["full"] | {"segments": {s: m2[s].get("diff") for s in ("llm_in_sample", "clean_holdout")},
                                                            "robust_known_ts_plus1d": res["mac2_robust_lag1"]["verdict"], "hy_first_obs": m2["hy_first_obs"]},
           note="full sample 2023-11..2026-11 (HY OAS licence window)", data=data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--as-of", default="")
    a = ap.parse_args()
    t = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")) if a.as_of else now()
    res = run(t.astimezone(UTC), refresh=not a.no_refresh)
    for seg in ("llm_in_sample", "clean_holdout"):
        for c in res["mac1"][seg]["cells"]:
            print(f"MAC1 {seg:14s} {c['regime']:16s} {c['h']:>2}d n={c['n_days']:4d} wk={c['n_weeks']:3d} mean={c['mean']:+.3f} "
                  f"p={c.get('p_one_sided', float('nan')):.3f} holm={c.get('p_holm', float('nan')):.3f} nw_t={c.get('nw_t', float('nan')):+.2f}")
    print("MAC1 verdicts", res["mac1"]["verdicts"], "robust", res["mac1_robust_lag1"]["verdicts"])
    print("MAC2", json.dumps({k: res["mac2"][k] for k in ("full", "llm_in_sample", "clean_holdout", "verdict", "segment_verdicts")}, default=str))
    print("TODAY", {k: res["today"][k] for k in ("last_obs", "regime", "d2_bp", "d10_bp", "d2_20obs_bp", "d10_20obs_bp", "regime_20d_counts")})
    if a.record:
        record_results(res)
        print("recorded H-MAC1, H-MAC2")


if __name__ == "__main__":
    main()
