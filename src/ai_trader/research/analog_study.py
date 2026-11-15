"""Multi-scale analog search for BTC (agent ANALOG, 2026-11-15) and the H-ANA1 walk-forward test. Research only.

Design frozen in the diary (2026-11-15 17:56 UTC, kind decision, "ANALOG design frozen before results") and copied to
data/research/analogs/design.json before any distance or outcome is computed. Summary:

  data     BTC 1h = Deribit BTC-PERPETUAL before the first Bybit BTCUSDT 1h bar (2020-01-01..2020-03-25), Bybit after.
           The lake holds no Binance BTC 1h (venue=binance is ETHUSDT only), so nothing before 2020. Funding (Bybit,
           2020-03-25 on) and 4h OI (Bybit, 2020-08-04 on) are as-of joined on known_ts; missing features are re-weighted.
  scales   L = 6, 24, 72, 168, 720 h. Shape = 1 - Pearson of the cumulative log path sampled at N points; amplitude =
           |ln A_now - ln A_cand|, A = path range / (sigma_ref sqrt L), sigma_ref = hourly sd over the 720 h before the
           window; state = standardised Euclidean (rv_pct, dd30, cell dir/vol, funding_pct, oi72, weekend), averaged
           over the features both hours have. d_s = 1.0 shape + 0.5 amp + 0.5 state.
  combined mean over scales of the percentile rank of d_s in the pool; decluster 168 h; top 20.
  H-ANA1   daily 08:00 UTC origins; pool = candidates whose 72 h outcome is known 7 days before the origin; vol-scaled
           analog variance vs log-HAR-t x weekday (options_compression.predictive); QLIKE NW t; Brier skill, 10-day blocks.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.analog_study design
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.analog_study search
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.analog_study walkforward [--record]
    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.analog_study_report
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.time import now
from ai_trader.research.calendar import add_calendar
from ai_trader.research.evaluate import knowledge_cutoff
from ai_trader.research.frame import enrich_bars
from ai_trader.research.regimes import label
from ai_trader.storage.lake import Lake

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "analogs"
SEED = 20261115

D: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "splice": {"before_bybit": {"venue": "deribit", "symbol": "BTC-PERPETUAL"}, "after": {"venue": "bybit", "symbol": "BTCUSDT"}},
    "scales_h": [6, 24, 72, 168, 720],
    "points": {"6": 7, "24": 25, "72": 25, "168": 25, "720": 25},
    "max_missing_frac": 0.10,
    "sigma_ref_window_h": 720,
    "sigma_ref_min_obs": 336,
    "w_shape": 1.0,
    "w_amp": 0.5,
    "w_state": 0.5,
    "state_features": {"rv_pct": 1.0, "dd30": 1.0, "cell_dir": 0.5, "cell_vol": 0.5, "funding_pct": 1.0, "oi72": 1.0, "weekend": 1.0},
    "rv_window_h": 24,
    "rv_pct_window_h": 8760,
    "rv_pct_min_h": 1440,
    "dd_window_h": 720,
    "funding_days": 90,
    "funding_min_days": 30,
    "oi_lag_h": 72,
    "exclude_recent_days": 30,
    "decluster_h": 168,
    "top_k": 20,
    "fwd_h": [6, 24, 72, 168],
    "path_h": 168,
    "wf_origin_hour_utc": 8,
    "wf_embargo_h": 168,
    "wf_outcome_h": 72,
    "wf_horizons_days": [1, 3],
    "wf_nw_lags": {"1": 7, "3": 9},
    "wf_scale_var_h": 168,
    "wf_block_days": 10,
    "nboot": 5000,
    "seed": SEED,
    "cutoff": "2026-07-01",
}
SCALES = D["scales_h"]
FEATS = list(D["state_features"])


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def seg_of(ts: datetime) -> str:
    return "llm_in_sample" if ts < knowledge_cutoff() else "clean_holdout"


# ====================================================================== data

def _clean(b: pl.DataFrame) -> pl.DataFrame:
    b = b.select("event_ts", "open", "high", "low", "close")
    b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                 & (pl.col("low") <= pl.min_horizontal("open", "close")))
    return b.with_columns(pl.col("event_ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us"))


def btc_bars(lake: Lake, as_of: datetime) -> tuple[pl.DataFrame, dict]:
    """Spliced BTC 1h bars (event_ts = bar open), only bars closed by as_of."""
    by = _clean(lake.bars("BTCUSDT", "1h", as_of, venue="bybit"))
    first = by["event_ts"].min()
    de = _clean(lake.bars("BTC-PERPETUAL", "1h", as_of, venue="deribit")).filter(pl.col("event_ts") < first)
    b = pl.concat([de.with_columns(pl.lit("deribit").alias("src")), by.with_columns(pl.lit("bybit").alias("src"))]).sort("event_ts")
    b = b.filter(pl.col("event_ts") + timedelta(hours=1) <= as_of).unique("event_ts", keep="last").sort("event_ts")
    meta = {"deribit_bars": de.height, "bybit_bars": by.height, "bybit_first": first, "first": b["event_ts"].min(), "last_open": b["event_ts"].max()}
    return b, meta


def symbol_bars(lake: Lake, sym: str, as_of: datetime) -> pl.DataFrame:
    b = _clean(lake.bars(sym, "1h", as_of, venue="bybit"))
    return b.filter(pl.col("event_ts") + timedelta(hours=1) <= as_of).unique("event_ts", keep="last").sort("event_ts").with_columns(pl.lit("bybit").alias("src"))


def _grid(bars: pl.DataFrame) -> pl.DataFrame:
    grid = pl.DataFrame({"event_ts": pl.datetime_range(bars["event_ts"].min(), bars["event_ts"].max(), "1h", time_zone="UTC", eager=True)}).with_columns(
        pl.col("event_ts").dt.cast_time_unit("us"))
    g = grid.join(bars, on="event_ts", how="left").sort("event_ts")
    return g.with_columns((pl.col("event_ts") + timedelta(hours=1)).alias("close_ts"))


def _rolling_pct(x: np.ndarray, window: int, min_obs: int, chunk: int = 400) -> np.ndarray:
    """Percentile of x[i] among the finite x[i-window+1..i] (inclusive), NaN if fewer than min_obs finite values."""
    n = len(x)
    out = np.full(n, np.nan)
    xp = np.r_[np.full(window - 1, np.nan), x]
    from numpy.lib.stride_tricks import sliding_window_view

    W = sliding_window_view(xp, window)
    for a in range(0, n, chunk):
        w = W[a:a + chunk]
        cur = x[a:a + chunk][:, None]
        fin = np.isfinite(w)
        cnt = fin.sum(1)
        le = (np.where(fin, w, np.inf) <= cur).sum(1)
        r = le / np.maximum(cnt, 1)
        r[(cnt < min_obs) | ~np.isfinite(x[a:a + chunk])] = np.nan
        out[a:a + chunk] = r
    return out


def _asof(grid_ts: np.ndarray, known: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Latest value with known <= grid time (ms arrays, known sorted)."""
    i = np.searchsorted(known, grid_ts, side="right") - 1
    out = np.where(i >= 0, vals[np.clip(i, 0, None)], np.nan)
    return out.astype(float)


def funding_rank(lake: Lake, sym: str, as_of: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = lake.read_asof("funding", as_of, symbols=[sym], filters={"venue": "bybit"}).sort("event_ts")
    if f.height == 0:
        return np.array([]), np.array([]), np.array([])
    ts = f["event_ts"].dt.epoch("ms").to_numpy()
    kn = f["known_ts"].dt.epoch("ms").to_numpy()
    r = f["rate"].to_numpy().astype(float)
    w = D["funding_days"] * 86_400_000
    lo = np.searchsorted(ts, ts - w, side="right")
    first = ts[0]
    rank = np.array([np.mean(r[lo[i]:i + 1] <= r[i]) if ts[i] - first >= D["funding_min_days"] * 86_400_000 else np.nan for i in range(len(r))])
    o = np.argsort(kn, kind="stable")
    return kn[o], rank[o], r[o]


def build_state(bars: pl.DataFrame, lake: Lake, as_of: datetime, sym: str = "BTCUSDT") -> dict[str, Any]:
    """Hourly arrays on the complete grid (index i = the bar closing at close_ts[i])."""
    g = _grid(bars)
    n = g.height
    present = g["close"].is_not_null().to_numpy()
    c = g["close"].forward_fill().to_numpy().astype(float)
    hi = g["high"].to_numpy().astype(float)
    lo = g["low"].to_numpy().astype(float)
    hi = np.where(present, hi, c)
    lo = np.where(present, lo, c)
    lc = np.log(c)
    ret = np.r_[np.nan, np.diff(lc)]
    ret[~present] = np.nan
    ret[np.r_[False, ~present[:-1]]] = np.nan  # return across a gap is unknown
    r0 = np.nan_to_num(ret)
    ts_ms = g["close_ts"].dt.epoch("ms").to_numpy()
    S: dict[str, Any] = {"close_ts": g["close_ts"].to_list(), "ts_ms": ts_ms, "c": c, "lc": lc, "hi": hi, "lo": lo, "ret": ret,
                         "present": present, "src": g["src"].to_list()}
    miss = pl.Series((~present).astype(float))
    S["miss"] = {L: miss.rolling_sum(L + 1, min_samples=1).to_numpy() / (L + 1) for L in SCALES}
    # sigma_ref: hourly sd over the 720 h ending at t-L
    rs = pl.Series(ret)
    sd720 = rs.rolling_std(D["sigma_ref_window_h"], min_samples=D["sigma_ref_min_obs"]).to_numpy()
    lcs = pl.Series(lc)
    S["logA"], S["path"], S["valid"] = {}, {}, {}
    for L in SCALES:
        N = D["points"][str(L)]
        stride = L // (N - 1)
        idx = np.arange(n)[:, None] - L + stride * np.arange(N)[None, :]
        ok = idx[:, 0] >= 0
        P = lc[np.clip(idx, 0, None)]
        P = P - P.mean(1, keepdims=True)
        sdp = P.std(1)
        Pz = np.where(sdp[:, None] > 0, P / np.where(sdp > 0, sdp, 1)[:, None], 0.0)
        rng = (lcs.rolling_max(L + 1).to_numpy() - lcs.rolling_min(L + 1).to_numpy())
        sref = np.r_[np.full(L, np.nan), sd720[:-L]]
        A = rng / (sref * math.sqrt(L))
        val = ok & (S["miss"][L] <= D["max_missing_frac"]) & (sdp > 0) & np.isfinite(A) & (A > 0) & present
        S["path"][L] = Pz.astype(np.float32)
        S["logA"][L] = np.where(val, np.log(np.where(A > 0, A, 1)), np.nan)
        S["valid"][L] = val
    # state features
    rv24 = np.sqrt(pl.Series(r0 ** 2).rolling_sum(D["rv_window_h"]).to_numpy())
    rv24[S["miss"][24] > D["max_missing_frac"]] = np.nan
    S["rv24"] = rv24
    S["rv168"] = np.sqrt(pl.Series(r0 ** 2).rolling_sum(168).to_numpy())
    S["s168sq"] = pl.Series(r0 ** 2).rolling_mean(D["wf_scale_var_h"]).to_numpy()
    S["rv_pct"] = _rolling_pct(rv24, D["rv_pct_window_h"], D["rv_pct_min_h"])
    S["rv168_pct"] = _rolling_pct(S["rv168"], D["rv_pct_window_h"], D["rv_pct_min_h"])
    S["dd30"] = lc - np.log(pl.Series(hi).rolling_max(D["dd_window_h"], min_samples=D["dd_window_h"] // 2).to_numpy())
    S["dd1y"] = lc - np.log(pl.Series(hi).rolling_max(8760, min_samples=24 * 180).to_numpy())
    # v0 cell on 1h bars (regimes.label, bar_hours 1), computed on the present bars then mapped to the grid
    b = bars.with_columns((pl.col("event_ts") + timedelta(hours=1)).alias("close_ts"))
    f = label(enrich_bars(b), 1.0).select("event_ts", "cell", "dir_score")
    gm = g.select("event_ts").join(f, on="event_ts", how="left")
    cell = gm["cell"].fill_null(strategy="forward").to_list()
    S["cell"] = cell
    k = np.array([int(x[1:]) if x else 0 for x in cell])
    S["cell_dir"] = np.where(k > 0, 1.0 - ((k - 1) // 3), np.nan)  # up=+1, range=0, down=-1
    S["cell_vol"] = np.where(k > 0, ((k - 1) % 3) - 1.0, np.nan)    # low=-1, normal=0, high=+1
    # funding percentile, OI 72h change (as of known_ts)
    kn, rank, rate = funding_rank(lake, sym, as_of)
    S["funding_pct"] = _asof(ts_ms, kn, rank) if len(kn) else np.full(n, np.nan)
    S["funding_rate"] = _asof(ts_ms, kn, rate) if len(kn) else np.full(n, np.nan)
    if len(kn):
        S["funding_pct"][ts_ms < kn[0]] = np.nan
    o = lake.read_asof("open_interest", as_of, symbols=[sym], filters={"venue": "bybit", "interval": "4h"}).sort("known_ts")
    if o.height:
        okn = o["known_ts"].dt.epoch("ms").to_numpy()
        oi = _asof(ts_ms, okn, np.log(o["open_interest"].to_numpy().astype(float)))
        oi[ts_ms < okn[0]] = np.nan
    else:
        oi = np.full(n, np.nan)
    S["log_oi"] = oi
    S["oi72"] = oi - np.r_[np.full(D["oi_lag_h"], np.nan), oi[:-D["oi_lag_h"]]]
    S["oi24"] = oi - np.r_[np.full(24, np.nan), oi[:-24]]
    cal = add_calendar(pl.DataFrame({"event_ts": g["close_ts"]}))
    S["weekend"] = cal["tradfi_weekend"].to_numpy().astype(float)
    S["session"] = cal["session"].to_list()
    S["n"] = n
    return S


def state_matrix(S: dict) -> np.ndarray:
    return np.column_stack([S[f] for f in FEATS]).astype(float)


# ====================================================================== distances

def distances(S: dict, q: int, pool: np.ndarray, X: np.ndarray, sd: np.ndarray) -> dict[str, np.ndarray]:
    """Per-scale distances d_s (and their parts) from query hour q to the pool hours."""
    w = np.array([D["state_features"][f] for f in FEATS])
    xq = X[q]
    Z = (X[pool] - xq[None, :]) / sd[None, :]
    m = np.isfinite(Z) & np.isfinite(xq)[None, :]
    wm = w[None, :] * m
    dstate = np.sqrt(np.sum(np.where(m, Z, 0.0) ** 2 * wm, 1) / np.maximum(wm.sum(1), 1e-12))
    out: dict[str, np.ndarray] = {"state": dstate, "state_wfrac": wm.sum(1) / w.sum()}
    for L in SCALES:
        N = D["points"][str(L)]
        rho = S["path"][L][pool].astype(np.float64) @ S["path"][L][q].astype(np.float64) / N
        amp = np.abs(S["logA"][L][pool] - S["logA"][L][q])
        out[f"rho_{L}"] = rho
        out[f"amp_{L}"] = amp
        out[f"d_{L}"] = D["w_shape"] * (1 - rho) + D["w_amp"] * amp + D["w_state"] * dstate
    return out


def pct_rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = (np.arange(len(x)) + 0.5) / len(x)
    return r


def combine(dist: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([pct_rank(dist[f"d_{L}"]) for L in SCALES], axis=0)


def decluster(order: np.ndarray, ts_h: np.ndarray, k: int, sep_h: int) -> list[int]:
    """Greedy: walk the pool in order of increasing distance, keep a hour if >= sep_h from every kept one."""
    kept: list[int] = []
    kt: list[int] = []
    for j in order:
        t = ts_h[j]
        if all(abs(t - u) >= sep_h for u in kt):
            kept.append(int(j))
            kt.append(int(t))
            if len(kept) >= k:
                break
    return kept


# ====================================================================== outcomes

def outcomes(S: dict, i: int) -> dict[str, Any]:
    n, lc, hi, lo, c = S["n"], S["lc"], S["hi"], S["lo"], S["c"]
    o: dict[str, Any] = {}
    for h in D["fwd_h"]:
        j = i + h
        ok = j < n and S["present"][i + 1:j + 1].all()
        o[f"r{h}"] = float(lc[j] - lc[i]) if ok else None
    for h in (72, 168):
        j = min(i + h, n - 1)
        if i + h >= n:
            o[f"mdd{h}"] = o[f"mru{h}"] = None
            continue
        o[f"mdd{h}"] = float(np.log(lo[i + 1:j + 1].min() / c[i]))
        o[f"mru{h}"] = float(np.log(hi[i + 1:j + 1].max() / c[i]))
    if i + 168 < n:
        r = np.nan_to_num(S["ret"][i + 1:i + 169])
        o["rv7d"] = float(math.sqrt(np.sum(r ** 2)))
        o["rv7d_ratio"] = float(o["rv7d"] / S["rv168"][i]) if S["rv168"][i] > 0 else None
        seg = lo[i + 1:i + 169]
        o["hours_to_low"] = int(np.argmin(seg) + 1)
        o["low7d"] = float(np.log(seg.min() / c[i]))
        for w in (24, 72):
            pl_ = lo[max(0, i - w + 1):i + 1].min()
            ph = hi[max(0, i - w + 1):i + 1].max()
            u = np.flatnonzero(seg < pl_)
            v = np.flatnonzero(hi[i + 1:i + 169] > ph)
            tu = int(u[0]) + 1 if u.size else None
            tv = int(v[0]) + 1 if v.size else None
            o[f"undercut{w}"] = tu is not None
            o[f"undercut{w}_h"] = tu
            o[f"exceed{w}_h"] = tv
            o[f"undercut{w}_first"] = (None if tu is None and tv is None else
                                       "tie" if tu is not None and tv is not None and tu == tv else
                                       bool(tu is not None and (tv is None or tu < tv)))
    else:
        o.update({"rv7d": None, "rv7d_ratio": None, "hours_to_low": None, "low7d": None})
    for lab, h in (("+1d", 24), ("+3d", 72), ("+7d", 168)):
        o[f"cell{lab}"] = S["cell"][i + h] if i + h < n else None
    return o


def fwd_path(S: dict, i: int, h: int) -> list[float] | None:
    if i + h >= S["n"]:
        return None
    return (S["lc"][i:i + h + 1] - S["lc"][i]).tolist()


def back_path(S: dict, i: int, h: int) -> list[float] | None:
    if i - h < 0:
        return None
    return (S["lc"][i - h:i + 1] - S["lc"][i]).tolist()


# ====================================================================== Part A: now

def describe_now(S: dict, sym: str) -> dict[str, Any]:
    q = S["n"] - 1
    while not S["present"][q]:
        q -= 1
    t = S["close_ts"][q]
    paths = {}
    for L in SCALES:
        if q - L < 0:
            continue
        p = S["lc"][q - L:q + 1] - S["lc"][q - L]
        sref = S["logA"][L][q]
        paths[str(L)] = {"ret": float(p[-1]), "max": float(p.max()), "min": float(p.min()), "range": float(p.max() - p.min()),
                         "range_vol_units": float(math.exp(sref)) if np.isfinite(sref) else None,
                         "pos_in_range": float((p[-1] - p.min()) / (p.max() - p.min())) if p.max() > p.min() else None,
                         "path": p.tolist()}
    ann = math.sqrt(24 * 365)
    hi, lo = S["hi"], S["lo"]
    d = {"symbol": sym, "as_of_close": t, "close": float(S["c"][q]), "paths": paths,
         "rv24_ann": float(S["rv24"][q] / math.sqrt(24) * ann) if np.isfinite(S["rv24"][q]) else None,
         "rv24_pct_1y": float(S["rv_pct"][q]) if np.isfinite(S["rv_pct"][q]) else None,
         "rv7d_ann": float(S["rv168"][q] / math.sqrt(168) * ann), "rv7d_pct_1y": float(S["rv168_pct"][q]) if np.isfinite(S["rv168_pct"][q]) else None,
         "dd30": float(S["dd30"][q]), "dd1y": float(S["dd1y"][q]) if np.isfinite(S["dd1y"][q]) else None,
         "high30": float(hi[q - 719:q + 1].max()), "high1y": float(hi[max(0, q - 8759):q + 1].max()),
         "high1y_ts": S["close_ts"][max(0, q - 8759) + int(np.argmax(hi[max(0, q - 8759):q + 1]))] - timedelta(hours=1),
         "low7d": float(lo[q - 167:q + 1].min()), "low7d_ts": S["close_ts"][q - 167 + int(np.argmin(lo[q - 167:q + 1]))] - timedelta(hours=1),
         "high7d": float(hi[q - 167:q + 1].max()), "low30": float(lo[q - 719:q + 1].min()),
         "low1y": float(lo[max(0, q - 8759):q + 1].min()),
         "cell": S["cell"][q], "cells_last_72h": _runs(S["cell"][q - 71:q + 1]),
         "funding_rate": float(S["funding_rate"][q]) if np.isfinite(S["funding_rate"][q]) else None,
         "funding_pct_90d": float(S["funding_pct"][q]) if np.isfinite(S["funding_pct"][q]) else None,
         "oi24": float(S["oi24"][q]) if np.isfinite(S["oi24"][q]) else None, "oi72": float(S["oi72"][q]) if np.isfinite(S["oi72"][q]) else None,
         "oi7d": float(S["log_oi"][q] - S["log_oi"][q - 168]) if np.isfinite(S["log_oi"][q] - S["log_oi"][q - 168]) else None,
         "weekend": bool(S["weekend"][q]), "session": S["session"][q], "weekday": t.strftime("%A")}
    # weekend position: hours since the tradfi weekend opened, hours to the CME reopen (Sun 18:00 ET)
    wk = S["weekend"]
    if wk[q]:
        j = q
        while j > 0 and wk[j - 1]:
            j -= 1
        d["hours_into_weekend"] = q - j
        k = q
        while k + 1 < S["n"] and wk[k + 1]:
            k += 1
        from zoneinfo import ZoneInfo

        ny = t.astimezone(ZoneInfo("America/New_York"))
        sun = (ny + timedelta(days=(6 - ny.weekday()) % 7)).replace(hour=18, minute=0, second=0, microsecond=0)
        d["hours_to_cme_reopen"] = (sun - ny).total_seconds() / 3600
    # the 24h high/low and ATR(14, 1h)
    tr = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - S["c"][:-1]), np.abs(lo[1:] - S["c"][:-1])))
    d["atr14_1h"] = float(np.mean(tr[q - 14:q]))
    d["high24"], d["low24"] = float(hi[q - 23:q + 1].max()), float(lo[q - 23:q + 1].min())
    d["high72"], d["low72"] = float(hi[q - 71:q + 1].max()), float(lo[q - 71:q + 1].min())
    d["q"] = q
    return d


def _runs(cells: list) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for c in cells:
        if out and out[-1][0] == c:
            out[-1] = (c, out[-1][1] + 1)
        else:
            out.append((c, 1))
    return out


# ====================================================================== Part B: search

def search(as_of: datetime | None = None) -> dict[str, Any]:
    as_of = as_of or now()
    lake = Lake()
    bars, meta = btc_bars(lake, as_of)
    S = build_state(bars, lake, as_of)
    now_d = describe_now(S, "BTCUSDT")
    q = now_d["q"]
    t_now = S["close_ts"][q]
    secondary = {}
    for sym in ("ETHUSDT", "SOLUSDT"):
        b = symbol_bars(lake, sym, as_of)
        Ss = build_state(b, lake, as_of, sym)
        dd = describe_now(Ss, sym)
        dd.pop("q")
        secondary[sym] = dd
    valid = np.all([S["valid"][L] for L in SCALES], axis=0)
    cut = S["ts_ms"][q] - D["exclude_recent_days"] * 86_400_000
    pool = np.flatnonzero(valid & (S["ts_ms"] <= cut))
    X = state_matrix(S)
    sd = np.nanstd(X[pool], axis=0)
    sd[~(sd > 0)] = 1.0
    dist = distances(S, q, pool, X, sd)
    comb = combine(dist)
    ts_h = (S["ts_ms"][pool] // 3_600_000).astype(np.int64)
    rankings: dict[str, list[dict]] = {}

    def row(j: int, scale: str, rank: int) -> dict:
        i = int(pool[j])
        r = {"rank": rank, "scale": scale, "close_ts": S["close_ts"][i], "idx": i, "close": float(S["c"][i]), "src": S["src"][i],
             "cell": S["cell"][i], "combined_rank": float(comb[j]), "similarity_combined": float(1 - comb[j]),
             "state_dist": float(dist["state"][j]), "state_wfrac": float(dist["state_wfrac"][j]),
             "state": {f: (float(X[i, k]) if np.isfinite(X[i, k]) else None) for k, f in enumerate(FEATS)}}
        for L in SCALES:
            r[f"rho_{L}"] = float(dist[f"rho_{L}"][j])
            r[f"amp_{L}"] = float(dist[f"amp_{L}"][j])
            r[f"d_{L}"] = float(dist[f"d_{L}"][j])
            r[f"rankpct_{L}"] = None
        r.update(outcomes(S, i))
        return r

    pr = {L: pct_rank(dist[f"d_{L}"]) for L in SCALES}
    for L in SCALES:
        order = np.argsort(dist[f"d_{L}"], kind="stable")
        kept = decluster(order, ts_h, D["top_k"], D["decluster_h"])
        rows = []
        for r_, j in enumerate(kept, 1):
            x = row(j, f"{L}h", r_)
            x["similarity"] = float(1 - dist[f"d_{L}"][j])
            for LL in SCALES:
                x[f"rankpct_{LL}"] = float(pr[LL][j])
            rows.append(x)
        rankings[f"{L}h"] = rows
    order = np.argsort(comb, kind="stable")
    kept = decluster(order, ts_h, D["top_k"], D["decluster_h"])
    rows = []
    for r_, j in enumerate(kept, 1):
        x = row(j, "combined", r_)
        x["similarity"] = float(1 - comb[j])
        for LL in SCALES:
            x[f"rankpct_{LL}"] = float(pr[LL][j])
        x["fwd_path"] = fwd_path(S, x["idx"], D["path_h"])
        x["back_path_720"] = back_path(S, x["idx"], 720)
        rows.append(x)
    rankings["combined"] = rows
    # unconditional: every pool hour (outcomes vectorised)
    lc, n = S["lc"], S["n"]
    unc: dict[str, np.ndarray] = {}
    for h in D["fwd_h"]:
        ok = pool[pool + h < n]
        unc[f"r{h}"] = lc[ok + h] - lc[ok]
    for h in (72, 168):
        ok = pool[pool + h < n]
        lo_ = pl.Series(S["lo"]).reverse().rolling_min(h).reverse().to_numpy()  # min lo[i..i+h-1]
        hi_ = pl.Series(S["hi"]).reverse().rolling_max(h).reverse().to_numpy()
        unc[f"mdd{h}"] = np.log(lo_[ok + 1] / S["c"][ok])
        unc[f"mru{h}"] = np.log(hi_[ok + 1] / S["c"][ok])
    fan = {"h": list(range(0, D["path_h"] + 1, 3))}
    for qq in (10, 25, 50, 75, 90):
        fan[f"q{qq}"] = []
    for h in fan["h"]:
        ok = pool[pool + h < n]
        x = lc[ok + h] - lc[ok]
        for qq in (10, 25, 50, 75, 90):
            fan[f"q{qq}"].append(float(np.quantile(x, qq / 100)))
    rng = np.random.default_rng(SEED)
    agg = aggregate(rankings["combined"], unc, rng)
    agg_scales = {k: aggregate(v, unc, rng, nboot=1000) for k, v in rankings.items() if k != "combined"}
    # coverage of state features in the pool, by year
    yrs = np.array([S["close_ts"][i].year for i in pool])
    cov = {int(y): {f: float(np.mean(np.isfinite(X[pool[yrs == y], k]))) for k, f in enumerate(FEATS)} for y in np.unique(yrs)}
    res = {"as_of": as_of, "t_now": t_now, "data": meta, "pool_n": len(pool), "pool_first": S["close_ts"][int(pool[0])],
           "pool_last": S["close_ts"][int(pool[-1])], "feature_sd": dict(zip(FEATS, sd.tolist(), strict=True)), "coverage_by_year": cov,
           "now": now_d, "now_state": {f: (float(X[q, k]) if np.isfinite(X[q, k]) else None) for k, f in enumerate(FEATS)},
           "now_path_720": back_path(S, q, 720), "secondary": secondary, "rankings": rankings, "aggregate": agg,
           "aggregate_scales": agg_scales, "unc_fan": fan, "design": D}
    write_json(OUT / "search.json", res)
    return res


def aggregate(rows: list[dict], unc: dict[str, np.ndarray], rng: np.random.Generator, nboot: int | None = None) -> dict[str, Any]:
    nboot = nboot or D["nboot"]
    out: dict[str, Any] = {}
    for k in [f"r{h}" for h in D["fwd_h"]] + ["mdd72", "mru72", "mdd168", "mru168"]:
        x = np.array([r[k] for r in rows if r.get(k) is not None], float)
        u = unc[k][np.isfinite(unc[k])]
        if len(x) < 3:
            continue
        bs = rng.choice(x, (nboot, len(x)))
        rs = rng.choice(u, (nboot, len(x)))
        med_b, mean_b = np.median(bs, 1), bs.mean(1)
        med_r, mean_r = np.median(rs, 1), rs.mean(1)
        out[k] = {"n": len(x), "median": float(np.median(x)), "mean": float(x.mean()), "q10": float(np.quantile(x, 0.1)),
                  "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)), "q90": float(np.quantile(x, 0.9)),
                  "share_up": float(np.mean(x > 0)), "sd": float(x.std(ddof=1)),
                  "median_ci": [float(np.quantile(med_b, 0.05)), float(np.quantile(med_b, 0.95))],
                  "mean_ci": [float(np.quantile(mean_b, 0.05)), float(np.quantile(mean_b, 0.95))],
                  "unc": {"n": len(u), "median": float(np.median(u)), "mean": float(u.mean()), "q10": float(np.quantile(u, 0.1)),
                          "q25": float(np.quantile(u, 0.25)), "q75": float(np.quantile(u, 0.75)), "q90": float(np.quantile(u, 0.9)),
                          "share_up": float(np.mean(u > 0)), "sd": float(u.std())},
                  "null_band_median": [float(np.quantile(med_r, 0.05)), float(np.quantile(med_r, 0.95))],
                  "null_band_mean": [float(np.quantile(mean_r, 0.05)), float(np.quantile(mean_r, 0.95))],
                  "null_band_sd": [float(np.quantile(rs.std(1, ddof=1), 0.05)), float(np.quantile(rs.std(1, ddof=1), 0.95))],
                  "p_median_le": float(np.mean(med_r <= np.median(x)))}
    for w in (24, 72):
        v = [r.get(f"undercut{w}_first") for r in rows]
        out[f"undercut{w}_first"] = {"yes": sum(1 for x in v if x is True), "no": sum(1 for x in v if x is False),
                                     "tie": sum(1 for x in v if x == "tie"), "neither": sum(1 for x in v if x is None)}
        out[f"undercut{w}_any"] = int(sum(1 for r in rows if r.get(f"undercut{w}")))
    h2l = [r["hours_to_low"] for r in rows if r.get("hours_to_low") is not None]
    out["hours_to_low"] = {"median": float(np.median(h2l)), "q25": float(np.quantile(h2l, 0.25)), "q75": float(np.quantile(h2l, 0.75)),
                           "share_within_24h": float(np.mean(np.array(h2l) <= 24))} if h2l else None
    for lab in ("+1d", "+3d", "+7d"):
        cs = [r.get(f"cell{lab}") for r in rows if r.get(f"cell{lab}")]
        out[f"cells{lab}"] = {c: cs.count(c) for c in sorted(set(cs))}
    rv = [r["rv7d_ratio"] for r in rows if r.get("rv7d_ratio") is not None]
    out["rv7d_ratio_median"] = float(np.median(rv)) if rv else None
    return out


# ====================================================================== Part D: H-ANA1 walk-forward

def nw_t(x: np.ndarray, lags: int) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    m = float(x.mean())
    e = x - m
    v = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
    se = math.sqrt(v / n) if v > 0 else float("nan")
    return m, m / se if se > 0 else float("nan"), n


def nw_neff(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    v0 = float(e @ e) / n
    v = v0
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
    return float(n * v0 / v) if v > 0 else float("nan")


def qlike(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    x = np.maximum(V, 1e-14) / F
    return x - np.log(x) - 1


def block_boot_skill(bs_a: np.ndarray, bs_b: np.ndarray, block: int, nboot: int, rng: np.random.Generator) -> list[float]:
    n = len(bs_a)
    nb = math.ceil(n / block)
    out = np.empty(nboot)
    for b in range(nboot):
        st = rng.integers(0, max(1, n - block + 1), nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = 1 - bs_a[idx].mean() / bs_b[idx].mean()
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def walkforward(as_of: datetime | None = None, log=print) -> dict[str, Any]:
    from ai_trader.research.options_compression import predictive

    as_of = as_of or now()
    lake = Lake()
    bars, meta = btc_bars(lake, as_of)
    S = build_state(bars, lake, as_of)
    X = state_matrix(S)
    valid = np.all([S["valid"][L] for L in SCALES], axis=0)
    vidx = np.flatnonzero(valid)
    n = S["n"]
    lc = S["lc"]
    pres = S["present"]
    csum_bad = np.r_[0, np.cumsum(~pres)]

    def fwd(i: np.ndarray, h: int) -> np.ndarray:
        j = i + h
        ok = j < n
        r = np.full(len(i), np.nan)
        r[ok] = lc[j[ok]] - lc[i[ok]]
        bad = np.zeros(len(i), bool)
        bad[ok] = (csum_bad[j[ok] + 1] - csum_bad[i[ok] + 1]) > 0
        r[bad] = np.nan
        return r

    Rv = {H: fwd(vidx, 24 * H) for H in D["wf_horizons_days"]}
    s2 = S["s168sq"]
    ts_h_all = (S["ts_ms"] // 3_600_000).astype(np.int64)
    pos = {int(t): i for i, t in enumerate(S["ts_ms"])}
    har: dict[int, pl.DataFrame] = {}
    grids: dict[int, dict] = {}
    hb = bars.select("event_ts", "open", "high", "low", "close")
    for H in D["wf_horizons_days"]:
        log(f"log-HAR-t x weekday H={H}d")
        df, gr = predictive(hb, H, log=None)
        har[H], grids[H] = df, gr
    rng = np.random.default_rng(SEED)
    res: dict[str, Any] = {"as_of": as_of, "data": meta, "design": D, "horizons": {}}
    rows_all: dict[int, list] = {}
    for H in D["wf_horizons_days"]:
        h = 24 * H
        df = har[H].filter(pl.col("s").is_not_nan() & pl.col("R").is_not_nan())
        Eu2 = {k: float(np.mean(np.asarray(g) ** 2)) for k, g in grids[H].items()}
        rows = []
        for o_ts, R, s, rid in zip(df["origin"].to_list(), df["R"].to_numpy(), df["s"].to_numpy(), df["refit"].to_numpy(), strict=True):
            q = pos.get(int(o_ts.timestamp() * 1000))
            if q is None or not valid[q]:
                continue
            lim = ts_h_all[q] - D["wf_embargo_h"] - D["wf_outcome_h"]
            npool = int(np.searchsorted(ts_h_all[vidx], lim, side="right"))
            if npool < 24 * 180:
                continue
            pool = vidx[:npool]
            sd = np.nanstd(X[pool], axis=0)
            sd[~(sd > 0)] = 1.0
            dist = distances(S, q, pool, X, sd)
            comb = combine(dist)
            kept = decluster(np.argsort(comb, kind="stable"), ts_h_all[pool], D["top_k"], D["decluster_h"])
            kk = np.array(kept)
            Rk = Rv[H][kk]
            ok = np.isfinite(Rk)
            Rk, ii = Rk[ok], pool[kk[ok]]
            if len(Rk) < 10:
                continue
            F_a = s2[q] * np.mean(Rk ** 2 / s2[ii])
            F_raw = float(np.mean(Rk ** 2))
            F_h = float(s) ** 2 * Eu2[int(rid)]
            p_up = (np.sum(Rk > 0) + 1) / (len(Rk) + 2)
            rows.append({"origin": o_ts, "R": float(R), "F_ana": float(F_a), "F_raw": F_raw, "F_har": F_h, "p_up": float(p_up),
                         "k": len(Rk), "pool": npool})
        log(f"  H={H}d origins {len(rows)}")
        rows_all[H] = rows
        r = pl.DataFrame(rows)
        org = r["origin"].to_list()
        R = r["R"].to_numpy()
        y = (R > 0).astype(float)
        # expanding base rate from origins whose outcome had resolved by the origin
        end = np.array([o + timedelta(hours=h) for o in org])
        base = np.empty(len(R))
        for i, o in enumerate(org):
            m = end[:i] <= o
            base[i] = y[:i][m].mean() if m.sum() >= 30 else 0.5
        V = R ** 2
        dq = qlike(V, r["F_ana"].to_numpy()) - qlike(V, r["F_har"].to_numpy())
        dq_raw = qlike(V, r["F_raw"].to_numpy()) - qlike(V, r["F_har"].to_numpy())
        bs_a = (r["p_up"].to_numpy() - y) ** 2
        bs_b = (base - y) ** 2
        segs = np.array([seg_of(o) for o in org])
        hr: dict[str, Any] = {}
        for sg in ("llm_in_sample", "clean_holdout"):
            m = segs == sg
            if m.sum() < 5:
                hr[sg] = {"n": int(m.sum())}
                continue
            lag = D["wf_nw_lags"][str(H)]
            md, t, nn = nw_t(dq[m], lag)
            mr, tr, _ = nw_t(dq_raw[m], lag)
            sk = 1 - bs_a[m].mean() / bs_b[m].mean()
            ci = block_boot_skill(bs_a[m], bs_b[m], D["wf_block_days"], D["nboot"], rng)
            hr[sg] = {"n": int(nn), "first": org[int(np.flatnonzero(m)[0])], "last": org[int(np.flatnonzero(m)[-1])],
                      "n_eff_nonoverlap": float(nn / H), "n_eff_nw": nw_neff(dq[m], lag),
                      "qlike_ana": float(np.mean(qlike(V[m], r["F_ana"].to_numpy()[m]))),
                      "qlike_har": float(np.mean(qlike(V[m], r["F_har"].to_numpy()[m]))),
                      "qlike_raw": float(np.mean(qlike(V[m], r["F_raw"].to_numpy()[m]))),
                      "dqlike": md, "dqlike_t": t, "dqlike_raw": mr, "dqlike_raw_t": tr,
                      "rel_dqlike": float(md / np.mean(qlike(V[m], r["F_har"].to_numpy()[m]))),
                      "share_ana_better": float(np.mean(dq[m] < 0)),
                      "log_ratio_F_median": float(np.median(np.log(r["F_ana"].to_numpy()[m] / r["F_har"].to_numpy()[m]))),
                      "mz_ana": _mz(V[m], r["F_ana"].to_numpy()[m]), "mz_har": _mz(V[m], r["F_har"].to_numpy()[m]),
                      "brier_ana": float(bs_a[m].mean()), "brier_base": float(bs_b[m].mean()), "brier_skill": float(sk),
                      "brier_skill_ci95": ci, "share_up": float(y[m].mean()), "p_up_mean": float(r["p_up"].to_numpy()[m].mean()),
                      "sign_hit": float(np.mean((r["p_up"].to_numpy()[m] > 0.5) == (y[m] > 0)))}
        res["horizons"][str(H)] = hr
    res["verdicts"] = {sg: verdict(res, sg) for sg in ("llm_in_sample", "clean_holdout")}
    write_json(OUT / "walkforward.json", res)
    write_json(OUT / "walkforward_rows.json", {str(k): v for k, v in rows_all.items()})
    return res


def _mz(V: np.ndarray, F: np.ndarray) -> dict[str, float]:
    """Mincer-Zarnowitz in logs: ln V = a + b ln F (V floored)."""
    y = np.log(np.maximum(V, 1e-12))
    x = np.log(F)
    b, a = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b), "corr_log": float(np.corrcoef(x, y)[0, 1])}


def verdict(res: dict, sg: str) -> str:
    hz = [res["horizons"][str(H)].get(sg, {}) for H in D["wf_horizons_days"]]
    if any("dqlike" not in h for h in hz):
        return "inconclusive"
    passes = [h["dqlike"] < 0 and h["dqlike_t"] < -2 for h in hz]
    worse = [h["dqlike"] > 0 and h["dqlike_t"] > 2 for h in hz]
    if all(passes):
        return "supports"
    if any(worse) and not any(passes):
        return "contradicts"
    return "inconclusive"


def record_results(res: dict) -> None:
    from ai_trader.research.hypotheses import record

    for sg in ("llm_in_sample", "clean_holdout"):
        m: dict[str, Any] = {"test": "walk-forward daily 08:00 UTC origins, 20 declustered multi-scale analogs (7-day embargo) vs log-HAR-t x weekday",
                             "trials": "1 hyp x 2 segments x 2 horizons x (QLIKE + Brier) + 1 secondary variant (unscaled analog variance)"}
        for H in D["wf_horizons_days"]:
            h = res["horizons"][str(H)].get(sg, {})
            for k in ("n", "n_eff_nonoverlap", "n_eff_nw", "qlike_ana", "qlike_har", "dqlike", "dqlike_t", "dqlike_raw", "dqlike_raw_t",
                      "share_ana_better", "brier_skill", "brier_skill_ci95", "share_up", "sign_hit"):
                if k in h:
                    m[f"{24 * H}h_{k}"] = h[k]
        record("H-ANA1", sg, res["verdicts"][sg], m,
               note="(a) QLIKE vs log-HAR-t x weekday decides the verdict; (b) Brier skill reported (expected null). BTC 1h spliced Deribit (2020-01..03) + Bybit; no pre-2020 BTC in the lake.",
               data="data/research/analogs/walkforward.json")


def design() -> None:
    write_json(OUT / "design.json", {"frozen": "2026-11-15T17:56:56Z (diary decision)", **D})


def main() -> None:
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "search", "walkforward"])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args()
    as_of = datetime.fromisoformat(a.as_of).astimezone(UTC) if a.as_of else None
    if a.cmd == "design":
        design()
    elif a.cmd == "search":
        r = search(as_of)
        print(json.dumps({k: r[k] for k in ("t_now", "pool_n", "pool_first", "pool_last")}, default=str))
    else:
        r = walkforward(as_of)
        print(json.dumps({"verdicts": r["verdicts"], "horizons": r["horizons"]}, default=str, indent=1))
        if a.record:
            record_results(r)
            print("recorded")


if __name__ == "__main__":
    main()
