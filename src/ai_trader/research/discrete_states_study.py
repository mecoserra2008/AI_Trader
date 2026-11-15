"""Discretised observation space: how much of the nonlinear next-day volatility signal a few cells carry, and whether
the cells keep their meaning across the knowledge cutoff (H-DS1..H-DS3, program 2026-11-12, agent C). Research only.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.discrete_states_study walk
    ... evaluate [--record]   verdicts, Holm within DS, research/reports/discrete_states_2026-11-12.{json,md}

Panel and protocol are those of H-NN1 (ml_signals_study): universe L, the 52 point-in-time features of
ai_trader.ml.features on complete rows (38,275 rows), decision at T = 00:00 UTC of D+1, expanding monthly refits from
2022-04 with training rows label_end <= month - 1 day (purge + embargo). The baseline is the same per-asset log-HAR,
re-fitted here month by month with the identical code path (checked against wf_preds.F_har); XGB forecasts are read
from H-NN1's wf_preds.parquet so the XGB gain is the exact H-NN1 reference on the same rows.

Cell target (vol). For each refit, u_i = log(RV_{D+1} / F_HAR,i) on the training rows, with F_HAR,i the fitted value of
that month's HAR fit (exp(xb + s^2/2)). Cell value = empirical-Bayes shrunk mean of u - mean(u) (discretize.eb_shrink:
date-clustered variances, one pooled kappa per refit and discretisation estimated on its own training sample, shrink
toward 0 = no adjustment). Forecast for a test row = F_HAR x exp(cell value).

Discretisations (fixed before evaluation; each fitted on training rows only, <= 64 cells):
  WD7   forecast weekday (7)                                            [calendar-only reference]
  V0    v0 regime: direction (down / range / up) x vol state (low / normal / high) (9)       [baseline]
  LAT1  weekday group (Mon / Tue-Fri / Sat / Sun) x vol_pct_1y terciles x v0 direction (36)
  LAT2  weekday group x lrv_w terciles (pooled 7d level) x mkt_ret_1d_z terciles (36)
  LAT3  weekday (7) x vol_pct_1y terciles x v0 direction (63)
  KM8, KM16, KM32, KM64   k-means codebooks on the top-10 in-sample H-NN3 TreeSHAP main-effect inputs (log_sigma,
        lrv_w, lrv_q, next_is_weekend, dow_sin, lrv_m, next_wk_hours, lrv_d, rq_ratio_d, v0_dir_score) plus dow_cos
        (completes the weekday circle), training-standardised; k-means++ / Lloyd, 3 inits, seed 0, fit on <= 20k rows
  HT3, HT6  honest regression trees of u on all 52 features + weekday, depth 3 / 6, min leaf max(150, 1% of the split
        half); split half = alternating calendar weeks of the training dates, leaf values from the other half
Direction (H-DS3) uses the same partitions except the vol trees, replaced by HT3r / HT6r grown on next-day
vol-scaled return (ret_next_z winsorised at +-5). Cell probability p = b + shrunk(mean(y - b)) with b = training base
rate of y = 1{ret_{D+1} > 0} (centre b, shrink toward b), clipped to [0.001, 0.999].

H-DS1 per segment (QLIKE on RV_{D+1}, paired difference averaged over symbols per date, NW t; rows where every H-NN1
forecast and every DS forecast exists). Gain share = diff(DS - HAR) / diff(XGB - HAR); 95% CI by a 10-day moving-block
bootstrap over dates (1000, seed 0). The BEST discretisation is chosen on llm_in_sample QLIKE only (walk-forward rows
before the cutoff) and then read in both segments. Per segment: supports iff share >= 0.8 and t < -2; contradicts if
diff >= 0 or the share CI upper < 0.8; else inconclusive. Pooled (registry criterion): supports iff both segments
support; contradicts if either contradicts; else inconclusive. p (per segment) = max(Phi(t), P*(share < 0.8)).

H-DS2 (pooled only; the statement is a cross-segment comparison). Frozen partition = the best discretisation as fitted
at the first refit on/after the cutoff (2026-07, labels end before the cutoff). All rows are assigned with it; per
segment the EB-shrunk mean of u_oos = log(RV / walk-forward F_HAR) by cell (segment's own kappa). Spearman over cells
with >= 30 rows in both segments; CI: 10-day block bootstrap of dates within each segment (1000, seed 0), recomputing
cell means, kappa and the >= 30 filter. supports iff rho > 0.8 and CI lower > 0.6; contradicts if rho <= 0.6 or CI
upper <= 0.8; else inconclusive. p = P*(rho <= 0.8).

H-DS3 (null). The best direction discretisation = highest walk-forward llm_in_sample Brier skill (vs the training base
rate). (i) Its frozen-partition cell ordering by EB-shrunk mean winsorised ret_next_z: Spearman between segments (as
H-DS2); (ii) its holdout Brier skill with the H-NN2 block-bootstrap CI. clean_holdout: supports iff skill <= 0;
contradicts if CI lower > 0; else inconclusive. llm_in_sample: the same on the in-sample walk-forward rows
(descriptive analogue). Pooled: supports iff rho <= 0.3 and holdout skill <= 0; contradicts if the rho CI lower > 0.3
or the holdout skill CI lower > 0; else inconclusive. p = max(P*(rho > 0.3), P*(skill > 0)). All discretisations are
reported (descriptive).

Holm within DS over {H-DS1 (max of its segment p), H-DS2, H-DS3}; a 'supports' with Holm p > 0.05 becomes inconclusive.
Cell table: the best vol discretisation's frozen cells in plain words, rows and shrunk effect (%) per segment.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ai_trader.core.config import load  # noqa: E402
from ai_trader.ml import discretize as dz  # noqa: E402
from ai_trader.ml.features import ML_FEATURES  # noqa: E402
from ai_trader.research.decision_budget_study import block_bootstrap_idx, holm, purged_train_mask, test_mask  # noqa: E402
from ai_trader.research.ml_signals_study import (_har_design, _months, _next_month, arrays, brier_block,  # noqa: E402
                                                  first_test_month, load_panel, paired)
from ai_trader.research.ml_signals_study import root as ml_root  # noqa: E402

TAG = "2026-11-12"
SEGS = ("llm_in_sample", "clean_holdout")
HAR_MIN_ROWS = 180
COLS = (*ML_FEATURES, "wd")
CI = {c: i for i, c in enumerate(COLS)}
KM_FEATS = ("log_sigma", "lrv_w", "lrv_q", "next_is_weekend", "dow_sin", "lrv_m", "next_wk_hours", "lrv_d", "rq_ratio_d",
            "v0_dir_score", "dow_cos")
RET_CLIP = 5.0
MIN_CELL = 30
N_BOOT = 1000
SHARE_TARGET = 0.8

WD4 = (((1.0,), (2.0, 3.0, 4.0, 5.0), (6.0,), (7.0,)), ("forecast day Monday", "forecast day Tue-Fri", "forecast day Saturday", "forecast day Sunday"))
WD7 = (tuple((float(k),) for k in range(1, 8)), tuple(f"forecast day {d}" for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")))
V0DIR = (((-1.0,), (0.0,), (1.0,)), ("v0 trend down", "v0 range", "v0 trend up"))
V0VOL = (((0.0,), (1.0,), (2.0,)), ("v0 low vol", "v0 normal vol", "v0 high vol"))
TERC = ("in the bottom third", "in the middle third", "in the top third")

VOL_DISCS = ("WD7", "V0", "LAT1", "LAT2", "LAT3", "KM8", "KM16", "KM32", "KM64", "HT3", "HT6")
DIR_DISCS = ("WD7", "V0", "LAT1", "LAT2", "LAT3", "KM8", "KM16", "KM32", "KM64", "HT3r", "HT6r")
ALL_DISCS = tuple(dict.fromkeys([*VOL_DISCS, *DIR_DISCS]))

WORDS = {
    "log_sigma": "robust 1h sigma (log)", "lrv_d": "day-D log RV", "lrv_w": "7-day mean log RV", "lrv_m": "30-day mean log RV",
    "lrv_q": "90-day mean log RV", "vol_pct_1y": "vol percentile in its own 1y range", "rv_surprise": "RV surprise",
    "rv_term": "RV term structure (short minus long)", "vol_of_vol_30d": "30d vol-of-vol", "rq_ratio_d": "day-D quarticity ratio (one-big-hour share)",
    "semivar_7d": "7d downside semivariance share", "jump_share_7d": "7d jump share", "neg_share_d": "day-D downside variance share",
    "jump_share_d": "day-D jump share", "pk_ratio_d": "day-D Parkinson/RV", "gk_ratio_d": "day-D Garman-Klass/RV", "day_range_z": "day-D range z",
    "ret_1d_z": "1d vol-scaled return", "ret_3d_z": "3d vol-scaled return", "ret_7d_z": "7d vol-scaled return", "ret_30d_z": "30d vol-scaled return",
    "abs_ret_1d_z": "|1d vol-scaled return|", "x_ret7_volpct": "7d return x vol percentile", "er_7d": "7d efficiency ratio",
    "er_30d": "30d efficiency ratio", "tstat_7d": "7d trend t-stat", "tstat_30d": "30d trend t-stat", "dvr_7d": "7d directional variance ratio",
    "dvr_30d": "30d directional variance ratio", "vr_30d": "30d variance ratio", "ac1_4h_30d": "30d lag-1 autocorrelation of 4h returns",
    "ac1_1d_30d": "30d lag-1 autocorrelation of 1d returns", "v0_dir": "v0 trend state", "v0_vol": "v0 vol state", "v0_dir_score": "v0 trend score",
    "range_pos_30d": "position in 30d range", "dd_90d_z": "90d drawdown z", "funding_1d": "1d funding", "funding_z_90d": "funding z (90d)",
    "x_ret1_fundz": "1d return x funding z", "oi_chg_1d": "1d OI change", "oi_chg_7d": "7d OI change", "next_is_weekend": "forecast day weekend",
    "next_is_monday": "forecast day Monday", "dow_sin": "weekday sin", "dow_cos": "weekday cos", "next_wk_hours": "TradFi-weekend hours in forecast day",
    "next_is_mexpiry": "forecast day is a Deribit monthly expiry", "btc_ret_1d_z": "BTC 1d vol-scaled return", "mkt_lrv_d": "market day-D log RV",
    "mkt_lrv_w": "market 7-day mean log RV", "mkt_ret_1d_z": "market mean 1d vol-scaled return", "wd": "forecast weekday",
}
CAL_FEATS = ("next_is_weekend", "next_is_monday", "dow_sin", "dow_cos", "wd")


def out_dir() -> Path:
    d = ml_root() / "discrete"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make(name: str):
    """A fresh, unfitted partition (fixed specification)."""
    if name == "WD7":
        return dz.Lattice([dz.CategoricalDim("weekday", CI["wd"], *WD7)])
    if name == "V0":
        return dz.Lattice([dz.CategoricalDim("v0 direction", CI["v0_dir"], *V0DIR), dz.CategoricalDim("v0 vol", CI["v0_vol"], *V0VOL)])
    if name == "LAT1":
        return dz.Lattice([dz.CategoricalDim("weekday group", CI["wd"], *WD4), dz.QuantileDim("vol (own 1y percentile)", CI["vol_pct_1y"], 3, TERC),
                           dz.CategoricalDim("v0 direction", CI["v0_dir"], *V0DIR)])
    if name == "LAT2":
        return dz.Lattice([dz.CategoricalDim("weekday group", CI["wd"], *WD4), dz.QuantileDim("7d vol level (pooled)", CI["lrv_w"], 3, TERC),
                           dz.QuantileDim("market day (mean 1d z)", CI["mkt_ret_1d_z"], 3, ("down", "flat", "up"))])
    if name == "LAT3":
        return dz.Lattice([dz.CategoricalDim("weekday", CI["wd"], *WD7), dz.QuantileDim("vol (own 1y percentile)", CI["vol_pct_1y"], 3, TERC),
                           dz.CategoricalDim("v0 direction", CI["v0_dir"], *V0DIR)])
    if name.startswith("KM"):
        return dz.KMeansCodebook(int(name[2:]), tuple(CI[f] for f in KM_FEATS), n_init=3, max_iter=100, seed=0)
    if name.startswith("HT"):
        return dz.HonestTree(max_depth=int(name[2]), min_leaf=0.01, min_leaf_abs=150, n_bins=32)
    raise KeyError(name)


def _wd(c: pl.DataFrame) -> np.ndarray:
    return c.select((pl.col("date") + pl.duration(days=1)).dt.weekday().alias("w"))["w"].to_numpy().astype(float)


class HarRefit:
    """Per-asset log-HAR exactly as ml_signals_study.har_forecasts, also returning fitted values for training rows."""

    def __init__(self, full: pl.DataFrame, c: pl.DataFrame) -> None:
        h = full.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in ("lrv_d", "lrv_w", "lrv_m", "lrv_next")])
                        & (pl.col("rv_next") > 0))
        self.sym = c["symbol"].to_numpy()
        self.Xc = np.c_[np.ones(c.height), _har_design(c)]
        self.per = {}
        for s in np.unique(self.sym):
            hs = h.filter(pl.col("symbol") == s)
            self.per[s] = (np.c_[np.ones(hs.height), _har_design(hs)], hs["lrv_next"].to_numpy().astype(float), hs["label_end"].to_numpy())

    def month(self, m: np.datetime64, rows: np.ndarray) -> np.ndarray:
        """Forecast exp(xb + s^2/2) of the month-m fit for complete rows `rows` (bool mask); nan if no fit."""
        out = np.full(len(rows), np.nan)
        for s, (Xh, yh, leh) in self.per.items():
            sel = rows & (self.sym == s)
            if not sel.any():
                continue
            tr = purged_train_mask(leh, m)
            if tr.sum() < HAR_MIN_ROWS:
                continue
            b, *_ = np.linalg.lstsq(Xh[tr], yh[tr], rcond=None)
            s2 = float(np.mean((yh[tr] - Xh[tr] @ b) ** 2))
            out[sel] = np.exp(self.Xc[sel] @ b + 0.5 * s2)
        return out


# ------------------------------------------------------------------ plain-word descriptions


def _pct(x: np.ndarray, v: float) -> float:
    return float(100 * np.mean(x < v))


def _cal_value(f: str, k: int) -> float:
    ang = 2 * np.pi * (k - 1) / 7
    return {"next_is_weekend": float(k >= 6), "next_is_monday": float(k == 1), "dow_sin": math.sin(ang), "dow_cos": math.cos(ang), "wd": float(k)}[f]


def path_words(path: list, Xtr: np.ndarray) -> str:
    """Tree leaf path -> plain words; calendar conditions become a weekday set; other bounds get training percentiles."""
    cal = [(j, op, t) for j, op, t in path if COLS[j] in CAL_FEATS]
    parts = []
    if cal:
        days = [k for k in range(1, 8) if all((_cal_value(COLS[j], k) < t) == (op == "<") for j, op, t in cal)]
        names = [dz.WEEKDAYS[k - 1] for k in days]
        parts.append("forecast day " + ("/".join(names) if names else "(none)"))
    bounds: dict[int, list] = {}
    for j, op, t in path:
        if COLS[j] in CAL_FEATS:
            continue
        lo, hi = bounds.get(j, [-np.inf, np.inf])
        if op == ">=":
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        bounds[j] = [lo, hi]
    for j, (lo, hi) in bounds.items():
        x = Xtr[:, j]
        w = WORDS.get(COLS[j], COLS[j])
        if np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"{w} between p{_pct(x, lo):.0f} and p{_pct(x, hi):.0f} ({lo:.3g} to {hi:.3g})")
        elif np.isfinite(lo):
            parts.append(f"{w} above p{_pct(x, lo):.0f} (>= {lo:.3g})")
        else:
            parts.append(f"{w} below p{_pct(x, hi):.0f} (< {hi:.3g})")
    return "; ".join(parts) if parts else "all rows"


def profile(Xc: np.ndarray) -> dict:
    """Composition of a cell's rows (descriptive, for naming k-means and tree cells)."""
    if not len(Xc):
        return {}
    wd = Xc[:, CI["wd"]].astype(int)
    share = {dz.WEEKDAYS[k - 1]: float(np.mean(wd == k)) for k in range(1, 8)}
    dirs = Xc[:, CI["v0_dir"]]
    return {"weekday_share": share, "vol_pct_1y_mean": float(Xc[:, CI["vol_pct_1y"]].mean()),
            "v0_dir_share": {"down": float(np.mean(dirs < -0.5)), "range": float(np.mean(np.abs(dirs) < 0.5)), "up": float(np.mean(dirs > 0.5))},
            "mkt_ret_1d_z_mean": float(Xc[:, CI["mkt_ret_1d_z"]].mean())}


def profile_words(p: dict) -> str:
    if not p:
        return "empty"
    sh = p["weekday_share"]
    top = sorted(sh, key=lambda d: -sh[d])
    days = [d for d in top if sh[d] >= 0.2]
    if not days or sum(sh[d] for d in days) < 0.5:
        dtxt = "all weekdays"
    else:
        dtxt = "mostly " + "/".join(days) + f" ({100 * sum(sh[d] for d in days):.0f}%)"
    vp = p["vol_pct_1y_mean"]
    vtxt = "vol low in its 1y range" if vp < 0.35 else "vol high in its 1y range" if vp > 0.65 else "vol mid-range"
    ds = p["v0_dir_share"]
    dd = max(ds, key=ds.get)
    ttxt = f"v0 {dd} {100 * ds[dd]:.0f}%"
    return f"{dtxt}; {vtxt} (mean pct {vp:.2f}); {ttxt}"


def describe(name: str, part, cell: int, Xtr_cell: np.ndarray, Xtr: np.ndarray) -> str:
    if isinstance(part, dz.Lattice):
        return part.describe(cell)
    if isinstance(part, dz.HonestTree):
        return path_words(part.leaf_paths()[cell], Xtr)
    if isinstance(part, dz.KMeansCodebook):
        c = part.centroids[cell]
        order = np.argsort(-np.abs(c))[:3]
        cen = ", ".join(f"{WORDS[KM_FEATS[j]]} {c[j]:+.1f} sd" for j in order)
        return f"{profile_words(profile(Xtr_cell))}; centroid: {cen}"
    return str(cell)


# ------------------------------------------------------------------ walk-forward


def run_walk(limit: int | None = None) -> None:
    full, c = load_panel()
    a = arrays(c)
    X = np.c_[a["X"], _wd(c)]
    rv, yd = a["rv"], a["yd"]
    rz = np.clip(c["ret_next_z"].to_numpy().astype(float), -RET_CLIP, RET_CLIP)
    T, le = a["T"], a["le"]
    days = T.astype("datetime64[D]")
    wf = pl.read_parquet(ml_root() / "wf_preds.parquet")
    assert wf.height == c.height and (wf["T"] == c["T"]).all() and (wf["symbol"] == c["symbol"]).all()
    m0 = first_test_month(T, le)
    months = [m for m in _months(T) if m >= m0]
    if limit:
        months = months[:limit]
    cut = np.datetime64(load().settings["research"]["knowledge_cutoff"] + "T00:00:00", "us")
    frozen_m = next(m for m in months if m >= cut) if any(m >= cut for m in months) else None
    har = HarRefit(full, c)
    n = c.height
    F = {d: np.full(n, np.nan) for d in VOL_DISCS}
    P = {d: np.full(n, np.nan) for d in DIR_DISCS}
    base = np.full(n, np.nan)
    F_har = np.full(n, np.nan)
    info, frozen = [], {}
    for m in months:
        t0 = time.time()
        te = test_mask(T, m, _next_month(m))
        if not te.any():
            continue
        trm = purged_train_mask(le, m)
        Fall = har.month(m, trm | te)
        F_har[te] = Fall[te]
        tr = np.flatnonzero(trm)
        ok = np.isfinite(Fall[tr])
        u = np.full(len(tr), np.nan)
        u[ok] = np.log(rv[tr][ok] / Fall[tr][ok])
        split = dz.honest_halves(days[tr])
        b = float(yd[tr].mean())
        base[te] = b
        Xtr, Xte = X[tr], X[te]
        rec = {"month": str(m)[:7], "n_train": int(len(tr)), "n_test": int(te.sum()), "n_train_har": int(ok.sum()), "discs": {}}
        for d in ALL_DISCS:
            part = make(d)
            if d in ("HT3", "HT6"):
                part.fit(Xtr[ok & split], u[ok & split])
                est_v, est_p = ok & ~split, None
            elif d in ("HT3r", "HT6r"):
                part.fit(Xtr[split], rz[tr][split])
                est_v, est_p = None, ~split
            else:
                part.fit(Xtr)
                est_v, est_p = ok, np.ones(len(tr), bool)
            ctr, cte = part.assign(Xtr), part.assign(Xte)
            r = {"n_cells": part.n_cells, "used_cells": int(len(np.unique(ctr)))}
            if d in VOL_DISCS:
                ce = dz.cell_effects(u[est_v], ctr[est_v], days[tr][est_v], part.n_cells)
                F[d][te] = Fall[te] * np.exp(ce.value(cte))
                r |= {"kappa_vol": ce.kappa, "tau2_vol": ce.tau2, "sd_shrunk_vol": float(np.sqrt(np.average(ce.shrunk ** 2, weights=np.maximum(ce.n, 1e-9))))}
            if d in DIR_DISCS:
                cp = dz.cell_effects(yd[tr][est_p], ctr[est_p], days[tr][est_p], part.n_cells, center=b)
                P[d][te] = np.clip(b + cp.value(cte), 0.001, 0.999)
                r |= {"kappa_dir": cp.kappa, "tau2_dir": cp.tau2}
            rec["discs"][d] = r
            if m == frozen_m:
                cells_all = part.assign(X)
                desc = {}
                for k in range(part.n_cells):
                    sel = ctr == k
                    desc[k] = {"words": describe(d, part, k, Xtr[sel], Xtr), "profile": profile(Xtr[sel]), "n_train": int(sel.sum())}
                fz = {"month": str(m)[:7], "spec": part.spec(), "cells": desc}
                if d in VOL_DISCS:
                    fz["train_vol"] = ce.to_dict()
                if d in DIR_DISCS:
                    fz["train_dir"] = cp.to_dict()
                frozen[d] = fz
                np.save(out_dir() / f"frozen_cells_{d}.npy", cells_all)
        rec["secs"] = round(time.time() - t0, 1)
        info.append(rec)
        print(json.dumps({k: rec[k] for k in ("month", "n_train", "n_test", "secs")}), flush=True)
    ref = wf["F_har"].to_numpy()
    both = np.isfinite(ref) & np.isfinite(F_har)
    maxrel = float(np.max(np.abs(F_har[both] / ref[both] - 1))) if both.any() else float("nan")
    print("HAR replication max rel diff", maxrel, "rows", int(both.sum()), "of", int(np.isfinite(ref).sum()), flush=True)
    out = wf.select("symbol", "date", "T", "segment", "rv_next", "lrv_next", "y_dir", "ret_next_z", "v0_vol", "next_is_weekend",
                    "F_har", "F_har_qs", "F_mlp", "F_xgb", "F_lin", "F_har_dow").with_columns(
        pl.Series("F_har_rep", F_har), pl.Series("base", base), *[pl.Series(f"F_{d}", v) for d, v in F.items()],
        *[pl.Series(f"p_{d}", v) for d, v in P.items()])
    out.write_parquet(out_dir() / "wf_ds.parquet")
    (out_dir() / "wf_ds_info.json").write_text(json.dumps({"first_test_month": str(m0)[:7], "frozen_month": str(frozen_m)[:7] if frozen_m is not None else None,
                                                           "har_replication_max_rel_diff": maxrel, "months": info}, indent=1, default=float), encoding="utf-8")
    (out_dir() / "frozen_partitions.json").write_text(json.dumps(frozen, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------------ evaluation helpers


def _phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2)) if np.isfinite(t) else 1.0


def _date_series(df: pl.DataFrame, a_col: str, b_col: str) -> tuple[np.ndarray, np.ndarray]:
    from ai_trader.ml.models import qlike

    la = qlike(df["rv_next"].to_numpy(), df[a_col].to_numpy())
    lb = qlike(df["rv_next"].to_numpy(), df[b_col].to_numpy())
    g = pl.DataFrame({"t": df["T"].to_numpy(), "d": la - lb}).group_by("t").agg(pl.col("d").mean()).sort("t")
    return g["t"].to_numpy(), g["d"].to_numpy()


def share_ci(df: pl.DataFrame, col: str) -> dict:
    _, dd = _date_series(df, col, "F_har")
    _, dx = _date_series(df, "F_xgb", "F_har")
    idx = block_bootstrap_idx(len(dd), 10, N_BOOT, 0)
    with np.errstate(all="ignore"):
        bs = dd[idx].mean(1) / dx[idx].mean(1)
    pt = float(dd.mean() / dx.mean())
    return {"share": pt, "ci95": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))],
            "p_lt_target": float((1 + np.sum(bs < SHARE_TARGET)) / (len(bs) + 1)), "xgb_diff_t_sign": float(np.sign(dx.mean()))}


def _codes(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(T.astype("datetime64[D]"), return_inverse=True)
    return u, inv


def seg_cell_values(values: np.ndarray, cells: np.ndarray, T: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    _, codes = _codes(T)
    return dz.date_cell_sums(values, cells, codes, n_cells)


def spearman_cells(SA, NA, SB, NB, min_n: int = MIN_CELL) -> tuple[float, int, np.ndarray, np.ndarray]:
    from ai_trader.ml.interpret import spearman

    ea, eb = dz.eb_shrink(SA, NA), dz.eb_shrink(SB, NB)
    keep = (ea.n >= min_n) & (eb.n >= min_n)
    if keep.sum() < 5:
        rho = float("nan")
    elif not (np.any(ea.shrunk[keep] != 0) and np.any(eb.shrunk[keep] != 0)):
        rho = 0.0  # a segment with tau2 = 0 has every cell shrunk to 0: no ordering (defined as rho = 0)
    else:
        rho = spearman(ea.shrunk[keep], eb.shrunk[keep], min_n=5)
    return rho, int(keep.sum()), ea.shrunk, eb.shrunk


def ordering_test(values: np.ndarray, cells: np.ndarray, T: np.ndarray, seg: np.ndarray, n_cells: int) -> dict:
    """Spearman of segment cell values (EB-shrunk, own kappa) with a date-block bootstrap within each segment."""
    ia, ib = seg == SEGS[0], seg == SEGS[1]
    SA, NA = seg_cell_values(values[ia], cells[ia], T[ia], n_cells)
    SB, NB = seg_cell_values(values[ib], cells[ib], T[ib], n_cells)
    rho, k, va, vb = spearman_cells(SA, NA, SB, NB)
    ixa = block_bootstrap_idx(SA.shape[0], 10, N_BOOT, 0)
    ixb = block_bootstrap_idx(SB.shape[0], 10, N_BOOT, 1)
    bs = np.array([spearman_cells(SA[ixa[i]], NA[ixa[i]], SB[ixb[i]], NB[ixb[i]])[0] for i in range(N_BOOT)])
    bs = bs[np.isfinite(bs)]
    fz = [float(np.tanh(np.arctanh(rho) + z / math.sqrt(k - 3))) for z in (-1.96, 1.96)] if k > 3 and np.isfinite(rho) and abs(rho) < 1 else [float("nan")] * 2
    return {"rho": rho, "n_cells_both": k, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if len(bs) else [float("nan")] * 2,
            "ci95_fisher": fz, "boot_median": float(np.median(bs)) if len(bs) else float("nan"),
            "boot": bs, "values_in": va, "values_ho": vb, "ea": dz.eb_shrink(SA, NA), "eb": dz.eb_shrink(SB, NB)}


# ------------------------------------------------------------------ evaluate


def run_evaluate(record: bool = False) -> dict:
    from ai_trader.ml.interpret import spearman

    w = pl.read_parquet(out_dir() / "wf_ds.parquet")
    frozen = json.loads((out_dir() / "frozen_partitions.json").read_text(encoding="utf-8"))
    info = json.loads((out_dir() / "wf_ds_info.json").read_text(encoding="utf-8"))
    need = ["F_har", "F_har_qs", "F_mlp", "F_xgb", "F_lin", *[f"F_{d}" for d in VOL_DISCS]]
    df = w.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in need]))
    res: dict = {"n_common": df.height, "har_replication_max_rel_diff": info["har_replication_max_rel_diff"],
                 "frozen_month": info["frozen_month"], "first_test_month": info["first_test_month"]}

    # ---------------- H-DS1
    ds1: dict = {}
    for sg in SEGS:
        s = df.filter(pl.col("segment") == sg)
        r = {"xgb_vs_har": paired(s, "F_xgb", "F_har"), "mlp_vs_har": paired(s, "F_mlp", "F_har"), "har_dow_vs_har": paired(s.filter(pl.col("F_har_dow").is_finite()), "F_har_dow", "F_har")}
        for d in VOL_DISCS:
            p = paired(s, f"F_{d}", "F_har")
            p["share_of_xgb"] = p["diff"] / r["xgb_vs_har"]["diff"]
            r[d] = p
        ds1[sg] = r
    best = min(VOL_DISCS, key=lambda d: ds1[SEGS[0]][d]["loss_model"])
    ds1["best"] = best
    for sg in SEGS:
        s = df.filter(pl.col("segment") == sg)
        r = ds1[sg]
        r["best_share"] = share_ci(s, f"F_{best}")
        r["best_vs_xgb"] = paired(s, f"F_{best}", "F_xgb")
        hd = s.filter(pl.col("F_har_dow").is_finite())
        r["best_vs_har_dow"] = paired(hd, f"F_{best}", "F_har_dow")
        r["by_weekend"] = {str(k): {"best": paired(s.filter(pl.col("next_is_weekend") == k), f"F_{best}", "F_har"),
                                    "xgb": paired(s.filter(pl.col("next_is_weekend") == k), "F_xgb", "F_har")} for k in (0.0, 1.0)}
        b, sh = r[best], r["best_share"]
        if sh["share"] >= SHARE_TARGET and b["diff"] < 0 and b["t"] < -2:
            v = "supports"
        elif b["diff"] >= 0 or sh["ci95"][1] < SHARE_TARGET:
            v = "contradicts"
        else:
            v = "inconclusive"
        r["verdict"] = v
        r["p"] = float(max(_phi(b["t"]), sh["p_lt_target"]))
        # which discretisations would pass in this segment (descriptive)
        r["passing"] = [d for d in VOL_DISCS if r[d]["share_of_xgb"] >= SHARE_TARGET and r[d]["diff"] < 0 and r[d]["t"] < -2]
    vs = [ds1[s]["verdict"] for s in SEGS]
    ds1["pooled_verdict"] = "supports" if vs == ["supports"] * 2 else ("contradicts" if "contradicts" in vs else "inconclusive")
    ds1["p"] = max(ds1[s]["p"] for s in SEGS)
    res["H-DS1"] = ds1

    # ---------------- H-DS2 (frozen best partition, OOS HAR residual)
    T = w["T"].to_numpy()
    seg = w["segment"].to_numpy()
    fh = w["F_har"].to_numpy()
    rvn = w["rv_next"].to_numpy()
    okh = np.isfinite(fh)
    u_oos = np.where(okh, np.log(rvn / np.where(okh, fh, 1.0)), np.nan)

    def frozen_cells(d: str) -> np.ndarray:
        return np.load(out_dir() / f"frozen_cells_{d}.npy")

    def ncells(d: str) -> int:
        return len(frozen[d]["cells"])

    cb = frozen_cells(best)
    ot = ordering_test(u_oos[okh], cb[okh], T[okh], seg[okh], ncells(best))
    rho, lo, hi = ot["rho"], ot["ci95"][0], ot["ci95"][1]
    if rho > 0.8 and lo > 0.6:
        v2 = "supports"
    elif rho <= 0.6 or hi <= 0.8:
        v2 = "contradicts"
    else:
        v2 = "inconclusive"
    ds2 = {"discretisation": best, "rho": rho, "ci95": ot["ci95"], "ci95_fisher": ot["ci95_fisher"], "boot_median": ot["boot_median"],
           "n_cells_both": ot["n_cells_both"], "n_cells": ncells(best),
           "p": float((1 + np.sum(ot["boot"] <= 0.8)) / (len(ot["boot"]) + 1)), "verdict": v2,
           "kappa_in": ot["ea"].kappa, "kappa_ho": ot["eb"].kappa,
           "sd_cells_in": float(np.std(ot["values_in"][(ot["ea"].n >= MIN_CELL) & (ot["eb"].n >= MIN_CELL)])),
           "sd_cells_ho": float(np.std(ot["values_ho"][(ot["ea"].n >= MIN_CELL) & (ot["eb"].n >= MIN_CELL)]))}
    # all discretisations (descriptive)
    ds2["all"] = {}
    for d in VOL_DISCS:
        o = ordering_test(u_oos[okh], frozen_cells(d)[okh], T[okh], seg[okh], ncells(d))
        ds2["all"][d] = {"rho": o["rho"], "ci95": o["ci95"], "ci95_fisher": o["ci95_fisher"], "boot_median": o["boot_median"], "n_cells_both": o["n_cells_both"]}
    res["H-DS2"] = ds2
    res["n_cells"] = {d: ncells(d) for d in ALL_DISCS}

    # ---------------- cell table of the best vol discretisation
    ea, eb = ot["ea"], ot["eb"]
    tv = frozen[best]["train_vol"]
    rows = []
    for k in range(ncells(best)):
        z_in = ea.mean[k] / math.sqrt(ea.var[k]) if ea.n[k] > 1 and ea.var[k] > 0 else float("nan")
        z_ho = eb.mean[k] / math.sqrt(eb.var[k]) if eb.n[k] > 1 and eb.var[k] > 0 else float("nan")
        rows.append({"cell": k, "words": frozen[best]["cells"][str(k)]["words"], "profile": frozen[best]["cells"][str(k)]["profile"],
                     "n_train_frozen": frozen[best]["cells"][str(k)]["n_train"], "train_effect": tv["shrunk"][k],
                     "n_in": int(ea.n[k]), "n_dates_in": int(ea.n_dates[k]), "effect_in": float(ea.shrunk[k]), "raw_in": float(np.nan_to_num(ea.mean[k])), "z_in": z_in,
                     "n_ho": int(eb.n[k]), "n_dates_ho": int(eb.n_dates[k]), "effect_ho": float(eb.shrunk[k]), "raw_ho": float(np.nan_to_num(eb.mean[k])), "z_ho": z_ho})
    rows.sort(key=lambda r: r["effect_in"])
    res["cell_table"] = {"discretisation": best, "rows": rows, "kappa_in": ea.kappa, "kappa_ho": eb.kappa, "kappa_train": tv["kappa"]}

    # ---------------- POST HOC (after the cell table was seen; descriptive, for rule wording only): merged LAT1 cells
    if best == "LAT1":
        wdg, vt, dr = cb // 9, (cb // 3) % 3, cb % 3  # weekday group Mon/Tue-Fri/Sat/Sun, vol tercile, v0 direction down/range/up
        groups = {
            "Saturday, v0 range or trend down": (wdg == 2) & (dr != 2), "Saturday, v0 trend up": (wdg == 2) & (dr == 2),
            "Sunday": wdg == 3, "Monday": wdg == 0, "Monday, vol in the top third and v0 trend up": (wdg == 0) & (vt == 2) & (dr == 2),
            "Tue-Fri, v0 trend up": (wdg == 1) & (dr == 2), "Tue-Fri, v0 range": (wdg == 1) & (dr == 1), "Tue-Fri, v0 trend down": (wdg == 1) & (dr == 0),
        }
        gres = {}
        for g, mk in groups.items():
            gres[g] = {}
            for sg in SEGS:
                ss = okh & (seg == sg)
                mu = float(np.mean(u_oos[ss]))
                sel = ss & mk
                e = u_oos[sel] - mu
                _, cd = _codes(T[sel])
                sd = np.bincount(cd, e - e.mean())
                se = math.sqrt(float((sd ** 2).sum())) / len(e) if len(e) > 1 else float("nan")
                gres[g][sg] = {"n": int(sel.sum()), "n_dates": int(cd.max() + 1) if len(cd) else 0, "effect": float(e.mean()), "z": float(e.mean() / se) if se and se > 0 else float("nan")}
        res["groups_post_hoc"] = gres

    # ---------------- H-DS3
    dfp = w.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in ["base", *[f"p_{d}" for d in DIR_DISCS]]]))
    ds3: dict = {"brier": {}}
    for sg in SEGS:
        s = dfp.filter(pl.col("segment") == sg)
        ds3["brier"][sg] = {d: brier_block(s, f"p_{d}") for d in DIR_DISCS}
    bestd = max(DIR_DISCS, key=lambda d: ds3["brier"][SEGS[0]][d]["skill"])
    ds3["best"] = bestd
    rz = np.clip(w["ret_next_z"].to_numpy().astype(float), -RET_CLIP, RET_CLIP)
    okr = np.isfinite(rz)
    o3 = ordering_test(rz[okr], frozen_cells(bestd)[okr], T[okr], seg[okr], ncells(bestd))
    ds3["ordering"] = {"rho": o3["rho"], "ci95": o3["ci95"], "ci95_fisher": o3["ci95_fisher"], "boot_median": o3["boot_median"], "n_cells_both": o3["n_cells_both"], "p_gt_03": float((1 + np.sum(o3["boot"] > 0.3)) / (len(o3["boot"]) + 1))}
    ds3["ordering_all"] = {}
    for d in DIR_DISCS:
        o = ordering_test(rz[okr], frozen_cells(d)[okr], T[okr], seg[okr], ncells(d))
        ds3["ordering_all"][d] = {"rho": o["rho"], "ci95": o["ci95"], "ci95_fisher": o["ci95_fisher"], "boot_median": o["boot_median"], "n_cells_both": o["n_cells_both"]}
    for sg in SEGS:
        bb = ds3["brier"][sg][bestd]
        ds3[sg] = {"skill": bb["skill"], "ci95": bb["ci95"], "auc": bb["auc"],
                   "verdict": "supports" if bb["skill"] <= 0 else ("contradicts" if bb["ci95"][0] > 0 else "inconclusive")}
    # bootstrap P*(holdout skill > 0) for the p-value
    s_ho = dfp.filter(pl.col("segment") == SEGS[1])
    p_skill = _p_skill_gt0(s_ho, f"p_{bestd}")
    ho_sk = ds3[SEGS[1]]
    if o3["rho"] <= 0.3 and ho_sk["skill"] <= 0:
        v3 = "supports"
    elif o3["ci95"][0] > 0.3 or ho_sk["ci95"][0] > 0:
        v3 = "contradicts"
    else:
        v3 = "inconclusive"
    ds3["pooled_verdict"] = v3
    ds3["p"] = float(max(ds3["ordering"]["p_gt_03"], p_skill))
    ds3["p_skill_gt0"] = p_skill
    res["H-DS3"] = ds3

    # ---------------- Holm within DS
    hp = holm({"H-DS1": ds1["p"], "H-DS2": ds2["p"], "H-DS3": ds3["p"]})
    res["holm"] = hp
    ver = {"H-DS1": {s: ds1[s]["verdict"] for s in SEGS} | {"pooled": ds1["pooled_verdict"]},
           "H-DS2": {"pooled": v2},
           "H-DS3": {s: ds3[s]["verdict"] for s in SEGS} | {"pooled": v3}}
    for h, vv in ver.items():
        for k in vv:
            if vv[k] == "supports" and hp[h] > 0.05:
                vv[k] = "inconclusive"
    res["verdicts"] = ver
    (out_dir() / "evaluate.json").write_text(json.dumps(_clean(res), indent=1, default=float), encoding="utf-8")
    if record:
        _record(res)
    return res


def _p_skill_gt0(s: pl.DataFrame, pcol: str) -> float:
    y = s["y_dir"].to_numpy().astype(float)
    p = s[pcol].to_numpy().astype(float)
    b = s["base"].to_numpy().astype(float)
    _, codes = _codes(s["T"].to_numpy())
    nd = int(codes.max()) + 1
    sq = np.bincount(codes, (p - y) ** 2, nd)
    ref = np.bincount(codes, (b - y) ** 2, nd)
    idx = block_bootstrap_idx(nd, 10, N_BOOT, 0)
    boot = 1 - sq[idx].sum(1) / ref[idx].sum(1)
    return float((1 + np.sum(boot > 0)) / (len(boot) + 1))


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not isinstance(v, (dz.CellEffects,))}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _record(res: dict) -> None:
    from ai_trader.research import hypotheses as hy

    data = f"universe L daily panel (ai_trader.ml.features, 52 features, complete rows), walk-forward {res['first_test_month']}..2026-11, monthly expanding refits; report research/reports/discrete_states_{TAG}.md"
    ds1, ds2, ds3, ver, hp = res["H-DS1"], res["H-DS2"], res["H-DS3"], res["verdicts"], res["holm"]
    best = ds1["best"]
    done = {(r["id"], r["segment"]) for h in ("H-DS1", "H-DS2", "H-DS3") for r in hy.results(h)}

    def rec(h, sg, v, m, note):
        if (h, sg) in done:
            print("already recorded", h, sg)
            return
        hy.record(h, sg, v, _clean(m), note=note, data=data)

    for sg in SEGS:
        r = ds1[sg]
        b = r[best]
        m = {"best": best, "diff": b["diff"], "rel": b["rel"], "t": b["t"], "n_dates": b["n_dates"], "xgb_diff": r["xgb_vs_har"]["diff"],
             "xgb_t": r["xgb_vs_har"]["t"], "share": r["best_share"]["share"], "share_ci95": r["best_share"]["ci95"], "holm_p": hp["H-DS1"],
             "passing": r["passing"], "all_rel": {d: r[d]["rel"] for d in VOL_DISCS}, "all_share": {d: r[d]["share_of_xgb"] for d in VOL_DISCS}}
        rec("H-DS1", sg, ver["H-DS1"][sg], m, f"best discretisation {best} chosen on in-sample QLIKE; share of the XGB gain {r['best_share']['share']:.2f} "
            f"[{r['best_share']['ci95'][0]:.2f}, {r['best_share']['ci95'][1]:.2f}], own diff {b['diff']:+.4f} (t {b['t']:.2f})")
    rec("H-DS1", "pooled", ver["H-DS1"]["pooled"], {"best": best, "segments": [ver["H-DS1"][s] for s in SEGS], "holm_p": hp["H-DS1"]},
        "registry criterion: >= 80% of the XGB gain with t < -2 in both segments")
    rec("H-DS2", "pooled", ver["H-DS2"]["pooled"], {k: ds2[k] for k in ("discretisation", "rho", "ci95", "n_cells_both", "n_cells", "p", "kappa_in", "kappa_ho")} | {"holm_p": hp["H-DS2"]},
        f"frozen {best} partition from the {res['frozen_month']} refit; Spearman of EB-shrunk cell log(RV/HAR) between segments over cells with >= 30 rows in both")
    for sg in SEGS:
        rec("H-DS3", sg, ver["H-DS3"][sg], {"best_dir": ds3["best"], **ds3[sg], "holm_p": hp["H-DS3"]},
            f"cell-probability direction rule of {ds3['best']} (best in-sample Brier skill); {'holdout criterion' if sg == SEGS[1] else 'in-sample analogue (descriptive)'}")
    rec("H-DS3", "pooled", ver["H-DS3"]["pooled"], {"best_dir": ds3["best"], "ordering": ds3["ordering"], "holdout_skill": ds3[SEGS[1]]["skill"],
                                                     "holdout_ci95": ds3[SEGS[1]]["ci95"], "p": ds3["p"], "holm_p": hp["H-DS3"],
                                                     "ordering_all_rho": {d: v["rho"] for d, v in ds3["ordering_all"].items()}},
        "null: cell ordering by next-day vol-scaled return Spearman <= 0.3 between segments and holdout Brier skill <= 0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["walk", "evaluate", "report"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.cmd == "walk":
        run_walk(a.limit)
    elif a.cmd == "evaluate":
        res = run_evaluate(record=a.record)
        from ai_trader.research.discrete_states_report import write

        write(res)
    else:
        from ai_trader.research.discrete_states_report import write

        write(json.loads((out_dir() / "evaluate.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
