"""Sparsity: the most important inputs, nodes and neurons of the daily vol forecaster (H-SP1..H-SP3, program 2026-11-12,
agent D). Deterministic research tools only; the LLM decision layer is not trained or backtested here (principle 1).

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.sparsity_study prune [--workers 4]
    ... select        H-SP2 stability selection (checkpoint per segment)
    ... units         H-SP3 surviving-unit analysis of the frozen pruned network
    ... evaluate [--record]   verdicts, Holm within SP, research/reports/sparsity_2026-11-12.{json,md}

Design (fixed before any evaluation; diary 2026-11-12 'Sparsity design fixed before evaluation'):

H-SP1 (lottery ticket). Same panel, rows, months, purge, validation block, standardisation, config (vol MLP (32, 16), wd
0.01; direction MLP (64, 32), wd 1e-4), 3 seeds and validation-block bias correction as H-NN1 (ml_signals_study.run_walk),
full monthly walk-forward 2022-04..2026-11 (56 refits). The dense network is ml.sparse.train_masked without a mask, which
is bit-identical to models.fit_mlp (checked). Iterative magnitude pruning per seed (ml.sparse.imp_chain): 22 rounds,
layer-wise 20% of surviving hidden weights and 10% of the head per round, rewinding to the end of epoch 1. Overall
sparsity by round is deterministic (sparse.sparsity_schedule): 95.4% at round 14 (the H-SP1 test level), 98.4% at 19,
99.04% at 22. Ensemble forecast per round = exp(mean of the 3 members' log forecasts + c), c from the validation block.
QLIKE per row, averaged over symbols per date; ratio = mean over dates of sparse / mean over dates of dense, per segment;
95% CI from a moving-block bootstrap over dates (10-day blocks, 1000 draws, seed 0). Segment verdict at round 14:
supports iff ratio <= 1.01 and CI upper <= 1.02; contradicts iff ratio > 1.02 or CI lower > 1.01; else inconclusive.
Pooled: supports iff both segments support; contradicts iff either contradicts; else inconclusive. p (Holm) = bootstrap
P*(ratio >= 1.02) (non-inferiority; pooled = max of segments). Descriptive: every round, 98% / 99% levels, single seed-0
member, vs log-HAR (the H-NN1 F_har), by vol state and weekend, input survival (inputs with a surviving path) over
months and seeds. Direction: the same pruning of the H-NN2 direction MLP; Brier skill vs the training base rate per
round, sd of the probabilities, number of surviving units (descriptive only, no verdict).

H-SP2 (stability selection). Library (sparsity_study.library, >= 200 columns): the 52 ml.features inputs; signed logs
sign(x) log(1 + |x|) of the non-binary inputs that are not already logs; lags 1-30 of lrv_d, lags 1-7 of abs_ret_1d_z and
mkt_lrv_d, lags 1-5 of ret_1d_z (same symbol, exact calendar lag); pairwise products of the standardised top-10 inputs of
H-NN3 (TreeSHAP main effects, in-sample: log_sigma, lrv_w, lrv_q, next_is_weekend, dow_sin, lrv_m, next_wk_hours, lrv_d,
rq_ratio_d, v0_dir_score); upper-quantile indicators 1{x > q20/q40/q60/q80} of 12 continuous inputs (edges from
llm_in_sample rows); 7 weekday dummies of D+1. Target lrv_next. Per segment (its own rows): 200 subsamples of half of the
dates, randomised lasso with penalty weights 1/W, W ~ U[0.5, 1], 40-value lambda grid (lambda_max of the segment to
0.005 lambda_max), coordinate descent on the Gram matrix (a subsample's path stops once its union exceeds 4 q_max; its
later lambdas count as union p, which can only shrink Lambda); Lambda = lambdas down to the last one with mean union size
<= q_max = sqrt((2 pi - 1) p) (E[V] <= 1), pi_thr = 0.8. Segment: supports iff 1 <= |S| <= 10; contradicts iff |S| > 10
or |S| = 0. Overlap = Jaccard |A n B| / |A u B|. Pooled: supports iff both segments support and Jaccard >= 0.7;
contradicts iff either segment contradicts or Jaccard < 0.7. p (Holm, every row) = hypergeometric probability of an
intersection at least as large as observed for random sets of the observed sizes (per-segment sizes are design-controlled
counts with the PFER bound reported).

H-SP3 (surviving neurons). Network = the seed-0 member of the 2026-07 refit (labels to 2026-06-30, as H-NN4) at the highest
of the 95% / 98% / 99% rounds whose H-SP1 test passes in both segments (95% if none). Surviving units = hidden units
reachable from an input and reaching the output through surviving weights (ml.sparse.surviving_units). Unit activation
fits: additive cubic splines (ml.sparse.gam_search) on the best single input and the best additive pair, chosen by
fitting-sample R^2; the unit's R^2 = evaluation-sample R^2 of the best pair. clean_holdout: fit on llm_in_sample rows, R^2
on clean_holdout rows; llm_in_sample (descriptive): fit on the first 80% of in-sample dates, R^2 on the last 20%. Units
with zero activation variance on the evaluation rows are not evaluable (counted, excluded from the share). Attribution =
mean |activation x d output / d activation| on llm_in_sample rows; top-3 units mean-ablated (in-sample mean activation);
QLIKE ratio ablated / intact (member's own validation bias), date-averaged, block bootstrap CI. Segment: supports iff
>= 80% of evaluable units have R^2 > 0.8 AND ablation ratio > 1.05 with CI lower > 1.00; contradicts iff < 80% meet the
R^2 condition or ratio <= 1.05; else inconclusive. Pooled = clean_holdout. p (Holm) = P*(ratio <= 1.05) if the R^2
condition holds, else 1.

Holm within SP per segment over H-SP1..3; a 'supports' with Holm p > 0.05 is downgraded to inconclusive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import models as md
from ai_trader.ml import sparse as sp
from ai_trader.ml.features import ML_FEATURES, Standardizer
from ai_trader.research.decision_budget_study import block_bootstrap_idx, holm, purged_train_mask, test_mask

TAG = "2026-11-12"
SEGS = ("llm_in_sample", "clean_holdout")
N_SEEDS = 3
ROUNDS = sp.N_ROUNDS
TARGETS = (0.95, 0.98, 0.99)
F = len(ML_FEATURES)


def root() -> Path:
    d = load().root / "data" / "models" / "ml" / "sparse"
    (d / "wf").mkdir(parents=True, exist_ok=True)
    return d


def schedules() -> dict:
    from ai_trader.research import ml_signals_study as ms

    cf = ms._cfgs()
    out = {}
    for task, key in (("vol", "mlp_vol"), ("dir", "mlp_dir")):
        h = tuple(cf[key]["hidden"])
        s = sp.sparsity_schedule(F, h, ROUNDS)
        out[task] = {"hidden": list(h), "schedule": s, "rounds_at": {str(t): sp.round_at(s, t) for t in TARGETS}}
    return out


# ------------------------------------------------------------------ H-SP1 walk-forward pruning

_W: dict = {}


def _init_worker() -> None:
    import torch

    md.TORCH_THREADS = 1
    torch.set_num_threads(1)


def _data():
    if "a" not in _W:
        from ai_trader.research import ml_signals_study as ms

        _, c = ms.load_panel()
        _W["a"] = ms.arrays(c)
        _W["cf"] = ms._cfgs()
    return _W["a"], _W["cf"]


def month_job(m_iso: str) -> dict:
    """One refit month: dense + 22 pruning rounds x 3 seeds for vol and direction; written atomically."""
    import torch

    a, cf = _data()
    m = np.datetime64(m_iso, "us")
    tag = m_iso[:7]
    out_npz = root() / "wf" / f"{tag}.npz"
    if out_npz.is_file():
        return {"month": tag, "skipped": True}
    t0 = time.time()
    m1 = np.datetime64(m.astype("datetime64[M]") + np.timedelta64(1, "M"), "us")
    te = np.flatnonzero(test_mask(a["T"], m, m1))
    tr = np.flatnonzero(purged_train_mask(a["le"], m))
    fit_l, va_l = md.val_split(a["T"][tr])
    fi, vi = tr[fit_l], tr[va_l]
    st = Standardizer().fit(a["X"][fi])
    Zf, Zv, Zt = st.transform(a["X"][fi]), st.transform(a["X"][vi]), st.transform(a["X"][te])
    R = ROUNDS + 1
    res: dict = {"te": te, "mu": st.mu, "sd": st.sd, "base": np.array(a["yd"][tr].mean())}
    states = {}
    for task, y, key in (("vol", a["ly"], "mlp_vol"), ("dir", a["yd"], "mlp_dir")):
        lt = np.zeros((N_SEEDS, R, len(te)))
        lv = np.zeros((N_SEEDS, R, len(vi)))
        ep = np.zeros((N_SEEDS, R), dtype=int)
        bv = np.zeros((N_SEEDS, R))
        paths = np.zeros((N_SEEDS, R, F), dtype=bool)
        nunits = np.zeros((N_SEEDS, R), dtype=int)
        okm = np.ones((N_SEEDS, R), dtype=bool)
        for s in range(N_SEEDS):
            ch = sp.imp_chain(Zf, y[fi], Zv, y[vi], task, s, cf[key], ROUNDS)
            for r, (fit, mk) in enumerate(zip(ch.fits, ch.masks, strict=True)):
                lt[s, r] = fit.predict(Zt)
                lv[s, r] = fit.predict(Zv)
                ep[s, r] = fit.epochs
                bv[s, r] = fit.best_val
                paths[s, r] = sp.input_paths(mk)
                nunits[s, r] = len(sp.surviving_units(fit, mk))
                okm[s, r] = sp.check_mask(fit, mk)
            if task == "vol":
                states[s] = [{"state": f.state, "masks": mk, "y_mu": f.y_mu, "y_sd": f.y_sd, "epochs": f.epochs} for f, mk in zip(ch.fits, ch.masks, strict=True)]
        if task == "vol":
            c_ens = np.array([md.bias_const(a["rv"][vi], lv[:, r].mean(0)) for r in range(R)])
            c_mem = np.array([[md.bias_const(a["rv"][vi], lv[s, r]) for r in range(R)] for s in range(N_SEEDS)])
            res |= {"vol_lf": lt.astype(np.float32), "vol_c_ens": c_ens, "vol_c_mem": c_mem}
        else:
            res |= {"dir_p": lt.astype(np.float32)}
        res |= {f"{task}_epochs": ep, f"{task}_best_val": bv, f"{task}_paths": paths, f"{task}_units": nunits, f"{task}_mask_ok": okm}
    torch.save({"month": tag, "hidden": list(cf["mlp_vol"]["hidden"]), "mu": st.mu, "sd": st.sd, "seeds": states},
               root() / "wf" / f"{tag}_vol_states.pt")
    tmp = root() / "wf" / f"{tag}.tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, out_npz)
    return {"month": tag, "n_test": len(te), "n_train": len(tr), "secs": round(time.time() - t0, 1),
            "vol_epochs_r0_r14_r22": [int(res["vol_epochs"][0, 0]), int(res["vol_epochs"][0, 14]), int(res["vol_epochs"][0, ROUNDS])]}


def months_list() -> list[str]:
    from ai_trader.research import ml_signals_study as ms

    a, _ = _data()
    m0 = ms.first_test_month(a["T"], a["le"])
    return [str(m) for m in ms._months(a["T"]) if m >= m0]


def run_prune(workers: int = 4) -> None:
    ms_ = months_list()
    a, _ = _data()
    todo = [m for m in ms_ if not (root() / "wf" / f"{m[:7]}.npz").is_file()]
    # largest training windows first (load balance); each month is its own checkpoint
    todo.sort(key=lambda m: -int(purged_train_mask(a["le"], np.datetime64(m, "us")).sum()))
    (root() / "schedule.json").write_text(json.dumps(schedules(), indent=1), encoding="utf-8")
    print(f"{len(ms_)} months, {len(todo)} to do", flush=True)
    log = root() / "prune_log.jsonl"
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = {ex.submit(month_job, m): m for m in todo}
        for f in as_completed(futs):
            r = f.result()
            print(json.dumps(r), flush=True)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")


def load_wf() -> tuple[pl.DataFrame, dict]:
    """Assemble per-row forecasts for every round from the month checkpoints."""
    from ai_trader.research import ml_signals_study as ms

    _, c = ms.load_panel()
    n = c.height
    R = ROUNDS + 1
    lf = np.full((N_SEEDS, R, n), np.nan, dtype=np.float32)
    Fens = np.full((R, n), np.nan)
    Fm0 = np.full((R, n), np.nan)
    pd_ = np.full((N_SEEDS, R, n), np.nan, dtype=np.float32)
    base = np.full(n, np.nan)
    meta = {"months": [], "vol_paths": [], "dir_paths": [], "vol_units": [], "dir_units": [], "vol_epochs": [], "mask_ok": True}
    for m in months_list():
        p = root() / "wf" / f"{m[:7]}.npz"
        if not p.is_file():
            continue
        z = np.load(p)
        te = z["te"]
        lf[:, :, te] = z["vol_lf"]
        Fens[:, te] = np.exp(z["vol_lf"].mean(0) + z["vol_c_ens"][:, None])
        Fm0[:, te] = np.exp(z["vol_lf"][0] + z["vol_c_mem"][0][:, None])
        pd_[:, :, te] = z["dir_p"]
        base[te] = float(z["base"])
        meta["months"].append(m[:7])
        for k in ("vol_paths", "dir_paths", "vol_units", "dir_units", "vol_epochs"):
            meta[k].append(z[k])
        meta["mask_ok"] &= bool(z["vol_mask_ok"].all() and z["dir_mask_ok"].all())
    df = c.select("symbol", "date", "T", "segment", "rv_next", "lrv_next", "y_dir", "v0_vol", "next_is_weekend")
    df = df.with_columns(pl.Series("base", base), *[pl.Series(f"F_r{r}", Fens[r]) for r in range(R)],
                         *[pl.Series(f"Fm0_r{r}", Fm0[r]) for r in range(R)],
                         *[pl.Series(f"p_r{r}", pd_[:, r].mean(0).astype(float)) for r in range(R)])
    wf = pl.read_parquet(ms.root() / "wf_preds.parquet").select("symbol", "T", "F_har", "F_mlp")
    df = df.join(wf, on=["symbol", "T"], how="left")
    return df.filter(pl.col("F_r0").is_not_null() & pl.col("F_r0").is_finite()), meta


# ------------------------------------------------------------------ helpers: date-averaged QLIKE and bootstrap


def _date_losses(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rv = df["rv_next"].to_numpy()
    q = {c: md.qlike(rv, df[c].to_numpy()) for c in cols}
    ok = np.all([np.isfinite(v) for v in q.values()], axis=0)
    g = pl.DataFrame({"t": df["T"].to_numpy()[ok], **{c: v[ok] for c, v in q.items()}}).group_by("t").agg(
        *[pl.col(c).mean() for c in cols]).sort("t")
    return g["t"].to_numpy(), {c: g[c].to_numpy() for c in cols}


def ratio_ci(num: np.ndarray, den: np.ndarray, block: int = 10, n_boot: int = 1000, seed: int = 0, thr: float | None = None) -> dict:
    idx = block_bootstrap_idx(len(num), block, n_boot, seed)
    boot = num[idx].mean(1) / den[idx].mean(1)
    out = {"ratio": float(num.mean() / den.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
           "n_dates": len(num), "loss_num": float(num.mean()), "loss_den": float(den.mean())}
    if thr is not None:
        out[f"p_ge_{thr}"] = float((1 + np.sum(boot >= thr)) / (n_boot + 1))
        out[f"p_le_{thr}"] = float((1 + np.sum(boot <= thr)) / (n_boot + 1))
    return out


def _nw(d: np.ndarray) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    m, t, n = newey_west_t(d)
    return {"diff": float(m), "t": float(t), "n_dates": int(n)}


def _sp1_verdict(r: dict) -> str:
    lo, hi = r["ci95"]
    if r["ratio"] <= 1.01 and hi <= 1.02:
        return "supports"
    if r["ratio"] > 1.02 or lo > 1.01:
        return "contradicts"
    return "inconclusive"


def eval_sp1(df: pl.DataFrame, meta: dict, sched: dict) -> dict:
    R = ROUNDS + 1
    rv_at = sched["vol"]["rounds_at"]
    r95, r98, r99 = rv_at["0.95"], rv_at["0.98"], rv_at["0.99"]
    out: dict = {"rounds_at": rv_at, "sparsity": [s["sparsity"] for s in sched["vol"]["schedule"]], "n_alive": [s["n_alive"] for s in sched["vol"]["schedule"]],
                 "months": len(meta["months"]), "mask_ok": meta["mask_ok"], "segments": {}}
    for sg in SEGS:
        s = df.filter(pl.col("segment") == sg)
        cols = [f"F_r{r}" for r in range(R)] + [f"Fm0_r{r}" for r in range(R)] + ["F_har", "F_mlp"]
        _, L = _date_losses(s, cols)
        curve = []
        for r in range(R):
            rr = ratio_ci(L[f"F_r{r}"], L["F_r0"], thr=1.02)
            rr["vs_har"] = _nw(L[f"F_r{r}"] - L["F_har"])
            rr["vs_har"]["rel"] = float(L[f"F_r{r}"].mean() / L["F_har"].mean() - 1)
            rr["member0"] = ratio_ci(L[f"Fm0_r{r}"], L["Fm0_r0"])
            curve.append(rr)
        seg = {"curve": curve, "n_obs": s.height, "n_symbols": s["symbol"].n_unique(),
               "dense_vs_hnn1_stored": ratio_ci(L["F_r0"], L["F_mlp"]),
               "levels": {lab: {"round": r, "sparsity": out["sparsity"][r], **curve[r], "verdict": _sp1_verdict(curve[r])}
                          for lab, r in (("95", r95), ("98", r98), ("99", r99))}}
        # sub-segments at the test level
        subs = {}
        for name, flt in (("vol_low", pl.col("v0_vol") == 0), ("vol_normal", pl.col("v0_vol") == 1), ("vol_high", pl.col("v0_vol") == 2),
                          ("weekend_next", pl.col("next_is_weekend") == 1), ("weekday_next", pl.col("next_is_weekend") == 0)):
            q = s.filter(flt)
            if q.height < 100:
                continue
            _, Lq = _date_losses(q, ["F_r0", f"F_r{r95}", f"F_r{r99}"])
            subs[name] = {"n_obs": q.height, "r95": ratio_ci(Lq[f"F_r{r95}"], Lq["F_r0"]), "r99": ratio_ci(Lq[f"F_r{r99}"], Lq["F_r0"])}
        seg["subsegments"] = subs
        seg["verdict"] = seg["levels"]["95"]["verdict"]
        seg["p"] = curve[r95]["p_ge_1.02"]
        out["segments"][sg] = seg
    vs = [out["segments"][g]["verdict"] for g in SEGS]
    out["pooled_verdict"] = "supports" if all(v == "supports" for v in vs) else "contradicts" if "contradicts" in vs else "inconclusive"
    out["pooled_p"] = max(out["segments"][g]["p"] for g in SEGS)
    passing = [lab for lab in ("95", "98", "99") if all(out["segments"][g]["levels"][lab]["verdict"] == "supports" for g in SEGS)]
    out["passing_levels"] = passing
    out["sp3_level"] = passing[-1] if passing else "95"
    out["sp3_round"] = {"95": r95, "98": r98, "99": r99}[out["sp3_level"]]
    # input survival over months x seeds
    P = np.stack(meta["vol_paths"])  # (months, seeds, R, F)
    U = np.stack(meta["vol_units"])
    out["input_survival"] = {lab: {f: float(P[:, :, r, j].mean()) for j, f in enumerate(ML_FEATURES)} for lab, r in (("95", r95), ("98", r98), ("99", r99))}
    out["units"] = {lab: {"mean": float(U[:, :, r].mean()), "min": int(U[:, :, r].min()), "max": int(U[:, :, r].max())}
                    for lab, r in (("0", 0), ("95", r95), ("98", r98), ("99", r99))}
    out["inputs_alive"] = {lab: {"mean": float(P[:, :, r].sum(-1).mean()), "min": int(P[:, :, r].sum(-1).min()), "max": int(P[:, :, r].sum(-1).max())}
                           for lab, r in (("95", r95), ("98", r98), ("99", r99))}
    ep = np.stack(meta["vol_epochs"])
    out["epochs_mean"] = [float(ep[:, :, r].mean()) for r in range(R)]
    return out


def brier_ci(y: np.ndarray, p: np.ndarray, b: np.ndarray, T: np.ndarray) -> dict:
    _, inv = np.unique(T.astype("datetime64[D]"), return_inverse=True)
    nd = int(inv.max()) + 1
    sq = np.bincount(inv, (p - y) ** 2, nd)
    ref = np.bincount(inv, (b - y) ** 2, nd)
    idx = block_bootstrap_idx(nd, 10, 1000, 0)
    boot = 1 - sq[idx].sum(1) / ref[idx].sum(1)
    return {"skill": float(1 - sq.sum() / ref.sum()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "sd_p": float(np.std(p)), "n_obs": len(y)}


def eval_dir(df: pl.DataFrame, meta: dict, sched: dict) -> dict:
    R = ROUNDS + 1
    ra = sched["dir"]["rounds_at"]
    out: dict = {"rounds_at": ra, "sparsity": [s["sparsity"] for s in sched["dir"]["schedule"]], "segments": {}}
    for sg in SEGS:
        s = df.filter(pl.col("segment") == sg)
        y, b, T = s["y_dir"].to_numpy().astype(float), s["base"].to_numpy(), s["T"].to_numpy()
        out["segments"][sg] = [brier_ci(y, s[f"p_r{r}"].to_numpy(), b, T) for r in range(R)]
    P = np.stack(meta["dir_paths"])
    U = np.stack(meta["dir_units"])
    out["units"] = {str(r): {"mean": float(U[:, :, r].mean()), "share_zero": float((U[:, :, r] == 0).mean())} for r in (0, ra["0.95"], ra["0.98"], ra["0.99"])}
    out["input_survival_99"] = {f: float(P[:, :, ra["0.99"], j].mean()) for j, f in enumerate(ML_FEATURES)}
    out["input_survival_95"] = {f: float(P[:, :, ra["0.95"], j].mean()) for j, f in enumerate(ML_FEATURES)}
    return out


# ------------------------------------------------------------------ H-SP2 library and stability selection

TOP10 = ("log_sigma", "lrv_w", "lrv_q", "next_is_weekend", "dow_sin", "lrv_m", "next_wk_hours", "lrv_d", "rq_ratio_d", "v0_dir_score")
QFEAT = ("log_sigma", "lrv_w", "lrv_q", "lrv_m", "lrv_d", "rq_ratio_d", "v0_dir_score", "vol_of_vol_30d", "vol_pct_1y", "abs_ret_1d_z",
         "mkt_lrv_d", "funding_z_90d")
NOLOG = {"lrv_d", "lrv_w", "lrv_m", "lrv_q", "log_sigma", "mkt_lrv_d", "mkt_lrv_w", "rq_ratio_d", "pk_ratio_d", "gk_ratio_d",
         "next_is_weekend", "next_is_monday", "next_is_mexpiry", "v0_dir", "v0_vol", "dow_sin", "dow_cos"}
LAGS = {"lrv_d": 30, "abs_ret_1d_z": 7, "mkt_lrv_d": 7, "ret_1d_z": 5}


def library() -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    """Wide point-in-time library on complete rows; returns (frame with symbol/T/segment/lrv_next + columns, names, family)."""
    from ai_trader.research import ml_signals_study as ms

    full, c = ms.load_panel()
    fam: dict[str, str] = {f: "base" for f in ML_FEATURES}
    # lags from the full panel (exact calendar lag within symbol)
    base = full.select("symbol", "date", *LAGS.keys()).sort("symbol", "date")
    lag_cols = []
    for v, kmax in LAGS.items():
        for k in range(1, kmax + 1):
            nm = f"{v}_lag{k}"
            src = base.select("symbol", (pl.col("date") + pl.duration(days=k)).alias("date"), pl.col(v).alias(nm))
            c = c.join(src, on=["symbol", "date"], how="left")
            lag_cols.append(nm)
            fam[nm] = "lag"
    ins = (c["segment"] == "llm_in_sample").to_numpy()
    exprs = []
    for f in ML_FEATURES:
        if f in NOLOG:
            continue
        exprs.append((pl.col(f).sign() * pl.col(f).abs().log1p()).alias(f"slog_{f}"))
        fam[f"slog_{f}"] = "log"
    X = c.select(ML_FEATURES).to_numpy().astype(float)
    mu, sd = X[ins].mean(0), X[ins].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    zc = {f: (X[:, j] - mu[j]) / sd[j] for j, f in enumerate(ML_FEATURES)}
    prods = {}
    for i in range(len(TOP10)):
        for j in range(i + 1, len(TOP10)):
            nm = f"{TOP10[i]}*{TOP10[j]}"
            prods[nm] = zc[TOP10[i]] * zc[TOP10[j]]
            fam[nm] = "product"
    qd = {}
    for f in QFEAT:
        x = c[f].to_numpy().astype(float)
        for q in (20, 40, 60, 80):
            e = float(np.quantile(x[ins], q / 100))
            nm = f"{f}>q{q}"
            qd[nm] = (x > e).astype(float)
            fam[nm] = "quantile_dummy"
    wd = c.select((pl.col("date") + pl.duration(days=1)).dt.weekday().alias("w"))["w"].to_numpy()
    days = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    wdd = {f"wd_{days[k - 1]}": (wd == k).astype(float) for k in range(1, 8)}
    for k in wdd:
        fam[k] = "weekday"
    c = c.with_columns(*exprs, *[pl.Series(k, v) for k, v in {**prods, **qd, **wdd}.items()])
    names = [*ML_FEATURES, *[e.meta.output_name() for e in exprs], *lag_cols, *prods, *qd, *wdd]
    c = c.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in lag_cols]))
    return c.select("symbol", "T", "segment", "lrv_next", *names), names, fam


def run_select(n_sub: int = 200) -> None:
    lib, names, fam = library()
    p = len(names)
    q_max = sp.q_for_bound(p, 0.8, 1.0)
    for sg in SEGS:
        outp = root() / f"stability_{sg}.npz"
        if outp.is_file():
            print("exists", outp, flush=True)
            continue
        s = lib.filter(pl.col("segment") == sg)
        X = s.select(names).to_numpy().astype(float)
        y = s["lrv_next"].to_numpy().astype(float)
        T = s["T"].to_numpy().astype("datetime64[D]")
        lams = sp.lambda_grid(X, y, 40, 0.005)
        t0 = time.time()
        res = sp.stability_selection(X, y, T, n_sub=n_sub, alpha=0.5, lams=lams, seed=0, stop_union=int(4 * q_max),
                                     progress=lambda i, sg=sg, t0=t0: print(sg, i, round(time.time() - t0, 1), flush=True) if i % 20 == 0 else None)
        np.savez_compressed(outp, probs=res.probs, lams=res.lams, union_size=res.union_size, n_sub=res.n_sub, alpha=res.alpha,
                            n_rows=len(y), n_dates=len(np.unique(T)), q_max=q_max)
    (root() / "library.json").write_text(json.dumps({"names": names, "family": fam, "p": p, "q_max": q_max,
                                                    "rows": {sg: int((lib["segment"] == sg).sum()) for sg in SEGS}}, indent=1), encoding="utf-8")


def _hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k), X ~ Hypergeometric(N population, K successes, n draws)."""
    def lc(a, b):
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    tot = 0.0
    for x in range(k, min(K, n) + 1):
        tot += math.exp(lc(K, x) + lc(N - K, n - x) - lc(N, n))
    return float(min(1.0, tot))


def eval_sp2() -> dict:
    meta = json.loads((root() / "library.json").read_text(encoding="utf-8"))
    names, fam, p = meta["names"], meta["family"], meta["p"]
    out: dict = {"p": p, "family_counts": {}, "segments": {}}
    for k in set(fam.values()):
        out["family_counts"][k] = sum(1 for n in names if fam[n] == k)
    sets = {}
    for sg in SEGS:
        z = np.load(root() / f"stability_{sg}.npz")
        res = sp.StabilityResult(z["probs"], z["lams"], z["union_size"], int(z["n_sub"]), float(z["alpha"]))
        q_max = float(z["q_max"])
        mp = res.max_prob(q_max)
        sel = res.selected(q_max, 0.8)
        q = res.q_hat(q_max)
        order = np.argsort(-mp)[:25]
        sets[sg] = {names[i] for i in sel}
        n_sel = len(sel)
        verdict = "supports" if 1 <= n_sel <= 10 else "contradicts"
        out["segments"][sg] = {"n_rows": int(z["n_rows"]), "n_dates": int(z["n_dates"]), "n_sub": int(z["n_sub"]), "q_max": q_max,
                               "q_hat": q, "region_idx": res.region(q_max), "lambda_min_rel": float(res.lams[res.region(q_max)] / res.lams[0]),
                               "pfer_bound": sp.pfer_bound(q, p, 0.8), "selected": [names[i] for i in sel],
                               "selected_probs": {names[i]: float(mp[i]) for i in sel}, "n_selected": n_sel,
                               "top25": [[names[i], float(mp[i]), fam[names[i]]] for i in order],
                               "family_of_selected": {k: sum(1 for i in sel if fam[names[i]] == k) for k in set(fam.values())},
                               "verdict": verdict,
                               "selected_at_0.6": [names[i] for i in res.selected(q_max, 0.6)]}
    A, B = sets[SEGS[0]], sets[SEGS[1]]
    inter, uni = A & B, A | B
    jac = len(inter) / len(uni) if uni else float("nan")
    out["overlap"] = {"intersection": sorted(inter), "union": sorted(uni), "jaccard": jac,
                      "share_of_smaller": len(inter) / min(len(A), len(B)) if min(len(A), len(B)) else float("nan"),
                      "p_hypergeom": _hypergeom_sf(len(inter), p, len(A), len(B)) if A and B else 1.0}
    vs = [out["segments"][g]["verdict"] for g in SEGS]
    if all(v == "supports" for v in vs) and jac >= 0.7:
        pv = "supports"
    else:
        pv = "contradicts"
    out["pooled_verdict"] = pv
    out["p_holm"] = out["overlap"]["p_hypergeom"]
    return out


# ------------------------------------------------------------------ H-SP3 surviving units

HUMAN = {
    "log_sigma": "robust sigma (log)", "lrv_d": "yesterday's log RV", "lrv_w": "7-day mean log RV", "lrv_m": "30-day mean log RV",
    "lrv_q": "90-day mean log RV", "vol_pct_1y": "1y vol percentile", "next_is_weekend": "next day is Sat/Sun",
    "next_is_monday": "next day is Monday", "dow_sin": "weekday (sin)", "dow_cos": "weekday (cos)", "next_wk_hours": "TradFi-weekend hours next day",
    "next_is_mexpiry": "monthly expiry next day", "mkt_lrv_d": "market log RV yesterday", "mkt_lrv_w": "market 7-day log RV",
    "mkt_ret_1d_z": "market return yesterday", "btc_ret_1d_z": "BTC return yesterday", "rq_ratio_d": "quarticity ratio",
    "v0_dir_score": "v0 trend score", "v0_vol": "v0 vol state", "v0_dir": "v0 direction state", "vol_of_vol_30d": "vol of vol",
    "rv_surprise": "RV surprise", "rv_term": "RV term structure", "abs_ret_1d_z": "|return| yesterday", "ret_1d_z": "return yesterday",
    "funding_z_90d": "funding z", "funding_1d": "funding rate", "ret_3d_z": "3-day return", "tstat_7d": "7d trend t-stat",
    "x_ret7_volpct": "7d return x vol percentile", "er_7d": "7d efficiency ratio", "dd_90d_z": "90d drawdown", "range_pos_30d": "30d range position", "day_range_z": "day range z",
}


# dow_sin = sin(2 pi (weekday - 1) / 7): high Tue-Thu, low Fri-Sun; dow_cos: high Sun-Tue, low Thu-Fri
CIRCULAR = {"dow_sin": {True: "the forecast day is Tue-Thu", False: "the forecast day is Fri-Sun"},
            "dow_cos": {True: "the forecast day is Sun-Tue", False: "the forecast day is Wed-Sat"}}


def _frozen_sparse(round_: int, seed: int = 0, month: str = "2026-07"):
    import torch

    d = torch.load(root() / "wf" / f"{month}_vol_states.pt", weights_only=False)
    e = d["seeds"][seed][round_]
    fit = md.MLPFit(e["state"], tuple(d["hidden"]), F, e["y_mu"], e["y_sd"], "vol", float("nan"), e["epochs"])
    z = np.load(root() / "wf" / f"{month}.npz")
    st = Standardizer()
    st.mu, st.sd = np.asarray(d["mu"]), np.asarray(d["sd"])
    return fit, e["masks"], st, float(z["vol_c_mem"][seed, round_])


def _describe(unit: tuple, fit, masks, A: np.ndarray, Z: np.ndarray, X: np.ndarray, gam: dict, grad_mean: float, wk: np.ndarray) -> dict:
    from ai_trader.ml.interpret import spearman

    li, u = unit
    a = A[:, u]
    d: dict = {"layer": li + 1, "unit": u, "fire_rate": float((a > 0).mean()), "mean_act": float(a.mean()),
               "d_logrv_per_act": grad_mean}
    if li == 0:
        w = fit.state["layers.0.weight"].numpy()[u]
        nz = np.flatnonzero(masks["layers.0.weight"][u])
        d["inputs"] = {ML_FEATURES[j]: float(w[j]) for j in nz[np.argsort(-np.abs(w[nz]))]}
    else:
        w = fit.state[f"layers.{li}.weight"].numpy()[u]
        nz = np.flatnonzero(masks[f"layers.{li}.weight"][u])
        d["inputs"] = {f"h{li}_{j}": float(w[j]) for j in nz}
    b1, b2 = gam["best1"], gam["best2"]
    d["best1"] = ML_FEATURES[b1]
    d["best2"] = [ML_FEATURES[j] for j in b2]
    d["rho_best2"] = {ML_FEATURES[j]: spearman(a, X[:, j]) for j in b2}
    d["weekend_mean"] = float(a[wk == 1].mean()) if (wk == 1).any() else float("nan")
    d["weekday_mean"] = float(a[wk == 0].mean()) if (wk == 0).any() else float("nan")
    parts = []
    for j in b2:
        r = d["rho_best2"][ML_FEATURES[j]]
        nm = HUMAN.get(ML_FEATURES[j], ML_FEATURES[j])
        if not np.isfinite(r):
            continue
        if ML_FEATURES[j] in CIRCULAR:
            parts.append(CIRCULAR[ML_FEATURES[j]][r > 0])
        else:
            parts.append(f"{'high' if r > 0 else 'low'} {nm}")
    eff = "raises" if grad_mean > 0 else "lowers"
    d["plain"] = (f"fires on {100 * d['fire_rate']:.0f}% of days, more when " + " and ".join(parts) + f"; when active it {eff} the vol forecast")
    return d


def run_units() -> dict:
    from ai_trader.research import ml_signals_study as ms

    _, c = ms.load_panel()
    a = ms.arrays(c)
    sp1 = json.loads((root() / "sp1_eval.json").read_text(encoding="utf-8"))
    rnd = int(sp1["sp3_round"])
    fit, masks, st, bias = _frozen_sparse(rnd)
    Z = st.transform(a["X"])
    ins, hol = a["seg"] == "llm_in_sample", a["seg"] == "clean_holdout"
    out_all, H = sp.forward_np(fit, Z)
    units = sp.surviving_units(fit, masks)
    attr = sp.unit_attribution(fit, Z[ins])
    # mean gradient d out / d a per unit (sign of effect), in-sample
    import torch

    mt = fit.model()
    x = torch.as_tensor(Z[ins].astype(np.float32))
    acts = []
    zz = x
    for lin in mt.layers:
        zz = torch.relu(lin(zz))
        zz.retain_grad()
        acts.append(zz)
    (mt.head(zz).squeeze(-1) * fit.y_sd).sum().backward()
    gmean = {(li, u): float(acts[li].grad[:, u][acts[li][:, u] > 0].mean()) if bool((acts[li][:, u] > 0).any()) else 0.0
             for li in range(len(acts)) for u in range(acts[li].shape[1])}
    wk = c["next_is_weekend"].to_numpy().astype(float)
    fit80, va20 = md.val_split(a["T"][ins], 0.2)
    Xins = Z[ins]
    rows = []
    for un in units:
        li, u = un
        act = H[li][:, u]
        rec: dict = {"unit": [li + 1, u], "attribution": attr[un]}
        for sg, (Xa, ya, Xb, yb) in {"clean_holdout": (Z[ins], act[ins], Z[hol], act[hol]),
                                     "llm_in_sample": (Xins[fit80], act[ins][fit80], Xins[va20], act[ins][va20])}.items():
            evaluable = bool(np.std(yb) > 1e-9 and np.std(ya) > 1e-9)
            if not evaluable:
                rec[sg] = {"evaluable": False, "sd_eval": float(np.std(yb))}
                continue
            g = sp.gam_search(Xa, ya, Xb, yb)
            i, j = g["best2"]
            rec[sg] = {"evaluable": True, "best1": ML_FEATURES[g["best1"]], "r2_1": g["r2_ev_1"], "best2": [ML_FEATURES[i], ML_FEATURES[j]],
                       "r2_2": g["r2_ev_2"], "r2_fit_2": g["r2_fit_2"], "r2_bivariate": sp.gam_bivariate(Xa, ya, Xb, yb, i, j)}
            if sg == "clean_holdout":
                rec["describe"] = _describe(un, fit, masks, H[li][hol], Z[hol], a["X"][hol], g, gmean[un], wk[hol])
        rows.append(rec)
    order = sorted(units, key=lambda q: -attr[q])
    top3 = order[:3]
    patch = {un: float(H[un[0]][ins, un[1]].mean()) for un in top3}
    out_abl, _ = sp.forward_np(fit, Z, patch)
    zero_patch = {un: 0.0 for un in top3}
    out_zero, _ = sp.forward_np(fit, Z, zero_patch)
    F0 = np.exp(out_all + bias)
    Fa = np.exp(out_abl + bias)
    Fz = np.exp(out_zero + bias)
    pr = c.select("symbol", "T", "segment", "rv_next").with_columns(pl.Series("F0", F0), pl.Series("Fa", Fa), pl.Series("Fz", Fz))
    abl = {}
    for sg in SEGS:
        s = pr.filter(pl.col("segment") == sg)
        _, L = _date_losses(s, ["F0", "Fa", "Fz"])
        abl[sg] = {"mean_ablation": ratio_ci(L["Fa"], L["F0"], thr=1.05), "zero_ablation": ratio_ci(L["Fz"], L["F0"]),
                   "intact_qlike": float(L["F0"].mean())}
        # each unit alone (descriptive)
        each = {}
        for un in order:
            o1, _ = sp.forward_np(fit, Z[pr["segment"].to_numpy() == sg], {un: float(H[un[0]][ins, un[1]].mean())})
            q1 = md.qlike(s["rv_next"].to_numpy(), np.exp(o1 + bias))
            q0 = md.qlike(s["rv_next"].to_numpy(), s["F0"].to_numpy())
            each[f"L{un[0] + 1}u{un[1]}"] = float(np.nanmean(q1) / np.nanmean(q0))
        abl[sg]["each_unit_ratio"] = each
    out = {"round": rnd, "level": sp1["sp3_level"], "sparsity": sp1["sparsity"][rnd], "bias": bias,
           "n_alive_weights": int(sum(int(m.sum()) for m in masks.values())),
           "alive_per_layer": {k: int(m.sum()) for k, m in masks.items()},
           "inputs_with_path": [ML_FEATURES[j] for j in np.flatnonzero(sp.input_paths(masks))],
           "n_units": len(units), "units": rows, "top3": [f"L{q[0] + 1}u{q[1]}" for q in top3], "ablation": abl}
    for sg in SEGS:
        ev = [r for r in rows if r[sg].get("evaluable")]
        meet = [r for r in ev if r[sg]["r2_2"] > 0.8]
        share = len(meet) / len(ev) if ev else float("nan")
        ab = abl[sg]["mean_ablation"]
        holds = ab["ratio"] > 1.05 and ab["ci95"][0] > 1.0
        r2ok = bool(ev) and share >= 0.8
        v = "supports" if (r2ok and holds) else "contradicts" if (not r2ok or ab["ratio"] <= 1.05) else "inconclusive"
        out[sg] = {"n_evaluable": len(ev), "n_meet_r2": len(meet), "share_meet": share, "share_meet_1input": (sum(1 for r in ev if r[sg]["r2_1"] > 0.8) / len(ev)) if ev else float("nan"),
                   "ablation_holds": holds, "verdict": v, "p": ab["p_le_1.05"] if r2ok else 1.0}
    out["pooled_verdict"] = out["clean_holdout"]["verdict"]
    # other seeds at the same round (descriptive): unit counts and inputs
    others = {}
    for s_ in range(1, N_SEEDS):
        f2, m2, _, _ = _frozen_sparse(rnd, s_)
        others[s_] = {"n_units": len(sp.surviving_units(f2, m2)), "inputs_with_path": [ML_FEATURES[j] for j in np.flatnonzero(sp.input_paths(m2))]}
    out["other_seeds"] = others
    (root() / "sp3_units.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


# ------------------------------------------------------------------ evaluate + report


def run_eval_sp1() -> dict:
    df, meta = load_wf()
    sched = json.loads((root() / "schedule.json").read_text(encoding="utf-8"))
    r1 = eval_sp1(df, meta, sched)
    r1["direction"] = eval_dir(df, meta, sched)
    (root() / "sp1_eval.json").write_text(json.dumps(r1, indent=1, default=float), encoding="utf-8")
    return r1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sparsity_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("select")
    s.add_argument("--n-sub", type=int, default=200)
    sub.add_parser("sp1")
    sub.add_parser("units")
    e = sub.add_parser("evaluate")
    e.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "prune":
        run_prune(args.workers)
    elif args.cmd == "select":
        run_select(args.n_sub)
    elif args.cmd == "sp1":
        print(json.dumps({k: v for k, v in run_eval_sp1().items() if k in ("passing_levels", "sp3_level", "pooled_verdict")}))
    elif args.cmd == "units":
        run_units()
    elif args.cmd == "evaluate":
        from ai_trader.research.sparsity_report import evaluate

        evaluate(record=args.record)


if __name__ == "__main__":
    main()


__all__ = ["eval_sp1", "eval_sp2", "library", "load_wf", "month_job", "ratio_ci", "run_prune", "run_select", "run_units"]
_ = (UTC, datetime, holm)


# ------------------------------------------------------------------ H-SP2 exploratory: cluster stability (post hoc)

CLUSTER_RHO = 0.8


def clusters(X: np.ndarray, names: list[str], rho: float = CLUSTER_RHO) -> list[list[int]]:
    """EXPLORATORY (added after the H-SP2 result, which selected nothing): single-linkage clusters of library columns with
    |Pearson| >= rho on llm_in_sample rows."""
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C)
    p = len(names)
    parent = list(range(p))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(np.abs(C) >= rho, 1))
    for i, j in zip(ii, jj, strict=True):
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(p):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def run_cluster_stability(n_sub: int = 200) -> dict:
    """Same subsamples / seed / grid as run_select (identical results), keeping per-subsample selections; probability that
    at least one member of a cluster is selected, max over Lambda."""
    lib, names, _fam = library()
    p = len(names)
    q_max = sp.q_for_bound(p, 0.8, 1.0)
    ins = lib.filter(pl.col("segment") == "llm_in_sample")
    cl = clusters(ins.select(names).to_numpy().astype(float), names)
    out: dict = {"rho": CLUSTER_RHO, "n_clusters": len(cl), "segments": {}}
    for sg in SEGS:
        s = lib.filter(pl.col("segment") == sg)
        X = s.select(names).to_numpy().astype(float)
        y = s["lrv_next"].to_numpy().astype(float)
        T = s["T"].to_numpy().astype("datetime64[D]")
        lams = sp.lambda_grid(X, y, 40, 0.005)
        res = sp.stability_selection(X, y, T, n_sub=n_sub, alpha=0.5, lams=lams, seed=0, stop_union=int(4 * q_max), keep_sel=True)
        k = res.region(q_max)
        z = np.load(root() / f"stability_{sg}.npz")
        same = bool(np.allclose(z["probs"], res.probs))
        rows = []
        for members in cl:
            pr = res.sel[:, : k + 1][:, :, members].any(axis=2).mean(axis=0).max()
            rows.append({"members": [names[i] for i in members], "size": len(members), "prob": float(pr)})
        rows.sort(key=lambda r: -r["prob"])
        out["segments"][sg] = {"identical_to_registered_run": same, "stable_clusters_0.8": [r for r in rows if r["prob"] >= 0.8], "top": rows[:12]}
    (root() / "sp2_clusters.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
