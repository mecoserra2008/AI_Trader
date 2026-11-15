"""Neural-network and tree signal generators with mechanistic interpretability (H-NN1..H-NN4, program 2026-11-12,
agent B). Deterministic research tools only: the LLM decision layer is never trained or backtested here (principle 1).

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.ml_signals_study build
    ... search      in-sample-only hyperparameter search (rows before the first test month), frozen afterwards
    ... walk        expanding monthly walk-forward: forecasts + per-refit attributions (TreeSHAP, IG, group ablation)
    ... interpret   frozen MLP (trained on llm_in_sample only): sparse autoencoder, probes, ablation (H-NN4)
    ... evaluate [--record]   verdicts, Holm within NN, research/reports/ml_signals_2026-11-12.{json,md}

Panel: ai_trader.ml.features (universe L = trend_zoo_study.UNIVERSE, 52 point-in-time features, complete rows only so
every model sees the identical sample). Decision at T = 00:00 UTC of D+1.

Walk-forward (both targets, every model): test month m = rows with T in [m, m+1); training rows = label_end <= m - 1 day
(decision_budget_study.purged_train_mask: 24h purge + 1 day embargo), expanding; first test month = the first with
>= 365 distinct training dates. Inside each window: models.val_split (last 15% of dates = validation, purged), fit-block
standardisation, early stopping, validation-block bias correction (models docstring).

H-NN1 next-day RV. Models: mlp (3-seed ensemble), xgb, lin (ridge OLS on the same 52 features; reference: 'more inputs'
without nonlinearity). Target log RV_{D+1}; variance forecast exp(f + c). Baseline har = per-asset log-HAR
(vol_panel.design('loghar'): log RV_d, mean log RV 7d / 30d), OLS per symbol on that symbol's own training rows (purged,
same months; >= 180 rows else no forecast), forecast exp(xb + s^2/2) as R-VOL-20; also har_qs = the same with the
in-sample QLIKE-optimal scale (robustness). Loss QLIKE in variance space; paired difference model - har, averaged over
symbols per date, NW t over dates, per segment, on rows where every forecast exists. Verdict per segment: supports if
some nonlinear model (mlp, xgb) has diff < 0 and t < -2; contradicts if no nonlinear model has diff < 0, or the better
one has t > 2; inconclusive otherwise. Pooled (the registry criterion): supports iff the SAME model supports in both
segments; contradicts if a segment contradicts; else inconclusive. p (per segment) = min over the two models of
2 Phi(t) (Bonferroni over models).

H-NN2 next-24h direction y = 1{ret_{D+1} > 0}. Models mlp, xgb (and lin, reference). Brier skill = 1 - sum (p - y)^2 /
sum (b - y)^2, b = training base rate of the refit, pooled per segment; 95% CI from a moving-block bootstrap over
decision dates (10-day blocks, 1000 draws, seed 0). Per segment: supports iff both models' CI upper <= +0.005;
contradicts if either model's CI lower > +0.005; else inconclusive. Pooled: supports iff both segments support;
contradicts if either contradicts. p = max over models of P*(skill > 0.005).

H-NN3 TreeSHAP for the xgb vol model on its out-of-sample test rows (each month's model on its own month):
interaction values (n, F+1, F+1), bias excluded; per segment interaction share = sum_{i != j} |phi_ij| / sum_ij |phi_ij|;
top-5 pairs by sum |phi_ij| + |phi_ji|. Bootstrap: 10-day blocks over dates (1000). Per segment: supports if share > 0.10
and its CI lower > 0.10; contradicts if CI upper < 0.10 (or the point <= 0.10); else inconclusive. Pooled: supports iff
both segments support AND the top-5 overlap (llm_in_sample vs clean_holdout) >= 3; contradicts if overlap < 3 or a segment
contradicts. p = max(P*(share <= 0.10) over segments, P*(overlap < 3)).

H-NN4 frozen MLP = the seed-0 member of the walk-forward vol ensemble for the first holdout month (trained only on rows
with label_end <= cutoff - 1 day). Sparse autoencoder (interpret.fit_sae, untied, dictionary 4 x last hidden width,
resampling) on its last hidden activations of the llm_in_sample rows; L1 chosen as the LARGEST of {0.03, 0.1, 0.3, 1.0}
with fraction of variance unexplained <= 0.10 on the last 20% of in-sample dates (SAE fitted on the first 80%), then refit
on all in-sample activations with that L1. 'Most used' = top 10 by firing frequency on the in-sample activations.
State variables (features.STATE_VARS): vol level = lrv_m, vol-of-vol = vol_of_vol_30d, recent jump = jump_share_7d,
weekend = next_is_weekend, trend strength = er_30d, funding z = funding_z_90d, drawdown = dd_90d_z. |Spearman| on the
clean_holdout rows; a feature 'ties' if max |rho| > 0.6. Ablation: H' = H - sum_{k in top 3} z_k a_k (error-preserving:
the SAE reconstruction error stays in H), output through the frozen head, QLIKE with the frozen model's validation bias;
Delta = QLIKE(ablated) - QLIKE(intact), averaged over symbols per date, NW t. clean_holdout: supports iff >= 6 of 10 tie
AND Delta > 0 with t > 2; contradicts if < 6 tie or Delta <= 0; else inconclusive. The llm_in_sample row applies the
same criteria to the SAE's own fitting sample (descriptive). Pooled = the clean_holdout verdict (the registry criterion).
p = Phi(-t_Delta) if >= 6 tie, else 1.

Holm within the NN block per segment over H-NN1..H-NN4; a 'supports' with Holm p > 0.05 is downgraded to inconclusive
(pooled uses the larger of the two segments' Holm p).

Descriptive extras: per-group ablation (inputs of a concept group set to their training mean) for mlp and xgb, both
targets; integrated gradients for the mlp (both targets) vs TreeSHAP main effects (Spearman of mean |attribution| over
features, top-10 overlap, grouped shares); linear probes of state variables on each hidden layer; xgb direction SHAP.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import interpret as ip
from ai_trader.ml import models as md
from ai_trader.ml.features import FEATURE_GROUP, GROUPS, ML_FEATURES, STATE_VARS, Standardizer, build_panel, complete_rows
from ai_trader.research.decision_budget_study import block_bootstrap_idx, holm, purged_train_mask, test_mask

AS_OF = datetime(2026, 11, 12, tzinfo=UTC)
TAG = "2026-11-12"
SEGS = ("llm_in_sample", "clean_holdout")
MIN_TRAIN_DAYS = 365
HAR_MIN_ROWS = 180
SAE_L1_GRID = (0.03, 0.1, 0.3, 1.0)
SAE_FVU_MAX = 0.10
SAE_MULT = 4
RHO_TIE = 0.6
DAY_US = np.timedelta64(86400 * 1_000_000, "us")
F = len(ML_FEATURES)


def root() -> Path:
    d = load().root / "data" / "models" / "ml"
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel_path() -> Path:
    return root() / "cache" / f"panel_{TAG}.parquet"


def load_panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    full = build_panel(AS_OF, UNIVERSE, cache=panel_path(), workers=3)
    return full, complete_rows(full)


def _months(T: np.ndarray) -> list[np.datetime64]:
    lo, hi = T.min().astype("datetime64[M]"), T.max().astype("datetime64[M]")
    return [np.datetime64(m, "us") for m in np.arange(lo, hi + np.timedelta64(1, "M"))]


def _next_month(m: np.datetime64) -> np.datetime64:
    return np.datetime64(m.astype("datetime64[M]") + np.timedelta64(1, "M"), "us")


def first_test_month(T: np.ndarray, le: np.ndarray) -> np.datetime64:
    for m in _months(T):
        if len(np.unique(T[purged_train_mask(le, m)].astype("datetime64[D]"))) >= MIN_TRAIN_DAYS:
            return m
    raise RuntimeError("no month with enough history")


def arrays(c: pl.DataFrame) -> dict:
    return {"X": c.select(ML_FEATURES).to_numpy().astype(float), "T": c["T"].to_numpy(), "le": c["label_end"].to_numpy(),
            "sym": c["symbol"].to_numpy(), "seg": c["segment"].to_numpy(), "rv": c["rv_next"].to_numpy().astype(float),
            "ly": c["lrv_next"].to_numpy().astype(float), "yd": c["y_dir"].to_numpy().astype(float)}


# ------------------------------------------------------------------ search


def run_search() -> dict:
    _, c = load_panel()
    a = arrays(c)
    m0 = first_test_month(a["T"], a["le"])
    pre = a["T"] < m0 - DAY_US  # labels end before the first test month
    Z = Standardizer().fit(a["X"][pre]).transform(a["X"][pre])
    out = {"first_test_month": str(m0)[:7], "n_rows": int(pre.sum()), "n_dates": int(len(np.unique(a["T"][pre].astype("datetime64[D]"))))}
    for task, y in (("vol", a["ly"][pre]), ("dir", a["yd"][pre])):
        for fam in ("mlp", "xgb"):
            t = time.time()
            r = md.search(Z, y, a["T"][pre], a["rv"][pre], task, fam)
            out[f"{fam}_{task}"] = {"best": {k: list(v) if isinstance(v, tuple) else v for k, v in r["best"].items()}, "scores": r["scores"],
                                    "secs": round(time.time() - t, 1)}
            print(fam, task, out[f"{fam}_{task}"], flush=True)
    (root() / "search.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _cfgs() -> dict:
    s = json.loads((root() / "search.json").read_text(encoding="utf-8"))
    cf = {}
    for k in ("mlp_vol", "mlp_dir", "xgb_vol", "xgb_dir"):
        b = dict(s[k]["best"])
        if "hidden" in b:
            b["hidden"] = tuple(b["hidden"])
        cf[k] = b
    return cf


# ------------------------------------------------------------------ walk-forward


def _har_design(c: pl.DataFrame) -> np.ndarray:
    return c.select("lrv_d", "lrv_w", "lrv_m").to_numpy().astype(float)


def har_forecasts(full: pl.DataFrame, c: pl.DataFrame, months: list[np.datetime64]) -> tuple[np.ndarray, np.ndarray]:
    """Per-asset log-HAR on each symbol's own rows (HAR inputs + target finite), purged monthly refits."""
    h = full.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in ("lrv_d", "lrv_w", "lrv_m", "lrv_next")])
                    & (pl.col("rv_next") > 0))
    T, sym = c["T"].to_numpy(), c["symbol"].to_numpy()
    Xc = _har_design(c)
    out = np.full(c.height, np.nan)
    out_qs = np.full(c.height, np.nan)
    for s in np.unique(sym):
        hs = h.filter(pl.col("symbol") == s)
        Xh = np.c_[np.ones(hs.height), _har_design(hs)]
        yh = hs["lrv_next"].to_numpy().astype(float)
        rvh = hs["rv_next"].to_numpy().astype(float)
        leh = hs["label_end"].to_numpy()
        rows_s = sym == s
        for m in months:
            te = rows_s & test_mask(T, m, _next_month(m))
            if not te.any():
                continue
            tr = purged_train_mask(leh, m)
            if tr.sum() < HAR_MIN_ROWS:
                continue
            b, *_ = np.linalg.lstsq(Xh[tr], yh[tr], rcond=None)
            fit = Xh[tr] @ b
            s2 = float(np.mean((yh[tr] - fit) ** 2))
            f = np.c_[np.ones(te.sum()), Xc[te]] @ b
            out[te] = np.exp(f + 0.5 * s2)
            out_qs[te] = np.exp(f + md.bias_const(rvh[tr], fit))
    return out, out_qs


def har_dow_forecasts(full: pl.DataFrame, c: pl.DataFrame, months: list[np.datetime64]) -> np.ndarray:
    """EXPLORATORY (added after the H-NN1 results were seen; not part of any verdict): per-asset log-HAR plus six
    day-of-week dummies of the forecast day D+1, same rows, months and purge as har_forecasts."""
    def des(d: pl.DataFrame) -> np.ndarray:
        wd = d.select((pl.col("date") + pl.duration(days=1)).dt.weekday().alias("w"))["w"].to_numpy()
        return np.c_[_har_design(d), np.column_stack([(wd == k).astype(float) for k in range(2, 8)])]

    h = full.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in ("lrv_d", "lrv_w", "lrv_m", "lrv_next")])
                    & (pl.col("rv_next") > 0))
    T, sym = c["T"].to_numpy(), c["symbol"].to_numpy()
    Xc = des(c)
    out = np.full(c.height, np.nan)
    for s_ in np.unique(sym):
        hs = h.filter(pl.col("symbol") == s_)
        Xh = np.c_[np.ones(hs.height), des(hs)]
        yh = hs["lrv_next"].to_numpy().astype(float)
        leh = hs["label_end"].to_numpy()
        for m in months:
            te = (sym == s_) & test_mask(T, m, _next_month(m))
            if not te.any():
                continue
            tr = purged_train_mask(leh, m)
            if tr.sum() < HAR_MIN_ROWS:
                continue
            b, *_ = np.linalg.lstsq(Xh[tr], yh[tr], rcond=None)
            s2 = float(np.mean((yh[tr] - Xh[tr] @ b) ** 2))
            out[te] = np.exp(np.c_[np.ones(te.sum()), Xc[te]] @ b + 0.5 * s2)
    return out


def calendar_effect(pr: pl.DataFrame, attr) -> dict:
    """Descriptive: TreeSHAP contribution (log RV units) of the calendar group by forecast weekday and v0 vol state."""
    S = attr["shap_vol"][:, :F].astype(float)
    ix = [ML_FEATURES.index(f) for f in GROUPS["calendar"]]
    cal = S[:, ix].sum(1)
    vl = S[:, [ML_FEATURES.index(f) for f in GROUPS["vol_level"]]].sum(1)
    d = pr.select("segment", "v0_vol", (pl.col("date") + pl.duration(days=1)).dt.weekday().alias("wd")).with_columns(
        pl.Series("cal", cal), pl.Series("vl", vl)).filter(pl.col("cal").is_finite())
    out = {}
    names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    for sg in SEGS:
        q = d.filter(pl.col("segment") == sg)
        out[sg] = {"by_weekday": {names[int(w)]: float(v) for w, v in q.group_by("wd").agg(pl.col("cal").mean()).sort("wd").iter_rows()},
                   "weekend_by_vol_state": {}}
        for k, lab in ((0.0, "low"), (1.0, "normal"), (2.0, "high")):
            z = q.filter(pl.col("v0_vol") == k)
            we = z.filter(pl.col("wd") >= 6)["cal"].mean()
            wk = z.filter(pl.col("wd") <= 5)["cal"].mean()
            out[sg]["weekend_by_vol_state"][lab] = {"weekend": float(we), "weekday": float(wk), "gap": float(we - wk), "n": z.height}
    return out


def dependence(pr: pl.DataFrame, attr, top: int = 16) -> dict:
    """Descriptive: direction of each important feature's effect = Spearman(feature value, its TreeSHAP value) on
    out-of-sample rows (xgb vol and dir models), plus the joint market-return contribution vs |mkt_ret_1d_z|."""
    _, c = load_panel()
    assert c.height == pr.height and (c["T"] == pr["T"]).all() and (c["symbol"] == pr["symbol"]).all()
    X = c.select(ML_FEATURES).to_numpy().astype(float)
    seg = pr["segment"].to_numpy()
    out: dict = {}
    for task in ("vol", "dir"):
        S = attr[f"shap_{task}"][:, :F].astype(float)
        out[task] = {}
        for sg in SEGS:
            m = (seg == sg) & np.isfinite(S).all(1)
            imp = np.abs(S[m]).mean(0)
            order = np.argsort(-imp)[:top]
            out[task][sg] = [{"feature": ML_FEATURES[i], "mean_abs_shap": float(imp[i]), "rho_value_shap": ip.spearman(X[m, i], S[m, i]),
                              "shap_q10": float(np.quantile(S[m, i], 0.1)), "shap_q90": float(np.quantile(S[m, i], 0.9))} for i in order]
            if task == "vol":
                j = [ML_FEATURES.index("btc_ret_1d_z"), ML_FEATURES.index("mkt_ret_1d_z")]
                out[task][sg + "_market_abs"] = ip.spearman(np.abs(X[m, j[1]]), S[m][:, j].sum(1))
    return out


def _group_cols() -> dict[str, list[int]]:
    return {g: [ML_FEATURES.index(f) for f in fs] for g, fs in GROUPS.items()}


def run_walk(limit: int | None = None) -> None:
    import torch

    full, c = load_panel()
    a = arrays(c)
    cf = _cfgs()
    months_all = _months(a["T"])
    m0 = first_test_month(a["T"], a["le"])
    months = [m for m in months_all if m >= m0]
    if limit:
        months = months[:limit]
    n = c.height
    cut = np.datetime64(load().settings["research"]["knowledge_cutoff"] + "T00:00:00", "us")
    pred = {k: np.full(n, np.nan) for k in ("F_mlp", "F_xgb", "F_lin", "lf_mlp", "lf_xgb", "lf_lin", "p_mlp", "p_xgb", "p_lin", "base")}
    shap_vol = np.full((n, F + 1), np.nan, dtype=np.float32)
    shap_dir = np.full((n, F + 1), np.nan, dtype=np.float32)
    ig_vol = np.full((n, F), np.nan, dtype=np.float32)
    ig_dir = np.full((n, F), np.nan, dtype=np.float32)
    gcols = _group_cols()
    abl = {f"{fam}_{task}_{g}": np.full(n, np.nan) for fam in ("mlp", "xgb") for task in ("vol", "dir") for g in gcols}
    days = a["T"].astype("datetime64[D]")
    udays, dcode = np.unique(days, return_inverse=True)
    inter_by_day = np.zeros((len(udays), F, F), dtype=np.float32)
    info = []
    mdir = root() / "wf"
    mdir.mkdir(exist_ok=True)
    frozen_done = False
    for m in months:
        t0 = time.time()
        te = test_mask(a["T"], m, _next_month(m))
        if not te.any():
            continue
        tr = np.flatnonzero(purged_train_mask(a["le"], m))
        fit_l, va_l = md.val_split(a["T"][tr])
        fi, vi = tr[fit_l], tr[va_l]
        st = Standardizer().fit(a["X"][fi])
        Zf, Zv, Zt = st.transform(a["X"][fi]), st.transform(a["X"][vi]), st.transform(a["X"][te])
        # ---- vol
        ens = md.fit_mlp_ensemble(Zf, a["ly"][fi], Zv, a["ly"][vi], "vol", cfg=cf["mlp_vol"])
        xg = md.fit_xgb(Zf, a["ly"][fi], Zv, a["ly"][vi], "vol", cfg=cf["xgb_vol"], names=ML_FEATURES)
        ln = md.fit_linear(Zf, a["ly"][fi], "vol")
        for name, mod in (("mlp", ens), ("xgb", xg), ("lin", ln)):
            cst = md.bias_const(a["rv"][vi], mod.predict(Zv))
            lf = mod.predict(Zt)
            pred[f"lf_{name}"][te] = lf
            pred[f"F_{name}"][te] = np.exp(lf + cst)
            if name in ("mlp", "xgb"):
                for g, cols in gcols.items():
                    Za = Zt.copy()
                    Za[:, cols] = 0.0
                    abl[f"{name}_vol_{g}"][te] = np.exp(mod.predict(Za) + cst)
        shap_vol[te] = ip.shap_main(xg, Zt)
        it = np.abs(ip.shap_interactions(xg, Zt)[:, :F, :F]).astype(np.float32)
        np.add.at(inter_by_day, dcode[te], it)
        ig_vol[te] = ip.integrated_gradients(ens.members, Zt)
        # ---- dir
        ensd = md.fit_mlp_ensemble(Zf, a["yd"][fi], Zv, a["yd"][vi], "dir", cfg=cf["mlp_dir"])
        xgd = md.fit_xgb(Zf, a["yd"][fi], Zv, a["yd"][vi], "dir", cfg=cf["xgb_dir"], names=ML_FEATURES)
        lnd = md.fit_linear(Zf, a["yd"][fi], "dir")
        pred["base"][te] = a["yd"][tr].mean()
        for name, mod in (("mlp", ensd), ("xgb", xgd), ("lin", lnd)):
            pred[f"p_{name}"][te] = mod.predict(Zt)
            if name in ("mlp", "xgb"):
                for g, cols in gcols.items():
                    Za = Zt.copy()
                    Za[:, cols] = 0.0
                    abl[f"{name}_dir_{g}"][te] = mod.predict(Za)
        shap_dir[te] = ip.shap_main(xgd, Zt)
        ig_dir[te] = _ig_logit(ensd.members, Zt)
        # ---- persist the month's models (small) for reproducibility
        tag = str(m)[:7]
        xg.booster.save_model(str(mdir / f"xgb_vol_{tag}.json"))
        xgd.booster.save_model(str(mdir / f"xgb_dir_{tag}.json"))
        torch.save({"vol": [mm.__dict__ for mm in ens.members], "dir": [mm.__dict__ for mm in ensd.members], "std": st.to_dict()},
                   mdir / f"mlp_{tag}.pt")
        if not frozen_done and m >= cut:
            frozen_done = True
            torch.save({"member": ens.members[0].__dict__, "std": st.to_dict(), "month": tag,
                        "bias": md.bias_const(a["rv"][vi], ens.members[0].predict(Zv)), "train_last_label_end": str(a["le"][tr].max())},
                       root() / "frozen_mlp_vol.pt")
        info.append({"month": tag, "n_train": int(len(tr)), "n_fit": int(len(fi)), "n_val": int(len(vi)), "n_test": int(te.sum()),
                     "mlp_vol_epochs": [mm.epochs for mm in ens.members], "mlp_dir_epochs": [mm.epochs for mm in ensd.members],
                     "xgb_vol_iter": xg.best_iteration, "xgb_dir_iter": xgd.best_iteration, "secs": round(time.time() - t0, 1)})
        print(json.dumps(info[-1]), flush=True)
    har, har_qs = har_forecasts(full, c, months)
    keep = list(dict.fromkeys(["symbol", "date", "T", "segment", "rv_next", "lrv_next", "y_dir", "ret_next_z", "v0_vol", "next_is_weekend",
                               "vol_pct_1y", "jump_share_7d", *STATE_VARS.values()]))
    out = c.select(keep).with_columns(
        pl.Series("F_har", har), pl.Series("F_har_qs", har_qs), *[pl.Series(k, v) for k, v in pred.items()],
        *[pl.Series(f"abl_{k}", v) for k, v in abl.items()])
    out.write_parquet(root() / "wf_preds.parquet")
    np.savez_compressed(root() / "wf_attr.npz", shap_vol=shap_vol, shap_dir=shap_dir, ig_vol=ig_vol, ig_dir=ig_dir,
                        inter_by_day=inter_by_day, udays=udays.astype("datetime64[D]").astype(str))
    (root() / "wf_info.json").write_text(json.dumps({"first_test_month": str(m0)[:7], "cfg": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                                                                               for kk, vv in v.items()} for k, v in cf.items()},
                                                    "months": info}, indent=1), encoding="utf-8")


def _ig_logit(fits, Z: np.ndarray, steps: int = 32) -> np.ndarray:
    """IG of the direction ensemble's logit (y_sd = 1, y_mu = 0 for 'dir' fits)."""
    return ip.integrated_gradients(fits, Z, steps)


# ------------------------------------------------------------------ interpret (H-NN4)


def _frozen():
    import torch

    d = torch.load(root() / "frozen_mlp_vol.pt", weights_only=False)
    fit = md.MLPFit(**d["member"])
    st = Standardizer()
    st.mu, st.sd = np.asarray(d["std"]["mu"]), np.asarray(d["std"]["sd"])
    return fit, st, float(d["bias"]), d


def _date_mean_diff(T: np.ndarray, d: np.ndarray) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    ok = np.isfinite(d)
    g = pl.DataFrame({"t": T[ok], "d": d[ok]}).group_by("t").agg(pl.col("d").mean()).sort("t")
    mean, t, nd = newey_west_t(g["d"].to_numpy())
    return {"mean": float(mean), "t": float(t), "n_dates": int(nd)}


def run_interpret() -> dict:
    _, c = load_panel()
    a = arrays(c)
    fit, st, bias, meta = _frozen()
    Z = st.transform(a["X"])
    H = ip.hidden_activations(fit, Z)
    Hl = H[-1]
    ins, hol = a["seg"] == "llm_in_sample", a["seg"] == "clean_holdout"
    width = Hl.shape[1]
    msz = SAE_MULT * width
    # L1 selection: largest L1 with FVU <= 0.10 on the last 20% of in-sample dates
    fit80, va20 = md.val_split(a["T"][ins], 0.2)
    Hin = Hl[ins]
    sel = []
    for l1 in SAE_L1_GRID:
        s = ip.fit_sae(Hin[fit80], msz, l1=l1, steps=8000, seed=0)
        mt = s.metrics(Hin[va20])
        sel.append({"l1": l1, "fvu_val": mt["fvu"], "l0_val": mt["l0"], "dead_frac_val": mt["dead_frac"]})
        print(sel[-1], flush=True)
    ok = [r["l1"] for r in sel if r["fvu_val"] <= SAE_FVU_MAX]
    l1 = max(ok) if ok else min(SAE_L1_GRID)
    sae = ip.fit_sae(Hin, msz, l1=l1, steps=12000, seed=0)
    met_in = sae.metrics(Hin)
    met_ho = sae.metrics(Hl[hol])
    freq = met_in["freq"]
    top = list(np.argsort(-freq, kind="stable")[:10])
    Zin_sae, Zho_sae = sae.encode(Hin), sae.encode(Hl[hol])
    sv = {k: c[v].to_numpy().astype(float) for k, v in STATE_VARS.items()}
    w_head = fit.state["head.weight"].numpy().astype(float).ravel() * fit.y_sd

    def corr_table(Zs: np.ndarray, mask: np.ndarray) -> list[dict]:
        rows = []
        for k in top:
            rh = {name: ip.spearman(Zs[:, k], x[mask]) for name, x in sv.items()}
            best = max(rh, key=lambda q: abs(rh[q]) if np.isfinite(rh[q]) else -1)
            # top input correlates (naming aid)
            inp = {f: ip.spearman(Zs[:, k], a["X"][mask, j]) for j, f in enumerate(ML_FEATURES)}
            tops = sorted(inp, key=lambda q: -abs(inp[q]) if np.isfinite(inp[q]) else 0)[:3]
            rows.append({"feature": int(k), "freq_in": float(freq[k]), "freq_ho": float((Zs[:, k] > 0).mean()),
                         "rho": rh, "best_state": best, "best_abs_rho": float(abs(rh[best])), "ties": bool(abs(rh[best]) > RHO_TIE),
                         "top_inputs": [[f, inp[f]] for f in tops],
                         "dlogrv_per_unit": float(w_head @ sae.atom(k)), "mean_act": float(Zs[:, k].mean())})
        return rows

    tab_ho = corr_table(Zho_sae, hol)
    tab_in = corr_table(Zin_sae, ins)

    def ablate(mask: np.ndarray, Zs: np.ndarray, feats: list[int]) -> dict:
        Hm = Hl[mask]
        base = np.exp(ip.head_from_last(fit, Hm) + bias)
        Hm2 = Hm - sum(np.outer(Zs[:, k], sae.atom(k)) for k in feats)
        abl_ = np.exp(ip.head_from_last(fit, Hm2) + bias)
        d = md.qlike(a["rv"][mask], abl_) - md.qlike(a["rv"][mask], base)
        r = _date_mean_diff(a["T"][mask], d)
        r["qlike_intact"] = float(np.nanmean(md.qlike(a["rv"][mask], base)))
        return r

    abl = {}
    for seg, mask, Zs in (("clean_holdout", hol, Zho_sae), ("llm_in_sample", ins, Zin_sae)):
        abl[seg] = {"top3": ablate(mask, Zs, top[:3]), "top10": ablate(mask, Zs, top), "each": {int(k): ablate(mask, Zs, [k]) for k in top}}
    # linear probes: state variables from inputs / each hidden layer; fit in-sample, R^2 holdout
    probes = {}
    layers = {"input": Z, **{f"hidden{i + 1}": h for i, h in enumerate(H)}}
    for lname, Hx in layers.items():
        probes[lname] = {}
        for name, x in sv.items():
            okr = np.isfinite(x)
            probes[lname][name] = ip.ridge_probe(Hx[ins & okr], x[ins & okr], Hx[hol & okr], x[hol & okr])
    # probes from SAE codes
    probes["sae_codes"] = {name: ip.ridge_probe(Zin_sae, x[ins], Zho_sae, x[hol]) for name, x in sv.items()}
    # frozen model quality on the holdout vs walk-forward
    Fh = np.exp(ip.head_from_last(fit, Hl[hol]) + bias)
    out = {"frozen": {"month": meta["month"], "train_last_label_end": meta["train_last_label_end"], "hidden": list(fit.hidden), "bias": bias,
                      "holdout_qlike": float(np.nanmean(md.qlike(a["rv"][hol], Fh)))},
           "sae": {"dict_size": msz, "width": width, "l1_selection": sel, "l1": l1, "metrics_in": {k: v for k, v in met_in.items() if k != "freq"},
                   "metrics_ho": {k: v for k, v in met_ho.items() if k != "freq"}, "resampled": sae.history["resampled"],
                   "n_alive_in": int((freq > 0).sum())},
           "top10": [int(k) for k in top], "corr_holdout": tab_ho, "corr_in_sample": tab_in,
           "n_ties_holdout": int(sum(r["ties"] for r in tab_ho)), "n_ties_in_sample": int(sum(r["ties"] for r in tab_in)),
           "ablation": abl, "probes": probes}
    (root() / "sae_vol.json").write_text(json.dumps(sae.to_dict()), encoding="utf-8")
    (root() / "interpret.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


# ------------------------------------------------------------------ evaluate


def _phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2)) if np.isfinite(t) else 1.0


def paired(df: pl.DataFrame, a_col: str, b_col: str) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    la = md.qlike(df["rv_next"].to_numpy(), df[a_col].to_numpy())
    lb = md.qlike(df["rv_next"].to_numpy(), df[b_col].to_numpy())
    ok = np.isfinite(la) & np.isfinite(lb)
    g = pl.DataFrame({"t": df["T"].to_numpy()[ok], "la": la[ok], "lb": lb[ok]}).group_by("t").agg(pl.col("la").mean(), pl.col("lb").mean()).sort("t")
    d = (g["la"] - g["lb"]).to_numpy()
    m, t, nd = newey_west_t(d)
    mb = float(g["lb"].mean())
    return {"diff": float(m), "t": float(t), "n_dates": int(nd), "n_obs": int(ok.sum()), "loss_model": float(g["la"].mean()), "loss_ref": mb,
            "rel": float(m / mb) if mb else float("nan")}


def brier_block(df: pl.DataFrame, pcol: str) -> dict:
    y = df["y_dir"].to_numpy().astype(float)
    p = df[pcol].to_numpy().astype(float)
    b = df["base"].to_numpy().astype(float)
    codes, nd = _codes(df["T"].to_numpy())
    sq = np.bincount(codes, (p - y) ** 2, nd)
    ref = np.bincount(codes, (b - y) ** 2, nd)
    idx = block_bootstrap_idx(nd, 10, 1000, 0)
    boot = 1 - sq[idx].sum(1) / ref[idx].sum(1)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))) if 0 < n1 < len(y) else float("nan")
    return {"skill": float(1 - sq.sum() / ref.sum()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "p_gt_0005": float((1 + np.sum(boot > 0.005)) / (len(boot) + 1)), "n_obs": int(len(y)), "n_dates": int(nd),
            "auc": auc, "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), "sd_p": float(p.std()),
            "base_rate_realised": float(y.mean())}


def _codes(T: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(T.astype("datetime64[D]"), return_inverse=True)
    return inv, int(inv.max()) + 1


def nn1(pr: pl.DataFrame) -> dict:
    need = ["F_har", "F_har_qs", "F_mlp", "F_xgb", "F_lin"]
    df = pr.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in need]))
    out: dict = {"models": ["mlp", "xgb", "lin"], "n_common": df.height}
    for seg in SEGS:
        s = df.filter(pl.col("segment") == seg)
        r = {f"{m}_vs_har": paired(s, f"F_{m}", "F_har") for m in ("mlp", "xgb", "lin")}
        r |= {f"{m}_vs_har_qs": paired(s, f"F_{m}", "F_har_qs") for m in ("mlp", "xgb", "lin")}
        r["mlp_vs_xgb"] = paired(s, "F_mlp", "F_xgb")
        r["mlp_vs_lin"] = paired(s, "F_mlp", "F_lin")
        r["xgb_vs_lin"] = paired(s, "F_xgb", "F_lin")
        r["har_qs_vs_har"] = paired(s, "F_har_qs", "F_har")
        r["n_symbols"] = s["symbol"].n_unique()
        r["mean_qlike"] = {m: float(np.nanmean(md.qlike(s["rv_next"].to_numpy(), s[f"F_{m}"].to_numpy()))) for m in ("har", "har_qs", "mlp", "xgb", "lin")}
        nl = {m: r[f"{m}_vs_har"] for m in ("mlp", "xgb")}
        if any(v["diff"] < 0 and v["t"] < -2 for v in nl.values()):
            v = "supports"
        elif all(not (v["diff"] < 0) for v in nl.values()) or min(nl.values(), key=lambda q: q["diff"])["t"] > 2:
            v = "contradicts"
        else:
            v = "inconclusive"
        r["verdict"] = v
        r["p"] = float(min(1.0, min(2 * _phi(nl[m]["t"]) for m in nl)))
        # descriptive segments: v0 vol state, weekend, symbol
        r["by_vol_state"] = {str(k): paired(s.filter(pl.col("v0_vol") == k), "F_xgb", "F_har") | {"mlp": paired(s.filter(pl.col("v0_vol") == k), "F_mlp", "F_har")}
                             for k in (0.0, 1.0, 2.0) if s.filter(pl.col("v0_vol") == k).height > 200}
        r["by_weekend"] = {str(k): {"xgb": paired(s.filter(pl.col("next_is_weekend") == k), "F_xgb", "F_har"),
                                    "mlp": paired(s.filter(pl.col("next_is_weekend") == k), "F_mlp", "F_har")} for k in (0.0, 1.0)}
        r["by_symbol_rel"] = {}
        for sym in sorted(s["symbol"].unique().to_list()):
            ss = s.filter(pl.col("symbol") == sym)
            q = {m: float(np.nanmean(md.qlike(ss["rv_next"].to_numpy(), ss[f"F_{m}"].to_numpy()))) for m in ("har", "mlp", "xgb")}
            r["by_symbol_rel"][sym] = {"n": ss.height, "mlp": q["mlp"] / q["har"] - 1, "xgb": q["xgb"] / q["har"] - 1}
        out[seg] = r
    win = [m for m in ("mlp", "xgb") if all(out[s][f"{m}_vs_har"]["diff"] < 0 and out[s][f"{m}_vs_har"]["t"] < -2 for s in SEGS)]
    vs = [out[s]["verdict"] for s in SEGS]
    out["pooled_verdict"] = "supports" if win else ("contradicts" if "contradicts" in vs else "inconclusive")
    out["winners_both_segments"] = win
    return out


def nn2(pr: pl.DataFrame) -> dict:
    df = pr.filter(pl.all_horizontal([pl.col(x).is_not_null() & pl.col(x).is_finite() for x in ("p_mlp", "p_xgb", "p_lin", "base")]))
    out: dict = {"n_common": df.height}
    for seg in SEGS:
        s = df.filter(pl.col("segment") == seg)
        r = {m: brier_block(s, f"p_{m}") for m in ("mlp", "xgb", "lin")}
        if all(r[m]["ci95"][1] <= 0.005 for m in ("mlp", "xgb")):
            v = "supports"
        elif any(r[m]["ci95"][0] > 0.005 for m in ("mlp", "xgb")):
            v = "contradicts"
        else:
            v = "inconclusive"
        r["verdict"] = v
        r["p"] = float(max(r[m]["p_gt_0005"] for m in ("mlp", "xgb")))
        out[seg] = r
    vs = [out[s]["verdict"] for s in SEGS]
    out["pooled_verdict"] = "supports" if vs == ["supports", "supports"] else ("contradicts" if "contradicts" in vs else "inconclusive")
    return out


def nn3(pr: pl.DataFrame, attr) -> dict:
    inter = attr["inter_by_day"].astype(float)
    udays = attr["udays"].astype("datetime64[D]")
    seg_by_day = {}
    dd = pr.select(pl.col("T").dt.date().alias("d"), "segment").unique()
    for d, s in dd.iter_rows():
        seg_by_day[np.datetime64(d, "D")] = s
    has = inter.reshape(len(udays), -1).sum(1) > 0
    out: dict = {}
    boots = {}
    for seg in SEGS:
        sel = np.array([seg_by_day.get(d) == seg for d in udays]) & has
        A = inter[sel]
        summ = ip.interaction_summary(A.sum(0), ML_FEATURES)
        idx = block_bootstrap_idx(int(sel.sum()), 10, 1000, 0)
        sh, sets = [], []
        for b in idx:
            sb = ip.interaction_summary(A[b].sum(0), ML_FEATURES)
            sh.append(sb["interaction_share"])
            sets.append(ip.pair_set(sb))
        sh = np.asarray(sh)
        boots[seg] = sets
        ci = [float(np.quantile(sh, 0.025)), float(np.quantile(sh, 0.975))]
        share = summ["interaction_share"]
        v = "supports" if share > 0.10 and ci[0] > 0.10 else ("contradicts" if share <= 0.10 or ci[1] < 0.10 else "inconclusive")
        # grouped (concept-level) attribution: main vs interaction within / across groups
        G = list(GROUPS)
        gi = np.array([G.index(FEATURE_GROUP[f]) for f in ML_FEATURES])
        M = A.sum(0)
        GM = np.zeros((len(G), len(G)))
        for i in range(F):
            for j in range(F):
                GM[gi[i], gi[j]] += M[i, j]
        tot = M.sum()
        gpairs = sorted(((G[i], G[j], float((GM[i, j] + GM[j, i]) / tot)) for i in range(len(G)) for j in range(i + 1, len(G))), key=lambda q: -q[2])[:8]
        out[seg] = {"n_days": int(sel.sum()), "interaction_share": share, "share_ci95": ci, "p_share_le_010": float((1 + np.sum(sh <= 0.10)) / (len(sh) + 1)),
                    "top_pairs": summ["top_pairs"], "top_main": sorted(summ["main_share"].items(), key=lambda q: -q[1])[:10],
                    "group_main_share": {G[i]: float(GM[i, i] / tot) for i in range(len(G))},
                    "group_within_interaction_share": float(sum(GM[i, i] for i in range(len(G))) / tot - sum(M[i, i] for i in range(F)) / tot),
                    "top_group_pairs": gpairs, "verdict": v}
    s_in, s_ho = ip.pair_set({"top_pairs": out["llm_in_sample"]["top_pairs"]}), ip.pair_set({"top_pairs": out["clean_holdout"]["top_pairs"]})
    overlap = len(s_in & s_ho)
    ov_b = np.array([len(a & b) for a, b in zip(boots["llm_in_sample"], boots["clean_holdout"], strict=True)])
    out["top5_overlap"] = overlap
    out["overlap_boot_mean"] = float(ov_b.mean())
    out["p_overlap_lt3"] = float((1 + np.sum(ov_b < 3)) / (len(ov_b) + 1))
    vs = [out[s]["verdict"] for s in SEGS]
    out["pooled_verdict"] = ("supports" if vs == ["supports", "supports"] and overlap >= 3 else
                             "contradicts" if overlap < 3 or "contradicts" in vs else "inconclusive")
    for seg in SEGS:
        out[seg]["p"] = out[seg]["p_share_le_010"]
    out["p_pooled"] = float(max(out["llm_in_sample"]["p"], out["clean_holdout"]["p"], out["p_overlap_lt3"]))
    return out


def nn4(it: dict) -> dict:
    out = {}
    for seg in SEGS:
        n_t = it["n_ties_holdout"] if seg == "clean_holdout" else it["n_ties_in_sample"]
        ab = it["ablation"][seg]["top3"]
        if n_t >= 6 and ab["mean"] > 0 and ab["t"] > 2:
            v = "supports"
        elif n_t < 6 or not ab["mean"] > 0:
            v = "contradicts"
        else:
            v = "inconclusive"
        out[seg] = {"n_ties": n_t, "ablation_top3": ab, "verdict": v, "p": float(_phi(-ab["t"])) if n_t >= 6 else 1.0}
    out["pooled_verdict"] = out["clean_holdout"]["verdict"]
    return out


def attributions(pr: pl.DataFrame, attr) -> dict:
    """IG (mlp) vs TreeSHAP main (xgb): mean |attribution| per feature, per segment; grouped shares; agreement."""
    seg = pr["segment"].to_numpy()
    out: dict = {}
    for task in ("vol", "dir"):
        S = attr[f"shap_{task}"][:, :F].astype(float)
        G = attr[f"ig_{task}"].astype(float)
        out[task] = {}
        for sg in SEGS:
            m = (seg == sg) & np.isfinite(S).all(1) & np.isfinite(G).all(1)
            ms, mg = np.abs(S[m]).mean(0), np.abs(G[m]).mean(0)
            rs, rg = np.argsort(-ms), np.argsort(-mg)
            grp = {}
            for gname, fs in GROUPS.items():
                ix = [ML_FEATURES.index(f) for f in fs]
                grp[gname] = {"xgb_shap": float(ms[ix].sum() / ms.sum()), "mlp_ig": float(mg[ix].sum() / mg.sum())}
            # signed dependence: corr(feature value, attribution) for the top features (direction of effect)
            out[task][sg] = {"n": int(m.sum()), "spearman_feature_ranking": ip.spearman(ms, mg),
                             "top10_overlap": len(set(rs[:10]) & set(rg[:10])),
                             "xgb_top10": [[ML_FEATURES[i], float(ms[i] / ms.sum())] for i in rs[:10]],
                             "mlp_top10": [[ML_FEATURES[i], float(mg[i] / mg.sum())] for i in rg[:10]],
                             "groups": grp, "group_spearman": ip.spearman(np.array([v["xgb_shap"] for v in grp.values()]), np.array([v["mlp_ig"] for v in grp.values()]), min_n=5)}
    return out


def ablation_summary(pr: pl.DataFrame) -> dict:
    out: dict = {}
    for task in ("vol", "dir"):
        out[task] = {}
        for sg in SEGS:
            s = pr.filter(pl.col("segment") == sg)
            r = {}
            for fam in ("mlp", "xgb"):
                for g in GROUPS:
                    col = f"abl_{fam}_{task}_{g}"
                    if task == "vol":
                        q = s.filter(pl.col(col).is_not_null() & pl.col(f"F_{fam}").is_not_null())
                        rr = paired(q, col, f"F_{fam}")
                        r[f"{fam}:{g}"] = {"dqlike": rr["diff"], "t": rr["t"], "rel": rr["rel"]}
                    else:
                        q = s.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & pl.col(f"p_{fam}").is_not_null()
                                     & pl.col(f"p_{fam}").is_finite() & pl.col("base").is_finite())
                        y = q["y_dir"].to_numpy()
                        b0 = np.mean((q[f"p_{fam}"].to_numpy() - y) ** 2)
                        b1 = np.mean((q[col].to_numpy() - y) ** 2)
                        ref = np.mean((q["base"].to_numpy() - y) ** 2)
                        r[f"{fam}:{g}"] = {"dskill": float(-(b1 - b0) / ref)}  # skill lost (negative = the group helped)
            out[task][sg] = r
    return out


def run_evaluate(record_rows: bool = False) -> dict:
    pr = pl.read_parquet(root() / "wf_preds.parquet")
    attr = np.load(root() / "wf_attr.npz")
    it = json.loads((root() / "interpret.json").read_text(encoding="utf-8"))
    info = json.loads((root() / "wf_info.json").read_text(encoding="utf-8"))
    search_ = json.loads((root() / "search.json").read_text(encoding="utf-8"))
    out = {"as_of": AS_OF.isoformat(), "cutoff": load().settings["research"]["knowledge_cutoff"], "features": list(ML_FEATURES),
           "groups": {k: list(v) for k, v in GROUPS.items()}, "search": search_, "walk": {"first_test_month": info["first_test_month"],
           "n_months": len(info["months"]), "cfg": info["cfg"], "secs_total": sum(m["secs"] for m in info["months"]),
           "mean_epochs_vol": float(np.mean([np.mean(m["mlp_vol_epochs"]) for m in info["months"]])),
           "mean_epochs_dir": float(np.mean([np.mean(m["mlp_dir_epochs"]) for m in info["months"]])),
           "mean_xgb_iter_vol": float(np.mean([m["xgb_vol_iter"] for m in info["months"]])),
           "mean_xgb_iter_dir": float(np.mean([m["xgb_dir_iter"] for m in info["months"]]))},
           "panel": {"rows": pr.height, "symbols": pr["symbol"].n_unique(), "first_T": str(pr["T"].min()), "last_T": str(pr["T"].max()),
                     "rows_by_segment": dict(pr.group_by("segment").len().iter_rows())}}
    if "F_har_dow" not in pr.columns:
        full, c = load_panel()
        months = [m for m in _months(c["T"].to_numpy()) if m >= np.datetime64(info["first_test_month"] + "-01T00:00:00", "us")]
        hd = c.select("symbol", "T").with_columns(pl.Series("F_har_dow", har_dow_forecasts(full, c, months)))
        pr = pr.join(hd, on=["symbol", "T"], how="left")
        pr.write_parquet(root() / "wf_preds.parquet")
    out["H-NN1"] = nn1(pr)
    out["exploratory_har_dow"] = {}
    for sg in SEGS:
        q = pr.filter((pl.col("segment") == sg) & pl.all_horizontal([pl.col(x).is_finite() for x in ("F_har", "F_har_dow", "F_mlp", "F_xgb", "F_lin")]))
        out["exploratory_har_dow"][sg] = {"har_dow_vs_har": paired(q, "F_har_dow", "F_har"), "mlp_vs_har_dow": paired(q, "F_mlp", "F_har_dow"),
                                          "xgb_vs_har_dow": paired(q, "F_xgb", "F_har_dow"), "lin_vs_har_dow": paired(q, "F_lin", "F_har_dow")}
    out["calendar_effect"] = calendar_effect(pr, attr)
    out["dependence"] = dependence(pr, attr)
    out["H-NN2"] = nn2(pr)
    out["H-NN3"] = nn3(pr, attr)
    out["H-NN4"] = nn4(it)
    out["interpret"] = it
    out["attributions"] = attributions(pr, attr)
    out["group_ablation"] = ablation_summary(pr)
    out["holm"] = {}
    for seg in SEGS:
        ps = {h: out[h][seg]["p"] for h in ("H-NN1", "H-NN2", "H-NN3", "H-NN4")}
        out["holm"][seg] = {"p": ps, "p_holm": holm(ps)}
    out["verdicts"] = {}
    for h in ("H-NN1", "H-NN2", "H-NN3", "H-NN4"):
        vs = {}
        for seg in SEGS:
            v = out[h][seg]["verdict"]
            vs[seg] = "inconclusive" if v == "supports" and out["holm"][seg]["p_holm"][h] > 0.05 else v
        pv = out[h]["pooled_verdict"]
        if pv == "supports" and max(out["holm"][s]["p_holm"][h] for s in SEGS if h != "H-NN4" or s == "clean_holdout") > 0.05:
            pv = "inconclusive"
        vs["pooled"] = pv
        out["verdicts"][h] = vs
    rep = load().root / "research" / "reports"
    (rep / f"ml_signals_{TAG}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    from ai_trader.research.ml_signals_report import build as build_report

    tab = tables_md(out)
    (root() / f"report_tables_{TAG}.md").write_text(tab, encoding="utf-8")
    (rep / f"ml_signals_{TAG}.md").write_text(build_report(out, tab), encoding="utf-8")
    print(json.dumps(out["verdicts"], indent=1))
    if record_rows:
        record_all(out)
    return out


def _f(x, k=4) -> str:
    try:
        return f"{float(x):.{k}f}"
    except (TypeError, ValueError):
        return str(x)


def tables_md(out: dict) -> str:
    """Auto-generated result tables (the report's narrative and synthesis are written around them)."""
    L = []
    n1 = out["H-NN1"]
    L.append("### H-NN1 next-day RV: QLIKE vs per-asset log-HAR\n")
    L.append("| segment | model | QLIKE model | QLIKE log-HAR | diff | rel | NW t | dates |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        for m in ("mlp", "xgb", "lin"):
            r = n1[s][f"{m}_vs_har"]
            L.append(f"| {s} | {m} | {_f(r['loss_model'])} | {_f(r['loss_ref'])} | {_f(r['diff'])} | {100 * r['rel']:+.1f}% | {_f(r['t'], 2)} | {r['n_dates']} |")
    L.append("\nOther pairs (diff, NW t):\n")
    L.append("| segment | mlp - xgb | mlp - lin | xgb - lin | har_qs - har | mlp - har_qs | xgb - har_qs |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SEGS:
        r = n1[s]
        c = [r["mlp_vs_xgb"], r["mlp_vs_lin"], r["xgb_vs_lin"], r["har_qs_vs_har"], r["mlp_vs_har_qs"], r["xgb_vs_har_qs"]]
        L.append(f"| {s} | " + " | ".join(f"{_f(x['diff'])} ({_f(x['t'], 1)})" for x in c) + " |")
    L.append("\nBy v0 vol state (xgb / mlp rel. QLIKE vs log-HAR, t):\n")
    L.append("| segment | vol state | n obs | xgb | mlp |")
    L.append("|---|---|---|---|---|")
    for s in SEGS:
        for k, r in n1[s]["by_vol_state"].items():
            L.append(f"| {s} | {['low', 'normal', 'high'][int(float(k))]} | {r['n_obs']} | {100 * r['rel']:+.1f}% ({_f(r['t'], 1)}) | {100 * r['mlp']['rel']:+.1f}% ({_f(r['mlp']['t'], 1)}) |")
        for k, r in n1[s]["by_weekend"].items():
            L.append(f"| {s} | next day weekend={int(float(k))} | {r['xgb']['n_obs']} | {100 * r['xgb']['rel']:+.1f}% ({_f(r['xgb']['t'], 1)}) | {100 * r['mlp']['rel']:+.1f}% ({_f(r['mlp']['t'], 1)}) |")
    n2 = out["H-NN2"]
    L.append("\n### H-NN2 next-24h direction: Brier skill vs training base rate\n")
    L.append("| segment | model | skill | 95% CI | AUC | log loss | sd(p) | P*(skill > 0.005) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        for m in ("mlp", "xgb", "lin"):
            r = n2[s][m]
            L.append(f"| {s} | {m} | {_f(r['skill'], 5)} | [{_f(r['ci95'][0], 5)}, {_f(r['ci95'][1], 5)}] | {_f(r['auc'])} | {_f(r['log_loss'])} | {_f(r['sd_p'])} | {_f(r['p_gt_0005'], 3)} |")
    n3 = out["H-NN3"]
    L.append("\n### H-NN3 TreeSHAP (xgb vol model, out-of-sample rows)\n")
    L.append("| segment | interaction share | 95% CI | top-5 pairs (share of total |phi|) |")
    L.append("|---|---|---|---|")
    for s in SEGS:
        r = n3[s]
        pairs = "; ".join(f"{a} x {b} {100 * v:.2f}%" for a, b, v in r["top_pairs"])
        L.append(f"| {s} | {100 * r['interaction_share']:.1f}% | [{100 * r['share_ci95'][0]:.1f}, {100 * r['share_ci95'][1]:.1f}] | {pairs} |")
    L.append(f"\nTop-5 overlap across segments: {n3['top5_overlap']} (bootstrap mean {n3['overlap_boot_mean']:.2f}, P*(overlap < 3) {n3['p_overlap_lt3']:.3f}).\n")
    L.append("| segment | top main effects (share of total |phi|) |")
    L.append("|---|---|")
    for s in SEGS:
        L.append(f"| {s} | " + "; ".join(f"{a} {100 * v:.1f}%" for a, v in n3[s]["top_main"]) + " |")
    L.append("\nConcept-group pairs (interaction share of total |phi|):\n")
    for s in SEGS:
        L.append(f"- {s}: " + "; ".join(f"{a} x {b} {100 * v:.2f}%" for a, b, v in n3[s]["top_group_pairs"][:6]))
    it = out["interpret"]
    L.append("\n### H-NN4 sparse autoencoder on the frozen MLP's last hidden layer\n")
    sa = it["sae"]
    L.append(f"Frozen model: month {it['frozen']['month']} member 0, hidden {it['frozen']['hidden']}, last training label {it['frozen']['train_last_label_end']}; "
             f"holdout QLIKE {it['frozen']['holdout_qlike']:.4f}. SAE dictionary {sa['dict_size']} (4 x {sa['width']}), L1 {sa['l1']} "
             f"(selection: " + ", ".join(f"{r['l1']}: FVU {r['fvu_val']:.3f}, L0 {r['l0_val']:.1f}" for r in sa["l1_selection"]) + "); "
             f"in-sample FVU {sa['metrics_in']['fvu']:.3f}, L0 {sa['metrics_in']['l0']:.2f}; holdout FVU {sa['metrics_ho']['fvu']:.3f}, L0 {sa['metrics_ho']['l0']:.2f}; "
             f"alive {sa['n_alive_in']} of {sa['dict_size']}.\n")
    L.append("| feature | freq in | freq ho | best state (holdout) | abs rho | ties | top inputs (rho) | d log RV per unit |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in it["corr_holdout"]:
        ti = ", ".join(f"{a} {b:+.2f}" for a, b in r["top_inputs"])
        L.append(f"| {r['feature']} | {r['freq_in']:.2f} | {r['freq_ho']:.2f} | {r['best_state']} | {r['best_abs_rho']:.2f} | {'yes' if r['ties'] else 'no'} | {ti} | {r['dlogrv_per_unit']:+.3f} |")
    L.append(f"\nTies: {it['n_ties_holdout']} / 10 on holdout ({it['n_ties_in_sample']} / 10 in-sample).\n")
    L.append("| segment | ablation | dQLIKE | NW t | intact QLIKE |")
    L.append("|---|---|---|---|---|")
    for s in ("clean_holdout", "llm_in_sample"):
        for k in ("top3", "top10"):
            r = it["ablation"][s][k]
            L.append(f"| {s} | {k} | {r['mean']:+.5f} | {_f(r['t'], 2)} | {r['qlike_intact']:.4f} |")
    L.append("\nLinear probes (R^2 on holdout, ridge fit in-sample):\n")
    sv = list(STATE_VARS)
    L.append("| layer | " + " | ".join(sv) + " |")
    L.append("|---|" + "---|" * len(sv))
    for lname, pr_ in it["probes"].items():
        L.append(f"| {lname} | " + " | ".join(_f(pr_[k], 2) for k in sv) + " |")
    at = out["attributions"]
    L.append("\n### Do trees and networks agree? (mean |attribution|, out-of-sample)\n")
    L.append("| target | segment | Spearman (52 features) | top-10 overlap | group Spearman | xgb top 5 | mlp top 5 |")
    L.append("|---|---|---|---|---|---|---|")
    for task in ("vol", "dir"):
        for s in SEGS:
            r = at[task][s]
            L.append(f"| {task} | {s} | {_f(r['spearman_feature_ranking'], 2)} | {r['top10_overlap']} | {_f(r['group_spearman'], 2)} | "
                     + ", ".join(a for a, _ in r["xgb_top10"][:5]) + " | " + ", ".join(a for a, _ in r["mlp_top10"][:5]) + " |")
    L.append("\nGrouped attribution shares (xgb TreeSHAP / mlp IG), clean_holdout:\n")
    L.append("| group | vol xgb | vol mlp | dir xgb | dir mlp |")
    L.append("|---|---|---|---|---|")
    for g in GROUPS:
        a_ = at["vol"]["clean_holdout"]["groups"][g]
        b_ = at["dir"]["clean_holdout"]["groups"][g]
        L.append(f"| {g} | {100 * a_['xgb_shap']:.1f}% | {100 * a_['mlp_ig']:.1f}% | {100 * b_['xgb_shap']:.1f}% | {100 * b_['mlp_ig']:.1f}% |")
    ga = out["group_ablation"]
    L.append("\n### Group ablation (inputs set to the training mean): QLIKE increase (rel., NW t) for vol\n")
    L.append("| group | " + " | ".join(f"{fam} {s}" for s in SEGS for fam in ("mlp", "xgb")) + " |")
    L.append("|---|" + "---|" * 4)
    for g in GROUPS:
        cells = []
        for s in SEGS:
            for fam in ("mlp", "xgb"):
                r = ga["vol"][s][f"{fam}:{g}"]
                cells.append(f"{100 * r['rel']:+.1f}% ({_f(r['t'], 1)})")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("\nDirection: Brier-skill change when a group is ablated (negative = the group carried skill):\n")
    L.append("| group | " + " | ".join(f"{fam} {s}" for s in SEGS for fam in ("mlp", "xgb")) + " |")
    L.append("|---|" + "---|" * 4)
    for g in GROUPS:
        L.append(f"| {g} | " + " | ".join(f"{ga['dir'][s][f'{fam}:{g}']['dskill']:+.5f}" for s in SEGS for fam in ("mlp", "xgb")) + " |")
    ex = out.get("exploratory_har_dow")
    if ex:
        L.append("\n### Exploratory (post hoc, not a verdict): per-asset log-HAR + day-of-week dummies\n")
        L.append("| segment | pair | diff | rel | NW t |")
        L.append("|---|---|---|---|---|")
        for s in SEGS:
            for k, r in ex[s].items():
                L.append(f"| {s} | {k.replace('_vs_', ' - ')} | {_f(r['diff'])} | {100 * r['rel']:+.1f}% | {_f(r['t'], 2)} |")
    ce = out.get("calendar_effect")
    if ce:
        L.append("\n### Calendar contribution to log RV forecast (xgb TreeSHAP, mean by forecast weekday; weekend gap by v0 vol state)\n")
        L.append("| segment | Mon | Tue | Wed | Thu | Fri | Sat | Sun | weekend gap low / normal / high vol |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s in SEGS:
            w = ce[s]["by_weekday"]
            g = ce[s]["weekend_by_vol_state"]
            L.append(f"| {s} | " + " | ".join(f"{w.get(d, float('nan')):+.3f}" for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
                     + f" | {g['low']['gap']:+.3f} / {g['normal']['gap']:+.3f} / {g['high']['gap']:+.3f} |")
    dp = out.get("dependence")
    if dp:
        L.append("\n### Direction of effects (Spearman of feature value vs its TreeSHAP value; clean_holdout, in-sample in brackets)\n")
        for task in ("vol", "dir"):
            ins = {r["feature"]: r for r in dp[task]["llm_in_sample"]}
            L.append(f"- {task}: " + "; ".join(f"{r['feature']} {r['rho_value_shap']:+.2f} [{ins[r['feature']]['rho_value_shap']:+.2f}]" if r["feature"] in ins
                                              else f"{r['feature']} {r['rho_value_shap']:+.2f}" for r in dp[task]["clean_holdout"][:12]))
        L.append(f"- vol: joint btc_ret_1d_z + mkt_ret_1d_z contribution vs |mkt_ret_1d_z|: rho {dp['vol']['clean_holdout_market_abs']:+.2f} holdout, "
                 f"{dp['vol']['llm_in_sample_market_abs']:+.2f} in-sample")
    L.append("\n### Verdicts (after Holm within NN)\n")
    L.append("| hypothesis | llm_in_sample | clean_holdout | pooled | Holm p (in / ho) |")
    L.append("|---|---|---|---|---|")
    for h, v in out["verdicts"].items():
        L.append(f"| {h} | {v['llm_in_sample']} | {v['clean_holdout']} | {v['pooled']} | {_f(out['holm']['llm_in_sample']['p_holm'][h], 3)} / {_f(out['holm']['clean_holdout']['p_holm'][h], 3)} |")
    return "\n".join(L) + "\n"


DATA = ("bybit 1h + 4h bars, universe L (23 crypto perps), market_frame; funding / 4h OI via Lake.read_asof (known_ts <= T); "
        "ai_trader.ml.features (52 point-in-time features); src/ai_trader/research/ml_signals_study.py; data/models/ml/")


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record, results

    v = out["verdicts"]
    for h in ("H-NN1", "H-NN2", "H-NN3", "H-NN4"):
        done = {r["segment"] for r in results(h)}
        for seg in (*SEGS, "pooled"):
            if seg in done:
                print(f"skip {h} {seg}: already recorded")
                continue
            m, note = _metrics(out, h, seg)
            record(h, seg, v[h][seg], m, note=note, data=DATA)


def _metrics(out: dict, h: str, seg: str) -> tuple[dict, str]:
    r = out[h]
    raw = r.get("pooled_verdict") if seg == "pooled" else r[seg]["verdict"]
    if h == "H-NN1":
        if seg == "pooled":
            return ({s: {k: r[s][f"{k}_vs_har"] for k in ("mlp", "xgb", "lin")} for s in SEGS} | {"winners_both_segments": r["winners_both_segments"]},
                    f"registry criterion: the same nonlinear model with QLIKE diff < 0, t < -2 in both segments; raw {raw}")
        m = {k: r[seg][f"{k}_vs_har"] for k in ("mlp", "xgb", "lin")} | {"vs_har_qs": {k: r[seg][f"{k}_vs_har_qs"] for k in ("mlp", "xgb")},
             "mlp_vs_xgb": r[seg]["mlp_vs_xgb"], "xgb_vs_lin": r[seg]["xgb_vs_lin"], "mean_qlike": r[seg]["mean_qlike"], "p": r[seg]["p"],
             "p_holm": out["holm"][seg]["p_holm"][h]}
        return m, (f"next-day RV, per-asset log-HAR baseline (exp(xb+s2/2)); expanding monthly purged refits; validation-block bias; "
                   f"date-averaged QLIKE diff, NW t; raw {raw}")
    if h == "H-NN2":
        if seg == "pooled":
            return {s: {k: {"skill": r[s][k]["skill"], "ci95": r[s][k]["ci95"]} for k in ("mlp", "xgb")} for s in SEGS}, f"both segments; raw {raw}"
        m = {k: r[seg][k] for k in ("mlp", "xgb", "lin")} | {"p": r[seg]["p"], "p_holm": out["holm"][seg]["p_holm"][h]}
        return m, f"sign of next-24h return; monthly purged (24h + 1d) refits; Brier skill vs training base rate, 10-day block bootstrap; raw {raw}"
    if h == "H-NN3":
        if seg == "pooled":
            return ({"top5_overlap": r["top5_overlap"], "overlap_boot_mean": r["overlap_boot_mean"], "p_overlap_lt3": r["p_overlap_lt3"],
                     "share": {s: r[s]["interaction_share"] for s in SEGS}, "top_pairs": {s: r[s]["top_pairs"] for s in SEGS}, "p": r["p_pooled"]},
                    f"share > 10% in both segments and top-5 pair overlap >= 3; raw {raw}")
        m = {k: r[seg][k] for k in ("n_days", "interaction_share", "share_ci95", "p_share_le_010", "top_pairs", "top_main", "group_main_share")}
        m["p_holm"] = out["holm"][seg]["p_holm"][h]
        return m, f"xgb vol model, TreeSHAP interaction values on each month's out-of-sample rows; 10-day block bootstrap; raw {raw}"
    it = out["interpret"]
    if seg == "pooled":
        return {"n_ties_holdout": it["n_ties_holdout"], "ablation_top3_holdout": it["ablation"]["clean_holdout"]["top3"]}, f"= clean_holdout (registry criterion); raw {raw}"
    tab = it["corr_holdout"] if seg == "clean_holdout" else it["corr_in_sample"]
    m = {"n_ties": r[seg]["n_ties"], "ablation_top3": r[seg]["ablation_top3"], "ablation_top10": it["ablation"][seg]["top10"],
         "features": [{"f": x["feature"], "freq": x["freq_in"], "best_state": x["best_state"], "rho": x["best_abs_rho"]} for x in tab],
         "sae": {"l1": it["sae"]["l1"], "dict_size": it["sae"]["dict_size"], "fvu_ho": it["sae"]["metrics_ho"]["fvu"], "l0_ho": it["sae"]["metrics_ho"]["l0"]},
         "p_holm": out["holm"][seg]["p_holm"][h]}
    note = ("SAE fit on in-sample activations of the frozen MLP; holdout correlations (Spearman) and error-preserving ablation" if seg == "clean_holdout"
            else "descriptive: same criteria on the SAE's own fitting sample")
    return m, f"{note}; raw {raw}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "search", "walk", "interpret", "evaluate"))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(md.TORCH_THREADS)
    if a.cmd == "build":
        full, c = load_panel()
        print(full.shape, c.shape)
    elif a.cmd == "search":
        run_search()
    elif a.cmd == "walk":
        run_walk(a.limit or None)
    elif a.cmd == "interpret":
        r = run_interpret()
        print(json.dumps({k: r[k] for k in ("n_ties_holdout", "n_ties_in_sample", "top10")}), json.dumps(r["ablation"]["clean_holdout"]["top3"]))
    else:
        run_evaluate(a.record)


if __name__ == "__main__":
    main()
