"""ML signals round 2 (program 2026-11-12, agent B): H-NN8 (nonlinear next-hour volatility vs robust_bc x hour-of-week)
and H-NN9 (64-unit last hidden layer + 1x-2x sparse autoencoder). Deterministic research tools only (principle 1).

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.ml_signals_round2_study build
    ... search8      in-sample-only config search for the hourly models (rows before the first test month)
    ... walk8        hourly walk-forward (H-NN8)
    ... walk9        daily-panel walk-forward of the 64-unit vol MLP (H-NN9; same protocol as H-NN1)
    ... interpret9   frozen 64-unit MLP: SAE size / L1 selection in-sample, holdout correlations and ablation
    ... evaluate [--record]   verdicts, Holm within round 2, research/reports/ml_signals_round2_2026-11-12.{json,md}

H-NN8 design (fixed before evaluation; diary 'Round-2 ML design fixed'):
  Data: the cached H-NS4 1h frames (data/research/new_strategies/f1h_<SYMBOL>.parquet, market_frame as of 2026-11-10),
  universe L, read only. Features ml.hourly.HOURLY_FEATURES (19); target log(r_t^2 / s2_t).
  Champion: s2_t x fac_t (robust_bc x the expanding H-NS4 hour-of-week factor). Context: s2_t alone.
  Walk-forward: monthly, expanding; training rows = bars opening before month start - 1 day; first test month = the
  first with >= 365 distinct training dates. RUNTIME: training rows are a deterministic random 1/3 subsample
  (hourly.train_subsample, seed 20261112; a fixed every-3rd-hour rule would alias with the 24 h cycle); test rows are
  ALL bars of the month. Inside each window: models.val_split (last 15% of training dates, purged 1 day), fit-block
  standardisation, early stopping, validation-block variance bias c (QLIKE-optimal constant), F = s2 exp(f + c).
  MLP: 2 ReLU layers, 3 seeds, AdamW, batch 2048, lr 2e-3 (larger batch for the 24x bigger panel), MSE.
  XGB: hist, eta 0.03, subsample / colsample 0.8, early stopping 50 / max 2000.
  Equal tuning effort: the round-1 grids (MLP hidden {(64,32),(32,16)} x wd {1e-4,1e-2}; XGB depth {3,6} x mcw {20,200})
  chosen on the subsampled rows whose bars open before the first test month - 1 day, 80/20 date split, validation MSE.
  Reference: ridge-linear on the same features.
  Loss: per-bar QLIKE difference (model - champion) = r^2 (1/F_a - 1/F_b) + log(F_a/F_b), mean over symbol-bars per UTC
  date, NW t over dates, per segment (bar open before / after the cutoff).
  Verdict per segment: supports if mlp or xgb has diff < 0 and t < -2; contradicts if neither has diff < 0 or the better
  one has t > 2; else inconclusive. Pooled (registry): supports iff the same model supports in both segments;
  contradicts if a segment contradicts. p = min over the two models of 2 Phi(t).
  Descriptive: TreeSHAP group shares (2,000 random test rows per month), group ablation (inputs at the training mean).

H-NN9 design (fixed before evaluation):
  Daily panel, inputs, months, purge, validation, bias and early stopping exactly as H-NN1 (ml_signals_study). Only the
  network changes: hidden (64, 64) (a 64-unit last layer after a 64-unit first layer: a 32-unit first layer would cap
  the last layer's pre-activations at rank 32), weight decay 1e-2 (the H-NN1 vol choice), 3 seeds. Frozen model = the
  seed-0 member for the first holdout month (2026-07; labels to 2026-06-30).
  SAE (interpret.fit_sae): dictionary size in {64 (1x), 128 (2x)}; for each size L1 = the largest of {0.03, 0.1, 0.3,
  1.0} with FVU <= 0.10 on the last 20% of in-sample dates (SAE fit on the first 80%); the size is then chosen by the
  number of its 10 most-used features (firing frequency on the 80% block) with |Spearman| > 0.6 to a named state on
  the 20% in-sample block (ties: lower FVU). In-sample only. The chosen (size, L1) is refit on all in-sample activations.
  Holdout: top 10 by in-sample firing frequency, |Spearman| with the H-NN4 state variables on clean_holdout rows;
  error-preserving ablation of the top 3, dQLIKE averaged per date, NW t.
  clean_holdout: supports iff >= 6 of 10 tie AND dQLIKE > 0 with t > 2; contradicts if < 6 tie or dQLIKE <= 0; else
  inconclusive. llm_in_sample row: the same criteria on in-sample activations (descriptive). Pooled = clean_holdout.
  p = Phi(-t) if >= 6 tie else 1. Also reported: walk-forward QLIKE of the 64-unit ensemble vs the 16-unit H-NN1 MLP.

Holm within round 2 per segment over H-NN8 and H-NN9; a 'supports' with Holm p > 0.05 becomes inconclusive.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import interpret as ip
from ai_trader.ml import models as md
from ai_trader.ml.features import ML_FEATURES, STATE_VARS, Standardizer
from ai_trader.ml.hourly import HOURLY_FEATURES, HOURLY_GROUPS, complete_hourly, qlike_diff, symbol_hourly, train_subsample
from ai_trader.research import ml_signals_study as r1
from ai_trader.research.decision_budget_study import holm, purged_train_mask, test_mask

TAG = "2026-11-12"
SEGS = ("llm_in_sample", "clean_holdout")
DAY_US = np.timedelta64(86400 * 1_000_000, "us")
HOUR_US = np.timedelta64(3600 * 1_000_000, "us")
MLP8 = {"batch": 2048, "lr": 2e-3}
MLP9 = {"hidden": (64, 64), "weight_decay": 1e-2}
SAE_SIZES = (64, 128)
SHAP_ROWS = 2000
F8 = len(HOURLY_FEATURES)


def root() -> Path:
    d = load().root / "data" / "models" / "ml" / "round2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cut() -> np.datetime64:
    return np.datetime64(load().settings["research"]["knowledge_cutoff"] + "T00:00:00", "us")


# ------------------------------------------------------------------ H-NN8 data


def hourly_path() -> Path:
    return root() / f"hourly_{TAG}.parquet"


def build_hourly() -> pl.DataFrame:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    p = hourly_path()
    if p.is_file():
        return pl.read_parquet(p)
    src = load().root / "data" / "research" / "new_strategies"
    parts = []
    for s in UNIVERSE:
        fp = src / f"f1h_{s}.parquet"
        if not fp.is_file():
            print(f"missing {s}", flush=True)
            continue
        h = complete_hourly(symbol_hourly(pl.read_parquet(fp)))
        parts.append(h.with_columns(pl.lit(s).alias("symbol")))
        print(s, h.height, flush=True)
    cut = load().settings["research"]["knowledge_cutoff"]
    df = pl.concat(parts).with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    df = df.with_columns(pl.when(pl.col("event_ts") < pl.lit(cut).str.to_datetime().dt.replace_time_zone("UTC"))
                         .then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment")).sort("event_ts", "symbol")
    df.write_parquet(p)
    return df


def arrays8(df: pl.DataFrame) -> dict:
    T = df["event_ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    return {"X": df.select(HOURLY_FEATURES).to_numpy().astype(float), "T": T, "le": T + HOUR_US, "y": df["y"].to_numpy().astype(float),
            "r2": df["r2"].to_numpy().astype(float), "s2": df["s2"].to_numpy().astype(float), "fac": df["fac"].to_numpy().astype(float),
            "seg": df["segment"].to_numpy(), "sub": train_subsample(df.height)}


def first_test_month8(T: np.ndarray, le: np.ndarray) -> np.datetime64:
    return r1.first_test_month(T, le)


def run_search8() -> dict:
    df = build_hourly()
    a = arrays8(df)
    m0 = first_test_month8(a["T"], a["le"])
    pre = (a["le"] <= m0 - DAY_US) & a["sub"]
    Z = Standardizer().fit(a["X"][pre]).transform(a["X"][pre])
    out = {"first_test_month": str(m0)[:7], "n_rows": int(pre.sum())}
    fit, va = md.val_split(a["T"][pre], 0.2)
    y = a["y"][pre]
    for fam, grid in (("mlp", md.MLP_GRID), ("xgb", md.XGB_GRID)):
        sc = []
        for cfg in grid:
            t = time.time()
            if fam == "mlp":
                m = md.fit_mlp(Z[fit], y[fit], Z[va], y[va], "vol", 0, {**cfg, **MLP8})
            else:
                m = md.fit_xgb(Z[fit], y[fit], Z[va], y[va], "vol", cfg)
            p = m.predict(Z[va])
            sc.append({"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, "val_mse": float(np.mean((p - y[va]) ** 2)),
                       "secs": round(time.time() - t, 1)})
            print(fam, sc[-1], flush=True)
        best = min(range(len(grid)), key=lambda i: sc[i]["val_mse"])
        out[fam] = {"best": sc[best]["cfg"], "scores": sc}
    (root() / "search8.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _cfg8() -> dict:
    s = json.loads((root() / "search8.json").read_text(encoding="utf-8"))
    m = dict(s["mlp"]["best"])
    m["hidden"] = tuple(m["hidden"])
    return {"mlp": {**m, **MLP8}, "xgb": dict(s["xgb"]["best"])}


def run_walk8(limit: int | None = None) -> None:
    df = build_hourly()
    a = arrays8(df)
    cf = _cfg8()
    m0 = first_test_month8(a["T"], a["le"])
    months = [m for m in r1._months(a["T"]) if m >= m0]
    if limit:
        months = months[:limit]
    n = df.height
    pred = {k: np.full(n, np.nan) for k in ("F_mlp", "F_xgb", "F_lin")}
    abl = {f"{fam}_{g}": np.full(n, np.nan) for fam in ("mlp", "xgb") for g in HOURLY_GROUPS}
    gcols = {g: [HOURLY_FEATURES.index(f) for f in fs] for g, fs in HOURLY_GROUPS.items()}
    shap_sum = {s: np.zeros(F8) for s in SEGS}
    shap_n = dict.fromkeys(SEGS, 0)
    rng = np.random.default_rng(0)
    info = []
    for m in months:
        t0 = time.time()
        te = test_mask(a["T"], m, r1._next_month(m))
        if not te.any():
            continue
        tr = np.flatnonzero(purged_train_mask(a["le"], m) & a["sub"])
        fl, vl = md.val_split(a["T"][tr])
        fi, vi = tr[fl], tr[vl]
        st = Standardizer().fit(a["X"][fi])
        Zf, Zv, Zt = st.transform(a["X"][fi]), st.transform(a["X"][vi]), st.transform(a["X"][te])
        ens = md.fit_mlp_ensemble(Zf, a["y"][fi], Zv, a["y"][vi], "vol", cfg=cf["mlp"])
        xg = md.fit_xgb(Zf, a["y"][fi], Zv, a["y"][vi], "vol", cfg=cf["xgb"], names=HOURLY_FEATURES)
        ln = md.fit_linear(Zf, a["y"][fi], "vol")
        ls_v, ls_t = np.log(a["s2"][vi]), np.log(a["s2"][te])
        for name, mod in (("mlp", ens), ("xgb", xg), ("lin", ln)):
            c = md.bias_const(a["r2"][vi], ls_v + mod.predict(Zv))
            pred[f"F_{name}"][te] = np.exp(ls_t + mod.predict(Zt) + c)
            if name in ("mlp", "xgb"):
                for g, cols in gcols.items():
                    Za = Zt.copy()
                    Za[:, cols] = 0.0
                    abl[f"{name}_{g}"][te] = np.exp(ls_t + mod.predict(Za) + c)
        ti = np.flatnonzero(te)
        pick = rng.choice(ti, size=min(SHAP_ROWS, len(ti)), replace=False)
        sv = np.abs(ip.shap_main(xg, st.transform(a["X"][pick]))[:, :F8])
        for s in SEGS:
            mk = a["seg"][pick] == s
            shap_sum[s] += sv[mk].sum(0)
            shap_n[s] += int(mk.sum())
        info.append({"month": str(m)[:7], "n_train": int(len(tr)), "n_test": int(te.sum()), "mlp_epochs": [x.epochs for x in ens.members],
                     "xgb_iter": xg.best_iteration, "secs": round(time.time() - t0, 1)})
        print(json.dumps(info[-1]), flush=True)
    out = df.select("symbol", "event_ts", "segment", "r2", "s2", "fac").with_columns(
        (pl.col("s2") * pl.col("fac")).alias("F_champ"), pl.col("s2").alias("F_base"),
        *[pl.Series(k, v) for k, v in pred.items()], *[pl.Series(f"abl_{k}", v) for k, v in abl.items()])
    out.filter(pl.col("F_xgb").is_not_null() & pl.col("F_xgb").is_finite()).write_parquet(root() / "wf8_preds.parquet")
    (root() / "wf8_info.json").write_text(json.dumps({"first_test_month": str(m0)[:7], "cfg": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                                     for kk, vv in v.items()} for k, v in cf.items()}, "months": info,
                                                     "shap_mean_abs": {s: (shap_sum[s] / max(shap_n[s], 1)).tolist() for s in SEGS},
                                                     "shap_n": shap_n}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ H-NN9


def run_walk9(limit: int | None = None) -> None:
    import torch

    full, c = r1.load_panel()
    a = r1.arrays(c)
    m0 = r1.first_test_month(a["T"], a["le"])
    months = [m for m in r1._months(a["T"]) if m >= m0]
    if limit:
        months = months[:limit]
    F = np.full(c.height, np.nan)
    done = False
    info = []
    for m in months:
        t0 = time.time()
        te = test_mask(a["T"], m, r1._next_month(m))
        if not te.any():
            continue
        tr = np.flatnonzero(purged_train_mask(a["le"], m))
        fl, vl = md.val_split(a["T"][tr])
        fi, vi = tr[fl], tr[vl]
        st = Standardizer().fit(a["X"][fi])
        Zf, Zv, Zt = st.transform(a["X"][fi]), st.transform(a["X"][vi]), st.transform(a["X"][te])
        ens = md.fit_mlp_ensemble(Zf, a["ly"][fi], Zv, a["ly"][vi], "vol", cfg=MLP9)
        cst = md.bias_const(a["rv"][vi], ens.predict(Zv))
        F[te] = np.exp(ens.predict(Zt) + cst)
        if not done and m >= _cut():
            done = True
            torch.save({"member": ens.members[0].__dict__, "std": st.to_dict(), "month": str(m)[:7],
                        "bias": md.bias_const(a["rv"][vi], ens.members[0].predict(Zv)), "train_last_label_end": str(a["le"][tr].max())},
                       root() / "frozen_mlp64_vol.pt")
        info.append({"month": str(m)[:7], "epochs": [x.epochs for x in ens.members], "secs": round(time.time() - t0, 1)})
        print(json.dumps(info[-1]), flush=True)
    c.select("symbol", "T", "segment", "rv_next").with_columns(pl.Series("F_mlp64", F)).write_parquet(root() / "wf9_preds.parquet")
    (root() / "wf9_info.json").write_text(json.dumps({"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in MLP9.items()}, "months": info}, indent=1),
                                          encoding="utf-8")


def _frozen9():
    import torch

    d = torch.load(root() / "frozen_mlp64_vol.pt", weights_only=False)
    fit = md.MLPFit(**d["member"])
    st = Standardizer()
    st.mu, st.sd = np.asarray(d["std"]["mu"]), np.asarray(d["std"]["sd"])
    return fit, st, float(d["bias"]), d


def _ties(Zs: np.ndarray, sv: dict, mask: np.ndarray, top: list[int]) -> list[dict]:
    rows = []
    for k in top:
        rh = {name: ip.spearman(Zs[:, k], x[mask]) for name, x in sv.items()}
        best = max(rh, key=lambda q: abs(rh[q]) if np.isfinite(rh[q]) else -1)
        rows.append({"feature": int(k), "rho": rh, "best_state": best, "best_abs_rho": float(abs(rh[best])) if np.isfinite(rh[best]) else float("nan"),
                     "ties": bool(np.isfinite(rh[best]) and abs(rh[best]) > r1.RHO_TIE)})
    return rows


def run_interpret9() -> dict:
    _, c = r1.load_panel()
    a = r1.arrays(c)
    fit, st, bias, meta = _frozen9()
    Z = st.transform(a["X"])
    H = ip.hidden_activations(fit, Z)
    Hl = H[-1]
    ins, hol = a["seg"] == "llm_in_sample", a["seg"] == "clean_holdout"
    sv = {k: c[v].to_numpy().astype(float) for k, v in STATE_VARS.items()}
    idx_in = np.flatnonzero(ins)
    f80, v20 = md.val_split(a["T"][ins], 0.2)
    Hin = Hl[ins]
    sel = {}
    for size in SAE_SIZES:
        rows = []
        for l1 in r1.SAE_L1_GRID:
            s = ip.fit_sae(Hin[f80], size, l1=l1, steps=8000, seed=0)
            mt = s.metrics(Hin[v20])
            rows.append({"l1": l1, "fvu_val": mt["fvu"], "l0_val": mt["l0"], "dead_frac_val": mt["dead_frac"], "sae": s})
            print(size, l1, round(mt["fvu"], 4), round(mt["l0"], 2), flush=True)
        ok = [r for r in rows if r["fvu_val"] <= r1.SAE_FVU_MAX]
        ch = max(ok, key=lambda r: r["l1"]) if ok else min(rows, key=lambda r: r["l1"])
        s = ch["sae"]
        freq = s.metrics(Hin[f80])["freq"]
        top = list(np.argsort(-freq, kind="stable")[:10])
        mask_v = np.zeros(len(a["T"]), bool)
        mask_v[idx_in[v20]] = True
        n_t = sum(r["ties"] for r in _ties(s.encode(Hin[v20]), sv, mask_v, top))
        sel[size] = {"l1": ch["l1"], "fvu_val": ch["fvu_val"], "l0_val": ch["l0_val"], "ties_val": int(n_t),
                     "grid": [{k: v for k, v in r.items() if k != "sae"} for r in rows]}
        print("size", size, sel[size]["l1"], n_t, flush=True)
    size = max(SAE_SIZES, key=lambda z: (sel[z]["ties_val"], -sel[z]["fvu_val"]))
    l1 = sel[size]["l1"]
    sae = ip.fit_sae(Hin, size, l1=l1, steps=12000, seed=0)
    met_in, met_ho = sae.metrics(Hin), sae.metrics(Hl[hol])
    freq = met_in["freq"]
    top = list(np.argsort(-freq, kind="stable")[:10])
    Zin, Zho = sae.encode(Hin), sae.encode(Hl[hol])
    tab_ho, tab_in = _ties(Zho, sv, hol, top), _ties(Zin, sv, ins, top)
    w_head = fit.state["head.weight"].numpy().astype(float).ravel() * fit.y_sd
    for tab, Zs, mask in ((tab_ho, Zho, hol), (tab_in, Zin, ins)):
        for r in tab:
            k = r["feature"]
            inp = {f: ip.spearman(Zs[:, k], a["X"][mask, j]) for j, f in enumerate(ML_FEATURES)}
            r["top_inputs"] = [[f, inp[f]] for f in sorted(inp, key=lambda q: -abs(inp[q]) if np.isfinite(inp[q]) else 0)[:3]]
            r["freq_in"] = float(freq[k])
            r["freq_seg"] = float((Zs[:, k] > 0).mean())
            r["dlogrv_per_unit"] = float(w_head @ sae.atom(k))

    def ablate(mask, Zs, feats):
        Hm = Hl[mask]
        base = np.exp(ip.head_from_last(fit, Hm) + bias)
        Hm2 = Hm - sum(np.outer(Zs[:, k], sae.atom(k)) for k in feats)
        d = md.qlike(a["rv"][mask], np.exp(ip.head_from_last(fit, Hm2) + bias)) - md.qlike(a["rv"][mask], base)
        out_ = r1._date_mean_diff(a["T"][mask], d)
        out_["qlike_intact"] = float(np.nanmean(md.qlike(a["rv"][mask], base)))
        return out_

    abl = {s: {"top3": ablate(m, Zs, top[:3]), "top10": ablate(m, Zs, top), "each": {int(k): ablate(m, Zs, [k]) for k in top}}
           for s, m, Zs in (("clean_holdout", hol, Zho), ("llm_in_sample", ins, Zin))}
    probes = {}
    for lname, Hx in {"input": Z, **{f"hidden{i + 1}": h for i, h in enumerate(H)}}.items():
        probes[lname] = {}
        for name, x in sv.items():
            okr = np.isfinite(x)
            probes[lname][name] = ip.ridge_probe(Hx[ins & okr], x[ins & okr], Hx[hol & okr], x[hol & okr])
    probes["sae_codes"] = {name: ip.ridge_probe(Zin, x[ins], Zho, x[hol]) for name, x in sv.items()}
    # effective dimensionality of the last layer (participation ratio of the activation covariance), vs the 16-unit model
    def pr_dim(Hx):
        ev = np.clip(np.linalg.eigvalsh(np.cov(Hx.T)), 0, None)
        return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else float("nan")

    f16, st16, _, _ = r1._frozen()
    H16 = ip.hidden_activations(f16, st16.transform(a["X"]))[-1]
    out = {"frozen": {"month": meta["month"], "train_last_label_end": meta["train_last_label_end"], "hidden": list(fit.hidden), "bias": bias,
                      "holdout_qlike": float(np.nanmean(md.qlike(a["rv"][hol], np.exp(ip.head_from_last(fit, Hl[hol]) + bias)))),
                      "dead_units_in": int(np.sum(Hin.max(0) <= 0)), "participation_ratio": pr_dim(Hin), "participation_ratio_16": pr_dim(H16[ins])},
           "selection": sel, "sae": {"dict_size": size, "l1": l1, "metrics_in": {k: v for k, v in met_in.items() if k != "freq"},
                                     "metrics_ho": {k: v for k, v in met_ho.items() if k != "freq"}, "n_alive_in": int((freq > 0).sum()),
                                     "resampled": sae.history["resampled"]},
           "top10": [int(k) for k in top], "corr_holdout": tab_ho, "corr_in_sample": tab_in,
           "n_ties_holdout": int(sum(r["ties"] for r in tab_ho)), "n_ties_in_sample": int(sum(r["ties"] for r in tab_in)),
           "ablation": abl, "probes": probes}
    (root() / "sae64_vol.json").write_text(json.dumps(sae.to_dict()), encoding="utf-8")
    (root() / "interpret9.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


# ------------------------------------------------------------------ evaluate


def _phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2)) if np.isfinite(t) else 1.0


def paired8(q: pl.DataFrame, a: str, b: str) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    d = qlike_diff(q["r2"].to_numpy(), q[a].to_numpy(), q[b].to_numpy())
    ok = np.isfinite(d)
    g = pl.DataFrame({"d": q["event_ts"].dt.date().to_numpy()[ok], "x": d[ok]}).group_by("d").agg(pl.col("x").mean()).sort("d")
    m, t, nd = newey_west_t(g["x"].to_numpy())
    return {"diff": float(m), "t": float(t), "n_dates": int(nd), "n_obs": int(ok.sum())}


def nn8(pr: pl.DataFrame, info: dict) -> dict:
    out: dict = {"n_rows": pr.height}
    for s in SEGS:
        q = pr.filter(pl.col("segment") == s)
        r = {f"{m}_vs_champ": paired8(q, f"F_{m}", "F_champ") for m in ("mlp", "xgb", "lin")}
        r["champ_vs_base"] = paired8(q, "F_champ", "F_base")
        r["mlp_vs_xgb"] = paired8(q, "F_mlp", "F_xgb")
        r["xgb_vs_lin"] = paired8(q, "F_xgb", "F_lin")
        r["n_symbols"] = q["symbol"].n_unique()
        nl = {m: r[f"{m}_vs_champ"] for m in ("mlp", "xgb")}
        if any(v["diff"] < 0 and v["t"] < -2 for v in nl.values()):
            v = "supports"
        elif all(not (v["diff"] < 0) for v in nl.values()) or min(nl.values(), key=lambda z: z["diff"])["t"] > 2:
            v = "contradicts"
        else:
            v = "inconclusive"
        r["verdict"] = v
        r["p"] = float(min(1.0, min(2 * _phi(nl[m]["t"]) for m in nl)))
        r["group_ablation"] = {}
        for fam in ("mlp", "xgb"):
            for g in HOURLY_GROUPS:
                rr = paired8(q.filter(pl.col(f"abl_{fam}_{g}").is_finite()), f"abl_{fam}_{g}", f"F_{fam}")
                r["group_ablation"][f"{fam}:{g}"] = {"dqlike": rr["diff"], "t": rr["t"]}
        sh = np.asarray(info["shap_mean_abs"][s])
        r["shap_group_share"] = {g: float(sh[[HOURLY_FEATURES.index(f) for f in fs]].sum() / sh.sum()) for g, fs in HOURLY_GROUPS.items()}
        r["shap_top"] = sorted(((HOURLY_FEATURES[i], float(sh[i] / sh.sum())) for i in range(F8)), key=lambda z: -z[1])[:8]
        out[s] = r
    win = [m for m in ("mlp", "xgb") if all(out[s][f"{m}_vs_champ"]["diff"] < 0 and out[s][f"{m}_vs_champ"]["t"] < -2 for s in SEGS)]
    vs = [out[s]["verdict"] for s in SEGS]
    out["winners_both_segments"] = win
    out["pooled_verdict"] = "supports" if win else ("contradicts" if "contradicts" in vs else "inconclusive")
    return out


def nn9(it: dict, pr9: pl.DataFrame) -> dict:
    out: dict = {}
    pr1 = pl.read_parquet(r1.root() / "wf_preds.parquet").select("symbol", "T", "F_mlp", "F_har", "F_xgb")
    j = pr9.join(pr1, on=["symbol", "T"], how="inner").filter(
        pl.all_horizontal([pl.col(x).is_finite() for x in ("F_mlp64", "F_mlp", "F_har", "F_xgb")]))
    for s in SEGS:
        n_t = it["n_ties_holdout"] if s == "clean_holdout" else it["n_ties_in_sample"]
        ab = it["ablation"][s]["top3"]
        if n_t >= 6 and ab["mean"] > 0 and ab["t"] > 2:
            v = "supports"
        elif n_t < 6 or not ab["mean"] > 0:
            v = "contradicts"
        else:
            v = "inconclusive"
        q = j.filter(pl.col("segment") == s)
        out[s] = {"n_ties": n_t, "ablation_top3": ab, "verdict": v, "p": float(_phi(-ab["t"])) if n_t >= 6 else 1.0,
                  "mlp64_vs_mlp16": r1.paired(q, "F_mlp64", "F_mlp"), "mlp64_vs_har": r1.paired(q, "F_mlp64", "F_har"),
                  "mlp64_vs_xgb": r1.paired(q, "F_mlp64", "F_xgb")}
    out["pooled_verdict"] = out["clean_holdout"]["verdict"]
    return out


def _f(x, k=4):
    try:
        return f"{float(x):.{k}f}"
    except (TypeError, ValueError):
        return str(x)


def run_evaluate(record_rows: bool = False) -> dict:
    pr8 = pl.read_parquet(root() / "wf8_preds.parquet")
    info8 = json.loads((root() / "wf8_info.json").read_text(encoding="utf-8"))
    s8 = json.loads((root() / "search8.json").read_text(encoding="utf-8"))
    it = json.loads((root() / "interpret9.json").read_text(encoding="utf-8"))
    pr9 = pl.read_parquet(root() / "wf9_preds.parquet")
    info9 = json.loads((root() / "wf9_info.json").read_text(encoding="utf-8"))
    out = {"cutoff": load().settings["research"]["knowledge_cutoff"], "hourly_features": list(HOURLY_FEATURES),
           "hourly_groups": {k: list(v) for k, v in HOURLY_GROUPS.items()}, "search8": s8,
           "walk8": {"first_test_month": info8["first_test_month"], "n_months": len(info8["months"]), "cfg": info8["cfg"],
                     "secs_total": sum(m["secs"] for m in info8["months"]), "mean_train_rows_last": info8["months"][-1]["n_train"],
                     "mean_epochs": float(np.mean([np.mean(m["mlp_epochs"]) for m in info8["months"]])),
                     "mean_xgb_iter": float(np.mean([m["xgb_iter"] for m in info8["months"]]))},
           "panel8": {"rows_tested": pr8.height, "symbols": pr8["symbol"].n_unique(), "rows_by_segment": dict(pr8.group_by("segment").len().iter_rows()),
                      "first": str(pr8["event_ts"].min()), "last": str(pr8["event_ts"].max())},
           "walk9": {"cfg": info9["cfg"], "n_months": len(info9["months"]), "secs_total": sum(m["secs"] for m in info9["months"]),
                     "mean_epochs": float(np.mean([np.mean(m["epochs"]) for m in info9["months"]]))}}
    out["H-NN8"] = nn8(pr8, info8)
    out["H-NN9"] = nn9(it, pr9)
    out["interpret9"] = it
    out["holm"] = {s: {"p": {h: out[h][s]["p"] for h in ("H-NN8", "H-NN9")}} for s in SEGS}
    for s in SEGS:
        out["holm"][s]["p_holm"] = holm(out["holm"][s]["p"])
    out["verdicts"] = {}
    for h in ("H-NN8", "H-NN9"):
        vs = {s: ("inconclusive" if out[h][s]["verdict"] == "supports" and out["holm"][s]["p_holm"][h] > 0.05 else out[h][s]["verdict"]) for s in SEGS}
        pv = out[h]["pooled_verdict"]
        segs_p = SEGS if h == "H-NN8" else ("clean_holdout",)
        if pv == "supports" and max(out["holm"][s]["p_holm"][h] for s in segs_p) > 0.05:
            pv = "inconclusive"
        vs["pooled"] = pv
        out["verdicts"][h] = vs
    rep = load().root / "research" / "reports"
    (rep / f"ml_signals_round2_{TAG}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    from ai_trader.research.ml_signals_round2_report import build as build_md

    (rep / f"ml_signals_round2_{TAG}.md").write_text(build_md(out), encoding="utf-8")
    print(json.dumps(out["verdicts"], indent=1))
    if record_rows:
        record_all(out)
    return out


DATA8 = ("bybit 1h bars, universe L (23 crypto perps), cached H-NS4 frames data/research/new_strategies/f1h_*.parquet (market_frame as of "
         "2026-11-10); ai_trader.ml.hourly; src/ai_trader/research/ml_signals_round2_study.py; data/models/ml/round2/")
DATA9 = ("daily panel of H-NN1 (ai_trader.ml.features, 52 features, universe L); src/ai_trader/research/ml_signals_round2_study.py; "
         "data/models/ml/round2/")


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record, results

    v = out["verdicts"]
    for h in ("H-NN8", "H-NN9"):
        done = {r["segment"] for r in results(h)}
        for s in (*SEGS, "pooled"):
            if s in done:
                print(f"skip {h} {s}")
                continue
            r = out[h]
            raw = r["pooled_verdict"] if s == "pooled" else r[s]["verdict"]
            if h == "H-NN8":
                if s == "pooled":
                    m = {x: {k: r[x][f"{k}_vs_champ"] for k in ("mlp", "xgb", "lin")} for x in SEGS} | {"winners_both_segments": r["winners_both_segments"]}
                    note = f"registry criterion: the same model with diff < 0, t < -2 in both segments; raw {raw}"
                else:
                    m = {k: r[s][f"{k}_vs_champ"] for k in ("mlp", "xgb", "lin")} | {"champ_vs_base": r[s]["champ_vs_base"],
                         "shap_group_share": r[s]["shap_group_share"], "p": r[s]["p"], "p_holm": out["holm"][s]["p_holm"][h]}
                    note = (f"next-hour QLIKE vs robust_bc x H-NS4 hour-of-week factor; monthly expanding refits, training on a random 1/3 of "
                            f"bars (runtime), all bars tested; per-bar diff averaged per date, NW t; raw {raw}")
                record(h, s, v[h][s], m, note=note, data=DATA8)
            else:
                it = out["interpret9"]
                if s == "pooled":
                    m = {"n_ties_holdout": it["n_ties_holdout"], "ablation_top3_holdout": it["ablation"]["clean_holdout"]["top3"]}
                    note = f"= clean_holdout (registry criterion); raw {raw}"
                else:
                    tab = it["corr_holdout"] if s == "clean_holdout" else it["corr_in_sample"]
                    m = {"n_ties": r[s]["n_ties"], "ablation_top3": r[s]["ablation_top3"], "mlp64_vs_mlp16": r[s]["mlp64_vs_mlp16"],
                         "mlp64_vs_har": r[s]["mlp64_vs_har"], "sae": {"dict_size": it["sae"]["dict_size"], "l1": it["sae"]["l1"],
                         "fvu_ho": it["sae"]["metrics_ho"]["fvu"], "l0_ho": it["sae"]["metrics_ho"]["l0"]},
                         "features": [{"f": x["feature"], "best_state": x["best_state"], "rho": x["best_abs_rho"]} for x in tab],
                         "p_holm": out["holm"][s]["p_holm"][h]}
                    note = ("64-unit last layer, SAE size and L1 chosen in-sample; holdout Spearman and error-preserving ablation" if s == "clean_holdout"
                            else "descriptive: same criteria on the SAE's fitting sample") + f"; raw {raw}"
                record(h, s, v[h][s], m, note=note, data=DATA9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "search8", "walk8", "walk9", "interpret9", "evaluate"))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(md.TORCH_THREADS)
    if a.cmd == "build":
        df = build_hourly()
        print(df.shape, df.group_by("segment").len())
    elif a.cmd == "search8":
        run_search8()
    elif a.cmd == "walk8":
        run_walk8(a.limit or None)
    elif a.cmd == "walk9":
        run_walk9(a.limit or None)
    elif a.cmd == "interpret9":
        r = run_interpret9()
        print(json.dumps({k: r[k] for k in ("n_ties_holdout", "n_ties_in_sample", "top10")}), json.dumps(r["ablation"]["clean_holdout"]["top3"]))
    else:
        run_evaluate(a.record)


if __name__ == "__main__":
    main()
