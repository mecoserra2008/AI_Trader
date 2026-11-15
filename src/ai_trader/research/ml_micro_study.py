"""H-NN5 / H-NN6: a 1D CNN on 1 s order-flow bars for next-minute volatility and direction (program 2026-11-12, agent B,
round 4). Deterministic research tools only (principle 1).

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.ml_micro_study freeze-days
    ... fit          leave-days-out folds: train, predict, attributions (one process; ~1 h CPU)
    ... evaluate [--record]   verdicts, Holm (NN5, NN6), research/reports/ml_micro_2026-11-12.{json,md}

Design (fixed before evaluation; diary 'Micro CNN design fixed'):
  Days: the verified days of data/research/micro_days at the start of this round, frozen in data/models/ml/micro/days.json
  (read only; agent A keeps adding days, and they are NOT used). Symbol-days with >= 10,000 traded seconds, all symbols
  pooled; inputs are normalised per symbol-day by trailing-1h statistics (ml.micro).
  Folds: the sorted days are split round-robin into 3 folds (day i -> fold i mod 3). For each fold, train on the other
  days and test on its days. Validation (early stopping and the variance bias) = the last training day in date order.
  NOTE: leave-days-out, not forward. Training days can lie after test days, as registered ('train on some days, test on
  others').
  RUNTIME: training windows are taken every 30 s, and each epoch draws 100,000 of them at random (patience 2, one CNN seed:
  cut from 200k / patience 3 / 2 seeds after the first model took 16 min, before any test result was seen); test windows every
  10 s; ablation on test windows every 60 s; integrated gradients on 2,000 random test windows per fold.
  Models:
    * cnn: ml.micro.make_cnn, 4 dilated conv layers (12 -> 16 channels, kernel 3, dilation 1/2/4/8) over the last 300 s,
      pooled (mean of the last 10 / 60 / 300 positions, max), MLP head (32), plus a linear skip on the 36 HAR aggregates;
      1 seed;
    * lin: the linear HAR-style model, i.e. the same skip alone (the same inputs as 10 / 60 / 300 s channel means);
    * same optimiser for both (Adam lr 2e-3, wd 1e-4, batch 1024, max 20 epochs, patience 2 on the validation day),
      HAR aggregates standardised with training statistics.
  H-NN5 target r1m^2; forecast F = 60 b_s exp(f + c), c = validation-day QLIKE-optimal constant; training loss QLIKE
  (r^2 / (base e^f) + f). Per held-out symbol-day: mean per-window QLIKE difference cnn - lin = r^2 (1/F_cnn - 1/F_lin)
  + log(F_cnn / F_lin) (no log r^2 term, so zero returns are fine). Verdict per segment (test days before 2026-07-01 =
  llm_in_sample, else clean_holdout; pooled = all): supports iff >= 70% of symbol-days have difference < 0 AND the mean
  over symbol-days has t < -2; contradicts if the mean >= 0 or the share < 50%; else inconclusive.
  p = max(one-sided sign-test p vs 0.5, Phi(t)). Also reported: day-clustered t, both models vs the 60 b_s baseline.
  H-NN6 target 1{r1m > 0} on windows with r1m != 0; BCE; AUC pooled over the segment's windows, 95% CI from a bootstrap
  over symbol-days (1,000; binned over 2,000 quantile bins of p for speed, point AUC exact). Supports iff the CI upper bound <= 0.52; contradicts if the CI lower bound > 0.52; else
  inconclusive. p = P*(AUC > 0.52).
  Holm per segment over H-NN5, H-NN6; a 'supports' with Holm p > 0.05 becomes inconclusive.
  Interpretability (descriptive): integrated gradients of the vol CNN (HAR skip recomputed from the input, baseline =
  all-zero input) aggregated by channel and lag bucket (0-9, 10-59, 60-299 s back); channel ablation (channel set to 0,
  HAR recomputed) dQLIKE for cnn and lin.
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
from ai_trader.ml import micro as mc

TAG = "2026-11-12"
CUT = "2026-07-01"
TRAIN_STEP, TEST_STEP, ABL_STEP = 30, 10, 60
EPOCH_N, BATCH, MAX_EP, PATIENCE, LR, WD = 100_000, 1024, 20, 2, 2e-3, 1e-4
SEEDS = (0,)  # runtime: 1 seed (diary 06:5x 'Micro CNN runtime cut'); was 2 seeds, 200k windows, patience 3
N_IG = 2000
LAG_BUCKETS = ((0, 10), (10, 60), (60, 300))


def root() -> Path:
    d = load().root / "data" / "models" / "ml" / "micro"
    d.mkdir(parents=True, exist_ok=True)
    return d


def micro_root() -> Path:
    return load().root / "data" / "research" / "micro_days"


def seg_of(day: str) -> str:
    return "llm_in_sample" if day < CUT else "clean_holdout"


def freeze_days() -> list[str]:
    p = root() / "days.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))["days"]
    days = []
    for d in sorted(micro_root().glob("20*-*-*")):
        q = d / "qa.json"
        if q.exists() and json.loads(q.read_text(encoding="utf-8")).get("verified", {}).get("ok"):
            days.append(d.name)
    from ai_trader.core.time import now

    p.write_text(json.dumps({"frozen_at": now().isoformat(), "days": days, "segments": {d: seg_of(d) for d in days}}, indent=1), encoding="utf-8")
    return days


# ------------------------------------------------------------------ data


def build_store(days: list[str]) -> tuple[mc.Store, dict]:
    parts, offsets, meta, aux = [], {}, {}, {}
    pos = 0
    for day in days:
        for tp in sorted((micro_root() / day / "trades_1s").glob("*.parquet")):
            sym = tp.stem
            r = mc.load_symbol_day(micro_root(), day, sym)
            if r is None or r["traded"] < mc.MIN_TRADED_SECONDS:
                continue
            key = (day, sym)
            parts.append(r["X"].astype(np.float16))
            offsets[key] = pos
            pos += r["X"].shape[0]
            meta[key] = {"day": day, "symbol": sym, "book": bool(r["X"][:, 8].max() > 0), "traded": r["traded"]}
            aux[key] = {"logp": r["logp"].astype(np.float64), "b": r["b"].astype(np.float32)}
            del r
        print(day, sum(1 for k in meta if k[0] == day), "symbol-days", flush=True)
    X = np.concatenate(parts)
    del parts
    return mc.Store(X, offsets, meta), aux


def window_set(store: mc.Store, aux: dict, keys: list, step: int, offset: int = 0) -> dict:
    rows, H, r1, base, kid, sec = [], [], [], [], [], []
    for i, k in enumerate(keys):
        s = mc.decision_seconds(step, offset)
        tg = mc.targets(aux[k]["logp"], aux[k]["b"], s)
        ok = tg["ok"]
        s = s[ok]
        rows.append(store.windows(k, s))
        o = store.offsets[k]
        H.append(mc.har_features(store.X[o:o + mc.DAY_S], s))
        r1.append(tg["r1"][ok])
        base.append(tg["base"][ok])
        kid.append(np.full(len(s), i))
        sec.append(s)
    return {"rows": np.concatenate(rows), "H": np.concatenate(H), "r1": np.concatenate(r1), "base": np.concatenate(base),
            "kid": np.concatenate(kid), "sec": np.concatenate(sec), "keys": keys}


class Gather:
    def __init__(self, store: mc.Store):
        import torch

        self.torch = torch
        self.X = torch.from_numpy(store.X)  # float16
        self.ar = torch.arange(-mc.WIN + 1, 1)

    def __call__(self, rows: np.ndarray):
        idx = self.torch.from_numpy(rows)[:, None] + self.ar[None, :]
        return self.X[idx].float()


# ------------------------------------------------------------------ training


def _loss(task, f, r1, base):
    import torch

    if task == "vol":
        return mc.qlike_loss(f, r1 * r1, base)
    return torch.nn.functional.binary_cross_entropy_with_logits(f, (r1 > 0).float())


def train(kind: str, task: str, seed: int, G: Gather, tr: dict, va: dict, hmu, hsd) -> tuple[object, dict]:
    import copy

    import torch

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = mc.make_cnn(task=task) if kind == "cnn" else mc.make_linear()
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    sel_tr = np.arange(len(tr["r1"])) if task == "vol" else np.flatnonzero(tr["r1"] != 0)
    sel_va = np.arange(len(va["r1"])) if task == "vol" else np.flatnonzero(va["r1"] != 0)
    rng = np.random.default_rng(seed)
    sel_va = np.sort(rng.choice(sel_va, size=min(60_000, len(sel_va)), replace=False))
    Hn_tr = torch.from_numpy(((tr["H"] - hmu) / hsd).astype(np.float32))
    Hn_va = torch.from_numpy(((va["H"] - hmu) / hsd).astype(np.float32))
    r_tr, b_tr = torch.from_numpy(tr["r1"].astype(np.float32)), torch.from_numpy(tr["base"].astype(np.float32))
    r_va, b_va = torch.from_numpy(va["r1"].astype(np.float32)), torch.from_numpy(va["base"].astype(np.float32))

    def val_loss():
        m.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(sel_va), 4096):
                j = sel_va[i:i + 4096]
                f = m(G(va["rows"][j]), Hn_va[j])
                tot += float(_loss(task, f, r_va[j], b_va[j])) * len(j)
                n += len(j)
        return tot / n

    best, best_state, best_ep, bad = val_loss(), copy.deepcopy(m.state_dict()), 0, 0
    for ep in range(1, MAX_EP + 1):
        m.train()
        pick = sel_tr[torch.randperm(len(sel_tr), generator=g)[:EPOCH_N].numpy()]
        for i in range(0, len(pick), BATCH):
            j = pick[i:i + BATCH]
            opt.zero_grad()
            loss = _loss(task, m(G(tr["rows"][j]), Hn_tr[j]), r_tr[j], b_tr[j])
            loss.backward()
            opt.step()
        v = val_loss()
        if v < best - 1e-6:
            best, best_state, best_ep, bad = v, copy.deepcopy(m.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, {"best_val": best, "best_epoch": best_ep, "epochs_run": ep}


def predict(models: list, G: Gather, ws: dict, hmu, hsd, sel: np.ndarray | None = None, x_mod=None) -> np.ndarray:
    import torch

    sel = np.arange(len(ws["r1"])) if sel is None else sel
    out = np.zeros(len(sel))
    with torch.no_grad():
        for i in range(0, len(sel), 4096):
            j = sel[i:i + 4096]
            x = G(ws["rows"][j])
            if x_mod is not None:
                x = x_mod(x)
                h = har_from_x(x)
            else:
                h = torch.from_numpy(ws["H"][j])
            hn = (h - torch.from_numpy(hmu.astype(np.float32))) / torch.from_numpy(hsd.astype(np.float32))
            out[i:i + len(j)] = np.mean([mm(x, hn).numpy() for mm in models], axis=0)
    return out


def har_from_x(x):
    import torch

    return torch.cat([x[:, -w:, :].mean(1) for w in mc.HAR_W], dim=1)


def integrated_gradients(models: list, x0, hmu, hsd, steps: int = 32):
    """IG of the ensemble's log-variance output w.r.t. the (B, 300, C) input, the HAR skip recomputed from the input."""
    import torch

    mu, sd = torch.from_numpy(hmu.astype(np.float32)), torch.from_numpy(hsd.astype(np.float32))
    tot = torch.zeros_like(x0)
    for k in range(steps):
        a = (k + 0.5) / steps
        x = (a * x0).clone().requires_grad_(True)
        y = torch.stack([mm(x, (har_from_x(x) - mu) / sd) for mm in models]).mean(0).sum()
        (gk,) = torch.autograd.grad(y, x)
        tot += gk
    return (x0 * tot / steps).detach().numpy()


def run_fit() -> None:
    import torch

    torch.set_num_threads(4)
    days = freeze_days()
    t0 = time.time()
    folds = {k: [d for i, d in enumerate(days) if i % 3 == k] for k in range(3)}
    dh0 = design_hash(days)
    if all(fold_done(k, dh0) for k in folds):
        aggregate(days, folds, dh0, t0)
        return
    store, aux = build_store(days)
    G = Gather(store)
    keys_all = sorted(store.meta)
    dh = design_hash(days)
    for fk, test_days in folds.items():
        if fold_done(fk, dh):
            print(json.dumps({"fold_skip": fk, "reason": "checkpoint present"}), flush=True)
            continue
        preds, info, ig_acc, abl_rows = [], [], {}, []
        train_days = [d for d in days if d not in test_days]
        val_day = train_days[-1]
        fit_days = [d for d in train_days if d != val_day]
        tr = window_set(store, aux, [k for k in keys_all if k[0] in fit_days], TRAIN_STEP)
        va = window_set(store, aux, [k for k in keys_all if k[0] == val_day], TRAIN_STEP, offset=15)
        te = window_set(store, aux, [k for k in keys_all if k[0] in test_days], TEST_STEP)
        hmu, hsd = tr["H"].mean(0), tr["H"].std(0)
        hsd = np.where(hsd > 0, hsd, 1.0)
        fold = {"fold": fk, "test_days": test_days, "fit_days": fit_days, "val_day": val_day, "n_train": len(tr["r1"]), "n_val": len(va["r1"]), "n_test": len(te["r1"])}
        res = {}
        for task in ("vol", "dir"):
            for kind in ("cnn", "lin"):
                ms, tinfo = [], []
                for sd_ in (SEEDS if kind == "cnn" else (0,)):
                    t1 = time.time()
                    m, ti = train(kind, task, sd_, G, tr, va, hmu, hsd)
                    ti["secs"] = round(time.time() - t1, 1)
                    ms.append(m)
                    tinfo.append(ti)
                    print(json.dumps({"fold": fk, "task": task, "kind": kind, "seed": sd_, **ti}), flush=True)
                fv = predict(ms, G, va, hmu, hsd)
                c = float(np.log(np.mean(va["r1"] ** 2 / (va["base"] * np.exp(fv))))) if task == "vol" else 0.0
                res[(task, kind)] = (ms, c)
                fold[f"{task}_{kind}"] = {"train": tinfo, "bias": c}
                f = predict(ms, G, te, hmu, hsd)
                te[f"out_{task}_{kind}"] = f + c
        # predictions
        kk = te["keys"]
        preds.append(pl.DataFrame({
            "day": [kk[i][0] for i in te["kid"]], "symbol": [kk[i][1] for i in te["kid"]], "sec": te["sec"], "r1": te["r1"], "base": te["base"],
            "F_cnn": te["base"] * np.exp(te["out_vol_cnn"]), "F_lin": te["base"] * np.exp(te["out_vol_lin"]),
            "p_cnn": 1 / (1 + np.exp(-te["out_dir_cnn"])), "p_lin": 1 / (1 + np.exp(-te["out_dir_lin"])),
            "book": [store.meta[kk[i]]["book"] for i in te["kid"]]}))
        # attributions (vol cnn): integrated gradients by channel x lag on random test windows
        rng = np.random.default_rng(fk)
        pick = np.sort(rng.choice(len(te["r1"]), size=min(N_IG, len(te["r1"])), replace=False))
        ms_v, _ = res[("vol", "cnn")]
        ig = integrated_gradients(ms_v, G(te["rows"][pick]), hmu, hsd)
        for sg in {seg_of(d) for d in test_days}:
            m_ = np.array([seg_of(kk[te["kid"][j]][0]) == sg for j in pick])
            if m_.any():
                ig_acc.setdefault(f"abs_{sg}", []).append(np.abs(ig[m_]).sum(0))       # (300, C) summed |IG|
                ig_acc.setdefault(f"sgn_{sg}", []).append(ig[m_].sum(0))
                ig_acc.setdefault(f"n_{sg}", []).append(int(m_.sum()))
        # channel ablation on test windows every 60 s
        asel = np.flatnonzero(te["sec"] % ABL_STEP == (mc.TRAIL + mc.WIN) % ABL_STEP)
        base_out = {kind: te[f"out_vol_{kind}"][asel] for kind in ("cnn", "lin")}
        for ci, ch in enumerate(mc.CHANNELS):
            def zero(x, ci=ci):
                x = x.clone()
                x[:, :, ci] = 0.0
                return x
            for kind in ("cnn", "lin"):
                ms_k, c_k = res[("vol", kind)]
                o = predict(ms_k, G, te, hmu, hsd, sel=asel, x_mod=zero) + c_k
                r2 = te["r1"][asel] ** 2
                Fa, Fb = te["base"][asel] * np.exp(o), te["base"][asel] * np.exp(base_out[kind])
                d = r2 * (1 / Fa - 1 / Fb) + np.log(Fa / Fb)
                abl_rows.append(pl.DataFrame({"fold": fk, "channel": ch, "kind": kind, "day": [kk[i][0] for i in te["kid"][asel]],
                                              "symbol": [kk[i][1] for i in te["kid"][asel]], "d": d}))
        info.append(fold)
        save_fold(fk, dh, preds, abl_rows, ig_acc, fold, store)
        print(json.dumps({"fold_done": fk, "secs": round(time.time() - t0)}), flush=True)
        del tr, va, te
    aggregate(days, folds, dh, t0)


def design_hash(days: list[str]) -> str:
    """Fingerprint of everything a fold checkpoint depends on (days, knobs, code of ml.micro and this module)."""
    import hashlib

    src = Path(__file__).resolve()
    knobs = [days, TRAIN_STEP, TEST_STEP, ABL_STEP, EPOCH_N, BATCH, MAX_EP, PATIENCE, LR, WD, list(SEEDS), N_IG, mc.MIN_TRADED_SECONDS]
    h = hashlib.sha256(json.dumps(knobs).encode())
    h.update((src.parent.parent / "ml" / "micro.py").read_bytes())
    return h.hexdigest()[:16]


def fold_dir(fk: int) -> Path:
    d = root() / f"fold_{fk}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_done(fk: int, dh: str) -> bool:
    p = root() / f"fold_{fk}" / "done.json"
    return p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("design_hash") == dh


def save_fold(fk: int, dh: str, preds, abl_rows, ig_acc, fold: dict, store) -> None:
    """Per-fold checkpoint; done.json is written last, so a crash mid-save leaves the fold incomplete."""
    d = fold_dir(fk)
    pl.concat(preds).write_parquet(d / "preds.parquet")
    pl.concat(abl_rows).group_by("fold", "channel", "kind", "day", "symbol").agg(pl.col("d").mean(), pl.len().alias("n")).write_parquet(d / "ablation.parquet")
    np.savez(d / "ig.npz", **{k: np.asarray(v[0]) for k, v in ig_acc.items() if not k.startswith("n_")},
             **{k: np.asarray(v) for k, v in ig_acc.items() if k.startswith("n_")})
    meta = [{**v, "key": list(k)} for k, v in store.meta.items() if k[0] in fold["test_days"]]
    (d / "fold.json").write_text(json.dumps({"fold": fold, "meta_test": meta}, indent=1, default=str), encoding="utf-8")
    (d / "done.json").write_text(json.dumps({"design_hash": dh}), encoding="utf-8")


def aggregate(days: list[str], folds: dict, dh: str, t0: float) -> None:
    preds, abl, folds_info, meta = [], [], [], []
    acc: dict = {}
    for fk in folds:
        d = root() / f"fold_{fk}"
        preds.append(pl.read_parquet(d / "preds.parquet"))
        abl.append(pl.read_parquet(d / "ablation.parquet"))
        fj = json.loads((d / "fold.json").read_text(encoding="utf-8"))
        folds_info.append(fj["fold"])
        meta += fj["meta_test"]
        z = np.load(d / "ig.npz")
        for k in z.files:
            acc.setdefault(k, []).append(z[k])
    pl.concat(preds).write_parquet(root() / "preds.parquet")
    pl.concat(abl).write_parquet(root() / "ablation.parquet")
    igs = {}
    for sg in ("llm_in_sample", "clean_holdout"):
        if f"abs_{sg}" in acc:
            n = int(sum(int(np.sum(x)) for x in acc[f"n_{sg}"]))
            igs[sg] = {"abs_mean": (sum(acc[f"abs_{sg}"]) / n).tolist(), "sgn_mean": (sum(acc[f"sgn_{sg}"]) / n).tolist(), "n": n}
    (root() / "ig.json").write_text(json.dumps(igs), encoding="utf-8")
    (root() / "fit_info.json").write_text(json.dumps({"days": days, "design_hash": dh, "n_symbol_days": len(meta), "folds": folds_info, "meta": meta,
                                                     "secs_total": round(time.time() - t0)}, indent=1, default=str), encoding="utf-8")


# ------------------------------------------------------------------ evaluate


def _phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2)) if np.isfinite(t) else 1.0


def _binom_ge(k: int, n: int, p: float) -> float:
    from scipy.stats import binom

    return float(binom.sf(k - 1, n, p)) if n else 1.0


def symbol_day_diffs(pr: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    r2 = pr["r1"].to_numpy() ** 2
    Fa, Fb = pr[a].to_numpy(), pr[b].to_numpy()
    d = r2 * (1 / Fa - 1 / Fb) + np.log(Fa / Fb)
    return pr.select("day", "symbol", "book").with_columns(pl.Series("d", d)).group_by("day", "symbol").agg(pl.col("d").mean(), pl.col("book").first(), pl.len().alias("n"))


def nn5_seg(pr: pl.DataFrame) -> dict:
    sd = symbol_day_diffs(pr, "F_cnn", "F_lin")
    d = sd["d"].to_numpy()
    n = len(d)
    k = int((d < 0).sum())
    share = k / n if n else float("nan")
    mean, sdv = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sdv / math.sqrt(n)) if n > 1 and sdv > 0 else float("nan")
    dm = sd.group_by("day").agg(pl.col("d").mean())["d"].to_numpy()
    t_day = float(dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm)))) if len(dm) > 1 and dm.std(ddof=1) > 0 else float("nan")
    if share >= 0.70 and t < -2:
        v = "supports"
    elif mean >= 0 or share < 0.5:
        v = "contradicts"
    else:
        v = "inconclusive"
    p = max(_binom_ge(k, n, 0.5), _phi(t))
    ctx = {}
    for m in ("cnn", "lin"):
        q = pr.with_columns(pl.col("base").alias("F_base"))
        s2 = symbol_day_diffs(q, f"F_{m}", "F_base")["d"].to_numpy()
        ctx[f"{m}_vs_base"] = {"mean": float(s2.mean()), "share_lt0": float((s2 < 0).mean()), "t": float(s2.mean() / (s2.std(ddof=1) / math.sqrt(len(s2)))) if len(s2) > 1 else float("nan")}
    byb = {}
    for bk in (True, False):
        x = sd.filter(pl.col("book") == bk)["d"].to_numpy()
        if len(x) > 1:
            byb["book" if bk else "no_book"] = {"n": len(x), "mean": float(x.mean()), "share_lt0": float((x < 0).mean()), "t": float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))}
    return {"n_symbol_days": n, "n_days": int(sd["day"].n_unique()), "share_lt0": share, "mean_diff": mean, "t": t, "t_day_clustered": t_day,
            "rel_to_lin_qlike": mean / float(np.mean(_qlike(pr, "F_lin"))), "verdict": v, "p": p, "context": ctx, "by_book": byb,
            "by_day": {r[0]: {"mean": r[1], "share_lt0": r[2]} for r in sd.group_by("day").agg(pl.col("d").mean(), (pl.col("d") < 0).mean().alias("s")).sort("day").iter_rows()}}


def _qlike(pr: pl.DataFrame, col: str) -> np.ndarray:
    r2 = pr["r1"].to_numpy() ** 2
    F = pr[col].to_numpy()
    with np.errstate(all="ignore"):
        q = np.where(r2 > 0, r2 / F - np.log(r2 / F) - 1, np.nan)
    return q[np.isfinite(q)]


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata

    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binned_auc_boot(y: np.ndarray, p: np.ndarray, groups: np.ndarray, B: int = 1000, bins: int = 2000, seed: int = 0) -> np.ndarray:
    """Cluster (symbol-day) bootstrap of the AUC: per-cluster histograms of positive / negative windows over quantile bins
    of p; a draw's AUC = sum_i pos_i (neg below i + neg_i / 2) / (P N) (ties within a bin count half; bin error ~1e-4)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1]))
    b = np.searchsorted(edges, p, side="right")
    nb = len(edges) + 1
    ng = int(groups.max()) + 1
    Hp = np.zeros((ng, nb))
    Hn = np.zeros((ng, nb))
    np.add.at(Hp, (groups[y == 1], b[y == 1]), 1.0)
    np.add.at(Hn, (groups[y == 0], b[y == 0]), 1.0)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, ng, ng), minlength=ng) for _ in range(B)]).astype(float)
    P, N = W @ Hp, W @ Hn
    below = np.cumsum(N, axis=1) - N
    return (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))


def nn6_seg(pr: pl.DataFrame, B: int = 1000) -> dict:
    q = pr.filter(pl.col("r1") != 0)
    y = (q["r1"].to_numpy() > 0).astype(int)
    keys = (q["day"] + "|" + q["symbol"]).to_numpy()
    uk, inv = np.unique(keys, return_inverse=True)
    out = {"n_windows": int(len(y)), "n_symbol_days": int(len(uk)), "base_rate": float(y.mean()), "bootstrap": f"cluster (symbol-day), B={B}, binned 2000"}
    for m in ("cnn", "lin"):
        p = q[f"p_{m}"].to_numpy()
        auc = _auc(y, p)
        bs = binned_auc_boot(y, p, inv, B)
        out[m] = {"auc": auc, "ci95": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))], "p_gt_052": float((1 + np.sum(bs > 0.52)) / (B + 1)),
                  "brier_skill": float(1 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2))}
    r = out["cnn"]
    out["verdict"] = "supports" if r["ci95"][1] <= 0.52 else ("contradicts" if r["ci95"][0] > 0.52 else "inconclusive")
    out["p"] = r["p_gt_052"]
    return out


def interpret_summary() -> dict:
    ig = json.loads((root() / "ig.json").read_text(encoding="utf-8"))
    out = {}
    for sg, v in ig.items():
        A = np.asarray(v["abs_mean"])  # (300, C), index 299 = the most recent second
        S = np.asarray(v["sgn_mean"])
        tot = A.sum()
        lag = np.arange(mc.WIN)[::-1]  # seconds back
        ch = {c: float(A[:, i].sum() / tot) for i, c in enumerate(mc.CHANNELS)}
        buckets = {f"{a}-{b - 1}s": float(A[(lag >= a) & (lag < b)].sum() / tot) for a, b in LAG_BUCKETS}
        cell = {c: {f"{a}-{b - 1}s": float(A[(lag >= a) & (lag < b), i].sum() / tot) for a, b in LAG_BUCKETS} for i, c in enumerate(mc.CHANNELS)}
        sign = {c: float(S[:, i].sum()) for i, c in enumerate(mc.CHANNELS)}
        out[sg] = {"n": v["n"], "channel_share": ch, "lag_share": buckets, "channel_lag_share": cell, "signed_sum": sign}
    return out


def ablation_summary() -> dict:
    a = pl.read_parquet(root() / "ablation.parquet")
    out = {}
    for sg in ("llm_in_sample", "clean_holdout", "pooled"):
        q = a if sg == "pooled" else a.filter(pl.col("day").map_elements(seg_of, return_dtype=pl.Utf8) == sg)
        r = {}
        for (ch, kind), g in q.group_by("channel", "kind"):
            d = g["d"].to_numpy()
            r[f"{kind}:{ch}"] = {"mean": float(d.mean()), "t": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if len(d) > 1 and d.std() > 0 else float("nan"),
                                 "share_gt0": float((d > 0).mean())}
        out[sg] = r
    return out


def run_evaluate(record_rows: bool = False) -> dict:
    from ai_trader.research.decision_budget_study import holm

    pr = pl.read_parquet(root() / "preds.parquet")
    info = json.loads((root() / "fit_info.json").read_text(encoding="utf-8"))
    pr = pr.with_columns(pl.col("day").map_elements(seg_of, return_dtype=pl.Utf8).alias("segment"))
    out = {"days": info["days"], "n_symbol_days": info["n_symbol_days"], "folds": info["folds"], "secs_total": info["secs_total"],
           "rows": pr.height, "rows_by_segment": dict(pr.group_by("segment").len().iter_rows())}
    for h, fn in (("H-NN5", nn5_seg), ("H-NN6", nn6_seg)):
        out[h] = {sg: fn(pr if sg == "pooled" else pr.filter(pl.col("segment") == sg)) for sg in ("llm_in_sample", "clean_holdout", "pooled")}
    out["holm"] = {}
    for sg in ("llm_in_sample", "clean_holdout", "pooled"):
        ps = {h: out[h][sg]["p"] for h in ("H-NN5", "H-NN6")}
        out["holm"][sg] = {"p": ps, "p_holm": holm(ps)}
    out["verdicts"] = {h: {sg: ("inconclusive" if out[h][sg]["verdict"] == "supports" and out["holm"][sg]["p_holm"][h] > 0.05 else out[h][sg]["verdict"])
                           for sg in ("llm_in_sample", "clean_holdout", "pooled")} for h in ("H-NN5", "H-NN6")}
    out["ig"] = interpret_summary()
    out["ablation"] = ablation_summary()
    rep = load().root / "research" / "reports"
    (rep / f"ml_micro_{TAG}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    from ai_trader.research.ml_micro_report import build

    (rep / f"ml_micro_{TAG}.md").write_text(build(out), encoding="utf-8")
    print(json.dumps(out["verdicts"], indent=1))
    if record_rows:
        record_all(out)
    return out


DATA = ("data/research/micro_days/<day>/{trades_1s,book_1s} (agent A aggregates, read only), days frozen in data/models/ml/micro/days.json; "
        "ai_trader.ml.micro; src/ai_trader/research/ml_micro_study.py")


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record, results

    for h in ("H-NN5", "H-NN6"):
        done = {r["segment"] for r in results(h)}
        for sg in ("llm_in_sample", "clean_holdout", "pooled"):
            if sg in done:
                print(f"skip {h} {sg}")
                continue
            r = out[h][sg]
            if h == "H-NN5":
                m = {k: r[k] for k in ("n_symbol_days", "n_days", "share_lt0", "mean_diff", "t", "t_day_clustered", "rel_to_lin_qlike", "context", "by_book")}
                note = ("leave-days-out (3 rotating folds), CNN vs linear HAR-style model on the same 1 s inputs, next-1m r^2 QLIKE, per symbol-day "
                        f"mean paired diff; sign share and t; training windows every 30 s; raw {r['verdict']}")
            else:
                m = {k: r[k] for k in ("n_windows", "n_symbol_days", "base_rate", "cnn", "lin")}
                note = f"leave-days-out AUC of the CNN for sign(next-1m return), bootstrap over symbol-days; raw {r['verdict']}"
            m["p_holm"] = out["holm"][sg]["p_holm"][h]
            record(h, sg, out["verdicts"][h][sg], m, note=note, data=DATA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("freeze-days", "fit", "evaluate"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.cmd == "freeze-days":
        print(freeze_days())
    elif a.cmd == "fit":
        run_fit()
    else:
        run_evaluate(a.record)


if __name__ == "__main__":
    main()
