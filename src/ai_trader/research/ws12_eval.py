"""WS12 evaluation: proper scores, DM (NW + HLN), MCS, seed spread, calibration and the H-WS12-1..5 verdicts.

Design in ai_trader.research.ws12_deep (docstring) and research/hypotheses_pending/WS12.yaml. Research only.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ai_trader.research import ws12_deep as wd

ARCHS = ("gru", "tcn", "mlp")
CLASSICAL = ("har", "base", "emp")
METRICS = ("pinball", "crps", "logscore", "qlike")
IDX7 = [round(t * 100) - 1 for t in wd.TAUS]  # positions of TAUS inside TAU99


def _cut() -> np.datetime64:
    from ai_trader.research.evaluate import knowledge_cutoff

    return np.datetime64(knowledge_cutoff().replace(tzinfo=None), "us")


def dm(d: np.ndarray) -> dict:
    """Per-date loss differences: mean, NW t with the HLN small-sample factor (h = 1 on per-date means), one-sided p."""
    from scipy import stats

    from ai_trader.research.evaluate import newey_west_t

    d = d[np.isfinite(d)]
    m, t, n = newey_west_t(d)
    if not np.isfinite(t):
        return {"mean": float(m), "t": float("nan"), "p": float("nan"), "n": int(n), "mde": float("nan")}
    t = t * math.sqrt((n - 1) / n)
    se = abs(m / t) if t != 0 else float("nan")
    return {"mean": float(m), "t": float(t), "p": float(stats.t.cdf(t, n - 1)), "n": int(n), "mde": float(2.8 * se)}


def _scores_t(par: np.ndarray, y: np.ndarray, v: np.ndarray) -> dict:
    """par (n, 4) mu, scale, df, logF of ONE model at one horizon -> per-row scores and quantiles (99)."""
    from scipy import stats

    mu, sc, df, lF = (par[:, k].astype(float) for k in range(4))
    q = wd.t_quantiles(mu, sc, df, wd.TAU99)
    return {"q": q, "lpdf": stats.t.logpdf(y, df, loc=mu, scale=sc), "cdf": stats.t.cdf(y, df, loc=mu, scale=sc), "F": np.exp(lF)}


def _from_q(q: np.ndarray, y: np.ndarray) -> dict:
    return {"pinball": wd.pinball(q[:, IDX7], y, wd.TAUS), "crps": 2 * wd.pinball(q, y, wd.TAU99)}


def evaluate(record: bool = False) -> dict:
    pn = wd.panel()
    cut = _cut()
    res: dict = {"horizons": {}, "seeds": {}, "calibration": {}, "dstar": None, "train": {}}
    # D*: gru vs tcn on the R1 inner-validation loss of the chosen config (fixed before scoring)
    logs = {}
    for a in ARCHS:
        for r in wd.REFITS:
            p = wd.root() / "logs" / f"train_{a}_{r}.json"
            if p.is_file():
                logs[(a, r)] = json.loads(p.read_text(encoding="utf-8"))
    vl = {a: min(x["val_loss"] for x in logs[(a, "R1")]["runs"] if x["seed"] == 0) for a in ("gru", "tcn") if (a, "R1") in logs}
    dstar = min(vl, key=vl.get)
    res["dstar"] = {"arch": dstar, "r1_val_loss": vl}
    res["train"] = {f"{a}_{r}": v for (a, r), v in logs.items()}
    rows_all, per = [], {}
    for r in wd.REFITS:
        te = np.load(wd.root() / "preds" / f"rows_{r}.npy")
        rows_all.append(te)
        per[r] = {"rows": te}
        for a in ARCHS:
            lg = logs[(a, r)]
            lr = lg["chosen_lr"]
            per[r][a] = [np.load(wd.root() / "preds" / f"{a}_{r}_lr{lr:g}_s{s}.npy") for s in wd.SEEDS]
        for c in ("har", "base"):
            per[r][c] = np.load(wd.root() / "preds" / f"{c}_{r}.npy")
        per[r]["emp_q"] = np.load(wd.root() / "preds" / f"emp_q_{r}.npy")
        per[r]["emp_F"] = np.load(wd.root() / "preds" / f"emp_F_{r}.npy")
    models = (*ARCHS, *CLASSICAL)
    for j, h in enumerate(wd.HS):
        loss = {m: {k: [] for k in METRICS} for m in models}
        seedloss = {a: {s: {"pinball": [], "qlike": []} for s in wd.SEEDS} for a in ARCHS}
        pit = {m: [] for m in (*ARCHS, "har", "base")}
        cov = {m: {"80": [], "90": []} for m in models}
        T_all, A_all = [], []
        for r in wd.REFITS:
            te = per[r]["rows"]
            hr = pn["hour"][te]
            sel = np.ones(len(te), bool) if h == 1 else ((hr % 4 == 0) if h == 4 else (hr == 0))
            sel &= np.isfinite(per[r]["emp_q"][:, j, 0]) & np.isfinite(per[r]["emp_F"][:, j])
            ii = np.flatnonzero(sel)
            rows = te[ii]
            y = pn["Y"][rows, j].astype(float)
            v = pn["V"][rows, j].astype(float)
            T_all.append(pn["T"][rows])
            A_all.append(pn["A"][rows])
            for a in ARCHS:
                qs, lp, cd, F = 0.0, [], 0.0, 0.0
                for s, P in zip(wd.SEEDS, per[r][a]):
                    o = _scores_t(P[ii, j], y, v)
                    sc = _from_q(o["q"], y)
                    seedloss[a][s]["pinball"].append(sc["pinball"])
                    seedloss[a][s]["qlike"].append(v / o["F"] + np.log(o["F"]))
                    qs = qs + o["q"] / len(wd.SEEDS)
                    lp.append(o["lpdf"])
                    cd = cd + o["cdf"] / len(wd.SEEDS)
                    F = F + o["F"] / len(wd.SEEDS)
                sc = _from_q(qs, y)
                loss[a]["pinball"].append(sc["pinball"])
                loss[a]["crps"].append(sc["crps"])
                lpm = np.logaddexp.reduce(np.vstack(lp), axis=0) - math.log(len(lp))
                loss[a]["logscore"].append(-lpm)
                loss[a]["qlike"].append(v / F + np.log(F))
                pit[a].append(cd)
                cov[a]["80"].append((y >= qs[:, 9]) & (y <= qs[:, 89]))
                cov[a]["90"].append((y >= qs[:, 4]) & (y <= qs[:, 94]))
            for c in ("har", "base"):
                o = _scores_t(per[r][c][ii, j], y, v)
                sc = _from_q(o["q"], y)
                loss[c]["pinball"].append(sc["pinball"])
                loss[c]["crps"].append(sc["crps"])
                loss[c]["logscore"].append(-o["lpdf"])
                loss[c]["qlike"].append(v / o["F"] + np.log(o["F"]))
                pit[c].append(o["cdf"])
                cov[c]["80"].append((y >= o["q"][:, 9]) & (y <= o["q"][:, 89]))
                cov[c]["90"].append((y >= o["q"][:, 4]) & (y <= o["q"][:, 94]))
            q = per[r]["emp_q"][ii, j].astype(float)
            sc = _from_q(q, y)
            Fe = per[r]["emp_F"][ii, j].astype(float)
            loss["emp"]["pinball"].append(sc["pinball"])
            loss["emp"]["crps"].append(sc["crps"])
            loss["emp"]["logscore"].append(np.full(len(y), np.nan))
            loss["emp"]["qlike"].append(v / Fe + np.log(Fe))
            cov["emp"]["80"].append((y >= q[:, 9]) & (y <= q[:, 89]))
            cov["emp"]["90"].append((y >= q[:, 4]) & (y <= q[:, 94]))
        T = np.concatenate(T_all)
        A = np.concatenate(A_all)
        seg = np.where(T < cut, "llm_in_sample", "clean_holdout")
        day = T.astype("datetime64[D]")
        cat = lambda lst: np.concatenate(lst)  # noqa: E731
        L = {m: {k: cat(loss[m][k]) for k in METRICS} for m in models}
        out_h: dict = {}
        for sg in ("llm_in_sample", "clean_holdout"):
            msk = seg == sg
            ud, inv = np.unique(day[msk], return_inverse=True)

            def per_date(x: np.ndarray) -> np.ndarray:
                xx = x[msk]
                ok = np.isfinite(xx)
                s_ = np.bincount(inv[ok], weights=xx[ok], minlength=len(ud))
                n_ = np.bincount(inv[ok], minlength=len(ud))
                return np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)

            o_s: dict = {"rows": int(msk.sum()), "dates": int(len(ud)), "mean": {}, "dm_vs_har": {}, "dm_dstar": {}, "mcs": {}}
            pdm = {m: {k: per_date(L[m][k]) for k in METRICS} for m in models}
            for m in models:
                o_s["mean"][m] = {k: float(np.nanmean(L[m][k][msk])) if np.isfinite(L[m][k][msk]).any() else None for k in METRICS}
            for m in models:
                if m == "har":
                    continue
                o_s["dm_vs_har"][m] = {k: dm(pdm[m][k] - pdm["har"][k]) for k in METRICS if np.isfinite(pdm[m][k]).any()}
            for m in ("mlp", "base", "emp", "tcn" if dstar == "gru" else "gru"):
                o_s["dm_dstar"][m] = {k: dm(pdm[dstar][k] - pdm[m][k]) for k in METRICS if np.isfinite(pdm[m][k]).any()}
            from ai_trader.features.vol_panel import mcs

            for k in METRICS:
                ms = [m for m in models if np.isfinite(pdm[m][k]).all()]
                Lm = np.column_stack([pdm[m][k] for m in ms])
                okr = np.isfinite(Lm).all(axis=1)
                mm = mcs(Lm[okr], alpha=0.10, block=10, B=1000, seed=0)
                o_s["mcs"][k] = {m: {"in": bool(mm["included"][i]), "p": float(mm["pvalues"][i])} for i, m in enumerate(ms)}
            # per asset (BTC, ETH) vs har, pinball and qlike
            o_s["per_asset"] = {}
            for ai, sym in enumerate(wd.SYMBOLS):
                ma = msk & (A == ai)
                udA, invA = np.unique(day[ma], return_inverse=True)

                def pdA(x: np.ndarray) -> np.ndarray:
                    xx = x[ma]
                    return np.bincount(invA, weights=xx, minlength=len(udA)) / np.bincount(invA, minlength=len(udA))

                o_s["per_asset"][sym] = {k: dm(pdA(L[dstar][k]) - pdA(L["har"][k])) for k in ("pinball", "qlike")}
            # seeds
            sd: dict = {}
            for a in ARCHS:
                sd[a] = {}
                for k in ("pinball", "qlike"):
                    vals = [float(np.nanmean(cat(seedloss[a][s][k])[msk])) for s in wd.SEEDS]
                    diffs = [dm(per_date(cat(seedloss[a][s][k])) - pdm["har"][k]) for s in wd.SEEDS]
                    sd[a][k] = {"per_seed": vals, "sd": float(np.std(vals, ddof=1)), "ens": o_s["mean"][a][k],
                                "seed_vs_har": [{"mean": x["mean"], "t": x["t"]} for x in diffs]}
            o_s["seeds"] = sd
            from scipy import stats

            o_s["calibration"] = {m: {"cov80": float(np.mean(cat(cov[m]["80"])[msk])), "cov90": float(np.mean(cat(cov[m]["90"])[msk]))} for m in models}
            for m in pit:
                pv = cat(pit[m])[msk]
                o_s["calibration"][m]["pit_ks"] = float(stats.kstest(pv, "uniform").statistic)
            out_h[sg] = o_s
        res["horizons"][f"{h}h"] = out_h
    res["verdicts"] = verdicts(res)
    try:
        res["pbo"] = pbo_configs()
    except Exception as e:  # noqa: BLE001
        res["pbo"] = {"error": repr(e)}
    res["learning"] = learning()
    p = wd.load().root / "research" / "reports" / f"ws12_{wd.TAG}.json"
    p.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["verdicts"], indent=1, default=float))
    if record:
        do_record(res)
    return res


def verdicts(res: dict) -> dict:
    ds = res["dstar"]["arch"]
    H = res["horizons"]
    out: dict = {}
    spec = {"H-WS12-1": ("1h", "pinball"), "H-WS12-2": ("24h", "pinball"), "H-WS12-3": ("1h", "qlike")}
    for sg in ("llm_in_sample", "clean_holdout"):
        ps = {hid: H[h][sg]["dm_vs_har"][ds][k]["p"] for hid, (h, k) in spec.items()}
        from ai_trader.research.decision_budget_study import holm

        hp = holm({k: (v if np.isfinite(v) else 1.0) for k, v in ps.items()})
        for hid, (h, k) in spec.items():
            d = H[h][sg]["dm_vs_har"][ds][k]
            v = "supports" if (d["mean"] < 0 and d["t"] < -2 and hp[hid] < 0.05) else ("contradicts" if d["mean"] >= 0 else "inconclusive")
            out.setdefault(hid, {})[sg] = {"verdict": v, "diff": d["mean"], "t": d["t"], "holm_p": hp[hid], "n_dates": d["n"], "mde": d["mde"]}
        # H-WS12-4 (null: sequence adds nothing over mlp)
        beat = []
        for k in ("pinball", "qlike"):
            d = H["1h"][sg]["dm_dstar"]["mlp"][k]
            beat.append(d["mean"] < 0 and d["t"] < -2)
        out.setdefault("H-WS12-4", {})[sg] = {"beats": beat, "pinball": H["1h"][sg]["dm_dstar"]["mlp"]["pinball"], "qlike": H["1h"][sg]["dm_dstar"]["mlp"]["qlike"]}
        # H-WS12-5 seeds
        s = H["1h"][sg]["seeds"][ds]["pinball"]
        ens = H["1h"][sg]["dm_vs_har"][ds]["pinball"]["mean"]
        signs = [np.sign(x["mean"]) == np.sign(ens) for x in s["seed_vs_har"]]
        out.setdefault("H-WS12-5", {})[sg] = {"same_sign": bool(all(signs)), "sd": s["sd"], "half_gap": abs(ens) / 2, "ok": bool(all(signs) and s["sd"] < abs(ens) / 2),
                                             "flip": bool(not all(signs))}
    for hid in spec:
        a, b = out[hid]["llm_in_sample"]["verdict"], out[hid]["clean_holdout"]["verdict"]
        out[hid]["pooled"] = "supports" if a == b == "supports" else ("contradicts" if a == "contradicts" or (a == b == "contradicts") else "inconclusive")
    b4 = [all(out["H-WS12-4"][sg]["beats"]) for sg in ("llm_in_sample", "clean_holdout")]
    any4 = [[out["H-WS12-4"][sg]["beats"][i] for sg in ("llm_in_sample", "clean_holdout")] for i in range(2)]
    both_seg = [all(x) for x in any4]  # per metric: beats in both segments
    out["H-WS12-4"]["pooled"] = "contradicts" if all(both_seg) else ("supports" if not any(both_seg) else "inconclusive")
    out["H-WS12-4"]["both_segments_all"] = b4
    for sg in ("llm_in_sample", "clean_holdout"):
        x = out["H-WS12-5"][sg]
        x["verdict"] = "supports" if x["ok"] else "inconclusive"
    flips = all(out["H-WS12-5"][sg]["flip"] for sg in ("llm_in_sample", "clean_holdout"))
    oks = all(out["H-WS12-5"][sg]["ok"] for sg in ("llm_in_sample", "clean_holdout"))
    out["H-WS12-5"]["pooled"] = "supports" if oks else ("contradicts" if flips else "inconclusive")
    return out


def do_record(res: dict) -> None:
    from ai_trader.research.hypotheses import record

    v = res["verdicts"]
    data = f"Bybit 1h {', '.join(wd.SYMBOLS)} as of {wd.AS_OF.date()}; refits R1 2025-07-01, R2 2026-07-01; D* = {res['dstar']['arch']}"
    for hid in ("H-WS12-1", "H-WS12-2", "H-WS12-3"):
        for sg in ("llm_in_sample", "clean_holdout"):
            record(hid, sg, v[hid][sg]["verdict"], v[hid][sg], data=data, note="D* 3-seed ensemble vs per-asset log-HAR-t; per-date NW t with HLN; Holm over 1-3")
        record(hid, "pooled", v[hid]["pooled"], {"segments": {sg: v[hid][sg]["verdict"] for sg in ("llm_in_sample", "clean_holdout")}}, data=data)
    for sg in ("llm_in_sample", "clean_holdout"):
        x = v["H-WS12-4"][sg]
        record("H-WS12-4", sg, "supports" if not any(x["beats"]) else ("contradicts" if all(x["beats"]) else "inconclusive"), x, data=data,
               note="null: D* does not beat the feature mlp on 1h pinball and QLIKE")
    record("H-WS12-4", "pooled", v["H-WS12-4"]["pooled"], {"both_segments_per_metric": v["H-WS12-4"]["both_segments_all"]}, data=data)
    for sg in ("llm_in_sample", "clean_holdout"):
        record("H-WS12-5", sg, v["H-WS12-5"][sg]["verdict"], v["H-WS12-5"][sg], data=data)
    record("H-WS12-5", "pooled", v["H-WS12-5"]["pooled"], {}, data=data)


__all__ = ["dm", "evaluate", "verdicts"]


def properties() -> dict:
    """Dossier section 2: empirical properties of the targets per asset (fit data = rows before the R1 refit)."""
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss

    pn = wd.panel()
    cut = wd._us(wd.REFITS["R1"][0])
    out = {}
    for ai, sym in enumerate(wd.SYMBOLS):
        o = {}
        for j, h in enumerate(wd.HS):
            m = (pn["A"] == ai) & np.isfinite(pn["Y"][:, j]) & (pn["T"] < cut)
            m &= np.ones_like(m) if h == 1 else ((pn["hour"] % 4 == 0) if h == 4 else (pn["hour"] == 0))
            y = pn["Y"][m, j].astype(float)
            lv = np.log(np.maximum(pn["V"][m, j].astype(float), 1e-6))
            ys = y[-6000:]
            lvs = lv[-6000:]
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_lv, kpss_lv = adfuller(lvs, autolag="AIC")[1], kpss(lvs, nlags="auto")[1]
            lb = acorr_ljungbox(y, lags=[10], return_df=False) if False else acorr_ljungbox(y, lags=[10])
            lb2 = acorr_ljungbox(y * y, lags=[10])
            df_, loc, sc = stats.t.fit(ys)
            o[f"{h}h"] = {"n": int(len(y)), "mean": float(y.mean()), "sd": float(y.std()), "skew": float(stats.skew(y)),
                          "exkurt": float(stats.kurtosis(y)), "t_df": float(df_), "lb10_p": float(lb["lb_pvalue"].iloc[0]),
                          "lb10_sq_p": float(lb2["lb_pvalue"].iloc[0]), "adf_p_logv": float(adf_lv), "kpss_p_logv": float(kpss_lv),
                          "acf1_logv": float(np.corrcoef(lv[1:], lv[:-1])[0, 1])}
        out[sym] = o
    p = wd.load().root / "research" / "reports" / f"ws12_properties_{wd.TAG}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def pbo_configs() -> dict:
    """CSCV PBO over the 6 seed-0 candidates (3 archs x 2 lr) on per-date 1h losses (test period, both refits)."""
    pn = wd.panel()
    out = {}
    names = [f"{a}_lr{lr:g}" for a in ARCHS for lr in wd.LRS]
    for k in ("pinball", "qlike"):
        cols, days = [], None
        for nm in names:
            a, lr = nm.split("_lr")
            parts_d, parts_l = [], []
            for r in wd.REFITS:
                te = np.load(wd.root() / "preds" / f"rows_{r}.npy")
                P = np.load(wd.root() / "preds" / f"{a}_{r}_lr{lr}_s0.npy")[:, 0]
                y, v = pn["Y"][te, 0].astype(float), pn["V"][te, 0].astype(float)
                o = _scores_t(P, y, v)
                l_ = _from_q(o["q"], y)["pinball"] if k == "pinball" else v / o["F"] + np.log(o["F"])
                parts_d.append(pn["T"][te].astype("datetime64[D]"))
                parts_l.append(l_)
            dd, ll = np.concatenate(parts_d), np.concatenate(parts_l)
            ud, inv = np.unique(dd, return_inverse=True)
            cols.append(np.bincount(inv, weights=ll) / np.bincount(inv))
            days = ud
        perf = -np.column_stack(cols)
        out[k] = {"configs": names, "n_dates": int(len(days)), **pbo_mean(perf, 16)}
    return out


def pbo_mean(perf: np.ndarray, n_blocks: int = 16) -> dict:
    """CSCV PBO (Bailey et al. 2017) with the MEAN per-period performance as the selection criterion (losses, not P&L)."""
    from itertools import combinations

    t, n = perf.shape
    Lb = t // n_blocks
    xb = perf[: Lb * n_blocks].reshape(n_blocks, Lb, n).mean(axis=1)  # (S, N) block means
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    lam, wins = [], np.zeros(n, int)
    for c in combos:
        m = np.zeros(n_blocks, bool)
        m[list(c)] = True
        is_, oos = xb[m].mean(axis=0), xb[~m].mean(axis=0)
        b = int(np.argmax(is_))
        wins[b] += 1
        rank = (oos < oos[b]).sum() + 0.5 * ((oos == oos[b]).sum() - 1) + 1
        w = rank / (n + 1)
        lam.append(math.log(w / (1 - w)))
    lam = np.array(lam)
    return {"pbo": float(np.mean(lam <= 0)), "logit_median": float(np.median(lam)), "n_splits": len(combos), "is_winner_counts": wins.tolist()}


def learning() -> dict:
    """Learning-curve summary per run: epochs to the best validation loss, best val, train loss at that epoch."""
    out = {}
    for f in sorted((wd.root() / "models").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        c = np.array(m["curve"])
        b = int(np.argmin(c[:, 1]))
        out[f.stem] = {"epochs_run": int(len(c)), "best_epoch": b + 1, "val_best": float(c[b, 1]), "train_at_best": float(c[b, 0]),
                       "val_first": float(c[0, 1]), "secs": float(m["secs"]), "params": int(m["params"])}
    return out


def evaluate_extra() -> dict:
    """Extra trials H-WS12-8 / H-WS12-9: D* vs harx and mlpx (and harx vs har, mlpx vs mlp), all horizons and metrics."""
    from ai_trader.features.vol_panel import mcs

    pn = wd.panel()
    cut = _cut()
    nets = ("gru", "tcn", "mlp", "mlpx")
    logs = {(a, r): json.loads((wd.root() / "logs" / f"train_{a}_{r}.json").read_text(encoding="utf-8")) for a in nets for r in wd.REFITS}
    out: dict = {"horizons": {}, "chosen_lr": {f"{a}_{r}": v["chosen_lr"] for (a, r), v in logs.items()}}
    for j, h in enumerate(wd.HS):
        L = {m: {k: [] for k in METRICS} for m in (*nets, "har", "harx")}
        T_all = []
        for r in wd.REFITS:
            te = np.load(wd.root() / "preds" / f"rows_{r}.npy")
            emp_q = np.load(wd.root() / "preds" / f"emp_q_{r}.npy")
            emp_F = np.load(wd.root() / "preds" / f"emp_F_{r}.npy")
            hr = pn["hour"][te]
            sel = np.ones(len(te), bool) if h == 1 else ((hr % 4 == 0) if h == 4 else (hr == 0))
            sel &= np.isfinite(emp_q[:, j, 0]) & np.isfinite(emp_F[:, j])  # the same rows as evaluate()
            ii = np.flatnonzero(sel)
            rows = te[ii]
            y, v = pn["Y"][rows, j].astype(float), pn["V"][rows, j].astype(float)
            T_all.append(pn["T"][rows])
            for a in nets:
                lr = logs[(a, r)]["chosen_lr"]
                qs, lp, F = 0.0, [], 0.0
                for s in wd.SEEDS:
                    o = _scores_t(np.load(wd.root() / "preds" / f"{a}_{r}_lr{lr:g}_s{s}.npy")[ii, j], y, v)
                    qs, F = qs + o["q"] / len(wd.SEEDS), F + o["F"] / len(wd.SEEDS)
                    lp.append(o["lpdf"])
                sc = _from_q(qs, y)
                L[a]["pinball"].append(sc["pinball"])
                L[a]["crps"].append(sc["crps"])
                L[a]["logscore"].append(-(np.logaddexp.reduce(np.vstack(lp), axis=0) - math.log(len(lp))))
                L[a]["qlike"].append(v / F + np.log(F))
            for c in ("har", "harx"):
                o = _scores_t(np.load(wd.root() / "preds" / f"{c}_{r}.npy")[ii, j], y, v)
                sc = _from_q(o["q"], y)
                L[c]["pinball"].append(sc["pinball"])
                L[c]["crps"].append(sc["crps"])
                L[c]["logscore"].append(-o["lpdf"])
                L[c]["qlike"].append(v / o["F"] + np.log(o["F"]))
        T = np.concatenate(T_all)
        LL = {m: {k: np.concatenate(x) for k, x in d.items()} for m, d in L.items()}
        seg = np.where(T < cut, "llm_in_sample", "clean_holdout")
        day = T.astype("datetime64[D]")
        oh = {}
        for sg in ("llm_in_sample", "clean_holdout"):
            msk = seg == sg
            ud, inv = np.unique(day[msk], return_inverse=True)
            cnt = np.bincount(inv, minlength=len(ud))
            pdm = {m: {k: np.bincount(inv, weights=LL[m][k][msk], minlength=len(ud)) / cnt for k in METRICS} for m in LL}
            pairs = (("gru", "harx"), ("gru", "mlpx"), ("harx", "har"), ("mlpx", "mlp"), ("mlpx", "harx"), ("tcn", "harx"))
            o = {"mean": {m: {k: float(LL[m][k][msk].mean()) for k in METRICS} for m in LL},
                 "dm": {f"{a}-{b}": {k: dm(pdm[a][k] - pdm[b][k]) for k in METRICS} for a, b in pairs}, "mcs": {}}
            for k in METRICS:
                ms = list(LL)
                mm = mcs(np.column_stack([pdm[m][k] for m in ms]), alpha=0.10, block=10, B=1000, seed=0)
                o["mcs"][k] = {m: {"in": bool(mm["included"][i]), "p": float(mm["pvalues"][i])} for i, m in enumerate(ms)}
            oh[sg] = o
        out["horizons"][f"{h}h"] = oh
    v: dict = {"H-WS12-8": {}, "H-WS12-9": {}}
    for sg in ("llm_in_sample", "clean_holdout"):
        d8 = [out["horizons"]["1h"][sg]["dm"]["gru-harx"][k] for k in ("pinball", "qlike")]
        d9 = [out["horizons"]["1h"][sg]["dm"]["gru-mlpx"][k] for k in ("pinball", "qlike")]
        v["H-WS12-8"][sg] = {"pinball": d8[0], "qlike": d8[1], "beats": [x["mean"] < 0 and x["t"] < -2 for x in d8],
                             "nonneg": [x["mean"] >= 0 for x in d8]}
        v["H-WS12-9"][sg] = {"pinball": d9[0], "qlike": d9[1], "beats": [x["mean"] < 0 and x["t"] < -2 for x in d9]}
        b8, b9 = v["H-WS12-8"][sg]["beats"], v["H-WS12-9"][sg]["beats"]
        v["H-WS12-8"][sg]["verdict"] = "supports" if all(b8) else ("contradicts" if any(v["H-WS12-8"][sg]["nonneg"]) else "inconclusive")
        v["H-WS12-9"][sg]["verdict"] = "supports" if not any(b9) else ("contradicts" if all(b9) else "inconclusive")
    segs = ("llm_in_sample", "clean_holdout")
    if all(all(v["H-WS12-8"][sg]["beats"]) for sg in segs):
        v["H-WS12-8"]["pooled"] = "supports"
    elif any(all(v["H-WS12-8"][sg]["nonneg"][i] for sg in segs) for i in range(2)):
        v["H-WS12-8"]["pooled"] = "contradicts"
    else:
        v["H-WS12-8"]["pooled"] = "inconclusive"
    per_metric = [all(v["H-WS12-9"][sg]["beats"][i] for sg in segs) for i in range(2)]
    v["H-WS12-9"]["pooled"] = "contradicts" if all(per_metric) else ("supports" if not any(per_metric) else "inconclusive")
    out["verdicts"] = v
    p = wd.load().root / "research" / "reports" / f"ws12_extra_{wd.TAG}.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(v, indent=1, default=float))
    return out
