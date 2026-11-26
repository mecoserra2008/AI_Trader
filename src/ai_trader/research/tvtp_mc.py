"""TVTP block, synthetic validation (H-TVTP0): Monte Carlo of the TVTP MS-VAR (src/ai_trader/research/tvtp_msvar.py).

    OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.tvtp_mc mc --reps 50
    ... cusum --reps 30

DGPs. K=2: the user's own synthetic check (gold leads silver in regime 0, silver leads in regime 1, P01 = logistic(-4 + 3z),
P10 = logistic(-4 - 3z), z AR(1) 0.98). K=3: adds an own-momentum, uncorrelated, higher-vol regime and z effects of both
signs on every off-diagonal transition. Every replication fits the true K with p = 1, l2 = 0.1, 3 random starts, aligns the
estimate to the truth (Hungarian on regime parameters), and filters 1000 fresh rows with the frozen parameters.
Outputs: data/research/tvtp/mc_*.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.core.config import load
from ai_trader.research.tvtp_msvar import TVTPMSVAR, companion_ccf, cusum, simulate_msvar

OUT = "data/research/tvtp"


def out_dir() -> Path:
    p = load().root / OUT
    p.mkdir(parents=True, exist_ok=True)
    return p


def dgp(K: int) -> dict[str, Any]:
    if K == 2:
        A = [np.array([[0.0, 0.0], [0.5, 0.0]]), np.array([[0.0, 0.4], [0.0, 0.0]])]
        S = [np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([[1.0, 0.7], [0.7, 1.0]])]
        W = np.array([[[4.0, -3.0]], [[-4.0, -3.0]]])
    elif K == 3:
        A = [np.array([[0.0, 0.0], [0.5, 0.0]]), np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [0.0, 0.2]])]
        S = [np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([[1.0, 0.7], [0.7, 1.0]]), np.array([[1.5, 0.0], [0.0, 1.5]])]
        # full logits L[i, j] = a + b z (stay 0, leave -4), b: 0->1 +2, 0->2 -1.5, 1->0 -2, 1->2 +1.5, 2->0 +1.5, 2->1 -1.5
        W = np.array([[[4.0, 1.5], [0.0, 3.5]], [[0.0, -3.5], [4.0, -1.5]], [[-4.0, 1.5], [-4.0, -1.5]]])
    else:
        raise ValueError(K)
    B = np.stack([np.vstack([np.zeros((1, 2)), a.T]) for a in A])
    return dict(B=B, S=np.stack(S), W=W, logxi0=np.log(np.full(K, 1.0 / K)), p=1)


def sim_z(T: int, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        z[t] = 0.98 * z[t - 1] + 0.2 * e[t]
    return z


def accuracy(prob: np.ndarray, s: np.ndarray) -> float:
    return float((prob.argmax(1) == s).mean())


def one_rep(K: int, T: int, seed: int, init: str = "blocks", n_init: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    par = dgp(K)
    T_all = T + 1000
    z = sim_z(T_all, rng)
    y, s = simulate_msvar(par, T_all, z[:, None], None, rng)
    t0 = time.time()
    m = TVTPMSVAR(K=K, p=1, l2=0.1, n_init=n_init, seed=seed + 1, init=init, max_iter=300, tol=1e-7)
    m.fit(y[:T], z[:T, None])
    sec = time.time() - t0
    m.align_to(par)                                                     # truth-aligned labels
    s_fit = s[1:T]
    acc_f, acc_s = accuracy(m.prob_filtered, s_fit), accuracy(m.prob_smoothed, s_fit)
    Yt, X, Zc = m.design(y, z[:, None])
    f = m.filter_design(Yt, X, Zc)
    acc_oos = accuracy(f["filt"][T - 1:], s[T:])
    acc_oos_pred = accuracy(f["pred"][T - 1:], s[T:])
    A_hat = np.stack([m.lag_matrix(j) for j in range(K)])
    A_true = np.stack([par["B"][j][1:3].T for j in range(K)])
    ccf_err = []
    for j in range(K):
        ch, ct = m.regime_ccf(j, 3), companion_ccf(A_true[j], par["S"][j], 3)
        ccf_err.append(max(float(np.nanmax(np.abs(ch[k] - ct[k]))) for k in range(-3, 4)))
    se = m.trans_se()
    slope_hat, slope_true, slope_se = m.W[:, :, 1], par["W"][:, :, 1], se[:, :, 1]
    nz = slope_true != 0
    return dict(K=K, T=T, seed=seed, init=init, sec=sec, ll=m.ll, iters=m.iters, converged=bool(m.converged),
                init_ll_spread=float(np.max(m.init_lls) - np.min(m.init_lls)), min_occupancy=m.min_occupancy,
                acc_filtered=acc_f, acc_smoothed=acc_s, acc_filtered_oos=acc_oos, acc_pred_oos=acc_oos_pred,
                A_err=(A_hat - A_true).tolist(), S_err=(m.S - par["S"]).tolist(), W_err=(m.W - par["W"]).tolist(),
                ccf_maxerr=ccf_err, slope_sign_ok=(np.sign(slope_hat[nz]) == np.sign(slope_true[nz])).tolist(),
                slope_cover=(np.abs(slope_hat[nz] - slope_true[nz]) <= 1.96 * slope_se[nz]).tolist(),
                slope_se=slope_se[nz].tolist())


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted({(r["K"], r["T"], r["init"]) for r in rows}):
        rr = [r for r in rows if (r["K"], r["T"], r["init"]) == key]
        d: dict[str, Any] = {"reps": len(rr)}
        for name in ("A_err", "S_err", "W_err"):
            e = np.array([r[name] for r in rr])
            d[name.replace("_err", "") + "_bias"] = np.round(e.mean(0), 3).tolist()
            d[name.replace("_err", "") + "_rmse"] = np.round(np.sqrt((e ** 2).mean(0)), 3).tolist()
            d[name.replace("_err", "") + "_rmse_mean"] = float(np.sqrt((e ** 2).mean()))
        for name in ("acc_filtered", "acc_smoothed", "acc_filtered_oos", "acc_pred_oos", "sec", "iters",
                     "init_ll_spread", "min_occupancy"):
            v = np.array([r[name] for r in rr], float)
            d[name] = {"mean": float(v.mean()), "median": float(np.median(v)), "q10": float(np.quantile(v, 0.1))}
        d["converged_share"] = float(np.mean([r["converged"] for r in rr]))
        d["degenerate_share"] = float(np.mean([r["min_occupancy"] < 0.02 for r in rr]))
        c = np.array([max(r["ccf_maxerr"]) for r in rr])
        d["ccf_maxerr_median"], d["ccf_maxerr_q90"] = float(np.nanmedian(c)), float(np.nanquantile(c, 0.9))
        sg = np.array([r["slope_sign_ok"] for r in rr], float)
        cv = np.array([r["slope_cover"] for r in rr], float)
        d["slope_sign_rate"] = np.round(sg.mean(0), 3).tolist()
        d["slope_sign_rate_min"] = float(sg.mean(0).min())
        d["slope_wald95_coverage"] = np.round(cv.mean(0), 3).tolist()
        out[f"K{key[0]}_T{key[1]}_{key[2]}"] = d
    return out


def h0_verdict(summ: dict[str, Any]) -> dict[str, Any]:
    crit = {}
    for K in (2, 3):
        d = summ.get(f"K{K}_T4000_blocks")
        if d is None:
            continue
        crit[f"K{K}"] = {"acc_oos": d["acc_filtered_oos"]["mean"] >= 0.85, "sign": d["slope_sign_rate_min"] >= 0.90,
                         "ccf": d["ccf_maxerr_median"] <= 0.05}
    allok = all(all(v.values()) for v in crit.values()) and len(crit) == 2
    k2fail = "K2" in crit and not all(crit["K2"].values())
    return {"criteria": crit, "verdict": "supports" if allok else ("contradicts" if k2fail else "inconclusive")}


def run_mc(reps: int, Ks: list[int], Ts: list[int], init_reps: int) -> dict[str, Any]:
    path = out_dir() / "mc_rows.jsonl"
    done = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["K"], r["T"], r["seed"], r["init"]))
    jobs = [(K, T, s, "blocks") for K in Ks for T in Ts for s in range(reps)]
    jobs += [(K, T, s, "dirichlet") for K in Ks for T in Ts for s in range(init_reps)]
    with path.open("a", encoding="utf-8") as fh:
        for K, T, s, init in jobs:
            if (K, T, s, init) in done:
                continue
            r = one_rep(K, T, 1000 * K + 10 * T + s, init)
            r["seed"] = s
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"K={K} T={T} rep={s} {init} acc_oos={r['acc_filtered_oos']:.3f} sec={r['sec']:.1f}", flush=True)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    summ = summarise(rows)
    res = {"summary": summ, "h0": h0_verdict(summ)}
    (out_dir() / "mc_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- synthetic CUSUM check


def corr_detector(y: np.ndarray, short: int = 24, long: int = 168, thr: float = 3.0, refractory: int = 24) -> np.ndarray:
    """Alarm when |fisher_z(corr over the last `short` rows) - fisher_z(corr over the last `long` rows)| /
    sqrt(1/(short-3) + 1/(long-3)) > thr. Uses rows up to and including t (causal)."""
    T = len(y)
    alarm = np.zeros(T, bool)
    last = -10 ** 9
    se = np.sqrt(1 / (short - 3) + 1 / (long - 3))
    for t in range(long - 1, T):
        a = np.corrcoef(y[t - short + 1:t + 1].T)[0, 1]
        b = np.corrcoef(y[t - long + 1:t + 1].T)[0, 1]
        d = abs(np.arctanh(np.clip(a, -0.999, 0.999)) - np.arctanh(np.clip(b, -0.999, 0.999))) / se
        if d > thr and t - last > refractory:
            alarm[t], last = True, t
    return alarm


def run_cusum_synth(reps: int) -> dict[str, Any]:
    """K=2 DGP, fit on 2000 rows, frozen-parameter filter on 1000 more; rows 2500..2699 come from an unmodelled regime
    (correlation -0.5, vol 1.5, no lead-lag). Detection delay and false alarms: CUSUM on the out-of-sample log score vs
    the rolling-correlation detector (both with the H-TVTP5 settings)."""
    par = dgp(2)
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(5000 + r)
        z = sim_z(3000, rng)
        y, s = simulate_msvar(par, 3000, z[:, None], None, rng)
        Sb = 2.25 * np.array([[1.0, -0.5], [-0.5, 1.0]])
        y[2500:2700] = rng.multivariate_normal(np.zeros(2), Sb, 200)
        m = TVTPMSVAR(K=2, p=1, l2=0.1, n_init=2, seed=r).fit(y[:2000], z[:2000, None])
        Yt, X, Zc = m.design(y, z[:, None])
        pll = m.filter_design(Yt, X, Zc)["pll"][1999:]                 # rows 2000..2999
        _, _, al_c = cusum(pll, burn=168, halflife=168, kappa=0.5, h=5.0, refractory=24)
        al_r = corr_detector(y[2000:])
        res = {}
        for name, al in (("cusum", al_c), ("corr", al_r)):
            idx = np.flatnonzero(al) + 2000
            hit = idx[(idx >= 2500) & (idx < 2700)]
            res[name] = {"delay": int(hit[0] - 2500) if len(hit) else None,
                         "false": int(((idx < 2500) & (idx >= 2000 + 168)).sum() + (idx >= 2724).sum())}
        rows.append(res)
    out = {}
    for name in ("cusum", "corr"):
        d = [x[name]["delay"] for x in rows]
        hits = [v for v in d if v is not None]
        out[name] = {"hit_rate": len(hits) / reps, "median_delay": float(np.median(hits)) if hits else None,
                     "false_per_1000": float(np.mean([x[name]["false"] for x in rows]) / ((2500 - 2168 + 276) / 1000))}
    res = {"reps": reps, "design": run_cusum_synth.__doc__, "result": out}
    (out_dir() / "mc_cusum.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mc")
    a.add_argument("--reps", type=int, default=50)
    a.add_argument("--K", default="2,3")
    a.add_argument("--T", default="1000,4000")
    a.add_argument("--init-reps", type=int, default=10)
    c = sub.add_parser("cusum")
    c.add_argument("--reps", type=int, default=30)
    args = ap.parse_args(argv)
    if args.cmd == "mc":
        res = run_mc(args.reps, [int(x) for x in args.K.split(",")], [int(x) for x in args.T.split(",")], args.init_reps)
        print(json.dumps(res["h0"], indent=1))
    else:
        print(json.dumps(run_cusum_synth(args.reps)["result"], indent=1))


if __name__ == "__main__":
    main()
