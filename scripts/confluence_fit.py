"""Walk-forward fit of forward returns and move size on the trader's confluence readings (research/desk/CONFLUENCE.md).

    PYTHONIOENCODING=utf-8 uv run python scripts/confluence_fit.py --hyp H-CONF1 [--horizons 4,24] [--lam 1.0] [--force-small]

Refuses to fit until >= 300 cycles are logged (data/journal/confluence/) and until the hypothesis named by --hyp is
registered in research/hypotheses.yaml (research protocol: register before testing). Models are linear-first and
regularised (house rules), fit on standardised layer readings (null = 0 = no view):
  1. ridge:    vol-scaled forward log return z = r / (sigma_1h sqrt(H))         on the 9 signed layers
  2. logistic: P(r > 0), L2-penalised (Newton)                                  on the 9 signed layers
  3. ridge:    move size |z|                                                    on |layer| and vol_liq
Walk-forward: cycles sorted by as_of, 5 expanding folds, training cycles within H hours of the test block purged.
Outcomes are read point in time from the lake (1h bars closed by as_of + H). Reports the fold-averaged weights with
their spread, out-of-sample IC, hit rate, Brier vs the base rate and R^2 vs zero, per horizon, as JSON + markdown
under research/reports/confluence_fit_<date>.{json,md}. `--force-small` runs the machinery on fewer cycles for
testing only and marks the output NOT EVIDENCE.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.narrative.confluence import LAYERS, MIN_CYCLES_TO_FIT, logged_cycles, rows


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]
    P = lam * np.eye(Xb.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + P, Xb.T @ y)


def logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 50) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    P = lam * np.eye(Xb.shape[1])
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + P @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def outcomes(df: pl.DataFrame, H: int) -> pl.DataFrame:
    from ai_trader.storage.lake import Lake

    lake = Lake()
    t_now = datetime.now(UTC)
    out = []
    for sym in df["symbol"].unique().to_list():
        sub = df.filter(pl.col("symbol") == sym)
        t0 = min(sub["ts"].to_list()) - timedelta(hours=730)
        b = lake.bars(sym, "1h", t_now, start=t0).select("event_ts", "close").sort("event_ts")
        b = b.with_columns((pl.col("event_ts") + pl.duration(hours=1)).alias("close_ts"))
        ct = b["close_ts"].to_list()
        c = b["close"].to_numpy().astype(float)
        lr = np.r_[np.nan, np.diff(np.log(c))]
        idx_of = {t: i for i, t in enumerate(ct)}
        for r in sub.iter_rows(named=True):
            ts = r["ts"].replace(minute=0, second=0, microsecond=0)
            i0 = idx_of.get(ts)
            if i0 is None or i0 + H >= len(c) or i0 < 720:
                continue
            sig = float(np.nanstd(lr[i0 - 719:i0 + 1], ddof=1))
            ret = math.log(c[i0 + H] / c[i0])
            z = ret / (sig * math.sqrt(H)) if sig > 0 else float("nan")
            out.append({"cycle_id": r["cycle_id"], "symbol": sym, f"r{H}": ret, f"z{H}": z})
    return pl.DataFrame(out) if out else pl.DataFrame({"cycle_id": [], "symbol": [], f"r{H}": [], f"z{H}": []})


def walk_forward(W: pl.DataFrame, H: int, lam: float, folds: int = 5) -> dict:
    W = W.sort("ts")
    X = W.select(list(LAYERS)).to_numpy().astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    z = W[f"z{H}"].to_numpy()
    up = (z > 0).astype(float)
    ts = W["ts"].to_list()
    cyc = W["cycle_id"].to_list()
    ucyc = list(dict.fromkeys(cyc))
    edges = np.linspace(len(ucyc) // 2, len(ucyc), folds + 1).astype(int)   # first half = initial training window
    pred_z, pred_p, pred_m, ys, ys_up, ys_m = [], [], [], [], [], []
    wz, wp, wm = [], [], []
    for k in range(folds):
        test_c = set(ucyc[edges[k]:edges[k + 1]])
        if not test_c:
            continue
        te = np.array([c in test_c for c in cyc])
        t_start = min(t for t, f in zip(ts, te, strict=True) if f)
        tr = np.array([(t + timedelta(hours=H) < t_start) for t in ts]) & ~te
        if tr.sum() < 30:
            continue
        mu_k, sd_k = X[tr].mean(0), X[tr].std(0)
        sd_k[sd_k == 0] = 1.0
        Xs = (X - mu_k) / sd_k
        Xa = np.c_[np.abs(X), X[:, LAYERS.index("vol_liq")]]
        mua, sda = Xa[tr].mean(0), Xa[tr].std(0)
        sda[sda == 0] = 1.0
        Xas = (Xa - mua) / sda
        b1 = ridge(Xs[tr], z[tr], lam)
        b2 = logistic(Xs[tr], up[tr], lam)
        b3 = ridge(Xas[tr], np.abs(z[tr]), lam)
        wz.append(b1[1:])
        wp.append(b2[1:])
        wm.append(b3[1:])
        pred_z += list(np.c_[np.ones(te.sum()), Xs[te]] @ b1)
        pred_p += list(1 / (1 + np.exp(-(np.c_[np.ones(te.sum()), Xs[te]] @ b2))))
        pred_m += list(np.c_[np.ones(te.sum()), Xas[te]] @ b3)
        ys += list(z[te])
        ys_up += list(up[te])
        ys_m += list(np.abs(z[te]))
    if not ys:
        return {"error": "no fold had enough training data"}
    pz, pp, pm, y, yu, ym = map(np.asarray, (pred_z, pred_p, pred_m, ys, ys_up, ys_m))
    base = float(np.mean(up))
    return {
        "n_rows": int(len(W)), "n_cycles": len(ucyc), "oos_rows": int(len(y)), "folds": len(wz), "lambda": lam,
        "weights_return": {L: [float(np.mean(c)), float(np.std(c))] for L, c in zip(LAYERS, np.array(wz).T, strict=True)},
        "weights_up_logit": {L: [float(np.mean(c)), float(np.std(c))] for L, c in zip(LAYERS, np.array(wp).T, strict=True)},
        "weights_move": {n: [float(np.mean(c)), float(np.std(c))] for n, c in
                         zip([f"|{L}|" for L in LAYERS] + ["vol_liq"], np.array(wm).T, strict=True)},
        "oos": {"ic_return": _spearman(pz, y), "r2_vs_zero": float(1 - np.sum((y - pz) ** 2) / np.sum(y**2)),
                "hit_rate": float(np.mean(np.sign(pz) == np.sign(y))), "brier": float(np.mean((pp - yu) ** 2)),
                "brier_base_rate": float(np.mean((base - yu) ** 2)), "ic_move": _spearman(pm, ym)},
        "note": "rows are (cycle, symbol); cycles overlap in time and symbols co-move: treat |IC| < 0.05 and small Brier gains as noise",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="registered hypothesis id this fit tests (e.g. H-CONF1)")
    ap.add_argument("--horizons", default="4,24")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--force-small", action="store_true")
    a = ap.parse_args()
    from ai_trader.research.hypotheses import registry

    if a.hyp not in registry():
        print(f"{a.hyp} is not registered in research/hypotheses.yaml: register the test before fitting", file=sys.stderr)
        return 2
    n = len(logged_cycles())
    if n < MIN_CYCLES_TO_FIT and not a.force_small:
        print(f"{n} cycles logged; the fit needs >= {MIN_CYCLES_TO_FIT}. Nothing fitted.")
        return 0
    long = pl.DataFrame(rows())
    if long.height == 0:
        print("no confluence rows")
        return 0
    W = (long.pivot(on="layer", index=["cycle_id", "as_of", "symbol"], values="value", aggregate_function="first")
         .with_columns(pl.col("as_of").str.to_datetime(time_zone="UTC").alias("ts")))
    for L in LAYERS:
        if L not in W.columns:
            W = W.with_columns(pl.lit(None, dtype=pl.Float64).alias(L))
    W = W.with_columns([pl.col(L).cast(pl.Float64).fill_null(0.0) for L in LAYERS])
    res = {"as_of": datetime.now(UTC).isoformat(), "hypothesis": a.hyp, "cycles": n, "evidence": not a.force_small, "horizons": {}}
    for H in (int(x) for x in a.horizons.split(",")):
        o = outcomes(W.select("cycle_id", "symbol", "ts"), H)
        J = W.join(o, on=["cycle_id", "symbol"], how="inner").filter(pl.col(f"z{H}").is_finite())
        res["horizons"][f"{H}h"] = walk_forward(J, H, a.lam) if J.height >= 60 else {"error": f"only {J.height} rows with outcomes"}
    out = load().root / "research" / "reports" / f"confluence_fit_{datetime.now(UTC):%Y-%m-%d}.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    lines = [f"# Confluence fit {res['as_of'][:16]} UTC ({a.hyp}{'' if res['evidence'] else ', NOT EVIDENCE: --force-small'})", ""]
    for h, r in res["horizons"].items():
        lines.append(f"## {h}")
        if "error" in r:
            lines += [r["error"], ""]
            continue
        lines.append(f"cycles {r['n_cycles']}, OOS rows {r['oos_rows']}, folds {r['folds']}, lambda {r['lambda']}; OOS IC {r['oos']['ic_return']:+.3f}, "
                     f"hit {r['oos']['hit_rate']:.3f}, Brier {r['oos']['brier']:.4f} vs base {r['oos']['brier_base_rate']:.4f}, move IC {r['oos']['ic_move']:+.3f}")
        lines += ["", "| layer | w(return) | w(P up) | w(move, abs) |", "|---|---|---|---|"]
        for L in LAYERS:
            wr, wp, wm = r["weights_return"][L], r["weights_up_logit"][L], r["weights_move"][f"|{L}|"]
            lines.append(f"| {L} | {wr[0]:+.3f} ± {wr[1]:.3f} | {wp[0]:+.3f} ± {wp[1]:.3f} | {wm[0]:+.3f} ± {wm[1]:.3f} |")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
