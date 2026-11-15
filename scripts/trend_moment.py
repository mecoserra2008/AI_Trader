"""H-TR5 / H-TR6: probit and XGBoost probability of a 'trend moment' week vs sideways (pre-registered 2026-11-08).

    PYTHONIOENCODING=utf-8 uv run python scripts/trend_moment.py [--record]

Target at each BTC UTC close t: y = 1 if |log C[t+7]/C[t]| >= sigma30_t * sqrt(7) (sigma30 = std of the 30 daily log
returns ending at t), else 0. Secondary y2 = 1 if next-7-day realised vol > sigma30_t. Features from
regime_context.state_vectors (causal) plus efficiency ratios and the 30d/180d vol ratio. Walk-forward from 2022-06-01,
refit every 21 days on rows with t <= fit_date - 8 (purged), fixed hyper-parameters. Metrics: Brier skill vs the
training base rate (14-day block bootstrap), AUC, log loss, reliability. --record appends verdicts to the registry.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from regime_context import state_vectors  # noqa: E402

from ai_trader.core.time import now  # noqa: E402
from ai_trader.storage.lake import Lake  # noqa: E402

warnings.filterwarnings("ignore")
FEATS_REGISTERED = ["r7", "r30", "dd90", "er7", "er30", "lrv30", "rv_ratio", "lrv_long", "ldvol", "vrp", "fng", "breadth_below_ma30",
                    "alt_rel30", "stable30", "fund7", "lvix", "hy_chg30", "spx_r7"]
# Primary spec (deviation logged before results): FRED's public HY OAS history starts 2023-11, so hy_chg30 is dropped.
FEATS = [f for f in FEATS_REGISTERED if f != "hy_chg30"]
START, REFIT, PURGE, CUTOFF = date(2022, 6, 1), 21, 8, date(2026, 7, 1)
XGB = {"max_depth": 2, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10, "lambda": 1.0,
       "objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": 4}
XGB_ROUNDS = 300


def build(t, feats=None) -> pl.DataFrame:
    feats = feats or FEATS
    sv = state_vectors(Lake(), t).sort("d")
    lc = pl.col("btc").log()
    r = lc.diff()
    sv = sv.with_columns(
        ((lc - lc.shift(7)).abs() / r.abs().rolling_sum(7)).alias("er7"),
        ((lc - lc.shift(30)).abs() / r.abs().rolling_sum(30)).alias("er30"),
        (pl.col("rv30") / (r.rolling_std(180) * math.sqrt(365))).log().alias("lrv_long"),
    )
    thr = pl.col("rv30") / math.sqrt(365) * math.sqrt(7)
    sv = sv.with_columns(
        pl.when(pl.col("fwd_r7").is_null()).then(None).otherwise((pl.col("fwd_r7").abs() >= thr).cast(pl.Int8)).alias("y"),
        pl.when(pl.col("fwd_rv7").is_null()).then(None).otherwise((pl.col("fwd_rv7") > pl.col("rv30")).cast(pl.Int8)).alias("y2"),
    )
    return sv.drop_nulls(feats).filter(pl.col("d") >= date(2021, 9, 1))


def roc_auc_score(y: np.ndarray, q: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    r = pl.Series(q).rank("average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(y: np.ndarray, q: np.ndarray) -> float:
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def fit_probit(X: np.ndarray, y: np.ndarray):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = sm.add_constant((X - mu) / sd, has_constant="add")
    try:
        m = sm.Probit(y, Z).fit(disp=0, method="bfgs", maxiter=500)
    except Exception:  # noqa: BLE001
        m = sm.Probit(y, Z).fit_regularized(disp=0, alpha=0.5)
    return lambda Xn: m.predict(sm.add_constant((Xn - mu) / sd, has_constant="add")), m, (mu, sd)


class _Booster:
    def __init__(self, b: xgb.Booster, k: int):
        g = b.get_score(importance_type="gain")
        tot = sum(g.values()) or 1.0
        self.feature_importances_ = np.array([g.get(f"f{i}", 0.0) / tot for i in range(k)])


def fit_xgb(X: np.ndarray, y: np.ndarray):
    b = xgb.train(XGB, xgb.DMatrix(X, label=y), num_boost_round=XGB_ROUNDS)
    return lambda Xn: b.predict(xgb.DMatrix(Xn)), _Booster(b, X.shape[1])


def walk_forward(df: pl.DataFrame, target: str) -> pl.DataFrame:
    d = df["d"].to_list()
    X = df.select(FEATS).to_numpy()
    y = df[target].to_numpy().astype(float)
    have = ~np.isnan(y)
    out = []
    fit_day = START
    last = d[-1]
    while fit_day <= last:
        train = np.array([(x <= fit_day - timedelta(days=PURGE)) for x in d]) & have
        test = np.array([(fit_day <= x < fit_day + timedelta(days=REFIT)) for x in d])
        if train.sum() > 200 and test.any():
            pp, *_ = fit_probit(X[train], y[train])
            px, _ = fit_xgb(X[train], y[train])
            base = y[train].mean()
            for i in np.where(test)[0]:
                out.append({"d": d[i], "y": y[i], "p_probit": float(pp(X[i:i + 1])[0]), "p_xgb": float(px(X[i:i + 1])[0]), "base": float(base)})
        fit_day += timedelta(days=REFIT)
    return pl.DataFrame(out).with_columns(pl.col("y").fill_nan(None))


def scores(p: pl.DataFrame, col: str, reps: int = 1000, block: int = 14) -> dict:
    p = p.drop_nulls("y")
    y, q, b = p["y"].to_numpy(), p[col].to_numpy(), p["base"].to_numpy()
    if len(y) < 30 or y.min() == y.max():
        return {"n": len(y)}
    bs, bref = np.mean((q - y) ** 2), np.mean((b - y) ** 2)
    rng = np.random.default_rng(0)
    n = len(y)
    starts = np.arange(0, n, block)
    boot = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, min(n, s + block)) for s in rng.choice(starts, size=len(starts))])
        boot.append(1 - np.mean((q[idx] - y[idx]) ** 2) / np.mean((b[idx] - y[idx]) ** 2))
    return {"n": n, "base_rate": float(y.mean()), "brier": float(bs), "brier_ref": float(bref), "bss": float(1 - bs / bref),
            "bss_ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))], "auc": float(roc_auc_score(y, q)),
            "logloss": float(log_loss(y, np.clip(q, 1e-6, 1 - 1e-6))), "logloss_ref": float(log_loss(y, np.clip(b, 1e-6, 1 - 1e-6)))}


def paired(p: pl.DataFrame, reps: int = 1000, block: int = 14) -> dict:
    p = p.drop_nulls("y")
    y, a, b = p["y"].to_numpy(), p["p_xgb"].to_numpy(), p["p_probit"].to_numpy()
    diff = (b - y) ** 2 - (a - y) ** 2  # > 0 = xgb better
    rng = np.random.default_rng(1)
    n = len(y)
    starts = np.arange(0, n, block)
    boot = [diff[np.concatenate([np.arange(s, min(n, s + block)) for s in rng.choice(starts, size=len(starts))])].mean() for _ in range(reps)]
    return {"brier_gain_xgb_vs_probit": float(diff.mean()), "ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]}


def reliability(p: pl.DataFrame, col: str) -> pl.DataFrame:
    return (p.drop_nulls("y").with_columns(pl.col(col).qcut(5, allow_duplicates=True).alias("bin"))
            .group_by("bin").agg(pl.col(col).mean().alias("mean_p"), pl.col("y").mean().alias("hit"), pl.len().alias("n")).sort("mean_p"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--registered", action="store_true", help="use the registered 18-feature spec (sample from 2023-12)")
    a = ap.parse_args()
    pl.Config.set_tbl_rows(40), pl.Config.set_tbl_cols(20), pl.Config.set_tbl_width_chars(220), pl.Config.set_float_precision(3)
    global FEATS
    if a.registered:
        FEATS = FEATS_REGISTERED
    print("spec:", "registered 18 features" if a.registered else "primary 17 features (no hy_chg30)")
    t = now()
    df = build(t)
    cur = df.tail(1)
    print(f"rows {df.height} ({df['d'][0]} .. {df['d'][-1]}); base rate y {df['y'].drop_nulls().mean():.3f}, y2 {df['y2'].drop_nulls().mean():.3f}")
    print(f"current state date {cur['d'][0]} (forecast window = next 7 days); BTC {cur['btc'][0]:.0f}; sigma30 {float(cur['rv30'][0])*100:.1f}% ann "
          f"-> trend threshold |7d move| >= {float(cur['rv30'][0])/math.sqrt(365)*math.sqrt(7)*100:.2f}%")
    results = {}
    for target in ["y", "y2"]:
        wf = walk_forward(df, target)
        wf = wf.with_columns((pl.col("d") + timedelta(days=7)).alias("t_end"))
        ins, hold = wf.filter(pl.col("t_end") < CUTOFF), wf.filter(pl.col("t_end") >= CUTOFF)
        sun = wf.filter(pl.col("d").dt.weekday() == 7)
        print(f"\n===== target {target} ({'trend moment' if target == 'y' else 'vol expansion'}) =====")
        for name, seg in [("llm_in_sample", ins), ("clean_holdout", hold), ("sundays_only (non-overlapping)", sun)]:
            for col in ["p_probit", "p_xgb"]:
                s = scores(seg, col)
                results[(target, name, col)] = s
                print(f"  {name:<31} {col:<9} " + (f"n={s['n']:<4} base={s['base_rate']:.2f} BSS={s['bss']:+.3f} CI90=[{s['bss_ci90'][0]:+.3f},{s['bss_ci90'][1]:+.3f}] AUC={s['auc']:.3f} "
                                                     f"logloss {s['logloss']:.3f} vs {s['logloss_ref']:.3f}" if "bss" in s else f"n={s['n']}"))
            if name != "sundays_only (non-overlapping)":
                pr = paired(seg)
                results[(target, name, "paired")] = pr
                print(f"  {'':<31} xgb vs probit Brier gain {pr['brier_gain_xgb_vs_probit']:+.4f} CI90 [{pr['ci90'][0]:+.4f},{pr['ci90'][1]:+.4f}]")
        print("  reliability (all walk-forward days), probit:")
        print(reliability(wf, "p_probit"))
        print("  reliability, xgb:")
        print(reliability(wf, "p_xgb"))
        # current probability from models fit on everything known now
        tr = df.filter((pl.col("d") <= cur["d"][0] - timedelta(days=PURGE)) & pl.col(target).is_not_null())
        X, y = tr.select(FEATS).to_numpy(), tr[target].to_numpy().astype(float)
        pp, pm, (mu, sd) = fit_probit(X, y)
        px, xm = fit_xgb(X, y)
        xc = cur.select(FEATS).to_numpy()
        p_pro, p_x = float(pp(xc)[0]), float(px(xc)[0])
        results[(target, "current")] = {"probit": p_pro, "xgb": p_x, "base": float(y.mean())}
        print(f"\n  CURRENT P({target}=1): probit {p_pro:.3f}   xgb {p_x:.3f}   base rate {y.mean():.3f}   (walk-forward history of these numbers above)")
        z = (xc[0] - mu) / sd
        coefs = np.asarray(pm.params)[1:]
        contrib = sorted(zip(FEATS, z, coefs, z * coefs), key=lambda r: -abs(r[3]))
        print("  probit drivers now (z-score x coefficient, top 8):", ", ".join(f"{f} z{zz:+.1f}·{c:+.2f}={v:+.2f}" for f, zz, c, v in contrib[:8]))
        imp = sorted(zip(FEATS, xm.feature_importances_), key=lambda r: -r[1])
        print("  xgb importance (top 8):", ", ".join(f"{f} {v:.3f}" for f, v in imp[:8]))
        # full-sample probit with HAC SEs for interpretation (overlapping targets -> maxlags 7)
        Zf = sm.add_constant((X - X.mean(0)) / (X.std(0) + 1e-12))
        try:
            hac = sm.Probit(y, Zf).fit(disp=0, method="bfgs", maxiter=500, cov_type="HAC", cov_kwds={"maxlags": 7})
            zt = sorted(zip(FEATS, hac.params[1:], hac.tvalues[1:]), key=lambda r: -abs(r[2]))
            print("  probit full-sample HAC t (top 8):", ", ".join(f"{f} {b:+.2f} (t {tt:+.1f})" for f, b, tt in zt[:8]))
        except Exception as e:  # noqa: BLE001
            print("  HAC probit failed:", e)
        if target == "y":
            # what the walk-forward probability has looked like recently
            print("  last 10 walk-forward predictions:")
            print(wf.tail(10).select("d", "y", "p_probit", "p_xgb", "base"))
    if a.record:
        from ai_trader.research.hypotheses import record
        for hid, col in [("H-TR5", "p_probit"), ("H-TR6", "p_xgb")]:
            for seg in ["llm_in_sample", "clean_holdout"]:
                s = results[("y", seg, col)]
                ok = s.get("bss", -1) > 0 and (seg == "clean_holdout" or s["bss_ci90"][0] > 0)
                if hid == "H-TR6":
                    ok = ok and results[("y", seg, "paired")]["brier_gain_xgb_vs_probit"] > 0
                verdict = "supports" if ok else ("inconclusive" if s.get("bss", -1) > 0 else "contradicts")
                m = {**s, "y2": results[("y2", seg, col)], **({"paired": results[("y", seg, "paired")]} if hid == "H-TR6" else {})}
                record(hid, seg, verdict, m, note=f"current P(y=1) {results[('y', 'current')]}", data="BTC daily since 2021-09, scripts/trend_moment.py")
        print("\nrecorded H-TR5/H-TR6")


if __name__ == "__main__":
    main()
