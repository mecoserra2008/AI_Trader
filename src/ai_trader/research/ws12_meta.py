"""WS12 ML benchmark, trees part (core of WS13): LightGBM meta-labelling of the 4h trend trades (H-WS12-6, H-WS12-7).

    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.ws12_meta features
    ... run [--record]

Trades: data/research/stress/trades/*.parquet, units zoo_ensemble and trend_4h, variant base, closed trades only (why !=
'end': positions still open at the stress as_of are excluded). Net R = stress.r_net (base costs: maker 2.0, taker 5.5,
pegged entry slippage, stop slippage by tier). Label = 1{R_net > 0}. Segment = entry_ts before / after the knowledge cutoff.
Features at entry (row t = the 1h bar opening at entry_ts; only bars t-1 and earlier; sigma = sqrt(s2_t), the robust base):
  ret24/72/240 = side * log(C_{t-1} / C_{t-1-k}) / (sigma sqrt(k)), same for BTC (btc24/72/240, side of the trade),
  lsig_rel1y, har_d/w/m (ws12_deep), lu_m24, jump24 (ml.hourly), fund = side * last funding (bp), doi = side * 24h
  change of log OI, lturn, stopsig = |entry - stop| / entry / sigma, hour, weekday, weekend, holiday, funding_win, tier
  (below 0, illiquid 1, mid 2, liquid 3), unit (trend_4h 0, zoo_ensemble 1), side.
Model: LightGBM binary, max_depth 3, num_leaves 7, 200 trees, lr 0.05, min_child_samples 40, bagging 0.8 / freq 1,
  feature_fraction 0.8, seeds 0, 1, 2. Walk-forward: test quarter Q from 2023Q1; training trades with exit_ts < Q start -
  1 day (purged) and entry before Q. Rule (fixed before results): take iff p_hat >= the training-fold base rate; the
  reported filter is the seed ensemble (mean p_hat), per-seed results alongside.
Tests: mean R of rejected trades with a date-clustered NW t (per-date sums of R - mean); accepted vs all; deflated Sharpe of
  the accepted per-trade stream with 6 trials (3 seeds x 2 units). SHAP: TreeSHAP (pred_contrib) mean |phi| on each
  fold's test trades; top-5 sets; mean pairwise Jaccard over folds x seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research import ws12_deep as wd

UNITS = ("trend_4h", "zoo_ensemble")
FEATS = ("ret24", "ret72", "ret240", "btc24", "btc72", "btc240", "lsig_rel1y", "har_d", "har_w", "har_m", "lu_m24", "jump24",
         "fund", "doi", "lturn", "stopsig", "hour", "weekday", "weekend", "holiday", "funding_win", "tier", "unit", "side")
TIER = {"below": 0, "illiquid": 1, "mid": 2, "liquid": 3}
SEEDS = (0, 1, 2)
PARAMS = {"objective": "binary", "max_depth": 3, "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 40,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1, "num_threads": 2}
N_TREES = 200


def trades() -> pl.DataFrame:
    from ai_trader.research.stress import r_net

    fs = glob.glob(str(load().root / "data" / "research" / "stress" / "trades" / "*.parquet"))
    d = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
    d = d.filter(pl.col("unit").is_in(UNITS) & (pl.col("variant") == "base") & (pl.col("why") != "end")).sort("entry_ts")
    return d.with_columns(pl.Series("R", r_net(d)))


def _sym_rows(sym: str) -> pl.DataFrame:
    df = wd.build_symbol(sym)
    r = df["r"].cast(pl.Float64).fill_null(0.0).to_numpy()
    cs = np.r_[0.0, np.cumsum(r)]
    n = len(r)
    idx = np.arange(n)
    s = np.sqrt(df["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy())
    cols = {}
    for k in (24, 72, 240):
        lo = idx - k
        v = np.where(lo >= 0, cs[idx] - cs[np.maximum(lo, 0)], np.nan)  # sum r[t-k..t-1]
        cols[f"m{k}"] = v / (s * math.sqrt(k))
    # step features of bar t-1 (known at the open of t)
    prev = {c: np.r_[np.nan, df[c].cast(pl.Float64).fill_null(np.nan).to_numpy()[:-1]] for c in ("fund", "doi", "lturn")}
    return df.select("event_ts", "lsig_rel1y", "har_d", "har_w", "har_m", "lu_m24", "jump24", "s2").with_columns(
        *[pl.Series(k, v) for k, v in {**cols, **prev}.items()])


def cmd_features() -> None:
    t = trades()
    syms = sorted(set(t["symbol"].to_list()) | {"BTCUSDT"})
    rows = {}
    for s in syms:
        rows[s] = _sym_rows(s)
        print(s, rows[s].height, flush=True)
    btc = rows["BTCUSDT"].select("event_ts", *[pl.col(f"m{k}").alias(f"b{k}") for k in (24, 72, 240)])
    out = []
    for s in syms:
        ts = t.filter(pl.col("symbol") == s)
        if ts.height == 0:
            continue
        ts = ts.with_columns(pl.col("entry_ts").dt.truncate("1h").alias("event_ts"))
        j = ts.join(rows[s], on="event_ts", how="left").join(btc, on="event_ts", how="left")
        side = j["side"].cast(pl.Float64)
        sig = np.sqrt(j["s2"].cast(pl.Float64).to_numpy())
        out.append(j.with_columns(
            (side * pl.col("m24")).alias("ret24"), (side * pl.col("m72")).alias("ret72"), (side * pl.col("m240")).alias("ret240"),
            (side * pl.col("b24")).alias("btc24"), (side * pl.col("b72")).alias("btc72"), (side * pl.col("b240")).alias("btc240"),
            (side * pl.col("fund")).alias("fund"), (side * pl.col("doi")).alias("doi"),
            pl.Series("stopsig", np.abs(j["entry"].to_numpy() - j["stop"].to_numpy()) / j["entry"].to_numpy() / sig),
            pl.col("entry_ts").dt.hour().cast(pl.Float64).alias("hour"), pl.col("entry_ts").dt.weekday().cast(pl.Float64).alias("weekday"),
            pl.col("weekend").cast(pl.Float64), pl.col("holiday").cast(pl.Float64), pl.col("funding_win").cast(pl.Float64),
            pl.col("tier").replace_strict(TIER, default=1).cast(pl.Float64).alias("tier"),
            (pl.col("unit") == "zoo_ensemble").cast(pl.Float64).alias("unit_f"), side.alias("side_f")))
    f = pl.concat(out, how="diagonal_relaxed").sort("entry_ts")
    f = f.select("unit", "symbol", "entry_ts", "exit_ts", "R", "seg", *[c for c in FEATS if c not in ("unit", "side")],
                 pl.col("unit_f").alias("unit_x"), pl.col("side_f").alias("side_x"))
    p = wd.root() / "meta_features.parquet"
    f.write_parquet(p)
    print(f.height, "trades ->", p)


def _cols() -> list[str]:
    return [c if c not in ("unit", "side") else f"{c}_x" for c in FEATS]


def cluster_t(R: np.ndarray, day: np.ndarray) -> tuple[float, float, int]:
    """Mean of R and a date-clustered NW t (Bartlett, automatic lag over dates)."""
    if len(R) < 10:
        return float(np.mean(R)) if len(R) else float("nan"), float("nan"), len(R)
    m = float(R.mean())
    ud, inv = np.unique(day, return_inverse=True)
    u = np.bincount(inv, weights=R - m, minlength=len(ud))
    D = len(ud)
    lag = int(np.floor(4 * (D / 100) ** (2 / 9)))
    lrv = u @ u / D
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / D
    se = math.sqrt(max(D * lrv, 1e-12)) / len(R)
    return m, m / se, len(R)


def cmd_run(record: bool = False) -> dict:
    import lightgbm as lgb

    from ai_trader.research.trend_zoo_study import deflated_sharpe

    f = pl.read_parquet(wd.root() / "meta_features.parquet")
    cols = _cols()
    X = f.select(cols).to_numpy().astype(float)
    R = f["R"].to_numpy().astype(float)
    y = (R > 0).astype(float)
    ent = f["entry_ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy()
    ex = f["exit_ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy()
    seg = f["seg"].to_numpy()
    unit = f["unit"].to_numpy()
    qs = np.arange(np.datetime64("2023-01"), np.datetime64("2027-01"), np.timedelta64(3, "M")).astype("datetime64[us]")
    P = np.full((len(SEEDS), len(R)), np.nan)
    base = np.full(len(R), np.nan)
    shap_top: list[set] = []
    shap_mean = []
    for qi in range(len(qs) - 1):
        a, b = qs[qi], qs[qi + 1]
        te = (ent >= a) & (ent < b)
        tr = (ex < a - np.timedelta64(1, "D")) & (ent < a)
        if te.sum() == 0 or tr.sum() < 200:
            continue
        base[te] = y[tr].mean()
        for si, s in enumerate(SEEDS):
            ds = lgb.Dataset(X[tr], y[tr], free_raw_data=False)
            bst = lgb.train({**PARAMS, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}, ds, N_TREES)
            P[si, te] = bst.predict(X[te])
            phi = np.abs(bst.predict(X[te], pred_contrib=True)[:, :-1]).mean(axis=0)
            shap_mean.append(phi)
            shap_top.append(set(np.argsort(-phi)[:5].tolist()))
    have = np.isfinite(base)
    pe = np.nanmean(P, axis=0)
    out: dict = {"n_trades": int(have.sum()), "features": cols, "segments": {}, "per_seed": {}, "per_unit": {}}
    day = ent.astype("datetime64[D]")

    def summary(take: np.ndarray, msk: np.ndarray) -> dict:
        rj = msk & ~take
        ac = msk & take
        m_all, t_all, n_all = cluster_t(R[msk], day[msk])
        m_rej, t_rej, n_rej = cluster_t(R[rj], day[rj])
        m_acc, t_acc, n_acc = cluster_t(R[ac], day[ac])
        return {"all": [m_all, t_all, n_all], "rejected": [m_rej, t_rej, n_rej], "accepted": [m_acc, t_acc, n_acc],
                "take_rate": float(take[msk].mean()) if msk.any() else float("nan"),
                "net_R_per_candidate_filtered": float((R[msk] * take[msk]).mean()) if msk.any() else float("nan"),
                "auc": _auc(pe[msk], y[msk]) if msk.any() else float("nan")}

    take = have & (pe >= base)
    for sg in ("llm_in_sample", "clean_holdout"):
        msk = have & (seg == sg)
        out["segments"][sg] = summary(take, msk)
        out["per_seed"][sg] = [summary(have & (P[si] >= base), msk) for si in range(len(SEEDS))]
        out["per_unit"][sg] = {u: summary(take, msk & (unit == u)) for u in UNITS}
    # deflated Sharpe of the accepted per-trade stream (6 trials: seeds x units), pooled segments
    srs, streams = [], {}
    for si in range(len(SEEDS)):
        for u in UNITS:
            m = have & (unit == u) & (P[si] >= base)
            x = R[m]
            streams[(si, u)] = x
            srs.append(x.mean() / x.std(ddof=1))
    out["deflated_sharpe"] = {f"{u}": deflated_sharpe(R[have & (unit == u) & take], np.array(srs), 6) for u in UNITS}
    # SHAP stability
    jac = [len(a & b) / len(a | b) for i, a in enumerate(shap_top) for b in shap_top[i + 1:]]
    mean_phi = np.mean(np.vstack(shap_mean), axis=0)
    order = np.argsort(-mean_phi)
    out["shap"] = {"mean_jaccard_top5": float(np.mean(jac)), "n_fits": len(shap_top),
                   "ranking": [[cols[i], float(mean_phi[i])] for i in order],
                   "top5_frequency": {cols[i]: float(np.mean([i in s for s in shap_top])) for i in order[:10]}}
    v6 = {}
    for sg in ("llm_in_sample", "clean_holdout"):
        m, t, n = out["segments"][sg]["rejected"]
        v6[sg] = "supports" if (m < 0 and (t < -2 if sg == "llm_in_sample" else True)) else ("contradicts" if m >= 0 else "inconclusive")
    v6["pooled"] = "supports" if (v6["llm_in_sample"] == "supports" and v6["clean_holdout"] == "supports") else (
        "contradicts" if v6["llm_in_sample"] == "contradicts" else "inconclusive")
    mj = out["shap"]["mean_jaccard_top5"]
    out["verdicts"] = {"H-WS12-6": v6, "H-WS12-7": "supports" if mj >= 0.5 else ("contradicts" if mj < 0.3 else "inconclusive")}
    p = load().root / "research" / "reports" / f"ws12_meta_{wd.TAG}.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"segments": out["segments"], "verdicts": out["verdicts"], "shap": out["shap"]["mean_jaccard_top5"]}, indent=1, default=float))
    if record:
        from ai_trader.research.hypotheses import record as rec

        data = "stress trades (base variant) zoo_ensemble + trend_4h, closed, as of 2026-11-25; quarterly purged walk-forward from 2023Q1"
        for sg in ("llm_in_sample", "clean_holdout"):
            rec("H-WS12-6", sg, v6[sg], out["segments"][sg], data=data, note="ensemble p_hat >= training base rate; rejected-trade mean R")
        rec("H-WS12-6", "pooled", v6["pooled"], {"deflated_sharpe": out["deflated_sharpe"]}, data=data)
        rec("H-WS12-7", "pooled", out["verdicts"]["H-WS12-7"], {"mean_jaccard_top5": mj, "top5_frequency": out["shap"]["top5_frequency"]}, data=data)
    return out


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    rk = rankdata(p)
    return float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    r = sub.add_parser("run")
    r.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "features":
        cmd_features()
    else:
        cmd_run(a.record)


if __name__ == "__main__":
    main()


