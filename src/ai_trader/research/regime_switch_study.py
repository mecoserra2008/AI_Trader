"""Trend-vs-mean-reversion regime switch: H-TR4 (single feature) and H-TR3 (walk-forward logistic classifier).

Data (RESEARCH_PROTOCOL §3b: everything on the FULL history, then split by segment at the knowledge cutoff)
  features   features.regime_features on Bybit 4h market_frame(funding=False, oi=False), one row per UTC day D,
             known at 00:00 UTC of D+1.
  MR         equal-weight signal of zscore_reversion and rsi_reversion (as configured, on 4h bars), evaluated
             with evaluate.strategy_pnl (risk units, 7.5 bps per side); daily net P&L = sum over rows by UTC
             date of event_ts (the H-X2b convention).
  trend      daily net P&L of the trend-zoo equal-weight ensemble, data/priors/trend_zoo_daily_4h.parquet
             (columns symbol, date, variant, pnl; variant == 'ensemble'), written by the trend-zoo study.

Design (fixed before results; choices the registry leaves open are marked *)
  sample     symbol-weeks: decision at Monday 00:00 UTC; features of Sunday D (known at the decision);
             outcome = sum of daily net P&L over dates D+1..D+7 (all 7 days present for both ensembles).
             Under the event_ts dating, P&L dated D+k is earned on returns from 04:00 D+k to 04:00 D+k+1, so the
             outcome never overlaps the feature window.
  segments   llm_in_sample if D+8 < cutoff (outcome fully realised before it); clean_holdout if D+1 >= cutoff;
             the straddling week is dropped.
  H-TR4      x = er_14d. * Tercile cut points are the in-sample 1/3 and 2/3 quantiles of er_14d over all
             in-sample symbol-weeks; the holdout reuses them (no holdout information). Statistic: per week, mean
             next-7d trend P&L of top-tercile symbols minus bottom-tercile symbols (weeks with both), mean and NW t
             over weeks (= clustered by week). Also the pooled OLS slope on 1{top} with week-clustered SE.
  H-TR3      inputs: features.regime_features.FEATURES (the 11 registry features); target y = 1{trend7 - MR7 > 0}.
             Monthly walk-forward: for month m, train on every symbol-week with D + 9 <= m (outcome fully known
             at the start of m), * >= MIN_TRAIN_WEEKS distinct weeks; features standardised with the training mean
             / sd; logistic regression with * a light ridge (RIDGE on standardised coefficients, intercept free)
             for numerical stability; predict p for decisions D+1 in month m. Rows with a missing feature are
             dropped.
             IC = Spearman of p vs realised (trend7 - MR7) over the pooled OOS symbol-weeks of the segment;
             * its t = OLS slope of standardised ranks, SE clustered by week. (Also: mean weekly cross-sectional
             IC, NW t.) Gated = trend-ensemble daily P&L on D+1..D+7 if p > 0.5 else 0; ungated = the trend
             ensemble on the same days. Sharpe = annualised (sqrt 365) Sharpe of each symbol's daily series
             (> MIN_DAYS days; a never-traded gated series counts as Sharpe 0), averaged over symbols.
  verdicts   (registry) H-TR4: diff > 0, NW t > 2 in-sample; diff > 0 in holdout.
             H-TR3: IC > 0 with t > 2 AND gated Sharpe > ungated in-sample; IC > 0 AND gated > ungated holdout.
             Holdout with < MIN_HOLDOUT_WEEKS weeks is inconclusive.
  checks     random-gate null (same exposure, N_NULL draws) for the gated Sharpe; walk-forward placebo with
             outcomes permuted across whole weeks (N_PLACEBO runs) for the IC t (leakage / selection check);
             coefficient paths of the monthly refits and per-calendar-year fits (stability).

    uv run python -m ai_trader.research.regime_switch_study --segment llm_in_sample [--record] [--out f.json]
    uv run python -m ai_trader.research.regime_switch_study --segment clean_holdout [--record] [--out f.json] [--save]
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features.regime_features import FEATURES, regime_features
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.frame import market_frame
from ai_trader.storage.lake import Lake
from ai_trader.strategies import REGISTRY

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
MR = ("zscore_reversion", "rsi_reversion")
SEGS = ("llm_in_sample", "clean_holdout")
HORIZON = 7
MIN_TRAIN_WEEKS = 52
RIDGE = 1.0
MIN_DAYS = 30
MIN_HOLDOUT_WEEKS = 8
N_NULL = 500
N_PLACEBO = 100
ZOO_FILE = "data/priors/trend_zoo_daily_4h.parquet"
MODEL_FILE = "data/priors/regime_switch_4h_v0.parquet"
PANEL_FILE = "data/priors/regime_switch_panel_4h_v0.parquet"


def _root() -> Path:
    return load().root


# ------------------------------------------------------------------ data

def symbol_daily(sym: str, as_of: datetime, lake: Lake | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(daily features, daily MR-ensemble net P&L) for one symbol, from its full 4h history."""
    f = market_frame(sym, "4h", as_of, funding=False, oi=False, lake=lake)
    feats = regime_features(f).with_columns(pl.lit(sym).alias("symbol"))
    s = 0.5 * (REGISTRY[MR[0]]()(f) + REGISTRY[MR[1]]()(f))
    p = strategy_pnl(f, s)
    mr = (p.group_by(pl.col("event_ts").dt.date().alias("date")).agg(pl.col("pnl").sum().alias("mr"), pl.len().alias("n_bars"))
          .with_columns(pl.lit(sym).alias("symbol")).sort("date"))
    # the last bar's forward return is unknown (P&L 0): drop the last, incomplete P&L date
    return feats, mr.filter(pl.col("date") < mr["date"].max())


def build_daily(as_of: datetime | None = None, symbols=UNIVERSE, cache: str = "") -> tuple[pl.DataFrame, pl.DataFrame]:
    if cache and Path(cache + ".feat.parquet").is_file():
        return pl.read_parquet(cache + ".feat.parquet"), pl.read_parquet(cache + ".mr.parquet")
    as_of = as_of or now()
    fs, ms = [], []
    for sym in symbols:
        f, m = symbol_daily(sym, as_of)
        if f.height:
            fs.append(f)
            ms.append(m)
    feats, mr = pl.concat(fs), pl.concat(ms)
    if cache:
        feats.write_parquet(cache + ".feat.parquet")
        mr.write_parquet(cache + ".mr.parquet")
    return feats, mr


def load_trend(path: str | Path | None = None) -> pl.DataFrame:
    p = Path(path) if path else _root() / ZOO_FILE
    z = pl.read_parquet(p).filter(pl.col("variant") == "ensemble")
    if z.schema["date"] != pl.Date:
        z = z.with_columns(pl.col("date").cast(pl.Date))
    return z.select("symbol", "date", pl.col("pnl").cast(pl.Float64).alias("trend")).unique(["symbol", "date"]).sort("symbol", "date")


def daily_join(mr: pl.DataFrame, trend: pl.DataFrame) -> pl.DataFrame:
    """Daily (symbol, date, trend, mr) on common dates, stamped with the decision Sunday D they belong to."""
    d = mr.select("symbol", "date", "mr").join(trend, on=["symbol", "date"], how="inner")
    return d.with_columns((pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("D")).sort("symbol", "date")


def weekly_panel(feats: pl.DataFrame, daily: pl.DataFrame, cut: date | None = None) -> pl.DataFrame:
    cut = cut or knowledge_cutoff().date()
    agg = daily.group_by("symbol", "D").agg(pl.col("trend").sum().alias("trend7"), pl.col("mr").sum().alias("mr7"), pl.len().alias("n_days"))
    agg = agg.filter(pl.col("n_days") == HORIZON)
    dec = feats.filter(pl.col("date").dt.weekday() == 7).rename({"date": "D"})
    p = dec.join(agg, on=["symbol", "D"], how="inner").with_columns((pl.col("trend7") - pl.col("mr7")).alias("diff"))
    seg = (pl.when(pl.col("D") + pl.duration(days=8) < cut).then(pl.lit("llm_in_sample"))
           .when(pl.col("D") + pl.duration(days=1) >= cut).then(pl.lit("clean_holdout")).otherwise(None))
    return p.with_columns(seg.alias("segment"), (pl.col("diff") > 0).cast(pl.Int8).alias("y")).sort("D", "symbol")


# ------------------------------------------------------------------ stats helpers

def _nw(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m, t, n = newey_west_t(x)
    if n < 10 and n >= 3 and x.std(ddof=1) > 0:
        t = float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
    return {"mean": m, "t": t, "n": n}


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = X * (y - X @ b)[:, None]
    _, inv = np.unique(groups, return_inverse=True)
    g = int(inv.max()) + 1
    U = np.zeros((g, k))
    np.add.at(U, inv, u)
    c = g / (g - 1) * (n - 1) / (n - k) if g > 1 else 1.0
    V = c * XtXi @ (U.T @ U) @ XtXi
    return b, b / np.sqrt(np.diag(V)), g


def fit_logit(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Ridge logistic regression by Newton; returns (b incl. intercept first, model-based SE)."""
    Z = np.c_[np.ones(len(y)), X]
    k = Z.shape[1]
    P = ridge * np.eye(k)
    P[0, 0] = 0.0
    b = np.zeros(k)
    H = np.eye(k)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ b, -35, 35)))
        g = Z.T @ (y - p) - P @ b
        H = (Z * (p * (1 - p))[:, None]).T @ Z + P
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _month_add(d: date, k: int = 1) -> date:
    y, m = divmod(d.month - 1 + k, 12)
    return date(d.year + y, m + 1, 1)


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= MIN_DAYS:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


# ------------------------------------------------------------------ walk-forward (H-TR3)

def walk_forward(panel: pl.DataFrame, through: date | None = None, y_col: str = "y", keep_models: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Monthly expanding-window logistic refits. Returns (OOS predictions, one model row per refit month).
    Models are also fitted for months up to `through` (e.g. the current month) even if they have no test rows."""
    ok = panel.filter(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURES]) & pl.col(y_col).is_not_null())
    X = ok.select(FEATURES).to_numpy()
    y = ok[y_col].to_numpy().astype(float)
    D = ok["D"].to_numpy().astype("datetime64[D]")
    dec = D + np.timedelta64(1, "D")
    m0 = ok["D"].min()
    last = max(ok["D"].max() + timedelta(days=1), through) if through else ok["D"].max() + timedelta(days=1)
    m = date(m0.year, m0.month, 1)
    preds = np.full(ok.height, np.nan)
    models = []
    while m <= last:
        m_np = np.datetime64(m, "D")
        tr = D + np.timedelta64(9, "D") <= m_np
        if len(np.unique(D[tr])) >= MIN_TRAIN_WEEKS:
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            b, se = fit_logit((X[tr] - mu) / sd, y[tr])
            te = (dec >= m_np) & (dec < np.datetime64(_month_add(m), "D"))
            if te.any():
                preds[te] = _sigmoid(b[0] + ((X[te] - mu) / sd) @ b[1:])
            if keep_models:
                models.append({"fit_month": m, "n_train": int(tr.sum()), "n_weeks": int(len(np.unique(D[tr]))), "base_rate": float(y[tr].mean()),
                               "intercept": float(b[0]), **{f"b_{c}": float(v) for c, v in zip(FEATURES, b[1:], strict=True)},
                               **{f"se_{c}": float(v) for c, v in zip(FEATURES, se[1:], strict=True)},
                               **{f"mu_{c}": float(v) for c, v in zip(FEATURES, mu, strict=True)},
                               **{f"sd_{c}": float(v) for c, v in zip(FEATURES, sd, strict=True)}})
        m = _month_add(m)
    out = ok.with_columns(pl.Series("p", preds)).filter(pl.col("p").is_not_nan())
    return out, pl.DataFrame(models)


def ic_stats(pred: pl.DataFrame) -> dict:
    p, d = pred["p"].to_numpy(), pred["diff"].to_numpy()
    wk = pred["D"].cast(pl.Int32).to_numpy()
    out = {"n_obs": pred.height, "n_weeks": int(len(np.unique(wk))), "n_symbols": pred["symbol"].n_unique()}
    if pred.height < 20:
        return {**out, "ic": float("nan"), "t": float("nan")}
    rx, ry = rankdata(p), rankdata(d)
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    b, t, g = ols_cluster(ry, np.c_[np.ones(len(rx)), rx], wk)
    out.update({"ic": float(spearmanr(p, d).statistic), "ic_slope": float(b[1]), "t": float(t[1]), "n_clusters": g})
    ics = []
    for _, sub in pred.group_by("D"):
        if sub.height >= 5 and sub["p"].std() > 0:
            ics.append(spearmanr(sub["p"].to_numpy(), sub["diff"].to_numpy()).statistic)
    w = _nw(np.array(ics))
    out.update({"xs_ic_mean": w["mean"], "xs_ic_t": w["t"], "xs_ic_weeks": w["n"]})
    yv = pred["y"].to_numpy()
    out.update({"base_rate_trend_wins": float(yv.mean()), "share_p_gt_half": float((p > 0.5).mean()),
                "accuracy": float(((p > 0.5) == (yv == 1)).mean()), "mean_p": float(p.mean()),
                "mean_diff_p_gt_half": float(d[p > 0.5].mean()) if (p > 0.5).any() else float("nan"),
                "mean_diff_p_le_half": float(d[p <= 0.5].mean()) if (p <= 0.5).any() else float("nan")})
    return out


def gate_stats(pred: pl.DataFrame, daily: pl.DataFrame, seed: int = 0) -> dict:
    """Gated vs ungated trend-ensemble Sharpe on the OOS days of `pred`, + random-gate null."""
    dd = daily.join(pred.select("symbol", "D", "p"), on=["symbol", "D"], how="inner").sort("symbol", "date")
    dd = dd.with_columns((pl.col("p") > 0.5).alias("gate"))
    rows = []
    for sym, sub in dd.group_by("symbol"):
        tr = sub["trend"].to_numpy()
        gt = sub["gate"].to_numpy()
        rows.append({"symbol": sym[0], "days": sub.height, "exposure": float(gt.mean()), "ungated": sharpe(tr), "gated": sharpe(tr * gt),
                     "mr": sharpe(sub["mr"].to_numpy())})
    per = pl.DataFrame(rows).filter(pl.col("ungated").is_finite() & pl.col("gated").is_finite()).sort("symbol")
    out = {"n_symbols": per.height, "days": int(per["days"].sum()) if per.height else 0}
    if not per.height:
        return {**out, "gated": float("nan"), "ungated": float("nan")}
    out.update({"ungated": float(per["ungated"].mean()), "gated": float(per["gated"].mean()), "mr": float(per["mr"].mean()),
                "exposure": float(dd["gate"].mean())})
    dif = (per["gated"] - per["ungated"]).to_numpy()
    out["diff"] = float(dif.mean())
    out["diff_t_over_symbols"] = float(dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif)))) if len(dif) > 2 and dif.std(ddof=1) > 0 else float("nan")
    out["share_symbols_gated_better"] = float((dif > 0).mean())
    # random-gate null: each OOS symbol-week traded with prob = observed exposure
    keep = set(per["symbol"].to_list())
    dk = dd.filter(pl.col("symbol").is_in(keep))
    syms = dk["symbol"].to_numpy()
    wkid = np.unique(np.char.add(syms.astype(str), dk["D"].cast(pl.Utf8).to_numpy().astype(str)), return_inverse=True)[1]
    tr = dk["trend"].to_numpy()
    q = float(dk["gate"].mean())
    rng = np.random.default_rng(seed)
    order = {s: np.flatnonzero(syms == s) for s in keep}
    null = []
    for _ in range(N_NULL):
        g = rng.random(wkid.max() + 1) < q
        gt = tr * g[wkid]
        null.append(np.mean([sharpe(gt[ix]) for ix in order.values()]))
    null = np.array(null)
    out.update({"null_gated_mean": float(null.mean()), "null_gated_q95": float(np.quantile(null, 0.95)),
                "null_p_value": float((null >= out["gated"]).mean())})
    out["per_symbol"] = per.to_dicts()
    return out


def placebo_ic(panel: pl.DataFrame, seg: str, cut: date, n: int = N_PLACEBO, seed: int = 1) -> dict:
    """Walk-forward IC t with outcomes permuted across whole weeks (cross-section kept): should be ~N(0,1)."""
    rng = np.random.default_rng(seed)
    weeks = np.array(sorted(panel["D"].unique().to_list()))
    ts = []
    for _ in range(n):
        perm = dict(zip(weeks, rng.permutation(weeks), strict=True))
        src = panel.select("symbol", "D", "diff").rename({"D": "D_src", "diff": "diff_p"})
        mp = pl.DataFrame({"D": list(perm.keys()), "D_src": list(perm.values())})
        pp = panel.drop("diff", "y").join(mp, on="D").join(src, on=["symbol", "D_src"], how="inner").rename({"diff_p": "diff"})
        pp = pp.with_columns((pl.col("diff") > 0).cast(pl.Int8).alias("y"))
        pred, _ = walk_forward(pp, keep_models=False)
        pred = pred.filter(pl.col("segment") == seg)
        ts.append(ic_stats(pred)["t"])
    ts = np.array(ts)
    return {"n": n, "t_mean": float(np.nanmean(ts)), "t_sd": float(np.nanstd(ts)), "share_t_gt_2": float(np.nanmean(ts > 2))}


# ------------------------------------------------------------------ H-TR4

def tr4_cuts(panel: pl.DataFrame) -> tuple[float, float]:
    x = panel.filter((pl.col("segment") == "llm_in_sample") & pl.col("er_14d").is_finite())["er_14d"].to_numpy()
    return float(np.quantile(x, 1 / 3)), float(np.quantile(x, 2 / 3))


def tr4_stats(panel: pl.DataFrame, seg: str, cuts: tuple[float, float]) -> dict:
    lo, hi = cuts
    s = panel.filter((pl.col("segment") == seg) & pl.col("er_14d").is_finite())
    s = s.with_columns(pl.when(pl.col("er_14d") >= hi).then(pl.lit("top")).when(pl.col("er_14d") <= lo).then(pl.lit("bottom"))
                       .otherwise(pl.lit("middle")).alias("terc"))
    out: dict = {"cut_lo": lo, "cut_hi": hi, "n_obs": s.height, "n_weeks": s["D"].n_unique()}
    out["by_tercile"] = {t: {"n": (g := s.filter(pl.col("terc") == t)).height, "trend7": float(g["trend7"].mean()) if g.height else float("nan"),
                             "mr7": float(g["mr7"].mean()) if g.height else float("nan"), "diff7": float(g["diff"].mean()) if g.height else float("nan")}
                         for t in ("bottom", "middle", "top")}
    wk = s.filter(pl.col("terc") != "middle").group_by("D", "terc").agg(pl.col("trend7").mean()).pivot(on="terc", index="D", values="trend7")
    if "top" in wk.columns and "bottom" in wk.columns:
        wk = wk.drop_nulls(["top", "bottom"]).sort("D")
        w = _nw((wk["top"] - wk["bottom"]).to_numpy())
    else:
        w = {"mean": float("nan"), "t": float("nan"), "n": 0}
    out.update({"diff": w["mean"], "t": w["t"], "weeks_both": w["n"]})
    tb = s.filter(pl.col("terc") != "middle")
    if tb.height > 10 and tb["terc"].n_unique() == 2:
        b, t, g = ols_cluster(tb["trend7"].to_numpy(), np.c_[np.ones(tb.height), (tb["terc"] == "top").to_numpy().astype(float)], tb["D"].cast(pl.Int32).to_numpy())
        out.update({"pooled_diff": float(b[1]), "pooled_t_cluster": float(t[1])})
    # diagnostic: within-week cross-sectional terciles
    xs = s.with_columns((pl.col("er_14d").rank("average").over("D") / pl.len().over("D")).alias("q"), pl.len().over("D").alias("nw"))
    xs = xs.filter(pl.col("nw") >= 6)
    xw = xs.group_by("D").agg(pl.col("trend7").filter(pl.col("q") > 2 / 3).mean().alias("top"), pl.col("trend7").filter(pl.col("q") <= 1 / 3).mean().alias("bottom"))
    xw = xw.drop_nulls().sort("D")
    xr = _nw((xw["top"] - xw["bottom"]).to_numpy())
    out.update({"xs_diff": xr["mean"], "xs_t": xr["t"], "xs_weeks": xr["n"]})
    return out


# ------------------------------------------------------------------ coefficients and the live table

def coef_summary(models: pl.DataFrame, panel: pl.DataFrame, years: list[int]) -> dict:
    out: dict = {"n_refits": models.height}
    if not models.height:
        return out
    path = {}
    for c in FEATURES:
        b = models[f"b_{c}"].to_numpy()
        med = np.median(b)
        path[c] = {"first": float(b[0]), "last": float(b[-1]), "median": float(med), "min": float(b.min()), "max": float(b.max()),
                   "share_same_sign_as_median": float((np.sign(b) == np.sign(med)).mean()), "last_z": float(b[-1] / models[f"se_{c}"][-1])}
    out["path"] = path
    snap = models.filter(pl.col("fit_month").dt.month().is_in([1, 7]))
    out["snapshots"] = snap.select("fit_month", "n_train", *[f"b_{c}" for c in FEATURES]).to_dicts()
    # per calendar-year fits (disjoint samples): the honest stability check for an expanding window
    ok = panel.filter(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURES]))
    yr = {}
    for y in years:
        s = ok.filter(pl.col("D").dt.year() == y)
        if s.height < 200:
            continue
        X = s.select(FEATURES).to_numpy()
        mu, sd = X.mean(0), X.std(0)
        b, se = fit_logit((X - mu) / np.where(sd > 0, sd, 1), s["y"].to_numpy().astype(float))
        yr[str(y)] = {"n": s.height, **{c: float(v) for c, v in zip(FEATURES, b[1:], strict=True)}}
    out["by_year"] = yr
    if len(yr) >= 2:
        B = np.array([[v[c] for c in FEATURES] for v in yr.values()])
        out["by_year_sign_agreement"] = {c: float(max((B[:, i] > 0).mean(), (B[:, i] < 0).mean())) for i, c in enumerate(FEATURES)}
    return out


def predict_row(model: dict, feats: dict) -> float:
    z = sum(model[f"b_{c}"] * (feats[c] - model[f"mu_{c}"]) / model[f"sd_{c}"] for c in FEATURES)
    return float(_sigmoid(np.array(model["intercept"] + z)))


def current_table(feats: pl.DataFrame, models: pl.DataFrame) -> pl.DataFrame:
    last = feats.filter(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURES])).sort("date").group_by("symbol").tail(1)
    rows = []
    for r in last.sort("symbol").to_dicts():
        mrow = models.filter(pl.col("fit_month") <= r["date"]).tail(1).to_dicts()
        if not mrow:
            continue
        rows.append({"symbol": r["symbol"], "date": r["date"], "fit_month": mrow[0]["fit_month"], "p_trend_wins": predict_row(mrow[0], r),
                     **{c: r[c] for c in ("er_14d", *FEATURES)}})
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ run

def verdict_tr4(r: dict, seg: str) -> str:
    if seg == "clean_holdout":
        return "inconclusive" if r["weeks_both"] < MIN_HOLDOUT_WEEKS or not np.isfinite(r["diff"]) else ("supports" if r["diff"] > 0 else "contradicts")
    return "supports" if r["diff"] > 0 and r["t"] > 2 else "contradicts"


def verdict_tr3(ic: dict, g: dict, seg: str) -> str:
    better = g["gated"] > g["ungated"]
    if seg == "clean_holdout":
        if ic["n_weeks"] < MIN_HOLDOUT_WEEKS or not np.isfinite(ic["ic"]):
            return "inconclusive"
        return "supports" if ic["ic"] > 0 and better else "contradicts"
    return "supports" if ic["ic"] > 0 and ic["t"] > 2 and better else "contradicts"


def run(seg: str, as_of: datetime | None = None, cache: str = "", zoo: str = "", placebo: bool = True, save: bool = False) -> dict:
    as_of = as_of or now()
    cut = knowledge_cutoff().date()
    feats, mr = build_daily(as_of, cache=cache)
    daily = daily_join(mr, load_trend(zoo or None))
    panel = weekly_panel(feats, daily, cut)
    out: dict = {"segment": seg, "cutoff": str(cut), "as_of": as_of.isoformat(),
                 "panel": {"symbols": panel["symbol"].n_unique(), "weeks": panel["D"].n_unique(), "first_D": str(panel["D"].min()), "last_D": str(panel["D"].max()),
                           "n_seg": panel.filter(pl.col("segment") == seg).height}}
    # H-TR4
    cuts = tr4_cuts(panel)
    out["H-TR4"] = tr4_stats(panel, seg, cuts)
    # H-TR3
    pred, models = walk_forward(panel, through=date(as_of.year, as_of.month, 1))
    ps = pred.filter(pl.col("segment") == seg)
    out["H-TR3"] = {"ic": ic_stats(ps), "gate": gate_stats(ps, daily), "oos_first_D": str(pred["D"].min())}
    if placebo:
        out["H-TR3"]["placebo"] = placebo_ic(panel, seg, cut)
    seg_models = models.filter(pl.col("fit_month") <= cut) if seg == "llm_in_sample" else models
    years = sorted({d.year for d in panel.filter(pl.col("segment") == "llm_in_sample")["D"].to_list()}) if seg == "llm_in_sample" else \
        sorted({d.year for d in panel["D"].to_list()})
    yp = panel.filter(pl.col("segment") == "llm_in_sample") if seg == "llm_in_sample" else panel.filter(pl.col("segment").is_not_null())
    out["coefficients"] = coef_summary(seg_models, yp, years)
    out["verdicts"] = {"H-TR4": verdict_tr4(out["H-TR4"], seg), "H-TR3": verdict_tr3(out["H-TR3"]["ic"], out["H-TR3"]["gate"], seg)}
    if seg == "clean_holdout":
        cur = current_table(feats, models)
        out["current"] = cur.to_dicts()
        if save:
            models.write_parquet(_root() / MODEL_FILE)
            pred.write_parquet(_root() / PANEL_FILE)
    return out


NOTES = {
    "H-TR4": "Monday 00:00 UTC decisions, er_14d from 4h bars (features.regime_features) at Sunday close; next-7d trend-zoo ensemble net P&L "
             "(dates D+1..D+7); tercile cut points = in-sample quantiles (reused in holdout); weekly top-minus-bottom mean, NW t over weeks",
    "H-TR3": "11 registry features; monthly expanding-window ridge(1) logistic, standardised on the training window, train on D+9<=refit month, >=52 weeks; "
             "target 1{trend7 - MR7 > 0}; pooled OOS Spearman IC with week-clustered t; gated = trend ensemble if p>0.5 else flat; "
             "cross-symbol mean annualised daily Sharpe on the OOS days",
}


def record_all(out: dict, prefix: str = "") -> None:
    from ai_trader.research.hypotheses import record

    seg = out["segment"]
    data = "bybit 4h, 23-symbol universe, market_frame(funding=False, oi=False); MR = mean(zscore_reversion, rsi_reversion) signal; trend = data/priors/trend_zoo_daily_4h.parquet ensemble"
    record("H-TR4", seg, out["verdicts"]["H-TR4"], out["H-TR4"], data=data, note=prefix + NOTES["H-TR4"])
    m = {"ic": out["H-TR3"]["ic"], "gate": {k: v for k, v in out["H-TR3"]["gate"].items() if k != "per_symbol"}, "placebo": out["H-TR3"].get("placebo")}
    record("H-TR3", seg, out["verdicts"]["H-TR3"], m, data=data, note=prefix + NOTES["H-TR3"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", choices=SEGS, required=True)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--save", action="store_true", help="write the refit models / OOS panel to data/priors (holdout run)")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="", help="path prefix for a daily feature / MR P&L cache")
    ap.add_argument("--zoo", default="")
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--note-prefix", default="")
    a = ap.parse_args()
    out = run(a.segment, cache=a.cache, zoo=a.zoo, placebo=not a.no_placebo, save=a.save)
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    if a.record:
        record_all(out, a.note_prefix)


if __name__ == "__main__":
    main()
