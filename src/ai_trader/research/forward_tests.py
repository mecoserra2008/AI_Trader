"""Self-evaluating forward tests: one evaluator per forward-only hypothesis (agent F, 2026-11-12).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/forward_tests.py [--record] [--only H-LEV6,H-OF6] [--as-of ISO]

Every hypothesis below was registered after its effect had been seen in both segments (or on the only data that
existed), so only data after its registration can confirm it. Each evaluator therefore

  (a) computes its statistic on rows whose decision time / event time is at or after FORWARD[hid].start (the first
      admissible time after registration; `forward_only` is the single gate). Anything fitted (HAR coefficients,
      per-symbol scales, flow thresholds, the centring of the cell effects) is FROZEN on data before that time and
      persisted once in data/research/forward_tests/frozen/<hid>.json, so later runs reuse exactly the same numbers.
      Features may look back (a 7-day RV mean at a forward origin is known at that origin); targets never precede it.
  (b) reports n (forward dates / events / sessions / symbol-days / weeks / refits, the unit of the registered sample
      condition), the required n, and the interim statistic with its sign;
  (c) returns verdict 'pending' while n < required and `run(record=True)` REFUSES to record then. Once n is reached the
      verdict follows the registered success criterion literally: supports = criterion met; contradicts = the point
      estimate is on the wrong side of the criterion (it cannot be met); inconclusive = right side but not significant.
      A result is recorded once (segment clean_holdout: every forward date is after research.knowledge_cutoff); an
      existing clean_holdout row in research/hypotheses_results.jsonl blocks a second record.

Data conventions (shared with the original studies):
  * daily panel = leverage_study.build (1h bars -> complete UTC days, calendar reindexed; origin = end of day t =
    decision time; target = RV of day t + 1); log-HAR regressors as ml.features / vol_panel 'loghar' (log RV_d, mean log
    RV over 7 (>= 5) and 30 (>= 20) calendar days);
  * forecast day = target day = origin date; "forecast dates after D" = origins >= D + 1 day 00:00 UTC;
  * order flow = features.flow_anomaly on flow_3s (per-symbol minute panels), liquidation clusters as H-OF3;
  * TradFi sessions = tradfi_session_study (NYSE sessions, 1h bars, 7.5 bp per side).

Out of scope: H-MM3 and H-NS7 need live paired market-making runs (blocked: research only). H-PD6 is a historical
in-sample / holdout test, not a forward one (listed, not evaluated here).
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

VERSION = "0.1.0"
WORKERS = 1          # single process by default (shared machine: keep RAM < ~1 GB); <= 4 if raised
MIN_FREE_GB = 6.0


@dataclass(frozen=True)
class Spec:
    hid: str
    registered: str          # registration date (registry block / statement)
    start: datetime          # first admissible decision / event time for the statistic
    need: int                # registered sample-size condition
    unit: str                # what n counts
    criterion: str           # registered success criterion (verbatim gist)


def _d(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=UTC)


FORWARD: dict[str, Spec] = {
    "H-LEV6": Spec("H-LEV6", "2026-11-10", _d(2026, 11, 11), 120, "forward decision dates",
                   "funding_z coefficient > 0 with t > 2 (H-LEV5 model, date-clustered SE)"),
    "H-VM6": Spec("H-VM6", "2026-11-10", _d(2026, 11, 11), 120, "forward forecast dates (min of L, T)",
                  "QLIKE(log-HAR + Sat/Sun/Mon) - QLIKE(log-HAR) < 0 with t < -2, universes L and T"),
    "H-OF6": Spec("H-OF6", "2026-11-10", _d(2026, 11, 11), 14, "forward UTC days of flow",
                  "pooled next-5m |r| ratio (top-1% |imbalance z| minutes vs same symbol-hour) >= 1.5, CI excluding 1"),
    "H-OF7": Spec("H-OF7", "2026-11-10", _d(2026, 11, 12), 40, "independent cluster-hours",
                  "signed 60m continuation after declustered liquidation clusters > 0 with t > 2"),
    "H-TP1b": Spec("H-TP1b", "2026-11-11", _d(2026, 11, 12), 150, "forward equity symbol-sessions with |on_z| > 1",
                   "net first-two-cash-hours return in the overnight direction > 0 with t > 2 (7.5 bp/side)"),
    "H-MS5b": Spec("H-MS5b", "2026-11-11", _d(2026, 11, 11), 10, "free-spread symbol-days after 2026-11-10",
                   "out-of-sample R^2 gain of the spread > 0 on >= 70% of symbol-days (5m and 15m)"),
    "H-NN7": Spec("H-NN7", "2026-11-12", _d(2026, 11, 13), 120, "forward forecast dates",
                  "weekend x high-vol interaction > 0 with t > 2 (pooled log-HAR, date-clustered SE)"),
    "H-LEV7": Spec("H-LEV7", "2026-11-12", _d(2026, 11, 13), 120, "forward forecast dates",
                   "b(market down) - b(market up) > 0 with t > 2"),
    "H-NN10": Spec("H-NN10", "2026-11-12", _d(2026, 11, 13), 60, "forward days of logged hourly forecasts",
                   "QLIKE(frozen XGB) - QLIKE(robust_bc x hour-of-week) < 0 with t < -2"),
    "H-DS4": Spec("H-DS4", "2026-11-12", _d(2026, 11, 13), 120, "forward forecast dates",
                  "Sat x (range|down) < 0, Monday > 0, Tue-Fri x up > 0, each |t| > 2 (log RV / frozen log-HAR)"),
    "H-SP4": Spec("H-SP4", "2026-11-12", _d(2026, 12, 1), 3, "monthly refits with training end after 2026-11-12",
                  "vol-level and calendar clusters > 0.8 and no other cluster > 0.5 in all 3 refits"),
    "H-PD5": Spec("H-PD5", "2026-11-11", _d(2026, 11, 16), 26, "forward week pairs (trailing, next)",
                  "mean weekly cross-strategy Spearman IC < 0 with t < -2"),
}
BLOCKED = {
    "H-MM3": "needs paired live market-making runs (exchange records); research only: blocked",
    "H-NS7": "needs paired live runs or post-registration 3 s replays of the live quoter; research only: blocked",
    "H-PD6": "not a forward test: registered as llm_in_sample with clean_holdout confirmation (historical re-run of the trend sleeves)",
}


# ====================================================================== shared helpers

def free_gb() -> float | None:
    """Free physical RAM in GB (Windows / psutil if available), None if unknown."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except ImportError:
        pass
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ext", ctypes.c_ulonglong)]

        m = MS()
        m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 2**30
    except Exception:  # noqa: BLE001
        return None


def wait_for_ram(min_gb: float = MIN_FREE_GB, max_wait_s: int = 1800) -> None:
    """Pause while free RAM is below `min_gb` (other jobs share the machine)."""
    t0 = time.time()
    while (g := free_gb()) is not None and g < min_gb and time.time() - t0 < max_wait_s:
        print(f"  [ram] {g:.1f} GB free < {min_gb} GB: waiting", flush=True)
        time.sleep(30)


def pmap(fn: Callable, *its, workers: int = WORKERS) -> list:
    """map() in-process when workers == 1 (default), else a process pool of <= 4 workers."""
    if workers <= 1:
        out = []
        for a in zip(*its, strict=True):
            wait_for_ram()
            out.append(fn(*a))
        return out
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        return list(ex.map(fn, *its))


def out_dir() -> Path:
    from ai_trader.core.config import load

    d = load().root / "data" / "research" / "forward_tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def forward_only(df: pl.DataFrame, col: str, start: datetime) -> pl.DataFrame:
    """THE gate: rows whose decision / event time `col` is at or after `start`."""
    if df.height == 0 or col not in df.columns:
        return df
    c = pl.col(col)
    if df.schema[col] == pl.Date:
        return df.filter(c >= start.date())
    return df.filter(c >= start)


def frozen(hid: str, fit: Callable[[], dict], path: Path | None = None) -> dict:
    """Parameters frozen as of registration: computed once by `fit` (which must use pre-start data only), persisted,
    and reused verbatim afterwards."""
    p = path or (out_dir() / "frozen" / f"{hid}.json")
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    par = fit()
    par["_frozen_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(par, default=float), encoding="utf-8")
    return par


def decide(n: int, need: int, met: bool, wrong_side: bool) -> str:
    if n < need:
        return "pending"
    if met:
        return "supports"
    if wrong_side:
        return "contradicts"
    return "inconclusive"


def sign_of(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return "+" if x > 0 else ("-" if x < 0 else "0")


def _f(x: Any, nd: int = 4) -> Any:
    if isinstance(x, dict):
        return {k: _f(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_f(v, nd) for v in x]
    if isinstance(x, float | np.floating):
        return round(float(x), nd) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def cluster_mean(x: np.ndarray, cl: np.ndarray) -> dict:
    """Mean with a CR1 cluster-robust SE (clusters = dates or hours)."""
    from ai_trader.research.flow_anomaly_study import cluster_t

    if len(x) == 0:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": 0, "clusters": 0}
    _, inv = np.unique(cl, return_inverse=True)
    return cluster_t(np.asarray(x, dtype=float), inv)


# ====================================================================== daily panel (H-LEV6, H-VM6, H-NN7, H-LEV7, H-DS4)

def _trail_mean_expr(c: str, w: int, min_n: int) -> pl.Expr:
    return pl.col(c).rolling_mean(window_size=w, min_samples=min_n).over("symbol")


def add_loghar(daily: pl.DataFrame) -> pl.DataFrame:
    """log-HAR regressors on the calendar-reindexed daily panel (per symbol, sorted by date)."""
    d = daily.sort("symbol", "date").with_columns(
        pl.when(pl.col("rv") > 0).then(pl.col("rv").log()).alias("lrv_d"),
        pl.when(pl.col("rv_next") > 0).then(pl.col("rv_next").log()).alias("lrv_next"))
    return d.with_columns(_trail_mean_expr("lrv_d", 7, 5).alias("lrv_w"), _trail_mean_expr("lrv_d", 30, 20).alias("lrv_m"))


def load_daily(as_of: datetime) -> pl.DataFrame:
    from ai_trader.research import leverage_study as ls

    B = ls.build(as_of=as_of, cache=None)
    return add_loghar(B["daily"])


# ---------------------------------------------------------------- H-LEV6

LEV6_X = ["rv_s", "rv_w_s", "rv_m_s", "rs_neg_s", "fz", "neg_fz"]


def lev6_freeze(daily: pl.DataFrame, start: datetime) -> dict:
    """Per-symbol mean RV on days before the forward start (the H-LEV5 unit-free scaling, frozen)."""
    pre = daily.filter((pl.col("universe") == "L") & (pl.col("origin") < start) & pl.col("rv").is_not_null())
    sc = pre.group_by("symbol").agg(pl.col("rv").mean().alias("c"))
    return {"scale": dict(sc.iter_rows())}


def lev6_stat(daily: pl.DataFrame, start: datetime, par: dict) -> dict:
    from ai_trader.research import leverage_study as ls

    sc = pl.DataFrame({"symbol": list(par["scale"]), "c": list(par["scale"].values())})
    d = forward_only(daily.filter(pl.col("universe") == "L"), "origin", start).join(sc, on="symbol", how="inner")
    d = d.with_columns(*[(pl.col(x) / pl.col("c")).alias(f"{x}_s") for x in ("rv_next", "rv", "rv_w", "rv_m", "rs_neg")])
    d = d.with_columns((pl.col("rs_neg_s") * pl.col("fz")).alias("neg_fz"))
    need = ["rv_next_s", *LEV6_X]
    d = d.drop_nulls(need).filter(pl.all_horizontal([pl.col(c).is_finite() for c in need]))
    n = d["origin"].n_unique() if d.height else 0
    res = ls.fe_ols(d, "rv_next_s", LEV6_X, cluster="origin") if d.height else {"ok": False}
    lc = ls.lincomb(res, {"fz": 1.0})
    return {"n": n, "n_obs": d.height, "est": lc["est"], "t": lc["t"], "se": lc["se"],
            "met": bool(lc["est"] > 0 and lc["t"] > 2), "wrong_side": bool(np.isfinite(lc["est"]) and lc["est"] <= 0)}


# ---------------------------------------------------------------- pooled FE log-HAR (H-VM6) and per-asset log-HAR (H-DS4)

HAR_X = ["lrv_d", "lrv_w", "lrv_m"]


def _wd_dummies(d: pl.DataFrame) -> pl.DataFrame:
    w = pl.col("tgt_wd")
    return d.with_columns((w == 6).cast(pl.Float64).alias("sat"), (w == 7).cast(pl.Float64).alias("sun"),
                          (w == 1).cast(pl.Float64).alias("mon"))


def _usable(d: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return d.drop_nulls(cols).filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols]))


def fe_fit(d: pl.DataFrame, y: str, xs: list[str]) -> dict:
    """Pooled OLS with symbol fixed effects; returns alpha per symbol, beta and the residual variance."""
    from ai_trader.research.leverage_study import demean

    d = _usable(d, [y, *xs])
    if d.height < len(xs) + 10:
        return {"xs": xs, "beta": [float("nan")] * len(xs), "alpha": {}, "s2": float("nan"), "n": int(d.height), "n_obs_sym": {}}
    syms, g = np.unique(d["symbol"].to_numpy(), return_inverse=True)
    X, yy = d.select(xs).to_numpy().astype(float), d[y].to_numpy().astype(float)
    beta, *_ = np.linalg.lstsq(demean(X, g), demean(yy, g), rcond=None)
    cnt = np.bincount(g)
    alpha = np.bincount(g, weights=yy) / cnt - np.column_stack([np.bincount(g, weights=X[:, j]) / cnt for j in range(X.shape[1])]) @ beta
    resid = yy - (alpha[g] + X @ beta)
    s2 = float(resid @ resid / max(len(yy) - len(beta) - len(syms), 1))
    return {"xs": xs, "beta": beta.tolist(), "alpha": dict(zip(syms.tolist(), alpha.tolist(), strict=True)), "s2": s2,
            "n": int(len(yy)), "n_obs_sym": dict(zip(syms.tolist(), cnt.tolist(), strict=True))}


def fe_predict(d: pl.DataFrame, fit: dict, min_sym_obs: int = 30) -> np.ndarray:
    """exp(alpha_i + x beta + s2 / 2) for rows whose symbol was in the frozen fit (>= min_sym_obs rows), else NaN."""
    X = d.select(fit["xs"]).to_numpy().astype(float)
    a = np.array([fit["alpha"].get(s, np.nan) if fit["n_obs_sym"].get(s, 0) >= min_sym_obs else np.nan for s in d["symbol"].to_list()])
    with np.errstate(all="ignore"):
        return np.exp(a + X @ np.asarray(fit["beta"]) + 0.5 * fit["s2"])


def vm6_freeze(daily: pl.DataFrame, start: datetime) -> dict:
    """Both models fitted per universe on rows whose target day is complete before the forward start."""
    out = {}
    pre = _wd_dummies(daily.filter(pl.col("origin") + pl.duration(days=1) <= start))
    for u in ("L", "T"):
        p = pre.filter(pl.col("universe") == u)
        out[u] = {"base": fe_fit(p, "lrv_next", HAR_X), "wd": fe_fit(p, "lrv_next", [*HAR_X, "sat", "sun", "mon"])}
    return out


def vm6_stat(daily: pl.DataFrame, start: datetime, par: dict) -> dict:
    from ai_trader.features.vol_models_x import qlike
    from ai_trader.research.leverage_study import date_avg_test

    res, ns, met, wrong = {}, [], True, False
    for u in ("L", "T"):
        d = _usable(_wd_dummies(forward_only(daily.filter(pl.col("universe") == u), "origin", start)),
                    [*HAR_X, "sat", "sun", "mon", "rv_next"]).filter(pl.col("rv_next") > 0)
        if d.height == 0:
            res[u] = {"mean": float("nan"), "t": float("nan"), "n_dates": 0, "n_obs": 0}
            ns.append(0)
            met = False
            continue
        fb, fw = fe_predict(d, par[u]["base"]), fe_predict(d, par[u]["wd"])
        rv = d["rv_next"].to_numpy()
        q = d.select("origin").with_columns(pl.Series("qa", qlike(rv, fw)), pl.Series("qb", qlike(rv, fb)))
        q = q.with_columns((pl.col("qa") - pl.col("qb")).alias("dq")).filter(pl.col("dq").is_finite())
        r = date_avg_test(q) if q.height else {"mean": float("nan"), "t": float("nan"), "n_dates": 0, "n_obs": 0}
        res[u] = {k: r[k] for k in ("mean", "t", "n_dates", "n_obs")}
        ns.append(r["n_dates"])
        met &= bool(r["mean"] < 0 and r["t"] < -2)
        wrong |= bool(np.isfinite(r["mean"]) and r["mean"] >= 0)
    est = res["L"]["mean"]
    return {"n": min(ns), "est": est, "t": res["L"]["t"], "by_universe": res, "met": met, "wrong_side": wrong}


# ---------------------------------------------------------------- H-NN7

def add_high_vol(d: pl.DataFrame, days: int = 365, min_n: int = 180) -> pl.DataFrame:
    """high = 1{lrv_w > its own trailing-1-year median (including today)}, point in time per symbol."""
    d = d.sort("symbol", "date").with_columns(pl.col("lrv_w").rolling_median(window_size=days, min_samples=min_n).over("symbol").alias("_med"))
    return d.with_columns(pl.when(pl.col("_med").is_null() | pl.col("lrv_w").is_null()).then(None)
                          .otherwise((pl.col("lrv_w") > pl.col("_med")).cast(pl.Float64)).alias("high")).drop("_med")


NN7_X = [*HAR_X, "sat", "sun", "mon", "high", "sat_high", "sun_high"]


def nn7_stat(daily: pl.DataFrame, start: datetime) -> dict:
    from ai_trader.research import leverage_study as ls

    d = add_high_vol(daily.filter(pl.col("universe") == "L"))
    d = _wd_dummies(forward_only(d, "origin", start)).with_columns(
        (pl.col("sat") * pl.col("high")).alias("sat_high"), (pl.col("sun") * pl.col("high")).alias("sun_high"))
    d = _usable(d, ["lrv_next", *NN7_X])
    n = d["origin"].n_unique() if d.height else 0
    res = ls.fe_ols(d, "lrv_next", NN7_X, cluster="origin") if d.height else {"ok": False}
    lc = ls.lincomb(res, {"sat_high": 0.5, "sun_high": 0.5})
    return {"n": n, "n_obs": d.height, "est": lc["est"], "t": lc["t"], "se": lc["se"],
            "met": bool(lc["est"] > 0 and lc["t"] > 2), "wrong_side": bool(np.isfinite(lc["est"]) and lc["est"] <= 0)}


# ---------------------------------------------------------------- H-LEV7

LEV7_X = ["l_neg", "l_pos", "lrv_w", "lrv_m", "mkt_dn", "mkt_up"]


def add_market(d: pl.DataFrame, min_syms: int = 5) -> pl.DataFrame:
    mk = (d.filter(pl.col("ret").is_not_null()).group_by("date")
          .agg(pl.col("ret").mean().alias("mkt"), pl.len().alias("_n")).filter(pl.col("_n") >= min_syms).drop("_n"))
    return d.join(mk, on="date", how="left").with_columns(
        pl.max_horizontal(-pl.col("mkt"), pl.lit(0.0)).alias("mkt_dn"), pl.max_horizontal(pl.col("mkt"), pl.lit(0.0)).alias("mkt_up"))


def lev7_stat(daily: pl.DataFrame, start: datetime) -> dict:
    from ai_trader.research import leverage_study as ls

    d = add_market(daily.filter(pl.col("universe") == "L"))
    d = forward_only(d, "origin", start).with_columns(pl.col("rs_neg_f").log().alias("l_neg"), pl.col("rs_pos_f").log().alias("l_pos"))
    d = _usable(d, ["lrv_next", *LEV7_X])
    n = d["origin"].n_unique() if d.height else 0
    res = ls.fe_ols(d, "lrv_next", LEV7_X, cluster="origin") if d.height else {"ok": False}
    lc = ls.lincomb(res, {"mkt_dn": 1.0, "mkt_up": -1.0})
    return {"n": n, "n_obs": d.height, "est": lc["est"], "t": lc["t"], "se": lc["se"],
            "met": bool(lc["est"] > 0 and lc["t"] > 2), "wrong_side": bool(np.isfinite(lc["est"]) and lc["est"] <= 0)}


# ---------------------------------------------------------------- H-DS4

DS4_GROUPS = {
    "sat_range_or_down": {"wd": (6,), "dir": ("range", "down"), "sign": -1},
    "monday": {"wd": (1,), "dir": None, "sign": +1},
    "tue_fri_up": {"wd": (2, 3, 4, 5), "dir": ("up",), "sign": +1},
}
DS4_MIN_ROWS = 180


def v0_direction(symbols: list[str], as_of: datetime) -> pl.DataFrame:
    """v0 direction state at the 20:00 UTC 4h bar of each date (its close = the decision time), as ml.features."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label
    from ai_trader.storage.lake import Lake

    lake = Lake()
    parts = []
    for s in symbols:
        f4 = market_frame(s, "4h", as_of, funding=False, oi=False, lake=lake)
        if f4.height == 0:
            continue
        lab = label(f4.sort("event_ts"), bar_hours=4.0).filter(pl.col("event_ts").dt.hour() == 20)
        parts.append(lab.select(pl.lit(s).alias("symbol"), pl.col("event_ts").dt.date().alias("date"), pl.col("dir_state").alias("v0_dir")))
    return pl.concat(parts) if parts else pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date, "v0_dir": pl.Utf8})


def ds4_freeze(daily: pl.DataFrame, start: datetime, train_end: datetime) -> dict:
    """Per-asset log-HAR (OLS of lrv_next on 1, lrv_d, lrv_w, lrv_m) on rows whose target day is complete by
    `train_end` (registration), plus the training mean of u = log(RV / F) used to centre the cell effects."""
    assert train_end <= start
    pre = _usable(daily.filter((pl.col("universe") == "L") & (pl.col("origin") + pl.duration(days=1) <= train_end)), ["lrv_next", *HAR_X])
    coefs, us = {}, []
    for (s,), g in pre.sort("symbol", "date").group_by("symbol", maintain_order=True):
        if g.height < DS4_MIN_ROWS:
            continue
        X = np.column_stack([np.ones(g.height), g.select(HAR_X).to_numpy()])
        y = g["lrv_next"].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        s2 = float(e @ e / max(len(y) - 4, 1))
        coefs[s] = {"b": b.tolist(), "s2": s2}
        us.append(e - 0.5 * s2)  # u = log RV - log F = e - s2 / 2
    ubar = float(np.mean(np.concatenate(us))) if us else 0.0
    return {"coefs": coefs, "ubar": ubar, "train_end": train_end.isoformat()}


def ds4_rows(daily: pl.DataFrame, v0: pl.DataFrame, start: datetime, par: dict) -> pl.DataFrame:
    d = forward_only(daily.filter(pl.col("universe") == "L"), "origin", start).join(v0, on=["symbol", "date"], how="left")
    d = _usable(d, ["lrv_next", *HAR_X]).filter(pl.col("symbol").is_in(list(par["coefs"])))
    if d.height == 0:
        return d.with_columns(pl.lit(None, pl.Float64).alias("u"))
    B = np.array([par["coefs"][s]["b"] for s in d["symbol"].to_list()])
    s2 = np.array([par["coefs"][s]["s2"] for s in d["symbol"].to_list()])
    X = np.column_stack([np.ones(d.height), d.select(HAR_X).to_numpy()])
    lf = (X * B).sum(1) + 0.5 * s2
    return d.with_columns(pl.Series("u", d["lrv_next"].to_numpy() - lf - par["ubar"]))


def ds4_stat(rows: pl.DataFrame) -> dict:
    groups, met, wrong = {}, True, False
    for k, g in DS4_GROUPS.items():
        m = rows.filter(pl.col("tgt_wd").is_in(list(g["wd"])))
        if g["dir"] is not None:
            m = m.filter(pl.col("v0_dir").is_in(list(g["dir"])))
        m = m.filter(pl.col("u").is_finite())
        r = cluster_mean(m["u"].to_numpy(), m["origin"].cast(pl.Int64).to_numpy()) if m.height else cluster_mean(np.array([]), np.array([]))
        r["n_dates"] = m["origin"].n_unique() if m.height else 0
        r["pct"] = float(math.exp(r["mean"]) - 1) * 100 if np.isfinite(r["mean"]) else float("nan")
        groups[k] = r
        met &= bool(np.isfinite(r["t"]) and g["sign"] * r["mean"] > 0 and g["sign"] * r["t"] > 2)
        wrong |= bool(np.isfinite(r["mean"]) and g["sign"] * r["mean"] <= 0)
    n = rows.filter(pl.col("u").is_finite())["origin"].n_unique() if rows.height else 0
    return {"n": n, "groups": groups, "est": groups["monday"]["mean"], "t": groups["monday"]["t"], "met": met, "wrong_side": wrong}


# ====================================================================== order flow (H-OF6, H-OF7)

OF6_FREEZE_AT = _d(2026, 11, 10, 22, 30)
OF6_Q = 0.99
FLOW_KEEP = ["symbol", "event_ts", "z_imbalance", "fwd_5", "fwd_60", "sigma_1m", "liq_cluster", "liq_side"]


def _flow_symbol(sym: str, as_of_iso: str, start_iso: str | None, score_from_iso: str | None, liq_start_iso: str | None, out: str) -> str:
    """One symbol's minute panel as of `as_of` (point in time), scored from `score_from`; forward 5m / 60m mid returns."""
    from ai_trader.features import flow_anomaly as fa
    from ai_trader.storage.lake import Lake

    as_of = datetime.fromisoformat(as_of_iso)
    lake = Lake()
    lake.con.execute("SET memory_limit = '512MB'")
    lake.con.execute("SET threads = 2")
    start = datetime.fromisoformat(start_iso) if start_iso else None
    df3 = lake.read_asof("flow_3s", as_of, symbols=[sym], start=start, filters={"category": "linear", "venue": "bybit"}, columns=fa.FLOW_COLUMNS)
    liq = lake.read_asof("liquidations", as_of, symbols=[sym], start=start, filters={"venue": "bybit"})
    if df3.height == 0:
        pl.DataFrame().write_parquet(out)
        return out
    # = fa.build_panel restricted to what H-OF6 / H-OF7 use (z_imbalance, sigma_1m, liquidation clusters): the same
    # functions in the same order, minus the other five robust z columns and z_liq (about 6x less compute)
    sf = datetime.fromisoformat(score_from_iso) if score_from_iso else None
    f = fa.add_robust_z(fa.add_features(fa.rollup_1m(df3, fa.MAX_LAG_S)), ("imbalance",), fa.WINDOW, fa.MIN_VALID, fa.REL_FLOOR, sf)
    del df3
    lm = fa.liquidation_minutes(liq if liq is not None else pl.DataFrame())
    if lm.height:
        f = f.join(lm.with_columns(pl.col("event_ts").cast(f["event_ts"].dtype)), on=["symbol", "event_ts"], how="left")
    else:
        f = f.with_columns(pl.lit(None, pl.Float64).alias("liq_notional"), pl.lit(None, pl.Int64).alias("liq_n"),
                           pl.lit(None, pl.Float64).alias("liq_long"), pl.lit(None, pl.Float64).alias("liq_short"))
    p = fa.liquidation_clusters(f, window=fa.WINDOW, min_valid=fa.MIN_VALID,
                                liq_start=datetime.fromisoformat(liq_start_iso) if liq_start_iso else None, score_from=sf)
    p = p.filter(pl.col("event_ts") + timedelta(minutes=1) <= as_of)
    lm = pl.col("mid").log()
    p = p.with_columns((lm.shift(-5) - lm).alias("fwd_5"), (lm.shift(-60) - lm).alias("fwd_60"))
    if score_from_iso:
        p = p.filter(pl.col("event_ts") >= datetime.fromisoformat(score_from_iso))
    p.select([c for c in FLOW_KEEP if c in p.columns]).write_parquet(out)
    return out


def flow_panel(as_of: datetime, start: datetime | None, score_from: datetime | None, workers: int = WORKERS) -> pl.DataFrame:
    from ai_trader.features import flow_anomaly as fa
    from ai_trader.storage.lake import Lake

    lake = Lake()
    syms = fa.flow_symbols(lake, as_of, "linear")
    ls_ = fa.liquidation_start(lake, as_of)
    with tempfile.TemporaryDirectory(prefix="fwd_flow_") as td:
        outs = [str(Path(td) / f"{s}.parquet") for s in syms]
        k = len(syms)
        pmap(_flow_symbol, syms, [as_of.isoformat()] * k, [start.isoformat() if start else None] * k,
             [score_from.isoformat() if score_from else None] * k, [ls_.isoformat() if ls_ else None] * k, outs, workers=workers)
        parts = [pl.read_parquet(o) for o in outs if Path(o).is_file()]
        parts = [p for p in parts if p.height]
        return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def of6_freeze_from_panel(pre: pl.DataFrame) -> dict:
    """Per-symbol q99 of |robust z| of 1m signed taker imbalance on minutes before the freeze time."""
    t = (pre.filter(pl.col("z_imbalance").is_not_null() & pl.col("z_imbalance").is_finite())
         .group_by("symbol").agg(pl.col("z_imbalance").abs().quantile(OF6_Q).alias("thr"), pl.len().alias("n")))
    return {"freeze_at": OF6_FREEZE_AT.isoformat(), "q": OF6_Q, "thr": {r[0]: r[1] for r in t.iter_rows()},
            "n_minutes": {r[0]: r[2] for r in t.iter_rows()}}


def of6_stat(fwd: pl.DataFrame, start: datetime, par: dict, seed: int = 20261112) -> dict:
    from ai_trader.research.flow_anomaly_study import _block_ids, ratio_boot

    thr = pl.DataFrame({"symbol": list(par["thr"]), "thr": list(par["thr"].values())})
    d = forward_only(fwd, "event_ts", start).join(thr, on="symbol", how="inner")
    d = d.filter(pl.col("z_imbalance").is_not_null() & pl.col("z_imbalance").is_finite() & pl.col("fwd_5").is_not_null())
    # a forward day counts once at least half of its symbol-minutes are scored (the collector has gaps)
    n_days = 0
    if d.height:
        cov = d.group_by(pl.col("event_ts").dt.date().alias("day")).agg(pl.len().alias("n"), pl.col("symbol").n_unique().alias("ns"))
        n_days = int(cov.filter(pl.col("n") >= 0.5 * 1440 * len(par["thr"])).height)
    if d.height == 0:
        return {"n": 0, "est": float("nan"), "ci95": [float("nan")] * 2, "met": False, "wrong_side": False}
    d = d.with_columns((pl.col("z_imbalance").abs() > pl.col("thr")).alias("anom"), pl.col("fwd_5").abs().alias("y"),
                       pl.col("event_ts").dt.hour().alias("hod"), pl.col("event_ts").dt.truncate("1h").alias("block"))
    base = d.filter(~pl.col("anom")).group_by("symbol", "hod").agg(pl.col("y").mean().alias("base"))
    ev = d.filter(pl.col("anom")).join(base, on=["symbol", "hod"], how="inner").filter(pl.col("base") > 0)
    if ev.height < 2:
        return {"n": n_days, "n_events": ev.height, "est": float("nan"), "ci95": [float("nan")] * 2, "met": False, "wrong_side": False}
    inv, nb = _block_ids(ev["block"])
    r = ratio_boot(ev["y"].to_numpy(), ev["base"].to_numpy(), inv, nb, np.random.default_rng(seed))
    return {"n": n_days, "n_days_any_data": int(d["event_ts"].dt.date().n_unique()), "n_events": ev.height,
            "anomaly_share": float(d["anom"].mean()), "est": r["ratio"], "ci95": r["ci95"],
            "est_minus_1": r["ratio"] - 1, "met": bool(r["ratio"] >= 1.5 and r["ci95"][0] > 1), "wrong_side": bool(r["ci95"][1] < 1.5)}


def of7_stat(fwd: pl.DataFrame, start: datetime) -> dict:
    """Declustered (30 min, per symbol) liquidation clusters from `start`; signed 60m return in sigma units,
    sign = -1 for long liquidations (forced sells: continuation = price down), clustered by calendar hour."""
    from ai_trader.research.flow_anomaly_study import decluster

    raw = forward_only(fwd, "event_ts", start)
    raw = raw.filter(pl.col("liq_cluster").fill_null(False)) if "liq_cluster" in raw.columns and raw.height else raw.head(0)
    if raw.height == 0:
        return {"n": 0, "n_events": 0, "est": float("nan"), "t": float("nan"), "met": False, "wrong_side": False}
    ev = decluster(raw, 30)
    sign = pl.when(pl.col("liq_side") == "long").then(-1.0).otherwise(1.0)
    e = ev.filter(pl.col("fwd_60").is_not_null() & (pl.col("sigma_1m") > 0)).with_columns(
        (sign * pl.col("fwd_60") / (pl.col("sigma_1m") * math.sqrt(60))).alias("s"), pl.col("event_ts").dt.truncate("1h").alias("block"))
    if e.height == 0:
        return {"n": 0, "n_events": 0, "est": float("nan"), "t": float("nan"), "met": False, "wrong_side": False}
    r = cluster_mean(e["s"].to_numpy(), e["block"].cast(pl.Int64).to_numpy())
    return {"n": int(r["clusters"]), "n_events": e.height, "est": r["mean"], "t": r["t"], "se": r["se"],
            "met": bool(r["mean"] > 0 and r["t"] > 2), "wrong_side": bool(np.isfinite(r["mean"]) and r["mean"] <= 0)}


# ====================================================================== H-TP1b

def tp1b_stat(sess: pl.DataFrame, start: datetime) -> dict:
    """Continuation trades: side = sign(overnight move) on sessions with |on_z| > 1, held 09:00 -> 11:00 ET, net of 15 bp
    round trip; trade-level NW t (trades in date order)."""
    from ai_trader.research.evaluate import newey_west_t
    from ai_trader.research.tradfi_session_study import COST_RT

    d = sess.filter(pl.col("open_utc") >= start) if sess.height else sess
    d = d.filter(pl.col("on_z").is_finite() & (pl.col("on_z").abs() > 1) & pl.col("first2").is_finite() & (pl.col("on") != 0)) if d.height else d
    if d.height == 0:
        return {"n": 0, "est": float("nan"), "t": float("nan"), "met": False, "wrong_side": False}
    d = d.sort("date", "symbol").with_columns(((pl.col("on").sign() * pl.col("first2") - COST_RT) * 1e4).alias("net_bps"))
    m, t, _ = newey_west_t(d["net_bps"].to_numpy())
    return {"n": d.height, "n_sessions": d["date"].n_unique(), "est": float(m), "t": float(t), "unit_est": "net bp per trade",
            "met": bool(m > 0 and np.isfinite(t) and t > 2), "wrong_side": bool(np.isfinite(m) and m <= 0)}


def load_tp1b(as_of: datetime) -> pl.DataFrame:
    from ai_trader.research import tradfi_session_study as ts

    df, _ = ts.load_equity(as_of)
    return ts.add_derived(df)


# ====================================================================== H-MS5b

MS5B_DAY0 = date(2026, 11, 10)
MS5B_FREE_SHARE = 0.25


def ms5b_days(root: Path | None = None) -> list[tuple[date, Path]]:
    """Pilot days after 2026-11-10 with a book_100ms directory (data/raw/pilot_YYYY-MM-DD/book_100ms)."""
    from ai_trader.core.config import load

    root = root or (load().root / "data" / "raw")
    out = []
    for p in sorted(root.glob("pilot_*")):
        try:
            d = date.fromisoformat(p.name.removeprefix("pilot_"))
        except ValueError:
            continue
        if d > MS5B_DAY0 and (p / "book_100ms").is_dir():
            out.append((d, p / "book_100ms"))
    return out


@contextmanager
def _book_day(day: date, book_dir: Path):
    """Point book_spillover_study at another pilot day (its functions read module-level day constants)."""
    from ai_trader.research import book_spillover_study as bs

    keep = {k: getattr(bs, k) for k in ("DAY0", "D0", "WINDOWS", "BOOK_DIR", "CPI_EX")}
    d0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    D0 = int(d0.timestamp() * 1000)
    try:
        bs.DAY0, bs.D0, bs.BOOK_DIR = d0, D0, book_dir
        bs.WINDOWS = {"fit": (D0, D0 + bs.HALF), "test": (D0 + bs.HALF, D0 + bs.DAY)}
        bs.CPI_EX = (D0, D0)  # no event exclusion on other days
        yield bs
    finally:
        for k, v in keep.items():
            setattr(bs, k, v)


def ms5b_day(day: date, book_dir: Path) -> list[dict]:
    """Per free-spread symbol (spread > 1 tick >= 25% of 100 ms states) the out-of-sample R^2 gain of the spread
    (FIT 00-12 UTC, TEST 12-24 UTC, H-MS5 regression; per symbol, its own intercept)."""
    with _book_day(day, book_dir) as bs:
        inst = bs.instruments()
        syms = sorted(p.stem for p in book_dir.glob("*.parquet"))
        free = []
        for s in syms:
            tick = (inst.get(s) or {}).get("tick_size")
            if not tick:
                continue
            b = pl.read_parquet(book_dir / f"{s}.parquet", columns=["ts", "bid1", "ask1"])
            ticks = np.round((b["ask1"] - b["bid1"]).to_numpy() / float(tick))
            share = float(np.mean(ticks > 1)) if len(ticks) else 0.0
            if share >= MS5B_FREE_SHARE:
                free.append((s, share))
        if not free:
            return []
        fac, _ = bs.how_factors([s for s, _ in free])
        rows = []
        for s, share in free:
            panel = bs.ms5_rows(s, fac[s])
            g = {}
            for h in bs.HORIZONS:
                try:
                    g[h] = bs.ms5_fit(panel, h)["r2_oos_gain"]
                except Exception:  # noqa: BLE001 - too few rows in a half
                    g[h] = float("nan")
            rows.append({"date": day.isoformat(), "symbol": s, "free_share": share, **{f"gain_{h}": g[h] for h in bs.HORIZONS}})
        return rows


def ms5b_stat(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "est": float("nan"), "met": False, "wrong_side": False, "share_pos": {}}
    df = pl.DataFrame(rows)
    sh = {}
    for h in (5, 15):
        x = df[f"gain_{h}"].drop_nulls().to_numpy()
        x = x[np.isfinite(x)]
        sh[h] = float(np.mean(x > 0)) if len(x) else float("nan")
    n = int(df.filter(pl.col("gain_5").is_finite() & pl.col("gain_15").is_finite()).height)
    est = float(np.nanmedian(df["gain_5"].to_numpy()))
    return {"n": n, "est": est, "share_pos": sh, "met": all(v >= 0.7 for v in sh.values()),
            "wrong_side": any(np.isfinite(v) and v < 0.5 for v in sh.values())}


# ====================================================================== H-SP4

SP4_REFITS = (date(2026, 12, 1), date(2027, 1, 1), date(2027, 2, 1))


def sp4_due(as_of: datetime) -> list[date]:
    """Refit month starts whose training window (labels ending before the month) is complete at `as_of`."""
    return [m for m in SP4_REFITS if as_of >= datetime(m.year, m.month, m.day, tzinfo=UTC) + timedelta(days=1)]


def sp4_check(cluster_rows: list[dict]) -> dict:
    """cluster_rows: [{'members': [...], 'prob': p}] of one refit. Vol-level cluster = the one holding lrv_w,
    calendar cluster = the one holding next_is_weekend."""
    def find(name: str) -> dict | None:
        return next((r for r in cluster_rows if name in r["members"]), None)

    vol, cal = find("lrv_w"), find("next_is_weekend")
    others = [r["prob"] for r in cluster_rows if r is not vol and r is not cal]
    pv, pc = (vol or {}).get("prob", 0.0), (cal or {}).get("prob", 0.0)
    mo = max(others) if others else 0.0
    return {"p_vol": pv, "p_cal": pc, "max_other": mo, "ok": bool(pv > 0.8 and pc > 0.8 and mo <= 0.5)}


def sp4_refit(month: date, n_sub: int = 200) -> dict:  # pragma: no cover - heavy, first due 2026-12-02
    """H-SP2 cluster-level stability selection on all rows whose label ends before `month` (panel rebuilt as of the
    refit), clusters (|rho| >= 0.8) on the same training rows."""
    from ai_trader.ml import sparse as sp
    from ai_trader.ml.features import build_panel, complete_rows
    from ai_trader.research import ml_signals_study as ms
    from ai_trader.research import sparsity_study as ss
    from ai_trader.research.trend_zoo_study import UNIVERSE

    as_of = datetime(month.year, month.month, month.day, tzinfo=UTC)
    cache = out_dir() / "cache" / f"ml_panel_{month.isoformat()}.parquet"
    full = build_panel(as_of, UNIVERSE, cache=cache, workers=1)
    keep = ms.load_panel
    ms.load_panel = lambda: (full, complete_rows(full))
    try:
        lib, names, _ = ss.library()
    finally:
        ms.load_panel = keep
    tr = lib.filter(pl.col("T") + pl.duration(days=1) <= as_of)
    X = tr.select(names).to_numpy().astype(float)
    y = tr["lrv_next"].to_numpy().astype(float)
    T = tr["T"].to_numpy().astype("datetime64[D]")
    cl = ss.clusters(X, names)
    q_max = sp.q_for_bound(len(names), 0.8, 1.0)
    lams = sp.lambda_grid(X, y, 40, 0.005)
    res = sp.stability_selection(X, y, T, n_sub=n_sub, alpha=0.5, lams=lams, seed=0, stop_union=int(4 * q_max), keep_sel=True)
    k = res.region(q_max)
    rows = [{"members": [names[i] for i in m], "prob": float(res.sel[:, : k + 1][:, :, m].any(axis=2).mean(axis=0).max())} for m in cl]
    rows.sort(key=lambda r: -r["prob"])
    return {"month": month.isoformat(), "rows": len(y), "check": sp4_check(rows), "top": rows[:8]}


# ====================================================================== H-PD5

PD5_FIRST_WEEK = date(2026, 11, 16)   # first Monday after registration (2026-11-11)


def pd5_weekly(panel: pl.DataFrame) -> pl.DataFrame:
    """Strategy-level weekly net P&L (universe L): mean over symbols of daily pnl_d, summed over complete Mon-Sun weeks."""
    d = forward_only(panel.filter(pl.col("universe") == "L"), "date", datetime(PD5_FIRST_WEEK.year, PD5_FIRST_WEEK.month, PD5_FIRST_WEEK.day, tzinfo=UTC))
    if d.height == 0:
        return pl.DataFrame(schema={"strategy": pl.Utf8, "week": pl.Date, "pnl": pl.Float64})
    day = d.group_by("strategy", "date").agg(pl.col("pnl_d").mean().alias("p"))
    day = day.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
    w = day.group_by("strategy", "week").agg(pl.col("p").sum().alias("pnl"), pl.col("date").n_unique().alias("nd"))
    return w.filter(pl.col("nd") == 7).drop("nd").sort("week", "strategy")


def pd5_stat(weekly: pl.DataFrame) -> dict:
    from ai_trader.research.evaluate import newey_west_t
    from ai_trader.research.strategy_panel import spearman

    weeks = sorted(set(weekly["week"].to_list())) if weekly.height else []
    ics = []
    for a, b in zip(weeks[:-1], weeks[1:], strict=False):
        if (b - a).days != 7:
            continue
        x = weekly.filter(pl.col("week") == a).select("strategy", pl.col("pnl").alias("x"))
        y = weekly.filter(pl.col("week") == b).select("strategy", pl.col("pnl").alias("y"))
        j = x.join(y, on="strategy", how="inner")
        if j.height >= 5:
            ics.append({"week": b.isoformat(), "ic": spearman(j["x"].to_numpy(), j["y"].to_numpy()), "n_strategies": j.height})
    if not ics:
        return {"n": 0, "est": float("nan"), "t": float("nan"), "met": False, "wrong_side": False, "ics": []}
    m, t, _ = newey_west_t(np.array([r["ic"] for r in ics]))
    return {"n": len(ics), "est": float(m), "t": float(t), "ics": ics,
            "met": bool(m < 0 and np.isfinite(t) and t < -2), "wrong_side": bool(np.isfinite(m) and m >= 0)}


def _pd5_job(sym: str, as_of_iso: str) -> pl.DataFrame | None:
    from ai_trader.research.strategy_panel import finalize_panel, symbol_job

    r = symbol_job(sym, "L", as_of_iso)
    if "skip" in r:
        return None
    return finalize_panel(r["daily"].filter(pl.col("date") >= PD5_FIRST_WEEK))


def pd5_forward_panel(as_of: datetime, workers: int = WORKERS) -> pl.DataFrame:
    """Forward extension of the strategy panel (universe L, dates >= the first forward Monday); the deterministic
    strategies run on the full history exactly as strategy_panel.build, only forward dates are kept."""
    from ai_trader.research.strategy_panel import UNIV_L

    syms = list(UNIV_L)
    res = pmap(_pd5_job, syms, [as_of.isoformat()] * len(syms), workers=workers)
    parts = [r for r in res if r is not None and r.height]
    p = pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()
    if p.height:
        p.write_parquet(out_dir() / "strategy_panel_fwd.parquet")
    return p


# ====================================================================== orchestration

class Context:
    """Lazy, per-run data cache shared by the evaluators."""

    def __init__(self, as_of: datetime, workers: int = WORKERS):
        self.as_of, self.workers, self._c = as_of, workers, {}

    def get(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._c:
            wait_for_ram()
            t = time.time()
            self._c[key] = fn()
            print(f"  [data] {key} loaded in {time.time() - t:.0f}s", flush=True)
        return self._c[key]

    def daily(self) -> pl.DataFrame:
        return self.get("daily", lambda: load_daily(self.as_of))

    def flow_fwd(self) -> pl.DataFrame:
        s = FORWARD["H-OF6"].start
        return self.get("flow_fwd", lambda: flow_panel(self.as_of, s - timedelta(days=1, hours=1), s, self.workers))


def _base(spec: Spec, as_of: datetime) -> dict:
    return {"hypothesis": spec.hid, "registered": spec.registered, "forward_from": spec.start.isoformat(), "need": spec.need,
            "unit": spec.unit, "criterion": spec.criterion}


def ev_lev6(ctx: Context) -> dict:
    sp = FORWARD["H-LEV6"]
    d = ctx.daily()
    par = frozen(sp.hid, lambda: lev6_freeze(d, sp.start))
    return lev6_stat(d, sp.start, par)


def ev_vm6(ctx: Context) -> dict:
    sp = FORWARD["H-VM6"]
    d = ctx.daily()
    par = frozen(sp.hid, lambda: vm6_freeze(d, sp.start))
    return vm6_stat(d, sp.start, par)


def ev_nn7(ctx: Context) -> dict:
    return nn7_stat(ctx.daily(), FORWARD["H-NN7"].start)


def ev_lev7(ctx: Context) -> dict:
    return lev7_stat(ctx.daily(), FORWARD["H-LEV7"].start)


def ev_ds4(ctx: Context) -> dict:
    sp = FORWARD["H-DS4"]
    d = ctx.daily()
    par = frozen(sp.hid, lambda: ds4_freeze(d, sp.start, _d(2026, 11, 12)))
    fwd_syms = sorted(set(forward_only(d.filter(pl.col("universe") == "L"), "origin", sp.start)["symbol"].to_list()))
    if not fwd_syms:
        return ds4_stat(ds4_rows(d.head(0), pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date, "v0_dir": pl.Utf8}), sp.start, par))
    v0 = ctx.get("v0", lambda: v0_direction(fwd_syms, ctx.as_of))
    return ds4_stat(ds4_rows(d, v0, sp.start, par))


def ev_of6(ctx: Context) -> dict:
    sp = FORWARD["H-OF6"]
    par = frozen(sp.hid, lambda: of6_freeze_from_panel(flow_panel(OF6_FREEZE_AT, None, None, ctx.workers)))
    return of6_stat(ctx.flow_fwd(), sp.start, par)


def ev_of7(ctx: Context) -> dict:
    return of7_stat(ctx.flow_fwd(), FORWARD["H-OF7"].start)


def ev_tp1b(ctx: Context) -> dict:
    sp = FORWARD["H-TP1b"]
    return tp1b_stat(ctx.get("tp1b", lambda: load_tp1b(ctx.as_of)), sp.start)


def ev_ms5b(ctx: Context) -> dict:
    rows = []
    cache = out_dir() / "ms5b_symbol_days.json"
    done = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    for day, bd in ms5b_days():
        if day.isoformat() not in done:
            done[day.isoformat()] = ms5b_day(day, bd)
        rows += done[day.isoformat()]
    if done:
        cache.write_text(json.dumps(done, default=float), encoding="utf-8")
    out = ms5b_stat(rows)
    out["days_found"] = sorted(done)
    return out


def ev_nn10(ctx: Context) -> dict:
    from ai_trader.research import vol_ml_forward_study as vf

    r = vf.run(ctx.as_of, record=False)
    lg = r["logged"]
    return {"n": int(r["forward_days"]), "est": lg.get("diff"), "t": lg.get("t"), "n_obs": lg.get("n_obs"),
            "reconstructed": r.get("reconstructed"), "met": vf.verdict({**lg, "n_dates": 10**6}) == "supports",
            "wrong_side": bool(np.isfinite(lg.get("diff", float("nan"))) and lg["diff"] >= 0), "delegate": "vol_ml_forward_study"}


def ev_sp4(ctx: Context) -> dict:
    due = sp4_due(ctx.as_of)
    res = []
    for m in due:
        p = out_dir() / f"sp4_{m.isoformat()}.json"
        if not p.is_file():
            p.write_text(json.dumps(sp4_refit(m), default=float), encoding="utf-8")
        res.append(json.loads(p.read_text(encoding="utf-8")))
    oks = [r["check"]["ok"] for r in res]
    return {"n": len(res), "refits": [{"month": r["month"], **r["check"]} for r in res], "est": float(np.mean(oks)) if oks else float("nan"),
            "met": bool(len(oks) >= 3 and all(oks)), "wrong_side": bool(any(not o for o in oks)),
            "next_due": next((m.isoformat() for m in SP4_REFITS if m not in due), None)}


def ev_pd5(ctx: Context) -> dict:
    first_pair_end = PD5_FIRST_WEEK + timedelta(days=14)   # two complete forward weeks
    if ctx.as_of.date() < first_pair_end:
        return {"n": 0, "est": float("nan"), "t": float("nan"), "met": False, "wrong_side": False,
                "note": f"first forward week pair completes {first_pair_end.isoformat()}; panel extension not built yet"}
    p = pd5_forward_panel(ctx.as_of, ctx.workers)
    return pd5_stat(pd5_weekly(p)) if p.height else {"n": 0, "est": float("nan"), "met": False, "wrong_side": False}


EVALUATORS: dict[str, Callable[[Context], dict]] = {
    "H-LEV6": ev_lev6, "H-VM6": ev_vm6, "H-NN7": ev_nn7, "H-LEV7": ev_lev7, "H-DS4": ev_ds4,
    "H-OF6": ev_of6, "H-OF7": ev_of7, "H-TP1b": ev_tp1b, "H-MS5b": ev_ms5b, "H-NN10": ev_nn10,
    "H-SP4": ev_sp4, "H-PD5": ev_pd5,
}


def already_recorded(hid: str) -> bool:
    from ai_trader.research.hypotheses import results

    return any(r["segment"] == "clean_holdout" for r in results(hid))


def evaluate_one(hid: str, ctx: Context, record: bool = False, recorder: Callable | None = None) -> dict:
    sp = FORWARD[hid]
    out = _base(sp, ctx.as_of)
    t0 = time.time()
    try:
        st = EVALUATORS[hid](ctx)
    except Exception as e:  # noqa: BLE001 - one failing evaluator must not stop the others
        out.update({"status": "error", "error": f"{type(e).__name__}: {e}"[:300], "verdict": "pending", "n": 0})
        return out
    out.update({k: v for k, v in st.items() if k not in ("met", "wrong_side")})
    out["n"] = int(st.get("n") or 0)
    out["interim_sign"] = sign_of(st.get("est"))
    out["verdict"] = decide(out["n"], sp.need, bool(st.get("met")), bool(st.get("wrong_side")))
    out["status"] = "collecting" if out["verdict"] == "pending" else "decidable"
    out["secs"] = round(time.time() - t0, 1)
    if record:
        out["record"] = record_result(hid, out, recorder)
    return out


def record_result(hid: str, out: dict, recorder: Callable | None = None) -> str:
    """Records once, only when the registered sample-size condition is met."""
    sp = FORWARD[hid]
    if out.get("verdict") == "pending" or out.get("n", 0) < sp.need:
        return f"refused: n {out.get('n', 0)} < required {sp.need} ({sp.unit})"
    if already_recorded(hid):
        return "skipped: already recorded (clean_holdout)"
    if hid == "H-NN10":   # its own runner records (same rule: refuses below 60 days, records once)
        from ai_trader.research import vol_ml_forward_study as vf

        return str(vf.run(datetime.fromisoformat(out["as_of"]) if out.get("as_of") else None, record=True).get("record"))
    from ai_trader.research.hypotheses import record as rec

    metrics = _f({k: v for k, v in out.items() if k not in ("hypothesis", "criterion", "status", "record", "verdict")})
    (recorder or rec)(hid, "clean_holdout", out["verdict"], metrics,
                      note=f"forward test (agent F runner v{VERSION}): data from {sp.start.date()} only, parameters frozen before; {sp.criterion}",
                      data="src/ai_trader/research/forward_tests.py; data/research/forward_tests/")
    return "recorded"


def run(as_of: datetime | None = None, record: bool = False, only: list[str] | None = None, workers: int = WORKERS) -> dict:
    from ai_trader.core.time import now

    as_of = as_of or now()
    ctx = Context(as_of, workers)
    rows = []
    for hid in (only or list(EVALUATORS)):
        print(f"{hid} ...", flush=True)
        r = evaluate_one(hid, ctx, record)
        r["as_of"] = as_of.isoformat()
        rows.append(_f(r))
        print(f"  {hid}: n {r['n']}/{FORWARD[hid].need} {FORWARD[hid].unit}; interim {r.get('interim_sign')} "
              f"(est {_f(r.get('est'))}, t {_f(r.get('t'), 2)}); {r['verdict']}" + (f"; {r['record']}" if record else "")
              + (f"; ERROR {r['error']}" if r.get("error") else ""), flush=True)
    status = {"as_of": as_of.isoformat(), "version": VERSION, "tests": rows,
              "blocked": [{"hypothesis": k, "reason": v} for k, v in BLOCKED.items()]}
    return status


def render_md(status: dict) -> str:
    L = [f"# Forward tests: status as of {status['as_of'][:16]} UTC", "",
         "_Generated by `scripts/forward_tests.py` (src/ai_trader/research/forward_tests.py). Each statistic uses only data "
         "after the hypothesis' registration, with parameters frozen on earlier data; nothing is recorded until the "
         "registered sample size is reached, then once (segment clean_holdout)._", "",
         "| hypothesis | forward from | n / needed | unit | interim est | t | sign | status |", "|---|---|---|---|---|---|---|---|"]
    for r in status["tests"]:
        est = r.get("est")
        L.append(f"| {r['hypothesis']} | {r['forward_from'][:10]} | {r.get('n', 0)} / {r['need']} | {r['unit']} | "
                 f"{'' if est is None else est} | {'' if r.get('t') is None else r.get('t')} | {r.get('interim_sign', 'n/a')} | "
                 f"{r.get('status')}{' (' + r['verdict'] + ')' if r.get('verdict') not in ('pending', None) else ''}"
                 f"{' ERROR: ' + r['error'] if r.get('error') else ''} |")
    L += ["", "Blocked or out of scope:", ""]
    L += [f"- **{b['hypothesis']}**: {b['reason']}" for b in status["blocked"]]
    L += ["", "Notes:", ""]
    for r in status["tests"]:
        extra = {k: r[k] for k in ("by_universe", "groups", "ci95", "share_pos", "days_found", "refits", "next_due", "note", "n_events", "n_obs")
                 if k in r and r[k] not in (None, [], {})}
        if extra:
            L.append(f"- {r['hypothesis']}: `{json.dumps(extra, default=str)[:400]}`")
    return "\n".join(L) + "\n"


def write_outputs(status: dict) -> tuple[Path, Path]:
    from ai_trader.core.config import load

    p = out_dir() / "status.json"
    p.write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    md = load().root / "research" / "reports" / "forward_tests_status.md"
    md.write_text(render_md(status), encoding="utf-8")
    return p, md


__all__ = ["BLOCKED", "EVALUATORS", "FORWARD", "Context", "decide", "evaluate_one", "forward_only", "frozen", "record_result",
           "render_md", "run", "write_outputs"]
