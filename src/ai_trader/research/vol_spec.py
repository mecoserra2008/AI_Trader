"""WS04 VOL (+ trimmed WS03 DIST core): volatility specification dossiers (specification program 2026-11-27).

Hypotheses H-WS04-1..7 (research/hypotheses.yaml), design frozen in the diary before any fit (2026-11-26 ~23:58 UTC).

Data: Bybit linear 5m klines (public REST). Crypto BTC ETH SOL XRP DOGE from the BOS cache (2022-01..2026-11-15), the
intraday_geometry extension (to 2026-11-25 22:00) and our own REST extension (data/raw/WS04); XAU / XAG from 2026-03-09
(data/raw/WS04). Per symbol an hourly grid of 5m realised measures (RV, bipower BV, sum r^4, downside semivariance) and
close-to-close returns is built, then 4h and 1d grids. A bar's measures are known at its close.

Forecasts (one per bar i of the own-frequency grid g in {1h, 4h, 1d}, made at the bar's open, i.e. from bars < i):
  rbc          robust_bc on g-bars (features.vol_models, bias-corrected, no weekend shrink) = the live sigma_risk_next
  rbc1h_flat   (4h, 1d) the 1h rbc forecast x hours in the bar
  rbc_how      the 1h rbc x the causal clock factor of the bar's hours (H-NS4 style; summed over the hours at 4h / 1d)
  ewma         RiskMetrics 0.94 of g-bar squared returns
  garch_n/t/st GARCH(1,1) Normal / Student-t / Hansen skew-t (arch, zero mean, returns x 100)
  gjr_t, egarch_t, aparch_t, figarch_t   GJR(1,1,1)-t, EGARCH(1,1,1)-t, APARCH(1,1,1)-t, FIGARCH(1,d,1)-t
  gjr_tc       GJR-t fit on clock-deflated returns r / sqrt(f), forecast x f
  har, harq, harj   level HAR (Corsi), HAR-Q (BPQ 2016: + RV_1 sqrt(RQ_1)), HAR-CJ (BV components + jump lag 1)
  loghar       log-HAR (log of the component means), Duan smearing back-transform
  loghar_c     log-HAR on clock-deflated RV, forecast x f (the intraday / weekly clock)
  loghar_cr    loghar_c + a dummy for a scheduled CPI / NFP / FOMC inside the bar (schedule known before the open)
  loghar_cny   (1h) loghar_c with the clock in New York local time (DST-aware)
HAR lags: 1h {1, 4, 24, 168, 720} h; 4h {1, 6, 42, 180} bars; 1d {1, 7, 30} days.
Clock factor (per refit, training data only): cell mean of min(RV_i / slow_i, 50), slow = the longest HAR component known
at the bar's open; shrunk toward the grand mean with kappa = 20; circular [.25, .5, .25] smoothing over hour of week at 1h;
normalised to mean 1 over cells. Cells: 168 hours of week (1h), 42 4h-blocks of week (4h), 7 weekdays (1d).
Refits on the first of each month (00:00 UTC); training = bars before the refit, capped at 3 years; level HAR insanity filter.

Evaluation: QLIKE (primary; Patton 2011 robust to proxy noise) and MSE on the 5m RV of the bar; Diebold-Mariano with a
Newey-West HAC variance and the Harvey-Leybourne-Newbold correction; Model Confidence Set (Hansen-Lunde-Nason, Tmax, 90%,
circular block bootstrap) per asset x horizon x segment and for the pooled crypto panel (cross-asset mean loss per origin);
Mincer-Zarnowitz in logs (HAC Wald); weekend / session / release splits. Distribution (WS03 core): z = r / sqrt(F);
Normal, Student-t and Hansen skew-t (scale fitted, location 0) on llm_in_sample z; GPD tails (POT at 5%); holdout log
score, Kupiec, Christoffersen, PIT KS and Berkowitz; 4-sigma tail and 24h first passage; VaR / ES of the $1.30 risk unit.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.vol_spec download|build|forecast|evaluate|dist [--symbols ..]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load

CRYPTO = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
METALS = ("XAUUSDT", "XAGUSDT")
GRIDS = {"1h": 1, "4h": 4, "1d": 24}                      # hours per bar
M5 = {"1h": 12, "4h": 48, "1d": 288}                     # 5m returns per bar
LAGS = {"1h": (1, 4, 24, 168, 720), "4h": (1, 6, 42, 180), "1d": (1, 7, 30)}
CELLS = {"1h": 168, "4h": 42, "1d": 7}
BLOCK = {"1h": 48, "4h": 12, "1d": 10}
EVAL_START = {"crypto": datetime(2023, 1, 1, tzinfo=UTC), "metals": datetime(2026, 5, 1, tzinfo=UTC)}
EVAL_START_1D_METALS = datetime(2026, 6, 1, tzinfo=UTC)
METALS_START = datetime(2026, 3, 9, tzinfo=UTC)
FRESH = datetime(2026, 11, 16, tzinfo=UTC)
CAP_YEARS = 3
KAPPA = 20.0
RATIO_CAP = 50.0
GARCH_SPECS = ("garch_n", "garch_t", "garch_st", "gjr_t", "egarch_t", "aparch_t", "figarch_t", "gjr_tc")
HAR_SPECS = ("har", "harq", "harj", "loghar", "loghar_c", "loghar_cr", "loghar_cny")
SIMPLICITY = ("rbc", "rbc_how", "rbc1h_flat", "ewma", "har", "loghar", "loghar_c", "loghar_cr", "loghar_cny", "harq", "harj",
              "garch_n", "garch_t", "garch_st", "gjr_t", "gjr_tc", "egarch_t", "aparch_t", "figarch_t")


def root() -> Path:
    p = load().root / "data" / "research" / "ws04"
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir() -> Path:
    p = load().root / "data" / "raw" / "WS04"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cutoff() -> datetime:
    from ai_trader.research.evaluate import knowledge_cutoff

    return knowledge_cutoff()


def group(sym: str) -> str:
    return "metals" if sym in METALS else "crypto"


# ------------------------------------------------------------------ statistics helpers


def nw_var(e: np.ndarray, lag: int | None = None) -> float:
    """Bartlett long-run variance of a demeaned series (Newey-West automatic lag unless given)."""
    n = len(e)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * float(e[k:] @ e[:-k]) / n
    return max(v, 1e-300)


def dm_test(d: np.ndarray, h: int = 1, lag: int | None = None) -> dict:
    """Diebold-Mariano on a loss differential d (a - b): mean, HAC t, the HLN small-sample corrected t and its p (t_{n-1})."""
    from scipy import stats

    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return {"mean": float(np.mean(d)) if n else float("nan"), "t": float("nan"), "t_hln": float("nan"), "p": float("nan"), "n": n}
    m = float(d.mean())
    v = nw_var(d - m, lag)
    t = m / math.sqrt(v / n)
    k = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    th = t * k
    p = float(2 * stats.t.sf(abs(th), n - 1))
    return {"mean": m, "t": float(t), "t_hln": float(th), "p": p, "n": n}


def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Patton's QLIKE RV/F - log(RV/F) - 1 (RV floored for the log; loss differences do not depend on the floor)."""
    rv = np.asarray(rv, float)
    f = np.asarray(f, float)
    x = np.maximum(rv, 1e-14) / f
    return x - np.log(x) - 1.0


def mse(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (np.asarray(rv, float) - np.asarray(f, float)) ** 2


def mcs(losses: np.ndarray, block: int, alpha: float = 0.10, B: int = 1000, seed: int = 0) -> dict:
    from ai_trader.features.vol_panel import mcs as _mcs

    return _mcs(losses, alpha=alpha, block=block, B=B, stat="Tmax", seed=seed)


def kupiec(hits: np.ndarray, p: float) -> dict:
    from scipy import stats

    h = np.asarray(hits, bool)
    n, x = len(h), int(h.sum())
    if n == 0:
        return {"n": 0, "x": 0, "rate": float("nan"), "p": float("nan")}
    ph = x / n
    ll0 = x * math.log(p) + (n - x) * math.log(1 - p)
    ll1 = (x * math.log(ph) if x else 0.0) + ((n - x) * math.log(1 - ph) if n - x else 0.0)
    lr = -2 * (ll0 - ll1)
    return {"n": n, "x": x, "rate": ph, "p": float(stats.chi2.sf(lr, 1))}


def christoffersen(hits: np.ndarray, p: float) -> dict:
    """Independence (first-order Markov) and conditional coverage LR tests."""
    from scipy import stats

    h = np.asarray(hits, int)
    if len(h) < 10:
        return {"p_ind": float("nan"), "p_cc": float("nan")}
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())

    def ll(k0, k1, q):
        return (k0 * math.log(1 - q) if k0 else 0.0) + (k1 * math.log(q) if k1 else 0.0) if 0 < q < 1 else 0.0

    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2 * (ll(n00 + n10, n01 + n11, p1) - ll(n00, n01, p01) - ll(n10, n11, p11))
    lr_uc = -2 * (ll(len(b) - b.sum(), b.sum(), p) - ll(len(b) - b.sum(), b.sum(), b.mean() if 0 < b.mean() < 1 else p))
    return {"p_ind": float(stats.chi2.sf(max(lr_ind, 0), 1)), "p_cc": float(stats.chi2.sf(max(lr_ind + lr_uc, 0), 2))}


def mz_log(rv: np.ndarray, f: np.ndarray) -> dict:
    """Mincer-Zarnowitz in logs: log RV = a + b log F; HAC Wald of (a, b) = (0, 1). Uses RV > 0 only."""
    from scipy import stats

    ok = np.isfinite(rv) & np.isfinite(f) & (rv > 0) & (f > 0)
    y, x = np.log(rv[ok]), np.log(f[ok])
    n = len(y)
    if n < 50:
        return {"n": n}
    X = np.column_stack([np.ones(n), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lag + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    dlt = b - np.array([0.0, 1.0])
    W = float(dlt @ np.linalg.solve(V, dlt))
    r2 = 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return {"n": n, "a": float(b[0]), "b": float(b[1]), "se_b": float(math.sqrt(V[1, 1])), "wald_p": float(stats.chi2.sf(W, 2)), "r2": r2}


# ------------------------------------------------------------------ data


def download(symbols: list[str]) -> dict:
    """Public REST 5m klines: metals from 2026-03-09, crypto from the end of the intraday_geometry extension."""
    from ai_trader.research.bos_download import fetch_range

    now_ms = int(datetime.now(UTC).timestamp() * 1000) // 300_000 * 300_000
    out = {}
    for s in symbols:
        start = METALS_START if s in METALS else datetime(2026, 11, 25, 20, 0, tzinfo=UTC)
        df = fetch_range(s, 5, int(start.timestamp() * 1000), now_ms)
        df.write_parquet(raw_dir() / f"{s}_5m.parquet", compression="zstd")
        out[s] = {"rows": df.height, "first": str(df["event_ts"].min()) if df.height else None, "last": str(df["event_ts"].max()) if df.height else None}
    return out


def load_5m(sym: str) -> pl.DataFrame:
    cols = ["event_ts", "open", "high", "low", "close"]
    base = load().root / "data" / "research"
    files = sorted((base / "bos" / "bars" / "5m").glob(f"{sym}_*.parquet"))
    files += [p for p in [base / "intraday_geometry" / "bars" / f"{sym}_ext.parquet", raw_dir() / f"{sym}_5m.parquet"] if p.exists()]
    parts = [pl.read_parquet(f, columns=cols).with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC"))) for f in files]
    d = pl.concat(parts).unique("event_ts", keep="last").sort("event_ts")
    return d.filter((pl.col("close") > 0) & (pl.col("low") > 0))


def hourly_grid(d5: pl.DataFrame) -> pl.DataFrame:
    """Complete hourly calendar: rv, bv, q4 (sum r^4), rsn (downside semivariance), n5 (valid 5m returns), close, ret."""
    d = d5.select("event_ts", "close").with_columns(pl.col("close").log().alias("lc"))
    d = d.with_columns((pl.col("lc") - pl.col("lc").shift(1)).alias("r"),
                       (pl.col("event_ts") - pl.col("event_ts").shift(1)).dt.total_seconds().alias("dt"))
    d = d.with_columns(pl.when(pl.col("dt") == 300).then(pl.col("r")).otherwise(None).alias("r"))
    d = d.with_columns((pl.col("r").abs() * pl.col("r").shift(1).abs()).alias("bp"))
    g = (d.group_by_dynamic("event_ts", every="1h", closed="left", label="left")
         .agg((pl.col("r") ** 2).sum().alias("rv"), pl.col("bp").sum().alias("bp"), pl.col("bp").is_not_null().sum().alias("nbp"),
              (pl.col("r") ** 4).sum().alias("q4"), (pl.col("r") ** 2 * (pl.col("r") < 0)).sum().alias("rsn"),
              pl.col("r").is_not_null().sum().alias("n5"), pl.col("lc").last().alias("lc_last")))
    t0, t1 = g["event_ts"].min(), g["event_ts"].max()
    cal = pl.DataFrame({"event_ts": pl.datetime_range(t0, t1, "1h", time_zone="UTC", eager=True)}).with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    g = cal.join(g, on="event_ts", how="left").sort("event_ts")
    ok = pl.col("n5") >= 10
    g = g.with_columns(
        pl.when(ok).then(pl.col("rv") * 12 / pl.col("n5")).otherwise(None).alias("rv"),
        pl.when(ok & (pl.col("nbp") > 0)).then(pl.col("bp") * (math.pi / 2) * 12 / pl.col("nbp")).otherwise(None).alias("bv"),
        pl.when(ok).then(pl.col("q4") * 12 / pl.col("n5")).otherwise(None).alias("q4"),
        pl.when(ok).then(pl.col("rsn") * 12 / pl.col("n5")).otherwise(None).alias("rsn"),
        (pl.col("lc_last") - pl.col("lc_last").shift(1)).alias("ret"),
    ).with_columns(pl.when(ok).then(pl.col("ret")).otherwise(None).alias("ret"), pl.col("lc_last").exp().alias("close")).drop("bp", "nbp", "lc_last")
    return g


def aggregate(h: pl.DataFrame, every: str) -> pl.DataFrame:
    """4h / 1d grids from the hourly grid; a bar needs >= 90% of its hours (rescaled), else null."""
    hrs = GRIDS[every]
    g = (h.with_columns(pl.col("close").log().alias("lc"))
         .group_by_dynamic("event_ts", every=every, closed="left", label="left")
         .agg(pl.col("rv").sum(), pl.col("bv").sum(), pl.col("q4").sum(), pl.col("rsn").sum(), pl.col("n5").sum(),
              pl.col("rv").is_not_null().sum().alias("nh"), pl.col("lc").drop_nulls().last().alias("lc_last")))
    ok = pl.col("nh") >= int(math.ceil(0.9 * hrs))
    sc = hrs / pl.col("nh")
    g = g.with_columns(
        *[pl.when(ok).then(pl.col(c) * sc).otherwise(None).alias(c) for c in ("rv", "bv", "q4", "rsn")],
        (pl.col("lc_last") - pl.col("lc_last").shift(1)).alias("ret"),
    ).with_columns(pl.when(ok).then(pl.col("ret")).otherwise(None).alias("ret"), pl.col("lc_last").exp().alias("close")).drop("nh", "lc_last")
    return g


def cells(ts: pl.Series, grid: str, ny: bool = False) -> np.ndarray:
    s = ts.dt.convert_time_zone("America/New_York") if ny else ts
    wd = s.dt.weekday().cast(pl.Int32).to_numpy() - 1
    hr = s.dt.hour().cast(pl.Int32).to_numpy()
    if grid == "1h":
        return (wd * 24 + hr).astype(np.int64)
    if grid == "4h":
        return (wd * 6 + hr // 4).astype(np.int64)
    return wd.astype(np.int64)


def release_flags(ts: pl.Series, hours: int) -> np.ndarray:
    """1 if a scheduled CPI / NFP / FOMC falls in [T_i, T_i + hours) and its schedule was known at T_i."""
    p = load().root / "data" / "research" / "news" / "events" / "events.parquet"
    ev = pl.read_parquet(p).filter(pl.col("kind").is_in(["cpi", "nfp", "fomc"]))
    st = ev["scheduled_ts"].cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
    kn = ev["schedule_known_ts"].cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
    t = ts.cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
    out = np.zeros(len(t), dtype=bool)
    span = hours * 3600 * 1_000_000
    for s, k in zip(st, kn):
        i = np.searchsorted(t, s, side="right") - 1
        if 0 <= i < len(t) and t[i] <= s < t[i] + span and k <= t[i]:
            out[i] = True
    return out


def build(symbols: list[str]) -> dict:
    from ai_trader.features.vol_models import VolModel, run
    from ai_trader.research.calendar import add_calendar

    info = {}
    for s in symbols:
        d5 = load_5m(s)
        if s in METALS:
            d5 = d5.filter(pl.col("event_ts") >= METALS_START)
        h = hourly_grid(d5)
        # last complete hour only
        h = h.filter(pl.col("event_ts") < d5["event_ts"].max().replace(minute=0) )
        grids = {"1h": h, "4h": aggregate(h, "4h"), "1d": aggregate(h, "1d")}
        for g, df in grids.items():
            if g != "1h":
                last_ok = d5["event_ts"].max() + timedelta(minutes=5)
                df = df.filter(pl.col("event_ts") + pl.duration(hours=GRIDS[g]) <= last_ok)
            r = np.nan_to_num(df["ret"].to_numpy().astype(float))
            wk = add_calendar(df.select("event_ts"))["tradfi_weekend"].to_numpy()
            risk = run(r, wk, VolModel(bias_correct=True, use_weekend=False))
            v = risk["v_next"]
            rbc = np.r_[np.nan, v[:-1]]                       # forecast for bar i made at the close of i-1
            df = df.with_columns(pl.Series("rbc", rbc), pl.Series("weekend", wk))
            df.write_parquet(root() / f"grid_{s}_{g}.parquet")
            info[f"{s}_{g}"] = {"rows": df.height, "first": str(df["event_ts"].min()), "last": str(df["event_ts"].max()),
                                "rv_null": int(df["rv"].null_count())}
    return info


def read_grid(s: str, g: str) -> pl.DataFrame:
    return pl.read_parquet(root() / f"grid_{s}_{g}.parquet")


# ------------------------------------------------------------------ forecasting building blocks


def trailing_mean(x: np.ndarray, w: int, min_frac: float = 0.8) -> np.ndarray:
    """Mean of x[i-w..i-1] (known at the open of bar i), NaN-aware, needs min_frac of the window."""
    s = pl.Series(np.asarray(x, float)).fill_nan(None).shift(1)
    return s.rolling_mean(w, min_samples=max(1, int(math.ceil(min_frac * w)))).to_numpy().astype(float)


def clock_factor(y: np.ndarray, slow: np.ndarray, cell: np.ndarray, ncell: int, train: np.ndarray, smooth: bool) -> np.ndarray:
    """Cell factor (mean 1 over cells) from training bars: mean of min(y / slow, cap), shrunk, smoothed."""
    ok = train & np.isfinite(y) & np.isfinite(slow) & (slow > 0)
    u = np.minimum(y[ok] / slow[ok], RATIO_CAP)
    c = cell[ok]
    s = np.bincount(c, weights=u, minlength=ncell)
    n = np.bincount(c, minlength=ncell).astype(float)
    gm = u.mean() if len(u) else 1.0
    m = (s + KAPPA * gm) / (n + KAPPA)
    if smooth:
        m = 0.25 * np.roll(m, 1) + 0.5 * m + 0.25 * np.roll(m, -1)
    return m / m.mean()


def refit_points(ts: np.ndarray, start: datetime) -> list[int]:
    """Indices of the first bar at or after each month start from `start`."""
    t = pl.Series(ts).cast(pl.Datetime("us", "UTC"))
    end = t.max()
    months, m = [], start
    while m <= end:
        months.append(m)
        m = datetime(m.year + (m.month // 12), m.month % 12 + 1, 1, tzinfo=UTC)
    tv = t.dt.epoch("us").to_numpy()
    idx = [int(np.searchsorted(tv, int(mm.timestamp() * 1_000_000))) for mm in months]
    return sorted(set(i for i in idx if i < len(tv)))


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def har_design(name: str, z: dict, g: str) -> np.ndarray:
    """Design matrix (with intercept) of the HAR variant from precomputed component arrays."""
    L = LAGS[g]
    n = len(z["rv"])
    one = np.ones(n)
    if name == "har":
        return np.column_stack([one] + [z[f"c{w}"] for w in L])
    if name == "harq":
        return np.column_stack([one] + [z[f"c{w}"] for w in L] + [z[f"c{L[0]}"] * np.sqrt(z["rq1"])])
    if name == "harj":
        return np.column_stack([one] + [z[f"b{w}"] for w in L] + [z["j1"]])
    if name == "loghar":
        return np.column_stack([one] + [np.log(np.maximum(z[f"c{w}"], z["floor"])) for w in L])
    raise KeyError(name)


def forecast_grid(s: str, g: str) -> pl.DataFrame:
    """All spec forecasts for one symbol and grid (columns = specs), with target, return and calendar columns."""
    df = read_grid(s, g)
    ts = df["event_ts"]
    n = df.height
    y = df["rv"].to_numpy().astype(float)
    bv = df["bv"].to_numpy().astype(float)
    r = df["ret"].to_numpy().astype(float)
    q4 = df["q4"].to_numpy().astype(float)
    L = LAGS[g]
    start = EVAL_START[group(s)] if not (g == "1d" and s in METALS) else EVAL_START_1D_METALS
    tsn = ts.cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
    cap = CAP_YEARS * 365 * 24 // GRIDS[g]
    refits = refit_points(ts.to_numpy(), start)
    out: dict[str, np.ndarray] = {}
    # ---------------- recursive incumbents
    out["rbc"] = df["rbc"].to_numpy().astype(float)
    lam = 0.94
    ew = np.full(n, np.nan)
    rr = np.nan_to_num(r)
    v = float(np.mean(rr[1:31] ** 2)) if n > 31 else np.nan
    for i in range(31, n):
        v = lam * v + (1 - lam) * rr[i - 1] ** 2
        ew[i] = v
    out["ewma"] = ew
    # ---------------- HAR components (known at the open of bar i)
    z = {"rv": y}
    for w in L:
        z[f"c{w}"] = trailing_mean(y, w)
        z[f"b{w}"] = trailing_mean(bv, w)
    z["rq1"] = np.r_[np.nan, (M5[g] / 3.0) * q4[:-1]]
    z["j1"] = np.r_[np.nan, np.maximum(y[:-1] - bv[:-1], 0.0)]
    slow = z[f"c{L[-1]}"]
    cell = cells(ts, g)
    cell_ny = cells(ts, g, ny=True) if g == "1h" else None
    rel = release_flags(ts, GRIDS[g]).astype(float)
    # 1h inputs for rbc_how / rbc1h_flat at coarser grids
    if g != "1h":
        h1 = read_grid(s, "1h")
        t1 = h1["event_ts"].cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
        y1, rbc1 = h1["rv"].to_numpy().astype(float), h1["rbc"].to_numpy().astype(float)
        cell1 = cells(h1["event_ts"], "1h")
        pos = np.searchsorted(t1, tsn)                      # index of the 1h bar that opens with bar i
        hrs = GRIDS[g]
        good = (pos + hrs <= len(t1)) & (pos < len(t1))
        rb = np.full(n, np.nan)
        rb[good] = rbc1[pos[good]]
        out["rbc1h_flat"] = rb * hrs
    for k in ("har", "harq", "harj", "loghar", "loghar_c", "loghar_cr", "loghar_cny", "rbc_how"):
        out[k] = np.full(n, np.nan)
    if g != "1h":
        out.pop("loghar_cny")
    for j, R in enumerate(refits):
        Rn = refits[j + 1] if j + 1 < len(refits) else n
        lo = max(0, R - cap)
        train = np.zeros(n, dtype=bool)
        train[lo:R] = True
        yt = y[lo:R]
        fin_y = np.isfinite(yt)
        if fin_y.sum() < 50:
            continue
        ymed = float(np.nanmedian(yt))
        z["floor"] = 0.01 * ymed
        ymin, ymax, ymean = float(np.nanmin(yt)), float(np.nanmax(yt)), float(np.nanmean(yt))
        sl = slice(R, Rn)
        # level HARs (+ insanity filter)
        for name in ("har", "harq", "harj"):
            X = har_design(name, z, g)
            ok = np.isfinite(X[lo:R]).all(axis=1) & fin_y
            if ok.sum() < 30:
                continue
            b = ols(X[lo:R][ok], yt[ok])
            f = X[sl] @ b
            bad = ~np.isfinite(f) | (f <= 0) | (f < ymin) | (f > ymax)
            f[bad & np.isfinite(X[sl]).all(axis=1)] = ymean
            out[name][sl] = f
        # log-HAR, calendar-blind
        ly = np.log(np.maximum(y, z["floor"]))
        X = har_design("loghar", z, g)
        ok = np.isfinite(X[lo:R]).all(axis=1) & fin_y
        if ok.sum() >= 30:
            b = ols(X[lo:R][ok], ly[lo:R][ok])
            smear = float(np.mean(np.exp(ly[lo:R][ok] - X[lo:R][ok] @ b)))
            out["loghar"][sl] = np.exp(X[sl] @ b) * smear
        # clock variants
        for name, cc in (("loghar_c", cell), ("loghar_cr", cell), ("loghar_cny", cell_ny)):
            if cc is None:
                continue
            f_cell = clock_factor(y, slow, cc, CELLS[g], train, smooth=(g == "1h"))
            fc = f_cell[cc]
            yd = y / fc
            comps = [np.log(np.maximum(trailing_mean(yd, w), z["floor"])) for w in L]
            Xd = np.column_stack([np.ones(n)] + comps + ([rel] if name == "loghar_cr" else []))
            lyd = np.log(np.maximum(yd, z["floor"]))
            ok = np.isfinite(Xd[lo:R]).all(axis=1) & fin_y
            if ok.sum() < 30:
                continue
            b = ols(Xd[lo:R][ok], lyd[lo:R][ok])
            smear = float(np.mean(np.exp(lyd[lo:R][ok] - Xd[lo:R][ok] @ b)))
            out[name][sl] = fc[sl] * np.exp(Xd[sl] @ b) * smear
            if name == "loghar_c":
                out.setdefault("_clock", np.full(n, np.nan))[sl] = fc[sl]
        # rbc x clock
        if g == "1h":
            f_r = clock_factor(y, out["rbc"], cell, 168, train, smooth=True)
            out["rbc_how"][sl] = out["rbc"][sl] * f_r[cell[sl]]
        else:
            tr1 = (t1 < tsn[R]) & (t1 >= tsn[lo])
            f_r = clock_factor(y1, rbc1, cell1, 168, tr1, smooth=True)
            idx = np.arange(R, Rn)
            gd = idx[good[idx]]
            fsum = np.array([f_r[cell1[pos[i]:pos[i] + hrs]].sum() for i in gd])
            out["rbc_how"][gd] = rbc1[pos[gd]] * fsum
    # ---------------- GARCH family
    out.update(garch_family(r, y, z, cell, g, refits, cap, slow))
    clk = out.pop("_clock", np.full(n, np.nan))
    fl = out.pop("_fails")
    (root() / f"fails_{s}_{g}.json").write_text(json.dumps(dict(zip(GARCH_SPECS, fl.tolist()))))
    cols = {"event_ts": ts, "rv": y, "ret": r, "weekend": df["weekend"], "release": rel.astype(bool), "clock": clk}
    res = pl.DataFrame(cols).with_columns(*[pl.Series(k, v.astype(np.float64)) for k, v in out.items()])
    first = tsn[refits[0]] if refits else tsn[-1]
    res = res.filter(pl.col("event_ts").cast(pl.Datetime("us", "UTC")).dt.epoch("us") >= first)
    return res


def _arch_model(name: str, x: np.ndarray):
    from arch import arch_model

    dist = {"n": "normal", "t": "t", "st": "skewt", "tc": "t"}[name.split("_")[1]]
    if name.startswith("garch"):
        return arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    if name.startswith("gjr"):
        return arch_model(x, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=False)
    if name.startswith("egarch"):
        return arch_model(x, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist=dist, rescale=False)
    if name.startswith("aparch"):
        return arch_model(x, mean="Zero", vol="APARCH", p=1, o=1, q=1, dist=dist, rescale=False)
    if name.startswith("figarch"):
        return arch_model(x, mean="Zero", vol="FIGARCH", p=1, q=1, dist=dist, rescale=False)
    raise KeyError(name)


def garch_family(r: np.ndarray, y: np.ndarray, z: dict, cell: np.ndarray, g: str, refits: list[int], cap: int,
                 slow: np.ndarray, specs: tuple[str, ...] = GARCH_SPECS) -> dict[str, np.ndarray]:
    """One-step GARCH-family variance forecasts with monthly refits (warm-started), filtered with fixed parameters."""
    import warnings

    n = len(r)
    x_all = np.nan_to_num(r) * 100.0
    out = {k: np.full(n, np.nan) for k in specs}
    prev: dict[str, np.ndarray | None] = {k: None for k in specs}
    fails = {k: 0 for k in specs}
    for j, R in enumerate(refits):
        Rn = refits[j + 1] if j + 1 < len(refits) else n
        lo = max(0, R - cap)
        if R - lo < 300:
            continue
        train = np.zeros(n, dtype=bool)
        train[lo:R] = True
        fc = None
        for name in specs:
            x = x_all
            if name == "gjr_tc":
                if fc is None:
                    f_cell = clock_factor(y, slow, cell, CELLS[g], train, smooth=(g == "1h"))
                    fc = f_cell[cell]
                x = x_all / np.sqrt(fc)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = _arch_model(name, x[lo:R])
                    res = m.fit(disp="off", starting_values=prev[name], options={"maxiter": 300}, show_warning=False)
                    p = np.asarray(res.params)
                    if not np.all(np.isfinite(p)):
                        raise ValueError("nan params")
                    prev[name] = p
            except Exception:  # noqa: BLE001 - a failed fit reuses the previous month's parameters
                fails[name] += 1
                if prev[name] is None:
                    continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mf = _arch_model(name, x[lo:Rn])
                    fx = mf.fix(prev[name])
                    cv = np.asarray(fx.conditional_volatility)[R - lo:]
                f = (cv / 100.0) ** 2
                if name == "gjr_tc":
                    f = f * fc[R:Rn]
                out[name][R:Rn] = f
            except Exception:  # noqa: BLE001
                fails[name] += 1
    out["_fails"] = np.array([fails[k] for k in specs], dtype=float)
    return out


def _one(task: tuple[str, str]) -> tuple[str, dict]:
    import os

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[v] = "1"
    s, g = task
    t0 = time.time()
    res = forecast_grid(s, g)
    res.write_parquet(root() / f"fc_{s}_{g}.parquet")
    return f"{s}_{g}", {"rows": res.height, "sec": round(time.time() - t0, 1)}


def forecast(symbols: list[str], grids: list[str], workers: int = 1) -> dict:
    """One task per (symbol, grid), largest first; workers > 1 runs single-threaded processes inside one governor job."""
    tasks = sorted(((s, g) for s in symbols for g in grids), key=lambda t: (-{"1h": 3, "4h": 2, "1d": 1}[t[1]], t[0] in METALS))
    info = {}
    if workers <= 1:
        for t in tasks:
            k, v = _one(t)
            info[k] = v
            print(k, v, flush=True)
        return info
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, v in ex.map(_one, tasks):
            info[k] = v
            print(k, v, flush=True)
    return info


# ------------------------------------------------------------------ evaluation


def segments(ts: pl.Series) -> dict[str, np.ndarray]:
    t = ts.cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
    c = int(cutoff().timestamp() * 1_000_000)
    f = int(FRESH.timestamp() * 1_000_000)
    return {"llm_in_sample": t < c, "clean_holdout": t >= c, "fresh": t >= f}


def specs_in(df: pl.DataFrame) -> list[str]:
    return [c for c in SIMPLICITY if c in df.columns]


def load_fc(s: str, g: str) -> pl.DataFrame:
    return pl.read_parquet(root() / f"fc_{s}_{g}.parquet")


def pooled(symbols: list[str], g: str, loss=qlike) -> tuple[pl.DataFrame, list[str]]:
    """Cross-asset mean loss per origin over the common origins of all symbols and all specs."""
    frames, specs = [], None
    for s in symbols:
        f = load_fc(s, g)
        sp = specs_in(f)
        specs = sp if specs is None else [c for c in specs if c in sp]
        frames.append((s, f))
    parts = []
    for s, f in frames:
        rv = f["rv"].to_numpy()
        L = {k: loss(rv, f[k].to_numpy()) for k in specs}
        parts.append(pl.DataFrame({"event_ts": f["event_ts"], "weekend": f["weekend"], "release": f["release"], **L}))
    allp = pl.concat(parts)
    ok = pl.all_horizontal([pl.col(k).is_finite() for k in specs])
    allp = allp.filter(ok)
    cnt = allp.group_by("event_ts").len()
    full = cnt.filter(pl.col("len") == len(symbols))["event_ts"]
    allp = allp.filter(pl.col("event_ts").is_in(full.implode()))
    agg = allp.group_by("event_ts").agg(pl.col("weekend").first(), pl.col("release").max(), *[pl.col(k).mean() for k in specs]).sort("event_ts")
    return agg, specs


def board(L: pl.DataFrame, specs: list[str], g: str, base: str = "rbc") -> dict:
    """Mean losses, ratio to base, DM-HLN vs base and vs the best, MCS membership."""
    M = L.select(specs).to_numpy()
    if M.shape[0] < 30:
        return {"n": int(M.shape[0])}
    mean = M.mean(axis=0)
    res = mcs(M, BLOCK[g])
    best = int(np.argmin(mean))
    bi = specs.index(base) if base in specs else best
    rows = []
    for k, sp in enumerate(specs):
        d_base = dm_test(M[:, k] - M[:, bi]) if k != bi else None
        d_best = dm_test(M[:, k] - M[:, best]) if k != best else None
        rows.append({"spec": sp, "loss": float(mean[k]), "ratio_base": float(mean[k] / mean[bi]),
                     "dm_base_t": d_base["t_hln"] if d_base else None, "dm_best_t": d_best["t_hln"] if d_best else None,
                     "in_mcs": bool(res["included"][k]), "mcs_p": float(res["pvalues"][k])})
    rows.sort(key=lambda x: x["loss"])
    return {"n": int(M.shape[0]), "best": specs[best], "mcs": [specs[k] for k in range(len(specs)) if res["included"][k]], "rows": rows}


def pair(L: pl.DataFrame, a: str, b: str) -> dict:
    if a not in L.columns or b not in L.columns:
        return {}
    d = (L[a] - L[b]).to_numpy()
    out = dm_test(d)
    out["rel"] = float(np.nanmean(d) / np.nanmean(L[b].to_numpy()))
    return out


def evaluate(symbols_c: list[str] = list(CRYPTO), symbols_m: list[str] = list(METALS)) -> dict:
    out: dict = {"pooled": {}, "asset": {}, "pairs": {}, "splits": {}, "mz": {}, "release": {}}
    pairs = [("rbc_how", "rbc"), ("loghar_c", "loghar"), ("loghar_cr", "loghar_c"), ("loghar_cny", "loghar_c"),
             ("loghar_c", "rbc"), ("loghar_c", "rbc_how"), ("gjr_t", "loghar_c"), ("garch_t", "loghar_c"), ("gjr_tc", "loghar_c"),
             ("gjr_tc", "gjr_t"), ("harq", "har"), ("harj", "har"), ("loghar", "har"), ("garch_t", "garch_n"),
             ("garch_st", "garch_t"), ("loghar_c", "rbc1h_flat"), ("rbc_how", "rbc1h_flat"), ("rbc1h_flat", "rbc")]
    for g in GRIDS:
        for name, syms in (("crypto", symbols_c),):
            for lossname, loss in (("qlike", qlike), ("mse", mse)):
                L, specs = pooled(syms, g, loss)
                seg = segments(L["event_ts"])
                key = f"{name}_{g}_{lossname}"
                out["pooled"][key] = {sg: board(L.filter(pl.Series(m)), specs, g) for sg, m in seg.items() if sg != "fresh" or lossname == "qlike"}
                if lossname == "qlike":
                    out["pairs"][f"{name}_{g}"] = {sg: {f"{a}-{b}": pair(L.filter(pl.Series(m)), a, b) for a, b in pairs} for sg, m in seg.items()}
                    sp = {}
                    for sg in ("llm_in_sample", "clean_holdout"):
                        Ls = L.filter(pl.Series(seg[sg]))
                        for wk in (True, False):
                            Lw = Ls.filter(pl.col("weekend") == wk)
                            sp[f"{sg}_{'weekend' if wk else 'weekday'}"] = {
                                "n": Lw.height, **{f"{a}-{b}": pair(Lw, a, b) for a, b in pairs[:7]},
                                "loss": {k: float(Lw[k].mean()) for k in specs} if Lw.height else {}}
                        Lr = Ls.filter(pl.col("release"))
                        out["release"][f"{name}_{g}_{sg}"] = {"n": Lr.height, **{f"{a}-{b}": pair(Lr, a, b) for a, b in (("loghar_cr", "loghar_c"), ("loghar_c", "rbc"), ("rbc_how", "rbc"))}}
                        if g == "1h":
                            hr = Ls["event_ts"].dt.hour().to_numpy()
                            for sess, (h0, h1) in (("asia", (0, 8)), ("europe", (8, 13)), ("us", (13, 20)), ("late", (20, 24))):
                                Lh = Ls.filter(pl.Series((hr >= h0) & (hr < h1)))
                                sp[f"{sg}_{sess}"] = {"n": Lh.height, **{f"{a}-{b}": pair(Lh, a, b) for a, b in pairs[:7]}}
                    out["splits"][f"{name}_{g}"] = sp
        # per asset boards (QLIKE), crypto and metals
        for s in list(symbols_c) + list(symbols_m):
            try:
                f = load_fc(s, g)
            except FileNotFoundError:
                continue
            specs = [k for k in specs_in(f) if f[k].is_finite().sum() > 0.5 * f.height]
            rv = f["rv"].to_numpy()
            L = pl.DataFrame({"event_ts": f["event_ts"], "weekend": f["weekend"], **{k: qlike(rv, f[k].to_numpy()) for k in specs}})
            L = L.filter(pl.all_horizontal([pl.col(k).is_finite() for k in specs]))
            seg = segments(L["event_ts"])
            out["asset"][f"{s}_{g}"] = {sg: board(L.filter(pl.Series(m)), specs, g) for sg, m in seg.items() if sg != "fresh"}
            out["pairs"][f"{s}_{g}"] = {sg: {f"{a}-{b}": pair(L.filter(pl.Series(m)), a, b) for a, b in pairs} for sg, m in seg.items() if sg != "fresh"}
            if s in METALS:
                for sg in ("llm_in_sample", "clean_holdout"):
                    Ls = L.filter(pl.Series(seg[sg]))
                    for wk in (True, False):
                        Lw = Ls.filter(pl.col("weekend") == wk)
                        out["splits"].setdefault(f"{s}_{g}", {})[f"{sg}_{'weekend' if wk else 'weekday'}"] = {
                            "n": Lw.height, **{f"{a}-{b}": pair(Lw, a, b) for a, b in pairs[:7]}}
            # Mincer-Zarnowitz (logs) for the main specs
            ff = f.filter(pl.all_horizontal([pl.col(k).is_finite() for k in specs]))
            sgm = segments(ff["event_ts"])
            out["mz"][f"{s}_{g}"] = {sg: {k: mz_log(ff.filter(pl.Series(m))["rv"].to_numpy(), ff.filter(pl.Series(m))[k].to_numpy())
                                          for k in ("rbc", "rbc_how", "loghar", "loghar_c", "gjr_t", "garch_t", "har") if k in specs}
                                     for sg, m in sgm.items() if sg != "fresh"}
    out["selection"] = select(out)
    (root() / "eval.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def select(out: dict) -> dict:
    """Frozen rule: the simplest spec in the pooled-crypto llm_in_sample 90% MCS that is also in the clean_holdout MCS;
    fallback = in-sample QLIKE winner."""
    sel = {}
    for g in GRIDS:
        b = out["pooled"].get(f"crypto_{g}_qlike", {})
        ins, ho = b.get("llm_in_sample", {}), b.get("clean_holdout", {})
        if "mcs" not in ins or "mcs" not in ho:
            continue
        both = [k for k in SIMPLICITY if k in ins["mcs"] and k in ho["mcs"]]
        sel[g] = {"pick": both[0] if both else ins["best"], "rule": "simplest in both MCS" if both else "fallback in-sample winner",
                  "in_sample_mcs": ins["mcs"], "holdout_mcs": ho["mcs"], "in_sample_best": ins["best"], "holdout_best": ho["best"]}
    return sel


# ------------------------------------------------------------------ distribution (WS03 core)


def _dist_ll(kind: str, z: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Per-observation log density of z under loc 0, scale exp(th[0]) and the shape parameters."""
    from arch.univariate import Normal, SkewStudent, StudentsT

    s2 = np.full(len(z), math.exp(2 * th[0]))
    if kind == "normal":
        return Normal().loglikelihood([], z, s2, individual=True)
    if kind == "t":
        return StudentsT().loglikelihood([2.05 + math.exp(th[1])], z, s2, individual=True)
    return SkewStudent().loglikelihood([2.05 + math.exp(th[1]), math.tanh(th[2])], z, s2, individual=True)


def _params(kind: str, th: np.ndarray) -> dict:
    p = {"scale": math.exp(th[0])}
    if kind in ("t", "skewt"):
        p["nu"] = 2.05 + math.exp(th[1])
    if kind == "skewt":
        p["lam"] = math.tanh(th[2])
    return p


def fit_dist(kind: str, z: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    x0 = {"normal": [0.0], "t": [0.0, math.log(3.0)], "skewt": [0.0, math.log(3.0), 0.0]}[kind]
    f = lambda th: -float(np.sum(_dist_ll(kind, z, np.asarray(th))))  # noqa: E731
    r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-4})
    return np.asarray(r.x)


def dist_cdf(kind: str, z: np.ndarray, th: np.ndarray) -> np.ndarray:
    from arch.univariate import SkewStudent, StudentsT
    from scipy import stats

    x = z / math.exp(th[0])
    if kind == "normal":
        return stats.norm.cdf(x)
    if kind == "t":
        return StudentsT().cdf(x, [2.05 + math.exp(th[1])])
    return SkewStudent().cdf(x, [2.05 + math.exp(th[1]), math.tanh(th[2])])


def dist_ppf(kind: str, p: float, th: np.ndarray) -> float:
    from arch.univariate import SkewStudent, StudentsT
    from scipy import stats

    if kind == "normal":
        q = stats.norm.ppf(p)
    elif kind == "t":
        q = StudentsT().ppf(np.array([p]), [2.05 + math.exp(th[1])])[0]
    else:
        q = SkewStudent().ppf(np.array([p]), [2.05 + math.exp(th[1]), math.tanh(th[2])])[0]
    return float(q * math.exp(th[0]))


def gpd_tail(z: np.ndarray, side: str, u_p: float = 0.05) -> dict:
    """POT/GPD on the lower (side='lo') or upper tail beyond the u_p quantile; returns threshold, xi, beta."""
    from scipy import stats

    x = -z if side == "lo" else z
    u = float(np.quantile(x, 1 - u_p))
    e = x[x > u] - u
    xi, _, beta = stats.genpareto.fit(e, floc=0)
    return {"u": u, "xi": float(xi), "beta": float(beta), "n_exc": int(len(e)), "u_p": u_p}


def gpd_quantile(t: dict, p: float) -> float:
    """Tail quantile of z with exceedance probability p (< u_p) on the GPD side (returns a z value)."""
    xi, beta, u = t["xi"], t["beta"], t["u"]
    q = u + (beta / xi) * ((p / t["u_p"]) ** (-xi) - 1) if abs(xi) > 1e-6 else u - beta * math.log(p / t["u_p"])
    return q


def berkowitz(u: np.ndarray) -> float:
    """Berkowitz (2001) LR test of iid N(0,1) of Phi^-1(PIT) against AR(1) with free mean / variance; p-value (3 df)."""
    from scipy import stats

    x = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    y, xl = x[1:], x[:-1]
    X = np.column_stack([np.ones(len(xl)), xl])
    b = ols(X, y)
    e = y - X @ b
    s2 = float(e @ e) / len(e)
    ll1 = float(np.sum(stats.norm.logpdf(e, scale=math.sqrt(s2))))
    ll0 = float(np.sum(stats.norm.logpdf(y)))
    return float(stats.chi2.sf(max(-2 * (ll0 - ll1), 0), 3))


def dist_block(z_in: np.ndarray, z_ho: np.ndarray, wk_in: np.ndarray | None = None, wk_ho: np.ndarray | None = None) -> dict:
    """Fit Normal / t / skew-t on in-sample z; holdout log score differences, VaR tests, PIT, 4-sigma tail, GPD tails;
    weekend-only 1% VaR coverage of the t (H-WS04-5b); empirical and t tail quantile / ES of the standardised move."""
    from scipy import stats

    res: dict = {"n_in": int(len(z_in)), "n_ho": int(len(z_ho)),
                 "moments_in": {"var": float(np.var(z_in)), "skew": float(stats.skew(z_in)), "exkurt": float(stats.kurtosis(z_in))},
                 "moments_ho": {"var": float(np.var(z_ho)), "skew": float(stats.skew(z_ho)), "exkurt": float(stats.kurtosis(z_ho))}}
    th = {k: fit_dist(k, z_in) for k in ("normal", "t", "skewt")}
    res["params"] = {k: _params(k, v) for k, v in th.items()}
    ll = {k: _dist_ll(k, z_ho, v) for k, v in th.items()}
    res["logscore_ho"] = {k: float(np.mean(v)) for k, v in ll.items()}
    res["ls_t_minus_normal"] = dm_test(ll["t"] - ll["normal"])
    res["ls_skewt_minus_t"] = dm_test(ll["skewt"] - ll["t"])
    var = {}
    for k, v in th.items():
        vk = {}
        for p in (0.01, 0.05):
            qlo, qhi = dist_ppf(k, p, v), dist_ppf(k, 1 - p, v)
            hlo, hhi = z_ho < qlo, z_ho > qhi
            vk[f"lo_{p}"] = {**kupiec(hlo, p), **christoffersen(hlo, p), "q": qlo}
            vk[f"hi_{p}"] = {**kupiec(hhi, p), **christoffersen(hhi, p), "q": qhi}
        u = dist_cdf(k, z_ho, v)
        vk["pit_ks_p"] = float(stats.kstest(u, "uniform").pvalue)
        vk["berkowitz_p"] = berkowitz(u)
        # 4-sigma one-bar tail (sigma = the forecast; z in forecast units)
        plo, phi = float(dist_cdf(k, np.array([-4.0]), v)[0]), float(1 - dist_cdf(k, np.array([4.0]), v)[0])
        vk["p4_lo_model"], vk["p4_hi_model"] = plo, phi
        var[k] = vk
    res["var"] = var
    n = len(z_ho)
    for side, x in (("lo", int((z_ho < -4).sum())), ("hi", int((z_ho > 4).sum()))):
        ci = stats.binomtest(x, n).proportion_ci(0.95) if n else None
        res[f"p4_{side}_emp"] = {"x": x, "n": n, "rate": x / n if n else float("nan"), "ci": [ci.low, ci.high] if ci else None}
    # in-sample empirical 4-sigma frequency too
    res["p4_in_emp"] = {"lo": float((z_in < -4).mean()), "hi": float((z_in > 4).mean())}
    gp = {}
    for side in ("lo", "hi"):
        try:
            t = gpd_tail(z_in, side)
        except Exception:  # noqa: BLE001
            continue
        cov = {}
        for p in (0.01, 0.005, 0.001):
            q = gpd_quantile(t, p)
            hits = (-z_ho > q) if side == "lo" else (z_ho > q)
            qt = -dist_ppf("t", p, th["t"]) if side == "lo" else dist_ppf("t", 1 - p, th["t"])
            hits_t = (-z_ho > qt) if side == "lo" else (z_ho > qt)
            cov[str(p)] = {"gpd_q": q, "gpd_rate": float(hits.mean()), "gpd_kupiec_p": kupiec(hits, p)["p"],
                           "t_q": qt, "t_rate": float(hits_t.mean()), "t_kupiec_p": kupiec(hits_t, p)["p"]}
        t["coverage_ho"] = cov
        gp[side] = t
    res["gpd"] = gp
    if wk_in is not None and wk_ho is not None:
        wv = {}
        for sg, zz, ww in (("llm_in_sample", z_in, wk_in), ("clean_holdout", z_ho, wk_ho)):
            for lab, m in (("weekend", ww), ("weekday", ~ww)):
                x = zz[m]
                d = {"n": int(len(x))}
                for p in (0.01, 0.05):
                    qlo, qhi = dist_ppf("t", p, th["t"]), dist_ppf("t", 1 - p, th["t"])
                    d[f"lo_{p}"] = kupiec(x < qlo, p)
                    d[f"hi_{p}"] = kupiec(x > qhi, p)
                d["var_z"] = float(np.var(x)) if len(x) else None
                wv[f"{sg}_{lab}"] = d
        res["weekend_var_t"] = wv
    tq = {}
    for sg, zz in (("llm_in_sample", z_in), ("clean_holdout", z_ho)):
        for side, x in (("long", -zz), ("short", zz)):
            q99 = float(np.quantile(x, 0.99))
            tq[f"{sg}_{side}"] = {"q99": q99, "es99": float(x[x >= q99].mean()), "q999": float(np.quantile(x, 0.999)), "max": float(x.max())}
    nu = res["params"]["t"]["nu"]
    sc = res["params"]["t"]["scale"]
    q = float(stats.t.ppf(0.99, nu)) * math.sqrt((nu - 2) / nu) * sc
    xs = stats.t.ppf(np.linspace(0.99, 1, 20001)[1:-1], nu) * math.sqrt((nu - 2) / nu) * sc
    tq["t_model"] = {"q99": q, "es99": float(xs.mean())}
    res["tail_move"] = tq
    return res


def dist(symbols: list[str], spec_by_grid: dict[str, str]) -> dict:
    """Distribution dossier for the selected spec per grid and for rbc (the incumbent), per asset and pooled crypto."""
    out: dict = {}
    c = cutoff()
    for g, spec in spec_by_grid.items():
        for sp in dict.fromkeys([spec, "rbc"]):
            pool_in, pool_ho = [], []
            for s in symbols:
                f = load_fc(s, g).filter(pl.col(sp).is_finite() & pl.col("ret").is_finite() & (pl.col(sp) > 0))
                zz = (f["ret"] / f[sp].sqrt()).to_numpy()
                wk = f["weekend"].to_numpy().astype(bool) if g != "1d" else (f["event_ts"].dt.weekday() >= 6).to_numpy()
                tin = (f["event_ts"] < c).to_numpy()
                z_in, z_ho = zz[tin], zz[~tin]
                if len(z_in) < 100 or len(z_ho) < 30:
                    continue
                out[f"{s}_{g}_{sp}"] = dist_block(z_in, z_ho, wk[tin], wk[~tin])
                if s in CRYPTO:
                    pool_in.append((z_in, wk[tin]))
                    pool_ho.append((z_ho, wk[~tin]))
            if pool_in:
                out[f"crypto_{g}_{sp}"] = dist_block(np.concatenate([a for a, _ in pool_in]), np.concatenate([a for a, _ in pool_ho]),
                                                     np.concatenate([b for _, b in pool_in]), np.concatenate([b for _, b in pool_ho]))
    (root() / "dist.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def touch_table(nu: float, H: int, n_sim: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """P(min of a driftless H-step random walk with unit-variance t(nu) steps and total variance 1 touches -x), on a grid
    of x, with the Broadie-Glasserman-Kou continuity correction (barrier moved in by 0.5826 step sd)."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    sd = 1 / math.sqrt(H)
    steps = stats.t.rvs(nu, size=(n_sim, H), random_state=rng) * math.sqrt((nu - 2) / nu) * sd
    mins = np.minimum.accumulate(np.cumsum(steps, axis=1), axis=1)[:, -1]
    xs = np.linspace(0.05, 12, 400)
    prob = np.array([np.mean(mins <= -(x - 0.5826 * sd)) for x in xs])
    return xs, prob


def first_passage(symbols: list[str], spec_by_grid: dict[str, str], ks=(2.0, 4.0), hold_h=(1, 4, 24), nu_1h: float = 4.0) -> dict:
    """Empirical touch frequency of an adverse k x sigma_1h move within the next H hours (5m lows for a long, highs for a
    short) at 00/04/../20 UTC origins (H = 24: 00 UTC only), with the reflection-principle benchmark
    2 P(r_H < -k sigma_1h) under a Normal with the horizon spec's variance F_H; plus the R loss of a resting long stop at
    k sigma_1h over 24 h (fill at the stop, or at the 5m open if that bar opens through it)."""
    from scipy import stats

    c = cutoff()
    res: dict = {"nu_1h": nu_1h}
    g_of = {1: "1h", 4: "4h", 24: "1d"}
    tables = {H: touch_table(nu_1h, H) for H in hold_h}
    for s in symbols:
        d5 = load_5m(s)
        t5 = d5["event_ts"].cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
        op, lo5, hi5, cl = (d5[k].to_numpy() for k in ("open", "low", "high", "close"))
        f1 = load_fc(s, "1h").filter(pl.col(spec_by_grid["1h"]).is_finite() & (pl.col("event_ts").dt.hour() % 4 == 0))
        f1 = f1.select("event_ts", pl.col(spec_by_grid["1h"]).alias("F1"))
        for H in (4, 24):
            fh = load_fc(s, g_of[H]).select("event_ts", pl.col(spec_by_grid[g_of[H]]).alias(f"F{H}"))
            f1 = f1.join(fh, on="event_ts", how="left")
        t0 = f1["event_ts"].cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()
        i0 = np.searchsorted(t5, t0)
        ok = (i0 > 0) & (i0 + 288 <= len(t5))
        ok[ok] &= (t5[i0[ok]] == t0[ok]) & (t5[i0[ok] + 287] - t0[ok] == 287 * 300_000_000)
        f1, i0 = f1.filter(pl.Series(ok)), i0[ok]
        if f1.height == 0:
            continue
        idx = i0[:, None] + np.arange(288)[None, :]
        e = cl[i0 - 1][:, None]
        lmin, hmax, lop = np.log(lo5[idx] / e), np.log(hi5[idx] / e), np.log(op[idx] / e)
        sig = np.sqrt(f1["F1"].to_numpy())
        tin = (f1["event_ts"] < c).to_numpy()
        is00 = (f1["event_ts"].dt.hour() == 0).to_numpy()
        FH = {1: sig ** 2, 4: f1["F4"].to_numpy().astype(float), 24: f1["F24"].to_numpy().astype(float)}
        rs: dict = {}
        for k in ks:
            b = k * sig
            for H in hold_h:
                m = H * 12
                hit_lo = (lmin[:, :m] <= -b[:, None]).any(axis=1)
                hit_hi = (hmax[:, :m] >= b[:, None]).any(axis=1)
                model = 2 * stats.norm.cdf(-b / np.sqrt(FH[H]))
                xs, pt = tables[H]
                model_t = np.interp(b / np.sqrt(FH[H]), xs, pt)
                for sg, msk in (("llm_in_sample", tin), ("clean_holdout", ~tin)):
                    mm = msk & np.isfinite(model) & (is00 if H == 24 else True)
                    rs[f"k{k}_H{H}_{sg}"] = {"n": int(mm.sum()), "long_touch": float(hit_lo[mm].mean()) if mm.any() else None,
                                             "short_touch": float(hit_hi[mm].mean()) if mm.any() else None,
                                             "model_normal_reflection": float(np.mean(np.minimum(model[mm], 1.0))) if mm.any() else None,
                                             "model_t_sim": float(np.mean(model_t[mm])) if mm.any() else None}
            hitmat = lmin <= -b[:, None]
            any_hit = hitmat.any(axis=1)
            first = hitmat.argmax(axis=1)
            fill = np.minimum(-b, lop[np.arange(len(first)), first])
            rloss = np.where(any_hit, -fill / b, np.nan)
            rs[f"k{k}_stop24h"] = {}
            for sg, msk in (("llm_in_sample", tin), ("clean_holdout", ~tin)):
                x = rloss[msk & any_hit]
                rs[f"k{k}_stop24h"][sg] = {"n": int(msk.sum()), "n_hit": int(len(x)),
                                           "mean_R": float(x.mean()) if len(x) else None,
                                           "p99_R": float(np.quantile(x, 0.99)) if len(x) > 20 else None,
                                           "max_R": float(x.max()) if len(x) else None,
                                           "share_gap": float(np.mean(x > 1.0 + 1e-9)) if len(x) else None}
        res[s] = rs
    (root() / "first_passage.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ------------------------------------------------------------------ report tables and mechanical verdicts


def _f(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{d}f}"


def tables() -> str:
    """Markdown tables from eval.json / dist.json / first_passage.json (written to data/research/ws04/tables.md)."""
    ev = json.loads((root() / "eval.json").read_text())
    L: list[str] = []
    for g in GRIDS:
        for seg in ("llm_in_sample", "clean_holdout", "fresh"):
            b = ev["pooled"].get(f"crypto_{g}_qlike", {}).get(seg, {})
            if "rows" not in b:
                continue
            L.append(f"\n**Pooled crypto, {g}, {seg}** ({b['n']} origins; QLIKE; ratio and DM-HLN t vs rbc; MCS 90% Tmax)\n")
            L.append("| rank | spec | QLIKE | ratio vs rbc | DM t vs rbc | DM t vs best | in MCS | MCS p |")
            L.append("|---|---|---|---|---|---|---|---|")
            for i, r in enumerate(b["rows"], 1):
                L.append(f"| {i} | {r['spec']} | {_f(r['loss'], 4)} | {_f(r['ratio_base'])} | {_f(r['dm_base_t'], 1)} | {_f(r['dm_best_t'], 1)} | {'**yes**' if r['in_mcs'] else 'no'} | {_f(r['mcs_p'], 3)} |")
    L.append("\n**MSE (level) MCS, pooled crypto**\n")
    L.append("| grid | segment | best | MCS members |")
    L.append("|---|---|---|---|")
    for g in GRIDS:
        for seg in ("llm_in_sample", "clean_holdout"):
            b = ev["pooled"].get(f"crypto_{g}_mse", {}).get(seg, {})
            if "mcs" in b:
                L.append(f"| {g} | {seg} | {b['best']} | {', '.join(b['mcs'])} |")
    L.append("\n**Per-asset QLIKE MCS (90%)**\n")
    L.append("| asset | grid | segment | n | best | size of MCS | MCS members (first 8) | rbc in MCS | pick in MCS |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    sel = ev.get("selection", {})
    for k, v in ev["asset"].items():
        s_, g = k.rsplit("_", 1)
        for seg, b in v.items():
            if "mcs" not in b:
                continue
            pk = sel.get(g, {}).get("pick")
            L.append(f"| {s_} | {g} | {seg} | {b['n']} | {b['best']} | {len(b['mcs'])} | {', '.join(b['mcs'][:8])} | {'rbc' in b['mcs']} | {pk in b['mcs'] if pk else 'n/a'} |")
    L.append("\n**Key pairs, pooled crypto (QLIKE difference a - b, relative to b, DM-HLN t)**\n")
    keys = ["rbc_how-rbc", "loghar_c-loghar", "loghar_cr-loghar_c", "loghar_cny-loghar_c", "loghar_c-rbc", "loghar_c-rbc_how",
            "gjr_t-loghar_c", "garch_t-loghar_c", "gjr_tc-loghar_c", "gjr_tc-gjr_t", "harq-har", "harj-har", "loghar-har",
            "garch_t-garch_n", "garch_st-garch_t", "loghar_c-rbc1h_flat", "rbc_how-rbc1h_flat", "rbc1h_flat-rbc"]
    L.append("| pair | " + " | ".join(f"{g} {sg}" for g in GRIDS for sg in ("in", "ho", "fresh")) + " |")
    L.append("|---|" + "---|" * 9)
    for kk in keys:
        cells_ = []
        for g in GRIDS:
            for sg in ("llm_in_sample", "clean_holdout", "fresh"):
                d = ev["pairs"].get(f"crypto_{g}", {}).get(sg, {}).get(kk, {})
                cells_.append(f"{100 * d['rel']:+.1f}% ({_f(d.get('t_hln'), 1)})" if d and "rel" in d and d.get("n", 0) >= 20 else "n/a")
        L.append(f"| {kk} | " + " | ".join(cells_) + " |")
    L.append("\n**Weekend / weekday and session splits, pooled crypto (QLIKE diff rel., DM-HLN t)**\n")
    L.append("| grid | split | n | rbc_how-rbc | loghar_c-loghar | loghar_c-rbc | loghar_c-rbc_how |")
    L.append("|---|---|---|---|---|---|---|")
    for g in GRIDS:
        for k, d in ev["splits"].get(f"crypto_{g}", {}).items():
            row = [f"{100 * d[x]['rel']:+.1f}% ({_f(d[x].get('t_hln'), 1)})" if d.get(x) and "rel" in d[x] else "n/a"
                   for x in ("rbc_how-rbc", "loghar_c-loghar", "loghar_c-rbc", "loghar_c-rbc_how")]
            L.append(f"| {g} | {k} | {d['n']} | " + " | ".join(row) + " |")
    L.append("\n**Release windows, pooled crypto**\n")
    L.append("| grid / segment | n windows | loghar_cr-loghar_c | loghar_c-rbc | rbc_how-rbc |")
    L.append("|---|---|---|---|---|")
    for k, d in ev["release"].items():
        row = [f"{_f(d[x].get('mean'), 3)} ({100 * d[x]['rel']:+.0f}%, t {_f(d[x].get('t_hln'), 1)})" if d.get(x) and "rel" in d[x] else "n/a"
               for x in ("loghar_cr-loghar_c", "loghar_c-rbc", "rbc_how-rbc")]
        L.append(f"| {k} | {d['n']} | " + " | ".join(row) + " |")
    L.append("\n**Metals (per asset) key pairs and weekend split**\n")
    L.append("| asset grid | segment | rbc_how-rbc | loghar_c-loghar | loghar_c-rbc | gjr_t-loghar_c |")
    L.append("|---|---|---|---|---|---|")
    for s_ in METALS:
        for g in GRIDS:
            for sg, d in ev["pairs"].get(f"{s_}_{g}", {}).items():
                row = [f"{100 * d[x]['rel']:+.0f}% ({_f(d[x].get('t_hln'), 1)})" if d.get(x) and "rel" in d[x] else "n/a"
                       for x in ("rbc_how-rbc", "loghar_c-loghar", "loghar_c-rbc", "gjr_t-loghar_c")]
                L.append(f"| {s_} {g} | {sg} | " + " | ".join(row) + " |")
    L.append("\n**Mincer-Zarnowitz in logs (log RV = a + b log F), slope b and HAC Wald p of (a, b) = (0, 1), R2**\n")
    L.append("| asset grid | segment | " + " | ".join(("rbc", "rbc_how", "loghar", "loghar_c", "gjr_t", "garch_t")) + " |")
    L.append("|---|---|" + "---|" * 6)
    for k, v in ev["mz"].items():
        for sg, d in v.items():
            row = [f"b {_f(d[x].get('b'), 2)}, p {_f(d[x].get('wald_p'), 2)}, R2 {_f(d[x].get('r2'), 2)}" if x in d and "b" in d[x] else "n/a"
                   for x in ("rbc", "rbc_how", "loghar", "loghar_c", "gjr_t", "garch_t")]
            L.append(f"| {k} | {sg} | " + " | ".join(row) + " |")
    dp = root() / "dist.json"
    if dp.exists():
        dd = json.loads(dp.read_text())
        L.append("\n**Innovation distributions (z = r / sqrt(F)), fitted on llm_in_sample, scored on clean_holdout**\n")
        L.append("| key | n in / ho | var z in / ho | exkurt in | t nu | skew-t nu, lambda | logscore N / t / skew-t (ho) | t-N DM t | skewt-t DM t | PIT KS p (t) | Berkowitz p (t) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, d in dd.items():
            pr, ls = d["params"], d["logscore_ho"]
            L.append(f"| {k} | {d['n_in']} / {d['n_ho']} | {_f(d['moments_in']['var'], 2)} / {_f(d['moments_ho']['var'], 2)} | {_f(d['moments_in']['exkurt'], 1)} | "
                     f"{_f(pr['t']['nu'], 2)} | {_f(pr['skewt']['nu'], 2)}, {_f(pr['skewt']['lam'], 3)} | {_f(ls['normal'])} / {_f(ls['t'])} / {_f(ls['skewt'])} | "
                     f"{_f(d['ls_t_minus_normal']['t_hln'], 1)} | {_f(d['ls_skewt_minus_t']['t_hln'], 1)} | {_f(d['var']['t']['pit_ks_p'], 3)} | {_f(d['var']['t']['berkowitz_p'], 3)} |")
        L.append("\n**VaR backtests on clean_holdout (exceedance rate; Kupiec p; Christoffersen CC p)**\n")
        L.append("| key | dist | lo 1% | hi 1% | lo 5% | hi 5% |")
        L.append("|---|---|---|---|---|---|")
        for k, d in dd.items():
            if not k.startswith("crypto_") and "_1h_" not in k:
                pass
            for dist_ in ("normal", "t", "skewt"):
                v = d["var"][dist_]
                row = [f"{100 * v[x]['rate']:.2f}% (p {_f(v[x]['p'], 2)}, cc {_f(v[x]['p_cc'], 2)})" for x in ("lo_0.01", "hi_0.01", "lo_0.05", "hi_0.05")]
                L.append(f"| {k} | {dist_} | " + " | ".join(row) + " |")
        L.append("\n**4-sigma one-bar tail (clean_holdout): empirical vs model**\n")
        L.append("| key | n | emp P(z<-4) [95% CI] | t model | Normal model | emp P(z>4) [95% CI] | t model | in-sample emp lo / hi |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, d in dd.items():
            lo, hi = d["p4_lo_emp"], d["p4_hi_emp"]
            L.append(f"| {k} | {lo['n']} | {100 * lo['rate']:.3f}% [{100 * lo['ci'][0]:.3f}, {100 * lo['ci'][1]:.3f}] | {100 * d['var']['t']['p4_lo_model']:.3f}% | {100 * d['var']['normal']['p4_lo_model']:.4f}% | "
                     f"{100 * hi['rate']:.3f}% [{100 * hi['ci'][0]:.3f}, {100 * hi['ci'][1]:.3f}] | {100 * d['var']['t']['p4_hi_model']:.3f}% | {100 * d['p4_in_emp']['lo']:.3f}% / {100 * d['p4_in_emp']['hi']:.3f}% |")
        L.append("\n**GPD tails (POT beyond the 5% quantile, fit in-sample) vs t: holdout exceedance rates**\n")
        L.append("| key | side | xi | 1%: GPD / t | 0.5%: GPD / t | 0.1%: GPD / t |")
        L.append("|---|---|---|---|---|---|")
        for k, d in dd.items():
            if not k.startswith("crypto_"):
                continue
            for side, t in d.get("gpd", {}).items():
                cv = t["coverage_ho"]
                row = [f"{100 * cv[p_]['gpd_rate']:.2f}% / {100 * cv[p_]['t_rate']:.2f}%" for p_ in ("0.01", "0.005", "0.001")]
                L.append(f"| {k} | {side} | {_f(t['xi'], 3)} | " + " | ".join(row) + " |")
        L.append("\n**Weekend vs weekday 1% VaR coverage of the t (pooled crypto)**\n")
        L.append("| key | split | n | var z | lo 1% rate (Kupiec p) | hi 1% rate (Kupiec p) |")
        L.append("|---|---|---|---|---|---|")
        for k, d in dd.items():
            if not k.startswith("crypto_"):
                continue
            for sp_, w in d.get("weekend_var_t", {}).items():
                L.append(f"| {k} | {sp_} | {w['n']} | {_f(w['var_z'], 2)} | {100 * w['lo_0.01']['rate']:.2f}% ({_f(w['lo_0.01']['p'], 2)}) | {100 * w['hi_0.01']['rate']:.2f}% ({_f(w['hi_0.01']['p'], 2)}) |")
        L.append("\n**Adverse standardised move: 99% quantile and ES (z units; one bar)**\n")
        L.append("| key | segment side | q99 | ES99 | q99.9 | max |")
        L.append("|---|---|---|---|---|---|")
        for k, d in dd.items():
            if not k.startswith("crypto_"):
                continue
            for sp_, w in d["tail_move"].items():
                L.append(f"| {k} | {sp_} | {_f(w['q99'], 2)} | {_f(w['es99'], 2)} | {_f(w.get('q999'), 2)} | {_f(w.get('max'), 1)} |")
    fp = root() / "first_passage.json"
    if fp.exists():
        f = json.loads(fp.read_text())
        L.append(f"\n**First passage: P(touch -k sigma_1h within H hours) for a long, k sigma_1h for a short (t nu_1h = {_f(f.get('nu_1h'), 2)})**\n")
        L.append("| asset | k | H | segment | n | long touch | short touch | t-sim model | Normal reflection |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s_, v in f.items():
            if not isinstance(v, dict):
                continue
            for kk, d in v.items():
                if "_H" not in kk:
                    continue
                k_, h_, sg = kk.split("_", 2)
                L.append(f"| {s_} | {k_[1:]} | {h_[1:]} | {sg} | {d['n']} | {_f(d['long_touch'])} | {_f(d['short_touch'])} | {_f(d['model_t_sim'])} | {_f(d['model_normal_reflection'])} |")
        L.append("\n**Resting long stop at k sigma_1h over 24 h: loss in R when hit (1.0 = exactly at the stop; gap = 5m open beyond the stop)**\n")
        L.append("| asset | k | segment | origins | hits | mean R | p99 R | max R | share gapped |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s_, v in f.items():
            if not isinstance(v, dict):
                continue
            for kk, d in v.items():
                if not kk.endswith("stop24h"):
                    continue
                for sg, x in d.items():
                    L.append(f"| {s_} | {kk.split('_')[0][1:]} | {sg} | {x['n']} | {x['n_hit']} | {_f(x['mean_R'], 3)} | {_f(x['p99_R'], 3)} | {_f(x['max_R'], 3)} | {_f(x['share_gap'], 4)} |")
    txt = "\n".join(L) + "\n"
    (root() / "tables.md").write_text(txt, encoding="utf-8")
    return txt


def verdicts() -> dict:
    """Mechanical verdicts of H-WS04-1..7 from the registered success rules (computed, not edited, after evaluation)."""
    ev = json.loads((root() / "eval.json").read_text())
    dd = json.loads((root() / "dist.json").read_text()) if (root() / "dist.json").exists() else {}
    fp = json.loads((root() / "first_passage.json").read_text()) if (root() / "first_passage.json").exists() else {}
    sel = ev.get("selection", {})
    P = lambda g, sg, k: ev["pairs"].get(f"crypto_{g}", {}).get(sg, {}).get(k, {})  # noqa: E731
    V: dict = {}

    def seg_verdicts(parts_in: list[dict], parts_ho: list[dict]) -> tuple[str, str]:
        ins = "supports" if all(d and d["mean"] < 0 and d["t_hln"] < -2 for d in parts_in) else (
            "contradicts" if any(d and d["mean"] > 0 and d["t_hln"] > 2 for d in parts_in) else "inconclusive")
        ho = "supports" if all(d and d["mean"] < 0 for d in parts_ho) else (
            "contradicts" if any(d and d["mean"] > 0 and d["t_hln"] > 2 for d in parts_ho) else "inconclusive")
        return ins, ho

    def m(d):
        return {k: d.get(k) for k in ("mean", "rel", "t_hln", "n")} if d else {}

    # H-WS04-1
    keys = [(g, k) for g in ("1h", "4h") for k in ("rbc_how-rbc", "loghar_c-loghar")]
    ins, ho = seg_verdicts([P(g, "llm_in_sample", k) for g, k in keys], [P(g, "clean_holdout", k) for g, k in keys])
    V["H-WS04-1"] = {"llm_in_sample": (ins, {f"{g}:{k}": m(P(g, "llm_in_sample", k)) for g, k in keys}),
                     "clean_holdout": (ho, {f"{g}:{k}": m(P(g, "clean_holdout", k)) for g, k in keys})}
    # H-WS04-2
    for sg in ("llm_in_sample", "clean_holdout"):
        ok_all, det = True, {}
        for g in GRIDS:
            b = ev["pooled"].get(f"crypto_{g}_qlike", {}).get(sg, {})
            rows = {r["spec"]: r for r in b.get("rows", [])}
            cand = [k for k in ("loghar_c", "loghar_cr") if k in rows]
            if not cand:
                ok_all = False
                continue
            best = min(cand, key=lambda k: rows[k]["loss"])
            in_mcs = best in b["mcs"]
            beats = {}
            for gs in GARCH_SPECS:
                if gs in rows:
                    d = dm_pair_from_rows(ev, g, sg, gs, best)
                    beats[gs] = d
            garch_wins = [k for k, d in beats.items() if d is not None and d < -2]
            det[g] = {"best_clock_har": best, "in_mcs": in_mcs, "garch_t_vs_best": beats, "garch_wins": garch_wins}
            ok_all &= in_mcs and not garch_wins
        V["H-WS04-2"] = V.get("H-WS04-2", {})
        V["H-WS04-2"][sg] = ("supports" if ok_all else "contradicts", det)
    # H-WS04-3
    parts = {sg: [P(g, sg, f"{sel[g]['pick']}-rbc") if sel.get(g, {}).get("pick") != "rbc" else {} for g in GRIDS] for sg in ("llm_in_sample", "clean_holdout")}
    for sg in parts:
        for i, g in enumerate(GRIDS):
            if not parts[sg][i] and sel.get(g, {}).get("pick") not in (None, "rbc"):
                parts[sg][i] = pair_from_eval(ev, g, sg, sel[g]["pick"], "rbc")
    ins, ho = seg_verdicts(parts["llm_in_sample"], parts["clean_holdout"])
    V["H-WS04-3"] = {"llm_in_sample": (ins, {"picks": {g: sel.get(g, {}).get("pick") for g in GRIDS}, **{g: m(d) for g, d in zip(GRIDS, parts["llm_in_sample"])}}),
                     "clean_holdout": (ho, {g: m(d) for g, d in zip(GRIDS, parts["clean_holdout"])})}
    # H-WS04-4
    out4 = {}
    for sg in ("llm_in_sample", "clean_holdout"):
        rel_ok, all_ok, det = True, True, {}
        for g in ("1h", "4h"):
            r = ev["release"].get(f"crypto_{g}_{sg}", {}).get("loghar_cr-loghar_c", {})
            a = P(g, sg, "loghar_cr-loghar_c")
            det[g] = {"release_windows": m(r), "all_windows": m(a)}
            if sg == "llm_in_sample":
                rel_ok &= bool(r) and r["mean"] < 0 and r["t_hln"] < -1.96
            else:
                rel_ok &= bool(r) and r["mean"] < 0
            all_ok &= not (a and a["mean"] > 0 and a["t_hln"] > 2)
        out4[sg] = ("supports" if rel_ok and all_ok else "contradicts" if (not all_ok) else "inconclusive", det)
    V["H-WS04-4"] = out4
    # H-WS04-5 (a) 1d clock log-HAR vs log-HAR (crypto); (b) weekend 1% VaR of the t under the clock spec
    a_in, a_ho = P("1d", "llm_in_sample", "loghar_c-loghar"), P("1d", "clean_holdout", "loghar_c-loghar")
    clock_spec = {g: ("loghar_c" if sel.get(g, {}).get("pick") not in ("loghar_c", "loghar_cr", "loghar_cny", "rbc_how") else sel[g]["pick"]) for g in GRIDS}
    wk = {}
    b_ok = {"llm_in_sample": True, "clean_holdout": True}
    for g in GRIDS:
        d = dd.get(f"crypto_{g}_{clock_spec[g]}", {}).get("weekend_var_t", {})
        for sg in b_ok:
            w = d.get(f"{sg}_weekend")
            if not w:
                b_ok[sg] = False
                continue
            okw = all(w[x]["rate"] <= 0.02 and w[x]["p"] > 0.05 for x in ("lo_0.01", "hi_0.01"))
            wk[f"{g}_{sg}"] = {"spec": clock_spec[g], "lo": w["lo_0.01"], "hi": w["hi_0.01"], "ok": okw}
            b_ok[sg] &= okw
    va = "supports" if (a_in and a_in["mean"] < 0 and a_in["t_hln"] < -2) else ("contradicts" if a_in and a_in["mean"] > 0 and a_in["t_hln"] > 2 else "inconclusive")
    vh = "supports" if (a_ho and a_ho["mean"] < 0) else ("contradicts" if a_ho and a_ho["t_hln"] > 2 else "inconclusive")
    comb = lambda x, y: "supports" if x == "supports" and y else ("contradicts" if x == "contradicts" or not y else "inconclusive")  # noqa: E731
    V["H-WS04-5"] = {"llm_in_sample": (comb(va, b_ok["llm_in_sample"]), {"a": m(a_in), "a_verdict": va, "b_ok": b_ok["llm_in_sample"], "weekend_var": {k: v for k, v in wk.items() if "llm" in k}}),
                     "clean_holdout": (comb(vh, b_ok["clean_holdout"]), {"a": m(a_ho), "a_verdict": vh, "b_ok": b_ok["clean_holdout"], "weekend_var": {k: v for k, v in wk.items() if "clean" in k}})}
    # H-WS04-6 (holdout test by construction: fit in-sample, score holdout)
    ok6, det6, contra6 = True, {}, False
    for g in GRIDS:
        d = dd.get(f"crypto_{g}_{sel.get(g, {}).get('pick')}")
        if not d:
            ok6 = False
            continue
        tn, st = d["ls_t_minus_normal"], d["ls_skewt_minus_t"]
        v = d["var"]["t"]
        var_ok = all(v[x]["p"] > 0.05 and v[x]["p_cc"] > 0.05 for x in ("lo_0.01", "hi_0.01"))
        det6[g] = {"nu": d["params"]["t"]["nu"], "t_minus_normal": m(tn), "skewt_minus_t": m(st), "var_t_1pct": {x: v[x] for x in ("lo_0.01", "hi_0.01")}, "var_ok": var_ok}
        ok6 &= tn["t_hln"] > 2 and abs(st["t_hln"]) < 2 and var_ok
        contra6 |= tn["t_hln"] < -2
    V["H-WS04-6"] = {"clean_holdout": ("supports" if ok6 else ("contradicts" if contra6 else "inconclusive"), det6)}
    # H-WS04-7
    from scipy import stats

    ok7, det7 = True, {}
    for g in GRIDS:
        d = dd.get(f"crypto_{g}_{sel.get(g, {}).get('pick')}")
        if not d:
            ok7 = False
            continue
        for side in ("lo", "hi"):
            e = d[f"p4_{side}_emp"]
            pm = d["var"]["t"][f"p4_{side}_model"]
            lo_, hi_ = stats.binom.ppf(0.025, e["n"], pm) / e["n"], stats.binom.ppf(0.975, e["n"], pm) / e["n"]
            inside = lo_ <= e["rate"] <= hi_
            det7[f"{g}_{side}"] = {"emp": e["rate"], "x": e["x"], "n": e["n"], "model": pm, "interval": [lo_, hi_], "inside": inside}
            ok7 &= inside
    ratios = []
    for s_ in CRYPTO:
        d = fp.get(s_, {}).get("k4.0_H24_clean_holdout")
        if d and d.get("model_t_sim"):
            ratios.append((d["long_touch"] + d["short_touch"]) / 2 / d["model_t_sim"])
    pooled_ratio = float(np.mean(ratios)) if ratios else float("nan")
    det7["first_passage_24h_ratio_mean"] = pooled_ratio
    det7["first_passage_ratios"] = ratios
    ok7 &= 0.67 <= pooled_ratio <= 1.5 if ratios else False
    V["H-WS04-7"] = {"clean_holdout": ("supports" if ok7 else "contradicts", det7)}
    (root() / "verdicts.json").write_text(json.dumps(V, indent=1, default=float))
    return V


def pair_from_eval(ev: dict, g: str, sg: str, a: str, b: str) -> dict:
    """Recompute a pooled pair that was not in the precomputed list."""
    L, specs = pooled(list(CRYPTO), g, qlike)
    m = segments(L["event_ts"])[sg]
    return pair(L.filter(pl.Series(m)), a, b)


def dm_pair_from_rows(ev: dict, g: str, sg: str, a: str, b: str) -> float | None:
    k = f"{a}-{b}"
    d = ev["pairs"].get(f"crypto_{g}", {}).get(sg, {}).get(k)
    if d is None:
        d = pair_from_eval(ev, g, sg, a, b)
    return d.get("t_hln") if d else None


def record_verdicts() -> None:
    """Append the verdicts to research/hypotheses_results.jsonl (run under `governor.py lock results`)."""
    from ai_trader.research.hypotheses import record

    V = verdicts()
    for hid, segs in V.items():
        for sg, (verdict, metrics) in segs.items():
            record(hid, sg, verdict, metrics, note="WS04 vol_spec (specification program 2026-11-27)", data="data/research/ws04/")
    print(json.dumps({h: {sg: v[0] for sg, v in segs.items()} for h, segs in V.items()}, indent=1))


def properties(symbols: list[str]) -> dict:
    """Empirical properties of the targets per asset and grid (dossier section 2): ADF + KPSS on log RV, moments of
    returns, Ljung-Box on returns and squared returns, ACF of log RV, missing share, jump share (1 - BV/RV), and the
    stability of the clock log-HAR coefficients by calendar year (OLS with HAC SEs)."""
    import warnings

    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, adfuller, kpss

    out: dict = {}
    for s_ in symbols:
        for g in GRIDS:
            df = read_grid(s_, g)
            y = df["rv"].to_numpy().astype(float)
            r = df["ret"].to_numpy().astype(float)
            bv = df["bv"].to_numpy().astype(float)
            ok = np.isfinite(y) & (y > 0)
            ly = np.log(y[ok])
            rr = r[np.isfinite(r)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = float(adfuller(ly, maxlag={"1h": 24, "4h": 6, "1d": 7}[g], autolag=None)[1])
                kpss_p = float(kpss(ly, regression="c", nlags="auto")[1])
            lags = {"1h": (1, 24, 168, 720), "4h": (1, 6, 42, 180), "1d": (1, 7, 30, 90)}[g]
            ac = acf(ly, nlags=max(lags), fft=True)
            lb_r = acorr_ljungbox(rr, lags=[20], return_df=True)
            lb_r2 = acorr_ljungbox(rr ** 2, lags=[20], return_df=True)
            zero_rv = float(np.mean(y[np.isfinite(y)] <= 0)) if np.isfinite(y).any() else float("nan")
            jump = 1 - np.nansum(np.minimum(bv, y)) / np.nansum(y)
            out[f"{s_}_{g}"] = {
                "n": int(df.height), "missing_share": float(1 - np.isfinite(y).mean()), "zero_rv_share": zero_rv,
                "logrv_adf_p": adf_p, "logrv_kpss_p": kpss_p, "logrv_acf": {str(k): float(ac[k]) for k in lags},
                "ret_skew": float(stats.skew(rr)), "ret_exkurt": float(stats.kurtosis(rr)),
                "lb20_ret_p": float(lb_r["lb_pvalue"].iloc[0]), "lb20_ret2_p": float(lb_r2["lb_pvalue"].iloc[0]),
                "lb20_ret_stat": float(lb_r["lb_stat"].iloc[0]), "lb20_ret2_stat": float(lb_r2["lb_stat"].iloc[0]),
                "jump_share": float(jump), "logrv_mean": float(ly.mean()), "logrv_sd": float(ly.std()),
            }
            # yearly stability of the clock log-HAR (clock factor estimated within the year; HAC SE)
            if s_ in CRYPTO:
                ts = df["event_ts"]
                L = LAGS[g]
                cell = cells(ts, g)
                yrs = ts.dt.year().to_numpy()
                rows = {}
                for yr in sorted(set(yrs.tolist())):
                    tr = yrs == yr
                    if tr.sum() < (300 if g == "1d" else 2000):
                        continue
                    slow = trailing_mean(y, L[-1])
                    f_cell = clock_factor(y, slow, cell, CELLS[g], tr, smooth=(g == "1h"))
                    yd = y / f_cell[cell]
                    floor = 0.01 * float(np.nanmedian(y[tr]))
                    X = np.column_stack([np.ones(len(y))] + [np.log(np.maximum(trailing_mean(yd, w), floor)) for w in L])
                    ly2 = np.log(np.maximum(yd, floor))
                    m_ = tr & np.isfinite(X).all(axis=1) & np.isfinite(ly2)
                    Xm, ym = X[m_], ly2[m_]
                    b = ols(Xm, ym)
                    e = ym - Xm @ b
                    n = len(ym)
                    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
                    S = (Xm * e[:, None]).T @ (Xm * e[:, None]) / n
                    for k in range(1, lag + 1):
                        G = (Xm[k:] * e[k:, None]).T @ (Xm[:-k] * e[:-k, None]) / n
                        S += (1 - k / (lag + 1)) * (G + G.T)
                    Q = np.linalg.inv(Xm.T @ Xm / n)
                    se = np.sqrt(np.diag(Q @ S @ Q / n))
                    rows[str(yr)] = {"b": [float(x) for x in b], "se": [float(x) for x in se], "n": n,
                                     "persistence": float(b[1:].sum()), "clock_range": [float(f_cell.min()), float(f_cell.max())]}
                out[f"{s_}_{g}"]["loghar_c_by_year"] = rows
    (root() / "properties.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def report(out_name: str = "ws04_2026-11-27.md") -> Path:
    """Assemble research/reports/<out_name> from the prose templates in data/research/ws04/tmp (report_head.txt,
    report_static.txt, report_tail.txt) and the generated tables (tables.md)."""
    t = root() / "tmp"
    parts = [t / "report_head.txt", t / "report_extras.txt", t / "report_static.txt", root() / "tables.md", t / "report_tail.txt"]
    txt = []
    for q in parts:
        if q.exists():
            body = q.read_text(encoding="utf-8")
            if q.name == "tables.md":
                body = "## Result tables (generated by `vol_spec tables`)\n" + body
            txt.append(body.rstrip() + "\n")
    out = load().root / "research" / "reports" / out_name
    out.write_text("\n".join(txt), encoding="utf-8")
    return out


def extras() -> dict:
    """Dossier sections 4, 6 and 7 for the decision: (a) minimum detectable effect (80% power, 5% two-sided) of the key
    pooled DM tests; (b) Ljung-Box Q(20) on z and z^2 per spec (does the spec absorb the clustering?), in-sample, per asset;
    (c) the stop-distance change sigma_pick / sigma_rbc by session / weekend (1h) and by 4h block; (d) the weekend 1%
    VaR of the t for the 1d clock log-HAR (needed by H-WS04-5, reported descriptively after the verdict)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    ev = json.loads((root() / "eval.json").read_text())
    sel = ev["selection"]
    out: dict = {"mde": {}, "ljung_box": {}, "stop_ratio": {}}
    for g in GRIDS:
        for sg in ("llm_in_sample", "clean_holdout", "fresh"):
            for k, d in ev["pairs"].get(f"crypto_{g}", {}).get(sg, {}).items():
                if d and d.get("t_hln") and math.isfinite(d["t_hln"]) and d["t_hln"] != 0:
                    se = abs(d["mean"] / d["t_hln"])
                    base = d["mean"] / d["rel"] if d.get("rel") else float("nan")
                    out["mde"][f"{g}|{sg}|{k}"] = {"se": se, "mde_abs": 2.8 * se, "mde_rel": 2.8 * se / abs(base) if base else None, "n": d["n"]}
    c = cutoff()
    for s_ in CRYPTO:
        for g in GRIDS:
            f = load_fc(s_, g).filter(pl.col("event_ts") < c)
            row = {}
            for sp in ("rbc", sel[g]["pick"], "loghar_c", "garch_t", "gjr_tc"):
                if sp not in f.columns:
                    continue
                ff = f.filter(pl.col(sp).is_finite() & (pl.col(sp) > 0) & pl.col("ret").is_finite())
                z = (ff["ret"] / ff[sp].sqrt()).to_numpy()
                lb = acorr_ljungbox(z, lags=[20], return_df=True)
                lb2 = acorr_ljungbox(z ** 2, lags=[20], return_df=True)
                row[sp] = {"q20_z": float(lb["lb_stat"].iloc[0]), "q20_z2": float(lb2["lb_stat"].iloc[0]), "p_z2": float(lb2["lb_pvalue"].iloc[0]), "n": len(z)}
            out["ljung_box"][f"{s_}_{g}"] = row
    # stop-distance ratio: sqrt(F_pick / F_rbc), pooled crypto, by calendar split
    for g in ("1h", "4h"):
        pk = sel[g]["pick"]
        parts = []
        for s_ in CRYPTO:
            f = load_fc(s_, g).filter(pl.col(pk).is_finite() & pl.col("rbc").is_finite() & (pl.col("rbc") > 0))
            parts.append(f.select("event_ts", "weekend", (pl.col(pk) / pl.col("rbc")).sqrt().alias("ratio")))
        R = pl.concat(parts).with_columns(pl.col("event_ts").dt.hour().alias("h"))
        res = {"all": R["ratio"].quantile(0.5), "p10": R["ratio"].quantile(0.1), "p90": R["ratio"].quantile(0.9)}
        if g == "1h":
            for sess, (h0, h1) in (("asia", (0, 8)), ("europe", (8, 13)), ("us", (13, 20)), ("late", (20, 24))):
                x = R.filter((pl.col("h") >= h0) & (pl.col("h") < h1) & ~pl.col("weekend"))["ratio"]
                res[f"weekday_{sess}"] = {"median": x.quantile(0.5), "p10": x.quantile(0.1), "p90": x.quantile(0.9)}
            for hh in (6, 14, 15):
                x = R.filter((pl.col("h") == hh) & ~pl.col("weekend"))["ratio"]
                res[f"weekday_{hh:02d}utc"] = {"median": x.quantile(0.5), "p10": x.quantile(0.1), "p90": x.quantile(0.9)}
        else:
            for b in range(0, 24, 4):
                x = R.filter((pl.col("h") == b) & ~pl.col("weekend"))["ratio"]
                res[f"weekday_{b:02d}"] = {"median": x.quantile(0.5), "p10": x.quantile(0.1), "p90": x.quantile(0.9)}
        x = R.filter(pl.col("weekend"))["ratio"]
        res["weekend"] = {"median": x.quantile(0.5), "p10": x.quantile(0.1), "p90": x.quantile(0.9)}
        out["stop_ratio"][g] = res
    # 1d clock spec weekend VaR (extra, descriptive)
    pool_in, pool_ho = [], []
    for s_ in CRYPTO:
        f = load_fc(s_, "1d").filter(pl.col("loghar_c").is_finite() & pl.col("ret").is_finite())
        z = (f["ret"] / f["loghar_c"].sqrt()).to_numpy()
        wk = (f["event_ts"].dt.weekday() >= 6).to_numpy()
        tin = (f["event_ts"] < c).to_numpy()
        pool_in.append((z[tin], wk[tin]))
        pool_ho.append((z[~tin], wk[~tin]))
    d = dist_block(np.concatenate([a for a, _ in pool_in]), np.concatenate([a for a, _ in pool_ho]),
                   np.concatenate([b for _, b in pool_in]), np.concatenate([b for _, b in pool_ho]))
    out["crypto_1d_loghar_c"] = {"weekend_var_t": d["weekend_var_t"], "nu": d["params"]["t"]["nu"], "p4": {k: d[k] for k in ("p4_lo_emp", "p4_hi_emp")}}
    (root() / "extras.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "build", "forecast", "evaluate", "dist", "passage", "tables", "verdicts", "record", "properties", "report", "extras"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--grids", default="1h,4h,1d")
    ap.add_argument("--spec", default="")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    syms = [x for x in a.symbols.split(",") if x] or list(CRYPTO) + list(METALS)
    grids = [x for x in a.grids.split(",") if x]
    t0 = time.time()
    if a.stage == "download":
        print(json.dumps(download(syms), indent=1, default=str))
    elif a.stage == "build":
        print(json.dumps(build(syms), indent=1, default=str))
    elif a.stage == "forecast":
        forecast(syms, grids, a.workers)
    elif a.stage == "evaluate":
        o = evaluate()
        print(json.dumps(o["selection"], indent=1, default=float))
    elif a.stage == "dist":
        sel = json.loads((root() / "eval.json").read_text())["selection"]
        spec = {g: (a.spec or v["pick"]) for g, v in sel.items()}
        o = dist(syms, spec)
        print(len(o))
    elif a.stage == "passage":
        sel = json.loads((root() / "eval.json").read_text())["selection"]
        spec = {g: (a.spec or v["pick"]) for g, v in sel.items()}
        dd = json.loads((root() / "dist.json").read_text())
        nu = dd[f"crypto_1h_{spec['1h']}"]["params"]["t"]["nu"]
        first_passage([s for s in syms if s in CRYPTO], spec, nu_1h=nu)
    elif a.stage == "tables":
        print(len(tables()))
    elif a.stage == "verdicts":
        print(json.dumps({h: {sg: v[0] for sg, v in segs.items()} for h, segs in verdicts().items()}, indent=1))
    elif a.stage == "record":
        record_verdicts()
    elif a.stage == "extras":
        o = extras()
        print(json.dumps({k: v for k, v in o.items() if k != "mde"}, indent=1, default=float)[:6000])
    elif a.stage == "report":
        print(report())
    elif a.stage == "properties":
        o = properties(syms)
        print(json.dumps({k: {kk: v[kk] for kk in ("logrv_adf_p", "logrv_kpss_p", "ret_exkurt", "jump_share")} for k, v in o.items()}, indent=0, default=float))
    print(f"done {a.stage} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
