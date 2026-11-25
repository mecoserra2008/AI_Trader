"""MACRO block (H-MAC3..H-MAC9): point-in-time macro forecasting models.

    PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 uv run python -m ai_trader.research.macro_models m1 [--record]
    ... m2 | m3 | m4 | nowcast | forward

Design frozen in the diary (2026-11-25 00:36 UTC, kind decision) before any fit.

Data. FRED/ALFRED all vintages, pulled with `atx alt fred --series ...` into lake alt_series (source=fred): one row per
(observation, vintage) with known_ts = vintage date + 1 day. `Vintages.asof(sid, T)` returns, per observation, the latest
value with known_ts <= T. When an origin precedes a series' first ALFRED vintage, the first vintage's values are used with
a conservative release-lag rule (`LAG_DAYS`) as their known_ts ("pseudo real time", flagged in the outputs). Unrevised daily
market series (target rates, bills, curve) come from ai_trader.narrative.macro (FRED CSV, known_ts = obs + 1 day).

M1  core PCE m/m (log x100): AR(p), Minnesota VAR(p), CPI/PPI bridge (h = 0), benchmarks RW / 12-month mean.
M2  policy reaction functions (Fed upper target, ECB DFR): monthly ordered logit; benchmarks climatology, persistence,
    market proxy (3m bill minus target midpoint).
M3  P(NBER recession within 12 months): probit, expanding window, labels purged 24 months.
M4  BTC 1d / 5d response to core PCE / CPI surprises vs the model expectation.

Pure functions (fit / forecast / scoring) are tested in tests/test_macro_models.py.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.core.config import load

OUT = "data/research/macro_models"
SEED = 11

# Conservative release lag (days after the end of the observation period) used only for pseudo-real-time rows
# (observations dated before a series' first ALFRED vintage).
LAG_DAYS: dict[str, int] = {
    "PCEPILFE": 32, "PCEPI": 32, "CPILFESL": 16, "CPIAUCSL": 16, "WPSFD4131": 16, "PPIFES": 16, "PPIFGS": 16,
    "AHETPI": 10, "CES0500000003": 10, "UNRATE": 10, "PAYEMS": 10, "SAHMREALTIME": 10,
    "CUSR0000SAH1": 16, "CUSR0000SACL1E": 16, "CUSR0000SASLE": 16, "CUSR0000SAM2": 16, "CUSR0000SETG01": 16,
    "PCU622110622110": 16, "PCU621111621111": 16, "PCU481111481111": 16, "PCU523920523920": 16,
    "NFCI": 5, "ANFCI": 5, "USREC": 365, "GS10": 2, "TB3MS": 2, "BAA10YM": 2, "BAA": 2, "AAA": 2, "FEDFUNDS": 2,
    "NROU": 400, "GDPC1": 30, "A191RL1Q225SBEA": 30, "GDPPOT": 400,
    "CP0000EZ19M086NEST": 17, "00XEFDEZ19M086NEST": 17, "ECBDFR": 1, "DFEDTAR": 1, "PCETRIM12M159SFRBDAL": 32,
}
WEEKLY = {"NFCI", "ANFCI"}
QUARTERLY = {"NROU", "GDPC1", "A191RL1Q225SBEA", "GDPPOT"}
DAILY_LAKE = {"ECBDFR", "DFEDTAR"}


def out_dir() -> Path:
    p = load().root / OUT
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc(d: datetime | date) -> datetime:
    if isinstance(d, datetime):
        return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def month_index(d: date) -> int:
    return d.year * 12 + d.month - 1


def index_month(i: int) -> date:
    return date(i // 12, i % 12 + 1, 1)


def period_end(d: date, sid: str) -> date:
    if sid in WEEKLY or sid in DAILY_LAKE:
        return d + timedelta(days=1)
    months = 3 if sid in QUARTERLY else 1
    k = month_index(d) + months
    return index_month(k)


# ====================================================================== vintages


class Vintages:
    """All FRED vintages from the lake, per series as numpy arrays (obs date ns, value, known_ts ns, pseudo flag)."""

    def __init__(self, sids: list[str] | None = None) -> None:
        root = load().root / "data" / "lake" / "alt_series" / "source=fred"
        df = pl.read_parquet(str(root / "**" / "*.parquet"), hive_partitioning=False,
                             columns=["series_id", "event_ts", "value", "known_ts"])
        if sids is not None:
            df = df.filter(pl.col("series_id").is_in(sids))
        df = df.drop_nulls("value")
        self._s: dict[str, dict[str, np.ndarray]] = {}
        for (sid,), g in df.group_by("series_id"):
            g = g.sort("event_ts", "known_ts")
            obs = g["event_ts"].dt.date().to_numpy()
            kt = g["known_ts"].dt.epoch("s").to_numpy()
            val = g["value"].to_numpy().astype(float)
            first_kt = kt.min()
            # pseudo-real-time copies of the first vintage, known at the release-lag rule (only if earlier than the vintage)
            first = kt == first_kt
            lag = LAG_DAYS.get(str(sid), 32)
            rule = np.array([int(utc(period_end(o, str(sid)) + timedelta(days=lag)).timestamp()) for o in obs[first].astype(object)])
            keep = rule < first_kt
            obs_all = np.concatenate([obs, obs[first][keep]])
            kt_all = np.concatenate([kt, rule[keep]])
            val_all = np.concatenate([val, val[first][keep]])
            pseudo = np.concatenate([np.zeros(len(obs), bool), np.ones(int(keep.sum()), bool)])
            o = np.lexsort((kt_all, obs_all))
            self._s[str(sid)] = {"obs": obs_all[o], "kt": kt_all[o], "val": val_all[o], "pseudo": pseudo[o],
                                 "first_vintage_ts": np.array([first_kt])}

    def has(self, sid: str) -> bool:
        return sid in self._s

    def asof(self, sid: str, t: datetime) -> tuple[np.ndarray, np.ndarray, bool]:
        """(obs dates, values, any_pseudo) as known at t: latest known value per observation."""
        s = self._s[sid]
        ts = int(utc(t).timestamp())
        m = s["kt"] <= ts
        obs, val, ps = s["obs"][m], s["val"][m], s["pseudo"][m]
        if len(obs) == 0:
            return obs, val, False
        # rows are sorted by (obs, kt): the last row of each obs block is the latest known
        last = np.r_[obs[1:] != obs[:-1], True]
        return obs[last].astype(object), val[last], bool(ps[last].any())

    def first_release(self, sid: str) -> dict[date, tuple[float, datetime]]:
        """obs -> (first-release value, its known_ts) using real vintages only."""
        s = self._s[sid]
        out: dict[date, tuple[float, datetime]] = {}
        for o, k, v, p in zip(s["obs"].astype(object), s["kt"], s["val"], s["pseudo"]):
            if p or o in out:
                continue
            out[o] = (float(v), datetime.fromtimestamp(int(k), UTC))
        return out

    def vintage_ts(self, sid: str) -> np.ndarray:
        s = self._s[sid]
        return np.unique(s["kt"][~s["pseudo"]])

    def first_vintage(self, sid: str) -> datetime:
        return datetime.fromtimestamp(int(self._s[sid]["first_vintage_ts"][0]), UTC)


def monthly(obs: np.ndarray, val: np.ndarray) -> dict[int, float]:
    return {month_index(o): float(v) for o, v in zip(obs, val)}


def logdiff(levels: dict[int, float]) -> dict[int, float]:
    return {k: 100.0 * math.log(v / levels[k - 1]) for k, v in levels.items() if k - 1 in levels and v > 0 and levels[k - 1] > 0}


# ====================================================================== pure statistics


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and residuals (x includes the constant column if wanted)."""
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    return b, y - x @ b


def ar_fit(y: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """AR(p) with intercept by OLS. Returns (coef [c, a1..ap], residual sd)."""
    n = len(y)
    if n <= p + 5:
        raise ValueError("series too short")
    x = np.column_stack([np.ones(n - p)] + [y[p - k - 1:n - k - 1] for k in range(p)])
    b, e = ols(y[p:], x)
    return b, float(e.std(ddof=p + 1))


def ar_forecast(y: np.ndarray, coef: np.ndarray, h: int) -> np.ndarray:
    """Iterated point forecasts for steps 1..h."""
    p = len(coef) - 1
    hist = list(y[-p:]) if p else []
    out = []
    for _ in range(h):
        f = coef[0] + sum(coef[k + 1] * hist[-k - 1] for k in range(p))
        out.append(f)
        hist.append(f)
    return np.array(out)


def var_minnesota_fit(Y: np.ndarray, p: int, lam: float | None, theta: float = 0.5) -> np.ndarray:
    """Equation-by-equation Minnesota-prior VAR(p) with intercept (prior mean 0 on every lag coefficient,
    sd = lam / l * (s_i / s_j) * (theta if j != i else 1), intercept diffuse). lam None = OLS.
    Y: (T, k). Returns B with shape (1 + k*p, k) for x_t = [1, y_{t-1}, ..., y_{t-p}]."""
    T, k = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l - 1:T - l - 1] for l in range(p)])
    Yt = Y[p:]
    s = np.array([ar_fit(Y[:, j], min(p, 4))[1] for j in range(k)])
    B = np.zeros((1 + k * p, k))
    for i in range(k):
        if lam is None:
            B[:, i], _ = ols(Yt[:, i], X)
            continue
        prec = np.zeros(1 + k * p)
        for l in range(p):
            for j in range(k):
                sd = lam / (l + 1) * (s[i] / s[j]) * (1.0 if i == j else theta)
                prec[1 + l * k + j] = 1.0 / sd**2
        A = X.T @ X / s[i] ** 2 + np.diag(prec)
        B[:, i] = np.linalg.solve(A, X.T @ Yt[:, i] / s[i] ** 2)
    return B


def var_forecast(Y: np.ndarray, B: np.ndarray, p: int, h: int) -> np.ndarray:
    k = Y.shape[1]
    hist = [Y[-l - 1] for l in range(p)]  # hist[0] = latest
    out = []
    for _ in range(h):
        x = np.concatenate([[1.0]] + hist[:p])
        f = x @ B
        out.append(f)
        hist = [f] + hist[:-1] if p else []
    return np.array(out).reshape(h, k)


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> dict[str, float]:
    """Diebold-Mariano on squared errors, Newey-West HAC with lag h-1 (>= 1 lag kept for robustness) and the
    Harvey-Leybourne-Newbold small-sample correction. One-sided p for H1: model has lower loss."""
    d = e_model**2 - e_bench**2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm": float("nan"), "p_one_sided": float("nan"), "n": n}
    dbar = d.mean()
    L = max(h - 1, 1)
    u = d - dbar
    lrv = u @ u / n
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        lrv += 2 * w * (u[l:] @ u[:-l]) / n
    lrv = max(lrv, 1e-12)
    dm = dbar / math.sqrt(lrv / n)
    hh = max(h, 1)
    corr = math.sqrt((n + 1 - 2 * hh + hh * (hh - 1) / n) / n)
    dm_c = dm * corr
    return {"dm": float(dm_c), "p_one_sided": float(stats.t.cdf(dm_c, df=n - 1)), "n": n}


def t_fit(errors: np.ndarray) -> tuple[float, float]:
    """Student-t (df, scale) fitted to errors with location fixed at 0; df bounded to [3, 200]."""
    e = errors[np.isfinite(errors)]
    df, _, sc = stats.t.fit(e, floc=0.0)
    if df < 3:
        df = 3.0
        sc = float(np.std(e) * math.sqrt((df - 2) / df))
    return float(min(df, 200.0)), float(sc)


def t_interval(mu: float, df: float, sc: float, level: float) -> tuple[float, float]:
    q = stats.t.ppf(0.5 + level / 2, df)
    return mu - q * sc, mu + q * sc


def pit(y: float, mu: float, df: float, sc: float) -> float:
    return float(stats.t.cdf((y - mu) / sc, df))


def rmse(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    return float(math.sqrt(np.mean(e**2))) if len(e) else float("nan")


def brier_multi(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row multi-class Brier score: p (n, K) probabilities, y (n,) class index 0..K-1."""
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return ((p - onehot) ** 2).sum(1)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def block_bootstrap_mean(d: np.ndarray, block: int, nboot: int = 4000, seed: int = SEED) -> tuple[float, float, float]:
    """Moving-block bootstrap of the mean of d: (mean, lo95, hi95)."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2 * block:
        block = max(1, n // 4)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(nboot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(nboot, -1)[:, :n]
    bm = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def fill_gaps(m: dict[int, float], log: bool = True) -> dict[int, float]:
    """Interpolates missing interior months (e.g. Oct 2025, no CPI during the shutdown): geometric for levels."""
    if not m:
        return m
    ks = sorted(m)
    out = dict(m)
    for a, b in zip(ks[:-1], ks[1:]):
        for k in range(a + 1, b):
            w = (k - a) / (b - a)
            out[k] = math.exp((1 - w) * math.log(m[a]) + w * math.log(m[b])) if log else (1 - w) * m[a] + w * m[b]
    return out


# ====================================================================== M1: core PCE

M1_VAR = ["PCEPILFE", "CPILFESL", "WPSFD4131", "AHETPI", "UNRATE"]
BRIDGE_SETS: dict[str, list[str]] = {
    "S1_core_cpi": ["CPILFESL"],
    "S2_core_cpi_ppi": ["CPILFESL", "WPSFD4131"],
    "S3_cpi_components": ["CUSR0000SACL1E", "CUSR0000SAH1", "CUSR0000SAM2", "CUSR0000SETG01", "CUSR0000SASLE"],
    # PPI portfolio management (PCU523920523920) was registered but FRED's series ends 2022-12: dropped (data gap).
    "S4_cpi_ppi_components": ["CUSR0000SACL1E", "CUSR0000SAH1", "CUSR0000SAM2", "CUSR0000SETG01", "CUSR0000SASLE",
                              "PCU622110622110", "PCU621111621111", "PCU481111481111"],
}
M1_EST_START = month_index(date(1985, 1, 1))     # post-Volcker estimation start (fixed before any fit)
BRIDGE_EST_START = month_index(date(1995, 1, 1))
VAL = (month_index(date(2003, 1, 1)), month_index(date(2014, 12, 1)))
BRIDGE_VAL = (month_index(date(2010, 1, 1)), month_index(date(2014, 12, 1)))
TEST_START = month_index(date(2015, 1, 1))
COVID = (month_index(date(2020, 3, 1)), month_index(date(2021, 12, 1)))
H = 3
AR_P = list(range(1, 13))
VAR_P = list(range(1, 7))
VAR_LAM: list[float | None] = [0.05, 0.1, 0.2, 0.5, 1.0, None]


def is_covid(k: int) -> bool:
    return COVID[0] <= k <= COVID[1]


@dataclass
class Origin:
    m: int              # month index of the not-yet-released target month (h = 0)
    t: datetime         # forecast origin: 00:00 UTC of the release date of month m


def targets(V: Vintages, sid: str) -> tuple[dict[int, float], dict[int, float], dict[int, datetime]]:
    """First-release log m/m x100 and simple y/y % per month, plus the release time (known_ts - 1 day)."""
    s = V._s[sid]
    mm: dict[int, float] = {}
    yy: dict[int, float] = {}
    rel: dict[int, datetime] = {}
    for kt in V.vintage_ts(sid):
        t = datetime.fromtimestamp(int(kt), UTC)
        o, v, _ = V.asof(sid, t)
        lv = monthly(o, v)
        new = [k for k in lv if k not in rel]
        for k in new:
            rel[k] = t - timedelta(days=1)
            if k - 1 in lv:
                mm[k] = 100 * math.log(lv[k] / lv[k - 1])
            if k - 12 in lv:
                yy[k] = 100 * (lv[k] / lv[k - 12] - 1)
    first_v = datetime.fromtimestamp(int(V.vintage_ts(sid)[0]), UTC)
    # observations present in the very first vintage were not released then: drop them as targets
    rel = {k: t for k, t in rel.items() if t > first_v - timedelta(days=1)}
    mm = {k: x for k, x in mm.items() if k in rel}
    yy = {k: x for k, x in yy.items() if k in rel}
    return mm, yy, rel


def series_asof(V: Vintages, sid: str, t: datetime, log: bool = True) -> dict[int, float]:
    o, v, _ = V.asof(sid, t)
    return fill_gaps(monthly(o, v), log=log)


def m1_panel(V: Vintages, t: datetime, last: int) -> np.ndarray:
    """VAR panel (months M1_EST_START..last) as known at t: m/m logs of the first four, change of UNRATE."""
    cols = []
    for sid in M1_VAR:
        lv = series_asof(V, sid, t, log=sid != "UNRATE")
        x = {k: lv[k] - lv[k - 1] for k in lv if k - 1 in lv} if sid == "UNRATE" else logdiff(lv)
        cols.append([x.get(k, np.nan) for k in range(M1_EST_START, last + 1)])
    Y = np.array(cols).T
    return Y


def m1_forecasts(V: Vintages, o: Origin) -> dict[str, np.ndarray]:
    """All AR / VAR configs plus benchmarks: arrays of h = 0..H forecasts of core PCE log m/m for months m..m+H."""
    last = o.m - 1
    Y = m1_panel(V, o.t, last)
    ok = np.isfinite(Y).all(1)
    # keep the longest finite tail
    if not ok[-1]:
        raise ValueError("panel not complete at m-1")
    bad = np.where(~ok)[0]
    Y = Y[bad[-1] + 1:] if len(bad) else Y
    y = Y[:, 0]
    out: dict[str, np.ndarray] = {}
    for p in AR_P:
        b, _ = ar_fit(y, p)
        out[f"AR{p}"] = ar_forecast(y, b, H + 1)
    for p in VAR_P:
        for lam in VAR_LAM:
            B = var_minnesota_fit(Y, p, lam)
            out[f"VAR{p}_l{lam if lam is not None else 'ols'}"] = var_forecast(Y, B, p, H + 1)[:, 0]
    out["RW"] = np.full(H + 1, y[-1])
    out["MEAN12"] = np.full(H + 1, y[-12:].mean())
    return out


def bridge_forecast(V: Vintages, o: Origin, sids: list[str]) -> float | None:
    """h = 0 nowcast of core PCE m/m from month m's component prints (None if a component of month m is not out yet)."""
    pce = logdiff(series_asof(V, "PCEPILFE", o.t))
    comps = [logdiff(series_asof(V, s, o.t)) for s in sids]
    if any(o.m not in c for c in comps) or (o.m - 1) not in pce:
        return None
    rows, ys = [], []
    for k in range(BRIDGE_EST_START, o.m):
        if k in pce and (k - 1) in pce and all(k in c for c in comps):
            rows.append([1.0, pce[k - 1]] + [c[k] for c in comps])
            ys.append(pce[k])
    if len(rows) < 40:
        return None
    b, _ = ols(np.array(ys), np.array(rows))
    x = np.array([1.0, pce[o.m - 1]] + [c[o.m] for c in comps])
    return float(x @ b)


def m1_origins(V: Vintages) -> list[Origin]:
    _, _, rel = targets(V, "PCEPILFE")
    return [Origin(k, t) for k, t in sorted(rel.items()) if k >= VAL[0]]


def implied_yoy(V: Vintages, o: Origin, f: np.ndarray) -> np.ndarray:
    """Simple y/y % for months m..m+H implied by the vintage at the origin plus the forecast m/m path."""
    pce = logdiff(series_asof(V, "PCEPILFE", o.t))
    out = []
    for h in range(H + 1):
        k = o.m + h
        known = sum(pce[j] for j in range(k - 11, o.m) if j < o.m)
        fc = float(np.sum(f[: h + 1]))
        out.append(100 * (math.exp((known + fc) / 100) - 1))
    return np.array(out)


def run_m1(V: Vintages, log=print) -> dict[str, Any]:
    mm, yy, rel = targets(V, "PCEPILFE")
    origins = m1_origins(V)
    recs: list[dict[str, Any]] = []
    for i, o in enumerate(origins):
        try:
            fc = m1_forecasts(V, o)
        except ValueError:
            continue
        br = {name: bridge_forecast(V, o, s) for name, s in BRIDGE_SETS.items()}
        recs.append({"m": o.m, "t": o.t, "fc": fc, "br": br, "yoy_base": {n: implied_yoy(V, o, f) for n, f in fc.items() if n in ("RW",)}})
        if i % 24 == 0:
            log(f"  M1 origin {index_month(o.m)} ({i + 1}/{len(origins)})")
    return {"recs": recs, "mm": mm, "yy": yy, "rel": rel, "_V": V}


def _errors(res: dict[str, Any], getter, h: int, sel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(errors, target months, forecasts) for origins passing sel(m, k)."""
    e, ks, fs = [], [], []
    for r in res["recs"]:
        k = r["m"] + h
        if k not in res["mm"] or not sel(r["m"], k):
            continue
        f = getter(r, h)
        if f is None or not np.isfinite(f):
            continue
        e.append(res["mm"][k] - f)
        ks.append(k)
        fs.append(f)
    return np.array(e), np.array(ks), np.array(fs)


def in_val(m: int, k: int) -> bool:
    return VAL[0] <= m and k <= VAL[1]


def in_bval(m: int, k: int) -> bool:
    return BRIDGE_VAL[0] <= m and k <= BRIDGE_VAL[1]


def in_test(m: int, k: int) -> bool:
    return m >= TEST_START and not is_covid(k)


def in_covid(m: int, k: int) -> bool:
    return m >= TEST_START and is_covid(k)


def in_holdout(m: int, k: int) -> bool:
    from ai_trader.research.evaluate import knowledge_cutoff
    return m >= TEST_START and index_month(k) >= knowledge_cutoff().date()


def calib(e: np.ndarray, df: float, sc: float) -> dict[str, Any]:
    if len(e) == 0:
        return {}
    z = stats.t.cdf(e / sc, df)
    q80 = stats.t.ppf(0.9, df) * sc
    q90 = stats.t.ppf(0.95, df) * sc
    hist = np.histogram(z, bins=10, range=(0, 1))[0].tolist()
    return {"cov80": float(np.mean(np.abs(e) <= q80)), "cov90": float(np.mean(np.abs(e) <= q90)), "pit_hist10": hist,
            "pit_ks_p": float(stats.kstest(z, "uniform").pvalue)}


def _fc_get(n: str):
    return lambda r, h: float(r["fc"][n][h])


def _br_get(name: str, sel: str):
    def g(r, h):
        if h != 0:
            return None
        b = r["br"][name]
        return b if b is not None else float(r["fc"][sel][0])
    return g


def evaluate_m1(res: dict[str, Any]) -> dict[str, Any]:
    names = list(res["recs"][0]["fc"].keys())
    model_names = [n for n in names if n not in ("RW", "MEAN12")]
    val_rmse = {n: [rmse(_errors(res, _fc_get(n), h, in_val)[0]) for h in range(H + 1)] for n in names}
    score = {n: float(np.mean(val_rmse[n][1:])) for n in names}
    sel = min(model_names, key=lambda n: score[n])
    best_ar = min([n for n in model_names if n.startswith("AR")], key=lambda n: score[n])
    best_var = min([n for n in model_names if n.startswith("VAR")], key=lambda n: score[n])
    # bridge selection (h = 0) on 2010-2014 origins; a missing bridge print falls back to the selected model's h = 0
    bval = {b: rmse(_errors(res, _br_get(b, sel), 0, in_bval)[0]) for b in BRIDGE_SETS}
    bsel = min(bval, key=lambda b: bval[b])
    out: dict[str, Any] = {"selected": sel, "best_ar": best_ar, "best_var": best_var, "bridge_selected": bsel,
                           "val_score_mean_h1_3": score,
                           "val_rmse": {n: val_rmse[n] for n in [sel, best_ar, best_var, "RW", "MEAN12"]},
                           "bridge_val_rmse": bval, "horizons": {}}
    tparams: dict[str, Any] = {}
    for h in range(H + 1):
        rows: dict[str, Any] = {}
        cands = {sel: _fc_get(sel), best_ar: _fc_get(best_ar), best_var: _fc_get(best_var), "RW": _fc_get("RW"), "MEAN12": _fc_get("MEAN12")}
        if h == 0:
            for b in BRIDGE_SETS:
                cands["BRIDGE_" + b] = _br_get(b, sel)
        for n, g in cands.items():
            ev, _, _ = _errors(res, g, h, in_val if not n.startswith("BRIDGE") else in_bval)
            df, sc = t_fit(ev) if len(ev) > 10 else (float("nan"), float("nan"))
            et, _, _ = _errors(res, g, h, in_test)
            ec, _, _ = _errors(res, g, h, in_covid)
            eh, _, _ = _errors(res, g, h, in_holdout)
            rows[n] = {"val_rmse": rmse(ev), "test_rmse": rmse(et), "test_mae": float(np.mean(np.abs(et))) if len(et) else None,
                       "test_bias": float(np.mean(et)) if len(et) else None, "n_test": len(et), "covid_rmse": rmse(ec),
                       "n_covid": len(ec), "holdout_rmse": rmse(eh), "n_holdout": len(eh), "t_df": df, "t_scale": sc,
                       **({"calib_test": calib(et, df, sc)} if np.isfinite(sc) else {})}
            tparams[f"{n}|{h}"] = (df, sc)
        bench = min(["RW", "MEAN12"], key=lambda n: rows[n]["test_rmse"])
        e_sel, k_sel, _ = _errors(res, _fc_get(sel), h, in_test)
        e_b, k_b, _ = _errors(res, _fc_get(bench), h, in_test)
        common = np.intersect1d(k_sel, k_b)
        dm = dm_test(e_sel[np.isin(k_sel, common)], e_b[np.isin(k_b, common)], max(h, 1))
        rows["_dm_selected_vs_" + bench] = dm | {"rmse_ratio": rows[sel]["test_rmse"] / rows[bench]["test_rmse"]}
        if h == 0:
            rival = min([sel, "RW"], key=lambda n: rows[n]["test_rmse"])
            e_br, k_br, _ = _errors(res, _br_get(bsel, sel), 0, in_test)
            e_r, k_r, _ = _errors(res, _fc_get(rival), 0, in_test)
            common = np.intersect1d(k_br, k_r)
            dm0 = dm_test(e_br[np.isin(k_br, common)], e_r[np.isin(k_r, common)], 1)
            rows["_dm_bridge_vs_" + rival] = dm0 | {"rmse_ratio": rows["BRIDGE_" + bsel]["test_rmse"] / rows[rival]["test_rmse"]}
            rows["_bridge_fallbacks_test"] = int(sum(1 for r in res["recs"] if r["m"] >= TEST_START and r["br"][bsel] is None))
        out["horizons"][h] = rows
    # y/y error distribution of the nowcast path (bridge at h = 0, selected model after)
    yy_val: dict[int, list[float]] = {h: [] for h in range(H + 1)}
    yy_test: dict[int, list[float]] = {h: [] for h in range(H + 1)}
    for r in res["recs"]:
        path = np.array(r["fc"][sel], dtype=float)
        if r["br"][bsel] is not None:
            path[0] = r["br"][bsel]
        yv = implied_yoy(res["_V"], Origin(r["m"], r["t"]), path)
        for h in range(H + 1):
            k = r["m"] + h
            if k not in res["yy"]:
                continue
            if BRIDGE_VAL[0] <= r["m"] and k <= VAL[1]:
                yy_val[h].append(res["yy"][k] - yv[h])
            elif in_test(r["m"], k):
                yy_test[h].append(res["yy"][k] - yv[h])
    out["yoy"] = {}
    for h in range(H + 1):
        ev, et = np.array(yy_val[h]), np.array(yy_test[h])
        df, sc = t_fit(ev)
        out["yoy"][h] = {"val_rmse": rmse(ev), "test_rmse": rmse(et), "t_df": df, "t_scale": sc, "calib_test": calib(et, df, sc)}
        tparams[f"YOY|{h}"] = (df, sc)
    out["_tparams"] = tparams
    return out


def m1_current(V: Vintages, ev: dict[str, Any], t: datetime, m: int) -> dict[str, Any]:
    """Forecast from origin t for months m..m+H with the selected models and their validation-fitted t intervals."""
    o = Origin(m, t)
    fc = m1_forecasts(V, o)
    sel, bsel = ev["selected"], ev["bridge_selected"]
    tp = ev["_tparams"]
    path = np.array(fc[sel], dtype=float)
    br = bridge_forecast(V, o, BRIDGE_SETS[bsel])
    src0 = "bridge_" + bsel
    if br is not None:
        path[0] = br
    else:
        src0 = sel + " (bridge input missing)"
    yv = implied_yoy(V, o, path)
    rows = []
    for h in range(H + 1):
        key = f"BRIDGE_{bsel}|0" if (h == 0 and br is not None) else f"{sel}|{h}"
        df, sc = tp[key]
        ydf, ysc = tp[f"YOY|{h}"]
        lo80, hi80 = t_interval(path[h], df, sc, 0.8)
        lo90, hi90 = t_interval(path[h], df, sc, 0.9)
        ylo80, yhi80 = t_interval(yv[h], ydf, ysc, 0.8)
        ylo90, yhi90 = t_interval(yv[h], ydf, ysc, 0.9)
        rows.append({"h": h, "month": str(index_month(m + h))[:7], "model": src0 if h == 0 else sel,
                     "mm_log_x100": round(float(path[h]), 4), "mm_simple_pct": round(100 * (math.exp(path[h] / 100) - 1), 4),
                     "mm_80": [round(lo80, 3), round(hi80, 3)], "mm_90": [round(lo90, 3), round(hi90, 3)],
                     "yoy_simple_pct": round(float(yv[h]), 3), "yoy_80": [round(ylo80, 3), round(yhi80, 3)],
                     "yoy_90": [round(ylo90, 3), round(yhi90, 3)], "t_df": round(df, 2), "t_scale": round(sc, 4),
                     "p_mm_ge_0.25": round(float(1 - stats.t.cdf((0.25 - path[h]) / sc, df)), 3),
                     "p_mm_ge_0.35": round(float(1 - stats.t.cdf((0.35 - path[h]) / sc, df)), 3)})
    alts: dict[str, Any] = {n: round(float(fc[n][0]), 4) for n in (sel, ev["best_ar"], ev["best_var"], "RW", "MEAN12")}
    for b, s in BRIDGE_SETS.items():
        x = bridge_forecast(V, o, s)
        alts[f"bridge_{b}"] = round(x, 4) if x is not None else None
    return {"origin": t.isoformat(), "target_month_h0": str(index_month(m))[:7], "path": rows, "h0_alternatives": alts}


# ====================================================================== GDP second estimate


def gdp_revisions(V: Vintages) -> list[dict[str, Any]]:
    """Per quarter: advance and second-estimate annualised real GDP growth (% SAAR, BEA convention) from GDPC1 vintages."""
    vts = V.vintage_ts("GDPC1")
    seen: dict[date, list[tuple[datetime, float]]] = {}
    for kt in vts:
        t = datetime.fromtimestamp(int(kt), UTC)
        o, v, _ = V.asof("GDPC1", t)
        qs = list(o)
        for i in range(max(1, len(qs) - 2), len(qs)):
            q = qs[i]
            lst = seen.setdefault(q, [])
            if len(lst) < 2:
                lst.append((t, 100 * ((v[i] / v[i - 1]) ** 4 - 1)))
    out = []
    first_v = datetime.fromtimestamp(int(vts[0]), UTC)
    for q, lst in sorted(seen.items()):
        if lst[0][0] <= first_v:
            continue
        row = {"quarter": str(q), "advance": lst[0][1], "advance_ts": lst[0][0].isoformat()}
        if len(lst) > 1:
            row |= {"second": lst[1][1], "second_ts": lst[1][0].isoformat(), "revision": lst[1][1] - lst[0][1]}
        out.append(row)
    return out


def gdp_second_forecast(V: Vintages, quarter: date) -> dict[str, Any]:
    """Second estimate = advance + mean revision; Student-t on 1992-2014 revisions (fit), coverage checked on 2015-2025."""
    rows = gdp_revisions(V)
    fit = np.array([r["revision"] for r in rows if "revision" in r and r["quarter"] < "2015-01-01"])
    tst = np.array([r["revision"] for r in rows if "revision" in r and "2015-01-01" <= r["quarter"] < "2026-01-01"
                    and not ("2020-01-01" <= r["quarter"] <= "2021-10-01")])
    mu = float(fit.mean())
    df, sc = t_fit(fit - mu)
    cur = next((r for r in rows if r["quarter"] == str(quarter)), None)
    if cur is None:
        raise ValueError(f"no advance estimate for {quarter}")
    c = calib(tst - mu, df, sc)
    lo80, hi80 = t_interval(cur["advance"] + mu, df, sc, 0.8)
    lo90, hi90 = t_interval(cur["advance"] + mu, df, sc, 0.9)
    return {"quarter": str(quarter), "advance_saar_pct": round(cur["advance"], 3), "advance_known": cur["advance_ts"],
            "second_median": round(cur["advance"] + mu, 3), "80": [round(lo80, 3), round(hi80, 3)], "90": [round(lo90, 3), round(hi90, 3)],
            "revision_mean_fit": round(mu, 3), "revision_sd_fit": round(float(fit.std(ddof=1)), 3), "n_fit": len(fit),
            "t_df": round(df, 2), "t_scale": round(sc, 3), "test_2015_2025_ex_covid": {"n": len(tst), "rmse_vs_advance_plus_mean": rmse(tst - mu), **c}}


# ====================================================================== forward files


def print_probs(mu: float, df: float, sc: float, values: list[float], log_mm: bool = True) -> dict[str, float]:
    """P(published one-decimal print = v) for a forecast of the (log x100) m/m or a y/y in simple %."""
    out = {}
    for v in values:
        lo, hi = v - 0.05, v + 0.05
        if log_mm:  # print is the simple % m/m; convert the bounds to log x100
            lo, hi = 100 * math.log(1 + lo / 100), 100 * math.log(1 + hi / 100)
        out[f"{v:.1f}"] = round(float(stats.t.cdf((hi - mu) / sc, df) - stats.t.cdf((lo - mu) / sc, df)), 3)
    return out


def recalibrated_scale(res: dict[str, Any], getter, h: int, last_month: int) -> tuple[float, float, int]:
    """t (df, scale) re-fitted on every ex-COVID error with a target month <= last_month (validation + test): a forward
    convenience, reported next to the frozen interval, never used to score H-MAC3/4."""
    e, _, _ = _errors(res, getter, h, lambda m, k: m >= VAL[0] and k <= last_month and not is_covid(k))
    df, sc = t_fit(e)
    return df, sc, len(e)


def pre_release(V: Vintages, res: dict[str, Any], ev: dict[str, Any], t: datetime, m: int,
                consensus: dict[str, Any], gdp_quarter: date | None) -> dict[str, Any]:
    cur = m1_current(V, ev, t, m)
    h0 = cur["path"][0]
    sel, bsel = ev["selected"], ev["bridge_selected"]
    tp = ev["_tparams"]
    df, sc = tp[f"BRIDGE_{bsel}|0"]
    ydf, ysc = tp["YOY|0"]
    rdf, rsc, rn = recalibrated_scale(res, _br_get(bsel, sel), 0, m - 1)
    mu = h0["mm_log_x100"]
    doc = {
        "written_ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "forecast_origin": t.isoformat(),
        "data_cutoff_rule": "every input has known_ts <= forecast_origin (ALFRED vintages; known_ts = vintage date + 1 day)",
        "hypotheses": ["H-MAC4", "H-MAC9"],
        "release": {"name": "BEA Personal Income and Outlays, October 2026", "scheduled_utc": "2026-11-25T13:30:00+00:00"},
        "core_pce_oct_2026": {
            "model": h0["model"], "selection": f"bridge {bsel} chosen on 2010-2014 validation origins; frozen before the test",
            "mm_median_simple_pct": h0["mm_simple_pct"], "mm_median_log_x100": mu,
            "mm_80_log_x100": h0["mm_80"], "mm_90_log_x100": h0["mm_90"],
            "mm_print_probabilities": print_probs(mu, df, sc, [0.1, 0.2, 0.3, 0.4, 0.5]),
            "yoy_median_simple_pct": h0["yoy_simple_pct"], "yoy_80": h0["yoy_80"], "yoy_90": h0["yoy_90"],
            "yoy_print_probabilities": print_probs(h0["yoy_simple_pct"], ydf, ysc, [3.0, 3.1, 3.2, 3.3, 3.4], log_mm=False),
            "interval_note": ("Frozen intervals: Student-t fitted to 2010-2014 validation errors. On the 2015-2026 test the "
                              "bridge's 80% interval covered only " + f"{ev['horizons'][0]['BRIDGE_' + bsel]['calib_test']['cov80']:.2f}"
                              + " (too narrow); the recalibrated interval below refits the scale on all ex-COVID errors through "
                              "the origin (forward convenience only)."),
            "mm_80_recalibrated": [round(x, 3) for x in t_interval(mu, rdf, rsc, 0.8)],
            "mm_90_recalibrated": [round(x, 3) for x in t_interval(mu, rdf, rsc, 0.9)],
            "recalibrated_t": {"df": round(rdf, 2), "scale": round(rsc, 4), "n": rn},
            "mm_print_probabilities_recalibrated": print_probs(mu, rdf, rsc, [0.1, 0.2, 0.3, 0.4, 0.5]),
            "alternatives_h0": cur["h0_alternatives"],
        },
        "core_pce_path": cur["path"],
        "consensus_as_given": consensus,
    }
    if gdp_quarter is not None:
        doc["real_gdp_q3_2026_second_estimate"] = gdp_second_forecast(V, gdp_quarter)
    return doc


def build_m1(log=print) -> tuple[Vintages, dict[str, Any], dict[str, Any]]:
    V = Vintages(M1_VAR + sorted({s for v in BRIDGE_SETS.values() for s in v}) + ["GDPC1", "PCEPI"])
    res = run_m1(V, log=log)
    ev = evaluate_m1(res)
    return V, res, ev


CONSENSUS_2026_11_25 = {
    "source": "as given in the user brief (web sources, 2026-11-25); not scraped, not verified point-in-time",
    "core_pce_oct_2026_mm_pct": 0.3, "core_pce_oct_2026_yoy_pct": 3.2, "real_gdp_q3_2026_second_saar_pct": 3.5,
}


def cmd_nowcast(path: str) -> dict[str, Any]:
    V, res, ev = build_m1()
    t = datetime(2026, 11, 25, tzinfo=UTC)
    doc = pre_release(V, res, ev, t, month_index(date(2026, 10, 1)), CONSENSUS_2026_11_25, date(2026, 7, 1))
    p = load().root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"{p} exists: pre-release files are never rewritten")
    p.write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    return doc
