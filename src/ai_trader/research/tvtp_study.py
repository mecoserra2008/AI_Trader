"""TVTP block (H-TVTP1..H-TVTP6c): the user's TVTP MS-VAR on gold / silver and crypto lead-lag, point in time.

    OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.tvtp_study <cmd>
    cmds: m1h | m1d | ame | m1m | crypto | xvenue | record

Design frozen in research/hypotheses.yaml (block TVTP) and the diary (2026-11-26 "TVTP design frozen before results")
before any real-data fit. Model: ai_trader.research.tvtp_msvar (reviewed copy of the user's tvtp_msvar.py).

Point in time. y_t = r_t / sigma_{t|t-1} with the EWMA variance built from returns up to t-1; ect_t and every z_t use data
with known_ts <= bar close t; the design row for y_t uses y_{t-1..t-p}, ect_{t-1}, z_{t-1}; every forecast row comes from
the causal filter with parameters fit on rows < its fold origin (walk_forward). Macro series: data/research/macro
(FRED CSV, known_ts = observation date + 1 day 00:00 UTC). Funding / OI: lake rows with known_ts <= t.
Outputs: data/research/tvtp/<dataset>_*.json / .npz.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import minimize

from ai_trader.core.config import load
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t
from ai_trader.research.tvtp_mc import corr_detector
from ai_trader.research.tvtp_msvar import TVTPMSVAR, cusum, kim_smoother, log_trans, marginal_effects, walk_forward

OUT = "data/research/tvtp"
CLIP = 8.0
EULER = 0.5772156649
COST = {"taker_bp": 5.5, "maker_bp": 2.0, "peg_slip_bp": 1.3}
THETAS = (0.5, 0.6, 0.7, 0.8, 0.9)
CS = (0.0, 0.5, 1.0, 1.5, 2.0)
SPEC_GRID = [(K, p, nu) for K in (2, 3) for p in (1, 2) for nu in (None, 5.0)]
END = datetime(2026, 11, 26, tzinfo=UTC)


def out_dir() -> Path:
    p = load().root / OUT
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(name: str, obj: Any) -> None:
    (out_dir() / name).write_text(json.dumps(obj, indent=1, default=_js), encoding="utf-8")


def _js(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


# ====================================================================== features (pure, tested)


def ewma_var(r: np.ndarray, hl: float) -> np.ndarray:
    """v[t] = sigma^2_{t|t-1}: EWMA of r^2 up to t-1 (v[0..hl] seeded by the mean of the first hl squares)."""
    a = 1 - 0.5 ** (1 / hl)
    v = np.empty(len(r))
    n0 = int(min(len(r), max(hl, 2)))
    v0 = float(np.nanmean(r[:n0] ** 2))
    v[0] = v0
    for t in range(1, len(r)):
        x = r[t - 1]
        v[t] = v[t - 1] if not np.isfinite(x) else (1 - a) * v[t - 1] + a * x * x
    return np.maximum(v, 1e-16)


def rolling_ect(lg: np.ndarray, ls: np.ndarray, w: int | None, min_n: int = 20) -> np.ndarray:
    """ect_t = residual at t of the OLS lg ~ a + b ls over the window ending at t (rolling w rows, or expanding when w is
    None), divided by the residual sd in that window. Uses rows <= t only."""
    T = len(lg)
    x, y = ls - ls[0], lg - lg[0]
    c = lambda v: np.r_[0.0, np.cumsum(v)]  # noqa: E731
    Sx, Sy, Sxx, Syy, Sxy = c(x), c(y), c(x * x), c(y * y), c(x * y)
    out = np.full(T, np.nan)
    for t in range(T):
        lo = 0 if w is None else t - w + 1
        if lo < 0 or t + 1 - lo < min_n:
            continue
        n = t + 1 - lo
        sx, sy = Sx[t + 1] - Sx[lo], Sy[t + 1] - Sy[lo]
        sxx = Sxx[t + 1] - Sxx[lo] - sx * sx / n
        syy = Syy[t + 1] - Syy[lo] - sy * sy / n
        sxy = Sxy[t + 1] - Sxy[lo] - sx * sy / n
        if sxx <= 0:
            continue
        b = sxy / sxx
        a = (sy - b * sx) / n
        rss = max(syy - b * sxy, 1e-18)
        out[t] = (y[t] - a - b * x[t]) / math.sqrt(rss / max(n - 2, 1))
    return out


def trailing_z(x: np.ndarray, w: int) -> np.ndarray:
    s = pl.Series(x)
    m, sd = s.rolling_mean(w, min_samples=w).to_numpy(), s.rolling_std(w, min_samples=w).to_numpy()
    return (x - m) / np.where(sd > 0, sd, np.nan)


def expanding_z(x: np.ndarray, min_n: int = 20) -> np.ndarray:
    """(x_t - mean(x_0..x_t)) / sd(x_0..x_t), NaN-aware, rows <= t only."""
    ok = np.isfinite(x)
    xv = np.where(ok, x, 0.0)
    n = np.cumsum(ok)
    s1, s2 = np.cumsum(xv), np.cumsum(xv * xv)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
        var = (s2 - n * m * m) / np.maximum(n - 1, 1)
        z = (x - m) / np.sqrt(var)
    z[(n < min_n) | ~ok] = np.nan
    return z


def build_design(Y: np.ndarray, Z: np.ndarray | None, Xe: np.ndarray | None, seg: np.ndarray, p: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like TVTPMSVAR.design but segment-aware: keeps rows t whose lags t-1..t-p are in the same segment and whose inputs
    are finite. Returns Yt, X, Zc, rows (original row index of each design row)."""
    T, n = Y.shape
    rows = np.arange(p, T)
    keep = np.ones(len(rows), bool)
    for lag in range(1, p + 1):
        keep &= seg[rows - lag] == seg[rows]
    lags = [Y[rows - lag] for lag in range(1, p + 1)]
    X = np.hstack([np.ones((len(rows), 1))] + lags)
    if Xe is not None:
        X = np.hstack([X, Xe.reshape(T, -1)[rows - 1]])
    Zc = np.ones((len(rows), 1)) if Z is None else np.hstack([np.ones((len(rows), 1)), Z.reshape(T, -1)[rows - 1]])
    keep &= np.isfinite(Y[rows]).all(1) & np.isfinite(X).all(1) & np.isfinite(Zc).all(1)
    return Y[rows][keep], X[keep], Zc[keep], rows[keep]


def dm(d: np.ndarray) -> dict[str, float]:
    m, t, n = newey_west_t(np.asarray(d, float))
    return {"mean": m, "t_hac": t, "n": n}


def deflated_sharpe(x: np.ndarray, sr_trials: np.ndarray, n_trials: float) -> dict[str, float]:
    """Bailey & Lopez de Prado (2014), per-trade Sharpe (copied from trend_zoo_study, no annualisation)."""
    nd = NormalDist()
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 10 or x.std(ddof=1) == 0:
        return {"sr": float("nan"), "dsr": float("nan"), "T": T}
    sr = x.mean() / x.std(ddof=1)
    zz = (x - x.mean()) / x.std(ddof=0)
    skew, kurt = float(np.mean(zz ** 3)), float(np.mean(zz ** 4))
    sr_trials = np.asarray(sr_trials, float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    v = float(np.var(sr_trials, ddof=1)) if len(sr_trials) > 1 else 0.0
    n = max(float(n_trials), 2.0)
    sr0 = math.sqrt(v) * ((1 - EULER) * nd.inv_cdf(1 - 1 / n) + EULER * nd.inv_cdf(1 - 1 / (n * math.e)))
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return {"sr": float(sr), "sr0": float(sr0), "T": T, "psr0": nd.cdf(sr * math.sqrt(T - 1) / den),
            "dsr": nd.cdf((sr - sr0) * math.sqrt(T - 1) / den), "n_trials": n}


# ====================================================================== data


def lake_pair(sym_a: str, sym_b: str, freq: str) -> pl.DataFrame:
    from ai_trader.storage.lake import Lake
    L = Lake()
    out = None
    for s, tag in ((sym_a, "a"), (sym_b, "b")):
        d = L.bars(s, freq, END).select("event_ts", pl.col("close").alias(f"c_{tag}"))
        out = d if out is None else out.join(d, on="event_ts", how="inner")
    assert out is not None
    step = {"1h": timedelta(hours=1), "1d": timedelta(days=1)}[freq]
    return out.sort("event_ts").with_columns((pl.col("event_ts") + step).alias("ts"))


def macro_frame(sids: list[str]) -> pl.DataFrame:
    """d5 change (log for DTWEXBGS) of each series, expanding z-scored over its full history, with known_ts."""
    from ai_trader.narrative.macro import read_asof
    out = None
    for sid in sids:
        d = read_asof(sid, END).drop_nulls("value").sort("obs_date")
        v = d["value"].to_numpy().astype(float)
        if sid == "DTWEXBGS":
            v = np.log(v)
        d5 = np.r_[np.full(5, np.nan), v[5:] - v[:-5]]
        z = expanding_z(d5, 250)
        g = pl.DataFrame({"known_ts": d["known_ts"], f"d5_{sid}": z}).drop_nulls().sort("known_ts")
        g = g.with_columns(pl.col("known_ts").dt.cast_time_unit("us"))
        out = g if out is None else out.join_asof(g, on="known_ts", strategy="backward")
    assert out is not None
    return out


def attach_asof(df: pl.DataFrame, right: pl.DataFrame, on_left: str = "ts", on_right: str = "known_ts") -> pl.DataFrame:
    left = df.with_columns(pl.col(on_left).dt.cast_time_unit("us"))
    right = right.with_columns(pl.col(on_right).dt.cast_time_unit("us")).sort(on_right)
    return left.sort(on_left).join_asof(right, left_on=on_left, right_on=on_right, strategy="backward").drop(on_right)


def metals_dataset(freq: str, macro: bool) -> dict[str, Any]:
    """M1h / M1h-macro / M1d (registered definitions)."""
    df = lake_pair("XAUUSDT", "XAGUSDT", freq)
    if freq == "1h":
        hl, burn, ect_w, rv_w = 24, 168, 336, 336
    else:
        hl, burn, ect_w, rv_w = 10, 20, None, None
    lg, ls = np.log(df["c_a"].to_numpy()), np.log(df["c_b"].to_numpy())
    r = np.c_[np.r_[np.nan, np.diff(lg)], np.r_[np.nan, np.diff(ls)]]
    r[0] = 0.0
    v = np.c_[ewma_var(r[:, 0], hl), ewma_var(r[:, 1], hl)]
    sig = np.sqrt(v)
    yraw = r / sig
    ect = rolling_ect(lg, ls, ect_w)
    # rv state: gold log vol for t+1 known at t (includes r_t)
    a = 1 - 0.5 ** (1 / hl)
    lv_next = 0.5 * np.log((1 - a) * v[:, 0] + a * r[:, 0] ** 2)
    rv = trailing_z(lv_next, rv_w) if rv_w else expanding_z(lv_next, 20)
    zc = {"ect": ect, "rv": rv}
    if macro or freq == "1d":
        mf = macro_frame(["DFII10", "DTWEXBGS", "T10YIE"])
        j = attach_asof(df.select("ts"), mf)
        for c in ("d5_DFII10", "d5_DTWEXBGS", "d5_T10YIE"):
            zc[c] = j[c].to_numpy().astype(float)
    Z = np.column_stack(list(zc.values()))
    T = len(df)
    valid = np.arange(T) >= burn
    seg = np.cumsum(~valid)                                  # burn-in rows get their own segment ids
    ts = df["ts"].to_numpy()
    name = {"1h": "m1h_macro" if macro else "m1h", "1d": "m1d"}[freq]
    return dict(name=name, ts=ts, y=np.clip(yraw, -CLIP, CLIP), yraw=yraw, sig=sig, r=r, Z=Z, z_names=list(zc),
                Xe=ect[:, None], seg=np.where(valid, 0, seg), valid=valid, assets=["XAUUSDT", "XAGUSDT"],
                half_spread_bp=np.array([0.026, 2.09]) / 2, lat_r=None, freq=freq)


def flow_1m(symbols: list[str], start: str = "2026-11-07", end: str = "2026-11-26") -> dict[str, pl.DataFrame]:
    """1m bars from flow_3s: close mid = last mid in the minute (>= 15 of 20 candles present), q = mid of the first 3 s
    candle of the next minute (the price about 3 s after the close: the latency fill), spread_q = its spread."""
    root = load().root / "data/lake/flow_3s/venue=bybit/category=linear"
    files = [f for f in sorted(glob.glob(str(root / "date=*" / "*.parquet")))
             if start <= Path(f).parent.name[5:] < end]
    lf = (pl.scan_parquet(files, hive_partitioning=False).filter(pl.col("symbol").is_in(symbols))
          .select("symbol", "event_ts", "mid", "close", "spread_bps", "known_ts"))
    d = lf.collect().sort("symbol", "event_ts", "known_ts").unique(["symbol", "event_ts"], keep="last")
    d = d.with_columns(pl.coalesce("mid", "close").alias("px"), pl.col("event_ts").dt.truncate("1m").alias("m"))
    out = {}
    for (s,), g in d.group_by("symbol"):
        g = g.sort("event_ts")
        bars = g.group_by("m").agg(pl.col("px").drop_nulls().last().alias("px"), pl.len().alias("n")).filter(pl.col("n") >= 15)
        first = (g.filter(pl.col("event_ts") == pl.col("m")).select(pl.col("m").alias("m_prev_close"),
                 pl.col("px").alias("q"), pl.col("spread_bps").alias("spread_q")))
        bars = bars.with_columns((pl.col("m") + timedelta(minutes=1)).alias("ts"))
        bars = bars.join(first, left_on="ts", right_on="m_prev_close", how="left")
        out[s] = bars.sort("ts").select("ts", "px", "q", "spread_q")
    return out


def lake_series(table: str, symbol: str, value: str, filters: dict | None = None) -> pl.DataFrame:
    from ai_trader.storage.lake import Lake
    d = Lake().read_asof(table, END, symbols=[symbol], start=datetime(2026, 8, 1, tzinfo=UTC), filters=filters)
    return d.select("known_ts", "event_ts", value).sort("known_ts")


def pair_1m_dataset(lead: str, lag: str, kind: str) -> dict[str, Any]:
    """kind 'crypto' (H-TVTP6a/b: z = BTC rv, alt funding, alt OI 1h change) or 'metals' (H-TVTP3b: ect, rv)."""
    bars = flow_1m([lead, lag])
    a, b = bars[lead], bars[lag]
    df = a.join(b, on="ts", how="full", coalesce=True, suffix="_b").sort("ts")
    full = pl.DataFrame({"ts": pl.datetime_range(df["ts"].min(), df["ts"].max(), "1m", eager=True,
                                                 time_unit=df["ts"].dtype.time_unit, time_zone="UTC")})
    df = full.join(df, on="ts", how="left")
    pa, pb = df["px"].to_numpy().astype(float), df["px_b"].to_numpy().astype(float)
    ok = np.isfinite(pa) & np.isfinite(pb)
    seg = np.cumsum(~ok)                                     # a missing minute breaks the segment
    la, lb = np.log(pa), np.log(pb)
    r = np.c_[np.r_[np.nan, np.diff(la)], np.r_[np.nan, np.diff(lb)]]
    r[~ok] = np.nan
    r[np.r_[True, ~ok[:-1]]] = np.nan                        # first row after a gap has no return
    hl = 60
    v = np.c_[ewma_var(r[:, 0], hl), ewma_var(r[:, 1], hl)]
    sig = np.sqrt(v)
    yraw = r / sig
    qa, qb = df["q"].to_numpy().astype(float), df["q_b"].to_numpy().astype(float)
    lat_r = np.c_[np.r_[np.nan, np.diff(np.log(qa))], np.r_[np.nan, np.diff(np.log(qb))]]
    lat_r[np.r_[True, seg[1:] != seg[:-1]]] = np.nan
    spread = np.c_[df["spread_q"].to_numpy().astype(float), df["spread_q_b"].to_numpy().astype(float)]
    aa = 1 - 0.5 ** (1 / hl)
    rr = np.nan_to_num(r)
    lv_next = 0.5 * np.log((1 - aa) * v + aa * rr ** 2)
    ts = df["ts"]
    if kind == "crypto":
        rv = trailing_z(lv_next[:, 0], 1440)
        fu = lake_series("funding", lag, "rate", {"venue": "bybit"})
        fu = fu.with_columns(pl.Series("fz", expanding_z(fu["rate"].to_numpy() * 1e4, 10)))
        oi = lake_series("open_interest", lag, "open_interest", {"venue": "bybit", "interval": "1h"})
        if oi.height < 100:                                  # alts carry only 4h OI in the lake (deviation, diary)
            oi = lake_series("open_interest", lag, "open_interest", {"venue": "bybit", "interval": "4h"})
        oiv = np.log(oi["open_interest"].to_numpy())
        oi = oi.with_columns(pl.Series("oiz", expanding_z(np.r_[np.nan, np.diff(oiv)], 24)))
        j = attach_asof(pl.DataFrame({"ts": ts}), fu.select("known_ts", "fz"))
        j = attach_asof(j, oi.select("known_ts", "oiz"))
        zc = {"btc_rv": rv, "funding": j["fz"].to_numpy().astype(float), "oi_chg": j["oiz"].to_numpy().astype(float)}
        Xe = None
    else:
        ect = rolling_ect(la, lb, 1440) if np.isfinite(la).all() else _ect_gappy(la, lb, 1440)
        zc = {"ect": ect, "rv": trailing_z(lv_next[:, 0], 1440)}
        Xe = ect[:, None]
    burn = 240                                               # burn-in at the very start only (EWMA carries over gaps)
    valid = ok & (np.arange(len(df)) >= burn)
    Z = np.column_stack(list(zc.values()))
    return dict(name=f"{kind}_{lag}", ts=ts.to_numpy(), y=np.clip(yraw, -CLIP, CLIP), yraw=yraw, sig=sig, r=r, Z=Z,
                z_names=list(zc), Xe=Xe, seg=np.where(valid, seg, -1 - np.arange(len(df))), valid=valid,
                assets=[lead, lag], spread_bp=spread, lat_r=lat_r, freq="1m")


def _ect_gappy(la: np.ndarray, lb: np.ndarray, w: int) -> np.ndarray:
    """Rolling ECT over the last w valid rows (gaps skipped)."""
    ok = np.isfinite(la) & np.isfinite(lb)
    out = np.full(len(la), np.nan)
    idx = np.flatnonzero(ok)
    e = rolling_ect(la[idx], lb[idx], w)
    out[idx] = e
    return out


def xvenue_dataset() -> dict[str, Any]:
    """BTC Bybit vs Binance mids from the xvenue journal on a 3-minute grid (last snapshot per bin)."""
    rows = []
    for f in sorted(glob.glob(str(load().root / "data/journal/xvenue/*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("sym") != "BTC":
                continue
            by, bn = r["venues"].get("by", {}), r["venues"].get("bn", {})
            if not by.get("mid") or not bn.get("mid"):
                continue
            rows.append((r["ts"], by["mid"], bn["mid"], by.get("funding"), bn.get("funding"), by.get("oi_usd")))
    d = pl.DataFrame(rows, schema=["ts", "by", "bn", "f_by", "f_bn", "oi"], orient="row")
    d = d.with_columns(pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC")).sort("ts")
    d = d.with_columns(pl.col("ts").dt.truncate("3m").alias("bin")).group_by("bin").agg(pl.all().last()).sort("bin")
    full = pl.DataFrame({"bin": pl.datetime_range(d["bin"].min(), d["bin"].max(), "3m", eager=True,
                                                  time_unit=d["bin"].dtype.time_unit, time_zone="UTC")})
    d = full.join(d, on="bin", how="left")
    pa, pb = d["by"].to_numpy().astype(float), d["bn"].to_numpy().astype(float)
    ok = np.isfinite(pa) & np.isfinite(pb)
    seg = np.cumsum(~ok)
    r = np.c_[np.r_[np.nan, np.diff(np.log(pa))], np.r_[np.nan, np.diff(np.log(pb))]]
    r[np.r_[True, ~ok[:-1]] | ~ok] = np.nan
    hl = 20
    v = np.c_[ewma_var(r[:, 0], hl), ewma_var(r[:, 1], hl)]
    sig = np.sqrt(v)
    yraw = r / sig
    aa = 1 - 0.5 ** (1 / hl)
    lv = 0.5 * np.log((1 - aa) * v[:, 0] + aa * np.nan_to_num(r[:, 0]) ** 2)
    fdiff = (d["f_by"].to_numpy().astype(float) - d["f_bn"].to_numpy().astype(float)) * 1e4
    oi = np.log(d["oi"].to_numpy().astype(float))
    oi_chg = np.r_[np.full(20, np.nan), oi[20:] - oi[:-20]]
    zc = {"btc_rv": trailing_z(lv, 480), "fund_diff": expanding_z(fdiff, 50), "oi_chg": expanding_z(oi_chg, 50)}
    valid = ok & (np.arange(len(d)) >= 60)
    return dict(name="xvenue_btc", ts=d["bin"].to_numpy(), y=np.clip(yraw, -CLIP, CLIP), yraw=yraw, sig=sig, r=r,
                Z=np.column_stack(list(zc.values())), z_names=list(zc), Xe=None,
                seg=np.where(valid, seg, -1 - np.arange(len(d))), valid=valid, assets=["bybit", "binance"],
                lat_r=None, freq="3m")


# ====================================================================== model runs


def ts_seconds(ts: np.ndarray) -> np.ndarray:
    return ts.astype("datetime64[s]").astype(np.int64)


def model_from(fit: dict[str, Any], K: int, p: int, nu: float | None) -> TVTPMSVAR:
    m = TVTPMSVAR(K=K, p=p, nu=nu)
    m.B, m.S, m.W, m.logxi0 = fit["B"], fit["S"], fit["W"], fit["logxi0"]
    m.n = m.B.shape[2]
    m.has_exog = m.B.shape[1] > 1 + m.n * p
    return m


def factory(K: int, p: int, nu: float | None, n_init: int = 3, seed: int = 0, max_iter: int = 200):
    return lambda: TVTPMSVAR(K=K, p=p, l2=1.0, cov_prior=2.0, n_init=n_init, seed=seed, nu=nu, max_iter=max_iter,
                             tol=1e-6)


def designs(ds: dict[str, Any], p: int) -> dict[str, Any]:
    Yt, X, Zc, rows = build_design(ds["y"], ds["Z"], ds["Xe"], ds["seg"], p)
    return dict(Yt=Yt, X=X, Zc=Zc, rows=rows, Zc0=np.ones((len(Yt), 1)))


def select_spec(ds: dict[str, Any], first_origin: datetime, grid: list, n_init: int = 3) -> dict[str, Any]:
    """Validation one-step log score on the rows before the first origin (70/30 chronological), same target rows for all
    specs (rows present for p = 2)."""
    tsec = ts_seconds(ds["ts"])
    o_s = int(first_origin.timestamp())
    pre = np.flatnonzero(ds["valid"] & (tsec < o_s))
    split_s = tsec[pre[int(0.7 * len(pre))]]
    d2 = designs(ds, 2)
    target = set(d2["rows"][(tsec[d2["rows"]] >= split_s) & (tsec[d2["rows"]] < o_s)].tolist())
    table = []
    for K, p, nu in grid:
        t0 = time.time()
        d = designs(ds, p)
        fit_m = tsec[d["rows"]] < split_s
        pre_m = tsec[d["rows"]] < o_s
        m = factory(K, p, nu, n_init)()
        m.fit_design(d["Yt"][fit_m], d["X"][fit_m], d["Zc"][fit_m])
        f = m.filter_design(d["Yt"][pre_m], d["X"][pre_m], d["Zc"][pre_m])
        rr = d["rows"][pre_m]
        sel = np.array([x in target for x in rr.tolist()])
        table.append(dict(K=K, p=p, nu=nu, val_logscore=float(f["pll"][sel].mean()), n_val=int(sel.sum()),
                          bic=float(m.bic), ll=float(m.ll), min_occupancy=m.min_occupancy, sec=time.time() - t0))
        print(f"  spec K={K} p={p} nu={nu}: val {table[-1]['val_logscore']:.4f} bic {m.bic:.1f} "
              f"({table[-1]['sec']:.0f}s)", flush=True)
    best = max(table, key=lambda r: r["val_logscore"])
    return {"table": table, "best": best, "split_ts": datetime.fromtimestamp(int(split_s), UTC),
            "n_fit_rows": int(len(pre))}


def origins_for(rows: np.ndarray, tsec: np.ndarray, first_origin: datetime, step: int) -> list[int]:
    o0 = int(np.searchsorted(tsec[rows], int(first_origin.timestamp())))
    return list(range(o0, len(rows), step))


def run_models(ds: dict[str, Any], K: int, p: int, nu: float | None, first_origin: datetime, step: int,
               rvar_window: int, window: int | None = None, n_init: int = 3) -> dict[str, Any]:
    d = designs(ds, p)
    tsec = ts_seconds(ds["ts"])
    orig = origins_for(d["rows"], tsec, first_origin, step)
    res = {"rows": d["rows"], "origins": orig}
    specs = {"tvtp": (factory(K, p, nu, n_init), d["Zc"], window), "ctp": (factory(K, p, nu, n_init), d["Zc0"], window),
             "var": (factory(1, p, nu), d["Zc0"], window), "rvar": (factory(1, p, nu), d["Zc0"], rvar_window)}
    for name, (fac, Zc, win) in specs.items():
        t0 = time.time()
        res[name] = walk_forward(fac, d["Yt"], d["X"], Zc, orig, window=win)
        res[name]["sec"] = time.time() - t0
        print(f"  {name}: {len(orig)} folds, {res[name]['sec']:.0f}s", flush=True)
    res["design"] = d
    return res


def seg_masks(ds: dict[str, Any], rows: np.ndarray, test: np.ndarray) -> dict[str, np.ndarray]:
    cut = int(knowledge_cutoff().timestamp())
    t = ts_seconds(ds["ts"])[rows]
    return {"llm_in_sample": test & (t < cut), "clean_holdout": test & (t >= cut)}


def logscore_table(ds: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    rows = res["rows"]
    test = np.isfinite(res["tvtp"]["pll"])
    out = {}
    for seg, msk in seg_masks(ds, rows, test).items():
        if msk.sum() < 20:
            continue
        o: dict[str, Any] = {"n": int(msk.sum()), "mean_logscore": {k: float(res[k]["pll"][msk].mean())
                                                                    for k in ("tvtp", "ctp", "var", "rvar")}}
        for b in ("ctp", "var", "rvar"):
            o[f"tvtp_minus_{b}"] = dm(res["tvtp"]["pll"][msk] - res[b]["pll"][msk])
        o["ctp_minus_var"] = dm(res["ctp"]["pll"][msk] - res["var"]["pll"][msk])
        out[seg] = o
    return out


def verdict_logscore(tab: dict[str, Any], need_ins: bool = True) -> dict[str, str]:
    v = {}
    ho, ins = tab.get("clean_holdout"), tab.get("llm_in_sample")
    if ins:
        ok = all(ins[f"tvtp_minus_{b}"]["mean"] > 0 and ins[f"tvtp_minus_{b}"]["t_hac"] >= 2 for b in ("ctp", "var", "rvar"))
        bad = ins["tvtp_minus_ctp"]["mean"] <= 0
        v["llm_in_sample"] = "supports" if ok else ("contradicts" if bad else "inconclusive")
    if ho:
        ok = all(ho[f"tvtp_minus_{b}"]["mean"] > 0 and ho[f"tvtp_minus_{b}"]["t_hac"] >= 2 for b in ("ctp", "var", "rvar"))
        if need_ins and ins:
            ok = ok and all(ins[f"tvtp_minus_{b}"]["mean"] > 0 for b in ("ctp", "var", "rvar"))
        bad = ho["tvtp_minus_ctp"]["mean"] <= 0 or any(
            ho[f"tvtp_minus_{b}"]["mean"] < 0 and ho[f"tvtp_minus_{b}"]["t_hac"] <= -2 for b in ("var", "rvar"))
        v["clean_holdout"] = "supports" if ok else ("contradicts" if bad else "inconclusive")
    return v


# ====================================================================== regimes (interpretation)


def regime_report(ds: dict[str, Any], res: dict[str, Any], K: int, p: int, nu: float | None) -> list[dict[str, Any]]:
    out = []
    zn = ds["z_names"]
    rows = res["rows"]
    Zc = res["design"]["Zc"]
    for f, fit in enumerate(res["tvtp"]["fits"]):
        m = model_from(fit, K, p, nu)
        lo, o = fit["lo"], fit["origin"]
        zbar = Zc[lo:o, 1:].mean(0)
        P = m.transition_matrix(zbar)
        reg = []
        for j in range(K):
            ccf = m.regime_ccf(j, 3)
            A1 = m.lag_matrix(j, 1)
            vol = np.sqrt(np.diag(m.S[j]))
            reg.append({"A1": A1.round(4).tolist(), "corr0": float(m.S[j][0, 1] / vol.prod()), "vol": vol.round(3).tolist(),
                        "ccf1_b_on_a": float(ccf[1][1, 0]), "ccf1_a_on_b": float(ccf[1][0, 1]),
                        "lead_asym": float(ccf[1][1, 0] - ccf[1][0, 1]), "stay_prob_at_mean_z": float(P[j, j]),
                        "expected_duration_rows": float(1 / max(1 - P[j, j], 1e-6))})
        ame = average_marginal_effects(m.W, Zc[lo:o])
        out.append({"fold": f, "origin_ts": datetime.fromtimestamp(int(ts_seconds(ds["ts"])[rows[o]]), UTC)
                    if o < len(rows) else None, "fit_rows": int(o - lo), "occupancy": m_occupancy(m, res, f, lo, o),
                    "regimes": reg, "W": fit["W"].round(3).tolist(), "z_names": zn,
                    "ame_leave_hazard": {zn[k]: ame[:, k].round(4).tolist() for k in range(len(zn))}})
    return out


def m_occupancy(m: TVTPMSVAR, res: dict[str, Any], f: int, lo: int, o: int) -> list[float]:
    d = res["design"]
    r = m.filter_design(d["Yt"][lo:o], d["X"][lo:o], d["Zc"][lo:o])
    return r["filt"].mean(0).round(3).tolist()


def average_marginal_effects(W: np.ndarray, Zc: np.ndarray) -> np.ndarray:
    """AME of each driver on the leave hazard 1 - P_ii, per regime: (K, d-1)."""
    K = W.shape[0]
    acc = np.zeros((K, Zc.shape[1] - 1))
    step = max(1, len(Zc) // 400)
    rows = Zc[::step]
    for zc in rows:
        me = marginal_effects(W, zc)
        acc += -np.stack([me[i, i] for i in range(K)])
    return acc / len(rows)


# ====================================================================== covariance forecasts (H-TVTP2)


def cov_rolling(y: np.ndarray, w: int) -> np.ndarray:
    T, n = y.shape
    H = np.full((T, n, n), np.nan)
    for t in range(w, T):
        H[t] = np.cov(y[t - w:t].T)
    return H


def cov_ewma(y: np.ndarray, lam: float, init: int = 50) -> np.ndarray:
    T, n = y.shape
    H = np.full((T, n, n), np.nan)
    h = np.cov(y[:init].T)
    for t in range(1, T):
        h = lam * h + (1 - lam) * np.outer(y[t - 1], y[t - 1])
        if t >= init:
            H[t] = h
    return H


def dcc_filter(e: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    T, n = e.shape
    R = np.empty((T, n, n))
    Q = Qbar.copy()
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(e[t - 1], e[t - 1]) + b * Q
        d = np.sqrt(np.diag(Q))
        R[t] = Q / np.outer(d, d)
    return R


def dcc_fit(e: np.ndarray) -> tuple[float, float, np.ndarray]:
    Qbar = np.corrcoef(e.T)

    def nll(x):
        a, b = 1 / (1 + np.exp(-x[0])) * 0.3, 1 / (1 + np.exp(-x[1]))
        if a + b >= 0.999:
            return 1e10
        R = dcc_filter(e, a, b, Qbar)
        det = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] ** 2
        q = (R[:, 1, 1] * e[:, 0] ** 2 - 2 * R[:, 0, 1] * e[:, 0] * e[:, 1] + R[:, 0, 0] * e[:, 1] ** 2) / det
        return 0.5 * float(np.sum(np.log(det) + q))
    r = minimize(nll, np.array([-1.5, 2.5]), method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
    return 1 / (1 + np.exp(-r.x[0])) * 0.3, 1 / (1 + np.exp(-r.x[1])), Qbar


def qlike(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    det = H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] ** 2
    q = (H[:, 1, 1] * y[:, 0] ** 2 - 2 * H[:, 0, 1] * y[:, 0] * y[:, 1] + H[:, 0, 0] * y[:, 1] ** 2) / det
    return np.log(det) + q


def covariance_study(ds: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    d = res["design"]
    rows, Yt = d["rows"], d["Yt"]
    y = ds["y"]
    test = np.isfinite(res["tvtp"]["pll"])
    full = np.where(ds["valid"][:, None], y, np.nan)
    yv = np.nan_to_num(full)
    Hr = cov_rolling(yv, 120)[rows]
    He = cov_ewma(yv, 0.97)[rows]
    # DCC: EWMA variances x DCC correlation, (a, b) by QMLE on each fold's fit rows
    ve = np.stack([cov_ewma(yv[:, [i]], 0.97)[:, 0, 0] for i in range(2)], 1)[rows]
    e = Yt / np.sqrt(ve)
    Hd = np.full_like(He, np.nan)
    params = []
    for fit, o2 in zip(res["tvtp"]["fits"], [f["origin"] for f in res["tvtp"]["fits"][1:]] + [len(rows)]):
        lo, o = fit["lo"], fit["origin"]
        fe = e[lo:o][np.isfinite(e[lo:o]).all(1)]
        a, b, Qbar = dcc_fit(fe)
        params.append((a, b))
        R = dcc_filter(np.nan_to_num(e[lo:o2]), a, b, Qbar)[o - lo:]
        sd = np.sqrt(ve[o:o2])
        Hd[o:o2] = R * sd[:, :, None] * sd[:, None, :]
    Ht = res["tvtp"]["cov"] + np.einsum("ta,tb->tab", res["tvtp"]["mean"], res["tvtp"]["mean"])
    Hc = res["ctp"]["cov"] + np.einsum("ta,tb->tab", res["ctp"]["mean"], res["ctp"]["mean"])
    models = {"tvtp": Ht, "ctp": Hc, "rolling120": Hr, "ewma097": He, "dcc": Hd}
    out: dict[str, Any] = {"dcc_params": params}
    for seg, msk in seg_masks(ds, rows, test).items():
        msk = msk & np.all([np.isfinite(H).all((1, 2)) for H in models.values()], 0)
        if msk.sum() < 20:
            continue
        L = {k: qlike(H[msk], Yt[msk]) for k, H in models.items()}
        M = {k: (H[msk][:, 0, 1] - Yt[msk][:, 0] * Yt[msk][:, 1]) ** 2 for k, H in models.items()}
        o: dict[str, Any] = {"n": int(msk.sum()), "qlike": {k: float(v.mean()) for k, v in L.items()},
                             "cov_mse": {k: float(v.mean()) for k, v in M.items()},
                             "mean_forecast_corr": {k: float(np.mean(H[msk][:, 0, 1] / np.sqrt(H[msk][:, 0, 0] * H[msk][:, 1, 1])))
                                                    for k, H in models.items()}}
        for b in ("ctp", "rolling120", "ewma097", "dcc"):
            o[f"qlike_tvtp_minus_{b}"] = dm(L["tvtp"] - L[b])
            o[f"covmse_tvtp_minus_{b}"] = dm(M["tvtp"] - M[b])
        out[seg] = o
    return out


def verdict_cov(tab: dict[str, Any]) -> dict[str, str]:
    v = {}
    bl = ("rolling120", "ewma097", "dcc")
    for seg in ("llm_in_sample", "clean_holdout"):
        s = tab.get(seg)
        if not s:
            continue
        ok = all(s[f"qlike_tvtp_minus_{b}"]["mean"] < 0 and s[f"qlike_tvtp_minus_{b}"]["t_hac"] <= -2 for b in bl)
        if seg == "clean_holdout" and "llm_in_sample" in tab:
            ok = ok and all(tab["llm_in_sample"][f"qlike_tvtp_minus_{b}"]["mean"] < 0 for b in bl)
        best = min(bl, key=lambda b: s["qlike"][b])
        bad = (s["qlike"]["tvtp"] > s["qlike"]["rolling120"] and s["qlike"]["tvtp"] > s["qlike"]["ewma097"]) or \
              (s[f"qlike_tvtp_minus_{best}"]["mean"] > 0 and s[f"qlike_tvtp_minus_{best}"]["t_hac"] >= 2)
        v[seg] = "supports" if ok else ("contradicts" if bad else "inconclusive")
    return v


# ====================================================================== trading (H-TVTP3 / 3b / 6b)


def roles(m: TVTPMSVAR, fixed_lead: int | None) -> dict[str, Any]:
    """Leader regimes from the first-lag CCF asymmetry corr(y1_{t+1}, y0_t) - corr(y0_{t+1}, y1_t) (> 0: asset 0 leads)."""
    asym = []
    for j in range(m.K):
        c = m.regime_ccf(j, 1)[1]
        asym.append(float(c[1, 0] - c[0, 1]) if np.isfinite(c).all() else 0.0)
    asym = np.array(asym)
    out: dict[str, Any] = {"asym": asym.tolist(), "lead0": None, "lead1": None}
    j0 = int(np.argmax(asym))
    if asym[j0] > 0:
        out["lead0"] = (j0, float(m.lag_matrix(j0)[1, 0]))          # asset 0 leads: trade asset 1
    if fixed_lead is None:
        j1 = int(np.argmin(asym))
        if asym[j1] < 0 and j1 != j0:
            out["lead1"] = (j1, float(m.lag_matrix(j1)[0, 1]))      # asset 1 leads: trade asset 0
    return out


def rule_positions(pred_next: np.ndarray, y_now: np.ndarray, rl: dict[str, Any], theta: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per decision row: asset traded (-1 none) and direction."""
    T = len(pred_next)
    asset, d = np.full(T, -1), np.zeros(T)
    if rl["lead0"] is not None:
        j, a = rl["lead0"]
        on = pred_next[:, j] > theta
        asset[on], d[on] = 1, np.sign(a * y_now[on, 0])
    if rl["lead1"] is not None:
        j, a = rl["lead1"]
        on = (pred_next[:, j] > theta) & (asset < 0)
        asset[on], d[on] = 0, np.sign(a * y_now[on, 1])
    asset[d == 0] = -1
    return asset, d


def free_positions(A1: np.ndarray, y_now: np.ndarray, c: float, fixed_lead: int | None) -> tuple[np.ndarray, np.ndarray]:
    if fixed_lead is None:
        lag = 1 if abs(A1[1, 0]) >= abs(A1[0, 1]) else 0
    else:
        lag = 1 - fixed_lead
    lead = 1 - lag
    T = len(y_now)
    on = np.abs(y_now[:, lead]) > c
    asset = np.where(on, lag, -1)
    d = np.where(on, np.sign(A1[lag, lead] * y_now[:, lead]), 0.0)
    asset[d == 0] = -1
    return asset, d


def trade_R(ds: dict[str, Any], rows_next: np.ndarray, asset: np.ndarray, d: np.ndarray, cost: str
            ) -> tuple[np.ndarray, np.ndarray]:
    """Net and gross R of one-bar trades. rows_next: original row index of the outcome bar."""
    on = asset >= 0
    R, G = np.full(len(asset), np.nan), np.full(len(asset), np.nan)
    if not on.any():
        return R, G
    rn, a = rows_next[on], asset[on]
    sig = ds["sig"][rn, a]
    if ds.get("lat_r") is not None and cost != "zero_latency":
        ret = ds["lat_r"][rn, a]
    else:
        ret = ds["r"][rn, a]
    gross = d[on] * ret / sig
    if "spread_bp" in ds:
        hs_in = ds["spread_bp"][rn - 1, a] / 2
        hs_out = ds["spread_bp"][rn, a] / 2
    else:
        hs_in = hs_out = ds["half_spread_bp"][a]
    if cost == "desk":
        c = COST["maker_bp"] + COST["peg_slip_bp"] + COST["taker_bp"] + hs_out
    elif cost == "taker":
        c = 2 * COST["taker_bp"] + hs_in + hs_out
    elif cost == "pegged":
        c = 2 * (COST["maker_bp"] + COST["peg_slip_bp"])
    elif cost == "maker":
        c = 2 * COST["maker_bp"]
    else:
        c = 0.0
    R[on] = gross - c / 1e4 / sig
    G[on] = gross
    return R, G


def summ(x: np.ndarray) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x))}
    m, t, n = newey_west_t(x)
    return {"n": n, "mean": m, "t_hac": t, "sr": float(x.mean() / x.std(ddof=1)), "sum": float(x.sum()),
            "hit": float((x > 0).mean())}


def trading_study(ds: dict[str, Any], res: dict[str, Any], K: int, p: int, nu: float | None, primary: str,
                  fixed_lead: int | None = None, n_trials: int = 40, seed: int = 7) -> dict[str, Any]:
    d = res["design"]
    rows = d["rows"]
    tsec = ts_seconds(ds["ts"])
    fits_t, fits_v = res["tvtp"]["fits"], res["var"]["fits"]
    ends = [f["origin"] for f in fits_t[1:]] + [len(rows)]
    # decision row r (design index) -> outcome row r + 1; require the outcome to be the next original row
    nxt_ok = np.r_[rows[1:] == rows[:-1] + 1, False]
    costs = ["desk", "taker", "pegged", "zero_latency"] if ds.get("lat_r") is None else \
        ["taker", "maker", "desk", "zero_latency"]
    picks, recs = [], {c: [] for c in costs}
    fixed = {th: [] for th in THETAS}
    free_rec = []
    sel_log = []
    for f, (ft, fv, o2) in enumerate(zip(fits_t, fits_v, ends)):
        lo, o = ft["lo"], ft["origin"]
        m = model_from(ft, K, p, nu)
        rl = roles(m, fixed_lead)
        # fit data: in-sample filtered one-step probabilities with this fold's parameters
        fr = m.filter_design(d["Yt"][lo:o], d["X"][lo:o], d["Zc"][lo:o])
        r_fit = np.arange(lo, o - 1)                                   # purge: the decision whose outcome is row o
        ok_fit = nxt_ok[r_fit]
        r_fit = r_fit[ok_fit]
        best, best_th = -np.inf, None
        for th in THETAS:
            a_, d_ = rule_positions(fr["pred"][r_fit + 1 - lo], d["Yt"][r_fit], rl, th)
            R, _ = trade_R(ds, rows[r_fit + 1], a_, d_, primary)
            x = R[np.isfinite(R)]
            if len(x) >= 30 and x.mean() > best:
                best, best_th = float(x.mean()), th
        if best_th is not None and best <= 0:
            best_th = None
        # regime-free comparator (pooled VAR of the same fold)
        A1 = model_from(fv, 1, p, nu).lag_matrix(0, 1)
        bc, bcv = None, -np.inf
        for c in CS:
            a_, d_ = free_positions(A1, d["Yt"][r_fit], c, fixed_lead)
            R, _ = trade_R(ds, rows[r_fit + 1], a_, d_, primary)
            x = R[np.isfinite(R)]
            if len(x) >= 30 and x.mean() > bcv:
                bcv, bc = float(x.mean()), c
        if bc is not None and bcv <= 0:
            bc = None
        sel_log.append({"fold": f, "theta": best_th, "fit_mean_R": best, "free_c": bc, "free_fit_mean_R": bcv,
                        "roles": rl})
        # test rows of this fold: decisions r in [o-1, o2-1) use pred[r+1] (row r+1 in this fold or the first of it)
        r_te = np.arange(max(o - 1, 0), o2 - 1)
        r_te = r_te[nxt_ok[r_te] & np.isfinite(res["tvtp"]["pll"][r_te + 1])]
        pn = res["tvtp"]["pred"][r_te + 1]
        for th in THETAS:
            a_, d_ = rule_positions(pn, d["Yt"][r_te], rl, th)
            R, G = trade_R(ds, rows[r_te + 1], a_, d_, primary)
            _, G0 = trade_R(ds, rows[r_te + 1], a_, d_, "zero_latency")
            fixed[th].append(np.c_[rows[r_te + 1], R, G, G0])
        if best_th is not None:
            a_, d_ = rule_positions(pn, d["Yt"][r_te], rl, best_th)
            for c in costs:
                R, G = trade_R(ds, rows[r_te + 1], a_, d_, c)
                recs[c].append(np.c_[rows[r_te + 1], R, G, a_, d_])
        if bc is not None:
            a_, d_ = free_positions(A1, d["Yt"][r_te], bc, fixed_lead)
            R, _ = trade_R(ds, rows[r_te + 1], a_, d_, primary)
            free_rec.append(np.c_[rows[r_te + 1], R])
        picks.append(best_th)
    cut = int(knowledge_cutoff().timestamp())
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"selection": sel_log, "primary_cost": primary}

    def cat(lst, k):
        return np.vstack(lst) if lst else np.zeros((0, k))
    for seg in ("llm_in_sample", "clean_holdout"):
        so: dict[str, Any] = {}
        for c in costs:
            M = cat(recs[c], 5)
            M = M[np.isfinite(M[:, 1])]
            t = tsec[M[:, 0].astype(int)] if len(M) else np.zeros(0)
            M = M[(t < cut) if seg == "llm_in_sample" else (t >= cut)]
            so[c] = summ(M[:, 1]) | {"gross": summ(M[:, 2])}
            if c == primary:
                prim = M
        F = cat(free_rec, 2)
        F = F[np.isfinite(F[:, 1])]
        tf = tsec[F[:, 0].astype(int)] if len(F) else np.zeros(0)
        F = F[(tf < cut) if seg == "llm_in_sample" else (tf >= cut)]
        so["regime_free"] = summ(F[:, 1])
        srs = []
        for th in THETAS:
            X = cat(fixed[th], 4)
            X = X[np.isfinite(X[:, 1])]
            tx = tsec[X[:, 0].astype(int)] if len(X) else np.zeros(0)
            X = X[(tx < cut) if seg == "llm_in_sample" else (tx >= cut)]
            s_ = summ(X[:, 1]) | {"gross": summ(X[:, 2]), "gross_zero_latency": summ(X[:, 3])}
            so[f"fixed_theta_{th}"] = s_
            srs.append(s_.get("sr", np.nan))
        if len(prim) >= 10:
            rn, a_, d_ = prim[:, 0].astype(int), prim[:, 3].astype(int), prim[:, 4]
            means = []
            for _ in range(2000):
                dd = rng.choice([-1.0, 1.0], len(rn))
                R, _ = trade_R(ds, rn, a_, dd, primary)
                means.append(np.nanmean(R))
            obs = float(np.mean(prim[:, 1]))
            so["random_sign"] = {"mean_of_means": float(np.mean(means)), "p_ge_observed": float(np.mean(np.array(means) >= obs))}
            so["dsr"] = deflated_sharpe(prim[:, 1], np.array(srs), n_trials)
            so["by_asset"] = {ds["assets"][k]: summ(prim[prim[:, 3] == k, 1]) for k in (0, 1)}
        out[seg] = so
    return out


def verdict_trading(tab: dict[str, Any], primary: str, single_segment: bool = False) -> dict[str, str]:
    v = {}
    ho, ins = tab.get("clean_holdout", {}), tab.get("llm_in_sample", {})
    p = ho.get(primary, {})
    if not p or p.get("n", 0) < (30 if not single_segment else 10) or "mean" not in p:
        v["clean_holdout"] = "inconclusive"
    elif p["mean"] <= 0:
        v["clean_holdout"] = "contradicts"
    else:
        ok = p["t_hac"] >= 2 and p["mean"] > ho.get("regime_free", {}).get("mean", -np.inf)
        if not single_segment:
            ok = ok and ho.get("random_sign", {}).get("p_ge_observed", 1) < 0.05 and ho.get("dsr", {}).get("dsr", 0) >= 0.95
            ok = ok and ins.get(primary, {}).get("mean", -1) > 0
        v["clean_holdout"] = "supports" if ok else "inconclusive"
    if ins.get(primary, {}).get("n", 0) >= 30:
        q = ins[primary]
        v["llm_in_sample"] = "contradicts" if q["mean"] <= 0 else ("supports" if q["t_hac"] >= 2 else "inconclusive")
    return v


# ====================================================================== CUSUM (H-TVTP5)


def cusum_study(ds: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    rows = res["rows"]
    pll = res["tvtp"]["pll"]
    test = np.isfinite(pll)
    tr = rows[test]
    _, _, al_c = cusum(pll[test], burn=168, halflife=168, kappa=0.5, h=5.0, refractory=24)
    yv = np.nan_to_num(np.where(ds["valid"][:, None], ds["y"], 0.0))
    al_r_full = corr_detector(yv)
    al_r = al_r_full[tr]
    al_r[:168] = False                                                # same burn as the CUSUM
    tsec = ts_seconds(ds["ts"])[tr]
    ev = pl.read_parquet(load().root / "data/research/news/events/events.parquet")
    ev = ev.filter(pl.col("kind").is_in(["cpi", "nfp", "fomc"]))
    ets = np.array([int(x.timestamp()) for x in ev["scheduled_ts"].to_list()])
    kinds = ev["kind"].to_list()
    t0, t1 = tsec[168], tsec[-1]
    sel = (ets >= t0) & (ets + 86400 <= t1 + 3600)
    ets, kinds = ets[sel], [k for k, s in zip(kinds, sel) if s]
    inwin = np.zeros(len(tsec), bool)
    for e in ets:
        inwin |= (tsec >= e) & (tsec < e + 86400)
    cut = int(knowledge_cutoff().timestamp())
    out: dict[str, Any] = {"n_episodes": int(len(ets))}
    for seg in ("llm_in_sample", "clean_holdout"):
        sm = (tsec < cut) if seg == "llm_in_sample" else (tsec >= cut)
        sm[:168] = False
        es = (ets < cut) if seg == "llm_in_sample" else (ets >= cut)
        so: dict[str, Any] = {"episodes": int(es.sum()), "rows": int(sm.sum())}
        for name, al in (("cusum", al_c), ("corr", al_r)):
            a = al & sm
            delays = []
            for e in ets[es]:
                hit = np.flatnonzero(a & (tsec >= e) & (tsec < e + 86400))
                if len(hit):
                    delays.append((tsec[hit[0]] - e) / 3600)
            n_out = int((sm & ~inwin).sum())
            false = int((a & ~inwin).sum())
            rate = a.sum() / max(sm.sum(), 1)
            chance = 1 - (1 - rate) ** 24
            k, n = len(delays), int(es.sum())
            so[name] = {"alarms": int(a.sum()), "hits": k, "hit_rate": k / n if n else None,
                        "median_delay_h": float(np.median(delays)) if delays else None,
                        "false_alarms": false, "false_per_1000": 1000 * false / max(n_out, 1),
                        "chance_hit_rate": float(chance),
                        "binom_p": float(stats.binomtest(k, n, chance, alternative="greater").pvalue) if n else None}
        out[seg] = so
    return out


def verdict_cusum(tab: dict[str, Any]) -> dict[str, str]:
    v = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        s = tab.get(seg)
        if not s or not s["episodes"]:
            continue
        c, r = s["cusum"], s["corr"]
        delay_ok = c["median_delay_h"] is None or r["median_delay_h"] is None or c["median_delay_h"] <= r["median_delay_h"]
        ok = (c["hit_rate"] >= r["hit_rate"] and c["false_per_1000"] <= r["false_per_1000"] and delay_ok
              and c["binom_p"] < 0.05)
        bad = (c["hit_rate"] < r["hit_rate"] and c["false_per_1000"] >= r["false_per_1000"]) or c["binom_p"] >= 0.5
        v[seg] = "supports" if ok else ("contradicts" if bad else "inconclusive")
    return v


# ====================================================================== AME bootstrap (H-TVTP4)


def ame_bootstrap(ds: dict[str, Any], K: int, p: int, nu: float | None, masks: dict[str, np.ndarray], reps: int,
                  seed: int = 3) -> dict[str, Any]:
    d = designs(ds, p)
    tsec = ts_seconds(ds["ts"])[d["rows"]]
    out: dict[str, Any] = {"z_names": ds["z_names"]}
    ref = None
    for seg, fn in masks.items():
        msk = fn(tsec)
        Yt, X, Zc = d["Yt"][msk], d["X"][msk], d["Zc"][msk]
        t0 = time.time()
        m = factory(K, p, nu, 3, seed)().fit_design(Yt, X, Zc)
        if ref is not None:
            m.align_to(ref)
        ref = ref or m
        ame0 = average_marginal_effects(m.W, Zc)
        rng = np.random.default_rng(seed + 1)
        boots = []
        nexo = X.shape[1] - (1 + m.n * p)
        for b in range(reps):
            ys, _ = sim_on_design(m, Yt, X, Zc, rng, nexo)
            mb = TVTPMSVAR(K=K, p=p, l2=1.0, cov_prior=2.0, n_init=1, seed=b, nu=nu, max_iter=100, tol=1e-6)
            Xb = rebuild_X(ys, X, p, m.n, nexo)
            mb.fit_design(ys, Xb, Zc, start=m.params())
            mb.align_to(m)
            boots.append(average_marginal_effects(mb.W, Zc))
        bt = np.array(boots)
        lo_, hi_ = np.quantile(bt, 0.025, axis=0), np.quantile(bt, 0.975, axis=0)
        so = {"n": int(msk.sum()), "ame": ame0.round(5).tolist(), "ci_lo": lo_.round(5).tolist(),
              "ci_hi": hi_.round(5).tolist(), "excl0": ((lo_ > 0) | (hi_ < 0)).tolist(), "sec": time.time() - t0,
              "W": m.W.round(3).tolist(), "regimes": [{"A1": m.lag_matrix(j).round(3).tolist(),
                                                       "corr0": float(m.S[j][0, 1] / np.sqrt(m.S[j][0, 0] * m.S[j][1, 1])),
                                                       "vol": np.sqrt(np.diag(m.S[j])).round(3).tolist()} for j in range(K)],
              "occupancy": m.prob_filtered.mean(0).round(3).tolist()}
        out[seg] = so
        print(f"  {seg}: AME done ({so['sec']:.0f}s)", flush=True)
    return out


def sim_on_design(m: TVTPMSVAR, Yt: np.ndarray, X: np.ndarray, Zc: np.ndarray, rng: np.random.Generator, nexo: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate y on the design rows with the observed Zc and exogenous columns (lags rebuilt from simulated y)."""
    T, n, p, K = len(Yt), m.n, m.p, m.K
    P = np.exp(log_trans(Zc, m.W))
    L = [np.linalg.cholesky(m.S[j]) for j in range(K)]
    y = np.zeros((T, n))
    s = rng.choice(K, p=np.exp(m.logxi0) / np.exp(m.logxi0).sum())
    hist = [X[0, 1 + n * l:1 + n * (l + 1)].copy() for l in range(p)]
    for t in range(T):
        if t > 0:
            pt = P[t][s]
            s = rng.choice(K, p=pt / pt.sum())
        x = np.r_[1.0, np.concatenate(hist), X[t, 1 + n * p:]] if nexo else np.r_[1.0, np.concatenate(hist)]
        e = L[s] @ rng.standard_normal(n)
        if m.nu is not None:
            e *= math.sqrt(m.nu / rng.chisquare(m.nu))
        y[t] = np.clip(x @ m.B[s] + e, -CLIP, CLIP)
        hist = [y[t]] + hist[:-1]
    return y, np.zeros(T)


def rebuild_X(ys: np.ndarray, X: np.ndarray, p: int, n: int, nexo: int) -> np.ndarray:
    Xb = X.copy()
    for l in range(1, p + 1):
        lagged = np.vstack([X[:l, 1 + n * (l - 1):1 + n * l], ys[:-l]])
        Xb[:, 1 + n * (l - 1):1 + n * l] = lagged
    return Xb


def verdict_ame(tab: dict[str, Any], drivers: list[str]) -> dict[str, str]:
    zn = tab["z_names"]
    A, B = tab.get("llm_in_sample"), tab.get("clean_holdout")
    if not A or not B:
        return {}
    ok, opp, anyB = False, False, False
    for dname in drivers:
        k = zn.index(dname)
        for i in range(len(A["ame"])):
            ea, eb = A["excl0"][i][k], B["excl0"][i][k]
            sa, sb = np.sign(A["ame"][i][k]), np.sign(B["ame"][i][k])
            anyB |= eb
            if ea and eb and sa == sb:
                ok = True
            if ea and eb and sa != sb:
                opp = True
    return {"clean_holdout": "supports" if ok else ("contradicts" if (not anyB or opp) else "inconclusive")}


# ====================================================================== drivers


def run_metals_1h(macro: bool = False, spec: dict | None = None) -> dict[str, Any]:
    ds = metals_dataset("1h", macro)
    first = datetime(2026, 5, 1, tzinfo=UTC)
    t0 = time.time()
    if spec is None:
        print("spec selection (M1h)", flush=True)
        sel = select_spec(ds, first, SPEC_GRID)
        save(f"{ds['name']}_spec.json", sel)
        spec = sel["best"]
    K, p, nu = spec["K"], spec["p"], spec["nu"]
    print(f"walk-forward {ds['name']} K={K} p={p} nu={nu}", flush=True)
    res = run_models(ds, K, p, nu, first, 336, 720)
    out = {"dataset": ds["name"], "spec": spec, "z_names": ds["z_names"], "n_rows": int(ds["valid"].sum()),
           "first_ts": ds["ts"][ds["valid"]][0], "last_ts": ds["ts"][-1]}
    out["logscore"] = logscore_table(ds, res)
    out["logscore_verdict"] = verdict_logscore(out["logscore"])
    out["regimes"] = regime_report(ds, res, K, p, nu)
    out["fits"] = [{k: v for k, v in f.items() if k in ("origin", "lo", "ll", "iters", "converged", "init_lls",
                                                         "min_occupancy")} for f in res["tvtp"]["fits"]]
    if not macro:
        out["covariance"] = covariance_study(ds, res)
        out["covariance_verdict"] = verdict_cov(out["covariance"])
        out["trading"] = trading_study(ds, res, K, p, nu, "desk")
        out["trading_verdict"] = verdict_trading(out["trading"], "desk")
        out["cusum"] = cusum_study(ds, res)
        out["cusum_verdict"] = verdict_cusum(out["cusum"])
    out["runtime_sec"] = time.time() - t0
    save(f"{ds['name']}_results.json", out)
    np.savez_compressed(out_dir() / f"{ds['name']}_walkforward.npz", rows=res["rows"],
                        **{f"{k}_pll": res[k]["pll"] for k in ("tvtp", "ctp", "var", "rvar")},
                        tvtp_pred=res["tvtp"]["pred"], tvtp_filt=res["tvtp"]["filt"], ts=ts_seconds(ds["ts"]))
    return out


def run_metals_1d() -> dict[str, Any]:
    ds = metals_dataset("1d", True)
    first = datetime(2026, 7, 1, tzinfo=UTC)
    t0 = time.time()
    K, p, nu = 2, 1, None
    res = run_models(ds, K, p, nu, first, 14, 60)
    out = {"dataset": "m1d", "spec": {"K": K, "p": p, "nu": nu}, "z_names": ds["z_names"],
           "n_rows": int(ds["valid"].sum())}
    out["logscore"] = logscore_table(ds, res)
    out["logscore_verdict"] = verdict_logscore(out["logscore"], need_ins=False)
    out["regimes"] = regime_report(ds, res, K, p, nu)
    print("AME bootstrap (M1d pooled)", flush=True)
    out["ame"] = ame_bootstrap(ds, K, p, nu, {"pooled": lambda t: np.ones(len(t), bool)}, 30)
    k1, k2 = ds["z_names"].index("d5_DFII10"), ds["z_names"].index("d5_DTWEXBGS")
    ex = out["ame"]["pooled"]["excl0"]
    out["ame_verdict"] = "supports" if any(ex[i][k] for i in range(K) for k in (k1, k2)) else "contradicts"
    out["runtime_sec"] = time.time() - t0
    save("m1d_results.json", out)
    return out


def run_ame(reps: int = 30) -> dict[str, Any]:
    spec = json.loads((out_dir() / "m1h_spec.json").read_text())["best"]
    ds = metals_dataset("1h", True)
    cut = int(knowledge_cutoff().timestamp())
    t0 = time.time()
    res = ame_bootstrap(ds, spec["K"], spec["p"], spec["nu"],
                        {"llm_in_sample": lambda t: t < cut, "clean_holdout": lambda t: t >= cut}, reps)
    res["verdict"] = verdict_ame(res, ["d5_DFII10", "d5_DTWEXBGS"])
    res["spec"] = spec
    res["runtime_sec"] = time.time() - t0
    save("m1h_macro_ame.json", res)
    return res


def run_pair_1m(ds: dict[str, Any], primary: str, fixed_lead: int | None, grid: list) -> dict[str, Any]:
    t0 = time.time()
    tv = ds["ts"][ds["valid"]]
    start = datetime.fromtimestamp(int(tv[0].astype("datetime64[s]").astype(np.int64)), UTC)
    # first origin: 4320 valid rows after the start
    tsec = ts_seconds(ds["ts"])
    vidx = np.flatnonzero(ds["valid"])
    first = datetime.fromtimestamp(int(tsec[vidx[min(4320, len(vidx) - 1)]]), UTC)
    print(f"{ds['name']}: {len(vidx)} valid rows from {start}, first origin {first}", flush=True)
    sel = select_spec(ds, first, grid, n_init=2)
    spec = sel["best"]
    K, p, nu = spec["K"], spec["p"], spec["nu"]
    res = run_models(ds, K, p, nu, first, 1440, 1440, window=4320, n_init=2)
    out = {"dataset": ds["name"], "spec_selection": sel, "spec": spec, "z_names": ds["z_names"],
           "n_valid_rows": int(len(vidx)), "first_origin": first}
    out["logscore"] = logscore_table(ds, res)
    out["logscore_verdict"] = verdict_logscore(out["logscore"], need_ins=False)
    out["regimes"] = regime_report(ds, res, K, p, nu)
    if primary:
        out["trading"] = trading_study(ds, res, K, p, nu, primary, fixed_lead=fixed_lead, n_trials=20)
        out["trading_verdict"] = verdict_trading(out["trading"], primary, single_segment=True)
    out["runtime_sec"] = time.time() - t0
    save(f"{ds['name']}_results.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["m1h", "m1h_macro", "m1d", "ame", "m1m", "crypto", "xvenue", "record", "record_dry", "svar"])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--alts", default="ETHUSDT,SOLUSDT,DOGEUSDT")
    a = ap.parse_args(argv)
    grid_1m = [(K, p, None) for K in (2, 3) for p in (1, 2)]
    if a.cmd == "m1h":
        r = run_metals_1h(False)
        print(json.dumps({k: r[k] for k in ("spec", "logscore_verdict", "covariance_verdict", "trading_verdict",
                                            "cusum_verdict")}, default=_js, indent=1))
    elif a.cmd == "m1h_macro":
        spec = json.loads((out_dir() / "m1h_spec.json").read_text())["best"]
        r = run_metals_1h(True, spec)
        print(json.dumps(r["logscore"], default=_js, indent=1))
    elif a.cmd == "m1d":
        r = run_metals_1d()
        print(json.dumps({k: r[k] for k in ("logscore", "logscore_verdict", "ame_verdict")}, default=_js, indent=1))
    elif a.cmd == "ame":
        r = run_ame(a.reps)
        print(json.dumps(r["verdict"], indent=1))
    elif a.cmd == "m1m":
        ds = pair_1m_dataset("XAUUSDT", "XAGUSDT", "metals")
        r = run_pair_1m(ds, "desk", None, grid_1m)
        print(json.dumps({k: r.get(k) for k in ("spec", "logscore_verdict", "trading_verdict")}, default=_js, indent=1))
    elif a.cmd == "crypto":
        for alt in a.alts.split(","):
            ds = pair_1m_dataset("BTCUSDT", alt, "crypto")
            r = run_pair_1m(ds, "taker", 0, grid_1m)
            print(json.dumps({k: r.get(k) for k in ("spec", "logscore_verdict", "trading_verdict")}, default=_js, indent=1))
    elif a.cmd == "svar":
        r = run_svar_diag()
        print(json.dumps({k: r[k] for k in ("verdict", "clean_holdout")}, default=_js, indent=1))
    elif a.cmd in ("record", "record_dry"):
        for r in record_all(dry=a.cmd == "record_dry"):
            print(r["id"], r["segment"], r["verdict"])
    elif a.cmd == "xvenue":
        ds = xvenue_dataset()
        r = run_pair_1m_xv(ds)
        print(json.dumps({k: r.get(k) for k in ("spec", "logscore_verdict")}, default=_js, indent=1))


def run_pair_1m_xv(ds: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    tsec = ts_seconds(ds["ts"])
    vidx = np.flatnonzero(ds["valid"])
    first = datetime.fromtimestamp(int(tsec[vidx[min(1440, len(vidx) - 1)]]), UTC)
    sel = select_spec(ds, first, [(K, p, None) for K in (2, 3) for p in (1, 2)], n_init=2)
    spec = sel["best"]
    res = run_models(ds, spec["K"], spec["p"], spec["nu"], first, 480, 480, window=1440, n_init=2)
    out = {"dataset": ds["name"], "spec_selection": sel, "spec": spec, "z_names": ds["z_names"],
           "n_valid_rows": int(len(vidx)), "first_origin": first}
    out["logscore"] = logscore_table(ds, res)
    out["logscore_verdict"] = verdict_logscore(out["logscore"], need_ins=False)
    out["regimes"] = regime_report(ds, res, spec["K"], spec["p"], spec["nu"])
    out["runtime_sec"] = time.time() - t0
    save("xvenue_btc_results.json", out)
    return out


# ====================================================================== H-TVTP1s and diagnostics (post hoc, registered)


def how_bucket(ts: np.ndarray, bar_s: int = 3600) -> np.ndarray:
    """Hour-of-week of the bar start (0 = Monday 00:00 UTC)."""
    start = ts_seconds(ts) - bar_s
    days = (start // 86400 + 3) % 7                                      # 1970-01-01 was a Thursday
    return (days * 24 + (start % 86400) // 3600).astype(int)


def run_svar_diag() -> dict[str, Any]:
    ds = metals_dataset("1h", False)
    r = json.loads((out_dir() / "m1h_results.json").read_text(encoding="utf-8"))
    K, p, nu = r["spec"]["K"], r["spec"]["p"], r["spec"]["nu"]
    wf = np.load(out_dir() / "m1h_walkforward.npz")
    d = designs(ds, p)
    rows = d["rows"]
    assert np.array_equal(rows, wf["rows"])
    hb = how_bucket(ds["ts"])
    n = 2
    folds = [f["origin"] for f in r["fits"]] + [len(rows)]
    pll_s = np.full(len(rows), np.nan)
    for o, o2 in zip(folds[:-1], folds[1:]):
        fit_rows = rows[:o]
        sfac = np.ones((168, n))
        yy = ds["y"][fit_rows]
        for h in range(168):
            mk = hb[fit_rows] == h
            if mk.sum() >= 5:
                sfac[h] = 0.5 * np.sqrt((yy[mk] ** 2).mean(0)) + 0.5
        S_all = sfac[hb]                                                 # per original row
        Ys = ds["y"] / S_all
        Ytd, Xd, Zcd, rr = build_design(Ys, None, ds["Xe"], ds["seg"], p)
        assert np.array_equal(rr, rows)
        m = factory(1, p, nu)().fit_design(Ytd[:o], Xd[:o], Zcd[:o])
        f = m.filter_design(Ytd[:o2], Xd[:o2], Zcd[:o2])
        pll_s[o:o2] = f["pll"][o:o2] - np.log(S_all[rows[o:o2]]).sum(1)
    test = np.isfinite(wf["tvtp_pll"])
    wd = hb[rows] < 4 * 24 + 21
    out: dict[str, Any] = {"note": "SVAR = K=1 t5 VAR on hour-of-week rescaled y, Jacobian-adjusted"}
    for seg, msk in seg_masks(ds, rows, test).items():
        o: dict[str, Any] = {"n": int(msk.sum()), "mean_logscore": {"tvtp": float(wf["tvtp_pll"][msk].mean()),
                             "svar": float(pll_s[msk].mean()), "var": float(wf["var_pll"][msk].mean()),
                             "ctp": float(wf["ctp_pll"][msk].mean())}}
        o["tvtp_minus_svar"] = dm(wf["tvtp_pll"][msk] - pll_s[msk])
        o["ctp_minus_svar"] = dm(wf["ctp_pll"][msk] - pll_s[msk])
        o["svar_minus_var"] = dm(pll_s[msk] - wf["var_pll"][msk])
        for lab, mm in (("weekday", msk & wd), ("weekend", msk & ~wd)):
            o[f"tvtp_minus_ctp_{lab}"] = dm(wf["tvtp_pll"][mm] - wf["ctp_pll"][mm])
            o[f"tvtp_minus_svar_{lab}"] = dm(wf["tvtp_pll"][mm] - pll_s[mm])
        out[seg] = o
    ho = out.get("clean_holdout", {})
    ok = ho and ho["tvtp_minus_svar"]["mean"] > 0 and ho["tvtp_minus_svar"]["t_hac"] >= 2 and         ho["tvtp_minus_ctp_weekday"]["mean"] > 0
    out["verdict"] = {"clean_holdout": "supports" if ok else ("contradicts" if ho and ho["tvtp_minus_svar"]["mean"] <= 0
                                                             else "inconclusive")}
    # trading diagnostics: gross and net per fixed theta, weekday vs weekend, with the fold roles of m1h_results
    sel = r["trading"]["selection"]
    nxt_ok = np.r_[rows[1:] == rows[:-1] + 1, False]
    diag: dict[str, Any] = {}
    cut = int(knowledge_cutoff().timestamp())
    tsec = ts_seconds(ds["ts"])
    for th in THETAS:
        recs = []
        for f, (o, o2) in enumerate(zip(folds[:-1], folds[1:])):
            rl = sel[f]["roles"]
            rl = {k: (tuple(v) if v is not None and k != "asym" else v) for k, v in rl.items()}
            r_te = np.arange(max(o - 1, 0), o2 - 1)
            r_te = r_te[nxt_ok[r_te] & np.isfinite(wf["tvtp_pll"][r_te + 1])]
            a_, d_ = rule_positions(wf["tvtp_pred"][r_te + 1], d["Yt"][r_te], rl, th)
            Rn, G = trade_R(ds, rows[r_te + 1], a_, d_, "desk")
            recs.append(np.c_[rows[r_te + 1], Rn, G, a_])
        M = np.vstack(recs)
        M = M[np.isfinite(M[:, 1])]
        wdm = hb[M[:, 0].astype(int)] < 4 * 24 + 21
        hold = tsec[M[:, 0].astype(int)] >= cut
        diag[str(th)] = {lab: {"net": summ(M[mk, 1]), "gross": summ(M[mk, 2])} for lab, mk in
                         (("holdout_all", hold), ("holdout_weekday", hold & wdm), ("holdout_weekend", hold & ~wdm),
                          ("insample_all", ~hold))}
    out["trading_fixed_theta"] = diag
    save("m1h_svar_diag.json", out)
    return out


# ====================================================================== recording


def _load(name: str) -> dict[str, Any] | None:
    f = out_dir() / name
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def record_all(dry: bool = False) -> list[dict[str, Any]]:
    """Append every available TVTP result to research/hypotheses_results.jsonl (failures included)."""
    from ai_trader.research.hypotheses import record
    rows: list[tuple] = []
    mc = _load("mc_summary.json")
    if mc:
        rows.append(("H-TVTP0", "pooled", mc["h0"]["verdict"], {"criteria": mc["h0"]["criteria"],
                     **{k: {kk: v[kk] for kk in ("acc_filtered_oos", "acc_smoothed", "slope_sign_rate_min",
                                                  "ccf_maxerr_median", "slope_wald95_coverage", "degenerate_share")}
                        for k, v in mc["summary"].items()}}, "synthetic Monte Carlo (tvtp_mc)", "synthetic"))
    m = _load("m1h_results.json")
    if m:
        data = "bybit 1h last XAUUSDT, XAGUSDT 2026-03-09..2026-11-25"
        for seg, v in m["logscore_verdict"].items():
            rows.append(("H-TVTP1", seg, v, m["logscore"][seg] | {"spec": m["spec"]}, "walk-forward log scores", data))
        for seg, v in m.get("covariance_verdict", {}).items():
            rows.append(("H-TVTP2", seg, v, m["covariance"][seg], "QLIKE primary, cov-element MSE secondary", data))
        for seg, v in m.get("trading_verdict", {}).items():
            rows.append(("H-TVTP3", seg, v, m["trading"][seg] | {"fixed_theta_gross": (_load("m1h_svar_diag.json") or {}).get(
                "trading_fixed_theta")}, "desk costs primary; selected rule took no trades (no threshold net positive on "
                "fit data in any fold); fixed thresholds all net negative", data))
        for seg, v in m.get("cusum_verdict", {}).items():
            rows.append(("H-TVTP5", seg, v, m["cusum"][seg], "CPI/NFP/FOMC 24h windows", data))
    d = _load("m1d_results.json")
    if d:
        data = "bybit 1d last XAUUSDT, XAGUSDT + FRED DFII10, DTWEXBGS, T10YIE"
        for seg, v in d["logscore_verdict"].items():
            rows.append(("H-TVTP1d", seg, v, d["logscore"][seg], "underpowered (about 145 test rows)", data))
        rows.append(("H-TVTP4d", "pooled", d["ame_verdict"], d["ame"]["pooled"] | {"z_names": d["z_names"]},
                     "full-sample fit, 30 parametric bootstrap reps", data))
    a = _load("m1h_macro_ame.json")
    if a:
        rows.append(("H-TVTP4", "clean_holdout", a["verdict"].get("clean_holdout", "inconclusive"),
                     {"z_names": a["z_names"], "llm_in_sample": a.get("llm_in_sample"), "clean_holdout": a.get("clean_holdout")},
                     "separate fits per subperiod, regimes aligned to the first", "M1h-macro"))
    mm = _load("metals_XAGUSDT_results.json")
    if mm:
        rows.append(("H-TVTP3b", "clean_holdout", mm["trading_verdict"].get("clean_holdout", "inconclusive"),
                     mm["trading"]["clean_holdout"] | {"logscore": mm["logscore"]}, "flow_3s 1m, 3 s latency",
                     "bybit flow_3s XAUUSDT, XAGUSDT 2026-11-07..11-25"))
    pairs = {alt: _load(f"crypto_{alt}_results.json") for alt in ("ETHUSDT", "SOLUSDT", "DOGEUSDT")}
    pairs = {k: v for k, v in pairs.items() if v}
    if pairs:
        data = "bybit flow_3s 1m mids BTCUSDT vs " + ", ".join(pairs)
        win, lose = 0, 0
        for alt, r in pairs.items():
            h = r["logscore"].get("clean_holdout", {})
            if h and all(h[f"tvtp_minus_{b}"]["mean"] > 0 and h[f"tvtp_minus_{b}"]["t_hac"] >= 2 for b in ("ctp", "var", "rvar")):
                win += 1
            if h and h["tvtp_minus_ctp"]["mean"] <= 0:
                lose += 1
        v6a = "supports" if win >= 2 else ("contradicts" if lose >= 2 else "inconclusive")
        rows.append(("H-TVTP6a", "clean_holdout", v6a, {alt: r["logscore"] for alt, r in pairs.items()} |
                     {"pairs_beating_all": win, "pairs_not_beating_ctp": lose}, "per-pair DM HAC", data))
        pos, nonpos = 0, 0
        for alt, r in pairs.items():
            t = r["trading"]["clean_holdout"].get("taker", {})
            if t.get("mean", -1) > 0 and t.get("t_hac", 0) >= 2:
                pos += 1
            if t.get("n", 0) >= 10 and t.get("mean", 1) <= 0:          # a pair with no selected trades is not evidence
                nonpos += 1
        v6b = "supports" if pos >= 2 else ("contradicts" if nonpos == len(pairs) == 3 else "inconclusive")
        rows.append(("H-TVTP6b", "clean_holdout", v6b, {alt: {c: r["trading"]["clean_holdout"].get(c) for c in
                     ("taker", "maker", "desk", "zero_latency", "fixed_theta_0.5", "fixed_theta_0.7", "fixed_theta_0.9")}
                     for alt, r in pairs.items()},
                     "taker primary; the selected rule took no trades on any pair (no threshold net positive on fit data); "
                     "fixed-threshold variants are reported", data))
    sv = _load("m1h_svar_diag.json")
    if sv:
        rows.append(("H-TVTP1s", "clean_holdout", sv["verdict"]["clean_holdout"],
                     {k: sv[k] for k in ("llm_in_sample", "clean_holdout") if k in sv},
                     "post hoc, registered after H-TVTP1; session-aware VAR baseline", "M1h"))
    x = _load("xvenue_btc_results.json")
    if x and not x["logscore_verdict"]:
        rows.append(("H-TVTP6c", "clean_holdout", "inconclusive",
                     {"n_grid_bins": 4851, "bins_with_valid_returns": 1224, "valid_rows_after_burn": x["n_valid_rows"],
                      "design_rows_p1": 877, "test_rows": 0, "spec_selection": x["spec_selection"]["table"]},
                     "not testable as registered: journal snapshots too irregular for a 3-minute grid (no test rows after "
                     "the 1440-row first window); no substitute design run", "data/journal/xvenue BTC bybit vs binance mids"))
    if x:
        for seg, v in x["logscore_verdict"].items():
            rows.append(("H-TVTP6c", seg, v, x["logscore"][seg], "3-minute grid of journal snapshots",
                         "data/journal/xvenue BTC bybit vs binance mids"))
    out = []
    for hid, seg, verdict, metrics, note, data in rows:
        if dry:
            out.append({"id": hid, "segment": seg, "verdict": verdict})
        else:
            out.append(record(hid, seg, verdict, json.loads(json.dumps(metrics, default=_js)), note=note, data=data))
    return out


if __name__ == "__main__":
    main()
