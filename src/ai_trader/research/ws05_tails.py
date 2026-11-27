"""WS05 tail risk: jumps, clustering and joint tails (specification program 2026-11-27; H-WS05-1..17).

Owns the trimmed core of WS06 DEPEND (co-jumps, tail dependence, DCC vs EWMA, effective bets, co-stopping, live limits).
Design frozen in the diary ("WS05 design frozen before results") before any fit. Research only: reads the lake, writes
data/research/ws05/.

Stages (each through the governor):
    python -m ai_trader.research.ws05_tails extract   flow_3s -> pool 3s (tmp, deleted at the end) + 1m bars
    python -m ai_trader.research.ws05_tails jumps     LM / BNS / ASJ on BTC 5m, flow 1m + 5m, ETH Binance 1m + 5m
    python -m ai_trader.research.ws05_tails hawkes    BTC 5m Hawkes family, walk-forward J1 / J4 forecasts vs field
    python -m ai_trader.research.ws05_tails density   SV-diffusion vs Merton vs Kou vs t on BTC 5m standardised returns
    python -m ai_trader.research.ws05_tails liq       liquidation clusters -> jump hazard, marked + multivariate Hawkes
    python -m ai_trader.research.ws05_tails stops     stop slips on the 3s path, live-limit losses
    python -m ai_trader.research.ws05_tails depend    tail dependence, copulas, DCC vs EWMA, ENB, co-stopping

Conventions: log returns; LM = Lee-Mykland (2008) with Gumbel critical value at alpha 1% per day (n = bars per day),
spot vol = bipower over the previous day of periodicity-adjusted returns (excludes the tested bar); periodicity =
hour-of-day mean |r| factor from PAST days only, shrunk to 1 with 3 pseudo-days. BNS = Huang-Tauchen ratio-max with
tri-power quarticity; ASJ = Ait-Sahalia-Jacod (2009) S(4, 2) with truncated power variations, M(4, 2) = 160 / 3.
Discrete-time Hawkes: Bernoulli per bin with cloglog link, P(event in bin) = 1 - exp(-lambda),
lambda = exp(baseline) + sum alpha * E(kappa), E_t = kappa E_{t-1} + src_{t-1}; forecasts of >= 1 event in the next H
bins are exact under the model (1 - exp(-sum lambda) with excitation from pre-origin events only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import optimize, signal, special, stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "ws05"
TMP = OUT / "tmp"
LAKE = ROOT / "data" / "lake"
CUTOFF_MS = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1000)
ALPHA = 0.01
H_MS, D_MS, M_MS = 3_600_000, 86_400_000, 60_000
POOL = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT",
        "DOTUSDT", "BNBUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT", "XLMUSDT",
        "SUIUSDT", "ENAUSDT", "HYPEUSDT"]
FOCUS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"]
SEGS = ("llm_in_sample", "clean_holdout")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seg(ms) -> np.ndarray:
    return np.where(np.asarray(ms) >= CUTOFF_MS, "clean_holdout", "llm_in_sample")


def save_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=_js), encoding="utf-8")


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def duck():
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='600MB'")
    return con


# ============================================================================ jump tests


def lm_threshold(n: int, alpha: float = ALPHA) -> float:
    """|z| critical value of the Lee-Mykland max statistic (z = r / sigma with sigma the proper spot sd)."""
    l = math.sqrt(2 * math.log(n))
    return l - (math.log(math.pi) + math.log(math.log(n))) / (2 * l) - math.log(-math.log(1 - alpha)) / l


def hod_factor(ts_ms: np.ndarray, r: np.ndarray, trail_days: int = 90, shrink: float = 3.0) -> np.ndarray:
    """Hour-of-day sd factor per bar from PAST days only (mean |r| per day-hour relative to the day's mean; trailing
    mean over the previous trail_days days, shrunk to 1 with `shrink` pseudo-days; normalised to mean f^2 = 1)."""
    day = ts_ms // D_MS
    hod = (ts_ms // H_MS) % 24
    di = (day - day.min()).astype(np.int64)
    nd = int(di.max()) + 1
    a = np.abs(r)
    ok = np.isfinite(a)
    S, C = np.zeros((nd, 24)), np.zeros((nd, 24))
    np.add.at(S, (di[ok], hod[ok]), a[ok])
    np.add.at(C, (di[ok], hod[ok]), 1.0)
    M = np.where(C >= 3, S / np.maximum(C, 1), np.nan)
    G = M / np.nanmean(M, axis=1, keepdims=True)
    val = np.isfinite(G)
    cs = np.vstack([np.zeros((1, 24)), np.cumsum(np.where(val, G, 0.0), 0)])
    cn = np.vstack([np.zeros((1, 24)), np.cumsum(val, 0)])
    idx = np.arange(nd)
    lo = np.maximum(idx - trail_days, 0)
    F = (cs[idx] - cs[lo] + shrink) / (cn[idx] - cn[lo] + shrink)
    F = F / np.sqrt(np.mean(F ** 2, axis=1, keepdims=True))
    return F[di, hod]


def spot_bv(x: np.ndarray, K: int, min_share: float = 0.8) -> np.ndarray:
    """sigma_i = sqrt(pi/2 * mean |x_j||x_{j-1}|) over j in [i-K+1, i-1] (bar i excluded)."""
    prod = np.concatenate([[np.nan], np.abs(x[1:]) * np.abs(x[:-1])])
    ok = np.isfinite(prod)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, prod, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(x))
    lo = np.maximum(i - K + 1, 0)
    s, n = cs[i] - cs[lo], cn[i] - cn[lo]
    v = (np.pi / 2) * s / np.maximum(n, 1)
    v[(n < min_share * (K - 1)) | (v <= 0)] = np.nan
    return np.sqrt(v)


@dataclass
class Series:
    """A regular grid of returns: ts (bar open, ms), r (log return of the bar, nan if invalid)."""
    name: str
    ts: np.ndarray
    r: np.ndarray
    n_day: int
    f: np.ndarray = field(default=None)
    sig: np.ndarray = field(default=None)
    z: np.ndarray = field(default=None)
    jump: np.ndarray = field(default=None)


def lm_detect(s: Series, trail_days: int = 90, min_share: float | None = None) -> Series:
    """min_share: coverage of the bipower window; 0.8 by default, 0.3 for the gappy flow_3s series (deviation logged
    in the diary: the collector covers ~13.6k of the ~27k minutes, in fragments)."""
    if min_share is None:
        min_share = 0.3 if "flow" in s.name else 0.8
    s.f = hod_factor(s.ts, s.r, trail_days)
    x = s.r / s.f
    s.sig = spot_bv(x, s.n_day, min_share)
    s.z = x / s.sig
    thr = lm_threshold(s.n_day)
    s.jump = np.isfinite(s.z) & (np.abs(s.z) > thr)
    return s


MU43 = 2 ** (2 / 3) * special.gamma(7 / 6) / special.gamma(0.5)
THETA_BNS = (np.pi / 2) ** 2 + np.pi - 5
M42 = 160.0 / 3.0


def day_tests(s: Series, min_share: float | None = None) -> pl.DataFrame:
    """Per day: RV, BV (raw), jump share, LM any, BNS ratio-max z, ASJ S(4,2) and its z; tests on x = r / f.
    Products use only adjacent valid bars (normalised by the number of valid pairs / triples); ASJ uses complete pairs.
    min_share: 0.9 of the day's bars by default, 0.4 for flow_3s series (deviation, collector gaps)."""
    if min_share is None:
        min_share = 0.4 if "flow" in s.name else 0.9
    day = s.ts // D_MS
    x = s.r / s.f
    rows = []
    za, zq = stats.norm.ppf(1 - ALPHA), stats.norm.ppf(1 - ALPHA)
    bounds = np.flatnonzero(np.diff(day)) + 1
    for a, b in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(day)]])):
        rr, xx, jj = s.r[a:b], x[a:b], s.jump[a:b]
        ok = np.isfinite(rr) & np.isfinite(xx)
        n = int(ok.sum())
        if n < min_share * s.n_day:
            continue
        r_, x_ = np.where(ok, rr, 0.0), np.where(ok, xx, 0.0)
        p2 = ok[1:] & ok[:-1]
        p3 = ok[2:] & ok[1:-1] & ok[:-2]
        n2, n3 = max(int(p2.sum()), 1), max(int(p3.sum()), 1)
        RV = float(np.sum(r_ ** 2))
        BV = float((np.pi / 2) * n / n2 * np.sum(np.abs(r_[1:]) * np.abs(r_[:-1])))
        RVx = float(np.sum(x_ ** 2))
        BVx = float((np.pi / 2) * n / n2 * np.sum(np.abs(x_[1:]) * np.abs(x_[:-1])))
        ax = np.abs(x_) ** (4 / 3)
        TQx = float(n * MU43 ** -3 * n / n3 * np.sum(ax[2:] * ax[1:-1] * ax[:-2]))
        z_bns = ((RVx - BVx) / RVx) / math.sqrt(THETA_BNS / n * max(1.0, TQx / BVx ** 2)) if RVx > 0 and BVx > 0 else np.nan
        u = 4.0 * math.sqrt(BVx / n)
        m2 = (len(x_) // 2) * 2
        cp = ok[0:m2:2] & ok[1:m2:2]
        B1 = float(np.sum((x_[0:m2:2] ** 4 + x_[1:m2:2] ** 4)[cp]))
        B2 = float(np.sum(((x_[0:m2:2] + x_[1:m2:2]) ** 4)[cp]))
        S = B2 / B1 if B1 > 0 else np.nan
        tr = np.abs(x_) <= u
        A4 = n / 3.0 * np.sum(x_[tr] ** 4)
        A8 = n ** 3 / 105.0 * np.sum(x_[tr] ** 8)
        V = (1.0 / n) * M42 * A8 / A4 ** 2 if A4 > 0 else np.nan
        z_asj = (2.0 - S) / math.sqrt(V) if V and V > 0 else np.nan
        rows.append({"day_ms": int(day[a] * D_MS), "n": n, "RV": RV, "BV": BV, "jshare": max(RV - BV, 0.0) / RV if RV > 0 else np.nan,
                     "lm": bool(jj.any()), "n_lm": int(jj.sum()), "z_bns": z_bns, "bns": bool(z_bns > zq), "S_asj": S, "z_asj": z_asj,
                     "asj": bool(z_asj > za)})
    return pl.DataFrame(rows)


def kappa(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0:
        return {"n": 0}
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {"n": n, "kappa": k, "jaccard": inter / union if union else np.nan, "rate_a": pa, "rate_b": pb, "both": inter}


# ============================================================================ data


def btc5m_bybit() -> tuple[np.ndarray, np.ndarray]:
    con = duck()
    g = (LAKE / "bars" / "venue=bybit" / "price_type=last" / "freq=5m").as_posix() + "/**/*.parquet"
    df = pl.from_arrow(con.sql(f"""SELECT epoch_ms(event_ts) ts, close, epoch_ms(known_ts) k FROM read_parquet('{g}', hive_partitioning=true)
                                   WHERE symbol='BTCUSDT'""").to_arrow_table())
    df = df.sort("k").unique("ts", keep="last").sort("ts")
    return grid_returns(df["ts"].to_numpy(), df["close"].to_numpy(), 300_000)


def grid_returns(ts: np.ndarray, close: np.ndarray, step: int, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Full regular grid from first to last ts; r = log close diff where both bars exist (and are valid)."""
    t0, t1 = int(ts.min()), int(ts.max())
    g = np.arange(t0, t1 + step, step, dtype=np.int64)
    c = np.full(len(g), np.nan)
    i = ((ts - t0) // step).astype(np.int64)
    c[i] = np.where(valid, close, np.nan) if valid is not None else close
    r = np.concatenate([[np.nan], np.diff(np.log(c))])
    return g, r


def flow1m() -> pl.DataFrame:
    return pl.read_parquet(OUT / "flow1m.parquet")


def flow_series(sym: str, m1: pl.DataFrame | None = None, freq_min: int = 1) -> Series:
    m1 = flow1m() if m1 is None else m1
    d = m1.filter(pl.col("symbol") == sym).sort("minute")
    ts = d["minute"].to_numpy().astype(np.int64) * M_MS
    ok = d["n3"].to_numpy() >= 17
    g, r = grid_returns(ts, d["close"].to_numpy(), M_MS, ok)
    if freq_min == 1:
        return Series(f"{sym}_flow1m", g, r, 1440)
    k = freq_min
    off = int(((-(g[0] // M_MS)) % k))
    g2, r2 = g[off:], r[off:]
    m = len(r2) // k
    R = r2[: m * k].reshape(m, k)
    return Series(f"{sym}_flow{k}m", g2[: m * k : k], R.sum(1), 1440 // k)  # nan if any minute invalid


def eth1m_binance() -> tuple[np.ndarray, np.ndarray]:
    con = duck()
    g = (LAKE / "bars" / "venue=binance" / "price_type=last" / "freq=1m").as_posix() + "/**/*.parquet"
    df = pl.from_arrow(con.sql(f"""SELECT epoch_ms(event_ts) ts, close, epoch_ms(known_ts) k FROM read_parquet('{g}', hive_partitioning=true)
                                   WHERE symbol='ETHUSDT'""").to_arrow_table())
    df = df.sort("k").unique("ts", keep="last").sort("ts")
    return grid_returns(df["ts"].to_numpy(), df["close"].to_numpy(), M_MS)


def agg_series(name: str, g: np.ndarray, r: np.ndarray, k: int, n_day: int) -> Series:
    off = int(((-(g[0] // M_MS)) % k)) if k > 1 else 0
    g2, r2 = g[off:], r[off:]
    m = len(r2) // k
    return Series(name, g2[: m * k : k], r2[: m * k].reshape(m, k).sum(1), n_day)


# ============================================================================ stage: extract


def stage_extract() -> None:
    con = duck()
    base = LAKE / "flow_3s" / "venue=bybit" / "category=linear"
    TMP.mkdir(parents=True, exist_ok=True)
    syms = ",".join(f"'{s}'" for s in POOL)
    parts, m1 = [], []
    for d in sorted(base.glob("date=*")):
        q = f"""SELECT symbol, epoch_ms(event_ts) ts, open, high, low, close, spread_bps sp, epoch_ms(known_ts) k
                FROM read_parquet('{d.as_posix()}/*.parquet', union_by_name=true) WHERE symbol IN ({syms})"""
        df = pl.from_arrow(con.sql(q).to_arrow_table()).sort("k").unique(["symbol", "ts"], keep="last").drop("k")
        df = df.with_columns(pl.col("open", "high", "low", "sp").cast(pl.Float32)).sort("symbol", "ts")
        parts.append(df)
        m1.append(df.with_columns((pl.col("ts") // M_MS).alias("minute")).group_by("symbol", "minute").agg(
            pl.col("open").drop_nulls().first().alias("open"), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.len().alias("n3"), pl.col("sp").mean().alias("spread_bps"),
            pl.col("close").first().alias("close_first")))
        log("extract", d.name, df.height)
    pl.concat(parts).write_parquet(TMP / "flow3s_pool.parquet", compression="zstd")
    pl.concat(m1).sort("symbol", "minute").write_parquet(OUT / "flow1m.parquet", compression="zstd")
    log("extract done")


# ============================================================================ stage: jumps


def detect_all() -> dict:
    """Builds every Series used later, detects LM jumps and day tests; saves bar-level jumps and day tables."""
    out = {}
    g, r = btc5m_bybit()
    out["btc5m"] = lm_detect(Series("btc5m_bybit", g, r, 288))
    m1 = flow1m()
    for sym in POOL:
        out[f"{sym}_1m"] = lm_detect(flow_series(sym, m1, 1))
        out[f"{sym}_5m"] = lm_detect(flow_series(sym, m1, 5))
    ge, re_ = eth1m_binance()
    out["eth1m"] = lm_detect(Series("eth1m_binance", ge, re_, 1440))
    out["eth5m"] = lm_detect(agg_series("eth5m_binance", ge, re_, 5, 288))
    return out


def stage_jumps() -> None:
    S = detect_all()
    res = {"lm_thr": {n: lm_threshold(n) for n in (288, 1440)}, "datasets": {}, "agreement": {}}
    days = {}
    for k, s in S.items():
        dt = day_tests(s)
        days[k] = dt
        valid = np.isfinite(s.z)
        sg = seg(s.ts)
        info = {}
        for sgn in SEGS:
            m = sg == sgn
            dsel = dt.filter(pl.Series(seg(dt["day_ms"].to_numpy()) == sgn)) if dt.height else dt
            if dsel.height == 0:
                continue
            info[sgn] = {"bars_valid": int((valid & m).sum()), "n_jumps": int((s.jump & m).sum()),
                         "jump_rate_per_bar": float((s.jump & m).sum() / max((valid & m).sum(), 1)), "days": dsel.height,
                         "share_lm": float(dsel["lm"].mean()), "share_bns": float(dsel["bns"].mean()), "share_asj": float(dsel["asj"].mean()),
                         "jshare_mean": float(dsel["jshare"].mean()), "jshare_median": float(dsel["jshare"].median())}
        res["datasets"][k] = info
    # agreement: tests within dataset; frequencies on the same days
    for k in ["btc5m", "eth1m", "eth5m", "BTCUSDT_1m", "BTCUSDT_5m"]:
        dt = days[k]
        for sgn in SEGS:
            d = dt.filter(pl.Series(seg(dt["day_ms"].to_numpy()) == sgn))
            if d.height < 5:
                continue
            res["agreement"][f"{k}|{sgn}|tests"] = {"lm_bns": kappa(d["lm"], d["bns"]), "lm_asj": kappa(d["lm"], d["asj"]), "bns_asj": kappa(d["bns"], d["asj"])}
    pairs = [("eth1m", "eth5m")] + [(f"{s}_1m", f"{s}_5m") for s in POOL]
    pool_rows = {t: ([], []) for t in ("lm", "bns", "asj")}
    for a, b in pairs:
        j = days[a].join(days[b], on="day_ms", suffix="_b")
        for sgn in SEGS:
            d = j.filter(pl.Series(seg(j["day_ms"].to_numpy()) == sgn))
            if d.height < 5:
                continue
            res["agreement"][f"{a}~{b}|{sgn}|freq"] = {t: kappa(d[t], d[f"{t}_b"]) for t in ("lm", "bns", "asj")}
            if a != "eth1m":
                for t in pool_rows:
                    pool_rows[t][0].extend(d[t].to_list())
                    pool_rows[t][1].extend(d[f"{t}_b"].to_list())
    res["agreement"]["flow_pool_1m~5m|clean_holdout|freq"] = {t: kappa(*pool_rows[t]) for t in pool_rows}
    # pooled flow tests agreement (1m)
    allf = pl.concat([days[f"{s}_1m"] for s in POOL])
    res["agreement"]["flow_pool_1m|clean_holdout|tests"] = {"lm_bns": kappa(allf["lm"], allf["bns"]), "lm_asj": kappa(allf["lm"], allf["asj"]),
                                                            "bns_asj": kappa(allf["bns"], allf["asj"])}
    # jump sizes (BTC 5m): |z| and bp, exceedance tails
    s = S["btc5m"]
    jz, jr, js = np.abs(s.z[s.jump]), np.abs(s.r[s.jump]), s.sig[s.jump] * s.f[s.jump]
    thr = lm_threshold(288)
    sizes = {}
    for sgn in SEGS:
        m = seg(s.ts[s.jump]) == sgn
        e = jz[m] - thr
        sz = {"n": int(m.sum()), "z_median": float(np.median(jz[m])), "z_p90": float(np.quantile(jz[m], 0.9)),
              "bp_median": float(1e4 * np.median(jr[m])), "bp_p90": float(1e4 * np.quantile(jr[m], 0.9)), "bp_max": float(1e4 * jr[m].max()),
              "up_share": float(np.mean(s.r[s.jump][m] > 0))}
        lam = 1 / e.mean()
        ll_exp = float(np.sum(np.log(lam) - lam * e))
        xi, _, sc = stats.genpareto.fit(e, floc=0)
        ll_gpd = float(np.sum(stats.genpareto.logpdf(e, xi, 0, sc)))
        sz.update({"exceed_mean_sd": float(e.mean()), "gpd_xi": float(xi), "lr_gpd_vs_exp": 2 * (ll_gpd - ll_exp),
                   "lr_p": float(stats.chi2.sf(max(2 * (ll_gpd - ll_exp), 0), 1))})
        lx = np.log(js[m])
        ly = np.log(jr[m])
        A = np.column_stack([np.ones_like(lx), lx])
        coef = np.linalg.lstsq(A, ly, rcond=None)[0]
        resid = ly - A @ coef
        se = math.sqrt(np.sum(resid ** 2) / (len(ly) - 2) / np.sum((lx - lx.mean()) ** 2))
        sz.update({"slope_logsize_on_logsig": float(coef[1]), "slope_se": se})
        sizes[sgn] = sz
    res["sizes_btc5m"] = sizes
    save_json("jumps.json", res)
    # bar-level jump tables and per-bar arrays for later stages
    rows = []
    for k, s in S.items():
        idx = np.flatnonzero(s.jump)
        rows.append(pl.DataFrame({"set": [k] * len(idx), "ts": s.ts[idx], "z": s.z[idx], "r": s.r[idx], "sig": (s.sig * s.f)[idx]}))
    pl.concat(rows).write_parquet(OUT / "jumps_bars.parquet")
    for k, dt in days.items():
        dt.with_columns(pl.lit(k).alias("set"))
    pl.concat([dt.with_columns(pl.lit(k).alias("set")) for k, dt in days.items() if dt.height]).write_parquet(OUT / "jump_days.parquet")
    b = S["btc5m"]
    pl.DataFrame({"ts": b.ts, "r": b.r, "f": b.f, "sig": b.sig, "z": b.z, "jump": b.jump}).write_parquet(OUT / "btc5m_bars.parquet")
    log("jumps done")


# ============================================================================ discrete Hawkes


def lf(src: np.ndarray, k: float) -> np.ndarray:
    """E_t = k E_{t-1} + src_{t-1} (along axis 0)."""
    return signal.lfilter([0.0, 1.0], [1.0, -k], src, axis=0)


@dataclass
class HModel:
    """lambda[t, c] = exp(offset + icpt[c] + X[t, c] @ b) + sum_terms exp(la[pidx[c]]) * E_term[t, c](kappa_g)."""
    Y: np.ndarray                # (T, C) 0/1
    V: np.ndarray                # (T, C) 1 if the bin enters the likelihood
    X: np.ndarray                # (T, C, P)
    offset: np.ndarray           # (T, C)
    icpt: np.ndarray             # (C,) intercept param index per component
    terms: list                  # (src (T,) or (T, C), group, pidx (C,) with -1 = off)
    n_i: int
    n_a: int
    n_g: int

    @property
    def n_par(self) -> int:
        return self.n_i + self.X.shape[2] + self.n_a + self.n_g

    def split(self, th):
        P = self.X.shape[2]
        a = th[: self.n_i]
        b = th[self.n_i: self.n_i + P]
        la = th[self.n_i + P: self.n_i + P + self.n_a]
        lb = th[self.n_i + P + self.n_a:]
        return a, b, la, lb

    def intensity(self, th, want_E: bool = False):
        a, b, la, lb = self.split(th)
        eta = self.offset + a[self.icpt][None, :] + (self.X @ b if self.X.shape[2] else 0.0)
        mu = np.exp(np.clip(eta, -40, 5))
        kap = np.exp(-np.exp(lb))
        lam = mu.copy()
        Es = []
        for src, g, pidx in self.terms:
            E = lf(src, kap[g])
            E = E if E.ndim == 2 else np.repeat(E[:, None], lam.shape[1], 1) if lam.shape[1] > 1 else E[:, None]
            al = np.where(pidx >= 0, np.exp(la[np.maximum(pidx, 0)]), 0.0)
            lam = lam + E * al[None, :]
            Es.append((E, g, pidx, al))
        return (lam, mu, Es, kap) if want_E else lam

    def nll(self, th):
        lam, mu, Es, kap = self.intensity(th, True)
        lam = np.maximum(lam, 1e-12)
        em = -np.expm1(-lam)
        y = self.Y > 0
        ll = np.where(y, np.log(np.maximum(em, 1e-300)), -lam) * self.V
        w = np.where(y, np.exp(-lam) / np.maximum(em, 1e-300), -1.0) * self.V
        a, b, la, lb = self.split(th)
        g_a = np.zeros(self.n_i)
        wm = w * mu
        np.add.at(g_a, self.icpt, wm.sum(0))
        g_b = np.einsum("tc,tcp->p", wm, self.X) if self.X.shape[2] else np.zeros(0)
        g_la, g_lb = np.zeros(self.n_a), np.zeros(self.n_g)
        for (E, g, pidx, al) in Es:
            on = pidx >= 0
            np.add.at(g_la, pidx[on], (al * (w * E).sum(0))[on])
            D = lf(E, kap[g])
            g_lb[g] += float(np.sum(al[None, :] * w * D)) * (-kap[g] * np.exp(lb[g]))
        return -float(ll.sum()), -np.concatenate([g_a, g_b, g_la, g_lb])

    def fit(self, x0=None, bounds=None):
        x0 = np.zeros(self.n_par) if x0 is None else x0
        if bounds is None:
            P = self.X.shape[2]
            bounds = [(-30, 5)] * self.n_i + [(-10, 10)] * P + [(-25, 3)] * self.n_a + [(math.log(1e-4), math.log(20))] * self.n_g
        r = optimize.minimize(self.nll, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        return r

    def rescaled(self, th, c: int = 0, seed: int = 0):
        """Discrete time-rescaling (Haslinger, Pipa, Brown 2010) for component c: iid Exp(1) under the model."""
        lam = self.intensity(th)[:, c]
        v = self.V[:, c] > 0
        lam, y = lam[v], self.Y[v, c] > 0
        p = -np.expm1(-lam)
        rng = np.random.default_rng(seed)
        ev = np.flatnonzero(y)
        cs = np.concatenate([[0.0], np.cumsum(np.where(y, 0.0, lam))])
        taus = []
        for i in range(1, len(ev)):
            a, b = ev[i - 1], ev[i]
            q = cs[b] - cs[a + 1]
            taus.append(q - math.log(1 - rng.random() * p[b]))
        return np.asarray(taus)


def gof(taus: np.ndarray) -> dict:
    if len(taus) < 10:
        return {"n": len(taus)}
    u = 1 - np.exp(-taus)
    ks = stats.kstest(u, "uniform")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(u, lags=[10], return_df=True)
    return {"n": len(taus), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "lb10_p": float(lb["lb_pvalue"].iloc[0])}


def num_hessian(m: HModel, th, eps: float = 1e-4) -> np.ndarray:
    k = len(th)
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = eps
        H[i] = (m.nll(th + e)[1] - m.nll(th - e)[1]) / (2 * eps)
    return 0.5 * (H + H.T)


# ============================================================================ forecast scoring


def nw_var(d: np.ndarray, L: int) -> float:
    d = d - d.mean()
    n = len(d)
    f = np.fft.rfft(d, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: L + 1] / n
    w = 1 - np.arange(L + 1) / (L + 1)
    return float(ac[0] + 2 * np.sum(w[1:] * ac[1:]))


def dm_hln(d: np.ndarray, L: int, h: int = 1) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return {"n": n}
    v = nw_var(d, L)
    t = d.mean() / math.sqrt(v / n) if v > 0 else np.nan
    t *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"n": n, "mean": float(d.mean()), "t": float(t), "p_two": float(2 * stats.t.sf(abs(t), n - 1))}


def bern_ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.where(y, np.log(p), np.log1p(-p))


def calib(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = y.astype(float)
    n = len(p)
    se = math.sqrt(max(y.mean() * (1 - y.mean()), 1e-12) / n)
    sz = float(np.sum((y - p) * (1 - 2 * p)) / math.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p))))
    lp = np.log(p / (1 - p))
    try:
        import statsmodels.api as sm
        g = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        slope, icpt, slope_se = float(g.params[1]), float(g.params[0]), float(g.bse[1])
    except Exception:
        slope = icpt = slope_se = np.nan
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    k = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    tab = [{"p_mean": float(p[k == i].mean()), "obs": float(y[k == i].mean()), "n": int((k == i).sum())} for i in range(len(qs) - 1) if (k == i).any()]
    return {"n": n, "base_rate": float(y.mean()), "mean_p": float(p.mean()), "citl_z": float((y.mean() - p.mean()) / se),
            "spiegelhalter_z": sz, "slope": slope, "slope_se": slope_se, "intercept": icpt,
            "brier": float(np.mean((p - y) ** 2)), "logscore": float(np.mean(bern_ll(p, y > 0))), "reliability": tab}


def mcs(L: np.ndarray, names: list[str], alpha: float = 0.10, B: int = 1000, block: int = 24, seed: int = 0) -> dict:
    """Hansen-Lunde-Nason MCS with the Tmax statistic and a moving-block bootstrap. L = losses (n, M)."""
    n, M = L.shape
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    cs = np.vstack([np.zeros((1, M)), np.cumsum(L, 0)])
    bsum = cs[block:] - cs[:-block]                     # (n - block + 1, M)
    boot = bsum[starts].sum(1) / (nb * block)           # (B, M) bootstrap means
    keep = list(range(M))
    pv = {}
    prun = 0.0
    while len(keep) > 1:
        Lm = L[:, keep].mean(0)
        db = Lm - Lm.mean()
        bb = boot[:, keep]
        dbb = bb - bb.mean(1, keepdims=True)
        var = np.mean((dbb - db[None, :]) ** 2, 0)
        t = db / np.sqrt(var)
        Tb = np.max((dbb - db[None, :]) / np.sqrt(var)[None, :], 1)
        p = float(np.mean(Tb >= t.max()))
        prun = max(prun, p)
        w = int(np.argmax(t))
        if prun >= alpha:
            break
        pv[names[keep[w]]] = prun
        keep.pop(w)
    for k in keep:
        pv[names[k]] = 1.0
    return {"pvalues": pv, "mcs": [names[k] for k in keep], "alpha": alpha}


# ============================================================================ stage: hawkes (BTC 5m, J1 / J4)


def release_bins(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ev = pl.read_parquet(ROOT / "data" / "research" / "news" / "events" / "events.parquet").filter(pl.col("kind").is_in(["cpi", "nfp", "fomc"]))
    t = ev["scheduled_ts"].dt.epoch("ms").to_numpy()
    k = ev["schedule_known_ts"].dt.epoch("ms").to_numpy()
    r0, r1 = np.zeros(len(ts)), np.zeros(len(ts))
    step = int(ts[1] - ts[0])
    for te, ke in zip(t, k):
        i = (te - ts[0]) // step
        if 0 <= i < len(ts) and ke <= te:
            r0[i] = 1.0
            r1[i + 1: i + 1 + (25 * M_MS) // step] = 1.0
    return r0, r1


def btc_hawkes_inputs():
    b = pl.read_parquet(OUT / "btc5m_bars.parquet")
    ts = b["ts"].to_numpy()
    y = b["jump"].to_numpy().astype(float)
    V = np.isfinite(b["z"].to_numpy()).astype(float)
    lf_ = np.log(b["f"].to_numpy())
    lsig = np.log(b["sig"].to_numpy())
    hs = ts % H_MS == 0
    vh = np.where(hs, lsig, np.nan)
    vh = pl.Series(vh).fill_nan(None).forward_fill().to_numpy().astype(float)   # log spot vol at the last hour close
    ok = np.isfinite(lf_) & np.isfinite(vh)
    V = V * ok
    lf_, vh = np.nan_to_num(lf_), np.nan_to_num(vh)
    r0, r1 = release_bins(ts)
    return ts, y, V, lf_, vh, r0, r1


def build_btc_model(kind: str, y, V, lf_, vh, r0, r1, vbar: float) -> HModel:
    T = len(y)
    cols = {"M1": [lf_], "M2": [lf_, vh - vbar], "M3": [lf_], "M4": [lf_, vh - vbar], "M5": [lf_, vh - vbar, r0, r1], "M4b": [lf_, vh - vbar]}[kind]
    X = np.stack(cols, 1)[:, None, :]
    terms, n_a, n_g = [], 0, 0
    if kind in ("M3", "M4", "M5"):
        terms, n_a, n_g = [(y, 0, np.array([0]))], 1, 1
    if kind == "M4b":
        terms, n_a, n_g = [(y, 0, np.array([0])), (y, 1, np.array([1]))], 2, 2
    return HModel(y[:, None], V[:, None], X, np.zeros((T, 1)), np.array([0]), terms, 1, n_a, n_g)


def hawkes_forecast(m: HModel, th, kind: str, origins: np.ndarray, H: int, lf_, vh, r0, r1, vbar) -> np.ndarray:
    a, b, la, lb = m.split(th)
    P = []
    base_cols = {"M1": 1, "M2": 2, "M3": 1, "M4": 2, "M5": 4, "M4b": 2}[kind]
    hh = np.arange(H)
    idx = origins[:, None] + hh[None, :]
    idx = np.minimum(idx, len(lf_) - 1)
    eta = a[0] + b[0] * lf_[idx]
    if base_cols >= 2:
        eta = eta + b[1] * (vh[origins] - vbar)[:, None]
    if base_cols == 4:
        eta = eta + b[2] * r0[idx] + b[3] * r1[idx]
    Lam = np.exp(np.clip(eta, -40, 5)).sum(1)
    if m.terms:
        kap = np.exp(-np.exp(lb))
        for src, g, pidx in m.terms:
            E = lf(src, kap[g])
            al = math.exp(la[pidx[0]])
            k = kap[g]
            Lam = Lam + al * E[origins] * (1 - k ** H) / (1 - k)
    P = -np.expm1(-Lam)
    return P


def field_p() -> pl.DataFrame:
    """Frozen cutoff field: p_jump per 1h state row (origin = bar close), the field's own posterior jump weight of the
    realised next return and its implied P(4-sigma flag next bar), plus the realised flag."""
    from ai_trader.field.kramers_moyal import _posterior
    from ai_trader.field.model import Field, coords_matrix
    from ai_trader.field.state import build_state, load_frame

    fld = Field.load(ROOT / "data" / "models" / "field" / "btc_field_cutoff_2026-07-01.json")
    fr = load_frame(datetime(2026, 11, 26, 23, 59, tzinfo=UTC), start=datetime(2020, 4, 1, tzinfo=UTC))
    st = build_state(fr, fld.profile, fld.cfg)
    st = st.with_columns(pl.col("flag").shift(-1).alias("flag_next"))
    X = coords_matrix(st)
    lnvf = np.log(st["vf"].to_numpy())
    good = np.all(np.isfinite(X), 1) & np.isfinite(lnvf)
    st, X, lnvf = st.filter(pl.Series(good)), X[good], lnvf[good]
    mu, D, p = fld.params(X, lnvf)
    r = st["r_next"].to_numpy().astype(float)
    okr = np.isfinite(r)
    w = np.full(len(r), np.nan)
    w[okr] = _posterior(r[okr], mu[okr], D[okr], p[okr], fld.kou, fld.nu())
    Lm = fld.cfg.clip * np.sqrt(st["f_next"].to_numpy() * st["vf"].to_numpy())
    pf = 1 - (fld.cdf(Lm, X, lnvf) - fld.cdf(-Lm, X, lnvf))
    return pl.DataFrame({"origin_ms": st["close_ts"].dt.epoch("ms").to_numpy().astype(np.int64), "p_field": p, "w_next": w,
                         "p_flag": pf, "flag_next": st["flag_next"].cast(pl.Float64).to_numpy(), "r_next": r})


def flow_btc5m_targets():
    """Clean-holdout extension: BTC 5m LM jumps from flow_3s (Nov 2026)."""
    s = lm_detect(flow_series("BTCUSDT", None, 5))
    return s


def stage_hawkes() -> None:
    ts, y, V, lf_, vh, r0, r1 = btc_hawkes_inputs()
    T = len(ts)
    hours = np.flatnonzero(ts % H_MS == 0)
    res = {"cutoff_fit": {}, "walk_forward": {}}
    # ---------- targets at hourly origins
    def target(H):
        cy = np.concatenate([[0], np.cumsum(y * V)])
        cv = np.concatenate([[0], np.cumsum(V)])
        o = hours[hours + H <= T]
        nj, nv = cy[o + H] - cy[o], cv[o + H] - cv[o]
        return o, (nj > 0), nv >= 0.83 * H
    # ---------- cutoff fit (H-WS05-4, H-WS05-9)
    tr = ts < CUTOFF_MS
    vbar = float(np.mean(vh[tr & (V > 0)]))
    fits = {}
    for kind in ("M1", "M2", "M3", "M4", "M4b", "M5"):
        m = build_btc_model(kind, y * tr, V * tr, lf_, vh, r0, r1, vbar)
        x0 = np.zeros(m.n_par)
        x0[0] = math.log(max(y[tr].mean(), 1e-5))
        if m.n_a:
            x0[m.n_i + m.X.shape[2]: m.n_i + m.X.shape[2] + m.n_a] = math.log(0.05)
            x0[-m.n_g:] = [math.log(0.2), math.log(0.01)][: m.n_g]
        r = m.fit(x0)
        fits[kind] = (m, r.x)
        taus = m.rescaled(r.x)
        info = {"nll": float(r.fun), "theta": r.x.tolist(), "conv": bool(r.success), "gof": gof(taus)}
        if m.n_a:
            a, b, la, lb = m.split(r.x)
            beta = np.exp(lb)
            info["alpha"] = np.exp(la).tolist()
            info["half_life_min"] = (5 * math.log(2) / beta).tolist()
            info["branching"] = float(np.sum(np.exp(la) / (1 - np.exp(-beta))))
        res["cutoff_fit"][kind] = info
        log("cutoff", kind, info["nll"], info.get("gof"))
    cf = res["cutoff_fit"]
    res["cutoff_tests"] = {
        "LR_M3_vs_M1": 2 * (cf["M1"]["nll"] - cf["M3"]["nll"]), "LR_M4_vs_M2": 2 * (cf["M2"]["nll"] - cf["M4"]["nll"]),
        "LR_M5_vs_M4": 2 * (cf["M4"]["nll"] - cf["M5"]["nll"]), "LR_M4b_vs_M4": 2 * (cf["M4"]["nll"] - cf["M4b"]["nll"]),
        "release_coef": cf["M5"]["theta"][3:5]}
    # LR for alpha at the boundary: 0.5 chi2_0 + 0.5 chi2_1
    for k in ("LR_M3_vs_M1", "LR_M4_vs_M2"):
        res["cutoff_tests"][k + "_p"] = 0.5 * float(stats.chi2.sf(max(res["cutoff_tests"][k], 0), 1))
    res["cutoff_tests"]["LR_M5_vs_M4_p"] = float(stats.chi2.sf(max(res["cutoff_tests"]["LR_M5_vs_M4"], 0), 2))
    # ---------- walk-forward monthly
    fp = field_p()
    months = []
    d = datetime(2021, 4, 1, tzinfo=UTC)
    end_ms = int(ts[-1])
    while int(d.timestamp() * 1000) < end_ms:
        months.append(int(d.timestamp() * 1000))
        d = datetime(d.year + (d.month // 12), d.month % 12 + 1, 1, tzinfo=UTC)
    months.append(end_ms + 1)
    kinds = ("M1", "M2", "M3", "M4", "M5")
    preds = {h: {k: [] for k in kinds} for h in (12, 48)}
    org = {h: [] for h in (12, 48)}
    warm = {k: None for k in kinds}
    for mi in range(len(months) - 1):
        m0, m1 = months[mi], months[mi + 1]
        trm = ts < m0
        vbar_m = float(np.mean(vh[trm & (V > 0)]))
        for kind in kinds:
            m = build_btc_model(kind, y * trm, V * trm, lf_, vh, r0, r1, vbar_m)
            x0 = warm[kind] if warm[kind] is not None else fits[kind][1]
            r = m.fit(x0)
            warm[kind] = r.x
            mf = build_btc_model(kind, y, V, lf_, vh, r0, r1, vbar_m)
            for H in (12, 48):
                o, _, _ = target(H)
                o = o[(ts[o] >= m0) & (ts[o] < m1)]
                preds[H][kind].append(hawkes_forecast(mf, r.x, kind, o, H, lf_, vh, r0, r1, vbar_m))
        for H in (12, 48):
            o, _, _ = target(H)
            org[H].append(o[(ts[o] >= m0) & (ts[o] < m1)])
        log("wf", datetime.fromtimestamp(m0 / 1000, UTC).date())
    for H in (12, 48):
        o_all, J_all, ok_all = target(H)
        pos = {int(v): i for i, v in enumerate(o_all)}
        o = np.concatenate(org[H])
        sel = np.array([pos[int(v)] for v in o])
        J, ok = J_all[sel], ok_all[sel]
        P = {k: np.concatenate(preds[H][k]) for k in kinds}
        # M0 expanding base rate of the target (origins strictly before, label fully observed: o + H <= origin)
        cJ = np.cumsum(np.where(ok_all, J_all, 0))
        cN = np.cumsum(ok_all)
        oa = ts[o_all]
        idx_prev = np.searchsorted(oa + H * 300_000, ts[o], side="right") - 1
        P["M0"] = np.where(idx_prev >= 0, (cJ[np.maximum(idx_prev, 0)] + 1) / (cN[np.maximum(idx_prev, 0)] + 2), 0.02)
        # pure diffusion: LM false-positive rate per bar
        P["PD"] = np.full(len(o), 1 - (1 - ALPHA) ** (H / 288))
        # field
        f = pl.DataFrame({"origin_ms": ts[o]}).join(fp, on="origin_ms", how="left")
        pf = f["p_field"].to_numpy().astype(float)
        P["FIELD"] = pf if H == 12 else 1 - (1 - pf) ** 4
        frame = pl.DataFrame({"ts": ts[o], "J": J, "ok": ok, **{k: v for k, v in P.items()}})
        frame.write_parquet(OUT / f"hawkes_preds_h{H}.parquet")
    # field's own definition
    fp.write_parquet(OUT / "field_p.parquet")
    save_json("hawkes_fits.json", res)
    log("hawkes done")


def flow_btc_holdout_preds() -> pl.DataFrame:
    """J1 / J4 for the Nov 2026 flow_3s BTC 5m with the last walk-forward fits (refit on all Bybit 5m data)."""
    ts, y, V, lf_, vh, r0, r1 = btc_hawkes_inputs()
    vbar = float(np.mean(vh[V > 0]))
    s = flow_btc5m_targets()
    t2 = s.ts
    y2 = s.jump.astype(float)
    V2 = np.isfinite(s.z).astype(float)
    lsig = np.log(s.sig)
    vh2 = pl.Series(np.where(t2 % H_MS == 0, lsig, np.nan)).fill_nan(None).forward_fill().to_numpy().astype(float)
    lf2 = np.log(s.f)
    ok = np.isfinite(vh2) & np.isfinite(lf2)
    V2 = V2 * ok
    vh2, lf2 = np.nan_to_num(vh2), np.nan_to_num(lf2)
    r02, r12 = release_bins(t2)
    fp = field_p()
    out = []
    kinds = ("M1", "M2", "M3", "M4", "M5")
    fitted = {}
    for kind in kinds:
        m = build_btc_model(kind, y, V, lf_, vh, r0, r1, vbar)
        x0 = np.zeros(m.n_par)
        x0[0] = math.log(max(y.mean(), 1e-5))
        if m.n_a:
            x0[m.n_i + m.X.shape[2]] = math.log(0.05)
            x0[-1] = math.log(0.2)
        fitted[kind] = m.fit(x0).x
    for H in (12, 48):
        hours = np.flatnonzero(t2 % H_MS == 0)
        o = hours[hours + H <= len(t2)]
        cy = np.concatenate([[0], np.cumsum(y2 * V2)])
        cv = np.concatenate([[0], np.cumsum(V2)])
        J, okk = (cy[o + H] - cy[o]) > 0, (cv[o + H] - cv[o]) >= 0.83 * H
        P = {}
        for kind in kinds:
            mf = build_btc_model(kind, y2, V2, lf2, vh2, r02, r12, vbar)
            P[kind] = hawkes_forecast(mf, fitted[kind], kind, o, H, lf2, vh2, r02, r12, vbar)
        # M0: base rate from all Bybit 5m targets
        cyb = np.concatenate([[0], np.cumsum(y * V)])
        cvb = np.concatenate([[0], np.cumsum(V)])
        hb = np.flatnonzero(ts % H_MS == 0)
        hb = hb[hb + H <= len(ts)]
        Jb = (cyb[hb + H] - cyb[hb]) > 0
        okb = (cvb[hb + H] - cvb[hb]) >= 0.83 * H
        P["M0"] = np.full(len(o), (Jb[okb].sum() + 1) / (okb.sum() + 2))
        P["PD"] = np.full(len(o), 1 - (1 - ALPHA) ** (H / 288))
        f = pl.DataFrame({"origin_ms": t2[o]}).join(fp, on="origin_ms", how="left")
        pf = f["p_field"].to_numpy().astype(float)
        P["FIELD"] = pf if H == 12 else 1 - (1 - pf) ** 4
        out.append(pl.DataFrame({"H": [H] * len(o), "ts": t2[o], "J": J, "ok": okk, **P}))
    return pl.concat(out)


def stage_score() -> None:
    """Scores (log score, Brier, calibration, DM, MCS) for J1 / J4 by segment, incl. the Nov flow extension."""
    res = {}
    nov = flow_btc_holdout_preds()
    nov.write_parquet(OUT / "hawkes_preds_nov.parquet")
    models = ["M0", "M1", "M2", "M3", "M4", "M5", "FIELD", "PD"]
    for H in (12, 48):
        d = pl.read_parquet(OUT / f"hawkes_preds_h{H}.parquet")
        dn = nov.filter(pl.col("H") == H).drop("H")
        # field recalibration fitted on llm_in_sample (logit-linear), applied out of sample
        dd = pl.concat([d.with_columns(pl.lit("bybit").alias("src")), dn.with_columns(pl.lit("flow_nov").alias("src"))], how="diagonal")
        dd = dd.filter(pl.col("ok") & pl.col("FIELD").is_not_null())
        sg = seg(dd["ts"].to_numpy())
        ins = sg == "llm_in_sample"
        import statsmodels.api as sm
        pf = np.clip(dd["FIELD"].to_numpy(), 1e-9, 1 - 1e-9)
        g = sm.GLM(dd["J"].to_numpy().astype(float)[ins], sm.add_constant(np.log(pf / (1 - pf))[ins]), family=sm.families.Binomial()).fit()
        dd = dd.with_columns(pl.Series("FIELD_R", special.expit(g.params[0] + g.params[1] * np.log(pf / (1 - pf)))))
        mods = models + ["FIELD_R"]
        out = {"field_recal": {"a": float(g.params[0]), "b": float(g.params[1])}}
        for sgn, mask in (("llm_in_sample", ins), ("clean_holdout", ~ins), ("clean_holdout_bybit", (~ins) & (dd["src"].to_numpy() == "bybit")),
                          ("clean_holdout_flow_nov", dd["src"].to_numpy() == "flow_nov")):
            sub = dd.filter(pl.Series(mask))
            if sub.height < 50:
                continue
            Jv = sub["J"].to_numpy()
            o = {"n": sub.height, "events": int(Jv.sum()), "models": {}}
            LS = {}
            for k in mods:
                if sgn == "llm_in_sample" and k == "FIELD_R":
                    continue
                p = sub[k].to_numpy().astype(float)
                LS[k] = bern_ll(p, Jv)
                o["models"][k] = calib(p, Jv)
            L = 24 if H == 12 else 48
            hh = 1 if H == 12 else 4
            o["dm"] = {f"{a}-{b}": dm_hln(LS[a] - LS[b], L, hh) for a, b in
                       [("M4", "M0"), ("M3", "M0"), ("M2", "M0"), ("M1", "M0"), ("M5", "M4"), ("M4", "FIELD"), ("FIELD", "M0"), ("M4", "M2"),
                        ("FIELD_R", "M0"), ("M4", "FIELD_R")] if a in LS and b in LS}
            names = [k for k in mods if k in LS and k != "PD"]
            o["mcs"] = mcs(-np.column_stack([LS[k] for k in names]), names, block=24)
            out[sgn] = o
        res[f"H{H // 12}h"] = out
        log("scored", H)
    save_json("hawkes_scores.json", res)
    # field own-definition calibration (H-WS05-7)
    fp = pl.read_parquet(OUT / "field_p.parquet").drop_nulls()
    own = {}
    for sgn in SEGS:
        f = fp.filter(pl.Series(seg(fp["origin_ms"].to_numpy()) == sgn))
        p, w, pf, fl = (f[c].to_numpy() for c in ("p_field", "w_next", "p_flag", "flag_next"))
        own[sgn] = {"n": f.height, "mean_p": float(p.mean()), "mean_w": float(w.mean()),
                    "z_w": float((w - p).mean() / math.sqrt(nw_var(w - p, 24) / len(p))),
                    "mean_p_flag": float(pf.mean()), "flag_rate": float(fl.mean()),
                    "z_flag": float((fl - pf).mean() / math.sqrt(nw_var(fl - pf, 24) / len(p)))}
    save_json("field_own.json", own)
    log("score done")


# ============================================================================ stage: density (H-WS05-3)


def _dens(kind, th, z):
    if kind == "gauss":
        s = math.exp(th[0])
        return stats.norm.logpdf(z, 0, s)
    if kind == "t":
        s, nu = math.exp(th[0]), 2.05 + math.exp(th[1])
        return stats.t.logpdf(z, nu, 0, s)
    if kind == "merton":
        s, p, mj, sj = math.exp(th[0]), special.expit(th[1]), th[2], math.exp(th[3])
        return np.logaddexp(np.log1p(-p) + stats.norm.logpdf(z, 0, s), np.log(p) + stats.norm.logpdf(z, mj, math.sqrt(s * s + sj * sj)))
    if kind == "kou":
        s, p, pu, eu, ed = math.exp(th[0]), special.expit(th[1]), special.expit(th[2]), math.exp(th[3]), math.exp(th[4])
        # N(0, s^2) + double exponential (rates eu, ed in z units): EMG densities
        def emg(x, eta):
            return np.log(eta) + 0.5 * (eta * s) ** 2 - eta * x + special.log_ndtr(x / s - eta * s)
        lj = np.logaddexp(np.log(pu) + emg(z, eu), np.log1p(-pu) + emg(-z, ed))
        return np.logaddexp(np.log1p(-p) + stats.norm.logpdf(z, 0, s), np.log(p) + lj)
    raise ValueError(kind)


X0 = {"gauss": [0.0], "t": [0.0, 1.0], "merton": [0.0, -3.0, 0.0, 1.0], "kou": [0.0, -3.0, 0.0, math.log(0.4), math.log(0.4)]}


def fit_dens(kind, z):
    f = lambda th: -float(np.sum(_dens(kind, th, z)))
    r = optimize.minimize(f, X0[kind], method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-2})
    r = optimize.minimize(f, r.x, method="L-BFGS-B")
    return r.x


def stage_density() -> None:
    b = pl.read_parquet(OUT / "btc5m_bars.parquet").filter(pl.col("z").is_finite())
    ts, z = b["ts"].to_numpy(), b["z"].to_numpy()
    yr = (ts // D_MS)
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    kinds = ("gauss", "t", "merton", "kou")
    rng = np.random.default_rng(0)
    LS = {k: np.full(len(z), np.nan) for k in kinds}
    params = {}
    for Y in years:
        t0 = int(datetime(Y, 1, 1, tzinfo=UTC).timestamp() * 1000)
        t1 = CUTOFF_MS if Y == 2026 else int(datetime(Y + 1, 1, 1, tzinfo=UTC).timestamp() * 1000)
        tr = np.flatnonzero(ts < t0)
        if len(tr) > 100_000:
            tr = np.sort(rng.choice(tr, 100_000, replace=False))
        te = (ts >= t0) & (ts < t1)
        for k in kinds:
            th = fit_dens(k, z[tr])
            params[f"{k}_{Y}"] = th.tolist()
            LS[k][te] = _dens(k, th, z[te])
        log("density", Y)
    tr = np.flatnonzero(ts < CUTOFF_MS)
    tr = np.sort(rng.choice(tr, min(len(tr), 100_000), replace=False))
    te = ts >= CUTOFF_MS
    for k in kinds:
        th = fit_dens(k, z[tr])
        params[f"{k}_cutoff"] = th.tolist()
        LS[k][te] = _dens(k, th, z[te])
    out = {"params": params}
    sg = seg(ts)
    for sgn in SEGS:
        m = (sg == sgn) & np.isfinite(LS["kou"])
        o = {"n": int(m.sum()), "mean_ls": {k: float(np.nanmean(LS[k][m])) for k in kinds}}
        o["dm"] = {f"{a}-{c}": dm_hln(LS[a][m] - LS[c][m], 288) for a, c in [("kou", "t"), ("kou", "gauss"), ("kou", "merton"), ("t", "gauss"), ("merton", "gauss")]}
        # PIT tails of each model: share beyond the model's 0.1% / 99.9% quantiles is reported via exceedances of |z| > 5
        o["tail_obs_absz_gt5"] = float(np.mean(np.abs(z[m]) > 5))
        out[sgn] = o
    th = params["kou_cutoff"]
    out["kou_cutoff_readable"] = {"s": math.exp(th[0]), "p": float(special.expit(th[1])), "p_up": float(special.expit(th[2])),
                                  "mean_up": 1 / math.exp(th[3]), "mean_down": 1 / math.exp(th[4])}
    tt = params["t_cutoff"]
    out["t_cutoff_readable"] = {"s": math.exp(tt[0]), "nu": 2.05 + math.exp(tt[1])}
    save_json("density.json", out)
    log("density done")


# ============================================================================ stage: liq (H-WS05-10, -11, -12)


LAG_BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 240)]


def pool_minutes():
    """Per symbol 1m arrays on a common minute grid: jump (LM 1m), valid, log f, log sig, n3, liquidation cluster flag."""
    from ai_trader.features.flow_anomaly import liquidation_clusters, liquidation_minutes

    m1 = flow1m()
    t0, t1 = int(m1["minute"].min()), int(m1["minute"].max())
    grid = np.arange(t0, t1 + 1, dtype=np.int64)
    T, C = len(grid), len(POOL)
    Y, V, LF, LS, LQ, SIDE = (np.zeros((T, C)) for _ in range(6))
    Rr = np.full((T, C), np.nan)
    from ai_trader.storage.lake import Lake
    lake = Lake()
    liq = lake.read_asof("liquidations", datetime(2026, 11, 26, 23, 59, tzinfo=UTC), symbols=POOL, filters={"venue": "bybit"})
    liqm = liquidation_minutes(liq)
    liq_start = liq["event_ts"].min()
    for c, sym in enumerate(POOL):
        s = lm_detect(flow_series(sym, m1, 1))
        i = (s.ts // M_MS - t0).astype(np.int64)
        Y[i, c] = s.jump
        V[i, c] = np.isfinite(s.z)
        LF[i, c] = np.log(s.f)
        LS[i, c] = np.log(s.sig)
        Rr[i, c] = s.r
        d = m1.filter(pl.col("symbol") == sym)
        g = pl.DataFrame({"symbol": [sym] * T, "event_ts": pl.Series(grid * M_MS).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")})
        n3 = pl.DataFrame({"event_ts": pl.Series(d["minute"].to_numpy() * M_MS).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"), "n3s": d["n3"]})
        g = g.join(n3, on="event_ts", how="left")
        lm_ = liqm.filter(pl.col("symbol") == sym).with_columns(pl.col("event_ts").cast(pl.Datetime("ms", "UTC")))
        g = g.join(lm_.drop("symbol"), on="event_ts", how="left")
        cl = liquidation_clusters(g, liq_start=liq_start)
        LQ[:, c] = cl["liq_cluster"].fill_null(False).to_numpy()
        SIDE[:, c] = np.where(cl["liq_side"].to_numpy() == "long", 1.0, -1.0)
    V = V * np.isfinite(LF) * np.isfinite(LS)
    LF, LS = np.nan_to_num(LF), np.nan_to_num(LS)
    return grid, Y, V, LF, LS, LQ, SIDE, Rr


def episodes(lq: np.ndarray, gap: int = 10) -> list[tuple[int, int]]:
    idx = np.flatnonzero(lq > 0)
    if len(idx) == 0:
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b <= gap:
            b = i
        else:
            out.append((a, b))
            a = b = i
    out.append((a, b))
    return out


def stage_liq() -> None:
    grid, Y, V, LF, LS, LQ, SIDE, Rr = pool_minutes()
    T, C = Y.shape
    np.savez_compressed(TMP / "pool_minutes.npz", grid=grid, Y=Y, V=V, LF=LF, LS=LS, LQ=LQ, SIDE=SIDE, Rr=Rr)
    res = {"n_minutes": T, "symbols": POOL}
    # ---------- baseline cells (hour factor tercile x spot-vol tercile per symbol), excluding post-episode windows
    eps = {c: episodes(LQ[:, c]) for c in range(C)}
    excl = np.zeros((T, C), bool)
    for c, E in eps.items():
        for a, b in E:
            excl[a: min(b + 241, T), c] = True
    base = np.zeros((T, C))
    for c in range(C):
        v = V[:, c] > 0
        qf = np.quantile(LF[v, c], [1 / 3, 2 / 3])
        qs = np.quantile(LS[v, c], [1 / 3, 2 / 3])
        cell = np.searchsorted(qf, LF[:, c]) * 3 + np.searchsorted(qs, LS[:, c])
        for k in range(9):
            m = v & (cell == k) & ~excl[:, c]
            base[cell == k, c] = (Y[m, c].sum() + 0.5) / (m.sum() + 1)
    def study(events: list[tuple[int, int]]):
        """events = (symbol c, t_end); obs and expected jump counts per lag bin."""
        obs = np.zeros((len(events), len(LAG_BINS)))
        exp_ = np.zeros_like(obs)
        for e, (c, t) in enumerate(events):
            for j, (a, b) in enumerate(LAG_BINS):
                sl = slice(t + 1 + a, min(t + 1 + b, T))
                v = V[sl, c] > 0
                obs[e, j] = Y[sl, c][v].sum()
                exp_[e, j] = base[sl, c][v].sum()
        return obs, exp_
    def ratios(obs, exp_, B=2000, seed=0):
        rng = np.random.default_rng(seed)
        n = len(obs)
        rr = obs.sum(0) / np.maximum(exp_.sum(0), 1e-12)
        if n < 3:
            return {"n": n}
        bi = rng.integers(0, n, (B, n))
        rb = obs[bi].sum(1) / np.maximum(exp_[bi].sum(1), 1e-12)
        lo, hi = np.quantile(rb, [0.025, 0.975], axis=0)
        return {"n": n, "bins": [f"({a},{b}]" for a, b in LAG_BINS], "ratio": rr.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                "obs": obs.sum(0).tolist(), "exp": exp_.sum(0).tolist()}
    ev_liq = [(c, b) for c, E in eps.items() for a, b in E]
    ev_liq_jump = [(c, b) for c, E in eps.items() for a, b in E if Y[max(a - 2, 0): b + 3, c].any()]
    ev_liq_nojump = [(c, b) for c, E in eps.items() for a, b in E if not Y[max(a - 2, 0): b + 3, c].any()]
    near = np.zeros((T, C), bool)
    for c, E in eps.items():
        for a, b in E:
            near[max(a - 10, 0): b + 11, c] = True
    jt = [(c, int(t)) for c in range(C) for t in np.flatnonzero((Y[:, c] > 0) & ~near[:, c])]
    # decluster jump events 30 min per symbol
    jd, last = [], {}
    for c, t in sorted(jt, key=lambda x: (x[0], x[1])):
        if t - last.get(c, -10 ** 9) > 30:
            jd.append((c, t))
            last[c] = t
    res["event_study"] = {"liq_all": ratios(*study(ev_liq)), "liq_with_jump": ratios(*study(ev_liq_jump)),
                          "liq_without_jump": ratios(*study(ev_liq_nojump)), "jump_without_liq": ratios(*study(jd))}
    res["n_episodes"] = len(ev_liq)
    res["episodes_per_symbol"] = {POOL[c]: len(E) for c, E in eps.items()}
    log("event study", res["event_study"]["liq_all"].get("ratio"))
    # ---------- marked panel Hawkes: self + liquidation + BTC cross terms
    off = np.log(np.maximum(base, 1e-7))
    vbar = np.array([LS[V[:, c] > 0, c].mean() for c in range(C)])
    X = np.stack([LS - vbar[None, :]], 2)
    btc = Y[:, [0]].repeat(C, 1)
    def panel(with_liq: bool, with_btc: bool = True):
        terms = [(Y, 0, np.zeros(C, int))]
        n_a, n_g = 1, 1
        if with_liq:
            terms.append((LQ, 1, np.ones(C, int)))
            n_a, n_g = 2, 2
        if with_btc:
            pidx = np.full(C, n_a)
            pidx[0] = -1
            terms.append((btc, 0, pidx))
            n_a += 1
        return HModel(Y, V, X, off, np.zeros(C, int), terms, 1, n_a, n_g)
    fits = {}
    for name, wl in (("liq", True), ("noliq", False)):
        m = panel(wl)
        x0 = np.zeros(m.n_par)
        x0[0] = -0.3
        x0[m.n_i + 1: m.n_i + 1 + m.n_a] = math.log(0.02)
        x0[-m.n_g:] = math.log(0.1)
        r = m.fit(x0)
        fits[name] = (m, r)
        log("panel", name, r.fun)
    m, r = fits["liq"]
    a, b, la, lb = m.split(r.x)
    H = num_hessian(m, r.x)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(r.x), np.nan)
    k_lb_liq = m.n_i + m.X.shape[2] + m.n_a + 1
    hl = math.log(2) / math.exp(lb[1])
    hl_ci = [math.log(2) / math.exp(lb[1] + 1.96 * se[k_lb_liq]), math.log(2) / math.exp(lb[1] - 1.96 * se[k_lb_liq])]
    lr = 2 * (fits["noliq"][1].fun - r.fun)
    gofs = {POOL[c]: gof(m.rescaled(r.x, c)) for c in (0, 1, 2)}
    res["marked_hawkes"] = {"alpha_self": math.exp(la[0]), "gamma_liq": math.exp(la[1]), "alpha_btc": math.exp(la[2]),
                            "half_life_self_min": math.log(2) / math.exp(lb[0]), "half_life_liq_min": hl, "half_life_liq_ci95": hl_ci,
                            "se_log_gamma": float(se[m.n_i + m.X.shape[2] + 1]), "LR_liq": lr, "LR_liq_p": 0.5 * float(stats.chi2.sf(max(lr, 0), 2)),
                            "beta_logvol": float(b[0]), "icpt": float(a[0]), "gof": gofs, "conv": bool(r.success)}
    # excess hazard after a cluster at lag L minutes relative to baseline (typical symbol): gamma * kappa^(L-1) / base
    kap = math.exp(-math.exp(lb[1]))
    mb = float(np.median(np.exp(off[V > 0])))
    res["marked_hawkes"]["excess_ratio_at_lag"] = {L: 1 + math.exp(la[1]) * kap ** (L - 1) / mb for L in (1, 5, 15, 30, 60, 90, 120)}
    log("marked", res["marked_hawkes"])
    # ---------- multivariate 4-component (BTC, ETH, SOL, alt basket) + co-jumps
    alt = (Y[:, 3:] * V[:, 3:]).max(1)
    YM = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], alt])
    VM = np.column_stack([V[:, 0], V[:, 1], V[:, 2], (V[:, 3:].sum(1) >= 0.8 * (C - 3)).astype(float)])
    LFM = np.column_stack([LF[:, 0]] * 4)
    XM = LFM[:, :, None]
    def mv(restrict: bool):
        terms = []
        k = 0
        for j in range(4):
            pidx = np.arange(k, k + 4)
            if restrict and j == 0:
                pidx[3] = -1
            terms.append((YM[:, j], 0, pidx))
            k += 4
        return HModel(YM, VM, XM, np.zeros_like(YM), np.arange(4), terms, 4, 16, 1)
    mres = {}
    for nm, rs in (("full", False), ("no_btc_to_alt", True)):
        m = mv(rs)
        x0 = np.zeros(m.n_par)
        x0[:4] = np.log(np.maximum(YM[VM > 0].mean(), 1e-4))
        x0[5:21] = math.log(0.01)
        x0[-1] = math.log(0.1)
        mres[nm] = (m, m.fit(x0))
    m, r = mres["full"]
    a, b, la, lb = m.split(r.x)
    A = np.exp(la).reshape(4, 4).T   # A[c, j]: effect of source j on component c
    lr = 2 * (mres["no_btc_to_alt"][1].fun - r.fun)
    names = ["BTC", "ETH", "SOL", "ALT"]
    res["multivariate"] = {"A": {f"{names[j]}->{names[c]}": float(A[c, j]) for c in range(4) for j in range(4)},
                           "half_life_min": math.log(2) / math.exp(lb[0]), "LR_btc_to_alt": lr, "LR_p": 0.5 * float(stats.chi2.sf(max(lr, 0), 1)),
                           "conv": bool(r.success)}
    # co-jumps: same-minute
    bj = (Y[:, 0] > 0) & (V[:, 0] > 0)
    ej = (Y[:, 1] > 0)
    nalt = Y[:, 3:].sum(1)
    def shares(shift_days=0):
        e = np.roll(ej, shift_days * 1440)
        na = np.roll(nalt, shift_days * 1440)
        return float(e[bj].mean()), float((na[bj] >= 3).mean())
    s0 = shares()
    plc = [shares(k) for k in (1, 2, 3, 4, 5)]
    n = int(bj.sum())
    res["cojumps"] = {"n_btc_jumps": n, "eth_same_min": s0[0], "ge3_alts_same_min": s0[1],
                      "placebo_eth": float(np.mean([p[0] for p in plc])), "placebo_ge3": float(np.mean([p[1] for p in plc])),
                      "eth_ci95": list(stats.binomtest(int(round(s0[0] * n)), n).proportion_ci()) if n else None,
                      "alt_ci95": list(stats.binomtest(int(round(s0[1] * n)), n).proportion_ci()) if n else None}
    # same-minute co-jump share across any two pool symbols vs placebo
    save_json("liq.json", res)
    log("liq done")


# ============================================================================ stage: stops (H-WS05-13, -17)


KS = (1, 2, 3, 4, 6, 8)
FEE = 0.00033 + 0.00055          # pegged maker entry (2.0 + 1.3 bp) + taker exit 5.5 bp
SIZING_EXTRA = 0.0002            # sizing budget's 2 bp slippage
TIERS = {"liquid": (200.0, 10.0), "mid": (100.0, 25.0)}


def sigma_1h(sym: str) -> pl.DataFrame:
    con = duck()
    g = (LAKE / "bars" / "venue=bybit" / "price_type=last" / "freq=1h").as_posix() + "/**/*.parquet"
    df = pl.from_arrow(con.sql(f"""SELECT epoch_ms(event_ts) ts, epoch_ms(close_ts) cts, close, epoch_ms(known_ts) k FROM read_parquet('{g}', hive_partitioning=true)
                                   WHERE symbol='{sym}' AND event_ts >= TIMESTAMP '2026-08-01'""").to_arrow_table())
    df = df.sort("k").unique("ts", keep="last").sort("ts")
    r = np.concatenate([[np.nan], np.diff(np.log(df["close"].to_numpy()))])
    v = np.empty(len(r))
    s2 = np.nanmean(r[1:73] ** 2)
    for i in range(len(r)):
        if np.isfinite(r[i]):
            s2 = 0.94 * s2 + 0.06 * r[i] ** 2
        v[i] = s2
    return pl.DataFrame({"origin_ms": df["cts"].to_numpy().astype(np.int64), "sig1h": np.sqrt(v)})


def stage_stops() -> None:
    f3 = pl.read_parquet(TMP / "flow3s_pool.parquet")
    pm = np.load(TMP / "pool_minutes.npz")
    grid, Y, LQ = pm["grid"], pm["Y"], pm["LQ"]
    Rr = pm["Rr"]
    con = duck()
    g = (LAKE / "bars" / "venue=bybit" / "price_type=last" / "freq=1h").as_posix() + "/**/*.parquet"
    tv = pl.from_arrow(con.sql(f"""SELECT symbol, sum(turnover) t FROM read_parquet('{g}', hive_partitioning=true)
                                   WHERE event_ts >= TIMESTAMP '2026-11-08' GROUP BY 1 ORDER BY 2 DESC""").to_arrow_table())
    rank = {s: i + 1 for i, s in enumerate(tv["symbol"].to_list())}
    rows = []
    for c, sym in enumerate(POOL):
        d = f3.filter(pl.col("symbol") == sym).sort("ts")
        ts = d["ts"].to_numpy()
        hi = d["high"].to_numpy().astype(float)
        lo = d["low"].to_numpy().astype(float)
        cl = d["close"].to_numpy().astype(float)
        op = d["open"].to_numpy().astype(float)
        sp = np.nan_to_num(d["sp"].to_numpy().astype(float), nan=np.nanmedian(d["sp"].to_numpy()))
        hi = np.where(np.isfinite(hi), hi, cl)
        lo = np.where(np.isfinite(lo), lo, cl)
        gap_before = np.concatenate([[True], np.diff(ts) > 6000])
        sg = sigma_1h(sym)
        tier = "liquid" if rank.get(sym, 99) <= 10 else "mid"
        cap, budget = TIERS[tier]
        mi = ((ts // M_MS) - grid[0]).astype(np.int64)
        jl = np.zeros(len(ts), bool)   # jump against a long in trigger or previous minute (down jump)
        js = np.zeros(len(ts), bool)
        lq = np.zeros(len(ts), bool)
        okm = (mi >= 1) & (mi < len(grid))
        md = np.sign(np.nan_to_num(Rr[:, c]))   # sign of the 1m log return of the jump minute
        yj = Y[:, c] > 0
        down = yj & (md < 0)
        up = yj & (md > 0)
        lqw = np.convolve(LQ[:, c], np.ones(5), "same") > 0
        jl[okm] = down[mi[okm]] | down[mi[okm] - 1]
        js[okm] = up[mi[okm]] | up[mi[okm] - 1]
        lq[okm] = lqw[mi[okm]]
        origins = sg.filter((pl.col("origin_ms") >= int(datetime(2026, 11, 8, tzinfo=UTC).timestamp() * 1000)) & (pl.col("origin_ms") <= ts[-1] - D_MS))
        for o_ms, s1 in zip(origins["origin_ms"].to_numpy(), origins["sig1h"].to_numpy()):
            i0 = int(np.searchsorted(ts, o_ms))
            if i0 == 0 or i0 >= len(ts) or ts[i0] - o_ms > 6000 or gap_before[i0]:
                continue
            P0 = cl[i0 - 1]
            i1 = int(np.searchsorted(ts, o_ms + D_MS))
            seg_lo = np.minimum.accumulate(lo[i0:i1])
            seg_hi = np.maximum.accumulate(hi[i0:i1])
            gaps = np.flatnonzero(gap_before[i0:i1])
            cens = gaps[0] if len(gaps) else i1 - i0
            for side in (1, -1):
                for k in KS:
                    d_ = k * s1
                    S = P0 * math.exp(-side * d_)
                    if side == 1:
                        j = int(np.searchsorted(-seg_lo, -S, side="left"))
                    else:
                        j = int(np.searchsorted(seg_hi, S, side="left"))
                    if j >= cens:
                        continue
                    t = i0 + j
                    hs = sp[t] / 2e4
                    if side == 1:
                        cen = (S - min(S, cl[t]) * (1 - hs)) / S
                        opt = hs
                        pes = (S - lo[t] * (1 - hs)) / S
                        gapped = bool(np.isfinite(op[t]) and op[t] < S and cl[t - 1] > S)
                        jt = jl[t]
                    else:
                        cen = (max(S, cl[t]) * (1 + hs) - S) / S
                        opt = hs
                        pes = (hi[t] * (1 + hs) - S) / S
                        gapped = bool(np.isfinite(op[t]) and op[t] > S and cl[t - 1] < S)
                        jt = js[t]
                    dfrac = 1 - math.exp(-d_)
                    N = min(cap, 1.30 / (dfrac + FEE + SIZING_EXTRA))
                    rows.append((sym, tier, int(o_ms), side, k, float(s1), int(ts[t]), cen * 1e4, opt * 1e4, pes * 1e4, gapped, bool(jt), bool(lq[t]),
                                 N, N * (dfrac + FEE + cen), budget))
        log("stops", sym, len(rows))
    df = pl.DataFrame(rows, schema=["symbol", "tier", "origin_ms", "side", "k", "sig1h", "trig_ms", "slip_c", "slip_o", "slip_p", "gapped", "jump",
                                    "liq", "notional", "loss_usd", "budget"], orient="row")
    df.write_parquet(OUT / "stops.parquet")
    res = {"n": df.height}
    for k in KS:
        dk = df.filter(pl.col("k") == k)
        o = {}
        for nm, sub in (("all", dk), ("jump", dk.filter(pl.col("jump"))), ("nonjump", dk.filter(~pl.col("jump"))), ("liq", dk.filter(pl.col("liq"))),
                        ("gapped", dk.filter(pl.col("gapped")))):
            if sub.height == 0:
                continue
            sc = sub["slip_c"].to_numpy()
            o[nm] = {"n": sub.height, "median_bp": float(np.median(sc)), "p90_bp": float(np.quantile(sc, 0.9)), "p99_bp": float(np.quantile(sc, 0.99)),
                     "max_bp": float(sc.max()), "mean_R": float(np.mean(sc / 1e4 / (k * sub["sig1h"].to_numpy()))),
                     "p99_R": float(np.quantile(sc / 1e4 / (k * sub["sig1h"].to_numpy()), 0.99)),
                     "share_over_budget": float(np.mean(sc > sub["budget"].to_numpy())),
                     "share_loss_gt_2usd": float(np.mean(sub["loss_usd"].to_numpy() > 2.0)),
                     "pess_median_bp": float(np.median(sub["slip_p"].to_numpy())), "pess_p99_bp": float(np.quantile(sub["slip_p"].to_numpy(), 0.99))}
        o["jump_share_of_triggers"] = float(dk["jump"].mean()) if dk.height else None
        res[f"k{k}"] = o
    # pair co-stops: same origin, side, k; triggers within 5 min
    pairs = []
    for (o_ms, side, k), g in df.group_by("origin_ms", "side", "k"):
        if g.height < 2:
            continue
        t = g["trig_ms"].to_numpy()
        L = g["loss_usd"].to_numpy()
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if abs(t[i] - t[j]) <= 5 * M_MS:
                    pairs.append((k, L[i] + L[j]))
    pr = np.array(pairs) if pairs else np.zeros((0, 2))
    res["pair_costops"] = {f"k{k}": {"n": int((pr[:, 0] == k).sum()), "p_gt_6usd": float(np.mean(pr[pr[:, 0] == k, 1] > 6)) if (pr[:, 0] == k).any() else None,
                                     "p_gt_4usd": float(np.mean(pr[pr[:, 0] == k, 1] > 4)) if (pr[:, 0] == k).any() else None,
                                     "max_usd": float(pr[pr[:, 0] == k, 1].max()) if (pr[:, 0] == k).any() else None} for k in KS}
    save_json("stops.json", res)
    log("stops done")


# ============================================================================ stage: depend (WS06 core: H-WS05-14, -15, -16)


def focus_bars(freq: str) -> pl.DataFrame:
    con = duck()
    g = (LAKE / "bars" / "venue=bybit" / "price_type=last" / f"freq={freq}").as_posix() + "/**/*.parquet"
    syms = ",".join(f"'{s}'" for s in FOCUS)
    df = pl.from_arrow(con.sql(f"""SELECT symbol, epoch_ms(event_ts) ts, high, low, close, epoch_ms(known_ts) k FROM read_parquet('{g}', hive_partitioning=true)
                                   WHERE symbol IN ({syms}) AND event_ts >= TIMESTAMP '2021-01-01'""").to_arrow_table())
    return df.sort("k").unique(["symbol", "ts"], keep="last").sort("symbol", "ts")


def wide(df: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    w = df.pivot(on="symbol", index="ts", values=col).sort("ts")
    return w["ts"].to_numpy(), np.column_stack([w[s].to_numpy().astype(float) if s in w.columns else np.full(w.height, np.nan) for s in FOCUS])


def ewma_sig(r: np.ndarray, lam: float = 0.94, init: int = 100) -> np.ndarray:
    """sigma_{t|t-1} per column (uses returns up to t-1)."""
    T, N = r.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        x = r[:, j]
        ok = np.flatnonzero(np.isfinite(x))
        if len(ok) < init + 2:
            continue
        s2 = np.mean(x[ok[:init]] ** 2)
        for i in range(ok[init], T):
            out[i, j] = math.sqrt(s2)
            if np.isfinite(x[i]):
                s2 = lam * s2 + (1 - lam) * x[i] ** 2
    return out


def corr_ll(e: np.ndarray, R: np.ndarray) -> np.ndarray:
    sgn, ld = np.linalg.slogdet(R)
    q = np.einsum("ti,ti->t", e, np.linalg.solve(R, e[..., None])[..., 0])
    return -0.5 * (ld + q)


def normalize(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.einsum("tii->ti", Q))
    return Q / (d[:, :, None] * d[:, None, :])


def q_filter(P: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    """Q_t = (1-a-b) Qbar + a P_{t-1} + b Q_{t-1}; P = e e' (T, N, N)."""
    T, N, _ = P.shape
    Z = signal.lfilter([0.0, a], [1.0, -b], (P - Qbar).reshape(T, N * N), axis=0,
                       zi=np.zeros((1, N * N)))[0].reshape(T, N, N)
    return Z + Qbar


def copula_ll(u: np.ndarray, R: np.ndarray, nu: float | None) -> np.ndarray:
    if nu is None:
        q = stats.norm.ppf(u)
        Ri = np.linalg.inv(R)
        _, ld = np.linalg.slogdet(R)
        return -0.5 * ld - 0.5 * np.einsum("ti,ij,tj->t", q, Ri - np.eye(len(R)), q)
    q = stats.t.ppf(u, nu)
    d = len(R)
    _, ld = np.linalg.slogdet(R)
    Ri = np.linalg.inv(R)
    m = np.einsum("ti,ij,tj->t", q, Ri, q)
    lj = special.gammaln((nu + d) / 2) - special.gammaln(nu / 2) - d / 2 * math.log(nu * math.pi) - 0.5 * ld - (nu + d) / 2 * np.log1p(m / nu)
    lm = np.sum(stats.t.logpdf(q, nu), 1)
    return lj - lm


def ecdf_u(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(train)
    s = np.sort(train)
    return np.clip((np.searchsorted(s, x, side="right")) / (n + 1), 0.5 / (n + 1), 1 - 0.5 / (n + 1))


def stage_depend() -> None:
    res = {}
    b1 = focus_bars("1h")
    ts, C = wide(b1, "close")
    _, Hh = wide(b1, "high")
    _, Ll = wide(b1, "low")
    r = np.vstack([np.full((1, len(FOCUS)), np.nan), np.diff(np.log(C), axis=0)])
    sig = ewma_sig(r)
    e = r / sig
    okall = np.all(np.isfinite(e), 1)
    first = int(np.flatnonzero(okall)[0])
    log("depend 1h rows", okall.sum(), "first", datetime.fromtimestamp(ts[first] / 1000, UTC))
    # ---------- tail co-exceedance (empirical vs Gaussian vs t-copula implied), per segment, BTC-alt pairs
    sg = seg(ts)
    tails = {}
    for sgn in SEGS:
        m = okall & (sg == sgn)
        E = e[m]
        U = (stats.rankdata(E, axis=0)) / (len(E) + 1)
        out = {}
        for q in (0.05, 0.01):
            emp, gau, tco = [], [], []
            for j in range(1, len(FOCUS)):
                ua, ub = U[:, 0], U[:, j]
                emp.append(np.mean((ua < q) & (ub < q)) / q)
                rho = np.corrcoef(stats.norm.ppf(ua), stats.norm.ppf(ub))[0, 1]
                z = stats.norm.ppf(q)
                gau.append(stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([z, z]) / q)
                tau = stats.kendalltau(ua[::5], ub[::5]).statistic
                rt = math.sin(math.pi * tau / 2)
                best = None
                for nu in (2.5, 3, 4, 5, 6, 8, 12, 20, 40):
                    ll = copula_ll(np.column_stack([ua, ub]), np.array([[1, rt], [rt, 1]]), nu).sum()
                    if best is None or ll > best[0]:
                        best = (ll, nu)
                nu = best[1]
                zt = stats.t.ppf(q, nu)
                tco.append(stats.multivariate_t([0, 0], [[1, rt], [rt, 1]], df=nu).cdf([zt, zt]) / q)
            out[f"q{q}"] = {"emp_lower": np.mean(emp), "gauss_lower": np.mean(gau), "t_lower": np.mean(tco), "ratio_emp_gauss": np.mean(emp) / np.mean(gau),
                            "per_pair_emp": dict(zip(FOCUS[1:], np.round(emp, 3))), "per_pair_gauss": dict(zip(FOCUS[1:], np.round(gau, 3)))}
        # upper tail at 1% for asymmetry
        q = 0.01
        out["upper_q0.01_emp"] = float(np.mean([np.mean((U[:, 0] > 1 - q) & (U[:, j] > 1 - q)) / q for j in range(1, len(FOCUS))]))
        out["n"] = int(m.sum())
        tails[sgn] = out
    res["tails_1h"] = tails
    log("tails", {k: v["q0.01"]["ratio_emp_gauss"] for k, v in tails.items()})
    # ---------- copula walk-forward (yearly; holdout = after cutoff)
    blocks = [(2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)]
    edges = [(int(datetime(a, 1, 1, tzinfo=UTC).timestamp() * 1000), int(datetime(bq, 1, 1, tzinfo=UTC).timestamp() * 1000)) for a, bq in blocks]
    edges.append((int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000), CUTOFF_MS))
    edges.append((CUTOFF_MS, int(ts[-1]) + 1))
    dcop = np.full(len(ts), np.nan)
    cop_info = []
    for t0, t1 in edges:
        tr = okall & (ts < t0)
        te = okall & (ts >= t0) & (ts < t1)
        Etr, Ete = e[tr], e[te]
        Utr = np.column_stack([ecdf_u(Etr[:, j], Etr[:, j]) for j in range(len(FOCUS))])
        Ute = np.column_stack([ecdf_u(Etr[:, j], Ete[:, j]) for j in range(len(FOCUS))])
        Rg = np.corrcoef(stats.norm.ppf(Utr).T)
        sub = Utr[::7]
        Rt = np.eye(len(FOCUS))
        for i in range(len(FOCUS)):
            for j in range(i + 1, len(FOCUS)):
                Rt[i, j] = Rt[j, i] = math.sin(math.pi * stats.kendalltau(sub[:, i], sub[:, j]).statistic / 2)
        w, V_ = np.linalg.eigh(Rt)
        Rt = V_ @ np.diag(np.maximum(w, 1e-4)) @ V_.T
        Rt = Rt / np.sqrt(np.outer(np.diag(Rt), np.diag(Rt)))
        best = max(((copula_ll(sub, Rt, nu).sum(), nu) for nu in (3, 4, 5, 6, 8, 10, 15, 20, 30, 50)))
        nu = best[1]
        dcop[te] = copula_ll(Ute, Rt, nu) - copula_ll(Ute, Rg, None)
        cop_info.append({"test_start": t0, "nu": nu, "n_test": int(te.sum())})
    res["copula"] = {"fits": cop_info}
    for sgn in SEGS:
        m = np.isfinite(dcop) & (sg == sgn)
        res["copula"][sgn] = dm_hln(dcop[m], 24)
    log("copula", res["copula"])
    # ---------- DCC vs EWMA vs rolling (1h and 4h)
    dccres = {}
    for freq in ("1h", "4h"):
        if freq == "1h":
            tsf, Cf = ts, C
        else:
            b4 = focus_bars("4h")
            tsf, Cf = wide(b4, "close")
        rf = np.vstack([np.full((1, len(FOCUS)), np.nan), np.diff(np.log(Cf), axis=0)])
        ef = rf / ewma_sig(rf)
        ok = np.all(np.isfinite(ef), 1)
        i0 = int(np.flatnonzero(ok)[0])
        tsf, ef, ok = tsf[i0:], ef[i0:], ok[i0:]
        efz = np.where(np.isfinite(ef), ef, 0.0)
        P = np.einsum("ti,tj->tij", efz, efz)
        W = 720 if freq == "1h" else 180
        lam = 0.97
        Qe = q_filter(P, 1 - lam, lam, np.zeros((len(FOCUS), len(FOCUS)))) + lam ** np.arange(len(tsf))[:, None, None] * np.eye(len(FOCUS))
        Re = normalize(Qe)
        cs = np.concatenate([np.zeros((1, len(FOCUS), len(FOCUS))), np.cumsum(P, 0)])
        idx = np.arange(len(tsf))
        lo = np.maximum(idx - W, 0)
        Qr = (cs[idx] - cs[lo]) / np.maximum(idx - lo, 1)[:, None, None]
        Qr[idx < W] = np.eye(len(FOCUS))
        Rr_ = normalize(Qr + 1e-9 * np.eye(len(FOCUS)))
        sgf = seg(tsf)
        ll_e, ll_r = corr_ll(efz, Re), corr_ll(efz, Rr_)
        ll_d = np.full(len(tsf), np.nan)
        fits = []
        for t0, t1 in edges:
            tr = ok & (tsf < t0) & (idx >= W)
            te = ok & (tsf >= t0) & (tsf < t1)
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            Qbar = np.corrcoef(efz[tr].T)
            def nll(th):
                a_, b_ = special.expit(th[0]) * 0.2, special.expit(th[1])
                if a_ + b_ >= 0.999:
                    return 1e12
                R = normalize(q_filter(P, a_, b_, Qbar))
                return -float(corr_ll(efz[tr], R[tr]).sum())
            rr = optimize.minimize(nll, [special.logit(0.02 / 0.2), special.logit(0.95)], method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-3, "fatol": 1.0})
            a_, b_ = special.expit(rr.x[0]) * 0.2, special.expit(rr.x[1])
            R = normalize(q_filter(P, a_, b_, Qbar))
            ll_d[te] = corr_ll(efz[te], R[te])
            fits.append({"test_start": t0, "a": a_, "b": b_})
            log("dcc", freq, t0, a_, b_)
        o = {"fits": fits}
        L = 24 if freq == "1h" else 6
        for sgn in SEGS:
            m = np.isfinite(ll_d) & (sgf == sgn) & ok & (idx >= W)
            o[sgn] = {"n": int(m.sum()), "dcc-ewma": dm_hln(ll_d[m] - ll_e[m], L), "dcc-rolling": dm_hln(ll_d[m] - ll_r[m], L),
                      "ewma-rolling": dm_hln(ll_e[m] - ll_r[m], L)}
        dccres[freq] = o
        if freq == "1h":
            R30 = Rr_   # 30-day rolling correlation at each hour (for ENB)
            ts_R = tsf
    res["dcc"] = dccres
    # ---------- ENB and co-stopping (4h entries, 2 x sigma_4h stops, 72h cap, 1h high/low path)
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label
    fb = label(market_frame("BTCUSDT", "4h", datetime(2026, 11, 26, 23, 59, tzinfo=UTC), funding=False, oi=False), 4.0)
    cells = dict(zip(fb["close_ts"].dt.epoch("ms").to_list(), fb["cell"].to_list()))
    b4 = focus_bars("4h")
    t4, C4 = wide(b4, "close")
    r4 = np.vstack([np.full((1, len(FOCUS)), np.nan), np.diff(np.log(C4), axis=0)])
    s4 = ewma_sig(r4)   # sigma_{t|t-1}; at the close of bar t we know sigma_{t+1|t}: use next row
    s4n = np.vstack([s4[1:], np.full((1, len(FOCUS)), np.nan)])
    pos1 = {int(v): i for i, v in enumerate(ts)}
    posR = {int(v): i for i, v in enumerate(ts_R)}
    stopt = {}
    entries = []
    for i4 in range(len(t4)):
        tc = int(t4[i4]) + 4 * H_MS          # close time of the 4h bar = entry time
        i1 = pos1.get(tc)                    # first 1h bar after entry
        if i1 is None or i1 + 72 > len(ts) or not np.all(np.isfinite(s4n[i4])) or not np.all(np.isfinite(C4[i4])):
            continue
        entries.append((i4, tc, i1))
    H72 = np.arange(72)
    hit = {}
    for side in (1, -1):
        T_ = np.full((len(entries), len(FOCUS)), np.inf)
        for n_, (i4, tc, i1) in enumerate(entries):
            P0 = C4[i4]
            S = P0 * np.exp(-side * 2 * s4n[i4])
            path = Ll[i1 + H72] if side == 1 else Hh[i1 + H72]
            crossed = (path <= S[None, :]) if side == 1 else (path >= S[None, :])
            crossed &= np.isfinite(path)
            anyc = crossed.any(0)
            T_[n_, anyc] = np.argmax(crossed, 0)[anyc]
        hit[side] = T_
    ent_cell = [cells.get(tc) for (_, tc, _) in entries]
    ent_ms = np.array([tc for (_, tc, _) in entries])
    def costop(idx_set):
        out = {}
        for grp, cs_ in (("calm", {"R1", "R4", "R5"}), ("stress", {"R6", "R8", "R9"}), ("all", None)):
            num = den = 0
            per_day = {}
            for side in (1, -1):
                T_ = hit[side][:, idx_set]
                for n_ in range(len(entries)):
                    if cs_ is not None and ent_cell[n_] not in cs_:
                        continue
                    t = T_[n_]
                    if not np.isfinite(t).any():
                        continue
                    allc = bool(np.all(np.isfinite(t)) and (t.max() - t.min()) <= 4)
                    num += allc
                    den += 1
                    dkey = ent_ms[n_] // D_MS
                    a, b = per_day.get(dkey, (0, 0))
                    per_day[dkey] = (a + allc, b + 1)
            out[grp] = (num, den, per_day)
        return out
    pairs = [(i, j) for i in range(len(FOCUS)) for j in range(i + 1, len(FOCUS))]
    triples = [(0, 1, j) for j in range(2, len(FOCUS))]
    rng = np.random.default_rng(0)
    cost = {}
    for nm, sets in (("pairs", pairs), ("btc_eth_triples", triples)):
        agg = {g: [0, 0, {}] for g in ("calm", "stress", "all")}
        for st_ in sets:
            o = costop(list(st_))
            for g in agg:
                agg[g][0] += o[g][0]
                agg[g][1] += o[g][1]
                for k, (a, b) in o[g][2].items():
                    x = agg[g][2].get(k, (0, 0))
                    agg[g][2][k] = (x[0] + a, x[1] + b)
        res_g = {g: {"p_all_within_4h": agg[g][0] / max(agg[g][1], 1), "n": agg[g][1]} for g in agg}
        # day-block bootstrap of the stress / calm ratio
        dc = agg["calm"][2]
        ds = agg["stress"][2]
        kc, ks_ = list(dc), list(ds)
        rat = []
        for _ in range(1000):
            sc = [dc[k] for k in rng.choice(kc, len(kc))] if kc else []
            ss = [ds[k] for k in rng.choice(ks_, len(ks_))] if ks_ else []
            pc = sum(a for a, _ in sc) / max(sum(b for _, b in sc), 1)
            ps = sum(a for a, _ in ss) / max(sum(b for _, b in ss), 1)
            rat.append(ps / pc if pc > 0 else np.nan)
        res_g["ratio_stress_calm"] = res_g["stress"]["p_all_within_4h"] / max(res_g["calm"]["p_all_within_4h"], 1e-9)
        res_g["ratio_ci95"] = np.nanquantile(rat, [0.025, 0.975]).tolist()
        cost[nm] = res_g
    res["costop"] = cost
    # ENB at 4h entries from the 30-day rolling correlation
    enb2, enb3 = [], []
    for (_, tc, _) in entries[::3]:
        k = posR.get(tc)
        if k is None:
            continue
        R = R30[k]
        for i, j in pairs:
            enb2.append(2 / (1 + R[i, j] ** 2))
        for t_ in triples:
            ev = np.linalg.eigvalsh(R[np.ix_(t_, t_)])
            enb3.append(ev.sum() ** 2 / np.sum(ev ** 2))
    res["enb"] = {"pairs_median": float(np.median(enb2)), "pairs_p10": float(np.quantile(enb2, 0.1)), "triples_median": float(np.median(enb3)),
                  "triples_p10": float(np.quantile(enb3, 0.1)), "n_pairs": len(enb2), "n_triples": len(enb3)}
    save_json("depend.json", res)
    log("depend done")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["extract", "jumps", "hawkes", "score", "density", "liq", "stops", "depend"])
    a = ap.parse_args(argv)
    fns = {"extract": stage_extract, "jumps": stage_jumps, "hawkes": stage_hawkes, "score": stage_score, "density": stage_density,
           "liq": stage_liq, "stops": stage_stops, "depend": stage_depend}
    for st in a.stages:   # sequential inside one governor slot
        t = time.time()
        try:
            fns[st]()
        except Exception as e:   # keep going: later stages may not depend on this one
            import traceback
            traceback.print_exc()
            log(f"{st} FAILED: {e!r}")
            continue
        log(f"{st} finished in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
