"""WMN studies: idiosyncratic jump detector, token / market event studies, decay of news-driven variance, Hawkes with
exogenous marks. Definitions are frozen in research/hypotheses_pending/WMN.yaml (header) and the diary entry
"WMN design frozen before results"; this module implements them. Point-in-time: every baseline, beta, sigma and hour
factor uses bars strictly before the bar it is applied to.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import optimize, special

from ai_trader.research import wmn_data as wd

BAR_MS = 300_000
DAY_MS = 86_400_000
LM_CRIT = 4.31
VOL_SURGE = 3.0
EPISODE_GAP_BARS = 12
BIPOWER_N = 288
BETA_DAYS = 7
HOUR_DAYS = 30
CUTOFF_MS = wd.ms(wd.CUTOFF)

PRECISE_GROUPS = {  # (source, category) -> analysis group
    ("bybit_ann", "listing_spot"): "bybit_listing_spot",
    ("bybit_ann", "listing_perp"): "bybit_listing_perp",
    ("bybit_ann", "delist_perp"): "bybit_delist",
    ("bybit_ann", "delist_spot"): "bybit_delist",
    ("bybit_ann", "maintenance"): "maintenance",
    ("okx_ann", "x_listing"): "x_listing",
    ("okx_ann", "x_delist"): "x_delist",
    ("okx_ann", "x_maintenance"): "maintenance",
    ("kucoin_ann", "x_listing"): "x_listing",
    ("kucoin_ann", "x_delist"): "x_delist",
    ("kucoin_ann", "x_maintenance"): "maintenance",
    ("hn", "hn_story"): "hn_story",
}
LOW_INFO = {("bybit_ann", "launch_promo"), ("bybit_ann", "contract_update"), ("bybit_ann", "other"), ("bybit_ann", "premarket"),
            ("okx_ann", "x_other"), ("kucoin_ann", "x_other")}


# ------------------------------------------------------------------------------------------------------ helpers
def seg_of(ts_ms) -> np.ndarray:
    return np.where(np.asarray(ts_ms) < CUTOFF_MS, "llm_in_sample", "clean_holdout")


def cluster_mean(y: np.ndarray, cl: np.ndarray) -> tuple[float, float, int, int]:
    """Mean, CR1 cluster-robust SE of the mean, n, number of clusters."""
    m = np.isfinite(y)
    y, cl = y[m], cl[m]
    n = y.size
    if n < 2:
        return (float(y.mean()) if n else float("nan")), float("nan"), n, 0
    mu = y.mean()
    u = y - mu
    _, inv = np.unique(cl, return_inverse=True)
    g = np.bincount(inv, weights=u)
    G = g.size
    if G < 2:
        return float(mu), float("nan"), n, G
    v = (g ** 2).sum() / n ** 2 * G / (G - 1)
    return float(mu), float(math.sqrt(v)), n, G


def ols_cluster(X: np.ndarray, y: np.ndarray, cl: np.ndarray, w: np.ndarray | None = None, chunk: int = 200_000):
    """Weighted OLS with CR1 cluster-robust covariance (chunked: memory O(chunk x k)). Returns beta, se, n, clusters."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    idx = np.nonzero(m)[0]
    n, k = idx.size, X.shape[1]
    w = np.ones(X.shape[0]) if w is None else np.asarray(w, float)
    XtX, Xty = np.zeros((k, k)), np.zeros(k)
    for a in range(0, n, chunk):
        ii = idx[a:a + chunk]
        Xc = X[ii].astype(np.float64)
        wc = w[ii]
        XtX += (Xc * wc[:, None]).T @ Xc
        Xty += (Xc * wc[:, None]).T @ y[ii].astype(np.float64)
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ Xty
    _, inv = np.unique(cl[idx], return_inverse=True)
    G = int(inv.max()) + 1 if n else 0
    S = np.zeros((G, k))
    for a in range(0, n, chunk):
        ii = idx[a:a + chunk]
        Xc = X[ii].astype(np.float64)
        u = (y[ii].astype(np.float64) - Xc @ b) * w[ii]
        gi = inv[a:a + chunk]
        for j in range(k):
            S[:, j] += np.bincount(gi, weights=Xc[:, j] * u, minlength=G)
    adj = G / (G - 1) * (n - 1) / max(n - k, 1) if G > 1 else float("nan")
    V = XtXi @ (S.T @ S) @ XtXi * adj
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), int(n), int(G)


def bh(pvals: list[float], q: float = 0.10) -> list[bool]:
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    ok = p[order] <= thr
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej.tolist()


def p_two(t: float) -> float:
    return float(special.erfc(abs(t) / math.sqrt(2))) if np.isfinite(t) else float("nan")


# ------------------------------------------------------------------------------------------------- bar loading
def load_bars(sym: str) -> pl.DataFrame | None:
    p = wd.BARS / f"{sym}.parquet"
    return pl.read_parquet(p) if p.is_file() else None


def returns(df: pl.DataFrame) -> pl.DataFrame:
    """log close-to-close return of consecutive bars (NaN across holes)."""
    return df.sort("ts").with_columns(
        pl.when(pl.col("ts").diff() == BAR_MS).then((pl.col("c").cast(pl.Float64).log()).diff()).otherwise(None).alias("r"))


# --------------------------------------------------------------------------------------------- detector (PIT)
def _daily_roll(vals: np.ndarray, days: int) -> np.ndarray:
    """Sum over the previous `days` rows (excluding the current row) along axis 0."""
    cs = np.cumsum(np.vstack([np.zeros((1,) + vals.shape[1:]), vals]), axis=0)
    out = np.zeros_like(vals, dtype=float)
    idx = np.arange(vals.shape[0])
    lo = np.maximum(idx - days, 0)
    out[:] = cs[idx] - cs[lo]
    return out


def detect(df: pl.DataFrame, btc: pl.DataFrame | None) -> pl.DataFrame:
    """Per-bar PIT idiosyncratic return e, hour factor f, spot sigma, z, turnover ratio and detector flags.

    btc None -> no market adjustment (used for BTC itself / indices)."""
    d = returns(df)
    if btc is not None:
        b = returns(btc).select("ts", pl.col("r").alias("rb"))
        d = d.join(b, on="ts", how="left")
    else:
        d = d.with_columns(pl.lit(0.0).alias("rb"))
    ts = d["ts"].to_numpy()
    r = d["r"].to_numpy().astype(float)
    rb = d["rb"].fill_null(float("nan")).to_numpy().astype(float)
    r = np.where(np.isfinite(r), r, np.nan)
    day = (ts // DAY_MS).astype(np.int64)
    d0 = day.min()
    di = day - d0
    nd = di.max() + 1
    hour = ((ts // 3_600_000) % 24).astype(np.int64)
    if btc is not None:
        ok = np.isfinite(r) & np.isfinite(rb)
        S = np.zeros((nd, 5))
        np.add.at(S, di[ok], np.column_stack([r[ok] * rb[ok], rb[ok] ** 2, r[ok], rb[ok], np.ones(ok.sum())]))
        R = _daily_roll(S, BETA_DAYS)
        n = R[:, 4]
        cov = R[:, 0] - R[:, 2] * R[:, 3] / np.maximum(n, 1)
        var = R[:, 1] - R[:, 3] ** 2 / np.maximum(n, 1)
        beta_d = np.where((n >= 500) & (var > 0), cov / np.where(var > 0, var, 1), np.nan)
        beta = beta_d[di]
        e = r - beta * np.where(np.isfinite(rb), rb, np.nan)
    else:
        beta = np.zeros_like(r)
        e = r.copy()
    ae = np.abs(e)
    okm = np.isfinite(ae)
    H = np.zeros((nd, 24))
    N = np.zeros((nd, 24))
    np.add.at(H, (di[okm], hour[okm]), ae[okm])
    np.add.at(N, (di[okm], hour[okm]), 1)
    H30, N30 = _daily_roll(H, HOUR_DAYS), _daily_roll(N, HOUR_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        mh = H30 / N30
        mall = H30.sum(1) / N30.sum(1)
        fr = mh / mall[:, None]
        ndays = _daily_roll((N.sum(1) > 0).astype(float)[:, None], HOUR_DAYS)[:, 0]
        f = (ndays[:, None] * np.where(np.isfinite(fr), fr, 1.0) + 3.0) / (ndays[:, None] + 3.0)
    fk = f[di, hour]
    fk = np.where(np.isfinite(fk) & (fk > 0.05), fk, 1.0)
    et = e / fk
    bp = np.abs(et) * np.abs(np.r_[np.nan, et[:-1]])
    bps = pl.Series(bp).rolling_mean(BIPOWER_N, min_samples=200).shift(1).to_numpy()
    sigma = np.sqrt(np.pi / 2 * bps)
    z = et / sigma
    tov = d["turnover"].to_numpy().astype(float)
    med = pl.Series(tov).rolling_median(BIPOWER_N, min_samples=200).shift(1).to_numpy()
    vr = tov / np.where(med > 0, med, np.nan)
    jp = np.abs(z) > LM_CRIT
    jd = jp & (vr >= VOL_SURGE)
    return pl.DataFrame({"ts": ts, "r": r, "rb": rb, "beta": beta, "e": e, "f": fk, "sigma": sigma, "z": z,
                         "vr": vr, "Dp": jp, "D": jd, "turnover": tov})


def episodes(det: pl.DataFrame, flag: str = "D") -> pl.DataFrame:
    """Merge detections within EPISODE_GAP_BARS of the previous one; episode known at its first bar's close."""
    ts = det["ts"].to_numpy()
    fl = det[flag].fill_null(False).to_numpy()
    idx = np.nonzero(fl)[0]
    starts, last = [], -10 ** 9
    for i in idx:
        if ts[i] - last > EPISODE_GAP_BARS * BAR_MS:
            starts.append(i)
        last = ts[i]
    s = np.array(starts, dtype=np.int64)
    if s.size == 0:
        return pl.DataFrame(schema={"ts": pl.Int64, "known_ts": pl.Int64, "z": pl.Float64, "e": pl.Float64, "vr": pl.Float64})
    return pl.DataFrame({"ts": ts[s], "known_ts": ts[s] + BAR_MS, "z": det["z"].to_numpy()[s], "e": det["e"].to_numpy()[s],
                         "vr": det["vr"].to_numpy()[s]})


def run_detector(symbols=wd.CORE, log=print) -> dict:
    """Detector over the core universe: writes det/<SYM>.parquet (compact) and episodes.parquet."""
    out = wd.OUT / "det"
    out.mkdir(parents=True, exist_ok=True)
    btc = load_bars("BTCUSDT")
    eps, stats = [], {}
    for sym in symbols:
        df = load_bars(sym)
        if df is None or df.height < 3000:
            continue
        det = detect(df, None if sym == "BTCUSDT" else btc)
        det.select("ts", pl.col("z").cast(pl.Float32), pl.col("beta").cast(pl.Float32)
                   ).write_parquet(out / f"{sym}.parquet", compression="zstd", compression_level=9)  # compact (quota)
        for flag in ("D", "Dp"):
            ep = episodes(det, flag).with_columns(pl.lit(sym).alias("symbol"), pl.lit(flag).alias("flag"))
            eps.append(ep)
        nb = int(np.isfinite(det["z"].to_numpy()).sum())
        stats[sym] = {"bars": nb, "D": int(det["D"].sum()), "Dp": int(det["Dp"].sum())}
        log(f"  det {sym}: {stats[sym]}")
    E = pl.concat(eps).sort("known_ts")
    E.write_parquet(wd.OUT / "episodes.parquet")
    return stats


def ew_index(symbols) -> pl.DataFrame:
    """Equal-weight alt index of 5m log returns (core alts with a return at that bar, >= 10 names), as a pseudo price
    series. Accumulated on a common 5m grid symbol by symbol (low memory)."""
    t0 = wd.ms(wd.START)
    t1 = max(int(pl.read_parquet(wd.BARS / f"{s}.parquet", columns=["ts"])["ts"].max()) for s in symbols if (wd.BARS / f"{s}.parquet").is_file())
    n = (t1 - t0) // BAR_MS + 1
    S, N, V = np.zeros(n), np.zeros(n, np.int32), np.zeros(n)
    for s in symbols:
        if s == "BTCUSDT":
            continue
        df = load_bars(s)
        if df is None:
            continue
        d = returns(df).drop_nulls("r")
        k = ((d["ts"].to_numpy() - t0) // BAR_MS).astype(np.int64)
        ok = (k >= 0) & (k < n)
        np.add.at(S, k[ok], d["r"].to_numpy()[ok])
        np.add.at(N, k[ok], 1)
        np.add.at(V, k[ok], d["turnover"].to_numpy().astype(float)[ok])
    m = N >= 10
    ts = t0 + BAR_MS * np.nonzero(m)[0]
    r = S[m] / N[m]
    return pl.DataFrame({"ts": ts.astype(np.int64), "c": np.exp(np.cumsum(r)).astype(np.float32), "turnover": V[m].astype(np.float32),
                         "n": N[m]})


# ------------------------------------------------------------------------------------------------ event table
def precise_token_events(ev: pl.DataFrame, include_low: bool = False) -> pl.DataFrame:
    """Explode token events to (event, symbol) rows with an analysis group; merge same-symbol events within 2 h."""
    rows = ev.filter(pl.col("scope") == "token").explode("symbols").rename({"symbols": "symbol"}).drop_nulls("symbol")
    g = [PRECISE_GROUPS.get((s, c)) or ("low_info" if include_low and (s, c) in LOW_INFO else None)
         for s, c in rows.select("source", "category").iter_rows()]
    rows = rows.with_columns(pl.Series("group", g, dtype=pl.Utf8)).drop_nulls("group")
    rows = rows.filter(pl.col("precision").str.starts_with("1ms") | (pl.col("precision") == "1s"))
    rows = rows.sort(["symbol", "known_ts"])
    keep, last = [], {}
    for sym, kt, grp in rows.select("symbol", "known_ts", "group").iter_rows():
        key = (sym, grp == "low_info")  # registered (precise) events are merged among themselves only
        lk = last.get(key)
        keep.append(lk is None or kt - lk > 2 * 3_600_000)
        if keep[-1]:
            last[key] = kt
    return rows.filter(pl.Series(keep))


def day_token_events(ev: pl.DataFrame) -> pl.DataFrame:
    rows = ev.filter((pl.col("scope") == "token") & (pl.col("precision") == "1d")).explode("symbols").rename({"symbols": "symbol"})
    return rows.drop_nulls("symbol").with_columns(pl.col("category").alias("group"))


# ------------------------------------------------------------------------------------------------- event study
@dataclass
class Window:
    ts: np.ndarray
    r: np.ndarray
    rb: np.ndarray
    tov: np.ndarray


def window_from(df: pl.DataFrame, btc: pl.DataFrame, a: int, b: int) -> Window | None:
    d = returns(df.filter((pl.col("ts") >= a - BAR_MS) & (pl.col("ts") < b)))
    if d.height < 100:
        return None
    bb = returns(btc.filter((pl.col("ts") >= a - BAR_MS) & (pl.col("ts") < b))).select("ts", pl.col("r").alias("rb"))
    d = d.join(bb, on="ts", how="left").filter(pl.col("ts") >= a)
    return Window(d["ts"].to_numpy(), d["r"].fill_null(float("nan")).to_numpy(), d["rb"].fill_null(float("nan")).to_numpy(),
                  d["turnover"].to_numpy().astype(float))


BINS = {"pre": (-2 * 12, 0), "h0_1": (0, 12), "h1_4": (12, 48), "h4_24": (48, 288), "h24_48": (288, 576), "h0_4": (0, 48),
        "h0_24": (0, 288), "h0_48": (0, 576), "h1_48": (12, 576)}


def event_metrics(w: Window, t0: int, market: bool = False) -> dict | None:
    """Metrics around an anchor t0 (ms). tau = 0 is the bar containing t0. market=True: no beta adjustment."""
    k0 = t0 // BAR_MS * BAR_MS
    rel = (w.ts - k0) // BAR_MS
    base = (rel >= -600) & (rel < -24)
    r, rb = w.r, w.rb
    okb = base & np.isfinite(r) & (market | np.isfinite(rb))
    if okb.sum() < 0.8 * 576:
        return None
    if market:
        beta = 0.0
    else:
        x, y = rb[okb], r[okb]
        vx = np.var(x)
        beta = float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else 0.0
    e = r - beta * np.where(np.isfinite(rb), rb, 0.0 if market else np.nan)
    eb = e[okb]
    bpv = np.pi / 2 * np.nanmean(np.abs(eb[1:]) * np.abs(eb[:-1]))
    if not np.isfinite(bpv) or bpv <= 0:
        return None
    tmed = np.nanmedian(w.tov[base])
    out = {"sigma_b": math.sqrt(bpv), "beta": beta, "tov_b": float(tmed)}
    for name, (a, b) in BINS.items():
        m = (rel >= a) & (rel < b)
        ee = e[m]
        nn = np.isfinite(ee).sum()
        need = 0.6 * (b - a)
        if nn < max(need, 1):
            out[f"ARV_{name}"] = out[f"AV_{name}"] = out[f"CAR_{name}"] = out[f"SCAR_{name}"] = float("nan")
            continue
        out[f"ARV_{name}"] = float(np.log(np.nanmean(ee ** 2) / bpv)) if np.nanmean(ee ** 2) > 0 else float("nan")
        tv = np.nanmean(w.tov[m])
        out[f"AV_{name}"] = float(np.log(tv / tmed)) if tmed > 0 and tv > 0 else float("nan")
        car = float(np.nansum(ee))
        out[f"CAR_{name}"] = car
        out[f"SCAR_{name}"] = car / (math.sqrt(bpv) * math.sqrt(nn))
    # post-shock squared standardised series for the decay study (k = 1..576), NaN where missing
    xs = np.full(577, np.nan)
    m = (rel >= 0) & (rel <= 576)
    xs[rel[m].astype(int)] = (e[m] ** 2) / bpv
    out["_x"] = xs
    ee_post = np.full(577, np.nan)
    ee_post[rel[m].astype(int)] = e[m] / math.sqrt(bpv)
    out["_z"] = ee_post
    return out


class Series:
    """Numpy view of one symbol's 5m bars (r, BTC r aligned, turnover) for fast window slicing."""

    def __init__(self, df: pl.DataFrame, btc: "Series | None"):
        d = returns(df)
        self.ts = d["ts"].to_numpy()
        self.r = d["r"].fill_null(float("nan")).to_numpy().astype(float)
        self.tov = d["turnover"].to_numpy().astype(float)
        if btc is None:
            self.rb = np.zeros_like(self.r)
        else:
            j = np.searchsorted(btc.ts, self.ts)
            j = np.clip(j, 0, btc.ts.size - 1)
            self.rb = np.where(btc.ts[j] == self.ts, btc.r[j], np.nan)

    def window(self, a: int, b: int) -> Window | None:
        i, k = np.searchsorted(self.ts, a), np.searchsorted(self.ts, b)
        if k - i < 100:
            return None
        return Window(self.ts[i:k], self.r[i:k], self.rb[i:k], self.tov[i:k])

    def covers(self, a: int, b: int) -> bool:
        return self.ts.size > 0 and self.ts[0] <= a and self.ts[-1] >= b


__all__ = ["detect", "episodes", "event_metrics", "cluster_mean", "ols_cluster", "bh", "precise_token_events",
           "run_detector", "ew_index", "window_from", "Window", "BINS"]


# ------------------------------------------------------------------------------------------- event-study driver
NOW_MS = None


def _cluster_day(ts_ms) -> np.ndarray:
    return (np.asarray(ts_ms) // DAY_MS).astype(np.int64)


def load_core_series(symbols=wd.CORE) -> dict[str, Series]:
    btc_df = load_bars("BTCUSDT")
    btc = Series(btc_df, None)
    out = {"BTCUSDT": btc}
    for s in symbols:
        if s == "BTCUSDT":
            continue
        df = load_bars(s)
        if df is not None and df.height > 3000:
            out[s] = Series(df, btc)
    return out


def _archive_series(sym: str, a: int, b: int, btc: Series, log=print) -> Series | None:
    """5m bars for a delisted perp rebuilt from the public trade archive for the UTC days covering [a, b)."""
    days = sorted({(a + i * DAY_MS) // DAY_MS for i in range(int((b - a) // DAY_MS) + 2)})
    frames = []
    for dd in days:
        if dd * DAY_MS >= b:
            continue
        day = wd.datetime.fromtimestamp(dd * 86400, wd.UTC).strftime("%Y-%m-%d")
        tr = wd.archive_trades(sym, day)
        if tr is not None and tr.height:
            frames.append(wd.trades_to_5m(tr))
    if not frames:
        return None
    df = pl.concat(frames).sort("ts").unique("ts", keep="first")
    return Series(df, btc)


def token_event_rows(ev: pl.DataFrame, inst: pl.DataFrame, start_ms: int, end_ms: int) -> pl.DataFrame:
    pt = precise_token_events(ev, include_low=True).with_columns(pl.col("known_ts").alias("anchor"))
    dy = day_token_events(ev).with_columns(pl.col("anchor_ts").alias("anchor"))
    cols = ["event_id", "source", "category", "group", "symbol", "known_ts", "anchor", "precision", "title"]
    rows = pl.concat([pt.select(cols), dy.select(cols)])
    st = dict(inst.select("symbol", "status").iter_rows())
    rows = rows.with_columns(pl.col("symbol").replace_strict(st, default="unknown").alias("status"))
    return rows.filter((pl.col("anchor") >= start_ms) & (pl.col("anchor") <= end_ms)).sort("anchor")


def run_event_study(log=print, delist_cap: int = 60, noncore_cap: int = 2500, seed: int = 20261127) -> dict:
    """Event and placebo metrics for token events (core continuous, non-core windows, delisted perps from archives),
    endogenous episodes (core) and market events. Writes es_token.parquet, es_x.npz (post-shock series)."""
    t_start0 = wd.time.time()
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    inst = pl.read_parquet(wd.OUT / "instruments.parquet")
    core = load_core_series()
    btc = core["BTCUSDT"]
    end_ms = int(btc.ts[-1])
    start_ms = wd.ms(wd.START) + 3 * DAY_MS
    rows = token_event_rows(ev, inst, start_ms, end_ms - 2 * 3_600_000).filter(pl.col("symbol") != "BTCUSDT")
    # all token-event anchors per symbol (any category) for placebo contamination checks
    allk: dict[str, np.ndarray] = {}
    for sym, g in rows.group_by("symbol"):
        allk[sym[0]] = np.sort(g["anchor"].to_numpy())
    rng = np.random.default_rng(seed)
    launch = dict(inst.select("symbol", "launch_ms").iter_rows())
    rows = rows.with_columns(pl.col("symbol").replace_strict(launch, default=0).alias("launch_ms"))
    is_core = rows["symbol"].is_in(list(core))
    # non-core symbols: only registered groups, and only when a full pre-event baseline can exist (listed >= 9 days before)
    ok_nc = (pl.col("group") != "low_info") & (pl.col("anchor") - 9 * DAY_MS >= pl.col("launch_ms"))
    rows = rows.filter(is_core | ok_nc)
    is_core = rows["symbol"].is_in(list(core))
    trading = (rows["status"] == "Trading") & ~is_core
    closed = (rows["status"] == "Closed") & (rows["group"] == "bybit_delist")
    sel_nc = np.nonzero(trading.to_numpy())[0]
    if sel_nc.size > noncore_cap:
        sel_nc = np.sort(rng.choice(sel_nc, noncore_cap, replace=False))
    sel_cl = np.nonzero(closed.to_numpy())[0]
    if sel_cl.size > delist_cap:
        sel_cl = np.sort(rng.choice(sel_cl, delist_cap, replace=False))
    use = np.zeros(rows.height, bool)
    use[np.nonzero(is_core.to_numpy())[0]] = True
    use[sel_nc] = True
    use[sel_cl] = True
    rows = rows.filter(pl.Series(use)).with_columns(
        pl.when(pl.col("symbol").is_in(list(core))).then(pl.lit("core")).when(pl.col("status") == "Trading")
        .then(pl.lit("window")).otherwise(pl.lit("archive")).alias("data"))
    log(f"token event rows: {rows.height} ({rows['data'].value_counts().to_dicts()})", flush=True)
    # prefetch non-core windows in parallel, one worker per symbol (cache files are per symbol)
    from concurrent.futures import ThreadPoolExecutor
    tasks: dict[str, list[int]] = {}
    for sym, t0, kind in rows.select("symbol", "anchor", "data").iter_rows():
        if kind == "window":
            tasks.setdefault(sym, []).extend([int(t0), int(t0) - 7 * DAY_MS])

    def pre(item):
        sym, ts_ = item
        for a0 in ts_:
            try:
                wd.window_bars(sym, a0)
            except Exception as e:  # noqa: BLE001
                log(f"  window {sym} {a0}: {e}")
        return sym

    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(pre, tasks.items())):
            if i % 50 == 0:
                log(f"  prefetched {i}/{len(tasks)} symbols, {wd.time.time() - t_start0:.0f} s", flush=True)
    out, X = [], []
    t_start = wd.time.time()
    for i, r in enumerate(rows.iter_rows(named=True)):
        sym, t0, kind = r["symbol"], int(r["anchor"]), r["data"]
        for plc in (0, 1):
            a0 = t0 - plc * 7 * DAY_MS
            if plc:
                ks = allk.get(sym, np.array([]))
                if ks.size and np.any(np.abs(ks - a0) < 2 * DAY_MS):
                    continue
            a, b = a0 - 50 * 3_600_000, a0 + 50 * 3_600_000
            ser = None
            if kind == "core":
                ser = core[sym]
            elif kind == "window":
                df = wd.window_bars(sym, a0)
                ser = Series(df, btc) if df.height > 100 else None
            else:
                ser = _archive_series(sym, a, b, btc, log)
            if ser is None:
                continue
            w = ser.window(a, b)
            if w is None:
                continue
            m = event_metrics(w, a0)
            if m is None:
                continue
            x = m.pop("_x")
            z = m.pop("_z")
            out.append({**{k: r[k] for k in ("event_id", "source", "category", "group", "symbol", "known_ts", "precision", "data")},
                        "anchor": a0, "placebo": bool(plc), **m})
            X.append(np.concatenate([x, z]).astype(np.float32))
        if i % 200 == 0:
            log(f"  {i}/{rows.height} events, {len(out)} rows, {wd.time.time() - t_start:.0f} s", flush=True)
    T = pl.DataFrame(out)
    T = T.with_columns(pl.Series("row", np.arange(T.height)))
    T.write_parquet(wd.OUT / "es_token.parquet")
    np.savez_compressed(wd.OUT / "es_token_x.npz", z=np.clip(np.vstack(X)[:, 577:], -6e4, 6e4).astype(np.float16))
    return {"rows": T.height, "events": int((~T["placebo"]).sum()), "placebos": int(T["placebo"].sum())}


def run_episode_study(log=print, flag: str = "D", max_per_symbol: int = 400, seed: int = 7) -> dict:
    """Event metrics for endogenous episodes on core alts (tau = 0 is the jump bar) and for BTC episodes (market=True).
    Episodes are labelled news-associated if a precise token event of the symbol has known_ts in [ep - 2h, ep + 30 min].
    Writes es_episodes.parquet and es_episodes_x.npz."""
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = precise_token_events(ev)
    pk: dict[str, np.ndarray] = {s[0]: np.sort(g["known_ts"].to_numpy()) for s, g in pt.group_by("symbol")}
    E = pl.read_parquet(wd.OUT / "episodes.parquet").filter(pl.col("flag") == flag)
    core = load_core_series()
    rng = np.random.default_rng(seed)
    out, X = [], []
    for sym, g in E.group_by("symbol"):
        sym = sym[0]
        if sym not in core:
            continue
        ser = core[sym]
        ts_all = g["ts"].to_numpy()
        zz = g["z"].to_numpy()
        idx = np.arange(ts_all.size)
        if idx.size > max_per_symbol:
            idx = np.sort(rng.choice(idx, max_per_symbol, replace=False))
        ks = pk.get(sym, np.array([], dtype=np.int64))
        for i in idx:
            t0 = int(ts_all[i])
            w = ser.window(t0 - 50 * 3_600_000, t0 + 50 * 3_600_000)
            if w is None:
                continue
            m = event_metrics(w, t0, market=(sym == "BTCUSDT"))
            if m is None:
                continue
            x, z = m.pop("_x"), m.pop("_z")
            near = bool(ks.size and np.any((ks >= t0 - 2 * 3_600_000) & (ks <= t0 + 30 * 60_000)))
            out.append({"symbol": sym, "anchor": t0, "z_jump": float(zz[i]), "news": near, "n_sym_eps": int(ts_all.size), **m})
            X.append(np.concatenate([x, z]).astype(np.float32))
        log(f"  episodes {sym}: {len(idx)} sampled of {ts_all.size}", flush=True)
    T = pl.DataFrame(out).with_columns(pl.Series("row", np.arange(len(out))))
    T.write_parquet(wd.OUT / "es_episodes.parquet")
    np.savez_compressed(wd.OUT / "es_episodes_x.npz", z=np.clip(np.vstack(X)[:, 577:], -6e4, 6e4).astype(np.float16))
    return {"rows": T.height, "news": int(T["news"].sum())}


def hn_market_bursts(ev: pl.DataFrame, k: int = 3, within_ms: int = 3_600_000, gap_ms: int = 6 * 3_600_000) -> np.ndarray:
    """Known times of HN market bursts: the k-th story on market terms within 1 h; next burst >= 6 h later."""
    t = np.sort(ev.filter((pl.col("source") == "hn") & (pl.col("scope") == "market"))["known_ts"].unique().to_numpy())
    out, last = [], -10 ** 15
    for i in range(k - 1, t.size):
        if t[i] - t[i - k + 1] <= within_ms and t[i] - last >= gap_ms:
            out.append(t[i])
            last = t[i]
    return np.array(out, dtype=np.int64)


def run_market_study(log=print) -> dict:
    """Market events (macro releases; HN market bursts; hacks >= 100 M USD at day resolution) on BTC and the EW alt index,
    with placebos 7 days earlier. Writes es_market.parquet and es_market_x.npz."""
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    btc = Series(load_bars("BTCUSDT"), None)
    idx_df = ew_index(wd.CORE)
    ew = Series(idx_df.select("ts", "c", "turnover"), None)
    start, end = wd.ms(wd.START) + 3 * DAY_MS, int(btc.ts[-1]) - 2 * 3_600_000
    mac = ev.filter(pl.col("category") == "macro_release")
    sets = {"macro_release": [(int(t), json.loads(x).get("kind", "")) for t, x in mac.select("known_ts", "extra").iter_rows()],
            "hn_burst": [(int(t), "") for t in hn_market_bursts(ev)],
            "hack_100m": [(int(t), "") for t in ev.filter((pl.col("source") == "defillama") & (pl.col("scope") == "market"))["anchor_ts"].to_list()]}
    out, X = [], []
    for grp, lst in sets.items():
        anchors = np.array(sorted(t for t, _ in lst))
        for t0, kind in lst:
            if not (start <= t0 <= end):
                continue
            for plc in (0, 1):
                a0 = t0 - plc * 7 * DAY_MS
                if plc and np.any(np.abs(anchors - a0) < 2 * DAY_MS):
                    continue
                for name, ser in (("BTC", btc), ("EW", ew)):
                    w = ser.window(a0 - 50 * 3_600_000, a0 + 50 * 3_600_000)
                    if w is None:
                        continue
                    m = event_metrics(w, a0, market=True)
                    if m is None:
                        continue
                    x, z = m.pop("_x"), m.pop("_z")
                    out.append({"group": grp, "kind": kind, "series": name, "anchor": a0, "known_ts": t0, "placebo": bool(plc), **m})
                    X.append(np.concatenate([x, z]).astype(np.float32))
    T = pl.DataFrame(out).with_columns(pl.Series("row", np.arange(len(out))))
    T.write_parquet(wd.OUT / "es_market.parquet")
    np.savez_compressed(wd.OUT / "es_market_x.npz", z=np.clip(np.vstack(X)[:, 577:], -6e4, 6e4).astype(np.float16))
    idx_df.write_parquet(wd.OUT / "ew_index.parquet")
    return {"rows": T.height, "by_group": T.group_by("group", "placebo").len().to_dicts()}


# ------------------------------------------------------------------------------------------------------- decay
TAU_MIN = 5.0 * np.arange(577)  # minutes after the shock bar start, k = 0..576


def m_exp(th: np.ndarray, tau: np.ndarray) -> np.ndarray:
    B, A, T = th
    return B + A * np.exp(-tau / T)


def m_pow(th: np.ndarray, tau: np.ndarray) -> np.ndarray:
    B, A, c, p = th
    return B + A * (1.0 + tau / c) ** (-p)


def _unpack(u: np.ndarray, spec: str) -> tuple[np.ndarray, float]:
    if spec == "exp":
        th = np.array([math.exp(u[0]), math.exp(u[1]), math.exp(u[2])])
    else:
        th = np.array([math.exp(u[0]), math.exp(u[1]), math.exp(u[2]), math.exp(u[3])])
    nu = 2.05 + math.exp(u[-1])
    return th, nu


def t_logpdf_scaled(z: np.ndarray, s2: np.ndarray, nu: float) -> np.ndarray:
    """log density of z with variance s2 under a unit-variance Student-t with nu dof."""
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log((nu - 2) * math.pi)
    return c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(z ** 2 / (s2 * (nu - 2)))


def _m_and_grad(u: np.ndarray, spec: str, tau: np.ndarray):
    """m(tau) and dm/du (rows: parameters in log space, excluding nu)."""
    if spec == "exp":
        B, A, T = np.exp(u[:3])
        g = np.exp(-tau / T)
        m = B + A * g
        J = np.vstack([np.full_like(tau, B), A * g, A * g * tau / T])
    else:
        B, A, c, p = np.exp(u[:4])
        base = 1.0 + tau / c
        g = base ** (-p)
        m = B + A * g
        J = np.vstack([np.full_like(tau, B), A * g, A * p * (tau / c) * base ** (-p - 1), -A * p * np.log(base) * g])
    return m, J


def decay_fit(Z: np.ndarray, spec: str, k0: int = 1, max_shocks: int = 4000, seed: int = 3) -> dict:
    """Pooled Student-t MLE of m(tau) = B + A g(tau) on post-shock standardised returns Z (n_shocks x 577), bars
    k >= k0; analytic gradient. At most max_shocks shocks (random subset) enter the fit."""
    if Z.shape[0] > max_shocks:
        Z = Z[np.sort(np.random.default_rng(seed).choice(Z.shape[0], max_shocks, replace=False))]
    tau = TAU_MIN[k0:]
    Zk = Z[:, k0:]
    ok = np.isfinite(Zk)
    z2 = np.where(ok, Zk, 0.0) ** 2
    nk = ok.sum(0).astype(float)
    N = nk.sum()
    npar = 3 if spec == "exp" else 4

    def f(u):
        m, J = _m_and_grad(u, spec, tau)
        if np.any(~np.isfinite(m)) or np.any(m <= 1e-8):
            return 1e10, np.zeros_like(u)
        nu = 2.05 + math.exp(u[-1])
        q = z2 / (m * (nu - 2))
        lq = np.where(ok, np.log1p(q), 0.0)
        C = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log((nu - 2) * math.pi)
        ll = C * N - 0.5 * np.sum(nk * np.log(m)) - (nu + 1) / 2 * lq.sum()
        r = np.where(ok, q / (1 + q), 0.0)
        Gk = -0.5 * nk / m + (nu + 1) / 2 * r.sum(0) / m
        g = np.zeros(npar + 1)
        g[:npar] = J @ Gk
        dC = 0.5 * special.digamma((nu + 1) / 2) - 0.5 * special.digamma(nu / 2) - 0.5 / (nu - 2)
        dnu = dC * N - 0.5 * lq.sum() + (nu + 1) / 2 * (r.sum() / (nu - 2))
        g[npar] = dnu * (nu - 2.05)
        return -ll / N, -g / N

    starts = ([0.0, 0.0, math.log(60.0), math.log(1.5)], [0.0, 1.0, math.log(15.0), math.log(1.0)]) if spec == "exp" else         ([0.0, 0.0, math.log(30.0), math.log(0.8), math.log(1.5)], [0.0, 1.0, math.log(5.0), math.log(1.5), math.log(1.0)])
    best = None
    for u0 in starts:
        r = optimize.minimize(f, np.array(u0), jac=True, method="L-BFGS-B",
                              bounds=[(-8, 8)] * (npar + 1), options={"maxiter": 1000})
        if best is None or r.fun < best.fun:
            best = r
    th = np.exp(best.x[:npar])
    nu = 2.05 + math.exp(best.x[-1])
    return {"spec": spec, "theta": th.tolist(), "nu": nu, "nll": float(best.fun), "n_obs": int(N),
            "n_shocks": int(Z.shape[0]), "converged": bool(best.success)}


def decay_scores(Z: np.ndarray, fit: dict, k0: int = 1) -> np.ndarray:
    """Per-shock mean log score over bars k >= k0 under a fitted spec."""
    f = m_exp if fit["spec"] == "exp" else m_pow
    s2 = f(np.array(fit["theta"]), TAU_MIN[k0:])
    Zk = Z[:, k0:]
    lp = t_logpdf_scaled(Zk, np.broadcast_to(s2, Zk.shape), fit["nu"])
    lp = np.where(np.isfinite(Zk), lp, np.nan)
    return np.nanmean(lp, axis=1)


def half_life(fit: dict) -> float:
    """Minutes until the excess A g(tau) is half its value at tau = 0."""
    th = fit["theta"]
    if fit["spec"] == "exp":
        return th[2] * math.log(2)
    return th[2] * (2 ** (1 / th[3]) - 1)


def decay_time(fit: dict, frac: float = 0.1) -> float:
    """Minutes until the excess falls to frac x the long-run level B (practical release time), capped at 1e5."""
    th = fit["theta"]
    B, A = th[0], th[1]
    target = frac * B
    if A <= target:
        return 0.0
    if fit["spec"] == "exp":
        return min(1e5, th[2] * math.log(A / target))
    return min(1e5, th[2] * ((A / target) ** (1 / th[3]) - 1))


def persistence_ratio(X: np.ndarray) -> float:
    """mean(x - 1) over (4h, 48h] / mean(x - 1) over (5 min, 1h], pooled over shocks."""
    late = np.nanmean(X[:, 49:577] - 1.0)
    early = np.nanmean(X[:, 1:13] - 1.0)
    return float(late / early) if early > 0 else float("nan")


def cluster_boot_ratio(Xa: np.ndarray, ca: np.ndarray, Xb: np.ndarray, cb: np.ndarray, B: int = 2000, seed: int = 1):
    """Day-cluster bootstrap of PR(a) / PR(b); returns ratio, 5% and 95% quantiles."""
    rng = np.random.default_rng(seed)
    ua, ia = np.unique(ca, return_inverse=True)
    ub, ib = np.unique(cb, return_inverse=True)
    ga = [np.nonzero(ia == j)[0] for j in range(ua.size)]
    gb = [np.nonzero(ib == j)[0] for j in range(ub.size)]
    base = persistence_ratio(Xa) / persistence_ratio(Xb)
    vals = []
    for _ in range(B):
        sa = np.concatenate([ga[j] for j in rng.integers(0, len(ga), len(ga))])
        sb = np.concatenate([gb[j] for j in rng.integers(0, len(gb), len(gb))])
        pa, pb = persistence_ratio(Xa[sa]), persistence_ratio(Xb[sb])
        vals.append(pa / pb)
    v = np.array(vals)
    v = v[np.isfinite(v)]
    return float(base), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)), int(v.size)


def dm_cluster(d: np.ndarray, cl: np.ndarray) -> dict:
    mu, se, n, G = cluster_mean(d, cl)
    return {"diff": mu, "se": se, "t": mu / se if se and se > 0 else float("nan"), "n": n, "clusters": G}


def decay_compare(Z: np.ndarray, anchors: np.ndarray, k0: int = 1) -> dict:
    """POW vs EXP: llm_in_sample by a 2-fold time split inside FIT; clean_holdout by fit on all FIT."""
    seg = seg_of(anchors)
    fit_i = np.nonzero(seg == "llm_in_sample")[0]
    hold_i = np.nonzero(seg == "clean_holdout")[0]
    res = {}
    if fit_i.size >= 40:
        med = np.median(anchors[fit_i])
        f1, f2 = fit_i[anchors[fit_i] < med], fit_i[anchors[fit_i] >= med]
        dd, cl = [], []
        for tr, te in ((f1, f2), (f2, f1)):
            fe, fp = decay_fit(Z[tr], "exp", k0), decay_fit(Z[tr], "pow", k0)
            dd.append(decay_scores(Z[te], fp, k0) - decay_scores(Z[te], fe, k0))
            cl.append(_cluster_day(anchors[te]))
        res["llm_in_sample"] = dm_cluster(np.concatenate(dd), np.concatenate(cl))
        fe, fp = decay_fit(Z[fit_i], "exp", k0), decay_fit(Z[fit_i], "pow", k0)
        res["fit_all"] = {"exp": fe, "pow": fp, "half_life_min": {"exp": half_life(fe), "pow": half_life(fp)},
                          "decay10_min": {"exp": decay_time(fe), "pow": decay_time(fp)}}
        if hold_i.size >= 10:
            d = decay_scores(Z[hold_i], fp, k0) - decay_scores(Z[hold_i], fe, k0)
            res["clean_holdout"] = dm_cluster(d, _cluster_day(anchors[hold_i]))
            res["holdout_fit"] = {"exp": decay_fit(Z[hold_i], "exp", k0), "pow": decay_fit(Z[hold_i], "pow", k0)}
    res["n"] = {"fit": int(fit_i.size), "holdout": int(hold_i.size)}
    return res


def mean_profile(X: np.ndarray) -> dict:
    """Mean excess-variance ratio by lag bucket (descriptive path for the report)."""
    edges = [(1, 2), (2, 4), (4, 7), (7, 13), (13, 25), (25, 49), (49, 97), (97, 193), (193, 289), (289, 577)]
    return {f"{5 * a}-{5 * (b - 1)}min": float(np.nanmean(X[:, a:b])) for a, b in edges}
