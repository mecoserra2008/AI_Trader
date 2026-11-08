"""Auction (volume-profile) and variance-ratio regime studies: H-AU1..H-AU4 and H-RG1 (research/hypotheses.yaml).

Data
  BTC  Bybit BTCUSDT 5m (2020-04 -> 2026-09): daily profiles (features.profile) + 1h decision bars aggregated from
       the same 5m bars (complete hours only), enriched with frame.enrich_bars / add_robust_vol.
  ETH  Binance ETHUSDT 1m (2023 -> 2026-07): same pipeline. Almost no holdout -> in-sample replication only.
  RG1  Bybit 1h, 23-symbol universe, market_frame(funding=False, oi=False).

Design (RESEARCH_PROTOCOL §3b): profiles, signals, strategy P&L and outcomes are computed on the FULL history;
events are assigned to a segment afterwards: `llm_in_sample` if the outcome is fully known before the knowledge
cutoff, `clean_holdout` if the decision is at/after it; the few events whose outcome straddles the cutoff are dropped.
sigma = sigma_risk_next (robust, bias-corrected EWMA; hourly) at the decision close. The first WARMUP_DAYS of each
hourly frame are skipped (vol model warm-up).

Definitions (fixed before results, transcribed from the registry; choices the registry leaves open are noted):
  day D = UTC calendar day; "prior VA" = profile of D-1 (known at the end of D-1). Only calendar-consecutive,
          complete profile days (>= 95% of bars) are used.
  H-AU1  first hour j in day D (both closes j-1, j inside day D) with two consecutive 1h closes > VAH (< VAL);
         outcome d * ln(C_{j+24}/C_j) / (sigma_j sqrt 24); one event per side per day; per-day mean, NW t over days.
  H-AU2  day opens (open of the 00:00 bar) above VAH (below VAL); first pair of consecutive closes in day D inside
         [VAL, VAH]; race on later 1h closes (cap RACE_CAP bars): target = opposite VA edge, barrier = re-entry edge
         -/+ 0.5 sigma. Driftless null b/(a+b), a = log distance to the target, b = to the barrier. NW t of hit - null.
  H-AU3  overlap = share of VA(D-1) inside VA(D-2) (features.profile.va_overlap); balance >= 0.70, imbalance <= 0.30.
         Daily net P&L of each strategy = sum of evaluate.strategy_pnl over rows whose close_ts falls on day D (the
         P&L earned during D). MR = mean(zscore_reversion, rsi_reversion), trend = mean(ema_stack, donchian_breakout).
         DiD = OLS slope of (MR - trend) on 1{balance} over balance + imbalance days, HAC (Bartlett, NW lag) t.
  H-AU4  days after a balance day: touched POC(D-1) within D (1h highs/lows of D); null 2(1 - Phi(d / sqrt 24)),
         d = |ln(C/POC)| / sigma at the last close of D-1. NW t over days of touch - null.
  H-RG1  decisions at each Monday 00:00 UTC close; VR from the trailing 168 1h log returns with Lo-MacKinlay unbiased
         variances (q = 24, overlapping 24h sums); outcome = next 168 bars' net P&L; diff = trend - MR;
         DiD = OLS slope of diff on 1{VR > 1.2} over VR > 1.2 or VR < 0.8 obs, SE clustered by week.

    uv run python -m ai_trader.research.auction_study --segment llm_in_sample [--record] [--out file.json]
    uv run python -m ai_trader.research.auction_study --segment clean_holdout [--record] [--out file.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from scipy.stats import norm

from ai_trader.core.time import now
from ai_trader.features.profile import daily_profiles, va_overlap
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.frame import add_robust_vol, enrich_bars, market_frame
from ai_trader.storage.lake import Lake
from ai_trader.strategies import REGISTRY

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
ASSETS = {"BTC": ("BTCUSDT", "5m", "bybit", 5), "ETH": ("ETHUSDT", "1m", "binance", 1)}
MR, TREND = ("zscore_reversion", "rsi_reversion"), ("ema_stack", "donchian_breakout")
SEGS = ("llm_in_sample", "clean_holdout")
BALANCE, IMBALANCE = 0.70, 0.30
VR_HI, VR_LO = 1.2, 0.8
RACE_CAP = 720
WARMUP_DAYS = 30
MIN_HOLDOUT = 8
H = 3600


# ------------------------------------------------------------------ helpers

def _hour(ts: pl.Series) -> np.ndarray:
    return (ts.dt.epoch("s").to_numpy() // H).astype(np.int64)


def _seg(end_ts: datetime | None, dec_ts: datetime, cut: datetime) -> str | None:
    if dec_ts >= cut:
        return "clean_holdout"
    if end_ts is not None and end_ts < cut:
        return "llm_in_sample"
    return None  # straddles the cutoff, or unresolved


def _nw(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m, t, n = newey_west_t(x)
    if n < 10 and n >= 3 and x.std(ddof=1) > 0:  # NW needs >= 10: fall back to the iid t (holdout only)
        t = float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
    return {"mean": m, "t": t, "n": n}


def _nw_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def ols_hac(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett) t-stats; rows must be time-ordered."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = X * (y - X @ b)[:, None]
    S = u.T @ u
    for lag in range(1, _nw_lag(n) + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1 - lag / (_nw_lag(n) + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, b / np.sqrt(np.diag(V))


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS coefficients and cluster-robust (CR1) t-stats; returns (b, t, n_clusters)."""
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


def did(y: np.ndarray, grp: np.ndarray, groups: np.ndarray | None = None) -> dict:
    """DiD of y between grp == 1 and grp == 0 (slope on the indicator), HAC or clustered t."""
    ok = np.isfinite(y)
    y, grp = y[ok], grp[ok].astype(float)
    n1, n0 = int(grp.sum()), int((1 - grp).sum())
    out = {"n_hi": n1, "n_lo": n0, "mean_hi": float(y[grp == 1].mean()) if n1 else float("nan"),
           "mean_lo": float(y[grp == 0].mean()) if n0 else float("nan")}
    if n1 < 2 or n0 < 2:
        return {**out, "did": out["mean_hi"] - out["mean_lo"], "t": float("nan")}
    X = np.c_[np.ones(len(y)), grp]
    if groups is None:
        b, t = ols_hac(y, X)
    else:
        b, t, g = ols_cluster(y, X, groups[ok])
        out["n_clusters"] = g
    return {**out, "did": float(b[1]), "t": float(t[1])}


# ------------------------------------------------------------------ data

def intraday(asset: str, as_of: datetime | None = None) -> pl.DataFrame:
    sym, freq, venue, _ = ASSETS[asset]
    return Lake().bars(sym, freq, as_of or now(), venue=venue).select("event_ts", "known_ts", "open", "high", "low", "close", "volume").sort("event_ts")


def hourly_from_intraday(b: pl.DataFrame, bar_minutes: int) -> pl.DataFrame:
    """Complete 1h bars from intraday bars + ret / sigma columns (frame.enrich_bars, add_robust_vol)."""
    need = 60 // bar_minutes
    h = (b.group_by(pl.col("event_ts").dt.truncate("1h").alias("event_ts"))
         .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
              pl.col("volume").sum(), pl.col("known_ts").max(), pl.len().alias("_n"))
         .filter(pl.col("_n") == need).drop("_n").sort("event_ts")
         .with_columns((pl.col("event_ts") + timedelta(hours=1)).alias("close_ts")))
    return add_robust_vol(enrich_bars(h))


def build(asset: str, as_of: datetime | None = None) -> dict:
    """Hourly frame, complete daily profiles and the per-day table (prior-day VA levels, overlap, bucket)."""
    sym, freq, venue, mins = ASSETS[asset]
    b = intraday(asset, as_of)
    prof = daily_profiles(b).filter(pl.col("n_bars") >= 0.95 * 1440 / mins)
    hf = hourly_from_intraday(b, mins)
    pd = {d.toordinal(): r for d, r in zip(prof["day"].to_list(), prof.iter_rows(named=True))}
    return {"asset": asset, "hf": hf, "profiles": prof, "prof_by_day": pd, "data": f"{venue} {sym} {freq} -> daily profiles (0.05% bins, 70% VA); 1h bars from {freq}"}


def _day_index(hf: pl.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    eh = _hour(hf["event_ts"])
    return eh, {int(h): i for i, h in enumerate(eh)}


def _epoch_day_to_ordinal(d: int) -> int:
    return d + datetime(1970, 1, 1).toordinal()


# ------------------------------------------------------------------ events

def events(ctx: dict) -> dict[str, pl.DataFrame]:
    hf, pdays = ctx["hf"], ctx["prof_by_day"]
    eh, pos = _day_index(hf)
    o, hi, lo, c = (hf[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    lc = np.log(c)
    sig = hf["sigma_risk_next"].to_numpy().astype(float)
    cts = hf["close_ts"].to_list()
    cut = knowledge_cutoff()
    first_day = int(eh[0] // 24) + WARMUP_DAYS
    days = sorted({int(x // 24) for x in eh})

    # strategy daily P&L by the day the P&L is earned (close_ts date)
    sigs = {s: REGISTRY[s]()(hf) for s in (*MR, *TREND)}
    pnl_hour = {s: strategy_pnl(hf, sigs[s])["pnl"].to_numpy() for s in sigs}

    au1, au2, au34 = [], [], []
    for D in days:
        if D < first_day:
            continue
        rows = [pos.get(D * 24 + k) for k in range(24)]
        if any(r is None for r in rows):
            continue
        p1, p2 = pdays.get(_epoch_day_to_ordinal(D - 1)), pdays.get(_epoch_day_to_ordinal(D - 2))
        if p1 is None:
            continue
        val, vah, poc = p1["val"], p1["vah"], p1["poc"]
        day_ts = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=D)
        # ---- H-AU1: acceptance outside value
        for d, beyond in ((1, c[rows] > vah), (-1, c[rows] < val)):
            js = np.flatnonzero(beyond[1:] & beyond[:-1])
            if not len(js):
                continue
            i = rows[int(js[0]) + 1]
            k = pos.get(int(eh[i]) + 24)
            if k is None or not (np.isfinite(sig[i]) and sig[i] > 0):
                continue
            seg = _seg(cts[k], cts[i], cut)
            if seg:
                au1.append({"day": D, "d": d, "event_ts": cts[i], "segment": seg, "sz": d * (lc[k] - lc[i]) / (sig[i] * np.sqrt(24))})
        # ---- H-AU2: failed auction outside value
        o0 = o[rows[0]]
        side = 1 if o0 > vah else (-1 if o0 < val else 0)
        if side:
            inside = (c[rows] >= val) & (c[rows] <= vah)
            js = np.flatnonzero(inside[1:] & inside[:-1])
            if len(js):
                i = rows[int(js[0]) + 1]
                s = sig[i]
                if np.isfinite(s) and s > 0:
                    if side == 1:  # opened above, re-entered at VAH: target VAL, barrier VAH + 0.5 sigma
                        t_lvl, b_lvl = np.log(val), np.log(vah) + 0.5 * s
                        a, bdist = lc[i] - t_lvl, b_lvl - lc[i]
                        path = lc[i + 1:i + 1 + RACE_CAP]
                        th, bh = np.flatnonzero(path <= t_lvl), np.flatnonzero(path >= b_lvl)
                    else:
                        t_lvl, b_lvl = np.log(vah), np.log(val) - 0.5 * s
                        a, bdist = t_lvl - lc[i], lc[i] - b_lvl
                        path = lc[i + 1:i + 1 + RACE_CAP]
                        th, bh = np.flatnonzero(path >= t_lvl), np.flatnonzero(path <= b_lvl)
                    tf, bf = (th[0] if len(th) else None), (bh[0] if len(bh) else None)
                    if a <= 0:
                        hit, end = 1, i
                    elif tf is None and bf is None:
                        hit, end = None, None
                    else:
                        hit = 1 if bf is None or (tf is not None and tf < bf) else 0
                        end = i + 1 + int(tf if hit else bf)
                    seg = _seg(cts[end] if end is not None else None, cts[i], cut)
                    au2.append({"day": D, "side": side, "event_ts": cts[i], "segment": seg, "resolved": hit is not None,
                                "hit": hit, "null": max(bdist, 0.0) / (max(a, 0.0) + max(bdist, 0.0)),
                                "a_sig": a / s, "b_sig": bdist / s, "bars": (end - i) if end is not None else None})
        # ---- H-AU3 / H-AU4 day table
        ov = float(va_overlap(np.array([val]), np.array([vah]), np.array([p2["val"]]), np.array([p2["vah"]]))[0]) if p2 else float("nan")
        j0 = pos.get(D * 24 - 1)  # the bar closing at 00:00 of D (decision close for D)
        row = {"day": D, "date": day_ts.date(), "overlap": ov,
               "segment": "llm_in_sample" if day_ts + timedelta(days=1) <= cut else ("clean_holdout" if day_ts >= cut else None)}
        # P&L earned during D: rows whose close_ts is in D, i.e. event hours 24D-1 .. 24D+22
        prow = [pos.get(D * 24 - 1 + k) for k in range(24)]
        for s in sigs:
            row[s] = float(sum(pnl_hour[s][r] for r in prow)) if all(r is not None for r in prow) else float("nan")
        if j0 is not None and np.isfinite(sig[j0]) and sig[j0] > 0:
            dist = abs(lc[j0] - np.log(poc)) / sig[j0]
            touched = hi[rows].max() >= poc if poc >= c[j0] else lo[rows].min() <= poc
            row.update(poc_dist_sig=dist, touch=float(touched), touch_null=float(2 * (1 - norm.cdf(dist / np.sqrt(24)))),
                       c0=float(c[j0]), s0=float(sig[j0]), val1=val, vah1=vah)
        au34.append(row)
    dt = pl.DataFrame(au34, infer_schema_length=None)
    dt = dt.with_columns(
        (pl.mean_horizontal(*MR) - pl.mean_horizontal(*TREND)).alias("mr_minus_trend"),
        pl.when(pl.col("overlap").is_nan()).then(pl.lit(None, dtype=pl.Utf8)).when(pl.col("overlap") >= BALANCE).then(pl.lit("balance")).when(pl.col("overlap") <= IMBALANCE).then(pl.lit("imbalance")).otherwise(pl.lit("middle")).alias("bucket"),
    )
    # row D's bucket classifies day D-1 (overlap of VA(D-1) with VA(D-2)): it is "the day following a balance day"
    # for H-AU4 and "the next day" for H-AU3.
    dt = dt.sort("day")
    return {"au1": pl.DataFrame(au1), "au2": pl.DataFrame(au2, infer_schema_length=None), "days": dt}


# ------------------------------------------------------------------ H-AU1 driftless null (correction)

BGK = 0.5826  # Broadie-Glasserman-Kou continuity correction for barriers monitored once per step (in step sigmas)


def au1_null_sim(days: pl.DataFrame, n_sims: int = 50, seed: int = 0) -> dict:
    """The H-AU1 statistic computed on driftless Gaussian random walks started at each day's real 00:00 close with
    the real prior-day VAH/VAL and sigma (48 hourly steps: day D + the 24h outcome window).

    Why: the per-day mean weights each event by 1 / (events that day). Whether a day has one or two events depends on
    the rest of the day's path, which overlaps the 24h outcome, so the per-day mean is NOT centred on 0 under a
    martingale (one-sided days are days the first acceptance did not reverse). Returns the null means of the per-day
    mean, of the event-weighted per-day sum and of the first-event-of-day outcome."""
    d = days.filter(pl.col("c0").is_not_null() & pl.col("s0").is_not_nan())
    if d.height == 0:
        return {"per_day_mean": float("nan"), "day_sum": float("nan"), "first_event": float("nan"), "n_sims": n_sims}
    rng = np.random.default_rng(seed)
    lc0, s0 = np.log(d["c0"].to_numpy()), d["s0"].to_numpy()
    lvah, lval = np.log(d["vah1"].to_numpy()), np.log(d["val1"].to_numpy())
    res = {"per_day_mean": [], "day_sum": [], "first_event": []}
    for _ in range(n_sims):
        path = lc0[:, None] + np.cumsum(rng.standard_normal((d.height, 48)), axis=1) * s0[:, None]  # closes k = 1..48
        day = path[:, :24]
        outs = []
        first_t = np.full(d.height, np.inf)
        first_v = np.full(d.height, np.nan)
        for sign, beyond in ((1, day > lvah[:, None]), (-1, day < lval[:, None])):
            pair = beyond[:, 1:] & beyond[:, :-1]
            has = pair.any(axis=1)
            j = np.argmax(pair, axis=1) + 1
            rows = np.flatnonzero(has)
            v = np.full(d.height, np.nan)
            v[rows] = sign * (path[rows, j[rows] + 24] - path[rows, j[rows]]) / (s0[rows] * np.sqrt(24))
            outs.append(v)
            earlier = has & (j < first_t)
            first_t[earlier], first_v[earlier] = j[earlier], v[earlier]
        o = np.c_[outs[0], outs[1]]
        k = np.isfinite(o).sum(axis=1)
        any_ = k > 0
        with np.errstate(invalid="ignore"):
            res["per_day_mean"].append(np.nanmean(o[any_], axis=1).mean())
            res["day_sum"].append(np.nansum(o[any_], axis=1).mean())
            res["first_event"].append(first_v[any_].mean())
    return {k: float(np.mean(v)) for k, v in res.items()} | {"n_sims": n_sims, "sd_per_day_mean_across_sims": float(np.std(res["per_day_mean"]))}


# ------------------------------------------------------------------ statistics per segment

def stats_au(ev: dict[str, pl.DataFrame], seg: str) -> dict:
    a1 = ev["au1"].filter(pl.col("segment") == seg).sort("event_ts")
    per_day = a1.group_by("day").agg(pl.col("sz").mean()).sort("day")["sz"].to_numpy()
    au1 = {**_nw(per_day), "n_events": a1.height,
           "up": _nw(a1.filter(pl.col("d") == 1)["sz"].to_numpy()), "down": _nw(a1.filter(pl.col("d") == -1)["sz"].to_numpy()),
           "p_positive": float((a1["sz"] > 0).mean()) if a1.height else float("nan")}
    # diagnostics (not registered, not used for the verdict): how the per-day mean depends on the weighting
    if a1.height:
        nside = a1.group_by("day").agg(pl.len().alias("_k"))
        a1k = a1.join(nside, on="day")
        au1["diag_event_weighted_day_sum"] = _nw(a1.group_by("day").agg(pl.col("sz").sum()).sort("day")["sz"].to_numpy())
        au1["diag_event_mean"] = float(a1["sz"].mean())
        au1["diag_first_event_of_day"] = _nw(a1.sort("event_ts").group_by("day", maintain_order=True).first().sort("day")["sz"].to_numpy())
        au1["diag_one_sided_days"] = _nw(a1k.filter(pl.col("_k") == 1).sort("event_ts")["sz"].to_numpy())
        au1["diag_two_sided_days_events"] = _nw(a1k.filter(pl.col("_k") == 2).sort("event_ts")["sz"].to_numpy())
    # correction: the registered per-day mean against its simulated driftless null (same SE)
    nul = au1_null_sim(ev["days"].filter(pl.col("segment") == seg))
    au1["null_sim"] = nul
    se = au1["mean"] / au1["t"] if np.isfinite(au1["t"]) and au1["t"] != 0 else float("nan")
    au1["excess_vs_null"] = au1["mean"] - nul["per_day_mean"]
    au1["t_excess_vs_null"] = au1["excess_vs_null"] / se if np.isfinite(se) else float("nan")
    a2all = ev["au2"]
    a2 = a2all.filter((pl.col("segment") == seg) & pl.col("resolved")).sort("event_ts")
    ex = (a2["hit"].cast(pl.Float64) - a2["null"]).to_numpy() if a2.height else np.array([])
    au2 = {"excess": _nw(ex)["mean"], "t": _nw(ex)["t"], "n_events": a2.height,
           "hit_rate": float(a2["hit"].mean()) if a2.height else float("nan"), "null": float(a2["null"].mean()) if a2.height else float("nan"),
           "n_unresolved_or_straddle": int(a2all.filter(pl.col("segment").is_null() | ~pl.col("resolved")).height) if seg == "llm_in_sample" else None,
           "median_bars": float(a2["bars"].median()) if a2.height else float("nan"),
           "null_bgk": float(((a2["b_sig"] + BGK) / (a2["a_sig"] + a2["b_sig"] + 2 * BGK)).mean()) if a2.height else float("nan"),
           "median_a_sig": float(a2["a_sig"].median()) if a2.height else float("nan"), "median_b_sig": float(a2["b_sig"].median()) if a2.height else float("nan")}
    for side, nm in ((1, "above"), (-1, "below")):
        s = a2.filter(pl.col("side") == side)
        au2[nm] = {"n": s.height, "hit": float(s["hit"].mean()) if s.height else float("nan"), "null": float(s["null"].mean()) if s.height else float("nan")}
    d = ev["days"].filter(pl.col("segment") == seg)
    d3 = d.filter(pl.col("bucket").is_in(["balance", "imbalance"]) & pl.col("mr_minus_trend").is_not_nan()).sort("day")
    au3 = did(d3["mr_minus_trend"].to_numpy(), (d3["bucket"] == "balance").to_numpy())
    au3["by_bucket"] = {}
    for b in ("balance", "middle", "imbalance"):
        x = d.filter(pl.col("bucket") == b)
        au3["by_bucket"][b] = {"days": x.height, **{s: (float(np.nanmean(x[s].to_numpy())) if x.height else float("nan")) for s in (*MR, *TREND, "mr_minus_trend")}}
    au3["share_balance"] = float((d["bucket"] == "balance").mean()) if d.height else float("nan")
    au3["share_imbalance"] = float((d["bucket"] == "imbalance").mean()) if d.height else float("nan")
    d4 = d.filter((pl.col("bucket") == "balance") & pl.col("touch").is_not_null()).sort("day")
    x4 = (d4["touch"] - d4["touch_null"]).to_numpy() if d4.height else np.array([])
    au4 = {"excess": _nw(x4)["mean"], "t": _nw(x4)["t"], "n_days": d4.height,
           "touch_rate": float(d4["touch"].mean()) if d4.height else float("nan"), "null": float(d4["touch_null"].mean()) if d4.height else float("nan"),
           "median_dist_sig": float(d4["poc_dist_sig"].median()) if d4.height else float("nan")}
    # context (not registered, not used for the verdict): the same magnet test on every day / after imbalance days
    for nm, cond in (("ctx_all_days", pl.col("touch").is_not_null()), ("ctx_after_imbalance", (pl.col("bucket") == "imbalance") & pl.col("touch").is_not_null())):
        s = d.filter(cond).sort("day")
        x = (s["touch"] - s["touch_null"]).to_numpy() if s.height else np.array([])
        au4[nm] = {"excess": _nw(x)["mean"], "t": _nw(x)["t"], "n": s.height}
    return {"H-AU1": au1, "H-AU2": au2, "H-AU3": au3, "H-AU4": au4}


# ------------------------------------------------------------------ H-RG1

def variance_ratio(r: np.ndarray, q: int = 24) -> float:
    """Lo-MacKinlay VR(q) with unbiased variance estimators and overlapping q-period sums."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    mu = r.mean()
    v1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    m = q * (n - q + 1) * (1 - q / n)
    vq = ((rq - q * mu) ** 2).sum() / m
    return float(vq / v1) if v1 > 0 else float("nan")  # m already carries the factor q


def rg1_panel(as_of: datetime | None = None, symbols: list[str] | tuple[str, ...] = UNIVERSE) -> pl.DataFrame:
    as_of = as_of or now()
    cut = knowledge_cutoff()
    rows = []
    for sym in symbols:
        f = market_frame(sym, "1h", as_of, funding=False, oi=False)
        if f.height < 600:
            continue
        pnl = {s: strategy_pnl(f, REGISTRY[s]()(f))["pnl"].to_numpy() for s in (*MR, *TREND)}
        eh = _hour(f["event_ts"])
        ret = f["ret"].to_numpy().astype(float)
        cts = f["close_ts"]
        dec = np.flatnonzero(((cts.dt.weekday() == 1) & (cts.dt.hour() == 0)).to_numpy())
        ctl = cts.to_list()
        for t in dec:
            t = int(t)
            if t < 167 or t + 168 >= len(eh) or eh[t] - eh[t - 167] != 167 or eh[t + 168] - eh[t] != 168:
                continue
            r = ret[t - 167:t + 1]
            if not np.isfinite(r).all():
                continue
            row = {"symbol": sym, "week": ctl[t], "vr": variance_ratio(r)}
            for s in pnl:
                row[s] = float(pnl[s][t:t + 168].sum())
            row["segment"] = _seg(ctl[t + 168], ctl[t], cut)
            rows.append(row)
    p = pl.DataFrame(rows)
    return p.with_columns((pl.mean_horizontal(*TREND) - pl.mean_horizontal(*MR)).alias("trend_minus_mr"))


def stats_rg1(p: pl.DataFrame, seg: str) -> dict:
    s = p.filter(pl.col("segment") == seg).sort("week")
    b = s.filter((pl.col("vr") > VR_HI) | (pl.col("vr") < VR_LO))
    wk = b["week"].dt.epoch("s").to_numpy()
    out = did(b["trend_minus_mr"].to_numpy(), (b["vr"] > VR_HI).to_numpy(), groups=wk)
    out["n_obs"] = s.height
    out["n_weeks"] = s["week"].n_unique()
    out["share_hi"] = float((s["vr"] > VR_HI).mean()) if s.height else float("nan")
    out["share_lo"] = float((s["vr"] < VR_LO).mean()) if s.height else float("nan")
    out["vr_median"] = float(s["vr"].median()) if s.height else float("nan")
    out["by_bucket"] = {}
    for nm, cond in (("vr_gt_1.2", pl.col("vr") > VR_HI), ("middle", (pl.col("vr") >= VR_LO) & (pl.col("vr") <= VR_HI)), ("vr_lt_0.8", pl.col("vr") < VR_LO)):
        x = s.filter(cond)
        out["by_bucket"][nm] = {"n": x.height, **{k: float(x[k].mean()) if x.height else float("nan") for k in (*MR, *TREND, "trend_minus_mr")}}
    return out


# ------------------------------------------------------------------ verdicts (mechanical, from the registry)

def verdict(hid: str, r: dict, seg: str) -> str:
    if hid == "H-AU1":
        # registered "mean > 0, t > 2" read against the simulated driftless null of the per-day mean (au1_null_sim),
        # not against 0: the per-day weighting alone gives ~+0.09 sigma under a random walk.
        val, t, n = r["excess_vs_null"], r["t_excess_vs_null"], r["n_events"]
        ok_in = val > 0 and t > 2
    elif hid in ("H-AU2", "H-AU4"):
        val, t = r["excess"], r["t"]
        n = r["n_events"] if hid == "H-AU2" else r["n_days"]
        ok_in = val > 0.05 and t > 2
    else:  # H-AU3, H-RG1: DiD
        val, t, n = r["did"], r["t"], min(r["n_hi"], r["n_lo"])
        ok_in = val > 0 and t > 2
    if seg == "clean_holdout":
        return "inconclusive" if n < MIN_HOLDOUT or not np.isfinite(val) else ("supports" if val > 0 else "contradicts")
    return "supports" if ok_in else "contradicts"


def run(seg: str, as_of: datetime | None = None, with_eth: bool = True) -> dict:
    out: dict = {"segment": seg, "cutoff": knowledge_cutoff().isoformat()}
    ctx = build("BTC", as_of)
    ev = events(ctx)
    out["BTC"] = stats_au(ev, seg)
    out["BTC_span"] = [str(ctx["hf"]["event_ts"].min()), str(ctx["hf"]["event_ts"].max())]
    out["data_BTC"] = ctx["data"]
    if with_eth and seg == "llm_in_sample":  # ETH has ~1 month of holdout: in-sample replication only
        ectx = build("ETH", as_of)
        out["ETH"] = stats_au(events(ectx), seg)
        out["data_ETH"] = ectx["data"]
    out["H-RG1"] = stats_rg1(rg1_panel(as_of), seg)
    out["verdicts"] = {h: verdict(h, out["BTC"][h], seg) for h in ("H-AU1", "H-AU2", "H-AU3", "H-AU4")}
    out["verdicts"]["H-RG1"] = verdict("H-RG1", out["H-RG1"], seg)
    if "ETH" in out:
        out["verdicts_ETH_replication"] = {h: verdict(h, out["ETH"][h], seg) for h in ("H-AU1", "H-AU2", "H-AU3", "H-AU4")}
    return out


NOTES = {
    "H-AU1": "first 2-close acceptance beyond prior-day VAH/VAL inside day D; signed 24h return / (sigma_risk_next*sqrt24); per-day mean, NW t over days; verdict on the excess over the simulated driftless null of the per-day mean (50 Gaussian RW sims from each real 00:00 close with real VA and sigma)",
    "H-AU2": "open outside prior VA, first 2 consecutive inside closes; race on 1h closes (cap 720): opposite edge vs re-entry edge -/+0.5 sigma; per-event null b/(a+b); NW t",
    "H-AU3": "overlap = share of VA(D-1) inside VA(D-2); daily net P&L by close_ts date; DiD = slope of (MR-trend) on 1{balance}, HAC t",
    "H-AU4": "days after a balance day; touch of POC(D-1) within D vs 2(1-Phi(d/sqrt24)), d from the 00:00 close in sigma_risk_next; NW t",
    "H-RG1": "Monday 00:00 UTC decisions; Lo-MacKinlay VR(24) on trailing 168 1h returns; next-168-bar net P&L; DiD slope of (trend-MR) on 1{VR>1.2} vs VR<0.8, clustered by week",
}


def record_all(out: dict, only: tuple[str, ...] = (), prefix: str = "") -> None:
    from ai_trader.research.hypotheses import record

    seg = out["segment"]
    for h in ("H-AU1", "H-AU2", "H-AU3", "H-AU4"):
        if only and h not in only:
            continue
        m = dict(out["BTC"][h])
        note = NOTES[h]
        if "ETH" in out:
            m["eth_replication"] = out["ETH"][h]
            note += f"; ETH (binance 1m, in-sample replication only) verdict-equivalent: {out['verdicts_ETH_replication'][h]}"
        record(h, seg, out["verdicts"][h], m, data=out["data_BTC"], note=prefix + note)
    if not only or "H-RG1" in only:
        record("H-RG1", seg, out["verdicts"]["H-RG1"], out["H-RG1"], data="bybit 1h, 23-symbol universe, market_frame(funding=False, oi=False), signals on full history", note=prefix + NOTES["H-RG1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", choices=SEGS, required=True)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="comma-separated ids to record (default: all)")
    ap.add_argument("--note-prefix", default="")
    a = ap.parse_args()
    out = run(a.segment)
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)
    if a.record:
        record_all(out, tuple(x for x in a.only.split(",") if x), a.note_prefix)


if __name__ == "__main__":
    main()
