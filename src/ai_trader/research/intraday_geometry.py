"""Intraday structure and market geometry (block GEO, H-GEO1..H-GEO17, research/hypotheses.yaml).

Design frozen in data/research/intraday_geometry/design.json and the diary (2026-11-25 22:25 UTC) before any outcome.

Data: Bybit linear 5m klines of universe L (data/research/bos/bars/5m, 2022-01 .. 2026-11-15, plus a public-REST
extension to 2026-11-25 in data/research/intraday_geometry/bars), lake funding / 4h bars (v0 regime) / flow_3s
(spread, depth) / liquidations, and the scheduled CPI / NFP / FOMC times (data/research/news/events/events.parquet).

Every price series is put on a complete 5m grid (missing bars carry the last close, zero volume). Returns are log
returns; sigma_slow = sqrt(trailing 2016-bar mean of r^2), shifted one bar (known at the bar's open). All forward
outcomes are divided by sigma_slow * sqrt(bars).

Stages (each writes to data/research/intraday_geometry/):
    map     part 1: hour-of-week profiles, release / funding / session / month-end windows, spread-depth (H-GEO1..8)
    events  part 2: geometry levels + the first-test event machine + placebo levels, per symbol (H-GEO9..15, 17)
    infer   part 2 inference: real minus placebo, BH, holdout, regime / session cells
    rules   part 3: executable rules with costs, walk-forward, DSR / PBO, random entries (H-GEO16)

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.intraday_geometry map|events|infer|rules
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ai_trader.core.config import load

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT", "DOTUSDT", "BNBUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT", "XLMUSDT", "SUIUSDT", "ENAUSDT", "HYPEUSDT"]
AS_OF = datetime(2026, 11, 25, 22, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 7, 1, tzinfo=UTC)
G0 = datetime(2022, 1, 1, tzinfo=UTC)          # global 5m grid origin (part 1 pooling)
BAR_NS = 300 * 10**9
DAY_NS = 86400 * 10**9
BPH, BPD = 12, 288
SIG_WIN = 2016                                  # 7 days of 5m bars
HORIZONS = (3, 12, 48)
OBJECTS = ("PD_HL", "PD_C", "PW_HL", "OR15", "OR30", "OR60", "ASIA_BOX", "VWAP_S", "VWAP_A", "VA_EDGE", "POC", "PIV4H", "ROUND")
EVENTS = ("touch", "sweep", "accept", "retest")
SESSIONS = ("asia", "europe", "us_cash", "us_off_hours", "weekend")
ARM_SIG, SWEEP_SIG, RETEST_BARS = 0.5, 0.1, 24
SEED = 7
NY, LDN = ZoneInfo("America/New_York"), ZoneInfo("Europe/London")
ND = NormalDist()


def out_dir() -> Path:
    p = load().root / "data" / "research" / "intraday_geometry"
    p.mkdir(parents=True, exist_ok=True)
    return p


def seg_of(ts_ns: np.ndarray) -> np.ndarray:
    """0 = llm_in_sample, 1 = clean_holdout."""
    return (np.asarray(ts_ns) >= int(CUTOFF.timestamp() * 1e9)).astype(np.int8)


# ====================================================================== bars
@dataclass
class Bars:
    sym: str
    t0: int                 # ns of bar 0 open
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray
    turn: np.ndarray
    real: np.ndarray        # bar existed in the source (not filled)
    r: np.ndarray           # log return close_{t-1} -> close_t
    sig: np.ndarray         # sigma_slow known at bar t open

    @property
    def n(self) -> int:
        return len(self.c)

    def ts(self) -> np.ndarray:
        return self.t0 + np.arange(self.n, dtype=np.int64) * BAR_NS


def grid_fill(ts_ns: np.ndarray, o, h, lo, c, v, turn) -> tuple[int, dict]:
    """Complete 5m grid from the first to the last bar; missing bars carry the previous close with zero volume."""
    t0 = int(ts_ns[0])
    idx = ((ts_ns - t0) // BAR_NS).astype(np.int64)
    n = int(idx[-1]) + 1
    real = np.zeros(n, bool)
    real[idx] = True
    cc = np.full(n, np.nan)
    cc[idx] = c
    # forward-fill closes
    pos = np.where(real, np.arange(n), 0)
    np.maximum.accumulate(pos, out=pos)
    cc = cc[pos]
    out = {"c": cc, "real": real}
    for name, arr in (("o", o), ("h", h), ("lo", lo)):
        x = cc.copy()
        x[idx] = arr
        out[name] = x
    for name, arr in (("v", v), ("turn", turn)):
        x = np.zeros(n)
        x[idx] = arr
        out[name] = x
    return t0, out


def sigma_slow(r: np.ndarray, win: int = SIG_WIN, min_n: int = BPD) -> np.ndarray:
    """sqrt(mean of r^2 over the previous `win` bars), excluding bar t itself (causal)."""
    r2 = np.where(np.isfinite(r), r * r, 0.0)
    ok = np.isfinite(r).astype(float)
    cs, cn = np.r_[0.0, np.cumsum(r2)], np.r_[0.0, np.cumsum(ok)]
    t = np.arange(len(r))
    a = np.maximum(t - win, 0)
    s, k = cs[t] - cs[a], cn[t] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(s / k)
    out[k < min_n] = np.nan
    return out


def load_bars(sym: str) -> Bars | None:
    from ai_trader.research.bos_regimes_study import _burn_start

    root = load().root / "data" / "research"
    fs = sorted((root / "bos" / "bars" / "5m").glob(f"{sym}_*.parquet")) + sorted((root / "intraday_geometry" / "bars").glob(f"{sym}_ext.parquet"))
    if not fs:
        return None
    b = pl.concat([pl.read_parquet(f).with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC"))) for f in fs], how="vertical_relaxed")
    b = b.unique("event_ts", keep="last").sort("event_ts")
    b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                 & (pl.col("low") <= pl.min_horizontal("open", "close")))
    st = _burn_start(sym)
    if st is not None:
        b = b.filter(pl.col("event_ts") >= st)
    b = b.filter((pl.col("event_ts") >= G0) & (pl.col("event_ts") + pl.duration(minutes=5) <= AS_OF))
    if b.height < 30 * BPD:
        return None
    ts = b["event_ts"].dt.epoch("ns").to_numpy()
    t0, g = grid_fill(ts, *(b[c].to_numpy().astype(float) for c in ("open", "high", "low", "close", "volume", "turnover")))
    lc = np.log(g["c"])
    r = np.r_[np.nan, np.diff(lc)]
    return Bars(sym, t0, g["o"], g["h"], g["lo"], g["c"], g["v"], g["turn"], g["real"], r, sigma_slow(r))


def session_codes(ts_ns: np.ndarray) -> np.ndarray:
    """ai_trader.research.calendar session labels as codes (SESSIONS order), from bar open times."""
    from ai_trader.research.calendar import add_calendar

    f = add_calendar(pl.DataFrame({"event_ts": pl.Series(ts_ns).cast(pl.Datetime("ns", "UTC"))}))
    m = {s: i for i, s in enumerate(SESSIONS)}
    return f["session"].replace_strict(m, return_dtype=pl.Int8).to_numpy()


def regime_arrays(sym: str, bars_close_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v0 cell (0..8, -1 unknown) and 4h vol tercile (0..2, -1) of the latest 4h bar closed by each 5m bar close."""
    from ai_trader.research import regimes
    from ai_trader.research.frame import market_frame

    f = market_frame(sym, "4h", AS_OF, funding=False, oi=False)
    if f.height < 50:
        return np.full(len(bars_close_ns), -1, np.int8), np.full(len(bars_close_ns), -1, np.int8)
    lab = regimes.label(f, bar_hours=4.0)
    sg = lab["sigma_next"].to_numpy().astype(float)
    w, mp = 2190, 1095
    vp = np.full(len(sg), np.nan)
    for t in range(mp, len(sg)):
        hh = sg[max(0, t - w):t]
        hh = hh[np.isfinite(hh)]
        if len(hh) >= mp and np.isfinite(sg[t]):
            vp[t] = (hh < sg[t]).mean()
    vt = np.where(np.isfinite(vp), np.minimum(np.floor(np.nan_to_num(vp) * 3), 2), -1).astype(np.int8)
    cell = np.array([int(c[1:]) - 1 if c else -1 for c in lab["cell"].to_list()], np.int8)
    cts = lab["close_ts"].cast(pl.Datetime("ns", "UTC")).dt.epoch("ns").to_numpy()
    k = np.searchsorted(cts, bars_close_ns, side="right") - 1
    ok = k >= 0
    return np.where(ok, cell[np.maximum(k, 0)], -1).astype(np.int8), np.where(ok, vt[np.maximum(k, 0)], -1).astype(np.int8)


# ====================================================================== statistics
def nw_var_mean(u: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) long-run variance of sum(u) for a zero-mean influence series u (one value per cluster)."""
    u = np.asarray(u, float)
    v = float(u @ u)
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * float(u[k:] @ u[:-k])
    return v


def diff_test(day_r: np.ndarray, s_r: np.ndarray, n_r: np.ndarray, s_p: np.ndarray, n_p: np.ndarray, lags: int = 2) -> dict:
    """Real minus placebo mean with a day-clustered NW SE. Inputs are per-day sums / counts on a common day axis."""
    Nr, Np = n_r.sum(), n_p.sum()
    if Nr < 2 or Np < 2:
        return {"n": int(Nr), "n_pl": int(Np), "diff": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan")}
    mr, mp = s_r.sum() / Nr, s_p.sum() / Np
    u = (s_r - mr * n_r) / Nr - (s_p - mp * n_p) / Np
    D = max(int(((n_r + n_p) > 0).sum()), 2)
    se = math.sqrt(max(nw_var_mean(u, lags), 1e-300) * D / (D - 1))
    t = (mr - mp) / se
    return {"n": int(Nr), "n_pl": int(Np), "days": D, "mean_real": float(mr), "mean_pl": float(mp), "diff": float(mr - mp), "se": se,
            "t": float(t), "p": float(2 * (1 - ND.cdf(abs(t))))}


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    rej = np.zeros(len(p), bool)
    if not ok.any():
        return rej
    idx = np.flatnonzero(ok)
    o = idx[np.argsort(p[idx])]
    m = len(o)
    thr = q * np.arange(1, m + 1) / m
    passed = p[o] <= thr
    if passed.any():
        k = np.flatnonzero(passed).max() + 1
        rej[o[:k]] = True
    return rej


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return float("nan"), float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    n = len(a)
    t = rho * math.sqrt((n - 2) / max(1 - rho**2, 1e-12))
    return rho, float(2 * (1 - ND.cdf(abs(t))))


def boot_ratio(num: np.ndarray, den: np.ndarray, nboot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Ratio of means sum(num)/sum(den) with an i.i.d. bootstrap over units (rows)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if len(num) < 3 or den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(num), (nboot, len(num)))
    rr = num[k].sum(1) / np.maximum(den[k].sum(1), 1e-300)
    return float(num.sum() / den.sum()), float(np.quantile(rr, 0.025)), float(np.quantile(rr, 0.975))


def boot_mean(x: np.ndarray, nboot: int = 2000, seed: int = SEED, q: tuple[float, float] = (0.025, 0.975)) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (nboot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, q[0])), float(np.quantile(m, q[1]))


# ====================================================================== part 1: the intraday map
def symbol_series(b: Bars) -> dict:
    """Per-symbol 5m u = r^2 / sigma^2 and z = r / sigma on the global grid, plus hourly RV / turnover / return."""
    ts = b.ts()
    g = ((ts - int(G0.timestamp() * 1e9)) // BAR_NS).astype(np.int64)
    ok = b.real & np.r_[False, b.real[:-1]] & np.isfinite(b.sig) & (b.sig > 0) & np.isfinite(b.r)
    u = np.where(ok, (b.r / np.where(ok, b.sig, 1.0)) ** 2, np.nan)
    z = np.where(ok, b.r / np.where(ok, b.sig, 1.0), np.nan)
    # hourly, aligned to UTC hours
    first = int((-(ts[0] // BAR_NS)) % BPH)
    m = (b.n - first) // BPH
    sl = slice(first, first + m * BPH)
    r = np.where(np.isfinite(b.r), b.r, 0.0)[sl].reshape(m, BPH)
    okh = ok[sl].reshape(m, BPH).sum(1) >= 10
    rv, rh = (r**2).sum(1), r.sum(1)
    tv = b.turn[sl].reshape(m, BPH).sum(1)
    hts = ts[sl][::BPH]

    def trail(x: np.ndarray, okx: np.ndarray, w: int = 168, mn: int = 72) -> np.ndarray:
        xx = np.where(okx, x, 0.0)
        cs, cn = np.r_[0.0, np.cumsum(xx)], np.r_[0.0, np.cumsum(okx.astype(float))]
        t = np.arange(len(x))
        a = np.maximum(t - w, 0)
        s, k = cs[t] - cs[a], cn[t] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s / k
        out[k < mn] = np.nan
        return out

    lv, lt = trail(rv, okh), trail(tv, okh)
    with np.errstate(invalid="ignore", divide="ignore"):
        uh = np.where(okh, rv / lv, np.nan)
        vh = np.where(okh, tv / lt, np.nan)
        zh = np.where(okh, rh / np.sqrt(lv), np.nan)
    gh = ((hts - int(G0.timestamp() * 1e9)) // (BAR_NS * BPH)).astype(np.int64)
    return {"g": g, "u": u, "z": z, "gh": gh, "uh": uh, "vh": vh, "zh": zh, "hts": hts}


def pool_add(acc: dict, key: str, gi: np.ndarray, x: np.ndarray, size: int) -> None:
    if key not in acc:
        acc[key] = (np.zeros(size), np.zeros(size))
    s, n = acc[key]
    ok = np.isfinite(x) & (gi >= 0) & (gi < size)
    np.add.at(s, gi[ok], x[ok])
    np.add.at(n, gi[ok], 1.0)


def how_cells(ts_ns: np.ndarray) -> np.ndarray:
    """Monday 00 UTC = 0 .. Sunday 23 = 167."""
    d = ts_ns // DAY_NS
    wd = (d + 3) % 7  # 1970-01-01 was a Thursday -> Monday = 0
    hr = (ts_ns % DAY_NS) // (3600 * 10**9)
    return (wd * 24 + hr).astype(np.int64)


def profile(cells: np.ndarray, x: np.ndarray, k: int, winsor: float | None = 0.995) -> np.ndarray:
    ok = np.isfinite(x)
    xx = x[ok]
    if winsor is not None and len(xx):
        xx = np.minimum(xx, np.quantile(xx, winsor))
    s = np.bincount(cells[ok], xx, minlength=k)
    n = np.bincount(cells[ok], minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n


def map_profiles(hts: np.ndarray, uh: np.ndarray, vh: np.ndarray, zh: np.ndarray) -> dict:
    """H-GEO1 / H-GEO2 on the pooled hourly series."""
    seg = seg_of(hts)
    how = how_cells(hts)
    out: dict = {}
    for name, x in (("var", uh), ("volume", vh)):
        pin = profile(how[seg == 0], x[seg == 0], 168)
        pho = profile(how[seg == 1], x[seg == 1], 168)
        rho, p = spearman(pin, pho)
        # monthly stability on 48 cells (hour x weekday/weekend)
        wkend = ((hts // DAY_NS + 3) % 7 >= 5).astype(np.int64)
        c48 = wkend * 24 + (hts % DAY_NS) // (3600 * 10**9)
        mon = hts.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64)
        rows = []
        for mm in np.unique(mon):
            sel = mon == mm
            if sel.sum() < 24 * 20:
                continue
            prof_m = profile(c48[sel], x[sel], 48)
            base_sel = (seg == 0) & ~sel
            prof_b = profile(c48[base_sel], x[base_sel], 48)
            r_, _ = spearman(prof_m, prof_b)
            rows.append({"month": f"{1970 + mm // 12}-{mm % 12 + 1:02d}", "rho": r_, "seg": int(seg[sel][0])})
        share = float(np.mean([r["rho"] >= 0.5 for r in rows])) if rows else float("nan")
        out[name] = {"rho_holdout": rho, "p": p, "month_share_rho_ge_0.5": share, "months": rows,
                     "profile_in": pin.tolist(), "profile_ho": pho.tolist(),
                     "hod_in": profile((how % 24)[seg == 0], x[seg == 0], 24).tolist(),
                     "hod_ho": profile((how % 24)[seg == 1], x[seg == 1], 24).tolist()}
    # direction
    wk = (hts // DAY_NS + 3) // 7
    res = {}
    for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
        sel = (seg == sgi) & np.isfinite(zh)
        m = profile(how[sel], zh[sel], 168, winsor=None)
        sd = np.array([np.nanstd(zh[sel][how[sel] == k], ddof=1) if (how[sel] == k).sum() > 2 else np.nan for k in range(168)])
        nn = np.bincount(how[sel], minlength=168)
        t = m / (sd / np.sqrt(nn))
        chi = float(np.nansum(t**2))
        from scipy.stats import chi2

        res[sname] = {"mean": m.tolist(), "t": t.tolist(), "n_weeks": len(np.unique(wk[sel])), "chi2": chi, "df": 168,
                      "p_joint": float(chi2.sf(chi, 168)), "n_abs_t_gt_2": int(np.nansum(np.abs(t) > 2)),
                      "hod_mean": profile((how % 24)[sel], zh[sel], 24, winsor=None).tolist()}
    tin, mho = np.array(res["llm_in_sample"]["t"]), np.array(res["clean_holdout"]["mean"])
    rho, p = spearman(np.array(res["llm_in_sample"]["mean"]), mho)
    # tilt: hold the in-sample sign of |t| > 2 cells in the holdout; 1 R = one hourly sigma; cost 15 bp per held hour
    pick = np.flatnonzero(np.abs(tin) > 2)
    sel = (seg == 1) & np.isfinite(zh)
    tilt = []
    for k in pick:
        zz = zh[sel][how[sel] == k]
        tilt.append(np.sign(tin[k]) * zz)
    tilt_g = np.concatenate(tilt) if tilt else np.array([])
    res["rank_corr_in_vs_ho"] = {"rho": rho, "p": p}
    res["tilt_holdout"] = {"cells": len(pick), "gross_R_per_hour": float(tilt_g.mean()) if len(tilt_g) else float("nan"), "n_hours": len(tilt_g)}
    out["direction"] = res
    return out


def release_windows(pool_ts: np.ndarray, pu: np.ndarray, events: pl.DataFrame) -> dict:
    """H-GEO3: pooled standardised r^2 in windows after CPI / NFP / FOMC vs the same UTC minute on 20 prior weekdays."""
    t0 = int(pool_ts[0])
    ev_ns = events["scheduled_ts"].cast(pl.Datetime("ns", "UTC")).dt.epoch("ns").to_numpy()
    ev_days = set((ev_ns // DAY_NS).tolist())
    wins = {"0-5": (0, 1), "5-15": (1, 3), "15-30": (3, 6), "30-60": (6, 12), "pre_-30-0": (-6, 0)}
    rows = []
    for k, e in zip(events["kind"].to_list(), ev_ns):
        i0 = (e - t0) // BAR_NS
        if i0 < BPD * 30 or i0 + 12 >= len(pu):
            continue
        d = e // DAY_NS
        base_days, dd = [], d - 1
        while len(base_days) < 20 and dd > d - 60:
            if (dd + 3) % 7 < 5 and dd not in ev_days:
                base_days.append(dd)
            dd -= 1
        row = {"kind": k, "ts": int(e), "seg": int(seg_of(np.array([e]))[0])}
        for wn, (a, bb) in wins.items():
            x = pu[i0 + a:i0 + bb]
            bi = np.array([(bd * DAY_NS + (e % DAY_NS) - t0) // BAR_NS for bd in base_days])
            bx = np.concatenate([pu[j + a:j + bb] for j in bi if j + a >= 0])
            row[f"ev_{wn}"] = float(np.nanmean(x))
            row[f"base_{wn}"] = float(np.nanmean(bx))
        rows.append(row)
    df = pl.DataFrame(rows)
    out = {"n_events": df.height}
    for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
        s = df.filter(pl.col("seg") == sgi)
        o = {"n": s.height}
        for wn in wins:
            rr = boot_ratio(s[f"ev_{wn}"].to_numpy(), s[f"base_{wn}"].to_numpy())
            o[wn] = {"ratio": rr[0], "lo": rr[1], "hi": rr[2]}
        for kind in ("cpi", "nfp", "fomc"):
            sk = s.filter(pl.col("kind") == kind)
            if sk.height >= 3:
                rr = boot_ratio(sk["ev_0-5"].to_numpy(), sk["base_0-5"].to_numpy())
                o[f"{kind}_0-5"] = {"n": sk.height, "ratio": rr[0], "lo": rr[1], "hi": rr[2]}
        out[sname] = o
    return out


def funding_windows(pool_ts: np.ndarray, pu: np.ndarray) -> dict:
    """H-GEO4: [hh:00, hh:05) at 00/08/16 UTC vs the same bar at hh-1 and hh+1, pooled, day bootstrap."""
    t0 = int(pool_ts[0])
    first_day = -(-t0 // DAY_NS)
    days = np.arange(first_day, (int(pool_ts[-1]) // DAY_NS))
    out = {}
    for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
        res = {}
        for label, hours in (("all", (0, 8, 16)), ("08_16", (8, 16)), ("00", (0,)), ("08", (8,)), ("16", (16,))):
            num, den = [], []
            for d in days:
                if int(seg_of(np.array([d * DAY_NS]))[0]) != sgi:
                    continue
                a, bsum, ok = 0.0, 0.0, True
                for hh in hours:
                    i = (d * DAY_NS + hh * 3600 * 10**9 - t0) // BAR_NS
                    nb = [i - BPH, i + BPH]
                    if i - BPH < 0 or i + BPH >= len(pu) or not np.isfinite(pu[i]) or not all(np.isfinite(pu[j]) for j in nb):
                        ok = False
                        break
                    a += pu[i]
                    bsum += 0.5 * (pu[nb[0]] + pu[nb[1]])
                if ok:
                    num.append(a)
                    den.append(bsum)
            rr = boot_ratio(np.array(num), np.array(den))
            res[label] = {"ratio": rr[0], "lo": rr[1], "hi": rr[2], "days": len(num)}
        out[sname] = res
    return out


def funding_direction(rows: pl.DataFrame) -> dict:
    """H-GEO5: rows = (sym, ts, pre, post, prev_rate) per settlement; pooled per settlement time, signed by -sign(prev)."""
    from ai_trader.research.evaluate import newey_west_t

    out = {}
    d = rows.with_columns((-pl.col("prev_rate").sign()).alias("sg")).filter(pl.col("sg") != 0)
    for label, flt in (("all", pl.lit(True)), ("abs_ge_1bp", pl.col("prev_rate").abs() >= 1e-4)):
        g = (d.filter(flt).with_columns((pl.col("sg") * pl.col("pre")).alias("spre"), (pl.col("sg") * pl.col("post")).alias("spost"))
             .group_by("ts").agg(pl.col("spre").mean(), pl.col("spost").mean(), pl.len().alias("nsym")).sort("ts"))
        seg = seg_of(g["ts"].to_numpy())
        res = {}
        for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
            o = {}
            for col in ("spre", "spost"):
                x = g[col].to_numpy()[seg == sgi]
                m, t, n = newey_west_t(x) if len(x) > 10 else (float("nan"), float("nan"), len(x))
                o[col] = {"mean": float(m), "t": float(t), "n": int(n)}
            res[sname] = o
        out[label] = res
    return out


def session_open_windows(pool_ts: np.ndarray, pu: np.ndarray, ev_days: set) -> dict:
    """H-GEO6: 30 min after vs 30 min before the session open; per day ratio of sums, mean over days; DST off-clock check."""
    from ai_trader.features.seasonality import is_nyse_day

    t0 = int(pool_ts[0])
    first_day = -(-t0 // DAY_NS)
    days = np.arange(first_day + 1, int(pool_ts[-1]) // DAY_NS)
    rows = []
    for d in days:
        dt = date(1970, 1, 1) + timedelta(days=int(d))
        wd = dt.weekday()
        if d in ev_days:
            continue
        opens = {"asia": datetime(dt.year, dt.month, dt.day, tzinfo=UTC)}
        opens["london"] = datetime(dt.year, dt.month, dt.day, 8, 0, tzinfo=LDN).astimezone(UTC)
        edt = False
        if is_nyse_day(dt):
            us = datetime(dt.year, dt.month, dt.day, 9, 30, tzinfo=NY).astimezone(UTC)
            opens["us"] = us
            edt = us.hour == 13
            opens["us_offclock"] = us + timedelta(hours=1) if edt else us - timedelta(hours=1)
        for name, o in opens.items():
            i = (int(o.timestamp() * 1e9) - t0) // BAR_NS
            if i - 6 < 0 or i + 6 >= len(pu):
                continue
            a, b_ = np.nansum(pu[i:i + 6]), np.nansum(pu[i - 6:i])
            if not (np.isfinite(pu[i:i + 6]).all() and np.isfinite(pu[i - 6:i]).all()) or b_ <= 0:
                continue
            rows.append({"sess": name, "day": int(d), "wd": wd, "seg": int(seg_of(np.array([d * DAY_NS]))[0]), "after": a, "before": b_, "edt": bool(edt)})
    df = pl.DataFrame(rows)
    out = {}
    for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
        o = {}
        for sess in ("asia", "london", "us", "us_offclock"):
            for lab, flt in (("all", pl.lit(True)), ("edt", pl.col("edt")), ("est", ~pl.col("edt"))):
                if sess in ("asia", "london") and lab != "all":
                    continue
                s = df.filter((pl.col("seg") == sgi) & (pl.col("sess") == sess) & flt & (pl.col("wd") < 5))
                if s.height < 5:
                    continue
                ratio = (s["after"] / s["before"]).to_numpy()
                m, lo, hi = boot_mean(np.log(ratio))  # CI on the log scale (per-day ratios are heavy-tailed)
                rm = boot_ratio(s["after"].to_numpy(), s["before"].to_numpy())
                o[f"{sess}_{lab}"] = {"days": s.height, "mean_of_ratios": float(ratio.mean()),
                                      "mean_ratio_ci": list(boot_mean(ratio)), "geo_mean_ratio": float(np.exp(m)),
                                      "geo_ci": [float(np.exp(lo)), float(np.exp(hi))], "ratio_of_means": rm}
        out[sname] = o
    return out


def month_end(hts: np.ndarray, uh: np.ndarray, zh: np.ndarray) -> dict:
    """H-GEO7: 15:00-16:00 UTC hour of the last calendar day vs the same hour on the other weekdays of the month."""
    from ai_trader.research.evaluate import newey_west_t

    hr = (hts % DAY_NS) // (3600 * 10**9)
    sel = hr == 15
    t, u, z = hts[sel], uh[sel], zh[sel]
    dts = [datetime.fromtimestamp(x / 1e9, UTC) for x in t]
    rows = {}
    for i, d in enumerate(dts):
        key = (d.year, d.month)
        last = (d + timedelta(days=1)).month != d.month
        rows.setdefault(key, {"me_u": np.nan, "me_z": np.nan, "base": [], "seg": int(seg_of(np.array([t[i]]))[0])})
        if last:
            rows[key]["me_u"], rows[key]["me_z"] = u[i], z[i]
        elif d.weekday() < 5:
            rows[key]["base"].append(u[i])
    out = {}
    for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
        me = np.array([v["me_u"] for v in rows.values() if v["seg"] == sgi])
        bs = np.array([np.nanmean(v["base"]) if v["base"] else np.nan for v in rows.values() if v["seg"] == sgi])
        mz = np.array([v["me_z"] for v in rows.values() if v["seg"] == sgi])
        rr = boot_ratio(me, bs)
        m, tt, _n = newey_west_t(mz[np.isfinite(mz)]) if np.isfinite(mz).sum() > 3 else (float("nan"), float("nan"), int(np.isfinite(mz).sum()))
        out[sname] = {"months": int(np.isfinite(me).sum()), "ratio": rr[0], "lo": rr[1], "hi": rr[2], "dir_mean_z": float(m), "dir_t": float(tt)}
    return out


def spread_depth() -> dict:
    """H-GEO8: hour-of-day profiles of spread, depth_25 and variance from lake flow_3s (rolled to 5m)."""
    import duckdb

    base = load().root / "data" / "lake" / "flow_3s" / "venue=bybit" / "category=linear"
    con = duckdb.connect()
    con.execute("SET threads TO 2; SET memory_limit = '900MB'")
    syms = ",".join(f"'{s}'" for s in UNIVERSE)
    q = f"""
      SELECT symbol, (epoch_ms(event_ts) // 300000) * 300000 AS b_ms,
             avg(spread_bps) AS spr, avg(coalesce(depth_bid_25,0) + coalesce(depth_ask_25,0)) AS dep, arg_max(close, event_ts) AS c
      FROM read_parquet('{base.as_posix()}/date=*/*.parquet', hive_partitioning=true)
      WHERE symbol IN ({syms}) AND book_cov > 0.5
      GROUP BY 1, 2"""
    d = pl.from_arrow(con.execute(q).to_arrow_table())
    d = d.with_columns(pl.from_epoch("b_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("b")).drop("b_ms")
    d = d.sort("symbol", "b").with_columns(pl.col("c").log().diff().over("symbol").alias("r"))
    d = d.with_columns(pl.col("b").dt.hour().alias("hr"), pl.col("b").dt.date().alias("day"))
    # per symbol z-scored hour-of-day profiles; median across symbols
    prof = d.group_by("symbol", "hr").agg(pl.col("spr").mean(), pl.col("dep").mean(), (pl.col("r") ** 2).mean().alias("var"))

    def zs(c: str) -> pl.Expr:
        return ((pl.col(c) - pl.col(c).mean().over("symbol")) / pl.col(c).std().over("symbol")).alias(c)

    pz = prof.with_columns(zs("spr"), zs("dep"), zs("var")).group_by("hr").agg(pl.col("spr", "dep", "var").median()).sort("hr")
    rs, _ = spearman(pz["spr"].to_numpy(), pz["var"].to_numpy())
    rd, _ = spearman(pz["dep"].to_numpy(), pz["var"].to_numpy())
    days = d["day"].unique().sort().to_list()
    rng = np.random.default_rng(SEED)
    bs, bd = [], []
    for _ in range(300):
        pick = [days[i] for i in rng.integers(0, len(days), len(days))]
        sub = pl.concat([d.filter(pl.col("day") == x) for x in pick])
        p = sub.group_by("symbol", "hr").agg(pl.col("spr").mean(), pl.col("dep").mean(), (pl.col("r") ** 2).mean().alias("var"))
        p = p.with_columns(zs("spr"), zs("dep"), zs("var")).group_by("hr").agg(pl.col("spr", "dep", "var").median()).sort("hr")
        bs.append(spearman(p["spr"].to_numpy(), p["var"].to_numpy())[0])
        bd.append(spearman(p["dep"].to_numpy(), p["var"].to_numpy())[0])
    med = d.group_by("symbol").agg(pl.col("spr").median().alias("median_spread_bps"), pl.col("dep").median().alias("median_depth25"))
    return {"days": len(days), "symbols": d["symbol"].n_unique(), "rho_spread_var": rs, "ci90_spread": [float(np.nanquantile(bs, 0.05)), float(np.nanquantile(bs, 0.95))],
            "rho_depth_var": rd, "ci90_depth": [float(np.nanquantile(bd, 0.05)), float(np.nanquantile(bd, 0.95))],
            "profile": pz.to_dicts(), "median_spread_by_symbol": {r["symbol"]: r["median_spread_bps"] for r in med.to_dicts()}}


def stage_map() -> dict:
    from ai_trader.storage.lake import Lake

    NG = int((AS_OF - G0).total_seconds() // 300)
    NH = NG // BPH
    acc: dict = {}
    frows = []
    lake = Lake()
    for sym in UNIVERSE:
        b = load_bars(sym)
        if b is None:
            continue
        s = symbol_series(b)
        pool_add(acc, "u", s["g"], s["u"], NG)
        pool_add(acc, "z", s["g"], s["z"], NG)
        pool_add(acc, "uh", s["gh"], s["uh"], NH)
        pool_add(acc, "vh", s["gh"], s["vh"], NH)
        pool_add(acc, "zh", s["gh"], s["zh"], NH)
        f = lake.read_asof("funding", AS_OF, symbols=[sym], filters={"venue": "bybit"})
        if f.height:
            f = f.sort("event_ts").unique("event_ts", keep="last").sort("event_ts")
            fts = f["event_ts"].cast(pl.Datetime("ns", "UTC")).dt.epoch("ns").to_numpy()
            rate = f["rate"].to_numpy().astype(float)
            ts = b.ts()
            for j in range(1, len(fts)):
                hh = (fts[j] % DAY_NS) // (3600 * 10**9)
                if hh not in (0, 8, 16) or fts[j] % (3600 * 10**9) != 0:
                    continue
                i = (fts[j] - ts[0]) // BAR_NS
                if i - BPH < 0 or i + BPH >= b.n:
                    continue
                zz = s["z"][i - BPH:i + BPH]
                if not np.isfinite(zz).all():
                    continue
                frows.append((sym, int(fts[j]), float(zz[:BPH].sum() / math.sqrt(BPH)), float(zz[BPH:].sum() / math.sqrt(BPH)), float(rate[j - 1])))
        print(f"map {sym} bars={b.n}", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pu = acc["u"][0] / acc["u"][1]
        pz = acc["z"][0] / acc["z"][1]
        uh = acc["uh"][0] / acc["uh"][1]
        vh = acc["vh"][0] / acc["vh"][1]
        zh = acc["zh"][0] / acc["zh"][1]
    g0 = int(G0.timestamp() * 1e9)
    pool_ts = g0 + np.arange(NG, dtype=np.int64) * BAR_NS
    hts = g0 + np.arange(NH, dtype=np.int64) * BAR_NS * BPH
    od = out_dir()
    pl.DataFrame({"ts": pool_ts, "u": pu, "z": pz, "n": acc["u"][1]}).write_parquet(od / "pool_5m.parquet")
    pl.DataFrame({"ts": hts, "u": uh, "v": vh, "z": zh, "n": acc["uh"][1]}).write_parquet(od / "pool_1h.parquet")
    fr = pl.DataFrame(frows, schema=["sym", "ts", "pre", "post", "prev_rate"], orient="row")
    fr.write_parquet(od / "funding_windows.parquet")
    ev = pl.read_parquet(load().root / "data" / "research" / "news" / "events" / "events.parquet").filter(pl.col("kind").is_in(["cpi", "nfp", "fomc"]))
    ev = ev.filter((pl.col("scheduled_ts") >= G0 + timedelta(days=40)) & (pl.col("scheduled_ts") < AS_OF - timedelta(hours=2)))
    ev_days = set((ev["scheduled_ts"].cast(pl.Datetime("ns", "UTC")).dt.epoch("ns").to_numpy() // DAY_NS).tolist())
    okh = np.isfinite(uh)
    res = {"profiles": map_profiles(hts[okh], uh[okh], vh[okh], zh[okh]),
           "releases": release_windows(pool_ts, pu, ev),
           "funding_vol": funding_windows(pool_ts, pu),
           "funding_dir": funding_direction(fr),
           "session_opens": session_open_windows(pool_ts, pu, ev_days),
           "month_end": month_end(hts[okh], uh[okh], zh[okh])}
    try:
        res["spread_depth"] = spread_depth()
    except Exception as e:  # noqa: BLE001 - the spread profile must not sink the whole map stage
        res["spread_depth"] = {"error": repr(e)}
    (od / "map_results.json").write_text(json.dumps(res, default=float, indent=1))
    return res


# ====================================================================== part 2: levels and the event machine
@dataclass
class Levels:
    """Level instances of one object. lo/hi are log prices (band edges; equal for point levels). For dynamic levels
    (VWAPs) `anchor` holds the bar index the VWAP starts at and lo/hi hold a constant log offset (0 for real)."""
    obj: int
    sub: np.ndarray
    start: np.ndarray
    W: int
    lo: np.ndarray
    hi: np.ndarray
    anchor: np.ndarray | None = None


def day_starts(b: Bars) -> np.ndarray:
    ts = b.ts()
    first = int((-(ts[0] // BAR_NS)) % BPD)
    return np.arange(first, b.n, BPD)


def lv_prior_day(b: Bars) -> tuple[Levels, Levels]:
    ds = day_starts(b)
    ds = ds[(ds - BPD >= 0)]
    lh, ll, lc = np.log(b.h), np.log(b.lo), np.log(b.c)
    hi = np.array([lh[d - BPD:d].max() for d in ds])
    lo = np.array([ll[d - BPD:d].min() for d in ds])
    pc = lc[ds - 1]
    st = np.r_[ds, ds]
    hl = Levels(0, np.r_[np.zeros(len(ds)), np.ones(len(ds))].astype(np.int8), st, BPD, np.r_[hi, lo], np.r_[hi, lo])
    return hl, Levels(1, np.zeros(len(ds), np.int8), ds, BPD, pc, pc)


def lv_prior_week(b: Bars) -> Levels:
    ts = b.ts()
    ds = day_starts(b)
    mon = ds[((ts[ds] // DAY_NS) + 3) % 7 == 0]
    mon = mon[mon - 7 * BPD >= 0]
    lh, ll = np.log(b.h), np.log(b.lo)
    hi = np.array([lh[d - 7 * BPD:d].max() for d in mon])
    lo = np.array([ll[d - 7 * BPD:d].min() for d in mon])
    return Levels(2, np.r_[np.zeros(len(mon)), np.ones(len(mon))].astype(np.int8), np.r_[mon, mon], 7 * BPD, np.r_[hi, lo], np.r_[hi, lo])


def session_open_index(b: Bars) -> dict[str, np.ndarray]:
    """Bar index of each session open: Asia 00:00 UTC (every day), London 08:00 Europe/London (every day), US 09:30
    America/New_York (NYSE days)."""
    from ai_trader.features.seasonality import is_nyse_day

    ts = b.ts()
    t0 = ts[0]
    out: dict[str, list] = {"asia": [], "london": [], "us": []}
    for d in range(int(-(-t0 // DAY_NS)), int(ts[-1] // DAY_NS) + 1):
        dt = date(1970, 1, 1) + timedelta(days=d)
        cand = {"asia": datetime(dt.year, dt.month, dt.day, tzinfo=UTC),
                "london": datetime(dt.year, dt.month, dt.day, 8, 0, tzinfo=LDN).astimezone(UTC)}
        if is_nyse_day(dt):
            cand["us"] = datetime(dt.year, dt.month, dt.day, 9, 30, tzinfo=NY).astimezone(UTC)
        for k, o in cand.items():
            i = (int(o.timestamp() * 1e9) - t0) // BAR_NS
            if 0 <= i < b.n:
                out[k].append(int(i))
    return {k: np.array(v, np.int64) for k, v in out.items()}


def lv_opening_ranges(b: Bars, opens: dict[str, np.ndarray]) -> list[Levels]:
    lh, ll = np.log(b.h), np.log(b.lo)
    res = []
    for oi, m in ((3, 15), (4, 30), (5, 60)):
        k = m // 5
        sub, st, hi, lo = [], [], [], []
        for si, name in enumerate(("asia", "london", "us")):
            o = opens[name]
            o = o[o + k + 1 < b.n]
            st.append(o + k)
            hi.append(np.array([lh[i:i + k].max() for i in o]))
            lo.append(np.array([ll[i:i + k].min() for i in o]))
            sub.append(np.full(len(o), si))
        st_, hi_, lo_, sub_ = (np.concatenate(x) for x in (st, hi, lo, sub))
        # high and low as separate instances; sub = session * 2 + (0 high, 1 low)
        res.append(Levels(oi, np.r_[sub_ * 2, sub_ * 2 + 1].astype(np.int8), np.r_[st_, st_], 72, np.r_[hi_, lo_], np.r_[hi_, lo_]))
    return res


def lv_asia_box(b: Bars) -> Levels:
    ds = day_starts(b)
    ds = ds[ds + 96 + 1 < b.n]
    lh, ll = np.log(b.h), np.log(b.lo)
    hi = np.array([lh[d:d + 96].max() for d in ds])
    lo = np.array([ll[d:d + 96].min() for d in ds])
    st = ds + 96
    return Levels(6, np.r_[np.zeros(len(ds)), np.ones(len(ds))].astype(np.int8), np.r_[st, st], 192, np.r_[hi, lo], np.r_[hi, lo])


def lv_vwap_session(b: Bars) -> Levels:
    ds = day_starts(b)
    ds = ds[ds + BPH < b.n]
    z = np.zeros(len(ds))
    return Levels(7, np.zeros(len(ds), np.int8), ds + BPH, BPD - BPH, z, z, anchor=ds)


def hourly_pivots(b: Bars, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1h fractal pivots from the 5m grid: (anchor bar index = pivot hour open, confirm bar index = open of hour p+k+1,
    kind 0 high / 1 low). Confirmation needs the close of hour p+k."""
    ts = b.ts()
    first = int((-(ts[0] // BAR_NS)) % BPH)
    m = (b.n - first) // BPH
    H = b.h[first:first + m * BPH].reshape(m, BPH).max(1)
    L = b.lo[first:first + m * BPH].reshape(m, BPH).min(1)
    anc, conf, kind = [], [], []
    from numpy.lib.stride_tricks import sliding_window_view

    if m < 2 * k + 2:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.int8)
    wh = sliding_window_view(H, 2 * k + 1).max(1)
    wl = sliding_window_view(L, 2 * k + 1).min(1)
    p = np.arange(k, m - k)
    ph = p[H[p] == wh]
    pl_ = p[L[p] == wl]
    for arr, kd in ((ph, 0), (pl_, 1)):
        anc.append(first + arr * BPH)
        conf.append(first + (arr + k + 1) * BPH)
        kind.append(np.full(len(arr), kd))
    return np.concatenate(anc), np.concatenate(conf), np.concatenate(kind).astype(np.int8)


def lv_vwap_anchored(b: Bars) -> Levels:
    anc, conf, kind = hourly_pivots(b)
    ok = conf + 1 < b.n
    anc, conf, kind = anc[ok], conf[ok], kind[ok]
    z = np.zeros(len(anc))
    return Levels(8, kind, conf, BPD, z, z, anchor=anc)


def lv_value_area(b: Bars) -> tuple[Levels, Levels]:
    from ai_trader.features.profile import session_profile

    ds = day_starts(b)
    ds = ds[ds - BPD >= 0]
    rows = []
    for d in ds:
        sl = slice(d - BPD, d)
        p = session_profile(b.lo[sl], b.h[sl], b.v[sl])
        if p is not None:
            rows.append((d, math.log(p.vah), math.log(p.val), math.log(p.poc)))
    a = np.array(rows) if rows else np.zeros((0, 4))
    st = a[:, 0].astype(np.int64)
    va = Levels(9, np.r_[np.zeros(len(st)), np.ones(len(st))].astype(np.int8), np.r_[st, st], BPD, np.r_[a[:, 1], a[:, 2]], np.r_[a[:, 1], a[:, 2]])
    return va, Levels(10, np.zeros(len(st), np.int8), st, BPD, a[:, 3], a[:, 3])


def pivot_zones(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, k: int = 3, top: int = 8, half_life_days: float = 30.0,
                piv: list | None = None) -> list[dict]:
    """scripts/levels.py zones() logic on arrays (4h bars), vectorised rejection count. `piv` (price, index, kind) may be
    passed precomputed (only pivots confirmed within the arrays)."""
    n = len(c)
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(lo[1:], c[:-1])
    atr = float(np.median(tr[-180:])) if len(tr) else 0.0
    if piv is None:
        piv = []
        for i in range(k, n - k):
            if h[i] == max(h[i - k:i + k + 1]):
                piv.append((h[i], i, "H"))
            if lo[i] == min(lo[i - k:i + k + 1]):
                piv.append((lo[i], i, "L"))
    piv = sorted(piv)
    clusters, cur = [], []
    for p in piv:
        if cur and p[0] - cur[0][0] > 0.6 * atr:
            clusters.append(cur)
            cur = []
        cur.append(p)
    if cur:
        clusters.append(cur)
    out = []
    for cl in clusters:
        prices = [p[0] for p in cl]
        zlo, zhi = min(prices), max(prices)
        if zhi - zlo < 0.25 * atr:
            mid = 0.5 * (zlo + zhi)
            zlo, zhi = mid - 0.125 * atr, mid + 0.125 * atr
        touched = (lo[1:] <= zhi) & (h[1:] >= zlo)
        rej = int((touched & ((c[1:] > zhi) | (c[1:] < zlo))).sum())
        last = max(p[1] for p in cl)
        age_days = (n - 1 - last) * 4 / 24
        score = (2 * len(cl) + rej) * math.exp(-math.log(2) * age_days / half_life_days)
        out.append({"lo": zlo, "hi": zhi, "pivots": len(cl), "rejections": rej, "score": score})
    out.sort(key=lambda z: -z["score"])
    return out[:top]


def lv_pivot_zones(b: Bars, lookback_days: int = 120, k: int = 3) -> Levels:
    ts = b.ts()
    first = int((-(ts[0] // BAR_NS)) % (BPH * 4))
    m = (b.n - first) // (BPH * 4)
    sl = slice(first, first + m * BPH * 4)
    O = b.o[sl][:: BPH * 4]
    H = b.h[sl].reshape(m, BPH * 4).max(1)
    L = b.lo[sl].reshape(m, BPH * 4).min(1)
    C = b.c[sl].reshape(m, BPH * 4)[:, -1]
    from numpy.lib.stride_tricks import sliding_window_view

    wh = sliding_window_view(H, 2 * k + 1).max(1)
    wl = sliding_window_view(L, 2 * k + 1).min(1)
    p = np.arange(k, m - k)
    ph, pll = p[H[p] == wh], p[L[p] == wl]
    ds = day_starts(b)
    lb = lookback_days * 6
    sub, st, lo_, hi_ = [], [], [], []
    for d in ds:
        e = (d - first) // (BPH * 4)          # 4h bars fully closed before the day start: indices < e
        if e < lb // 2 or d - first < 0:
            continue
        a = max(0, e - lb)
        # pivot i confirmed at the close of bar i + k  ->  i + k < e
        piv = [(H[i] - 0.0, i - a, "H") for i in ph[(ph >= a) & (ph + k < e)]] + [(L[i], i - a, "L") for i in pll[(pll >= a) & (pll + k < e)]]
        zs = pivot_zones(O[a:e], H[a:e], L[a:e], C[a:e], k=k, piv=piv)
        for rank, z in enumerate(zs):
            sub.append(rank)
            st.append(d)
            lo_.append(math.log(z["lo"]))
            hi_.append(math.log(z["hi"]))
    return Levels(11, np.array(sub, np.int8), np.array(st, np.int64), BPD, np.array(lo_), np.array(hi_))


def round_step(price: float) -> float:
    return 10.0 ** (math.floor(math.log10(price)) - 1)


def lv_round(b: Bars) -> Levels:
    ds = day_starts(b)
    sub, st, lv = [], [], []
    for d in ds:
        p = b.o[d]
        s = round_step(p)
        below = math.floor(p / s) * s
        above = below + s
        for j, x in enumerate((above, below)):
            if x > 0:
                sub.append(j)
                st.append(d)
                lv.append(math.log(x))
    a = np.array(lv)
    return Levels(12, np.array(sub, np.int8), np.array(st, np.int64), BPD, a, a)


def all_levels(b: Bars) -> list[Levels]:
    hl, pc = lv_prior_day(b)
    va, poc = lv_value_area(b)
    return [hl, pc, lv_prior_week(b), *lv_opening_ranges(b, session_open_index(b)), lv_asia_box(b), lv_vwap_session(b),
            lv_vwap_anchored(b), va, poc, lv_pivot_zones(b), lv_round(b)]


def first_true(M: np.ndarray) -> np.ndarray:
    W = M.shape[1]
    return np.where(M.any(1), M.argmax(1), W)


def machine(lh: np.ndarray, ll: np.ndarray, lc: np.ndarray, s1h: np.ndarray, start: np.ndarray, W: int,
            LO: np.ndarray, HI: np.ndarray, LO_prev: np.ndarray, HI_prev: np.ndarray) -> dict:
    """First-test event machine on 5m bars for N level instances over windows [start, start + W).

    LO / HI: (N, W) band edges per window bar (log), LO_prev / HI_prev: (N,) edges for the bar before the window.
    Returns per instance the approach side s (+1 from below, -1 from above, 0 never armed) and the window offsets of
    touch / sweep / accept / retest (W = no event), plus the mirrored near / far edges per bar for outcome references."""
    n = len(lc)
    N = len(start)
    J = np.arange(W)
    idx = start[:, None] + J[None, :]
    valid = idx < n
    idx = np.minimum(idx, n - 1)
    H, L, C, A = lh[idx], ll[idx], lc[idx], ARM_SIG * s1h[idx]
    D = SWEEP_SIG * s1h[idx]
    pi = np.maximum(start - 1, 0)
    cp, ap = lc[pi], ARM_SIG * s1h[pi]
    Af = np.where(np.isfinite(A), A, np.inf)
    up = valid & (C <= LO - Af)
    dn = valid & (C >= HI + Af)
    fu = np.where(np.isfinite(ap) & (cp <= LO_prev - ap), -1, first_true(up))
    fd = np.where(np.isfinite(ap) & (cp >= HI_prev + ap), -1, first_true(dn))
    s = np.where((fu < W) & (fu <= fd), 1, np.where(fd < W, -1, 0)).astype(np.int8)
    ja = np.where(s > 0, fu, np.where(s < 0, fd, W))
    sp = (s >= 0)[:, None]
    Hm = np.where(sp, H, -L)
    Lm = np.where(sp, L, -H)
    Cm = np.where(sp, C, -C)
    NEAR = np.where(sp, LO, -HI)
    FAR = np.where(sp, HI, -LO)
    ok = valid & (J[None, :] > ja[:, None]) & (s != 0)[:, None]
    touch = first_true(ok & (Hm >= NEAR))
    acc = first_true(ok & (Cm > FAR))
    sweep = first_true(ok & (Hm >= FAR + D) & (Cm < NEAR) & (J[None, :] < acc[:, None]))
    after_acc = valid & (J[None, :] > acc[:, None])
    brk = first_true(after_acc & (Cm <= FAR))
    ret = first_true(after_acc & (J[None, :] <= acc[:, None] + RETEST_BARS) & (J[None, :] < brk[:, None]) & (Lm <= FAR) & (Cm > FAR))
    return {"s": s, "touch": touch, "sweep": sweep, "accept": acc, "retest": ret, "NEAR": NEAR, "FAR": FAR, "N": N, "W": W}


def vwap_matrix(cpv: np.ndarray, cv: np.ndarray, anchor: np.ndarray, start: np.ndarray, W: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Log VWAP from `anchor` through the bar BEFORE each window bar (known at that bar's open), (N, W) and the
    value for the bar before the window."""
    J = np.arange(W)
    j = np.minimum(start[:, None] + J[None, :] - 1, n - 1)       # last bar included
    a = anchor[:, None]
    num = cpv[j + 1] - cpv[a]
    den = cv[j + 1] - cv[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.log(num / den)
    jp = np.minimum(start - 2, n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prev = np.where(jp >= anchor, np.log((cpv[jp + 1] - cpv[anchor]) / (cv[jp + 1] - cv[anchor])), np.nan)
    return m, prev


def run_object(b: Bars, lv: Levels, rng: np.random.Generator, n_placebo: int = 2, chunk_cells: int = 1_500_000) -> pl.DataFrame:
    lh, ll, lc = np.log(b.h), np.log(b.lo), np.log(b.c)
    s1h = b.sig * math.sqrt(BPH)
    n = b.n
    if len(lv.start) == 0:
        return pl.DataFrame()
    st0 = lv.start
    keep = (st0 >= 1) & (st0 < n - 1) & np.isfinite(b.sig[np.minimum(st0, n - 1)])
    st0, sub0, lo0, hi0 = st0[keep], lv.sub[keep], lv.lo[keep], lv.hi[keep]
    anc0 = lv.anchor[keep] if lv.anchor is not None else None
    sday = b.sig[st0] * math.sqrt(BPD)
    # instances: real (plac 0) + placebos (plac 1..n_placebo), offsets in log units
    reps = n_placebo + 1
    off = np.zeros((reps, len(st0)))
    for p in range(1, reps):
        off[p] = rng.choice([-1.0, 1.0], len(st0)) * rng.uniform(0.5, 1.5, len(st0)) * sday
    plac = np.repeat(np.arange(reps), len(st0)).astype(np.int8)
    st = np.tile(st0, reps)
    sub = np.tile(sub0, reps)
    o_ = off.ravel()
    lo_, hi_ = np.tile(lo0, reps) + o_, np.tile(hi0, reps) + o_
    anc = np.tile(anc0, reps) if anc0 is not None else None
    if anc is not None:
        pv = (b.h + b.lo + b.c) / 3 * b.v
        cpv, cv = np.r_[0.0, np.cumsum(pv)], np.r_[0.0, np.cumsum(b.v)]
    W = lv.W
    step = max(1, chunk_cells // W)
    out = []
    for a in range(0, len(st), step):
        sl = slice(a, a + step)
        sst = st[sl]
        if anc is not None:
            vm, vp = vwap_matrix(cpv, cv, anc[sl], sst, W, n)
            LO = vm + o_[sl][:, None]
            HI = LO
            LOp = vp + o_[sl]
            HIp = LOp
        else:
            LO = np.broadcast_to(lo_[sl][:, None], (len(sst), W))
            HI = np.broadcast_to(hi_[sl][:, None], (len(sst), W))
            LOp, HIp = lo_[sl], hi_[sl]
        res = machine(lh, ll, lc, s1h, sst, W, LO, HI, LOp, HIp)
        s = res["s"]
        for ei, ev in enumerate(EVENTS):
            j = res[ev]
            m = j < W
            if not m.any():
                continue
            rows = np.flatnonzero(m)
            jj = j[rows]
            i = sst[rows] + jj
            ss = s[rows].astype(float)
            near_m = res["NEAR"][rows, jj]
            far_m = res["FAR"][rows, jj]
            ref = near_m if ev == "touch" else ss * lc[i]
            sig = b.sig[i]
            ys = {}
            for hz in HORIZONS:
                k = np.minimum(i + hz, n - 1)
                y = (ss * lc[k] - ref) / (sig * math.sqrt(hz))
                y[i + hz >= n] = np.nan
                ys[f"y{hz}"] = y.astype(np.float32)
            out.append(pl.DataFrame({"obj": np.full(len(rows), lv.obj, np.int8), "sub": sub[sl][rows], "ev": np.full(len(rows), ei, np.int8),
                                     "plac": plac[sl][rows], "s": s[rows], "i": i.astype(np.int64), "near": (ss * near_m), "far": (ss * far_m), **ys}))
    return pl.concat(out) if out else pl.DataFrame()


def stage_events(symbols: list[str] | None = None) -> None:
    od = out_dir() / "events"
    od.mkdir(exist_ok=True)
    for sym in symbols or UNIVERSE:
        f = od / f"{sym}.parquet"
        if f.exists():
            print(f"events {sym} cached", flush=True)
            continue
        b = load_bars(sym)
        if b is None:
            continue
        rng = np.random.default_rng(SEED + UNIVERSE.index(sym))
        parts = []
        for lv in all_levels(b):
            d = run_object(b, lv, rng)
            if d.height:
                parts.append(d)
            print(f"  {sym} {OBJECTS[lv.obj]} instances={len(lv.start)} events={d.height}", flush=True)
        ev = pl.concat(parts, how="vertical_relaxed")
        ts = b.ts()
        i = ev["i"].to_numpy()
        cell, vt = regime_arrays(sym, ts + BAR_NS)
        sess = session_codes(ts)
        ev = ev.with_columns(pl.Series("ts", ts[i]), pl.Series("cell", cell[i]), pl.Series("vt", vt[i]), pl.Series("sess", sess[i]),
                             pl.lit(sym).alias("sym"), pl.Series("seg", seg_of(ts[i])))
        ev.write_parquet(f)
        print(f"events {sym} n={ev.height}", flush=True)


# ====================================================================== part 2 inference
def cell_stats(ev: pl.DataFrame, hz: int) -> dict:
    """Real vs placebo for one (object, event) slice at horizon hz, per segment."""
    out = {}
    y = f"y{hz}"
    d = ev.filter(pl.col(y).is_not_nan() & pl.col(y).is_not_null()).with_columns((pl.col("ts") // DAY_NS).alias("day"),
                                                                                  (pl.col("plac") > 0).alias("pl"))
    for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
        s = d.filter(pl.col("seg") == sgi)
        if s.height == 0:
            out[sname] = {"n": 0}
            continue
        g = s.group_by("day", "pl").agg(pl.col(y).sum().alias("S"), pl.len().alias("N"), (pl.col(y) > 0).sum().alias("P"))
        days = np.sort(g["day"].unique().to_numpy())
        D = {dd: k for k, dd in enumerate(days)}
        sr, nr, sp_, np_ = (np.zeros(len(days)) for _ in range(4))
        pr = pp = 0
        for row in g.iter_rows(named=True):
            k = D[row["day"]]
            if row["pl"]:
                sp_[k] += row["S"]
                np_[k] += row["N"]
                pp += row["P"]
            else:
                sr[k] += row["S"]
                nr[k] += row["N"]
                pr += row["P"]
        r = diff_test(days, sr, nr, sp_, np_)
        r["p_cont_real"] = float(pr / max(nr.sum(), 1))
        r["p_cont_pl"] = float(pp / max(np_.sum(), 1))
        out[sname] = r
    return out


def load_events(cols: list[str] | None = None) -> pl.LazyFrame:
    return pl.scan_parquet(str(out_dir() / "events" / "*.parquet")).select(cols) if cols else pl.scan_parquet(str(out_dir() / "events" / "*.parquet"))


NAMED = {"H-GEO10": ("PD_HL", "sweep", 12, None), "H-GEO11": ("OR30", "accept", 48, None), "H-GEO12": ("VWAP_S", "accept", 12, None),
         "H-GEO13": ("ROUND", "touch", 12, None), "H-GEO14": ("ASIA_BOX", "sweep", 12, (7, 11))}


def stage_infer() -> dict:
    cols = ["obj", "sub", "ev", "plac", "s", "ts", "seg", "cell", "vt", "sess", "sym", "y3", "y12", "y48"]
    grid = []
    for oi, on in enumerate(OBJECTS):
        e_obj = load_events(cols).filter(pl.col("obj") == oi).collect()
        for ei, en in enumerate(EVENTS):
            e = e_obj.filter(pl.col("ev") == ei)
            for hz in HORIZONS:
                cs = cell_stats(e, hz)
                grid.append({"obj": on, "ev": en, "h": hz, **{f"{k}_{sg[:3]}": v for sg, d in cs.items() for k, v in d.items()}})
        print(f"infer {on}", flush=True)
    G = pl.DataFrame(grid)
    p_in = G["p_llm"].to_numpy()
    G = G.with_columns(pl.Series("bh_in", bh(p_in, 0.10)))
    # holdout: one-sided in the in-sample direction, BH among survivors
    sgn = np.sign(G["diff_llm"].to_numpy())
    t_ho = G["t_cle"].to_numpy()
    p1 = 1 - np.array([ND.cdf(x) if np.isfinite(x) else np.nan for x in sgn * t_ho])
    G = G.with_columns(pl.Series("p_ho_1s", p1))
    surv = G["bh_in"].to_numpy()
    bh_ho = np.zeros(G.height, bool)
    if surv.any():
        bh_ho[np.flatnonzero(surv)] = bh(p1[surv], 0.10)
    G = G.with_columns(pl.Series("bh_ho", bh_ho))
    od = out_dir()
    G.write_parquet(od / "grid.parquet")
    named = {}
    for hid, (on, en, hz, hrs) in NAMED.items():
        e = load_events(cols).filter((pl.col("obj") == OBJECTS.index(on)) & (pl.col("ev") == EVENTS.index(en))).collect()
        if hrs:
            hh = (pl.col("ts") % DAY_NS) // (3600 * 10**9)
            e = e.filter((hh >= hrs[0]) & (hh < hrs[1]))
        named[hid] = cell_stats(e, hz)
        if hid == "H-GEO11":
            named[hid]["by_session"] = {nm: cell_stats(e.filter(pl.col("sub") // 2 == si), hz) for si, nm in enumerate(("asia", "london", "us"))}
    res = {"grid_n": G.height, "bh_in_survivors": int(surv.sum()), "bh_ho_pass": int(bh_ho.sum()), "named": named}
    (od / "infer_results.json").write_text(json.dumps(res, default=float, indent=1))
    return res


def heterogeneity(cells: list[tuple[str, str, int, tuple | None]], kappa: float = 10.0) -> dict:
    """H-GEO15: per v0 cell and per session real - placebo effects, shrunk (w = episodes / (episodes + kappa)); Wald test
    of equality across groups (in-sample) and Spearman of in-sample vs holdout group effects."""
    from scipy.stats import chi2

    cols = ["obj", "ev", "plac", "ts", "seg", "cell", "sess", "y3", "y12", "y48"]
    out = {}
    for on, en, hz, hrs in cells:
        e = load_events(cols).filter((pl.col("obj") == OBJECTS.index(on)) & (pl.col("ev") == EVENTS.index(en))).collect()
        if hrs:
            hh = (pl.col("ts") % DAY_NS) // (3600 * 10**9)
            e = e.filter((hh >= hrs[0]) & (hh < hrs[1]))
        key = f"{on}/{en}/{hz}"
        res = {}
        for gcol, groups in (("cell", list(range(9))), ("sess", list(range(len(SESSIONS))))):
            eff = {0: {}, 1: {}}
            for g in groups:
                st = cell_stats(e.filter(pl.col(gcol) == g), hz)
                for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
                    r = st[sname]
                    if r.get("n", 0) >= 100 and np.isfinite(r.get("diff", np.nan)):
                        eff[sgi][g] = (r["diff"], r["se"], r.get("days", 0))
            ins = eff[0]
            if len(ins) >= 2:
                d = np.array([v[0] for v in ins.values()])
                se = np.array([v[1] for v in ins.values()])
                w = 1 / se**2
                pooled = (w * d).sum() / w.sum()
                W_ = float(((d - pooled) ** 2 * w).sum())
                pw = float(chi2.sf(W_, len(d) - 1))
            else:
                W_, pw = float("nan"), float("nan")
            common = [g for g in ins if g in eff[1]]
            rho = spearman(np.array([ins[g][0] for g in common]), np.array([eff[1][g][0] for g in common]))[0] if len(common) >= 4 else float("nan")
            pooled_all = np.mean([v[0] for v in ins.values()]) if ins else float("nan")
            shrunk = {str(g): float((v[2] / (v[2] + kappa)) * v[0] + (kappa / (v[2] + kappa)) * pooled_all) for g, v in ins.items()}
            res[gcol] = {"wald": W_, "p": pw, "groups_in": len(ins), "common": len(common), "rho_in_ho": rho,
                         "in": {str(g): v[0] for g, v in ins.items()}, "ho": {str(g): v[0] for g, v in eff[1].items()}, "shrunk_in": shrunk}
        out[key] = res
    return out


# ====================================================================== part 3: rules
COST = {"peg_slip": 1.3, "maker": 2.0, "taker": 5.5, "slip": 2.0, "through": 1.0}
VARIANTS = [(k, tg, T, sc) for k in (1.0, 2.0) for tg in (2.0, 0.0) for T in (12, 48) for sc in (0, 1)]


def hour_factor(b: Bars) -> np.ndarray:
    """Causal hour-of-week variance factor (features.seasonality.seasonal_factor on the hourly u) mapped to 5m bars."""
    from ai_trader.features.seasonality import seasonal_factor

    s = symbol_series(b)
    how = how_cells(s["hts"])
    f = seasonal_factor(how, s["uh"])
    ts = b.ts()
    k = np.clip((ts - s["hts"][0]) // (BAR_NS * BPH), 0, len(f) - 1)
    out = f[k]
    out[(ts < s["hts"][0])] = np.nan
    return out


def simulate(b: Bars, entry_i: np.ndarray, entry_px: np.ndarray, direction: np.ndarray, entry_cost_bp: np.ndarray, sig_i: np.ndarray,
             k: float, tgt: float, T: int, scale: np.ndarray, fcum: tuple[np.ndarray, np.ndarray], taker_extra_bp: float = 0.0,
             fill_bar_stop: bool = True) -> np.ndarray:
    """Net R per trade (worst-case intrabar). entry_i = bar index of the fill; for fills inside a bar (limit / stop entries)
    fill_bar_stop applies the stop on the fill bar too (worst case); for open fills the fill bar is fully live.
    1 R = k * sigma_1h * scale in log units; stop exit taker + slippage, gaps at the open; target maker on a 1 bp
    trade-through; time exit taker at the close of bar entry_i + T - 1; funding = direction * sum of settled rates."""
    n = b.n
    N = len(entry_i)
    if N == 0:
        return np.array([])
    lh, ll, lc, lo_ = np.log(b.h), np.log(b.lo), np.log(b.c), np.log(b.o)
    R = k * sig_i * math.sqrt(BPH) * scale
    E = np.log(entry_px)
    d = direction.astype(float)
    stop = E - d * R
    targ = E + d * tgt * R if tgt > 0 else np.full(N, np.nan)
    J = np.arange(T)
    idx = np.minimum(entry_i[:, None] + J[None, :], n - 1)
    valid = (entry_i[:, None] + J[None, :]) < n
    Hm = np.where(d[:, None] > 0, lh[idx], -ll[idx])
    Lm = np.where(d[:, None] > 0, ll[idx], -lh[idx])
    Om = d[:, None] * lo_[idx]
    Em, Sm = d * E, d * stop
    hit_s = valid & (Lm <= Sm[:, None])
    js = first_true(hit_s)
    th = 1e-4 * COST["through"]
    if tgt > 0:
        Tm = d * targ
        hit_t = valid & (Hm >= Tm[:, None] + th) & (J[None, :] >= (1 if fill_bar_stop else 0))
        jt = first_true(hit_t)
    else:
        jt = np.full(N, T)
    jx = np.minimum(np.minimum(js, jt), np.where(valid.all(1), T - 1, valid.sum(1) - 1))
    stop_first = (js <= jt) & (js < T)
    targ_first = (jt < js) & (jt < T)
    rows = np.arange(N)
    xi = idx[rows, np.minimum(jx, T - 1)]
    # exit price (mirrored log)
    gap = Om[rows, np.minimum(js, T - 1)]
    Xm_stop = np.minimum(Sm, np.where(js > 0, gap, Sm)) - 1e-4 * (COST["slip"])
    Xm_t = d * targ if tgt > 0 else np.zeros(N)
    Xm_time = d * lc[xi] - 1e-4 * COST["slip"]
    Xm = np.where(stop_first, Xm_stop, np.where(targ_first, Xm_t, Xm_time))
    fee_exit = np.where(targ_first, COST["maker"], COST["taker"] + taker_extra_bp)
    gross = Xm - Em
    ts = b.ts()
    ft, fc = fcum
    a = np.searchsorted(ft, ts[entry_i], side="right")
    z = np.searchsorted(ft, ts[xi] + BAR_NS, side="right")
    fund = d * (fc[np.maximum(z, 0)] - fc[np.maximum(a, 0)])
    net = gross - 1e-4 * (entry_cost_bp + fee_exit) - fund
    out = net / R
    out[~np.isfinite(R) | (R <= 0) | ~valid[:, 0]] = np.nan
    return out


def funding_cum(sym: str) -> tuple[np.ndarray, np.ndarray]:
    from ai_trader.storage.lake import Lake

    f = Lake().read_asof("funding", AS_OF, symbols=[sym], filters={"venue": "bybit"})
    if f.height == 0:
        return np.array([0], np.int64), np.array([0.0, 0.0])
    f = f.sort("event_ts").unique("event_ts", keep="last").sort("event_ts")
    ts = f["event_ts"].cast(pl.Datetime("ns", "UTC")).dt.epoch("ns").to_numpy()
    return ts, np.r_[0.0, np.cumsum(np.nan_to_num(f["rate"].to_numpy().astype(float)))]


def rule_entries(b: Bars, e: pl.DataFrame, ev: str, dsign: int, latency: int) -> dict:
    """Entries for one rule. dsign = +1 trade with the approach (continuation), -1 against it (reversal).
    accept / sweep / retest: pegged limit at the open of bar i + 1 + latency (+1.3 bp, maker 2.0).
    touch, reversal: limit at the level on the touch bar (fill on a 1 bp trade-through, maker 2.0).
    touch, continuation: stop entry at the level (taker 5.5 + 2.0 bp)."""
    i = e["i"].to_numpy()
    s = e["s"].to_numpy().astype(float)
    direction = (dsign * s).astype(np.int8)
    if ev == "touch":
        lvl = np.exp(e["near"].to_numpy())
        if dsign < 0:
            # touched bar must trade through the level by 1 bp to fill the resting limit
            lh, ll = np.log(b.h[i]), np.log(b.lo[i])
            thr = 1e-4 * COST["through"]
            filled = np.where(s > 0, lh >= np.log(lvl) + thr, ll <= np.log(lvl) - thr)
            cost = np.full(len(i), COST["maker"])
            px = lvl
        else:
            filled = np.ones(len(i), bool)
            cost = np.full(len(i), COST["taker"])
            px = lvl * np.exp(direction * 1e-4 * COST["slip"])
        ent = i
        fill_bar_stop = True
    else:
        ent = i + 1 + latency
        filled = ent < b.n
        ent = np.minimum(ent, b.n - 1)
        px = b.o[ent] * np.exp(direction * 1e-4 * COST["peg_slip"])
        cost = np.full(len(i), COST["maker"])
        fill_bar_stop = False
    return {"ent": ent, "px": px, "dir": direction, "cost": cost, "filled": filled, "fill_bar_stop": fill_bar_stop, "sig_i": b.sig[i]}


def stage_rules(cells: list[dict]) -> dict:
    """cells: [{"name", "obj", "ev", "h", "dsign", "hours": (a, b) | None}]. Writes per-trade net R per variant."""
    od = out_dir() / "rules"
    od.mkdir(exist_ok=True)
    spreads = {}
    mp = out_dir() / "map_results.json"
    if mp.exists():
        spreads = json.loads(mp.read_text()).get("spread_depth", {}).get("median_spread_by_symbol", {})
    rows = []
    rnd_rows = []
    for sym in UNIVERSE:
        b = load_bars(sym)
        if b is None:
            continue
        evf = out_dir() / "events" / f"{sym}.parquet"
        E = pl.read_parquet(evf).filter(pl.col("plac") == 0)
        fc = funding_cum(sym)
        hf = hour_factor(b)
        ts = b.ts()
        cell_a, _ = regime_arrays(sym, ts + BAR_NS)
        sess_a = session_codes(ts)
        half = 0.5 * float(spreads.get(sym, 0.0) or 0.0)
        rng = np.random.default_rng(SEED + 100 + UNIVERSE.index(sym))
        for c in cells:
            e = E.filter((pl.col("obj") == OBJECTS.index(c["obj"])) & (pl.col("ev") == EVENTS.index(c["ev"])))
            if c.get("hours"):
                hh = (pl.col("ts") % DAY_NS) // (3600 * 10**9)
                e = e.filter((hh >= c["hours"][0]) & (hh < c["hours"][1]))
            e = e.filter(pl.col("i") + 60 < b.n)
            if e.height == 0:
                continue
            base = {"rule": c["name"], "sym": sym}
            for lat in (0, 1):
                en = rule_entries(b, e, c["ev"], c["dsign"], lat)
                f = en["filled"] & np.isfinite(en["sig_i"])
                cols = {}
                for vi, (k, tg, T, sc) in enumerate(VARIANTS):
                    scale = np.sqrt(np.where(np.isfinite(hf[en["ent"]]), hf[en["ent"]], 1.0)) if sc else np.ones(len(f))
                    r = simulate(b, en["ent"][f], en["px"][f], en["dir"][f], en["cost"][f], en["sig_i"][f], k, tg, T, scale[f], fc,
                                 fill_bar_stop=en["fill_bar_stop"])
                    cols[f"v{vi}"] = r.astype(np.float32)
                    if lat == 0 and vi in (0, 5) and half > 0:
                        cols[f"v{vi}_spr"] = simulate(b, en["ent"][f], en["px"][f] * np.exp(en["dir"][f] * 1e-4 * (half if c["ev"] == "touch" and c["dsign"] > 0 else 0)),
                                                      en["dir"][f], en["cost"][f], en["sig_i"][f], k, tg, T, scale[f], fc, taker_extra_bp=half,
                                                      fill_bar_stop=en["fill_bar_stop"]).astype(np.float32)
                rows.append(pl.DataFrame({**{kk: [vv] * int(f.sum()) for kk, vv in base.items()}, "lat": np.full(int(f.sum()), lat, np.int8),
                                          "ts": ts[e["i"].to_numpy()[f]], "cell": e["cell"].to_numpy()[f], "sess": e["sess"].to_numpy()[f],
                                          "dir": en["dir"][f], "seg": e["seg"].to_numpy()[f], **cols}))
                if lat == 0:
                    rows[-1] = rows[-1].with_columns(pl.lit(len(f)).alias("n_signals"))
            # regime-matched random entries: same symbol, cell, session, side; within +-30 days; pegged at the next open
            i = e["i"].to_numpy()
            en0 = rule_entries(b, e, c["ev"], c["dsign"], 0)
            keyb = cell_a.astype(np.int64) * 10 + sess_a
            ri, rdir, rsrc = [], [], []
            for t_ in range(len(i)):
                lo_, hi_ = max(BPD * 8, i[t_] - 30 * BPD), min(b.n - 60, i[t_] + 30 * BPD)
                if hi_ <= lo_:
                    continue
                cand = rng.integers(lo_, hi_, 60)
                cand = cand[(keyb[cand] == keyb[i[t_]]) & (np.abs(cand - i[t_]) > BPH)][:10]
                ri.extend(cand.tolist())
                rdir.extend([en0["dir"][t_]] * len(cand))
                rsrc.extend([t_] * len(cand))
            if ri:
                ri_ = np.array(ri, np.int64)
                rdir_ = np.array(rdir, np.int8)
                ent = ri_ + 1
                px = b.o[ent] * np.exp(rdir_ * 1e-4 * COST["peg_slip"])
                cols = {}
                for vi, (k, tg, T, sc) in enumerate(VARIANTS):
                    scale = np.sqrt(np.where(np.isfinite(hf[ent]), hf[ent], 1.0)) if sc else np.ones(len(ent))
                    cols[f"v{vi}"] = simulate(b, ent, px, rdir_, np.full(len(ent), COST["maker"]), b.sig[ri_], k, tg, T, scale, fc,
                                              fill_bar_stop=False).astype(np.float32)
                rnd_rows.append(pl.DataFrame({"rule": [c["name"]] * len(ent), "sym": [sym] * len(ent), "ts": ts[ri_], "src_ts": ts[i[np.array(rsrc)]],
                                              "seg": seg_of(ts[i[np.array(rsrc)]]), **cols}))
        print(f"rules {sym}", flush=True)
    T_ = pl.concat(rows, how="diagonal_relaxed")
    T_.write_parquet(od / "trades.parquet")
    if rnd_rows:
        pl.concat(rnd_rows, how="diagonal_relaxed").write_parquet(od / "random.parquet")
    return {"trades": T_.height}


def day_boot_mean(ts: np.ndarray, x: np.ndarray, nboot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean per trade with a day-clustered bootstrap CI."""
    ok = np.isfinite(x)
    ts, x = ts[ok], x[ok]
    if len(x) < 10:
        return float("nan"), float("nan"), float("nan")
    day = ts // DAY_NS
    u, inv = np.unique(day, return_inverse=True)
    S = np.bincount(inv, x)
    Nn = np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(u), (nboot, len(u)))
    m = S[k].sum(1) / Nn[k].sum(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def evaluate_rules(n_trials_total: int | None = None) -> dict:
    from ai_trader.research.bos_regimes_infer import dsr, pbo

    od = out_dir() / "rules"
    T = pl.read_parquet(od / "trades.parquet")
    Rn = pl.read_parquet(od / "random.parquet") if (od / "random.parquet").exists() else None
    vcols = [f"v{i}" for i in range(len(VARIANTS))]
    years = [(2023, datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)), (2024, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
             (2025, datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)), (2026, datetime(2026, 1, 1, tzinfo=UTC), CUTOFF)]
    ns = lambda x: int(x.timestamp() * 1e9)
    rules = T["rule"].unique().sort().to_list()
    n_trials = n_trials_total or len(rules) * len(VARIANTS)
    all_sr = []
    per = {}
    for rn in rules:
        t0 = T.filter((pl.col("rule") == rn) & (pl.col("lat") == 0)).sort("ts")
        ins = t0.filter(pl.col("seg") == 0)
        # daily R per variant (in-sample) for SR trials and PBO
        dly = ins.with_columns((pl.col("ts") // DAY_NS).alias("d")).group_by("d").agg([pl.col(v).fill_nan(None).sum() for v in vcols]).sort("d")
        M = dly.select(vcols).to_numpy().astype(float)
        sr = np.nanmean(M, 0) / np.nanstd(M, 0, ddof=1)
        all_sr.extend(sr.tolist())
        # walk-forward
        wf_ts, wf_r, picks = [], [], []
        for y, a, z in years:
            tr = t0.filter((pl.col("ts") < ns(a) - DAY_NS) & (pl.col("seg") == 0))
            te = t0.filter((pl.col("ts") >= ns(a)) & (pl.col("ts") < ns(z)))
            if tr.height < 30 or te.height == 0:
                continue
            means = [float(np.nanmean(tr[v].to_numpy())) for v in vcols]
            best = int(np.nanargmax(means))
            picks.append((y, best))
            wf_ts.append(te["ts"].to_numpy())
            wf_r.append(te[vcols[best]].to_numpy().astype(float))
        wf_ts_ = np.concatenate(wf_ts) if wf_ts else np.array([], np.int64)
        wf_r_ = np.concatenate(wf_r) if wf_r else np.array([])
        full_means = [float(np.nanmean(ins[v].to_numpy())) for v in vcols]
        best_full = int(np.nanargmax(full_means))
        ho = t0.filter(pl.col("seg") == 1)
        ho_r = ho[vcols[best_full]].to_numpy().astype(float)
        lat1 = T.filter((pl.col("rule") == rn) & (pl.col("lat") == 1))
        res = {"n_signals_in": int(ins.height), "n_signals_ho": int(ho.height), "wf_picks": picks,
               "wf_mean_R": day_boot_mean(wf_ts_, wf_r_), "wf_n": int(np.isfinite(wf_r_).sum()),
               "best_variant_in": VARIANTS[best_full], "in_mean_R_best": day_boot_mean(ins["ts"].to_numpy(), ins[vcols[best_full]].to_numpy().astype(float)),
               "in_mean_R_all_variants": dict(zip([str(v) for v in VARIANTS], full_means)),
               "ho_mean_R": day_boot_mean(ho["ts"].to_numpy(), ho_r), "ho_n": int(np.isfinite(ho_r).sum()),
               "lat5_in_mean_R_best": float(np.nanmean(lat1.filter(pl.col("seg") == 0)[vcols[best_full]].to_numpy())) if lat1.height else float("nan"),
               "lat5_ho_mean_R_best": float(np.nanmean(lat1.filter(pl.col("seg") == 1)[vcols[best_full]].to_numpy())) if lat1.height else float("nan"),
               "pbo": pbo(np.nan_to_num(M), n_blocks=16) if M.shape[0] >= 64 else None}
        for vi in (0, 5):
            if f"v{vi}_spr" in t0.columns:
                res[f"spread_sens_v{vi}"] = {"in": float(np.nanmean(ins[f"v{vi}_spr"].to_numpy())), "ho": float(np.nanmean(ho[f"v{vi}_spr"].to_numpy())),
                                             "in_base": float(np.nanmean(ins[f"v{vi}"].to_numpy())), "ho_base": float(np.nanmean(ho[f"v{vi}"].to_numpy()))}
        if wf_ts_.size:
            dd = pl.DataFrame({"d": wf_ts_ // DAY_NS, "r": wf_r_}).group_by("d").agg(pl.col("r").fill_nan(None).sum()).sort("d")
            res["_wf_daily"] = dd["r"].to_numpy()
        if Rn is not None:
            rr = Rn.filter(pl.col("rule") == rn)
            for sgi, sname in ((0, "llm_in_sample"), (1, "clean_holdout")):
                rs = rr.filter(pl.col("seg") == sgi)
                real = t0.filter(pl.col("seg") == sgi)
                if rs.height and real.height:
                    a_ = real[vcols[best_full]].to_numpy().astype(float)
                    b_ = rs[vcols[best_full]].to_numpy().astype(float)
                    # difference of means with a day bootstrap of the real and random sets (independent resampling)
                    m1 = day_boot_mean(real["ts"].to_numpy(), a_)
                    m2 = day_boot_mean(rs["src_ts"].to_numpy(), b_)
                    rng = np.random.default_rng(SEED)
                    da, db = real["ts"].to_numpy() // DAY_NS, rs["src_ts"].to_numpy() // DAY_NS
                    days = np.unique(np.r_[da, db])
                    ia, ib = np.searchsorted(days, da), np.searchsorted(days, db)
                    okA, okB = np.isfinite(a_), np.isfinite(b_)
                    SA, NA = np.bincount(ia[okA], a_[okA], len(days)), np.bincount(ia[okA], minlength=len(days))
                    SB, NB = np.bincount(ib[okB], b_[okB], len(days)), np.bincount(ib[okB], minlength=len(days))
                    k = rng.integers(0, len(days), (2000, len(days)))
                    diff = SA[k].sum(1) / np.maximum(NA[k].sum(1), 1) - SB[k].sum(1) / np.maximum(NB[k].sum(1), 1)
                    res[f"vs_random_{sname}"] = {"rule": m1[0], "random": m2[0], "diff": m1[0] - m2[0],
                                                 "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))], "n_random": int(okB.sum())}
        per[rn] = res
    all_sr = np.array([x for x in all_sr if np.isfinite(x)])
    for rn, res in per.items():
        x = res.pop("_wf_daily", None)
        if x is not None and len(x) > 30:
            res["dsr_wf"] = dsr(np.asarray(x, float), all_sr, n_trials, 365)
    out = {"n_trials": n_trials, "variants": [list(v) for v in VARIANTS], "rules": per}
    (out_dir() / "rules_results.json").write_text(json.dumps(out, default=float, indent=1))
    return out


# ====================================================================== liquidation clusters (H-GEO17, exploratory)
def stage_liq() -> dict:
    from ai_trader.storage.lake import Lake

    lake = Lake()
    rows = []
    rng = np.random.default_rng(SEED + 999)
    for sym in UNIVERSE:
        q = lake.read_asof("liquidations", AS_OF, symbols=[sym])
        if q.height < 1000:
            continue
        b = load_bars(sym)
        if b is None:
            continue
        lt = q["event_ts"].cast(pl.Datetime("ns", "UTC")).dt.epoch("ns").to_numpy()
        lp = q["price"].to_numpy().astype(float)
        notional = lp * q["size"].to_numpy().astype(float)
        ts = b.ts()
        w = math.log1p(0.0025)
        st, lo_, sub = [], [], []
        for hstart in range(int(-(-lt.min() // (3600 * 10**9))) + 24, int(ts[-1] // (3600 * 10**9)) - 24):
            t = hstart * 3600 * 10**9
            i = (t - ts[0]) // BAR_NS
            if i < 1 or i >= b.n - BPD:
                continue
            m = (lt >= t - DAY_NS) & (lt < t)
            if m.sum() < 5:
                continue
            k = np.floor(np.log(lp[m]) / w).astype(np.int64)
            uk, inv = np.unique(k, return_inverse=True)
            tot = np.bincount(inv, notional[m])
            px = math.log(b.c[i - 1])
            centre = (uk + 0.5) * w
            for side, sel in ((0, centre > px), (1, centre < px)):
                if sel.any():
                    j = np.flatnonzero(sel)[np.argmax(tot[sel])]
                    st.append(i)
                    lo_.append(centre[j])
                    sub.append(side)
        if not st:
            continue
        lv = Levels(13, np.array(sub, np.int8), np.array(st, np.int64), BPD, np.array(lo_), np.array(lo_))
        d = run_object(b, lv, rng)
        if d.height:
            rows.append(d.with_columns(pl.Series("ts", ts[d["i"].to_numpy()]), pl.lit(sym).alias("sym"), pl.Series("seg", seg_of(ts[d["i"].to_numpy()]))))
        print(f"liq {sym} instances={len(st)}", flush=True)
    out = {}
    if rows:
        E = pl.concat(rows, how="vertical_relaxed")
        E.write_parquet(out_dir() / "liq_events.parquet")
        for ei, en in enumerate(EVENTS[:3]):
            out[en] = cell_stats(E.filter(pl.col("ev") == ei), 12)
    (out_dir() / "liq_results.json").write_text(json.dumps(out, default=float, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["map", "events", "infer", "liq"])
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.stage == "map":
        stage_map()
    elif a.stage == "events":
        stage_events(a.symbols.split(",") if a.symbols else None)
    elif a.stage == "infer":
        print(json.dumps(stage_infer(), default=float)[:2000])
    elif a.stage == "liq":
        print(json.dumps(stage_liq(), default=float)[:2000])


if __name__ == "__main__":
    main()


# ====================================================================== verdicts (registered success rules, applied once)
SEGS = ("llm_in_sample", "clean_holdout")


def _v(sup: bool, con: bool) -> str:
    return "supports" if sup else ("contradicts" if con else "inconclusive")


def hourly_sigma_bp() -> float:
    """Median over symbols of the median hourly sigma (sqrt of the trailing 168-hour mean RV) in the holdout, in bp."""
    vals = []
    for sym in UNIVERSE:
        b = load_bars(sym)
        if b is None:
            continue
        s = b.sig[b.ts() >= int(CUTOFF.timestamp() * 1e9)]
        s = s[np.isfinite(s)]
        if len(s):
            vals.append(float(np.median(s)) * math.sqrt(BPH) * 1e4)
    return float(np.median(vals))


def verdicts_map(res: dict, sigma_1h_bp: float) -> list[tuple[str, str, str, dict]]:
    out = []
    P = res["profiles"]
    v, vo = P["var"], P["volume"]

    def share(x, sg):
        r = [m["rho"] >= 0.5 for m in x["months"] if m["seg"] == sg and np.isfinite(m["rho"])]
        return float(np.mean(r)) if r else float("nan")

    m = {"var_rho_holdout": v["rho_holdout"], "vol_rho_holdout": vo["rho_holdout"], "var_month_share_all": v["month_share_rho_ge_0.5"],
         "vol_month_share_all": vo["month_share_rho_ge_0.5"], "var_share_in": share(v, 0), "vol_share_in": share(vo, 0),
         "var_share_ho": share(v, 1), "vol_share_ho": share(vo, 1)}
    out.append(("H-GEO1", "llm_in_sample", _v(m["var_share_in"] >= 0.8 and m["vol_share_in"] >= 0.8, m["var_share_in"] < 0.5 or m["vol_share_in"] < 0.5), m))
    out.append(("H-GEO1", "clean_holdout", _v(v["rho_holdout"] >= 0.8 and vo["rho_holdout"] >= 0.8 and m["var_share_ho"] >= 0.8 and m["vol_share_ho"] >= 0.8,
                                              v["rho_holdout"] < 0.5 or vo["rho_holdout"] < 0.5), m))
    out.append(("H-GEO1", "pooled", _v(v["rho_holdout"] >= 0.8 and vo["rho_holdout"] >= 0.8 and m["var_month_share_all"] >= 0.8 and m["vol_month_share_all"] >= 0.8,
                                       v["rho_holdout"] < 0.5 or vo["rho_holdout"] < 0.5), m))
    D = P["direction"]
    cost_R = 15.0 / sigma_1h_bp
    tilt = D["tilt_holdout"]
    net = tilt["gross_R_per_hour"] - cost_R if tilt["cells"] else float("nan")
    m2 = {"p_joint_in": D["llm_in_sample"]["p_joint"], "chi2_in": D["llm_in_sample"]["chi2"], "n_abs_t_gt_2_in": D["llm_in_sample"]["n_abs_t_gt_2"],
          "p_joint_ho": D["clean_holdout"]["p_joint"], "rank_rho": D["rank_corr_in_vs_ho"]["rho"], "rank_p": D["rank_corr_in_vs_ho"]["p"],
          "tilt_cells": tilt["cells"], "tilt_gross_R_h": tilt["gross_R_per_hour"], "cost_R_h": cost_R, "tilt_net_R_h": net, "sigma_1h_bp": sigma_1h_bp}
    out.append(("H-GEO2", "llm_in_sample", _v(m2["p_joint_in"] < 0.01, m2["p_joint_in"] > 0.5), m2))
    rr = D["rank_corr_in_vs_ho"]
    out.append(("H-GEO2", "clean_holdout", _v(rr["rho"] > 0.2 and rr["p"] < 0.05 and np.isfinite(net) and net > 0,
                                              rr["rho"] <= 0 or (np.isfinite(net) and net < 0)), m2))
    R = res["releases"]
    for sg in SEGS:
        r = R[sg]
        a, z = r["0-5"], r["30-60"]
        out.append(("H-GEO3", sg, _v(a["ratio"] >= 3 and a["lo"] >= 1.5 and z["ratio"] < 0.5 * a["ratio"], a["ratio"] < 1.5), r))
    F = res["funding_vol"]
    for sg in SEGS:
        r = F[sg]["08_16"]
        out.append(("H-GEO4", sg, _v(r["ratio"] >= 1.2 and r["lo"] > 1, r["ratio"] < 1.05), F[sg]))
    FD = res["funding_dir"]
    hi_, ho_ = FD["abs_ge_1bp"]["llm_in_sample"]["spre"], FD["abs_ge_1bp"]["clean_holdout"]["spre"]
    out.append(("H-GEO5", "llm_in_sample", _v(hi_["mean"] > 0 and hi_["t"] > 2, not (hi_["t"] > 0)), FD))
    out.append(("H-GEO5", "clean_holdout", _v(np.sign(ho_["mean"]) == np.sign(hi_["mean"]) and hi_["mean"] > 0 and ho_["t"] > 1.2816,
                                              np.sign(ho_["mean"]) != np.sign(hi_["mean"]) or not (hi_["t"] > 0)), FD))
    S = res["session_opens"]
    for sg in SEGS:
        us = S[sg].get("us_all", {})
        r_, ci = us.get("mean_of_ratios", float("nan")), us.get("mean_ratio_ci", [np.nan] * 3)
        off = [S[sg].get(f"us_offclock_{k}", {}).get("mean_of_ratios", float("nan")) for k in ("edt", "est")]
        sup = r_ >= 1.5 and ci[1] > 1.2 and (sg == "clean_holdout" or all(x < 1.2 for x in off))
        out.append(("H-GEO6", sg, _v(sup, r_ < 1.2), S[sg]))
    M = res["month_end"]
    for sg in SEGS:
        r = M[sg]
        out.append(("H-GEO7", sg, _v(r["ratio"] >= 1.3 and r["lo"] > 1, r["ratio"] < 1.1), r))
    SD = res.get("spread_depth", {})
    if "rho_spread_var" in SD:
        sup = SD["rho_spread_var"] > 0.5 and SD["ci90_spread"][0] > 0 and SD["rho_depth_var"] < -0.5 and SD["ci90_depth"][1] < 0
        con = SD["rho_spread_var"] < 0 or SD["rho_depth_var"] > 0
        out.append(("H-GEO8", "clean_holdout", _v(sup, con), {k: SD[k] for k in ("days", "symbols", "rho_spread_var", "ci90_spread", "rho_depth_var", "ci90_depth")}))
    return out


def record_all(rows: list[tuple[str, str, str, dict]], note: str, data: str) -> None:
    from ai_trader.research.hypotheses import record

    for hid, sg, verdict, metrics in rows:
        record(hid, sg, verdict, json.loads(json.dumps(metrics, default=float)), note=note, data=data)
        print(hid, sg, verdict, flush=True)
