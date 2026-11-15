"""TradFi session effects on Bybit TradFi perps (block TF: H-TP1, H-TF1..H-TF5, Program 2, 2026-11-11).

Design fixed before any result was seen (diary 2026-11-10 "B2 TradFi session block: design fixed before results").

Off-hours behaviour (measured first, diary 2026-11-10): Bybit equity perps trade 24/7. Last and index 1h bars exist for
every hour including weekends; weekday overnight hours carry ~0.3x regular-hours turnover and ~0.45x the mean |r|, the
weekend ~0.1x turnover and ~0.2x |r| (quiet, not frozen); the index moves on weekends too and the last-index basis is
~4-5 bp at all hours. So an overnight "gap" is a *traded drift* on the perp between the NYSE close and the NYSE open, not
a discontinuity in the bar series.

Calendar
  * NYSE regular session 09:30-16:00 America/New_York (13:30-20:00 UTC under EDT, 14:30-21:00 UTC under EST), weekdays
    except NYSE holidays; early closes at 13:00 ET. Holidays/early closes are hard-coded for the years covered by the data
    (2026); any other year raises, so the calendar can't silently treat a holiday as a session.
  * CME Globex Sunday reopen 18:00 ET (22:00 UTC under EDT, 23:00 UTC under EST).

Prices from 1h bars (bar = [event_ts, event_ts + 1h))
  * P(T) = close of the bar ending at T (whole hours only). NYSE close 16:00 ET is a bar boundary. The open 09:30 ET is
    not: the **first cash hour** is the 1h bar containing the open, 09:00-10:00 ET, and the session "open" price is
    P(09:00 ET). The 30 pre-open minutes 09:00-09:30 are thus assigned to the cash session: overnight = prev close
    (16:00 ET) -> 09:00 ET, intraday = 09:00 ET -> 16:00 ET (7 bars; 4 on early closes). Robustness: open at 10:00 ET.
  * sigma_on   RMS of the previous 20 overnight log returns (>= 10), known at the open (weekends/holidays included).
  * sigma_bar  sqrt(mean per-bar regular-session realised variance over the previous 20 sessions) (>= 10): per-bar
               cash-session sigma known at the open; sigma_day = sigma_bar * sqrt(7).
  * Risk units for strategies: net log return / sigma_bar (position 1/sigma fixed at entry, as research.evaluate).

Segments by session open (TF5: Monday 07:00 UTC) vs research.knowledge_cutoff. Holm over the six primary one-sided
p-values of the block, per segment. Costs 7.5 bp per side (15 bp round trip).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

NY = ZoneInfo("America/New_York")
H = 3600

# Source: NYSE Group press release "NYSE Group Announces 2026, 2027 and 2028 Holiday and Early Closings Calendar"
# (Business Wire, Dec 2025; nyse.com/markets/hours-calendars). 2026 full-day closures and 13:00 ET early closes.
NYSE_HOLIDAYS: dict[int, frozenset[date]] = {
    2026: frozenset({
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # Martin Luther King, Jr. Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day (observed; Jul 4 is a Saturday)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day
    }),
}
NYSE_EARLY_CLOSES: dict[int, frozenset[date]] = {
    2026: frozenset({date(2026, 11, 27), date(2026, 12, 24)}),  # 13:00 ET
}

EQUITY = ["TSLAUSDT", "INTCUSDT", "GOOGLUSDT", "AAPLUSDT", "MSTRUSDT", "NVDAUSDT", "MUUSDT", "QQQUSDT", "SOXLUSDT"]
COMMODITIES = ["CLUSDT", "BZUSDT", "XAUUSDT", "XAGUSDT"]
EXCLUDED = {
    "SPCXUSDT": "under 120 days of history",
    "SPYUSDT/MSFTUSDT/METAUSDT/AVGOUSDT/TSMUSDT/ORCLUSDT/HOODUSDT/...": "listed 2026-11-05: too short",
}
COST_SIDE_BPS = 7.5
COST_RT = 2 * COST_SIDE_BPS / 1e4
TRAIL, TRAIL_MIN = 20, 10
MIN_DATES, MIN_WEEKENDS = 30, 10
N_DRAWS = 1000
HYPS = ["H-TP1", "H-TF1", "H-TF2", "H-TF3", "H-TF4", "H-TF5"]
SEGS = ("llm_in_sample", "clean_holdout", "pooled")


# ---------------------------------------------------------------- calendar

def _covered(y: int) -> None:
    if y not in NYSE_HOLIDAYS:
        raise ValueError(f"NYSE calendar not hard-coded for {y}; add its holidays and early closes")


def is_nyse_trading_day(d: date) -> bool:
    _covered(d.year)
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS[d.year]


def nyse_session(d: date) -> tuple[datetime, datetime] | None:
    """(open, close) in UTC for NYSE trading day d, None if the exchange is closed. DST via America/New_York."""
    if not is_nyse_trading_day(d):
        return None
    close_t = time(13, 0) if d in NYSE_EARLY_CLOSES[d.year] else time(16, 0)
    o = datetime.combine(d, time(9, 30), NY).astimezone(UTC)
    c = datetime.combine(d, close_t, NY).astimezone(UTC)
    return o, c


def cme_sunday_reopen(d: date) -> datetime:
    """CME Globex Sunday reopen (18:00 ET) in UTC for Sunday d."""
    if d.weekday() != 6:
        raise ValueError(f"{d} is not a Sunday")
    return datetime.combine(d, time(18, 0), NY).astimezone(UTC)


def bar_containing(ts: datetime) -> datetime:
    """Start of the 1h bar containing ts (bars start on whole UTC hours; NY offsets are whole hours)."""
    return ts.replace(minute=0, second=0, microsecond=0)


def first_cash_bar(open_utc: datetime) -> datetime:
    """The first cash hour = the 1h bar containing the open (09:30 ET -> the 09:00 ET bar)."""
    return bar_containing(open_utc)


def nyse_sessions(start: date, end: date) -> list[dict]:
    """Trading sessions in [start, end] with the previous session's close (across weekends/holidays/early closes)."""
    out, prev = [], None
    d = start - timedelta(days=10)
    while d <= end:
        s = nyse_session(d) if d.year in NYSE_HOLIDAYS else None
        if s is not None:
            o, c = s
            if d >= start and prev is not None:
                fb = first_cash_bar(o)
                out.append({"date": d, "open_utc": o, "close_utc": c, "fb_start": fb, "prev_close_utc": prev,
                            "early_close": d in NYSE_EARLY_CLOSES[d.year], "n_bars": int((c - fb).total_seconds() // H),
                            "gap_hours": int((fb - prev).total_seconds() // H),
                            "after_closure": (fb - prev) > timedelta(hours=24)})
            prev = c
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------- price grid

@dataclass
class Grid:
    """Hourly bars of one symbol keyed by bar start (epoch hours)."""
    o: dict[int, float] = field(default_factory=dict)
    h: dict[int, float] = field(default_factory=dict)
    l: dict[int, float] = field(default_factory=dict)  # noqa: E741
    c: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_bars(cls, bars: pl.DataFrame) -> Grid:
        g = cls()
        for ts, o, hi, lo, c in bars.select("event_ts", "open", "high", "low", "close").iter_rows():
            k = _eh(ts)
            g.o[k], g.h[k], g.l[k], g.c[k] = float(o), float(hi), float(lo), float(c)
        return g

    def p(self, t: datetime | int) -> float:
        """P(T) = close of the bar ending at T."""
        k = t if isinstance(t, int) else _eh(t)
        return self.c.get(k - 1, math.nan)

    def path(self, t0: datetime, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Prices P(t0 + k h) k=0..n, and high/low of the n bars starting at t0. None if any bar is missing."""
        k0 = _eh(t0)
        ks = range(k0, k0 + n)
        if (k0 - 1) not in self.c or any(k not in self.c for k in ks):
            return None
        px = np.array([self.c[k0 - 1]] + [self.c[k] for k in ks])
        return px, np.array([self.h[k] for k in ks]), np.array([self.l[k] for k in ks])


def _eh(ts: datetime) -> int:
    return int(ts.timestamp() // H)


def _from_eh(k: int) -> datetime:
    return datetime.fromtimestamp(k * H, UTC)


# ---------------------------------------------------------------- session table (equities)

def session_table(grid: Grid, sessions: list[dict]) -> pl.DataFrame:
    """One row per NYSE session with complete bars from the previous close to this close. Point in time: trailing
    sigmas use only sessions strictly before this one (all known at the open)."""
    rows = []
    for s in sessions:
        n = s["n_bars"]
        pa = grid.path(s["fb_start"], n)
        p_prev = grid.p(s["prev_close_utc"])
        if pa is None or not np.isfinite(p_prev):
            continue
        px, hi, lo = pa
        r = np.diff(np.log(px))
        on = math.log(px[0] / p_prev)
        rows.append({
            "date": s["date"], "open_utc": s["open_utc"], "fb_start": s["fb_start"], "close_utc": s["close_utc"],
            "n_bars": n, "early_close": s["early_close"], "after_closure": s["after_closure"], "gap_hours": s["gap_hours"],
            "p_prev": p_prev, "p_open": px[0], "p_10": px[1], "p_11": px[2] if n >= 2 else math.nan, "p_close": px[-1],
            "p_last_start": px[-2], "or_high": hi[0], "or_low": lo[0],
            "sess_high": float(hi.max()), "sess_low": float(lo.min()),
            "on": on, "intraday": math.log(px[-1] / px[0]),
            "on_10": math.log(px[1] / p_prev), "intraday_10": math.log(px[-1] / px[1]),
            "first2": math.log(px[2] / px[0]) if n >= 2 else math.nan,
            "fh": math.log(px[1] / p_prev), "last": float(r[-1]),
            "rv_bar": float(np.mean(r**2)), "rv_sess": float(np.sum(r**2)),
            "px": px.tolist(), "hi": hi.tolist(), "lo": lo.tolist(),
        })
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows).sort("date")

    def trail(x: pl.Expr) -> pl.Expr:  # mean over the previous TRAIL sessions (this one excluded), then sqrt
        return x.rolling_mean(TRAIL, min_samples=TRAIL_MIN).shift(1).sqrt()

    return df.with_columns(
        trail(pl.col("on") ** 2).alias("sigma_on"),
        trail(pl.col("rv_bar")).alias("sigma_bar"),
        trail(pl.col("fh") ** 2).alias("sigma_fh"),
        trail(pl.col("last") ** 2).alias("sigma_last"),
    ).with_columns((pl.col("sigma_bar") * math.sqrt(7)).alias("sigma_day"))


def gap_fill(row: dict) -> bool:
    """Did price revisit the previous close during the session (any cash bar's [low, high] contains p_prev)?
    Gap up (p_open > p_prev): filled if the session low <= p_prev; gap down: session high >= p_prev."""
    if row["p_open"] >= row["p_prev"]:
        return row["sess_low"] <= row["p_prev"]
    return row["sess_high"] >= row["p_prev"]


def orb_trade(row: dict) -> dict | None:
    """First 1h close beyond the first-cash-bar high/low (bars 2..n-1; the last bar closes at the bell), entered at that
    close in its direction and held to the NYSE close."""
    px, n = row["px"], row["n_bars"]
    for k in range(2, n):  # px[k] = close of bar k-1 (0-based), bars after the first one
        side = 1 if px[k] > row["or_high"] else (-1 if px[k] < row["or_low"] else 0)
        if side:
            return {"side": side, "entry_k": k, "gross": side * math.log(px[n] / px[k])}
    return None


# ---------------------------------------------------------------- statistics

def nw_t(x) -> tuple[float, float, int]:
    from ai_trader.research.evaluate import newey_west_t

    return newey_west_t(np.asarray(x, dtype=float))


def date_mean_test(df: pl.DataFrame, col: str, by: str = "date") -> dict:
    """Average across symbols per date (clusters by time), NW t over dates."""
    if df.height == 0:
        return {"mean": math.nan, "t": math.nan, "n_dates": 0, "n_obs": 0}
    d = df.filter(pl.col(col).is_finite()).group_by(by).agg(pl.col(col).mean()).sort(by)
    m, t, n = nw_t(d[col].to_numpy())
    return {"mean": m, "t": t, "n_dates": n, "n_obs": int(df.filter(pl.col(col).is_finite()).height)}


def cluster_ols(x: np.ndarray, y: np.ndarray, cl: np.ndarray) -> dict:
    """OLS y = a + b x with CR1 standard errors clustered by `cl`."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((2, 2))
    groups = np.unique(cl)
    for g in groups:
        s = X[cl == g].T @ u[cl == g]
        meat += np.outer(s, s)
    n, k, G = len(y), 2, len(groups)
    adj = G / (G - 1) * (n - 1) / (n - k) if G > 1 else 1.0
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"a": float(beta[0]), "b": float(beta[1]), "t_a": float(beta[0] / se[0]), "t_b": float(beta[1] / se[1]),
            "n": int(n), "n_dates": int(G)}


def p_one_sided(t: float, df: int) -> float:
    from scipy.stats import t as tdist

    if not np.isfinite(t) or df < 2:
        return math.nan
    return float(tdist.sf(t, df))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(((k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)), key=lambda kv: kv[1])
    out = {k: math.nan for k in pvals}
    run, m = 0.0, len(items)
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def segment_of(ts: datetime) -> str:
    from ai_trader.research.evaluate import knowledge_cutoff

    return "llm_in_sample" if ts < knowledge_cutoff() else "clean_holdout"


def seg_filter(df: pl.DataFrame, seg: str) -> pl.DataFrame:
    return df if seg == "pooled" or df.height == 0 else df.filter(pl.col("segment") == seg)


# ---------------------------------------------------------------- data

def load_equity(as_of: datetime, symbols: list[str] = EQUITY) -> tuple[pl.DataFrame, dict[str, Grid]]:
    from ai_trader.research.frame import market_frame

    tabs, grids = [], {}
    for s in symbols:
        f = market_frame(s, "1h", as_of, funding=False, oi=False)
        if f.height == 0:
            continue
        g = Grid.from_bars(f)
        grids[s] = g
        d0, d1 = f["event_ts"].min().date(), (f["event_ts"].max() - timedelta(hours=1)).date()
        st = session_table(g, nyse_sessions(d0 + timedelta(days=1), d1))
        if st.height:
            tabs.append(st.with_columns(pl.lit(s).alias("symbol")))
    df = pl.concat(tabs, how="diagonal_relaxed")
    df = df.with_columns(pl.col("open_utc").map_elements(segment_of, return_dtype=pl.Utf8).alias("segment"))
    return df, grids


def weekend_table(grid: Grid, symbol: str, d0: date, d1: date, literal_23: bool = False) -> list[dict]:
    """H-TF5 rows: R1 = Sunday CME reopen -> Monday 07:00 UTC, R2 = Monday 07:00 -> 12:00 UTC."""
    rows = []
    d = d0 + timedelta(days=(6 - d0.weekday()) % 7)
    while d <= d1:
        reopen = datetime.combine(d, time(23, 0), UTC) if literal_23 else cme_sunday_reopen(d)
        mon = d + timedelta(days=1)
        t7, t12 = datetime.combine(mon, time(7, 0), UTC), datetime.combine(mon, time(12, 0), UTC)
        p0, p7, p12 = grid.p(reopen), grid.p(t7), grid.p(t12)
        # sigma: RMS of 1h returns over the previous Monday-Friday (UTC weekday bars), known at Monday 07:00
        prev_mon = d - timedelta(days=6)
        k0, k1 = _eh(datetime.combine(prev_mon, time(0, 0), UTC)), _eh(datetime.combine(d - timedelta(days=1), time(0, 0), UTC))
        ks = [k for k in range(k0, k1) if k in grid.c and (k - 1) in grid.c]
        r = np.array([math.log(grid.c[k] / grid.c[k - 1]) for k in ks])
        mid = all((k in grid.c) for k in range(_eh(t7), _eh(t12)))
        if all(np.isfinite([p0, p7, p12])) and len(r) >= 60 and mid:
            sig = float(np.sqrt(np.mean(r**2)))
            r1, r2 = math.log(p7 / p0), math.log(p12 / p7)
            rows.append({"symbol": symbol, "sunday": d, "monday": mon, "t7": t7, "reopen_utc": reopen, "r1": r1, "r2": r2,
                         "sigma_bar": sig, "r1_z": r1 / (sig * math.sqrt(_eh(t7) - _eh(reopen))),
                         "cont": (np.sign(r1) * r2 / (sig * math.sqrt(5))) if r1 != 0 else math.nan,
                         "segment": segment_of(t7)})
        d += timedelta(days=7)
    return rows


def load_commodities(as_of: datetime, symbols: list[str] = COMMODITIES, literal_23: bool = False) -> tuple[pl.DataFrame, dict[str, Grid]]:
    from ai_trader.research.frame import market_frame

    rows, grids = [], {}
    for s in symbols:
        f = market_frame(s, "1h", as_of, funding=False, oi=False)
        if f.height == 0:
            continue
        g = Grid.from_bars(f)
        grids[s] = g
        rows += weekend_table(g, s, f["event_ts"].min().date() + timedelta(days=7), f["event_ts"].max().date(), literal_23)
    return pl.DataFrame(rows), grids


# ---------------------------------------------------------------- hypothesis tests

def _valid(df: pl.DataFrame, *cols: str) -> pl.DataFrame:
    return df.filter(pl.all_horizontal([pl.col(c).is_finite() & pl.col(c).is_not_null() for c in cols]))


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    df = _valid(df, "sigma_on", "sigma_bar")
    from ai_trader.research.calendar import gap_fill_baseline

    fills = [gap_fill(r) for r in df.select("p_open", "p_prev", "sess_low", "sess_high").iter_rows(named=True)]
    gap_bar = (df["on"].abs() / df["sigma_bar"]).to_numpy()
    base = gap_fill_baseline(gap_bar, df["n_bars"].to_numpy())
    base_rv = gap_fill_baseline(df["on"].abs().to_numpy() / np.sqrt(df["rv_sess"].to_numpy()), np.ones(df.height))
    return df.with_columns(
        (pl.col("on") / pl.col("sigma_on")).alias("on_z"),
        pl.Series("filled", fills, dtype=pl.Float64),
        pl.Series("fill_base", base),
        pl.Series("fill_base_rv", base_rv),
        (-pl.col("on").sign() * pl.col("first2") / (pl.col("sigma_bar") * math.sqrt(2))).alias("rev"),
        (pl.col("on") - pl.col("intraday")).alias("night_minus_day"),
        ((pl.col("on") - pl.col("intraday")) / pl.col("sigma_day")).alias("night_minus_day_z"),
        (pl.col("on_10") - pl.col("intraday_10")).alias("night_minus_day_10"),
        (pl.col("fh") / pl.col("sigma_fh")).alias("x_fh"),
        (pl.col("last") / pl.col("sigma_last")).alias("y_last"),
    )


def test_tp1(df: pl.DataFrame) -> dict:
    d = _valid(df, "rev")
    out = date_mean_test(d, "rev")
    big = d.filter(pl.col("on_z").abs() > 1)
    out["variant_abs_on_gt_1sigma"] = date_mean_test(big, "rev")
    out["after_closure"] = date_mean_test(d.filter(pl.col("after_closure")), "rev")
    out["weekday_overnight"] = date_mean_test(d.filter(~pl.col("after_closure")), "rev")
    out["frac_reverting"] = float((d["rev"] > 0).mean()) if d.height else math.nan
    out["per_symbol"] = {s: date_mean_test(d.filter(pl.col("symbol") == s), "rev") for s in sorted(d["symbol"].unique())}
    return out


def test_tf1(df: pl.DataFrame) -> dict:
    g = df.filter(pl.col("on_z").abs() > 1)
    if g.height == 0:
        return {"n_events": 0}
    e = (g["filled"] - g["fill_base"]).to_numpy()
    dates = g["date"].to_numpy()
    sums = pl.DataFrame({"d": dates, "e": e}).group_by("d").agg(pl.col("e").sum())["e"].to_numpy()
    se = math.sqrt(float(np.sum(sums**2))) / len(e)
    diff = float(np.mean(e))
    e_rv = (g["filled"] - g["fill_base_rv"]).to_numpy()
    sums_rv = pl.DataFrame({"d": dates, "e": e_rv}).group_by("d").agg(pl.col("e").sum())["e"].to_numpy()
    per = {}
    for s in sorted(g["symbol"].unique()):
        x = g.filter(pl.col("symbol") == s)
        per[s] = {"n": x.height, "fill_rate": float(x["filled"].mean()), "baseline": float(x["fill_base"].mean())}
    return {"n_events": int(len(e)), "n_dates": int(len(sums)), "fill_rate": float(g["filled"].mean()),
            "baseline": float(g["fill_base"].mean()), "diff": diff, "se": se, "z": diff / se if se > 0 else math.nan,
            "diff_expost_rv_baseline": float(np.mean(e_rv)),
            "z_expost_rv_baseline": float(np.mean(e_rv) / (math.sqrt(float(np.sum(sums_rv**2))) / len(e_rv))),
            "mean_abs_gap_on_sigma": float(g["on_z"].abs().mean()), "per_symbol": per}


def test_tf2(df: pl.DataFrame) -> dict:
    out = date_mean_test(df, "night_minus_day")
    out["mean_on"] = date_mean_test(df, "on")["mean"]
    out["mean_intraday"] = date_mean_test(df, "intraday")["mean"]
    out["t_on"] = date_mean_test(df, "on")["t"]
    out["t_intraday"] = date_mean_test(df, "intraday")["t"]
    out["vol_scaled"] = date_mean_test(_valid(df, "night_minus_day_z"), "night_minus_day_z")
    out["open_at_10ET"] = date_mean_test(df, "night_minus_day_10")
    out["per_symbol"] = {s: {**date_mean_test(df.filter(pl.col("symbol") == s), "night_minus_day"),
                             "mean_on": float(df.filter(pl.col("symbol") == s)["on"].mean()),
                             "mean_intraday": float(df.filter(pl.col("symbol") == s)["intraday"].mean())}
                         for s in sorted(df["symbol"].unique())}
    return out


def orb_trades(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in df.iter_rows(named=True):
        tr = orb_trade(r)
        if tr:
            net = tr["gross"] - COST_RT
            rows.append({"symbol": r["symbol"], "date": r["date"], "segment": r["segment"], "n_bars": r["n_bars"],
                         "side": tr["side"], "entry_k": tr["entry_k"], "gross_bps": tr["gross"] * 1e4, "net_bps": net * 1e4,
                         "net_ru": net / r["sigma_bar"], "gross_ru": tr["gross"] / r["sigma_bar"]})
    return pl.DataFrame(rows)


def test_tf4(df: pl.DataFrame) -> dict:
    d = _valid(df, "x_fh", "y_last")
    if d.height < 10:
        return {"n": d.height}
    reg = cluster_ols(d["x_fh"].to_numpy(), d["y_last"].to_numpy(), np.array([str(x) for x in d["date"].to_list()]))
    agree = float(((d["x_fh"] * d["y_last"]) > 0).mean())
    per = {}
    for s in sorted(d["symbol"].unique()):
        x = d.filter(pl.col("symbol") == s)
        if x.height >= 10:
            r = cluster_ols(x["x_fh"].to_numpy(), x["y_last"].to_numpy(), np.arange(x.height))
            per[s] = {"b": r["b"], "t_b": r["t_b"], "n": r["n"]}
    return {**reg, "sign_agreement": agree, "per_symbol": per}


def test_tf5(w: pl.DataFrame) -> dict:
    d = _valid(w, "cont") if w.height else w
    if d.height == 0:
        return {"n_dates": 0}
    out = date_mean_test(d, "cont", by="sunday")
    out["sign_agreement"] = float((d["cont"] > 0).mean())
    out["per_symbol"] = {s: date_mean_test(d.filter(pl.col("symbol") == s), "cont", by="sunday") for s in sorted(d["symbol"].unique())}
    big = d.filter(pl.col("r1_z").abs() > 1)
    out["variant_abs_r1_gt_1sigma"] = date_mean_test(big, "cont", by="sunday")
    return out


# ---------------------------------------------------------------- strategy views

def _trade_stats(t: pl.DataFrame, by: str = "date") -> dict:
    if t.height == 0:
        return {"n_trades": 0}
    s = date_mean_test(t, "net_ru", by=by)
    return {"n_trades": t.height, "n_dates": s["n_dates"], "mean_net_bps": float(t["net_bps"].mean()),
            "mean_gross_bps": float(t["gross_bps"].mean()), "mean_net_ru": s["mean"], "t_net_ru": s["t"],
            "hit_rate": float((t["net_bps"] > 0).mean()), "long_frac": float((t["side"] > 0).mean())}


def random_session_baseline(trades: pl.DataFrame, sess: pl.DataFrame, payoff, seed: int = 7, draws: int = N_DRAWS) -> dict:
    """Random entries matched in time and holding: each trade is replayed on a random session of the same symbol,
    segment and session length, with the same side, entry offset and exit rule. payoff(row, trade) -> net log ret."""
    if trades.height == 0:
        return {}
    rng = np.random.default_rng(seed)
    pools: dict[tuple, list[dict]] = {}
    for r in sess.iter_rows(named=True):
        pools.setdefault((r["symbol"], r["segment"], r["n_bars"]), []).append(r)
    tl = trades.to_dicts()
    actual = float(trades["net_ru"].mean())
    means = np.empty(draws)
    for i in range(draws):
        v = []
        for tr in tl:
            pool = pools[(tr["symbol"], tr["segment"], tr["n_bars"])]
            r = pool[rng.integers(len(pool))]
            v.append(payoff(r, tr) / r["sigma_bar"])
        means[i] = np.mean(v)
    return {"actual_mean_net_ru": actual, "random_mean_net_ru": float(means.mean()), "random_p05": float(np.quantile(means, 0.05)),
            "random_p95": float(np.quantile(means, 0.95)), "beats_random_frac": float((actual > means).mean()), "draws": draws}


def _pay_tp1(r: dict, tr: dict) -> float:
    return tr["side"] * r["first2"] - COST_RT


def _pay_orb(r: dict, tr: dict) -> float:
    px = r["px"]
    return tr["side"] * math.log(px[r["n_bars"]] / px[tr["entry_k"]]) - COST_RT


def _pay_tf4(r: dict, tr: dict) -> float:
    return tr["side"] * r["last"] - COST_RT


def _pay_gapfade(r: dict, tr: dict) -> float:
    """Fade entered at the open with a take-profit `dist` away, else exit at the close."""
    side, dist = tr["side"], tr["dist"]
    target = r["p_open"] * math.exp(side * dist)
    hit = r["sess_high"] >= target if side > 0 else r["sess_low"] <= target
    return (dist if hit else side * r["intraday"]) - COST_RT


def _mk(t: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(t) if t else pl.DataFrame()


def strategy_tp1(df: pl.DataFrame, conditional: bool) -> tuple[pl.DataFrame, dict]:
    d = _valid(df, "first2")
    if conditional:
        d = d.filter(pl.col("on_z").abs() > 1)
    t = []
    for r in d.iter_rows(named=True):
        if r["on"] == 0:
            continue
        side = -int(np.sign(r["on"]))
        net = side * r["first2"] - COST_RT
        t.append({"symbol": r["symbol"], "date": r["date"], "segment": r["segment"], "n_bars": r["n_bars"], "side": side,
                  "gross_bps": side * r["first2"] * 1e4, "net_bps": net * 1e4, "net_ru": net / r["sigma_bar"]})
    return _mk(t), {}


def strategy_gapfade(df: pl.DataFrame) -> pl.DataFrame:
    g = df.filter(pl.col("on_z").abs() > 1)
    t = []
    for r in g.iter_rows(named=True):
        side, dist = -int(np.sign(r["on"])), abs(r["on"])
        net = _pay_gapfade(r, {"side": side, "dist": dist})
        t.append({"symbol": r["symbol"], "date": r["date"], "segment": r["segment"], "n_bars": r["n_bars"], "side": side,
                  "dist": dist, "gross_bps": (net + COST_RT) * 1e4, "net_bps": net * 1e4, "net_ru": net / r["sigma_bar"]})
    return _mk(t)


def strategy_tf4(df: pl.DataFrame) -> pl.DataFrame:
    t = []
    for r in df.iter_rows(named=True):
        if r["fh"] == 0:
            continue
        side = int(np.sign(r["fh"]))
        net = side * r["last"] - COST_RT
        t.append({"symbol": r["symbol"], "date": r["date"], "segment": r["segment"], "n_bars": r["n_bars"], "side": side,
                  "gross_bps": side * r["last"] * 1e4, "net_bps": net * 1e4, "net_ru": net / r["sigma_bar"]})
    return _mk(t)


def strategy_overnight(df: pl.DataFrame, grids: dict[str, Grid], seed: int = 11) -> dict:
    """Long from the NYSE close to the next 09:00 ET (the overnight leg), vs long intraday and vs random windows of the
    same length (any start hour of the same symbol and segment, long)."""
    out = {}
    for seg in SEGS:
        d = seg_filter(df, seg)
        if d.height == 0:
            continue
        t_on = d.with_columns(((pl.col("on") - COST_RT) / pl.col("sigma_bar")).alias("net_ru"))
        t_id = d.with_columns(((pl.col("intraday") - COST_RT) / pl.col("sigma_bar")).alias("net_ru"))
        rng = np.random.default_rng(seed)
        # random windows: per trade, a random start hour within the segment's span of that symbol, same holding length
        rows = d.select("symbol", "fb_start", "gap_hours", "sigma_bar", "segment").to_dicts()
        spans: dict[tuple, list[int]] = {}
        for r in rows:
            spans.setdefault((r["symbol"], r["segment"]), []).append(_eh(r["fb_start"]))
        means = []
        for _ in range(200):
            v = []
            for r in rows:
                ks = spans[(r["symbol"], r["segment"])]
                lo_k, hi_k = min(ks) - 24 * 3, max(ks)
                g = grids[r["symbol"]]
                for _try in range(20):
                    k = int(rng.integers(lo_k, hi_k))
                    a, b = g.p(k), g.p(k + r["gap_hours"])
                    if np.isfinite(a) and np.isfinite(b):
                        v.append((math.log(b / a) - COST_RT) / r["sigma_bar"])
                        break
            means.append(np.mean(v))
        means = np.array(means)
        a_on = float(t_on["net_ru"].mean())
        s_on, s_id = date_mean_test(t_on, "net_ru"), date_mean_test(t_id, "net_ru")
        out[seg] = {"long_overnight": {"n_trades": d.height, "mean_net_bps": float(d["on"].mean() * 1e4 - 2 * COST_SIDE_BPS),
                                       "mean_net_ru": s_on["mean"], "t_net_ru": s_on["t"], "hit_rate": float((d["on"] > COST_RT).mean())},
                    "long_intraday": {"n_trades": d.height, "mean_net_bps": float(d["intraday"].mean() * 1e4 - 2 * COST_SIDE_BPS),
                                      "mean_net_ru": s_id["mean"], "t_net_ru": s_id["t"], "hit_rate": float((d["intraday"] > COST_RT).mean())},
                    "random_windows": {"random_mean_net_ru": float(means.mean()), "beats_random_frac": float((a_on > means).mean()), "draws": len(means)}}
    return out


def strategy_tf5(w: pl.DataFrame, grids: dict[str, Grid], seed: int = 13) -> dict:
    d = _valid(w, "cont")
    if d.height == 0:
        return {}
    t = d.with_columns(pl.col("r1").sign().cast(pl.Int64).alias("side")).with_columns(
        (pl.col("side") * pl.col("r2") * 1e4).alias("gross_bps"),
        ((pl.col("side") * pl.col("r2") - COST_RT) * 1e4).alias("net_bps"),
        ((pl.col("side") * pl.col("r2") - COST_RT) / pl.col("sigma_bar")).alias("net_ru"))
    out = {}
    rng = np.random.default_rng(seed)
    for seg in SEGS:
        x = seg_filter(t, seg)
        if x.height == 0:
            continue
        st = _trade_stats(x, by="sunday")
        # random baseline: same symbol, side and window (07:00 -> 12:00 UTC) on a random weekday in the segment
        rows = x.to_dicts()
        means = []
        for _ in range(N_DRAWS):
            v = []
            for r in rows:
                g = grids[r["symbol"]]
                for _try in range(30):
                    day = r["monday"] + timedelta(days=int(rng.integers(-70, 71)))
                    if day.weekday() >= 5 or (seg != "pooled" and segment_of(datetime.combine(day, time(7), UTC)) != seg):
                        continue
                    a, b = g.p(datetime.combine(day, time(7), UTC)), g.p(datetime.combine(day, time(12), UTC))
                    if np.isfinite(a) and np.isfinite(b):
                        v.append((r["side"] * math.log(b / a) - COST_RT) / r["sigma_bar"])
                        break
            means.append(np.mean(v) if v else np.nan)
        means = np.array(means)
        actual = float(x["net_ru"].mean())  # trade mean, the same statistic as each draw
        st["random"] = {"actual_mean_net_ru": actual, "random_mean_net_ru": float(np.nanmean(means)),
                        "random_p05": float(np.nanquantile(means, 0.05)), "random_p95": float(np.nanquantile(means, 0.95)),
                        "beats_random_frac": float((actual > means).mean()), "draws": N_DRAWS}
        out[seg] = st
    return out


def strategies(df: pl.DataFrame, w: pl.DataFrame, grids_eq: dict[str, Grid], grids_cm: dict[str, Grid]) -> dict:
    out: dict = {}
    sets = {
        "TP1_fade_overnight_first2h": (strategy_tp1(df, False)[0], _pay_tp1),
        "TP1_fade_overnight_first2h_if_gt_1sigma": (strategy_tp1(df, True)[0], _pay_tp1),
        "TF1_gap_fade_to_prev_close": (strategy_gapfade(df), _pay_gapfade),
        "TF3_orb_to_close": (orb_trades(df), _pay_orb),
        "TF4_last_hour_follow_first_hour": (strategy_tf4(_valid(df, "x_fh", "y_last")), _pay_tf4),
    }
    for name, (t, pay) in sets.items():
        out[name] = {}
        for seg in SEGS:
            x = seg_filter(t, seg)
            if x.height == 0:
                continue
            out[name][seg] = {**_trade_stats(x), "random": random_session_baseline(x, seg_filter(df, seg), pay)}
    out["TF2_long_overnight"] = strategy_overnight(df, grids_eq)
    out["TF5_follow_reopen_into_europe"] = strategy_tf5(w, grids_cm)
    return out


# ---------------------------------------------------------------- block evaluation

def off_hours_profile(as_of: datetime, symbols: list[str]) -> dict:
    """Turnover and |r| per NY-time bucket relative to regular hours (10-15 ET), last and index price."""
    from ai_trader.storage.lake import Lake

    lake, out = Lake(), {}
    for s in symbols:
        L = lake.bars(s, "1h", as_of).select("event_ts", pl.col("close").alias("cl"), "turnover")
        idx = lake.bars(s, "1h", as_of, price_type="index").select("event_ts", pl.col("close").alias("ci"))
        b = L.join(idx, on="event_ts", how="left").sort("event_ts").with_columns(
            pl.col("cl").log().diff().alias("rl"), pl.col("ci").log().diff().alias("ri"),
            ((pl.col("cl") / pl.col("ci")).log() * 1e4).alias("basis"),
            pl.col("event_ts").dt.convert_time_zone("America/New_York").alias("ny"))
        b = b.with_columns(pl.col("ny").dt.weekday().alias("wd"), pl.col("ny").dt.hour().alias("h"))
        sess = (pl.when((pl.col("wd") >= 6) | ((pl.col("wd") == 5) & (pl.col("h") >= 20)) | ((pl.col("wd") == 7))).then(pl.lit("weekend"))
                .when(pl.col("h") == 9).then(pl.lit("first_cash_bar_09ET"))
                .when(pl.col("h").is_between(10, 15)).then(pl.lit("regular_10_15ET"))
                .when(pl.col("h").is_between(4, 8)).then(pl.lit("pre_04_08ET"))
                .when(pl.col("h").is_between(16, 19)).then(pl.lit("post_16_19ET"))
                .otherwise(pl.lit("overnight_20_03ET")))
        b = b.with_columns(sess.alias("bucket"))
        ref = b.filter(pl.col("bucket") == "regular_10_15ET")
        rt, ra = ref["turnover"].median(), ref["rl"].abs().mean()
        g = b.group_by("bucket").agg(pl.len().alias("bars"), (pl.col("turnover").median() / rt).alias("turnover_rel"),
                                     (pl.col("rl").abs().mean() / ra).alias("abs_ret_rel"), (pl.col("rl") == 0).mean().alias("zero_ret_frac"),
                                     (pl.col("ri") == 0).mean().alias("index_zero_ret_frac"), pl.col("ri").abs().mean().alias("index_abs_ret"),
                                     pl.col("basis").abs().median().alias("abs_basis_bp_median"))
        out[s] = {r["bucket"]: {k: v for k, v in r.items() if k != "bucket"} for r in g.iter_rows(named=True)}
    return out


def _judge(enough: bool, met: bool, sign_ok: bool, holm_ok: bool) -> str:
    if not enough:
        return "inconclusive"
    if met and holm_ok:
        return "supports"
    if not sign_ok:
        return "contradicts"
    return "inconclusive"


def run(as_of: datetime) -> dict:
    eq, grids_eq = load_equity(as_of)
    eq = add_derived(eq)
    w, grids_cm = load_commodities(as_of)
    w23, _ = load_commodities(as_of, literal_23=True)
    res: dict = {"segments": {}, "meta": {"as_of": as_of.isoformat(), "equity": sorted(eq["symbol"].unique().to_list()),
                                          "commodities": sorted(w["symbol"].unique().to_list()) if w.height else [],
                                          "excluded": EXCLUDED, "cost_bps_per_side": COST_SIDE_BPS}}
    for seg in SEGS:
        d = seg_filter(eq, seg)
        ws = seg_filter(w, seg)
        r = {"H-TP1": test_tp1(d), "H-TF1": test_tf1(d), "H-TF2": test_tf2(d), "H-TF4": test_tf4(d), "H-TF5": test_tf5(ws)}
        r["H-TF5"]["literal_23UTC"] = test_tf5(seg_filter(w23, seg))
        orb = orb_trades(d)
        r["H-TF3"] = {**_trade_stats(orb), "random": random_session_baseline(orb, d, _pay_orb),
                      "per_symbol": {s: _trade_stats(orb.filter(pl.col("symbol") == s)) for s in sorted(orb["symbol"].unique())} if orb.height else {}}
        r["coverage"] = {"sessions": int(d.height), "dates": int(d["date"].n_unique()) if d.height else 0,
                         "first": str(d["date"].min()) if d.height else None, "last": str(d["date"].max()) if d.height else None,
                         "weekends_tf5": int(ws["sunday"].n_unique()) if ws.height else 0}
        res["segments"][seg] = r
    # primary statistics, Holm, verdicts
    for seg in SEGS:
        r = res["segments"][seg]
        prim = {
            "H-TP1": (r["H-TP1"]["mean"], r["H-TP1"]["t"], r["H-TP1"]["n_dates"]),
            "H-TF1": (r["H-TF1"].get("diff", math.nan), r["H-TF1"].get("z", math.nan), r["H-TF1"].get("n_dates", 0)),
            "H-TF2": (r["H-TF2"]["mean"], r["H-TF2"]["t"], r["H-TF2"]["n_dates"]),
            "H-TF3": (r["H-TF3"].get("mean_net_ru", math.nan), r["H-TF3"].get("t_net_ru", math.nan), r["H-TF3"].get("n_dates", 0)),
            "H-TF4": (r["H-TF4"].get("b", math.nan), r["H-TF4"].get("t_b", math.nan), r["H-TF4"].get("n_dates", 0)),
            "H-TF5": (r["H-TF5"].get("mean", math.nan), r["H-TF5"].get("t", math.nan), r["H-TF5"].get("n_dates", 0)),
        }
        p = {h: p_one_sided(t, n - 1) for h, (_, t, n) in prim.items()}
        ph = holm(p)
        v = {}
        for h, (m, t, n) in prim.items():
            enough = n >= (MIN_WEEKENDS if h == "H-TF5" else MIN_DATES)
            if h == "H-TF1":
                met = np.isfinite(m) and m > 0.05 and t > 2
            elif h == "H-TF3":
                met = np.isfinite(m) and m > 0 and t > 2 and r["H-TF3"].get("random", {}).get("beats_random_frac", 0) > 0.95
            else:
                met = np.isfinite(m) and m > 0 and t > 2
            sign_ok = bool(np.isfinite(m) and m > 0)
            verdict = _judge(enough, bool(met), sign_ok, bool(np.isfinite(ph[h]) and ph[h] < 0.05))
            reason = "" if enough else f"insufficient data: {n} {'weekends' if h == 'H-TF5' else 'dates'} < {MIN_WEEKENDS if h == 'H-TF5' else MIN_DATES}"
            v[h] = {"stat": m, "t": t, "n_units": n, "p_raw": p[h], "p_holm": ph[h], "criterion_met": bool(met), "verdict": verdict, "reason": reason}
        r["verdicts"] = v
    res["off_hours"] = off_hours_profile(as_of, EQUITY + COMMODITIES)
    res["strategies"] = strategies(eq, w, grids_eq, grids_cm)
    return res


# ---------------------------------------------------------------- report

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, float | np.floating):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, date | datetime):
        return x.isoformat()
    return x


def write_json(res: dict, path: Path) -> None:
    path.write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    import sys

    from ai_trader.core.config import load
    from ai_trader.core.time import now

    out = run(now())
    stem = sys.argv[1] if len(sys.argv) > 1 else "tradfi_sessions_2026-11-11"
    p = load().root / "research" / "reports" / f"{stem}.json"
    write_json(out, p)
    print(p)
