"""Block OPX: options after a volatility compression, the user's call + short deep-ITM put structure, tail-profit
continuation and scaling out (H-OPX1..H-OPX4, research/hypotheses.yaml, registered 2026-11-14 before any data).

Design frozen in the diary before evaluation (2026-11-14 16:57 UTC, "OPX design frozen before evaluation"). Summary:

  data      Deribit BTC option trades (history API) aggregated per instrument-hour (options_download); Bybit BTCUSDT /
            ETHUSDT 1h bars, Bybit funding, deribit_svi daily smile summary and DVOL from the lake (point-in-time).
  detector  at 08:01 UTC on day d: 7-day RV (168 hourly log returns ending 08:00) in the bottom quintile of its trailing
            365 daily values AND 30-day ATM IV (iv_atm_30, DVOL/100 where missing) in the bottom tercile; entry at the
            day d + 1 08:00 fixing; one event per 14 days.
  pricing   tier 1 'real': the instrument's hourly row in 05:00-10:59 whose last trade is nearest 08:00, its trade IV
            transported to the 08:00 forward; tier 2 'model': Black-76 on a 3-point smile (quadratic in log-moneyness
            through the 25-delta put / ATM / 25-delta call, linear outside the pillars) rebuilt from ATM / RR25 / BF25.
  costs     per option leg per transaction max(half-spread, 0.0003 BTC + 0.0125% x 1 BTC) in BTC; perp / spot 5 bps per
            side; Bybit funding on perp legs.
  units     USD per 1-BTC option (Black-76 value; Deribit BTC price = V / S); BTC-settled P&L reported alongside.

Nothing here places orders. Everything is deterministic and testable (tests/test_options_compression.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from scipy import optimize
from scipy.special import ndtr, ndtri

# ====================================================================== constants (frozen in the diary)

RV_HOURS = 168
TRAIL_DAYS = 365
MIN_TRAIL = 300
RV_PCT = 0.20
IV_PCT = 1.0 / 3.0
DECLUSTER_DAYS = 14
HOLD_DAYS = 14
MIN_DTE = 21
LISTING_DAYS = 7
CALL_DELTA = 0.25
PUT_DELTA = -0.85
FEE_FLOOR_BTC = 0.0003 + 0.000125   # 0.0003 BTC per option + 0.0125% of the 1-BTC notional
PERP_COST = 0.0005                  # 5 bps per side (perp and spot)
PILLARS = (7, 30, 90)
Z25 = float(-ndtri(0.25))           # 0.6745
QUOTE_HOURS = (5, 11)               # rows with hour in [05:00, 11:00)
PROT_PUTS = 0.5                     # downside plan (descriptive): 0.5 x 25-delta puts
PROT_PERP = 0.25                    #                              + 0.25 short perp
Q_GRID = (50, 75, 90, 95, 99)
SEGS = ("llm_in_sample", "clean_holdout")


# ====================================================================== Black-76 (r = 0, forward measure)

def _npdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / math.sqrt(2 * math.pi)


def b76(F, K, T, sigma, cp):
    """Black-76 value (USD per 1 unit of underlying). Vectorised; cp 'C' / 'P' (scalar) or +1 / -1 array."""
    F, K, T, sigma = (np.asarray(x, float) for x in (F, K, T, sigma))
    w = _w(cp)
    intr = np.maximum(w * (F - K), 0.0)
    ok = (T > 0) & (sigma > 0)
    v = np.where(ok, sigma * np.sqrt(np.where(ok, T, 1.0)), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = (np.log(F / K) + 0.5 * v * v) / v
        d2 = d1 - v
        val = w * (F * ndtr(w * d1) - K * ndtr(w * d2))
    out = np.where(ok, val, intr)
    return float(out) if out.ndim == 0 else out


def _w(cp):
    a = np.asarray(cp)
    if a.dtype.kind in "if":
        return a.astype(float)
    return np.where(a == "C", 1.0, -1.0)


def b76_greeks(F, K, T, sigma, cp) -> dict:
    """Forward delta, vega (per 1.00 vol), theta (per calendar day), gamma. Vectorised."""
    F, K, T, sigma = (np.asarray(x, float) for x in (F, K, T, sigma))
    w = _w(cp)
    ok = (T > 0) & (sigma > 0)
    Ts = np.where(ok, T, 1.0)
    v = sigma * np.sqrt(Ts)
    d1 = (np.log(F / K) + 0.5 * v * v) / np.where(ok, v, 1.0)
    delta = np.where(ok, w * ndtr(w * d1), np.where(w * (F - K) > 0, w, 0.0))
    vega = np.where(ok, F * _npdf(d1) * np.sqrt(Ts), 0.0)
    theta = np.where(ok, -F * _npdf(d1) * sigma / (2 * np.sqrt(Ts)) / 365.0, 0.0)
    gamma = np.where(ok, _npdf(d1) / (F * np.where(ok, v, 1.0)), 0.0)
    return {"delta": delta, "vega": vega, "theta": theta, "gamma": gamma}


def implied_vol(V, F, K, T, cp, lo=0.01, hi=6.0) -> float:
    """Black-76 implied vol by Brent; NaN outside the no-arbitrage bounds."""
    w = 1.0 if cp == "C" else -1.0
    intr = max(w * (F - K), 0.0)
    if not (T > 0) or not (V > intr) or V >= (F if w > 0 else K):
        return float("nan")
    f = lambda s: b76(F, K, T, s, cp) - V
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi, xtol=1e-7))


# ====================================================================== smile model (fallback pricing)

def pillar_interp(Tdays, atm, rr, bf):
    """ATM / RR / BF at tenor Tdays from the 7/30/90-day pillars (arrays shape (n, 3)): ATM total variance linear in T,
    RR and BF linear in T, flat outside [7, 90]."""
    T = np.clip(np.asarray(Tdays, float), PILLARS[0], PILLARS[-1])
    atm, rr, bf = (np.atleast_2d(np.asarray(x, float)) for x in (atm, rr, bf))
    P = np.asarray(PILLARS, float)
    j = np.clip(np.searchsorted(P, T, side="right") - 1, 0, len(P) - 2)
    t0, t1 = P[j], P[j + 1]
    a = (T - t0) / (t1 - t0)
    idx = np.arange(len(T)) if atm.shape[0] == len(T) else np.zeros(len(T), int)
    w0, w1 = atm[idx, j] ** 2 * t0, atm[idx, j + 1] ** 2 * t1
    atm_t = np.sqrt((w0 + a * (w1 - w0)) / T)
    rr_t = rr[idx, j] + a * (rr[idx, j + 1] - rr[idx, j])
    bf_t = bf[idx, j] + a * (bf[idx, j + 1] - bf[idx, j])
    return atm_t, rr_t, bf_t


def smile_iv(k, T_years, atm, rr, bf):
    """3-point smile at log-moneyness k = ln(K/F): quadratic through (k_25P, s_25P), (0, ATM), (k_25C, s_25C), linear
    outside the 25-delta pillars with the quadratic's slope there, clipped to [0.5, 3] x ATM. Vectorised."""
    k, t, atm, rr, bf = (np.asarray(x, float) for x in (k, T_years, atm, rr, bf))
    sc, sp = atm + bf + rr / 2, atm + bf - rr / 2
    st = np.sqrt(np.maximum(t, 1e-8))
    kc = Z25 * sc * st + 0.5 * sc * sc * t
    kp = -Z25 * sp * st + 0.5 * sp * sp * t
    # solve s = atm + b k + c k^2 through both wings
    det = kc * kp * kp - kp * kc * kc
    b = ((sc - atm) * kp * kp - (sp - atm) * kc * kc) / det
    c = ((sp - atm) * kc - (sc - atm) * kp) / det
    q = atm + b * k + c * k * k
    right = sc + (b + 2 * c * kc) * (k - kc)
    left = sp + (b + 2 * c * kp) * (k - kp)
    s = np.where(k > kc, right, np.where(k < kp, left, q))
    return np.clip(s, 0.5 * atm, 3.0 * atm)


# ====================================================================== costs and settlement

def option_cost_btc(half_spread_btc) -> np.ndarray | float:
    """Registered cost per option leg per transaction (BTC per 1-BTC option): max(half-spread, fee floor)."""
    hs = np.nan_to_num(np.asarray(half_spread_btc, float), nan=0.0)
    out = np.maximum(hs, FEE_FLOOR_BTC)
    return float(out) if out.ndim == 0 else out


def inverse_pnl(qty: float, V0: float, S0: float, V1: float, S1: float, cost_btc: float = 0.0) -> dict:
    """P&L of `qty` Deribit inverse options (BTC-priced) between two marks. V are USD (Black-76) values.
    usd   = linear USD P&L (premium funded in USD): qty (V1 - V0) - cost_btc x S at each transaction is the caller's job;
    btc   = BTC-settled P&L qty (V1/S1 - V0/S0) - cost_btc;  btc_usd = btc x S1."""
    btc = qty * (V1 / S1 - V0 / S0) - cost_btc
    return {"usd": qty * (V1 - V0), "btc": btc, "btc_usd": btc * S1}


# ====================================================================== compression detector

def trailing_pct(x: np.ndarray, window: int = TRAIL_DAYS, min_n: int = MIN_TRAIL) -> np.ndarray:
    """Share of the trailing `window` values (including today) <= today's value; NaN with < min_n finite values."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        w = x[max(0, i - window + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_n:
            out[i] = float(np.mean(w <= x[i]))
    return out


def rv7_at_0800(bars: pl.DataFrame) -> pl.DataFrame:
    """Per UTC day: sum of squared hourly log returns of the 168 bars ending 08:00 (bars with event_ts 08:00 - 168 h ..
    07:00), >= 150 present. bars: event_ts (bar open), close."""
    b = bars.select(pl.col("event_ts").dt.replace_time_zone("UTC") if bars["event_ts"].dtype.time_zone is None else pl.col("event_ts"),
                    "close").sort("event_ts")
    grid = pl.DataFrame({"event_ts": pl.datetime_range(b["event_ts"].min(), b["event_ts"].max(), "1h", time_zone="UTC", eager=True)})
    g = grid.join(b, on="event_ts", how="left").with_columns(pl.col("event_ts").dt.cast_time_unit("us"))
    lc = np.log(g["close"].to_numpy().astype(float))
    r = np.r_[np.nan, np.diff(lc)]
    r2 = pl.Series(r * r)
    s = r2.rolling_sum(RV_HOURS, min_samples=150).to_numpy()
    g = g.with_columns(pl.Series("rv7", s))
    # value at the bar opening 07:00 = window ending at 08:00 on that day
    out = g.filter(pl.col("event_ts").dt.hour() == 7).select(pl.col("event_ts").dt.date().alias("date"), "rv7")
    return out


def detector(rv: pl.DataFrame, iv: pl.DataFrame) -> pl.DataFrame:
    """rv: date, rv7 (window ending 08:00); iv: date, iv (known 08:01). Returns per date: rv7, iv, pct_rv, pct_iv,
    trigger, entry (entry date = d + 1, declustered by DECLUSTER_DAYS). Causal: row d uses rows <= d only."""
    d = rv.join(iv, on="date", how="full", coalesce=True).sort("date")
    grid = pl.DataFrame({"date": pl.date_range(d["date"].min(), d["date"].max(), "1d", eager=True)})
    d = grid.join(d, on="date", how="left")
    prv = trailing_pct(d["rv7"].fill_null(float("nan")).to_numpy())
    piv = trailing_pct(d["iv"].fill_null(float("nan")).to_numpy())
    trig = np.isfinite(prv) & np.isfinite(piv) & (prv <= RV_PCT) & (piv <= IV_PCT)
    dates = d["date"].to_list()
    entry = np.zeros(len(dates), bool)
    last = None
    for i, t in enumerate(trig):
        if not t:
            continue
        e = dates[i] + timedelta(days=1)
        if last is None or (e - last).days >= DECLUSTER_DAYS:
            entry[i] = True
            last = e
    return d.with_columns(pl.Series("pct_rv", prv), pl.Series("pct_iv", piv), pl.Series("trigger", trig),
                          pl.Series("event", entry), (pl.col("date") + timedelta(days=1)).alias("entry_date"))


# ====================================================================== market data container

def fixing(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 8, tzinfo=UTC)


@dataclass
class Market:
    """Everything needed to price and hedge on daily 08:00 fixings (all point-in-time by construction: the value used at
    fixing d is known by d 08:00 for prices / d 08:01 for the smile summary)."""
    S: dict            # date -> index at 08:00 (option trades), fallback Bybit open
    perp: dict         # date -> Bybit BTCUSDT open of the 08:00 bar
    basis: dict        # date -> iv_basis_ann_30 (last known)
    pillars: dict      # date -> (atm[3], rr[3], bf[3])
    listing: dict      # date -> polars frame (instrument, expiry, strike, cp) traded that day
    quotes: dict       # (instrument, date) -> (iv, last_price, last_index, mad_mark, n)
    hs_bucket: dict    # (date, bucket) -> trailing-90-day median half-spread (BTC) of real quotes
    funding: pl.DataFrame  # event_ts, rate, px
    hourly_px: pl.DataFrame | None = None  # Bybit 1h: event_ts, high, low, open (fill checks of perp / spot legs)

    def F(self, d: date, T_years: float) -> float:
        return self.S[d] * math.exp(self.basis.get(d, 0.0) * T_years)

    def model_iv(self, d: date, K: float, T_years: float) -> float:
        p = self.pillars.get(d)
        if p is None:
            return float("nan")
        atm, rr, bf = pillar_interp(np.array([T_years * 365]), p[0][None, :], p[1][None, :], p[2][None, :])
        F = self.F(d, T_years)
        return float(smile_iv(np.log(K / F), T_years, atm[0], rr[0], bf[0]))


def parse_name(name: str):
    from ai_trader.research.options_download import parse_instrument

    return parse_instrument(name)


def tau(d: date, expiry: date) -> float:
    return (expiry - d).days / 365.0


def bucket(delta: float) -> int:
    a = abs(delta)
    return 0 if a < 0.35 else (1 if a <= 0.65 else 2)


@dataclass
class Quote:
    v: float       # USD value per 1-BTC option
    iv: float
    src: str       # real | model | settle
    hs_btc: float  # half-spread (BTC)
    S: float
    F: float
    T: float
    delta: float
    vega: float
    theta: float


def quote(m: Market, d: date, inst: str, K: float, expiry: date, cp: str) -> Quote | None:
    """Tiered quote of one option at the d 08:00 fixing; None if neither tier is available."""
    if d not in m.S:
        return None
    S = m.S[d]
    T = tau(d, expiry)
    if T <= 0:
        v = max((S - K) if cp == "C" else (K - S), 0.0)
        return Quote(v, float("nan"), "settle", 0.0, S, S, 0.0, float(np.sign(v) * (1 if cp == "C" else -1)) if v > 0 else 0.0, 0.0, 0.0)
    F = m.F(d, T)
    q = m.quotes.get((inst, d))
    iv, src, hs = float("nan"), "model", float("nan")
    if q is not None:
        iv_t, lp, li, mad, _n = q
        iv = iv_t / 100.0 if iv_t and iv_t > 0 else implied_vol(lp * li, li * math.exp(m.basis.get(d, 0.0) * T), K, T, cp)
        if np.isfinite(iv) and iv > 0:
            src, hs = "real", float(mad) if mad is not None and np.isfinite(mad) else float("nan")
        else:
            iv = float("nan")
    if not np.isfinite(iv):
        iv = m.model_iv(d, K, T)
        if not np.isfinite(iv):
            return None
    g = b76_greeks(F, K, T, iv, cp)
    if src == "model" or not np.isfinite(hs):
        hs = m.hs_bucket.get((d, bucket(float(g["delta"]))), FEE_FLOOR_BTC)
    return Quote(float(b76(F, K, T, iv, cp)), iv, src, hs, S, F, T, float(g["delta"]), float(g["vega"]), float(g["theta"]))


def listed(m: Market, d: date) -> pl.DataFrame:
    """Instruments traded in the LISTING_DAYS full days before d (point-in-time listing)."""
    parts = [m.listing[x] for x in (d - timedelta(days=i) for i in range(1, LISTING_DAYS + 1)) if x in m.listing]
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts).unique("instrument")


def choose_expiry(lst: pl.DataFrame, d: date, min_dte: int = MIN_DTE) -> date | None:
    if lst.height == 0:
        return None
    ex = sorted(e for e in lst["expiry"].unique().to_list() if (e - d).days >= min_dte)
    return ex[0] if ex else None


def strikes_for(lst: pl.DataFrame, expiry: date) -> np.ndarray:
    return np.unique(lst.filter(pl.col("expiry") == expiry)["strike"].to_numpy())


def inst_name(expiry: date, K: float, cp: str) -> str:
    mon = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")[expiry.month - 1]
    ks = str(int(K)) if float(K).is_integer() else str(K)
    return f"BTC-{expiry.day}{mon}{expiry.strftime('%y')}-{ks}-{cp}"


def pick_by_delta(m: Market, d: date, expiry: date, strikes: np.ndarray, target: float, cp: str) -> float | None:
    """Listed strike whose model-smile Black-76 delta is nearest `target`."""
    if len(strikes) == 0 or d not in m.pillars:
        return None
    T = tau(d, expiry)
    best, bk = 9.0, None
    for K in strikes:
        iv = m.model_iv(d, float(K), T)
        if not np.isfinite(iv):
            continue
        dl = float(b76_greeks(m.F(d, T), K, T, iv, cp)["delta"])
        if abs(dl - target) < best:
            best, bk = abs(dl - target), float(K)
    return bk


# ====================================================================== position simulation

@dataclass
class Leg:
    kind: str          # 'opt' | 'perp' | 'spot'
    qty: float         # + long / - short (options: number of 1-BTC options; perp / spot: BTC)
    inst: str = ""
    K: float = 0.0
    expiry: date | None = None
    cp: str = ""


def leg_value(m: Market, d: date, leg: Leg) -> tuple[float, Quote | None]:
    """USD value per unit of the leg at fixing d (options: tiered quote; perp / spot: Bybit 08:00 open)."""
    if leg.kind == "opt":
        q = quote(m, d, leg.inst, leg.K, leg.expiry, leg.cp)
        return (q.v, q) if q is not None else (float("nan"), None)
    return m.perp.get(d, float("nan")), None


def funding_cost(m: Market, qty: float, t0: datetime, t1: datetime) -> float:
    """USD paid by a perp position of `qty` BTC held over (t0, t1] (long pays positive rates)."""
    if qty == 0 or m.funding.height == 0:
        return 0.0
    f = m.funding.filter((pl.col("event_ts") > t0) & (pl.col("event_ts") <= t1))
    return float(qty * (f["rate"] * f["px"]).sum()) if f.height else 0.0


def run_position(m: Market, d0: date, legs: list[Leg], days: int, hedge: bool = False) -> dict | None:
    """Open `legs` at the d0 fixing, mark daily, close at d0 + days (or settlement if an option expires first).
    hedge=True adds a daily Bybit perp delta hedge (h_k = -net option delta at fixing k). Returns USD / BTC P&L parts
    and the daily mark path (for scale-out rules), or None if a price is missing."""
    path_days = [d0 + timedelta(days=k) for k in range(days + 1)]
    marks, quotes = [], []
    for d in path_days:
        row = []
        qrow = []
        for lg in legs:
            v, q = leg_value(m, d, lg)
            if not np.isfinite(v):
                return None
            row.append(v)
            qrow.append(q)
        marks.append(row)
        quotes.append(qrow)
    V = np.asarray(marks)  # (days + 1, n_legs) USD per unit
    qty = np.array([lg.qty for lg in legs])
    S = np.array([m.S[d] for d in path_days])
    # costs at entry and exit (options: per leg registered cost in BTC; perp / spot: 5 bps)
    cost_usd, cost_btc = 0.0, 0.0
    for j, lg in enumerate(legs):
        for k in (0, days):
            if lg.kind == "opt":
                q = quotes[k][j]
                if k == days and q is not None and q.src == "settle":
                    continue
                c = option_cost_btc(q.hs_btc) * abs(lg.qty)
                cost_btc += c
                cost_usd += c * S[k]
            else:
                c = PERP_COST * abs(lg.qty) * V[k, j]
                cost_usd += c
                cost_btc += c / S[k]
    fund = sum(funding_cost(m, lg.qty, fixing(d0), fixing(path_days[-1])) for lg in legs if lg.kind == "perp")
    gross_usd = float(qty @ (V[-1] - V[0]))
    opt = np.array([lg.kind == "opt" for lg in legs])
    gross_btc = float(np.sum(qty[opt] * (V[-1, opt] / S[-1] - V[0, opt] / S[0]))) + float(np.sum(qty[~opt] * (V[-1, ~opt] - V[0, ~opt]))) / S[-1]
    hedge_usd, hedge_cost, hedge_fund = 0.0, 0.0, 0.0
    if hedge:
        P = np.array([m.perp.get(d, float("nan")) for d in path_days])
        if not np.all(np.isfinite(P)):
            return None
        h = np.zeros(days + 1)
        for k in range(days):
            dl = sum(lg.qty * (quotes[k][j].delta if quotes[k][j] is not None else 1.0) for j, lg in enumerate(legs))
            h[k] = -dl
        hedge_usd = float(np.sum(h[:-1] * np.diff(P)))
        hedge_cost = PERP_COST * float(np.sum(np.abs(np.diff(np.r_[0.0, h])) * P))
        hedge_fund = sum(funding_cost(m, h[k], fixing(path_days[k]), fixing(path_days[k + 1])) for k in range(days) if h[k] != 0)
    net_usd = gross_usd - cost_usd - fund + hedge_usd - hedge_cost - hedge_fund
    net_btc = gross_btc - cost_btc - fund / S[-1] + (hedge_usd - hedge_cost - hedge_fund) / S[-1]
    prem = float(qty[opt] @ V[0, opt]) if opt.any() else 0.0
    return {"S0": float(S[0]), "ST": float(S[-1]), "prem_usd": prem, "gross_usd": gross_usd, "cost_usd": cost_usd, "fund_usd": fund,
            "hedge_usd": hedge_usd, "hedge_cost_usd": hedge_cost, "hedge_fund_usd": hedge_fund, "net_usd": net_usd,
            "net_btc": net_btc, "net_btc_usd": net_btc * S[-1], "marks": V, "S_path": S,
            "src": [q.src if q is not None else lg.kind for q, lg in zip(quotes[0], legs)],
            "src_exit": [q.src if q is not None else lg.kind for q, lg in zip(quotes[-1], legs)],
            "q0": quotes[0], "days": path_days}


# ====================================================================== H-OPX3 predictive model (log-HAR-t x weekday)

def hourly_state(bars: pl.DataFrame) -> pl.DataFrame:
    """Complete hourly grid by bar CLOSE time with ret = ln(close_t / close_{t-1}) (NaN across gaps)."""
    b = bars.select(pl.col("event_ts").dt.replace_time_zone("UTC") if bars["event_ts"].dtype.time_zone is None else pl.col("event_ts"),
                    "close", "high", "low", "open").sort("event_ts")
    b = b.with_columns(pl.col("event_ts").dt.cast_time_unit("us"))
    grid = pl.DataFrame({"event_ts": pl.datetime_range(b["event_ts"].min(), b["event_ts"].max(), "1h", time_zone="UTC", eager=True)}).with_columns(
        pl.col("event_ts").dt.cast_time_unit("us"))
    g = grid.join(b, on="event_ts", how="left")
    lc = np.log(g["close"].to_numpy().astype(float))
    return g.with_columns((pl.col("event_ts") + timedelta(hours=1)).alias("close_ts"), pl.Series("ret", np.r_[np.nan, np.diff(lc)]))


def weekday_dummies(dates: list[date]) -> np.ndarray:
    wd = np.array([d.weekday() for d in dates])
    return np.column_stack([(wd == k).astype(float) for k in range(1, 7)])


def fwd_sum_strict(r: np.ndarray, h: int) -> np.ndarray:
    """sum r[t+1..t+h], NaN if any is missing or the window runs past the end."""
    r = np.asarray(r, float)
    n = len(r)
    c = np.r_[0.0, np.cumsum(np.nan_to_num(r))]
    bad = np.r_[0, np.cumsum(~np.isfinite(r))]
    out = np.full(n, np.nan)
    t = np.arange(n - h)
    s = c[t + h + 1] - c[t + 1]
    nb = bad[t + h + 1] - bad[t + 1]
    out[: n - h] = np.where(nb == 0, s, np.nan)
    return out


def _std_grid(model) -> tuple[np.ndarray, np.ndarray]:
    """CDF of the standardised variable u = R / s on a grid (u = m + e)."""
    u = np.linspace(-40, 40, 16001)
    c = model.cdf(np.zeros((len(u), model.k)), u)
    return u, np.maximum.accumulate(c)


def _quant(u: np.ndarray, c: np.ndarray, p: float) -> float:
    return float(np.interp(p, c, u))


def _sd(u: np.ndarray, c: np.ndarray) -> float:
    pdf = np.diff(c)
    mid = 0.5 * (u[1:] + u[:-1])
    mu = float(np.sum(pdf * mid))
    return float(math.sqrt(max(np.sum(pdf * (mid - mu) ** 2), 1e-12)))


def predictive(bars: pl.DataFrame, H_days: int, min_days: int = 365, log=None) -> tuple[pl.DataFrame, dict]:
    """Daily 08:00 origins: realised R_H (log return over the next H days), predictive scale s, the standardised
    quantiles for Q_GRID (both tails) and the predictive sd. Expanding window, refit on the 1st of each month on origins
    whose target ended before the refit (log-HAR-t + constant Kou of field_study, + 6 origin-weekday dummies).
    Also returns {refit id: 400-point equal-probability grid of the standardised variable u = R / s}."""
    from ai_trader.research.field_study import har_regressors

    st = hourly_state(bars)
    Z0 = har_regressors(st)
    h = 24 * H_days
    R = fwd_sum_strict(st["ret"].to_numpy(), h)
    # har windows need complete-ish history: require the 720-bar window to have >= 90% present
    pres = pl.Series(np.isfinite(st["ret"].to_numpy()).astype(float)).rolling_mean(720, min_samples=720).to_numpy()
    is_o = (st["close_ts"].dt.hour() == 8).to_numpy()
    idx = np.where(is_o & (np.nan_to_num(pres) >= 0.9))[0]
    ct = st["close_ts"].to_list()
    dates = [ct[i].date() for i in idx]
    Z = np.column_stack([Z0[idx], weekday_dummies(dates)])
    Rr = R[idx]
    end_ts = np.array([ct[i] + timedelta(hours=h) for i in idx])
    org = np.array([ct[i] for i in idx])
    n = len(idx)
    out = {k: np.full(n, np.nan) for k in ["s", "sd_u", *[f"qz_hi_{q}" for q in Q_GRID], *[f"qz_lo_{q}" for q in Q_GRID], "qz_95", "nu", "p_jump"]}
    out["refit"] = np.full(n, -1)
    grids: dict[int, np.ndarray] = {}
    model = _safe_loghar_t()(Z.shape[1])
    first = org[0] + timedelta(days=min_days)
    refit = datetime(first.year, first.month, 1, tzinfo=UTC) + timedelta(days=32)
    refit = refit.replace(day=1)
    while refit <= org[-1] + timedelta(days=1):
        nxt = (refit + timedelta(days=32)).replace(day=1)
        tr = np.array([e <= refit for e in end_ts]) & np.isfinite(Rr) & np.isfinite(Z).all(axis=1)
        te = np.array([(o >= refit) and (o < nxt) for o in org]) & np.isfinite(Z).all(axis=1)
        if tr.sum() >= min_days and te.any():
            model.fit(Z[tr], Rr[tr])
            u, c = _std_grid(model)
            par = model.params()
            s = np.exp(0.5 * Z[te] @ np.asarray(par["b"]))
            out["s"][te] = s
            out["sd_u"][te] = _sd(u, c)
            for q in Q_GRID:
                out[f"qz_hi_{q}"][te] = _quant(u, c, q / 100)
                out[f"qz_lo_{q}"][te] = _quant(u, c, 1 - q / 100)
            out["qz_95"][te] = _quant(u, c, 0.95)
            out["nu"][te] = par["nu"]
            out["p_jump"][te] = par["p"]
            rid = len(grids)
            grids[rid] = std_quantile_grid(u, c)
            out["refit"][te] = rid
            if log:
                log(f"  fit H={H_days} refit {refit.date()} n={int(tr.sum())} nu={par['nu']:.2f}")
        refit = nxt
    df = pl.DataFrame({"origin": pl.Series("origin", list(org), dtype=pl.Datetime("us", "UTC")), "date": pl.Series("date", dates, dtype=pl.Date),
                       "R": Rr, **out})
    return df, grids


# ====================================================================== H-OPX3 statistics

def cluster_t(x: np.ndarray, groups: np.ndarray) -> float:
    """Cluster-robust t of the mean (CR1)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(groups)[ok]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    if G < 3:
        return float("nan")
    sums = np.bincount(inv, weights=e)
    var = float(np.sum(sums ** 2)) / n**2 * G / (G - 1)
    return float(x.mean() / math.sqrt(var)) if var > 0 else float("nan")


def tail_events(p: pl.DataFrame, H_days: int) -> pl.DataFrame:
    """For each origin with a forecast: direction-signed forward return in sigma units, the exceedance level reached
    (0..5 = none, q50, q75, q90, q95, q99) and the origin's block id."""
    R = p["R"].to_numpy()
    s = p["s"].to_numpy()
    dates = p["date"].to_list()
    pos = {d: i for i, d in enumerate(dates)}
    fwd_i = np.array([pos.get(d + timedelta(days=H_days), -1) for d in dates])
    has = fwd_i >= 0
    Rf = np.where(has, R[np.where(has, fwd_i, 0)], np.nan)
    sdf = np.where(has, (s * p["sd_u"].to_numpy())[np.where(has, fwd_i, 0)], np.nan)
    lvl = np.zeros(len(R), int)
    dirn = np.zeros(len(R))
    for j, q in enumerate(Q_GRID, start=1):
        hi = R > s * p[f"qz_hi_{q}"].to_numpy()
        lo = R < s * p[f"qz_lo_{q}"].to_numpy()
        lvl = np.where(hi | lo, j, lvl)
        dirn = np.where(hi, 1.0, np.where(lo, -1.0, dirn))
    L = max(7, 2 * H_days)
    blk = np.array([(d - date(2020, 1, 1)).days // L for d in dates])
    return p.select("origin", "date", "R").with_columns(
        pl.Series("level", lvl), pl.Series("dir", dirn), pl.Series("z_fwd", dirn * Rf / sdf),
        pl.Series("ret_fwd_signed", dirn * Rf), pl.Series("block", blk), pl.Series("ok", np.isfinite(s) & np.isfinite(Rf) & np.isfinite(R)))


def tail_table(ev: pl.DataFrame, n_boot: int = 2000, seed: int = 3) -> dict:
    """Conditional mean of the signed forward z for events above each q, cluster and NW t, slope across q."""
    from ai_trader.research.evaluate import newey_west_t

    e = ev.filter(pl.col("ok")).sort("origin")
    rows = {}
    for j, q in enumerate(Q_GRID, start=1):
        x = e.filter(pl.col("level") >= j)
        z = x["z_fwd"].to_numpy()
        m, tnw, n = newey_west_t(z) if len(z) else (float("nan"), float("nan"), 0)
        rows[q] = {"n": len(z), "mean_z": float(np.mean(z)) if len(z) else float("nan"), "t_cluster": cluster_t(z, x["block"].to_numpy()),
                   "t_nw": float(tnw), "mean_ret_pct": float(np.mean(x["ret_fwd_signed"].to_numpy()) * 100) if len(z) else float("nan"),
                   "hit": float(np.mean(z > 0)) if len(z) else float("nan"), "n_up": int((x["dir"] > 0).sum()), "n_down": int((x["dir"] < 0).sum()),
                   "mean_z_up": float(x.filter(pl.col("dir") > 0)["z_fwd"].mean() or float("nan")) if x.height else float("nan"),
                   "mean_z_down": float(x.filter(pl.col("dir") < 0)["z_fwd"].mean() or float("nan")) if x.height else float("nan")}
    means = np.array([rows[q]["mean_z"] for q in Q_GRID])
    xs = np.arange(len(Q_GRID), dtype=float)

    def slope(mm):
        ok = np.isfinite(mm)
        return float(np.polyfit(xs[ok], mm[ok], 1)[0]) if ok.sum() >= 3 else float("nan")

    sl = slope(means)
    # block bootstrap of the slope
    blocks = e["block"].to_numpy()
    ub, inv = np.unique(blocks, return_inverse=True)
    lv = e["level"].to_numpy()
    z = e["z_fwd"].to_numpy()
    S = np.zeros((len(ub), len(Q_GRID)))
    N = np.zeros((len(ub), len(Q_GRID)))
    for j in range(len(Q_GRID)):
        m_ = lv >= j + 1
        S[:, j] = np.bincount(inv[m_], weights=z[m_], minlength=len(ub))
        N[:, j] = np.bincount(inv[m_], minlength=len(ub))
    rng = np.random.default_rng(seed)
    bs = []
    if len(ub) >= 5:
        for _ in range(n_boot):
            c = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mm = (c @ S) / (c @ N)
            bs.append(slope(mm))
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    ci = (float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))) if len(bs) > 50 else (float("nan"), float("nan"))
    return {"by_q": rows, "slope": sl, "slope_ci90": ci, "n_origins": int(e.height)}


def horizon_call(tab: dict, min_n: int = 10) -> str:
    """'continuation' / 'reversal' / 'neither' for one horizon (frozen rule)."""
    qs = [q for q in (90, 95, 99) if tab["by_q"][q]["n"] >= min_n]
    if 90 not in qs:
        return "neither"
    lo, hi = tab["slope_ci90"]
    cont = all(tab["by_q"][q]["mean_z"] > 0 and tab["by_q"][q]["t_cluster"] > 2 for q in qs) and tab["slope"] > 0 and np.isfinite(lo) and lo > 0
    rev = all(tab["by_q"][q]["mean_z"] < 0 and tab["by_q"][q]["t_cluster"] < -2 for q in qs) and tab["slope"] < 0 and np.isfinite(hi) and hi < 0
    return "continuation" if cont else ("reversal" if rev else "neither")


def opx3_verdict(calls: list[str]) -> str:
    if sum(c == "continuation" for c in calls) >= 2:
        return "supports"
    if sum(c == "reversal" for c in calls) >= 2:
        return "contradicts"
    return "inconclusive"


# ====================================================================== bootstrap helpers (ISO-week blocks)

def week_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def boot_diff_cond_uncond(vals: np.ndarray, is_event: np.ndarray, weeks: np.ndarray, B: int = 5000, seed: int = 1) -> dict:
    """mean(event) - mean(all), week-block bootstrap (weeks resampled with replacement)."""
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    vals, ev, wk = vals[ok], np.asarray(is_event, bool)[ok], np.asarray(weeks)[ok]
    if ev.sum() == 0:
        return {"diff": float("nan"), "ci95": (float("nan"), float("nan")), "n_event": 0, "n_all": len(vals)}
    uw, inv = np.unique(wk, return_inverse=True)
    Sa = np.bincount(inv, weights=vals, minlength=len(uw))
    Na = np.bincount(inv, minlength=len(uw)).astype(float)
    Se = np.bincount(inv[ev], weights=vals[ev], minlength=len(uw))
    Ne = np.bincount(inv[ev], minlength=len(uw)).astype(float)
    d = float(vals[ev].mean() - vals.mean())
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        c = np.bincount(rng.integers(0, len(uw), len(uw)), minlength=len(uw)).astype(float)
        ne = c @ Ne
        if ne == 0:
            continue
        out.append((c @ Se) / ne - (c @ Sa) / (c @ Na))
    out = np.asarray(out)
    return {"diff": d, "ci95": (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))), "n_event": int(ev.sum()),
            "n_all": len(vals), "mean_event": float(vals[ev].mean()), "mean_all": float(vals.mean()),
            "median_event": float(np.median(vals[ev])), "median_all": float(np.median(vals))}


def boot_mean(vals: np.ndarray, weeks: np.ndarray, B: int = 5000, seed: int = 2) -> dict:
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    vals, wk = vals[ok], np.asarray(weeks)[ok]
    if len(vals) == 0:
        return {"mean": float("nan"), "ci95": (float("nan"), float("nan")), "n": 0}
    uw, inv = np.unique(wk, return_inverse=True)
    Sw = np.bincount(inv, weights=vals, minlength=len(uw))
    Nw = np.bincount(inv, minlength=len(uw)).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        c = np.bincount(rng.integers(0, len(uw), len(uw)), minlength=len(uw)).astype(float)
        out[b] = (c @ Sw) / (c @ Nw)
    return {"mean": float(vals.mean()), "ci95": (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))), "n": len(vals),
            "n_weeks": len(uw), "median": float(np.median(vals)), "share_pos": float(np.mean(vals > 0))}


def verdict_diff(mean: float, lo: float, n: int, seg: str, min_holdout: int = 8, need_ci_in_holdout: bool = True) -> str:
    if not np.isfinite(mean) or n == 0:
        return "inconclusive"
    if seg == "clean_holdout" and n < min_holdout:
        return "inconclusive"
    if mean > 0 and (np.isfinite(lo) and lo > 0 or (seg == "clean_holdout" and not need_ci_in_holdout)):
        return "supports"
    if mean <= 0:
        return "contradicts"
    return "inconclusive"


# ====================================================================== H-OPX4: scale-out rules

def scale_out_trade(lev: np.ndarray, fr: np.ndarray, lc: np.ndarray, lh: np.ndarray, ll: np.ndarray, i: int, j: int,
                    thr: np.ndarray, c_frac: float) -> dict:
    """One trade on bars i..j (position lev[k] earns fr[k] = ln(c_{k+1}/c_k)). thr[k] = profit threshold (log) at close k.
    At the first three closes t in (i, j] where side x (lc_t - lc_i) > thr[t], a limit for 1/3 of the position rests at
    close_t and fills only if the next bar trades through it (lh[t+1] > lc_t for longs, ll[t+1] < lc_t for shorts); the
    filled third stops earning from fr[t]. Returns gross / cost for own-exit and scaled versions (bar-risk units)."""
    side = float(np.sign(lev[i]))
    f = np.ones(j - i + 1)
    trig, fills = 0, []
    frac = 1.0
    for t in range(i + 1, j + 1):
        if trig >= 3:
            break
        if side * (lc[t] - lc[i]) > thr[t]:
            trig += 1
            through = (lh[t + 1] > lc[t]) if side > 0 else (ll[t + 1] < lc[t])
            if through:
                frac -= 1.0 / 3.0
                fills.append(t)
                f[t - i:] = max(frac, 0.0)
    lk = lev[i:j + 1]
    g = lk * fr[i:j + 1]
    gross0 = float(g.sum())
    cost0 = c_frac * (abs(lev[i]) + float(np.abs(np.diff(lk)).sum()) + abs(lev[j]))
    gross1 = float((f * g).sum())
    # f[k] multiplies the position held over bar k; a fill at t sells 1/3 of the position at close t
    d_lk = np.abs(np.diff(lk)) * f[1:]
    fill_cost = sum(abs(lev[t]) / 3.0 for t in fills)
    cost1 = c_frac * (abs(lev[i]) + float(d_lk.sum()) + fill_cost + f[-1] * abs(lev[j]))
    return {"gross0": gross0, "cost0": cost0, "gross1": gross1, "cost1": cost1, "n_trig": trig, "n_fill": len(fills)}


def pnl_quantile(legs_fn, R_grid: np.ndarray, q: float = 0.95) -> float:
    """q-quantile of P&L(R) over an equal-probability grid of R (the predictive distribution's quantiles)."""
    v = np.asarray(legs_fn(np.asarray(R_grid, float)), float)  # legs_fn is vectorised over R
    return float(np.quantile(v, q))


def std_quantile_grid(u: np.ndarray, c: np.ndarray, n: int = 400) -> np.ndarray:
    p = (np.arange(n) + 0.5) / n
    return np.interp(p, c, u)


# ====================================================================== loaders (lake + downloaded option aggregates)

def _utc(df: pl.DataFrame, col: str = "event_ts") -> pl.DataFrame:
    tz = getattr(df[col].dtype, "time_zone", None)
    e = pl.col(col).dt.replace_time_zone("UTC") if tz is None else pl.col(col).dt.convert_time_zone("UTC")
    return df.with_columns(e.dt.cast_time_unit("us"))


def lake_inputs(as_of: datetime, symbol: str = "BTCUSDT") -> dict:
    """Bybit 1h bars, funding, deribit_svi pillars and DVOL, all read point-in-time at `as_of`."""
    from ai_trader.storage.lake import Lake

    L = Lake()
    bars = _utc(L.bars(symbol, "1h", as_of)).select("event_ts", "open", "high", "low", "close").sort("event_ts")
    fund = _utc(L.read_asof("funding", as_of, symbols=[symbol], filters={"venue": "bybit"})).select("event_ts", "rate").sort("event_ts")
    ent = symbol.replace("USDT", "")
    svi = _utc(L.read_asof("alt_series", as_of, filters={"source": "deribit_svi", "entity": ent})).select("series_id", "event_ts", "value")
    dv = _utc(L.read_asof("alt_series", as_of, filters={"source": "deribit", "series_id": "dvol_close", "entity": ent})).select("event_ts", "value")
    return {"bars": bars, "funding": fund, "svi": svi, "dvol": dv}


def svi_wide(svi: pl.DataFrame) -> pl.DataFrame:
    w = svi.with_columns(pl.col("event_ts").dt.date().alias("date")).pivot(on="series_id", index="date", values="value", aggregate_function="last")
    return w.sort("date")


def iv_series(svi_w: pl.DataFrame, dvol: pl.DataFrame) -> pl.DataFrame:
    """Detector IV per date: iv_atm_30 (08:00 window, known 08:01), DVOL/100 of the bar ending 08:00 where missing."""
    dv = dvol.filter(pl.col("event_ts").dt.hour() == 7).select(pl.col("event_ts").dt.date().alias("date"), (pl.col("value") / 100).alias("dvol"))
    d = svi_w.select("date", pl.col("iv_atm_30").alias("svi")).join(dv, on="date", how="full", coalesce=True).sort("date")
    return d.select("date", pl.coalesce("svi", "dvol").alias("iv"),
                    pl.when(pl.col("svi").is_not_null()).then(pl.lit("svi")).otherwise(pl.lit("dvol")).alias("iv_src"), "svi", "dvol")


def load_options(log=print) -> dict:
    """Listing per date, best real quote per (instrument, date) near 08:00, index at 08:00 per date."""
    from ai_trader.research.options_download import OUT

    fs = sorted(str(f) for f in OUT.glob("btc_*.parquet"))
    lf = pl.scan_parquet(fs)
    lst = (lf.select(pl.col("hour").dt.date().alias("date"), "instrument", "expiry", "strike", "cp")
           .unique(["date", "instrument"]).collect())
    listing = {d: g.drop("date") for (d,), g in lst.group_by("date")}
    q = (lf.filter((pl.col("hour").dt.hour() >= QUOTE_HOURS[0]) & (pl.col("hour").dt.hour() < QUOTE_HOURS[1]))
         .select("hour", "instrument", "expiry", "strike", "cp", "n", "last_price", "last_iv", "last_index", "last_ts", "mad_mark")
         .with_columns(pl.col("hour").dt.date().alias("date"))
         .with_columns((pl.col("last_ts") - (pl.col("date").cast(pl.Datetime("ms")).dt.epoch("ms") + 8 * 3_600_000)).abs().alias("dist"))
         .sort("dist").group_by("instrument", "date").first().collect())
    idx = (lf.filter(pl.col("hour").dt.hour() == 7).select(pl.col("hour").dt.date().alias("date"), "last_ts", "last_index")
           .sort("last_ts").group_by("date").last().collect())
    log(f"options: {len(fs)} months, {lst.height} listing rows, {q.height} quote rows near 08:00")
    return {"listing": listing, "quotes": q, "index": dict(zip(idx["date"].to_list(), idx["last_index"].to_list()))}


def build_market(as_of: datetime, log=print) -> tuple[Market, dict]:
    li = lake_inputs(as_of)
    op = load_options(log)
    bars = li["bars"]
    o8 = bars.filter(pl.col("event_ts").dt.hour() == 8)
    perp = dict(zip(o8["event_ts"].dt.date().to_list(), o8["open"].to_list()))
    S = {d: v for d, v in op["index"].items() if v and v > 0}
    for d, v in perp.items():
        S.setdefault(d, v)
    sw = svi_wide(li["svi"])
    basis, pillars = {}, {}
    last_rr = last_bf = None

    def cols(pre):
        return [f"{pre}_{t}" for t in PILLARS]

    for r in sw.iter_rows(named=True):
        if r.get("iv_basis_ann_30") is not None:
            basis[r["date"]] = float(r["iv_basis_ann_30"])
        a = [r.get(c) for c in cols("iv_atm")]
        rr = [r.get(c) for c in cols("iv_rr25")]
        bf = [r.get(c) for c in cols("iv_bf25")]
        if all(x is not None for x in a + rr + bf):
            pillars[r["date"]] = (np.array(a, float), np.array(rr, float), np.array(bf, float))
            last_rr, last_bf = np.array(rr, float), np.array(bf, float)
    # after the smile summary ends: ATM = DVOL/100 at every tenor, RR / BF = last known (frozen in the diary)
    dv = li["dvol"].filter(pl.col("event_ts").dt.hour() == 7)
    last_svi = max(pillars) if pillars else None
    for d, v in zip(dv["event_ts"].dt.date().to_list(), dv["value"].to_list()):
        if last_svi is not None and d > last_svi and v:
            pillars[d] = (np.full(3, v / 100.0), last_rr, last_bf)
    if basis:  # carry the basis forward over gaps (last known)
        cur = None
        for d in sorted(set(S) | set(basis) | set(pillars)):
            if d in basis:
                cur = basis[d]
            elif cur is not None:
                basis[d] = cur
    px = bars.select("event_ts", pl.col("open").alias("px"))
    fund = li["funding"].join(px, on="event_ts", how="left").with_columns(pl.col("px").fill_null(strategy="forward"))
    q = op["quotes"]
    quotes = {(i, d): (iv, lp, li_, mad, n) for i, d, iv, lp, li_, mad, n in
              zip(q["instrument"].to_list(), q["date"].to_list(), q["last_iv"].to_list(), q["last_price"].to_list(),
                  q["last_index"].to_list(), q["mad_mark"].to_list(), q["n"].to_list())}
    m = Market(S=S, perp=perp, basis=basis, pillars=pillars, listing=op["listing"], quotes=quotes, hs_bucket={}, funding=fund,
               hourly_px=bars)
    m.hs_bucket = half_spread_buckets(m, q)
    return m, {"bars": bars, "svi_w": sw, "dvol": li["dvol"], "quotes": q}


def real_quote_table(m: Market, q: pl.DataFrame) -> pl.DataFrame:
    """Real 08:00 quotes with T, F, IV (trade iv), Black-76 delta, the model IV and value at the same F / T."""
    d = q.filter(pl.col("last_iv") > 0).with_columns(((pl.col("expiry") - pl.col("date")).dt.total_days()).alias("dte"))
    d = d.filter(pl.col("dte") > 0)
    dates = d["date"].to_list()
    S = np.array([m.S.get(x, np.nan) for x in dates])
    b = np.array([m.basis.get(x, 0.0) for x in dates])
    T = d["dte"].to_numpy() / 365.0
    F = S * np.exp(b * T)
    K = d["strike"].to_numpy()
    cp = d["cp"].to_numpy()
    iv = d["last_iv"].to_numpy() / 100.0
    g = b76_greeks(F, K, T, iv, cp)
    nanp = (np.full(3, np.nan),) * 3
    P = [m.pillars.get(x, nanp) for x in dates]
    atm = np.array([p[0] for p in P])
    rr = np.array([p[1] for p in P])
    bf = np.array([p[2] for p in P])
    a_t, r_t, b_t = pillar_interp(T * 365, atm, rr, bf)
    with np.errstate(invalid="ignore", divide="ignore"):
        miv = smile_iv(np.log(K / F), T, a_t, r_t, b_t)
        v_real = b76(F, K, T, iv, cp)
        v_mod = np.where(np.isfinite(miv), b76(F, K, T, np.nan_to_num(miv, nan=0.0), cp), np.nan)
    return d.with_columns(pl.Series("S", S), pl.Series("F", F), pl.Series("T", T), pl.Series("iv", iv), pl.Series("delta", g["delta"]),
                          pl.Series("model_iv", miv), pl.Series("v_real", v_real), pl.Series("v_model", v_mod))


def half_spread_buckets(m: Market, q: pl.DataFrame, window: int = 90) -> dict:
    """(date, |delta| bucket) -> median half-spread (BTC) of real quotes over the `window` days before date."""
    t = real_quote_table(m, q).filter(pl.col("mad_mark").is_not_null() & pl.col("delta").is_finite())
    t = t.with_columns(pl.col("delta").abs().alias("ad"))
    t = t.with_columns(pl.when(pl.col("ad") < 0.35).then(0).when(pl.col("ad") <= 0.65).then(1).otherwise(2).alias("bk"))
    out = {}
    all_dates = sorted(m.S)
    for bk in (0, 1, 2):
        x = t.filter(pl.col("bk") == bk).sort("date")
        dd = np.array([(v - date(2020, 1, 1)).days for v in x["date"].to_list()])
        mm = x["mad_mark"].to_numpy()
        for d in all_dates:
            di = (d - date(2020, 1, 1)).days
            lo, hi = np.searchsorted(dd, di - window), np.searchsorted(dd, di)
            if hi - lo >= 20:
                out[(d, bk)] = float(np.median(mm[lo:hi]))
    return out


def _safe_loghar_t():
    """field_study.LogHarT with p_up clipped to [1e-9, 1 - 1e-9] and the jump rates to [1e-6, 1e6]: the original raises a
    math domain error when expit saturates at 1.0 or exp underflows to 0 on some symbols; the clip changes nothing
    whenever the original runs."""
    from ai_trader.research.field_study import LogHarT

    class SafeLogHarT(LogHarT):
        def unpack(self, th):
            b, nu, m, p, pu, eu, ed = super().unpack(th)
            return (b, nu, m, p, float(np.clip(pu, 1e-9, 1 - 1e-9)), float(np.clip(eu, 1e-6, 1e6)),
                    float(np.clip(ed, 1e-6, 1e6)))

    return SafeLogHarT
