"""Weekend options study: H-OPT1 / H-OPT2 / H-OPT3 (research/hypotheses.yaml, registered 2026-11-07).

Instrument: Deribit daily-expiry ATM straddle, sold at the 08:00 UTC sampling window, held to expiry 3 days later.
  * WEEKEND trade: sell Friday 08:00 UTC, expiry Monday 08:00 UTC.   WEEKDAY control: Monday 08:00 -> Thursday 08:00.

Design (fixed before any P&L was computed; every judgement call is listed here and in the report):
  * Raw trades: BACKTEST_TREND_FOLLOWING/data/{btc,eth}_options/raw_trades/<yyyy>/<date>.csv.gz (read-only). The
    sampling window is 04:00-08:00 UTC. Only inverse instruments named "<U>-<DMMMYY>-<K>-<C|P>" are used.
  * Forward at entry F0 = index_price of the last trade in the window (any instrument); r = 0, forward = index.
  * Strike grid: step = smallest positive gap between traded strikes of that expiry within +-10% of F0 (fallback:
    smallest gap overall). ATM strike K = round(F0 / step) * step (always a listed strike on Deribit's grid).
  * Entry IV: median `iv` of the expiry's trades with |strike - K| <= 2 * step if there are >= MIN_ATM_TRADES of them;
    otherwise the amount-weighted mean iv of the expiry's trades with |ln(strike/F0)| <= NEAR_MONEY; otherwise skip.
  * Premium = Black-76 call + put at (F0, K, IV, T = 3/365 exactly, 08:00 -> 08:00).
  * Half-spread = median |price - mark_price| * index_price over the same trades that set the IV (USD per coin),
    charged on both legs at entry; settlement has no spread. Fee per leg = min(0.03% * F0, 12.5% * leg price),
    entry only (registry: no settlement fee is charged).
  * Hedge: the Bybit perp (lake 1h bars; perp ~ Deribit index, an approximation: basis of a few bps is ignored).
    Hedge times t_k = entry + k hours, k = 0..71; S(t_0) = open of the 08:00 bar, S(t_k) = close of the bar opening
    at t_{k-1}. The short straddle is hedged with h_k = +(2 N(d1) - 1) perp units, Black-76 delta at the ENTRY IV,
    F = S(t_k), remaining time (72 - k) hours. Hedge P&L = sum h_k (S_{k+1} - S_k); cost 5 bps * S_k * |h_k - h_{k-1}|
    including the opening trade (from 0) and the unwind at settlement. A window with a missing bar is skipped.
  * Settlement S_T = open of the bar opening at expiry 08:00 UTC (Deribit settles on a 30-min index TWAP; approx).
  * P&L (short 1 straddle on 1 coin, USD per coin): premium - |S_T - K| + hedge P&L - spread - fees - hedge costs.
    Reported in USD, in units of mid premium and as % of underlying notional F0 (the primary, scale-free unit).
    Inverse (coin-margined) settlement is ignored: P&L is computed as for a linear USD option.
  * Segments by entry vs research.knowledge_cutoff; a trade whose expiry is on/after the cutoff but whose entry is
    before it is dropped from both segments. H-OPT2 pairs the Friday trade with the Monday control of the same ISO week.
  * Statistics: Newey-West t on the weekly series (n >= 10), plain t also reported. Pooled H-OPT2 averages BTC and
    ETH per week first (clustering by time).
  * H-OPT3: per underlying, notional fraction w = 1% / q99(in-sample loss in % of notional) (numpy linear quantile);
    the same w is applied to the holdout. Annualised return = 52 * w * mean P&L%; worst loss = w * max loss%.

Run:  uv run python -m ai_trader.research.options_study insample   (records in-sample verdicts first)
      uv run python -m ai_trader.research.options_study holdout
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.time import now
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t
from ai_trader.research.verdicts import MIN_HOLDOUT
from ai_trader.storage.lake import Lake

DATA = Path(r"C:\Users\Asus\Documents\BACKTEST_TREND_FOLLOWING\data")
UNDERLYINGS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
HOLD_DAYS = 3
T_YEARS = HOLD_DAYS / 365.0
MIN_ATM_TRADES = 3
NEAR_MONEY = 0.05  # |ln K/F| for the fallback IV
FEE_RATE = 0.0003
FEE_CAP = 0.125
HEDGE_COST = 0.0005
SEGS = ("llm_in_sample", "clean_holdout")
DATA_DESC = ("Deribit raw option trades 04:00-08:00 UTC (BACKTEST_TREND_FOLLOWING), daily-expiry ATM straddles; "
             "Bybit perp 1h bars (lake) for hedge and settlement")


# ------------------------------------------------------------------ Black-76

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    v = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return d1, d1 - v


def black76(F: float, K: float, T: float, sigma: float, cp: str, r: float = 0.0) -> float:
    """Black-76 price of a European call ('C') or put ('P') on a forward F; discount exp(-rT)."""
    df = math.exp(-r * T)
    if T <= 0 or sigma <= 0:
        return df * max(F - K, 0.0) if cp == "C" else df * max(K - F, 0.0)
    d1, d2 = _d1d2(F, K, T, sigma)
    if cp == "C":
        return df * (F * _ncdf(d1) - K * _ncdf(d2))
    return df * (K * _ncdf(-d2) - F * _ncdf(-d1))


def black76_delta(F: float, K: float, T: float, sigma: float, cp: str, r: float = 0.0) -> float:
    """Forward delta dV/dF."""
    df = math.exp(-r * T)
    if T <= 0 or sigma <= 0:
        itm = F > K if cp == "C" else F < K
        return df * (1.0 if itm else 0.0) * (1 if cp == "C" else -1)
    d1, _ = _d1d2(F, K, T, sigma)
    return df * _ncdf(d1) if cp == "C" else -df * _ncdf(-d1)


def straddle(F: float, K: float, T: float, sigma: float) -> tuple[float, float, float]:
    """(call, put, straddle delta)."""
    c, p = black76(F, K, T, sigma, "C"), black76(F, K, T, sigma, "P")
    return c, p, black76_delta(F, K, T, sigma, "C") + black76_delta(F, K, T, sigma, "P")


def option_fee(F: float, price: float) -> float:
    """Deribit taker fee per option leg: 0.03% of the underlying, capped at 12.5% of the option price."""
    return min(FEE_RATE * F, FEE_CAP * price)


# ------------------------------------------------------------------ raw trades

def _expiry_code(d: date) -> str:
    return f"{d.day}{d.strftime('%b%y').upper()}"


def load_trades(u: str, d: date) -> pl.DataFrame | None:
    f = DATA / f"{u.lower()}_options" / "raw_trades" / str(d.year) / f"{d.isoformat()}.csv.gz"
    if not f.is_file():
        return None
    df = pl.read_csv(f, schema_overrides={"block_trade_id": pl.Utf8, "combo_id": pl.Utf8, "trade_id": pl.Utf8,
                                          "instrument_name": pl.Utf8})
    return df.filter(pl.col("instrument_name").str.starts_with(f"{u}-"))


def entry_quote(trades: pl.DataFrame, expiry: date) -> dict:
    """Strike, forward, IV and half-spread for the ATM straddle of `expiry` from one day's window of trades.
    Returns {'skip': reason} if no usable quote."""
    if trades is None or trades.height == 0:
        return {"skip": "no_file"}
    F0 = float(trades.sort("timestamp")["index_price"][-1])
    parts = trades["instrument_name"].str.split("-")
    t = trades.with_columns(parts.list.get(1).alias("exp"), parts.list.get(2).cast(pl.Float64, strict=False).alias("strike"))
    t = t.filter((pl.col("exp") == _expiry_code(expiry)) & pl.col("strike").is_not_null() & (pl.col("iv") > 0))
    if t.height == 0:
        return {"skip": "no_trades_in_expiry", "F0": F0}
    ks = np.unique(t["strike"].to_numpy())
    near = ks[np.abs(ks / F0 - 1) <= 0.10]
    gaps = np.diff(near) if len(near) > 1 else np.diff(ks)
    gaps = gaps[gaps > 0]
    step = float(gaps.min()) if len(gaps) else float("nan")
    K = float(round(F0 / step) * step) if np.isfinite(step) else float(ks[np.argmin(np.abs(ks - F0))])
    if np.isfinite(step):
        atm = t.filter((pl.col("strike") - K).abs() <= 2 * step + 1e-9)
    else:
        atm = t.filter(pl.col("strike") == K)
    if atm.height >= MIN_ATM_TRADES:
        iv, used, method = float(atm["iv"].median()) / 100, atm, "atm_median"
    else:
        nm = t.filter((pl.col("strike") / F0).log().abs() <= NEAR_MONEY)
        if nm.height == 0:
            return {"skip": "no_near_money_trades", "F0": F0}
        w = nm["amount"].to_numpy()
        iv, used, method = float(np.average(nm["iv"].to_numpy(), weights=w)) / 100, nm, "near_money_weighted"
    hs = float(((used["price"] - used["mark_price"]).abs() * used["index_price"]).median())
    return {"F0": F0, "K": K, "step": step, "iv": iv, "half_spread": hs, "n_iv_trades": used.height,
            "n_expiry_trades": t.height, "iv_method": method}


# ------------------------------------------------------------------ simulation

def simulate(q: dict, path: np.ndarray) -> dict:
    """P&L of a short straddle given the entry quote and the hedge price path S(t_0..t_72) (hourly)."""
    F0, K, iv = q["F0"], q["K"], q["iv"]
    c, p, _ = straddle(F0, K, T_YEARS, iv)
    prem = c + p
    n = len(path) - 1
    h = np.empty(n)
    for k in range(n):
        tau = (n - k) / 24 / 365
        h[k] = 2 * _ncdf(_d1d2(path[k], K, tau, iv)[0]) - 1
    hedge_pnl = float(np.sum(h * np.diff(path)))
    turnover = np.abs(np.diff(np.r_[0.0, h])) * path[:-1]
    hedge_cost = HEDGE_COST * (float(turnover.sum()) + abs(h[-1]) * path[-1])
    ST = float(path[-1])
    payoff = abs(ST - K)
    spread = 2 * q["half_spread"]
    fees = option_fee(F0, c) + option_fee(F0, p)
    net = prem - payoff + hedge_pnl - spread - fees - hedge_cost
    unhedged = prem - payoff - spread - fees
    rv = float(np.sum(np.diff(np.log(path)) ** 2))
    return {"premium": prem, "call": c, "put": p, "payoff": payoff, "hedge_pnl": hedge_pnl, "hedge_cost": hedge_cost,
            "spread_cost": spread, "fees": fees, "net": net, "unhedged": unhedged, "S0": float(path[0]), "ST": ST,
            "ret": math.log(ST / path[0]), "max_up": float(np.log(path.max() / path[0])),
            "max_dn": float(np.log(path.min() / path[0])), "rv": rv, "iv_var": iv * iv * T_YEARS,
            "net_pct": net / F0 * 100, "unhedged_pct": unhedged / F0 * 100, "net_prem": net / prem,
            "unhedged_prem": unhedged / prem, "gross_pct": (prem - payoff + hedge_pnl) / F0 * 100,
            "cost_pct": (spread + fees + hedge_cost) / F0 * 100}


def hourly_path(bars: pl.DataFrame, t0: datetime) -> np.ndarray | None:
    hrs = 24 * HOLD_DAYS
    b = bars.filter((pl.col("event_ts") >= t0) & (pl.col("event_ts") <= t0 + timedelta(hours=hrs))).sort("event_ts")
    if b.height != hrs + 1:
        return None
    exp_ts = [t0 + timedelta(hours=i) for i in range(hrs + 1)]
    if [x.replace(tzinfo=UTC) for x in b["event_ts"].to_list()] != exp_ts:
        return None
    o, c = b["open"].to_numpy(), b["close"].to_numpy()
    return np.r_[o[0], c[:hrs - 1], o[hrs]]  # S(t_72) = open of the expiry 08:00 bar


def run(as_of: datetime | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """All trades (both kinds, both underlyings) and the skip log. P&L columns are computed for every date; the
    caller decides which segment it looks at."""
    as_of = as_of or now()
    lake = Lake()
    rows, skips = [], []
    for u, sym in UNDERLYINGS.items():
        bars = lake.bars(sym, "1h", as_of).with_columns(pl.col("event_ts").dt.replace_time_zone("UTC"))
        days = sorted(p.name[:10] for p in (DATA / f"{u.lower()}_options" / "raw_trades").glob("*/*.csv.gz"))
        for ds in days:
            d = date.fromisoformat(ds)
            if d.weekday() not in (0, 4):
                continue
            kind = "weekend" if d.weekday() == 4 else "weekday"
            expiry = d + timedelta(days=HOLD_DAYS)
            q = entry_quote(load_trades(u, d), expiry)
            base = {"u": u, "kind": kind, "entry": d, "expiry": expiry, "year": d.year,
                    "week": d - timedelta(days=d.weekday())}
            if "skip" in q:
                skips.append({**base, "reason": q["skip"]})
                continue
            t0 = datetime(d.year, d.month, d.day, 8, tzinfo=UTC)
            path = hourly_path(bars, t0)
            if path is None:
                skips.append({**base, "reason": "missing_bars"})
                continue
            rows.append({**base, **{k: q[k] for k in ("F0", "K", "step", "iv", "half_spread", "n_iv_trades",
                                                         "n_expiry_trades", "iv_method")}, **simulate(q, path)})
    tr = pl.DataFrame(rows)
    k = knowledge_cutoff().date()
    tr = tr.with_columns(
        pl.when(pl.col("expiry") < k).then(pl.lit("llm_in_sample"))
        .when(pl.col("entry") >= k).then(pl.lit("clean_holdout")).otherwise(pl.lit("straddles_cutoff")).alias("segment"))
    return tr, pl.DataFrame(skips)


# ------------------------------------------------------------------ statistics

def _t(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m, t_nw, _ = newey_west_t(x)
    t_plain = float(x.mean() / (x.std(ddof=1) / np.sqrt(n))) if n >= 3 else float("nan")
    return {"n": n, "mean": float(m) if n else float("nan"), "t_nw": float(t_nw), "t": t_plain,
            "median": float(np.median(x)) if n else float("nan")}


def describe(d: pl.DataFrame) -> dict:
    if d.height == 0:
        return {"n": 0}
    out = {"net_pct": _t(d["net_pct"].to_numpy()), "net_prem": _t(d["net_prem"].to_numpy()),
           "unhedged_pct": _t(d["unhedged_pct"].to_numpy()), "unhedged_prem": _t(d["unhedged_prem"].to_numpy())}
    out["n"] = d.height
    out["mean_net_usd"] = float(d["net"].mean())
    out["loss_share"] = float((d["net"] < 0).mean())
    out["loss_share_unhedged"] = float((d["unhedged"] < 0).mean())
    out["mean_premium_pct"] = float((d["premium"] / d["F0"]).mean() * 100)
    out["mean_cost_pct"] = float(d["cost_pct"].mean())
    out["mean_gross_pct"] = float(d["gross_pct"].mean())
    out["mean_iv"] = float(d["iv"].mean())
    vr = (d["rv"] / d["iv_var"]).to_numpy()
    out["rv_iv_ratio_median"] = float(np.median(vr))
    out["rv_iv_ratio_mean"] = float(np.mean(vr))
    out["rv_iv_ratio_pooled"] = float(d["rv"].sum() / d["iv_var"].sum())
    out["share_atm_median_iv"] = float((d["iv_method"] == "atm_median").mean())
    out["worst_net_pct"] = float(d["net_pct"].min())
    return out


def h_opt1(tr: pl.DataFrame, seg: str) -> tuple[str, dict]:
    w = tr.filter((pl.col("kind") == "weekend") & (pl.col("segment") == seg))
    m = {u: describe(w.filter(pl.col("u") == u)) for u in UNDERLYINGS}
    m["by_year"] = {u: {str(y): describe(g) for (y,), g in w.filter(pl.col("u") == u).group_by("year", maintain_order=True)}
                    for u in UNDERLYINGS}
    means = [m[u]["net_pct"]["mean"] if m[u]["n"] else float("nan") for u in UNDERLYINGS]
    ts = [m[u]["net_pct"]["t_nw"] if m[u]["n"] else float("nan") for u in UNDERLYINGS]
    n = min(m[u]["n"] for u in UNDERLYINGS)
    sign_ok = all(np.isfinite(means)) and all(x > 0 for x in means)
    ok = sign_ok and all(np.isfinite(ts)) and all(t > 2 for t in ts)
    return _verdict(seg, ok, sign_ok, n), m


def _verdict(seg: str, ok_in: bool, sign_ok: bool, n: int) -> str:
    if seg == "clean_holdout":
        if n < MIN_HOLDOUT:
            return "inconclusive"
        return "supports" if sign_ok else "contradicts"
    return "supports" if ok_in else ("contradicts" if not sign_ok else "inconclusive")


def pairs(tr: pl.DataFrame, seg: str) -> pl.DataFrame:
    s = tr.filter(pl.col("segment") == seg)
    fri = s.filter(pl.col("kind") == "weekend").select("u", "week", pl.col("net_prem").alias("fri"), pl.col("entry").alias("fri_date"))
    mon = s.filter(pl.col("kind") == "weekday").select("u", "week", pl.col("net_prem").alias("mon"))
    return fri.join(mon, on=["u", "week"], how="inner").with_columns((pl.col("fri") - pl.col("mon")).alias("diff")).sort("week")


def h_opt2(tr: pl.DataFrame, seg: str) -> tuple[str, dict]:
    p = pairs(tr, seg)
    m = {u: {**_t(p.filter(pl.col("u") == u)["diff"].to_numpy()),
             "mean_fri": float(p.filter(pl.col("u") == u)["fri"].mean() or float("nan")),
             "mean_mon": float(p.filter(pl.col("u") == u)["mon"].mean() or float("nan"))} for u in UNDERLYINGS}
    pooled = p.group_by("week").agg(pl.col("diff").mean()).sort("week")
    m["pooled"] = _t(pooled["diff"].to_numpy())
    m["by_year"] = {u: {str(y): _t(g["diff"].to_numpy()) for (y,), g in
                        p.filter(pl.col("u") == u).with_columns(pl.col("week").dt.year().alias("y")).group_by("y", maintain_order=True)}
                    for u in UNDERLYINGS}
    pm, pt, n = m["pooled"]["mean"], m["pooled"]["t_nw"], m["pooled"]["n"]
    sign_ok = np.isfinite(pm) and pm > 0
    ok = sign_ok and np.isfinite(pt) and pt > 2
    m["unit"] = "net P&L in units of entry mid premium, Fri-Mon minus Mon-Thu of the same ISO week"
    return _verdict(seg, ok, sign_ok, n), m


def sizing(tr: pl.DataFrame) -> dict[str, dict]:
    """In-sample sizing rule (H-OPT3): notional fraction so that the 99th-percentile loss = 1% of capital."""
    w = tr.filter((pl.col("kind") == "weekend") & (pl.col("segment") == "llm_in_sample"))
    out = {}
    for u in UNDERLYINGS:
        loss = -w.filter(pl.col("u") == u)["net_pct"].to_numpy()
        q99 = float(np.quantile(loss, 0.99))
        out[u] = {"q99_loss_pct_notional": q99, "notional_fraction": 1.0 / q99 if q99 > 0 else float("nan"),
                  "n_insample": len(loss)}
    return out


def h_opt3(tr: pl.DataFrame, seg: str) -> tuple[str, dict]:
    sz = sizing(tr)
    w = tr.filter((pl.col("kind") == "weekend") & (pl.col("segment") == seg))
    m, oks, worst_oks, n = {}, [], [], 10**9
    for u in UNDERLYINGS:
        x = w.filter(pl.col("u") == u)["net_pct"].to_numpy()
        f = sz[u]["notional_fraction"]
        r = x * f  # % of capital per weekend
        ann = 52 * float(r.mean()) if len(r) else float("nan")
        worst = float(-r.min()) if len(r) else float("nan")
        m[u] = {**sz[u], "n": len(r), "ann_return_pct": ann, "worst_loss_pct_capital": worst,
                "mean_weekend_pct_capital": float(r.mean()) if len(r) else float("nan"),
                "ann_return_unhedged_pct_at_same_size": 52 * float(w.filter(pl.col("u") == u)["unhedged_pct"].mean() * f) if len(r) else float("nan")}
        oks.append(ann > 5 and worst < 3)
        worst_oks.append(worst < 3)
        n = min(n, len(r))
    if seg == "clean_holdout":
        v = "inconclusive" if n < MIN_HOLDOUT else ("supports" if all(worst_oks) else "contradicts")
    else:
        v = "supports" if all(oks) else "contradicts"
    return v, m


def tail_table(tr: pl.DataFrame, seg: str, k: int = 10) -> pl.DataFrame:
    return (tr.filter((pl.col("kind") == "weekend") & (pl.col("segment") == seg)).sort("net_pct").head(k)
            .select("u", "entry", "K", "S0", "ST", (pl.col("ret") * 100).alias("ret_pct"),
                    (pl.col("max_up") * 100).alias("max_up_pct"), (pl.col("max_dn") * 100).alias("max_dn_pct"),
                    (pl.col("iv") * 100).alias("iv_pct"), (pl.col("rv") / pl.col("iv_var")).alias("rv_iv"),
                    "net_pct", "net_prem", "unhedged_pct"))


# ------------------------------------------------------------------ CLI

HYPS = {"H-OPT1": h_opt1, "H-OPT2": h_opt2, "H-OPT3": h_opt3}
NOTES = {
    "H-OPT1": "short Fri 08:00 -> Mon 08:00 ATM daily straddle, hourly perp delta hedge at entry IV; primary unit net P&L % of notional, NW t; both underlyings required",
    "H-OPT2": "paired by ISO week Fri-Mon minus Mon-Thu net P&L in premium units; primary = BTC/ETH averaged per week, NW t",
    "H-OPT3": "w = 1%/q99(in-sample loss % notional) per underlying, same w on holdout; 52 weekends/yr; both underlyings required",
}


def main(mode: str, record: bool = True) -> dict:
    from ai_trader.research import hypotheses

    tr, sk = run()
    seg = "llm_in_sample" if mode == "insample" else "clean_holdout"
    out = {"segment": seg, "n_trades": tr.filter(pl.col("segment") == seg).group_by("u", "kind").len().sort("u", "kind").rows(),
           "skips": sk.filter(pl.col("entry") >= date(2025, 1, 1)).group_by("u", "kind", "reason").len().sort("u", "kind", "reason").rows()
           if sk.height else [], "dropped_straddling_cutoff": tr.filter(pl.col("segment") == "straddles_cutoff").height}
    for hid, fn in HYPS.items():
        v, m = fn(tr, seg)
        out[hid] = {"verdict": v, "metrics": m}
        if record:
            hypotheses.record(hid, seg, v, m, data=DATA_DESC, note=NOTES[hid])
    out["weekday_control"] = {u: describe(tr.filter((pl.col("segment") == seg) & (pl.col("kind") == "weekday") & (pl.col("u") == u)))
                              for u in UNDERLYINGS}
    out["tail"] = [{k: (v.isoformat() if isinstance(v, date) else v) for k, v in r.items()}
                   for r in tail_table(tr, seg).to_dicts()]
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "insample"
    if mode not in ("insample", "holdout", "dry"):
        raise SystemExit("usage: options_study insample|holdout|dry")
    res = main("insample" if mode == "dry" else mode, record=mode != "dry")
    print(json.dumps(res, indent=1, default=str))
