"""H-X1: resting vs close-confirmed structural stops on 1h trend entries, simulated on 5m bars.

Entries come from a 1h strategy signal computed on the FULL history (protocol §3b). A trade opens at
the 1h close where sign(signal) changes from 0/opposite to nonzero; it is closed by the strategy
at the first later 1h close where sign(signal) != direction (flat or opposite; an opposite sign also
opens the next trade at that same close, so trades never overlap).

Structural stop (all causal):
  * pivot low at bar i: low_i < min(low_{i-k..i-1}) and low_i <= min(low_{i+1..i+k}); it needs the k
    bars after it, so it is *confirmed* (usable) only at the close of bar i+k (k = 3). Pivot highs
    mirror this.
  * at entry bar t the stop references the most recent pivot confirmed at or before t whose level is
    on the protective side of the entry price (below it for a long, above for a short).
  * stop = pivot_low - buf (long) / pivot_high + buf (short), buf = 0.3 * sigma * close.
  * sigma = sigma_next at the entry bar: the EWMA forecast of the next 1h log-return std, known at
    the close of t. In price units one-bar sigma is sigma * close (log return ~ fractional move).
  * skip the trade if no valid pivot exists or |entry - stop| is outside [0.5, 4] * sigma * close
    (config/risk.yaml placement constants).
  * no target: exits only by stop or by strategy exit.

Exit policies for the SAME entries:
  A  resting: first 5m bar whose low (long) / high (short) touches the stop; fill at the stop, or at
     that 5m bar's open if it opened beyond the stop (worse fill).
  B  close-confirmed: exit at a 1h close beyond the stop (fill at that close); plus a resting
     catastrophe stop at 2x the original distance from entry (5m touch, same fill rules).
Within an hour the intrabar stop is checked first (it happens before the close), then the 1h close
(close-confirmed stop for B, then the strategy exit). If an hour has no 5m bars inside the 5m
coverage window the 1h bar itself is used as the single intrabar bar (counted as a fallback); hours
after the 5m coverage end make the trade unresolved and it is dropped.

Costs (config/risk.yaml): market entry and market/stop exit both pay taker + slippage. R follows
at-risk sizing: risk = |entry - stop| + entry * round_trip_cost, pnl = dir*(exit-entry) -
side_cost*(entry + exit), R = pnl / risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t

PIVOT_K = 3
BUFFER_SIGMA = 0.3
MIN_STOP_SIGMA = 0.5
MAX_STOP_SIGMA = 4.0
CATASTROPHE_MULT = 2.0

# Intrabar provider: hour index j -> (open, high, low) arrays of sub-bars inside 1h bar j, in time
# order; None if j is beyond data coverage (trade unresolved).
Intrabar = Callable[[int], "tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None"]


def side_cost() -> float:
    c = load().risk["costs"]
    return (float(c["taker_bps"]) + float(c["slippage_bps"])) / 1e4


# ---------------------------------------------------------------- causal building blocks
def pivots(high: np.ndarray, low: np.ndarray, k: int = PIVOT_K) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Confirmed pivots as (confirm_bar, level), sorted by confirm_bar. confirm_bar = i + k."""
    n = len(low)
    lows, highs = [], []
    for i in range(k, n - k):
        lo, hi = low[i], high[i]
        if lo < low[i - k:i].min() and lo <= low[i + 1:i + k + 1].min():
            lows.append((i + k, float(lo)))
        if hi > high[i - k:i].max() and hi >= high[i + 1:i + k + 1].max():
            highs.append((i + k, float(hi)))
    return lows, highs


def entries(signal: np.ndarray) -> list[tuple[int, int, int | None]]:
    """(entry_bar, direction, strategy_exit_bar or None)."""
    s = np.sign(np.nan_to_num(signal))
    out = []
    for t in range(len(s)):
        if s[t] != 0 and (t == 0 or s[t - 1] != s[t]):
            d = int(s[t])
            nxt = np.nonzero(s[t + 1:] != d)[0]
            out.append((t, d, int(t + 1 + nxt[0]) if len(nxt) else None))
    return out


def structural_stop(t: int, d: int, close: float, sigma: float, piv_lo: list, piv_hi: list) -> float | None:
    """Stop for an entry at the close of bar t, or None if no valid pivot / distance out of band."""
    if not (np.isfinite(sigma) and sigma > 0):
        return None
    sp = sigma * close
    pv = piv_lo if d == 1 else piv_hi
    level = None
    for cb, lv in reversed(pv):  # most recent confirmed at or before t, on the protective side
        if cb > t:
            continue
        if (d == 1 and lv < close) or (d == -1 and lv > close):
            level = lv
            break
    if level is None:
        return None
    stop = level - BUFFER_SIGMA * sp if d == 1 else level + BUFFER_SIGMA * sp
    dist = abs(close - stop)
    if dist < MIN_STOP_SIGMA * sp or dist > MAX_STOP_SIGMA * sp:
        return None
    return stop


def _touch(d: int, level: float, o: np.ndarray, h: np.ndarray, lo: np.ndarray) -> float | None:
    """Fill of a resting stop at `level` on the first touching sub-bar (worse of level and open)."""
    hit = np.nonzero(lo <= level)[0] if d == 1 else np.nonzero(h >= level)[0]
    if not len(hit):
        return None
    op = o[hit[0]]
    return float(min(level, op)) if d == 1 else float(max(level, op))


@dataclass
class Exit:
    bar: int
    price: float
    reason: str          # stop | close_stop | catastrophe | signal
    reverted: bool       # resting stop touched but the hour closed back inside the stop
    fallback_hours: int  # hours simulated on the 1h bar because 5m bars were missing


def simulate(policy: str, t: int, d: int, stop: float, exit_bar: int | None, close: np.ndarray, intrabar: Intrabar, n_bars: int) -> Exit | None:
    entry = close[t]
    cat = entry - d * CATASTROPHE_MULT * abs(entry - stop)
    last = exit_bar if exit_bar is not None else n_bars - 1
    fb = 0
    for j in range(t + 1, last + 1):
        ib = intrabar(j)
        if ib is None:
            return None
        o, h, lo, used_fallback = ib
        fb += int(used_fallback)
        c = close[j]
        if policy == "A":
            f = _touch(d, stop, o, h, lo)
            if f is not None:
                return Exit(j, f, "stop", bool(d * (c - stop) > 0), fb)
        elif policy == "B":
            f = _touch(d, cat, o, h, lo)
            if f is not None:
                return Exit(j, f, "catastrophe", False, fb)
            if d * (c - stop) <= 0:
                return Exit(j, float(c), "close_stop", False, fb)
        else:
            raise ValueError(policy)
        if j == exit_bar:
            return Exit(j, float(c), "signal", False, fb)
    return None  # still open at the end of the data


def r_multiple(d: int, entry: float, stop: float, fill: float, cost: float) -> float:
    risk = abs(entry - stop) + entry * 2 * cost
    pnl = d * (fill - entry) - cost * (entry + fill)
    return pnl / risk


# ---------------------------------------------------------------- intrabar providers
def intrabar_5m(hour_ts: np.ndarray, hour_bar: tuple[np.ndarray, np.ndarray, np.ndarray], m5: pl.DataFrame) -> Intrabar:
    """5m bars inside each hour; the 1h bar itself where 5m is missing inside coverage."""
    ts = m5["event_ts"].dt.epoch("ms").to_numpy()
    o, h, lo = (m5[c].to_numpy() for c in ("open", "high", "low"))
    end = ts[-1] + 5 * 60_000
    ho, hh, hl = hour_bar

    def f(j: int):
        a = hour_ts[j]
        if a + 3_600_000 > end or a < ts[0]:
            return None
        i0, i1 = np.searchsorted(ts, a), np.searchsorted(ts, a + 3_600_000)
        if i1 - i0 == 0:
            return ho[j:j + 1], hh[j:j + 1], hl[j:j + 1], True
        return o[i0:i1], h[i0:i1], lo[i0:i1], False

    return f


def intrabar_1h(hour_bar: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Intrabar:
    """Coarse version: the 1h bar is the only sub-bar (stop touched by the 1h high/low, worst-case)."""
    ho, hh, hl = hour_bar
    return lambda j: (ho[j:j + 1], hh[j:j + 1], hl[j:j + 1], False)


# ---------------------------------------------------------------- study
def trades(frame: pl.DataFrame, signal: np.ndarray, intrabar: Intrabar) -> pl.DataFrame:
    """One row per entry with a valid stop that both policies resolve within the data."""
    c = frame["close"].to_numpy()
    h, lo = frame["high"].to_numpy(), frame["low"].to_numpy()
    sig = frame["sigma_next"].to_numpy()
    ts = frame["event_ts"].to_list()
    piv_lo, piv_hi = pivots(h, lo)
    cost = side_cost()
    rows, skipped = [], {"no_stop": 0, "unresolved": 0}
    for t, d, e in entries(signal):
        stop = structural_stop(t, d, c[t], sig[t], piv_lo, piv_hi)
        if stop is None:
            skipped["no_stop"] += 1
            continue
        a = simulate("A", t, d, stop, e, c, intrabar, len(c))
        b = simulate("B", t, d, stop, e, c, intrabar, len(c))
        if a is None or b is None:
            skipped["unresolved"] += 1
            continue
        rows.append({
            "entry_ts": ts[t], "dir": d, "entry": c[t], "stop": stop,
            "stop_sigma": abs(c[t] - stop) / (sig[t] * c[t]),
            "r_a": r_multiple(d, c[t], stop, a.price, cost), "exit_a": a.reason, "bars_a": a.bar - t, "reverted_a": a.reverted,
            "r_b": r_multiple(d, c[t], stop, b.price, cost), "exit_b": b.reason, "bars_b": b.bar - t,
            "fallback_hours": max(a.fallback_hours, b.fallback_hours),
        })
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    return df.with_columns((pl.col("r_b") - pl.col("r_a")).alias("diff")) if rows else df


def segment(df: pl.DataFrame, cutoff: datetime | None = None) -> dict[str, pl.DataFrame]:
    k = cutoff or knowledge_cutoff()
    return {"llm_in_sample": df.filter(pl.col("entry_ts") < k), "clean_holdout": df.filter(pl.col("entry_ts") >= k)}


def summarize(df: pl.DataFrame) -> dict:
    n = df.height
    if n == 0:
        return {"n": 0}
    dv = df["diff"].to_numpy()
    sd = dv.std(ddof=1) if n > 1 else float("nan")
    t_simple = float(dv.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    _, t_nw, _ = newey_west_t(dv)
    stops_a = df.filter(pl.col("exit_a") == "stop")
    return {
        "n": n,
        "mean_r_a": float(df["r_a"].mean()), "mean_r_b": float(df["r_b"].mean()),
        "diff_mean": float(dv.mean()), "t_simple": t_simple, "t_nw": t_nw,
        "win_a": float((df["r_a"] > 0).mean()), "win_b": float((df["r_b"] > 0).mean()),
        "worst_r_a": float(df["r_a"].min()), "worst_r_b": float(df["r_b"].min()),
        "stop_exits_a": stops_a.height,
        "stop_touch_reverted": float(stops_a["reverted_a"].mean()) if stops_a.height else float("nan"),
        "close_stop_exits_b": int((df["exit_b"] == "close_stop").sum()),
        "catastrophe_exits_b": int((df["exit_b"] == "catastrophe").sum()),
        "median_stop_sigma": float(df["stop_sigma"].median()),
        "fallback_trades": int((df["fallback_hours"] > 0).sum()),
    }


def passes_in_sample(m: dict) -> bool:
    """Pre-declared: diff > 0, min(simple t, NW t) > 2, worst R under B >= -2.5."""
    return m.get("n", 0) > 0 and m["diff_mean"] > 0 and min(m["t_simple"], m["t_nw"]) > 2 and m["worst_r_b"] >= -2.5


def passes_holdout(m: dict) -> bool:
    return m["diff_mean"] > 0 and m["worst_r_b"] >= -2.5
