"""Market Cipher B approximation (H-MC1): LazyBear WaveTrend + the VuManChu Cipher B money-flow proxy.

WaveTrend (LazyBear, "WaveTrend Oscillator [WT]", public TradingView script; n1 = 9 channel, n2 = 12 average):
    ap  = hlc3 = (high + low + close) / 3
    esa = EMA(ap, 9)
    d   = EMA(|ap - esa|, 9)
    ci  = (ap - esa) / (0.015 * d)
    wt1 = EMA(ci, 12)
    wt2 = SMA(wt1, 4)
Dots (as in VuManChu Cipher B: wtCross & wtCrossUp & wtOversold, osLevel = -53 / obLevel = 53):
    green: wt1 crosses above wt2 (wt1 - wt2 goes from <= 0 to > 0) with wt2 <= -53
    red:   wt1 crosses below wt2 (wt1 - wt2 goes from >= 0 to < 0) with wt2 >= 53
Money flow ("RSI+MFI area" of the open-source VuManChu Cipher B, f_rsimfi with rsiMFIperiod = 60,
rsiMFIMultiplier = 150, rsiMFIPosY = 2.5):
    mf = SMA(((close - open) / (high - low)) * 150, 60) - 2.5
    (a bar with high == low contributes 0 instead of Pine's na). MF filter: longs need mf > 0, shorts mf < 0.

EMA follows Pine's `ema` seeding (first value = first input, alpha = 2 / (n + 1)); SMA is a trailing window
(nan until full). All outputs are causal: value t uses bars <= t only, and the recursion starts at the first
bar of the input, so truncating the input never changes earlier values (tests/test_cipher.py).
"""

from __future__ import annotations

import numpy as np

OB_LEVEL = 53.0
OS_LEVEL = -53.0


def ema(x, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    a = 2.0 / (n + 1.0)
    prev = np.nan
    for i, v in enumerate(x.tolist()):
        if v != v:  # nan input: carry the previous value
            out[i] = prev
            continue
        prev = v if prev != prev else a * v + (1.0 - a) * prev
        out[i] = prev
    return out


def sma(x, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).mean(axis=1)
    return out


def wavetrend(high, low, close, n1: int = 9, n2: int = 12, n_sig: int = 4) -> tuple[np.ndarray, np.ndarray]:
    h, lo, c = (np.asarray(v, dtype=float) for v in (high, low, close))
    ap = (h + lo + c) / 3.0
    esa = ema(ap, n1)
    d = ema(np.abs(ap - esa), n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(d > 0, (ap - esa) / (0.015 * d), 0.0)
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, n_sig)
    return wt1, wt2


def money_flow(open_, high, low, close, period: int = 60, mult: float = 150.0, pos_y: float = 2.5) -> np.ndarray:
    o, h, lo, c = (np.asarray(v, dtype=float) for v in (open_, high, low, close))
    rng = h - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(rng > 0, (c - o) / rng, 0.0) * mult
    return sma(x, period) - pos_y


def crosses(wt1: np.ndarray, wt2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cross_up, cross_down) boolean arrays; row t compares t with t-1 only."""
    dlt = np.asarray(wt1, dtype=float) - np.asarray(wt2, dtype=float)
    prev = np.r_[np.nan, dlt[:-1]]
    ok = np.isfinite(dlt) & np.isfinite(prev)
    up = ok & (prev <= 0) & (dlt > 0)
    dn = ok & (prev >= 0) & (dlt < 0)
    return up, dn


def dots(open_, high, low, close, mf_filter: bool) -> dict[str, np.ndarray]:
    """Green / red dots (and the raw crosses used for exits)."""
    wt1, wt2 = wavetrend(high, low, close)
    mf = money_flow(open_, high, low, close)
    up, dn = crosses(wt1, wt2)
    green = up & (wt2 <= OS_LEVEL)
    red = dn & (wt2 >= OB_LEVEL)
    if mf_filter:
        green &= np.nan_to_num(mf, nan=0.0) > 0
        red &= np.nan_to_num(mf, nan=0.0) < 0
    return {"wt1": wt1, "wt2": wt2, "mf": mf, "cross_up": up, "cross_down": dn, "green": green, "red": red}


def cipher_trades(open_, high, low, close, mf_filter: bool, max_hold: int = 12, start: int = 0) -> list[tuple[int, int, int]]:
    """Non-overlapping trades (entry index, exit index, side): enter at the close of a dot bar (green -> +1,
    red -> -1), exit at the close of the first opposite cross within max_hold bars, else after max_hold bars.
    One position at a time; a new dot on the exit bar may open the next trade. Trades whose exit would fall
    beyond the data are dropped. Entries before `start` (warm-up) are ignored."""
    d = dots(open_, high, low, close, mf_filter)
    n = len(d["green"])
    g, r, up, dn = d["green"], d["red"], d["cross_up"], d["cross_down"]
    out = []
    i = max(1, int(start))
    while i < n:
        s = 1 if g[i] else (-1 if r[i] else 0)
        if s == 0:
            i += 1
            continue
        opp = dn if s == 1 else up
        j = None
        for k in range(i + 1, min(i + max_hold, n - 1) + 1):
            if opp[k]:
                j = k
                break
        if j is None:
            if i + max_hold > n - 1:
                break
            j = i + max_hold
        out.append((i, j, s))
        i = j
    return out
