"""Causal daily volume profile (auction / market-profile levels) from intraday bars.

Definition (research/hypotheses.yaml, batch 2026-11-07 night):
  * session = one UTC calendar day; a bar belongs to the day of its event_ts (bar open time);
  * each bar's volume is spread **uniformly in price over [low, high]** (a zero-range bar puts all its
    volume in the bin containing its price);
  * price grid: geometric bins of 0.05% of price, bin k = [e^(k*w), e^((k+1)*w)), w = ln(1.0005). The grid is
    global (the same edges every day), so levels of different days are directly comparable;
  * POC = the max-volume bin (ties: the tied bin nearest the session's volume-weighted mean bin);
  * value area (VA) = grown contiguously from the POC, each step adding the larger of the two adjacent bins
    (an exact tie adds both) until it holds >= 70% of the session's volume. VAL = lower edge of its lowest
    bin, VAH = upper edge of its highest bin; POC price = geometric centre of the POC bin.

Causality: the profile of day D uses only bars of day D and is stamped `known_ts` = the latest known_ts of
those bars, i.e. it is known only at the end of day D. `asof_profile` attaches, to each row of a decision
frame, the latest profile whose known_ts <= the row's known_ts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

BIN_FRAC = 0.0005
VA_FRAC = 0.70


@dataclass(frozen=True)
class Profile:
    poc: float
    val: float
    vah: float
    volume: float
    va_share: float          # share of volume inside [val, vah] (>= va_frac)
    edges: np.ndarray        # bin edges (K + 1)
    bins: np.ndarray         # volume per bin (K)
    poc_bin: int
    va_bins: tuple[int, int]  # inclusive bin index range of the VA


def bin_volumes(low: np.ndarray, high: np.ndarray, volume: np.ndarray, bin_frac: float = BIN_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Volume per bin on the global geometric grid covering [min low, max high]. Returns (edges, bins)."""
    low, high, volume = (np.asarray(a, dtype=float) for a in (low, high, volume))
    ok = np.isfinite(low) & np.isfinite(high) & np.isfinite(volume) & (low > 0) & (high >= low) & (volume >= 0)
    low, high, volume = low[ok], high[ok], volume[ok]
    if len(low) == 0:
        return np.array([]), np.array([])
    w = np.log1p(bin_frac)
    k0 = int(np.floor(np.log(low.min()) / w)) - 1
    k1 = int(np.floor(np.log(high.max()) / w)) + 2
    edges = np.exp(np.arange(k0, k1 + 1) * w)
    # F(x) = volume traded below price x; bin volume = F(e_{k+1}) - F(e_k).
    rng = high - low
    pos = rng > 0
    F = np.zeros(len(edges))
    if pos.any():
        frac = np.clip((edges[None, :] - low[pos, None]) / rng[pos, None], 0.0, 1.0)
        F += volume[pos] @ frac
    bins = np.diff(F)
    if (~pos).any():
        idx = np.clip(np.searchsorted(edges, low[~pos], side="right") - 1, 0, len(bins) - 1)
        np.add.at(bins, idx, volume[~pos])
    return edges, np.clip(bins, 0.0, None)


def _close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))


def value_area(bins: np.ndarray, va_frac: float = VA_FRAC) -> tuple[int, int, int]:
    """(poc_bin, lo_bin, hi_bin) with the greedy contiguous growth rule."""
    total = bins.sum()
    m = bins.max()
    tied = np.flatnonzero(np.isclose(bins, m, rtol=1e-9, atol=1e-12))
    centre = (np.arange(len(bins)) * bins).sum() / total if total > 0 else (len(bins) - 1) / 2
    poc = int(tied[np.argmin(np.abs(tied - centre))])
    lo = hi = poc
    acc = bins[poc]
    target = va_frac * total
    K = len(bins)
    while acc < target * (1 - 1e-12) and (lo > 0 or hi < K - 1):
        up = bins[hi + 1] if hi < K - 1 else -1.0
        dn = bins[lo - 1] if lo > 0 else -1.0
        if up >= 0 and dn >= 0 and _close(up, dn):
            hi, lo, acc = hi + 1, lo - 1, acc + up + dn
        elif up > dn:
            hi, acc = hi + 1, acc + up
        else:
            lo, acc = lo - 1, acc + dn
    return poc, lo, hi


def session_profile(low: np.ndarray, high: np.ndarray, volume: np.ndarray, bin_frac: float = BIN_FRAC, va_frac: float = VA_FRAC) -> Profile | None:
    edges, bins = bin_volumes(low, high, volume, bin_frac)
    if len(bins) == 0 or bins.sum() <= 0:
        return None
    poc, lo, hi = value_area(bins, va_frac)
    return Profile(
        poc=float(np.sqrt(edges[poc] * edges[poc + 1])), val=float(edges[lo]), vah=float(edges[hi + 1]),
        volume=float(bins.sum()), va_share=float(bins[lo:hi + 1].sum() / bins.sum()),
        edges=edges, bins=bins, poc_bin=poc, va_bins=(lo, hi),
    )


def daily_profiles(bars: pl.DataFrame, bin_frac: float = BIN_FRAC, va_frac: float = VA_FRAC) -> pl.DataFrame:
    """One row per UTC day: day, poc, val, vah, volume, va_share, n_bars, first_ts, known_ts.

    `bars` needs event_ts, high, low, volume (+ known_ts; if absent, known_ts = last event_ts + bar length is
    NOT inferred: the column is required so the as-of stamp is the true publication time)."""
    b = bars.sort("event_ts").with_columns(pl.col("event_ts").dt.date().alias("_day"))
    days = b["_day"].cast(pl.Int32).to_numpy()
    day_vals = b["_day"].to_list()
    lo, hi, vol = (b[c].cast(pl.Float64).to_numpy() for c in ("low", "high", "volume"))
    ev, kn = b["event_ts"], b["known_ts"]
    starts = np.r_[0, np.flatnonzero(days[1:] != days[:-1]) + 1]
    ends = np.r_[starts[1:], len(days)]
    rows = []
    for s, e in zip(starts, ends):
        p = session_profile(lo[s:e], hi[s:e], vol[s:e], bin_frac, va_frac)
        if p is None:
            continue
        rows.append({"day": day_vals[s], "poc": p.poc, "val": p.val, "vah": p.vah, "volume": p.volume, "va_share": p.va_share,
                     "n_bars": int(e - s), "first_ts": ev[int(s)], "known_ts": kn[s:e].max()})
    if not rows:
        return pl.DataFrame(schema={"day": pl.Date, "poc": pl.Float64, "val": pl.Float64, "vah": pl.Float64})
    return pl.DataFrame(rows).with_columns(pl.col("day").cast(pl.Date))


def va_overlap(val: np.ndarray, vah: np.ndarray, val_ref: np.ndarray, vah_ref: np.ndarray) -> np.ndarray:
    """Share of the VA [val, vah] that lies inside the reference VA [val_ref, vah_ref] (0..1)."""
    inter = np.clip(np.minimum(vah, vah_ref) - np.maximum(val, val_ref), 0.0, None)
    width = vah - val
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(width > 0, inter / width, np.nan)


def asof_profile(frame: pl.DataFrame, profiles: pl.DataFrame, prefix: str = "pf_") -> pl.DataFrame:
    """Attach to each frame row the latest profile with known_ts <= row known_ts (backward as-of join)."""
    cols = [c for c in ("day", "poc", "val", "vah") if c in profiles.columns]
    r = profiles.select(pl.col("known_ts").alias("_k"), *[pl.col(c).alias(prefix + c) for c in cols]).sort("_k")
    return frame.sort("known_ts").join_asof(r, left_on="known_ts", right_on="_k", strategy="backward").drop("_k")
