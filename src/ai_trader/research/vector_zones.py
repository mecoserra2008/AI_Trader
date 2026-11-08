"""Vector-candle OPEN levels on 4h, active until taken (research/hypotheses.yaml H-VZ1..H-VZ4).

Vector (PVSRA climax): volume >= 2 x mean(prior 10) OR volume*range >= max(prior 10 volume*range).
Level L = open of the vector bar, known at its close. Active until the first later bar whose range
contains L (the first touch: "taken"). At a touch the race runs on subsequent closes in the orientation
of the side price came from: reject = close >= L + sigma (oriented), break = close <= L - sigma, with a
per-event driftless null p0 = clip((y0 + sigma) / (2 sigma), 0, 1) from the touch close y0.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from scipy.stats import norm

from ai_trader.core.time import now
from ai_trader.research.frame import market_frame

MAX_WAIT = 500   # bars to look for a touch
MAX_RACE = 200   # bars for a race to resolve
MAGNET_H = 60


def pvsra_vectors(f: pl.DataFrame) -> np.ndarray:
    vol = pl.col("volume")
    vr = vol * (pl.col("high") - pl.col("low"))
    flags = f.select(
        ((vol >= 2 * vol.shift(1).rolling_mean(10, min_samples=10)) | (vr >= vr.shift(1).rolling_max(10, min_samples=10))).fill_null(False)
    ).to_series()
    return flags.to_numpy()


def _race(close: np.ndarray, start: int, L: float, s: float, sig: float) -> tuple[int, float, int] | None:
    """Returns (outcome 1 reject / 0 break, p0, resolution bar) or None if unresolved."""
    y0 = s * (close[start] - L)
    p0 = float(np.clip((y0 + sig) / (2 * sig), 0.0, 1.0))
    for k in range(start, min(start + MAX_RACE, len(close))):
        y = s * (close[k] - L)
        if y >= sig:
            return 1, p0, k
        if y <= -sig:
            return 0, p0, k
    return None


def _first_touch(low: np.ndarray, high: np.ndarray, start: int, L: float) -> int | None:
    end = min(start + MAX_WAIT, len(low))
    hit = np.nonzero((low[start:end] <= L) & (high[start:end] >= L))[0]
    return int(start + hit[0]) if len(hit) else None


def symbol_events(f: pl.DataFrame, sym: str, control_every: int = 4) -> list[dict]:
    o, h, lo, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    sigp = f["sigma_risk_next"].to_numpy() * c  # per-bar sigma in price units
    ts = f["event_ts"].to_list()
    vec = pvsra_vectors(f)
    rows: list[dict] = []
    n = len(c)
    for t in range(20, n - 2):
        is_vec = bool(vec[t])
        if not is_vec and t % control_every:
            continue
        L, s0 = o[t], np.sign(c[t] - o[t])
        if s0 == 0 or not np.isfinite(sigp[t]) or sigp[t] <= 0:
            continue
        if not is_vec:
            # control: price must close >= 1 sigma away from L (same side) before the first touch
            away = None
            for k in range(t + 1, min(t + MAX_WAIT, n)):
                if lo[k] <= L <= h[k]:
                    break
                if s0 * (c[k] - L) >= sigp[t]:
                    away = k
                    break
            if away is None:
                continue
            search_from = away + 1
        else:
            search_from = t + 1
        d = abs(c[t] - L) / sigp[t]
        row = {"symbol": sym, "kind": "vector" if is_vec else "control", "event_ts": ts[t],
               "magnet_base": float(2 * (1 - norm.cdf(d / np.sqrt(MAGNET_H)))) if is_vec else None}
        j = _first_touch(lo, h, search_from, L)
        row["taken_60"] = bool(j is not None and j - t <= MAGNET_H) if is_vec else None
        if j is None or not np.isfinite(sigp[j]) or sigp[j] <= 0:
            rows.append(row)
            continue
        s = np.sign(c[j - 1] - L) or s0  # side price approached from
        r1 = _race(c, j, L, s, sigp[j])
        row.update(touch_ts=ts[j], wait=j - t)
        if r1:
            row.update(rej1=r1[0], p0_1=r1[1])
            k2 = _first_touch(lo, h, r1[2] + 1, L)
            if k2 is not None and np.isfinite(sigp[k2]) and sigp[k2] > 0:
                s2 = np.sign(c[k2 - 1] - L) or s
                r2 = _race(c, k2, L, s2, sigp[k2])
                if r2:
                    row.update(rej2=r2[0], p0_2=r2[1], touch2_ts=ts[k2])
        rows.append(row)
    return rows


def study(symbols: list[str], freq: str = "4h", as_of: datetime | None = None) -> pl.DataFrame:
    as_of = as_of or now()
    rows: list[dict] = []
    for sym in symbols:
        f = market_frame(sym, freq, as_of, funding=False, oi=False)
        if f.height > 300:
            rows += symbol_events(f, sym)
    return pl.DataFrame(rows, infer_schema_length=None)
