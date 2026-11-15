"""Book features on a fixed clock grid from Bybit ob200 archive files (snapshot + delta JSON lines), for the
microstructure pilot (H-MS9 registered features: spread and depth imbalance).

The archive is replayed message by message; the book state at grid time t is the state after every message with
exchange ts <= t, so a trade at time T may only use the grid row with t <= T (point in time). Grid rows inside a
gap of the feed carry the last state forward (`age_ms` says how stale it is).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.sim.l2_features data/raw/pilot_2026-11-10/ob200/<file>.zip out.parquet
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

GRID_MS = 250


def _best(side: dict[float, float], bid: bool) -> float:
    if not side:
        return np.nan
    return max(side) if bid else min(side)


def _band(side: dict[float, float], lo: float, hi: float) -> float:
    return float(sum(p * q for p, q in side.items() if lo <= p <= hi))


def book_grid(path: str | Path, grid_ms: int = GRID_MS, band_bps: float = 5.0) -> pl.DataFrame:
    """Columns: t_ms (grid time), bid, ask, spread_bps, obi_l1 = (qb - qa) / (qb + qa) at the touch,
    obi_5 = (Db - Da) / (Db + Da) for notional within `band_bps` of mid, depth_5 (USD, both sides), age_ms."""
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows: list[tuple] = []
    next_t = None
    last_ts = None

    def emit(t: int) -> None:
        b = _best(bids, True)
        a = _best(asks, False)
        if not (np.isfinite(b) and np.isfinite(a)) or a <= b:
            rows.append((t, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, t - (last_ts or t)))
            return
        mid = 0.5 * (a + b)
        qb, qa = bids[b], asks[a]
        lo, hi = mid * (1 - band_bps * 1e-4), mid * (1 + band_bps * 1e-4)
        db, da = _band(bids, lo, mid), _band(asks, mid, hi)
        obi5 = (db - da) / (db + da) if db + da > 0 else np.nan
        rows.append((t, b, a, (a - b) / mid * 1e4, (qb - qa) / (qb + qa), obi5, db + da, t - last_ts))

    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                if not line.strip():
                    continue
                m = json.loads(line)
                ts = int(m["ts"])
                if next_t is None:
                    next_t = (ts // grid_ms + 1) * grid_ms
                while ts > next_t:  # state after all messages with ts <= next_t
                    emit(next_t)
                    next_t += grid_ms
                d = m["data"]
                if m.get("type") == "snapshot":
                    bids.clear()
                    asks.clear()
                for p, q in d.get("b", ()):
                    pf, qf = float(p), float(q)
                    if qf == 0.0:
                        bids.pop(pf, None)
                    else:
                        bids[pf] = qf
                for p, q in d.get("a", ()):
                    pf, qf = float(p), float(q)
                    if qf == 0.0:
                        asks.pop(pf, None)
                    else:
                        asks[pf] = qf
                last_ts = ts
    if next_t is not None:
        emit(next_t)
    return pl.DataFrame(rows, schema=["t_ms", "bid", "ask", "spread_bps", "obi_l1", "obi_5", "depth_5", "age_ms"], orient="row")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    df = book_grid(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst)
    print(dst, df.height, df.select(pl.col("spread_bps").median()).item())
