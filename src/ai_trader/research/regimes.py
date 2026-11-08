"""Regime labeller v0: the 9-cell grid (ARCHITECTURE §3.2) from transparent causal rules.

direction: D_t = mean_k clip(z_k, −3, 3),  z_k = log(c_t / c_{t−k}) / (σ̂_{t+1|t} · √k)
           up if D > θ, down if D < −θ, else range
vol:       σ̂_{t+1|t} vs its own trailing quantiles (low < q_low ≤ normal ≤ q_high < high)
persistence: a state change needs `confirm_bars` consecutive bars of agreement (hysteresis)

Cells: up/{low,normal,high} = R1/R2/R3, range = R4/R5/R6, down = R7/R8/R9.
This is a *baseline*: no parameters are fitted, so it can't overfit, but it isn't optimal either.
The factorial HMM / jump model (tools.yaml regime.direction / regime.vol) must beat it out of sample.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ai_trader.core.config import load

DIR = ("up", "range", "down")
VOL = ("low", "normal", "high")


def cell_name(d: str, v: str) -> str:
    return f"R{DIR.index(d) * 3 + VOL.index(v) + 1}"


CELL_LABELS = {
    "R1": "quiet trend up", "R2": "trend up", "R3": "momentum burst",
    "R4": "compression", "R5": "balanced range", "R6": "whipsaw",
    "R7": "grind down", "R8": "trend down", "R9": "liquidation",
}


def _hysteresis(raw: np.ndarray, confirm: int) -> np.ndarray:
    """Causal debounce: switch to a new state only after `confirm` consecutive raw observations."""
    out = np.empty(len(raw), dtype=object)
    state, cand, run = None, None, 0
    for t, r in enumerate(raw):
        if r is None:
            out[t] = state
            continue
        if state is None:
            state = r
        elif r != state:
            run = run + 1 if r == cand else 1
            cand = r
            if run >= confirm:
                state, run, cand = r, 0, None
        else:
            run, cand = 0, None
        out[t] = state
    return out


def label(frame: pl.DataFrame, bar_hours: float | None = None) -> pl.DataFrame:
    """Adds dir_score, dir_state, vol_state, cell (null during warm-up)."""
    s = load().settings["regimes"]
    v0 = s["v0"]
    if bar_hours is None:
        bar_hours = (frame["close_ts"][0] - frame["event_ts"][0]).total_seconds() / 3600
    ks = sorted({max(1, round(h / bar_hours)) for h in v0["direction_horizons_hours"]})
    w = max(20, round(v0["vol_window_hours"] / bar_hours))
    lc, sig = pl.col("close").log(), pl.col("sigma_next")
    zk = [((lc - lc.shift(k)) / (sig * np.sqrt(k))).clip(-3, 3) for k in ks]
    f = frame.with_columns(pl.mean_horizontal(zk).alias("dir_score"))
    f = f.with_columns(
        pl.col("sigma_next").rolling_quantile(v0["vol_low_quantile"], window_size=w, min_samples=w // 2).alias("_vq_lo"),
        pl.col("sigma_next").rolling_quantile(v0["vol_high_quantile"], window_size=w, min_samples=w // 2).alias("_vq_hi"),
    )
    th = v0["direction_threshold"]
    d_raw = f.select(
        pl.when(pl.col("dir_score").is_null() | pl.col("dir_score").is_nan()).then(None)
        .when(pl.col("dir_score") > th).then(pl.lit("up"))
        .when(pl.col("dir_score") < -th).then(pl.lit("down"))
        .otherwise(pl.lit("range"))
    ).to_series().to_numpy()
    v_raw = f.select(
        pl.when(pl.col("_vq_lo").is_null()).then(None)
        .when(pl.col("sigma_next") < pl.col("_vq_lo")).then(pl.lit("low"))
        .when(pl.col("sigma_next") > pl.col("_vq_hi")).then(pl.lit("high"))
        .otherwise(pl.lit("normal"))
    ).to_series().to_numpy()
    d = _hysteresis(d_raw, s["confirm_bars"])
    v = _hysteresis(v_raw, s["confirm_bars"])
    cells = [cell_name(a, b) if a is not None and b is not None else None for a, b in zip(d, v, strict=True)]
    return f.drop("_vq_lo", "_vq_hi").with_columns(
        pl.Series("dir_state", list(d), dtype=pl.Utf8),
        pl.Series("vol_state", list(v), dtype=pl.Utf8),
        pl.Series("cell", cells, dtype=pl.Utf8),
    )
