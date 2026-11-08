"""Calendar strategies: the weekend / TradFi-closure family (research/hypotheses.yaml H-W2..H-W4).

These are tagged *seasonal*, the one family allowed a calendar exit (ARCHITECTURE §3.12): the
edge is the TradFi absorption window itself, so positions opened at the CME reopen are closed
at Monday's NYSE close unless a price condition exits them first.

Each needs bars of 1h or finer (a 4h/1d bar can't resolve the weekend window); on coarser bars
the signal is all zeros.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ai_trader.research.calendar import weekend_context

from .base import Strategy, StrategySpec, register


def _bar_hours(frame: pl.DataFrame) -> float:
    if frame.height < 2:
        return np.inf
    return float((frame["close_ts"][0] - frame["event_ts"][0]).total_seconds() / 3600)


def _run(frame: pl.DataFrame, entry_fn, exit_fn) -> pl.Series:
    """Shared state machine: enter on the first bar after a weekend, exit on a price condition
    or when the absorption window ends (Monday NYSE close)."""
    if _bar_hours(frame) > 1.0:
        return pl.Series([0.0] * frame.height)
    c = weekend_context(frame)
    close = c["close"].to_numpy()
    first = c["wk_first_bar"].to_numpy()
    absorbed = c["wk_absorbed"].to_numpy()
    ctx = {k: c[f"ctx_{k}"].to_numpy() for k in ("fri_close", "wk_z", "vol_ratio", "wk_high", "wk_low")}
    out = np.zeros(c.height)
    pos = 0.0
    entry_ctx = None
    for t in range(c.height):
        if pos != 0.0 and (absorbed[t] or exit_fn(pos, close[t], entry_ctx)):
            pos, entry_ctx = 0.0, None
        if pos == 0.0 and first[t] and not absorbed[t] and np.isfinite(ctx["wk_z"][t]):
            snap = {k: v[t] for k, v in ctx.items()}
            pos = entry_fn(snap)
            entry_ctx = snap if pos != 0.0 else None
        out[t] = pos
    return pl.Series(out)


@register
class WeekendShockFollow(Strategy):
    spec = StrategySpec(
        name="weekend_shock_follow",
        version="0.1.0",
        family="calendar",
        description="At the CME reopen, follow a large weekend move that happened on above-normal activity "
        "(information the closed TradFi venues couldn't trade on). Exit if price crosses back through the "
        "Friday close (thesis invalid) or at Monday's NYSE close (absorption window over).",
        hypothesis="Weekend moves of |z| >= 2 on turnover above the weekday baseline continue into Monday's TradFi session (H-W3).",
        requires=("bars", "calendar"),
        expected_regimes=("R2", "R3", "R6", "R8", "R9"),
        params={"z_min": 2.0, "z_full": 4.0, "vol_ratio_min": 1.2},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params

        def entry(s):
            if abs(s["wk_z"]) >= p["z_min"] and s["vol_ratio"] >= p["vol_ratio_min"]:
                return float(np.sign(s["wk_z"]) * min(1.0, abs(s["wk_z"]) / p["z_full"]))
            return 0.0

        def exit_(pos, px, s):
            return (pos > 0 and px < s["fri_close"]) or (pos < 0 and px > s["fri_close"])

        return _run(frame, entry, exit_)


@register
class WeekendDriftFade(Strategy):
    spec = StrategySpec(
        name="weekend_drift_fade",
        version="0.1.0",
        family="calendar",
        description="At the CME reopen, fade a sizeable weekend move that happened on below-normal activity "
        "(thin-liquidity drift, no information). Target: the Friday close. Invalidation: a new extreme beyond the "
        "weekend high/low in the drift direction. Calendar exit: Monday's NYSE close.",
        hypothesis="Weekend moves of |z| >= 1 on turnover below the weekday baseline revert toward the Friday close by Monday's NYSE close (H-W2).",
        requires=("bars", "calendar"),
        expected_regimes=("R4", "R5", "R1", "R7"),
        params={"z_min": 1.0, "z_full": 3.0, "vol_ratio_max": 0.9},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params

        def entry(s):
            if abs(s["wk_z"]) >= p["z_min"] and s["vol_ratio"] <= p["vol_ratio_max"]:
                return float(-np.sign(s["wk_z"]) * min(1.0, abs(s["wk_z"]) / p["z_full"]))
            return 0.0

        def exit_(pos, px, s):
            if pos > 0:  # faded a down-drift: target Friday close above, invalid below weekend low
                return px >= s["fri_close"] or px < s["wk_low"]
            return px <= s["fri_close"] or px > s["wk_high"]

        return _run(frame, entry, exit_)


@register
class CmeGapFill(Strategy):
    spec = StrategySpec(
        name="cme_gap_fill",
        version="0.1.0",
        family="calendar",
        description="At the CME reopen, fade the weekend gap toward the Friday close regardless of activity; exit "
        "on fill or at Monday's NYSE close. The unconditional version of the drift fade, kept separate so the "
        "activity filter's contribution can be measured.",
        hypothesis="The Friday CME close is revisited by Monday's NYSE close more often than a driftless random walk implies (H-W4).",
        requires=("bars", "calendar"),
        expected_regimes=("R4", "R5", "R6"),
        params={"z_min": 0.75, "z_full": 3.0},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        p = self.params

        def entry(s):
            if abs(s["wk_z"]) >= p["z_min"]:
                return float(-np.sign(s["wk_z"]) * min(1.0, abs(s["wk_z"]) / p["z_full"]))
            return 0.0

        def exit_(pos, px, s):
            return (pos > 0 and px >= s["fri_close"]) or (pos < 0 and px <= s["fri_close"])

        return _run(frame, entry, exit_)
