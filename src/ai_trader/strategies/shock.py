"""Shock strategies: what happens after a large vol-standardised bar (research/hypotheses.yaml H-S2, H-S3).

`shock_regime` trades the in-sample finding that a |z| >= 4 shock into a *calm* book (vol_state low)
continues, while a shock into an already normal/high-vol book reverses. It was designed AFTER seeing
that evidence, so its in-sample/holdout results are not evidence for it: only forward data counts.

Exits are price/state only (ARCHITECTURE §3.12), no clocks. All exit levels are fixed at entry from
the shock bar (open, high/low, close, sigma_next), like a bracket order.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ai_trader.research.regimes import label

from .base import Strategy, StrategySpec, register


@register
class ShockRegime(Strategy):
    spec = StrategySpec(
        name="shock_regime",
        version="0.1.0",
        family="shock",
        description="On a declustered |z| >= z_min shock bar, read the causal v0 vol_state at that bar. Low vol: "
        "follow the shock (continuation); normal/high vol: fade it (reversal). Size min(1, |z|/z_full). "
        "Continuation exits: close back through the shock bar's open (invalidation) or a close more than "
        "trail_sigma x sigma_next x entry_close below (above) the best close since entry (vol-scaled trail). "
        "Reversal exits: close back at the shock bar's open (target) or a close beyond the shock bar's "
        "high (low) by more than invalid_sigma x sigma_next x close (invalidation). A new accepted shock "
        "replaces the position. No time-based exits.",
        hypothesis="H-S2/H-S3: after |z| >= 4 shocks the 24h signed return is > 0 when vol_state at shock time is "
        "low (R1/R4/R7) and < 0 when it is normal/high (incl. R9). DATA-GENERATED: this strategy was designed "
        "after seeing the in-sample and holdout evidence, so only forward results (event_ts >= 2026-11-08) count.",
        requires=("bars",),
        # Low-vol cells (continuation leg) plus the normal/high cells where reversal was observed (R5, R6, R8, R9).
        # R2/R3 are left out: no supporting evidence there yet (few events), though the rule still trades them.
        expected_regimes=("R1", "R4", "R7", "R5", "R6", "R8", "R9"),
        params={"z_min": 4.0, "z_full": 8.0, "decluster_bars": 24, "trail_sigma": 2.0, "invalid_sigma": 0.5},
    )

    def signal(self, frame: pl.DataFrame) -> pl.Series:
        n = frame.height
        if n < 2:
            return pl.Series([0.0] * n)
        p = self.params
        vol_state = label(frame)["vol_state"].to_list()
        z = np.nan_to_num(frame["z"].to_numpy().astype(float), nan=0.0)
        o, h, lo, c = (frame[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
        sig = np.nan_to_num(frame["sigma_next"].to_numpy().astype(float), nan=0.0)

        out = np.zeros(n)
        pos = 0.0
        mode = None  # "cont" | "rev"
        ctx: dict[str, float] = {}
        last_shock = -(10**9)
        for t in range(n):
            if abs(z[t]) >= p["z_min"] and t - last_shock > p["decluster_bars"]:
                # Accepted shock (causal decluster: only past accepted shocks matter). Replaces any position.
                last_shock = t
                vs = vol_state[t]
                d = float(np.sign(z[t]))
                size = min(1.0, abs(z[t]) / p["z_full"])
                if vs is None:  # regime warm-up: no classification, no trade
                    pos, mode, ctx = 0.0, None, {}
                else:
                    mode = "cont" if vs == "low" else "rev"
                    pos = d * size if mode == "cont" else -d * size
                    ctx = {"open": o[t], "high": h[t], "low": lo[t], "close": c[t], "sig": sig[t], "best": c[t]}
                out[t] = pos
                continue
            if pos != 0.0:
                px = c[t]
                if mode == "cont":
                    ctx["best"] = max(ctx["best"], px) if pos > 0 else min(ctx["best"], px)
                    trail = p["trail_sigma"] * ctx["sig"] * ctx["close"]
                    if pos > 0:
                        exit_ = px < ctx["open"] or px < ctx["best"] - trail
                    else:
                        exit_ = px > ctx["open"] or px > ctx["best"] + trail
                else:
                    buf = p["invalid_sigma"] * ctx["sig"] * ctx["close"]
                    if pos < 0:  # faded an up-shock: target the shock open below, invalid above the high
                        exit_ = px <= ctx["open"] or px > ctx["high"] + buf
                    else:  # faded a down-shock
                        exit_ = px >= ctx["open"] or px < ctx["low"] - buf
                if exit_:
                    pos, mode, ctx = 0.0, None, {}
            out[t] = pos
        return pl.Series(out)
