"""Relative value tools (live use): factor fair value vs BTC/ETH, and pair spreads.

Both read Bybit 1h `last` closes known at `as_of`, on synchronous bars (inner join on event_ts), and use the
causal features in ai_trader.features.relative_value. Validation: research/reports/relative_value_2026-11-07.md
(H-RV1 / H-RV2 / H-RV3); see config/tools.yaml for status.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from ai_trader.core.registry import tool
from ai_trader.core.schemas import ToolResult
from ai_trader.features.relative_value import ar1_half_life, factor_residuals, spread_z
from ai_trader.storage.lake import Lake

FACTORS = ("BTCUSDT", "ETHUSDT")


def _closes(lake: Lake, sym: str, as_of: datetime) -> pl.DataFrame:
    b = lake.bars(sym, "1h", as_of)
    if b.height == 0:
        raise ValueError(f"{sym}: no 1h bars known at {as_of}")
    return b.filter(pl.col("volume") > 0).select("event_ts", "known_ts", pl.col("close").alias(sym)).unique("event_ts").sort("event_ts")


def _sync(lake: Lake, syms: list[str], as_of: datetime) -> pl.DataFrame:
    out = _closes(lake, syms[0], as_of)
    for s in syms[1:]:
        f = _closes(lake, s, as_of)
        out = out.join(f, on="event_ts", how="inner").with_columns(pl.max_horizontal("known_ts", "known_ts_right").alias("known_ts")).drop("known_ts_right")
    return out.sort("event_ts")


def _logret(px: np.ndarray, ts: np.ndarray) -> np.ndarray:
    r = np.full(len(px), np.nan)
    r[1:] = np.where(np.diff(ts) == np.timedelta64(1, "h"), np.diff(np.log(px)), np.nan)
    return r


@tool("relval.factor_gap", "0.1.0")
def factor_gap(symbol: str, *, as_of: datetime, window: int = 720, lookback: int = 72, lake: Lake | None = None) -> ToolResult:
    """Fair value of `symbol` vs a BTC/ETH factor model (ETH vs BTC only). Betas are the OLS fit on the last
    `window` hours (applies to the next bar); residual z = last-`lookback`h residual sum / (σ_e √lookback) with the
    causal betas; gap % = price vs the price its factor exposure alone implies over the last `window` hours."""
    lake = lake or Lake()
    factors = [f for f in FACTORS if f != symbol]
    if not factors:
        raise ValueError("BTCUSDT is the factor itself")
    d = _sync(lake, [symbol, *factors], as_of).tail(2 * window + lookback + 2)
    ts = d["event_ts"].to_numpy()
    F = np.column_stack([_logret(d[f].to_numpy(), ts) for f in factors])
    px = d[symbol].to_numpy()
    out = factor_residuals(_logret(px, ts), F, window, lookback, alt_price=px)
    z, gap = float(out["z"][-1]), float(out["gap_pct"][-1])
    b = out["beta_next"][-1]
    if not np.isfinite(b).all():
        raise ValueError(f"{symbol}: not enough synchronous history for a {window}h fit at {as_of}")
    betas = {f"beta_{f.removesuffix('USDT').lower()}": float(v) for f, v in zip(factors, b, strict=True)}
    value = {**betas, "resid_z": z, "cum_resid_pct": 100 * float(np.expm1(out["cum_resid"][-1])), "gap_pct": 100 * gap,
             "fair_price": float(out["fair_price"][-1]), "price": float(px[-1]), "alpha_bps_per_h": 1e4 * float(out["alpha_next"][-1]),
             "window_h": window, "lookback_h": lookback}
    bs = ", ".join(f"β_{k[5:].upper()} {v:.2f}" for k, v in betas.items())
    flag = " |z|>=2 (H-RV1 signal level)" if abs(z) >= 2 else ""
    return ToolResult(
        tool="relval.factor_gap", version="0.1.0", asset=symbol, as_of=as_of, data_max_ts=d["known_ts"].max(),
        value=value,
        uncertainty={"resid_std_h": float(out["resid_std"][-1]), "r2": float(out["r2"][-1])},
        quality={"n_bars": d.height, "validated_by": "H-RV1 (see config/tools.yaml status)"},
        summary=f"{symbol} vs {'+'.join(factors)}: {bs}; {lookback}h residual z {z:+.2f}{flag}; "
        f"{100 * gap:+.1f}% vs {window}h factor fair value {value['fair_price']:.6g}",
    )


@tool("relval.pair", "0.1.0")
def pair(sym_a: str, sym_b: str, *, as_of: datetime, window: int = 720, hedge: bool = False, lake: Lake | None = None) -> ToolResult:
    """Log spread ln A - ln B (or the rolling-OLS hedged residual), z vs the trailing `window` synchronous hours,
    and the half-life of mean reversion from an AR(1) fit on the trailing window."""
    lake = lake or Lake()
    d = _sync(lake, [sym_a, sym_b], as_of).tail(2 * window + 2)
    la, lb = np.log(d[sym_a].to_numpy()), np.log(d[sym_b].to_numpy())
    sz = spread_z(la, lb, window, hedge=hedge)
    s = sz["spread"]
    z = float(sz["z"][-1])
    if not np.isfinite(z):
        raise ValueError(f"{sym_a}/{sym_b}: fewer than {window} synchronous bars at {as_of}")
    hl = ar1_half_life(s[-window:])
    value = {"spread_bps": 1e4 * float(s[-1]), "mean_bps": 1e4 * float(sz["mean"][-1]), "std_bps": 1e4 * float(sz["std"][-1]),
             "z": z, "half_life_h": hl["half_life"], "ar1_phi": hl["phi"], "hedge_ratio": float(sz["beta"][-1]), "window_h": window}
    flag = " |z|>=2" if abs(z) >= 2 else ""
    return ToolResult(
        tool="relval.pair", version="0.1.0", asset=f"{sym_a}/{sym_b}", as_of=as_of, data_max_ts=d["known_ts"].max(),
        value=value, uncertainty={"ar1_phi_se": hl.get("phi_se", float("nan")), "n_ar1": hl["n"]},
        quality={"n_bars": d.height, "hedged": hedge, "validated_by": "H-RV2 / H-RV3 (see config/tools.yaml status)"},
        summary=f"{sym_a}/{sym_b} spread {value['spread_bps']:+.1f} bps vs mean {value['mean_bps']:+.1f} (σ {value['std_bps']:.1f}): "
        f"z {z:+.2f}{flag}; AR(1) half-life {hl['half_life']:.1f} h",
    )
