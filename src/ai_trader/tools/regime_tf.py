"""Trend-vs-mean-reversion regime tool (live use): the 11 causal daily regime features of a symbol and the walk-forward
logistic P(trend-zoo ensemble beats the MR ensemble over the next 7 days).

Features: ai_trader.features.regime_features on Bybit 4h bars known at `as_of` (latest complete UTC day).
Model: the monthly refit (research.regime_switch_study.walk_forward) whose fit month is the latest one <= the feature
date, read from data/priors/regime_switch_4h_v0.parquet (refresh: rerun the study with --save). Validation:
research/reports/regime_switch_2026-11-08.md (H-TR3 / H-TR4); see config/tools.yaml for status.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.registry import tool
from ai_trader.core.schemas import ToolResult
from ai_trader.features.regime_features import FEATURES, regime_features
from ai_trader.storage.lake import Lake

MODEL_FILE = "data/priors/regime_switch_4h_v0.parquet"


def _model(path: Path, day) -> dict:
    m = pl.read_parquet(path).filter(pl.col("fit_month") <= day).sort("fit_month")
    if m.height == 0:
        raise ValueError(f"no regime-switch model fitted on or before {day} in {path}")
    return m.tail(1).to_dicts()[0]


@tool("regime.trend_vs_mr", "0.1.0")
def trend_vs_mr(symbol: str, *, as_of: datetime, lake: Lake | None = None, model_path: str | Path | None = None) -> ToolResult:
    """Latest daily regime features of `symbol` (4h bars) and P(trend ensemble beats MR ensemble over the next 7 days)."""
    from ai_trader.research.frame import market_frame

    lake = lake or Lake()
    f = market_frame(symbol, "4h", as_of, funding=False, oi=False, lake=lake)
    if f.height == 0:
        raise ValueError(f"{symbol}: no 4h bars known at {as_of}")
    feats = regime_features(f).filter(pl.col("known_ts") <= as_of)
    ok = feats.filter(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURES]))
    if ok.height == 0:
        raise ValueError(f"{symbol}: not enough 4h history for the regime features at {as_of} (needs ~180 days)")
    row = ok.tail(1).to_dicts()[0]
    path = Path(model_path) if model_path else load().root / MODEL_FILE
    m = _model(path, row["date"])
    zc = {c: m[f"b_{c}"] * (row[c] - m[f"mu_{c}"]) / m[f"sd_{c}"] for c in FEATURES}
    p = float(1.0 / (1.0 + np.exp(-(m["intercept"] + sum(zc.values())))))
    top = sorted(zc.items(), key=lambda kv: -abs(kv[1]))[:3]
    status = load().tools["tools"]["regime.trend_vs_mr"].get("status", "planned")
    value = {"date": str(row["date"]), "p_trend_wins": p, "gate": "trend" if p > 0.5 else "flat", "base_rate": m["base_rate"],
             "fit_month": str(m["fit_month"]), **{c: row[c] for c in ("er_14d", *FEATURES)},
             "logit_contrib": {c: float(v) for c, v in zc.items()}}
    drivers = ", ".join(f"{c} {v:+.2f}" for c, v in top)
    return ToolResult(
        tool="regime.trend_vs_mr", version="0.1.0", asset=symbol, as_of=as_of, data_max_ts=f["known_ts"].max(),
        value=value,
        uncertainty={"n_train": m["n_train"], "n_weeks": m["n_weeks"]},
        quality={"status": status, "validated_by": "H-TR3 / H-TR4 (see config/tools.yaml status)", "n_days": feats.height},
        summary=f"{symbol} {row['date']}: P(trend beats MR, next 7d) {p:.2f} (base rate {m['base_rate']:.2f}); ER7/30 "
        f"{row['er_7d']:.2f}/{row['er_30d']:.2f}, DVR30 {row['dvr_30d']:+.2f}, VR30 {row['vr_30d']:.2f}, vol pct {row['vol_pct_1y']:.2f}; "
        f"top drivers {drivers}" + ("" if status == "active" else " [descriptive only: H-TR3 not supported]"),
    )
