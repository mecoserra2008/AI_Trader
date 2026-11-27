"""Point-in-time next-hour volatility from the frozen H-NN8 XGBoost model (data/models/ml/hourly_vol_v1), next to the
validated baseline robust_bc x hour-of-week factor (R-VOL-26). Status: probation. The forward test is H-NN10, and the
decision layer must not rely on the ML number before it passes (R-ML-7: central expectations only, never stops or risk).

    from ai_trader.tools.vol_ml import hourly_vol_forecast
    hourly_vol_forecast("BTCUSDT", as_of)

The forecast is for the 1h bar that opens at the close of the last 1h bar known at as_of (`forecast_for`). Inputs are
the market_frame 1h bars with known_ts <= as_of (Lake.read_asof). The features of that next bar are computed by
appending an empty placeholder bar and running ml.hourly.symbol_hourly. Every feature of bar t uses bars <= t-1 only, so
the row is identical to the one the model was trained on (tests/test_vol_ml.py checks this). `flags` contains 'stale'
when as_of is more than 1h past forecast_for, i.e. the forecast bar has already closed or data are late.

Not registered in config/tools.yaml (proposed key: vol.hourly_ml, status probation); `tool_result` wraps the output in
a ToolResult for the journal.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.ml.frozen import FrozenHourly, load_frozen
from ai_trader.ml.hourly import HOURLY_FEATURES, symbol_hourly

TOOL = "vol.hourly_ml"
VERSION = "0.1.0"


@lru_cache(maxsize=4)
def _model(path: str | None) -> FrozenHourly:
    return load_frozen(Path(path) if path else None)


def inputs_hash(model_id: str, symbol: str, forecast_for: str, x: np.ndarray) -> str:
    payload = json.dumps([model_id, symbol, forecast_for, [float(f"{v:.12g}") for v in np.asarray(x, dtype=float)]])
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _next_bar_row(f: pl.DataFrame) -> tuple[pl.DataFrame, datetime]:
    from ai_trader.risk.sigma import pin_base

    f = pin_base(f).select("event_ts", "ret", "sigma_risk_next").sort("event_ts")   # robust_bc base (risk.sigma roles)
    nxt = f["event_ts"][-1] + timedelta(hours=1)
    ph = pl.DataFrame({"event_ts": [nxt], "ret": [None], "sigma_risk_next": [None]}, schema=f.schema)
    h = symbol_hourly(pl.concat([f, ph]))
    return h.filter(pl.col("event_ts") == nxt), nxt


def hourly_vol_forecast(symbol: str, as_of: datetime, *, lake=None, model_path: str | Path | None = None) -> dict:
    from ai_trader.research.frame import market_frame

    m = _model(str(model_path) if model_path else None)
    f = market_frame(symbol, "1h", as_of, funding=False, oi=False, lake=lake)
    out = {"tool": TOOL, "version": VERSION, "model_id": m.card["model_id"], "symbol": symbol, "as_of": as_of.isoformat(), "flags": []}
    if f.height < 24 * 60:
        out["flags"].append("insufficient_history")
        return out
    row, nxt = _next_bar_row(f)
    data_max = f["known_ts"].max()
    x = np.array([row[c][0] if row[c][0] is not None else np.nan for c in HOURLY_FEATURES], dtype=float)
    s2 = row["s2"][0]
    fac = row["fac"][0]
    out.update({"forecast_for": nxt.isoformat(), "data_max_ts": data_max.isoformat(), "features": dict(zip(HOURLY_FEATURES, x.tolist(), strict=True))})
    if as_of - nxt > timedelta(hours=1):
        out["flags"].append("stale")
    if s2 is None or fac is None or not np.isfinite(x).all():
        out["flags"].append("incomplete_features")
        out["missing"] = [c for c, v in zip(HOURLY_FEATURES, x, strict=True) if not np.isfinite(v)] + (["fac"] if fac is None else [])
        if s2 is not None:
            out["sigma_next_hour_baseline"] = math.sqrt(s2 * fac) if fac is not None else None
        return out
    X = x[None, :]
    var_ml = float(m.variance(X, np.array([s2]))[0])
    var_base = float(s2 * fac)
    contr = m.contribs(X)[0]
    groups = {g: float(sum(contr[HOURLY_FEATURES.index(c)] for c in cols)) for g, cols in m.card["groups"].items()}
    top = sorted(groups.items(), key=lambda kv: -abs(kv[1]))[:3]
    out.update({
        "sigma_next_hour_ml": math.sqrt(var_ml), "sigma_next_hour_baseline": math.sqrt(var_base), "ratio": math.sqrt(var_ml / var_base),
        "s2": float(s2), "hour_of_week_factor": float(fac), "log_margin": float(contr.sum()), "bias": float(m.card["bias"]),
        "top_shap_groups": [[g, round(v, 6)] for g, v in top], "shap_groups": {g: round(v, 6) for g, v in groups.items()},
        "inputs_hash": inputs_hash(m.card["model_id"], symbol, nxt.isoformat(), x),
    })
    return out


def tool_result(symbol: str, *, as_of: datetime, lake=None, model_path: str | Path | None = None):
    """The forecast as a ToolResult (for the journal; not registered in config/tools.yaml)."""
    from ai_trader.core.schemas import ToolResult
    from ai_trader.core.time import utc

    v = hourly_vol_forecast(symbol, as_of, lake=lake, model_path=model_path)
    dm = utc(datetime.fromisoformat(v["data_max_ts"])) if v.get("data_max_ts") else as_of
    if "sigma_next_hour_ml" in v:
        summ = (f"{symbol} next hour ({v['forecast_for']}): ML sigma {v['sigma_next_hour_ml']:.3%} vs baseline {v['sigma_next_hour_baseline']:.3%} "
                f"(ratio {v['ratio']:.2f}; top {', '.join(f'{g} {c:+.2f}' for g, c in v['top_shap_groups'])}); probation, H-NN10 forward test")
    else:
        summ = f"{symbol}: no ML forecast ({', '.join(v['flags'])})"
    return ToolResult(tool=TOOL, version=VERSION, asset=symbol, as_of=as_of, data_max_ts=dm, value=v,
                      quality={"status": "probation", "forward_test": "H-NN10", "flags": v["flags"]}, summary=summ)


__all__ = ["TOOL", "VERSION", "hourly_vol_forecast", "inputs_hash", "tool_result"]
