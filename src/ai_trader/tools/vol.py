"""Volatility tools. GARCH-t is *fit* here (weekly) and *filtered* in Rust per bar
(crates/at-filters/src/garch.rs) from the JSON parameters written below."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import polars as pl
from arch import arch_model

from ai_trader.core.config import load
from ai_trader.core.registry import tool
from ai_trader.core.schemas import ToolResult
from ai_trader.features.transforms import ewma_sigma_ahead, log_returns
from ai_trader.storage.lake import Lake

# Fit on returns scaled by 100 (the optimiser is far better conditioned); converted back on output.
_SCALE = 100.0


def fit_garch_t(returns: np.ndarray) -> dict:
    """GARCH(1,1), Student-t innovations, constant mean. Returns parameters in raw log-return units."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    res = arch_model(r * _SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    p = res.params
    h_next = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    return {
        "mu": float(p["mu"]) / _SCALE,
        "omega": float(p["omega"]) / _SCALE**2,
        "alpha": float(p["alpha[1]"]),
        "beta": float(p["beta[1]"]),
        "nu": float(p["nu"]),
        "h_next": h_next / _SCALE**2,
        "n_obs": int(len(r)),
        "loglik": float(res.loglikelihood),
        "converged": bool(res.convergence_flag == 0),
    }


def _bars_returns(lake: Lake, symbol: str, freq: str, as_of: datetime, lookback: int):
    df = lake.bars(symbol, freq, as_of)
    if df.height < 50:
        raise ValueError(f"{symbol} {freq}: only {df.height} bars known at {as_of}")
    df = df.tail(lookback + 1)
    return log_returns(df["close"].to_numpy()), df["known_ts"].max()


@tool("vol.ewma", "0.1.0")
def ewma(symbol: str, freq: str, *, as_of: datetime, lam: float = 0.94, lookback: int = 500, lake: Lake | None = None) -> ToolResult:
    lake = lake or Lake()
    r, data_max = _bars_returns(lake, symbol, freq, as_of, lookback)
    sig = ewma_sigma_ahead(r[1:], lam=lam, warmup=20)
    # σ for the next (unseen) bar: one more update with the last return.
    var_next = lam * sig[-1] ** 2 + (1 - lam) * r[-1] ** 2
    s = math.sqrt(var_next)
    return ToolResult(
        tool="vol.ewma", version="0.1.0", asset=symbol, as_of=as_of, data_max_ts=data_max,
        value={"sigma_next": s, "lambda": lam, "freq": freq},
        quality={"n_obs": int(np.isfinite(r).sum())},
        summary=f"EWMA σ next {freq} bar {s:.2%} (λ={lam})",
    )


@tool("vol.garch_t", "0.1.0")
def garch_t(symbol: str, freq: str, *, as_of: datetime, lookback: int = 2000, lake: Lake | None = None, save: bool = True) -> ToolResult:
    lake = lake or Lake()
    r, data_max = _bars_returns(lake, symbol, freq, as_of, lookback)
    p = fit_garch_t(r)
    pers = p["alpha"] + p["beta"]
    half_life = math.log(0.5) / math.log(pers) if 0 < pers < 1 else float("inf")
    if save:
        d = load().path("model_states") / "vol.garch_t"
        d.mkdir(parents=True, exist_ok=True)
        rust_params = {k: p[k] for k in ("mu", "omega", "alpha", "beta", "nu")}
        (d / f"{symbol}_{freq}.json").write_text(json.dumps({"params": rust_params, "h_next": p["h_next"], "fit_as_of": as_of.isoformat(), "data_max_ts": str(data_max), "n_obs": p["n_obs"]}, indent=2))
    s = math.sqrt(p["h_next"])
    # House rules (ARCHITECTURE §1.2): ν ≥ 3 and a stationary variance process. Violations are
    # flagged rather than silently clipped: they usually mean too little history for this freq.
    flags = [f for f, bad in (("persistence_at_boundary", pers > 0.999), ("nu_below_floor", p["nu"] < 3.0), ("not_converged", not p["converged"])) if bad]
    return ToolResult(
        tool="vol.garch_t", version="0.1.0", asset=symbol, as_of=as_of, data_max_ts=data_max,
        value={"sigma_next": s, "persistence": pers, "half_life_bars": half_life, "nu": p["nu"], "freq": freq},
        uncertainty={"n_obs": p["n_obs"]},
        quality={"converged": p["converged"], "loglik": p["loglik"], "flags": flags},
        summary=f"GARCH-t σ next {freq} bar {s:.2%}, persistence {pers:.3f} (half-life {half_life:.0f} bars), ν={p['nu']:.1f}"
        + (f" [UNRELIABLE: {', '.join(flags)}]" if flags else ""),
    )


@tool("vol.robust_bc", "0.1.0")
def robust_bc(symbol: str, freq: str, *, as_of: datetime, horizon: int = 24, lake: Lake | None = None) -> ToolResult:
    """Validated σ (H-V1b): central forecast with the weekend multiplier, risk forecast without it."""
    from ai_trader.features.vol_models import horizon_variance
    from ai_trader.research.calendar import add_calendar
    from ai_trader.research.frame import market_frame

    f = market_frame(symbol, freq, as_of, lake=lake, funding=False, oi=False)
    if f.height < 200:
        raise ValueError(f"{symbol} {freq}: not enough bars at {as_of}")
    last = f.tail(1).row(0, named=True)
    # future bar calendar is known: bars t+1..t+h
    step = last["close_ts"] - last["event_ts"]
    fut = pl.DataFrame({"event_ts": [last["close_ts"] + step * j for j in range(horizon)]}).with_columns(pl.col("event_ts").dt.cast_time_unit("ms"))
    fwk = add_calendar(fut)["tradfi_weekend"].to_numpy()
    base_var = last["sigma_risk_next"] ** 2
    central = float(np.sqrt(base_var * ((~fwk).sum() + fwk.sum() * last["wk_m"] ** 2)))
    risk = float(np.sqrt(base_var * horizon))
    return ToolResult(
        tool="vol.robust_bc", version="0.1.0", asset=symbol, as_of=as_of, data_max_ts=f["known_ts"].max(),
        value={"sigma_next_central": last["sigma_rb_next"], "sigma_next_risk": last["sigma_risk_next"], "weekend_m": last["wk_m"],
               f"sigma_{horizon}bars_central": central, f"sigma_{horizon}bars_risk": risk, "weekend_bars_ahead": int(fwk.sum()), "freq": freq},
        quality={"validated_by": "H-V1b"},
        summary=f"robust σ next {freq} bar {last['sigma_risk_next']:.2%} (risk); next {horizon} bars {risk:.2%} risk / {central:.2%} central "
        f"({int(fwk.sum())} weekend bars ahead, m={last['wk_m']:.2f})",
    )


@tool("vol.har", "0.1.0")
def har(symbol: str, *, as_of: datetime, lake: Lake | None = None) -> ToolResult:
    """Next-UTC-day realised vol forecast from HAR on 1h-derived daily RV (H-VCL1)."""
    from ai_trader.features.vol_models import har_forecast
    from ai_trader.research.frame import market_frame

    f = market_frame(symbol, "1h", as_of, lake=lake, funding=False, oi=False)
    daily = (f.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
             .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n")).filter(pl.col("n") == 24).sort("d"))
    fc = har_forecast(daily["rv"].to_numpy())
    if not np.isfinite(fc[-1]):
        raise ValueError(f"{symbol}: not enough complete days for HAR at {as_of}")
    s = float(np.sqrt(fc[-1]))
    return ToolResult(
        tool="vol.har", version="0.1.0", asset=symbol, as_of=as_of, data_max_ts=f["known_ts"].max(),
        value={"sigma_next_day": s, "last_complete_day": str(daily["d"][-1]), "rv_last_day": float(np.sqrt(daily["rv"][-1]))},
        quality={"validated_by": "H-VCL1", "n_days": daily.height},
        summary=f"HAR next-day σ {s:.2%} (last full day realised {np.sqrt(daily['rv'][-1]):.2%})",
    )
