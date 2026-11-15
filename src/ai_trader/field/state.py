"""Point-in-time BTC state vector for the probability field (program 2026-11-13, agent F1; H-FLD1-3).

Row t is "the world at the close of 1h bar t" (known at the bar's known_ts). The field maps X_t to the law of the next
bar's log return r_{t+1}. All recursions are causal; the same `step` function moves the state in the Monte-Carlo
propagator, so a simulated path evolves exactly as the real state would.

Vol coordinates (multi-speed, jump-robust, deseasonalised EWMAs of 1h log returns; an EWMA form of log-HAR):
    f_t        hour-of-week variance factor of bar t (features.seasonality.profile_asof, frozen per fit)
    x_t        r_t / sqrt(f_t)                                      deseasonalised return
    flag_t     |x_t| > clip * sqrt(vf_{t-1})                         jump flag (the bar was winsorised)
    xc_t       clip(x_t, +-clip * sqrt(vf_{t-1}))
    vk_t       lam_k vk_{t-1} + (1 - lam_k) xc_t^2,  k in fast/med/slow (lam 0.94 / 0.995 / 0.9993: half-lives ~11 h,
               ~6 d, ~41 d)
State coordinates (raw units; the field standardises them with training means / sds):
    x1   0.5 ln(vf / vm)       short vs medium log vol (a vol shock)
    x2   0.5 ln(vm / vs)       medium vs long log vol
    L    0.5 ln vs             long-run log vol level (per hour, deseasonalised)
    s    0.5 ln f_{k(t+1)}     hour-of-week log-vol factor of the NEXT bar (calendar; weekday shape included)
    jx   sum_j 2^{-(t - t_j)/3} over past jump flags (jump excitation, half-life 3 h as H-S4 / H-V1b)
    m24  clip(sum_{24 bars} r / sqrt(24 vf), +-4)                 standardised 24 h momentum
    fz   (funding - mean_720h) / sd_720h, clip +-4                funding z (known at the print's known_ts)
    oi   clip(ln OI_t - ln OI_{t-24}, +-0.3)                      24 h open-interest change (4h prints)
    dir  regime v0 direction score (research.regimes), clip +-3
Offset: the field's diffusion is D(X) = vf_t * exp(poly(X)), so ln vf is an offset, not a coordinate.

Signed flow (delta): Bybit flow_bars 1h exist for BTC only from 2026-05-11 and the lake's Binance bars carry no
taker-buy volume (bars schema has volume/turnover only; Binance 1h/1m bars hold ETHUSDT only), so delta is NOT a core
coordinate; `flow_delta` returns it where it exists for an exploratory check. News intensity is not yet available (F3).

Per-feature known_ts: bar-derived coordinates are known at the bar's known_ts; fz carries fz_known_ts (the funding print
used), oi carries oi_known_ts (the OI print used); both are <= the row's known_ts by construction (backward as-of join on
known_ts, never on event_ts).

Aggregation: `aggregate_state(state, hours)` samples the hourly state at block closes (4h: close hour % 4 == 0; 1d:
00:00 UTC) and adds the block's log return and realised variance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.seasonality import HOW, hour_of_week, profile_asof
from ai_trader.storage.lake import Lake

SYMBOL = "BTCUSDT"
COORDS = ("x1", "x2", "L", "s", "jx", "m24", "fz", "oi", "dir")
ENDOGENOUS = ("x1", "x2", "L", "s", "jx", "m24")  # moved by `step` inside a simulated path
EXOGENOUS = ("fz", "oi", "dir")                     # frozen at the origin inside a simulated path


@dataclass(frozen=True)
class StateConfig:
    lam_fast: float = 0.94
    lam_med: float = 0.995
    lam_slow: float = 0.9993
    clip: float = 4.0
    jx_half_life: float = 3.0
    m24_bars: int = 24
    warmup: int = 720          # bars used to initialise the EWMAs (rows before are dropped)
    fund_window: int = 720     # hours for the funding z
    oi_lag: int = 24

    @property
    def jx_decay(self) -> float:
        return 2.0 ** (-1.0 / self.jx_half_life)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ data


def load_frame(as_of: datetime, *, lake: Lake | None = None, start: datetime | None = None) -> pl.DataFrame:
    """BTC 1h market frame (bars, ret, sigma_risk_next, dir_score) known at as_of, with point-in-time funding / OI."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label

    lake = lake or Lake()
    f = market_frame(SYMBOL, "1h", as_of, lake=lake, start=start, funding=False, oi=False)
    if f.height == 0:
        return f
    f = label(f, 1.0)
    keep = ["event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "ret", "sigma_risk_next", "dir_score", "cell"]
    f = f.select([c for c in keep if c in f.columns]).sort("event_ts")
    fu = lake.read_asof("funding", as_of, symbols=[SYMBOL], start=start, filters={"venue": "bybit"})
    f = _asof(f, fu, "rate", "funding_rate", "fz_known_ts")
    oi = lake.read_asof("open_interest", as_of, symbols=[SYMBOL], start=start, filters={"venue": "bybit", "interval": "4h"})
    f = _asof(f, oi, "open_interest", "open_interest", "oi_known_ts")
    return f.sort("event_ts")


def _asof(left: pl.DataFrame, right: pl.DataFrame, col: str, name: str, kname: str) -> pl.DataFrame:
    if right.height == 0 or col not in right.columns:
        return left.with_columns(pl.lit(None, dtype=pl.Float64).alias(name), pl.lit(None, dtype=left["known_ts"].dtype).alias(kname))
    r = right.select(pl.col("known_ts").alias(kname), pl.col(col).cast(pl.Float64).alias(name)).drop_nulls().sort(kname)
    r = r.with_columns(pl.col(kname).cast(left["known_ts"].dtype))
    return left.sort("known_ts").join_asof(r, left_on="known_ts", right_on=kname, strategy="backward", coalesce=False)


def flow_delta(as_of: datetime, *, lake: Lake | None = None) -> pl.DataFrame:
    """Signed taker flow per 1h bar where Bybit flow_bars exist: delta = (buy - sell) / (buy + sell), delta24 = 24 h
    volume-weighted imbalance. Known at the flow bar's known_ts. Empty before 2026-05-11 for BTC."""
    lake = lake or Lake()
    fl = lake.read_asof("flow_bars", as_of, symbols=[SYMBOL], filters={"venue": "bybit", "freq": "1h"})
    if fl.height == 0:
        return pl.DataFrame()
    fl = fl.sort("event_ts").with_columns(
        ((pl.col("buy_volume") - pl.col("sell_volume")) / (pl.col("buy_volume") + pl.col("sell_volume"))).alias("delta"),
        ((pl.col("buy_volume") - pl.col("sell_volume")).rolling_sum(24) / (pl.col("buy_volume") + pl.col("sell_volume")).rolling_sum(24)).alias("delta24"),
    )
    return fl.select("event_ts", pl.col("known_ts").alias("delta_known_ts"), "delta", "delta24")


def hour_profile(frame: pl.DataFrame, as_of: datetime) -> np.ndarray:
    """168-vector of hour-of-week variance factors known at as_of (H-NS4 construction on robust_bc residuals)."""
    sig = frame["sigma_risk_next"].shift(1).to_numpy()
    r = frame["ret"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        u = r * r / (sig * sig)
    return profile_asof(frame["event_ts"], u, as_of)


# ------------------------------------------------------------------ the recursion


def step(vf, vm, vs, jx, r, f, cfg: StateConfig):
    """One causal update with the return r of a bar whose hour-of-week factor is f. Works on scalars or arrays.
    Returns (vf, vm, vs, jx, flag)."""
    x = r / np.sqrt(f)
    lim = cfg.clip * np.sqrt(vf)
    flag = np.abs(x) > lim
    xc = np.clip(x, -lim, lim)
    x2 = xc * xc
    vf = cfg.lam_fast * vf + (1 - cfg.lam_fast) * x2
    vm = cfg.lam_med * vm + (1 - cfg.lam_med) * x2
    vs = cfg.lam_slow * vs + (1 - cfg.lam_slow) * x2
    jx = cfg.jx_decay * jx + flag
    return vf, vm, vs, jx, flag


def coords_from(vf, vm, vs, jx, sum24, f_next, cfg: StateConfig) -> dict[str, np.ndarray]:
    return {
        "x1": 0.5 * np.log(vf / vm),
        "x2": 0.5 * np.log(vm / vs),
        "L": 0.5 * np.log(vs),
        "s": 0.5 * np.log(f_next),
        "jx": np.asarray(jx, dtype=float),
        "m24": np.clip(sum24 / np.sqrt(cfg.m24_bars * vf), -4.0, 4.0),
    }


def build_state(frame: pl.DataFrame, profile: np.ndarray, cfg: StateConfig = StateConfig()) -> pl.DataFrame:
    """Adds vf/vm/vs (after bar t), flag, jx, f (this bar), f_next, and the COORDS. Rows of the EWMA warm-up dropped."""
    n = frame.height
    prof = np.asarray(profile, dtype=float)
    ts = frame["event_ts"]
    k = hour_of_week(ts)
    f_now = prof[k]
    f_next = prof[(k + 1) % HOW]
    r = np.nan_to_num(frame["ret"].to_numpy().astype(float))
    w = min(cfg.warmup, n)
    v0 = float(np.mean((r[1:w] / np.sqrt(f_now[1:w])) ** 2)) if w > 2 else 1e-4
    v0 = v0 if v0 > 0 else 1e-6
    vf = vm = vs = v0
    jx = 0.0
    out_vf, out_vm, out_vs = np.empty(n), np.empty(n), np.empty(n)
    out_jx, out_flag = np.empty(n), np.zeros(n, bool)
    lf, lm, ls, c, dec = cfg.lam_fast, cfg.lam_med, cfg.lam_slow, cfg.clip, cfg.jx_decay
    for t in range(n):  # scalar mirror of `step` (tested equal)
        x = r[t] / math.sqrt(f_now[t])
        lim = c * math.sqrt(vf)
        fl = abs(x) > lim
        xc = min(max(x, -lim), lim)
        x2 = xc * xc
        vf = lf * vf + (1 - lf) * x2
        vm = lm * vm + (1 - lm) * x2
        vs = ls * vs + (1 - ls) * x2
        jx = dec * jx + fl
        out_vf[t], out_vm[t], out_vs[t], out_jx[t], out_flag[t] = vf, vm, vs, jx, fl
    rs = pl.Series(r)
    sum24 = rs.rolling_sum(cfg.m24_bars, min_samples=cfg.m24_bars).fill_null(0.0).to_numpy()
    co = coords_from(out_vf, out_vm, out_vs, out_jx, sum24, f_next, cfg)
    fr = frame.get_column("funding_rate").to_numpy().astype(float) if "funding_rate" in frame.columns else np.full(n, np.nan)
    fs = pl.Series(fr).fill_nan(None)
    mu = fs.rolling_mean(cfg.fund_window, min_samples=cfg.fund_window // 4).to_numpy().astype(float)
    sd = fs.rolling_std(cfg.fund_window, min_samples=cfg.fund_window // 4).to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fz = np.clip((fr - mu) / np.where(sd > 1e-9, sd, np.nan), -4, 4)
    oi = frame.get_column("open_interest").to_numpy().astype(float) if "open_interest" in frame.columns else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        loi = np.log(np.where(oi > 0, oi, np.nan))
    d_oi = np.full(n, np.nan)
    d_oi[cfg.oi_lag:] = loi[cfg.oi_lag:] - loi[: -cfg.oi_lag]
    d_oi = np.clip(d_oi, -0.3, 0.3)
    dsc = frame.get_column("dir_score").to_numpy().astype(float) if "dir_score" in frame.columns else np.full(n, np.nan)
    dsc = np.clip(dsc, -3, 3)
    # unavailable exogenous inputs are set to their neutral value 0 and flagged (OI starts 2020-08; delta/news absent)
    avail = np.isfinite(fz) & np.isfinite(d_oi) & np.isfinite(dsc)
    out = frame.with_columns(
        pl.Series("f", f_now), pl.Series("f_next", f_next), pl.Series("how_next", (k + 1) % HOW),
        pl.Series("vf", out_vf), pl.Series("vm", out_vm), pl.Series("vs", out_vs),
        pl.Series("flag", out_flag), pl.Series("sum24", sum24),
        *[pl.Series(c_, co[c_]) for c_ in ENDOGENOUS],
        pl.Series("fz", np.nan_to_num(fz)), pl.Series("oi", np.nan_to_num(d_oi)), pl.Series("dir", np.nan_to_num(dsc)),
        pl.Series("exo_ok", avail),
        pl.col("ret").shift(-1).alias("r_next"),
    )
    if "fz_known_ts" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=out["known_ts"].dtype).alias("fz_known_ts"))
    if "oi_known_ts" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=out["known_ts"].dtype).alias("oi_known_ts"))
    return out.slice(w)


def last_returns(state: pl.DataFrame, idx: np.ndarray, n: int = 24) -> np.ndarray:
    """(len(idx), n) raw log returns of bars idx-n+1..idx (oldest first), for the m24 window inside a simulated path."""
    r = np.nan_to_num(state["ret"].to_numpy().astype(float))
    idx = np.asarray(idx)
    j = idx[:, None] + np.arange(-n + 1, 1)[None, :]
    return np.where(j >= 0, r[np.clip(j, 0, None)], 0.0)


def future_factors(state: pl.DataFrame, idx: np.ndarray, profile: np.ndarray, h: int) -> np.ndarray:
    """(len(idx), h + 1) hour-of-week factors of bars t+1..t+h+1 (pure calendar: known at t)."""
    k0 = state["how_next"].to_numpy()[np.asarray(idx)]
    return np.asarray(profile)[(k0[:, None] + np.arange(h + 1)[None, :]) % HOW]


def aggregate_state(state: pl.DataFrame, hours: int) -> pl.DataFrame:
    """Hourly state sampled at block closes of `hours` (4 or 24) with the NEXT block's log return and realised variance
    as labels (r_block_next, rv_block_next) and the closing block's own return / RV as features (r_block, rv_block)."""
    s = state.with_columns(
        pl.col("ret").rolling_sum(hours).alias("r_block"),
        (pl.col("ret") ** 2).rolling_sum(hours).alias("rv_block"),
    ).with_columns(
        pl.col("r_block").shift(-hours).alias("r_block_next"),
        pl.col("rv_block").shift(-hours).alias("rv_block_next"),
    )
    return s.filter(pl.col("close_ts").dt.hour() % hours == 0 if hours < 24 else pl.col("close_ts").dt.hour() == 0)


def state_asof(as_of: datetime, cfg: StateConfig = StateConfig(), *, lake: Lake | None = None, profile: np.ndarray | None = None
               ) -> tuple[pl.DataFrame, np.ndarray]:
    """Convenience: (state, profile) with every row known at as_of and the profile known at as_of."""
    fr = load_frame(as_of, lake=lake)
    prof = hour_profile(fr, as_of) if profile is None else profile
    st = build_state(fr, prof, cfg)
    return st.filter(pl.col("known_ts") <= as_of), prof


__all__ = ["COORDS", "ENDOGENOUS", "EXOGENOUS", "SYMBOL", "StateConfig", "aggregate_state", "build_state", "coords_from", "flow_delta",
           "future_factors", "hour_profile", "last_returns", "load_frame", "state_asof", "step", "timedelta"]
