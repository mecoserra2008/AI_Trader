"""Risk sigma: ONE entry point for the per-bar sigma behind stops, sizing and the EV gate (decision D1, user-approved
2026-11-27 ~21:35 UTC: replace robust_bc at 1h / 4h, keep log-HAR at 1d, no weekend shrink in the stop sigma; shadow
first, then switch).

    from ai_trader.risk.sigma import sigma_risk, spec_in_force
    s = sigma_risk("BTCUSDT", "1h", as_of)                    # the spec in force at as_of (settings.yaml risk_sigma)
    s = sigma_risk("BTCUSDT", "4h", as_of, spec="loghar_cr_range")
    s.sigma, s.nu, s.bar_start, s.spec, s.fallback

Every market frame gets its stop sigma here: research.frame.add_robust_vol calls stop_sigma_column(), so every consumer
of the frame's sigma_risk_next (agent.session packet geometry, execution.gate placement / intents' sigma.per_bar,
research.stress, research.trend_spec, state_now, ...) follows the spec in force without code changes.

Specs (the horizon is the bar being forecast; any other frequency, e.g. 15m, always uses robust_bc):
  robust_bc        the incumbent: features.vol_models robust EWMA, bias-corrected, use_weekend=False (H-V1b), run on
                   bars of the horizon = the frame's sigma_rbc_next.
  loghar_cr        WS04 pick at 1h / 4h (H-WS04-1/3/4, R-WS04-1): log-HAR on 5m realised variance deflated by the
                   hour-of-week (1h) / 4h-block-of-week (4h) clock, plus a dummy for a scheduled CPI / NFP / FOMC inside
                   the bar (schedule known at the bar open); HAR lags 1/4/24/168/720 h (1h) and 1/6/42/180 bars (4h);
                   refit on the first of every month on up to 3 years of bars before it; Duan smearing. The arithmetic is
                   ai_trader.research.vol_spec.forecast_grid's loghar_cr (same helpers; tested equal on the WS04 grids).
                   1d: vol.har (H-VCL1, plain log-HAR on 1h-derived daily RV; WS04: in the 1d MCS, kept).
  loghar_cr_range  loghar_cr + the WS12 high-low range terms (R-WS12-1, H-WS12-8): log of the clock-deflated Parkinson
                   range variance ln(H/L)^2 / (4 ln 2) averaged over the last 1 / 6 / 24 bars (1h) and 1 / 6 bars (4h,
                   i.e. 4 h and 24 h), floored at 1% of its training median. 1d: vol.har.
                   NEW COMBINATION: WS04 scored loghar_cr and WS12 scored range terms on a robust_bc-based log-HAR-t; this
                   joint spec is scored forward by the shadow (H-SIGMA-1/2) before it is used.
Stop sigma vs central: R-VOL-2 / H-WS04-5 (the hour-of-week shrink under-covers the weekend lower tail, 1.44% at 1%).
  For a forecast bar touching the TradFi weekend (research.calendar) the clock factor is floored at the Mon-Thu mean
  factor of the same UTC hour (4h block): the weekend never shrinks the stop sigma. The unfloored value is `central`.
Innovations: Student-t with nu per horizon (WS04 R-WS04-2: 1h 3.80, 4h 3.53, 1d 3.57; settings risk_sigma.nu).
Fallback: a NaN new-spec forecast (no 5m store, stale store, short history) falls back to robust_bc; the result says so.
Switch (shadow first): settings.yaml risk_sigma {spec, shadow, switch_on, switch_gate}.
  spec_in_force(as_of) = `spec` before switch_on. From switch_on: `shadow` iff switch_gate is 'none', or the gate decision
  for this switch_on (scripts/sigma_shadow_score.py --gate appends it to data/journal/sigma_shadow/gate.jsonl once, the
  registered rule of H-SIGMA-1) passed and was decided at or before as_of; else `spec`. Flip by hand: set spec.
Roles: sigma_risk_next = this module's STOP sigma. sigma_rbc_next = robust_bc always: the base the frozen networks
  were trained on and the calibrated features use. They never switch:
  - pin_base(frame) in tools.vol_ml (H-NN10), ml.hourly_xasset, field.state, features.regime_features, tools.market_state,
    tools.vol (vol.robust_bc) and execution.gate's vol percentile;
  - pin_process_base() in scripts/live_nn.py (live_nn v1 / v2, xs_v3 context): those modules are hash-guarded;
  - xs_v4 and the research grids build frames without a symbol, which always gives robust_bc.
  ml.hourly.symbol_hourly itself is NOT pinned (xs_v3 / xs_v4 hash it): callers pass pin_base(frame).
Point in time: store rows with known_ts <= as_of; release schedules known at the bar open; gate decisions decided_at <=
  as_of; the monthly fit uses rows known at the month start only.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import numpy as np
import polars as pl

from ai_trader.core.config import load

SPECS = ("robust_bc", "loghar_cr", "loghar_cr_range")
HORIZONS = {"1h": 1, "4h": 4, "1d": 24}
LAGS = {"1h": (1, 4, 24, 168, 720), "4h": (1, 6, 42, 180)}
RANGE_LAGS = {"1h": (1, 6, 24), "4h": (1, 6)}
CELLS = {"1h": 168, "4h": 42}
NU = {"1h": 3.80, "4h": 3.53, "1d": 3.57}
CAP_YEARS = 3
MIN_HISTORY_DAYS = 365          # first refit needs a year of 5m history (WS04: data from 2022-01, first refit 2023-01)
BASE_COL = "sigma_rbc_next"
VERSION = "risk.sigma.v1"
UNIVERSE_L = ("BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
              "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT").split()
DEFAULTS = {"spec": "robust_bc", "shadow": "loghar_cr_range", "switch_on": None, "switch_gate": "score", "nu": NU,
            "symbols": UNIVERSE_L}

_FORCE_RBC: contextvars.ContextVar[bool] = contextvars.ContextVar("risk_sigma_force_rbc", default=False)
_PROCESS_BASE = False


def pin_process_base(on: bool = True) -> None:
    """Keep sigma_risk_next = robust_bc in every frame of this process, whatever spec is in force. For processes that
    only run frozen networks on market frames (scripts/live_nn.py: live_nn v1 / v2 and the xs_v3 context line). Their
    modules (live/nn.py, live/nn_v2.py, live/xs_v3.py, ml/hourly.py) are code-hash guarded or logged, so they are not
    edited: the pin lives in the launching script. They never place stops."""
    global _PROCESS_BASE
    _PROCESS_BASE = bool(on)


# ====================================================================== config / switch


def settings() -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(load().settings.get("risk_sigma") or {})
    cfg["nu"] = {**NU, **(cfg.get("nu") or {})}
    for k in ("spec", "shadow"):
        if cfg.get(k) is not None and cfg[k] not in SPECS:
            raise ValueError(f"risk_sigma.{k} = {cfg[k]!r}: not one of {SPECS}")
    if cfg.get("switch_gate") not in ("score", "none"):
        raise ValueError(f"risk_sigma.switch_gate = {cfg.get('switch_gate')!r}: 'score' or 'none'")
    return cfg


def _utc(t: datetime | str | date | None) -> datetime | None:
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    if isinstance(t, date) and not isinstance(t, datetime):
        return datetime(t.year, t.month, t.day, tzinfo=UTC)
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def gate_path():
    p = load().root / "data" / "journal" / "sigma_shadow"
    p.mkdir(parents=True, exist_ok=True)
    return p / "gate.jsonl"


@lru_cache(maxsize=8)
def _gate_lines(p: str, mtime_ns: int) -> tuple[dict, ...]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return tuple(out)


def gate_decision(switch_on: str, as_of: datetime) -> dict | None:
    """The first gate decision for this switch_on, if it was decided at or before as_of (latched: later lines for the
    same switch_on are ignored)."""
    p = gate_path()
    if not p.is_file():
        return None
    lines = [d for d in _gate_lines(str(p), p.stat().st_mtime_ns) if str(d.get("switch_on")) == str(switch_on) and d.get("decided_at")]
    if not lines:
        return None
    first = min(lines, key=lambda d: _utc(d["decided_at"]))
    return first if _utc(first["decided_at"]) <= _utc(as_of) else None


def spec_in_force(as_of: datetime | None, cfg: dict | None = None) -> str:
    """The stop-sigma spec in force at as_of (module docstring, 'Switch')."""
    cfg = cfg or settings()
    spec, shadow, sw = cfg["spec"], cfg.get("shadow"), cfg.get("switch_on")
    if as_of is None or not shadow or not sw or shadow == spec:
        return spec
    if _utc(as_of) < _utc(sw):
        return spec
    if cfg["switch_gate"] == "none":
        return shadow
    d = gate_decision(str(sw), _utc(as_of))
    return shadow if d and d.get("pass") is True and d.get("shadow") == shadow else spec


# ====================================================================== roles


def pin_base(f: pl.DataFrame) -> pl.DataFrame:
    """The frame with sigma_risk_next replaced by the robust_bc base (sigma_rbc_next) when present: for trained models
    and calibrated features that must not follow a stop-sigma switch. A frame without the column is returned as is."""
    if BASE_COL in f.columns:
        return f.with_columns(pl.col(BASE_COL).alias("sigma_risk_next"))
    return f


def base_values(f: pl.DataFrame) -> np.ndarray:
    return f[BASE_COL if BASE_COL in f.columns else "sigma_risk_next"].cast(pl.Float64).fill_null(np.nan).to_numpy()


# ====================================================================== grids


def _ts64(s: pl.Series) -> np.ndarray:
    return s.cast(pl.Datetime("us", "UTC")).dt.epoch("us").to_numpy()


def grid(h1: pl.DataFrame, g: str) -> pl.DataFrame:
    """1h / 4h / 1d grid (event_ts, rv, pk = Parkinson range variance, close) from store hours. A 4h (1d) bar needs >= 4 (22)
    hours with rv (vol_spec.aggregate's 90% rule, rescaled); its range is from the hours' high / low."""
    from ai_trader.risk.sigma_store import parkinson

    if g == "1h":
        return h1.select("event_ts", "rv", parkinson(pl.col("high"), pl.col("low")).alias("pk"), "close", "known_ts")
    hrs = HORIZONS[g]
    a = (h1.group_by_dynamic("event_ts", every=g, closed="left", label="left")
         .agg(pl.col("rv").sum(), pl.col("rv").is_not_null().sum().alias("nh"), pl.col("high").max(), pl.col("low").min(),
              pl.col("close").drop_nulls().last(), pl.col("known_ts").max(), pl.len().alias("n")))
    ok = (pl.col("nh") >= int(math.ceil(0.9 * hrs))) & (pl.col("n") == hrs)
    return a.with_columns(
        pl.when(ok).then(pl.col("rv") * hrs / pl.col("nh")).otherwise(None).alias("rv"),
        pl.when(ok).then(parkinson(pl.col("high"), pl.col("low"))).otherwise(None).alias("pk"),
        pl.when(ok).then(pl.col("close")).otherwise(None).alias("close"),
    ).select("event_ts", "rv", "pk", "close", "known_ts")


def weekend_bars(ts: pl.Series, g: str) -> np.ndarray:
    """True if any hour of the bar [T, T + h) is inside the TradFi weekend (research.calendar.tradfi_weekend)."""
    from ai_trader.research.calendar import add_calendar

    hrs = HORIZONS[g]
    t = ts.cast(pl.Datetime("us", "UTC"))
    wk = np.zeros(len(t), dtype=bool)
    for k in range(hrs):
        wk |= add_calendar(pl.DataFrame({"event_ts": t + timedelta(hours=k)}))["tradfi_weekend"].to_numpy().astype(bool)
    return wk


def weekday_reference(f_cell: np.ndarray, g: str) -> np.ndarray:
    """Per cell: the mean clock factor of the same UTC hour (1h) / 4h block (4h) over Monday-Thursday."""
    per_day = CELLS[g] // 7
    m = f_cell.reshape(7, per_day)[:4].mean(axis=0)
    return np.tile(m, 7)


def _release(ts: pl.Series, g: str) -> tuple[np.ndarray, str | None]:
    from ai_trader.research.vol_spec import release_flags

    try:
        return release_flags(ts, HORIZONS[g]).astype(float), None
    except Exception as e:  # noqa: BLE001 - missing events table: no dummy, flagged
        return np.zeros(len(ts)), f"release schedule unavailable ({type(e).__name__})"


# ====================================================================== the log-HAR forecaster


def forecast(G: pl.DataFrame, g: str, spec: str, start: datetime | None = None) -> dict:
    """Forecasts for every bar of grid G (event_ts, rv, pk) at 1h / 4h under loghar_cr or loghar_cr_range, made at the bar
    open from bars before it; refits on each month start from `start` (default: MIN_HISTORY_DAYS after the first bar).
    Returns {'stop', 'central', 'clock'} arrays (NaN where no fit) and 'flags'."""
    from ai_trader.research.vol_spec import cells, clock_factor, ols, refit_points, trailing_mean

    if spec not in ("loghar_cr", "loghar_cr_range") or g not in LAGS:
        raise ValueError(f"forecast: {spec} at {g}")
    n = G.height
    out = {"stop": np.full(n, np.nan), "central": np.full(n, np.nan), "clock": np.full(n, np.nan), "flags": []}
    if n == 0:
        return out
    ts = G["event_ts"].cast(pl.Datetime("us", "UTC"))
    y = G["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    pk = G["pk"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    L = LAGS[g]
    cap = CAP_YEARS * 365 * 24 // HORIZONS[g]
    if start is None:
        start = first_refit(ts.min())
    refits = refit_points(ts.to_numpy(), start)
    if not refits:
        return out
    z_slow = trailing_mean(y, L[-1])
    cell = cells(ts, g)
    rel, fl = _release(ts, g)
    if fl:
        out["flags"].append(fl)
    wk = weekend_bars(ts, g)
    for j, R in enumerate(refits):
        Rn = refits[j + 1] if j + 1 < len(refits) else n
        lo = max(0, R - cap)
        train = np.zeros(n, dtype=bool)
        train[lo:R] = True
        yt = y[lo:R]
        fin_y = np.isfinite(yt)
        if fin_y.sum() < 50:
            continue
        floor = 0.01 * float(np.nanmedian(yt))
        f_cell = clock_factor(y, z_slow, cell, CELLS[g], train, smooth=(g == "1h"))
        fc = f_cell[cell]
        yd = y / fc
        comps = [np.log(np.maximum(trailing_mean(yd, w), floor)) for w in L]
        cols = [np.ones(n)] + comps + [rel]
        if spec == "loghar_cr_range":
            pt = pk[lo:R]
            if np.isfinite(pt).sum() < 50:
                continue
            pfloor = 0.01 * float(np.nanmedian(pt))
            pd_ = pk / fc
            cols += [np.log(np.maximum(trailing_mean(pd_, k), pfloor)) for k in RANGE_LAGS[g]]
        Xd = np.column_stack(cols)
        lyd = np.log(np.maximum(yd, floor))
        ok = np.isfinite(Xd[lo:R]).all(axis=1) & fin_y
        if ok.sum() < 30:
            continue
        b = ols(Xd[lo:R][ok], lyd[lo:R][ok])
        smear = float(np.mean(np.exp(lyd[lo:R][ok] - Xd[lo:R][ok] @ b)))
        sl = slice(R, Rn)
        base = np.exp(Xd[sl] @ b) * smear
        out["central"][sl] = fc[sl] * base
        ref = weekday_reference(f_cell, g)[cell[sl]]
        out["stop"][sl] = np.where(wk[sl], np.maximum(fc[sl], ref), fc[sl]) * base
        out["clock"][sl] = fc[sl]
        if j == len(refits) - 1:
            out["fit"] = {"refit_ts": str(ts[R]), "n_train": int(ok.sum()), "coef": [float(x) for x in b], "smear": smear,
                          "floor": floor, "clock_sha": hashlib.sha256(np.round(f_cell, 12).tobytes()).hexdigest()[:12]}
    return out


def first_refit(first_ts: datetime) -> datetime:
    """The first month start at least MIN_HISTORY_DAYS after the first bar of a symbol's 5m history."""
    t0 = _utc(first_ts) + timedelta(days=MIN_HISTORY_DAYS)
    m = datetime(t0.year, t0.month, 1, tzinfo=UTC)
    if m < t0:
        m = datetime(t0.year + (t0.month // 12), t0.month % 12 + 1, 1, tzinfo=UTC)
    return m


def _with_placeholder(G: pl.DataFrame, bar_start: datetime, g: str) -> pl.DataFrame:
    """G truncated to bars before bar_start, on a complete calendar up to bar_start, plus the (unknown) bar itself."""
    step = timedelta(hours=HORIZONS[g])
    b = pl.lit(bar_start).cast(pl.Datetime("us", "UTC"))
    G = G.filter(pl.col("event_ts") < b)
    t0 = G["event_ts"].min() if G.height else bar_start
    cal = pl.DataFrame({"event_ts": pl.datetime_range(t0, bar_start, step, time_zone="UTC", eager=True)}).with_columns(
        pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    return cal.join(G, on="event_ts", how="left").sort("event_ts")


@lru_cache(maxsize=48)
def _series_cached(symbol: str, g: str, spec: str, as_of_us: int, store_mtime: int) -> pl.DataFrame:
    from ai_trader.risk import sigma_store

    as_of = datetime.fromtimestamp(as_of_us / 1e6, tz=UTC)
    h1 = sigma_store.load(symbol, as_of)
    if h1.height == 0:
        return pl.DataFrame(schema={"event_ts": pl.Datetime("us", "UTC"), "stop": pl.Float64, "central": pl.Float64})
    G = grid(h1, g)
    step = timedelta(hours=HORIZONS[g])
    last = G["event_ts"].max()
    G = _with_placeholder(G, last + step, g)       # the bar after the newest complete one
    fc = forecast(G, g, spec)
    return pl.DataFrame({"event_ts": G["event_ts"], "stop": fc["stop"], "central": fc["central"]})


def _store_mtime(symbol: str) -> int:
    from ai_trader.risk import sigma_store

    p = sigma_store.path(symbol)
    return p.stat().st_mtime_ns if p.is_file() else 0


# ====================================================================== frame route (all bars)


def stop_sigma_column(df: pl.DataFrame, *, symbol: str | None, freq: str | None, as_of: datetime | None,
                      spec: str | None = None, lake=None) -> tuple[np.ndarray, str]:
    """sigma_risk_next for every row of an add_robust_vol frame (row t -> the bar opening at close_ts[t]) under the spec
    in force at as_of (default: the frame's newest known_ts). Returns (values, spec used). robust_bc returns the frame's
    sigma_rbc_next unchanged. New-spec rows without a forecast fall back to robust_bc."""
    rbc = df[BASE_COL].cast(pl.Float64).fill_null(np.nan).to_numpy()
    if as_of is None and "known_ts" in df.columns and df.height:
        as_of = _utc(df["known_ts"].max())
    spec = spec or ("robust_bc" if (_FORCE_RBC.get() or _PROCESS_BASE) else spec_in_force(as_of))
    if spec == "robust_bc" or symbol is None or freq not in HORIZONS or df.height == 0:
        return rbc, "robust_bc"
    try:
        if freq == "1d":
            fc = _har_daily_series(symbol, _utc(as_of), lake)     # the 5m store (1h / 4h) is its own source
        else:
            fc = _series_cached(symbol, freq, spec, int(_utc(as_of).timestamp() * 1e6), _store_mtime(symbol)).select(
                "event_ts", pl.col("stop").alias("_s"))
    except Exception:  # noqa: BLE001 - any failure of the new spec leaves the incumbent in place
        return rbc, "robust_bc"
    if fc.height == 0:
        return rbc, "robust_bc"
    key = df.select(pl.col("close_ts").cast(pl.Datetime("us", "UTC")).alias("event_ts"))
    s = key.join(fc.with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC"))), on="event_ts", how="left")["_s"]
    v = np.sqrt(s.cast(pl.Float64).fill_null(np.nan).to_numpy())
    out = np.where(np.isfinite(v) & (v > 0), v, rbc)
    return out, spec


def _frame_rbc(symbol: str, freq: str, as_of: datetime, lake=None) -> pl.DataFrame:
    """market_frame (no funding / OI) with the stop column forced to robust_bc (no recursion into the new specs).
    With the default lake it is cached per (symbol, freq, as_of) for the current wall-clock minute (one shadow run reads
    each frame once; a lake write after the minute is seen)."""
    if lake is None:
        return _frame_rbc_cached(symbol, freq, int(_utc(as_of).timestamp() * 1e6), int(time.time() // 60))
    return _frame_rbc_uncached(symbol, freq, _utc(as_of), lake)


@lru_cache(maxsize=6)
def _frame_rbc_cached(symbol: str, freq: str, as_of_us: int, minute: int) -> pl.DataFrame:
    return _frame_rbc_uncached(symbol, freq, datetime.fromtimestamp(as_of_us / 1e6, tz=UTC), None)


def _frame_rbc_uncached(symbol: str, freq: str, as_of: datetime, lake) -> pl.DataFrame:
    from ai_trader.research.frame import market_frame

    tok = _FORCE_RBC.set(True)
    try:
        return market_frame(symbol, freq, as_of, lake=lake, funding=False, oi=False)
    finally:
        _FORCE_RBC.reset(tok)


def _har_daily_series(symbol: str, as_of: datetime, lake=None) -> pl.DataFrame:
    """vol.har's next-UTC-day variance for every day (the tool's arithmetic on the same 1h frame): event_ts = the
    forecast day's 00:00 (the 1d bar it is for)."""
    from ai_trader.features.vol_models import har_forecast

    f = _frame_rbc(symbol, "1h", as_of, lake)
    daily = (f.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
             .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n")).filter(pl.col("n") == 24).sort("d"))
    if daily.height == 0:
        return pl.DataFrame(schema={"event_ts": pl.Datetime("us", "UTC"), "_s": pl.Float64})
    fc = har_forecast(daily["rv"].to_numpy())
    nxt = daily.select((pl.col("d").cast(pl.Datetime("us")) + pl.duration(days=1)).dt.replace_time_zone("UTC").alias("event_ts"))
    return nxt.with_columns(pl.Series("_s", fc))


# ====================================================================== point route (one sigma)


class _Fallback(Exception):
    pass


@dataclass(frozen=True)
class RiskSigma:
    symbol: str
    horizon: str
    spec: str                  # the spec that produced `sigma` (after any fallback)
    requested: str             # the spec asked for (or in force)
    sigma: float               # sd of the log return of the forecast bar (stop sigma: no weekend shrink)
    central: float | None      # unfloored clock forecast (weekend-shrunk), None for robust_bc / vol.har
    nu: float                  # Student-t degrees of freedom for this horizon
    bar_start: datetime | None  # open of the forecast bar
    as_of: datetime
    data_max_ts: datetime | None
    fallback: str | None = None
    fit: dict | None = None
    version: str = VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("bar_start", "as_of", "data_max_ts"):
            d[k] = d[k].isoformat() if d[k] is not None else None
        return d


def _floor_bar(t: datetime, g: str) -> datetime:
    t = _utc(t)
    if g == "1d":
        return t.replace(hour=0, minute=0, second=0, microsecond=0)
    h = HORIZONS[g]
    return t.replace(hour=(t.hour // h) * h, minute=0, second=0, microsecond=0)


def _rbc_point(symbol: str, horizon: str, as_of: datetime, lake=None, requested: str = "robust_bc", why: str | None = None) -> RiskSigma:
    f = _frame_rbc(symbol, horizon, as_of, lake)
    nu = settings()["nu"].get(horizon, NU.get(horizon, float("nan")))
    if f.height == 0:
        return RiskSigma(symbol, horizon, "robust_bc", requested, float("nan"), None, nu, None, as_of, None, why or "no bars")
    last = f.tail(1).row(0, named=True)
    v = last[BASE_COL]
    return RiskSigma(symbol, horizon, "robust_bc", requested, float(v) if v is not None else float("nan"), None, nu,
                     _utc(last["close_ts"]), as_of, _utc(f["known_ts"].max()), why)


def sigma_risk(symbol: str, horizon: str, as_of: datetime, spec: str | None = None, *, lake=None, fallback: bool = True) -> RiskSigma:
    """The stop sigma of the next `horizon` bar (the one open at as_of, forecast from the bars closed before it)."""
    as_of = _utc(as_of)
    requested = spec or spec_in_force(as_of)
    if requested not in SPECS:
        raise ValueError(f"unknown risk sigma spec {requested!r}")
    nu = settings()["nu"].get(horizon, NU.get(horizon, float("nan")))
    if requested == "robust_bc" or horizon not in HORIZONS:
        return _rbc_point(symbol, horizon, as_of, lake, requested)
    bar = _floor_bar(as_of, horizon)
    why = None
    try:
        if horizon == "1d":
            # vol.har (tools.vol.har) arithmetic on the same 1h frame, for exactly the UTC day open at as_of (NaN when
            # the previous day is incomplete in the lake, instead of silently forecasting an older day)
            fc = _har_daily_series(symbol, as_of, lake)
            row = fc.filter(pl.col("event_ts") == pl.lit(bar).cast(pl.Datetime("us", "UTC")))
            v = float(row["_s"][0]) if row.height and row["_s"][0] is not None else float("nan")
            if np.isfinite(v) and v > 0:
                return RiskSigma(symbol, horizon, requested, requested, math.sqrt(v), None, nu, bar, as_of, None, None,
                                 {"tool": "vol.har", "n_days": int(fc.height)})
            why = "no vol.har forecast for this day (previous UTC day incomplete)"
            raise _Fallback(why)
        from ai_trader.risk import sigma_store

        h1 = sigma_store.load(symbol, as_of)
        if h1.height == 0:
            why = "no 5m store"
        else:
            m0 = datetime(bar.year, bar.month, 1, tzinfo=UTC)
            if m0 < first_refit(h1["event_ts"].min()):
                why = "5m history shorter than a year"
            else:
                # same arithmetic as the frame route for the month of `bar`: a refit only sees cap + the longest HAR
                # window of bars before the month start, so older rows are dropped (tested equal to the full series)
                G = _with_placeholder(grid(h1, horizon), bar, horizon)
                keep = (CAP_YEARS * 365 * 24 // HORIZONS[horizon]) + max(LAGS[horizon]) + 8
                G = G.filter(pl.col("event_ts") >= pl.lit(m0 - timedelta(hours=HORIZONS[horizon] * keep)).cast(pl.Datetime("us", "UTC")))
                fc = forecast(G, horizon, requested, start=m0)
                v, c = float(fc["stop"][-1]), float(fc["central"][-1])
                if np.isfinite(v) and v > 0:
                    return RiskSigma(symbol, horizon, requested, requested, math.sqrt(v), math.sqrt(c), nu, bar, as_of,
                                     _utc(h1["known_ts"].max()), None, {**fc.get("fit", {}), "flags": fc["flags"]})
                why = "no forecast (stale or missing 5m hours)"
    except _Fallback as e:
        why = str(e)
    except Exception as e:  # noqa: BLE001
        why = f"error: {type(e).__name__}: {str(e)[:200]}"
    if not fallback:
        return RiskSigma(symbol, horizon, requested, requested, float("nan"), None, nu, bar, as_of, None, why)
    return _rbc_point(symbol, horizon, as_of, lake, requested, why)
