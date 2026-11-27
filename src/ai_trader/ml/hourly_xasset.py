"""Hourly volatility features for TradFi perps and a cross-asset activity block (H-NN11, H-NN12; program 2026-11-12,
agent G). Extends ml.hourly (H-NN8): same grid, same per-symbol features, same target, same timing convention.

Timing (as ml.hourly): row = 1h bar t = [event_ts, event_ts + 1h) of one symbol; the forecast is made at its open, so
every input uses bars t-1 and earlier, or the calendar of bar t (known in advance).

Session flags (calendar of bar t, known in advance; ET = America/New_York with DST via zoneinfo)
  cash_sess  equities / ETFs: 1 if the NY date is an NYSE trading day and the bar starts at or after 09:00 ET and before
             the close (16:00 ET, 13:00 on early closes). The 09:00-10:00 ET bar contains the 09:30 open and is the first
             cash hour (tradfi_session_study convention). NYSE calendar: tradfi_session_study (2026, with early closes);
             other years fall back to features.seasonality.NYSE_HOLIDAYS with a 16:00 close.
  cme_sess   commodities (incl. the gold tokens PAXG / XAUT): CME Globex metals / energy hours, Sunday 18:00 ET to Friday
             17:00 ET with the daily 17:00-18:00 ET maintenance break Monday-Thursday. CME holiday schedules are not
             modelled (caveat). Flag = 1 if bar t starts inside those hours.
  For the other class the flag is 0.

Cross-asset block (XA_FEATURES; leaders BTC, ETH, QQQ). Per leader, on the leader's own regular hourly grid, exactly
as ml.hourly (a return is only used if the previous bar exists):
  x{tag}_lu1, _lu2, _lu3, _lu6   log(max(r_{L,t-k}^2 / s2_{L,t}, 1e-4)): the leader's |returns| at 1-6 h lags in the
                                 leader's own base units (s2_{L,t} = sigma_risk_next of the leader's bar t-1, squared)
  x{tag}_lum6                    log(mean_{k=1..6} r_{L,t-k}^2 / s2_{L,t}) (>= 5 finite)
  x{tag}_lsig2                   log s2_{L,t} (the leader's robust per-bar variance known at t)
  x{tag}_ok                      1 if the six values are finite for this row, else 0
Joined to targets on event_ts (bar t of the leader = bar t of the target, so the leader inputs end at the same close
as the target's own lag-1 input). The target's own copy is excluded: for target == leader the leader's columns are
null and x{tag}_ok = 0. Missing values are standardised to 0 (the training mean of the available rows, BlockStd).
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from ai_trader.ml.hourly import HOURLY_FEATURES, _shift, _trail_mean_lagged

LEADERS = {"BTCUSDT": "btc", "ETHUSDT": "eth", "QQQUSDT": "qqq"}
XA_SUFFIX = ("lu1", "lu2", "lu3", "lu6", "lum6", "lsig2", "ok")
XA_GROUPS = {f"x_{tag}": tuple(f"x{tag}_{s}" for s in XA_SUFFIX) for tag in LEADERS.values()}
XA_FEATURES = tuple(c for g in XA_GROUPS.values() for c in g)
assert len(XA_FEATURES) == 21
SESSION_FEATURES = ("cash_sess", "cme_sess")
T_FEATURES = (*HOURLY_FEATURES, *SESSION_FEATURES)

TRADFI_EQUITY = ("TSLAUSDT", "INTCUSDT", "GOOGLUSDT", "AAPLUSDT", "MSTRUSDT", "NVDAUSDT", "MUUSDT", "QQQUSDT", "SOXLUSDT", "SPCXUSDT")
TRADFI_COMMODITY = ("XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "PAXGUSDT", "XAUTUSDT")
GOLD_TOKENS = ("PAXGUSDT", "XAUTUSDT")
MIN_COMPLETE_DAYS = 120


def target_class(symbol: str) -> str:
    if symbol in TRADFI_EQUITY:
        return "equity"
    if symbol in GOLD_TOKENS:
        return "gold_token"
    if symbol in TRADFI_COMMODITY:
        return "commodity"
    if symbol in ("BTCUSDT", "ETHUSDT"):
        return symbol[:3].lower()
    return "crypto_alt"


def complete_days(f: pl.DataFrame) -> int:
    """Number of UTC dates with all 24 hourly bars."""
    d = f.select(pl.col("event_ts").unique()).with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d").len()
    return int((d["len"] == 24).sum())


# ------------------------------------------------------------------ session calendars


def _ny(ts: pl.Series) -> pl.DataFrame:
    s = ts.cast(pl.Datetime("us", "UTC")) if ts.dtype != pl.Datetime("us", "UTC") else ts
    ny = s.dt.convert_time_zone("America/New_York")
    return pl.DataFrame({"utc": s, "d": ny.dt.date(), "h": ny.dt.hour(), "wd": ny.dt.weekday()})


def _nyse_close_utc(d: date) -> datetime | None:
    """NYSE close (UTC) of NY date d, None if closed."""
    from datetime import UTC, time

    from ai_trader.features import seasonality as sz
    from ai_trader.research import tradfi_session_study as tf

    if d.year in tf.NYSE_HOLIDAYS:
        s = tf.nyse_session(d)
        return None if s is None else s[1]
    if not sz.is_nyse_day(d):
        return None
    return datetime.combine(d, time(16, 0), tf.NY).astimezone(UTC)


def cash_session_flag(ts: pl.Series) -> np.ndarray:
    """1.0 if the 1h bar starting at ts (UTC) is a NYSE cash-session hour (09:00 ET bar .. last bar before the close)."""
    n = _ny(ts)
    closes = {d: _nyse_close_utc(d) for d in n["d"].unique().to_list()}
    utc = n["utc"].to_list()
    out = np.zeros(n.height)
    for i, (d, h) in enumerate(zip(n["d"].to_list(), n["h"].to_list(), strict=True)):
        c = closes[d]
        if c is not None and h >= 9 and utc[i] < c:
            out[i] = 1.0
    return out


def cme_session_flag(ts: pl.Series) -> np.ndarray:
    """1.0 if the 1h bar starting at ts (UTC) is inside CME Globex metals/energy hours (Sun 18:00 ET - Fri 17:00 ET,
    daily break 17:00-18:00 ET Mon-Thu). Holidays are not modelled."""
    n = _ny(ts)
    h, wd = n["h"].to_numpy(), n["wd"].to_numpy()  # weekday: Mon 1 .. Sun 7
    on = ((wd == 7) & (h >= 18)) | ((wd >= 1) & (wd <= 4) & (h != 17)) | ((wd == 5) & (h < 17))
    return on.astype(float)


def session_flags(ts: pl.Series, symbol: str) -> dict[str, np.ndarray]:
    z = np.zeros(len(ts))
    cls = target_class(symbol)
    return {"cash_sess": cash_session_flag(ts) if cls == "equity" else z,
            "cme_sess": cme_session_flag(ts) if cls in ("commodity", "gold_token") else z.copy()}


# ------------------------------------------------------------------ cross-asset block


def _grid_r_s2(f: pl.DataFrame) -> tuple[pl.Series, np.ndarray, np.ndarray]:
    """Regular hourly grid of one symbol (as ml.symbol_hourly): event_ts, gap-safe r, s2 = sigma_risk_next[t-1]^2."""
    from ai_trader.risk.sigma import pin_base

    f = pin_base(f).sort("event_ts").unique("event_ts", keep="last", maintain_order=True).select("event_ts", "ret", "sigma_risk_next")
    f = f.with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    grid = pl.DataFrame({"event_ts": pl.datetime_range(f["event_ts"].min(), f["event_ts"].max(), "1h", eager=True)}).with_columns(
        pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    g = grid.join(f, on="event_ts", how="left")
    prev_ok = g["ret"].is_not_null().to_numpy() & np.r_[False, g["sigma_risk_next"].is_not_null().to_numpy()[:-1]]
    r = np.where(prev_ok, g["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(), np.nan)
    s_pre = _shift(g["sigma_risk_next"].cast(pl.Float64).fill_null(np.nan).to_numpy(), 1)
    return g["event_ts"], r, s_pre * s_pre


def leader_block(f: pl.DataFrame, tag: str) -> pl.DataFrame:
    """The leader's block on its own grid: event_ts + x{tag}_* (all from leader bars t-1 and earlier)."""
    ts, r, s2 = _grid_r_s2(f)
    r2 = r * r
    cols: dict[str, np.ndarray] = {}
    with np.errstate(all="ignore"):
        for k in (1, 2, 3, 6):
            cols[f"x{tag}_lu{k}"] = np.log(np.maximum(_shift(r2, k) / s2, 1e-4))
        cols[f"x{tag}_lum6"] = np.log(_trail_mean_lagged(r2, 6, 5) / s2)
        cols[f"x{tag}_lsig2"] = np.log(s2)
    for k, v in cols.items():
        cols[k] = np.where(np.isfinite(v), v, np.nan)
    ok = np.all(np.isfinite(np.column_stack(list(cols.values()))), axis=1)
    out = pl.DataFrame({"event_ts": ts, **cols}).with_columns(pl.col(pl.Float64).fill_nan(None))
    return out.with_columns(pl.Series(f"x{tag}_ok", ok.astype(float)))


def attach_leaders(panel: pl.DataFrame, blocks: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Left-join every leader block on event_ts; null + ok = 0 where the leader has no row, and for the leader's own
    rows (the target's own series is never a cross-asset input)."""
    p = panel.with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    for sym, tag in LEADERS.items():
        b = blocks[sym]
        vals = [f"x{tag}_{s}" for s in XA_SUFFIX if s != "ok"]
        p = p.join(b, on="event_ts", how="left")
        own = pl.col("symbol") == sym
        p = p.with_columns(*[pl.when(own).then(None).otherwise(pl.col(c)).alias(c) for c in vals],
                           pl.when(own | pl.col(f"x{tag}_ok").is_null()).then(0.0).otherwise(pl.col(f"x{tag}_ok")).alias(f"x{tag}_ok"))
    return p


class BlockStd:
    """Standardisation of a block with missing values: training mean / sd over the finite training values; missing
    entries become 0 (= the training mean). A column with no finite training value is all 0."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> BlockStd:
        X = np.asarray(X, dtype=float)
        fin = np.isfinite(X)
        n = fin.sum(0)
        with np.errstate(all="ignore"):
            mu = np.where(n > 0, np.nansum(np.where(fin, X, 0.0), 0) / np.maximum(n, 1), 0.0)
            var = np.where(n > 1, np.nansum(np.where(fin, (X - mu) ** 2, 0.0), 0) / np.maximum(n, 1), 0.0)
        sd = np.sqrt(var)
        self.mu, self.sd = mu, np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sd is not None, "fit first"
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0)


__all__ = ["BlockStd", "GOLD_TOKENS", "LEADERS", "MIN_COMPLETE_DAYS", "SESSION_FEATURES", "TRADFI_COMMODITY", "TRADFI_EQUITY", "T_FEATURES",
           "XA_FEATURES", "XA_GROUPS", "XA_SUFFIX", "attach_leaders", "cash_session_flag", "cme_session_flag", "complete_days", "leader_block",
           "session_flags", "target_class"]
