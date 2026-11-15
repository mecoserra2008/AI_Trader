"""Calendar seasonality for 24/7 perps: an hour-of-week volatility profile, crypto session labels, the Deribit expiry
calendar, Bybit funding-settlement mapping and the DST-aware NYSE opening bar (program 2026-11-11, agent B3).

Hour-of-week volatility profile (H-NS4). For bar t with log return r_t and a base variance forecast F0_t = sigma_t^2
made before the bar (e.g. the frame's sigma_risk_next of bar t-1), the standardised square u_t = r_t^2 / F0_t measures
how loud bar t was relative to the base model. With k = hour of week (Monday 00:00 UTC = 0 ... Sunday 23:00 = 167):

    m_k   = mean of min(u, cap) over past bars of hour k        (expanding, or the last `window_weeks` occurrences)
    m~_k  = (n_k m_k + kappa m_bar) / (n_k + kappa)             shrink toward the grand mean of the same window
    s_k   = sum_j w_j m~_{k+j mod 168}                          circular triangular kernel, half-width `bandwidth` hours
    f_k   = max(s_k / mean_k(s), floor)                         shape only: the factor averages 1 over the week

The seasonal forecast is F1_t = F0_t * f_{k_t}. The factor for bar t uses u_j for j < t only (u_t is known at the close
of t, when the forecast for t+1 is made), so `seasonal_factor(...)[t]` is known at the open of bar t. Normalising to a
weekly mean of 1 makes the profile a pure reshaping of the base model: the base model's overall level (and the uniform
part of the cap's downward bias) is left alone. QLIKE-optimal forecasts are conditional means, hence a mean of u, not a
median; the cap (default 50, a ~7 sigma bar) and the shrinkage keep single jumps from defining an hour.

    from ai_trader.features.seasonality import hour_of_week, seasonal_factor, profile_asof
    k = hour_of_week(frame["event_ts"])
    f = seasonal_factor(k, frame["ret"] ** 2 / frame["sigma_risk_next"].shift(1) ** 2)   # known at each bar's open
    prof = profile_asof(...)  # the 168-vector known at a given time, for quoting widths and stops

Other helpers (pure functions of timestamps; no data):
  crypto_session(hour)      Asia 00-08, Europe 08-13, US 13-20, late 20-24 UTC (H-NS3)
  deribit_expiry_kind(d)    weekly / monthly / quarterly for a Friday (expiry Friday 08:00 UTC; monthly = last Friday of
                            the month, quarterly = last Friday of Mar / Jun / Sep / Dec), None otherwise (H-NS2)
  expiry_calendar(a, b)     the expiry timestamps in [a, b)
  funding_settlements(df)   per-symbol settlement times with the interval since the previous print (H-NS1)
  nyse_open_bar(dates)      UTC open time of the 1h bar containing 09:30 America/New_York, NYSE holidays excluded (H-NS5)
"""

from __future__ import annotations

from calendar import monthrange
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

HOW = 168
NY = ZoneInfo("America/New_York")
SESSIONS = ("asia", "europe", "us", "late")
SESSION_HOURS = {"asia": (0, 8), "europe": (8, 13), "us": (13, 20), "late": (20, 24)}
DERIBIT_EXPIRY_HOUR = 8

# NYSE full-day closures 2020-2027 (early closes are trading days and kept).
NYSE_HOLIDAYS = frozenset(date.fromisoformat(d) for d in ["2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25", "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24", "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26", "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25", "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"])


# ------------------------------------------------------------------ hour of week and sessions


def hour_of_week(ts) -> np.ndarray:
    """Monday 00:00 UTC = 0 ... Sunday 23:00 UTC = 167, from bar open times (polars Series, numpy datetime64 or list)."""
    s = ts if isinstance(ts, pl.Series) else pl.Series(np.asarray(ts, dtype="datetime64[us]"))
    if s.dtype == pl.Datetime and getattr(s.dtype, "time_zone", None) not in (None, "UTC"):
        s = s.dt.convert_time_zone("UTC")
    return ((s.dt.weekday().cast(pl.Int32) - 1) * 24 + s.dt.hour().cast(pl.Int32)).to_numpy().astype(np.int64)


def crypto_session(hour) -> np.ndarray:
    """Session label per UTC hour (array-like of ints 0..23)."""
    h = np.asarray(hour, dtype=int)
    out = np.empty(h.shape, dtype=object)
    for name, (a, b) in SESSION_HOURS.items():
        out[(h >= a) & (h < b)] = name
    return out


# ------------------------------------------------------------------ hour-of-week volatility profile


def _kernel(bandwidth: int) -> np.ndarray:
    w = np.array([bandwidth + 1 - abs(j) for j in range(-bandwidth, bandwidth + 1)], dtype=float)
    return w / w.sum()


def _shape(sums: np.ndarray, counts: np.ndarray, kappa: float, kern: np.ndarray, floor: float) -> np.ndarray:
    tot_n = counts.sum()
    if tot_n <= 0:
        return np.ones(HOW)
    grand = sums.sum() / tot_n
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (sums + kappa * grand) / (counts + kappa)
    b = (len(kern) - 1) // 2
    if b:
        mm = np.concatenate([m[-b:], m, m[:b]])
        m = np.convolve(mm, kern, mode="valid")
    mean = m.mean()
    if not (np.isfinite(mean) and mean > 0):
        return np.ones(HOW)
    return np.maximum(m / mean, floor)


def seasonal_factor(how: np.ndarray, u: np.ndarray, *, window_weeks: int | None = None, kappa: float = 20.0, bandwidth: int = 1,
                    cap: float = 50.0, floor: float = 0.1, min_weeks: int = 8) -> np.ndarray:
    """Causal hour-of-week factor: out[t] uses u[j], j < t, only. NaN until `min_weeks` weeks of finite u are seen
    (a factor of 1 would silently mix the warm-up into a comparison). u = r^2 / F0 (NaN entries are skipped)."""
    how = np.asarray(how, dtype=np.int64)
    u = np.asarray(u, dtype=float)
    n = len(u)
    out = np.full(n, np.nan)
    kern = _kernel(bandwidth)
    sums = np.zeros(HOW)
    counts = np.zeros(HOW)
    hist: list[list[float]] | None = [[] for _ in range(HOW)] if window_weeks else None
    need = min_weeks * HOW
    seen = 0
    cache = None
    for t in range(n):
        if seen >= need:
            if cache is None:
                cache = _shape(sums, counts, kappa, kern, floor)
            out[t] = cache[how[t]]
        x = u[t]
        if np.isfinite(x):
            x = min(max(x, 0.0), cap)
            k = how[t]
            sums[k] += x
            counts[k] += 1
            if hist is not None:
                hist[k].append(x)
                if len(hist[k]) > window_weeks:
                    sums[k] -= hist[k].pop(0)
                    counts[k] -= 1
            seen += 1
            cache = None
    return out


def profile_asof(ts, u: np.ndarray, as_of: datetime, **kw) -> np.ndarray:
    """The 168-vector of factors known at `as_of`: built from bars with close time <= as_of (bar open + 1h)."""
    s = ts if isinstance(ts, pl.Series) else pl.Series(np.asarray(ts, dtype="datetime64[us]"))
    opens = s.dt.replace_time_zone(None) if getattr(s.dtype, "time_zone", None) else s
    lim = np.datetime64(as_of.astimezone(UTC).replace(tzinfo=None), "us") - np.timedelta64(1, "h")
    ok = opens.to_numpy() <= lim  # bar open <= as_of - 1h  <=>  bar close <= as_of
    how = hour_of_week(s)[ok]
    uu = np.asarray(u, dtype=float)[ok]
    kappa, bandwidth = kw.get("kappa", 20.0), kw.get("bandwidth", 1)
    cap, floor, window = kw.get("cap", 50.0), kw.get("floor", 0.1), kw.get("window_weeks")
    fin = np.isfinite(uu)
    how, uu = how[fin], np.clip(uu[fin], 0.0, cap)
    if window:
        keep = np.zeros(len(uu), bool)
        for k in range(HOW):
            idx = np.flatnonzero(how == k)
            keep[idx[-window:]] = True
        how, uu = how[keep], uu[keep]
    sums = np.bincount(how, weights=uu, minlength=HOW).astype(float)
    counts = np.bincount(how, minlength=HOW).astype(float)
    return _shape(sums, counts, kappa, _kernel(bandwidth), floor)


# ------------------------------------------------------------------ Deribit expiry calendar


def last_friday(year: int, month: int) -> date:
    d = date(year, month, monthrange(year, month)[1])
    return d - timedelta(days=(d.weekday() - 4) % 7)


def deribit_expiry_kind(d: date) -> str | None:
    """'quarterly' / 'monthly' / 'weekly' for a Friday (Deribit expires at 08:00 UTC), None for other days. Quarterly
    implies monthly implies weekly; the label is the most senior one."""
    if d.weekday() != 4:
        return None
    if d == last_friday(d.year, d.month):
        return "quarterly" if d.month in (3, 6, 9, 12) else "monthly"
    return "weekly"


def expiry_calendar(start: date, end: date) -> pl.DataFrame:
    """Every Deribit Friday expiry with start <= date < end: date, ts (08:00 UTC), kind, is_monthly, is_quarterly."""
    d = start + timedelta(days=(4 - start.weekday()) % 7)
    rows = []
    while d < end:
        k = deribit_expiry_kind(d)
        rows.append({"date": d, "ts": datetime.combine(d, time(DERIBIT_EXPIRY_HOUR), tzinfo=UTC), "kind": k,
                     "is_monthly": k in ("monthly", "quarterly"), "is_quarterly": k == "quarterly"})
        d += timedelta(days=7)
    return pl.DataFrame(rows, schema={"date": pl.Date, "ts": pl.Datetime("us", "UTC"), "kind": pl.Utf8, "is_monthly": pl.Boolean,
                                      "is_quarterly": pl.Boolean})


# ------------------------------------------------------------------ funding settlements


def funding_settlements(funding: pl.DataFrame, interval_min: int | None = 480, hours: tuple[int, ...] | None = (0, 8, 16)) -> pl.DataFrame:
    """Map each symbol's actual settlement times from the lake's funding table (symbol, event_ts, rate, known_ts).

    interval_min = minutes since the symbol's previous print (null for its first print). With `interval_min` set, only
    settlements reached after exactly that interval are kept (e.g. 480: the 8h regime; ENA's 4h period from 2025-01 and
    SOL's 2h episode in 2022 drop out); with `hours`, only settlements at those UTC hours, minute 0."""
    f = funding.sort(["symbol", "event_ts"]).with_columns(
        pl.col("event_ts").diff().over("symbol").dt.total_minutes().alias("interval_min"),
        pl.col("event_ts").dt.hour().alias("hour"),
        pl.col("event_ts").dt.minute().alias("minute"),
    )
    if interval_min is not None:
        f = f.filter(pl.col("interval_min") == interval_min)
    if hours is not None:
        f = f.filter(pl.col("hour").is_in(list(hours)) & (pl.col("minute") == 0))
    return f.drop("minute")


# ------------------------------------------------------------------ NYSE opening bar


def is_nyse_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def nyse_open_bar(dates) -> list[datetime]:
    """UTC open time of the 1h bar containing 09:30 America/New_York on each NYSE trading day in `dates` (09:00 ET bar:
    13:00 UTC under EDT, 14:00 UTC under EST). Non-trading days are dropped."""
    out = []
    for d in dates:
        if is_nyse_day(d):
            out.append(datetime.combine(d, time(9, 0), tzinfo=NY).astimezone(UTC))
    return out


__all__ = ["HOW", "NYSE_HOLIDAYS", "SESSIONS", "SESSION_HOURS", "crypto_session", "deribit_expiry_kind", "expiry_calendar",
           "funding_settlements", "hour_of_week", "is_nyse_day", "last_friday", "nyse_open_bar", "profile_asof", "seasonal_factor"]
