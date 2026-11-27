"""Per-token market state card (research/plans/2026-11-10_volatility_program.md section 5): deterministic and point in
time. Every input is an as-of read of the lake (rows with known_ts <= as_of); nothing here is fitted to future data.

    from ai_trader.tools.market_state import state_card, market_state
    card = state_card("BTCUSDT", as_of)            # one dict, JSON-serialisable
    cards = market_state(["BTCUSDT", "XAUUSDT"], as_of)

Fields (all optional: a missing input gives None plus a data-quality flag, never an exception):
  price          last 1h close (and the latest flow_3s close if fresher)
  ret_sigma      1h / 24h / 7d log return in sigma units: ln(C_t / C_{t-h}) / (sigma_risk_{t-h+1|t-h} * sqrt(h)), i.e. the
                 robust risk sigma known at the START of the window (no look-ahead in the scale)
  regime         v0 9-cell grid on 1h bars (research.regimes.label)
  vol            robust_bc per-bar sigma (risk and central), daily (x sqrt 24) and annualised (x sqrt 8760); HAR next-UTC-day
                 sigma on 1h-derived daily RV (features.realized + features.vol_models.har_forecast, as vol.har);
                 1-year percentile of the current robust daily sigma vs its own hourly history
  vol_surprise   yesterday's realised vol / its HAR forecast made at the end of the day before (vol units, sqrt of the
                 variance ratio)
  asymmetry      drawdown flag: ln(high_30d / close) > 1.5 * sigma_30d * sqrt(30), sigma_30d = sqrt(mean daily RV, last
                 30 complete days); RS- share of RV over the last 5 complete days
  jumps          1h bars in the last 7 days with |r / sigma_risk_prev| > 4, Lee-Mykland jump count (features.jump_diffusion)
                 and the jump-variation share of RV over 5 days
  funding        latest rate, annualised by the instrument's funding interval, z vs the trailing 90 days (strictly before)
  oi             open interest (4h series) and its 24h log change, with its age
  liquidations   last 24h by side (notional, count) and the largest one
  flow           1m signed taker imbalance (buy - sell) / (buy + sell) from flow_3s; robust z vs the trailing 24h
                 median / MAD (strictly before the minute); count of |z| > 4 minutes in the last 24h and the latest one
  zones          nearest 4h structural zones above / below the price (scripts/levels.py zones() on 120 days of 4h bars)
  quality        history length, stale last bar, gaps, coverage, short-history flags
"""

from __future__ import annotations

import importlib.util
import math
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.registry import tool
from ai_trader.core.schemas import ToolResult
from ai_trader.core.time import utc
from ai_trader.storage.lake import Lake

VERSION = "0.1.0"
TRADFI_TYPES = {"stock", "ETF", "commodity", "forex"}
H_YEAR = 24 * 365
FLOW_Z = 4.0
FLOW_BASE_MIN = 24 * 60          # trailing baseline window (minutes)
FLOW_MIN_BASE = 360              # need >= 6h of minutes in the baseline to score a minute
STALE_BAR_H = 2.0
SHORT_HISTORY_DAYS = 30


# ---------------------------------------------------------------- small helpers

def _f(x: Any, nd: int = 6) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _iso(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return utc(ts).isoformat(timespec="seconds")
    return str(ts)


def _hours(a: datetime, b: datetime) -> float:
    return (utc(a) - utc(b)).total_seconds() / 3600.0


class _Max:
    """Tracks the newest known_ts of every input used (the card's data_max_ts)."""

    def __init__(self) -> None:
        self.ts: datetime | None = None

    def see(self, ts: Any) -> None:
        if ts is None:
            return
        ts = utc(ts)
        if self.ts is None or ts > self.ts:
            self.ts = ts


@lru_cache(maxsize=1)
def _zones_fn():
    """scripts/levels.py zones() (the market maker's 4h structural zones), loaded by path."""
    p = load().root / "scripts" / "levels.py"
    spec = importlib.util.spec_from_file_location("_at_levels", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.zones


# ---------------------------------------------------------------- instruments / asset class

def instruments(as_of: datetime, lake: Lake | None = None) -> tuple[dict[str, dict], bool]:
    """Linear-perp instrument rows known at as_of. Falls back to the latest snapshot (static metadata) when none was
    known yet; the bool says whether the fallback was used."""
    lake = lake or Lake()
    cols = ["symbol", "symbol_type", "funding_interval_min", "launch_ts", "known_ts"]
    try:
        d = lake.read_asof("instruments", as_of, filters={"venue": "bybit", "category": "linear"}, columns=cols)
    except Exception:  # noqa: BLE001
        d = pl.DataFrame()
    fallback = False
    if d.height == 0:
        try:
            d = lake.read_asof("instruments", datetime.now(UTC), filters={"venue": "bybit", "category": "linear"}, columns=cols)
            fallback = d.height > 0
        except Exception:  # noqa: BLE001
            d = pl.DataFrame()
    return ({r["symbol"]: r for r in d.iter_rows(named=True)} if d.height else {}), fallback


def asset_class(inst: dict | None) -> tuple[str, str]:
    st = (inst or {}).get("symbol_type")
    if st is None:
        return "unknown", ""
    return ("tradfi", st) if st in TRADFI_TYPES else ("crypto", st or "")


# ---------------------------------------------------------------- order flow (local causal detector)

def _flow_minutes(symbols: list[str], as_of: datetime, hours: float, lake: Lake, category: str = "linear") -> pl.DataFrame:
    """1m aggregates of flow_3s rows known at as_of, for complete minutes in [as_of - hours, floor_min(as_of)).
    Reads only the date partitions it needs, with column projection (flow_3s is the largest table)."""
    end = utc(as_of).replace(second=0, microsecond=0)
    start = end - timedelta(hours=hours)
    base = lake.root / "flow_3s" / "venue=bybit" / f"category={category}"
    globs = []
    d = start.date()
    while d <= end.date():
        p = base / f"date={d:%Y-%m-%d}"
        if p.is_dir() and any(p.glob("*.parquet")):
            globs.append((p / "*.parquet").as_posix())
        d += timedelta(days=1)
    empty = pl.DataFrame(schema={"symbol": pl.Utf8, "m": pl.Datetime("us", "UTC"), "buy": pl.Float64, "sell": pl.Float64, "close": pl.Float64, "known_ts": pl.Datetime("us", "UTC")})
    if not globs or not symbols:
        return empty
    sql = f"""
        WITH r AS (
            SELECT symbol, event_ts,
                   arg_max(buy_notional, known_ts) AS bv, arg_max(sell_notional, known_ts) AS sv, arg_max(close, known_ts) AS c, max(known_ts) AS k
            FROM read_parquet({globs!r}, union_by_name = true)
            WHERE known_ts <= ? AND event_ts >= ? AND event_ts < ? AND symbol IN (SELECT unnest(?))
            GROUP BY symbol, event_ts
        )
        SELECT symbol, time_bucket(INTERVAL 1 MINUTE, event_ts) AS m, sum(bv)::DOUBLE AS buy, sum(sv)::DOUBLE AS sell, arg_max(c, event_ts)::DOUBLE AS close, max(k) AS known_ts
        FROM r GROUP BY 1, 2 ORDER BY 1, 2
    """
    out = pl.from_arrow(lake.con.sql(sql, params=[utc(as_of), start, end, list(symbols)]).to_arrow_table())
    if out.height == 0:
        return empty
    return out.with_columns(pl.col("m").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
                            pl.col("known_ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us"))


def _robust_z(x: np.ndarray, n_target: int, base: int = FLOW_BASE_MIN, min_base: int = FLOW_MIN_BASE) -> np.ndarray:
    """z of the last n_target values vs the median / MAD of the `base` values strictly before each one (causal)."""
    n = len(x)
    out = np.full(n_target, np.nan)
    if n <= base:
        x = np.r_[np.full(base + 1 - n, np.nan), x]
        n = len(x)
    from numpy.lib.stride_tricks import sliding_window_view

    w = sliding_window_view(x[:-1], base)[-n_target:]            # row j = the `base` minutes before target j
    tgt = x[-n_target:]
    cnt = np.isfinite(w).sum(axis=1)
    ok = cnt >= min_base
    if not ok.any():
        return out
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(w[ok], axis=1)
            mad = np.nanmedian(np.abs(w[ok] - med[:, None]), axis=1) * 1.4826
        z = (tgt[ok] - med) / np.where(mad > 0, mad, np.nan)
    out[ok] = z
    return out


def flow_anomalies_local(symbols: list[str], as_of: datetime, lookback_h: int = 24, *, lake: Lake | None = None,
                         z_thr: float = FLOW_Z) -> dict[str, dict]:
    """Minimal causal detector, same feature as features.flow_anomaly (A4): signed taker imbalance = buy - sell notional
    per complete 1m bar; robust z vs the median / MAD of the trailing 24h (strictly before; >= 6h of minutes needed).
    Always computed for coverage and the latest flow price; its counts are the fallback when A4's module fails."""
    lake = lake or Lake()
    as_of = utc(as_of)
    m = _flow_minutes(symbols, as_of, lookback_h + FLOW_BASE_MIN / 60, lake)
    end = as_of.replace(second=0, microsecond=0)
    n_t = lookback_h * 60
    grid = pl.DataFrame({"m": pl.datetime_range(end - timedelta(minutes=FLOW_BASE_MIN + n_t), end - timedelta(minutes=1),
                                                 interval="1m", time_unit="us", time_zone="UTC", eager=True)})
    out: dict[str, dict] = {}
    for s in symbols:
        g = m.filter(pl.col("symbol") == s)
        if g.height == 0:
            out[s] = {"source": "local", "minutes_24h": 0, "coverage_24h": 0.0, "n_anom_24h": None, "latest": None}
            continue
        g = grid.join(g, on="m", how="left").with_columns((pl.col("buy") - pl.col("sell")).alias("imb"),
                                                          (pl.col("buy") + pl.col("sell")).alias("notional"))
        x = g["imb"].fill_null(np.nan).to_numpy().astype(float)
        z = _robust_z(x, n_t)
        tail = g.tail(n_t).with_columns(pl.Series("z", z).fill_nan(None))  # polars orders NaN above every number
        an = tail.filter(pl.col("z").abs() > z_thr)
        have = int(tail["imb"].is_not_null().sum())
        latest = None
        if an.height:
            r = an.tail(1).row(0, named=True)
            latest = {"ts": _iso(r["m"]), "z": _f(r["z"], 2), "imbalance_usd": _f(r["imb"], 0), "side": "buy" if r["z"] > 0 else "sell", "notional_usd": _f(r["notional"], 0),
                      "age_min": _f(_hours(end, r["m"]) * 60, 0)}
        zz = tail["z"].drop_nulls()
        last_row = g.filter(pl.col("close").is_not_null()).tail(1)
        out[s] = {
            "source": "local",
            "minutes_24h": have, "coverage_24h": _f(have / n_t, 3),
            "n_scored_24h": int(np.isfinite(z).sum()),
            "n_anom_24h": int(an.height) if np.isfinite(z).any() else None,
            "n_anom_buy": int((an["z"] > 0).sum()), "n_anom_sell": int((an["z"] < 0).sum()),
            "n_anom_1h": int(an.filter(pl.col("m") >= end - timedelta(hours=1)).height),
            "max_abs_z_24h": _f(zz.abs().max(), 2) if zz.len() else None,
            "latest": latest,
            "last_close": _f(last_row["close"][0], 8) if last_row.height else None,
            "last_ts": _iso(last_row["m"][0] + timedelta(minutes=1)) if last_row.height else None,
            "known_max": _iso(g["known_ts"].max()),
        }
    return out


class FlowLake(Lake):
    """A Lake whose flow_3s as-of reads touch only the date partitions in [start, as_of] (same semantics as
    Lake.read_asof: known_ts <= as_of, latest revision per primary key, column projection). Passed to A4's
    latest_anomalies, whose generic read otherwise scans every flow_3s file."""

    def read_asof(self, table, as_of, *, symbols=None, start=None, end=None, filters=None, columns=None):
        if table != "flow_3s" or start is None:
            return super().read_asof(table, as_of, symbols=symbols, start=start, end=end, filters=filters, columns=columns)
        as_of, start = utc(as_of), utc(start)
        filters = dict(filters or {})
        venue, category = str(filters.pop("venue", "bybit")), str(filters.pop("category", "linear"))
        if filters:
            return super().read_asof(table, as_of, symbols=symbols, start=start, end=end, filters={"venue": venue, "category": category, **filters}, columns=columns)
        spec = self.spec(table)
        cols = list(columns or [c["name"] for c in spec["columns"]])
        base = self.root / table / f"venue={venue}" / f"category={category}"
        globs, d = [], start.date()
        while d <= as_of.date():
            q = base / f"date={d:%Y-%m-%d}"
            if q.is_dir() and any(q.glob("*.parquet")):
                globs.append((q / "*.parquet").as_posix())
            d += timedelta(days=1)
        if not globs:
            return pl.DataFrame(schema={c: pl.Null for c in cols})
        where, params = ["known_ts <= ?", "event_ts >= ?"], [as_of, start]
        if end is not None:
            where.append("event_ts < ?")
            params.append(utc(end))
        if symbols is not None:
            where.append("symbol IN (SELECT unnest(?))")
            params.append(list(symbols))
        sel = ", ".join(f'"{c}"' for c in cols)
        sql = f"""SELECT {sel} FROM (
                    SELECT *, row_number() OVER (PARTITION BY symbol, event_ts ORDER BY known_ts DESC, filename DESC) AS _rn
                    FROM read_parquet({globs!r}, hive_partitioning = true, union_by_name = true, filename = true)
                    WHERE {' AND '.join(where)}) WHERE _rn = 1 ORDER BY symbol, event_ts"""
        return pl.from_arrow(self.con.sql(sql, params=params).to_arrow_table())


def flow_anomalies(symbols: list[str], as_of: datetime, lookback_h: int = 24, *, lake: Lake | None = None,
                   use_a4: bool = False) -> dict[str, dict]:
    """Flow anomaly flags per symbol (`source` says which detector produced the counts).

    use_a4=True: counts and the latest anomaly come from A4's ai_trader.features.flow_anomaly.latest_anomalies
    (imbalance type = |robust z| > 4 minutes of 1m signed taker imbalance; its other anomaly types are kept in
    `counts_all`), falling back to the local detector if it fails. Default False because on 2026-11-10 it took ~45 s for
    8 symbols (its rolling MADs, not the read) and raised MemoryError for the 57-symbol universe next to the other
    research jobs. Cross-check the same evening: local vs A4 imbalance counts BTC 53/53, TRX 157/156, GRAM 106/104.
    TODO: make use_a4=True the default once latest_anomalies is fast enough for the daily universe.
    Coverage and the latest flow price always come from the local 1m aggregation."""
    lake = lake or Lake()
    local = flow_anomalies_local(symbols, as_of, lookback_h, lake=lake)
    a4, err = None, None
    if use_a4:
        try:
            from ai_trader.features.flow_anomaly import latest_anomalies

            a4 = latest_anomalies(list(symbols), utc(as_of), lookback_h, lake=FlowLake(config=lake.config, root=lake.root))
        except Exception as e:  # noqa: BLE001
            a4, err = None, f"{type(e).__name__}: {e}"
    end = utc(as_of).replace(second=0, microsecond=0)
    out: dict[str, dict] = {}
    for s in symbols:
        c = dict(local.get(s, {}))
        if a4 is None:
            if err:
                c["a4_error"] = err
            out[s] = c
            continue
        r = a4.get(s)
        c["local"] = {k: c.get(k) for k in ("n_anom_24h", "n_scored_24h", "latest", "max_abs_z_24h")}
        c["source"] = "features.flow_anomaly.latest_anomalies"
        if not r:
            c.update({"n_anom_24h": None, "n_scored_24h": 0, "latest": None, "counts_all": {}, "latest_any": None})
            out[s] = c
            continue
        li = (r.get("latest_by_type") or {}).get("imbalance")
        latest = None
        if li:
            latest = {"ts": li["time"], "z": _f(li["z"], 2), "side": li["side"], "age_min": _f(_hours(end, utc(li["time"])) * 60, 0)}
        la = r.get("latest")
        c.update({
            "n_scored_24h": int(r.get("minutes") or 0),
            "n_anom_24h": int((r.get("counts") or {}).get("imbalance", 0)) if r.get("minutes") else None,
            "latest": latest,
            "counts_all": r.get("counts") or {}, "n_total_all": int(r.get("n_total") or 0),
            "latest_any": {"type": la["type"], "ts": la["time"], "z": _f(la["z"], 2), "side": la["side"]} if la else None,
        })
        for k in ("n_anom_buy", "n_anom_sell", "n_anom_1h", "max_abs_z_24h"):
            c.pop(k, None)
        out[s] = c
    return out


# ---------------------------------------------------------------- per-section computations

def _returns_block(f: pl.DataFrame) -> dict:
    ts = f["event_ts"].to_list()
    idx = {t: i for i, t in enumerate(ts)}
    c = f["close"].to_numpy()
    sr = f["sigma_risk_next"].to_numpy()
    last = len(ts) - 1
    out = {}
    for lab, h in (("1h", 1), ("24h", 24), ("7d", 168)):
        j = idx.get(ts[last] - timedelta(hours=h))
        if j is None or not np.isfinite(sr[j]) or sr[j] <= 0:
            out[lab] = {"ret": None, "sigma_units": None}
            continue
        r = math.log(c[last] / c[j])
        out[lab] = {"ret": _f(r), "sigma_units": _f(r / (sr[j] * math.sqrt(h)), 2)}
    return out


def _vol_block(f: pl.DataFrame, daily: pl.DataFrame, flags: list[str]) -> dict:
    from ai_trader.features.vol_models import har_forecast

    last = f.tail(1).row(0, named=True)
    s_h = last["sigma_risk_next"]
    out: dict[str, Any] = {
        "robust_sigma_1h_risk": _f(s_h), "robust_sigma_1h_central": _f(last["sigma_rb_next"]), "weekend_m": _f(last["wk_m"], 3),
        "robust_sigma_day": _f(s_h * math.sqrt(24)) if s_h is not None else None,
        "robust_sigma_ann": _f(s_h * math.sqrt(H_YEAR)) if s_h is not None else None,
    }
    # 1-year percentile of the current robust daily sigma vs its own hourly history (same scale, so sqrt(24) cancels)
    w = f.filter(pl.col("event_ts") > last["event_ts"] - timedelta(days=365))["sigma_risk_next"].drop_nulls().drop_nans().to_numpy()
    if len(w) >= 24 * 30 and s_h is not None:
        out["vol_pct_1y"] = _f(float((w < s_h).mean() + 0.5 * (w == s_h).mean()), 3)
        out["vol_pct_window_days"] = _f(len(w) / 24, 0)
        if len(w) < 24 * 360:
            flags.append("vol_pct_window_short")
    else:
        out["vol_pct_1y"], out["vol_pct_window_days"] = None, _f(len(w) / 24, 0)
        flags.append("vol_pct_unavailable")

    out.update({"har_sigma_day": None, "har_for_day": None, "rv_last_day": None, "last_complete_day": None,
                "vol_surprise": None, "vol_surprise_day": None})
    if daily.height >= 31:
        rv = daily["rv"].to_numpy()
        fc = har_forecast(rv)
        if not np.isfinite(fc[-1]):
            fc = har_forecast(rv, min_days=60)
            if np.isfinite(fc[-1]):
                flags.append("har_short_fit")
        dates = daily["date"].to_list()
        out["last_complete_day"] = str(dates[-1])
        out["rv_last_day"] = _f(math.sqrt(rv[-1]))
        if np.isfinite(fc[-1]):
            out["har_sigma_day"] = _f(math.sqrt(fc[-1]))
            out["har_for_day"] = str(dates[-1] + timedelta(days=1))
        else:
            flags.append("har_unavailable")
        if len(fc) >= 2 and np.isfinite(fc[-2]) and fc[-2] > 0 and (dates[-1] - dates[-2]).days == 1:
            out["vol_surprise"] = _f(math.sqrt(rv[-1] / fc[-2]), 3)
            out["vol_surprise_day"] = str(dates[-1])
    else:
        flags.append("har_unavailable")
    return out


def _asym_block(f: pl.DataFrame, daily: pl.DataFrame) -> dict:
    last = f.tail(1).row(0, named=True)
    out: dict[str, Any] = {"drawdown_flag": None, "drawdown_log": None, "drawdown_sigma30": None, "high_30d": None,
                           "rs_neg_share_5d": None, "jv_share_5d": None}
    w30 = f.filter(pl.col("event_ts") > last["event_ts"] - timedelta(days=30))
    d30 = daily.tail(30)
    if w30.height and d30.height >= 20:
        hi = float(w30["high"].max())
        sig30 = math.sqrt(float(d30["rv"].mean()))
        dd = math.log(hi / last["close"])
        scale = sig30 * math.sqrt(30)
        out.update({"high_30d": _f(hi, 8), "drawdown_log": _f(dd, 4), "sigma_30d_daily": _f(sig30),
                    "drawdown_sigma30": _f(dd / scale, 2) if scale > 0 else None, "drawdown_flag": bool(dd > 1.5 * scale)})
    d5 = daily.tail(5)
    if d5.height == 5 and float(d5["rv"].sum()) > 0:
        tot = float(d5["rv"].sum())
        out["rs_neg_share_5d"] = _f(float(d5["rs_neg"].sum()) / tot, 3)
        out["jv_share_5d"] = _f(float(d5["jv"].sum()) / tot, 3)
    return out


def _jump_block(f: pl.DataFrame) -> dict:
    from ai_trader.features.jump_diffusion import bipower_sigma, lee_mykland

    g = f.tail(168 + 80)
    r = g["ret"].fill_nan(None).fill_null(0.0).to_numpy()
    sp = g["sigma_risk_next"].shift(1).to_numpy()
    with np.errstate(all="ignore"):
        zr = np.abs(r / sp)[-168:]
    out = {"n_exceed_4sigma_7d": int(np.nansum(zr > 4)), "max_abs_z_7d": _f(np.nanmax(zr), 2) if np.isfinite(zr).any() else None}
    if len(r) >= 168 + 74:
        bs = bipower_sigma(r, 72)
        L, thr = lee_mykland(r, bs, 24)
        with np.errstate(all="ignore"):
            out["n_jumps_lm_7d"] = int(np.nansum(L[-168:] > thr))
    else:
        out["n_jumps_lm_7d"] = None
    return out


def _funding_block(symbol: str, as_of: datetime, lake: Lake, inst: dict | None, seen: _Max, flags: list[str]) -> dict:
    d = lake.read_asof("funding", as_of, symbols=[symbol], start=as_of - timedelta(days=95), filters={"venue": "bybit"})
    out: dict[str, Any] = {"rate": None, "ts": None, "rate_ann": None, "z_90d": None, "n_90d": 0, "interval_min": None}
    if d.height == 0:
        flags.append("no_funding")
        return out
    d = d.sort("event_ts")
    seen.see(d["known_ts"].max())
    last = d.tail(1).row(0, named=True)
    iv = (inst or {}).get("funding_interval_min")
    if not iv and d.height >= 2:
        iv = int(round((d["event_ts"][-1] - d["event_ts"][-2]).total_seconds() / 60))
    prev = d.filter((pl.col("event_ts") < last["event_ts"]) & (pl.col("event_ts") >= last["event_ts"] - timedelta(days=90)))["rate"].to_numpy()
    z = None
    if len(prev) >= 30 and np.std(prev, ddof=1) > 0:
        z = (last["rate"] - float(np.mean(prev))) / float(np.std(prev, ddof=1))
    else:
        flags.append("funding_z_short")
    age = _hours(as_of, last["known_ts"])
    if iv and age > 2 * iv / 60 + 0.5:
        flags.append("stale_funding")
    crowd = None
    if z is not None:
        crowd = "longs" if (z >= 2 and last["rate"] > 0) else "shorts" if (z <= -2 and last["rate"] < 0) else None
    out.update({"rate": _f(last["rate"], 8), "ts": _iso(last["event_ts"]), "interval_min": iv,
                "rate_ann": _f(last["rate"] * H_YEAR * 60 / iv, 4) if iv else None, "z_90d": _f(z, 2), "n_90d": int(len(prev)),
                "mean_90d": _f(np.mean(prev), 8) if len(prev) else None, "crowded": crowd, "age_h": _f(age, 2)})
    return out


def _oi_block(symbol: str, as_of: datetime, lake: Lake, price: float, seen: _Max, flags: list[str]) -> dict:
    d = lake.read_asof("open_interest", as_of, symbols=[symbol], start=as_of - timedelta(days=4), filters={"venue": "bybit", "interval": "4h"})
    out: dict[str, Any] = {"oi": None, "oi_usd": None, "chg_24h_log": None, "ts": None, "age_h": None}
    if d.height == 0:
        flags.append("no_oi")
        return out
    d = d.sort("event_ts")
    seen.see(d["known_ts"].max())
    last = d.tail(1).row(0, named=True)
    prev = d.filter(pl.col("event_ts") <= last["event_ts"] - timedelta(hours=24)).tail(1)
    chg = None
    if prev.height and prev["open_interest"][0] > 0 and last["open_interest"] > 0:
        chg = math.log(last["open_interest"] / prev["open_interest"][0])
    age = _hours(as_of, last["known_ts"])
    if age > 9:
        flags.append("stale_oi")
    out.update({"oi": _f(last["open_interest"], 2), "oi_usd": _f(last["open_interest"] * price, 0), "chg_24h_log": _f(chg, 4),
                "ts": _iso(last["event_ts"]), "age_h": _f(age, 2)})
    return out


def liquidations_window(as_of: datetime, symbols: list[str] | None = None, hours: int = 24, lake: Lake | None = None) -> pl.DataFrame:
    lake = lake or Lake()
    d = lake.read_asof("liquidations", as_of, symbols=symbols, start=as_of - timedelta(hours=hours), filters={"venue": "bybit"})
    if d.height == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "event_ts": pl.Datetime("us", "UTC"), "side": pl.Utf8, "usd": pl.Float64,
                                    "known_ts": pl.Datetime("us", "UTC")})
    return d.with_columns((pl.col("price") * pl.col("size")).alias("usd")).select("symbol", "event_ts", "side", "usd", "known_ts")


def _liq_block(d: pl.DataFrame, as_of: datetime, seen: _Max) -> dict:
    out: dict[str, Any] = {}
    for side in ("long", "short"):
        g = d.filter(pl.col("side") == side)
        out[f"{side}_usd"] = _f(g["usd"].sum(), 0) if g.height else 0.0
        out[f"{side}_n"] = int(g.height)
    h1 = d.filter(pl.col("event_ts") >= as_of - timedelta(hours=1))
    out["n_1h"] = int(h1.height)
    out["usd_1h"] = _f(h1["usd"].sum(), 0) if h1.height else 0.0
    if d.height:
        seen.see(d["known_ts"].max())
        b = d.sort("usd").tail(1).row(0, named=True)
        out["largest"] = {"usd": _f(b["usd"], 0), "side": b["side"], "ts": _iso(b["event_ts"])}
    else:
        out["largest"] = None
    return out


def _zones_block(symbol: str, as_of: datetime, lake: Lake, price: float, sigma_day: float | None, seen: _Max, flags: list[str]) -> dict:
    b = lake.bars(symbol, "4h", as_of, start=as_of - timedelta(days=120)).filter(
        (pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.col("low") <= pl.min_horizontal("open", "close"))).sort("event_ts")
    out: dict[str, Any] = {"below": None, "above": None, "inside": None, "n_bars_4h": int(b.height)}
    if b.height < 30:
        flags.append("no_4h_zones")
        return out
    seen.see(b["known_ts"].max())
    zs, atr = _zones_fn()(b["event_ts"].to_list(), b["open"].to_numpy(), b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())

    def fmt(z: dict, ref: float) -> dict:
        dist = math.log(ref / price)
        return {"lo": z["lo"], "hi": z["hi"], "score": z["score"], "pivots": z["pivots"], "dist_pct": _f(100 * (math.exp(dist) - 1), 2),
                "dist_sigma_day": _f(dist / sigma_day, 2) if sigma_day else None}

    below = [z for z in zs if z["hi"] < price]
    above = [z for z in zs if z["lo"] > price]
    inside = [z for z in zs if z["lo"] <= price <= z["hi"]]
    if below:
        z = max(below, key=lambda z: z["hi"])
        out["below"] = fmt(z, z["hi"])
    if above:
        z = min(above, key=lambda z: z["lo"])
        out["above"] = fmt(z, z["lo"])
    if inside:
        z = max(inside, key=lambda z: z["score"])
        out["inside"] = {"lo": z["lo"], "hi": z["hi"], "score": z["score"]}
    out["atr_4h"] = _f(atr, 8)
    return out


# ---------------------------------------------------------------- the card

def state_card(symbol: str, as_of: datetime, *, lake: Lake | None = None, flow: dict | bool | None = None,
               liquidations: pl.DataFrame | None = None, instrument: dict | None = None, instruments_fallback: bool = False) -> dict:
    """Market state of one symbol at `as_of`. `flow`, `liquidations` and `instrument` may be passed precomputed (the
    batch report reads them once for every symbol); otherwise they are read here."""
    from ai_trader.features.realized import daily_measures
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import CELL_LABELS, label

    as_of = utc(as_of)
    lake = lake or Lake()
    seen = _Max()
    flags: list[str] = []
    if instrument is None:
        inst_all, instruments_fallback = instruments(as_of, lake)
        instrument = inst_all.get(symbol)
    if instruments_fallback:
        flags.append("instruments_not_pit")
    cls, sub = asset_class(instrument)
    card: dict[str, Any] = {"symbol": symbol, "as_of": _iso(as_of), "asset_class": cls, "symbol_type": sub, "version": VERSION}

    f = market_frame(symbol, "1h", as_of, lake=lake, funding=False, oi=False)
    if f.height < 24 * 7:
        # The listing burn-in (market_frame) drops the first 14-30 days after the first lake bar. For contracts whose lake
        # history simply starts late (backfill began 2026-11-05) that removes everything: use the raw history, flagged.
        f2 = market_frame(symbol, "1h", as_of, lake=lake, funding=False, oi=False, burn_in=False)
        if f2.height > f.height:
            launch = (instrument or {}).get("launch_ts")
            rs = load().settings["research"]
            late = launch is not None and f2["event_ts"][0] >= utc(launch) + timedelta(days=rs["burn_in_days_new"])
            flags.append("burn_in_skipped:history_starts_after_listing" if late else "burn_in_skipped:inside_listing_burn_in")
            f = f2
    from ai_trader.risk.sigma import pin_base

    f = pin_base(f)   # the card's vol / ret_sigma / jump fields are robust_bc by definition (risk.sigma roles)
    if f.height < 50:
        flags.append("no_1h_bars" if f.height == 0 else "too_few_1h_bars")
        card.update({"price": None, "flags": flags, "data_max_ts": None, "history_days": _f(f.height / 24, 1)})
        return card
    seen.see(f["known_ts"].max())
    f = label(f, 1.0)
    last = f.tail(1).row(0, named=True)
    price = float(last["close"])

    # quality
    hist_days = _hours(last["event_ts"], f["event_ts"][0]) / 24 + 1 / 24
    age_h = _hours(as_of, last["close_ts"])
    n7 = f.filter(pl.col("event_ts") > last["event_ts"] - timedelta(days=7)).height
    if age_h > STALE_BAR_H:
        flags.append("stale_last_bar")
    exp7 = min(168, int(round(_hours(last["event_ts"], f["event_ts"][0]))) + 1)
    if n7 < exp7:
        flags.append(f"gaps_7d:{exp7 - n7}")
    if hist_days < SHORT_HISTORY_DAYS:
        flags.append("short_history")

    daily = daily_measures(f.select("event_ts", "open", "high", "low", "close"), "1h").filter(pl.col("known_ts") <= as_of)
    vol = _vol_block(f, daily, flags)
    card.update({
        "price": _f(price, 8), "last_bar": _iso(last["event_ts"]), "last_bar_age_h": _f(age_h, 2),
        "returns": _returns_block(f),
        "regime": {"cell": last["cell"], "label": CELL_LABELS.get(last["cell"]) if last["cell"] else None,
                   "dir_state": last["dir_state"], "vol_state": last["vol_state"], "dir_score": _f(last["dir_score"], 3)},
        "vol": vol,
        "asymmetry": _asym_block(f, daily),
        "jumps": _jump_block(f),
    })
    card["funding"] = _funding_block(symbol, as_of, lake, instrument, seen, flags)
    card["oi"] = _oi_block(symbol, as_of, lake, price, seen, flags)

    liq = liquidations if liquidations is not None else liquidations_window(as_of, [symbol], lake=lake)
    card["liquidations_24h"] = _liq_block(liq.filter(pl.col("symbol") == symbol) if liq.height else liq, as_of, seen)

    card["zones_4h"] = _zones_block(symbol, as_of, lake, price, vol.get("har_sigma_day") or vol.get("robust_sigma_day"), seen, flags)
    card["history_days"] = _f(hist_days, 1)
    card["first_bar"] = _iso(f["event_ts"][0])
    card["flags"] = flags
    card["data_max_ts"] = _iso(seen.ts)
    if flow is not False:  # False = the caller attaches flow later (attach_flow), e.g. the batch report
        fl = flow if flow is not None else flow_anomalies([symbol], as_of, lake=lake).get(symbol, {})
        attach_flow(card, fl)
    if card["data_max_ts"] is not None and utc(card["data_max_ts"]) > as_of:  # defensive: the lake reader refuses this
        raise RuntimeError(f"{symbol}: input known after as_of")
    return card


def attach_flow(card: dict, fl: dict | None) -> dict:
    """Adds the flow block (from flow_anomalies) to a card, with its quality flags and the fresher flow price."""
    fl = dict(fl or {})
    flags = card.setdefault("flags", [])
    if card.get("price") is None:
        card["flow"] = fl
        return card
    if fl.get("known_max") and (card.get("data_max_ts") is None or utc(fl["known_max"]) > utc(card["data_max_ts"])):
        card["data_max_ts"] = _iso(utc(fl["known_max"]))
    if not fl or not fl.get("minutes_24h"):
        flags.append("no_flow")
    else:
        if (fl.get("coverage_24h") or 0) < 0.5:
            flags.append("flow_coverage_low")
        if (fl.get("n_scored_24h") or 0) < 360:
            flags.append("flow_few_scored_minutes")
    card["flow"] = fl
    last_close_ts = utc(card["last_bar"]) + timedelta(hours=1)
    if fl.get("last_close") and fl.get("last_ts") and utc(fl["last_ts"]) > last_close_ts:
        card["price_flow"] = fl["last_close"]
        card["price_flow_ts"] = fl["last_ts"]
    return card


def market_state(symbols: list[str], as_of: datetime, *, lake: Lake | None = None) -> list[dict]:
    """Cards for many symbols; shared inputs (instruments, liquidations, flow) are read once. A failing symbol gives a
    card with an `error` field instead of raising."""
    as_of = utc(as_of)
    lake = lake or Lake()
    inst, fb = instruments(as_of, lake)
    liq = liquidations_window(as_of, list(symbols), lake=lake)
    try:
        flows = flow_anomalies(list(symbols), as_of, lake=lake)
    except Exception as e:  # noqa: BLE001
        flows = {s: {"error": f"{type(e).__name__}: {e}"} for s in symbols}
    out = []
    for s in symbols:
        try:
            out.append(state_card(s, as_of, lake=lake, flow=flows.get(s, {}), liquidations=liq.filter(pl.col("symbol") == s),
                                  instrument=inst.get(s), instruments_fallback=fb))
        except Exception as e:  # noqa: BLE001
            out.append({"symbol": s, "as_of": _iso(as_of), "error": f"{type(e).__name__}: {e}", "flags": ["error"]})
    return out


@tool("state.market_card", VERSION)
def market_card(symbol: str, *, as_of: datetime, lake: Lake | None = None) -> ToolResult:
    """Registered wrapper: the state card as a ToolResult (descriptive only; no tested signal behind any field)."""
    c = state_card(symbol, as_of, lake=lake)
    dm = utc(c["data_max_ts"]) if c.get("data_max_ts") else as_of
    v = c.get("vol") or {}
    rg = c.get("regime") or {}
    summ = (f"{symbol} {c.get('price')} {rg.get('cell')} {rg.get('label') or ''}; robust σ/day {v.get('robust_sigma_day')}, "
            f"HAR σ/day {v.get('har_sigma_day')}, vol pct 1y {v.get('vol_pct_1y')}, surprise {v.get('vol_surprise')}; flags {c.get('flags')}")
    return ToolResult(tool="state.market_card", version=VERSION, asset=symbol, as_of=as_of, data_max_ts=dm, value=c,
                      quality={"flags": c.get("flags", [])}, summary=summ)
