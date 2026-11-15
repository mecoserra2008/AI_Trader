"""Cross-venue study: Bybit linear perps vs BlackBull cTrader CFDs on the same underlying (H-ARB1, H-ARB2).

Data (never the lake; all under data/research/xvenue/):
  bybit_book_1s/<SYM>/<date>.parquet   Bybit best bid / ask at the END of each 1 s bin of the ob200 push time (sim.book);
                                       micro_days/<date>/book_1s/<SYM>.parquet is reused where it already exists
  ctrader_ticks/<name>/<date>_<side>.parquet   cTrader historical ticks (atx ctrader ticks): event_ts, price
  ctrader_map.json                     resolved names / ids / symbol details (atx ctrader symbols --resolve)
  usdcusdt_1m.parquet                  Bybit spot USDCUSDT 1 m klines (USDT basis)
  funding.parquet                      Bybit funding history for the mapped perps (public REST; the lake has gaps)

Grid. One row per UTC second T (state known at T): Bybit bid / ask = book state at the end of (T-1 s, T]; cTrader
bid / ask = last tick at or before T per side (as-of join, no look-ahead). A second is *common* when the CFD is inside
its schedule (when known) and both venues quoted within `live_age_s`.

Gaps (log, bp). corr = ln(USDCUSDT) for usdt_correct_classes else 0, m = multiplier.
  mid gap         g   = ln(mid_b / (m mid_c)) - corr
  sell Bybit / buy CFD   e_sb = ln(bid_b / (m ask_c)) - corr - fees
  buy Bybit / sell CFD   e_bb = ln(m bid_c / ask_b) + corr - fees
  fees = round-trip taker fees on both venues (entry + exit): 2 (bybit_taker + ctrader_commission) in bp.
The executable gap is max(e_sb, e_bb); a second is executable when it is > 0. Episodes = maximal runs of executable
seconds in one direction. Latency: an episode only counts for H-ARB1 if it lasts > 2 x RTT (RTT = measured order
round trip); the H-ARB2 simulation fills every leg `delay_s` seconds after its signal at the then-current touch.

H-ARB2 trade. Signal at the first executable second of an episode (direction d); fill at T + delay at the touch; exit
signal when |g| <= |g_entry| / 2 or after 60 minutes or at the last common second, fill at exit + delay at the touch.
Net = gross (touch to touch, both legs) - fees + funding received on the perp leg - CFD swap where held over the
broker's rollover. One trade at a time per instrument-day. Mean net per trade with a day-clustered bootstrap CI.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from ai_trader.core.config import load

DAY_S = 86_400
NY = ZoneInfo("America/New_York")
US_HOLIDAYS_2026 = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
                    "2026-11-26", "2026-12-25"}
TRADFI = ("stock", "etf", "metal", "energy", "fx")
WEEKEND_TRADING_CLASSES = ("crypto",)
EXIT_MAX_S = 3600


def root() -> Path:
    return load().root


def xdir() -> Path:
    return root() / "data" / "research" / "xvenue"


def cfg() -> dict[str, Any]:
    c = yaml.safe_load((root() / "config" / "xvenue.yaml").read_text(encoding="utf-8"))
    syms = [i["bybit"] for i in c["instruments"]]
    if "SPYUSDT" in syms:
        raise ValueError("SPYUSDT must never be part of the cross-venue map")
    return c


def study_instruments(c: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    c = c or cfg()
    return [i for i in c["instruments"] if i.get("study")]


# ----------------------------------------------------------------------------------------------------------------------
# day split (written before any cTrader data is looked at)
# ----------------------------------------------------------------------------------------------------------------------


def micro_days_cached(sym: str) -> list[str]:
    base = root() / "data" / "research" / "micro_days"
    return sorted(p.parent.parent.name for p in base.glob(f"*/book_1s/{sym}.parquet"))


def day_pool(inst: dict[str, Any], start: str, end: str, launch: dict[str, str], cached: Sequence[str] = ()) -> list[str]:
    """Candidate days in [start, end]: weekdays (any day for crypto), after the perp's listing + 7 days, no US market
    holiday for stocks / ETFs. If the Bybit book is already reduced for some days (micro_days), the pool is those days
    only (no duplicate ob200 downloads); the rule depends on data availability, never on outcomes."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    lst = date.fromisoformat(launch[inst["bybit"]][:10]) + timedelta(days=7) if inst["bybit"] in launch else d0
    out = []
    d = max(d0, lst)
    while d <= d1:
        s = d.isoformat()
        weekday_ok = d.weekday() < 5 or inst["class"] in WEEKEND_TRADING_CLASSES
        hol_ok = not (inst["class"] in ("stock", "etf") and s in US_HOLIDAYS_2026)
        if weekday_ok and hol_ok:
            out.append(s)
        d += timedelta(days=1)
    if cached:
        c = [x for x in out if x in set(cached)]
        if len(c) >= 5:
            return c
    return out


def draw_split(insts: Sequence[dict[str, Any]], start: str, end: str, launch: dict[str, str], seed: int, n: int = 5,
               cached: dict[str, Sequence[str]] | None = None) -> dict[str, Any]:
    cached = cached or {}
    out: dict[str, Any] = {}
    for k, inst in enumerate(insts):
        pool = day_pool(inst, start, end, launch, cached.get(inst["bybit"], ()))
        rng = np.random.default_rng([seed, k])
        pick = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()) if pool else []
        # weekend reopen (descriptive): the Sunday before a sampled Monday, for CFDs that reopen on Sunday evening
        extra = []
        if inst["class"] in ("metal", "energy", "fx"):
            extra = [(date.fromisoformat(d) - timedelta(days=1)).isoformat() for d in pick if date.fromisoformat(d).weekday() == 0]
        out[inst["bybit"]] = {"class": inst["class"], "pool_size": len(pool), "pool_from_cache": bool(cached.get(inst["bybit"])) and set(pool) <= set(cached[inst["bybit"]]),
                              "days": pick, "reopen_sundays": extra}
    return out


def write_split(path: Path, split: dict[str, Any], meta: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists: the split is fixed once written")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "instruments": split}, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------------


def _epoch_s(col: str = "ts") -> pl.Expr:
    return (pl.col(col).dt.epoch("ms") // 1000).cast(pl.Int64)


def bybit_book_path(sym: str, d: str) -> Path | None:
    p = xdir() / "bybit_book_1s" / sym / f"{d}.parquet"
    if p.exists():
        return p
    q = root() / "data" / "research" / "micro_days" / d / "book_1s" / f"{sym}.parquet"
    return q if q.exists() else None


def load_bybit(sym: str, d: str) -> pl.DataFrame | None:
    """t (epoch s, state known at t), bid_b, ask_b, age_b (s since the last book update)."""
    p = bybit_book_path(sym, d)
    if p is None:
        return None
    b = pl.read_parquet(p, columns=["ts", "bid1", "ask1", "n_updates"]).with_columns(_epoch_s().alias("t"))
    b = b.filter(pl.col("bid1").is_finite() & pl.col("ask1").is_finite() & (pl.col("ask1") > pl.col("bid1")))
    last_upd = pl.when(pl.col("n_updates") > 0).then(pl.col("t")).otherwise(None).forward_fill()
    return b.with_columns((pl.col("t") - last_upd).fill_null(10**9).alias("age_b")).select("t", pl.col("bid1").alias("bid_b"), pl.col("ask1").alias("ask_b"), "age_b")


def ctrader_tick_path(name: str, d: str, side: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
    return xdir() / "ctrader_ticks" / safe / f"{d}_{side}.parquet"


def ticks_to_grid(ev_ms: np.ndarray, px: np.ndarray, t_grid: np.ndarray, prev: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """As-of: for each grid second T the last tick with event time <= T (ms <= T*1000). Returns (price, age_s)."""
    if len(ev_ms) == 0:
        return np.full(len(t_grid), np.nan), np.full(len(t_grid), np.inf)
    o = np.argsort(ev_ms, kind="stable")
    ev_ms, px = ev_ms[o], px[o]
    idx = np.searchsorted(ev_ms, t_grid * 1000, side="right") - 1
    ok = idx >= 0
    p = np.where(ok, px[np.clip(idx, 0, None)], np.nan)
    age = np.where(ok, t_grid - ev_ms[np.clip(idx, 0, None)] / 1000.0, np.inf)
    if prev is not None:
        p = np.where(ok, p, prev[0])
        age = np.where(ok, age, t_grid - prev[1])
    return p, age


def load_ctrader(name: str, d: str, t_grid: np.ndarray) -> pl.DataFrame | None:
    """bid_c, ask_c, age_c on the grid. The previous day's last tick seeds the start of the day."""
    cols = {}
    for side in ("bid", "ask"):
        p = ctrader_tick_path(name, d, side)
        if not p.exists():
            return None
        t = pl.read_parquet(p, columns=["event_ts", "price"])
        ev = t["event_ts"].dt.epoch("ms").to_numpy()
        prev = None
        pp = ctrader_tick_path(name, (date.fromisoformat(d) - timedelta(days=1)).isoformat(), side)
        if pp.exists():
            y = pl.read_parquet(pp, columns=["event_ts", "price"]).sort("event_ts").tail(1)
            if y.height:
                prev = (float(y["price"][0]), float(y["event_ts"].dt.epoch("ms")[0]) / 1000.0)
        cols[side] = ticks_to_grid(ev, t["price"].to_numpy(), t_grid, prev)
    return pl.DataFrame({"t": t_grid, "bid_c": cols["bid"][0], "ask_c": cols["ask"][0], "age_c": np.maximum(cols["bid"][1], cols["ask"][1])})


def usdt_corr(t_grid: np.ndarray) -> np.ndarray:
    """ln(USDCUSDT) as of each second (1 m close of the minute that ended at or before T); 0 where unavailable."""
    p = xdir() / "usdcusdt_1m.parquet"
    if not p.exists():
        return np.zeros(len(t_grid))
    k = pl.read_parquet(p).sort("open_s")
    end = k["open_s"].to_numpy() + 60
    idx = np.searchsorted(end, t_grid, side="right") - 1
    v = np.log(k["close"].to_numpy())
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# schedule (ProtoOASymbol.schedule: [start, end) seconds from Sunday 00:00 in scheduleTimeZone)
# ----------------------------------------------------------------------------------------------------------------------


def in_schedule(t_grid: np.ndarray, schedule: Sequence[Sequence[int]] | None, tz: str | None, holidays: Sequence[dict] | None = None) -> np.ndarray:
    if not schedule:
        return np.ones(len(t_grid), dtype=bool)
    z = ZoneInfo(tz or "UTC")
    out = np.zeros(len(t_grid), dtype=bool)
    # offsets are constant within a day except across DST switches: compute per hour block
    # wall-clock seconds from Sunday 00:00 local; the UTC offset is constant within an hour block (DST switches on the hour)
    for h0 in range(0, len(t_grid), 3600):
        blk = t_grid[h0:h0 + 3600]
        dt = datetime.fromtimestamp(int(blk[0]), UTC).astimezone(z)
        sow0 = ((dt.weekday() + 1) % 7) * DAY_S + dt.hour * 3600 + dt.minute * 60 + dt.second
        sow = np.mod(sow0 + (blk - blk[0]), 7 * DAY_S)
        m = np.zeros(len(blk), dtype=bool)
        for a, b in schedule:
            m |= (sow >= a) & (sow < b)
        out[h0:h0 + 3600] = m
    for h in holidays or ():
        hz = ZoneInfo(h.get("tz") or tz or "UTC")
        dd = date(1970, 1, 1) + timedelta(days=int(h["date_days"]))
        if h.get("recurring"):
            dd = dd.replace(year=datetime.fromtimestamp(int(t_grid[0]), UTC).year)
        start = datetime(dd.year, dd.month, dd.day, tzinfo=hz) + timedelta(seconds=int(h.get("start_s") or 0))
        stop = datetime(dd.year, dd.month, dd.day, tzinfo=hz) + timedelta(seconds=int(h.get("end_s") or DAY_S))
        out &= ~((t_grid >= start.timestamp()) & (t_grid < stop.timestamp()))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# costs
# ----------------------------------------------------------------------------------------------------------------------


def bybit_taker_bp(inst: dict[str, Any]) -> float:
    f = yaml.safe_load((root() / "config" / "fees.yaml").read_text(encoding="utf-8"))
    o = f.get("symbols", {}).get(inst["bybit"])
    if isinstance(o, dict):
        return float(o["taker_bps"])
    cls = o if isinstance(o, str) else ("crypto" if inst["class"] == "crypto" else "tradfi")
    return float(f["classes"][cls]["taker_bps"])


def ctrader_commission_bp(details: dict[str, Any] | None, price: float, ref_notional_usd: float = 10_000.0) -> float | None:
    """Per-side commission in bp of notional from ProtoOASymbol fields (None if unknown). Per-lot types use lotSize
    (cents of units); the minimum commission is applied at `ref_notional_usd` per leg."""
    if not details or details.get("commission_rate") is None:
        return None
    rate, typ = float(details["commission_rate"]), details.get("commission_type")
    units_per_lot = (details.get("lot_size_cents") or 100) / 100.0
    if typ == "USD_PER_MILLION_USD":
        bp = rate / 100.0
    elif typ == "PERCENTAGE_OF_VALUE":
        bp = rate * 100.0
    elif typ in ("USD_PER_LOT", "QUOTE_CCY_PER_LOT"):
        bp = rate / (units_per_lot * price) * 1e4 if price > 0 else math.nan
    else:
        return None
    mn = details.get("precise_min_commission")
    if mn:
        bp = max(bp, float(mn) / ref_notional_usd * 1e4)
    return bp


@dataclass
class Costs:
    bybit_taker_bp: float
    ctrader_comm_bp: float
    ctrader_comm_known: bool

    @property
    def round_trip_bp(self) -> float:
        return 2.0 * (self.bybit_taker_bp + self.ctrader_comm_bp)


# ----------------------------------------------------------------------------------------------------------------------
# grid, gaps, episodes
# ----------------------------------------------------------------------------------------------------------------------


def build_grid(b: pl.DataFrame, c: pl.DataFrame, corr: np.ndarray | float, multiplier: float, live_age_s: float, sched: np.ndarray | None = None) -> pl.DataFrame:
    g = b.join(c, on="t", how="inner").sort("t")
    n = g.height
    corr_v = np.broadcast_to(np.asarray(corr, dtype=float), (n,)) if np.ndim(corr) == 0 else np.asarray(corr)[: n]
    if sched is None:
        sched = np.ones(n, dtype=bool)
    m = multiplier
    bid_b, ask_b, bid_c, ask_c = (g[x].to_numpy() for x in ("bid_b", "ask_b", "bid_c", "ask_c"))
    with np.errstate(invalid="ignore", divide="ignore"):
        mid_b, mid_c = (bid_b + ask_b) / 2, m * (bid_c + ask_c) / 2
        gap = (np.log(mid_b / mid_c) - corr_v) * 1e4
        gap_sb = (np.log(bid_b / (m * ask_c)) - corr_v) * 1e4
        gap_bb = (np.log(m * bid_c / ask_b) + corr_v) * 1e4
        spr_b = (ask_b - bid_b) / mid_b * 1e4
        spr_c = m * (ask_c - bid_c) / mid_c * 1e4
    common = sched & np.isfinite(gap) & (g["age_b"].to_numpy() <= live_age_s) & (g["age_c"].to_numpy() <= live_age_s) & (ask_c > bid_c)
    return g.with_columns(pl.Series("gap_bp", gap), pl.Series("gap_sb_bp", gap_sb), pl.Series("gap_bb_bp", gap_bb), pl.Series("spr_b_bp", spr_b),
                          pl.Series("spr_c_bp", spr_c), pl.Series("corr_bp", corr_v * 1e4), pl.Series("common", common))


def exec_gaps(g: pl.DataFrame, fees_rt_bp: float) -> tuple[np.ndarray, np.ndarray]:
    """(executable gap after fees in bp, direction: +1 = sell Bybit / buy CFD, -1 = buy Bybit / sell CFD, 0 = none)."""
    sb = g["gap_sb_bp"].to_numpy() - fees_rt_bp
    bb = g["gap_bb_bp"].to_numpy() - fees_rt_bp
    common = g["common"].to_numpy()
    e = np.where(sb >= bb, sb, bb)
    d = np.where(sb >= bb, 1, -1)
    ok = common & np.isfinite(e) & (e > 0)
    return np.where(common, e, np.nan), np.where(ok, d, 0)


def episodes(t: np.ndarray, direction: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of consecutive seconds (t step 1) with the same nonzero direction: (start index, end index incl., dir)."""
    out = []
    i, n = 0, len(direction)
    while i < n:
        if direction[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and direction[j + 1] == direction[i] and t[j + 1] == t[j] + 1:
            j += 1
        out.append((i, j, int(direction[i])))
        i = j + 1
    return out


def arb1_day(g: pl.DataFrame, fees_rt_bp: float, min_len_s: float) -> dict[str, Any]:
    t = g["t"].to_numpy()
    e, d = exec_gaps(g, fees_rt_bp)
    n_common = int(g["common"].sum())
    eps = episodes(t, d)
    lens = np.array([j - i + 1 for i, j, _ in eps], dtype=float)
    long_eps = [(i, j, s) for (i, j, s), L in zip(eps, lens, strict=True) if L > min_len_s]
    sec_long = int(sum(j - i + 1 for i, j, _ in long_eps))
    return {
        "common_s": n_common,
        "exec_s": int((d != 0).sum()),
        "exec_share": float((d != 0).sum() / n_common) if n_common else math.nan,
        "n_episodes": len(eps),
        "n_long_episodes": len(long_eps),
        "long_share": sec_long / n_common if n_common else math.nan,
        "episode_len_s": {"median": float(np.median(lens)) if len(lens) else None, "p90": float(np.quantile(lens, 0.9)) if len(lens) else None,
                          "max": float(lens.max()) if len(lens) else None},
        "max_exec_gap_bp": float(np.nanmax(e)) if np.isfinite(e).any() else None,
        "share_sell_bybit": float(np.mean([s == 1 for *_, s in long_eps])) if long_eps else None,
        "pass": bool(n_common and sec_long / n_common >= 0.005 and long_eps),
    }


# ----------------------------------------------------------------------------------------------------------------------
# H-ARB2 convergence trade
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class Carry:
    """Funding events (epoch s, rate; positive = longs pay shorts) and the CFD swap (bp of notional per night, per
    side; negative = paid) charged at `swap_time_s` UTC seconds-of-day."""
    funding: list[tuple[int, float]] = field(default_factory=list)
    swap_long_bp: float = 0.0
    swap_short_bp: float = 0.0
    swap_time_s: int = 22 * 3600
    triple_weekday: int | None = 2      # Python weekday (Mon = 0) of triple swap; Wednesday by default


def carry_bp(c: Carry, d: int, t_in: int, t_out: int) -> tuple[float, float]:
    """(funding, swap) in bp received by the position over (t_in, t_out]. d = +1: short perp / long CFD."""
    fund = sum(rate * 1e4 * d for ts, rate in c.funding if t_in < ts <= t_out)   # short perp receives +rate
    swap = 0.0
    day = t_in - t_in % DAY_S
    while day <= t_out:
        ts = day + c.swap_time_s
        if t_in < ts <= t_out:
            k = 3.0 if c.triple_weekday is not None and datetime.fromtimestamp(ts, UTC).weekday() == c.triple_weekday else 1.0
            swap += k * (c.swap_long_bp if d == 1 else c.swap_short_bp)   # the CFD leg is long when d = +1
        day += DAY_S
    return fund, swap


def arb2_day(g: pl.DataFrame, fees_rt_bp: float, delay_s: int, carry: Carry, exit_max_s: int = EXIT_MAX_S) -> list[dict[str, Any]]:
    t = g["t"].to_numpy()
    gap = g["gap_bp"].to_numpy()
    sb, bb = g["gap_sb_bp"].to_numpy(), g["gap_bb_bp"].to_numpy()
    common = g["common"].to_numpy()
    _, d = exec_gaps(g, fees_rt_bp)
    pos = {int(x): i for i, x in enumerate(t)}
    last_common = np.where(common)[0].max() if common.any() else -1
    trades = []
    i, n = 0, len(t)
    while i < n:
        if d[i] == 0 or (i > 0 and d[i - 1] == d[i] and t[i - 1] == t[i] - 1):
            i += 1
            continue
        k_in = pos.get(int(t[i]) + delay_s)
        if k_in is None or not common[k_in]:
            i += 1
            continue
        side = int(d[i])
        # entry fill: side +1 sells Bybit at bid, buys CFD at ask -> locked spread = gap_sb at the fill second
        entry = sb[k_in] if side == 1 else bb[k_in]
        g0 = gap[i]
        j, reason = k_in + 1, "end"
        while j < n:
            if not common[j] or j == last_common:
                reason = "close"
                break
            if t[j] - t[k_in] >= exit_max_s:
                reason = "time"
                break
            if abs(gap[j]) <= abs(g0) / 2:
                reason = "halved"
                break
            j += 1
        k_out = pos.get(int(t[min(j, n - 1)]) + delay_s, min(j, n - 1)) if reason != "close" else min(j, n - 1)
        # exit: side +1 buys Bybit back at ask, sells CFD at bid -> pays -gap_bb at exit
        exit_cost = -(bb[k_out] if side == 1 else sb[k_out])
        gross = entry - exit_cost
        fund, swap = carry_bp(carry, side, int(t[k_in]), int(t[k_out]))
        net = gross - fees_rt_bp + fund + swap
        trades.append({"t_signal": int(t[i]), "t_in": int(t[k_in]), "t_out": int(t[k_out]), "side": side, "gap_signal_bp": float(g0),
                       "entry_bp": float(entry), "exit_bp": float(exit_cost), "gross_bp": float(gross), "fees_bp": fees_rt_bp,
                       "funding_bp": float(fund), "swap_bp": float(swap), "net_bp": float(net), "hold_s": int(t[k_out] - t[k_in]), "exit": reason})
        i = max(k_out + 1, i + 1)
    return trades


def cluster_boot_mean(x: np.ndarray, cl: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict[str, float]:
    x, cl = np.asarray(x, float), np.asarray(cl)
    if len(x) == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0, "clusters": 0}
    ids = np.unique(cl)
    groups = [x[cl == k] for k in ids]
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)), "n": int(len(x)), "clusters": int(len(ids))}


# ----------------------------------------------------------------------------------------------------------------------
# descriptive: basis by hour, US open / close, weekend reopen
# ----------------------------------------------------------------------------------------------------------------------


def swap_bp_per_night(details: dict[str, Any] | None, price: float) -> tuple[float, float]:
    """(long, short) CFD swap in bp of notional per night (negative = paid) from ProtoOASymbol swap fields."""
    if not details or price <= 0:
        return 0.0, 0.0
    lo, sh = float(details.get("swap_long") or 0.0), float(details.get("swap_short") or 0.0)
    typ = details.get("swap_calculation_type") or "PIPS"
    if typ == "PERCENTAGE":                      # annual percent
        f = 1e4 / 100.0 / 365.0
        return lo * f, sh * f
    unit = 10.0 ** -(details.get("pip_position") or 0) if typ == "PIPS" else 10.0 ** -(details.get("digits") or 0)
    return lo * unit / price * 1e4, sh * unit / price * 1e4


def triple_weekday(details: dict[str, Any] | None) -> int | None:
    names = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    v = (details or {}).get("swap_rollover_3days")
    return names.index(v) if v in names else 2


def us_open_close_utc(d: str) -> tuple[int, int]:
    dd = date.fromisoformat(d)
    o = datetime(dd.year, dd.month, dd.day, 9, 30, tzinfo=NY).timestamp()
    c = datetime(dd.year, dd.month, dd.day, 16, 0, tzinfo=NY).timestamp()
    return int(o), int(c)


def window_stats(g: pl.DataFrame, t0: int, before_s: int, after_s: int, fees_rt_bp: float) -> dict[str, Any] | None:
    w = g.filter((pl.col("t") >= t0 - before_s) & (pl.col("t") < t0 + after_s) & pl.col("common"))
    if w.height < 30:
        return None
    _, d = exec_gaps(w, fees_rt_bp)
    gp = w["gap_bp"].to_numpy()
    return {"n_s": w.height, "gap_median_bp": float(np.median(gp)), "abs_gap_p90_bp": float(np.quantile(np.abs(gp - np.median(gp)), 0.9)),
            "exec_share": float((d != 0).mean()), "spr_c_median_bp": float(w["spr_c_bp"].median()), "spr_b_median_bp": float(w["spr_b_bp"].median())}


def reopen_stats(g: pl.DataFrame) -> dict[str, Any] | None:
    """Weekend reopen on a reopen day (Sunday file for FX / metals / energy, Monday file for 24/5 stocks): the first
    common second of the day vs the median gap 1-2 h later (the perp kept trading through the CFD's closure)."""
    t = g["t"].to_numpy()
    idx = np.where(g["common"].to_numpy())[0]
    if len(idx) < 2:
        return None
    k = idx[0]
    t0 = int(t[k])
    later = g.filter((pl.col("t") >= t0 + 3600) & (pl.col("t") < t0 + 7200) & pl.col("common"))["gap_bp"]
    first5 = g.filter((pl.col("t") >= t0) & (pl.col("t") < t0 + 300) & pl.col("common"))["gap_bp"]
    return {"reopen_t": t0, "reopen_utc": datetime.fromtimestamp(t0, UTC).strftime("%a %H:%M:%S"), "gap_at_reopen_bp": float(g["gap_bp"][int(k)]),
            "gap_first5m_median_bp": float(first5.median()) if first5.len() else None, "gap_1to2h_median_bp": float(later.median()) if later.len() else None}


def hour_profile(g: pl.DataFrame) -> pl.DataFrame:
    return (g.filter(pl.col("common")).with_columns(((pl.col("t") % DAY_S) // 3600).alias("hour"))
            .group_by("hour").agg(pl.col("gap_bp").median().alias("gap_median_bp"), pl.col("spr_c_bp").median().alias("spr_c_bp"),
                                  pl.col("spr_b_bp").median().alias("spr_b_bp"), pl.len().alias("n_s")).sort("hour"))
