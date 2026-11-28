"""Live entry gates for `atx live` (user decisions D2 and D3, approved 2026-11-27 ~21:35 UTC): data/live/gates.json.

D2, release-hour rule (R-WS05-2, H-WS05-9, H-GEO3, H-WS14-8). No NEW entry whose stop is closer than 2 sigma (sigma of
the intent's horizon) from 5 min before to 30 min after a scheduled CPI, NFP or FOMC release. The 1h / 4h bar that holds
the release minute and the bar after it are flagged for band widening (`release_flag_bars`; the paper / packet side reads
the flag; WS14: the 80% bands cover ~45% of release hours and ~62% of the next hour).

D3, jump-intensity gate (H-WS05-6..11b). The field P(jump) is no longer a jump gate (it is a latent mixture weight and
scores worse than a constant for observable jumps). R-FLOW-2's fixed 60-min post-liquidation window becomes a Hawkes
intensity gate per symbol: while the excess jump hazard is at or above the level WS05 measured 60 min after a cascade,
new entries need a stop of at least 2 sigma.

Model (WS05 frozen fits, src/ai_trader/research/ws05_tails.py; the constants below are checked against
data/research/ws05/*.json in tests). Per symbol, on the 5m grid of flow_3s:
- A minute is valid with >= 17 of its 20 3s candles. A 5m return is invalid when any of its minutes is.
- y_t: Lee-Mykland 5m jump, |r / f| / sigma_bv > 4.30. f is the past-days hour-of-day factor; sigma_bv is bipower over
  the previous day at 30% coverage (WS05's flow conventions).
- mu_t = exp(a + b_f log f_t + b_v (log sigma_h(t) - vbar)): the M4b baseline (no excitation). sigma_h is the spot vol at
  the last hour close; vbar is its expanding mean (point in time).
- lambda_t = mu_t * exp(c0 R0_t + c1 R1_t)        M5 release marks (release 5m bin; +5..+30 min)
           + a1 E1_t + a2 E2_t                    M4b two-speed self-excitation (half-lives 4.9 and 51 min)
           + mu_t * gamma L_t / base_1m           marked-panel liquidation mark (half-life 31 min; R-FLOW-2 cluster minutes)
  with E_t = kappa E_{t-1} + y_{t-1} (5m bins) and L the same recursion on the 1m cluster minutes.
- ratio = lambda / mu ("excess jump risk" + 1). elevated = ratio >= THRESHOLD.

THRESHOLD = 1.543 is the marked-Hawkes hazard ratio 60 min after one cluster minute (liq.json
`excess_ratio_at_lag["60"]`), chosen before any replay: the level the old 60-min rule stopped protecting at. The gate
keeps every state at or above it. A lone cluster minute therefore closes at ~60 min, as before. Multi-minute cascades,
cascades with a jump (309 of 760 episodes) and waves keep it on longer, as do jump clusters (the slow 51-min kernel
holds a single 5m jump above 1.54x for ~2 h) and releases. That is WS05's ~2 h excess (+26% [1%, 56%] at 60-120 min).
Cost on Nov 2026 data: see `replay` and the diary entry "D2/D3 live gates".

Coverage. The gates cover WS05's 23-symbol pool (the marked-Hawkes fit universe) from their own flow. Any other symbol
that could have flagged a cluster minute in the last 7 h (a minute with >= 3 prints) is written as a proxy: BTC's
self-excitation and release terms plus its own liquidation term. `atx live` falls back to BTCUSDT for symbols that are
not listed: with no cluster minute their liquidation term is 0, so BTC's state is exactly their proxy state.

Fail safe. An error in any part writes that part as unsafe with a reason: jump `elevated: true`, or `calendar.ok: false`
(an incomplete calendar counts as inside a window). Stale flow (no valid minute for 20 min) is elevated. A file that is
missing or older than 5 min makes `atx live` refuse every new entry ("gates stale"); closes, stops, reconcile and the
watchdog never read it.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
GATES_PATH = ROOT / "data" / "live" / "gates.json"
CACHE_DIR = ROOT / "data" / "live" / "gates_cache"
EVENTS_PATH = ROOT / "data" / "research" / "news" / "events" / "events.parquet"
FLOW_DIR = ROOT / "data" / "lake" / "flow_3s" / "venue=bybit" / "category=linear"
LIQ_DIR = ROOT / "data" / "lake" / "liquidations" / "venue=bybit"

VERSION = "gates-v1"
M_MS, B_MS, H_MS, D_MS = 60_000, 300_000, 3_600_000, 86_400_000

# ---- D2
RELEASE_KINDS = ("cpi", "nfp", "fomc")
PRE_MS, POST_MS = 5 * M_MS, 30 * M_MS
MIN_STOP_SIGMA = 2.0
FLAG_FREQS = {"1h": H_MS, "4h": 4 * H_MS}
WINDOWS_AHEAD_MS = 14 * D_MS
CALENDAR_WARN_MS = 21 * D_MS

# ---- D3 (WS05 frozen fits: hawkes_fits.json cutoff_fit M4b / M5, liq.json marked_hawkes)
M4B_THETA = (-5.509107402213408, 0.04756067203127604, -0.6525990203918293, -2.3091196782341474, -4.563237228008463,
             -0.3489080705461079, -2.686106624321017)
M5_RELEASE = (4.334410610561216, 1.4803053185804234)
LIQ_GAMMA = 0.005973180728295226
LIQ_HALF_LIFE_MIN = 30.834309090229137
LIQ_RATIO_LAG1 = 3.045217081368285
LIQ_BASE_1M = LIQ_GAMMA / (LIQ_RATIO_LAG1 - 1.0)        # WS05's median 1m baseline, so gamma L / base = its table
THRESHOLD = 1.5429160428195066                           # excess_ratio_at_lag["60"]
LM_THR_5M = 4.3046                                       # lm_threshold(288), alpha 1% per day (lm_detect computes it)
POOL = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT",
        "DOTUSDT", "BNBUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT", "XLMUSDT",
        "SUIUSDT", "ENAUSDT", "HYPEUSDT"]
PROXY = "BTCUSDT"
HISTORY_DAYS = 90
LIQ_LOOKBACK_MS = 31 * H_MS                              # 24h trailing quantile + 6h of kernel + 1h margin
FLOW_STALE_MS = 20 * M_MS
PROJECT_MS = 6 * H_MS
MAX_AGE_MS = 5 * M_MS                                    # atx live: older -> "gates stale"


def _m4b():
    a, bf, bv, la1, la2, lb1, lb2 = M4B_THETA
    k1, k2 = math.exp(-math.exp(lb1)), math.exp(-math.exp(lb2))
    return a, bf, bv, math.exp(la1), math.exp(la2), k1, k2


A0, B_F, B_V, ALPHA1, ALPHA2, KAPPA1, KAPPA2 = _m4b()
KAPPA_LIQ = 0.5 ** (1.0 / LIQ_HALF_LIFE_MIN)


def iso(ms: int | None) -> str | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_ms() -> int:
    return int(time.time() * 1000)


# ============================================================================ D2: release windows and flag bars


def load_events(path: Path = EVENTS_PATH) -> pl.DataFrame:
    """CPI / NFP / FOMC rows: kind, name, release_ms, known_ms."""
    ev = pl.read_parquet(path).filter(pl.col("kind").is_in(list(RELEASE_KINDS)))
    return ev.select("kind", "name", pl.col("scheduled_ts").dt.epoch("ms").alias("release_ms"),
                     pl.col("schedule_known_ts").dt.epoch("ms").alias("known_ms")).sort("release_ms")


def known_releases(ev: pl.DataFrame, as_of: int) -> pl.DataFrame:
    """Releases whose schedule is known at as_of (point in time)."""
    return ev.filter(pl.col("known_ms") <= as_of)


def release_windows(ev: pl.DataFrame, as_of: int, ahead_ms: int = WINDOWS_AHEAD_MS) -> list[dict]:
    """[release - 5 min, release + 30 min] windows that are active now or start within `ahead_ms`."""
    out = []
    for r in known_releases(ev, as_of).iter_rows(named=True):
        t = int(r["release_ms"])
        a, b = t - PRE_MS, t + POST_MS
        if b >= as_of and a <= as_of + ahead_ms:
            out.append({"event": r["name"], "kind": r["kind"], "release_ms": t, "release_utc": iso(t), "start_ms": a, "end_ms": b,
                        "active": a <= as_of <= b})
    return out


def release_flag_bars(ev: pl.DataFrame, as_of: int, ahead_ms: int = WINDOWS_AHEAD_MS) -> list[dict]:
    """For each known release within (as_of - 8 h, as_of + ahead]: the 1h / 4h bar holding the release minute and the
    next bar, flagged for band widening (WS14 R-WS14 candidate; R-WS05-2)."""
    out = []
    for r in known_releases(ev, as_of).iter_rows(named=True):
        t = int(r["release_ms"])
        if not (as_of - 8 * H_MS < t <= as_of + ahead_ms):
            continue
        for freq, step in FLAG_FREQS.items():
            b0 = t // step * step
            for which, o in (("release", b0), ("next", b0 + step)):
                if o + step <= as_of:
                    continue
                out.append({"event": r["name"], "kind": r["kind"], "release_ms": t, "freq": freq, "bar": which, "bar_open_ms": o,
                            "bar_close_ms": o + step, "bar_open_utc": iso(o), "active": o <= as_of < o + step})
    return out


def calendar_state(ev: pl.DataFrame, as_of: int) -> dict:
    """ok = every kind has at least one release scheduled after as_of in the table (else the next one is unknown and
    the release gate cannot tell whether a window is near: atx live then treats every moment as inside a window)."""
    cov, warn, missing = {}, [], []
    for k in RELEASE_KINDS:
        nxt = ev.filter((pl.col("kind") == k) & (pl.col("release_ms") > as_of))["release_ms"]
        last = ev.filter(pl.col("kind") == k)["release_ms"]
        cov[k] = int(last.max()) if last.len() else None
        if nxt.len() == 0:
            missing.append(k)
        elif int(last.max()) < as_of + CALENDAR_WARN_MS:
            warn.append(f"{k}: the calendar ends {iso(int(last.max()))}; add the next schedule to src/ai_trader/news/events.py")
    ok = not missing
    return {"ok": ok, "reason": None if ok else f"no future {', '.join(missing)} release in {EVENTS_PATH.name}: next release unknown",
            "covered_until_ms": cov, "warn": warn}


# ============================================================================ D3: jump intensity (numpy core)


def _lf(src: np.ndarray, k: float) -> np.ndarray:
    """E_t = k E_{t-1} + src_{t-1}."""
    from scipy import signal
    return signal.lfilter([0.0, 1.0], [1.0, -k], np.asarray(src, float))


def release_marks(bin_ts: np.ndarray, releases_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R0 = 1 on the 5m bin holding the release, R1 = 1 on the next five bins (+5..+30 min), as ws05_tails.release_bins."""
    r0, r1 = np.zeros(len(bin_ts)), np.zeros(len(bin_ts))
    if len(bin_ts) == 0:
        return r0, r1
    t0 = int(bin_ts[0])
    for te in np.asarray(releases_ms, dtype=np.int64):
        i = int((te - t0) // B_MS)
        if 0 <= i < len(bin_ts):
            r0[i] = 1.0
        lo, hi = max(i + 1, 0), min(i + 1 + 5, len(bin_ts))
        if lo < hi:
            r1[lo:hi] = 1.0
    return r0, r1


def release_mult(t_ms: int, releases_ms: np.ndarray) -> float:
    """exp(c0 R0 + c1 R1) for the 5m bin holding t_ms."""
    b = t_ms // B_MS * B_MS
    for te in np.asarray(releases_ms, dtype=np.int64):
        tb = int(te) // B_MS * B_MS
        if b == tb:
            return math.exp(M5_RELEASE[0])
        if tb < b <= tb + 5 * B_MS:
            return math.exp(M5_RELEASE[1])
    return 1.0


def five_minute(minute: np.ndarray, close: np.ndarray, n3: np.ndarray, end_min: int):
    """1m flow candles -> regular 5m grid (bar open ms, log return) up to and including the 5m bin holding end_min;
    a minute is valid with n3 >= 17, a 5m return needs all five 1m returns (WS05 flow_series)."""
    from ai_trader.research.ws05_tails import grid_returns
    keep = minute <= end_min
    minute, close, n3 = minute[keep], close[keep], n3[keep]
    last_min = (end_min // 5) * 5 + 4
    ts = np.append(minute.astype(np.int64) * M_MS, last_min * M_MS) if minute[-1] < last_min else minute.astype(np.int64) * M_MS
    cl = np.append(close, np.nan) if minute[-1] < last_min else close
    ok = np.append(n3 >= 17, False) if minute[-1] < last_min else n3 >= 17
    g, r = grid_returns(ts, cl, M_MS, ok)
    off = int((-(g[0] // M_MS)) % 5)
    g2, r2 = g[off:], r[off:]
    m = len(r2) // 5
    return g2[: m * 5: 5], r2[: m * 5].reshape(m, 5).sum(1)


def self_series(bin_ts: np.ndarray, r5: np.ndarray, name: str = "flow5m") -> dict:
    """LM 5m jumps and the M4b baseline / self-excitation per 5m bin (point in time: every quantity at bin t uses
    bars before t only)."""
    from ai_trader.research.ws05_tails import Series, lm_detect
    s = lm_detect(Series(f"{name}_flow5m", bin_ts, r5, 288))
    y = (s.jump & np.isfinite(s.z)).astype(float)
    lsig = np.log(s.sig)
    vh = np.where(bin_ts % H_MS == 0, lsig, np.nan)
    vh = pl.Series(vh).fill_nan(None).forward_fill().to_numpy().astype(float)
    valid = np.isfinite(s.z) & np.isfinite(vh)
    cs = np.concatenate([[0.0], np.cumsum(np.where(valid, vh, 0.0))])[:-1]
    cn = np.concatenate([[0], np.cumsum(valid)])[:-1]
    vbar = np.where(cn > 0, cs / np.maximum(cn, 1), vh)
    mu = np.exp(A0 + B_F * np.log(s.f) + B_V * (vh - vbar))
    e1, e2 = _lf(y, KAPPA1), _lf(y, KAPPA2)
    return {"ts": bin_ts, "y": y, "mu": mu, "e1": e1, "e2": e2, "self_abs": ALPHA1 * e1 + ALPHA2 * e2, "f": s.f, "vh": vh, "vbar": vbar}


def liq_kernel(lq: np.ndarray) -> np.ndarray:
    """L_m on the minute grid (cluster minutes before m)."""
    return _lf(np.nan_to_num(np.asarray(lq, float)), KAPPA_LIQ)


def ratio_now(ss: dict, b: int, liq_l: float, t_ms: int, releases_ms: np.ndarray) -> dict:
    mu = float(ss["mu"][b])
    rel = release_mult(t_ms, releases_ms)
    se = float(ss["self_abs"][b]) / mu if mu > 0 else float("nan")
    le = LIQ_GAMMA * liq_l / LIQ_BASE_1M
    return {"mu": mu, "release": rel, "self": se, "liq": le, "ratio": rel + se + le}


def project(ss: dict, b: int, liq_l: float, t_ms: int, releases_ms: np.ndarray, horizon_ms: int = PROJECT_MS) -> dict:
    """With no new events: ratio path per minute from t_ms, the first minute it is below THRESHOLD, and P(>= 1 5m LM
    jump in the next hour) (exact under the model for the next 12 bins, mu held at its current value)."""
    mu = float(ss["mu"][b])
    e1, e2 = float(ss["e1"][b]), float(ss["e2"][b])
    b0 = t_ms // B_MS
    rl = np.asarray(releases_ms, dtype=np.int64)
    releases_ms = rl[(rl > t_ms - 2 * H_MS) & (rl < t_ms + horizon_ms + H_MS)]
    until = None
    for k in range(0, horizon_ms // M_MS + 1):
        t = t_ms + k * M_MS
        j = t // B_MS - b0
        r = release_mult(t, releases_ms) + (ALPHA1 * e1 * KAPPA1 ** j + ALPHA2 * e2 * KAPPA2 ** j) / mu + LIQ_GAMMA * liq_l * KAPPA_LIQ ** k / LIQ_BASE_1M
        if r < THRESHOLD:
            until = t // M_MS * M_MS
            break
    lam = 0.0
    for j in range(12):
        tb = (b0 + j) * B_MS
        k = max((tb - t_ms) // M_MS, 0)
        lam += mu * (release_mult(tb, releases_ms) + LIQ_GAMMA * liq_l * KAPPA_LIQ ** k / LIQ_BASE_1M) + ALPHA1 * e1 * KAPPA1 ** j + ALPHA2 * e2 * KAPPA2 ** j
    return {"until_ms": until if until is not None else t_ms + horizon_ms, "p_jump_1h": -math.expm1(-lam), "p_jump_1h_base": -math.expm1(-12 * mu)}


def jump_state(ss: dict, liq_l: np.ndarray | None, liq_t0_min: int | None, t_ms: int, releases_ms: np.ndarray, *,
               last_valid_min: int | None, liq_minutes_6h: int = 0, source: str = "own") -> dict:
    """Gate state at t_ms for one symbol. ss = self_series of its own (or the proxy's) 5m grid."""
    b = int(np.searchsorted(ss["ts"], t_ms // B_MS * B_MS))
    if b >= len(ss["ts"]) or int(ss["ts"][b]) != t_ms // B_MS * B_MS:
        raise ValueError("the 5m grid does not reach the current bin")
    m = t_ms // M_MS
    L = 0.0
    if liq_l is not None and liq_t0_min is not None and 0 <= m - liq_t0_min < len(liq_l):
        L = float(liq_l[m - liq_t0_min])
    c = ratio_now(ss, b, L, t_ms, releases_ms)
    reasons = []
    if not (np.isfinite(c["mu"]) and c["mu"] > 0 and np.isfinite(c["ratio"])):
        reasons.append("no spot vol / baseline (too little valid flow in the previous day)")
    if last_valid_min is None or t_ms - (last_valid_min + 1) * M_MS > FLOW_STALE_MS:
        reasons.append(f"flow stale: last valid minute {iso(None if last_valid_min is None else last_valid_min * M_MS)}")
    elevated = bool(reasons) or c["ratio"] >= THRESHOLD
    pr = project(ss, b, L, t_ms, releases_ms) if not reasons else {"until_ms": t_ms + PROJECT_MS, "p_jump_1h": None, "p_jump_1h_base": None}
    y = ss["y"][max(b - 72, 0): b]
    return {"intensity": _r(c["mu"] * c["ratio"]), "baseline": _r(c["mu"]), "ratio": _r(c["ratio"], 4), "elevated": elevated,
            "until_ms": pr["until_ms"] if elevated else None, "until_utc": iso(pr["until_ms"]) if elevated else None,
            "p_jump_1h": _r(pr["p_jump_1h"]), "p_jump_1h_base": _r(pr["p_jump_1h_base"]),
            "components": {"release": _r(c["release"], 4), "self": _r(c["self"], 4), "liq": _r(c["liq"], 4)},
            "jumps_5m_6h": int(y.sum()), "liq_cluster_minutes_6h": int(liq_minutes_6h), "last_valid_minute_ms": None if last_valid_min is None else last_valid_min * M_MS,
            "source": source, "reason": "; ".join(reasons) if reasons else None}


def _r(x, nd: int = 6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def unsafe_jump(reason: str) -> dict:
    return {"intensity": None, "baseline": None, "ratio": None, "elevated": True, "until_ms": None, "until_utc": None,
            "p_jump_1h": None, "p_jump_1h_base": None, "components": None, "source": "error", "reason": reason}


# ============================================================================ data (flow_3s -> 1m, liquidations)


def _duck():
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='600MB'")
    return con


def _q(con, files: list[str], symbols: list[str]) -> pl.DataFrame:
    fl = ", ".join("'" + f.replace("\\", "/") + "'" for f in files)
    syms = ", ".join(f"'{s}'" for s in symbols)
    q = f"""SELECT symbol, epoch_ms(event_ts)::BIGINT ts, close, epoch_ms(known_ts)::BIGINT k
            FROM read_parquet([{fl}], union_by_name=true) WHERE symbol IN ({syms})"""
    return pl.from_arrow(con.sql(q).to_arrow_table())


def to_minutes(rows: pl.DataFrame) -> pl.DataFrame:
    """3s candles (symbol, ts, close, k) -> 1m (symbol, minute, close, n3); latest known row per (symbol, ts)."""
    if rows.height == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "minute": pl.Int64, "close": pl.Float64, "n3": pl.UInt32})
    d = rows.sort("k").unique(["symbol", "ts"], keep="last").sort("symbol", "ts")
    return (d.with_columns((pl.col("ts") // M_MS).alias("minute")).group_by("symbol", "minute")
            .agg(pl.col("close").drop_nulls().last().alias("close"), pl.len().alias("n3")).sort("symbol", "minute"))


class FlowCache:
    """1m flow candles per symbol, point in time. Complete past days are cached under data/live/gates_cache/; the
    last two days are read incrementally (lake part files are immutable)."""

    def __init__(self, symbols: list[str], flow_dir: Path = FLOW_DIR, cache_dir: Path = CACHE_DIR):
        self.symbols, self.flow_dir, self.cache_dir = list(symbols), flow_dir, cache_dir
        self.days: dict[str, pl.DataFrame] = {}
        self.rows: dict[str, pl.DataFrame] = {}
        self.seen: dict[str, set[str]] = {}
        self.con = None

    def _con(self):
        if self.con is None:
            self.con = _duck()
        return self.con

    def _key(self) -> str:
        import hashlib
        return hashlib.sha256(",".join(sorted(self.symbols)).encode()).hexdigest()[:8]

    def minutes(self, as_of: int) -> pl.DataFrame:
        today = as_of // D_MS
        dates = sorted(p.name[5:] for p in self.flow_dir.glob("date=*") if p.is_dir())
        lo = datetime.fromtimestamp((today - HISTORY_DAYS) * D_MS / 1000, UTC).date().isoformat()
        dates = [d for d in dates if d >= lo and int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp() * 1000) <= as_of]
        out = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for d in dates:
            d_ms = int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp() * 1000)
            complete = as_of > d_ms + D_MS + 30 * M_MS
            if complete:
                if d not in self.days:
                    cp = self.cache_dir / f"flow1m_{d}_{self._key()}.parquet"
                    if cp.is_file():
                        self.days[d] = pl.read_parquet(cp)
                    else:
                        files = [str(f) for f in (self.flow_dir / f"date={d}").glob("*.parquet")]
                        m1 = to_minutes(_q(self._con(), files, self.symbols)) if files else to_minutes(pl.DataFrame())
                        tmp = cp.with_suffix(".tmp")
                        m1.write_parquet(tmp)
                        os.replace(tmp, cp)
                        self.days[d] = m1
                    self.rows.pop(d, None)
                    self.seen.pop(d, None)
                out.append(self.days[d])
            else:
                files = sorted(str(f) for f in (self.flow_dir / f"date={d}").glob("*.parquet"))
                new = [f for f in files if f not in self.seen.setdefault(d, set())]
                if new:
                    add = _q(self._con(), new, self.symbols)
                    self.rows[d] = pl.concat([self.rows[d], add]) if d in self.rows else add
                    self.seen[d].update(new)
                if d in self.rows:
                    out.append(to_minutes(self.rows[d].filter(pl.col("k") <= as_of)))
        for d in [d for d in self.days if d < lo]:
            del self.days[d]
        for f in self.cache_dir.glob("flow1m_*.parquet"):
            if f.name[7:17] < lo:
                f.unlink(missing_ok=True)
        if not out:
            return to_minutes(pl.DataFrame())
        return pl.concat([o.with_columns(pl.col("n3").cast(pl.UInt32)) for o in out]).filter(pl.col("minute") * M_MS < as_of)


class LiqCache:
    """Bybit liquidation prints of the last LIQ_LOOKBACK_MS, read incrementally from the lake's immutable part files
    (year=/month= partitions), point in time (known_ts <= as_of), one row per primary key."""

    def __init__(self, liq_dir: Path = LIQ_DIR):
        self.liq_dir, self.seen, self.rows, self.con = liq_dir, set(), None, None

    def read(self, as_of: int, start: int) -> pl.DataFrame:
        months = set()
        for t in (start, as_of):
            d = datetime.fromtimestamp(t / 1000, UTC)
            months.add(self.liq_dir / f"year={d.year}" / f"month={d.month}")
        new = sorted(str(f) for m in months if m.is_dir() for f in m.glob("*.parquet") if str(f) not in self.seen)
        if new:
            if self.con is None:
                self.con = _duck()
            fl = ", ".join("'" + f.replace("\\", "/") + "'" for f in new)
            q = f"""SELECT symbol, epoch_ms(event_ts)::BIGINT ts, side, price, size, epoch_ms(known_ts)::BIGINT k
                    FROM read_parquet([{fl}], union_by_name=true) WHERE epoch_ms(event_ts) >= {start - H_MS}"""
            add = pl.from_arrow(self.con.sql(q).to_arrow_table())
            self.rows = add if self.rows is None else pl.concat([self.rows, add])
            self.seen.update(new)
        if self.rows is None or self.rows.height == 0:
            return pl.DataFrame(schema={"venue": pl.Utf8, "symbol": pl.Utf8, "event_ts": pl.Datetime("ms", "UTC"), "side": pl.Utf8,
                                        "price": pl.Float64, "size": pl.Float64})
        self.rows = self.rows.filter(pl.col("ts") >= start - H_MS)
        d = (self.rows.filter((pl.col("k") <= as_of) & (pl.col("ts") >= start)).sort("k")
             .unique(["symbol", "ts", "side", "price", "size"], keep="last").sort("symbol", "ts"))
        return d.select(pl.lit("bybit").alias("venue"), "symbol", pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC").alias("event_ts"),
                        "side", "price", "size")


def liquidation_flags(liq: pl.DataFrame, alive_min: np.ndarray, t0_min: int, t1_min: int, score_from_min: int | None = None) -> dict[str, np.ndarray]:
    """R-FLOW-2 cluster minutes per symbol on the minute grid [t0_min, t1_min) (features.flow_anomaly definition).
    alive_min: minutes in which the collector wrote any flow candle; other minutes are unknown, not zero.
    score_from_min: flag only from this minute on (earlier minutes still feed the trailing 24h quantile)."""
    from ai_trader.features.flow_anomaly import liquidation_clusters, liquidation_minutes
    out = {}
    if liq.height == 0:
        return out
    lm = liquidation_minutes(liq).with_columns(pl.col("event_ts").cast(pl.Datetime("ms", "UTC")))
    if score_from_min is not None:   # a cluster minute needs >= 3 prints: symbols without one in the scored span cannot flag
        sf = datetime.fromtimestamp(score_from_min * 60, UTC)
        can = lm.filter((pl.col("event_ts") >= sf) & (pl.col("liq_n") >= 3))["symbol"].unique().to_list()
        lm = lm.filter(pl.col("symbol").is_in(can))
    grid_min = np.arange(t0_min, t1_min, dtype=np.int64)
    alive = pl.DataFrame({"m": pl.Series(np.asarray(alive_min, dtype=np.int64)).unique()}).with_columns(pl.lit(1, pl.UInt32).alias("n3s"))
    for sym in sorted(lm["symbol"].unique().to_list()):
        g = pl.DataFrame({"symbol": [sym] * len(grid_min), "m": grid_min}).join(alive, on="m", how="left")
        g = g.with_columns(pl.from_epoch(pl.col("m") * M_MS, time_unit="ms").dt.replace_time_zone("UTC").cast(pl.Datetime("ms", "UTC")).alias("event_ts"))
        g = g.join(lm.filter(pl.col("symbol") == sym).drop("symbol"), on="event_ts", how="left").drop("m")
        cl = liquidation_clusters(g, score_from=None if score_from_min is None else datetime.fromtimestamp(score_from_min * 60, UTC))
        out[sym] = cl["liq_cluster"].fill_null(False).to_numpy().astype(float)
    return out


# ============================================================================ assemble / write


def build(as_of: int | None = None, *, flow: FlowCache | None = None, events_path: Path = EVENTS_PATH, symbols: list[str] | None = None,
          read_liq=None) -> dict:
    """The full gates document at as_of. Never raises: a failing part is written unsafe with a reason."""
    t = now_ms() if as_of is None else int(as_of)
    syms = list(symbols or POOL)
    doc: dict = {"written_ms": None, "written_utc": None, "as_of_ms": t, "as_of_utc": iso(t), "version": VERSION, "max_age_ms": MAX_AGE_MS,
                 "min_stop_sigma": MIN_STOP_SIGMA, "release_windows": [], "release_flag_bars": [],
                 "calendar": {"ok": False, "reason": "not computed"}, "jump": {},
                 "jump_model": {"threshold": THRESHOLD, "fallback_symbol": PROXY, "spec": "M4b two-speed self-excitation + M5 release marks + marked-panel liquidation mark (ws05_tails)",
                                "threshold_source": "data/research/ws05/liq.json marked_hawkes.excess_ratio_at_lag[60]"},
                 "errors": []}
    releases = np.zeros(0, dtype=np.int64)
    try:
        ev = load_events(events_path)
        doc["release_windows"] = release_windows(ev, t)
        doc["release_flag_bars"] = release_flag_bars(ev, t)
        doc["calendar"] = calendar_state(ev, t)
        releases = known_releases(ev, t)["release_ms"].to_numpy().astype(np.int64)
    except Exception as e:  # fail safe: an unreadable calendar counts as inside a window
        doc["calendar"] = {"ok": False, "reason": f"events calendar unreadable: {e!r}"}
        doc["errors"].append(f"calendar: {e!r}")
    try:
        doc["jump"] = jump_states(t, syms, releases, flow or FlowCache(syms), read_liq)
    except Exception as e:
        doc["errors"].append(f"jump: {e!r}")
        doc["jump"] = {s: unsafe_jump(f"jump model failed: {e!r}") for s in syms}
        doc["traceback"] = traceback.format_exc(limit=4)
    return doc


def jump_states(t: int, syms: list[str], releases: np.ndarray, flow: FlowCache, read_liq=None) -> dict:
    if read_liq is None:
        lc = getattr(flow, "liq_cache", None)
        if lc is None:
            lc = LiqCache()
            try:
                flow.liq_cache = lc
            except AttributeError:
                pass
        read_liq = lc.read
    m1 = flow.minutes(t)
    end_min = t // M_MS
    t0_min = (t - LIQ_LOOKBACK_MS) // M_MS
    alive = m1.filter(pl.col("minute") >= t0_min)["minute"].to_numpy()
    liq = read_liq(t, t0_min * M_MS)
    # the grid holds the current minute (L(now) sees every minute before it); flags only for the last 7 h (kernel 2^-13.6 after)
    flags = liquidation_flags(liq, alive, t0_min, end_min + 1, score_from_min=end_min - 420)
    liq_l = {s: liq_kernel(v) for s, v in flags.items()}
    six = (end_min + 1 - t0_min) - 360
    out, ss_proxy, lv_proxy = {}, None, None
    for sym in syms:
        try:
            d = m1.filter(pl.col("symbol") == sym).sort("minute")
            if d.height < 288 * 5:
                out[sym] = unsafe_jump(f"too little flow history ({d.height} minutes)")
                continue
            mn, cl, n3 = d["minute"].to_numpy().astype(np.int64), d["close"].to_numpy().astype(float), d["n3"].to_numpy().astype(int)
            bt, r5 = five_minute(mn, cl, n3, end_min)
            ss = self_series(bt, r5, sym)
            valid = mn[n3 >= 17]
            lv = int(valid.max()) if len(valid) else None
            fl = flags.get(sym)
            out[sym] = jump_state(ss, liq_l.get(sym), t0_min, t, releases, last_valid_min=lv,
                                  liq_minutes_6h=int(fl[six:].sum()) if fl is not None else 0, source="own")
            if sym == PROXY:
                ss_proxy, lv_proxy = ss, lv
        except Exception as e:
            out[sym] = unsafe_jump(f"{sym}: {e!r}")
    # proxies: symbols with liquidation prints in the window but no own flow model -> BTC self / release + own liquidations
    for sym, fl in flags.items():
        if sym in out:
            continue
        if ss_proxy is None:
            out[sym] = unsafe_jump(f"proxy {PROXY} unavailable")
            continue
        try:
            out[sym] = jump_state(ss_proxy, liq_l[sym], t0_min, t, releases, last_valid_min=lv_proxy, liq_minutes_6h=int(fl[six:].sum()),
                                  source=f"proxy:{PROXY}+own_liq")
        except Exception as e:
            out[sym] = unsafe_jump(f"{sym} proxy: {e!r}")
    return out


def write_atomic(doc: dict, path: Path = GATES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = now_ms()
    doc["written_ms"], doc["written_utc"] = w, iso(w)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    for i in range(20):  # Windows: the reader may hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, path)


def run_once(flow: FlowCache | None = None, path: Path = GATES_PATH, symbols: list[str] | None = None) -> dict:
    doc = build(flow=flow, symbols=symbols)
    write_atomic(doc, path)
    return doc


def loop(interval_s: int = 60, path: Path = GATES_PATH, symbols: list[str] | None = None, log=print) -> None:
    """Every interval_s on the wall clock (at second 5 of each minute for 60 s). Suspend-safe: sleeps in short steps and
    re-reads the wall clock, so a resume runs immediately instead of finishing a long monotonic sleep."""
    flow = FlowCache(list(symbols or POOL))
    while True:
        t0 = time.time()
        try:
            doc = run_once(flow, path, symbols)
            el = sorted(s for s, j in doc["jump"].items() if j.get("elevated"))
            act = [w["event"] for w in doc["release_windows"] if w.get("active")]
            log(f"{iso(doc['written_ms'])} ok {time.time() - t0:.1f}s | windows active {act or '-'} | calendar {'ok' if doc['calendar'].get('ok') else 'NOT OK'}"
                f" | elevated {len(el)}/{len(doc['jump'])} {el[:8]}{' | errors ' + str(doc['errors']) if doc['errors'] else ''}")
        except Exception as e:  # the write itself failed: the file goes stale and atx live refuses entries (fail closed)
            log(f"{iso(now_ms())} FAILED {e!r}")
        nxt = (math.floor(time.time() / interval_s) + 1) * interval_s + 5
        while time.time() < nxt:
            time.sleep(min(5.0, max(nxt - time.time(), 0.1)))


# ============================================================================ packet helper (replaces the field P(jump) sentence)


def load(path: Path = GATES_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def packet_line(symbols: list[str] | None = None, *, gates: dict | None = None, now: int | None = None, path: Path = GATES_PATH) -> str:
    """One packet line with the live gate state (D2 / D3). It replaces the field card's "jump probability x%" sentence,
    which is a latent mixture weight, not the chance of a visible jump (H-WS05-6, R-WS05-1)."""
    g = gates if gates is not None else load(path)
    t = now_ms() if now is None else now
    if not g or not g.get("written_ms"):
        return "Live gates: data/live/gates.json missing: atx live refuses every new entry (gates stale)."
    age = t - int(g["written_ms"])
    if age > MAX_AGE_MS:
        return f"Live gates STALE (written {age / 60000:.0f} min ago): atx live refuses every new entry until scripts/live_gates.py --loop runs."
    bits = []
    act = [w for w in g.get("release_windows", []) if w["start_ms"] <= t <= w["end_ms"]]
    if act:
        w = act[0]
        bits.append(f"RELEASE WINDOW {w['event']} {iso(w['release_ms'])[11:16]}Z ({iso(w['start_ms'])[11:16]}-{iso(w['end_ms'])[11:16]}Z): no new entry with a stop < {g.get('min_stop_sigma', 2)} sigma (D2)")
    else:
        nxt = sorted((w for w in g.get("release_windows", []) if w["start_ms"] > t), key=lambda w: w["start_ms"])
        bits.append(f"next release {nxt[0]['event']} {iso(nxt[0]['release_ms'])[:16].replace('T', ' ')}Z" if nxt else "no release in the next 14 days")
    fb = [b for b in g.get("release_flag_bars", []) if b["bar_open_ms"] <= t < b["bar_close_ms"]]
    if fb:
        bits.append("bands: widen this " + "/".join(sorted({f"{b['freq']} ({b['bar']} bar of {b['event']})" for b in fb})))
    cal = g.get("calendar", {})
    if not cal.get("ok"):
        bits.append(f"release calendar NOT OK ({cal.get('reason')}): stops < 2 sigma refused")
    jm = g.get("jump", {})
    want = symbols or [s for s in ("BTCUSDT", "ETHUSDT") if s in jm]
    el = sorted((s for s, j in jm.items() if j.get("elevated")), key=lambda s: -(jm[s].get("ratio") or 99))
    for s in want:
        j = jm.get(s) or jm.get(g.get("jump_model", {}).get("fallback_symbol", PROXY))
        if not j:
            bits.append(f"{s} jump state unknown (counts as elevated)")
            continue
        p, pb = j.get("p_jump_1h"), j.get("p_jump_1h_base")
        pj = f", P(>=1 5m jump next 1h) {100 * p:.1f}% vs {100 * pb:.1f}% base" if p is not None and pb is not None else ""
        st = (f"ELEVATED until {j['until_utc'][11:16]}Z" if j.get("until_utc") else "ELEVATED") if j.get("elevated") else "normal"
        bits.append(f"{s} jump intensity x{j['ratio']:.2f} {st}{pj}" if j.get("ratio") is not None else f"{s} jump gate {st} ({j.get('reason')})")
    others = [s for s in el if s not in want]
    if others:
        bits.append(f"jump gate elevated also on {', '.join(others[:6])}{' +' + str(len(others) - 6) if len(others) > 6 else ''}")
    return ("Live gates (WS05 Hawkes, not the field's P(jump); stops < 2 sigma refused while elevated, D3): " + "; ".join(bits) + ".")


# ============================================================================ replay (documentation of the gate's cost)


def replay(m1: pl.DataFrame, liq: pl.DataFrame, releases: np.ndarray, symbols: list[str] | None = None) -> pl.DataFrame:
    """Per symbol and minute: the gate ratio and elevated flag on a historical 1m flow table (symbol, minute, close, n3),
    computed point in time with the live formulas (vectorised), plus the R-FLOW-2 60-min window for comparison."""
    rows = []
    syms = symbols or sorted(m1["symbol"].unique().to_list())
    t0_min, t1_min = int(m1["minute"].min()), int(m1["minute"].max()) + 1
    flags = liquidation_flags(liq.filter(pl.col("symbol").is_in(syms)), m1["minute"].unique().to_numpy(), t0_min, t1_min)
    for sym in syms:
        d = m1.filter(pl.col("symbol") == sym).sort("minute")
        if d.height < 288 * 5:
            continue
        mn, n3 = d["minute"].to_numpy().astype(np.int64), d["n3"].to_numpy().astype(int)
        bt, r5 = five_minute(mn, d["close"].to_numpy().astype(float), n3, t1_min - 1)
        ss = self_series(bt, r5, sym)
        r0, r1 = release_marks(bt, releases)
        grid = np.arange(t0_min, t1_min, dtype=np.int64)
        b = np.clip(np.searchsorted(bt, grid * M_MS // B_MS * B_MS), 0, len(bt) - 1)
        fl = flags.get(sym, np.zeros(len(grid)))
        L = liq_kernel(fl)
        rel = np.exp(M5_RELEASE[0] * r0[b] + M5_RELEASE[1] * r1[b])
        ratio = rel + ss["self_abs"][b] / ss["mu"][b] + LIQ_GAMMA * L / LIQ_BASE_1M
        valid_min = np.isin(grid, mn[n3 >= 17])
        last_cl = np.where(fl > 0, grid, -10 ** 12)
        last_cl = np.maximum.accumulate(last_cl)
        rows.append(pl.DataFrame({"symbol": [sym] * len(grid), "minute": grid, "ratio": ratio, "elevated": ratio >= THRESHOLD,
                                  "liq_cluster": fl > 0, "rflow2_60": (grid - last_cl) <= 60, "valid": valid_min & np.isfinite(ratio)}))
    return pl.concat(rows) if rows else pl.DataFrame()


__all__ = ["GATES_PATH", "THRESHOLD", "VERSION", "build", "calendar_state", "jump_state", "load", "loop", "packet_line",
           "release_flag_bars", "release_windows", "replay", "run_once", "write_atomic"]
