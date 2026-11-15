"""FDR-controlled hypothesis mining on the micro aggregates (H-MINE1; overnight program 2026-11-12, agent E).

Protocol (pre-registered in research/hypotheses.yaml, H-MINE1):

1. DISCOVERY on 16 days (batches 1-3 + the 2026-11-10 pilot). Every effect of a family declared below, before any
   statistic is computed, is estimated with day-clustered inference (one estimate per day, t-test over days, df = D - 1)
   and Benjamini-Hochberg is applied at FDR 10% over the WHOLE family (effects that cannot be estimated get p = 1).
2. The discovery list (effect id, direction, size) is frozen in data/research/mining/discoveries_frozen.json and in a
   diary entry BEFORE any confirmation day is read. ``load_day`` refuses confirmation days while the file is missing.
3. CONFIRMATION on batches 4-6 (15 days): each discovery is tested once, one-sided in its discovered direction,
   p < 0.05, day-clustered. H-MINE1 success: replication rate >= 50% and vol-effect rate > direction-effect rate.

State tables (per symbol-day, built from data/research/micro_days/<date>/trades_1s and book_1s, read-only):

T1, one row per UTC minute m (decision time = end of minute m, i.e. (m + 1) x 60 s):
  flow_q   signed-flow quintile 0..4: notional imbalance (buy - sell) / (buy + sell) of minute m, ranked (mid-rank) among
           the same symbol-day's PAST minutes 0..m-1 (at least WARMUP_MIN past values; no future data)
  vol_t    recent-vol tercile 0..2: log of the sum of 5 s last-trade squared log returns over minutes m-14..m, ranked among
           the symbol-day's past values of the same quantity (normalised per symbol-day by construction)
  spread   L2-book symbols only: 1 = free (time-average spread over minute m > 1.5 ticks), 0 = tick-bound, -1 = no book
  hour     UTC block of minute m: 0 = [00, 06), 1 = [06, 12), 2 = [12, 16), 3 = [16, 24)
  cls      symbol class (microstructure_laws_study.asset_class): major / alt / tradfi
  BTC/ETH/SOL flow_q and vol_t of the same minute m (for alts; the leader's own past-only states)
  targets  a1 = |r(m, m+1]| and a5 = |r(m, m+5]| (last-trade closes), s1 = r(m, m+1], s5 = r(m, m+5]. Each target is
           normalised over the symbol-day's analysis rows (|r| / mean |r|, r / sd r) and then demeaned within the
           symbol-day: an effect is the mean target in a state cell minus the symbol-day mean (descriptive conditional
           effect; the normalisation is on the target side only, never in a state)
T5, one row per 5-minute block (decision times at the end of minutes m with (m + 1) % 5 == 0):
  flow_q   quintile of the 5-minute notional imbalance (minutes m-4..m), ranked among past minutes' rolling values
  vol_t    tercile of log RV over minutes m-29..m (past-ranked)
  hour, cls as in T1; targets a5, s5 (normalised over the symbol-day's T5 rows)

Inclusion: a symbol-day enters if at least MIN_COVER of its 1440 minutes have a print. Rows enter once every state of
the table is defined (past warm-up) and the target exists. A day-cell estimate needs >= MIN_ROWS rows (pooled over
symbols); an effect is testable on discovery if >= MIN_DAYS_DISC of the 16 days give an estimate.

Caveat fixed in advance: returns are last-trade returns, so bid-ask bounce makes "flow -> next-minute reversal"
partly mechanical; vol targets are much less affected.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DAYS_ROOT = ROOT / "data" / "research" / "micro_days"
MINING_DIR = ROOT / "data" / "research" / "mining"
FROZEN = MINING_DIR / "discoveries_frozen.json"
CACHE = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\c--Users-Asus-Desktop-AI-Trader\31faef80-e320-4ca6-ab63-04fa013eeb03\scratchpad\agentE\cache")

DISCOVERY_DAYS = ("2025-09-05", "2025-12-18", "2026-07-12", "2026-07-13", "2026-09-23", "2025-08-31", "2025-11-19", "2026-07-22",
                  "2026-08-17", "2026-10-25", "2026-05-11", "2026-06-21", "2026-07-03", "2026-08-16", "2026-10-02", "2026-11-10")
CONFIRMATION_DAYS = ("2026-05-20", "2026-06-14", "2026-07-07", "2026-07-18", "2026-10-09", "2025-11-21", "2026-01-25", "2026-07-17",
                     "2026-08-04", "2026-08-18", "2025-09-02", "2025-09-10", "2026-10-16", "2026-11-02", "2026-11-08")
CUTOFF = "2026-07-01"   # segment split (llm_in_sample < cutoff <= clean_holdout), as micro_days.segment_of

HOUR_BLOCKS = ((0, 6), (6, 12), (12, 16), (16, 24))
LEADERS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CLASSES = ("major", "alt", "tradfi")
WARMUP_MIN = 60          # past values needed before a within-day rank is defined
VOL_WIN_T1 = 15          # minutes in the T1 recent-vol window
VOL_WIN_T5 = 30          # minutes in the T5 recent-vol window
FLOW_WIN_T5 = 5
SPREAD_FREE_TICKS = 1.5
MIN_BOOK_STATES = 30     # 1 s book states in a minute for a spread regime
MIN_COVER = 0.75
MIN_ROWS = 30
MIN_DAYS_DISC = 10
MIN_DAYS_CONF = 5
FDR_Q = 0.10
ALPHA_CONF = 0.05
VOL_TARGETS = ("a1", "a5")
DIR_TARGETS = ("s1", "s5")
DAY_MS = 86_400_000


def segment_of(date: str) -> str:
    return "llm_in_sample" if date < CUTOFF else "clean_holdout"


def asset_class(sym: str) -> str:
    from ai_trader.research.microstructure_laws_study import asset_class as ac

    return ac(sym)


def day0_ms(date: str) -> int:
    from datetime import UTC, datetime

    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


# ----------------------------------------------------------------------------------------------------------------------
# Effect family (declared before any statistic is computed)
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    id: str
    family: str
    table: str                          # "t1" | "t5"
    target: str                         # a1 a5 s1 s5
    filt: tuple[tuple[str, Any], ...]   # row filter (column, value)
    keys: tuple[str, ...]               # grouping columns
    cell: tuple[Any, ...]               # key values of the cell
    minus: tuple[Any, ...] | None = None  # contrast cell (cell - minus); None = cell vs symbol-day mean
    kind: str = field(default="")       # "vol" | "direction"

    def describe(self) -> str:
        f = ", ".join(f"{k}={v}" for k, v in self.filt)
        c = ", ".join(f"{k}={v}" for k, v in zip(self.keys, self.cell, strict=True))
        if self.minus is None:
            return f"[{self.table}] {self.target} | {f + '; ' if f else ''}{c} vs symbol-day mean"
        m = ", ".join(f"{k}={v}" for k, v in zip(self.keys, self.minus, strict=True))
        return f"[{self.table}] {self.target} | {f + '; ' if f else ''}({c}) minus ({m})"


def _kind(target: str) -> str:
    return "vol" if target in VOL_TARGETS else "direction"


def effect_family() -> list[Effect]:
    """The declared family. F1 own-state cross, F2 flow Q5-Q1 contrasts, F3 spread regime (L2 symbols), F3c free-tick
    contrasts, F4a leader state for alts, F4b leader x own flow for alts, F5 the 5-minute table cross."""
    out: list[Effect] = []
    t1_targets = ("a1", "a5", "s1", "s5")
    for tg in t1_targets:
        for c in CLASSES:
            for fq in range(5):
                for vt in range(3):
                    for h in range(4):
                        out.append(Effect(f"F1:{tg}:{c}:f{fq}:v{vt}:h{h}", "F1", "t1", tg, (("cls", c),), ("flow_q", "vol_t", "hour"),
                                          (fq, vt, h), None, _kind(tg)))
        for c in CLASSES:
            for vt in range(3):
                for h in range(4):
                    out.append(Effect(f"F2:{tg}:{c}:v{vt}:h{h}", "F2", "t1", tg, (("cls", c),), ("flow_q", "vol_t", "hour"),
                                      (4, vt, h), (0, vt, h), _kind(tg)))
        for sp in (0, 1):
            for fq in range(5):
                for vt in range(3):
                    out.append(Effect(f"F3:{tg}:sp{sp}:f{fq}:v{vt}", "F3", "t1", tg, (("has_book", 1),), ("spread", "flow_q", "vol_t"),
                                      (sp, fq, vt), None, _kind(tg)))
        for fq in range(5):
            for vt in range(3):
                out.append(Effect(f"F3c:{tg}:f{fq}:v{vt}", "F3c", "t1", tg, (("has_book", 1),), ("spread", "flow_q", "vol_t"),
                                  (1, fq, vt), (0, fq, vt), _kind(tg)))
        for ld in LEADERS:
            p = ld[:3]
            for lq in range(5):
                for lv in range(3):
                    out.append(Effect(f"F4a:{tg}:{p}:lf{lq}:lv{lv}", "F4a", "t1", tg, (("cls", "alt"),), (f"{p}_flow_q", f"{p}_vol_t"),
                                      (lq, lv), None, _kind(tg)))
            for lq in range(5):
                for fq in range(5):
                    out.append(Effect(f"F4b:{tg}:{p}:lf{lq}:f{fq}", "F4b", "t1", tg, (("cls", "alt"),), (f"{p}_flow_q", "flow_q"),
                                      (lq, fq), None, _kind(tg)))
    for tg in ("a5", "s5"):
        for c in CLASSES:
            for fq in range(5):
                for vt in range(3):
                    for h in range(4):
                        out.append(Effect(f"F5:{tg}:{c}:f{fq}:v{vt}:h{h}", "F5", "t5", tg, (("cls", c),), ("flow_q", "vol_t", "hour"),
                                          (fq, vt, h), None, _kind(tg)))
    ids = [e.id for e in out]
    assert len(ids) == len(set(ids))
    return out


def family_counts(fam: list[Effect] | None = None) -> dict[str, Any]:
    fam = fam or effect_family()
    by_f: dict[str, int] = {}
    for e in fam:
        by_f[e.family] = by_f.get(e.family, 0) + 1
    return {"total": len(fam), "by_family": by_f, "vol": sum(e.kind == "vol" for e in fam),
            "direction": sum(e.kind == "direction" for e in fam)}


# ----------------------------------------------------------------------------------------------------------------------
# Point-in-time state construction
# ----------------------------------------------------------------------------------------------------------------------


def past_percentile(x: np.ndarray, min_past: int = WARMUP_MIN) -> np.ndarray:
    """Mid-rank of x[i] among the finite values x[0..i-1] (strictly past). NaN if fewer than min_past past values or
    x[i] is not finite. Uses only data before i."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    valid = np.isfinite(x)
    xv = np.where(valid, x, 0.0)
    past = np.tril(np.ones((n, n), dtype=bool), -1) & valid[None, :]
    cnt = past.sum(1)
    less = ((xv[None, :] < xv[:, None]) & past).sum(1)
    eq = ((xv[None, :] == xv[:, None]) & past).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = (less + 0.5 * eq) / cnt
    ok = valid & (cnt >= min_past)
    out[ok] = pct[ok]
    return out


def to_bins(pct: np.ndarray, k: int) -> np.ndarray:
    """Percentile in [0, 1] -> bin 0..k-1; NaN -> -1."""
    b = np.full(len(pct), -1, dtype=np.int64)
    ok = np.isfinite(pct)
    b[ok] = np.clip(np.floor(pct[ok] * k).astype(np.int64), 0, k - 1)
    return b


def hour_block(minute: np.ndarray) -> np.ndarray:
    h = np.asarray(minute) // 60
    out = np.full(len(h), -1, dtype=np.int64)
    for i, (a, b) in enumerate(HOUR_BLOCKS):
        out[(h >= a) & (h < b)] = i
    return out


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over [i-w+1, i]; NaN until w values exist or if any NaN in the window."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    c = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))])
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(x))])
    s = c[w:] - c[:-w]
    nb = bad[w:] - bad[:-w]
    out[w - 1:] = np.where(nb == 0, s, np.nan)
    return out


def minute_series(tr: pl.DataFrame, d0: int) -> dict[str, np.ndarray]:
    """Per-minute raw quantities from the sparse 1 s bars: close (last trade, forward-filled; NaN before the first
    print), 5 s RV, buy / sell notional, n_trades."""
    sec = ((tr["ts"].to_numpy() - d0) // 1000).astype(np.int64)
    keep = (sec >= 0) & (sec < 86400)
    sec = sec[keep]
    close = tr["close"].to_numpy()[keep]
    px = np.full(86400, np.nan)
    px[sec] = close
    # forward fill
    idx = np.where(np.isfinite(px), np.arange(86400), 0)
    np.maximum.accumulate(idx, out=idx)
    px = px[idx]
    px[: sec.min() if len(sec) else 86400] = np.nan
    lp = np.log(px)
    p5 = lp[4::5]                       # last price at the end of each 5 s bin
    r5 = np.diff(np.concatenate([[np.nan], p5]))
    rv = np.nansum(np.where(np.isfinite(r5), r5 ** 2, np.nan).reshape(1440, 12), axis=1)
    rv_ok = np.isfinite(r5).reshape(1440, 12).sum(1) >= 6
    rv = np.where(rv_ok, rv, np.nan)
    mclose = lp[59::60]
    minute = sec // 60
    bn = np.bincount(minute, weights=tr["buy_notional"].to_numpy()[keep], minlength=1440)
    sn = np.bincount(minute, weights=tr["sell_notional"].to_numpy()[keep], minlength=1440)
    nt = np.bincount(minute, weights=tr["n_trades"].to_numpy()[keep].astype(float), minlength=1440)
    return {"lclose": mclose, "rv": rv, "buy": bn, "sell": sn, "n_trades": nt}


def spread_minutes(book: pl.DataFrame, d0: int, tick: float) -> np.ndarray:
    """Spread regime per minute: 1 free (TWA spread > 1.5 ticks), 0 tick-bound, -1 undefined. A book row with ts (state
    at ts) in (m x 60 s, (m + 1) x 60 s] belongs to minute m (known at the end of minute m)."""
    col = book["ts"]
    if isinstance(col.dtype, pl.Datetime):
        col = col.dt.cast_time_unit("ms")
    ts = col.cast(pl.Int64).to_numpy()     # epoch ms
    m =(ts - d0 - 1) // 60000
    sp = (book["ask1"].to_numpy() - book["bid1"].to_numpy()) / tick
    ok = (m >= 0) & (m < 1440) & np.isfinite(sp)
    s = np.bincount(m[ok], weights=sp[ok], minlength=1440)
    c = np.bincount(m[ok], minlength=1440)
    out = np.full(1440, -1, dtype=np.int64)
    good = c >= MIN_BOOK_STATES
    out[good] = (s[good] / c[good] > SPREAD_FREE_TICKS).astype(np.int64)
    return out


def _norm_targets(df: dict[str, np.ndarray], rows: np.ndarray, names: Iterable[str]) -> None:
    """Normalise over the analysis rows (|r| / mean |r|, r / sd r), then demean within the symbol-day."""
    for k in names:
        y = df[k]
        v = y[rows]
        if k.startswith("a"):
            mu = np.mean(v) if len(v) else np.nan
            z = y / mu if mu and mu > 0 else np.full_like(y, np.nan)
        else:
            sd = np.std(v) if len(v) else np.nan
            z = y / sd if sd and sd > 0 else np.full_like(y, np.nan)
        z = z - np.nanmean(z[rows]) if rows.any() else z
        df[k] = np.where(rows, z, np.nan)


def symbol_tables(ms: dict[str, np.ndarray], spread: np.ndarray | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """State tables T1 (per minute) and T5 (per 5-minute block) from the per-minute series. States at minute m use
    only minutes <= m; targets use minutes > m."""
    lc, rv, buy, sell = ms["lclose"], ms["rv"], ms["buy"], ms["sell"]
    minute = np.arange(1440)
    tot = buy + sell
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(tot > 0, (buy - sell) / tot, 0.0)
    imb = np.where(np.isfinite(lc), imb, np.nan)
    flow_q = to_bins(past_percentile(imb), 5)
    v15 = np.log(_rolling_sum(rv, VOL_WIN_T1) + 1e-18)
    vol_t = to_bins(past_percentile(v15), 3)
    hour = hour_block(minute)
    nxt1 = np.concatenate([lc[1:], [np.nan]])
    nxt5 = np.concatenate([lc[5:], [np.nan] * 5])
    s1, s5 = nxt1 - lc, nxt5 - lc
    t1 = {"minute": minute, "flow_q": flow_q, "vol_t": vol_t, "hour": hour,
          "spread": spread if spread is not None else np.full(1440, -1, dtype=np.int64),
          "a1": np.abs(s1), "a5": np.abs(s5), "s1": s1, "s5": s5}
    rows1 = (flow_q >= 0) & (vol_t >= 0) & np.isfinite(s1) & np.isfinite(s5)
    _norm_targets(t1, rows1, ("a1", "a5", "s1", "s5"))
    t1["ok"] = rows1
    # T5
    b5, s5n = _rolling_sum(buy, FLOW_WIN_T5), _rolling_sum(sell, FLOW_WIN_T5)
    with np.errstate(invalid="ignore", divide="ignore"):
        imb5 = np.where(b5 + s5n > 0, (b5 - s5n) / (b5 + s5n), 0.0)
    imb5 = np.where(np.isfinite(lc) & np.isfinite(b5), imb5, np.nan)
    fq5 = to_bins(past_percentile(imb5), 5)
    v30 = np.log(_rolling_sum(rv, VOL_WIN_T5) + 1e-18)
    vt5 = to_bins(past_percentile(v30), 3)
    sel = (minute + 1) % 5 == 0
    t5 = {"minute": minute[sel], "flow_q": fq5[sel], "vol_t": vt5[sel], "hour": hour[sel], "a5": np.abs(s5)[sel], "s5": s5[sel]}
    rows5 = (t5["flow_q"] >= 0) & (t5["vol_t"] >= 0) & np.isfinite(t5["s5"])
    _norm_targets(t5, rows5, ("a5", "s5"))
    t5["ok"] = rows5
    return t1, t5


# ----------------------------------------------------------------------------------------------------------------------
# Day loading (with the discovery / confirmation wall)
# ----------------------------------------------------------------------------------------------------------------------


class FrozenMissing(RuntimeError):
    pass


def require_frozen(frozen: Path = FROZEN) -> dict[str, Any]:
    if not frozen.is_file():
        raise FrozenMissing(f"{frozen} does not exist: the discovery list must be frozen before confirmation days are read")
    obj = json.loads(frozen.read_text(encoding="utf-8"))
    if not obj.get("frozen_at") or "discoveries" not in obj:
        raise FrozenMissing(f"{frozen} is not a frozen discovery file")
    return obj


def _check_wall(date: str, frozen: Path = FROZEN) -> None:
    if date in CONFIRMATION_DAYS:
        require_frozen(frozen)
    elif date not in DISCOVERY_DAYS:
        raise ValueError(f"{date} is neither a discovery nor a confirmation day of H-MINE1")


def day_ready(date: str) -> bool:
    q = DAYS_ROOT / date / "qa.json"
    if not q.is_file():
        return False
    try:
        return bool(json.loads(q.read_text(encoding="utf-8")).get("verified", {}).get("ok"))
    except (json.JSONDecodeError, OSError):
        return False


def build_day(date: str) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """T1 and T5 rows of every included symbol of a day (leader states joined), plus inclusion info."""
    _check_wall(date)
    d0 = day0_ms(date)
    root = DAYS_ROOT / date
    per: dict[str, tuple[dict, dict]] = {}
    info: dict[str, Any] = {"included": [], "excluded": {}}
    for f in sorted((root / "trades_1s").glob("*.parquet")):
        sym = f.stem
        tr = pl.read_parquet(f, columns=["ts", "close", "buy_notional", "sell_notional", "n_trades"])
        ms = minute_series(tr, d0)
        cover = float((ms["n_trades"] > 0).mean())
        if cover < MIN_COVER:
            info["excluded"][sym] = round(cover, 3)
            continue
        spread = None
        bf = root / "book_1s" / f"{sym}.parquet"
        lf = root / "laws" / f"{sym}.json"
        if bf.is_file() and lf.is_file():
            tick = json.loads(lf.read_text(encoding="utf-8")).get("tick")
            if tick and tick > 0:
                spread = spread_minutes(pl.read_parquet(bf, columns=["ts", "bid1", "ask1"]), d0, float(tick))
        per[sym] = symbol_tables(ms, spread)
        info["included"].append(sym)
    t1s, t5s = [], []
    leader_cols = {}
    for ld in LEADERS:
        p = ld[:3]
        if ld in per:
            leader_cols[f"{p}_flow_q"] = per[ld][0]["flow_q"]
            leader_cols[f"{p}_vol_t"] = per[ld][0]["vol_t"]
        else:
            leader_cols[f"{p}_flow_q"] = np.full(1440, -1, dtype=np.int64)
            leader_cols[f"{p}_vol_t"] = np.full(1440, -1, dtype=np.int64)
    for sym, (t1, t5) in per.items():
        c = asset_class(sym)
        ok1 = t1["ok"]
        d1 = {k: v[ok1] for k, v in t1.items() if k != "ok"}
        d1.update({k: v[ok1] for k, v in leader_cols.items()})
        d1["has_book"] = (d1["spread"] >= 0).astype(np.int64)
        t1s.append(pl.DataFrame(d1).with_columns(pl.lit(date).alias("day"), pl.lit(sym).alias("sym"), pl.lit(c).alias("cls")))
        ok5 = t5["ok"]
        d5 = {k: v[ok5] for k, v in t5.items() if k != "ok"}
        t5s.append(pl.DataFrame(d5).with_columns(pl.lit(date).alias("day"), pl.lit(sym).alias("sym"), pl.lit(c).alias("cls")))
    return pl.concat(t1s, how="vertical_relaxed"), pl.concat(t5s, how="vertical_relaxed"), info


def _build_cached(date: str) -> dict[str, Any]:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(k, "1")
    CACHE.mkdir(parents=True, exist_ok=True)
    p1, p5, pi = CACHE / f"{date}_t1.parquet", CACHE / f"{date}_t5.parquet", CACHE / f"{date}_info.json"
    if p1.is_file() and p5.is_file() and pi.is_file():
        return json.loads(pi.read_text(encoding="utf-8"))
    t1, t5, info = build_day(date)
    t1.write_parquet(str(p1) + ".tmp")
    t5.write_parquet(str(p5) + ".tmp")
    os.replace(str(p1) + ".tmp", p1)
    os.replace(str(p5) + ".tmp", p5)
    pi.write_text(json.dumps(info), encoding="utf-8")
    return info


def load_days(days: Iterable[str], workers: int = 4) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    days = list(days)
    for d in days:
        _check_wall(d)
    with ProcessPoolExecutor(max_workers=min(workers, 4, len(days))) as ex:
        infos = dict(zip(days, ex.map(_build_cached, days), strict=True))
    t1 = pl.concat([pl.read_parquet(CACHE / f"{d}_t1.parquet") for d in days], how="vertical_relaxed")
    t5 = pl.concat([pl.read_parquet(CACHE / f"{d}_t5.parquet") for d in days], how="vertical_relaxed")
    return t1, t5, infos


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------


def bh(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean array of rejections at FDR q."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    thr = q * np.arange(1, m + 1) / m
    below = p[order] <= thr
    rej = np.zeros(m, dtype=bool)
    if below.any():
        k = np.max(np.nonzero(below)[0])
        rej[order[: k + 1]] = True
    return rej


def bh_adjusted(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def day_ttest(est: np.ndarray, sided: int = 0) -> dict[str, float]:
    """t-test of the mean of per-day estimates (day-clustered). sided 0 = two-sided, +1 / -1 = one-sided in that
    direction."""
    from scipy import stats

    est = np.asarray(est, dtype=float)
    est = est[np.isfinite(est)]
    n = len(est)
    if n < 2:
        return {"n_days": n, "mean": float(est.mean()) if n else math.nan, "se": math.nan, "t": math.nan, "p": 1.0}
    mu, sd = float(est.mean()), float(est.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mu / se if se > 0 else (math.inf if mu > 0 else -math.inf if mu < 0 else 0.0)
    if sided == 0:
        p = float(2 * stats.t.sf(abs(t), n - 1))
    else:
        p = float(stats.t.sf(sided * t, n - 1))
    return {"n_days": n, "mean": mu, "se": se, "t": t, "p": min(p, 1.0)}


def _cell_table(df: pl.DataFrame, e: Effect, cache: dict) -> pl.DataFrame:
    key = (e.table, e.filt, e.keys, e.target)
    if key not in cache:
        d = df
        for c, v in e.filt:
            d = d.filter(pl.col(c) == v)
        d = d.filter(pl.col(e.target).is_finite())
        for k in e.keys:
            d = d.filter(pl.col(k) >= 0) if d.schema[k] != pl.Utf8 else d
        cache[key] = d.group_by(["day", *e.keys]).agg(pl.col(e.target).sum().alias("s"), pl.len().alias("n"))
    return cache[key]


def day_estimates(tables: dict[str, pl.DataFrame], e: Effect, days: list[str], cache: dict) -> np.ndarray:
    """Per-day estimate of an effect (NaN where a cell has < MIN_ROWS rows)."""
    g = _cell_table(tables[e.table], e, cache)

    def cell_mean(cell) -> dict[str, float]:
        f = g
        for k, v in zip(e.keys, cell, strict=True):
            f = f.filter(pl.col(k) == v)
        return {r["day"]: (r["s"] / r["n"] if r["n"] >= MIN_ROWS else math.nan) for r in f.iter_rows(named=True)}

    a = cell_mean(e.cell)
    b = cell_mean(e.minus) if e.minus is not None else None
    out = np.full(len(days), np.nan)
    for i, d in enumerate(days):
        if d in a:
            out[i] = a[d] - (b.get(d, math.nan) if b is not None else 0.0)
    return out


def _fast_estimates(tables: dict[str, pl.DataFrame], fam: list[Effect], days: list[str]) -> dict[str, np.ndarray]:
    """Vectorised day_estimates for the whole family (same numbers as day_estimates)."""
    cache: dict = {}
    lookups: dict = {}
    out = {}
    di = {d: i for i, d in enumerate(days)}
    for e in fam:
        key = (e.table, e.filt, e.keys, e.target)
        if key not in lookups:
            g = _cell_table(tables[e.table], e, cache)
            lk: dict = {}
            for r in g.iter_rows():
                day, *kv, s, n = r
                lk.setdefault(tuple(kv), np.full(len(days), np.nan))
                if day in di and n >= MIN_ROWS:
                    lk[tuple(kv)][di[day]] = s / n
            lookups[key] = lk
        lk = lookups[key]
        a = lk.get(tuple(e.cell), np.full(len(days), np.nan))
        if e.minus is not None:
            a = a - lk.get(tuple(e.minus), np.full(len(days), np.nan))
        out[e.id] = a
    return out


def discovery(t1: pl.DataFrame, t5: pl.DataFrame, days: list[str] | None = None) -> dict[str, Any]:
    days = list(days or DISCOVERY_DAYS)
    for d in days:
        if d not in DISCOVERY_DAYS:
            raise ValueError(f"{d} is not a discovery day")
    fam = effect_family()
    est = _fast_estimates({"t1": t1, "t5": t5}, fam, days)
    rows = []
    for e in fam:
        x = est[e.id]
        nd = int(np.isfinite(x).sum())
        r = day_ttest(x) if nd >= MIN_DAYS_DISC else {"n_days": nd, "mean": float(np.nanmean(x)) if nd else math.nan,
                                                      "se": math.nan, "t": math.nan, "p": 1.0}
        rows.append({"id": e.id, "family": e.family, "kind": e.kind, "target": e.target, "table": e.table, "describe": e.describe(),
                     "testable": nd >= MIN_DAYS_DISC, **r, "day_est": [None if not np.isfinite(v) else float(v) for v in x]})
    p = np.array([r["p"] for r in rows])
    rej = bh(p, FDR_Q)
    adj = bh_adjusted(p)
    for r, j, a in zip(rows, rej, adj, strict=True):
        r["bh_reject"] = bool(j)
        r["p_bh"] = float(a)
        r["direction"] = int(np.sign(r["mean"])) if j else 0
    return {"days": days, "family": family_counts(fam), "effects": rows}


def confirmation(t1: pl.DataFrame, t5: pl.DataFrame, days: list[str], frozen: Path = FROZEN) -> dict[str, Any]:
    fz = require_frozen(frozen)
    for d in days:
        if d not in CONFIRMATION_DAYS:
            raise ValueError(f"{d} is not a confirmation day")
    fam = {e.id: e for e in effect_family()}
    disc = [fam[x["id"]] for x in fz["discoveries"]]
    dirs = {x["id"]: int(x["direction"]) for x in fz["discoveries"]}
    est = _fast_estimates({"t1": t1, "t5": t5}, disc, days)
    segs = {"pooled": days, "llm_in_sample": [d for d in days if segment_of(d) == "llm_in_sample"],
            "clean_holdout": [d for d in days if segment_of(d) == "clean_holdout"]}
    rows = []
    for e in disc:
        x = est[e.id]
        row = {"id": e.id, "family": e.family, "kind": e.kind, "direction": dirs[e.id], "describe": e.describe()}
        for s, sd in segs.items():
            idx = [days.index(d) for d in sd]
            xs = x[idx] if idx else np.array([])
            nd = int(np.isfinite(xs).sum())
            need = MIN_DAYS_CONF if s == "pooled" else 3
            r = day_ttest(xs, sided=dirs[e.id]) if nd >= need else {"n_days": nd, "mean": math.nan, "se": math.nan, "t": math.nan, "p": 1.0}
            row[s] = {**r, "testable": nd >= need, "replicated": bool(nd >= need and r["p"] < ALPHA_CONF)}
        row["day_est"] = [None if not np.isfinite(v) else float(v) for v in x]
        rows.append(row)
    return {"days": days, "segments": segs, "effects": rows, "summary": replication_summary(rows)}


def replication_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in ("pooled", "llm_in_sample", "clean_holdout"):
        d = {}
        for k in ("all", "vol", "direction"):
            sel = [r for r in rows if k == "all" or r["kind"] == k]
            n = len(sel)
            rep = sum(r[s]["replicated"] for r in sel)
            tst = sum(r[s]["testable"] for r in sel)
            d[k] = {"discoveries": n, "testable": tst, "replicated": rep, "rate": rep / n if n else math.nan}
        out[s] = d
    p = out["pooled"]
    out["criterion"] = {"rate_ge_50": bool(p["all"]["rate"] >= 0.5),
                        "vol_gt_direction": bool((p["vol"]["rate"] if p["vol"]["discoveries"] else 0.0)
                                                 > (p["direction"]["rate"] if p["direction"]["discoveries"] else 0.0)),
                        }
    out["criterion"]["met"] = out["criterion"]["rate_ge_50"] and out["criterion"]["vol_gt_direction"]
    return out


def freeze(disc: dict[str, Any], frozen: Path = FROZEN) -> dict[str, Any]:
    """Write the frozen discovery list (refuses to overwrite)."""
    from datetime import UTC, datetime

    if frozen.exists():
        raise FileExistsError(f"{frozen} already exists; the discovery list is frozen once")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    ds = [{"id": r["id"], "family": r["family"], "kind": r["kind"], "direction": r["direction"], "size": r["mean"], "t": r["t"],
           "p": r["p"], "p_bh": r["p_bh"], "n_days": r["n_days"], "describe": r["describe"]}
          for r in disc["effects"] if r["bh_reject"]]
    obj = {"hypothesis": "H-MINE1", "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "discovery_days": disc["days"],
           "confirmation_days": list(CONFIRMATION_DAYS), "family": disc["family"], "fdr_q": FDR_Q, "alpha_confirmation": ALPHA_CONF,
           "n_discoveries": len(ds), "discoveries": ds}
    tmp = frozen.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, frozen)
    return obj
