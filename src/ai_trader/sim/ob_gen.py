"""Queue-reactive BTC order-flow generator (agent F5, H-OBG1 / H-OBG2; plan research/plans/2026-11-13_btc_field_program.md).

Generated data is for deterministic models, strategies and the execution engine only, never for the LLM layer
(CLAUDE.md principle 1). Research only.

Pipeline
  1. download    one BTC day at a time (public.bybit.com trades + quote-saver ob200) into data/raw/obgen/<date>/, after a
                 storage check (data/lake + data/research + data/raw < budget); processed, verified, then deleted. The
                 manifest data/research/obgen/manifest.json records every download / calibration / generated day, so nothing
                 is downloaded or recomputed twice.
  2. calibrate   `extract_calibration` streams the ob200 deltas (100 ms pushes) with the day's prints and accumulates, per
                 side-agnostic level d (distance in ticks from the own-side best quote), per queue-size bin q (in units of
                 the day's average touch event size, AES), per spread state (1 tick / wider) and per recent-volatility state
                 (tercile of the trailing-60 s mid RV): the time at risk, the number of 100 ms intervals with a net limit
                 inflow and with a net cancellation (net of the prints at that price), and reservoir samples of the event
                 sizes. Market consumption is attributed from the prints. This is the Huang-Lehalle-Rosenbaum (2015)
                 queue-reactive estimator lambda(q) = events in state q / time in state q, at the 100 ms resolution of the
                 feed (net events per interval; gross flows inside one push are not observable). Also: the stationary
                 profile (occupancy and size per d), the in-spread improvement rate and size, the touch-depletion statistics,
                 and the real aggregates of the day (data/research/obgen/real/<date>/, micro_days schema).
  3. day cal     `day_calibration`: per-day scale factors from the FIT half (00-12 UTC) of a micro_days day: touch and deep
                 queue scales, order arrival law (expmix3), order size law (log-normal body + Pareto tail), order-sign
                 model (TranslatedSignModel), all taken from the day's laws/BTCUSDT.json (H-MD1: traits vary by day).
  4. generate    `generate`: 100 ms steps. Per level, net limit / cancel events are drawn from the calibrated
                 queue-reactive tables given the current queue size, spread state and volatility state; taker (market)
                 orders arrive by the day's bursty arrival law with the day's sizes and long-memory signs, with their count,
                 volume and buy / sell split per bar taken from an external PATH (real 1 s or 1 min bars, or any model's
                 paths); the reference price follows the path's anchors through repricing (touch cancelled, the other side
                 improves), whose intensity grows with the gap between the path and the book mid, so market-order impact is
                 transient and the bar-level path is honoured. Every action is emitted as an at-match JSONL event
                 (docs/MATCHING_ENGINE.md): {"ts_ms", "kind": new|cancel|amend, "id", "side", "type", "tif", "price",
                 "qty", "reduce_only", "owner"}.
  5. engine      `run_engine`: `atx match-sim` when the binary has the subcommand (crates/at-match, agent F4), else the
                 generator's own book (identical by construction; checked against `RefBook`, a minimal price-time FIFO
                 reference book, in tests/test_ob_gen.py).
  6. aggregate   `aggregate_engine_output` / `aggregate_internal`: trades -> orders (same-ms same-side runs) -> trades_1s,
                 book states -> book_1s (sim.book.book_metrics on the 200-level view), the micro_days schema exactly.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.sim.ob_gen calibrate 2026-04-15
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "obgen"
OUT = ROOT / "data" / "research" / "obgen"
MANIFEST = OUT / "manifest.json"
MICRO = ROOT / "data" / "research" / "micro_days"
SYMBOL = "BTCUSDT"
TICK = 0.1
QTY_STEP = 0.001
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
DAY_MS = 86_400_000
HALF_MS = 43_200_000
STEP_MS = 100
DEPTH_LEVELS = 200          # the public feed's view (and the engine's --depth)
D = 480                     # ticks per side held by the generator (200 non-empty levels of BTC span ~300 ticks)
D_NEAR = 10                 # levels 0..D_NEAR-1 have their own tables; deeper levels are pooled in bands
BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 80, 120, 180, 260, 360, D)   # band b = [BANDS[b], BANDS[b+1])
NB = len(BANDS) - 1
QBINS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, np.inf)  # q / AES; bin 0 = empty level
NQ = len(QBINS)             # bin 0 = exactly empty, bins 1.. = (QBINS[k-1], QBINS[k]]
NV = 3                      # volatility terciles
NS = 2                      # spread state: 1 tick | wider
NRES = 4000                 # reservoir size per cell
SIZE_PHI = 0.995            # generator: AR(1) coefficient (per order) of the latent order-size level
SIZE_RUN_COUPLING = 4.0     # generator: weight of the order's alignment with the local sign run in the size score
TRADE_TIME_FLOOR = 0.2      # generator: calendar-time share floor of the trade-time noise weights
FLOW_QV_CAP = 0.7           # generator: largest share of a bar's realized variance carried by the flow bridge
IMPACT_DELTA = 0.5          # generator: concave (square-root) impact of a taker order in the flow bridge
DEFICIT_SHARE = 0.3         # generator: share of a bar side's orders (top size scores) that absorb the notional deficit
INFORMED_SIZE_BOOST = 2.0   # generator: size-score boost of informed orders (latency-arbitrage takers trade larger)
HEAL_REFILL = 0.75          # generator: share of sweep gaps closed by a refill of the swept side (else the other side follows)
HEAL_NOW = 0.95             # generator: share of sweep gaps re-quoted in the same ms (the rest at the next 100 ms step)
DEEP_REFRESH_PER_S = 1 / 60  # generator: rate at which a deep level (d >= D_FINE) is re-drawn from the profile
GB = 1024 ** 3


# ----------------------------------------------------------------------------------------------------------------------
# manifest and storage
# ----------------------------------------------------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1), encoding="utf-8")
    tmp.replace(path)


class Manifest:
    """data/research/obgen/manifest.json: {section: {key: record}}. Every write is atomic; `done` decides skips."""

    def __init__(self, path: Path = MANIFEST):
        self.path = Path(path)
        self.data: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    def get(self, section: str, key: str) -> dict[str, Any] | None:
        return self.data.get(section, {}).get(key)

    def done(self, section: str, key: str) -> bool:
        r = self.get(section, key)
        return bool(r and r.get("status") == "done")

    def put(self, section: str, key: str, rec: dict[str, Any]) -> None:
        """Locked read-merge-write, so several processes (<= 2 workers) can share the manifest."""
        lock = self.path.with_suffix(".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(600):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > 120:
                    lock.unlink(missing_ok=True)
                time.sleep(0.1)
        else:
            fd = None
        try:
            if self.path.is_file():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.setdefault(section, {})[key] = {**rec, "updated": datetime.now(UTC).isoformat(timespec="seconds")}
            write_json(self.path, self.data)
        finally:
            if fd is not None:
                os.close(fd)
                lock.unlink(missing_ok=True)


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def storage_gb(root: Path = ROOT) -> dict[str, float]:
    d = root / "data"
    lake, res, raw = du_bytes(d / "lake"), du_bytes(d / "research"), du_bytes(d / "raw")
    return {"lake_gb": lake / GB, "research_gb": res / GB, "raw_gb": raw / GB, "total_gb": (lake + res + raw) / GB,
            "obgen_gb": du_bytes(OUT) / GB}


def budget_gb() -> float:
    import yaml

    s = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return float(s.get("storage", {}).get("budget_gb", 30))


def http_get(url: str, dest: Path, retries: int = 4, timeout: float = 180.0) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AI_Trader research; public archives)"})
            with urllib.request.urlopen(req, timeout=timeout) as r, part.open("wb") as f:
                expected = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f, 1 << 20)
            size = part.stat().st_size
            if expected is not None and int(expected) != size:
                raise OSError(f"short read {size} of {expected}")
            part.replace(dest)
            return {"status": "ok", "bytes": size, "attempts": attempt + 1}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"status": "missing", "bytes": 0}
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - retried, reported
            last = repr(e)
        time.sleep(5 * (attempt + 1))
    part.unlink(missing_ok=True)
    return {"status": "failed", "bytes": 0, "error": last}


def fetch_day(date: str, man: Manifest | None = None, fetch=http_get, max_gb: float | None = None) -> dict[str, Path] | None:
    """Download the BTC trade and ob200 archives of `date` into data/raw/obgen/<date>/ unless the manifest says the day was
    already processed (then None). Refuses (RuntimeError) when data/lake + data/research + data/raw >= budget."""
    man = man or Manifest()
    if man.done("calibration", date):
        return None
    st = storage_gb()
    cap = budget_gb() if max_gb is None else max_gb
    if st["total_gb"] >= cap - 1.0:
        raise RuntimeError(f"storage {st['total_gb']:.2f} GB >= budget {cap} - 1 GB headroom: no download")
    d = RAW / date
    paths = {"trades": d / f"{SYMBOL}{date}.csv.gz", "ob200": d / f"{date}_{SYMBOL}_ob200.data.zip"}
    urls = {"trades": TRADES_URL.format(s=SYMBOL, d=date), "ob200": OB_URL.format(s=SYMBOL, d=date)}
    for k in ("trades", "ob200"):
        key = f"{date}:{k}"
        if paths[k].is_file() and paths[k].stat().st_size > 0:
            continue
        r = fetch(urls[k], paths[k])
        man.put("downloads", key, {**r, "url": urls[k], "path": str(paths[k].relative_to(ROOT)), "storage_before_gb": st["total_gb"]})
        if r["status"] != "ok":
            raise RuntimeError(f"download {key}: {r}")
    return paths


def cleanup_day(date: str, man: Manifest) -> None:
    d = RAW / date
    nbytes = du_bytes(d)
    shutil.rmtree(d, ignore_errors=True)
    for k in ("trades", "ob200"):
        r = man.get("downloads", f"{date}:{k}") or {}
        man.put("downloads", f"{date}:{k}", {**r, "deleted": True})
    man.put("raw_deleted", date, {"status": "done", "bytes": nbytes})


# ----------------------------------------------------------------------------------------------------------------------
# cells and small helpers
# ----------------------------------------------------------------------------------------------------------------------

D_FINE = 30                 # levels updated every 100 ms (tables from 100 ms intervals); deeper levels every 1 s
BAND_OF = np.searchsorted(np.asarray(BANDS), np.arange(D), side="right") - 1          # d -> band
QEDGES = np.asarray(QBINS[1:-1], dtype=float)                                          # interior edges of bins 1..


def qbin(q_aes: np.ndarray) -> np.ndarray:
    """Queue-size bin: 0 = empty, k >= 1 for q in (QBINS[k-1], QBINS[k]] (q in AES units)."""
    q = np.asarray(q_aes, dtype=float)
    return np.where(q <= 0, 0, 1 + np.searchsorted(QEDGES, q, side="left")).astype(np.int64)


def cell_index(band, qb, v, s) -> np.ndarray:
    return ((np.asarray(band) * NQ + np.asarray(qb)) * NV + np.asarray(v)) * NS + np.asarray(s)


NCELL = NB * NQ * NV * NS


class Reservoir:
    """Fixed-size uniform reservoir per key (Vitter's algorithm R), deterministic given the seed."""

    def __init__(self, k: int = NRES, seed: int = 0):
        self.k, self.rng, self.n, self.v = k, np.random.default_rng(seed), {}, {}

    def add(self, key, x: float) -> None:
        n = self.n.get(key, 0)
        if n < self.k:
            self.v.setdefault(key, []).append(float(x))
        else:
            j = int(self.rng.integers(0, n + 1))
            if j < self.k:
                self.v[key][j] = float(x)
        self.n[key] = n + 1

    def quantiles(self, key, nq: int = 41) -> list[float] | None:
        x = self.v.get(key)
        if not x:
            return None
        return [float(v) for v in np.quantile(np.asarray(x), np.linspace(0, 1, nq))]


def vol_state_edges(rv60: np.ndarray) -> list[float]:
    x = rv60[np.isfinite(rv60) & (rv60 > 0)]
    return [float(np.quantile(x, 1 / 3)), float(np.quantile(x, 2 / 3))] if len(x) > 100 else [0.0, 0.0]


def rv60_from_last(last_px: np.ndarray) -> np.ndarray:
    """Trailing-60 s sum of squared 1 s log returns of the last trade price (per second, row s = seconds s-59..s)."""
    lp = np.log(last_px)
    r = np.nan_to_num(np.diff(lp, prepend=lp[0]), nan=0.0)
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    n = np.arange(1, len(r) + 1)
    return c[n] - c[np.maximum(n - 60, 0)]


def last_price_per_second(ms: np.ndarray, price: np.ndarray, d0: int, n_sec: int = 86400) -> np.ndarray:
    sec = ((ms - d0) // 1000).astype(np.int64)
    ok = (sec >= 0) & (sec < n_sec)
    z = np.full(n_sec, np.nan)
    z[sec[ok]] = price[ok]          # later prints overwrite earlier ones within a second (ms sorted)
    s = pl.Series(z).fill_nan(None).forward_fill().backward_fill().to_numpy()
    return s.astype(float)


def vstate_of(rv: float, edges: list[float]) -> int:
    return 0 if rv <= edges[0] else (1 if rv <= edges[1] else 2)


def day0_ms(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() * 1000)


# ----------------------------------------------------------------------------------------------------------------------
# calibration from event-level ob200 deltas + prints
# ----------------------------------------------------------------------------------------------------------------------

G = 200_000                 # dense absolute-tick grid per side (+-10,000 USD at a 0.1 tick)


class DenseBook:
    """Two dense arrays over an absolute tick grid (size in base units, 0 = no level) with the best quotes tracked."""

    def __init__(self, center_tick: int):
        self.base = int(center_tick) - G // 2
        self.bid = np.zeros(G)
        self.ask = np.zeros(G)
        self.ib = -1
        self.ia = -1

    def idx(self, px: float) -> int:
        return int(round(px / TICK)) - self.base

    def recenter(self, center_tick: int) -> None:
        nb = int(center_tick) - G // 2
        sh = nb - self.base
        for arr in (self.bid, self.ask):
            if abs(sh) >= G:
                arr[:] = 0.0
            elif sh > 0:
                arr[:-sh] = arr[sh:].copy()
                arr[-sh:] = 0.0
            elif sh < 0:
                arr[-sh:] = arr[:sh].copy()
                arr[:-sh] = 0.0
        self.base = nb
        self.ib -= sh
        self.ia -= sh

    def clear(self) -> None:
        self.bid[:] = 0.0
        self.ask[:] = 0.0

    def apply(self, levels_b, levels_a) -> None:
        for arr, lv in ((self.bid, levels_b), (self.ask, levels_a)):
            for p, s in lv:
                i = self.idx(float(p))
                if 0 <= i < G:
                    arr[i] = float(s)
        nzb = np.flatnonzero(self.bid)
        nza = np.flatnonzero(self.ask)
        self.ib = int(nzb[-1]) if len(nzb) else -1
        self.ia = int(nza[0]) if len(nza) else -1

    def deepest(self) -> tuple[int, int]:
        w = 1500
        lo = max(self.ib - w, 0)
        nb = np.flatnonzero(self.bid[lo:self.ib + 1])
        na = np.flatnonzero(self.ask[self.ia:self.ia + w])
        return (lo + int(nb[0]) if len(nb) else self.ib), (self.ia + int(na[-1]) if len(na) else self.ia)

    def rel(self, side: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(abs indices, sizes) of distances d = 0..n-1 from the own-side best (side +1 bid, -1 ask)."""
        if side > 0:
            ix = self.ib - np.arange(n)
            return ix, self.bid[np.clip(ix, 0, G - 1)] * (ix >= 0)
        ix = self.ia + np.arange(n)
        return ix, self.ask[np.clip(ix, 0, G - 1)] * (ix < G)

    def view(self, depth: int = DEPTH_LEVELS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prices and sizes of the first `depth` non-empty levels per side (the public feed's view)."""
        lo = max(self.ib - 3000, 0)
        nb = np.flatnonzero(self.bid[lo:self.ib + 1])[::-1][:depth] + lo
        na = (np.flatnonzero(self.ask[self.ia:self.ia + 3000]) + self.ia)[:depth]
        return ((nb + self.base) * TICK, self.bid[nb], (na + self.base) * TICK, self.ask[na])


def _book_fast_apply(bk: DenseBook, d: dict) -> None:
    """Apply a delta with incremental best tracking (scans only when a best level was removed)."""
    ib, ia = bk.ib, bk.ia
    rm_b = rm_a = False
    for p, s in d.get("b", ()):
        i = int(round(float(p) / TICK)) - bk.base
        if not 0 <= i < G:
            continue
        sz = float(s)
        bk.bid[i] = sz
        if sz > 0 and i > ib:
            ib = i
        elif sz == 0 and i == ib:
            rm_b = True
    for p, s in d.get("a", ()):
        i = int(round(float(p) / TICK)) - bk.base
        if not 0 <= i < G:
            continue
        sz = float(s)
        bk.ask[i] = sz
        if sz > 0 and (i < ia or ia < 0):
            ia = i
        elif sz == 0 and i == ia:
            rm_a = True
    if ib >= 0 and (rm_b or bk.bid[ib] == 0):
        lo = max(ib - 5000, 0)
        nz = np.flatnonzero(bk.bid[lo:ib + 1])
        ib = lo + int(nz[-1]) if len(nz) else -1
    if ia >= 0 and (rm_a or bk.ask[ia] == 0):
        nz = np.flatnonzero(bk.ask[ia:ia + 5000])
        ia = ia + int(nz[0]) if len(nz) else -1
    bk.ib, bk.ia = ib, ia


@dataclass
class CalAcc:
    """Accumulators of one calibration day (see the module docstring)."""
    aes: float
    vedges: list[float]
    risk: np.ndarray = field(default_factory=lambda: np.zeros(NCELL))      # 100 ms intervals at risk (near) / 1 s (deep)
    n_add: np.ndarray = field(default_factory=lambda: np.zeros(NCELL))
    n_can: np.ndarray = field(default_factory=lambda: np.zeros(NCELL))
    vol_add: np.ndarray = field(default_factory=lambda: np.zeros(NCELL))
    vol_can: np.ndarray = field(default_factory=lambda: np.zeros(NCELL))
    mk_risk: np.ndarray = field(default_factory=lambda: np.zeros((NQ, NV)))   # touch, unchanged intervals
    mk_hit: np.ndarray = field(default_factory=lambda: np.zeros((NQ, NV)))
    trans: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))     # touch: previous x current (none/add/cancel)
    occ_n: np.ndarray = field(default_factory=lambda: np.zeros(D))
    occ_k: np.ndarray = field(default_factory=lambda: np.zeros(D))
    res: Reservoir = field(default_factory=Reservoir)
    counts: dict = field(default_factory=dict)

    def inc(self, k: str, v: float = 1.0) -> None:
        self.counts[k] = self.counts.get(k, 0.0) + float(v)


def _snapshot_book(bk: DenseBook | None, d: dict) -> DenseBook:
    bb = max(float(p) for p, _ in d["b"])
    if bk is None:
        bk = DenseBook(int(round(bb / TICK)))
    elif abs(int(round(bb / TICK)) - bk.base - G // 2) > G // 4:
        bk.recenter(int(round(bb / TICK)))
    bk.clear()
    bk.apply(d.get("b", ()), d.get("a", ()))
    return bk


def _estimate_aes(messages, tr: dict[str, np.ndarray], n_msg: int = 30_000) -> float:
    """Average event size at the touch: mean |net flow| over 100 ms intervals with unchanged best quotes and a net flow."""
    t_ms, t_side, t_px, t_sz = tr["ms"], tr["side"], tr["price"], tr["size"]
    bk = None
    j = 0
    vals = []
    last_cts = None
    for k, m in enumerate(messages):
        if k >= n_msg:
            break
        d = m["data"]
        cts = int(m.get("cts", m["ts"]))
        if m["type"] == "snapshot":
            bk = _snapshot_book(bk, d)
            last_cts = cts
            continue
        if bk is None or bk.ib < 0 or bk.ia < 0:
            continue
        ib0, ia0 = bk.ib, bk.ia
        qb0, qa0 = bk.bid[ib0], bk.ask[ia0]
        _book_fast_apply(bk, d)
        mb = ma = 0.0
        while j < len(t_ms) and t_ms[j] <= cts:
            if last_cts is None or t_ms[j] > last_cts:
                i = int(round(t_px[j] / TICK)) - bk.base
                if t_side[j] < 0 and i == ib0:
                    mb += t_sz[j]
                elif t_side[j] > 0 and i == ia0:
                    ma += t_sz[j]
            j += 1
        last_cts = cts
        if bk.ib == ib0 and bk.ia == ia0:
            for f in (bk.bid[ib0] - qb0 + mb, bk.ask[ia0] - qa0 + ma):
                if abs(f) >= 0.5 * QTY_STEP:
                    vals.append(abs(f))
    return float(np.mean(vals)) if vals else 1.0


def _accumulate(acc: CalAcc, c: np.ndarray, fo: np.ndarray, bo: np.ndarray, qo: np.ndarray, q0: np.ndarray | None = None) -> None:
    aes = acc.aes
    np.add.at(acc.risk, c, 1.0)
    add = fo >= 0.5 * QTY_STEP
    can = fo <= -0.5 * QTY_STEP
    if not (add.any() or can.any()):
        return
    np.add.at(acc.n_add, c[add], 1.0)
    np.add.at(acc.n_can, c[can], 1.0)
    np.add.at(acc.vol_add, c[add], fo[add] / aes)
    np.add.at(acc.vol_can, c[can], -fo[can] / aes)
    for kk in np.flatnonzero(add | can):
        if fo[kk] > 0:
            acc.res.add(("add", int(bo[kk]), int(qo[kk])), fo[kk] / aes)
        else:
            # cancellations are stored as a FRACTION of the queue (bounded by 1), so they rescale with the queue
            qq = float(q0[kk]) if q0 is not None else 0.0
            acc.res.add(("canf", int(bo[kk]), int(qo[kk])), min(-fo[kk] / qq, 1.0) if qq > 0 else 1.0)


def extract_calibration(ob_path: Path, tr: dict[str, np.ndarray], date: str, real_out: Path | None = None,
                        max_messages: int | None = None, log=print) -> dict[str, Any]:
    """One pass over the ob200 stream (module docstring, step 2). `tr`: prints {ms, side (+1 taker buy), price, size}
    sorted by ms. Returns the calibration dict of the day (JSON-able); writes the real 1 s aggregates of the day when
    `real_out` is given (book_1s from the same book states, trades_1s and orders from the prints)."""
    from ai_trader.sim.book import book_metrics, iter_messages

    d0 = day0_ms(date)
    t_ms, t_side, t_px, t_sz = tr["ms"], tr["side"], tr["price"], tr["size"]
    last = last_price_per_second(t_ms, t_px, d0)
    rv60 = rv60_from_last(last)
    vedges = vol_state_edges(rv60)
    aes = _estimate_aes(iter_messages(ob_path), tr)
    log(f"{date}: AES {aes:.4f} BTC, vol edges {vedges}")
    acc = CalAcc(aes=aes, vedges=vedges)
    res = acc.res
    bk: DenseBook | None = None
    j = 0
    last_cts = None
    n_msg = 0
    cur_sec = None
    deep_prev = None
    book_rows = []
    n_upd: list[int] = []
    n_in_bin = 0
    band_near = BAND_OF[:D_FINE]
    band_deep = BAND_OF[D_FINE:]
    prev_touch = [(-1, 0.0, -1), (-1, 0.0, -1)]
    t0 = time.time()
    for m in iter_messages(ob_path):
        n_msg += 1
        if max_messages and n_msg > max_messages:
            break
        d = m["data"]
        ts = int(m["ts"])
        cts = int(m.get("cts", ts))
        if m["type"] == "snapshot":
            bk = _snapshot_book(bk, d)
            acc.inc("snapshots")
            last_cts = cts
            deep_prev = None
            continue
        if bk is None or bk.ib < 0 or bk.ia < 0:
            continue
        if bk.ib < G // 8 or bk.ib > G - G // 8:
            bk.recenter(bk.ib + bk.base)
            deep_prev = None
        sec_bin = (ts - 1) // 1000                     # 1 s bin of the push time (bin end = (sec_bin + 1) * 1000)
        if cur_sec is not None and sec_bin != cur_sec:
            # state at the end of bin cur_sec (before this message): book_1s row + 1 s deep / profile statistics
            bpx, bsz, apx, asz = bk.view()
            book_rows.append(((cur_sec + 1) * 1000, last_cts, *book_metrics(bpx, bsz, apx, asz)))
            n_upd.append(n_in_bin)
            n_in_bin = 0
            s_idx = int(((cur_sec + 1) * 1000 - d0) // 1000) - 1
            v = vstate_of(rv60[min(max(s_idx, 0), 86399)], vedges)
            dbm, dam = bk.deepest()
            ixb, qb = bk.rel(1, D)
            ixa, qa = bk.rel(-1, D)
            obs_b = ixb >= dbm
            obs_a = ixa <= dam
            acc.occ_n += obs_b.astype(float) + obs_a
            acc.occ_k += (obs_b & (qb > 0)).astype(float) + (obs_a & (qa > 0))
            if deep_prev is not None:
                _, _, pixb, pixa, pqb, pqa, pdb, pda, pv = deep_prev
                for sgn, pix, pq, arr, pdeep, ndeep in ((1, pixb, pqb, bk.bid, pdb, dbm), (-1, pixa, pqa, bk.ask, pda, dam)):
                    ix = pix[D_FINE:]
                    q0 = pq[D_FINE:]
                    obs = ((ix >= max(pdeep, ndeep)) & (ix >= 0)) if sgn > 0 else ((ix <= min(pdeep, ndeep)) & (ix < G))
                    f = arr[np.clip(ix, 0, G - 1)] - q0
                    qbn = qbin(q0 / aes)
                    c = cell_index(band_deep, qbn, pv, 0)[obs]
                    _accumulate(acc, c, f[obs], band_deep[obs], qbn[obs], q0[obs])
                    acc.inc("deep_events", float((np.abs(f[obs]) >= 0.5 * QTY_STEP).sum()))
                acc.inc("deep_seconds")
            for dd in (*range(0, D_FINE), *range(D_FINE, D, 3)):
                if obs_b[dd] and qb[dd] > 0:
                    res.add(("prof", int(BAND_OF[dd])), qb[dd] / aes)
                if obs_a[dd] and qa[dd] > 0:
                    res.add(("prof", int(BAND_OF[dd])), qa[dd] / aes)
            deep_prev = (bk.ib, bk.ia, ixb, ixa, qb.copy(), qa.copy(), dbm, dam, v)
        cur_sec = sec_bin
        n_in_bin += 1
        # ---- the 100 ms interval: before state
        ib0, ia0 = bk.ib, bk.ia
        spread0 = ia0 - ib0
        s_state = 0 if spread0 <= 1 else 1
        s_idx = int((cts - d0) // 1000) - 1
        v = vstate_of(rv60[min(max(s_idx, 0), 86399)], vedges)
        ixb, qb0 = bk.rel(1, D_FINE)
        ixa, qa0 = bk.rel(-1, D_FINE)
        dbm, dam = bk.deepest()
        _book_fast_apply(bk, d)
        if bk.ib < 0 or bk.ia < 0:
            last_cts = cts
            continue
        mb = np.zeros(D_FINE)
        ma = np.zeros(D_FINE)
        traded_b = traded_a = 0.0
        while j < len(t_ms) and t_ms[j] <= cts:
            if last_cts is None or t_ms[j] > last_cts:
                i = int(round(t_px[j] / TICK)) - bk.base
                if t_side[j] < 0:
                    dd = ib0 - i
                    traded_b += t_sz[j]
                    if 0 <= dd < D_FINE:
                        mb[dd] += t_sz[j]
                else:
                    dd = i - ia0
                    traded_a += t_sz[j]
                    if 0 <= dd < D_FINE:
                        ma[dd] += t_sz[j]
            j += 1
        last_cts = cts
        if spread0 >= 2:
            acc.inc("wide_intervals")
        if bk.ib == ib0 and bk.ia == ia0:
            acc.inc("intervals_unchanged")
            for si, (q0, ix, arr, mm, ok) in enumerate(((qb0, ixb, bk.bid, mb, ixb >= dbm), (qa0, ixa, bk.ask, ma, ixa <= dam))):
                f = arr[np.clip(ix, 0, G - 1)] - q0 + mm
                qbn = qbin(q0 / aes)
                c = cell_index(band_near, qbn, v, s_state)[ok]
                _accumulate(acc, c, f[ok], band_near[ok], qbn[ok], q0[ok])
                # touch event memory: (previous, current) net event type in consecutive unchanged intervals
                cur = 1 if f[0] >= 0.5 * QTY_STEP else (2 if f[0] <= -0.5 * QTY_STEP else 0)
                pt, psz, ppx = prev_touch[si]
                if ppx == int(ix[0]) and pt >= 0:
                    acc.trans[pt, cur] += 1
                    if pt == 1 and cur == 2:
                        res.add(("flick",), -f[0] / max(psz, 1e-12))
                prev_touch[si] = (cur, abs(float(f[0])), int(ix[0]))
                qt = int(qbn[0])
                acc.mk_risk[qt, v] += 1
                if mm[0] > 0:
                    acc.mk_hit[qt, v] += 1
                    res.add(("mk_frac",), min(mm[0] / max(q0[0], 1e-12), 1.0))
        else:
            prev_touch = [(-1, 0.0, -1), (-1, 0.0, -1)]
            dmid2 = (bk.ib + bk.ia) - (ib0 + ia0)        # mid change in half ticks
            acc.inc("intervals_changed")
            acc.inc("half_ticks_moved", abs(dmid2))
            if dmid2 != 0:
                up = dmid2 > 0
                if up:
                    imp_new = bk.bid[bk.ib] if bk.ib > ib0 else None
                    old_keep = bk.bid[ib0] / qb0[0] if (bk.ib > ib0 and qb0[0] > 0) else None
                    rec_new = bk.ask[bk.ia] if bk.ia > ia0 else None
                    taker = traded_a > 0
                else:
                    imp_new = bk.ask[bk.ia] if bk.ia < ia0 else None
                    old_keep = bk.ask[ia0] / qa0[0] if (bk.ia < ia0 and qa0[0] > 0) else None
                    rec_new = bk.bid[bk.ib] if bk.ib < ib0 else None
                    taker = traded_b > 0
                if imp_new is not None:
                    res.add(("imp_new",), imp_new / aes)
                if old_keep is not None:
                    res.add(("old_keep",), min(old_keep, 1.0))
                if rec_new is not None:
                    res.add(("rec_new",), rec_new / aes)
                acc.inc("moves")
                acc.inc("moves_with_taker", float(taker))
                res.add(("move_ticks",), abs(dmid2) / 2.0)
            if spread0 >= 2:
                acc.inc("wide_closed", float((bk.ia - bk.ib) < spread0))
        if n_msg % 200_000 == 0:
            log(f"{date}: {n_msg} messages, {time.time() - t0:.0f} s")
    if cur_sec is not None and bk is not None:
        bpx, bsz, apx, asz = bk.view()
        book_rows.append(((cur_sec + 1) * 1000, last_cts, *book_metrics(bpx, bsz, apx, asz)))
        n_upd.append(n_in_bin)
    acc.inc("messages", n_msg)
    out = finalize_calibration(acc, date)
    if real_out is not None:
        write_real_aggregates(real_out, tr, d0, book_rows, np.asarray(n_upd))
    out["seconds"] = time.time() - t0
    return out


def finalize_calibration(acc: CalAcc, date: str) -> dict[str, Any]:
    """Accumulators -> JSON-able calibration of one day (counts kept, so days can be pooled exactly)."""
    res = acc.res
    sizes = {}
    for key in list(res.v):
        sizes["|".join(str(k) for k in key)] = {"n": res.n[key], "q": res.quantiles(key)}
    return {"date": date, "aes": acc.aes, "vedges": acc.vedges, "risk": acc.risk, "n_add": acc.n_add, "n_can": acc.n_can,
            "vol_add": acc.vol_add, "vol_can": acc.vol_can, "mk_risk": acc.mk_risk, "mk_hit": acc.mk_hit, "trans": acc.trans,
            "extractor_version": 2,
            "occ_n": acc.occ_n, "occ_k": acc.occ_k, "sizes": sizes, "counts": acc.counts,
            "layout": {"D": D, "D_FINE": D_FINE, "BANDS": list(BANDS), "QBINS": [q if math.isfinite(q) else 1e9 for q in QBINS],
                       "NV": NV, "NS": NS, "step_ms": STEP_MS}}


BOOK_COLS = (["ts", "cts", "bid1", "ask1", "bid1_size", "ask1_size", "mid", "rel_spread_bp"]
             + [f"depth_{side}_{b}" for b in (5, 10, 25) for side in ("bid", "ask")] + ["imb_5", "imb_10", "span_bid_bp", "span_ask_bp"])


def book_frame(rows: list[tuple], n_updates: np.ndarray | None = None) -> pl.DataFrame:
    """Rows (ts_end_ms, cts, *book_metrics) -> the micro_days book_1s schema (dense 1 s grid, forward-filled; seconds
    without a row get n_updates = 0)."""
    df = pl.DataFrame(rows, schema=BOOK_COLS, orient="row").with_columns(pl.col("ts").cast(pl.Int64), pl.col("cts").cast(pl.Int64))
    nu = np.full(df.height, 10, dtype=np.int32) if n_updates is None else np.asarray(n_updates, dtype=np.int32)
    df = df.with_columns(pl.Series("n_updates", nu)).unique("ts", keep="last").sort("ts")
    grid = pl.DataFrame({"ts": np.arange(int(df["ts"][0]), int(df["ts"][-1]) + 1000, 1000, dtype=np.int64)})
    full = grid.join(df, on="ts", how="left").with_columns([pl.col(c).forward_fill() for c in BOOK_COLS if c != "ts"])
    return (full.with_columns(pl.col("n_updates").fill_null(0).cast(pl.Int32))
            .with_columns(pl.col("ts").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")))


def write_real_aggregates(out: Path, tr: dict[str, np.ndarray], d0: int, book_rows: list[tuple], n_upd: np.ndarray | None = None) -> None:
    from ai_trader.research import micro_days as md

    od = md.merge_orders(tr)
    for sub in ("trades_1s", "book_1s", "orders"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    md.bars_1s(tr, od, d0).write_parquet(out / "trades_1s" / f"{SYMBOL}.parquet", compression="zstd")
    pl.DataFrame(od).write_parquet(out / "orders" / f"{SYMBOL}.parquet", compression="zstd")
    book_frame(book_rows, n_upd).write_parquet(out / "book_1s" / f"{SYMBOL}.parquet", compression="zstd")


CAL_DAYS = ("2026-04-15", "2026-09-13", "2026-10-28")   # ob200 calibration days (none of them is a micro_days day)


def calibrate_day(date: str, man: Manifest | None = None, keep_raw: bool = False, log=print) -> dict[str, Any]:
    """download -> extract -> write data/research/obgen/cal/<date>.json + real/<date>/ -> verify -> delete raw (manifest)."""
    from ai_trader.research import micro_days as md

    man = man or Manifest()
    out_json = OUT / "cal" / f"{date}.json"
    if man.done("calibration", date) and out_json.is_file():
        log(f"{date}: calibration already done (manifest)")
        return json.loads(out_json.read_text(encoding="utf-8"))
    st0 = storage_gb()
    paths = fetch_day(date, man)
    if paths is None:
        raise RuntimeError("manifest says done but the calibration file is missing")
    tr = md.load_csv_trades(paths["trades"], day0_ms(date))
    cal = extract_calibration(paths["ob200"], tr, date, real_out=OUT / "real" / date, log=log)
    write_json(out_json, cal)
    ok = out_json.is_file() and all((OUT / "real" / date / s / f"{SYMBOL}.parquet").is_file() for s in ("trades_1s", "book_1s", "orders"))
    if not ok:
        raise RuntimeError(f"{date}: outputs missing, raw kept")
    man.put("calibration", date, {"status": "done", "messages": cal["counts"].get("messages"), "aes": cal["aes"],
                                  "seconds": cal.get("seconds"), "storage_before_gb": st0["total_gb"]})
    if not keep_raw:
        cleanup_day(date, man)
    log(f"{date}: calibration done, storage {storage_gb()['total_gb']:.2f} GB")
    return cal


# ----------------------------------------------------------------------------------------------------------------------
# pooled queue-reactive tables
# ----------------------------------------------------------------------------------------------------------------------

MIN_RISK = 300.0            # cells with fewer intervals at risk back off (pool v, then s, then neighbouring q bins)


class QSampler:
    """Mixture of per-day empirical quantile functions (weights = sample counts); `draw` inverts by interpolation."""

    def __init__(self, parts: list[tuple[float, list[float]]]):
        parts = [(w, q) for w, q in parts if q]
        self.w = np.asarray([w for w, _ in parts], dtype=float)
        self.q = np.asarray([q for _, q in parts], dtype=float) if parts else np.zeros((0, 41))
        self.w = self.w / self.w.sum() if len(self.w) else self.w
        self.grid = np.linspace(0, 1, self.q.shape[1]) if len(self.q) else np.linspace(0, 1, 41)
        self._ql = None
        self._cw = None

    @property
    def ok(self) -> bool:
        return len(self.w) > 0

    def draw(self, rng: np.random.Generator, n: int | None = None):
        if n is None:
            if self._ql is None:
                self._ql = self.q.tolist()
                self._cw = np.cumsum(self.w).tolist()
            u = rng.random()
            c = 0
            if len(self._cw) > 1:
                while c < len(self._cw) - 1 and u * 1.0 > self._cw[c]:
                    c += 1
                u = rng.random()
            row = self._ql[c]
            pos = u * (len(row) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(row) - 1)
            fr = pos - lo
            return row[lo] * (1 - fr) + row[hi] * fr
        k = n
        comp = rng.choice(len(self.w), size=k, p=self.w) if len(self.w) > 1 else np.zeros(k, dtype=int)
        u = rng.random(k)
        pos = u * (self.q.shape[1] - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, self.q.shape[1] - 1)
        fr = pos - lo
        x = self.q[comp, lo] * (1 - fr) + self.q[comp, hi] * fr
        return float(x[0]) if n is None else x

    def median(self) -> float:
        if not self.ok:
            return float("nan")
        return float(np.sum(self.w * self.q[:, self.q.shape[1] // 2]))


@dataclass
class QRTables:
    """Queue-reactive tables pooled over the calibration days (counts summed, then p = events / intervals at risk)."""
    aes: float                      # pooled average event size at the touch (BTC)
    p_add: np.ndarray               # (NB, NQ, NV, NS) per interval (near bands: 100 ms; deep bands: 1 s)
    p_can: np.ndarray
    occ: np.ndarray                 # (D,) occupancy of tick distance d
    samplers: dict                  # key -> QSampler (sizes in AES units, cancel sizes as queue fractions)
    days: list
    counts: dict
    vedges_ref: list
    touch_med_btc: float            # median real touch size (BTC) over the calibration days' book_1s
    depth_med_btc: float            # median real 200-level depth (BTC, both sides) over the calibration days' book_1s
    mk_rate: np.ndarray             # (NQ, NV) P(prints at the touch in a 100 ms interval | q, v): descriptive, lambda_M(q)
    touch_mult: np.ndarray = field(default_factory=lambda: np.ones((3, 2)))   # [prev none/add/cancel, (add, cancel)]

    def sampler(self, *key) -> QSampler | None:
        return self.samplers.get("|".join(str(k) for k in key))

    def size_sampler(self, kind: str, band: int, qb: int) -> QSampler:
        """Back off over neighbouring q bins, then over bands, until a sampler with data exists."""
        for dq in (0, -1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -9, 9, -13, 13):
            s = self.sampler(kind, band, qb + dq)
            if s is not None and s.ok:
                return s
        for db in (1, -1, 2, -2, 3, -3):
            for dq in range(0, NQ):
                for sg in (1, -1):
                    s = self.sampler(kind, band + db, qb + sg * dq)
                    if s is not None and s.ok:
                        return s
        raise KeyError((kind, band, qb))


def _backoff_probs(risk: np.ndarray, n: np.ndarray) -> np.ndarray:
    """p = n / risk per cell with backoff: (b,q,v,s) -> (b,q,.,s) -> (b,q,.,.) -> nearest q of (b,.,.,.) with data."""
    p = np.full(risk.shape, np.nan)
    ok = risk >= MIN_RISK
    p[ok] = n[ok] / risk[ok]
    r1, n1 = risk.sum(axis=2, keepdims=True), n.sum(axis=2, keepdims=True)
    p1 = np.where(r1 >= MIN_RISK, n1 / np.maximum(r1, 1), np.nan)
    p = np.where(np.isnan(p), np.broadcast_to(p1, p.shape), p)
    r2, n2 = risk.sum(axis=(2, 3), keepdims=True), n.sum(axis=(2, 3), keepdims=True)
    p2 = np.where(r2 >= MIN_RISK / 3, n2 / np.maximum(r2, 1), np.nan)
    p = np.where(np.isnan(p), np.broadcast_to(p2, p.shape), p)
    for b in range(risk.shape[0]):
        rb, nb = risk[b].sum(axis=(1, 2)), n[b].sum(axis=(1, 2))
        have = np.flatnonzero(rb >= 30)
        for q in range(risk.shape[1]):
            if np.isnan(p[b, q]).any():
                if len(have):
                    k = have[np.argmin(np.abs(have - q))]
                    p[b, q] = np.where(np.isnan(p[b, q]), nb[k] / rb[k], p[b, q])
                else:
                    p[b, q] = np.where(np.isnan(p[b, q]), 0.0, p[b, q])
    return np.clip(p, 0.0, 1.0)


def load_tables(days: tuple[str, ...] | list[str] = CAL_DAYS, root: Path = OUT) -> QRTables:
    cals = [json.loads((root / "cal" / f"{d}.json").read_text(encoding="utf-8")) for d in days]
    shp = (NB, NQ, NV, NS)
    risk = sum(np.asarray(c["risk"], float) for c in cals).reshape(shp)
    na = sum(np.asarray(c["n_add"], float) for c in cals).reshape(shp)
    nc = sum(np.asarray(c["n_can"], float) for c in cals).reshape(shp)
    occ = sum(np.asarray(c["occ_k"], float) for c in cals) / np.maximum(sum(np.asarray(c["occ_n"], float) for c in cals), 1.0)
    keys = set()
    for c in cals:
        keys |= set(c["sizes"])
    samplers = {k: QSampler([(c["sizes"][k]["n"], c["sizes"][k]["q"]) for c in cals if k in c["sizes"]]) for k in keys}
    w = np.asarray([c["counts"].get("intervals_unchanged", 1.0) for c in cals])
    aes = float(np.sum(w * np.asarray([c["aes"] for c in cals])) / w.sum())
    counts = {}
    for c in cals:
        for k, v in c["counts"].items():
            counts[k] = counts.get(k, 0.0) + float(v)
    tm, dm = [], []
    for d in days:
        b = pl.read_parquet(root / "real" / d / "book_1s" / f"{SYMBOL}.parquet")
        tm.append(np.concatenate([b["bid1_size"].to_numpy(), b["ask1_size"].to_numpy()]))
        dm.append(((b["depth_bid_25"] + b["depth_ask_25"]) / b["mid"]).to_numpy())
    mk_r = sum(np.asarray(c["mk_risk"], float) for c in cals)
    mk_h = sum(np.asarray(c["mk_hit"], float) for c in cals)
    tr_ = sum((np.asarray(c["trans"], float) for c in cals if "trans" in c), np.zeros((3, 3)))
    touch_mult = np.ones((3, 2))
    if tr_.sum() > 1000:
        marg = tr_.sum(axis=0) / tr_.sum()
        cond = tr_ / np.maximum(tr_.sum(axis=1, keepdims=True), 1.0)
        touch_mult[:, 0] = cond[:, 1] / max(marg[1], 1e-9)
        touch_mult[:, 1] = cond[:, 2] / max(marg[2], 1e-9)
    return QRTables(aes=aes, p_add=_backoff_probs(risk, na), p_can=_backoff_probs(risk, nc), occ=occ, samplers=samplers,
                    days=list(days), counts=counts, vedges_ref=cals[0]["vedges"],
                    touch_med_btc=float(np.nanmedian(np.concatenate(tm))), depth_med_btc=float(np.nanmedian(np.concatenate(dm))),
                    mk_rate=mk_h / np.maximum(mk_r, 1.0), touch_mult=touch_mult)


# ----------------------------------------------------------------------------------------------------------------------
# per-day calibration (FIT half of a micro_days day) and the external path
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class DayCal:
    """Per-day traits (H-MD1: they vary by day). Sizes: AES_day = tables.aes x s_touch; deep levels x s_deep / s_touch."""
    date: str
    s_touch: float = 1.0
    s_deep: float = 1.0
    vedges: list = field(default_factory=lambda: [0.0, 0.0])
    arrival: dict = field(default_factory=dict)       # expmix3 law dict of order durations (s)
    size: dict = field(default_factory=dict)          # SizeLaw dict (order notional, USD): fallback when size_q is empty
    size_q: list = field(default_factory=list)        # 513 empirical quantiles of the FIT-half order notional (USD)
    sign: dict = field(default_factory=dict)          # TranslatedSignModel dict (order signs)
    orders_per_s: float = 3.0
    theta_scale: float = 1.0                          # reprice threshold multiplier
    phi: float = 0.0                                  # informed share of the taker orders just before a reprice (0 = off)
    lam: float = 1e-7                                 # flow-bridge impact: log price per (USD of taker notional)^IMPACT_DELTA
    inform_ms: int = 100                              # window before a reprice in which informed orders arrive
    deep_corr: float = 1.0                            # deep-level size correction (indirect inference on the FIT median depth)
    notes: dict = field(default_factory=dict)


def fit_half_frames(date: str, half: str = "fit", root: Path = MICRO) -> dict[str, pl.DataFrame]:
    d0 = day0_ms(date)
    lo, hi = (d0, d0 + HALF_MS) if half == "fit" else (d0 + HALF_MS, d0 + DAY_MS)
    tr = pl.read_parquet(root / date / "trades_1s" / f"{SYMBOL}.parquet").filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
    bk = pl.read_parquet(root / date / "book_1s" / f"{SYMBOL}.parquet")
    bms = bk["ts"].dt.epoch("ms") if bk["ts"].dtype != pl.Int64 else bk["ts"]
    bk = bk.with_columns(bms.alias("_ms")).filter((pl.col("_ms") > lo) & (pl.col("_ms") <= hi)).drop("_ms")
    od = pl.read_parquet(root / date / "orders" / f"{SYMBOL}.parquet").filter((pl.col("ms") >= lo) & (pl.col("ms") < hi))
    return {"trades": tr, "book": bk, "orders": od, "lo": lo, "hi": hi}


def laws_from_orders(od: pl.DataFrame) -> dict[str, dict]:
    """Arrival / size / sign laws of a taker-order stream (for days without a micro_days laws file, e.g. the ob200
    calibration days): expmix3 of the order durations (s), log-normal body + CSN Pareto tail of the order notional,
    TranslatedSignModel from the empirical sign ACF (short lags 1-20 + power-law fit over lags 1-1000)."""
    from ai_trader.sim import laws as lw

    ms = np.sort(od["ms"].to_numpy())
    dur = np.diff(ms) / 1000.0
    dur = dur[dur > 0]
    arrival = lw.fit_law("expmix3", np.maximum(dur, 5e-4)).to_dict()
    v = od["notional"].to_numpy()
    tail = lw.fit_powerlaw(v)
    size = lw.fit_size_law(v, tail).to_dict()
    sg = od.sort("ms")["side"].to_numpy().astype(float)
    rho = lw.acf(sg, 1000)
    k, vv = lw.binned(rho, lw.log_bins(1000))
    pw = lw.fit_power_acf(k, vv, floor=2 / math.sqrt(len(sg)))
    short = {int(a): float(rho[a]) for a in (1, 2, 3, 5, 10, 20)}
    sign = lw.TranslatedSignModel(pw["c"], pw["gamma"], float((sg > 0).mean()), short).to_dict()
    return {"arrival": arrival, "size": size, "sign": sign}


def day_calibration(date: str, tables: QRTables, root: Path = MICRO) -> DayCal:
    """Per-day traits from the FIT half (00-12 UTC): queue scales from book_1s, laws from laws/BTCUSDT.json (FIT half)."""
    f = fit_half_frames(date, "fit", root)
    bk, tr = f["book"], f["trades"]
    touch = float(np.nanmedian(np.concatenate([bk["bid1_size"].to_numpy(), bk["ask1_size"].to_numpy()])))
    depth = float(np.nanmedian(((bk["depth_bid_25"] + bk["depth_ask_25"]) / bk["mid"]).to_numpy()))
    lp = root / date / "laws" / f"{SYMBOL}.json"
    if lp.is_file():
        laws = json.loads(lp.read_text(encoding="utf-8"))["halves"]["fit"]
        arrival, size, sign = (laws["durations"]["event"]["laws"]["expmix3"], laws["size"]["order"]["size_law"],
                               laws["signs"]["order"]["sign_model"])
    else:
        lf = laws_from_orders(f["orders"])
        arrival, size, sign = lf["arrival"], lf["size"], lf["sign"]
    last = np.full(HALF_MS // 1000, np.nan)
    sec = ((tr["ts"].to_numpy() - f["lo"]) // 1000).astype(int)
    last[sec] = tr["close"].to_numpy()
    last = pl.Series(last).fill_nan(None).forward_fill().backward_fill().to_numpy()
    vedges = vol_state_edges(rv60_from_last(last))
    return DayCal(date=date, s_touch=touch / tables.touch_med_btc, s_deep=depth / tables.depth_med_btc, vedges=vedges,
                  arrival=arrival, size=size, sign=sign, orders_per_s=f["orders"].height / (HALF_MS / 1000),
                  size_q=np.quantile(f["orders"]["notional"].to_numpy(), np.linspace(0, 1, 513)).tolist(),
                  notes={"touch_med_btc": touch, "depth_med_btc": depth})


@dataclass
class FlowPath:
    """External price-action / volume path: bars of `bar_ms` from t0_ms; close = mid (or last price) at each bar end;
    open0 = the price at t0. Optional per bar: rv (sum of squared 1 s log returns inside the bar), buy / sell taker
    notional (USD) and buy / sell taker order counts. Missing fields are filled by `complete`."""
    t0_ms: int
    bar_ms: int
    open0: float
    close: np.ndarray
    rv: np.ndarray | None = None
    buy_notional: np.ndarray | None = None
    sell_notional: np.ndarray | None = None
    n_buy: np.ndarray | None = None
    n_sell: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.close)

    def complete(self, cal: DayCal, rng: np.random.Generator, notional_per_s: float | None = None) -> FlowPath:
        """Fill missing fields: rv = max(r^2, 1e-10 bar_s) smoothed, order counts from the day's rate scaled by
        sqrt(rv / median rv), notional = count x the day's mean order notional, buy share 0.5 + 0.25 tanh(r / sigma)."""
        lp = np.log(np.r_[self.open0, self.close])
        r = np.diff(lp)
        bar_s = self.bar_ms / 1000
        rv = self.rv
        if rv is None:
            k = np.ones(15) / 15
            rv = np.maximum(np.convolve(r * r, k, mode="same"), 1e-10 * bar_s)
        act = np.sqrt(rv / np.median(rv))
        n_tot = self.n_buy + self.n_sell if self.n_buy is not None and self.n_sell is not None else rng.poisson(cal.orders_per_s * bar_s * act)
        sd = float(np.sqrt(np.median(rv)))
        share = 0.5 + 0.25 * np.tanh(r / max(sd, 1e-12))
        nb = self.n_buy if self.n_buy is not None else rng.binomial(n_tot, share)
        ns = self.n_sell if self.n_sell is not None else n_tot - nb
        mean_v = notional_per_s / max(cal.orders_per_s, 1e-9) if notional_per_s else float(np.exp(cal.size["mu"] + cal.size["sigma"] ** 2 / 2))
        bn = self.buy_notional if self.buy_notional is not None else nb * mean_v
        sn = self.sell_notional if self.sell_notional is not None else ns * mean_v
        return FlowPath(self.t0_ms, self.bar_ms, self.open0, self.close, rv, np.asarray(bn, float), np.asarray(sn, float),
                    np.asarray(nb, int), np.asarray(ns, int))


def real_path(date: str, lo_ms: int, hi_ms: int, bar_ms: int = 60_000, root: Path = MICRO) -> FlowPath:
    """Path from a real day: bar closes = book mid at each bar end (book_1s), rv = sum of squared 1 s mid log returns in
    the bar, taker buy / sell notional and order counts per bar from trades_1s."""
    tr = pl.read_parquet(root / date / "trades_1s" / f"{SYMBOL}.parquet").filter((pl.col("ts") >= lo_ms) & (pl.col("ts") < hi_ms))
    bk = pl.read_parquet(root / date / "book_1s" / f"{SYMBOL}.parquet")
    bms = (bk["ts"].dt.epoch("ms") if bk["ts"].dtype != pl.Int64 else bk["ts"]).to_numpy()
    n_sec = (hi_ms - lo_ms) // 1000
    mid = np.full(n_sec + 1, np.nan)
    k = (bms - lo_ms) // 1000          # book ts = end of second k-1 -> state at time lo + k s
    ok = (k >= 0) & (k <= n_sec)
    mid[k[ok]] = bk["mid"].to_numpy()[ok]
    mid = pl.Series(mid).fill_nan(None).forward_fill().backward_fill().to_numpy()
    lr = np.diff(np.log(mid))
    spb = bar_ms // 1000
    nb = n_sec // spb
    rv = (lr[: nb * spb] ** 2).reshape(nb, spb).sum(axis=1)
    close = mid[spb: nb * spb + 1: spb]
    b = ((tr["ts"].to_numpy() - lo_ms) // bar_ms).astype(int)
    okb = b < nb

    def agg(col):
        return np.bincount(b[okb], tr[col].to_numpy().astype(float)[okb], minlength=nb)

    return FlowPath(lo_ms, bar_ms, float(mid[0]), close, rv, agg("buy_notional"), agg("sell_notional"),
                agg("n_buy_orders").astype(int), agg("n_sell_orders").astype(int))


def bootstrap_path(p: FlowPath, rng: np.random.Generator, block_bars: int = 30, n_bars: int | None = None) -> FlowPath:
    """Stationary-in-blocks bootstrap of a path's bars (returns, rv, flows, counts move together): a new path with the
    same marginal bar dynamics, used for generated days that are not tied to one real price path."""
    n = n_bars or p.n
    lp = np.log(np.r_[p.open0, p.close])
    r = np.diff(lp)
    idx = []
    while len(idx) < n:
        s = int(rng.integers(0, max(p.n - block_bars, 1)))
        idx.extend(range(s, min(s + block_bars, p.n)))
    idx = np.asarray(idx[:n])
    close = p.open0 * np.exp(np.cumsum(r[idx]))
    pick = (lambda a: None if a is None else np.asarray(a)[idx])
    return FlowPath(p.t0_ms, p.bar_ms, p.open0, close, pick(p.rv), pick(p.buy_notional), pick(p.sell_notional), pick(p.n_buy), pick(p.n_sell))


def target_path(p: FlowPath, rng: np.random.Generator, step_ms: int = STEP_MS, flow: tuple[np.ndarray, np.ndarray] | None = None,
                lam: float = 0.0) -> np.ndarray:
    """Log reference price at every step start (n_bars x bar_ms / step_ms + 1 values). Inside each bar: the straight
    line from the previous close to the bar close, plus a FLOW BRIDGE lam (F(t) - (t/T) F(T)) with F the cumulative
    signed taker impact sum of sign x notional^IMPACT_DELTA of the bar up to t (permanent-within-the-bar impact of the generated orders, zero at the
    bar ends so the path's closes are kept exactly; per bar lam is capped so the flow part carries at most FLOW_QV_CAP of
    the bar's rv), plus a Brownian bridge whose variance tops the quadratic variation up to the bar's rv."""
    m = p.bar_ms // step_ms
    lp = np.log(np.r_[p.open0, p.close])
    rv = np.maximum(np.asarray(p.rv, float), 0.0)
    fb = np.zeros((p.n, m))
    if flow is not None and lam > 0 and len(flow[0]):
        ms, sv = flow
        k = np.clip((ms - p.t0_ms) // step_ms, 0, p.n * m - 1)           # order inside step k -> F counts it from k + 1
        inc = np.bincount(k + 1, sv, minlength=p.n * m + 1)[1:p.n * m + 1].reshape(p.n, m)
        F = np.cumsum(inc, axis=1)
        frac = np.arange(1, m + 1) / m
        unit = F - frac[None, :] * F[:, -1:]
        spb = max(1, 1000 // step_ms)                                       # QV measured on 1 s increments (clusters count)
        u1 = np.concatenate([np.zeros((p.n, 1)), unit[:, spb - 1::spb]], axis=1)
        s2 = (np.diff(u1, axis=1) ** 2).sum(axis=1)
        lam_b = np.minimum(lam, np.sqrt(FLOW_QV_CAP * rv / np.maximum(s2, 1e-30)))     # flow part <= FLOW_QV_CAP of rv
        fb = lam_b[:, None] * unit
        rv = rv - lam_b ** 2 * s2
    if flow is not None and len(flow[0]):
        # trade-time noise (Clark 1973): step variance proportional to the taker orders in the step (+ a floor), so
        # volatility clusters with the bursty arrivals inside the bar; total per bar = rv
        k = np.clip((flow[0] - p.t0_ms) // step_ms, 0, p.n * m - 1)
        cnt = np.bincount(k, minlength=p.n * m)[: p.n * m].reshape(p.n, m).astype(float)
        wts = cnt + TRADE_TIME_FLOOR * np.maximum(cnt.mean(axis=1, keepdims=True), 1e-3)
        wts /= wts.sum(axis=1, keepdims=True)
    else:
        wts = np.full((p.n, m), 1.0 / m)
    z = rng.standard_normal((p.n, m)) * np.sqrt(rv[:, None] * wts)
    w = np.cumsum(z, axis=1)
    frac = np.arange(1, m + 1) / m
    bridge = w - frac[None, :] * w[:, -1:]
    x = lp[:-1, None] + frac[None, :] * (lp[1:] - lp[:-1])[:, None] + bridge + fb
    return np.r_[lp[0], x.ravel()]


# ----------------------------------------------------------------------------------------------------------------------
# the generator
# ----------------------------------------------------------------------------------------------------------------------


def _fmt_px(i_abs: int) -> str:
    return f"{i_abs * TICK:.1f}"


def _lots(q: float) -> int:
    return int(round(q / QTY_STEP))


class EventSink:
    """Writes at-match JSONL events (or only counts them when path is None)."""

    def __init__(self, path: Path | None):
        self.f = open(path, "w", encoding="utf-8", newline="\n") if path is not None else None
        self.n = 0
        self.buf: list[str] = []

    def new(self, ts: int, oid: int, side: int, typ: str, i_abs: int | None, lots: int, owner: str) -> None:
        self.n += 1
        if self.f is None:
            return
        sd = "Buy" if side > 0 else "Sell"
        if typ == "Limit":
            self.buf.append(f'{{"ts_ms":{ts},"kind":"new","id":{oid},"side":"{sd}","type":"Limit","tif":"PostOnly","price":{_fmt_px(i_abs)},'
                            f'"qty":{lots * QTY_STEP:.3f},"reduce_only":false,"owner":"{owner}"}}\n')
        else:
            self.buf.append(f'{{"ts_ms":{ts},"kind":"new","id":{oid},"side":"{sd}","type":"Market","tif":"IOC","price":null,'
                            f'"qty":{lots * QTY_STEP:.3f},"reduce_only":false,"owner":"{owner}"}}\n')
        if len(self.buf) >= 20000:
            self.flush()

    def cancel(self, ts: int, oid: int, owner: str = "mk") -> None:
        self.n += 1
        if self.f is None:
            return
        self.buf.append(f'{{"ts_ms":{ts},"kind":"cancel","id":{oid},"owner":"{owner}"}}\n')
        if len(self.buf) >= 20000:
            self.flush()

    def flush(self) -> None:
        if self.f is not None and self.buf:
            self.f.write("".join(self.buf))
            self.buf.clear()

    def close(self) -> None:
        self.flush()
        if self.f is not None:
            self.f.close()


class GenBook:
    """The generator's own price-time FIFO book in lots on an absolute tick grid (the same matching as at-match for the
    events the generator emits: PostOnly makers that never cross, market IOC takers, cancels of whole orders)."""

    def __init__(self, center_tick: int, sink: EventSink):
        self.base = int(center_tick) - G // 2
        self.lots = {1: np.zeros(G, dtype=np.int64), -1: np.zeros(G, dtype=np.int64)}
        self.q: dict[tuple[int, int], list[list[int]]] = {}
        self.best = {1: -1, -1: -1}
        self.sink = sink
        self.next_id = 1
        self.trades: list[tuple[int, int, int, int]] = []      # (ts, taker side, price idx abs, lots)

    # abs tick index (tick count) <-> array index
    def a(self, i_abs: int) -> int:
        return i_abs - self.base

    def add(self, ts: int, side: int, i_abs: int, lots: int) -> None:
        if lots <= 0:
            return
        i = i_abs - self.base
        oid = self.next_id
        self.next_id += 1
        self.q.setdefault((side, i_abs), []).append([oid, lots])
        self.lots[side][i] += lots
        if side > 0 and (self.best[1] < 0 or i_abs > self.best[1]):
            self.best[1] = i_abs
        if side < 0 and (self.best[-1] < 0 or i_abs < self.best[-1]):
            self.best[-1] = i_abs
        self.sink.new(ts, oid, side, "Limit", i_abs, lots, "mk")

    def cancel_lots(self, ts: int, side: int, i_abs: int, lots: int) -> int:
        """Remove `lots` from the BACK of the level: whole orders are cancelled; when the last order is larger than what
        is left to remove, it is cancelled and its remainder re-posted at the back (same queue position: it was last)."""
        lv = self.q.get((side, i_abs))
        if not lv or lots <= 0:
            return 0
        removed = 0
        while lv and removed < lots:
            oid, ql = lv[-1]
            self.sink.cancel(ts, oid)
            lv.pop()
            need = lots - removed
            if ql <= need:
                removed += ql
            else:
                removed += need
                nid = self.next_id
                self.next_id += 1
                lv.append([nid, ql - need])
                self.sink.new(ts, nid, side, "Limit", i_abs, ql - need, "mk")
        self.lots[side][i_abs - self.base] -= removed
        if not lv:
            del self.q[(side, i_abs)]
            if self.best[side] == i_abs:
                self._rescan(side)
        return removed

    def cancel_level(self, ts: int, side: int, i_abs: int) -> int:
        return self.cancel_lots(ts, side, i_abs, int(self.lots[side][i_abs - self.base]))

    def _rescan(self, side: int) -> None:
        b = self.best[side]
        arr = self.lots[side]
        i = b - self.base
        if side > 0:
            lo = max(i - 8000, 0)
            nz = np.flatnonzero(arr[lo:i + 1])
            self.best[1] = int(nz[-1]) + lo + self.base if len(nz) else -1
        else:
            nz = np.flatnonzero(arr[i:i + 8000])
            self.best[-1] = int(nz[0]) + i + self.base if len(nz) else -1

    def market(self, ts: int, side: int, lots: int, band_frac: float = 0.01) -> int:
        """Taker IOC: walks the opposite side from its best within the protection band; returns the lots filled."""
        if lots <= 0:
            return 0
        oid = self.next_id
        self.next_id += 1
        self.sink.new(ts, oid, side, "Market", None, lots, "tk")
        opp = -side
        left = lots
        b0 = self.best[opp]
        if b0 < 0:
            return 0
        limit = int(math.floor(b0 * (1 + band_frac) + 1e-9)) if side > 0 else int(math.ceil(b0 * (1 - band_frac) - 1e-9))
        while left > 0 and self.best[opp] >= 0:
            p = self.best[opp]
            if (side > 0 and p > limit) or (side < 0 and p < limit):
                break
            lv = self.q[(opp, p)]
            while lv and left > 0:
                o = lv[0]
                take = min(o[1], left)
                o[1] -= take
                left -= take
                self.lots[opp][p - self.base] -= take
                self.trades.append((ts, side, p, take))
                if o[1] == 0:
                    lv.pop(0)
            if not lv:
                del self.q[(opp, p)]
                self._rescan(opp)
        return lots - left

    def rel(self, side: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        b = self.best[side]
        if side > 0:
            ix = b - np.arange(n)
        else:
            ix = b + np.arange(n)
        return ix, self.lots[side][ix - self.base]

    def view_metrics(self, depth: int = DEPTH_LEVELS) -> tuple[float, ...]:
        from ai_trader.sim.book import book_metrics

        bi, ai = self.best[1] - self.base, self.best[-1] - self.base
        lo = max(bi - 3000, 0)
        nb = np.flatnonzero(self.lots[1][lo:bi + 1])[::-1][:depth] + lo
        na = (np.flatnonzero(self.lots[-1][ai:ai + 3000]) + ai)[:depth]
        return book_metrics(np.round((nb + self.base) * TICK, 1), self.lots[1][nb] * QTY_STEP, np.round((na + self.base) * TICK, 1),
                            self.lots[-1][na] * QTY_STEP)


@dataclass
class GenOut:
    book_rows: list
    trades: np.ndarray            # structured (ms, side, price, size)
    n_events: int
    stats: dict


class Generator:
    """Queue-reactive generator for one path (module docstring, step 4)."""

    def __init__(self, tables: QRTables, cal: DayCal, seed: int = 0):
        self.T, self.cal = tables, cal
        self.rng = np.random.default_rng(seed)
        self.aes_lots = tables.aes * cal.s_touch / QTY_STEP
        self.deep_mult = cal.s_deep / max(cal.s_touch, 1e-9) * cal.deep_corr
        self.band_near = BAND_OF[:D_FINE]
        self.band_deep = BAND_OF[D_FINE:]
        # per-cell samplers are resolved lazily and cached
        self._samp: dict = {}
        self.flick_q = tables.sampler("flick")
        self.touch_prev = {1: (0, 0, -1), -1: (0, 0, -1)}       # side -> (prev event type, prev add lots, touch price)
        self.move_q = tables.sampler("move_ticks")
        self.imp_q = tables.sampler("imp_new")
        self.keep_q = tables.sampler("old_keep")
        self.rec_q = tables.sampler("rec_new")

    def _s(self, kind: str, band: int, qb: int) -> QSampler:
        k = (kind, band, qb)
        s = self._samp.get(k)
        if s is None:
            s = self.T.size_sampler(kind, band, qb)
            self._samp[k] = s
        return s

    def _prof_sampler(self, b: int) -> QSampler:
        s = self._samp.get(("prof", b))
        if s is None:
            for db in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
                bb = b + db
                if 0 <= bb < NB and (bb > 0 or b == 0):          # never borrow the touch law for a deeper level
                    s = self.T.sampler("prof", bb)
                    if s is not None and s.ok:
                        break
            self._samp[("prof", b)] = s
        return s

    def _prof_lots(self, d: int) -> int:
        b = int(BAND_OF[min(d, D - 1)])
        s = self._prof_sampler(b)
        mult = self.deep_mult if d >= D_FINE else 1.0
        return max(1, int(round(s.draw(self.rng) * self.aes_lots * mult)))

    # ---------------------------------------------------------------- book initialisation and window maintenance
    def _populate(self, bk: GenBook, ts: int, side: int, d_from: int, d_to: int) -> None:
        """Fill empty tick distances d_from..d_to-1 of `side` (relative to its best) from the stationary profile."""
        b = bk.best[side]
        for d in range(d_from, d_to):
            if self.rng.random() < self.T.occ[min(d, D - 1)]:
                i_abs = b - d if side > 0 else b + d
                if bk.lots[side][i_abs - bk.base] == 0:
                    bk.add(ts, side, i_abs, self._prof_lots(d))

    def _trim(self, bk: GenBook, ts: int, side: int) -> None:
        """Cancel the orders of `side` farther than D ticks from its best (they would leave the modelled window)."""
        b = bk.best[side]
        far = [(s, i) for (s, i) in list(bk.q) if s == side and ((side > 0 and i <= b - D) or (side < 0 and i >= b + D))]
        for s, i in far:
            bk.cancel_level(ts, s, i)

    def _redraw_entering(self, bk: GenBook, ts: int, side: int, k: int) -> None:
        """After the improving side shifted by k ticks, its levels that crossed into the deep region (now at
        D_FINE..D_FINE+k-1) are re-drawn from the deep profile: market makers' far ladders follow the price."""
        if k <= 0:
            return
        b = bk.best[side]
        for d in range(D_FINE, min(D_FINE + k, D)):
            i_abs = b - d if side > 0 else b + d
            if bk.lots[side][i_abs - bk.base] > 0:
                bk.cancel_level(ts, side, i_abs)
            if self.rng.random() < self.T.occ[d]:
                bk.add(ts, side, i_abs, self._prof_lots(d))

    def init_book(self, bk: GenBook, ts: int, mid_tick: float) -> None:
        bid = int(math.floor(mid_tick))
        ask = bid + 1
        bk.add(ts, 1, bid, self._prof_lots(0))
        bk.add(ts, -1, ask, self._prof_lots(0))
        self._populate(bk, ts, 1, 1, D)
        self._populate(bk, ts, -1, 1, D)

    def reprice(self, bk: GenBook, ts: int, new_bid: int) -> None:
        """Reference-price move: the book is re-quoted around new_bid / new_bid + 1 (transition laws of the calibration
        days: old touch keeps a fraction, the improving side posts a new touch, the receding side's stale levels inside
        the new spread are cancelled and its new touch is the existing level or a fresh one)."""
        ob, oa = bk.best[1], bk.best[-1]
        up = (new_bid + 0.5) > 0.5 * (ob + oa)
        new_ask = new_bid + 1
        if up:
            # receding = ask: cancel asks at or below new_bid
            while bk.best[-1] >= 0 and bk.best[-1] <= new_bid:
                bk.cancel_level(ts, -1, bk.best[-1])
            if bk.best[-1] != new_ask:
                bk.add(ts, -1, new_ask, max(1, int(round(self.rec_q.draw(self.rng) * self.aes_lots))))
            # improving = bid: old touch keeps a fraction, new touch at new_bid
            if ob >= 0:
                keep = self.keep_q.draw(self.rng)
                ql = int(bk.lots[1][ob - bk.base])
                bk.cancel_lots(ts, 1, ob, ql - int(round(keep * ql)))
            if bk.lots[1][new_bid - bk.base] == 0:
                bk.add(ts, 1, new_bid, max(1, int(round(self.imp_q.draw(self.rng) * self.aes_lots))))
            self._trim(bk, ts, 1)
            self._redraw_entering(bk, ts, 1, new_bid - ob if ob >= 0 else 0)
            if oa >= 0:
                k = bk.best[-1] - oa
                if k > 0:
                    self._populate(bk, ts, -1, max(D - k, 1), D)
        else:
            while bk.best[1] >= 0 and bk.best[1] >= new_ask:
                bk.cancel_level(ts, 1, bk.best[1])
            if bk.best[1] != new_bid:
                bk.add(ts, 1, new_bid, max(1, int(round(self.rec_q.draw(self.rng) * self.aes_lots))))
            if oa >= 0:
                keep = self.keep_q.draw(self.rng)
                ql = int(bk.lots[-1][oa - bk.base])
                bk.cancel_lots(ts, -1, oa, ql - int(round(keep * ql)))
            if bk.lots[-1][new_ask - bk.base] == 0:
                bk.add(ts, -1, new_ask, max(1, int(round(self.imp_q.draw(self.rng) * self.aes_lots))))
            self._trim(bk, ts, -1)
            self._redraw_entering(bk, ts, -1, oa - new_ask if oa >= 0 else 0)
            if ob >= 0:
                k = ob - bk.best[1]
                if k > 0:
                    self._populate(bk, ts, 1, max(D - k, 1), D)

    def _heal(self, bk: GenBook, ts: int) -> None:
        """A side emptied by a sweep is re-quoted one tick from the other side (then refilled from the profile)."""
        if bk.best[1] < 0 and bk.best[-1] >= 0:
            bk.add(ts, 1, bk.best[-1] - 1, max(1, int(round(self.rec_q.draw(self.rng) * self.aes_lots))))
            self._populate(bk, ts, 1, 1, D)
        elif bk.best[-1] < 0 and bk.best[1] >= 0:
            bk.add(ts, -1, bk.best[1] + 1, max(1, int(round(self.rec_q.draw(self.rng) * self.aes_lots))))
            self._populate(bk, ts, -1, 1, D)

    # ---------------------------------------------------------------- queue-reactive level events
    def level_events(self, bk: GenBook, ts: int, v: int, near: bool) -> None:
        spread = bk.best[-1] - bk.best[1]
        s = 0 if spread <= 1 else 1
        if near:
            d_lo, d_hi, bands = 0, D_FINE, self.band_near
        else:
            d_lo, d_hi, bands, s = D_FINE, D, self.band_deep, 0
        ev = []
        for side in (1, -1):
            ix, q = bk.rel(side, d_hi)
            ix, q = ix[d_lo:], q[d_lo:]
            mult = 1.0 if near else self.deep_mult
            qb = qbin(q / (self.aes_lots * mult))
            c = (bands, qb, v, s)
            pa = self.T.p_add[c]
            pc = self.T.p_can[c] * (q > 0)
            if near:
                pt, _, ppx = self.touch_prev[side]
                if ppx != int(ix[0]):
                    pt = 0
                pa = pa.copy()
                pc = pc.copy()
                pa[0] *= self.T.touch_mult[pt, 0]
                pc[0] *= self.T.touch_mult[pt, 1]
                tot = pa[0] + pc[0]
                if tot > 0.98:
                    pa[0] *= 0.98 / tot
                    pc[0] *= 0.98 / tot
            u = self.rng.random(len(q))
            for k in np.flatnonzero(u < pa + pc):
                ev.append((side, int(ix[k]), int(q[k]), int(bands[k]), int(qb[k]), bool(u[k] < pa[k]), mult, k + d_lo))
        if near:
            touched = {e_[0] for e_ in ev if e_[7] == 0}
            for side in (1, -1):
                if side not in touched:
                    self.touch_prev[side] = (0, 0, bk.best[side])
        if not ev:
            return
        for j in self.rng.permutation(len(ev)):
            side, i_abs, ql, band, qbn, is_add, mult, d = ev[j]
            ql = int(bk.lots[side][i_abs - bk.base])          # the level may have changed since the draw
            if is_add:
                x = self._s("add", band, qbn).draw(self.rng)
                n = max(1, int(round(x * self.aes_lots * mult)))
                bk.add(ts, side, i_abs, n)
                if near and d == 0:
                    self.touch_prev[side] = (1, n, i_abs)
            elif ql > 0:
                pt, plots, ppx = self.touch_prev[side] if d == 0 else (0, 0, -1)
                if d == 0 and pt == 1 and ppx == i_abs and self.flick_q is not None and self.flick_q.ok:
                    n = int(round(self.flick_q.draw(self.rng) * plots))     # cancel of the order just placed (flicker)
                else:
                    n = int(round(self._s("canf", band, qbn).draw(self.rng) * ql))
                if d == 0:
                    n = min(n, ql - 1)                       # tables come from intervals without a touch depletion
                    self.touch_prev[side] = (2, 0, i_abs)
                if n > 0:
                    bk.cancel_lots(ts, side, i_abs, n)

    def deep_refresh(self, bk: GenBook, ts: int) -> None:
        """Levels d >= D_FINE: each is re-drawn from the stationary profile (occupancy x size of its band) at rate
        DEEP_REFRESH_PER_S. This keeps the 200-level depth profile at its calibrated law with few events; the deep
        queue-reactive tables are estimated and reported but deliberately not used (their per-level chain, estimated
        on a moving 200-level window, does not preserve the profile, and deep turnover affects none of the facts)."""
        for side in (1, -1):
            u = self.rng.random(D - D_FINE)
            for k in np.flatnonzero(u < DEEP_REFRESH_PER_S):
                d = int(k) + D_FINE
                i_abs = bk.best[side] - d if side > 0 else bk.best[side] + d
                if bk.lots[side][i_abs - bk.base] > 0:
                    bk.cancel_level(ts, side, i_abs)
                if self.rng.random() < self.T.occ[d]:
                    bk.add(ts, side, i_abs, self._prof_lots(d))

    def sweep_heal(self, bk: GenBook, ts: int, taker_side: int) -> None:
        """After a taker emptied the touch and left a gap (spread >= 2 ticks), the book is re-quoted at once (real
        pushes show a 1-tick spread > 99.8% of the time): with probability HEAL_REFILL the swept side refills one tick
        from the other best (fresh touch law), else the other side follows the price into the gap (improving touch law)."""
        if bk.best[1] < 0 or bk.best[-1] < 0:
            self._heal(bk, ts)
            return
        if bk.best[-1] - bk.best[1] < 2 or self.rng.random() > HEAL_NOW:
            return
        swept = -taker_side
        if self.rng.random() < HEAL_REFILL:
            i_abs = bk.best[1] + 1 if swept < 0 else bk.best[-1] - 1
            bk.add(ts, swept, i_abs, max(1, int(round(self.rec_q.draw(self.rng) * self.aes_lots))))
        else:
            i_abs = bk.best[-1] - 1 if swept < 0 else bk.best[1] + 1
            bk.add(ts, -swept, i_abs, max(1, int(round(self.imp_q.draw(self.rng) * self.aes_lots))))

    # ---------------------------------------------------------------- taker orders
    def taker_orders(self, p: FlowPath, moves: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ms, side, lots) of every taker order of the path: per bar the path's buy / sell counts and notional, times by
        the day's bursty duration law compressed into the bar, long-memory signs, informed share phi of the orders in the
        `inform_ms` before a reprice takes the move's direction (counts per bar restored by flipping uninformed ones)."""
        from ai_trader.sim.laws import SizeLaw, law_from_dict

        rng = self.rng
        nb_, ns_ = np.asarray(p.n_buy, int), np.asarray(p.n_sell, int)
        n_tot = nb_ + ns_
        N = int(n_tot.sum())
        if N == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0, np.int64)
        arr = law_from_dict(self.cal.arrival)
        dur = arr.sample(N + p.n, rng)
        ms = np.empty(N, np.int64)
        pos = 0
        dpos = 0
        for b in range(p.n):
            k = n_tot[b]
            if k == 0:
                dpos += 1
                continue
            x = dur[dpos:dpos + k + 1]
            dpos += k + 1
            c = np.cumsum(x)
            t = p.t0_ms + b * p.bar_ms + np.floor(c[:-1] / c[-1] * p.bar_ms).astype(np.int64)
            ms[pos:pos + k] = t
            pos += k
        sm = law_from_dict(self.cal.sign)
        sg = sm.sample(N, rng).astype(np.int8)
        # per-bar buy / sell counts of the path imposed on the long-memory sign sequence (random flips of the excess)
        start = np.r_[0, np.cumsum(n_tot)[:-1]]
        for b in range(p.n):
            k = n_tot[b]
            if k == 0:
                continue
            sl = slice(start[b], start[b] + k)
            s_b = sg[sl]
            need = int(nb_[b]) - int((s_b > 0).sum())
            if need != 0:
                frm = -1 if need > 0 else 1
                cand = np.flatnonzero(s_b == frm)
                fl = rng.choice(cand, size=min(abs(need), len(cand)), replace=False)
                s_b[fl] = -frm
                sg[sl] = s_b
        # informed: orders in (t_move - inform_ms, t_move] take the move's sign with probability phi (after the count
        # step, so the bar counts move with the informed flow; the bar notional per side is still imposed below)
        inf = np.zeros(N, bool)
        if moves and self.cal.phi > 0:
            mt = np.asarray([m[0] for m in moves], np.int64)
            md = np.asarray([m[1] for m in moves], np.int8)
            j = np.searchsorted(mt, ms, side="left")          # first move at or after the order
            okj = j < len(mt)
            jj = np.minimum(j, len(mt) - 1)
            near = okj & (mt[jj] - ms <= self.cal.inform_ms) & (mt[jj] >= ms)
            pick = near & (rng.random(N) < self.cal.phi)
            sg[pick] = md[jj[pick]]
            inf = pick
        # sizes per bar and side: drawn from the day's law and assigned by the rank of a latent score (a slow AR(1)
        # level in order time plus the order's alignment with the local sign run, so large orders cluster in time and
        # go with the prevailing flow); then scaled to the bar's notional (factor clipped to [1/3, 3]), any remaining
        # deficit added to the side's largest order
        if self.cal.size_q:
            lq = np.log(np.maximum(np.asarray(self.cal.size_q, float), 1e-6))
            pos = rng.random(N) * (len(lq) - 1)
            lo_ = np.floor(pos).astype(int)
            v = np.exp(lq[lo_] + (pos - lo_) * (lq[np.minimum(lo_ + 1, len(lq) - 1)] - lq[lo_]))
        else:
            sl_ = law_from_dict(self.cal.size)
            assert isinstance(sl_, SizeLaw)
            v = sl_.sample(N, rng)
        lvl = np.zeros(N)
        e = rng.standard_normal(N) * math.sqrt(1 - SIZE_PHI ** 2)
        acc_b = acc_s = 0.0                                     # one slow size level per side (big buys cluster with big buys)
        for i in range(N):
            if sg[i] > 0:
                acc_b = SIZE_PHI * acc_b + e[i]
                lvl[i] = acc_b
            else:
                acc_s = SIZE_PHI * acc_s + e[i]
                lvl[i] = acc_s
        run = np.zeros(N)
        ew = 0.0
        for i in range(N):
            run[i] = ew
            ew = 0.9 * ew + 0.1 * sg[i]
        score = lvl + SIZE_RUN_COUPLING * sg * run + INFORMED_SIZE_BOOST * inf + 0.5 * rng.standard_normal(N)
        px = np.repeat(np.r_[p.open0, p.close[:-1]], n_tot)
        lots = np.zeros(N, np.int64)
        for b in range(p.n):
            k = n_tot[b]
            if k == 0:
                continue
            sl = slice(start[b], start[b] + k)
            for side, tgt in ((1, p.buy_notional[b]), (-1, p.sell_notional[b])):
                m = np.flatnonzero(sg[sl] == side)
                if len(m) == 0:
                    continue
                vv = np.sort(v[sl][m])
                order = np.argsort(score[sl][m])
                vs = np.empty(len(m))
                vs[order] = vv                                  # higher score -> larger order
                tot = vs.sum()
                dominant = tgt >= (p.sell_notional[b] if side > 0 else p.buy_notional[b])
                if tot > 0 and tgt > 0:
                    if tot > tgt:
                        # excess removed from the orders above the side's median (the median order size is kept)
                        big = vs > np.median(vs)
                        sb = vs[big].sum()
                        if sb > 0:
                            vs[big] *= max((sb - (tot - tgt)) / sb, 0.2)
                    elif not dominant:
                        vs *= min(tgt / tot, 3.0)                # the minor side is scaled, no metaorder
                    deficit = tgt - vs.sum()
                    if deficit > 0:
                        # spread over the top decile of the side's scores (clustered in time: a metaorder)
                        top = order[-max(1, int(math.ceil(DEFICIT_SHARE * len(order)))):]
                        vs[top] += deficit * vs[top] / vs[top].sum()
                lots[start[b] + m] = np.maximum(1, np.round(vs / px[sl][m] / QTY_STEP)).astype(np.int64)
        lots = np.minimum(lots, int(100 / QTY_STEP))
        # same-ms same-side neighbours would merge into one order: shift the later one by 1 ms
        o = np.argsort(ms, kind="stable")
        ms, sg, lots = ms[o], sg[o], lots[o]
        for i in range(1, N):
            if ms[i] <= ms[i - 1] and sg[i] == sg[i - 1]:
                ms[i] = ms[i - 1] + 1
        o = np.argsort(ms, kind="stable")
        return ms[o], sg[o], lots[o]

    def move_schedule(self, x: np.ndarray, t0_ms: int, mid0_tick: float) -> list[tuple[int, int, int]]:
        """Threshold reprice schedule on the target path: when |target - mid| (ticks) reaches the current threshold
        (theta_scale x a draw from the calibration days' move-size law), the reference mid jumps to the target.
        Returns (ts_ms, direction, new_bid_tick)."""
        tgt = np.exp(x) / TICK
        mid = mid0_tick
        th = max(1.0, self.cal.theta_scale * self.move_q.draw(self.rng))
        out = []
        for k in range(1, len(tgt)):
            e = tgt[k] - mid
            if abs(e) >= th:
                new_bid = int(math.floor(tgt[k]))
                new_mid = new_bid + 0.5
                if new_mid != mid:
                    out.append((t0_ms + k * STEP_MS, 1 if new_mid > mid else -1, new_bid))
                    mid = new_mid
                th = max(1.0, self.cal.theta_scale * self.move_q.draw(self.rng))
        return out

    # ---------------------------------------------------------------- run
    def run(self, p: FlowPath, events_path: Path | None = None, warmup_s: int = 300, record_100ms: bool = False,
            diag: bool = False) -> GenOut:
        rng = self.rng
        prof_sum = np.zeros(D)
        prof_occ = np.zeros(D)
        prof_n = 0
        n_steps = p.n * p.bar_ms // STEP_MS
        t0 = p.t0_ms
        if self.cal.phi > 0:
            # informed takers need the move schedule: target without the flow bridge first
            x = target_path(p, rng)
            moves = self.move_schedule(x, t0, math.floor(math.exp(x[0]) / TICK) + 0.5)
            t_ms, t_sd, t_lots = self.taker_orders(p, [(m[0], m[1]) for m in moves])
        else:
            t_ms, t_sd, t_lots = self.taker_orders(p, [])
            bar = np.clip((t_ms - t0) // p.bar_ms, 0, p.n - 1)
            px = np.r_[p.open0, p.close][bar]
            x = target_path(p, rng, flow=(t_ms, t_sd * (t_lots * QTY_STEP * px) ** IMPACT_DELTA), lam=self.cal.lam)
            moves = self.move_schedule(x, t0, math.floor(math.exp(x[0]) / TICK) + 0.5)
        mid0 = math.exp(x[0]) / TICK
        tgt = np.exp(x) / TICK                                  # reference price in ticks at every step start
        steps_per_bar = p.bar_ms // STEP_MS
        ref_mid = math.floor(mid0) + 0.5
        th = max(1.0, self.cal.theta_scale * self.move_q.draw(rng))
        off = 0.0                                               # sweep displacement carried by the reference (ticks)
        sink = EventSink(events_path)
        bk = GenBook(int(round(mid0)), sink)
        ts_w = t0 - warmup_s * 1000
        self.init_book(bk, ts_w, mid0)
        vedges = self.cal.vedges
        # warm-up: queue dynamics only, flat reference price
        for k in range(warmup_s * 1000 // STEP_MS):
            ts = ts_w + k * STEP_MS
            self.level_events(bk, ts, 1, True)
            if k % 10 == 0:
                self.deep_refresh(bk, ts)
        bk.trades.clear()
        rows = []
        rows100 = []
        last_trade_px = mid0 * TICK
        rv_hist = np.zeros(60)
        rv_sum = 0.0
        prev_lp = math.log(last_trade_px)
        v = 1
        j = 0
        n_tk = len(t_ms)
        n_moves = 0
        n_sweeps = 0
        n_wide = 0
        for k in range(n_steps):
            ts = t0 + k * STEP_MS
            if k % 10 == 0:
                sec = k // 10
                if sec > 0:
                    rows.append((ts, ts, *bk.view_metrics()))
                    if diag:
                        for sd_ in (1, -1):
                            _, qq = bk.rel(sd_, D)
                            prof_sum += qq
                            prof_occ += qq > 0
                            prof_n += 1
                    lp = math.log(last_trade_px)
                    r2 = (lp - prev_lp) ** 2
                    prev_lp = lp
                    rv_sum += r2 - rv_hist[sec % 60]
                    rv_hist[sec % 60] = r2
                    v = vstate_of(rv_sum, vedges)
                self.deep_refresh(bk, ts)
            # threshold reprice on the reference (+ the decaying sweep offset, zero again at the bar end)
            rem = steps_per_bar - (k % steps_per_bar)
            off *= (rem - 1) / rem if rem > 1 else 0.0
            ref = tgt[k] + off
            if abs(ref - ref_mid) >= th and int(math.floor(ref)) + 0.5 != ref_mid:
                nbid = int(math.floor(ref))
                self.reprice(bk, ts, nbid)
                ref_mid = nbid + 0.5
                th = max(1.0, self.cal.theta_scale * self.move_q.draw(rng))
                n_moves += 1
            else:
                self.level_events(bk, ts, v, True)
            if bk.best[-1] - bk.best[1] >= 2:
                n_wide += 1
                # spread closing (calibration: wide states close within one push almost always)
                if rng.random() < 0.95:
                    side = 1 if rng.random() < 0.5 else -1
                    i_abs = bk.best[-1] - 1 if side > 0 else bk.best[1] + 1
                    bk.add(ts, side, i_abs, max(1, int(round(self.rec_q.draw(rng) * self.aes_lots))))
            hi = ts + STEP_MS
            while j < n_tk and t_ms[j] < hi:
                tms = max(int(t_ms[j]), ts + 1)
                nt0 = len(bk.trades)
                b0, a0 = bk.best[1], bk.best[-1]
                bk.market(tms, int(t_sd[j]), int(t_lots[j]))
                if len(bk.trades) > nt0:
                    last_trade_px = bk.trades[-1][2] * TICK
                if bk.best[1] != b0 or bk.best[-1] != a0:
                    n_sweeps += 1
                    self.sweep_heal(bk, tms, int(t_sd[j]))
                    if bk.best[1] >= 0 and bk.best[-1] >= 0:
                        new_mid = 0.5 * (bk.best[1] + bk.best[-1])
                        off += new_mid - ref_mid                # the sweep's displacement is kept (permanent in the bar)
                        ref_mid = new_mid
                j += 1
            if record_100ms:
                rows100.append((hi, hi, *bk.view_metrics()))
        rows.append((t0 + n_steps * STEP_MS, t0 + n_steps * STEP_MS, *bk.view_metrics()))
        sink.close()
        tr = np.asarray(bk.trades, dtype=np.int64).reshape(-1, 4)
        stats = {"n_events": sink.n, "n_moves": n_moves, "n_sweeps": n_sweeps, "n_wide_steps": n_wide, "n_taker": n_tk,
                 "n_prints": int(tr.shape[0]), "steps": n_steps}
        if record_100ms:
            stats["rows100"] = rows100
        if diag:
            stats["profile_mean_btc"] = (prof_sum / max(prof_n, 1) * QTY_STEP).tolist()
            stats["profile_occ"] = (prof_occ / max(prof_n, 1)).tolist()
        return GenOut(rows, tr, sink.n, stats)


# ----------------------------------------------------------------------------------------------------------------------
# aggregation to the micro_days schema
# ----------------------------------------------------------------------------------------------------------------------


def prints_dict(trades: np.ndarray) -> dict[str, np.ndarray]:
    """(ts, taker side, price tick, lots) -> {ms, side, price, size} sorted by ms (archive order kept)."""
    if trades.size == 0:
        return {"ms": np.zeros(0, np.int64), "side": np.zeros(0, np.int8), "price": np.zeros(0), "size": np.zeros(0)}
    o = np.argsort(trades[:, 0], kind="stable")
    t = trades[o]
    return {"ms": t[:, 0].astype(np.int64), "side": t[:, 1].astype(np.int8), "price": np.round(t[:, 2] * TICK, 1),
            "size": np.round(t[:, 3] * QTY_STEP, 3)}


def aggregate(out_dir: Path, prints: dict[str, np.ndarray], book_rows: list[tuple], d0: int, n_updates: np.ndarray | None = None) -> dict[str, int]:
    """Write trades_1s / book_1s / orders parquet files (micro_days schema) for generated prints and book states."""
    from ai_trader.research import micro_days as md

    od = md.merge_orders(prints)
    for sub in ("trades_1s", "book_1s", "orders"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    t1 = md.bars_1s(prints, od, d0)
    t1.write_parquet(out_dir / "trades_1s" / f"{SYMBOL}.parquet", compression="zstd")
    pl.DataFrame(od).write_parquet(out_dir / "orders" / f"{SYMBOL}.parquet", compression="zstd")
    b1 = book_frame(book_rows, n_updates)
    b1.write_parquet(out_dir / "book_1s" / f"{SYMBOL}.parquet", compression="zstd")
    return {"trades_1s": t1.height, "book_1s": b1.height, "orders": len(od["ms"])}


# ----------------------------------------------------------------------------------------------------------------------
# the matching engine (atx match-sim) and its outputs
# ----------------------------------------------------------------------------------------------------------------------

RULES_BTC = {"symbol": SYMBOL, "tick_size": TICK, "qty_step": QTY_STEP, "min_qty": QTY_STEP, "max_qty": 1190.0, "max_mkt_qty": 119.0,
             "min_notional": 5.0, "max_leverage": 100.0}
_ATX: dict[str, Any] = {}


def atx_binary() -> Path | None:
    """The atx binary when it has the `match-sim` subcommand (agent F4), else None. Release preferred; cached."""
    if "bin" in _ATX:
        return _ATX["bin"]
    found = None
    for p in (ROOT / "target" / "release" / "atx.exe", ROOT / "target" / "release" / "atx", ROOT / "target" / "debug" / "atx.exe",
              ROOT / "target" / "debug" / "atx"):
        if p.is_file():
            try:
                r = subprocess.run([str(p), "match-sim", "--help"], capture_output=True, text=True, timeout=60)
            except Exception:  # noqa: BLE001
                continue
            if r.returncode == 0 and "--events" in r.stdout:
                found = p
                break
    _ATX["bin"] = found
    return found


def run_engine(events: Path, out_dir: Path, depth: int = DEPTH_LEVELS, snapshot_ms: int = STEP_MS, timeout: float = 3600) -> dict[str, Any]:
    """`atx match-sim` on an event file; returns stats.json (+ wall seconds). Raises when the engine is unavailable."""
    exe = atx_binary()
    if exe is None:
        raise RuntimeError("atx match-sim is not available")
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = out_dir / "rules.json"
    rules.write_text(json.dumps(RULES_BTC), encoding="utf-8")
    t0 = time.time()
    cmd = [str(exe), "match-sim", "--events", str(events), "--symbol", SYMBOL, "--out", str(out_dir), "--snapshot-every-ms", str(snapshot_ms),
           "--depth", str(depth), "--rules", str(rules), "--fee-class", "crypto"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"match-sim failed ({r.returncode}): {r.stderr[-2000:]}")
    st = json.loads((out_dir / "stats.json").read_text(encoding="utf-8")) if (out_dir / "stats.json").is_file() else {}
    st["wall_s"] = time.time() - t0
    return st


def engine_prints(out_dir: Path) -> dict[str, np.ndarray]:
    """trades.jsonl -> prints {ms, side (+1 taker buy), price, size} in engine order."""
    p = out_dir / "trades.jsonl"
    if not p.is_file() or p.stat().st_size == 0:
        return prints_dict(np.zeros((0, 4), np.int64))
    df = pl.read_ndjson(p, schema={"ts": pl.Int64, "price": pl.Float64, "size": pl.Float64, "side": pl.Utf8})
    return {"ms": df["ts"].to_numpy().astype(np.int64), "side": np.where(df["side"].to_numpy() == "Buy", 1, -1).astype(np.int8),
            "price": df["price"].to_numpy(), "size": df["size"].to_numpy()}


def engine_book_1s(out_dir: Path) -> pl.DataFrame:
    """book.jsonl (ob200 shape) -> micro_days book_1s via sim.book (the same reconstruction as the real archives)."""
    from ai_trader.sim.book import build_series

    df, _ = build_series(out_dir / "book.jsonl", bin_ms=1000)
    return df            # same columns as the micro_days BTC book_1s files (no symbol column there)


def engine_rejects(out_dir: Path) -> int:
    p = out_dir / "rejects.jsonl"
    if not p.is_file():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ----------------------------------------------------------------------------------------------------------------------
# a generated day: generate -> (engine) -> aggregate -> manifest
# ----------------------------------------------------------------------------------------------------------------------

SCRATCH = Path(os.environ.get("OBGEN_SCRATCH", r"C:\Users\Asus\AppData\Local\Temp\claude\c--Users-Asus-Desktop-AI-Trader\31faef80-e320-4ca6-ab63-04fa013eeb03\scratchpad\F5\work"))


def generate_day(key: str, tables: QRTables, cal: DayCal, path: FlowPath, out_dir: Path, d0: int, seed: int,
                 use_engine: bool | None = None, man: Manifest | None = None, log=print) -> dict[str, Any]:
    """Generate one path, run it through at-match when available, write the micro_days aggregates to out_dir and record
    the day in the manifest (section 'generated', key). Skips when the manifest says done and the files exist."""
    man = man or Manifest()
    if man.done("generated", key) and (out_dir / "book_1s" / f"{SYMBOL}.parquet").is_file():
        return man.get("generated", key)
    use_engine = (atx_binary() is not None) if use_engine is None else use_engine
    work = SCRATCH / key.replace(":", "_").replace("/", "_")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    ev = work / "events.jsonl" if use_engine else None
    t0 = time.time()
    g = Generator(tables, cal, seed=seed)
    out = g.run(path, events_path=ev)
    t_gen = time.time() - t0
    rec = {"status": "done", "date": cal.date, "seed": seed, "engine": "atx match-sim" if use_engine else "internal",
           "gen_s": t_gen, **{k: v for k, v in out.stats.items() if k != "rows100"}, "cal": asdict(cal)}
    if use_engine:
        st = run_engine(ev, work / "engine")
        rec["engine_stats"] = st
        rec["engine_rejects"] = engine_rejects(work / "engine")
        prints = engine_prints(work / "engine")
        t_end = path.t0_ms + path.n * path.bar_ms
        b1 = engine_book_1s(work / "engine")
        bms = b1["ts"].dt.epoch("ms")
        b1 = b1.filter((bms > path.t0_ms) & (bms <= t_end))          # drop the warm-up states
        internal = prints_dict(out.trades)
        rec["engine_vs_internal"] = {"n_prints_engine": int(len(prints["ms"])), "n_prints_internal": int(len(internal["ms"])),
                                     "qty_engine": float(prints["size"].sum()), "qty_internal": float(internal["size"].sum())}
        from ai_trader.research import micro_days as md

        od = md.merge_orders(prints)
        for sub in ("trades_1s", "book_1s", "orders"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        md.bars_1s(prints, od, d0).write_parquet(out_dir / "trades_1s" / f"{SYMBOL}.parquet", compression="zstd")
        pl.DataFrame(od).write_parquet(out_dir / "orders" / f"{SYMBOL}.parquet", compression="zstd")
        b1.write_parquet(out_dir / "book_1s" / f"{SYMBOL}.parquet", compression="zstd")
    else:
        rec["rows"] = aggregate(out_dir, prints_dict(out.trades), out.book_rows, d0)
    shutil.rmtree(work, ignore_errors=True)
    rec["total_s"] = time.time() - t0
    rec["bytes"] = du_bytes(out_dir)
    man.put("generated", key, rec)
    log(f"{key}: {rec['engine']} gen {t_gen:.0f} s total {rec['total_s']:.0f} s events {out.n_events} moves {out.stats['n_moves']}")
    return rec


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="ai_trader.sim.ob_gen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("dates", nargs="*", default=list(CAL_DAYS))
    c.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        os.environ.setdefault("POLARS_MAX_THREADS", "2")
        for dt in a.dates:
            calibrate_day(dt, keep_raw=a.keep_raw)


if __name__ == "__main__":
    main()
