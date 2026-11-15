"""Gate G1 leak-audit harness (SPEC section 6, automated leak tests 1 to 3). Written by the leak auditor.

Nothing in ``vecopen/src`` imports this module. It never edits ``src``: every instrument is applied from outside
(a class-level spy on ``LevelRegistry.update`` that returns the original result unchanged, timestamp-tracking
provenance attached to the bar stream, pickled state snapshots, and independent recomputation from the bars).

Design (the full argument is in ``vecopen/reports/phase_4_g1.md``)
------------------------------------------------------------------
The event pipeline is exactly the loop of ``phase3.event_step``: base 5m bars and the closed bars of the 8 primary
timeframes (phase 1 ``resample`` with the asset's session) merged by ``levels.bar_stream`` and fed one bar at a time
to four ``TouchDetector``s (treated + 3 placebo families) plus the ``PlaceboGenerator``. ``Pipeline`` replicates that
loop line for line (a test checks it against ``phase3.event_step``).

Per asset, ONE full run in the parent process. 200 timestamps t (seeded from ``config.seed``) are drawn among the base
bar closes; each t gets a warm-start point s = t - W (W log-uniform in [1, 30] days; s = None, i.e. a from-scratch
literal run, when that falls before the first bar, and for one seeded t per asset). During the full run the parent

* takes ``pickle`` snapshots of the complete pipeline state at every s, and its canonical digest;
* digests every bar fed and every output emitted inside each window (s, t] (and since the start, for the literal t);
* takes the canonical digest of the complete state at every t (registry, detectors, live levels, pending candidates,
  ATR/sigma accumulators, placebo generator and its RNG state) and audits that it holds no timestamp after t and no
  object outside the pipeline's own classes (the state is closed: no frame, iterator or file is reachable from it);
* audits every output (test 3) and recomputes its fields independently from the bars (rules 1 to 3).

A separate WORKER process (fresh interpreter, own globals) then, for each t:

* test 1: truncates the data at t, re-runs ``resample`` on the truncated frame, checks that every resampled frame
  equals the full-data frame on ``ts_close <= t`` (exact, all rows, all columns), restores the snapshot at s (checked
  to digest identically), replays the truncated stream over (s, t] and compares the window digest of bars + outputs
  and the complete state digest at t with the full run's;
* test 2: multiplies every price (and volume) after t by random factors on the data up to t + 1 day, re-resamples,
  checks the frames on ``ts_close <= t`` are unchanged, replays (s, t + 1 day] from the same snapshot, and compares
  the window digest and the state digest at t, and counts outputs stamped <= t that were emitted after t (must be 0).

Equivalence to the spec's literal test: the pipeline is a deterministic state machine S_k = F(S_{k-1}, x_k). The
truncated literal run differs from the full run only if (a) its input prefix differs, which is tested exactly for all
rows <= t, or (b) F reads something other than (S_{k-1}, x_k), which is excluded by the closed-state audit, the
fresh-process replay from pickled state and the static import/IO audit; any look-ahead inside F would also have to
surface in the state or outputs at t, which are compared in full. One literal from-scratch run per asset checks the
equivalence directly.
"""

from __future__ import annotations

import array
import hashlib
import math
import pickle
import time
import traceback
import zlib
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from vecopen.bars import load_sessions, resample
from vecopen.config import PROJECT_ROOT, load_config
from vecopen.data import load_bars
from vecopen.events import LevelInfo, TouchDetector, TouchEvent
from vecopen.legs import Bar
from vecopen.levels import FirstFill, LevelBorn, LevelExpired, LevelRegistry, PolarityFlip, bar_stream
from vecopen.placebo import FAMILIES, P1, P2, P3, PlaceboGenerator, PlaceboSpec, rng_for

BASE_MS = 5 * 60_000
DAY_MS = 86_400_000
TS_LO, TS_HI = 1_400_000_000_000, 2_200_000_000_000      # any int in this range is an epoch-ms timestamp
COLS = ("ts_close", "o", "h", "l", "c", "v")
TREATED = "treated"
EVENTS_DIR = PROJECT_ROOT / "data" / "research" / "events"
LABEL_COLUMNS = {"entry", "entry_ts", "bars_seen", "censored", "res", "res_side", "res_end_ts", "mae_atr",
                 "mae_end_ts", "horizon_end_ts"}
LABEL_PREFIXES = ("fade_", "follow_")
TS_FIELDS = frozenset({"ts", "t0", "birth_ts", "origin_ts_close", "origin_ts_open", "ts_close", "ts_open", "fill_ts",
                       "expire_ts"})


def peak_rss_mb() -> float:
    """Peak working set of this process in MB (Windows; 0 elsewhere)."""
    import sys

    if sys.platform != "win32":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2**20


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def iso(t: int | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t / 1000, UTC).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- spy on the registry (behaviour-preserving)

_SINK: list[Any] = []
_ORIG_UPDATE = LevelRegistry.update


def _spy_update(self: LevelRegistry, bar: Bar) -> list[Any]:
    out = _ORIG_UPDATE(self, bar)
    if out:
        _SINK.append(out)
    return out


class spy:
    """Context manager: capture every ``LevelRegistry.update`` result (births, fills, flips, expiries)."""

    def __enter__(self) -> spy:
        _SINK.clear()
        LevelRegistry.update = _spy_update  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        LevelRegistry.update = _ORIG_UPDATE  # type: ignore[method-assign]
        _SINK.clear()


def _drain() -> tuple[Any, ...]:
    out = tuple(e for x in _SINK for e in x)
    _SINK.clear()
    return out


# ---------------------------------------------------------------- the event pipeline (== phase3.event_step loop)

@dataclass
class StepOut:
    outs: list[Any]
    touches: list[tuple[TouchDetector, TouchEvent]]
    births: list[LevelInfo]                      # treated births at this bar
    placebos: list[tuple[PlaceboSpec, LevelInfo]]
    regev: list[tuple[TouchDetector, tuple[Any, ...]]]


class Pipeline:
    """The four touch detectors and the placebo generator of one asset, fed exactly as ``phase3.event_step``."""

    def __init__(self, cfg: Any, session: Any, asset: str) -> None:
        tfs = list(cfg.data["levels"]["timeframes_primary"])
        self.treated = TouchDetector(LevelRegistry(cfg, timeframes=tfs, session=session), cfg, session=session)
        self.fams = {f: TouchDetector(LevelRegistry(cfg, timeframes=tfs, session=session, family=f), cfg,
                                      session=session) for f in FAMILIES}
        self.gen = PlaceboGenerator(cfg, self.treated.registry.timeframes, rng=rng_for(asset, cfg))
        self.base_tf = self.treated.base_tf
        self.tds = (self.treated, *self.fams.values())

    def step(self, b: Bar) -> StepOut:
        so = StepOut([], [], [], [], [])
        for td in self.tds:
            ev = td.update(b)
            reg = _drain()
            so.outs.append((td.family, reg, tuple(ev), tuple(td.expired)))
            so.touches += [(td, e) for e in ev]
            so.regev.append((td, reg))
        so.births = list(self.treated.new_levels)
        so.outs.append(("treated_new", tuple(so.births)))
        if b.tf != self.base_tf:
            for sp in self.gen.update(b, self.treated.new_levels):
                x = self.fams[sp.family].register(sp.leg, source_level_id=sp.source_level_id, variant=sp.variant,
                                                  displacement_atr=sp.displacement_atr)
                so.placebos.append((sp, x))
        so.outs.append(("placebo", tuple(so.placebos)))
        return so


# ---------------------------------------------------------------- canonical state digest + closure audit

class ClosureViolation(TypeError):
    """The pipeline state reaches an object that is not plain data or one of vecopen's own classes."""


_SLOTS: dict[type, tuple[str, ...]] = {}


def _slots(tx: type) -> tuple[str, ...]:
    r = _SLOTS.get(tx)
    if r is None:
        r = tuple(n for cls in tx.__mro__ for n in cls.__dict__.get("__slots__", ()) if n not in ("__dict__", "__weakref__"))
        _SLOTS[tx] = r
    return r


def canon(root: Any) -> tuple[str, int, int]:
    """(digest, max timestamp found, number of objects) of the object graph under ``root``.

    Exact: floats by ``float.hex``, arrays by their bytes, dict insertion order, list/heap order, aliasing (shared
    references) by back-reference numbers. Raises ``ClosureViolation`` on any foreign type (a polars frame, an
    iterator, a file, a path...): the state must be closed."""
    toks: list[str] = []
    memo: dict[int, int] = {}
    keep: list[Any] = []
    mx = [-1]
    ap = toks.append

    def v(y: Any) -> None:                 # one value: primitives inline, everything else through w
        ty = type(y)
        if ty is float:
            ap(y.hex())
        elif ty is int:
            ap("i%d" % y)
            if TS_LO <= y <= TS_HI and y > mx[0]:
                mx[0] = y
        elif y is None:
            ap("N")
        elif ty is str:
            ap(repr(y))
        else:
            w(y)

    def w(x: Any) -> None:
        tx = type(x)
        if tx is float or tx is int or x is None or tx is str:
            v(x)
            return
        if tx is bool:
            ap("T" if x else "F")
            return
        if isinstance(x, (float, np.floating)):
            ap("f" + tx.__name__ + float(x).hex())
            return
        if isinstance(x, (int, np.integer)):
            ap("j" + tx.__name__ + str(int(x)))
            return
        i = id(x)
        m = memo.get(i)
        if m is not None:
            ap("@%d" % m)
            return
        memo[i] = len(memo)
        keep.append(x)
        if tx is tuple or tx is list:
            ap(("t%d" if tx is tuple else "l%d") % len(x))
            for y in x:
                v(y)
            return
        if tx is dict:
            ap("d%d" % len(x))
            for k, y in x.items():
                v(k)
                v(y)
            return
        if tx is deque:
            ap("q%d/%s" % (len(x), x.maxlen))
            for y in x:
                v(y)
            return
        if tx is set or tx is frozenset:
            ap("S%d" % len(x))
            for y in sorted(x, key=repr):
                v(y)
            return
        if tx is array.array:
            ap("a%s%d:%s" % (x.typecode, len(x), hashlib.blake2b(x.tobytes(), digest_size=20).hexdigest()))
            return
        if isinstance(x, np.random.Generator):
            ap("rng" + repr(x.bit_generator.state))
            return
        if tx is ZoneInfo:
            ap("z" + x.key)
            return
        mod = tx.__module__ or ""
        if not (mod.startswith("vecopen.") or mod in (__name__, "leakkit")):
            raise ClosureViolation(f"state reaches a {mod}.{tx.__qualname__} object")
        ap("C" + tx.__qualname__)
        if isinstance(x, tuple):                      # NamedTuple (Bar)
            ap("%d" % len(x))
            for y in x:
                v(y)
            return
        d = getattr(x, "__dict__", None)
        if d is not None:
            ap("d%d" % len(d))
            for k, y in d.items():
                ap(k)
                v(y)
        for sname in _slots(tx):
            ap(sname)
            try:
                y = getattr(x, sname)
            except AttributeError:
                ap("<unset>")
                continue
            v(y)

    w(root)
    h = hashlib.blake2b("".join(toks).encode("utf-8"), digest_size=20).hexdigest()
    return h, mx[0], len(memo)


def max_ts(x: Any, depth: int = 0) -> int:
    """Largest timestamp-named int field anywhere inside an output object (-1 if none)."""
    best = -1
    if isinstance(x, tuple) and not hasattr(x, "__dataclass_fields__") and not hasattr(x, "_fields"):
        for y in x:
            best = max(best, max_ts(y, depth + 1))
        return best
    names = getattr(x, "__dataclass_fields__", None) or (dict.fromkeys(x._fields) if hasattr(x, "_fields") else None)
    if not names:
        return best
    for n in names:
        v = getattr(x, n, None)
        if type(v) is int:
            if n in TS_FIELDS:
                best = max(best, v)
            elif TS_LO <= v <= TS_HI:
                raise AssertionError(f"{type(x).__name__}.{n}={v} looks like a timestamp outside TS_FIELDS")
        elif v is not None and (hasattr(v, "__dataclass_fields__") or isinstance(v, tuple)) and depth < 4:
            best = max(best, max_ts(v, depth + 1))
    return best


def stamp(o: Any) -> int:
    for a in ("t0", "ts", "birth_ts"):
        v = getattr(o, a, None)
        if isinstance(v, int):
            return v
    if isinstance(o, PlaceboSpec):
        return o.leg.birth_ts
    raise TypeError(f"no stamp on {type(o).__name__}")


def iter_outputs(so: StepOut) -> Iterator[tuple[str, Any]]:
    """(family, output) for every output of one bar."""
    for fam, reg, ev, exp in so.outs[:4]:
        for o in (*reg, *ev, *exp):
            yield fam, o
    for x in so.births:
        yield TREATED, x
    for sp, x in so.placebos:
        yield sp.family, sp
        yield sp.family, x


# ---------------------------------------------------------------- inputs: resample variants and frame digests

def resample_clean(df: pl.DataFrame, tf: str, session: Any) -> pl.DataFrame:
    return resample(df, tf, session)


def resample_open_keyed(df: pl.DataFrame, tf: str, session: Any) -> pl.DataFrame:
    """NON-VACUITY ONLY: the forbidden join on open time (a bar becomes visible at its first base close)."""
    r = resample(df, tf, session, complete_only=False)
    return r.with_columns(ts_close=pl.col("ts_open") + pl.duration(minutes=5))


RESAMPLERS: dict[str, Callable[..., pl.DataFrame]] = {"clean": resample_clean, "open_keyed": resample_open_keyed,
                                                      "peek": resample_clean}


def peek_stream(it: Iterator[Bar]) -> Iterator[Bar]:
    """NON-VACUITY ONLY: a one-bar look-ahead (each base bar's high includes the next base bar's high)."""
    pending: Bar | None = None
    held: list[Bar] = []
    for b in it:
        if b.tf == "5m":
            if pending is not None:
                yield from held_flush(pending, b, held)
                held = []
            pending = b
        elif pending is not None:
            held.append(b)
        else:
            yield b
    if pending is not None:
        yield pending
        yield from held


def held_flush(prev: Bar, nxt: Bar, held: list[Bar]) -> Iterator[Bar]:
    yield prev._replace(h=max(prev.h, nxt.h))
    yield from held


def make_stream(base: pl.DataFrame, higher: dict[str, pl.DataFrame], variant: str) -> Iterator[Bar]:
    it = bar_stream(base, higher, "5m")
    return peek_stream(it) if variant == "peek" else it


class FrameSet:
    """Column arrays of the base frame and every resampled frame, for exact digests of the rows ``ts_close <= t``."""

    def __init__(self, base: pl.DataFrame, higher: dict[str, pl.DataFrame]) -> None:
        self.frames: list[tuple[str, np.ndarray, list[tuple[str, np.ndarray]]]] = []
        for name, fr in (("5m", base), *higher.items()):
            cols = []
            for c in fr.columns:
                x = fr[c]
                a = x.dt.epoch("ms").to_numpy() if isinstance(x.dtype, pl.Datetime) else x.to_numpy()
                cols.append((c, np.ascontiguousarray(a)))
            ts = dict(cols)["ts_close"]
            if len(ts) > 1 and not (np.diff(ts) > 0).all():
                raise AssertionError(f"{name} frame is not strictly sorted by ts_close")
            self.frames.append((name, ts, cols))

    def digests(self, t: int) -> dict[str, str]:
        out = {}
        for name, ts, cols in self.frames:
            k = int(np.searchsorted(ts, t, side="right"))
            h = hashlib.blake2b(digest_size=20)
            h.update(f"{k}|{[c for c, _ in cols]}".encode())
            for c, a in cols:
                h.update(c.encode())
                h.update(a[:k].tobytes())
            out[name] = h.hexdigest()
        return out


def prefix_digests(base: pl.DataFrame, higher: dict[str, pl.DataFrame], t: int) -> dict[str, str]:
    return FrameSet(base, higher).digests(t)


def after(fr: pl.DataFrame, s: int | None) -> pl.DataFrame:
    return fr if s is None else fr.filter(pl.col("ts_close").dt.epoch("ms") > s)


def upto(fr: pl.DataFrame, t: int) -> pl.DataFrame:
    return fr.filter(pl.col("ts_close").dt.epoch("ms") <= t)


def perturb(df: pl.DataFrame, t: int, fp: float, fv: float) -> pl.DataFrame:
    late = pl.col("ts_close").dt.epoch("ms") > t
    return df.with_columns([pl.when(late).then(pl.col(c) * fp).otherwise(pl.col(c)).alias(c) for c in "ohlc"]
                           + [pl.when(late).then(pl.col("v") * fv).otherwise(pl.col("v")).alias("v")])


# ---------------------------------------------------------------- provenance of every resampled bar (test 3)

def provenance(df: pl.DataFrame, higher: dict[str, pl.DataFrame], session: Any, variant: str
               ) -> tuple[dict[str, dict[int, int]], dict[str, Any]]:
    """Source timestamps of every higher-timeframe bar, measured by running the SAME resampler on a frame whose
    prices are the base bars' own close timestamps: then h = max and l = min source timestamp of the bucket."""
    fake = df.select("ts_close").with_columns(src=pl.col("ts_close").dt.epoch("ms").cast(pl.Float64))
    fake = fake.select("ts_close", o=pl.col("src"), h=pl.col("src"), l=pl.col("src"), c=pl.col("src"),
                       v=pl.lit(1.0))
    rs = RESAMPLERS[variant]
    prov: dict[str, dict[int, int]] = {}
    rep: dict[str, Any] = {}
    for tf, real in higher.items():
        p = rs(fake, tf, session)
        key = p["ts_close"].dt.epoch("ms").to_numpy()
        opn = p["ts_open"].dt.epoch("ms").to_numpy()
        mx, mn = p["h"].to_numpy().astype(np.int64), p["l"].to_numpy().astype(np.int64)
        same_keys = (real.height == p.height and np.array_equal(key, real["ts_close"].dt.epoch("ms").to_numpy()))
        late = int((mx > key).sum())
        early = int((mn < opn + BASE_MS).sum())
        rep[tf] = {"bars": int(p.height), "keys_match_real_frame": bool(same_keys), "max_src_after_close": late,
                   "min_src_before_open": early, "count_matches_n_base": bool(np.array_equal(
                       p["v"].to_numpy().astype(np.int64), p["n_base"].to_numpy()))}
        if late:
            i = int(np.flatnonzero(mx > key)[0])
            rep[tf]["example"] = {"key": iso(int(key[i])), "max_src": iso(int(mx[i]))}
        prov[tf] = dict(zip(key.tolist(), mx.tolist()))
    return prov, rep


# ---------------------------------------------------------------- independent recomputation (rules 1 to 3)

class TfArrays:
    """Trade bars of one timeframe and the shifted statistics, recomputed from the bars by the auditor."""

    def __init__(self, fr: pl.DataFrame, tf: str, legs: dict[str, Any], a_max: int) -> None:
        ts = fr["ts_close"].dt.epoch("ms").to_list()
        to = fr["ts_open"].dt.epoch("ms").to_list()
        o, h, l, c, v = (fr[x].to_list() for x in "ohlcv")  # noqa: E741
        keep = [i for i in range(len(ts)) if not (h[i] == l[i] and (v[i] is None or v[i] == 0))]
        self.ts = array.array("q", (ts[i] for i in keep))
        self.to = array.array("q", (to[i] for i in keep))
        self.o, self.h, self.l, self.c, self.v = (array.array("d", (a[i] for i in keep)) for a in (o, h, l, c, v))
        self.n_no_trade = len(ts) - len(keep)
        del ts, to, o, h, l, c, v
        n = int(legs["atr_len"])
        N = len(self.ts)
        nan = float("nan")
        self.atr_after = array.array("d", [nan]) * N          # ATR through bar i (nan during warm-up)
        self.mtr_before = array.array("d", [nan]) * N         # mean TR of up to a_max trade bars through i-1
        self.mtr_err = 0.0                                    # running sum vs exact fsum, max relative error
        s = 0.0
        cnt = 0
        val: float | None = None
        pc: float | None = None
        trs: deque[float] = deque(maxlen=max(a_max, n))
        tr_sum = 0.0
        for i in range(N):
            hh, ll, cc = self.h[i], self.l[i], self.c[i]
            tr = hh - ll if pc is None else max(hh - ll, abs(hh - pc), abs(ll - pc))
            m = len(trs)
            if m >= n:
                self.mtr_before[i] = tr_sum / m
                ex = math.fsum(trs) / m
                if ex:
                    self.mtr_err = max(self.mtr_err, abs(self.mtr_before[i] - ex) / ex)
            if m == trs.maxlen:
                tr_sum -= trs[0]
            trs.append(tr)
            tr_sum += tr
            pc = cc
            if cnt < n:
                s += tr
                cnt += 1
                if cnt == n:
                    val = s / n
            else:
                val = (val * (n - 1) + tr) / n  # type: ignore[operator]
            self.atr_after[i] = nan if val is None else val
        # volume baseline: mean of the previous N sessions' volume in the same intraday slot (shifted)
        from vecopen.bars import INTRADAY_MINUTES

        tf_min = INTRADAY_MINUTES.get(tf) if tf not in ("1d", "1w", "1M") else None
        vn = int(legs["volume_baseline_sessions"])
        dqs: dict[int, deque[float]] = {}
        self.vbase = array.array("d", [nan]) * N
        for i in range(N):
            slot = 0 if tf_min is None else ((self.to[i] // 60_000) % 1440) // tf_min
            dq = dqs.setdefault(slot, deque(maxlen=vn))
            self.vbase[i] = sum(dq) / vn if len(dq) == vn else nan
            dq.append(float(self.v[i]))

    def atr(self, i: int) -> float | None:
        """ATR through trade bar i (None during warm-up or for i < 0)."""
        if i < 0:
            return None
        x = self.atr_after[i]
        return None if x != x else x

    def regime(self, c: int) -> float | None:
        a = self.atr(c - 1)
        m = self.mtr_before[c]
        return math.log(a / m) if (a is not None and a > 0 and m == m and m) else None


class BaseArrays:
    def __init__(self, df: pl.DataFrame) -> None:
        self.ts = array.array("q", df["ts_close"].dt.epoch("ms").to_list())
        self.o, self.h, self.l, self.c = (array.array("d", df[x].to_list()) for x in "ohlc")
        self.t_first = self.ts[0]
        self.contiguous = all(self.ts[k] - self.ts[k - 1] == BASE_MS for k in range(1, len(self.ts)))
        self.idx = None if self.contiguous else {t: i for i, t in enumerate(self.ts)}
        cum = array.array("d", [0.0])
        for k in range(1, len(self.c)):
            d = self.c[k] - self.c[k - 1]
            cum.append(cum[-1] + d * d)
        self.cum = cum

    def index(self, t: int) -> int:
        return (t - self.t_first) // BASE_MS if self.idx is None else self.idx[t]

    def sigma_before(self, j: int, w: int) -> float | None:
        """RMS of the close changes of base bars 0..j-1 over the last w changes (bar j itself excluded)."""
        n = j - 1
        if w < 1 or n < w:
            return None
        return math.sqrt(max(self.cum[n] - self.cum[n - w], 0.0) / w)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Audit:
    """Test 3 (timestamp audit of every output) plus the independent rule checks, applied during the full run."""

    def __init__(self, asset: str, cfg: Any, session: Any, df: pl.DataFrame, higher: dict[str, pl.DataFrame],
                 prov: dict[str, dict[int, int]], *, independent: bool = True) -> None:
        d = cfg.data
        self.legs, self.lv = d["legs"], d["levels"]
        self.kappa = float(self.lv["kappa_atr"])
        self.theta = float(self.legs["theta_L_atr"])
        self.omega_max = float(self.legs["omega_max"])
        self.prov = prov
        self.base = BaseArrays(df)
        self.independent = independent
        self.tfa = {tf: TfArrays(fr, tf, self.legs, int(self.lv["a_max_bars"])) for tf, fr in higher.items()} \
            if independent else {}
        lo, hi = (float(x) for x in d["stats"]["placebo_displacement_atr"])
        self.u_lo, self.u_hi = lo, hi
        self.rng_ref = np.random.default_rng([int(d["seed"]), zlib.crc32(asset.encode("utf-8"))])   # D-P2 formula
        self.fail: Counter[str] = Counter()
        self.ok: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}
        self.hwm = -1
        self.last_key: tuple[int, int] | None = None
        self.ptr = {tf: 0 for tf in higher}
        self.fed = {tf: 0 for tf in higher}
        self.tf_ts = {tf: array.array("q", fr["ts_close"].dt.epoch("ms").to_list()) for tf, fr in higher.items()}
        self.live: dict[tuple[str, int], LevelInfo] = {}
        self.zones: dict[tuple[str, int], tuple[float, float, float, float, int]] = {}
        self.treated_born: dict[int, LevelInfo] = {}
        self.cov_rows: list[tuple[str, str, int, float, float, float | None]] = []
        self.n_treated_events = 0
        self.n_treated_births = 0
        self.nonvac: Counter[str] = Counter()
        self.ok_placebo_any: set[str] = set()

    # -- bookkeeping
    def bad(self, check: str, msg: str) -> None:
        self.fail[check] += 1
        ex = self.examples.setdefault(check, [])
        if len(ex) < 3:
            ex.append(msg)

    def chk(self, check: str, cond: bool, msg: str = "", fam: str | None = None) -> None:
        """Successes are counted only for treated (or family-free) checks: a count of passing checks over placebo
        events would be a placebo touch count, an H2 outcome (phase 3 D-P6). Failures are counted for every family."""
        if cond:
            if fam is None or fam == TREATED:
                self.ok[check] += 1
            else:
                self.ok_placebo_any.add(check)
        else:
            self.bad(check, msg)

    # -- per bar
    def bar(self, b: Bar, so: StepOut) -> None:
        t = b.ts_close
        is_base = b.tf == "5m"
        self.treated_born = {}                      # family 2/3 sources are born at the same bar
        key = (t, 1 if is_base else 0)
        if self.last_key is not None and key < self.last_key:
            self.bad("r1_stream_order", f"{b.tf} bar at {iso(t)} after {self.last_key}")
        self.last_key = key
        if is_base:
            src = t
        else:
            src = self.prov[b.tf].get(t, TS_HI + 1)
            self.fed[b.tf] += 1
        self.chk("t3_bar_provenance", src <= t, f"{b.tf} bar keyed {iso(t)} carries data up to {iso(src)}")
        self.hwm = max(self.hwm, src)
        if is_base:          # rule 1: the as-of join on close_time <= t, for every timeframe
            for tf, lst in self.tf_ts.items():
                p = self.ptr[tf]
                while p < len(lst) and lst[p] <= t:
                    p += 1
                self.ptr[tf] = p
                if self.fed[tf] != p:
                    self.bad("r1_asof_close_join", f"at {iso(t)}: {self.fed[tf]} {tf} bars fed, {p} closed")
            self.ok["r1_asof_close_join"] += 1
        for fam, o in iter_outputs(so):
            st = stamp(o)
            self.chk("t3_output_stamp_is_now", st == t, f"{type(o).__name__} stamped {iso(st)} emitted at {iso(t)}", fam)
            m = max_ts(o)
            self.chk("t3_output_fields_le_stamp", m <= st, f"{type(o).__name__} at {iso(st)} holds {iso(m)}", fam)
            self.chk("t3_hwm_le_stamp", self.hwm <= st, f"{type(o).__name__} at {iso(st)}: inputs up to {iso(self.hwm)}",
                     fam)
        for x in so.births:
            self.birth(x, b, TREATED)
        for sp, x in so.placebos:
            self.birth(x, b, sp.family, sp)
        for td, e in so.touches:
            self.touch(td, e, b)
        for td, reg in so.regev:
            for e in reg:
                if isinstance(e, FirstFill):
                    for oid in e.overlapping:
                        ol = td.registry.active.get(oid)
                        self.chk("t3_confluence_inputs_le_t0", ol is not None and ol.birth_ts <= e.t0,
                                 f"{td.family} fill {e.level_id} at {iso(e.t0)} counts level {oid}", td.family)
                elif isinstance(e, LevelExpired):
                    self.live.pop((td.family, e.level_id), None)
                    self.zones.pop((td.family, e.level_id), None)
                elif isinstance(e, (LevelBorn, PolarityFlip)):
                    pass

    # -- births (rules 2 and 3)
    def birth(self, x: LevelInfo, b: Bar, fam: str, sp: PlaceboSpec | None = None) -> None:
        def ck(c: str, cond: bool, msg: str = "") -> None:
            self.chk(c, cond, msg, fam)

        k = (fam, x.level_id)
        self.live[k] = x
        self.zones[k] = (x.price, x.zone_lo, x.zone_hi, x.atr, x.birth_ts)
        ck("r3_birth_at_bar_close", x.birth_ts == b.ts_close and x.tf == b.tf,
                 f"{fam} {x.level_id} born {iso(x.birth_ts)} on a {b.tf} bar at {iso(b.ts_close)}")
        half = self.kappa * x.atr
        ck("r3_zone_from_birth_atr", x.zone_lo == x.price - half and x.zone_hi == x.price + half,
                 f"{fam} {x.level_id} zone {x.zone_lo}..{x.zone_hi} vs price {x.price} atr {x.atr}")
        if fam == TREATED:
            self.n_treated_births += 1
            self.treated_born[x.level_id] = x
        if not self.independent:
            return
        leg = x.leg
        assert leg is not None
        a = self.tfa[x.tf]
        c, kk = leg.origin_index, leg.k_star
        conf = c + kk
        tag = f"{fam} {x.tf} level {x.level_id} born {iso(x.birth_ts)}"
        if conf >= len(a.ts) or c < 1:
            self.bad("r3_indices", tag)
            return
        ck("r3_born_at_confirmation_close", a.ts[conf] == x.birth_ts and leg.confirm_index == conf, tag)
        o_ok = a.o[c] == leg.origin_o if fam in (TREATED, P1) else leg.origin_o == x.price
        ck("r3_origin_candle", a.ts[c] == leg.origin_ts_close and o_ok and x.price == leg.origin_o
                 and a.h[c] == leg.origin_h and a.l[c] == leg.origin_l and a.c[c] == leg.origin_c, tag)
        ck("r2_atr_shifted", a.atr(c - 1) == x.atr, f"{tag}: atr {x.atr} vs ATR_(c-1) {a.atr(c - 1)}")
        if fam == TREATED:
            self.nonvac["treated_atr_checked"] += 1
            self.nonvac["treated_atr_equal_to_unshifted_atr"] += a.atr(c) == x.atr
            self.nonvac["treated_sigma_birth_equal_to_unshifted"] += x.sigma_birth is not None and                 x.sigma_birth == self.base.sigma_before(self.base.index(x.birth_ts) + 1, x.h_base)
        ck("r2_regime_shifted", x.regime == a.regime(c), f"{tag}: regime {x.regime} vs {a.regime(c)}")
        jb = self.base.index(x.birth_ts)
        ck("r2_sigma_birth_shifted", x.sigma_birth == self.base.sigma_before(jb, x.h_base),
                 f"{tag}: sigma_birth {x.sigma_birth} vs {self.base.sigma_before(jb, x.h_base)}")
        ck("r3_ref_close_is_confirmation_close", x.ref_close == a.c[conf] == leg.confirm_close, tag)
        g = _sign(x.ref_close - x.price) or (1 if leg.direction >= 0 else -1)
        dist = abs(x.ref_close - x.price)
        ck("r3_side_departure_dedge", x.g == g and x.departure_atr == dist / x.atr
                 and x.d_edge == max(dist - half, 0.0), tag)
        s = leg.direction
        o0 = leg.origin_o
        thr = self.theta * x.atr
        if fam in (TREATED, P1):
            ok_thr = s * (a.c[conf] - o0) >= thr
            ok_back = all(s * ((a.h[c + j] if s < 0 else a.l[c + j]) - o0) > 0 for j in range(1, kk + 1))
            ok_min = all(s * (a.c[c + j] - o0) < thr for j in range(0, kk))
            ck("r3_leg_rule_threshold_tradeback_first_k", ok_thr and ok_back and ok_min and kk <= int(
                self.legs["n_confirm"]), f"{tag}: thr {ok_thr} back {ok_back} first {ok_min} k {kk}")
        if fam == TREATED:
            rng_ = a.h[c] - a.l[c]
            om = (a.h[c] - o0) / rng_ if s < 0 else (o0 - a.l[c]) / rng_
            ck("r3_treated_candle_rule", s == _sign(a.c[c] - a.o[c]) and om == leg.omega
                     and om <= self.omega_max and leg.leg_size_atr == s * (a.c[conf] - o0) / x.atr, tag)
            vs = [a.v[j] for j in range(c, conf + 1)]
            bs = [a.vbase[j] for j in range(c, conf + 1)]
            if any(q != q for q in bs):
                exp_lv = None
            else:
                v_sum = 0.0
                b_sum = 0.0
                for q, r in zip(vs, bs):
                    v_sum += q
                    b_sum += r
                exp_lv = v_sum / b_sum if b_sum else None
            ck("r2_leg_vol_baseline_shifted", x.leg_vol == exp_lv, f"{tag}: leg_vol {x.leg_vol} vs {exp_lv}")
        elif fam == P2:
            src = self.treated_born.get(x.source_level_id)
            u = float(self.rng_ref.uniform(self.u_lo, self.u_hi))
            sg = 1 if x.variant == "up" else -1
            ck("rng_p2_draw_is_next_fixed_draw", x.displacement_atr == sg * u, f"{tag}: {x.displacement_atr} vs {sg * u}")
            ck("r3_p2_born_with_source", src is not None and src.birth_ts == x.birth_ts and src.atr == x.atr
                     and x.price == src.price + sg * u * src.atr, tag)
        elif fam == P3:
            src = self.treated_born.get(x.source_level_id)
            want = {"close": a.c[c], "mid": (a.h[c] + a.l[c]) / 2,
                    "extreme": a.l[c] if (src is not None and src.leg_direction < 0) else a.h[c]}
            ck("r3_p3_candle_prices", src is not None and src.birth_ts == x.birth_ts
                     and x.price == want.get(x.variant or "", float("nan")), tag)

    # -- touch events (rules 1 to 3, test 3, matching covariates)
    def touch(self, td: TouchDetector, e: TouchEvent, b: Bar) -> None:
        fam = td.family

        def ck(c: str, cond: bool, msg: str = "") -> None:
            self.chk(c, cond, msg, fam)

        fam = td.family
        k = (fam, e.level_id)
        info = self.live.pop(k, None)
        tag = f"{fam} {e.tf} level {e.level_id} t0 {iso(e.t0)}"
        if fam == TREATED:
            self.n_treated_events += 1
        ck("t3_event_at_base_close", b.tf == "5m" and e.t0 == b.ts_close, tag)
        ck("t3_event_level_is_birth_record", info is not None and repr(replace(info, leg=None)) == repr(e.level),
                 tag)
        ck("r3_no_fill_on_birth_bar", e.level.birth_ts < e.t0, tag)
        if fam in (TREATED, P1):
            self.cov_rows.append((fam, f"{fam}:{e.level_id}", datetime.fromtimestamp(e.t0 / 1000, UTC).year,
                                  math.log1p(e.age), e.level.departure_atr, e.level.regime))
        if not self.independent or info is None:
            return
        a = self.tfa[e.tf]
        j = self.base.index(e.t0)
        B = self.base
        n_after_birth = _bisect_right(a.ts, e.t0) - _bisect_right(a.ts, e.level.birth_ts)
        ck("cov_age_counts_tau_bars_in_birth_t0", e.age == n_after_birth, f"{tag}: age {e.age} vs {n_after_birth}")
        ip = _bisect_right(a.ts, e.t0) - 1
        last = a.c[ip]
        pol = "resistance" if last < e.level.zone_lo else "support" if last > e.level.zone_hi else \
            ("resistance" if info.leg_direction < 0 else "support")
        ck("r1_polarity_from_last_tau_close_le_t0", e.polarity == pol, f"{tag}: {e.polarity} vs {pol}")
        ck("t3_touch_bar_is_bar_at_t0", (e.touch_o, e.touch_h, e.touch_l, e.touch_c) ==
                 (B.o[j], B.h[j], B.l[j], B.c[j]) and B.l[j] <= e.level.zone_hi and B.h[j] >= e.level.zone_lo, tag)
        pc = B.c[j - 1] if j >= 1 else None
        ck("t3_prev_close_is_bar_before_t0", e.prev_close == pc, tag)
        ck("r2_sigma_t0_excludes_touch_bar", e.sigma_t0 == B.sigma_before(j, e.level.h_base),
                 f"{tag}: {e.sigma_t0} vs {B.sigma_before(j, e.level.h_base)}")
        if fam == TREATED:
            self.nonvac["treated_sigma_t0_checked"] += 1
            self.nonvac["treated_sigma_t0_equal_to_unshifted"] += e.sigma_t0 is not None and                 e.sigma_t0 == B.sigma_before(j + 1, e.level.h_base)
        # explicit source timestamps of the event's inputs
        conf_bar = a.ts[e.level.origin_index + e.level.k_star]
        srcs = [e.t0, B.ts[j - 1] if j >= 1 else -1, e.level.birth_ts, conf_bar,
                self.prov[e.tf].get(conf_bar, TS_HI + 1), a.ts[ip], self.prov[e.tf].get(a.ts[ip], TS_HI + 1)]
        ck("t3_max_source_ts_le_t0", max(srcs) <= e.t0, f"{tag}: max source {iso(max(srcs))}")

    # -- at a sample instant t (state level checks)
    def at_t(self, pipe: Pipeline, t: int) -> None:
        for td in pipe.tds:
            for lid, lvl in td.registry.active.items():
                z = self.zones.get((td.family, lid))
                self.chk("r3_zone_frozen_since_birth", z is not None and z == (lvl.price, lvl.zone_lo, lvl.zone_hi,
                                                                              lvl.atr, lvl.birth_ts),
                         f"{td.family} level {lid} at {iso(t)}", td.family)
        if not self.independent:
            return
        self.chk("rng_state_is_fixed_sequence_position",
                 pipe.gen.rng.bit_generator.state == self.rng_ref.bit_generator.state, f"at {iso(t)}")

    def summary(self) -> dict[str, Any]:
        checks = sorted(set(self.ok) | set(self.fail) | self.ok_placebo_any)
        # ok = successes on treated outputs (or family-free bar checks); placebo successes are never counted (H2 blind)
        return {"checks": {c: {"ok": self.ok[c], "fail": self.fail[c], "placebo_checked": c in self.ok_placebo_any}
                           for c in checks},
                "examples": self.examples,
                "treated_events": self.n_treated_events, "treated_births": self.n_treated_births,
                "nonvacuity": dict(self.nonvac),
                "regime_running_sum_max_rel_err": max((x.mtr_err for x in self.tfa.values()), default=0.0)}


def _bisect_right(a: list[int], x: int) -> int:
    import bisect

    return bisect.bisect_right(a, x)


# ---------------------------------------------------------------- sampling plan (seeded from config)

RESTORE_CHECK_EVERY = 10          # snapshot-restore exactness is digested for every 10th timestamp (diagnostic)


@dataclass
class Point:
    i: int
    t: int
    s: int | None
    literal: bool
    fp: float
    fv: float
    t_end: int


def plan(asset: str, cfg: Any, base_ts: list[int], n: int, variant: str = "clean") -> list[Point]:
    rng = np.random.default_rng([int(cfg.data["seed"]), zlib.crc32(asset.encode("utf-8")),
                                 zlib.crc32(b"vecopen-g1-leak-" + variant.encode())])
    n = min(n, len(base_ts) - 1)
    idx = np.sort(rng.choice(np.arange(1, len(base_ts)), size=n, replace=False))
    lit = int(rng.integers(n))
    out = []
    for k, i in enumerate(idx.tolist()):
        t = base_ts[i]
        w = int(math.exp(rng.uniform(math.log(DAY_MS), math.log(30 * DAY_MS)))) // BASE_MS * BASE_MS
        s: int | None = t - w
        if s < base_ts[0] or k == lit:
            s = None
        fp = fv = 1.0
        while abs(math.log(fp)) < 0.05:
            fp = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        while abs(math.log(fv)) < 0.05:
            fv = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        out.append(Point(k, t, s, k == lit, fp, fv, t + DAY_MS))
    return out


# ---------------------------------------------------------------- the worker (fresh process)

def point_request(p: Point, snap: bytes | None, ds: str | None, *, df: pl.DataFrame, cfg: Any, session: Any,
                  asset: str, tfs: list[str], variant: str) -> dict[str, Any]:
    """Tests 1 and 2 at one timestamp, in the worker.

    Test 1: truncate at t, resample, digest the frames (rows <= t), restore the snapshot at s (or start fresh),
    replay (s, t], digest bars and bars + outputs, digest the complete state at t.
    Test 2: perturb every price and volume after t (data up to t + 1 day), resample, digest the frames (rows <= t),
    digest the bars <= t of the perturbed stream (they must equal test 1's bar for bar, so the pipeline's outputs
    and state up to t are those of test 1), then continue the SAME pipeline with the perturbed bars after t and count
    outputs stamped <= t emitted after t."""
    rs = RESAMPLERS[variant]
    T = time.perf_counter()
    d = upto(df, p.t)
    hi = {tf: rs(d, tf, session) for tf in tfs}
    pre1 = FrameSet(d, hi).digests(p.t)
    if snap is None:
        pipe = Pipeline(cfg, session, asset)
        restore = None
    else:
        pipe = pickle.loads(snap)
        restore = None if ds is None else canon(pipe)[0] == ds
    hw, hb = hashlib.blake2b(digest_size=20), hashlib.blake2b(digest_size=20)
    n1 = 0
    for b in make_stream(after(d, p.s), {tf: after(f, p.s) for tf, f in hi.items()}, variant):
        so = pipe.step(b)
        rb = repr(b)
        hb.update(rb.encode())
        hw.update((rb + repr(so.outs)).encode())
        n1 += 1
    st, st_max, _ = canon(pipe)
    del d, hi
    T1 = time.perf_counter()
    dp = perturb(upto(df, p.t_end), p.t, p.fp, p.fv)
    hp = {tf: rs(dp, tf, session) for tf in tfs}
    pre2 = FrameSet(dp, hp).digests(p.t)
    hb2 = hashlib.blake2b(digest_size=20)
    n_le = n_gt = deferred = 0
    for b in make_stream(after(dp, p.s), {tf: after(f, p.s) for tf, f in hp.items()}, variant):
        if b.ts_close <= p.t:
            hb2.update(repr(b).encode())
            n_le += 1
        else:
            so = pipe.step(b)
            n_gt += 1
            deferred += sum(1 for _, o in iter_outputs(so) if stamp(o) <= p.t)
    return {"t1": {"prefix": pre1, "restore_ok": restore, "window": hw.hexdigest(), "bars": hb.hexdigest(),
                   "state": st, "state_max_ts": st_max, "n_bars": n1, "seconds": T1 - T},
            "t2": {"prefix": pre2, "bars": hb2.hexdigest(), "n_bars": n_le, "n_after": n_gt, "deferred": deferred,
                   "seconds": time.perf_counter() - T1}}


def worker_main(conn: Any, asset: str, variant: str, df_path: str | None) -> None:
    import sys

    sys.setrecursionlimit(10_000)
    try:
        cfg = load_config()
        session = load_sessions().for_asset(asset)
        df = pl.read_parquet(df_path) if df_path else load_bars(asset).select(list(COLS))
        tfs = list(cfg.data["levels"]["timeframes_primary"])
        conn.send({"ready": True, "rows": df.height, "polars_threads": pl.thread_pool_size()})
    except Exception:  # pragma: no cover
        conn.send({"error": traceback.format_exc()})
        return
    with spy():
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                conn.send({"peak_rss_mb": peak_rss_mb()})
                break
            try:
                _, p, snap, ds = msg
                conn.send(point_request(p, snap, ds, df=df, cfg=cfg, session=session, asset=asset, tfs=tfs,
                                        variant=variant))
            except Exception:
                conn.send({"error": traceback.format_exc()})


class Worker:
    def __init__(self, asset: str, variant: str, df_path: str | None) -> None:
        import multiprocessing as mp

        import os

        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(child, asset, variant, df_path), daemon=True)
        old = {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS")}
        os.environ["POLARS_MAX_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"    # parent + worker = 2 threads
        try:
            self.proc.start()
        finally:
            for k, x in old.items():
                if x is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = x
        r = self.conn.recv()
        if "error" in r:
            raise RuntimeError(r["error"])
        self.pid = self.proc.pid
        self.polars_threads = r.get("polars_threads")
        self.outstanding: Any = None

    def submit(self, msg: tuple[Any, ...], ctx: Any) -> Any:
        """Send one request; return the (context, result) of the previous one (at most one outstanding)."""
        prev = self.collect()
        self.conn.send(msg)
        self.outstanding = ctx
        return prev

    def collect(self) -> Any:
        if self.outstanding is None:
            return None
        r = self.conn.recv()
        ctx, self.outstanding = self.outstanding, None
        if "error" in r:
            raise RuntimeError(r["error"])
        return ctx, r

    def close(self) -> float | None:
        peak = None
        try:
            self.conn.send(("stop",))
            if self.conn.poll(60):
                peak = self.conn.recv().get("peak_rss_mb")
        except Exception:
            pass
        self.proc.join(30)
        if self.proc.is_alive():
            self.proc.terminate()
        return peak


# ---------------------------------------------------------------- one asset

def session_anchor_report(higher: dict[str, pl.DataFrame], session: Any) -> dict[str, Any]:
    """Rule 4: bar boundaries follow the configured session (00:00 UTC for crypto)."""
    from vecopen.bars import INTRADAY_MINUTES

    rep: dict[str, Any] = {"session": session.name, "tz": session.tz, "day_start": session.day_start}
    for tf, fr in higher.items():
        to = fr["ts_open"].dt.epoch("ms").to_numpy()
        tc = fr["ts_close"].dt.epoch("ms").to_numpy()
        nb = fr["n_base"].to_numpy()
        if tf in INTRADAY_MINUTES:
            step = INTRADAY_MINUTES[tf] * 60_000
            aligned = (to % step == 0) & (tc - to == step)
        elif tf == "1d":
            aligned = (to % DAY_MS == 0) & (tc - to == DAY_MS)
        elif tf == "1w":
            aligned = (to % DAY_MS == 0) & (((to // DAY_MS) + 3) % 7 == 0) & (tc - to == 7 * DAY_MS)
        else:
            dto = fr["ts_open"].dt
            dtc = fr["ts_close"].dt
            aligned = ((dto.day() == 1) & (dto.hour() == 0) & (dto.minute() == 0) & (dtc.day() == 1)
                       & (dtc.hour() == 0)).to_numpy()
        full = nb == (tc - to) // BASE_MS
        partial = np.flatnonzero(~full).tolist()
        rep[tf] = {"bars": int(len(to)), "misaligned": int((~aligned).sum()), "partial_buckets": len(partial),
                   "partial_only_first_bar": partial in ([], [0])}
    return rep


def run_asset(asset: str, n: int = 200, *, variant: str = "clean", df: pl.DataFrame | None = None,
              df_path: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The G1 audit of one asset (tests 1-3 plus rule checks). ``df``/``df_path`` replace the research frame (for
    the pytest slice and the non-vacuity variants); ``variant`` injects a known leak (non-vacuity only)."""
    import sys

    sys.setrecursionlimit(10_000)
    T = time.perf_counter()
    cfg = load_config()
    sessions = load_sessions()
    session = sessions.for_asset(asset)
    if df is None:
        df = load_bars(asset).select(list(COLS))
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    rs = RESAMPLERS[variant]
    higher = {tf: rs(df, tf, session) for tf in tfs}
    prov, prov_rep = provenance(df, higher, session, variant)
    anchors = session_anchor_report(higher, session) if variant == "clean" else {}
    base_ts = df["ts_close"].dt.epoch("ms").to_list()
    pts = plan(asset, cfg, base_ts, n, variant)
    snaps_at: dict[int, list[Point]] = {}
    ts_at: dict[int, list[Point]] = {}
    for p in pts:
        if p.s is not None:
            snaps_at.setdefault(p.s, []).append(p)
        ts_at.setdefault(p.t, []).append(p)
    audit = Audit(asset, cfg, session, df, higher, prov, independent=(variant == "clean"))
    worker = Worker(asset, variant, df_path)
    res1: list[dict[str, Any]] = []
    res2: list[dict[str, Any]] = []
    state_max_ok = True
    state_max_viol: list[str] = []
    t_setup = time.perf_counter() - T
    fs = FrameSet(df, higher)

    def judge(done: Any) -> None:
        if done is None:
            return
        (q, pre, wd, bd, dig_t), r = done
        res1.append(_judge1(q, r["t1"], pre, wd, bd, dig_t))
        res2.append(_judge2(q, r["t2"], r["t1"], pre, bd))
        bad = [x for x in (res1[-1], res2[-1]) if not x["pass"]]
        if bad:
            log(f"  {asset} t={iso(q.t)}: FAIL {[x['why'] for x in bad]}")

    try:
        with spy():
            pipe = Pipeline(cfg, session, asset)
            win: dict[int, tuple[Any, Any]] = {}          # point -> (bars + outputs digest, bars-only digest)
            snap: dict[int, tuple[bytes, str | None]] = {}
            for p in pts:
                if p.s is None:
                    win[p.i] = (hashlib.blake2b(digest_size=20), hashlib.blake2b(digest_size=20))
            n_bars = 0
            for b in make_stream(df, higher, variant):
                so = pipe.step(b)
                n_bars += 1
                audit.bar(b, so)
                if win:
                    rb = repr(b)
                    rec = (rb + repr(so.outs)).encode()
                    rbb = rb.encode()
                    for hw, hb in win.values():
                        hw.update(rec)
                        hb.update(rbb)
                if b.tf != "5m":
                    continue
                t = b.ts_close
                if t in snaps_at:
                    check = [q for q in snaps_at[t] if q.i % RESTORE_CHECK_EVERY == 0]
                    dig = None
                    if check:
                        dig, mx, _ = canon(pipe)
                        if mx > t:
                            state_max_ok = False
                            state_max_viol.append(f"state at {iso(t)} holds {iso(mx)}")
                    blob = pickle.dumps(pipe, protocol=5)
                    for q in snaps_at[t]:
                        snap[q.i] = (blob, dig if q in check else None)
                        win[q.i] = (hashlib.blake2b(digest_size=20), hashlib.blake2b(digest_size=20))
                if t in ts_at:
                    dig_t, mx, _ = canon(pipe)
                    if mx > t:
                        state_max_ok = False
                        state_max_viol.append(f"state at {iso(t)} holds {iso(mx)}")
                    audit.at_t(pipe, t)
                    pre = fs.digests(t)
                    for q in ts_at[t]:
                        hw, hb = win.pop(q.i)
                        blob, ds = snap.pop(q.i, (None, None))
                        judge(worker.submit(("point", q, blob, ds), (q, pre, hw.hexdigest(), hb.hexdigest(), dig_t)))
            judge(worker.collect())
            assert not win and not snap, "unclosed windows"
            late_origin = len(pipe.gen.late_origin)
    finally:
        worker_peak = worker.close()
    import sys as _sys

    labels_loaded = "vecopen.labels" in _sys.modules
    out = {"asset": asset, "variant": variant, "n_timestamps": len(pts), "base_bars": len(base_ts),
           "stream_bars": n_bars, "first_bar": iso(base_ts[0]), "last_bar": iso(base_ts[-1]),
           "seconds": {"setup": round(t_setup, 1), "total": round(time.perf_counter() - T, 1)},
           "peak_rss_mb": {"parent": round(peak_rss_mb(), 1), "worker": None if worker_peak is None else round(worker_peak, 1)},
           "test1": _agg(res1), "test2": _agg(res2), "provenance": prov_rep, "session_anchors": anchors,
           "state_timestamps_le_t": state_max_ok, "state_timestamp_violations": state_max_viol[:5],
           "audit": audit.summary(), "late_origin_flags": late_origin,
           "labels_module_loaded_during_run": labels_loaded, "_cov_rows": audit.cov_rows,
           "points": [{"t": iso(r["t"]), "s": iso(r["s"]), "literal": r["literal"], "pass1": r["pass"],
                       "pass2": q["pass"]} for r, q in zip(res1, res2)]}
    a = out["audit"]["checks"]
    t3 = [c for c in a if c.startswith("t3_")]
    out["test3"] = {"pass": all(a[c]["fail"] == 0 for c in t3) and all(
        v["max_src_after_close"] == 0 and v["min_src_before_open"] == 0 and v["keys_match_real_frame"]
        for v in prov_rep.values()), "checks": {c: a[c] for c in t3}}
    out["rules_pass"] = all(a[c]["fail"] == 0 for c in a if not c.startswith("t3_"))
    return out


def _judge1(p: Point, r: dict[str, Any], pre: dict[str, str], wd: str, bd: str, dig_t: str) -> dict[str, Any]:
    why = []
    bad_frames = [k for k in pre if r["prefix"].get(k) != pre[k]]
    if bad_frames:
        why.append(f"resampled frames of the truncated data differ on ts_close<=t: {bad_frames}")
    if r["restore_ok"] is False:
        why.append("snapshot restore not exact")
    if r["bars"] != bd:
        why.append("bars fed in (s,t] differ")
    if r["window"] != wd:
        why.append("outputs in (s,t] differ")
    if r["state"] != dig_t:
        why.append("state at t differs")
    if r["state_max_ts"] > p.t:
        why.append("replayed state holds a timestamp after t")
    return {"t": p.t, "s": p.s, "literal": p.literal, "pass": not why, "why": why, "bars": r["n_bars"],
            "bars_after_t": 0, "seconds": r["seconds"], "restore_checked": r["restore_ok"] is not None}


def _judge2(p: Point, r: dict[str, Any], r1: dict[str, Any], pre: dict[str, str], bd: str) -> dict[str, Any]:
    why = []
    bad_frames = [k for k in pre if r["prefix"].get(k) != pre[k]]
    if bad_frames:
        why.append(f"resampled frames of the perturbed data differ on ts_close<=t: {bad_frames}")
    if r["bars"] != bd or r["bars"] != r1["bars"]:
        why.append("perturbed stream differs on bars <= t")
    if r["deferred"]:
        why.append(f"{r['deferred']} outputs stamped <= t emitted after t")
    return {"t": p.t, "s": p.s, "literal": p.literal, "pass": not why, "why": why, "bars": r["n_bars"],
            "bars_after_t": r["n_after"], "seconds": r["seconds"], "fp": p.fp, "fv": p.fv}


def _agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
    fails = [r for r in rs if not r["pass"]]
    lit = [r for r in rs if r["literal"]]
    win = [(r["t"] - r["s"]) / DAY_MS for r in rs if r["s"] is not None]
    return {"n": len(rs), "passed": len(rs) - len(fails), "failed": len(fails), "pass": not fails and bool(rs),
            "literal_runs": len(lit), "literal_pass": all(r["pass"] for r in lit) if lit else None,
            "window_days_min_median_max": [round(min(win), 2), round(float(np.median(win)), 2), round(max(win), 2)]
            if win else None,
            "bars_replayed": int(sum(r["bars"] for r in rs)),
            "bars_after_t_replayed": int(sum(r["bars_after_t"] for r in rs)),
            "worker_seconds": round(sum(r["seconds"] for r in rs), 1),
            "failures": [{"t": iso(r["t"]), "s": iso(r["s"]), "why": r["why"]} for r in fails[:10]]}


# ---------------------------------------------------------------- stored tables (covariates only, no outcome)

def stored_covariates_check(asset: str, rows: list[tuple[str, str, int, float, float, float | None]]
                            ) -> dict[str, Any]:
    """The matching covariates in the stored events table equal the audited stream's values at t0."""
    p = EVENTS_DIR / asset / "events.parquet"
    if not p.exists():
        return {"present": False}
    schema = pl.read_parquet_schema(p)
    label_cols = sorted(c for c in schema if c in LABEL_COLUMNS or c.startswith(LABEL_PREFIXES))
    st = (pl.read_parquet(p, columns=["family", "event_key", "year", "age_log", "departure_atr", "regime"])
          .filter(pl.col("family").is_in([TREATED, P1])))
    mine = pl.DataFrame(rows, schema={"family": pl.String, "event_key": pl.String, "year": pl.Int64,
                                      "age_log": pl.Float64, "departure_atr": pl.Float64, "regime": pl.Float64},
                        orient="row")
    st = st.with_columns(pl.col("year").cast(pl.Int64)).sort("event_key")
    mine = mine.sort("event_key")
    out = {"present": True, "label_columns_in_events_table": label_cols}
    for fam in (TREATED, P1):
        a, b = st.filter(pl.col("family") == fam), mine.filter(pl.col("family") == fam)
        same = a.height == b.height and a.equals(b, null_equal=True)
        out[f"{fam}_identical"] = bool(same)
        if fam == TREATED:
            out["treated_rows"] = a.height
            if not same:
                j = a.join(b, on="event_key", how="full", suffix="_m")
                out["treated_mismatch_rows"] = int(j.filter(
                    ~((pl.col("age_log") == pl.col("age_log_m")) & (pl.col("departure_atr") == pl.col("departure_atr_m"))
                      & (pl.col("year") == pl.col("year_m")))).height)
    lv = EVENTS_DIR / asset / "levels.parquet"
    if lv.exists():
        out["levels_outcome_columns"] = sorted(c for c in pl.read_parquet_schema(lv) if c in ("fill_ts", "expire_ts"))
    return out


def matching_timing_check(assets: list[str]) -> dict[str, Any]:
    """Family-1 matching (phase 3 ``match_all``) seen from the timestamps only: how often the matched placebo event
    lies after the treated event, and whether both t0 are in the same calendar year. Reads the matches table and the
    ``t0`` column of treated and family-1 events; no label, no outcome, no placebo count."""
    ms_, ev = [], []
    for a in assets:
        mp, ep = EVENTS_DIR / a / "matches.parquet", EVENTS_DIR / a / "events.parquet"
        if not (mp.exists() and ep.exists()):
            continue
        ms_.append(pl.read_parquet(mp, columns=["asset", "tf", "year", "treated_key", "placebo_asset", "placebo_key"]))
        ev.append(pl.read_parquet(ep, columns=["asset", "event_key", "family", "t0"])
                  .filter(pl.col("family").is_in([TREATED, P1])).drop("family"))
    if not ms_:
        return {}
    m = pl.concat(ms_)
    e = pl.concat(ev)
    j = (m.join(e.rename({"event_key": "treated_key", "t0": "t0_t"}), on=["asset", "treated_key"], how="left")
         .join(e.rename({"asset": "placebo_asset", "event_key": "placebo_key", "t0": "t0_p"}),
               on=["placebo_asset", "placebo_key"], how="left"))
    j = j.with_columns(group=pl.when(pl.col("tf").is_in(["1d", "1w", "1M"])).then(pl.lit("daily_plus"))
                       .otherwise(pl.lit("intraday")))
    out: dict[str, Any] = {"groups": {}}
    for g in ("intraday", "daily_plus", "all"):
        x = j if g == "all" else j.filter(pl.col("group") == g)
        x = x.filter(pl.col("placebo_key").is_not_null())
        n = x.height
        out["groups"][g] = {
            "pairs": n,
            "placebo_later_frac": float((x["t0_p"] > x["t0_t"]).mean()) if n else float("nan"),
            "same_year_frac": float(((x["t0_p"].dt.year() == x["t0_t"].dt.year())
                                     & (x["t0_t"].dt.year() == x["year"])).mean()) if n else float("nan"),
            "cross_asset_frac": float((x["placebo_asset"] != x["asset"]).mean()) if n else float("nan"),
            "missing_t0": int(x["t0_t"].is_null().sum() + x["t0_p"].is_null().sum())}
    return out
