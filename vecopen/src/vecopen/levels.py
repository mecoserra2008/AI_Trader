"""Phase 2: the level registry and its state machine (SPEC section 2), streaming only (standing rule 2).

``LevelRegistry.update(bar) -> list[event]`` takes one closed bar, either a base (5m) bar or a closed higher-timeframe
bar, and returns what happened at that bar's close: ``LevelBorn``, ``FirstFill``, ``PolarityFlip``, ``LevelExpired``.
A batch run is a loop over ``update`` (``run`` below); no function takes a frame and returns per-bar outputs.

Multi-timeframe join (SPEC section 6, construction rule 1)
----------------------------------------------------------
Every bar is keyed by its close. Bars are fed in increasing ``ts_close``; at one instant ``t`` every higher-timeframe
bar closing at ``t`` comes before the base bar closing at ``t``. So when the base bar closing at ``t`` is processed,
the registry holds exactly the higher-timeframe information with ``close_time <= t``: the stream order IS the backward
as-of join on close time. Joining on open time cannot happen, because a higher-timeframe bar is never fed before its
close. ``update`` enforces the order and raises on a violation. ``bar_stream`` builds that order from frames.

Levels
------
A leg from ``LegOriginDetector`` (``vecopen.legs``) gives a level ``price = O_t`` born at the close of the
confirmation bar ``t+k*``. Its zone is ``price -/+ kappa_atr * atr`` with ``atr = ATR_{t-1}``, the same ATR as the
detection threshold and the leg size (decision, SPEC silent on which ATR). ``zone_lo``, ``zone_hi`` and ``atr`` are
frozen at birth: ``Level`` refuses to rebind them.

State machine
-------------
* ``untouched`` from birth until the first base bar closing strictly after ``birth_ts`` whose range intersects the
  zone (``l <= zone_hi and h >= zone_lo``, closed intervals). That bar is the decision event ``FirstFill`` with
  ``t0`` = its close. The base bar closing at ``birth_ts`` itself traded before the level was known, so it is never a
  fill.
* ``reference`` after the fill. Its initial polarity comes from the last close of the level's own timeframe as of
  t0: ``resistance`` if it is below the zone, ``support`` if above (inside the zone, which cannot happen for an
  untouched level on gap-free data, falls back to the leg side: down-leg -> resistance). It flips each time a
  tau-bar closes beyond the zone on the other side (close > zone_hi flips resistance to support, close < zone_lo
  flips support to resistance); closes inside the zone change nothing.
* ``expired`` at the close of the ``a_max_bars``-th tau-bar after birth (age counted from birth, in trade bars of the
  level's timeframe), from either live state. An expired level never fills and never counts for confluence.

Confluence at t0 (SPEC section 2)
---------------------------------
Sum of ``timeframe_weights[tf]`` over the OTHER active levels (untouched or reference, any timeframe in the registry,
born at or before t0, not expired) whose zones overlap the touched zone (closed intervals). Levels filled by the same
base bar are active and count for each other.

Placebo entry point (phase 3)
-----------------------------
``LevelRegistry(family=...)`` tags every level and event with a family. Only the ``treated`` registry runs the leg
detectors. A registry of any other family (the placebo families of ``vecopen.placebo``) has no detectors; its levels
come from ``register(leg)``, which calls the same ``_birth`` as a detected leg, with the same tau-bar count and last
close, so the zone, heaps, fills, polarity, expiry and confluence code is shared line for line. ``register`` must be
called after the registry has processed the leg's own tau-bar closing at ``leg.birth_ts`` and before the base bar of
that close, which is exactly where a detected leg is born (D7). For a placebo ``Leg`` the field ``origin_o`` holds
the placebo price and ``direction`` its side g (see ``vecopen.placebo``).

No-trade bars (decision): a bar with ``h == l`` and zero or absent volume (``vecopen.legs.is_no_trade``) is skipped:
it fills nothing, does not age levels, does not move polarity and does not reach the detector.

Indexing (engineering): untouched levels sit in two heaps, by ``zone_lo`` for levels above the last price and by
``zone_hi`` for levels below. A base bar pops only the levels its range reaches, so a fill check is O(log n) per
level touched instead of a scan of every untouched level. A level that a gapping bar jumps over moves to the other
heap. Expiry is a FIFO per timeframe (all levels of a timeframe share ``a_max_bars``).
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vecopen.bars import BASE_MINUTES, TIMEFRAMES, Session
from vecopen.config import LoadedConfig
from vecopen.legs import Bar, Leg, LegOriginDetector, config_data, is_no_trade

UNTOUCHED, REFERENCE, EXPIRED = "untouched", "reference", "expired"
TREATED = "treated"
RESISTANCE, SUPPORT = "resistance", "support"
TF_RANK = {tf: i for i, tf in enumerate(TIMEFRAMES)}


# ---------------------------------------------------------------- level

_FROZEN = frozenset({"id", "tf", "direction", "price", "zone_lo", "zone_hi", "atr", "birth_ts", "birth_count", "leg",
                     "family"})


@dataclass(eq=False)
class Level:
    """One registry level. The fields in ``_FROZEN`` are set once at birth and can never be rebound."""

    id: int
    tf: str
    direction: int
    price: float
    zone_lo: float
    zone_hi: float
    atr: float
    birth_ts: int
    birth_count: int          # tau-bar count of its timeframe at birth (age = count - birth_count)
    leg: Leg
    family: str = TREATED
    state: str = UNTOUCHED
    polarity: str | None = None
    fill_ts: int | None = None
    fill_age: int | None = None
    n_flips: int = 0
    expire_ts: int | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FROZEN and name in self.__dict__:
            raise AttributeError(f"Level.{name} is frozen at birth")
        object.__setattr__(self, name, value)

    @property
    def half_width(self) -> float:
        return (self.zone_hi - self.zone_lo) / 2

    def overlaps(self, lo: float, hi: float) -> bool:
        return self.zone_lo <= hi and lo <= self.zone_hi


# ---------------------------------------------------------------- events

@dataclass(frozen=True, slots=True)
class LevelBorn:
    kind: ClassVar[str] = "birth"
    ts: int
    level_id: int
    tf: str
    direction: int
    price: float
    zone_lo: float
    zone_hi: float
    atr: float
    origin_ts_close: int
    k_star: int
    leg_size_atr: float
    leg_vol: float | None
    family: str = TREATED


@dataclass(frozen=True, slots=True)
class FirstFill:
    """The decision event: first base bar into an untouched zone. Every field is known at ``t0``."""

    kind: ClassVar[str] = "first_fill"
    t0: int
    level_id: int
    tf: str
    direction: int
    price: float
    zone_lo: float
    zone_hi: float
    atr: float
    birth_ts: int
    origin_ts_close: int
    age: int                 # tau-bars of the level's timeframe since birth
    k_star: int
    leg_size_atr: float
    leg_vol: float | None
    polarity: str            # initial polarity of the reference level
    confluence: float
    overlapping: tuple[int, ...] = field(default=())
    family: str = TREATED

    @property
    def ts(self) -> int:
        return self.t0


@dataclass(frozen=True, slots=True)
class PolarityFlip:
    kind: ClassVar[str] = "flip"
    ts: int
    level_id: int
    tf: str
    polarity: str            # the new polarity
    n_flips: int


@dataclass(frozen=True, slots=True)
class LevelExpired:
    kind: ClassVar[str] = "expire"
    ts: int
    level_id: int
    tf: str
    state_before: str
    age: int


Event = LevelBorn | FirstFill | PolarityFlip | LevelExpired


# ---------------------------------------------------------------- registry

class LevelRegistry:
    """Streaming multi-timeframe level registry (module docstring)."""

    def __init__(self, config: LoadedConfig | Mapping[str, Any] | None = None, *,
                 timeframes: Iterable[str] | None = None, base_tf: str | None = None,
                 session: Session | None = None, family: str = TREATED) -> None:
        cfg = config_data(config)
        lv = cfg["levels"]
        if not isinstance(family, str) or not family:
            raise ValueError("family must be a non-empty string")
        self.family = family
        self.detect = family == TREATED          # only the treated registry detects legs
        self.kappa = float(lv["kappa_atr"])
        self.a_max = int(lv["a_max_bars"])
        if self.kappa < 0 or self.a_max < 1:
            raise ValueError(f"invalid levels parameters {dict(lv)}")
        self.weights = {str(k): float(w) for k, w in lv["timeframe_weights"].items()}
        self.base_tf = str(base_tf if base_tf is not None else cfg["events"]["base_timeframe"])
        tfs = list(timeframes if timeframes is not None else lv["timeframes_primary"])
        if self.base_tf not in TIMEFRAMES:
            raise ValueError(f"unknown base timeframe {self.base_tf!r}")
        for tf in tfs:
            if tf not in TIMEFRAMES or tf not in self.weights:
                raise ValueError(f"timeframe {tf!r} unknown or without a weight in levels.timeframe_weights")
            if TF_RANK[tf] < TF_RANK[self.base_tf]:
                raise ValueError(f"level timeframe {tf} is finer than the base timeframe {self.base_tf}")
        self.timeframes = tuple(sorted(dict.fromkeys(tfs), key=TF_RANK.__getitem__))
        self.detectors = ({tf: LegOriginDetector(tf, cfg, session=session) for tf in self.timeframes}
                          if self.detect else {})
        self._count = dict.fromkeys(self.timeframes, 0)            # trade tau-bars seen per timeframe
        self._last_close: dict[str, float] = {}
        self._fifo: dict[str, deque[Level]] = {tf: deque() for tf in self.timeframes}
        self._reference: dict[str, dict[int, Level]] = {tf: {} for tf in self.timeframes}
        self.active: dict[int, Level] = {}                          # untouched + reference, by id
        self._above: list[tuple[float, int, Level]] = []           # untouched, keyed by zone_lo (min-heap)
        self._below: list[tuple[float, int, Level]] = []           # untouched, keyed by -zone_hi
        self._n_untouched = 0
        self._next_id = 0
        self._t: int | None = None
        self._base_at_t = False
        self._tfs_at_t: set[str] = set()
        self.n_no_trade = dict.fromkeys((self.base_tf, *self.timeframes), 0)

    # -- read-only views
    def untouched(self) -> list[Level]:
        return [x for x in self.active.values() if x.state == UNTOUCHED]

    def age(self, level: Level) -> int:
        return self._count[level.tf] - level.birth_count

    # -- the streaming step
    def update(self, bar: Bar) -> list[Event]:
        """Process one closed bar (base or higher timeframe) and return the events at its close."""
        self._check_order(bar)
        out: list[Event] = []
        if bar.tf in self._count:
            out.extend(self._tau_bar(bar))
        if bar.tf == self.base_tf:
            out.extend(self._base_bar(bar))
        return out

    def _check_order(self, bar: Bar) -> None:
        tf, t = bar.tf, bar.ts_close
        if tf != self.base_tf and tf not in self._count:
            raise ValueError(f"registry has no timeframe {tf!r} (base {self.base_tf}, levels {self.timeframes})")
        if self._t is not None and t < self._t:
            raise ValueError(f"bars must be fed in increasing ts_close ({tf} bar at {t} after {self._t})")
        if self._t is None or t > self._t:
            self._t, self._base_at_t, self._tfs_at_t = t, False, set()
        if tf in self._tfs_at_t:
            raise ValueError(f"duplicate {tf} bar at {t}")
        if self._base_at_t:
            raise ValueError(f"{tf} bar at {t} arrived after the base bar of the same close: higher-timeframe bars "
                             "closing at t must precede the base bar closing at t (backward as-of join on close)")
        self._tfs_at_t.add(tf)
        if tf == self.base_tf:
            self._base_at_t = True

    # -- tau-bars: expiry, polarity, births
    def _tau_bar(self, bar: Bar) -> list[Event]:
        tf, t = bar.tf, bar.ts_close
        legs = self.detectors[tf].update(bar) if self.detect else ()   # the detector skips no-trade bars itself
        if is_no_trade(bar):
            self.n_no_trade[tf] += 1
            return []
        out: list[Event] = []
        self._count[tf] += 1
        count = self._count[tf]
        self._last_close[tf] = bar.c

        fifo = self._fifo[tf]
        while fifo and count - fifo[0].birth_count >= self.a_max:
            lvl = fifo.popleft()
            if lvl.state == EXPIRED:
                continue
            out.append(LevelExpired(t, lvl.id, tf, lvl.state, count - lvl.birth_count))
            if lvl.state == UNTOUCHED:
                self._n_untouched -= 1
            else:
                del self._reference[tf][lvl.id]
            lvl.state, lvl.expire_ts = EXPIRED, t
            del self.active[lvl.id]

        c = bar.c
        for lvl in self._reference[tf].values():
            new = None
            if lvl.polarity == RESISTANCE and c > lvl.zone_hi:
                new = SUPPORT
            elif lvl.polarity == SUPPORT and c < lvl.zone_lo:
                new = RESISTANCE
            if new is not None:
                lvl.polarity = new
                lvl.n_flips += 1
                out.append(PolarityFlip(t, lvl.id, tf, new, lvl.n_flips))

        for leg in legs:
            out.append(self._birth(leg, count, bar.c))
        if len(self._above) + len(self._below) > 2 * self._n_untouched + 4096:
            self._compact()
        return out

    def register(self, leg: Leg) -> LevelBorn:
        """Birth of an externally given level (placebo families) through the same ``_birth`` as a detected leg.

        Allowed only on a non-treated registry, at the instant ``leg.birth_ts`` after this registry processed the
        leg's own tau-bar closing then and before the base bar of that close (the birth point of a detected leg)."""
        if self.detect:
            raise ValueError("register() is for placebo registries; the treated registry detects its own legs")
        tf = leg.tf
        if tf not in self._count:
            raise ValueError(f"registry has no timeframe {tf!r}")
        if self._t != leg.birth_ts or tf not in self._tfs_at_t or self._base_at_t or tf not in self._last_close:
            raise ValueError(f"register({tf} leg born {leg.birth_ts}): the registry must be at that instant, after its "
                             f"{tf} bar and before the base bar of the same close (now at {self._t})")
        born = self._birth(leg, self._count[tf], self._last_close[tf])
        if len(self._above) + len(self._below) > 2 * self._n_untouched + 4096:
            self._compact()
        return born

    def _birth(self, leg: Leg, count: int, last_price: float) -> LevelBorn:
        half = self.kappa * leg.atr
        lvl = Level(id=self._next_id, tf=leg.tf, direction=leg.direction, price=leg.origin_o,
                    zone_lo=leg.origin_o - half, zone_hi=leg.origin_o + half, atr=leg.atr,
                    birth_ts=leg.birth_ts, birth_count=count, leg=leg, family=self.family)
        self._next_id += 1
        self.active[lvl.id] = lvl
        self._fifo[leg.tf].append(lvl)
        self._n_untouched += 1
        if last_price < lvl.zone_hi:
            heapq.heappush(self._above, (lvl.zone_lo, lvl.id, lvl))
        else:
            heapq.heappush(self._below, (-lvl.zone_hi, lvl.id, lvl))
        return LevelBorn(leg.birth_ts, lvl.id, lvl.tf, lvl.direction, lvl.price, lvl.zone_lo, lvl.zone_hi, lvl.atr,
                         leg.origin_ts_close, leg.k_star, leg.leg_size_atr, leg.leg_vol, self.family)

    def _compact(self) -> None:
        self._above = [e for e in self._above if e[2].state == UNTOUCHED]
        self._below = [e for e in self._below if e[2].state == UNTOUCHED]
        heapq.heapify(self._above)
        heapq.heapify(self._below)

    # -- base bars: first fills
    def _base_bar(self, bar: Bar) -> list[Event]:
        if is_no_trade(bar):
            self.n_no_trade[self.base_tf] += 1
            return []
        t, h, l = bar.ts_close, bar.h, bar.l  # noqa: E741
        filled: list[Level] = []
        defer_a: list[tuple[float, int, Level]] = []
        defer_b: list[tuple[float, int, Level]] = []
        above, below = self._above, self._below
        while above and above[0][0] <= h:
            e = heapq.heappop(above)
            lvl = e[2]
            if lvl.state != UNTOUCHED:
                continue
            if lvl.birth_ts >= t:
                defer_a.append(e)
            elif l <= lvl.zone_hi:
                filled.append(lvl)
            else:                                   # the bar jumped over the zone: now below price
                heapq.heappush(below, (-lvl.zone_hi, lvl.id, lvl))
        while below and -below[0][0] >= l:
            e = heapq.heappop(below)
            lvl = e[2]
            if lvl.state != UNTOUCHED:
                continue
            if lvl.birth_ts >= t:
                defer_b.append(e)
            elif h >= lvl.zone_lo:
                filled.append(lvl)
            else:
                heapq.heappush(above, (lvl.zone_lo, lvl.id, lvl))
        for e in defer_a:
            heapq.heappush(above, e)
        for e in defer_b:
            heapq.heappush(below, e)
        if not filled:
            return []
        filled.sort(key=lambda x: x.id)
        events: list[Event] = []
        polarities: list[str] = []
        for lvl in filled:                      # confluence first, while every filled level is still untouched
            conf, ids = self._confluence(lvl)
            last = self._last_close[lvl.tf]
            if last < lvl.zone_lo:
                pol = RESISTANCE
            elif last > lvl.zone_hi:
                pol = SUPPORT
            else:
                pol = RESISTANCE if lvl.direction < 0 else SUPPORT
            polarities.append(pol)
            age = self._count[lvl.tf] - lvl.birth_count
            events.append(FirstFill(t, lvl.id, lvl.tf, lvl.direction, lvl.price, lvl.zone_lo, lvl.zone_hi, lvl.atr,
                                    lvl.birth_ts, lvl.leg.origin_ts_close, age, lvl.leg.k_star, lvl.leg.leg_size_atr,
                                    lvl.leg.leg_vol, pol, conf, ids, self.family))
        for lvl, pol in zip(filled, polarities):
            lvl.state, lvl.polarity, lvl.fill_ts = REFERENCE, pol, t
            lvl.fill_age = self._count[lvl.tf] - lvl.birth_count
            self._reference[lvl.tf][lvl.id] = lvl
            self._n_untouched -= 1
        return events

    def _confluence(self, lvl: Level) -> tuple[float, tuple[int, ...]]:
        lo, hi = lvl.zone_lo, lvl.zone_hi
        ids = [o.id for o in self.active.values() if o is not lvl and o.zone_lo <= hi and lo <= o.zone_hi]
        ids.sort()
        w = self.weights
        return float(sum(w[self.active[i].tf] for i in ids)), tuple(ids)


# ---------------------------------------------------------------- batch helpers (loops over update)

def _bars_from_frame(df: Any, tf: str, chunk: int = 50_000) -> Iterator[Bar]:
    """Bars of one polars frame (``ts_close`` and optional ``ts_open`` UTC datetimes; ``o, h, l, c``, optional ``v``).
    Yields one bar at a time; no per-bar output is computed here. Rows are converted ``chunk`` at a time, so memory
    does not grow with the length of the frame."""
    for part in df.iter_slices(chunk):
        tc = part["ts_close"].dt.epoch("ms").to_list()
        to = part["ts_open"].dt.epoch("ms").to_list() if "ts_open" in part.columns else \
            [x - BASE_MINUTES * 60_000 for x in tc] if tf == "5m" else [None] * len(tc)
        v = part["v"].to_list() if "v" in part.columns else [None] * len(tc)
        for row in zip(tc, part["o"].to_list(), part["h"].to_list(), part["l"].to_list(), part["c"].to_list(), v, to):
            yield Bar(tf, row[0], row[1], row[2], row[3], row[4], row[5], row[6])


def bar_stream(base: Any, higher: Mapping[str, Any], base_tf: str = "5m") -> Iterator[Bar]:
    """Merge base bars and closed higher-timeframe bars into the registry's feed order: increasing ``ts_close``;
    at equal close, higher timeframes (coarsest last) before the base bar. This order is the backward as-of join
    on ``close_time <= t`` (module docstring)."""
    def keyed(df: Any, tf: str, base_flag: int) -> Iterator[tuple[int, int, int, Bar]]:
        for b in _bars_from_frame(df, tf):
            yield (b.ts_close, base_flag, TF_RANK[tf], b)

    its = [keyed(df, tf, 0) for tf, df in higher.items()]
    its.append(keyed(base, base_tf, 1))
    for item in heapq.merge(*its, key=lambda x: (x[0], x[1], x[2])):
        yield item[3]


def run(registry: LevelRegistry, bars: Iterable[Bar]) -> Iterator[Event]:
    """Feed ``bars`` one by one into ``registry.update`` and yield its events (a loop over the streaming API)."""
    for b in bars:
        yield from registry.update(b)
