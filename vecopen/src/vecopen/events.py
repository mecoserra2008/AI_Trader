"""Phase 3: touch events (SPEC section 2), streaming only (standing rule 2).

``TouchDetector(registry).update(bar) -> list[TouchEvent]`` wraps one ``LevelRegistry`` (of any family: treated or a
placebo family) and turns each ``FirstFill`` into a ``TouchEvent``. An event fires at the close ``t0`` of the first
base (5m) bar whose range intersects an untouched zone (the registry's rule, unchanged). Every field of a
``TouchEvent`` is known at ``t0``: nothing here reads a bar after the one being processed. Labels, which look forward,
are a separate step (``vecopen.labels``) and nothing in this module imports it.

``TouchDetector.register(leg, ...)`` passes an externally given level (a placebo) to ``LevelRegistry.register``, the
same ``_birth`` as a detected leg.

Covariates recorded at birth (decisions, SPEC silent; marked D in ``reports/phase_3.md``)
---------------------------------------------------------------------------------------
* ``ref_close``: the last close of the level's own timeframe at birth (``Leg.confirm_close``). For a treated level
  it is ``C_{t+k*}``.
* side ``g = sign(ref_close - price)``, falling back to ``Leg.direction`` if zero: the side price is on at birth.
  For a treated level ``g = s_t`` exactly (the confirmation close is theta_L ATR beyond the open in direction s_t),
  and ``_born`` checks it. The fade trades in direction g and the follow-through in -g (``vecopen.labels``).
* ``departure_atr = |ref_close - price| / atr``: the departure distance at birth, in the level's frozen ATR. For a
  treated level it equals ``leg_size_atr``.
* ``regime = ln(ATR_{c-1} / mean TR_{c-1})`` at the level's source candle c (the origin candle for treated levels,
  families 2 and 3; the non-origin candle for family 1): the level's own ATR unit relative to the mean true range of
  the up to ``levels.a_max_bars`` trade tau-bars through c-1 (at least ``legs.atr_len``), so both terms are shifted
  by one bar (construction rule 2). The ATR is a replica of the detector's (same class, length and method); ``_born``
  checks that it equals the level's frozen ATR exactly, which also proves that the bar indices agree.
* ``sigma_birth`` / ``sigma_t0``: root mean square of the base-bar close-to-close price changes over the last
  ``h_base`` base bars before the instant (shifted: the base bar closing at ``t0`` is not included in ``sigma_t0``;
  at a birth the base bar of the same close has not been processed yet). Driftless RMS, in price units. None before
  ``h_base`` changes exist.
* ``d_edge = max(|ref_close - price| - kappa * atr, 0)``: the distance from price at birth to the near zone edge, in
  price units, the ``d`` of the touch baseline ``2 Phi(-d / (sigma sqrt(T)))`` with ``T = h_base``.
* ``approach_side = sign(prev_close - price)`` with ``prev_close`` the close of the base bar before the touch bar
  (that bar did not reach the zone, so it lies wholly on one side). ``approach_side != g`` flags a level that a gap
  jumped over before its fill.

Horizon in base bars (decision): ``h_base = labels.h_tau_bars * base bars per tau-bar``, with 3, 6, 12, 24, 48 base
bars per 15m, 30m, 1h, 2h, 4h bar, 288 per day, 2016 per week and 8766 (= 365.25 / 12 days * 288) per month. That is
exact wall-clock time on a 24-hour session (every crypto asset). Other sessions raise until CFD data exists.
"""

from __future__ import annotations

import math
from array import array
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from vecopen.bars import BASE_MINUTES, INTRADAY_MINUTES, UTC_24H, Session
from vecopen.config import LoadedConfig
from vecopen.legs import ATR, DEFAULT_ATR_METHOD, Bar, Leg, config_data, is_no_trade
from vecopen.levels import TREATED, FirstFill, LevelBorn, LevelExpired, LevelRegistry

BASE_PER_DAY = 24 * 60 // BASE_MINUTES                   # 288
BASE_PER_TAU_24H = {**{tf: m // BASE_MINUTES for tf, m in INTRADAY_MINUTES.items()},
                    "1d": BASE_PER_DAY, "1w": 7 * BASE_PER_DAY, "1M": round(365.25 / 12 * BASE_PER_DAY)}
_RING = 64                                                # tau-bars of (ATR_{i-1}, regime_i) kept per timeframe


def base_bars_per_tau(tf: str, session: Session | None = None) -> int:
    """Nominal number of base bars in one tau-bar (module docstring). Only 24-hour sessions are defined."""
    s = session if session is not None else UTC_24H
    if s.tz != "UTC" or s.day_start_hm != (0, 0):
        raise NotImplementedError(f"session {s.name}: the label horizon in base bars is defined for 24-hour UTC "
                                  "sessions only (no CFD data exists yet)")
    return BASE_PER_TAU_24H[tf]


def horizon_base_bars(tf: str, config: LoadedConfig | Mapping[str, Any] | None = None,
                      session: Session | None = None) -> int:
    """``h`` = ``labels.h_tau_bars`` tau-bars, in base bars (SPEC section 7)."""
    return int(config_data(config)["labels"]["h_tau_bars"]) * base_bars_per_tau(tf, session)


def side_at_birth(ref_close: float, price: float, fallback: int) -> int:
    """``g = sign(ref_close - price)``, or ``fallback`` when price sits exactly on the level."""
    return (ref_close > price) - (ref_close < price) or (1 if fallback >= 0 else -1)


# ---------------------------------------------------------------- per-timeframe volatility state

class TfVol:
    """Streaming ATR replica plus the mean true range of the last ``long_len`` trade bars, per timeframe.

    ``update(bar)`` returns the trade-bar index of ``bar`` (None for a no-trade bar, which is skipped exactly as the
    detector skips it) and records ``(ATR_{i-1}, regime_i)`` for that index, both computed BEFORE bar i enters."""

    __slots__ = ("atr", "min_len", "trs", "tr_sum", "prev_c", "i", "ring")

    def __init__(self, atr_len: int, method: str = DEFAULT_ATR_METHOD, long_len: int = 200) -> None:
        self.atr = ATR(atr_len, method)
        self.min_len = atr_len
        self.trs: deque[float] = deque(maxlen=max(long_len, atr_len))
        self.tr_sum = 0.0
        self.prev_c: float | None = None
        self.i = 0
        self.ring: dict[int, tuple[float | None, float | None]] = {}

    def update(self, bar: Bar) -> int | None:
        if is_no_trade(bar):
            return None
        i = self.i
        a = self.atr.value
        n = len(self.trs)
        mean_tr = self.tr_sum / n if n >= self.min_len else None
        regime = math.log(a / mean_tr) if (a is not None and a > 0 and mean_tr) else None
        self.ring[i] = (a, regime)
        self.ring.pop(i - _RING, None)
        pc = self.prev_c
        tr = bar.h - bar.l if pc is None else max(bar.h - bar.l, abs(bar.h - pc), abs(bar.l - pc))
        if n == self.trs.maxlen:
            self.tr_sum -= self.trs[0]
        self.trs.append(tr)
        self.tr_sum += tr
        self.prev_c = bar.c
        self.atr.update(bar.h, bar.l, bar.c)
        self.i = i + 1
        return i

    def at(self, i: int) -> tuple[float | None, float | None]:
        """``(ATR_{i-1}, regime_i)`` for a recent trade-bar index i."""
        try:
            return self.ring[i]
        except KeyError:
            raise KeyError(f"trade-bar index {i} is not in the last {_RING} bars (now {self.i})") from None


class Sigma:
    """Cumulative squared base-bar close changes; ``rms(w)`` over the last ``w`` changes (None if fewer exist)."""

    __slots__ = ("cum", "prev_c")

    def __init__(self) -> None:
        self.cum = array("d", [0.0])
        self.prev_c: float | None = None

    def update(self, c: float) -> None:
        if self.prev_c is not None:
            d = c - self.prev_c
            self.cum.append(self.cum[-1] + d * d)
        self.prev_c = c

    def rms(self, w: int) -> float | None:
        n = len(self.cum) - 1
        if w < 1 or n < w:
            return None
        return math.sqrt(max(self.cum[n] - self.cum[n - w], 0.0) / w)


# ---------------------------------------------------------------- outputs

@dataclass(frozen=True, slots=True)
class LevelInfo:
    """A level at birth, any family (the levels table). Every field is known at ``birth_ts``."""

    family: str
    level_id: int
    tf: str
    birth_ts: int
    price: float
    zone_lo: float
    zone_hi: float
    atr: float
    g: int
    leg_direction: int          # Leg.direction (s_t for treated levels)
    origin_index: int
    origin_ts_close: int
    k_star: int
    leg_size_atr: float
    leg_vol: float | None
    ref_close: float
    departure_atr: float
    d_edge: float
    regime: float | None
    sigma_birth: float | None
    h_base: int
    source_level_id: int | None = None      # the treated level a family-2/3 placebo derives from
    variant: str | None = None              # family 2: up/down; family 3: close/mid/extreme
    displacement_atr: float | None = None   # family 2: signed upsilon
    leg: Leg | None = None                  # the registry input (not written to tables)


@dataclass(frozen=True, slots=True)
class TouchEvent:
    """A first fill (decision event) with the level's birth covariates. Every field is known at ``t0``."""

    level: LevelInfo
    t0: int
    age: int                    # tau-bars of the level's timeframe since birth
    polarity: str
    confluence: float           # within the registry's own family
    approach_side: int
    prev_close: float | None
    sigma_t0: float | None
    touch_o: float
    touch_h: float
    touch_l: float
    touch_c: float

    @property
    def family(self) -> str:
        return self.level.family

    @property
    def level_id(self) -> int:
        return self.level.level_id

    @property
    def tf(self) -> str:
        return self.level.tf


# ---------------------------------------------------------------- detector

class TouchDetector:
    """Streaming touch-event detector over one ``LevelRegistry`` (module docstring)."""

    def __init__(self, registry: LevelRegistry, config: LoadedConfig | Mapping[str, Any] | None = None, *,
                 session: Session | None = None) -> None:
        cfg = config_data(config)
        legs = cfg["legs"]
        self.registry = registry
        self.family = registry.family
        self.base_tf = registry.base_tf
        self.kappa = float(cfg["levels"]["kappa_atr"])
        long_len = int(cfg["levels"]["a_max_bars"])
        method = str(legs.get("atr_method", DEFAULT_ATR_METHOD))
        self.vol = {tf: TfVol(int(legs["atr_len"]), method, long_len) for tf in registry.timeframes}
        self.h_base = {tf: horizon_base_bars(tf, cfg, session) for tf in registry.timeframes}
        self.sigma = Sigma()
        self.live: dict[int, LevelInfo] = {}      # untouched levels, by id
        self.new_levels: list[LevelInfo] = []     # born during the last update() / register() calls
        self.expired: list[LevelExpired] = []     # expiries during the last update()
        self._prev_close: float | None = None

    def update(self, bar: Bar) -> list[TouchEvent]:
        """Process one closed bar (base or higher timeframe); return the touch events at its close."""
        self.new_levels = []
        self.expired = []
        if bar.tf in self.vol:
            self.vol[bar.tf].update(bar)          # before the registry: a k* = 0 origin is this very bar
        out: list[TouchEvent] = []
        for e in self.registry.update(bar):
            if isinstance(e, LevelBorn):
                self._born(e)
            elif isinstance(e, FirstFill):
                out.append(self._touch(e, bar))
            elif isinstance(e, LevelExpired):
                self.expired.append(e)
                self.live.pop(e.level_id, None)
        if bar.tf == self.base_tf:
            self.sigma.update(bar.c)
            self._prev_close = bar.c
        return out

    def register(self, leg: Leg, *, source_level_id: int | None = None, variant: str | None = None,
                 displacement_atr: float | None = None) -> LevelInfo:
        """Birth of a placebo level through ``LevelRegistry.register`` (the same ``_birth`` as a detected leg)."""
        born = self.registry.register(leg)
        return self._born(born, source_level_id, variant, displacement_atr)

    def _born(self, e: LevelBorn, source_level_id: int | None = None, variant: str | None = None,
              displacement_atr: float | None = None) -> LevelInfo:
        lvl = self.registry.active[e.level_id]
        leg = lvl.leg
        atr_prev, regime = self.vol[leg.tf].at(leg.origin_index)
        if atr_prev != leg.atr:
            raise AssertionError(f"{self.family} level {e.level_id}: ATR replica {atr_prev!r} != frozen ATR {leg.atr!r}")
        ref = leg.confirm_close
        g = side_at_birth(ref, lvl.price, leg.direction)
        dist = abs(ref - lvl.price)
        if self.family == TREATED and (g != leg.direction or dist / leg.atr != leg.leg_size_atr):
            raise AssertionError(f"treated level {e.level_id}: side {g} / departure {dist / leg.atr} disagree with "
                                 f"the leg ({leg.direction}, {leg.leg_size_atr})")
        h = self.h_base[leg.tf]
        info = LevelInfo(self.family, lvl.id, lvl.tf, lvl.birth_ts, lvl.price, lvl.zone_lo, lvl.zone_hi, lvl.atr, g,
                         leg.direction, leg.origin_index, leg.origin_ts_close, leg.k_star, leg.leg_size_atr,
                         leg.leg_vol, ref, dist / leg.atr, max(dist - self.kappa * leg.atr, 0.0), regime,
                         self.sigma.rms(h), h, source_level_id, variant, displacement_atr, leg)
        self.live[lvl.id] = info
        self.new_levels.append(info)
        return info

    def _touch(self, e: FirstFill, bar: Bar) -> TouchEvent:
        info = self.live.pop(e.level_id)
        pc = self._prev_close
        side = 0 if pc is None else (pc > info.price) - (pc < info.price)
        return TouchEvent(replace(info, leg=None), e.t0, e.age, e.polarity, e.confluence, side, pc, self.sigma.rms(info.h_base),
                          bar.o, bar.h, bar.l, bar.c)
