"""Phase 3: outcome labels (SPEC section 3). THE LABELLING STEP: this module looks forward by definition.

It consumes touch events (``vecopen.events.TouchEvent``, all fields known at t0) plus the base bars AFTER t0, and
nothing else. No module that builds events, features or signals imports it; labels are written to their own table
and joined to events only by later analysis phases.

``label_event(event, bars, config)`` labels one event from an iterable of the base bars that follow ``t0`` (it stops
reading at the horizon). ``LabelStream`` labels many events in one pass over the base bars: ``add(event)`` when the
base bar closing at ``t0`` has been seen, ``update(bar)`` for every later base bar, ``finish()`` at the end of data.
Both drive the same ``LabelState.step``, so they cannot disagree.

Definitions (SPEC section 3; the choices where it is silent are marked D in ``reports/phase_3.md``)
--------------------------------------------------------------------------------------------------
* Unit ``A`` = the level's frozen ATR at birth (``atr`` of the event; the ATR_{t-1} of the origin candle that also
  sizes the zone). Level ``l``; zone ``[zone_lo, zone_hi] = l -/+ kappa A``; side ``g`` from the event.
* Horizon: the ``h_base`` base bars after t0, ``j = 1 .. h_base`` (all base bars, no-trade bars included, so the
  horizon is wall-clock time on a 24-hour session). The label window ends at the close of bar ``h_base``.
* Entry ``E`` = open of the first trade bar after t0 (normally bar j = 1). No-trade bars (``legs.is_no_trade``) are
  skipped for entry and barrier checks: a venue cannot fill there.
* Resolution: ``res = 1`` if some bar in the window has ``h >= l + k_pt A`` or ``l <= l - k_pt A``, else 0.
  ``res_side`` is +1 (up), -1 (down) or 0 (both in the first resolving bar). ``mae_atr`` is the maximum absolute
  excursion ``max(max h - l, l - min l) / A`` over the whole window (from the entry bar on).
* Fade (direction ``g``): target ``l + g k_pt A``, stop ``k_sl A`` beyond the opposite (far) zone edge,
  i.e. ``zone_hi + k_sl A`` when g = -1 and ``zone_lo - k_sl A`` when g = +1.
* Follow-through (direction ``-g``): target ``l - g k_pt A``, stop ``k_sl A`` beyond the zone edge on the approach
  side, ``zone_lo - k_sl A`` when g = -1 and ``zone_hi + k_sl A`` when g = +1.
  So the two targets are exactly the edges of the resolution band.
* ``a = dir (target - E)``, ``b = dir (E - stop)`` (price units). If ``a <= 0`` or ``b <= 0`` the entry is already
  at or through a barrier: the label is void (``reason = void_target`` / ``void_stop``, label and R None).
* Barrier check per trade bar: an open at or through the stop (target) exits there at the open; otherwise a bar
  whose range reaches both barriers is a STOP (tie rule ``stop_first``, ``tie = True``); otherwise the barrier the
  range reaches. Exits are at the barrier price (or the open on a gap through it). At the close of bar ``h_base`` an
  open label exits at that close (``reason = time``).
* Label 1 iff the target is hit first, else 0. Realised ``R = dir (exit - E) / b``: -1 at the stop, ``a / b`` at
  the target, anything in between at the time barrier. ``pnl_atr = dir (exit - E) / A`` is the same result in ATR
  units. R explodes when the entry sits a hair inside the stop (b of a fraction of a tick), so means of R are not
  robust before the cost model of phase 7; ``pnl_atr`` is.
* End times (for purging and uniqueness, phase 8): ``res_end_ts`` (first band exit, else the horizon end),
  ``fade_end_ts`` / ``follow_end_ts`` (the exit bar), ``mae_end_ts`` = ``horizon_end_ts`` (close of bar h_base).
* Censoring: if the data ends before the horizon, every label not yet decided is None and ``censored`` is True.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from vecopen.bars import BASE_MINUTES
from vecopen.config import LoadedConfig
from vecopen.legs import Bar, config_data, is_no_trade

FADE, FOLLOW = "fade", "follow"
_BASE_MS = BASE_MINUTES * 60_000


def label_params(config: LoadedConfig | Mapping[str, Any] | None = None) -> tuple[float, float]:
    lab = config_data(config)["labels"]
    if lab.get("tie_rule", "stop_first") != "stop_first":
        raise ValueError(f"labels.tie_rule must be stop_first (SPEC section 3), got {lab.get('tie_rule')!r}")
    k_pt, k_sl = float(lab["k_pt_atr"]), float(lab["k_sl_atr"])
    if k_pt <= 0 or k_sl < 0:
        raise ValueError(f"invalid label barriers {dict(lab)}")
    return k_pt, k_sl


@dataclass(frozen=True, slots=True)
class BarrierOutcome:
    """One directional triple-barrier label."""

    side: str                  # fade or follow
    direction: int             # +1 long, -1 short
    target: float
    stop: float
    a: float | None            # dir (target - E), price units
    b: float | None            # dir (E - stop), price units
    label: int | None          # 1 target first, 0 otherwise; None if void or censored
    r: float | None            # realised R multiple
    pnl_atr: float | None      # dir (exit - E) / A: the same result in ATR units (stable when b is tiny)
    exit: float | None
    reason: str                # target, stop, time, void_target, void_stop, no_entry, censored
    end_ts: int | None
    tie: bool                  # both barriers inside one bar, resolved as a stop


@dataclass(frozen=True, slots=True)
class Label:
    family: str
    level_id: int
    tf: str
    t0: int
    h_base: int
    horizon_end_ts: int        # close of bar h_base (nominal t0 + h_base * 5m while censored)
    entry_ts: int | None       # open time of the entry bar
    entry: float | None
    bars_seen: int
    censored: bool
    res: int | None
    res_side: int | None
    res_end_ts: int | None
    mae_atr: float | None
    mae_end_ts: int | None
    fade: BarrierOutcome
    follow: BarrierOutcome


class _Barrier:
    __slots__ = ("side", "dir", "target", "stop", "unit", "a", "b", "label", "r", "pnl", "exit", "reason", "end_ts",
                 "tie", "done")

    def __init__(self, side: str, direction: int, target: float, stop: float, unit: float) -> None:
        self.side, self.dir, self.target, self.stop, self.unit = side, direction, target, stop, unit
        self.a = self.b = self.label = self.r = self.pnl = self.exit = self.end_ts = None
        self.reason = "censored"
        self.tie = False
        self.done = False

    def enter(self, e: float, ts: int) -> None:
        d = self.dir
        self.a, self.b = d * (self.target - e), d * (e - self.stop)
        if self.a <= 0 or self.b <= 0:
            self.reason = "void_target" if self.a <= 0 else "void_stop"
            self.done = True
            self.end_ts = ts

    def _close(self, x: float, reason: str, ts: int, e: float) -> None:
        self.exit, self.reason, self.end_ts, self.done = x, reason, ts, True
        self.label = 1 if reason == "target" else 0
        self.r = self.dir * (x - e) / self.b
        self.pnl = self.dir * (x - e) / self.unit

    def step(self, o: float, h: float, l: float, ts: int, e: float) -> None:  # noqa: E741
        if self.dir > 0:
            gap_s, gap_t, hit_s, hit_t = o <= self.stop, o >= self.target, l <= self.stop, h >= self.target
        else:
            gap_s, gap_t, hit_s, hit_t = o >= self.stop, o <= self.target, h >= self.stop, l <= self.target
        if gap_s:
            self._close(o, "stop", ts, e)
        elif gap_t:
            self._close(o, "target", ts, e)
        elif hit_s:                       # stop-first tie rule: a bar reaching both barriers is a stop
            self.tie = hit_t
            self._close(self.stop, "stop", ts, e)
        elif hit_t:
            self._close(self.target, "target", ts, e)

    def outcome(self) -> BarrierOutcome:
        return BarrierOutcome(self.side, self.dir, self.target, self.stop, self.a, self.b, self.label, self.r, self.pnl,
                              self.exit, self.reason, self.end_ts, self.tie)


class LabelState:
    """The forward walk of one event; ``step(bar)`` returns True once the horizon is complete."""

    __slots__ = ("ev", "lvl", "a", "up", "dn", "j", "e", "entry_ts", "res", "res_side", "res_end", "hi", "lo",
                 "barriers", "open_b", "complete", "last_ts")

    def __init__(self, event: Any, k_pt: float, k_sl: float) -> None:
        lv = event.level
        self.ev, self.lvl, self.a = event, lv.price, lv.atr
        g, a = lv.g, lv.atr
        self.up, self.dn = lv.price + k_pt * a, lv.price - k_pt * a
        far, near = (lv.zone_hi, lv.zone_lo) if g < 0 else (lv.zone_lo, lv.zone_hi)
        self.barriers = (_Barrier(FADE, g, lv.price + g * k_pt * a, far - g * k_sl * a, a),
                         _Barrier(FOLLOW, -g, lv.price - g * k_pt * a, near + g * k_sl * a, a))
        self.open_b = list(self.barriers)
        self.j = 0
        self.e: float | None = None
        self.entry_ts: int | None = None
        self.res: int | None = None
        self.res_side: int | None = None
        self.res_end: int | None = None
        self.hi = -float("inf")
        self.lo = float("inf")
        self.complete = False
        self.last_ts = event.t0

    def step(self, bar: Bar) -> bool:
        if self.complete:
            return True
        if bar.ts_close <= self.last_ts:
            raise ValueError(f"label of level {self.ev.level_id}: bar at {bar.ts_close} is not after {self.last_ts}")
        self.last_ts = ts = bar.ts_close
        self.j += 1
        if not is_no_trade(bar):
            o, h, l = bar.o, bar.h, bar.l  # noqa: E741
            if self.e is None:
                self.e = o
                self.entry_ts = bar.ts_open if bar.ts_open is not None else ts - _BASE_MS
                for b in self.barriers:
                    b.enter(o, ts)
                self.open_b = [b for b in self.barriers if not b.done]
            e = self.e
            if h > self.hi:
                self.hi = h
            if l < self.lo:
                self.lo = l
            if self.res is None:
                up, dn = h >= self.up, l <= self.dn
                if up or dn:
                    self.res, self.res_side, self.res_end = 1, (1 if up else 0) - (1 if dn else 0), ts
            if self.open_b:
                for b in self.open_b:
                    b.step(o, h, l, ts, e)
                self.open_b = [b for b in self.open_b if not b.done]
        if self.j >= self.ev.level.h_base:
            if self.res is None:
                self.res, self.res_side, self.res_end = 0, 0, ts
            if self.e is not None:
                for b in self.open_b:
                    b._close(bar.c, "time", ts, self.e)
            else:
                for b in self.barriers:
                    b.reason, b.done, b.end_ts = "no_entry", True, ts
            self.open_b = []
            self.complete = True
        return self.complete

    def result(self) -> Label:
        ev, lv = self.ev, self.ev.level
        done = self.complete
        horizon_end = self.last_ts if done else ev.t0 + lv.h_base * _BASE_MS
        mae = (max(self.hi - self.lvl, self.lvl - self.lo) / self.a) if (done and self.e is not None) else None
        return Label(lv.family, lv.level_id, lv.tf, ev.t0, lv.h_base, horizon_end, self.entry_ts, self.e, self.j,
                     not done, self.res, self.res_side, self.res_end, mae, horizon_end if done else None,
                     self.barriers[0].outcome(), self.barriers[1].outcome())


def label_event(event: Any, bars: Iterable[Bar], config: LoadedConfig | Mapping[str, Any] | None = None) -> Label:
    """Label one touch event from the base bars that FOLLOW its t0 (reads at most ``h_base`` of them)."""
    k_pt, k_sl = label_params(config)
    st = LabelState(event, k_pt, k_sl)
    for b in bars:
        if st.step(b):
            break
    return st.result()


class LabelStream:
    """Labels many events in one pass over the base bars (module docstring)."""

    def __init__(self, config: LoadedConfig | Mapping[str, Any] | None = None, base_tf: str | None = None) -> None:
        cfg = config_data(config)
        self.k_pt, self.k_sl = label_params(cfg)
        self.base_tf = str(base_tf if base_tf is not None else cfg["events"]["base_timeframe"])
        self.open: list[LabelState] = []
        self._t: int | None = None

    def add(self, event: Any) -> None:
        """Start labelling ``event``; the next base bar passed to ``update`` must close after its t0."""
        if self._t is not None and event.t0 < self._t:
            raise ValueError(f"event at {event.t0} added after the stream reached {self._t}")
        self.open.append(LabelState(event, self.k_pt, self.k_sl))

    def update(self, bar: Bar) -> list[Label]:
        if bar.tf != self.base_tf:
            raise ValueError(f"LabelStream takes base ({self.base_tf}) bars, got {bar.tf}")
        self._t = bar.ts_close
        done: list[Label] = []
        keep: list[LabelState] = []
        for st in self.open:
            if bar.ts_close <= st.ev.t0:
                keep.append(st)
            elif st.step(bar):
                done.append(st.result())
            else:
                keep.append(st)
        self.open = keep
        return done

    def finish(self) -> list[Label]:
        """End of data: the remaining labels, censored where undecided."""
        out = [st.result() for st in self.open]
        self.open = []
        return out
