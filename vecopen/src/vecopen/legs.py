"""Phase 2: leg-origin detection (SPEC section 2), streaming only (standing rule 2).

``LegOriginDetector(tf).update(bar) -> list[Leg]`` takes one CLOSED bar of timeframe ``tf`` and returns the legs
confirmed at that bar's close (zero, one, or one per direction). Nothing else is exposed: there is no function that
takes a frame and returns per-bar outputs. A batch run is a loop over ``update``.

Rule (SPEC section 2, parameters from ``load_config()["legs"]``)
-----------------------------------------------------------------
* Candidate bar t: ``s_t = sign(C_t - O_t) != 0`` and open-side wick ``omega_t <= omega_max`` with
  ``omega_t = (H_t - O_t)/(H_t - L_t)`` for ``s = -1`` and ``(O_t - L_t)/(H_t - L_t)`` for ``s = +1``.
* Confirmation at the first ``k in [0, n_confirm]`` with ``s_t (C_{t+k} - O_t) >= theta_L * ATR_{t-1}`` and
  ``s_t (X_{t+j} - O_t) > 0`` for every ``j in [1, k]`` (``X = H`` for down-legs, ``L`` for up-legs). The second
  condition is strict, so a bar whose high (down-leg) or low (up-leg) merely equals the open kills the candidate.
* One level per leg: when a leg of direction s confirms at bar ``t+k*``, every other pending candidate of direction
  s is discarded and no bar with index ``<= t+k*`` can start a new direction-s candidate. If several pending
  candidates of direction s confirm at the same bar, the earliest origin wins. Opposite directions are independent.
* The leg is born at the close of the confirmation bar ``t+k*`` (``Leg.birth_ts``), never at bar t.

ATR (SPEC section 6, construction rule 2)
-----------------------------------------
Wilder's ATR (RMA) of length ``legs.atr_len``: ``TR_1 = H - L``; ``TR_i = max(H - L, |H - C_{i-1}|, |L - C_{i-1}|)``;
seeded with the simple mean of the first ``atr_len`` TRs, then ``ATR_i = ((len - 1) ATR_{i-1} + TR_i) / len``.
It is updated incrementally and shifted by one bar: bar t sees ``ATR_{t-1}``, the value after bar t-1 closed, so
the origin bar's own range never enters its threshold. ``ATR_{t-1}`` is undefined for the first ``atr_len`` bars of
a timeframe, which therefore cannot be candidates. ``legs.atr_method`` may be ``wilder`` (default; the key is absent
from the frozen baseline, so the default applies) or ``simple`` (rolling mean of the last ``atr_len`` TRs).

No-trade bars (decision, SPEC silent)
-------------------------------------
A bar with zero range and zero (or absent) volume (``h == l`` and ``v in (0, None)``) is a no-trade bar: Binance
prints them during venue maintenance (``venue_flat``) and for illiquid minutes (``symbol_flat``). Such a bar is
invisible to the detector, exactly like a missing bar: it creates no candidate, does not advance a confirmation
window, does not kill or confirm a pending candidate, does not enter the ATR or the volume baseline, and its index
is not counted. The rule is evaluated on the bar alone (no lookup in ``gaps.json``, which is a whole-sample file).

Relative leg volume (decision, SPEC silent on the exact baseline)
-----------------------------------------------------------------
``Leg.leg_vol = sum(v over bars t..t+k*) / sum(baseline over the same bars)``. A bar's baseline is the mean volume
of the same intraday slot (position of the bar inside its trade day, from the asset's session) over the previous
``legs.volume_baseline_sessions`` sessions that printed that slot; for 1d, 1w and 1M bars there is one slot, so it
is the mean of the previous N bars. The baseline uses earlier sessions only (shifted by construction). ``leg_vol``
is ``None`` when the bar stream has no volume, when a baseline is not yet full, or when it is zero.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from vecopen.bars import CALENDAR_TFS, INTRADAY_MINUTES, TIMEFRAMES, UTC_24H, Session
from vecopen.config import LoadedConfig, load_config

ATR_METHODS = ("wilder", "simple")
DEFAULT_ATR_METHOD = "wilder"


class Bar(NamedTuple):
    """One closed bar. Timestamps are epoch milliseconds, UTC; the bar is keyed by its close (SPEC section 2)."""

    tf: str
    ts_close: int
    o: float
    h: float
    l: float  # noqa: E741
    c: float
    v: float | None = None
    ts_open: int | None = None


def is_no_trade(bar: Bar) -> bool:
    """Zero range and zero or absent volume: a venue_flat / symbol_flat bar (module docstring)."""
    return bar.h == bar.l and (bar.v is None or bar.v == 0)


def config_data(config: LoadedConfig | Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The config mapping: ``load_config().data`` when ``config`` is None."""
    if config is None:
        return load_config().data
    if isinstance(config, LoadedConfig):
        return config.data
    return config


# ---------------------------------------------------------------- ATR

class ATR:
    """Incremental ATR. ``value`` is the ATR through the last bar passed to ``update`` (None during warm-up)."""

    __slots__ = ("n", "method", "value", "_sum", "_count", "_prev_c", "_trs")

    def __init__(self, n: int, method: str = DEFAULT_ATR_METHOD) -> None:
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"atr_len must be a positive integer, got {n!r}")
        if method not in ATR_METHODS:
            raise ValueError(f"atr_method must be one of {ATR_METHODS}, got {method!r}")
        self.n, self.method = n, method
        self.value: float | None = None
        self._sum = 0.0
        self._count = 0
        self._prev_c: float | None = None
        self._trs: deque[float] = deque(maxlen=n)

    def update(self, h: float, l: float, c: float) -> None:  # noqa: E741
        pc = self._prev_c
        tr = h - l if pc is None else max(h - l, abs(h - pc), abs(l - pc))
        self._prev_c = c
        if self.method == "simple":
            if len(self._trs) == self.n:
                self._sum -= self._trs[0]
            self._trs.append(tr)
            self._sum += tr
            if len(self._trs) == self.n:
                self.value = self._sum / self.n
            return
        if self._count < self.n:
            self._sum += tr
            self._count += 1
            if self._count == self.n:
                self.value = self._sum / self.n
        else:
            self.value = (self.value * (self.n - 1) + tr) / self.n  # type: ignore[operator]


# ---------------------------------------------------------------- outputs

@dataclass(frozen=True, slots=True)
class Leg:
    """A confirmed leg. ``price = origin_o`` is the level; it exists from ``birth_ts`` (confirmation close) on."""

    tf: str
    direction: int            # s_t: -1 down-leg (level above price), +1 up-leg (level below price)
    origin_index: int         # index of bar t among this timeframe's trade bars
    origin_ts_open: int | None
    origin_ts_close: int
    origin_o: float
    origin_h: float
    origin_l: float
    origin_c: float
    omega: float              # open-side wick of the origin candle, fraction of its range
    confirm_index: int        # t + k*
    k_star: int
    birth_ts: int             # close of the confirmation bar t + k*
    confirm_close: float      # C_{t+k*}
    atr: float                # ATR_{t-1}, the unit of the threshold, the leg size and the zone
    leg_size_atr: float       # s_t (C_{t+k*} - O_t) / ATR_{t-1}
    leg_vol: float | None     # relative leg volume (module docstring), None without volume

    @property
    def price(self) -> float:
        return self.origin_o


class _Pending:
    __slots__ = ("idx", "s", "o", "h", "l", "c", "ts_open", "ts_close", "omega", "atr", "thr", "v_sum", "b_sum")

    def __init__(self, idx: int, s: int, bar: Bar, omega: float, atr: float, thr: float) -> None:
        self.idx, self.s, self.omega, self.atr, self.thr = idx, s, omega, atr, thr
        self.o, self.h, self.l, self.c = bar.o, bar.h, bar.l, bar.c
        self.ts_open, self.ts_close = bar.ts_open, bar.ts_close
        self.v_sum: float | None = 0.0
        self.b_sum: float | None = 0.0

    def add_volume(self, v: float | None, base: float | None) -> None:
        if v is None or base is None or self.v_sum is None or self.b_sum is None:
            self.v_sum = self.b_sum = None
        else:
            self.v_sum += v
            self.b_sum += base


# ---------------------------------------------------------------- detector

class LegOriginDetector:
    """Streaming leg-origin detector for one timeframe (module docstring). ``update(bar)`` is the only entry point."""

    def __init__(self, tf: str, config: LoadedConfig | Mapping[str, Any] | None = None, *,
                 session: Session | None = None) -> None:
        if tf not in TIMEFRAMES:
            raise ValueError(f"unknown timeframe {tf!r}; expected one of {TIMEFRAMES}")
        legs = config_data(config)["legs"]
        self.tf = tf
        self.theta = float(legs["theta_L_atr"])
        self.n_confirm = int(legs["n_confirm"])
        self.omega_max = float(legs["omega_max"])
        self.atr_len = int(legs["atr_len"])
        self.vol_n = int(legs["volume_baseline_sessions"])
        self.atr_method = str(legs.get("atr_method", DEFAULT_ATR_METHOD))
        if self.n_confirm < 0 or self.vol_n < 1 or self.theta <= 0 or not 0 <= self.omega_max <= 1:
            raise ValueError(f"invalid legs parameters {dict(legs)}")
        self.session = session if session is not None else UTC_24H
        self._atr = ATR(self.atr_len, self.atr_method)
        self._pending: dict[int, list[_Pending]] = {-1: [], 1: []}
        self._lock: dict[int, int] = {-1: -1, 1: -1}   # no new direction-s origin with index <= lock[s]
        self._i = 0                                     # trade-bar index of the next bar
        self._last_ts: int | None = None
        self._vol: dict[int, deque[float]] = {}
        self._tf_min = INTRADAY_MINUTES.get(tf)
        self._tz = None if self.session.tz == "UTC" else ZoneInfo(self.session.tz)
        h, m = self.session.day_start_hm
        self._day_start_min = 60 * h + m
        self.n_bars = 0
        self.n_no_trade = 0

    # -- point-in-time views (read-only; for tests and later phases)
    @property
    def atr(self) -> float | None:
        """ATR through the last trade bar seen (what the next bar will use as ATR_{t-1})."""
        return self._atr.value

    @property
    def pending(self) -> int:
        return len(self._pending[-1]) + len(self._pending[1])

    # -- volume baseline
    def _slot(self, bar: Bar) -> int:
        if self._tf_min is None or self.tf in CALENDAR_TFS:
            return 0
        ts_open = bar.ts_open if bar.ts_open is not None else bar.ts_close - self._tf_min * 60_000
        if self._tz is None:
            minute = (ts_open // 60_000) % 1440
        else:
            loc = datetime.fromtimestamp(ts_open / 1000, UTC).astimezone(self._tz)
            minute = loc.hour * 60 + loc.minute
        return ((minute - self._day_start_min) % 1440) // self._tf_min

    def _baseline(self, bar: Bar) -> float | None:
        """Baseline of this bar from earlier sessions, then record the bar's own volume for later sessions."""
        if bar.v is None:
            return None
        dq = self._vol.get(slot := self._slot(bar))
        if dq is None:
            dq = self._vol[slot] = deque(maxlen=self.vol_n)
        base = sum(dq) / self.vol_n if len(dq) == self.vol_n else None
        dq.append(float(bar.v))
        return base

    # -- the streaming step
    def update(self, bar: Bar) -> list[Leg]:
        """Process one closed bar of this timeframe; return the legs confirmed at its close."""
        if bar.tf != self.tf:
            raise ValueError(f"detector {self.tf} got a {bar.tf} bar")
        if self._last_ts is not None and bar.ts_close <= self._last_ts:
            raise ValueError(f"{self.tf}: bars must arrive in strictly increasing ts_close order "
                             f"({bar.ts_close} after {self._last_ts})")
        self._last_ts = bar.ts_close
        self.n_bars += 1
        if is_no_trade(bar):
            self.n_no_trade += 1
            return []
        i = self._i
        atr_prev = self._atr.value          # ATR_{i-1}: shifted by one bar
        base = self._baseline(bar)
        out: list[Leg] = []

        for s in (-1, 1):
            pend = self._pending[s]
            if not pend:
                continue
            x = bar.h if s < 0 else bar.l
            keep: list[_Pending] = []
            won: _Pending | None = None
            for p in pend:
                if s * (x - p.o) <= 0:        # traded back through (or to) the open: dead
                    continue
                p.add_volume(bar.v, base)
                if won is None and s * (bar.c - p.o) >= p.thr:
                    won = p
                    continue
                if i - p.idx >= self.n_confirm:   # window exhausted at k = n
                    continue
                keep.append(p)
            if won is not None:
                out.append(self._leg(won, i, bar))
                self._lock[s] = i
                self._pending[s] = []
            else:
                self._pending[s] = keep

        s = (bar.c > bar.o) - (bar.c < bar.o)
        if s and atr_prev is not None and atr_prev > 0 and i > self._lock[s]:
            rng = bar.h - bar.l
            omega = (bar.h - bar.o) / rng if s < 0 else (bar.o - bar.l) / rng
            if omega <= self.omega_max:
                p = _Pending(i, s, bar, omega, atr_prev, self.theta * atr_prev)
                p.add_volume(bar.v, base)
                if s * (bar.c - bar.o) >= p.thr:          # k* = 0
                    out.append(self._leg(p, i, bar))
                    self._lock[s] = i
                elif self.n_confirm > 0:
                    self._pending[s].append(p)

        self._atr.update(bar.h, bar.l, bar.c)
        self._i = i + 1
        return out

    def _leg(self, p: _Pending, i: int, bar: Bar) -> Leg:
        leg_vol = (p.v_sum / p.b_sum) if (p.v_sum is not None and p.b_sum) else None
        return Leg(tf=self.tf, direction=p.s, origin_index=p.idx, origin_ts_open=p.ts_open, origin_ts_close=p.ts_close,
                   origin_o=p.o, origin_h=p.h, origin_l=p.l, origin_c=p.c, omega=p.omega, confirm_index=i,
                   k_star=i - p.idx, birth_ts=bar.ts_close, confirm_close=bar.c, atr=p.atr,
                   leg_size_atr=p.s * (bar.c - p.o) / p.atr, leg_vol=leg_vol)
