"""Phase 2 acceptance: 12 hand-built synthetic cases for LegOriginDetector and LevelRegistry (SPEC sections 2, 8).

Every case is a short explicit bar sequence; the expected outputs are written by hand in the assertions, and the
comments give the arithmetic. All cases use the frozen baseline (theta_L = 2.0, n = 6, omega_max = 0.25,
atr_len = 14, kappa = 0.10, A_max = 200, weights 15m=2, 1h=4, 4h=6); ``test_baseline_values_used_by_the_cases``
pins those values, so a config change cannot silently change what the cases mean.

Warm-up: 14 bars (o, h, l, c) = (100, 100.5, 99.5, 100) have TR = 1 each, so Wilder's seed ATR_13 = 14/14 = 1.0
exactly and bar 14 is the first bar that can be a candidate (it sees ATR_13; ATR is shifted by one bar).
Timestamps are epoch ms; 15m bar i closes at (i + 1) * 15 min.

C01 birth at the confirmation bar (k* = 2), never at the origin bar
C02 k* = 0: born at the close of the origin bar itself; the threshold uses ATR_{t-1}, not ATR_t
C03 a bar that trades back to the open kills the candidate (the condition is strict)
C04 the confirmation window: confirmation at k = n works, at k = n + 1 it does not
C05 one level per leg: the earliest origin wins, the confirmation bar cannot start a new leg, the next bar can
C06 open-side wick filter: omega = omega_max is a candidate, omega > omega_max is not
C07 zone freeze: the zone is fixed at birth while ATR moves, and fills use the frozen zone
C08 first fill: the first base bar after birth into the zone is the event, t0 = its close; later touches are not
C09 multi-timeframe as-of join with phase 1's resample: a level born at T is not filled by the base bar closing at T
C10 polarity flip of a reference level on tau-bar closes beyond the zone
C11 expiry after A_max tau-bars: no fill afterwards
C12 confluence: weighted count of other active levels with overlapping zones (expired excluded, reference included)
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import polars as pl
import pytest
from vecopen.bars import UTC_24H, resample
from vecopen.legs import ATR, Bar, Leg, LegOriginDetector
from vecopen.levels import (
    RESISTANCE,
    SUPPORT,
    FirstFill,
    LevelBorn,
    LevelExpired,
    LevelRegistry,
    PolarityFlip,
    bar_stream,
)

from vecopen.config import load_config

CFG = load_config().data
M5, M15, H = 300_000, 900_000, 3_600_000
TF_MS = {"5m": M5, "15m": M15, "1h": H, "4h": 4 * H, "1d": 24 * H}
WARM = [(100.0, 100.5, 99.5, 100.0)] * 14       # TR = 1 each -> ATR_13 = 1.0


def bar(tf: str, ts_close: int, o: float, h: float, l: float, c: float, v: float | None = 1.0) -> Bar:  # noqa: E741
    return Bar(tf, ts_close, o, h, l, c, v, ts_close - TF_MS[tf])


def T(i: int) -> int:
    """Close of 15m bar i."""
    return (i + 1) * M15


def detect(rows, tf: str = "15m", config=CFG) -> tuple[LegOriginDetector, list[list[Leg]]]:
    det = LegOriginDetector(tf, config)
    return det, [det.update(bar(tf, (i + 1) * TF_MS[tf], *r)) for i, r in enumerate(rows)]


def legs_of(out: list[list[Leg]]) -> dict[int, list[Leg]]:
    return {i: x for i, x in enumerate(out) if x}


def feed15(reg: LevelRegistry, i: int, row, base=None) -> list:
    """Feed 15m bar i with its three 5m bars in the as-of order: 5m at T-10m, 5m at T-5m, the 15m bar at T,
    then the 5m bar at T. Default 5m bars: (o, h, l, c) then two zero-range bars at c (a consistent split)."""
    o, h, l, c = row  # noqa: E741
    base = base or [(o, h, l, c), (c, c, c, c), (c, c, c, c)]
    t = T(i)
    ev = reg.update(bar("5m", t - 2 * M5, *base[0]))
    ev += reg.update(bar("5m", t - M5, *base[1]))
    ev += reg.update(bar("15m", t, *row))
    ev += reg.update(bar("5m", t, *base[2]))
    return ev


def reg15(config=CFG) -> LevelRegistry:
    return LevelRegistry(config, timeframes=["15m"], session=UTC_24H)


# The leg used by the registry cases C07, C08, C10, C11: down-leg from bar 14 (open 100), confirmed at bar 16.
# Zone = 100 -/+ 0.1 * ATR_13 = [99.9, 100.1].
LEG = WARM + [
    (100.0, 100.0, 99.25, 99.5),    # 14 origin: down, omega 0, 0.5 ATR
    (99.5, 99.92, 98.75, 99.0),     # 15 high 99.92 < 100 keeps it alive (inside the future zone, before birth)
    (99.0, 99.95, 97.75, 98.0),     # 16 close 2.0 below the open = 2.0 * ATR_13: confirmed, k* = 2
]
LEG_BASE = {16: [(99.0, 99.5, 98.5, 98.6), (98.6, 98.7, 97.75, 97.9), (97.9, 99.95, 97.9, 98.0)]}


def feed_leg(reg: LevelRegistry) -> list:
    ev = []
    for i, row in enumerate(LEG):
        ev += feed15(reg, i, row, LEG_BASE.get(i))
    return ev


def test_baseline_values_used_by_the_cases():
    legs, lv = CFG["legs"], CFG["levels"]
    assert (legs["theta_L_atr"], legs["n_confirm"], legs["omega_max"], legs["atr_len"]) == (2.0, 6, 0.25, 14)
    assert (lv["kappa_atr"], lv["a_max_bars"]) == (0.10, 200)
    assert (lv["timeframe_weights"]["15m"], lv["timeframe_weights"]["1h"], lv["timeframe_weights"]["4h"]) == (2, 4, 6)
    assert CFG["events"]["base_timeframe"] == "5m"


# ---------------------------------------------------------------- C01

def test_c01_birth_at_confirmation_bar_not_origin_bar():
    rows = WARM + [
        (100.0, 100.0, 99.25, 99.5),   # 14 candidate: down, omega = 0, |C - O| = 0.5 ATR -> pending
        (99.5, 99.9, 98.75, 99.0),     # 15 high 99.9 < 100: alive; 1.0 ATR; own wick 0.4/1.15 > 0.25: no candidate
        (99.0, 99.5, 97.75, 98.0),     # 16 close 98.0: 100 - 98 = 2.0 >= 2.0 * ATR_13 = 2.0 -> confirmed, k* = 2
    ]
    _, out = detect(rows)
    assert legs_of(out).keys() == {16}            # nothing at the origin bar 14 or at 15
    (leg,) = out[16]
    assert (leg.direction, leg.origin_index, leg.k_star, leg.price) == (-1, 14, 2, 100.0)
    assert (leg.origin_ts_close, leg.birth_ts) == (T(14), T(16))
    assert (leg.atr, leg.leg_size_atr, leg.confirm_close) == (1.0, 2.0, 98.0)

    # the registry: no level exists after bars 14 and 15; the level is born by the 15m bar closing at T(16)
    reg = reg15()
    for i, row in enumerate(rows[:16]):
        assert not [e for e in feed15(reg, i, row) if isinstance(e, LevelBorn)]
        assert reg.active == {}
    ev = reg.update(bar("5m", T(16) - 2 * M5, 99.0, 99.5, 98.5, 98.6)) + reg.update(bar("5m", T(16) - M5, 98.6, 98.6, 97.75, 97.9))
    assert ev == [] and reg.active == {}          # still unknown before the 15m close
    (born,) = reg.update(bar("15m", T(16), *rows[16]))
    assert isinstance(born, LevelBorn)
    assert (born.ts, born.price, born.zone_lo, born.zone_hi, born.k_star) == (T(16), 100.0, 99.9, 100.1, 2)


# ---------------------------------------------------------------- C02

def test_c02_k_zero_and_atr_shifted_by_one_bar():
    # 14: down, omega 0, |C - O| = 2.5 >= 2 * ATR_13 = 2.0 -> confirmed at its own close (k* = 0).
    # Its TR is 16, so the unshifted ATR_14 = (13 * 1 + 16) / 14 = 29/14 and 2 * 29/14 = 4.14 > 2.5: using ATR_t
    # instead of ATR_{t-1} would reject it.
    det, out = detect(WARM + [(100.0, 100.0, 84.0, 97.5)])
    (leg,) = out[14]
    assert (leg.k_star, leg.origin_index, leg.birth_ts, leg.origin_ts_close) == (0, 14, T(14), T(14))
    assert (leg.atr, leg.leg_size_atr) == (1.0, 2.5)
    assert det.atr == (13 * 1.0 + 16.0) / 14
    assert 2.5 < CFG["legs"]["theta_L_atr"] * det.atr
    # during warm-up no bar can be a candidate (ATR_{t-1} undefined): the same bar at index 13 gives nothing
    _, out13 = detect(WARM[:13] + [(100.0, 100.0, 84.0, 97.5)])
    assert legs_of(out13) == {}


# ---------------------------------------------------------------- C03

def test_c03_trading_back_to_the_open_kills_the_candidate():
    origin = (100.0, 100.0, 99.25, 99.5)          # 14 down candidate, open 100
    confirm = (99.25, 99.5, 97.5, 97.75)          # 16 close 97.75: 2.25 ATR below 100 (would confirm 14)
    # 15: high 100.0 equals the open: s * (H - O) = 0 is not > 0 -> candidate 14 dies. Bar 16 is its own candidate
    # (omega 0.25/2 = 0.125) but only 1.5 < 2 * ATR_15 = 1.967 from its open, so it stays pending.
    _, out = detect(WARM + [origin, (99.5, 100.0, 99.0, 99.25), confirm])
    assert legs_of(out) == {}
    # the same with high 99.99 at bar 15: 14 survives and is confirmed at 16 with k* = 2
    _, out = detect(WARM + [origin, (99.5, 99.99, 99.0, 99.25), confirm])
    (leg,) = out[16]
    assert (leg.origin_index, leg.k_star, leg.leg_size_atr) == (14, 2, 2.25)


# ---------------------------------------------------------------- C04

STEPS = [                                         # k = 1..5: closes 99.5 .. 98.5, highs < 100, upper wick 0.2
    (99.75, 99.95, 99.4, 99.5),                   # omega 0.2/0.55 = 0.36: none of these is a candidate
    (99.5, 99.7, 99.15, 99.25),
    (99.25, 99.45, 98.9, 99.0),
    (99.0, 99.2, 98.65, 98.75),
    (98.75, 98.95, 98.4, 98.5),
]


def test_c04_confirmation_window_n():
    origin = (100.0, 100.0, 99.5, 99.75)          # 14 down candidate, 0.25 ATR
    # k = 6 = n: close 98.0 -> 2.0 ATR: confirmed at the last bar of the window
    _, out = detect(WARM + [origin, *STEPS, (98.5, 98.8, 97.9, 98.0)])
    (leg,) = out[20]
    assert (leg.origin_index, leg.k_star, leg.birth_ts) == (14, 6, T(20))
    assert legs_of(out).keys() == {20}
    # k = 6 closes at 98.25 (1.75 ATR) and k = 7 at 97.75 (2.25 ATR): k = 7 > n, so no leg at all
    _, out = detect(WARM + [origin, *STEPS, (98.5, 98.8, 98.15, 98.25), (98.25, 98.55, 97.65, 97.75)])
    assert legs_of(out) == {}


# ---------------------------------------------------------------- C05

def test_c05_one_level_per_leg():
    rows = WARM + [
        (100.0, 100.0, 99.5, 99.6),   # 14 A: down candidate, open 100, threshold 2 * ATR_13 = 2.0
        (99.6, 99.6, 99.0, 99.1),     # 15 B: also a down candidate (omega 0), threshold 2 * ATR_14; A alive (99.6 < 100)
        (99.1, 99.1, 96.9, 97.0),     # 16: A at 3.0 ATR -> leg A, k* = 2. B (2.6 from its open) is dropped. Bar 16
                                      #     would qualify on its own (2.1 >= 2 * ATR_15) but it is not after t+k*
        (97.0, 97.0, 94.4, 94.5),     # 17: first bar after t + k*: new origin, 2.5 >= 2 * ATR_16 -> leg, k* = 0
    ]
    atr14 = (13 * 1.0 + 0.5) / 14                 # TR_14 = 0.5
    atr15 = (13 * atr14 + 0.6) / 14               # TR_15 = 0.6
    atr16 = (13 * atr15 + 2.2) / 14               # TR_16 = 2.2
    assert 99.1 - 97.0 >= 2 * atr15               # bar 16 qualifies by itself: only the lockout rejects it
    _, out = detect(rows)
    assert legs_of(out).keys() == {16, 17}
    (a,), (d,) = out[16], out[17]
    assert (a.origin_index, a.k_star, a.price, a.leg_size_atr) == (14, 2, 100.0, 3.0)
    assert (d.origin_index, d.k_star, d.price) == (17, 0, 97.0)
    assert d.atr == pytest.approx(atr16, rel=1e-12)   # TRs like 99.6 - 99.0 are not exact binary fractions
    # the registry turns them into exactly two levels, at 100 and 97
    reg = reg15()
    born = [e for i, r in enumerate(rows) for e in feed15(reg, i, r) if isinstance(e, LevelBorn)]
    assert [(e.ts, e.price) for e in born] == [(T(16), 100.0), (T(17), 97.0)]


# ---------------------------------------------------------------- C06

def test_c06_open_side_wick_filter():
    # omega = (H - O)/(H - L) = (101 - 100)/(101 - 97) = 0.25 = omega_max: a candidate; 2.5 ATR -> leg at k* = 0
    _, out = detect(WARM + [(100.0, 101.0, 97.0, 97.5)])
    (leg,) = out[14]
    assert (leg.omega, leg.k_star) == (0.25, 0)
    # high 101.2: omega = 1.2/4.2 = 0.286 > 0.25: not a candidate, although the body is 2.5 ATR. Bar 15 is a down
    # candidate (omega 0.2/1.2) but only 0.9 from its own open: pending, nothing confirmed.
    _, out = detect(WARM + [(100.0, 101.2, 97.0, 97.5), (97.5, 97.7, 96.5, 96.6)])
    assert legs_of(out) == {}


# ---------------------------------------------------------------- C07

def test_c07_zone_frozen_at_birth():
    reg = reg15()
    feed_leg(reg)
    (lvl,) = reg.active.values()
    assert (lvl.zone_lo, lvl.zone_hi, lvl.atr) == (99.9, 100.1, 1.0)
    # 20 wide bars well below the zone (TR 9.8): ATR_15m rises towards ~5, the zone does not move
    for i in range(17, 37):
        feed15(reg, i, (95.0, 99.8, 90.0, 95.0), [(95.0, 99.8, 90.0, 95.0), (95.0,) * 4, (95.0,) * 4])
        assert (lvl.zone_lo, lvl.zone_hi, lvl.atr) == (99.9, 100.1, 1.0)
    atr_now = reg.detectors["15m"].atr
    assert atr_now > 4.5
    with pytest.raises(AttributeError):
        lvl.zone_lo = 100.0 - 0.1 * atr_now
    # a base bar reaching 99.85 would be inside a zone re-sized with today's ATR (100 -/+ 0.45); with the frozen
    # zone it is not a fill. A bar reaching exactly zone_lo = 99.9 is.
    t = T(37)
    assert reg.update(bar("5m", t - 2 * M5, 95.0, 99.85, 95.0, 96.0)) == []
    (fill,) = reg.update(bar("5m", t - M5, 96.0, 99.9, 96.0, 97.0))
    assert isinstance(fill, FirstFill) and (fill.zone_lo, fill.zone_hi, fill.t0) == (99.9, 100.1, t - M5)


# ---------------------------------------------------------------- C08

def test_c08_first_fill_is_the_decision_event():
    reg = reg15()
    ev = feed_leg(reg)
    # bar 15's high 99.92 and the last 5m bar of bar 16 (high 99.95, closing at T(16)) were inside the future zone
    # but traded before the level existed: no fill
    assert [type(e) for e in ev] == [LevelBorn]
    t = T(17)
    ev = reg.update(bar("5m", t - 2 * M5, 98.0, 99.5, 97.0, 99.0))      # high 99.5 < 99.9: no touch
    assert ev == []
    (fill,) = reg.update(bar("5m", t - M5, 99.0, 99.9, 98.9, 99.2))    # high = zone_lo: the first fill
    assert fill == FirstFill(t0=t - M5, level_id=0, tf="15m", direction=-1, price=100.0, zone_lo=99.9, zone_hi=100.1,
                             atr=1.0, birth_ts=T(16), origin_ts_close=T(14), age=0, k_star=2, leg_size_atr=2.0,
                             leg_vol=None, polarity=RESISTANCE, confluence=0.0, overlapping=())
    # age 0: no 15m bar has closed since birth; polarity resistance: the last 15m close (98.0) is below the zone.
    # leg_vol is None: the 20-session volume baseline is not full after 17 bars.
    assert reg.update(bar("15m", t, 98.0, 100.05, 97.0, 99.3)) == []   # closes below the zone: no flip
    assert reg.update(bar("5m", t, 99.2, 100.05, 99.1, 99.3)) == []    # a second touch is not an event
    (lvl,) = reg.active.values()
    assert (lvl.state, lvl.fill_ts, lvl.fill_age) == ("reference", t - M5, 0)


# ---------------------------------------------------------------- C09

def _df5(rows5: list[tuple[float, float, float, float]], t0: int) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_close": pl.Series([t0 + (i + 1) * M5 for i in range(len(rows5))]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "o": [r[0] for r in rows5], "h": [r[1] for r in rows5], "l": [r[2] for r in rows5], "c": [r[3] for r in rows5],
        "v": [1.0] * len(rows5),
    })


def test_c09_multi_timeframe_asof_join_on_close():
    # 5m bars, resampled to 15m by phase 1's resample (UTC session). Warm-up: 42 bars (100, 100.5, 99.5, 100) ->
    # 14 15m bars with TR 1. Then the LEG bars 14..16 split into 5m bars, then three 5m bars of bucket 17.
    t0 = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    rows5 = [(100.0, 100.5, 99.5, 100.0)] * 42 + [
        (100.0, 100.0, 99.25, 99.5), (99.5, 99.5, 99.5, 99.5), (99.5, 99.5, 99.5, 99.5),       # bucket 14
        (99.5, 99.92, 98.75, 99.0), (99.0, 99.0, 99.0, 99.0), (99.0, 99.0, 99.0, 99.0),         # bucket 15
        *LEG_BASE[16],                                                                         # bucket 16
        (98.0, 99.5, 97.0, 99.0), (99.0, 99.9, 98.9, 99.2), (99.2, 100.05, 99.1, 99.3),        # bucket 17
    ]
    df5 = _df5(rows5, t0)
    df15 = resample(df5, "15m", UTC_24H)
    assert df15.height == 18 and df15.row(16)[2:6] == LEG[16]      # resample reproduces the hand-built 15m bar
    birth_close = t0 + 17 * M15
    reg = LevelRegistry(CFG, timeframes=["15m"], session=UTC_24H)
    seen15: list[int] = []
    log = []
    for b in bar_stream(df5, {"15m": df15}):
        if b.tf == "15m":
            seen15.append(b.ts_close)
        else:   # backward as-of join: at base bar t the registry has seen exactly the 15m bars with close <= t
            assert seen15 == [x for x in (t0 + (i + 1) * M15 for i in range(18)) if x <= b.ts_close]
        log.append((b.tf, b.ts_close, reg.update(b)))
    born = [(tf, ts) for tf, ts, ev in log for e in ev if isinstance(e, LevelBorn)]
    fills = [(tf, ts) for tf, ts, ev in log for e in ev if isinstance(e, FirstFill)]
    assert born == [("15m", birth_close)]
    i_born = next(i for i, (tf, ts, ev) in enumerate(log) if ev and isinstance(ev[0], LevelBorn))
    assert log[i_born - 1][:2] == ("5m", birth_close - M5) and log[i_born + 1][:2] == ("5m", birth_close)
    assert log[i_born + 1][2] == []                # the 5m bar closing at T (high 99.95, in the zone): not a fill
    assert fills == [("5m", birth_close + 2 * M5)]  # first 5m bar after birth into the zone (high 99.9)
    # a 15m bar arriving after the base bar of the same close would be joined late: refused
    reg2 = LevelRegistry(CFG, timeframes=["15m"], session=UTC_24H)
    reg2.update(bar("5m", T(0), 100.0, 100.5, 99.5, 100.0))
    with pytest.raises(ValueError, match="as-of"):
        reg2.update(bar("15m", T(0), 100.0, 100.5, 99.5, 100.0))


# ---------------------------------------------------------------- C10

def test_c10_polarity_flips_on_tau_closes_beyond_the_zone():
    reg = reg15()
    feed_leg(reg)
    feed15(reg, 17, (98.0, 100.05, 97.0, 99.3), [(98.0, 99.5, 97.0, 99.0), (99.0, 99.9, 98.9, 99.2), (99.2, 100.05, 99.1, 99.3)])
    (lvl,) = reg.active.values()
    assert (lvl.state, lvl.polarity) == ("reference", RESISTANCE)       # filled at T(17) - 5m, as in C08
    rows = {                                     # (o, h, l, c); big wicks so that none of them starts a leg
        18: (99.3, 100.08, 98.3, 100.05),        # close 100.05 inside the zone [99.9, 100.1]: no flip
        19: (100.05, 100.6, 99.6, 100.5),        # close 100.5 > 100.1: resistance -> support (flip 1)
        20: (100.5, 100.7, 99.95, 100.0),        # close 100.0 inside the zone: support stays
        21: (100.0, 100.3, 99.4, 99.5),          # close 99.5 < 99.9: support -> resistance (flip 2)
    }
    ev = {i: feed15(reg, i, r) for i, r in rows.items()}
    assert ev == {18: [], 19: [PolarityFlip(T(19), 0, "15m", SUPPORT, 1)], 20: [],
                  21: [PolarityFlip(T(21), 0, "15m", RESISTANCE, 2)]}
    assert (lvl.polarity, lvl.n_flips) == (RESISTANCE, 2)


# ---------------------------------------------------------------- C11

def test_c11_expiry_after_a_max_tau_bars():
    reg = reg15()
    feed_leg(reg)                                # born at bar 16
    feed15(reg, 17, (98.0, 99.5, 94.5, 95.0))    # upper wick 1.5/5 = 0.3: no new leg
    for i in range(18, 216):                     # flat bars at 95, far below the zone
        assert feed15(reg, i, (95.0, 95.5, 94.5, 95.0)) == []
    (lvl,) = reg.active.values()
    assert reg.age(lvl) == 199                   # 199 15m bars since birth (bars 17..215)
    (exp,) = feed15(reg, 216, (95.0, 95.5, 94.5, 95.0))
    assert exp == LevelExpired(T(216), 0, "15m", "untouched", 200)
    assert reg.active == {} and lvl.state == "expired" and lvl.expire_ts == T(216)
    # price returns into the zone: an expired level never fills
    assert reg.update(bar("5m", T(217) - 2 * M5, 95.0, 100.0, 95.0, 99.95)) == []


# ---------------------------------------------------------------- C12

def test_c12_confluence_weighted_count_of_overlapping_active_levels():
    """Levels (id: tf, price, zone):
      0 E  15m 100.02 [99.92, 100.12]  born 3.75h, filled 16h, expired 53.75h (200 15m bars after birth)
      1 R  1h  100.10 [100.00, 100.20] born 15h, filled 16h, reference from then on
      2 L15 15m 100.00 [~99.9, ~100.1] born 54h
      3 N  15m 101.00 [~100.889, ~101.111] born 54.25h, never reached
      4 L4 4h  100.00 [99.9, 100.1]    born 60h
      5 L1 1h  100.05 [~99.938, ~100.162] born 61h
    The 5m bar at 16h (high 100.05) fills E and R: each counts the other (E: w1h = 4; R: w15m = 2).
    The 5m bar at 62h (high 100.0) fills L15, L4 and L1. Each counts R (reference, weight 4) and the other two; N
    does not overlap; E overlaps but is expired: L15 = 4 + 6 + 4 = 14, L4 = 4 + 2 + 4 = 10, L1 = 4 + 2 + 6 = 12."""
    warm = (100.0, 100.5, 99.5, 100.0)
    bars = []
    bars += [bar("4h", (i + 1) * 4 * H, *warm) for i in range(14)]
    bars.append(bar("4h", 60 * H, 100.0, 100.0, 97.4, 97.5))            # L4: 2.5 >= 2 * 1.0
    bars += [bar("1h", (j + 1) * H, *warm) for j in range(14)]
    bars.append(bar("1h", 15 * H, 100.1, 100.1, 97.4, 97.5))            # R: 2.6 >= 2 * 1.0
    bars.append(bar("1h", 61 * H, 100.05, 100.05, 97.4, 97.5))          # L1: 2.55 >= 2 * ATR_14 = 2 * 15.7/14
    bars += [bar("15m", T(k), *warm) for k in range(14)]
    bars.append(bar("15m", T(14), 100.02, 100.02, 97.4, 97.5))          # E: 2.52 >= 2 * 1.0
    bars += [bar("15m", T(k), 97.5, 98.0, 97.0, 97.5) for k in range(15, 215)]
    bars.append(bar("15m", T(215), 100.0, 100.0, 97.4, 97.5))           # L15: 2.5 >= 2 * ATR (~1.0000)
    bars.append(bar("15m", T(216), 101.0, 101.0, 98.4, 98.5))           # N: 2.5 >= 2 * ATR (~1.114)
    bars.append(bar("5m", 16 * H, 99.5, 100.05, 99.5, 99.6))
    bars.append(bar("5m", 62 * H, 97.5, 100.0, 97.4, 97.6))
    rank = {"5m": 9, "15m": 1, "1h": 2, "4h": 3}                        # at equal close: higher tf before base
    bars.sort(key=lambda b: (b.ts_close, rank[b.tf]))
    reg = LevelRegistry(CFG, timeframes=["15m", "1h", "4h"], session=UTC_24H)
    events = [e for b in bars for e in reg.update(b)]
    born = [(e.level_id, e.tf, e.price) for e in events if isinstance(e, LevelBorn)]
    assert born == [(0, "15m", 100.02), (1, "1h", 100.1), (2, "15m", 100.0), (3, "15m", 101.0), (4, "4h", 100.0),
                    (5, "1h", 100.05)]
    fills = [(e.t0, e.level_id, e.confluence, e.overlapping) for e in events if isinstance(e, FirstFill)]
    assert fills == [(16 * H, 0, 4.0, (1,)), (16 * H, 1, 2.0, (0,)),
                     (62 * H, 2, 14.0, (1, 4, 5)), (62 * H, 4, 10.0, (1, 2, 5)), (62 * H, 5, 12.0, (1, 2, 4))]
    (exp,) = [e for e in events if isinstance(e, LevelExpired)]
    assert exp == LevelExpired(T(214), 0, "15m", "reference", 200)
    assert sorted(reg.active) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- extra unit tests

def test_no_trade_bars_are_invisible():
    """venue_flat / symbol_flat bars (zero range, zero volume) create no candidate, do not count in the window and
    cannot kill a candidate; they never fill. A zero-range bar WITH volume is a real trade."""
    origin = (100.0, 100.0, 99.5, 99.75)
    flat_at_open = (100.0, 100.0, 100.0, 100.0)
    seq = WARM + [origin, *STEPS[:3]]
    # a flat bar priced exactly at the open, with v = 0, then k = 4, 5, 6 (C04's confirming sequence)
    det = LegOriginDetector("15m", CFG)
    out = [det.update(bar("15m", T(i), *r)) for i, r in enumerate(seq)]
    out.append(det.update(bar("15m", T(len(seq)), *flat_at_open, v=0.0)))
    tail = [*STEPS[3:], (98.5, 98.8, 97.9, 98.0)]
    out += [det.update(bar("15m", T(len(seq) + 1 + j), *r)) for j, r in enumerate(tail)]
    (leg,) = out[-1]
    assert (leg.origin_index, leg.k_star, leg.birth_ts) == (14, 6, T(21))     # k counts trade bars only
    assert (det.n_no_trade, det.n_bars) == (1, 22)
    # the same bar with volume 1 is a trade at the open: it kills candidate 14
    det = LegOriginDetector("15m", CFG)
    for i, r in enumerate(seq):
        det.update(bar("15m", T(i), *r))
    det.update(bar("15m", T(len(seq)), *flat_at_open, v=1.0))
    assert [x for j, r in enumerate(tail) for x in det.update(bar("15m", T(len(seq) + 1 + j), *r))] == []
    # registry: a flat base bar at the zone price with v = 0 fills nothing; with v > 0 it fills
    reg = reg15()
    feed_leg(reg)
    assert reg.update(bar("5m", T(17) - 2 * M5, 100.0, 100.0, 100.0, 100.0, v=0.0)) == []
    assert reg.n_no_trade["5m"] == 1
    (fill,) = reg.update(bar("5m", T(17) - M5, 100.0, 100.0, 100.0, 100.0, v=2.0))
    assert isinstance(fill, FirstFill)


def test_opposite_direction_is_not_locked():
    rows = WARM + [
        (100.0, 100.0, 99.25, 99.5),  # 14 down candidate A (open 100)
        (97.0, 99.0, 97.0, 98.9),     # 15 up candidate U (open 97, omega 0); high 99 < 100 keeps A
        (98.9, 99.5, 97.5, 97.9),     # 16 A confirmed (2.1 ATR), down lock until 16; U alive (low 97.5 > 97)
        (97.9, 99.6, 97.6, 99.5),     # 17 U: 99.5 - 97 = 2.5 >= 2 * ATR_14 -> up-leg, origin 15 <= 16 is fine
    ]
    _, out = detect(rows)
    assert [(x.direction, x.origin_index, x.k_star) for i in sorted(legs_of(out)) for x in out[i]] == [(-1, 14, 2), (1, 15, 2)]


def test_reference_level_expires_too():
    reg = reg15()
    feed_leg(reg)
    feed15(reg, 17, (98.0, 100.05, 97.0, 99.3), [(98.0, 99.5, 97.0, 99.0), (99.0, 99.9, 98.9, 99.2), (99.2, 100.05, 99.1, 99.3)])
    events = []
    for i in range(18, 217):
        events += feed15(reg, i, (99.3, 99.8, 98.8, 99.3))
    assert events == [LevelExpired(T(216), 0, "15m", "reference", 200)]


def test_a_bar_that_gaps_over_the_zone_is_not_a_fill_and_the_level_stays_live():
    reg = reg15()
    feed_leg(reg)                                       # zone [99.9, 100.1], above price (98.0)
    t = T(17)
    assert reg.update(bar("5m", t - 2 * M5, 100.5, 101.0, 100.2, 100.8)) == []   # gap up: low 100.2 > 100.1
    assert reg.update(bar("5m", t - M5, 100.8, 101.5, 100.3, 101.0)) == []       # now above the zone, untouched
    (fill,) = reg.update(bar("5m", t, 101.0, 101.0, 100.1, 100.4))              # back down to zone_hi: the fill
    assert (fill.t0, fill.level_id) == (t, 0)
    # the same gap inside a consistent 15m bar (100.5, 101.0, 100.2, 100.8): lower wick 0.3/0.8 > 0.25, so it
    # starts no up-leg; the level is jumped over, never touched, and stays the only live level
    reg = reg15()
    feed_leg(reg)
    ev = feed15(reg, 17, (100.5, 101.0, 100.2, 100.8),
                [(100.5, 101.0, 100.2, 100.8), (100.8, 100.9, 100.6, 100.8), (100.8, 100.9, 100.6, 100.8)])
    assert ev == []
    (lvl,) = reg.active.values()
    assert lvl.state == "untouched"


def test_feed_order_is_enforced():
    reg = LevelRegistry(CFG, timeframes=["15m", "1h"], session=UTC_24H)
    reg.update(bar("1h", H, *WARM[0]))
    reg.update(bar("15m", H, *WARM[0]))          # two higher timeframes at the same close: any order between them
    reg.update(bar("5m", H, *WARM[0]))
    with pytest.raises(ValueError, match="increasing"):
        reg.update(bar("5m", H - M5, *WARM[0]))
    with pytest.raises(ValueError, match="duplicate"):
        reg.update(bar("5m", H, *WARM[0]))
    with pytest.raises(ValueError, match="no timeframe"):
        reg.update(bar("4h", 4 * H, *WARM[0]))
    det = LegOriginDetector("15m", CFG)
    det.update(bar("15m", T(1), *WARM[0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        det.update(bar("15m", T(1), *WARM[0]))
    with pytest.raises(ValueError, match="got a 1h bar"):
        det.update(bar("1h", 10 * H, *WARM[0]))


def test_parameters_come_from_the_config():
    det = LegOriginDetector("15m")                # default: load_config()
    legs = load_config().data["legs"]
    assert (det.theta, det.n_confirm, det.omega_max, det.atr_len, det.vol_n, det.atr_method) == (
        legs["theta_L_atr"], legs["n_confirm"], legs["omega_max"], legs["atr_len"], legs["volume_baseline_sessions"],
        "wilder")
    reg = LevelRegistry()
    assert reg.timeframes == tuple(CFG["levels"]["timeframes_primary"]) and reg.base_tf == "5m"
    assert (reg.kappa, reg.a_max) == (0.10, 200)
    # a different theta changes the outcome of C02 (2.5 ATR < 3.0 ATR): the detector reads it, nothing is hard-coded
    alt = copy.deepcopy(CFG)
    alt["legs"]["theta_L_atr"] = 3.0
    _, out = detect(WARM + [(100.0, 100.0, 84.0, 97.5)], config=alt)
    assert legs_of(out) == {}
    alt = copy.deepcopy(CFG)
    alt["levels"]["kappa_atr"] = 0.25
    reg = reg15(alt)
    (born,) = feed_leg(reg)
    assert (born.zone_lo, born.zone_hi) == (99.75, 100.25)


def test_wilder_and_simple_atr_by_hand():
    w, s = ATR(3, "wilder"), ATR(3, "simple")
    for h, l, c in [(2.0, 1.0, 1.5), (3.0, 1.5, 2.5), (2.5, 2.0, 2.0), (6.0, 2.0, 5.0)]:  # TR 1, 1.5, 0.5, 4
        w.update(h, l, c)
        s.update(h, l, c)
    assert w.value == ((1 + 1.5 + 0.5) / 3 * 2 + 4) / 3     # seed mean(1, 1.5, 0.5) = 1, then (2 * 1 + 4) / 3 = 2
    assert s.value == (1.5 + 0.5 + 4) / 3
    with pytest.raises(ValueError):
        ATR(3, "ema")


def test_relative_leg_volume_by_hand():
    """1d bars (one slot per session): baseline = mean volume of the previous 20 daily bars."""
    rows = [(100.0, 100.5, 99.5, 100.0)] * 20     # v = 10 each
    det = LegOriginDetector("1d", CFG)
    for i, r in enumerate(rows):
        det.update(bar("1d", (i + 1) * 24 * H, *r, v=10.0))
    # 20: down candidate with v = 30 (baseline 10); 21: confirms (100 - 97.5 = 2.5 >= 2 * ATR_19), v = 20,
    # baseline (19 * 10 + 30) / 20 = 11.  leg_vol = (30 + 20) / (10 + 11)
    assert det.update(bar("1d", 21 * 24 * H, 100.0, 100.0, 99.0, 99.2, v=30.0)) == []
    (leg,) = det.update(bar("1d", 22 * 24 * H, 99.2, 99.5, 97.4, 97.5, v=20.0))
    assert leg.leg_vol == pytest.approx(50.0 / 21.0, rel=1e-12)
    # no volume in the stream -> None
    det = LegOriginDetector("1d", CFG)
    for i, r in enumerate(rows):
        det.update(bar("1d", (i + 1) * 24 * H, *r, v=None))
    det.update(bar("1d", 21 * 24 * H, 100.0, 100.0, 99.0, 99.2, v=None))
    (leg,) = det.update(bar("1d", 22 * 24 * H, 99.2, 99.5, 97.4, 97.5, v=None))
    assert leg.leg_vol is None


def test_intraday_volume_slots_follow_the_session():
    from vecopen.bars import load_sessions

    ny = load_sessions().conventions["fx_ny17"]
    det = LegOriginDetector("1h", CFG, session=ny)
    # 2024-07-01 21:00 UTC = 17:00 New York (EDT): the first 1h bar of the FX trade day -> slot 0
    ts_open = int(datetime(2024, 7, 1, 21, tzinfo=UTC).timestamp() * 1000)
    assert det._slot(Bar("1h", ts_open + H, 1, 1, 1, 1, 1.0, ts_open)) == 0
    # in January (EST) slot 0 starts at 22:00 UTC
    ts_open = int(datetime(2024, 1, 2, 22, tzinfo=UTC).timestamp() * 1000)
    assert det._slot(Bar("1h", ts_open + H, 1, 1, 1, 1, 1.0, ts_open)) == 0
    det = LegOriginDetector("15m", CFG, session=UTC_24H)
    ts_open = int(datetime(2024, 1, 2, 13, 45, tzinfo=UTC).timestamp() * 1000)
    assert det._slot(Bar("15m", ts_open + M15, 1, 1, 1, 1, 1.0, ts_open)) == 55
