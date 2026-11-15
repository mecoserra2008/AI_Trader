"""Phase 3: triple-barrier labels (SPEC section 3), hand-built cases plus a stream-vs-function property.

Geometry used by the hand cases (frozen baseline: k_pt = 1.0, k_sl = 0.5, kappa = 0.10), level l = 100, frozen
ATR A = 1, zone [99.9, 100.1]:

* g = -1 (down-leg: level above price). Fade = short: target 99.0, stop 100.1 + 0.5 = 100.6.
  Follow = long: target 101.0, stop 99.9 - 0.5 = 99.4. Band [99.0, 101.0].
* g = +1 (up-leg: level below price). Fade = long: target 101.0, stop 99.9 - 0.5 = 99.4.
  Follow = short: target 99.0, stop 100.1 + 0.5 = 100.6.

With the entry at the near edge (99.9 for g = -1): fade a = 0.9, b = 0.7; follow a = 1.1, b = 0.5.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from vecopen.bars import Session
from vecopen.events import LevelInfo, TouchEvent, base_bars_per_tau, horizon_base_bars
from vecopen.labels import LabelStream, label_event, label_params
from vecopen.legs import Bar

from vecopen.config import load_config

CFG = load_config().data
M5 = 300_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % M5)


def event(g: int = -1, price: float = 100.0, atr: float = 1.0, h_base: int = 18, t0: int = T0, lid: int = 0):
    k = CFG["levels"]["kappa_atr"]
    info = LevelInfo("treated", lid, "15m", t0 - 10 * M5, price, price - k * atr, price + k * atr, atr, g, g, 0, 0, 0,
                     2.0, None, price + g * 2 * atr, 2.0, (2 - k) * atr, 0.0, None, h_base)
    return TouchEvent(info, t0, 0, "resistance", 0.0, g, None, None, 0.0, 0.0, 0.0, 0.0)


def bars(rows, t0: int = T0):
    """Base bars j = 1, 2, ... after t0 from (o, h, l, c[, v]) rows."""
    out = []
    for j, r in enumerate(rows, start=1):
        o, h, l, c = r[:4]  # noqa: E741
        v = r[4] if len(r) > 4 else 1.0
        out.append(Bar("5m", t0 + j * M5, o, h, l, c, v, t0 + (j - 1) * M5))
    return out


def test_frozen_label_parameters():
    assert label_params(CFG) == (1.0, 0.5)
    assert CFG["labels"]["tie_rule"] == "stop_first" and CFG["levels"]["kappa_atr"] == 0.10


def test_barrier_geometry_both_sides():
    for g, fade, follow in ((-1, (-1, 99.0, 100.6), (1, 101.0, 99.4)), (1, (1, 101.0, 99.4), (-1, 99.0, 100.6))):
        lab = label_event(event(g), bars([(100.0 - 0.1 * g, 100.05, 99.95, 100.0)]), CFG)
        assert (lab.fade.direction, lab.fade.target, pytest.approx(lab.fade.stop)) == fade
        assert (lab.follow.direction, lab.follow.target, pytest.approx(lab.follow.stop)) == follow
    lab = label_event(event(-1), bars([(99.9, 99.95, 99.85, 99.9)]), CFG)
    assert lab.entry == 99.9
    assert (lab.fade.a, lab.fade.b) == (pytest.approx(0.9), pytest.approx(0.7))
    assert (lab.follow.a, lab.follow.b) == (pytest.approx(1.1), pytest.approx(0.5))


# ---------------------------------------------------------------- the stop-first tie rule

def test_stop_first_tie_rule_fade_short():
    # bar 1 reaches the fade target 99.0 AND the fade stop 100.6: stop first; the follow only reaches its stop 99.4
    lab = label_event(event(-1), bars([(99.9, 100.7, 98.9, 99.0)]), CFG)
    f = lab.fade
    assert (f.label, f.reason, f.tie, f.exit) == (0, "stop", True, 100.6)
    assert f.r == pytest.approx(-1.0) and f.pnl_atr == pytest.approx(-0.7)
    assert f.end_ts == T0 + M5
    w = lab.follow
    assert (w.label, w.reason, w.tie, w.exit) == (0, "stop", False, 99.4)
    assert w.pnl_atr == pytest.approx(-0.5)


def test_stop_first_tie_rule_both_labels_and_both_sides():
    for g, row in ((-1, (99.9, 101.2, 98.8, 100.0)), (1, (100.1, 101.2, 98.8, 100.0))):
        lab = label_event(event(g), bars([row]), CFG)
        for b in (lab.fade, lab.follow):
            assert (b.label, b.reason, b.tie) == (0, "stop", True), (g, b)
            assert b.exit == pytest.approx(b.stop) and b.r == pytest.approx(-1.0)
        assert (lab.res, lab.res_side) == (1, 0)          # the band was left on both sides in one bar


def test_tie_only_when_both_barriers_are_in_the_same_bar():
    # target in bar 1, stop in bar 2: target first, no tie
    lab = label_event(event(-1), bars([(99.9, 100.0, 98.95, 99.0), (99.0, 100.7, 98.9, 100.5)]), CFG)
    assert (lab.fade.label, lab.fade.reason, lab.fade.tie, lab.fade.exit) == (1, "target", False, 99.0)
    assert lab.fade.r == pytest.approx(0.9 / 0.7)


def test_open_through_a_barrier_exits_at_the_open_and_is_not_a_tie():
    # bar 2 opens at 98.5, below the fade target 99.0, and its range also reaches the fade stop 100.6:
    # the open is known to come first, so the fade exits at the target side, at the open price
    lab = label_event(event(-1), bars([(99.9, 100.0, 99.5, 99.6), (98.5, 100.8, 98.4, 99.0)]), CFG)
    assert (lab.fade.label, lab.fade.reason, lab.fade.tie, lab.fade.exit) == (1, "target", False, 98.5)
    assert lab.fade.r == pytest.approx((99.9 - 98.5) / 0.7)
    assert (lab.follow.label, lab.follow.reason, lab.follow.exit) == (0, "stop", 98.5)    # long, gapped below 99.4
    assert lab.follow.r == pytest.approx((98.5 - 99.9) / 0.5)


# ---------------------------------------------------------------- time barrier, voids, entry, resolution, censoring

def test_time_barrier_and_reads_at_most_h_bars():
    rows = [(99.9, 100.2, 99.6, 100.0), (100.0, 100.3, 99.7, 99.8), (99.8, 100.0, 99.5, 99.7)]

    def gen():
        yield from bars(rows)
        raise AssertionError("label_event read past the horizon")

    lab = label_event(event(-1, h_base=3), gen(), CFG)
    assert not lab.censored and lab.bars_seen == 3
    assert (lab.fade.reason, lab.fade.label, lab.fade.exit) == ("time", 0, 99.7)
    assert lab.fade.r == pytest.approx(-(99.7 - 99.9) / 0.7) and lab.fade.pnl_atr == pytest.approx(0.2)
    assert (lab.follow.reason, lab.follow.label) == ("time", 0)
    assert lab.follow.r == pytest.approx((99.7 - 99.9) / 0.5)
    assert (lab.res, lab.res_side, lab.res_end_ts) == (0, 0, T0 + 3 * M5)
    assert lab.horizon_end_ts == lab.mae_end_ts == T0 + 3 * M5
    assert lab.mae_atr == pytest.approx(0.5)           # max(100.3 - 100, 100 - 99.5) / 1


def test_void_when_the_entry_is_already_through_a_barrier():
    lab = label_event(event(-1), bars([(100.7, 100.8, 100.6, 100.7)]), CFG)    # above the fade stop 100.6
    assert (lab.fade.reason, lab.fade.label, lab.fade.r, lab.fade.pnl_atr) == ("void_stop", None, None, None)
    assert lab.follow.reason != "void_stop" and lab.follow.a == pytest.approx(0.3) and lab.follow.b == pytest.approx(1.3)
    lab = label_event(event(-1), bars([(98.9, 99.0, 98.8, 98.9)]), CFG)        # below the fade target 99.0
    assert (lab.fade.reason, lab.fade.label) == ("void_target", None)
    assert (lab.follow.reason, lab.follow.label) == ("void_stop", None)       # long with E below its stop 99.4


def test_entry_skips_no_trade_bars_but_time_counts_them():
    rows = [(99.8, 99.8, 99.8, 99.8, 0.0), (99.85, 99.9, 99.8, 99.85), (99.85, 99.9, 98.9, 99.0)]
    lab = label_event(event(-1, h_base=3), bars(rows), CFG)
    assert lab.entry == 99.85 and lab.entry_ts == T0 + M5         # open of bar 2 (bar 1 is venue_flat)
    assert (lab.fade.label, lab.fade.end_ts) == (1, T0 + 3 * M5)
    assert lab.bars_seen == 3 and not lab.censored


def test_resolution_side_end_time_and_mae_over_the_whole_window():
    rows = [(99.9, 100.2, 99.5, 99.6), (99.6, 99.7, 98.95, 99.1), (99.1, 99.3, 97.0, 97.5), (97.5, 98.0, 97.2, 97.8)]
    lab = label_event(event(-1, h_base=4), bars(rows), CFG)
    assert (lab.res, lab.res_side, lab.res_end_ts) == (1, -1, T0 + 2 * M5)
    assert lab.mae_atr == pytest.approx(3.0)            # 100 - 97.0, after the resolution bar
    assert lab.fade.end_ts == T0 + 2 * M5 and lab.fade.label == 1


def test_censored_when_data_ends_before_the_horizon():
    rows = [(99.9, 100.0, 99.5, 99.6), (99.6, 100.7, 99.5, 100.5)]
    lab = label_event(event(-1, h_base=10), bars(rows), CFG)
    assert lab.censored and lab.mae_atr is None and lab.mae_end_ts is None
    assert (lab.fade.label, lab.fade.reason) == (0, "stop")         # decided before the data ended: kept
    assert (lab.follow.label, lab.follow.reason) == (None, "censored")
    assert lab.res is None                                          # band [99, 101] not left yet
    rows[1] = (99.6, 101.1, 99.5, 101.0)
    lab = label_event(event(-1, h_base=10), bars(rows), CFG)
    assert lab.censored and lab.res == 1 and lab.follow.label == 1


def test_horizon_in_base_bars_scales_with_the_level_timeframe():
    assert {tf: base_bars_per_tau(tf) for tf in ("15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M")} == \
        {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "1d": 288, "1w": 2016, "1M": 8766}
    assert horizon_base_bars("15m", CFG) == 18 and horizon_base_bars("1d", CFG) == 1728
    assert horizon_base_bars("1M", CFG) == 6 * 8766
    with pytest.raises(NotImplementedError):
        base_bars_per_tau("1h", Session("fx_ny17", "America/New_York", "17:00"))


# ---------------------------------------------------------------- stream == function

@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10**6))
def test_label_stream_equals_label_event(seed):
    rnd = random.Random(seed)
    rows, c = [], 100.0
    for _ in range(300):
        o = c + rnd.choice([0, 0, 0, rnd.uniform(-0.6, 0.6)])
        c = o + rnd.uniform(-0.5, 0.5)
        if rnd.random() < 0.03:
            rows.append((o, o, o, o, 0.0))
            c = o
            continue
        rows.append((o, max(o, c) + rnd.uniform(0, 0.4), min(o, c) - rnd.uniform(0, 0.4), c, 1.0))
    allbars = bars(rows)
    evs = []
    for k in sorted(rnd.sample(range(len(allbars)), 25)):
        b = allbars[k]
        evs.append(event(rnd.choice([-1, 1]), round(b.c + rnd.uniform(-0.5, 0.5), 2), rnd.uniform(0.3, 1.5),
                         rnd.choice([3, 18, 40]), b.ts_close, len(evs)))
    ls, got, i = LabelStream(CFG), {}, 0
    for b in allbars:
        for x in ls.update(b):
            got[x.level_id] = x
        while i < len(evs) and evs[i].t0 == b.ts_close:
            ls.add(evs[i])
            i += 1
    for x in ls.finish():
        got[x.level_id] = x
    for e in evs:
        want = label_event(e, [b for b in allbars if b.ts_close > e.t0], CFG)
        assert got[e.level_id] == want
