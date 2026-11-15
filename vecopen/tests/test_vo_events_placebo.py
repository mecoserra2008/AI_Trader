"""Phase 3: touch events, placebo generation, the identical registry code path, matching and balance.

* ``TouchDetector`` on the phase 2 hand case (LEG): every field of the touch event by hand.
* The whole event step (treated + 3 placebo families + generator) is invariant to truncation and to a perturbation
  of every price after the cut (the streaming contract, with placebos and the seeded draws included).
* Identical code path: re-registering the treated registry's own legs into a placebo-type registry (family tag only)
  reproduces the treated event stream exactly, on random streams and on real BTCUSDT bars; every placebo birth of the
  phase 3 pipeline goes through ``LevelRegistry._birth`` of a plain ``LevelRegistry``.
* Family 1, 2 and 3 generation by hand; matching and SMD by hand.
* No forward-looking module is imported by the event, placebo or level modules.
"""

from __future__ import annotations

import ast
import copy
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_vo_levels_cases import LEG, LEG_BASE, M5, WARM, T, bar, feed15
from test_vo_levels_properties import BASE_CFG, SMALL_CFG, frame5, make_rows, step
from vecopen import phase3
from vecopen.bars import UTC_24H, resample
from vecopen.events import TouchDetector, side_at_birth
from vecopen.levels import LevelRegistry, bar_stream
from vecopen.placebo import (
    FAMILIES,
    P1,
    P2,
    P3,
    Moments,
    PlaceboGenerator,
    match_events,
    rng_for,
    smd,
    smd_moments,
)

CFG = BASE_CFG
SRC = Path(__file__).resolve().parents[1] / "src" / "vecopen"
PROP = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


# ---------------------------------------------------------------- TouchDetector by hand

class Rec:
    """Feeds a TouchDetector and records base closes (for the sigma check)."""

    def __init__(self, td):
        self.td, self.closes, self.out = td, [], []

    def update(self, b):
        ev = self.td.update(b)
        if b.tf == "5m":
            self.closes.append(b.c)
        self.out += ev
        return ev


def rms_last(closes, w):
    d = np.diff(np.asarray(closes, float))[-w:]
    return math.sqrt(float((d * d).sum()) / w)


def test_touch_event_fields_by_hand():
    td = TouchDetector(LevelRegistry(CFG, timeframes=["15m"], session=UTC_24H), CFG)
    rec = Rec(td)
    for i, row in enumerate(LEG):
        feed15(rec, i, row, LEG_BASE.get(i))
    closes_at_birth = rec.closes[:-1]                  # the base bar closing at T16 comes after the 15m bar
    (info,) = td.live.values()
    assert (info.price, info.zone_lo, info.zone_hi, info.atr, info.g, info.k_star) == (100.0, 99.9, 100.1, 1.0, -1, 2)
    assert (info.ref_close, info.departure_atr, info.regime) == (98.0, 2.0, 0.0)   # ln(ATR_13 / mean TR) = ln(1/1)
    assert info.d_edge == pytest.approx(1.9)
    assert info.h_base == 18 and info.birth_ts == T(16)
    assert info.sigma_birth == pytest.approx(rms_last(closes_at_birth, 18))
    base17 = [(98.0, 99.5, 97.9, 99.4), (99.4, 99.9, 99.3, 99.8), (99.8, 99.85, 99.0, 99.0)]
    feed15(rec, 17, (98.0, 99.9, 97.9, 99.0), base17)
    assert len(rec.out) == 1
    e = rec.out[0]
    assert e.t0 == T(17) - M5 and e.family == "treated" and e.level_id == 0
    assert (e.age, e.polarity, e.confluence, e.approach_side, e.prev_close) == (0, "resistance", 0.0, -1, 99.4)
    assert (e.touch_o, e.touch_h, e.touch_l, e.touch_c) == (99.4, 99.9, 99.3, 99.8)
    closes_before = rec.closes[:rec.closes.index(99.8)]           # up to the bar before the touch bar
    assert e.sigma_t0 == pytest.approx(rms_last(closes_before, 18))
    assert e.level.leg is None and e.level.price == 100.0 and not td.live


def test_side_at_birth():
    assert side_at_birth(98.0, 100.0, 1) == -1 and side_at_birth(102.0, 100.0, -1) == 1
    assert side_at_birth(100.0, 100.0, -1) == -1 and side_at_birth(100.0, 100.0, 1) == 1


# ---------------------------------------------------------------- streaming contract for the whole event step

def run_events(df5, cfg, tfs=("15m", "1h")):
    higher = {tf: resample(df5, tf, UTC_24H) for tf in tfs}
    out = phase3.event_step("TEST", df5, higher, cfg, UTC_24H)
    return out


@PROP
@given(steps=st.lists(step, min_size=60, max_size=500), cut=st.floats(0.1, 0.9), factor=st.floats(0.5, 2.0),
       cfg=st.sampled_from([BASE_CFG, SMALL_CFG]))
def test_event_step_truncation_and_future_perturbation_invariance(steps, cut, factor, cfg):
    rows = make_rows(steps)
    k = int(len(rows) * cut)
    t_cut = frame5(rows)["ts_close"][k - 1]
    t_ms = int(t_cut.timestamp() * 1000)
    pert = rows[:k] + [(o * factor, h * factor, l * factor, c * factor, v) for o, h, l, c, v in rows[k:]]
    runs = [run_events(frame5(r), cfg) for r in (rows, rows[:k], pert)]

    def upto(r):
        ev = [e for e in r["events"] if e.t0 <= t_ms]
        lv = [x for x in r["level_rows"] if x["birth_ts"] <= t_cut]
        return repr(ev), repr(lv)

    assert upto(runs[0]) == upto(runs[1]) == upto(runs[2])


def test_event_step_produces_every_family():
    import random

    rnd = random.Random(CFG["seed"])
    steps = [(rnd.randint(-10, 10), rnd.choice([0, 0, rnd.randint(0, 6)]), rnd.choice([0, 0, rnd.randint(0, 6)]),
              0, rnd.randint(1, 40), 0) for _ in range(3000)]
    r = run_events(frame5(make_rows(steps)), SMALL_CFG)
    fam_lv = {x["family"] for x in r["level_rows"]}
    fam_ev = {e.family for e in r["events"]}
    assert fam_lv == fam_ev == {"treated", *FAMILIES}


# ---------------------------------------------------------------- identical registry code path

def probe_run(df5, cfg, tfs):
    """Treated registry, and a placebo-type registry (family tag only) that receives the treated legs through
    ``register``. Returns the two event logs per bar."""
    higher = {tf: resample(df5, tf, UTC_24H) for tf in tfs}
    tre = LevelRegistry(cfg, timeframes=list(tfs), session=UTC_24H)
    probe = LevelRegistry(cfg, timeframes=list(tfs), session=UTC_24H, family="probe")
    assert type(probe) is type(tre) is LevelRegistry and probe.detectors == {} and not probe.detect
    log_t, log_p = [], []
    for b in bar_stream(df5, higher):
        ev_t = tre.update(b)
        ev_p = probe.update(b)
        for e in ev_t:
            if e.kind == "birth":
                ev_p.append(probe.register(tre.active[e.level_id].leg))
        log_t.append(ev_t)
        log_p.append([replace(e, family="treated") if hasattr(e, "family") else e for e in ev_p])
        assert all(getattr(e, "family", "probe") == "probe" for e in ev_p)
    return log_t, log_p


@PROP
@given(steps=st.lists(step, min_size=30, max_size=700), cfg=st.sampled_from([BASE_CFG, SMALL_CFG]))
def test_registered_levels_follow_the_identical_state_machine(steps, cfg):
    log_t, log_p = probe_run(frame5(make_rows(steps)), cfg, ("15m", "1h"))
    assert log_t == log_p


def test_identical_code_path_non_vacuous_and_on_real_bars():
    import random

    rnd = random.Random(7)
    steps = [(rnd.randint(-10, 10), rnd.choice([0, 0, rnd.randint(0, 6)]), rnd.choice([0, 0, rnd.randint(0, 6)]),
              0, rnd.randint(1, 40), 0) for _ in range(3000)]
    log_t, log_p = probe_run(frame5(make_rows(steps)), SMALL_CFG, ("15m", "1h"))
    assert log_t == log_p
    assert {e.kind for x in log_t for e in x} == {"birth", "first_fill", "flip", "expire"}

    from vecopen.data import load_bars, research_path

    if not research_path("BTCUSDT").exists():
        pytest.skip("research data not present")
    df = load_bars("BTCUSDT")
    lo, hi = datetime(2021, 2, 1, tzinfo=UTC), datetime(2021, 3, 18, tzinfo=UTC)
    sl = df.filter((pl.col("ts_close") > lo) & (pl.col("ts_close") <= hi)).select("ts_close", "o", "h", "l", "c", "v")
    log_t, log_p = probe_run(sl, BASE_CFG, ("15m", "30m", "1h", "2h", "4h", "1d"))
    assert log_t == log_p
    assert sum(e.kind == "first_fill" for x in log_t for e in x) > 30


def test_every_placebo_birth_goes_through_the_same_birth_method(monkeypatch):
    calls: dict[str, int] = {}
    orig = LevelRegistry._birth

    def spy(self, leg, count, last_price):
        calls[self.family] = calls.get(self.family, 0) + 1
        assert type(self) is LevelRegistry
        return orig(self, leg, count, last_price)

    monkeypatch.setattr(LevelRegistry, "_birth", spy)
    import random

    rnd = random.Random(3)
    steps = [(rnd.randint(-10, 10), rnd.choice([0, 0, rnd.randint(0, 6)]), rnd.choice([0, 0, rnd.randint(0, 6)]),
              0, rnd.randint(1, 40), 0) for _ in range(2000)]
    r = run_events(frame5(make_rows(steps)), SMALL_CFG)
    rows = {}
    for x in r["level_rows"]:
        rows[x["family"]] = rows.get(x["family"], 0) + 1
    assert calls == rows and set(calls) == {"treated", *FAMILIES}
    for fam, td in r["detectors"].items():
        assert type(td.registry) is LevelRegistry and td.registry.family == fam
        assert (td.registry.detectors == {}) == (fam != "treated")
    src = ast.unparse(ast.parse((SRC / "levels.py").read_text(encoding="utf-8")))
    assert "self._birth(leg, self._count[tf], self._last_close[tf])" in src      # register() -> _birth


def test_register_guards():
    tre = LevelRegistry(CFG, timeframes=["15m"], session=UTC_24H)
    probe = LevelRegistry(CFG, timeframes=["15m"], session=UTC_24H, family="probe")
    legs = []
    for i, row in enumerate(LEG):
        b = bar("15m", T(i), *row)
        for e in tre.update(b):
            if e.kind == "birth":
                legs.append(tre.active[e.level_id].leg)
        if i < 16:
            probe.update(b)
    leg = legs[0]
    with pytest.raises(ValueError, match="placebo registries"):
        tre.register(leg)
    with pytest.raises(ValueError, match="must be at that instant"):
        probe.register(leg)                              # probe has not processed the 15m bar of T16 yet
    probe.update(bar("15m", T(16), *LEG[16]))
    probe.update(bar("5m", T(16), 98.0, 98.0, 98.0, 98.0))
    with pytest.raises(ValueError, match="must be at that instant"):
        probe.register(leg)                              # the base bar of the same close has been processed
    with pytest.raises(ValueError):
        LevelRegistry(CFG, timeframes=["15m"], family="")


# ---------------------------------------------------------------- the three families by hand

def gen_run(rows, asset="TEST"):
    td = TouchDetector(LevelRegistry(CFG, timeframes=["15m"], session=UTC_24H), CFG)
    gen = PlaceboGenerator(CFG, ["15m"], rng=rng_for(asset, CFG))
    out = []
    for i, row in enumerate(rows):
        b = bar("15m", T(i), *row)
        td.update(b)
        out.append((list(td.new_levels), gen.update(b, td.new_levels)))
    return out


def test_family1_non_origin_open_born_at_its_departure():
    # bar 14: down candle, open 100, wick (100.4 - 100) / 1.0 = 0.4 > 0.25: NOT an origin.
    # bar 15: high 99.9 < 100 (no trade back), close 98.0 = 2.0 ATR_13 below the open of bar 14 -> family-1 birth
    out = gen_run(WARM + [(100.0, 100.4, 99.4, 99.5), (99.5, 99.9, 97.9, 98.0)])
    assert all(not tl for tl, _ in out)                                 # no treated level at all
    assert all(not sp for _, sp in out[:15])
    (sp,) = out[15][1]
    lg = sp.leg
    assert sp.family == P1 and sp.source_level_id is None
    assert (lg.origin_index, lg.origin_o, lg.direction, lg.k_star, lg.confirm_close, lg.atr, lg.leg_size_atr) == \
        (14, 100.0, -1, 1, 98.0, 1.0, 2.0)
    assert lg.birth_ts == T(15) and lg.omega == pytest.approx(0.4)


def test_family1_strict_trade_back_and_origin_exclusion():
    out = gen_run(WARM + [(100.0, 100.4, 99.4, 99.5), (99.5, 100.0, 97.9, 98.0)])    # high = open: dead
    assert all(not sp for _, sp in out)
    out = gen_run(LEG)                     # bar 14 IS the origin (confirmed at 16): no family-1 level for it
    tl16, sp16 = out[16]
    assert [x.leg.origin_index for x in tl16] == [14]
    assert not [s for s in sp16 if s.family == P1]
    assert all(not sp for _, sp in out[:16])


def test_families_2_and_3_by_hand_and_seeded():
    out = gen_run(LEG)
    tl16, sp16 = out[16]
    rng = rng_for("TEST", CFG)
    u1, u2 = rng.uniform(1.0, 3.0), rng.uniform(1.0, 3.0)
    p2 = [s for s in sp16 if s.family == P2]
    assert [(s.variant, s.source_level_id) for s in p2] == [("up", 0), ("down", 0)]
    assert p2[0].leg.origin_o == 100.0 + u1 and p2[1].leg.origin_o == 100.0 - u2
    assert (p2[0].displacement_atr, p2[1].displacement_atr) == (u1, -u2)
    for s in p2:
        assert s.leg.direction == side_at_birth(98.0, s.leg.origin_o, -1)
        assert s.leg.atr == 1.0 and s.leg.birth_ts == T(16) and s.leg.confirm_close == 98.0
        assert s.leg.leg_size_atr == pytest.approx(abs(98.0 - s.leg.origin_o))
    p3 = [s for s in sp16 if s.family == P3]
    assert [(s.variant, s.leg.origin_o, s.leg.direction) for s in p3] == \
        [("close", 99.5, -1), ("mid", 99.625, -1), ("extreme", 99.25, -1)]      # down-leg: extreme = the low
    # same asset -> same draws; another asset -> different draws
    again = [s.leg.origin_o for s in gen_run(LEG)[16][1] if s.family == P2]
    other = [s.leg.origin_o for s in gen_run(LEG, "OTHER")[16][1] if s.family == P2]
    assert again == [p.leg.origin_o for p in p2] and other != again
    assert CFG["stats"]["placebo_displacement_atr"] == [1.0, 3.0]


def test_family3_extreme_of_an_up_leg_is_the_high():
    up = [(100.0, 100.75, 100.0, 100.5), (100.5, 101.25, 100.08, 101.0), (101.0, 102.25, 100.05, 102.0)]
    out = gen_run(WARM + up)
    tl16, sp16 = out[16]
    assert [x.leg.direction for x in tl16] == [1]
    p3 = {s.variant: s.leg.origin_o for s in sp16 if s.family == P3}
    assert p3 == {"close": 100.5, "mid": 100.375, "extreme": 100.75}


# ---------------------------------------------------------------- matching and balance

def test_match_events_by_hand():
    t = {"event_key": ["t1", "t2", "t3", "t4"], "year": [2021, 2021, 2022, 2021],
         "age_log": [1.0, 3.0, 1.0, 1.05], "departure_atr": [2.0, 3.0, 2.0, 2.0], "regime": [0.0, 0.1, 0.0, 0.0]}
    p = {"event_key": ["p1", "p2", "p3", "p4"], "year": [2021, 2021, 2023, 2022],
         "age_log": [1.1, 2.9, 1.0, np.nan], "departure_atr": [2.0, 3.1, 2.0, 2.0], "regime": [0.0, 0.1, 0.0, 0.0]}
    m = match_events({k: np.asarray(v) for k, v in t.items()}, {k: np.asarray(v) for k, v in p.items()})
    assert list(m["placebo_key"]) == ["p1", "p2", None, "p1"]          # exact on year, with replacement
    assert np.isnan(m["distance"][2]) and m["distance"][0] < m["distance"][1]


def test_smd_by_hand_and_pooled_moments():
    assert smd([1, 2, 3], [2, 3, 4]) == pytest.approx(-1.0)
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 500), rng.normal(0.2, 1.5, 300)
    ma, mb = Moments(), Moments()
    ma.add(a[:200])
    ma.add(a[200:])
    mb.merge(len(b), b.sum(), (b * b).sum())
    assert smd_moments(ma, mb) == pytest.approx(smd(a, b))


# ---------------------------------------------------------------- separation of the forward-looking step

def test_event_and_level_modules_never_import_labels():
    for name in ("legs.py", "levels.py", "events.py", "placebo.py", "features.py"):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m and m.endswith("labels") for m in mods), name


def test_real_slice_events_equal_registry_fills_and_every_event_is_labelled():
    from vecopen.data import load_bars, research_path

    if not research_path("ETHUSDT").exists():
        pytest.skip("research data not present")
    df = load_bars("ETHUSDT")
    lo, hi = datetime(2022, 5, 1, tzinfo=UTC), datetime(2022, 6, 15, tzinfo=UTC)
    sl = df.filter((pl.col("ts_close") > lo) & (pl.col("ts_close") <= hi)).select("ts_close", "o", "h", "l", "c", "v")
    tfs = ("15m", "30m", "1h", "2h", "4h", "1d")
    r = run_events(sl, BASE_CFG, tfs)
    higher = {tf: resample(sl, tf, UTC_24H) for tf in tfs}
    plain = LevelRegistry(BASE_CFG, timeframes=list(tfs), session=UTC_24H)
    fills = [(e.t0, e.level_id) for b in bar_stream(sl, higher) for e in plain.update(b) if e.kind == "first_fill"]
    assert [(e.t0, e.level_id) for e in r["events"] if e.family == "treated"] == fills and len(fills) > 50
    assert {e.family for e in r["events"]} == {"treated", *FAMILIES}
    ev = sorted(r["events"], key=lambda e: e.t0)
    labels = list(phase3.label_step(ev, sl, BASE_CFG, "5m"))
    assert len(labels) == len(ev)
    assert {(x.family, x.level_id) for x in labels} == {(e.family, e.level_id) for e in ev}
    assert all(x.entry is not None and x.entry_ts == x.t0 for x in labels)   # entry at the open of the next bar
    cfg = copy.deepcopy(BASE_CFG)
    assert cfg["labels"]["h_tau_bars"] == 6
