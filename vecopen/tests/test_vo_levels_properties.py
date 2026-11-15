"""Phase 2 property tests (hypothesis) for the streaming level layer.

* The detector equals an independent, non-streaming oracle that evaluates the SPEC section 2 formulas per candidate
  (k* by direct search) and then applies the one-level-per-leg rule greedily in confirmation order.
* Registry invariants, checked after every single bar against a brute-force model kept in the test:
  a level never exists before its confirmation close; its zone never changes after birth; a first fill fires at a
  base bar exactly when that bar is the first one after birth to intersect the zone of a live untouched level (so the
  heap index equals a full scan); confluence equals a full scan; flips and expiry follow the state machine.
* Truncation and future-perturbation invariance of all outputs (the streaming contract in action).
* The same invariants on a slice of real BTCUSDT research bars.

The oracles live here, in the tests: they take whole sequences, which src/ may never do (standing rule 2).
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from vecopen.bars import UTC_24H, resample
from vecopen.legs import ATR, Bar, LegOriginDetector, is_no_trade
from vecopen.levels import FirstFill, LevelBorn, LevelExpired, LevelRegistry, PolarityFlip, bar_stream

from vecopen.config import load_config

BASE_CFG = load_config().data
SMALL_CFG = copy.deepcopy(BASE_CFG)   # more events per example: short ATR, short window, low threshold, wide zones,
SMALL_CFG["legs"].update(atr_len=3, n_confirm=3, theta_L_atr=1.0, volume_baseline_sessions=2)   # short expiry
SMALL_CFG["levels"].update(a_max_bars=6, kappa_atr=0.5)
CONFIGS = [BASE_CFG, SMALL_CFG]
M5, M15 = 300_000, 900_000
T0 = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1000)
PROP = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


# ---------------------------------------------------------------- random bars on a 0.25 grid

wick = st.one_of(st.just(0), st.integers(0, 6), st.integers(0, 40))   # zero often; rare spikes reach far zones
step = st.tuples(
    st.integers(-10, 10),                                        # body, quarters
    wick,                                                        # upper wick
    wick,                                                        # lower wick
    st.sampled_from([0] * 10 + [-3, 3, -12, 12]),                # gap from the previous close (jumps over zones)
    st.integers(0, 40),                                          # volume (0 with a flat bar = no-trade)
    st.sampled_from([0] * 6 + [1, 2, 3]),                        # retest: extend a wick exactly to an open 1-3 bars back
)


def make_rows(steps, start=400.0):
    """Bars on a 0.25 grid. ``retest`` puts the high (or low) exactly on an earlier open, so the strict
    trade-back condition and closed zone edges are hit with equality."""
    rows, c = [], start
    for body, uw, lw, gap, v, retest in steps:
        o = c + gap * 0.25
        if v == 0:                                               # a venue_flat-style bar: zero range, zero volume
            rows.append((o, o, o, o, 0.0))
            c = o
            continue
        c = o + body * 0.25
        h, l = max(o, c) + uw * 0.25, min(o, c) - lw * 0.25  # noqa: E741
        if retest and len(rows) >= retest:
            x = rows[-retest][0]
            h, l = (max(h, x), l) if x >= o else (h, min(l, x))  # noqa: E741
        rows.append((o, h, l, c, float(v)))
    return rows


# ---------------------------------------------------------------- detector oracle

def oracle_legs(rows, cfg):
    """SPEC section 2 by direct evaluation: returns [(direction, origin_index, confirm_index)] over trade bars."""
    lg = cfg["legs"]
    theta, n, wmax, alen = lg["theta_L_atr"], lg["n_confirm"], lg["omega_max"], lg["atr_len"]
    tr_bars = [r for r in rows if not (r[1] == r[2] and r[4] == 0)]
    atr = ATR(alen)
    atr_prev = []
    for o, h, l, c, v in tr_bars:  # noqa: E741
        atr_prev.append(atr.value)
        atr.update(h, l, c)
    cands = []                                                   # (confirm index c, origin t, s)
    for t, (o, h, l, c, v) in enumerate(tr_bars):  # noqa: E741
        s = (c > o) - (c < o)
        a = atr_prev[t]
        if not s or a is None or a <= 0:
            continue
        w = (h - o) / (h - l) if s < 0 else (o - l) / (h - l)
        if w > wmax:
            continue
        for k in range(0, n + 1):
            if t + k >= len(tr_bars):
                break
            j = t + k
            if k >= 1:
                x = tr_bars[j][1] if s < 0 else tr_bars[j][2]
                if not s * (x - o) > 0:
                    break
            if s * (tr_bars[j][3] - o) >= theta * a:
                cands.append((j, t, s))
                break
    out, lock = [], {-1: -1, 1: -1}
    for j, t, s in sorted(cands):                                # confirmation order, earliest origin first
        if t > lock[s]:
            out.append((s, t, j))
            lock[s] = j
    return sorted(out, key=lambda x: (x[2], x[0]))


@PROP
@given(steps=st.lists(step, min_size=5, max_size=160), cfg=st.sampled_from(CONFIGS))
def test_detector_equals_the_direct_oracle(steps, cfg):
    rows = make_rows(steps)
    det = LegOriginDetector("15m", cfg)
    got = []
    trade_index = {}
    k = 0
    for i, (o, h, l, c, v) in enumerate(rows):  # noqa: E741
        b = Bar("15m", T0 + (i + 1) * M15, o, h, l, c, v, T0 + i * M15)
        if not is_no_trade(b):
            trade_index[k] = b.ts_close
            k += 1
        for leg in det.update(b):
            assert leg.birth_ts == b.ts_close                    # born at the close of the bar that confirmed it
            assert leg.birth_ts >= leg.origin_ts_close
            assert leg.k_star == leg.confirm_index - leg.origin_index <= cfg["legs"]["n_confirm"]
            got.append((leg.direction, leg.origin_index, leg.confirm_index))
    assert sorted(got, key=lambda x: (x[2], x[0])) == oracle_legs(rows, cfg)


# ---------------------------------------------------------------- registry brute-force model

class Model:
    """Brute-force state machine fed alongside the registry; checks every output after every bar."""

    def __init__(self, reg: LevelRegistry, cfg):
        self.reg, self.kappa, self.amax = reg, cfg["levels"]["kappa_atr"], cfg["levels"]["a_max_bars"]
        self.w = cfg["levels"]["timeframe_weights"]
        self.lv: dict[int, dict] = {}
        self.count = dict.fromkeys(reg.timeframes, 0)
        self.last_close: dict[str, float] = {}
        self.n_fills = 0

    def live(self):
        return {i: x for i, x in self.lv.items() if x["state"] != "expired"}

    def step(self, b: Bar, events) -> None:
        t = b.ts_close
        ev_born = [e for e in events if isinstance(e, LevelBorn)]
        ev_fill = [e for e in events if isinstance(e, FirstFill)]
        ev_flip = [e for e in events if isinstance(e, PolarityFlip)]
        ev_exp = [e for e in events if isinstance(e, LevelExpired)]
        exp_expired, exp_flips = [], []
        if b.tf in self.count and not is_no_trade(b):
            self.count[b.tf] += 1
            self.last_close[b.tf] = b.c
            for i, x in self.live().items():
                if x["tf"] != b.tf:
                    continue
                if self.count[b.tf] - x["birth_count"] >= self.amax:
                    exp_expired.append((i, x["state"]))
                    x["state"] = "expired"
                elif x["state"] == "reference":
                    if x["pol"] == "resistance" and b.c > x["zone"][1]:
                        x["pol"] = "support"
                        exp_flips.append((i, "support"))
                    elif x["pol"] == "support" and b.c < x["zone"][0]:
                        x["pol"] = "resistance"
                        exp_flips.append((i, "resistance"))
        assert sorted((e.level_id, e.state_before) for e in ev_exp) == sorted(exp_expired)
        assert all(e.ts == t and e.age == self.amax for e in ev_exp)
        assert sorted((e.level_id, e.polarity) for e in ev_flip) == sorted(exp_flips)
        for e in ev_born:
            assert e.ts == t and e.tf == b.tf                     # born at the confirmation close, on its own tf
            assert e.origin_ts_close <= t and (e.k_star > 0 or e.origin_ts_close == t)
            assert e.zone_hi - e.zone_lo == pytest.approx(2 * self.kappa * e.atr, rel=1e-9, abs=1e-12)
            assert e.zone_lo <= e.price <= e.zone_hi
            self.lv[e.level_id] = {"tf": e.tf, "zone": (e.zone_lo, e.zone_hi), "atr": e.atr, "birth": t,
                                   "birth_count": self.count[e.tf], "state": "untouched", "pol": None}
        if b.tf == self.reg.base_tf:
            hit = set()
            if not is_no_trade(b):
                hit = {i for i, x in self.live().items() if x["state"] == "untouched" and x["birth"] < t
                       and b.l <= x["zone"][1] and b.h >= x["zone"][0]}
            assert {e.level_id for e in ev_fill} == hit           # heap index == full scan; first fill only
            live = self.live()
            for e in ev_fill:
                lo, hi = self.lv[e.level_id]["zone"]
                assert (e.zone_lo, e.zone_hi, e.t0) == (lo, hi, t)
                others = sorted(i for i, x in live.items() if i != e.level_id and x["zone"][0] <= hi and lo <= x["zone"][1])
                assert e.overlapping == tuple(others)
                assert e.confluence == pytest.approx(sum(self.w[live[i]["tf"]] for i in others))
                assert e.age == self.count[e.tf] - self.lv[e.level_id]["birth_count"]
                last = self.last_close[e.tf]
                fallback = "resistance" if e.direction < 0 else "support"
                assert e.polarity == ("resistance" if last < lo else "support" if last > hi else fallback)
            for e in ev_fill:
                self.lv[e.level_id].update(state="reference", pol=e.polarity)
            self.n_fills += len(ev_fill)
        else:
            assert ev_fill == []
        # after every bar: nothing exists before its birth, and every zone equals the zone at birth
        assert set(self.reg.active) == set(self.live())
        for i, lvl in self.reg.active.items():
            assert lvl.birth_ts <= t
            assert (lvl.zone_lo, lvl.zone_hi, lvl.atr) == (*self.lv[i]["zone"], self.lv[i]["atr"])
            assert lvl.state == self.lv[i]["state"]


def frame5(rows) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_close": pl.Series([T0 + (i + 1) * M5 for i in range(len(rows))]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "o": [r[0] for r in rows], "h": [r[1] for r in rows], "l": [r[2] for r in rows], "c": [r[3] for r in rows],
        "v": [r[4] for r in rows],
    })


def run_registry(df5: pl.DataFrame, cfg, tfs=("15m", "1h"), check=True):
    higher = {tf: resample(df5, tf, UTC_24H) for tf in tfs}
    reg = LevelRegistry(cfg, timeframes=list(tfs), session=UTC_24H)
    model = Model(reg, cfg) if check else None
    log = []
    for b in bar_stream(df5, higher):
        ev = reg.update(b)
        if model is not None:
            model.step(b, ev)
        log.append((b.tf, b.ts_close, ev))
    return log, model


@PROP
@given(steps=st.lists(step, min_size=30, max_size=700), cfg=st.sampled_from(CONFIGS))
def test_registry_invariants_against_brute_force(steps, cfg):
    run_registry(frame5(make_rows(steps)), cfg)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(steps=st.lists(step, min_size=60, max_size=500), cut=st.floats(0.1, 0.9), factor=st.floats(0.5, 2.0),
       cfg=st.sampled_from(CONFIGS))
def test_truncation_and_future_perturbation_invariance(steps, cut, factor, cfg):
    rows = make_rows(steps)
    k = int(len(rows) * cut)
    t_cut = T0 + k * M5                                          # close of the last kept base bar
    full, _ = run_registry(frame5(rows), cfg, check=False)
    trunc, _ = run_registry(frame5(rows[:k]), cfg, check=False)
    pert = rows[:k] + [(o * factor, h * factor, l * factor, c * factor, v) for o, h, l, c, v in rows[k:]]
    perturbed, _ = run_registry(frame5(pert), cfg, check=False)

    def upto(log):
        return [x for x in log if x[1] <= t_cut]

    assert upto(full) == upto(trunc) == upto(perturbed)


def test_property_generators_produce_events():
    """Non-vacuity: the random streams do produce births, fills, flips and expiries for the checks above."""
    import random

    rnd = random.Random(BASE_CFG["seed"])
    steps = [(rnd.randint(-10, 10), rnd.choice([0, 0, rnd.randint(0, 6)]), rnd.choice([0, 0, rnd.randint(0, 6)]),
              0, rnd.randint(1, 40), 0) for _ in range(3000)]
    log, model = run_registry(frame5(make_rows(steps)), SMALL_CFG)
    kinds = {e.kind for _, _, ev in log for e in ev}
    assert kinds == {"birth", "first_fill", "flip", "expire"}
    assert model.n_fills > 10


def test_real_btcusdt_slice_invariants():
    from vecopen.data import load_bars, research_path

    if not research_path("BTCUSDT").exists():
        pytest.skip("research data not present")
    df = load_bars("BTCUSDT")
    # 45 days that contain the 2021-03-02 venue_flat run (11 flat 5m bars)
    lo, hi = datetime(2021, 2, 1, tzinfo=UTC), datetime(2021, 3, 18, tzinfo=UTC)
    sl = df.filter((pl.col("ts_close") > lo) & (pl.col("ts_close") <= hi)).select("ts_close", "o", "h", "l", "c", "v")
    log, model = run_registry(sl, BASE_CFG, tfs=("15m", "30m", "1h", "2h", "4h", "1d"))
    kinds = [e.kind for _, _, ev in log for e in ev]
    assert kinds.count("birth") > 50 and kinds.count("first_fill") > 30
    assert model.reg.n_no_trade["5m"] >= 11
