"""Cross-venue study (H-ARB1 / H-ARB2): grid alignment, gaps, episodes, trade simulation, costs, schedule, split."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from ai_trader.research import xvenue as X


def _grid(n: int, bid_b, ask_b, bid_c, ask_c, t0: int = 1_794_000_000, age: float = 0.0) -> pl.DataFrame:
    t = np.arange(t0, t0 + n)
    b = pl.DataFrame({"t": t, "bid_b": np.broadcast_to(bid_b, n).astype(float), "ask_b": np.broadcast_to(ask_b, n).astype(float), "age_b": np.full(n, age)})
    c = pl.DataFrame({"t": t, "bid_c": np.broadcast_to(bid_c, n).astype(float), "ask_c": np.broadcast_to(ask_c, n).astype(float), "age_c": np.full(n, age)})
    return X.build_grid(b, c, 0.0, 1.0, 60)


def test_ticks_to_grid_is_as_of_without_lookahead():
    ev = np.array([1000_500, 1001_000, 1003_200])  # ms
    px = np.array([1.0, 2.0, 3.0])
    p, age = X.ticks_to_grid(ev, px, np.array([1000, 1001, 1002, 1003, 1004]))
    assert math.isnan(p[0]) and age[0] == np.inf          # nothing known at 1000 s (first tick at 1000.5 s)
    assert list(p[1:]) == [2.0, 2.0, 2.0, 3.0]            # tick at exactly 1001.000 s is known at 1001
    assert age[3] == pytest.approx(2.0)
    p2, _ = X.ticks_to_grid(ev, px, np.array([1000]), prev=(0.5, 990.0))
    assert p2[0] == 0.5                                    # previous day's last tick seeds the start


def test_gaps_and_executable_direction():
    # Bybit rich by ~20 bp: sell Bybit at 100.20, buy CFD at 100.00
    g = _grid(10, 100.20, 100.21, 99.99, 100.00)
    assert g["gap_sb_bp"][0] == pytest.approx(math.log(100.20 / 100.00) * 1e4)
    e, d = X.exec_gaps(g, fees_rt_bp=10.0)
    assert (d == 1).all() and e[0] == pytest.approx(math.log(1.002) * 1e4 - 10.0)
    _, d2 = X.exec_gaps(g, fees_rt_bp=25.0)
    assert (d2 == 0).all()


def test_episodes_and_arb1_criterion():
    t = np.arange(20)
    d = np.array([0, 1, 1, 1, 0, -1, -1, 1, 0, 0] + [1] * 10)
    eps = X.episodes(t, d)
    assert eps == [(1, 3, 1), (5, 6, -1), (7, 7, 1), (10, 19, 1)]
    t2 = t.copy()
    t2[15:] += 5                                            # a hole in the grid splits a run
    assert len(X.episodes(t2, d)) == 5

    n = 1000
    bid_b = np.full(n, 100.0)
    bid_b[100:110] = 100.5                                   # one 10 s episode = 1% of seconds
    g = _grid(n, bid_b, bid_b + 0.01, 99.99, 100.0)
    r = X.arb1_day(g, fees_rt_bp=10.0, min_len_s=4)
    assert r["n_long_episodes"] == 1 and r["long_share"] == pytest.approx(0.01) and r["pass"]
    assert not X.arb1_day(g, fees_rt_bp=10.0, min_len_s=12)["pass"]


def test_arb2_convergence_trade_pnl():
    n = 400
    bid_b = np.full(n, 100.00)
    bid_b[10:200] = 100.50                                   # Bybit rich for 190 s, then converges
    g = _grid(n, bid_b, bid_b + 0.01, 99.99, 100.00)
    tr = X.arb2_day(g, fees_rt_bp=10.0, delay_s=2, carry=X.Carry())
    assert len(tr) == 1
    x = tr[0]
    assert x["side"] == 1 and x["exit"] == "halved"
    assert x["t_in"] == x["t_signal"] + 2
    entry = math.log(100.50 / 100.00) * 1e4                  # sold Bybit bid 100.50, bought CFD ask 100.00
    exit_ = math.log(99.99 / 100.01) * 1e4                   # bought Bybit ask 100.01, sold CFD bid 99.99
    assert x["gross_bp"] == pytest.approx(entry + exit_)
    assert x["net_bp"] == pytest.approx(entry + exit_ - 10.0)

    # a persistent (structural) gap never halves: time exit after 60 minutes
    m = 4000
    g2 = _grid(m, 100.5, 100.51, 99.99, 100.0)
    tr2 = X.arb2_day(g2, fees_rt_bp=10.0, delay_s=1, carry=X.Carry())
    assert tr2[0]["exit"] == "time" and tr2[0]["hold_s"] == pytest.approx(3600, abs=2)


def test_carry_funding_and_swap():
    t_in = int(datetime(2026, 11, 10, 7, 0, tzinfo=UTC).timestamp())        # Tuesday
    t_out = int(datetime(2026, 11, 11, 9, 0, tzinfo=UTC).timestamp())
    c = X.Carry(funding=[(t_in + 3600, 0.0001), (t_out + 1, 0.01)], swap_long_bp=-0.5, swap_short_bp=0.1, swap_time_s=22 * 3600, triple_weekday=None)
    f, s = X.carry_bp(c, +1, t_in, t_out)                   # short perp receives +1 bp; long CFD pays one night
    assert f == pytest.approx(1.0) and s == pytest.approx(-0.5)
    f, s = X.carry_bp(c, -1, t_in, t_out)
    assert f == pytest.approx(-1.0) and s == pytest.approx(0.1)
    c.triple_weekday = 1                                    # Tuesday night counts three times
    assert X.carry_bp(c, +1, t_in, t_out)[1] == pytest.approx(-1.5)


def test_schedule_mask_wall_clock_and_holiday():
    # Mon-Fri 09:30-16:00 New York; 2026-11-10 is a Tuesday (EST, UTC-5)
    sched = [[d * 86400 + 34200, d * 86400 + 57600] for d in range(1, 6)]
    t = np.array([int(datetime(2026, 11, 10, h, m, tzinfo=UTC).timestamp()) for h, m in ((14, 29), (14, 30), (20, 59), (21, 0))])
    assert X.in_schedule(t, sched, "America/New_York").tolist() == [False, True, True, False]
    hol = [{"date_days": (datetime(2026, 11, 10).date() - datetime(1970, 1, 1).date()).days, "recurring": False, "tz": "America/New_York"}]
    assert not X.in_schedule(t, sched, "America/New_York", hol).any()
    assert X.in_schedule(t, None, None).all()


def test_commission_and_swap_conversion():
    assert X.ctrader_commission_bp({"commission_type": "USD_PER_MILLION_USD", "commission_rate": 30.0}, 100.0) == pytest.approx(0.3)
    assert X.ctrader_commission_bp({"commission_type": "PERCENTAGE_OF_VALUE", "commission_rate": 0.01}, 100.0) == pytest.approx(1.0)
    # 0.02 USD per lot of 1 share at 200 USD = 1 bp
    assert X.ctrader_commission_bp({"commission_type": "USD_PER_LOT", "commission_rate": 0.02, "lot_size_cents": 100}, 200.0) == pytest.approx(1.0)
    assert X.ctrader_commission_bp({"commission_type": "USD_PER_LOT", "commission_rate": 0.0, "lot_size_cents": 100, "precise_min_commission": 1.0}, 200.0) == pytest.approx(1.0)
    assert X.ctrader_commission_bp(None, 1.0) is None
    lo, sh = X.swap_bp_per_night({"swap_long": -7.3, "swap_short": 3.65, "swap_calculation_type": "PERCENTAGE"}, 100.0)
    assert lo == pytest.approx(-2.0) and sh == pytest.approx(1.0)
    lo, _ = X.swap_bp_per_night({"swap_long": -10, "swap_short": 0, "swap_calculation_type": "PIPS", "pip_position": 2}, 100.0)
    assert lo == pytest.approx(-10.0)                       # 10 pips of 0.01 on 100 = 10 bp


def test_cluster_bootstrap():
    x = np.array([1.0, 1.0, 1.0, -1.0, 3.0])
    r = X.cluster_boot_mean(x, np.array([0, 0, 1, 1, 2]), n_boot=500)
    assert r["mean"] == pytest.approx(1.0) and r["clusters"] == 3 and r["lo"] <= 1.0 <= r["hi"]


def test_split_is_seeded_and_respects_pool_rules():
    insts = [{"bybit": "NVDAUSDT", "class": "stock"}, {"bybit": "BTCUSDT", "class": "crypto"}, {"bybit": "XAUUSDT", "class": "metal"}]
    launch = {"NVDAUSDT": "2026-05-06"}
    a = X.draw_split(insts, "2026-08-31", "2026-10-30", launch, seed=7)
    b = X.draw_split(insts, "2026-08-31", "2026-10-30", launch, seed=7)
    assert a == b and all(len(v["days"]) == 5 for v in a.values())
    assert all(datetime.fromisoformat(d).weekday() < 5 for d in a["NVDAUSDT"]["days"])
    assert "2026-09-07" not in X.day_pool(insts[0], "2026-08-31", "2026-10-30", launch)     # Labor Day
    assert len(X.day_pool(insts[1], "2026-08-31", "2026-09-06", {})) == 7                   # crypto: every day
    cached = {"XAUUSDT": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-08", "2026-09-09"]}
    c = X.draw_split(insts, "2026-08-31", "2026-10-30", launch, seed=7, cached=cached)
    assert set(c["XAUUSDT"]["days"]) <= set(cached["XAUUSDT"]) and c["XAUUSDT"]["pool_from_cache"]


def test_config_never_maps_spy():
    c = X.cfg()
    assert "SPYUSDT" not in [i["bybit"] for i in c["instruments"]]
    assert len(X.study_instruments(c)) >= 5
