from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import evaluate as ev
from ai_trader.research.calendar import add_calendar, gap_fill_baseline, weekend_context, weekend_table
from ai_trader.research.frame import enrich_bars
from ai_trader.research.regimes import cell_name, label
from ai_trader.strategies.base import REGISTRY, assert_causal


def hourly_frame(n: int = 24 * 120, start: datetime = datetime(2025, 1, 6, tzinfo=UTC), seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ev_ts = [start + timedelta(hours=i) for i in range(n)]
    ny_weekend = add_calendar(pl.DataFrame({"event_ts": ev_ts}).with_columns(pl.col("event_ts").dt.cast_time_unit("ms")))["tradfi_weekend"].to_numpy()
    vol = np.where(ny_weekend, 0.004, 0.008)
    drift = np.where((np.arange(n) // 500) % 3 == 0, 0.0008, 0.0)  # trend segments
    r = drift + vol * rng.standard_t(5, n) / np.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    turnover = np.where(ny_weekend, 0.6, 1.0) * rng.lognormal(0, 0.3, n) * 1e6
    bars = pl.DataFrame({
        "event_ts": ev_ts,
        "close_ts": [t + timedelta(hours=1) for t in ev_ts],
        "known_ts": [t + timedelta(hours=1, seconds=2) for t in ev_ts],
        "open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
        "close": close, "volume": turnover / close, "turnover": turnover,
    }).with_columns(pl.col(["event_ts", "close_ts", "known_ts"]).dt.cast_time_unit("ms"))
    return enrich_bars(bars)


@pytest.fixture(scope="module")
def frame():
    return hourly_frame()


def _ts(s):
    return pl.DataFrame({"event_ts": [datetime.fromisoformat(s)]}).with_columns(pl.col("event_ts").dt.cast_time_unit("ms"))


def test_weekend_window_follows_new_york_dst():
    # Winter (EST, UTC-5): Fri 17:00 ET = 22:00 UTC. Summer (EDT, UTC-4): 21:00 UTC.
    assert not add_calendar(_ts("2025-01-10T21:00:00+00:00"))["tradfi_weekend"][0]
    assert add_calendar(_ts("2025-01-10T22:00:00+00:00"))["tradfi_weekend"][0]
    assert add_calendar(_ts("2025-07-11T21:00:00+00:00"))["tradfi_weekend"][0]
    # Sunday 18:00 ET reopen: 23:00 UTC in winter, 22:00 UTC in summer.
    assert add_calendar(_ts("2025-01-12T22:00:00+00:00"))["tradfi_weekend"][0]
    assert not add_calendar(_ts("2025-01-12T23:00:00+00:00"))["tradfi_weekend"][0]
    assert not add_calendar(_ts("2025-07-13T22:00:00+00:00"))["tradfi_weekend"][0]
    # NYSE cash session and weekend id.
    assert add_calendar(_ts("2025-01-13T14:30:00+00:00"))["nyse_open"][0]
    assert add_calendar(_ts("2025-01-11T12:00:00+00:00"))["weekend_id"][0] == "2025-01-10"


def test_weekend_table_and_context_are_causal(frame):
    wt = weekend_table(frame)
    assert wt.height >= 15
    # Fri 17:00 -> Sun 18:00 ET is 49 hours, except DST-change weekends (48 in March, 50 in November).
    assert set(wt["bars"].to_list()) <= {48, 49, 50} and (wt["bars"] == 49).mean() > 0.9
    assert wt["rv_ratio"].median() < 1.0, "synthetic weekends have half the vol"
    full = weekend_context(frame)
    k = 24 * 60 + 7
    part = weekend_context(frame.head(k))
    cols = ["ctx_wk_z", "ctx_fri_close", "wk_first_bar", "wk_absorbed"]
    assert full.head(k).select(cols).equals(part.select(cols))


def test_regime_labels_are_causal_and_complete(frame):
    full = label(frame)
    part = label(frame.head(2000))
    assert full["cell"].head(2000).to_list() == part["cell"].to_list()
    assert set(full["cell"].drop_nulls().unique()) <= {f"R{i}" for i in range(1, 10)}
    assert cell_name("up", "low") == "R1" and cell_name("down", "high") == "R9" and cell_name("range", "normal") == "R5"


@pytest.mark.parametrize("name", ["weekend_shock_follow", "weekend_drift_fade", "cme_gap_fill"])
def test_calendar_strategies_are_causal_and_flat_on_weekends(frame, name):
    s = REGISTRY[name]()
    assert_causal(s, frame)
    sig = s(frame)
    wk = add_calendar(frame)["tradfi_weekend"].to_numpy()
    assert np.all(sig.to_numpy()[wk] == 0.0), "positions only in the post-weekend absorption window"
    ctx = weekend_context(frame)
    assert not ctx.filter(pl.col("wk_first_bar"))["wk_absorbed"].any(), "the Sunday reopen bar opens the absorption window"
    if name == "cme_gap_fill":
        assert (sig.to_numpy() != 0).sum() > 0, "gap fill must trade on a sample with many weekends"


def test_pnl_costs_and_alignment(frame):
    sig = pl.Series(np.r_[np.zeros(100), np.ones(frame.height - 100)])
    p = ev.strategy_pnl(frame, sig, cost_bps=10.0)
    assert p["cost"][100] > 0 and p["cost"][101] == 0, "costs only when the position changes"
    expected = frame["fwd_ret"][150] / frame["sigma_next"][100]  # sized at entry (bar 100), held
    assert abs(p["gross"][150] - expected) < 1e-12, "signal at t earns the next bar's return at the entry size"


def test_newey_west_and_episodes():
    rng = np.random.default_rng(0)
    m, t, n = ev.newey_west_t(rng.normal(0, 1, 5000))
    assert abs(t) < 3.5 and n == 5000
    m, t, _ = ev.newey_west_t(rng.normal(0.1, 1, 5000))
    assert t > 4
    assert ev.episodes(pl.Series(["R1", "R1", "R2", None, "R1", "R1"])) == {"R1": 2, "R2": 1}


def test_gap_fill_baseline():
    assert gap_fill_baseline(np.array([0.0]), np.array([10]))[0] == pytest.approx(1.0)
    p = gap_fill_baseline(np.array([1.0, 3.0]), np.array([9, 9]))
    assert p[0] > p[1] and 0 < p[1] < 1


def test_hypothesis_registry_parses_and_is_complete():
    import yaml

    from ai_trader.core.config import load

    reg = yaml.safe_load((load().root / "research" / "hypotheses.yaml").read_text(encoding="utf-8"))["hypotheses"]
    for hid, h in reg.items():
        missing = {"title", "statement", "test", "success", "prior", "status"} - set(h)
        assert not missing, f"{hid} missing {missing}"
        assert 0 < h["prior"] < 1
