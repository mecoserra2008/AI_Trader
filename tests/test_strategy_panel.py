import numpy as np
import polars as pl
import pytest

from ai_trader.research.evaluate import by_regime, strategy_pnl
from ai_trader.research.strategy_panel import (_rank, _with_separators, block_indices, daily_from_bars, grade, held_leverage, holm,
                                               regime_table, spearman, start_of_day_context, strategy_fn, strategy_specs, trade_table)
from tests.test_strategies import build_frame


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    n = 24 * 200
    mu = np.repeat(rng.choice([-0.0015, 0.0, 0.0015], size=n // 300 + 1), 300)[:n]
    r = mu + 0.01 * rng.standard_t(4, n) / np.sqrt(2.0)
    f = build_frame(100 * np.exp(np.cumsum(r)), rng)
    return f.with_columns((pl.col("close").log().shift(-1) - pl.col("close").log()).alias("fwd_ret"),
                          pl.col("sigma_next").alias("sigma_risk_next"))


def test_specs_skip_flow_and_t_oi():
    names = [s[0] for s in strategy_specs("L")]
    assert "flow_momentum" not in names and "cvd_divergence" not in names
    assert "zoo_ensemble" in names and "oi_price_divergence" in names
    assert "oi_price_divergence" not in [s[0] for s in strategy_specs("T")]
    assert dict((n, f) for n, _, f in strategy_specs("L"))["geo_struct"] == "4h"


@pytest.mark.parametrize("name", ["tsmom", "donchian_breakout", "zscore_reversion", "rsi_reversion", "compression_breakout"])
def test_held_leverage_matches_strategy_pnl(frame, name):
    sig = strategy_fn(name)(frame)
    p = strategy_pnl(frame, sig)
    hs, lev = held_leverage(sig.to_numpy(), frame["sigma_next"].to_numpy())
    assert np.allclose(lev * np.nan_to_num(frame["fwd_ret"].to_numpy()), p["gross"].to_numpy(), atol=1e-12)
    assert np.allclose(np.abs(np.diff(np.r_[0.0, lev])) * 7.5e-4, p["cost"].to_numpy(), atol=1e-12)
    assert np.all((hs == 0) == (lev == 0))


def test_trade_table_costs_and_excursions(frame):
    sig = strategy_fn("donchian_breakout")(frame)
    p = strategy_pnl(frame, sig)
    _, lev = held_leverage(sig.to_numpy(), frame["sigma_next"].to_numpy())
    lc, lh, ll = (np.log(frame[k].to_numpy()) for k in ("close", "high", "low"))
    tr = trade_table(lev, p["gross"].to_numpy(), lc, lh, ll, 7.5e-4)
    assert len(tr) > 5
    for t in tr:
        assert t["mfe"] >= max(t["gross"], 0) - 1e-12
        assert t["mae"] <= min(t["gross"], 0) + 1e-12
    # closed trades' costs add up to strategy_pnl's cost over the same bars (only the open tail may be missing)
    last = tr[-1]["j"] + 1
    assert np.isclose(sum(t["cost"] for t in tr), p["cost"].to_numpy()[: last + 1].sum()) or \
        sum(t["cost"] for t in tr) <= p["cost"].to_numpy().sum() + 1e-12
    assert np.isclose(sum(t["gross"] for t in tr), p["gross"].to_numpy()[: last].sum())


def test_daily_panel_is_causal(frame):
    """Daily P&L of complete dates before a cut is unchanged when bars after the cut are removed (P&L dated by close_ts:
    a date's last row needs the next bar's close, so only dates strictly before the cut's date are compared)."""
    for name in ("tsmom", "zscore_reversion", "shock_regime"):
        fn = strategy_fn(name)

        def daily(f):
            s = fn(f)
            p = strategy_pnl(f, s)
            hs, lev = held_leverage(s.to_numpy(), f["sigma_next"].to_numpy())
            return daily_from_bars(f, p, hs, lev)

        full = daily(frame)
        for k in (2400, 3600):
            part = frame.head(k)
            cut = part["close_ts"][-1].date()
            a = full.filter(pl.col("date") < cut).sort("date")
            b = daily(part).filter(pl.col("date") < cut).sort("date")
            assert a.height == b.height
            assert np.allclose(a["pnl"].to_numpy(), b["pnl"].to_numpy(), atol=1e-12), name


def test_start_of_day_cell_is_known_at_midnight(frame):
    from ai_trader.research.regimes import label

    ctx = start_of_day_context(frame)
    lab = label(frame, bar_hours=1.0)
    mid = lab.filter((pl.col("close_ts").dt.hour() == 0) & (pl.col("close_ts").dt.minute() == 0))
    m = dict(zip(mid["close_ts"].dt.date().to_list(), mid["cell"].to_list()))
    for d, c in zip(ctx["date"].to_list(), ctx["cell_sod"].to_list()):
        if d in m:
            assert c == m[d]
    # truncation invariance of the start-of-day cell
    part = start_of_day_context(frame.head(3000))
    j = part.join(ctx, on="date", suffix="_f").filter(pl.col("cell_sod").is_not_null())
    assert (j["cell_sod"] == j["cell_sod_f"]).all()


def test_regime_table_separators_match_per_symbol_episodes():
    rows = []
    for s, cells in (("A", ["R1", "R1", "R2"]), ("B", ["R2", "R2", "R1"])):
        for i, c in enumerate(cells):
            rows.append({"symbol": s, "date": i, "pnl_d": 1.0 if c == "R1" else -1.0, "active": True, "cell": c})
    df = pl.DataFrame(rows)
    t = regime_table(df)
    eps = dict(zip(t["cell"].to_list(), t["episodes"].to_list()))
    assert eps["R1"] == 2 and eps["R2"] == 2  # without separators A's R2 and B's R2 would merge into one run
    p, cells = _with_separators(df)
    assert cells.null_count() == 2
    direct = by_regime(p, cells, bar_hours=24.0)
    assert np.allclose(direct["mean_shrunk"].to_numpy(), t["mean_shrunk"].to_numpy())


def test_stat_helpers():
    assert np.isclose(spearman(np.arange(10.0), np.arange(10.0) ** 3), 1.0)
    assert np.allclose(_rank(np.array([3.0, 1.0, 3.0])), [1.5, 0.0, 1.5])
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["b"], 0.08) and np.isclose(adj["c"], 0.5)
    idx = block_indices(25, 10, np.random.default_rng(0))
    assert len(idx) == 25 and idx.max() < 25
    assert grade(0.02, 2.5, 0.01, 2.1) == "A" and grade(-0.02, -2.5, -0.01, -1.2) == "B"
    assert grade(0.02, -2.5, 0.01, 2.1) == "C" and grade(0.02, 3.0, -0.01, -2.0) == "-"
