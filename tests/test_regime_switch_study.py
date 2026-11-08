from datetime import date, timedelta

import numpy as np
import polars as pl

from ai_trader.features.regime_features import FEATURES
from ai_trader.research import regime_switch_study as R

SUN0 = date(2021, 1, 3)  # a Sunday


def _daily(n_days: int, syms=("AAA", "BBB")) -> pl.DataFrame:
    rows = []
    for s in syms:
        for i in range(n_days):
            d = SUN0 + timedelta(days=i)
            rows.append({"symbol": s, "date": d, "mr": float(i), "trend": float(2 * i)})
    df = pl.DataFrame(rows)
    return R.daily_join(df.select("symbol", "date", "mr"), df.select("symbol", "date", "trend"))


def _panel(n_weeks: int, n_sym: int = 8, seed: int = 0, beta: float = 1.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_weeks):
        for s in range(n_sym):
            x = rng.standard_normal(len(FEATURES))
            diff = beta * x[0] + rng.standard_normal()
            rows.append({"symbol": f"S{s}", "D": SUN0 + timedelta(weeks=w), **dict(zip(FEATURES, x, strict=True)), "diff": diff,
                         "y": int(diff > 0), "er_14d": abs(x[0]), "trend7": diff, "mr7": 0.0, "segment": "llm_in_sample"})
    return pl.DataFrame(rows)


def test_weekly_outcome_is_the_next_seven_days():
    daily = _daily(40)
    # decision Sunday D gets dates D+1..D+7
    assert daily.filter(pl.col("date") == SUN0 + timedelta(days=1))["D"][0] == SUN0
    assert daily.filter(pl.col("date") == SUN0 + timedelta(days=7))["D"][0] == SUN0
    assert daily.filter(pl.col("date") == SUN0 + timedelta(days=8))["D"][0] == SUN0 + timedelta(days=7)
    feats = pl.DataFrame({"symbol": ["AAA"] * 40, "date": [SUN0 + timedelta(days=i) for i in range(40)], **{c: [0.0] * 40 for c in FEATURES}})
    p = R.weekly_panel(feats, daily, cut=SUN0 + timedelta(days=15))
    w0 = p.filter(pl.col("D") == SUN0)
    assert w0["mr7"][0] == sum(range(1, 8)) and w0["trend7"][0] == 2 * sum(range(1, 8))
    # D+8 < cut -> in-sample; D+1 >= cut -> holdout; straddling -> dropped (null)
    seg = dict(zip(p["D"].to_list(), p["segment"].to_list(), strict=True))
    assert seg[SUN0] == "llm_in_sample" and seg[SUN0 + timedelta(days=7)] is None and seg[SUN0 + timedelta(days=14)] == "clean_holdout"


def test_logit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 3))
    b = np.array([0.3, 1.0, -0.5, 0.0])
    y = (rng.random(20000) < 1 / (1 + np.exp(-(b[0] + X @ b[1:])))).astype(float)
    est, se = R.fit_logit(X, y, ridge=0.0)
    assert np.all(np.abs(est - b) < 4 * se + 1e-3)


def test_walk_forward_is_causal():
    panel = _panel(150)
    pred, models = R.walk_forward(panel)
    assert models.height > 0 and pred.height > 0
    cut = SUN0 + timedelta(weeks=100)
    # scramble every outcome from week 100 on: predictions made at refits m <= cut + 9 days must not move
    rng = np.random.default_rng(9)
    late = pl.col("D") >= cut
    p2 = panel.with_columns(pl.when(late).then(pl.lit(rng.standard_normal())).otherwise(pl.col("diff")).alias("diff"))
    p2 = p2.with_columns(pl.when(late).then(1 - pl.col("y")).otherwise(pl.col("y")).alias("y"))
    pred2, _ = R.walk_forward(p2)
    safe = pl.col("D") + pl.duration(days=1) < date((cut + timedelta(days=9)).year, (cut + timedelta(days=9)).month, 1)
    a = pred.filter(safe).sort("D", "symbol")["p"].to_numpy()
    b = pred2.filter(safe).sort("D", "symbol")["p"].to_numpy()
    assert len(a) > 100 and np.allclose(a, b)
    # and the model picks up the planted signal out of sample
    ic = R.ic_stats(pred)
    assert ic["ic"] > 0.3 and ic["t"] > 5


def test_tr4_detects_planted_effect_and_gate_null_is_sane():
    panel = _panel(120, seed=2)
    cuts = R.tr4_cuts(panel)
    r = R.tr4_stats(panel, "llm_in_sample", cuts)
    assert r["diff"] > 0 and r["t"] > 2 and r["by_tercile"]["top"]["n"] > 0
    pure = _panel(120, seed=3, beta=0.0)
    pure = pure.with_columns(pl.Series("diff", np.random.default_rng(4).standard_normal(pure.height)))
    r0 = R.tr4_stats(pure.with_columns(pl.col("diff").alias("trend7")), "llm_in_sample", R.tr4_cuts(pure))
    assert abs(r0["t"]) < 3.5


def test_live_tool_uses_point_in_time_features_and_model(lake, tmp_path):
    from datetime import UTC, datetime

    from ai_trader.tools.regime_tf import trend_vs_mr
    from tests.conftest import make_bars

    lake.write("bars", make_bars(6 * 320, "TESTUSDT", freq_h=4, seed=5))
    rows = []
    for k, (m, icpt) in enumerate(((date(2026, 1, 1), -1.0), (date(2026, 8, 1), 1.0), (date(2026, 12, 1), 3.0))):
        rows.append({"fit_month": m, "n_train": 100 + k, "n_weeks": 60, "base_rate": 0.4, "intercept": icpt,
                     **{f"b_{c}": 0.0 for c in FEATURES}, **{f"mu_{c}": 0.0 for c in FEATURES}, **{f"sd_{c}": 1.0 for c in FEATURES}})
    mp = tmp_path / "model.parquet"
    pl.DataFrame(rows).write_parquet(mp)
    as_of = datetime(2026, 9, 15, 12, tzinfo=UTC)
    r = trend_vs_mr("TESTUSDT", as_of=as_of, lake=lake, model_path=mp)
    assert r.value["fit_month"] == "2026-08-01"  # never the December model
    assert np.isclose(r.value["p_trend_wins"], 1 / (1 + np.exp(-1.0)))
    assert r.value["date"] == "2026-09-14" and r.data_max_ts <= as_of
    assert r.value["gate"] == "trend" and np.isfinite(r.value["vol_pct_1y"])
