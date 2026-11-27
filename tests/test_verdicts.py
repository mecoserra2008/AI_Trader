"""D7a (2026-11-27): a clean_holdout "supports" must pass a declared one-sided test, not just keep the sign."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import verdicts as V


def test_judge_holdout_needs_a_passing_one_sided_test():
    h = "clean_holdout"
    assert V._judge(h, False, True, 20) == "inconclusive"             # sign only: no longer "supports"
    assert V._judge(h, False, True, 20, p_hold=0.20) == "inconclusive"
    assert V._judge(h, False, True, 20, p_hold=0.04) == "supports"
    assert V._judge(h, False, True, 20, p_hold=0.04, alpha=0.025) == "inconclusive"   # a registered stricter level
    assert V._judge(h, False, False, 20, p_hold=0.001) == "contradicts"               # wrong sign
    assert V._judge(h, True, True, V.MIN_HOLDOUT - 1, p_hold=1e-9) == "inconclusive"   # MIN_HOLDOUT kept
    assert V._judge(h, False, True, 20, p_hold=float("nan")) == "inconclusive"
    # in-sample is unchanged: the registered rule decides
    assert V._judge("llm_in_sample", True, True, 5) == "supports"
    assert V._judge("llm_in_sample", False, False, 500) == "contradicts"
    assert V._judge("llm_in_sample", False, True, 500) == "inconclusive"


def test_one_sided_p_helpers():
    assert V.p_one_sided_t(1.6449, 10**7, +1) == pytest.approx(0.05, abs=1e-4)
    assert V.p_one_sided_t(-1.6449, None, -1) == pytest.approx(0.05, abs=1e-4)
    assert V.p_one_sided_t(2.0, 10, +1) > V.p_one_sided_t(2.0, 1000, +1)            # small samples: Student-t tails
    assert V.p_one_sided_t(2.0, 50, -1) > 0.95                                       # wrong direction
    assert np.isnan(V.p_one_sided_t(float("nan"), 50))
    assert V.p_one_sided_binom(15, 18) == pytest.approx(0.00377, abs=1e-4)           # sign test
    assert V.p_one_sided_binom(9, 18) > 0.5
    assert V.p_one_sided_boot(np.r_[np.full(99, 2.0), 0.5], null=1.0) == pytest.approx(2 / 101)
    est, se = 0.2, 0.1
    assert V.p_one_sided_ci(est, est - 1.96 * se, est + 1.96 * se) == pytest.approx(0.0228, abs=1e-3)
    assert V.p_one_sided_ci(1.5, 1.2, 1.9, null=1.0, log=True) < 0.01


def _weekends(n_in: int, n_hold: int, mean_hold: float, sd_hold: float, seed: int = 0) -> pl.DataFrame:
    """One row per (weekend, symbol); the post-weekend signed z of each weekend in the holdout is drawn N(mean_hold, sd_hold)."""
    rng = np.random.default_rng(seed)
    k = datetime(2026, 7, 1, tzinfo=UTC)
    rows = []
    for i in range(n_in):
        rows.append({"event_ts": k - timedelta(days=7 * (i + 1)), "weekend_id": -i - 1, "wk_z": 1.5, "vol_ratio": 0.5,
                     "signed_post_z": float(rng.normal(-0.4, 1.0))})
    for i in range(n_hold):
        rows.append({"event_ts": k + timedelta(days=7 * i + 1), "weekend_id": i, "wk_z": 1.5, "vol_ratio": 0.5,
                     "signed_post_z": float(rng.normal(mean_hold, sd_hold))})
    return pl.DataFrame(rows)


def test_h_w2_holdout_sign_only_is_now_inconclusive_and_a_strong_holdout_still_supports():
    weak = V.h_w2(_weekends(200, 10, -0.05, 1.0, seed=1))["clean_holdout"]
    assert weak[1]["mean_signed_post_z"] < 0 or weak[0] == "contradicts"
    if weak[1]["mean_signed_post_z"] < 0:
        assert weak[0] == "inconclusive" and weak[1]["p_one_sided"] > V.HOLDOUT_ALPHA
    strong = V.h_w2(_weekends(200, 30, -1.0, 0.5, seed=2))["clean_holdout"]
    assert strong[0] == "supports" and strong[1]["p_one_sided"] < V.HOLDOUT_ALPHA and strong[1]["alpha"] == V.HOLDOUT_ALPHA


def test_h_w1_uses_a_sign_test_in_the_holdout():
    rows = []
    k = datetime(2026, 7, 1, tzinfo=UTC)
    for i, x in enumerate([0.8] * 6 + [1.2] * 4):      # 6 of 10 weekends below 1: sign right, sign test p 0.38
        rows.append({"event_ts": k + timedelta(days=7 * i + 1), "weekend_id": i, "rv_ratio": x})
    v, m = V.h_w1(pl.DataFrame(rows))["clean_holdout"]
    assert m["median_rv_ratio"] < 1 and v == "inconclusive" and m["p_one_sided"] == pytest.approx(0.377, abs=1e-3)


def test_h_s1_intersection_union_over_cells():
    def frame(t_r3: float) -> pl.DataFrame:
        rows = []
        for seg in ("llm_in_sample", "clean_holdout"):
            for cell, m, t in (("R3", 0.2, t_r3), ("R9", 0.3, 3.0), ("R4", -0.2, -3.0), ("R5", -0.25, -3.0)):
                rows.append({"segment": seg, "h_bars": 24, "cell": cell, "n_events": 60, "mean_signed_z": m, "t": t})
        return pl.DataFrame(rows)
    assert V.h_s1(frame(3.0), 24)["clean_holdout"][0] == "supports"
    v, m = V.h_s1(frame(0.5), 24)["clean_holdout"]    # one weak cell: the signs hold but the joint test fails
    assert v == "inconclusive" and m["p_one_sided"] > V.HOLDOUT_ALPHA
