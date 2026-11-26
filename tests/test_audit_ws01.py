import math

import numpy as np
from statsmodels.stats.multitest import multipletests

from ai_trader.research import audit_ws01 as A


def test_bh_by_match_statsmodels():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=80), rng.uniform(0, 0.002, size=20)])
    for q in (0.05, 0.10):
        assert (A.bh(p, q) == multipletests(p, q, method="fdr_bh")[0]).all()
        assert (A.by(p, q) == multipletests(p, q, method="fdr_by")[0]).all()
    assert np.allclose(A.bh_adjusted(p), multipletests(p, method="fdr_bh")[1])
    assert np.allclose(A.bh_adjusted(p, dependence=True), multipletests(p, method="fdr_by")[1])


def test_simes():
    assert math.isclose(A.simes([0.01, 0.5]), 0.02)
    assert math.isclose(A.simes([0.04, 0.04]), 0.04)


def _row(metrics, seg="llm_in_sample", verdict="contradicts", hid="H-X"):
    return {"id": hid, "segment": seg, "verdict": verdict, "metrics": metrics}


def test_extract_priority_and_conversions():
    s = A.extract(_row({"p": 0.03, "t": 5.0, "n": 50}))
    assert s["source"] == "p:p" and s["p"] == 0.03
    s = A.extract(_row({"t": 2.0}))
    assert s["source"] == "t:t" and math.isclose(s["p"], 0.0455, abs_tol=1e-3)
    s = A.extract(_row({"t_nw": -3.0, "t": 9.0, "n_weeks": 20}))
    assert s["source"] == "t:t_nw" and s["n_key"] == "n_weeks" and s["p"] > 0.0027      # Student-t(19) is wider than Normal
    s = A.extract(_row({"ratio": 2.0, "ci95": [1.5, 2.6667]}))
    assert s["source"].startswith("ci:") and s["p"] < 1e-5
    s = A.extract(_row({"p_holm": 0.2}))
    assert s["flag"] == "adjusted_only" and s["p"] == 0.2
    s = A.extract(_row({"BTC": {"t": 3.0}, "ETH": {"t": 0.1}}))
    assert s["flag"] == "cell_combined" and math.isclose(s["p"], 2 * 0.0026998, rel_tol=1e-3)
    s = A.extract(_row({"clean_holdout": {"t": 2.5}, "llm_in_sample": {"t": 9.0}}, seg="clean_holdout"))
    assert s["scope"] == "clean_holdout" and s["t"] == 2.5
    s = A.extract(_row({"median": 0.7}))
    assert s["source"] == "none" and s["p"] == 1.0


def test_manual_map_only_for_supports():
    m = {"frac_below_1": 0.8, "n_weekends": 20}
    s = A.extract(_row(m, verdict="supports", hid="H-W1"))
    assert s["source"] == "manual:H-W1" and s["p"] < 0.02
    assert A.extract(_row(m, verdict="contradicts", hid="H-W1"))["source"] == "none"


def test_claim_type():
    assert A.claim_type({"title": "News sentiment does not predict direction (null)"}) == "null"
    assert A.claim_type({"title": "Weekend volatility is lower than weekday volatility"}) == "effect"


def test_run_on_snapshot_is_deterministic():
    res = A.run(write=False)
    f = res["fdr"]["F_all"]
    assert f["m"] == A.SNAPSHOT_ROWS
    assert 0 <= f["BY0.10"]["supports_surviving"] <= f["BH0.10"]["supports_surviving"] <= f["effect_supports"]
    assert set(res["verdicts"]) >= {f"H-WS01-{i}" for i in range(1, 9)}
