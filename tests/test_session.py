from ai_trader.agent.session import Candidate, p_first


def test_driftless_null():
    # log space: ln(110/100) < ln(100/90), so the target is slightly closer -> 0.525
    assert abs(p_first(100, 90, 110) - 0.525) < 0.001
    assert p_first(100, 95, 115) < 0.3


def test_candidate_has_shock_ts_default():
    c = Candidate("x", "BTCUSDT", "trend_4h", "long", "limit", 1, 0.9, 1.3, 0.25, 0.28, 3, None, "r", "majors")
    assert c.shock_ts is None
