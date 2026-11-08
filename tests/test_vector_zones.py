import numpy as np

from ai_trader.research.vector_zones import _first_touch, _race


def test_race_null_accounts_for_wick_touch_and_resolves():
    close = np.array([100.5, 100.8, 101.2, 99.0])
    out = _race(close, 0, L=100.0, s=1.0, sig=1.0)   # touch close already 0.5 sigma back above the level
    assert out[0] == 1 and abs(out[1] - 0.75) < 1e-12 and out[2] == 2
    out = _race(np.array([99.8, 98.9]), 0, L=100.0, s=1.0, sig=1.0)
    assert out[0] == 0 and abs(out[1] - 0.4) < 1e-12


def test_first_touch_finds_wick_and_respects_start():
    low = np.array([101, 101.5, 99.9, 98.0])
    high = np.array([103, 102.0, 101.0, 99.0])
    assert _first_touch(low, high, 0, 100.0) == 2
    assert _first_touch(low, high, 3, 100.0) is None
