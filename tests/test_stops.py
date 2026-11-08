import numpy as np

from ai_trader.research import stops as st


def _ib(sub):
    """sub: {hour_index: [(open, high, low), ...]} -> Intrabar provider."""
    def f(j):
        if j not in sub:
            return None
        a = np.array(sub[j], dtype=float)
        return a[:, 0], a[:, 1], a[:, 2], False
    return f


def test_resting_stop_fills_worse_on_gap():
    close = np.array([100.0, 99.0, 99.0])
    # long, stop 98: second 5m bar opens at 97 (gapped through) -> fill at 97, not 98
    ib = _ib({1: [(100, 100.5, 99.5), (97, 97.5, 96.8)], 2: [(99, 99, 99)]})
    ex = st.simulate("A", 0, 1, 98.0, 2, close, ib, 3)
    assert ex.reason == "stop" and ex.price == 97.0 and ex.bar == 1
    # touch without gap fills at the stop
    ib2 = _ib({1: [(100, 100.5, 97.5)], 2: [(99, 99, 99)]})
    assert st.simulate("A", 0, 1, 98.0, 2, close, ib2, 3).price == 98.0
    # short mirror: open 103 above stop 102 -> fill 103
    ib3 = _ib({1: [(103, 103.2, 102.5)]})
    assert st.simulate("A", 0, -1, 102.0, None, np.array([100.0, 101.0]), ib3, 2).price == 103.0


def test_close_confirmed_ignores_wick_that_closes_inside():
    close = np.array([100.0, 99.0, 101.0])
    # hour 1 wicks to 97.5 (< stop 98) but closes 99 (inside); hour 2 is the strategy exit
    ib = _ib({1: [(100, 100, 97.5), (98.5, 99.2, 98.4)], 2: [(99, 101, 99)]})
    a = st.simulate("A", 0, 1, 98.0, 2, close, ib, 3)
    b = st.simulate("B", 0, 1, 98.0, 2, close, ib, 3)
    assert a.reason == "stop" and a.reverted
    assert b.reason == "signal" and b.price == 101.0
    # a 1h close beyond the stop exits B at that close
    close2 = np.array([100.0, 97.0, 101.0])
    b2 = st.simulate("B", 0, 1, 98.0, 2, close2, _ib({1: [(100, 100, 96.9)], 2: [(97, 101, 97)]}), 3)
    assert b2.reason == "close_stop" and b2.price == 97.0 and b2.bar == 1


def test_catastrophe_stop_triggers():
    close = np.array([100.0, 99.0])
    # stop 98 -> distance 2 -> catastrophe at 96; low 95.5 touches it (open 99 so fill at 96)
    b = st.simulate("B", 0, 1, 98.0, None, close, _ib({1: [(99, 99, 95.5)]}), 2)
    assert b.reason == "catastrophe" and b.price == 96.0
    # gap through the catastrophe level fills at the open
    b2 = st.simulate("B", 0, 1, 98.0, None, close, _ib({1: [(99, 99, 98.5), (94, 95, 93)]}), 2)
    assert b2.price == 94.0


def test_pivots_are_causal_and_stop_band():
    low = np.array([10, 9, 8, 5, 8, 9, 10, 11], dtype=float)
    high = low + 1
    lows, _ = st.pivots(high, low, k=3)
    assert lows == [(6, 5.0)]  # pivot at bar 3 usable only from bar 6
    assert st.structural_stop(5, 1, 11.0, 0.2, lows, []) is None  # not yet confirmed
    s = st.structural_stop(6, 1, 11.0, 0.2, lows, [])  # sigma*close = 2.2, buf 0.66
    assert np.isclose(s, 5.0 - 0.3 * 2.2)
    assert st.structural_stop(6, 1, 11.0, 0.05, lows, []) is None  # distance > 4 sigma


def test_entries_and_r_multiple():
    e = st.entries(np.array([0, 0.5, 0.7, 0, -0.3, 0.2]))
    assert e == [(1, 1, 3), (4, -1, 5), (5, 1, None)]
    c = 0.00075
    r = st.r_multiple(1, 100.0, 98.0, 98.0, c)
    assert np.isclose(r, (-2 - c * 198) / (2 + 100 * 2 * c))
