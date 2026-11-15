import numpy as np
import polars as pl

from ai_trader.research.exit_policy_study import trail_sigma, trailing_exit


def _path():
    # long signal throughout; closes rise to 110 then fall
    close = np.array([100, 102, 105, 110, 109, 108, 106, 104, 103, 107, 111], dtype=float)
    sig = np.ones(len(close))
    sigma = np.full(len(close), 0.01)  # 1% per bar
    return sig, close, sigma


def test_exits_at_first_bar_where_retracement_exceeds_k_sigma():
    sig, close, sigma = _path()
    k = 2.5  # 2.5% retracement in log terms from the best close 110
    out, stop = trailing_exit(sig, close, sigma, k)
    retr = np.log(np.maximum.accumulate(close)) - np.log(close)
    first = int(np.argmax(retr > k * 0.01))
    assert first == 6  # 110 -> 106 is log 3.7% > 2.5%; 108 is 1.8%
    assert np.all(out[:first] == 1.0)
    assert out[first] == 0.0 and stop[first]
    assert stop.sum() == 1


def test_no_reentry_without_fresh_signal_change():
    sig, close, sigma = _path()
    out, stop = trailing_exit(sig, close, sigma, 2.5)
    # raw stays long after the stop (new highs at the end): no re-entry
    assert np.all(out[6:] == 0.0)
    # raw goes flat at bar 8 then long again at bar 9: re-entry at bar 9 (fresh signal change)
    sig2 = sig.copy()
    sig2[8] = 0.0
    out2, _ = trailing_exit(sig2, close, sigma, 2.5)
    assert np.all(out2[6:9] == 0.0) and out2[9] == 1.0 and out2[10] == 1.0
    # an opposite raw signal opens the opposite trade at that same close
    sig3 = sig.copy()
    sig3[7:] = -1.0
    out3, _ = trailing_exit(sig3, close, sigma, 2.5)
    assert out3[6] == 0.0 and out3[7] == -1.0


def test_running_best_resets_per_trade_and_shorts_mirror():
    close = np.array([100, 98, 95, 96, 97, 99, 100], dtype=float)
    sig = -np.ones(len(close))
    sigma = np.full(len(close), 0.01)
    out, stop = trailing_exit(sig, close, sigma, 2.5)
    # best (min) 95; 97 is +2.08% (no exit), 99 is +4.1% -> exit at bar 5
    assert np.all(out[:5] == -1.0) and out[5] == 0.0 and stop[5]
    # strategy's own exit (raw flat) still applies and resets the trade
    sig2 = sig.copy()
    sig2[3] = 0.0
    out2, stop2 = trailing_exit(sig2, close, sigma, 2.5)
    assert out2[3] == 0.0 and not stop2[3]
    # after the own exit, new trade starts at 97 (bar 4): best 97, 100 is +3.0% > 2.5% -> exit at bar 6, not 5
    assert out2[4] == -1.0 and out2[5] == -1.0 and out2[6] == 0.0 and stop2[6]


def test_causal_uses_only_prices_and_sigma_known_at_each_bar():
    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    sig = np.sign(np.sin(np.arange(n) / 25.0)) * 0.7
    sigma = np.abs(rng.normal(0.01, 0.002, n))
    out, stop = trailing_exit(sig, close, sigma, 1.5)
    for cut in (50, 150, 299):
        c2, s2, g2 = close.copy(), sig.copy(), sigma.copy()
        c2[cut + 1:] *= np.exp(rng.normal(0, 0.05, n - cut - 1))
        s2[cut + 1:] = -s2[cut + 1:]
        g2[cut + 1:] = 1e-6
        o2, st2 = trailing_exit(s2, c2, g2, 1.5)
        assert np.array_equal(out[:cut + 1], o2[:cut + 1]) and np.array_equal(stop[:cut + 1], st2[:cut + 1])


def test_strict_threshold_and_invalid_sigma():
    close = np.array([100.0, 100.0 * np.exp(-0.0249)])
    out, _ = trailing_exit(np.ones(2), close, np.full(2, 0.01), 2.5)
    assert out[1] == 1.0  # just inside k sigma: no exit
    close = np.array([100.0, 100.0 * np.exp(-0.0251)])
    out, _ = trailing_exit(np.ones(2), close, np.full(2, 0.01), 2.5)
    assert out[1] == 0.0  # just beyond: exit
    out, _ = trailing_exit(np.ones(3), np.array([100.0, 90.0, 80.0]), np.array([0.01, np.nan, 0.0]), 1.5)
    assert np.all(out == 1.0)  # no valid sigma: the trail cannot fire


def test_trail_sigma_prefers_robust_risk_sigma():
    f = pl.DataFrame({"sigma_next": [0.01, 0.02, 0.03], "sigma_risk_next": [0.015, None, 0.0]})
    assert np.allclose(trail_sigma(f), [0.015, 0.02, 0.03])
