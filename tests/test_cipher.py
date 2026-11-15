import numpy as np
import pytest

from ai_trader.features.cipher import cipher_trades, crosses, dots, ema, money_flow, sma, wavetrend


def _ohlc(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_t(4, n) / np.sqrt(2)))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + 0.003 * rng.random(n))
    lo = np.minimum(o, c) * (1 - 0.003 * rng.random(n))
    return o, h, lo, c


def test_ema_sma_basic():
    x = np.arange(10, dtype=float)
    e = ema(x, 3)
    assert e[0] == 0.0 and e[1] == pytest.approx(0.5)
    s = sma(x, 4)
    assert np.isnan(s[2]) and s[3] == pytest.approx(1.5)


def test_wavetrend_definition():
    o, h, lo, c = _ohlc(300)
    ap = (h + lo + c) / 3
    esa = ema(ap, 9)
    d = ema(np.abs(ap - esa), 9)
    ci = np.where(d > 0, (ap - esa) / (0.015 * np.where(d > 0, d, 1)), 0.0)
    wt1 = ema(ci, 12)
    w1, w2 = wavetrend(h, lo, c)
    assert np.allclose(w1, wt1)
    assert np.allclose(w2[3:], np.convolve(wt1, np.ones(4) / 4, "valid"))


@pytest.mark.parametrize("mf", [True, False])
def test_wavetrend_dots_and_mf_are_causal(mf):
    o, h, lo, c = _ohlc(3000, seed=5)
    full = dots(o, h, lo, c, mf)
    for k in (250, 999, 2001, 2999):
        part = dots(o[:k], h[:k], lo[:k], c[:k], mf)
        for key in ("wt1", "wt2", "mf"):
            assert np.allclose(part[key], full[key][:k], equal_nan=True), (key, k)
        for key in ("green", "red", "cross_up", "cross_down"):
            assert np.array_equal(part[key], full[key][:k]), (key, k)


def test_future_bar_perturbation_does_not_change_past():
    o, h, lo, c = _ohlc(1500, seed=9)
    base = dots(o, h, lo, c, True)
    c2 = c.copy()
    h2 = h.copy()
    c2[1000:] *= 1.5
    h2[1000:] *= 1.5
    pert = dots(o, h2, lo, c2, True)
    assert np.allclose(base["wt2"][:1000], pert["wt2"][:1000], equal_nan=True)
    assert np.array_equal(base["green"][:1000], pert["green"][:1000])


def test_dot_conditions_and_mf_filter():
    o, h, lo, c = _ohlc(5000, seed=11)
    d = dots(o, h, lo, c, False)
    assert d["green"].sum() > 0 and d["red"].sum() > 0
    assert np.all(d["wt2"][d["green"]] <= -53) and np.all(d["wt2"][d["red"]] >= 53)
    up, dn = crosses(d["wt1"], d["wt2"])
    assert np.all(up[d["green"]]) and np.all(dn[d["red"]])
    f = dots(o, h, lo, c, True)
    assert np.all(f["mf"][f["green"]] > 0) and np.all(f["mf"][f["red"]] < 0)
    assert f["green"].sum() <= d["green"].sum()


def test_money_flow_formula():
    o, h, lo, c = _ohlc(200, seed=2)
    mf = money_flow(o, h, lo, c)
    x = (c - o) / (h - lo) * 150
    assert mf[59] == pytest.approx(x[:60].mean() - 2.5)


def test_trades_exit_rules_and_no_overlap():
    o, h, lo, c = _ohlc(5000, seed=13)
    d = dots(o, h, lo, c, False)
    tr = cipher_trades(o, h, lo, c, False, max_hold=12, start=200)
    assert tr
    prev_exit = -1
    for i, j, s in tr:
        assert i >= prev_exit and 0 < j - i <= 12 and i >= 200
        assert (d["green"][i] and s == 1) or (d["red"][i] and s == -1)
        opp = d["cross_down"] if s == 1 else d["cross_up"]
        # exit is the first opposite cross, or the time stop if none occurred
        assert not opp[i + 1:j].any()
        assert opp[j] or j - i == 12
        prev_exit = j
    # causal: trades entered and closed within the first k bars are unchanged on the truncated input
    k = 3000
    part = cipher_trades(o[:k], h[:k], lo[:k], c[:k], False, max_hold=12, start=200)
    assert [t for t in tr if t[1] < k - 1] [: len(part) - 1] == part[: len(part) - 1]
