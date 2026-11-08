import numpy as np
import pytest

from ai_trader.strategies.trend_zoo import ZOO, _filter_z, ensemble, lookback_bars, mac_weights, slope_weights, tsmom_weights, zoo
from tests.test_strategies import build_frame


def _close(r: np.ndarray) -> np.ndarray:
    return 100 * np.exp(np.cumsum(r))


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 2500
    mu = np.repeat(rng.choice([-0.002, 0.0, 0.002], size=n // 250 + 1), 250)[:n]
    r = mu + 0.01 * rng.standard_t(4, n) / np.sqrt(2.0)
    return build_frame(_close(r), rng)


def _check_causal(fn, frame, cuts=5, seed=0):
    full = fn(frame).to_numpy()
    rng = np.random.default_rng(seed)
    n = frame.height
    for k in sorted(rng.integers(max(10, n // 4), n, size=cuts)):
        part = fn(frame.head(int(k))).to_numpy()
        assert np.allclose(part, full[: int(k)], atol=1e-10), f"{fn.__name__} not causal at cut {k}"


def test_zoo_size_and_names():
    assert len(ZOO) == 21
    assert {"tsmom_1d", "mac_14_60", "donch_60", "ema_4_16_64", "slope_30d", "cfg_tsmom"} <= set(ZOO)


@pytest.mark.parametrize("name", sorted(ZOO))
def test_causal_and_bounded(name, frame):
    fn = ZOO[name]
    _check_causal(fn, frame)
    v = fn(frame).to_numpy()
    assert len(v) == frame.height
    assert np.isfinite(v).all() and (np.abs(v) <= 1).all()
    assert (v != 0).any()


def test_label_column_ignored(frame):
    f2 = frame.with_columns((frame["close"].log().shift(-1) - frame["close"].log()).alias("fwd_ret"))
    for name, fn in ZOO.items():
        assert np.array_equal(fn(frame).to_numpy(), fn(f2).to_numpy()), name


def test_ensemble_causal_and_mean(frame):
    _check_causal(ensemble, frame, cuts=3)
    e = ensemble(frame).to_numpy()
    m = np.mean([fn(frame).to_numpy() for fn in ZOO.values()], axis=0)
    assert np.allclose(e, m)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("bpd", [6, 1])
def test_sustained_trend_gives_signal_in_its_direction(direction, bpd):
    rng = np.random.default_rng(3)
    n = 3000
    r = 0.005 * rng.standard_normal(n)
    r[1200:] += direction * 0.0025  # 0.5 sigma per bar for 1800 bars
    f = build_frame(_close(r))
    for name, fn in zoo(bpd).items():
        s = fn(f).to_numpy()
        # short horizons are noisy bar to bar: judge the average over the trend's last 300 bars
        assert direction * s[-300:].mean() > 0.15, (name, s[-300:].mean())
        assert direction * np.median(s[-300:]) > 0, name
    assert direction * ensemble(f, bpd).to_numpy()[-300:].mean() > 0.4


def test_filter_z_is_unit_variance_under_random_walk():
    rng = np.random.default_rng(11)
    n = 60000
    r = 0.01 * rng.standard_normal(n)
    close = _close(r)
    sigma = np.full(n, 0.01)
    for w in (tsmom_weights(18), mac_weights(6, 42), slope_weights(84)):
        z = _filter_z(close, sigma, w)
        z = z[np.isfinite(z)][:: len(w)]  # non-overlapping windows
        assert abs(z.std() - 1) < 0.05 and abs(z.mean()) < 0.05


def test_weights_sum_to_zero_and_lookback():
    for w in (tsmom_weights(5), mac_weights(6, 42), slope_weights(30)):
        assert abs(w.sum()) < 1e-12
    assert lookback_bars(6) >= 384  # slowest own variant: 64-day EMA on 4h bars
