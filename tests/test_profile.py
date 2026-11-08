from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.profile import asof_profile, bin_volumes, daily_profiles, session_profile, va_overlap, value_area

W = np.log1p(0.0005)


def _bars(days: int, per_day: int = 288, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * per_day
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[100.0, c[:-1]]
    ev = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(n)]
    return pl.DataFrame({
        "event_ts": ev, "open": o, "close": c,
        "high": np.maximum(o, c) * (1 + rng.uniform(0, 5e-4, n)), "low": np.minimum(o, c) * (1 - rng.uniform(0, 5e-4, n)),
        "volume": rng.uniform(1, 10, n), "known_ts": [t + timedelta(minutes=5, seconds=2) for t in ev],
    })


def test_bin_volumes_conserve_volume_and_spread_uniformly():
    edges, bins = bin_volumes(np.array([100.0, 100.0]), np.array([101.0, 100.0]), np.array([10.0, 3.0]))
    assert abs(bins.sum() - 13.0) < 1e-9
    assert np.allclose(np.diff(np.log(edges)), W)
    # the zero-range bar's volume lands in the bin containing 100
    k = np.searchsorted(edges, 100.0, side="right") - 1
    inside = (edges[:-1] >= 100.0) & (edges[1:] <= 101.0)
    dens = bins[inside] / (edges[1:][inside] - edges[:-1][inside])
    assert np.allclose(dens, 10.0, rtol=1e-6)  # 10 units of volume per unit of price
    assert bins[k] > 3.0


def test_poc_and_value_area_on_a_known_histogram():
    bins = np.array([1, 2, 3, 10, 6, 5, 1, 1, 1], dtype=float)  # total 30, target 21
    poc, lo, hi = value_area(bins)
    assert poc == 3
    # 10 -> +6 (16) -> +5 (21): the larger neighbour is added each step, lower side never needed
    assert (lo, hi) == (3, 5)
    assert bins[lo:hi + 1].sum() >= 0.7 * bins.sum()
    # ties add both sides
    poc, lo, hi = value_area(np.array([1, 4, 10, 4, 1], dtype=float))
    assert (poc, lo, hi) == (2, 1, 3)


def test_session_profile_va_holds_70pct_and_contains_poc():
    rng = np.random.default_rng(1)
    mid = 100 * np.exp(rng.normal(0, 0.004, 500))
    low, high = mid * 0.999, mid * 1.001
    p = session_profile(low, high, rng.uniform(1, 5, 500))
    assert p.va_share >= 0.70 - 1e-9
    assert p.val <= p.poc <= p.vah
    assert low.min() * 0.999 <= p.val and p.vah <= high.max() * 1.001
    # removing either end bin of the VA drops it below 70%
    lo, hi = p.va_bins
    assert min(p.bins[lo + 1:hi + 1].sum(), p.bins[lo:hi].sum()) < 0.70 * p.bins.sum()


def test_bimodal_volume_puts_poc_on_heavy_node():
    low = np.r_[np.full(50, 100.0), np.full(10, 105.0)]
    high = low * 1.002
    vol = np.r_[np.full(50, 1.0), np.full(10, 20.0)]  # 50 at 100, 200 at 105
    p = session_profile(low, high, vol)
    assert 105.0 <= p.poc <= 105.21
    assert p.val >= 104.9  # the light node is outside the 70% VA


def test_daily_profiles_are_causal_and_stamped_end_of_day():
    b = _bars(5)
    full = daily_profiles(b)
    assert full.height == 5
    # profile for day D only uses day D: truncating after day 3 or perturbing day 4 leaves days 0..2 unchanged
    cut = b.filter(pl.col("event_ts") < datetime(2026, 1, 4, tzinfo=UTC))
    part = daily_profiles(cut)
    assert np.allclose(part["poc"].to_numpy(), full["poc"].to_numpy()[:3])
    shocked = b.with_columns(pl.when(pl.col("event_ts") >= datetime(2026, 1, 4, tzinfo=UTC)).then(pl.col("high") * 1.2).otherwise(pl.col("high")).alias("high"))
    assert np.allclose(daily_profiles(shocked)["vah"].to_numpy()[:3], full["vah"].to_numpy()[:3])
    # known only at the end of the day
    for r in full.iter_rows(named=True):
        assert r["known_ts"] >= datetime.combine(r["day"], datetime.min.time(), tzinfo=UTC) + timedelta(days=1)


def test_asof_profile_never_sees_the_current_day():
    b = _bars(4)
    prof = daily_profiles(b)
    hourly = pl.DataFrame({"event_ts": [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(96)]}).with_columns(
        (pl.col("event_ts") + timedelta(hours=1, seconds=2)).alias("known_ts"))
    j = asof_profile(hourly, prof)
    for r in j.iter_rows(named=True):
        if r["pf_day"] is not None:
            # the attached profile's day ended no later than the decision bar closed
            assert datetime.combine(r["pf_day"], datetime.min.time(), tzinfo=UTC) + timedelta(days=1) <= r["event_ts"] + timedelta(hours=1)
    assert j.filter(pl.col("event_ts") < datetime(2026, 1, 1, 23, tzinfo=UTC))["pf_day"].null_count() == 23


def test_va_overlap():
    assert va_overlap(np.array([1.0]), np.array([3.0]), np.array([2.0]), np.array([10.0]))[0] == 0.5
    assert va_overlap(np.array([1.0]), np.array([3.0]), np.array([5.0]), np.array([6.0]))[0] == 0.0


def test_variance_ratio_is_unbiased_under_iid_and_detects_trend():
    from ai_trader.research.auction_study import variance_ratio

    rng = np.random.default_rng(0)
    assert abs(np.mean([variance_ratio(rng.normal(0, 1, 168)) for _ in range(500)]) - 1.0) < 0.05
    e = rng.normal(0, 1, 20000)
    ar = np.zeros_like(e)
    for t in range(1, len(e)):
        ar[t] = 0.3 * ar[t - 1] + e[t]  # positively autocorrelated -> VR > 1
    assert variance_ratio(ar) > 1.5
