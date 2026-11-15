"""Parameter recovery of the fitted laws in ai_trader.sim.laws (microstructure pilot, H-MS1..H-MS4)."""

import math

import numpy as np
import pytest
from scipy import stats

from ai_trader.sim import laws as L

N = 40_000


def _rng(s=7):
    return np.random.default_rng(s)


@pytest.mark.parametrize("family, true", [
    ("exponential", {"scale": 2.0}),
    ("weibull", {"c": 0.6, "scale": 1.5}),
    ("gamma", {"a": 0.45, "scale": 3.0}),
    ("lognormal", {"s": 1.3, "scale": 0.8}),
    ("gengamma", {"a": 2.0, "c": 0.5, "scale": 0.3}),
    ("burr12", {"c": 1.4, "d": 0.7, "scale": 2.0}),
    ("loglogistic", {"c": 1.8, "scale": 0.5}),
])
def test_family_recovers_parameters(family, true):
    x = L.Law(family, true).sample(N, _rng())
    fit = L.fit_law(family, x)
    for k, v in true.items():
        assert fit.params[k] == pytest.approx(v, rel=0.12), (k, fit.params)
    # the fitted law is at least as good as the truth on its own sample
    assert fit.loglik_fit >= L.Law(family, true).loglik(x) - 1e-6 * abs(fit.loglik_fit)


def test_logpdf_matches_scipy():
    x = np.geomspace(1e-3, 50, 30)
    cases = [("weibull", {"c": 0.7, "scale": 1.2}, stats.weibull_min(0.7, scale=1.2)),
             ("gengamma", {"a": 1.5, "c": 0.6, "scale": 0.4}, stats.gengamma(1.5, 0.6, scale=0.4)),
             ("burr12", {"c": 1.2, "d": 0.8, "scale": 2.0}, stats.burr12(1.2, 0.8, scale=2.0)),
             ("loglogistic", {"c": 2.1, "scale": 0.7}, stats.fisk(2.1, scale=0.7)),
             ("lognormal", {"s": 0.9, "scale": 1.1}, stats.lognorm(0.9, scale=1.1)),
             ("gamma", {"a": 0.5, "scale": 2.0}, stats.gamma(0.5, scale=2.0))]
    for fam, p, ref in cases:
        np.testing.assert_allclose(L.Law(fam, p).logpdf(x), ref.logpdf(x), rtol=1e-9, atol=1e-9)


def test_exponential_mixture_em():
    true = L.Law("expmix2", {"w1": 0.7, "w2": 0.3, "rate1": 10.0, "rate2": 0.5})
    x = true.sample(N, _rng())
    fit = L.fit_law("expmix2", x)
    assert fit.params["w1"] == pytest.approx(0.7, abs=0.03)
    assert fit.params["rate1"] == pytest.approx(10.0, rel=0.1)
    assert fit.params["rate2"] == pytest.approx(0.5, rel=0.1)
    f3 = L.fit_law("expmix3", x)
    assert f3.loglik_fit >= fit.loglik_fit - 1.0


def test_csn_recovers_alpha():
    rng = _rng(3)
    xmin, alpha = 100.0, 2.5
    tail = L.PowerLawTail(xmin, alpha, 0, 1, 0.0).sample(8000, rng)
    body = rng.uniform(1.0, xmin, 12000)
    x = np.concatenate([body, tail])
    fit = L.fit_powerlaw(x)
    assert fit.alpha == pytest.approx(alpha, abs=0.12)
    assert 50 <= fit.xmin <= 200
    p = L.powerlaw_bootstrap_p(x, fit, B=30, rng=rng)
    assert p > 0.05
    v = L.vuong_lognormal(x, fit)
    assert not (v["R"] < 0 and v["p"] < 0.05)  # a truncated log-normal can mimic a pure power law (R near 0)


def test_csn_rejects_lognormal_as_powerlaw_vuong():
    x = _rng(4).lognormal(0.0, 0.6, 30000)
    fit = L.fit_powerlaw(x)
    assert L.vuong_lognormal(x, fit)["R"] < 0


def test_acd_recovers_parameters():
    true = L.ACD(0.05, 0.10, 0.85, "weibull", {"k": 0.8})
    x = true.sample(60_000, _rng(5))
    fit = L.fit_acd(x, "weibull")
    assert fit.alpha == pytest.approx(0.10, abs=0.02)
    assert fit.beta == pytest.approx(0.85, abs=0.03)
    assert fit.shape["k"] == pytest.approx(0.8, abs=0.02)
    assert fit.unconditional_mean() == pytest.approx(1.0, rel=0.1)
    g = L.fit_acd(x, "gengamma", start=fit)
    assert g.loglik_fit >= fit.loglik_fit - 1.0
    # residuals of the true model are closer to exponential (Weibull k=0.8) than the raw durations are
    ks_raw = L.ks_exponential(x)[0]
    ks_res = L.ks_exponential(fit.residuals(x))[0]
    assert ks_res < ks_raw


def test_trailing_scale_is_causal():
    t = np.cumsum(_rng(6).exponential(1.0, 5000))
    s = L.trailing_scale(t, window_s=100.0)
    t2 = t.copy()
    t2[3000:] += 1000.0  # changing the future does not change the past scales
    s2 = L.trailing_scale(t2, window_s=100.0)
    np.testing.assert_allclose(s[:2999], s2[:2999], equal_nan=True)
    assert np.nanmedian(s[200:]) == pytest.approx(1.0, rel=0.1)


def test_fgn_sign_generator_reproduces_power_law_acf():
    target = L.FGNSignModel(c=0.15, gamma=0.5)
    sg = target.sample(400_000, _rng(8))
    rho = L.acf(sg, 1000)
    k, v = L.binned(rho, L.log_bins(1000))
    fit = L.fit_power_acf(k, v)
    assert fit["gamma"] == pytest.approx(0.5, abs=0.12)
    assert fit["c"] == pytest.approx(0.15, rel=0.3)
    np.testing.assert_allclose(target.model_acf(k[k > 3]), 0.15 * k[k > 3] ** -0.5, rtol=0.25)


def test_lmf_generator_long_memory():
    sg = L.LMFSignModel(a=1.5, n_meta=5).sample(200_000, _rng(9))
    rho = L.acf(sg, 500)
    k, v = L.binned(rho, L.log_bins(500))
    pw, ex = L.fit_power_acf(k, v), L.fit_exp_acf(k, v)
    assert 0.2 < pw["gamma"] < 0.9
    mp, me, _ = L.heldout_log_mse(k, v, pw, ex)
    assert mp < me


def test_pacf_ar1():
    rng = _rng(10)
    e = rng.standard_normal(50_000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.5 * x[i - 1] + e[i]
    p = L.pacf_from_acf(L.acf(x, 10), 10)
    assert p[0] == pytest.approx(0.5, abs=0.02)
    assert np.all(np.abs(p[1:]) < 0.03)


def test_dfa_hurst():
    rng = _rng(11)
    assert L.dfa(rng.standard_normal(40_000))["H"] == pytest.approx(0.5, abs=0.05)
    assert L.dfa(L.fgn(40_000, 0.75, rng))["H"] == pytest.approx(0.75, abs=0.06)


def test_size_law_and_hurdle_roundtrip():
    rng = _rng(12)
    s = L.SizeLaw(mu=3.0, sigma=1.0, xmin=200.0, alpha=2.5, p_tail=0.1)
    x = s.sample(50_000, rng)
    assert np.mean(x >= 200.0) == pytest.approx(0.1, abs=0.01)
    pl_ = L.fit_powerlaw(x)
    assert pl_.alpha == pytest.approx(2.5, abs=0.2)
    h = L.HurdleLaw(0.3, L.Law("gamma", {"a": 0.5, "scale": 2.0}))
    v = h.sample(20_000, rng)
    assert np.mean(v == 0) == pytest.approx(0.3, abs=0.02)
    for obj in (s, h, L.FGNSignModel(0.1, 0.6), L.ACD(0.1, 0.1, 0.8, "weibull", {"k": 0.9}), pl_):
        d = obj.to_dict()
        back = L.law_from_dict(d)
        assert type(back) is type(obj)


def test_aft_logpdf_integrates_to_one():
    t = np.geomspace(1e-6, 1e6, 200_001)
    for dist in ("normal", "logistic", "extreme"):
        f = np.exp(L.aft_logpdf(t, np.zeros_like(t), 0.8, dist))
        assert np.trapezoid(f, t) == pytest.approx(1.0, abs=2e-3)
    assert math.isfinite(L.aft_logpdf(np.array([1.0]), np.array([0.0]), 1.0, "extreme")[0])


def test_nesting_families_never_below_sub_families():
    rng = _rng(13)
    for x in (L.Law("weibull", {"c": 0.45, "scale": 2.0}).sample(30_000, rng), rng.lognormal(0, 2.5, 30_000),
              L.Law("gamma", {"a": 3.0, "scale": 1.0}).sample(30_000, rng)):
        f = L.fit_all(x, families=("weibull", "gamma", "loglogistic", "gengamma", "burr12"))
        assert f["gengamma"].loglik_fit >= max(f["weibull"].loglik_fit, f["gamma"].loglik_fit) - 1e-3
        assert f["burr12"].loglik_fit >= f["loglogistic"].loglik_fit - 1e-3
        assert f["burr12"].loglik_fit >= f["weibull"].loglik_fit - 1.0


def test_translated_sign_model_reproduces_short_and_long_lags():
    m = L.TranslatedSignModel(c=0.3, gamma=0.6, p_buy=0.5, short={1: 0.5, 2: 0.38, 3: 0.32, 5: 0.25, 10: 0.08})
    sg = m.sample(300_000, _rng(14))
    rho = L.acf(sg, 300)
    for k in (1, 3, 10, 100):
        assert rho[k] == pytest.approx(float(m.target_acf([k])[0]), abs=0.03), k
    assert L.law_from_dict(m.to_dict()).short == m.short


def test_acd_sample_anchored_when_near_integrated():
    m = L.ACD(0.02, 0.15, 0.84995, "weibull", {"k": 0.6}, psi0=1.0)
    x = m.sample(50_000, _rng(15))
    assert 0.5 < x.mean() < 2.0
