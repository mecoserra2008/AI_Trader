import math

import numpy as np

from ai_trader.features import vol_panel as vp


def _panel_inputs(n=700, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = 0.95 * lv[t - 1] + 0.3 * rng.standard_normal()
    rv = scale * np.exp(lv - 8) * rng.chisquare(24, n) / 24
    share = rng.uniform(0.3, 0.7, n)
    return {"rv": rv, "rs_pos": rv * share, "rs_neg": rv * (1 - share), "rq": rv**2 * rng.uniform(1, 3, n),
            "jv": rv * rng.uniform(0, 0.2, n), "gk": rv * rng.uniform(0.6, 1.0, n), "pk": rv * rng.uniform(0.6, 1.0, n)}


def test_forward_mean_and_trailing():
    x = np.arange(10.0)
    np.testing.assert_allclose(vp.forward_mean(x, 2)[:3], [1.5, 2.5, 3.5])
    assert np.isnan(vp.forward_mean(x, 2)[-2:]).all()
    x[5] = np.nan
    f = vp.forward_mean(x, 2)
    assert np.isnan(f[3]) and np.isnan(f[4]) and np.isfinite(f[5])
    m = vp.trailing_nanmean(x, 7, 5)
    assert np.isfinite(m[7]) and np.isnan(m[3])


def test_panel_reduces_to_per_asset_with_one_asset():
    a = _panel_inputs()
    for name in ("har", "harq", "shar", "loghar"):
        X, p = vp.design(name, a)
        for h in (1, 5):
            y = vp.target(a["rv"], h, vp.is_log(name))
            pa = vp.per_asset_forecast(X, y, h, log=vp.is_log(name), min_days=180, first=0)
            pn = vp.panel_forecast([X], [y], h, log=vp.is_log(name), powers=p, refit_days=list(range(180, len(y), 30)))[0]
            np.testing.assert_allclose(pn, pa, rtol=1e-8, equal_nan=True)
            assert np.isfinite(pa[200:-10]).all()


def test_panel_scale_invariance_of_level_slopes():
    """Level panel: a symbol scaled by 100x gets exactly 100x its forecasts (symbol-normalised pooling)."""
    a = _panel_inputs(seed=3)
    b = _panel_inputs(seed=4)
    b100 = {k: v * (100.0 if k != "rq" else 1e4) for k, v in b.items()}
    for name in ("har", "harq"):
        Xa, p = vp.design(name, a)
        Xb, _ = vp.design(name, b)
        Xc, _ = vp.design(name, b100)
        ya, yb, yc = (vp.target(z["rv"], 1, False) for z in (a, b, b100))
        f1 = vp.panel_forecast([Xa, Xb], [ya, yb], 1, log=False, powers=p, start=200)
        f2 = vp.panel_forecast([Xa, Xc], [ya, yc], 1, log=False, powers=p, start=200)
        np.testing.assert_allclose(f2[1], 100 * f1[1], rtol=1e-7, equal_nan=True)
        np.testing.assert_allclose(f2[0], f1[0], rtol=1e-7, equal_nan=True)


def test_no_lookahead_regressions():
    a = _panel_inputs()
    b = _panel_inputs(seed=1)
    cut = 400
    for name in ("har", "harq", "loghar"):
        X, p = vp.design(name, a)
        Xb, _ = vp.design(name, b)
        a2 = {k: v.copy() for k, v in a.items()}
        for k in a2:
            a2[k][cut:] *= 30.0
        X2, _ = vp.design(name, a2)
        for h in (1, 5):
            lg = vp.is_log(name)
            y, y2, yb = vp.target(a["rv"], h, lg), vp.target(a2["rv"], h, lg), vp.target(b["rv"], h, lg)
            f = vp.per_asset_forecast(X, y, h, log=lg)
            g = vp.per_asset_forecast(X2, y2, h, log=lg)
            np.testing.assert_array_equal(f[:cut], g[:cut])
            pf = vp.panel_forecast([X, Xb], [y, yb], h, log=lg, powers=p)
            pg = vp.panel_forecast([X2, Xb], [y2, yb], h, log=lg, powers=p)
            np.testing.assert_array_equal(pf[:, :cut], pg[:, :cut])


def test_no_lookahead_ewma_arch_sv():
    rng = np.random.default_rng(5)
    r = 0.03 * rng.standard_t(5, 400)
    r2 = r.copy()
    r2[300:] *= 10
    np.testing.assert_array_equal(vp.ewma_daily(r)[:300], vp.ewma_daily(r2)[:300])
    fa = vp.arch_forecast(r, "gjr", min_days=200, refit_every=60)
    fb = vp.arch_forecast(r2, "gjr", min_days=200, refit_every=60)
    for h in (1, 5):
        np.testing.assert_allclose(fa[h][:300], fb[h][:300], rtol=1e-12, equal_nan=True)
        assert np.isfinite(fa[h][200:]).all()
    y, y2 = vp.fuller_log_r2(r), vp.fuller_log_r2(r2)
    sa = vp.sv_forecast(y, min_days=200, refit_every=60)
    sb = vp.sv_forecast(y2, min_days=200, refit_every=60)
    np.testing.assert_allclose(sa[1][:300], sb[1][:300], rtol=1e-12, equal_nan=True)


def test_sv_filter_recovers_known_ar1():
    r, h = vp.simulate_sv(6000, mu=-8.0, phi=0.95, s_eta=0.25, seed=11)
    y = np.log(r * r)
    pr = vp.sv_fit(y, "r2")
    assert abs(pr["phi"] - 0.95) < 0.03
    assert abs(pr["mu"] + 8.0) < 0.25
    assert abs(pr["s_eta"] - 0.25) < 0.08
    m, P, _ = vp.sv_filter(y, pr["mu"], pr["phi"], pr["s_eta"], pr["c"], pr["H"])
    # one-step predicted state tracks the true log variance
    assert np.corrcoef(m[:-1], h[1:])[0, 1] > 0.6


def test_arch_filter_matches_arch_package():
    from arch import arch_model

    rng = np.random.default_rng(2)
    r = 0.02 * rng.standard_t(6, 1500)
    for kind in ("garch", "gjr", "egarch"):
        pr = vp.fit_arch(r, kind)
        am = arch_model(r * 100, mean="Constant", dist="t", rescale=False, **vp.ARCH_SPECS[kind])
        res = am.fit(disp="off")
        cv = (res.conditional_volatility / 100) ** 2  # sigma^2_t (forecast of t made at t-1)
        mine = vp.arch_filter(r, pr, kind)  # sigma^2_{t+1}
        np.testing.assert_allclose(mine[500:-1], cv[501:], rtol=2e-3)


def test_garch_multi_step_analytic():
    pr = {"omega": 1e-5, "alpha": 0.1, "gamma": 0.0, "beta": 0.85}
    s1 = np.array([4e-4])
    f5 = vp.arch_multi(s1, pr, "garch", 5)
    v = [4e-4]
    for _ in range(4):
        v.append(1e-5 + 0.95 * v[-1])
    assert math.isclose(f5[0], np.mean(v), rel_tol=1e-12)


def test_mcs_excludes_bad_model():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((800, 1)) + 5
    L = np.hstack([base + 0.05 * rng.standard_normal((800, 1)), base + 0.05 * rng.standard_normal((800, 1)), base + 0.5 + 0.05 * rng.standard_normal((800, 1))])
    for stat in ("Tmax", "TR"):
        out = vp.mcs(L, alpha=0.1, block=10, B=500, stat=stat)
        assert not out["included"][2]
        assert out["included"][:2].all()
        assert out["pvalues"][2] < 0.1


def test_holm():
    adj = vp.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)
