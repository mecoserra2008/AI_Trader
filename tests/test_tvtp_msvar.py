"""Tests for ai_trader.research.tvtp_msvar (the reviewed copy of the user's TVTP MS-VAR). One regression test per fix
F1..F13 plus the pure pieces (filter, smoother, marginal effects, CCF)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t

from ai_trader.research.tvtp_msvar import (
    TVTPMSVAR, chol_psd, companion_ccf, cusum, emission_logpdf, hamilton_filter, kim_smoother, log_trans,
    marginal_effects, mixture_moments, permute_W, simulate_msvar, walk_forward)

A2 = [np.array([[0.0, 0.0], [0.5, 0.0]]), np.array([[0.0, 0.4], [0.0, 0.0]])]
S2 = [np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([[1.0, 0.7], [0.7, 1.0]])]
W2 = np.array([[[4.0, -3.0]], [[-4.0, -3.0]]])           # reference category 1: P01 = logistic(-4 + 3z), P10 = logistic(-4 - 3z)


def true_par(p: int = 1) -> dict:
    B = np.stack([np.vstack([np.zeros((1, 2)), A.T]) for A in A2])
    return dict(B=B, S=np.stack(S2), W=W2, logxi0=np.log([0.5, 0.5]), p=p)


def sim(T: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.98 * z[t - 1] + 0.2 * rng.standard_normal()
    y, s = simulate_msvar(true_par(), T, z[:, None], None, rng)
    return y, s, z


# ---------------------------------------------------------------- log-space reference (the user's original code)
def ref_filter(logeta, logP, logxi0):
    T, K = logeta.shape
    pred, filt, pll = np.empty((T, K)), np.empty((T, K)), np.empty(T)
    for t in range(T):
        lp = logxi0 if t == 0 else logsumexp(filt[t - 1][:, None] + logP[t], axis=0)
        la = lp + logeta[t]
        c = logsumexp(la)
        pred[t], filt[t], pll[t] = lp, la - c, c
    return pred, filt, pll


def ref_smooth(pred, filt, logP):
    T, K = filt.shape
    sm = np.empty((T, K))
    sm[-1] = filt[-1]
    joint = np.full((T, K, K), -np.inf)
    for t in range(T - 2, -1, -1):
        J = filt[t][:, None] + logP[t + 1] + (sm[t + 1] - pred[t + 1])[None, :]
        sm[t] = logsumexp(J, axis=1)
        joint[t + 1] = J
    return sm, joint


def test_filter_smoother_match_log_space_reference():
    rng = np.random.default_rng(0)
    T, K = 300, 3
    logeta = rng.normal(-2, 3, (T, K))
    W = rng.normal(0, 1, (K, K - 1, 2))
    Zc = np.c_[np.ones(T), rng.normal(size=T)]
    logP = log_trans(Zc, W)
    xi0 = np.array([0.2, 0.3, 0.5])
    pred, filt, pll = hamilton_filter(logeta, np.exp(logP), xi0)
    rp, rf, rl = ref_filter(logeta, logP, np.log(xi0))
    assert np.allclose(np.log(pred), rp, atol=1e-9) and np.allclose(np.log(filt), rf, atol=1e-9)
    assert np.allclose(pll, rl, atol=1e-9)
    sm, joint = kim_smoother(pred, filt, np.exp(logP))
    rs, rj = ref_smooth(rp, rf, logP)
    assert np.allclose(np.log(sm), rs, atol=1e-8)
    assert np.allclose(joint[1:], np.exp(rj[1:]), atol=1e-10)
    # joint marginals: sum_j joint[t][i, j] = sm[t-1][i], sum_i joint[t][i, j] = sm[t][j]
    assert np.allclose(joint[1:].sum(2), sm[:-1]) and np.allclose(joint[1:].sum(1), sm[1:])


def test_filter_handles_extreme_densities():
    logeta = np.array([[-1e4, 0.0], [0.0, -1e4], [-800.0, -805.0]])
    P = np.tile(np.array([[1.0 - 1e-300, 1e-300], [1e-300, 1.0 - 1e-300]]), (3, 1, 1))
    pred, filt, pll = hamilton_filter(logeta, P, np.array([0.5, 0.5]))
    assert np.all(np.isfinite(pll)) and np.allclose(filt.sum(1), 1)


def test_marginal_effects_match_finite_differences():
    rng = np.random.default_rng(3)
    K, d = 3, 3
    W = rng.normal(0, 1, (K, K - 1, d))
    z = np.array([0.3, -0.7])
    me = marginal_effects(W, np.r_[1.0, z])
    h = 1e-6
    for k in range(d - 1):
        dz = np.zeros(d - 1)
        dz[k] = h
        Pp = np.exp(log_trans(np.r_[1.0, z + dz][None], W)[0])
        Pm = np.exp(log_trans(np.r_[1.0, z - dz][None], W)[0])
        assert np.allclose(me[:, :, k], (Pp - Pm) / (2 * h), atol=1e-7)
    assert np.allclose(me.sum(1), 0, atol=1e-12)                       # rows of P sum to one


def test_ccf_matches_simulated_ccf():
    A = np.array([[0.1, 0.0], [0.5, 0.2]])
    S = np.array([[1.0, 0.3], [0.3, 1.0]])
    ccf = companion_ccf(A, S, 3)
    rng = np.random.default_rng(5)
    T = 200_000
    e = rng.multivariate_normal(np.zeros(2), S, T)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + e[t]
    y = y[1000:]
    for k in (0, 1, 2, -1):
        a, b = (y[k:], y[:len(y) - k]) if k >= 0 else (y[:k], y[-k:])
        emp = np.corrcoef(a[:, 1], b[:, 0])[0, 1]                       # corr(y_1,t+k, y_0,t)
        assert abs(emp - ccf[k][1, 0]) < 0.01
    assert ccf[1][1, 0] > 0.3 and abs(ccf[1][0, 1]) < 0.15              # y0 leads y1


def test_ccf_nan_when_unstable():                                        # F10
    ccf = companion_ccf(np.array([[1.01, 0.0], [0.0, 0.5]]), np.eye(2), 2)
    assert np.isnan(ccf[0]).all()


def test_permute_W_permutes_transition_matrix():                         # F3
    rng = np.random.default_rng(7)
    K, d = 3, 2
    W = rng.normal(0, 1, (K, K - 1, d))
    perm = np.array([2, 0, 1])
    W2p = permute_W(W, perm)
    zc = np.array([[1.0, 0.4]])
    P = np.exp(log_trans(zc, W)[0])
    P2 = np.exp(log_trans(zc, W2p)[0])
    assert np.allclose(P2, P[perm][:, perm])


def test_align_to_undoes_a_label_permutation():                           # F3
    y, s, z = sim(1500, 2)
    m = TVTPMSVAR(K=2, p=1, l2=0.1, n_init=2, seed=1).fit(y, z[:, None])
    ll = m.ll
    ref = {k: v.copy() for k, v in m.params().items()}
    m.permute(np.array([1, 0]))
    assert np.isclose(m.ll, ll, atol=1e-6)                               # likelihood invariant to labels
    perm = m.align_to(ref)
    assert list(perm) == [1, 0] and np.allclose(m.B, ref["B"]) and np.allclose(m.W, ref["W"], atol=1e-10)


def test_expected_corr_next_uses_exog_at_origin():                       # F2
    rng = np.random.default_rng(11)
    T = 1200
    ect = np.zeros(T)
    for t in range(1, T):
        ect[t] = 0.95 * ect[t - 1] + rng.normal(0, 0.3)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = np.array([0.0, -0.8]) * ect[t - 1] + rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
    m = TVTPMSVAR(K=1, p=1).fit(y, None, ect[:, None])
    with pytest.raises(ValueError):
        m.expected_corr_next()
    x_new = np.r_[1.0, y[-1], 2.0]
    pj, R = m.expected_corr_next(exog_now=2.0)
    pj2, R2 = m.expected_corr_next(x_next=x_new)
    assert np.allclose(R, R2)
    # the design row for the last fitted observation carries ect_{T-2}; the forecast must not reuse it
    assert not np.isclose(m._X[-1, -1], ect[-1])


def test_mixture_moments_two_regimes():
    w = np.array([[0.3, 0.7]])
    mu = np.array([[[1.0, 0.0], [-1.0, 2.0]]])
    S = np.stack([np.eye(2), 2 * np.eye(2)])
    m, C = mixture_moments(w, mu, S)
    rng = np.random.default_rng(0)
    N = 400_000
    k = rng.random(N) < 0.3
    x = np.where(k[:, None], rng.normal(size=(N, 2)) + mu[0, 0], rng.normal(size=(N, 2)) * np.sqrt(2) + mu[0, 1])
    assert np.allclose(m[0], x.mean(0), atol=0.01) and np.allclose(C[0], np.cov(x.T), atol=0.03)


def test_student_t_logpdf_matches_scipy():                               # F12
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(50, 2))
    X = np.ones((50, 1))
    B = np.array([[[0.1, -0.2]]])
    S = np.array([[[1.5, 0.4], [0.4, 0.8]]])
    lp, _ = emission_logpdf(Y, X, B, S, nu=5.0)
    assert np.allclose(lp[:, 0], multivariate_t(loc=[0.1, -0.2], shape=S[0], df=5.0).logpdf(Y))
    lg, _ = emission_logpdf(Y, X, B, S, nu=None)
    assert np.allclose(lg[:, 0], multivariate_normal([0.1, -0.2], S[0]).logpdf(Y))


def test_chol_psd_singular():                                            # F5
    L = chol_psd(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert np.all(np.isfinite(L))


def test_degenerate_regime_stays_finite():                               # F4
    rng = np.random.default_rng(4)
    y = rng.normal(size=(800, 2))                                        # one regime; ask for three
    m = TVTPMSVAR(K=3, p=1, n_init=2, seed=0, max_iter=100).fit(y)
    assert np.isfinite(m.ll)
    assert min(np.linalg.eigvalsh(S).min() for S in m.S) > 1e-3


def test_K1_is_ols_var_and_ll_is_final():                                # F7, F8
    y, _, _ = sim(1000, 3)
    m = TVTPMSVAR(K=1, p=2, ridge=0.0, cov_prior=0.0).fit(y)
    Yt, X, _ = m.design(y)
    B = np.linalg.lstsq(X, Yt, rcond=None)[0]
    assert np.allclose(m.B[0], B, atol=1e-8)
    E = Yt - X @ B
    ll = multivariate_normal(np.zeros(2), E.T @ E / len(E)).logpdf(E).sum()
    assert np.isclose(m.ll, ll, rtol=1e-6)
    assert m.n_params == B.size + 3


def test_em_ll_nondecreasing_and_ll_consistent():                        # F7
    y, _, z = sim(1500, 6)
    m = TVTPMSVAR(K=2, p=1, l2=1e-3, cov_prior=0.0, n_init=1, seed=2).fit(y, z[:, None])
    d = np.diff(m.ll_path)
    assert (d > -1e-3 * np.abs(np.array(m.ll_path[1:]))).all() or (d > -0.5).all()
    f = m.filter_design(m._Yt, m._X, m._Zc)
    assert np.isclose(f["pll"].sum(), m.ll)


def test_mstep_trans_recovers_logit_from_known_states():                 # Zc[1:] <-> joint[1:] alignment
    y, s, z = sim(20_000, 8)
    m = TVTPMSVAR(K=2, p=1, l2=1e-4)
    m.K = 2
    Yt, X, Zc = m.design(y, z[:, None])
    st = s[1:]
    joint = np.zeros((len(st), 2, 2))
    joint[np.arange(1, len(st)), st[:-1], st[1:]] = 1.0                  # one-hot transitions t-1 -> t
    W = m._mstep_trans(Zc, joint, np.zeros((2, 1, 2)))
    assert np.allclose(W, W2, atol=0.6)
    # misaligning by one row destroys the z effect
    Wbad = m._mstep_trans(np.r_[Zc[1:], Zc[-1:]], joint, np.zeros((2, 1, 2)))
    assert np.abs(Wbad[:, 0, 1] - W2[:, 0, 1]).max() > np.abs(W[:, 0, 1] - W2[:, 0, 1]).max()


def test_walk_forward_has_no_look_ahead():                               # F1
    y, _, z = sim(1400, 9)
    fac = lambda: TVTPMSVAR(K=2, p=1, l2=0.1, n_init=1, seed=0, max_iter=30)  # noqa: E731
    m = fac()
    Yt, X, Zc = m.design(y, z[:, None])
    wf = walk_forward(fac, Yt, X, Zc, origins=[800, 1100])
    y2 = y.copy()
    y2[1200:] += 5.0 * np.random.default_rng(0).normal(size=y2[1200:].shape)
    Yt2, X2, Zc2 = m.design(y2, z[:, None])
    wf2 = walk_forward(fac, Yt2, X2, Zc2, origins=[800, 1100])
    t_ok = 1199 - 1                                                      # design row for y_1198 (p = 1)
    assert np.allclose(wf["pll"][800:t_ok], wf2["pll"][800:t_ok])
    assert np.allclose(wf["pred"][800:t_ok + 1], wf2["pred"][800:t_ok + 1])
    assert np.isnan(wf["pll"][:800]).all() and np.isfinite(wf["pll"][800:]).all()


def test_cusum_out_of_sample():                                          # F9
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 1000)
    x[700:760] -= 3.0
    u, c, alarm = cusum(x, burn=200, halflife=200, kappa=0.5, h=5.0)
    assert not alarm[:200].any()
    first = np.flatnonzero(alarm)
    assert len(first) and 700 <= first[first >= 700][0] < 710
    assert alarm[:700].sum() <= 2


def test_trans_se_and_simulate():                                        # F13
    y, _, z = sim(2000, 10)
    m = TVTPMSVAR(K=2, p=1, l2=0.1, n_init=1, seed=0).fit(y, z[:, None])
    se = m.trans_se()
    assert se.shape == m.W.shape and np.all(np.isfinite(se)) and np.all(se > 0)
    ys, ss = m.simulate(500, z[:500, None], rng=np.random.default_rng(0))
    assert ys.shape == (500, 2) and set(np.unique(ss)) <= {0, 1}


def test_blocks_init_recovers_synthetic():                               # F6 (the user's own synthetic check)
    y, s, z = sim(3000, 1)
    m = TVTPMSVAR(K=2, p=1, l2=0.1, n_init=2, seed=3).fit(y, z[:, None])
    est = m.prob_filtered.argmax(1)
    acc = max((est == s[1:]).mean(), (est != s[1:]).mean())
    assert acc > 0.9


# ---------------------------------------------------------------- tvtp_study pure pieces (point in time, costs)
from ai_trader.research import tvtp_study as st  # noqa: E402


def test_ewma_var_is_causal():
    rng = np.random.default_rng(0)
    r = rng.normal(size=300)
    v = st.ewma_var(r, 24)
    r2 = r.copy()
    r2[200:] *= 10
    v2 = st.ewma_var(r2, 24)
    assert np.allclose(v[:201], v2[:201]) and not np.isclose(v[201], v2[201])


def test_rolling_ect_matches_direct_ols_and_is_causal():
    rng = np.random.default_rng(1)
    ls = np.cumsum(rng.normal(0, 0.01, 400)) + 3.0
    lg = 0.8 * ls + rng.normal(0, 0.005, 400) + 5.0
    e = st.rolling_ect(lg, ls, 100)
    t = 250
    x, y = ls[t - 99:t + 1], lg[t - 99:t + 1]
    b, a = np.polyfit(x, y, 1)
    res = y - a - b * x
    assert np.isclose(e[t], res[-1] / np.sqrt((res ** 2).sum() / 98), rtol=1e-6)
    lg2 = lg.copy()
    lg2[t + 1:] += 1.0
    assert np.isclose(st.rolling_ect(lg2, ls, 100)[t], e[t]) and np.isnan(e[:99]).all()
    ee = st.rolling_ect(lg, ls, None, min_n=20)
    assert np.isnan(ee[:19]).all() and np.isfinite(ee[19:]).all()


def test_expanding_z_causal():
    x = np.r_[np.arange(30.0), np.nan, 100.0]
    z = st.expanding_z(x, 5)
    assert np.isnan(z[:4]).all() and np.isnan(z[30])
    assert np.isclose(z[10], (10 - np.mean(x[:11])) / np.std(x[:11], ddof=1))


def test_build_design_matches_model_design_and_breaks_at_gaps():
    y, _, z = sim(300, 4)
    Xe = np.random.default_rng(0).normal(size=(300, 1))
    m = TVTPMSVAR(K=2, p=2)
    Yt, X, Zc = m.design(y, z[:, None], Xe)
    Yb, Xb, Zb, rows = st.build_design(y, z[:, None], Xe, np.zeros(300, int), 2)
    assert np.allclose(Yt, Yb) and np.allclose(X, Xb) and np.allclose(Zc, Zb) and rows[0] == 2
    seg = np.r_[np.zeros(150, int), np.ones(150, int)]
    _, _, _, rows2 = st.build_design(y, z[:, None], Xe, seg, 2)
    assert 150 not in rows2 and 151 not in rows2 and 152 in rows2


def test_trade_R_costs():
    ds = {"r": np.array([[0.0, 0.0], [0.001, 0.002]]), "sig": np.array([[0.01, 0.01], [0.01, 0.02]]),
          "half_spread_bp": np.array([0.0, 1.0]), "lat_r": None}
    asset, d = np.array([1]), np.array([1.0])
    R, G = st.trade_R(ds, np.array([1]), asset, d, "desk")
    assert np.isclose(G[0], 0.1) and np.isclose(R[0], 0.1 - (2.0 + 1.3 + 5.5 + 1.0) / 1e4 / 0.02)
    R, _ = st.trade_R(ds, np.array([1]), asset, d, "taker")
    assert np.isclose(R[0], 0.1 - (11.0 + 2.0) / 1e4 / 0.02)
    R, _ = st.trade_R(ds, np.array([1]), np.array([-1]), d, "taker")
    assert np.isnan(R[0])


def test_rule_positions_direction_and_threshold():
    rl = {"lead0": (0, 0.5), "lead1": (1, -0.3)}
    pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]])
    y_now = np.array([[1.0, -2.0], [1.0, -2.0], [1.0, 1.0]])
    a, d = st.rule_positions(pred, y_now, rl, 0.6)
    assert list(a) == [1, 0, -1] and d[0] == 1.0 and d[1] == 1.0          # -0.3 * -2 > 0


def test_qlike_minimised_at_truth():
    rng = np.random.default_rng(2)
    S = np.array([[1.0, 0.6], [0.6, 1.0]])
    y = rng.multivariate_normal([0, 0], S, 20000)
    H = np.tile(S, (len(y), 1, 1))
    H2 = np.tile(np.array([[1.0, 0.2], [0.2, 1.0]]), (len(y), 1, 1))
    assert st.qlike(H, y).mean() < st.qlike(H2, y).mean()
