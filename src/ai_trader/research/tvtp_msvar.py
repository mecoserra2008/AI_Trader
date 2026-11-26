"""TVTP Markov-switching VAR for state-dependent lead-lag (gold / silver, BTC / alts).

Original methodology and code: the user (Americo Serra), `tvtp_msvar.py` at the repo root (2026-11-26, untracked, left
unchanged). This module is a reviewed copy (agent TVTP, 2026-11-26); every change is listed below as F1..F13 and has a
regression test in tests/test_tvtp_msvar.py.

Model (unchanged)
-----------------
y_t in R^n (log returns, vol-standardised), regime S_t in {0..K-1}:
    y_t = c_j + sum_{l=1..p} A_{j,l} y_{t-l} [+ alpha_j * ect_{t-1}] + e_t,   e_t ~ N(0, Sigma_j)   (or Student-t, F12)
    P(S_t = j | S_{t-1} = i, z_{t-1}) = softmax_j( W[i, j] . [1, z_{t-1}] ),  W[i, K-1] = 0
EM: Hamilton filter, Kim smoother, weighted least squares per regime, L2-penalised weighted multinomial logit for W.

Fixes and additions (each documented where it lives)
---------------------------------------------------
F1  look-ahead. `prob_smoothed`, the in-sample `pred_ll_` and `break_cusum()` on it use parameters fit on the whole
    sample (and, for smoothed, future data): they are in-sample diagnostics only. Forecasts and signals must come from
    `filter_design()` / `walk_forward()`: parameters fit on rows < origin, causal Hamilton filter, one-step predictive
    regime probabilities `pred[t] = P(S_t | F_{t-1})`.
F2  `expected_corr_next` built the exogenous part of x_{T+1} from `self._X[-1, 1+n*p:]` = ect_{T-1}; forecasting y_{T+1}
    needs ect_T. It now takes `exog_now` explicitly (raises if the model has an exogenous block and it is missing), and
    uses the lags y_T..y_{T-p+1} (unchanged, they were right).
F3  label switching across refits: `align_to(ref)` permutes regimes (B, Sigma, W, xi0) by Hungarian matching on the
    regime parameters; `walk_forward` aligns every refit to the previous one. Permuting W re-normalises the reference
    category (full logits, permute rows and columns, subtract the new reference column).
F4  degenerate regimes: a regime with ~0 weight got Sigma = 0 + 1e-10 I (unbounded likelihood spike). Sigma_j is now the
    MAP under an inverse-Wishart-type prior with `cov_prior` pseudo-observations at the pooled covariance:
    Sigma_j = (E' W E + kappa * S0) / (sum w + kappa). `min_occupancy` (smallest expected regime share) is reported.
F5  Cholesky failures: jitter retry (1e-10 .. 1e-4 x trace/n) instead of an exception.
F6  initialisation: the original random Dirichlet responsibilities per row make every regime's first M-step almost the
    pooled fit (weights average out over T rows), so EM starts near a symmetric saddle. `init="blocks"` (default) assigns
    random contiguous blocks to regimes; `init="dirichlet"` keeps the original. Compared in the Monte Carlo.
F7  `ll` was the log-likelihood of the parameters *before* the last M-step while B, S, W were after it; `ll` is now
    recomputed at the final parameters. The BIC count adds the K-1 initial-distribution parameters.
F8  K=1 (plain VAR(p), the baseline) and Z=None (constant transition probabilities) are supported by the same code.
F9  CUSUM: `cusum()` is a pure function of an out-of-sample predictive log-score series; mean / variance start from a
    burn-in that is not scored, and alarms reset. The original initialised from the first `halflife` scores it then scored.
F10 `regime_ccf` returns NaN when the regime's companion matrix is not stable (spectral radius >= 1); the Lyapunov
    solution is meaningless there. The CCF assumes the regime persists forever (no switching): documented caveat.
F11 speed: filter / smoother in scaled probability space (one small matvec per row), joint smoothed transition
    probabilities vectorised. Same numbers as the log-space version (tested).
F12 optional multivariate Student-t emissions (`nu`, fixed), ECM weights u = (nu+n)/(nu+delta). House rule
    ARCHITECTURE 1.2; Gaussian (the user's choice) stays the default.
F13 `trans_se()`: Wald standard errors of W from the penalised weighted-logit Hessian at the soft counts (ignores
    regime uncertainty, so expected to under-cover; measured in the Monte Carlo). `simulate()` for Monte Carlo and
    parametric bootstrap.

Checked and found correct (kept): design lags and z_{t-1} alignment, the Kim smoother joint
P(S_t=i, S_{t+1}=j | Y) = filt_t(i) P_{t+1}(i,j) sm_{t+1}(j) / pred_{t+1}(j), the M-step alignment of Zc[1:] with
joint[1:] (joint[t] is the t-1 -> t transition driven by Zc[t] = z_{t-1}), the logit gradient, the marginal-effect formula,
the CCF Lyapunov construction and its sign convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_triangular
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import gammaln, log_softmax

LOG2PI = math.log(2 * math.pi)


# ====================================================================== pure pieces


def chol_psd(S: np.ndarray) -> np.ndarray:
    """F5: Cholesky with a jitter ladder."""
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        n = S.shape[0]
        sc = max(float(np.trace(S)) / n, 1e-12)
        for j in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            try:
                return np.linalg.cholesky(S + j * sc * np.eye(n))
            except np.linalg.LinAlgError:
                continue
        raise


def log_trans(Zc: np.ndarray, W: np.ndarray) -> np.ndarray:
    """log P(S_t=j | S_{t-1}=i, z) for each row of Zc; W is (K, K-1, d) with reference category K-1. Returns (T,K,K)."""
    T, K = Zc.shape[0], W.shape[0]
    if K == 1:
        return np.zeros((T, 1, 1))
    logits = np.einsum("td,ijd->tij", Zc, W)
    logits = np.concatenate([logits, np.zeros(logits.shape[:2] + (1,))], axis=2)
    return log_softmax(logits, axis=2)


def hamilton_filter(logeta: np.ndarray, P: np.ndarray, xi0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal filter. logeta (T,K) emission log densities, P (T,K,K) transition probabilities (P[t] drives t-1 -> t,
    P[0] unused), xi0 prior over S_0. Returns pred (T,K) = P(S_t|F_{t-1}), filt (T,K) = P(S_t|F_t), pll (T,) =
    log p(y_t|F_{t-1}). F11: scaled probability space."""
    T, K = logeta.shape
    m = logeta.max(axis=1)
    eta = np.exp(logeta - m[:, None])
    pred, filt, c = np.empty((T, K)), np.empty((T, K)), np.empty(T)
    f = np.asarray(xi0, float)
    floor = 1e-300
    for t in range(T):
        pr = f if t == 0 else f @ P[t]
        a = pr * eta[t]
        s = a.sum()
        if not s > floor:                                   # every regime's density underflowed relative to its prior
            a = (pr + 1e-12) * eta[t]
            s = a.sum()
        f = a / s
        pred[t], filt[t], c[t] = pr, f, s
    return pred, filt, np.log(c) + m


def kim_smoother(pred: np.ndarray, filt: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kim smoother. Returns sm (T,K) = P(S_t|F_T) and joint (T,K,K) with joint[t][i,j] = P(S_{t-1}=i, S_t=j | F_T)
    for t >= 1 (joint[0] = 0)."""
    T, K = filt.shape
    sm = np.empty((T, K))
    sm[-1] = filt[-1]
    pr = np.maximum(pred, 1e-300)
    for t in range(T - 2, -1, -1):
        sm[t] = filt[t] * (P[t + 1] @ (sm[t + 1] / pr[t + 1]))
    joint = np.zeros((T, K, K))
    joint[1:] = filt[:-1, :, None] * P[1:] * (sm[1:] / pr[1:])[:, None, :]
    return sm, joint


def emission_logpdf(Y: np.ndarray, X: np.ndarray, B: np.ndarray, S: np.ndarray, nu: float | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-regime log density (T,K) and Mahalanobis distances (T,K)."""
    T, n = Y.shape
    K = B.shape[0]
    out, dist = np.empty((T, K)), np.empty((T, K))
    for j in range(K):
        E = Y - X @ B[j]
        L = chol_psd(S[j])
        sol = solve_triangular(L, E.T, lower=True)
        d = (sol ** 2).sum(0)
        half_logdet = np.log(np.diag(L)).sum()
        if nu is None:
            out[:, j] = -0.5 * d - half_logdet - 0.5 * n * LOG2PI
        else:
            out[:, j] = (gammaln((nu + n) / 2) - gammaln(nu / 2) - 0.5 * n * math.log(nu * math.pi) - half_logdet
                         - 0.5 * (nu + n) * np.log1p(d / nu))
        dist[:, j] = d
    return out, dist


def marginal_effects(W: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """dP_ij/dz_k = P_ij (b_ijk - sum_m P_im b_imk) at zc = [1, z]. Returns (K,K,d-1)."""
    K = W.shape[0]
    P = np.exp(log_trans(zc[None, :], W)[0])
    Wf = np.concatenate([W, np.zeros((K, 1, W.shape[2]))], axis=1)[:, :, 1:]
    bbar = np.einsum("ij,ijk->ik", P, Wf)
    return P[:, :, None] * (Wf - bbar[:, None, :])


def full_logits(W: np.ndarray) -> np.ndarray:
    K = W.shape[0]
    return np.concatenate([W, np.zeros((K, 1, W.shape[2]))], axis=1)


def permute_W(W: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """New regime a = old regime perm[a]. F3: permute rows and columns of the full logits, re-normalise the reference."""
    Wf = full_logits(W)[perm][:, perm]
    Wf = Wf - Wf[:, -1:, :]
    return Wf[:, :-1, :]


def companion_ccf(A: np.ndarray, Sigma: np.ndarray, max_lag: int = 5) -> dict[int, np.ndarray]:
    """Implied cross-correlation of a stable VAR(p): out[k][a,b] = corr(y_{a,t+k}, y_{b,t}). A is (n, n*p) = [A1..Ap].
    F10: NaN if the companion matrix is not stable."""
    n = Sigma.shape[0]
    p = A.shape[1] // n
    F = np.zeros((n * p, n * p))
    F[:n] = A
    if p > 1:
        F[n:, :-n] = np.eye(n * (p - 1))
    if np.max(np.abs(np.linalg.eigvals(F))) >= 0.999:
        return {k: np.full((n, n), np.nan) for k in range(-max_lag, max_lag + 1)}
    Q = np.zeros_like(F)
    Q[:n, :n] = Sigma
    G0 = solve_discrete_lyapunov(F, Q)
    sd = np.sqrt(np.diag(G0)[:n])
    out, Fk = {}, np.eye(n * p)
    for k in range(0, max_lag + 1):
        Gk = (Fk @ G0)[:n, :n]
        out[k] = Gk / np.outer(sd, sd)
        out[-k] = out[k].T
        Fk = F @ Fk
    return dict(sorted(out.items()))


def cusum(score: np.ndarray, burn: int = 168, halflife: int = 168, kappa: float = 0.5, h: float = 5.0,
          refractory: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F9: one-sided CUSUM on standardised drops of an OUT-OF-SAMPLE predictive log score. Mean / variance start from the
    first `burn` scores (not scored), then EWMA-update with information up to t-1. Alarm when c_t > h; c resets after an
    alarm and alarms are suppressed for `refractory` rows. Returns (u, c, alarm bool)."""
    x = np.asarray(score, float)
    T = len(x)
    u, c, alarm = np.zeros(T), np.zeros(T), np.zeros(T, bool)
    if T <= burn + 1:
        return u, c, alarm
    a = 1 - 0.5 ** (1 / halflife)
    m, v = float(np.mean(x[:burn])), float(np.var(x[:burn])) + 1e-12
    cc, last = 0.0, -10 ** 9
    for t in range(burn, T):
        u[t] = (x[t] - m) / math.sqrt(v)
        cc = max(0.0, cc - u[t] - kappa)
        if cc > h and t - last > refractory:
            alarm[t], last, cc = True, t, 0.0
        c[t] = cc
        d = x[t] - m
        m += a * d
        v = (1 - a) * (v + a * d * d)
    return u, c, alarm


# ====================================================================== model


class TVTPMSVAR:
    def __init__(self, K: int = 3, p: int = 2, l2: float = 1.0, ridge: float = 1e-6, max_iter: int = 300,
                 tol: float = 1e-7, n_init: int = 5, seed: int = 0, min_var: float = 1e-10, cov_prior: float = 2.0,
                 init: str = "blocks", nu: float | None = None, verbose: bool = False):
        self.K, self.p, self.l2, self.ridge = K, p, l2, ridge
        self.max_iter, self.tol, self.n_init = max_iter, tol, n_init
        self.rng = np.random.default_rng(seed)
        self.min_var, self.cov_prior, self.init, self.nu, self.verbose = min_var, cov_prior, init, nu, verbose
        if init not in ("blocks", "dirichlet"):
            raise ValueError("init must be 'blocks' or 'dirichlet'")

    # ---------- design ----------
    def design(self, Y: np.ndarray, Z: np.ndarray | None = None, X_extra: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row t (t = p..T-1 of the inputs): Yt = y_t, X = [1, y_{t-1}..y_{t-p}, x_extra_{t-1}], Zc = [1, z_{t-1}]."""
        Y = np.asarray(Y, float)
        T, n = Y.shape
        p = self.p
        lags = [Y[p - l:T - l] for l in range(1, p + 1)]
        X = np.hstack([np.ones((T - p, 1))] + lags)
        if X_extra is not None:
            Xe = np.asarray(X_extra, float).reshape(T, -1)
            X = np.hstack([X, Xe[p - 1:T - 1]])
        if Z is None:
            Zc = np.ones((T - p, 1))
        else:
            Zz = np.asarray(Z, float).reshape(T, -1)
            Zc = np.hstack([np.ones((T - p, 1)), Zz[p - 1:T - 1]])
        return Y[p:], X, Zc

    # ---------- M-steps ----------
    def _mstep_regimes(self, Y: np.ndarray, X: np.ndarray, w: np.ndarray, u: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
        K, n, k = self.K, Y.shape[1], X.shape[1]
        B, S = np.empty((K, k, n)), np.empty((K, n, n))
        S0 = self._S0
        for j in range(K):
            wj = w[:, j] if u is None else w[:, j] * u[:, j]
            XtW = X.T * wj
            B[j] = np.linalg.solve(XtW @ X + self.ridge * np.eye(k), XtW @ Y)
            E = Y - X @ B[j]
            num = (E.T * wj) @ E + self.cov_prior * S0                       # F4
            S[j] = num / max(w[:, j].sum() + self.cov_prior, 1e-12) + self.min_var * np.eye(n)
        return B, S

    def _row_objective(self, Zs: np.ndarray, N: np.ndarray, pen: np.ndarray) -> Callable:
        K, d = self.K, Zs.shape[1]
        Ns = N.sum(1, keepdims=True)

        def f(wf: np.ndarray) -> tuple[float, np.ndarray]:
            Wi = wf.reshape(K - 1, d)
            lg = np.hstack([Zs @ Wi.T, np.zeros((len(Zs), 1))])
            lp = log_softmax(lg, axis=1)
            Pr = np.exp(lp)
            obj = -(N * lp).sum() + 0.5 * (pen * Wi ** 2).sum()
            G = -((N[:, :K - 1] - Ns * Pr[:, :K - 1]).T @ Zs) + pen * Wi
            return obj, G.ravel()
        return f

    def _pen(self, d: int) -> np.ndarray:
        pen = np.full(d, self.l2)
        pen[0] = 1e-4 * self.l2                                               # barely penalise the intercept
        return pen

    def _mstep_trans(self, Zc: np.ndarray, joint: np.ndarray, W: np.ndarray) -> np.ndarray:
        K, d = self.K, Zc.shape[1]
        if K == 1:
            return W
        Wn = W.copy()
        Zs, Js = Zc[1:], joint[1:]                     # joint[t] is the (t-1 -> t) transition, driven by Zc[t] = z_{t-1}
        pen = self._pen(d)
        for i in range(K):
            r = minimize(self._row_objective(Zs, Js[:, i, :], pen), W[i].ravel(), jac=True, method="L-BFGS-B")
            Wn[i] = r.x.reshape(K - 1, d)
        return Wn

    # ---------- E-step ----------
    def _estep(self, Y, X, Zc, B, S, W, xi0):
        P = np.exp(log_trans(Zc, W))
        logeta, dist = emission_logpdf(Y, X, B, S, self.nu)
        pred, filt, pll = hamilton_filter(logeta, P, xi0)
        return P, logeta, dist, pred, filt, pll

    # ---------- init ----------
    def _init_w(self, T: int) -> np.ndarray:
        K = self.K
        if self.init == "dirichlet":
            return self.rng.dirichlet(np.ones(K), size=T)                     # the original
        L = max(10, T // (8 * K))                                             # F6
        w = np.full((T, K), 0.1 / max(K - 1, 1))
        starts = np.arange(0, T, L)
        labs = self.rng.integers(0, K, len(starts))
        labs[:K] = self.rng.permutation(K)[: min(K, len(labs))]               # every regime gets a block
        for s0, lab in zip(starts, labs):
            w[s0:s0 + L] = 0.1 / max(K - 1, 1)
            w[s0:s0 + L, lab] = 0.9
        return w / w.sum(1, keepdims=True)

    def _init_W(self, d: int) -> np.ndarray:
        K = self.K
        W = np.zeros((K, max(K - 1, 0), d))
        if K > 1:
            W[:, :, 0] = np.where(np.arange(K)[:, None] == np.arange(K - 1)[None, :], 3.0, 0.0)
            W[K - 1, :, 0] = -3.0                                             # sticky diagonal
        return W

    # ---------- EM ----------
    def _em(self, Y, X, Zc, start: dict | None = None) -> dict[str, Any]:
        T, K, d = len(Y), self.K, Zc.shape[1]
        if start is None:
            w = self._init_w(T)
            B, S = self._mstep_regimes(Y, X, w)
            W = self._init_W(d)
            xi0 = np.full(K, 1.0 / K)
        else:
            B, S, W, xi0 = start["B"].copy(), start["S"].copy(), start["W"].copy(), np.exp(start["logxi0"])
        ll_old, path, conv = -np.inf, [], False
        it = 0
        for it in range(self.max_iter):
            P, logeta, dist, pred, filt, pll = self._estep(Y, X, Zc, B, S, W, xi0)
            ll = float(pll.sum())
            path.append(ll)
            sm, joint = kim_smoother(pred, filt, P)
            u = None if self.nu is None else (self.nu + Y.shape[1]) / (self.nu + dist)
            B, S = self._mstep_regimes(Y, X, sm, u)
            W = self._mstep_trans(Zc, joint, W)
            xi0 = np.clip(sm[0], 1e-8, None)
            xi0 = xi0 / xi0.sum()
            if abs(ll - ll_old) < self.tol * max(1.0, abs(ll)):
                conv = True
                break
            ll_old = ll
        ll = float(self._estep(Y, X, Zc, B, S, W, xi0)[5].sum())               # F7: ll at the final parameters
        return dict(B=B, S=S, W=W, logxi0=np.log(xi0), ll=ll, iters=it + 1, converged=conv, ll_path=path)

    def fit(self, Y, Z=None, X_extra=None, start: dict | None = None) -> TVTPMSVAR:
        Y = np.asarray(Y, float)
        self.n = Y.shape[1]
        self.has_exog = X_extra is not None
        Yt, X, Zc = self.design(Y, Z, X_extra)
        self.fit_design(Yt, X, Zc, start=start)
        self._Zlast = None if Z is None else np.asarray(Z, float).reshape(len(Y), -1)[-1]
        return self

    def fit_design(self, Yt: np.ndarray, X: np.ndarray, Zc: np.ndarray, start: dict | None = None) -> TVTPMSVAR:
        self.n = Yt.shape[1]
        self.n_lagcols = 1 + self.n * self.p
        if not hasattr(self, "has_exog"):
            self.has_exog = X.shape[1] > self.n_lagcols
        self._S0 = np.cov(Yt.T) if len(Yt) > 2 else np.eye(self.n)
        runs = []
        if start is not None:                                                  # warm start (walk-forward refits)
            runs.append(self._em(Yt, X, Zc, start))
        n_rand = self.n_init if self.K > 1 else (0 if runs else 1)
        for _ in range(n_rand):
            runs.append(self._em(Yt, X, Zc))
        best = max(runs, key=lambda r: r["ll"])
        self.init_lls = [r["ll"] for r in runs]
        self.__dict__.update(best)
        self._Yt, self._X, self._Zc = Yt, X, Zc
        self._post()
        return self

    def params(self) -> dict[str, np.ndarray]:
        return dict(B=self.B, S=self.S, W=self.W, logxi0=self.logxi0)

    def _post(self) -> None:
        P, logeta, dist, pred, filt, pll = self._estep(self._Yt, self._X, self._Zc, self.B, self.S, self.W,
                                                       np.exp(self.logxi0))
        sm, _ = kim_smoother(pred, filt, P)
        # IN-SAMPLE ONLY (F1): parameters fit on the whole sample; smoothed also uses future data.
        self.pred_, self.filt_, self.pred_ll_ = np.log(np.maximum(pred, 1e-300)), np.log(np.maximum(filt, 1e-300)), pll
        self.smooth_ = np.log(np.maximum(sm, 1e-300))
        self.prob_filtered, self.prob_smoothed = filt, sm
        self.ll = float(pll.sum())
        self.min_occupancy = float(sm.mean(0).min())
        self.n_params = int(self.B.size + self.K * self.n * (self.n + 1) // 2 + self.W.size + (self.K - 1))   # F7
        self.bic = -2 * self.ll + self.n_params * np.log(len(self._Yt))

    # ---------- causal filtering with frozen parameters (F1) ----------
    def filter_design(self, Yt: np.ndarray, X: np.ndarray, Zc: np.ndarray) -> dict[str, np.ndarray]:
        """Causal pass with the fitted parameters over any design rows (typically the full history up to the end of the
        test block, starting at the fit's first row). pred[t] = P(S_t | F_{t-1}) is the one-step regime forecast;
        pll[t] = log p(y_t | F_{t-1}); mean / cov = one-step predictive mixture moments of y_t."""
        P, logeta, dist, pred, filt, pll = self._estep(Yt, X, Zc, self.B, self.S, self.W, np.exp(self.logxi0))
        mu = np.einsum("tk,jkn->tjn", X, self.B)                               # (T,K,n) regime means
        mean, cov = mixture_moments(pred, mu, self.S * self._cov_scale())
        return dict(pred=pred, filt=filt, pll=pll, mu=mu, mean=mean, cov=cov)

    def _cov_scale(self) -> float:
        return 1.0 if self.nu is None else self.nu / (self.nu - 2.0)

    # ---------- interpretation ----------
    def transition_matrix(self, z=None) -> np.ndarray:
        zc = np.r_[1.0, np.atleast_1d(z)] if self.W.shape[2] > 1 else np.array([1.0])
        return np.exp(log_trans(zc[None, :], self.W)[0])

    def transition_marginal_effects(self, z) -> np.ndarray:
        """dP_ij / dz_k = P_ij (b_ijk - sum_m P_im b_imk). Returns (K,K,d-1)."""
        return marginal_effects(self.W, np.r_[1.0, np.atleast_1d(z)])

    def regime_ccf(self, j: int, max_lag: int = 5) -> dict[int, np.ndarray]:
        """corr(y_{a,t+k}, y_{b,t}), k=-max_lag..max_lag, if regime j persisted forever (AR part only, no exogenous
        block). With y = [gold, silver], mass at positive k in [1, 0] means gold leads silver in regime j."""
        n, p = self.n, self.p
        A = self.B[j][1:1 + n * p].T
        return companion_ccf(A, self.S[j], max_lag)

    def lag_matrix(self, j: int, lag: int = 1) -> np.ndarray:
        n = self.n
        return self.B[j][1 + n * (lag - 1):1 + n * lag].T                    # A_{j,lag}[a, b]: effect of y_b,t-lag on y_a,t

    def expected_corr_next(self, z_now=None, exog_now=None, x_next=None) -> tuple[np.ndarray, np.ndarray]:
        """Mixture covariance of y_{T+1} given F_T and z_T -> (regime probabilities, correlation matrix).
        F2: the exogenous block of x_{T+1} is x_extra_T (`exog_now`), not the stored x_extra_{T-1}."""
        if self.W.shape[2] > 1:
            z_now = self._Zlast if z_now is None else z_now
        P = self.transition_matrix(z_now)
        pj = np.exp(self.filt_[-1]) @ P
        if x_next is None:
            lags = self._Yt[::-1][: self.p].ravel()                           # y_T, y_{T-1}, ... (matches the design)
            if self.has_exog:
                if exog_now is None:
                    raise ValueError("model has an exogenous block: pass exog_now = x_extra at the forecast origin T")
                x_next = np.r_[1.0, lags, np.atleast_1d(exog_now)]
            else:
                x_next = np.r_[1.0, lags]
        mu = np.stack([x_next @ self.B[j] for j in range(self.K)])
        m, C = mixture_moments(pj[None], mu[None], self.S * self._cov_scale())
        C = C[0]
        sd = np.sqrt(np.diag(C))
        return pj, C / np.outer(sd, sd)

    def break_cusum(self, halflife=60, kappa=0.5):
        """IN-SAMPLE diagnostic only (F1, F9): CUSUM on the in-sample predictive log score. Use `cusum()` on walk-forward
        scores for anything that claims to detect breaks in time."""
        u, c, _ = cusum(self.pred_ll_, burn=halflife, halflife=halflife, kappa=kappa, h=np.inf)
        return u, c

    # ---------- alignment (F3) ----------
    def regime_signature(self) -> np.ndarray:
        n = self.n
        sig = [np.r_[self.B[j][1:1 + n * self.p].ravel(), self.B[j][0], self.S[j][np.triu_indices(n)]]
               for j in range(self.K)]
        return np.array(sig)

    def match(self, ref_B: np.ndarray, ref_S: np.ndarray) -> np.ndarray:
        """perm with new regime a = current regime perm[a] best matching reference regime a (Hungarian)."""
        n, p = self.n, self.p
        def sig(B, S):
            return np.array([np.r_[B[j][1:1 + n * p].ravel(), B[j][0], S[j][np.triu_indices(n)]] for j in range(len(B))])
        a, b = sig(ref_B, ref_S), self.regime_signature()
        cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        perm = np.empty(self.K, int)
        perm[r] = c
        return perm

    def permute(self, perm: np.ndarray) -> TVTPMSVAR:
        perm = np.asarray(perm)
        self.B, self.S = self.B[perm], self.S[perm]
        self.W = permute_W(self.W, perm) if self.K > 1 else self.W
        self.logxi0 = self.logxi0[perm]
        if hasattr(self, "_Yt"):
            self._post()
        return self

    def align_to(self, ref: TVTPMSVAR | dict) -> np.ndarray:
        rB, rS = (ref.B, ref.S) if isinstance(ref, TVTPMSVAR) else (ref["B"], ref["S"])
        perm = self.match(rB, rS)
        if not np.array_equal(perm, np.arange(self.K)):
            self.permute(perm)
        return perm

    # ---------- inference (F13) ----------
    def trans_se(self) -> np.ndarray:
        """Wald SEs of W (K, K-1, d) from the inverse Hessian of the penalised weighted logit at the smoothed soft counts.
        Conditional on the soft counts (no Louis correction): anti-conservative by construction."""
        K, d = self.K, self._Zc.shape[1]
        if K == 1:
            return np.zeros_like(self.W)
        P = np.exp(log_trans(self._Zc, self.W))
        pred = np.exp(self.pred_)
        filt = np.exp(self.filt_)
        _, joint = kim_smoother(pred, filt, P)
        Zs, Js = self._Zc[1:], joint[1:]
        pen = self._pen(d)
        se = np.zeros_like(self.W)
        for i in range(K):
            N = Js[:, i, :]
            Ns = N.sum(1)
            lg = np.hstack([Zs @ self.W[i].T, np.zeros((len(Zs), 1))])
            Pr = np.exp(log_softmax(lg, axis=1))[:, :K - 1]
            H = np.zeros(((K - 1) * d, (K - 1) * d))
            for a in range(K - 1):
                for b in range(K - 1):
                    wts = Ns * (Pr[:, a] * ((a == b) - Pr[:, b]))
                    H[a * d:(a + 1) * d, b * d:(b + 1) * d] = (Zs.T * wts) @ Zs
            H += np.diag(np.tile(pen, K - 1))
            try:
                V = np.linalg.inv(H)
                se[i] = np.sqrt(np.clip(np.diag(V), 0, None)).reshape(K - 1, d)
            except np.linalg.LinAlgError:
                se[i] = np.nan
        return se

    # ---------- simulation ----------
    def simulate(self, T: int, Z: np.ndarray | None = None, X_extra: np.ndarray | None = None,
                 rng: np.random.Generator | None = None, y0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate from the fitted parameters (exogenous block held at the given path). Z, X_extra are (T, .) aligned
        with the output rows; the first p rows are y0 (zeros by default)."""
        return simulate_msvar(self.params() | {"p": self.p}, T, Z, X_extra, rng or self.rng, y0, self.nu)


def mixture_moments(w: np.ndarray, mu: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w (T,K) weights, mu (T,K,n) regime means, S (K,n,n) -> mixture mean (T,n), covariance (T,n,n)."""
    m = np.einsum("tk,tkn->tn", w, mu)
    second = np.einsum("tk,kab->tab", w, S) + np.einsum("tk,tka,tkb->tab", w, mu, mu)
    return m, second - np.einsum("ta,tb->tab", m, m)


def simulate_msvar(par: dict, T: int, Z: np.ndarray | None, X_extra: np.ndarray | None, rng: np.random.Generator,
                   y0: np.ndarray | None = None, nu: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    B, S, W, p = par["B"], par["S"], par["W"], int(par["p"])
    K, _, n = B.shape
    xi0 = np.exp(par["logxi0"]) if "logxi0" in par else np.full(K, 1 / K)
    Zz = None if Z is None else np.asarray(Z, float).reshape(T, -1)
    Xe = None if X_extra is None else np.asarray(X_extra, float).reshape(T, -1)
    L = [np.linalg.cholesky(S[j]) for j in range(K)]
    y = np.zeros((T, n))
    if y0 is not None:
        y[:p] = y0
    s = np.zeros(T, int)
    s[p - 1] = rng.choice(K, p=xi0 / xi0.sum())
    for t in range(p, T):
        zc = np.array([1.0]) if Zz is None else np.r_[1.0, Zz[t - 1]]
        Pt = np.exp(log_trans(zc[None, :], W)[0])[s[t - 1]]
        s[t] = rng.choice(K, p=Pt / Pt.sum())
        x = np.r_[1.0, y[t - p:t][::-1].ravel()]
        if Xe is not None:
            x = np.r_[x, Xe[t - 1]]
        e = L[s[t]] @ rng.standard_normal(n)
        if nu is not None:
            e = e * math.sqrt(nu / rng.chisquare(nu))
        y[t] = x @ B[s[t]] + e
    return y, s


# ====================================================================== walk-forward (F1)


def walk_forward(model_factory: Callable[[], TVTPMSVAR], Yt: np.ndarray, X: np.ndarray, Zc: np.ndarray,
                 origins: list[int], end: int | None = None, window: int | None = None, align: bool = True,
                 warm: bool = True, on_fit: Callable[[int, TVTPMSVAR], None] | None = None) -> dict[str, Any]:
    """Expanding (window=None) or rolling refits at design-row `origins`. The model fit at origin o uses rows
    [lo, o) only, lo = 0 or o - window; the causal filter then runs over rows [lo, next origin) with those frozen
    parameters, and rows [o, next origin) are kept as out-of-sample one-step forecasts. Every refit is aligned to the
    previous one (Hungarian on regime parameters) and warm-started from it in addition to fresh random starts."""
    T = len(Yt) if end is None else end
    K = model_factory().K
    n = Yt.shape[1]
    out = {k: np.full((T,) + s, np.nan) for k, s in
           (("pll", ()), ("pred", (K,)), ("filt", (K,)), ("mean", (n,)), ("cov", (n, n)), ("mu", (K, n)))}
    out["fold"] = np.full(T, -1)
    fits: list[dict[str, Any]] = []
    prev: TVTPMSVAR | None = None
    bounds = list(origins) + [T]
    for f, (o, o2) in enumerate(zip(bounds[:-1], bounds[1:])):
        if o >= T:
            break
        lo = 0 if window is None else max(0, o - window)
        m = model_factory()
        start = prev.params() if (warm and prev is not None) else None
        m.fit_design(Yt[lo:o], X[lo:o], Zc[lo:o], start=start)
        if align and prev is not None and m.K > 1:
            m.align_to(prev)
        r = m.filter_design(Yt[lo:o2], X[lo:o2], Zc[lo:o2])
        sl = slice(o - lo, o2 - lo)
        for k in ("pll", "pred", "filt", "mean", "cov", "mu"):
            out[k][o:o2] = r[k][sl]
        out["fold"][o:o2] = f
        fits.append(dict(origin=o, lo=lo, ll=m.ll, bic=m.bic, iters=m.iters, converged=m.converged,
                         init_lls=m.init_lls, min_occupancy=m.min_occupancy, B=m.B.copy(), S=m.S.copy(),
                         W=m.W.copy(), logxi0=m.logxi0.copy()))
        if on_fit is not None:
            on_fit(f, m)
        prev = m
    out["fits"] = fits
    return out


__all__ = ["TVTPMSVAR", "chol_psd", "companion_ccf", "cusum", "emission_logpdf", "hamilton_filter", "kim_smoother",
           "log_trans", "marginal_effects", "mixture_moments", "permute_W", "simulate_msvar", "walk_forward"]
