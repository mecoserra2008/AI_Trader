"""Jump-diffusion model for bar returns: Kou double-exponential jumps on a bipower-scaled diffusion, Hawkes intensity.

    r_t = sigma_t * e_t + 1[jump_t] * J_t,   e_t ~ N(0, 1),   P(jump_t) = p_t,
    J ~ Kou: +Exp(eta_up) with prob p_up, -Exp(eta_down) otherwise.

- `bipower_sigma`: jump-robust per-bar diffusive sigma from bars strictly before t (causal).
- `lee_mykland`: the Lee-Mykland (2008) jump test statistic and critical value.
- `fit_mixture`: MLE of (p, p_up, eta_up, eta_down) given sigma_t, using the closed-form density of a Gaussian plus an
  exponential (exponentially modified Gaussian, evaluated with erfcx for stability).
- `posterior`: P(jump | r_t) per bar.
- `fit_hawkes`: exponential-kernel Hawkes MLE on detected jump times (bar units); `hawkes_intensity` filters it causally.
- `simulate_bracket`: Monte-Carlo first passage of a stop/target bracket with jumps (Hawkes excitation inside the path),
  Brownian-bridge touch correction between bar closes, and jump slippage through the stop.

Registered tests: H-JD1-4 (research/hypotheses.yaml). Fit in Python; a Rust filter can consume `to_params()` later.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import optimize, special

SQRT2 = math.sqrt(2.0)
MU1 = math.sqrt(2.0 / math.pi)  # E|N(0,1)|


def bipower_sigma(r: np.ndarray, window: int = 72) -> np.ndarray:
    """sigma_t from |r_{i}||r_{i-1}| over the `window` bars before t (excludes t). NaN during warm-up."""
    r = np.asarray(r, float)
    a = np.abs(np.nan_to_num(r, nan=0.0))
    prod = np.r_[np.nan, a[1:] * a[:-1]]
    c = np.nancumsum(np.nan_to_num(prod, nan=0.0))
    out = np.full(len(r), np.nan)
    for t in range(window + 1, len(r)):
        s = (c[t - 1] - c[t - 1 - window]) / window
        out[t] = math.sqrt(max(s, 1e-18) / MU1**2)
    return out


def lee_mykland(r: np.ndarray, sigma: np.ndarray, n_per_day: int = 24, alpha: float = 0.01) -> tuple[np.ndarray, float]:
    """Returns (|L_t| statistic, critical |L| threshold) for a test at level alpha with n = bars per day."""
    L = np.abs(r) / sigma
    c = MU1
    ln = math.log(n_per_day)
    cn = math.sqrt(2 * ln) / c - (math.log(math.pi) + math.log(ln)) / (2 * c * math.sqrt(2 * ln))
    sn = 1 / (c * math.sqrt(2 * ln))
    beta = -math.log(-math.log(1 - alpha))
    return L, cn + sn * beta


def _emg_logpdf(x: np.ndarray, sigma: np.ndarray, eta: float) -> np.ndarray:
    """log density of N(0, sigma^2) + Exp(rate eta) at x (vectorised)."""
    b = (eta * sigma**2 - x) / (SQRT2 * sigma)
    a = eta * (eta * sigma**2 / 2.0 - x)
    # exp(a) * erfc(b) = erfcx(b) * exp(a - b^2)
    return math.log(eta / 2.0) + np.log(special.erfcx(b)) + a - b * b


def _norm_logpdf(x, sigma):
    return -0.5 * np.log(2 * math.pi) - np.log(sigma) - 0.5 * (x / sigma) ** 2


@dataclass
class JumpParams:
    p: float          # per-bar jump probability (constant-rate stage)
    p_up: float       # share of up jumps
    eta_up: float     # rate of up jump sizes (mean size 1/eta_up, log-return units)
    eta_down: float   # rate of down jump sizes
    n_obs: int = 0
    loglik: float = float("nan")
    hawkes_mu: float = float("nan")     # baseline jump intensity per bar
    hawkes_alpha: float = float("nan")  # excitation per jump
    hawkes_beta: float = float("nan")   # decay per bar
    hawkes_lr: float = float("nan")     # likelihood-ratio statistic vs constant intensity

    @property
    def mean_up(self) -> float:
        return 1 / self.eta_up

    @property
    def mean_down(self) -> float:
        return 1 / self.eta_down

    def to_params(self) -> dict:
        return asdict(self)


def jump_logpdf(x: np.ndarray, sigma: np.ndarray, p_up: float, eta_up: float, eta_down: float) -> np.ndarray:
    """log density of r given a jump: Gaussian diffusion plus a Kou jump."""
    lu = math.log(p_up) + _emg_logpdf(x, sigma, eta_up)
    ld = math.log(1 - p_up) + _emg_logpdf(-x, sigma, eta_down)
    return np.logaddexp(lu, ld)


def mixture_logpdf(x, sigma, p, p_up, eta_up, eta_down):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.logaddexp(np.log1p(-p) + _norm_logpdf(x, sigma), np.log(p) + jump_logpdf(x, sigma, p_up, eta_up, eta_down))


def fit_mixture(r: np.ndarray, sigma: np.ndarray, x0: tuple[float, float, float, float] | None = None) -> JumpParams:
    m = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0)
    x, s = np.asarray(r)[m], np.asarray(sigma)[m]
    sd = float(np.median(s))
    if x0 is None:
        x0 = (0.01, 0.5, 1 / (4 * sd), 1 / (4 * sd))

    def unpack(th):
        return special.expit(th[0]) * 0.2, special.expit(th[1]), math.exp(th[2]), math.exp(th[3])

    def nll(th):
        p, pu, eu, ed = unpack(th)
        return -float(np.sum(mixture_logpdf(x, s, p, pu, eu, ed)))

    th0 = [special.logit(min(0.199, x0[0]) / 0.2), special.logit(x0[1]), math.log(x0[2]), math.log(x0[3])]
    res = optimize.minimize(nll, th0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    p, pu, eu, ed = unpack(res.x)
    return JumpParams(p=p, p_up=pu, eta_up=eu, eta_down=ed, n_obs=int(m.sum()), loglik=-float(res.fun))


def posterior(r: np.ndarray, sigma: np.ndarray, prm: JumpParams, p_t: np.ndarray | None = None) -> np.ndarray:
    """P(jump | r_t) per bar; p_t overrides the constant prior (e.g. Hawkes)."""
    p = prm.p if p_t is None else np.clip(p_t, 1e-9, 1 - 1e-9)
    lj = np.log(p) + jump_logpdf(r, sigma, prm.p_up, prm.eta_up, prm.eta_down)
    ln = np.log1p(-p) + _norm_logpdf(r, sigma)
    return np.exp(lj - np.logaddexp(lj, ln))


def _hawkes_nll(th, times: np.ndarray, T: float) -> float:
    mu, a, b = math.exp(th[0]), math.exp(th[1]), math.exp(th[2])
    if a >= b:  # stationarity: branching ratio a/b < 1
        return 1e12
    A, ll, prev = 0.0, 0.0, None
    for t in times:
        if prev is not None:
            A = math.exp(-b * (t - prev)) * (1 + A)
        ll += math.log(mu + a * A)
        prev = t
    comp = mu * T + (a / b) * np.sum(1 - np.exp(-b * (T - times)))
    return -(ll - comp)


def fit_hawkes(times: np.ndarray, T: float) -> tuple[float, float, float, float]:
    """(mu, alpha, beta, LR vs constant) for event times in bar units on [0, T]."""
    times = np.sort(np.asarray(times, float))
    n = len(times)
    if n < 10:
        return n / T if T else float("nan"), 0.0, 1.0, 0.0
    mu0 = n / T
    ll0 = n * math.log(mu0) - mu0 * T
    best = None
    for b0 in (0.05, 0.2, 1.0):
        res = optimize.minimize(_hawkes_nll, [math.log(mu0 * 0.7), math.log(b0 * 0.3), math.log(b0)], args=(times, T), method="Nelder-Mead",
                                options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    mu, a, b = (math.exp(v) for v in best.x)
    return mu, a, b, max(0.0, 2 * (-best.fun - ll0))


def hawkes_intensity(jump_flags: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Causal per-bar intensity lambda_t from jumps strictly before t."""
    lam = np.empty(len(jump_flags))
    A = 0.0
    decay = math.exp(-beta)
    for t in range(len(jump_flags)):
        lam[t] = mu + alpha * A
        A = decay * (A + (1.0 if jump_flags[t] else 0.0))
    return lam


def p_from_intensity(lam: np.ndarray) -> np.ndarray:
    return 1 - np.exp(-np.asarray(lam))


def p_jump_within(lam_now_excess: float, mu: float, alpha: float, beta: float, bars: int) -> float:
    """P(>= 1 jump in the next `bars`) ignoring further excitation (lower bound): intensity mu + excess * exp(-beta k)."""
    k = np.arange(bars)
    lam = mu + lam_now_excess * np.exp(-beta * k)
    return float(1 - math.exp(-lam.sum()))


def simulate_bracket(entry: float, stop: float, target: float, sigma_bar: float, prm: JumpParams, bars: int, *, excess0: float = 0.0,
                     n_paths: int = 4000, seed: int = 0, use_jumps: bool = True) -> dict:
    """Monte-Carlo first passage of a resting stop / target bracket over `bars` bars.

    Diffusion with per-bar sigma, Brownian-bridge probability of touching a barrier between closes, Kou jumps with
    Hawkes intensity (mu, alpha, beta; excess0 = current excitation) including self-excitation inside each path. A jump
    that crosses the stop fills at the post-jump price (slippage); a jump crossing the target fills at the target.
    Returns p_tp, p_sl, p_open, mean R (in units of |entry - stop|), and P(stop filled worse than 1.5 R).
    """
    rng = np.random.default_rng(seed)
    side = 1.0 if target > entry else -1.0
    up_b, dn_b = (math.log(target / entry), math.log(stop / entry)) if side > 0 else (math.log(stop / entry), math.log(target / entry))
    risk = abs(math.log(stop / entry))
    x = np.zeros(n_paths)
    alive = np.ones(n_paths, bool)
    res_r = np.zeros(n_paths)
    hit = np.zeros(n_paths, int)  # 1 tp, -1 sl
    worse = np.zeros(n_paths, bool)
    mu, a, b = (prm.hawkes_mu, prm.hawkes_alpha, prm.hawkes_beta) if np.isfinite(prm.hawkes_mu) else (prm.p, 0.0, 1.0)
    A = np.full(n_paths, excess0 / a if (a and excess0) else 0.0)
    decay = math.exp(-b)
    s2 = sigma_bar**2
    for _ in range(bars):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        x0 = x[idx]
        dx = rng.standard_normal(len(idx)) * sigma_bar
        x1d = x0 + dx
        # bridge touch probabilities for the diffusion leg
        def touch(bar_level):
            d0, d1 = bar_level - x0, bar_level - x1d
            pr = np.where(d0 * d1 > 0, np.exp(-2 * d0 * d1 / s2), 1.0)
            return rng.random(len(idx)) < pr
        t_up, t_dn = touch(up_b), touch(dn_b)
        both = t_up & t_dn
        first_up = np.where(both, rng.random(len(idx)) < 0.5, t_up)
        first_dn = np.where(both, ~first_up, t_dn)
        x1 = x1d.copy()
        jumped = np.zeros(len(idx), bool)
        if use_jumps:
            lam = mu + a * A[idx]
            jumped = rng.random(len(idx)) < (1 - np.exp(-lam))
            nj = int(jumped.sum())
            if nj:
                upj = rng.random(nj) < prm.p_up
                size = np.where(upj, rng.exponential(1 / prm.eta_up, nj), -rng.exponential(1 / prm.eta_down, nj))
                x1[jumped] += size
            A[idx] = decay * (A[idx] + jumped)
        # resolve (vectorised): a diffusion touch within the bar counts first; otherwise a close beyond a barrier
        tp_r = abs(math.log(target / entry)) / risk
        diff_hit = first_up | first_dn
        diff_tp = np.where(first_up, side > 0, side < 0)
        cross_up, cross_dn = x1 >= up_b, x1 <= dn_b
        close_hit = ~diff_hit & (cross_up | cross_dn)
        close_tp = np.where(cross_up, side > 0, side < 0)
        loss = np.where(jumped, np.abs(x1) / risk, 1.0)  # a jump through the stop fills at the post-jump price
        done = diff_hit | close_hit
        is_tp = np.where(diff_hit, diff_tp, close_tp)
        r_out = np.where(is_tp, tp_r, np.where(diff_hit, -1.0, -loss))
        k = idx[done]
        hit[k] = np.where(is_tp[done], 1, -1)
        res_r[k] = r_out[done]
        worse[k] = (~is_tp[done]) & (r_out[done] < -1.5)
        alive[k] = False
        x[idx[~done]] = x1[~done]
    res_r[alive] = side * x[alive] / risk
    return {"p_tp": float((hit == 1).mean()), "p_sl": float((hit == -1).mean()), "p_open": float(alive.mean()), "mean_R": float(res_r.mean()),
            "p_stop_worse_1p5R": float(worse.mean()), "n_paths": n_paths, "bars": bars, "use_jumps": use_jumps}
