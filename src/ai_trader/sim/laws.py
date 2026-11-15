"""Fitted empirical laws of trading activity with `sample()` methods (microstructure pilot, H-MS1..H-MS4, H-MS9).

Every law serialises to a plain dict (`to_dict` / `law_from_dict`), so a simulator can rebuild it from
data/models/microstructure_laws/params_<date>.json without refitting.

Contents
  Law              positive-support distribution families fitted by MLE: exponential, weibull, gamma, lognormal,
                   gengamma (Stacy), burr12, loglogistic (Fisk), expmix2 / expmix3 (exponential mixtures, EM)
  ACD              ACD(1,1) of Engle and Russell (1998) on deseasonalised durations, innovations exponential / weibull /
                   gengamma with unit mean
  trailing_scale   causal deseasonalisation factor (mean duration over a trailing clock window)
  PowerLawTail     Clauset-Shalizi-Newman (2009): x_min by KS minimisation, MLE alpha (pdf exponent), semi-parametric
                   bootstrap p, Vuong test vs a truncated log-normal; SizeLaw = log-normal body spliced to the tail
  acf / pacf / log_bins / fit_power_acf / fit_exp_acf   correlogram tools (Bartlett / Durbin-Levinson)
  TranslatedSignModel  sign generator for an arbitrary target ACF (empirical short lags + power-law tail) by the
                   inverse arcsine law and circulant embedding; use it for real targets
  FGNSignModel     long-memory trade-sign generator: sign of (fractional Gaussian noise + white noise); sign ACF is
                   (2/pi) arcsin(w rho_H(k)) ~ c k^-gamma with H = 1 - gamma/2 (Davies-Harte exact fGn)
  LMFSignModel     Lillo-Mike-Farmer (2005) metaorder generator: N concurrent hidden orders with Pareto lengths
                   (tail exponent a = gamma + 1), each print takes the sign of a random active metaorder
  dfa              detrended fluctuation analysis (DFA-1) Hurst exponent
  HurdleLaw        volume per interval: P(zero) + a positive family
  aft_logpdf       log density of an AFT model log T = mu + sigma Z (normal / logistic / extreme), xgboost conventions

Conventions: durations in seconds, sizes in quote notional (USD), all logs natural. Nothing here reads data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, signal, special, stats

FAMILIES = ("exponential", "weibull", "gamma", "lognormal", "gengamma", "burr12", "loglogistic", "expmix2", "expmix3")
_SCIPY = {"exponential": "expon", "weibull": "weibull_min", "gamma": "gamma", "lognormal": "lognorm", "gengamma": "gengamma",
          "burr12": "burr12", "loglogistic": "fisk"}
_SHAPES = {"exponential": (), "weibull": ("c",), "gamma": ("a",), "lognormal": ("s",), "gengamma": ("a", "c"),
           "burr12": ("c", "d"), "loglogistic": ("c",)}
EULER = 0.5772156649015329


# ----------------------------------------------------------------------------------------------------------------------
# Positive-support families
# ----------------------------------------------------------------------------------------------------------------------

def _logpdf_std(family: str, lz: np.ndarray, shapes: tuple[float, ...]) -> np.ndarray:
    """log density of the standard (scale 1) family at z = exp(lz), WITHOUT the -log(scale) Jacobian, plus -lz
    already included where the density has a 1/z factor. Fast closed forms (no scipy argument checking)."""
    if family == "exponential":
        return -np.exp(lz)
    if family == "weibull":
        (c,) = shapes
        return math.log(c) + (c - 1) * lz - np.exp(c * lz)
    if family == "gamma":
        (a,) = shapes
        return -special.gammaln(a) + (a - 1) * lz - np.exp(lz)
    if family == "lognormal":
        (s,) = shapes
        return -0.5 * (lz / s) ** 2 - lz - math.log(s) - 0.5 * math.log(2 * math.pi)
    if family == "gengamma":
        a, c = shapes
        return math.log(abs(c)) - special.gammaln(a) + (c * a - 1) * lz - np.exp(c * lz)
    if family == "burr12":
        c, d = shapes
        return math.log(c) + math.log(d) + (c - 1) * lz - (d + 1) * np.logaddexp(0.0, c * lz)
    if family == "loglogistic":
        (c,) = shapes
        return math.log(c) + (c - 1) * lz - 2 * np.logaddexp(0.0, c * lz)
    raise ValueError(family)


@dataclass
class Law:
    """A fitted positive law. params: shape names of the scipy family + 'scale' (scipy parametrisation), or for
    expmixK: w1..wK (weights) and rate1..rateK (per unit of x)."""
    family: str
    params: dict[str, float]
    n_fit: int = 0
    loglik_fit: float = float("nan")

    @property
    def n_params(self) -> int:
        if self.family.startswith("expmix"):
            return 2 * int(self.family[-1]) - 1
        return len(_SHAPES[self.family]) + 1

    def _mix(self) -> tuple[np.ndarray, np.ndarray]:
        k = int(self.family[-1])
        return (np.array([self.params[f"w{j + 1}"] for j in range(k)]), np.array([self.params[f"rate{j + 1}"] for j in range(k)]))

    def _shapes(self) -> tuple[float, ...]:
        return tuple(self.params[s] for s in _SHAPES[self.family])

    def logpdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.family.startswith("expmix"):
            w, r = self._mix()
            return special.logsumexp(np.log(w)[None, :] + np.log(r)[None, :] - x[:, None] * r[None, :], axis=1)
        sc = self.params["scale"]
        with np.errstate(divide="ignore"):
            lz = np.log(x) - math.log(sc)
        return _logpdf_std(self.family, lz, self._shapes()) - math.log(sc)

    def loglik(self, x) -> float:
        return float(np.sum(self.logpdf(x)))

    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.family.startswith("expmix"):
            w, r = self._mix()
            return 1.0 - (w[None, :] * np.exp(-x[:, None] * r[None, :])).sum(axis=1)
        return getattr(stats, _SCIPY[self.family]).cdf(x, *self._shapes(), scale=self.params["scale"])

    def mean(self) -> float:
        if self.family.startswith("expmix"):
            w, r = self._mix()
            return float((w / r).sum())
        return float(getattr(stats, _SCIPY[self.family]).mean(*self._shapes(), scale=self.params["scale"]))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.family.startswith("expmix"):
            w, r = self._mix()
            comp = rng.choice(len(w), size=n, p=w / w.sum())
            return rng.exponential(1.0 / r[comp])
        return getattr(stats, _SCIPY[self.family]).rvs(*self._shapes(), scale=self.params["scale"], size=n, random_state=rng)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "params": {k: float(v) for k, v in self.params.items()}, "n_fit": int(self.n_fit),
                "loglik_fit": float(self.loglik_fit), "k": self.n_params}


def _start(family: str, ly: np.ndarray, y: np.ndarray) -> list[tuple[float, ...]]:
    """Starting values (shapes..., scale) in standardised units y (median 1)."""
    mly, sly = float(ly.mean()), float(max(ly.std(), 1e-3))
    m = float(y.mean())
    s_ = max(math.log(m) - mly, 1e-6)
    a_g = (3 - s_ + math.sqrt((s_ - 3) ** 2 + 24 * s_)) / (12 * s_)
    c_w = math.pi / (math.sqrt(6) * sly)
    c_l = math.pi / (math.sqrt(3) * sly)
    if family == "weibull":
        return [(c_w, math.exp(mly + EULER / c_w))]
    if family == "gamma":
        return [(a_g, m / a_g)]
    if family == "loglogistic":
        return [(c_l, math.exp(float(np.median(ly))))]
    if family == "gengamma":
        return [(1.0, c_w, math.exp(mly + EULER / c_w)), (a_g, 1.0, m / a_g), (2.0, 0.5, math.exp(mly)), (0.5, 2.0, math.exp(mly))]
    if family == "burr12":
        return [(c_l, 1.0, math.exp(float(np.median(ly)))), (c_w, 2.0, 2 * math.exp(mly)), (1.0, 0.5, 1.0)]
    raise ValueError(family)


def _fit_mixture(y: np.ndarray, k: int, iters: int = 300, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, float]:
    qs = np.quantile(y, np.linspace(0, 1, k + 1))
    rates = np.array([1.0 / max(float(y[(y >= qs[j]) & (y <= qs[j + 1])].mean()), 1e-9) for j in range(k)])
    w = np.full(k, 1.0 / k)
    prev = -np.inf
    ll = -np.inf
    for _ in range(iters):
        lp = np.log(w)[None, :] + np.log(rates)[None, :] - y[:, None] * rates[None, :]
        mx = lp.max(axis=1, keepdims=True)
        resp = np.exp(lp - mx)
        tot = resp.sum(axis=1, keepdims=True)
        ll = float((np.log(tot) + mx).sum())
        resp /= tot
        nk = resp.sum(axis=0) + 1e-12
        w = nk / nk.sum()
        rates = nk / (resp * y[:, None]).sum(axis=0)
        if abs(ll - prev) < tol * abs(ll):
            break
        prev = ll
    o = np.argsort(-rates)
    return w[o], rates[o], ll


def _multistart(nll, starts: list[np.ndarray], n_best: int = 2):
    """L-BFGS-B from the `n_best` starting points with the lowest objective, then a Nelder-Mead polish of the winner."""
    vals = [nll(th) for th in starts]
    order = np.argsort(vals)[:n_best]
    best = None
    for i in order:
        try:
            r = optimize.minimize(nll, starts[i], method="L-BFGS-B", options={"maxiter": 500})
        except Exception:  # noqa: BLE001 - optimiser failures are tried with the next start
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    if best is not None:
        try:
            r = optimize.minimize(nll, best.x, method="Nelder-Mead", options={"maxiter": 600 * len(best.x), "xatol": 1e-7, "fatol": 1e-12})
            if np.isfinite(r.fun) and r.fun < best.fun:
                best = r
        except Exception:  # noqa: BLE001
            pass
    return best


def _nested_starts(family: str, nested: dict[str, Law] | None, med: float) -> list[tuple[float, ...]]:
    """Exact nesting points (standardised units) from already fitted sub-families: gengamma(a=1) = Weibull,
    gengamma(c=1) = gamma, burr12(d=1) = log-logistic, burr12(d -> inf) ~ Weibull."""
    if not nested:
        return []
    out = []
    w, g, ll = nested.get("weibull"), nested.get("gamma"), nested.get("loglogistic")
    if family == "gengamma":
        if isinstance(w, Law):
            out.append((1.0, w.params["c"], w.params["scale"] / med))
        if isinstance(g, Law):
            out.append((g.params["a"], 1.0, g.params["scale"] / med))
    if family == "burr12":
        if isinstance(ll, Law):
            out.append((ll.params["c"], 1.0, ll.params["scale"] / med))
        if isinstance(w, Law):
            d = 200.0
            out.append((w.params["c"], d, w.params["scale"] / med * d ** (1.0 / w.params["c"])))
    return out


def fit_law(family: str, x, max_n: int | None = 200_000, nested: dict[str, Law] | None = None) -> Law:
    """MLE of one family on positive x. Fits in standardised units y = x / median(x) and rescales.
    `max_n`: deterministic thinning (every k-th value) for the optimiser; the reported loglik_fit is on all of x.
    `nested`: fitted sub-families (weibull, gamma, loglogistic) used as exact starting points for gengamma / burr12, so
    the nesting family never ends below them."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 20:
        raise ValueError("fewer than 20 positive values")
    med = float(np.median(x))
    xs = x if max_n is None or len(x) <= max_n else x[:: int(math.ceil(len(x) / max_n))]
    y = xs / med
    ly = np.log(y)
    if family == "exponential":
        law = Law("exponential", {"scale": float(x.mean())})
    elif family == "lognormal":
        law = Law("lognormal", {"s": float(np.log(x).std()), "scale": float(math.exp(np.log(x).mean()))})
    elif family.startswith("expmix"):
        k = int(family[-1])
        w, r, _ = _fit_mixture(y[:: int(math.ceil(len(y) / 200_000))] if len(y) > 200_000 else y, k)
        law = Law(family, {**{f"w{j + 1}": float(w[j]) for j in range(k)}, **{f"rate{j + 1}": float(r[j] / med) for j in range(k)}})
    else:
        names = _SHAPES[family]

        def unpack(th):  # the SAME transform for the objective and the reported parameters
            return np.exp(np.clip(th, -60.0, 60.0))

        def nll(th):
            p = unpack(th)
            v = _logpdf_std(family, ly - math.log(p[-1]), tuple(p[:-1])) - math.log(p[-1])
            out = -float(np.mean(v))
            return out if np.isfinite(out) else 1e10

        starts = [*_nested_starts(family, nested, med), *_start(family, ly, y)]
        best = _multistart(nll, [np.log(np.asarray(st, dtype=float)) for st in starts])
        if best is None or best.fun >= 1e10:
            raise RuntimeError(f"{family}: MLE failed")
        p = unpack(best.x)
        law = Law(family, {**{nm: float(p[i]) for i, nm in enumerate(names)}, "scale": float(p[-1] * med)})
    law.n_fit = len(x)
    law.loglik_fit = law.loglik(x)
    return law


def fit_all(x, families=FAMILIES, max_n: int | None = 200_000) -> dict[str, Law | str]:
    """All families; a failed fit is returned as its error string (never raised)."""
    out: dict[str, Law | str] = {}
    order = sorted(families, key=lambda f: f in ("gengamma", "burr12"))  # sub-families first
    for f in order:
        try:
            out[f] = fit_law(f, x, max_n=max_n, nested={k: v for k, v in out.items() if isinstance(v, Law)})
        except Exception as e:  # noqa: BLE001 - recorded as a failure
            out[f] = f"failed: {e}"
    return {f: out[f] for f in families}


def law_from_dict(d: dict[str, Any]):
    kind = d.get("kind", "law")
    if kind == "acd":
        return ACD.from_dict(d)
    if kind == "powerlaw":
        return PowerLawTail(**{k: d[k] for k in ("xmin", "alpha", "n_tail", "n", "ks")})
    if kind == "size":
        return SizeLaw(**{k: d[k] for k in ("mu", "sigma", "xmin", "alpha", "p_tail")})
    if kind == "fgn_sign":
        return FGNSignModel(**{k: d[k] for k in ("c", "gamma", "p_buy")})
    if kind == "translated_sign":
        return TranslatedSignModel(d["c"], d["gamma"], d.get("p_buy", 0.5), {int(a): float(b) for a, b in d.get("short", {}).items()})
    if kind == "hurdle":
        return HurdleLaw(p_zero=d["p_zero"], law=law_from_dict(d["law"]))
    return Law(d["family"], dict(d["params"]), int(d.get("n_fit", 0)), float(d.get("loglik_fit", float("nan"))))


def ks_exponential(x) -> tuple[float, float]:
    """KS statistic and p-value of x against an exponential with the sample mean (scale estimated: the p-value is
    conservative, Lilliefors; with n in the thousands any rejection here is decisive)."""
    x = np.asarray(x, dtype=float)
    r = stats.kstest(x, "expon", args=(0, float(x.mean())))
    return float(r.statistic), float(r.pvalue)


def ks_stat(x, cdf) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    F = cdf(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


# ----------------------------------------------------------------------------------------------------------------------
# Deseasonalisation and ACD(1,1)
# ----------------------------------------------------------------------------------------------------------------------

def trailing_scale(t_s: np.ndarray, window_s: float = 1800.0, min_events: int = 5) -> np.ndarray:
    """Causal duration scale for the duration that STARTS at event i-1 (so it ends at t_s[i]), i = 1..n-1:
    s_i = elapsed / count over events in (t_{i-1} - window, t_{i-1}], elapsed = min(window, t_{i-1} - t_0).
    Returns an array aligned with np.diff(t_s); NaN where fewer than `min_events` events are in the window."""
    t = np.asarray(t_s, dtype=float)
    start = t[:-1]
    lo = np.searchsorted(t, start - window_s, side="right")
    cnt = np.arange(len(start)) - lo + 1
    elapsed = np.minimum(window_s, start - t[0])
    s = np.where((cnt >= min_events) & (elapsed > 0), elapsed / np.maximum(cnt, 1), np.nan)
    return s


_INNOV = {"exponential": (), "weibull": ("k",), "gengamma": ("a", "c")}


def _innov_logpdf(innov: str, e: np.ndarray, sh: tuple[float, ...]) -> np.ndarray:
    """log density of the unit-mean innovation at e > 0."""
    if innov == "exponential":
        return -e
    le = np.log(e)
    if innov == "weibull":
        (k,) = sh
        lam = 1.0 / math.gamma(1.0 + 1.0 / k)
        lz = le - math.log(lam)
        return math.log(k) - math.log(lam) + (k - 1) * lz - np.exp(k * lz)
    if innov == "gengamma":
        a, c = sh
        llam = special.gammaln(a) - special.gammaln(a + 1.0 / c)
        lz = le - llam
        return math.log(c) - special.gammaln(a) + (c * a - 1) * lz - np.exp(c * lz) - llam
    raise ValueError(innov)


def _innov_sample(innov: str, sh: tuple[float, ...], n: int, rng: np.random.Generator) -> np.ndarray:
    if innov == "exponential":
        return rng.exponential(1.0, n)
    if innov == "weibull":
        (k,) = sh
        return rng.weibull(k, n) / math.gamma(1.0 + 1.0 / k)
    a, c = sh
    lam = math.exp(special.gammaln(a) - special.gammaln(a + 1.0 / c))
    return lam * rng.gamma(a, 1.0, n) ** (1.0 / c)


@dataclass
class ACD:
    """psi_i = omega + alpha x_{i-1} + beta psi_{i-1};  x_i = psi_i eps_i, E eps = 1 (x = duration / seasonal scale)."""
    omega: float
    alpha: float
    beta: float
    innov: str = "exponential"
    shape: dict[str, float] = field(default_factory=dict)
    psi0: float = 1.0
    n_fit: int = 0
    loglik_fit: float = float("nan")

    @property
    def n_params(self) -> int:
        return 3 + len(_INNOV[self.innov])

    def _sh(self) -> tuple[float, ...]:
        return tuple(self.shape[s] for s in _INNOV[self.innov])

    def filter(self, x: np.ndarray, psi0: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.empty_like(x)
        u[0] = self.psi0 if psi0 is None else psi0
        u[1:] = self.omega + self.alpha * x[:-1]
        return signal.lfilter([1.0], [1.0, -self.beta], u)

    def logpdf(self, x: np.ndarray, psi: np.ndarray | None = None) -> np.ndarray:
        """Per-observation log density of x (deseasonalised durations). Add -log(scale) for raw durations."""
        x = np.asarray(x, dtype=float)
        psi = self.filter(x) if psi is None else psi
        return _innov_logpdf(self.innov, x / psi, self._sh()) - np.log(psi)

    def residuals(self, x: np.ndarray, psi: np.ndarray | None = None) -> np.ndarray:
        psi = self.filter(x) if psi is None else psi
        return np.asarray(x, dtype=float) / psi

    def m2_factor(self) -> float:
        """E[(alpha eps + beta)^2] = alpha^2 E[eps^2] + 2 alpha beta + beta^2: the process has a finite second moment
        (and is usable as a generator) only if this is < 1. Many fits here are good one-step filters with m2 >> 1."""
        sh = self._sh()
        if self.innov == "exponential":
            e2 = 2.0
        elif self.innov == "weibull":
            (k,) = sh
            e2 = math.gamma(1 + 2 / k) / math.gamma(1 + 1 / k) ** 2
        else:
            a, c = sh
            e2 = math.exp(special.gammaln(a + 2 / c) + special.gammaln(a) - 2 * special.gammaln(a + 1 / c))
        return self.alpha ** 2 * e2 + 2 * self.alpha * self.beta + self.beta ** 2

    def unconditional_mean(self) -> float:
        p = self.alpha + self.beta
        return self.omega / (1 - p) if p < 1 else float("inf")

    def sample(self, n: int, rng: np.random.Generator | None = None, scale: np.ndarray | None = None,
               max_persistence: float = 0.99) -> np.ndarray:
        """n durations; multiplied by `scale` (seasonal factor per duration) if given.
        Simulation-time regularisation: near-integrated fits (alpha + beta at the 0.99995 bound, common on this data)
        are degenerate to simulate (the multiplicative recursion drifts to ~0 and the long-run mean is not identified),
        so if alpha + beta > `max_persistence` both are scaled down to sum to it, and omega is set to psi0 (1 - alpha -
        beta), anchoring the process at its in-sample mean. Pass max_persistence=1 to sample the raw fit. Even capped, a fit
        with m2_factor() >= 1 has no finite variance and gives far too dispersed paths: use such fits as filters only."""
        rng = rng or np.random.default_rng()
        eps = _innov_sample(self.innov, self._sh(), n, rng)
        p = self.alpha + self.beta
        al, be, omega = self.alpha, self.beta, self.omega
        if p > max_persistence:
            al, be = al * max_persistence / p, be * max_persistence / p
            omega = self.psi0 * (1.0 - max_persistence)
            p = max_persistence
        x = np.empty(n)
        psi = omega / (1.0 - p) if p < 1.0 else self.psi0
        prev = psi
        for i in range(n):
            psi = omega + al * prev + be * psi if i else psi
            x[i] = psi * eps[i]
            prev = x[i]
        return x * scale if scale is not None else x

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "acd", "omega": self.omega, "alpha": self.alpha, "beta": self.beta, "innov": self.innov,
                "shape": dict(self.shape), "psi0": self.psi0, "n_fit": self.n_fit, "loglik_fit": self.loglik_fit, "k": self.n_params,
                "m2_factor": self.m2_factor()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ACD:
        return cls(d["omega"], d["alpha"], d["beta"], d["innov"], dict(d["shape"]), d.get("psi0", 1.0), d.get("n_fit", 0),
                   d.get("loglik_fit", float("nan")))


def fit_acd(x, innov: str = "weibull", start: ACD | None = None) -> ACD:
    """MLE of ACD(1,1) on deseasonalised durations x (finite, > 0). Persistence p = alpha + beta < 1 and alpha share are
    logit-parametrised, omega and shapes log-parametrised. Starts from (alpha, beta) = (0.05, 0.90) and, for gengamma,
    from the Weibull solution."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    names = _INNOV[innov]

    def unpack(th):
        p = special.expit(th[0]) * 0.99995
        a = special.expit(th[1])
        omega = math.exp(th[2])
        sh = tuple(np.exp(th[3:]))
        return omega, p * a, p * (1 - a), sh

    def nll(th):
        omega, al, be, sh = unpack(th)
        u = np.empty_like(x)
        u[0] = m
        u[1:] = omega + al * x[:-1]
        psi = signal.lfilter([1.0], [1.0, -be], u)
        v = _innov_logpdf(innov, x / psi, sh) - np.log(psi)
        out = -float(np.mean(v))
        return out if np.isfinite(out) else 1e10

    def th_of(al, be, omega, sh):
        p = al + be
        return np.array([special.logit(min(p / 0.99995, 1 - 1e-9)), special.logit(al / p), math.log(omega), *np.log(sh)])

    starts = []
    if start is not None:
        sh0 = start._sh() if start.innov == innov else ()
        if innov == "gengamma" and start.innov == "weibull":
            sh0 = (1.0, start.shape["k"])
        if len(sh0) == len(names):
            starts.append(th_of(start.alpha, start.beta, start.omega, sh0))
    dflt = {"exponential": (), "weibull": (0.8,), "gengamma": (1.0, 0.8)}[innov]
    starts.append(th_of(0.05, 0.90, 0.05 * m, dflt))
    starts.append(th_of(0.15, 0.80, 0.05 * m, dflt))
    best = _multistart(nll, starts, n_best=2)
    if best is None or best.fun >= 1e10:
        raise RuntimeError(f"ACD-{innov}: MLE failed")
    omega, al, be, sh = unpack(best.x)
    mod = ACD(float(omega), float(al), float(be), innov, {nm: float(sh[i]) for i, nm in enumerate(names)}, psi0=m, n_fit=len(x))
    mod.loglik_fit = float(np.sum(mod.logpdf(x)))
    return mod


# ----------------------------------------------------------------------------------------------------------------------
# Power-law tails (Clauset, Shalizi and Newman 2009)
# ----------------------------------------------------------------------------------------------------------------------

def _csn_scan(xs: np.ndarray, cand_idx: np.ndarray) -> tuple[int, float, float]:
    """xs sorted ascending; for each candidate index j (x_min = xs[j]) MLE alpha on xs[j:] and the KS distance.
    Returns (best j, alpha, D)."""
    n = len(xs)
    lx = np.log(xs)
    csum = np.concatenate([np.cumsum(lx[::-1])[::-1], [0.0]])  # csum[j] = sum lx[j:]
    best = (int(cand_idx[0]), float("nan"), float("inf"))
    for j in cand_idx:
        m = n - j
        xmin = xs[j]
        s = csum[j] - m * lx[j]
        if m < 2 or s <= 0:
            continue
        a = 1.0 + m / s
        tail = xs[j:]
        F = 1.0 - np.exp(-(a - 1.0) * (lx[j:] - lx[j]))
        i = np.arange(1, m + 1)
        D = max(float(np.max(i / m - F)), float(np.max(F - (i - 1) / m)))
        if D < best[2]:
            best = (int(j), float(a), D)
        del tail, xmin
    return best


def _candidates(xs: np.ndarray, n_cand: int, min_tail: int, q_lo: float) -> np.ndarray:
    n = len(xs)
    lo = int(q_lo * n)
    hi = n - min_tail
    if hi <= lo:
        return np.array([max(0, hi)])
    # geometric in the tail size: dense in the far tail, sparse in the body
    tails = np.unique(np.geomspace(n - lo, min_tail, n_cand).astype(int))
    idx = n - tails
    # move each candidate to the first occurrence of its value (ties: x_min is a value, the tail is x >= x_min)
    idx = np.searchsorted(xs, xs[idx], side="left")
    return np.unique(idx[(idx >= 0) & (idx <= hi)])


@dataclass
class PowerLawTail:
    """pdf p(x) = (alpha - 1) / xmin (x / xmin)^-alpha for x >= xmin. Tail index (CCDF exponent) zeta = alpha - 1."""
    xmin: float
    alpha: float
    n_tail: int
    n: int
    ks: float

    @property
    def zeta(self) -> float:
        return self.alpha - 1.0

    @property
    def p_tail(self) -> float:
        return self.n_tail / self.n

    def logpdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return math.log(self.alpha - 1) - math.log(self.xmin) - self.alpha * np.log(x / self.xmin)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return self.xmin * (1.0 - rng.random(n)) ** (-1.0 / (self.alpha - 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "powerlaw", "xmin": self.xmin, "alpha": self.alpha, "zeta": self.zeta, "n_tail": self.n_tail, "n": self.n,
                "ks": self.ks, "p_tail": self.p_tail}


def fit_powerlaw(x, n_cand: int = 80, min_tail: int = 50, q_lo: float = 0.5) -> PowerLawTail:
    """CSN: x_min minimises the KS distance over candidates (tail sizes geometric between the upper (1 - q_lo) share and
    `min_tail` points), alpha = 1 + m / sum log(x / x_min)."""
    xs = np.sort(np.asarray(x, dtype=float))
    xs = xs[np.isfinite(xs) & (xs > 0)]
    j, a, D = _csn_scan(xs, _candidates(xs, n_cand, min_tail, q_lo))
    return PowerLawTail(float(xs[j]), a, int(len(xs) - j), int(len(xs)), D)


def powerlaw_bootstrap_p(x, fit: PowerLawTail, B: int = 200, rng: np.random.Generator | None = None, **kw) -> float:
    """CSN semi-parametric bootstrap goodness of fit: share of synthetic samples (power law above x_min with
    probability n_tail / n, the empirical body below x_min otherwise) whose re-fitted KS distance >= the observed one."""
    rng = rng or np.random.default_rng(0)
    xs = np.sort(np.asarray(x, dtype=float))
    xs = xs[np.isfinite(xs) & (xs > 0)]
    body = xs[xs < fit.xmin]
    n = len(xs)
    hits = 0
    for _ in range(B):
        nt = rng.binomial(n, fit.n_tail / n)
        syn = np.concatenate([fit.sample(nt, rng), rng.choice(body, n - nt) if len(body) and n - nt > 0 else np.empty(0)])
        f2 = fit_powerlaw(syn, **kw)
        hits += f2.ks >= fit.ks
    return hits / B


def vuong_lognormal(x, fit: PowerLawTail) -> dict[str, float]:
    """Vuong normalised log-likelihood ratio of the power law vs a log-normal truncated to x >= x_min (both MLE on the
    tail). R > 0 favours the power law; p is two-sided."""
    x = np.asarray(x, dtype=float)
    t = x[x >= fit.xmin]
    lt = np.log(t)
    lxm = math.log(fit.xmin)

    def ln_ll(th):
        mu, s = th[0], math.exp(th[1])
        return stats.norm.logpdf(lt, mu, s) - lt - stats.norm.logsf(lxm, mu, s)

    best = None
    for mu0, s0 in ((float(lt.mean()), float(max(lt.std(), 1e-2))), (lxm, 1.0), (lxm - 2.0, 2.0)):
        r = optimize.minimize(lambda th: -float(np.mean(ln_ll(th))), [mu0, math.log(s0)], method="Nelder-Mead",
                              options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    l_ln = ln_ll(best.x)
    l_pl = fit.logpdf(t)
    d = l_pl - l_ln
    R = float(d.sum())
    sd = float(d.std())
    z = R / (math.sqrt(len(d)) * sd) if sd > 0 else 0.0
    return {"R": R, "z": z, "p": float(2 * stats.norm.sf(abs(z))), "ln_mu": float(best.x[0]), "ln_sigma": float(math.exp(best.x[1]))}


def hill(x, k: int) -> float:
    """Hill estimate of the tail index zeta from the k largest values."""
    xs = np.sort(np.asarray(x, dtype=float))[::-1]
    k = min(k, len(xs) - 1)
    return float(1.0 / np.mean(np.log(xs[:k] / xs[k])))


@dataclass
class SizeLaw:
    """Trade notional: log-normal(mu, sigma) truncated to (0, xmin) with probability 1 - p_tail, Pareto tail above."""
    mu: float
    sigma: float
    xmin: float
    alpha: float
    p_tail: float

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        tail = rng.random(n) < self.p_tail
        out = np.empty(n)
        nt = int(tail.sum())
        out[tail] = self.xmin * (1.0 - rng.random(nt)) ** (-1.0 / (self.alpha - 1.0))
        cmax = stats.norm.cdf((math.log(self.xmin) - self.mu) / self.sigma)
        u = rng.random(n - nt) * cmax
        out[~tail] = np.exp(self.mu + self.sigma * stats.norm.ppf(np.clip(u, 1e-300, None)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "size", "mu": self.mu, "sigma": self.sigma, "xmin": self.xmin, "alpha": self.alpha, "p_tail": self.p_tail}


def fit_size_law(x, tail: PowerLawTail) -> SizeLaw:
    x = np.asarray(x, dtype=float)
    body = np.log(x[(x > 0) & (x < tail.xmin)])
    lxm = math.log(tail.xmin)
    if len(body) < 10:
        return SizeLaw(lxm - 1.0, 1.0, tail.xmin, tail.alpha, tail.p_tail)

    def nll(th):
        mu, s = th[0], math.exp(th[1])
        return -float(np.mean(stats.norm.logpdf(body, mu, s) - stats.norm.logcdf(lxm, mu, s)))

    r = optimize.minimize(nll, [float(body.mean()), math.log(max(body.std(), 1e-2))], method="Nelder-Mead")
    return SizeLaw(float(r.x[0]), float(math.exp(r.x[1])), tail.xmin, tail.alpha, tail.p_tail)


# ----------------------------------------------------------------------------------------------------------------------
# Correlograms, long memory and sign generators
# ----------------------------------------------------------------------------------------------------------------------

def acf(x, maxlag: int) -> np.ndarray:
    """Sample ACF at lags 0..maxlag (FFT, biased denominator n, demeaned)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    r = np.fft.irfft(f * np.conj(f), nfft)[: maxlag + 1]
    return r / r[0]


def pacf_from_acf(rho: np.ndarray, maxlag: int) -> np.ndarray:
    """Durbin-Levinson: PACF at lags 1..maxlag from ACF values rho[0..maxlag]."""
    phi = np.zeros(maxlag + 1)
    out = np.zeros(maxlag)
    prev = np.zeros(maxlag + 1)
    v = 1.0
    for k in range(1, maxlag + 1):
        a = (rho[k] - np.dot(prev[1:k], rho[k - 1:0:-1])) / v
        phi[1:k] = prev[1:k] - a * prev[k - 1:0:-1]
        phi[k] = a
        v *= 1 - a * a
        out[k - 1] = a
        prev = phi.copy()
    return out


def log_bins(maxlag: int, per_decade: int = 8) -> list[np.ndarray]:
    edges = np.unique(np.round(np.logspace(0, math.log10(maxlag + 1), per_decade * int(math.ceil(math.log10(maxlag + 1))) + 1)).astype(int))
    bins = [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bins = [b[b <= maxlag] for b in bins]
    return [b for b in bins if len(b)]


def binned(rho: np.ndarray, bins: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(geometric-centre lag, mean ACF) per log bin; rho indexed by lag (rho[0] = 1)."""
    k = np.array([math.exp(np.log(b).mean()) for b in bins])
    v = np.array([float(rho[b].mean()) for b in bins])
    return k, v


def fit_power_acf(k: np.ndarray, v: np.ndarray, floor: float = 0.0) -> dict[str, float]:
    """OLS of log ACF on log lag over bins with ACF > floor (e.g. the 2 / sqrt(n) noise band): ACF = c k^-gamma."""
    ok = v > max(floor, 0.0)
    if ok.sum() < 3:
        return {"c": float("nan"), "gamma": float("nan"), "n_bins": int(ok.sum())}
    b = np.polyfit(np.log(k[ok]), np.log(v[ok]), 1)
    return {"c": float(math.exp(b[1])), "gamma": float(-b[0]), "n_bins": int(ok.sum())}


def fit_exp_acf(k: np.ndarray, v: np.ndarray, floor: float = 0.0) -> dict[str, float]:
    """OLS of log ACF on lag over bins with ACF > floor: ACF = c exp(-k / tau)."""
    ok = v > max(floor, 0.0)
    if ok.sum() < 3:
        return {"c": float("nan"), "tau": float("nan"), "n_bins": int(ok.sum())}
    b = np.polyfit(k[ok], np.log(v[ok]), 1)
    return {"c": float(math.exp(b[1])), "tau": float(-1.0 / b[0]) if b[0] < 0 else float("inf"), "n_bins": int(ok.sum())}


def heldout_log_mse(k: np.ndarray, v: np.ndarray, pw: dict, ex: dict, floor: float = 0.0) -> tuple[float, float, int]:
    """MSE in log ACF on held-out bins with ACF > floor of the power and exponential fits."""
    ok = v > max(floor, 0.0)
    if ok.sum() < 3 or not (np.isfinite(pw["c"]) and np.isfinite(ex["c"])):
        return float("nan"), float("nan"), int(ok.sum())
    lv = np.log(v[ok])
    lp = math.log(pw["c"]) - pw["gamma"] * np.log(k[ok])
    le = math.log(ex["c"]) - (k[ok] / ex["tau"] if np.isfinite(ex["tau"]) else 0.0)
    return float(np.mean((lv - lp) ** 2)), float(np.mean((lv - le) ** 2)), int(ok.sum())


def fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise (unit variance) by Davies-Harte circulant embedding."""
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(row).real
    lam = np.clip(lam, 0, None)
    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.fft(np.sqrt(lam / m) * z)
    return w.real[:n]


def fgn_rho(k: np.ndarray, H: float) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))


@dataclass
class FGNSignModel:
    """Signs = 1{ sqrt(w) G_H + sqrt(1 - w) N > q }, G_H fGn, N white noise, q set by p_buy. Target sign ACF c k^-gamma
    for k >= 1: H = 1 - gamma / 2 (clipped to (0.5, 1)), w by least squares of (2/pi) arcsin(w rho_H(k)) on lags 1-1000."""
    c: float
    gamma: float
    p_buy: float = 0.5

    @property
    def H(self) -> float:
        return float(np.clip(1.0 - self.gamma / 2.0, 0.505, 0.995))

    @property
    def w(self) -> float:
        k = np.unique(np.round(np.geomspace(1, 1000, 40)))
        tgt = self.c * k ** (-self.gamma)
        rh = fgn_rho(k, self.H)
        r = optimize.minimize_scalar(lambda w: float(np.sum((np.log(np.maximum((2 / np.pi) * np.arcsin(w * rh), 1e-12)) - np.log(tgt)) ** 2)),
                                     bounds=(1e-6, 1.0), method="bounded")
        return float(r.x)

    def model_acf(self, k) -> np.ndarray:
        return (2 / np.pi) * np.arcsin(self.w * fgn_rho(np.asarray(k, dtype=float), self.H))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        w = self.w
        z = math.sqrt(w) * fgn(n, self.H, rng) + math.sqrt(1 - w) * rng.standard_normal(n)
        q = stats.norm.ppf(1 - self.p_buy)
        return np.where(z > q, 1, -1).astype(np.int8)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fgn_sign", "c": self.c, "gamma": self.gamma, "p_buy": self.p_buy, "H_latent": self.H, "w": self.w}


@dataclass
class TranslatedSignModel:
    """Gaussian-translation sign generator for an arbitrary target sign ACF a(k): signs = 1{Z > q} with Z a stationary
    Gaussian process whose correlation is rho(k) = sin(pi/2 a(k)) (the inverse of the arcsine law for p_buy = 1/2), drawn
    exactly by circulant embedding (negative eigenvalues clipped). a(k) = log-log interpolation of `short` (lag -> ACF,
    the empirical short lags, where order splitting makes the ACF much larger than the power-law fit) up to its last
    lag, then c k^-gamma. Reproduces strong short-lag persistence that FGNSignModel cannot."""
    c: float
    gamma: float
    p_buy: float = 0.5
    short: dict[int, float] = field(default_factory=dict)

    def target_acf(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = self.c * np.maximum(k, 1.0) ** (-self.gamma)
        sk = sorted((int(a), float(b)) for a, b in self.short.items() if float(b) > 0)
        if sk:
            lk = np.log([a for a, _ in sk])
            lv = np.log([b for _, b in sk])
            m = k <= sk[-1][0]
            out[m] = np.exp(np.interp(np.log(np.maximum(k[m], 1.0)), lk, lv))
        return np.clip(out, 0.0, 0.999)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        k = np.arange(1, n + 1, dtype=float)
        rho = np.r_[1.0, np.sin(np.pi / 2 * self.target_acf(k))]
        row = np.concatenate([rho, rho[-2:0:-1]])
        lam = np.clip(np.fft.fft(row).real, 0.0, None)
        m = len(row)
        z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        g = np.fft.fft(np.sqrt(lam / m) * z).real[:n]
        g /= g.std()
        q = stats.norm.ppf(1 - self.p_buy)
        return np.where(g > q, 1, -1).astype(np.int8)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "translated_sign", "c": self.c, "gamma": self.gamma, "p_buy": self.p_buy, "short": {str(a): float(b) for a, b in self.short.items()}}


@dataclass
class LMFSignModel:
    """Lillo-Mike-Farmer: n_meta concurrent metaorders with lengths P(L >= l) ~ l^-a (a = gamma + 1 gives sign ACF ~
    k^-gamma); each print picks an active metaorder uniformly, emits its sign and shortens it; finished ones are replaced
    by a fresh metaorder with a random sign (P(+) = p_buy)."""
    a: float
    n_meta: int = 10
    p_buy: float = 0.5

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        new_len = lambda: int(np.floor((1.0 - rng.random()) ** (-1.0 / self.a)))  # noqa: E731 - Pareto(a), L >= 1
        sg = np.where(rng.random(self.n_meta) < self.p_buy, 1, -1)
        left = np.array([new_len() for _ in range(self.n_meta)])
        pick = rng.integers(0, self.n_meta, n)
        out = np.empty(n, dtype=np.int8)
        for i in range(n):
            j = pick[i]
            out[i] = sg[j]
            left[j] -= 1
            if left[j] <= 0:
                sg[j] = 1 if rng.random() < self.p_buy else -1
                left[j] = new_len()
        return out


def dfa(x, scales=None, order: int = 1) -> dict[str, Any]:
    """DFA Hurst exponent: profile Y = cumsum(x - mean); F(s) = RMS of residuals of a degree-`order` polynomial fit in
    non-overlapping windows of size s; H = slope of log F on log s."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if scales is None:
        scales = np.unique(np.geomspace(10, max(n // 10, 20), 20).astype(int))
    Y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        m = n // s
        if m < 4:
            F.append(np.nan)
            continue
        seg = Y[: m * s].reshape(m, s)
        t = np.arange(s, dtype=float)
        V = np.vander(t, order + 1)
        coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        res = seg.T - V @ coef
        F.append(float(np.sqrt(np.mean(res ** 2))))
    F = np.array(F)
    ok = np.isfinite(F) & (F > 0)
    b = np.polyfit(np.log(np.asarray(scales)[ok]), np.log(F[ok]), 1)
    return {"H": float(b[0]), "scales": [int(s) for s in np.asarray(scales)[ok]], "F": [float(f) for f in F[ok]]}


# ----------------------------------------------------------------------------------------------------------------------
# Volume per interval, AFT densities
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class HurdleLaw:
    """Volume per interval: zero with probability p_zero, otherwise drawn from `law` (positive)."""
    p_zero: float
    law: Law

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        out = self.law.sample(n, rng)
        out[rng.random(n) < self.p_zero] = 0.0
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "hurdle", "p_zero": self.p_zero, "law": self.law.to_dict()}


def aft_logpdf(t: np.ndarray, mu: np.ndarray, sigma: float, dist: str) -> np.ndarray:
    """log density of T with log T = mu + sigma Z: normal, logistic or extreme (xgboost 'extreme' = Gumbel-min /
    log-Weibull: f_Z(z) = exp(z - e^z))."""
    lt = np.log(np.asarray(t, dtype=float))
    z = (lt - mu) / sigma
    if dist == "normal":
        lf = -0.5 * z * z - 0.5 * math.log(2 * math.pi)
    elif dist == "logistic":
        lf = -z - 2 * np.logaddexp(0.0, -z)
    elif dist == "extreme":
        lf = z - np.exp(z)
    else:
        raise ValueError(dist)
    return lf - math.log(sigma) - lt
