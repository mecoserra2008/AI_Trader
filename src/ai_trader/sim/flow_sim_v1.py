"""Event-level order-flow simulator v1 (H-MS8): generates taker ORDERS, splits them into prints, moves a latent mid with
their impact, and returns 3 s candles in the flow_3s `Grid` shape (so `sim.replay` runs on them) plus the trade list.

Generated data is for deterministic strategies and the execution engine only, never for the LLM layer (CLAUDE.md
principle 1).

Model (per symbol, all parameters in `V1Params`; fitted on the FIT half by `research.population_simulator_study`, or
extrapolated to any instrument by `sim.population.Population.params_for`):

  intensity  orders arrive at rate `rate` x m(t), log m(t) = u1(t) + u2(t) - (v1 + v2) / 2, uk independent Gaussian AR(1)
             on a 1 s grid with per-minute coefficient phik and variance vk (fast and slow activity components, fitted to
             the ACF of log order counts per bin; C1: arrivals cluster on a slow, near-integrated scale)
  arrivals   operational durations x_i i.i.d. from a 3-component exponential mixture with mean 1 (fitted on FIT durations
             deseasonalised by the local order rate), real time t_i = Lambda^-1(sum x / rate) with Lambda(t) = int m;
             the fitted ACDs are NOT used as generators (C1: near-integrated, explode or collapse)
  signs      TranslatedSignModel (Gaussian translation of the short empirical lags + power-law tail), event time
  sizes      order notional from the spliced law (log-normal body, Pareto tail above xmin), capped at `size_cap`, assigned
             to orders by the rank of a latent score (slow size level in time + |sign path| coupling + noise), so the
             marginal is exact while large orders cluster in time and inside one-sided runs (`order_stream`)
  prints     prints per order from the pmf {1..10, >10 Pareto}, rank-coupled to the order notional (Gaussian copula,
             Spearman `rho_kv`), notional split by a flat Dirichlet; print j of a buy is priced at the ask before the
             order plus one tick per `level_usd` of the order's notional already filled (sweeps)
  impact     each order moves the log mid by sign x k_imp x (v / v_ref)^delta bp, a share g_inf permanent and the rest
             decaying with time constant tau_s
  noise      mid innovations carried by the orders (trade time, Clark 1973 subordination; default) with variance
             sigma0^2 m(t)^beta per second on average, or Brownian in calendar time (noise_time = "clock")
  jumps      exogenous compound Poisson (rate jump_rate per second, sizes drawn from jump_pool, bp), each spread over an
             exponential profile with time constant jump_tau_s (a fast move over tens of seconds, as the 1-min
             exceedances they are calibrated on)
  spread     latent log spread in ticks, Gaussian AR(1) on the 1 s grid (mean spr_mu, sd spr_sd, coefficient spr_phi)
             plus spr_elast x log m(t); spread = max(1 tick, exp(latent)): censoring at one tick makes the tick-bound
             regime (latent mean below 0) and the free-spread regime the same process

Candles: mid at the candle close, spread = time average over the candle (1 s states), high / low = extreme print
prices (NaN without prints), buy / sell notional and print counts per side, obi = 0 (no book side is simulated),
depth = depth25 (constant). Everything is deterministic given the seed.

`calibrate` fits the parameters that have no direct estimator (size coupling, impact scale and exponent, decay, noise
level and activity elasticity, jump rate) by indirect inference: the same estimators are applied to real data and to
simulated data and the parameters are moved until they agree (`trade_stats` on prints for the flow targets,
`grid_stats` on 3 s book-mid candles for the price targets).
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.signal import lfilter

from ai_trader.sim.flow_data import STEP_MS, Grid
from ai_trader.sim.laws import SizeLaw, TranslatedSignModel

BIN_S = 0.1          # latent mid grid (s)
PMF_KEYS = [str(k) for k in range(1, 11)] + [">10"]


@dataclass
class V1Params:
    symbol: str = "SIM"
    asset_class: str = "alt"
    price0: float = 100.0
    tick_bps: float = 1.0                     # tick / price0 x 1e4
    # arrivals
    rate: float = 0.3                         # orders per second (mean)
    mix_w: list = field(default_factory=lambda: [0.3, 0.4, 0.3])
    mix_mean: list = field(default_factory=lambda: [0.02, 0.6, 2.4])   # component means, units of the mean duration
    int_v: list = field(default_factory=lambda: [0.10, 0.20])           # variances of the fast / slow log-intensity parts
    int_phi: list = field(default_factory=lambda: [0.5, 0.98])          # AR(1) coefficient per minute
    # sizes (USD notional of an order)
    size_mu: float = 5.0
    size_sigma: float = 2.0
    size_xmin: float = 5000.0
    size_alpha: float = 2.3                   # pdf exponent of the tail (CCDF index zeta = alpha - 1)
    size_p_tail: float = 0.02
    size_cap: float = 5e6
    size_xi_share: float = 0.0                # rank-copula share of a slow size level (AR(1) in time)
    size_xi_phi: float = 0.9                  # its per-minute AR coefficient
    size_g_share: float = 0.0                 # rank-copula share of |g| (size-sign-run coupling)
    # signs (order level)
    sign_c: float = 0.3
    sign_gamma: float = 0.6
    sign_short: dict = field(default_factory=lambda: {"1": 0.35, "2": 0.2, "3": 0.15, "5": 0.1, "10": 0.06, "20": 0.04})
    p_buy: float = 0.5
    # prints per order
    ppo_pmf: dict = field(default_factory=lambda: {"1": 0.5, "2": 0.15, "3": 0.08, "4": 0.06, "5": 0.05, "6": 0.04, "7": 0.03, "8": 0.025,
                                                    "9": 0.02, "10": 0.015, ">10": 0.04})
    ppo_tail_zeta: float = 1.0
    rho_kv: float = 0.5                       # Spearman(order notional, prints per order)
    level_usd: float = 2e4                    # notional per tick level when an order sweeps
    # impact / noise / jumps (bp)
    impact_kind: str = "pow"                  # pow (exponent impact_delta) | sqrt | linear
    impact_delta: float = 0.5
    k_imp: float = 0.2                        # bp per (v / v_ref)^delta
    imp_sat: float = 1e12                     # impact saturates above imp_sat x v_ref (concave tail)
    quote_eta: float = 0.25                   # hysteresis of the quoted book (ticks)
    v_ref: float = 100.0                      # reference notional (median order)
    g_inf: float = 0.6                        # permanent share of impact
    tau_s: float = 30.0                       # decay time of the transient share (s)
    sigma0: float = 0.3                       # diffusive noise, bp per sqrt(second) at m = 1
    noise_beta: float = 1.0                   # noise variance per second proportional to m(t)^beta (normalised per path)
    noise_time: str = "trade"                 # trade: innovations at order times | clock: Brownian in calendar time
    jump_rate: float = 0.0                    # per second
    jump_pool: list = field(default_factory=list)  # bp
    jump_tau_s: float = 20.0                  # a jump is spread over an exponential profile with this time constant
    # spread (ticks)
    spr_mu: float = -0.5                      # latent log-spread mean (log ticks)
    spr_sd: float = 0.3
    spr_phi: float = 0.9                      # AR(1) per second
    spr_elast: float = 0.07                   # d log spread / d log activity
    depth25: float = 1e6                      # bid + ask notional within 25 bp (constant, for the replay engine)
    info: dict = field(default_factory=dict)

    @property
    def tick(self) -> float:
        return self.tick_bps * self.price0 / 1e4

    def size_law(self) -> SizeLaw:
        return SizeLaw(self.size_mu, self.size_sigma, self.size_xmin, self.size_alpha, self.size_p_tail)

    def sign_model(self) -> TranslatedSignModel:
        return TranslatedSignModel(self.sign_c, self.sign_gamma, self.p_buy, {int(a): float(b) for a, b in self.sign_short.items()})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["jump_pool"] = [float(x) for x in self.jump_pool]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> V1Params:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in names})


@dataclass
class SimOut:
    grid: Grid
    trades: dict[str, np.ndarray]     # prints: ms, side (+1 / -1), price, notional, order
    orders: dict[str, np.ndarray]     # orders: ms, side, notional, n_prints
    m: np.ndarray                     # activity multiplier on the 1 s grid


# ---------------------------------------------------------------------------------------------------------- pieces

def _ar1(rng: np.random.Generator, n: int, phi: float, var: float) -> np.ndarray:
    """Stationary Gaussian AR(1), unit step, variance var."""
    if var <= 0 or n == 0:
        return np.zeros(n)
    phi = min(max(phi, 0.0), 0.999999)
    e = rng.standard_normal(n) * math.sqrt(var * (1 - phi * phi))
    y, _ = lfilter([1.0], [1.0, -phi], e, zi=[phi * rng.standard_normal() * math.sqrt(var)])
    return y


def activity_path(p: V1Params, seconds: int, rng: np.random.Generator) -> np.ndarray:
    """m(t) on the 1 s grid, mean one."""
    lm = np.zeros(seconds)
    vt = 0.0
    for v, ph in zip(p.int_v, p.int_phi, strict=True):
        lm += _ar1(rng, seconds, float(ph) ** (1 / 60), float(v))
        vt += float(v)
    return np.exp(lm - 0.5 * vt)


def mixture_sample(w, means, n: int, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    k = rng.choice(len(w), n, p=w)
    return rng.exponential(1.0, n) * np.asarray(means, dtype=float)[k]


def arrival_times(p: V1Params, m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order times (s) in [0, len(m)) from the operational-time mixture on activity m."""
    seconds = len(m)
    Lam = np.r_[0.0, np.cumsum(m)]                      # Lambda at integer seconds
    total = Lam[-1] * p.rate
    out = []
    tau0 = 0.0
    while True:
        n = int(total - tau0 + 10 * math.sqrt(max(total - tau0, 1.0)) + 50)
        x = mixture_sample(p.mix_w, p.mix_mean, n, rng)
        tau = tau0 + np.cumsum(x)
        out.append(tau)
        tau0 = float(tau[-1])
        if tau0 >= total:
            break
    tau = np.concatenate(out)
    tau = tau[tau < total] / p.rate
    return np.interp(tau, Lam, np.arange(seconds + 1, dtype=float))


def prints_per_order(p: V1Params, notional: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(notional)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pmf = np.array([float(p.ppo_pmf.get(k, 0.0)) for k in PMF_KEYS])
    pmf = pmf / pmf.sum()
    cdf = np.cumsum(pmf)
    rho = 2 * math.sin(math.pi * float(np.clip(p.rho_kv, -0.99, 0.99)) / 6)
    zv = stats.norm.ppf((stats.rankdata(notional) - 0.5) / n)
    u = stats.norm.cdf(rho * zv + math.sqrt(max(1 - rho * rho, 0.0)) * rng.standard_normal(n))
    k = np.searchsorted(cdf, np.clip(u, 0, 1 - 1e-12)) + 1
    big = k == 11
    if big.any():
        z = max(float(p.ppo_tail_zeta), 0.3)
        k[big] = np.minimum(np.ceil(10.0 * (1 - rng.random(int(big.sum()))) ** (-1 / z)).astype(np.int64), 400)
        k[big] = np.maximum(k[big], 11)
    return k.astype(np.int64)


def quote_path(c: np.ndarray, eta: float) -> np.ndarray:
    """Integer bid index following a continuous target c (in ticks) with hysteresis: it moves to round(c) only when
    |c - current| > 0.5 + eta (uncertainty zones, Robert and Rosenbaum 2011); eta = 0 is plain rounding."""
    out = np.empty(len(c), dtype=np.int64)
    if len(c) == 0:
        return out
    if eta <= 0:
        return np.rint(c).astype(np.int64)
    thr = 0.5 + eta
    cur = int(round(float(c[0])))
    cl = c.tolist()
    for i, x in enumerate(cl):
        if abs(x - cur) > thr:
            cur = int(round(x))
        out[i] = cur
    return out


def _seeds(rng: np.random.Generator | int, k: int) -> list[np.random.Generator]:
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    return [np.random.default_rng(s) for s in np.random.SeedSequence(int(rng.integers(2**63))).spawn(k)]


def sign_path(p: V1Params, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Signs (+1 / -1) and the latent unit Gaussian path g of the TranslatedSignModel (sign = 1{g > q}); the same
    construction as sim.laws.TranslatedSignModel.sample, which does not expose g."""
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    tm = p.sign_model()
    k = np.arange(1, n + 1, dtype=float)
    rho = np.r_[1.0, np.sin(np.pi / 2 * tm.target_acf(k))]
    row = np.concatenate([rho, rho[-2:0:-1]])
    lam = np.clip(np.fft.fft(row).real, 0.0, None)
    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    g = np.fft.fft(np.sqrt(lam / m) * z).real[:n]
    if n > 1 and g.std() > 0:
        g = g / g.std()
    q = stats.norm.ppf(1 - p.p_buy)
    return np.where(g > q, 1, -1).astype(np.int64), g


def _nscore(x: np.ndarray) -> np.ndarray:
    return stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x)) if len(x) else x


def order_stream(p: V1Params, seconds: int, rng: np.random.Generator | int) -> dict[str, np.ndarray]:
    """Order times (s), notional, signs, latent sign path g and the activity path m (1 s grid).
    Sizes keep their spliced marginal exactly and are assigned by the rank of a latent score
    h = sqrt(a) xi(t) + sqrt(b) nscore(|g - q|) + sqrt(1 - a - b) e: xi = unit AR(1) in time (per-minute coefficient
    size_xi_phi), a = size_xi_share (slow size level), b = size_g_share (large orders when the sign path is far from its
    threshold, i.e. inside persistent one-sided runs: the coupling that makes clock-time signed notional autocorrelated)."""
    r_act, r_arr, r_size, r_sign, r_h, r_xi = _seeds(rng, 6)
    seconds = int(seconds // 3 * 3)
    m = activity_path(p, seconds, r_act)
    t = arrival_times(p, m, r_arr)
    n = len(t)
    v = np.minimum(p.size_law().sample(n, r_size), p.size_cap) if n else np.zeros(0)
    s, g = sign_path(p, n, r_sign)
    a = float(np.clip(p.size_xi_share, 0.0, 0.95))
    bb = float(np.clip(p.size_g_share, 0.0, 0.95 - a))
    if n > 1 and (a > 0 or bb > 0):
        xi = _ar1(r_xi, seconds, float(p.size_xi_phi) ** (1 / 60), 1.0)
        h = math.sqrt(a) * xi[np.minimum(t.astype(np.int64), seconds - 1)]
        h = h + math.sqrt(bb) * _nscore(np.abs(g - stats.norm.ppf(1 - p.p_buy)))
        h = h + math.sqrt(max(1.0 - a - bb, 0.0)) * r_h.standard_normal(n)
        v = np.sort(v)[stats.rankdata(h, method="ordinal").astype(np.int64) - 1]
    return {"t": t, "v": v, "s": s, "g": g, "m": m}


# ---------------------------------------------------------------------------------------------------------- generate

def generate(p: V1Params, seconds: int, rng: np.random.Generator | int, *, t0_ms: int = 0, mid0: float | None = None,
             jumps: bool = True, stream: dict | None = None) -> SimOut:
    """`seconds` of flow (a multiple of 3 is used); returns candles, prints and orders. Deterministic given the seed."""
    seconds = int(seconds // 3 * 3)
    r_stream, r_ppo, r_split, r_noise, r_jump, r_spr = _seeds(rng, 6)
    st = stream if stream is not None else order_stream(p, seconds, r_stream)
    t, v, s, m = st["t"], st["v"], st["s"], st["m"]
    n = len(t)
    mid0 = float(p.price0 if mid0 is None else mid0)
    tick = p.tick_bps * mid0 / 1e4
    # latent mid on the 100 ms grid
    nb = int(round(seconds / BIN_S))
    b = np.minimum((t / BIN_S).astype(np.int64), nb - 1)
    delta = 0.5 if p.impact_kind == "sqrt" else 1.0 if p.impact_kind == "linear" else float(p.impact_delta)
    imp = s * p.k_imp * (np.minimum(v, p.imp_sat * p.v_ref) / p.v_ref) ** delta
    ib = np.bincount(b, imp, minlength=nb)
    perm = np.cumsum(p.g_inf * ib)
    trans = lfilter([1.0 - p.g_inf], [1.0, -math.exp(-BIN_S / max(p.tau_s, 1e-6))], ib) if p.g_inf < 1 else np.zeros(nb)
    if p.noise_time == "trade" and n > 0:
        # innovations carried by the orders themselves (subordination: the price moves when trades happen); per-order
        # variance prop. to m^(beta - 1), so the variance per second is prop. to m^beta; mean variance per second = sigma0^2
        wv = m[np.minimum(t.astype(np.int64), seconds - 1)] ** (float(p.noise_beta) - 1.0)
        wv = wv / wv.mean()
        e_o = r_noise.standard_normal(n) * p.sigma0 * math.sqrt(seconds / n) * np.sqrt(wv)
        noise = np.cumsum(np.bincount(b, e_o, minlength=nb))
    else:
        mb = m ** float(p.noise_beta)
        mb = mb / mb.mean()
        m_bin = np.repeat(mb, int(round(1 / BIN_S)))[:nb]
        noise = np.cumsum(r_noise.standard_normal(nb) * p.sigma0 * np.sqrt(BIN_S * m_bin))
    jmp = np.zeros(nb)
    if jumps and p.jump_rate > 0 and len(p.jump_pool):
        nj = r_jump.poisson(p.jump_rate * seconds)
        if nj:
            pos = r_jump.integers(0, nb, nj)
            pool = np.asarray(p.jump_pool, dtype=float)
            np.add.at(jmp, pos, pool[r_jump.integers(0, len(pool), nj)])
    if p.jump_tau_s > 0 and jmp.any():
        # a jump unfolds over about jump_tau_s (exponential profile, total size preserved), not inside one 100 ms bin
        aj = math.exp(-BIN_S / p.jump_tau_s)
        jmp = lfilter([1.0 - aj], [1.0, -aj], jmp)
    lmid = perm + trans + noise + np.cumsum(jmp)          # bp, state at the END of each 100 ms bin
    mid = mid0 * np.exp(lmid / 1e4)
    # spread (ticks) on the 1 s grid
    phi = min(max(p.spr_phi, 0.0), 0.9999)
    e = r_spr.standard_normal(seconds) * p.spr_sd * math.sqrt(1 - phi * phi)
    ls, _ = lfilter([1.0], [1.0, -phi], e, zi=[phi * r_spr.standard_normal() * p.spr_sd])
    ls = p.spr_mu + ls + p.spr_elast * np.log(m)
    spr_t = np.maximum(1.0, np.exp(ls))                    # latent spread, ticks
    n_spr = np.maximum(1, np.rint(spr_t)).astype(np.int64)  # quoted spread, integer ticks
    # quoted book on the 1 s grid: bid index moves only when the latent leaves its zone by eta ticks (Robert-Rosenbaum)
    lat1 = mid[np.arange(1, seconds + 1) * int(round(1 / BIN_S)) - 1] / tick
    bid_i = quote_path(lat1 - n_spr / 2.0, p.quote_eta)
    q_bid = bid_i * tick
    q_ask = q_bid + n_spr * tick
    q_mid = 0.5 * (q_bid + q_ask)
    # prints
    k = prints_per_order(p, v, r_ppo)
    oid = np.repeat(np.arange(n), k)
    w = r_split.exponential(1.0, len(oid))
    tot = np.bincount(oid, w, minlength=n)
    pn = v[oid] * w / tot[oid] if n else np.zeros(0)        # print notional
    if n:
        cum = np.cumsum(pn)
        first = np.r_[0, np.cumsum(k)[:-1]]
        before = cum - pn - (cum[first] - pn[first])[oid]   # order notional filled before this print
    else:
        before = np.zeros(0)
    lvl = np.floor(before / max(p.level_usd, 1e-9))
    sec_prev = np.maximum(t.astype(np.int64) - 1, 0)            # quote state at the end of the previous second
    touch = np.where(s > 0, q_ask[sec_prev], q_bid[sec_prev])
    side_p = s[oid]
    price = np.maximum(touch[oid] + side_p * lvl * tick, tick)
    ms = (t0_ms + np.floor(t * 1000)).astype(np.int64)
    # candles
    nc = seconds // 3
    c = np.minimum((t / 3.0).astype(np.int64), nc - 1)[oid]
    buy = np.bincount(c, np.where(side_p > 0, pn, 0.0), minlength=nc)
    sell = np.bincount(c, np.where(side_p < 0, pn, 0.0), minlength=nc)
    n_buy = np.bincount(c, side_p > 0, minlength=nc).astype(float)
    n_sell = np.bincount(c, side_p < 0, minlength=nc).astype(float)
    hi = np.full(nc, -np.inf)
    lo = np.full(nc, np.inf)
    np.maximum.at(hi, c, price)
    np.minimum.at(lo, c, price)
    has = np.isfinite(hi)
    hi = np.where(has, hi, np.nan)
    lo = np.where(has, lo, np.nan)
    mid_c = q_mid[np.arange(1, nc + 1) * 3 - 1]
    spr_bps = (n_spr * tick / q_mid * 1e4).reshape(nc, 3).mean(axis=1)
    z = np.zeros(nc)
    g = Grid(p.symbol, t0_ms + np.arange(nc, dtype=np.int64) * STEP_MS, np.ones(nc, bool), np.ones(nc, bool), mid_c, spr_bps, hi, lo,
             buy, sell, n_buy, n_sell, z, z.copy(), np.full(nc, float(p.depth25)))
    trades = {"ms": ms[oid], "side": side_p.astype(np.int8), "price": price, "notional": pn, "order": oid}
    orders = {"ms": ms, "side": s.astype(np.int8), "notional": v, "n_prints": k}
    return SimOut(g, trades, orders, m)


# ------------------------------------------------------------------------------------- trade-based estimators

def merge_orders(ms: np.ndarray, side: np.ndarray, notional: np.ndarray) -> dict[str, np.ndarray]:
    """Orders = maximal runs of consecutive prints with the same ms and taker side (C1); prints time-sorted."""
    if len(ms) == 0:
        return {"ms": ms, "side": side, "notional": notional, "n_prints": np.zeros(0, dtype=np.int64)}
    new = np.ones(len(ms), dtype=bool)
    new[1:] = (ms[1:] != ms[:-1]) | (side[1:] != side[:-1])
    gid = np.cumsum(new) - 1
    st = np.flatnonzero(new)
    return {"ms": ms[st], "side": side[st], "notional": np.bincount(gid, notional), "n_prints": np.bincount(gid)}


def mid_proxy(ms: np.ndarray, side: np.ndarray, price: np.ndarray, t0_ms: int, seconds: int, bin_s: int = 1) -> np.ndarray:
    """(last buy price + last sell price) / 2 at the end of each bin, carried forward (NaN until both sides traded).
    Prints must be time-sorted."""
    nb = seconds // bin_s
    bi = ((ms - t0_ms) // (bin_s * 1000)).astype(np.int64)
    keep = (bi >= 0) & (bi < nb)
    out = []
    for sd in (1, -1):
        sel = keep & (side == sd)
        last = np.full(nb, np.nan)
        bsel, psel = bi[sel], price[sel]
        if len(bsel):
            endb = np.flatnonzero(np.r_[bsel[1:] != bsel[:-1], True])
            last[bsel[endb]] = psel[endb]
        idx = np.where(np.isfinite(last), np.arange(nb), -1)
        idx = np.maximum.accumulate(idx)
        out.append(np.where(idx >= 0, last[np.maximum(idx, 0)], np.nan))
    return 0.5 * (out[0] + out[1])


def last_price(ms: np.ndarray, price: np.ndarray, t0_ms: int, seconds: int, bin_s: int = 1) -> np.ndarray:
    nb = seconds // bin_s
    bi = ((ms - t0_ms) // (bin_s * 1000)).astype(np.int64)
    keep = (bi >= 0) & (bi < nb)
    bsel, psel = bi[keep], price[keep]
    last = np.full(nb, np.nan)
    if len(bsel):
        endb = np.flatnonzero(np.r_[bsel[1:] != bsel[:-1], True])
        last[bsel[endb]] = psel[endb]
    idx = np.maximum.accumulate(np.where(np.isfinite(last), np.arange(nb), -1))
    return np.where(idx >= 0, last[np.maximum(idx, 0)], np.nan)


def _ols_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 30:
        return float("nan"), float("nan")
    xx, yy = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    sxx = float(xx @ xx)
    if sxx <= 0:
        return float("nan"), float("nan")
    b = float(xx @ yy) / sxx
    e = yy - b * xx
    se = math.sqrt(float((xx * xx) @ (e * e))) / sxx   # HC0
    return b, se


def acf_mean(x: np.ndarray, lags: int) -> float:
    """Mean sample ACF over lags 1..lags (biased denominator, NaN dropped pairwise by zero-filling the demeaned series)."""
    ok = np.isfinite(x)
    if ok.sum() < 3 * lags:
        return float("nan")
    y = np.where(ok, x - np.nanmean(x), 0.0)
    v = float(y @ y)
    if v <= 0:
        return float("nan")
    return float(np.mean([float(y[k:] @ y[:-k]) / v for k in range(1, lags + 1)]))


def impact_curve(order_ms: np.ndarray, order_side: np.ndarray, order_v: np.ndarray, mp1: np.ndarray, t0_ms: int,
                 n_bins: int = 8) -> dict[str, float]:
    """Mean signed mid-proxy move from the second before to the second after an order, by order-notional bin
    (equal-count bins of log notional); delta = OLS slope of log mean move on log bin median notional (bins with a
    positive mean move, at least 4)."""
    sec = ((order_ms - t0_ms) // 1000).astype(np.int64)
    ok = (sec >= 1) & (sec < len(mp1) - 1)
    sec, sd, v = sec[ok], order_side[ok], order_v[ok]
    if len(v) < 200:
        return {"delta": float("nan"), "top_move": float("nan")}
    resp = sd * 1e4 * (mp1[sec + 1] / mp1[sec - 1] - 1.0)
    f = np.isfinite(resp)
    resp, v = resp[f], v[f]
    edges = np.quantile(np.log(v), np.linspace(0, 1, n_bins + 1))
    bi = np.clip(np.searchsorted(edges, np.log(v), side="right") - 1, 0, n_bins - 1)
    mv = np.array([resp[bi == j].mean() if (bi == j).any() else np.nan for j in range(n_bins)])
    xv = np.array([np.median(v[bi == j]) if (bi == j).any() else np.nan for j in range(n_bins)])
    good = np.isfinite(mv) & (mv > 0) & np.isfinite(xv)
    if good.sum() < 4:
        return {"delta": float("nan"), "top_move": float(mv[-1]), "moves": mv.tolist()}
    b = np.polyfit(np.log(xv[good]), np.log(mv[good]), 1)
    return {"delta": float(b[0]), "top_move": float(mv[-1]), "moves": mv.tolist(), "bin_v": xv.tolist()}


def trade_stats(ms: np.ndarray, side: np.ndarray, price: np.ndarray, notional: np.ndarray, t0_ms: int, seconds: int,
                jump_k: float = 6.0) -> dict[str, Any]:
    """Estimators shared by real FIT trades and simulated trades (indirect-inference targets). Prints time-sorted.

    lambda_mid / lambda_trade: OLS slope of the 1 s log return (bp) of the mid proxy / last trade price on 1 s signed
    notional (USD 1k), HC0 SE; sigma_1m: std of 1-min mid-proxy log returns (bp), sigma_day = sigma_1m x sqrt(1440);
    vr5 = Var(5-min) / (5 Var(1-min)); exceedances: 1-min returns beyond jump_k robust sd (1.4826 MAD), rate per
    second and values (bp); flow_acf_3s: mean ACF lags 1-20 of 3 s signed notional; absr_acf_3s: mean ACF lags 1-60 of
    |3 s mid-proxy return|; size_ns_acf1 / 10: ACF of the normal scores of order notional at order lags 1 / 10;
    delta: impact-curve exponent (impact_curve)."""
    seconds = int(seconds)
    mp1 = mid_proxy(ms, side, price, t0_ms, seconds, 1)
    lp1 = last_price(ms, price, t0_ms, seconds, 1)
    bi = ((ms - t0_ms) // 1000).astype(np.int64)
    keep = (bi >= 0) & (bi < seconds)
    F = np.bincount(bi[keep], (side * notional)[keep], minlength=seconds)[:seconds] / 1e3
    r_mid = np.r_[np.nan, 1e4 * np.diff(np.log(mp1))]
    r_tr = np.r_[np.nan, 1e4 * np.diff(np.log(lp1))]
    lam_mid, se_mid = _ols_slope(r_mid, F)
    lam_tr, se_tr = _ols_slope(r_tr, F)
    sq = np.sign(F) * np.sqrt(np.abs(F))
    lsq_mid, _ = _ols_slope(r_mid, sq)
    ok = np.isfinite(r_mid)
    r2_lin = float(np.corrcoef(r_mid[ok], F[ok])[0, 1] ** 2) if ok.sum() > 30 else float("nan")
    r2_sqrt = float(np.corrcoef(r_mid[ok], sq[ok])[0, 1] ** 2) if ok.sum() > 30 else float("nan")
    m60 = mp1[59::60]
    r1m = 1e4 * np.diff(np.log(m60))
    r1m = r1m[np.isfinite(r1m)]
    m300 = mp1[299::300]
    r5m = 1e4 * np.diff(np.log(m300))
    r5m = r5m[np.isfinite(r5m)]
    s1 = float(np.std(r1m)) if len(r1m) > 10 else float("nan")
    vr5 = float(np.var(r5m) / (5 * np.var(r1m))) if len(r5m) > 10 and len(r1m) > 10 and np.var(r1m) > 0 else float("nan")
    rob = 1.4826 * float(np.median(np.abs(r1m - np.median(r1m)))) if len(r1m) else float("nan")
    exc = r1m[np.abs(r1m - np.median(r1m)) > jump_k * rob] if rob > 0 else np.zeros(0)
    # 3 s clock facts from trades
    nc = seconds // 3
    c3 = ((ms - t0_ms) // 3000).astype(np.int64)
    k3 = (c3 >= 0) & (c3 < nc)
    F3 = np.bincount(c3[k3], (side * notional)[k3], minlength=nc)[:nc]
    mp3 = mp1[2::3][:nc]
    r3 = np.r_[np.nan, 1e4 * np.diff(np.log(mp3))]
    od = merge_orders(ms[keep], side[keep], notional[keep])
    ns = _nscore(od["notional"]) if len(od["notional"]) > 50 else np.zeros(0)
    ic = impact_curve(od["ms"], od["side"], od["notional"], mp1, t0_ms) if len(od["ms"]) > 200 else {"delta": float("nan")}

    def _acfk(x, k):
        return float(np.corrcoef(x[k:], x[:-k])[0, 1]) if len(x) > k + 30 else float("nan")

    rr = r3[np.isfinite(r3)]
    kurt = float(np.mean((rr - rr.mean()) ** 4) / np.var(rr) ** 2 - 3) if len(rr) > 100 and np.var(rr) > 0 else float("nan")

    return {"lambda_mid": lam_mid, "lambda_mid_se": se_mid, "lambda_trade": lam_tr, "lambda_trade_se": se_tr,
            "lambda_sqrt_mid": lsq_mid, "r2_lin_1s": r2_lin, "r2_sqrt_1s": r2_sqrt,
            "sigma_1m": s1, "sigma_day": s1 * math.sqrt(1440) if np.isfinite(s1) else float("nan"), "vr5": vr5,
            "robust_sd_1m": rob, "exc_rate": float(len(exc) / seconds), "exc_values": exc.tolist(),
            "corr_F_r1s": float(np.corrcoef(r_mid[ok], F[ok])[0, 1]) if ok.sum() > 30 else float("nan"),
            "flow_acf_3s": acf_mean(F3, 20), "absr_acf_3s": acf_mean(np.abs(r3), 60),
            "corr_F_r3s": float(np.corrcoef(r3[np.isfinite(r3)], F3[np.isfinite(r3)])[0, 1]) if np.isfinite(r3).sum() > 30 else float("nan"),
            "kurt_3s": kurt, "size_ns_acf1": _acfk(ns, 1), "size_ns_acf10": _acfk(ns, 10), "delta": ic["delta"], "impact_moves": ic.get("moves"),
            "n_orders": int(len(od["ms"]))}


def sim_trade_stats(out: SimOut, seconds: int, t0_ms: int = 0) -> dict[str, Any]:
    tr = out.trades
    o = np.argsort(tr["ms"], kind="stable")
    return trade_stats(tr["ms"][o], tr["side"][o].astype(np.int64), tr["price"][o], tr["notional"][o], t0_ms, seconds)


# ------------------------------------------------------------------------------------------------- calibration

def grid_stats(g: Grid, mask: np.ndarray | None = None, min_len: int = 100, jump_k: float = 6.0) -> dict[str, Any]:
    """Book-mid statistics of a 3 s Grid (real flow_3s or simulated), within contiguous ok segments: the H-SIM2 facts
    (kurtosis, |r| ACF, flow ACF, impact = corr(r, F), spread median / p90) plus sigma_3s, sigma_1m (20-candle returns),
    vr_3s_1m = Var(1 m) / (20 Var(3 s)), vr5 = Var(5 m) / (5 Var(1 m)) and 1-min exceedances beyond jump_k robust sd
    (rate per second, values in units of the robust sd)."""
    from ai_trader.sim import flow_sim as fs

    f, _ = fs.facts_of_grid(g, mask, min_len)
    gg = g if mask is None else g.slice(mask)
    r3, r1, r5 = [], [], []
    for a, b in gg.segments(min_len):
        lm = np.log(gg.mid[a:b])
        r3.append(1e4 * np.diff(lm))
        r1.append(1e4 * np.diff(lm[::20]))
        r5.append(1e4 * np.diff(lm[::100]))

    def cat(x):
        return np.concatenate(x) if x else np.zeros(0)

    r3, r1, r5 = cat(r3), cat(r1), cat(r5)
    v3, v1, v5 = (float(np.var(x)) if len(x) > 10 else float("nan") for x in (r3, r1, r5))
    rob = 1.4826 * float(np.median(np.abs(r1 - np.median(r1)))) if len(r1) > 10 else float("nan")
    if not (np.isfinite(rob) and rob > 0):
        rob = float(np.std(r1)) if len(r1) > 10 else float("nan")
    exc = r1[np.abs(r1 - np.median(r1)) > jump_k * rob] if np.isfinite(rob) and rob > 0 else np.zeros(0)
    secs = 3.0 * len(r3)
    return {**f, "sigma_3s": math.sqrt(v3) if np.isfinite(v3) else float("nan"), "sigma_1m": math.sqrt(v1) if np.isfinite(v1) else float("nan"),
            "vr_3s_1m": v1 / (20 * v3) if np.isfinite(v3) and v3 > 0 else float("nan"),
            "vr5": v5 / (5 * v1) if np.isfinite(v1) and v1 > 0 else float("nan"),
            "robust_sd_1m": rob, "exc_rate": float(len(exc) / secs) if secs > 0 else 0.0,
            "exc_z": (exc / rob).tolist() if np.isfinite(rob) and rob > 0 else [], "n_3s": int(len(r3))}


def _stream_stats(p: V1Params, seconds: int, seeds: list[int]) -> dict[str, float]:
    """Size-ACF and 3 s flow-ACF of order streams only (no prices needed)."""
    a1, a10, fa = [], [], []
    for sd in seeds:
        st = order_stream(p, seconds, sd)
        ns = _nscore(st["v"])
        a1.append(float(np.corrcoef(ns[1:], ns[:-1])[0, 1]) if len(ns) > 40 else np.nan)
        a10.append(float(np.corrcoef(ns[10:], ns[:-10])[0, 1]) if len(ns) > 60 else np.nan)
        c3 = (st["t"] / 3.0).astype(np.int64)
        fa.append(acf_mean(np.bincount(c3, st["s"] * st["v"], minlength=seconds // 3)[: seconds // 3], 20))
    return {"size_ns_acf1": float(np.nanmean(a1)), "size_ns_acf10": float(np.nanmean(a10)), "flow_acf_3s": float(np.nanmean(fa))}


PRICE_TARGETS = ("sigma_1m", "impact", "kurtosis", "absr_acf", "vr_3s_1m", "vr5", "exc_rate")


def calibrate(p: V1Params, target: dict[str, float], *, seconds: int = 4 * 3600, iters: int = 7, seed: int = 11,
              jump_z: list | None = None, n_pilots: int = 2, fit_stream: bool = True) -> tuple[V1Params, dict]:
    """Indirect inference on pilots with common seeds; missing / non-finite targets leave their parameter unchanged.

    Stage A (order stream only, trade-space targets): size_xi_share to the normal-score size ACF at order lag 10
    (size_ns_acf10) and size_g_share to the 3 s signed-flow ACF (flow_acf_3s).
    Stage B (book-mid space, the generator's own Grid and `grid_stats`): sigma0 to sigma_1m (with the impact-only
    variance measured on a noise-free pilot; the impact may carry at most 80% of it), k_imp to the 3 s flow-return
    correlation (impact), impact_delta to the 3 s kurtosis, tau_s to vr_3s_1m, g_inf to vr5, noise_beta to the |r| ACF
    (absr_acf on the grid, or absr_acf_3s of the trade mid proxy via `trade_stats` when only that is given), jump_rate to the 1-min exceedance rate (exc_rate) with jump sizes = jump_z x the target robust sd."""
    q = copy.deepcopy(p)
    seeds = [seed + 101 * j for j in range(n_pilots)]

    def tg(k):
        v = target.get(k)
        return float(v) if v is not None and np.isfinite(v) else None

    hist_a = []
    if fit_stream:
        t10, tf = tg("size_ns_acf10"), tg("flow_acf_3s")
        for it in range(iters):
            ss = _stream_stats(q, seconds, seeds)
            hist_a.append({"iter": it, "xi": q.size_xi_share, "g": q.size_g_share, **ss})
            if t10 is not None and np.isfinite(ss["size_ns_acf10"]):
                q.size_xi_share = float(np.clip(q.size_xi_share + 1.5 * (t10 - ss["size_ns_acf10"]), 0.0, 0.6))
            if tf is not None and np.isfinite(ss["flow_acf_3s"]):
                q.size_g_share = float(np.clip(q.size_g_share + 6.0 * (tf - ss["flow_acf_3s"]), 0.0, 0.9 - q.size_xi_share))
    T = {k: tg(k) for k in PRICE_TARGETS}
    rob_t = tg("robust_sd_1m")
    if jump_z is not None and len(jump_z) and rob_t:
        q.jump_pool = [float(z) * rob_t for z in jump_z]
    hist = []
    for it in range(iters + 1):
        outs = [generate(q, seconds, sd) for sd in seeds]
        st = [grid_stats(o.grid) for o in outs]
        S = {k: float(np.nanmean([x[k] for x in st])) for k in PRICE_TARGETS}
        S["absr_acf_3s"] = float(np.nanmean([sim_trade_stats(o, seconds)["absr_acf_3s"] for o in outs])) if tg("absr_acf_3s") is not None else float("nan")
        hist.append({"iter": it, "k_imp": q.k_imp, "delta": q.impact_delta, "g_inf": q.g_inf, "tau_s": q.tau_s, "sigma0": q.sigma0,
                     "beta": q.noise_beta, "jump_rate": q.jump_rate, **{f"sim_{k}": v for k, v in S.items()}})
        if it == iters:
            break
        q0 = copy.deepcopy(q)
        q0.sigma0, q0.jump_rate = 0.0, 0.0
        v_imp = float(np.nanmean([grid_stats(generate(q0, seconds, sd, jumps=False).grid)["sigma_1m"] ** 2 for sd in seeds]))
        k_old = q.k_imp
        cr, cr_t = S["impact"], T["impact"]
        if cr_t is not None and np.isfinite(cr) and 0 < cr < 1:
            q.k_imp *= float(np.clip((max(cr_t, 0.01) / (1 - min(cr_t, 0.99))) / (cr / (1 - cr)), 0.25, 4.0))
        s_t = T["sigma_1m"]
        if s_t is not None and np.isfinite(v_imp):
            v_new = v_imp * (q.k_imp / k_old) ** 2 if k_old > 0 else 0.0
            if v_new > 0.8 * s_t * s_t and v_imp > 0:
                q.k_imp = k_old * math.sqrt(0.8 * s_t * s_t / v_imp)
                v_new = 0.8 * s_t * s_t
            pool = np.asarray(q.jump_pool, dtype=float)
            v_jump = q.jump_rate * 60.0 * float(np.mean(pool ** 2)) if len(pool) else 0.0
            v_jump = min(v_jump, 0.5 * s_t * s_t)
            q.sigma0 = math.sqrt(max(s_t * s_t - v_new - v_jump, 0.02 * s_t * s_t) / 60.0)
        ku, ku_t = S["kurtosis"], T["kurtosis"]
        if q.impact_kind == "pow" and ku_t is not None and np.isfinite(ku) and ku > 0 and ku_t > 0:
            q.impact_delta = float(np.clip(q.impact_delta + 0.12 * math.log(ku_t / ku), 0.1, 1.0))
        vr, vr_t = S["vr_3s_1m"], T["vr_3s_1m"]
        if vr_t is not None and np.isfinite(vr) and vr > 0:
            q.tau_s = float(np.clip(q.tau_s * (vr_t / vr) ** 3, 3.0, 1800.0))
        vr, vr_t = S["vr5"], T["vr5"]
        if vr_t is not None and np.isfinite(vr) and vr > 0:
            q.g_inf = float(np.clip(q.g_inf * (vr_t / vr) ** 2, 0.05, 1.0))
        ar, ar_t = (S["absr_acf"], T["absr_acf"]) if T["absr_acf"] is not None else (S["absr_acf_3s"], tg("absr_acf_3s"))
        if ar_t is not None and np.isfinite(ar):
            q.noise_beta = float(np.clip(q.noise_beta + 6.0 * (ar_t - ar), 0.0, 3.0))
        ex, ex_t = S["exc_rate"], T["exc_rate"]
        if q.jump_pool and ex_t is not None and np.isfinite(ex):
            q.jump_rate = float(max(q.jump_rate + 0.7 * (ex_t - ex), 0.0))
    final = hist[-1]
    fin = {k: final[f"sim_{k}"] for k in (*PRICE_TARGETS, "absr_acf_3s")}
    if hist_a:
        fin.update({k: hist_a[-1][k] for k in ("size_ns_acf1", "size_ns_acf10", "flow_acf_3s")})
    q.info = {**q.info, "calibration": {"target": {**T, "size_ns_acf10": tg("size_ns_acf10"), "flow_acf_3s": tg("flow_acf_3s"),
                                                   "absr_acf_3s": tg("absr_acf_3s")},
                                        "stream_history": hist_a, "history": hist, "final": fin}}
    return q, fin


# ------------------------------------------------------------------------------------------------- fitting helpers

def fit_intensity(order_ms: np.ndarray, t0_ms: int, seconds: int, min_mean: float = 10.0, max_lag_min: int = 120) -> dict[str, Any]:
    """Two-component AR fit of log order counts: bins of b minutes (b >= 1 so that the mean count >= min_mean);
    ACF(k) = a1 phi1^k + a2 phi2^k on lags 1..max_lag (least squares); vk = ak Var(y); phi converted to per minute."""
    from scipy.optimize import minimize

    rate_min = len(order_ms) / seconds * 60.0
    bmin = max(1, int(math.ceil(min_mean / max(rate_min, 1e-9))))
    bs = bmin * 60
    nb = seconds // bs
    bi = ((order_ms - t0_ms) // (bs * 1000)).astype(np.int64)
    bi = bi[(bi >= 0) & (bi < nb)]
    cnt = np.bincount(bi, minlength=nb).astype(float)
    y = np.log(cnt + 0.5)
    vy = float(np.var(y))
    L = max(3, min(max_lag_min // bmin, nb // 3))
    yc = y - y.mean()
    acf = np.array([float(yc[k:] @ yc[:-k]) / (len(yc) * vy) for k in range(1, L + 1)]) if vy > 0 else np.zeros(L)
    k = np.arange(1, L + 1, dtype=float)

    def unpack(z):
        z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
        a = 1 / (1 + np.exp(-z[:2]))
        a1, a2 = 0.98 * a[0] * a[1], 0.98 * a[0] * (1 - a[1])
        f1 = float(1 / (1 + np.exp(-z[2])))
        f2 = f1 + (1 - f1) / float(1 + np.exp(-z[3]))
        return a1, a2, f1, min(f2, 0.99999)

    def loss(z):
        a1, a2, f1, f2 = unpack(z)
        return float(np.sum((acf - a1 * f1 ** k - a2 * f2 ** k) ** 2))

    best = None
    for z0 in ([0.0, 0.0, 0.0, 1.0], [1.0, -1.0, -1.0, 2.0], [1.0, 1.0, 1.0, 3.0], [-1.0, 0.0, -2.0, 4.0]):
        r = minimize(loss, np.array(z0), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    a1, a2, f1, f2 = unpack(best.x)
    to_min = 1.0 / bmin
    return {"bin_min": bmin, "var_y": vy, "acf": acf.tolist(), "a": [a1, a2], "phi_bin": [f1, f2],
            "int_v": [a1 * vy, a2 * vy], "int_phi": [f1 ** to_min, f2 ** to_min], "loss": float(best.fun)}


def local_rate(order_ms: np.ndarray, t0_ms: int, seconds: int, window_s: int) -> np.ndarray:
    """Centered moving order rate (orders per second) at each order time over window_s."""
    sec = ((order_ms - t0_ms) // 1000).astype(np.int64)
    sec = np.clip(sec, 0, seconds - 1)
    c = np.bincount(sec, minlength=seconds).astype(float)
    cs = np.r_[0.0, np.cumsum(c)]
    h = window_s // 2
    lo = np.clip(np.arange(seconds) - h, 0, seconds)
    hi = np.clip(np.arange(seconds) + h + 1, 0, seconds)
    rate = (cs[hi] - cs[lo]) / (hi - lo)
    return rate[sec]


def fit_mixture_normalized(order_ms: np.ndarray, t0_ms: int, seconds: int, bin_min: int) -> dict[str, Any]:
    """Expmix3 on durations x local rate (centered window of 5 intensity bins), rescaled to mean 1."""
    from ai_trader.sim.laws import fit_law

    d = np.diff(order_ms).astype(float) / 1000.0
    d = np.where(d <= 0, 0.0005, d)
    lr = local_rate(order_ms[:-1], t0_ms, seconds, max(300, 5 * 60 * bin_min))
    x = d * lr
    x = x[np.isfinite(x) & (x > 0)]
    law = fit_law("expmix3", x)
    w = np.array([law.params[f"w{j}"] for j in (1, 2, 3)])
    mean = 1.0 / np.array([law.params[f"rate{j}"] for j in (1, 2, 3)])
    o = np.argsort(mean)
    w, mean = w[o], mean[o]
    mu = float(w @ mean)
    return {"mix_w": (w / w.sum()).tolist(), "mix_mean": (mean / mu).tolist(), "cv_norm": float(np.std(x) / np.mean(x)),
            "cv_raw": float(np.std(d) / np.mean(d)), "n": int(len(x))}


def censored_spread_fit(spread_ticks: np.ndarray) -> dict[str, float]:
    """(mu, sd) of a latent normal log-spread (log ticks) censored at 0 (one tick), by least squares on the quantiles
    q = 0.5, 0.75, 0.9, 0.95, 0.99 of log(spread / tick); shares at the floor reported."""
    from scipy.optimize import minimize

    ls = np.log(np.maximum(spread_ticks[np.isfinite(spread_ticks) & (spread_ticks > 0)], 1e-9))
    qs = np.array([0.5, 0.75, 0.9, 0.95, 0.99])
    emp = np.maximum(np.quantile(ls, qs), 0.0)
    zq = stats.norm.ppf(qs)
    floor_share = float(np.mean(ls < math.log(1.5)))

    def loss(z):
        return float(np.sum((np.maximum(0.0, z[0] + math.exp(z[1]) * zq) - emp) ** 2))

    best = None
    for z0 in ([float(np.median(ls)), math.log(max(float(np.std(ls)), 0.05))], [-0.5, math.log(0.3)], [0.5, math.log(0.5)]):
        r = minimize(loss, np.array(z0), method="Nelder-Mead", options={"maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    return {"spr_mu": float(best.x[0]), "spr_sd": float(math.exp(best.x[1])), "share_lt_1p5_tick": floor_share,
            "median_ticks": float(np.exp(np.median(ls))), "p90_ticks": float(np.exp(np.quantile(ls, 0.9)))}
