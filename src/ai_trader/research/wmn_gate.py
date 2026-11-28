"""WMN time-decayed news-intensity gate for the weekend market maker (program WMM2; rule frozen in the H-WMN registry
header, parameters frozen in data/research/wmn/gate_params.json before any holdout gate statistic).

State per symbol s at time t (computable live from Friday to Sunday; only information known at or before t):

    lam_tok,s(t) = sum over precise token events e of s with known_ts <= t:  A_cat(e) g_cat(e)(t - bar_start(known_ts))
                 + sum over D episodes j of s with bar close <= t:          A_jump  g_jump(t - bar_start(j))
    lam_mkt(t)   = sum over macro releases m with release <= t:            A_macro g_macro(t - release)
                 + sum over BTC D episodes b with bar close <= t:          A_btc   g_btc(t - bar_start(b))
    M_s(t)       = 1 + w_s lam_mkt(t) + (1 - w_s) lam_tok,s(t),  w_s = beta_s^2 var(r_BTC) / var(r_s) (previous 7 days)
    lambda_news  = M_s - 1   (predicted excess variance as a multiple of normal; kernels truncated at 48 h)
    pull         = lambda_news >= PULL (3: predicted sigma >= 2x normal) or t in [release - 2 min, release)
    widen        = sqrt(1 + min(lambda_news, PULL))  in [1, 2], multiplies the quoted distance from mid

g is EXP exp(-tau / T) or POW (1 + tau / c)^-p (minutes), A and the spec per category are fitted on FIT shocks
(llm_in_sample, before 2026-07-01) by wmn_study.decay_* and stored in gate_params.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PULL = 3.0
MACRO_PRE_MIN = 2.0
TRUNC_MIN = 48 * 60.0
BAR_MS = 300_000
MIN_MS = 60_000


@dataclass
class Kernel:
    spec: str  # "exp" | "pow"
    A: float
    theta: list[float]  # exp: [T]; pow: [c, p] (minutes)

    def g(self, tau_min: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_min, float)
        if self.spec == "exp":
            v = np.exp(-tau / self.theta[0])
        else:
            v = (1.0 + tau / self.theta[0]) ** (-self.theta[1])
        return np.where((tau >= 0) & (tau <= TRUNC_MIN), v, 0.0)

    def value(self, tau_min: np.ndarray) -> np.ndarray:
        return self.A * self.g(tau_min)


@dataclass
class GateParams:
    kernels: dict[str, Kernel]
    category_map: dict[str, str] = field(default_factory=dict)  # event group -> kernel name
    pull: float = PULL
    macro_pre_min: float = MACRO_PRE_MIN
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> GateParams:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        ks = {k: Kernel(v["spec"], float(v["A"]), [float(x) for x in v["theta"]]) for k, v in d["kernels"].items()}
        return cls(ks, d.get("category_map", {}), float(d.get("pull", PULL)), float(d.get("macro_pre_min", MACRO_PRE_MIN)), d.get("meta", {}))

    def dump(self, path: str | Path) -> None:
        d = {"kernels": {k: {"spec": v.spec, "A": v.A, "theta": v.theta} for k, v in self.kernels.items()},
             "category_map": self.category_map, "pull": self.pull, "macro_pre_min": self.macro_pre_min, "meta": self.meta}
        Path(path).write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")

    def kernel_for(self, group: str) -> Kernel | None:
        name = self.category_map.get(group, group)
        return self.kernels.get(name) or self.kernels.get("news_pooled")


def accumulate(t_ms: np.ndarray, shocks: list[tuple[int, int, Kernel]]) -> np.ndarray:
    """sum_k A_k g_k(t - origin_k) over shocks with active_from_k <= t. shocks: (origin_ms, active_from_ms, kernel)."""
    t = np.asarray(t_ms, dtype=np.int64)
    out = np.zeros(t.size)
    for origin, active, k in shocks:
        i0 = np.searchsorted(t, active, side="left")
        i1 = np.searchsorted(t, origin + int(TRUNC_MIN * MIN_MS), side="right")
        if i1 <= i0:
            continue
        out[i0:i1] += k.value((t[i0:i1] - origin) / MIN_MS)
    return out


def token_shocks(params: GateParams, news: list[tuple[int, str]], episodes_bar_ms: list[int]) -> list[tuple[int, int, Kernel]]:
    """news: (known_ts, group); episodes: bar start of each D episode (known at bar close)."""
    sh = []
    for kt, grp in news:
        k = params.kernel_for(grp)
        if k is not None:
            sh.append((int(kt) // BAR_MS * BAR_MS, int(kt), k))
    kj = params.kernels.get("jump")
    if kj is not None:
        for b in episodes_bar_ms:
            sh.append((int(b), int(b) + BAR_MS, kj))
    return sh


def market_shocks(params: GateParams, macro_ms: list[int], btc_eps_bar_ms: list[int]) -> list[tuple[int, int, Kernel]]:
    sh = []
    km, kb = params.kernels.get("macro"), params.kernels.get("btc_jump")
    if km is not None:
        for m in macro_ms:
            sh.append((int(m), int(m), km))
    if kb is not None:
        for b in btc_eps_bar_ms:
            sh.append((int(b), int(b) + BAR_MS, kb))
    return sh


def macro_pre(t_ms: np.ndarray, macro_ms: list[int], pre_min: float = MACRO_PRE_MIN) -> np.ndarray:
    t = np.asarray(t_ms, dtype=np.int64)
    out = np.zeros(t.size, bool)
    for m in macro_ms:
        out |= (t >= m - int(pre_min * MIN_MS)) & (t < m)
    return out


def gate_values(t_ms: np.ndarray, lam_tok: np.ndarray, lam_mkt: np.ndarray, w: np.ndarray | float, pre: np.ndarray,
                pull_at: float = PULL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lambda_news, pull, widen from the components (w = market variance share of the symbol)."""
    w = np.clip(np.asarray(w, float), 0.0, 1.0)
    lam = w * lam_mkt + (1.0 - w) * lam_tok
    pull = (lam >= pull_at) | pre
    widen = np.sqrt(1.0 + np.minimum(lam, pull_at))
    return lam, pull, widen


def symbol_gate(params: GateParams, t_ms: np.ndarray, news: list[tuple[int, str]], episodes_bar_ms: list[int],
                macro_ms: list[int], btc_eps_bar_ms: list[int], w: np.ndarray | float):
    lt = accumulate(t_ms, token_shocks(params, news, episodes_bar_ms))
    lm = accumulate(t_ms, market_shocks(params, macro_ms, btc_eps_bar_ms))
    pre = macro_pre(t_ms, macro_ms, params.macro_pre_min)
    return gate_values(t_ms, lt, lm, w, pre, params.pull) + (lt, lm)


def market_gate(params: GateParams, t_ms: np.ndarray, macro_ms: list[int], btc_eps_bar_ms: list[int]):
    lm = accumulate(t_ms, market_shocks(params, macro_ms, btc_eps_bar_ms))
    pre = macro_pre(t_ms, macro_ms, params.macro_pre_min)
    return gate_values(t_ms, np.zeros_like(lm), lm, 1.0, pre, params.pull)


def describe() -> str:
    return __doc__ or ""


__all__ = ["GateParams", "Kernel", "PULL", "accumulate", "gate_values", "macro_pre", "market_gate", "symbol_gate",
           "token_shocks", "market_shocks"]
