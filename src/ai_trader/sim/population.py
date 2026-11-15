"""Population model of the event-level simulator (H-MS7): every parameter of `flow_sim_v1.V1Params` as a function of
trading activity W, asset class and tick size, so a calibrated generator can be built for an instrument we never
recorded (`Population.params_for`).

Generated data is for deterministic strategies and the execution engine only, never for the LLM layer (CLAUDE.md
principle 1).

Definitions (fixed in research.population_simulator_study.METHOD before the tests):
  W        trading activity = sigma_day x P V_day (USD per day): sigma_day = std of 1-min log returns of the trade mid
           proxy ((last buy + last sell) / 2) x sqrt(1440) as a fraction, P V_day = notional traded per day (Kyle and
           Obizhaeva 2016, "market microstructure invariance")
  classes  major (BTC ETH SOL XRP DOGE) | alt (other crypto) | tradfi (equity, index, commodity perps)
  models   OLS of each scaled parameter on [1, log W, alt, tradfi] across symbols ("lin"); leave-one-symbol-out R^2
           (PRESS) for the cross-sectional test; parameters with no stable relation are carried as class medians
           ("med"); the tick regime is a logistic on [1, log tick_bps, log W]; the free-spread level is an OLS of log
           spread (bp) on [1, log W, log tick_bps] over the free-spread symbols

Invariance predictions (Kyle-Obizhaeva): bets per day ~ W^(2/3), bet notional ~ W^(1/3), spread and impact cost in
return units ~ 1/L = (sigma^2 / P V)^(1/3) = sigma W^(-1/3), Kyle lambda (return per dollar) ~ sigma^2 W^(-2/3).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.sim.flow_sim_v1 import PMF_KEYS, V1Params

CLASSES = ("major", "alt", "tradfi")
KO_SLOPES = {"log_rate": 2 / 3, "log_v_ref": 1 / 3, "log_spread_over_sigma": -1 / 3, "log_lambda_over_sigma2": -2 / 3,
             "log_spread_vs_illiq": 1.0}


def design(log_w: np.ndarray, cls: list[str], extra: list[np.ndarray] | None = None) -> np.ndarray:
    cols = [np.ones(len(log_w)), np.asarray(log_w, dtype=float), np.array([c == "alt" for c in cls], float),
            np.array([c == "tradfi" for c in cls], float)]
    for e in extra or []:
        cols.append(np.asarray(e, dtype=float))
    return np.column_stack(cols)


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def hc1_se(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    V = A @ (X.T * e ** 2) @ X @ A * n / max(n - k, 1)
    return np.sqrt(np.clip(np.diag(V), 0, None))


def loo_r2(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None) -> float:
    """1 - PRESS / TSS, leaving out one group (symbol) at a time (all rows of the group); TSS around the full mean."""
    y = np.asarray(y, dtype=float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    pred = np.empty(len(y))
    for gval in np.unique(groups):
        m = groups == gval
        if (~m).sum() <= X.shape[1]:
            return float("nan")
        b, *_ = np.linalg.lstsq(X[~m], y[~m], rcond=None)
        pred[m] = X[m] @ b
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / tss if tss > 0 else float("nan")


def loo_r2_boot(y: np.ndarray, X: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap over symbols of the LOO R^2 (rows resampled with replacement; the held-out unit is a symbol, all of
    its copies are dropped from the fit)."""
    n = len(y)
    out = np.full(B, np.nan)
    for i in range(B):
        ix = rng.integers(0, n, n)
        if np.linalg.matrix_rank(X[ix]) < X.shape[1]:
            continue
        out[i] = loo_r2(y[ix], X[ix], groups=ix)
    return out


def logistic(y: np.ndarray, X: np.ndarray, ridge: float = 1e-3, iters: int = 100) -> np.ndarray:
    """IRLS logistic regression with a small ridge (separable data stay finite)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def universality(F: np.ndarray, T: np.ndarray, B: int, rng: np.random.Generator) -> dict[str, float]:
    """D = Var((F+T)/2) / (Var(F-T)/4): between-symbol dispersion of a parameter relative to its estimation noise
    (FIT-vs-TEST difference). D = 1 if all dispersion is noise; D > 1 iff the two halves covary across symbols.
    Bootstrap over symbols: CI and p = P*(D <= 1) (small p = dispersion significantly above noise)."""
    ok = np.isfinite(F) & np.isfinite(T)
    F, T = F[ok], T[ok]
    n = len(F)

    def D(f, t):
        noise = np.var(f - t, ddof=1) / 4
        return float(np.var((f + t) / 2, ddof=1) / noise) if noise > 0 else float("nan")

    d0 = D(F, T)
    bs = np.array([D(F[ix], T[ix]) for ix in (rng.integers(0, n, n) for _ in range(B))])
    return {"n": int(n), "D": d0, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
            "p_disp": float(np.nanmean(bs <= 1.0)), "between_sd": float(np.std((F + T) / 2, ddof=1)),
            "noise_sd": float(np.std(F - T, ddof=1) / 2), "spearman_fit_test": float(_spearman(F, T)), "mean": float(np.mean((F + T) / 2))}


def _spearman(a, b) -> float:
    from scipy import stats

    return float(stats.spearmanr(a, b).statistic) if len(a) > 3 else float("nan")


# ------------------------------------------------------------------------------------------------- the population

LIN_PARAMS = ("log_rate", "log_v_ref", "log_sigma_day", "log_depth25", "log_spread_bps", "log_lambda", "sign_gamma", "sign_acf1",
              "zeta_order", "size_mu", "log_k_imp", "impact_corr", "flow_acf_3s", "absr_acf_3s", "ppo_mean")
MED_PARAMS = ("size_sigma", "log_xmin_over_vref", "size_p_tail", "log_cap_over_vref", "mix_w", "mix_mean", "int_v", "int_phi",
              "sign_c", "sign_short", "ppo_pmf", "ppo_tail_zeta", "rho_kv", "g_inf", "tau_s", "noise_beta", "size_xi_share",
              "size_g_share", "log_sigma0_over_sigma1m", "jump_rate", "jump_z", "spr_phi", "vr_3s_1m", "vr5", "size_ns_acf10",
              "exc_rate", "rob_over_sigma1m", "impact_delta")


@dataclass
class Population:
    lin: dict[str, dict] = field(default_factory=dict)        # name -> {coef, names, n, r2, resid_sd}
    regime: dict = field(default_factory=dict)                # logistic coef on [1, log tick_bps, log W]
    free_spread: dict = field(default_factory=dict)           # OLS coef on [1, log W, log tick_bps] (free-spread symbols)
    med: dict[str, dict] = field(default_factory=dict)        # name -> {class: value, "all": value}
    tick_bound_spread: dict = field(default_factory=dict)     # median spr_mu, spr_sd of tick-bound symbols
    free_spread_sd: float = 0.3
    symbols: list = field(default_factory=list)
    excluded: str | None = None

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, rows: list[dict], exclude: str | None = None) -> Population:
        rows = [r for r in rows if r["symbol"] != exclude]
        pop = cls(symbols=[r["symbol"] for r in rows], excluded=exclude)
        lw = np.array([math.log(r["W"]) for r in rows])
        cl = [r["class"] for r in rows]
        X = design(lw, cl)
        for name in LIN_PARAMS:
            y = np.array([r.get(name, np.nan) for r in rows], dtype=float)
            ok = np.isfinite(y)
            if ok.sum() < 8:
                continue
            b, e = ols(y[ok], X[ok])
            pop.lin[name] = {"coef": b.tolist(), "names": ["const", "log_W", "alt", "tradfi"], "n": int(ok.sum()),
                             "resid_sd": float(np.std(e, ddof=X.shape[1])), "r2": _r2(y[ok], e)}
        tb = np.array([float(r["tick_bound"]) for r in rows])
        lt = np.array([math.log(r["tick_bps"]) for r in rows])
        pop.regime = {"coef": logistic(tb, np.column_stack([np.ones(len(lw)), lt, lw])).tolist(), "names": ["const", "log_tick_bps", "log_W"]}
        fr = tb == 0
        ys = np.array([r["log_spread_bps"] for r in rows])
        if fr.sum() >= 4:
            Xf = np.column_stack([np.ones(int(fr.sum())), lw[fr], lt[fr]])
            b, e = ols(ys[fr], Xf)
            pop.free_spread = {"coef": b.tolist(), "names": ["const", "log_W", "log_tick_bps"], "n": int(fr.sum()),
                               "resid_sd": float(np.std(e, ddof=3))}
            pop.free_spread_sd = float(np.median([r["spr_sd"] for r, f in zip(rows, fr, strict=True) if f]))
        tbr = [r for r, f in zip(rows, fr, strict=True) if not f]
        pop.tick_bound_spread = {"spr_mu": float(np.median([r["spr_mu"] for r in tbr])) if tbr else 0.0,
                                 "spr_sd": float(np.median([r["spr_sd"] for r in tbr])) if tbr else 0.02}
        for name in MED_PARAMS:
            pop.med[name] = {}
            for c in (*CLASSES, "all"):
                sel = [r[name] for r in rows if (c == "all" or r["class"] == c) and r.get(name) is not None]
                if not sel:
                    continue
                pop.med[name][c] = [float(x) for v in sel for x in v] if name == "jump_z" else _median_of(sel)
        return pop

    # ------------------------------------------------------------------ predict
    def predict(self, name: str, W: float, asset_class: str) -> float:
        m = self.lin[name]
        x = np.array([1.0, math.log(W), float(asset_class == "alt"), float(asset_class == "tradfi")])
        return float(x @ np.asarray(m["coef"]))

    def m(self, name: str, asset_class: str):
        d = self.med.get(name, {})
        return copy.deepcopy(d.get(asset_class, d.get("all")))

    def p_tick_bound(self, W: float, tick_bps: float) -> float:
        b = np.asarray(self.regime["coef"])
        return float(1 / (1 + math.exp(-float(b @ [1.0, math.log(tick_bps), math.log(W)]))))

    def spread_bps(self, W: float, asset_class: str, tick_bps: float) -> dict[str, float]:
        """Regime and median quoted spread (bp): tick-bound -> about one tick; free -> the free-spread OLS, at least 1.5 ticks."""
        p_tb = self.p_tick_bound(W, tick_bps)
        if p_tb >= 0.5 or not self.free_spread:
            mu, sd = self.tick_bound_spread["spr_mu"], self.tick_bound_spread["spr_sd"]
            return {"regime": "tick_bound", "p_tick_bound": p_tb, "spread_bps": tick_bps * max(1.0, math.exp(mu)), "spr_mu": mu, "spr_sd": sd}
        b = np.asarray(self.free_spread["coef"])
        s = float(math.exp(b @ [1.0, math.log(W), math.log(tick_bps)]))
        s = max(s, 1.5 * tick_bps)
        return {"regime": "free", "p_tick_bound": p_tb, "spread_bps": s, "spr_mu": math.log(s / tick_bps), "spr_sd": self.free_spread_sd}

    def params_for(self, W: float, asset_class: str, tick_rel: float, *, sigma_day: float | None = None, price: float = 100.0,
                   symbol: str = "SIM", calibrate: bool = False, seconds: int = 4 * 3600, seed: int = 11) -> V1Params:
        """Full V1Params for an instrument with activity W (USD per day), class and tick size tick_rel = tick / price.
        sigma_day (daily return sd as a fraction) defaults to the population prediction. With calibrate=True the
        impact / noise / coupling parameters are re-fitted (indirect inference) to the predicted targets."""
        c = asset_class if asset_class in CLASSES else "alt"
        tick_bps = tick_rel * 1e4
        sig_day = sigma_day if sigma_day is not None else math.exp(self.predict("log_sigma_day", W, c))
        sigma_1m = sig_day * 1e4 / math.sqrt(1440)
        v_ref = math.exp(self.predict("log_v_ref", W, c))
        sp = self.spread_bps(W, c, tick_bps)
        depth = math.exp(self.predict("log_depth25", W, c))
        mu = self.predict("size_mu", W, c)
        mix_w, mix_mean = np.asarray(self.m("mix_w", c), float), np.asarray(self.m("mix_mean", c), float)
        mix_w = mix_w / mix_w.sum()
        mix_mean = mix_mean / float(mix_w @ mix_mean)
        pmf = self.m("ppo_pmf", c)
        tot = sum(pmf.values())
        pmf = {k: float(v) / tot for k, v in pmf.items()}
        zeta = float(np.clip(self.predict("zeta_order", W, c), 0.6, 3.0))
        p = V1Params(symbol=symbol, asset_class=c, price0=price, tick_bps=tick_bps,
                     rate=math.exp(self.predict("log_rate", W, c)), mix_w=mix_w.tolist(), mix_mean=mix_mean.tolist(),
                     int_v=list(self.m("int_v", c)), int_phi=list(self.m("int_phi", c)),
                     size_mu=mu, size_sigma=float(self.m("size_sigma", c)), size_xmin=math.exp(math.log(v_ref) + float(self.m("log_xmin_over_vref", c))),
                     size_alpha=zeta + 1.0, size_p_tail=float(self.m("size_p_tail", c)),
                     size_cap=math.exp(math.log(v_ref) + float(self.m("log_cap_over_vref", c))),
                     size_xi_share=float(self.m("size_xi_share", c)), size_g_share=float(self.m("size_g_share", c)),
                     sign_c=float(self.m("sign_c", c)), sign_gamma=float(np.clip(self.predict("sign_gamma", W, c), 0.1, 1.5)),
                     sign_short=self.m("sign_short", c), p_buy=0.5, ppo_pmf=pmf, ppo_tail_zeta=float(self.m("ppo_tail_zeta", c)),
                     rho_kv=float(self.m("rho_kv", c)), level_usd=max(depth / 2.0 * min(1.0, tick_bps / 25.0), 10.0),
                     impact_delta=float(self.m("impact_delta", c)), k_imp=math.exp(self.predict("log_k_imp", W, c)), v_ref=v_ref,
                     g_inf=float(self.m("g_inf", c)), tau_s=float(self.m("tau_s", c)),
                     sigma0=sigma_1m * math.exp(float(self.m("log_sigma0_over_sigma1m", c))), noise_beta=float(self.m("noise_beta", c)),
                     jump_rate=float(self.m("jump_rate", c)), jump_pool=[z * sigma_1m * float(self.m("rob_over_sigma1m", c)) for z in self.m("jump_z", c) or []],
                     spr_mu=sp["spr_mu"], spr_sd=sp["spr_sd"], spr_phi=float(self.m("spr_phi", c)), depth25=depth,
                     info={"population": {"W": W, "class": c, "tick_rel": tick_rel, "sigma_day": sig_day, "spread": sp,
                                          "excluded": self.excluded}})
        if calibrate:
            from ai_trader.sim.flow_sim_v1 import calibrate as _cal

            tgt = {"sigma_1m": sigma_1m, "robust_sd_1m": sigma_1m * float(self.m("rob_over_sigma1m", c)),
                   "impact": float(np.clip(self.predict("impact_corr", W, c), 0.02, 0.9)), "kurtosis": None, "absr_acf": None,
                   "absr_acf_3s": self.predict("absr_acf_3s", W, c), "vr_3s_1m": float(self.m("vr_3s_1m", c)), "vr5": float(self.m("vr5", c)),
                   "exc_rate": float(self.m("exc_rate", c)), "flow_acf_3s": self.predict("flow_acf_3s", W, c),
                   "size_ns_acf10": float(self.m("size_ns_acf10", c))}
            p, fin = _cal(p, tgt, jump_z=list(self.m("jump_z", c) or []), seconds=seconds, seed=seed)
            p.info["population"]["targets"] = tgt
        return p

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return {"lin": self.lin, "regime": self.regime, "free_spread": self.free_spread, "med": self.med,
                "tick_bound_spread": self.tick_bound_spread, "free_spread_sd": self.free_spread_sd, "symbols": self.symbols,
                "excluded": self.excluded, "ko_slopes": KO_SLOPES}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Population:
        return cls(**{k: d[k] for k in ("lin", "regime", "free_spread", "med", "tick_bound_spread", "free_spread_sd", "symbols", "excluded") if k in d})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), default=float, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Population:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _median_of(vals: list):
    """Median of scalars, element-wise median of equal-length lists, key-wise median of dicts."""
    v0 = vals[0]
    if isinstance(v0, dict):
        keys = [k for k in v0 if all(k in v for v in vals)]
        return {k: float(np.median([float(v[k]) for v in vals])) for k in keys}
    if isinstance(v0, (list, tuple)):
        L = min(len(v) for v in vals)
        return np.median(np.array([list(map(float, v[:L])) for v in vals]), axis=0).tolist()
    return float(np.median([float(v) for v in vals]))


def rows_from_symbols(sym_data: list[dict], params_json: dict) -> list[dict]:
    """Population rows from the per-symbol study outputs (research.population_simulator_study stage symbol) and the C1
    parameter file: every value below is measured on the FIT half (spread on pre-TEST flow_3s days)."""
    rows = []
    for d in sym_data:
        s = d["symbol"]
        fm = d["meas"]["fit"]
        p = d["params"]
        h = params_json["symbols"][s]["halves"]["fit"]
        ht = params_json["symbols"][s]["halves"]["test"]
        sz = h["size"]["order"]["size_law"]
        acf = h["signs"]["order"]["acf"]
        sp = d["spread"]
        sig1m = fm["sigma_1m"]
        cal = p.get("info", {}).get("calibration", {})
        rows.append({
            "symbol": s, "class": d["class"], "W": fm["W"], "sigma_day": fm["sigma_day"] * 1e-4, "pv_day": fm["pv_day"], "price": fm["price"],
            "tick_bps": sp["tick_bps"], "tick_bound": bool(sp["tick_bound"]), "spr_mu": sp["spr_mu"], "spr_sd": sp["spr_sd"],
            "log_rate": math.log(fm["orders_per_s"]), "log_v_ref": math.log(fm["order_notional_med"]), "log_sigma_day": math.log(fm["sigma_day"] * 1e-4),
            "log_depth25": math.log(max(sp["depth25_med"], 1.0)), "log_spread_bps": math.log(sp["spread_med_bps"]),
            "lambda": fm["lambda_mid"], "log_lambda": math.log(fm["lambda_mid"]) if fm["lambda_mid"] and fm["lambda_mid"] > 0 else None,
            "lambda_trade": fm["lambda_trade"],
            "sign_gamma": h["signs"]["order"]["acf_power"]["gamma"], "sign_gamma_test": ht["signs"]["order"]["acf_power"]["gamma"],
            "sign_acf1": acf["1"], "sign_acf1_test": ht["signs"]["order"]["acf"]["1"],
            "zeta_order": h["size"]["order"]["powerlaw"]["zeta"], "zeta_order_test": ht["size"]["order"]["powerlaw"]["zeta"],
            "zeta_print": h["size"]["print"]["powerlaw"]["zeta"], "zeta_print_test": ht["size"]["print"]["powerlaw"]["zeta"],
            "size_mu": sz["mu"], "size_sigma": sz["sigma"], "log_xmin_over_vref": math.log(sz["xmin"] / fm["order_notional_med"]),
            "size_p_tail": sz["p_tail"], "log_cap_over_vref": math.log(p["size_cap"] / fm["order_notional_med"]),
            "mix_w": p["mix_w"], "mix_mean": p["mix_mean"], "mix_w1": p["mix_w"][0], "mix_w2": p["mix_w"][1],
            "log_mix_m1": math.log(max(p["mix_mean"][0], 1e-6)), "log_mix_m2": math.log(max(p["mix_mean"][1], 1e-6)), "log_mix_m3": math.log(p["mix_mean"][2]),
            "int_v": p["int_v"], "int_phi": p["int_phi"], "sign_c": p["sign_c"], "sign_short": p["sign_short"],
            "ppo_pmf": p["ppo_pmf"], "ppo_tail_zeta": p["ppo_tail_zeta"], "rho_kv": p["rho_kv"],
            "ppo_mean": h["rates"]["prints_per_order_mean"],
            "log_k_imp": math.log(p["k_imp"]) if p["k_imp"] > 0 else None, "g_inf": p["g_inf"], "tau_s": p["tau_s"], "noise_beta": p["noise_beta"],
            "size_xi_share": p["size_xi_share"], "size_g_share": p["size_g_share"], "impact_delta": p["impact_delta"],
            "log_sigma0_over_sigma1m": math.log(max(p["sigma0"], 1e-6) / sig1m) if sig1m and sig1m > 0 else None,
            "jump_rate": p["jump_rate"], "jump_z": [x / max(fm["robust_sd_1m"], 1e-9) for x in fm.get("exc_values", [])] if fm.get("robust_sd_1m") else [],
            "spr_phi": p["spr_phi"],
            "impact_corr": d["pre_test"]["impact"], "vr_3s_1m": d["pre_test"]["vr_3s_1m"], "vr5": d["pre_test"]["vr5"],
            "flow_acf_3s": fm["flow_acf_3s"], "absr_acf_3s": fm["absr_acf_3s"], "size_ns_acf10": fm["size_ns_acf10"],
            "exc_rate": fm["exc_rate"], "rob_over_sigma1m": fm["robust_sd_1m"] / sig1m if sig1m and sig1m > 0 else 1.0,
            "illiq": fm["illiq"], "calib_final": cal.get("final"),
        })
    return rows


__all__ = ["CLASSES", "KO_SLOPES", "Population", "design", "loo_r2", "loo_r2_boot", "logistic", "ols", "rows_from_symbols",
           "universality", "PMF_KEYS"]


def _r2(y: np.ndarray, e: np.ndarray) -> float:
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(e @ e) / tss if tss > 0 else float("nan")
