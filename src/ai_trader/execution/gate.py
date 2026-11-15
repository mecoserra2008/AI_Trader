"""Bias gate: the tested biases (research/rules.yaml via research.playbook) applied to every strategy event, plus the
vol-based placement and expectations each OrderIntent carries (agent K, 2026-11-13; contract docs/EXECUTION_PIPELINE.md).

The gate never sizes and never sets a direction: the side comes from the strategy signal only (R-PX-6, R-ML-5, R-DS-6;
no tilts from calendar or regime cells), exits come from the strategy signal only (R-EXIT-1). Rules are read from
research/rules.yaml at run time, so a retired rule stops gating and a rule's WHEN cells are parsed, not hard-coded.

Machine-checkable rules (entries only; exits always pass, a gate must never trap a position):
  R-AVOID-1   strategies whose family failed its test (funding contrarian, OI-price divergence, pivot-geometry trend
              = geo_struct / geo_pivot_bo / geo_flag_bo, matched by the phrases in the rule text)        -> block
  R-MAP-3..7  regime-cell avoid lists: the strategy is named in the rule text and the START-OF-DAY v0 cell of the symbol
              (research.regimes.label on 1h, the row closing at 00:00 UTC) is in the rule's WHEN cells (playbook.conditions);
              R-MAP-7 also needs vol_state normal at the decision bar and the shock_regime reversal leg   -> block (A/B
              active) or warn (C / provisional)
  R-PX-2      never fade a >= 2 sigma hourly deviation from trend: dev = log(c / EMA48) / rolling std (240) of that
              deviation on 1h (the H-T1 / zscore_reversion definition); an entry against the sign of dev with |dev| >= 2
              is a fade. shock_regime's reversal leg is included until H-S3 has a clean_holdout 'supports' result
              (R-PX-1 precedence note)                                                                     -> block
  R-PX-9      no directional trades from weekend price: (a) no entries while the next bar is inside the TradFi weekend
              window (research.calendar tradfi_weekend), except shock_regime (agent.session playbook S4);
              (b) the weekend-move strategies weekend_shock_follow, weekend_drift_fade, cme_gap_fill     -> block
  provisional (grade C) avoid rules only warn; a rule whose scope assets exclude the symbol's class is not applied.

Placement and expectations (every entry):
  sigma_risk  robust_bc risk sigma sigma_risk_next (no weekend shrink: R-VOL-2, R-VOL-3) per bar of the strategy's
              frequency = sigma.per_bar. The contract's stop unit (L's check) is |ln(entry/stop)| / sigma.per_bar
  stop        STOP_K = 1.5 daily risk sigmas (the H-JD4 bracket behind R-EXEC-2; agent.session's trend sleeve) = 1.5 sqrt(bars
              per day) per-bar sigmas, clamped to [min_stop_distance_sigma, max_stop_distance_sigma] per-bar sigmas and to
              max_stop_distance_pct (config/risk.yaml). With max 4 that is 4 sigma_1h (~0.8 daily sigma) for the 1h strategies
              and 3.67 sigma_4h (1.5 daily) for the 4h ones: the open placement conflict of research/BELIEFS.md (cap vs wide
              stops) is flagged per intent (stop_clamped_max_sigma), not resolved here. Vol-only placement (ARCHITECTURE 3.11
              'no structure' fallback, flagged): structural levels are not built yet. Resting exchange stop (R-EXEC-1)
  target      a reference target at TP_R = 3 R (H-JD4 geometry) for p_tp / net_er only, in context.reference_target. Every
              registered strategy exits on its own signal (R-EXIT-1), so `targets` is [] by default and net_er_r is sent
              (the engine then checks the intent's net_er_r); resting_target=True puts it in targets[0] as a resting TP
  expectation central daily variance = per-asset log-HAR (features.vol_panel 'loghar', expanding window, monthly refit;
              R-VOL-20) x weekday factor of the forecast day (R-ML-1, crypto only; factors parsed from the rule text:
              Sat 0.6, Sun 0.85, Mon 1.3, Tue-Fri so that the week averages 1). TradFi: factor 1 (R-VOL-9: no blanket
              Monday multiplier). The bracket horizon uses the direct 5-day log-HAR level (R-VOL-21: no flat scaling)
  p_tp        crypto: jump_diffusion.simulate_bracket (R-EXEC-2, H-JD4): Kou jumps + Hawkes fitted monthly on the trailing
              2 years of 1h bars strictly before the month (as scripts/jump_study.py), current Hawkes excitation from
              bars <= t, diffusion sigma = central hourly sigma, 2000 paths, seed from the event, run for SIM_BARS = 1344
              1h bars (8 weeks: the placement has no time barrier, ARCHITECTURE 3.11; after 8 weeks < 1% of paths are
              open and they get the driftless odds). A 168-bar cap (H-JD4's) would leave ~1/3 of paths open, and
              conditioning on resolution would bias p_tp toward the nearer stop (0.12 vs 0.23 for 1.5 sigma / 3R). TradFi, or
              crypto without 2000 bars of history: agent.session.p_first (driftless Gaussian first passage; cheaper,
              not jump-aware), flagged in context.p_source
  edge        p_tp = null + a documented prior edge only for the sleeves agent.session lists (PRIOR_EDGE: shock_regime
              +0.05, the 4h trend sleeve +0.03 -> zoo_ensemble). Every other strategy gets the driftless null, so its
              net_er_r is <= 0 and the risk engine's min_net_expected_r rejects it: no registered strategy has a
              validated positive net edge (R-MAP-1, R-PX-3)
  net_er_r    agent.session.net_er (the at-risk formula: round-trip costs added to both legs)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research.playbook import (
    ACTIVE,
    asset_set,
    conditions,
    rules_hash,
    select,
)
from ai_trader.research.rules import load_rules

from .signals import SignalEvent, SymbolFrames

VERSION = "0.1.0"
STOP_K = 1.5            # daily risk sigmas (H-JD4 bracket)
TP_R = 3.0              # target in R (H-JD4 bracket)
TICK_MARGIN = 0.01      # keep the stop 1% inside the placement sigma bounds (tick rounding in the engine)
SIM_BARS = 1344         # 1h bars: 8 weeks, effectively no time barrier (ARCHITECTURE 3.11)
N_PATHS = 2000
TRAIN_BARS = 2 * 365 * 24
MIN_FIT_BARS = 2000
DEV_SPAN, DEV_STD = 48, 240  # R-PX-2 / H-T1 deviation definition (zscore_reversion's)
FADE_SIGMA = 2.0
WEEKEND_STRATEGIES = ("weekend_shock_follow", "weekend_drift_fade", "cme_gap_fill")
WEEKEND_EXEMPT = ("shock_regime",)      # agent.session playbook S4
AVOID1_PHRASES = {r"funding contrarian": ("funding_contrarian",), r"oi-price divergence": ("oi_price_divergence",),
                  r"pivot-geometry trend": ("geo_struct", "geo_pivot_bo", "geo_flag_bo")}
EDGE_SLEEVE = {"shock_regime": "shock_regime", "zoo_ensemble": "trend_4h"}  # strategy -> agent.session.PRIOR_EDGE key
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WD_DEFAULT = {"Sat": 0.6, "Sun": 0.85, "Mon": 1.3}
# rules the gate checks or applies mechanically (excluded from the LLM context list)
MACHINE_RULES = ("R-AVOID-1", "R-MAP-3", "R-MAP-4", "R-MAP-5", "R-MAP-6", "R-MAP-7", "R-PX-2", "R-PX-9", "R-PX-6", "R-ML-5", "R-DS-6",
                 "R-EXIT-1", "R-VOL-2", "R-VOL-3", "R-VOL-20", "R-VOL-21", "R-ML-1", "R-EXEC-1", "R-EXEC-2", "R-PD-1", "R-PD-2", "R-MAP-1", "R-PX-3")


def exec_dir() -> Path:
    return load().root / "data" / "exec"


# ====================================================================== rule access

class RuleBook:
    """rules.yaml as data: status, grade, scope and parsed WHEN conditions per rule id."""

    def __init__(self, rules: list[dict] | None = None):
        self.rules = load_rules() if rules is None else rules
        self.by_id = {r["id"]: r for r in self.rules}
        self.hash = rules_hash(self.rules)
        from ai_trader.strategies import REGISTRY

        names = [*REGISTRY, "zoo_ensemble"]
        self.cell_rules: list[dict] = []
        for r in self.rules:
            if not r["id"].startswith("R-MAP-") or r["kind"] != "avoid":
                continue
            txt = f"{r['rule']} {r['when']} {r['then']}"
            named = tuple(n for n in names if re.search(rf"\b{n}\b", txt))
            cells = conditions(r).get("cell", set())
            if named and cells:
                self.cell_rules.append({"id": r["id"], "strategies": named, "cells": cells,
                                        "vol_state": conditions(r).get("vol_state") if "vol_state" in str(r["when"]) else None,
                                        "reversal_leg": "reversal leg" in str(r["rule"]).lower()})
        r1 = self.by_id.get("R-AVOID-1")
        self.avoid1: tuple[str, ...] = ()
        if r1:
            t = str(r1["rule"]).lower()
            self.avoid1 = tuple(s for p, ss in AVOID1_PHRASES.items() if re.search(p, t) for s in ss)
        self.weekday_factor = self._weekday_factors()

    def active(self, rid: str) -> bool:
        r = self.by_id.get(rid)
        return bool(r) and r["status"] in ACTIVE

    def applies(self, rid: str, asset_class: str) -> bool:
        r = self.by_id.get(rid)
        return self.active(rid) and asset_class in asset_set(r["scope"].get("assets"))

    def severity(self, rid: str) -> str:
        r = self.by_id[rid]
        return "block" if r["status"] == "active" and r["confidence"] in ("A", "B") else "warn"

    def _weekday_factors(self) -> dict[str, float]:
        """R-ML-1 weekday variance factors (Sat / Sun / Mon parsed from the rule's THEN; Tue-Fri fill the week to 7)."""
        wf = dict(WD_DEFAULT)
        r = self.by_id.get("R-ML-1")
        if r:
            t = str(r["then"])
            for day, key in (("Saturday", "Sat"), ("Sunday", "Sun"), ("Monday", "Mon")):
                m = re.search(rf"{day} about ([0-9.]+)x", t)
                if m:
                    wf[key] = float(m.group(1))
        rest = (7.0 - wf["Sat"] - wf["Sun"] - wf["Mon"]) / 4.0
        return {d: wf.get(d, rest) for d in WEEKDAYS}


def holdout_supports(hid: str, as_of: datetime) -> bool:
    """True if hypothesis `hid` has a clean_holdout 'supports' verdict recorded at or before as_of (read-only)."""
    p = load().root / "research" / "hypotheses_results.jsonl"
    if not p.is_file():
        return False
    ok = False
    for line in p.read_text(encoding="utf-8").splitlines():
        if f'"{hid}"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("id") == hid and d.get("segment") == "clean_holdout" and datetime.fromisoformat(d["ts"]) <= as_of:
            ok = str(d.get("verdict", "")).startswith("support")
    return ok


# ====================================================================== per-symbol state series

def _daily_loghar(f1: pl.DataFrame) -> dict[date, tuple[float, float]]:
    """Forecast day D -> (F1, F5): per-asset log-HAR forecasts of next-day variance and of the 5-day mean daily variance,
    made at the end of day D-1 from complete days <= D-1 (vol_panel.per_asset_forecast; expanding, refit every 30 days)."""
    from ai_trader.features import vol_panel as vp
    from ai_trader.features.realized import daily_measures

    d = daily_measures(f1.select("event_ts", "open", "high", "low", "close"), "1h").select("date", "rv")
    if d.height < 40:
        return {}
    cal = pl.DataFrame({"date": pl.date_range(d["date"].min(), d["date"].max(), "1d", eager=True)})
    d = cal.join(d, on="date", how="left").sort("date")
    rv = d["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    X, _ = vp.design("loghar", {"rv": rv})
    f1d = vp.per_asset_forecast(X, vp.target(rv, 1, True), 1, log=True, refit_every=30, min_days=180, first=0)
    f5d = vp.per_asset_forecast(X, vp.target(rv, 5, True), 5, log=True, refit_every=30, min_days=180, first=0)
    dates = d["date"].to_list()
    return {dates[k] + timedelta(days=1): (float(f1d[k]), float(f5d[k])) for k in range(len(dates))}


@dataclass
class SymbolState:
    """Causal per-bar series on the symbol's 1h frame (every value at row t uses rows <= t)."""

    symbol: str
    asset_class: str
    ts_known: np.ndarray        # known_ts of 1h rows (datetime64[us])
    close_ts: list
    close: np.ndarray
    cell: list
    vol_state: list
    dir_state: list
    sod_cell: dict
    dev: np.ndarray
    z: np.ndarray
    ret: np.ndarray
    sig_risk: np.ndarray
    sig_rb: np.ndarray
    next_weekend: np.ndarray
    loghar: dict
    f4_known: np.ndarray | None = None
    f4_sig_risk: np.ndarray | None = None
    jump_cache: dict = field(default_factory=dict)
    r_1h: np.ndarray | None = None
    event_ts: list | None = None

    @classmethod
    def build(cls, sf: SymbolFrames) -> SymbolState:
        from ai_trader.research.calendar import add_calendar
        from ai_trader.research.regimes import label
        from ai_trader.strategies.trend import ema, rstd, safe_div

        f = label(sf.f1, 1.0)
        c = f["close"].to_numpy().astype(float)
        raw = np.log(c) - np.log(ema(c, DEV_SPAN))
        sd = rstd(raw, DEV_STD, DEV_STD // 2)
        dev = safe_div(raw, np.where(sd > 0, sd, np.nan))
        sod = (f.filter((pl.col("close_ts").dt.hour() == 0) & (pl.col("close_ts").dt.minute() == 0))
               .select(pl.col("close_ts").dt.date().alias("d"), "cell"))
        nxt = add_calendar(f.select(pl.col("close_ts").alias("event_ts")))["tradfi_weekend"].to_numpy().astype(bool)
        st = cls(sf.symbol, sf.asset_class, f["known_ts"].to_numpy(), f["close_ts"].to_list(), c, f["cell"].to_list(), f["vol_state"].to_list(),
                 f["dir_state"].to_list(), dict(zip(sod["d"].to_list(), sod["cell"].to_list(), strict=True)), dev,
                 f["z"].to_numpy().astype(float), f["ret"].to_numpy().astype(float), f["sigma_risk_next"].to_numpy().astype(float),
                 f["sigma_rb_next"].to_numpy().astype(float), nxt, _daily_loghar(sf.f1))
        st.r_1h = np.nan_to_num(st.ret, nan=0.0)
        st.event_ts = f["event_ts"].to_list()
        if sf.f4 is not None and sf.f4.height:
            st.f4_known = sf.f4["known_ts"].to_numpy()
            st.f4_sig_risk = sf.f4["sigma_risk_next"].to_numpy().astype(float)
        return st

    def row_at(self, ts: datetime) -> int:
        """Last 1h row with known_ts <= ts (-1 if none)."""
        return int(np.searchsorted(self.ts_known, np.datetime64(ts.replace(tzinfo=None), "us"), side="right")) - 1

    def sigma_per_bar(self, freq: str, ts: datetime, i1: int) -> float:
        if freq == "4h" and self.f4_known is not None:
            j = int(np.searchsorted(self.f4_known, np.datetime64(ts.replace(tzinfo=None), "us"), side="right")) - 1
            if j >= 0 and np.isfinite(self.f4_sig_risk[j]):
                return float(self.f4_sig_risk[j])
            return float(self.sig_risk[i1] * 2.0)
        return float(self.sig_risk[i1])

    def vol_pctile(self, i: int) -> float | None:
        """1-year percentile of the current robust risk sigma vs its own hourly history (as tools.market_state)."""
        lo = max(0, i - 24 * 365 + 1)
        w = self.sig_risk[lo:i + 1]
        w = w[np.isfinite(w)]
        s = self.sig_risk[i]
        if len(w) < 24 * 30 or not np.isfinite(s):
            return None
        return float((w < s).mean() + 0.5 * (w == s).mean())

    def last_shock(self, i: int, lookback: int = 72, z_min: float = 4.0) -> dict | None:
        for k in range(i, max(-1, i - lookback), -1):
            if np.isfinite(self.z[k]) and abs(self.z[k]) >= z_min:
                return {"row": k, "z": float(self.z[k]), "sign": int(np.sign(self.ret[k])), "vol_state": self.vol_state[k], "age_bars": i - k}
        return None


# ====================================================================== jump model (monthly, point in time)

def _month_start(ts: datetime) -> datetime:
    return datetime(ts.year, ts.month, 1, tzinfo=UTC)


def fit_jump_params(st: SymbolState, month: datetime, cache_dir: Path | None = None):
    """Kou + Hawkes fit on the trailing 2 years of 1h returns with event_ts < month (scripts/jump_study.py). Cached."""
    from ai_trader.features.jump_diffusion import (
        JumpParams,
        bipower_sigma,
        fit_hawkes,
        fit_mixture,
        lee_mykland,
        posterior,
    )

    key = f"{st.symbol}_{month:%Y-%m}"
    if key in st.jump_cache:
        return st.jump_cache[key]
    cdir = cache_dir or (exec_dir() / "models" / "jump")
    p = cdir / f"{key}.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        prm = None if d.get("params") is None else JumpParams(**d["params"])
        st.jump_cache[key] = prm
        return prm
    ets = np.array([np.datetime64(t.replace(tzinfo=None), "us") for t in st.event_ts])
    i1 = int(np.searchsorted(ets, np.datetime64(month.replace(tzinfo=None), "us"), side="left"))
    i0 = max(0, i1 - TRAIN_BARS)
    prm = None
    meta: dict[str, Any] = {"symbol": st.symbol, "month": f"{month:%Y-%m}", "train_rows": i1 - i0, "fit_on": "1h event_ts < month start"}
    if i1 - i0 >= MIN_FIT_BARS:
        r = np.r_[np.nan, np.diff(np.log(st.close[i0:i1]))]
        sig = bipower_sigma(r, 72)
        ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
        prm = fit_mixture(r[ok], sig[ok])
        L, thr = lee_mykland(r, sig)
        with np.errstate(all="ignore"):
            post = posterior(np.nan_to_num(r), np.where(np.isfinite(sig) & (sig > 0), sig, 1.0), prm)
        det = (post > 0.9) & (L > thr) & np.isfinite(sig)
        times = np.flatnonzero(det[80:]).astype(float)
        mu, a, b, lr = fit_hawkes(times, float(len(r) - 80))
        prm.hawkes_mu, prm.hawkes_alpha, prm.hawkes_beta, prm.hawkes_lr = mu, a, b, lr
    cdir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**meta, "params": None if prm is None else prm.to_params()}, default=float), encoding="utf-8")
    st.jump_cache[key] = prm
    return prm


def hawkes_excess(st: SymbolState, i: int, prm, window: int = 400) -> float:
    """Current Hawkes excitation (lambda_{t+1} - mu) from detected jumps in bars <= t (causal)."""
    from ai_trader.features.jump_diffusion import bipower_sigma, lee_mykland, posterior

    if prm is None or not np.isfinite(prm.hawkes_mu) or not prm.hawkes_alpha:
        return 0.0
    lo = max(0, i - window - 80)
    r = np.r_[np.nan, np.diff(np.log(st.close[lo:i + 1]))]
    sig = bipower_sigma(r, 72)
    L, thr = lee_mykland(r, sig)
    with np.errstate(all="ignore"):
        post = posterior(np.nan_to_num(r), np.where(np.isfinite(sig) & (sig > 0), sig, 1.0), prm)
        det = (post > 0.9) & (L > thr) & np.isfinite(sig)
    A, decay = 0.0, math.exp(-prm.hawkes_beta)
    for flag in det:
        A = decay * (A + (1.0 if flag else 0.0))
    return float(prm.hawkes_alpha * A)


# ====================================================================== results

@dataclass
class GateResult:
    decision: str                       # pass | block (reduce is never emitted by K)
    rules_applied: list[str]
    reasons: list[str]
    blocked_by: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Placement:
    entry: float
    stop: float | None
    targets: list[float]
    sigma: dict
    p_tp: float | None
    net_er_r: float | None
    stop_sigma: float | None
    details: dict = field(default_factory=dict)


# ====================================================================== the gate

class BiasGate:
    def __init__(self, rules: list[dict] | None = None, *, jump: bool = True, jump_cache_dir: Path | None = None, n_paths: int = N_PATHS,
                 resting_target: bool = False):
        self.book = RuleBook(rules)
        self.resting_target = resting_target
        cfg = load()
        self.risk = cfg.risk
        pl_ = self.risk["placement"]
        self.min_sig, self.max_sig = float(pl_["min_stop_distance_sigma"]), float(pl_["max_stop_distance_sigma"])
        # 2026-11-14 (agent D): the cap crates/at-risk enforces is the H-STOP1 posterior when data/models/risk/stop_cap.json
        # exists (yaml value = prior / fallback); place against the same number so intents are not rejected on it.
        from ai_trader.risk.stop_posterior import applied_cap

        self.max_sig = applied_cap()[0]
        self.max_pct = float(self.risk["max_stop_distance_pct"]) / 100.0
        self.min_er = float(self.risk["min_net_expected_r"])
        self.jump = jump
        self.jump_cache_dir = jump_cache_dir
        self.n_paths = n_paths
        self._llm_cache: dict[tuple, list[str]] = {}
        self._hs3: dict[str, bool] = {}

    # ---------------------------------------------------------------- context
    def context(self, ev: SignalEvent, st: SymbolState) -> dict:
        i = st.row_at(ev.decision_ts)
        fday = ev.bar_close_ts.date()          # forecast day = UTC day of the next bar (the first bar held)
        wd = WEEKDAYS[ev.bar_close_ts.weekday()]
        sod = st.sod_cell.get(ev.decision_ts.date())   # the 1h row closing at 00:00 UTC of the decision day (known by then)
        return {"i": i, "regime_cell": sod, "cell_now": st.cell[i] if i >= 0 else None, "vol_state": st.vol_state[i] if i >= 0 else None,
                "dir_state": st.dir_state[i] if i >= 0 else None, "weekday": wd, "forecast_day": fday.isoformat(),
                "vol_pctile": st.vol_pctile(i) if i >= 0 else None, "asset_class": st.asset_class,
                "next_bar_tradfi_weekend": bool(st.next_weekend[i]) if i >= 0 else None,
                "dev_1h": float(st.dev[i]) if i >= 0 and np.isfinite(st.dev[i]) else None}

    # ---------------------------------------------------------------- rules
    def _h_s3(self, ts: datetime) -> bool:
        k = ts.date().isoformat()
        if k not in self._hs3:
            self._hs3[k] = holdout_supports("H-S3", ts)
        return self._hs3[k]

    def check(self, ev: SignalEvent, ctx: dict, st: SymbolState) -> GateResult:
        b = self.book
        applied, reasons, blocked, warns = [], [], [], []

        def hit(rid: str, why: str) -> None:
            sev = b.severity(rid)
            applied.append(rid)
            if sev == "block":
                blocked.append(rid)
                reasons.append(f"block {rid}: {why}")
            else:
                warns.append(rid)
                reasons.append(f"warn {rid} ({b.by_id[rid]['confidence']}/{b.by_id[rid]['status']}): {why}")

        if ev.kind == "exit":
            if b.active("R-EXIT-1"):
                applied.append("R-EXIT-1")
            reasons.append("exit from the strategy's own signal (R-EXIT-1); exits are never blocked")
            return GateResult("pass", applied, reasons)

        ac = ev.asset_class
        # side comes from the strategy only: no calendar / cell / ML direction tilt
        for rid in ("R-PX-6", "R-ML-5", "R-DS-6"):
            if b.active(rid) and b.applies(rid, ac):
                applied.append(rid)
        if "R-PX-6" in applied:
            reasons.append(f"side {ev.side} from the {ev.strategy} signal ({ev.signal:+.2f}); no direction tilt from calendar or regime cell")
        # R-AVOID-1
        if b.applies("R-AVOID-1", ac) and ev.strategy in b.avoid1:
            hit("R-AVOID-1", f"{ev.strategy} belongs to a family that failed its registered test")
        # regime-cell avoid lists
        cell = ctx["regime_cell"]
        shock = st.last_shock(ctx["i"]) if ev.strategy == "shock_regime" and ctx["i"] >= 0 else None
        reversal = bool(shock and np.sign(ev.held_signal) == -shock["sign"])
        for cr in b.cell_rules:
            rid = cr["id"]
            if ev.strategy not in cr["strategies"] or not b.applies(rid, ac):
                continue
            if cell is None:
                reasons.append(f"info {rid}: start-of-day cell unknown (warm-up), not checked")
                continue
            if cell not in cr["cells"]:
                continue
            if cr["vol_state"] and ctx["vol_state"] not in cr["vol_state"]:
                continue
            if cr["reversal_leg"] and not reversal:
                continue
            hit(rid, f"{ev.strategy} in start-of-day cell {cell} (avoid cells {sorted(cr['cells'])})")
        # R-PX-2: do not fade 2-sigma hourly deviations
        if b.applies("R-PX-2", ac) and ctx["dev_1h"] is not None:
            d = ctx["dev_1h"]
            side = 1 if ev.side == "long" else -1
            exempt = ev.strategy == "shock_regime" and self._h_s3(ev.decision_ts)
            if abs(d) >= FADE_SIGMA and side == -int(np.sign(d)) and not exempt:
                hit("R-PX-2", f"{ev.side} entry fades a {d:+.2f} sigma 1h deviation from the 48-bar EMA"
                    + (" (shock_regime reversal leg: H-S3 not yet recorded in the holdout)" if ev.strategy == "shock_regime" else ""))
            elif "R-PX-2" not in applied:
                applied.append("R-PX-2")
        # R-PX-9: no weekend directional trades
        if b.applies("R-PX-9", ac):
            if ev.strategy in WEEKEND_STRATEGIES:
                hit("R-PX-9", f"{ev.strategy} takes its direction from the weekend move")
            elif ctx["next_bar_tradfi_weekend"] and ev.strategy not in WEEKEND_EXEMPT:
                hit("R-PX-9", "next bar is inside the TradFi weekend window (Fri 17:00 ET - Sun 18:00 ET)")
            elif "R-PX-9" not in applied:
                applied.append("R-PX-9")
        if ev.cluster == "trend" and b.active("R-PD-2"):
            applied.append("R-PD-2")
            reasons.append("trend-family signal: one crypto TSMOM bet, cluster 'trend' (R-PD-1/R-PD-2)")
        decision = "block" if blocked else "pass"
        return GateResult(decision, list(dict.fromkeys(applied)), reasons, blocked, warns)

    # ---------------------------------------------------------------- placement
    def placement(self, ev: SignalEvent, ctx: dict, st: SymbolState) -> Placement:
        from ai_trader.agent.session import PRIOR_EDGE, Candidate, net_er, p_first
        from ai_trader.features.jump_diffusion import simulate_bracket

        i = ctx["i"]
        entry = float(ev.price)
        per_bar = st.sigma_per_bar(ev.freq, ev.decision_ts, i)
        s1 = float(st.sig_risk[i])
        risk_daily = s1 * math.sqrt(24)
        F1, F5 = st.loghar.get(date.fromisoformat(ctx["forecast_day"]), (float("nan"), float("nan")))
        crypto = st.asset_class == "crypto"
        wf_map = self.book.weekday_factor if crypto and self.book.active("R-ML-1") else dict.fromkeys(WEEKDAYS, 1.0)
        wf = wf_map[ctx["weekday"]]
        daily_lh = math.sqrt(F1) if np.isfinite(F1) and F1 > 0 else None
        sigma = {"per_bar": per_bar, "daily_logHAR": daily_lh, "weekday_factor": wf, "source": "vol.har",
                 "risk_daily": risk_daily, "freq": ev.freq, "central_1h": float(st.sig_rb[i])}
        det: dict[str, Any] = {"placement": "vol_only", "flags": ["no_structure"], "stop_type": "resting_mark (R-EXEC-1)",
                               "target_role": "resting" if self.resting_target else "evaluation", "exit_policy": "strategy signal (R-EXIT-1)"}
        if not (np.isfinite(per_bar) and per_bar > 0 and np.isfinite(risk_daily) and risk_daily > 0 and entry > 0):
            return Placement(entry, None, [], sigma, None, None, None, {**det, "error": "no valid risk sigma"})
        # stop: STOP_K daily risk sigmas, expressed in per-bar sigmas of the strategy's frequency (the contract's unit:
        # |ln(entry/stop)| / sigma.per_bar) and clamped to the placement bounds and the % cap
        bpd = 6 if ev.freq == "4h" else 24
        k_want = STOP_K * math.sqrt(bpd)
        # TICK_MARGIN: the engine rounds the stop away from entry to the tick, so a stop exactly at the bound would fail it
        k = min(max(k_want, self.min_sig * (1 + TICK_MARGIN)), self.max_sig * (1 - TICK_MARGIN))
        if k != k_want:
            det["flags"].append(f"stop_clamped_{'max' if k < k_want else 'min'}_sigma")
        if k * per_bar > self.max_pct:
            det["flags"].append("stop_capped_by_max_stop_distance_pct")
            k = self.max_pct / per_bar
        side = 1.0 if ev.side == "long" else -1.0
        dist = k * per_bar
        stop = entry * math.exp(-side * dist)
        tgt = entry * math.exp(side * TP_R * dist)
        det.update({"stop_k_bar_sigma": k, "stop_k_wanted": k_want, "stop_daily_sigma": dist / risk_daily, "reference_target": tgt})
        # central hourly sigma over the bracket horizon: direct 5-day log-HAR level x mean weekday factor of the horizon
        days = [(ev.bar_close_ts + timedelta(hours=h)).weekday() for h in range(SIM_BARS)]
        mwf = float(np.mean([wf_map[WEEKDAYS[d]] for d in days]))
        lvl = F5 if np.isfinite(F5) and F5 > 0 else (F1 if np.isfinite(F1) and F1 > 0 else (float(st.sig_rb[i]) ** 2) * 24)
        sig_c = math.sqrt(lvl * mwf / 24.0)
        det.update({"tp_r": TP_R, "sigma_central_1h_sim": sig_c, "loghar_5d_daily": math.sqrt(lvl),
                    "horizon_mean_weekday_factor": mwf})
        prm = None
        if crypto and self.jump and self.book.active("R-EXEC-2"):
            prm = fit_jump_params(st, _month_start(ev.decision_ts), self.jump_cache_dir)
        if prm is not None:
            seed = int(hashlib.sha256(f"{ev.symbol}|{ev.strategy}|{ev.decision_ts.isoformat()}|{ev.side}".encode()).hexdigest()[:8], 16)
            ex = hawkes_excess(st, i, prm)
            sim = simulate_bracket(entry, stop, tgt, sig_c, prm, SIM_BARS, excess0=ex, n_paths=self.n_paths, seed=seed, use_jumps=True)
            # no time barrier: the few paths still open after SIM_BARS resolve with the driftless odds from the entry
            p_null = sim["p_tp"] + sim["p_open"] * p_first(entry, stop, tgt)
            det.update({"p_source": "jump_mc (R-EXEC-2)", "p_tp_mc": sim["p_tp"], "p_sl_mc": sim["p_sl"], "p_open_mc": sim["p_open"],
                        "mean_R_mc": sim["mean_R"], "p_stop_worse_1p5R": sim["p_stop_worse_1p5R"], "hawkes_excess": ex, "sim_bars_1h": SIM_BARS,
                        "jump_month": f"{_month_start(ev.decision_ts):%Y-%m}"})
        else:
            p_null = p_first(entry, stop, tgt)
            det["p_source"] = "p_first (driftless Gaussian; " + ("TradFi: R-EXEC-2 is crypto-scoped)" if not crypto else "no jump fit)")
        sleeve = EDGE_SLEEVE.get(ev.strategy)
        edge = float(PRIOR_EDGE.get(sleeve, 0.0)) if sleeve else 0.0
        p_tp = min(0.95, max(0.0, p_null + edge))
        det.update({"p_null": p_null, "edge_prior": edge, "edge_source": f"agent.session.PRIOR_EDGE[{sleeve}]" if sleeve else "none (driftless null)"})
        cand = Candidate("x", ev.symbol, ev.strategy, ev.side, "market", entry, stop, tgt, p_null, p_tp, TP_R, None, "", ev.cluster)
        er = float(net_er(cand))
        return Placement(entry, stop, [tgt] if self.resting_target else [], sigma, p_tp, er, abs(math.log(stop / entry)) / per_bar, det)

    # ---------------------------------------------------------------- llm context
    def llm_rules(self, ctx: dict) -> list[str]:
        key = (ctx["asset_class"], ctx["weekday"], ctx["vol_state"], ctx["dir_state"], ctx["regime_cell"])
        if key not in self._llm_cache:
            stt = {"asset_class": key[0], "weekday": key[1], "vol_state": key[2], "dir_state": key[3], "cell": key[4]}
            s = select("strategy_choice", {k: v for k, v in stt.items() if v is not None}, self.book.rules)
            self._llm_cache[key] = [r["id"] for r in s["rules"] if r["id"] not in MACHINE_RULES]
        return self._llm_cache[key]

    # ---------------------------------------------------------------- one event
    def evaluate(self, ev: SignalEvent, st: SymbolState) -> tuple[GateResult, Placement | None, dict]:
        ctx = self.context(ev, st)
        if ctx["i"] < 0:
            g = GateResult("block", [], ["block: no 1h bar known at the decision time"], ["no_data"])
            return g, None, ctx
        g = self.check(ev, ctx, st)
        plc = None
        if ev.kind == "entry":
            plc = self.placement(ev, ctx, st)
            for rid in ("R-VOL-2", "R-VOL-3", "R-VOL-21", "R-EXEC-1"):
                if self.book.active(rid):
                    g.rules_applied.append(rid)
            if ev.asset_class == "crypto" and self.book.active("R-ML-1"):
                g.rules_applied.append("R-ML-1")
            if str(plc.details.get("p_source", "")).startswith("jump_mc"):
                g.rules_applied.append("R-EXEC-2")
            if plc.stop is None:
                g.decision = "block"
                g.blocked_by.append("no_placement")
                g.reasons.append("block: no valid risk sigma for the stop")
            elif plc.stop_sigma is not None and plc.stop_sigma < self.min_sig:
                g.decision = "block"
                g.blocked_by.append("stop_bounds")
                g.reasons.append(f"block: stop {plc.stop_sigma:.2f} per-bar sigma < min {self.min_sig} after the % cap")
            if plc.net_er_r is not None and plc.net_er_r < self.min_er:
                g.reasons.append(f"info: net_er_r {plc.net_er_r:+.3f} < min_net_expected_r {self.min_er} (the risk engine rejects it)")
            g.rules_applied = list(dict.fromkeys(g.rules_applied))
        ctx["llm_rules"] = self.llm_rules(ctx)
        return g, plc, ctx
