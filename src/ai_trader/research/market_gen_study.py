"""Backtests on generated markets: does a parameter-uncertain generator calibrated in-sample cover the holdout?
(block GEN: H-GEN1, H-GEN2, H-GEN3; generator in ai_trader.sim.market_gen). Research only: generated data is for the
deterministic strategies, never for the LLM layer (CLAUDE.md principle 1).

Design (fixed in the diary before any holdout comparison)
---------------------------------------------------------
* Calibration: llm_in_sample only (market_frame(as_of = knowledge cutoff), rows known before it), per universe:
  L = the strategy panel's 23 crypto perps; T = its 15 TradFi perps (13 of them have only 12-80 in-sample days, so T is
  calibrated mostly on PAXG / XAUT and is reported as feasible-but-weak).
* Generators, all continuing the real in-sample prefix from 2026-06-30 23:00 UTC to 2026-11-09 23:00 (3169 hours, the
  panel's holdout window 2026-07-01..2026-11-09 = 132 days):
    param  parametric generator, a fresh bootstrap replicate of every parameter per path (500 paths)
    fixed  the same generator at the point estimate on every path (200 paths)
    boot   B5-style stationary block bootstrap of in-sample relative bars (sim.resample), joint across symbols
           (one end-aligned index path, Politis-White mean block = median over symbols) (200 paths)
* Strategies: exactly the strategy panel's engines (strategy_panel.strategy_specs / strategy_fn, evaluate.strategy_pnl,
  7.5 bps per side), 1h frames with a 40-day real prefix, 4h frames aggregated from 1h with an 80-day prefix, funding
  (and OI for L) from the generator; daily P&L dated by close_ts, pnl_d = pnl / sqrt(bars per day), only dates in the
  holdout window and, per T symbol, on/after its panel start (as the real panel).
* Sharpe of a family x universe pair = sqrt(365) mean / sd of the daily family book (per date: mean pnl_d over the
  family's strategies x available symbols), identically for real (data/priors/strategy_panel_daily.parquet, clean_holdout)
  and generated paths.
* H-GEN1: per pair, rank q of the realised holdout Sharpe in the generated distribution; covered if 0.05 <= q <= 0.95;
  coverage with a Wilson 95% CI; same for `fixed` and `boot`. Verdict: supports if coverage(param) >= 0.80 and >
  coverage(fixed); contradicts if coverage(param) < 0.80 or < coverage(fixed); inconclusive if >= 0.80 but tied.
* H-GEN2 (universe L, param generator): 10 facts of the holdout window: (1) daily and (2) hourly excess kurtosis (mean
  over symbols), (3) mean |r| ACF at 1-24 h, (4) mean |r| ACF at 1-20 d, (5) weekday vol profile (7 log ratios),
  (6) hour-of-week vol profile (168 log ratios), (7) return factor share (first-eigenvalue share of the daily return
  correlation matrix), (8) vol factor share (same for daily log RV), (9) jump frequency (share of hours with
  |r| / bipower sigma(72h) above the Lee-Mykland 1% critical value), (10) median v0 episode length (hours, pooled).
  Scalars: inside if within the generated 5-95% band. Profiles: inside if the squared distance of the real profile to
  the generated median profile is <= the generated paths' 95th percentile of the same distance. Supports if >= 8 of 10.
* H-GEN3: Spearman of median generated Sharpe (param) vs realised holdout Sharpe over family x universe pairs; 95% CI
  from 10,000 bootstrap draws of pairs. Supports if rho > 0.3 and the CI lower bound > 0; inconclusive if rho > 0.3 but
  the CI includes 0; contradicts if rho <= 0.3.
* shock_regime is data-generated (strategy_panel docstring): it stays in its family as in the panel; reported.

    uv run --extra ml python -m ai_trader.research.market_gen_study calibrate --universe L
    uv run --extra ml python -m ai_trader.research.market_gen_study diag --universe L
    uv run --extra ml python -m ai_trader.research.market_gen_study run --universe L --gen param
    uv run --extra ml python -m ai_trader.research.market_gen_study evaluate [--record]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.sim import market_gen as mg

SCRATCH = Path(os.environ.get("AT_MARKET_GEN_DIR", r"C:\Users\Asus\AppData\Local\Temp\claude\c--Users-Asus-Desktop-AI-Trader"
                                                  r"\31faef80-e320-4ca6-ab63-04fa013eeb03\scratchpad\agentN\market_gen"))
GENS = ("param", "fixed", "boot")
N_PATHS = {"param": 500, "fixed": 200, "boot": 200}
N_REP = 500
CHUNK = 10
PREFIX_1H_DAYS = 40
PREFIX_4H_DAYS = 80
EVAL_END = date(2026, 11, 10)        # exclusive: the panel keeps dates < its as_of date (2026-11-10)
SEED = 20261113
REPORT = "market_gen_2026-11-13"
FACT_KEYS = ("kurt_daily", "kurt_hourly", "acf_abs_1h", "acf_abs_1d", "weekday_profile", "how_profile", "ret_factor_share",
             "vol_factor_share", "jump_freq", "regime_median_hours")
VECTOR_FACTS = ("weekday_profile", "how_profile")


def cutoff() -> datetime:
    from ai_trader.research.evaluate import knowledge_cutoff

    return knowledge_cutoff()


def eval_dates() -> list[date]:
    d0 = cutoff().date()
    return [d0 + timedelta(days=k) for k in range((EVAL_END - d0).days)]


def panel_meta() -> dict:
    from ai_trader.research.strategy_panel import priors_dir

    return json.loads((priors_dir() / "strategy_panel_meta.json").read_text(encoding="utf-8"))


def universe_symbols(u: str) -> list[str]:
    return list(panel_meta()["symbols"][u])


def gen_spec(u: str) -> mg.GenSpec:
    d0 = cutoff() - timedelta(days=1)
    return mg.GenSpec(day0=d0, n_days=(EVAL_END - d0.date()).days, first_hour=23, has_oi=(u == "L"))


# ====================================================================== calibration
_BASE: dict = {}


def _rep_init(path: str) -> None:
    with open(path, "rb") as f:
        _BASE.update(pickle.load(f))


def _rep_job(b: int) -> dict:
    cd, dv, jm = _BASE["cd"], _BASE["dv"], _BASE["jm"]
    w = mg.block_weights(cd.n_days, np.random.default_rng([SEED, 7, b]))
    r = mg.estimate(cd, dv, w, jumps=jm)
    r.pop("_zc", None)
    return r


def calib_path(u: str) -> Path:
    return SCRATCH / f"calib_{u}.pkl"


def calibrate(u: str, workers: int = 4, n_rep: int = N_REP) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    co = cutoff()
    cd = mg.load_calibration(u, universe_symbols(u), co)
    dv = mg.derive(cd)
    p0 = mg.estimate(cd, dv, np.ones(cd.n_days))
    jf, jm = mg.fit_jumps(p0.pop("_zc"), np.random.default_rng(SEED))
    point = mg.estimate(cd, dv, np.ones(cd.n_days), jumps=jm)
    point.pop("_zc", None)
    hp = mg.smm(point, jf)
    js = mg.jump_structure(point, hp, jf)
    drift = mg.profile_drift(cd, dv, point, jumps=jm)
    print(f"[{u}] point + jumps + smm in {time.time() - t0:.0f}s; smm objective {hp['objective']:.1f}; jump structure {js}; drift {drift}",
          flush=True)
    base_path = SCRATCH / f"calib_{u}_base.pkl"
    with open(base_path, "wb") as f:
        pickle.dump({"cd": cd, "dv": dv, "jm": jm}, f)
    reps_path = SCRATCH / f"reps_{u}.pkl"
    reps: list = []
    if reps_path.exists():
        reps = pickle.loads(reps_path.read_bytes())
    todo = list(range(len(reps), n_rep))
    with ProcessPoolExecutor(max_workers=workers, initializer=_rep_init, initargs=(str(base_path),)) as ex:
        for k in range(0, len(todo), 50):
            batch = todo[k:k + 50]
            reps.extend(ex.map(_rep_job, batch))
            reps_path.write_bytes(pickle.dumps(reps))     # checkpoint
            print(f"[{u}] replicates {len(reps)}/{n_rep} ({time.time() - t0:.0f}s)", flush=True)
    state = mg.start_state(cd, dv)
    bs = mg.bootstrap_source(cd)
    H = cd.C.shape[0]
    k0 = H - 24 * PREFIX_4H_DAYS
    ts = (int(cd.day0.timestamp()) + 3600 * np.arange(k0, H)).astype(np.int64)
    prefix = {}
    for i, s in enumerate(cd.symbols):
        m = np.isfinite(cd.C[k0:, i]) & np.isfinite(cd.O[k0:, i]) & np.isfinite(cd.Hi[k0:, i]) & np.isfinite(cd.Lo[k0:, i])
        prefix[s] = {"ts": ts[m], "open": cd.O[k0:, i][m], "high": cd.Hi[k0:, i][m], "low": cd.Lo[k0:, i][m], "close": cd.C[k0:, i][m],
                     "turnover": np.nan_to_num(cd.TV[k0:, i][m]), "funding": cd.FU[k0:, i][m],
                     "oi": cd.OI[k0:, i][m] if cd.OI is not None else None}
    diag = calib_diagnostics(cd, dv, point, hp, jf, reps)
    diag.update({"jump_structure": js, "profile_drift": drift, "cojump_phi": point["stats"]["cojump_phi"],
                 "tail_rate_data": point["stats"]["tail_rate"], "gpd": [point["stats"]["gpd_xi"], point["stats"]["gpd_scale"]]})
    out = {"universe": u, "symbols": cd.symbols, "point": point, "hp": hp, "jf": jf, "js": js, "drift": drift, "reps": reps,
           "state": state, "bs": bs,
           "prefix": prefix, "known_max": cd.known_max.isoformat(), "cutoff": co.isoformat(), "n_days": cd.n_days,
           "day0": cd.day0.isoformat(), "diag": diag}
    calib_path(u).write_bytes(pickle.dumps(out))
    print(f"[{u}] calibration saved ({time.time() - t0:.0f}s)", flush=True)
    return out


def calib_diagnostics(cd: mg.CalibData, dv: mg.Derived, point: dict, hp: dict, jf: mg.JumpFit, reps: list) -> dict:
    """Point estimates and replicate dispersion of the headline parameters (for the report)."""
    def q(vals):
        v = np.asarray(vals, float)
        return {"p05": float(np.nanquantile(v, 0.05)), "p50": float(np.nanquantile(v, 0.5)), "p95": float(np.nanquantile(v, 0.95))}

    hps = [mg.hourly_params(r, point, hp) for r in reps]
    kou = mg.JumpFit.unpack(jf.theta)
    out = {
        "n_symbols": cd.n_sym, "n_days": cd.n_days, "known_max": cd.known_max.isoformat(), "day0": cd.day0.isoformat(),
        "har_F": point["har_F"].tolist(), "har_F_persistence": q([r["har_F"][1:].sum() for r in reps]),
        "har_e": point["har_e"].tolist(), "har_e_persistence": q([r["har_e"].sum() for r in reps]),
        "sig_F": q([r["sig_F"] for r in reps]), "nu_F": point["nu_F"], "nu_e": point["nu_e"],
        "loading_b_median": float(np.median(point["b"])), "weekday_effect_pool": point["wk_pool"].tolist(),
        "gamma_cells": point["gamma"].tolist(), "phi_hS": point["phi_hS"].tolist(), "phi_dS": point["phi_dS"].tolist(),
        "phi_d_pooled": q([r["phi_d_pooled"] for r in reps]), "phi_h_pooled": q([r["phi_h_pooled"] for r in reps]),
        "stay_prob_daily": np.diag(point["P"]).tolist(),
        "hourly": {k: q([h[k] for h in hps]) for k in ("sig_h", "phi_h", "psi", "nu_eps")},
        "hourly_point": {k: hp[k] for k in mg.HOURLY_KEYS}, "smm_objective": hp["objective"],
        "smm_target": hp["target"].tolist(), "smm_fit": hp["fit_moments"].tolist(),
        "lam_median": float(np.median(point["lam_raw"] * hp["lam_scale"])),
        "kou": {"p": kou[0], "p_up": kou[1], "mean_up_z": 1 / kou[2], "mean_down_z": 1 / kou[3], "sigma0": jf.sigma0},
        "hawkes_branching_median": float(np.median([1 / (1 + math.exp(-h["th"][0])) for h in jf.hawkes])),
        "hawkes_beta_median": float(np.median([math.exp(h["th"][1]) for h in jf.hawkes])),
        "kappa_w": point["kappa_w"], "turnover": {"b_tau": point["b_tau"], "c_tau": point["c_tau"], "phi_v": point["phi_v"]},
        "funding": {k: point[k] for k in ("f_phi", "f_beta", "f_sig", "f_nu")},
    }
    if "o_phi" in point:
        out["oi"] = {k: point[k] for k in ("o_phi", "o_beta", "o_sig", "o_nu")}
    return out


# ====================================================================== frames and strategies on one path
def _dt(ts: np.ndarray) -> pl.Series:
    return pl.Series(np.asarray(ts, np.int64) * 1_000_000).cast(pl.Datetime("us", "UTC"))


def build_frames(pre: dict, gen: dict, gen_ts: np.ndarray, use_oi: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """1h frame (40-day real prefix + generated) and 4h frame (80-day prefix, complete UTC 4h groups), enriched like
    research.frame (enrich_bars) plus funding_rate / open_interest step series."""
    from ai_trader.research.frame import enrich_bars

    ts = np.r_[pre["ts"], gen_ts]
    cols = {k: np.r_[pre[k], gen[k]] for k in ("open", "high", "low", "close", "turnover", "funding")}
    base = pl.DataFrame({"event_ts": _dt(ts), "open": cols["open"], "high": cols["high"], "low": cols["low"], "close": cols["close"],
                         "turnover": cols["turnover"]}).with_columns(
        (pl.col("event_ts") + pl.duration(hours=1)).alias("close_ts"),
        (pl.col("event_ts") + pl.duration(hours=1, seconds=2)).alias("known_ts"),
        (pl.col("turnover") / pl.col("close")).alias("volume"))
    t1 = int(gen_ts[0]) - 3600 * 24 * PREFIX_1H_DAYS
    f1 = enrich_bars(base.filter(pl.col("event_ts") >= _dt(np.array([t1]))[0]).select(
        "event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover"))
    k1 = ts >= t1
    f1 = f1.with_columns(pl.Series("funding_rate", cols["funding"][k1]))
    if use_oi and pre.get("oi") is not None and "oi" in gen:
        oi = np.r_[pre["oi"], gen["oi"]][k1]
        loi = np.log(np.where(oi > 0, oi, np.nan))
        f1 = f1.with_columns(pl.Series("open_interest", oi), pl.Series("oi_chg_1", np.r_[np.nan, np.diff(loi)]))
    g = (base.group_by_dynamic("event_ts", every="4h").agg(
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
        pl.col("turnover").sum(), pl.col("known_ts").max(), pl.len().alias("_n"))
        .filter(pl.col("_n") == 4).drop("_n").with_columns((pl.col("event_ts") + pl.duration(hours=4)).alias("close_ts")))
    f4 = enrich_bars(g.select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover"))
    return f1, f4


def strategy_daily(f1: pl.DataFrame, f4: pl.DataFrame, specs: list, d0_epoch: int, n_dates: int, first_date_idx: int) -> np.ndarray:
    """(n_strategies, n_dates) daily pnl_d of every panel strategy (NaN before the symbol's first panel date)."""
    from ai_trader.research.evaluate import strategy_pnl
    from ai_trader.research.strategy_panel import BPD, strategy_fn

    out = np.full((len(specs), n_dates), np.nan)
    cache = {}
    for k, (name, _fam, freq) in enumerate(specs):
        f = f1 if freq == "1h" else f4
        if freq not in cache:
            ct = f["close_ts"].dt.epoch("s").to_numpy()
            di = (ct - d0_epoch) // 86400
            cache[freq] = (di, (di >= first_date_idx) & (di < n_dates))
        di, ok = cache[freq]
        p = strategy_pnl(f, strategy_fn(name)(f))["pnl"].to_numpy()
        s = np.bincount(di[ok], weights=p[ok], minlength=n_dates)[:n_dates] / math.sqrt(BPD[freq])
        s[:first_date_idx] = np.nan
        out[k] = s
    return out


# ====================================================================== stylised facts
def _acf_nan(x: np.ndarray, lags) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 10:
        return np.full(len(lags), np.nan)
    xc = np.where(m, x - np.nanmean(x), 0.0)
    den = np.sum(xc**2)
    return np.array([np.sum(xc[k:] * xc[:-k]) / den for k in lags])


def _excess_kurt(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    d = x - x.mean()
    return float(np.mean(d**4) / np.mean(d**2) ** 2 - 3.0)


def _pc1_share(X: np.ndarray) -> float:
    """First-eigenvalue share of the pairwise-complete correlation matrix of the columns of X (T, N)."""
    N = X.shape[1]
    C = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            C[i, j] = C[j, i] = np.corrcoef(X[m, i], X[m, j])[0, 1] if m.sum() > 10 else 0.0
    vals = np.linalg.eigvalsh(np.nan_to_num(C))
    return float(vals[-1] / N)


def facts(R: np.ndarray, wd0: int, runs: np.ndarray, n_jump: int, n_hours: int) -> dict:
    """Stylised facts of hourly returns R (H, N) on a UTC-day-aligned grid starting on weekday wd0 (0 = Monday)."""
    H, N = R.shape
    D = H // 24
    Rd = R[: D * 24].reshape(D, 24, N)
    comp = np.isfinite(Rd).all(axis=1)
    ret_d = np.where(comp, Rd.sum(axis=1), np.nan)
    rv_d = np.where(comp, (Rd**2).sum(axis=1), np.nan)
    wd = (wd0 + np.arange(D)) % 7
    how = np.repeat(wd, 24) * 24 + np.tile(np.arange(24), D)
    out = {
        "kurt_daily": float(np.nanmean([_excess_kurt(ret_d[:, i]) for i in range(N)])),
        "kurt_hourly": float(np.nanmean([_excess_kurt(R[:, i]) for i in range(N)])),
        "acf_abs_1h": float(np.nanmean([np.mean(_acf_nan(np.abs(R[:, i]), range(1, 25))) for i in range(N)])),
        "acf_abs_1d": float(np.nanmean([np.mean(_acf_nan(np.abs(ret_d[:, i]), range(1, 21))) for i in range(N)])),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = rv_d / np.nanmean(rv_d, axis=0, keepdims=True)
        out["weekday_profile"] = np.log(np.array([np.nanmean(rel[wd == k]) for k in range(7)]))
        s = R[: D * 24] ** 2 / np.nanmean(R[: D * 24] ** 2, axis=0, keepdims=True)
        out["how_profile"] = np.log(np.array([np.nanmean(s[how == h]) for h in range(mg.HOW)]))
        out["ret_factor_share"] = _pc1_share(ret_d)
        out["vol_factor_share"] = _pc1_share(np.log(rv_d))
    out["jump_freq"] = n_jump / max(n_hours, 1)
    out["regime_median_hours"] = float(np.median(runs)) if len(runs) else float("nan")
    return out


def symbol_fact_inputs(f1: pl.DataFrame, t_from: datetime, t_to: datetime) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """From a 1h frame: hourly returns on the [t_from, t_to) grid, v0 run lengths inside it, jump count / hours."""
    from ai_trader.features.jump_diffusion import bipower_sigma, lee_mykland
    from ai_trader.research.regimes import label
    from ai_trader.sim.resample import run_lengths

    n = int((t_to - t_from).total_seconds() // 3600)
    ets = f1["event_ts"].dt.epoch("s").to_numpy()
    idx = (ets - int(t_from.timestamp())) // 3600
    r = f1["ret"].to_numpy()
    # a return is valid only if the previous bar is the previous hour
    gap = np.r_[True, np.diff(ets) != 3600]
    r = np.where(gap, np.nan, r)
    ok = (idx >= 0) & (idx < n)
    R = np.full(n, np.nan)
    R[idx[ok]] = r[ok]
    sig = bipower_sigma(np.nan_to_num(r), 72)
    with np.errstate(invalid="ignore", divide="ignore"):
        Lst, crit = lee_mykland(np.nan_to_num(r), sig)
    jm = ok & np.isfinite(Lst) & np.isfinite(r)
    n_jump = int(np.sum(Lst[jm] > crit))
    cells = label(f1, bar_hours=1.0)["cell"].to_list()
    runs = run_lengths([c if o else None for c, o in zip(cells, ok, strict=True)])
    return R, runs, n_jump, int(jm.sum()), idx


# ====================================================================== simulation chunks
_CAL: dict = {}


def _calib(u: str) -> dict:
    if u not in _CAL:
        _CAL[u] = pickle.loads(calib_path(u).read_bytes())
    return _CAL[u]


def run_path_dir() -> Path:
    return SCRATCH / "runs"


def chunk_file(u: str, gen: str, chunk: int) -> Path:
    return run_path_dir() / f"{u}_{gen}_{chunk:03d}.npz"


def generate_chunk(u: str, gen: str, chunk: int) -> tuple[dict, int]:
    """Generated arrays (P, N, H_kept) of one chunk of paths and the number of paths."""
    C = _calib(u)
    N = len(C["symbols"])
    j0 = chunk * CHUNK
    P = min(CHUNK, N_PATHS[gen] - j0)
    spec = gen_spec(u)
    rng = np.random.default_rng([SEED, ord(u), GENS.index(gen), chunk])
    if gen == "boot":
        H = 24 * spec.n_days - spec.first_hour
        return mg.bootstrap_paths(C["bs"], C["state"], P, H, rng), P
    ps = []
    for j in range(j0, j0 + P):
        if gen == "param":
            ps.append(mg.path_params(C["reps"][j % len(C["reps"])], C["point"], C["hp"], C["jf"], N, np.random.default_rng([SEED, 99, j]),
                                     js=C["js"], drift=C["drift"]))
        else:
            ps.append(mg.path_params(C["point"], C["point"], C["hp"], C["jf"], N, None, js=C["js"]))
    pp = mg.stack_params(ps)
    return mg.simulate(pp, C["state"], spec, rng), P


def run_chunk(u: str, gen: str, chunk: int, with_facts: bool | None = None) -> str:
    from ai_trader.research.strategy_panel import strategy_specs

    out = chunk_file(u, gen, chunk)
    if out.exists():
        return str(out)
    t0 = time.time()
    C = _calib(u)
    syms = C["symbols"]
    g, P = generate_chunk(u, gen, chunk)
    spec = gen_spec(u)
    gen_ts = int((spec.day0 + timedelta(hours=spec.first_hour)).timestamp()) + 3600 * np.arange(g["close"].shape[2])
    specs = strategy_specs(u)
    dates = eval_dates()
    d0 = int(datetime(dates[0].year, dates[0].month, dates[0].day, tzinfo=UTC).timestamp())
    starts = panel_meta()["starts"]
    first_idx = {s: max(0, (date.fromisoformat(starts[s]) - dates[0]).days) for s in syms}
    S = np.zeros((P, len(specs), len(dates)))
    Nn = np.zeros((P, len(specs), len(dates)))
    with_facts = (u == "L") if with_facts is None else with_facts
    fact_rows = []
    co = cutoff()
    t_to = datetime(EVAL_END.year, EVAL_END.month, EVAL_END.day, tzinfo=UTC)
    for p in range(P):
        Rs, runs_all, nj, nh = [], [], 0, 0
        for i, s in enumerate(syms):
            gi = {k: g[k][p, i] for k in ("open", "high", "low", "close", "turnover", "funding")}
            if "oi" in g:
                gi["oi"] = g["oi"][p, i]
            f1, f4 = build_frames(C["prefix"][s], gi, gen_ts, use_oi=(u == "L"))
            x = strategy_daily(f1, f4, specs, d0, len(dates), first_idx[s])
            ok = np.isfinite(x)
            S[p] += np.where(ok, x, 0.0)
            Nn[p] += ok
            if with_facts:
                R, runs, a, b, _ = symbol_fact_inputs(f1, co, t_to)
                Rs.append(R)
                runs_all.append(runs)
                nj += a
                nh += b
        if with_facts:
            fact_rows.append(facts(np.column_stack(Rs), co.weekday(), np.concatenate(runs_all), nj, nh))
    run_path_dir().mkdir(parents=True, exist_ok=True)
    payload = {"S": S, "N": Nn, "strategies": np.array([s[0] for s in specs]), "families": np.array([s[1] for s in specs])}
    if fact_rows:
        for k in FACT_KEYS:
            payload[f"fact_{k}"] = np.array([fr[k] for fr in fact_rows])
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out)
    return f"{out.name} {time.time() - t0:.0f}s"


def run(u: str, gens=GENS, workers: int = 4) -> None:
    jobs = [(u, g, c) for g in gens for c in range(math.ceil(N_PATHS[g] / CHUNK)) if not chunk_file(u, g, c).exists()]
    print(f"[{u}] {len(jobs)} chunks to run", flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(_run_star, jobs):
            print("  done", msg, flush=True)


def _run_star(a):
    return run_chunk(*a)


# ====================================================================== in-sample diagnostics (no holdout)
def diag(u: str = "L", n_paths: int = 40, n_windows: int = 8) -> dict:
    """Design check on llm_in_sample only: facts of n_windows in-sample 132-day windows vs the generated envelope
    of paths from the cutoff (param generator), plus a timing estimate."""
    from ai_trader.research.frame import market_frame

    C = _calib(u)
    syms = C["symbols"]
    co = cutoff()
    L = len(eval_dates())
    real = []
    frames = {s: market_frame(s, "1h", co, funding=False, oi=False) for s in syms}
    for w in range(n_windows):
        t_to = co - timedelta(days=L * w)
        t_from = t_to - timedelta(days=L)
        Rs, runs, nj, nh = [], [], 0, 0
        for s in syms:
            f = frames[s].filter(pl.col("event_ts") < t_to)
            if f.filter(pl.col("event_ts") >= t_from).height < 24 * L * 0.9:
                continue
            R, rr, a, b, _ = symbol_fact_inputs(f, t_from, t_to)
            Rs.append(R)
            runs.append(rr)
            nj += a
            nh += b
        if len(Rs) >= 5:
            real.append({"window_end": t_to.date().isoformat(), "n_sym": len(Rs), **facts(np.column_stack(Rs), t_from.weekday(), np.concatenate(runs), nj, nh)})
    # generated (param) paths with facts (the first verdict seeds; stored separately, compared with in-sample windows only)
    t0 = time.time()
    gen_facts = []
    for c in range(math.ceil(n_paths / CHUNK)):
        f = SCRATCH / "diag" / f"{u}_diag_{c:03d}.npz"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            g, P = generate_chunk(u, "param", c)
            spec = gen_spec(u)
            gen_ts = int((spec.day0 + timedelta(hours=spec.first_hour)).timestamp()) + 3600 * np.arange(g["close"].shape[2])
            t_to = datetime(EVAL_END.year, EVAL_END.month, EVAL_END.day, tzinfo=UTC)
            rows = []
            for p in range(P):
                Rs, runs, nj, nh = [], [], 0, 0
                for i, s in enumerate(syms):
                    gi = {k: g[k][p, i] for k in ("open", "high", "low", "close", "turnover", "funding")}
                    f1, _ = build_frames(C["prefix"][s], gi, gen_ts, use_oi=False)
                    R, rr, a, b, _ = symbol_fact_inputs(f1, co, t_to)
                    Rs.append(R)
                    runs.append(rr)
                    nj += a
                    nh += b
                rows.append(facts(np.column_stack(Rs), co.weekday(), np.concatenate(runs), nj, nh))
            np.savez(f, **{k: np.array([r[k] for r in rows]) for k in FACT_KEYS})
        z = np.load(f)
        gen_facts.append({k: z[k] for k in FACT_KEYS})
    G = {k: np.concatenate([g[k] for g in gen_facts]) for k in FACT_KEYS}
    out = {"n_gen_paths": int(len(G["jump_freq"])), "secs_gen_facts": round(time.time() - t0, 1), "windows": [], "per_fact": {}}
    for r in real:
        inside = fact_inside(r, G)
        out["windows"].append({"window_end": r["window_end"], "n_sym": r["n_sym"], "inside": inside,
                               **{k: (float(r[k]) if k not in VECTOR_FACTS else None) for k in FACT_KEYS}})
    for k in FACT_KEYS:
        if k in VECTOR_FACTS:
            continue
        out["per_fact"][k] = {"gen_p05": float(np.nanquantile(G[k], 0.05)), "gen_p50": float(np.nanquantile(G[k], 0.5)),
                              "gen_p95": float(np.nanquantile(G[k], 0.95)),
                              "insample_windows": [float(r[k]) for r in real]}
    out["per_fact_share_inside"] = {k: float(np.mean([w["inside"][k] for w in out["windows"]])) for k in FACT_KEYS}
    (SCRATCH / f"diag_{u}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def fact_inside(real: dict, G: dict) -> dict:
    """Envelope test of every fact: scalars within the generated 5-95% band; profiles by squared distance to the
    generated median profile vs the generated 95th percentile of that distance."""
    res = {}
    for k in FACT_KEYS:
        g = np.asarray(G[k], float)
        if k in VECTOR_FACTS:
            med = np.nanmedian(g, axis=0)
            dg = np.nansum((g - med) ** 2, axis=1)
            dr = float(np.nansum((np.asarray(real[k]) - med) ** 2))
            res[k] = bool(dr <= np.nanquantile(dg, 0.95))
        else:
            v = float(real[k])
            res[k] = bool(np.nanquantile(g, 0.05) <= v <= np.nanquantile(g, 0.95))
    return res


# ====================================================================== evaluation
def load_runs(u: str, gen: str) -> dict | None:
    files = sorted(run_path_dir().glob(f"{u}_{gen}_*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if not files:
        return None
    zs = [np.load(f, allow_pickle=False) for f in files]
    out = {"S": np.concatenate([z["S"] for z in zs]), "N": np.concatenate([z["N"] for z in zs]),
           "strategies": [str(s) for s in zs[0]["strategies"]], "families": [str(s) for s in zs[0]["families"]], "n_files": len(files)}
    if "fact_jump_freq" in zs[0]:
        out["facts"] = {k: np.concatenate([z[f"fact_{k}"] for z in zs]) for k in FACT_KEYS}
    return out


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    """Annualised Sharpe of each row (paths, dates), NaN dates ignored (ddof 1, sqrt 365)."""
    m = np.nanmean(x, axis=1)
    sd = np.nanstd(x, axis=1, ddof=1)
    n = np.sum(np.isfinite(x), axis=1)
    return np.where((sd > 0) & (n >= 20), m / sd * math.sqrt(365), np.nan)


def gen_sharpes(r: dict, by: str) -> dict[str, np.ndarray]:
    """Per family (by='family') or strategy: Sharpe of the daily book on every path."""
    keys = r["families"] if by == "family" else r["strategies"]
    out = {}
    for k in sorted(set(keys)):
        ix = [i for i, x in enumerate(keys) if x == k]
        s = r["S"][:, ix, :].sum(axis=1)
        n = r["N"][:, ix, :].sum(axis=1)
        with np.errstate(invalid="ignore"):
            book = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out[k] = _sharpe_rows(book)
    return out


def real_sharpes(u: str, by: str) -> dict[str, float]:
    from ai_trader.research.strategy_panel import panel_path, sharpe_ann

    p = pl.read_parquet(panel_path()).filter((pl.col("universe") == u) & (pl.col("seg") == "clean_holdout")
                                             & (pl.col("date") < EVAL_END))
    col = "family" if by == "family" else "strategy"
    out = {}
    for k in sorted(p[col].unique().to_list()):
        book = p.filter(pl.col(col) == k).group_by("date").agg(pl.col("pnl_d").mean()).sort("date")["pnl_d"].to_numpy()
        out[k] = sharpe_ann(book)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def coverage(pairs: list[dict], gen: str) -> dict:
    rows = [p for p in pairs if p.get(f"q_{gen}") is not None and np.isfinite(p[f"q_{gen}"])]
    k = sum(1 for p in rows if 0.05 <= p[f"q_{gen}"] <= 0.95)
    return {"covered": k, "n": len(rows), "rate": k / len(rows) if rows else float("nan"), "wilson95": wilson(k, len(rows)),
            "median_band_width": float(np.median([p[f"p95_{gen}"] - p[f"p05_{gen}"] for p in rows])) if rows else float("nan")}


def pair_table(by: str) -> list[dict]:
    pairs = []
    for u in ("L", "T"):
        real = real_sharpes(u, by)
        gens = {g: load_runs(u, g) for g in GENS}
        sh = {g: gen_sharpes(r, by) for g, r in gens.items() if r is not None}
        for k, v in real.items():
            row = {"universe": u, "key": k, "real": v}
            for g, d in sh.items():
                x = d.get(k)
                if x is None:
                    continue
                x = x[np.isfinite(x)]
                if len(x) < 20 or not np.isfinite(v):
                    continue
                row.update({f"n_{g}": int(len(x)), f"q_{g}": float((np.sum(x < v) + 0.5 * np.sum(x == v)) / len(x)),
                            f"p05_{g}": float(np.quantile(x, 0.05)), f"p50_{g}": float(np.quantile(x, 0.5)),
                            f"p95_{g}": float(np.quantile(x, 0.95)), f"mean_{g}": float(np.mean(x)), f"sd_{g}": float(np.std(x))})
            pairs.append(row)
    return pairs


def spearman_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 31) -> dict:
    from ai_trader.research.strategy_panel import spearman

    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = spearman(a, b)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(a), len(a))
        bs.append(spearman(a[ix], b[ix]))
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]
    return {"rho": rho, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n_pairs": int(len(a)),
            "p_rho_le_0": float((np.sum(bs <= 0) + 1) / (len(bs) + 1))}


def real_facts_L() -> dict:
    """Holdout facts of universe L (evaluation stage only)."""
    from ai_trader.research.frame import market_frame

    co = cutoff()
    t_to = datetime(EVAL_END.year, EVAL_END.month, EVAL_END.day, tzinfo=UTC)
    as_of = datetime.fromisoformat(panel_meta()["as_of"])
    Rs, runs, nj, nh = [], [], 0, 0
    for s in universe_symbols("L"):
        f = market_frame(s, "1h", as_of, start=co - timedelta(days=PREFIX_1H_DAYS), funding=False, oi=False, burn_in=False)
        f = f.filter(pl.col("event_ts") < t_to)
        R, rr, a, b, _ = symbol_fact_inputs(f, co, t_to)
        Rs.append(R)
        runs.append(rr)
        nj += a
        nh += b
    return facts(np.column_stack(Rs), co.weekday(), np.concatenate(runs), nj, nh)


def evaluate(record: bool = False) -> dict:
    fam = pair_table("family")
    strat = pair_table("strategy")
    cov = {g: coverage(fam, g) for g in GENS}
    cov_s = {g: coverage(strat, g) for g in GENS}
    c_p, c_f = cov["param"]["rate"], cov["fixed"]["rate"]
    if c_p >= 0.8 and c_p > c_f:
        v1 = "supports"
    elif c_p < 0.8 or c_p < c_f:
        v1 = "contradicts"
    else:
        v1 = "inconclusive"
    # H-GEN2
    rf = real_facts_L()
    gl = load_runs("L", "param")
    G = gl["facts"]
    inside = fact_inside(rf, G)
    fact_rows = {}
    for k in FACT_KEYS:
        g = np.asarray(G[k], float)
        if k in VECTOR_FACTS:
            med = np.nanmedian(g, axis=0)
            dg = np.nansum((g - med) ** 2, axis=1)
            dr = float(np.nansum((np.asarray(rf[k]) - med) ** 2))
            lo, hi = np.nanquantile(g, 0.05, axis=0), np.nanquantile(g, 0.95, axis=0)
            fact_rows[k] = {"real_distance": dr, "gen_distance_p95": float(np.nanquantile(dg, 0.95)),
                            "pointwise_share_inside": float(np.mean((rf[k] >= lo) & (rf[k] <= hi))), "inside": inside[k],
                            "real": np.round(rf[k], 3).tolist() if k == "weekday_profile" else None,
                            "gen_median": np.round(med, 3).tolist() if k == "weekday_profile" else None}
        else:
            fact_rows[k] = {"real": float(rf[k]), "gen_p05": float(np.nanquantile(g, 0.05)), "gen_p50": float(np.nanquantile(g, 0.5)),
                            "gen_p95": float(np.nanquantile(g, 0.95)), "q": float(np.nanmean(g < rf[k])), "inside": inside[k]}
    other = {}
    for g in ("fixed", "boot"):
        r = load_runs("L", g)
        if r is not None and "facts" in r:
            other[g] = {"inside": fact_inside(rf, r["facts"]), "n_inside": int(sum(fact_inside(rf, r["facts"]).values()))}
    n_in = int(sum(inside.values()))
    v2 = "supports" if n_in >= 8 else "contradicts"
    # H-GEN3
    ok = [p for p in fam if "p50_param" in p]
    sp = spearman_ci(np.array([p["p50_param"] for p in ok]), np.array([p["real"] for p in ok]))
    if sp["rho"] > 0.3 and sp["ci95"][0] > 0:
        v3 = "supports"
    elif sp["rho"] > 0.3:
        v3 = "inconclusive"
    else:
        v3 = "contradicts"
    ok_s = [p for p in strat if "p50_param" in p]
    sp_s = spearman_ci(np.array([p["p50_param"] for p in ok_s]), np.array([p["real"] for p in ok_s]))
    sp_other = {}
    for g in ("fixed", "boot"):
        okg = [p for p in fam if f"p50_{g}" in p]
        if len(okg) >= 4:
            sp_other[g] = spearman_ci(np.array([p[f"p50_{g}"] for p in okg]), np.array([p["real"] for p in okg]), n_boot=2000)
    out = {
        "design": {"n_paths": N_PATHS, "n_replicates": N_REP, "chunk": CHUNK, "eval_window": [eval_dates()[0].isoformat(), eval_dates()[-1].isoformat()],
                   "prefix_days_1h": PREFIX_1H_DAYS, "prefix_days_4h": PREFIX_4H_DAYS, "seed": SEED},
        "calibration": {u: _calib(u)["diag"] for u in ("L", "T") if calib_path(u).exists()},
        "known_max": {u: _calib(u)["known_max"] for u in ("L", "T") if calib_path(u).exists()},
        "paths_loaded": {u: {g: (load_runs(u, g)["S"].shape[0] if load_runs(u, g) is not None else 0) for g in GENS} for u in ("L", "T")},
        "H-GEN1": {"verdict": v1, "coverage": cov, "coverage_strategy_level": cov_s, "pairs": fam},
        "H-GEN2": {"verdict": v2, "n_inside": n_in, "facts": fact_rows, "other_generators": other},
        "H-GEN3": {"verdict": v3, **sp, "strategy_level": sp_s, "other_generators": sp_other},
        "strategy_pairs": strat,
    }
    rp = Path(__file__).resolve().parents[3] / "research" / "reports" / f"{REPORT}.json"
    rp.write_text(json.dumps(out, indent=1, default=_json_default), encoding="utf-8")
    if record:
        record_results(out)
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer | np.floating | np.bool_):
        return o.item()
    return str(o)


def record_results(out: dict) -> list[str]:
    from ai_trader.research.hypotheses import record, results

    done = []
    data = f"research/reports/{REPORT}.json"
    g1, g2, g3 = out["H-GEN1"], out["H-GEN2"], out["H-GEN3"]
    rows = {
        "H-GEN1": (g1["verdict"], {"coverage_param": g1["coverage"]["param"], "coverage_fixed": g1["coverage"]["fixed"],
                                   "coverage_boot": g1["coverage"]["boot"], "coverage_strategy_level": g1["coverage_strategy_level"]},
                   "family x universe pairs (L 6, T 6); rank of realised clean_holdout Sharpe in >= 500 param / 200 fixed / 200 bootstrap paths"),
        "H-GEN2": (g2["verdict"], {"n_inside": g2["n_inside"], "facts": {k: v["inside"] for k, v in g2["facts"].items()},
                                   "other_generators": g2["other_generators"]}, "universe L, 500 param paths, holdout 2026-07-01..2026-11-09"),
        "H-GEN3": (g3["verdict"], {"rho": g3["rho"], "ci95": g3["ci95"], "n_pairs": g3["n_pairs"],
                                   "strategy_level": {k: g3["strategy_level"][k] for k in ("rho", "ci95", "n_pairs")}},
                   "Spearman of median generated Sharpe (param) vs realised holdout Sharpe over family x universe pairs"),
    }
    for hid, (verdict, metrics, note) in rows.items():
        if any(r["segment"] == "clean_holdout" for r in results(hid)):
            continue
        record(hid, "clean_holdout", verdict, metrics, note=note, data=data)
        done.append(hid)
    return done


# ====================================================================== report tables (markdown)
def _f(x, nd: int = 2) -> str:
    try:
        return "-" if x is None or not np.isfinite(float(x)) else f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def tables_md(out: dict) -> str:
    """Markdown tables of the evaluation output (the narrative is written around them)."""
    L = []
    g1 = out["H-GEN1"]
    L.append("| generator | covered / pairs | coverage | Wilson 95% | median band width (Sharpe) | strategy-level coverage |")
    L.append("|---|---|---|---|---|---|")
    for g in GENS:
        c, cs = g1["coverage"][g], g1["coverage_strategy_level"][g]
        L.append(f"| {g} | {c['covered']}/{c['n']} | {c['rate']:.2f} | [{c['wilson95'][0]:.2f}, {c['wilson95'][1]:.2f}] | "
                 f"{c['median_band_width']:.2f} | {cs['covered']}/{cs['n']} = {cs['rate']:.2f} |")
    L.append("")
    L.append("| universe | family | realised holdout Sharpe | param p05 / p50 / p95 (q) | fixed p05 / p95 (q) | boot p05 / p95 (q) |")
    L.append("|---|---|---|---|---|---|")
    for p in g1["pairs"]:
        def cell(g, full=False, p=p):
            if f"q_{g}" not in p:
                return "-"
            mark = "" if 0.05 <= p[f"q_{g}"] <= 0.95 else " **out**"
            mid = f" / {_f(p[f'p50_{g}'])}" if full else ""
            return f"{_f(p[f'p05_{g}'])}{mid} / {_f(p[f'p95_{g}'])} ({p[f'q_{g}']:.2f}){mark}"
        L.append(f"| {p['universe']} | {p['key']} | {_f(p['real'])} | {cell('param', True)} | {cell('fixed')} | {cell('boot')} |")
    L.append("")
    g2 = out["H-GEN2"]
    L.append("| fact (universe L, holdout) | realised | generated p05 / p50 / p95 | q | inside |")
    L.append("|---|---|---|---|---|")
    for k, v in g2["facts"].items():
        if k in VECTOR_FACTS:
            L.append(f"| {k} | distance {v['real_distance']:.3f} | gen p95 distance {v['gen_distance_p95']:.3f} "
                     f"(pointwise inside {v['pointwise_share_inside']:.2f}) | - | {'yes' if v['inside'] else 'no'} |")
        else:
            L.append(f"| {k} | {v['real']:.4g} | {v['gen_p05']:.4g} / {v['gen_p50']:.4g} / {v['gen_p95']:.4g} | {v['q']:.2f} | "
                     f"{'yes' if v['inside'] else 'no'} |")
    L.append(f"| **total** | | | | **{g2['n_inside']}/10** |")
    for g, v in g2["other_generators"].items():
        L.append(f"| ({g} generator) | | | | {v['n_inside']}/10 |")
    L.append("")
    g3 = out["H-GEN3"]
    L.append("| ranking statistic | Spearman | 95% CI | pairs |")
    L.append("|---|---|---|---|")
    L.append(f"| family x universe, param median vs realised | {g3['rho']:+.2f} | [{g3['ci95'][0]:+.2f}, {g3['ci95'][1]:+.2f}] | {g3['n_pairs']} |")
    sl = g3["strategy_level"]
    L.append(f"| strategy x universe (secondary) | {sl['rho']:+.2f} | [{sl['ci95'][0]:+.2f}, {sl['ci95'][1]:+.2f}] | {sl['n_pairs']} |")
    for g, v in g3["other_generators"].items():
        L.append(f"| family x universe, {g} median vs realised | {v['rho']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | {v['n_pairs']} |")
    return "\n".join(L)


# ====================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser(prog="market_gen_study")
    ap.add_argument("cmd", choices=("calibrate", "diag", "run", "evaluate"))
    ap.add_argument("--universe", default="L")
    ap.add_argument("--gen", default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.universe, a.workers)
    elif a.cmd == "diag":
        print(json.dumps(diag(a.universe), indent=1, default=float)[:6000])
    elif a.cmd == "run":
        run(a.universe, GENS if a.gen == "all" else tuple(a.gen.split(",")), a.workers)
    else:
        o = evaluate(a.record)
        print(json.dumps({k: o[k]["verdict"] for k in ("H-GEN1", "H-GEN2", "H-GEN3")}))


if __name__ == "__main__":
    main()
