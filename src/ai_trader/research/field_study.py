"""H-FLD1-3: the BTC probability field against log-HAR-t + constant Kou jumps (program 2026-11-13, agent F1).

Design fixed before evaluation (diary 2026-11-13, "H-FLD design fixed before evaluation").

Data: Bybit BTCUSDT 1h bars (lake, from 2020-04-08 after burn-in), funding, 4h OI, regime v0 direction score; state
per ai_trader.field.state. Training rows from 2020-09-01 (OI available); first out-of-sample month 2022-01.

Field (ai_trader.field.kramers_moyal.fit_field_em): libraries = all monomials of degree <= 3 in the 9 standardised
coordinates for ln D (220 terms) and the drift (220), degree <= 2 for the jump intensity (55; ~1-3% of bars are jump
bars). Supports by stability selection (40 subsamples of half the dates, randomised lasso alpha 0.5, pi_thr 0.7, PFER
E[V] <= 1 per component) at the first refit, every January and at the knowledge cutoff (2026-07-01); coefficients
re-estimated by EM at every monthly refit (expanding window, warm start). Primary law (fixed after the
2022 pilot, before any holdout evaluation): non-jump branch a unit-variance Student-t (df by EM; the house default),
jump branch N(0,1) + Kou with mean jump size >= 3 diffusion sds (eta <= 1/3). The pilot's Gaussian core let the jump
branch absorb 7% -> 25% of bars as a second diffusion scale. Secondary 'field_gauss' (same fit, Gaussian core,
1h only) reported, not used for verdicts.

Baselines (per horizon h in {1, 4, 24} bars, fitted by MLE on the training origins of the same grid, same refits):
  B0 log-HAR-t + constant Kou (H-FLD1's comparator): R_h = s_t (m + e), ln s_t^2 = b0 + b1 ln RV24 + b2 ln RV168/7
     + b3 ln RV720/30, e ~ (1 - p) t_nu + p (N(0,1) + Kou) with constant p, Kou.
  B1 B0 + calendar: adds c * sbar_h, sbar_h = 0.5 ln mean of the hour-of-week factor over the next h bars (report only).
Origins: 1h every hour (closed-form field density); 4h at closes with hour % 4 == 0 and 1d at 00:00 UTC (MC, 1000 /
2000 paths, Rao-Blackwellised last step). Scores: log predictive density of R_h. Paired differences per origin are
averaged per UTC date, Newey-West t on the daily series, one-sided normal p. Segments by origin time: llm_in_sample
(< 2026-07-01), clean_holdout (>=).

Verdicts (literal):
  H-FLD1 per segment: supports if >= 2 of 3 horizons have diff > 0, t > 2 and Holm p < 0.05 (Holm over the FLD family
    of the segment: FLD1's three horizons, plus FLD3's three in the holdout); contradicts if >= 2 horizons have
    diff <= 0; else inconclusive.
  H-FLD2 per segment, at 1h and 1d: Brier skill of P(R_h > 0) vs the expanding training base rate (10-day block
    bootstrap, 90% CI) and the drift-on minus drift-off log-score difference (NW t). Supports if at both horizons the
    CI upper bound <= 0.005 and |t| < 2; contradicts if at either horizon the CI lower bound > 0 or t > 2; else
    inconclusive.
  H-FLD3 (clean_holdout): field frozen at the cutoff vs the same field with RLS-with-forgetting updates (memory 1440 h,
    prior = the cutoff fit); state construction frozen at the cutoff for both. Primary horizon 1h; supports if diff > 0,
    t > 2 and Holm p < 0.05; contradicts if diff <= 0; else inconclusive. 4h / 1d reported.

    uv run --extra ml python -m ai_trader.research.field_study pilot
    uv run --extra ml python -m ai_trader.research.field_study run
    uv run --extra ml python -m ai_trader.research.field_study analyse [--record]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
from scipy import optimize, special

from ai_trader.core.config import load
from ai_trader.field.kramers_moyal import _emg_cdf, fit_field_em, jump_logpdf
from ai_trader.field.model import Field, coords_matrix
from ai_trader.field.poly import Standardizer, monomials
from ai_trader.field.propagate import origins_from_state, simulate
from ai_trader.field.state import (
    COORDS,
    StateConfig,
    build_state,
    hour_profile,
    load_frame,
)
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t

SCR = Path(os.environ.get("FIELD_SCRATCH", "C:/Users/Asus/AppData/Local/Temp/claude/c--Users-Asus-Desktop-AI-Trader/31faef80-e320-4ca6-ab63-04fa013eeb03/scratchpad/F1"))
TRAIN_START = datetime(2020, 9, 1, tzinfo=UTC)
EVAL_START = datetime(2022, 1, 1, tzinfo=UTC)
HORIZONS = (1, 4, 24)
N_PATHS = {4: 1000, 24: 2000}
FORGET = 1 - 1 / 1440
REPORT_DATE = "2026-11-13"
LIB_DEG = {"mu": 3, "D": 3, "lam": 2}


def _models_dir() -> Path:
    d = load().root / "data" / "models" / "field"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _month_starts(a: datetime, b: datetime) -> list[datetime]:
    out, t = [], datetime(a.year, a.month, 1, tzinfo=UTC)
    while t < b:
        out.append(t)
        t = datetime(t.year + (t.month == 12), t.month % 12 + 1, 1, tzinfo=UTC)
    return out


def _ts_np(s: pl.Series) -> np.ndarray:
    return s.dt.replace_time_zone(None).to_numpy().astype("datetime64[ms]")


def _np(t: datetime) -> np.datetime64:
    return np.datetime64(t.astimezone(UTC).replace(tzinfo=None), "ms")


# ------------------------------------------------------------------ baseline: log-HAR-t + constant Kou

TLOGC = None


def _t_logpdf(z, nu):
    return special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - (nu + 1) / 2 * np.log1p(z * z / nu)


class LogHarT:
    """R = s (m + e), ln s^2 = Z b; e ~ (1 - p) t_nu + p (N(0,1) + Kou)."""

    def __init__(self, k: int):
        self.k = k
        self.theta: np.ndarray | None = None

    def unpack(self, th):
        k = self.k
        b = th[:k]
        nu = 2.05 + math.exp(th[k])
        m = th[k + 1]
        p = 0.3 * special.expit(th[k + 2])
        pu = special.expit(th[k + 3])
        eu, ed = math.exp(th[k + 4]), math.exp(th[k + 5])
        return b, nu, m, p, pu, eu, ed

    def _lp(self, th, Z, R):
        b, nu, m, p, pu, eu, ed = self.unpack(th)
        ls = 0.5 * (Z @ b)
        z = R / np.exp(ls) - m
        ln = _t_logpdf(z, nu)
        lj = jump_logpdf(z, np.ones_like(z), pu, eu, ed)
        return np.logaddexp(math.log1p(-p) + ln, math.log(p) + lj) - ls

    def fit(self, Z, R):
        if self.theta is None:
            y = np.log(R * R + 1e-12)
            b, *_ = np.linalg.lstsq(Z, y, rcond=None)
            b[0] += 1.27
            th0 = np.r_[b, math.log(2.0), 0.0, special.logit(0.02 / 0.3), 0.0, math.log(0.6), math.log(0.6)]
        else:
            th0 = self.theta
        f = lambda th: -float(np.sum(self._lp(th, Z, R))) / len(R)
        res = optimize.minimize(f, th0, method="L-BFGS-B", options={"maxiter": 400})
        self.theta = res.x
        return self

    def logpdf(self, Z, R):
        return self._lp(self.theta, Z, R)

    def cdf(self, Z, R):
        from scipy import stats

        b, nu, m, p, pu, eu, ed = self.unpack(self.theta)
        z = R / np.exp(0.5 * (Z @ b)) - m
        cn = stats.t.cdf(z, nu)
        cj = pu * _emg_cdf(z, eu) + (1 - pu) * (1 - _emg_cdf(-z, ed))
        return np.clip((1 - p) * cn + p * cj, 0, 1)

    def params(self) -> dict:
        b, nu, m, p, pu, eu, ed = self.unpack(self.theta)
        return {"b": b.tolist(), "nu": nu, "m": m, "p": p, "p_up": pu, "eta_up": eu, "eta_down": ed}


def har_regressors(state: pl.DataFrame) -> np.ndarray:
    r2 = pl.Series(np.nan_to_num(state["ret"].to_numpy().astype(float)) ** 2)
    a = [np.log(np.maximum(r2.rolling_sum(w).to_numpy().astype(float) / (w / 24), 1e-12)) for w in (24, 168, 720)]
    return np.column_stack([np.ones(state.height), *a])


def sbar(state: pl.DataFrame, profile: np.ndarray, h: int) -> np.ndarray:
    k0 = state["how_next"].to_numpy()
    f = np.asarray(profile)[(k0[:, None] + np.arange(h)[None, :]) % 168]
    return 0.5 * np.log(f.mean(1))


def fwd_sum(r: np.ndarray, h: int) -> np.ndarray:
    r = np.nan_to_num(r)
    c = np.r_[0.0, np.cumsum(r)]
    n = len(r)
    out = np.full(n, np.nan)
    out[: n - h] = c[1 + h: n + 1] - c[1: n - h + 1]
    return out


def origin_mask(state: pl.DataFrame, h: int) -> np.ndarray:
    hr = state["close_ts"].dt.hour().to_numpy()
    return np.ones(state.height, bool) if h == 1 else (hr % 4 == 0 if h == 4 else hr == 0)


# ------------------------------------------------------------------ one refit + its out-of-sample month


def libraries():
    p = len(COORDS)
    return {k: monomials(p, d) for k, d in LIB_DEG.items()}


def fit_field_at(frame: pl.DataFrame, t0: datetime, *, prev: dict | None, select: bool, cfg=StateConfig(), log=print):
    """Field fitted on rows whose next-bar return is known at t0. prev: the previous fit dict (support / warm start)."""
    prof = hour_profile(frame, t0)
    st = build_state(frame, prof, cfg)
    kts = _ts_np(st["known_ts"])
    k_next = np.r_[kts[1:], np.datetime64("2100-01-01", "ms")]
    tr = (kts >= _np(TRAIN_START)) & (k_next <= _np(t0)) & st["r_next"].is_not_null().to_numpy()
    X = coords_matrix(st)
    std = Standardizer.fit(X[tr], COORDS)
    Xs = std(X[tr])
    r = st["r_next"].to_numpy().astype(float)[tr]
    lnvf = np.log(st["vf"].to_numpy()[tr])
    groups = (kts[tr].astype("datetime64[D]").astype(np.int64))
    lib = libraries()
    diag = {}
    t = time.time()
    if select or prev is None:
        em, diag = fit_field_em(r, lnvf, Xs, COORDS, lib["mu"], lib["D"], lib["lam"], groups=groups, select=True, n_sub=40, pi_thr=0.7, ev=1.0,
                                progress=lambda k: log(f"    selected {k} ({time.time() - t:.0f}s)"))
    else:
        pf = Field.from_dict(prev["field"])
        from ai_trader.field.kramers_moyal import EMFit

        start = EMFit(pf.mu, pf.logD, pf.loglam, pf.kou, np.zeros(0), 0.0, 0, 0)
        em, _ = fit_field_em(r, lnvf, Xs, COORDS, [tuple(m) for m in pf.mu.monos], [tuple(m) for m in pf.logD.monos],
                             [tuple(m) for m in pf.loglam.monos], start=start, n_em=4)
        diag = {"support": prev.get("diag", {}).get("support"), "selected_at": prev.get("selected_at")}
    fld = Field.from_em(em, std, prof, cfg, {"fit_as_of": t0.isoformat(), "train_start": TRAIN_START.isoformat(), "n_train": int(tr.sum()),
                                             "jump_share": float(em.w.mean()), "em_iters": em.iters, "loglik_per_bar": em.loglik / max(em.n, 1)})
    return fld, st, tr, diag


def fit_baselines(st: pl.DataFrame, prof, t0: datetime, prev: dict | None):
    Z = har_regressors(st)
    kts = _ts_np(st["known_ts"])
    r = st["ret"].to_numpy().astype(float)
    out = {}
    for h in HORIZONS:
        R = fwd_sum(r, h)
        k_t = np.r_[kts[h:], np.full(h, np.datetime64("2100-01-01", "ms"))]
        ok = origin_mask(st, h) & (kts >= _np(TRAIN_START)) & (k_t <= _np(t0)) & np.isfinite(R) & np.isfinite(Z).all(1)
        Z1 = np.column_stack([Z, sbar(st, prof, h)])
        for name, ZZ in (("B0", Z), ("B1", Z1)):
            m = LogHarT(ZZ.shape[1])
            if prev and f"{name}_{h}" in prev:
                m.theta = np.asarray(prev[f"{name}_{h}"], float)
            m.fit(ZZ[ok], R[ok])
            out[f"{name}_{h}"] = m
        out[f"base_rate_{h}"] = float(np.mean(R[ok] > 0))
    return out, Z


def evaluate_month(fld: Field, st: pl.DataFrame, base: dict, Z: np.ndarray, m0: datetime, m1: datetime, *, mc: bool = True, seed: int = 0,
                   coef_hist=None) -> pl.DataFrame:
    kts = _ts_np(st["known_ts"])
    r = st["ret"].to_numpy().astype(float)
    X = coords_matrix(st)
    lnvf = np.log(st["vf"].to_numpy())
    inm = (kts >= _np(m0)) & (kts < _np(m1))
    rows = []
    for h in HORIZONS:
        if h > 1 and not mc:
            continue
        R = fwd_sum(r, h)
        idx = np.flatnonzero(inm & origin_mask(st, h) & np.isfinite(R) & np.isfinite(Z).all(1))
        if not len(idx):
            continue
        Zh = Z[idx]
        Z1 = np.column_stack([Zh, sbar(st, fld.profile, h)[idx]])
        res = {}
        for name, ZZ in (("B0", Zh), ("B1", Z1)):
            mdl = base[f"{name}_{h}"]
            lp = mdl.logpdf(ZZ, R[idx])
            cd = mdl.cdf(ZZ, R[idx])
            pu = 1 - mdl.cdf(ZZ, np.zeros(len(idx)))
            res[name] = (lp, cd, pu)
        if h == 1:
            for name, kw in (("field", {}), ("field_nodrift", {"drift": False}), ("field_gauss", {"core": "gauss"})):
                lp = fld.logpdf(R[idx], X[idx], lnvf[idx], **kw)
                cd = fld.cdf(R[idx], X[idx], lnvf[idx], **kw)
                pu = 1 - fld.cdf(np.zeros(len(idx)), X[idx], lnvf[idx], **kw)
                res[name] = (lp, cd, pu)
        else:
            org = origins_from_state(st, idx, fld.profile, h)
            for name, dr in (("field", True), ("field_nodrift", False)):
                sim = simulate(fld, org, h, N_PATHS[h], seed=seed + h, drift=dr, target=R[idx], keep_terminal=False)
                res[name] = (sim["logpdf"], sim["pit"], sim["p_up"])
        ts = st["known_ts"].gather(idx)
        for name, (lp, cd, pu) in res.items():
            rows.append(pl.DataFrame({"origin": ts, "h": np.full(len(idx), h, np.int32), "model": [name] * len(idx), "logpdf": lp, "pit": cd,
                                      "p_up": pu, "R": R[idx], "base_rate": np.full(len(idx), base[f"base_rate_{h}"])}))
    return pl.concat(rows) if rows else pl.DataFrame()


# ------------------------------------------------------------------ driver


def run(months: list[datetime], *, as_of: datetime, mc: bool = True, out_dir: Path = SCR / "months", log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_dir = out_dir.parent / ("fits" if out_dir.name == "months" else f"fits_{out_dir.name}")
    fits_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(as_of)
    cut = knowledge_cutoff()
    prev = None
    for i, m0 in enumerate(months):
        m1 = months[i + 1] if i + 1 < len(months) else as_of
        tag = f"{m0:%Y-%m}"
        fp, mp = fits_dir / f"{tag}.json", out_dir / f"{tag}.parquet"
        if fp.exists() and mp.exists():
            prev = json.loads(fp.read_text(encoding="utf-8"))
            continue
        t = time.time()
        select = prev is None or m0.month == 1 or m0 == cut
        fld, st, tr, diag = fit_field_at(frame, m0, prev=prev, select=select, log=log)
        base, Z = fit_baselines(st, fld.profile, m0, prev.get("base") if prev else None)
        ev = evaluate_month(fld, st, base, Z, m0, m1, mc=mc, seed=int(m0.timestamp()) % 100000)
        rec = {"month": tag, "field": fld.to_dict(), "diag": diag, "selected_at": tag if select else (prev or {}).get("selected_at"),
               "base": {k: v.theta.tolist() for k, v in base.items() if isinstance(v, LogHarT)},
               "base_params": {k: v.params() for k, v in base.items() if isinstance(v, LogHarT)},
               "base_rates": {k: v for k, v in base.items() if k.startswith("base_rate")}}
        if ev.height:
            ev.write_parquet(mp)
        if m0 == cut and out_dir.name == "months":
            fld.save(_models_dir() / f"btc_field_cutoff_{cut:%Y-%m-%d}.json")
        fp.write_text(json.dumps(rec, default=float), encoding="utf-8")
        prev = rec
        msg = " ".join(f"h{h}:{_quick(ev, h)}" for h in HORIZONS)
        log(f"[{tag}] select={select} n_train={fld.meta['n_train']} D_terms={len(fld.logD.monos)} mu_terms={len(fld.mu.monos)} "
            f"lam_terms={len(fld.loglam.monos)} jump_share={fld.meta['jump_share']:.3f} {msg} ({time.time() - t:.0f}s)")


def _quick(ev: pl.DataFrame, h: int) -> str:
    if ev.height == 0 or ev.filter(pl.col("h") == h).height == 0:
        return "-"
    e = ev.filter(pl.col("h") == h)
    f = e.filter(pl.col("model") == "field")["logpdf"].mean()
    b = e.filter(pl.col("model") == "B0")["logpdf"].mean()
    return f"{f - b:+.4f}"


# ------------------------------------------------------------------ H-FLD3: online vs frozen on the holdout


def run_fld3(as_of: datetime, log=print) -> pl.DataFrame:
    from ai_trader.field.kramers_moyal import mixture_cdf, mixture_logpdf
    from ai_trader.field.online import params_varying, run_online

    outp = SCR / "fld3.parquet"
    if outp.exists():
        return pl.read_parquet(outp)
    cut = knowledge_cutoff()
    fld = Field.load(_models_dir() / f"btc_field_cutoff_{cut:%Y-%m-%d}.json")
    frame = load_frame(as_of)
    st = build_state(frame, fld.profile, fld.cfg)
    kts = _ts_np(st["known_ts"])
    i0 = int(np.flatnonzero(kts >= _np(cut))[0])
    st = st.slice(i0 - 30, st.height - (i0 - 30))  # keep 30 bars of history for the 24h momentum window
    kts = _ts_np(st["known_ts"])
    X = coords_matrix(st)
    lnvf = np.log(st["vf"].to_numpy())
    r = st["ret"].to_numpy().astype(float)
    rn = np.r_[r[1:], np.nan]
    hold = kts >= _np(cut)
    Cm, Cd, Cl = run_online(fld, X[hold], lnvf[hold], rn[hold], forget=FORGET)
    hidx = np.flatnonzero(hold)
    rows = []
    for h in HORIZONS:
        R = fwd_sum(r, h)
        okh = origin_mask(st, h)[hidx] & np.isfinite(R[hidx])
        sel = np.flatnonzero(okh)          # positions within the holdout arrays
        idx = hidx[sel]
        if h == 1:
            lp_f = fld.logpdf(R[idx], X[idx], lnvf[idx])
            mu, D, p = params_varying(fld, X[idx], lnvf[idx], Cm[sel], Cd[sel], Cl[sel])
            lp_o = mixture_logpdf(R[idx], mu, D, p, fld.kou, fld.nu())
            pit_o = mixture_cdf(R[idx], mu, D, p, fld.kou, fld.nu())
            pit_f = fld.cdf(R[idx], X[idx], lnvf[idx])
        else:
            lp_f, pit_f = np.empty(len(idx)), np.empty(len(idx))
            lp_o, pit_o = np.empty(len(idx)), np.empty(len(idx))
            for k, (a, j) in enumerate(zip(sel, idx, strict=True)):  # common random numbers per origin for both arms
                o1 = origins_from_state(st, [j], fld.profile, h)
                sd_ = 11 + h + 100003 * k
                s_f = simulate(fld, o1, h, N_PATHS[h], seed=sd_, target=R[[j]], keep_terminal=False)
                s_o = simulate(fld, o1, h, N_PATHS[h], seed=sd_, target=R[[j]], keep_terminal=False, coefs=(Cm[a], Cd[a], Cl[a]))
                lp_f[k], pit_f[k] = s_f["logpdf"][0], s_f["pit"][0]
                lp_o[k], pit_o[k] = s_o["logpdf"][0], s_o["pit"][0]
        ts = st["known_ts"].gather(idx)
        for name, lp, pit in (("frozen", lp_f, pit_f), ("online", lp_o, pit_o)):
            rows.append(pl.DataFrame({"origin": ts, "h": np.full(len(idx), h, np.int32), "model": [name] * len(idx), "logpdf": lp, "pit": pit, "R": R[idx]}))
        log(f"FLD3 h={h}: frozen {np.mean(lp_f):.4f} online {np.mean(lp_o):.4f} n={len(idx)}")
    out = pl.concat(rows)
    out.write_parquet(outp)
    # coefficient drift summary
    drift = {"logD": {"start": Cd[0].tolist(), "end": Cd[-1].tolist()}, "mu": {"start": Cm[0].tolist(), "end": Cm[-1].tolist()},
             "loglam": {"start": Cl[0].tolist(), "end": Cl[-1].tolist()}}
    (SCR / "fld3_coef_drift.json").write_text(json.dumps(drift), encoding="utf-8")
    return out


# ------------------------------------------------------------------ analysis helpers


def paired(ev: pl.DataFrame, a: str, b: str, h: int) -> dict:
    e = ev.filter(pl.col("h") == h)
    x = e.filter(pl.col("model") == a).select("origin", pl.col("logpdf").alias("la"))
    y = e.filter(pl.col("model") == b).select("origin", pl.col("logpdf").alias("lb"))
    j = x.join(y, on="origin").with_columns((pl.col("la") - pl.col("lb")).alias("d"), pl.col("origin").dt.date().alias("date"))
    j = j.filter(pl.col("d").is_finite())
    if j.height < 10:
        return {"n": j.height, "diff": float("nan"), "t": float("nan"), "p": float("nan")}
    daily = j.group_by("date").agg(pl.col("d").mean()).sort("date")["d"].to_numpy()
    m, t, nd = newey_west_t(daily)
    p = 1 - NormalDist().cdf(t) if np.isfinite(t) else float("nan")
    return {"n": j.height, "days": nd, "diff": float(j["d"].mean()), "diff_daily": float(m), "t": float(t), "p": float(p),
            "mean_a": float(j["la"].mean()), "mean_b": float(j["lb"].mean())}


def pit_table(ev: pl.DataFrame, model: str, h: int) -> dict:
    e = ev.filter((pl.col("h") == h) & (pl.col("model") == model))
    u = e["pit"].to_numpy()
    u = u[np.isfinite(u)]
    if not len(u):
        return {}
    hist = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
    cov = {f"{int(100 * c)}": float(np.mean(np.abs(u - 0.5) < c / 2)) for c in (0.5, 0.8, 0.9, 0.95)}
    return {"n": len(u), "hist": hist.round(4).tolist(), "coverage": cov, "tail_lo_5": float(np.mean(u < 0.05)), "tail_hi_5": float(np.mean(u > 0.95))}


def brier_skill(ev: pl.DataFrame, model: str, h: int, block_days: int = 10, reps: int = 1000, seed: int = 0) -> dict:
    e = ev.filter((pl.col("h") == h) & (pl.col("model") == model)).with_columns(pl.col("origin").dt.date().alias("date"))
    e = e.filter(pl.col("p_up").is_finite())
    y = (e["R"].to_numpy() > 0).astype(float)
    p = e["p_up"].to_numpy()
    b = e["base_rate"].to_numpy()
    if len(y) < 30:
        return {"n": len(y)}
    d = e["date"].to_numpy()
    ud, inv = np.unique(d, return_inverse=True)
    blk = inv // block_days
    nb = blk.max() + 1
    se_m = np.bincount(blk, weights=(p - y) ** 2, minlength=nb)
    se_b = np.bincount(blk, weights=(b - y) ** 2, minlength=nb)
    bss = 1 - se_m.sum() / se_b.sum()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        k = rng.integers(0, nb, nb)
        boot.append(1 - se_m[k].sum() / se_b[k].sum())
    lo, hi = np.percentile(boot, [5, 95])
    return {"n": len(y), "bss": float(bss), "ci90": [float(lo), float(hi)], "mean_p": float(p.mean()), "hit_rate": float(y.mean())}


def holm(p: dict[str, float]) -> dict[str, float]:
    from ai_trader.features.vol_panel import holm as _h

    return _h(p)


__all__ = ["LogHarT", "brier_skill", "evaluate_month", "fit_baselines", "fit_field_at", "holm", "paired", "pit_table", "run", "run_fld3"]


def main() -> None:
    ap = argparse.ArgumentParser(prog="field_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sub.add_parser("run")
    a = sub.add_parser("analyse")
    a.add_argument("--record", action="store_true")
    args = ap.parse_args()
    as_of = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    if args.cmd == "pilot":
        run(_month_starts(EVAL_START, datetime(2023, 1, 1, tzinfo=UTC)), as_of=datetime(2023, 1, 1, tzinfo=UTC), mc=False, out_dir=SCR / "pilot")
    elif args.cmd == "run":
        run(_month_starts(EVAL_START, as_of), as_of=as_of)
    else:
        from ai_trader.research import field_report

        field_report.analyse(as_of, record=args.record)


if __name__ == "__main__":
    main()


_ = timedelta
