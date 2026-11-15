"""BTC polynomial-neuron network study (H-PNN1..H-PNN3, BTC field program 2026-11-13, agent F2). Research only.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.pnn_study build
    ... search                 in-sample config search (rows before each horizon's first test month only)
    ... walk --h 1h|1d [--months 2022-01:2022-12]   walk-forward with a checkpoint per refit month
    ... evaluate [--record]    conformal, verdicts, Holm, artifacts, research/reports/btc_pnn_2026-11-13.{json,md}

Design (fixed before any holdout evaluation; diary 'PNN design fixed'):

Data (point-in-time; F1's field.state did not exist when this study started, so the existing ML features are used):
  1h  BTCUSDT bybit 1h market_frame as of AS_OF, ml.hourly.symbol_hourly -> complete_hourly (19 HOURLY_FEATURES, bars
      t-1 and earlier). Target y = r_t / sqrt(s2_t) (vol-standardised next-hour log return; s2 = robust_bc base,
      known at the bar open). Decision time = bar open; label end = open + 1 h; segment = bar open before / after the
      knowledge cutoff. Exogenous u_t for the latent dynamics = calendar of bar t+1 (hour-of-day and weekday sin/cos).
  1d  BTCUSDT rows of the cached universe-L daily panel (ml.features, as of 2026-11-12; decision T = 00:00 UTC of D+1),
      DAILY_FEATURES (14 of ML_FEATURES: vol level, recent returns, calendar, funding, jumps). Target
      y = ret_{D+1} / exp(lrv_w / 2). Segment = the panel's. u_t = weekday sin/cos and weekend flag of day D+2.
  Signed flow (delta) has no multi-year history in the lake (flow_bars start 2026), so it is not an input.

Walk-forward: monthly, expanding; training rows have label_end <= month start - 1 day (decision_budget purge +
  embargo); first test month = first with >= 365 training dates. Inside a window: models.val_split (last 15% of
  training dates, purged 1 day) = validation block (early stopping, split-conformal calibration); Standardizer on the
  fit block; inputs clipped to +/- pnn.CLIP.
Models (all on the same rows):
  pnn   KAN [n_in -> 4 latent -> 7 quantiles], Chebyshev degree 4 + SiLU residual; 2 seeds, quantiles averaged.
  mlp   equal-parameter ReLU MLP [n_in -> H -> 4 -> 7], same head, same training protocol, 2 seeds.
  har   log-HAR-t: OLS of the log variance proxy on the HAR terms (1d: log RV_{D+1} on lrv_d/w/m, F = exp(xb + s^2/2),
        R-VOL-20; 1h: log max(r^2/s2, 1e-6) on lsig2, lfac, har_d/w/m, F = s2 exp(xb + c), c the QLIKE-optimal
        constant on the training rows), then a Student-t (loc, scale, df by MLE) on r / sqrt(F) of the training rows.
  ctrl / fb   (H-PNN3, seed 0) the seed-0 pnn's encoder frozen; its head retrained (same cfg, seed, early stopping)
        without (ctrl) or with (fb) the predicted next latent zhat_{t+1} = A z_t + B u_t + c fitted on the fit block's
        consecutive pairs. Extra fb edges start at zero.
  Config search (equal effort, in-sample only): lr {1e-3, 3e-3} x weight decay {1e-4, 1e-2} per family and horizon,
  seed 0, rows before the first test month (80/20 purged date split), criterion validation pinball. Batch 1024 (1h),
  128 (1d); max 200 epochs, patience 10.
Losses and tests (per segment):
  H-PNN1  per-row pinball averaged over the 7 taus, in vol-standardised units (the common point-in-time scale);
          paired difference pnn - mlp and pnn - har averaged per UTC date, NW t over dates.
          Segment supports iff at >= 1 horizon both diffs < 0 with t < -2; contradicts iff at every horizon a baseline
          is at least as good (diff >= 0); else inconclusive. p = min over horizons of max(Phi(t_mlp), Phi(t_har)).
          Pooled: supports iff both segments support; contradicts iff either contradicts; else inconclusive.
  H-PNN2  ACI (gamma 0.005; window 1440 bars at 1h, 250 days at 1d; warm start = the first refit's validation
          scores) around the 2-seed pnn: 80% from [q10, q90], 90% and 95% from [q5, q95] (CQR scores). Coverage per
          level and horizon on the segment, width in return units vs the unconformalised log-HAR-t central interval
          at the same nominal level. Supports iff all six |coverage - nominal| <= 2 pp and all six mean widths <=
          the log-HAR-t widths; contradicts iff any coverage misses by > 4 pp or any width ratio > 1.10; else
          inconclusive. Split conformal (calibration = each refit's validation block) reported alongside. 10-day
          block-bootstrap CIs over dates. Registry segment = clean_holdout (pooled = clean_holdout); the
          llm_in_sample row applies the same rule to the in-sample walk-forward.
  H-PNN3  (a) latent predictability: R^2 gain of the dynamics over persistence for z_{t+1} (latent standardised by
          fit-block mean / sd, dims weighted equally, SST over the segment), rows with a consecutive next step under
          the same refit network; NW t of the per-date mean gain contribution. (b) pinball fb - ctrl per date, NW t.
          Supports iff at >= 1 horizon gain > 0.05 with t > 2 and the pinball diff < 0 with t < -2; contradicts iff
          at every horizon gain <= 0.05 or diff >= 0; else inconclusive. p = min over horizons of
          max(Phi(-t_gain), Phi(t_pin)). Registry segment = clean_holdout (pooled = clean_holdout).
  Holm within PNN per segment over the t-test p-values of H-PNN1 and H-PNN3 (H-PNN2 is a calibration criterion
  without a p-value); a 'supports' with Holm p > 0.05 becomes inconclusive.
Symbolic extraction: the seed-0 pnn of the first holdout refit (2026-07, labels to 2026-06-30); every edge fitted with
  degree 1-3 polynomials in the standardised input on its training rows; report the lowest degree with R^2 >= 0.99.

Artifacts (data/models/pnn/): wf_<h>/<YYYY-MM>.npz + .pt (per-refit checkpoint: predictions, latents, states,
  standardiser, dynamics), latent_<h>.parquet (dashboard; schema in SCHEMA below and schema.json), frozen_<h>.json
  (latest refit: KAN coefficients, standardiser, dynamics - the 'fit in Python, filter in Rust' hand-off).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import models as md
from ai_trader.ml import pnn
from ai_trader.ml.features import Standardizer
from ai_trader.ml.hourly import HOURLY_FEATURES, complete_hourly, symbol_hourly
from ai_trader.research.decision_budget_study import block_bootstrap_idx, holm, purged_train_mask, test_mask
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t

TAG = "2026-11-13"
SYMBOL = "BTCUSDT"
AS_OF = datetime(2026, 11, 13, tzinfo=UTC)
PANEL = "data/models/ml/cache/panel_2026-11-12.parquet"
DAILY_FEATURES = ("log_sigma", "lrv_d", "lrv_w", "lrv_m", "lrv_q", "ret_1d_z", "ret_7d_z", "abs_ret_1d_z", "dow_sin", "dow_cos",
                  "next_wk_hours", "funding_z_90d", "jump_share_7d", "vol_of_vol_30d")
assert len(DAILY_FEATURES) == 14
FEATURES = {"1h": HOURLY_FEATURES, "1d": DAILY_FEATURES}
HORIZONS = ("1h", "1d")
SEGS = ("llm_in_sample", "clean_holdout")
N_SEEDS = 2
BATCH = {"1h": 1024, "1d": 128}
MIN_TRAIN_DAYS = 365
DAY_US = np.timedelta64(86400 * 1_000_000, "us")
HOUR_US = np.timedelta64(3600 * 1_000_000, "us")
STEP = {"1h": HOUR_US, "1d": DAY_US}
LEVELS = (0.80, 0.90, 0.95)
BAND = {0.80: (1, 5), 0.90: (0, 6), 0.95: (0, 6)}  # quantile indices (lo, hi) in pnn.TAUS
ACI_GAMMA = 0.005
ACI_WINDOW = {"1h": 1440, "1d": 250}
FROZEN_MONTH = "2026-07"
QCOLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


def root() -> Path:
    p = load().root / "data" / "models" / "pnn"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cut() -> np.datetime64:
    return np.datetime64(knowledge_cutoff().replace(tzinfo=None), "us")


def _us(s: pl.Series) -> np.ndarray:
    return s.dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy()


# ------------------------------------------------------------------ data


def build_1h() -> pl.DataFrame:
    p = root() / "cache" / f"btc_1h_{TAG}.parquet"
    if p.is_file():
        return pl.read_parquet(p)
    from ai_trader.research.frame import market_frame

    f = market_frame(SYMBOL, "1h", AS_OF, funding=False, oi=False)
    h = complete_hourly(symbol_hourly(f))
    rr = f.select("event_ts", pl.col("ret").alias("r"))
    h = h.join(rr, on="event_ts", how="left").filter(pl.col("r").is_not_null() & pl.col("r").is_finite())
    assert np.allclose(h["r"].to_numpy() ** 2, h["r2"].to_numpy(), rtol=1e-9, atol=1e-15)
    p.parent.mkdir(parents=True, exist_ok=True)
    h.write_parquet(p)
    return h


def build_1d() -> pl.DataFrame:
    p = root() / "cache" / f"btc_1d_{TAG}.parquet"
    if p.is_file():
        return pl.read_parquet(p)
    d = pl.read_parquet(load().root / PANEL).filter(pl.col("symbol") == SYMBOL).sort("T")
    cols = [*DAILY_FEATURES, "ret_next", "lrv_next"]
    d = d.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cols]) & pl.col("segment").is_not_null())
    d = d.select("date", "T", "label_end", "segment", *DAILY_FEATURES, "ret_next", "lrv_next", "rv_next")
    p.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(p)
    return d


def arrays(h: str) -> dict:
    """Uniform arrays per horizon: X, y (vol-standardised target), sig (scale), r (raw return), T (decision, us),
    le (label end), seg, day, u (next-step calendar), har inputs."""
    if h == "1h":
        df = build_1h()
        T = _us(df["event_ts"])
        s2 = df["s2"].to_numpy()
        sig = np.sqrt(s2)
        r = df["r"].to_numpy()
        nxt = df["event_ts"] + timedelta(hours=1)
        hr = nxt.dt.hour().to_numpy().astype(float)
        wd = nxt.dt.weekday().to_numpy().astype(float)
        u = np.c_[np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), np.sin(2 * np.pi * (wd - 1) / 7), np.cos(2 * np.pi * (wd - 1) / 7)]
        seg = np.where(T < _cut(), "llm_in_sample", "clean_holdout")
        har = {"X": df.select("lsig2", "lfac", "har_d", "har_w", "har_m").to_numpy(), "ylog": df["y"].to_numpy(), "s2": s2}
        le = T + HOUR_US
    else:
        df = build_1d()
        T = _us(df["T"])
        sig = np.exp(0.5 * df["lrv_w"].to_numpy())
        r = df["ret_next"].to_numpy()
        d2 = df["date"] + timedelta(days=2)
        wd = d2.dt.weekday().to_numpy().astype(float)
        u = np.c_[np.sin(2 * np.pi * (wd - 1) / 7), np.cos(2 * np.pi * (wd - 1) / 7), (wd >= 6).astype(float)]
        seg = df["segment"].to_numpy()
        har = {"X": df.select("lrv_d", "lrv_w", "lrv_m").to_numpy(), "ylog": df["lrv_next"].to_numpy()}
        le = _us(df["label_end"])
    X = df.select(*FEATURES[h]).to_numpy().astype(float)
    return {"h": h, "X": X, "y": r / sig, "sig": sig, "r": r, "T": T, "le": le, "seg": seg,
            "day": T.astype("datetime64[D]"), "u": u, "har": har, "n": len(T)}


def months_of(T: np.ndarray) -> list[np.datetime64]:
    lo, hi = T.min().astype("datetime64[M]"), T.max().astype("datetime64[M]")
    return [np.datetime64(m, "us") for m in np.arange(lo, hi + np.timedelta64(1, "M"))]


def next_month(m: np.datetime64) -> np.datetime64:
    return np.datetime64(m.astype("datetime64[M]") + np.timedelta64(1, "M"), "us")


def first_test_month(a: dict) -> np.datetime64:
    for m in months_of(a["T"]):
        if len(np.unique(a["day"][purged_train_mask(a["le"], m)])) >= MIN_TRAIN_DAYS:
            return m
    raise RuntimeError("no month with enough history")


def next_index(a: dict) -> np.ndarray:
    """Index of the row exactly one step later (or -1)."""
    T = a["T"]
    j = np.searchsorted(T, T + STEP[a["h"]])
    ok = (j < len(T)) & (T[np.minimum(j, len(T) - 1)] == T + STEP[a["h"]])
    return np.where(ok, j, -1)


# ------------------------------------------------------------------ log-HAR-t


def fit_har_t(a: dict, tr: np.ndarray) -> dict:
    from scipy import stats

    H = a["har"]
    Xtr = np.c_[np.ones(len(tr)), H["X"][tr]]
    b, *_ = np.linalg.lstsq(Xtr, H["ylog"][tr], rcond=None)
    res = H["ylog"][tr] - Xtr @ b
    if a["h"] == "1d":
        s2r = float(np.var(res, ddof=Xtr.shape[1]))
        logF = lambda idx: np.c_[np.ones(len(idx)), H["X"][idx]] @ b + s2r / 2  # noqa: E731
        c = s2r / 2
    else:
        lf = np.log(H["s2"][tr]) + Xtr @ b
        c = md.bias_const(a["r"][tr] ** 2, lf)
        logF = lambda idx: np.log(H["s2"][idx]) + np.c_[np.ones(len(idx)), H["X"][idx]] @ b + c  # noqa: E731
    zt = a["r"][tr] / np.exp(0.5 * logF(tr))
    df, loc, scale = stats.t.fit(zt)
    return {"b": b.tolist(), "c": float(c), "df": float(df), "loc": float(loc), "scale": float(scale), "logF": logF}


def har_quantiles(loc, scale, df, sd_z, taus) -> np.ndarray:
    from scipy import stats

    """Student-t quantiles in model units, row-wise parameters (arrays of equal length)."""
    loc, scale, df, sd_z = (np.asarray(v, dtype=float) for v in (loc, scale, df, sd_z))
    z = stats.t.ppf(np.asarray(taus, dtype=float)[None, :], df[:, None])
    return (loc[:, None] + scale[:, None] * z) * sd_z[:, None]


# ------------------------------------------------------------------ search


def run_search() -> dict:
    out = {}
    for h in HORIZONS:
        a = arrays(h)
        m0 = first_test_month(a)
        rows = np.flatnonzero(purged_train_mask(a["le"], m0))
        fl, vl = md.val_split(a["T"][rows], 0.2)
        fi, vi = rows[fl], rows[vl]
        st = Standardizer().fit(a["X"][fi])
        Zf, Zv = (np.clip(st.transform(a["X"][i]), -pnn.CLIP, pnn.CLIP) for i in (fi, vi))
        k = pnn.pnn_params(Zf.shape[1])
        H = pnn.mlp_hidden_for(Zf.shape[1], k)
        res = {"pnn": [], "mlp": []}
        for cfg in pnn.GRID:
            c = {**cfg, "batch": BATCH[h]}
            t0 = time.time()
            p = pnn.fit_pnn(Zf, a["y"][fi], Zv, a["y"][vi], seed=0, cfg=c)
            res["pnn"].append({"cfg": cfg, "val": p.best_val, "epochs": p.epochs, "secs": round(time.time() - t0, 1)})
            t0 = time.time()
            m = pnn.fit_mlp(Zf, a["y"][fi], Zv, a["y"][vi], seed=0, hidden=H, cfg=c)
            res["mlp"].append({"cfg": cfg, "val": m.best_val, "epochs": m.epochs, "secs": round(time.time() - t0, 1)})
            print(h, cfg, res["pnn"][-1], res["mlp"][-1], flush=True)
        out[h] = {"first_test_month": str(m0)[:7], "n_fit": int(len(fi)), "n_val": int(len(vi)), "pnn_params": k, "mlp_hidden": H,
                  "mlp_params": pnn.mlp_params(Zf.shape[1], H), "scores": res,
                  "best": {f: min(res[f], key=lambda r: r["val"])["cfg"] for f in ("pnn", "mlp")}}
    (root() / "search.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _cfgs(h: str) -> dict:
    s = json.loads((root() / "search.json").read_text(encoding="utf-8"))[h]
    return {"pnn": {**s["best"]["pnn"], "batch": BATCH[h]}, "mlp": {**s["best"]["mlp"], "batch": BATCH[h]}, "hidden": s["mlp_hidden"]}


# ------------------------------------------------------------------ walk-forward


def wf_dir(h: str) -> Path:
    p = root() / f"wf_{h}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_walk(h: str, months: tuple[str, str] | None = None) -> None:
    import torch

    torch.set_num_threads(pnn.TORCH_THREADS)
    a = arrays(h)
    cf = _cfgs(h)
    nx = next_index(a)
    m0 = first_test_month(a)
    ms = [m for m in months_of(a["T"]) if m >= m0]
    if months:
        ms = [m for m in ms if months[0] <= str(m)[:7] <= months[1]]
    for m in ms:
        tag = str(m)[:7]
        f_npz, f_pt = wf_dir(h) / f"{tag}.npz", wf_dir(h) / f"{tag}.pt"
        if f_npz.is_file() and f_pt.is_file():
            continue
        te = np.flatnonzero(test_mask(a["T"], m, next_month(m)))
        if not len(te):
            continue
        t0 = time.time()
        tr = np.flatnonzero(purged_train_mask(a["le"], m))
        fl, vl = md.val_split(a["T"][tr])
        fi, vi = tr[fl], tr[vl]
        st = Standardizer().fit(a["X"][fi])
        Z = np.clip(st.transform(a["X"]), -pnn.CLIP, pnn.CLIP)
        y = a["y"]
        pn = [pnn.fit_pnn(Z[fi], y[fi], Z[vi], y[vi], seed=s, cfg=cf["pnn"]) for s in range(N_SEEDS)]
        ml = [pnn.fit_mlp(Z[fi], y[fi], Z[vi], y[vi], seed=s, hidden=cf["hidden"], cfg=cf["mlp"]) for s in range(N_SEEDS)]
        ev = np.r_[vi, te]
        q_pnn = pnn.ensemble_quantiles([p.predict(Z[ev]) for p in pn])
        q_mlp = pnn.ensemble_quantiles([p.predict(Z[ev]) for p in ml])
        # log-HAR-t on all training rows
        ht = fit_har_t(a, tr)
        sd_z = np.exp(0.5 * ht["logF"](ev)) / a["sig"][ev]
        # latent dynamics on the seed-0 pnn (fit block consecutive pairs)
        b0 = pn[0]
        zf_all = b0.latent(Z)
        pair = fi[(nx[fi] >= 0) & np.isin(nx[fi], fi)]
        dyn = pnn.fit_latent_dynamics(zf_all[pair], zf_all[nx[pair]], a["u"][pair])
        ctrl = pnn.fit_head_variant(b0, Z[fi], y[fi], Z[vi], y[vi], seed=0, cfg=cf["pnn"])
        fb = pnn.fit_head_variant(b0, Z[fi], y[fi], Z[vi], y[vi], seed=0, cfg=cf["pnn"], dyn=dyn, Uf=a["u"][fi], Uv=a["u"][vi])
        q_ctrl = ctrl.predict(Z[ev])
        q_fb = fb.predict(Z[ev], a["u"][ev])
        z_ev = zf_all[ev]
        zhat_ev = dyn.predict(z_ev, a["u"][ev])
        nxe = nx[ev]
        z_next = np.where((nxe >= 0)[:, None], zf_all[np.maximum(nxe, 0)], np.nan)
        np.savez_compressed(
            f_npz, idx=ev, is_test=np.r_[np.zeros(len(vi), bool), np.ones(len(te), bool)], q_pnn=q_pnn, q_mlp=q_mlp,
            q_ctrl=q_ctrl, q_fb=q_fb, har=np.c_[np.full(len(ev), ht["loc"]), np.full(len(ev), ht["scale"]), np.full(len(ev), ht["df"]), sd_z],
            q_pnn_seed=np.stack([p.predict(Z[ev]) for p in pn]), z=z_ev, zhat=zhat_ev, z_next=z_next, zmu=dyn.zmu, zsd=dyn.zsd)
        torch.save({"pnn": [p.state for p in pn], "pnn_meta": pn[0].meta, "mlp": [p.state for p in ml], "mlp_meta": ml[0].meta,
                    "ctrl": ctrl.state, "fb": fb.state, "fb_meta": fb.meta, "dyn": dyn.to_dict(), "std": st.to_dict(),
                    "har": {k: v for k, v in ht.items() if k != "logF"}, "fit_idx_range": [int(fi.min()), int(fi.max())],
                    "val_idx_range": [int(vi.min()), int(vi.max())],
                    "info": {"month": tag, "n_fit": int(len(fi)), "n_val": int(len(vi)), "n_test": int(len(te)),
                             "epochs_pnn": [p.epochs for p in pn], "epochs_mlp": [p.epochs for p in ml], "epochs_ctrl": ctrl.epochs,
                             "epochs_fb": fb.epochs, "secs": round(time.time() - t0, 1)}}, f_pt)
        print(json.dumps({"h": h, "month": tag, "n_fit": int(len(fi)), "n_test": int(len(te)), "ep_pnn": [p.epochs for p in pn],
                          "ep_mlp": [p.epochs for p in ml], "ep_ctrl": ctrl.epochs, "ep_fb": fb.epochs, "secs": round(time.time() - t0, 1)}), flush=True)


# ------------------------------------------------------------------ assemble


def assemble(h: str) -> dict:
    """Concatenate the per-refit checkpoints: test rows (time-ordered) and per-month validation blocks."""
    a = arrays(h)
    files = sorted(wf_dir(h).glob("*.npz"))
    te, val = [], {}
    for f in files:
        d = dict(np.load(f))
        tm = d["is_test"]
        te.append({k: (v[tm] if (np.ndim(v) >= 1 and len(v) == len(tm)) else v) for k, v in d.items()} | {"month": f.stem})
        val[f.stem] = {k: v[~tm] for k, v in d.items() if np.ndim(v) >= 1 and len(v) == len(tm)}
    cat = {}
    for k in ("idx", "q_pnn", "q_mlp", "q_ctrl", "q_fb", "har", "z", "zhat", "z_next"):
        cat[k] = np.concatenate([t[k] for t in te])
    cat["month"] = np.concatenate([np.full(len(t["idx"]), t["month"]) for t in te])
    cat["zmu"] = np.concatenate([np.tile(t["zmu"], (len(t["idx"]), 1)) for t in te])
    cat["zsd"] = np.concatenate([np.tile(t["zsd"], (len(t["idx"]), 1)) for t in te])
    o = np.argsort(a["T"][cat["idx"]], kind="stable")
    cat = {k: v[o] for k, v in cat.items()}
    i = cat["idx"]
    cat["q_har"] = har_quantiles(cat["har"][:, 0], cat["har"][:, 1], cat["har"][:, 2], cat["har"][:, 3], pnn.TAUS)
    for k in ("y", "sig", "r", "T", "seg", "day", "le"):
        cat[k] = a[k][i]
    return {"a": a, "te": cat, "val": val}


# ------------------------------------------------------------------ statistics


def _phi(t: float) -> float:
    return 0.5 * (1 + math.erf(t / math.sqrt(2))) if np.isfinite(t) else 1.0


def per_date(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    df = pl.DataFrame({"d": day[ok].astype("datetime64[D]").astype(np.int64), "x": x[ok]}).group_by("d").agg(pl.col("x").mean()).sort("d")
    return df["x"].to_numpy()


def paired(x: np.ndarray, day: np.ndarray) -> dict:
    d = per_date(x, day)
    m, t, n = newey_west_t(d)
    return {"diff": m, "t": t, "dates": n}


def boot_ci(daily: np.ndarray, fn=np.mean, seed: int = 0) -> list[float]:
    if len(daily) < 5:
        return [float("nan"), float("nan")]
    idx = block_bootstrap_idx(len(daily), 10, 1000, seed)
    v = np.array([fn(daily[i]) for i in idx])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def run_conformal(A: dict) -> dict:
    """ACI and split conformal around the pnn for every level; HAR-t nominal intervals. Returns per-row bands."""
    h = A["a"]["h"]
    te, val = A["te"], A["val"]
    y = te["y"]
    out = {}
    first = sorted(val)[0]
    for lev in LEVELS:
        a_ = 1 - lev
        il, ih = BAND[lev]
        lo, hi = te["q_pnn"][:, il], te["q_pnn"][:, ih]
        warm_v = val[first]
        yv = A["a"]["y"][warm_v["idx"]]
        warm = pnn.cqr_scores(warm_v["q_pnn"][:, il], warm_v["q_pnn"][:, ih], yv)
        r = pnn.aci(lo, hi, y, a_, gamma=ACI_GAMMA, window=ACI_WINDOW[h], warm=warm)
        # split conformal per refit month
        sc_lo, sc_hi = np.empty(len(y)), np.empty(len(y))
        for mo in np.unique(te["month"]):
            k = te["month"] == mo
            v = val[mo]
            s = pnn.cqr_scores(v["q_pnn"][:, il], v["q_pnn"][:, ih], A["a"]["y"][v["idx"]])
            sc_lo[k], sc_hi[k], _ = pnn.split_conformal(lo[k], hi[k], s, a_)
        hq = har_quantiles(te["har"][:, 0], te["har"][:, 1], te["har"][:, 2], te["har"][:, 3], (a_ / 2, 1 - a_ / 2))
        out[lev] = {"aci_lo": r["lo"], "aci_hi": r["hi"], "aci_alpha": r["alpha_t"], "sc_lo": sc_lo, "sc_hi": sc_hi,
                    "har_lo": hq[:, 0], "har_hi": hq[:, 1], "raw_lo": lo, "raw_hi": hi}
    return out


def eval_horizon(A: dict, C: dict) -> dict:
    te = A["te"]
    y, day, seg, sig = te["y"], te["day"], te["seg"], te["sig"]
    pin = {k: pnn.pinball_np(y, te[f"q_{k}"]) for k in ("pnn", "mlp", "har", "ctrl", "fb")}
    res = {}
    for s in SEGS:
        k = seg == s
        if not k.any():
            continue
        r = {"n": int(k.sum()), "dates": int(len(np.unique(day[k])))}
        r["pinball"] = {m: float(np.mean(v[k])) for m, v in pin.items()}
        r["pinball_raw_bps"] = {m: float(np.mean(v[k] * sig[k]) * 1e4) for m, v in pin.items()}
        r["pnn_vs_mlp"] = paired(pin["pnn"][k] - pin["mlp"][k], day[k])
        r["pnn_vs_har"] = paired(pin["pnn"][k] - pin["har"][k], day[k])
        r["mlp_vs_har"] = paired(pin["mlp"][k] - pin["har"][k], day[k])
        r["fb_vs_ctrl"] = paired(pin["fb"][k] - pin["ctrl"][k], day[k])
        r["ctrl_vs_pnn_seed0_note"] = "ctrl = seed-0 encoder frozen, head retrained; pnn = 2-seed ensemble"
        # per-tau pinball
        t = np.asarray(pnn.TAUS)[None, :]
        r["per_tau"] = {}
        for m in ("pnn", "mlp", "har"):
            e = y[k][:, None] - te[f"q_{m}"][k]
            r["per_tau"][m] = np.mean(np.maximum(t * e, (t - 1) * e), axis=0).tolist()
        # quantile calibration (fraction below each quantile)
        r["hit_rate"] = {m: np.mean(y[k][:, None] <= te[f"q_{m}"][k], axis=0).tolist() for m in ("pnn", "mlp", "har")}
        # conformal
        cov = {}
        for lev in LEVELS:
            c = C[lev]
            row = {}
            for name in ("aci", "sc", "har", "raw"):
                if name == "raw" and lev == 0.95:
                    continue
                lo, hi = c[f"{name}_lo"][k], c[f"{name}_hi"][k]
                hit = ((y[k] >= lo) & (y[k] <= hi)).astype(float)
                wid = (hi - lo) * sig[k] * 1e4  # bps of return
                dh = per_date(hit, day[k])
                dw = per_date(wid, day[k])
                row[name] = {"coverage": float(hit.mean()), "cov_ci": boot_ci(dh), "width_bps": float(wid.mean()), "width_ci": boot_ci(dw)}
            wa = (c["aci_hi"][k] - c["aci_lo"][k]) * sig[k]
            wh = (c["har_hi"][k] - c["har_lo"][k]) * sig[k]
            dwa, dwh = per_date(wa, day[k]), per_date(wh, day[k])
            row["aci_over_har_width"] = float(wa.mean() / wh.mean())
            row["aci_over_har_ci"] = boot_ci(np.c_[dwa, dwh], fn=lambda x: x[:, 0].mean() / x[:, 1].mean()) if len(dwa) == len(dwh) else [np.nan, np.nan]
            row["aci_alpha_last"] = float(c["aci_alpha"][k][-1])
            cov[str(lev)] = row
        r["conformal"] = cov
        # latent predictability
        zmu, zsd = te["zmu"][k], te["zsd"][k]
        ok = np.all(np.isfinite(te["z_next"][k]), axis=1)
        zt = ((te["z"][k] - zmu) / zsd)[ok]
        zn = ((te["z_next"][k] - zmu) / zsd)[ok]
        zp = ((te["zhat"][k] - zmu) / zsd)[ok]
        var = zn.var(axis=0)
        var = np.where(var > 0, var, 1.0)
        g = (((zn - zt) ** 2 - (zn - zp) ** 2) / var[None, :]).mean(axis=1)
        sst = ((zn - zn.mean(0)) ** 2).sum(0)
        r["latent"] = {"pairs": int(ok.sum()), "r2_model": (1 - ((zn - zp) ** 2).sum(0) / sst).tolist(),
                       "r2_persist": (1 - ((zn - zt) ** 2).sum(0) / sst).tolist(), "gain": float(g.mean()),
                       "gain_t": paired(g, day[k][ok])["t"], "gain_dates": paired(g, day[k][ok])["dates"],
                       "lag1_autocorr": [float(np.corrcoef(zt[:, j], zn[:, j])[0, 1]) for j in range(zt.shape[1])]}
        res[s] = r
    return res


# ------------------------------------------------------------------ verdicts


def v_pnn1(E: dict, s: str) -> tuple[str, float, dict]:
    hs = {h: E[h][s] for h in HORIZONS if s in E[h]}
    win = [h for h, r in hs.items() if r["pnn_vs_mlp"]["diff"] < 0 and r["pnn_vs_mlp"]["t"] < -2 and r["pnn_vs_har"]["diff"] < 0 and r["pnn_vs_har"]["t"] < -2]
    lose = [h for h, r in hs.items() if r["pnn_vs_mlp"]["diff"] >= 0 or r["pnn_vs_har"]["diff"] >= 0]
    p = min(max(_phi(r["pnn_vs_mlp"]["t"]), _phi(r["pnn_vs_har"]["t"])) for r in hs.values())
    v = "supports" if win else ("contradicts" if len(lose) == len(hs) else "inconclusive")
    return v, p, {h: {"vs_mlp": r["pnn_vs_mlp"], "vs_har": r["pnn_vs_har"], "pinball": r["pinball"]} for h, r in hs.items()}


def v_pnn2(E: dict, s: str) -> tuple[str, float, dict]:
    ok_all, bad = True, False
    m = {}
    for h in HORIZONS:
        for lev in LEVELS:
            c = E[h][s]["conformal"][str(lev)]
            cov, ratio = c["aci"]["coverage"], c["aci_over_har_width"]
            ok_all &= abs(cov - lev) <= 0.02 and ratio <= 1.0
            bad |= abs(cov - lev) > 0.04 or ratio > 1.10
            m[f"{h}_{int(lev * 100)}"] = {"aci_cov": cov, "sc_cov": c["sc"]["coverage"], "har_cov": c["har"]["coverage"],
                                          "aci_width_bps": c["aci"]["width_bps"], "har_width_bps": c["har"]["width_bps"], "width_ratio": ratio}
    v = "supports" if ok_all else ("contradicts" if bad else "inconclusive")
    return v, float("nan"), m


def v_pnn3(E: dict, s: str) -> tuple[str, float, dict]:
    hs = {h: E[h][s] for h in HORIZONS if s in E[h]}
    win = [h for h, r in hs.items() if r["latent"]["gain"] > 0.05 and r["latent"]["gain_t"] > 2 and r["fb_vs_ctrl"]["diff"] < 0 and r["fb_vs_ctrl"]["t"] < -2]
    lose = [h for h, r in hs.items() if r["latent"]["gain"] <= 0.05 or r["fb_vs_ctrl"]["diff"] >= 0]
    p = min(max(_phi(-r["latent"]["gain_t"]), _phi(r["fb_vs_ctrl"]["t"])) for r in hs.values())
    v = "supports" if win else ("contradicts" if len(lose) == len(hs) else "inconclusive")
    return v, p, {h: {"latent_gain": r["latent"]["gain"], "gain_t": r["latent"]["gain_t"], "fb_vs_ctrl": r["fb_vs_ctrl"]} for h, r in hs.items()}


def verdicts(E: dict) -> dict:
    out = {}
    for s in SEGS:
        v1, p1, m1 = v_pnn1(E, s)
        v2, p2, m2 = v_pnn2(E, s)
        v3, p3, m3 = v_pnn3(E, s)
        hp = holm({"H-PNN1": p1, "H-PNN3": p3})
        rows = {"H-PNN1": [v1, p1, m1], "H-PNN2": [v2, p2, m2], "H-PNN3": [v3, p3, m3]}
        for hid in ("H-PNN1", "H-PNN3"):
            if rows[hid][0] == "supports" and hp[hid] > 0.05:
                rows[hid][0] = "inconclusive"
        out[s] = {hid: {"verdict": r[0], "p": r[1], "holm_p": hp.get(hid, float("nan")), "metrics": r[2]} for hid, r in rows.items()}
    i, c = out["llm_in_sample"], out["clean_holdout"]
    v1 = i["H-PNN1"]["verdict"], c["H-PNN1"]["verdict"]
    pooled1 = "supports" if v1 == ("supports", "supports") else ("contradicts" if "contradicts" in v1 else "inconclusive")
    out["pooled"] = {"H-PNN1": {"verdict": pooled1, "rule": "supports iff both segments support"},
                     "H-PNN2": {"verdict": c["H-PNN2"]["verdict"], "rule": "registry test is on clean_holdout"},
                     "H-PNN3": {"verdict": c["H-PNN3"]["verdict"], "rule": "registry test is on clean_holdout"}}
    return out


# ------------------------------------------------------------------ symbolic extraction


def symbolic(h: str, month: str = FROZEN_MONTH) -> dict:
    import torch

    a = arrays(h)
    ck = torch.load(wf_dir(h) / f"{month}.pt", weights_only=False)
    st = Standardizer()
    st.mu, st.sd = np.asarray(ck["std"]["mu"]), np.asarray(ck["std"]["sd"])
    lo, hi = ck["fit_idx_range"]
    Z = np.clip(st.transform(a["X"][lo:hi + 1]), -pnn.CLIP, pnn.CLIP)
    m = pnn.make_pnn(**ck["pnn_meta"])
    m.load_state_dict(ck["pnn"][0])
    m.eval()
    names = list(FEATURES[h])
    lat = [f"z{j}" for j in range(m.latent)]
    enc = pnn.edge_table(m.enc, Z, names, lat)
    with torch.no_grad():
        zz = m.latent_of(torch.as_tensor(Z, dtype=torch.float32)).numpy().astype(float)
    head = pnn.edge_table(m.head, zz, lat, ["o05", "o10", "o25", "o50", "o75", "o90", "o95"])
    # input importance = summed edge importance into the latents
    imp = {}
    for r in enc:
        imp[r["in"]] = imp.get(r["in"], 0.0) + r["importance"]
    # what each latent tracks: Spearman with the inputs
    from scipy.stats import spearmanr

    tracks = {}
    for j in range(m.latent):
        cors = {n: float(spearmanr(zz[:, j], Z[:, i]).statistic) for i, n in enumerate(names)}
        tracks[lat[j]] = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:4]
    share_poly = {d: float(np.mean([r["deg"] <= d and r["r2"] >= 0.99 for r in enc])) for d in (1, 2, 3)}
    strip = lambda rows: [{k: v for k, v in r.items() if k != "fits"} | {"r2_by_deg": [round(f["r2"], 4) for f in r["fits"]]} for r in rows]  # noqa: E731
    return {"month": month, "n_train": int(len(Z)), "std_mu": dict(zip(names, st.mu.tolist(), strict=True)),
            "std_sd": dict(zip(names, st.sd.tolist(), strict=True)), "encoder_edges": strip(enc), "head_edges": strip(head),
            "input_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])), "latent_tracks": tracks,
            "share_edges_poly_r2_099": share_poly, "latent_bias": m.enc.bias.detach().numpy().tolist()}


# ------------------------------------------------------------------ artifacts


SCHEMA = {
    "file": "data/models/pnn/latent_<h>.parquet, h in {1h, 1d}; one row per walk-forward test step (out of sample)",
    "columns": {
        "ts": "decision time UTC (1h: bar open; 1d: 00:00 UTC of the forecast day D+1)",
        "horizon": "'1h' | '1d'", "segment": "llm_in_sample | clean_holdout (knowledge cutoff 2026-07-01)",
        "refit_month": "YYYY-MM of the walk-forward refit that produced the row (latent coordinates are per refit)",
        "ret": "realised log return over the horizon (label; known only after ts + horizon)",
        "sigma_ref": "point-in-time scale (1h sqrt(robust_bc s2); 1d exp(lrv_w / 2)); quantiles / sigma_ref = model units",
        "z0..z3": "latent (bottleneck) state of the seed-0 PNN of that refit, raw scale",
        "zhat0..zhat3": "predicted next latent A z + B u + c (H-PNN3 dynamics), raw scale",
        "zf0..zf3, zfhat0..zfhat3": "latent / predicted next latent under the LATEST refit network for every row (one fixed coordinate "
                                    "system for plotting; rows before that refit's training cut are in-sample for it)",
        "q05..q95": "PNN 2-seed quantiles of ret (return units)", "har_q05..har_q95": "log-HAR-t quantiles (return units)",
        "aci_lo80/aci_hi80, ..90, ..95": "adaptive conformal band (Gibbs-Candes) around the PNN, return units",
        "sc_lo80/sc_hi80, ..90, ..95": "split-conformal band (calibration = refit validation block), return units",
        "aci_alpha80, aci_alpha90, aci_alpha95": "ACI running miscoverage level alpha_t used for the row",
    },
    "frozen": "data/models/pnn/frozen_<h>.json: latest refit KAN (enc/head coef [out,in,deg] for Chebyshev T1..T4 of tanh(0.5 x), "
              "w_res for silu(x), bias), standardiser mu/sd, clip, feature names, dynamics A/B/c/zmu/zsd, taus",
}


def write_artifacts(h: str, A: dict, C: dict) -> dict:
    import torch

    te, a = A["te"], A["a"]
    sig = te["sig"]
    cols = {"ts": pl.Series("ts", te["T"]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            "horizon": pl.Series("horizon", [h] * len(sig)), "segment": pl.Series("segment", te["seg"]),
            "refit_month": pl.Series("refit_month", te["month"]), "ret": pl.Series("ret", te["r"]), "sigma_ref": pl.Series("sigma_ref", sig)}
    for j in range(te["z"].shape[1]):
        cols[f"z{j}"] = pl.Series(f"z{j}", te["z"][:, j])
        cols[f"zhat{j}"] = pl.Series(f"zhat{j}", te["zhat"][:, j])
    for i, c in enumerate(QCOLS):
        cols[c] = pl.Series(c, te["q_pnn"][:, i] * sig)
        cols[f"har_{c}"] = pl.Series(f"har_{c}", te["q_har"][:, i] * sig)
    for lev in LEVELS:
        L = int(round(lev * 100))
        for nm in ("aci", "sc"):
            cols[f"{nm}_lo{L}"] = pl.Series(f"{nm}_lo{L}", C[lev][f"{nm}_lo"] * sig)
            cols[f"{nm}_hi{L}"] = pl.Series(f"{nm}_hi{L}", C[lev][f"{nm}_hi"] * sig)
        cols[f"aci_alpha{L}"] = pl.Series(f"aci_alpha{L}", C[lev]["aci_alpha"])
    # latest refit network over every test row
    last = sorted(wf_dir(h).glob("*.pt"))[-1]
    ck = torch.load(last, weights_only=False)
    st = Standardizer()
    st.mu, st.sd = np.asarray(ck["std"]["mu"]), np.asarray(ck["std"]["sd"])
    m = pnn.QFit("pnn", ck["pnn"][0], ck["pnn_meta"], 0.0, 0)
    Zt = np.clip(st.transform(a["X"][te["idx"]]), -pnn.CLIP, pnn.CLIP)
    zf = m.latent(Zt)
    dy = ck["dyn"]
    dyn = pnn.LatentDynamics(*(np.asarray(dy[k]) for k in ("A", "B", "c", "zmu", "zsd")))
    zfh = dyn.predict(zf, a["u"][te["idx"]])
    for j in range(zf.shape[1]):
        cols[f"zf{j}"] = pl.Series(f"zf{j}", zf[:, j])
        cols[f"zfhat{j}"] = pl.Series(f"zfhat{j}", zfh[:, j])
    df = pl.DataFrame(list(cols.values())).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    p = root() / f"latent_{h}.parquet"
    df.write_parquet(p)
    st_ = ck["pnn"][0]
    frozen = {"horizon": h, "refit_month": last.stem, "features": list(FEATURES[h]), "std_mu": st.mu.tolist(), "std_sd": st.sd.tolist(),
              "clip": pnn.CLIP, "degree": pnn.DEGREE, "scale": pnn.SCALE, "taus": list(pnn.TAUS), "target_units": "ret / sigma_ref",
              "enc": {k[4:]: v.numpy().tolist() for k, v in st_.items() if k.startswith("enc.")},
              "head": {k[5:]: v.numpy().tolist() for k, v in st_.items() if k.startswith("head.")}, "dynamics": dy,
              "head_rule": "q50=o3; q75=q50+sp(o4); q90=q75+sp(o5); q95=q90+sp(o6); q25=q50-sp(o2); q10=q25-sp(o1); q05=q10-sp(o0)"}
    (root() / f"frozen_{h}.json").write_text(json.dumps(frozen), encoding="utf-8")
    (root() / "schema.json").write_text(json.dumps(SCHEMA, indent=1), encoding="utf-8")
    return {"rows": df.height, "cols": df.width, "bytes": p.stat().st_size, "latest_refit": last.stem}


# ------------------------------------------------------------------ evaluate / report


def run_evaluate(record_rows: bool = False) -> dict:
    E, art, wf = {}, {}, {}
    for h in HORIZONS:
        A = assemble(h)
        C = run_conformal(A)
        E[h] = eval_horizon(A, C)
        art[h] = write_artifacts(h, A, C)
        import torch

        infos = [torch.load(f, weights_only=False)["info"] for f in sorted(wf_dir(h).glob("*.pt"))]
        wf[h] = {"refits": len(infos), "first": infos[0]["month"], "last": infos[-1]["month"],
                 "mean_epochs_pnn": float(np.mean([np.mean(i["epochs_pnn"]) for i in infos])),
                 "mean_epochs_mlp": float(np.mean([np.mean(i["epochs_mlp"]) for i in infos])),
                 "mean_epochs_ctrl": float(np.mean([i["epochs_ctrl"] for i in infos])), "mean_epochs_fb": float(np.mean([i["epochs_fb"] for i in infos])),
                 "cpu_minutes": float(sum(i["secs"] for i in infos) / 60), "n_fit_last": infos[-1]["n_fit"]}
    V = verdicts(E)
    sym = {h: symbolic(h) for h in HORIZONS}
    s = json.loads((root() / "search.json").read_text(encoding="utf-8"))
    out = {"tag": TAG, "as_of": AS_OF.isoformat(), "cutoff": str(_cut())[:10], "search": s, "walk": wf, "eval": E, "verdicts": V,
           "symbolic": sym, "artifacts": art, "schema": SCHEMA,
           "params": {h: {"pnn": s[h]["pnn_params"], "mlp": s[h]["mlp_params"], "mlp_hidden": s[h]["mlp_hidden"],
                          "pnn_fb": pnn.pnn_params(len(FEATURES[h]), fb=True)} for h in HORIZONS}}
    rep = load().root / "research" / "reports"
    (rep / f"btc_pnn_{TAG}.json").write_text(json.dumps(out, indent=1, default=_js), encoding="utf-8")
    from ai_trader.research.pnn_report import write_md

    write_md(json.loads(json.dumps(out, default=_js)), rep / f"btc_pnn_{TAG}.md")
    if record_rows:
        record_all(V)
    return out


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


DATA = (f"BTCUSDT bybit 1h bars (ml.hourly features, market_frame as of {AS_OF:%Y-%m-%d}) and BTC rows of the universe-L daily panel "
        "(ml.features, as of 2026-11-12); monthly expanding walk-forward; code src/ai_trader/ml/pnn.py, src/ai_trader/research/pnn_study.py")


def record_all(V: dict) -> None:
    from ai_trader.research import hypotheses as hy

    for hid in ("H-PNN1", "H-PNN2", "H-PNN3"):
        done = {r["segment"] for r in hy.results(hid)}
        for s in (*SEGS, "pooled"):
            if s in done:
                continue
            row = V[s][hid]
            metrics = json.loads(json.dumps({k: v for k, v in row.items() if k != "verdict"}, default=_js))
            hy.record(hid, s, row["verdict"], metrics, note=f"btc_pnn_{TAG}; Holm within PNN per segment", data=DATA)


def main() -> None:
    ap = argparse.ArgumentParser(prog="pnn_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("search")
    w = sub.add_parser("walk")
    w.add_argument("--h", required=True, choices=HORIZONS)
    w.add_argument("--months", default="")
    e = sub.add_parser("evaluate")
    e.add_argument("--record", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        for h in HORIZONS:
            a = arrays(h)
            print(h, a["n"], str(a["T"].min()), str(a["T"].max()), "first test", str(first_test_month(a))[:7],
                  {s: int((a["seg"] == s).sum()) for s in SEGS})
    elif args.cmd == "search":
        print(json.dumps(run_search(), indent=1, default=_js))
    elif args.cmd == "walk":
        mo = tuple(args.months.split(":")) if args.months else None
        run_walk(args.h, mo)
    else:
        out = run_evaluate(args.record)
        print(json.dumps(out["verdicts"], indent=1, default=_js))


if __name__ == "__main__":
    main()
