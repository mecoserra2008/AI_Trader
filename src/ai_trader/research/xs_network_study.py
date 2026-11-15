"""Cross-sectional network v3 study (H-XS1..H-XS3, block XS, 2026-11-15, agent X3). Research only; nothing goes live.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.xs_network_study build
    ... search                         in-sample config search (training rows of the first test quarter only)
    ... walk --seg llm_in_sample       purged quarterly walk-forward, checkpoint per refit and arm
    ... evaluate --seg llm_in_sample [--record]
    ... walk --seg clean_holdout       (after the in-sample record; same frozen configuration)
    ... evaluate --seg clean_holdout [--record]    holdout looked at once; writes the pooled verdicts
    ... report                         research/reports/xs_network_2026-11-15.{md,json} + figures
    ... freeze                         data/models/xs_v3/ (latest refit: weights, spec, hashes, ACI state)

Design (fixed in the diary entry 'XS v3 design frozen' before any result; see ml.xs_net for the panel and network):

Data: universe L (trend_zoo_study.UNIVERSE, 23 crypto perps), market_frame(sym, '1h', AS_OF, funding, 4h OI), the
  panel of ml.xs_net (features at the open of hour t from bars <= t-1; targets over hours t .. t+h-1). Hours with
  fewer than MIN_TOKENS present tokens are dropped. Segment of an hour = its decision time before / after the cutoff.
Walk-forward: quarterly, expanding. Training hours have t + 24 h <= quarter start - 1 day (purge of the longest label +
  a 1-day embargo). First test quarter = first with >= 365 training days. Validation = the last 15% of training days
  (models.val_split, purged 1 day): early stopping only. Standardisation (mean / sd over present cells of the fit
  block), inputs clipped to +/- 6, missing optional inputs -> 0. Warm start: each refit starts from the previous
  refit's weights of the same arm and seed (the first refit from scratch with the head bias at the unconditional
  quantiles).
Arms (same rows, same protocol): kan_mix (= v3), mlp_mix (equal-size MLP encoder / head), kan_nomix (H-XS3: keys
  restricted to the token itself). Seeds N_SEEDS, quantiles averaged over seeds.
Config search (in-sample, equal effort): lr {1e-3, 3e-3} x weight decay {1e-4, 1e-2} for kan_mix and mlp_mix, seed 0,
  on the first test quarter's training rows; kan_nomix uses the kan_mix config.
Baselines:
  rel (H-XS1)  mom24 = past 24h relative return, mom7d = past 168h relative return, rev1 = minus past 1h relative
               return (raw log units, the same relative definition as the target)
  vol (H-XS2)  panel log-HAR-t x hour-of-week per horizon h: pooled OLS of log max(R_h^2 / (h s2), 1e-6) on
               [1, lsig2, lfac_h, har_d, har_w, har_m] (lfac_1 = lfac), F = h s2 exp(xb + c), c QLIKE-optimal on the
               training rows (models.bias_const); a zero-location Student-t (df, scale by MLE on <= 200k training cells)
               on R_h / sqrt(F); magnitude quantiles a_tau = sqrt(F / (h s2)) scale t^-1((1 + tau) / 2, df).
Tests (per segment; NW t over days):
  H-XS1  origins: 4h -> hours 00, 04, .., 20 UTC; 24h -> 00 UTC (non-overlapping). IC_t = Spearman(model median rel
         return in raw units, realised rel return) over tokens with a target (>= MIN_TOKENS); per-day mean; NW t.
         Best baseline per horizon = highest in-sample mean IC (the same choice is applied in the holdout). Paired
         per-origin difference. Segment supports iff at >= 1 horizon IC > 0 with t > 2 and IC - best > 0 with t > 2;
         contradicts iff at every horizon IC <= 0 or IC - best <= 0; else inconclusive.
  H-XS2  vol pinball (TAUS, units of sqrt(h s2)) model - baseline per token-origin, per-day mean, NW t; h = 1h (every
         hour) and 4h (origins 00, 04, ..). Supports iff at >= 1 horizon diff < 0 with t < -2; contradicts iff at every
         horizon diff >= 0; else inconclusive.
  H-XS3  vol pinball kan_nomix - kan_mix on non-BTC tokens, same origins and horizons as H-XS2. Supports iff at >= 1
         horizon diff > 0 with t > 2; contradicts iff at every horizon diff <= 0; else inconclusive. The IC part
         (kan_mix - kan_nomix rank IC) is reported.
  p-values: H-XS1 min over horizons of max(Phi(-t_ic), Phi(-t_diff)); H-XS2 min Phi(t); H-XS3 min Phi(-t). Holm over
  the three per segment; a 'supports' with Holm p > 0.05 becomes inconclusive. Pooled: supports iff both segments
  support; contradicts iff either contradicts; else inconclusive.
Descriptive: IC by sector and by BTC v0 cell (rank-product decomposition of the IC), attention maps (mean weight from
  query i to key j over test hours; all hours and BTC v0 high-vol hours; 'received' = mean weight x number of keys, 1 =
  uniform), pairwise accuracy / Brier, top-minus-bottom quintile spread (equal weight, 7.5 bp per side per leg on
  turnover), ACI-conformalised 80% rel bands per token (gamma 0.005, window 500 origins, warm start = first refit's
  validation scores), 2D PCA map of the per-token mean representation, KAN vs MLP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import xs_net as xs

TAG = "2026-11-15"
AS_OF = datetime(2026, 11, 15, 10, tzinfo=UTC)
SEGS = ("llm_in_sample", "clean_holdout")
ARMS = ("kan_mix", "mlp_mix", "kan_nomix")
ARM_SPEC = {"kan_mix": ("kan", True), "mlp_mix": ("mlp", True), "kan_nomix": ("kan", False)}
N_SEEDS = 3
MIN_TRAIN_DAYS = 365
CLIP = 6.0
GRID = [{"lr": lr, "wd": wd} for lr in (1e-3, 3e-3) for wd in (1e-4, 1e-2)]
HOUR_US = 3600 * 1_000_000
DAY_US = 24 * HOUR_US
COST_BP = 7.5
ACI_GAMMA = 0.005
ACI_WINDOW = 500
REL_LEVEL = 0.80


def root() -> Path:
    p = load().root / "data" / "models" / "xs_v3"
    p.mkdir(parents=True, exist_ok=True)
    return p


def universe() -> list[str]:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    return list(UNIVERSE)


def sectors(symbols: list[str]) -> list[str]:
    import yaml

    y = yaml.safe_load((load().root / "config" / "sectors.yaml").read_text(encoding="utf-8"))
    base, ov = y.get("sectors", {}), y.get("overrides", {})
    return [ov.get(s, base.get(s, "other")) for s in symbols]


def cut_us() -> int:
    from ai_trader.research.evaluate import knowledge_cutoff

    return int(knowledge_cutoff().timestamp() * 1_000_000)


# ------------------------------------------------------------------ build


def _frame(sym: str) -> pl.DataFrame:
    p = root() / "cache" / f"f1h_{sym}.parquet"
    if p.is_file():
        return pl.read_parquet(p)
    from ai_trader.research.frame import market_frame

    f = market_frame(sym, "1h", AS_OF, funding=True, oi=True, oi_interval="4h")
    f = f.select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "ret", "sigma_risk_next", "funding_rate", "open_interest")
    p.parent.mkdir(parents=True, exist_ok=True)
    f.write_parquet(p)
    return f


def token_rows(f: pl.DataFrame) -> pl.DataFrame:
    """One symbol on its own regular hourly grid: HOURLY_FEATURES (row t uses bars <= t-1), s2, lfac4 / lfac24, log
    close of bar t, funding / OI as known at the close of bar t (lagged later)."""
    from ai_trader.features import seasonality as sz
    from ai_trader.ml.hourly import _shift, symbol_hourly

    h = symbol_hourly(f)
    f = f.sort("event_ts").unique("event_ts", keep="last", maintain_order=True)
    grid = h.select("event_ts")
    g = grid.join(f.select("event_ts", "close", "ret", "sigma_risk_next", "funding_rate", "open_interest"), on="event_ts", how="left")
    prev_ok = g["ret"].is_not_null().to_numpy() & np.r_[False, g["sigma_risk_next"].is_not_null().to_numpy()[:-1]]
    r = np.where(prev_ok, g["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(), np.nan)
    s_pre = _shift(g["sigma_risk_next"].cast(pl.Float64).fill_null(np.nan).to_numpy(), 1)
    with np.errstate(all="ignore"):
        u = r * r / (s_pre * s_pre)
    sm = xs.seasonal_multi(sz.hour_of_week(g["event_ts"]), u, (4, 24))
    with np.errstate(all="ignore"):
        extra = {"lfac4": np.log(sm[4]), "lfac24": np.log(sm[24]), "logc": np.log(g["close"].cast(pl.Float64).fill_null(np.nan).to_numpy()),
                 "fund_raw": g["funding_rate"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                 "oi_raw": g["open_interest"].cast(pl.Float64).fill_null(np.nan).to_numpy()}
    return h.with_columns(*[pl.Series(k, np.where(np.isfinite(v), v, np.nan)) for k, v in extra.items()]).with_columns(pl.col(pl.Float64).fill_nan(None))


def btc_cells(ts_us: np.ndarray) -> np.ndarray:
    """BTC v0 cell (4h labeller) known at each hour's decision time (latest 4h bar with close_ts <= t)."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label

    f4 = market_frame("BTCUSDT", "4h", AS_OF, funding=False, oi=False)
    lab = label(f4.sort("event_ts"), bar_hours=4.0).select("close_ts", "cell").drop_nulls()
    c_us = lab["close_ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).cast(pl.Int64).to_numpy()
    cells = lab["cell"].to_numpy()
    j = np.searchsorted(c_us, ts_us, side="right") - 1
    return np.where(j >= 0, cells[np.maximum(j, 0)], None).astype(object)


def assemble(rows: dict[str, pl.DataFrame], symbols: list[str]) -> dict:
    """Dense (T, N, .) arrays on the global hourly grid. Pure function of the per-token rows (tested for look-ahead)."""
    from ai_trader.ml.hourly import HOURLY_FEATURES

    lo = min(r["event_ts"].min() for r in rows.values())
    hi = max(r["event_ts"].max() for r in rows.values())
    G = pl.DataFrame({"event_ts": pl.datetime_range(lo, hi, "1h", eager=True)}).with_columns(pl.col("event_ts").cast(next(iter(rows.values()))["event_ts"].dtype))
    T, N = G.height, len(symbols)
    cols = (*HOURLY_FEATURES, "s2", "lfac4", "lfac24", "logc", "fund_raw", "oi_raw")
    A = {c: np.full((T, N), np.nan) for c in cols}
    for j, s in enumerate(symbols):
        if s not in rows:
            continue
        m = G.join(rows[s].select("event_ts", *cols), on="event_ts", how="left")
        for c in cols:
            A[c][:, j] = m[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
    ts = G["event_ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).cast(pl.Int64).to_numpy()
    s2 = A["s2"]
    with np.errstate(all="ignore"):
        s = np.sqrt(np.where(s2 > 0, s2, np.nan))
    logc = A["logc"]
    feats = {c: A[c] for c in HOURLY_FEATURES}
    feats["lfac4"], feats["lfac24"] = A["lfac4"], A["lfac24"]
    hr = G["event_ts"].dt.hour().to_numpy().astype(float)
    feats["hod_sin"] = np.broadcast_to(np.sin(2 * np.pi * hr / 24)[:, None], (T, N)).copy()
    feats["hod_cos"] = np.broadcast_to(np.cos(2 * np.pi * hr / 24)[:, None], (T, N)).copy()
    back = {h: xs.back_ret(logc, h) for h in (1, 4, 24, 168)}
    relb = {h: xs.relative(back[h], np.isfinite(back[h]), min_n=3) for h in (1, 4, 24, 168)}
    with np.errstate(all="ignore"):
        for h in (1, 4, 24, 168):
            feats[f"ret{h}_z"] = back[h] / (np.sqrt(h) * s)
            feats[f"rel{h}_z"] = relb[h] / (np.sqrt(h) * s)
    r1 = np.full((T, N), np.nan)
    r1[1:] = logc[1:] - logc[:-1]  # bar t's own return (used only through windows ending at t-1)
    fin = np.isfinite(r1)
    with np.errstate(all="ignore"):
        r_ew = np.where(fin.sum(1) >= 3, np.nansum(r1, axis=1) / fin.sum(1), np.nan)
    btc = symbols.index("BTCUSDT")
    feats["beta_btc"] = xs.rolling_beta(r1, r1[:, btc])
    feats["beta_ew"] = xs.rolling_beta(r1, r_ew)
    fund = np.full((T, N), np.nan)
    fund[1:] = A["fund_raw"][:-1]
    oi = np.full((T, N), np.nan)
    oi[1:] = A["oi_raw"][:-1]
    with np.errstate(all="ignore"):
        loi = np.log(np.where(oi > 0, oi, np.nan))
        feats["fund_bp"] = np.clip(fund * 1e4, -30, 30)
        feats["fund_z90"] = np.clip(xs.lagged_rolling_z(fund), -10, 10)
        o4 = np.full((T, N), np.nan)
        o24 = np.full((T, N), np.nan)
        o4[4:] = loi[4:] - loi[:-4]
        o24[24:] = loi[24:] - loi[:-24]
    feats["oi4"], feats["oi24"] = np.clip(o4, -1, 1), np.clip(o24, -1, 1)
    feats["oi_ok"] = (np.isfinite(o4) & np.isfinite(o24)).astype(float)
    names = xs.feature_names()
    del A, r1, fin, fund, oi, loi, o4, o24
    lu1, lsig2 = feats["lu_1"].astype(np.float32), feats["lsig2"].astype(np.float32)
    X = np.empty((T, N, len(names)), dtype=np.float32)
    for i, c in enumerate(names):
        X[..., i] = feats.pop(c)
    del feats
    core = [i for i, c in enumerate(names) if c not in xs.OPTIONAL]
    present = np.all(np.isfinite(X[..., core]), axis=-1) & np.isfinite(s) & (s > 0)
    npres = present.sum(1)
    # market token (present tokens)
    with np.errstate(all="ignore"):
        cnt = np.maximum(npres, 1)
        ms = np.where(present, s, 0).sum(1) / cnt
        mk = {}
        for h in (1, 4, 24, 168):
            mk[f"mkt_r{h}_z"] = np.where(present, back[h], 0).sum(1) / cnt / (np.sqrt(h) * ms)
        mk["mkt_lu1"] = np.where(present, lu1, 0).sum(1) / cnt
        mk["mkt_lsig2"] = np.where(present, lsig2, 0).sum(1) / cnt
        b24 = np.where(present, back[24], np.nan)
        mk["mkt_disp24"] = np.log(np.nanstd(b24, axis=1) / (np.sqrt(24) * ms))
        mk["n_frac"] = npres / N
    M = np.stack([mk[c] for c in xs.MKT_FEATURES], axis=-1)
    valid = (npres >= xs.MIN_TOKENS) & np.all(np.isfinite(M), axis=1)
    present &= valid[:, None]
    X[~present] = np.nan
    # targets
    Rf = {h: xs.fwd_ret(logc, h) for h in sorted(set(xs.VOL_H) | set(xs.REL_H))}
    with np.errstate(all="ignore"):
        Av = np.stack([np.where(present, np.abs(Rf[h]) / (np.sqrt(h) * s), np.nan) for h in xs.VOL_H], axis=-1)
        rel_raw = np.stack([xs.relative(Rf[h], present) for h in xs.REL_H], axis=-1)
        Yr = np.stack([rel_raw[..., k] / (np.sqrt(h) * s) for k, h in enumerate(xs.REL_H)], axis=-1)
    # baselines' inputs (raw relative past returns, same relative definition over present tokens)
    base = np.stack([xs.relative(back[h], present) for h in (1, 24, 168)], axis=-1)
    return {"ts": ts, "symbols": np.array(symbols), "features": np.array(names), "X": X, "M": M.astype(np.float32),
            "present": present, "valid": valid, "s2": s2.astype(np.float64), "Av": Av.astype(np.float32), "Yr": Yr.astype(np.float32),
            "rel_raw": rel_raw.astype(np.float32), "Rf_vol": np.stack([Rf[h] for h in xs.VOL_H], -1).astype(np.float32), "base_rel": base.astype(np.float32)}


def panel_path() -> Path:
    return root() / "cache" / f"panel_{TAG}.npz"


def build() -> dict:
    p = panel_path()
    if p.is_file():
        return load_panel()
    syms = universe()
    rows = {}
    for s in syms:
        t0 = time.time()
        f = _frame(s)
        if f.height < 24 * 120:
            print(s, "skipped (short)", flush=True)
            continue
        rows[s] = token_rows(f)
        print(s, f.height, f["event_ts"].min(), round(time.time() - t0, 1), flush=True)
    P = assemble(rows, syms)
    P["sector"] = np.array(sectors(syms))
    P["cell"] = btc_cells(P["ts"]).astype(str)
    keep = P["valid"]
    first = int(np.argmax(keep))
    P = {k: (v[first:] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(P["ts"]) else v) for k, v in P.items()}
    np.savez(p, **P)
    meta = {"as_of": AS_OF.isoformat(), "symbols": syms, "sectors": P["sector"].tolist(), "features": P["features"].tolist(),
            "hours": len(P["ts"]), "valid_hours": int(P["valid"].sum()), "first": str(np.datetime64(int(P["ts"][0]), "us")),
            "present_cells": int(P["present"].sum()),
            "first_present": {s: str(np.datetime64(int(P["ts"][int(np.argmax(P["present"][:, j]))]), "us")) if P["present"][:, j].any() else None
                              for j, s in enumerate(syms)}}
    (root() / "cache" / f"panel_{TAG}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return load_panel()


def load_panel() -> dict:
    d = np.load(panel_path(), allow_pickle=False)
    return {k: d[k] for k in d.files}


# ------------------------------------------------------------------ blocks, standardisation, quarters


def sec_idx(P: dict) -> np.ndarray:
    return np.array([xs.SECTOR_ORDER.index(s) for s in P["sector"]], dtype=np.int64)


def quarter_starts(P: dict) -> list[int]:
    t0 = np.datetime64(int(P["ts"][0]), "us").astype("datetime64[M]")
    t1 = np.datetime64(int(P["ts"][-1]), "us").astype("datetime64[M]")
    out = []
    for m in np.arange(t0, t1 + np.timedelta64(1, "M")):
        if (m.astype(int) % 12) % 3 == 0:
            out.append(int(np.datetime64(m, "us").astype(np.int64)))
    return out


def qtag(q_us: int) -> str:
    d = np.datetime64(q_us, "us").astype(datetime)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def train_idx(P: dict, q_us: int) -> np.ndarray:
    """Valid hours whose longest label (24 h) ends at least the 1-day embargo before the quarter start."""
    return np.flatnonzero(P["valid"] & (P["ts"] + 24 * HOUR_US <= q_us - DAY_US))


def test_idx(P: dict, q_us: int, q_next: int) -> np.ndarray:
    return np.flatnonzero(P["valid"] & (P["ts"] >= q_us) & (P["ts"] < q_next))


def first_test_quarter(P: dict) -> int:
    for q in quarter_starts(P):
        tr = train_idx(P, q)
        if len(np.unique(P["ts"][tr] // DAY_US)) >= MIN_TRAIN_DAYS:
            return q
    raise RuntimeError("no quarter with enough history")


def quarters(P: dict, seg: str) -> list[tuple[int, int]]:
    qs = quarter_starts(P)
    q0 = first_test_quarter(P)
    nxt = {a: b for a, b in zip(qs, [*qs[1:], int(P["ts"][-1]) + HOUR_US], strict=True)}
    c = cut_us()
    keep = [q for q in qs if q >= q0 and ((q < c) if seg == "llm_in_sample" else (q >= c))]
    return [(q, nxt[q]) for q in keep]


def split_fit_val(P: dict, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from ai_trader.ml import models as md

    fl, vl = md.val_split(P["ts"][tr].astype("datetime64[us]"))
    return tr[fl], tr[vl]


def fit_std(P: dict, fit: np.ndarray) -> dict:
    F = P["X"].shape[2]
    mu, sd = np.zeros(F), np.ones(F)
    for f in range(F):
        v = P["X"][fit, :, f]
        v = v[np.isfinite(v)].astype(float)
        mu[f] = v.mean()
        s = v.std()
        sd[f] = s if s > 1e-12 else 1.0
    Mf = P["M"][fit].astype(float)
    mmu, msd = Mf.mean(0), Mf.std(0)
    return {"mu": mu, "sd": sd, "mmu": mmu, "msd": np.where(msd > 1e-12, msd, 1.0)}


def block(P: dict, idx: np.ndarray, st: dict) -> xs.Batchable:
    Z = P["X"][idx]  # a copy (fancy index); standardised in place to keep peak memory low
    Z -= st["mu"].astype(np.float32)
    Z /= st["sd"].astype(np.float32)
    np.clip(Z, -CLIP, CLIP, out=Z)
    np.nan_to_num(Z, copy=False, nan=0.0)
    M = np.clip((P["M"][idx] - st["mmu"]) / st["msd"], -CLIP, CLIP).astype(np.float32)
    pres = P["present"][idx]
    Av, Yr = P["Av"][idx], P["Yr"][idx]
    Mv = np.isfinite(Av) & pres[..., None]
    Mr = np.isfinite(Yr) & pres[..., None]
    return xs.Batchable(Z, M, pres, np.where(Mv, Av, 0).astype(np.float32), Mv, np.where(Mr, Yr, 0).astype(np.float32), Mr)


def new_model(P: dict, arm: str):
    kind, mix = ARM_SPEC[arm]
    return xs.make(P["X"].shape[2], P["M"].shape[1], len(xs.SECTOR_ORDER), kind=kind, mix=mix)


def init_model_bias(model, fitb: xs.Batchable) -> None:
    a = [fitb.Yv[..., k][fitb.Mv[..., k]] for k in range(len(xs.VOL_H))]
    y = [fitb.Yr[..., k][fitb.Mr[..., k]] for k in range(len(xs.REL_H))]
    xs.init_bias(model, a, y)


# ------------------------------------------------------------------ panel log-HAR-t x hour-of-week (H-XS2 baseline)


HAR_COLS = {1: "lfac", 4: "lfac4", 24: "lfac24"}


def _har_design(P: dict, idx: np.ndarray, h: int) -> np.ndarray:
    names = list(P["features"])
    cols = ["lsig2", HAR_COLS[h], "har_d", "har_w", "har_m"]
    Xr = P["X"][:, :, [names.index(c) for c in cols]][idx].astype(float)
    return np.concatenate([np.ones((*Xr.shape[:-1], 1)), Xr], axis=-1)


def fit_har_t(P: dict, tr: np.ndarray, seed: int = 0) -> dict:
    from scipy import stats

    from ai_trader.ml import models as md

    out = {}
    rng = np.random.default_rng(seed)
    for k, h in enumerate(xs.VOL_H):
        Z = _har_design(P, tr, h)
        a = P["Av"][tr][..., k].astype(float)
        ok = P["present"][tr] & np.isfinite(a) & np.all(np.isfinite(Z), -1)
        Zc, ac = Z[ok], a[ok]
        y = np.log(np.maximum(ac * ac, 1e-6))
        b, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        lf = Zc @ b
        c = md.bias_const(ac * ac, lf)
        z = ac / np.exp(0.5 * (lf + c))
        sub = rng.choice(len(z), size=min(200_000, len(z)), replace=False)
        df, _, scale = stats.t.fit(z[sub], floc=0.0)
        out[h] = {"b": b.tolist(), "c": float(c), "df": float(df), "scale": float(scale), "n": len(z)}
    return out


def har_quantiles(P: dict, idx: np.ndarray, har: dict) -> np.ndarray:
    """(n, N, 3, 7) magnitude quantiles in units of sqrt(h s2)."""
    from scipy import stats

    out = np.full((len(idx), P["X"].shape[1], len(xs.VOL_H), len(xs.TAUS)), np.nan, dtype=np.float32)
    for k, h in enumerate(xs.VOL_H):
        p = har[h]
        Z = _har_design(P, idx, h)
        sd = np.exp(0.5 * (Z @ np.asarray(p["b"]) + p["c"]))
        tq = p["scale"] * stats.t.ppf((1 + np.asarray(xs.TAUS)) / 2, p["df"])
        out[..., k, :] = (sd[..., None] * tq).astype(np.float32)
    return out


# ------------------------------------------------------------------ search


def search() -> dict:
    P = build()
    q0 = first_test_quarter(P)
    tr = train_idx(P, q0)
    # 80 / 20 purged split of the first test quarter's training rows (as the BTC PNN search)
    from ai_trader.ml import models as md

    fl, vl = md.val_split(P["ts"][tr].astype("datetime64[us]"), 0.2)
    fi, vi = tr[fl], tr[vl]
    st = fit_std(P, fi)
    fb, vb = block(P, fi, st), block(P, vi, st)
    sec = sec_idx(P)
    res = {}
    for arm in ("kan_mix", "mlp_mix"):
        res[arm] = []
        for cfg in GRID:
            t0 = time.time()
            m = new_model(P, arm)
            init_model_bias(m, fb)
            r = xs.train(m, fb, vb, sec, seed=0, lr=cfg["lr"], wd=cfg["wd"])
            res[arm].append({"cfg": cfg, "val": r["best_val"], "epoch": r["epoch"], "epochs_run": r["epochs_run"], "secs": round(time.time() - t0, 1)})
            print(arm, res[arm][-1], flush=True)
    out = {"first_test_quarter": qtag(q0), "n_fit_hours": len(fi), "n_val_hours": len(vi),
           "params": {a: xs.n_params(new_model(P, a)) for a in ARMS}, "scores": res,
           "best": {a: min(res[a], key=lambda r: r["val"])["cfg"] for a in res}}
    out["best"]["kan_nomix"] = out["best"]["kan_mix"]
    (root() / "search.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def cfgs() -> dict:
    return json.loads((root() / "search.json").read_text(encoding="utf-8"))["best"]


# ------------------------------------------------------------------ walk-forward


def wf_dir() -> Path:
    p = root() / "wf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def representation(model, B: xs.Batchable, sec: np.ndarray) -> np.ndarray:
    """Mean head input [z_i, a_i] per token over the block's present hours: (N, 2D) (NaN if never present)."""
    torch = xs._torch()
    S = torch.as_tensor(sec, dtype=torch.long)
    N = B.P.shape[1]
    acc, cnt = np.zeros((N, 2 * xs.D)), np.zeros(N)
    with torch.no_grad():
        for i in range(0, len(B.P), 512):
            sl = slice(i, i + 512)
            Z, M, Pm = (torch.as_tensor(np.ascontiguousarray(a[sl]), dtype=dt) for a, dt in ((B.Z, torch.float32), (B.M, torch.float32), (B.P, torch.bool)))
            z, h = model.encode(Z, S)
            a, _ = model.attend(h, M, Pm)
            r = torch.cat([z, a], -1).numpy()
            pm = B.P[sl]
            acc += (r * pm[..., None]).sum(0)
            cnt += pm.sum(0)
    with np.errstate(all="ignore"):
        return acc / cnt[:, None]


def run_walk(seg: str) -> None:
    import torch

    torch.set_num_threads(xs.TORCH_THREADS)
    P = build()
    cf = cfgs()
    sec = sec_idx(P)
    allq = quarters(P, "llm_in_sample") + quarters(P, "clean_holdout")
    for q, qn in quarters(P, seg):
        tag = qtag(q)
        f_npz, f_pt = wf_dir() / f"{tag}.npz", wf_dir() / f"{tag}.pt"
        if f_npz.is_file() and f_pt.is_file():
            continue
        t0 = time.time()
        tr = train_idx(P, q)
        fi, vi = split_fit_val(P, tr)
        te = test_idx(P, q, qn)
        st = fit_std(P, fi)
        fb, vb, tb = block(P, fi, st), block(P, vi, st), block(P, te, st)
        k = [x[0] for x in allq].index(q)
        prev = wf_dir() / f"{qtag(allq[k - 1][0])}.pt" if k > 0 else None
        prev_states = torch.load(prev, weights_only=False)["states"] if prev is not None and prev.is_file() else None
        save = {"idx_val": vi, "idx_test": te}
        states, info = {}, {}
        for arm in ARMS:
            c = cf[arm]
            vq, rq, wq, rv_val, reps = [], [], [], [], []
            states[arm], info[arm] = [], []
            for s in range(N_SEEDS):
                torch.manual_seed(1000 * s + 7)
                m = new_model(P, arm)
                if prev_states is not None:
                    m.load_state_dict(prev_states[arm][s])
                else:
                    init_model_bias(m, fb)
                r = xs.train(m, fb, vb, sec, seed=1000 * s + k, lr=c["lr"], wd=c["wd"])
                out = xs.predict(m, tb, sec, attn=ARM_SPEC[arm][1])
                vq.append(out[0])
                rq.append(out[1])
                if ARM_SPEC[arm][1]:
                    wq.append(out[2].astype(np.float32))
                if arm == "kan_mix":
                    rv_val.append(xs.predict(m, vb, sec)[1])
                    reps.append(representation(m, tb, sec))
                states[arm].append(copy_state(m))
                info[arm].append({k2: v for k2, v in r.items() if k2 != "hist"} | {"hist": [round(x, 5) for x in r["hist"]]})
            save[f"vol_{arm}"] = np.mean(vq, 0).astype(np.float32)
            save[f"rel_{arm}"] = np.mean(rq, 0).astype(np.float32)
            if arm == "kan_mix":
                save["vol_kan_mix_seeds"] = np.stack(vq).astype(np.float32)
                save["rel_kan_mix_seeds"] = np.stack(rq).astype(np.float32)
                save["relval_kan_mix"] = np.mean(rv_val, 0).astype(np.float32)
                save["rep_kan_mix"] = np.mean(reps, 0)
            if wq:
                save[f"attn_{arm}"] = np.mean(wq, 0).astype(np.float16)
            print(json.dumps({"q": tag, "arm": arm, "epochs": [i["epoch"] for i in info[arm]], "val": [round(i["best_val"], 5) for i in info[arm]],
                              "secs": round(time.time() - t0, 1)}), flush=True)
        har = fit_har_t(P, tr)
        save["vol_har"] = har_quantiles(P, te, har)
        np.savez_compressed(f_npz, **save)
        torch.save({"states": states, "std": {k2: v.tolist() for k2, v in st.items()}, "har": har, "cfg": cf,
                    "info": info | {"quarter": tag, "n_fit": len(fi), "n_val": len(vi), "n_test": len(te),
                                    "fit_range": [int(P["ts"][fi[0]]), int(P["ts"][fi[-1]])], "secs": round(time.time() - t0, 1)}}, f_pt)
        print(json.dumps({"q": tag, "done": True, "secs": round(time.time() - t0, 1)}), flush=True)
        del fb, vb, tb


def copy_state(m) -> dict:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def timing() -> None:
    """One short fit per arm on the first test quarter's rows: seconds per epoch (no metrics are inspected)."""
    import torch

    torch.set_num_threads(xs.TORCH_THREADS)
    P = build()
    q0 = first_test_quarter(P)
    fi, vi = split_fit_val(P, train_idx(P, q0))
    st = fit_std(P, fi)
    fb, vb = block(P, fi, st), block(P, vi, st)
    sec = sec_idx(P)
    for arm in ARMS:
        m = new_model(P, arm)
        init_model_bias(m, fb)
        t0 = time.time()
        r = xs.train(m, fb, vb, sec, seed=0, lr=3e-3, wd=1e-4, max_epochs=2, patience=99)
        print(arm, "params", xs.n_params(m), "secs/epoch", round((time.time() - t0) / r["epochs_run"], 1), "fit hours", len(fi), flush=True)


# ------------------------------------------------------------------ evaluation helpers


def collect(P: dict, segs=SEGS, seeds: bool = False) -> dict:
    """Concatenated test rows of every available refit in `segs` (time ordered) + the first refit's validation block."""
    files = []
    for s in segs:
        files += [(q, wf_dir() / f"{qtag(q)}.npz") for q, _ in quarters(P, s)]
    files = [(q, f) for q, f in files if f.is_file()]
    if not files:
        raise RuntimeError(f"no walk-forward checkpoints for {segs}")
    keys = ["idx_test", *[f"vol_{a}" for a in ARMS], *[f"rel_{a}" for a in ARMS], "vol_har", "attn_kan_mix", "attn_mlp_mix"]
    if seeds:
        keys += ["rel_kan_mix_seeds"]
    parts: dict[str, list] = {k: [] for k in keys}
    reps, qt = [], []
    for q, f in files:
        d = np.load(f)
        for k in keys:
            parts[k].append(d[k])
        reps.append(d["rep_kan_mix"])
        qt.append(np.full(len(d["idx_test"]), qtag(q)))
    first = np.load(files[0][1])
    out = {k: np.concatenate(v, axis=1 if k.endswith("_seeds") else 0) for k, v in parts.items()}
    out["idx"] = out.pop("idx_test")
    out["quarter"] = np.concatenate(qt)
    out["reps"] = np.stack(reps)
    out["rep_quarters"] = [qtag(q) for q, _ in files]
    out["val_idx"] = first["idx_val"]
    out["val_rel"] = first["relval_kan_mix"]
    return out


def seg_of(P: dict, idx: np.ndarray) -> np.ndarray:
    return np.where(P["ts"][idx] < cut_us(), "llm_in_sample", "clean_holdout")


def origin_mask(P: dict, idx: np.ndarray, h: int) -> np.ndarray:
    hr = (P["ts"][idx] // HOUR_US) % 24
    return np.ones(len(idx), bool) if h == 1 else (hr % 4 == 0 if h == 4 else hr == 0)


def pinball_cells(y: np.ndarray, q: np.ndarray, taus) -> np.ndarray:
    t = np.asarray(taus, dtype=np.float32)
    e = y[..., None] - q
    return np.maximum(t * e, (t - 1) * e).mean(-1)


def per_day(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Mean of finite x per day (x and day of equal shape)."""
    x, day = np.ravel(x), np.ravel(day)
    ok = np.isfinite(x)
    if not ok.any():
        return np.array([])
    _, inv = np.unique(day[ok], return_inverse=True)
    s = np.bincount(inv, weights=x[ok])
    n = np.bincount(inv)
    return s / n


def nw(daily: np.ndarray) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    m, t, n = newey_west_t(np.asarray(daily, dtype=float))
    return {"mean": float(m), "t": float(t), "days": int(n)}


def _phi(t: float) -> float:
    return 0.5 * (1 + math.erf(t / math.sqrt(2))) if np.isfinite(t) else 1.0


def raw_rel_quantiles(P: dict, idx: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Standardised rel quantiles (n, N, 2, 3) -> raw log-return units."""
    s = np.sqrt(P["s2"][idx])
    sc = np.stack([np.sqrt(h) * s for h in xs.REL_H], axis=-1)
    return rel * sc[..., None]


BASELINES = {"mom24": (1, 1.0), "mom7d": (2, 1.0), "rev1": (0, -1.0)}  # base_rel column, sign


def spread_series(score: np.ndarray, real: np.ndarray, mask: np.ndarray) -> dict:
    """Top-minus-bottom quintile (equal weight) at consecutive non-overlapping origins: gross, turnover, net."""
    T, N = score.shape
    wl_prev, ws_prev = np.zeros(N), np.zeros(N)
    gross, to, net = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    for t in range(T):
        k = mask[t] & np.isfinite(score[t]) & np.isfinite(real[t])
        n = int(k.sum())
        if n < xs.MIN_TOKENS:
            continue
        q = max(1, n // 5)
        ids = np.flatnonzero(k)
        o = ids[np.argsort(score[t, ids], kind="stable")]
        lo_, hi_ = o[:q], o[-q:]
        wl, ws = np.zeros(N), np.zeros(N)
        wl[hi_], ws[lo_] = 1.0 / q, 1.0 / q
        tt = np.abs(wl - wl_prev).sum() + np.abs(ws - ws_prev).sum()
        g = real[t, hi_].mean() - real[t, lo_].mean()
        gross[t], to[t], net[t] = g, tt, g - COST_BP * 1e-4 * tt
        wl_prev, ws_prev = wl, ws
    return {"gross": gross, "turnover": to, "net": net}


def spread_summary(sp: dict, day: np.ndarray, h: int) -> dict:
    ok = np.isfinite(sp["net"])
    per_year = 365 * 24 / h
    n, g = sp["net"][ok], sp["gross"][ok]
    sd = n.std()
    return {"origins": int(ok.sum()), "gross_bp": float(g.mean() * 1e4), "net_bp": float(n.mean() * 1e4),
            "turnover_per_rebalance": float(np.nanmean(sp["turnover"][ok]) / 2), "sharpe_net_ann": float(n.mean() / sd * math.sqrt(per_year)) if sd > 0 else float("nan"),
            "sharpe_gross_ann": float(g.mean() / g.std() * math.sqrt(per_year)) if g.std() > 0 else float("nan"),
            "net_t_daily": nw(per_day(sp["net"], day))["t"]}


def attention_stats(P: dict, idx: np.ndarray, W: np.ndarray, rows: np.ndarray) -> dict:
    """Mean attention from query i to key j (N x (N+1)) over hours `rows` where both are present; received ratio per
    key (weight x number of keys; 1 = uniform), self share, market-token share; per-day received ratios."""
    pres = P["present"][idx][rows]
    w = W[rows].astype(np.float32)
    T, N, K = w.shape
    keyp = np.concatenate([pres, np.ones((T, 1), bool)], 1)
    both = pres[:, :, None] & keyp[:, None, :]
    w = np.where(both, w, 0.0)
    cnt = both.sum(0)
    with np.errstate(all="ignore"):
        mean_w = w.sum(0) / cnt
    nk = keyp.sum(1).astype(np.float32)
    ratio = w * nk[:, None, None]
    # received ratio per key: mean over (hour, query) pairs with both present
    with np.errstate(all="ignore"):
        recv = ratio.sum((0, 1)) / cnt.sum(0)
    # received from others (excluding self-attention)
    eye = np.zeros((N, K), bool)
    eye[np.arange(N), np.arange(N)] = True
    ratio_o = np.where(eye[None], 0.0, ratio)
    cnt_o = np.where(eye, 0, cnt)
    with np.errstate(all="ignore"):
        recv_o = ratio_o.sum((0, 1)) / cnt_o.sum(0)
    self_share = float(np.einsum("tii->", w[:, :, :N]) / pres.sum())
    mkt_share = float(w[:, :, N].sum() / pres.sum())
    day = (P["ts"][idx][rows] // DAY_US)
    # per-day received-from-others ratio per key (for hub tests)
    ud, inv = np.unique(day, return_inverse=True)
    num = np.zeros((len(ud), K))
    den = np.zeros((len(ud), K))
    np.add.at(num, inv, ratio_o.sum(1))
    np.add.at(den, inv, (both & ~eye[None]).sum(1))
    with np.errstate(all="ignore"):
        daily = num / den
    return {"mean_w": mean_w, "recv": recv, "recv_others": recv_o, "self_share": self_share, "mkt_share": mkt_share, "daily_recv_others": daily, "hours": int(T)}


def hub_test(daily: np.ndarray, j: int, N: int) -> dict:
    others = [k for k in range(N) if k != j]
    with np.errstate(all="ignore"):
        d = daily[:, j] - np.nanmean(daily[:, others], axis=1)
    return nw(d[np.isfinite(d)])


# ------------------------------------------------------------------ evaluate


def evaluate(seg: str, record: bool = False) -> dict:
    P = build()
    segs = ("llm_in_sample",) if seg == "llm_in_sample" else SEGS
    C = collect(P, segs)
    idx_all = C["idx"]
    sg = seg_of(P, idx_all)
    syms = list(P["symbols"])
    N = len(syms)
    btc = syms.index("BTCUSDT")
    res: dict = {"segment": seg}
    # ---- ACI on the whole available stream (causal; the in-sample part is identical in both runs)
    aci = run_aci(P, C)
    k = sg == seg
    idx = idx_all[k]
    sub = {kk: (v[k] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(idx_all) else v) for kk, v in C.items()}
    day = P["ts"][idx] // DAY_US
    pres = P["present"][idx]
    res["hours"] = len(idx)
    res["days"] = len(np.unique(day))
    res["quarters"] = sorted(set(sub["quarter"].tolist()))
    # ---- vol pinball (H-XS2, H-XS3)
    Av = P["Av"][idx]
    pin = {m: pinball_cells(Av, sub[f"vol_{m}"], xs.TAUS) for m in (*ARMS, "har")}
    vol = {}
    for kk, h in enumerate(xs.VOL_H):
        om = origin_mask(P, idx, h)
        ok = pres & np.isfinite(Av[..., kk]) & om[:, None]
        dd = np.broadcast_to(day[:, None], ok.shape)
        r = {"cells": int(ok.sum()), "pinball": {m: float(pin[m][..., kk][ok].mean()) for m in pin}}
        r["v3_minus_har"] = nw(per_day(np.where(ok, pin["kan_mix"][..., kk] - pin["har"][..., kk], np.nan), dd))
        r["mlp_minus_har"] = nw(per_day(np.where(ok, pin["mlp_mix"][..., kk] - pin["har"][..., kk], np.nan), dd))
        r["kan_minus_mlp"] = nw(per_day(np.where(ok, pin["kan_mix"][..., kk] - pin["mlp_mix"][..., kk], np.nan), dd))
        okn = ok.copy()
        okn[:, btc] = False
        r["nomix_minus_mix_nonbtc"] = nw(per_day(np.where(okn, pin["kan_nomix"][..., kk] - pin["kan_mix"][..., kk], np.nan), dd))
        r["nomix_minus_mix_all"] = nw(per_day(np.where(ok, pin["kan_nomix"][..., kk] - pin["kan_mix"][..., kk], np.nan), dd))
        r["gain_vs_har_pct"] = 100 * (1 - r["pinball"]["kan_mix"] / r["pinball"]["har"])
        r["per_token_gain_vs_har_pct"] = {syms[j]: float(100 * (1 - pin["kan_mix"][..., kk][ok[:, j], j].mean() / pin["har"][..., kk][ok[:, j], j].mean()))
                                          for j in range(N) if ok[:, j].sum() > 100}
        r["per_token_mix_gain_pct"] = {syms[j]: float(100 * (pin["kan_nomix"][..., kk][ok[:, j], j].mean() / pin["kan_mix"][..., kk][ok[:, j], j].mean() - 1))
                                       for j in range(N) if ok[:, j].sum() > 100}
        r["hit_rate"] = {m: np.mean(Av[..., kk][ok][:, None] <= sub[f"vol_{m}"][..., kk, :][ok], axis=0).tolist() for m in ("kan_mix", "har")}
        vol[f"{h}h"] = r
    res["vol"] = vol
    # ---- relative returns (H-XS1)
    rel = {}
    best_in = None
    if seg == "clean_holdout":
        best_in = json.loads((root() / "eval_llm_in_sample.json").read_text(encoding="utf-8"))["best_baseline"]
    res["best_baseline"] = {}
    cells = P["cell"][idx]
    sec = P["sector"]
    for kk, h in enumerate(xs.REL_H):
        om = origin_mask(P, idx, h)
        rr = np.flatnonzero(om)
        real = P["rel_raw"][idx][rr][..., kk].astype(float)
        msk = pres[rr] & np.isfinite(real)
        dd = day[rr]
        q_raw = {a: raw_rel_quantiles(P, idx[rr], sub[f"rel_{a}"][rr])[..., kk, :] for a in ARMS}
        scores = {a: q_raw[a][..., 1] for a in ARMS}
        base = P["base_rel"][idx][rr].astype(float)
        for b, (col, sgn) in BASELINES.items():
            scores[b] = sgn * base[..., col]
        ic = {m: xs.spearman_rows(sc, real, msk) for m, sc in scores.items()}
        r = {"origins": int(np.isfinite(ic["kan_mix"]).sum()), "ic": {m: nw(per_day(v, dd)) for m, v in ic.items()}}
        best = best_in[f"{h}h"] if best_in else max(BASELINES, key=lambda b: r["ic"][b]["mean"])
        res["best_baseline"][f"{h}h"] = best
        r["best_baseline"] = best
        r["v3_minus_best"] = nw(per_day(ic["kan_mix"] - ic[best], dd))
        r["mlp_minus_best"] = nw(per_day(ic["mlp_mix"] - ic[best], dd))
        r["mix_minus_nomix"] = nw(per_day(ic["kan_mix"] - ic["kan_nomix"], dd))
        r["kan_minus_mlp"] = nw(per_day(ic["kan_mix"] - ic["mlp_mix"], dd))
        # IC by sector and by token (rank-product decomposition), by BTC v0 cell
        con = xs.rank_contrib(scores["kan_mix"], real, msk)
        r["ic_by_sector"] = {s: {"contrib": float(np.nanmean(con[:, sec == s])), "tokens": int((sec == s).sum())} for s in xs.SECTOR_ORDER}
        r["ic_by_token"] = {syms[j]: float(np.nanmean(con[:, j])) for j in range(N) if np.isfinite(con[:, j]).sum() > 20}
        cc = cells[rr]
        r["ic_by_cell"] = {c: nw(per_day(np.where(cc == c, ic["kan_mix"], np.nan), dd)) | {"origins": int(((cc == c) & np.isfinite(ic["kan_mix"])).sum())}
                           for c in sorted(set(cc.tolist())) if c.startswith("R")}
        r["ic_best_by_cell"] = {c: nw(per_day(np.where(cc == c, ic[best], np.nan), dd))["mean"] for c in r["ic_by_cell"]}
        # pairwise P(i beats j)
        r["pairwise"] = pairwise_eval(q_raw["kan_mix"], scores[best], real, msk, dd)
        # quintile spreads
        r["spread"] = {m: spread_summary(spread_series(scores[m], real, msk), dd, h) for m in ("kan_mix", "mlp_mix", "kan_nomix", best)}
        # Descriptive robustness (added after the in-sample look, not part of any verdict; diary 2026-11-15):
        # log-return targets reward low volatility mechanically (median / mean log return = arithmetic - sigma^2 / 2),
        # so report the IC and the spread on simple-return relative returns too, and a pure low-vol ranking (-lsig2).
        Rs = np.expm1(P["Rf_vol"][idx[rr]][..., list(xs.VOL_H).index(h)].astype(float))
        rel_s = xs.relative(np.where(pres[rr], Rs, np.nan), pres[rr])
        msk_s = msk & np.isfinite(rel_s)
        scores["lowvol"] = -P["X"][idx[rr]][..., list(P["features"]).index("lsig2")].astype(float)
        ic["lowvol"] = xs.spearman_rows(scores["lowvol"], real, msk)
        r["ic"]["lowvol"] = nw(per_day(ic["lowvol"], dd))
        r["v3_minus_lowvol"] = nw(per_day(ic["kan_mix"] - ic["lowvol"], dd))
        ic_s = {m: xs.spearman_rows(scores[m], rel_s, msk_s) for m in ("kan_mix", "mlp_mix", "kan_nomix", "lowvol", *BASELINES)}
        r["ic_simple"] = {m: nw(per_day(v, dd)) for m, v in ic_s.items()}
        r["v3_minus_best_simple"] = nw(per_day(ic_s["kan_mix"] - ic_s[best], dd))
        r["spread_simple"] = {m: spread_summary(spread_series(scores[m], rel_s, msk_s), dd, h) for m in ("kan_mix", "lowvol", best)}
        r["spread"]["lowvol"] = spread_summary(spread_series(scores["lowvol"], real, msk), dd, h)
        sp = spread_series(scores["kan_mix"], real, msk)
        r["spread_series"] = {"ts": P["ts"][idx][rr].tolist(), "net": np.nan_to_num(sp["net"]).tolist(), "gross": np.nan_to_num(sp["gross"]).tolist()}
        spb = spread_series(scores[best], real, msk)
        r["spread_series_best"] = {"net": np.nan_to_num(spb["net"]).tolist()}
        r["ic_series"] = {"ts": P["ts"][idx][rr].tolist(), "v3": np.nan_to_num(ic["kan_mix"]).tolist(), "best": np.nan_to_num(ic[best]).tolist()}
        rel[f"{h}h"] = r
    res["rel"] = rel
    # ---- ACI bands
    res["aci"] = {h: aci[h]["summary"][seg] for h in aci}
    # ---- attention
    att = {}
    hv = np.isin(cells, ["R3", "R6", "R9"])
    for arm in ("kan_mix", "mlp_mix"):
        W = sub[f"attn_{arm}"]
        a_all = attention_stats(P, idx, W, np.ones(len(idx), bool))
        a_hv = attention_stats(P, idx, W, hv) if hv.sum() > 100 else None
        rec = {"all": _att_json(a_all, syms), "high_vol": _att_json(a_hv, syms) if a_hv else None,
               "hubs": {s: hub_test(a_all["daily_recv_others"], syms.index(s), N) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT")}}
        if a_hv:
            rec["hubs_high_vol"] = {s: hub_test(a_hv["daily_recv_others"], syms.index(s), N) for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT")}
        att[arm] = rec
    res["attention"] = att
    res["high_vol_hours"] = int(hv.sum())
    # ---- verdicts
    res["verdicts"] = verdicts(res)
    (root() / f"eval_{seg}.json").write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    if record:
        record_results(seg, res)
    print(json.dumps(summary_lines(res), indent=1))
    return res


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _att_json(a: dict, syms: list[str]) -> dict:
    return {"hours": a["hours"], "self_share": a["self_share"], "mkt_share": a["mkt_share"],
            "recv": dict(zip([*syms, "MARKET"], np.round(a["recv"], 4).tolist(), strict=True)),
            "recv_others": dict(zip([*syms, "MARKET"], np.round(a["recv_others"], 4).tolist(), strict=True)),
            "mean_w": np.round(np.nan_to_num(a["mean_w"]), 4).tolist()}


def pairwise_eval(q_raw: np.ndarray, base_score: np.ndarray, real: np.ndarray, msk: np.ndarray, dd: np.ndarray) -> dict:
    Pp = xs.pairwise_prob(q_raw)
    T, N = real.shape
    iu = np.triu_indices(N, 1)
    acc, brier, acc_b, n = (np.full(T, np.nan) for _ in range(4))
    for t in range(T):
        k = msk[t] & np.isfinite(q_raw[t, :, 1])
        pair = k[iu[0]] & k[iu[1]]
        if k.sum() < xs.MIN_TOKENS:
            continue
        i, j = iu[0][pair], iu[1][pair]
        y = (real[t, i] > real[t, j]).astype(float)
        p = Pp[t, i, j]
        acc[t] = np.mean((p > 0.5) == (y == 1))
        brier[t] = np.mean((p - y) ** 2)
        bs = base_score[t]
        acc_b[t] = np.mean(((bs[i] - bs[j]) > 0) == (y == 1))
        n[t] = len(i)
    return {"pairs": int(np.nansum(n)), "accuracy": nw(per_day(acc, dd)), "brier": nw(per_day(brier, dd)),
            "brier_skill_vs_half": float(1 - np.nanmean(brier) / 0.25), "accuracy_best_baseline": nw(per_day(acc_b, dd)),
            "acc_minus_baseline": nw(per_day(acc - acc_b, dd))}


def run_aci(P: dict, C: dict) -> dict:
    """ACI (gamma ACI_GAMMA, window ACI_WINDOW origins) on the v3 80% rel band [q10, q90], per token and horizon, over
    non-overlapping origins of the walk-forward stream; warm start = the first refit's validation scores."""
    from ai_trader.ml import pnn

    idx = C["idx"]
    sg = seg_of(P, idx)
    syms = list(P["symbols"])
    out = {}
    for kk, h in enumerate(xs.REL_H):
        om = origin_mask(P, idx, h)
        vom = origin_mask(P, C["val_idx"], h)
        y_all = P["Yr"][idx][..., kk]
        q = C["rel_kan_mix"][..., kk, :]
        yv = P["Yr"][C["val_idx"]][..., kk]
        qv = C["val_rel"][..., kk, :]
        s_raw = np.sqrt(h) * np.sqrt(P["s2"][idx])
        per = {s: {"cov": [], "cov_raw": [], "width_bp": [], "width_raw_bp": [], "seg": []} for s in syms}
        rows = {"seg": [], "hit": [], "hit_raw": [], "w": [], "w_raw": [], "tok": []}
        for j, s in enumerate(syms):
            r = np.flatnonzero(om & np.isfinite(y_all[:, j]) & P["present"][idx][:, j])
            if len(r) < 50:
                continue
            rv = np.flatnonzero(vom & np.isfinite(yv[:, j]))
            warm = pnn.cqr_scores(qv[rv, j, 0], qv[rv, j, 2], yv[rv, j]) if len(rv) else None
            lo, hi, y = q[r, j, 0], q[r, j, 2], y_all[r, j]
            a = pnn.aci(lo, hi, y, 1 - REL_LEVEL, gamma=ACI_GAMMA, window=ACI_WINDOW, warm=warm)
            hit = (y >= a["lo"]) & (y <= a["hi"])
            hit_raw = (y >= lo) & (y <= hi)
            rows["seg"] += sg[r].tolist()
            rows["hit"] += hit.tolist()
            rows["hit_raw"] += hit_raw.tolist()
            rows["w"] += ((a["hi"] - a["lo"]) * s_raw[r, j] * 1e4).tolist()
            rows["w_raw"] += ((hi - lo) * s_raw[r, j] * 1e4).tolist()
            rows["tok"] += [s] * len(r)
            per[s] = {"alpha_last": float(a["alpha_t"][-1]), "scores_tail": pnn.cqr_scores(lo, hi, y)[-ACI_WINDOW:].tolist()}
        R = {k: np.asarray(v) for k, v in rows.items()}
        summ = {}
        for sgn in SEGS:
            m = R["seg"] == sgn
            if not m.any():
                continue
            tok = {}
            for s in syms:
                mm = m & (R["tok"] == s)
                if mm.sum() > 10:
                    tok[s] = {"coverage": float(R["hit"][mm].mean()), "coverage_raw": float(R["hit_raw"][mm].mean()), "width_bp": float(R["w"][mm].mean()), "n": int(mm.sum())}
            covs = [v["coverage"] for v in tok.values()]
            summ[sgn] = {"coverage": float(R["hit"][m].mean()), "coverage_raw": float(R["hit_raw"][m].mean()), "width_bp": float(R["w"][m].mean()),
                         "width_raw_bp": float(R["w_raw"][m].mean()), "token_coverage_min": float(min(covs)), "token_coverage_max": float(max(covs)),
                         "tokens_within_3pp": int(sum(abs(c - REL_LEVEL) <= 0.03 for c in covs)), "tokens": len(covs), "per_token": tok}
        out[f"{h}h"] = {"summary": summ, "state": per}
    return out


def verdicts(res: dict) -> dict:
    # H-XS1
    rel = res["rel"]
    win = [h for h, r in rel.items() if r["ic"]["kan_mix"]["mean"] > 0 and r["ic"]["kan_mix"]["t"] > 2 and r["v3_minus_best"]["mean"] > 0 and r["v3_minus_best"]["t"] > 2]
    lose = [h for h, r in rel.items() if r["ic"]["kan_mix"]["mean"] <= 0 or r["v3_minus_best"]["mean"] <= 0]
    p1 = min(max(_phi(-r["ic"]["kan_mix"]["t"]), _phi(-r["v3_minus_best"]["t"])) for r in rel.values())
    v1 = "supports" if win else ("contradicts" if len(lose) == len(rel) else "inconclusive")
    # H-XS2 (1h, 4h)
    vol = {h: res["vol"][h] for h in ("1h", "4h")}
    win = [h for h, r in vol.items() if r["v3_minus_har"]["mean"] < 0 and r["v3_minus_har"]["t"] < -2]
    lose = [h for h, r in vol.items() if r["v3_minus_har"]["mean"] >= 0]
    p2 = min(_phi(r["v3_minus_har"]["t"]) for r in vol.values())
    v2 = "supports" if win else ("contradicts" if len(lose) == len(vol) else "inconclusive")
    # H-XS3 (1h, 4h; non-BTC)
    win = [h for h, r in vol.items() if r["nomix_minus_mix_nonbtc"]["mean"] > 0 and r["nomix_minus_mix_nonbtc"]["t"] > 2]
    lose = [h for h, r in vol.items() if r["nomix_minus_mix_nonbtc"]["mean"] <= 0]
    p3 = min(_phi(-r["nomix_minus_mix_nonbtc"]["t"]) for r in vol.values())
    v3 = "supports" if win else ("contradicts" if len(lose) == len(vol) else "inconclusive")
    from ai_trader.research.decision_budget_study import holm

    raw = {"H-XS1": p1, "H-XS2": p2, "H-XS3": p3}
    adj = holm(raw)
    out = {}
    for hid, v in (("H-XS1", v1), ("H-XS2", v2), ("H-XS3", v3)):
        vv = "inconclusive" if (v == "supports" and adj[hid] > 0.05) else v
        out[hid] = {"verdict": vv, "raw_verdict": v, "p": raw[hid], "holm_p": adj[hid]}
    return out


def _metrics(res: dict, hid: str) -> dict:
    if hid == "H-XS1":
        return {h: {"ic_v3": r["ic"]["kan_mix"], "ic_baselines": {b: r["ic"][b] for b in BASELINES}, "best_baseline": r["best_baseline"],
                    "v3_minus_best": r["v3_minus_best"], "origins": r["origins"], "ic_mlp": r["ic"]["mlp_mix"]["mean"],
                    "ic_nomix": r["ic"]["kan_nomix"]["mean"]} for h, r in res["rel"].items()}
    if hid == "H-XS2":
        return {h: {"pinball": r["pinball"], "v3_minus_har": r["v3_minus_har"], "gain_vs_har_pct": r["gain_vs_har_pct"], "cells": r["cells"]}
                for h, r in res["vol"].items()}
    return {h: {"nomix_minus_mix_nonbtc": r["nomix_minus_mix_nonbtc"], "pinball_mix": r["pinball"]["kan_mix"], "pinball_nomix": r["pinball"]["kan_nomix"]}
            for h, r in res["vol"].items()} | {"ic_mix_minus_nomix": {h: r["mix_minus_nomix"] for h, r in res["rel"].items()}}


def record_results(seg: str, res: dict) -> None:
    from ai_trader.research.hypotheses import record

    data = "universe L bybit 1h 2021-2026 (panel data/models/xs_v3/cache/panel_2026-11-15.npz); src/ai_trader/research/xs_network_study.py"
    for hid, v in res["verdicts"].items():
        m = _metrics(res, hid) | {"p": v["p"], "holm_p": v["holm_p"], "raw_verdict": v["raw_verdict"], "days": res["days"], "quarters": res["quarters"]}
        record(hid, seg, v["verdict"], m, note=f"XS v3 quarterly purged walk-forward, {N_SEEDS} seeds, Holm over H-XS1..3", data=data)
    if seg == "clean_holdout":
        ins = json.loads((root() / "eval_llm_in_sample.json").read_text(encoding="utf-8"))["verdicts"]
        for hid, v in res["verdicts"].items():
            a, b = ins[hid]["verdict"], v["verdict"]
            pooled = "supports" if (a == b == "supports") else ("contradicts" if "contradicts" in (a, b) else "inconclusive")
            record(hid, "pooled", pooled, {"llm_in_sample": a, "clean_holdout": b}, note="pooled = supports iff both support; contradicts iff either contradicts", data=data)


def summary_lines(res: dict) -> dict:
    o = {"segment": res["segment"], "days": res["days"], "verdicts": {k: v["verdict"] for k, v in res["verdicts"].items()}}
    for h, r in res["rel"].items():
        o[f"ic_{h}"] = {m: round(v["mean"], 4) for m, v in r["ic"].items()} | {"v3-best t": round(r["v3_minus_best"]["t"], 2), "v3 t": round(r["ic"]["kan_mix"]["t"], 2)}
    for h, r in res["vol"].items():
        o[f"vol_{h}"] = {"gain_vs_har_%": round(r["gain_vs_har_pct"], 3), "t": round(r["v3_minus_har"]["t"], 2), "nomix-mix t": round(r["nomix_minus_mix_nonbtc"]["t"], 2)}
    return o


# ------------------------------------------------------------------ report


REPORT = "xs_network_2026-11-15"
C_V3, C_BASE, C_ALT, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def fig_dir() -> Path:
    p = load().root / "research" / "figures" / REPORT
    p.mkdir(parents=True, exist_ok=True)
    return p


def _style(ax, title: str = "", ylabel: str = "") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _years(ax) -> None:
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _f(x: float, d: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _mt(r: dict, d: int = 4, scale: float = 1.0) -> str:
    return f"{_f(r['mean'] * scale, d)} ({_f(r['t'], 2)})"


def make_figures(E: dict, P: dict) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    syms = list(P["symbols"])
    short = [s.replace("USDT", "") for s in syms]
    # 1. IC bars
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    models = [("kan_mix", "v3 (KAN + mixing)", C_V3), ("mlp_mix", "MLP + mixing", C_ALT), ("kan_nomix", "v3 without mixing", C_GRAY),
              ("mom24", "24h rel. momentum", C_BASE), ("mom7d", "7d rel. momentum", "#eda100"), ("rev1", "1h rel. reversal", "#4a3aa7")]
    for ax, h in zip(axes, ("4h", "24h"), strict=True):
        w = 0.13
        for i, (m, lab, c) in enumerate(models):
            for j, sgn in enumerate(SEGS):
                if sgn not in E:
                    continue
                r = E[sgn]["rel"][h]["ic"][m]
                se = abs(r["mean"] / r["t"]) if r["t"] and np.isfinite(r["t"]) else 0
                ax.bar(j + (i - 2.5) * w, r["mean"], w * 0.9, color=c, label=lab if j == 0 else None)
                ax.errorbar(j + (i - 2.5) * w, r["mean"], yerr=1.96 * se, color=INK2, lw=0.8, capsize=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(SEGS)), ["llm_in_sample", "clean_holdout"])
        _style(ax, f"Rank IC of relative return, {h} ahead", "mean Spearman IC (95% NW CI)" if h == "4h" else "")
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    p = fig_dir() / "ic_vs_baselines.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Mean cross-sectional rank IC of v3 and its arms vs the three momentum / reversal baselines"))
    # 2. cumulative IC
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, h in zip(axes, ("4h", "24h"), strict=True):
        ts, v3, bb = [], [], []
        for sgn in SEGS:
            if sgn in E:
                s = E[sgn]["rel"][h]["ic_series"]
                ts += s["ts"]
                v3 += s["v3"]
                bb += s["best"]
        t = np.array(ts, dtype="datetime64[us]")
        ax.plot(t, np.cumsum(v3), color=C_V3, lw=1.5, label="v3")
        ax.plot(t, np.cumsum(bb), color=C_BASE, lw=1.5, label=f"best baseline ({E['llm_in_sample']['rel'][h]['best_baseline']})")
        ax.axvline(np.datetime64("2026-07-01"), color=INK2, lw=0.8, ls="--")
        _style(ax, f"Cumulative IC, {h} origins", "sum of per-origin IC")
        _years(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "ic_cumulative.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Cumulative per-origin IC through time (dashed line = knowledge cutoff)"))
    # 3. vol gain by token
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, h in zip(axes, ("1h", "4h"), strict=True):
        x = np.arange(len(syms))
        for j, (sgn, c) in enumerate(zip(SEGS, (C_V3, C_BASE), strict=True)):
            if sgn not in E:
                continue
            g = E[sgn]["vol"][h]["per_token_gain_vs_har_pct"]
            ax.bar(x + (j - 0.5) * 0.4, [g.get(s, np.nan) for s in syms], 0.38, color=c, label=sgn)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, short, rotation=90, fontsize=7)
        _style(ax, f"Vol pinball gain of v3 over panel log-HAR-t, {h}", "% lower pinball" if h == "1h" else "")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "vol_gain_by_token.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Per-token % reduction in vol pinball loss, v3 vs panel log-HAR-t x hour-of-week"))
    # 4. attention heatmaps
    A = E["llm_in_sample"]["attention"]["kan_mix"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, ttl in zip(axes, ("all", "high_vol"), ("all hours", "BTC v0 high-vol cells (R3/R6/R9)"), strict=True):
        if not A.get(key):
            continue
        W = np.array(A[key]["mean_w"])
        nk = np.array([*([1] * len(syms)), 1])
        im = ax.imshow(W * (len(syms) + 1) * nk[None, :], cmap="Blues", vmin=0, vmax=np.nanquantile(W * (len(syms) + 1), 0.98))
        ax.set_xticks(range(len(syms) + 1), [*short, "MKT"], rotation=90, fontsize=6)
        ax.set_yticks(range(len(syms)), short, fontsize=6)
        ax.set_xlabel("key (attended to)", fontsize=8, color=INK2)
        ax.set_ylabel("query (token being forecast)", fontsize=8, color=INK2)
        ax.set_title(f"v3 attention x number of keys, {ttl}\n(llm_in_sample test hours; 1 = uniform)", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    p = fig_dir() / "attention_maps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Who attends to whom: mean attention weight scaled by the number of keys (1 = uniform)"))
    # 5. received attention
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(len(syms) + 1)
    for j, (sgn, c) in enumerate(zip(SEGS, (C_V3, C_BASE), strict=True)):
        if sgn not in E:
            continue
        rv = E[sgn]["attention"]["kan_mix"]["all"]["recv_others"]
        ax.bar(x + (j - 0.5) * 0.4, [rv[k] for k in [*syms, "MARKET"]], 0.38, color=c, label=sgn)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x, [*short, "MKT"], rotation=90, fontsize=7)
    _style(ax, "Attention received from other tokens (ratio to uniform)", "ratio")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "attention_received.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Hubs: attention each key receives from the other tokens' queries, relative to uniform"))
    # 6. cumulative quintile spread
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, h in zip(axes, ("4h", "24h"), strict=True):
        ts, nv, nb = [], [], []
        for sgn in SEGS:
            if sgn in E:
                s = E[sgn]["rel"][h]
                ts += s["spread_series"]["ts"]
                nv += s["spread_series"]["net"]
                nb += s["spread_series_best"]["net"]
        t = np.array(ts, dtype="datetime64[us]")
        ax.plot(t, np.cumsum(nv) * 100, color=C_V3, lw=1.5, label="v3 top - bottom quintile")
        ax.plot(t, np.cumsum(nb) * 100, color=C_BASE, lw=1.5, label="best baseline")
        ax.axvline(np.datetime64("2026-07-01"), color=INK2, lw=0.8, ls="--")
        ax.axhline(0, color=INK2, lw=0.6)
        _style(ax, f"Quintile spread net of 7.5 bp/side/leg, {h} rebalance", "sum of net log returns, %")
        _years(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "quintile_spread.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Cumulative net top-minus-bottom quintile relative return (research only, no sizing)"))
    # 7. embedding map (latest refit representation)
    rep = E["_reps"][-1]
    ok = np.all(np.isfinite(rep), axis=1)
    R = rep[ok] - rep[ok].mean(0)
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    xy = R @ Vt[:2].T
    secs = P["sector"][ok]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cols = dict(zip(xs.SECTOR_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"], strict=True))
    for s in xs.SECTOR_ORDER:
        m = secs == s
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=46, color=cols[s], label=s, edgecolor="white", lw=1.5, zorder=3)
    for (a, b), nm in zip(xy, np.array(short)[ok], strict=True):
        ax.annotate(nm, (a, b), fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
    ev = S[:2] ** 2 / (S ** 2).sum()
    _style(ax, f"Token map: PCA of v3's mean head input, refit {E['_rep_quarters'][-1]}")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=8, color=INK2)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="best", title="sector", title_fontsize=7)
    fig.tight_layout()
    p = fig_dir() / "token_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "2D map of the per-token representation (encoder + mixing output), coloured by sector"))
    E["_pca"] = {"tokens": np.array(syms)[ok].tolist(), "xy": xy.tolist(), "explained": ev.tolist()}
    # 8. IC by sector / cell
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, (sgn, c) in enumerate(zip(SEGS, (C_V3, C_BASE), strict=True)):
        if sgn not in E:
            continue
        r = E[sgn]["rel"]["24h"]
        axes[0].bar(np.arange(len(xs.SECTOR_ORDER)) + (j - 0.5) * 0.4, [r["ic_by_sector"][s]["contrib"] for s in xs.SECTOR_ORDER], 0.38, color=c, label=sgn)
        cl = [f"R{i}" for i in range(1, 10)]
        axes[1].bar(np.arange(9) + (j - 0.5) * 0.4, [r["ic_by_cell"].get(cc, {"mean": np.nan})["mean"] for cc in cl], 0.38, color=c, label=sgn)
    axes[0].set_xticks(range(len(xs.SECTOR_ORDER)), xs.SECTOR_ORDER, fontsize=7, rotation=30)
    axes[1].set_xticks(range(9), [f"R{i}" for i in range(1, 10)], fontsize=7)
    for ax, t in zip(axes, ("24h IC contribution by sector", "24h IC by BTC v0 regime cell"), strict=True):
        ax.axhline(0, color=INK2, lw=0.8)
        _style(ax, t, "mean")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "ic_by_sector_cell.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append((p.name, "Where the 24h IC comes from: sector contributions and BTC regime cells"))
    return out


def report() -> None:
    P = build()
    E = {}
    for sgn in SEGS:
        f = root() / f"eval_{sgn}.json"
        if f.is_file():
            E[sgn] = json.loads(f.read_text(encoding="utf-8"))
    C = collect(P, tuple(E))
    E["_reps"] = C["reps"]
    E["_rep_quarters"] = C["rep_quarters"]
    figs = make_figures(E, P)
    S = json.loads((root() / "search.json").read_text(encoding="utf-8"))
    infos = []
    import torch

    for q, _ in quarters(P, "llm_in_sample") + quarters(P, "clean_holdout"):
        f = wf_dir() / f"{qtag(q)}.pt"
        if f.is_file():
            infos.append(torch.load(f, weights_only=False)["info"])
    md = render_md(E, S, infos, figs)
    rp = load().root / "research" / "reports"
    old = (rp / f"{REPORT}.md").read_text(encoding="utf-8") if (rp / f"{REPORT}.md").is_file() else ""
    # keep the hand-written sections (findings, rules, live note) across regenerations
    for tag in ("FINDINGS", "RULES", "LIVE"):
        a, b = f"<!-- {tag} -->", f"<!-- /{tag} -->"
        if a in old and b in old:
            keep = old[old.index(a): old.index(b) + len(b)]
            md = md.replace(f"{a}\n{b}", keep)
    (rp / f"{REPORT}.md").write_text(md, encoding="utf-8")
    slim = {k: v for k, v in E.items() if not k.startswith("_reps") and k != "_rep_quarters"}
    for sgn in SEGS:
        if sgn in slim:
            for h in slim[sgn]["rel"].values():
                for k in ("spread_series", "spread_series_best", "ic_series"):
                    h.pop(k, None)
            for arm in slim[sgn]["attention"].values():
                for k in ("all", "high_vol"):
                    if arm.get(k):
                        arm[k].pop("mean_w", None)
    slim["search"] = S
    slim["refits"] = infos
    slim["figures"] = [f"research/figures/{REPORT}/{n}" for n, _ in figs]
    (rp / f"{REPORT}.json").write_text(json.dumps(slim, indent=1, default=_json_default), encoding="utf-8")
    print("wrote", rp / f"{REPORT}.md")


def render_md(E: dict, S: dict, infos: list, figs: list) -> str:
    L = []
    a = L.append
    a("# Cross-sectional network v3: relative strength, volatility and spillovers across universe L (H-XS1 to H-XS3)")
    a("")
    a("_Block XS, agent X3, 2026-11-15. Data: Bybit 1h bars, funding and 4h OI for the 23 universe-L perps, 2021-05-25 to 2026-11-14, as of "
      "2026-11-15 10:00 UTC; knowledge cutoff 2026-07-01. Code: `src/ai_trader/ml/xs_net.py` (panel maths, network), "
      "`src/ai_trader/research/xs_network_study.py` (design in its docstring); tests `tests/test_xs_net.py`; artifacts `data/models/xs_v3/`. "
      "Design frozen in the diary before any fit ('XS v3 design frozen'). Research only: deterministic tools, nothing live, no sizing._")
    a("")
    a("## Verdicts")
    a("")
    a("| hypothesis | llm_in_sample | clean_holdout | pooled | Holm p (in / hold) |")
    a("|---|---|---|---|---|")
    for hid in ("H-XS1", "H-XS2", "H-XS3"):
        vi = E.get("llm_in_sample", {}).get("verdicts", {}).get(hid, {})
        vh = E.get("clean_holdout", {}).get("verdicts", {}).get(hid, {})
        pooled = "n/a"
        if vi and vh:
            x, y = vi["verdict"], vh["verdict"]
            pooled = "supports" if x == y == "supports" else ("contradicts" if "contradicts" in (x, y) else "inconclusive")
        a(f"| {hid} | {vi.get('verdict', 'n/a')} | {vh.get('verdict', 'n/a')} | **{pooled}** | {_f(vi.get('holm_p', np.nan), 3)} / {_f(vh.get('holm_p', np.nan), 3)} |")
    a("")
    pr = S["params"]
    a(f"**Sizes.** v3 (kan_mix) {pr['kan_mix']} parameters, MLP arm {pr['mlp_mix']}, ablation {pr['kan_nomix']} (cap 5,000). "
      f"Config search (in-sample, first test quarter {S['first_test_quarter']}): kan_mix {S['best']['kan_mix']}, mlp_mix {S['best']['mlp_mix']}.")
    if infos:
        secs = sum(i["secs"] for i in infos)
        ep = {arm: np.mean([e["epoch"] for i in infos for e in i[arm]]) for arm in ARMS}
        a(f"**Walk-forward.** {len(infos)} quarterly refits ({infos[0]['quarter']} to {infos[-1]['quarter']}), {N_SEEDS} seeds per arm, warm-started; "
          f"mean best epoch kan_mix {ep['kan_mix']:.1f}, mlp_mix {ep['mlp_mix']:.1f}, kan_nomix {ep['kan_nomix']:.1f}; {secs / 60:.0f} CPU-minutes (2 threads).")
    a("")
    a("<!-- FINDINGS -->\n<!-- /FINDINGS -->")
    a("")
    for n, cap in figs:
        if n in ("ic_vs_baselines.png", "ic_cumulative.png"):
            a(f"![{cap}](../figures/{REPORT}/{n})")
            a("")
    a("## H-XS1: rank IC of the relative-return median vs momentum and reversal")
    a("")
    a("Mean per-origin Spearman IC between the predicted median relative return and the realised one (token minus the equal-weight universe), "
      "non-overlapping origins (4h: every 4 hours; 24h: 00 UTC), NW t over days in brackets. Best baseline chosen in-sample per horizon.")
    a("")
    a("| segment | horizon | origins | v3 | MLP arm | v3 no mixing | mom24 | mom7d | rev1 | best | v3 - best |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn not in E:
            continue
        for h, r in E[sgn]["rel"].items():
            ic = r["ic"]
            a(f"| {sgn} | {h} | {r['origins']} | {_mt(ic['kan_mix'])} | {_mt(ic['mlp_mix'])} | {_mt(ic['kan_nomix'])} | {_mt(ic['mom24'])} | {_mt(ic['mom7d'])} | "
              f"{_mt(ic['rev1'])} | {r['best_baseline']} | {_mt(r['v3_minus_best'])} |")
    a("")
    a("**Descriptive robustness (added after the in-sample look; not part of any verdict).** A pure low-volatility ranking (minus lsig2) "
      "earns a large rank IC on its own. Rank IC is identical on log and simple relative returns (ranks are invariant to exp and to "
      "subtracting the cross-sectional mean), so this is not a log-return artefact in the ranks; it says high-vol tokens land below the "
      "median more often. Whether that is worth anything shows in the simple-return spread below, where it is not.")
    a("")
    a("| segment | horizon | v3 IC | low-vol IC | v3 - low-vol (t) |")
    a("|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn not in E:
            continue
        for h, r in E[sgn]["rel"].items():
            if "lowvol" in r["ic"]:
                a(f"| {sgn} | {h} | {_mt(r['ic']['kan_mix'])} | {_mt(r['ic']['lowvol'])} | {_mt(r['v3_minus_lowvol'])} |")
    a("")
    a("Pairwise view (from the rel heads, P(i beats j) = Phi((m_i - m_j) / sqrt(s_i^2 + s_j^2)), all token pairs at the same origins):")
    a("")
    a("| segment | horizon | pairs | accuracy v3 (t vs 0) | accuracy best baseline | v3 - baseline (t) | Brier skill vs 1/2 |")
    a("|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn in E:
            for h, r in E[sgn]["rel"].items():
                pw = r["pairwise"]
                a(f"| {sgn} | {h} | {pw['pairs']} | {_f(pw['accuracy']['mean'])} | {_f(pw['accuracy_best_baseline']['mean'])} | {_mt(pw['acc_minus_baseline'])} | {_f(pw['brier_skill_vs_half'], 5)} |")
    a("")
    a("### Where the IC comes from")
    a("")
    a("IC contribution = mean of z(rank predicted) x z(rank realised) over the tokens of the sector (the token-average of these products is the IC). "
      "Regime cells are BTC's v0 cell known at the origin.")
    a("")
    for h in ("4h", "24h"):
        a(f"**{h}, by sector:** " + "; ".join(
            f"{sgn}: " + ", ".join(f"{s} {E[sgn]['rel'][h]['ic_by_sector'][s]['contrib']:+.3f}" for s in xs.SECTOR_ORDER) for sgn in SEGS if sgn in E))
        a("")
        a(f"**{h}, by BTC v0 cell (v3 IC / best-baseline IC, origins):**")
        a("")
        a("| segment | " + " | ".join(f"R{i}" for i in range(1, 10)) + " |")
        a("|---|" + "---|" * 9)
        for sgn in SEGS:
            if sgn in E:
                r = E[sgn]["rel"][h]
                a(f"| {sgn} | " + " | ".join(
                    (f"{r['ic_by_cell'][f'R{i}']['mean']:+.3f} / {r['ic_best_by_cell'][f'R{i}']:+.3f} ({r['ic_by_cell'][f'R{i}']['origins']})" if f"R{i}" in r["ic_by_cell"] else "-")
                    for i in range(1, 10)) + " |")
        a("")
    for n, cap in figs:
        if n == "ic_by_sector_cell.png":
            a(f"![{cap}](../figures/{REPORT}/{n})")
            a("")
    a("## H-XS2: per-token volatility quantiles vs panel log-HAR-t x hour-of-week")
    a("")
    a("Pinball loss of the return-magnitude quantiles (5..95, in units of sqrt(h) s_t), averaged over token-origins; paired difference per day, NW t.")
    a("")
    a("| segment | horizon | cells | v3 | MLP arm | log-HAR-t | v3 - HAR (t) | gain % | MLP - HAR (t) | KAN - MLP (t) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn in E:
            for h, r in E[sgn]["vol"].items():
                p = r["pinball"]
                a(f"| {sgn} | {h}{' (descriptive)' if h == '24h' else ''} | {r['cells']} | {_f(p['kan_mix'], 5)} | {_f(p['mlp_mix'], 5)} | {_f(p['har'], 5)} | "
                  f"{_mt(r['v3_minus_har'], 5)} | {r['gain_vs_har_pct']:.2f} | {_mt(r['mlp_minus_har'], 5)} | {_mt(r['kan_minus_mlp'], 5)} |")
    a("")
    a("Calibration (share of outcomes below each quantile, nominal 0.05 .. 0.95):")
    a("")
    a("| segment | horizon | model | " + " | ".join(f"q{int(t * 100):02d}" for t in xs.TAUS) + " |")
    a("|---|---|---|" + "---|" * len(xs.TAUS))
    for sgn in SEGS:
        if sgn in E:
            for h in ("1h", "4h"):
                for m in ("kan_mix", "har"):
                    a(f"| {sgn} | {h} | {m} | " + " | ".join(f"{x:.3f}" for x in E[sgn]["vol"][h]["hit_rate"][m]) + " |")
    a("")
    for n, cap in figs:
        if n == "vol_gain_by_token.png":
            a(f"![{cap}](../figures/{REPORT}/{n})")
            a("")
    a("## H-XS3: ablation without the mixing layer")
    a("")
    a("Same encoder, heads, rows, seeds and config; the attention keys are restricted to the token itself (no other tokens, no market token).")
    a("")
    a("| segment | horizon | pinball mixing | pinball no mixing | no-mix - mix, non-BTC (t) | no-mix - mix, all (t) |")
    a("|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn in E:
            for h, r in E[sgn]["vol"].items():
                a(f"| {sgn} | {h} | {_f(r['pinball']['kan_mix'], 5)} | {_f(r['pinball']['kan_nomix'], 5)} | {_mt(r['nomix_minus_mix_nonbtc'], 5)} | {_mt(r['nomix_minus_mix_all'], 5)} |")
    a("")
    a("IC part (reported): v3 - no-mixing rank IC: " + "; ".join(
        f"{sgn} {h} {_mt(E[sgn]['rel'][h]['mix_minus_nomix'])}" for sgn in SEGS if sgn in E for h in ("4h", "24h")) + ".")
    a("")
    a("Per-token effect of mixing at 1h (% pinball increase when it is removed; positive = mixing helps): " + "; ".join(
        f"{sgn}: " + ", ".join(f"{k.replace('USDT', '')} {v:+.2f}" for k, v in sorted(E[sgn]["vol"]["1h"]["per_token_mix_gain_pct"].items(), key=lambda kv: -kv[1])[:6])
        for sgn in SEGS if sgn in E) + ".")
    a("")
    a("## Attention: who attends to whom")
    a("")
    a("Mean attention weights of v3 (averaged over heads and seeds) on test hours. 'Received from others' = the weight a key gets from the "
      "other tokens' queries times the number of keys (1 = uniform). Hub test = per-day received ratio of the token minus the mean of the "
      "other tokens, NW t. High-vol hours = BTC v0 cells R3/R6/R9.")
    a("")
    a("| segment | arm | hours | self share | market-token share | top keys (received from others) | ETH hub (t) | SOL hub (t) | BTC hub (t) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn not in E:
            continue
        for arm in ("kan_mix", "mlp_mix"):
            for key, hk in (("all", "hubs"), ("high_vol", "hubs_high_vol")):
                A = E[sgn]["attention"][arm]
                if not A.get(key):
                    continue
                rv = A[key]["recv_others"]
                top = sorted(((k, v) for k, v in rv.items() if np.isfinite(v)), key=lambda kv: -kv[1])[:5]
                hb = A.get(hk, {})
                a(f"| {sgn} | {arm} {'(high vol)' if key == 'high_vol' else ''} | {A[key]['hours']} | {A[key]['self_share']:.3f} | {A[key]['mkt_share']:.3f} | "
                  + ", ".join(f"{k.replace('USDT', '')} {v:.2f}" for k, v in top) + " | "
                  + " | ".join(_mt(hb[s], 3) if s in hb else "-" for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT")) + " |")
    a("")
    for n, cap in figs:
        if n in ("attention_maps.png", "attention_received.png"):
            a(f"![{cap}](../figures/{REPORT}/{n})")
            a("")
    a("## Top-minus-bottom quintile relative-return spread (descriptive, research only)")
    a("")
    a("Equal-weight top quintile minus bottom quintile by predicted median relative return, rebalanced at each non-overlapping origin; log returns; "
      f"cost {COST_BP} bp per side per leg on turnover (sum |dw| over both legs). Turnover = one-way share of the book replaced per rebalance. "
      "Sizing belongs to the risk engine; this measures ranking value only.")
    a("")
    a("| segment | horizon | ranking | rebalances | gross bp | net bp | turnover | Sharpe net (ann.) | net t (days) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn in E:
            for h, r in E[sgn]["rel"].items():
                for m, s in r["spread"].items():
                    a(f"| {sgn} | {h} | {m} | {s['origins']} | {s['gross_bp']:.2f} | {s['net_bp']:.2f} | {s['turnover_per_rebalance']:.2f} | {_f(s['sharpe_net_ann'], 2)} | {_f(s['net_t_daily'], 2)} |")
    a("")
    a("The same spread in simple returns (what an equal-weight long-short book would earn before funding; descriptive):")
    a("")
    a("| segment | horizon | ranking | gross bp | net bp | turnover | Sharpe net (ann.) | net t (days) |")
    a("|---|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn in E:
            for h, r in E[sgn]["rel"].items():
                for m, s in r.get("spread_simple", {}).items():
                    a(f"| {sgn} | {h} | {m} | {s['gross_bp']:.2f} | {s['net_bp']:.2f} | {s['turnover_per_rebalance']:.2f} | {_f(s['sharpe_net_ann'], 2)} | {_f(s['net_t_daily'], 2)} |")
    a("")
    for n, cap in figs:
        if n == "quintile_spread.png":
            a(f"![{cap}](../figures/{REPORT}/{n})")
            a("")
    a("## Conformal relative-return bands (ACI per token)")
    a("")
    a(f"80% band [q10, q90] of the rel head, adaptive conformal per token (gamma {ACI_GAMMA}, window {ACI_WINDOW} origins, warm start = first refit's validation scores), non-overlapping origins.")
    a("")
    a("| segment | horizon | coverage ACI | coverage raw | tokens within 3 pp | token range | width ACI bp | width raw bp |")
    a("|---|---|---|---|---|---|---|---|")
    for sgn in SEGS:
        if sgn in E:
            for h, r in E[sgn]["aci"].items():
                a(f"| {sgn} | {h} | {r['coverage']:.3f} | {r['coverage_raw']:.3f} | {r['tokens_within_3pp']}/{r['tokens']} | {r['token_coverage_min']:.3f}-{r['token_coverage_max']:.3f} | {r['width_bp']:.0f} | {r['width_raw_bp']:.0f} |")
    a("")
    a("## Token map")
    a("")
    for n, cap in figs:
        if n == "token_map.png":
            a(f"![{cap}](../figures/{REPORT}/{n})")
            a("")
    a("<!-- LIVE -->\n<!-- /LIVE -->")
    a("")
    a("<!-- RULES -->\n<!-- /RULES -->")
    a("")
    return "\n".join(L)


# ------------------------------------------------------------------ freeze


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze() -> None:
    """data/models/xs_v3/frozen/: the latest refit's v3 (kan_mix) seeds, spec with features and standardisation, ACI state,
    hashes. A live runner rebuilds the panel row for hour t with xs_network_study.token_rows / assemble (same code)."""
    import subprocess

    import torch

    P = build()
    allq = quarters(P, "llm_in_sample") + quarters(P, "clean_holdout")
    q = [qq for qq, _ in allq if (wf_dir() / f"{qtag(qq)}.pt").is_file()][-1]
    ck = torch.load(wf_dir() / f"{qtag(q)}.pt", weights_only=False)
    fd = root() / "frozen"
    fd.mkdir(parents=True, exist_ok=True)
    files = {}
    for s, stt in enumerate(ck["states"]["kan_mix"]):
        p = fd / f"v3_seed{s}.pt"
        torch.save(stt, p)
        files[p.name] = _sha(p)
        pj = fd / f"v3_seed{s}.json"
        pj.write_text(json.dumps({k: v.tolist() for k, v in stt.items()}), encoding="utf-8")
        files[pj.name] = _sha(pj)
    m = new_model(P, "kan_mix")
    C = collect(P, SEGS)
    aci = run_aci(P, C)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, cwd=load().root).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    code = {f: _sha(load().root / f) for f in ("src/ai_trader/ml/xs_net.py", "src/ai_trader/research/xs_network_study.py", "src/ai_trader/ml/hourly.py", "src/ai_trader/ml/pnn.py")}
    spec = {
        "name": "xs_v3", "version": f"xs_v3-{qtag(q)}", "created": datetime.now(UTC).isoformat(), "git_commit": commit, "code_sha256": code,
        "refit_quarter": qtag(q), "fit_range_us": ck["info"]["fit_range"], "n_fit_hours": ck["info"]["n_fit"],
        "model": {"class": "ml.xs_net.XSNet", **m.cfg, "n_params": xs.n_params(m), "seeds": N_SEEDS, "ensemble": "mean of seed quantiles",
                  "config": ck["cfg"]["kan_mix"]},
        "symbols": P["symbols"].tolist(), "sectors": P["sector"].tolist(), "sector_order": list(xs.SECTOR_ORDER), "sector_index": sec_idx(P).tolist(),
        "features": P["features"].tolist(), "market_features": list(xs.MKT_FEATURES), "optional_features": list(xs.OPTIONAL),
        "standardisation": {"mu": ck["std"]["mu"], "sd": ck["std"]["sd"], "mkt_mu": ck["std"]["mmu"], "mkt_sd": ck["std"]["msd"], "clip": CLIP,
                            "missing": "optional features -> 0 after standardisation; absent tokens masked"},
        "outputs": {"vol": {"horizons_h": list(xs.VOL_H), "taus": list(xs.TAUS), "units": "|R_h| / sqrt(h s2_t), s2_t = sigma_risk_next(bar t-1)^2"},
                    "rel": {"horizons_h": list(xs.REL_H), "taus": list(xs.REL_TAUS), "units": "(R_i - EW mean R) / (sqrt(h) s_t)"},
                    "pairwise": "ml.xs_net.pairwise_prob on raw rel quantiles"},
        "timing": "row for hour t = decision at the open of t (close of bar t-1 + 2 s known_ts); inputs bars <= t-1",
        "min_tokens": xs.MIN_TOKENS, "as_of_data": AS_OF.isoformat(),
        "aci": {"level": REL_LEVEL, "gamma": ACI_GAMMA, "window": ACI_WINDOW, "state": {h: aci[h]["state"] for h in aci}},
        "har_baseline": ck["har"], "files_sha256": files,
        "evidence": "research/reports/xs_network_2026-11-15.md",
    }
    (fd / "spec.json").write_text(json.dumps(spec, indent=1, default=_json_default), encoding="utf-8")
    print("frozen", qtag(q), "->", fd)


# ------------------------------------------------------------------ CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search", "walk", "evaluate", "report", "freeze", "timing"])
    ap.add_argument("--seg", default="llm_in_sample")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "timing":
        timing()
    elif a.cmd == "search":
        search()
    elif a.cmd == "walk":
        run_walk(a.seg)
    elif a.cmd == "evaluate":
        evaluate(a.seg, record=a.record)
    elif a.cmd == "report":
        report()
    elif a.cmd == "freeze":
        freeze()


if __name__ == "__main__":
    main()
