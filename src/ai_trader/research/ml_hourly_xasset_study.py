"""Hourly volatility ML for TradFi perps (H-NN11) and a cross-asset activity block (H-NN12); program 2026-11-12, agent G.
Deterministic research tools only (principle 1). Extends H-NN8 (ml_signals_round2_study) with the same feature builder,
refit schedule, models, bias correction and loss, so the numbers are comparable.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.ml_hourly_xasset_study build
    ... walk --univ T          monthly walk-forward on universe T (checkpoint per refit)
    ... walk --univ L          same for universe L (crypto, the H-NN8 panel)
    ... status
    ... evaluate [--record]    verdicts, Holm within {H-NN11, H-NN12}, research/reports/ml_hourly_xasset_2026-11-12.{json,md}

Design (fixed before evaluation; diary 'H-NN11 / H-NN12 design fixed'):
  Universe T = Bybit TradFi perps whose cached 1h frame (data/research/new_strategies/f1h_<S>.parquet, the H-NS4 cache,
  market_frame as of 2026-11-10) has >= 120 complete UTC days (24 bars). Candidates: ml.hourly_xasset.TRADFI_EQUITY +
  TRADFI_COMMODITY (SPCXUSDT has 116 -> out). Universe L = the H-NN8 panel (data/models/ml/round2/hourly_2026-11-12.parquet,
  23 crypto perps, rows identical to H-NN8), read only.
  T features: ml.hourly.HOURLY_FEATURES (19, symbol_hourly) + cash_sess (equities/ETFs) + cme_sess (commodities and the
  gold tokens) = 21 (ml.hourly_xasset.T_FEATURES). No class dummy, no symbol id. L features: the 19 H-NN8 features.
  Cross-asset block (H-NN12): ml.hourly_xasset.XA_FEATURES (21 = BTC, ETH, QQQ x {|r| at lags 1, 2, 3, 6 h, mean 1-6 h,
  log robust variance, availability}), the target's own copy excluded, missing standardised to the training mean.
  Target, champion, forecast: as H-NN8 (y = log r^2/s2; champion s2 x hour-of-week factor; F = s2 exp(f + c), c the
  validation-block QLIKE-optimal constant).
  Walk-forward: monthly expanding; training rows = bars ending <= month start - 1 day (purge + embargo as H-NN8); first test
  month = the first with >= 365 distinct training dates in the pooled universe (H-NN8 rule). L trains on the H-NN8 random
  1/3 subsample (same seed, same rows); T is small and trains on all rows. Test rows = all bars of the month.
  Models: the frozen H-NN8 configs (data/models/ml/round2/search8.json: MLP (64, 32), wd 1e-2, batch 2048, lr 2e-3, 3 seeds;
  XGB depth 3, mcw 200, eta 0.03, early stopping 50) - no new search, so nothing is tuned on any test month. Ridge-linear
  reference on the same inputs. Per refit two feature sets are fitted on identical rows: base (no block) and xa (+ block).
  Pooled training across the universe: one model per universe and feature set.
  Loss: per-bar QLIKE difference r^2 (1/Fa - 1/Fb) + log(Fa/Fb), mean per UTC date over symbol-bars, NW t over dates;
  segment = bar opens before / after research.knowledge_cutoff (2026-07-01).
  H-NN11 (universe T; base models vs champion): per segment supports if mlp or xgb has diff < 0 and t < -2; contradicts if
  neither has diff < 0 or the better one has t > 2; else inconclusive; a segment with < 30 test dates is inconclusive
  (too short). Pooled (registry) = clean_holdout (the registered criterion). p = min over the two models of 2 Phi(t).
  H-NN12 (xa vs base, same family): components crypto = L minus BTC and ETH ("the other crypto perps"), TradFi = all of T
  (QQQ with its own copy excluded). Each component x segment judged by the H-NN11 rule on (xa_mlp - base_mlp,
  xa_xgb - base_xgb). Registry: llm_in_sample = crypto in-sample; clean_holdout = supports iff crypto and TradFi both
  support (contradicts if either contradicts); pooled = supports iff the same family supports crypto in both segments and
  TradFi supports in clean_holdout (contradicts if any of those three contradicts). p (intersection-union) = max of the
  required components' p.
  Holm per segment over {H-NN11, H-NN12}; a 'supports' with Holm p > 0.05 becomes inconclusive; pooled uses the max of
  the Holm p of its required segments.
  Descriptive: grouped TreeSHAP (|sum of phi| per group per row) of the xa XGB by target class (crypto_alt, btc, eth,
  equity, commodity, gold_token) on up to 2,000 (L) / 5,000 (T) random test rows per month; leader ablation (a leader's
  block at its training mean) per class for xa mlp/xgb; base-model group ablation and SHAP for T (session flags).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import hourly_xasset as hx
from ai_trader.ml import interpret as ip
from ai_trader.ml import models as md
from ai_trader.ml.features import Standardizer
from ai_trader.ml.hourly import HOURLY_FEATURES, HOURLY_GROUPS, complete_hourly, symbol_hourly, train_subsample
from ai_trader.research import ml_signals_study as r1
from ai_trader.research.decision_budget_study import holm, purged_train_mask, test_mask
from ai_trader.research.ml_signals_round2_study import paired8

TAG = "2026-11-12"
SEGS = ("llm_in_sample", "clean_holdout")
HOUR_US = np.timedelta64(3600 * 1_000_000, "us")
MLP_CFG = {"batch": 2048, "lr": 2e-3}
SHAP_ROWS = {"L": 2000, "T": 5000}
MIN_SEG_DATES = 30
FAMS = ("mlp", "xgb", "lin")
T_GROUPS = {**HOURLY_GROUPS, "session": hx.SESSION_FEATURES}
CLASSES = ("crypto_alt", "btc", "eth", "equity", "commodity", "gold_token")
# shared machine (orchestrator 2026-11-12): <= 2 threads, ~1.5 GB per process, pause while free RAM < 6 GB
THREADS = 2
MIN_FREE_GB = 6.0


def _mem() -> tuple[float, float]:
    """(free physical RAM GB, this process's peak working set GB); Windows only, (inf, nan) elsewhere."""
    import ctypes

    try:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (k, ctypes.c_ulonglong) for k in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (k, ctypes.c_size_t) for k in ("peak_ws", "ws", "qpp", "qp", "qnpp", "qnp", "pf", "peak_pf")]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return m.ap / 2**30, c.peak_ws / 2**30
    except Exception:  # noqa: BLE001
        return float("inf"), float("nan")


def _wait_for_ram() -> None:
    while (free := _mem()[0]) < MIN_FREE_GB:
        print(f"free RAM {free:.1f} GB < {MIN_FREE_GB} GB: pausing 60 s", flush=True)
        time.sleep(60)


def root() -> Path:
    d = load().root / "data" / "models" / "ml" / "xasset"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cut() -> str:
    return load().settings["research"]["knowledge_cutoff"]


def _segment(df: pl.DataFrame) -> pl.DataFrame:
    c = pl.lit(_cut()).str.to_datetime().dt.replace_time_zone("UTC")
    return df.with_columns(pl.when(pl.col("event_ts") < c).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"))


def _src(s: str) -> Path:
    return load().root / "data" / "research" / "new_strategies" / f"f1h_{s}.parquet"


def _blocks() -> dict[str, pl.DataFrame]:
    return {s: hx.leader_block(pl.read_parquet(_src(s)), tag) for s, tag in hx.LEADERS.items()}


def _attach(p: pl.DataFrame, blocks) -> pl.DataFrame:
    p = p.with_row_index("_i")
    return hx.attach_leaders(p, blocks).sort("_i").drop("_i")


def build() -> dict:
    blocks = _blocks()
    out: dict = {"T_candidates": {}}
    pT = root() / "panel_T.parquet"
    if not pT.is_file():
        parts = []
        for s in (*hx.TRADFI_EQUITY, *hx.TRADFI_COMMODITY):
            if not _src(s).is_file():
                out["T_candidates"][s] = {"complete_days": 0, "included": False}
                continue
            f = pl.read_parquet(_src(s))
            nd = hx.complete_days(f)
            out["T_candidates"][s] = {"complete_days": nd, "included": nd >= hx.MIN_COMPLETE_DAYS, "first": str(f["event_ts"].min())}
            if nd < hx.MIN_COMPLETE_DAYS:
                continue
            h = symbol_hourly(f)
            h = h.with_columns(*[pl.Series(k, v) for k, v in hx.session_flags(h["event_ts"], s).items()])
            h = complete_hourly(h).with_columns(pl.lit(s).alias("symbol"))
            parts.append(h)
            print(s, nd, h.height, flush=True)
        df = pl.concat(parts).with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC"))).sort("event_ts", "symbol")
        _attach(_segment(df), blocks).write_parquet(pT)
        (root() / "build_T.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    pL = root() / "panel_L.parquet"
    if not pL.is_file():
        src = load().root / "data" / "models" / "ml" / "round2" / f"hourly_{TAG}.parquet"
        _attach(pl.read_parquet(src), blocks).write_parquet(pL)
    return out


def panel(univ: str) -> pl.DataFrame:
    return pl.read_parquet(root() / f"panel_{univ}.parquet")


def base_features(univ: str) -> tuple[str, ...]:
    return hx.T_FEATURES if univ == "T" else HOURLY_FEATURES


def arrays(df: pl.DataFrame, univ: str) -> dict:
    T = df["event_ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    # the block is stored as float32 (memory cap); base features stay float64 as in H-NN8
    Xa = df.select(hx.XA_FEATURES).with_columns(pl.all().cast(pl.Float64)).fill_null(np.nan).to_numpy().astype(np.float32)
    return {"Xb": df.select(base_features(univ)).to_numpy().astype(float), "Xa": Xa, "T": T, "le": T + HOUR_US,
            "y": df["y"].to_numpy().astype(float), "r2": df["r2"].to_numpy().astype(float), "s2": df["s2"].to_numpy().astype(float),
            "seg": df["segment"].to_numpy(), "sym": df["symbol"].to_numpy(),
            "sub": train_subsample(df.height) if univ == "L" else np.ones(df.height, bool)}


def mmap_arrays(univ: str) -> dict:
    """arrays() cached as .npy and memory-mapped (memory cap on the shared machine); row order = the panel's."""
    d = root() / f"arr_{univ}"
    keys = ("Xb", "Xa", "T", "le", "y", "r2", "s2")
    if not all((d / f"{k}.npy").is_file() for k in keys):
        d.mkdir(parents=True, exist_ok=True)
        cols = ["event_ts", "symbol", "segment", "y", "r2", "s2", *base_features(univ), *hx.XA_FEATURES]
        a = arrays(pl.read_parquet(root() / f"panel_{univ}.parquet", columns=cols), univ)
        for k in keys:
            np.save(d / f"{k}.npy", a[k])
        del a
    out = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in keys}
    n = len(out["T"])
    out["sub"] = train_subsample(n) if univ == "L" else np.ones(n, bool)
    return out


def _cfg() -> dict:
    s = json.loads((load().root / "data" / "models" / "ml" / "round2" / "search8.json").read_text(encoding="utf-8"))
    m = dict(s["mlp"]["best"])
    m["hidden"] = tuple(m["hidden"])
    return {"mlp": {**m, **MLP_CFG}, "xgb": dict(s["xgb"]["best"])}


def months_for(a: dict) -> list[np.datetime64]:
    m0 = r1.first_test_month(a["T"], a["le"])
    return [m for m in r1._months(a["T"]) if m >= m0]


def _ckpt(univ: str, m: np.datetime64) -> Path:
    d = root() / univ
    d.mkdir(parents=True, exist_ok=True)
    return d / f"m_{str(m)[:7]}.npz"


def _fit3(Zf, yf, Zv, yv, names, cf):
    ens = md.fit_mlp_ensemble(Zf, yf, Zv, yv, "vol", cfg=cf["mlp"])
    xg = md.fit_xgb(Zf, yf, Zv, yv, "vol", cfg=cf["xgb"], names=names, nthread=THREADS)
    ln = md.fit_linear(Zf, yf, "vol")
    return {"mlp": ens, "xgb": xg, "lin": ln}


def run_walk(univ: str, limit: int | None = None) -> None:
    md.TORCH_THREADS = THREADS  # read by models._torch at every fit
    a = mmap_arrays(univ)
    n_rows = len(a["T"])
    cf = _cfg()
    bnames = list(base_features(univ))
    xnames = [*bnames, *hx.XA_FEATURES]
    nb = len(bnames)
    months = months_for(a)
    if limit:
        months = months[:limit]
    lcols = {g: [nb + hx.XA_FEATURES.index(f) for f in fs] for g, fs in hx.XA_GROUPS.items()}
    bgroups = T_GROUPS if univ == "T" else {}
    bcols = {g: [bnames.index(f) for f in fs] for g, fs in bgroups.items()}
    print(f"{univ}: {n_rows} rows, {len(months)} months from {str(months[0])[:7]}", flush=True)
    for m in months:
        ck = _ckpt(univ, m)
        if ck.is_file():
            continue
        _wait_for_ram()
        t0 = time.time()
        te = np.flatnonzero(test_mask(a["T"], m, r1._next_month(m)))
        if not len(te):
            continue
        tr = np.flatnonzero(purged_train_mask(a["le"], m) & a["sub"])
        fl, vl = md.val_split(a["T"][tr])
        fi, vi = tr[fl], tr[vl]
        sb = Standardizer().fit(a["Xb"][fi])
        sx = hx.BlockStd().fit(a["Xa"][fi])

        def Z(idx, xa: bool):
            zb = sb.transform(a["Xb"][idx])
            return np.hstack([zb, sx.transform(a["Xa"][idx])]) if xa else zb

        ls_v, ls_t = np.log(a["s2"][vi]), np.log(a["s2"][te])
        res: dict[str, np.ndarray] = {"idx": te}
        info = {"month": str(m)[:7], "n_train": int(len(tr)), "n_test": int(len(te))}
        rng = np.random.default_rng(int(str(m)[:7].replace("-", "")))
        pick_pos = np.sort(rng.choice(len(te), size=min(SHAP_ROWS[univ], len(te)), replace=False))
        res["shap_pos"] = pick_pos
        for fs, xa in (("base", False), ("xa", True)):
            Zf, Zv, Zt = Z(fi, xa), Z(vi, xa), Z(te, xa)
            mods = _fit3(Zf, a["y"][fi], Zv, a["y"][vi], xnames if xa else bnames, cf)
            info[f"{fs}_mlp_epochs"] = [x.epochs for x in mods["mlp"].members]
            info[f"{fs}_xgb_iter"] = mods["xgb"].best_iteration
            for fam, mod in mods.items():
                c = md.bias_const(a["r2"][vi], ls_v + mod.predict(Zv))
                res[f"F_{fs}_{fam}"] = np.exp(ls_t + mod.predict(Zt) + c)
                if fam == "lin":
                    continue
                for g, cols in (lcols if xa else bcols).items():
                    Za = Zt.copy()
                    Za[:, cols] = 0.0
                    res[f"abl_{fs}_{fam}_{g}"] = np.exp(ls_t + mod.predict(Za) + c)
            res[f"shap_{fs}"] = ip.shap_main(mods["xgb"], Zt[pick_pos]).astype(np.float32)
        info["secs"] = round(time.time() - t0, 1)
        info["threads"] = THREADS
        info["free_gb"], info["peak_ws_gb"] = (round(x, 2) for x in _mem())
        tmp = ck.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, info=np.array(json.dumps(info)), **res)
        os.replace(tmp, ck)
        print(json.dumps(info), flush=True)


def status() -> None:
    for u in ("T", "L"):
        d = root() / u
        n = len(list(d.glob("m_*.npz"))) if d.is_dir() else 0
        print(u, n, "checkpoints")


# ------------------------------------------------------------------ collect and evaluate


def collect(univ: str) -> tuple[pl.DataFrame, list[dict], dict]:
    """Predictions table (test rows) + per-month info + SHAP rows (per feature set: phi matrix and row indices)."""
    df = pl.read_parquet(root() / f"panel_{univ}.parquet", columns=["symbol", "event_ts", "segment", "r2", "s2", "fac"])
    n = df.height
    cols: dict[str, np.ndarray] = {}
    infos, shap = [], {"base": [], "xa": [], "idx": []}
    for ck in sorted((root() / univ).glob("m_*.npz")):
        z = np.load(ck)
        idx = z["idx"]
        for k in z.files:
            if k.startswith(("F_", "abl_")):
                cols.setdefault(k, np.full(n, np.nan))[idx] = z[k]
        infos.append(json.loads(str(z["info"])))
        shap["idx"].append(idx[z["shap_pos"]])
        shap["base"].append(z["shap_base"])
        shap["xa"].append(z["shap_xa"])
    shap = {k: np.concatenate(v) for k, v in shap.items()}
    out = df.select("symbol", "event_ts", "segment", "r2", "s2", "fac").with_columns(
        (pl.col("s2") * pl.col("fac")).alias("F_champ"), pl.col("s2").alias("F_s2"), *[pl.Series(k, v) for k, v in cols.items()])
    out = out.with_columns(pl.col("symbol").map_elements(hx.target_class, return_dtype=pl.Utf8).alias("cls"))
    keep = np.isfinite(cols["F_xa_xgb"])
    return out.filter(pl.Series(keep)), infos, shap


def _phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2)) if np.isfinite(t) else 1.0


def judge(pair: dict[str, dict], n_dates: int) -> tuple[str, float]:
    """H-NN8 rule over the mlp / xgb rows of `pair`; < MIN_SEG_DATES dates -> inconclusive."""
    nl = {k: pair[k] for k in ("mlp", "xgb")}
    p = float(min(1.0, min(2 * _phi(v["t"]) for v in nl.values())))
    if n_dates < MIN_SEG_DATES:
        return "inconclusive", p
    if any(v["diff"] < 0 and v["t"] < -2 for v in nl.values()):
        return "supports", p
    best = min(nl.values(), key=lambda z: z["diff"])
    if all(not (v["diff"] < 0) for v in nl.values()) or best["t"] > 2:
        return "contradicts", p
    return "inconclusive", p


def _sub(q: pl.DataFrame, **flt) -> pl.DataFrame:
    for k, v in flt.items():
        q = q.filter(pl.col(k).is_in(v) if isinstance(v, (list, tuple)) else (pl.col(k) == v))
    return q


def per_symbol(q: pl.DataFrame, a: str, b: str) -> dict:
    out = {}
    for s in sorted(q["symbol"].unique().to_list()):
        r = paired8(q.filter(pl.col("symbol") == s), a, b)
        out[s] = {"diff": r["diff"], "t": r["t"], "n_dates": r["n_dates"]}
    return out


def nn11(pT: pl.DataFrame) -> dict:
    out: dict = {}
    for s in SEGS:
        q = pT.filter(pl.col("segment") == s)
        if q.height == 0:
            out[s] = {"verdict": "inconclusive", "p": 1.0, "n_dates": 0}
            continue
        r = {fam: paired8(q, f"F_base_{fam}", "F_champ") for fam in FAMS}
        r["champ_vs_s2"] = paired8(q, "F_champ", "F_s2")
        r["xgb_vs_lin"] = paired8(q, "F_base_xgb", "F_base_lin")
        r["mlp_vs_xgb"] = paired8(q, "F_base_mlp", "F_base_xgb")
        r["mlp_vs_lin"] = paired8(q, "F_base_mlp", "F_base_lin")
        nd = r["xgb"]["n_dates"]
        r["verdict"], r["p"] = judge(r, nd)
        r["n_dates"], r["n_symbols"] = nd, q["symbol"].n_unique()
        r["by_class"] = {c: {fam: paired8(_sub(q, cls=c), f"F_base_{fam}", "F_champ") for fam in FAMS} | {
            "champ_vs_s2": paired8(_sub(q, cls=c), "F_champ", "F_s2")} for c in sorted(q["cls"].unique().to_list())}
        r["per_symbol_xgb"] = per_symbol(q, "F_base_xgb", "F_champ")
        r["per_symbol_mlp"] = per_symbol(q, "F_base_mlp", "F_champ")
        r["group_ablation"] = {}
        for fam in ("mlp", "xgb"):
            for g in T_GROUPS:
                k = f"abl_base_{fam}_{g}"
                rr = paired8(q.filter(pl.col(k).is_finite()), k, f"F_base_{fam}")
                r["group_ablation"][f"{fam}:{g}"] = {"dqlike": rr["diff"], "t": rr["t"]}
        out[s] = r
    out["pooled_verdict"] = out["clean_holdout"]["verdict"]
    return out


def xa_component(q: pl.DataFrame) -> dict:
    if q.height == 0:
        return {"verdict": "inconclusive", "p": 1.0, "n_dates": 0}
    r = {fam: paired8(q, f"F_xa_{fam}", f"F_base_{fam}") for fam in FAMS}
    r["xa_xgb_vs_champ"] = paired8(q, "F_xa_xgb", "F_champ")
    r["xa_mlp_vs_champ"] = paired8(q, "F_xa_mlp", "F_champ")
    r["verdict"], r["p"] = judge(r, r["xgb"]["n_dates"])
    r["n_dates"], r["n_symbols"] = r["xgb"]["n_dates"], q["symbol"].n_unique()
    return r


def leader_ablation(q: pl.DataFrame) -> dict:
    out = {}
    for fam in ("mlp", "xgb"):
        for g in hx.XA_GROUPS:
            k = f"abl_xa_{fam}_{g}"
            rr = paired8(q.filter(pl.col(k).is_finite()), k, f"F_xa_{fam}")
            out[f"{fam}:{g}"] = {"dqlike": rr["diff"], "t": rr["t"], "n_dates": rr["n_dates"]}
    return out


def nn12(pL: pl.DataFrame, pT: pl.DataFrame) -> dict:
    out: dict = {"crypto": {}, "tradfi": {}, "descriptive": {}}
    crypto = pL.filter(~pl.col("symbol").is_in(["BTCUSDT", "ETHUSDT"]))
    for s in SEGS:
        out["crypto"][s] = xa_component(crypto.filter(pl.col("segment") == s))
        out["tradfi"][s] = xa_component(pT.filter(pl.col("segment") == s))
        d = {}
        for name, q in (("btc", pL.filter(pl.col("symbol") == "BTCUSDT")), ("eth", pL.filter(pl.col("symbol") == "ETHUSDT")),
                        ("tradfi_ex_qqq", pT.filter(pl.col("symbol") != "QQQUSDT")), ("qqq", pT.filter(pl.col("symbol") == "QQQUSDT")),
                        *[(c, pT.filter(pl.col("cls") == c)) for c in ("equity", "commodity", "gold_token")]):
            qq = q.filter(pl.col("segment") == s)
            if qq.height:
                d[name] = {fam: paired8(qq, f"F_xa_{fam}", f"F_base_{fam}") for fam in ("mlp", "xgb")}
        out["descriptive"][s] = d
        if s == "clean_holdout":
            out["crypto"][s]["per_symbol_xgb"] = per_symbol(crypto.filter(pl.col("segment") == s), "F_xa_xgb", "F_base_xgb")
            out["tradfi"][s]["per_symbol_xgb"] = per_symbol(pT.filter(pl.col("segment") == s), "F_xa_xgb", "F_base_xgb")
            out["tradfi"][s]["per_symbol_mlp"] = per_symbol(pT.filter(pl.col("segment") == s), "F_xa_mlp", "F_base_mlp")
    cr, tf = out["crypto"], out["tradfi"]
    out["seg_verdict"] = {"llm_in_sample": cr["llm_in_sample"]["verdict"]}
    vs = (cr["clean_holdout"]["verdict"], tf["clean_holdout"]["verdict"])
    out["seg_verdict"]["clean_holdout"] = "supports" if all(v == "supports" for v in vs) else ("contradicts" if "contradicts" in vs else "inconclusive")
    out["seg_p"] = {"llm_in_sample": cr["llm_in_sample"]["p"], "clean_holdout": max(cr["clean_holdout"]["p"], tf["clean_holdout"]["p"])}
    same = [f for f in ("mlp", "xgb") if all(cr[s][f]["diff"] < 0 and cr[s][f]["t"] < -2 for s in SEGS)]
    req = (cr["llm_in_sample"]["verdict"], cr["clean_holdout"]["verdict"], tf["clean_holdout"]["verdict"])
    out["crypto_same_family_both_segments"] = same
    out["pooled_verdict"] = ("supports" if same and tf["clean_holdout"]["verdict"] == "supports"
                             else ("contradicts" if "contradicts" in req else "inconclusive"))
    out["pooled_p"] = max(cr["llm_in_sample"]["p"], cr["clean_holdout"]["p"], tf["clean_holdout"]["p"])
    return out


def shap_groups(univ: str, pr_full: pl.DataFrame, shap: dict) -> dict:
    """Grouped TreeSHAP (mean |sum phi in group| per row), by segment x class, for the xa and base XGB."""
    df = pl.read_parquet(root() / f"panel_{univ}.parquet", columns=["symbol", "segment"])
    seg = df["segment"].to_numpy()[shap["idx"]]
    cls = np.array([hx.target_class(s) for s in df["symbol"].to_numpy()[shap["idx"]]])
    bnames = list(base_features(univ))
    groups_b = T_GROUPS if univ == "T" else HOURLY_GROUPS
    out: dict = {}
    for fs, names, groups in (("xa", [*bnames, *hx.XA_FEATURES], {**groups_b, **hx.XA_GROUPS}), ("base", bnames, groups_b)):
        phi = shap[fs][:, : len(names)].astype(float)
        G = {g: np.abs(phi[:, [names.index(f) for f in fs_]].sum(1)) for g, fs_ in groups.items()}
        res: dict = {}
        for s in SEGS:
            for c in sorted(set(cls)):
                mk = (seg == s) & (cls == c)
                if mk.sum() < 50:
                    continue
                mean = {g: float(v[mk].mean()) for g, v in G.items()}
                tot = sum(mean.values())
                r = {"n_rows": int(mk.sum()), "share": {g: mean[g] / tot for g in mean}, "mean_abs": mean}
                if fs == "xa":
                    blk = {g: mean[g] for g in hx.XA_GROUPS}
                    bt = sum(blk.values())
                    r["block_share"] = bt / tot
                    r["within_block"] = {g: blk[g] / bt if bt > 0 else float("nan") for g in blk}
                res[f"{s}:{c}"] = r
        out[fs] = res
    return out


def _walk_info(infos: list[dict]) -> dict:
    return {"n_months": len(infos), "first": infos[0]["month"], "last": infos[-1]["month"], "secs_total": sum(i["secs"] for i in infos),
            "n_train_last": infos[-1]["n_train"], "mean_xgb_iter_base": float(np.mean([i["base_xgb_iter"] for i in infos])),
            "mean_xgb_iter_xa": float(np.mean([i["xa_xgb_iter"] for i in infos])),
            "mean_epochs_base": float(np.mean([np.mean(i["base_mlp_epochs"]) for i in infos])),
            "mean_epochs_xa": float(np.mean([np.mean(i["xa_mlp_epochs"]) for i in infos])), "months": infos}


def replication_check(pL: pl.DataFrame) -> dict:
    """Our base L run against the H-NN8 walk-forward (same rows / configs; refit noise only)."""
    p = load().root / "data" / "models" / "ml" / "round2" / "wf8_preds.parquet"
    if not p.is_file():
        return {}
    w = pl.read_parquet(p, columns=["symbol", "event_ts", "F_xgb", "F_mlp"]).rename({"F_xgb": "F8_xgb", "F_mlp": "F8_mlp"})
    j = pL.join(w, on=["symbol", "event_ts"], how="inner")
    out = {"n": j.height}
    for fam in ("xgb", "mlp"):
        out[f"corr_log_{fam}"] = float(np.corrcoef(np.log(j[f"F_base_{fam}"].to_numpy()), np.log(j[f"F8_{fam}"].to_numpy()))[0, 1])
        for s in SEGS:
            out[f"{s}:{fam}_vs_champ_hnn8"] = paired8(j.filter(pl.col("segment") == s), f"F8_{fam}", "F_champ")
            out[f"{s}:{fam}_vs_champ_ours"] = paired8(j.filter(pl.col("segment") == s), f"F_base_{fam}", "F_champ")
    return out


def run_evaluate(record_rows: bool = False) -> dict:
    pT, infT, shT = collect("T")
    pL, infL, shL = collect("L")
    bT = json.loads((root() / "build_T.json").read_text(encoding="utf-8"))
    out: dict = {"cutoff": _cut(), "T_candidates": bT["T_candidates"], "t_features": list(hx.T_FEATURES), "xa_features": list(hx.XA_FEATURES),
                 "xa_groups": {k: list(v) for k, v in hx.XA_GROUPS.items()}, "cfg": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                                                                                  for k, v in _cfg().items()},
                 "walk_T": _walk_info(infT), "walk_L": _walk_info(infL)}
    for u, p in (("T", pT), ("L", pL)):
        out[f"panel_{u}"] = {"rows_tested": p.height, "symbols": sorted(p["symbol"].unique().to_list()),
                             "rows_by_segment": dict(p.group_by("segment").len().iter_rows()),
                             "dates_by_segment": {s: int(p.filter(pl.col("segment") == s)["event_ts"].dt.date().n_unique()) for s in SEGS},
                             "first": str(p["event_ts"].min()), "last": str(p["event_ts"].max())}
    out["H-NN11"] = nn11(pT)
    out["H-NN12"] = nn12(pL, pT)
    out["attribution"] = {"T": shap_groups("T", pT, shT), "L": shap_groups("L", pL, shL)}
    out["leader_ablation"] = {s: {**{f"L:{c}": leader_ablation(pL.filter((pl.col("segment") == s) & (pl.col("cls") == c))) for c in ("crypto_alt", "btc", "eth")},
                                  **{f"T:{c}": leader_ablation(pT.filter((pl.col("segment") == s) & (pl.col("cls") == c)))
                                     for c in ("equity", "commodity", "gold_token") if pT.filter((pl.col("segment") == s) & (pl.col("cls") == c)).height}}
                              for s in SEGS}
    out["replication_hnn8"] = replication_check(pL)
    # Holm per segment over the two hypotheses
    p11 = {s: out["H-NN11"][s]["p"] for s in SEGS}
    p12 = out["H-NN12"]["seg_p"]
    out["holm"] = {s: {"p": {"H-NN11": p11[s], "H-NN12": p12[s]}} for s in SEGS}
    for s in SEGS:
        out["holm"][s]["p_holm"] = holm(out["holm"][s]["p"])
    raw = {"H-NN11": {s: out["H-NN11"][s]["verdict"] for s in SEGS} | {"pooled": out["H-NN11"]["pooled_verdict"]},
           "H-NN12": dict(out["H-NN12"]["seg_verdict"]) | {"pooled": out["H-NN12"]["pooled_verdict"]}}
    out["raw_verdicts"] = raw
    ver = {}
    for h in ("H-NN11", "H-NN12"):
        v = {s: ("inconclusive" if raw[h][s] == "supports" and out["holm"][s]["p_holm"][h] > 0.05 else raw[h][s]) for s in SEGS}
        segs_p = ("clean_holdout",) if h == "H-NN11" else SEGS
        pv = raw[h]["pooled"]
        if pv == "supports" and max(out["holm"][s]["p_holm"][h] for s in segs_p) > 0.05:
            pv = "inconclusive"
        v["pooled"] = pv
        ver[h] = v
    out["verdicts"] = ver
    rep = load().root / "research" / "reports"
    (rep / f"ml_hourly_xasset_{TAG}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    from ai_trader.research.ml_hourly_xasset_report import build as build_md

    (rep / f"ml_hourly_xasset_{TAG}.md").write_text(build_md(out), encoding="utf-8")
    print(json.dumps(ver, indent=1), "free / peak GB", _mem())
    if record_rows:
        record_all(out)
    return out


DATA11 = ("bybit 1h bars, universe T (Bybit TradFi perps with >= 120 complete days), cached H-NS4 frames "
          "data/research/new_strategies/f1h_*.parquet (market_frame as of 2026-11-10); ai_trader.ml.hourly + ai_trader.ml.hourly_xasset; "
          "src/ai_trader/research/ml_hourly_xasset_study.py; data/models/ml/xasset/")
DATA12 = ("bybit 1h bars, universe L (H-NN8 panel, 23 crypto perps) and universe T; leaders BTC/ETH/QQQ from the same cache; "
          "ai_trader.ml.hourly_xasset; src/ai_trader/research/ml_hourly_xasset_study.py; data/models/ml/xasset/")


def _slim(r: dict) -> dict:
    return {k: r[k] for k in ("mlp", "xgb", "lin") if k in r} | {k: r[k] for k in ("n_dates", "n_symbols", "verdict", "p") if k in r}


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record, results

    v = out["verdicts"]
    for h in ("H-NN11", "H-NN12"):
        done = {r["segment"] for r in results(h)}
        for s in (*SEGS, "pooled"):
            if s in done:
                print(f"skip {h} {s}")
                continue
            if h == "H-NN11":
                r = out[h]
                if s == "pooled":
                    m = {"clean_holdout": _slim(r["clean_holdout"]), "llm_in_sample": _slim(r["llm_in_sample"])}
                    note = f"= clean_holdout (registered criterion); raw {out['raw_verdicts'][h][s]}"
                else:
                    m = _slim(r[s]) | {"champ_vs_s2": r[s].get("champ_vs_s2"), "by_class": r[s].get("by_class"),
                                       "p_holm": out["holm"][s]["p_holm"][h]}
                    note = ("next-hour QLIKE, H-NN8 models (frozen configs) + cash/CME session flags vs robust_bc x hour-of-week; pooled over "
                            f"universe T, monthly expanding refits; per-bar diff averaged per date, NW t; raw {out['raw_verdicts'][h][s]}")
                    if s == "llm_in_sample":
                        note += "; in-sample is short (TradFi perps listed 2026-04..06; equities enter the test set only after 8 weeks of factor history)"
                record(h, s, v[h][s], m, note=note, data=DATA11)
            else:
                r = out[h]
                if s == "pooled":
                    m = {"crypto": {x: _slim(r["crypto"][x]) for x in SEGS}, "tradfi_clean_holdout": _slim(r["tradfi"]["clean_holdout"]),
                         "crypto_same_family_both_segments": r["crypto_same_family_both_segments"], "p": r["pooled_p"]}
                    note = ("registered criterion: crypto (L minus BTC, ETH) diff < 0, t < -2 in both segments (same family) and TradFi (T) in "
                            f"clean_holdout; raw {out['raw_verdicts'][h][s]}")
                else:
                    m = {"crypto": _slim(r["crypto"][s]), "tradfi": _slim(r["tradfi"][s]), "p": r["seg_p"][s], "p_holm": out["holm"][s]["p_holm"][h]}
                    note = ("paired next-hour QLIKE with vs without the BTC/ETH/QQQ block (same family, same rows, same refits); "
                            + ("segment verdict = crypto component (TradFi in-sample reported only)" if s == "llm_in_sample"
                               else "segment verdict requires crypto and TradFi") + f"; raw {out['raw_verdicts'][h][s]}")
                record(h, s, v[h][s], m, note=note, data=DATA12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "walk", "status", "evaluate"))
    ap.add_argument("--univ", choices=("T", "L"), default="T")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch

    md.TORCH_THREADS = THREADS
    torch.set_num_threads(THREADS)
    if a.cmd == "build":
        print(json.dumps(build(), indent=1))
        for u in ("T", "L"):
            p = panel(u)
            print(u, p.shape, dict(p.group_by("segment").len().iter_rows()), p["symbol"].n_unique())
    elif a.cmd == "walk":
        run_walk(a.univ, a.limit or None)
    elif a.cmd == "status":
        status()
    else:
        run_evaluate(a.record)


if __name__ == "__main__":
    main()
