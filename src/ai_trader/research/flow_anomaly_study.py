"""H-OF1..H-OF5 (research/hypotheses.yaml, block "Program 2026-11-10"): order-flow anomaly detectors.

All data (flow_3s from 2026-11-07, liquidations from 2026-11-08) is after the knowledge cutoff, so every result is
segment clean_holdout: a short first sample (about 3 days, one market state); the forward record continues.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.flow_anomaly_study --as-of 2026-11-10T22:30:00Z [--record]

Tests, as registered (the choices that the registry leaves open are fixed in METHOD and stated in the report):
  H-OF1  |z_imbalance| > 4 minutes: mean forward |return| (5m, 15m) vs matched non-anomaly minutes (same symbol,
         same UTC hour of day); ratio of sums, 1h block bootstrap. Supports: pooled ratio >= 1.5 at both
         horizons with 95% CI excluding 1.
  H-OF2  same events: sign rate of the next-5m/15m mid return vs the imbalance sign, mean signed return in bp,
         t clustered by calendar hour. Supports: sign rate > 55% and mean > 1 bp with t > 2, both horizons.
  H-OF3  liquidation clusters (1m notional > trailing-24h 99th percentile and >= 3 prints), declustered 30 min per
         symbol; return signed against the liquidated side, sigma units; t clustered by hour. Supports: 60m mean
         > 0 with t > 2.
  H-OF4  robust Mahalanobis over 7 z features, fitted on the first half (calendar time), AUC for
         |r_15m| > 2 sigma_15m on the second half vs the best single |z|; 1h block bootstrap of the difference.
  H-OF5  P(alt anomaly in t+1..t+3 | BTC anomaly at t) / unconditional 3-minute rate, and the reverse direction.
Holm over the five primary p-values (alpha 0.05) is required on top of each success criterion for "supports".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from ai_trader.features import flow_anomaly as fa

HORIZONS = (5, 15, 60)
B = 2000
SEED = 20261110
ALPHA = 0.05
MIN_EVENTS = 30
TRADFI = {
    "SOXLUSDT", "QQQUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT", "MUUSDT", "GOOGLUSDT", "AAPLUSDT", "INTCUSDT", "SPCXUSDT",
    "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "USDJPYUSDT", "EURUSDUSDT", "SPYUSDT", "IWMUSDT", "TLTUSDT", "SMHUSDT",
    "XLEUSDT", "GDXUSDT", "UVXYUSDT", "IBITUSDT", "CRWVUSDT", "IRENUSDT", "CORZUSDT", "COINUSDT", "HOODUSDT", "MARAUSDT",
    "CRCLUSDT", "AMDSTOCKUSDT", "AVGOUSDT", "TSMUSDT", "METAUSDT", "MSFTUSDT", "ORCLUSDT", "PAXGUSDT",
}
NOT_ALT = TRADFI | {"BTCUSDT", "XAUTUSDT"}  # H-OF5 alts: crypto perps other than BTC and tokenised gold
METHOD = {
    "z": "robust z vs trailing 1440 1m bars (t-1440..t-1), >= 360 valid (collector gaps); scale 1.4826*MAD, fallback 1.2533*meanAD when MAD = 0, floored at 1% of |median|; 3 s candles finalised > 60 s late dropped",
    "features": "imbalance = buy_notional - sell_notional; trades = n_buy + n_sell; volume = buy+sell notional; spread = TWA spread_bps; depth = depth_bid_25 + depth_ask_25; abs_ret = |1m log mid return|; liq = 1m liquidation notional",
    "anomaly": "|z_imbalance| > 4 (H-OF1, H-OF2, H-OF5)",
    "forward": "log(mid_{t+h} / mid_t), mid at the close of minute t (entry after the anomaly minute is known)",
    "sigma": "sigma_h = sqrt(h * trailing-1440 mean of squared 1m mid returns)",
    "warmup": "first 1440 minutes of each symbol dropped (no trailing window); liquidation features need 1440 minutes after 2026-11-08 00:46 (collection start); minutes in collector outages are null (never zero) for flow and liquidations",
    "blocks": "calendar-hour blocks pooled across symbols (cross-sectional dependence kept inside a block), B = 2000",
    "universe": "Bybit linear perps in flow_3s (spot streams excluded); asset class TradFi = the TradFi stream groups in settings.flow.streams",
    "verdict": "supports = registered criterion met and Holm-adjusted p < 0.05; contradicts = criterion fails and the 95% CI excludes the required effect; else inconclusive; < 30 events -> inconclusive",
}


# ------------------------------------------------------------------------------------------ data

def _symbols(lake, as_of: datetime) -> list[str]:
    return fa.flow_symbols(lake, as_of, "linear")


def _liq_start(lake, as_of: datetime) -> datetime | None:
    return fa.liquidation_start(lake, as_of)


def build_symbol(sym: str, as_of_iso: str, liq_start_iso: str | None, cache_dir: str, min_valid: int = fa.MIN_VALID,
                 rel_floor: float = fa.REL_FLOOR) -> str:
    """Read one symbol point in time, build the per-minute panel with forward outcomes; cache to parquet."""
    from ai_trader.storage.lake import Lake

    as_of = datetime.fromisoformat(as_of_iso)
    out = Path(cache_dir) / f"{sym}.parquet"
    if out.is_file():
        return str(out)
    lake = Lake()
    lake.con.execute("SET memory_limit = '1GB'")  # shared machine: keep each worker small
    lake.con.execute("SET threads = 2")
    df3 = lake.read_asof("flow_3s", as_of, symbols=[sym], filters={"category": "linear", "venue": "bybit"}, columns=fa.FLOW_COLUMNS)
    liq = lake.read_asof("liquidations", as_of, symbols=[sym], filters={"venue": "bybit"})
    p = fa.build_panel(df3, liq, liq_start=datetime.fromisoformat(liq_start_iso) if liq_start_iso else None,
                       min_valid=min_valid, rel_floor=rel_floor)
    # only complete minutes as of as_of
    p = p.filter(pl.col("event_ts") + timedelta(minutes=1) <= as_of)
    lm = pl.col("mid").log()
    p = p.with_columns([(lm.shift(-h) - lm).alias(f"fwd_{h}") for h in HORIZONS])
    p.write_parquet(out)
    return str(out)


def load_panel(as_of: datetime, cache_dir: Path, workers: int = 4, symbols: list[str] | None = None,
               min_valid: int = fa.MIN_VALID, rel_floor: float = fa.REL_FLOOR) -> pl.DataFrame:
    from ai_trader.storage.lake import Lake

    lake = Lake()
    syms = symbols or _symbols(lake, as_of)
    ls = _liq_start(lake, as_of)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        k = len(syms)
        paths = list(ex.map(build_symbol, syms, [as_of.isoformat()] * k, [ls.isoformat() if ls else None] * k, [str(cache_dir)] * k,
                            [min_valid] * k, [rel_floor] * k))
    sys.stdout.write(f"built {len(paths)} symbols in {time.time() - t:.0f}s\n")
    p = pl.concat([pl.read_parquet(x) for x in paths], how="diagonal_relaxed")
    return p.with_columns(
        pl.col("event_ts").dt.hour().alias("hod"),
        pl.col("event_ts").dt.truncate("1h").alias("block"),
        pl.col("symbol").is_in(list(TRADFI)).alias("tradfi"),
    )


# ------------------------------------------------------------------------------------------ stats

def _boot_counts(n_blocks: int, rng: np.random.Generator, b: int = B) -> np.ndarray:
    """(b, n_blocks) multinomial resampling counts of blocks."""
    return rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=b).astype(float)


def _block_ids(blocks: pl.Series) -> tuple[np.ndarray, int]:
    u, inv = np.unique(blocks.cast(pl.Int64).to_numpy(), return_inverse=True)
    return inv, len(u)


def ratio_boot(num: np.ndarray, den: np.ndarray, blocks: np.ndarray, nb: int, rng) -> dict[str, float]:
    """Ratio of sums with a block bootstrap; p = share of replicates with ratio <= 1 (one-sided)."""
    sn, sd = np.bincount(blocks, num, nb), np.bincount(blocks, den, nb)
    w = _boot_counts(nb, rng)
    r = (w @ sn) / (w @ sd)
    est = sn.sum() / sd.sum()
    return {"ratio": float(est), "ci95": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))],
            "p_le_1": float((np.sum(r <= 1) + 1) / (len(r) + 1)), "n": int(len(num)), "blocks": int(np.count_nonzero(np.bincount(blocks, minlength=nb)))}


def cluster_t(x: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    """Mean with a cluster-robust SE (clusters = calendar hour)."""
    ok = np.isfinite(x)
    x, c = x[ok], clusters[ok]
    n = len(x)
    if n < 3:
        return {"mean": float(np.mean(x)) if n else float("nan"), "se": float("nan"), "t": float("nan"), "n": n, "clusters": 0}
    m = x.mean()
    _, inv = np.unique(c, return_inverse=True)
    g = np.bincount(inv, x - m)
    G = len(g)
    se = math.sqrt(G / max(G - 1, 1) * np.sum(g ** 2)) / n
    return {"mean": float(m), "se": float(se), "t": float(m / se) if se > 0 else float("nan"), "n": int(n), "clusters": int(G)}


def _auc_prep(score: np.ndarray):
    u, inv = np.unique(score, return_inverse=True)
    return inv, len(u)


def auc_w(inv: np.ndarray, ng: int, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Mann-Whitney AUC (ties count 1/2) for score groups `inv` (sorted ascending)."""
    wp = np.bincount(inv, w * y, ng)
    wn = np.bincount(inv, w * (1 - y), ng)
    below = np.cumsum(wn) - wn
    den = wp.sum() * wn.sum()
    return float(np.sum(wp * (below + 0.5 * wn)) / den) if den > 0 else float("nan")


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=lambda k: p[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def _p_from_t(t: float) -> float:
    return float(1 - norm.cdf(t)) if np.isfinite(t) else 1.0


# ------------------------------------------------------------------------------------- hypotheses

def h_of1(p: pl.DataFrame, rng, flag: str = "anom_z4") -> dict[str, Any]:
    out: dict[str, Any] = {"horizons": {}}
    for h in (5, 15):
        d = p.filter(pl.col(flag).is_not_null() & pl.col(f"fwd_{h}").is_not_null()).with_columns(
            pl.col(f"fwd_{h}").abs().alias("y"), pl.col(flag).alias("anom"))
        base = d.filter(~pl.col("anom")).group_by("symbol", "hod").agg(pl.col("y").mean().alias("base"), pl.len().alias("n_base"))
        ev = d.filter(pl.col("anom")).join(base, on=["symbol", "hod"], how="inner").filter(pl.col("base") > 0)
        inv, nb = _block_ids(ev["block"])
        pooled = ratio_boot(ev["y"].to_numpy(), ev["base"].to_numpy(), inv, nb, rng)
        pooled["median_event_ratio"] = float((ev["y"] / ev["base"]).median()) if ev.height else float("nan")
        pooled["anomaly_share"] = float(d["anom"].mean())
        by_class = {}
        for cls, g in ev.sort("tradfi").group_by("tradfi", maintain_order=True):
            i2, n2 = _block_ids(g["block"])
            by_class["tradfi" if cls[0] else "crypto"] = ratio_boot(g["y"].to_numpy(), g["base"].to_numpy(), i2, n2, rng)
        per_sym = []
        for (sym,), g in ev.sort("symbol").group_by("symbol", maintain_order=True):
            if g.height < 5:
                per_sym.append({"symbol": sym, "n": g.height, "ratio": float(g["y"].sum() / g["base"].sum())})
                continue
            i2, n2 = _block_ids(g["block"])
            r = ratio_boot(g["y"].to_numpy(), g["base"].to_numpy(), i2, n2, rng)
            per_sym.append({"symbol": sym, "n": g.height, "ratio": r["ratio"], "ci95": r["ci95"]})
        out["horizons"][str(h)] = {"pooled": pooled, "by_class": by_class, "per_symbol": sorted(per_sym, key=lambda r: -r["n"])}
    hz = out["horizons"]
    ok = all(hz[k]["pooled"]["ratio"] >= 1.5 and hz[k]["pooled"]["ci95"][0] > 1 for k in ("5", "15"))
    ruled_out = any(hz[k]["pooled"]["ci95"][1] < 1.5 for k in ("5", "15"))
    n = min(hz[k]["pooled"]["n"] for k in ("5", "15"))
    out["criterion_met"], out["ruled_out"], out["n_events"] = ok, ruled_out, n
    out["p"] = max(hz[k]["pooled"]["p_le_1"] for k in ("5", "15"))
    return out


def h_of2(p: pl.DataFrame, rng, flag: str = "anom_z4") -> dict[str, Any]:
    out: dict[str, Any] = {"horizons": {}}
    for h in (5, 15):
        ev = p.filter(pl.col(flag).fill_null(False) & pl.col(f"fwd_{h}").is_not_null()).with_columns(
            (pl.col("z_imbalance").sign() * pl.col(f"fwd_{h}") * 1e4).alias("s_bp"),
            (pl.col("z_imbalance").sign() * pl.col(f"fwd_{h}") / (pl.col("sigma_1m") * math.sqrt(h))).alias("s_sig"))
        inv, nb = _block_ids(ev["block"])
        s = ev["s_bp"].to_numpy()
        nz = s != 0
        agree = (s > 0).astype(float)
        sn, sd = np.bincount(inv, agree * nz, nb), np.bincount(inv, nz.astype(float), nb)
        w = _boot_counts(nb, rng)
        sr_b = (w @ sn) / (w @ sd)
        sr = float(sn.sum() / sd.sum()) if sd.sum() else float("nan")
        t_bp = cluster_t(s, inv)
        t_sig = cluster_t(ev["s_sig"].to_numpy(), inv)
        by_class = {("tradfi" if k[0] else "crypto"): {**cluster_t(g["s_bp"].to_numpy(), _block_ids(g["block"])[0]),
                                                       "sign_rate": float((g["s_bp"] > 0).sum() / max((g["s_bp"] != 0).sum(), 1))}
                    for k, g in ev.sort("tradfi").group_by("tradfi", maintain_order=True)}
        per_sym = []
        for (sym,), g in ev.sort("symbol").group_by("symbol", maintain_order=True):
            r = cluster_t(g["s_bp"].to_numpy(), _block_ids(g["block"])[0])
            per_sym.append({"symbol": sym, "n": g.height, "mean_bp": r["mean"], "t": r["t"],
                            "sign_rate": float((g["s_bp"] > 0).sum() / max((g["s_bp"] != 0).sum(), 1))})
        out["horizons"][str(h)] = {
            "sign_rate": sr, "sign_rate_ci95": [float(np.nanquantile(sr_b, 0.025)), float(np.nanquantile(sr_b, 0.975))],
            "zero_return_share": float(1 - nz.mean()) if len(s) else float("nan"),
            "mean_bp": t_bp, "mean_sigma": t_sig, "by_class": by_class, "per_symbol": sorted(per_sym, key=lambda r: -r["n"]),
        }
    hz = out["horizons"]
    ok = all(hz[k]["sign_rate"] > 0.55 and hz[k]["mean_bp"]["mean"] > 1 and hz[k]["mean_bp"]["t"] > 2 for k in ("5", "15"))
    ruled_out = any(hz[k]["sign_rate_ci95"][1] < 0.55 or hz[k]["mean_bp"]["mean"] + 1.96 * hz[k]["mean_bp"]["se"] < 1 for k in ("5", "15"))
    out["criterion_met"], out["ruled_out"] = ok, ruled_out
    out["n_events"] = min(hz[k]["mean_bp"]["n"] for k in ("5", "15"))
    out["p"] = max(_p_from_t(hz[k]["mean_bp"]["t"]) for k in ("5", "15"))
    return out


def decluster(ev: pl.DataFrame, minutes: int) -> pl.DataFrame:
    keep = []
    for (_sym,), g in ev.sort("symbol", "event_ts").group_by("symbol", maintain_order=True):
        last = None
        for i, t in enumerate(g["event_ts"]):
            if last is None or t >= last + timedelta(minutes=minutes):
                keep.append(g[i])
                last = t
    return pl.concat(keep) if keep else ev.head(0)


def h_of3(p: pl.DataFrame, rng) -> dict[str, Any]:
    raw = p.filter(pl.col("liq_cluster"))
    ev = decluster(raw, 30)
    sign = pl.when(pl.col("liq_side") == "long").then(1.0).otherwise(-1.0)
    out: dict[str, Any] = {"n_raw_clusters": raw.height, "n_declustered": ev.height, "horizons": {},
                           "side_counts": {k: int(v) for k, v in ev["liq_side"].value_counts().iter_rows()} if ev.height else {}}
    for h in (15, 60):
        e = ev.filter(pl.col(f"fwd_{h}").is_not_null() & (pl.col("sigma_1m") > 0)).with_columns(
            (sign * pl.col(f"fwd_{h}") / (pl.col("sigma_1m") * math.sqrt(h))).alias("s_sig"), (sign * pl.col(f"fwd_{h}") * 1e4).alias("s_bp"))
        inv = _block_ids(e["block"])[0] if e.height else np.array([], dtype=int)
        by_side = {k[0]: cluster_t(g["s_sig"].to_numpy(), _block_ids(g["block"])[0]) for k, g in e.sort("liq_side").group_by("liq_side", maintain_order=True)} if e.height else {}
        per_sym = [{"symbol": k[0], "n": g.height, "mean_sigma": float(g["s_sig"].mean()), "mean_bp": float(g["s_bp"].mean())}
                   for k, g in e.sort("symbol").group_by("symbol", maintain_order=True)] if e.height else []
        out["horizons"][str(h)] = {"mean_sigma": cluster_t(e["s_sig"].to_numpy(), inv), "mean_bp": cluster_t(e["s_bp"].to_numpy(), inv),
                                   "hit_rate": float((e["s_sig"] > 0).mean()) if e.height else float("nan"),
                                   "by_side": by_side, "per_symbol": sorted(per_sym, key=lambda r: -r["n"])}
    m = out["horizons"]["60"]["mean_sigma"]
    out["criterion_met"] = bool(m["mean"] > 0 and m["t"] > 2)
    out["ruled_out"] = bool(np.isfinite(m["se"]) and m["mean"] + 1.96 * m["se"] < 0)
    out["n_events"], out["p"] = m["n"], _p_from_t(m["t"])
    return out


def h_of4(p: pl.DataFrame, rng, model_path: Path | None) -> dict[str, Any]:
    zc = [f"z_{c}" for c in fa.MAHA_FEATURES]
    d = p.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in zc]) & pl.col("fwd_15").is_not_null() & (pl.col("sigma_1m") > 0))
    d = d.with_columns((pl.col("fwd_15").abs() > 2 * pl.col("sigma_1m") * math.sqrt(15)).cast(pl.Float64).alias("y"))
    split = d["event_ts"].sort()[d.height // 2]
    tr, te = d.filter(pl.col("event_ts") < split), d.filter(pl.col("event_ts") >= split)
    xtr = tr.select(zc).to_numpy()
    model = fa.RobustMahalanobis().fit(xtr)
    xte = np.clip(np.nan_to_num(te.select(zc).to_numpy(), posinf=model.clip, neginf=-model.clip), -model.clip, model.clip)
    y = te["y"].to_numpy()
    scores = {"mahalanobis": model.score(te.select(zc).to_numpy())}
    for c, col in zip(fa.MAHA_FEATURES, range(len(zc)), strict=True):
        scores[c] = np.abs(xte[:, col])
    # training-half AUCs pick the best single feature without looking at the test half (robustness)
    ytr = (tr["fwd_15"].abs() > 2 * tr["sigma_1m"] * math.sqrt(15)).cast(pl.Float64).to_numpy()
    xtr_c = np.clip(np.nan_to_num(xtr, posinf=model.clip, neginf=-model.clip), -model.clip, model.clip)
    auc_train = {c: auc_w(*_auc_prep(np.abs(xtr_c[:, i])), ytr, np.ones(len(ytr))) for i, c in enumerate(fa.MAHA_FEATURES)}
    auc_train["mahalanobis"] = auc_w(*_auc_prep(model.score(xtr)), ytr, np.ones(len(ytr)))
    inv_b, nb = _block_ids(te["block"])
    prep = {k: _auc_prep(v) for k, v in scores.items()}
    one = np.ones(len(y))
    auc = {k: auc_w(*prep[k], y, one) for k in scores}
    best = max((k for k in scores if k != "mahalanobis"), key=lambda k: auc[k])
    best_train = max((k for k in auc_train if k != "mahalanobis"), key=lambda k: auc_train[k])
    w_blocks = _boot_counts(nb, rng, 1000)
    bs = {k: np.empty(len(w_blocks)) for k in scores}
    for j, wb in enumerate(w_blocks):
        w = wb[inv_b]
        for k in scores:
            bs[k][j] = auc_w(*prep[k], y, w)
    diff = bs["mahalanobis"] - bs[best]
    diff_max = bs["mahalanobis"] - np.max(np.column_stack([bs[k] for k in scores if k != "mahalanobis"]), axis=1)
    diff_tr = bs["mahalanobis"] - bs[best_train]
    res = {
        "split_ts": split.isoformat(), "n_train": tr.height, "n_test": te.height, "base_rate_test": float(y.mean()),
        "base_rate_train": float(ytr.mean()), "test_symbols": te["symbol"].n_unique(), "test_blocks": nb,
        "auc_test": {k: {"auc": auc[k], "ci95": [float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))]} for k in scores},
        "auc_train": auc_train, "best_single_test": best, "best_single_train": best_train,
        "diff_vs_best_test": {"diff": auc["mahalanobis"] - auc[best], "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
                              "p_le_0": float((np.sum(diff <= 0) + 1) / (len(diff) + 1))},
        "diff_vs_max_per_replicate": {"ci95": [float(np.quantile(diff_max, 0.025)), float(np.quantile(diff_max, 0.975))]},
        "diff_vs_best_train": {"diff": auc["mahalanobis"] - auc[best_train], "ci95": [float(np.quantile(diff_tr, 0.025)), float(np.quantile(diff_tr, 0.975))]},
        "model": {"shrinkage": model.shrinkage, "location": model.location.tolist(), "train_quantiles": model.train_quantiles,
                  "corr": np.round(model.covariance / np.sqrt(np.outer(np.diag(model.covariance), np.diag(model.covariance))), 3).tolist()},
    }
    dd = res["diff_vs_best_test"]
    res["criterion_met"] = bool(dd["diff"] > 0 and dd["ci95"][0] > 0)
    res["ruled_out"] = bool(dd["ci95"][1] < 0)
    res["n_events"], res["p"] = int(y.sum()), dd["p_le_0"]
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**model.to_json(), "fit_window": [tr["event_ts"].min().isoformat(), split.isoformat()], "version": "v1",
                "source": "research/flow_anomaly_study.py (H-OF4 first-half fit)", "z_window_min": fa.WINDOW}
        model_path.write_text(json.dumps(meta, indent=1))
    return res


def h_of5(p: pl.DataFrame, rng, flag: str = "anom_z4") -> dict[str, Any]:
    d = p.filter(pl.col("symbol").is_in(list(NOT_ALT - {"BTCUSDT"})).not_())
    piv_z = (d.select("event_ts", "symbol", pl.col(flag).cast(pl.Float64).alias("f"))
             .pivot(on="symbol", index="event_ts", values="f").sort("event_ts"))
    full = pl.DataFrame({"event_ts": pl.datetime_range(piv_z["event_ts"].min(), piv_z["event_ts"].max(), "1m", eager=True, time_zone="UTC").cast(piv_z["event_ts"].dtype)})
    piv_z = full.join(piv_z, on="event_ts", how="left")
    ts = piv_z["event_ts"]
    alts = [c for c in piv_z.columns if c not in ("event_ts", "BTCUSDT")]
    Z = piv_z.select(["BTCUSDT", *alts]).cast(pl.Float64).fill_null(np.nan).to_numpy()
    V = ~np.isnan(Z)  # flag null = not scored (warm-up / gap); +-inf z counts as scored
    A = V & (np.nan_to_num(Z) > 0.5)
    T = len(Z)
    # next-3 window (t+1..t+3): any anomaly, all three minutes valid
    NA = np.zeros_like(A)
    NV = np.zeros_like(V)
    NV[: T - 3] = V[1:T - 2] & V[2:T - 1] & V[3:T]
    NA[: T - 3] = A[1:T - 2] | A[2:T - 1] | A[3:T]
    blocks = ts.dt.truncate("1h").cast(pl.Int64).to_numpy()
    _, binv = np.unique(blocks, return_inverse=True)
    nb = binv.max() + 1
    b_ev, b_val, b_na, b_nv = A[:, 0], V[:, 0], NA[:, 0], NV[:, 0]
    a_ev, a_val, a_na, a_nv = A[:, 1:], V[:, 1:], NA[:, 1:], NV[:, 1:]
    # BTC -> alt
    c_n = (b_ev[:, None] & a_nv).sum(axis=1)
    c_h = (b_ev[:, None] & a_nv & a_na).sum(axis=1)
    u_n = (b_val[:, None] & a_nv).sum(axis=1)
    u_h = (b_val[:, None] & a_nv & a_na).sum(axis=1)
    # alt -> BTC
    r_n = (a_ev & b_nv[:, None]).sum(axis=1)
    r_h = (a_ev & (b_nv & b_na)[:, None]).sum(axis=1)
    ru_n = (a_val & b_nv[:, None]).sum(axis=1)
    ru_h = (a_val & (b_nv & b_na)[:, None]).sum(axis=1)
    agg = {k: np.bincount(binv, v.astype(float), nb) for k, v in dict(c_n=c_n, c_h=c_h, u_n=u_n, u_h=u_h, r_n=r_n, r_h=r_h, ru_n=ru_n, ru_h=ru_h).items()}

    def lifts(w):
        s = {k: w @ v for k, v in agg.items()} if w.ndim == 2 else {k: float(np.dot(w, v)) for k, v in agg.items()}
        fwd = (s["c_h"] / s["c_n"]) / (s["u_h"] / s["u_n"])
        rev = (s["r_h"] / s["r_n"]) / (s["ru_h"] / s["ru_n"])
        return fwd, rev, s

    fwd, rev, s = lifts(np.ones(nb))
    W = _boot_counts(nb, rng)
    fb, rb, _ = lifts(W)
    diff = fb - rb
    # per-lag and same-minute co-occurrence (descriptive)
    lag = {}
    for k in (0, 1, 2, 3):
        av = V[k:, 1:] & b_ev[: T - k, None] & V[: T - k, 0][:, None]
        hit = (av & A[k:, 1:]).sum()
        un = (V[k:, 1:] & V[: T - k, 0][:, None])
        lag[str(k)] = {"p_cond": float(hit / max(av.sum(), 1)), "p_uncond": float((un & A[k:, 1:]).sum() / max(un.sum(), 1))}
        lag[str(k)]["lift"] = lag[str(k)]["p_cond"] / lag[str(k)]["p_uncond"] if lag[str(k)]["p_uncond"] > 0 else float("nan")
    per_alt = []
    for j, a in enumerate(alts):
        n1 = int((b_ev & a_nv[:, j]).sum())
        if n1 == 0:
            continue
        pc = (b_ev & a_nv[:, j] & a_na[:, j]).sum() / n1
        pu = (b_val & a_nv[:, j] & a_na[:, j]).sum() / max((b_val & a_nv[:, j]).sum(), 1)
        per_alt.append({"symbol": a, "n_btc_events": n1, "p_cond": float(pc), "p_uncond": float(pu), "lift": float(pc / pu) if pu > 0 else float("nan")})
    q = lambda x: [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]  # noqa: E731
    res = {
        "n_btc_anomalies": int(b_ev.sum()), "n_alt_anomalies": int(a_ev.sum()), "n_alts": len(alts), "blocks": int(nb),
        "btc_to_alt": {"p_cond": float(s["c_h"] / s["c_n"]), "p_uncond": float(s["u_h"] / s["u_n"]), "lift": float(fwd), "ci95": q(fb),
                       "p_le_1": float((np.sum(fb <= 1) + 1) / (len(fb) + 1)), "pairs": int(s["c_n"])},
        "alt_to_btc": {"p_cond": float(s["r_h"] / s["r_n"]), "p_uncond": float(s["ru_h"] / s["ru_n"]), "lift": float(rev), "ci95": q(rb), "pairs": int(s["r_n"])},
        "lift_diff": {"diff": float(fwd - rev), "ci95": q(diff), "p_le_0": float((np.sum(diff <= 0) + 1) / (len(diff) + 1))},
        "by_lag_btc_to_alt": lag, "per_alt": sorted(per_alt, key=lambda r: -r["lift"] if np.isfinite(r["lift"]) else 0),
    }
    f5 = res["btc_to_alt"]
    res["criterion_met"] = bool(f5["lift"] >= 2 and f5["ci95"][0] > 1 and fwd > rev)
    res["ruled_out"] = bool(f5["ci95"][1] < 2 or res["lift_diff"]["ci95"][1] < 0)
    res["n_events"] = res["n_btc_anomalies"]
    res["p"] = max(f5["p_le_1"], res["lift_diff"]["p_le_0"])
    return res


# ------------------------------------------------------------------------------------------- run

def verdict(r: dict[str, Any], p_holm: float) -> str:
    if r["n_events"] < MIN_EVENTS:
        return "inconclusive"
    if r["criterion_met"] and p_holm < ALPHA:
        return "supports"
    if not r["criterion_met"] and r["ruled_out"]:
        return "contradicts"
    return "inconclusive"


def sample_summary(p: pl.DataFrame) -> dict[str, Any]:
    ok = p.filter(pl.col("z_imbalance").is_not_null())
    return {
        "symbols": p["symbol"].n_unique(), "symbols_scored": ok["symbol"].n_unique(),
        "tradfi_symbols": ok.filter(pl.col("tradfi"))["symbol"].n_unique(),
        "minutes_total": p.height, "minutes_scored": ok.height,
        "first_minute": p["event_ts"].min().isoformat(), "first_scored": ok["event_ts"].min().isoformat(),
        "last_minute": p["event_ts"].max().isoformat(),
        "gap_minutes": int(p["n3s"].is_null().sum()),
        "imbalance_anomalies": int((ok["z_imbalance"].abs() > fa.Z_THRESH).sum()),
        "imbalance_anomaly_share": float((ok["z_imbalance"].abs() > fa.Z_THRESH).mean()),
        "inf_z_imbalance": int(ok["z_imbalance"].is_infinite().sum()),
        "liq_clusters": int(p["liq_cluster"].sum()),
        "anomaly_share_by_feature": {c: float((ok[f"z_{c}"].abs() > fa.Z_THRESH).mean()) for c in fa.Z_FEATURES},
        "firing_rate_per_symbol": {r[0]: float(r[1]) for r in ok.group_by("symbol").agg((pl.col("z_imbalance").abs() > fa.Z_THRESH).mean()).sort("symbol").iter_rows()},
    }


def run(as_of: datetime, cache_dir: Path, workers: int = 4, model_path: Path | None = None, symbols: list[str] | None = None,
        min_valid: int = fa.MIN_VALID, rel_floor: float = fa.REL_FLOOR, exploratory: bool = True) -> dict[str, Any]:
    p = load_panel(as_of, cache_dir, workers, symbols, min_valid, rel_floor)
    rng = np.random.default_rng(SEED)
    res = {"as_of": as_of.isoformat(), "segment": "clean_holdout", "method": METHOD, "spec": {"min_valid": min_valid, "rel_floor": rel_floor,
           "max_lag_s": fa.MAX_LAG_S, "window": fa.WINDOW, "z_thresh": fa.Z_THRESH}, "sample": sample_summary(p)}
    p = p.with_columns(pl.when(pl.col("z_imbalance").is_null()).then(None).otherwise(pl.col("z_imbalance").abs() > fa.Z_THRESH).alias("anom_z4"))
    res["H-OF1"] = h_of1(p, rng)
    res["H-OF2"] = h_of2(p, rng)
    res["H-OF3"] = h_of3(p, rng)
    res["H-OF4"] = h_of4(p, rng, model_path)
    res["H-OF5"] = h_of5(p, rng)
    raw = {k: res[k]["p"] for k in ("H-OF1", "H-OF2", "H-OF3", "H-OF4", "H-OF5")}
    adj = holm(raw)
    fr = np.array(list(res["sample"]["firing_rate_per_symbol"].values()))
    firing = {"pooled": res["sample"]["imbalance_anomaly_share"], "median_symbol": float(np.median(fr)),
              "p10_symbol": float(np.quantile(fr, 0.1)), "p90_symbol": float(np.quantile(fr, 0.9))}
    res["firing_rate_z4"] = firing
    for k in raw:
        res[k]["p_raw"], res[k]["p_holm"] = raw[k], adj[k]
        res[k]["verdict"] = verdict(res[k], adj[k])
    if exploratory:
        res["exploratory_quantile"] = exploratory_quantile(p, np.random.default_rng(SEED + 1))
    return res


def exploratory_quantile(p: pl.DataFrame, rng) -> dict[str, Any]:
    """EXPLORATORY, not a registered test and never recorded as a verdict: anomaly = |z_imbalance| above the
    symbol's own q-quantile over the whole scored sample (uses the full sample, so it is NOT causal), q in
    {0.99, 0.999}; the H-OF1 / H-OF2 / H-OF5 statistics re-computed with that flag."""
    out = {}
    for q in (0.99, 0.999):
        thr = p.filter(pl.col("z_imbalance").is_not_null() & pl.col("z_imbalance").is_finite()).group_by("symbol").agg(
            pl.col("z_imbalance").abs().quantile(q).alias("thr"))
        flag = f"anom_q{q}"
        pq = p.join(thr, on="symbol", how="left").with_columns(
            pl.when(pl.col("z_imbalance").is_null()).then(None).otherwise(pl.col("z_imbalance").abs() > pl.col("thr")).alias(flag))
        o1, o2, o5 = h_of1(pq, rng, flag), h_of2(pq, rng, flag), h_of5(pq, rng, flag)
        out[str(q)] = {
            "median_threshold_abs_z": float(thr["thr"].median()),
            "H-OF1": {h: {k: o1["horizons"][h]["pooled"][k] for k in ("ratio", "ci95", "n")} for h in ("5", "15")},
            "H-OF2": {h: {"sign_rate": o2["horizons"][h]["sign_rate"], "sign_rate_ci95": o2["horizons"][h]["sign_rate_ci95"],
                          "mean_bp": o2["horizons"][h]["mean_bp"]["mean"], "t": o2["horizons"][h]["mean_bp"]["t"],
                          "n": o2["horizons"][h]["mean_bp"]["n"]} for h in ("5", "15")},
            "H-OF5": {"btc_to_alt": {k: o5["btc_to_alt"][k] for k in ("lift", "ci95", "p_cond", "p_uncond")},
                      "alt_to_btc": {k: o5["alt_to_btc"][k] for k in ("lift", "ci95")}, "lift_diff": o5["lift_diff"],
                      "n_btc_anomalies": o5["n_btc_anomalies"]},
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="research/reports/flow_anomaly_2026-11-10.json")
    ap.add_argument("--model", default=str(fa.MODEL_PATH))
    ap.add_argument("--symbols", default="", help="comma list (debug); default: every linear flow_3s symbol")
    ap.add_argument("--min-valid", type=int, default=fa.MIN_VALID)
    ap.add_argument("--rel-floor", type=float, default=fa.REL_FLOOR)
    ap.add_argument("--no-exploratory", action="store_true")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC)
    cache = Path(a.cache) if a.cache else Path("data/cache/flow_anomaly") / as_of.strftime("%Y%m%dT%H%M")
    syms = [x for x in a.symbols.split(",") if x] or None
    res = run(as_of, cache, min(a.workers, 4), Path(a.model) if a.model else None, syms, a.min_valid, a.rel_floor, not a.no_exploratory)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for k in ("H-OF1", "H-OF2", "H-OF3", "H-OF4", "H-OF5"):
        sys.stdout.write(f"{k}: {res[k]['verdict']} (p_raw {res[k]['p_raw']:.4f}, holm {res[k]['p_holm']:.4f}, n {res[k]['n_events']})\n")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------ report

def _f(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    if isinstance(x, float):
        return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def markdown_tables(res: dict[str, Any], sens: dict[str, Any] | None = None, top: int = 15) -> str:
    """Mechanical tables for the report (the narrative, caveats and rule candidates are written by hand)."""
    s, L = res["sample"], []
    L += ["## Sample", "", "| item | value |", "|---|---|"]
    for k in ("symbols", "symbols_scored", "tradfi_symbols", "minutes_total", "minutes_scored", "gap_minutes", "first_minute",
              "first_scored", "last_minute", "imbalance_anomalies", "imbalance_anomaly_share", "inf_z_imbalance", "liq_clusters"):
        L.append(f"| {k} | {_f(s[k], 4)} |")
    L += ["", "Share of scored minutes with |robust z| > 4, by feature: " + ", ".join(f"{k} {v:.3f}" for k, v in s["anomaly_share_by_feature"].items()), ""]
    L += ["## Verdicts (clean_holdout; Holm over the 5 OF hypotheses)", "", "| id | primary statistic | p raw | p Holm | verdict |", "|---|---|---|---|---|"]
    o1, o2, o3, o4, o5 = (res[k] for k in ("H-OF1", "H-OF2", "H-OF3", "H-OF4", "H-OF5"))
    h1 = o1["horizons"]
    h2 = o2["horizons"]
    m60, m15 = o3["horizons"]["60"]["mean_sigma"], o3["horizons"]["15"]["mean_sigma"]
    prim = {
        "H-OF1": "; ".join(f"{h}m ratio {_f(h1[h]['pooled']['ratio'])} CI {_f(h1[h]['pooled']['ci95'])} (n {h1[h]['pooled']['n']})" for h in ("5", "15")),
        "H-OF2": "; ".join(f"{h}m sign {_f(h2[h]['sign_rate'], 3)}, mean {_f(h2[h]['mean_bp']['mean'])} bp (t {_f(h2[h]['mean_bp']['t'])})" for h in ("5", "15")),
        "H-OF3": f"60m signed {_f(m60['mean'], 3)} sigma (t {_f(m60['t'])}, n {m60['n']}); 15m {_f(m15['mean'], 3)} (t {_f(m15['t'])})",
        "H-OF4": (f"AUC maha {_f(o4['auc_test']['mahalanobis']['auc'], 3)} vs best single ({o4['best_single_test']}) "
                  f"{_f(o4['auc_test'][o4['best_single_test']]['auc'], 3)}; diff {_f(o4['diff_vs_best_test']['diff'], 3)} CI {_f(o4['diff_vs_best_test']['ci95'], 3)}"),
        "H-OF5": (f"BTC->alt lift {_f(o5['btc_to_alt']['lift'])} CI {_f(o5['btc_to_alt']['ci95'])}; alt->BTC {_f(o5['alt_to_btc']['lift'])} "
                  f"CI {_f(o5['alt_to_btc']['ci95'])}; diff CI {_f(o5['lift_diff']['ci95'])}"),
    }
    for k, txt in prim.items():
        L.append(f"| {k} | {txt} | {_f(res[k]['p_raw'], 4)} | {_f(res[k]['p_holm'], 4)} | **{res[k]['verdict']}** |")
    if sens:
        L += ["", f"Sensitivity, original spec (min_valid {sens['spec']['min_valid']}, rel_floor {sens['spec']['rel_floor']}): "
              + ", ".join(f"{k} {sens[k]['verdict']} (p Holm {_f(sens[k]['p_holm'], 3)})" for k in prim) + "."]
    # H-OF1
    L += ["", "## H-OF1: forward |return| after |z_imbalance| > 4 vs matched minutes", "",
          "| horizon | class | ratio | 95% CI | n | median event ratio |", "|---|---|---|---|---|---|"]
    for h in ("5", "15"):
        hp = h1[h]
        L.append(f"| {h}m | pooled | {_f(hp['pooled']['ratio'])} | {_f(hp['pooled']['ci95'])} | {hp['pooled']['n']} | {_f(hp['pooled']['median_event_ratio'])} |")
        for c, v in sorted(hp["by_class"].items()):
            L.append(f"| {h}m | {c} | {_f(v['ratio'])} | {_f(v['ci95'])} | {v['n']} | |")
    L += ["", f"Per symbol, 15m (top {top} by events; all in the JSON):", "", "| symbol | n | ratio | 95% CI |", "|---|---|---|---|"]
    L += [f"| {r['symbol']} | {r['n']} | {_f(r['ratio'])} | {_f(r.get('ci95'))} |" for r in h1["15"]["per_symbol"][:top]]
    ps = [r["ratio"] for r in h1["15"]["per_symbol"] if r["n"] >= 10 and np.isfinite(r["ratio"])]
    if ps:
        L += ["", f"Symbols with >= 10 events: {len(ps)}; 15m ratio > 1 in {sum(x > 1 for x in ps)}, >= 1.5 in {sum(x >= 1.5 for x in ps)}; median {np.median(ps):.2f}."]
    # H-OF2
    L += ["", "## H-OF2: direction after |z_imbalance| > 4", "",
          "| horizon | sign rate | 95% CI | zero-return share | mean bp | t (hour-clustered) | mean sigma | t |", "|---|---|---|---|---|---|---|---|"]
    for h in ("5", "15"):
        v = h2[h]
        L.append(f"| {h}m | {_f(v['sign_rate'], 3)} | {_f(v['sign_rate_ci95'], 3)} | {_f(v['zero_return_share'], 3)} | {_f(v['mean_bp']['mean'])} | "
                 f"{_f(v['mean_bp']['t'])} | {_f(v['mean_sigma']['mean'], 3)} | {_f(v['mean_sigma']['t'])} |")
    L += ["", "| horizon | class | n | sign rate | mean bp | t |", "|---|---|---|---|---|---|"]
    for h in ("5", "15"):
        for c, v in sorted(h2[h]["by_class"].items()):
            L.append(f"| {h}m | {c} | {v['n']} | {_f(v['sign_rate'], 3)} | {_f(v['mean'])} | {_f(v['t'])} |")
    L += ["", f"Per symbol, 15m (top {top} by events):", "", "| symbol | n | sign rate | mean bp | t |", "|---|---|---|---|---|"]
    L += [f"| {r['symbol']} | {r['n']} | {_f(r['sign_rate'], 3)} | {_f(r['mean_bp'])} | {_f(r['t'])} |" for r in h2["15"]["per_symbol"][:top]]
    # H-OF3
    L += ["", "## H-OF3: liquidation clusters (declustered 30 min), return signed against the liquidated side", "",
          f"Raw cluster minutes {o3['n_raw_clusters']}, declustered events {o3['n_declustered']}, by side {o3['side_counts']}.", "",
          "| horizon | mean sigma | t | n | hour clusters | mean bp | t bp | hit rate |", "|---|---|---|---|---|---|---|---|"]
    for h in ("15", "60"):
        v = o3["horizons"][h]
        L.append(f"| {h}m | {_f(v['mean_sigma']['mean'], 3)} | {_f(v['mean_sigma']['t'])} | {v['mean_sigma']['n']} | {v['mean_sigma']['clusters']} | "
                 f"{_f(v['mean_bp']['mean'])} | {_f(v['mean_bp']['t'])} | {_f(v['hit_rate'], 3)} |")
    L += ["", "| horizon | liquidated side | mean sigma | t | n |", "|---|---|---|---|---|"]
    for h in ("15", "60"):
        for side, v in sorted(o3["horizons"][h]["by_side"].items()):
            L.append(f"| {h}m | {side} | {_f(v['mean'], 3)} | {_f(v['t'])} | {v['n']} |")
    L += ["", f"Per symbol, 60m (top {top} by events):", "", "| symbol | n | mean sigma | mean bp |", "|---|---|---|---|"]
    L += [f"| {r['symbol']} | {r['n']} | {_f(r['mean_sigma'], 3)} | {_f(r['mean_bp'])} |" for r in o3["horizons"]["60"]["per_symbol"][:top]]
    # H-OF4
    L += ["", "## H-OF4: multivariate robust Mahalanobis vs single features (AUC for |r_15m| > 2 sigma_15m)", "",
          f"Fit on minutes before {o4['split_ts']} (n {o4['n_train']}, base rate {o4['base_rate_train']:.3f}); test n {o4['n_test']} "
          f"(base rate {o4['base_rate_test']:.3f}, {o4['test_symbols']} symbols, {o4['test_blocks']} hour blocks). "
          f"Ledoit-Wolf shrinkage {o4['model']['shrinkage']:.3f}.", "",
          "| score | test AUC | 95% CI | train AUC |", "|---|---|---|---|"]
    for k, v in sorted(o4["auc_test"].items(), key=lambda kv: -kv[1]["auc"]):
        L.append(f"| {k} | {_f(v['auc'], 3)} | {_f(v['ci95'], 3)} | {_f(o4['auc_train'].get(k), 3)} |")
    L += ["", f"Difference vs the best single feature on the test half ({o4['best_single_test']}): {_f(o4['diff_vs_best_test']['diff'], 3)} "
          f"CI {_f(o4['diff_vs_best_test']['ci95'], 3)}; vs the best per replicate: CI {_f(o4['diff_vs_max_per_replicate']['ci95'], 3)}; "
          f"vs the feature best on the training half ({o4['best_single_train']}): {_f(o4['diff_vs_best_train']['diff'], 3)} "
          f"CI {_f(o4['diff_vs_best_train']['ci95'], 3)}."]
    # H-OF5
    L += ["", "## H-OF5: BTC-led vs alt-led flow anomalies (next 3 minutes)", "",
          f"BTC anomalies {o5['n_btc_anomalies']}, alt anomalies {o5['n_alt_anomalies']} over {o5['n_alts']} alts, {o5['blocks']} hour blocks.", "",
          "| direction | P(cond) | P(uncond) | lift | 95% CI | pairs |", "|---|---|---|---|---|---|"]
    for k in ("btc_to_alt", "alt_to_btc"):
        v = o5[k]
        L.append(f"| {k} | {_f(v['p_cond'], 3)} | {_f(v['p_uncond'], 3)} | {_f(v['lift'])} | {_f(v['ci95'])} | {v['pairs']} |")
    L += ["", f"Lift difference (BTC->alt minus alt->BTC): {_f(o5['lift_diff']['diff'], 3)} CI {_f(o5['lift_diff']['ci95'], 3)}.", "",
          "By lag (BTC anomaly at t, alt anomaly at exactly t+k): "
          + ", ".join(f"k={k}: lift {_f(v['lift'])}" for k, v in o5["by_lag_btc_to_alt"].items()) + "."]
    pa = [r for r in o5["per_alt"] if r["n_btc_events"] >= 20 and np.isfinite(r["lift"])]
    if pa:
        L += ["", f"Per-alt lifts (alts with >= 20 BTC events): median {np.median([r['lift'] for r in pa]):.2f}, "
              f">= 2 in {sum(r['lift'] >= 2 for r in pa)} of {len(pa)}."]
    return "\n".join(L) + "\n"
