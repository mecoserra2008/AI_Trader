"""Shadow xs.v4: the xs.v3 network over every liquid Bybit crypto perp (block XS4, H-XS4; research agent, 2026-11-16).

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.xs_v4_study build     panel from the REST cache
    ... walk        purged quarterly expanding walk-forward (kan_mix, 3 seeds, warm start), checkpoint per quarter
    ... evaluate    walk-forward IC (4h / 24h) of v4 vs the low-vol ranking, vol pinball vs panel log-HAR-t, by segment
    ... freeze      final refit on every labelled hour (warm start from the last quarter) -> data/models/xs_v4/frozen

Same design as xs_network_study (docstring there): token_rows / assemble unchanged; v3's chosen hyper-parameters (no new
search); 14 sector embeddings (config/live_xs_v4.yaml). New: a point-in-time liquidity mask. Token i is present at hour t
only if its turnover over bars t-24..t-1 (1h klines) >= 20 M USD; the mask is applied before assemble by nulling a core
feature, so the market token, the relative basket and the targets all use the liquid set of that hour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.ml import xs_net as xs
from ai_trader.research import xs_network_study as study
from ai_trader.research.xs_v4_data import cfg as v4cfg
from ai_trader.research.xs_v4_data import root as data_root

AS_OF = datetime(2026, 11, 16, 1, tzinfo=UTC)
HOUR_US, DAY_US = study.HOUR_US, study.DAY_US
N_SEEDS = 3
SEGS = study.SEGS


def mroot() -> Path:
    p = load().root / "data" / "models" / "xs_v4"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sector_order() -> tuple[str, ...]:
    return tuple(v4cfg()["model"]["sectors"])


def sectors(symbols: list[str]) -> list[str]:
    so = sector_order()
    return [s if s in so else "other" for s in study.sectors(symbols)]


# ------------------------------------------------------------------ frames from the REST cache


def frame(sym: str, as_of: datetime | None = None) -> pl.DataFrame:
    """market_frame-equivalent (FRAME_COLS + turnover) from the REST cache, point in time at as_of."""
    from ai_trader.research.frame import _asof_join, add_robust_vol, enrich_bars

    raw = data_root() / "raw"
    k = pl.read_parquet(raw / f"kline_{sym}.parquet")
    ts = pl.col("ts").cast(pl.Datetime("ms", "UTC")).cast(pl.Datetime("us", "UTC"))
    b = k.with_columns(ts.alias("event_ts")).with_columns((pl.col("event_ts") + pl.duration(hours=1)).alias("close_ts"))
    b = b.with_columns((pl.col("close_ts") + pl.duration(seconds=2)).alias("known_ts"))
    if as_of is not None:
        b = b.filter(pl.col("known_ts") <= as_of)
    b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close")) & (pl.col("low") <= pl.min_horizontal("open", "close")))
    if b.height == 0:
        return b
    start = datetime.fromisoformat(v4cfg()["history_start"]).replace(tzinfo=UTC)
    first = b["event_ts"].min()
    if first > start + timedelta(days=1):           # listed inside the window: listing burn-in as market_frame (30 days)
        b = b.filter(pl.col("event_ts") >= first + timedelta(days=load().settings["research"]["burn_in_days_new"]))
    if b.height < 40:
        return b.head(0)
    b = b.sort("event_ts").select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")
    df = add_robust_vol(enrich_bars(b))
    fpath, opath = raw / f"funding_{sym}.parquet", raw / f"oi_{sym}.parquet"
    f = pl.read_parquet(fpath) if fpath.exists() else pl.DataFrame(schema={"ts": pl.Int64, "rate": pl.Float64})
    f = f.with_columns((pl.col("ts").cast(pl.Datetime("ms", "UTC")).cast(pl.Datetime("us", "UTC")) + pl.duration(seconds=2)).alias("known_ts")).rename({"rate": "funding_rate"})
    o = pl.read_parquet(opath) if opath.exists() else pl.DataFrame(schema={"ts": pl.Int64, "oi": pl.Float64})
    o = o.with_columns((pl.col("ts").cast(pl.Datetime("ms", "UTC")).cast(pl.Datetime("us", "UTC")) + pl.duration(hours=4, seconds=2)).alias("known_ts")).rename({"oi": "open_interest"})
    if as_of is not None:
        f, o = f.filter(pl.col("known_ts") <= as_of), o.filter(pl.col("known_ts") <= as_of)
    df = _asof_join(df, f, ["funding_rate"])
    df = _asof_join(df, o, ["open_interest"])
    return df.sort("event_ts").select(*[c for c in ("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "ret", "sigma_risk_next",
                                                  "funding_rate", "open_interest", "turnover")])


def liquid_rows(f: pl.DataFrame, rows: pl.DataFrame, floor: float) -> pl.DataFrame:
    """Null lsig2 (a core feature) at hours t whose turnover over bars t-24..t-1 is below the floor."""
    t = f.select("event_ts", "turnover").sort("event_ts")
    t = t.with_columns(pl.col("turnover").rolling_sum(24, min_samples=24).shift(1).alias("to24"))
    r = rows.join(t.select("event_ts", "to24"), on="event_ts", how="left")
    return r.with_columns(pl.when(pl.col("to24") >= floor).then(pl.col("lsig2")).otherwise(None).alias("lsig2")).drop("to24")


# ------------------------------------------------------------------ panel


def panel_path() -> Path:
    return data_root() / "panel_v4.npz"


def symbols_all() -> list[str]:
    from ai_trader.research.xs_v4_data import candidates

    s = candidates()
    return ["BTCUSDT", *[x for x in s if x != "BTCUSDT"]]


def build() -> dict:
    p = panel_path()
    if p.is_file():
        return load_panel()
    floor = v4cfg()["universe"]["min_turnover_usd"]
    syms = symbols_all()
    rows = {}
    for s in syms:
        t0 = time.time()
        f = frame(s, AS_OF)
        if f.height < 24 * 120:
            print(s, "skipped (short)", f.height, flush=True)
            continue
        rows[s] = liquid_rows(f, study.token_rows(f.drop("turnover")), floor)
        print(s, f.height, round(time.time() - t0, 1), flush=True)
    syms = [s for s in syms if s in rows]
    P = study.assemble(rows, syms)
    P["sector"] = np.array(sectors(syms))
    keep = P["valid"]
    first = int(np.argmax(keep))
    P = {k: (v[first:] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(P["ts"]) else v) for k, v in P.items()}
    np.savez(p, **P)
    meta = {"as_of": AS_OF.isoformat(), "symbols": syms, "sectors": P["sector"].tolist(), "hours": len(P["ts"]), "valid_hours": int(P["valid"].sum()),
            "first": str(np.datetime64(int(P["ts"][0]), "us")), "present_cells": int(P["present"].sum()),
            "mean_present": float(P["present"].sum(1)[P["valid"]].mean()), "max_present": int(P["present"].sum(1).max())}
    (data_root() / "panel_v4.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return load_panel()


def load_panel() -> dict:
    d = np.load(panel_path(), allow_pickle=False)
    return {k: d[k] for k in d.files}


def sec_idx(P: dict) -> np.ndarray:
    so = sector_order()
    return np.array([so.index(s) for s in P["sector"]], dtype=np.int64)


def new_model(P: dict):
    return xs.make(P["X"].shape[2], P["M"].shape[1], len(sector_order()), kind="kan", mix=True)


# ------------------------------------------------------------------ walk-forward


def wf_dir() -> Path:
    p = mroot() / "wf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def all_quarters(P: dict) -> list[tuple[int, int]]:
    return study.quarters(P, "llm_in_sample") + study.quarters(P, "clean_holdout")


def fit_seeds(P: dict, fi: np.ndarray, vi: np.ndarray, prev_states, k: int, sec: np.ndarray):
    import torch

    c = v4cfg()["model"]
    st = study.fit_std(P, fi)
    fb, vb = study.block(P, fi, st), study.block(P, vi, st)
    models, info = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(1000 * s + 7)
        m = new_model(P)
        if prev_states is not None:
            m.load_state_dict(prev_states[s])
        else:
            study.init_model_bias(m, fb)
        r = xs.train(m, fb, vb, sec, seed=1000 * s + k, lr=c["lr"], wd=c["wd"])
        models.append(m)
        info.append({"best_val": r["best_val"], "epoch": r["epoch"], "epochs_run": r["epochs_run"]})
    del fb, vb
    return models, st, info


def run_walk() -> None:
    import torch

    torch.set_num_threads(2)
    P = build()
    sec = sec_idx(P)
    qs = all_quarters(P)
    for k, (q, qn) in enumerate(qs):
        tag = study.qtag(q)
        f_npz, f_pt = wf_dir() / f"{tag}.npz", wf_dir() / f"{tag}.pt"
        if f_npz.is_file() and f_pt.is_file():
            continue
        t0 = time.time()
        tr = study.train_idx(P, q)
        fi, vi = study.split_fit_val(P, tr)
        te = study.test_idx(P, q, qn)
        prev = wf_dir() / f"{study.qtag(qs[k - 1][0])}.pt" if k > 0 else None
        prev_states = torch.load(prev, weights_only=False)["states"] if prev is not None and prev.is_file() else None
        models, st, info = fit_seeds(P, fi, vi, prev_states, k, sec)
        tb = study.block(P, te, st)
        outs = [xs.predict(m, tb, sec) for m in models]
        har = study.fit_har_t(P, tr)
        np.savez_compressed(f_npz, idx_test=te, vol=np.mean([o[0] for o in outs], 0).astype(np.float32),
                            rel=np.mean([o[1] for o in outs], 0).astype(np.float32), vol_har=study.har_quantiles(P, te, har))
        torch.save({"states": [study.copy_state(m) for m in models], "std": {a: v.tolist() for a, v in st.items()}, "har": har,
                    "info": {"quarter": tag, "seeds": info, "n_fit": len(fi), "n_val": len(vi), "n_test": len(te), "secs": round(time.time() - t0, 1)}}, f_pt)
        print(json.dumps({"q": tag, "seeds": info, "secs": round(time.time() - t0, 1)}), flush=True)
        del tb


def evaluate() -> dict:
    """Walk-forward IC of the rel median vs the low-vol ranking; vol pinball (1h, 4h) vs panel log-HAR-t."""
    P = load_panel()
    names = list(P["features"])
    il = names.index("lsig2")
    cut = study.cut_us()
    res: dict = {}
    files = sorted(wf_dir().glob("*.npz"))
    idx = np.concatenate([np.load(f)["idx_test"] for f in files])
    rel = np.concatenate([np.load(f)["rel"] for f in files])
    vol = np.concatenate([np.load(f)["vol"] for f in files])
    vh = np.concatenate([np.load(f)["vol_har"] for f in files])
    ts = P["ts"][idx]
    hr = (ts // HOUR_US) % 24
    day = ts // DAY_US
    pres = P["present"][idx]
    for sg in SEGS:
        m_seg = (ts < cut) if sg == "llm_in_sample" else (ts >= cut)
        r = {}
        for k, h in enumerate(xs.REL_H):
            org = m_seg & ((hr % 4 == 0) if h == 4 else (hr == 0))
            real = P["rel_raw"][idx][org][..., k]
            msk = pres[org] & np.isfinite(real)
            s2 = P["s2"][idx][org]
            med = rel[org][..., k, 1] * np.sqrt(h * s2)       # raw units
            lowvol = -P["X"][idx][org][..., il]
            ic_m = xs.spearman_rows(med, real, msk)
            ic_l = xs.spearman_rows(lowvol, real, msk)
            d = day[org]
            dm = study.per_day(ic_m, d)
            dl = study.per_day(ic_l, d)
            dd = study.per_day(ic_m - ic_l, d)
            r[f"{h}h"] = {"ic_v4": study.nw(dm), "ic_lowvol": study.nw(dl), "diff": study.nw(dd), "n_origins": int(np.isfinite(ic_m).sum()),
                          "mean_tokens": float(msk.sum(1).mean()) if msk.size else None}
        for k, h in enumerate((1, 4)):
            org = m_seg & ((hr % 4 == 0) if h == 4 else np.ones_like(m_seg))
            y = P["Av"][idx][org][..., k]
            msk = pres[org] & np.isfinite(y)
            pm = study.pinball_cells(y, vol[org][..., k, :], xs.TAUS)
            ph = study.pinball_cells(y, vh[org][..., k, :], xs.TAUS)
            dd = np.where(msk, pm - ph, np.nan)
            with np.errstate(all="ignore"):
                per_o = np.nanmean(dd, 1)
            r[f"vol{h}h_pinball_v4_minus_har"] = study.nw(study.per_day(per_o, day[org]))
        res[sg] = r
    (mroot() / "eval_walk.json").write_text(json.dumps(res, indent=1, default=study._json_default), encoding="utf-8")
    return res


# ------------------------------------------------------------------ freeze


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


CODE_FILES = ("src/ai_trader/ml/xs_net.py", "src/ai_trader/research/xs_network_study.py", "src/ai_trader/ml/hourly.py",
              "src/ai_trader/research/xs_v4_study.py", "src/ai_trader/live/xs_v4.py")


def freeze() -> None:
    """Final refit: every valid hour whose 24h label ends >= 1 day before AS_OF (fit / val split as the walk), warm start
    from the last walk-forward quarter. Writes data/models/xs_v4/frozen/{spec.json, v4_seed*.pt}."""
    import subprocess

    import torch

    torch.set_num_threads(2)
    P = load_panel()
    sec = sec_idx(P)
    qs = all_quarters(P)
    last = wf_dir() / f"{study.qtag(qs[-1][0])}.pt"
    prev = torch.load(last, weights_only=False)["states"]
    cut_us = int(AS_OF.timestamp() * 1e6)
    tr = np.flatnonzero(P["valid"] & (P["ts"] + 24 * HOUR_US <= cut_us - DAY_US))
    fi, vi = study.split_fit_val(P, tr)
    t0 = time.time()
    models, st, info = fit_seeds(P, fi, vi, prev, len(qs), sec)
    har = study.fit_har_t(P, tr)
    fz = mroot() / "frozen"
    fz.mkdir(parents=True, exist_ok=True)
    files = {}
    for s, m in enumerate(models):
        f = fz / f"v4_seed{s}.pt"
        torch.save(m.state_dict(), f)
        files[f.name] = _sha(f)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=load().root).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    spec = {"model_id": "xs.v4", "version": v4cfg()["version"], "trained_at": datetime.now(UTC).isoformat(), "as_of_data": AS_OF.isoformat(),
            "symbols_trained": list(P["symbols"]), "sectors_trained": list(P["sector"]), "sector_order": list(sector_order()),
            "features": list(P["features"]), "mkt_features": list(xs.MKT_FEATURES),
            "model": {"n_in": int(P["X"].shape[2]), "n_mkt": int(P["M"].shape[1]), "n_sector": len(sector_order()), "kind": "kan", "mix": True,
                      "seeds": N_SEEDS, "d": xs.D, "lr": v4cfg()["model"]["lr"], "wd": v4cfg()["model"]["wd"]},
            "standardisation": {"mu": st["mu"].tolist(), "sd": st["sd"].tolist(), "mkt_mu": st["mmu"].tolist(), "mkt_sd": st["msd"].tolist()},
            "har_baseline": {str(k): v for k, v in har.items()}, "fit_range_us": [int(P["ts"][fi[0]]), int(P["ts"][fi[-1]])],
            "train_info": info, "secs": round(time.time() - t0, 1), "files_sha256": files, "git_commit": commit,
            "code_sha256": {f: _sha(load().root / f) for f in CODE_FILES if (load().root / f).exists()},
            "evidence": "research/reports/xs_v4_2026-11-16.md"}
    (fz / "spec.json").write_text(json.dumps(spec, indent=1, default=study._json_default), encoding="utf-8")
    print(json.dumps({"frozen": str(fz), "secs": spec["secs"], "seeds": info}), flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="xs_v4_study")
    ap.add_argument("cmd", choices=["build", "walk", "evaluate", "freeze"])
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build()
    elif a.cmd == "walk":
        run_walk()
    elif a.cmd == "evaluate":
        print(json.dumps(evaluate(), indent=1, default=study._json_default))
    else:
        freeze()


if __name__ == "__main__":
    main()
