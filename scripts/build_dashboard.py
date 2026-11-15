"""BTC field dashboard: one self-contained HTML page (plotly.js from jsDelivr, data embedded as JSON).

    PYTHONIOENCODING=utf-8 uv run --extra ml --extra viz python scripts/build_dashboard.py [--as-of 2026-11-13T21:00] \
        [--out research/dashboard/btc_field.html] [--wait-min 0] [--synthetic]

Panels (one finding each; every input is filtered to <= --as-of):
  1 state     BTC state space (agent F1's state coordinates, standardised) projected to 2 dims by PCA (numpy SVD), the
              last 7 days' trajectory, and the empirical Kramers-Moyal field of the projected state: drift arrows
              (mean 24 h displacement per cell) and diffusion ellipses (1-sd ellipse of the 24 h displacement). Underlay:
              F1's fitted field's predicted 1 h sd per cell when data/models/field/*.json exists.
  2 latent    agent F2's network latent (data/models/pnn/latent_1h.parquet, latest-refit coordinates zf*): trajectory in
              its 2-dim PCA, the predicted next latent, and the feedback error |z_{t+1} - zhat_{t+1}| over time with a
              rolling 10-90% band.
  3 card      the current distribution card: F1 field quantiles, P(touch +-1/2/3%) at 1 h / 4 h / 1 d, jump
              probability, and F2's conformal bands (ACI 80/90/95) for the latest row.
  4 news      BTC daily close, realised vol, and GDELT abnormal attention ('bitcoin' query) as three aligned small
              multiples (one y-axis each), with scheduled CPI / NFP / FOMC / crypto event markers.
  5 mi        mutual information (bits, 8x8 quantile bins, Miller-Madow corrected) between state / news variables and
              the latent dims; hover shows the Spearman correlation.
Missing inputs render a clearly labelled 'pending' panel; --wait-min polls for F1 / F2 artifacts before building.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
NEWS_VARS = ("att_btc", "att_ind", "tone_btc", "ev_macro_next", "oc_mempool")


# ------------------------------------------------------------------ numerics (pure numpy, tested on synthetic data)

def pca2(X: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardise columns; return (scores n x k, components k x d, mean, sd, explained-variance share)."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Z = np.nan_to_num((X - mu) / sd)
    _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    comp = vt[:k]
    ev = (s[:k] ** 2) / max((s ** 2).sum(), 1e-12)
    return Z @ comp.T, comp, mu, sd, ev


def km_grid(Y: np.ndarray, step: int = 24, nbin: int = 10, min_n: int = 40) -> list[dict]:
    """Empirical Kramers-Moyal field of a 2-D path: per grid cell the mean `step`-ahead displacement (drift) and the
    covariance of the displacement (diffusion)."""
    Y = np.asarray(Y, float)
    if len(Y) <= step + 10:
        return []
    d = Y[step:] - Y[:-step]
    y0 = Y[:-step]
    ok = np.isfinite(d).all(axis=1) & np.isfinite(y0).all(axis=1)
    y0, d = y0[ok], d[ok]
    ex = [np.quantile(y0[:, j], np.linspace(0.02, 0.98, nbin + 1)) for j in range(2)]
    ix = np.clip(np.searchsorted(ex[0], y0[:, 0]) - 1, 0, nbin - 1)
    iy = np.clip(np.searchsorted(ex[1], y0[:, 1]) - 1, 0, nbin - 1)
    cells = []
    for a in range(nbin):
        for b in range(nbin):
            m = (ix == a) & (iy == b)
            if m.sum() < min_n:
                continue
            dd = d[m]
            c = np.cov(dd.T)
            cells.append({"x": float(y0[m, 0].mean()), "y": float(y0[m, 1].mean()), "dx": float(dd[:, 0].mean()), "dy": float(dd[:, 1].mean()),
                          "cov": [[float(c[0, 0]), float(c[0, 1])], [float(c[1, 0]), float(c[1, 1])]], "n": int(m.sum()), "ia": a, "ib": b})
    return cells


def ellipse(cx: float, cy: float, cov, scale: float = 1.0, n: int = 40) -> tuple[list, list]:
    w, v = np.linalg.eigh(np.asarray(cov, float))
    w = np.sqrt(np.maximum(w, 0)) * scale
    t = np.linspace(0, 2 * np.pi, n)
    pts = (v @ np.vstack([w[0] * np.cos(t), w[1] * np.sin(t)])).T + [cx, cy]
    return pts[:, 0].round(4).tolist(), pts[:, 1].round(4).tolist()


def mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """MI in bits between two variables after quantile binning, Miller-Madow bias corrected, floored at 0."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 50:
        return float("nan")

    def qbin(v):
        e = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
        return np.searchsorted(e, v, side="right")
    bx, by = qbin(x), qbin(y)
    joint = np.zeros((bx.max() + 1, by.max() + 1))
    np.add.at(joint, (bx, by), 1)
    p = joint / n
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log2(p[nz] / (px[:, None] * py[None, :])[nz])).sum()
    k_xy, k_x, k_y = nz.sum(), (px > 0).sum(), (py > 0).sum()
    mi -= (k_xy - k_x - k_y + 1) / (2 * n * math.log(2))
    return float(max(mi, 0.0))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def rolling_band(e: np.ndarray, w: int = 24) -> tuple[list, list, list]:
    lo, md, hi = [], [], []
    for i in range(len(e)):
        s = e[max(0, i - w + 1): i + 1]
        s = s[np.isfinite(s)]
        if len(s) < max(4, w // 4):
            lo.append(None), md.append(None), hi.append(None)
        else:
            q = np.quantile(s, [0.1, 0.5, 0.9])
            lo.append(round(float(q[0]), 4)), md.append(round(float(q[1]), 4)), hi.append(round(float(q[2]), 4))
    return lo, md, hi


def _r(a, nd=4):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in np.asarray(a, float)]


# ------------------------------------------------------------------ panel builders from arrays

def state_panel(ts: list[str], X: np.ndarray, names: list[str], sigma_field: np.ndarray | None, traj_hours: int = 168) -> dict:
    Y, comp, _, _, ev = pca2(X)
    cells = km_grid(Y)
    # glyph scale: the median 1-sd ellipse major semi-axis is drawn at 40% of the median cell spacing (stated on the chart)
    scale = 1.0
    if cells:
        semi = np.median([math.sqrt(max(np.linalg.eigvalsh(np.asarray(c["cov"])).max(), 1e-12)) for c in cells])
        spacing = np.median([abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]) for a in cells for b in cells if a is not b and abs(a["ia"] - b["ia"]) + abs(a["ib"] - b["ib"]) == 1] or [1.0])
        scale = float(0.4 * spacing / semi) if semi > 0 else 1.0
    ell = []
    for c in cells:
        ex, ey = ellipse(c["x"], c["y"], c["cov"], scale)
        ell.append({"x": ex, "y": ey})
        c["dx"], c["dy"] = c["dx"] * scale, c["dy"] * scale
    heat = None
    if sigma_field is not None and len(cells):
        heat = [{"x": c["x"], "y": c["y"], "v": None} for c in cells]
        nb = 10
        ex0 = np.quantile(Y[:-24, 0], np.linspace(0.02, 0.98, nb + 1))
        ey0 = np.quantile(Y[:-24, 1], np.linspace(0.02, 0.98, nb + 1))
        ia = np.clip(np.searchsorted(ex0, Y[:, 0]) - 1, 0, nb - 1)
        ib = np.clip(np.searchsorted(ey0, Y[:, 1]) - 1, 0, nb - 1)
        for h, c in zip(heat, cells, strict=True):
            m = (ia == c["ia"]) & (ib == c["ib"]) & np.isfinite(sigma_field)
            h["v"] = round(float(np.nanmedian(sigma_field[m])), 4) if m.any() else None
    tr = slice(max(0, len(Y) - traj_hours), len(Y))
    loads = [{"var": n, "pc1": round(float(comp[0, j]), 3), "pc2": round(float(comp[1, j]), 3)} for j, n in enumerate(names)]
    return {"status": "ok", "ev": [round(float(v), 3) for v in ev], "loadings": loads, "glyph_scale": round(scale, 3),
            "cells": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != "cov"} for c in cells], "ellipses": ell,
            "heat": heat, "traj": {"ts": ts[tr], "x": _r(Y[tr, 0]), "y": _r(Y[tr, 1])}, "now": {"ts": ts[-1], "x": round(float(Y[-1, 0]), 4), "y": round(float(Y[-1, 1]), 4)},
            "cloud": {"x": _r(Y[::max(1, len(Y) // 3000), 0], 3), "y": _r(Y[::max(1, len(Y) // 3000), 1], 3)}}


def latent_panel(ts: list[str], Z: np.ndarray, Zhat: np.ndarray, traj: int = 72, err_days: int = 30) -> dict:
    """Z: (n, k) latent per decision time, Zhat: (n, k) predicted NEXT latent made at the same row."""
    mu, sd = np.nanmean(Z, 0), np.nanstd(Z, 0)
    sd[sd == 0] = 1
    Zs, Hs = (Z - mu) / sd, (Zhat - mu) / sd
    _, s, vt = np.linalg.svd(np.nan_to_num(Zs - np.nanmean(Zs, 0)), full_matrices=False)
    comp = vt[:2]
    c0 = np.nanmean(Zs, 0)
    Y = np.nan_to_num(Zs - c0) @ comp.T
    Yh = np.nan_to_num(Hs - c0) @ comp.T
    err = np.full(len(Z), np.nan)
    err[1:] = np.sqrt(np.nansum((Zs[1:] - Hs[:-1]) ** 2, axis=1))  # error of the prediction made one step earlier
    pers = np.full(len(Z), np.nan)
    pers[1:] = np.sqrt(np.nansum((Zs[1:] - Zs[:-1]) ** 2, axis=1))
    n_err = min(len(Z), err_days * 24)
    lo, md, hi = rolling_band(err[-n_err:], 24)
    tr = slice(max(0, len(Y) - traj), len(Y))
    return {"status": "ok", "ev": [round(float(v), 3) for v in (s[:2] ** 2 / max((s ** 2).sum(), 1e-12))],
            "traj": {"ts": ts[tr], "x": _r(Y[tr, 0]), "y": _r(Y[tr, 1])}, "now": {"ts": ts[-1], "x": round(float(Y[-1, 0]), 4), "y": round(float(Y[-1, 1]), 4)},
            "pred": {"x": round(float(Yh[-1, 0]), 4), "y": round(float(Yh[-1, 1]), 4)},
            "err": {"ts": ts[-n_err:], "e": _r(err[-n_err:]), "lo": lo, "md": md, "hi": hi},
            "err_summary": {"model_median": float(np.nanmedian(err[-n_err:])), "persistence_median": float(np.nanmedian(pers[-n_err:]))}}


def mi_panel(rows: list[str], cols: list[str], A: np.ndarray, B: np.ndarray) -> dict:
    mi = [[mutual_info(A[:, i], B[:, j]) for j in range(B.shape[1])] for i in range(A.shape[1])]
    rho = [[spearman(A[:, i], B[:, j]) for j in range(B.shape[1])] for i in range(A.shape[1])]
    return {"status": "ok", "rows": rows, "cols": cols, "mi": [_r(r, 4) for r in mi], "rho": [_r(r, 3) for r in rho], "n": int(len(A))}


# ------------------------------------------------------------------ real inputs (each guarded; failures -> pending)

def _pending(msg: str) -> dict:
    return {"status": "pending", "msg": msg}


def latest_field_json() -> Path | None:
    d = ROOT / "data" / "models" / "field"
    idx = d / "index.json"
    if idx.is_file():
        try:
            p = d / json.loads(idx.read_text(encoding="utf-8"))["latest"]
            if p.is_file():
                return p
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    fs = sorted(glob.glob(str(ROOT / "data" / "models" / "field" / "*.json")), key=os.path.getmtime)
    for f in reversed(fs):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            if str(d.get("schema", "")).startswith("field.v"):
                return Path(f)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def latent_file() -> Path | None:
    for h in ("1h", "1d"):
        p = ROOT / "data" / "models" / "pnn" / f"latent_{h}.parquet"
        if p.is_file():
            return p
    return None


F1_HEADLINE = ("Agent F1 (research/reports/btc_field_2026-11-13.md): the field beats log-HAR-t with constant jumps only through the "
               "hour-of-week shape; against log-HAR-t x hour-of-week it ties at 1h. Its drift has no direction skill (H-FLD2 null).")


def field_drivers(fld, st) -> dict:
    """Evaluate F1's fitted field around the current state: (a) 1h variance and jump probability across the 168 hours of the
    week (s = 0.5 ln f(h), other coordinates held at now); (b) variance multiplier over a momentum (m24) x trend (dir) grid."""
    from ai_trader.field.model import coords_matrix
    from ai_trader.field.state import COORDS

    X0 = coords_matrix(st.slice(st.height - 1, 1))[0]
    lnvf = float(np.log(st["vf"][st.height - 1]))
    ci = {c: i for i, c in enumerate(COORDS)}
    prof = np.asarray(fld.profile, float)
    Xh = np.repeat(X0[None, :], 168, axis=0)
    Xh[:, ci["s"]] = 0.5 * np.log(np.maximum(prof, 1e-9))
    _, Dh, ph = fld.params(Xh, np.full(168, lnvf))
    Dh, ph = np.asarray(Dh, float), np.asarray(ph, float)
    g = np.linspace(-3, 3, 13)
    M, P = np.zeros((13, 13)), np.zeros((13, 13))
    Xg = np.repeat(X0[None, :], 169, axis=0)
    mm, dd = np.meshgrid(g, g, indexing="ij")
    Xg[:, ci["m24"]] = mm.ravel()
    Xg[:, ci["dir"]] = dd.ravel()
    _, Dg, pg = fld.params(Xg, np.full(169, lnvf))
    Xc = X0.copy()
    Xc[ci["m24"]] = 0.0
    Xc[ci["dir"]] = 0.0
    _, Dc, _ = fld.params(Xc[None, :], np.array([lnvf]))
    M = (np.asarray(Dg, float) / float(np.asarray(Dc)[0])).reshape(13, 13)
    P = np.asarray(pg, float).reshape(13, 13)
    now_how = int(((st["known_ts"][st.height - 1].weekday()) * 24 + st["known_ts"][st.height - 1].hour) % 168)
    return {"status": "ok", "how": {"mult": _r(Dh / Dh.mean(), 3), "p_jump": _r(ph, 4), "now": now_how},
            "grid": {"m24": g.round(2).tolist(), "dir": g.round(2).tolist(), "log2_mult": [_r(np.log2(r), 3) for r in M],
                     "mult": [_r(r, 3) for r in M], "p_jump": [_r(r, 4) for r in P],
                     "now": {"m24": round(float(X0[ci["m24"]]), 2), "dir": round(float(X0[ci["dir"]]), 2)}},
            "headline": F1_HEADLINE}


def f2_findings() -> dict:
    """Headline findings of agent F2 (research/reports/btc_pnn_*.json), for the latent and conformal panels."""
    out = {"latent": "Agent F2: the latent is predictable (R^2 gain over persistence 0.14 at 1h, 0.89 at 1d) but feeding the "
                     "predicted latent back adds nothing; one 1d latent is a weekend clock; KAN neurons add nothing over an equal-size MLP.",
           "conformal": ""}
    fs = sorted(glob.glob(str(ROOT / "research" / "reports" / "btc_pnn_*.json")))
    if not fs:
        return out
    try:
        v = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))["verdicts"]
        for seg in ("clean_holdout", "llm_in_sample"):
            m = (v.get(seg) or {}).get("H-PNN2", {}).get("metrics", {})
            if all(f"1h_{lv}" in m for lv in (80, 90, 95)):
                cov = "/".join(f"{100 * m[f'1h_{lv}']['aci_cov']:.1f}" for lv in (80, 90, 95))
                wr = "/".join(f"{m[f'1h_{lv}']['width_ratio']:.2f}" for lv in (80, 90, 95))
                out["conformal"] = f"ACI coverage at 1h ({seg}): {cov}% for nominal 80/90/95; width vs log-HAR-t x{wr}. Source: {Path(fs[-1]).name}."
                break
    except Exception:  # noqa: BLE001
        pass
    return out


def gather(as_of: datetime, log=print) -> dict:
    import polars as pl

    panels: dict = {}
    meta = {"as_of": as_of.isoformat(), "built_at": datetime.now(UTC).isoformat(timespec="seconds"), "sources": {}}
    # ---- F1 state (+ field)
    st = fld = None
    try:
        from ai_trader.field.state import COORDS, state_asof

        st, _ = state_asof(as_of)
        st = st.filter(pl.col("known_ts") <= as_of).drop_nulls(list(COORDS))
        st = st.filter(pl.col("known_ts") >= as_of - timedelta(days=730))
        meta["sources"]["state"] = f"ai_trader.field.state ({st.height} hourly rows)"
    except Exception as e:  # noqa: BLE001
        log(f"state: {e}")
    fpath = latest_field_json()
    if fpath is not None:
        try:
            from ai_trader.field.model import Field

            fld = Field.load(fpath)
            meta["sources"]["field"] = fpath.relative_to(ROOT).as_posix()
        except Exception as e:  # noqa: BLE001
            log(f"field: {e}")
    if st is not None and st.height > 500:
        from ai_trader.field.model import coords_matrix
        from ai_trader.field.state import COORDS

        X = coords_matrix(st)
        sig = None
        if fld is not None:
            try:
                mu, D, p = fld.params(X, np.log(st["vf"].to_numpy().astype(float)))
                sig = 100 * np.sqrt(np.asarray(D) * (1 + np.asarray(p) * fld.kou.second))
            except Exception as e:  # noqa: BLE001
                log(f"field params: {e}")
        ts = [str(t)[:16] for t in st["known_ts"].to_list()]
        panels["state"] = state_panel(ts, X, list(COORDS), sig)
        panels["state"]["field"] = meta["sources"].get("field")
        if fld is None:
            panels["state"]["note"] = "field underlay pending: no data/models/field/*.json yet (agent F1)"
    else:
        panels["state"] = _pending("BTC state not available (agent F1's ai_trader.field.state)")
    # ---- card
    if fld is not None and st is not None and st.height:
        try:
            from ai_trader.field.cards import card

            c = card(fld, st, st.height - 1, moves=(0.01, 0.02, 0.03), n_paths=4000)
            panels["card"] = {"status": "ok", "field": c}
        except Exception as e:  # noqa: BLE001
            panels["card"] = _pending(f"field card failed: {e}")
    else:
        panels["card"] = _pending("distribution card pending: agent F1's fitted field (data/models/field/*.json) not found")
    # ---- field drivers: hour-of-week amplification and momentum x trend
    if fld is not None and st is not None and st.height:
        try:
            panels["drivers"] = field_drivers(fld, st)
        except Exception as e:  # noqa: BLE001
            panels["drivers"] = _pending(f"field drivers failed: {e}")
    else:
        panels["drivers"] = _pending("field drivers pending: agent F1's fitted field (data/models/field/) not found")
    # ---- F2 latent
    lf = latent_file()
    lat = None
    if lf is not None:
        try:
            lat = pl.read_parquet(lf).sort("ts").filter(pl.col("ts") <= as_of)
            zc = sorted(c for c in lat.columns if c.startswith("zf") and not c.startswith("zfhat"))
            hc = sorted(c for c in lat.columns if c.startswith("zfhat"))
            if not zc:
                zc = sorted(c for c in lat.columns if c[:1] == "z" and c[1:].isdigit())
                hc = sorted(c for c in lat.columns if c.startswith("zhat"))
            lat = lat.drop_nulls(zc + hc)
            ts = [str(t)[:16] for t in lat["ts"].to_list()]
            panels["latent"] = latent_panel(ts, lat.select(zc).to_numpy(), lat.select(hc).to_numpy())
            panels["latent"]["file"] = lf.relative_to(ROOT).as_posix()
            panels["latent"]["coords"] = "latest-network coordinates (zf*)" if zc[0].startswith("zf") else "per-refit coordinates (z*)"
            panels["latent"]["f2"] = f2_findings()
            meta["sources"]["latent"] = panels["latent"]["file"]
            last = lat.row(lat.height - 1, named=True)
            bands = {lv: [last.get(f"aci_lo{lv}"), last.get(f"aci_hi{lv}")] for lv in (80, 90, 95) if last.get(f"aci_lo{lv}") is not None}
            qs = {k: last[k] for k in ("q05", "q10", "q25", "q50", "q75", "q90", "q95") if k in last}
            panels.setdefault("card", {})
            panels["card"]["conformal"] = {"horizon": last.get("horizon"), "ts": str(last["ts"]), "bands": bands, "pnn_q": qs,
                                           "note": panels["latent"]["f2"]["conformal"]}
        except Exception as e:  # noqa: BLE001
            log(f"latent: {e}")
            panels["latent"] = _pending(f"latent file unreadable: {e}")
    else:
        panels["latent"] = _pending("latent trajectory pending: agent F2's data/models/pnn/latent_1h.parquet not found")
    # ---- news timeline
    try:
        from ai_trader.features.realized import daily_panel
        from ai_trader.news import events
        from ai_trader.news.store import Manifest

        man = Manifest()
        d = daily_panel(["BTCUSDT"], as_of=as_of).sort("date").filter(pl.col("date") >= (as_of - timedelta(days=548)).date())
        fe = pl.read_parquet(man.root / "features" / "daily.parquet").filter(pl.col("asof") <= as_of)
        fe = fe.with_columns((pl.col("asof").dt.date() - timedelta(days=1)).alias("date")).select("date", "att_btc", "gdelt_norm_z")
        d = d.join(fe, on="date", how="left")
        ev = events.load_or_build(man).filter((pl.col("scheduled_ts") <= as_of) & (pl.col("scheduled_ts").dt.date() >= d["date"].min()))
        panels["news"] = {"status": "ok", "date": [str(x) for x in d["date"].to_list()], "close": _r(d["close"].to_numpy(), 1),
                          "vol": _r(100 * np.sqrt(365 * d["rv"].to_numpy()), 2), "att": _r(d["att_btc"].to_numpy(), 3),
                          "norm_z": _r(d["gdelt_norm_z"].to_numpy(), 3),
                          "events": [{"kind": k, "name": n, "ts": str(t)[:16]} for k, n, t in ev.select("kind", "name", "scheduled_ts").iter_rows()]}
        meta["sources"]["news"] = "data/research/news/features/daily.parquet"
    except Exception as e:  # noqa: BLE001
        log(f"news: {e}")
        panels["news"] = _pending(f"news features unavailable: {e}")
    # ---- MI map
    if lat is not None and st is not None and st.height:
        try:
            from ai_trader.field.state import COORDS

            zc = sorted(c for c in lat.columns if c.startswith("zf") and not c.startswith("zfhat")) or sorted(c for c in lat.columns if c[:1] == "z" and c[1:].isdigit())
            # F1 known_ts = bar close + ~2 s; F2 ts = decision time (next bar open): match on the hour (descriptive map)
            j = st.select(pl.col("known_ts").dt.truncate("1h").alias("ts"), *COORDS).join(lat.select("ts", *zc), on="ts", how="inner")
            rows = list(COORDS)
            try:
                from ai_trader.news.store import Manifest

                hd = pl.read_parquet(Manifest().root / "features" / "daily.parquet").select("asof", *NEWS_VARS).sort("asof")
                j = j.sort("ts").join_asof(hd, left_on="ts", right_on="asof", strategy="backward")
                rows += list(NEWS_VARS)
            except Exception as e:  # noqa: BLE001
                log(f"mi news join: {e}")
            A = j.select(rows).to_numpy().astype(float)
            B = j.select(zc).to_numpy().astype(float)
            panels["mi"] = mi_panel(rows, zc, A, B)
        except Exception as e:  # noqa: BLE001
            panels["mi"] = _pending(f"MI map failed: {e}")
    else:
        panels["mi"] = _pending("MI map pending: needs agent F1's state and agent F2's latent file")
    return {"meta": meta, "panels": panels}


# ------------------------------------------------------------------ synthetic inputs (tests / demo)

def synthetic_inputs(seed: int = 0, n: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ts = [(t0 + timedelta(hours=i)).isoformat()[:16] for i in range(n)]
    X = np.cumsum(rng.standard_normal((n, 6)) * 0.05, axis=0) + rng.standard_normal((n, 6)) * 0.2
    Z = np.cumsum(rng.standard_normal((n, 4)) * 0.1, axis=0)
    Zh = Z + rng.standard_normal((n, 4)) * 0.05
    q = {"q05": -2.1, "q10": -1.5, "q25": -0.7, "q50": 0.0, "q75": 0.7, "q90": 1.5, "q95": 2.1}
    card = {"status": "ok", "field": {"as_of": ts[-1], "close": 90000.0, "one_hour": {"sd_pct": 0.5, "p_jump": 0.03, "drift_pct": 0.0},
                                      "horizons": {h: {"quantiles_pct": {k: v * s for k, v in q.items()}, "sd_pct": s, "p_jump": 0.03 * s,
                                                       "p_touch_pct": {f"{m:+d}%": min(0.99, s / (abs(m) * 3)) for m in (1, 2, 3, -1, -2, -3)}}
                                                   for h, s in (("1h", 0.4), ("4h", 0.9), ("24h", 2.0))}, "text": "synthetic"},
            "conformal": {"horizon": "1h", "ts": ts[-1], "bands": {80: [-0.006, 0.006], 90: [-0.008, 0.008], 95: [-0.01, 0.01]}}}
    days = [(t0 + timedelta(days=i)).date().isoformat() for i in range(120)]
    news = {"status": "ok", "date": days, "close": _r(90000 * np.exp(np.cumsum(rng.standard_normal(120) * 0.02))), "vol": _r(40 + 10 * rng.random(120)),
            "att": _r(rng.standard_normal(120) * 0.4), "norm_z": _r(rng.standard_normal(120) * 0.1),
            "events": [{"kind": "cpi", "name": "CPI", "ts": days[10] + "T13:30"}, {"kind": "fomc", "name": "FOMC decision", "ts": days[40] + "T19:00"}]}
    panels = {"state": state_panel(ts, X, [f"x{i}" for i in range(6)], np.abs(X[:, 0]) + 0.3), "latent": latent_panel(ts, Z, Zh),
              "card": card, "news": news, "mi": mi_panel([f"x{i}" for i in range(6)], [f"z{j}" for j in range(4)], X, Z)}
    return {"meta": {"as_of": ts[-1], "built_at": "synthetic", "sources": {"synthetic": f"seed {seed}"}}, "panels": panels}


# ------------------------------------------------------------------ HTML

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--seq1:#cde2fb;--seq2:#86b6ef;--seq3:#2a78d6;--seq4:#104281;--pend:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--seq1:#184f95;--seq2:#256abf;--seq3:#5598e7;--seq4:#cde2fb;--pend:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--seq1:#184f95;--seq2:#256abf;--seq3:#5598e7;--seq4:#cde2fb;--pend:#383835}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 48px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 14px 8px;min-width:0}
.card.wide{grid-column:1/-1}.finding{color:var(--ink2);margin:0 0 8px;font-size:13px}.plot{width:100%;height:380px}.plot.tall{height:560px}.plot.short{height:260px}
.pending{background:var(--pend);border-radius:8px;padding:28px 16px;color:var(--ink2);text-align:center}
.pending b{display:block;color:var(--ink);margin-bottom:4px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
.meta{color:var(--muted);font-size:12px;margin-top:20px}.note{color:var(--muted);font-size:12px;margin:4px 0 0}
.tbl-wrap{overflow-x:auto}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = () => ({ink:css('--ink'),ink2:css('--ink2'),muted:css('--muted'),grid:css('--grid'),axis:css('--axis'),surface:css('--surface'),
  s1:css('--s1'),s2:css('--s2'),s3:css('--s3'),s4:css('--s4'),s5:css('--s5'),seq:[css('--seq1'),css('--seq2'),css('--seq3'),css('--seq4')]});
const base = (c, x, y, extra={}) => Object.assign({paper_bgcolor:c.surface, plot_bgcolor:c.surface, font:{family:'system-ui,-apple-system,Segoe UI,sans-serif', size:12, color:c.ink2},
  margin:{l:56,r:16,t:8,b:44}, hovermode:'closest', showlegend:true, legend:{orientation:'h', y:-0.18, font:{color:c.ink2}},
  xaxis:{title:{text:x}, gridcolor:c.grid, linecolor:c.axis, zeroline:false, tickfont:{color:c.muted}},
  yaxis:{title:{text:y}, gridcolor:c.grid, linecolor:c.axis, zeroline:false, tickfont:{color:c.muted}}}, extra);
const cfg = {displaylogo:false, responsive:true, modeBarButtonsToRemove:['lasso2d','select2d']};
function pending(id, p){ document.getElementById(id).innerHTML = `<div class="pending"><b>Pending</b>${p.msg||'input not available yet'}</div>`; }

function drawState(c){
  const p = D.panels.state; if(!p || p.status!=='ok') return pending('p-state', p||{});
  const tr = [];
  if(p.heat){ tr.push({type:'scatter', mode:'markers', x:p.heat.map(h=>h.x), y:p.heat.map(h=>h.y), name:'field 1h sd (%)',
     marker:{size:22, symbol:'square', color:p.heat.map(h=>h.v), colorscale:[[0,c.seq[0]],[0.5,c.seq[2]],[1,c.seq[3]]], opacity:0.4,
     colorbar:{title:{text:'1h sd %', font:{color:c.ink2}}, thickness:10, tickfont:{color:c.muted}}}, hovertemplate:'field 1h sd %{marker.color:.3f}%<extra></extra>'}); }
  tr.push({type:'scattergl', mode:'markers', x:p.cloud.x, y:p.cloud.y, name:'history (2 y)', marker:{size:3, color:c.muted, opacity:0.25}, hoverinfo:'skip'});
  p.ellipses.forEach((e,i)=>tr.push({type:'scatter', mode:'lines', x:e.x, y:e.y, line:{color:c.s3, width:1}, name:`24 h spread, 1 sd (drawn x${p.glyph_scale})`, legendgroup:'ell', showlegend:i===0, hoverinfo:'skip'}));
  tr.push({type:'scatter', mode:'lines', x:p.traj.x, y:p.traj.y, text:p.traj.ts, name:'last 7 days', line:{color:c.s1, width:2}, hovertemplate:'%{text}<br>PC1 %{x:.2f} PC2 %{y:.2f}<extra></extra>'});
  tr.push({type:'scatter', mode:'markers+text', x:[p.now.x], y:[p.now.y], text:['now'], textposition:'top right', textfont:{color:c.ink}, name:'now ('+p.now.ts+')',
     marker:{size:11, color:c.s1, line:{color:c.surface, width:2}}, hovertemplate:p.now.ts+'<extra>now</extra>'});
  const ann = p.cells.map(k=>({x:k.x+k.dx, y:k.y+k.dy, ax:k.x, ay:k.y, xref:'x', yref:'y', axref:'x', ayref:'y', showarrow:true, arrowhead:2, arrowsize:1, arrowwidth:1.5, arrowcolor:c.s2, text:''}));
  tr.push({type:'scatter', mode:'lines', x:[null], y:[null], name:`24 h mean move (drawn x${p.glyph_scale})`, line:{color:c.s2, width:1.5}});
  Plotly.newPlot('p-state', tr, base(c, `PC1 (${Math.round(100*p.ev[0])}% of variance)`, `PC2 (${Math.round(100*p.ev[1])}%)`, {annotations:ann}), cfg);
  const L = p.loadings.map(l=>`<tr><td>${l.var}</td><td>${l.pc1.toFixed(2)}</td><td>${l.pc2.toFixed(2)}</td></tr>`).join('');
  document.getElementById('t-state').innerHTML = `<details><summary class="note">PCA loadings (table view)</summary><div class="tbl-wrap"><table><tr><th>coordinate</th><th>PC1</th><th>PC2</th></tr>${L}</table></div></details>` + (p.note?`<p class="note">${p.note}</p>`:'');
}

function drawDrivers(c){
  const p = D.panels.drivers; if(!p || p.status!=='ok'){ pending('p-drivers', p||{}); pending('p-grid', {msg:'momentum x trend map pending'}); return; }
  const h = p.how, x = h.mult.map((_,i)=>i), days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  Plotly.newPlot('p-drivers', [
    {type:'scatter', mode:'lines', x:x, y:h.mult, name:'1h variance multiplier', line:{color:c.s1, width:2}, customdata:h.p_jump,
     hovertemplate:'hour of week %{x} (UTC)<br>variance x%{y:.2f}<br>P(jump) %{customdata:.1%}<extra></extra>', showlegend:false},
    {type:'scatter', mode:'markers+text', x:[h.now], y:[h.mult[h.now]], text:['now'], textposition:'top center', textfont:{color:c.ink}, marker:{size:10, color:c.s1, line:{color:c.surface,width:2}}, showlegend:false, hoverinfo:'skip'}],
    base(c, 'hour of week (UTC)', 'field 1h variance / weekly mean', {xaxis:{tickvals:[0,24,48,72,96,120,144], ticktext:days, gridcolor:c.grid, linecolor:c.axis, tickfont:{color:c.muted}},
      shapes:[{type:'line', xref:'paper', x0:0, x1:1, y0:1, y1:1, line:{color:c.axis, width:1}}]}), cfg);
  const g = p.grid;
  const txt = g.mult.map((r,i)=>r.map((v,j)=>`variance x${v===null?'n/a':v.toFixed(2)}<br>P(jump) ${(100*g.p_jump[i][j]).toFixed(1)}%`));
  const zmax = Math.max(...g.log2_mult.flat().filter(v=>v!==null).map(Math.abs), 0.1);
  Plotly.newPlot('p-grid', [{type:'heatmap', z:g.log2_mult, y:g.m24, x:g.dir, text:txt, zmin:-zmax, zmax:zmax, hovertemplate:'momentum m24 %{y}, trend %{x}<br>%{text}<extra></extra>',
      colorscale:[[0,c.s1],[0.5,css('--pend')],[1,'#e34948']], colorbar:{title:{text:'log2 variance', font:{color:c.ink2}}, thickness:10, tickfont:{color:c.muted}}, xgap:1, ygap:1},
     {type:'scatter', mode:'markers+text', x:[g.now.dir], y:[g.now.m24], text:['now'], textposition:'top center', textfont:{color:c.ink}, marker:{size:10, color:c.surface, line:{color:c.ink, width:2}}, hoverinfo:'skip'}],
    base(c, 'trend score (dir, v0 regime)', '24 h momentum (sd units)', {showlegend:false}), cfg);
  document.getElementById('t-drivers').innerHTML = `<p class="note">${p.headline}</p>`;
}

function drawLatent(c){
  const p = D.panels.latent; if(!p || p.status!=='ok'){ pending('p-latent', p||{}); pending('p-err', {msg:'feedback error pending'}); return; }
  Plotly.newPlot('p-latent', [
    {type:'scatter', mode:'lines', x:p.traj.x, y:p.traj.y, text:p.traj.ts, name:'latent path (last 72 steps)', line:{color:c.s1, width:2}, hovertemplate:'%{text}<br>%{x:.2f}, %{y:.2f}<extra></extra>'},
    {type:'scatter', mode:'markers+text', x:[p.now.x], y:[p.now.y], text:['now'], textposition:'top left', textfont:{color:c.ink}, name:'now', marker:{size:11, color:c.s1, line:{color:c.surface,width:2}}},
    {type:'scatter', mode:'markers+text', x:[p.pred.x], y:[p.pred.y], text:['predicted next'], textposition:'bottom right', textfont:{color:c.ink}, name:'predicted next latent', marker:{size:11, color:c.surface, line:{color:c.s2, width:2}}}],
    base(c, `latent PC1 (${Math.round(100*p.ev[0])}%)`, `latent PC2 (${Math.round(100*p.ev[1])}%)`, {annotations:[{x:p.pred.x, y:p.pred.y, ax:p.now.x, ay:p.now.y, xref:'x', yref:'y', axref:'x', ayref:'y', arrowhead:2, arrowwidth:1.5, arrowcolor:c.s2, text:''}]}), cfg);
  const e = p.err;
  Plotly.newPlot('p-err', [
    {type:'scatter', mode:'lines', x:e.ts, y:e.hi, line:{width:0}, hoverinfo:'skip', showlegend:false},
    {type:'scatter', mode:'lines', x:e.ts, y:e.lo, fill:'tonexty', fillcolor:'rgba(42,120,214,0.18)', line:{width:0}, name:'rolling 10-90% (24 steps)', hoverinfo:'skip'},
    {type:'scatter', mode:'lines', x:e.ts, y:e.md, line:{color:c.s1, width:2}, name:'rolling median', hovertemplate:'%{x}<br>median %{y:.2f}<extra></extra>'}],
    base(c, '', 'feedback error (latent sd)', {hovermode:'x unified'}), cfg);
  document.getElementById('t-latent').innerHTML = `<p class="note">Median error over the window: model ${p.err_summary.model_median.toFixed(2)} vs persistence ${p.err_summary.persistence_median.toFixed(2)} (lower is better). Coordinates: ${p.coords||''}. Source: ${p.file||''}.</p>` + (p.f2?`<p class="note">${p.f2.latent}</p>`:'');
}

function drawCard(c){
  const p = D.panels.card || {};
  const el = document.getElementById('t-card');
  if(!p.field){ pending('p-card', p.status==='pending'?p:{msg:'field card pending: agent F1'}); }
  if(p.field){
    const f = p.field, H = Object.keys(f.horizons);
    const qk = ['q05','q10','q25','q50','q75','q90','q95'];
    const rows = H.map(h=>`<tr><td>${h}</td>${qk.map(k=>`<td>${f.horizons[h].quantiles_pct[k].toFixed(2)}</td>`).join('')}<td>${f.horizons[h].sd_pct.toFixed(2)}</td><td>${(100*f.horizons[h].p_jump).toFixed(1)}</td></tr>`).join('');
    el.innerHTML = `<div class="tbl-wrap"><table><tr><th>horizon</th>${qk.map(k=>`<th>${k}</th>`).join('')}<th>sd</th><th>P(jump) %</th></tr>${rows}</table></div><p class="note">Return quantiles in %, from the field's Monte-Carlo (as of ${f.as_of}, close ${f.close}). ${f.text||''}</p>`;
    const lv = ['-3%','-2%','-1%','+1%','+2%','+3%'], cols=[c.s1,c.s2,c.s3];
    const tr = H.map((h,i)=>({type:'scatter', mode:'lines+markers', x:lv, y:lv.map(l=>f.horizons[h].p_touch_pct[l]), name:h, line:{color:cols[i%3], width:2}, marker:{size:8, color:cols[i%3], line:{color:c.surface,width:2}},
      hovertemplate:h+' P(touch %{x}) = %{y:.1%}<extra></extra>'}));
    Plotly.newPlot('p-card', tr, base(c, 'level (move from now)', 'P(touch before horizon)', {yaxis:{title:{text:'P(touch before horizon)'}, tickformat:'.0%', range:[0,1], gridcolor:c.grid, tickfont:{color:c.muted}}}), cfg);
  }
  const cf = p.conformal, ce = document.getElementById('p-conf');
  if(!cf || !cf.bands || !Object.keys(cf.bands).length){ pending('p-conf', {msg:'conformal bands pending: agent F2 latent file'}); return; }
  const lv = Object.keys(cf.bands);
  const tr = lv.map((k,i)=>({type:'scatter', mode:'lines', x:[100*cf.bands[k][0], 100*cf.bands[k][1]], y:[k+'%', k+'%'], name:'ACI '+k+'%', line:{color:[c.s1,c.s2,c.s3][i%3], width:6},
      hovertemplate:'ACI '+k+'%: %{x:.2f}%<extra></extra>'}));
  if(cf.pnn_q && cf.pnn_q.q50!==undefined) tr.push({type:'scatter', mode:'markers', x:[100*cf.pnn_q.q50], y:['80%'], name:'network median', marker:{size:10, color:c.ink, symbol:'line-ns-open', line:{width:2}}});
  const xs = lv.flatMap(k=>cf.bands[k].map(v=>100*v)), pad = 0.08*(Math.max(...xs)-Math.min(...xs));
  if(cf.note) document.getElementById('t-card').insertAdjacentHTML('beforeend', `<p class="note">${cf.note}</p>`);
  Plotly.newPlot('p-conf', tr, base(c, `return over the next ${cf.horizon||''} (%)`, 'nominal coverage', {margin:{l:56,r:16,t:8,b:60}, legend:{orientation:'h', y:-0.42, font:{color:c.ink2}},
    yaxis:{title:{text:'nominal coverage'}, type:'category', gridcolor:c.grid, tickfont:{color:c.muted}},
    xaxis:{title:{text:`return over the next ${cf.horizon||''} (%)`}, range:[Math.min(...xs)-pad, Math.max(...xs)+pad], gridcolor:c.grid, linecolor:c.axis, zeroline:false, tickfont:{color:c.muted}}}), cfg);
}

function drawNews(c){
  const p = D.panels.news; if(!p || p.status!=='ok') return pending('p-news', p||{});
  const ecol = {cpi:c.s2, nfp:c.s4, fomc:c.s5, fomc_unscheduled:c.s5, halving:c.s3, etf_reg:c.s3};
  const shapes = [], seen = {};
  const marks = p.events.map(e=>{ shapes.push({type:'line', xref:'x', yref:'paper', x0:e.ts, x1:e.ts, y0:0, y1:1, line:{color:ecol[e.kind]||c.muted, width:1}, opacity:0.3}); return e; });
  const tr = [
    {type:'scatter', mode:'lines', x:p.date, y:p.close, name:'BTC close (USD)', line:{color:c.s1, width:2}, xaxis:'x', yaxis:'y', hovertemplate:'%{x}<br>%{y:,.0f} USD<extra></extra>'},
    {type:'scatter', mode:'lines', x:p.date, y:p.vol, name:'realised vol (annualised %)', line:{color:c.s1, width:1.5}, xaxis:'x', yaxis:'y2', hovertemplate:'%{x}<br>%{y:.1f}%<extra></extra>', showlegend:false},
    {type:'bar', x:p.date, y:p.att, name:'GDELT abnormal attention, bitcoin (log share vs 60 d)', marker:{color:p.att.map(v=>v>0?c.s2:c.muted)}, xaxis:'x', yaxis:'y3', hovertemplate:'%{x}<br>%{y:+.2f}<extra></extra>', showlegend:false}];
  Object.keys(ecol).forEach(k=>{ const ev = p.events.filter(e=>e.kind===k); if(!ev.length || seen[ecol[k]+k]) return; seen[ecol[k]+k]=1;
    tr.push({type:'scatter', mode:'markers', x:ev.map(e=>e.ts), y:ev.map(()=>0), text:ev.map(e=>e.name), name:k.toUpperCase().replace('_',' '), xaxis:'x', yaxis:'y3',
      marker:{symbol:'triangle-up', size:9, color:ecol[k], line:{color:c.surface,width:1}}, hovertemplate:'%{text}<br>%{x}<extra></extra>'}); });
  const ax = t => ({title:{text:t}, gridcolor:c.grid, linecolor:c.axis, zeroline:false, tickfont:{color:c.muted}});
  const lay = base(c, '', '', {grid:{rows:3, columns:1, pattern:'coupled', roworder:'top to bottom'}, shapes:shapes, hovermode:'x unified',
     xaxis:{gridcolor:c.grid, linecolor:c.axis, tickfont:{color:c.muted}}, yaxis:Object.assign(ax('BTC (USD, log)'), {type:'log'}), yaxis2:ax('realised vol (ann. %)'), yaxis3:ax('abnormal attention (log)'),
     legend:{orientation:'h', y:-0.12, font:{color:c.ink2}}});
  Plotly.newPlot('p-news', tr, lay, cfg);
}

function drawMI(c){
  const p = D.panels.mi; if(!p || p.status!=='ok') return pending('p-mi', p||{});
  const txt = p.mi.map((r,i)=>r.map((v,j)=>`MI ${v===null?'n/a':v.toFixed(3)} bits<br>Spearman ${p.rho[i][j]===null?'n/a':p.rho[i][j].toFixed(2)}`));
  Plotly.newPlot('p-mi', [{type:'heatmap', z:p.mi, x:p.cols, y:p.rows, text:txt, hovertemplate:'%{y} x %{x}<br>%{text}<extra></extra>',
     colorscale:[[0,c.surface],[0.25,c.seq[0]],[0.6,c.seq[2]],[1,c.seq[3]]], colorbar:{title:{text:'MI (bits)', font:{color:c.ink2}}, thickness:10, tickfont:{color:c.muted}}, xgap:2, ygap:2}],
     base(c, 'network latent dimension', '', {showlegend:false, margin:{l:110,r:16,t:8,b:44}, yaxis:{autorange:'reversed', automargin:true, tickfont:{color:c.muted}}}), cfg);
  document.getElementById('t-mi').innerHTML = `<p class="note">${p.n} matched hours; 8x8 quantile bins, Miller-Madow corrected. Hover for the Spearman correlation.</p>`;
}

function drawAll(){ const c = C(); try{drawState(c)}catch(e){console.error(e)} try{drawDrivers(c)}catch(e){console.error(e)} try{drawLatent(c)}catch(e){console.error(e)} try{drawCard(c)}catch(e){console.error(e)}
  try{drawNews(c)}catch(e){console.error(e)} try{drawMI(c)}catch(e){console.error(e)} }
window.addEventListener('load', drawAll);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
"""


def findings(data: dict) -> dict:
    P = data["panels"]
    f = {}
    s = P.get("state", {})
    if s.get("status") == "ok":
        f["state"] = (f"Where BTC's state sits now and where it is pushed: the two leading components explain {round(100 * sum(s['ev']))}% of the "
                      f"variance; orange arrows are the mean 24 h move per region, green ellipses the 24 h spread.")
    dr = P.get("drivers", {})
    if dr.get("status") == "ok":
        m = [v for v in dr["how"]["mult"] if v is not None]
        g = np.array([[np.nan if v is None else v for v in r] for r in dr["grid"]["mult"]], float)
        f["drivers"] = (f"The field's 1h variance swings x{min(m):.2f} to x{max(m):.2f} of its weekly mean across the week; when momentum and trend "
                        f"agree (same sign) variance rises, up to x{np.nanmax(g):.2f} the neutral state at the |3| corners; when they disagree it falls (x{np.nanmin(g):.2f}). "
                        "Other coordinates held at now; the corners are rare states, so read them as the polynomial's extrapolation.")
        if "state" in f:
            f["state"] += " " + dr["headline"]
    la = P.get("latent", {})
    if la.get("status") == "ok":
        m = la["err_summary"]
        f["latent"] = (f"The network's latent path and its predicted next step; over the last 30 days the median feedback error is "
                       f"{m['model_median']:.2f} vs {m['persistence_median']:.2f} for 'no change'.")
    c = P.get("card", {})
    if c.get("field"):
        h = c["field"]["horizons"]
        k = "24h" if "24h" in h else list(h)[-1]
        f["card"] = (f"Over {k} the field puts the 5-95% return range at {h[k]['quantiles_pct']['q05']:+.1f}% to {h[k]['quantiles_pct']['q95']:+.1f}%; "
                     f"P(touch -2%) {h[k]['p_touch_pct'].get('-2%', float('nan')):.0%}, P(touch +2%) {h[k]['p_touch_pct'].get('+2%', float('nan')):.0%}.")
    n = P.get("news", {})
    if n.get("status") == "ok":
        f["news"] = "News attention and scheduled events against BTC's price and realised vol (no direction is implied: H-NEWS2 is a registered null)."
    m = P.get("mi", {})
    if m.get("status") == "ok":
        arr = np.array([[np.nan if v is None else v for v in r] for r in m["mi"]], float)
        if np.isfinite(arr).any():
            i, j = np.unravel_index(np.nanargmax(arr), arr.shape)
            f["mi"] = f"Which state variables the latent dims encode: strongest link {m['rows'][i]} - {m['cols'][j]} ({arr[i, j]:.2f} bits)."
    return f


def render(data: dict) -> str:
    data = _clean(data)
    fd = findings(data)
    meta = data["meta"]
    payload = json.dumps(data, default=str, allow_nan=False).replace("</", "<\\/")

    def card(pid, title, extra_ids="", cls="plot", wide=False):
        return (f'<section class="card{" wide" if wide else ""}"><h2>{title}</h2><p class="finding">{fd.get(pid, "")}</p>'
                f'<div id="p-{pid}" class="{cls}"></div>{extra_ids}<div id="t-{pid}"></div></section>')
    src = ", ".join(f"{k}: {v}" for k, v in meta.get("sources", {}).items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Field Dashboard</title><style>{CSS}</style>
<script src="{PLOTLY}" charset="utf-8"></script></head>
<body><main>
<h1>BTC field dashboard</h1>
<p class="sub">State space, network latent space, distribution card and news for BTCUSDT, as of {meta['as_of']}. Research only: no direction calls; position size comes from the risk engine.</p>
<div class="grid">
{card("state", "1 · State space and its field", cls="plot tall", wide=True)}
{card("drivers", "1b · What moves the field's variance: hour of week, momentum x trend", extra_ids='<div id="p-grid" class="plot"></div>', wide=True)}
{card("latent", "2 · Network latent trajectory", extra_ids='<div id="p-err" class="plot short"></div>')}
{card("card", "3 · Distribution card", extra_ids='<div id="p-conf" class="plot short"></div>')}
{card("news", "4 · News intensity and events on the BTC timeline", cls="plot tall", wide=True)}
{card("mi", "5 · What the latent dims encode (mutual information)", wide=True)}
</div>
<p class="meta">Built {meta['built_at']} by scripts/build_dashboard.py (BTC field program, agent F3). Sources: {src}. Regenerate with --as-of.</p>
</main>
<script id="data" type="application/json">{payload}</script>
<script>{JS}</script>
</body></html>
"""


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--out", default=str(ROOT / "research" / "dashboard" / "btc_field.html"))
    ap.add_argument("--wait-min", type=float, default=0.0, help="poll up to N minutes for F1 / F2 artifacts")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of).replace(tzinfo=UTC) if a.as_of else datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    if a.wait_min > 0:
        t_end = time.time() + 60 * a.wait_min
        while time.time() < t_end and (latest_field_json() is None or latent_file() is None):
            print("waiting for F1 / F2 artifacts ...", flush=True)
            time.sleep(60)
    data = synthetic_inputs() if a.synthetic else gather(as_of)
    data = _clean(data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    st = {k: v.get("status", "ok" if v else "pending") for k, v in data["panels"].items()}
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB); panels {st}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
