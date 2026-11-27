"""WS14 INTERP: mechanistic interpretability of the live networks (specification program 2026-11-27). Research only.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.ws14_interp core      H-WS14-1..5 (PNN part)
    ... probe       H-WS14-6, 7, 9 (probes, ablations, integrated gradients, patching)
    ... failure     H-WS14-8 (release hours, BTC v0 cells)
    ... xs          H-WS14-10 and the xs part of H-WS14-5
    ... report      research/reports/ws14_2026-11-27.md from the JSON results

Design frozen in the diary ('WS14 design frozen before results') and research/hypotheses_pending/WS14.yaml. Reads the
frozen checkpoints in data/models (never writes there); outputs go to data/research/ws14/.

Networks. The live PNN (live_nn.v1 and v2.1 share it): data/models/pnn/wf_<h>/<month>.pt, 2 seeds, quantiles averaged;
inputs standardised with the checkpoint's fit-block mean / sd and clipped to +/- 6 (pnn_study / live.nn). sigma =
(q90 - q10) / 2.5631 in model units (y = r / sqrt(s2) at 1h, ret / exp(lrv_w / 2) at 1d).

Distillation (H-WS14-1..3). Per refit, OLS of the network's own log sigma (also q50 and the log tail ratio
(q95 - q05) / (q90 - q10)) on a feature set over the refit's fit-block rows; scored on the refit's test month.
R^2 pooled over the segment's test rows, SST around the segment mean; 10-day moving-block bootstrap over dates.
Rebuilt surrogate quantiles: q50_s + w_s c_tau (w = q90 - q10, c_tau the fit-block mean standardised offsets), tails
split by the fit-block mean left share of the tail extension e = tr - 1. Non-inferiority on realised pinball with margin
0.25 x (log-HAR-t - PNN), NW mean / SE over per-date means.

Monotonicity (H-WS14-4). Rows of the live 2026-11 refit whose last bar was not itself a jump; x = |r_{t-1}| / sqrt(s2)
on 0..8 (step 0.25); lu_1 = log(max(x^2, 1e-4)), lu_m6 / lu_m24 shift their mean by (x^2 - exp(lu_1)) / 6 or / 24,
x > 4 sets jump24 = 1 and since_jump = 1 / 168; all other inputs fixed.

Invariance (H-WS14-5). OHLC x c through enrich_bars / add_robust_vol / symbol_hourly and the network.
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

NORM80 = 2.5631031310892007
CUT = np.datetime64("2026-07-01T00:00:00", "us")
TAG = "2026-11-27"
LIVE_MONTH = "2026-11"
HAR_1H = ("lsig2", "lfac", "har_d", "har_w", "har_m")
HAR_1D = ("log_sigma", "lrv_d", "lrv_w", "lrv_m", "lrv_q", "next_wk_hours", "dow_sin", "dow_cos")
GRID_X = np.arange(0.0, 8.0001, 0.25)


def out_dir() -> Path:
    from ai_trader.core.config import load

    p = load().root / "data" / "research" / "ws14"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return None if not math.isfinite(o) else float(f"{o:.7g}")
    return o


def save(name: str, obj: dict) -> Path:
    p = out_dir() / f"{name}.json"
    p.write_text(json.dumps(_json(obj), indent=1), encoding="utf-8")
    return p


# ------------------------------------------------------------------ statistics


def nw(d: np.ndarray) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    m, t, n = newey_west_t(np.asarray(d, float))
    se = abs(m / t) if (t is not None and np.isfinite(t) and t != 0) else float("nan")
    return {"mean": m, "t": t, "n": n, "se": se}


def per_date(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    d = pl.DataFrame({"d": day[ok].astype("datetime64[D]").astype(np.int64), "x": x[ok]}).group_by("d").agg(pl.col("x").mean()).sort("d")
    return d["x"].to_numpy()


def r2(y: np.ndarray, f: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - float(np.sum((y - f) ** 2)) / sst if sst > 0 else float("nan")


def r2_boot(y: np.ndarray, f: np.ndarray, day: np.ndarray, block: int = 10, reps: int = 500, seed: int = 0) -> list[float]:
    """95% moving-block (dates) bootstrap CI of the pooled R^2."""
    from ai_trader.research.decision_budget_study import block_bootstrap_idx

    ok = np.isfinite(y) & np.isfinite(f)
    df = pl.DataFrame({"d": day[ok].astype("datetime64[D]").astype(np.int64), "y": y[ok], "e2": (y[ok] - f[ok]) ** 2})
    g = df.group_by("d").agg(pl.len().alias("n"), pl.col("y").sum().alias("sy"), (pl.col("y") ** 2).sum().alias("syy"),
                             pl.col("e2").sum().alias("sse")).sort("d")
    n, sy, syy, sse = (g[c].to_numpy().astype(float) for c in ("n", "sy", "syy", "sse"))
    if len(n) < 5:
        return [float("nan"), float("nan")]
    idx = block_bootstrap_idx(len(n), block, reps, seed)
    N, SY, SYY, SSE = n[idx].sum(1), sy[idx].sum(1), syy[idx].sum(1), sse[idx].sum(1)
    v = 1 - SSE / (SYY - SY ** 2 / N)
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    from ai_trader.ml import pnn

    return pnn.pinball_np(y, q)


# ------------------------------------------------------------------ PNN loading / forward


def pnn_dir() -> Path:
    from ai_trader.core.config import load

    return load().root / "data" / "models" / "pnn"


class Refit:
    """One walk-forward refit of the PNN (read-only view of its checkpoint)."""

    def __init__(self, h: str, month: str):
        import torch

        from ai_trader.ml import pnn

        self.h, self.month = h, month
        d = pnn_dir() / f"wf_{h}"
        ck = torch.load(d / f"{month}.pt", weights_only=False)
        self.ck = ck
        self.fits = [pnn.QFit("pnn", s, ck["pnn_meta"], 0.0, 0) for s in ck["pnn"]]
        self.models = [f.model() for f in self.fits]
        self.mu = np.asarray(ck["std"]["mu"], float)
        self.sd = np.asarray(ck["std"]["sd"], float)
        self.fit_rows = np.arange(ck["fit_idx_range"][0], ck["fit_idx_range"][1] + 1)
        self.val_rows = np.arange(ck["val_idx_range"][0], ck["val_idx_range"][1] + 1)
        z = np.load(d / f"{month}.npz")
        self.test_rows = z["idx"][z["is_test"]]
        self.q_saved = z["q_pnn"][z["is_test"]]
        self.har = ck["har"]
        self.har_test = z["har"][z["is_test"]]  # [loc, scale, df, sd_z] per test row (pnn_study)

    def har_test_q(self) -> np.ndarray:
        from ai_trader.ml import pnn
        from ai_trader.research.pnn_study import har_quantiles

        h = self.har_test
        return har_quantiles(h[:, 0], h[:, 1], h[:, 2], h[:, 3], pnn.TAUS)

    def Z(self, X: np.ndarray) -> np.ndarray:
        from ai_trader.ml import pnn

        return np.clip(np.nan_to_num((np.asarray(X, float) - self.mu) / self.sd, nan=0.0), -pnn.CLIP, pnn.CLIP)

    def forward(self, X: np.ndarray, models=None) -> dict:
        """Per-seed latent z (k, n, 4), raw head o (k, n, 7), quantiles (k, n, 7) and the ensemble quantiles."""
        import torch

        from ai_trader.ml import pnn

        qf = pnn.classes()[3]
        Zt = torch.as_tensor(self.Z(X), dtype=torch.float32)
        zs, os_, qs = [], [], []
        with torch.no_grad():
            for m in models or self.models:
                z = m.enc(Zt)
                o = m.head(z)
                zs.append(z.numpy().astype(float))
                os_.append(o.numpy().astype(float))
                qs.append(qf(o).numpy().astype(float))
        q = np.mean(qs, axis=0)
        return {"z": np.stack(zs), "o": np.stack(os_), "q_seed": np.stack(qs), "q": q}

    def quantiles(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["q"]


def summaries(q: np.ndarray) -> dict:
    w = q[:, 5] - q[:, 1]
    return {"lsig": np.log(w / NORM80), "q50": q[:, 3], "ltr": np.log((q[:, 6] - q[:, 0]) / w), "w": w}


def months(h: str) -> list[str]:
    return sorted(p.stem for p in (pnn_dir() / f"wf_{h}").glob("*.pt"))


def har_q(har: dict, sd_z: np.ndarray) -> np.ndarray:
    from ai_trader.ml import pnn
    from ai_trader.research.pnn_study import har_quantiles

    n = len(sd_z)
    return har_quantiles(np.full(n, har["loc"]), np.full(n, har["scale"]), np.full(n, har["df"]), sd_z, pnn.TAUS)


def har_sdz(A: dict, har: dict, idx: np.ndarray) -> np.ndarray:
    """log-HAR-t predicted sd of the model-unit target (pnn_study.fit_har_t's logF with the checkpoint's coefficients)."""
    H = A["har"]
    b = np.asarray(har["b"], float)
    X = np.c_[np.ones(len(idx)), H["X"][idx]]
    if A["h"] == "1d":
        logF = X @ b + har["c"]
    else:
        logF = np.log(H["s2"][idx]) + X @ b + har["c"]
    return np.exp(0.5 * logF) / A["sig"][idx]


# ------------------------------------------------------------------ distillation


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    D = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    return b


def _pred(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ b


def _spline_fit(X: np.ndarray, y: np.ndarray, binary: np.ndarray):
    """Additive cubic-spline surrogate (5 knots per continuous input, ridge 1e-3): the GAM-lite of the descriptive arm."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer

    cont = ~binary
    st = SplineTransformer(n_knots=5, degree=3, extrapolation="linear").fit(X[:, cont])

    def design(Xn):
        return np.c_[st.transform(Xn[:, cont]), Xn[:, binary]]

    m = Ridge(alpha=1e-3).fit(design(X), y)
    return lambda Xn: m.predict(design(Xn))


def rebuild_quantiles(q50: np.ndarray, w: np.ndarray, tr: np.ndarray, shape: dict) -> np.ndarray:
    c10 = shape["c10"]
    e = np.maximum(tr - 1.0, 0.01)
    lam = shape["lam"]
    q10, q90 = q50 + c10 * w, q50 + (c10 + 1.0) * w
    q25, q75 = q50 + shape["c25"] * w, q50 + shape["c75"] * w
    q05, q95 = q10 - lam * e * w, q90 + (1 - lam) * e * w
    q = np.c_[q05, q10, q25, q50, q75, q90, q95]
    return np.sort(q, axis=1)


def fit_shape(q: np.ndarray) -> dict:
    w = q[:, 5] - q[:, 1]
    e = (q[:, 6] - q[:, 0]) / w - 1
    left = (q[:, 1] - q[:, 0]) / w
    return {"c10": float(np.mean((q[:, 1] - q[:, 3]) / w)), "c25": float(np.mean((q[:, 2] - q[:, 3]) / w)),
            "c75": float(np.mean((q[:, 4] - q[:, 3]) / w)), "lam": float(np.mean(left) / np.mean(e))}


def distill(h: str, sets: dict[str, tuple], spline: bool = False, live_rows: dict | None = None) -> dict:
    """Per refit: fit the surrogates on fit rows to the network's outputs, score on the test month."""
    from ai_trader.research.pnn_study import FEATURES, arrays

    A = arrays(h)
    feats = list(FEATURES[h])
    X = A["X"]
    binary = np.array([len(np.unique(X[:5000, j])) <= 2 for j in range(X.shape[1])])
    rows = {k: [] for k in ("idx", "seg", "lsig", "q50", "ltr", "y", "pin_net", "pin_har")}
    sur = {s: {"lsig": [], "q50": [], "ltr": [], "pin": []} for s in sets}
    if spline:
        sur["spline_all"] = {"lsig": [], "q50": [], "ltr": [], "pin": []}
    per_refit, coefs, repro = [], {}, []
    for mo in months(h):
        R = Refit(h, mo)
        te, fi = R.test_rows, R.fit_rows
        qf, qt = R.quantiles(X[fi]), R.quantiles(X[te])
        repro.append(float(np.max(np.abs(qt - R.q_saved))))
        Sf, St = summaries(qf), summaries(qt)
        y = A["y"][te]
        rows["idx"].append(te)
        rows["seg"].append(np.where(A["T"][te] < CUT, "llm_in_sample", "clean_holdout"))
        for k in ("lsig", "q50", "ltr"):
            rows[k].append(St[k])
        rows["y"].append(y)
        rows["pin_net"].append(pinball(y, qt))
        rows["pin_har"].append(pinball(y, R.har_test_q()))
        shape = fit_shape(qf)
        rr = {"month": mo, "n_fit": len(fi), "n_test": len(te)}
        allsets = dict(sets)
        for s, cols in allsets.items():
            j = [feats.index(c) for c in cols]
            pr = {}
            for k in ("lsig", "q50", "ltr"):
                b = _ols(X[fi][:, j], Sf[k])
                pr[k] = _pred(b, X[te][:, j])
                if k == "lsig":
                    coefs.setdefault(s, {})[mo] = b.tolist()
                    rr[f"r2_{s}"] = r2(St[k], pr[k])
                sur[s][k].append(pr[k])
            qs = rebuild_quantiles(pr["q50"], NORM80 * np.exp(pr["lsig"]), np.exp(pr["ltr"]), shape)
            sur[s]["pin"].append(pinball(y, qs))
            if live_rows is not None and mo == LIVE_MONTH:
                L = live_rows
                Sl = summaries(R.quantiles(L["X"]))
                pl_ = {k: _pred(_ols(X[fi][:, j], Sf[k]), L["X"][:, j]) for k in ("lsig", "q50", "ltr")}
                ql = rebuild_quantiles(pl_["q50"], NORM80 * np.exp(pl_["lsig"]), np.exp(pl_["ltr"]), shape)
                L.setdefault("res", {})[s] = {"r2_lsig": r2(Sl["lsig"], pl_["lsig"]), "r2_q50": r2(Sl["q50"], pl_["q50"]),
                                              "pin_sur": float(np.mean(pinball(L["y"], ql))), "pin_net": float(np.mean(pinball(L["y"], R.quantiles(L["X"]))))}
        if spline:
            pr = {}
            for k in ("lsig", "q50", "ltr"):
                f = _spline_fit(X[fi], Sf[k], binary)
                pr[k] = f(X[te])
                sur["spline_all"][k].append(pr[k])
            rr["r2_spline_all"] = r2(St["lsig"], pr["lsig"])
            qs = rebuild_quantiles(pr["q50"], NORM80 * np.exp(pr["lsig"]), np.exp(pr["ltr"]), shape)
            sur["spline_all"]["pin"].append(pinball(y, qs))
        per_refit.append(rr)
        print(json.dumps({"h": h, "month": mo, **{k: round(v, 4) for k, v in rr.items() if k.startswith("r2")}}), flush=True)
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    day = A["day"][cat["idx"]]
    out = {"h": h, "n_refits": len(per_refit), "max_abs_repro_err": max(repro), "per_refit": per_refit, "coef_lsig": coefs, "segments": {}}
    for seg in ("llm_in_sample", "clean_holdout"):
        k = cat["seg"] == seg
        dd = day[k]
        gain = per_date(cat["pin_har"][k] - cat["pin_net"][k], dd)
        g = nw(gain)
        margin = 0.25 * g["mean"]
        so = {"rows": int(k.sum()), "dates": len(np.unique(dd)), "pinball_net": float(cat["pin_net"][k].mean()),
              "pinball_har": float(cat["pin_har"][k].mean()), "har_minus_net": g, "margin": margin, "surrogates": {}}
        for s, v in sur.items():
            c = {kk: np.concatenate(vv) for kk, vv in v.items()}
            d = nw(per_date(c["pin"][k] - cat["pin_net"][k], dd))
            ub, lb = d["mean"] + 1.645 * d["se"], d["mean"] - 1.645 * d["se"]
            so["surrogates"][s] = {
                "r2_lsig": r2(cat["lsig"][k], c["lsig"][k]), "r2_lsig_ci": r2_boot(cat["lsig"][k], c["lsig"][k], dd),
                "r2_q50": r2(cat["q50"][k], c["q50"][k]), "r2_ltr": r2(cat["ltr"][k], c["ltr"][k]),
                "r2_lsig_raw": r2(cat["lsig"][k] + 0.5 * np.log(A["sig"][cat["idx"]][k] ** 2), c["lsig"][k] + 0.5 * np.log(A["sig"][cat["idx"]][k] ** 2)),
                "pinball": float(c["pin"][k].mean()), "sur_minus_net": d, "upper95": ub, "lower95": lb,
                "noninferior": bool(ub < margin), "inferior": bool(lb > margin),
                "share_of_gain_kept": float((cat["pin_har"][k].mean() - c["pin"][k].mean()) / (cat["pin_har"][k].mean() - cat["pin_net"][k].mean())),
                "median_refit_r2": float(np.median([r[f"r2_{s}"] for r in per_refit if (np.datetime64(r["month"] + "-01") < CUT) == (seg == "llm_in_sample")]))}
        out["segments"][seg] = so
    if live_rows is not None:
        out["live"] = {"rows": int(len(live_rows["y"])), "first": live_rows["first"], "last": live_rows["last"], "res": live_rows.get("res", {}),
                       "consistency": live_rows.get("consistency")}
    return out


# ------------------------------------------------------------------ live rows (point in time, lake)


def live_rows_1h(as_of: datetime) -> dict:
    """ml.hourly features of every closed BTC bar since the live refit month, exactly as live.nn builds them."""
    from ai_trader.live import nn
    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.frame import market_frame

    f1 = market_frame("BTCUSDT", "1h", as_of, funding=False, oi=False)
    H, _S = nn.hourly_frame(f1)
    H = H.filter(pl.col("event_ts") >= datetime(2026, 11, 1, tzinfo=UTC))
    cols = [*HOURLY_FEATURES, "s2", "r"]
    H = H.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cols]) & (pl.col("s2") > 0))
    X = np.column_stack([H[c].cast(pl.Float64).to_numpy() for c in HOURLY_FEATURES])
    s2 = H["s2"].to_numpy()
    ts = H["event_ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy()
    return {"X": X, "y": H["r"].to_numpy() / np.sqrt(s2), "s2": s2, "ts": ts, "first": str(ts[0]), "last": str(ts[-1])}


def live_consistency(L: dict) -> dict:
    """Max abs feature difference between the lake rebuild and (a) the study cache, (b) the live v1 journal inputs."""
    from ai_trader.core.config import load
    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.pnn_study import arrays

    A = arrays("1h")
    pos = {t: i for i, t in enumerate(A["T"])}
    common = [(i, pos[t]) for i, t in enumerate(L["ts"]) if t in pos]
    out = {"cache_rows": len(common)}
    if common:
        a, b = zip(*common, strict=True)
        out["cache_max_abs"] = float(np.max(np.abs(L["X"][list(a)] - A["X"][list(b)])))
    j = load().root / "data" / "journal" / "live_nn"
    diffs, n = [], 0
    lpos = {str(t)[:13]: i for i, t in enumerate(L["ts"])}
    for f in sorted(j.glob("2026-11-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = r["forecast_hour"][:13]
            if k in lpos and "replay" not in r.get("flags", []):
                x = np.array([r["inputs"]["1h"][c] for c in HOURLY_FEATURES], float)
                if np.isfinite(x).all():
                    diffs.append(np.abs(x - L["X"][lpos[k]]))
                    n += 1
    if diffs:
        D = np.stack(diffs)
        out["journal_lines"] = n
        out["journal_max_abs"] = float(D.max())
        out["journal_share_rows_diff_gt_1e-6"] = float(np.mean(D.max(1) > 1e-6))
        out["journal_worst_feature"] = HOURLY_FEATURES[int(np.argmax(D.max(0)))]
    return out


# ------------------------------------------------------------------ monotonicity (H-WS14-4)


def perturb_last_return(X: np.ndarray, x: float, feats: list[str]) -> np.ndarray:
    Xn = X.copy()
    i1, i6, i24, ij, isj = (feats.index(c) for c in ("lu_1", "lu_m6", "lu_m24", "jump24", "since_jump"))
    u_old = np.exp(X[:, i1])
    u_new = x * x
    Xn[:, i1] = np.log(max(u_new, 1e-4))
    Xn[:, i6] = np.log(np.maximum(np.exp(X[:, i6]) + (u_new - u_old) / 6, 1e-12))
    Xn[:, i24] = np.log(np.maximum(np.exp(X[:, i24]) + (u_new - u_old) / 24, 1e-12))
    if x > 4.0:
        Xn[:, ij] = 1.0
        Xn[:, isj] = 1.0 / 168
    return Xn


def monotonicity(R: Refit, X: np.ndarray, ts: np.ndarray, y: np.ndarray) -> dict:
    from ai_trader.ml.hourly import HOURLY_FEATURES

    feats = list(HOURLY_FEATURES)
    keep = np.exp(X[:, feats.index("lu_1")]) < 16.0  # last bar not itself a jump (|r| / s < 4)
    X, ts, y = X[keep], ts[keep], y[keep]
    S = np.stack([np.exp(summaries(R.quantiles(perturb_last_return(X, x, feats)))["lsig"]) for x in GRID_X], axis=1)  # (n, G)
    drop = np.maximum(0.0, (S[:, :-1] - S[:, 1:]) / S[:, :-1]).max(axis=1)
    where = np.argmax(np.maximum(0.0, (S[:, :-1] - S[:, 1:]) / S[:, :-1]), axis=1)
    ratio = S / S[:, :1]
    # empirical response: realised y^2 by bins of the observed x (descriptive)
    xo = np.sqrt(np.exp(X[:, feats.index("lu_1")]))
    out = {"rows_excluded_jump": int((~keep).sum()), "grid": GRID_X.tolist(), "segments": {}}
    for seg, k in (("llm_in_sample", ts < CUT), ("clean_holdout", ts >= CUT)):
        if not k.any():
            continue
        d = drop[k]
        out["segments"][seg] = {"rows": int(k.sum()), "share_drop_gt_0.5pct": float(np.mean(d > 0.005)), "share_drop_gt_2pct": float(np.mean(d > 0.02)),
                                "max_drop": float(d.max()), "median_ratio_profile": np.median(ratio[k], axis=0).tolist(),
                                "drop_location_hist": np.bincount(where[k][d > 0.005], minlength=len(GRID_X) - 1).tolist()}
    bins = [0, 0.5, 1, 1.5, 2, 3, 4]
    emp = []
    s0 = S[:, 0]
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        m = (xo >= a) & (xo < b)
        if m.sum() > 30:
            gi = np.searchsorted(GRID_X, 0.5 * (a + b))
            emp.append({"bin": [a, b], "n": int(m.sum()), "realised_rms_y": float(np.sqrt(np.mean(y[m] ** 2))),
                        "net_sigma_at_observed": float(np.median(np.exp(summaries(R.quantiles(X[m]))["lsig"]))),
                        "net_ratio_vs_x0_median": float(np.median(S[m, gi] / s0[m]))})
    out["empirical_vs_net"] = emp
    return out


# ------------------------------------------------------------------ invariance (H-WS14-5, PNN part)


def invariance_pnn(R: Refit) -> dict:
    from ai_trader.core.config import load
    from ai_trader.ml.hourly import HOURLY_FEATURES, symbol_hourly
    from ai_trader.research.frame import add_robust_vol, enrich_bars

    f = pl.read_parquet(load().root / "data" / "models" / "xs_v3" / "cache" / "f1h_BTCUSDT.parquet")
    f = f.filter(pl.col("event_ts") >= datetime(2025, 6, 1, tzinfo=UTC)).select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close")
    outs = {}
    for c in (1.0, 1e-3, 1e3):
        b = f.with_columns([pl.col(k) * c for k in ("open", "high", "low", "close")])
        g = add_robust_vol(enrich_bars(b)).select("event_ts", "ret", "sigma_risk_next")
        hh = symbol_hourly(g)
        hh = hh.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in HOURLY_FEATURES]))
        X = np.column_stack([hh[k].cast(pl.Float64).to_numpy() for k in HOURLY_FEATURES])
        outs[c] = (hh["event_ts"].to_numpy(), X, R.quantiles(X))
    t0, X0, q0 = outs[1.0]
    res = {"rows": len(t0)}
    for c in (1e-3, 1e3):
        t, Xc, qc = outs[c]
        assert np.array_equal(t, t0)
        w0 = q0[:, 5] - q0[:, 1]
        res[str(c)] = {"max_rel_q": float(np.max(np.abs(qc - q0) / np.maximum(np.abs(q0), 1e-12))), "max_abs_feature": float(np.max(np.abs(Xc - X0))),
                       "max_abs_q_over_width": float(np.max(np.abs(qc - q0) / w0[:, None])), "features_bitwise_equal": bool(np.array_equal(Xc, X0))}
    return res


# ------------------------------------------------------------------ probes, ablations, IG, patching (H-WS14-6, 7, 9)

GROUPS_1H = {"clock": ("lfac", "dow_sin", "dow_cos"), "base_sigma": ("lsig2",), "funding_hour": ("fund_after", "fund_before"),
             "recent_returns": ("lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24", "lu_m6", "lu_m24"),
             "jump": ("jump24", "since_jump"), "daily_har": ("har_d", "har_w", "har_m")}


def btc_side(ts_us: np.ndarray) -> dict:
    """BTC v0 cell and fund_z90 at each hour (xs_v3 panel, point in time; NaN / '' outside it)."""
    from ai_trader.core.config import load

    d = np.load(load().root / "data" / "models" / "xs_v3" / "cache" / "panel_2026-11-15.npz", allow_pickle=False)
    pts = d["ts"]
    names = list(d["features"])
    sym = list(d["symbols"])
    fz = d["X"][:, sym.index("BTCUSDT"), names.index("fund_z90")].astype(float)
    cell = d["cell"]
    t = ts_us.astype("datetime64[us]").astype(np.int64)
    j = np.searchsorted(pts, t)
    ok = (j < len(pts)) & (pts[np.minimum(j, len(pts) - 1)] == t)
    jj = np.minimum(j, len(pts) - 1)
    return {"fund_z90": np.where(ok, fz[jj], np.nan), "cell": np.where(ok, cell[jj], "")}


def ridge_probe(Af: np.ndarray, yf: np.ndarray, Ae: np.ndarray, ye: np.ndarray) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    ok_f, ok_e = np.isfinite(yf) & np.isfinite(Af).all(1), np.isfinite(ye) & np.isfinite(Ae).all(1)
    mu, sd = Af[ok_f].mean(0), Af[ok_f].std(0) + 1e-12
    m = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=KFold(5, shuffle=False)).fit((Af[ok_f] - mu) / sd, yf[ok_f])
    return r2(ye[ok_e], m.predict((Ae[ok_e] - mu) / sd))


def class_probe(Af, yf, Ae, ye) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    mu, sd = Af.mean(0), Af.std(0) + 1e-12
    m = LogisticRegression(max_iter=2000, C=1.0).fit((Af - mu) / sd, yf)
    return float(balanced_accuracy_score(ye, m.predict((Ae - mu) / sd)))


def random_pnn(seed: int, n_in: int = 19):
    import torch

    from ai_trader.ml import pnn

    torch.manual_seed(seed)
    m = pnn.make_pnn(n_in)
    m.eval()
    return m


def latent_of(R: Refit, X: np.ndarray, models) -> np.ndarray:
    z = R.forward(X, models)["z"]  # (k, n, 4)
    return np.concatenate(list(z), axis=1)


def lsig_fn(models, Zt):
    """log sigma (model units) of the ensemble mean quantiles, differentiable in the standardised input Zt."""
    import torch

    from ai_trader.ml import pnn

    qf = pnn.classes()[3]
    q = torch.stack([qf(m.head(m.enc(Zt))) for m in models]).mean(0)
    return torch.log((q[:, 5] - q[:, 1]) / NORM80)


def integrated_gradients(models, Z: np.ndarray, steps: int = 64) -> tuple[np.ndarray, np.ndarray]:
    import torch

    Zt = torch.as_tensor(Z, dtype=torch.float32)
    base = torch.zeros_like(Zt)
    tot = torch.zeros_like(Zt)
    for k in range(steps):
        a = (k + 0.5) / steps
        x = (base + a * (Zt - base)).requires_grad_(True)
        f = lsig_fn(models, x).sum()
        (g,) = torch.autograd.grad(f, x)
        tot += g
    ig = ((Zt - base) * tot / steps).numpy().astype(float)
    with torch.no_grad():
        delta = (lsig_fn(models, Zt) - lsig_fn(models, base)).numpy().astype(float)
    return ig, delta


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ra = np.apply_along_axis(rankdata, 1, a)
    rb = np.apply_along_axis(rankdata, 1, b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-300)


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def retrain_permuted(R: Refit, A: dict, seed_base: int = 900) -> list:
    """Data randomisation (Adebayo): the 2 seeds retrained with the study's protocol on permuted labels."""
    from ai_trader.ml import pnn

    s = json.loads((pnn_dir() / "search.json").read_text(encoding="utf-8"))["1h"]["best"]["pnn"]
    cfg = {"lr": s["lr"], "weight_decay": s["weight_decay"], "batch": 1024}
    rng = np.random.default_rng(seed_base)
    fi, vi = R.fit_rows, R.val_rows
    yf, yv = rng.permutation(A["y"][fi]), rng.permutation(A["y"][vi])
    fits = [pnn.fit_pnn(R.Z(A["X"][fi]), yf, R.Z(A["X"][vi]), yv, seed=k, cfg=cfg) for k in range(2)]
    return [f.model() for f in fits], [f.epochs for f in fits]


def group_resample(R: Refit, X: np.ndarray, y: np.ndarray, cols: list[int], draws: int = 5, seed: int = 0) -> np.ndarray:
    """Per-row pinball increase when `cols` are jointly replaced by those of random other rows (mean over draws)."""
    rng = np.random.default_rng(seed)
    base = pinball(y, R.quantiles(X))
    inc = np.zeros(len(y))
    for _ in range(draws):
        Xa = X.copy()
        Xa[:, cols] = X[rng.permutation(len(X))][:, cols]
        inc += pinball(y, R.quantiles(Xa)) - base
    return inc / draws


def cmd_probe() -> dict:
    import torch

    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.pnn_study import arrays

    t0 = time.time()
    feats = list(HOURLY_FEATURES)
    A = arrays("1h")
    R = Refit("1h", LIVE_MONTH)
    rng = np.random.default_rng(0)
    fi = R.fit_rows[rng.choice(len(R.fit_rows), 20000, replace=False)]
    fi.sort()
    ev = np.r_[R.val_rows, R.test_rows]
    seg_ev = np.where(A["T"][ev] < CUT, "llm_in_sample", "clean_holdout")
    out: dict = {"fit_rows": len(fi), "eval_rows": len(ev), "eval_first": str(A["T"][ev].min()), "eval_last": str(A["T"][ev].max())}

    # ---- targets
    side = btc_side(A["T"])
    how = ((A["T"].astype("datetime64[h]").astype(np.int64) + 72) % 168)  # hour of week (Mon 00 = 0); epoch was a Thursday
    ctrl_val = np.random.default_rng(7).normal(size=168)
    lab = A["X"][:, feats.index("lfac")]
    blocks = (np.arange(len(A["T"])) // 168)
    perm_b = np.random.default_rng(11).permutation(blocks.max() + 1)
    order = np.argsort(perm_b[blocks], kind="stable")
    targets = {
        "lfac": lab,
        "abs_weekly_lrv": A["X"][:, feats.index("har_w")] + A["X"][:, feats.index("lsig2")],
        "abs_rv24": A["X"][:, feats.index("lu_m24")] + A["X"][:, feats.index("lsig2")],
        "hour_of_day_sin": np.sin(2 * np.pi * (how % 24) / 24),
        "control_random_how": ctrl_val[how],
        "shuffled_lfac": lab[order],
        "fund_z90": side["fund_z90"],
    }
    acts = {"trained": (latent_of(R, A["X"][fi], R.models), latent_of(R, A["X"][ev], R.models))}
    rand_sets = []
    for k in range(10):
        ms = [random_pnn(1000 + 2 * k), random_pnn(1001 + 2 * k)]
        rand_sets.append((latent_of(R, A["X"][fi], ms), latent_of(R, A["X"][ev], ms)))
    inp = (R.Z(A["X"][fi]), R.Z(A["X"][ev]))
    probes = {}
    for name, v in targets.items():
        yf, ye = v[fi], v[ev]
        row = {}
        for seg in ("llm_in_sample", "clean_holdout"):
            k = seg_ev == seg
            tr = ridge_probe(acts["trained"][0], yf, acts["trained"][1][k], ye[k])
            rr = [ridge_probe(a, yf, b[k], ye[k]) for a, b in rand_sets]
            ip = ridge_probe(inp[0], yf, inp[1][k], ye[k])
            row[seg] = {"r2_trained": tr, "r2_random_mean": float(np.mean(rr)), "r2_random_max": float(np.max(rr)), "selectivity": tr - float(np.mean(rr)),
                        "r2_inputs_linear": ip, "latent_minus_inputs": tr - ip}
        probes[name] = row
        print(name, json.dumps(_json(row)), flush=True)
    # regime vol tier (3 classes) from the v0 cell
    tier = np.array([{"1": 0, "4": 0, "7": 0, "2": 1, "5": 1, "8": 1, "3": 2, "6": 2, "9": 2}.get(c[-1:], -1) if c else -1 for c in side["cell"]])
    cls = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        k = seg_ev == seg
        okf, oke = tier[fi] >= 0, tier[ev][k] >= 0
        a_tr = class_probe(acts["trained"][0][okf], tier[fi][okf], acts["trained"][1][k][oke], tier[ev][k][oke])
        a_rd = [class_probe(a[okf], tier[fi][okf], b[k][oke], tier[ev][k][oke]) for a, b in rand_sets[:5]]
        a_in = class_probe(inp[0][okf], tier[fi][okf], inp[1][k][oke], tier[ev][k][oke])
        cls[seg] = {"bal_acc_trained": a_tr, "bal_acc_random_mean": float(np.mean(a_rd)), "bal_acc_inputs": a_in, "chance": 1 / 3, "n": int(oke.sum())}
    probes["vol_tier_v0"] = cls
    out["probes"] = probes

    # ---- ablations across walk-forward test months (H-WS14-7)
    abl = {g: [] for g in GROUPS_1H}
    seg_all, day_all = [], []
    for mo in months("1h"):
        Rm = Refit("1h", mo)
        te = Rm.test_rows
        for g, cols in GROUPS_1H.items():
            abl[g].append(group_resample(Rm, A["X"][te], A["y"][te], [feats.index(c) for c in cols], seed=hash(mo) % 1000))
        seg_all.append(np.where(A["T"][te] < CUT, "llm_in_sample", "clean_holdout"))
        day_all.append(A["day"][te])
    seg_all, day_all = np.concatenate(seg_all), np.concatenate(day_all)
    ab = {}
    for g, v in abl.items():
        v = np.concatenate(v)
        ab[g] = {seg: {**nw(per_date(v[seg_all == seg], day_all[seg_all == seg])), "pct_of_pinball": None} for seg in ("llm_in_sample", "clean_holdout")}
    out["group_ablation"] = ab
    print("ablation", json.dumps(_json(ab))[:800], flush=True)

    # ---- single-input and unit resample ablation on the live refit eval rows (descriptive + H-WS14-9 ranking)
    yev = A["y"][ev]
    single = {c: float(group_resample(R, A["X"][ev], yev, [feats.index(c)], draws=3, seed=3).mean()) for c in feats}
    out["single_input_ablation_live"] = single
    base_pin = pinball(yev, R.quantiles(A["X"][ev]))
    unit = {}
    Zt = torch.as_tensor(R.Z(A["X"][ev]), dtype=torch.float32)
    from ai_trader.ml import pnn as _p

    qf = _p.classes()[3]
    with torch.no_grad():
        zs = [m.enc(Zt) for m in R.models]
        for s in range(2):
            for j in range(4):
                incs = []
                for d in range(3):
                    p = torch.as_tensor(np.random.default_rng(50 + d).permutation(len(ev)))
                    qs = []
                    for s2_, m in enumerate(R.models):
                        z = zs[s2_].clone()
                        if s2_ == s:
                            z[:, j] = z[p, j]
                        qs.append(qf(m.head(z)).numpy().astype(float))
                    incs.append(pinball(yev, np.mean(qs, 0)) - base_pin)
                v = np.mean(incs, 0)
                unit[f"seed{s}.z{j}"] = {seg: nw(per_date(v[seg_ev == seg], A["day"][ev][seg_ev == seg])) for seg in ("llm_in_sample", "clean_holdout")}
    out["unit_ablation_live"] = unit

    # ---- integrated gradients + Adebayo (H-WS14-9)
    pick = np.sort(np.random.default_rng(5).choice(len(ev), 2000, replace=False))
    Zp = R.Z(A["X"][ev][pick])
    ig, delta = integrated_gradients(R.models, Zp)
    comp = float(np.mean(np.abs(ig.sum(1) - delta)) / np.mean(np.abs(delta)))
    mabs = np.abs(ig).mean(0)
    ig_rand, _ = integrated_gradients([random_pnn(77), random_pnn(78)], Zp)
    head_rand = []
    for s, m in enumerate(R.models):
        mm = _p.make_pnn(19)
        mm.load_state_dict(m.state_dict())
        torch.manual_seed(500 + s)
        fresh = _p.make_pnn(19)
        mm.head.load_state_dict(fresh.head.state_dict())
        mm.eval()
        head_rand.append(mm)
    ig_head, _ = integrated_gradients(head_rand, Zp)
    perm_models, perm_epochs = retrain_permuted(R, A)
    ig_perm, _ = integrated_gradients(perm_models, Zp)
    abl_rank = np.array([single[c] for c in feats])
    out["ig"] = {"rows": len(pick), "completeness_rel_err": comp, "mean_abs_ig": dict(zip(feats, mabs.tolist(), strict=True)),
                 "spearman_rowwise_median": {"full_random": float(np.median(rowwise_spearman(np.abs(ig), np.abs(ig_rand)))),
                                             "head_random": float(np.median(rowwise_spearman(np.abs(ig), np.abs(ig_head)))),
                                             "label_permuted": float(np.median(rowwise_spearman(np.abs(ig), np.abs(ig_perm))))},
                 "global_rank_spearman": {"full_random": spearman(mabs, np.abs(ig_rand).mean(0)), "head_random": spearman(mabs, np.abs(ig_head).mean(0)),
                                          "label_permuted": spearman(mabs, np.abs(ig_perm).mean(0)), "vs_ablation": spearman(mabs, abl_rank)},
                 "perm_epochs": perm_epochs,
                 "input_x_spearman_median": float(np.median(rowwise_spearman(np.abs(ig), np.abs(Zp))))}
    print("ig", json.dumps(_json(out["ig"]))[:900], flush=True)

    # ---- activation patching (descriptive extra, declared in the diary before running)
    Xe, te_ = A["X"][ev], A["T"][ev]
    how_e = how[ev]
    lu1, lum6 = Xe[:, feats.index("lu_1")], Xe[:, feats.index("lu_m6")]
    calm = (lu1 < np.quantile(lu1, 1 / 3)) & (lum6 < np.quantile(lum6, 1 / 3))
    shock = lu1 > np.quantile(lu1, 0.9)
    prng = np.random.default_rng(9)
    pairs = []
    for i in np.flatnonzero(shock):
        c = np.flatnonzero(calm & (how_e == how_e[i]))
        if len(c) >= 2:
            a, b = prng.choice(c, 2, replace=False)
            pairs.append((i, a, b))
    pairs = np.array(pairs)
    S_, C_, N_ = pairs[:, 0], pairs[:, 1], pairs[:, 2]

    def f_of(Xm):
        return summaries(R.quantiles(Xm))["lsig"]

    fS, fC = f_of(Xe[S_]), f_of(Xe[C_])
    dlt = fS - fC
    patch = {"pairs": len(pairs), "mean_delta_log_sigma": float(dlt.mean())}
    zS = [m.enc(torch.as_tensor(R.Z(Xe[S_]), dtype=torch.float32)).detach() for m in R.models]
    zC = [m.enc(torch.as_tensor(R.Z(Xe[C_]), dtype=torch.float32)).detach() for m in R.models]
    zN = [m.enc(torch.as_tensor(R.Z(Xe[N_]), dtype=torch.float32)).detach() for m in R.models]

    def from_latents(zz):
        with torch.no_grad():
            q = np.mean([qf(m.head(z)).numpy().astype(float) for m, z in zip(R.models, zz, strict=True)], 0)
        return summaries(q)["lsig"]

    units = {}
    for s in range(2):
        for j in range(4):
            zz = [z.clone() for z in zC]
            zz[s][:, j] = zS[s][:, j]
            zn = [z.clone() for z in zC]
            zn[s][:, j] = zN[s][:, j]
            units[f"seed{s}.z{j}"] = {"restored_share": float(np.mean(from_latents(zz) - fC) / dlt.mean()),
                                      "null_share": float(np.mean(from_latents(zn) - fC) / dlt.mean())}
    best = {}
    for s in range(2):
        top2 = sorted(range(4), key=lambda j: -units[f"seed{s}.z{j}"]["restored_share"])[:2]
        best[f"seed{s}"] = top2
    zz = [z.clone() for z in zC]
    for s in range(2):
        for j in best[f"seed{s}"]:
            zz[s][:, j] = zS[s][:, j]
    patch["top2_per_seed"] = best
    patch["top2_restored_share"] = float(np.mean(from_latents(zz) - fC) / dlt.mean())
    patch["units"] = units
    grp = {}
    for g, cols in GROUPS_1H.items():
        Xp = Xe[C_].copy()
        ci = [feats.index(c) for c in cols]
        Xp[:, ci] = Xe[S_][:, ci]
        Xn = Xe[C_].copy()
        Xn[:, ci] = Xe[N_][:, ci]
        grp[g] = {"restored_share": float(np.mean(f_of(Xp) - fC) / dlt.mean()), "null_share": float(np.mean(f_of(Xn) - fC) / dlt.mean())}
    patch["input_groups"] = grp
    out["patching"] = patch
    out["runtime_s"] = round(time.time() - t0, 1)
    save("probe_1h", out)
    print(json.dumps({"runtime_s": out["runtime_s"]}), flush=True)
    return out


# ------------------------------------------------------------------ failure modes (H-WS14-8)


def cmd_failure() -> dict:
    from scipy.stats import binomtest

    from ai_trader.core.config import load
    from ai_trader.ml import pnn

    d = pl.read_parquet(pnn_dir() / "latent_1h.parquet").sort("ts")
    ts = d["ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy()
    sig = d["sigma_ref"].to_numpy()
    y = d["ret"].to_numpy() / sig
    q = np.column_stack([d[c].to_numpy() for c in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")]) / sig[:, None]
    qh = np.column_stack([d[f"har_{c}"].to_numpy() for c in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")]) / sig[:, None]
    dpin = pnn.pinball_np(y, q) - pnn.pinball_np(y, qh)
    hit80 = (y >= q[:, 1]) & (y <= q[:, 5])
    hit_aci = (d["ret"].to_numpy() >= d["aci_lo80"].to_numpy()) & (d["ret"].to_numpy() <= d["aci_hi80"].to_numpy())
    hit_har = (y >= qh[:, 1]) & (y <= qh[:, 5])
    seg = np.where(ts < CUT, "llm_in_sample", "clean_holdout")
    day = ts.astype("datetime64[D]")
    ev = pl.read_parquet(load().root / "data" / "research" / "news" / "events" / "events.parquet").filter(pl.col("kind").is_in(["cpi", "nfp", "fomc"]))
    evh = set(ev["scheduled_ts"].dt.replace_time_zone(None).dt.truncate("1h").cast(pl.Datetime("us")).to_numpy().tolist())
    rel = np.array([t in evh for t in ts.tolist()])
    post = np.array([(t - np.timedelta64(1, "h")) in evh for t in ts])
    side = btc_side(ts)
    cell = side["cell"]
    out = {"rows": len(ts), "first": str(ts[0]), "last": str(ts[-1])}

    def block(mask) -> dict:
        n = int(mask.sum())
        k80 = int(hit80[mask].sum())
        r = {"n": n, "cov80_raw": k80 / n if n else None, "cov80_aci": float(hit_aci[mask].mean()) if n else None,
             "cov80_har": float(hit_har[mask].mean()) if n else None,
             "binom_p_less_080": float(binomtest(k80, n, 0.8, alternative="less").pvalue) if n else None,
             "rms_y": float(np.sqrt(np.mean(y[mask] ** 2))) if n else None,
             "mean_sigma_net": float(np.mean((q[mask, 5] - q[mask, 1]) / NORM80)) if n else None,
             "pnn_minus_har": nw(per_date(dpin[mask], day[mask])) if n else None}
        return r

    out["release"] = {"all": block(rel), "post_hour": block(post), "non_release": block(~rel & ~post)}
    for kind in ("cpi", "nfp", "fomc"):
        hk = set(ev.filter(pl.col("kind") == kind)["scheduled_ts"].dt.replace_time_zone(None).dt.truncate("1h").cast(pl.Datetime("us")).to_numpy().tolist())
        out["release"][kind] = block(np.array([t in hk for t in ts.tolist()]))
    groups = {"low_R1R4R7": ("R1", "R4", "R7"), "normal_R2R5R8": ("R2", "R5", "R8"), "high_R3R6R9": ("R3", "R6", "R9")}
    cells = {}
    for sg in ("llm_in_sample", "clean_holdout"):
        cells[sg] = {g: block((seg == sg) & np.isin(cell, cs)) for g, cs in groups.items()}
        cells[sg]["by_cell"] = {c: block((seg == sg) & (cell == c)) for c in [f"R{i}" for i in range(1, 10)]}
    out["cells"] = cells
    wd = (ts.astype("datetime64[D]").astype(np.int64) + 3) % 7  # Mon = 0
    out["weekend"] = {sg: {"weekend": block((seg == sg) & (wd >= 5)), "weekday": block((seg == sg) & (wd < 5))} for sg in ("llm_in_sample", "clean_holdout")}
    save("failure_1h", out)
    print(json.dumps(_json(out["release"]))[:1500], flush=True)
    return out


# ------------------------------------------------------------------ xs_v3 / xs_v4 (H-WS14-10, xs part of H-WS14-5)

XS_SUR = ("lsig2", "lfac", "fund_after", "fund_before", "lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24", "lu_m6", "lu_m24",
          "jump24", "since_jump", "har_d", "har_w", "har_m", "dow_sin", "dow_cos", "lfac4", "lfac24", "hod_sin", "hod_cos")
XS_HAR = ("lsig2", "lfac", "har_d", "har_w", "har_m")


def xs_load(kind: str, tag: str) -> dict:
    import torch

    from ai_trader.core.config import load
    from ai_trader.research import xs_network_study as study

    if kind == "v3":
        P = study.load_panel()
        ck = torch.load(load().root / "data" / "models" / "xs_v3" / "wf" / f"{tag}.pt", weights_only=False)
        states, sec = ck["states"]["kan_mix"], study.sec_idx(P)
        newm = lambda: study.new_model(P, "kan_mix")  # noqa: E731
        z = np.load(load().root / "data" / "models" / "xs_v3" / "wf" / f"{tag}.npz")
        saved = z["vol_kan_mix"]
    else:
        from ai_trader.research import xs_v4_study as v4

        P = v4.load_panel()
        ck = torch.load(load().root / "data" / "models" / "xs_v4" / "wf" / f"{tag}.pt", weights_only=False)
        states, sec = ck["states"], v4.sec_idx(P)
        newm = lambda: v4.new_model(P)  # noqa: E731
        z = np.load(load().root / "data" / "models" / "xs_v4" / "wf" / f"{tag}.npz")
        saved = z["vol"]
    models = []
    for s in states:
        m = newm()
        m.load_state_dict(s)
        m.eval()
        models.append(m)
    st = {k: np.asarray(v, float) for k, v in ck["std"].items()}
    return {"P": P, "models": models, "sec": sec, "st": st, "te": z["idx_test"], "saved_vol": saved, "kind": kind, "tag": tag}


def xs_forward(L: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ai_trader.ml import xs_net as xs
    from ai_trader.research import xs_network_study as study

    B = study.block(L["P"], idx, L["st"])
    outs = [xs.predict(m, B, L["sec"]) for m in L["models"]]
    return np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0), B.P


def xs_analyse(kind: str, tag: str) -> dict:
    from ai_trader.research import xs_network_study as study

    t0 = time.time()
    L = xs_load(kind, tag)
    P = L["P"]
    names = list(P["features"])
    q = int(np.datetime64(f"{tag[:4]}-{(int(tag[-1]) - 1) * 3 + 1:02d}-01", "us").astype(np.int64))
    tr = study.train_idx(P, q)
    fi, _vi = study.split_fit_val(P, tr)
    fi = np.sort(np.random.default_rng(0).choice(fi, min(5000, len(fi)), replace=False))
    te = L["te"]
    vf, rf, pf = xs_forward(L, fi)
    vt, rt, pt = xs_forward(L, te)
    repro = float(np.nanmax(np.abs(vt - L["saved_vol"])))
    out = {"kind": kind, "tag": tag, "fit_hours": len(fi), "test_hours": len(te), "repro_max_abs": repro,
           "test_first": str(np.datetime64(int(P["ts"][te[0]]), "us")), "test_last": str(np.datetime64(int(P["ts"][te[-1]]), "us"))}

    def cells(idx, pres, v):
        Xr = P["X"][idx].astype(float)
        return Xr[pres], v[pres]

    res = {}
    for hk, h in ((0, 1), (1, 4), (2, 24)):
        Xf, yf = cells(fi, pf, np.log(vf[..., hk, 3]))
        Xt, yt = cells(te, pt, np.log(vt[..., hk, 3]))
        okf = np.isfinite(Xf).all(1)
        r = {}
        for nm, cols in (("har_clock", (*XS_HAR[:1], {1: "lfac", 4: "lfac4", 24: "lfac24"}[h], *XS_HAR[2:])), ("hourly23", XS_SUR), ("all38", tuple(names))):
            j = [names.index(c) for c in cols]
            Af, At = np.nan_to_num(Xf[:, j]), np.nan_to_num(Xt[:, j])
            b = _ols(Af[okf], yf[okf])
            r[nm] = r2(yt, _pred(b, At))
        res[f"vol{h}h_logq50"] = r
    out["vol_surrogate_r2"] = res
    # rel 24h median in raw units vs the vol-drag surrogate -b * 24 * s2
    s2f, s2t = P["s2"][fi], P["s2"][te]
    medf = rf[..., 1, 1] * np.sqrt(24 * s2f)
    medt = rt[..., 1, 1] * np.sqrt(24 * s2t)
    mf, xf = medf[pf], 24 * s2f[pf]
    ok = np.isfinite(mf) & np.isfinite(xf)
    # hour fixed effects removed (the rel target is relative to the basket): demean per hour
    hours = np.repeat(np.arange(len(fi))[:, None], P["X"].shape[1], 1)[pf][ok]
    dm = lambda v: v - np.bincount(hours, v, minlength=len(fi))[hours] / np.maximum(np.bincount(hours, minlength=len(fi))[hours], 1)  # noqa: E731
    b_vd = float(np.sum(dm(xf[ok]) * dm(mf[ok])) / np.sum(dm(xf[ok]) ** 2))
    rho, rho_std = [], []
    for t in range(len(te)):
        k = pt[t] & np.isfinite(medt[t]) & np.isfinite(s2t[t])
        if k.sum() >= 5:
            rho.append(spearman(medt[t, k], b_vd * s2t[t, k]))
            rho_std.append(spearman(medt[t, k], rt[t, k, 1, 1]))
    real = P["rel_raw"][te][..., 1]
    org = (P["ts"][te] // 3_600_000_000) % 24 == 0
    ic_net, ic_vd, ic_std = [], [], []
    for t in np.flatnonzero(org):
        k = pt[t] & np.isfinite(real[t]) & np.isfinite(medt[t])
        if k.sum() >= 5:
            ic_net.append(spearman(medt[t, k], real[t, k]))
            ic_vd.append(spearman(b_vd * s2t[t, k], real[t, k]))
            ic_std.append(spearman(rt[t, k, 1, 1], real[t, k]))
    out["rel24"] = {"b_voldrag_per_24s2": b_vd, "mean_spearman_net_vs_voldrag": float(np.mean(rho)), "hours": len(rho),
                    "mean_spearman_net_raw_vs_std_median": float(np.mean(rho_std)),
                    "ic_net": nw(np.array(ic_net)), "ic_voldrag": nw(np.array(ic_vd)), "ic_std_median_only": nw(np.array(ic_std)),
                    "ic_net_minus_voldrag": nw(np.array(ic_net) - np.array(ic_vd)), "origins": len(ic_net),
                    "std_median_sd_cross_section": float(np.nanmean(np.nanstd(np.where(pt, rt[..., 1, 1], np.nan), axis=1)))}
    out["runtime_s"] = round(time.time() - t0, 1)
    print(json.dumps(_json(out))[:1500], flush=True)
    return out


def xs_invariance_v3() -> dict:
    """xs_v3 pipeline with BTC and ETH OHLC x c (2026-03-01 onward, cache frames), frozen-equivalent 2026Q4 weights."""
    from ai_trader.core.config import load
    from ai_trader.research import xs_network_study as study
    from ai_trader.research.frame import add_robust_vol, enrich_bars

    syms = study.universe()
    cache = load().root / "data" / "models" / "xs_v3" / "cache"
    raw = {s: pl.read_parquet(cache / f"f1h_{s}.parquet").filter(pl.col("event_ts") >= datetime(2026, 3, 1, tzinfo=UTC)) for s in syms}

    def panel(c):
        rows = {}
        for s, f in raw.items():
            k = c if s in ("BTCUSDT", "ETHUSDT") else 1.0
            b = f.select("event_ts", "close_ts", "known_ts", *[(pl.col(x) * k).alias(x) for x in ("open", "high", "low", "close")])
            g = add_robust_vol(enrich_bars(b)).join(f.select("event_ts", "funding_rate", "open_interest"), on="event_ts", how="left")
            rows[s] = study.token_rows(g)
        return study.assemble(rows, syms)

    L = xs_load("v3", "2026Q4")
    base = panel(1.0)
    out = {}
    for c in (1e-3, 1e3):
        Pc = panel(c)
        dx = np.nanmax(np.abs(Pc["X"].astype(float) - base["X"].astype(float)))
        same_nan = bool(np.array_equal(np.isnan(Pc["X"]), np.isnan(base["X"])))
        last = np.flatnonzero(base["valid"])[-720:]
        o = {}
        for nm, PP in (("base", base), ("scaled", Pc)):
            Lx = {**L, "P": PP}
            v, r_, pres = xs_forward(Lx, last)
            o[nm] = (v, r_, pres)
        dv = np.nanmax(np.abs(o["scaled"][0] - o["base"][0]) / np.maximum(o["base"][0], 1e-12))
        dr = np.nanmax(np.abs(o["scaled"][1] - o["base"][1])) / np.nanmean(np.abs(o["base"][1][..., 2] - o["base"][1][..., 0]))
        out[str(c)] = {"max_abs_feature": float(dx), "nan_pattern_equal": same_nan, "max_rel_vol": float(dv), "max_rel_rel_vs_band": float(dr),
                       "features_bitwise_equal": bool(np.array_equal(np.nan_to_num(Pc["X"], nan=-999.0), np.nan_to_num(base["X"], nan=-999.0)))}
    return out


def xs_invariance_v4(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT")) -> dict:
    """xs_v4 pipeline (REST-cache frames, liquidity mask on unscaled turnover) with BTC and ETH OHLC x c, frozen v4 weights.
    The raw cache is read only between minutes 8 and 55 of the hour (the live xs_v4 loop writes it at HH:02:30)."""
    import torch

    from ai_trader.core.config import load
    from ai_trader.ml import xs_net as xs
    from ai_trader.research import xs_network_study as study
    from ai_trader.research import xs_v4_study as v4
    from ai_trader.research.frame import add_robust_vol, enrich_bars
    from ai_trader.research.xs_v4_data import cfg as v4cfg

    while not (8 <= datetime.now(UTC).minute <= 55):
        time.sleep(20)
    fr = {s: v4.frame(s).filter(pl.col("event_ts") >= datetime(2026, 3, 1, tzinfo=UTC)) for s in symbols}
    floor = v4cfg()["universe"]["min_turnover_usd"]
    spec = json.loads((load().root / "data" / "models" / "xs_v4" / "frozen" / "spec.json").read_text(encoding="utf-8"))
    sd = spec["standardisation"]
    st = {"mu": np.asarray(sd["mu"]), "sd": np.asarray(sd["sd"]), "mmu": np.asarray(sd["mkt_mu"]), "msd": np.asarray(sd["mkt_sd"])}
    models = []
    for k in range(3):
        m = xs.make(38, 8, 14, kind="kan", mix=True)
        m.load_state_dict(torch.load(load().root / "data" / "models" / "xs_v4" / "frozen" / f"v4_seed{k}.pt", weights_only=False))
        m.eval()
        models.append(m)
    sec = np.zeros(len(symbols), dtype=np.int64)

    def panel(c):
        rows = {}
        for s, f in fr.items():
            k = c if s in ("BTCUSDT", "ETHUSDT") else 1.0
            b = f.select("event_ts", "close_ts", "known_ts", *[(pl.col(x) * k).alias(x) for x in ("open", "high", "low", "close")])
            g = add_robust_vol(enrich_bars(b)).join(f.select("event_ts", "funding_rate", "open_interest"), on="event_ts", how="left")
            rows[s] = v4.liquid_rows(f, study.token_rows(g), floor)
        return study.assemble(rows, list(symbols))

    base = panel(1.0)
    last = np.flatnonzero(base["valid"])[-720:]
    out = {"symbols": list(symbols), "hours": int(len(last))}

    def fwd(P):
        B = study.block(P, last, st)
        o = [xs.predict(m, B, sec) for m in models]
        return np.mean([x[0] for x in o], 0), np.mean([x[1] for x in o], 0)

    vb, rb = fwd(base)
    for c in (1e-3, 1e3):
        Pc = panel(c)
        vc, rc = fwd(Pc)
        out[str(c)] = {"max_abs_feature": float(np.nanmax(np.abs(Pc["X"].astype(float) - base["X"].astype(float)))),
                       "nan_pattern_equal": bool(np.array_equal(np.isnan(Pc["X"]), np.isnan(base["X"]))),
                       "present_equal": bool(np.array_equal(Pc["present"], base["present"])),
                       "max_rel_vol": float(np.nanmax(np.abs(vc - vb) / np.maximum(vb, 1e-12))),
                       "max_rel_rel_vs_band": float(np.nanmax(np.abs(rc - rb)) / np.nanmean(np.abs(rb[..., 2] - rb[..., 0])))}
    return out


def cmd_xs_inv4() -> dict:
    r = xs_invariance_v4()
    p = out_dir() / "xs.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["invariance_v4"] = r
    save("xs", d)
    print(json.dumps(_json(r)), flush=True)
    return r


# ------------------------------------------------------------------ WS12 deep models (H-WS14-12, 13)


def ws12_surrogate_X(pn: dict, rows: np.ndarray, which: str) -> np.ndarray:
    from ai_trader.research import ws12_deep as wd

    S = pn["S"][rows].astype(float)
    si = {c: i for i, c in enumerate(wd.STATIC_FEATURES)}
    if which == "1h":
        lfacn = pn["P"][rows - 1, wd.STEP_FEATURES.index("lfacn")].astype(float)
        lu = np.stack([pn["P"][rows - k, wd.STEP_FEATURES.index("lu")] for k in range(1, 7)], axis=1).astype(float)
        return np.c_[S[:, [si["lsig_rel1y"], si["har_d"], si["har_w"], si["har_m"]]], lfacn, lu]
    return S[:, [si["har_w"], si["lfac_n24"], si["lsig_rel1y"]]]


def cmd_ws12() -> dict:
    import torch

    from ai_trader.research import ws12_deep as wd

    base = out_dir().parent / "ws12"
    pn = wd.panel()
    out: dict = {}
    for arch in ("gru", "tcn"):
        for refit in ("R1", "R2"):
            lg = base / "logs" / f"train_{arch}_{refit}.json"
            if not lg.is_file():
                out[f"{arch}_{refit}"] = "not trained yet"
                continue
            lr = json.loads(lg.read_text(encoding="utf-8"))["chosen_lr"]
            rows = wd.split_rows(pn, refit)
            te = np.load(base / "preds" / f"rows_{refit}.npy")
            fi = np.sort(np.random.default_rng(0).choice(rows["fit"], min(20000, len(rows["fit"])), replace=False))
            pf, pt = [], []
            for s in wd.SEEDS:
                name = f"{arch}_{refit}_lr{lr:g}_s{s}"
                meta = json.loads((base / "models" / f"{name}.json").read_text(encoding="utf-8"))
                net = wd.make_net(arch, len(wd.STEP_FEATURES), len(wd.STATIC_FEATURES), pn["M"].shape[1])
                net.load_state_dict(torch.load(base / "models" / f"{name}.pt", weights_only=False))
                net.eval()
                bat = wd.Batcher.__new__(wd.Batcher)
                bat.pn, bat.arch = pn, arch
                for k, v in meta["standardiser"].items():
                    setattr(bat, k, np.asarray(v, dtype=np.float32))
                pf.append(wd.predict(net, bat, fi))
                pt.append(np.load(base / "preds" / f"{name}.npy"))
            pf, pt = np.mean(pf, 0), np.mean(pt, 0)
            day = pn["T"][te].astype("datetime64[D]")
            res = {"chosen_lr": lr, "fit_rows": len(fi), "test_rows": len(te), "test_first": str(pn["T"][te].min()), "test_last": str(pn["T"][te].max())}
            for which, hk in (("1h", 0), ("24h", 2)):
                Xf, Xt = ws12_surrogate_X(pn, fi, which), ws12_surrogate_X(pn, te, which)
                b = _ols(Xf, pf[:, hk, 3])
                f = _pred(b, Xt)
                binary = np.zeros(Xf.shape[1], bool)
                g = _spline_fit(Xf, pf[:, hk, 3], binary)(Xt)
                res[which] = {"r2_linear": r2(pt[:, hk, 3], f), "r2_linear_ci": r2_boot(pt[:, hk, 3], f, day), "coef": b.tolist(),
                              "r2_spline_descriptive": r2(pt[:, hk, 3], g),
                              "df_corr_with_logF": float(np.corrcoef(pt[:, hk, 2], pt[:, hk, 3])[0, 1])}
            out[f"{arch}_{refit}"] = res
            print(arch, refit, json.dumps(_json(res))[:600], flush=True)
    save("ws12_pickup", out)
    return out


def cmd_xs() -> dict:
    t0 = time.time()
    res = {"v3_2026Q4": xs_analyse("v3", "2026Q4"), "v3_2025Q2": xs_analyse("v3", "2025Q2")}
    save("xs", res)
    res["v4_2026Q4"] = xs_analyse("v4", "2026Q4")
    save("xs", res)
    res["invariance_v3"] = xs_invariance_v3()
    res["runtime_s"] = round(time.time() - t0, 1)
    save("xs", res)
    print(json.dumps(_json(res["invariance_v3"])), flush=True)
    return res


# ------------------------------------------------------------------ commands


def cmd_core() -> dict:
    t0 = time.time()
    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.pnn_study import FEATURES, arrays

    as_of = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(minutes=5)
    L = live_rows_1h(as_of)
    L["consistency"] = live_consistency(L)
    print("live rows", len(L["y"]), L["first"], L["last"], L["consistency"], flush=True)
    sets1 = {"har_clock": HAR_1H, "all19": tuple(HOURLY_FEATURES),
             "har_clock_recent": (*HAR_1H, "lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24", "lu_m6", "lu_m24")}
    d1 = distill("1h", sets1, spline=True, live_rows=L)
    save("distill_1h", d1)
    d24 = distill("1d", {"har_weekend": HAR_1D, "all14": tuple(FEATURES["1d"])}, spline=False)
    save("distill_1d", d24)
    R = Refit("1h", LIVE_MONTH)
    A = arrays("1h")
    rows = np.r_[R.val_rows, R.test_rows]
    Xm = np.r_[A["X"][rows], L["X"][L["ts"] > A["T"][rows].max()]]
    tm = np.r_[A["T"][rows], L["ts"][L["ts"] > A["T"][rows].max()]]
    ym = np.r_[A["y"][rows], L["y"][L["ts"] > A["T"][rows].max()]]
    mono = monotonicity(R, Xm, tm, ym)
    mono["rows_first"], mono["rows_last"] = str(tm.min()), str(tm.max())
    save("monotonicity_1h", mono)
    inv = invariance_pnn(R)
    save("invariance_pnn", inv)
    print(json.dumps({"runtime_s": round(time.time() - t0, 1)}), flush=True)
    return {"distill_1h": d1, "distill_1d": d24, "mono": mono, "inv": inv}


def main(argv: list[str] | None = None) -> None:
    import os

    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", 2)))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["core", "probe", "failure", "xs", "xs_inv4", "ws12", "report"])
    a = ap.parse_args(argv)
    if a.cmd == "core":
        cmd_core()
    else:
        globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
