"""WS07 REGIME: regime specifications compared on the same data and the same point-in-time constraints.

Specification program 2026-11-27 (research/programs/2026-11-27_specification_program.md, WS07). Hypotheses H-WS07-1..10
(research/hypotheses.yaml), design frozen in the diary before any fit ("WS07 design frozen before results").

Unit: one row per (symbol, UTC day d), universe L (23 Bybit crypto perps). Outcomes of day d: r_d (close-to-close log
return 00:00 -> 24:00 UTC) and RV_d (sum of 24 squared 1h log returns). The STATE used for day d is known at 00:00 UTC
of d:
  * v0: research.regimes.label on the 1h frame, the row closing at 00:00 UTC (exactly the live gate's start-of-day cell);
  * fitted models on daily data: FILTERED P(S_{d-1} | F_{d-1}) (argmax = label, used for gates and agreement) and the
    one-step PREDICTED P(S_d | F_{d-1}) (used for forecasts). Smoothed probabilities appear only in the descriptive
    diagnostics (smoothed-vs-filtered ARI), never in a test.
Walk-forward: expanding refits at ORIGINS, training days <= origin - 2 (1-day embargo), >= MIN_TRAIN days per symbol,
the HMM family uses 2 random block starts at a symbol's first fit and the previous fold's parameters as the only start
afterwards (tvtp3 starts from the same fold's msvar3 fit with zero driver slopes, so its fit nests constant transitions),
the causal filter runs with frozen parameters through the fold's out-of-sample days [origin, next origin).

Specs (K fixed in advance): v0_cell / v0_vol / v0_dir; ghmm2, ghmm3 (Gaussian HMM on r); thmm3 (t, nu = 5); msar2
(MS-AR(1)); msvar3 (MS-VAR(1) on [r, log RV]); tvtp3 (msvar3 with funding / OI-driven transitions); bocpd (Adams-MacKay on
log RV); pelt (online PELT on log RV); kmeans9 / gmm3 (start-of-day state features). The HMM family uses the reviewed
TVTP module (src/ai_trader/research/tvtp_msvar.py: K regimes, VAR(p), constant transitions when Z is None).

    uv run python -m ai_trader.research.regime_specs panel
    uv run python -m ai_trader.research.regime_specs fit --specs ghmm2,ghmm3
    uv run python -m ai_trader.research.regime_specs eval
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "ws07"
AS_OF = datetime(2026, 11, 26, tzinfo=UTC)
CUTOFF = date(2026, 7, 1)
ORIGINS = [date(y, m, 1) for y in range(2022, 2027) for m in (1, 7)]          # 2022-01-01 .. 2026-07-01
MIN_TRAIN = 365
EMBARGO = 1
HMM_SPECS = {   # name: (K, p, nu, columns)
    "ghmm2": (2, 0, None, ("r",)), "ghmm3": (3, 0, None, ("r",)), "thmm3": (3, 0, 5.0, ("r",)),
    "msar2": (2, 1, None, ("r",)), "msvar3": (3, 1, None, ("r", "lrv")), "tvtp3": (3, 1, None, ("r", "lrv")),
}
OTHER_SPECS = ("bocpd", "pelt", "kmeans9", "gmm3")
V0_SPECS = ("v0_cell", "v0_vol", "v0_dir")
ALL_SPECS = V0_SPECS + tuple(HMM_SPECS) + OTHER_SPECS
CELLS = tuple(f"R{i}" for i in range(1, 10))
FEATS = ("vpct", "dir_score", "fz", "oiz")


def seg_of(d: date) -> str:
    return "clean_holdout" if d >= CUTOFF else "llm_in_sample"


def fold_of(d: date) -> int:
    f = -1
    for i, o in enumerate(ORIGINS):
        if d >= o:
            f = i
    return f


# ====================================================================== panel


def trailing_pct(x: np.ndarray, w: int) -> np.ndarray:
    """Share of the trailing w values (current included) <= the current one; NaN until w // 2 values exist."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        a = x[max(0, i - w + 1): i + 1]
        a = a[np.isfinite(a)]
        if len(a) >= w // 2 and np.isfinite(x[i]):
            out[i] = (a <= x[i]).mean()
    return out


def trailing_z(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        a = x[max(0, i - w + 1): i + 1]
        a = a[np.isfinite(a)]
        if len(a) >= w // 2 and np.isfinite(x[i]):
            sd = a.std()
            out[i] = (x[i] - a.mean()) / sd if sd > 0 else 0.0
    return out


def symbol_panel(sym: str, as_of: datetime = AS_OF) -> pl.DataFrame:
    from ai_trader.features.realized import daily_measures
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label

    f1 = market_frame(sym, "1h", as_of, funding=True, oi=True, oi_interval="4h")
    lab = label(f1, bar_hours=1.0)
    sod = lab.filter((pl.col("close_ts").dt.hour() == 0) & (pl.col("close_ts").dt.minute() == 0)).select(
        pl.col("close_ts").dt.date().alias("date"), pl.col("cell"), pl.col("dir_score"), pl.col("dir_state"),
        pl.col("vol_state"), pl.col("sigma_next").alias("sigma_sod"), pl.col("funding_rate").alias("funding"),
        pl.col("open_interest").alias("oi"))
    dm = daily_measures(f1.select("event_ts", "open", "high", "low", "close"), "1h").select("date", "ret", "rv", "n_bars")
    d = sod.join(dm, on="date", how="left").filter(pl.col("date") < as_of.date()).sort("date")
    lsig = np.log(d["sigma_sod"].to_numpy())
    oi = d["oi"].to_numpy().astype(float)
    loi = np.log(np.where(oi > 0, oi, np.nan))
    oi7 = np.r_[np.full(7, np.nan), loi[7:] - loi[:-7]]
    return d.with_columns(
        pl.lit(sym).alias("symbol"),
        pl.Series("vpct", trailing_pct(lsig, 180)),
        pl.Series("fz", trailing_z(d["funding"].to_numpy().astype(float), 90)),
        pl.Series("oiz", trailing_z(oi7, 180)),
        (24.0 * pl.col("sigma_sod") ** 2).alias("ewma_var"),
        ((pl.col("n_bars") == 24) & (pl.col("rv") > 0)).fill_null(False).alias("ok"),
    )


def build_panel() -> Path:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in UNIVERSE:
        t = time.time()
        parts.append(symbol_panel(s))
        print(s, parts[-1].height, round(time.time() - t, 1), flush=True)
    p = pl.concat(parts, how="vertical_relaxed").with_columns(
        pl.col("date").map_elements(lambda x: seg_of(x), return_dtype=pl.Utf8).alias("seg"))
    path = OUT / "daily_panel.parquet"
    p.write_parquet(path)
    return path


def load_panel() -> pl.DataFrame:
    return pl.read_parquet(OUT / "daily_panel.parquet")


# ====================================================================== change-point specs


def bocpd(x: np.ndarray, mu0: float, kappa0: float, alpha0: float, beta0: float, hazard: float = 1 / 30,
          rmax: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adams-MacKay BOCPD, Gaussian with unknown mean and variance (Normal-inverse-gamma). For each t returns the
    predictive mean and variance of x_t given x_{<t} and the MAP run length before x_t. Causal by construction."""
    T = len(x)
    pm, pv, rl = np.empty(T), np.empty(T), np.empty(T)
    mu, ka, al, be = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    w = np.array([1.0])
    for t in range(T):
        df = 2 * al
        sc2 = be * (ka + 1) / (al * ka)
        pm[t] = w @ mu
        pv[t] = w @ (sc2 * df / (df - 2) + mu ** 2) - pm[t] ** 2
        rl[t] = float(np.argmax(w))
        z2 = (x[t] - mu) ** 2 / sc2
        logp = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * math.pi * sc2) - (df + 1) / 2 * np.log1p(z2 / df)
        pr = w * np.exp(logp - logp.max())
        s = pr.sum()
        w = np.r_[s * hazard, pr * (1 - hazard)]
        w /= w.sum()
        mu_n = (ka * mu + x[t]) / (ka + 1)
        be_n = be + ka * (x[t] - mu) ** 2 / (2 * (ka + 1))
        mu, ka, al, be = np.r_[mu0, mu_n], np.r_[kappa0, ka + 1], np.r_[alpha0, al + 0.5], np.r_[beta0, be_n]
        if len(w) > rmax:
            w, mu, ka, al, be = w[:rmax], mu[:rmax], ka[:rmax], al[:rmax], be[:rmax]
            w /= w.sum()
    return pm, pv, rl


def pelt_last_start(x: np.ndarray, beta: float, sigma2: float) -> np.ndarray:
    """Online PELT (Killick et al. 2012), Gaussian mean-change cost with known variance sigma2. last[t] = start index of
    the last segment of the optimal partition of x[0:t] (so the segment is x[last[t]:t]); last[0] = 0."""
    T = len(x)
    cs, cs2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
    F = np.empty(T + 1)
    F[0] = -beta
    last = np.zeros(T + 1, dtype=np.int64)
    R = np.array([0])
    for t in range(1, T + 1):
        n = t - R
        s1 = cs[t] - cs[R]
        cost = (cs2[t] - cs2[R] - s1 * s1 / n) / sigma2
        v = F[R] + cost
        i = int(np.argmin(v))
        F[t] = v[i] + beta
        last[t] = R[i]
        R = np.r_[R[v <= F[t]], t]
    return last


def tercile_vs_trailing(level: np.ndarray, x: np.ndarray, w: int = 180) -> np.ndarray:
    """Label 0/1/2 of level[i] against the 1/3, 2/3 quantiles of x[i-w:i] (strictly past values); -1 when unknown."""
    out = np.full(len(x), -1, dtype=np.int64)
    for i in range(len(x)):
        a = x[max(0, i - w): i]
        if len(a) < w // 2 or not np.isfinite(level[i]):
            continue
        q1, q2 = np.quantile(a, [1 / 3, 2 / 3])
        out[i] = 0 if level[i] < q1 else (2 if level[i] > q2 else 1)
    return out


# ====================================================================== fitting


def _sym_arrays(g: pl.DataFrame) -> dict:
    v = g.filter(pl.col("ok")).sort("date")
    return {"date": v["date"].to_list(), "r": v["ret"].to_numpy() * 100.0, "lrv": np.log(v["rv"].to_numpy() * 1e4),
            "fz": np.nan_to_num(v["fz"].to_numpy().astype(float)), "oiz": np.nan_to_num(v["oiz"].to_numpy().astype(float)),
            "feats": v.select(FEATS).to_numpy().astype(float)}


def _rows(spec: str, sym: str, dates: list, fold: int, oos: np.ndarray, label: np.ndarray, probs: np.ndarray | None,
          var_native: np.ndarray | None, ls_native: np.ndarray | None) -> pl.DataFrame:
    n = len(dates)
    cols = {"spec": [spec] * n, "symbol": [sym] * n, "date": dates, "fold": np.full(n, fold, np.int8), "oos": oos,
            "label": label.astype(np.int8)}
    if probs is not None:
        for k in range(probs.shape[1]):
            cols[f"p{k}"] = probs[:, k].astype(np.float32)
    cols["var_native"] = (np.full(n, np.nan) if var_native is None else var_native).astype(np.float64)
    cols["ls_native"] = (np.full(n, np.nan) if ls_native is None else ls_native).astype(np.float64)
    return pl.DataFrame(cols)


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    m = (a >= 0) & (b >= 0)
    return float(adjusted_rand_score(a[m], b[m])) if m.sum() > 10 else float("nan")


def fit_hmm_symbol(spec: str, A: dict, sym: str, start_from: dict | None = None) -> tuple[list[pl.DataFrame], list[dict], dict]:
    """Walk-forward for one HMM-family spec on one symbol. Returns row frames, fit diagnostics and the last params per fold
    (used as the warm start of tvtp3 from msvar3)."""
    from ai_trader.research.tvtp_msvar import TVTPMSVAR

    K, p, nu, cols = HMM_SPECS[spec]
    Y = np.column_stack([A[c] for c in cols])
    Z = np.column_stack([A["fz"], A["oiz"]]) if spec == "tvtp3" else None
    dates = A["date"]
    proto = TVTPMSVAR(K=K, p=p, nu=nu)
    Yt, X, Zc = proto.design(Y, Z)
    dd = dates[p:]
    frames, diags, params_by_fold = [], [], {}
    prev = None
    for f, o in enumerate(ORIGINS):
        o2 = ORIGINS[f + 1] if f + 1 < len(ORIGINS) else date(2100, 1, 1)
        ntr = sum(1 for x in dd if x <= o - timedelta(days=EMBARGO + 1))
        nend = sum(1 for x in dd if x < o2)
        if ntr < MIN_TRAIN or nend <= ntr:
            continue
        m = TVTPMSVAR(K=K, p=p, nu=nu, n_init=0 if (spec == "tvtp3" or prev is not None) else 2, max_iter=150, tol=1e-6, seed=f)
        if spec == "tvtp3":
            st = start_from[f]
            W = np.zeros((K, K - 1, Zc.shape[1]))
            W[:, :, :st["W"].shape[2]] = st["W"]
            start = dict(B=st["B"], S=st["S"], W=W, logxi0=st["logxi0"])
        else:
            start = prev.params() if prev is not None else None
        t0 = time.time()
        m.fit_design(Yt[:ntr], X[:ntr], Zc[:ntr], start=start)
        perm = np.argsort(m.S[:, 0, 0])
        if not np.array_equal(perm, np.arange(K)):
            m.permute(perm)
        r = m.filter_design(Yt[:nend], X[:nend], Zc[:nend])
        pred, filt = r["pred"], r["filt"]
        lab = np.r_[-1, np.argmax(filt[:-1], axis=1)]                     # label for day t = filtered state at t-1
        cs = m._cov_scale()
        if Y.shape[1] == 1:
            var_n = (pred * (r["mu"][:, :, 0] ** 2 + cs * m.S[None, :, 0, 0])).sum(1) / 1e4
            ls_n = r["pll"] + math.log(100.0)
        else:
            var_n = (pred * np.exp(r["mu"][:, :, 1] + 0.5 * m.S[None, :, 1, 1])).sum(1) / 1e4
            s0 = m.S[:, 0, 0]
            lg = -0.5 * (Yt[:nend, 0:1] - r["mu"][:, :, 0]) ** 2 / s0 - 0.5 * np.log(2 * math.pi * s0)
            ls_n = logsumexp(lg + np.log(np.maximum(pred, 1e-300)), axis=1) + math.log(100.0)
        oos = np.array([x >= o for x in dd[:nend]])
        keep = oos | (np.arange(nend) < ntr)
        frames.append(_rows(spec, sym, [x for x, k in zip(dd[:nend], keep) if k], f, oos[keep], lab[keep], pred[keep],
                            var_n[keep], ls_n[keep]))
        P = m.transition_matrix() if Z is None else m.transition_matrix(np.zeros(Z.shape[1]))
        dg = {"spec": spec, "symbol": sym, "fold": f, "n_train": ntr, "ll": m.ll, "bic": m.bic, "iters": m.iters,
              "converged": bool(m.converged), "min_occupancy": m.min_occupancy, "secs": round(time.time() - t0, 2),
              "exp_dur": (1.0 / np.maximum(1 - np.diag(P), 1e-9)).round(2).tolist(),
              "sd_r": np.sqrt(m.S[:, 0, 0] * cs / 1e4).round(5).tolist()}
        dg["ari_smooth_vs_filt"] = _ari(np.argmax(m.prob_smoothed, axis=1), np.argmax(m.prob_filtered, axis=1))  # descriptive
        if prev is not None:
            ntr_prev = prev._Yt.shape[0]
            a = np.argmax(prev.filter_design(Yt[:ntr_prev], X[:ntr_prev], Zc[:ntr_prev])["filt"], axis=1)
            b = np.argmax(m.filter_design(Yt[:ntr_prev], X[:ntr_prev], Zc[:ntr_prev])["filt"], axis=1)
            dg["ari_refit"] = _ari(a, b)
        diags.append(dg)
        params_by_fold[f] = {k: v.copy() for k, v in m.params().items()}
        prev = m
    return frames, diags, params_by_fold


def fit_cp_symbol(spec: str, A: dict, sym: str) -> tuple[list[pl.DataFrame], list[dict]]:
    x, dates = A["lrv"], A["date"]
    frames, diags = [], []
    prev_lab = None
    for f, o in enumerate(ORIGINS):
        o2 = ORIGINS[f + 1] if f + 1 < len(ORIGINS) else date(2100, 1, 1)
        ntr = sum(1 for d in dates if d <= o - timedelta(days=EMBARGO + 1))
        nend = sum(1 for d in dates if d < o2)
        if ntr < MIN_TRAIN or nend <= ntr:
            continue
        xt = x[:ntr]
        s2 = float(np.var(np.diff(xt)) / 2)
        if spec == "bocpd":
            pm, pv, rl = bocpd(x[:nend], mu0=float(xt.mean()), kappa0=1.0, alpha0=2.0, beta0=s2)
            level, var_n = pm, np.exp(pm + 0.5 * pv) / 1e4
            extra = {"median_run_length": float(np.median(rl[ntr:nend]))}
        else:
            last = pelt_last_start(x[:nend], beta=2 * math.log(ntr), sigma2=s2)
            cs = np.r_[0.0, np.cumsum(x[:nend])]
            idx = np.arange(nend)
            st = last[idx]
            level = np.where(idx > st, (cs[idx] - cs[st]) / np.maximum(idx - st, 1), np.nan)
            var_n = np.exp(level + 0.5 * s2) / 1e4
            extra = {"segments_train": int(len(np.unique(last[1:ntr + 1])))}
        lab = tercile_vs_trailing(level, x[:nend])
        oos = np.array([d >= o for d in dates[:nend]])
        keep = oos | (np.arange(nend) < ntr)
        probs = np.eye(3)[np.maximum(lab, 0)] * (lab >= 0)[:, None]
        frames.append(_rows(spec, sym, [d for d, k in zip(dates[:nend], keep) if k], f, oos[keep], lab[keep], probs[keep],
                            var_n[keep], None))
        dg = {"spec": spec, "symbol": sym, "fold": f, "n_train": ntr, "sigma2": s2, **extra}
        if prev_lab is not None:
            n0 = len(prev_lab)
            dg["ari_refit"] = _ari(prev_lab, lab[:n0])
        prev_lab = lab[:ntr]
        diags.append(dg)
    return frames, diags


def fit_cluster_symbol(spec: str, A: dict, sym: str) -> tuple[list[pl.DataFrame], list[dict]]:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    Fm, dates = A["feats"], A["date"]
    ok = np.isfinite(Fm).all(1)
    frames, diags = [], []
    prev = None
    for f, o in enumerate(ORIGINS):
        o2 = ORIGINS[f + 1] if f + 1 < len(ORIGINS) else date(2100, 1, 1)
        ntr = sum(1 for d in dates if d <= o - timedelta(days=EMBARGO + 1))
        nend = sum(1 for d in dates if d < o2)
        tr = ok[:ntr]
        if tr.sum() < MIN_TRAIN or nend <= ntr:
            continue
        mu, sd = Fm[:ntr][tr].mean(0), Fm[:ntr][tr].std(0) + 1e-12
        Zs = (np.nan_to_num(Fm[:nend]) - mu) / sd
        if spec == "kmeans9":
            m = KMeans(n_clusters=9, n_init=10, random_state=f).fit(Zs[:ntr][tr])
            order = np.lexsort((m.cluster_centers_[:, 1], m.cluster_centers_[:, 0]))
            raw = m.predict(Zs)
            K = 9
            probs = None
        else:
            m = GaussianMixture(3, covariance_type="full", n_init=2, random_state=f).fit(Zs[:ntr][tr])
            order = np.argsort(m.means_[:, 0])
            raw = m.predict(Zs)
            K = 3
            probs = m.predict_proba(Zs)[:, order]
        rank = np.empty(K, int)
        rank[order] = np.arange(K)
        lab = np.where(ok[:nend], rank[raw], -1)
        if probs is None:
            probs = np.eye(K)[np.maximum(lab, 0)] * (lab >= 0)[:, None]
        else:
            probs = probs * (lab >= 0)[:, None]
        oos = np.array([d >= o for d in dates[:nend]])
        keep = oos | (np.arange(nend) < ntr)
        frames.append(_rows(spec, sym, [d for d, k in zip(dates[:nend], keep) if k], f, oos[keep], lab[keep], probs[keep],
                            None, None))
        dg = {"spec": spec, "symbol": sym, "fold": f, "n_train": int(tr.sum())}
        if prev is not None:
            n0, mprev, (pmu, psd), prank = prev
            old = np.where(ok[:n0], prank[mprev.predict((np.nan_to_num(Fm[:n0]) - pmu) / psd)], -1)
            dg["ari_refit"] = _ari(old, lab[:n0])
        prev = (ntr, m, (mu, sd), rank)
        diags.append(dg)
    return frames, diags


def run_fit(specs: list[str]) -> None:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    panel = load_panel()
    (OUT / "fits").mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if spec == "tvtp3":
            continue                                                       # fitted together with msvar3
        t0 = time.time()
        frames, diags, tv_frames, tv_diags = [], [], [], []
        for sym in UNIVERSE:
            A = _sym_arrays(panel.filter(pl.col("symbol") == sym))
            if spec in HMM_SPECS:
                fr, dg, pars = fit_hmm_symbol(spec, A, sym)
                if spec == "msvar3" and "tvtp3" in specs:
                    a, b, _ = fit_hmm_symbol("tvtp3", A, sym, start_from=pars)
                    tv_frames += a
                    tv_diags += b
            elif spec in ("bocpd", "pelt"):
                fr, dg = fit_cp_symbol(spec, A, sym)
            else:
                fr, dg = fit_cluster_symbol(spec, A, sym)
            frames += fr
            diags += dg
            print(spec, sym, len(dg), round(time.time() - t0, 1), flush=True)
        _write_spec(spec, frames, diags)
        if tv_frames:
            _write_spec("tvtp3", tv_frames, tv_diags)


def _write_spec(spec: str, frames: list[pl.DataFrame], diags: list[dict]) -> None:
    pl.concat(frames, how="diagonal_relaxed").write_parquet(OUT / "fits" / f"{spec}.parquet")
    (OUT / "fits" / f"{spec}_diag.json").write_text(json.dumps(diags, default=float, indent=0))


def v0_rows(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """v0 labels (fold-independent): integer labels in a fixed order, one-hot probabilities."""
    out = {}
    for spec, col, levels in (("v0_cell", "cell", CELLS), ("v0_vol", "vol_state", ("low", "normal", "high")),
                              ("v0_dir", "dir_state", ("down", "range", "up"))):
        m = {v: i for i, v in enumerate(levels)}
        g = panel.filter(pl.col("ok")).select(
            "symbol", "date", pl.col(col).replace_strict(m, default=-1, return_dtype=pl.Int8).fill_null(-1).alias("label"))
        lab = g["label"].to_numpy().astype(int)
        P = np.eye(len(levels))[np.maximum(lab, 0)] * (lab >= 0)[:, None]
        out[spec] = g.with_columns([pl.Series(f"p{k}", P[:, k].astype(np.float32)) for k in range(len(levels))]
                                   + [pl.lit(None, dtype=pl.Float64).alias("var_native"), pl.lit(None, dtype=pl.Float64).alias("ls_native")])
    return out


# ====================================================================== statistics


def nw_t(x: np.ndarray, lag: int | None = None) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan"), n
    e = x - x.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9))) if lag is None else lag
    v = e @ e / n
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(v, 1e-300) / n)
    return float(x.mean()), float(x.mean() / se), n


def dm_test(d: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano on a loss-difference series (NW HAC), Harvey-Leybourne-Newbold small-sample correction, t(n-1)."""
    m, t, n = nw_t(d)
    if not np.isfinite(t):
        return {"mean": m, "t": t, "p": float("nan"), "n": n}
    k = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    th = t * k
    return {"mean": m, "t": th, "p": float(2 * stats.t.sf(abs(th), n - 1)), "n": n,
            "mde": 2.8 * abs(m / t) if t else float("nan")}


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted((v, k) for k, v in p.items() if np.isfinite(v))
    m, out, run = len(items), {}, 0.0
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def block_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(math.ceil(n / block))
    st = rng.integers(0, max(n - block + 1, 1), nb)
    return (st[:, None] + np.arange(block)[None, :]).ravel()[:n]


def mcs(L: np.ndarray, names: list[str], alpha: float = 0.10, B: int = 2000, block: int = 10, seed: int = 7) -> dict:
    """Hansen-Lunde-Nason MCS with the T_max statistic. L: (T, M) losses (lower is better). Returns the surviving set
    and MCS p-values."""
    rng = np.random.default_rng(seed)
    T, M = L.shape
    idx = [block_idx(T, block, rng) for _ in range(B)]
    alive = list(range(M))
    pvals = {}
    pmax = 0.0
    while len(alive) > 1:
        Lm = L[:, alive]
        dbar = Lm.mean(0) - Lm.mean()                                     # d_i. = mean_i - mean over models
        boot = np.array([Lm[ix].mean(0) - Lm[ix].mean() for ix in idx]) - dbar
        se = boot.std(0) + 1e-300
        tstat = dbar / se
        tmax = tstat.max()
        bt = (boot / se).max(1)
        p = float((bt >= tmax).mean())
        pmax = max(pmax, p)
        worst = alive[int(np.argmax(tstat))]
        pvals[names[worst]] = pmax
        if p >= alpha:
            break
        alive.remove(worst)
    for a in alive:
        pvals.setdefault(names[a], 1.0)
    surv = [names[a] for a in alive] if len(alive) > 1 or pmax < alpha else [names[a] for a in alive]
    return {"set": surv, "p": pvals}


def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    m = (a >= 0) & (b >= 0)
    a, b = a[m], b[m]
    if len(a) < 10:
        return float("nan")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    C = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(C, (ia, ib), 1)
    P = C / C.sum()
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    Ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    Hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    I = (P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum()
    return float(Ha + Hb - 2 * I)


def t_nu_mle(z: np.ndarray) -> float:
    z = z[np.isfinite(z)]
    grid = np.r_[np.arange(3.0, 10.0, 0.25), np.arange(10.0, 51.0, 1.0)]

    def ll(nu):
        c = math.sqrt((nu - 2) / nu)
        return (stats.t.logpdf(z / c, nu) - math.log(c)).sum()
    return float(grid[int(np.argmax([ll(v) for v in grid]))])


def t_logpdf_var(x: np.ndarray, mu: np.ndarray, var: np.ndarray, nu: float) -> np.ndarray:
    """Student-t log density with mean mu and VARIANCE var (scale = sqrt(var (nu-2)/nu))."""
    sc = np.sqrt(var * (nu - 2) / nu)
    return stats.t.logpdf((x - mu) / sc, nu) - np.log(sc)


def t_cdf_var(x: np.ndarray, mu, var: np.ndarray, nu: float) -> np.ndarray:
    return stats.t.cdf((x - mu) / np.sqrt(var * (nu - 2) / nu), nu)


def qlike(rv: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = rv / h
    return q - np.log(q) - 1.0


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 10 and x.std() > 0 else float("nan")


def deflated_sharpe(sr_daily: float, n: int, trials_sr: np.ndarray, skew: float, kurt: float) -> float:
    """Bailey & Lopez de Prado DSR: P(true SR > SR0), SR0 = expected max of N trial SRs under the null (daily units)."""
    N = max(len(trials_sr), 2)
    v = np.var(trials_sr, ddof=1) if len(trials_sr) > 1 else 0.0
    em = 0.5772156649
    sr0 = math.sqrt(max(v, 1e-12)) * ((1 - em) * stats.norm.ppf(1 - 1 / N) + em * stats.norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2, 1e-12))
    return float(stats.norm.cdf((sr_daily - sr0) * math.sqrt(n - 1) / den))


# ====================================================================== evaluation

LAMBDA = 20.0
KAPPA = 10.0
RULES = {   # R-MAP-3..7 as written in research/rules.yaml (strategies named in the rule, v0 WHEN cells)
    "R-MAP-3": (("zscore_reversion", "rsi_reversion"), ("R1", "R2", "R4", "R7")),
    "R-MAP-4": (("tsmom", "ema_stack"), ("R5", "R6")),
    "R-MAP-5": (("funding_contrarian", "oi_price_divergence"), ("R5", "R7")),
    "R-MAP-6": (("geo_struct", "cme_gap_fill", "weekend_drift_fade"), ("R4",)),
    "R-MAP-7": (("shock_regime",), ("R2",)),
}


def spec_tables(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out = v0_rows(panel)
    for sp in tuple(HMM_SPECS) + OTHER_SPECS + ("msgarch2",):
        f = OUT / "fits" / f"{sp}.parquet"
        if f.exists():
            out[sp] = pl.read_parquet(f)
    return out


def _pcols(t: pl.DataFrame) -> list[str]:
    return [c for c in t.columns if c.startswith("p") and c[1:].isdigit()]


def _episodes(labels: np.ndarray, syms: np.ndarray, K: int) -> np.ndarray:
    e = np.zeros(K)
    prev_l, prev_s = None, None
    for lab, s in zip(labels, syms, strict=True):
        if lab >= 0 and (lab != prev_l or s != prev_s):
            e[lab] += 1
        prev_l, prev_s = lab, s
    return e


def _ridge(X: np.ndarray, y: np.ndarray, nfree: int) -> np.ndarray:
    lam = np.r_[np.zeros(nfree), np.full(X.shape[1] - nfree, LAMBDA)]
    return np.linalg.solve(X.T @ X + np.diag(lam), X.T @ y)


def _lookup(tt: pl.DataFrame, pc: list[str], dord: np.ndarray, mask: np.ndarray):
    m = dict(zip([d.toordinal() for d in tt["date"].to_list()], range(tt.height), strict=True))
    ii = np.array([m.get(dord[i], -1) for i in np.where(mask)[0]], dtype=np.int64)
    P = tt.select(pc).to_numpy().astype(float)
    lab = tt["label"].to_numpy().astype(int)
    vn, lsn = tt["var_native"].to_numpy(), tt["ls_native"].to_numpy()
    ok = ii >= 0
    out_P = np.full((len(ii), len(pc)), np.nan)
    out_P[ok] = P[ii[ok]]
    out_l = np.full(len(ii), -1)
    out_l[ok] = lab[ii[ok]]
    out_v = np.full(len(ii), np.nan)
    out_v[ok] = vn[ii[ok]]
    out_s = np.full(len(ii), np.nan)
    out_s[ok] = lsn[ii[ok]]
    out_P[~ok | (out_l < 0)] = np.nan
    return out_P, out_l, out_v, out_s


def forecasts(panel: pl.DataFrame, tabs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Out-of-sample next-day variance forecasts h and log scores ls of r_d for every model, one row per
    (symbol, date, model). Models: ewma, har, har|spec (log-HAR + ridge state shifts), harmu|spec (+ shrunk state means of
    the standardised return; ls only), nat|spec (the regime model's own forecasts)."""
    recs = []
    for sym in panel["symbol"].unique().sort().to_list():
        g = panel.filter((pl.col("symbol") == sym) & pl.col("ok")).sort("date")
        dts = g["date"].to_list()
        n = len(dts)
        if n < MIN_TRAIN + 30:
            continue
        rv, r, ew = g["rv"].to_numpy(), g["ret"].to_numpy(), g["ewma_var"].to_numpy()
        lrv = np.log(rv)
        x1 = np.r_[np.nan, lrv[:-1]]
        c = np.r_[0.0, np.cumsum(rv)]
        idx = np.arange(n)
        x5 = np.where(idx >= 5, np.log(np.maximum(c[idx] - c[np.maximum(idx - 5, 0)], 1e-300) / 5), np.nan)
        x22 = np.where(idx >= 22, np.log(np.maximum(c[idx] - c[np.maximum(idx - 22, 0)], 1e-300) / 22), np.nan)
        H = np.column_stack([np.ones(n), x1, x5, x22])
        okh = np.isfinite(H).all(1) & np.isfinite(ew) & (ew > 0)
        dord = np.array([d.toordinal() for d in dts])
        stab = {sp: t.filter(pl.col("symbol") == sym) for sp, t in tabs.items()}
        for f, o in enumerate(ORIGINS):
            o2 = ORIGINS[f + 1] if f + 1 < len(ORIGINS) else date(2100, 1, 1)
            tr = okh & (dord <= (o - timedelta(days=EMBARGO + 1)).toordinal())
            te = okh & (dord >= o.toordinal()) & (dord < o2.toordinal())
            if tr.sum() < MIN_TRAIN or te.sum() == 0:
                continue
            b = np.linalg.lstsq(H[tr], lrv[tr], rcond=None)[0]
            s2 = float(np.var(lrv[tr] - H[tr] @ b))
            h_tr, h_te = np.exp(H[tr] @ b + s2 / 2), np.exp(H[te] @ b + s2 / 2)
            nu = t_nu_mle(r[tr] / np.sqrt(h_tr))
            nu_e = t_nu_mle(r[tr] / np.sqrt(ew[tr]))
            te_d = [dts[i] for i in np.where(te)[0]]
            nte = len(te_d)
            kw = {"symbol": [sym] * nte, "fold": [f] * nte, "date": te_d}
            recs.append(pl.DataFrame({**kw, "model": "har", "h": h_te, "ls": t_logpdf_var(r[te], 0.0, h_te, nu),
                                      "pit": t_cdf_var(r[te], 0.0, h_te, nu)}))
            recs.append(pl.DataFrame({**kw, "model": "ewma", "h": ew[te], "ls": t_logpdf_var(r[te], 0.0, ew[te], nu_e),
                                      "pit": t_cdf_var(r[te], 0.0, ew[te], nu_e)}))
            for sp, t in stab.items():
                if t.height == 0:
                    continue
                pc = _pcols(t)
                if sp in V0_SPECS:
                    ttr = tte = t
                else:
                    tf = t.filter(pl.col("fold") == f)
                    ttr, tte = tf.filter(~pl.col("oos")), tf.filter(pl.col("oos"))
                if ttr.height == 0 or tte.height == 0:
                    continue
                Ptr, ltr, _, _ = _lookup(ttr, pc, dord, tr)
                Pte, lte, vte, ste = _lookup(tte, pc, dord, te)
                g_tr = np.isfinite(Ptr).all(1)
                if g_tr.sum() < MIN_TRAIN // 2:
                    continue
                Xtr = np.column_stack([H[tr][g_tr], Ptr[g_tr][:, 1:]])
                bb = _ridge(Xtr, lrv[tr][g_tr], 4)
                s2c = float(np.var(lrv[tr][g_tr] - Xtr @ bb))
                hc_tr = np.exp(Xtr @ bb + s2c / 2)
                nuc = t_nu_mle(r[tr][g_tr] / np.sqrt(hc_tr))
                g_te = np.isfinite(Pte).all(1)
                hc = np.full(nte, np.nan)
                hc[g_te] = np.exp(np.column_stack([H[te][g_te], Pte[g_te][:, 1:]]) @ bb + s2c / 2)
                lsc = t_logpdf_var(r[te], 0.0, hc, nuc)
                K = len(pc)
                ztr = r[tr][g_tr] / np.sqrt(hc_tr)
                lt = ltr[g_tr]
                e = _episodes(lt, np.zeros(len(lt)), K)
                mu_k = np.array([ztr[lt == k].mean() if (lt == k).any() else 0.0 for k in range(K)]) * e / (e + KAPPA)
                loc = np.where(lte >= 0, mu_k[np.maximum(lte, 0)], np.nan) * np.sqrt(hc)
                lsm = t_logpdf_var(r[te], loc, hc, nuc)
                recs.append(pl.DataFrame({**kw, "model": f"har|{sp}", "h": hc, "ls": lsc, "pit": t_cdf_var(r[te], 0.0, hc, nuc)}))
                recs.append(pl.DataFrame({**kw, "model": f"harmu|{sp}", "h": np.full(nte, np.nan), "ls": lsm,
                                          "pit": t_cdf_var(r[te], loc, hc, nuc)}))
                if np.isfinite(vte).any() or np.isfinite(ste).any():
                    recs.append(pl.DataFrame({**kw, "model": f"nat|{sp}", "h": np.where(g_te, vte, np.nan),
                                              "ls": np.where(g_te, ste, np.nan), "pit": np.full(nte, np.nan)}))
        print("forecasts", sym, flush=True)
    fc = pl.concat(recs, how="vertical_relaxed")
    return fc.join(panel.select("symbol", "date", "rv", "ret"), on=["symbol", "date"], how="left").with_columns(
        pl.col("date").map_elements(seg_of, return_dtype=pl.Utf8).alias("seg"))


def date_matrix(fc: pl.DataFrame, value: str, models: list[str], seg: str) -> tuple[np.ndarray, list[int]]:
    """Date x model matrix of the cross-symbol mean of `value`, on the intersection of (symbol, date) rows where every
    model has a finite value."""
    w = (fc.filter(pl.col("model").is_in(models) & (pl.col("seg") == seg) & pl.col(value).is_finite())
         .pivot(on="model", index=["symbol", "date"], values=value).drop_nulls())
    w = w.select(["symbol", "date"] + models)
    d = w.group_by("date").agg([pl.col(m).mean() for m in models]).sort("date")
    return d.select(models).to_numpy(), [w.height, d.height]


def forecast_tests(fc: pl.DataFrame) -> dict:
    fc = fc.with_columns(qlike(pl.col("rv"), pl.col("h")).alias("ql"), (-pl.col("ls")).alias("nls"))
    models = fc["model"].unique().to_list()
    has_h = set(fc.filter(pl.col("h").is_finite())["model"].unique().to_list())
    has_ls = set(fc.filter(pl.col("ls").is_finite())["model"].unique().to_list())
    var_models = sorted(m for m in models if m in has_h and (m in ("har", "ewma") or m.startswith(("har|", "nat|"))))
    den_models = sorted(m for m in models if m in has_ls and (m in ("har", "ewma") or m.startswith(("har|", "harmu|", "nat|"))))
    out = {"var_models": var_models, "den_models": den_models}
    for seg in ("llm_in_sample", "clean_holdout"):
        for key, val, ms in (("qlike", "ql", var_models), ("logscore", "nls", den_models)):
            M, (nrows, ndates) = date_matrix(fc, val, ms, seg)
            res = {"n_rows": nrows, "n_dates": ndates, "mean": dict(zip(ms, M.mean(0).tolist(), strict=True))}
            ib = ms.index("har")
            res["dm_vs_har"] = {m: dm_test(M[:, j] - M[:, ib]) for j, m in enumerate(ms) if m != "har"}
            if key == "qlike":
                for v0 in ("har|v0_cell", "har|v0_vol", "har|v0_dir"):
                    if v0 in ms:
                        res[f"dm_vs_{v0}"] = {m: dm_test(M[:, j] - M[:, ms.index(v0)]) for j, m in enumerate(ms) if m != v0}
            else:
                res["dm_mu_vs_cond"] = {m: dm_test(M[:, j] - M[:, ms.index("har|" + m.split("|")[1])])
                                        for j, m in enumerate(ms) if m.startswith("harmu|") and "har|" + m.split("|")[1] in ms}
            res["mcs"] = mcs(M, ms)
            out[f"{key}_{seg}"] = res
        out[f"calibration_{seg}"] = calibration(fc.filter(pl.col("seg") == seg))
    return out


def calibration(fc: pl.DataFrame) -> dict:
    """PIT checks per t-based model: 90% central coverage, KS vs U(0,1), Berkowitz moments of z = Phi^-1(PIT) with the
    LR test of (mean 0, variance 1) (iid version; the AR(1) term is reported, not tested)."""
    out = {}
    for m in fc.filter(pl.col("pit").is_finite())["model"].unique().sort().to_list():
        u = fc.filter((pl.col("model") == m) & pl.col("pit").is_finite()).sort("symbol", "date")["pit"].to_numpy()
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = stats.norm.ppf(u)
        n = len(z)
        mu, var = z.mean(), z.var()
        lr = n * (var - 1 - math.log(var)) + n * mu ** 2
        out[m] = {"n": n, "cov90": float(((u > 0.05) & (u < 0.95)).mean()), "tail_lo_1pct": float((u < 0.01).mean()),
                  "tail_hi_1pct": float((u > 0.99).mean()), "ks": float(stats.kstest(u, "uniform").statistic),
                  "z_mean": float(mu), "z_var": float(var), "z_ar1": float(np.corrcoef(z[1:], z[:-1])[0, 1]),
                  "berkowitz_lr2_p": float(stats.chi2.sf(lr, 2))}
    return out


def properties(panel: pl.DataFrame) -> dict:
    """Empirical properties of the daily inputs per symbol (llm_in_sample days only): ADF / KPSS, Ljung-Box(10) on r, r^2,
    log RV, moments; summarised across symbols."""
    import warnings

    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss

    rows = []
    for sym in panel["symbol"].unique().sort().to_list():
        g = panel.filter((pl.col("symbol") == sym) & pl.col("ok") & (pl.col("seg") == "llm_in_sample")).sort("date")
        if g.height < 400:
            continue
        r, lrv = g["ret"].to_numpy(), np.log(g["rv"].to_numpy())
        rec = {"symbol": sym, "n": g.height, "skew_r": float(stats.skew(r)), "kurt_r": float(stats.kurtosis(r, fisher=False))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for nm, x in (("r", r), ("lrv", lrv)):
                rec[f"adf_p_{nm}"] = float(adfuller(x, autolag="AIC")[1])
                rec[f"kpss_p_{nm}"] = float(kpss(x, regression="c", nlags="auto")[1])
            for nm, x in (("r", r), ("r2", r ** 2), ("lrv", lrv)):
                rec[f"lb10_p_{nm}"] = float(acorr_ljungbox(x, lags=[10])["lb_pvalue"].iloc[0])
        rec["acf1_lrv"] = float(np.corrcoef(lrv[1:], lrv[:-1])[0, 1])
        rec["acf22_lrv"] = float(np.corrcoef(lrv[22:], lrv[:-22])[0, 1])
        rows.append(rec)
    d = pl.DataFrame(rows)
    summ = {c: float(d[c].median()) for c in d.columns if c not in ("symbol",)}
    summ["share_adf_reject_r"] = float((d["adf_p_r"] < 0.05).mean())
    summ["share_kpss_reject_r"] = float((d["kpss_p_r"] < 0.05).mean())
    summ["share_adf_reject_lrv"] = float((d["adf_p_lrv"] < 0.05).mean())
    summ["share_kpss_reject_lrv"] = float((d["kpss_p_lrv"] < 0.05).mean())
    summ["share_lb_reject_r"] = float((d["lb10_p_r"] < 0.05).mean())
    summ["share_lb_reject_r2"] = float((d["lb10_p_r2"] < 0.05).mean())
    return {"median": summ, "n_symbols": d.height}


def _romano_wolf(comps: dict[str, tuple[float, np.ndarray]], keys: list[str]) -> dict[str, float]:
    """Studentised one-sided stepdown (Romano-Wolf) on bootstrap draws of the statistics (centred at the estimate)."""
    obs = np.array([comps[k][0] for k in keys])
    bs = np.array([comps[k][1] for k in keys]).T
    se = np.nanstd(bs, axis=0) + 1e-12
    tob = obs / se
    tb = (bs - obs[None, :]) / se[None, :]
    order = list(np.argsort(-tob))
    p_adj, run, alive = {}, 0.0, list(order)
    for j in order:
        mx = np.nanmax(tb[:, alive], axis=1)
        run = max(run, float(np.mean(mx >= tob[j])))
        p_adj[keys[j]] = run
        alive.remove(j)
    return p_adj


def strategy_tests(panel: pl.DataFrame, tabs: dict[str, pl.DataFrame], n_boot: int = 2000, seed: int = 3) -> dict:
    sp = (pl.read_parquet(ROOT / "data" / "priors" / "strategy_panel_daily.parquet")
          .filter(pl.col("universe") == "L").select("symbol", "date", "strategy", "pnl_d", "cell"))
    strategies = sorted(sp["strategy"].unique().to_list())
    S = len(strategies)
    sidx = {s: i for i, s in enumerate(strategies)}
    wide = sp.pivot(on="strategy", index=["symbol", "date"], values="pnl_d").sort("symbol", "date")
    cellw = sp.filter(pl.col("strategy") == strategies[0]).select("symbol", "date", "cell")
    wide = wide.join(cellw, on=["symbol", "date"], how="left")
    PN = wide.select(strategies).to_numpy()
    kd = {(s, d): i for i, (s, d) in enumerate(zip(wide["symbol"].to_list(), wide["date"].to_list(), strict=True))}
    dates_all = wide["date"].to_list()
    gates, labs = {}, {}
    common = np.ones(len(kd), bool)
    for spn, t in tabs.items():
        K = len(_pcols(t))
        lab_oos, fold_oos = np.full(len(kd), -1), np.full(len(kd), -1)
        gates[spn] = {}
        for f, o in enumerate(ORIGINS):
            o2 = ORIGINS[f + 1] if f + 1 < len(ORIGINS) else date(2100, 1, 1)
            if spn in V0_SPECS:
                ttr = t.filter(pl.col("date") <= o - timedelta(days=EMBARGO + 1))
                tte = t.filter((pl.col("date") >= o) & (pl.col("date") < o2))
            else:
                tf = t.filter(pl.col("fold") == f)
                ttr, tte = tf.filter(~pl.col("oos")), tf.filter(pl.col("oos"))
            if ttr.height == 0 or tte.height == 0:
                continue
            ii = np.array([kd.get(k, -1) for k in zip(ttr["symbol"].to_list(), ttr["date"].to_list(), strict=True)])
            lt = ttr["label"].to_numpy().astype(int)
            m = (ii >= 0) & (lt >= 0)
            if m.sum() < 100:
                continue
            ii, lt, st = ii[m], lt[m], np.array(ttr["symbol"].to_list())[m]
            X = PN[ii]
            pool = np.nanmean(X, axis=0)
            e = _episodes(lt, st, K)
            means = np.array([np.nanmean(X[lt == k], axis=0) if (lt == k).any() else pool for k in range(K)])
            w = (e / (e + KAPPA))[:, None]
            gates[spn][f] = ((w * means + (1 - w) * pool[None, :]) > 0, pool > 0)
            jj = np.array([kd.get(k, -1) for k in zip(tte["symbol"].to_list(), tte["date"].to_list(), strict=True)])
            lj = tte["label"].to_numpy().astype(int)
            mm = (jj >= 0) & (lj >= 0)
            lab_oos[jj[mm]] = lj[mm]
            fold_oos[jj[mm]] = f
        labs[spn] = (lab_oos, fold_oos)
        common &= (lab_oos >= 0) & np.isin(fold_oos, list(gates[spn]))
    rows = np.where(common)[0]
    out = {"n_symbol_days": int(len(rows)), "strategies": strategies}
    zoo = sidx["zoo_ensemble"]
    Xn = np.nan_to_num(PN[rows])
    daily = {}
    fold_ref = labs["v0_cell"][1][rows]
    U = np.zeros((len(rows), S), bool)
    for f in np.unique(fold_ref):
        U[fold_ref == f] = gates["v0_cell"][int(f)][1][None, :]
    nu_ = U.sum(1)
    daily["uncond"] = np.where(nu_ > 0, (Xn * U).sum(1) / np.maximum(nu_, 1), 0.0)
    daily["ew"] = Xn.mean(1)
    daily["sleeve|ungated"] = Xn[:, zoo]
    cells = wide["cell"].to_numpy()[rows]
    blocked = np.zeros((len(rows), S), bool)
    for sts, cl in RULES.values():
        inc = np.isin(cells, cl)
        for s_ in sts:
            if s_ in sidx:
                blocked[inc, sidx[s_]] = True
    daily["ew_rules"] = (Xn * ~blocked).sum(1) / S
    gate_share = {}
    for spn in tabs:
        lab, fold = labs[spn][0][rows], labs[spn][1][rows]
        G = np.zeros((len(rows), S), bool)
        for f in np.unique(fold):
            m = fold == f
            G[m] = gates[spn][int(f)][0][lab[m]]
        nsel = G.sum(1)
        daily[f"book|{spn}"] = np.where(nsel > 0, (Xn * G).sum(1) / np.maximum(nsel, 1), 0.0)
        daily[f"sleeve|{spn}"] = Xn[:, zoo] * G[:, zoo]
        gate_share[spn] = {"zoo_on": float(G[:, zoo].mean()), "mean_selected": float(nsel.mean())}
    out["gate_share"] = gate_share
    dd = pl.DataFrame({"date": [dates_all[i] for i in rows], **daily})
    dser = dd.group_by("date").agg([pl.col(c).mean() for c in daily]).sort("date")
    dl = dser["date"].to_list()
    rng = np.random.default_rng(seed)
    for seg in ("llm_in_sample", "clean_holdout"):
        msk = np.array([seg_of(d) == seg for d in dl])
        Dm = {c: dser[c].to_numpy()[msk] for c in daily}
        T = int(msk.sum())
        idx = [block_idx(T, 10, rng) for _ in range(n_boot)]
        res = {"n_dates": T, "sharpe": {c: sharpe(v) for c, v in Dm.items()},
               "mean_x100": {c: float(np.mean(v) * 100) for c, v in Dm.items()}}
        sh_b = {c: np.array([sharpe(v[ix]) for ix in idx]) for c, v in Dm.items()}
        comps = {}
        for spn in tabs:
            comps[f"sleeve|{spn} - sleeve|ungated"] = (res["sharpe"][f"sleeve|{spn}"] - res["sharpe"]["sleeve|ungated"],
                                                       sh_b[f"sleeve|{spn}"] - sh_b["sleeve|ungated"])
            comps[f"book|{spn} - uncond"] = (res["sharpe"][f"book|{spn}"] - res["sharpe"]["uncond"], sh_b[f"book|{spn}"] - sh_b["uncond"])
            if spn != "v0_cell":
                comps[f"book|{spn} - book|v0_cell"] = (res["sharpe"][f"book|{spn}"] - res["sharpe"]["book|v0_cell"],
                                                       sh_b[f"book|{spn}"] - sh_b["book|v0_cell"])
        for a, b_ in (("ew_rules", "ew"), ("uncond", "ew"), ("book|v0_cell", "ew")):
            comps[f"{a} - {b_}"] = (res["sharpe"][a] - res["sharpe"][b_], sh_b[a] - sh_b[b_])
        cres = {}
        for k, (obs, bs) in comps.items():
            se = float(np.nanstd(bs))
            cres[k] = {"diff": float(obs), "se": se, "p_one": float(np.nanmean((bs - obs) >= obs)),
                       "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "mde": 2.8 * se}
        for fam in ([k for k in comps if k.startswith("sleeve|")], [k for k in comps if k.startswith("book|") and k.endswith("uncond")],
                    [k for k in comps if k.endswith("book|v0_cell")]):
            for k, v in _romano_wolf(comps, fam).items():
                cres[k]["p_rw"] = v
        res["comparisons"] = cres
        sl = {k: v for k, v in Dm.items() if k.startswith("sleeve|") and k != "sleeve|ungated"}
        srs = np.array([np.mean(v) / np.std(v, ddof=1) for v in sl.values()])
        best = max(sl, key=lambda k: np.mean(sl[k]) / np.std(sl[k], ddof=1))
        v = sl[best]
        res["dsr_best_sleeve"] = {"spec": best, "dsr": deflated_sharpe(float(np.mean(v) / np.std(v, ddof=1)), len(v), srs,
                                                                    float(stats.skew(v)), float(stats.kurtosis(v, fisher=False)))}
        out[seg] = res
    return out


def encompassing(panel: pl.DataFrame, tabs: dict[str, pl.DataFrame]) -> dict:
    import statsmodels.api as sm

    sp = (pl.read_parquet(ROOT / "data" / "priors" / "strategy_panel_daily.parquet")
          .filter((pl.col("universe") == "L") & pl.col("active")).select("symbol", "date", "strategy", "pnl_d", "cell"))
    g3 = tabs["ghmm3"].filter(pl.col("oos")).select("symbol", "date", pl.col("label").alias("g3"))
    feats = panel.select("symbol", "date", "vpct", "dir_score")
    out = {}
    for rid, (sts, cells) in RULES.items():
        x = (sp.filter(pl.col("strategy").is_in(sts)).join(g3, on=["symbol", "date"], how="inner")
             .join(feats, on=["symbol", "date"], how="inner").drop_nulls(["cell", "vpct", "dir_score"])
             .filter((pl.col("g3") >= 0) & pl.col("vpct").is_finite() & pl.col("dir_score").is_finite())
             .with_columns(pl.col("date").map_elements(seg_of, return_dtype=pl.Utf8).alias("seg")))
        res = {}
        for seg in ("llm_in_sample", "clean_holdout"):
            xs = x.filter(pl.col("seg") == seg)
            if xs.height < 50:
                res[seg] = {"n": xs.height}
                continue
            y = xs["pnl_d"].to_numpy() * 100
            ind = xs["cell"].is_in(list(cells)).to_numpy().astype(float)
            sfe = (np.column_stack([(xs["strategy"] == s).to_numpy().astype(float) for s in sts[1:]])
                   if len(sts) > 1 else np.zeros((xs.height, 0)))
            g = xs["g3"].to_numpy()
            gd = np.column_stack([(g == k).astype(float) for k in (1, 2)])
            ds = xs["dir_score"].to_numpy()
            cont = np.column_stack([xs["vpct"].to_numpy(), ds, np.abs(ds)])
            grp = np.array([d.toordinal() for d in xs["date"].to_list()])
            r = {"n": xs.height, "share_in_cells": float(ind.mean())}
            for name, extra in (("raw", np.zeros((xs.height, 0))), ("ghmm3", gd), ("continuous", cont)):
                Xm = sm.add_constant(np.column_stack([ind, sfe, extra]), has_constant="add")
                try:
                    fit = sm.OLS(y, Xm).fit(cov_type="cluster", cov_kwds={"groups": grp})
                    r[name] = {"coef": float(fit.params[1]), "t": float(fit.tvalues[1])}
                except Exception as ex:                                      # noqa: BLE001 (singular design)
                    r[name] = {"error": str(ex)[:80]}
            res[seg] = r
        out[rid] = res
    return out


def rvol8(panel: pl.DataFrame, tabs: dict[str, pl.DataFrame]) -> dict:
    """H-WS07-9: high-minus-low vol-state contrast of log(RV_7d / 7 EWMA var) and of log(RV_7d / direct log-HAR 7d)."""
    import statsmodels.api as sm

    recs = []
    for sym in panel["symbol"].unique().sort().to_list():
        g = panel.filter((pl.col("symbol") == sym) & pl.col("ok")).sort("date")
        n = g.height
        if n < MIN_TRAIN + 60:
            continue
        dts = g["date"].to_list()
        dord = np.array([d.toordinal() for d in dts])
        rv = g["rv"].to_numpy()
        c = np.r_[0.0, np.cumsum(rv)]
        idx = np.arange(n)
        consec = np.r_[dord[6:] - dord[:-6] == 6, np.zeros(6, bool)]
        s7 = np.r_[c[7:] - c[:-7], np.ones(6)]
        f7 = np.where(consec, np.log(np.maximum(s7, 1e-300)), np.nan)
        y_e = f7 - np.log(7 * g["ewma_var"].to_numpy())
        lrv = np.log(rv)
        x1 = np.r_[np.nan, lrv[:-1]]
        x5 = np.where(idx >= 5, np.log(np.maximum(c[idx] - c[np.maximum(idx - 5, 0)], 1e-300) / 5), np.nan)
        x22 = np.where(idx >= 22, np.log(np.maximum(c[idx] - c[np.maximum(idx - 22, 0)], 1e-300) / 22), np.nan)
        H = np.column_stack([np.ones(n), x1, x5, x22])
        y_h = np.full(n, np.nan)
        for f, o in enumerate(ORIGINS):
            o2 = ORIGINS[f + 1] if f + 1 < len(ORIGINS) else date(2100, 1, 1)
            tr = np.isfinite(H).all(1) & np.isfinite(f7) & (dord + 6 <= (o - timedelta(days=EMBARGO + 1)).toordinal())  # purged
            te = np.isfinite(H).all(1) & (dord >= o.toordinal()) & (dord < o2.toordinal())
            if tr.sum() < MIN_TRAIN or te.sum() == 0:
                continue
            b = np.linalg.lstsq(H[tr], f7[tr], rcond=None)[0]
            s2 = float(np.var(f7[tr] - H[tr] @ b))
            y_h[te] = f7[te] - (H[te] @ b + s2 / 2)
        recs.append(pl.DataFrame({"symbol": [sym] * n, "date": dts, "y_e": y_e, "y_h": y_h}))
    Y = pl.concat(recs)
    out = {}
    for spn in ("v0_vol", "ghmm3", "bocpd"):
        if spn not in tabs:
            continue
        t = tabs[spn] if spn in V0_SPECS else tabs[spn].filter(pl.col("oos"))
        x = (Y.join(t.select("symbol", "date", "label"), on=["symbol", "date"], how="inner")
             .filter((pl.col("label") >= 0) & (pl.col("date") >= ORIGINS[0]))
             .with_columns(pl.col("date").map_elements(seg_of, return_dtype=pl.Utf8).alias("seg")))
        res = {}
        for seg in ("llm_in_sample", "clean_holdout"):
            xs = x.filter(pl.col("seg") == seg)
            lab = xs["label"].to_numpy()
            grp = np.array([d.toordinal() // 14 for d in xs["date"].to_list()])
            r = {"n": xs.height}
            for yc in ("y_e", "y_h"):
                y = xs[yc].to_numpy()
                m = np.isfinite(y)
                Xm = sm.add_constant(np.column_stack([lab[m] == 1, lab[m] == 2]).astype(float), has_constant="add")
                fit = sm.OLS(y[m], Xm).fit(cov_type="cluster", cov_kwds={"groups": grp[m]})
                r[yc] = {"high_minus_low": float(fit.params[2]), "t": float(fit.tvalues[2]), "n": int(m.sum())}
            res[seg] = r
        out[spn] = res
    return out


def describe(tabs: dict[str, pl.DataFrame]) -> dict:
    """Agreement (ARI, VI) on common OOS rows, persistence, refit stability (from the fit diagnostics)."""
    labs = {}
    for spn, t in tabs.items():
        tt = t if spn in V0_SPECS else t.filter(pl.col("oos"))
        labs[spn] = tt.filter(pl.col("date") >= ORIGINS[0]).select("symbol", "date", pl.col("label").cast(pl.Int16).alias(spn))
    names = list(labs)
    J = labs[names[0]]
    for nme in names[1:]:
        J = J.join(labs[nme], on=["symbol", "date"], how="inner")
    J = J.sort("symbol", "date")
    L = {nme: J[nme].to_numpy().astype(int) for nme in names}
    ari = {a: {b: round(_ari(L[a], L[b]), 3) for b in names} for a in names}
    vi = {a: {b: round(variation_of_information(L[a], L[b]), 3) for b in names} for a in names}
    sy = J["symbol"].to_numpy()
    pers = {}
    for nme in names:
        lab = L[nme]
        ch = (lab[1:] != lab[:-1]) | (sy[1:] != sy[:-1])
        pers[nme] = {"mean_run_days": float(len(lab) / (int(ch.sum()) + 1)),
                     "share_by_state": (np.bincount(lab[lab >= 0]) / max((lab >= 0).sum(), 1)).round(3).tolist()}
    diag = {}
    for spn in tabs:
        f = OUT / "fits" / f"{spn}_diag.json"
        if f.exists():
            dg = json.loads(f.read_text())
            ar = [d["ari_refit"] for d in dg if d.get("ari_refit") is not None and np.isfinite(d["ari_refit"])]
            sf = [d["ari_smooth_vs_filt"] for d in dg if d.get("ari_smooth_vs_filt") is not None]
            ed = [d["exp_dur"] for d in dg if "exp_dur" in d]
            diag[spn] = {"median_ari_refit": float(np.median(ar)) if ar else None,
                         "q25_ari_refit": float(np.percentile(ar, 25)) if ar else None, "n_refits": len(ar),
                         "median_ari_smooth_vs_filt": float(np.median(sf)) if sf else None,
                         "median_exp_dur_days": np.median(np.array(ed), axis=0).round(1).tolist() if ed else None,
                         "converged_share": float(np.mean([d.get("converged", True) for d in dg])),
                         "median_fit_secs": float(np.median([d.get("secs", 0) for d in dg]))}
    return {"n_common_rows": J.height, "ari": ari, "vi": vi, "persistence": pers, "fit_diag": diag}


def evaluate() -> None:
    panel = load_panel()
    tabs = spec_tables(panel)
    t0 = time.time()
    fc = forecasts(panel, tabs)
    fc.write_parquet(OUT / "forecasts.parquet")
    res = {"specs": list(tabs), "forecast": forecast_tests(fc)}
    print("forecast tests", round(time.time() - t0, 1), flush=True)
    res["strategy"] = strategy_tests(panel, tabs)
    print("strategy tests", round(time.time() - t0, 1), flush=True)
    res["encompassing"] = encompassing(panel, tabs)
    res["rvol8"] = rvol8(panel, tabs)
    res["describe"] = describe(tabs)
    res["properties"] = properties(panel)
    (OUT / "results.json").write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1))
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["panel", "fit", "eval"])
    ap.add_argument("--specs", default=",".join(tuple(HMM_SPECS) + OTHER_SPECS))
    a = ap.parse_args()
    if a.cmd == "panel":
        print(build_panel())
    elif a.cmd == "fit":
        run_fit(a.specs.split(","))
    else:
        evaluate()
