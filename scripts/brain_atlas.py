"""Neuron atlas of the frozen live 1h network (descriptive, historical; research only).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/brain_atlas.py

Runs the exported live network (data/models/pnn/brain_1h.json, both seeds, float64 forward = ai_trader.livedash.brain)
over the cached hourly panel (data/models/pnn/cache/btc_1h_*.parquet, the H-PNN1 features) from 2025-01-01, and
describes every latent unit (A.z1..A.z4 = seed 0, B.z1..B.z4 = seed 1) and head unit (A.o1..B.o7):
  receptive field   share of the unit's variance-of-edges (sd of phi_ji(x_i) over hours), the edge curves; for the
                    network's sigma (q90 - q10) / 2.5631, the partial dependence on every input (ICE mean, grid p2..p98)
  when active       mean standardised activation by regime cell (regimes v0 of the bar before S, known at S), by
                    hour of week, and around jump bars (|r| / s_pre > 4) from -12 h to +24 h
  sparsity          share of hours with |a - mean| > 1 sd (and > 2 sd), excess kurtosis
  redundancy        correlation matrix of the 8 latent units; mean-ablation (unit set to its 2025-26 mean) and the
                    change in mean pinball (7 taus, model units y = r / sqrt(s2)); same for every input (set to its
                    training mean, standardised 0) and for each whole seed (hemisphere)
Segments (research protocol): llm_in_sample (< research.knowledge_cutoff 2026-07-01) and clean_holdout (>=). The frozen
net was fit on rows to 2025-11-12 and early-stopped on 2025-11-14..2026-10-30, so 2025-2026 is in-sample or validation
for the network itself: this atlas describes, it does not test.
Outputs: research/reports/brain_atlas_2026-11-15.md (+ .json) and research/figures/brain_atlas_2026-11-15/*.png.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("POLARS_MAX_THREADS", "2")

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.livedash import brain as B

TAG = "2026-11-15"
START = datetime(2025, 1, 1, tzinfo=UTC)
CUTOFF = datetime(2026, 7, 1, tzinfo=UTC)
FIT_END = datetime(2025, 11, 13, tzinfo=UTC)
VAL_END = datetime(2026, 10, 31, tzinfo=UTC)
FIG = ROOT / "research" / "figures" / f"brain_atlas_{TAG}"
REP = ROOT / "research" / "reports" / f"brain_atlas_{TAG}.md"
SEEDS = ("A", "B")
NORM80 = B.NORM80


def load_panel(net: dict) -> pl.DataFrame:
    p = B.panel_path(ROOT / "data" / "models" / "pnn")
    d = pl.read_parquet(p).filter(pl.col("event_ts") >= START).sort("event_ts")
    keep = pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in (*net["features"], "r", "s2")]) & (pl.col("s2") > 0)
    return d.filter(keep), p.name


def regime_cells(ts_min: datetime, ts_max: datetime) -> pl.DataFrame | None:
    """regimes v0 cell of each 1h bar, shifted so row S carries the cell of bar S-1 (known at S)."""
    try:
        from ai_trader.research.frame import market_frame
        from ai_trader.research.regimes import label

        f = market_frame("BTCUSDT", "1h", ts_max + timedelta(hours=2), funding=False, oi=False)
        f = f.filter(pl.col("event_ts") >= ts_min - timedelta(days=60))
        lab = label(f).select("event_ts", "cell")
        return lab.with_columns((pl.col("event_ts") + timedelta(hours=1)).alias("event_ts")).rename({"cell": "cell_prev"})
    except Exception as e:  # noqa: BLE001 - descriptive extra; the atlas runs without it
        print("regime labels unavailable:", type(e).__name__, e)
        return None


def pinball_mean(y: np.ndarray, q: np.ndarray, mask: np.ndarray) -> float:
    from ai_trader.ml.pnn import pinball_np

    return float(pinball_np(y[mask], q[mask]).mean())


def head_q(net: dict, s: int, z: np.ndarray) -> np.ndarray:
    """Quantiles of seed s from its latent z (N, 4)."""
    sc, D = float(net["scale"]), int(net["degree"])
    L = net["seeds"][s]["head"]
    coef, w, b = (np.asarray(L[k], float) for k in ("coef", "w_res", "bias"))
    u = np.tanh(sc * z)
    ts = [u, 2 * u * u - 1]
    for _ in range(2, D):
        ts.append(2 * u * ts[-1] - ts[-2])
    Bz = np.stack(ts[:D], axis=-1)
    o = np.einsum("nid,oid->no", Bz, coef) + (z / (1 + np.exp(-z))) @ w.T + b
    return np.stack([B.quantiles_from_raw(r) for r in o])


def kurt(a: np.ndarray) -> float:
    a = (a - a.mean()) / a.std()
    return float(np.mean(a ** 4) - 3)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    pnn_dir = ROOT / "data" / "models" / "pnn"
    net = B.load_net(pnn_dir)
    feats = net["features"]
    d, panel_name = load_panel(net)
    ts = d["event_ts"].to_list()
    X = np.column_stack([d[c].cast(pl.Float64).to_numpy() for c in feats])
    y = d["r"].to_numpy() / np.sqrt(d["s2"].to_numpy())
    N = len(y)
    tsn = np.array([t.timestamp() for t in ts])
    seg = {"all": np.ones(N, bool), "llm_in_sample": tsn < CUTOFF.timestamp(), "clean_holdout": tsn >= CUTOFF.timestamp()}
    blk = {"fit": tsn < FIT_END.timestamp(), "val": (tsn >= FIT_END.timestamp()) & (tsn < VAL_END.timestamp()), "test": tsn >= VAL_END.timestamp()}
    fb = B.forward_batch(net, X)
    Z = [s["z"] for s in fb["seeds"]]
    names = [f"{SEEDS[s]}.z{j + 1}" for s in range(2) for j in range(4)]
    Zall = np.column_stack(Z)  # (N, 8)
    mu, sd = Zall.mean(0), Zall.std(0)
    Zs = (Zall - mu) / sd
    q = fb["q"]
    base_pin = {k: pinball_mean(y, q, m) for k, m in seg.items()}
    sigma = (q[:, 5] - q[:, 1]) / NORM80
    out: dict = {"panel": panel_name, "n": N, "from": str(ts[0]), "to": str(ts[-1]), "base_pinball": base_pin,
                 "blocks": {k: int(v.sum()) for k, v in blk.items()}, "segments": {k: int(v.sum()) for k, v in seg.items()}, "units": {}}

    # ---- receptive fields (edge sd shares) and edge curves
    rf = np.zeros((8, len(feats)))
    for s in range(2):
        E = fb["seeds"][s]["enc_edges"]  # (N, 4, 19)
        esd = E.std(axis=0)
        for j in range(4):
            rf[s * 4 + j] = esd[j] / esd[j].sum()
    corr_in = np.zeros((8, len(feats)))  # sign of each edge: corr(phi, x_i)
    for s in range(2):
        E = fb["seeds"][s]["enc_edges"]
        for j in range(4):
            for i in range(len(feats)):
                a, b = E[:, j, i], fb["x"][:, i]
                corr_in[s * 4 + j, i] = np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else 0.0
    # head receptive fields over latents
    head_rf = {}
    for s in range(2):
        E2 = fb["seeds"][s]["head_edges"]  # (N, 7, 4)
        esd = E2.std(axis=0)
        for k in range(7):
            head_rf[f"{SEEDS[s]}.o{k + 1}"] = (esd[k] / esd[k].sum()).tolist()

    # ---- partial dependence of the ensemble sigma (model units) on each input: ICE mean over a subsample, grid p2..p98
    rng = np.random.default_rng(0)
    sub = rng.choice(N, min(N, 3000), replace=False)
    pd_sig = {}
    for i, c in enumerate(feats):
        lo, hi = np.percentile(X[:, i], [2, 98])
        grid = np.unique(np.linspace(lo, hi, 9)) if hi > lo else np.array([lo])
        vals = []
        for g in grid:
            Xg = X[sub].copy()
            Xg[:, i] = g
            qg = B.forward_batch(net, Xg)["q"]
            vals.append(float(np.mean((qg[:, 5] - qg[:, 1]) / NORM80)))
        pd_sig[c] = {"grid": grid.tolist(), "sigma": vals, "range": float(max(vals) - min(vals)),
                     "slope_sign": int(np.sign(vals[-1] - vals[0])) if len(vals) > 1 else 0}

    # ---- when active: regime cells, hour of week, jumps
    cells = regime_cells(ts[0], ts[-1])
    by_cell = None
    if cells is not None:
        jj = pl.DataFrame({"event_ts": ts}).join(cells, on="event_ts", how="left")["cell_prev"].to_list()
        cl = np.array([c if c is not None else "" for c in jj])
        by_cell = {}
        for c in sorted(set(cl) - {""}):
            m = cl == c
            by_cell[c] = {"n": int(m.sum()), "mean_z": Zs[m].mean(0).tolist(), "mean_abs_z": np.abs(Zs[m]).mean(0).tolist()}
    how = np.array([(t.weekday() * 24 + t.hour) for t in ts])
    hw = np.full((8, 168), np.nan)
    for h in range(168):
        m = how == h
        if m.any():
            hw[:, h] = Zs[m].mean(0)
    s_pre = np.sqrt(d["s2"].to_numpy())
    jump = np.abs(d["r"].to_numpy()) / s_pre > 4.0  # bar S itself jumped (its return is the row's target)
    jidx = np.flatnonzero(jump)
    lags = np.arange(-12, 25)
    ev = np.full((8, len(lags)), np.nan)
    for k, L in enumerate(lags):
        ii = jidx + L
        ii = ii[(ii >= 0) & (ii < N)]
        ev[:, k] = Zs[ii].mean(0)

    # ---- sparsity
    for u, nm in enumerate(names):
        a = Zs[:, u]
        out["units"][nm] = {"mean": float(mu[u]), "sd": float(sd[u]), "share_gt1sd": float(np.mean(np.abs(a) > 1)),
                            "share_gt2sd": float(np.mean(np.abs(a) > 2)), "excess_kurtosis": kurt(Zall[:, u]),
                            "rf": {feats[i]: float(rf[u, i]) for i in np.argsort(-rf[u])[:5]},
                            "rf_sign": {feats[i]: float(corr_in[u, i]) for i in np.argsort(-rf[u])[:5]},
                            "corr_sigma": float(np.corrcoef(Zall[:, u], sigma)[0, 1]), "corr_q50": float(np.corrcoef(Zall[:, u], q[:, 3])[0, 1]),
                            "hour_of_week_max": int(np.nanargmax(hw[u])), "hour_of_week_min": int(np.nanargmin(hw[u])),
                            "hour_of_week_amp": float(np.nanmax(hw[u]) - np.nanmin(hw[u])),
                            "jump_lag0": float(ev[u, list(lags).index(0)]), "jump_pre": float(np.nanmean(ev[u, lags < 0])),
                            "jump_post": float(np.nanmean(ev[u, (lags > 0) & (lags <= 6)])),
                            "jump_lag1": float(ev[u, list(lags).index(1)])}
        if by_cell:
            out["units"][nm]["by_cell"] = {c: float(v["mean_z"][u]) for c, v in by_cell.items()}

    # ---- redundancy: correlations and ablations
    C = np.corrcoef(Zall.T)
    out["corr"] = C.tolist()
    abl = {}
    for s in range(2):
        for j in range(4):
            z2 = Z[s].copy()
            z2[:, j] = Z[s][:, j].mean()
            qs = [fb["seeds"][t]["q"] if t != s else head_q(net, s, z2) for t in range(2)]
            qa = np.mean(qs, axis=0)
            abl[f"{SEEDS[s]}.z{j + 1}"] = {k: 100 * (pinball_mean(y, qa, m) / base_pin[k] - 1) for k, m in seg.items()}
    for s in range(2):
        abl[f"seed {SEEDS[s]} alone"] = {k: 100 * (pinball_mean(y, fb["seeds"][s]["q"], m) / base_pin[k] - 1) for k, m in seg.items()}
    abl_in = {}
    for i, c in enumerate(feats):
        Xa = X.copy()
        Xa[:, i] = net["std_mu"][i]
        qa = B.forward_batch(net, Xa)["q"]
        abl_in[c] = {k: 100 * (pinball_mean(y, qa, m) / base_pin[k] - 1) for k, m in seg.items()}
    # unconditional baseline for scale: the train-sample quantiles of y (constant forecast) over the same rows
    qc = np.tile(np.quantile(y[seg["llm_in_sample"]], B.TAUS), (N, 1))
    out["const_quantiles_pinball_pct"] = {k: 100 * (pinball_mean(y, qc, m) / base_pin[k] - 1) for k, m in seg.items()}
    out["ablation_units_pct"] = abl
    out["ablation_inputs_pct"] = abl_in
    out["head_rf"] = head_rf
    out["pd_sigma"] = pd_sig
    # cross-seed matching
    out["cross_seed_best"] = {names[j]: [names[4 + int(np.argmax(np.abs(C[j, 4:])))], float(C[j, 4 + int(np.argmax(np.abs(C[j, 4:])))])] for j in range(4)}

    # ---- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
    cmap = "RdBu_r"

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(rf, cmap="Blues", aspect="auto", vmin=0, vmax=rf.max())
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
    ax.set_yticks(range(8), names)
    for u in range(8):
        for i in range(len(feats)):
            if rf[u, i] >= 0.12:
                ax.text(i, u, f"{rf[u, i]:.2f}", ha="center", va="center", fontsize=7, color="white" if rf[u, i] > 0.25 else "black")
    fig.colorbar(im, ax=ax, label="share of the unit's edge sd")
    ax.set_title("Receptive fields: which inputs drive each latent unit (2025-01 to 2026-11)")
    fig.tight_layout()
    fig.savefig(FIG / "receptive_fields.png", dpi=130)
    plt.close(fig)

    # top-3 edge curves per latent unit
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.2), sharey=False)
    for u in range(8):
        s, j = divmod(u, 4)
        a = axs[s, j]
        for rank, i in enumerate(np.argsort(-rf[u])[:3]):
            xs = fb["x"][:, i]
            g = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99), 60)
            L = net["seeds"][s]["enc"]
            phi = [float(np.asarray(L["w_res"])[j, i] * (v / (1 + np.exp(-v)))
                         + sum(np.asarray(L["coef"])[j, i, k] * t for k, t in enumerate(B._basis(np.array([v]), net["scale"], net["degree"])[0]))) for v in g]
            a.plot(g, phi, lw=2, ls=("-", "--", ":")[rank], color=("#2a78d6", "#eb6834", "#1baf7a")[rank], label=feats[i])
        a.axhline(0, color="#999", lw=0.6)
        a.set_title(names[u])
        a.legend(fontsize=7, frameon=False)
        if s == 1:
            a.set_xlabel("standardised input x (p1 to p99)")
    fig.suptitle("Top-3 edge functions phi(x) into each latent unit")
    fig.tight_layout()
    fig.savefig(FIG / "edge_functions.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    v = np.nanmax(np.abs(hw))
    im = ax.imshow(hw, cmap=cmap, aspect="auto", vmin=-v, vmax=v)
    ax.set_yticks(range(8), names)
    ax.set_xticks([d_ * 24 for d_ in range(7)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    fig.colorbar(im, ax=ax, label="mean standardised activation")
    ax.set_title("When each latent unit is active: hour of week (UTC, forecast bar S)")
    fig.tight_layout()
    fig.savefig(FIG / "hour_of_week.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    cols = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    for u in range(8):
        s, j = divmod(u, 4)
        axs[s].plot(lags, ev[u], color=cols[j], lw=2, label=names[u])
    for s in range(2):
        axs[s].axvline(0, color="#999", lw=0.8)
        axs[s].axhline(0, color="#999", lw=0.6)
        axs[s].set_title(f"hemisphere {SEEDS[s]}: around jump bars (n = {len(jidx)})")
        axs[s].set_xlabel("hours from the jump bar (0 = the bar whose |r| / s_pre > 4)")
        axs[s].legend(fontsize=7, frameon=False)
    axs[0].set_ylabel("mean standardised activation")
    fig.tight_layout()
    fig.savefig(FIG / "jump_event.png", dpi=130)
    plt.close(fig)

    if by_cell:
        cs = sorted(by_cell)
        M = np.array([[by_cell[c]["mean_z"][u] for c in cs] for u in range(8)])
        fig, ax = plt.subplots(figsize=(8, 3.6))
        v = np.abs(M).max()
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=-v, vmax=v)
        ax.set_xticks(range(len(cs)), [f"{c}\nn={by_cell[c]['n']}" for c in cs], fontsize=7)
        ax.set_yticks(range(8), names)
        for u in range(8):
            for k in range(len(cs)):
                ax.text(k, u, f"{M[u, k]:+.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="mean standardised activation")
        ax.set_title("Latent activation by regime cell (regimes v0 of bar S-1)")
        fig.tight_layout()
        fig.savefig(FIG / "regime_cells.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(8), names, rotation=45, ha="right")
    ax.set_yticks(range(8), names)
    for a_ in range(8):
        for b_ in range(8):
            ax.text(b_, a_, f"{C[a_, b_]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Redundancy: correlation of latent units")
    fig.tight_layout()
    fig.savefig(FIG / "latent_correlation.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ks = list(abl)
    for k_, segn, colr in ((0, "llm_in_sample", "#2a78d6"), (1, "clean_holdout", "#eb6834")):
        axs[0].barh(np.arange(len(ks)) + (k_ - 0.5) * 0.38, [abl[x][segn] for x in ks], height=0.36, color=colr, label=segn)
    axs[0].set_yticks(range(len(ks)), ks)
    axs[0].axvline(0, color="#999", lw=0.8)
    axs[0].set_xlabel("pinball change when ablated (%)")
    axs[0].set_title("Units and hemispheres")
    axs[0].legend(fontsize=7, frameon=False)
    order = sorted(feats, key=lambda c: -abl_in[c]["all"])
    for k_, segn, colr in ((0, "llm_in_sample", "#2a78d6"), (1, "clean_holdout", "#eb6834")):
        axs[1].barh(np.arange(len(order)) + (k_ - 0.5) * 0.38, [abl_in[c][segn] for c in order], height=0.36, color=colr, label=segn)
    axs[1].set_yticks(range(len(order)), order)
    axs[1].axvline(0, color="#999", lw=0.8)
    axs[1].set_xlabel("pinball change when the input is set to its training mean (%)")
    axs[1].set_title("Inputs")
    for a in axs:
        a.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=130)
    plt.close(fig)

    top = sorted(feats, key=lambda c: -pd_sig[c]["range"])[:8]
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    for a, c in zip(axs.flat, top, strict=True):
        a.plot(pd_sig[c]["grid"], pd_sig[c]["sigma"], color="#2a78d6", lw=2, marker="o", ms=3)
        a.set_title(f"{c} (range {pd_sig[c]['range']:.3f})")
        a.set_xlabel("raw input (p2 to p98)")
    axs[0, 0].set_ylabel("mean network sigma (model units)")
    axs[1, 0].set_ylabel("mean network sigma (model units)")
    fig.suptitle("Partial dependence of the network's sigma on its 8 strongest inputs")
    fig.tight_layout()
    fig.savefig(FIG / "pd_sigma.png", dpi=130)
    plt.close(fig)

    out["figures"] = sorted(p.name for p in FIG.glob("*.png"))
    out["hour_of_week"] = hw.tolist()
    out["jump_event"] = {"lags": lags.tolist(), "mean_z": ev.tolist(), "n_jumps": len(jidx)}
    if by_cell:
        out["by_cell_n"] = {c: v["n"] for c, v in by_cell.items()}
    (REP.with_suffix(".json")).write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "from", "to", "base_pinball", "blocks", "segments")}, default=str, indent=1))
    print("units", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("by_cell",)}
                                for k, v in out["units"].items()}, indent=1, default=str))
    print("ablation units", json.dumps(abl, indent=1))
    print("ablation inputs", json.dumps({k: {s: round(v, 3) for s, v in d_.items()} for k, d_ in abl_in.items()}, indent=1))
    print("const", out["const_quantiles_pinball_pct"])
    print("pd ranges", {c: round(v["range"], 4) for c, v in pd_sig.items()})
    print("head_rf", json.dumps({k: [round(x, 2) for x in v] for k, v in head_rf.items()}))
    print("cross seed", out["cross_seed_best"])
    if by_cell:
        print("cells", {c: v["n"] for c, v in by_cell.items()})
        print("by_cell", {nm: {c: round(v, 2) for c, v in out["units"][nm]["by_cell"].items()} for nm in names})


if __name__ == "__main__":
    main()
