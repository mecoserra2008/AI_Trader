"""Figures for the deep-analogs report (research/figures/deep_analogs_2026-11-15/). Reads data/research/deep_analogs/
results.json + aligned.parquet. Colour = outcome class (bottomed vs lower low within 14 days), NOW in ink, so identity
does not need 11 hues; every series is direct-labelled at its line end."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ai_trader.research.deep_analogs_study import OUT  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "research" / "figures" / "deep_analogs_2026-11-15"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_BOT, C_LL = "#2a78d6", "#eb6834"   # categorical slots 1, 2 (validated pair)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def load():
    res = json.loads((OUT / "results.json").read_text())
    al = pl.read_parquet(OUT / "aligned.parquet")
    cls = {}
    for e, fp in res["fingerprints"].items():
        if e == "NOW":
            cls[e] = "now"
        else:
            cls[e] = "lower" if fp["outcome"].get("undercut14d") else "bottom"
    return res, al, cls


def _style(e, cls):
    if cls[e] == "now":
        return {"color": INK, "lw": 2.6, "zorder": 5}
    return {"color": C_LL if cls[e] == "lower" else C_BOT, "lw": 1.2, "alpha": 0.75, "zorder": 3}


def overlay(al, cls, col, xcol, fname, ylabel, title, scale=1.0, xlim=(-72, 168), step=False, ref0=True):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for e in sorted(cls, key=lambda x: x == "NOW"):
        d = al.filter(pl.col("ep") == e).sort(xcol).drop_nulls(col)
        if not d.height:
            continue
        x, y = d[xcol].to_numpy(), d[col].to_numpy() * scale
        m = (x >= xlim[0]) & (x <= xlim[1])
        x, y = x[m], y[m]
        if not len(x):
            continue
        (ax.step if step else ax.plot)(x, y, **_style(e, cls), **({"where": "post"} if step else {}))
        ax.annotate(e, (x[-1], y[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK2 if e != "NOW" else INK,
                    va="center", fontweight="bold" if e == "NOW" else None)
    if ref0:
        ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(xlim[0], xlim[1] + 14)
    ax.set_xlabel("hours from " + ("the analog hour T" if xcol == "k" else "the flush low"))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], color=INK, lw=2.6, label="NOW (2026-11-15 17:00)"),
                       Line2D([], [], color=C_BOT, lw=1.2, label="analog: no lower low within 14 d"),
                       Line2D([], [], color=C_LL, lw=1.2, label="analog: lower low within 14 d")], loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def liq_profiles(res, al, cls):
    eps = [e for e in res["fingerprints"]]
    fig, axs = plt.subplots(4, 3, figsize=(10, 9), sharex=True)
    for ax, e in zip(axs.flat, eps, strict=False):
        d = al.filter(pl.col("ep") == e).sort("k")
        d = d.filter((pl.col("k") >= -72) & (pl.col("k") <= 72))
        ax.bar(d["k"].to_numpy(), -d["fs_h"].fill_null(0).to_numpy(), width=1.0, color=C_LL, label="forced selling (proxy)")
        ax.bar(d["k"].to_numpy(), d["fb_h"].fill_null(0).to_numpy(), width=1.0, color=C_BOT, label="forced buying (proxy)")
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.axhline(0, color=INK2, lw=0.6)
        oc = res["fingerprints"][e]["outcome"]
        tag = "NOW" if e == "NOW" else ("lower low" if oc.get("undercut14d") else "no lower low")
        ax.set_title(f"{e} {res['fingerprints'][e]['T'][:10]} ({tag})", fontsize=8, loc="left", color=INK)
    ax = axs.flat[len(eps)]
    lq = OUT / "NOW" / "bybit_liq.parquet"
    if lq.exists():
        from datetime import datetime

        q = pl.read_parquet(lq).with_columns((pl.col("price") * pl.col("size")).alias("usd"))
        T = datetime.fromisoformat(res["fingerprints"]["NOW"]["T"])
        q = q.with_columns(((pl.col("event_ts") - T).dt.total_minutes() // 60).alias("k")).group_by("k", "side").agg(pl.col("usd").sum())
        lo = q.filter(pl.col("side") == "long").sort("k")
        sh = q.filter(pl.col("side") == "short").sort("k")
        ax.bar(lo["k"].to_numpy(), -lo["usd"].to_numpy() / 1e6, width=1.0, color=C_LL)
        ax.bar(sh["k"].to_numpy(), sh["usd"].to_numpy() / 1e6, width=1.0, color=C_BOT)
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_title("NOW: real Bybit liquidations, $M/h (longs down)", fontsize=8, loc="left", color=INK)
    for a in axs.flat[len(eps) + 1:]:
        a.axis("off")
    axs.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Liquidation-wave profile, hours from T. Proxy: 1m bars beyond 3 sigma with 5x volume and >60% taker share on the move side,\n"
                 "in hours of median volume (Binance UM perp). Real prints exist only for NOW (Bybit lake).", fontsize=8.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "liquidation_profiles.png", dpi=130)
    plt.close(fig)


def heatmap(res, cls):
    sims = res["similarity"]["sims"]
    eps = sorted(sims, key=lambda e: -(sims[e]["micro"] or 0))
    cols = list(res["groups"]) + ["micro", "overall"]
    M = np.array([[np.nan if sims[e].get(c) is None else sims[e][c] for c in cols] for e in eps])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and v > 0.6) else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    lab = [f"{e} {res['fingerprints'][e]['T'][:10]} ({'LL' if cls[e] == 'lower' else 'bottom'})" for e in eps]
    ax.set_yticks(range(len(eps)), lab)
    ax.grid(False)
    ax.axvline(len(res["groups"]) - 0.5, color=INK, lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.03, label="similarity to NOW, exp(-d^2/2)")
    ax.set_title("Feature-group similarity to NOW (sorted by microstructure = mean of flow, liquidation, positioning, liquidity)",
                 loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "similarity_heatmap.png", dpi=130)
    plt.close(fig)


def level_chart(res, lvl_table):
    """lvl_table: list of (source, p_down_first, p_up_first, p_neither, p58_given_dn, p68_given_up)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    names = [r[0] for r in lvl_table]
    y = np.arange(len(names))[::-1]
    dn = np.array([r[1] for r in lvl_table])
    up = np.array([r[2] for r in lvl_table])
    ne = np.array([r[3] for r in lvl_table])
    a1.barh(y, dn, color=C_LL, height=0.6, label="63k first")
    a1.barh(y, ne, left=dn, color=GRID, height=0.6, label="neither")
    a1.barh(y, up, left=dn + ne, color=C_BOT, height=0.6, label="66k first")
    for yi, d_, n_, u_ in zip(y, dn, ne, up, strict=True):
        a1.text(d_ / 2, yi, f"{d_:.0%}", ha="center", va="center", color="white", fontsize=8)
        a1.text(d_ + n_ + u_ / 2, yi, f"{u_:.0%}", ha="center", va="center", color="white", fontsize=8)
    a1.set_yticks(y, names, fontsize=8)
    a1.set_xlim(0, 1)
    a1.grid(False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    a1.set_title(f"Which level first? (from {res['P0']:,.0f})", loc="left", fontsize=9, color=INK)
    w = 0.38
    p58 = np.array([np.nan if r[4] is None else r[4] for r in lvl_table])
    p68 = np.array([np.nan if r[5] is None else r[5] for r in lvl_table])
    a2.barh(y + w / 2, p58, height=w, color=C_LL, label="P(58k | lost 63k)")
    a2.barh(y - w / 2, p68, height=w, color=C_BOT, label="P(68k | broke 66k)")
    for yi, a, b in zip(y, p58, p68, strict=True):
        if not np.isnan(a):
            a2.text(a + 0.01, yi + w / 2, f"{a:.0%}", va="center", fontsize=7, color=INK2)
        if not np.isnan(b):
            a2.text(b + 0.01, yi - w / 2, f"{b:.0%}", va="center", fontsize=7, color=INK2)
    a2.set_yticks(y, [""] * len(y))
    a2.set_xlim(0, 1.1)
    a2.grid(False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    a2.set_title("Extension after the first break (14 d)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "level_scenarios.png", dpi=130)
    plt.close(fig)


def make_all(lvl_table) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    res, al, cls = load()
    overlay(al, cls, "lp", "k", "price_aligned_T.png", "log % vs the close at T", "Price on the analog clock (T = analog hour)", 100)
    al2 = al.with_columns((pl.col("low").log() - pl.col("low").filter(pl.col("kL") == 0).first().over("ep").log()).alias("lpl"))
    overlay(al2, cls, "lpl", "kL", "price_aligned_low.png", "log % vs the flush-low hour's low", "Price on the flush-low clock", 100,
            xlim=(-48, 168))
    overlay(al, cls, "cvd", "k", "cvd.png", "CVD since T-72h, hours of median volume", "Cumulative taker delta (Binance UM perp)", ref0=True)
    overlay(al, cls, "loi", "k", "oi.png", "ln(OI / OI at T), %", "Open interest (Binance UM, BTC units)", 100)
    overlay(al, cls, "fund_bp", "k", "funding.png", "funding per 8h, bp", "Funding (Binance UM BTCUSDT)", step=True)
    overlay(al, cls, "basis", "k", "basis.png", "perp vs spot, bp (hourly mean)", "Spot-perp basis (Binance)")
    overlay(al, cls, "dvol", "k", "dvol.png", "DVOL", "Deribit DVOL", ref0=False)
    liq_profiles(res, al, cls)
    heatmap(res, cls)
    level_chart(res, lvl_table)
