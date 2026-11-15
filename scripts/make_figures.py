"""Render the research figures (PNG) and research/figures/INDEX.md.

Every figure states one finding in its title, reads its numbers from the
research reports (research/reports/*.json) or research data (data/research/**),
and notes source and segment (llm_in_sample = before research.knowledge_cutoff
2026-07-01, clean_holdout = after) in a caption. Run:

    PYTHONIOENCODING=utf-8 uv run --extra viz --extra ml python scripts/make_figures.py
    ... --only vol            # figures whose topic/name contains 'vol' (INDEX.md untouched)
    ... --list                # list registered figures

Design rules: one message per figure, one y-axis per panel, a fixed
colour-blind-safe categorical palette (tab10 order), single-hue sequential and
RdBu_r diverging colormaps centred on 0, CIs wherever the source has them,
units on axes, 150 dpi, white background.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "research" / "reports"
DATA = ROOT / "data"
OUT = ROOT / "research" / "figures"
CUTOFF = "2026-07-01"

# fixed categorical order (tab10); never re-ordered by rank
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
SEG = ["llm_in_sample", "clean_holdout"]
SEG_COLOR = {"llm_in_sample": C[0], "clean_holdout": C[1], "pooled": C[7]}
SEG_LABEL = {
    "llm_in_sample": "llm_in_sample (before 2026-07-01)",
    "clean_holdout": "clean_holdout (after 2026-07-01)",
    "pooled": "pooled",
}
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

FIGURES: list[dict] = []


def figure(topic: str, name: str, finding: str, source: str):
    """Register a figure function. `fn()` returns (fig, caption). The finding is the title."""

    def deco(fn):
        FIGURES.append({"fn": fn, "topic": topic, "name": name, "finding": finding, "source": source})
        return fn

    return deco


def rj(rel: str):
    """Load a report JSON (path relative to research/reports/, or absolute/ROOT-relative)."""
    p = Path(rel)
    if not p.is_absolute():
        p = REPORTS / rel if (REPORTS / rel).exists() else ROOT / rel
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def num(x, default=np.nan):
    """Float or NaN for None/str/non-finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def ci_err(est, lo, hi):
    """Asymmetric yerr array for errorbar from point estimates and CI bounds."""
    est, lo, hi = (np.asarray(a, float) for a in (est, lo, hi))
    return np.vstack([np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)])


def annotate_bars(ax, bars, fmt="{:.2f}", inside=False, fontsize=10, color="#222", offset=3):
    """Write each bar's value at its end (handles vertical bars, negatives included)."""
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        x = b.get_x() + b.get_width() / 2
        va = "bottom" if h >= 0 else "top"
        ax.annotate(fmt.format(h), (x, h), xytext=(0, offset if h >= 0 else -offset),
                    textcoords="offset points", ha="center", va=va, fontsize=fontsize, color=color)


def annotate_barh(ax, bars, fmt="{:.2f}", fontsize=10, color="#222", offset=3):
    for b in bars:
        w = b.get_width()
        if not np.isfinite(w):
            continue
        y = b.get_y() + b.get_height() / 2
        ax.annotate(fmt.format(w), (w, y), xytext=(offset if w >= 0 else -offset, 0),
                    textcoords="offset points", ha="left" if w >= 0 else "right", va="center",
                    fontsize=fontsize, color=color)


def heatmap(ax, M, xlabels, ylabels, cmap=SEQ_CMAP, center=None, fmt="{:.2f}", cbar_label="",
            vmin=None, vmax=None, annot=True, fontsize=9.5, mask=None):
    """Annotated heatmap; center=0 gives a diverging TwoSlopeNorm with a neutral midpoint."""
    M = np.asarray(M, float)
    if mask is not None:
        M = np.where(mask, np.nan, M)
    finite = M[np.isfinite(M)]
    if center is not None:
        span = vmax if vmax is not None else (np.nanmax(np.abs(finite - center)) if finite.size else 1.0)
        span = span if span > 0 else 1.0
        norm = TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annot:
        cm = im.get_cmap()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = cm(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                        color="white" if lum < 0.5 else "#111")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    return im


def caption(source: str, segment: str, extra: str = "") -> str:
    s = f"Source: {source}  |  Segment: {segment}"
    return s + (f"  |  {extra}" if extra else "")


def _finish(fig, title: str, cap: str):
    # fig.text (not suptitle) so tight_layout's rect alone reserves the title band
    import textwrap

    w, h = fig.get_figwidth(), fig.get_figheight()
    width = max(40, int(w * 0.97 / 0.100))  # ~chars per line at 12.5 pt bold
    title = "\n".join(textwrap.fill(line, width) for line in title.split("\n"))
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.006, cap, ha="left", va="bottom", fontsize=10, color="#555555", wrap=True)
    n_title = title.count("\n") + 1
    n_cap = sum(max(1, math.ceil(len(line) * 0.078 / (w * 0.97))) for line in cap.split("\n"))
    top = 1 - (0.12 + 0.24 * n_title) / h
    bottom = (0.08 + 0.19 * n_cap) / h
    if getattr(fig, "_manual_layout", False):
        fig.subplots_adjust(top=top - 0.06, bottom=bottom + 0.1)
    else:
        fig.tight_layout(rect=(0, bottom, 1, top))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="substring of topic/name; INDEX.md is not rewritten")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for f in FIGURES:
            print(f"{f['topic']}/{f['name']}.png  {f['finding']}")
        return 0
    todo = [f for f in FIGURES if a.only is None or a.only in f"{f['topic']}/{f['name']}"]
    done, failed = [], []
    for f in todo:
        path = OUT / f["topic"] / f"{f['name']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig, cap = f["fn"]()
            _finish(fig, f["finding"], cap)
            fig.savefig(path, dpi=150, facecolor="white")
            plt.close(fig)
            done.append(f)
            print(f"ok   {path.relative_to(ROOT)}")
        except Exception as e:  # report and continue: one bad source must not kill the set
            plt.close("all")
            failed.append((f, repr(e)))
            print(f"FAIL {f['topic']}/{f['name']}: {e!r}")
            traceback.print_exc(limit=3)
    if a.only is None:
        lines = [
            "# Research figures",
            "",
            f"_Generated by `scripts/make_figures.py` ({len(done)} figures). Do not edit by hand; rerun the script. "
            "Segments: llm_in_sample = before research.knowledge_cutoff (2026-07-01), clean_holdout = after._",
            "",
            "| figure | finding | source |",
            "|---|---|---|",
        ]
        for f in done:
            rel = f"{f['topic']}/{f['name']}.png"
            lines.append(f"| [{rel}]({rel}) | {f['finding'].replace(chr(10), ' ')} | {f['source']} |")
        if failed:
            lines += ["", "Not rendered this run:", ""]
            lines += [f"- {f['topic']}/{f['name']}: {err}" for f, err in failed]
        (OUT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {(OUT / 'INDEX.md').relative_to(ROOT)}")
    print(f"{len(done)} ok, {len(failed)} failed")
    return 1 if failed else 0


# ======================================================================= figures


# ---------------------------------------------------------------- volatility I-III

_VT = "vol_tournament_2026-11-10.json"
_CO_MODEL_LABEL = {
    "ewma": "EWMA (single lambda)", "robust_bc": "robust_bc x 24", "garch": "GARCH-t", "gjr": "GJR-GARCH",
    "egarch": "EGARCH", "sv_rv": "stoch. vol (RV)", "har": "HAR (reference)", "harq": "HAR-Q", "shar": "SHAR",
    "harj": "HAR-J", "har_gk": "HAR on GK range", "loghar": "log-HAR", "p_har": "panel HAR",
    "p_harq": "panel HAR-Q", "p_loghar": "panel log-HAR", "p_loghar_f": "panel log-HAR + factor",
}
_CO_MODELS = ["ewma", "robust_bc", "garch", "gjr", "egarch", "sv_rv", "har_gk", "har", "harq", "shar", "harj",
              "p_har", "p_harq", "loghar", "p_loghar", "p_loghar_f"]


def _co_lb_ratio(vt, key):
    lb = vt["leaderboards"].get(key)
    if lb is None:
        return {}, set(), None
    return {r["model"]: r["ratio_vs_ref"] for r in lb["rows"]}, set(lb.get("mcs", [])), lb


def _co_tournament(universe, title_unit):
    vt = rj(_VT)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    y = np.arange(len(_CO_MODELS))
    h = 0.38
    info = []
    for i, seg in enumerate(SEG):
        ratio, mcs, lb = _co_lb_ratio(vt, f"{universe}|h1|{seg}|qlike|all")
        vals = np.array([ratio.get(m, np.nan) for m in _CO_MODELS])
        bars = ax.barh(y + (i - 0.5) * h, vals - 1, left=1, height=h, color=SEG_COLOR[seg],
                       label=f"{SEG_LABEL[seg]}" + (f": {lb['n_dates']} dates, {lb['n_symbols']} symbols" if lb else ""))
        for m, b, v in zip(_CO_MODELS, bars, vals):
            if not np.isfinite(v):
                continue
            star = " *" if m in mcs else ""
            ax.annotate(f"{v:.2f}{star}", (v, b.get_y() + b.get_height() / 2), xytext=(3 if v >= 1 else -3, 0),
                        textcoords="offset points", va="center", ha="left" if v >= 1 else "right", fontsize=9.5)
        info.append(lb)
    ax.axvline(1, color="#333", lw=1)
    ax.set_yticks(y, [_CO_MODEL_LABEL.get(m, m) for m in _CO_MODELS])
    ax.invert_yaxis()
    ax.set_xlabel(f"QLIKE relative to HAR, next-day {title_unit} variance (< 1 = better)")
    lo = np.nanmin([b.get_width() + 1 for b in ax.patches])
    hi = np.nanmax([b.get_width() + 1 for b in ax.patches])
    ax.set_xlim(min(0.8, lo - 0.08), hi + 0.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return fig, caption(f"research/reports/{_VT} (leaderboards {universe}|h1|*|qlike|all)",
                        "llm_in_sample vs clean_holdout", "* = in the 90% model confidence set")


@figure("vol", "tournament_crypto_daily",
        "Daily crypto vol: log-HAR beats HAR by 12% in-sample and 4% in the holdout;\nGARCH, EWMA and range estimators lose",
        "vol_tournament_2026-11-10")
def _co_fig_tournament_crypto():
    return _co_tournament("L", "crypto (23 perps)")


@figure("vol", "tournament_tradfi_daily",
        "TradFi perps: pooled panel HARs forecast best; robust_bc x 24 is ~1.5x worse than HAR in the holdout",
        "vol_tournament_2026-11-10")
def _co_fig_tournament_tradfi():
    return _co_tournament("T", "TradFi (15 perps)")


@figure("vol", "weekday_rv_ratio",
        "Weekends are quiet: crypto Saturdays run ~0.5x Tue-Fri realised vol, TradFi perps ~0.05-0.1x",
        "vol_tournament_2026-11-10")
def _co_fig_weekday():
    w = rj(_VT)["weekday_breakdown_1d"]
    days = ["Sat", "Sun", "Mon", "Tue-Fri"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (u, name) in zip(axes, [("L", "Crypto (23 perps)"), ("T", "TradFi perps (15)")]):
        x = np.arange(len(days))
        for i, seg in enumerate(SEG):
            d = w[f"{u}|{seg}"]
            vals = [d[k]["median_rv_ratio_to_tue_fri"] for k in days]
            ns = [d[k]["n"] for k in days]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=SEG_COLOR[seg],
                          label=f"{SEG_LABEL[seg]} (n Sat = {ns[0]})")
            annotate_bars(ax, bars, "{:.2f}", fontsize=9.5)
        ax.axhline(1, color="#333", lw=1)
        ax.set_xticks(x, ["Saturday", "Sunday", "Monday", "Tue-Fri"])
        ax.set_title(name)
        ax.set_ylim(0, 1.3)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median daily RV / median Tue-Fri RV (x)")
    axes[1].legend(loc="upper left")
    return fig, caption(f"research/reports/{_VT} (weekday_breakdown_1d, forecast-day RV from 1h bars)",
                        "llm_in_sample vs clean_holdout", "TradFi perps listed Mar-Jun 2026: short in-sample")


@figure("vol", "weekday_forecast_bias",
        "Without a calendar term, log-HAR over-forecasts Saturday vol ~2x relative to Tue-Fri (crypto)",
        "vol_tournament_2026-11-10")
def _co_fig_weekday_bias():
    w = rj(_VT)["weekday_breakdown_1d"]
    days = ["Sat", "Sun", "Mon", "Tue-Fri"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (u, name) in zip(axes, [("L", "Crypto (23 perps)"), ("T", "TradFi perps (15)")]):
        x = np.arange(len(days))
        for i, seg in enumerate(SEG):
            d = w[f"{u}|{seg}"]
            ref = d["Tue-Fri"]["loghar"]["mean_log_F_over_RV"]
            vals = [d[k]["loghar"]["mean_log_F_over_RV"] - ref for k in days]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=SEG_COLOR[seg], label=SEG_LABEL[seg])
            for b, v in zip(bars, vals):
                ax.annotate(f"{np.exp(v):.1f}x", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -3),
                            textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(x, ["Saturday", "Sunday", "Monday", "Tue-Fri"])
        ax.set_title(name)
        ax.grid(axis="x", visible=False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.5), hi * 1.12)
    axes[0].set_ylabel("mean log(forecast / RV), minus Tue-Fri")
    axes[0].legend(loc="upper right")
    return fig, caption(f"research/reports/{_VT} (weekday_breakdown_1d, model loghar)",
                        "llm_in_sample vs clean_holdout", "labels = exp(bias): forecast / realised, relative to Tue-Fri")


# hour of week ------------------------------------------------------------
_HOW_CACHE: dict = {}


def _co_how(group):
    """(weeks x 168) matrix of vol-standardised squared 1h log returns, mean over symbols; week start dates."""
    if group in _HOW_CACHE:
        return _HOW_CACHE[group]
    from datetime import datetime, timezone
    import polars as pl
    from ai_trader.storage.lake import Lake

    vt = rj(_VT)
    syms = vt["universe_L"] if group == "L" else vt["universe_T"]
    as_of = datetime.fromisoformat(vt["as_of"].replace("Z", "+00:00")) if isinstance(vt.get("as_of"), str) else \
        datetime(2026, 11, 10, tzinfo=timezone.utc)
    lk = Lake()
    parts = []
    used = []
    for s in syms:
        b = lk.bars(s, "1h", as_of)
        if b.height < 24 * 60:
            continue
        b = b.sort("event_ts")
        ts = b["event_ts"].dt.epoch("s").to_numpy()
        c = b["close"].to_numpy().astype(float)
        r = np.diff(np.log(c))
        ok = np.diff(ts) == 3600
        t1 = ts[1:]
        r2 = np.where(ok, r * r, np.nan)
        n = 720  # trailing 30 days, known before the bar
        filled = np.nan_to_num(r2)
        cnt = np.concatenate([[0], np.cumsum(np.isfinite(r2))])
        cs = np.concatenate([[0], np.cumsum(filled)])
        base = np.full(r2.size, np.nan)
        num_ = cs[n:-1] - cs[:-n - 1]
        den = cnt[n:-1] - cnt[:-n - 1]
        base[n:] = np.where(den > n // 2, num_ / np.maximum(den, 1), np.nan)
        u = np.clip(r2 / base, 0, 30)
        keep = np.isfinite(u) & (base > 0)
        days = t1 // 86400
        how = ((days + 3) % 7) * 24 + (t1 % 86400) // 3600  # Monday 00 UTC = 0
        week = (days + 3) // 7
        parts.append(pl.DataFrame({"week": week[keep], "how": how[keep], "u": u[keep]}))
        used.append(s)
    df = pl.concat(parts).group_by(["week", "how"]).agg(pl.col("u").mean())
    weeks = np.sort(df["week"].unique().to_numpy())
    M = np.full((weeks.size, 168), np.nan)
    wi = np.searchsorted(weeks, df["week"].to_numpy())
    M[wi, df["how"].to_numpy()] = df["u"].to_numpy()
    full = np.isfinite(M).sum(1) >= 160
    M, weeks = M[full], weeks[full]
    start = weeks * 7 - 3  # Monday, days since epoch
    _HOW_CACHE[group] = (M, start, used)
    return _HOW_CACHE[group]


def _co_profile(M, rng=None, n_boot=0):
    p = np.nanmean(M, 0)
    p = p / np.nanmean(p)
    if not n_boot:
        return p, None, None
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, M.shape[0], M.shape[0])
        q = np.nanmean(M[idx], 0)
        bs.append(q / np.nanmean(q))
    bs = np.array(bs)
    return p, np.nanpercentile(bs, 5, 0), np.nanpercentile(bs, 95, 0)


_CUT_DAY = 20635  # 2026-07-01 in days since epoch


@figure("vol", "hour_of_week_crypto",
        "Crypto hour-of-week vol shape is large and stable: peaks at the Sunday-night reopen and the\nUS open (13-15 UTC), troughs on weekend mornings",
        "new_strategies_2026-11-11 (H-NS4); recomputed from lake 1h bars")
def _co_fig_how_crypto():
    M, start, used = _co_how("L")
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    x = np.arange(168)
    clip = np.nan
    for seg in SEG:
        sel = start < _CUT_DAY if seg == "llm_in_sample" else start >= _CUT_DAY
        p, lo, hi = _co_profile(M[sel], rng, 300)
        ins = seg == "llm_in_sample"
        ax.fill_between(x, lo, hi, color=SEG_COLOR[seg], alpha=0.3 if ins else 0.12, lw=0, zorder=3 if ins else 1)
        ax.plot(x, p, color=SEG_COLOR[seg], lw=2 if ins else 1.5, zorder=4 if ins else 2,
                label=f"{SEG_LABEL[seg]}: {int(sel.sum())} weeks (90% band)")
        if seg == "llm_in_sample":
            pk, tr = int(np.nanargmax(p)), int(np.nanargmin(p))
            dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            ax.annotate(f"peak {p[pk]:.1f}x ({dn[pk // 24]} {pk % 24:02d} UTC)", (pk, p[pk]), xytext=(10, 4),
                        textcoords="offset points", fontsize=10)
            ax.annotate(f"trough {p[tr]:.2f}x ({dn[tr // 24]} {tr % 24:02d} UTC)", (tr, p[tr]), xytext=(-150, 60),
                        textcoords="offset points", fontsize=10, arrowprops={"arrowstyle": "-", "color": "#555", "lw": 0.8})
        else:
            clip = float(np.nanmax(p))
    ax.axhline(1, color="#333", lw=1)
    ax.set_ylim(0, 3.8)
    for d in range(1, 7):
        ax.axvline(24 * d, color="#999", lw=0.6)
    ax.set_xticks(np.arange(12, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(np.arange(0, 168, 6), minor=True)
    ax.set_xlim(0, 167)
    ax.set_xlabel("hour of week (UTC; minor ticks every 6 h)")
    ax.set_ylabel("variance factor vs weekly mean (x)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    ax.grid(axis="x", visible=False)
    return fig, caption(f"lake 1h bars, {len(used)} crypto perps; squared log return / trailing-30d mean, clipped at 30x",
                        "llm_in_sample vs clean_holdout",
                        f"90% week-bootstrap bands; holdout max {clip:.1f}x is off scale; H-NS4 QLIKE gain -0.040 / -0.076")


@figure("vol", "hour_of_week_heatmap",
        "TradFi perps have a far deeper hour-of-week shape than crypto: weekends ~0.1x, US cash open ~3x",
        "new_strategies_2026-11-11 (H-NS4); recomputed from lake 1h bars")
def _co_fig_how_heat():
    from matplotlib.colors import TwoSlopeNorm
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.035]})
    axes, cax = axs[:2], axs[2]
    res = {}
    for g in ["L", "T"]:
        M, start, used = _co_how(g)
        p, _, _ = _co_profile(M)
        res[g] = (np.log2(np.clip(p, 1 / 64, None)).reshape(7, 24), len(used), M.shape[0])
    span = max(np.nanmax(np.abs(v[0])) for v in res.values())
    norm = TwoSlopeNorm(vcenter=0, vmin=-span, vmax=span)
    for ax, (g, name) in zip(axes, [("L", "Crypto"), ("T", "TradFi perps")]):
        Z, ns, nw = res[g]
        im = ax.imshow(Z, cmap=DIV_CMAP, norm=norm, aspect="auto")
        ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xticks(range(0, 24, 3), [f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlabel("hour (UTC)")
        ax.set_title(f"{name}: {ns} symbols, {nw} weeks")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.text(j, i, f"{2 ** Z[i, j]:.1f}x", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        i, j = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.text(j, i, f"{2 ** Z[i, j]:.2f}x", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    cb = fig.colorbar(im, cax=cax)
    ticks = [t for t in range(-6, 7) if abs(t) <= span]
    cb.set_ticks(ticks, labels=[f"{2.0 ** t:g}x" for t in ticks])
    cb.set_label("variance factor vs weekly mean (log scale)")
    cb.outline.set_visible(False)
    return fig, caption("lake 1h bars; squared log return / trailing-30d mean, clipped at 30x",
                        "all data to 2026-11-10 (crypto mostly llm_in_sample; TradFi listed Mar-Jun 2026)")


@figure("vol", "hour_of_week_qlike_gain",
        "Multiplying robust sigma^2 by an hour-of-week factor cuts next-hour QLIKE in both segments,\n7x more for TradFi",
        "new_strategies_2026-11-11 (H-NS4)")
def _co_fig_how_gain():
    m = rj("new_strategies_2026-11-11.json")["H-NS4"]["main"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups = [("L", "Crypto (23)"), ("T", "TradFi perps (15)")]
    x = np.arange(len(groups))
    for i, seg in enumerate(SEG):
        est = np.array([m[g][seg]["mean"] for g, _ in groups])
        se = np.abs(est / np.array([m[g][seg]["t"] for g, _ in groups]))
        bars = ax.bar(x + (i - 0.5) * 0.38, est, 0.38, color=SEG_COLOR[seg], yerr=1.96 * se, capsize=4,
                      label=SEG_LABEL[seg], error_kw={"lw": 1.2})
        for b, v, s, g in zip(bars, est, se, groups):
            ax.annotate(f"{v:+.3f}\n(t {v / s:.1f})", (b.get_x() + b.get_width() / 2, v - 1.96 * s),
                        xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=9.5)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, [g[1] for g in groups])
    ax.set_ylabel("QLIKE change vs robust sigma^2 (< 0 = better)")
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo * 1.35, 0.05)
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    return fig, caption("research/reports/new_strategies_2026-11-11.json (H-NS4 main; date-clustered t)",
                        "llm_in_sample vs clean_holdout", "error bars: 95% CI")


# leverage ------------------------------------------------------------------
_LEV = "leverage_2026-11-10.json"


@figure("vol", "leverage_semivariance",
        "No leverage effect: a down day's semivariance raises next-day vol no more than an up day's",
        "leverage_2026-11-10 (H-LEV1, H-LEV2)")
def _co_fig_leverage():
    L = rj(_LEV)["segments"]
    rows = [("H-LEV2", "llm_in_sample", "Crypto\nllm_in_sample"), ("H-LEV2", "clean_holdout", "Crypto\nclean_holdout"),
            ("H-LEV1", "clean_holdout", "TradFi equity\nclean_holdout")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(rows))
    series = [("rs_neg_s", "b(RS-): down-move semivariance", C[3]), ("rs_pos_s", "b(RS+): up-move semivariance", C[2])]
    for i, (k, lab, col) in enumerate(series):
        est, se = [], []
        for h, seg, _ in rows:
            cf = L[seg]["results"][h][1]["shar"]["primary_level"]["coefs"][k]
            est.append(cf["b"])
            se.append(cf["se"])
        est, se = np.array(est), np.array(se)
        ax.errorbar(x + (i - 0.5) * 0.25, est, yerr=1.96 * se, fmt="o", ms=7, color=col, capsize=4, lw=1.6, label=lab)
    for j, (h, seg, _) in enumerate(rows):
        pl_ = L[seg]["results"][h][1]["shar"]["primary_level"]
        ax.annotate(f"diff {pl_['diff']:+.2f}\nt {pl_['diff_t']:.2f}\n{pl_['n_dates']} dates", (j, 0.0),
                    xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=9.5)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, [r[2] for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.25), hi)
    ax.set_ylabel("SHAR coefficient on next-day RV (vol-standardised)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig, caption(f"research/reports/{_LEV} (SHAR primary_level; diff = b(RS-) - b(RS+), Driscoll-Kraay)",
                        "llm_in_sample vs clean_holdout", "bars: 95% CI")


@figure("vol", "leverage_gjr_assets",
        "None of 23 crypto GJR-GARCH asymmetry terms is significantly positive (median gamma 0.004)",
        "leverage_2026-11-10 (H-LEV2 gjr_assets)")
def _co_fig_gjr():
    r = rj(_LEV)["segments"]["llm_in_sample"]["results"]["H-LEV2"][1]
    a = [(s, v["gjr"]) for s, v in r["gjr_assets"].items() if v.get("status") == "ok" and "gjr" in v]
    a.sort(key=lambda t: t[1]["gamma"])
    g = np.array([v["gamma"] for _, v in a])
    se = np.array([v["se"] for _, v in a])
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(a))
    sig = np.abs(g / se) > 1.96
    ax.errorbar(g[~sig], y[~sig], xerr=1.96 * se[~sig], fmt="o", color=C[0], capsize=3, lw=1.4,
                label="not significant (|t| < 1.96)")
    if sig.any():
        ax.errorbar(g[sig], y[sig], xerr=1.96 * se[sig], fmt="s", color=C[3], capsize=3, lw=1.4,
                    label="significant (both negative: up-moves add more vol)")
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y, [s.replace("USDT", "") for s, _ in a])
    ax.set_xlabel("GJR gamma (extra ARCH loading after a down day), 95% CI")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.98, f"median gamma {r['gjr_summary']['median_gjr_gamma']:.3f}\n"
                        f"t > 2: {r['gjr_summary']['n_gamma_t_gt2']} of {r['gjr_summary']['n_assets_fit']}",
            transform=ax.transAxes, va="top", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    return fig, caption(f"research/reports/{_LEV} (GJR-GARCH-t per asset, daily)", "llm_in_sample (years per asset)")


@figure("vol", "post_spike_overforecast",
        "After vol spikes forecasts run hot: in the high-vol state next-week RV lands further below the EWMA",
        "dow_regime_2026-11-08 (H-RL2)")
def _co_fig_rl2():
    r = rj("dow_regime_2026-11-08.json")["regime"]["RL2"]
    states = ["low", "normal", "high"]
    groups = [("crypto", "Crypto"), ("tradfi", "TradFi / macro")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (g, name) in zip(axes, groups):
        x = np.arange(3)
        for i, seg in enumerate(SEG):
            d = r[seg].get(g)
            if not d:
                continue
            vals = [d["mean_log_rv_ratio"][s] for s in states]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=SEG_COLOR[seg],
                          label=f"{SEG_LABEL[seg]}: high-low {d['high_minus_low']:+.2f} (t {d['t']:.1f})")
            annotate_bars(ax, bars, "{:.2f}", fontsize=9.5)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(x, ["low vol", "normal", "high vol"])
        ax.set_title(name)
        ax.set_xlabel("v0 vol state at forecast time")
        ax.legend(loc="lower left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(top=0.06)
    axes[0].set_ylabel("mean log(next-week RV / EWMA forecast)")
    lo = min(a.get_ylim()[0] for a in axes)
    axes[0].set_ylim(lo * 1.45, 0.06)
    return fig, caption("research/reports/dow_regime_2026-11-08.json (regime.RL2)", "llm_in_sample vs clean_holdout")


# --------------------------------------------------------------- strategies / regimes
_SM = "strategy_map_2026-11-11.json"
_CELLS = ["R1 quiet\ntrend up", "R2 trend\nup", "R3 momentum\nburst", "R4 com-\npression", "R5 balanced\nrange",
          "R6 whip-\nsaw", "R7 grind\ndown", "R8 trend\ndown", "R9 liqui-\ndation"]


@figure("strategies", "family_regime_map",
        "Regime cells say where families lose (mean reversion in trend cells), not who wins: holdout signs shift",
        "strategy_map_2026-11-11 (H-RC1, context_family)")
def _co_fig_family_map():
    d = rj(_SM)
    rows = [r for r in d["context_family"] if r["universe"] == "L"]
    fams = ["trend", "mean_reversion", "shock", "volatility", "positioning", "calendar"]
    cells = [f"R{i}" for i in range(1, 10)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2))
    for ax, (pre, name) in zip(axes, [("is", "llm_in_sample"), ("ho", "clean_holdout")]):
        M = np.full((len(fams), 9), np.nan)
        G = [[""] * 9 for _ in fams]
        for r in rows:
            i, j = fams.index(r["name"]), cells.index(r["cell"])
            M[i, j] = num(r[f"{pre}_shrunk"]) * 100
            G[i][j] = r["grade"]
        heatmap(ax, M, _CELLS if pre == "ho" else [""] * 9, [f.replace("_", " ") for f in fams], cmap=DIV_CMAP,
                center=0, fmt="{:+.1f}", cbar_label="% of daily sigma per\nactive symbol-day", fontsize=9.5)
        if pre == "is":
            for i in range(len(fams)):
                for j in range(9):
                    if G[i][j] in ("A", "B"):
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#111", lw=2))
        ax.set_title(f"{SEG_LABEL[name]}" + ("  (boxed: grade A/B, same sign and t >= 2 / >= 1)" if pre == "is" else ""),
                     loc="left", fontsize=10.5)
    return fig, caption(f"research/reports/{_SM} (context_family, crypto L; shrunk net mean, 7.5 bp/side)",
                        "llm_in_sample vs clean_holdout",
                        f"rank persistence across segments: Spearman {d['H-RC1']['spearman']:.3f}, "
                        f"95% CI [{d['H-RC1']['ci95'][0]:.2f}, {d['H-RC1']['ci95'][1]:.2f}]")


@figure("strategies", "regime_rank_persistence",
        "Per-cell strategy rankings do not persist into the holdout (Spearman 0.085, CI includes 0)",
        "strategy_map_2026-11-11 (H-RC1)")
def _co_fig_rc1():
    d = rj(_SM)["H-RC1"]
    p = d["pairs"]
    strategies = sorted({q["strategy"] for q in p})
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    x = np.array([q["is_shrunk"] for q in p]) * 100
    y = np.array([q["ho_shrunk"] for q in p]) * 100
    ax.scatter(x, y, s=22, color=C[0], alpha=0.7, edgecolor="white", lw=0.4,
               label=f"strategy x regime cell ({len(p)} pairs, {len(strategies)} strategies)")
    ax.axhline(0, color="#333", lw=0.9)
    ax.axvline(0, color="#333", lw=0.9)
    b = np.polyfit(x, y, 1)
    xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 50)
    ax.plot(xs, np.polyval(b, xs), color=C[1], lw=1.8, label=f"OLS fit, slope {b[0]:.2f}")
    ax.set_xlabel("in-sample shrunk mean (% of daily sigma per symbol-day)")
    ax.set_ylabel("holdout shrunk mean (% of daily sigma)")
    ax.text(0.02, 0.98, f"Spearman {d['spearman']:.3f}\n95% CI [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]\n"
                        f"sign agreement {d['sign_agreement']:.0%}", transform=ax.transAxes, va="top", fontsize=10.5)
    ax.legend(loc="lower right")
    return fig, caption(f"research/reports/{_SM} (H-RC1 pairs; episode bootstrap CI)",
                        "llm_in_sample (x) vs clean_holdout (y)")


@figure("strategies", "regime_conditional_book",
        "Dropping in-sample-losing cells beat equal weight in the holdout, but the book stayed negative",
        "strategy_map_2026-11-11 (H-RC2)")
def _co_fig_rc2():
    d = rj(_SM)["H-RC2"]
    books = [("eq", "equal weight, all strategies"), ("un", "unconditional winners"), ("co", "cell-conditional sets")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(SEG))
    for i, (k, lab) in enumerate(books):
        vals = [d[s]["sharpe"][k] for s in SEG]
        bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26, color=C[[7, 0, 2][i]], label=lab)
        annotate_bars(ax, bars, "{:+.2f}", fontsize=9.5)
    for j, s in enumerate(SEG):
        lo, hi = d[s]["ci95_vs_eq"]
        ax.text(j, 0.97, f"conditional - equal weight: {d[s]['diff_vs_eq']:+.2f}\n95% CI [{lo:.2f}, {hi:.2f}]",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, [f"{SEG_LABEL[s]}\n{d[s]['n_days']} days" for s in SEG])
    ax.set_ylabel("annualised Sharpe (net of 7.5 bp/side)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.3, hi + 0.9)
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    return fig, caption(f"research/reports/{_SM} (H-RC2)", "llm_in_sample vs clean_holdout")


@figure("strategies", "regime_durations",
        "Regimes are persistent: median episode 5-7 days, daily stay probability 0.84-0.89",
        "dow_regime_2026-11-08 (H-RL3)")
def _co_fig_durations():
    de = rj("dow_regime_2026-11-08.json")["regime"]["desc"]
    cells = [f"R{i}" for i in range(1, 10)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    med = [de["duration_by_cell"][c]["median"] for c in cells]
    mean = [de["duration_by_cell"][c]["mean"] for c in cells]
    x = np.arange(9)
    b1 = axes[0].bar(x - 0.19, med, 0.38, color=C[0], label="median")
    axes[0].bar(x + 0.19, mean, 0.38, color=C[9], label="mean")
    annotate_bars(axes[0], b1, "{:.0f}", fontsize=9.5)
    axes[0].set_ylabel("episode length (days)")
    axes[0].legend(loc="upper right", ncols=2)
    axes[0].set_ylim(0, max(mean) * 1.25)
    st = [de["stay_prob_daily"][c] for c in cells]
    b2 = axes[1].bar(x, st, 0.6, color=C[0])
    annotate_bars(axes[1], b2, "{:.2f}", fontsize=9.5)
    axes[1].set_ylim(0.7, 0.95)
    axes[1].set_ylabel("P(same cell tomorrow)")
    for ax in axes:
        ax.set_xticks(x, [c.replace("\n", " ").replace("- ", "") for c in _CELLS], rotation=40, ha="right")
        ax.grid(axis="x", visible=False)
    axes[0].set_title("Episode length by cell")
    axes[1].set_title("Daily stay probability")
    n = sum(de["duration_by_cell"][c]["n_episodes"] for c in cells)
    return fig, caption("research/reports/dow_regime_2026-11-08.json (regime.desc)",
                        "all history (llm_in_sample dominated)", f"{n} episodes across assets")


@figure("strategies", "selection_overfit_pbo",
        "Choosing the best variant per symbol is a coin flip: PBO median 0.44 (trend zoo), 0.65 (breakouts)",
        "synthetic_robustness_2026-11-11 (H-SYN1, H-SYN2)")
def _co_fig_pbo():
    s = rj("synthetic_robustness_2026-11-11.json")["overfit"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    rng = np.random.default_rng(3)
    for i, (k, lab) in enumerate([("zoo", "21-variant trend zoo"), ("bo", "breakout grid")]):
        v = np.array([num(s["per_symbol"][sym][k]["pbo"]) for sym in s["per_symbol"]])
        v = v[np.isfinite(v)]
        xj = i + rng.uniform(-0.12, 0.12, v.size)
        ax.scatter(xj, v, s=34, color=C[i], alpha=0.8, edgecolor="white", lw=0.5, label=f"{lab} ({v.size} symbols)")
        m = np.median(v)
        ax.hlines(m, i - 0.25, i + 0.25, color="#111", lw=2)
        ax.annotate(f"median {m:.2f}\n{(v > 0.5).mean():.0%} of symbols > 0.5", (i + 0.27, m), va="center", fontsize=10)
    ax.axhline(0.5, color=C[3], lw=1.2, ls="--", label="0.5 = selection no better than chance")
    ax.set_xticks([0, 1], ["trend zoo", "breakouts"])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability of backtest overfitting (CSCV, S = 16)")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    return fig, caption("research/reports/synthetic_robustness_2026-11-11.json (overfit.per_symbol)",
                        "llm_in_sample history per symbol")


# --------------------------------------------------------------- PnL decomposition
_PD = "pnl_decomposition_2026-11-11.json"


@figure("pnl", "effective_bets",
        "The 16-sleeve book is roughly one bet: ~8 eigen-bets, but 1.2 effective bets equal-weighted (2.5 holdout)",
        "pnl_decomposition_2026-11-11 (H-PD1)")
def _co_fig_enb():
    d = rj(_PD)["H-PD1"]
    keys = [("llm_in_sample|L", "crypto sleeves\nllm_in_sample"), ("clean_holdout|L", "crypto sleeves\nclean_holdout"),
            ("clean_holdout|LT", "crypto + TradFi\nclean_holdout")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(keys))
    enb = np.array([d[k]["enb"] for k, _ in keys])
    lo = np.array([d[k]["ci95_enb"][0] for k, _ in keys])
    hi = np.array([d[k]["ci95_enb"][1] for k, _ in keys])
    ns = np.array([d[k]["n_sleeves"] for k, _ in keys])
    b0 = ax.bar(x - 0.27, ns, 0.27, color=C[7], label="number of sleeves")
    b1 = ax.bar(x, enb, 0.27, color=C[0], yerr=ci_err(enb, lo, hi), capsize=4,
                label="eigen-bets of the correlation matrix (95% CI)")
    b2 = ax.bar(x + 0.27, [d[k]["enb_meucci_ew_book"] for k, _ in keys], 0.27, color=C[1],
                label="effective bets in the equal-weight book (Meucci)")
    for bars in (b0, b2):
        annotate_bars(ax, bars, "{:.1f}", fontsize=9.5)
    for b, v, h_ in zip(b1, enb, hi):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, h_), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9.5)
    ax.set_xticks(x, [f"{lab}\n({d[k]['n_days']} days)" for k, lab in keys])
    ax.set_ylabel("number of bets")
    ax.set_ylim(0, max(ns) * 1.3)
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig, caption(f"research/reports/{_PD} (H-PD1)", "llm_in_sample vs clean_holdout")


@figure("pnl", "factor_loadings",
        "Trend sleeves load on the crypto TSMOM factor; mean reversion loads short it; little alpha is left",
        "pnl_decomposition_2026-11-11 (H-PD2)")
def _co_fig_loadings():
    d = rj(_PD)["H-PD2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    seg_keys = [("llm_in_sample", "llm_in_sample"), ("clean_holdout", "clean_holdout")]
    sl = list(d["llm_in_sample"]["sleeves"].keys())
    # order sleeves by family cluster
    fam = rj(_PD)["clusters"]["map"]
    sl.sort(key=lambda s: (str(fam.get(s.split("|")[0], "")), s))
    factors = ["MKT", "TSMOM", "VOL", "CARRY", "LIQ"]
    for ax, (k, name) in zip(axes, seg_keys):
        sv = d[k]["sleeves"]
        M = np.array([[num(sv.get(s, {}).get("std_loading", {}).get(f)) for f in factors] + [
            num(sv.get(s, {}).get("alpha_t"))] for s in sl])
        L = M[:, :-1]
        heatmap(ax, L, factors, [s.split("|")[0].replace("_", " ") for s in sl], cmap=DIV_CMAP, center=0,
                fmt="{:+.2f}", cbar_label="standardised loading", vmax=0.8, fontsize=9)
        ax.set_title(f"{SEG_LABEL[name]}  (median R^2 {d[k]['median_r2']:.2f})", fontsize=10.5)
        for i, at in enumerate(M[:, -1]):
            if np.isfinite(at):
                ax.text(len(factors) - 0.4, i, f"alpha t {at:+.1f}", va="center", ha="left", fontsize=9,
                        color=C[3] if abs(at) >= 2 else "#444")
        ax.set_xlim(-0.5, len(factors) + 0.9)
    return fig, caption(f"research/reports/{_PD} (H-PD2 factor regressions, daily, HAC t)",
                        "llm_in_sample vs clean_holdout", "sleeves grouped by return-correlation cluster")


@figure("pnl", "exit_leakage",
        "Edge leaks through exits, not fees: ~70% of the favourable excursion is given back, costs ~5%",
        "pnl_decomposition_2026-11-11 (H-PD4)")
def _co_fig_leak():
    d = rj(_PD)["H-PD4"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    parts = [("D_mfe", "max favourable\nexcursion"), ("exit_leak", "given back\nbefore exit"),
             ("dir_loss", "losers'\nadverse moves"), ("costs", "fees +\nslippage"), ("net", "net P&L")]
    for ax, seg in zip(axes, SEG):
        a = d[seg]["all"]
        tot = a["D_mfe"]
        vals = [a["D_mfe"], -a["exit_leak"], -a["dir_loss"], -a["costs"]]
        cum = 0.0
        for i, (v, (k, lab)) in enumerate(zip(vals, parts[:-1])):
            bottom = cum if v >= 0 else cum + v
            col = C[2] if i == 0 else [C[3], C[1], C[4]][i - 1]
            ax.bar(i, abs(v) / tot * 100, bottom=bottom / tot * 100, color=col, width=0.65)
            ax.annotate(f"{v / tot * 100:+.0f}%", (i, (bottom + abs(v)) / tot * 100), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=10)
            cum += v
        net = a["net"]
        ax.bar(4, net / tot * 100, color=C[7], width=0.65)
        ax.annotate(f"{net / tot * 100:+.1f}%", (4, max(net, 0) / tot * 100), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(range(5), [p[1] for p in parts], fontsize=9.5)
        lo, hi = d[seg]["ci95_share"]
        ax.set_title(f"{SEG_LABEL[seg]}: {a['n_trades']:,} trades\nleak share {a['share']:.2f} (95% CI {lo:.2f}-{hi:.2f})",
                     fontsize=10.5)
        ax.set_ylim(-12, 118)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of total max favourable excursion")
    return fig, caption(f"research/reports/{_PD} (H-PD4; MFE from bar highs/lows, so leakage is an upper bound)",
                        "llm_in_sample vs clean_holdout")


@figure("pnl", "exit_policy_trails",
        "Tight k-sigma trails do not recover the leakage: they cut exposure from 0.86 to 0.10-0.25\nand lose in-sample",
        "exit_policy_2026-11-12 (H-PD6)")
def _co_fig_exit_policy():
    e = rj("exit_policy_2026-11-12.json")
    ks = ["k1.5", "k2.5", "k3.5"]
    labels = ["original\nexits", "k = 1.5", "k = 2.5", "k = 3.5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(4)
    for i, seg in enumerate(SEG):
        m = e["main"][seg]
        sh = [m["orig_sharpe"]] + [m["k"][k]["sharpe"] for k in ks]
        bars = axes[0].bar(x + (i - 0.5) * 0.38, sh, 0.38, color=SEG_COLOR[seg], label=f"{SEG_LABEL[seg]} ({m['days']} d)")
        annotate_bars(axes[0], bars, "{:+.2f}", fontsize=9.5)
        ex = [e["exposure"][seg]["orig"]] + [e["exposure"][seg][k] for k in ks]
        bars = axes[1].bar(x + (i - 0.5) * 0.38, ex, 0.38, color=SEG_COLOR[seg], label=SEG_LABEL[seg])
        annotate_bars(axes[1], bars, "{:.2f}", fontsize=9.5)
    axes[0].axhline(0, color="#333", lw=1)
    axes[0].set_ylabel("annualised Sharpe of the book (net)")
    axes[0].set_title("Book Sharpe")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.4, hi + 0.4)
    axes[1].set_ylabel("average exposure (share of time in a position)")
    axes[1].set_title("Exposure")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper right", fontsize=10)
    t = [e["main"]["llm_in_sample"]["k"][k]["t_nw"] for k in ks]
    axes[0].text(0.98, 0.98, "in-sample diff vs original:\nNW t " + ", ".join(f"{v:.1f}" for v in t),
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9.5)
    return fig, caption("research/reports/exit_policy_2026-11-12.json (main, exposure; close-based k-sigma trails)",
                        "llm_in_sample vs clean_holdout")


# --------------------------------------------------------------- TradFi sessions
_TS = "tradfi_sessions_2026-11-11.json"


@figure("tradfi", "session_strategies",
        "TradFi session lore has no edge on Bybit perps: every session rule is <= 0 net in the holdout",
        "tradfi_sessions_2026-11-11 (H-TP1, H-TF1-5)")
def _co_fig_sessions():
    s = rj(_TS)["strategies"]
    names = [("TP1_fade_overnight_first2h", "fade overnight move,\nfirst 2 h"),
             ("TP1_fade_overnight_first2h_if_gt_1sigma", "fade overnight\nif > 1 sigma"),
             ("TF1_gap_fade_to_prev_close", "gap fade to\nprior close"),
             ("TF3_orb_to_close", "opening-range\nbreakout"),
             ("TF4_last_hour_follow_first_hour", "last hour follows\nfirst hour"),
             ("TF2_long_overnight", "long overnight"),
             ("TF5_follow_reopen_into_europe", "follow CME reopen\ninto Europe")]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(names))
    ntr = []
    for i, seg in enumerate(SEG):
        est, se, nt = [], [], []
        for k, _ in names:
            d = s[k][seg]
            d = d.get("long_overnight", d)
            m, t = num(d.get("mean_net_ru")), num(d.get("t_net_ru"))
            est.append(m)
            se.append(abs(m / t) if np.isfinite(t) and t != 0 else np.nan)
            nt.append(d.get("n_trades", 0))
        est, se = np.array(est), np.array(se)
        ax.errorbar(x + (i - 0.5) * 0.3, est, yerr=1.96 * se, fmt="o", ms=7, color=SEG_COLOR[seg], capsize=4,
                    lw=1.5, label=SEG_LABEL[seg])
        for xi, v in zip(x, est):
            ax.annotate(f"{v:+.2f}", (xi + (i - 0.5) * 0.3, v), xytext=(7 if i else -7, 0),
                        textcoords="offset points", ha="left" if i else "right", va="center", fontsize=10)
        ntr.append(nt)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, [f"{n[1]}\nn = {a_} / {b_}" for n, a_, b_ in zip(names, *ntr)], fontsize=9.5)
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel("mean net P&L per trade (R units, 7.5 bp/side)")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    return fig, caption(f"research/reports/{_TS} (strategies; 9 equity + 4 commodity perps)",
                        "llm_in_sample (Mar-Jun 2026, short) vs clean_holdout", "bars: 95% CI from date-clustered t; n = trades in-sample / holdout")


@figure("tradfi", "off_hours_activity",
        "Equity perps are half alive off-hours: |1h returns| run 0.4-0.7x the cash session, weekends ~0.3x",
        "tradfi_sessions_2026-11-11 (off_hours)")
def _co_fig_offhours():
    o = rj(_TS)["off_hours"]
    sess = [("pre_04_08ET", "pre-market\n04-08 ET"), ("first_cash_bar_09ET", "first cash\nbar 09 ET"),
            ("regular_10_15ET", "regular\n10-15 ET"), ("post_16_19ET", "post-market\n16-19 ET"),
            ("overnight_20_03ET", "overnight\n20-03 ET"), ("weekend", "weekend")]
    syms = list(o.keys())
    M = np.array([[num(o[s].get(k, {}).get("abs_ret_rel")) for k, _ in sess] for s in syms])
    fig, ax = plt.subplots(figsize=(10, 6.4))
    from matplotlib.colors import LogNorm
    im = ax.imshow(M, cmap=SEQ_CMAP, norm=LogNorm(vmin=0.1, vmax=4), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color="white" if v > 1.2 else "#111")
    ax.set_xticks(range(len(sess)), [s[1] for s in sess])
    ax.set_yticks(range(len(syms)), [s.replace("USDT", "") for s in syms])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("mean |1h return| relative to the regular 10-15 ET session")
    cb.set_ticks([0.1, 0.25, 0.5, 1, 2, 4], labels=["0.1x", "0.25x", "0.5x", "1x", "2x", "4x"])
    cb.minorticks_off()
    cb.outline.set_visible(False)
    return fig, caption(f"research/reports/{_TS} (off_hours.abs_ret_rel)", "pooled (TradFi perps listed Mar-Jun 2026)")


# ================================================================ ML, sparsity, states, mining, calibration, budget

# ----------------------------------------------------------------- ML, sparsity, states, mining,
# calibration and decision budget (KNOWLEDGE.md sections 1, 4, 6, 11, 12)


def _ml_half(diff, t):
    """95% half-width from a point estimate and its (HAC / clustered) t statistic."""
    d, tt = num(diff), num(t)
    if not (np.isfinite(d) and np.isfinite(tt)) or tt == 0:
        return np.nan
    return 1.96 * abs(d / tt)


def _ml_wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _ml_model_colors():
    # fixed model colours (segments keep C[0] / C[1])
    return {"lin": C[2], "mlp": C[4], "xgb": C[5], "cnn": C[9]}


def _ml_label_ends(ax, xs, vals, lo, hi, fmt, fontsize=10, color="#222"):
    """Write each value just beyond the end of its CI whisker (no overlap with the error bar)."""
    for x, v, l, h in zip(xs, vals, lo, hi):
        if not np.isfinite(v):
            continue
        l = 0.0 if not np.isfinite(l) else l
        h = 0.0 if not np.isfinite(h) else h
        if v >= 0:
            ax.annotate(fmt.format(v), (x, v + h), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize, color=color)
        else:
            ax.annotate(fmt.format(v), (x, v - l), xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=fontsize, color=color)


def _ml_pad_y(ax, ymin, ymax, frac=0.2):
    span = (ymax - ymin) or 1.0
    ax.set_ylim(min(ymin, 0) - frac * span, max(ymax, 0) + frac * span)


def _ml_segbars(ax, labels, vals, halfs, fmt="{:.3f}", segs=None, width=0.38):
    """Grouped bars: one group per label, one bar per segment (segment colours), symmetric CIs."""
    segs = segs or SEG
    xs0 = np.arange(len(labels))
    lo_all, hi_all = [], []
    for i, seg in enumerate(segs):
        v = np.asarray(vals[seg], float)
        e = np.asarray(halfs[seg], float)
        x = xs0 + (i - (len(segs) - 1) / 2) * width
        ax.bar(x, v, width * 0.95, color=SEG_COLOR[seg], label=SEG_LABEL[seg])
        ax.errorbar(x, v, yerr=np.where(np.isfinite(e), e, 0), fmt="none", ecolor="#333",
                    elinewidth=1.2, capsize=3)
        _ml_label_ends(ax, x, v, e, e, fmt)
        lo_all.append(np.nanmin(v - np.nan_to_num(e)))
        hi_all.append(np.nanmax(v + np.nan_to_num(e)))
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(xs0, labels)
    ax.grid(axis="x", visible=False)
    _ml_pad_y(ax, min(lo_all), max(hi_all), 0.18)


# --------------------------------------------------------------------------------- ml


@figure("ml", "daily_vol_vs_har",
        "Daily vol: MLP and XGB beat log-HAR in both segments, but weekday dummies do most of it",
        "ml_signals_2026-11-12.json (H-NN1)")
def _ml_daily_vol_vs_har():
    d = rj("ml_signals_2026-11-12.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [4, 3]})
    specs = [
        (axes[0], "vs per-asset log-HAR",
         [("log-HAR\n+ weekday", lambda s: d["exploratory_har_dow"][s]["har_dow_vs_har"]),
          ("linear\n(52 inputs)", lambda s: d["H-NN1"][s]["lin_vs_har"]),
          ("MLP", lambda s: d["H-NN1"][s]["mlp_vs_har"]),
          ("XGB", lambda s: d["H-NN1"][s]["xgb_vs_har"])]),
        (axes[1], "vs log-HAR + weekday dummies",
         [("linear", lambda s: d["exploratory_har_dow"][s]["lin_vs_har_dow"]),
          ("MLP", lambda s: d["exploratory_har_dow"][s]["mlp_vs_har_dow"]),
          ("XGB", lambda s: d["exploratory_har_dow"][s]["xgb_vs_har_dow"])]),
    ]
    for ax, ttl, items in specs:
        vals = {s: [100 * num(f(s)["rel"]) for _, f in items] for s in SEG}
        halfs = {s: [100 * _ml_half(f(s)["rel"], f(s)["t"]) for _, f in items] for s in SEG}
        _ml_segbars(ax, [lab for lab, _ in items], vals, halfs, fmt="{:+.1f}%")
        ax.set_title(ttl)
        ax.set_ylabel("next-day QLIKE change vs reference (%), lower = better")
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    axes[0].legend(loc="lower left")
    n_in = d["H-NN1"]["llm_in_sample"]["mlp_vs_har"]["n_dates"]
    n_ho = d["H-NN1"]["clean_holdout"]["mlp_vs_har"]["n_dates"]
    return fig, caption("research/reports/ml_signals_2026-11-12.json (H-NN1, exploratory_har_dow)",
                        f"llm_in_sample ({n_in} dates) and clean_holdout ({n_ho} dates), 23 crypto perps",
                        "95% CI = +-1.96 x |estimate / date-clustered t|; winning family flips between segments")


@figure("ml", "hourly_crypto_vs_baseline",
        "Nonlinear models beat the hourly vol baseline in both segments; half the gain is inputs",
        "ml_signals_round2_2026-11-12.json (H-NN8)")
def _ml_hourly_crypto():
    d = rj("ml_signals_round2_2026-11-12.json")["H-NN8"]
    items = [("linear\n(same inputs)", "lin_vs_champ"), ("MLP", "mlp_vs_champ"), ("XGB", "xgb_vs_champ")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    vals = {s: [num(d[s][k]["diff"]) for _, k in items] for s in SEG}
    halfs = {s: [_ml_half(d[s][k]["diff"], d[s][k]["t"]) for _, k in items] for s in SEG}
    _ml_segbars(ax, [a for a, _ in items], vals, halfs, fmt="{:+.3f}")
    ax.set_ylabel("next-hour QLIKE difference vs baseline (lower = better)")
    ax.set_title("baseline = robust_bc sigma^2 x expanding hour-of-week factor")
    ax.legend(loc="lower left")
    cb = {s: num(d[s]["champ_vs_base"]["diff"]) for s in SEG}
    return fig, caption("research/reports/ml_signals_round2_2026-11-12.json (H-NN8)",
                        f"llm_in_sample ({d['llm_in_sample']['mlp_vs_champ']['n_dates']} dates) and clean_holdout "
                        f"({d['clean_holdout']['mlp_vs_champ']['n_dates']} dates), 23 crypto perps",
                        f"the baseline itself beats bare robust sigma by {cb['llm_in_sample']:+.3f} / "
                        f"{cb['clean_holdout']:+.3f}; 95% CI from date-clustered t")


@figure("ml", "hourly_tradfi",
        "TradFi perps gain most: holdout MLP cuts next-hour QLIKE by 0.176 (t -6.9), 15 of 15 symbols",
        "ml_hourly_xasset_2026-11-12.json (H-NN11)")
def _ml_hourly_tradfi():
    d = rj("ml_hourly_xasset_2026-11-12.json")["H-NN11"]
    mc = _ml_model_colors()
    models = [("lin", "linear"), ("mlp", "MLP"), ("xgb", "XGB")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [3, 4]}, sharey=True)
    lo_all, hi_all = [], []
    for ax, seg in zip(axes, SEG):
        x = d[seg]
        cls = [("all", x)] + [(c, x["by_class"][c]) for c in ("commodity", "equity", "gold_token") if c in x["by_class"]]
        xs0 = np.arange(len(cls))
        w = 0.26
        for j, (m, lab) in enumerate(models):
            v = np.array([num(c[1][m]["diff"]) for c in cls])
            e = np.array([_ml_half(c[1][m]["diff"], c[1][m]["t"]) for c in cls])
            xx = xs0 + (j - 1) * w
            ax.bar(xx, v, w * 0.95, color=mc[m], label=lab)
            ax.errorbar(xx, v, yerr=e, fmt="none", ecolor="#333", elinewidth=1.2, capsize=2.5)
            keep = np.array([i == 0 or m == "mlp" for i in range(len(cls))])
            _ml_label_ends(ax, xx[keep], v[keep], e[keep], e[keep], "{:.2f}", fontsize=10)
            lo_all.append(np.nanmin(v - e))
            hi_all.append(np.nanmax(v + e))
        ax.set_xticks(xs0, [f"{c[0].replace('_', ' ')}" + (f"\n({x['n_symbols']} symbols)" if c[0] == "all" else "")
                            for c in cls])
        ax.axhline(0, color="#333", lw=1)
        ax.grid(axis="x", visible=False)
        ax.set_title(SEG_LABEL[seg], color=SEG_COLOR[seg])
    _ml_pad_y(axes[0], min(lo_all), max(hi_all), 0.12)
    axes[0].set_ylabel("next-hour QLIKE difference vs baseline (lower = better)")
    axes[1].legend(loc="lower right", ncol=3)
    axes[1].text(0.99, 0.13, "labels: every model for 'all', MLP for each class", transform=axes[1].transAxes,
                 ha="right", fontsize=10, color="#555")
    return fig, caption("research/reports/ml_hourly_xasset_2026-11-12.json (H-NN11)",
                        f"llm_in_sample ({d['llm_in_sample']['n_dates']} dates, 5 commodity / gold perps) and "
                        f"clean_holdout ({d['clean_holdout']['n_dates']} dates, 15 TradFi perps)",
                        "baseline = robust_bc x hour of week; 95% CI from date-clustered t")


@figure("ml", "hourly_cross_asset",
        "Adding BTC, ETH and QQQ activity does not help next-hour vol: XGB ~0, the MLP gets worse",
        "ml_hourly_xasset_2026-11-12.json (H-NN12)")
def _ml_cross_asset():
    d = rj("ml_hourly_xasset_2026-11-12.json")["H-NN12"]
    items = [("linear", "lin"), ("MLP", "mlp"), ("XGB", "xgb")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    for ax, (key, ttl) in zip(axes, [("crypto", "23 crypto perps"), ("tradfi", "TradFi perps")]):
        vals = {s: [num(d[key][s][k]["diff"]) for _, k in items] for s in SEG}
        halfs = {s: [_ml_half(d[key][s][k]["diff"], d[key][s][k]["t"]) for _, k in items] for s in SEG}
        _ml_segbars(ax, [a for a, _ in items], vals, halfs, fmt="{:+.3f}")
        ax.set_title(ttl)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    axes[0].set_ylim(lo, hi + 0.25 * (hi - lo))
    axes[0].set_ylabel("QLIKE change from adding leader inputs\n(same model; > 0 = worse)")
    axes[0].legend(loc="lower left")
    return fig, caption("research/reports/ml_hourly_xasset_2026-11-12.json (H-NN12)",
                        "llm_in_sample and clean_holdout (TradFi in-sample: 5 perps, 91 dates)",
                        "difference = model with BTC / ETH / QQQ activity minus the same model without; 95% CI from date-clustered t")


@figure("ml", "next_minute_vol",
        "Next-minute vol: a linear model of recent activity does the work; a CNN adds only 0.8% QLIKE",
        "ml_micro_2026-11-12.json (H-NN5)")
def _ml_next_minute():
    d = rj("ml_micro_2026-11-12.json")["H-NN5"]
    labels = ["linear (10/60/300 s\naggregates) vs baseline", "CNN (raw 1 s path)\nvs baseline", "CNN vs linear"]
    vals, halfs = {}, {}
    for s in SEG:
        x = d[s]
        lb, cb = x["context"]["lin_vs_base"], x["context"]["cnn_vs_base"]
        vals[s] = [num(lb["mean"]), num(cb["mean"]), num(x["mean_diff"])]
        halfs[s] = [_ml_half(lb["mean"], lb["t"]), _ml_half(cb["mean"], cb["t"]),
                    _ml_half(x["mean_diff"], x["t_day_clustered"])]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    _ml_segbars(ax, labels, vals, halfs, fmt="{:+.3f}")
    ax.set_ylabel("next-minute QLIKE difference per window (lower = better)")
    p = d["pooled"]
    ratio = num(p["context"]["lin_vs_base"]["mean"]) / num(p["mean_diff"])
    ax.annotate(f"pooled: linear's gain over the baseline is {ratio:.0f}x\nthe CNN's extra gain "
                f"({100 * num(p['rel_to_lin_qlike']):+.1f}% QLIKE)", xy=(2, num(p["mean_diff"])),
                xytext=(1.35, -0.06), fontsize=10, arrowprops=dict(arrowstyle="->", color="#555"))
    ax.legend(loc="lower right")
    return fig, caption("research/reports/ml_micro_2026-11-12.json (H-NN5)",
                        f"llm_in_sample ({d['llm_in_sample']['n_symbol_days']} symbol-days, {d['llm_in_sample']['n_days']} days) and "
                        f"clean_holdout ({d['clean_holdout']['n_symbol_days']} symbol-days, {d['clean_holdout']['n_days']} days)",
                        "baseline = 60 x trailing-1h per-second variance; CI: symbol-day t (vs baseline), day-clustered t (CNN vs linear)")


def _ml_group_name(g):
    return {"vol_level": "vol level", "vol_shape": "vol shape", "jumps_semivar": "jumps / semivariance",
            "range_measures": "range estimators", "returns": "returns", "trend_regime": "trend / regime",
            "funding_oi": "funding / OI", "calendar": "calendar (weekday)", "market": "market (BTC, index)",
            "base_sigma": "robust sigma", "seasonal": "hour of week", "funding_hour": "funding hour",
            "recent_returns": "recent |r| (1-24 h)", "jump": "jump flags", "daily_har": "daily HAR level",
            "weekday": "weekday", "interactions": "pairwise interactions"}.get(g, g)


def _ml_hbars_seg(ax, groups, vals, fmt="{:.0%}", errs=None):
    ys = np.arange(len(groups))
    h = 0.38
    for i, s in enumerate(SEG):
        v = np.asarray(vals[s], float)
        y = ys + (i - 0.5) * h
        bars = ax.barh(y, v, h * 0.95, color=SEG_COLOR[s], label=SEG_LABEL[s])
        if errs is not None:
            e = np.asarray(errs[s], float)
            ax.errorbar(v, y, xerr=e, fmt="none", ecolor="#333", elinewidth=1.1, capsize=2.5)
            for yy, vv, ee in zip(y, v, e):
                xx = vv + (ee if vv >= 0 else -ee)
                ax.annotate(fmt.format(vv), (xx, yy), xytext=(4 if vv >= 0 else -4, 0), textcoords="offset points",
                            ha="left" if vv >= 0 else "right", va="center", fontsize=10)
        else:
            annotate_barh(ax, bars, fmt=fmt)
    ax.set_yticks(ys, [_ml_group_name(g) for g in groups])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


@figure("ml", "attribution_daily",
        "Daily vol models run on the vol level (~40% of TreeSHAP) and the weekday; the rest is small",
        "ml_signals_2026-11-12.json (H-NN3)")
def _ml_attr_daily():
    d = rj("ml_signals_2026-11-12.json")["H-NN3"]
    gm = {s: dict(d[s]["group_main_share"]) for s in SEG}
    for s in SEG:
        gm[s]["interactions"] = num(d[s]["interaction_share"])
    groups = sorted([g for g in gm["llm_in_sample"] if g != "interactions"], key=lambda g: -gm["llm_in_sample"][g])
    groups.append("interactions")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    _ml_hbars_seg(ax, groups, {s: [gm[s][g] for g in groups] for s in SEG})
    ax.set_xlabel("share of mean |TreeSHAP| (main effects by input group; interactions pooled)")
    ax.set_xlim(0, max(max(gm[s].values()) for s in SEG) * 1.2)
    ax.legend(loc="center right")
    return fig, caption("research/reports/ml_signals_2026-11-12.json (H-NN3, XGB TreeSHAP interaction values)",
                        f"llm_in_sample ({d['llm_in_sample']['n_days']} days) and clean_holdout ({d['clean_holdout']['n_days']} days)",
                        "ablating the vol-level group costs +75% QLIKE; trend, funding and market are 1-7% each")


@figure("ml", "attribution_hourly",
        "Hourly vol: recent |r| (41%), hour of week (24%) and the daily HAR level (21%) carry the signal",
        "ml_signals_round2_2026-11-12.json (H-NN8)")
def _ml_attr_hourly():
    d = rj("ml_signals_round2_2026-11-12.json")["H-NN8"]
    sh = {s: d[s]["shap_group_share"] for s in SEG}
    groups = sorted(sh["clean_holdout"], key=lambda g: -sh["clean_holdout"][g])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    _ml_hbars_seg(axes[0], groups, {s: [num(sh[s][g]) for g in groups] for s in SEG})
    axes[0].set_xlabel("share of mean |TreeSHAP| (XGB)")
    axes[0].set_xlim(0, 0.55)
    axes[0].set_title("attribution")
    abl = {s: [num(d[s]["group_ablation"][f"xgb:{g}"]["dqlike"]) for g in groups] for s in SEG}
    err = {s: [_ml_half(d[s]["group_ablation"][f"xgb:{g}"]["dqlike"], d[s]["group_ablation"][f"xgb:{g}"]["t"])
               for g in groups] for s in SEG}
    # re-draw with error bars on the right panel
    _ml_hbars_seg(axes[1], groups, abl, fmt="{:+.3f}", errs=err)
    axes[1].invert_yaxis()  # sharey: undo the second inversion
    axes[1].tick_params(labelleft=False)
    axes[1].axvline(0, color="#333", lw=1)
    axes[1].set_xlabel("QLIKE increase when the group is ablated (XGB)")
    axes[1].set_title("ablation (> 0 = the group helps)")
    xmax = max(np.nanmax(np.array(abl[s]) + np.array(err[s])) for s in SEG)
    axes[1].set_xlim(-0.04, xmax * 1.3)
    axes[0].legend(loc="lower right")
    return fig, caption("research/reports/ml_signals_round2_2026-11-12.json (H-NN8 shap_group_share, group_ablation)",
                        "llm_in_sample (1,826 dates) and clean_holdout (132 dates), 23 crypto perps",
                        "funding-hour and jump flags add nothing; 95% CI from date-clustered t")


# --------------------------------------------------------------------------- sparsity


@figure("sparsity", "pruning_curve",
        "95% of the daily vol net's weights prune away with no loss (QLIKE ratio 0.986 / 0.974)",
        "sparsity_2026-11-12.json (H-SP1)")
def _ml_pruning_curve():
    d = rj("sparsity_2026-11-12.json")["sp1"]
    alive = np.array(d["n_alive"], float)
    sp = np.array(d["sparsity"], float)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for s in SEG:
        c = d["segments"][s]["curve"]
        r = np.array([num(x["ratio"]) for x in c])
        lo = np.array([num(x["ci95"][0]) for x in c])
        hi = np.array([num(x["ci95"][1]) for x in c])
        ax.fill_between(alive, lo, hi, color=SEG_COLOR[s], alpha=0.15, lw=0)
        ax.plot(alive, r, "-o", color=SEG_COLOR[s], ms=3.5, lw=1.8, label=SEG_LABEL[s])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ticks = [2192, 1126, 465, 192, 100, 44, 21]
    lab = []
    for t in ticks:
        i = int(np.argmin(abs(alive - t)))
        lab.append(f"{int(alive[i])}\n({100 * sp[i]:.0f}%)")
    ax.set_xticks(ticks, lab)
    ax.minorticks_off()
    ax.axhline(1.0, color="#333", lw=1)
    lv = {s: d["segments"][s]["levels"]["95"] for s in SEG}
    x95 = alive[int(lv["llm_in_sample"]["round"])]
    ax.axvline(x95, color="#555", ls="--", lw=1.2)
    ax.annotate(f"95% pruned ({int(x95)} weights left)\nratio {lv['llm_in_sample']['ratio']:.3f} "
                f"[{lv['llm_in_sample']['ci95'][0]:.3f}, {lv['llm_in_sample']['ci95'][1]:.3f}] in-sample\n"
                f"ratio {lv['clean_holdout']['ratio']:.3f} [{lv['clean_holdout']['ci95'][0]:.3f}, "
                f"{lv['clean_holdout']['ci95'][1]:.3f}] holdout",
                xy=(x95, 0.97), xytext=(900, 1.16), fontsize=10, arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_ylim(0.8, 1.3)
    ax.set_xlabel("surviving weights (log scale; % of 2,192 pruned in brackets)")
    ax.set_ylabel("QLIKE ratio, pruned / dense MLP (< 1 = better)")
    ax.legend(loc="upper left")
    return fig, caption("research/reports/sparsity_2026-11-12.json (H-SP1, iterative magnitude pruning)",
                        "llm_in_sample (1,552 dates) and clean_holdout (133 dates)",
                        "bands = 95% date-block bootstrap CI; beyond 98% pruning the in-sample loss rises")


def _ml_unit_category(pair, groups):
    vol = set(groups["vol_level"]) | set(groups["vol_shape"]) | {"mkt_lrv_d", "mkt_lrv_w"}
    cal = set(groups["calendar"])
    nv = sum(p in vol for p in pair)
    nc = sum(p in cal for p in pair)
    if nv == 2:
        return "vol level x vol level"
    if nv == 1 and nc == 1:
        return "vol level x calendar"
    if nv == 1:
        return "vol level x trend / returns / funding"
    if nc == 2:
        return "calendar only"
    return "no vol-level input"


@figure("sparsity", "surviving_units",
        "17 of 24 surviving units are simple 2-input functions; the strongest pair vol level with weekday",
        "sparsity_2026-11-12.json (H-SP3)")
def _ml_surviving_units():
    s = rj("sparsity_2026-11-12.json")
    groups = rj("ml_signals_2026-11-12.json")["groups"]
    units = sorted(s["sp3"]["units"], key=lambda u: -num(u["attribution"]))
    cats = ["vol level x vol level", "vol level x calendar", "vol level x trend / returns / funding",
            "calendar only", "no vol-level input"]
    ccol = {c: C[i + 2] for i, c in enumerate(cats)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    ys = np.arange(len(units))
    for y, u in zip(ys, units):
        pair = u["clean_holdout"]["best2"]
        cat = _ml_unit_category(pair, groups)
        ax.barh(y, num(u["clean_holdout"]["r2_2"]), 0.75, color=ccol[cat])
        ax.plot(num(u["llm_in_sample"]["r2_2"]), y, "|", color="#111", ms=9, mew=2)
    ax.set_yticks(ys, [f"L{u['unit'][0]}u{u['unit'][1]}: {' + '.join(u['clean_holdout']['best2'])}" for u in units],
                  fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0.8, color="#333", ls="--", lw=1.2)
    ax.set_ylim(len(units) - 0.4, -1.9)
    ax.text(0.79, -1.1, "R^2 = 0.8 bar", ha="right", va="center", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("holdout R^2 of the best additive 2-input spline")
    ax.set_title("units ordered by attribution (top = largest)")
    ax.grid(axis="y", visible=False)
    # right: count by category, met vs not met
    ax2 = axes[1]
    met = {c: 0 for c in cats}
    miss = {c: 0 for c in cats}
    for u in units:
        cat = _ml_unit_category(u["clean_holdout"]["best2"], groups)
        (met if num(u["clean_holdout"]["r2_2"]) > 0.8 else miss)[cat] += 1
    yy = np.arange(len(cats))
    ax2.barh(yy, [met[c] for c in cats], 0.6, color=[ccol[c] for c in cats])
    ax2.barh(yy, [miss[c] for c in cats], 0.6, left=[met[c] for c in cats], color=[ccol[c] for c in cats],
             alpha=0.35, hatch="//", edgecolor="white")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hnd = [Patch(color="#777", label="holdout R^2 > 0.8"),
           Patch(facecolor="#777", alpha=0.35, hatch="//", edgecolor="white", label="holdout R^2 <= 0.8"),
           Line2D([], [], marker="|", ls="none", color="#111", ms=9, mew=2, label="in-sample R^2 (left)")]
    for y, c in zip(yy, cats):
        ax2.text(met[c] + miss[c] + 0.15, y, f"{met[c]} of {met[c] + miss[c]}", va="center", fontsize=10)
    ax2.set_yticks(yy, [c.replace(" / ", "/\n").replace("x trend", "x\ntrend") if len(c) > 22 else c for c in cats])
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(met[c] + miss[c] for c in cats) + 8)
    from matplotlib.ticker import MaxNLocator
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel("surviving units (holdout fit)")
    ax2.set_title("input pair type")
    ax2.grid(axis="y", visible=False)
    ax2.legend(handles=hnd, loc="lower right", frameon=True, framealpha=0.95)
    n_ok = sum(met.values())
    return fig, caption("research/reports/sparsity_2026-11-12.json (H-SP3, seed-0 net at 95% pruning)",
                        "clean_holdout fit (bars), llm_in_sample fit (ticks)",
                        f"{n_ok} / {len(units)} units with holdout R^2 > 0.8; H-SP3 failed on its ablation criterion (redundant circuit)")


# ----------------------------------------------------------------------------- states


@figure("states", "cell_table",
        "The 36-cell vol table is mostly weekday: Saturdays far below log-HAR, Mondays well above",
        "discrete_states_2026-11-12.json (H-DS1 cell table, LAT1)")
def _ml_cell_table():
    d = rj("discrete_states_2026-11-12.json")
    rows = {r["cell"]: r for r in d["cell_table"]["rows"]}
    wg = ["Mon", "Tue-Fri", "Sat", "Sun"]
    dr = ["down", "range", "up"]
    ylabels = [f"{w}, v0 {x}" for w in wg for x in dr]
    xlabels = ["vol bottom third", "middle third", "top third"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    M = {}
    for key, seg in [("in", "llm_in_sample"), ("ho", "clean_holdout")]:
        A = np.full((12, 3), np.nan)
        for g in range(4):
            for t in range(3):
                for k in range(3):
                    r = rows.get(g * 9 + t * 3 + k)
                    if r is not None:
                        A[g * 3 + k, t] = 100 * num(r[f"effect_{key}"])
        M[seg] = A
    span = max(np.nanmax(np.abs(M[s])) for s in SEG)
    for ax, s in zip(axes, SEG):
        heatmap(ax, M[s], xlabels, ylabels, cmap=DIV_CMAP, center=0, fmt="{:+.0f}", vmax=span,
                cbar_label="next-day RV vs log-HAR (%)", fontsize=10)
        ax.set_title(SEG_LABEL[s], color=SEG_COLOR[s])
        for yb in (2.5, 5.5, 8.5):
            ax.axhline(yb, color="white", lw=2.5)
        ax.tick_params(axis="x", labelrotation=0)
    axes[1].set_yticklabels([])
    nho = [r["n_ho"] for r in d["cell_table"]["rows"]]
    return fig, caption("research/reports/discrete_states_2026-11-12.json (cell_table, partition frozen at the 2026-07 refit)",
                        f"llm_in_sample and clean_holdout (holdout cells {min(nho)}-{max(nho)} rows)",
                        "cell = forecast weekday group x own-1y vol tercile x v0 trend; EB-shrunk exp(cell value) - 1")


@figure("states", "gain_vs_xgb",
        "36 cells beat log-HAR by 6-7% in both segments but recover only half of XGB's holdout gain",
        "discrete_states_2026-11-12.json (H-DS1)")
def _ml_states_gain():
    d = rj("discrete_states_2026-11-12.json")["H-DS1"]
    items = [("log-HAR\n+ weekday", "har_dow_vs_har"), ("weekday\n(7 cells)", "WD7"), ("v0 regime\n(9)", "V0"),
             ("LAT1\n(36)", "LAT1"), ("LAT2\n(36)", "LAT2"), ("k-means\n(64)", "KM64"), ("honest tree\n(HT3)", "HT3"),
             ("MLP", "mlp_vs_har"), ("XGB", "xgb_vs_har")]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    vals = {s: [100 * num(d[s][k]["rel"]) for _, k in items] for s in SEG}
    halfs = {s: [100 * _ml_half(d[s][k]["rel"], d[s][k]["t"]) for _, k in items] for s in SEG}
    _ml_segbars(ax, [a for a, _ in items], vals, halfs, fmt="{:+.1f}")
    ax.axvspan(2.5, 6.5, color="#999", alpha=0.08, lw=0)
    ax.axvline(6.5, color="#777", lw=1, ls=":")
    sh = d["clean_holdout"]["best_share"]
    ax.annotate(f"LAT1 holdout share of XGB gain\n{sh['share']:.2f} [{sh['ci95'][0]:.2f}, {sh['ci95'][1]:.2f}]",
                xy=(3.19, 100 * num(d["clean_holdout"]["LAT1"]["rel"])), xytext=(3.6, -27), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_ylabel("next-day QLIKE change vs log-HAR (%), lower = better")
    ax.text(4.5, ax.get_ylim()[1] * 0.92, "discrete cell tables (log-HAR x cell factor)", ha="center", fontsize=10,
            color="#444")
    ax.legend(loc="lower left")
    return fig, caption("research/reports/discrete_states_2026-11-12.json (H-DS1)",
                        "llm_in_sample (1,552 dates) and clean_holdout (133 dates), 23 crypto perps",
                        "95% CI from date-clustered t; the cell shortfall sits on weekend forecast days")


# ----------------------------------------------------------------------------- mining


@figure("mining", "replication",
        "Mined vol effects replicate out of sample (54 of 79); mined return effects at chance (1 of 11)",
        "daily_mining_2026-11-12.json (H-MINE2)")
def _ml_mining_replication():
    d = rj("daily_mining_2026-11-12.json")["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    panels = [
        (axes[0], "by kind of effect",
         [("all\ndiscoveries", d["overall"]), ("vol effects\n(next-day RV)", d["by_kind"]["vol"]),
          ("return effects\n(next 1 d / 5 d)", d["by_kind"]["return"])]),
        (axes[1], "by relation to known rules",
         [("restates a\nknown rule", d["by_known"]["known"]), ("contradicts a\nknown rule", d["by_known"]["contradicts"]),
          ("novel", d["by_known"]["novel"])]),
    ]
    for ax, ttl, items in panels:
        xs = np.arange(len(items))
        r = np.array([num(x["rate"]) for _, x in items])
        lo = np.array([num(x["wilson95"][0]) for _, x in items])
        hi = np.array([num(x["wilson95"][1]) for _, x in items])
        ax.bar(xs, r, 0.6, color=SEG_COLOR["clean_holdout"], label="replication rate on clean_holdout")
        ax.errorbar(xs, r, yerr=ci_err(r, lo, hi), fmt="none", ecolor="#333", elinewidth=1.2, capsize=3)
        for x, (_, it), h in zip(xs, items, hi):
            ax.annotate(f"{it['replicated']}/{it['n']}", (x, h), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.axhline(0.05, color=C[3], ls="--", lw=1.5, label="chance (one-sided p < 0.05)")
        ax.set_xticks(xs, [a for a, _ in items])
        ax.set_title(ttl)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("share of discoveries replicated (Wilson 95% CI)")
    axes[0].legend(loc="upper right")
    rk = d["restating_known"]
    return fig, caption("research/reports/daily_mining_2026-11-12.json (H-MINE2: 495 effects, BH-FDR 10%)",
                        "discovery on llm_in_sample, confirmation once on clean_holdout",
                        f"{rk['discoveries']} of {d['overall']['n']} discoveries restate known rules; "
                        f"2 of {d['by_known']['contradicts']['n'] + d['by_known']['novel']['n']} others replicate")


# ------------------------------------------------------------------------- calibration


def _ml_prior_bins():
    return [(0.0, 0.30), (0.30, 0.45), (0.45, 0.60), (0.60, 1.0001)]


@figure("calibration", "prior_reliability",
        "Our priors were ~2.3x too optimistic (mean 0.45 vs 19% supported) yet ranked hypotheses well",
        "data/research/knowledge.json (calibration); hypotheses.yaml + hypotheses_results.jsonl")
def _ml_prior_reliability():
    import yaml

    k = rj("data/research/knowledge.json")["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    # left: protocol (strict) outcome, from the knowledge book
    ax = axes[0]
    xs, ys, lo, hi, ns = [], [], [], [], []
    for b in k["bins"]:
        n = int(b["n"])
        kk = int(round(num(b["hit_rate"]) * n))
        w = _ml_wilson(kk, n)
        xs.append(num(b["mean_prior"]))
        ys.append(kk / n)
        lo.append(w[0])
        hi.append(w[1])
        ns.append(n)
    xs, ys = np.array(xs), np.array(ys)
    ax.errorbar(xs, ys, yerr=ci_err(ys, lo, hi), fmt="o-", color=C[7], ecolor=C[7], capsize=3, lw=1.8, ms=6,
                label="strict (protocol) outcome")
    for x, y, h, n in zip(xs, ys, hi, ns):
        ax.annotate(f"n={n}", (x, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    st = k["strict"]
    ax.text(0.02, 0.97, f"n = {st['n']} decided\nmean prior {st['mean_prior']:.2f}, hit rate {st['hit_rate']:.2f}\n"
                        f"Brier {st['brier']:.3f} (base rate {st['brier_base_rate']:.3f})\nAUC {st['auc']:.2f}",
            transform=ax.transAxes, va="top", fontsize=10)
    ax.set_title("supported = holds in both segments")
    # right: per segment, from the registry (latest verdict per hypothesis and segment)
    H = yaml.safe_load(open(ROOT / "research" / "hypotheses.yaml", encoding="utf-8"))["hypotheses"]
    last = {}
    with open(ROOT / "research" / "hypotheses_results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                last[(r.get("id"), r.get("segment"))] = r.get("verdict")
    ax2 = axes[1]
    tot = {}
    for j, s in enumerate(SEG):
        pts = [(num(H[i].get("prior")), v == "supports") for (i, sg), v in last.items()
               if sg == s and i in H and v in ("supports", "contradicts") and np.isfinite(num(H[i].get("prior")))]
        tot[s] = (len(pts), sum(o for _, o in pts))
        bx, by, blo, bhi, bn = [], [], [], [], []
        for a, b in _ml_prior_bins():
            sel = [o for p, o in pts if a <= p < b]
            if not sel:
                continue
            n = len(sel)
            kk = int(sum(sel))
            w = _ml_wilson(kk, n)
            bx.append(np.mean([p for p, _ in pts if a <= p < b]) + (j - 0.5) * 0.012)
            by.append(kk / n)
            blo.append(w[0])
            bhi.append(w[1])
            bn.append(n)
        by = np.array(by)
        ax2.errorbar(bx, by, yerr=ci_err(by, blo, bhi), fmt="o-", color=SEG_COLOR[s], ecolor=SEG_COLOR[s],
                     capsize=3, lw=1.8, ms=6, label=f"{SEG_LABEL[s]}: {tot[s][1]}/{tot[s][0]} supported")
        for x, y, n in zip(bx, by, bn):
            ax2.annotate(f"{n}", (x, y), xytext=(8 if j else -8, 0), textcoords="offset points",
                         ha="left" if j else "right", va="center", fontsize=10, color=SEG_COLOR[s])
    ax2.set_title("per segment (latest decisive verdict)")
    for a in axes:
        a.plot([0, 1], [0, 1], color="#333", ls="--", lw=1.2, label="perfect calibration")
        a.set_xlim(0.1, 0.85)
        a.set_ylim(0, 1)
        a.set_xlabel("prior P(hypothesis holds), bin mean")
    axes[0].set_ylabel("observed share supported (Wilson 95% CI)")
    axes[0].legend(loc="lower right")
    ax2.legend(loc="upper left", fontsize=10)
    return fig, caption("data/research/knowledge.json (calibration), research/hypotheses.yaml (prior), "
                        "research/hypotheses_results.jsonl",
                        "strict = both segments; right: llm_in_sample and clean_holdout separately",
                        "bins [0, .30) [.30, .45) [.45, .60) [.60, 1]; inconclusive verdicts excluded")


# ----------------------------------------------------------------------------- budget


@figure("budget", "features_k",
        "Direction skill peaks at one feature and falls to -0.015 with 32 (in-sample); holdout CIs span 0",
        "decision_budget_2026-11-10.json (H-IC1)")
def _ml_budget_k():
    d = rj("decision_budget_2026-11-10.json")["H-IC1"]
    ks = [1, 2, 4, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for j, s in enumerate(SEG):
        x = np.array(ks, float) * (1 + (j - 0.5) * 0.06)
        v = np.array([num(d[s]["skill"][str(k)]) for k in ks])
        lo = np.array([num(d[s]["ci95"][str(k)][0]) for k in ks])
        hi = np.array([num(d[s]["ci95"][str(k)][1]) for k in ks])
        ax.errorbar(x, v, yerr=ci_err(v, lo, hi), fmt="o-", color=SEG_COLOR[s], ecolor=SEG_COLOR[s], capsize=3,
                    lw=1.8, ms=6, label=f"{SEG_LABEL[s]} ({d[s]['n_dates']} dates)")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    vi = d["llm_in_sample"]["skill"]
    ax.annotate(f"k = 1: {num(vi['1']):+.4f}", (1, num(vi["1"])), xytext=(1.08, -0.0095), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate(f"k = 32: {num(vi['32']):+.4f}", (32 * 0.97, num(vi["32"])), xytext=(9, -0.021), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlabel("number of features k (in order of in-sample importance)")
    ax.set_ylabel("next-day direction Brier skill vs base rate")
    ax.legend(loc="lower left")
    return fig, caption("research/reports/decision_budget_2026-11-10.json (H-IC1)",
                        f"llm_in_sample ({d['llm_in_sample']['n_obs']:,} symbol-days) and clean_holdout "
                        f"({d['clean_holdout']['n_obs']:,} symbol-days)",
                        "95% date-block bootstrap CI; holdout peak at k = 8, every CI includes 0")


@figure("budget", "latency_decay",
        "The 4h trend keeps 97% of its edge after 4 h; shock edges keep 61% after 1 h, 10% after 2 h",
        "decision_budget_2026-11-10.json (H-IC3)")
def _ml_budget_latency():
    d = rj("decision_budget_2026-11-10.json")["H-IC3"]["llm_in_sample"]
    lags = [0, 1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ylo, yhi = -0.8, 1.4
    for j, (leg, lab, col) in enumerate([("trend", "4h trend", C[2]), ("shock", "shock (1h)", C[4])]):
        x = np.array(lags, float) + (j - 0.5) * 0.12
        v = np.array([num(d[leg]["ratio"][str(l)]) for l in lags])
        lo = np.array([num(d[leg]["ratio_ci95"][str(l)][0]) for l in lags])
        hi = np.array([num(d[leg]["ratio_ci95"][str(l)][1]) for l in lags])
        lo_c, hi_c = np.clip(lo, ylo, yhi), np.clip(hi, ylo, yhi)
        ax.errorbar(x, v, yerr=ci_err(v, lo_c, hi_c), fmt="o-", color=col, ecolor=col, capsize=3, lw=1.8, ms=6,
                    label=f"{lab} ({d[leg]['n_dates']} dates)")
        for xx, l, h in zip(x, lo, hi):
            if l < ylo:
                ax.annotate("", (xx, ylo), xytext=(xx, ylo + 0.12), arrowprops=dict(arrowstyle="->", color=col))
            if h > yhi:
                ax.annotate("", (xx, yhi), xytext=(xx, yhi - 0.12), arrowprops=dict(arrowstyle="->", color=col))
        for xx, vv, l in zip(x, v, lags):
            if l in (1, 2, 4) and ((leg == "trend" and l == 4) or (leg == "shock" and l in (1, 2))):
                ax.annotate(f"{vv:.0%}", (xx, vv), xytext=(10, 8 if leg == "trend" else -16), textcoords="offset points",
                            fontsize=10.5, color=col, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.axhline(1, color="#333", lw=1, ls="--")
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(lags, [f"{l} h" for l in lags])
    ax.set_xlabel("execution delay after the signal")
    ax.set_ylabel("edge kept (delayed edge / edge at 0 h)")
    ax.legend(loc="lower left")
    return fig, caption("research/reports/decision_budget_2026-11-10.json (H-IC3)",
                        "llm_in_sample only (the holdout edge at 0 h is <= 0 for both signals, so the ratio is undefined)",
                        "95% bootstrap CI; whiskers clipped at the axis (arrows); shock edge half-life ~1.2 h")


@figure("budget", "window_length",
        "Short fitting windows never win: 90-day windows are worse for direction and vol in both segments",
        "decision_budget_2026-11-10.json (H-IC2)")
def _ml_budget_window():
    d = rj("decision_budget_2026-11-10.json")["H-IC2"]
    wins = ["90", "180", "365", "730"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (m, ttl, unit) in zip(axes, [("logit", "direction (logit, Brier)", "Brier"),
                                         ("har", "next-day vol (HAR, QLIKE)", "QLIKE")]):
        vals = {s: [num(d[s][m]["windows"][w]["diff_vs_expanding"]) for w in wins] for s in SEG}
        halfs = {s: [_ml_half(d[s][m]["windows"][w]["diff_vs_expanding"], d[s][m]["windows"][w]["t_nw"])
                     for w in wins] for s in SEG}
        _ml_segbars(ax, [f"{w} d" for w in wins], vals, halfs, fmt="{:+.4f}" if m == "logit" else "{:+.3f}")
        ax.set_title(ttl)
        ax.set_xlabel("rolling fitting window")
        ax.set_ylabel(f"{unit} difference vs expanding window (> 0 = worse)")
    b0, t0 = axes[0].get_ylim()
    axes[0].set_ylim(b0, t0 + 0.3 * (t0 - b0))
    axes[0].legend(loc="upper left")
    return fig, caption("research/reports/decision_budget_2026-11-10.json (H-IC2)",
                        "llm_in_sample and clean_holdout, 23 crypto perps",
                        "95% CI from Newey-West t; no window shorter than expanding wins anywhere")


# ================================================================ microstructure, spillover, simulator, MM, flow

# ----------------------------------------------------------------- micro / spillover / simulator / flow / mm
# Trade-level laws (H-MS*), multi-day stability (H-MD*), spillovers, population scaling, simulator,
# flow anomalies and live market-making P&L. Every micro/flow source is after the 2026-07-01 cutoff
# (clean_holdout) except the multi-day batch, which has both segments and is shown per segment.


def _mi_cls_color():
    return {"major": C[0], "alt": C[1], "tradfi": C[2]}


def _mi_short(sym):
    s = sym.replace("USDT", "")
    return s.replace("1000PEPE", "PEPE")


def _mi_params():
    return rj(DATA / "models" / "microstructure_laws" / "params_2026-11-10.json")


def _mi_wilson(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _mi_boot_ci_sum(x, B=2000, seed=7):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = rng.choice(x, size=(B, x.size), replace=True).sum(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ------------------------------------------------------------------------------------------------ micro
@figure("micro", "duration_mixture",
        "Trade arrivals are a 3-speed mixture, not Poisson: the exponential is rejected for 51 of 51 perps",
        "data/models/microstructure_laws/params_2026-11-10.json; microstructure_laws_2026-11-11.json")
def _mi_duration_mixture():
    P = _mi_params()
    rep = rj("microstructure_laws_2026-11-11.json")["H-MS1"]
    comp_c = [C[3], C[4], C[5]]
    comp_lab = ["fast (bursts)", "medium", "slow (gaps)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    sym = "BTCUSDT"
    p = P["symbols"][sym]["halves"]["test"]["durations"]["event"]["laws"]["expmix3"]["params"]
    w = np.array([p["w1"], p["w2"], p["w3"]])
    r = np.array([p["rate1"], p["rate2"], p["rate3"]])
    u = np.linspace(-4.5, 1.8, 600)  # log10 seconds
    d = 10.0 ** u
    comps = [w[k] * r[k] * np.exp(-r[k] * d) * d * math.log(10) for k in range(3)]
    mix = np.sum(comps, axis=0)
    mu = float(np.sum(w / r))
    expo = (1 / mu) * np.exp(-d / mu) * d * math.log(10)
    for k in range(3):
        a1.plot(u, comps[k], color=comp_c[k], lw=1.5, ls="--", label=f"{comp_lab[k]}: w {w[k]:.2f}, mean {1 / r[k] * 1000:.0f} ms")
    a1.plot(u, mix, color="#222", lw=2, label="3-component mixture")
    a1.plot(u, expo, color=C[7], lw=2, ls=":", label=f"exponential, same mean ({mu * 1000:.0f} ms)")
    a1.set_xlabel("time between arrival events (log10 seconds)")
    a1.set_ylabel("density per log10 unit")
    a1.set_title(f"{_mi_short(sym)}, TEST half (12-24 UTC)", fontsize=11)
    a1.legend(loc="upper left", fontsize=9, frameon=False)
    a1.set_ylim(0, max(mix.max(), expo.max()) * 1.45)

    # panel b: component weight vs component mean, all symbols
    cls = _mi_cls_color()
    for k in range(3):
        xs, ys = [], []
        for s, v in P["symbols"].items():
            q = v["halves"]["test"]["durations"]["event"]["laws"].get("expmix3", {}).get("params")
            if not q:
                continue
            xs.append(1000.0 / q[f"rate{k + 1}"])
            ys.append(q[f"w{k + 1}"])
        a2.scatter(xs, ys, s=30, color=comp_c[k], alpha=0.8, edgecolor="white", linewidth=0.5,
                   label=f"{comp_lab[k]} (median {np.median(xs):.0f} ms)")
    a2.set_xscale("log")
    a2.set_xlabel("component mean duration (ms, log scale)")
    a2.set_ylabel("component weight (share of arrivals)")
    a2.set_title("all 51 perps, TEST half", fontsize=11)
    a2.legend(loc="upper right", fontsize=9, frameon=False)
    a2.set_ylim(0, 1.0)
    wins = sum(rep["wins_by_class"][c]["expmix3"] for c in rep["wins_by_class"])
    a2.text(0.17, 0.03, f"exponential rejected (KS p<0.01): {rep['part1'][0]}/{rep['part1'][1]}\n"
            f"3-exp mixture wins held-out log-lik: {wins}/51\n(Burr XII or another heavy law: the rest)",
            transform=a2.transAxes, va="bottom", fontsize=10)

    return fig, caption("data/models/microstructure_laws/params_2026-11-10.json, microstructure_laws_2026-11-11.json (H-MS1)",
                        "clean_holdout (post-cutoff, 1 day: 2026-11-10)",
                        "fit 00-12 UTC, scored 12-24 UTC; arrival event = distinct ms")


@figure("micro", "size_tail",
        "Trade sizes are heavy-tailed (median tail index 1.76) but not a clean power law, and unstable",
        "microstructure_laws_2026-11-11.json")
def _mi_size_tail():
    R = rj("microstructure_laws_2026-11-11.json")["H-MS2"]
    ps = R["per_symbol"]
    cls = _mi_cls_color()
    order = ["major", "alt", "tradfi"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    rng = np.random.default_rng(3)
    a1.axhspan(1, 2, color=C[7], alpha=0.12, lw=0)
    a1.text(-0.45, 1.03, "registered range [1, 2]", fontsize=9, color="#555", ha="left")
    for i, c in enumerate(order):
        z = np.array([num(v["zeta"]) for v in ps.values() if v["class"] == c])
        nr = np.array([bool(v["not_rejected"]) for v in ps.values() if v["class"] == c])
        x = i + rng.uniform(-0.22, 0.22, z.size)
        a1.scatter(x[~nr], z[~nr], s=36, facecolor="white", edgecolor=cls[c], linewidth=1.5)
        a1.scatter(x[nr], z[nr], s=36, color=cls[c], edgecolor=cls[c])
        a1.hlines(np.median(z), i - 0.3, i + 0.3, color="#222", lw=2)
        a1.annotate(f"{np.median(z):.2f}", (i + 0.31, np.median(z)), fontsize=10, va="center")
    a1.set_xticks(range(3), [f"{c}\n(n={sum(v['class'] == c for v in ps.values())})" for c in order])
    a1.set_xlim(-0.5, 2.6)
    a1.set_ylabel("CCDF tail index zeta of print notional")
    a1.set_title("per perp, TEST half; bar = class median", fontsize=11)
    a1.scatter([], [], s=36, facecolor="white", edgecolor="#555", label="power law rejected (CSN p<=0.1)")
    a1.scatter([], [], s=36, color="#555", label="not rejected")
    a1.legend(loc="upper left", fontsize=9, frameon=False)
    a1.text(0.98, 0.98, f"in [1, 2]: {R['in_range'][0]}/51\nnot rejected: {R['not_rejected'][0]}/51\n"
            f"preferred to log-normal: {R['b'][0]}/51", transform=a1.transAxes, ha="right", va="top", fontsize=10)

    zf = np.array([num(v["zeta_fit"]) for v in ps.values()])
    zt = np.array([num(v["zeta"]) for v in ps.values()])
    cc = [cls[v["class"]] for v in ps.values()]
    a2.scatter(zf, zt, s=30, c=cc, edgecolor="white", linewidth=0.5)
    lim = [min(np.nanmin(zf), np.nanmin(zt)) - 0.1, max(np.nanmax(zf), np.nanmax(zt)) + 0.1]
    a2.plot(lim, lim, color=C[7], ls="--", lw=1.5, label="same tail index both halves")
    for c in order:
        a2.scatter([], [], s=30, color=cls[c], label=c)
    a2.set_xlim(lim)
    a2.set_ylim(lim)
    a2.set_xlabel("zeta, FIT half (00-12 UTC)")
    a2.set_ylabel("zeta, TEST half (12-24 UTC)")
    a2.set_title("same perp, two halves of one day", fontsize=11)
    a2.legend(loc="upper left", fontsize=9, frameon=False)
    a2.text(0.98, 0.02, f"Spearman {R['fit_test_spearman']:.2f}\nmedian |diff| {R['fit_test_median_abs_diff']:.2f}",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=10)
    return fig, caption("microstructure_laws_2026-11-11.json (H-MS2: contradicts)", "clean_holdout (post-cutoff, 1 day: 2026-11-10)",
                        "CSN x_min/MLE fit on print notional (USD); bootstrap p, Vuong vs truncated log-normal")


@figure("micro", "sign_acf",
        "Order signs have long memory: ACF decays as a power law, gamma 0.45-0.75 by class",
        "data/models/microstructure_laws/params_2026-11-10.json; microstructure_laws_2026-11-11.json")
def _mi_sign_acf():
    P = _mi_params()
    R = rj("microstructure_laws_2026-11-11.json")["H-MS3"]
    cls = _mi_cls_color()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    for sym in ["ETHUSDT", "LINKUSDT", "NVDAUSDT"]:
        v = P["symbols"][sym]
        so = v["halves"]["test"]["signs"]["order"]
        lags = np.array([int(k) for k in so["acf"]])
        acf = np.array([so["acf"][k] for k in so["acf"]])
        ok = acf > 0
        col = cls[v["class"]]
        a1.scatter(lags[ok], acf[ok], s=40, color=col, zorder=3)
        g, c0 = so["acf_power"]["gamma"], so["acf_power"]["c"]
        kk = np.logspace(0, 3, 50)
        a1.plot(kk, c0 * kk ** (-g), color=col, lw=1.75, label=f"{_mi_short(sym)} ({v['class']}): gamma {g:.2f}")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("lag (orders)")
    a1.set_ylabel("ACF of order signs (log scale)")
    a1.set_title("dots = empirical ACF, lines = fitted c k^-gamma (TEST)", fontsize=11)
    a1.legend(loc="lower left", fontsize=9.5, frameon=False)

    ps = R["per_symbol"]
    order = ["major", "alt", "tradfi"]
    rng = np.random.default_rng(5)
    a2.axhspan(0, 1, color=C[7], alpha=0.10, lw=0)
    for i, c in enumerate(order):
        g = np.array([num(v["gamma"]) for v in ps.values() if v["class"] == c and v.get("eligible")])
        x = i + rng.uniform(-0.22, 0.22, g.size)
        a2.scatter(x, g, s=34, color=cls[c], edgecolor="white", linewidth=0.5)
        m = R["gamma_median_by_class"][c]
        a2.hlines(m, i - 0.3, i + 0.3, color="#222", lw=2)
        a2.annotate(f"{m:.2f}", (i + 0.31, m), fontsize=10, va="center")
    a2.set_xticks(range(3), [f"{c}\n(n={sum(v['class'] == c and bool(v.get('eligible')) for v in ps.values())})" for c in order])
    a2.set_xlim(-0.5, 2.6)
    a2.set_ylabel("power-law exponent gamma (TEST)")
    a2.set_title("perps with >= 20k prints; bar = class median", fontsize=11)
    a2.text(0.02, 0.97, f"power law beats exponential on held-out lags\nand 0 < gamma < 1: {R['a'][0]}/{R['a'][1]}",
            transform=a2.transAxes, va="top", fontsize=10)
    a2.set_ylim(0, max(1.25, a2.get_ylim()[1]))
    return fig, caption("params_2026-11-10.json (signs.order), microstructure_laws_2026-11-11.json (H-MS3)",
                        "clean_holdout (post-cutoff, 1 day: 2026-11-10)",
                        "order = same-ms same-side prints merged; fit on FIT half, scored on TEST lags above the noise band")


@figure("micro", "hurst",
        "Signed order flow is mildly persistent: Hurst 0.53-0.54 with CI above 0.5, on 1 day and on 16 days",
        "microstructure_laws_2026-11-11.json; multiday_micro_batch3.json")
def _mi_hurst():
    R = rj("microstructure_laws_2026-11-11.json")["H-MS3"]
    M = rj("multiday_micro_batch3.json")
    cls = _mi_cls_color()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    ps = R["per_symbol"]
    order = ["major", "alt", "tradfi"]
    rng = np.random.default_rng(11)
    h = R["hurst"]
    a1.axhspan(h["lo"], h["hi"], color=C[7], alpha=0.25, lw=0, label=f"pooled mean {h['mean']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}]")
    a1.axhline(0.5, color="#222", ls="--", lw=1.5, label="0.5 = no memory")
    for i, c in enumerate(order):
        y = np.array([num(v["hurst"]) for v in ps.values() if v["class"] == c and v.get("eligible")])
        x = i + rng.uniform(-0.22, 0.22, y.size)
        a1.scatter(x, y, s=32, color=cls[c], edgecolor="white", linewidth=0.5)
    a1.set_xticks(range(3), order)
    a1.set_ylabel("Hurst exponent (DFA-1, 1 s signed notional)")
    a1.set_title(f"2026-11-10, per perp (n={h['n']}); {h['share_above'] * 100:.0f}% above 0.5", fontsize=11)
    a1.legend(loc="upper right", fontsize=9, frameon=False)
    lo, hi = a1.get_ylim()
    a1.set_ylim(lo, hi + 0.08)

    seg_days = M["segments_days"]
    pooled = M["results"]["pooled"]["H-MD3"]["hurst"]
    days = M["days"]
    y = [pooled["per_day_mean"][d] for d in days]
    seg = ["llm_in_sample" if d in seg_days["llm_in_sample"] else "clean_holdout" for d in days]
    x = np.arange(len(days))
    a2.axhspan(pooled["lo"], pooled["hi"], color=C[7], alpha=0.25, lw=0,
               label=f"16-day mean {pooled['mean']:.3f} [{pooled['lo']:.3f}, {pooled['hi']:.3f}]")
    a2.axhline(0.5, color="#222", ls="--", lw=1.5)
    for s in ["llm_in_sample", "clean_holdout"]:
        m = np.array(seg) == s
        a2.scatter(x[m], np.array(y)[m], s=50, color=SEG_COLOR[s], label=SEG_LABEL[s], zorder=3)
    a2.set_xticks(x, [d[2:] for d in days], rotation=60, ha="right", fontsize=9.5)
    a2.set_ylabel("mean Hurst over symbol-days")
    a2.set_title("per day, perps with >= 20k prints", fontsize=11)
    a2.legend(loc="upper left", fontsize=9, frameon=False)
    a2.set_ylim(0.49, max(y) + 0.03)
    return fig, caption("microstructure_laws_2026-11-11.json (H-MS3), multiday_micro_batch3.json (H-MD3)",
                        "left: clean_holdout (post-cutoff, 1 day); right: llm_in_sample 6 days + clean_holdout 10 days",
                        "CIs: bootstrap over symbols (left) and days (right)")


@figure("micro", "multiday_icc",
        "Per-symbol microstructure traits are mostly day-to-day noise: ICC 0.23-0.31, a pass needs > 0.5",
        "multiday_micro_batch3.json; data/research/micro_days/*/laws/*.json")
def _mi_multiday_icc():
    M = rj("multiday_micro_batch3.json")
    traits = ["gamma", "acf1", "zeta", "mix_weights"]
    tlab = {"gamma": "sign ACF\ngamma", "acf1": "sign ACF\nlag 1", "zeta": "size tail\nzeta", "mix_weights": "duration mix\nweights"}
    segs = ["llm_in_sample", "clean_holdout", "pooled"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for j, s in enumerate(segs):
        T = M["results"][s]["H-MD1"]["traits"]
        est = [T[t]["icc"] for t in traits]
        lo = [T[t]["lo"] for t in traits]
        hi = [T[t]["hi"] for t in traits]
        x = np.arange(len(traits)) + (j - 1) * 0.22
        nd = M["results"][s]["H-MD1"]["n_days"]
        a1.errorbar(x, est, yerr=ci_err(est, lo, hi), fmt="o", ms=7, color=SEG_COLOR[s], capsize=3, lw=1.5,
                    label=f"{s} ({nd} days)")
    a1.axhline(0.5, color="#222", ls="--", lw=1.5)
    a1.text(3.45, 0.51, "pass: ICC > 0.5", ha="right", va="bottom", fontsize=9.5)
    a1.axhline(0.3, color="#222", ls=":", lw=1.2)
    a1.text(3.45, 0.31, "and CI low > 0.3", ha="right", va="bottom", fontsize=9.5)
    a1.set_xticks(range(len(traits)), [tlab[t] for t in traits])
    a1.set_ylabel("ICC (share of variance between symbols)")
    a1.set_ylim(0, 0.75)
    a1.set_title("95% bootstrap CI over symbols", fontsize=11)
    a1.legend(loc="upper left", fontsize=9, frameon=False)

    # panel b: sign gamma per symbol across days
    days = M["days"]
    insample = set(M["segments_days"]["llm_in_sample"])
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT", "NEARUSDT",
            "XAUUSDT", "NVDAUSDT", "TSLAUSDT"]
    base = DATA / "research" / "micro_days"
    rows = {}
    for sym in syms:
        vals = []
        for d in days:
            p = base / d / "laws" / f"{sym}.json"
            if not p.exists():
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    g = num(json.load(fh)["tests"]["sign_order"]["gamma_test"])
            except Exception:
                continue
            if np.isfinite(g):
                vals.append((g, d in insample))
        if vals:
            rows[sym] = vals
    syms = [s for s in syms if s in rows]
    rng = np.random.default_rng(2)
    for i, sym in enumerate(syms):
        g = np.array([v[0] for v in rows[sym]])
        ins = np.array([v[1] for v in rows[sym]])
        y = i + rng.uniform(-0.18, 0.18, g.size)
        a2.scatter(g[ins], y[ins], s=22, color=SEG_COLOR["llm_in_sample"], alpha=0.85)
        a2.scatter(g[~ins], y[~ins], s=22, color=SEG_COLOR["clean_holdout"], alpha=0.85)
        a2.vlines(np.mean(g), i - 0.32, i + 0.32, color="#222", lw=2)
    a2.scatter([], [], s=22, color=SEG_COLOR["llm_in_sample"], label="llm_in_sample day")
    a2.scatter([], [], s=22, color=SEG_COLOR["clean_holdout"], label="clean_holdout day")
    a2.plot([], [], color="#222", lw=2, label="symbol mean")
    a2.set_yticks(range(len(syms)), [_mi_short(s) for s in syms])
    a2.invert_yaxis()
    a2.set_xlabel("order-sign ACF exponent gamma (TEST half of each day)")
    a2.set_title("one dot per day: spread within a symbol >= between symbols", fontsize=11)
    a2.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    a2.set_xlim(right=max(a2.get_xlim()[1], 1.68))
    return fig, caption("multiday_micro_batch3.json (H-MD1), data/research/micro_days/<day>/laws/<SYM>.json",
                        "llm_in_sample 6 days + clean_holdout (post-cutoff, 10 days); pooled 16",
                        "ICC(1), symbol as group; 12 illustrative symbols on the right")


@figure("micro", "multiday_powerlaw_share",
        "A power-law sign ACF wins on 79% of symbol-days, just under the registered 80%",
        "multiday_micro_batch3.json")
def _mi_multiday_powerlaw():
    M = rj("multiday_micro_batch3.json")
    days = M["days"]
    ins = set(M["segments_days"]["llm_in_sample"])
    pd_ = M["results"]["pooled"]["H-MD3"]["per_day"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(days))
    for s in ["llm_in_sample", "clean_holdout"]:
        idx = [i for i, d in enumerate(days) if (d in ins) == (s == "llm_in_sample")]
        k = np.array([pd_[days[i]]["k"] for i in idx])
        n = np.array([pd_[days[i]]["n"] for i in idx])
        sh = k / n
        lo, hi = zip(*[_mi_wilson(a, b) for a, b in zip(k, n)])
        ax.errorbar(x[idx], sh, yerr=ci_err(sh, lo, hi), fmt="o", ms=7, color=SEG_COLOR[s], capsize=3, lw=1.5,
                    label=SEG_LABEL[s])
    ax.axhline(0.8, color="#222", ls="--", lw=1.5)
    ax.text(len(days) - 0.5, 0.805, "registered criterion 0.80", ha="right", va="bottom", fontsize=10)
    R = M["results"]
    txt = []
    for s in ["llm_in_sample", "clean_holdout", "pooled"]:
        r = R[s]["H-MD3"]
        lo, hi = _mi_wilson(r["k"], r["n"])
        txt.append(f"{s}: {r['k']}/{r['n']} = {r['share']:.3f} [{lo:.2f}, {hi:.2f}]")
    ax.text(0.01, 0.03, "\n".join(txt), transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xticks(x, days, rotation=45, ha="right")
    ax.set_ylabel("share of symbol-days passing")
    ax.set_ylim(0.12, 1.05)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False, ncol=2)
    return fig, caption("multiday_micro_batch3.json (H-MD3, interim)", "llm_in_sample 6 days + clean_holdout (post-cutoff, 10 days)",
                        "pass = power law preferred on held-out lags and 0 < gamma < 1; symbol-days with >= 20k prints; Wilson 95% CIs")


# -------------------------------------------------------------------------------------------- spillover
def _mi_sym_color():
    return {"ETHUSDT": C[0], "SOLUSDT": C[1], "BTCUSDT": C[2], "DOGEUSDT": C[3]}


@figure("spillover", "net_transmitters",
        "ETH and SOL transmit volatility to the rest of crypto on 15 of 15 days; BTC only on 9",
        "multiday_micro_batch3.json")
def _mi_net_transmitters():
    M = rj("multiday_micro_batch3.json")
    R = M["results"]["pooled"]["H-MD5"]
    per = R["per_day"]
    days = list(per.keys())
    ins = set(M["segments_days"]["llm_in_sample"])
    sc = _mi_sym_color()
    syms = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "BTCUSDT"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(days))
    for j, s in enumerate(syms):
        est, lo, hi, xs = [], [], [], []
        for i, d in enumerate(days):
            v = per[d]["net_day"].get(s)
            ci = per[d]["net_ci_day"].get(s)
            if v is None or ci is None:
                continue
            est.append(v)
            lo.append(ci[0])
            hi.append(ci[1])
            xs.append(i + (j - 1.5) * 0.18)
        r = R[s]
        ax.errorbar(xs, est, yerr=ci_err(est, lo, hi), fmt="o", ms=5.5, color=sc[s], capsize=2, lw=1.3,
                    label=f"{_mi_short(s)}: net > 0 on {r['k_pos']}/{r['n']} days, mean {r['mean_net']:+.1f}")
    ax.axhline(0, color="#222", lw=1.2)
    n_in = sum(d in ins for d in days)
    ax.axvline(n_in - 0.5, color="#555", ls="--", lw=1.2)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    ax.text(n_in - 0.45, lo_ + 0.02 * (hi_ - lo_), " knowledge cutoff 2026-07-01 ->", va="bottom", fontsize=9.5, color="#555")
    ax.set_xticks(x, days, rotation=45, ha="right")
    ax.set_ylabel("net vol spillover (to - from, % of FEV)")
    ax.legend(loc="upper center", ncol=2, fontsize=9.5, frameon=False)
    return fig, caption("multiday_micro_batch3.json (H-MD5, interim)",
                        f"llm_in_sample {n_in} days + clean_holdout (post-cutoff, {len(days) - n_in} days)",
                        "Diebold-Yilmaz generalized FEVD of log 1 m RV, 15 tokens; 30-min block-bootstrap 95% CI")


@figure("spillover", "leadlag_fine",
        "No exploitable lead: alts co-move with ETH, SOL and BTC within +-20 ms; 0 of 39 peak at +0.1-30 s",
        "book_spillover_2026-11-11.json")
def _mi_leadlag_fine():
    D = rj("book_spillover_2026-11-11.json")
    pairs = D["pairs"]
    crypto = D["universe"]["crypto"]
    lags = np.array([-500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500])
    sc = _mi_sym_color()
    leaders = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.axvspan(-20, 20, color=C[7], alpha=0.15, lw=0, label="+-20 ms (timestamp jitter)")
    counts = {}
    for ld in leaders:
        R = []
        cnt = {"neg": 0, "zero": 0, "pos": 0}
        for t in crypto:
            if t == ld or ld not in pairs.get(t, {}):
                continue
            st = pairs[t][ld]["test"]
            if st.get("rho_fine") is not None:
                R.append(st["rho_fine"])
            pl_ = num(st.get("peak_lag_s"))
            if not np.isfinite(pl_):
                continue
            if pl_ < 0:
                cnt["neg"] += 1
            elif pl_ == 0:
                cnt["zero"] += 1
            else:
                cnt["pos"] += 1
        counts[ld] = cnt
        R = np.array(R, float)
        med = np.nanmedian(R, axis=0)
        q1, q3 = np.nanpercentile(R, 25, axis=0), np.nanpercentile(R, 75, axis=0)
        a1.fill_between(lags, q1, q3, color=sc[ld], alpha=0.15, lw=0)
        a1.plot(lags, med, color=sc[ld], lw=2, marker="o", ms=4, label=f"{_mi_short(ld)} leads (median, IQR band)")
    a1.set_xscale("symlog", linthresh=10)
    a1.set_xticks([-500, -100, -20, 0, 20, 100, 500], ["-500", "-100", "-20", "0", "20", "100", "500"])
    a1.set_xlabel("lag l (ms): corr(leader at t, target at t + l)")
    a1.set_ylabel("Hayashi-Yoshida correlation")
    a1.set_title("TEST half, crypto targets", fontsize=11)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9, frameon=False, ncol=2)

    cats = [("neg", "peak before 0 (target leads)", plt.get_cmap(SEQ_CMAP)(0.35)),
            ("zero", "peak at lag 0", plt.get_cmap(SEQ_CMAP)(0.85)),
            ("pos", "peak > 0 (leader leads)", C[3])]
    y = np.arange(len(leaders))
    left = np.zeros(len(leaders))
    for key, lab, col in cats:
        v = np.array([counts[ld][key] for ld in leaders], float)
        a2.barh(y, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.6, label=lab)
        for yi, (l0, vi) in enumerate(zip(left, v)):
            if vi >= 2:
                a2.text(l0 + vi / 2, yi, f"{int(vi)}", ha="center", va="center", fontsize=10,
                        color="white" if key == "zero" else "#111")
        left += v
    a2.set_yticks(y, [_mi_short(l) for l in leaders])
    a2.invert_yaxis()
    a2.set_xlabel("crypto targets (count)")
    a2.set_title("where the coarse-grid HY peak sits", fontsize=11)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9, frameon=False, ncol=1)
    a2.grid(axis="y", visible=False)
    return fig, caption("book_spillover_2026-11-11.json (H-MS6: contradicts)", "clean_holdout (post-cutoff, 1 day: 2026-11-10, TEST 12-24 UTC)",
                        "trade prices, same-ms HY estimator; coarse grid 0, +-0.1..60 s")


# -------------------------------------------------------------------------------------------- simulator
@figure("simulator", "population_scaling",
        "Activity W predicts order rate and price impact across 51 perps, with slopes off the KO 2/3 law",
        "data/research/micro_days/2026-11-10/laws/*.json; population_simulator_2026-11-11.json")
def _mi_population_scaling():
    P = rj("population_simulator_2026-11-11.json")
    ko = P["H-MS7"]["ko"]
    base = DATA / "research" / "micro_days" / "2026-11-10" / "laws"
    cls = _mi_cls_color()
    W, rate, lam, cl = [], [], [], []
    for p in sorted(base.glob("*.json")):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        a = (d.get("activity") or {}).get("fit") or {}
        w_, r_, l_, s_ = num(a.get("W")), num(a.get("orders_per_s")), num(a.get("lambda_mid")), num(a.get("sigma_day"))
        if not (np.isfinite(w_) and w_ > 0):
            continue
        W.append(w_)
        rate.append(r_ * 86400 if np.isfinite(r_) else np.nan)
        lam.append(l_ / (s_ / 1e4) ** 2 if (np.isfinite(l_) and l_ > 0 and np.isfinite(s_) and s_ > 0) else np.nan)
        cl.append(d.get("class", "alt"))
    W, rate, lam = (np.array(v, float) for v in (W, rate, lam))
    lw = np.log10(W)
    fig, axs = plt.subplots(1, 3, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1, 0.9]})

    def _panel(ax, y, key, ylab, ttl, loc):
        m = np.isfinite(y) & (y > 0)
        ly = np.log10(y[m])
        x = lw[m]
        b, a0 = np.polyfit(x, ly, 1)
        for c in ["major", "alt", "tradfi"]:
            mm = np.array(cl)[m] == c
            ax.scatter(10 ** x[mm], 10 ** ly[mm], s=26, color=cls[c], edgecolor="white", linewidth=0.5, label=c)
        xx = np.linspace(x.min(), x.max(), 20)
        ax.plot(10 ** xx, 10 ** (a0 + b * xx), color="#222", lw=2, label=f"fit {b:+.2f}")
        pred = ko[key]["prediction"]
        ax.plot(10 ** xx, 10 ** (ly.mean() + pred * (xx - x.mean())), color=C[7], lw=2, ls="--",
                label=f"KO {pred:+.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("activity W = sigma_day x daily USD volume")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        ax.legend(loc=loc, fontsize=9, frameon=False)
        return b, int(m.sum())

    b1, n1 = _panel(axs[0], rate, "orders per day vs W (bets ~ W^2/3)", "taker orders per day", "order rate", "upper left")
    b2, n2 = _panel(axs[1], lam, "lambda / sigma^2 vs W (~ W^-2/3)", "lambda / sigma_day^2 (bp per USD 1k)", "price impact", "upper right")

    ax = axs[2]
    pr = P["H-MS7"]["primary"]
    sec = P["secondary"]
    rows = [("order rate", pr["rate"]["coef"][1], pr["rate"]["se"][1], ko["orders per day vs W (bets ~ W^2/3)"]["prediction"]),
            ("depth 25 bp", sec["log_depth25"]["coef"][1], sec["log_depth25"]["se"][1], np.nan),
            ("lambda / sigma^2", ko["lambda / sigma^2 vs W (~ W^-2/3)"]["slope"], ko["lambda / sigma^2 vs W (~ W^-2/3)"]["se"],
             ko["lambda / sigma^2 vs W (~ W^-2/3)"]["prediction"])]
    y = np.arange(len(rows))
    est = np.array([r[1] for r in rows])
    se = np.array([r[2] for r in rows])
    ax.errorbar(est, y, xerr=1.96 * se, fmt="o", ms=7, color="#222", capsize=3, lw=1.5, label="estimate, 95% CI")
    kp = np.array([r[3] for r in rows])
    ax.scatter(kp, y, marker="D", s=50, facecolor="white", edgecolor=C[7], linewidth=2, zorder=3, label="KO prediction")
    for yi, e in zip(y, est):
        ax.annotate(f"{e:+.2f}", (e, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("slope on log W")
    ax.set_title("slopes on log W", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_ylim(2.5, -1.4)
    return fig, caption("micro_days/2026-11-10/laws/<SYM>.json (activity, FIT half); population_simulator_2026-11-11.json (H-MS7)",
                        "clean_holdout (post-cutoff, 1 day: 2026-11-10, FIT 00-12 UTC)",
                        f"right: rate and depth slopes include class dummies; left two recomputed, W-only OLS (n={n1}, {n2}); report: {ko['orders per day vs W (bets ~ W^2/3)']['slope']:+.2f}, "
                        f"{ko['lambda / sigma^2 vs W (~ W^-2/3)']['slope']:+.2f}")


@figure("simulator", "facts_inside",
        "Simulator v1 hits more stylised facts than v0 in every class; majors get 3 of 5, not 4",
        "population_simulator_2026-11-11.json")
def _mi_facts_inside():
    P = rj("population_simulator_2026-11-11.json")["H-MS8"]
    classes = ["major", "alt", "tradfi"]
    gens = [("v0", "v0 (per-symbol FlowSim)", C[0]), ("v1", "v1 (event-level)", C[1]), ("pop", "population-only v1", C[2])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(len(classes))
    for j, (g, lab, col) in enumerate(gens):
        v = np.array([P["counts"][c][g] for c in classes], float)
        bars = a1.bar(x + (j - 1) * 0.26, v, width=0.24, color=col, label=lab, edgecolor="white", linewidth=1)
        if g in ("v0", "v1"):
            lo, hi = [], []
            for c in classes:
                h = np.array(P["boot_counts"][c][g], float)
                cdf = np.cumsum(h) / h.sum()
                lo.append(int(np.searchsorted(cdf, 0.025)))
                hi.append(int(np.searchsorted(cdf, 0.975)))
            a1.errorbar(x + (j - 1) * 0.26, v, yerr=ci_err(v, lo, hi), fmt="none", ecolor="#333", capsize=3, lw=1.2)
        for b_, vv in zip(bars, v):
            a1.text(b_.get_x() + b_.get_width() / 2, 0.08 if vv > 0 else 0.05, f"{vv:.0f}", ha="center", va="bottom",
                    fontsize=10, color="white" if vv > 0 else "#222", fontweight="bold")
    a1.axhline(4, color="#222", ls="--", lw=1.5)
    a1.text(-0.45, 4.08, "majors need 4/5", fontsize=9.5)
    a1.set_xticks(x, classes)
    a1.set_ylabel("facts inside the TEST envelope (of 5)")
    a1.set_ylim(0, 5.4)
    a1.set_title("bars: class medians; whiskers: 95% symbol bootstrap", fontsize=11)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, frameon=False, ncol=2)

    facts = ["kurtosis", "absr_acf", "flow_acf", "impact", "spread"]
    flab = ["kurtosis", "|r| ACF", "flow ACF", "impact", "spread"]
    ylabs, M = [], []
    for c in classes:
        for g in ("v0", "v1"):
            ylabs.append(f"{c} {g}")
            M.append([P["own_envelope_share"][c][g][f] for f in facts])
    heatmap(a2, np.array(M) * 100, flab, ylabs, cmap=SEQ_CMAP, fmt="{:.0f}", cbar_label="% of symbols inside own envelope",
            vmin=0, vmax=100)
    a2.set_title("per fact: share of symbols inside their own envelope", fontsize=11)
    return fig, caption("population_simulator_2026-11-11.json (H-MS8: contradicts)", "clean_holdout (post-cutoff, 1 day: 2026-11-10, TEST 12-24 UTC)",
                        "envelope = 2.5-97.5% of the class median under a 1 h block bootstrap; 10 paths per symbol")


@figure("simulator", "mm_ranking",
        "Simulated paths cannot rank market-making settings: Spearman -0.09 (v0) and 0.30 (v1) vs real replay",
        "simulator_2026-11-11.json; population_simulator_2026-11-11.json")
def _mi_mm_ranking():
    S4 = rj("simulator_2026-11-11.json")["sim4"]
    RP = rj("population_simulator_2026-11-11.json")["replay"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    rx = np.array([r["real_bps"] for r in S4["table"]])
    sx = np.array([r["sim_bps"] for r in S4["table"]])
    a1.scatter(rx, sx, s=34, color=C[0], edgecolor="white", linewidth=0.5, label=f"36 parameter sets, {S4['symbols']} perps")
    a1.set_title(f"v0: Spearman {S4['spearman']:+.2f}, 95% CI [{S4['ci'][0]:+.2f}, {S4['ci'][1]:+.2f}]", fontsize=11)
    rows = RP["rows"][2:]
    ry = np.array([r["real_bps"] for r in rows])
    vy = np.array([r["v1_bps"] for r in rows])
    a2.scatter(ry, vy, s=34, color=C[1], edgecolor="white", linewidth=0.5, label=f"36 parameter sets, {RP['symbols']} perps")
    lv, bs = RP["live"], RP["best"]
    a2.scatter([lv["real_bps"]], [lv["v1_bps"]], s=90, marker="*", color="#222", zorder=4, label="live mm_multi set")
    a2.annotate(f"live: real {lv['real_bps']:+.1f} bp, v1 {lv['v1_bps']:+.1f} bp", (lv["real_bps"], lv["v1_bps"]),
                xytext=(-10, -16), textcoords="offset points", ha="right", fontsize=9.5)
    a2.scatter([bs["real_bps"]], [bs["v1_bps"]], s=70, marker="s", facecolor="none", edgecolor="#222", lw=1.5, zorder=4,
               label=f"best on real ({bs['real_bps']:+.1f} bp; v1 {bs['v1_bps']:+.1f})")
    a2.set_title(f"v1: Spearman {RP['spearman_bps']:+.2f} (turnover {RP['spearman_turnover']:.3f})", fontsize=11)
    for ax, (x, y) in ((a1, (rx, sx)), (a2, (ry, vy))):
        lo = min(x.min(), y.min()) - 1
        hi = max(x.max(), y.max()) + 1
        ax.plot([lo, hi], [lo, hi], color=C[7], ls="--", lw=1.5, label="simulated = real")
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_xlabel("real replay: net bp per turnover")
        ax.set_ylabel("simulated paths: net bp per turnover")
        ax.legend(loc="upper left" if ax is a1 else "lower right", fontsize=9, frameon=False)
    return fig, caption("simulator_2026-11-11.json (H-SIM4: contradicts), population_simulator_2026-11-11.json (replay, descriptive)",
                        "clean_holdout (post-cutoff, 2026-11-08..10 for v0; TEST half of 2026-11-10 for v1)",
                        "same 36-set grid (a, b, take-profit, requote); fills from the touch fill model")


# ------------------------------------------------------------------------------------------------- flow
@figure("flow", "detector_forward",
        "Flow anomalies barely predict: |z|>4 gives 1.14x forward |return| at 5 m; top-1% 1.48x, CI spans 1",
        "flow_anomaly_2026-11-10.json; data/research/forward_tests/status.json")
def _mi_flow_detector():
    F = rj("flow_anomaly_2026-11-10.json")
    st = rj(DATA / "research" / "forward_tests" / "status.json")
    of6 = next((t for t in st.get("tests", []) if t.get("hypothesis") == "H-OF6"), None)
    q = F["exploratory_quantile"]["0.99"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    hz = ["5", "15"]
    xs = np.arange(2)
    ser = [
        ("|z| > 4 (registered, H-OF1)", C[0],
         [F["H-OF1"]["horizons"][h]["pooled"]["ratio"] for h in hz],
         [F["H-OF1"]["horizons"][h]["pooled"]["ci95"] for h in hz],
         F["H-OF1"]["horizons"]["5"]["pooled"]["n"]),
        ("top 1% |z| (exploratory, same sample)", C[1],
         [q["H-OF1"][h]["ratio"] for h in hz], [q["H-OF1"][h]["ci95"] for h in hz], q["H-OF1"]["5"]["n"]),
    ]
    for j, (lab, col, est, ci, n) in enumerate(ser):
        x = xs + (j - 1) * 0.22
        a1.errorbar(x, est, yerr=ci_err(est, [c[0] for c in ci], [c[1] for c in ci]), fmt="o", ms=8, color=col, capsize=4, lw=1.75,
                    label=f"{lab}, n={n:,}")
        for xi, e in zip(x, est):
            a1.annotate(f"{e:.2f}x", (xi, e), xytext=(-8 if j == 0 else 8, 0), textcoords="offset points",
                        ha="right" if j == 0 else "left", va="center", fontsize=10)
    if of6:
        e, ci = of6["est"], of6["ci95"]
        a1.errorbar([0.42], [e], yerr=ci_err([e], [ci[0]], [ci[1]]), fmt="D", ms=7, color=C[2], capsize=4, lw=1.75,
                    label=f"top 1% forward (H-OF6 interim), n={of6['n_events']}")
        a1.annotate(f"{e:.2f}x", (0.42, e), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10)
    a1.axhline(1, color="#222", lw=1.2)
    a1.axhline(1.5, color="#222", ls="--", lw=1.2)
    a1.text(1.48, 1.64, "H-OF6 needs 1.5x", ha="right", va="bottom", fontsize=9.5)
    a1.set_xticks(xs, ["next 5 min", "next 15 min"])
    a1.set_xlim(-0.6, 1.55)
    a1.set_ylabel("forward |return| ratio vs same symbol-hour")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, frameon=False)

    ser2 = [("|z| > 4", C[0], [F["H-OF2"]["horizons"][h]["sign_rate"] for h in hz],
             [F["H-OF2"]["horizons"][h]["sign_rate_ci95"] for h in hz]),
            ("top 1% |z|", C[1], [q["H-OF2"][h]["sign_rate"] for h in hz], [q["H-OF2"][h]["sign_rate_ci95"] for h in hz])]
    for j, (lab, col, est, ci) in enumerate(ser2):
        x = xs + (j - 0.5) * 0.22
        a2.errorbar(x, est, yerr=ci_err(est, [c[0] for c in ci], [c[1] for c in ci]), fmt="o", ms=8, color=col, capsize=4, lw=1.75,
                    label=lab)
        for xi, e in zip(x, est):
            a2.annotate(f"{e:.3f}", (xi, e), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=10)
    a2.axhline(0.5, color="#222", ls="--", lw=1.5)
    a2.text(1.45, 0.501, "coin flip", ha="right", va="bottom", fontsize=9.5)
    a2.set_xticks(xs, ["next 5 min", "next 15 min"])
    a2.set_xlim(-0.6, 1.5)
    a2.set_ylabel("share of returns in the flow direction")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, frameon=False, ncol=2)
    return fig, caption("flow_anomaly_2026-11-10.json (H-OF1/H-OF2), data/research/forward_tests/status.json (H-OF6)",
                        "clean_holdout (post-cutoff, scored minutes 2026-11-08 17:53 to 11-10 22:29 UTC; forward 2026-11-11..12)",
                        "signed-imbalance robust z vs trailing 1440 1 m bars; 95% block-bootstrap CIs")


@figure("flow", "firing_rate",
        "A |z|>4 flow rule is not rare: it fires on 22% of minutes, against 0.006% for a Gaussian",
        "flow_anomaly_2026-11-10.json")
def _mi_flow_firing():
    F = rj("flow_anomaly_2026-11-10.json")
    fr = F["firing_rate_z4"]
    per = np.array(list(F["sample"]["firing_rate_per_symbol"].values()), float) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    a1.hist(per, bins=np.arange(0, per.max() + 2.5, 2.5), color=C[0], edgecolor="white", linewidth=1)
    a1.axvline(fr["pooled"] * 100, color="#222", lw=2, label=f"pooled {fr['pooled'] * 100:.1f}%")
    a1.axvline(fr["median_symbol"] * 100, color="#222", lw=1.5, ls="--", label=f"median symbol {fr['median_symbol'] * 100:.1f}%")
    a1.axvspan(fr["p10_symbol"] * 100, fr["p90_symbol"] * 100, color=C[7], alpha=0.15, lw=0,
               label=f"p10-p90 {fr['p10_symbol'] * 100:.0f}-{fr['p90_symbol'] * 100:.0f}%")
    a1.set_xlabel("share of scored minutes with |imbalance z| > 4 (%)")
    a1.set_ylabel("symbols (count)")
    a1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    a1.set_title(f"per symbol (n={per.size})", fontsize=11)
    a1.legend(loc="upper right", fontsize=9.5, frameon=False)

    sh = F["sample"]["anomaly_share_by_feature"]
    keys = ["imbalance", "volume", "trades", "spread", "abs_ret", "depth"]
    lab = ["signed imbalance", "volume", "trade count", "spread", "|return|", "depth"]
    v = np.array([sh[k] for k in keys]) * 100
    y = np.arange(len(keys))
    bars = a2.barh(y, v, color=C[0], height=0.6)
    annotate_barh(a2, bars, fmt="{:.1f}%")
    gauss = 2 * 0.5 * math.erfc(4 / math.sqrt(2)) * 100
    a2.axvline(gauss, color=C[3], lw=2, ls="--", label=f"Gaussian |z|>4: {gauss:.4f}%")
    a2.set_yticks(y, lab)
    a2.invert_yaxis()
    a2.set_xlabel("share of minutes with |z| > 4 (%)")
    a2.set_title("by feature (pooled)", fontsize=11)
    a2.set_xlim(0, v.max() * 1.25)
    a2.legend(loc="lower right", fontsize=9.5, frameon=False)
    a2.grid(axis="y", visible=False)
    return fig, caption("flow_anomaly_2026-11-10.json", "clean_holdout (post-cutoff, scored minutes 2026-11-08 17:53 to 11-10 22:29 UTC, with collector gaps)",
                        f"{F['sample']['minutes_scored']:,} scored symbol-minutes; robust z = (x - median) / 1.4826 MAD")


# --------------------------------------------------------------------------------------------------- mm
def _mi_mm_execs(*runs):
    out = {}
    for r in runs:
        p = DATA / "mm" / "analysis" / r / "executions.json"
        with open(p, encoding="utf-8") as fh:
            for x in json.load(fh):
                if x.get("execType") == "Trade" and x.get("symbol") != "SPYUSDT":  # SPY = the user's own position
                    out[x["execId"]] = x
    return sorted(out.values(), key=lambda x: int(x["execTime"]))


def _mi_fifo(E):
    """FIFO round trips per symbol, as scripts/mm_analyze.py: net of prorated fees, tagged by the exit fill's maker flag."""
    from collections import defaultdict, deque
    book = defaultdict(deque)
    trips = []
    for r in E:
        q = num(r["execQty"])
        sgn = 1 if r["side"] == "Buy" else -1
        px = num(r["execPrice"])
        fp = num(r["execFee"], 0.0) / q if q else 0.0
        dq = book[r["symbol"]]
        while q > 1e-12 and dq and np.sign(dq[0][0]) != sgn:
            lq, lpx, lts, lf = dq[0]
            m = min(q, abs(lq))
            pnl = (px - lpx) * m if lq > 0 else (lpx - px) * m
            trips.append({"sym": r["symbol"], "net": pnl - (lf + fp) * m, "maker": bool(r["isMaker"]), "ts": int(r["execTime"])})
            q -= m
            if m >= abs(lq) - 1e-12:
                dq.popleft()
            else:
                dq[0] = (lq - np.sign(lq) * m, lpx, lts, lf)
        if q > 1e-12:
            dq.append((sgn * q, px, int(r["execTime"]), fp))
    return trips


@figure("mm", "pnl_by_exit",
        "Live MM: passive exits made money (87% winners), market-order exits gave most of it back (14%)",
        "data/mm/analysis/mainnet-*/executions.json")
def _mi_mm_pnl_by_exit():
    runs = [("Nov 8 day\n3 symbols", ["mainnet-20261108T2258"]),
            ("Nov 8-9 night\n42 symbols", ["mainnet-20261109T0006", "mainnet-20261109T0817"]),
            ("Nov 11-12 NVDA\nmicro (last hour)", ["mainnet-20261112T0050"])]
    kinds = [(True, "passive (maker) exit", C[0]), (False, "market-order (taker) exit", C[1])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(runs))
    night = None
    for j, (mk, lab, col) in enumerate(kinds):
        est, lo, hi, ann = [], [], [], []
        for name, dirs in runs:
            T = _mi_fifo(_mi_mm_execs(*dirs))
            if "night" in name:
                night = T
            v = [t["net"] for t in T if t["maker"] == mk]
            if not v:
                est.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                ann.append("")
                continue
            est.append(sum(v))
            l, h = _mi_boot_ci_sum(v)
            lo.append(l)
            hi.append(h)
            ann.append(f"{len(v)} trips\n{np.mean(np.array(v) > 0) * 100:.0f}% win")
        xx = x + (j - 0.5) * 0.36
        bars = a1.bar(xx, est, width=0.34, color=col, label=lab)
        ok = np.isfinite(est)
        a1.errorbar(xx[ok], np.array(est)[ok], yerr=ci_err(np.array(est)[ok], np.array(lo)[ok], np.array(hi)[ok]),
                    fmt="none", ecolor="#333", capsize=3, lw=1.2)
        for xi, e, h_, l_, t in zip(xx, est, hi, lo, ann):
            if not np.isfinite(e):
                a1.text(xi, 0.02, "none", ha="center", va="bottom", fontsize=9, color="#777")
                continue
            yy = h_ if e >= 0 else l_
            a1.annotate(f"{e:+.2f}\n{t}", (xi, yy), xytext=(0, 3 if e >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if e >= 0 else "top", fontsize=9)
    a1.axhline(0, color="#222", lw=1)
    a1.set_xticks(x, [r[0] for r in runs])
    a1.set_ylabel("round-trip P&L net of fees (USD)")
    a1.set_title("FIFO round trips by how they were closed; 95% bootstrap CI", fontsize=11)
    a1.legend(loc="upper right", fontsize=9, frameon=False)
    lo_, hi_ = a1.get_ylim()
    a1.set_ylim(lo_ - 0.5, hi_ + 0.35)

    rng = np.random.default_rng(4)
    for j, (mk, lab, col) in enumerate(kinds):
        v = np.array([t["net"] for t in night if t["maker"] == mk]) * 100
        xj = j + rng.uniform(-0.18, 0.18, v.size)
        a2.scatter(xj, v, s=20, color=col, alpha=0.75, edgecolor="white", linewidth=0.4)
        a2.hlines(np.median(v), j - 0.28, j + 0.28, color="#222", lw=2)
        a2.annotate(f"median {np.median(v):+.1f}c", (j + 0.3, np.median(v)), fontsize=9.5, va="center")
    a2.axhline(0, color="#999", lw=1)
    a2.set_xticks([0, 1], ["passive exit", "market-order exit"])
    a2.set_xlim(-0.5, 1.9)
    a2.set_ylabel("P&L per round trip (US cents)")
    a2.set_title("Nov 8-9 night, one dot per trip", fontsize=11)
    return fig, caption("data/mm/analysis/mainnet-2026110{8,9}T*/ and mainnet-20261112T0050/executions.json (recomputed FIFO as scripts/mm_analyze.py)",
                        "clean_holdout (post-cutoff, live mainnet, 3 sessions)",
                        "night = deduplicated union of two pulls; 42 of 44 taker exits are the forced flatten 23:04-23:07 UTC "
                        "(first-hour pull alone: 43 passive +0.55, 42 taker -0.54)")


@figure("mm", "nvda_micro",
        "NVDA micro MM earned on its quotes but held a long book while NVDA fell: about -0.16 USD",
        "data/mm/analysis/mainnet-20261112T0050/*.json")
def _mi_mm_nvda():
    import datetime as _dt
    d = DATA / "mm" / "analysis" / "mainnet-20261112T0050"
    E = _mi_mm_execs("mainnet-20261112T0050")
    with open(d / "positions.json", encoding="utf-8") as fh:
        pos = [p for p in json.load(fh) if p.get("symbol") == "NVDAUSDT"]
    with open(d / "closed_pnl.json", encoding="utf-8") as fh:
        closed = sum(num(c["closedPnl"], 0.0) for c in json.load(fh) if c.get("symbol") == "NVDAUSDT")
    pos_end = num(pos[0]["size"]) * (1 if pos[0]["side"] == "Buy" else -1) if pos else 0.0
    mark_end = num(pos[0]["markPrice"]) if pos else num(E[-1]["markPrice"])
    upnl = num(pos[0]["unrealisedPnl"], 0.0) if pos else 0.0
    net_flow = sum((1 if x["side"] == "Buy" else -1) * num(x["execQty"]) for x in E)
    p = pos_end - net_flow  # position at the first fill
    t = [_dt.datetime.fromtimestamp(int(x["execTime"]) / 1000, tz=_dt.timezone.utc) for x in E]
    edge, inv, prev = 0.0, 0.0, None
    path = []
    for x in E:
        q, px, mk = num(x["execQty"]), num(x["execPrice"]), num(x["markPrice"])
        s = 1 if x["side"] == "Buy" else -1
        if prev is not None:
            inv += p * (mk - prev)
        edge += s * (mk - px) * q
        p += s * q
        prev = mk
        path.append(p)
    inv += p * (mark_end - prev)
    import matplotlib.dates as mdates
    fig, axs = plt.subplots(1, 3, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    a1, a2, a3 = axs
    for side, col in (("Buy", C[0]), ("Sell", C[1])):
        ii = [i for i, x in enumerate(E) if x["side"] == side]
        a1.scatter([t[i] for i in ii], [num(E[i]["execPrice"]) for i in ii], s=14, color=col, label=f"{side.lower()} fills ({len(ii)})")
    a1.set_ylabel("NVDA fill price (USD)")
    a1.set_title("141 maker fills, 23:48-00:50 UTC", fontsize=11)
    a1.legend(loc="upper right", fontsize=9, frameon=False)
    a2.step(t, np.array(path) * mark_end, where="post", color=C[4], lw=1.75)
    a2.axhline(13, color="#222", ls="--", lw=1.2, label="inventory target 13 USD")
    a2.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    a2.set_ylabel("inventory (USD at final mark)")
    a2.set_title("long almost all hour", fontsize=11)
    for ax in (a1, a2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("time (UTC)")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
    vals = [edge, inv, edge + inv, closed + upnl]
    labs = ["edge\nvs mark", "inventory\nvs mark", "hour\nMTM", "closed +\nopen"]
    cols = [C[2], C[4], C[7], C[7]]
    bars = a3.bar(range(4), vals, color=cols, width=0.6)
    annotate_bars(a3, bars, fmt="{:+.2f}")
    a3.axhline(0, color="#222", lw=1)
    a3.set_xticks(range(4), labs, fontsize=9)
    a3.set_ylabel("P&L (USD)")
    a3.set_title("decomposition", fontsize=11)
    lo, hi = a3.get_ylim()
    a3.set_ylim(lo * 1.2, hi + 0.15)
    return fig, caption("data/mm/analysis/mainnet-20261112T0050/ (executions, closed_pnl, positions)",
                        "clean_holdout (post-cutoff, live mainnet, 1 hour)",
                        f"edge = sum of side x (Bybit mark at fill - fill price) x qty; mark lags fills, so the split is rough. "
                        f"Closed {closed:+.3f}, open {upnl:+.3f}; diary 2026-11-12 (1 m mid proxy): +9 c/NVDA capture, ~-0.25 inventory")


if __name__ == "__main__":
    sys.exit(main())
