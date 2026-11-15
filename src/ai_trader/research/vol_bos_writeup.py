"""Figures and the Markdown/JSON report of the VBOS study, built from the saved stage outputs only
(<cache>/insample.json, selection.json, holdout.json, grid_surfaces.json, insample_series.npz). Nothing here
computes a new statistic that feeds a verdict.

    uv run python -m ai_trader.research.vol_bos_writeup [--cache DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

from ai_trader.core.config import load
from ai_trader.features import vol_bos as vb
from ai_trader.research import vol_bos_study as st

REPORT = "vol_bos_2026-11-13"
C1, C2, C3, GREY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"


def _f(x, nd: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ax(title: str, xlabel: str = "", ylabel: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", wrap=True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _done(fig, path: Path) -> str:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path.relative_to(load().root)).replace("\\", "/")


def _day(i: int) -> _dt.date:
    return st.DAY0 + _dt.timedelta(days=int(i))


def figures(cache: Path, ins: dict, ho: dict, grid: dict) -> list[str]:
    out_dir = load().root / "research" / "figures" / "vol_bos"
    out_dir.mkdir(parents=True, exist_ok=True)
    L, HL, vi = ins["L"], ho["L"], ins["verdicts"]
    paths = []

    # 1. VBOS1 walk-forward t by family x freq, holdout t as dots
    keys = list(L["vbos1"].keys())
    wf_t = [L["vbos1"][k]["h10"]["t"] or 0.0 for k in keys]
    ho_t = [HL["vbos1"].get(k, {}).get("h10", {}).get("t") for k in keys]
    fig, ax = _ax("H-VBOS1: no vol-BoS family raises trend P&L; range breaks and CUSUM alarms precede WORSE 10-day ensemble P&L "
                  f"(range|1d WF t {_f(L['vbos1']['range|1d']['h10']['t'])}, holdout t {_f(HL['vbos1']['range|1d']['h10']['t'])})",
                  "t-stat of 10-day ensemble excess P&L after a declustered onset (10-day-bucket clustered NW)")
    y = np.arange(len(keys))
    ax.barh(y, wf_t, height=0.55, color=C1, label="llm_in_sample, nested walk-forward picks")
    ax.scatter([t if t is not None else np.nan for t in ho_t], y, s=64, color=C2, zorder=3, edgecolor=SURF, lw=2,
               label="clean_holdout, frozen detector (missing: < 5 clusters)")
    for xv in (-2, 2):
        ax.axvline(xv, color=GREY, ls="--", lw=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, keys)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    paths.append(_done(fig, out_dir / "fig1_vbos1_walk_forward_t.png"))

    # 2. VBOS1 descriptive: every detector's in-sample vs holdout t (range family highlighted)
    fig, ax = _ax("Range-break detectors are negative almost everywhere: 89-96% of grid points below 0 in-sample, 89-100% in the holdout "
                  "(descriptive, every detector, not used to select)", "in-sample t (all llm_in_sample events)", "holdout t")
    for fr, mk in (("1d", "o"), ("4h", "s")):
        g = grid[fr]
        fam = np.array([n.split("_")[0] for n in g["names"]])
        a, b = np.array(g["t_in"], float), np.array(g["t_ho"], float)
        ax.scatter(a[fam != "range"], b[fam != "range"], s=34, marker=mk, color=GREY, edgecolor=SURF, lw=1, label=f"{fr} other families")
        ax.scatter(a[fam == "range"], b[fam == "range"], s=40, marker=mk, color=C2, edgecolor=SURF, lw=1, label=f"{fr} range break")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    paths.append(_done(fig, out_dir / "fig2_vbos1_all_detectors.png"))

    # 3. VBOS2 signed returns by horizon
    labels, a, b = [], [], []
    for fr in L["vbos2"]:
        for h in st.H:
            labels.append(f"{fr} h{h}")
            a.append(L["vbos2"][fr][f"h{h}"]["mean"] or 0.0)
            b.append(HL["vbos2"].get(fr, {}).get(f"h{h}", {}).get("mean") or 0.0)
    fig, ax = _ax(f"H-VBOS2: semivariance breaks do not point the trend in-sample (WF 10-day signed return 1d {_f(L['vbos2']['1d']['h10']['mean'], 3, True)}, "
                  f"4h {_f(L['vbos2']['4h']['h10']['mean'], 3, True)} sigma); the 4h holdout sign is positive but fragile",
                  "", "net return in the dominant side's direction (daily sigma)")
    x = np.arange(len(labels))
    ax.bar(x - 0.2, a, width=0.38, color=C1, label="llm_in_sample, nested walk-forward")
    ax.bar(x + 0.2, b, width=0.38, color=C2, label="clean_holdout, frozen threshold")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, fontsize=8)
    paths.append(_done(fig, out_dir / "fig3_vbos2_signed_returns.png"))

    # 4. VBOS3 cumulative P&L
    z = np.load(cache / "insample_series.npz")
    h3, s3 = HL["vbos3"], L["vbos3"]
    xs = np.r_[z["wf"], np.array(h3["series"])]
    ys = np.r_[z["wf_ens"], np.array(h3["ens_series"])]
    dts = [_day(i) for i in np.r_[z["wf_days"], np.array(h3["days_idx"])]]
    fig, ax = _ax(f"H-VBOS3: the walk-forward-selected filters sit flat for months (sparse CUSUM/HAR gates) and end below the unfiltered ensemble "
                  f"(WF Sharpe {_f(s3['walk_forward']['wf_sharpe'])} vs {_f(s3['walk_forward']['wf_ens_sharpe'])}; "
                  f"holdout {_f(h3['sharpe'])} vs {_f(h3['ens_sharpe'])})", "", "cumulative EW P&L (risk units per symbol)")
    ax.plot(dts, np.cumsum(xs), color=C1, lw=2, label="vol BoS + structure + semivariance filter (WF picks, then frozen final)")
    ax.plot(dts, np.cumsum(ys), color=GREY, lw=2, label="unfiltered trend ensemble (same freq)")
    ax.axvline(_day(st.cutoff_day()), color=C2, ls="--", lw=1, label="knowledge cutoff 2026-07-01")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    paths.append(_done(fig, out_dir / "fig4_vbos3_cumulative.png"))

    # 5. PBO logits
    lg = z["logits"]
    fig, ax = _ax(f"H-VBOS3: CSCV over 992 configs gives PBO {_f(s3['pbo']['pbo'])}: the in-sample winner usually lands below the "
                  f"out-of-sample median (median logit {_f(s3['pbo']['logit_median'])})", "logit of the in-sample winner's out-of-sample rank",
                  "splits (of 12870)")
    ax.hist(lg[np.isfinite(lg)], bins=40, color=C1, edgecolor=SURF, lw=0.5)
    ax.axvline(0, color=C2, lw=1.5)
    paths.append(_done(fig, out_dir / "fig5_vbos3_pbo_logits.png"))

    # 6. placebo
    pb = h3.get("placebo")
    if pb:
        fig, ax = _ax(f"H-VBOS3 timing placebo: the frozen filter's holdout Sharpe {_f(h3['sharpe'])} is the median of 200 time-shifted gates "
                      f"(beats {100 * pb['pct_below_actual']:.0f}%): the vol-BoS timing adds nothing", "holdout Sharpe of the shifted-gate book", "draws")
        ax.hist(np.array(pb["draws"], float), bins=30, color=GREY, edgecolor=SURF, lw=0.5)
        ax.axvline(h3["sharpe"], color=C1, lw=2, label="frozen config")
        ax.axvline(h3["ens_sharpe"], color=C2, lw=2, ls="--", label="unfiltered 4h ensemble")
        ax.legend(frameon=False, fontsize=8)
        paths.append(_done(fig, out_dir / "fig6_vbos3_placebo.png"))

    # 7-8. VBOS4 surfaces
    for i, (fam, lab) in enumerate((("range", "range-break grid"), ("semi", "semivariance grid"))):
        hits = {fr: ho["vbos4"][f"{fam}|{fr}"]["check"].get("hit") for fr in st.FREQS}
        rho = {fr: ho["vbos4"][f"{fam}|{fr}"]["check"].get("spearman") for fr in st.FREQS}
        fig, ax = _ax(f"H-VBOS4 {lab}: holdout optimum inside the in-sample top quintile 1d {hits['1d']}, 4h {hits['4h']}; "
                      f"surface rank corr 1d {_f(rho['1d'])}, 4h {_f(rho['4h'])}", "in-sample objective (clustered t, 27 grid points)",
                      "holdout objective (clustered t)")
        for fr, col in zip(st.FREQS, (C1, C2)):
            a = np.array(ho["selection"]["vbos4_in_sample"][f"{fam}|{fr}"], float)
            b = np.array(ho["vbos4"][f"{fam}|{fr}"]["holdout"], float)
            ax.scatter(a, b, color=col, s=40, edgecolor=SURF, lw=1, label=f"{fr} grid points")
            ch = ho["vbos4"][f"{fam}|{fr}"]["check"]
            if ch.get("threshold") is not None:
                ax.axvline(ch["threshold"], color=col, ls="--", lw=1, label=f"{fr} in-sample top-quintile edge")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.legend(frameon=False, fontsize=8)
        paths.append(_done(fig, out_dir / f"fig{7 + i}_vbos4_{fam}_surface.png"))

    # 9. BTC price with onsets of the frozen 1d range detector
    d = st.load_symbol(cache, "BTCUSDT", "1d")
    s = ho["selection"]["L"]["vbos1"]["range|1d"]
    if d is not None and s["cfg"] >= 0:
        on = vb.decluster(d["det"][s["cfg"]] & (np.arange(len(d["ts"])) >= d["lb"]), st.DECLUSTER_DAYS)
        tt = d["ts"].astype("datetime64[us]")
        fig, ax = _ax(f"BTC 1d: onsets of the frozen range-break detector ({s['name']}: 20-day RV above its 180-day 90% quantile) "
                      f"{int(on.sum())} declustered events", "", "log price")
        ax.plot(tt, np.log(d["close"]), color=GREY, lw=1, label="BTC close (log)")
        ax.scatter(tt[on], np.log(d["close"][on]), color=C2, s=34, zorder=3, edgecolor=SURF, lw=1, label="vol BoS onset")
        ax.axvline(np.datetime64(st.ddate(st.cutoff_day())), color=C1, ls="--", lw=1, label="knowledge cutoff")
        ax.legend(frameon=False, fontsize=8)
        paths.append(_done(fig, out_dir / "fig9_btc_range_onsets.png"))
    return paths


# ====================================================================== narrative (written after the results)
WORKED = [
    "A robust NEGATIVE pattern: after a realised-vol range break (5-20 day RV above its trailing 60-180 day 90-100% quantile) the trend "
    "ensemble earns LESS than its own mean over the next 5-20 days. Walk-forward in-sample range|1d -0.037/day (t -2.43), range|4h -0.075 "
    "(t -2.37); the frozen detector in the holdout -0.086 (t -2.97, 24 events, 1d) and -0.102 (t -1.59, 4h). Across the whole range grid 89-96% "
    "of configurations are negative in-sample and 89-100% in the holdout (descriptive, fig2). CUSUM alarms point the same way in-sample "
    "(cusum|4h t -2.33) but not in the holdout. This is the opposite of the registered direction, so it is a candidate to register, not knowledge.",
    "The nested purged walk-forward, DSR and PBO did their job: they show that every in-sample 'winner' here is a selection artefact "
    "(final strategy DSR 0.22 against SR0 0.68 for 992 trials; PBO 0.84; the WF picks switch family or frequency in 5 of 8 block transitions).",
    "The frozen machinery is reusable: causal detectors with quantile thresholds (unit-tested: appending bars never changes past flags, a planted "
    "x4 vol shift is caught within 5 bars by range, CUSUM and HAR), fractal BoS/CHoCH, and a trade builder that reproduces evaluate.strategy_pnl exactly.",
]
DID_NOT = [
    "H-VBOS1: vol BoS is not the timing filter trend was missing. No family x freq test is positive with t > 2 (best vov|4h +0.028/day, t 0.81; "
    "all Holm p = 1); the frozen vov|4h detector is negative in the holdout (-0.057, t -1.16).",
    "H-VBOS2: which semivariance side breaks first does not point the next 10 days. WF signed return 1d -0.077 sigma (t -1.04), 4h -0.037 (t -0.59). "
    "The dominant side agrees with the sign of the trailing return in 69-80% of events, so it is a momentum proxy, and short-horizon momentum after "
    "vol expansion does not continue (cf. R-VSIG-2). The 4h holdout bucket-weighted mean is +0.165 (t 1.20), which literally meets the holdout sign "
    "criterion, but the event-weighted mean is about -0.02 (up +0.005, down -0.060) and n is 45: the pooled verdict stays 'contradicts'.",
    "H-VBOS3: 'trend following with market-structure awareness' as specified (enter on a fractal BoS in the ensemble's direction while a vol BoS "
    "is active and the dominant semivariance agrees; ensemble or opposite-BoS exit) does not beat the plain ensemble. The in-sample final pick "
    "(4h, har_1.5_3, fractal k 3, semivariance on, own exits) has Sharpe 0.35 vs 0.44 for the unfiltered 4h ensemble on the same days at 43% "
    "exposure; holdout -0.46 vs -0.12. Its holdout Sharpe equals the median of 200 time-shifted gates, so the vol-BoS timing contributes nothing; "
    "the filter is an exposure cut (R-EXIT-2), and here it even cut the good days.",
    "The opposite-BoS exit hurt in-sample (ablation: 0.35 -> 0.13 when switched on), the same lesson as R-EXIT-1: structure-based exits are tight trails.",
    "H-VBOS4: thresholds are not stable. 2 of 4 holdout optima fall inside the in-sample top quintile (range|1d, semi|1d) and 2 do not; surface "
    "rank correlations in-sample vs holdout are -0.11 / +0.50 (range 1d / 4h) and +0.23 / -0.08 (semi); the split-half in-sample analogue also gives 2 of 4.",
    "TradFi T cannot test the selection: history starts 2026-04..08, so the 180-day trailing windows and the HAR fit (180-day minimum) never warm up; "
    "the frozen VBOS3 config never trades on T. Only the vov/CUSUM families fire on T (all |t| < 1.5).",
    "BTC alone is too thin for event-level selection (fewer than 5 ten-day clusters per half-year fold for most detectors); its per-symbol strategy pick "
    "had in-sample Sharpe 0.77 vs 0.59 but DSR 0.09 and walk-forward Sharpe 0.15 vs 0.65.",
]
CAVEATS = [
    "The holdout is 135 days (2026-07-01 .. 2026-11-12) with few independent vol episodes; holdout event counts for the frozen detectors are 4-113.",
    "Holdout t-values rest on about 10-14 ten-day clusters, so their magnitudes (down to -8 for some range grid points) are overstated; read the holdout for sign and rough size only.",
    "Ten-day-bucket clustering with NW over buckets is a judgement call; event-weighted and bucket-weighted means differ materially when n is small (VBOS2 4h holdout).",
    "The ensemble's EW in-sample Sharpe here (4h 0.44, 1d 0.31 over 2021-01..2026-06) is below the trend-zoo report's 0.96 because each symbol starts after "
    "200 warm-up days (the zoo includes 2020H2); the base itself reproduces data/priors/trend_zoo_daily_4h.parquet (monthly corr 0.997).",
    "Universe L is today's survivors; risk units, not dollars; 7.5 bps per side; no funding.",
    "The range-break finding is data-generated in sign (the registered direction was positive) and should be registered as its own hypothesis before it is used.",
]
RULES_YAML = '''- id: "R-VBOS-1"
  kind: "avoid"
  rule: "WHEN a realised-volatility range break fires (5-20 day RV above its trailing 60-180 day 90-100% quantile) THEN don't expect trend-following to earn more than usual over the next 5-20 days; the trend ensemble has earned less than its own mean after such breaks"
  when: "universe L crypto perps, 1d or 4h bars; onset = first bar the w-day log RV exceeds its trailing N-day q-quantile, declustered 20 days"
  then: "don't add or scale trend exposure because volatility broke out; read a vol range break as a whipsaw warning, not a confirmation; it is not an exit signal either (R-EXIT-1)"
  scope: {assets: "crypto", horizon: "5-20d", sides: "both"}
  effect: "10-day ensemble excess P&L walk-forward in-sample -0.037/day (t -2.43, 1d) and -0.075 (t -2.37, 4h); frozen detector holdout -0.086 (t -2.97, 24 events, 1d), -0.102 (t -1.59, 4h); 89-96% of range-grid points negative in-sample, 89-100% in the holdout"
  confidence: "C"
  evidence: ["H-VBOS1", "research/reports/vol_bos_2026-11-13.md"]
  review: "register the negative direction as its own hypothesis and keep it only if the forward excess after range breaks stays < 0 with t < -2 over the next 100 events"
  status: "provisional"
- id: "R-VBOS-2"
  kind: "expect"
  rule: "WHEN upside or downside realised semivariance breaks above its own trailing quantile while the other side does not THEN don't read the dominant side as the direction of the next 5-20 days; it mostly restates the trailing return"
  when: "1d or 4h; RS+ / RS- over 5-20 days vs their trailing 60-180 day 80-95% quantiles; first bar of a one-sided break"
  then: "no directional preference from semivariance dominance; use the semivariances for the vol forecast only"
  scope: {assets: "crypto", horizon: "5-20d", sides: "both"}
  effect: "walk-forward signed 10-day net return -0.077 sigma (t -1.04, 1d) and -0.037 (t -0.59, 4h); dominant side agrees with the trailing-return sign in 69-80% of events; 4h holdout +0.165 bucket-weighted (t 1.20) but about -0.02 event-weighted (45 events)"
  confidence: "B"
  evidence: ["H-VBOS2", "research/reports/vol_bos_2026-11-13.md"]
  review: "a registered re-test that conditions on semivariance dominance AGAINST the trailing return (the non-momentum part) shows t > 2 in both segments"
  status: "active"
- id: "R-VBOS-3"
  kind: "avoid"
  rule: "WHEN tempted to gate trend entries by a volatility break of structure plus a fractal price break and semivariance agreement THEN don't expect a better trend book; the in-sample-best gate is a selection artefact and its timing is no better than a time-shifted gate"
  when: "trend-zoo ensemble, 1d/4h, 992-config grid of vol-BoS detector x fractal k x semivariance filter x exit"
  then: "keep the plain ensemble's entries and exits; if exposure must be cut, say so and cut it directly rather than through a vol-BoS timing story"
  scope: {assets: "crypto", horizon: "1-20d", sides: "both"}
  effect: "final pick Sharpe 0.35 vs 0.44 unfiltered in-sample (43% exposure), DSR 0.22 (SR0 0.68, 992 trials), PBO 0.84 (CSCV 16 blocks); holdout -0.46 vs -0.12; holdout Sharpe at the 50th percentile of 200 time-shifted gates; adding the opposite-BoS exit cut in-sample Sharpe 0.35 -> 0.13"
  confidence: "B"
  evidence: ["H-VBOS3", "research/reports/vol_bos_2026-11-13.md"]
  review: "a pre-registered filter beats the ensemble in a forward window with DSR > 0.95 and PBO < 0.5 on its own grid"
  status: "active"
- id: "R-VBOS-4"
  kind: "expect"
  rule: "WHEN a vol-BoS or semivariance-break threshold is learned on history THEN expect the best threshold to move out of sample; don't trust a single tuned quantile"
  when: "quantile thresholds 0.80-1.00 over 60-180 day trailing windows, 1d/4h, pooled crypto"
  then: "use fixed, coarse thresholds chosen a priori, or report the whole surface; never act on a grid optimum alone"
  scope: {assets: "crypto", horizon: "5-20d", sides: "both"}
  effect: "holdout optimum inside the in-sample top quintile for 2 of 4 surfaces (split-half in-sample analogue also 2 of 4); in-sample vs holdout surface rank corr -0.11 to +0.50; walk-forward picks change family or freq in 5 of 8 block transitions"
  confidence: "C"
  evidence: ["H-VBOS4", "research/reports/vol_bos_2026-11-13.md"]
  review: "a forward window where >= 3 of 4 surfaces keep their optimum in the frozen top quintile"
  status: "provisional"
'''


# ====================================================================== markdown
def _row(*c) -> str:
    return "| " + " | ".join(str(x).replace("|", "\|") for x in c) + " |"


def _ev(r: dict | None, nd: int = 3) -> str:
    if not r or not r.get("n_events"):
        return "n/a (0)"
    return f"{_f(r.get('mean'), nd, True)} (t {_f(r.get('t'))}, n {r['n_events']})"


def markdown(ins: dict, ho: dict, grid: dict, figs: list[str]) -> str:
    L, B, vi, HL = ins["L"], ins["BTC"], ins["verdicts"], ho["L"]
    vh = ho["verdicts"]
    s3, h3 = L["vbos3"], HL["vbos3"]
    md = []
    md.append("# Volatility break of structure as a trend-following filter (H-VBOS1-4) - 2026-11-13\n")
    md.append("Agent F6. Plan: `research/plans/2026-11-13_vol_bos_trend.md`. Design (detector families, grids, objectives, nested purged "
              "walk-forward, purge/embargo, declustering, verdict rules) fixed in the diary 2026-11-13 21:03 UTC (kind 'hypothesis') before any "
              "result; in-sample verdicts recorded and logged (diary 21:19 UTC, with the pooled-verdict rules) before the single clean_holdout look. "
              "Code: `src/ai_trader/features/vol_bos.py`, `src/ai_trader/research/vol_bos_study.py` (stages), `vol_bos_report.py` (verdicts, "
              "records), `vol_bos_writeup.py` (this report). Tests: `tests/test_vol_bos.py` (14).\n")
    md.append(f"Data: bybit, universe L = 23 crypto perps pooled (verdicts), T = 16 TradFi perps (holdout robustness), BTC alone (report only); "
              f"1d and 4h bars; intrabar RV / semivariances from 1h returns; trend base = the 21-variant zoo ensemble (`trend_zoo.ensemble`) through "
              f"`evaluate.strategy_pnl` (7.5 bps per side, risk units). Segments: llm_in_sample < 2026-07-01 <= clean_holdout "
              f"({ho['window'][0]} .. {ho['window'][1]}, 135 days).\n")
    md.append("## Verdicts\n")
    md.append(_row("hypothesis", "llm_in_sample", "clean_holdout", "pooled", "key numbers"))
    md.append(_row("---", "---", "---", "---", "---"))
    p1 = vi["vbos1_primary"]
    md.append(_row("H-VBOS1 vol BoS -> more trend P&L", vi["H-VBOS1"], vh["clean_holdout"]["H-VBOS1"], vh["pooled"]["H-VBOS1"],
                   f"best family {p1}: WF {_ev(L['vbos1'][p1]['h10'])}, all Holm p 1.00; holdout {_ev(HL['vbos1'][p1]['h10'])}"))
    p2 = vi["vbos2_primary"]
    md.append(_row("H-VBOS2 semivariance side -> direction", vi["H-VBOS2"], vh["clean_holdout"]["H-VBOS2"], vh["pooled"]["H-VBOS2"],
                   f"WF 1d {_ev(L['vbos2']['1d']['h10'])}, 4h {_ev(L['vbos2']['4h']['h10'])}; holdout {p2} {_ev(HL['vbos2'][p2]['h10'])}"))
    md.append(_row("H-VBOS3 filtered beats unfiltered", vi["H-VBOS3"], vh["clean_holdout"]["H-VBOS3"], vh["pooled"]["H-VBOS3"],
                   f"DSR {_f(s3['dsr_final']['dsr'])}, PBO {_f(s3['pbo']['pbo'])}, holdout Sharpe {_f(h3['sharpe'])} vs {_f(h3['ens_sharpe'])} "
                   f"(paired diff t {_f(h3['diff']['t'])})"))
    md.append(_row("H-VBOS4 thresholds stable", vi["H-VBOS4"], vh["clean_holdout"]["H-VBOS4"], vh["pooled"]["H-VBOS4"],
                   f"holdout optimum in in-sample top quintile {sum(bool(x) for x in vh['vbos4_holdout_hits'])}/4; split-half {sum(bool(x) for x in vi['vbos4_split_hits'])}/4"))
    hv = vi["holm_vbos"]
    md.append(f"\nHolm within VBOS (in-sample one-sided p: VBOS1 best Holm-adjusted family p, VBOS2 best Holm-adjusted freq p, VBOS3 WF paired "
              f"diff): raw {', '.join(f'{k} {_f(v)}' for k, v in hv['p'].items())}; Holm {', '.join(f'{k} {_f(v)}' for k, v in hv['p_holm'].items())}.\n")

    md.append("## Selected configuration and thresholds (frozen in-sample)\n")
    lab = s3["walk_forward"]["final"]["label"]
    md.append(f"- **Strategy (H-VBOS3):** {lab['freq']} bars; vol BoS = `{lab['detector']}` (persistent HAR surprise: 4h realised variance above "
              f"1.5x the causal log-HAR level forecast for 3 consecutive bars), active for 20 days after the onset; entry on a close beyond the last "
              f"unbroken fractal swing (k = {lab['fractal_k_bars']} bars each side) in the ensemble's direction; semivariance filter "
              f"{'on' if lab['semi_filter'] else 'off'} (10-day RS+ - RS- must have the trade's sign); exit = {lab['exit']} (ensemble sign change).")
    md.append("- **VBOS1 final detectors per family:** " + "; ".join(f"{k} `{v['final'].get('name')}`" for k, v in L["vbos1"].items()) + ".")
    md.append("- **VBOS2 final semivariance thresholds (w days, N days, quantile):** " +
              "; ".join(f"{fr} {v['final'].get('params')}" for fr, v in L["vbos2"].items()) + ".")
    md.append("- Thresholds are quantiles of each symbol's own trailing history, so they transfer across assets; nothing was tuned on the holdout.\n")

    md.append("## H-VBOS1 - trend-ensemble P&L after a vol BoS (event study)\n")
    md.append("Excess = mean daily ensemble P&L over the h days after a declustered onset minus that symbol's mean daily ensemble P&L in the same "
              "block / segment. Clustered t over 10-day calendar buckets. WF = nested purged walk-forward (outer half-years 2022H1-2026H1, inner "
              "median-of-folds, 25-day gap). Holdout = the frozen final detector.\n")
    md.append(_row("family|freq", "WF h5", "WF h10", "WF h20", "Holm p", "final detector", "holdout h10", "holdout h20"))
    md.append(_row(*["---"] * 8))
    for k, v in L["vbos1"].items():
        hv1 = HL["vbos1"].get(k, {})
        md.append(_row(k, _ev(v["h5"]), _ev(v["h10"]), _ev(v["h20"]), _f(L["vbos1_holm"]["p_holm"][k]), f"`{v['final'].get('name')}`",
                       _ev(hv1.get("h10")), _ev(hv1.get("h20"))))
    md.append("\nDescriptive (every detector, full segment, not used for selection): share of grid points with t < 0 and the median t.\n")
    md.append(_row("freq", "family", "in-sample median t", "in-sample share < 0", "holdout median t", "holdout share < 0"))
    md.append(_row(*["---"] * 6))
    for fr, g in grid.items():
        fam = np.array([n.split("_")[0] for n in g["names"]])
        for f in vb.FAMILIES:
            a, b = np.array(g["t_in"], float)[fam == f], np.array(g["t_ho"], float)[fam == f]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            md.append(_row(fr, f, _f(float(np.median(a)) if len(a) else None), _f(float(np.mean(a < 0)) if len(a) else None),
                           _f(float(np.median(b)) if len(b) else None), _f(float(np.mean(b < 0)) if len(b) else None)))

    md.append("\n## H-VBOS2 - directional semivariance breaks\n")
    md.append("Signed net return (dominant side's direction, minus 2 x 7.5 bps) over h days in daily-sigma units.\n")
    md.append(_row("freq", "WF h5", "WF h10", "WF h20", "final (w, N, q)", "agrees with trailing return", "holdout h10", "holdout up / down (h10, event mean)"))
    md.append(_row(*["---"] * 8))
    for fr, v in L["vbos2"].items():
        hh = HL["vbos2"].get(fr, {})
        bs = hh.get("by_side", {})
        md.append(_row(fr, _ev(v["h5"]), _ev(v["h10"]), _ev(v["h20"]), v["final"].get("params"), _f(v["final"].get("momentum_agreement")),
                       _ev(hh.get("h10")), f"{_f(bs.get('1', {}).get('event_mean'), 3, True)} / {_f(bs.get('-1', {}).get('event_mean'), 3, True)}"))

    md.append("\n## H-VBOS3 - filtered trend vs the unfiltered ensemble\n")
    wf = s3["walk_forward"]
    fi = s3["final_in_sample"]
    md.append(f"Grid: 62 detectors x fractal k (1d 2/5, 4h 3/12 bars) x semivariance filter x exit (own / own + opposite BoS) x 1d/4h = "
              f"{s3['n_configs']} configs; EW book over live L symbols, flat days count 0. In-sample window {s3['window'][0]} .. {s3['window'][1]}.\n")
    md.append(_row("quantity", "value"))
    md.append(_row("---", "---"))
    rows = [
        ("WF Sharpe (selected per block) vs same-freq ensemble", f"{_f(wf['wf_sharpe'])} vs {_f(wf['wf_ens_sharpe'])}; paired daily diff t {_f(wf['wf_diff_t'])} ({wf['wf_days']} days)"),
        ("WF picks per outer block", "; ".join(f"{p['block'][0][:7]} {p['label']['freq']} {p['label']['detector']} k{p['label']['fractal_k_bars']} "
                                              f"semi {int(p['label']['semi_filter'])} {p['label']['exit']}" for p in wf["picks"] if p["label"])),
        ("final in-sample Sharpe vs ensemble (same days)", f"{_f(fi['sharpe'])} vs 4h {_f(fi.get('ens_sharpe_4h'))} / 1d {_f(fi.get('ens_sharpe_1d'))}; paired diff t {_f(fi['diff_vs_ens']['t'])}"),
        ("final exposure / trades / CHoCH share of entries", f"{_f(fi['exposure_days'])} of symbol-days / {fi['trades']} / {_f(fi['choch_share'])}"),
        ("DSR of the final (992 trials)", f"{_f(s3['dsr_final']['dsr'])} (SR {_f(s3['dsr_final']['sr_ann'])}, SR0 {_f(s3['dsr_final']['sr0_ann'])}, skew {_f(s3['dsr_final']['skew'])}, kurt {_f(s3['dsr_final']['kurt'], 1)})"),
        ("DSR of the WF series", _f(s3["dsr_walk_forward"].get("dsr"))),
        ("PBO (CSCV, 16 blocks, 12870 splits)", f"{_f(s3['pbo']['pbo'])}; median logit {_f(s3['pbo']['logit_median'])}; IS-winner mean OOS rank {_f(s3['pbo']['oos_rel_rank_mean_of_is_winner'])}; degradation slope {_f(s3['pbo']['degradation_slope'])}"),
        ("grid Sharpe quantiles (0/10/50/90/100%)", " / ".join(_f(x) for x in s3["grid_sharpe_quantiles"])),
        ("share of configs beating the ensemble in-sample", f"vs 1d {_f(s3['share_configs_beating_ens']['1d'])}, vs 4h {_f(s3['share_configs_beating_ens']['4h'])}"),
        ("ablation of the final (in-sample Sharpe)", "; ".join(f"{k} {_f(v['sharpe'])}" for k, v in s3["ablation"].items())),
        ("**holdout** Sharpe vs same-freq ensemble", f"{_f(h3['sharpe'])} vs {_f(h3['ens_sharpe'])} (diff {_f(h3['sharpe_diff'], 2, True)}); paired daily diff {_f(h3['diff']['mean'], 4, True)} (t {_f(h3['diff']['t'])}); 1d ensemble {_f(h3['ens_sharpe_other_freq'].get('1d'))}"),
        ("holdout exposure", _f(h3["exposure_days"])),
        ("timing placebo (200 time-shifted gates)", f"holdout: beats {_f(h3['placebo']['pct_below_actual'])} of draws (q05/q50/q95 {' / '.join(_f(x) for x in h3['placebo']['sharpe_q05_q50_q95'])}); "
                                                     f"in-sample (from 2020-10): actual {_f(h3['placebo_in_sample']['actual'])}, beats {_f(h3['placebo_in_sample']['pct_below_actual'])}"),
    ]
    for a, b in rows:
        md.append(_row(a, b))

    md.append("\n## H-VBOS4 - threshold surfaces (description only, never used to select)\n")
    md.append(_row("surface", "split-half hit", "holdout hit", "in-sample argmax", "holdout argmax", "rank corr in vs holdout"))
    md.append(_row(*["---"] * 6))
    names = {"range": [vb.Detector("range", p).name for p in [(w, n, q) for w in vb.RANGE_W for n in vb.RANGE_N for q in vb.RANGE_Q]],
             "semi": [str(p) for p in vb.semi_grid()]}
    for k, v in ho["vbos4"].items():
        c, sp = v["check"], L["vbos4_split"][k]["check"]
        fam = k.split("|")[0]
        md.append(_row(k, sp.get("hit"), c.get("hit"), names[fam][c["in_argmax"]] if c.get("in_argmax") is not None else "n/a",
                       names[fam][c["out_argmax"]] if c.get("out_argmax") is not None else "n/a", _f(c.get("spearman"))))

    md.append("\n## BTC look and TradFi robustness (report only)\n")
    bt, Tt = ho["BTC"], ho["T"]
    md.append(f"- **BTC** (min 3 events per inner fold; most event folds still undefined): strategy pick `{B['vbos3']['walk_forward']['final']['label']['detector']}` "
              f"({B['vbos3']['walk_forward']['final']['label']['freq']}), in-sample Sharpe {_f(B['vbos3']['final_in_sample']['sharpe'])} vs ensemble "
              f"{_f(B['vbos3']['final_in_sample']['ens_sharpe_same_freq'])}, DSR {_f(B['vbos3']['dsr_final']['dsr'])}, PBO {_f(B['vbos3']['pbo']['pbo'])}, "
              f"WF Sharpe {_f(B['vbos3']['walk_forward']['wf_sharpe'])} vs {_f(B['vbos3']['walk_forward']['wf_ens_sharpe'])}; holdout "
              f"{_f(bt['vbos3']['sharpe'])} vs {_f(bt['vbos3']['ens_sharpe'])} (one symbol, 135 days: noise).")
    md.append(f"- **T** ({Tt['n_symbols']['4h']} symbols, history from 2026-04..08): the frozen VBOS3 config never trades (HAR needs 180 days); "
              f"vov|4h holdout {_ev(Tt['vbos1'].get('vov|4h', {}).get('h10'))}, cusum|4h {_ev(Tt['vbos1'].get('cusum|4h', {}).get('h10'))}; "
              f"semivariance 1d {_ev(Tt['vbos2'].get('1d', {}).get('h10'))}; T ensemble holdout Sharpe 4h {_f(Tt['vbos3']['ens_sharpe'])}.")

    md.append("\n## What worked\n")
    md += [f"- {x}" for x in WORKED]
    md.append("\n## What did not\n")
    md += [f"- {x}" for x in DID_NOT]
    md.append("\n## Caveats\n")
    md += [f"- {x}" for x in CAVEATS]
    md.append("\n## Figures\n")
    md += [f"- `{p}`" for p in figs]
    md.append("\n## Rule candidates\n")
    md.append("```yaml\n" + RULES_YAML + "```\n")
    return "\n".join(md) + "\n"


def write(cache: Path) -> dict:
    import yaml

    rd = lambda n: json.loads((cache / n).read_text(encoding="utf-8"))  # noqa: E731
    ins, ho, grid = rd("insample.json"), rd("holdout.json"), rd("grid_surfaces.json")
    figs = figures(cache, ins, ho, grid)
    rules = yaml.safe_load(RULES_YAML)
    out_dir = load().root / "research" / "reports"
    hl = json.loads(json.dumps(ho))
    hl["L"]["vbos3"].pop("series", None)
    hl["L"]["vbos3"].pop("ens_series", None)
    hl["L"]["vbos3"].pop("days_idx", None)
    if "placebo" in hl["L"]["vbos3"]:
        hl["L"]["vbos3"]["placebo"].pop("draws", None)
    for u in ("T", "BTC"):
        for k in ("series", "ens_series", "days_idx"):
            hl[u]["vbos3"].pop(k, None)
    rep = {"report": REPORT, "agent": "F6", "design_diary": "research/diary/2026-11-13.md (21:03 UTC hypothesis entry)",
           "in_sample": ins, "holdout": hl, "descriptive_grid_event_t": grid, "figures": figs, "rule_candidates": rules,
           "worked": WORKED, "did_not": DID_NOT, "caveats": CAVEATS}
    (out_dir / f"{REPORT}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out_dir / f"{REPORT}.md").write_text(markdown(ins, ho, grid, figs), encoding="utf-8")
    return {"figures": figs, "json": f"research/reports/{REPORT}.json", "md": f"research/reports/{REPORT}.md"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    print(write(Path(a.cache) if a.cache else st.default_cache()))


if __name__ == "__main__":
    main()
