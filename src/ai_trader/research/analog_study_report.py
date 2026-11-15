"""Report for the multi-scale analog search and H-ANA1 (agent ANALOG, 2026-11-15).

Reads data/research/analogs/{search,walkforward}.json (analog_study), adds point-in-time context for the top analogs
(Fear & Greed, DVOL, Wikipedia attention, scheduled macro events) and writes
research/reports/analogs_2026-11-15.{md,json} with figures in research/figures/analogs_2026-11-15/.

    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.analog_study_report
"""

from __future__ import annotations

import glob
import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import yaml

from ai_trader.research.analog_study import OUT, ROOT, SCALES, write_json
from ai_trader.research.regimes import CELL_LABELS

FIG = ROOT / "research" / "figures" / "analogs_2026-11-15"
REPORT = ROOT / "research" / "reports" / "analogs_2026-11-15"
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
SURF = "#fcfcfb"
SEGN = {"llm_in_sample": "llm_in_sample (pre 2026-07-01)", "clean_holdout": "clean_holdout (2026-07-01 on)"}

# Market episodes used to group analogs on the timeline (descriptive labels, dates inclusive).
ERAS = [
    ("2020-03-01", "2020-04-30", "2020 COVID crash"),
    ("2020-05-01", "2020-12-31", "2020 recovery/bull"),
    ("2021-01-01", "2021-04-30", "2021 H1 bull"),
    ("2021-05-01", "2021-07-31", "2021 May crash (China ban)"),
    ("2021-08-01", "2021-12-31", "2021 H2 top"),
    ("2022-01-01", "2022-04-30", "2022 Q1 Fed tightening"),
    ("2022-05-01", "2022-05-31", "2022 May LUNA"),
    ("2022-06-01", "2022-07-15", "2022 June Celsius/3AC"),
    ("2022-07-16", "2022-10-31", "2022 bear"),
    ("2022-11-01", "2022-12-31", "2022 Nov FTX"),
    ("2023-01-01", "2023-12-31", "2023 (SVB, recovery)"),
    ("2024-01-01", "2024-12-31", "2024 ETF era (Mt Gox, Aug yen unwind)"),
    ("2025-01-01", "2025-06-30", "2025 H1 (Bybit hack, tariffs)"),
    ("2025-07-01", "2025-12-31", "2025 H2 (post-Oct-10 deleveraging)"),
    ("2026-01-01", "2026-06-30", "2026 H1 bear (Warsh, June rout)"),
    ("2026-07-01", "2026-12-31", "2026 H2"),
]

# "Why" for the top-10 combined analogs. Web content was read as data (sources listed); dates in UTC.
WHY: dict[str, dict[str, Any]] = {
    "2025-11-22": {
        "cause": "A six-week slide from the October 2025 record (about 126k) on spot-ETF outflows, fading odds of a December Fed cut and AI-equity doubts after Nvidia's Nov 20 earnings reversal; BTC hit about 80.5k on Friday Nov 21. The analog hour is the Saturday after that low.",
        "resolved": "NY Fed President Williams said on Nov 21 there was 'room for a further adjustment in the near term'; December cut odds jumped above 70% and BTC bounced to about 86k at once and about 91k within the week. A policy-expectations rescue.",
        "sources": ["https://www.albawaba.com/business/pr/bitcoin-jumps-805k-lows-86k-after-fed-1617407", "https://cryptoslate.com/?p=508072",
                    "https://cointelegraph.com/markets/bitcoin-stocks-crumble-after-nvidia-earnings-and-fed-uncertainty-over-next-rate-cut"]},
    "2026-02-01": {
        "cause": "A weekend liquidation cascade after Trump nominated Kevin Warsh as Fed chair (read as hawkish) and a record precious-metals crash on Jan 30-31 forced margin calls across portfolios; spot ETFs had lost about $681M the week before. BTC fell below 76k, its first time under 80k since April 2025, with about $2.2B liquidated in 24h.",
        "resolved": "Not within the week. The slide extended to about 60k on Thursday Feb 5 (the worst day since FTX, about $2.7B liquidated), and BTC then rebounded 11-15% on Feb 6. The low came 4 days later and about 20% lower.",
        "sources": ["https://www.kucoin.com/news/flash/crypto-market-sees-2-2b-in-liquidations-as-bitcoin-drops-below-76k",
                    "https://www.ibtimes.co.uk/warsh-shock-how-hawkish-fed-nominee-triggered-16bn-weekend-liquidity-trap-bitcoin-1775533",
                    "https://www.theblock.co/post/388759/bitcoin-sees-flash-crash-60000", "https://fortune.com/2026/02/06/bitcoin-claws-back-above-70000-after-worst-day-since-ftx-crash/"]},
    "2026-06-07": {
        "cause": "The worst week since FTX (BTC -17%, ETH -22% in the week to Jun 6). A strong May payrolls print (172k vs 88k expected) lifted rates, spot ETFs had a 13-day outflow streak as money rotated to AI infrastructure and the SpaceX IPO, and Strategy made its first BTC sale in years to fund preferred dividends. The 2026 low was 59.1k on Jun 5-6.",
        "resolved": "An 'oversold relief rally' over the 7-8 June weekend (+7.5% to about 63.8k) and an in-line May CPI on Jun 10 (core below forecast). The week ended slightly up, still range-bound.",
        "sources": ["https://www.theblock.co/news/markets/2026-06-04-the-rally-that-wasnt-bitcoin-slides-14-in-one-week-as-etf-outflows-strategy-sale-and-oil-prices-hit-sentiment-403659",
                    "https://www.theblock.co/news/markets/2026-06-07-bitcoin-recovers-63000-kospi-plunges-403919",
                    "https://www.blockhead.co/2026/06/08/crypto-markets-weekend-selloff-eases-key-data-ahead/"]},
    "2022-01-22": {
        "cause": "Markets repriced faster Fed tightening ahead of the Jan 26 FOMC, alongside a global tech sell-off and Russia-Ukraine tension. BTC fell from about 40.7k on Jan 21 to about 34.3k on the weekend of Jan 22, about 50% below the November 2021 record: its worst week in 8 months.",
        "resolved": "A final low near 33k on Monday Jan 24, when US stocks reversed intraday, then chop around the hawkish FOMC and a February rally toward 45k. The flush was the low of that leg, not of the bear market.",
        "sources": ["https://fortune.com/2022/01/21/crypto-is-crashing-and-experts-predict-more-pain-ahead/",
                    "https://forklog.com/en/bitcoin-ends-its-worst-week-in-eight-months/",
                    "https://www.fxstreet.com/amp/cryptocurrencies/news/crypto-markets-shakeout-as-federal-reserve-leaves-us-interest-rates-on-hold-202201270904"]},
    "2023-03-11": {
        "cause": "Silicon Valley Bank failed on Friday Mar 10, with part of Circle's USDC reserves inside it. USDC depegged to about $0.87 on Saturday Mar 11 and BTC traded under 20k.",
        "resolved": "On Sunday Mar 12, the Fed, Treasury and FDIC backstopped all SVB deposits and created the Bank Term Funding Program. USDC repegged and BTC rallied about 19% in 72h and 31% in 7 days: an external policy rescue, the largest upside outcome in the set.",
        "sources": ["https://www.bloomberg.com/news/articles/2023-03-13/bitcoin-btc-increases-more-than-5-after-us-supports-banking-sector",
                    "https://techcrunch.com/2023/03/13/bitcoin-rallies-18-percent-in-wake-of-svb-crisis/",
                    "https://coinloan.io/blog/coinloan-weekly-market-rebound-usdc-recovery-impact-of-svb-shutdown/"]},
    "2024-07-06": {
        "cause": "A supply overhang. The Mt. Gox estate began repaying creditors (47k BTC moved on Jul 5) while the German government sold seized BTC; BTC hit about 53.5k on Friday Jul 5, a four-month low.",
        "resolved": "The German wallet was emptied around Jul 12 and many Mt. Gox coins were held rather than sold. BTC chopped for a week, then rallied to about 68k after the Jul 13 assassination attempt on Trump raised his election odds. The supply shock ended, and a political catalyst followed.",
        "sources": ["https://www.theblock.co/news/markets/2024-07-04-bitcoin-dips-mt-gox-moves-btc-303594",
                    "https://forklog.com/en/bitcoin-dips-below-54000-amid-mt-gox-movements-experts-warn-of-further-decline/",
                    "https://fortune.com/crypto/2024/07/08/bitcoin-slumps-mt-gox-governments-liquidating-crypto/"]},
    "2022-06-14": {
        "cause": "The crypto credit unwind after LUNA. Celsius froze withdrawals on Jun 12-13 and Three Arrows Capital was being liquidated, while the hot May CPI (Jun 10) set up a 75 bp Fed hike on Jun 15, the largest since 1994. BTC lost 20k.",
        "resolved": "It did not resolve. After a brief post-FOMC bounce to about 22.5k, BTC made a new low at about 17.6k on Saturday Jun 18 as forced sellers dumped. The bear market's final low came only after FTX in November.",
        "sources": ["https://cointelegraph.com/markets/bitcoin-price-climbs-to-22-5k-after-fed-75-basis-point-hike-aims-to-cap-runaway-inflation",
                    "https://forklog.com/en/june-2022-in-figures-bitcoin-below-20000-steth-discount-and-collapse-of-well-known-crypto-firms/amp",
                    "https://www.coinbase.com/institutional/research-insights/research/weekly-market-commentary/weekly-market-commentary-june-17-2022"]},
    "2025-11-15": {
        "cause": "The start of the same November 2025 leg as analog #1. BTC broke below 95k on Friday Nov 14 (a six-month low) on the largest spot-ETF outflow since February ($867M on Nov 13), with December cut odds falling under 50%.",
        "resolved": "Not within the week. The slide continued to about 80.5k on Nov 21, until the Williams comments of analog #1. This is the 'low still ahead' branch.",
        "sources": ["https://www.marketpulse.com/markets/bitcoin-drops-below-95000-market-under-pressure-as-investor-confidence-wanes/",
                    "https://www.blockhead.co/2025/11/17/bitcoin-drops-below-95k-in-deepest-weekly-drawdown-since-may/"]},
    "2025-02-27": {
        "cause": "Three things at once: the Feb 21 Bybit hack (about $1.5B of ETH stolen, the largest ever), heavy spot-ETF outflows (about $2.1B in six days) and new tariff threats (25% on EU goods, Canada/Mexico tariffs due Mar 4). BTC fell below 83k on Feb 27 and to about 78-80k on Feb 28.",
        "resolved": "On Sunday Mar 2, Trump posted about a US strategic crypto reserve and BTC spiked to about 94k. Most of that gain was given back within days once the tariffs took effect. The path ran through a squeeze and a reversal.",
        "sources": ["https://www.thenationalnews.com/business/money/2025/02/27/bitcoin-cryptos/", "https://cointelegraph.com/magazine/timeline-trump-tariffs-bitcoin-drop",
                    "https://www.theblock.co/amp/post/344559/the-daily-trumps-crypto-reserve-plan-sparks-criticism-bitcoin-and-ether-slide-as-new-tariffs-take-effect-and-more"]},
    "2022-05-12": {
        "cause": "The Terra collapse. UST lost its peg from May 9 and LUNA went to about zero; the Luna Foundation Guard sold its BTC reserve, Tether briefly slipped under $0.99, and BTC hit about 26.7k on May 12, with more than $200B wiped from crypto in a day.",
        "resolved": "Forced selling ended once the LFG reserve was spent and Terra halted its chain. BTC bounced to about 30k within days, then ranged until the June credit unwind (analog #7).",
        "sources": ["https://cointelegraph.com/news/bitcoin-falls-below-27k-to-december-2020-lows-as-tether-stablecoin-peg-slips-under-99-cents",
                    "https://fortune.com/2022/05/12/crypto-crash-bitcoin-ether-bnb-luna-prices-dropping/",
                    "https://forklog.com/en/week-in-review-ust-stablecoin-crashes-crypto-market-as-elon-musk-freezes-twitter-deal/"]},
}


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def pct(x: float | None, k: int = 1, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:+.{k}f}%" if sign else f"{100 * x:.{k}f}%"


def era_of(t: datetime) -> str:
    d = t.date().isoformat()
    for a, b, name in ERAS:
        if a <= d <= b:
            return name
    return "other"


# ====================================================================== point-in-time context

def context(rows: list[dict], as_of: datetime) -> dict[str, dict]:
    from ai_trader.storage.lake import Lake

    lake = Lake()
    alt = lake.read_asof("alt_series", as_of, filters={"source": "alternative_me"})
    fng = alt.filter(pl.col("series_id") == "fear_greed").sort("known_ts")
    dv = lake.read_asof("alt_series", as_of, filters={"source": "deribit"}).filter((pl.col("series_id") == "dvol_close") & (pl.col("entity") == "BTC")).sort("known_ts")
    wiki = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "data/research/news/alt/wiki/pv_Bitcoin/*.parquet")))]).unique("date").sort("date")
    ev = pl.read_parquet(ROOT / "data/research/news/events/events.parquet")
    fk, fv = fng["known_ts"].dt.epoch("ms").to_numpy(), fng["value"].to_numpy()
    dk, dvv = dv["known_ts"].dt.epoch("ms").to_numpy(), dv["value"].to_numpy()
    wd, wv = wiki["date"].to_list(), np.log(np.maximum(wiki["value"].to_numpy().astype(float), 1))
    out = {}
    for r in rows:
        t = _ts(r["close_ts"]) if isinstance(r["close_ts"], str) else r["close_ts"]
        ms = int(t.timestamp() * 1000)
        c: dict[str, Any] = {}
        i = np.searchsorted(fk, ms, side="right") - 1
        if i >= 0:
            c["fng"] = float(fv[i])
            j = np.searchsorted(fk, ms - 7 * 86_400_000, side="right") - 1
            c["fng_7d_ago"] = float(fv[j]) if j >= 0 else None
        i = np.searchsorted(dk, ms, side="right") - 1
        if i >= 0 and ms - dk[i] < 6 * 3_600_000:
            lo = np.searchsorted(dk, ms - 365 * 86_400_000)
            c["dvol"] = float(dvv[i])
            c["dvol_pct_1y"] = float(np.mean(dvv[lo:i + 1] <= dvv[i])) if i - lo > 100 else None
        # Wikipedia 'Bitcoin' views of the last complete day (known after the day ends), z vs the trailing 90 days
        d0 = (t - timedelta(days=1)).date()
        k = int(np.searchsorted(np.array(wd, dtype="datetime64[D]"), np.datetime64(d0), side="right")) - 1
        if k >= 90:
            base = wv[k - 90:k]
            c["wiki_z"] = float((wv[k] - base.mean()) / base.std()) if base.std() > 0 else None
        e = ev.filter((pl.col("scheduled_ts") >= t - timedelta(days=3)) & (pl.col("scheduled_ts") <= t + timedelta(days=7))).sort("scheduled_ts")
        c["events_m3_p7"] = [f"{x['name']} {x['scheduled_ts']:%m-%d}" for x in e.iter_rows(named=True)]
        out[t.isoformat()] = c
    nowc = {}
    ms = int(as_of.timestamp() * 1000)
    i = np.searchsorted(fk, ms, side="right") - 1
    nowc["fng"] = float(fv[i]) if i >= 0 else None
    i = np.searchsorted(dk, ms, side="right") - 1
    lo = np.searchsorted(dk, ms - 365 * 86_400_000)
    nowc["dvol"] = float(dvv[i])
    nowc["dvol_pct_1y"] = float(np.mean(dvv[lo:i + 1] <= dvv[i]))
    lo90 = np.searchsorted(dk, ms - 90 * 86_400_000)
    nowc["dvol_pct_90d"] = float(np.mean(dvv[lo90:i + 1] <= dvv[i]))
    out["now"] = nowc
    return out


# ====================================================================== figures

def _style(plt) -> None:
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GREY,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def figures(S: dict, W: dict | None) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    figs: list[tuple[str, str]] = []
    now = S["now"]
    # 1. current-state panel: normalised log-price path per scale, BTC vs ETH / SOL
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.0))
    for ax, L in zip(axs, SCALES, strict=True):
        for sym, col, d in (("BTC", BLUE, now), ("ETH", ORANGE, S["secondary"]["ETHUSDT"]), ("SOL", AQUA, S["secondary"]["SOLUSDT"])):
            p = d["paths"].get(str(L))
            if p:
                x = np.arange(-L, 1)
                ax.plot(x, 100 * np.array(p["path"]), color=col, lw=1.6 if sym == "BTC" else 1.0, label=sym)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_title(f"last {L} h")
        ax.set_xlabel("hours to now")
    axs[0].set_ylabel("log return from window start, %")
    axs[0].legend(loc="lower left")
    fig.suptitle(f"BTC, ETH, SOL at the {now['as_of_close'][:16]} UTC close: the market's shape at five scales", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "current_state.png", dpi=130)
    plt.close(fig)
    figs.append(("current_state.png", "Normalised log-price path over the last 6h, 24h, 72h, 7d and 30d (BTC bold, ETH, SOL)."))
    # 2. overlay: preceding paths of the top-20 combined analogs vs now (last 30 d and last 7 d), aligned at t = 0
    rows = S["rankings"]["combined"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    cur = np.array(S["now_path_720"])
    for ax, L in zip(axs, (720, 168), strict=True):
        for r in rows:
            p = r.get("back_path_720")
            if p:
                p = np.array(p)[-(L + 1):]
                ax.plot(np.arange(-L, 1), 100 * (p - p[-1]), color=GREY, lw=0.7, alpha=0.6)
        c = cur[-(L + 1):]
        ax.plot(np.arange(-L, 1), 100 * (c - c[-1]), color=ORANGE, lw=2.0, label="now (BTC)")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_title(f"preceding {L // 24} days, rebased to 0 at the analog hour")
        ax.set_xlabel("hours before t")
        ax.set_ylabel("log price vs t, %")
    axs[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "overlay.png", dpi=130)
    plt.close(fig)
    figs.append(("overlay.png", "Preceding paths of the 20 combined analogs (grey) overlaid on the current BTC path (orange), rebased at t = 0."))
    # 3. fan chart: forward paths of the 20 combined analogs vs the unconditional fan
    fig, ax = plt.subplots(figsize=(9, 4))
    P = np.array([r["fwd_path"] for r in rows if r.get("fwd_path")])
    x = np.arange(P.shape[1])
    u = S["unc_fan"]
    ax.fill_between(u["h"], 100 * np.array(u["q10"]), 100 * np.array(u["q90"]), color=GREY, alpha=0.18, lw=0, label="unconditional 10-90%")
    ax.fill_between(u["h"], 100 * np.array(u["q25"]), 100 * np.array(u["q75"]), color=GREY, alpha=0.28, lw=0, label="unconditional 25-75%")
    ax.fill_between(x, 100 * np.quantile(P, 0.1, 0), 100 * np.quantile(P, 0.9, 0), color=BLUE, alpha=0.15, lw=0, label="analogs 10-90%")
    ax.fill_between(x, 100 * np.quantile(P, 0.25, 0), 100 * np.quantile(P, 0.75, 0), color=BLUE, alpha=0.25, lw=0, label="analogs 25-75%")
    for p in P:
        ax.plot(x, 100 * p, color=BLUE, lw=0.5, alpha=0.35)
    ax.plot(x, 100 * np.median(P, 0), color=BLUE, lw=2, label="analogs median")
    ax.plot(u["h"], 100 * np.array(u["q50"]), color=INK2, lw=1, ls="--", label="unconditional median")
    ax.axhline(0, color=GREY, lw=0.6)
    for h in (24, 72):
        ax.axvline(h, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("hours after the analog hour")
    ax.set_ylabel("forward log return, %")
    ax.set_title("What followed: forward 7-day paths of the 20 combined analogs vs every hour since 2020-03")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fan.png", dpi=130)
    plt.close(fig)
    figs.append(("fan.png", "Forward 7-day log-return paths of the 20 combined analogs (blue) against the unconditional fan of all pool hours (grey)."))
    # 4. timeline: analog dates per scale over the BTC history, by era
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 5.2), gridspec_kw={"height_ratios": [2, 1.3]})
    labels = [f"{L}h" for L in SCALES] + ["combined"]
    for y, k in enumerate(labels):
        ts = [_ts(r["close_ts"]) for r in S["rankings"][k]]
        ax.scatter(ts, [y] * len(ts), s=[60 - 2.2 * r["rank"] for r in S["rankings"][k]], color=ORANGE if k == "combined" else BLUE, alpha=0.75)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Where in history the top-20 analogs sit (bigger dot = higher rank)")
    for a, b, name in ERAS:
        if any(a <= r["close_ts"][:10] <= b for k in labels for r in S["rankings"][k]):
            ax.axvspan(datetime.fromisoformat(a), datetime.fromisoformat(b), color=GREY, alpha=0.08, lw=0)
    allr = [r for k in labels for r in S["rankings"][k]]
    eras = {}
    for r in allr:
        e = era_of(_ts(r["close_ts"]))
        eras.setdefault(e, {"combined": 0, "scales": 0})
        eras[e]["combined" if r["scale"] == "combined" else "scales"] += 1
    order = [name for _, _, name in ERAS if name in eras]
    xx = np.arange(len(order))
    ax2.bar(xx - 0.2, [eras[e]["scales"] / 5 for e in order], 0.4, color=BLUE, label="per-scale lists (mean count of the 5)")
    ax2.bar(xx + 0.2, [eras[e]["combined"] for e in order], 0.4, color=ORANGE, label="combined list")
    ax2.set_xticks(xx, order, rotation=25, ha="right", fontsize=7)
    ax2.set_ylabel("analogs (of 20)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "timeline.png", dpi=130)
    plt.close(fig)
    figs.append(("timeline.png", "Dates of the top-20 analogs per scale and combined, and their count by market episode."))
    # 5. outcome distributions: analogs vs unconditional, with the random-20 null band
    A = S["aggregate"]
    keys = ["r6", "r24", "r72", "r168", "mdd72", "mru72", "mdd168", "mru168"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, k in enumerate(keys):
        a = A.get(k)
        if not a:
            continue
        xs = np.array([r[k] for r in rows if r.get(k) is not None])
        ax.scatter(np.full(len(xs), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(xs)), 100 * xs, s=10, color=BLUE, alpha=0.5)
        ax.plot([i - 0.25, i + 0.25], [100 * a["median"]] * 2, color=BLUE, lw=2.5)
        ax.vlines(i + 0.32, 100 * a["unc"]["q25"], 100 * a["unc"]["q75"], color=GREY, lw=5)
        ax.vlines(i + 0.32, 100 * a["unc"]["q10"], 100 * a["unc"]["q90"], color=GREY, lw=1.5)
        ax.vlines(i - 0.32, 100 * a["null_band_median"][0], 100 * a["null_band_median"][1], color=ORANGE, lw=3)
    ax.set_xticks(range(len(keys)), ["+6h", "+24h", "+72h", "+7d", "maxDD 72h", "maxRU 72h", "maxDD 7d", "maxRU 7d"])
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylabel("log %")
    ax.set_title("20 combined analogs (dots, median bar) vs unconditional (grey, 10-90 / 25-75) and the median of 20 random hours (orange, 5-95%)")
    fig.tight_layout()
    fig.savefig(FIG / "outcomes.png", dpi=130)
    plt.close(fig)
    figs.append(("outcomes.png", "Outcomes of the 20 combined analogs against the unconditional distribution and the 5-95% band of the median of 20 random hours."))
    # 6. H-ANA1: QLIKE difference by segment and horizon
    if W:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, (lab, key, tk) in zip(axs, (("QLIKE analog - HAR (< 0 = analogs better)", "dqlike", "dqlike_t"), ("Brier skill of the sign", "brier_skill", None)), strict=True):
            xs, ys, es, cols = [], [], [], []
            for j, H in enumerate(("1", "3")):
                for k, sg in enumerate(("llm_in_sample", "clean_holdout")):
                    h = W["horizons"][H].get(sg, {})
                    if key not in h:
                        continue
                    xs.append(j * 3 + k)
                    ys.append(h[key])
                    if tk:
                        se = abs(h[key] / h[tk]) if h[tk] else 0
                        es.append([1.96 * se, 1.96 * se])
                    else:
                        ci = h["brier_skill_ci95"]
                        es.append([h[key] - ci[0], ci[1] - h[key]])
                    cols.append(BLUE if k == 0 else ORANGE)
            ax.bar(xs, ys, color=cols)
            ax.errorbar(xs, ys, yerr=np.array(es).T, fmt="none", ecolor=INK2, capsize=3)
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_xticks([0.5, 3.5], ["24h", "72h"])
            ax.set_title(lab)
        axs[0].text(0.02, 0.95, "blue = llm_in_sample, orange = clean_holdout; bars 95% (NW / block bootstrap)", transform=axs[0].transAxes, fontsize=7, va="top")
        fig.tight_layout()
        fig.savefig(FIG / "hana1.png", dpi=130)
        plt.close(fig)
        figs.append(("hana1.png", "H-ANA1: mean QLIKE difference (analogs minus log-HAR-t x weekday) and Brier skill of the sign, per horizon and segment."))
    return figs


# ====================================================================== rules

def rule_candidates(S: dict, W: dict | None) -> list[dict[str, Any]]:
    if not W:
        return []
    v = W["verdicts"]
    h1, h3 = W["horizons"]["1"], W["horizons"]["3"]

    def q(h: dict, sg: str) -> str:
        x = h.get(sg, {})
        return f"{x.get('dqlike', float('nan')):+.3f} (t {x.get('dqlike_t', float('nan')):+.1f})"

    def b(h: dict, sg: str) -> str:
        x = h.get(sg, {})
        ci = x.get("brier_skill_ci95", [float("nan")] * 2)
        return f"{x.get('brier_skill', float('nan')):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"

    # B (negative): contradicts in-sample and the holdout keeps the sign (analogs worse) at both horizons; else C
    keeps = all(W["horizons"][H].get("clean_holdout", {}).get("dqlike", -1) > 0 for H in ("1", "3"))
    grade = "B" if v["llm_in_sample"] == "contradicts" and keeps else "C"
    A = S["aggregate"]
    rules = [{
        "id": "R-ANA-1", "kind": "avoid",
        "rule": "WHEN the market looks like a past episode (multi-scale analog search on price shape, vol percentile, drawdown, cell, funding, OI, weekend) THEN do not use what followed the analogs as a forecast of the next 24h/72h move size; use log-HAR-t x weekday",
        "when": "an analog list or 'this looks like <date>' argument is offered for BTC",
        "then": "move-size expectations come from the volatility tools (log-HAR-t x weekday, field card), not from analog outcomes",
        "scope": {"assets": ["BTCUSDT"], "horizon": "24h-72h", "sides": "both"},
        "effect": f"QLIKE analog minus HAR: 24h in-sample {q(h1, 'llm_in_sample')}, holdout {q(h1, 'clean_holdout')}; 72h in-sample {q(h3, 'llm_in_sample')}, holdout {q(h3, 'clean_holdout')}; verdicts {v['llm_in_sample']} / {v['clean_holdout']}",
        "confidence": grade, "evidence": ["H-ANA1", "research/reports/analogs_2026-11-15.md"],
        "review": "a registered analog variant with QLIKE t < -2 vs log-HAR-t in both segments",
        "status": "active" if grade in ("A", "B") else "provisional"}]
    rules.append({
        "id": "R-ANA-2", "kind": "avoid",
        "rule": "WHEN reading the direction of past analogs THEN treat their up/down split as no information about the next 24h/72h sign",
        "when": "analog up-share is quoted as a direction call",
        "then": "no directional weight from analogs; the sign stays at the base rate",
        "scope": {"assets": ["BTCUSDT"], "horizon": "24h-72h", "sides": "both"},
        "effect": f"Brier skill vs expanding base rate: 24h in-sample {b(h1, 'llm_in_sample')}, holdout {b(h1, 'clean_holdout')}; 72h in-sample {b(h3, 'llm_in_sample')}, holdout {b(h3, 'clean_holdout')}",
        "confidence": "C", "evidence": ["H-ANA1", "research/reports/analogs_2026-11-15.md"],
        "review": "Brier skill CI above 0 in both segments for a registered variant",
        "status": "provisional"})
    a168 = A.get("r168", {})
    rules.append({
        "id": "R-ANA-3", "kind": "expect",
        "rule": "WHEN BTC is in a weekend post-crash state like 2026-11-15 (R9, 30d drawdown > 20%, rv pct > 0.95, funding pct < 0.1) THEN expect a two-branch week (policy or relief rebound vs a further leg down days later), not a known direction",
        "when": "cell R9, dd30 < -0.20, rv_pct > 0.95, funding_pct < 0.10, weekend",
        "then": "plan for both branches; the 24h low being undercut within the week is common; no edge in either direction from the analogs",
        "scope": {"assets": ["BTCUSDT"], "horizon": "1d-7d", "sides": "both"},
        "effect": f"20 combined analogs (chosen after seeing today): 7d median {pct(a168.get('median'))}, q10 {pct(a168.get('q10'))}, q90 {pct(a168.get('q90'))}, up {100 * a168.get('share_up', float('nan')):.0f}%; unconditional median {pct(a168.get('unc', {}).get('median'))}; trailing 24h low undercut within 7d {A['undercut24_any']}/20",
        "confidence": "C", "evidence": ["H-ANA1", "H-RB1", "research/reports/analogs_2026-11-15.md"],
        "review": "descriptive only; retire if the forward sample of such states shows a consistent direction",
        "status": "provisional"})
    return rules


# ====================================================================== markdown

def _row_outcome(r: dict) -> str:
    return (f"{pct(r.get('r6'))} | {pct(r.get('r24'))} | {pct(r.get('r72'))} | {pct(r.get('r168'))} | {pct(r.get('mdd72'))} / {pct(r.get('mru72'))} | "
            f"{pct(r.get('mdd168'))} / {pct(r.get('mru168'))} | {100 * r['rv7d']:.1f}% ({r['rv7d_ratio']:.2f}x) | {r.get('cell+1d')} > {r.get('cell+3d')} > {r.get('cell+7d')} | "
            f"{_uc(r, 24)} / {_uc(r, 72)} | {r.get('hours_to_low')}")


def _uc(r: dict, w: int) -> str:
    f = r.get(f"undercut{w}_first")
    if f is None:
        return "neither"
    if f == "tie":
        return f"tie {r.get(f'undercut{w}_h')}h"
    return f"yes {r.get(f'undercut{w}_h')}h" if f else ("later" if r.get(f"undercut{w}") else "no")


def markdown(S: dict, W: dict | None, C: dict, figs: list, rules: list) -> str:
    n = S["now"]
    e, s = S["secondary"]["ETHUSDT"], S["secondary"]["SOLUSDT"]
    nc = C["now"]
    A = S["aggregate"]
    L: list[str] = []
    a = L.append
    a("# BTC analogs: where today's market has been before, and what followed (2026-11-15)\n")
    a("Agent ANALOG. Research only. The design (distance, weights, pool, outcomes, H-ANA1 test) was frozen in the diary (2026-11-15 17:56 UTC, kind decision) and in `data/research/analogs/design.json` before any distance or outcome was computed. Code: `src/ai_trader/research/analog_study.py`, `analog_study_report.py`. Percent moves are log returns unless marked otherwise.\n")
    a("> **Read this first.** The analogs were chosen *after* seeing today's market. They describe the past and do not predict. "
      + (f"H-ANA1, the test of whether this kind of analog search forecasts anything, came out **{W['verdicts']['llm_in_sample']}** before the cutoff and **{W['verdicts']['clean_holdout']}** after it, so the analogs' outcomes below are history, not a forecast." if W else "") + "\n")
    a("**Data note.** The brief assumed Binance BTC 1h from 2017. The lake has no Binance BTC (venue=binance holds only ETHUSDT), and it was not backfilled (the user leads data decisions). "
      f"BTC 1h here is Deribit BTC-PERPETUAL from 2020-01-01 to the first Bybit bar ({S['data']['bybit_first'][:16]}), then Bybit BTCUSDT. The candidate pool runs {S['pool_first'][:10]} to {S['pool_last'][:10]} ({S['pool_n']:,} hours; the last 30 days are excluded). "
      "Funding exists from 2020-03-25 and OI from 2020-08-04. Missing state features are re-weighted (each distance is averaged over the features both hours have); in 2020, OI covers 48% of hours and funding 82%, and from 2021 on everything is covered. So 2017-2019 (including the 2018 bear) cannot appear as analogs.\n")

    a("## Part A: the market now\n")
    a(f"As of the {n['as_of_close'][:16]} UTC close (the latest closed 1h bar; {n['weekday']}, inside the TradFi weekend, {n.get('hours_into_weekend')} h in, {n.get('hours_to_cme_reopen', 0):.0f} h to the CME reopen at 23:00 UTC).\n")
    a("| | BTC | ETH | SOL |\n|---|---|---|---|")
    a(f"| close | {n['close']:,.0f} | {e['close']:,.1f} | {s['close']:,.2f} |")
    for Lh in SCALES:
        k = str(Lh)
        a(f"| {Lh}h return (range in vol units, position in range) | {pct(n['paths'][k]['ret'])} ({n['paths'][k]['range_vol_units']:.1f}, {n['paths'][k]['pos_in_range']:.2f}) | "
          f"{pct(e['paths'][k]['ret'])} ({(e['paths'][k]['range_vol_units'] or float('nan')):.1f}, {e['paths'][k]['pos_in_range']:.2f}) | "
          f"{pct(s['paths'][k]['ret'])} ({(s['paths'][k]['range_vol_units'] or float('nan')):.1f}, {s['paths'][k]['pos_in_range']:.2f}) |")
    for lab, k, k2 in (("24h realised vol, ann. (1y percentile)", "rv24_ann", "rv24_pct_1y"), ("7d realised vol, ann. (1y percentile)", "rv7d_ann", "rv7d_pct_1y")):
        a(f"| {lab} | {pct(n[k], 0, False)} (p{100 * n[k2]:.0f}) | {pct(e[k], 0, False)} (p{100 * e[k2]:.0f}) | {pct(s[k], 0, False)} (p{100 * s[k2]:.0f}) |")
    a(f"| drawdown from the 30d high | {pct(n['dd30'])} (high {n['high30']:,.0f}) | {pct(e['dd30'])} | {pct(s['dd30'])} |")
    a(f"| drawdown from the 1y high | {pct(n['dd1y'])} (high {n['high1y']:,.0f}, {n['high1y_ts'][:10]}) | {pct(e['dd1y'])} | {pct(s['dd1y'])} |")
    a(f"| v0 cell (1h) | {n['cell']} {CELL_LABELS.get(n['cell'], '')} | {e['cell']} | {s['cell']} |")
    a(f"| funding, last print (90d percentile) | {100 * n['funding_rate']:+.4f}% (p{100 * n['funding_pct_90d']:.0f}) | {100 * e['funding_rate']:+.4f}% (p{100 * e['funding_pct_90d']:.0f}) | {100 * s['funding_rate']:+.4f}% (p{100 * s['funding_pct_90d']:.0f}) |")
    a(f"| OI change 24h / 72h / 7d | {pct(n['oi24'])} / {pct(n['oi72'])} / {pct(n['oi7d'])} | {pct(e['oi24'])} / {pct(e['oi72'])} / {pct(e['oi7d'])} | {pct(s['oi24'])} / {pct(s['oi72'])} / {pct(s['oi7d'])} |")
    a(f"| 7d low (time) | {n['low7d']:,.0f} ({n['low7d_ts'][5:16]}) | {e['low7d']:,.1f} | {s['low7d']:,.2f} |\n")
    a(f"Other context (point in time): DVOL {nc['dvol']:.1f} (1y percentile p{100 * nc['dvol_pct_1y']:.0f}, 90d p{100 * nc['dvol_pct_90d']:.0f}); Fear & Greed {nc['fng']:.0f}. The BTC cell has been R9 for {n['cells_last_72h'][-1][1]} of the last 72 hours (before that, R8 for 7 h).\n")
    atr = n["atr14_1h"]
    c = n["close"]
    lv = [("24h high (the bounce high)", n["high24"]), ("72h high", n["high72"]), ("7d high", n["high7d"]), ("30d high", n["high30"]),
          ("flush low = 7d / 30d low (11-15 01:00)", n["low7d"]), ("1y low", n["low1y"]), ("round 65,000", 65000.0), ("round 60,000", 60000.0),
          ("24h VAH (desk geometry 17:23)", 65462.0), ("24h VAL (desk geometry 17:23)", 63338.0), ("est. long-liq cluster (desk, a model)", 61705.0),
          ("7d POC (desk)", 74455.0)]
    a("**Distance to key levels** (BTC; log %, and in 1h ATR14 units of " + f"{atr:,.0f}):\n")
    a("| level | price | distance | ATRs |\n|---|---|---|---|")
    for name, p in sorted(lv, key=lambda x: -x[1]):
        a(f"| {name} | {p:,.0f} | {pct(math.log(p / c), 2)} | {(p - c) / atr:+.1f} |")
    a("")
    a("**Shape in plain words.** BTC lost about 23% over 30 days and 16% over 7 days, with most of the 7-day loss (-10%) in the last 72 hours. Price sits in the bottom fifth of its 7-day and 30-day ranges, but is 64% up the last 24 hours' range after bouncing off the 61,899 flush low. The 72h range was 4.3 vol units (measured against the month before), the most stretched of the five scales. Short-term vol is at its 1-year extreme (p99), funding is near its 90-day low, and OI is up 2.4% in 72h while price fell (new shorts). ETH and SOL show the same picture with deeper 30-day drawdowns.\n")
    a("![current state](../figures/analogs_2026-11-15/current_state.png)\n")
    a("_" + figs[0][1] + "_\n")

    a("## Part B: the analog search\n")
    a("**How similarity is measured** (frozen before results). For each past hour and each scale (6h, 24h, 72h, 7d, 30d):\n"
      "- **shape**: 1 minus the Pearson correlation of the z-normalised cumulative log-price path (7 points for 6h, 25 for the others);\n"
      "- **amplitude**: |log ratio| of the path ranges in vol units (range / (sd of hourly returns over the 720 h before the window x sqrt L));\n"
      "- **state**: standardised Euclidean over rv percentile, drawdown from the 30d high, v0 cell (direction and vol, half weight each), funding percentile, 72h OI change and the weekend flag.\n\n"
      "d = 1.0 shape + 0.5 amplitude + 0.5 state. The **combined** score is the mean percentile rank across the five scales. At most one analog per 7 days (so one episode counts once); the last 30 days are excluded. Similarity in the tables is 1 - d for the per-scale lists, and 1 - mean rank for the combined list (0.99 = in the best 1% on average across scales).\n")
    a("### Top 20 combined (all scales together)\n")
    a("Columns: forward log returns; max drawdown / max run-up from the close; 7-day realised vol (and vs the 7 days before); v0 cell at +1d > +3d > +7d; whether the trailing 24h / 72h low was undercut *before* the trailing high was exceeded (hours after t; 'later' = undercut after the high was exceeded); hours to the 7-day low.\n")
    a("| # | analog hour (UTC) | day | episode | sim | rho 6h/24h/72h/7d/30d | cell | +6h | +24h | +72h | +7d | DD/RU 72h | DD/RU 7d | RV 7d | cells | 24h/72h low first | h to low |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in S["rankings"]["combined"]:
        t = _ts(r["close_ts"])
        rh = "/".join(f"{r[f'rho_{L_}']:.2f}" for L_ in SCALES)
        a(f"| {r['rank']} | {r['close_ts'][:16]} | {t:%a} | {era_of(t)} | {r['similarity']:.3f} | {rh} | {r['cell']} | {_row_outcome(r)} |")
    a("")
    a("### Top 20 per scale\n")
    a("Per scale, only the outcome columns that matter for reading the list; full rows are in the JSON.\n")
    for k in [f"{L_}h" for L_ in SCALES]:
        a(f"**{k}**\n")
        a("| # | analog hour | sim | rho | amp | state d | cell | +24h | +72h | +7d | 7d DD / RU |\n|---|---|---|---|---|---|---|---|---|---|---|")
        L_ = int(k[:-1])
        for r in S["rankings"][k]:
            a(f"| {r['rank']} | {r['close_ts'][:16]} | {r['similarity']:.3f} | {r[f'rho_{L_}']:.2f} | {r[f'amp_{L_}']:.2f} | {r['state_dist']:.2f} | {r['cell']} | {pct(r.get('r24'))} | {pct(r.get('r72'))} | {pct(r.get('r168'))} | {pct(r.get('mdd168'))} / {pct(r.get('mru168'))} |")
        g = S["aggregate_scales"][k]
        a(f"\n_{k} list: +24h median {pct(g['r24']['median'])}, +72h {pct(g['r72']['median'])}, +7d {pct(g['r168']['median'])} ({100 * g['r168']['share_up']:.0f}% up)._\n")
    a("### What followed, in aggregate (top 20 combined vs every hour)\n")
    a("'Random-20 band' is the 5-95% range of the median of 20 random pool hours (5000 draws): an analog median outside it is unusual for a random set of 20 hours, but still says nothing about forecasting (Part D).\n")
    a("| outcome | analogs median [90% bootstrap] | analogs q10 / q90 | share > 0 | unconditional median | unconditional q10 / q90 | random-20 band (median) | analogs sd vs random-20 sd band |")
    a("|---|---|---|---|---|---|---|---|")
    names = {"r6": "+6h", "r24": "+24h", "r72": "+72h", "r168": "+7d", "mdd72": "max DD 72h", "mru72": "max run-up 72h", "mdd168": "max DD 7d", "mru168": "max run-up 7d"}
    for k, nm in names.items():
        x = A.get(k)
        if not x:
            continue
        a(f"| {nm} | {pct(x['median'])} [{pct(x['median_ci'][0])}, {pct(x['median_ci'][1])}] | {pct(x['q10'])} / {pct(x['q90'])} | {100 * x['share_up']:.0f}% | {pct(x['unc']['median'])} | {pct(x['unc']['q10'])} / {pct(x['unc']['q90'])} | "
          f"[{pct(x['null_band_median'][0])}, {pct(x['null_band_median'][1])}] | {pct(x['sd'], 1, False)} vs [{pct(x['null_band_sd'][0], 1, False)}, {pct(x['null_band_sd'][1], 1, False)}] |")
    a("")
    u24, u72 = A["undercut24_first"], A["undercut72_first"]
    h2l = A["hours_to_low"]
    a(f"- **Path.** The trailing 24h low was undercut before the 24h high was exceeded in {u24['yes']} of 20 analogs ({A['undercut24_any']} undercut it at some point within 7 days); for the 72h low: {u72['yes']} first, {A['undercut72_any']} at any point. The 7-day low came a median {h2l['median']:.0f} h after the analog hour (IQR {h2l['q25']:.0f}-{h2l['q75']:.0f} h); {100 * h2l['share_within_24h']:.0f}% within 24 h.")
    a(f"- **Vol.** The following week's realised vol was a median {A['rv7d_ratio_median']:.2f}x the week before: about as high as the crash week, not a quick calm-down (the spread is wide: 0.5x to 1.9x).")
    a(f"- **A tilt up, but not a forecast.** {100 * A['r24']['share_up']:.0f}% of the 20 were up 24h later and the 7-day median was {pct(A['r168']['median'])}, outside the random-20 band; but the 7-day q10 was {pct(A['r168']['q10'])} and the 7-day max drawdown q10 {pct(A['mdd168']['q10'])} (unconditional {pct(A['mdd168']['unc']['q10'])}). The same tilt, measured walk-forward (Part D), has *negative* Brier skill in-sample: past analog up-shares made worse sign calls than the base rate. Part of the tilt comes from picking hours just after a low: the search matches the 24h bounce off the flush low, and several analog hours (#1, #5, #10, #20) were within 1-5 h of their 7-day low.")
    a(f"- **Regime.** v0 cell at +1d: {A['cells+1d']}; +3d: {A['cells+3d']}; +7d: {A['cells+7d']}. R9 hands over mostly to R6 (whipsaw) and R5, in line with R-PX-8 / RB1.")
    a("- **The spread is the message.** The set holds both policy-rescue rebounds (SVB 2023 +31% in 7 days, Williams Nov 2025) and 'the low was still ahead' legs (Jun 2022 Celsius, Nov 2025 week 1, Feb 2026 Warsh, May 2021). On these features the past doesn't tell you which branch you are on.\n")
    a("![fan](../figures/analogs_2026-11-15/fan.png)\n")
    a("_" + figs[2][1] + "_\n")
    a("![overlay](../figures/analogs_2026-11-15/overlay.png)\n")
    a("_" + figs[1][1] + "_\n")
    a("![outcomes](../figures/analogs_2026-11-15/outcomes.png)\n")
    a("_" + figs[4][1] + "_\n")
    a("### Where in history the analogs cluster\n")
    labels = [f"{L_}h" for L_ in SCALES] + ["combined"]
    by_era: dict[str, dict[str, int]] = {}
    by_year: dict[int, dict[str, int]] = {}
    cells: dict[str, int] = {}
    for k in labels:
        for r in S["rankings"][k]:
            t = _ts(r["close_ts"])
            by_era.setdefault(era_of(t), {}).setdefault(k, 0)
            by_era[era_of(t)][k] += 1
            by_year.setdefault(t.year, {}).setdefault(k, 0)
            by_year[t.year][k] += 1
            if k == "combined":
                cells[r["cell"] or "none (warm-up)"] = cells.get(r["cell"] or "none (warm-up)", 0) + 1
    a("| episode | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
    for _, _, name in ERAS:
        if name in by_era:
            a(f"| {name} | " + " | ".join(str(by_era[name].get(k, 0)) for k in labels) + " |")
    a("\n| year | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
    for y in sorted(by_year):
        a(f"| {y} | " + " | ".join(str(by_year[y].get(k, 0)) for k in labels) + " |")
    a(f"\nv0 cell at the combined analog hours: {cells}.\n")
    a("- The combined list is dominated by **recent bear-phase flushes**: Nov 2025, Feb and Jun 2026, Feb-Mar 2025. These have the same market structure as today (ETF-era, perp-heavy, weekend liquidity holes) and similar vol percentiles.")
    a("- The second cluster is **2022** (Jan Fed repricing x2, May LUNA, Jun Celsius/3AC, Nov FTX), then **2024** (May, Jul Mt Gox, Sep), with single episodes from 2020 (Mar 1, before the COVID crash), 2021 (May China ban) and 2023 (SVB). Aug 2024 (yen carry unwind) appears on the 72h and 7d lists but not in the combined top 20.")
    a("- The longer scales (7d, 30d) pick the same few months (Jun 2026, Nov 2025, Feb 2026, May 2022): few past months had a 23% 30-day loss with this shape. The 6h scale is the noisiest (only 7 points, so many weak matches across the whole history).")
    nwk = sum(1 for r in S["rankings"]["combined"] if r["state"].get("weekend"))
    a(f"- Weekends are over-represented: {nwk} of the 20 combined analog hours are inside the TradFi weekend, vs about 29% of all hours. The weekend flag is in the state vector, and many past flushes also bottomed on thin weekend liquidity.\n")
    a("![timeline](../figures/analogs_2026-11-15/timeline.png)\n")
    a("_" + figs[3][1] + "_\n")

    a("## Part C: why. What was happening at the top 10 combined analogs\n")
    a("Our own tables first. Point-in-time context at the analog hour: Fear & Greed (alternative.me, daily), DVOL and its 1y percentile (Deribit, from 2021-03), Wikipedia 'Bitcoin' views of the previous day as a z-score vs the 90 days before, and scheduled macro events from `data/research/news/events` within -3/+7 days. `research/narratives/events.yaml` and `research/calendar/` only cover 2026-11, so they add nothing for past analogs. The narrative for each date comes from a web search (treated as data, sources listed).\n")
    a("| # | analog | F&G (7d before) | DVOL (1y pct) | wiki z | scheduled events -3d..+7d |\n|---|---|---|---|---|---|")
    for r in S["rankings"]["combined"][:10]:
        cx = C.get(_ts(r["close_ts"]).isoformat(), {})
        dv = f"{cx['dvol']:.0f} (p{100 * cx['dvol_pct_1y']:.0f})" if cx.get("dvol") is not None and cx.get("dvol_pct_1y") is not None else "n/a"
        fg = f"{cx['fng']:.0f} ({cx['fng_7d_ago']:.0f})" if cx.get("fng") is not None and cx.get("fng_7d_ago") is not None else "n/a"
        wz = f"{cx['wiki_z']:+.1f}" if cx.get("wiki_z") is not None else "n/a"
        a(f"| {r['rank']} | {r['close_ts'][:16]} | {fg} | {dv} | {wz} | {', '.join(cx.get('events_m3_p7', [])) or '-'} |")
    a(f"| now | {n['as_of_close'][:16]} | {nc['fng']:.0f} | {nc['dvol']:.0f} (p{100 * nc['dvol_pct_1y']:.0f}) | | NVDA earnings 11-17, FOMC minutes 11-18 (desk calendar) |\n")
    for r in S["rankings"]["combined"][:10]:
        d = r["close_ts"][:10]
        w = WHY.get(d)
        t = _ts(r["close_ts"])
        a(f"**{r['rank']}. {r['close_ts'][:16]} UTC ({t:%a}), similarity {r['similarity']:.3f}: +24h {pct(r.get('r24'))}, +72h {pct(r.get('r72'))}, +7d {pct(r.get('r168'))}; 7d max DD {pct(r.get('mdd168'))}, low after {r.get('hours_to_low')} h.**")
        if w:
            a(f"Cause: {w['cause']} What resolved it: {w['resolved']} Sources: " + ", ".join(f"[{i + 1}]({u})" for i, u in enumerate(w["sources"])) + "\n")
        else:
            a("(no write-up)\n")
    a("**Pattern across the ten.** Five resolved through an outside catalyst arriving within days: SVB backstop, Williams' rate-cut signal, a sovereign seller finishing (Germany 2024), the Terra reserve exhausted, a policy headline (the crypto reserve). Two were just oversold relief rallies (Jan 2022, Jun 2026), and these chopped rather than trended. Three ran into a further leg down because the forced selling wasn't finished: a Fed shock on top of a credit unwind (Jun 2022), and ETF outflows plus fading cut odds (Nov 2025 week 1, Feb 2026). Today's setup (ETF outflow streak, a hawkish-Fed narrative, AI-capex doubt, and Nvidia earnings and FOMC minutes inside the week) has elements of both branches.\n")

    if W:
        a("## Part D: H-ANA1, does the analog search forecast anything? (walk-forward, as registered)\n")
        a("Origins are daily 08:00 UTC closes, the same origins as the project's log-HAR-t x weekday (`options_compression.predictive`: log-HAR-t + constant Kou jumps, weekday dummies, monthly expanding refits, 365-day warm-up). "
          "For each origin, the 20 declustered combined analogs come only from hours whose 72h outcome was known 7 days before the origin; feature sds also come from that pool. "
          "The analog variance forecast is vol-scaled: s168_t^2 x mean_k(R_k^2 / s168_k^2). HAR's is s^2 E[u^2]. Loss: QLIKE on R^2; paired difference, NW t (lag 7 / 9). Sign: p_up = (ups + 1) / 22 vs an expanding base rate, Brier skill with a 10-day block bootstrap.\n")
        a("| horizon | segment | origins (n eff: non-overlap / NW) | QLIKE analog | QLIKE HAR | diff (NW t) | share of days analog better | unscaled analog diff (t) | Brier skill [95%] | sign hit rate | verdict |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for H in ("1", "3"):
            for sg in ("llm_in_sample", "clean_holdout"):
                h = W["horizons"][H].get(sg, {})
                if "dqlike" not in h:
                    continue
                a(f"| {int(H) * 24}h | {sg} | {h['n']} ({h['n_eff_nonoverlap']:.0f} / {h['n_eff_nw']:.0f}) | {h['qlike_ana']:.3f} | {h['qlike_har']:.3f} | {h['dqlike']:+.3f} ({h['dqlike_t']:+.1f}) | {100 * h['share_ana_better']:.0f}% | "
                  f"{h['dqlike_raw']:+.3f} ({h['dqlike_raw_t']:+.1f}) | {h['brier_skill']:+.3f} [{h['brier_skill_ci95'][0]:+.3f}, {h['brier_skill_ci95'][1]:+.3f}] | {100 * h['sign_hit']:.0f}% | {W['verdicts'][sg]} |")
        a("")
        h1 = W["horizons"]["1"]["llm_in_sample"]
        a(f"Segments: llm_in_sample {h1['first'][:10]} to {h1['last'][:10]}; clean_holdout from 2026-07-01 ({W['horizons']['1']['clean_holdout'].get('n', 0)} origins, about {W['horizons']['1']['clean_holdout'].get('n', 0) / 30:.1f} months). "
          "Calibration (Mincer-Zarnowitz in logs, ln R^2 = a + b ln F), in-sample 24h: analog b " + f"{h1['mz_ana']['b']:.2f} (corr {h1['mz_ana']['corr_log']:.2f}), HAR b {h1['mz_har']['b']:.2f} (corr {h1['mz_har']['corr_log']:.2f}).\n")
        a(f"**Verdict: llm_in_sample {W['verdicts']['llm_in_sample']}, clean_holdout {W['verdicts']['clean_holdout']}.** " + verdict_text(W) + "\n")
        a("![H-ANA1](../figures/analogs_2026-11-15/hana1.png)\n")
        a("_" + figs[5][1] + "_\n")
        a("Trials: 1 hypothesis x 2 segments x 2 horizons x (QLIKE + Brier), plus 1 secondary variant (unscaled). Recorded with `hypotheses.record` for both segments.\n")

    a("## What this means\n")
    a("- **The analogs are a map of past episodes, not a forecast.** " + ("H-ANA1 shows that the same machinery, run walk-forward every day since 2021, does not beat log-HAR-t at forecasting the size of the next 24h/72h move, and has no skill on the sign. " if W else "") + "Use the volatility tools for size (R-ANA-1) and treat the sign as unknown (R-ANA-2, R-ML-5).")
    a("- **Descriptively, the analogs agree with RB1.** After states like today's, the following week was wide (7-day max drawdown and max run-up both large), the 24h low was often undercut, and R9 handed over to R6/R5. Policy or supply-exhaustion rescues produced the big up-weeks; unfinished forced selling produced the further legs down.")
    a("- **What to watch for this week** (from the 'why' pattern, not a signal): a catalyst that ends the forced selling (ETF flows turning, a dovish Fed read in the FOMC minutes on 11-18, Nvidia on 11-17 for the AI-capex leg) versus a continuation of outflows into thin liquidity.\n")
    a("## Rule candidates\n")
    a("Validated with `ai_trader.research.rules.validate` (no problems, alone and together with research/rules.yaml). Not merged into rules.yaml.\n")
    a("```yaml\n" + yaml.safe_dump(rules, sort_keys=False, allow_unicode=True, width=200) + "```\n")
    return "\n".join(L)


def verdict_text(W: dict) -> str:
    h1, h3 = W["horizons"]["1"], W["horizons"]["3"]
    s = []
    for sg in ("llm_in_sample", "clean_holdout"):
        a, b = h1.get(sg, {}), h3.get(sg, {})
        if "dqlike" in a and "dqlike" in b:
            s.append(f"{sg}: QLIKE difference {a['dqlike']:+.3f} (t {a['dqlike_t']:+.1f}) at 24h and {b['dqlike']:+.3f} (t {b['dqlike_t']:+.1f}) at 72h")
    return "; ".join(s) + ". A negative difference with t < -2 at both horizons was needed: instead the analog variance forecast is significantly *worse* than log-HAR-t in-sample (about 16% higher QLIKE), and still worse, though not significantly, in the holdout. The unscaled variant is no better (it only ties in the 72h holdout, t -0.3). The analogs' log-variance forecasts are also weakly correlated with what happened (MZ corr about 0.2 vs 0.36 for HAR). Brier skill of the sign is negative in-sample, with CIs below 0 at both horizons, and about zero in the holdout, as expected: direction is not predictable (R-ML-5)."


def build() -> None:
    S = json.loads((OUT / "search.json").read_text(encoding="utf-8"))
    wp = OUT / "walkforward.json"
    W = json.loads(wp.read_text(encoding="utf-8")) if wp.exists() else None
    as_of = _ts(S["as_of"])
    C = context(S["rankings"]["combined"], as_of)
    figs = figures(S, W)
    rules = rule_candidates(S, W)
    from ai_trader.research.rules import load_rules, validate

    errs = validate(rules) + validate(load_rules() + rules)
    if errs:
        print("RULE PROBLEMS:", errs)
    md = markdown(S, W, C, figs, rules)
    REPORT.with_suffix(".md").write_text(md, encoding="utf-8")
    slim = {k: v for k, v in S.items() if k not in ("rankings",)}
    slim["now"] = {k: v for k, v in S["now"].items() if k != "paths"} | {"paths": {L: {k: v for k, v in p.items() if k != "path"} for L, p in S["now"]["paths"].items()}}
    slim["secondary"] = {sym: {k: v for k, v in d.items() if k != "paths"} | {"paths": {L: {k: v for k, v in p.items() if k != "path"} for L, p in d["paths"].items()}} for sym, d in S["secondary"].items()}
    slim.pop("now_path_720", None)
    slim["rankings"] = {k: [{kk: vv for kk, vv in r.items() if kk not in ("fwd_path", "back_path_720")} for r in v] for k, v in S["rankings"].items()}
    out = {"report": "research/reports/analogs_2026-11-15.md", "search": slim, "walkforward": W, "context": C, "why": WHY, "rules": rules,
           "rule_problems": errs, "figures": [f"research/figures/analogs_2026-11-15/{f}" for f, _ in figs]}
    write_json(REPORT.with_suffix(".json"), out)
    print("wrote", REPORT.with_suffix(".md"), "rule problems:", errs)


if __name__ == "__main__":
    build()
