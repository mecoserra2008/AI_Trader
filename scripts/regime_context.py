"""Cross-asset regime context and historical analogs (descriptive; the analog *forecast* is registered as H-AN1/H-AN2).

    PYTHONIOENCODING=utf-8 uv run python scripts/regime_context.py [--as-of 2026-11-08T12:00:00Z] [--out DIR]

Point-in-time lake reads (Lake.read_asof). FRED market series (S&P 500, VIX, WTI, broad dollar, 2y/10y, HY OAS) come
from FRED's public CSV, latest vintage: these market series are not revised, so latest == point-in-time.

1. Crypto state per symbol: 9-cell regime on 1h and 4h (v0 labeller), returns, vol vs its 90d level, funding, OI.
2. Cross-asset: crypto vs TradFi perps on Bybit (gold, silver, WTI, Brent, QQQ, single stocks) and FRED macro.
3. Intra-week: daily returns matrix, the largest BTC hours with what else moved in the same hour, session variance
   shares, BTC-QQQ hourly correlation this week vs before.
4. Positioning / sentiment: Fear & Greed, DVOL and its premium over realised, funding cross-section, OI changes.
5. Analogs: daily state vectors since 2021-03 (DVOL start); nearest past dates (>= 21 days apart, last 45 days
   excluded), their next-7-day outcomes vs the unconditional distribution. Two feature sets: crypto-only and
   crypto + macro (macro from FRED since 2021).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.time import now
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import CELL_LABELS, label
from ai_trader.storage.lake import Lake

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
          "BCHUSDT", "DOTUSDT", "AAVEUSDT", "ETCUSDT", "FILUSDT", "XLMUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT",
          "SUIUSDT", "ENAUSDT", "HYPEUSDT"]
TRADFI = ["XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "QQQUSDT", "NVDAUSDT", "MSTRUSDT", "SPCXUSDT", "PAXGUSDT"]
FRED = {"SP500": "spx", "VIXCLS": "vix", "DCOILWTICO": "wti", "DTWEXBGS": "dxy", "DGS10": "y10", "DGS2": "y2", "BAMLH0A0HYM2": "hy"}
ALTS_OLD = ["LTCUSDT", "LINKUSDT", "BCHUSDT", "ETHUSDT", "DOTUSDT", "ADAUSDT", "AAVEUSDT", "XRPUSDT", "DOGEUSDT", "ETCUSDT",
            "BNBUSDT", "FILUSDT", "XLMUSDT", "TRXUSDT", "AVAXUSDT", "ATOMUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT"]


def fred(series: str, as_of: datetime) -> pl.DataFrame:
    with urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=30) as r:
        txt = r.read().decode()
    df = pl.read_csv(io.StringIO(txt), null_values=["."], try_parse_dates=True)
    df = df.rename({df.columns[0]: "d", df.columns[1]: "v"}).drop_nulls()
    # A daily observation is known after its US close; be conservative: usable from the next UTC day.
    return df.with_columns(pl.col("d").cast(pl.Date), pl.col("v").cast(pl.Float64)).filter(pl.col("d") < as_of.date())


def daily_closes(lake: Lake, sym: str, as_of: datetime, start: datetime) -> pl.DataFrame:
    b = lake.bars(sym, "1h", as_of, start=start).select("event_ts", "close", "high", "low").sort("event_ts")
    b = b.filter(pl.col("event_ts") + timedelta(hours=1) <= as_of)
    return (b.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
            .agg(pl.col("close").last(), pl.col("high").max(), pl.col("low").min(), pl.len().alias("n")).sort("d"))


def ann(x: float) -> float:
    return x * math.sqrt(365) * 100


def section_crypto(lake: Lake, t: datetime) -> pl.DataFrame:
    rows = []
    for s in CRYPTO + TRADFI:
        try:
            f1 = label(market_frame(s, "1h", t, start=t - timedelta(days=200), lake=lake, burn_in=False), 1.0)
            f4 = label(market_frame(s, "4h", t, start=t - timedelta(days=500), lake=lake, burn_in=False, oi=False), 4.0)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: {e}")
            continue
        if f1.height < 400:
            continue
        c = f1["close"].to_numpy()
        r = np.diff(np.log(c))
        last1, last4 = f1.tail(1).row(0, named=True), f4.tail(1).row(0, named=True) if f4.height else {}
        def ret(h):
            return 100 * (c[-1] / c[-1 - h] - 1) if len(c) > h else float("nan")
        rv7 = float(np.sqrt(np.mean(r[-168:] ** 2)) * math.sqrt(24 * 365) * 100)
        rv90 = float(np.sqrt(np.mean(r[-2160:] ** 2)) * math.sqrt(24 * 365) * 100)
        fr = last1.get("funding_rate")
        oi = f1["open_interest"].drop_nulls() if "open_interest" in f1.columns else pl.Series([])
        oi7 = 100 * (oi[-1] / oi[-min(len(oi), 169)] - 1) if len(oi) > 50 else float("nan")
        rows.append({"sym": s.removesuffix("USDT"), "cell_1h": last1.get("cell"), "cell_4h": last4.get("cell"),
                     "r24h": ret(24), "r7d": ret(168), "r30d": ret(720), "dd_30d_hi": 100 * (c[-1] / c[-720:].max() - 1),
                     "rv7": rv7, "rv90": rv90, "rv_ratio": rv7 / rv90 if rv90 else float("nan"),
                     "fund_ann": (fr or float("nan")) * 3 * 365 * 100, "oi_7d": oi7})
    return pl.DataFrame(rows)


def section_intraweek(lake: Lake, t: datetime) -> dict:
    start = t - timedelta(days=75)
    h = {}
    for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "QQQUSDT", "NVDAUSDT", "MSTRUSDT", "XAUUSDT", "CLUSDT", "BZUSDT", "XAGUSDT"]:
        b = lake.bars(s, "1h", t, start=start).select("event_ts", "close").sort("event_ts")
        h[s] = b.with_columns(pl.col("close").log().diff().alias(s.removesuffix("USDT"))).select("event_ts", s.removesuffix("USDT"))
    alts = []
    for s in ALTS_OLD:
        b = lake.bars(s, "1h", t, start=start).select("event_ts", "close").sort("event_ts")
        alts.append(b.with_columns(pl.col("close").log().diff().alias(s)).select("event_ts", s))
    wide = h["BTCUSDT"]
    for k, v in h.items():
        if k != "BTCUSDT":
            wide = wide.join(v, on="event_ts", how="left")
    a = alts[0]
    for v in alts[1:]:
        a = a.join(v, on="event_ts", how="left")
    a = a.with_columns(pl.mean_horizontal([c for c in a.columns if c != "event_ts"]).alias("ALTS")).select("event_ts", "ALTS")
    wide = wide.join(a, on="event_ts", how="left").sort("event_ts")
    wide = wide.with_columns([pl.col(c).fill_nan(None) for c in wide.columns if c != "event_ts"])

    wk = wide.filter(pl.col("event_ts") >= t - timedelta(days=9))
    daily = (wk.with_columns(pl.col("event_ts").dt.date().alias("day")).group_by("day")
             .agg([(100 * (pl.col(c).fill_null(0).sum().exp() - 1)).alias(c) for c in wide.columns if c != "event_ts"]).sort("day"))

    # Largest BTC hours in the last 7 days, scaled by the trailing 30-day hourly sigma.
    sig = float(wide.filter(pl.col("event_ts") < t - timedelta(days=7))["BTC"].drop_nulls().tail(720).std())
    last7 = wide.filter(pl.col("event_ts") >= t - timedelta(days=7))
    big = last7.with_columns((pl.col("BTC") / sig).alias("z")).filter(pl.col("z").abs() >= 2.5).sort("event_ts")
    big = big.with_columns([(pl.col(c) * 1e4).round(0) for c in wide.columns if c not in ("event_ts",)])

    def sess(df):
        hr = df["event_ts"].dt.hour().to_numpy()
        r2 = df["BTC"].fill_null(0).to_numpy() ** 2
        tot = r2.sum()
        cuts = {"Asia 00-08": (0, 8), "Europe 08-13": (8, 13), "US 13-21": (13, 21), "late US 21-24": (21, 24)}
        return {k: round(100 * r2[(hr >= a) & (hr < b)].sum() / tot, 1) for k, (a, b) in cuts.items()}

    prev = wide.filter((pl.col("event_ts") < t - timedelta(days=7)))
    def corr(df, x, y):
        d = df.select(x, y).drop_nulls().filter((pl.col(y) != 0))
        return round(float(np.corrcoef(d[x], d[y])[0, 1]), 2) if d.height > 30 else float("nan")
    corrs = {f"BTC~{y}": {"last_7d": corr(last7, "BTC", y), "prior_60d": corr(prev, "BTC", y)} for y in ["QQQ", "NVDA", "MSTR", "XAU", "CL", "ALTS", "ETH"]}
    rv = {k: {"last_7d_ann": round(float(last7[k].drop_nulls().std()) * math.sqrt(24 * 365) * 100, 1),
              "prior_60d_ann": round(float(prev[k].drop_nulls().std()) * math.sqrt(24 * 365) * 100, 1)} for k in ["BTC", "ETH", "ALTS", "QQQ", "NVDA", "XAU", "CL"]}
    return {"daily": daily, "big": big, "sess_last7": sess(last7), "sess_prior60": sess(prev), "corr": corrs, "rv": rv, "sigma_1h_bp": sig * 1e4}


def section_positioning(lake: Lake, t: datetime) -> dict:
    out = {}
    alt = lake.read_asof("alt_series", t, start=t - timedelta(days=800))
    fg = alt.filter(pl.col("series_id") == "fear_greed").sort("event_ts")
    if fg.height:
        v = fg["value"].to_numpy()
        out["fear_greed"] = {"now": v[-1], "7d_ago": v[-8], "30d_ago": v[-31], "pct_2y": round(100 * (v[-730:] < v[-1]).mean(), 0)}
    for e in ["BTC", "ETH"]:
        d = alt.filter((pl.col("series_id") == "dvol_close") & (pl.col("entity") == e)).sort("event_ts")
        if d.height:
            dd = d.group_by_dynamic("event_ts", every="1d").agg(pl.col("value").last()).sort("event_ts")
            v = dd["value"].to_numpy()
            out[f"dvol_{e}"] = {"now": round(float(d["value"][-1]), 1), "7d_ago": round(float(v[-8]), 1), "30d_ago": round(float(v[-31]), 1),
                                "pct_1y": round(100 * (v[-365:] < d["value"][-1]).mean(), 0)}
    st = alt.filter(pl.col("series_id") == "stablecoins_total_usd").sort("event_ts")
    if st.height:
        v = st["value"].to_numpy()
        out["stablecoins"] = {"now_bn": round(v[-1] / 1e9, 1), "chg_7d_pct": round(100 * (v[-1] / v[-8] - 1), 2), "chg_30d_pct": round(100 * (v[-1] / v[-31] - 1), 2)}
    f = lake.read_asof("funding", t, start=t - timedelta(days=3)).sort("event_ts").group_by("symbol").agg(pl.col("rate").last())
    f = f.filter(pl.col("symbol").is_in(CRYPTO))
    out["funding_xs"] = {"median_ann_pct": round(float(f["rate"].median()) * 3 * 365 * 100, 1), "share_negative": round(float((f["rate"] < 0).mean()), 2), "n": f.height}
    liq = lake.read_asof("liquidations", t, start=t - timedelta(days=2))
    if liq.height:
        l2 = liq.with_columns((pl.col("price") * pl.col("size")).alias("usd")).group_by("side").agg(pl.col("usd").sum(), pl.len())
        out["liquidations_since_collector_start"] = {r["side"]: {"usd": round(r["usd"]), "n": r["len"]} for r in l2.iter_rows(named=True)}
    return out


def state_vectors(lake: Lake, t: datetime) -> pl.DataFrame:
    """One row per UTC day since 2021-03-25 with the analog features (all causal: computed from data up to that day)."""
    start = datetime(2020, 9, 1, tzinfo=UTC)
    btc = daily_closes(lake, "BTCUSDT", t, start).filter(pl.col("n") >= 20)
    df = btc.select("d", pl.col("close").alias("btc"))
    lc = pl.col("btc").log()
    r1 = lc.diff()
    df = df.with_columns(
        (lc - lc.shift(7)).alias("r7"), (lc - lc.shift(30)).alias("r30"),
        (lc - pl.col("btc").rolling_max(90).log()).alias("dd90"),
        (r1.rolling_std(30) * math.sqrt(365)).alias("rv30"), (r1.rolling_std(7) * math.sqrt(365)).alias("rv7"),
    ).with_columns(pl.col("rv30").log().alias("lrv30"), (pl.col("rv7") / pl.col("rv30")).log().alias("rv_ratio"))
    # alt breadth and relative strength
    acl = []
    for s in ALTS_OLD:
        x = daily_closes(lake, s, t, start).filter(pl.col("n") >= 20).select("d", pl.col("close").alias(s))
        acl.append(x)
    a = acl[0]
    for x in acl[1:]:
        a = a.join(x, on="d", how="full", coalesce=True)
    a = a.sort("d")
    below = [(pl.col(s) < pl.col(s).rolling_mean(30)).cast(pl.Float64) for s in ALTS_OLD]
    r30s = [(pl.col(s).log() - pl.col(s).log().shift(30)) for s in ALTS_OLD]
    a = a.with_columns(pl.mean_horizontal(below).alias("breadth_below_ma30"), pl.mean_horizontal(r30s).alias("alt_r30"),
                       pl.mean_horizontal([(pl.col(s).log() - pl.col(s).log().shift(7)) for s in ALTS_OLD]).alias("alt_r7")).select("d", "breadth_below_ma30", "alt_r30", "alt_r7")
    df = df.join(a, on="d", how="left").with_columns((pl.col("alt_r30") - pl.col("r30")).alias("alt_rel30"))
    # DVOL, F&G, stablecoins, funding
    alt = lake.read_asof("alt_series", t, start=start)
    def daily_alt(sid, ent=None):
        x = alt.filter(pl.col("series_id") == sid)
        if ent:
            x = x.filter(pl.col("entity") == ent)
        return x.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d").agg(pl.col("value").last().alias(sid)).sort("d")
    df = df.join(daily_alt("dvol_close", "BTC"), on="d", how="left").join(daily_alt("fear_greed"), on="d", how="left") \
           .join(daily_alt("stablecoins_total_usd"), on="d", how="left").sort("d")
    df = df.with_columns(pl.col("stablecoins_total_usd").forward_fill(), pl.col("fear_greed").forward_fill())
    df = df.with_columns((pl.col("dvol_close") / 100).log().alias("ldvol"),
                         ((pl.col("dvol_close") / 100) / pl.col("rv30")).log().alias("vrp"),
                         (pl.col("stablecoins_total_usd").log() - pl.col("stablecoins_total_usd").log().shift(30)).alias("stable30"),
                         pl.col("fear_greed").alias("fng"))
    fu = lake.read_asof("funding", t, symbols=["BTCUSDT"], start=start).with_columns(pl.col("event_ts").dt.date().alias("d")) \
        .group_by("d").agg(pl.col("rate").mean().alias("fr")).sort("d")
    df = df.join(fu, on="d", how="left").with_columns((pl.col("fr").rolling_mean(7, min_samples=3) * 3 * 365).alias("fund7"))
    # FRED macro (known the next UTC day: shift the date by one)
    for sid, name in FRED.items():
        try:
            x = fred(sid, t).with_columns((pl.col("d") + timedelta(days=1)).alias("d")).rename({"v": name})
            df = df.join(x, on="d", how="left")
        except Exception as e:  # noqa: BLE001
            print(f"  FRED {sid}: {e}")
    df = df.sort("d").with_columns([pl.col(n).forward_fill(limit=10) for n in FRED.values()])
    df = df.with_columns((pl.col("spx").log() - pl.col("spx").log().shift(30)).alias("spx_r30"),
                         (pl.col("spx").log() - pl.col("spx").log().shift(7)).alias("spx_r7"),
                         pl.col("vix").log().alias("lvix"),
                         (pl.col("hy") - pl.col("hy").shift(30)).alias("hy_chg30"),
                         (pl.col("wti").log() - pl.col("wti").log().shift(30)).alias("wti_r30"),
                         (pl.col("dxy").log() - pl.col("dxy").log().shift(30)).alias("dxy_r30"))
    # forward outcomes (for analog reporting only; never used as features)
    df = df.with_columns(
        (lc.shift(-7) - lc).alias("fwd_r7"),
        (pl.col("alt_r7").shift(-7) - (lc.shift(-7) - lc)).alias("fwd_alt_rel7"),
        (r1.shift(-7).rolling_std(7).shift(-0) * math.sqrt(365)).alias("_tmp"),
    )
    rr = df["btc"].log().diff().to_numpy()
    n = len(rr)
    fwd_rv = np.full(n, np.nan)
    fwd_mdd = np.full(n, np.nan)
    fwd_mdu = np.full(n, np.nan)
    c = df["btc"].to_numpy()
    for i in range(n - 7):
        seg = rr[i + 1:i + 8]
        fwd_rv[i] = np.std(seg) * math.sqrt(365)
        path = c[i + 1:i + 8] / c[i]
        fwd_mdd[i] = path.min() - 1
        fwd_mdu[i] = path.max() - 1
    return df.drop("_tmp").with_columns(pl.Series("fwd_rv7", fwd_rv), pl.Series("fwd_mdd7", fwd_mdd), pl.Series("fwd_mdu7", fwd_mdu))


CRYPTO_FEATS = ["r7", "r30", "dd90", "lrv30", "rv_ratio", "ldvol", "vrp", "fng", "breadth_below_ma30", "alt_rel30", "stable30", "fund7"]
MACRO_FEATS = ["spx_r30", "spx_r7", "lvix", "hy", "hy_chg30", "wti_r30", "dxy_r30"]


def analogs(sv: pl.DataFrame, feats: list[str], k: int = 12, sep_days: int = 21, exclude_days: int = 45) -> tuple[pl.DataFrame, dict]:
    d = sv.drop_nulls(feats)
    cur = d.tail(1)
    hist = d.filter(pl.col("d") <= cur["d"][0] - timedelta(days=exclude_days)).drop_nulls(["fwd_r7", "fwd_rv7"])
    mu = {f: d[f].mean() for f in feats}
    sd = {f: d[f].std() for f in feats}
    z = lambda df: np.column_stack([((df[f] - mu[f]) / sd[f]).to_numpy() for f in feats])  # noqa: E731
    dist = np.sqrt(((z(hist) - z(cur)) ** 2).mean(axis=1))
    hist = hist.with_columns(pl.Series("dist", dist)).sort("dist")
    picked, days = [], []
    for r in hist.iter_rows(named=True):
        if all(abs((r["d"] - x).days) >= sep_days for x in days):
            picked.append(r)
            days.append(r["d"])
        if len(picked) == k:
            break
    an = pl.DataFrame(picked)
    unc = d.drop_nulls(["fwd_r7", "fwd_rv7"])
    def summ(x):
        return {"fwd_r7_mean_%": round(100 * float(x["fwd_r7"].mean()), 2), "fwd_r7_median_%": round(100 * float(x["fwd_r7"].median()), 2),
                "p_up": round(float((x["fwd_r7"] > 0).mean()), 2), "fwd_rv7_median_%": round(100 * float(x["fwd_rv7"].median()), 1),
                "fwd_maxdd7_median_%": round(100 * float(x["fwd_mdd7"].median()), 2), "fwd_maxdu7_median_%": round(100 * float(x["fwd_mdu7"].median()), 2),
                "fwd_alt_minus_btc7_mean_%": round(100 * float(x["fwd_alt_rel7"].drop_nulls().mean()), 2), "n": x.height}
    cur_row = {f: round(float(cur[f][0]), 3) for f in feats}
    zc = {f: round(float((cur[f][0] - mu[f]) / sd[f]), 2) for f in feats}
    return an, {"current_date": str(cur["d"][0]), "current": cur_row, "current_z": zc, "analogs": summ(an), "unconditional": summ(unc),
                "rv30_now_%": round(100 * float(cur["rv30"][0]), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--out", default="research/reports")
    a = ap.parse_args()
    t = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")) if a.as_of else now()
    lake = Lake()
    pl.Config.set_tbl_rows(60), pl.Config.set_tbl_cols(30), pl.Config.set_tbl_width_chars(260), pl.Config.set_float_precision(2)
    print(f"# regime context as of {t:%Y-%m-%d %H:%M} UTC\n")
    cs = section_crypto(lake, t)
    print("## 1. per-symbol state (cells:", ", ".join(f"{k} {v}" for k, v in CELL_LABELS.items()), ")")
    print(cs)
    print("\ncell counts 1h:", cs.group_by("cell_1h").len().sort("len", descending=True).rows(), " 4h:", cs.group_by("cell_4h").len().sort("len", descending=True).rows())
    iw = section_intraweek(lake, t)
    print("\n## 2. daily returns, last 9 UTC days (%)")
    print(iw["daily"])
    print(f"\n## 3. BTC hours with |z| >= 2.5 in the last 7 days (bp; 30d hourly sigma {iw['sigma_1h_bp']:.0f} bp)")
    print(iw["big"])
    print("\nBTC variance share by session, last 7d:", iw["sess_last7"], " prior 60d:", iw["sess_prior60"])
    print("hourly correlations:", json.dumps(iw["corr"]))
    print("realised vol (ann %):", json.dumps(iw["rv"]))
    pos = section_positioning(lake, t)
    print("\n## 4. positioning / sentiment")
    print(json.dumps(pos, indent=1, default=str))
    sv = state_vectors(lake, t)
    macro_now = sv.drop_nulls(["spx"]).tail(8).select("d", "spx", "vix", "wti", "dxy", "y2", "y10", "hy")
    print("\n## 5. FRED macro, last observations (date = first UTC day the print is known)")
    print(macro_now)
    out = {}
    for name, feats in [("crypto", CRYPTO_FEATS), ("crypto_macro", CRYPTO_FEATS + MACRO_FEATS)]:
        an, s = analogs(sv, feats)
        out[name] = s
        print(f"\n## 6. analogs ({name}): current {s['current_date']}")
        print("current z-scores:", s["current_z"])
        print(an.select("d", "dist", "r7", "r30", "dd90", "rv30", "dvol_close", "fng", "breadth_below_ma30", "fwd_r7", "fwd_rv7", "fwd_mdd7", "fwd_mdu7", "fwd_alt_rel7"))
        print("analogs:", s["analogs"])
        print("unconditional:", s["unconditional"], " rv30 now %:", s["rv30_now_%"])
    p = Path(a.out) / f"regime_context_{t:%Y-%m-%d}.json"
    p.write_text(json.dumps({"as_of": t.isoformat(), "positioning": pos, "analogs": out, "intraweek": {k: v for k, v in iw.items() if k in ("sess_last7", "sess_prior60", "corr", "rv")}}, indent=1, default=str))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
