"""Deep analogs stage 2: fingerprints on an aligned clock, similarity per feature group, outcomes, level probabilities.
Reads the aggregates written by deep_analogs_study (data/research/deep_analogs/<EP>/) and writes
data/research/deep_analogs/results.json + aligned.parquet. Descriptive only (chosen after seeing today; H-ANA1 stands).

Feature definitions (T = analog hour = close time of the reference 1h bar; everything in the fingerprint uses data < T):
  flush low L, t_L     min 1m low (Binance UM perp) over [T-72h, T)
  shape    dd30 = ln(P_T / max 1m high over [T-30d, T)); ret72 = ln(P_T / P_{T-72h}); rebound = ln(P_T / L); h_since_low
  vol      rv24 = 5m-return RV over [T-24h, T), annualised (sqrt 365); rv_ratio = rv24 / median daily RV over [T-30d, T-3d);
           dvol_T, ln(dvol_T / median dvol 30d), ln(dvol_T / dvol_{T-72h}) (Deribit DVOL, lake)
  flow     climax = max hourly quote volume over [T-72h, T) / median hourly volume over [T-30d, T-3d); vol_at_low = the low
           hour's ratio; delta = (2 taker-buy - volume) / volume over [T-72h, T) and over [t_L, T); CVD lead = hours by which
           the 72h CVD minimum preceded the price low; retrace gap = price retrace off the low minus CVD retrace (both as
           a share of the 72h high-to-low leg); Bybit tick (T-24h, T): taker delta share, >= $250k print net share
  liq      proxy, same rule every episode: 1m bars with return < -3 sigma (1m sd over [T-30d, T-3d)), quote volume > 5x
           its median and taker sell share > 0.6 (forced selling); mirror for forced buying. Counts and notional (as hours
           of median volume) over [T-72h, T). Real prints: Bybit (lake, NOW only) and Binance CM BTCUSD_PERP (2023-06..)
  pos      Binance metrics: OI flush = ln(min OI in [t_L-24h, min(t_L+6h, T)) / max OI in [T-7d, t_L]); OI rebuild =
           ln(OI_T / that min); funding last print, 72h min, number of negative prints in 72h (Binance 8h); top-trader
           account and position L/S and global account L/S: level at T and ln change over 72h; taker L/S vol ratio 24h
  liquidity basis = 1e4 ln(perp / spot) (1m closes): mean of the last 6h, 72h min, change vs [T-72h, T-60h); Binance
           bookDepth +-1% notional: last hour and 72h min vs the [T-72h, T-48h) median, ln(bid/ask) at T (2023+);
           NOW: Bybit flow_3s depth within 100 bp is used because bookDepth for 2026-11-15 is not published yet
  sentiment Fear & Greed on T's date and its 7-day min
Similarity: per group, d = RMS over features both episodes have of (x_i - x_now) / sd_across_episodes; S = exp(-d^2/2).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research.deep_analogs_study import EPISODES, OUT, ep_T

H = timedelta(hours=1)
GROUPS: dict[str, list[str]] = {
    "shape": ["dd30", "ret72", "rebound", "h_since_low"],
    "vol": ["rv24", "rv_ratio", "dvol", "dvol_vs30", "dvol_chg72"],
    "flow": ["climax", "vol_at_low", "delta72", "delta_post_low", "cvd_lead_h", "retrace_gap", "bybit_delta24", "bybit_lg_net24"],
    "liquidation": ["n_forced_sell", "forced_sell_h", "n_forced_buy", "forced_sell_share"],
    "positioning": ["oi_flush", "oi_rebuild", "f_last_bp", "f_min72_bp", "n_neg72", "top_acc", "top_acc_chg72", "top_pos",
                    "top_pos_chg72", "glob_acc_chg72", "taker_ratio24"],
    "liquidity": ["basis6h", "basis_min72", "basis_chg", "depth_last_ratio", "depth_min_ratio", "depth_asym"],
    "sentiment": ["fng", "fng_min7"],
}
MICRO = ["flow", "liquidation", "positioning", "liquidity"]


def rd(ep: str, name: str) -> pl.DataFrame | None:
    p = OUT / ep / f"{name}.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p)
    tcol = "ts" if "ts" in d.columns else "event_ts"
    d = d.rename({tcol: "ts"}) if tcol != "ts" else d
    if d.schema["ts"] != pl.Datetime("us", "UTC"):
        d = d.with_columns(pl.col("ts").dt.cast_time_unit("us").dt.convert_time_zone("UTC") if d.schema["ts"].time_zone else
                           pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))
    return d.sort("ts")


def win(d: pl.DataFrame | None, a: datetime, b: datetime) -> pl.DataFrame | None:
    if d is None:
        return None
    x = d.filter((pl.col("ts") >= a) & (pl.col("ts") < b))
    return x if x.height else None


def fnum(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def asof_val(d: pl.DataFrame | None, t: datetime, col: str) -> float | None:
    if d is None:
        return None
    x = d.filter((pl.col("ts") < t) & pl.col(col).is_not_null())
    return fnum(x[col][-1]) if x.height else None


# ---------------------------------------------------------------------------------------------------------------- core


def hourly(k: pl.DataFrame, T: datetime) -> pl.DataFrame:
    return (k.with_columns(((pl.col("ts") - T).dt.total_minutes() // 60).cast(pl.Int64).alias("k"))
            .group_by("k").agg(pl.col("ts").first(), pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
                               pl.col("close").last(), pl.col("quote_volume").sum().alias("qv"),
                               pl.col("taker_buy_quote_volume").sum().alias("tbqv"))
            .sort("k"))


def fingerprint(ep: str, P0: float) -> dict[str, Any]:
    T = ep_T(ep)
    k = rd(ep, "bn_perp_1m")
    out: dict[str, Any] = {"ep": ep, "T": T.isoformat(), "label": {e[0]: e[2] for e in EPISODES}[ep]}
    f: dict[str, float | None] = {}
    pre = k.filter(pl.col("ts") < T)
    P_T = float(pre["close"][-1])
    w72 = pre.filter(pl.col("ts") >= T - 72 * H)
    iL = int(w72["low"].arg_min())
    L, tL = float(w72["low"][iL]), w72["ts"][iL]
    H30 = float(pre.filter(pl.col("ts") >= T - timedelta(days=30))["high"].max())
    H72 = float(w72["high"].max())
    out.update(P_T=P_T, low=L, t_low=tL.isoformat(), high30=H30, high72=H72)
    f["dd30"] = math.log(P_T / H30)
    f["ret72"] = math.log(P_T / float(w72["open"][0]))
    f["rebound"] = math.log(P_T / L)
    f["h_since_low"] = (T - tL).total_seconds() / 3600

    # vol: 5m RV
    k5 = pre.filter(pl.col("ts") >= T - timedelta(days=31)).group_by_dynamic("ts", every="5m").agg(pl.col("close").last())
    r5 = k5.with_columns(pl.col("close").log().diff().alias("r")).drop_nulls()
    rv_d = r5.group_by_dynamic("ts", every="1d", offset=f"{T.hour}h").agg((pl.col("r") ** 2).sum().sqrt().alias("rv"))
    last24 = r5.filter(pl.col("ts") >= T - 24 * H)
    f["rv24"] = float(math.sqrt((last24["r"] ** 2).sum()) * math.sqrt(365))
    base = rv_d.filter((pl.col("ts") >= T - timedelta(days=30)) & (pl.col("ts") < T - timedelta(days=3)))["rv"]
    f["rv_ratio"] = f["rv24"] / (float(base.median()) * math.sqrt(365)) if base.len() else None
    rv7 = r5.filter(pl.col("ts") >= T - timedelta(days=7))
    out["rv7"] = float(math.sqrt((rv7["r"] ** 2).sum() / 7) * math.sqrt(365))
    dv = rd(ep, "dvol_1h")
    if dv is not None and dv.height:
        dT = asof_val(dv, T, "value")
        d30 = win(dv, T - timedelta(days=30), T)
        d72 = asof_val(dv, T - 72 * H, "value")
        f["dvol"] = dT
        f["dvol_vs30"] = math.log(dT / float(d30["value"].median())) if dT and d30 is not None else None
        f["dvol_chg72"] = math.log(dT / d72) if dT and d72 else None

    # flow (Binance)
    hb = hourly(k.filter(pl.col("ts") >= T - timedelta(days=30)), T)
    med_v = float(hb.filter((pl.col("k") < -72))["qv"].median())
    w = hb.filter((pl.col("k") >= -72) & (pl.col("k") < 0))
    f["climax"] = float(w["qv"].max()) / med_v
    kL = int(((tL - T).total_seconds()) // 3600)
    f["vol_at_low"] = float(w.filter(pl.col("k") == kL)["qv"].sum()) / med_v
    dl = 2 * w["tbqv"] - w["qv"]
    f["delta72"] = float(dl.sum() / w["qv"].sum())
    wpl = w.filter(pl.col("k") >= kL)
    f["delta_post_low"] = float((2 * wpl["tbqv"] - wpl["qv"]).sum() / wpl["qv"].sum()) if wpl.height else None
    cvd = np.cumsum(dl.to_numpy()) / med_v
    kmin = int(w["k"][int(np.argmin(cvd))])
    f["cvd_lead_h"] = float(kL - kmin)
    cmax_pre = float(cvd[: int(np.argmin(cvd)) + 1].max()) if np.argmin(cvd) > 0 else float(cvd[0])
    c_ret = (cvd[-1] - cvd.min()) / (cmax_pre - cvd.min()) if cmax_pre > cvd.min() else 0.0
    p_ret = (P_T - L) / (H72 - L) if H72 > L else 0.0
    f["retrace_gap"] = float(p_ret - c_ret)
    out["cvd_retrace"], out["price_retrace"] = float(c_ret), float(p_ret)

    bt = rd(ep, "bybit_trades_1m")
    if ep == "NOW":
        fl = rd(ep, "bybit_flow_1m")
        if fl is not None:
            fl = fl.filter(pl.col("ts") >= datetime(2026, 11, 15, tzinfo=UTC)).select("ts", "buy_usd", "sell_usd", "n", "close", "low", "high")
            bt = pl.concat([bt, fl], how="diagonal_relaxed").sort("ts") if bt is not None else fl
    b24 = win(bt, T - 24 * H, T)
    if b24 is not None:
        f["bybit_delta24"] = float((b24["buy_usd"].sum() - b24["sell_usd"].sum()) / (b24["buy_usd"].sum() + b24["sell_usd"].sum()))
        if "lg_buy_usd" in b24.columns:
            lb, ls = b24["lg_buy_usd"].fill_null(0).sum(), b24["lg_sell_usd"].fill_null(0).sum()
            # NOW: 2026-11-15 rows come from flow_3s (no per-print sizes), so large prints cover only the archive part
            f["bybit_lg_net24"] = float((lb - ls) / (lb + ls)) if lb + ls > 0 else None
            out["bybit_lg_share24"] = float((lb + ls) / (b24["buy_usd"].sum() + b24["sell_usd"].sum()))

    # liquidation proxy
    r1 = k.filter(pl.col("ts") >= T - timedelta(days=30)).with_columns(pl.col("close").log().diff().alias("r"))
    bwin = r1.filter(pl.col("ts") < T - timedelta(days=3))
    sig, mv = float(bwin["r"].std()), float(bwin["quote_volume"].median())
    r1 = r1.with_columns(
        ((pl.col("r") < -3 * sig) & (pl.col("quote_volume") > 5 * mv) & (1 - pl.col("taker_buy_quote_volume") / pl.col("quote_volume") > 0.6)).alias("fs"),
        ((pl.col("r") > 3 * sig) & (pl.col("quote_volume") > 5 * mv) & (pl.col("taker_buy_quote_volume") / pl.col("quote_volume") > 0.6)).alias("fb"))
    out["liq_proxy_sig1m"], out["liq_proxy_medvol1m"] = sig, mv
    lw = r1.filter((pl.col("ts") >= T - 72 * H) & (pl.col("ts") < T))
    f["n_forced_sell"] = float(lw["fs"].sum())
    f["n_forced_buy"] = float(lw["fb"].sum())
    f["forced_sell_h"] = float(lw.filter(pl.col("fs"))["quote_volume"].sum() / med_v)
    fb_h = float(lw.filter(pl.col("fb"))["quote_volume"].sum() / med_v)
    f["forced_sell_share"] = f["forced_sell_h"] / (f["forced_sell_h"] + fb_h) if f["forced_sell_h"] + fb_h > 0 else None
    proxy_h = (r1.filter((pl.col("ts") >= T - 72 * H) & (pl.col("ts") < T + timedelta(days=7)))
               .with_columns(((pl.col("ts") - T).dt.total_minutes() // 60).cast(pl.Int64).alias("k"))
               .group_by("k").agg((pl.col("quote_volume") * pl.col("fs")).sum().alias("fs_usd"), (pl.col("quote_volume") * pl.col("fb")).sum().alias("fb_usd"))
               .sort("k"))
    real = {}
    cm = rd(ep, "bn_cm_liq")
    if cm is not None:
        cw = win(cm, T - 72 * H, T)
        if cw is not None:
            usd = cw.with_columns((pl.col("contracts") * 100).alias("usd"))
            real["binance_cm"] = {s: {"n": int(g.height), "usd": float(g["usd"].sum())} for s, g in
                                  ((s, usd.filter(pl.col("side") == s)) for s in ("SELL", "BUY"))}
            real["binance_cm_note"] = "BTCUSD_PERP coin-margined only; SELL = long liquidated; 1 contract = 100 USD"
    if ep == "NOW":
        lq = rd(ep, "bybit_liq")
        if lq is not None:
            lw2 = win(lq, T - 72 * H, T)
            u = lw2.with_columns((pl.col("price") * pl.col("size")).alias("usd"))
            real["bybit"] = {s: {"n": int(g.height), "usd": float(g["usd"].sum())} for s, g in
                             ((s, u.filter(pl.col("side") == s)) for s in ("long", "short"))}
    out["liq_real"] = real

    # positioning
    m = rd(ep, "bn_metrics")
    if m is not None:
        mp = m.filter(pl.col("ts") < T)
        oi = mp.filter(pl.col("sum_open_interest").is_not_null())
        omax = oi.filter((pl.col("ts") >= T - timedelta(days=7)) & (pl.col("ts") <= tL))["sum_open_interest"].max()
        omin_w = oi.filter((pl.col("ts") >= tL - 24 * H) & (pl.col("ts") < min(tL + 6 * H, T)))
        if omax is not None and omin_w.height:
            omin = float(omin_w["sum_open_interest"].min())
            f["oi_flush"] = math.log(omin / float(omax))
            f["oi_rebuild"] = math.log(float(oi["sum_open_interest"][-1]) / omin)
        for col, key in (("count_toptrader_long_short_ratio", "top_acc"), ("sum_toptrader_long_short_ratio", "top_pos"),
                         ("count_long_short_ratio", "glob_acc")):
            if col in mp.columns:
                now_v, then_v = asof_val(mp, T, col), asof_val(mp, T - 72 * H, col)
                if key != "glob_acc":
                    f[key] = now_v
                f[f"{key}_chg72"] = math.log(now_v / then_v) if now_v and then_v else None
        if "sum_taker_long_short_vol_ratio" in mp.columns:
            tr = win(mp, T - 24 * H, T)
            v = tr["sum_taker_long_short_vol_ratio"].drop_nulls() if tr is not None else None
            f["taker_ratio24"] = float(np.exp(np.log(v.to_numpy()).mean())) if v is not None and v.len() else None
    fr = rd(ep, "bn_funding")
    if fr is not None:
        fp = fr.filter(pl.col("ts") < T)
        f["f_last_bp"] = float(fp["rate"][-1]) * 1e4
        f72 = fp.filter(pl.col("ts") >= T - 72 * H)
        f["f_min72_bp"] = float(f72["rate"].min()) * 1e4
        f["n_neg72"] = float((f72["rate"] < 0).sum())

    # liquidity
    sp = rd(ep, "bn_spot_1m")
    if sp is not None:
        bs = k.select("ts", pl.col("close").alias("perp")).join(sp.select("ts", pl.col("close").alias("spot")), on="ts")
        bs = bs.with_columns((1e4 * (pl.col("perp") / pl.col("spot")).log()).alias("basis"))
        f["basis6h"] = fnum(win(bs, T - 6 * H, T)["basis"].mean()) if win(bs, T - 6 * H, T) is not None else None
        b72 = win(bs, T - 72 * H, T)
        if b72 is not None:
            hb_ = b72.group_by_dynamic("ts", every="1h").agg(pl.col("basis").mean())
            f["basis_min72"] = fnum(hb_["basis"].min())
            b0 = win(bs, T - 72 * H, T - 60 * H)
            f["basis_chg"] = f["basis6h"] - float(b0["basis"].mean()) if b0 is not None and f["basis6h"] is not None else None
    bd = rd(ep, "bn_bookdepth_1m")
    depth_src = None
    if ep == "NOW":
        fl = rd(ep, "bybit_flow_1m")
        if fl is not None and "dep_b100" in fl.columns:
            dd = fl.select("ts", (pl.col("dep_b100") + pl.col("dep_a100")).alias("dep"), pl.col("dep_b100").alias("b"), pl.col("dep_a100").alias("a"))
            depth_src = "bybit flow_3s within 100 bp"
    elif bd is not None:
        dd = (bd.filter(pl.col("pct").abs() == 1.0).pivot(on="pct", index="ts", values="notional")
              .rename({"-1.0": "b", "1.0": "a"}).with_columns((pl.col("b") + pl.col("a")).alias("dep")))
        depth_src = "binance bookDepth +-1%"
    if depth_src:
        hd = dd.filter((pl.col("ts") >= T - 72 * H) & (pl.col("ts") < T + timedelta(days=7))).group_by_dynamic("ts", every="1h").agg(
            pl.col("dep").mean(), pl.col("b").mean(), pl.col("a").mean())
        b0 = hd.filter(pl.col("ts") < T - 48 * H)["dep"].median()
        pre_d = hd.filter(pl.col("ts") < T)
        if b0 and pre_d.height:
            f["depth_last_ratio"] = float(pre_d["dep"][-1]) / float(b0)
            f["depth_min_ratio"] = float(pre_d["dep"].min()) / float(b0)
            f["depth_asym"] = math.log(float(pre_d["b"][-1]) / float(pre_d["a"][-1]))
            out["depth_hourly"] = {"src": depth_src, "k": [int((t - T).total_seconds() // 3600) for t in hd["ts"]],
                                   "ratio": [float(x) / float(b0) if x is not None else None for x in hd["dep"]]}
    out["depth_src"] = depth_src

    fg = rd(ep, "fng")
    if fg is not None:
        fgp = fg.filter(pl.col("ts") <= T)
        f["fng"] = fnum(fgp["value"][-1])
        f["fng_min7"] = fnum(fgp.filter(pl.col("ts") >= T - timedelta(days=7))["value"].min())

    out["features"] = {kk: (fnum(v) if v is not None else None) for kk, v in f.items()}
    out["aligned"] = aligned(ep, k, T, tL, med_v, proxy_h)
    out["outcome"] = outcome(ep, k, T, L, P_T, P0, out["rv7"])
    out["bybit_tick"] = bybit_tick_summary(bt, T)
    out["okx_check"] = okx_check(ep, k, T)
    out["ob200"] = ob200_summary(ep, T)
    return out


def aligned(ep: str, k: pl.DataFrame, T: datetime, tL: datetime, med_v: float, proxy_h: pl.DataFrame) -> pl.DataFrame:
    hb = hourly(k.filter((pl.col("ts") >= T - 72 * H) & (pl.col("ts") < T + timedelta(days=7))), T)
    P_T = float(k.filter(pl.col("ts") < T)["close"][-1])
    hb = hb.with_columns((pl.col("close") / P_T).log().alias("lp"), ((2 * pl.col("tbqv") - pl.col("qv")) / med_v).alias("dlt"),
                         (pl.col("qv") / med_v).alias("vratio"))
    hb = hb.with_columns(pl.col("dlt").cum_sum().alias("cvd"))
    kL = int((tL - T).total_seconds() // 3600)
    hb = hb.with_columns((pl.col("k") - kL).alias("kL"), pl.lit(ep).alias("ep"))
    m = rd(ep, "bn_metrics")
    if m is not None:
        mh = (m.filter((pl.col("ts") >= T - 72 * H) & (pl.col("ts") < T + timedelta(days=7)))
              .with_columns(((pl.col("ts") - T).dt.total_minutes() // 60).cast(pl.Int64).alias("k"))
              .group_by("k").agg(pl.col("sum_open_interest").last().alias("oi"), pl.col("count_toptrader_long_short_ratio").last().alias("top_acc"),
                                 pl.col("sum_toptrader_long_short_ratio").last().alias("top_pos")))
        oiT = asof_val(m, T, "sum_open_interest")
        hb = hb.join(mh, on="k", how="left").with_columns((pl.col("oi") / oiT).log().alias("loi"))
    fr = rd(ep, "bn_funding")
    if fr is not None:
        frk = fr.filter((pl.col("ts") >= T - 96 * H) & (pl.col("ts") < T + timedelta(days=7))).with_columns(
            ((pl.col("ts") - T).dt.total_minutes() // 60).cast(pl.Int64).alias("k"), (pl.col("rate") * 1e4).alias("fund_bp")).select("k", "fund_bp")
        hb = hb.sort("k").join_asof(frk.sort("k"), on="k")
    dv = rd(ep, "dvol_1h")
    if dv is not None:
        dk = dv.with_columns(((pl.col("ts") - T).dt.total_minutes() // 60).cast(pl.Int64).alias("k")).select("k", pl.col("value").alias("dvol"))
        hb = hb.join(dk, on="k", how="left")
    sp = rd(ep, "bn_spot_1m")
    if sp is not None:
        bs = (k.select("ts", pl.col("close").alias("perp")).join(sp.select("ts", pl.col("close").alias("spot")), on="ts")
              .with_columns((1e4 * (pl.col("perp") / pl.col("spot")).log()).alias("basis"),
                            ((pl.col("ts") - T).dt.total_minutes() // 60).cast(pl.Int64).alias("k")).group_by("k").agg(pl.col("basis").mean()))
        hb = hb.join(bs, on="k", how="left")
    hb = hb.join(proxy_h.with_columns((pl.col("fs_usd") / med_v).alias("fs_h"), (pl.col("fb_usd") / med_v).alias("fb_h")).select("k", "fs_h", "fb_h"),
                 on="k", how="left")
    return hb.drop("ts")


def first_hit(path: pl.DataFrame, T: datetime, P_T: float, up: float, dn: float, horizon_h: float) -> tuple[str, float | None]:
    """Which log barrier (vs P_T) is touched first within the horizon, on 1m high/low ('both' if in the same minute)."""
    p = path.filter((pl.col("ts") >= T) & (pl.col("ts") < T + timedelta(hours=horizon_h)))
    hu = (p["high"] >= P_T * math.exp(up)).to_numpy()
    hd = (p["low"] <= P_T * math.exp(dn)).to_numpy()
    iu = int(np.argmax(hu)) if hu.any() else None
    idn = int(np.argmax(hd)) if hd.any() else None
    if iu is None and idn is None:
        return "neither", None
    if idn is None or (iu is not None and iu < idn):
        return "up", float(iu / 60)
    if iu is None or idn < iu:
        return "down", float(idn / 60)
    return "both", float(iu / 60)


def outcome(ep: str, k: pl.DataFrame, T: datetime, L: float, P_T: float, P0: float, rv7: float) -> dict[str, Any]:
    o: dict[str, Any] = {}
    post = k.filter(pl.col("ts") >= T)
    by = rd(ep, "bybit_1h")
    o["data_until"] = post["ts"][-1].isoformat() if post.height else None
    if ep == "NOW" or post.height < 60:
        return o
    for hz in (7, 14):
        p = post.filter(pl.col("ts") < T + timedelta(days=hz))
        i = int(p["low"].arg_min())
        o[f"min{hz}d"] = float(p["low"][i])
        o[f"min{hz}d_vs_low"] = math.log(float(p["low"][i]) / L)
        o[f"min{hz}d_h"] = (p["ts"][i] - T).total_seconds() / 3600
        o[f"undercut{hz}d"] = bool(float(p["low"][i]) < L)
        j = int(p["high"].arg_max())
        o[f"max{hz}d_vs_PT"] = math.log(float(p["high"][j]) / P_T)
        o[f"max{hz}d_h"] = (p["ts"][j] - T).total_seconds() / 3600
        o[f"ret{hz}d"] = math.log(float(p["close"][-1]) / P_T)
    under = post.filter((pl.col("ts") < T + timedelta(days=14)) & (pl.col("low") < L))
    o["first_undercut_h"] = (under["ts"][0] - T).total_seconds() / 3600 if under.height else None
    pre_u = post.filter(pl.col("ts") < (under["ts"][0] if under.height else T + timedelta(days=14)))
    o["max_before_undercut_vs_PT"] = math.log(float(pre_u["high"].max()) / P_T) if pre_u.height else 0.0
    if by is not None:
        b30 = by.filter((pl.col("ts") >= T) & (pl.col("ts") < T + timedelta(days=30)))
        if b30.height:
            i = int(b30["low"].arg_min())
            o["min30d_vs_low"] = math.log(float(b30["low"][i]) / L)
            o["min30d_h"] = (b30["ts"][i] - T).total_seconds() / 3600
            o["max30d_vs_PT"] = math.log(float(b30["high"].max()) / P_T)
            o["ret30d"] = math.log(float(b30["close"][-1]) / P_T)
    # level stats: the user's levels as log distances from the current price P0, applied to P_T
    lv = levels(P0)
    for tag, scale in (("raw", 1.0), ("volscaled", rv7 / NOW_RV7[0] if NOW_RV7[0] else 1.0)):
        up, dn = lv["66000"] * scale, lv["63000"] * scale
        who, hrs = first_hit(post, T, P_T, up, dn, 14 * 24)
        s = {"first": who, "hours": hrs}
        p14 = post.filter(pl.col("ts") < T + timedelta(days=14))
        s["touch_dn"] = bool((p14["low"] <= P_T * math.exp(dn)).any())
        s["touch_up"] = bool((p14["high"] >= P_T * math.exp(up)).any())
        s["touch_58k"] = bool((p14["low"] <= P_T * math.exp(lv["58000"] * scale)).any())
        s["touch_68k"] = bool((p14["high"] >= P_T * math.exp(lv["68000"] * scale)).any())
        s["touch_72k"] = bool((p14["high"] >= P_T * math.exp(lv["72058"] * scale)).any())
        s["undercut_flush_equiv"] = bool((p14["low"] <= P_T * math.exp(lv["61899"] * scale)).any())
        # after the first barrier: did it extend before the opposite barrier?
        if who in ("down", "up"):
            t1 = T + timedelta(hours=hrs)
            rest = post.filter((pl.col("ts") >= t1) & (pl.col("ts") < T + timedelta(days=14)))
            tgt, opp = (lv["58000"] * scale, up) if who == "down" else (lv["68000"] * scale, dn)
            if who == "down":
                ht = (rest["low"] <= P_T * math.exp(tgt)).to_numpy()
                ho = (rest["high"] >= P_T * math.exp(opp)).to_numpy()
            else:
                ht = (rest["high"] >= P_T * math.exp(tgt)).to_numpy()
                ho = (rest["low"] <= P_T * math.exp(opp)).to_numpy()
            s["extend_before_reverse"] = bool(ht.any() and (not ho.any() or np.argmax(ht) < np.argmax(ho)))
        o[f"levels_{tag}"] = s
    o["vol_scale"] = rv7 / NOW_RV7[0] if NOW_RV7[0] else None
    return o


NOW_RV7: list[float] = [0.0]


def levels(P0: float) -> dict[str, float]:
    return {str(x): math.log(x / P0) for x in (58000, 61899, 63000, 66000, 68000, 72058)}


def bybit_tick_summary(bt: pl.DataFrame | None, T: datetime) -> dict[str, Any] | None:
    if bt is None:
        return None
    res = {}
    absd = (bt["buy_usd"] - bt["sell_usd"]).abs()
    thr = 10 * float(absd.median())
    for nm, a, b in (("pre24", T - 24 * H, T), ("post24", T, T + 24 * H)):
        w = win(bt, a, b)
        if w is None:
            continue
        net = w["buy_usd"] - w["sell_usd"]
        r = {"delta_share": float(net.sum() / (w["buy_usd"].sum() + w["sell_usd"].sum())),
             "burst_sell_min": int((net < -thr).sum()), "burst_buy_min": int((net > thr).sum()), "usd": float(w["buy_usd"].sum() + w["sell_usd"].sum())}
        if "lg_buy_usd" in w.columns:
            r["lg_buy_usd"], r["lg_sell_usd"] = float(w["lg_buy_usd"].fill_null(0).sum()), float(w["lg_sell_usd"].fill_null(0).sum())
            r["n_1m_buy"], r["n_1m_sell"] = int(w["n_1m_buy"].fill_null(0).sum()), int(w["n_1m_sell"].fill_null(0).sum())
        res[nm] = r
    res["burst_threshold_usd_per_min"] = thr
    return res


def okx_check(ep: str, k: pl.DataFrame, T: datetime) -> dict[str, Any] | None:
    ok = rd(ep, "okx_1h")
    if ok is None:
        return None
    hb = k.group_by_dynamic("ts", every="1h").agg(pl.col("close").last().alias("bn"))
    j = ok.select("ts", pl.col("close").alias("okx")).join(hb, on="ts")
    if not j.height:
        return None
    d = (1e4 * (j["okx"] / j["bn"]).log())
    lo = ok.filter((pl.col("ts") >= T - 72 * H) & (pl.col("ts") < T))
    return {"n": j.height, "median_gap_bp": float(d.median()), "max_abs_gap_bp": float(d.abs().max()),
            "okx_low72": float(lo["low"].min()) if lo.height else None}


def ob200_summary(ep: str, T: datetime) -> dict[str, Any] | None:
    ob = rd(ep, "bybit_ob200_1m")
    if ob is None:
        return None
    ob = ob.with_columns((pl.col("dep_b_all") + pl.col("dep_a_all")).alias("dep_all"), (pl.col("span_bid_bp") + pl.col("span_ask_bp")).alias("span"))
    hh = ob.group_by_dynamic("ts", every="1h").agg(pl.col("dep_all").median(), pl.col("span").median(), pl.col("spread_bp").mean(),
                                                     pl.col("dep_b10").median(), pl.col("dep_a10").median())
    i = int(hh["dep_all"].arg_min())
    return {"day": str(ob["ts"][0].date()), "dep200_median_usd": float(ob["dep_all"].median()), "dep200_min_hour": hh["ts"][i].isoformat(),
            "dep200_min_hour_ratio": float(hh["dep_all"][i] / ob["dep_all"].median()), "span_bp_median": float(ob["span"].median()),
            "span_bp_max_hour": float(hh["span"].max()), "dep10_bid_ask_median": [float(ob["dep_b10"].median()), float(ob["dep_a10"].median())]}


# ------------------------------------------------------------------------------------------------------------ similarity


def similarity(fps: dict[str, dict]) -> dict[str, Any]:
    eps = [e for e in fps if e != "NOW"]
    feats = sorted({f for g in GROUPS.values() for f in g})
    X = {f: {e: fps[e]["features"].get(f) for e in fps} for f in feats}
    sd = {}
    for f in feats:
        v = np.array([x for x in X[f].values() if x is not None], float)
        sd[f] = float(v.std(ddof=1)) if len(v) >= 3 and v.std() > 0 else None
    sims: dict[str, dict[str, float | None]] = {e: {} for e in eps}
    gaps: dict[str, dict[str, float]] = {e: {} for e in eps}
    for e in eps:
        for g, fl in GROUPS.items():
            z = []
            for f in fl:
                a, b = X[f][e], X[f]["NOW"]
                if a is None or b is None or sd[f] is None:
                    continue
                zz = (a - b) / sd[f]
                z.append(zz)
                gaps[e][f] = float(zz)
            sims[e][g] = float(math.exp(-0.5 * float(np.mean(np.square(z))))) if z else None
        gs = [sims[e][g] for g in GROUPS if sims[e][g] is not None]
        ms = [sims[e][g] for g in MICRO if sims[e][g] is not None]
        sims[e]["overall"] = float(np.mean(gs)) if gs else None
        sims[e]["micro"] = float(np.mean(ms)) if ms else None
        sims[e]["n_micro_groups"] = len(ms)
    return {"sd": sd, "sims": sims, "gaps": gaps}


# ---------------------------------------------------------------------------------------------------------- level models


def mc_barriers(sig_d: float, mu_d: float, lv: dict[str, float], days_: float, n: int = 40_000, dt_h: float = 0.25, seed: int = 7) -> dict[str, float]:
    """Brownian log-price (drift mu_d, vol sig_d per day) monitored every dt_h with the Broadie-Glasserman barrier shift."""
    rng = np.random.default_rng(seed)
    steps = int(days_ * 24 / dt_h)
    s = sig_d * math.sqrt(dt_h / 24)
    m = mu_d * dt_h / 24
    sh = 0.5826 * s
    up, dn = lv["66000"] - sh, lv["63000"] + sh
    l58, l68, l62 = lv["58000"] + sh, lv["68000"] - sh, lv["61899"] + sh
    res = {"up_first": 0, "dn_first": 0, "neither": 0, "t58": 0, "t68": 0, "t62": 0, "dn_then58": 0, "up_then68": 0, "t_dn": 0, "t_up": 0}
    chunk = 4000
    for c0 in range(0, n, chunk):
        x = np.zeros(chunk)
        first = np.zeros(chunk, np.int8)  # 1 up, -1 dn
        hit = {k: np.zeros(chunk, bool) for k in ("dn", "up", "58", "68", "62")}
        for _ in range(steps):
            x += m + s * rng.standard_normal(chunk)
            hit["dn"] |= x <= dn
            hit["up"] |= x >= up
            hit["58"] |= x <= l58
            hit["68"] |= x >= l68
            hit["62"] |= x <= l62
            first = np.where((first == 0) & (x >= up), 1, np.where((first == 0) & (x <= dn), -1, first))
        res["up_first"] += int((first == 1).sum())
        res["dn_first"] += int((first == -1).sum())
        res["neither"] += int((first == 0).sum())
        res["t58"] += int(hit["58"].sum())
        res["t68"] += int(hit["68"].sum())
        res["t62"] += int(hit["62"].sum())
        res["dn_then58"] += int((hit["dn"] & hit["58"]).sum())
        res["up_then68"] += int((hit["up"] & hit["68"]).sum())
        res["t_dn"] += int(hit["dn"].sum())
        res["t_up"] += int(hit["up"].sum())
    out = {k: v / n for k, v in res.items()}
    out["p58_given_dn"] = res["dn_then58"] / res["t_dn"] if res["t_dn"] else None
    out["p68_given_up"] = res["up_then68"] / res["t_up"] if res["t_up"] else None
    return out


def touch_prob(a: float, mu: float, sig: float, T: float) -> float:
    """P(max_{t<=T} W_t >= a) for W with drift mu, vol sig (a > 0); for a < 0 pass -a and -mu."""
    from scipy.stats import norm

    st = sig * math.sqrt(T)
    return float(norm.cdf((-a + mu * T) / st) + math.exp(2 * mu * a / sig**2) * norm.cdf((-a - mu * T) / st))


def fit_field(card: dict[str, dict[str, float]]) -> dict[str, float]:
    """Fit drift + vol (per day) of a Brownian log-price to the field card's 24h touch probabilities (+-1/2/3 %)."""
    from scipy.optimize import minimize

    pts = [(lev / 100, p) for lev, p in card.items()]

    def loss(th):
        mu, ls = th
        sig = math.exp(ls)
        e = 0.0
        for a, p in pts:
            q = touch_prob(a, mu, sig, 1.0) if a > 0 else touch_prob(-a, -mu, sig, 1.0)
            e += (q - p) ** 2
        return e

    r = minimize(loss, x0=[0.0, math.log(0.04)], method="Nelder-Mead")
    return {"mu_d": float(r.x[0]), "sig_d": float(math.exp(r.x[1])), "rmse": float(math.sqrt(r.fun / len(pts)))}


def top20_levels(P0: float) -> dict[str, Any]:
    """The top-20 combined analog list (analogs_2026-11-15) on Bybit/Deribit 1h bars from the lake: first passage of the
    same log barriers from each analog close (1h resolution; a bar touching both counts as 'both')."""
    from ai_trader.storage.lake import Lake

    rows = json.loads((ROOT_ANALOG / "search.json").read_text())["rankings"]["combined"]
    lk = Lake()
    bars = lk.read_asof("bars", datetime.now(UTC), symbols=["BTCUSDT"], filters={"venue": "bybit", "freq": "1h", "price_type": "last"},
                        start=datetime(2020, 3, 1, tzinfo=UTC)).select("event_ts", "high", "low", "close").sort("event_ts").unique("event_ts", keep="last").sort("event_ts")
    lv = levels(P0)
    out = []
    for r in rows:
        t = datetime.fromisoformat(r["close_ts"])
        P_T = float(r["close"])
        p = bars.filter((pl.col("event_ts") >= t) & (pl.col("event_ts") < t + timedelta(days=14)))
        if p.height < 300:
            continue
        hu = (p["high"] >= P_T * math.exp(lv["66000"])).to_numpy()
        hd = (p["low"] <= P_T * math.exp(lv["63000"])).to_numpy()
        iu = int(np.argmax(hu)) if hu.any() else 10**9
        idn = int(np.argmax(hd)) if hd.any() else 10**9
        first = "neither" if iu == idn == 10**9 else ("both" if iu == idn else ("up" if iu < idn else "down"))
        lo_before = float(bars.filter((pl.col("event_ts") >= t - timedelta(hours=72)) & (pl.col("event_ts") < t))["low"].min())
        out.append({"t": r["close_ts"], "first": first, "touch_dn": bool(hd.any()), "touch_up": bool(hu.any()),
                    "touch_58k": bool((p["low"] <= P_T * math.exp(lv["58000"])).any()), "touch_68k": bool((p["high"] >= P_T * math.exp(lv["68000"])).any()),
                    "lower_low14": bool(float(p["low"].min()) < lo_before), "min14_vs_low72": math.log(float(p["low"].min()) / lo_before)})
    return {"rows": out}


ROOT_ANALOG = OUT.parents[0] / "analogs"


def run(P0: float, field_card: dict[str, float]) -> dict[str, Any]:
    fps = {}
    NOW_RV7[0] = 0.0
    fpn = fingerprint("NOW", P0)
    NOW_RV7[0] = fpn["rv7"]
    fps["NOW"] = fpn
    for e, _, _ in EPISODES:
        if e != "NOW":
            fps[e] = fingerprint(e, P0)
    sim = similarity(fps)
    lv = levels(P0)
    rv7, rv24 = fpn["rv7"], fpn["features"]["rv24"]
    dvol = fpn["features"].get("dvol") or 44.0
    base = {}
    for nm, s_ann in (("rv7d", rv7), ("dvol", dvol / 100), ("rv24", rv24)):
        sd = s_ann / math.sqrt(365)
        base[nm] = {"sig_ann": s_ann, "d14": mc_barriers(sd, 0.0, lv, 14), "d1": mc_barriers(sd, 0.0, lv, 1)}
    # analytic infinite-horizon driftless: P(up first) = |dn| / (up + |dn|)
    base["analytic_inf"] = {"p_up_first": -lv["63000"] / (lv["66000"] - lv["63000"]),
                            "p58_given_63_before_66": (lv["66000"] - lv["63000"]) / (lv["66000"] - lv["58000"]),
                            "p68_given_66_before_63": (lv["66000"] - lv["63000"]) / (lv["68000"] - lv["63000"])}
    fit = fit_field(field_card)
    field = {"fit": fit, "card": field_card, "d1": mc_barriers(fit["sig_d"], fit["mu_d"], lv, 1),
             "d14_extrapolated": mc_barriers(fit["sig_d"], fit["mu_d"], lv, 14)}
    t20 = top20_levels(P0)
    al = pl.concat([fps[e].pop("aligned") for e in fps], how="diagonal_relaxed")
    al.write_parquet(OUT / "aligned.parquet")
    res = {"as_of": datetime.now(UTC).isoformat(timespec="seconds"), "P0": P0, "levels_log": lv, "fingerprints": fps, "similarity": sim,
           "baseline": base, "field": field, "top20": t20, "groups": GROUPS, "micro_groups": MICRO}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    import sys

    P0 = float(sys.argv[1])
    card = {1: 0.87, -1: 0.82, 2: 0.73, -2: 0.66, 3: 0.60, -3: 0.53}
    run(P0, card)
    print("ok")
