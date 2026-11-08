"""Studies for H-VCL2..H-VCL5 (extended HAR), H-FU1 / H-FU2 (stablecoins, VRP) and H-C1 (funding carry).

Design (RESEARCH_PROTOCOL §3b), fixed before any result was seen:
  * Everything is computed causally on each symbol's full history, then split by segment.
  * Decision time for day d = (d+1) 00:05 UTC: the last 1h bar of day d, the 23:00 DVOL print (known 00:00:02)
    and the 08:00 SVI fit of day d are all known by then. Alt data are read with known_ts <= decision time.
  * Daily RV = Σ squared 1h log returns over UTC days with 24 bars (H-VCL1 construction).
  * Segment by forecast origin; an observation whose outcome window straddles the knowledge cutoff is dropped
    from both segments (in-sample outcomes must not contain holdout data).
  * Cross-symbol statistics average per origin date first (clustering by time), then Newey-West t.
  * Units: DVOL in vol points, SVI ATM IV in decimals; both -> daily variance (IV)^2/365, the RV unit.
  * Fixed choices: RS- floored at 1% of RV before the log (H-VCL2); DVOL is the primary IV for H-VCL4,
    SVI iv_atm_7 a secondary variant; H-VCL5 compression = 20-day RV at or below the 10th percentile of
    the previous 365 daily values (>= 300 finite); HAR horizons are direct (target log Σ RV over h days);
    H-FU2 uses a direct 30-day HAR; regressions use HAC lag max(NW auto, 5) for the overlapping 30d outcomes;
    H-C1 rebalances Mondays 00:05 UTC, signal = mean funding settled in (t-7d, t], P&L = settlements in
    (t, t+7d], funding burn-in = market_frame's listing burn-in applied to the first settlement.

Run:  uv run python -m ai_trader.research.vol_fund_study insample|holdout   (in-sample first; recorded before the
holdout is computed).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features.vol_models_x import har_forecast_x, iv_to_daily_var, qlike
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t
from ai_trader.research.frame import market_frame
from ai_trader.research.verdicts import MIN_HOLDOUT, _judge
from ai_trader.storage.lake import Lake

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
DECISION_LAG = timedelta(minutes=5)
SEGS = ("llm_in_sample", "clean_holdout")
COST_PER_LEG = 0.001  # 10 bps per leg per unit of notional changed


# ------------------------------------------------------------------ data

def _ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def daily_panel(symbols: list[str], as_of: datetime | None = None) -> dict[str, pl.DataFrame]:
    """Per symbol: complete UTC days (24 bars) with rv, rs_neg, ret (daily log return), close (last), plus
    `origin` = (d+1) 00:00 UTC. Also returns the full close series at each day end for return studies."""
    as_of = as_of or now()
    out = {}
    for sym in symbols:
        f = market_frame(sym, "1h", as_of, funding=False, oi=False)
        if f.height < 24 * 60:
            continue
        r = pl.col("ret").fill_null(0)
        dly = (f.with_columns(pl.col("event_ts").dt.date().alias("d"))
               .group_by("d").agg((r ** 2).sum().alias("rv"), pl.when(r < 0).then(r ** 2).otherwise(0.0).sum().alias("rs_neg"),
                                   r.sum().alias("ret"), pl.col("close").last().alias("close"), pl.len().alias("n"),
                                   pl.col("event_ts").max().alias("last_bar"))
               .sort("d"))
        out[sym] = dly
    return out


def _complete(dly: pl.DataFrame) -> pl.DataFrame:
    return dly.filter(pl.col("n") == 24).with_columns(pl.col("d").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("day_ts"))


def _contiguous(days: np.ndarray, h: int) -> np.ndarray:
    """True at index d if days[d+1..d+h] are the h calendar days after days[d]."""
    n = len(days)
    ok = np.zeros(n, dtype=bool)
    if n > h:
        dd = days.astype("datetime64[D]").astype(np.int64)
        ok[: n - h] = (dd[h:] - dd[: n - h]) == h
    return ok


def alt_series(source: str, series: str, entity: str, as_of: datetime | None = None) -> pl.DataFrame:
    a = Lake().read_asof("alt_series", as_of or now(), filters={"source": source, "series_id": series, "entity": entity})
    return a.select("event_ts", "known_ts", "value").sort("known_ts")


def _asof_value(origins: pl.Series, alt: pl.DataFrame, max_stale: timedelta) -> np.ndarray:
    """Latest alt value with known_ts <= origin + DECISION_LAG and event_ts >= decision - max_stale."""
    q = pl.DataFrame({"decision": origins + DECISION_LAG}).with_row_index("i").sort("decision")
    j = q.join_asof(alt.rename({"known_ts": "decision_k"}), left_on="decision", right_on="decision_k", strategy="backward")
    j = j.with_columns(pl.when(pl.col("event_ts") >= pl.col("decision") - max_stale).then(pl.col("value")).otherwise(None).alias("v")).sort("i")
    return j["v"].cast(pl.Float64).fill_null(np.nan).to_numpy()


def split(df: pl.DataFrame, start: str = "origin", end: str = "end") -> dict[str, pl.DataFrame]:
    k = knowledge_cutoff()
    return {"llm_in_sample": df.filter(pl.col(end) <= k), "clean_holdout": df.filter(pl.col(start) >= k)}


def ols_hac(y: np.ndarray, x: np.ndarray, min_lag: int = 5) -> dict:
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 5:
        return {"slope": float("nan"), "t": float("nan"), "n": n}
    xc = x - x.mean()
    b = float(xc @ (y - y.mean()) / (xc @ xc))
    e = y - y.mean() - b * xc
    u = xc * e
    lag = max(int(np.floor(4 * (n / 100) ** (2 / 9))), min_lag)
    lag = min(lag, n - 1)
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    se = np.sqrt(max(s, 1e-300) * n) / (xc @ xc)
    return {"slope": b, "t": b / se, "n": n, "slope_per_sd": b * float(x.std()), "corr": float(np.corrcoef(x, y)[0, 1])}


def _paired(panel: pl.DataFrame, a: str, b: str) -> dict:
    """mean over origins of the cross-symbol mean of (q_b - q_a), NW t."""
    g = panel.group_by("origin").agg((pl.col(b) - pl.col(a)).mean().alias("dq"), pl.col(a).mean(), pl.col(b).mean()).sort("origin")
    m, t, n = newey_west_t(g["dq"].to_numpy())
    return {"dqlike": m, "t": t, "n_origins": n, f"qlike_{a}": float(g[a].mean()) if g.height else float("nan"),
            f"qlike_{b}": float(g[b].mean()) if g.height else float("nan"), "n_obs": panel.height}


# ------------------------------------------------------------------ H-VCL2 / H-VCL3 (next-day, H-VCL1 design)

def _next_day_rows(sym: str, c: pl.DataFrame, f0: np.ndarray, f1: np.ndarray) -> list[dict]:
    rv = c["rv"].to_numpy()
    days = c["day_ts"].to_list()
    rows = []
    for d in range(len(rv) - 1):
        if np.isfinite(f0[d]) and np.isfinite(f1[d]) and rv[d + 1] > 0:
            o = days[d] + timedelta(days=1)
            rows.append({"symbol": sym, "origin": o, "end": days[d + 1] + timedelta(days=1),
                         "q_har": float(qlike(rv[d + 1], f0[d])), "q_x": float(qlike(rv[d + 1], f1[d]))})
    return rows


def vcl2_panel(panel: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for sym, dly in panel.items():
        c = _complete(dly)
        rv, rs = c["rv"].to_numpy(), c["rs_neg"].to_numpy()
        ex = np.log(np.maximum(np.maximum(rs, 0.01 * rv), 1e-12))
        rows += _next_day_rows(sym, c, har_forecast_x(rv), har_forecast_x(rv, ex))
    return pl.DataFrame(rows)


def vcl3_panel(panel: dict[str, pl.DataFrame]) -> pl.DataFrame:
    btc = _complete(panel["BTCUSDT"])
    lb = np.log(np.maximum(btc["rv"].to_numpy(), 1e-12))
    btc = btc.select("d", pl.Series("b_d", lb), pl.Series("b_w", lb).rolling_mean(7).alias("b_w"))
    rows = []
    for sym, dly in panel.items():
        if sym == "BTCUSDT":
            continue
        c = _complete(dly).join(btc, on="d", how="left").sort("d")
        ex = np.column_stack([c["b_d"].fill_null(np.nan).to_numpy(), c["b_w"].fill_null(np.nan).to_numpy()])
        rv = c["rv"].to_numpy()
        rows += _next_day_rows(sym, c, har_forecast_x(rv), har_forecast_x(rv, ex))
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ H-VCL4 (7-day, implied vol)

def vcl4_panel(panel: dict[str, pl.DataFrame], as_of: datetime | None = None) -> pl.DataFrame:
    rows = []
    for sym, ent in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
        c = _complete(panel[sym])
        rv = c["rv"].to_numpy()
        days = c["day_ts"]
        origins = days + timedelta(days=1)
        cont = _contiguous(c["d"].to_numpy(), 7)
        dvol = _asof_value(origins, alt_series("deribit", "dvol_close", ent, as_of), timedelta(hours=3))
        svi = _asof_value(origins, alt_series("deribit_svi", "iv_atm_7", ent, as_of), timedelta(hours=48))
        ivs = {"dvol": np.log(7 * iv_to_daily_var(dvol, True)), "svi7": np.log(7 * iv_to_daily_var(svi, False))}
        f0 = har_forecast_x(rv, horizon=7, min_obs=180, target_mask=cont)
        fx = {k: har_forecast_x(rv, v, horizon=7, min_obs=180, target_mask=cont) for k, v in ivs.items()}
        real = np.r_[np.convolve(rv, np.ones(7), "valid")[1:], [np.nan] * 7][: len(rv)]
        for d in range(len(rv)):
            o = origins[d]
            if o.weekday() != 0 or not cont[d] or not np.isfinite(real[d]):
                continue
            row = {"symbol": sym, "origin": o, "end": o + timedelta(days=7), "q_har": float(qlike(real[d], f0[d])) if np.isfinite(f0[d]) else np.nan}
            for k in fx:
                row[f"q_{k}"] = float(qlike(real[d], fx[k][d])) if np.isfinite(fx[k][d]) else np.nan
            rows.append(row)
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ H-VCL5 (compression)

def vcl5_panel(panel: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for sym, dly in panel.items():
        c = _complete(dly)
        rv, ret = c["rv"].to_numpy(), c["ret"].to_numpy()
        n = len(rv)
        if n < 450:
            continue
        cont = _contiguous(c["d"].to_numpy(), 5)
        f5 = har_forecast_x(rv, horizon=5, target_mask=cont)
        rv20 = pl.Series(rv).rolling_sum(20).to_numpy()
        pct = np.full(n, np.nan)
        for d in range(365, n):
            w = rv20[d - 365:d]
            w = w[np.isfinite(w)]
            if len(w) >= 300 and np.isfinite(rv20[d]):
                pct[d] = float((w <= rv20[d]).mean())
        days = c["day_ts"].to_list()
        for d in range(n - 5):
            if not cont[d] or not np.isfinite(f5[d]) or not np.isfinite(pct[d]):
                continue
            o = days[d] + timedelta(days=1)
            real = float(rv[d + 1:d + 6].sum())
            rows.append({"symbol": sym, "origin": o, "end": o + timedelta(days=5), "pct": pct[d], "event": bool(pct[d] <= 0.10),
                         "ratio": float(real / f5[d]), "ret5": float(ret[d + 1:d + 6].sum())})
    return pl.DataFrame(rows)


def _cluster_boot_median(df: pl.DataFrame, col: str, reps: int = 2000, seed: int = 7) -> tuple[float, float]:
    if df.height == 0:
        return (float("nan"), float("nan"))
    wk = df.with_columns(pl.col("origin").dt.truncate("1w").alias("wk"))
    groups = [g[col].to_numpy() for _, g in wk.group_by("wk")]
    rng = np.random.default_rng(seed)
    meds = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        meds[r] = np.median(np.concatenate([groups[i] for i in pick]))
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


# ------------------------------------------------------------------ H-FU1 / H-FU2 (weekly return predictability)

def _closes(panel: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Close at (d+1) 00:00 UTC (= close of the 23:00 bar of day d), long format."""
    parts = []
    for sym, dly in panel.items():
        p = dly.filter(pl.col("last_bar").dt.hour() == 23).select(
            pl.lit(sym).alias("symbol"), (pl.col("last_bar") + timedelta(hours=1)).alias("t"), pl.col("close"))
        parts.append(p)
    return pl.concat(parts)


def _fwd30(closes: pl.DataFrame) -> pl.DataFrame:
    fut = closes.select("symbol", (pl.col("t") - timedelta(days=30)).alias("t"), pl.col("close").alias("close30"))
    return closes.join(fut, on=["symbol", "t"], how="inner").with_columns((pl.col("close30") / pl.col("close")).log().alias("fwd30"))


def fu1_panel(panel: dict[str, pl.DataFrame], as_of: datetime | None = None) -> pl.DataFrame:
    s = alt_series("defillama", "stablecoins_total_usd", "all", as_of)
    # known_ts = event day + 1 -> value usable at that known date 00:00 (<= decision time)
    s = s.with_columns(pl.col("known_ts").dt.truncate("1d").alias("t")).group_by("t").agg(pl.col("value").last()).sort("t")
    s = s.upsample("t", every="1d").with_columns(pl.col("value").forward_fill())
    lv = np.log(s["value"].to_numpy())
    x = np.full(len(lv), np.nan)
    x[30:] = lv[30:] - lv[:-30]
    xs = pl.Series(x)
    z = (xs - xs.rolling_mean(365, min_samples=300)) / xs.rolling_std(365, min_samples=300)
    s = s.with_columns(pl.Series("chg30", x), z.alias("z"))
    r = _fwd30(_closes(panel))
    btc = r.filter(pl.col("symbol") == "BTCUSDT").select("t", pl.col("fwd30").alias("btc"))
    ew = r.group_by("t").agg(pl.col("fwd30").mean().alias("ew"), pl.len().alias("n_sym"))
    out = (s.filter(pl.col("t").dt.weekday() == 1).join(btc, on="t", how="left").join(ew, on="t", how="left")
           .rename({"t": "origin"}).with_columns((pl.col("origin") + timedelta(days=30)).alias("end"))
           .filter(pl.col("z").is_not_nan() & pl.col("btc").is_not_null()))
    return out


def fu2_panel(panel: dict[str, pl.DataFrame], as_of: datetime | None = None) -> pl.DataFrame:
    c = _complete(panel["BTCUSDT"])
    rv = c["rv"].to_numpy()
    cont = _contiguous(c["d"].to_numpy(), 30)
    f30 = har_forecast_x(rv, horizon=30, target_mask=cont)
    origins = c["day_ts"] + timedelta(days=1)
    dvol = _asof_value(origins, alt_series("deribit", "dvol_close", "BTC", as_of), timedelta(hours=3))
    df = pl.DataFrame({"origin": origins, "iv_var": (dvol / 100) ** 2, "har_var": f30 * 365 / 30}).with_columns(
        (pl.col("iv_var") - pl.col("har_var")).alias("vrp"))
    r = _fwd30(_closes({"BTCUSDT": panel["BTCUSDT"]})).select(pl.col("t").alias("origin"), pl.col("fwd30").alias("btc"))
    return (df.join(r, on="origin", how="inner").filter(pl.col("origin").dt.weekday() == 1)
            .with_columns((pl.col("origin") + timedelta(days=30)).alias("end"))
            .filter(pl.col("vrp").is_not_nan() & pl.col("vrp").is_not_null()))


# ------------------------------------------------------------------ H-C1 (funding carry)

def carry_book(symbols: list[str], as_of: datetime | None = None, conditional: bool = True) -> pl.DataFrame:
    """Weekly delta-neutral carry: short perp / long spot, spot assumed to hedge price exactly.
    Returns per week: origin, end, n_active, gross (funding per unit notional), cost, net, turnover."""
    as_of = as_of or now()
    f = Lake().read_asof("funding", as_of, symbols=symbols).select("symbol", "event_ts", "rate").sort("symbol", "event_ts")
    rs = load().settings["research"]
    first = f.group_by("symbol").agg(pl.col("event_ts").min().alias("first"))
    first = first.with_columns(pl.when(pl.col("first").dt.year() >= rs["burn_in_new_listing_year"])
                               .then(pl.col("first") + timedelta(days=rs["burn_in_days_new"]))
                               .otherwise(pl.col("first") + timedelta(days=rs["burn_in_days"])).alias("live"))
    f = f.join(first, on="symbol").filter(pl.col("event_ts") >= pl.col("live")).drop("first", "live")
    t0 = f["event_ts"].min()
    t0 = _ts((t0 + timedelta(days=7 - t0.weekday() + 7)).date())  # second Monday after the first settlement
    last = f["event_ts"].max()
    weeks = []
    t = t0
    while t + timedelta(days=7) <= last:
        weeks.append(t)
        t += timedelta(days=7)
    # bucket each settlement into the week whose (t, t+7d] contains it
    f = f.with_columns(((pl.col("event_ts") - timedelta(microseconds=1)).dt.truncate("1w")).alias("hold_wk"))
    hold = f.group_by("symbol", "hold_wk").agg(pl.col("rate").sum().alias("fund"), pl.len().alias("n_set"))
    # signal at t: settlements in (t-7d, t] = the hold bucket of week t-7d
    sig = hold.select("symbol", (pl.col("hold_wk") + timedelta(days=7)).alias("wk"), (pl.col("fund") / pl.col("n_set")).alias("sig"))
    hold = hold.rename({"hold_wk": "wk"})
    grid = pl.DataFrame({"wk": weeks}).with_columns(pl.col("wk").cast(pl.Datetime("us", "UTC")))
    g = grid.join(sig, on="wk", how="left").join(hold, on=["symbol", "wk"], how="left").filter(pl.col("symbol").is_not_null())
    g = g.with_columns((pl.col("sig") > 0 if conditional else pl.lit(True)).alias("on"))
    n_on = g.group_by("wk").agg(pl.col("on").sum().alias("n_active"))
    g = g.join(n_on, on="wk").with_columns(pl.when(pl.col("on")).then(1.0 / pl.col("n_active")).otherwise(0.0).alias("w"))
    wide = g.pivot(on="symbol", index="wk", values="w").sort("wk")
    wide = grid.join(wide, on="wk", how="left").fill_null(0.0).sort("wk")
    W = wide.drop("wk").to_numpy()
    turnover = np.abs(np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0)).sum(axis=1)
    gross = g.group_by("wk").agg((pl.col("w") * pl.col("fund").fill_null(0.0)).sum().alias("gross"))
    out = (grid.join(gross, on="wk", how="left").join(n_on, on="wk", how="left").fill_null(0).sort("wk")
           .with_columns(pl.Series("turnover", turnover), pl.Series("cost", 2 * COST_PER_LEG * turnover))
           .with_columns((pl.col("gross") - pl.col("cost")).alias("net"))
           .rename({"wk": "origin"}).with_columns((pl.col("origin") + timedelta(days=7)).alias("end")))
    return out


def carry_stats(w: pl.DataFrame) -> dict:
    out = {"n_weeks": w.height}
    if w.height == 0:
        return out
    for col in ("gross", "net"):
        x = w[col].to_numpy()
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        cum = np.cumsum(x)
        dd = float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min())
        m, t, _ = newey_west_t(x)
        out[col] = {"ann_return": float(x.mean() * 52), "sharpe": float(x.mean() / sd * np.sqrt(52)) if sd > 0 else float("nan"),
                    "nw_t": t, "max_dd": dd, "worst_week": float(x.min()), "frac_neg_weeks": float((x < 0).mean())}
    out["mean_n_active"] = float(w["n_active"].mean())
    out["ann_turnover"] = float(w["turnover"].mean() * 52)
    out["ann_cost"] = float(w["cost"].mean() * 52)
    return out


def carry_by_year(w: pl.DataFrame) -> list[dict]:
    rows = []
    for (y,), g in w.with_columns(pl.col("origin").dt.year().alias("y")).group_by("y"):
        s = carry_stats(g)
        rows.append({"year": y, "weeks": g.height, "gross_ann": s["gross"]["ann_return"], "net_ann": s["net"]["ann_return"],
                     "net_sharpe": s["net"]["sharpe"], "worst_week": s["net"]["worst_week"], "frac_neg": s["gross"]["frac_neg_weeks"],
                     "n_active": s["mean_n_active"]})
    return sorted(rows, key=lambda r: r["year"])


# ------------------------------------------------------------------ verdicts

def summarise(seg: str, P: dict[str, pl.DataFrame], books: dict[str, pl.DataFrame]) -> dict[str, tuple[str, dict]]:
    res: dict[str, tuple[str, dict]] = {}
    for hid in ("H-VCL2", "H-VCL3"):
        d = split(P[hid])[seg]
        m = _paired(d, "q_har", "q_x")
        ok = m["dqlike"] < 0 and m["t"] < -2
        res[hid] = (_judge(seg, ok, m["dqlike"] < 0, m["n_origins"]), m)

    d = split(P["H-VCL4"])[seg]
    m4 = {}
    for k in ("dvol", "svi7"):
        dk = d.filter(pl.col(f"q_{k}").is_not_nan() & pl.col("q_har").is_not_nan())
        m4[k] = _paired(dk, "q_har", f"q_{k}")
    ok = m4["dvol"]["dqlike"] < 0 and m4["dvol"]["t"] < -2
    res["H-VCL4"] = (_judge(seg, ok, m4["dvol"]["dqlike"] < 0, m4["dvol"]["n_origins"]), {"primary": "dvol", **m4})

    d = split(P["H-VCL5"])[seg]
    ev_ = d.filter(pl.col("event"))
    med = float(ev_["ratio"].median()) if ev_.height else float("nan")
    lo, hi = _cluster_boot_median(ev_, "ratio")
    rr = ev_.group_by("origin").agg(pl.col("ret5").mean()).sort("origin")["ret5"].to_numpy()
    mr, tr, nr = newey_west_t(rr)
    m5 = {"median_ratio_events": med, "ci95": [lo, hi], "median_ratio_all_days": float(d["ratio"].median()) if d.height else float("nan"),
          "mean_log_ratio_events": float(np.log(ev_["ratio"]).mean()) if ev_.height else float("nan"),
          "mean_log_ratio_all": float(np.log(d["ratio"]).mean()) if d.height else float("nan"),
          "n_events": ev_.height, "n_event_dates": nr, "mean_ret5": mr, "t_ret5": tr}
    ok = med > 1.1 and lo > 1 and np.isfinite(tr) and abs(tr) < 2
    res["H-VCL5"] = (_judge(seg, ok, med > 1, nr), m5)

    for hid, ycols in (("H-FU1", ("btc", "ew")), ("H-FU2", ("btc",))):
        d = split(P[hid])[seg]
        xcol = "z" if hid == "H-FU1" else "vrp"
        m = {y: ols_hac(d[y].cast(pl.Float64).fill_null(np.nan).to_numpy(), d[xcol].to_numpy()) for y in ycols}
        ok = m["btc"]["slope"] > 0 and m["btc"]["t"] > 2
        if hid == "H-FU1":
            ok = ok and m["ew"]["slope"] > 0
        res[hid] = (_judge(seg, ok, m["btc"]["slope"] > 0, m["btc"]["n"]), m)

    w = split(books["conditional"])[seg]
    s = carry_stats(w)
    s["by_year"] = carry_by_year(w)
    s["context_always_on"] = {k: v for k, v in carry_stats(split(books["always_on"])[seg]).items() if k in ("gross", "net")}
    if seg == "clean_holdout":
        verdict = "inconclusive" if w.height < MIN_HOLDOUT else ("supports" if s["net"]["ann_return"] > 0 else "contradicts")
    else:
        verdict = "supports" if s["net"]["sharpe"] > 1 else ("contradicts" if s["net"]["ann_return"] <= 0 else "inconclusive")
    res["H-C1"] = (verdict, s)
    return res


DATA = {
    "H-VCL2": "bybit 1h->daily RV (24-bar UTC days), 23 symbols; HAR vs HAR+log RS-_d; expanding OLS refit 30d, min 180d",
    "H-VCL3": "22 alts; HAR vs HAR + BTC log RV_d and 7d mean log RV (joined by date)",
    "H-VCL4": "BTC/ETH; direct HAR-7d vs +log(7*DVOL^2/365) [primary] / +SVI iv_atm_7 [secondary]; Monday origins, non-overlapping",
    "H-VCL5": "23 symbols; RV20 <= 10th pct of trailing 365d; realised 5d / direct HAR-5d; week-cluster bootstrap",
    "H-FU1": "DefiLlama total stablecoins (revisable, backfilled), z of 30d log change vs trailing 365d; Monday samples; HAC lag>=5",
    "H-FU2": "BTC: DVOL^2 - direct HAR-30d*365/30; Monday samples from 2021-03; HAC lag>=5",
    "H-C1": "Bybit realised funding, 23 symbols; weekly Monday rebalance, EW among trailing-7d-mean>0; 10 bps/leg/turnover; spot hedge assumed perfect",
}


def build(as_of: datetime | None = None) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    panel = daily_panel(UNIVERSE, as_of)
    P = {"H-VCL2": vcl2_panel(panel), "H-VCL3": vcl3_panel(panel), "H-VCL4": vcl4_panel(panel, as_of),
         "H-VCL5": vcl5_panel(panel), "H-FU1": fu1_panel(panel, as_of), "H-FU2": fu2_panel(panel, as_of)}
    books = {"conditional": carry_book(UNIVERSE, as_of), "always_on": carry_book(UNIVERSE, as_of, conditional=False)}
    return P, books


def main(seg_arg: str) -> None:
    """insample | holdout record that segment; dry = in-sample only, printed, not recorded (debugging)."""
    from ai_trader.research.hypotheses import record

    seg = {"insample": "llm_in_sample", "holdout": "clean_holdout", "dry": "llm_in_sample"}[seg_arg]
    P, books = build()
    res = summarise(seg, P, books)
    for hid, (verdict, metrics) in res.items():
        if seg_arg != "dry":
            record(hid, seg, verdict, metrics, data=DATA[hid])
        print(hid, seg, verdict)
        print(json.dumps(metrics, default=float, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "insample")
