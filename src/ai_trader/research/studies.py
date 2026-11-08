"""Study runners: turn the lake into evidence for the hypothesis registry.

playbook()       strategy × regime-cell performance + forward-path priors + shock study, pooled across
                 symbols as a cell-conditional equal-weight portfolio (averaging across symbols per
                 timestamp clusters by time, so crypto-wide co-movement doesn't inflate t-stats)
weekend_study()  per-weekend table with post-weekend outcomes for H-W1..H-W5

Every output is split into llm_in_sample / clean_holdout at research.knowledge_cutoff.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.research import evaluate as ev
from ai_trader.research import evaluate as ev_mod
from ai_trader.research.calendar import NY, add_calendar, gap_fill_baseline, weekend_table
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import label
from ai_trader.strategies import all_strategies

BAR_HOURS = {"1h": 1.0, "4h": 4.0, "1d": 24.0, "15m": 0.25}


def _segments(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {"pooled": df, **ev.split_by_cutoff(df)}


def _cell_portfolio_stats(panel: pl.DataFrame, kappa: float) -> pl.DataFrame:
    """panel: symbol, event_ts, cell, pnl, active. Stats per cell on the per-timestamp mean across symbols."""
    act = panel.filter(pl.col("active"))
    pooled_ts = act.group_by("event_ts").agg(pl.col("pnl").mean()).sort("event_ts")
    pm, pt, pn = ev.newey_west_t(pooled_ts["pnl"].to_numpy())
    eps = {}
    for sym in panel["symbol"].unique().to_list():
        for c, k in ev.episodes(panel.filter(pl.col("symbol") == sym).sort("event_ts")["cell"]).items():
            eps[c] = eps.get(c, 0) + k
    rows = []
    for c in sorted(x for x in act["cell"].unique().to_list() if x is not None):
        ts = act.filter(pl.col("cell") == c).group_by("event_ts").agg(pl.col("pnl").mean()).sort("event_ts")
        m, t, n = ev.newey_west_t(ts["pnl"].to_numpy())
        w = eps.get(c, 0) / (eps.get(c, 0) + kappa)
        rows.append({"cell": c, "episodes": eps.get(c, 0), "n_ts": n, "mean": m, "t_nw": t,
                     "hit": float((ts["pnl"] > 0).mean()), "shrink_w": w, "mean_shrunk": w * m + (1 - w) * pm})
    rows.append({"cell": "ALL", "episodes": sum(eps.values()), "n_ts": pn, "mean": pm, "t_nw": pt,
                 "hit": float((pooled_ts["pnl"] > 0).mean()) if pn else float("nan"), "shrink_w": 1.0, "mean_shrunk": pm})
    return pl.DataFrame(rows)


def playbook(symbols: list[str], freq: str = "1h", as_of: datetime | None = None, start: datetime | None = None) -> dict[str, pl.DataFrame]:
    as_of = as_of or now()
    rs = load().settings["research"]
    bh = BAR_HOURS[freq]
    horizons = sorted({max(1, round(h / bh)) for h in rs["path_horizons_hours"]})
    strategies = all_strategies()
    pnl_rows, path_frames, shock_frames = [], [], []
    for sym in symbols:
        f = market_frame(sym, freq, as_of, start=start, flow=True)
        if f.height < 500:
            continue
        f = label(f, bh)
        cells = f["cell"]
        for s in strategies:
            p = ev.strategy_pnl(f, s(f))
            pnl_rows.append(p.select("event_ts", "pnl", "active").with_columns(pl.lit(sym).alias("symbol"), cells.alias("cell"), pl.lit(s.spec.name).alias("strategy")))
        fp = ev.forward_paths(f, horizons).with_columns(cells.alias("cell"), pl.lit(sym).alias("symbol"))
        path_frames.append(fp)
        evs = ev.shock_events(f, rs["shock_z"], round(rs["shock_decluster_hours"] / bh))
        if evs.height:
            shock_frames.append(fp[evs["i"].to_list()].with_columns(evs["shock_sign"].alias("sign")))
    panel = pl.concat(pnl_rows)
    paths = pl.concat(path_frames)
    shocks = pl.concat(shock_frames) if shock_frames else pl.DataFrame()

    out: dict[str, pl.DataFrame] = {}
    perf = []
    for seg, d in _segments(panel).items():
        for name in d["strategy"].unique().sort().to_list():
            st = _cell_portfolio_stats(d.filter(pl.col("strategy") == name), rs["shrinkage_kappa"])
            perf.append(st.with_columns(pl.lit(name).alias("strategy"), pl.lit(seg).alias("segment")))
    out["strategy_by_cell"] = pl.concat(perf)

    prow = []
    for seg, d in _segments(paths).items():
        for c in sorted(x for x in d["cell"].unique().to_list() if x is not None):
            sub = d.filter(pl.col("cell") == c)
            for h in horizons:
                z = sub[f"fwd_z_{h}"].drop_nulls().drop_nans().to_numpy()
                if len(z) < 50:
                    continue
                q = np.quantile(z, [0.1, 0.25, 0.5, 0.75, 0.9])
                prow.append({"segment": seg, "cell": c, "h_bars": h, "n": len(z), "mean_z": float(z.mean()), "p_up": float((z > 0).mean()),
                             "q10": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q90": q[4],
                             "mfe_med": float(np.nanmedian(sub[f"mfe_{h}"].to_numpy())), "mae_med": float(np.nanmedian(sub[f"mae_{h}"].to_numpy()))})
    out["path_priors"] = pl.DataFrame(prow)

    srow = []
    if shocks.height:
        for seg, d in _segments(shocks).items():
            for c in [*sorted(x for x in d["cell"].unique().to_list() if x is not None), "ALL"]:
                g = d if c == "ALL" else d.filter(pl.col("cell") == c)
                for h in horizons:
                    x = (g[f"fwd_z_{h}"] * g["sign"]).drop_nulls().drop_nans().to_numpy()
                    if len(x) < 5:
                        continue
                    m, t, n = ev.newey_west_t(x) if len(x) >= 10 else (float(x.mean()), float("nan"), len(x))
                    srow.append({"segment": seg, "cell": c, "h_bars": h, "n_events": n, "mean_signed_z": m, "t": t, "p_continue": float((x > 0).mean())})
    out["shock_study"] = pl.DataFrame(srow)
    return out


# ---------------------------------------------------------------- weekend

def weekend_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    """weekend_table + what happened next: Sunday reopen -> Monday NYSE close (the absorption window)."""
    base = add_calendar(frame.sort("event_ts"))
    wt = weekend_table(base)  # its first_i/last_i index rows of `base`
    # Drop warm-up weekends (σ̂ or the turnover baseline not yet defined).
    wt = wt.filter(pl.col("sigma_fri").is_finite() & (pl.col("sigma_fri") > 0) & pl.col("vol_ratio").is_finite() & pl.col("wk_z").is_finite())
    f = base.with_row_index("_i")
    if wt.height == 0:
        return wt
    ny = pl.col("event_ts").dt.convert_time_zone(NY)
    # Monday NYSE close bar = the bar opening at 15:00 ET on Monday (1h bars).
    mon_close = f.filter((ny.dt.weekday() == 1) & (ny.dt.hour() == 15)).select(pl.col("_i").alias("mon_i"))
    # NYSE opening window: the 09:00, 10:00 and 11:00 ET bars (the 09:00 bar contains the 09:30 open).
    open_abs = (
        f.filter((ny.dt.weekday() <= 5) & ny.dt.hour().is_between(9, 11))
        .group_by(ny.dt.date().alias("ny_date"))
        .agg(pl.col("z").abs().fill_nan(None).mean().alias("open_abs_z"), (ny.dt.weekday() == 1).first().alias("is_mon"))
    )
    open_by_date = dict(zip(open_abs["ny_date"].to_list(), open_abs["open_abs_z"].to_list(), strict=True))
    other_open = open_abs.filter(~pl.col("is_mon"))["open_abs_z"].mean()
    # H-W5b: |ret| in the window scaled by the σ̂ known before the session (08:00 ET bar's sigma_next).
    pre = f.filter((ny.dt.weekday() <= 5) & (ny.dt.hour() == 8)).select(ny.dt.date().alias("ny_date"), pl.col("sigma_next").alias("sig_pre"))
    win_abs = f.filter((ny.dt.weekday() <= 5) & ny.dt.hour().is_between(9, 11)).group_by(ny.dt.date().alias("ny_date")).agg(
        pl.col("ret").abs().fill_nan(None).mean().alias("abs_ret"), (ny.dt.weekday() == 1).first().alias("is_mon"))
    win_abs = win_abs.join(pre, on="ny_date", how="left")
    absret_by_date = dict(zip(win_abs["ny_date"].to_list(), win_abs["abs_ret"].to_list(), strict=True))
    other_rs = win_abs.filter(~pl.col("is_mon")).select((pl.col("abs_ret") / pl.col("sig_pre")).fill_nan(None).mean()).item()
    lows, highs, closes = f["low"].to_numpy(), f["high"].to_numpy(), f["close"].to_numpy()
    rows = []
    mon_idx = np.sort(mon_close["mon_i"].to_numpy())
    for r in wt.iter_rows(named=True):
        j0 = r["last_i"] + 1  # first bar after the weekend
        k = mon_idx[mon_idx >= j0]
        if len(k) == 0 or k[0] - j0 > 30:
            continue
        j1 = int(k[0])
        n_post = j1 - r["last_i"]
        post_ret = np.log(closes[j1] / r["sun_close"])
        d = np.log(r["sun_close"] / r["fri_close"]) / r["sigma_fri"]
        filled = bool(lows[j0 : j1 + 1].min() <= r["fri_close"] <= highs[j0 : j1 + 1].max()) if r["sun_close"] != r["fri_close"] else True
        mon_date = (datetime.fromisoformat(r["weekend_id"]) + timedelta(days=3)).date()
        rows.append({**r, "post_bars": n_post,
                     "post_z": post_ret / (r["sigma_fri"] * np.sqrt(n_post)),
                     "signed_post_z": np.sign(r["wk_z"]) * post_ret / (r["sigma_fri"] * np.sqrt(n_post)),
                     "gap_sigma": d, "gap_filled": filled,
                     "fill_baseline": float(gap_fill_baseline(np.array([d]), np.array([n_post]))[0]),
                     "mon_open_abs_z": open_by_date.get(mon_date),
                     "mon_open_abs_rs": (absret_by_date.get(mon_date) or np.nan) / r["sigma_fri"]})
    out = pl.DataFrame(rows)
    return out.with_columns(pl.lit(other_open).alias("other_days_open_abs_z"), pl.lit(other_rs).alias("other_days_open_abs_rs"))


def weekend_study(symbols: list[str], as_of: datetime | None = None, start: datetime | None = None) -> pl.DataFrame:
    as_of = as_of or now()
    frames = []
    for sym in symbols:
        f = market_frame(sym, "1h", as_of, start=start, funding=False, oi=False)
        if f.height < 24 * 30:
            continue
        w = weekend_outcomes(f)
        if w.height:
            frames.append(w.with_columns(pl.lit(sym).alias("symbol"), pl.col("end_ts").alias("event_ts")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


# ---------------------------------------------------------------- weekend variance risk premium (H-W7)

def weekend_vrp(entity: str, symbol: str, as_of: datetime | None = None) -> pl.DataFrame:
    """Per week: log(realised / implied variance) for the TradFi weekend window (Fri 17:00 -> Sun 18:00 ET,
    IV from Friday 08:00 UTC) and for an equal-length weekday window (Tue 17:00 -> Thu 18:00 ET, IV from
    Tuesday 08:00 UTC). Implied variance = IV_7d_ATM^2 * hours / 8760 (options price calendar time)."""
    from ai_trader.storage.lake import Lake

    as_of = as_of or now()
    lake = Lake()
    b = lake.bars(symbol, "1h", as_of).select("event_ts", "close").sort("event_ts")
    b = b.with_columns(pl.col("close").log().diff().alias("r"))
    ny = pl.col("event_ts").dt.convert_time_zone(NY)
    wd, mins = ny.dt.weekday(), ny.dt.hour().cast(pl.Int32) * 60 + ny.dt.minute().cast(pl.Int32)
    b = b.with_columns(
        (((wd == 5) & (mins >= 1020)) | (wd == 6) | ((wd == 7) & (mins < 1080))).alias("wk"),
        (((wd == 2) & (mins >= 1020)) | (wd == 3) | ((wd == 4) & (mins < 1080))).alias("wd"),
        # anchor date = the Friday (weekend) or Tuesday (weekday) that opens the window
        (ny.dt.date() - pl.duration(days=(wd - 5) % 7)).alias("fri"),
        (ny.dt.date() - pl.duration(days=(wd - 2) % 7)).alias("tue"),
    )
    iv = lake.read_asof("alt_series", as_of, filters={"source": "deribit_svi", "series_id": "iv_atm_7", "entity": entity})
    iv = iv.select(pl.col("event_ts").dt.date().alias("d"), pl.col("value").alias("iv"))

    def window(flag: str, anchor: str) -> pl.DataFrame:
        w = b.filter(pl.col(flag)).group_by(anchor).agg((pl.col("r") ** 2).sum().alias("rv"), pl.len().alias("h"), pl.col("event_ts").min().alias("event_ts"))
        w = w.filter(pl.col("h") >= 47).join(iv, left_on=anchor, right_on="d", how="inner")
        return w.with_columns((pl.col("rv") / (pl.col("iv") ** 2 * pl.col("h") / 8760)).log().alias("log_ratio"),
                              pl.col(anchor).dt.truncate("1w").alias("week"))

    we, wk = window("wk", "fri"), window("wd", "tue")
    return we.select("week", "event_ts", pl.col("log_ratio").alias("lr_weekend"), pl.col("iv").alias("iv_fri")).join(
        wk.select("week", pl.col("log_ratio").alias("lr_weekday"), pl.col("iv").alias("iv_tue")), on="week", how="inner"
    ).with_columns((pl.col("lr_weekend") - pl.col("lr_weekday")).alias("diff")).sort("week")


# ---------------------------------------------------------------- post-jump volatility (H-S4)

def post_jump_vol(symbols: list[str], freq: str = "1h", z_thresh: float = 5.0, horizon: int = 24, as_of: datetime | None = None) -> pl.DataFrame:
    """Per jump event (|z| >= z_thresh, declustered over `horizon` bars): squared returns over the next
    `horizon` bars in units of the PRE-jump variance (sigma_prev^2), plus the ratio of realised post-jump
    vol to the EWMA forecast made right after the jump (sigma_next)."""
    as_of = as_of or now()
    rows = []
    for sym in symbols:
        f = market_frame(sym, freq, as_of, funding=False, oi=False)
        if f.height < 500:
            continue
        r = f["ret"].to_numpy()
        sp, sn = f["sigma_prev"].to_numpy(), f["sigma_next"].to_numpy()
        ev = ev_mod.shock_events(f, z_thresh, horizon)
        for i in ev["i"].to_list():
            if i + horizon >= len(r) or not (np.isfinite(sp[i]) and sp[i] > 0):
                continue
            post = r[i + 1 : i + 1 + horizon]
            rows.append({"symbol": sym, "event_ts": f["event_ts"][i], "z": float(r[i] / sp[i]),
                         **{f"x{h + 1}": float(post[h] ** 2 / sp[i] ** 2) for h in range(horizon)},
                         "rv_over_forecast": float(np.sqrt(np.mean(post**2)) / sn[i])})
    return pl.DataFrame(rows)


def jump_half_life(events: pl.DataFrame, horizon: int = 24) -> dict:
    """Mean excess variance e(h) = E[r_{t+h}^2 / sigma_pre^2] - 1, and the first h where e(h) <= e(1)/2."""
    e = np.array([events[f"x{h}"].mean() - 1 for h in range(1, horizon + 1)])
    half = next((h + 1 for h in range(horizon) if e[h] <= e[0] / 2), None)
    return {"excess_h1": float(e[0]), "excess_h6": float(e[5]), "excess_h24": float(e[-1]), "half_life_bars": half,
            "rv_over_forecast_median": float(events["rv_over_forecast"].median()), "n_events": events.height}


# ---------------------------------------------------------------- realised vol vs trend returns (H-X3)

def rv_trend_ic(symbols: list[str], strategy: str, freq: str = "4h", rv_days: int = 20, fwd_days: int = 10, pct_window_days: int = 365, as_of: datetime | None = None) -> pl.DataFrame:
    """Daily panel: RV20 time-series percentile (trailing 1y, causal) vs the strategy's next-10-day net P&L.
    Signals are computed on the full history; P&L is aggregated per UTC day."""
    from ai_trader.strategies import REGISTRY

    as_of = as_of or now()
    out = []
    for sym in symbols:
        f = market_frame(sym, freq, as_of, funding=False, oi=False)
        if f.height < 500:
            continue
        p = ev_mod.strategy_pnl(f, REGISTRY[strategy]()(f))
        daily = (
            f.select("event_ts", "ret").with_columns(p["pnl"])
            .group_by(pl.col("event_ts").dt.date().alias("d")).agg(pl.col("ret").sum().alias("r_d"), pl.col("pnl").sum())
            .sort("d")
        )
        daily = daily.with_columns(pl.col("r_d").rolling_std(rv_days, min_samples=rv_days).alias("rv"))
        daily = daily.with_columns(
            pl.col("rv").rolling_map(lambda s: float((s < s[-1]).mean()) if s.null_count() == 0 else None, window_size=pct_window_days, min_samples=pct_window_days // 2).alias("rv_pct"),
            # next fwd_days of P&L, strictly after day d
            pl.col("pnl").rolling_sum(fwd_days).shift(-fwd_days).alias("fwd_pnl"),
        )
        out.append(daily.select("d", "rv_pct", "fwd_pnl").with_columns(pl.lit(sym).alias("symbol")))
    return pl.concat(out).drop_nulls()


# ---------------------------------------------------------------- vol model validation (H-V1, H-V2)

def vol_forecast_eval(symbols: list[str], h: int = 24, as_of: datetime | None = None) -> pl.DataFrame:
    """Non-overlapping h-bar windows (every h-th bar): QLIKE of the cumulative-variance forecast for
    plain EWMA, robust EWMA, robust + weekend-seasonal EWMA. Also flags weekend overlap and jumps."""
    from ai_trader.features.vol_models import VolModel, horizon_variance, run

    as_of = as_of or now()
    models = {"plain": VolModel(clip=np.inf, use_weekend=False), "robust": VolModel(use_weekend=False), "robust_wk": VolModel(),
              "robust_bc": VolModel(use_weekend=False, bias_correct=True)}
    rows = []
    for sym in symbols:
        f = market_frame(sym, "1h", as_of, funding=False, oi=False)
        if f.height < 2000:
            continue
        wk = add_calendar(f)["tradfi_weekend"].to_numpy()
        r = np.nan_to_num(f["ret"].to_numpy())
        r2c = np.concatenate([[0.0], np.cumsum(r**2)])
        n = len(r)
        idx = np.arange(0, n - h - 1, h)
        realised = r2c[idx + h + 1] - r2c[idx + 1]
        wk_c = np.concatenate([[0], np.cumsum(wk)])
        n_wk = wk_c[idx + h + 1] - wk_c[idx + 1]
        z = f["z"].to_numpy()
        out = {"symbol": sym, "event_ts": f["event_ts"].to_numpy()[idx], "realised": realised, "n_wk": n_wk, "jump": np.abs(np.nan_to_num(z[idx])) >= 5}
        for name, cfg in models.items():
            o = run(r, wk, cfg)
            F = horizon_variance(o["v_next"], o["m"], wk, h, use_weekend=cfg.use_weekend)[idx]
            q = realised / F - np.log(realised / F) - 1
            out[f"F_{name}"] = F
            out[f"qlike_{name}"] = q
        rows.append(pl.DataFrame(out))
    df = pl.concat(rows)
    return df.with_columns(pl.col("event_ts").dt.replace_time_zone("UTC") if df["event_ts"].dtype.time_zone is None else pl.col("event_ts"))


# ---------------------------------------------------------------- liquidity crunches (H-LC2) and HAR (H-VCL1)

def illiquidity_events(symbols: list[str], z_thresh: float = 2.0, h: int = 24, window: int = 720, as_of: datetime | None = None) -> pl.DataFrame:
    """Amihud ILLIQ = |ret| / turnover; log z vs the PRIOR `window` bars. Events z >= z_thresh (declustered
    over h). Ratio = sqrt(realised var over next h bars / (h * sigma_risk_next^2)). Also a baseline sample
    (every h-th bar) for comparison."""
    as_of = as_of or now()
    rows = []
    for sym in symbols:
        f = market_frame(sym, "1h", as_of, funding=False, oi=False)
        if f.height < window + 100:
            continue
        x = (pl.col("ret").abs() / pl.col("turnover")).log()
        f = f.with_columns(pl.when(pl.col("turnover") > 0).then(x).otherwise(None).alias("lil"))
        f = f.with_columns(((pl.col("lil") - pl.col("lil").shift(1).rolling_mean(window, min_samples=window // 2))
                            / pl.col("lil").shift(1).rolling_std(window, min_samples=window // 2)).alias("illiq_z"))
        r2 = np.nan_to_num(f["ret"].to_numpy()) ** 2
        c = np.concatenate([[0.0], np.cumsum(r2)])
        sig = f["sigma_risk_next"].to_numpy()
        zz = np.nan_to_num(f["illiq_z"].to_numpy(), nan=-9)
        n = len(r2)
        last = -10**9
        ts = f["event_ts"].to_list()
        for i in range(n - h - 1):
            is_event = zz[i] >= z_thresh and i - last > h
            if is_event:
                last = i
            if is_event or i % h == 0:
                if not (np.isfinite(sig[i]) and sig[i] > 0):
                    continue
                rv = c[i + h + 1] - c[i + 1]
                rows.append({"symbol": sym, "event_ts": ts[i], "event": bool(is_event), "illiq_z": float(zz[i]), "ratio": float(np.sqrt(rv / (h * sig[i] ** 2)))})
    return pl.DataFrame(rows)


def har_vs_robust(symbols: list[str], refit_days: int = 30, min_days: int = 180, as_of: datetime | None = None) -> pl.DataFrame:
    """Per symbol and UTC day d: forecast of next-day realised variance by HAR on log RV (daily/weekly/monthly,
    expanding-window OLS refit every `refit_days`, lognormal mean correction) vs robust_bc (24 * sigma_risk_next^2
    at the last bar of day d). Returns per-day QLIKE of both."""
    as_of = as_of or now()
    out = []
    for sym in symbols:
        f = market_frame(sym, "1h", as_of, funding=False, oi=False)
        if f.height < 24 * (min_days + 60):
            continue
        daily = (f.with_columns(pl.col("event_ts").dt.date().alias("d"))
                 .group_by("d").agg((pl.col("ret").fill_null(0) ** 2).sum().alias("rv"), pl.col("sigma_risk_next").last().alias("sig"), pl.len().alias("n"))
                 .filter(pl.col("n") == 24).sort("d"))
        lrv = np.log(np.maximum(daily["rv"].to_numpy(), 1e-12))
        rv = daily["rv"].to_numpy()
        sig = daily["sig"].to_numpy()
        n = len(rv)
        X = np.column_stack([np.ones(n), lrv,
                             pl.Series(lrv).rolling_mean(7).to_numpy(), pl.Series(lrv).rolling_mean(30).to_numpy()])
        coef, s2, last_fit = None, None, -10**9
        for d in range(30, n - 1):
            if d >= min_days and (coef is None or d - last_fit >= refit_days):
                # fit on pairs (X_j, lrv_{j+1}) with j+1 <= d: only information available at day d
                Xj, yj = X[30:d], lrv[31 : d + 1]
                ok = np.isfinite(Xj).all(axis=1) & np.isfinite(yj)
                coef, *_ = np.linalg.lstsq(Xj[ok], yj[ok], rcond=None)
                s2 = float(np.var(yj[ok] - Xj[ok] @ coef))
                last_fit = d
            if coef is None or not np.isfinite(X[d]).all() or not np.isfinite(sig[d]):
                continue
            f_har = float(np.exp(X[d] @ coef + 0.5 * s2))
            f_rob = 24 * sig[d] ** 2
            real = rv[d + 1]
            if real <= 0:
                continue
            q = lambda F: real / F - np.log(real / F) - 1  # noqa: E731
            out.append({"symbol": sym, "d": daily["d"][d], "q_har": q(f_har), "q_rob": q(f_rob)})
    return pl.DataFrame(out)
