"""WMS study (program WMM2): per symbol-weekend Friday-known indices, weekend outcomes and weekend BoS events.

    uv run python -m ai_trader.research.wms_study features      # -> data/research/wms/features.parquet, bos_events.parquet, btc.parquet
    uv run python -m ai_trader.research.wms_study analyze       # -> data/research/wms/results.json, selection.parquet (see wms_tests)

Grid per weekend W (Saturday weekend_start): 5m bars opened in [Fri-7d 22:00, Mon 00:00) = 2016 (FW) + 24 (gap) + 576 (weekend).
Missing bars are forward-filled (high = low = close) after the coverage check. Log prices of the last-trade close.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.research import wms_select as ws
from ai_trader.research.wms_data import band_table, root, saturdays

N = ws.FW_BARS + ws.GAP_BARS + ws.WE_BARS  # 2616
WE0 = ws.FW_BARS + ws.GAP_BARS  # 2040: first weekend bar
PREV_WE = (24, 24 + 576)  # previous weekend inside the FW
TUE_THU = (888, 1464)  # Tue 00:00 -> Thu 00:00 of the same week
MON = 600  # Mon 00:00 of the same week (weekday window Mon 00:00 -> Sat 00:00 = [600, 2040))
BAND = (5e6, 50e6)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def grid(df_ts: np.ndarray, cols: dict[str, np.ndarray], t0: int) -> tuple[dict[str, np.ndarray], np.ndarray, float | None]:
    """Values on the N-bar grid starting at t0 (ms), the presence mask and the close of the bar before t0 (or None)."""
    want = t0 + ws.BAR_MS * np.arange(-1, N, dtype=np.int64)
    idx = np.searchsorted(df_ts, want)
    idx_c = np.minimum(idx, df_ts.size - 1)
    present = (idx < df_ts.size) & (df_ts[idx_c] == want)
    out = {}
    for k, v in cols.items():
        x = np.where(present, v[idx_c], np.nan)
        out[k] = x
    pre = float(out["close"][0]) if present[0] else None
    pres = present[1:]
    c = out["close"][1:]
    # forward fill closes; leading gaps take the first present value
    good = np.where(~np.isnan(c), np.arange(c.size), -1)
    np.maximum.accumulate(good, out=good)
    if (good < 0).all():
        return {}, pres, pre
    first = int(np.argmax(good >= 0))
    good[:first] = first
    cf = c[good]
    hf = np.where(pres, out["high"][1:], cf)
    lf = np.where(pres, out["low"][1:], cf)
    tf = np.where(pres, out["turnover"][1:], 0.0)
    return {"close": cf, "high": hf, "low": lf, "turnover": tf}, pres, pre


def seg_stats(lc: np.ndarray, a: int, b: int, p_before: float) -> tuple[float, float, float]:
    """Sum r, sum r^2, sum |r| of the log returns of bars [a, b) (first return from p_before)."""
    r = np.diff(np.concatenate([[p_before], lc[a:b]]))
    return float(r.sum()), float(r @ r), float(np.abs(r).sum())


def event_rows(tf: str, h: np.ndarray, lo: np.ndarray, c: np.ndarray, btc_c: np.ndarray, we0: int, bars_per: int,
               sig5: float, hor: dict[str, int]) -> list[dict]:
    rows = []
    last = c.size - 1
    for t, d, lvl in ws.bos_events(h, lo, c, 3):
        if t < we0:
            continue
        row = {"tf": tf, "bar": t - we0, "dir": d, "excess_bp": 1e4 * d * (c[t] - lvl)}
        for name, k in hor.items():
            j = last if k < 0 else t + k
            if j > last or j <= t:
                row[f"c_{name}"] = row[f"ch_{name}"] = row[f"bp_{name}"] = None
                continue
            sd = sig5 * math.sqrt((j - t) * bars_per)
            ra = c[j] - c[t]
            rb = btc_c[j] - btc_c[t]
            row[f"c_{name}"] = d * ra / sd
            row[f"ch_{name}"] = d * (ra - rb) / sd
            row[f"bp_{name}"] = 1e4 * d * ra
        rows.append(row)
    return rows


def symbol_weekends(sym: str, weekends: list[date], btc: dict) -> tuple[list[dict], list[dict]]:
    f = root() / "bars5m" / f"{sym}.parquet"
    if not f.exists():
        return [], []
    df = pl.read_parquet(f)
    ts = df["ts"].to_numpy()
    cols = {k: df[k].to_numpy().astype(np.float64) for k in ("high", "low", "close", "turnover")}
    feats, events = [], []
    for wk in weekends:
        t0 = ms(ws.window_start(wk))
        g, pres, pre = grid(ts, cols, t0)
        if not g:
            continue
        b = btc.get(wk)
        cov_fw = float(pres[: ws.FW_BARS].mean())
        cov_we = float(pres[WE0:].mean())
        zero_to = float((g["turnover"][: ws.FW_BARS] <= 0).mean())
        row = {"weekend_start": wk, "symbol": sym, "cov_fw": cov_fw, "cov_we": cov_we, "zero_to_fw": zero_to}
        if cov_fw < 0.98 or cov_we < 0.98 or b is None:
            feats.append(row)
            continue
        lc, lh, ll = np.log(g["close"]), np.log(g["high"]), np.log(g["low"])
        p0 = math.log(pre) if pre else float(lc[0])
        fw = ws.indices(lh[: ws.FW_BARS], ll[: ws.FW_BARS], lc[: ws.FW_BARS], p0)
        row.update({k: v for k, v in fw.items()})
        h15, l15, c15 = ws.agg(lh[:WE0], ll[:WE0], lc[:WE0], 3)
        we = ws.indices(lh[WE0:], ll[WE0:], lc[WE0:], float(lc[WE0 - 1]), dfa_scales=ws.DFA_SCALES_WE, bos_lead=(h15, l15, c15))
        row.update({f"w_{k}": v for k, v in we.items()})
        s_fw, rv_fw, _ = seg_stats(lc, 0, ws.FW_BARS, p0)
        _, rv_d1, _ = seg_stats(lc, ws.FW_BARS - 288, ws.FW_BARS, float(lc[ws.FW_BARS - 289]))
        _, rv_pw, _ = seg_stats(lc, PREV_WE[0], PREV_WE[1], float(lc[PREV_WE[0] - 1]))
        s_we, rv_we, a_we = seg_stats(lc, WE0, N, float(lc[WE0 - 1]))
        _, rv_tt, _ = seg_stats(lc, TUE_THU[0], TUE_THU[1], float(lc[TUE_THU[0] - 1]))
        exc = np.abs(lc[WE0:] - lc[WE0 - 1])
        bc = b["lc"]
        row.update({
            "rv_fw": rv_fw, "rv_d1": rv_d1, "rv_pw": rv_pw, "rv_we": rv_we, "rv_tt": rv_tt,
            "ret_we": s_we, "abs_we": a_we, "mx_we": float(exc.max()),
            "trend_we": abs(s_we) / math.sqrt(rv_we) if rv_we > 0 else None,
            "rtr_we": math.log(a_we / max(abs(s_we), 1e-8)) if a_we > 0 else None,
            "s_we": s_we - (bc[N - 1] - bc[WE0 - 1]),
            "s_tt": (lc[TUE_THU[1] - 1] - lc[TUE_THU[0] - 1]) - (bc[TUE_THU[1] - 1] - bc[TUE_THU[0] - 1]),
            "ret_fri24": float(lc[ws.FW_BARS - 1] - lc[ws.FW_BARS - 289]),
            "ret_fw": float(lc[ws.FW_BARS - 1] - p0),
            "s_fw": float(lc[ws.FW_BARS - 1] - p0) - (bc[ws.FW_BARS - 1] - b["p0"]),
            "turnover_fw": float(g["turnover"][: ws.FW_BARS].sum()),
        })
        feats.append(row)
        sig5 = math.sqrt(rv_fw / ws.FW_BARS)
        bh15, bl15, bc15 = ws.agg(lh, ll, lc, 3)
        btc15 = bc[2::3]
        for e in event_rows("15m", bh15, bl15, bc15, btc15, WE0 // 3, 3, sig5, {"1h": 4, "4h": 16, "close": -1}):
            events.append({"weekend_start": wk, "symbol": sym, **e})
        bh1, bl1, bc1 = ws.agg(lh, ll, lc, 12)
        btc1 = bc[11::12]
        for e in event_rows("1h", bh1, bl1, bc1, btc1, WE0 // 12, 12, sig5, {"1h": 1, "4h": 4, "close": -1}):
            events.append({"weekend_start": wk, "symbol": sym, **e})
    return feats, events


def btc_table(weekends: list[date]) -> tuple[dict, pl.DataFrame]:
    df = pl.read_parquet(root() / "bars5m" / "BTCUSDT.parquet")
    ts = df["ts"].to_numpy()
    cols = {k: df[k].to_numpy().astype(np.float64) for k in ("high", "low", "close", "turnover")}
    out, rows = {}, []
    for wk in weekends:
        g, pres, pre = grid(ts, cols, ms(ws.window_start(wk)))
        if not g or pres[: ws.FW_BARS].mean() < 0.98:
            continue
        lc = np.log(g["close"])
        p0 = math.log(pre) if pre else float(lc[0])
        have_we = pres[WE0:].mean() >= 0.98
        out[wk] = {"lc": lc, "p0": p0}
        s_fw, rv_fw, _ = seg_stats(lc, 0, ws.FW_BARS, p0)
        _, rv_d1, _ = seg_stats(lc, ws.FW_BARS - 288, ws.FW_BARS, float(lc[ws.FW_BARS - 289]))
        _, rv_pw, _ = seg_stats(lc, PREV_WE[0], PREV_WE[1], float(lc[PREV_WE[0] - 1]))
        row = {"weekend_start": wk, "rv_fw": rv_fw, "rv_d1": rv_d1, "rv_pw": rv_pw, "ret_fw": s_fw,
               "ret_fri24": float(lc[ws.FW_BARS - 1] - lc[ws.FW_BARS - 289]), "have_we": bool(have_we)}
        if have_we:
            s_we, rv_we, a_we = seg_stats(lc, WE0, N, float(lc[WE0 - 1]))
            r = np.diff(lc[WE0 - 1:])
            sig = math.sqrt(rv_fw / ws.FW_BARS)
            row.update({"ret_we": s_we, "rv_we": rv_we, "n_jump5": int((np.abs(r) > 5 * sig).sum()),
                        "max_abs_z": float(np.abs(r).max() / sig), "ret_wd": float(lc[WE0 - 1] - lc[MON - 1]),
                        "trend_we": abs(s_we) / math.sqrt(rv_we), "mx_we": float(np.abs(lc[WE0:] - lc[WE0 - 1]).max())})
        rows.append(row)
    return out, pl.DataFrame(rows)


_BTC: dict = {}


def _init(btc: dict) -> None:
    global _BTC
    _BTC = btc


def _work(args: tuple[str, list[date]]) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    sym, wks = args
    f, e = symbol_weekends(sym, wks, _BTC)
    return (pl.DataFrame(f, infer_schema_length=None) if f else None, pl.DataFrame(e, infer_schema_length=None) if e else None)


def features(procs: int = 2) -> None:
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    d = pl.read_parquet(root() / "daily.parquet")
    band = band_table(d, *BAND)
    weekends = [s.date() for s in saturdays()]
    btc, bt = btc_table(weekends)
    bt.write_parquet(root() / "btc.parquet")
    syms = sorted(band["symbol"].unique().to_list())
    jobs = [(sym, sorted(band.filter(pl.col("symbol") == sym)["weekend_start"].to_list())) for sym in syms]
    feats, events = [], []
    nf = ne = 0
    with ProcessPoolExecutor(procs, initializer=_init, initargs=(btc,)) as ex:
        for i, (f, e) in enumerate(ex.map(_work, jobs, chunksize=4)):
            if f is not None:
                feats.append(f)
                nf += f.height
            if e is not None:
                events.append(e)
                ne += e.height
            if i % 50 == 0:
                print(f"{i}/{len(syms)} rows={nf} events={ne} {time.time() - t0:.0f}s", flush=True)
    F = pl.concat(feats, how="diagonal_relaxed").join(band, on=["weekend_start", "symbol"], how="left")
    F.write_parquet(root() / "features.parquet")
    E = pl.concat(events, how="diagonal_relaxed")
    E.write_parquet(root() / "bos_events.parquet")
    events = E
    print(f"features: {F.height} rows, events {len(events)}, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- H-WMS-13: bounce bias on flow_3s (last trade vs mid)

def bounce_check() -> dict:
    lake = load_root() / "data" / "lake" / "flow_3s" / "venue=bybit" / "category=linear"
    gaps = pl.read_parquet(load_root() / "data" / "research" / "ws02" / "gaps.parquet")
    deficit = gaps.filter(pl.col("kind") == "trade_deficit").with_columns(
        pl.from_epoch("gap_start", time_unit="ms").dt.date().alias("day")).select("symbol", "day").unique()
    d = pl.read_parquet(root() / "daily.parquet")
    rows = []
    for part in sorted(lake.glob("date=*")):
        day = date.fromisoformat(part.name[5:])
        band = d.filter((pl.col("day") == day) & (pl.col("turnover") >= BAND[0]) & (pl.col("turnover") <= BAND[1]))["symbol"].to_list()
        if not band:
            continue
        f = pl.scan_parquet(str(part / "*.parquet")).filter(pl.col("symbol").is_in(band)).select("symbol", "event_ts", "close", "mid", "spread_bps")
        f = f.collect().unique(["symbol", "event_ts"]).sort("symbol", "event_ts")
        bad = set(deficit.filter(pl.col("day") == day)["symbol"].to_list())
        for (sym,), g in f.group_by(["symbol"]):
            if sym in bad or g.height < 0.95 * 28800:
                continue
            g = g.with_columns(pl.col("event_ts").cast(pl.Int64).alias("t"))
            for mins in (1, 5):
                step = mins * 60_000
                # value at the end of each bucket = the last 3 s candle in it
                b = g.with_columns((pl.col("t") // step).alias("k")).group_by("k").agg(pl.col("close").last(), pl.col("mid").last(),
                                                                                         pl.col("spread_bps").mean().alias("sp")).sort("k")
                b = b.drop_nulls(["close", "mid"])
                k = b["k"].to_numpy()
                ok = np.diff(k) == 1
                for col in ("close", "mid"):
                    r = np.diff(np.log(b[col].to_numpy()))
                    r = np.where(ok, r, np.nan)
                    x, y = r[:-1], r[1:]
                    m = np.isfinite(x) & np.isfinite(y)
                    rho = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 50 else math.nan
                    rows.append({"day": day, "symbol": sym, "mins": mins, "price": col, "ar1": rho, "n": int(m.sum()),
                                 "spread_bps": float(b["sp"].mean())})
    R = pl.DataFrame(rows)
    R.write_parquet(root() / "bounce.parquet")
    out = {}
    for mins in (1, 5):
        w = R.filter(pl.col("mins") == mins).pivot(on="price", index=["day", "symbol"], values="ar1").drop_nulls()
        w = w.with_columns((pl.col("close") - pl.col("mid")).alias("diff"))
        syms = w["symbol"].unique().to_list()
        med = float(w["diff"].median())
        arr = {s_: w.filter(pl.col("symbol") == s_)["diff"].to_numpy() for s_ in syms}
        boots = []
        for _ in range(2000):
            pick = RNG.choice(len(syms), len(syms))
            boots.append(float(np.median(np.concatenate([arr[syms[i]] for i in pick]))))
        out[f"{mins}m"] = {"median_diff": med, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           "median_ar1_last": float(w["close"].median()), "median_ar1_mid": float(w["mid"].median()),
                           "symbol_days": w.height, "symbols": len(syms)}
    one, five = out["1m"], out["5m"]
    if one["median_diff"] < -0.02 and one["ci"][1] < 0 and abs(five["median_diff"]) < abs(one["median_diff"]):
        v = "supports"
    elif one["ci"][0] > -0.02:
        v = "contradicts"
    else:
        v = "inconclusive"
    out["verdict"] = v
    return out


RNG = np.random.default_rng(20261127)


def load_root():
    from ai_trader.core.config import load
    return load().root


# ---------------------------------------------------------------- forward weekend (Friday-known only)

def forward(weekend: date = date(2026, 11, 28)) -> None:
    """Fetch the Friday window of the forward weekend for its in-band names (+ BTC), funding and 1d OI; nothing after T_F."""
    from ai_trader.research import wms_data as wd
    tf = ws.friday_cutoff(weekend)
    d = pl.read_parquet(root() / "daily.parquet")
    fri = tf.date()
    win = d.filter((pl.col("day") >= fri - timedelta(days=7)) & (pl.col("day") <= fri - timedelta(days=1)))
    first = d.group_by("symbol").agg(pl.col("day").min().alias("first_day"))
    g = win.group_by("symbol").agg(pl.col("turnover").median().alias("med_turnover"), pl.len().alias("n_days")).join(first, on="symbol")
    g = g.filter((pl.col("n_days") == 7) & (pl.col("first_day") <= fri - timedelta(days=30)) & pl.col("med_turnover").is_between(*BAND))
    syms = sorted(g["symbol"].to_list()) + ["BTCUSDT"]
    a, z = ms(ws.window_start(weekend)) - 3_600_000, ms(tf)
    frames, fund, oi = [], [], []
    for s in syms:
        rows = wd.klines(s, "5", a, z, ws.BAR_MS)
        df = wd._frame(s, rows, ws.BAR_MS, z)
        if df.height:
            frames.append(df.select("symbol", "ts", "high", "low", "close", "turnover"))
        res = wd._get("funding/history", {"category": "linear", "symbol": s, "startTime": z - 9 * 86_400_000, "endTime": z, "limit": 200})
        fund += [(s, int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in res.get("list", [])]
        res = wd._get("open-interest", {"category": "linear", "symbol": s, "intervalTime": "1d", "startTime": z - 10 * 86_400_000,
                                        "endTime": z, "limit": 200})
        oi += [(s, int(x["timestamp"]), float(x["openInterest"])) for x in res.get("list", [])]
    out = root() / "forward"
    out.mkdir(exist_ok=True)
    tag = weekend.isoformat()
    pl.concat(frames).filter(pl.col("ts") + ws.BAR_MS <= z).write_parquet(out / f"bars5m_{tag}.parquet")
    pl.DataFrame(fund, schema=["symbol", "ts", "rate"], orient="row").filter(pl.col("ts") <= z).write_parquet(out / f"funding_{tag}.parquet")
    pl.DataFrame(oi, schema=["symbol", "ts", "oi"], orient="row").write_parquet(out / f"oi_{tag}.parquet")
    g.write_parquet(out / f"band_{tag}.parquet")
    print(f"forward {tag}: {len(syms) - 1} names in band, bars {sum(f.height for f in frames)}")


def forward_features(weekend: date = date(2026, 11, 28)) -> tuple[pl.DataFrame, dict]:
    out = root() / "forward"
    tag = weekend.isoformat()
    bars = pl.read_parquet(out / f"bars5m_{tag}.parquet")
    band = pl.read_parquet(out / f"band_{tag}.parquet")
    t0 = ms(ws.window_start(weekend))
    rows, btc = [], {}
    for (sym,), df in bars.group_by(["symbol"]):
        df = df.sort("ts")
        ts = df["ts"].to_numpy()
        cols = {k: df[k].to_numpy().astype(np.float64) for k in ("high", "low", "close", "turnover")}
        g, pres, pre = grid(ts, cols, t0)
        if not g:
            continue
        cov = float(pres[: ws.FW_BARS].mean())
        lc, lh, ll = (np.log(g[k][: ws.FW_BARS]) for k in ("close", "high", "low"))
        p0 = math.log(pre) if pre else float(lc[0])
        s_fw, rv_fw, _ = seg_stats(lc, 0, ws.FW_BARS, p0)
        _, rv_d1, _ = seg_stats(lc, ws.FW_BARS - 288, ws.FW_BARS, float(lc[ws.FW_BARS - 289]))
        _, rv_pw, _ = seg_stats(lc, PREV_WE[0], PREV_WE[1], float(lc[PREV_WE[0] - 1]))
        r = {"weekend_start": weekend, "symbol": sym, "cov_fw": cov, "zero_to_fw": float((g["turnover"][: ws.FW_BARS] <= 0).mean()),
             "rv_fw": rv_fw, "rv_d1": rv_d1, "rv_pw": rv_pw, "ret_fw": s_fw, "ret_fri24": float(lc[-1] - lc[-289]),
             "ret_wd": float(lc[-1] - lc[MON - 1]), "p0": p0, "lc_end": float(lc[-1])}
        if sym == "BTCUSDT":
            btc = r
            continue
        r |= ws.indices(lh, ll, lc, p0)
        rows.append(r)
    F = pl.DataFrame(rows).join(band.select("symbol", "med_turnover"), on="symbol", how="left")
    if btc:
        F = F.with_columns((pl.col("ret_fw") - btc["ret_fw"]).alias("s_fw"))
    return F, btc


# ---------------------------------------------------------------- null distributions of the indices (dossier)

def nulls(n_sim: int = 1000) -> dict:
    """Sampling distribution of every index on a 2016-bar window under (a) iid Student-t(3.5) 5m returns (random walk) and
    (b) the same efficient price observed with bid-ask bounce (Roll: last = efficient +- half-spread, sign iid), spread 8 bp,
    sigma_5m 25 bp (a typical 5-50 M USD alt), and (c) a mean-reverting OU log price (half-life 2 h) for scale."""
    rng = np.random.default_rng(7)
    sig, half = 25e-4, 4e-4
    out = {}
    for case in ("rw", "rw_bounce", "ou_2h"):
        vals: dict[str, list] = {}
        for _ in range(n_sim):
            e = rng.standard_t(3.5, ws.FW_BARS + 1) * sig / math.sqrt(3.5 / 1.5)
            if case == "ou_2h":
                b = 1 - 0.5 ** (1 / 24)
                x = np.zeros(e.size)
                for t in range(1, e.size):
                    x[t] = (1 - b) * x[t - 1] + e[t]
                p = x + np.cumsum(e) * 0.3
            else:
                p = np.cumsum(e)
            if case == "rw_bounce":
                p = p + half * rng.choice([-1.0, 1.0], p.size)
            h, lo = p + half, p - half
            ind = ws.indices(h[1:], lo[1:], p[1:], p[0])
            for k, v in ind.items():
                vals.setdefault(k, []).append(v)
        out[case] = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "p05": float(np.nanpercentile(v, 5)),
                         "p95": float(np.nanpercentile(v, 95))} for k, v in vals.items()}
    return out


# ---------------------------------------------------------------- analysis driver

def analyze(fwd_weekend: date = date(2026, 11, 28)) -> dict:
    import json

    from ai_trader.research import wms_analysis as A
    t0 = time.time()
    U = A.load_universe()
    U, har = A.loghar(U)
    U = A.add_scores(A.outcomes(U))
    U = A.deriv_inputs(U)
    res: dict = {"loghar": har}
    seg_counts = U.group_by("segment").agg(pl.len().alias("rows"), pl.col("weekend_start").n_unique().alias("weekends"),
                                           pl.col("symbol").n_unique().alias("symbols")).to_dicts()
    F = pl.read_parquet(root() / "features.parquet")
    res["universe"] = {"segments": seg_counts, "band_rows": F.height,
                       "dropped_cov": int(F.filter((pl.col("cov_fw") < 0.98) | (pl.col("cov_we") < 0.98)).height),
                       "dropped_zero_turnover": int(F.filter((pl.col("cov_fw") >= 0.98) & (pl.col("cov_we") >= 0.98) & (pl.col("zero_to_fw") > 0.10)).height),
                       "per_weekend_median": float(U.group_by("weekend_start").len()["len"].median()),
                       "delisted_symbols": int(U.join(pl.read_parquet(root() / "daily.parquet").filter(pl.col("status") == "Closed")
                                                      .select("symbol").unique(), on="symbol").select(pl.col("symbol").n_unique()).item()),
                       "symbols": int(U["symbol"].n_unique())}
    res |= A.h_persistence(U)
    print("persistence done", time.time() - t0, flush=True)
    res |= A.h_outcomes(U)
    res["H-WMS-5"] = A.h_rule(U, "S0")
    print("outcomes done", time.time() - t0, flush=True)
    res["H-WMS-6"] = A.h_bos(U)
    B = pl.read_parquet(root() / "btc.parquet")
    res |= A.h_drift(U, B)
    print("drift done", time.time() - t0, flush=True)
    try:
        res["properties"] = A.properties(U, B)
    except Exception as e:  # diagnostics only
        res["properties"] = {"error": repr(e)}
    res["H-WMS-11"], U2 = A.h_trenders(U)
    print("trenders done", time.time() - t0, flush=True)
    # forward BTC row for the calm classifier
    FF, btc_f = forward_features(fwd_weekend) if (root() / "forward" / f"bars5m_{fwd_weekend}.parquet").exists() else (None, {})
    extra = None
    if btc_f:
        extra = {"weekend_start": fwd_weekend, "rv_fw": btc_f["rv_fw"], "rv_d1": btc_f["rv_d1"], "rv_pw": btc_f["rv_pw"],
                 "ret_fw": btc_f["ret_fw"], "ret_fri24": btc_f["ret_fri24"], "have_we": False}
    C = A.calm_frame(B, extra)
    C.write_parquet(root() / "calm_frame.parquet")
    res["H-WMS-12"], cm, nrm, ptab = A.h_calm(C, B)
    print("calm done", time.time() - t0, flush=True)
    # ---- the frozen rule
    s0_pass = res["H-WMS-3"]["outcomes"]["O4_trend"]["llm_in_sample"]["mean"] is not None and         res["H-WMS-3"]["outcomes"]["O4_trend"]["llm_in_sample"]["mean"] < 0 and (res["H-WMS-3"]["outcomes"]["O4_trend"]["q_fit"] or 1) < 0.05
    s1 = res["H-WMS-4"].get("fit_passing", [])
    s1_fit = res["H-WMS-4"].get("S1", {}).get("llm_in_sample", {})
    use = "S0" if s0_pass or not s1 or not ((s1_fit.get("mean") or 0) < 0 and (s1_fit.get("p") or 1) < 0.05) else "S1"
    gate = bool(res["H-WMS-12"]["fit_pass"])
    res["rule"] = {"score": use, "gate_enabled": gate, "s0_fit_pass": bool(s0_pass), "s1_components": s1}
    if use == "S1":
        comps = {k: "neg" for k in s1}
        U = ws.composite(U, components=comps, name="S1")
        res["H-WMS-5_S1"] = A.h_rule(U, "S1")
    sel = U.join(ptab, on="weekend_start", how="left")
    sel = ws.select(sel.with_columns(pl.col(use).alias("score")), calm_p="calm_p", gate=gate)
    keep = ["weekend_start", "symbol", "segment", "med_turnover", *A.IDX, "S0", "score", "eligible_name", "calm_p", "eligible"]
    sel = sel.with_columns([pl.col(c).exp().alias(c) for c in A.LOGT])  # back to raw scale
    out = sel.select(keep).with_columns(pl.lit(gate).alias("gate_on"), pl.lit(use).alias("score_kind"))
    # ---- forward weekend rows
    fwd = {}
    if FF is not None and FF.height:
        f = FF.filter((pl.col("cov_fw") >= 0.98) & (pl.col("zero_to_fw") <= 0.10)).drop_nulls(list(ws.S0_COMPONENTS))
        f = ws.composite(f, name="S0")
        if use == "S1":
            f = ws.composite(f.with_columns([pl.col(c).log().alias(f"_l_{c}") for c in s1 if c in A.LOGT]),
                             components={(f"_l_{c}" if c in A.LOGT else c): "neg" for c in s1}, name="S1")
        row = {c: (C.filter(pl.col("weekend_start") == fwd_weekend)[c][0] if C.filter(pl.col("weekend_start") == fwd_weekend).height else None)
               for c in A.CALM_X + ["dvol", "med_rv52", "holiday"]}
        p_calm = A.predict_calm(cm, nrm, row)
        f = f.with_columns(pl.lit("forward").alias("segment"), pl.lit(p_calm, pl.Float64).alias("calm_p"), pl.col(use).alias("score"))
        f = ws.select(f, calm_p="calm_p", gate=gate)
        out = pl.concat([out, f.select(keep).with_columns(pl.lit(gate).alias("gate_on"), pl.lit(use).alias("score_kind"))], how="diagonal_relaxed")
        fwd = {"weekend_start": str(fwd_weekend), "calm_p": p_calm, "inputs": row, "btc": btc_f, "n_universe": f.height,
               "n_band": FF.height, "eligible_name": f.filter(pl.col("eligible_name")).sort("score", descending=True)
               .select("symbol", "score", "med_turnover", "vr15_4", "vr15_16", "ar1_15", "dfa", "er_1h", "bos_imb").to_dicts(),
               "eligible": f.filter(pl.col("eligible"))["symbol"].to_list()}
        # where does this Friday sit in history?
        hist = C.drop_nulls("x1_rvfw_vs_med")
        for c in ("x1_rvfw_vs_med", "x2_day_vs_week", "x3_dvol_vs_med", "dvol"):
            if row.get(c) is not None:
                fwd[f"pct_{c}"] = float((hist.filter(pl.col("weekend_start") < fwd_weekend)[c].drop_nulls() < row[c]).mean())
        fwd["ret_fri24_btc_pct"] = float((B["ret_fri24"] < btc_f["ret_fri24"]).mean()) if btc_f else None
    res["forward"] = fwd
    out.write_parquet(root() / "selection.parquet")
    U.drop([c for c in U.columns if c.startswith("w_")]).write_parquet(root() / "panel.parquet")
    res["runtime_s"] = time.time() - t0
    (root() / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"analysis done {time.time() - t0:.0f}s; rule {res['rule']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "bounce", "forward", "analyze", "nulls"])
    a = ap.parse_args()
    if a.step == "features":
        features()
    elif a.step == "forward":
        forward()
    elif a.step == "analyze":
        analyze()
    elif a.step == "nulls":
        import json
        res = nulls()
        (root() / "nulls.json").write_text(json.dumps(res, indent=1))
        for case, d in res.items():
            print(case, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in d.items()})
    elif a.step == "bounce":
        import json
        res = bounce_check()
        (root() / "bounce.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
