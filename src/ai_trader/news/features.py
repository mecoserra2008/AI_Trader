"""Point-in-time daily and hourly news / event / on-chain feature tables.

Daily table: one row per decision time `asof` = t + 1 00:00 UTC (the end of day t), used to forecast day t + 1.
Every component carries its own known_ts column (kt_*) and the builder guarantees kt_* <= asof:
  news (GDELT)  bucket for day d has known_ts = d + 1 00:00 (end of the publication window). Primary features use the
                latest day with known_ts <= asof (day t); the *_l1 variants use day t - 1 (robust to GDELT's ~15 min
                processing lag).
  on-chain      day d known at d + 2 00:00 (onchain.ONCHAIN_LAG), so at asof = t + 1 the latest usable day is t - 1.
  events        "next" flags: an event scheduled on day t + 1 whose schedule was public (schedule_known_ts <= asof);
                "today" flags: an event that happened on day t (scheduled_ts <= asof).

GDELT monitored volume (the API's `norm`) is not stationary: it fell from ~0.7 M articles/day (2017) to ~0.15 M (2023-
2026) and to ~0.02 M by November 2026. Raw counts are therefore never used; features are the log share
ls_d = ln((count_d + 1) / norm_d) minus its trailing 60-day mean (abnormal attention, log units), and tone minus its
trailing 60-day mean. `gdelt_norm_z` (log norm vs its trailing 60-day mean) records coverage shifts.

Feature block for H-NEWS1 / H-NEWS2 (fixed before the evaluation, diary 2026-11-13):
  att_btc      abnormal log share, 'bitcoin'
  att_btc7     mean of att_btc over days t-6..t
  att_ind      mean abnormal log share over the industry queries (hack, regulation, etf, stablecoin, binance, coinbase, tether)
  att_max      max abnormal log share over all queries
  tone_btc     abnormal tone, 'bitcoin'
  tone_ind     mean abnormal tone over the industry queries
  ev_macro_next, ev_crypto_next, ev_macro_today   0/1 event flags (macro = CPI, NFP, scheduled FOMC; crypto = halving, ETF)
  oc_ntx, oc_fees, oc_mempool   ln(value_d) - mean ln(value) over d-30..d-1 (latest usable day)
  oc_hash      ln hash_d - ln hash_{d-7}
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.news import gdelt, onchain, rss
from ai_trader.news.store import Manifest

INDUSTRY = ("hack", "regulation", "etf", "stablecoin", "binance", "coinbase", "tether")
NEWS_COLS = ("att_btc", "att_btc7", "att_ind", "att_max", "tone_btc", "tone_ind")
EVENT_COLS = ("ev_macro_next", "ev_crypto_next", "ev_macro_today")
ONCHAIN_COLS = ("oc_ntx", "oc_fees", "oc_mempool", "oc_hash")
BLOCK = NEWS_COLS + EVENT_COLS + ONCHAIN_COLS
MACRO = ("cpi", "nfp", "fomc")
CRYPTO = ("halving", "etf_reg")
TRAIL = 60


def _trail_mean(x: np.ndarray, w: int, min_n: int) -> np.ndarray:
    """mean of x[t-w..t-1] (strictly before t)."""
    s = pl.Series(np.asarray(x, dtype=float)).fill_nan(None).shift(1)
    return s.rolling_mean(w, min_samples=min_n).to_numpy().astype(float)


def gdelt_daily(man: Manifest, queries=None) -> pl.DataFrame:
    """date, known_ts and per query: ls_<q>, abn_<q>, tone_<q>, tabn_<q>; plus gdelt_norm_z (from the 'bitcoin' norm)."""
    out = None
    for q in queries or gdelt.QUERIES:
        v = gdelt.load_chunks(man, "timelinevolraw", q, daily=True)
        if v.height == 0:
            continue
        v = v.with_columns(pl.col("ts").dt.date().alias("date")).sort("date")
        ls = np.log((v["value"].to_numpy() + 1.0) / np.maximum(v["norm"].to_numpy(), 1.0))
        cols = {"date": v["date"], f"ls_{q}": ls, f"abn_{q}": ls - _trail_mean(ls, TRAIL, 30)}
        if q == "bitcoin":
            ln = np.log(np.maximum(v["norm"].to_numpy(), 1.0))
            cols["gdelt_norm_z"] = ln - _trail_mean(ln, TRAIL, 30)
        f = pl.DataFrame(cols)
        t = gdelt.load_chunks(man, "timelinetone", q, daily=True)
        if t.height:
            t = t.with_columns(pl.col("ts").dt.date().alias("date")).sort("date")
            tv = t["value"].to_numpy()
            f = f.join(pl.DataFrame({"date": t["date"], f"tone_{q}": tv, f"tabn_{q}": tv - _trail_mean(tv, TRAIL, 30)}), on="date", how="left")
        out = f if out is None else out.join(f, on="date", how="full", coalesce=True)
    if out is None:
        return pl.DataFrame()
    out = out.with_columns(pl.col(pl.Float64).fill_nan(None))
    return out.sort("date").with_columns((pl.col("date").cast(pl.Datetime("us", "UTC")) + timedelta(days=1)).alias("known_ts"))


def news_block(g: pl.DataFrame) -> pl.DataFrame:
    """Compact news block from gdelt_daily output (one row per news day)."""
    if g.height == 0:
        return g
    ind_abn = [f"abn_{q}" for q in INDUSTRY if f"abn_{q}" in g.columns]
    ind_tone = [f"tabn_{q}" for q in INDUSTRY if f"tabn_{q}" in g.columns]
    all_abn = [c for c in g.columns if c.startswith("abn_")]
    ex = [pl.col("abn_bitcoin").alias("att_btc") if "abn_bitcoin" in g.columns else pl.lit(None, pl.Float64).alias("att_btc"),
          pl.col("abn_bitcoin").rolling_mean(7, min_samples=5).alias("att_btc7") if "abn_bitcoin" in g.columns else pl.lit(None, pl.Float64).alias("att_btc7"),
          (pl.mean_horizontal(ind_abn) if ind_abn else pl.lit(None, pl.Float64)).alias("att_ind"),
          (pl.max_horizontal(all_abn) if all_abn else pl.lit(None, pl.Float64)).alias("att_max"),
          (pl.col("tabn_bitcoin") if "tabn_bitcoin" in g.columns else pl.lit(None, pl.Float64)).alias("tone_btc"),
          (pl.mean_horizontal(ind_tone) if ind_tone else pl.lit(None, pl.Float64)).alias("tone_ind")]
    keep = ["date", "known_ts"] + (["gdelt_norm_z"] if "gdelt_norm_z" in g.columns else [])
    return g.sort("date").select(*keep, *ex)


def onchain_block(man: Manifest) -> pl.DataFrame:
    d = onchain.daily(man)
    if d.height == 0:
        return d
    cols = {"date": d["date"], "known_ts": d["known_ts"]}
    for src, name in (("n_transactions", "oc_ntx"), ("transaction_fees_usd", "oc_fees"), ("mempool_size", "oc_mempool")):
        if src in d.columns:
            x = np.log(np.maximum(d[src].to_numpy().astype(float), 1e-9))
            cols[name] = x - _trail_mean(x, 30, 20)
    if "hash_rate" in d.columns:
        h = np.log(np.maximum(d["hash_rate"].to_numpy().astype(float), 1e-9))
        cols["oc_hash"] = h - np.r_[np.full(7, np.nan), h[:-7]]
    return pl.DataFrame(cols).with_columns(pl.col(pl.Float64).fill_nan(None)).sort("known_ts")


def event_flags(ev: pl.DataFrame, asof: pl.Series) -> pl.DataFrame:
    """Daily event flags per decision time asof (= t + 1 00:00): next = scheduled in [asof, asof + 1 d) and schedule
    known by asof; today = scheduled in [asof - 1 d, asof)."""
    a = asof.to_numpy()
    st = ev["scheduled_ts"].to_numpy()
    sk = ev["schedule_known_ts"].to_numpy()
    kind = ev["kind"].to_numpy()
    one = np.timedelta64(1, "D")
    out = {"asof": asof}
    kt_next = np.full(len(a), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    for name, kinds, mode in (("ev_macro_next", MACRO, "next"), ("ev_crypto_next", CRYPTO, "next"), ("ev_macro_today", MACRO, "today")):
        m = np.isin(kind, kinds)
        s, k = st[m], sk[m]
        flag = np.zeros(len(a), dtype=float)
        for i, t in enumerate(a):
            if mode == "next":
                hit = (s >= t) & (s < t + one) & (k <= t)
                if hit.any():
                    flag[i] = 1.0
                    km = k[hit].max()
                    kt_next[i] = km if np.isnat(kt_next[i]) else max(kt_next[i], km)
            else:
                flag[i] = float(((s >= t - one) & (s < t)).any())
        out[name] = flag
    out["kt_events"] = kt_next
    return pl.DataFrame(out).with_columns(pl.col("kt_events").cast(pl.Datetime("us", "UTC")))


def daily_table(man: Manifest, ev: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    """One row per asof in [start, end] (UTC midnights) with the full block, *_l1 news variants and kt_* columns."""
    asof = pl.datetime_range(start, end, interval="1d", time_zone="UTC", eager=True).alias("asof").cast(pl.Datetime("us", "UTC"))
    base = pl.DataFrame({"asof": asof})
    nb = news_block(gdelt_daily(man))
    if nb.height:
        n0 = nb.rename({"known_ts": "kt_news", "date": "news_date"}).sort("kt_news")
        base = base.join_asof(n0, left_on="asof", right_on="kt_news", strategy="backward")
        n1 = nb.select("known_ts", *NEWS_COLS).with_columns((pl.col("known_ts") + timedelta(days=1)).alias("kt_news_l1")).drop("known_ts")
        n1 = n1.rename({c: f"{c}_l1" for c in NEWS_COLS}).sort("kt_news_l1")
        base = base.join_asof(n1, left_on="asof", right_on="kt_news_l1", strategy="backward")
        # a stale match (no news for > 2 days) is missing data, not a feature
        base = base.with_columns([pl.when(pl.col("asof") - pl.col("kt_news") <= timedelta(days=2)).then(pl.col(c)).otherwise(None).alias(c) for c in NEWS_COLS])
    oc = onchain_block(man)
    if oc.height:
        oc = oc.rename({"known_ts": "kt_onchain", "date": "onchain_date"})
        base = base.join_asof(oc, left_on="asof", right_on="kt_onchain", strategy="backward")
        base = base.with_columns([pl.when(pl.col("asof") - pl.col("kt_onchain") <= timedelta(days=3)).then(pl.col(c)).otherwise(None).alias(c)
                                  for c in ONCHAIN_COLS if c in base.columns])
    if ev.height:
        base = base.join(event_flags(ev, base["asof"]), on="asof", how="left")
    return base.sort("asof")


def hourly_table(man: Manifest, ev: pl.DataFrame, start: datetime, end: datetime, daily: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per hour H (decision at H for the hour [H, H + 1 h)). Columns:
      g<q>_n, g<q>_share  GDELT 15-min buckets of [H - 1 h, H) summed (known at H), for the intraday queries
      g<q>_abn            ln share minus its trailing 7-day mean of hourly ln shares (needs >= 3 days)
      ev_next_1h          a scheduled macro / crypto event in [H, H + 1 h) with the schedule known by H
      h_to_event, h_since_event   hours to the next known scheduled event / since the last one (capped at 720)
      mempool             last 15-min mempool sample with sample_ts + 1 h <= H (bytes)
      rss_n               RSS items first seen in [H - 1 h, H) (forward-only; null before the log starts)
      + the daily block of the latest daily asof <= H (prefixed d_)."""
    hours = pl.datetime_range(start, end, interval="1h", time_zone="UTC", eager=True).alias("asof").cast(pl.Datetime("us", "UTC"))
    base = pl.DataFrame({"asof": hours})
    for q in gdelt.CORE_INTRADAY:
        v = gdelt.load_chunks(man, "timelinevolraw", q, daily=False)
        if v.height == 0:
            continue
        h = (v.with_columns((pl.col("ts").dt.truncate("1h") + timedelta(hours=1)).alias("asof"))
             .group_by("asof").agg(pl.col("value").sum().alias(f"g{q}_n"), pl.col("norm").sum().alias("_norm"), pl.len().alias("_k"))
             .filter(pl.col("_k") == 4).sort("asof"))
        ls = np.log((h[f"g{q}_n"].to_numpy() + 1.0) / np.maximum(h["_norm"].to_numpy(), 1.0))
        h = h.with_columns(pl.Series(f"g{q}_share", np.exp(ls)), pl.Series(f"g{q}_abn", ls - _trail_mean(ls, 168, 72))).drop("_norm", "_k")
        base = base.join(h, on="asof", how="left")
    if ev.height:
        a = base["asof"].to_numpy()
        e = ev.filter(pl.col("kind").is_in(MACRO + CRYPTO))
        st, sk = e["scheduled_ts"].to_numpy(), e["schedule_known_ts"].to_numpy()
        hr = np.timedelta64(1, "h")
        nxt = np.zeros(len(a))
        to_e = np.full(len(a), 720.0)
        since = np.full(len(a), 720.0)
        for i, t in enumerate(a):
            known = sk <= t
            fut = known & (st >= t)
            nxt[i] = float((fut & (st < t + hr)).any())
            if fut.any():
                to_e[i] = min(720.0, (st[fut].min() - t) / hr)
            past = st <= t
            if past.any():
                since[i] = min(720.0, (t - st[past].max()) / hr)
        base = base.with_columns(pl.Series("ev_next_1h", nxt), pl.Series("h_to_event", to_e), pl.Series("h_since_event", since))
    mp = onchain.load(man, "mempool-size")
    if mp.height:
        mp = mp.with_columns((pl.col("ts") + onchain.MEMPOOL_LAG).alias("kt_mempool")).select("kt_mempool", pl.col("value").alias("mempool")).sort("kt_mempool")
        base = base.join_asof(mp, left_on="asof", right_on="kt_mempool", strategy="backward")
    rc = rss.hourly_counts(man)
    if rc.height:
        r = rc.group_by("known_ts").agg(pl.col("n").sum().alias("rss_n"), pl.col("first_poll").any().alias("rss_first_poll"))
        base = base.join(r.rename({"known_ts": "asof"}), on="asof", how="left")
        first = rc["known_ts"].min()
        base = base.with_columns(pl.when(pl.col("asof") >= first).then(pl.col("rss_n").fill_null(0)).otherwise(None).alias("rss_n"))
    if daily is not None and daily.height:
        d = daily.select("asof", *[c for c in BLOCK if c in daily.columns]).rename({c: f"d_{c}" for c in BLOCK if c in daily.columns})
        base = base.join_asof(d.rename({"asof": "d_asof"}).sort("d_asof"), left_on="asof", right_on="d_asof", strategy="backward")
    return base.sort("asof")


def assert_point_in_time(df: pl.DataFrame) -> None:
    """Every kt_* column must be <= asof wherever it is set."""
    for c in df.columns:
        if c.startswith("kt_") or c in ("d_asof",):
            bad = df.filter(pl.col(c).is_not_null() & (pl.col(c) > pl.col("asof")))
            if bad.height:
                raise AssertionError(f"{c} after asof in {bad.height} rows, e.g. {bad.row(0, named=True)}")


def build_all(man: Manifest, ev: pl.DataFrame, start: datetime, end: datetime, hourly_days: int = 84) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = daily_table(man, ev, start, end)
    assert_point_in_time(d)
    h = hourly_table(man, ev, end - timedelta(days=hourly_days), end, daily=d)
    assert_point_in_time(h)
    out = man.root / "features"
    out.mkdir(parents=True, exist_ok=True)
    d.write_parquet(out / "daily.parquet")
    h.write_parquet(out / "hourly.parquet")
    return d, h


__all__ = ["BLOCK", "EVENT_COLS", "NEWS_COLS", "ONCHAIN_COLS", "assert_point_in_time", "build_all", "daily_table", "event_flags",
           "gdelt_daily", "hourly_table", "news_block", "onchain_block"]
