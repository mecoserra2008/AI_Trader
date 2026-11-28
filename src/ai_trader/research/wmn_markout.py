"""WMN markouts of passive (maker) fills around news and jumps vs quiet windows (H-WMN-10 / 11 / 14).

Per symbol-day of public trades (public.bybit.com archive; raw deleted after parsing):
  maker markout MO(h) = d (p - m(t + h)) / m(t-) x 1e4 bp, d = +1 for a taker buy (the maker sold);
  m = midpoint proxy (last taker-buy price + last taker-sell price) / 2 when both are within 60 s, else the last price;
  m(t-) uses trades strictly before the trade; m(t + h) the last trade at or before t + h.
Controls (past trades only): aligned = d x OFI_5m, |OFI_5m| (OFI = signed / total notional over the previous 300 s),
VPIN_1h (sum |buy - sell| over the 12 previous complete 5-min clock buckets / their volume), log size minus log of the
previous clock hour's median size, log RV_1h (squared 1-min proxy-mid returns over the previous 60 complete minutes).
States: age (minutes) of the latest precise token news event known <= t, of the latest D episode known <= t, of the
latest macro release <= t. Trades in the first hour of the day are dropped (control warm-up).
Sampling: all trades within 60 min after a news event or an episode, plus a uniform sample of the rest (<= 15k per
symbol-day); weight = 1 / inclusion probability.
"""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl

from ai_trader.research import wmn_data as wd
from ai_trader.research.wmn_study import CUTOFF_MS, DAY_MS, ols_cluster, precise_token_events

HORIZONS = (1, 10, 60, 300, 1800)
OUTDIR = wd.OUT / "markouts"


def _ffill_idx(mask: np.ndarray) -> np.ndarray:
    """Index of the latest True at or before each position (-1 if none)."""
    idx = np.where(mask, np.arange(mask.size), -1)
    return np.maximum.accumulate(idx)


def day_features(tr: pl.DataFrame, day_ms: int, news_ts: np.ndarray, ep_ts: np.ndarray, mac_ts: np.ndarray,
                 rng: np.random.Generator, cap: int = 15_000) -> pl.DataFrame | None:
    """Selected trades of one symbol-day with markouts, controls and states. The sample is chosen first (states only
    need trade times), then outcomes and controls are computed for the selected trades only (memory)."""
    t = tr["t"].to_numpy()
    n = t.size
    if n < 500:
        return None
    tms = (t * 1000).astype(np.int64)

    def age(src: np.ndarray) -> np.ndarray:
        if src.size == 0:
            return np.full(n, np.inf, dtype=np.float32)
        k = np.searchsorted(src, tms, side="right") - 1
        return np.where(k >= 0, (tms - src[np.maximum(k, 0)]) / 60_000.0, np.inf).astype(np.float32)

    news_age, jump_age, mac_age = age(news_ts), age(ep_ts), age(mac_ts)
    del tms
    keep = t >= day_ms / 1000 + 3600
    hot = keep & ((news_age < 60) | (jump_age < 60))
    rest = np.nonzero(keep & ~hot)[0]
    pr = min(1.0, cap / max(rest.size, 1))
    pick = rest[rng.random(rest.size) < pr] if pr < 1 else rest
    sel = np.sort(np.r_[np.nonzero(hot)[0], pick])
    w = np.where(hot[sel], 1.0, 1.0 / pr)
    del keep, rest, pick

    d = tr["d"].to_numpy().astype(np.int8)
    p = tr["price"].to_numpy()
    q = tr["notional"].to_numpy().astype(np.float64)
    ib, is_ = _ffill_idx(d == 1), _ffill_idx(d == -1)
    both = (t - np.where(ib >= 0, t[np.maximum(ib, 0)], -1e18) <= 60) & (t - np.where(is_ >= 0, t[np.maximum(is_, 0)], -1e18) <= 60)
    m_after = np.where(both, 0.5 * (p[np.maximum(ib, 0)] + p[np.maximum(is_, 0)]), p)  # proxy mid after trade i
    del ib, is_, both
    ts_, ds, ps = t[sel], d[sel].astype(np.float64), p[sel]
    mb = m_after[np.maximum(sel - 1, 0)]
    mb = np.where(sel >= 1, mb, np.nan)
    out = {"t": ts_, "d": ds.astype(np.float32), "eff": (ds * (ps - mb) / mb * 1e4).astype(np.float32)}
    for h in HORIZONS:
        j = np.searchsorted(t, ts_ + h, side="right") - 1
        out[f"mo{h}"] = np.where(ts_ + h <= t[-1], ds * (ps - m_after[j]) / mb * 1e4, np.nan).astype(np.float32)
    ct = np.r_[0.0, np.cumsum(q)]
    cs = np.r_[0.0, np.cumsum(d * q)]
    i0 = np.searchsorted(t, ts_ - 300, side="left")
    tot = ct[sel] - ct[i0]
    ofi = np.where(tot > 0, (cs[sel] - cs[i0]) / np.where(tot > 0, tot, 1), 0.0)
    out["aligned"] = (ds * ofi).astype(np.float32)
    out["absofi"] = np.abs(ofi).astype(np.float32)
    del ct, cs
    b5 = (t // 300).astype(np.int64)
    b0 = b5.min()
    bi = b5 - b0
    nb = int(bi.max()) + 1
    B = np.bincount(bi, weights=np.where(d == 1, q, 0.0), minlength=nb)
    S = np.bincount(bi, weights=np.where(d == -1, q, 0.0), minlength=nb)
    ci, cv = np.r_[0, np.cumsum(np.abs(B - S))], np.r_[0, np.cumsum(B + S)]
    bs = bi[sel]
    lo = np.maximum(bs - 12, 0)
    vv = cv[bs] - cv[lo]
    out["vpin"] = np.where(vv > 0, (ci[bs] - ci[lo]) / np.where(vv > 0, vv, 1), np.nan).astype(np.float32)
    hr = (t // 3600).astype(np.int64)
    lq = np.log(np.maximum(q, 1e-9))
    hm = dict(pl.DataFrame({"h": hr, "lq": lq}).group_by("h").agg(pl.col("lq").median()).iter_rows())
    hs = hr[sel]
    prev = np.array([hm.get(int(h) - 1, np.nan) for h in hs])
    out["lsize"] = (lq[sel] - prev).astype(np.float32)
    del hr, lq
    mn = (t // 60).astype(np.int64)
    mi = mn - mn.min()
    nm = int(mi.max()) + 1
    last_idx = np.full(nm, -1)
    np.maximum.at(last_idx, mi, np.arange(n))
    mm = np.where(last_idx >= 0, m_after[np.maximum(last_idx, 0)], np.nan)
    mm = pl.Series(mm).fill_null(strategy="forward").fill_nan(None).fill_null(strategy="forward").to_numpy()
    r1 = np.r_[np.nan, np.diff(np.log(mm))]
    c2 = np.r_[0, np.cumsum(np.where(np.isfinite(r1), r1 ** 2, 0.0))]
    ms_ = mi[sel]
    rv = c2[ms_] - c2[np.maximum(ms_ - 60, 0)]
    out["lrv"] = np.log(np.maximum(rv, 1e-12)).astype(np.float32)
    out["news_age"], out["jump_age"], out["mac_age"] = news_age[sel], jump_age[sel], mac_age[sel]
    df = pl.DataFrame(out)
    return df.with_columns(pl.Series("w", w), pl.lit(int(n)).alias("n_day"), pl.lit(float(np.median(q))).alias("med_notional"))


def select_days(seed: int = 20261127, n_news: int = 20, n_jump: int = 15) -> pl.DataFrame:
    """Symbol-days: news days (precise token events on MM-relevant core alts), unlabelled-jump days, and a control day
    7 days earlier (14 if that one has an event or episode within a day). Per segment n_news / n_jump."""
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = precise_token_events(ev)
    pool = [s for s in wd.MM_NAMES + wd.LAKE_ALTS]
    E = pl.read_parquet(wd.OUT / "episodes.parquet").filter((pl.col("flag") == "D") & pl.col("symbol").is_in(pool))
    pt = pt.filter(pl.col("symbol").is_in(pool) & (pl.col("known_ts") >= wd.ms(wd.START) + 8 * DAY_MS))
    rng = np.random.default_rng(seed)
    nk = {s[0]: np.sort(g["known_ts"].to_numpy()) for s, g in pt.group_by("symbol")}
    ek = {s[0]: np.sort(g["ts"].to_numpy()) for s, g in E.group_by("symbol")}
    busy = lambda s, a: bool(np.any(np.abs(nk.get(s, np.array([])) - a) < DAY_MS) or np.any(np.abs(ek.get(s, np.array([])) - a) < DAY_MS))  # noqa: E731
    rows = []
    for seg_lo, seg_hi, seg in ((wd.ms(wd.START) + 8 * DAY_MS, CUTOFF_MS, "llm_in_sample"), (CUTOFF_MS, wd.ms(wd.datetime.now(wd.UTC)) - 2 * DAY_MS, "clean_holdout")):
        cand = pt.filter((pl.col("known_ts") >= seg_lo) & (pl.col("known_ts") < seg_hi) & (pl.col("group") != "hn_story"))
        if cand.height > n_news:
            cand = cand[np.sort(rng.choice(cand.height, n_news, replace=False)).tolist()]
        for sym, kt in cand.select("symbol", "known_ts").iter_rows():
            rows.append((sym, kt // DAY_MS * DAY_MS, "news", seg))
            if (kt % DAY_MS) >= 12 * 3_600_000:
                rows.append((sym, kt // DAY_MS * DAY_MS + DAY_MS, "news_next", seg))
        ej = E.filter((pl.col("ts") >= seg_lo) & (pl.col("ts") < seg_hi))
        ej = ej.filter(pl.Series([not np.any(np.abs(nk.get(s, np.array([])) - t) < 2 * 3_600_000) for s, t in ej.select("symbol", "ts").iter_rows()]))
        if ej.height > n_jump:
            ej = ej[np.sort(rng.choice(ej.height, n_jump, replace=False)).tolist()]
        for sym, t in ej.select("symbol", "ts").iter_rows():
            rows.append((sym, t // DAY_MS * DAY_MS, "jump", seg))
    base = list(rows)
    for sym, day, kind, seg in base:
        if kind == "news_next":
            continue
        for back in (7, 14):
            c = day - back * DAY_MS
            if not busy(sym, c + DAY_MS // 2):
                rows.append((sym, c, "control", seg))
                break
    df = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "day_ms": pl.Int64, "kind": pl.Utf8, "segment": pl.Utf8}, orient="row")
    return df.unique(["symbol", "day_ms"], keep="first").sort("day_ms")


def run(log=print, seed: int = 20261127) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    days = select_days(seed)
    days.write_parquet(OUTDIR / "days.parquet")
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = precise_token_events(ev)
    E = pl.read_parquet(wd.OUT / "episodes.parquet").filter(pl.col("flag") == "D")
    mac = np.sort(ev.filter(pl.col("category") == "macro_release")["known_ts"].to_numpy())
    rng = np.random.default_rng(seed)
    done = {p.stem for p in OUTDIR.glob("*_*.parquet") if p.stem != "days"}
    stats = []
    for sym, day_ms, kind, seg in days.iter_rows():
        tag = f"{sym}_{day_ms // DAY_MS}"
        if tag in done:
            continue
        day = wd.datetime.fromtimestamp(day_ms / 1000, wd.UTC).strftime("%Y-%m-%d")
        nskip = len(wd.ARCHIVE_SKIPPED)
        tr = wd.archive_trades(sym, day)
        if tr is None:
            stats.append({"symbol": sym, "day": day, "status": "too_large" if len(wd.ARCHIVE_SKIPPED) > nskip else "missing"})
            continue
        news = np.sort(pt.filter(pl.col("symbol") == sym)["known_ts"].to_numpy())
        eps = np.sort(E.filter(pl.col("symbol") == sym)["known_ts"].to_numpy())
        f = day_features(tr, day_ms, news, eps, mac, rng)
        if f is None:
            stats.append({"symbol": sym, "day": day, "status": "few_trades", "n": tr.height})
            continue
        f.with_columns(pl.lit(sym).alias("symbol"), pl.lit(day_ms).alias("day_ms"), pl.lit(kind).alias("kind"),
                       pl.lit(seg).alias("segment")).write_parquet(OUTDIR / f"{tag}.parquet")
        stats.append({"symbol": sym, "day": day, "status": "ok", "n": tr.height, "kept": f.height})
        log(f"  {tag} {kind} {seg}: {tr.height} trades, kept {f.height}", flush=True)
    (OUTDIR / "run_stats.json").write_text(json.dumps(stats, indent=1))
    return {"days": days.height, "ok": sum(1 for s in stats if s["status"] == "ok")}


def load_sample() -> pl.DataFrame:
    fs = [p for p in OUTDIR.glob("*_*.parquet") if p.stem not in ("days",)]
    out = []
    for p in fs:
        d = pl.read_parquet(p)
        out.append(d.with_columns([pl.col(c).cast(pl.Float32) for c, t in d.schema.items() if t == pl.Float64 and c not in ("t",)]))
    return pl.concat(out, how="diagonal")


def regress(S: pl.DataFrame, state_cols: list[str], h: int, extra: list[str] | None = None) -> dict:
    """Weighted OLS of MO(h) on states + registered controls + symbol FE, CR1 by symbol-day."""
    ctrl = ["aligned", "absofi", "vpin", "lsize", "lrv"] + (extra or [])
    S = S.select(["symbol", "day_ms", "w", f"mo{h}"] + list(dict.fromkeys(state_cols + ctrl)))
    S = S.filter(pl.col(f"mo{h}").is_finite() & pl.all_horizontal([pl.col(c).is_finite() for c in ctrl]))
    if S.height < 1000:
        return {"n": S.height}
    # symbol fixed effects by the weighted within transformation (no dummy matrix)
    _, sid = np.unique(S["symbol"].to_numpy(), return_inverse=True)
    w = S["w"].to_numpy().astype(float)
    sw = np.bincount(sid, weights=w)

    def within(v: np.ndarray) -> np.ndarray:
        return v - (np.bincount(sid, weights=w * v) / sw)[sid]

    X = np.column_stack([within(S[c].to_numpy().astype(float)).astype(np.float32) for c in state_cols + ctrl])
    y = within(np.clip(S[f"mo{h}"].to_numpy().astype(float), -500, 500))
    cl = sid.astype(np.int64) * 100_000 + (S["day_ms"].to_numpy() // DAY_MS)
    b, se, n, G = ols_cluster(X, y, cl, w)
    names = state_cols + ctrl
    return {"n": n, "clusters": G, **{nm: {"coef": float(b[i]), "se": float(se[i]), "t": float(b[i] / se[i]) if se[i] > 0 else float("nan")}
                                      for i, nm in enumerate(names)}}


def with_states(S: pl.DataFrame) -> pl.DataFrame:
    bins = [(0, 15), (15, 60), (60, 240), (240, 720), (720, 2880)]
    cols = [(pl.col("news_age") < 60).cast(pl.Float64).alias("news60"), (pl.col("jump_age") < 60).cast(pl.Float64).alias("jump60"),
            (pl.col("mac_age") < 60).cast(pl.Float64).alias("mac60")]
    for a, b in bins:
        cols.append(((pl.col("news_age") >= a) & (pl.col("news_age") < b)).cast(pl.Float64).alias(f"news_{a}_{b}"))
        cols.append(((pl.col("jump_age") >= a) & (pl.col("jump_age") < b)).cast(pl.Float64).alias(f"jump_{a}_{b}"))
    return S.with_columns([c.cast(pl.Float32) for c in cols])


def analyse(S: pl.DataFrame | None = None) -> dict:
    S = with_states(load_sample() if S is None else S).drop(["med_notional", "n_day", "kind"], strict=False)
    res = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        s = S.filter(pl.col("segment") == seg)
        r = {"n_trades": s.height, "symbol_days": s.select("symbol", "day_ms").unique().height}
        for h in HORIZONS:
            r[f"h{h}"] = regress(s, ["news60", "jump60"], h)
            r[f"h{h}_bins"] = regress(s, [f"news_{a}_{b}" for a, b in [(0, 15), (15, 60), (60, 240), (240, 720), (720, 2880)]]
                                      + [f"jump_{a}_{b}" for a, b in [(0, 15), (15, 60), (60, 240), (240, 720), (720, 2880)]], h)
            r[f"h{h}_nocontrols"] = _raw_means(s, h)
            r[f"h{h}_macro"] = regress(s, ["news60", "jump60", "mac60"], h)
        res[seg] = r
    return res


def _raw_means(s: pl.DataFrame, h: int) -> dict:
    out = {}
    for name, m in (("news60", pl.col("news_age") < 60), ("jump60", (pl.col("jump_age") < 60) & (pl.col("news_age") >= 60)),
                    ("quiet", (pl.col("jump_age") >= 240) & (pl.col("news_age") >= 240))):
        x = s.filter(m & pl.col(f"mo{h}").is_finite())
        if x.height == 0:
            out[name] = None
            continue
        w = x["w"].to_numpy()
        y = np.clip(x[f"mo{h}"].to_numpy(), -500, 500)
        out[name] = {"mean_bp": float(np.sum(w * y) / np.sum(w)), "n": x.height,
                     "eff_bp": float(np.nansum(w * x["eff"].to_numpy()) / np.sum(w))}
    return out
