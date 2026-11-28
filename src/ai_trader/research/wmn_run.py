"""WMN pipeline stages (each run through scripts/ops/governor.py):

    python -m ai_trader.research.wmn_run detect | es_token | es_episodes | es_market | tests | hawkes | gate_fit |
                                         gate_eval | gate_write | markouts | markout_tests | live
Results go to data/research/wmn/results/*.json; nothing is recorded here (wmn_record.py records under the lock).
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.research import wmn_data as wd
from ai_trader.research import wmn_gate as wg
from ai_trader.research import wmn_study as ws

RES = wd.OUT / "results"
PRECISE = ["bybit_listing_spot", "bybit_listing_perp", "bybit_delist", "x_listing", "x_delist", "maintenance", "hn_story"]
EXCH = ["bybit_listing_spot", "bybit_listing_perp", "bybit_delist", "x_listing", "x_delist", "maintenance"]
H4_GROUPS = ["bybit_listing_spot", "bybit_delist", "x_listing", "x_delist", "maintenance", "hn_story", "hack"]


def save(name: str, obj) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda x: float(x) if isinstance(x, np.floating) else str(x)),
                                      encoding="utf-8")


def load(name: str):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def seg_col(col: str = "known_ts") -> pl.Expr:
    return pl.when(pl.col(col) < ws.CUTOFF_MS).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment")


def paired(T: pl.DataFrame, metric: str, absval: bool = False) -> pl.DataFrame:
    ev = T.filter(~pl.col("placebo"))
    pl_ = T.filter(pl.col("placebo")).select("event_id", "symbol", pl.col(metric).alias("p"))
    j = ev.join(pl_, on=["event_id", "symbol"], how="inner")
    x = pl.col(metric).abs() if absval else pl.col(metric)
    p = pl.col("p").abs() if absval else pl.col("p")
    return j.with_columns((x - p).alias("diff"))


def test_paired(T: pl.DataFrame, metric: str, absval: bool = False) -> dict:
    out = {}
    j = paired(T, metric, absval)
    for seg in ("llm_in_sample", "clean_holdout"):
        s = j.filter(pl.col("segment") == seg)
        d = s["diff"].to_numpy().astype(float)
        cl = ws._cluster_day(s["anchor"].to_numpy())
        mu, se, n, G = ws.cluster_mean(d, cl)
        ev_mu = float(np.nanmean(s[metric].abs().to_numpy() if absval else s[metric].to_numpy())) if s.height else float("nan")
        out[seg] = {"diff": mu, "se": se, "t": mu / se if se and se > 0 else float("nan"), "n": n, "clusters": G,
                    "event_mean": ev_mu, "placebo_mean": float(np.nanmean(s["p"].abs().to_numpy() if absval else s["p"].to_numpy())) if s.height else float("nan")}
    return out


# ------------------------------------------------------------------------------------------- stage: tests
def stage_tests() -> dict:
    T = pl.read_parquet(wd.OUT / "es_token.parquet").with_columns(seg_col())
    res = {"counts": T.group_by("group", "placebo", "segment").len().sort("group").to_dicts()}
    prec = T.filter(pl.col("group").is_in(PRECISE))
    # H-WMN-3
    res["H-WMN-3"] = test_paired(prec, "ARV_h0_1")
    res["H-WMN-3_extra"] = {m: test_paired(prec, m) for m in ("ARV_pre", "ARV_h1_4", "ARV_h4_24", "AV_h0_1", "CAR_h0_48")}
    # weekend / weekday split (descriptive inside H-WMN-3)
    # H-WMN-4
    fam, rows = [], []
    for g in H4_GROUPS:
        sub = T.filter(pl.col("group") == g)
        if g == "hack":
            mets = [("ARV_h0_24", False), ("ARV_h24_48", False), ("AV_h0_24", False), ("CAR_h0_48", True)]
        else:
            mets = [("ARV_h0_1", False), ("ARV_h1_4", False), ("AV_h0_1", False), ("CAR_h0_48", True)]
        for m, ab in mets:
            r = test_paired(sub, m, ab)
            rows.append({"group": g, "metric": ("|" + m + "|") if ab else m, **{f"{s}_{k}": v for s in r for k, v in r[s].items()}})
    tab = pl.DataFrame(rows)
    elig = tab.filter(pl.col("llm_in_sample_n") >= 30)
    pv = [float(1 - 0.5 * (1 + math.erf(t / math.sqrt(2)))) if np.isfinite(t) else 1.0 for t in elig["llm_in_sample_t"].to_list()]
    rej = ws.bh(pv, 0.10) if pv else []
    elig = elig.with_columns(pl.Series("p_one", pv), pl.Series("bh", rej))
    res["H-WMN-4"] = {"table": tab.to_dicts(), "bh_family": elig.select("group", "metric", "llm_in_sample_diff", "llm_in_sample_t", "p_one", "bh",
                                                                       "clean_holdout_diff", "clean_holdout_n").to_dicts()}
    # H-WMN-5 continuation
    E = pl.read_parquet(wd.OUT / "es_episodes.parquet").with_columns(seg_col("anchor"))
    ev = prec.filter(~pl.col("placebo") & (pl.col("SCAR_h0_1").abs() >= 2))
    jp = E.filter(~pl.col("news") & (pl.col("symbol") != "BTCUSDT") & (pl.col("SCAR_h0_1").abs() >= 2))
    h5 = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        a = ev.filter(pl.col("segment") == seg)
        b = jp.filter(pl.col("segment") == seg)
        x = np.r_[a["CAR_h0_1"].to_numpy(), b["CAR_h0_1"].to_numpy()]
        y = np.r_[a["CAR_h1_48"].to_numpy(), b["CAR_h1_48"].to_numpy()]
        nw = np.r_[np.ones(a.height), np.zeros(b.height)]
        cl = ws._cluster_day(np.r_[a["anchor"].to_numpy(), b["anchor"].to_numpy()])
        X = np.column_stack([np.ones_like(x), x, nw, nw * x])
        bb, se, n, G = ws.ols_cluster(X, y, cl)
        h5[seg] = {"b_jump": float(bb[1]), "b_news": float(bb[1] + bb[3]), "diff": float(bb[3]), "se_diff": float(se[3]),
                   "t_diff": float(bb[3] / se[3]) if se[3] > 0 else float("nan"), "t_b_jump": float(bb[1] / se[1]) if se[1] > 0 else float("nan"),
                   "n_news": a.height, "n_jump": b.height, "clusters": G}
    res["H-WMN-5"] = h5
    # H-WMN-6 weekend vs weekday ARV[0,4h)
    evr = prec.filter(~pl.col("placebo")).select("anchor", "ARV_h0_4", pl.col("group").alias("type"))
    epr = E.filter(pl.col("symbol") != "BTCUSDT").select("anchor", "ARV_h0_4", pl.lit("episode").alias("type"))
    P = pl.concat([evr, epr]).with_columns(seg_col("anchor"),
                                          (pl.from_epoch("anchor", "ms").dt.weekday() >= 6).cast(pl.Float64).alias("wkend"))
    h6 = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        s = P.filter((pl.col("segment") == seg) & pl.col("ARV_h0_4").is_finite())
        types = s["type"].unique().sort().to_list()
        X = np.column_stack([s["wkend"].to_numpy()] + [(s["type"] == t).to_numpy().astype(float) for t in types])
        bb, se, n, G = ws.ols_cluster(X, s["ARV_h0_4"].to_numpy(), ws._cluster_day(s["anchor"].to_numpy()))
        h6[seg] = {"diff": float(bb[0]), "se": float(se[0]), "t": float(bb[0] / se[0]) if se[0] > 0 else float("nan"), "n": n,
                   "clusters": G, "n_weekend": int(s["wkend"].sum())}
        for typ in ("episode",):
            q = s.filter(pl.col("type") == typ)
            Xq = np.column_stack([np.ones(q.height), q["wkend"].to_numpy()])
            b2, s2, n2, G2 = ws.ols_cluster(Xq, q["ARV_h0_4"].to_numpy(), ws._cluster_day(q["anchor"].to_numpy()))
            h6[seg][f"{typ}_only"] = {"diff": float(b2[1]), "t": float(b2[1] / s2[1]) if s2[1] > 0 else float("nan"), "n": n2}
    res["H-WMN-6"] = h6
    save("event_tests", res)
    return res


def stage_detector_tests() -> dict:
    """H-WMN-1 (recall lift) and H-WMN-2 (explained share)."""
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = ws.precise_token_events(ev)
    E = pl.read_parquet(wd.OUT / "episodes.parquet")
    core = [s for s in wd.CORE if s != "BTCUSDT"]
    allk = {s[0]: np.sort(g["known_ts"].to_numpy()) for s, g in ws.precise_token_events(ev, include_low=True).group_by("symbol")}
    cov = {}
    for s in core:
        p = wd.BARS / f"{s}.parquet"
        if p.is_file():
            ts = pl.read_parquet(p, columns=["ts"])["ts"]
            cov[s] = (int(ts.min()) + 8 * ws.DAY_MS, int(ts.max()))
    res = {}
    for flag in ("D", "Dp"):
        ep = {s[0]: np.sort(g["ts"].to_numpy()) for s, g in E.filter(pl.col("flag") == flag).group_by("symbol")}
        x = pt.filter(pl.col("symbol").is_in(core) & pl.col("source").is_in(["bybit_ann", "okx_ann", "kucoin_ann"]))
        rows = []
        for sym, kt in x.select("symbol", "known_ts").iter_rows():
            if sym not in cov or not (cov[sym][0] <= kt - 7 * ws.DAY_MS and kt + 3_600_000 <= cov[sym][1]):
                continue
            e = ep.get(sym, np.array([], dtype=np.int64))
            hit = bool(np.any((e >= kt // ws.BAR_MS * ws.BAR_MS) & (e < kt + 3_600_000)))
            p0 = kt - 7 * ws.DAY_MS
            ks = allk.get(sym, np.array([]))
            if np.any(np.abs(ks - p0) < 2 * ws.DAY_MS):
                continue
            hp = bool(np.any((e >= p0 // ws.BAR_MS * ws.BAR_MS) & (e < p0 + 3_600_000)))
            rows.append((sym, kt, hit, hp))
        R = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "kt": pl.Int64, "hit": pl.Boolean, "hit_p": pl.Boolean}, orient="row").with_columns(seg_col("kt"))
        out = {}
        for seg in ("llm_in_sample", "clean_holdout"):
            s = R.filter(pl.col("segment") == seg)
            d = (s["hit"].cast(pl.Float64) - s["hit_p"].cast(pl.Float64)).to_numpy()
            mu, se, n, G = ws.cluster_mean(d, ws._cluster_day(s["kt"].to_numpy()))
            rec, recp = float(s["hit"].mean()) if s.height else float("nan"), float(s["hit_p"].mean()) if s.height else float("nan")
            out[seg] = {"recall": rec, "placebo_rate": recp, "lift": rec / recp if recp and recp > 0 else (float("nan") if not rec else float("inf")), "diff": mu,
                        "t": mu / se if se and se > 0 else float("nan"), "n": n, "clusters": G}
        res[f"H-WMN-1_{flag}"] = out
        # H-WMN-2: explained share of episodes on core alts
        rows2 = []
        pk = {s[0]: np.sort(g["known_ts"].to_numpy()) for s, g in pt.group_by("symbol")}
        for sym, e in ep.items():
            if sym not in core:
                continue
            k = pk.get(sym, np.array([], dtype=np.int64))
            for t in e:
                ex = bool(k.size and np.any((k >= t - 2 * 3_600_000) & (k <= t + 30 * 60_000)))
                t2 = t - 7 * ws.DAY_MS
                exp_ = bool(k.size and np.any((k >= t2 - 2 * 3_600_000) & (k <= t2 + 30 * 60_000)))
                rows2.append((sym, int(t), ex, exp_))
        R2 = pl.DataFrame(rows2, schema={"symbol": pl.Utf8, "t": pl.Int64, "ex": pl.Boolean, "ex_p": pl.Boolean}, orient="row").with_columns(seg_col("t"))
        out2 = {}
        for seg in ("llm_in_sample", "clean_holdout"):
            s = R2.filter(pl.col("segment") == seg)
            out2[seg] = {"episodes": s.height, "explained_share": float(s["ex"].mean()) if s.height else float("nan"),
                         "placebo_share": float(s["ex_p"].mean()) if s.height else float("nan"),
                         "by_source_note": "precise token events (exchange announcements + HN), [-2h, +30min]"}
        res[f"H-WMN-2_{flag}"] = out2
    save("detector_tests", res)
    return res


# -------------------------------------------------------------------------------------------- stage: decay
def _X(name: str) -> np.ndarray:
    """Post-shock arrays [x = z^2 (577), z (577)] per row. Stored as float16 z only (x is derived)."""
    d = np.load(wd.OUT / f"{name}_x.npz")
    if "z" in d:
        z = d["z"].astype(np.float32)
        return np.concatenate([z ** 2, z], axis=1)
    return d["x"]


def compact_x(name: str) -> None:
    """Rewrite <name>_x.npz as float16 z only (storage quota)."""
    p = wd.OUT / f"{name}_x.npz"
    d = np.load(p)
    if "z" in d:
        return
    z = np.clip(d["x"][:, 577:], -6e4, 6e4).astype(np.float16)
    np.savez_compressed(p, z=z)


def stage_decay() -> dict:
    T = pl.read_parquet(wd.OUT / "es_token.parquet")
    XT = _X("es_token")
    E = pl.read_parquet(wd.OUT / "es_episodes.parquet")
    XE = _X("es_episodes")
    M = pl.read_parquet(wd.OUT / "es_market.parquet")
    XM = _X("es_market")
    res = {}
    news = T.filter(~pl.col("placebo") & pl.col("group").is_in(PRECISE))
    unl = E.filter(~pl.col("news") & (pl.col("symbol") != "BTCUSDT"))
    sets = {"a_news": (XT[news["row"].to_numpy()], news["anchor"].to_numpy()),
            "b_unlabelled": (XE[unl["row"].to_numpy()], unl["anchor"].to_numpy())}
    for name, (X, anc) in sets.items():
        Z = X[:, 577:]
        res[name] = ws.decay_compare(Z, anc)
        res[name]["profile"] = {seg: ws.mean_profile(X[ws.seg_of(anc) == seg, :577]) for seg in ("llm_in_sample", "clean_holdout")}
    # H-WMN-8 persistence ratio
    newsj = E.filter(pl.col("news") & (pl.col("symbol") != "BTCUSDT"))
    big = news.filter(pl.col("SCAR_h0_1").abs() >= 2)
    Xn = np.vstack([XE[newsj["row"].to_numpy(), :577], XT[big["row"].to_numpy(), :577]])
    an = np.r_[newsj["anchor"].to_numpy(), big["anchor"].to_numpy()]
    Xu, au = XE[unl["row"].to_numpy(), :577], unl["anchor"].to_numpy()
    h8 = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        mn, mu_ = ws.seg_of(an) == seg, ws.seg_of(au) == seg
        if mn.sum() < 5 or mu_.sum() < 5:
            h8[seg] = {"n_news": int(mn.sum()), "n_unl": int(mu_.sum())}
            continue
        r, lo, hi, nb = ws.cluster_boot_ratio(Xn[mn], ws._cluster_day(an[mn]), Xu[mu_], ws._cluster_day(au[mu_]))
        h8[seg] = {"ratio": r, "ci90": [lo, hi], "boot": nb, "PR_news": ws.persistence_ratio(Xn[mn]), "PR_unl": ws.persistence_ratio(Xu[mu_]),
                   "n_news": int(mn.sum()), "n_unl": int(mu_.sum()), "profile_news": ws.mean_profile(Xn[mn]), "profile_unl": ws.mean_profile(Xu[mu_])}
    res["H-WMN-8"] = h8
    # per-category / market kernels (FIT only) for the gate
    kern = {}
    spec_news = "pow" if res["a_news"].get("llm_in_sample", {}).get("diff", -1) > 0 else "exp"
    spec_jump = "pow" if res["b_unlabelled"].get("llm_in_sample", {}).get("diff", -1) > 0 else "exp"
    fitm = ws.seg_of(news["anchor"].to_numpy()) == "llm_in_sample"
    Zn = XT[news["row"].to_numpy(), 577:]
    kern["news_pooled"] = ws.decay_fit(Zn[fitm], spec_news)
    for g in PRECISE:
        m = (news["group"].to_numpy() == g) & fitm
        if m.sum() >= 30:
            kern[g] = ws.decay_fit(Zn[m], spec_news)
            kern[g]["n_fit"] = int(m.sum())
    allep = E.filter(pl.col("symbol") != "BTCUSDT")
    fm = ws.seg_of(allep["anchor"].to_numpy()) == "llm_in_sample"
    kern["jump"] = ws.decay_fit(XE[allep["row"].to_numpy(), 577:][fm], spec_jump)
    be = E.filter(pl.col("symbol") == "BTCUSDT")
    fb = ws.seg_of(be["anchor"].to_numpy()) == "llm_in_sample"
    Zb = XE[be["row"].to_numpy(), 577:][fb]
    mac = M.filter((pl.col("group") == "macro_release") & (pl.col("series") == "BTC") & ~pl.col("placebo"))
    fmac = ws.seg_of(mac["anchor"].to_numpy()) == "llm_in_sample"
    Zm = XM[mac["row"].to_numpy(), 577:][fmac]
    for name, Z, anc in (("btc_jump", Zb, be["anchor"].to_numpy()[fb]), ("macro", Zm, mac["anchor"].to_numpy()[fmac])):
        cmp_ = ws.decay_compare(Z, anc) if Z.shape[0] >= 40 else {}
        sp = "pow" if cmp_.get("llm_in_sample", {}).get("diff", -1) > 0 else "exp"
        kern[name] = ws.decay_fit(Z, sp)
        kern[name]["choice_dm"] = cmp_.get("llm_in_sample")
        kern[name]["n_fit"] = int(Z.shape[0])
    for k, v in kern.items():
        v["half_life_min"] = ws.half_life(v)
        v["decay10_min"] = ws.decay_time(v)
    res["kernels_fit"] = kern
    res["spec_choice"] = {"news": spec_news, "jump": spec_jump}
    save("decay", res)
    return res


def stage_gate_fit() -> dict:
    d = load("decay")
    ks = {}
    for k, v in d["kernels_fit"].items():
        th = v["theta"]
        ks[k] = wg.Kernel(v["spec"], float(th[1]), [float(x) for x in th[2:]])
    cmap = {g: (g if g in ks else "news_pooled") for g in PRECISE}
    p = wg.GateParams(ks, cmap, meta={"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"), "fit": "llm_in_sample shocks < 2026-07-01",
                                      "rule": "lambda_news = w*lam_mkt + (1-w)*lam_tok; pull if lambda_news >= 3 or [release-2min, release); widen = sqrt(1+min(lambda,3))",
                                      "source": "wmn_run.stage_decay kernels_fit", "spec_choice": d["spec_choice"]})
    p.dump(wd.OUT / "gate_params.json")
    return {"kernels": {k: {"spec": v.spec, "A": v.A, "theta": v.theta} for k, v in ks.items()}, "map": cmap}


# ---------------------------------------------------------------------------------------------- gate eval
def _weekend_windows(lo_ms: int, hi_ms: int) -> list[tuple[int, int]]:
    d = datetime.fromtimestamp(lo_ms / 1000, UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    d -= timedelta(days=(d.weekday() - 4) % 7)
    out = []
    while wd.ms(d) < hi_ms:
        a, b = wd.ms(d), wd.ms(d + timedelta(days=3))
        if b > lo_ms:
            out.append((max(a, lo_ms), min(b, hi_ms)))
        d += timedelta(days=7)
    return out


def _w_daily(sym_ts: np.ndarray, r: np.ndarray, rb: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """w per bar from daily 7-day variance of r and r_BTC (previous days) and the detector's daily beta."""
    day = sym_ts // ws.DAY_MS
    d0 = day.min()
    di = day - d0
    nd = di.max() + 1
    ok = np.isfinite(r) & np.isfinite(rb)
    S = np.zeros((nd, 5))
    np.add.at(S, di[ok], np.column_stack([r[ok] ** 2, r[ok], rb[ok] ** 2, rb[ok], np.ones(ok.sum())]))
    R = ws._daily_roll(S, 7)
    n = np.maximum(R[:, 4], 1)
    vs = R[:, 0] / n - (R[:, 1] / n) ** 2
    vb = R[:, 2] / n - (R[:, 3] / n) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (beta ** 2) * vb[di] / vs[di]
    return np.clip(np.nan_to_num(w, nan=0.5), 0, 1)


def gate_inputs():
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = ws.precise_token_events(ev)
    news = {s[0]: list(zip(g["known_ts"].to_list(), g["group"].to_list(), strict=True)) for s, g in pt.group_by("symbol")}
    E = pl.read_parquet(wd.OUT / "episodes.parquet").filter(pl.col("flag") == "D")
    eps = {s[0]: g["ts"].to_list() for s, g in E.group_by("symbol")}
    mac = sorted(ev.filter(pl.col("category") == "macro_release")["known_ts"].to_list())
    return news, eps, mac


def stage_gate_eval() -> dict:
    """H-WMN-13: QLIKE of sigma_base^2 x M vs sigma_base^2 for the next 30 min, weekend bars, per segment (compact arrays)."""
    p = wg.GateParams.load(wd.OUT / "gate_params.json")
    news, eps, mac = gate_inputs()
    btc = ws.load_bars("BTCUSDT")
    rb_df = ws.returns(btc).select("ts", "r")
    acc = {k: [] for k in ("t", "q", "d", "pull", "M")}
    for sym in wd.CORE:
        if sym == "BTCUSDT":
            continue
        df = ws.load_bars(sym)
        if df is None:
            continue
        tot = ws.detect(df, None).select("ts", "r", "sigma", "f")
        ts = tot["ts"].to_numpy()
        r = tot["r"].to_numpy()
        sig = tot["sigma"].to_numpy()
        f = tot["f"].to_numpy()
        del tot
        bj = pl.DataFrame({"ts": ts}).join(pl.read_parquet(wd.OUT / "det" / f"{sym}.parquet", columns=["ts", "beta"]), on="ts", how="left")["beta"].to_numpy().astype(float)
        rb = pl.DataFrame({"ts": ts}).join(rb_df, on="ts", how="left")["r"].fill_null(float("nan")).to_numpy()
        w = _w_daily(ts, r, rb, np.nan_to_num(bj, nan=1.0))
        n = ts.size
        k = np.arange(n - 7)
        cons = (ts[k + 6] - ts[k] == 6 * ws.BAR_MS)
        r2 = np.where(np.isfinite(r), r ** 2, np.nan)
        RV = sum(r2[k + j] for j in range(1, 7))
        Fb = sum((sig[k + 1] * f[k + j]) ** 2 for j in range(1, 7))
        t_or = ts[k] + ws.BAR_MS
        wk = np.zeros(k.size, bool)
        for a_, b_ in _weekend_windows(int(ts[0]), int(ts[-1])):
            i0, i1 = np.searchsorted(t_or, a_), np.searchsorted(t_or, b_)
            wk[i0:i1] = True
        m = cons & wk & np.isfinite(RV) & np.isfinite(Fb) & (Fb > 0)
        if m.sum() == 0:
            continue
        tt = t_or[m]
        lam, pull, _, _, _ = wg.symbol_gate(p, tt, news.get(sym, []), eps.get(sym, []), mac, eps.get("BTCUSDT", []), w[k][m])
        Mv = 1.0 + lam
        rv, fb = RV[m], Fb[m]
        acc["t"].append(tt)
        acc["q"].append((rv / fb).astype(np.float32))
        acc["d"].append(((np.log(fb * Mv) + rv / (fb * Mv)) - (np.log(fb) + rv / fb)).astype(np.float32))
        acc["pull"].append(pull)
        acc["M"].append(Mv.astype(np.float32))
        del ts, r, sig, f, bj, rb, w, RV, Fb, r2
    A = {k: np.concatenate(v) for k, v in acc.items()}
    seg = ws.seg_of(A["t"])
    wkid = ((A["t"] - 4 * ws.DAY_MS) // (7 * ws.DAY_MS)).astype(np.int64)
    res = {}
    rng = np.random.default_rng(13)
    for sg in ("llm_in_sample", "clean_holdout"):
        mm = seg == sg
        q, d, pm, Mv, wks = A["q"][mm].astype(float), A["d"][mm].astype(float), A["pull"][mm], A["M"][mm], wkid[mm]
        uw, inv = np.unique(wks, return_inverse=True)
        per = np.bincount(inv, weights=d) / np.bincount(inv)
        mu, sd = float(per.mean()), float(per.std(ddof=1)) if per.size > 1 else float("nan")
        ratio = float(q[pm].mean() / q[~pm].mean()) if pm.any() else float("nan")
        sq_p = np.bincount(inv, weights=q * pm, minlength=uw.size)
        n_p = np.bincount(inv, weights=pm.astype(float), minlength=uw.size)
        sq_n = np.bincount(inv, weights=q * ~pm, minlength=uw.size)
        n_n = np.bincount(inv, weights=(~pm).astype(float), minlength=uw.size)
        bs = []
        for _ in range(2000):
            j = rng.integers(0, uw.size, uw.size)
            if n_p[j].sum() > 0 and n_n[j].sum() > 0:
                bs.append((sq_p[j].sum() / n_p[j].sum()) / (sq_n[j].sum() / n_n[j].sum()))
        dd = np.array([np.mean(d[(inv == u) & pm]) - 0 if np.any((inv == u) & pm) else np.nan for u in range(0)])  # placeholder
        res[sg] = {"weekends": int(uw.size), "bars": int(mm.sum()), "qlike_diff": mu,
                   "t": mu / (sd / math.sqrt(per.size)) if per.size > 1 and sd > 0 else float("nan"),
                   "share_weekends_improved": float((per < 0).mean()),
                   "pull_share": float(pm.mean()), "ratio_norm": ratio,
                   "ratio_norm_ci90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))] if bs else None,
                   "mean_q_pull": float(q[pm].mean()) if pm.any() else float("nan"), "mean_q_nopull": float(q[~pm].mean()),
                   "mean_M": float(Mv.mean()), "share_M_gt_1_5": float((Mv > 1.5).mean()),
                   "qlike_diff_pull_bars": float(d[pm].mean()) if pm.any() else float("nan"), "qlike_diff_other_bars": float(d[~pm].mean())}
        del dd
    save("gate_eval", res)
    return res


def stage_gate_write(lo: str = "2026-07-03", hi: str | None = None) -> dict:
    """Per-minute gate values for the replayed weekends (Fri 00:00 - Mon 00:00 UTC) of every core symbol + MARKET."""
    p = wg.GateParams.load(wd.OUT / "gate_params.json")
    news, eps, mac = gate_inputs()
    btc = ws.load_bars("BTCUSDT")
    lo_ms = wd.ms(datetime.fromisoformat(lo).replace(tzinfo=UTC))
    hi_ms = wd.ms(datetime.fromisoformat(hi).replace(tzinfo=UTC)) if hi else int(btc["ts"].max()) + ws.BAR_MS
    wins = _weekend_windows(lo_ms, hi_ms)
    frames = []
    rb_df = ws.returns(btc).select("ts", pl.col("r").alias("rb"))
    for sym in wd.CORE:
        df = ws.load_bars(sym)
        if df is None:
            continue
        for a, b in wins:
            t = np.arange(a, b, 60_000, dtype=np.int64)
            if sym == "BTCUSDT":
                continue
            d = ws.returns(df.filter((pl.col("ts") >= a - 8 * ws.DAY_MS) & (pl.col("ts") < b))).join(rb_df, on="ts", how="left")
            if d.height < 500:
                continue
            idio = pl.read_parquet(wd.OUT / "det" / f"{sym}.parquet", columns=["ts", "beta"]).filter((pl.col("ts") >= a - 8 * ws.DAY_MS) & (pl.col("ts") < b))
            dd = d.join(idio, on="ts", how="left")
            wbar = _w_daily(dd["ts"].to_numpy(), dd["r"].fill_null(float("nan")).to_numpy(), dd["rb"].fill_null(float("nan")).to_numpy(),
                            dd["beta"].fill_null(1.0).to_numpy().astype(float))
            j = np.clip(np.searchsorted(dd["ts"].to_numpy(), t, side="right") - 1, 0, dd.height - 1)
            lam, pull, widen, _, _ = wg.symbol_gate(p, t, news.get(sym, []), eps.get(sym, []), mac, eps.get("BTCUSDT", []), wbar[j])
            frames.append(pl.DataFrame({"ts": t, "symbol": sym, "lambda_news": lam.astype(np.float32), "pull": pull, "widen": widen.astype(np.float32)}))
    for a, b in wins:
        t = np.arange(a, b, 60_000, dtype=np.int64)
        lam, pull, widen = wg.market_gate(p, t, mac, eps.get("BTCUSDT", []))
        frames.append(pl.DataFrame({"ts": t, "symbol": "MARKET", "lambda_news": lam.astype(np.float32), "pull": pull, "widen": widen.astype(np.float32)}))
    G = pl.concat(frames).with_columns(pl.from_epoch("ts", "ms").dt.replace_time_zone("UTC").alias("ts")).sort("symbol", "ts")
    G.write_parquet(wd.OUT / "gate.parquet", compression="zstd")
    summ = G.group_by("symbol").agg(pl.col("pull").mean().alias("pull_share"), pl.col("lambda_news").mean().alias("mean_lambda"),
                                    pl.col("lambda_news").max().alias("max_lambda")).sort("pull_share", descending=True)
    out = {"rows": G.height, "weekends": len(wins), "symbols": G["symbol"].n_unique(), "first": str(G["ts"].min()), "last": str(G["ts"].max()),
           "pull_share_all": float(G.filter(pl.col("symbol") != "MARKET")["pull"].mean()), "by_symbol": summ.to_dicts()}
    save("gate_write", out)
    return out


# ------------------------------------------------------------------------------------------------- markouts
def stage_markout_tests() -> dict:
    from ai_trader.research import wmn_markout as wm
    S = wm.load_sample().select("symbol", "day_ms", "segment", "t", "w", "mo60", "mo300", "aligned", "absofi", "vpin", "lsize", "lrv",
                                "news_age", "jump_age", "mac_age", "eff", "mo1", "mo10", "mo1800")
    res = wm.analyse(S)
    # H-WMN-14: pull dummy from the frozen gate at each sampled trade minute
    p = wg.GateParams.load(wd.OUT / "gate_params.json")
    news, eps, mac = gate_inputs()
    parts = []
    for (sym, day), g in S.group_by("symbol", "day_ms"):
        tm = (g["t"].to_numpy() * 1000).astype(np.int64)
        tmin = tm // 60_000 * 60_000
        lam, pull, widen, _, _ = wg.symbol_gate(p, tmin, news.get(sym, []), eps.get(sym, []), mac, eps.get("BTCUSDT", []), 0.5)
        parts.append(g.select("symbol", "day_ms", "segment", "w", "mo60", "mo300", "aligned", "absofi", "vpin", "lsize", "lrv")
                     .with_columns(pl.Series("pull", pull.astype(np.float32))))
    del S
    S2 = pl.concat(parts)
    h14 = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        s = S2.filter(pl.col("segment") == seg)
        h14[seg] = {f"h{h}": wm.regress(s, ["pull"], h) for h in (60, 300)}
        h14[seg]["pull_share"] = float(s["pull"].mean()) if s.height else float("nan")
    res["H-WMN-14"] = h14
    res["H-WMN-14_note"] = "w fixed at 0.5 for trade-level gate values (market share not recomputed per trade day)"
    save("markouts", res)
    return res


def stage_live(as_of: datetime | None = None) -> dict:
    """Current weekend news state: events known in the last 72 h, gate values now (frozen parameters)."""
    as_of = as_of or datetime.now(UTC)
    t = wd.ms(as_of)
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    # fresh announcements (first pages only, in memory; events.parquet is not changed)
    try:
        inst = pl.read_parquet(wd.OUT / "instruments.parquet")
        by = (wd.get("https://api.bybit.com/v5/announcements/index", {"locale": "en-US", "limit": 100, "page": 1}) or {}).get("result", {}).get("list", [])
        ok = []
        for ty in wd.OKX_TYPES:
            d = wd.get("https://www.okx.com/api/v5/support/announcements", {"annType": ty, "page": 1}, host_gap=0.5)
            ok.extend(((d or {}).get("data") or [{}])[0].get("details") or [])
        ku = wd.fetch_kucoin_ann(start=as_of - timedelta(days=4), log=lambda *a, **k: None)
        fresh = wd.build_events(by, ok, ku, [], {}, [], {}, inst, None)
        ev = pl.concat([ev, fresh.select(ev.columns)], how="vertical").unique("event_id", keep="first")
        news_fresh = ws.precise_token_events(fresh)
        n_fresh = fresh.filter(pl.col("known_ts") > int(pl.read_parquet(wd.OUT / "events.parquet")["known_ts"].max())).height
    except Exception as e:  # noqa: BLE001
        news_fresh, n_fresh = None, f"failed: {e}"
    recent = ev.filter((pl.col("known_ts") <= t) & (pl.col("known_ts") >= t - 72 * 3_600_000)).sort("known_ts")
    up = ev.filter(pl.col("occur_ts") > t).filter(pl.col("occur_ts") <= t + 72 * 3_600_000)
    p = wg.GateParams.load(wd.OUT / "gate_params.json")
    news, eps, mac = gate_inputs()
    if news_fresh is not None:
        for sym, g in news_fresh.group_by("symbol"):
            have = {k for k, _ in news.get(sym[0], [])}
            news.setdefault(sym[0], []).extend((k, gg) for k, gg in zip(g["known_ts"].to_list(), g["group"].to_list(), strict=True) if k not in have)
    now_vals = {}
    for sym in wd.CORE:
        if sym == "BTCUSDT":
            continue
        lam, pull, widen, lt, lm = wg.symbol_gate(p, np.array([t]), news.get(sym, []), eps.get(sym, []), mac, eps.get("BTCUSDT", []), 0.5)
        now_vals[sym] = {"lambda_news": float(lam[0]), "pull": bool(pull[0]), "widen": float(widen[0]), "lam_tok": float(lt[0]), "lam_mkt": float(lm[0])}
    E = pl.read_parquet(wd.OUT / "episodes.parquet").filter((pl.col("flag") == "D") & (pl.col("known_ts") >= t - 72 * 3_600_000))
    last_bar = {s_: int(pl.read_parquet(wd.BARS / f"{s_}.parquet", columns=["ts"])["ts"].max()) for s_ in ("BTCUSDT", "ATOMUSDT")}
    out = {"as_of": as_of.isoformat(timespec="seconds"), "fresh_new_events": n_fresh, "last_bar_ms": last_bar, "recent_events": recent.select("known_dt", "source", "category", "scope", "symbols", "title").to_dicts(),
           "scheduled_next_72h": up.select("source", "category", "symbols", "occur_ts", "title").to_dicts(), "gate_now": now_vals,
           "episodes_72h": E.select("symbol", "ts", "z", "vr").to_dicts(), "macro_next": [m for m in mac if t <= m <= t + 7 * ws.DAY_MS]}
    save("live", out)
    return out


def _market_paired(M: pl.DataFrame, group: str, series: str, metric: str) -> dict:
    s = M.filter((pl.col("group") == group) & (pl.col("series") == series))
    ev = s.filter(~pl.col("placebo"))
    pl_ = s.filter(pl.col("placebo")).select("known_ts", pl.col(metric).alias("p"))
    j = ev.join(pl_, on="known_ts", how="inner").with_columns((pl.col(metric) - pl.col("p")).alias("diff"), seg_col("known_ts"))
    out = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        q = j.filter(pl.col("segment") == seg)
        mu, se, n, G = ws.cluster_mean(q["diff"].to_numpy().astype(float), ws._cluster_day(q["anchor"].to_numpy()))
        out[seg] = {"diff": mu, "se": se, "t": mu / se if se and se > 0 else float("nan"), "n": n, "clusters": G,
                    "event_mean": float(q[metric].mean()) if q.height else float("nan"), "placebo_mean": float(q["p"].mean()) if q.height else float("nan")}
    return out


def stage_market_tests() -> dict:
    M = pl.read_parquet(wd.OUT / "es_market.parquet")
    res = {"macro_BTC": _market_paired(M, "macro_release", "BTC", "ARV_h0_1")}
    sec = []
    for grp, ser, met in (("macro_release", "EW", "ARV_h0_1"), ("hn_burst", "BTC", "ARV_h0_1"), ("hn_burst", "EW", "ARV_h0_1"),
                          ("hack_100m", "BTC", "ARV_h0_24"), ("hack_100m", "EW", "ARV_h0_24")):
        r = _market_paired(M, grp, ser, met)
        sec.append({"group": grp, "series": ser, "metric": met, **{f"{a}_{k}": v for a in r for k, v in r[a].items()}})
    pv = [float(1 - 0.5 * (1 + math.erf(x["llm_in_sample_t"] / math.sqrt(2)))) if np.isfinite(x["llm_in_sample_t"]) else 1.0 for x in sec]
    rej = ws.bh(pv, 0.10)
    for x, p_, r_ in zip(sec, pv, rej, strict=True):
        x["p_one"], x["bh"] = p_, r_
    res["secondary"] = sec
    res["macro_extra"] = {m: _market_paired(M, "macro_release", "BTC", m) for m in ("ARV_pre", "ARV_h1_4", "ARV_h4_24", "AV_h0_1")}
    save("market_tests", res)
    return res


def stage_ts_alignment() -> dict:
    """Descriptive: for Bybit announcements on core symbols whose publishTime and dateTimestamp differ by > 1 h, the
    abnormal variance / volume in the first hour after each of the two timestamps (which one the market reacts to)."""
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = ws.precise_token_events(ev).filter(pl.col("source") == "bybit_ann")
    core = ws.load_core_series()
    rows = []
    for sym, extra, grp in pt.select("symbol", "extra", "group").iter_rows():
        if sym not in core or sym == "BTCUSDT":
            continue
        x = json.loads(extra)
        tp, tdt = int(x.get("ts_publish") or 0), int(x.get("ts_date") or 0)
        if not tp or not tdt:
            continue
        gap_h = (tp - tdt) / 3.6e6
        r = {"symbol": sym, "group": grp, "gap_h": gap_h}
        for name, t0 in (("pub", tp), ("date", tdt)):
            w = core[sym].window(t0 - 50 * 3_600_000, t0 + 50 * 3_600_000)
            m = ws.event_metrics(w, t0) if w is not None else None
            r[f"ARV1_{name}"] = m["ARV_h0_1"] if m else float("nan")
            r[f"AV1_{name}"] = m["AV_h0_1"] if m else float("nan")
        rows.append(r)
    T = pl.DataFrame(rows)
    out = {}
    for lab, cond in (("agree_le_1h", pl.col("gap_h").abs() <= 1), ("pub_later_1_24h", (pl.col("gap_h") > 1) & (pl.col("gap_h") <= 24)),
                      ("pub_later_gt_24h", pl.col("gap_h") > 24), ("pub_earlier_gt_1h", pl.col("gap_h") < -1)):
        q = T.filter(cond)
        out[lab] = {"n": q.height, **{c: float(np.nanmean(q[c].to_numpy())) if q.height else float("nan") for c in ("ARV1_pub", "ARV1_date", "AV1_pub", "AV1_date")},
                    "n_finite_pub": int(np.isfinite(q["ARV1_pub"].to_numpy()).sum()) if q.height else 0}
    save("ts_alignment", out)
    return out


def _f(x, nd=3):
    try:
        return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def stage_tables() -> str:
    """Markdown tables of every result file (printed; pasted into the report)."""
    L = []
    if (RES / "event_tests.json").is_file():
        et = load("event_tests")
        L.append("### H-WMN-3 / H-WMN-4 (event - placebo, date-clustered)\n")
        L.append("| group | metric | in: n | in: event | in: placebo | in: diff | in: t | BH | hold: n | hold: diff | hold: t |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        bhm = {(r["group"], r["metric"]): r["bh"] for r in et["H-WMN-4"]["bh_family"]}
        for r in et["H-WMN-4"]["table"]:
            L.append(f"| {r['group']} | {r['metric']} | {r['llm_in_sample_n']} | {_f(r['llm_in_sample_event_mean'])} | {_f(r['llm_in_sample_placebo_mean'])} | "
                     f"{_f(r['llm_in_sample_diff'])} | {_f(r['llm_in_sample_t'], 2)} | {bhm.get((r['group'], r['metric']), '-')} | {r['clean_holdout_n']} | "
                     f"{_f(r['clean_holdout_diff'])} | {_f(r['clean_holdout_t'], 2)} |")
        h3 = et["H-WMN-3"]
        L.append("\nPooled precise token news, ARV[0,1h): " + "; ".join(f"{k}: diff {_f(v['diff'])} (t {_f(v['t'], 2)}, n {v['n']}, clusters {v['clusters']})" for k, v in h3.items()))
        for m, v in et["H-WMN-3_extra"].items():
            L.append(f"- {m}: " + "; ".join(f"{k} event {_f(x['event_mean'])} placebo {_f(x['placebo_mean'])} diff {_f(x['diff'])} (t {_f(x['t'], 2)})" for k, x in v.items()))
        L.append("\nH-WMN-5: " + json.dumps(et["H-WMN-5"], default=str))
        L.append("\nH-WMN-6: " + json.dumps(et["H-WMN-6"], default=str))
    for name in ("detector_tests", "market_tests", "decay", "hawkes", "markouts", "gate_eval", "gate_write", "ts_alignment"):
        if (RES / f"{name}.json").is_file():
            txt = json.dumps(load(name), default=str)
            L.append(f"\n### {name}\n\n" + txt[:6000])
    out = "\n".join(L)
    print(out)
    return "printed"


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    t0 = time.time()
    st = argv[0]
    fn = {"detect": lambda: ws.run_detector(), "es_token": lambda: ws.run_event_study(), "es_episodes": lambda: ws.run_episode_study(),
          "es_market": lambda: ws.run_market_study(), "tests": stage_tests, "det_tests": stage_detector_tests, "decay": stage_decay,
          "gate_fit": stage_gate_fit, "gate_eval": stage_gate_eval, "gate_write": stage_gate_write, "markout_tests": stage_markout_tests,
          "live": stage_live, "market_tests": stage_market_tests, "ts_alignment": stage_ts_alignment, "tables": stage_tables}
    if st == "hawkes":
        from ai_trader.research import wmn_hawkes as wh
        r = wh.compare(wh.inputs())
        save("hawkes", r)
    elif st == "markouts":
        from ai_trader.research import wmn_markout as wm
        r = wm.run()
    else:
        r = fn[st]()
    print(json.dumps({"stage": st, "secs": round(time.time() - t0, 1), "result": r if st in ("detect", "es_token", "es_episodes", "es_market", "markouts", "gate_fit", "gate_write") else "saved"},
                     default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
