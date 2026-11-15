"""H-DOW1-3 (Monday effect) and H-RL1-3 (regime cells, vol states, regime lifecycle) on crypto and TradFi (registered 2026-11-08).

    PYTHONIOENCODING=utf-8 uv run python scripts/dow_regime_study.py [--record]

Daily data: BTC / ETH / equal-weight alt basket from the lake (1h -> UTC days); S&P 500, Nasdaq-100, gold, WTI, TLT, dollar
index from Yahoo daily (2005+). z_t = r_t / sigma_t|t-1 (EWMA 0.94 of prior days). Regimes: the v0 9-cell labeller on
daily bars (horizons 1/3/10 bars, vol quantiles over 180 bars, 3-bar hysteresis). Forward window: 7 days crypto,
5 trading days TradFi. Segments by date vs the knowledge cutoff. Verdict rules are the registered ones, coded below.
Data note: WTI's negative-price days (Apr 2020) are dropped (log returns undefined).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from regime_context import ALTS_OLD, daily_closes  # noqa: E402

from ai_trader.core.time import now  # noqa: E402
from ai_trader.research.frame import enrich_bars  # noqa: E402
from ai_trader.research.regimes import label  # noqa: E402
from ai_trader.storage.lake import Lake  # noqa: E402

warnings.filterwarnings("ignore")
CUTOFF = date(2026, 7, 1)
CRYPTO = ["BTC", "ETH", "ALTS"]
TRADFI = {"SPX": "^GSPC", "NDX": "^NDX", "GOLD": "GC=F", "WTI": "CL=F", "TLT": "TLT", "DXY": "DX-Y.NYB"}
LAM = 0.94


def yahoo(sym: str) -> pl.DataFrame:
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=1104537600&period2=9999999999&interval=1d"
    d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["chart"]["result"][0]
    q = d["indicators"]["quote"][0]
    adj = (d["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
    df = pl.DataFrame({"d": [datetime.fromtimestamp(x, UTC).date() for x in d["timestamp"]], "open": q["open"], "close_raw": q["close"], "close": adj},
                      schema_overrides={"open": pl.Float64, "close_raw": pl.Float64, "close": pl.Float64})
    return df.drop_nulls().filter((pl.col("close") > 0) & (pl.col("open") > 0)).unique("d", keep="last").sort("d")


def ewma_sigma(r: np.ndarray) -> np.ndarray:
    """sigma_t|t-1 from returns strictly before t."""
    s2 = np.full(len(r), np.nan)
    v = np.nanvar(r[1:31]) if len(r) > 31 else np.nanvar(r)
    for i in range(1, len(r)):
        prev = r[i - 1]
        if np.isfinite(prev):
            v = LAM * v + (1 - LAM) * prev * prev
        s2[i] = v
    return np.sqrt(s2)


def build(t) -> dict[str, pl.DataFrame]:
    lake = Lake()
    out = {}
    start = datetime(2018, 1, 1, tzinfo=UTC)
    for name, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        x = daily_closes(lake, sym, t, start).filter(pl.col("n") >= 20).select("d", "close")
        out[name] = x
    # equal-weight alt basket: mean of daily log returns over available alts (>= 5), cumulated into a price
    rets = []
    for s in ALTS_OLD:
        if s == "ETHUSDT":
            continue
        x = daily_closes(lake, s, t, datetime(2020, 9, 1, tzinfo=UTC)).filter(pl.col("n") >= 20).select("d", pl.col("close").log().diff().alias(s))
        rets.append(x)
    a = rets[0]
    for x in rets[1:]:
        a = a.join(x, on="d", how="full", coalesce=True)
    cols = [c for c in a.columns if c != "d"]
    a = a.sort("d").with_columns(pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in cols]).alias("k"), pl.mean_horizontal(cols).alias("r"))
    a = a.filter(pl.col("k") >= 5).drop_nulls("r").with_columns(pl.col("r").cum_sum().exp().alias("close")).select("d", "close")
    out["ALTS"] = a
    for name, sym in TRADFI.items():
        out[name] = yahoo(sym)
    return out


def prep(df: pl.DataFrame, crypto: bool) -> pl.DataFrame:
    df = df.sort("d")
    c = df["close"].to_numpy()
    r = np.r_[np.nan, np.diff(np.log(c))]
    sig = ewma_sigma(r)
    df = df.with_columns(pl.Series("r", r), pl.Series("sig", sig)).with_columns((pl.col("r") / pl.col("sig")).alias("z"), pl.col("d").dt.weekday().alias("wd"))
    n = 7 if crypto else 5
    pw = df["r"].rolling_sum(n).shift(1).to_numpy() / (sig * math.sqrt(n))
    return df.with_columns(pl.Series("prior_week_z", pw)).drop_nulls(["z"]).filter(pl.col("z").is_finite())


def hac(y, X, lags=5):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def block_ratio(a: np.ndarray, mask: np.ndarray, base: np.ndarray, reps=1000, block=14, seed=0) -> tuple[float, float, float]:
    """E[a | mask] / E[a | base] with a block bootstrap over time."""
    rng = np.random.default_rng(seed)
    n = len(a)
    starts = np.arange(0, n, block)
    est = a[mask].mean() / a[base].mean()
    bs = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, min(n, s + block)) for s in rng.choice(starts, size=len(starts))])
        mm, bb = mask[idx], base[idx]
        if mm.sum() and bb.sum():
            bs.append(a[idx][mm].mean() / a[idx][bb].mean())
    return float(est), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def dow_tests(data: dict[str, pl.DataFrame]) -> dict:
    res = {"DOW1": {}, "DOW2": {}, "DOW3": {}}
    pooled = {"crypto": [], "tradfi": []}
    for name, raw in data.items():
        crypto = name in CRYPTO
        df = prep(raw, crypto)
        for seg, part in [("llm_in_sample", df.filter(pl.col("d") < CUTOFF)), ("clean_holdout", df.filter(pl.col("d") >= CUTOFF))]:
            if part.height < 40:
                continue
            z, wd = part["z"].to_numpy(), part["wd"].to_numpy()
            mon = (wd == 1).astype(float)
            m1 = hac(z, mon)
            weekday = (wd >= 1) & (wd <= 5)
            # DOW1: Monday vs all other days (crypto incl. weekend), and vs Tue-Fri
            sub = weekday
            m1b = hac(z[sub], mon[sub])
            z2 = z * z
            r_wk = block_ratio(z2, wd == 1, (wd >= 2) & (wd <= 5))
            r_we = block_ratio(z2, wd == 1, wd >= 6) if crypto else (float("nan"),) * 3
            res["DOW1"].setdefault(seg, {})[name] = {"n_mon": int(mon.sum()), "mean_z_mon": float(z[wd == 1].mean()), "mean_z_rest": float(z[wd != 1].mean()),
                                                      "coef": float(m1.params[1]), "t": float(m1.tvalues[1]), "p": float(m1.pvalues[1]),
                                                      "coef_vs_tue_fri": float(m1b.params[1]), "t_vs_tue_fri": float(m1b.tvalues[1])}
            m2 = hac(z2[sub], mon[sub])
            res["DOW2"].setdefault(seg, {})[name] = {"ratio_vs_tue_fri": r_wk[0], "ci90": [r_wk[1], r_wk[2]], "t_z2": float(m2.tvalues[1]),
                                                      "ratio_vs_weekend": r_we[0], "ci90_weekend": [r_we[1], r_we[2]]}
            # DOW3: Monday z conditional on prior week
            pm = part.filter(pl.col("wd") == 1).drop_nulls("prior_week_z")
            if pm.height > 30:
                pw = pm["prior_week_z"].to_numpy()
                X = np.column_stack([(pw < -1).astype(float), (pw > 1).astype(float)])
                m3 = hac(pm["z"].to_numpy(), X, lags=2)
                res["DOW3"].setdefault(seg, {})[name] = {"n": pm.height, "n_down": int((pw < -1).sum()), "coef_down": float(m3.params[1]), "t_down": float(m3.tvalues[1]),
                                                          "coef_up": float(m3.params[2]), "t_up": float(m3.tvalues[2]), "mean_z_after_down": float(pm["z"].to_numpy()[pw < -1].mean()) if (pw < -1).any() else None}
                pooled["crypto" if crypto else "tradfi"].append(pm.with_columns(pl.lit(name).alias("asset"), pl.lit(seg).alias("seg")).select("d", "asset", "seg", "z", "prior_week_z"))
        # Monday gap vs intraday for US indices
        if name in ("SPX", "NDX"):
            g = raw.sort("d").with_columns((pl.col("open") / pl.col("close_raw").shift(1)).log().alias("gap"), (pl.col("close_raw") / pl.col("open")).log().alias("intra"),
                                           pl.col("d").dt.weekday().alias("wd")).drop_nulls()
            g = g.filter(pl.col("d") >= date(2010, 1, 1))  # Yahoo opens for indices are unreliable earlier
            share = g.group_by(pl.col("wd") == 1).agg((pl.col("gap") ** 2).mean().alias("gap_var"), (pl.col("intra") ** 2).mean().alias("intra_var")).sort("wd")
            res.setdefault("gap", {})[name] = {("monday" if r["wd"] else "tue_fri"): {"gap_var_share": r["gap_var"] / (r["gap_var"] + r["intra_var"]),
                                               "gap_vol_bp": math.sqrt(r["gap_var"]) * 1e4, "intra_vol_bp": math.sqrt(r["intra_var"]) * 1e4} for r in share.iter_rows(named=True)}
    for k, lst in pooled.items():
        if not lst:
            continue
        P = pl.concat(lst)
        for seg in ["llm_in_sample", "clean_holdout"]:
            q = P.filter(pl.col("seg") == seg)
            if q.height < 30:
                continue
            pw = q["prior_week_z"].to_numpy()
            X = sm.add_constant(np.column_stack([(pw < -1).astype(float), (pw > 1).astype(float)]))
            groups = pl.Series(q["d"]).rank("dense").to_numpy().astype(int)
            m = sm.OLS(q["z"].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
            res["DOW3"].setdefault(seg, {})[f"pooled_{k}"] = {"n": q.height, "coef_down": float(m.params[1]), "t_down": float(m.tvalues[1]), "coef_up": float(m.params[2]), "t_up": float(m.tvalues[2])}
    return res


def regime_frames(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for name, raw in data.items():
        crypto = name in CRYPTO
        h = 7 if crypto else 5
        df = raw.sort("d").with_columns(pl.col("d").cast(pl.Datetime("us", "UTC")).alias("event_ts"))
        df = df.with_columns((pl.col("event_ts") + timedelta(days=1)).alias("close_ts"))
        f = label(enrich_bars(df.select("event_ts", "close_ts", "close")), 24.0).drop_nulls("cell")
        lc = np.log(f["close"].to_numpy())
        r = np.r_[np.nan, np.diff(lc)]
        sig = f["sigma_next"].to_numpy()
        n = len(lc)
        zf = np.full(n, np.nan)
        lrv = np.full(n, np.nan)
        for i in range(n - h):
            zf[i] = (lc[i + h] - lc[i]) / (sig[i] * math.sqrt(h))
            seg = r[i + 1:i + h + 1]
            lrv[i] = math.log(math.sqrt(np.mean(seg * seg)) / sig[i]) if np.all(np.isfinite(seg)) and sig[i] > 0 else np.nan
        cells = f["cell"].to_list()
        ep, age, end5 = np.zeros(n, int), np.zeros(n, int), np.full(n, np.nan)
        e, a = 0, 0
        for i in range(n):
            if i and cells[i] != cells[i - 1]:
                e += 1
                a = 0
            a += 1
            ep[i], age[i] = e, a
        for i in range(n - 5):
            end5[i] = float(any(cells[j] != cells[i] for j in range(i + 1, i + 6)))
        rows.append(f.select("event_ts", "cell", "dir_state", "vol_state").with_columns(
            pl.lit(name).alias("asset"), pl.lit("crypto" if crypto else "tradfi").alias("cls"),
            pl.Series("z_fwd", zf), pl.Series("lrv_fwd", lrv), pl.Series("episode", [f"{name}-{x}" for x in ep]), pl.Series("age", age), pl.Series("end5", end5),
            pl.col("event_ts").dt.date().alias("d")))
    return pl.concat(rows)


def clustered(y, X, groups, logit=False):
    X = sm.add_constant(X, has_constant="add")
    g = pl.Series(groups).rank("dense").to_numpy().astype(int)
    if logit:
        return sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})


def regime_tests(R: pl.DataFrame) -> dict:
    res = {"RL1": {}, "RL2": {}, "RL3": {}, "desc": {}}
    for seg, part in [("llm_in_sample", R.filter(pl.col("d") < CUTOFF - timedelta(days=7))), ("clean_holdout", R.filter(pl.col("d") >= CUTOFF))]:
        for scope, q in [("pooled", part), ("crypto", part.filter(pl.col("cls") == "crypto")), ("tradfi", part.filter(pl.col("cls") == "tradfi"))]:
            a = q.drop_nulls("z_fwd").filter(pl.col("z_fwd").is_finite())
            if a.height < 100:
                continue
            up, dn = (a["dir_state"] == "up").cast(float).to_numpy(), (a["dir_state"] == "down").cast(float).to_numpy()
            m = clustered(a["z_fwd"].to_numpy(), np.column_stack([up, dn]), a["episode"].to_list())
            diff = m.params[1] - m.params[2]
            cv = m.cov_params()
            se = math.sqrt(cv[1, 1] + cv[2, 2] - 2 * cv[1, 2])
            res["RL1"].setdefault(seg, {})[scope] = {"n": a.height, "up": float(m.params[1]), "down": float(m.params[2]), "up_minus_down": float(diff), "t": float(diff / se),
                                                     "mean_z_by_cell": {r["cell"]: round(r["z"], 3) for r in a.group_by("cell").agg(pl.col("z_fwd").mean().alias("z")).sort("cell").iter_rows(named=True)}}
            b = q.drop_nulls("lrv_fwd").filter(pl.col("lrv_fwd").is_finite())
            hi, lo = (b["vol_state"] == "high").cast(float).to_numpy(), (b["vol_state"] == "low").cast(float).to_numpy()
            m2 = clustered(b["lrv_fwd"].to_numpy(), np.column_stack([hi, lo]), b["episode"].to_list())
            cv2 = m2.cov_params()
            d2 = m2.params[1] - m2.params[2]
            res["RL2"].setdefault(seg, {})[scope] = {"n": b.height, "high_minus_low": float(d2), "t": float(d2 / math.sqrt(cv2[1, 1] + cv2[2, 2] - 2 * cv2[1, 2])),
                                                     "mean_log_rv_ratio": {s: round(v, 3) for s, v in b.group_by("vol_state").agg(pl.col("lrv_fwd").mean()).sort("vol_state").iter_rows()}}
            c = q.filter(pl.col("end5").is_not_null() & pl.col("end5").is_not_nan())
            dums = np.column_stack([(c["cell"] == k).cast(float).to_numpy() for k in sorted(set(c["cell"].to_list()))[1:]])
            X = np.column_stack([np.log(c["age"].to_numpy()), dums])
            try:
                m3 = clustered(c["end5"].to_numpy(), X, c["episode"].to_list(), logit=True)
                res["RL3"].setdefault(seg, {})[scope] = {"n": c.height, "age_coef": float(m3.params[1]), "z": float(m3.tvalues[1])}
            except Exception as e:  # noqa: BLE001
                res["RL3"].setdefault(seg, {})[scope] = {"error": str(e)[:120]}
    # descriptive lifecycle (all data)
    ep = R.group_by(["asset", "episode", "cell"]).agg(pl.len().alias("dur"))
    res["desc"]["duration_by_cell"] = {r["cell"]: {"median": r["med"], "mean": round(r["mean"], 1), "n_episodes": r["n"]} for r in
                                       ep.group_by("cell").agg(pl.col("dur").median().alias("med"), pl.col("dur").mean().alias("mean"), pl.len().alias("n")).sort("cell").iter_rows(named=True)}
    tr = R.sort(["asset", "event_ts"]).with_columns(pl.col("cell").shift(-1).over("asset").alias("nxt")).drop_nulls("nxt")
    res["desc"]["stay_prob_daily"] = {r["cell"]: round(r["p"], 3) for r in tr.group_by("cell").agg((pl.col("nxt") == pl.col("cell")).mean().alias("p")).sort("cell").iter_rows(named=True)}
    ends = tr.filter(pl.col("nxt") != pl.col("cell"))
    res["desc"]["next_cell_after_exit"] = {}
    for cell in sorted(set(ends["cell"].to_list())):
        e = ends.filter(pl.col("cell") == cell)
        res["desc"]["next_cell_after_exit"][cell] = {r["nxt"]: round(r["len"] / e.height, 2) for r in e.group_by("nxt").len().sort("len", descending=True).head(3).iter_rows(named=True)}
    cur = R.sort("event_ts").group_by("asset").agg(pl.col("cell").last(), pl.col("age").last(), pl.col("d").last())
    res["desc"]["current"] = {r["asset"]: {"cell": r["cell"], "age_days": r["age"], "as_of": str(r["d"])} for r in cur.iter_rows(named=True)}
    return res


def verdicts(dw: dict, rg: dict) -> list[tuple[str, str, str, dict]]:
    out = []
    # DOW1: Holm across assets in-sample; holdout: same sign for the passing assets
    ins = dw["DOW1"].get("llm_in_sample", {})
    adj = holm({k: v["p"] for k, v in ins.items()})
    passing = [k for k, p in adj.items() if p < 0.05]
    out.append(("H-DOW1", "llm_in_sample", "supports" if passing else "contradicts", {"holm_p": adj, "assets": ins}))
    ho = dw["DOW1"].get("clean_holdout", {})
    same = [k for k in passing if k in ho and np.sign(ho[k]["coef"]) == np.sign(ins[k]["coef"])]
    out.append(("H-DOW1", "clean_holdout", "supports" if (passing and len(same) == len(passing)) else ("inconclusive" if not passing else "contradicts"), {"assets": ho, "passing_in_sample": passing}))
    # DOW2: primary assets BTC and SPX
    for seg in ["llm_in_sample", "clean_holdout"]:
        d = dw["DOW2"].get(seg, {})
        ok = [a for a in ("BTC", "SPX") if a in d and ((d[a]["ci90"][0] > 1) if seg == "llm_in_sample" else (d[a]["ratio_vs_tue_fri"] > 1))]
        out.append(("H-DOW2", seg, "supports" if len(ok) == 2 else ("inconclusive" if ok else "contradicts"), {"primary_ok": ok, "assets": d}))
    # DOW3: pooled |t| > 2 in-sample, same sign holdout
    for seg in ["llm_in_sample", "clean_holdout"]:
        d = dw["DOW3"].get(seg, {})
        pc = {k: v for k, v in d.items() if k.startswith("pooled")}
        if seg == "llm_in_sample":
            ok = any(abs(v["t_down"]) > 2 for v in pc.values())
        else:
            ins3 = dw["DOW3"].get("llm_in_sample", {})
            ok = any(abs(ins3.get(k, {}).get("t_down", 0)) > 2 and np.sign(v["coef_down"]) == np.sign(ins3[k]["coef_down"]) for k, v in pc.items())
        verdict = ("inconclusive" if not pc else ("supports" if ok else "contradicts"))
        out.append(("H-DOW3", seg, verdict, d or {"note": "too few Mondays after a < -1 sigma week in this segment"}))
    # RL1 / RL2 / RL3 on the pooled scope
    for hid, key, cond_in, cond_ho in [
        ("H-RL1", "RL1", lambda v: v["up_minus_down"] > 0 and v["t"] > 2, lambda v: v["up_minus_down"] > 0),
        ("H-RL2", "RL2", lambda v: v["high_minus_low"] < 0 and v["t"] < -2, lambda v: v["high_minus_low"] < 0),
        ("H-RL3", "RL3", lambda v: v.get("age_coef", -1) > 0 and v.get("z", 0) > 2, lambda v: v.get("age_coef", -1) > 0),
    ]:
        for seg, cond in [("llm_in_sample", cond_in), ("clean_holdout", cond_ho)]:
            v = rg[key].get(seg, {}).get("pooled")
            if v is None or "error" in v:
                if v is not None:
                    out.append((hid, seg, "inconclusive", rg[key][seg]))
                continue
            out.append((hid, seg, "supports" if cond(v) else "contradicts", {**rg[key][seg], **({"lifecycle": rg["desc"]} if hid == "H-RL3" and seg == "llm_in_sample" else {})}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    t = now()
    data = build(t)
    for k, v in data.items():
        print(f"{k:<5} {v['d'][0]} .. {v['d'][-1]}  n={v.height}")
    dw = dow_tests(data)
    R = regime_frames(data)
    rg = regime_tests(R)
    out = Path(__file__).resolve().parents[1] / "research" / "reports" / "dow_regime_2026-11-08.json"
    out.write_text(json.dumps({"dow": dw, "regime": rg}, indent=1, default=str))
    print(json.dumps({"dow": dw, "regime": rg}, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else str(x))[:20000])
    v = verdicts(dw, rg)
    print("\nVERDICTS")
    for hid, seg, verdict, _ in v:
        print(f"  {hid:<7} {seg:<14} {verdict}")
    if a.record:
        from ai_trader.research.hypotheses import record
        for hid, seg, verdict, m in v:
            record(hid, seg, verdict, json.loads(json.dumps(m, default=str)), data="crypto: lake 1h -> UTC days; TradFi: Yahoo daily 2005+; scripts/dow_regime_study.py")
        print("recorded")


if __name__ == "__main__":
    main()
