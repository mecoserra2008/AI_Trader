"""Microstructure pilot, trade-level laws: H-MS1, H-MS2, H-MS3, H-MS4, H-MS9 (research/hypotheses.yaml, block
"Microstructure pilot"). Agent C1. Fitted laws live in ai_trader.sim.laws; every fitted parameter per symbol and half is
written to data/models/microstructure_laws/params_2026-11-10.json (+ a tidy parquet) for the population / simulator work.

Data: lake table trades_raw (every Bybit linear print of 2026-11-10 UTC; 51 symbols = config/mm_multi.yaml + BTCUSDT),
read with DuckDB over data/lake/trades_raw/**/*.parquet (hive partitions, date = 2026-11-10 only; BTC has other dates).
All of it is after the knowledge cutoff: every result is segment clean_holdout. FIT = 00:00-12:00 UTC, TEST = 12:00-24:00
UTC (contains the 13:30 UTC CPI release; a robustness row drops 13:00-16:00, the registered verdict uses the full TEST half).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.microstructure_laws_study --stage laws|aft|report|all [--record]

Choices fixed BEFORE any fit (the registry leaves them open), see METHOD:

* Prints vs orders. Bybit prints one aggressive order as several trades with the same millisecond and taker side
  (BTC 2.10M prints -> 0.60M same-ms same-side groups; up to 825 prints per group). ORDER = maximal run of consecutive
  prints (archive order) with the same ms and the same side; ARRIVAL EVENT = distinct ms (merges the ~1% of ms that
  carry both sides), so event durations are >= 1 ms and continuous laws apply. Primary level: H-MS1 = arrival events,
  H-MS3 = orders, H-MS2 = prints (the registered 'trade size'); the other level is always reported. Print-level
  durations of 0 ms (prints of one order) are set to 0.5 ms (half the clock resolution).
* H-MS2 'index alpha in [1, 2]' is the TAIL index zeta of the CCDF, P(X > x) ~ x^-zeta (the ~1.5 of Gopikrishnan et al.
  and Gabaix et al. in the rationale); the CSN pdf exponent is alpha_pdf = zeta + 1 and is reported too.
* Deseasonalisation: causal trailing 30-minute mean duration s_i (a one-day sample has no out-of-sample time-of-day
  profile; flow_3s for 2026-11-10 starts near 12:00). x_i = d_i / s_i; the first 30 minutes of FIT are warm-up.
"""

from __future__ import annotations

import argparse
import json
import warnings
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize, stats

from ai_trader.sim import laws as L

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\c--Users-Asus-Desktop-AI-Trader\31faef80-e320-4ca6-ab63-04fa013eeb03\scratchpad\c1")
CACHE = SCRATCH / "sym"
BOOKS = SCRATCH / "book"
DATE = "2026-11-10"
DAY0 = 1_794_268_800_000  # 2026-11-10 00:00 UTC in ms
HALF = DAY0 + 43_200_000
DAY1 = DAY0 + 86_400_000
CPI = (DAY0 + 13 * 3_600_000, DAY0 + 16 * 3_600_000)
WARMUP_MS = 1_800_000
SEED = 20261111
WORKERS = 4
B_CSN = 200
B_CSN_SECONDARY = 50
B_BOOT = 2000
MAX_LAG = 1000
MIN_TRADES_MS3 = 20_000
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"}
TRADFI = {"CLUSDT", "BZUSDT", "XAUUSDT", "XAGUSDT", "QQQUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT", "CRWVUSDT", "GOOGLUSDT"}
L2_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "HYPEUSDT", "CARVUSDT", "KASUSDT", "ATOMUSDT", "NVDAUSDT", "QQQUSDT", "XAUUSDT"}
PARAMS_PATH = ROOT / "data" / "models" / "microstructure_laws" / f"params_{DATE}.json"
AFT_DIR = ROOT / "data" / "models" / "microstructure_laws" / f"aft_{DATE}"
REPORT = ROOT / "research" / "reports" / "microstructure_laws_2026-11-11"

METHOD = {
    "data": "trades_raw 2026-11-10 (Bybit linear public archive): event_ts (ms), side (taker), price, size; 51 symbols; DuckDB read of the "
            "lake parquet, archive row order kept within a millisecond (file_row_number)",
    "split": "FIT = durations / prints with start and end in [00:00, 12:00) UTC; TEST = [12:00, 24:00); robustness TEST-exCPI drops durations or "
             "prints starting in [13:00, 16:00); registered verdicts use the full TEST half",
    "levels": "order = maximal run of consecutive prints with the same ms and taker side; arrival event = distinct ms; primary levels: H-MS1 "
              "events, H-MS2 prints, H-MS3 orders; print durations of 0 ms set to 0.5 ms",
    "classes": "major = BTC ETH SOL XRP DOGE; tradfi = CL BZ XAU XAG QQQ NVDA TSLA MSTR COIN CRWV GOOGL; alt = the rest (incl. BNB)",
    "H-MS1": "part 1: KS of raw TEST durations vs an exponential with the sample mean, p < 0.01, for EVERY symbol (Anderson-Darling at 1% reported); "
             "part 2: exponential, weibull, gamma, lognormal, gengamma, burr12, loglogistic (MLE, loc 0) and 2-3 component exponential mixtures (EM) "
             "fitted on FIT, winner = highest TEST log-likelihood (AIC on TEST reported), holds if a non-exponential family wins for > 80% of symbols; "
             "part 3: x = d / s with s the causal trailing 30-min mean duration, ACD(1,1) with exponential / Weibull / generalized-gamma unit-mean "
             "innovations fitted on FIT (after a 30-min warm-up), filtered through the whole day; residuals e = x / psi on TEST; holds if "
             "KS(e vs exponential) <= 0.5 KS(raw TEST durations vs exponential) for > 80% of symbols, using the ACD with the best FIT log-likelihood; "
             "supports = all three parts; p = max of the three one-sided binomial p-values (part 1 and 3 vs 0.8, part 2 vs 0.8)",
    "H-MS2": "notional = price x size (USD); CSN: x_min minimises KS over 80 candidates (tail sizes geometric from the upper half to 50 points), MLE "
             "alpha_pdf, zeta = alpha_pdf - 1; bootstrap p (B = 200 on TEST prints, 50 elsewhere), power law not rejected if p > 0.1 (CSN); "
             "Vuong vs a log-normal truncated at x_min, preferred if R > 0 and two-sided p < 0.05; Hill on the same tail; per symbol on TEST "
             "prints; supports = zeta in [1, 2] and not rejected for > 70% of symbols AND preferred for > 50%; p = max of binomial p vs 0.7 and 0.5",
    "H-MS3": "order signs (+1 buy, -1 sell) per half; ACF lags 1-1000 averaged in log bins (8 per decade); power law = OLS of log ACF on log lag, "
             "exponential = OLS of log ACF on lag, over bins whose ACF exceeds the 2/sqrt(n) noise band of that half (rule added after the "
             "CARV/KAS smoke test, before the full run: near-zero noisy bins dominated the log MSE); both fitted on FIT, held-out MSE of log "
             "ACF on the TEST bins above the TEST band; "
             "gamma = power-law fit on TEST; per symbol with >= 20k prints in the day: power preferred (lower held-out MSE) and 0 < gamma < 1; "
             "PACF lags 1-50 (Durbin-Levinson) with +-1.96/sqrt(n) bands, count of significant positive lags 11-50 (reported, not in the success line); "
             "Hurst = DFA-1 of 1 s signed notional on TEST (scales 10-4320 s), pooled = mean over the >= 20k symbols, 95% CI by bootstrap over symbols; "
             "supports = share > 0.8 and the CI excludes 0.5 from above; p = max(binomial p vs 0.8, bootstrap p of mean H <= 0.5)",
    "H-MS4": "notional volume per 10 s and per 60 s clock bin; positive bins: gamma, gengamma, lognormal fitted on FIT, TEST log-likelihood (zero "
             "bins share reported, identical for all families); distribution holds if max(gengamma, lognormal) > gamma; ACF of log(1 + v / median "
             "positive FIT v), lags 1-360 (10 s) / 1-120 (60 s), log bins, power vs exponential as in H-MS3 (FIT fit, TEST held-out MSE); "
             "co-movement: Spearman of v and |log return| (last price per bin, carried forward) on TEST > 0; a symbol passes if all three hold "
             "at both intervals; supports = > 80% of symbols; p = binomial p vs 0.8",
    "H-MS9": "arrival-event durations; xgboost survival:aft (hist, depth 6, eta 0.05, subsample 0.8, early stopping on the last 20% of FIT "
             "after warm-up) with distributions normal / logistic / extreme x training scale {0.8, 1.5}; sigma of the density re-estimated by "
             "MLE on the validation part; chosen config = best validation log-likelihood, refitted on all of FIT with 1.25x the best rounds; "
             "features at the start of each duration: log of the last 5 durations, log trailing 30-min scale, trailing 10 s / 60 s counts, "
             "|log return| over the last 10 / 100 events and 10 s / 60 s, signed-notional imbalance of the last 10 events and 60 s, last event "
             "sign / log notional / log prints, effective spread proxy |last buy - last sell| / price, L2 spread / touch imbalance / 5 bp depth "
             "imbalance / log 5 bp depth (12 ob200 symbols, 250 ms grid, point in time), hour of day; comparator = the ACD(1,1) with the best "
             "TEST log-likelihood (conservative); per-duration log-lik difference in seconds units, class value = mean over symbols of the "
             "per-symbol mean, 95% CI by a 5-minute block bootstrap shared across symbols (B = 2000); supports = difference > 0 with CI "
             "excluding 0 in >= 2 of 3 classes; p = second-smallest class bootstrap p",
    "holm": "Holm over the five summary p-values; supports only if the literal criterion holds and the Holm-adjusted p < 0.05; criterion met "
            "but Holm p >= 0.05 -> inconclusive; criterion not met -> contradicts",
}


def asset_class(sym: str) -> str:
    return "major" if sym in MAJORS else "tradfi" if sym in TRADFI else "alt"


def symbols() -> list[str]:
    import yaml

    syms = yaml.safe_load((ROOT / "config" / "mm_multi.yaml").read_text(encoding="utf-8"))["symbols"]
    return list(dict.fromkeys([*syms, "BTCUSDT"]))


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------

def load_trades(sym: str) -> dict[str, np.ndarray]:
    import duckdb

    glob = (ROOT / "data" / "lake" / "trades_raw" / "venue=bybit" / f"symbol={sym}" / f"date={DATE}" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    df = con.sql(f"SELECT epoch_ms(event_ts) AS ms, side, price, size, filename, file_row_number FROM read_parquet('{glob}', "
                 f"file_row_number=true, filename=true) ORDER BY ms, filename, file_row_number").pl()
    ms = df["ms"].to_numpy().astype(np.int64)
    keep = (ms >= DAY0) & (ms < DAY1)
    side = np.where(df["side"].to_numpy() == "Buy", 1, -1).astype(np.int8)
    return {"ms": ms[keep], "side": side[keep], "price": df["price"].to_numpy()[keep], "size": df["size"].to_numpy()[keep]}


def build_levels(tr: dict[str, np.ndarray]) -> dict[str, Any]:
    ms, side, px, sz = tr["ms"], tr["side"], tr["price"], tr["size"]
    notional = px * sz
    new = np.ones(len(ms), dtype=bool)
    new[1:] = (ms[1:] != ms[:-1]) | (side[1:] != side[:-1])
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    orders = {"ms": ms[starts], "side": side[starts], "notional": np.bincount(gid, notional), "n_prints": np.bincount(gid),
              "last_price": px[np.r_[starts[1:] - 1, len(ms) - 1]]}
    enew = np.ones(len(ms), dtype=bool)
    enew[1:] = ms[1:] != ms[:-1]
    eid = np.cumsum(enew) - 1
    es = np.flatnonzero(enew)
    events = {"ms": ms[es], "notional": np.bincount(eid, notional), "signed": np.bincount(eid, notional * side), "n_prints": np.bincount(eid),
              "last_price": px[np.r_[es[1:] - 1, len(ms) - 1]], "two_sided": np.bincount(eid, side == 1) * np.bincount(eid, side == -1) > 0}
    return {"prints": {"ms": ms, "side": side, "notional": notional, "price": px}, "orders": orders, "events": events}


def _halves(start_ms: np.ndarray, end_ms: np.ndarray | None = None) -> dict[str, np.ndarray]:
    end_ms = start_ms if end_ms is None else end_ms
    fit = (start_ms >= DAY0) & (end_ms < HALF)
    test = (start_ms >= HALF) & (end_ms < DAY1)
    ex = test & ~((start_ms >= CPI[0]) & (start_ms < CPI[1]))
    return {"fit": fit, "test": test, "test_ex": ex}


def _fin(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, np.bool_ | bool):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        return _fin(o)
    return o


# ----------------------------------------------------------------------------------------------------------------------
# H-MS1: durations
# ----------------------------------------------------------------------------------------------------------------------

def _anderson_exp(x: np.ndarray) -> dict[str, Any]:
    try:
        r = stats.anderson(x, dist="expon")
        return {"stat": float(r.statistic), "crit_1pct": float(r.critical_values[-1]), "reject_1pct": bool(r.statistic > r.critical_values[-1])}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def duration_block(t_ms: np.ndarray, zero_fill: float | None) -> dict[str, Any]:
    d = np.diff(t_ms) / 1000.0
    if zero_fill is not None:
        d = np.where(d <= 0, zero_fill, d)
    st, en = t_ms[:-1], t_ms[1:]
    H = _halves(st, en)
    out: dict[str, Any] = {"n": {h: int(m.sum()) for h, m in H.items()}, "zero_share": float(np.mean(np.diff(t_ms) == 0)) if zero_fill else 0.0}
    fits = {h: L.fit_all(d[H[h]]) for h in ("fit", "test")}
    out["laws"] = {h: {f: (v.to_dict() if isinstance(v, L.Law) else v) for f, v in fits[h].items()} for h in fits}
    for h in ("fit", "test", "test_ex"):
        x = d[H[h]]
        ks, p = L.ks_exponential(x)
        out[f"ks_exp_{h}"] = {"stat": ks, "p": p, "mean_s": float(x.mean()), "cv": float(x.std() / x.mean()), **{"ad": _anderson_exp(x)}}
    held = {}
    for h in ("test", "test_ex"):
        x = d[H[h]]
        rows = {}
        for f, law in fits["fit"].items():
            if not isinstance(law, L.Law):
                continue
            ll = law.loglik(x)
            rows[f] = {"ll": ll, "ll_per_obs": ll / len(x), "aic": -2 * ll + 2 * law.n_params, "ks": L.ks_stat(x, law.cdf)}
        rank = sorted(rows, key=lambda f: -rows[f]["ll"] if math.isfinite(rows[f]["ll"]) else math.inf)
        held[h] = {"by_family": rows, "ranking": rank, "winner": rank[0] if rank else None}
    out["heldout"] = held
    return out


def acd_block(t_ms: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    t_s = (t_ms - DAY0) / 1000.0
    d = np.diff(t_s)
    s = L.trailing_scale(t_s, 1800.0)
    st, en = t_ms[:-1], t_ms[1:]
    first = int(np.flatnonzero(np.isfinite(s))[0])
    s = s.copy()
    # forward-fill a quiet window (< 5 events in 30 min): the last finite scale is still causal
    idx = np.where(np.isfinite(s), np.arange(len(s)), -1)
    idx = np.maximum.accumulate(idx)
    s[first:] = s[idx[first:]]
    x = d / s
    H = _halves(st, en)
    fitm = H["fit"] & (st >= DAY0 + WARMUP_MS) & np.isfinite(x)
    out: dict[str, Any] = {"deseason": {"kind": "trailing_mean_duration", "window_s": 1800.0}, "models": {}, "test_models": {}}
    mods = {}
    for innov in ("exponential", "weibull", "gengamma"):
        try:
            mods[innov] = L.fit_acd(x[fitm], innov, start=mods.get("weibull"))
        except Exception as e:  # noqa: BLE001
            out["models"][innov] = f"failed: {e}"
    ll_arrays: dict[str, np.ndarray] = {"start_ms": st[first:], "end_ms": en[first:], "log_s": np.log(s[first:])}
    tm = H["test"][first:]
    tex = H["test_ex"][first:]
    ks_raw = L.ks_exponential(d[H["test"]])[0]
    ks_des = L.ks_exponential(x[H["test"]])[0]
    for innov, m in mods.items():
        xx = x[first:]
        psi = m.filter(xx, psi0=m.psi0)
        ll = m.logpdf(xx, psi) - np.log(s[first:])
        e = xx / psi
        ll_arrays[f"ll_{innov}"] = ll
        out["models"][innov] = {**m.to_dict(), "aic_fit": -2 * m.loglik_fit + 2 * m.n_params,
                                "ll_test": float(ll[tm].sum()), "ll_test_per_obs": float(ll[tm].mean()), "ll_test_ex": float(ll[tex].sum()),
                                "ks_resid_test": L.ks_exponential(e[tm])[0], "ks_resid_test_ex": L.ks_exponential(e[tex])[0],
                                "resid_mean_test": float(e[tm].mean()), "resid_cv_test": float(e[tm].std() / e[tm].mean())}
    ok = [k for k in mods]
    best_fit = max(ok, key=lambda k: mods[k].loglik_fit) if ok else None
    best_test = max(ok, key=lambda k: out["models"][k]["ll_test"]) if ok else None
    out.update({"best_by_fit": best_fit, "best_by_test": best_test, "ks_raw_test": ks_raw, "ks_deseason_test": ks_des,
                "ks_raw_test_ex": L.ks_exponential(d[H["test_ex"]])[0]})
    if best_fit:
        mb = out["models"][best_fit]
        out["ks_ratio_test"] = mb["ks_resid_test"] / ks_raw
        out["ks_ratio_test_ex"] = mb["ks_resid_test_ex"] / out["ks_raw_test_ex"]
    # per-half parameters for the simulator (TEST refit, after the same causal scale)
    testm = H["test"] & np.isfinite(x)
    for innov in ("weibull", "gengamma"):
        try:
            mt = L.fit_acd(x[testm], innov, start=mods.get("weibull"))
            out["test_models"][innov] = mt.to_dict()
        except Exception as e:  # noqa: BLE001
            out["test_models"][innov] = f"failed: {e}"
    return out, ll_arrays


# ----------------------------------------------------------------------------------------------------------------------
# H-MS2: sizes
# ----------------------------------------------------------------------------------------------------------------------

def size_block(notional: np.ndarray, ms: np.ndarray, B_test: int, B_fit: int, rng: np.random.Generator) -> dict[str, Any]:
    H = _halves(ms)
    out = {}
    for h, B in (("fit", B_fit), ("test", B_test), ("test_ex", 0)):
        x = notional[H[h] & (notional > 0)]
        if len(x) < 200:
            out[h] = {"n": int(len(x)), "error": "too few"}
            continue
        f = L.fit_powerlaw(x)
        r = {"n": int(len(x)), "powerlaw": f.to_dict(), "hill_zeta": L.hill(x, f.n_tail), "vuong": L.vuong_lognormal(x, f),
             "median": float(np.median(x)), "p99": float(np.quantile(x, 0.99)), "max": float(x.max())}
        if B:
            r["p_boot"] = L.powerlaw_bootstrap_p(x, f, B=B, rng=rng)
            r["B"] = B
        if h != "test_ex":
            r["size_law"] = L.fit_size_law(x, f).to_dict()
        out[h] = r
    return out


# ----------------------------------------------------------------------------------------------------------------------
# H-MS3: signs, signed flow
# ----------------------------------------------------------------------------------------------------------------------

def seg_acf(x: np.ndarray, segs: list[np.ndarray], maxlag: int) -> np.ndarray:
    """ACF pooled over segments (global mean and variance; lagged products never cross a segment boundary)."""
    m = x[np.concatenate(segs)].mean()
    num = np.zeros(maxlag + 1)
    den = 0.0
    for sg in segs:
        y = x[sg] - m
        n = len(y)
        if n < 2:
            continue
        nfft = 1 << int(math.ceil(math.log2(2 * n)))
        f = np.fft.rfft(y, nfft)
        r = np.fft.irfft(f * np.conj(f), nfft)[: maxlag + 1]
        num[: len(r)] += r
        den += float(y @ y)
    return num / den


def _segments(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.split(idx, cuts)


def acf_compare(x: np.ndarray, Hm: dict[str, np.ndarray], maxlag: int, pacf_lags: int = 50) -> dict[str, Any]:
    rho = {h: seg_acf(x.astype(float), _segments(Hm[h]), maxlag) for h in ("fit", "test", "test_ex")}
    bins = L.log_bins(maxlag)
    kb = {h: L.binned(rho[h], bins) for h in rho}
    fl = {h: 2.0 / math.sqrt(max(int(Hm[h].sum()), 1)) for h in rho}  # noise band: bins below it are not fitted / scored
    pw = {h: L.fit_power_acf(*kb[h], floor=fl[h]) for h in rho}
    ex = {h: L.fit_exp_acf(*kb[h], floor=fl[h]) for h in rho}
    out: dict[str, Any] = {"n": {h: int(Hm[h].sum()) for h in Hm}, "noise_floor": fl}
    for h in ("test", "test_ex"):
        mp, me, nb = L.heldout_log_mse(*kb[h], pw["fit"], ex["fit"], floor=fl[h])
        out[f"heldout_{h}"] = {"mse_power": mp, "mse_exp": me, "n_bins": nb, "power_preferred": bool(mp < me) if math.isfinite(mp) and math.isfinite(me) else None}
    for h in rho:
        n = int(Hm[h].sum())
        pac = L.pacf_from_acf(rho[h][: pacf_lags + 1], pacf_lags)
        band = 1.96 / math.sqrt(max(n, 1))
        out[h] = {"power": pw[h], "exp": ex[h], "acf": {str(k): float(rho[h][k]) for k in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000) if k <= maxlag},
                  "acf_bins": {"k": kb[h][0].tolist(), "v": kb[h][1].tolist()}, "pacf": pac.tolist(), "band": band,
                  "pacf_sig_pos_11_50": int(np.sum(pac[10:50] > band)) if pacf_lags >= 50 else None,
                  "pacf_sig_neg_11_50": int(np.sum(pac[10:50] < -band)) if pacf_lags >= 50 else None}
    return out


def clock_flow(ms: np.ndarray, signed: np.ndarray, count_sign: np.ndarray, bin_ms: int) -> tuple[np.ndarray, np.ndarray]:
    nb = 86_400_000 // bin_ms
    b = ((ms - DAY0) // bin_ms).astype(np.int64)
    return np.bincount(b, signed, minlength=nb)[:nb], np.bincount(b, count_sign, minlength=nb)[:nb]


def sign_block(lv: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in ("order", "print"):
        src = lv["orders"] if level == "order" else lv["prints"]
        Hm = _halves(src["ms"])
        sg = src["side"].astype(float)
        r = acf_compare(sg, Hm, MAX_LAG)
        for h in ("fit", "test"):
            pw = r[h]["power"]
            p_buy = float(np.mean(sg[Hm[h]] > 0))
            if np.isfinite(pw["c"]) and np.isfinite(pw["gamma"]):
                r[h]["fgn_sign"] = L.FGNSignModel(pw["c"], pw["gamma"], p_buy).to_dict()
            r[h]["p_buy"] = p_buy
        out[level] = r
    o = lv["orders"]
    Hm = _halves(o["ms"])
    out["order_signed_notional"] = acf_compare(o["side"] * o["notional"], Hm, MAX_LAG)
    # clock-time signed flow
    p = lv["prints"]
    for bin_s, maxlag in ((1, 300), (10, 360)):
        fl, cnt = clock_flow(p["ms"], p["side"] * p["notional"], p["side"].astype(float), bin_s * 1000)
        nb = len(fl)
        t0 = np.arange(nb) * bin_s * 1000 + DAY0
        Hc = _halves(t0, t0 + bin_s * 1000 - 1)
        r = acf_compare(fl, Hc, maxlag, pacf_lags=30)
        if bin_s == 1:
            for h in ("fit", "test", "test_ex"):
                r[h]["hurst_dfa"] = L.dfa(fl[Hc[h]], scales=np.unique(np.geomspace(10, 4320, 20).astype(int)))["H"]
                r[h]["hurst_dfa_count"] = L.dfa(cnt[Hc[h]], scales=np.unique(np.geomspace(10, 4320, 20).astype(int)))["H"]
        out[f"clock_{bin_s}s"] = r
    return out


# ----------------------------------------------------------------------------------------------------------------------
# H-MS4: volume per interval
# ----------------------------------------------------------------------------------------------------------------------

def volume_block(lv: dict[str, Any]) -> dict[str, Any]:
    p = lv["prints"]
    out = {}
    for bin_s, maxlag in ((10, 360), (60, 120)):
        bm = bin_s * 1000
        nb = 86_400_000 // bm
        b = ((p["ms"] - DAY0) // bm).astype(np.int64)
        v = np.bincount(b, p["notional"], minlength=nb)[:nb]
        lastp = np.full(nb, np.nan)
        endb = np.flatnonzero(np.r_[b[1:] != b[:-1], True])  # last print of each non-empty bin (prints are time-sorted)
        lastp[b[endb]] = p["price"][endb]
        idx = np.where(np.isfinite(lastp), np.arange(nb), -1)
        idx = np.maximum.accumulate(idx)
        lp = np.where(idx >= 0, lastp[np.maximum(idx, 0)], np.nan)
        r = np.r_[np.nan, np.diff(np.log(lp))]
        t0 = np.arange(nb) * bm + DAY0
        Hc = _halves(t0, t0 + bm - 1)
        blk: dict[str, Any] = {"zero_share": {h: float(np.mean(v[Hc[h]] == 0)) for h in Hc}}
        fits = {}
        for h in ("fit", "test"):
            pos = v[Hc[h] & (v > 0)]
            fits[h] = L.fit_all(pos, families=("gamma", "gengamma", "lognormal"))
            blk[f"laws_{h}"] = {f: (law.to_dict() if isinstance(law, L.Law) else law) for f, law in fits[h].items()}
            blk[f"p_zero_{h}"] = float(np.mean(v[Hc[h]] == 0))
        for h in ("test", "test_ex"):
            pos = v[Hc[h] & (v > 0)]
            ll = {f: (law.loglik(pos) if isinstance(law, L.Law) else float("nan")) for f, law in fits["fit"].items()}
            blk[f"heldout_{h}"] = {"ll": ll, "n_pos": int(len(pos)),
                                   "heavy_beats_gamma": bool(max(ll.get("gengamma", -np.inf), ll.get("lognormal", -np.inf)) > ll.get("gamma", -np.inf))}
        med = float(np.median(v[Hc["fit"] & (v > 0)]))
        y = np.log1p(v / med)
        blk["acf"] = acf_compare(y, Hc, maxlag, pacf_lags=20)
        for h in ("fit", "test", "test_ex"):
            m = Hc[h] & np.isfinite(r)
            rho, pv = stats.spearmanr(v[m], np.abs(r[m]))
            blk[f"spearman_absret_{h}"] = {"rho": float(rho), "p": float(pv), "n": int(m.sum())}
        blk["median_pos_fit"] = med
        out[f"{bin_s}s"] = blk
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Stage 'laws': everything except the AFT
# ----------------------------------------------------------------------------------------------------------------------

def rates_block(lv: dict[str, Any]) -> dict[str, Any]:
    out = {}
    o, e, p = lv["orders"], lv["events"], lv["prints"]
    for h, (a, b) in (("fit", (DAY0, HALF)), ("test", (HALF, DAY1))):
        mo = (o["ms"] >= a) & (o["ms"] < b)
        me = (e["ms"] >= a) & (e["ms"] < b)
        mp = (p["ms"] >= a) & (p["ms"] < b)
        npo = o["n_prints"][mo]
        pmf = {str(k): float(np.mean(npo == k)) for k in range(1, 11)}
        pmf[">10"] = float(np.mean(npo > 10))
        tail = npo[npo >= 2].astype(float)
        out[h] = {"prints_per_s": float(mp.sum() / 43200), "orders_per_s": float(mo.sum() / 43200), "events_per_s": float(me.sum() / 43200),
                  "prints_per_order_mean": float(npo.mean()), "prints_per_order_pmf": pmf,
                  "prints_per_order_tail_zeta": float(1.0 / np.mean(np.log(tail / 1.5))) if len(tail) > 50 else None,
                  "p_buy_order": float(np.mean(o["side"][mo] > 0)), "two_sided_ms_share": float(np.mean(e["two_sided"][me])),
                  "notional_usd": float(p["notional"][mp].sum()), "median_print_usd": float(np.median(p["notional"][mp])),
                  "hourly_events": np.bincount(((e["ms"][me] - a) // 3_600_000).astype(int), minlength=12)[:12].tolist()}
    return out


def analyse_symbol(sym: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    rng = np.random.default_rng(SEED + sum(map(ord, sym)))
    tr = load_trades(sym)
    lv = build_levels(tr)
    res: dict[str, Any] = {"symbol": sym, "class": asset_class(sym), "n_prints": int(len(tr["ms"])), "n_orders": int(len(lv["orders"]["ms"])),
                           "n_events": int(len(lv["events"]["ms"]))}
    res["rates"] = rates_block(lv)
    res["durations"] = {"event": duration_block(lv["events"]["ms"], None), "print": duration_block(lv["prints"]["ms"], 0.0005)}
    acd, arrays = acd_block(lv["events"]["ms"])
    res["acd"] = acd
    np.savez_compressed(CACHE / f"{sym}_acd.npz", **arrays)
    res["size"] = {"print": size_block(lv["prints"]["notional"], lv["prints"]["ms"], B_CSN, B_CSN_SECONDARY, rng),
                   "order": size_block(lv["orders"]["notional"], lv["orders"]["ms"], B_CSN_SECONDARY, B_CSN_SECONDARY, rng)}
    res["signs"] = sign_block(lv)
    res["volume"] = volume_block(lv)
    res["seconds"] = time.time() - t0
    (CACHE / f"{sym}.json").write_text(json.dumps(_jsonable(res)), encoding="utf-8")
    return {"symbol": sym, "seconds": res["seconds"]}


def refit_symbol(sym: str) -> dict[str, Any]:
    """Stage 'refit': recompute the duration-family and volume blocks only (after the nested-start fix in sim.laws,
    diary 'mistake' 2026-11-11) and patch the cached symbol result; ACD, CSN, sign and AFT results are unchanged."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    f = CACHE / f"{sym}.json"
    res = json.loads(f.read_text(encoding="utf-8"))
    lv = build_levels(load_trades(sym))
    res["durations"] = {"event": duration_block(lv["events"]["ms"], None), "print": duration_block(lv["prints"]["ms"], 0.0005)}
    res["volume"] = volume_block(lv)
    res["refit"] = "nested-start fix"
    f.write_text(json.dumps(_jsonable(res)), encoding="utf-8")
    (CACHE / f"{sym}.refit_done.json").write_text("{}", encoding="utf-8")
    return {"symbol": sym, "seconds": time.time() - t0}


# ----------------------------------------------------------------------------------------------------------------------
# Stage 'aft': H-MS9
# ----------------------------------------------------------------------------------------------------------------------

def _trail_sum(t: np.ndarray, v: np.ndarray, w_ms: int) -> np.ndarray:
    c = np.r_[0.0, np.cumsum(v)]
    lo = np.searchsorted(t, t - w_ms, side="right")
    return c[np.arange(1, len(t) + 1)] - c[lo]


def _price_at(t: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """last price at or before times q (NaN before the first)."""
    i = np.searchsorted(t, q, side="right") - 1
    return np.where(i >= 0, p[np.maximum(i, 0)], np.nan)


def aft_features(sym: str, lv: dict[str, Any]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Features known at the start of each arrival-event duration (event j -> j+1), the duration (s) and its start ms."""
    e, p = lv["events"], lv["prints"]
    t = e["ms"]
    n = len(t)
    d = np.diff(t) / 1000.0
    st = t[:-1]
    F: dict[str, np.ndarray] = {}
    prev = np.r_[np.nan, d]  # prev[j] = duration that ended at t_j
    for k in range(5):
        F[f"log_d_lag{k + 1}"] = np.log(np.r_[np.full(k, np.nan), prev[: n - k]])[:-1]
    F["log_scale30m"] = np.log(L.trailing_scale((t - DAY0) / 1000.0, 1800.0))
    ones = np.ones(n)
    F["cnt_10s"] = _trail_sum(t, ones, 10_000)[:-1]
    F["cnt_60s"] = _trail_sum(t, ones, 60_000)[:-1]
    lp = np.log(e["last_price"])
    for k in (10, 100):
        F[f"absret_{k}ev"] = np.abs(lp - np.r_[np.full(k, np.nan), lp[:-k]])[:-1]
    for w in (10_000, 60_000):
        F[f"absret_{w // 1000}s"] = np.abs(lp - np.log(_price_at(t, e["last_price"], t - w)))[:-1]
    sn = e["signed"]
    an = e["notional"]
    c_s, c_a = np.r_[0.0, np.cumsum(sn)], np.r_[0.0, np.cumsum(an)]
    j = np.arange(1, n + 1)
    lo10 = np.maximum(j - 10, 0)
    F["imb_10ev"] = ((c_s[j] - c_s[lo10]) / (c_a[j] - c_a[lo10]))[:-1]
    F["imb_60s"] = (_trail_sum(t, sn, 60_000) / _trail_sum(t, an, 60_000))[:-1]
    F["last_sign"] = np.sign(sn)[:-1]
    F["last_log_notional"] = np.log(an)[:-1]
    F["last_log_prints"] = np.log(e["n_prints"])[:-1]
    # effective spread proxy from prints: last buy and last sell price at or before t_j
    pm, ps, pp = p["ms"], p["side"], p["price"]
    out_sp = []
    for sgn in (1, -1):
        m = ps == sgn
        out_sp.append(_price_at(pm[m], pp[m], t))
    F["eff_spread_bps"] = (np.abs(out_sp[0] - out_sp[1]) / e["last_price"] * 1e4)[:-1]
    bk = BOOKS / f"{sym}.parquet"
    if bk.exists():
        b = pl.read_parquet(bk)
        bt = b["t_ms"].to_numpy()
        i = np.searchsorted(bt, st, side="right") - 1
        ok = i >= 0
        for col in ("spread_bps", "obi_l1", "obi_5"):
            v = b[col].to_numpy()
            F[f"l2_{col}"] = np.where(ok, v[np.maximum(i, 0)], np.nan)
        dep = b["depth_5"].to_numpy()
        F["l2_log_depth5"] = np.where(ok, np.log(np.maximum(dep[np.maximum(i, 0)], 1e-9)), np.nan)
    F["hour_of_day"] = (st - DAY0) / 3_600_000.0
    names = list(F)
    X = np.column_stack([F[k] for k in names]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, names, d, st


def aft_symbol(sym: str, nthread: int = 2) -> dict[str, Any]:
    import xgboost as xgb

    warnings.filterwarnings("ignore")
    t0 = time.time()
    tr = load_trades(sym)
    lv = build_levels(tr)
    X, names, d, st = aft_features(sym, lv)
    en = lv["events"]["ms"][1:]
    H = _halves(st, en)
    fitm = H["fit"] & (st >= DAY0 + WARMUP_MS)
    fi = np.flatnonzero(fitm)
    cut = fi[int(0.8 * len(fi))]
    tr_i, va_i = fi[fi < cut], fi[fi >= cut]
    te_i = np.flatnonzero(H["test"])

    def dm(idx):
        m = xgb.DMatrix(X[idx], feature_names=names, nthread=nthread)
        m.set_float_info("label_lower_bound", d[idx])
        m.set_float_info("label_upper_bound", d[idx])
        return m

    dtr, dva = dm(tr_i), dm(va_i)
    base = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "tree_method": "hist", "learning_rate": 0.05, "max_depth": 6,
            "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "nthread": nthread, "seed": SEED, "verbosity": 0}
    grid = []
    for dist in ("normal", "logistic", "extreme"):
        for sc in (0.8, 1.5):
            prm = {**base, "aft_loss_distribution": dist, "aft_loss_distribution_scale": sc}
            bst = xgb.train(prm, dtr, num_boost_round=2000, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
            mu = bst.predict(dva, output_margin=True, iteration_range=(0, bst.best_iteration + 1))
            r = optimize.minimize_scalar(lambda ls: -float(np.sum(L.aft_logpdf(d[va_i], mu, math.exp(ls), dist))), bounds=(-4, 2), method="bounded")
            sig = math.exp(r.x)
            grid.append({"dist": dist, "train_scale": sc, "best_iter": int(bst.best_iteration), "sigma": sig,
                         "ll_valid_per_obs": float(-r.fun / len(va_i))})
    best = max(grid, key=lambda g: g["ll_valid_per_obs"])
    prm = {**base, "aft_loss_distribution": best["dist"], "aft_loss_distribution_scale": best["train_scale"]}
    dall = dm(fi)
    rounds = max(20, int(1.25 * (best["best_iter"] + 1)))
    bst = xgb.train(prm, dall, num_boost_round=rounds, verbose_eval=False)
    AFT_DIR.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(AFT_DIR / f"{sym}.json"))
    dte = xgb.DMatrix(X[te_i], feature_names=names, nthread=nthread)
    mu = bst.predict(dte, output_margin=True)
    ll_aft = L.aft_logpdf(d[te_i], mu, best["sigma"], best["dist"])
    # robustness: same config without the hour-of-day feature
    keep = [k for k in names if k != "hour_of_day"]
    ki = [names.index(k) for k in keep]

    def dm2(idx):
        m = xgb.DMatrix(X[idx][:, ki], feature_names=keep, nthread=nthread)
        m.set_float_info("label_lower_bound", d[idx])
        m.set_float_info("label_upper_bound", d[idx])
        return m

    bst2 = xgb.train(prm, dm2(fi), num_boost_round=rounds, verbose_eval=False)
    mu2 = bst2.predict(xgb.DMatrix(X[te_i][:, ki], feature_names=keep, nthread=nthread), output_margin=True)
    ll_aft_notod = L.aft_logpdf(d[te_i], mu2, best["sigma"], best["dist"])
    # ACD per-duration log-likelihoods from stage 'laws', aligned by start ms
    z = np.load(CACHE / f"{sym}_acd.npz")
    pos = np.searchsorted(z["start_ms"], st[te_i])
    okp = (pos < len(z["start_ms"])) & (z["start_ms"][np.minimum(pos, len(z["start_ms"]) - 1)] == st[te_i])
    acd_ll = {k[3:]: z[k][np.minimum(pos, len(z["start_ms"]) - 1)] for k in z.files if k.startswith("ll_")}
    gain = {}
    imp = bst.get_score(importance_type="gain")
    tot = sum(imp.values()) or 1.0
    gain = {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    np.savez_compressed(CACHE / f"{sym}_aft.npz", start_ms=st[te_i], ok=okp, ll_aft=ll_aft, ll_aft_notod=ll_aft_notod,
                        **{f"ll_acd_{k}": v for k, v in acd_ll.items()})
    res = {"symbol": sym, "class": asset_class(sym), "n_train": int(len(fi)), "n_test": int(len(te_i)), "aligned": float(okp.mean()),
           "grid": grid, "best": best, "rounds": rounds, "features": names, "has_l2": bool((BOOKS / f"{sym}.parquet").exists()),
           "ll_aft_test_per_obs": float(ll_aft.mean()), "ll_aft_notod_test_per_obs": float(ll_aft_notod.mean()),
           "ll_acd_test_per_obs": {k: float(v[okp].mean()) for k, v in acd_ll.items()}, "importance_gain": gain, "seconds": time.time() - t0}
    (CACHE / f"{sym}_aft.json").write_text(json.dumps(_jsonable(res)), encoding="utf-8")
    return {"symbol": sym, "seconds": res["seconds"]}


# ----------------------------------------------------------------------------------------------------------------------
# Stage 'report': verdicts, Holm, params JSON / parquet, report
# ----------------------------------------------------------------------------------------------------------------------

def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=lambda k: p[k])
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def _binom_p(k: int, n: int, p0: float) -> float:
    """one-sided P(X >= k) under share p0 (evidence that the share exceeds p0)."""
    return float(stats.binom.sf(k - 1, n, p0)) if n else 1.0


def _share(flags: list[bool]) -> tuple[int, int, float]:
    f = [bool(x) for x in flags if x is not None]
    return sum(f), len(f), (sum(f) / len(f) if f else float("nan"))


def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _f(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        return "n/a"
    a = abs(x)
    if a != 0 and (a < 10 ** -nd or a >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _block_boot(per_sym: dict[str, tuple[np.ndarray, np.ndarray]], B: int, rng: np.random.Generator, nblk: int) -> dict[str, float]:
    """per_sym: symbol -> (block sums of the difference, block counts), blocks aligned in clock time. Statistic = mean over
    symbols of the per-symbol mean difference; bootstrap draws the same blocks for every symbol (keeps cross-symbol dependence)."""
    syms = list(per_sym)
    S = np.array([per_sym[s][0] for s in syms])
    N = np.array([per_sym[s][1] for s in syms])
    ok = N.sum(axis=1) > 0
    S, N = S[ok], N[ok]
    stat = float(np.mean(S.sum(axis=1) / N.sum(axis=1)))
    bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, nblk, nblk)
        ns = N[:, idx].sum(axis=1)
        m = ns > 0
        bs[b] = np.mean(S[:, idx].sum(axis=1)[m] / ns[m])
    return {"diff": stat, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "p_le0": float(np.mean(bs <= 0)), "n_symbols": int(ok.sum())}


def _flatten(prefix: str, o, rows: list, base: dict) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, rows, base)
    elif isinstance(o, list):
        if all(isinstance(v, int | float) or v is None for v in o):
            for i, v in enumerate(o):
                if v is not None:
                    rows.append({**base, "path": f"{prefix}[{i}]", "value": float(v)})
        else:
            for i, v in enumerate(o):
                _flatten(f"{prefix}[{i}]", v, rows, base)
    elif isinstance(o, bool):
        rows.append({**base, "path": prefix, "value": float(o)})
    elif isinstance(o, int | float):
        if o is not None and math.isfinite(o):
            rows.append({**base, "path": prefix, "value": float(o)})
    elif isinstance(o, str):
        rows.append({**base, "path": prefix, "value": float("nan"), "text": o})


def _half_params(r: dict, a: dict | None, h: str) -> dict[str, Any]:
    """The simulator-facing parameter set of one symbol and half (laws fitted on that half)."""
    D = r["durations"]
    out: dict[str, Any] = {"rates": r["rates"][h]}
    out["durations"] = {}
    for lvl in ("event", "print"):
        laws = {f: v for f, v in D[lvl]["laws"][h].items()}
        ok = {f: v for f, v in laws.items() if isinstance(v, dict)}
        best_own = max(ok, key=lambda f: ok[f]["loglik_fit"] - ok[f]["k"]) if ok else None  # lowest AIC
        out["durations"][lvl] = {"laws": laws, "best_aic_own_half": best_own, "zero_share": D[lvl].get("zero_share"),
                                 "heldout_winner_fit_to_test": D[lvl]["heldout"]["test"]["winner"] if h == "fit" else None}
    A = r["acd"]
    if h == "fit":
        out["acd"] = {"deseason": A["deseason"], "best_by_fit": A["best_by_fit"], "best_by_test": A["best_by_test"],
                      "models": {k: ({**{x: v[x] for x in ("kind", "omega", "alpha", "beta", "innov", "shape", "psi0", "n_fit", "loglik_fit", "k")},
                                      "m2_factor": L.ACD.from_dict(v).m2_factor()}
                                     if isinstance(v, dict) else v) for k, v in A["models"].items()}}
    else:
        out["acd"] = {"deseason": A["deseason"], "models": {k: ({**v, "m2_factor": L.ACD.from_dict(v).m2_factor()} if isinstance(v, dict) else v)
                                                            for k, v in A["test_models"].items()}}
    out["size"] = {}
    for lvl in ("print", "order"):
        z = r["size"][lvl][h]
        out["size"][lvl] = {k: z.get(k) for k in ("n", "powerlaw", "size_law", "hill_zeta", "vuong", "p_boot", "B", "median", "p99", "max")}
    out["signs"] = {}
    for lvl in ("order", "print"):
        z = r["signs"][lvl][h]
        pw = z["power"]
        tsm = (L.TranslatedSignModel(pw["c"], pw["gamma"], z.get("p_buy", 0.5),
                                     {int(k): float(v) for k, v in z["acf"].items() if int(k) <= 20 and v is not None and v > 0}).to_dict()
               if pw.get("c") is not None and pw.get("gamma") is not None else None)
        out["signs"][lvl] = {"acf_power": z["power"], "acf_exp": z["exp"], "sign_model": tsm, "fgn_sign": z.get("fgn_sign"), "p_buy": z.get("p_buy"),
                             "acf": z["acf"], "pacf_1_20": z["pacf"][:20], "noise_floor": r["signs"][lvl]["noise_floor"][h]}
    out["signed_flow"] = {}
    for key in ("order_signed_notional", "clock_1s", "clock_10s"):
        z = r["signs"][key][h]
        out["signed_flow"][key] = {"acf_power": z["power"], "acf_exp": z["exp"], "acf": z["acf"], "pacf_1_20": z["pacf"][:20],
                                   **({"hurst_dfa": z.get("hurst_dfa"), "hurst_dfa_count": z.get("hurst_dfa_count")} if key == "clock_1s" else {})}
    out["volume"] = {}
    for b in ("10s", "60s"):
        V = r["volume"][b]
        laws = V[f"laws_{h}"]
        ok = {f: v for f, v in laws.items() if isinstance(v, dict)}
        out["volume"][b] = {"unit": "USD notional per bin", "p_zero": V[f"p_zero_{h}"], "laws_positive": laws,
                            "best_own_half": max(ok, key=lambda f: ok[f]["loglik_fit"] - ok[f]["k"]) if ok else None,
                            "acf_log1p_power": V["acf"][h]["power"], "acf_log1p_exp": V["acf"][h]["exp"], "acf_log1p": V["acf"][h]["acf"],
                            "log1p_divisor_usd": V["median_pos_fit"], "spearman_absret": V[f"spearman_absret_{h}"]}
    if h == "fit" and a:
        out["aft"] = {"model_file": f"data/models/microstructure_laws/aft_{DATE}/{r['symbol']}.json", "best": a["best"], "rounds": a["rounds"],
                      "features": a["features"], "importance_gain": a["importance_gain"]}
    return out


PARAMS_SCHEMA = {
    "file": "data/models/microstructure_laws/params_2026-11-10.json (+ params_2026-11-10.parquet: tidy rows symbol, class, half, path, value, text)",
    "top": "schema_version, date, created, method (METHOD dict), units, classes, symbols{SYM}, class_medians{class}",
    "symbols.SYM": "class, n_prints, n_orders, n_events, halves{fit, test}: every law below is fitted on THAT half (fit = 00-12 UTC, test = 12-24 UTC)",
    "rates": "prints_per_s, orders_per_s, events_per_s, prints_per_order_mean, prints_per_order_pmf{1..10, >10}, prints_per_order_tail_zeta "
             "(Hill on n >= 2), p_buy_order, two_sided_ms_share, notional_usd, median_print_usd, hourly_events[12]",
    "durations.{event|print}.laws.FAMILY": "Law dict {family, params, n_fit, loglik_fit, k}: params use scipy names with loc = 0: exponential{scale}, "
             "weibull{c, scale} = weibull_min, gamma{a, scale}, lognormal{s, scale} = lognorm, gengamma{a, c, scale}, burr12{c, d, scale}, "
             "loglogistic{c, scale} = fisk, expmixK{w1..wK, rate1..rateK (per second)}; durations in seconds; rebuild with "
             "ai_trader.sim.laws.law_from_dict(d).sample(n)",
    "acd": "deseason{kind: trailing_mean_duration, window_s: 1800} (x = d / s, s = mean event duration over the trailing 30 min); "
           "models{exponential|weibull|gengamma}: {omega, alpha, beta, innov, shape{k} or {a, c} (unit-mean innovation), psi0}; "
           "duration = s * psi * eps; ai_trader.sim.laws.ACD.from_dict(d).sample(n, scale=s) (most fits are near-integrated, so "
           "sample() caps alpha + beta at 0.99 and anchors at psi0 unless max_persistence=1); m2_factor = E[(alpha eps + beta)^2]: only "
           "fits with m2_factor < 1 are stable generators, the others are one-step filters",
    "size.{print|order}": "powerlaw{xmin (USD), alpha (pdf exponent), zeta = alpha - 1 (CCDF tail index), n_tail, n, ks, p_tail}, "
                          "size_law{kind: size, mu, sigma (log-normal body below xmin), xmin, alpha, p_tail} -> SizeLaw.sample, hill_zeta, "
                          "vuong{R, z, p, ln_mu, ln_sigma}, p_boot (CSN), median, p99, max",
    "signs.{order|print}": "acf_power{c, gamma} (ACF ~ c k^-gamma, log-binned lags 1-1000 above the noise band), acf_exp{c, tau}, "
                           "sign_model{kind: translated_sign, c, gamma, p_buy, short{lag: ACF for lags 1-20}} -> law_from_dict(d).sample(n) "
                           "gives +1/-1 signs matching the empirical short lags and the power-law tail (USE THIS ONE); fgn_sign{c, gamma, "
                           "p_buy, H_latent, w} (pure power law; cannot reach the lag-1 ACF of ~0.35 when gamma > ~0.5), p_buy, "
                           "acf{lag: value}, pacf_1_20",
    "signed_flow": "order_signed_notional (per order, trade time), clock_1s / clock_10s (signed USD per clock bin): acf_power, acf_exp, acf, "
                   "pacf_1_20; clock_1s also hurst_dfa (signed notional) and hurst_dfa_count (signed count)",
    "volume.{10s|60s}": "p_zero, laws_positive{gamma|gengamma|lognormal} (USD per bin, positive bins), best_own_half, acf_log1p_power / "
                        "acf_log1p_exp (ACF of log(1 + v / log1p_divisor_usd)), spearman_absret{rho, p, n}",
    "aft (fit half)": "model_file (xgboost JSON, survival:aft; predict(output_margin=True) = mu, log T = mu + sigma Z), best{dist, sigma, "
                      "train_scale, best_iter}, rounds, features (ordered names, see aft_features), importance_gain",
}


def build_report(syms: list[str], record: bool = False) -> None:
    from ai_trader.research import diary, hypotheses

    rng = np.random.default_rng(SEED)
    R = {}
    A = {}
    for s_ in syms:
        f = CACHE / f"{s_}.json"
        if f.exists():
            R[s_] = json.loads(f.read_text(encoding="utf-8"))
        fa = CACHE / f"{s_}_aft.json"
        if fa.exists():
            A[s_] = json.loads(fa.read_text(encoding="utf-8"))
    missing = [s_ for s_ in syms if s_ not in R]
    classes = {s_: asset_class(s_) for s_ in R}
    CL = ("major", "alt", "tradfi")
    out: dict[str, Any] = {"as_of": DATE, "method": METHOD, "n_symbols": len(R), "missing": missing, "missing_aft": [s_ for s_ in R if s_ not in A]}

    # ---------------- H-MS1
    ms1 = {}
    for s_, r in R.items():
        D = r["durations"]["event"]
        Dp = r["durations"]["print"]
        ac = r["acd"]
        ms1[s_] = {"class": classes[s_], "n_test": D["n"]["test"], "ks_raw": D["ks_exp_test"]["stat"], "ks_p": D["ks_exp_test"]["p"],
                   "ad_reject": _g(D, "ks_exp_test", "ad", "reject_1pct"), "cv": D["ks_exp_test"]["cv"],
                   "part1": D["ks_exp_test"]["p"] < 0.01, "winner": D["heldout"]["test"]["winner"], "ranking": D["heldout"]["test"]["ranking"],
                   "part2": D["heldout"]["test"]["winner"] not in (None, "exponential"),
                   "winner_ex": D["heldout"]["test_ex"]["winner"], "part2_ex": D["heldout"]["test_ex"]["winner"] not in (None, "exponential"),
                   "ll_gain_vs_exp": (D["heldout"]["test"]["by_family"][D["heldout"]["test"]["winner"]]["ll_per_obs"]
                                      - D["heldout"]["test"]["by_family"]["exponential"]["ll_per_obs"]) if D["heldout"]["test"]["winner"] else None,
                   "acd_best_fit": ac["best_by_fit"], "acd_best_test": ac["best_by_test"], "ks_resid": _g(ac, "models", ac["best_by_fit"], "ks_resid_test"),
                   "ks_deseason": ac["ks_deseason_test"], "ks_ratio": ac.get("ks_ratio_test"), "part3": (ac.get("ks_ratio_test") or 9) <= 0.5,
                   "ks_ratio_ex": ac.get("ks_ratio_test_ex"), "part3_ex": (ac.get("ks_ratio_test_ex") or 9) <= 0.5,
                   "acd": {k: {x: v.get(x) for x in ("alpha", "beta", "omega", "shape", "ll_test_per_obs", "ks_resid_test")} for k, v in ac["models"].items() if isinstance(v, dict)},
                   "print_part1": Dp["ks_exp_test"]["p"] < 0.01, "print_winner": Dp["heldout"]["test"]["winner"],
                   "print_part2": Dp["heldout"]["test"]["winner"] not in (None, "exponential"), "print_zero_share": Dp["zero_share"]}
    k1, n1, s1 = _share([v["part1"] for v in ms1.values()])
    k2, n2, s2 = _share([v["part2"] for v in ms1.values()])
    k3, n3, s3 = _share([v["part3"] for v in ms1.values()])
    crit1 = (k1 == n1) and s2 > 0.8 and s3 > 0.8
    p_ms1 = max(_binom_p(k1, n1, 0.8), _binom_p(k2, n2, 0.8), _binom_p(k3, n3, 0.8))
    fam_rank = {f: float(np.mean([v["ranking"].index(f) + 1 for v in ms1.values() if f in v["ranking"]])) for f in L.FAMILIES}
    wins_by_class = {c: {f: sum(1 for v in ms1.values() if v["class"] == c and v["winner"] == f) for f in L.FAMILIES} for c in CL}
    out["H-MS1"] = {"per_symbol": ms1, "part1": [k1, n1, s1], "part2": [k2, n2, s2], "part3": [k3, n3, s3],
                    "part2_ex": list(_share([v["part2_ex"] for v in ms1.values()])), "part3_ex": list(_share([v["part3_ex"] for v in ms1.values()])),
                    "print_part1": list(_share([v["print_part1"] for v in ms1.values()])), "print_part2": list(_share([v["print_part2"] for v in ms1.values()])),
                    "median_ks_ratio": float(np.nanmedian([v["ks_ratio"] for v in ms1.values() if v["ks_ratio"] is not None])),
                    "mean_rank": fam_rank, "wins_by_class": wins_by_class, "criterion": crit1, "p": p_ms1}

    # ---------------- H-MS2
    ms2 = {}
    for s_, r in R.items():
        z = r["size"]["print"]["test"]
        zo = r["size"]["order"]["test"]
        zf = r["size"]["print"]["fit"]
        zx = r["size"]["print"]["test_ex"]
        if "powerlaw" not in z:
            continue
        zeta = z["powerlaw"]["zeta"]
        ms2[s_] = {"class": classes[s_], "zeta": zeta, "xmin": z["powerlaw"]["xmin"], "n_tail": z["powerlaw"]["n_tail"], "p_tail": z["powerlaw"]["p_tail"],
                   "ks": z["powerlaw"]["ks"], "p_boot": z.get("p_boot"), "hill": z["hill_zeta"], "R": z["vuong"]["R"], "p_vuong": z["vuong"]["p"],
                   "in_range": 1 <= zeta <= 2, "not_rejected": (z.get("p_boot") or 0) > 0.1,
                   "preferred": z["vuong"]["R"] > 0 and z["vuong"]["p"] < 0.05,
                   "zeta_fit": _g(zf, "powerlaw", "zeta"), "zeta_ex": _g(zx, "powerlaw", "zeta"),
                   "zeta_order": _g(zo, "powerlaw", "zeta"), "p_boot_order": zo.get("p_boot"), "R_order": _g(zo, "vuong", "R"), "p_vuong_order": _g(zo, "vuong", "p"),
                   "median_usd": z["median"], "p99_usd": z["p99"]}
        ms2[s_]["a_ok"] = ms2[s_]["in_range"] and ms2[s_]["not_rejected"]
        zo_z = ms2[s_]["zeta_order"]
        ms2[s_]["a_ok_order"] = zo_z is not None and 1 <= zo_z <= 2 and (ms2[s_]["p_boot_order"] or 0) > 0.1
        ms2[s_]["pref_order"] = (ms2[s_]["R_order"] or -1) > 0 and (ms2[s_]["p_vuong_order"] or 1) < 0.05
    ka, na, sa = _share([v["a_ok"] for v in ms2.values()])
    kb, nb, sb = _share([v["preferred"] for v in ms2.values()])
    crit2 = sa > 0.7 and sb > 0.5
    p_ms2 = max(_binom_p(ka, na, 0.7), _binom_p(kb, nb, 0.5))
    zf_ = np.array([[v["zeta_fit"], v["zeta"]] for v in ms2.values() if v["zeta_fit"] is not None])
    out["H-MS2"] = {"per_symbol": ms2, "a": [ka, na, sa], "b": [kb, nb, sb], "in_range": list(_share([v["in_range"] for v in ms2.values()])),
                    "not_rejected": list(_share([v["not_rejected"] for v in ms2.values()])),
                    "order_a": list(_share([v["a_ok_order"] for v in ms2.values()])), "order_b": list(_share([v["pref_order"] for v in ms2.values()])),
                    "zeta_median": float(np.median([v["zeta"] for v in ms2.values()])),
                    "zeta_order_median": float(np.nanmedian([v["zeta_order"] for v in ms2.values() if v["zeta_order"] is not None])),
                    "fit_test_spearman": float(stats.spearmanr(zf_[:, 0], zf_[:, 1])[0]) if len(zf_) > 3 else None,
                    "fit_test_median_abs_diff": float(np.median(np.abs(zf_[:, 0] - zf_[:, 1]))) if len(zf_) else None,
                    "criterion": crit2, "p": p_ms2}

    # ---------------- H-MS3
    ms3 = {}
    for s_, r in R.items():
        g = r["signs"]
        o, pr = g["order"], g["print"]
        c1 = g["clock_1s"]
        ms3[s_] = {"class": classes[s_], "eligible": r["n_prints"] >= MIN_TRADES_MS3, "n_orders_test": o["n"]["test"],
                   "acf1": o["test"]["acf"]["1"], "acf10": o["test"]["acf"]["10"], "acf100": o["test"]["acf"]["100"], "acf1000": o["test"]["acf"]["1000"],
                   "gamma": o["test"]["power"]["gamma"], "c": o["test"]["power"]["c"], "gamma_fit": o["fit"]["power"]["gamma"], "tau_exp_fit": o["fit"]["exp"]["tau"],
                   "mse_power": o["heldout_test"]["mse_power"], "mse_exp": o["heldout_test"]["mse_exp"], "preferred": o["heldout_test"]["power_preferred"],
                   "preferred_ex": o["heldout_test_ex"]["power_preferred"], "gamma_ex": o["test_ex"]["power"]["gamma"],
                   "pacf_pos_11_50": o["test"]["pacf_sig_pos_11_50"], "pacf_neg_11_50": o["test"]["pacf_sig_neg_11_50"],
                   "print_gamma": pr["test"]["power"]["gamma"], "print_acf1": pr["test"]["acf"]["1"], "print_preferred": pr["heldout_test"]["power_preferred"],
                   "print_pacf_pos_11_50": pr["test"]["pacf_sig_pos_11_50"],
                   "hurst": c1["test"]["hurst_dfa"], "hurst_count": c1["test"]["hurst_dfa_count"], "hurst_fit": c1["fit"]["hurst_dfa"], "hurst_ex": c1["test_ex"]["hurst_dfa"],
                   "flow1s_acf1": c1["test"]["acf"]["1"], "flow10s_acf1": g["clock_10s"]["test"]["acf"]["1"],
                   "signed_notional_acf1": g["order_signed_notional"]["test"]["acf"]["1"], "signed_notional_gamma": g["order_signed_notional"]["test"]["power"]["gamma"]}
        v = ms3[s_]
        v["ok"] = bool(v["preferred"]) and v["gamma"] is not None and 0 < v["gamma"] < 1
        v["ok_ex"] = bool(v["preferred_ex"]) and v["gamma_ex"] is not None and 0 < v["gamma_ex"] < 1
        v["print_ok"] = bool(v["print_preferred"]) and v["print_gamma"] is not None and 0 < v["print_gamma"] < 1
    el = {s_: v for s_, v in ms3.items() if v["eligible"]}
    k3a, n3a, s3a = _share([v["ok"] for v in el.values()])
    Hs = np.array([v["hurst"] for v in el.values()])
    bH = np.array([rng.choice(Hs, len(Hs)).mean() for _ in range(B_BOOT)])
    hurst = {"mean": float(Hs.mean()), "median": float(np.median(Hs)), "lo": float(np.quantile(bH, 0.025)), "hi": float(np.quantile(bH, 0.975)),
             "p_le_half": float(np.mean(bH <= 0.5)), "share_above": float(np.mean(Hs > 0.5)), "n": int(len(Hs))}
    Hc = np.array([v["hurst_count"] for v in el.values()])
    bHc = np.array([rng.choice(Hc, len(Hc)).mean() for _ in range(B_BOOT)])
    hurst_count = {"mean": float(Hc.mean()), "lo": float(np.quantile(bHc, 0.025)), "hi": float(np.quantile(bHc, 0.975)), "share_above": float(np.mean(Hc > 0.5))}
    Hx = np.array([v["hurst_ex"] for v in el.values()])
    crit3 = s3a > 0.8 and hurst["lo"] > 0.5
    p_ms3 = max(_binom_p(k3a, n3a, 0.8), hurst["p_le_half"])
    out["H-MS3"] = {"per_symbol": ms3, "a": [k3a, n3a, s3a], "a_ex": list(_share([v["ok_ex"] for v in el.values()])),
                    "print_a": list(_share([v["print_ok"] for v in el.values()])), "hurst": hurst, "hurst_count": hurst_count,
                    "hurst_ex_mean": float(Hx.mean()), "preferred_share": list(_share([v["preferred"] for v in el.values()])),
                    "gamma_in_01": list(_share([v["gamma"] is not None and 0 < v["gamma"] < 1 for v in el.values()])),
                    "pacf_any_pos_beyond10": list(_share([(v["pacf_pos_11_50"] or 0) >= 4 for v in el.values()])),
                    "gamma_median_by_class": {c: float(np.median([v["gamma"] for v in el.values() if v["class"] == c])) for c in CL if any(v["class"] == c for v in el.values())},
                    "criterion": crit3, "p": p_ms3}

    # ---------------- H-MS4
    ms4 = {}
    for s_, r in R.items():
        row = {"class": classes[s_]}
        allok = True
        for b in ("10s", "60s"):
            V = r["volume"][b]
            ll = V["heldout_test"]["ll"]
            n_pos = max(V["heldout_test"]["n_pos"], 1)
            dist_ok = V["heldout_test"]["heavy_beats_gamma"]
            acf_ok = bool(V["acf"]["heldout_test"]["power_preferred"])
            sp = V["spearman_absret_test"]["rho"]
            sp_ok = sp is not None and sp > 0
            row[b] = {"zero_share": V["zero_share"]["test"], "d_gg_vs_gamma": (ll["gengamma"] - ll["gamma"]) / n_pos if ll.get("gengamma") is not None else None,
                      "d_ln_vs_gamma": (ll["lognormal"] - ll["gamma"]) / n_pos, "dist_ok": dist_ok, "acf_gamma": V["acf"]["test"]["power"]["gamma"],
                      "acf1": V["acf"]["test"]["acf"]["1"], "acf_ok": acf_ok, "spearman": sp, "sp_ok": sp_ok,
                      "dist_ok_ex": V["heldout_test_ex"]["heavy_beats_gamma"], "acf_ok_ex": bool(V["acf"]["heldout_test_ex"]["power_preferred"]),
                      "spearman_ex": V["spearman_absret_test_ex"]["rho"], "ok": dist_ok and acf_ok and sp_ok}
            allok = allok and row[b]["ok"]
        row["ok"] = allok
        row["ok_ex"] = all(row[b]["dist_ok_ex"] and row[b]["acf_ok_ex"] and (row[b]["spearman_ex"] or -1) > 0 for b in ("10s", "60s"))
        ms4[s_] = row
    k4, n4, s4 = _share([v["ok"] for v in ms4.values()])
    crit4 = s4 > 0.8
    p_ms4 = _binom_p(k4, n4, 0.8)
    parts4 = {f"{b}_{x}": list(_share([v[b][x] for v in ms4.values()])) for b in ("10s", "60s") for x in ("dist_ok", "acf_ok", "sp_ok")}
    out["H-MS4"] = {"per_symbol": ms4, "share": [k4, n4, s4], "share_ex": list(_share([v["ok_ex"] for v in ms4.values()])), "parts": parts4,
                    "criterion": crit4, "p": p_ms4}

    # ---------------- H-MS9
    nblk = 144
    per = {}
    variants = {"primary": {}, "no_tod": {}, "ex_cpi": {}}
    for s_, a in A.items():
        z = np.load(CACHE / f"{s_}_aft.npz")
        ok = z["ok"]
        acd_keys = [k for k in z.files if k.startswith("ll_acd_")]
        best = max(acd_keys, key=lambda k: float(z[k][ok].mean()))
        dif = (z["ll_aft"] - z[best])[ok]
        dif2 = (z["ll_aft_notod"] - z[best])[ok]
        st = z["start_ms"][ok]
        blk = np.clip((st - HALF) // 300_000, 0, nblk - 1).astype(int)
        cpi = (st >= CPI[0]) & (st < CPI[1])
        variants["primary"][s_] = (np.bincount(blk, dif, nblk), np.bincount(blk, minlength=nblk).astype(float))
        variants["no_tod"][s_] = (np.bincount(blk, dif2, nblk), np.bincount(blk, minlength=nblk).astype(float))
        variants["ex_cpi"][s_] = (np.bincount(blk[~cpi], dif[~cpi], nblk), np.bincount(blk[~cpi], minlength=nblk).astype(float))
        per[s_] = {"class": classes[s_], "acd_best": best[7:], "ll_acd": float(z[best][ok].mean()), "ll_aft": float(z["ll_aft"][ok].mean()),
                   "diff": float(dif.mean()), "diff_no_tod": float(dif2.mean()), "dist": a["best"]["dist"], "sigma": a["best"]["sigma"], "has_l2": a["has_l2"],
                   "top_features": list(a["importance_gain"])[:5], "n_test": int(ok.sum())}
    cls9 = {}
    for c in CL:
        for vname, vv in variants.items():
            sub = {s_: x for s_, x in vv.items() if classes[s_] == c}
            if sub:
                cls9.setdefault(c, {})[vname] = _block_boot(sub, B_BOOT, rng, nblk)
        sub = {s_: x for s_, x in variants["primary"].items() if classes[s_] == c and per[s_]["has_l2"]}
        if sub:
            cls9.setdefault(c, {})["l2_only"] = _block_boot(sub, B_BOOT, rng, nblk)
    wins9 = [c for c in CL if c in cls9 and cls9[c]["primary"]["lo"] > 0]
    crit9 = len(wins9) >= 2
    pcl = sorted(cls9[c]["primary"]["p_le0"] for c in cls9)
    p_ms9 = pcl[1] if len(pcl) >= 2 else 1.0
    out["H-MS9"] = {"per_symbol": per, "by_class": cls9, "classes_won": wins9, "criterion": crit9, "p": p_ms9,
                    "share_symbols_aft_better": list(_share([v["diff"] > 0 for v in per.values()]))}

    # ---------------- Holm and verdicts
    pv = {h: out[h]["p"] for h in ("H-MS1", "H-MS2", "H-MS3", "H-MS4", "H-MS9")}
    ph = holm(pv)
    verdicts = {}
    for h in pv:
        crit = out[h]["criterion"]
        verdicts[h] = "supports" if crit and ph[h] < 0.05 else "inconclusive" if crit else "contradicts"
    out["holm"] = {"p": pv, "p_holm": ph}
    out["verdicts"] = verdicts

    # ---------------- params JSON + parquet
    params = {"schema_version": 1, "date": DATE, "created": datetime.now(UTC).isoformat(timespec="seconds"), "schema": PARAMS_SCHEMA,
              "method": METHOD, "units": {"duration": "seconds", "size": "USD notional (price x size)", "volume": "USD notional per clock bin",
                                          "time": "ms since epoch UTC", "halves": {"fit": [DAY0, HALF], "test": [HALF, DAY1]}},
              "classes": {c: sorted(s_ for s_ in R if classes[s_] == c) for c in CL}, "symbols": {}}
    for s_, r in R.items():
        params["symbols"][s_] = {"class": classes[s_], "n_prints": r["n_prints"], "n_orders": r["n_orders"], "n_events": r["n_events"],
                                 "halves": {h: _half_params(r, A.get(s_), h) for h in ("fit", "test")}}
    med = {}
    for c in CL:
        ss = [s_ for s_ in R if classes[s_] == c]
        if not ss:
            continue
        def m_(fn):
            vals = [fn(s_) for s_ in ss]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            return float(np.median(vals)) if vals else None
        P = params["symbols"]
        med[c] = {h: {"events_per_s": m_(lambda s_: P[s_]["halves"][h]["rates"]["events_per_s"]),
                      "prints_per_order_mean": m_(lambda s_: P[s_]["halves"][h]["rates"]["prints_per_order_mean"]),
                      "weibull_c_event": m_(lambda s_: _g(P[s_]["halves"][h], "durations", "event", "laws", "weibull", "params", "c")),
                      "lognormal_s_event": m_(lambda s_: _g(P[s_]["halves"][h], "durations", "event", "laws", "lognormal", "params", "s")),
                      "acd_weibull_alpha": m_(lambda s_: _g(P[s_]["halves"][h], "acd", "models", "weibull", "alpha")),
                      "acd_weibull_beta": m_(lambda s_: _g(P[s_]["halves"][h], "acd", "models", "weibull", "beta")),
                      "acd_weibull_k": m_(lambda s_: _g(P[s_]["halves"][h], "acd", "models", "weibull", "shape", "k")),
                      "size_zeta_print": m_(lambda s_: _g(P[s_]["halves"][h], "size", "print", "powerlaw", "zeta")),
                      "size_zeta_order": m_(lambda s_: _g(P[s_]["halves"][h], "size", "order", "powerlaw", "zeta")),
                      "sign_gamma_order": m_(lambda s_: _g(P[s_]["halves"][h], "signs", "order", "acf_power", "gamma")),
                      "sign_c_order": m_(lambda s_: _g(P[s_]["halves"][h], "signs", "order", "acf_power", "c")),
                      "sign_acf1_order": m_(lambda s_: _g(P[s_]["halves"][h], "signs", "order", "acf", "1")),
                      "hurst_dfa_1s": m_(lambda s_: _g(P[s_]["halves"][h], "signed_flow", "clock_1s", "hurst_dfa")),
                      "vol10s_p_zero": m_(lambda s_: _g(P[s_]["halves"][h], "volume", "10s", "p_zero")),
                      "vol60s_lognormal_s": m_(lambda s_: _g(P[s_]["halves"][h], "volume", "60s", "laws_positive", "lognormal", "params", "s"))}
                  for h in ("fit", "test")}
    params["class_medians"] = med
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(_jsonable(params), indent=1), encoding="utf-8")
    rows: list[dict] = []
    for s_, P in params["symbols"].items():
        for h, hp in P["halves"].items():
            _flatten("", hp, rows, {"symbol": s_, "class": P["class"], "half": h})
    for r_ in rows:
        r_.setdefault("text", None)
    pl.DataFrame(rows, schema={"symbol": pl.Utf8, "class": pl.Utf8, "half": pl.Utf8, "path": pl.Utf8, "value": pl.Float64, "text": pl.Utf8}).write_parquet(
        PARAMS_PATH.with_suffix(".parquet"))
    out["params_rows"] = len(rows)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".json").write_text(json.dumps(_jsonable(out), indent=1), encoding="utf-8")
    write_markdown(out, R, A)
    print(json.dumps(_jsonable({"verdicts": verdicts, "holm": out["holm"]}), indent=1))
    if record:
        _record(out)


def _record(out: dict) -> None:
    from ai_trader.research import hypotheses

    data = f"trades_raw {DATE}, {out['n_symbols']} symbols; FIT 00-12 UTC / TEST 12-24 UTC; research/reports/microstructure_laws_2026-11-11.md"
    for h in ("H-MS1", "H-MS2", "H-MS3", "H-MS4", "H-MS9"):
        if hypotheses.results(h):
            print(f"{h} already recorded, skipped")
            continue
        o = out[h]
        m = {"p_summary": out["holm"]["p"][h], "p_holm": out["holm"]["p_holm"][h], "criterion_met": o["criterion"]}
        if h == "H-MS1":
            m.update({"part1_ks_reject": o["part1"], "part2_nonexp_wins": o["part2"], "part3_ks_halved": o["part3"], "median_ks_ratio": o["median_ks_ratio"],
                      "part2_exCPI": o["part2_ex"], "part3_exCPI": o["part3_ex"], "print_level_part1": o["print_part1"], "print_level_part2": o["print_part2"],
                      "wins_by_class": o["wins_by_class"]})
            note = ("event level (distinct ms); part 3 fails: deseasonalised ACD residuals stay far from exponential "
                    f"(median KS ratio {o['median_ks_ratio']:.2f})")
        elif h == "H-MS2":
            m.update({"alpha_ok_share": o["a"], "vuong_pref_share": o["b"], "in_range": o["in_range"], "not_rejected": o["not_rejected"],
                      "zeta_median_print": o["zeta_median"], "zeta_median_order": o["zeta_order_median"], "order_level": [o["order_a"], o["order_b"]],
                      "fit_test_spearman": o["fit_test_spearman"]})
            note = "zeta = CCDF tail index (CSN pdf alpha - 1); print-level notional on TEST"
        elif h == "H-MS3":
            m.update({"ok_share": o["a"], "ok_share_exCPI": o["a_ex"], "print_level": o["print_a"], "hurst": o["hurst"], "hurst_count": o["hurst_count"],
                      "preferred_share": o["preferred_share"], "gamma_in_01": o["gamma_in_01"], "pacf_pos_beyond10": o["pacf_any_pos_beyond10"],
                      "gamma_median_by_class": o["gamma_median_by_class"]})
            note = "order-level signs, symbols with >= 20k prints; Hurst = DFA of 1 s signed notional"
        elif h == "H-MS4":
            m.update({"share": o["share"], "share_exCPI": o["share_ex"], "parts": o["parts"]})
            note = "10 s and 60 s notional; all three parts at both intervals"
        else:
            m.update({"by_class": o["by_class"], "classes_won": o["classes_won"], "share_symbols_aft_better": o["share_symbols_aft_better"]})
            note = "xgboost survival:aft vs best-on-TEST ACD(1,1); per-duration log-lik difference, 5-min block bootstrap"
        row = hypotheses.record(h, "clean_holdout", out["verdicts"][h], _jsonable(m), note=note, data=data)
        print("recorded", h, row["verdict"])


NOTES: dict[str, str] = {
    "intro": (
        "Every print of one full UTC day (2026-11-10, the CPI day) for 51 Bybit perps: 6.9k (CARV) to 2.1M (BTC) prints. Laws are fitted on "
        "00-12 UTC and scored on 12-24 UTC, which is a harder half: it holds the 13:30 CPI release and about twice the activity (majors 1.3 -> 2.5 "
        "arrival events per second). This is one day, so every number is a first measurement (confidence C).\n\n"
        "In short: arrivals are bursty and far from Poisson. A 3-component exponential mixture (or Burr XII) is the best marginal law, but deseasonalised "
        "ACD(1,1) residuals stay far from exponential, so H-MS1 fails on its third part. Trade-size tails are heavy (tail index 1.4-1.8) but curved: "
        "CSN rejects the pure power law and the log-normal is never worse (H-MS2 fails). Order signs have long memory: a power-law ACF with gamma "
        "0.45-0.75, and the 1 s signed-flow Hurst exponent is 0.53 with a CI above 0.5. That meets the H-MS3 criterion, but it is inconclusive after "
        "Holm. Volume is log-normal, persistent and tied to |returns|, but the power-law ACF part fails at 1 minute for too many symbols (H-MS4 just "
        "misses: 78% vs > 80%). The xgboost AFT beats the best ACD only for majors (H-MS9 fails)."),
    "data": (
        "Merging matters. A Bybit taker order is printed as one trade per maker it hits, all with the same millisecond and side: majors average 4 "
        "prints per order and TradFi perps 1.9. Treating prints as independent orders creates mechanical zero durations and a print-level sign ACF "
        "of 0.7-0.84 at lag 1 that comes from splitting, not from behaviour. Only 0.2-1% of milliseconds carry both sides, so arrival events "
        "(distinct ms) and orders are nearly the same series."),
    "ms1": (
        "Durations have a coefficient of variation of about 2.1-2.3 (an exponential has 1), and the exponential is rejected for every symbol "
        "(median KS 0.33). Out of sample, a non-exponential family wins for every symbol. The 3-component exponential mixture wins for 30 of 51 "
        "(mean rank 2.4), then Burr XII (3.0) and Weibull (3.3). The mixture reads as three regimes: about 25-32% of durations are "
        "bursts with a mean of about 5 ms, 35-45% have a mean of 0.2-0.4 s, and 25-38% are quiet gaps with a mean of 1.2-4.8 s (class medians, TEST "
        "fit). The generalized gamma nests Weibull and gamma but ranks only 5.0 out of sample: its extra shape overfits the FIT half.\n\n"
        "Part 3 fails clearly. Causal deseasonalisation (trailing 30-min mean duration) cuts the KS distance only from 0.33 to 0.29. The ACD residuals "
        "reach 0.22-0.26: a median ratio of 0.78 (exponential innovations 0.77, Weibull 0.78, generalized gamma 0.78), and only 2 of 51 symbols halve "
        "it (0 of 51 without 13-16 UTC). The fitted ACD sits at the integrated boundary (median alpha + beta = 0.99995, alpha about 0.1-0.2), and the "
        "unit-mean innovations stay strongly over-dispersed (Weibull k about 0.47-0.61). Conditional means absorb the slow clustering but not the "
        "millisecond bursts. A generator needs a burst component (a self-exciting or mixture innovation), not only a moving conditional mean."),
    "ms2": (
        "Print notional has a heavy upper tail. The CSN tail index zeta (CCDF exponent; CSN pdf alpha = zeta + 1) has medians of 1.84 for majors, 1.76 "
        "for alts and 1.39 for TradFi, and 35 of 51 symbols fall in [1, 2]. At order level the tails are heavier (zeta 1.1-1.4), because merging the "
        "prints of one order rebuilds large orders. But the pure power law is rejected by the bootstrap for 45 of 51 symbols (p <= 0.1). "
        "The Vuong test never prefers it to a log-normal truncated at x_min (0 of 51); the log-normal is significantly better or tied. With tens of "
        "thousands of tail points, any curvature in log-log space is detected. Lot-size discreteness adds to the KS distance (the five most "
        "common sizes are 6-8% of the tail for liquid symbols, 22% for KAS) but does not explain the Vuong result. The index is also unstable "
        "across halves (Spearman 0.27 between FIT and TEST zeta, median absolute change 0.33). For simulation, use the spliced law (log-normal "
        "body, Pareto tail above x_min) per symbol and half. Treat zeta as roughly 1.3-1.8 with a large uncertainty, not as a universal 1.5."),
    "ms3": (
        "Order signs persist strongly: ACF at lag 1 is 0.33-0.39 in every class, about 0.04-0.09 at lag 10 and 0.01-0.05 at lag 100. A power law "
        "predicts the held-out correlogram better than an exponential for every eligible symbol (48 of 48). The fitted gamma is 0.64 for majors, "
        "0.74 for alts and 0.45 for TradFi (longest memory). It is in (0, 1) for 44 of 48. The four failures (AAVE, DOT, FIL, ZEC) have gamma "
        "1.00-1.09, just outside. 39 of 48 symbols have 4 or more significant positive PACF lags in 11-50, so the memory is not an AR(1). In clock time "
        "the memory is diluted by size heterogeneity: 1 s signed notional has a lag-1 ACF of only 0.03-0.04, but DFA gives Hurst 0.531 [0.522, "
        "0.541] for notional and 0.579 [0.568, 0.590] for signed counts (every symbol above 0.5). The literal criterion is met (44/48 = 0.92 > 0.8; "
        "Hurst CI above 0.5). The share is not significantly above 0.8 (binomial p 0.025, Holm 0.12), so the pre-set rule gives inconclusive. "
        "Without 13-16 UTC the share drops to 37 of 48 (more gammas just above 1), while at print level it is 47 of 48."),
    "ms4": (
        "Volume per 10 s and per minute is log-normal-like. Out of sample the log-normal beats the gamma by 0.18-0.65 nats per bin (all 51 at "
        "10 s, 49 at 1 m), and the generalized gamma does nearly as well. The co-movement with |return| in the same bin holds for every symbol "
        "(Spearman 0.40-0.55). Log volume is persistent (lag-1 ACF 0.35-0.64) and decays slowly. The power law is the better held-out decay for "
        "46 of 51 at 10 s but only 42 of 51 at 1 minute, where 12 h give only 720 bins and a correlogram truncated at 2 h. 40 of 51 symbols pass all "
        "six parts (78%, below the registered 80%), and 31 of 51 without 13-16 UTC. The failures are about ACF shape, not about distribution or "
        "co-movement."),
    "ms9": (
        "The AFT model is the ML law of the next arrival: log T = mu(x) + sigma Z. For majors it beats the best ACD(1,1) by 0.071 nats per duration "
        "[0.063, 0.078] (still 0.063 without 13-16 UTC). The extreme-value (log-Weibull) error distribution wins validation for most symbols. "
        "The most useful features are trades in the last 10 s and 60 s, the last durations, and the number of prints in the last order "
        "(a big sweep is followed by fast follow-on trades). For alts the difference is -0.012 [-0.030, +0.012], and for TradFi -0.027 [-0.034, "
        "-0.019]: with a few thousand to tens of thousands of FIT durations, trees fitted on the quiet morning do not transfer to the busier "
        "afternoon better than a 4-parameter ACD. The registered hour-of-day feature is outside its FIT support in TEST; dropping it helps a little "
        "(alts -0.003, TradFi -0.016) and changes no class verdict. The three TradFi L2 symbols (NVDA, QQQ, XAU) gain +0.017 [0.010, 0.025], "
        "a hint that book features help where trades are sparse. The criterion holds in 1 of 3 classes, so H-MS9 is contradicted."),
    "generator": (
        "Targets for the event-level simulator v1 (agent C3). Each line is a held-out fact with its class range, and the parameter file has the "
        "per-symbol values.\n\n"
        "1. **Arrivals in event time, by order, not by print.** CV of durations about 2.2. The marginal is a 3-component exponential mixture "
        "(about 30% bursts at about 5 ms, about 40% at 0.2-0.4 s, about 30% gaps at 1-5 s). The intensity moves slowly (trailing 30-min level, "
        "doubling from FIT to TEST) and clustering is integrated (ACD alpha + beta near 1). Plain ACD leaves the burst component unexplained "
        "(KS ratio 0.78), so add a self-exciting burst term or draw innovations from the fitted mixture. The fitted ACDs are FILTERS, not "
        "generators: E[(alpha eps + beta)^2] (`acd.models.*.m2_factor`) is >= 1 for all 51 Weibull fits (median 1.20) and for 28 of 51 "
        "exponential / generalized-gamma fits (median 1.00), so simulated ACD paths explode or collapse (CV 3-35 instead of about 2). "
        "Generate arrivals from the mixture marginal on a slow intensity path (or a Hawkes process), and use the ACD as a one-step "
        "forecast only.\n"
        "2. **Prints per order.** Mean 1.9 (TradFi) to 4.0 (majors), heavy-tailed pmf (`rates.prints_per_order_pmf`). Split simulated orders into "
        "prints at the same ms.\n"
        "3. **Sizes.** Log-normal body plus a heavy tail: print zeta about 1.4-1.8, order zeta about 1.1-1.5 (`size.*.size_law`). Do not rely on "
        "a universal exponent.\n"
        "4. **Order signs.** Lag-1 ACF 0.33-0.39 and a power-law decay with gamma 0.45 (TradFi) to 0.75 (alts), still about 0.01-0.05 at lag 100, "
        "plus positive PACF beyond lag 10. `TranslatedSignModel` (`signs.order.sign_model`) reproduces the empirical short lags and the power-law "
        "tail (tests/test_laws.py). The pure fGn generator cannot reach a lag-1 ACF of 0.35 when gamma is above about 0.5. The v0 generator had "
        "near-zero flow autocorrelation, and this is the missing piece.\n"
        "5. **Clock-time signed flow.** Lag-1 ACF of 1 s signed notional only 0.03-0.04 but Hurst 0.53 (notional) and 0.58 (counts). Long sign "
        "memory combined with heavy-tailed sizes must give both.\n"
        "6. **Volume per bin.** Log-normal positive part, zero share 13-16% of 10 s bins for alts and TradFi (none for majors), lag-1 ACF of log "
        "volume 0.35-0.64 with slow decay, Spearman with |return| 0.40-0.55.\n"
        "7. **Conditional arrival model.** For majors, the saved xgboost AFT models (`aft_2026-11-10/<SYM>.json`) give a better next-arrival "
        "density than ACD. For alts and TradFi, use the ACD."),
    "caveats": (
        "- One day, one market state (a crypto sell-off on CPI day). The TEST half is about twice as active as FIT, so every FIT-fitted scale is "
        "off in TEST: this penalises all families alike and favours models with an adaptive level (ACD, AFT count features).\n"
        "- Deseasonalisation is a causal trailing 30-minute mean. It absorbs slow intensity swings, not a true time-of-day profile (one day cannot "
        "identify one). With more days, use an hour-of-week profile (features.seasonality) instead.\n"
        "- ACD persistence sits at the integrated bound for most symbols. The model is close to IACD, so its unconditional mean is not defined, and "
        "simulating it raw drifts to near-zero durations. `ACD.sample` therefore caps alpha + beta at 0.99 and anchors at the in-sample mean "
        "(a simulation-time regularisation, not a fitted quantity). Even capped, fits with m2_factor >= 1 stay far too dispersed.\n"
        "- Print-level duration fits are dominated by the 0.5 ms atom and are reported for completeness only.\n"
        "- CSN with 10^3-10^5 tail points rejects any curvature. Sizes are partly discrete (round lots).\n"
        "- The ACF noise-floor rule (bins above 2/sqrt(n) only) was set after a two-symbol smoke test (CARV, KAS), before the full run. "
        "A generalized gamma / Burr XII optimiser bug (stuck below the families they nest) was found and fixed before any verdict was recorded "
        "(diary 'mistake'). Duration and volume laws were refitted for all symbols.\n"
        "- The Hurst CI bootstraps symbols as if independent. Crypto flows co-move, so the true CI is wider, but 40 of 48 symbols are "
        "individually above 0.5 for notional and all 48 for counts.\n"
        "- The H-MS9 comparator is the ACD that is best on TEST (conservative toward the AFT). Spread and depth imbalance come from ob200 for 12 "
        "symbols only; the others use a trade-based effective-spread proxy. Hour of day is outside its FIT support (no-tod variant reported).\n"
        "- Holm runs over the five summary p-values. Hypotheses that fail get p = 1, which makes the step-down strict for H-MS3.\n"
        "- Classes: BNB is an alt here (task definition); in simulator v0 it was a major."),
    "rules": """
- id: R-MS-1
  kind: expect
  rule: "WHEN reasoning about or simulating aggressive trade arrivals on a Bybit perp THEN expect bursts, not a Poisson stream: durations have a coefficient of variation of about 2 and roughly a quarter to a third of order gaps are a few milliseconds"
  when: "inter-arrival times of taker orders (same-ms same-side prints merged)"
  then: "use a 3-component exponential mixture or Burr XII marginal with a slowly moving intensity; never an exponential law"
  scope: {assets: all, horizon: "1ms-1m", sides: both}
  effect: "exponential rejected 51/51 (median KS 0.33); mixture or Burr XII best held-out for 37/51; burst component weight 0.24-0.32 at ~5 ms"
  confidence: C
  evidence: [H-MS1]
  review: "a week of trades_raw with the same winners in >= 80% of symbol-days"
  status: provisional
- id: R-MS-2
  kind: avoid
  rule: "WHEN building a duration model for a generator or a fill-timing estimate THEN avoid relying on ACD(1,1) alone to remove clustering; its residuals keep most of the non-exponential shape"
  when: "conditional duration modelling, event time"
  then: "add a burst / self-exciting component or draw innovations from the fitted mixture; check residual KS"
  scope: {assets: all, horizon: "1ms-1m", sides: both}
  effect: "residual KS / raw KS median 0.78; halved for 2/51 symbols; ACD persistence ~1 (integrated); second-moment factor >= 1 for 51/51 Weibull fits, so they are filters, not generators"
  confidence: C
  evidence: [H-MS1]
  review: "a Hawkes or mixture-innovation ACD that halves the KS on >= 80% of symbols out of sample"
  status: provisional
- id: R-MS-3
  kind: expect
  rule: "WHEN taker orders have been mostly on one side THEN expect the next orders to stay on that side more often than not, with memory decaying slowly (power law) over hundreds of orders"
  when: "order-level trade signs (prints of one order merged)"
  then: "treat order flow as persistent in event time: lag-1 sign autocorrelation about 0.35, still 0.01-0.05 at lag 100; longest memory in TradFi perps"
  scope: {assets: all, horizon: "1-1000 orders", sides: both}
  effect: "power law beats exponential held-out for 48/48; gamma 0.64 majors, 0.74 alts, 0.45 TradFi; 1 s signed-flow Hurst 0.53 (CI 0.52-0.54)"
  confidence: C
  evidence: [H-MS3]
  review: "a week of trades_raw; gamma in (0, 1) for >= 80% of symbol-days"
  status: provisional
- id: R-MS-4
  kind: avoid
  rule: "WHEN counting prints as trades (signs, durations, sizes) THEN avoid treating same-millisecond same-side prints as separate orders"
  when: "any statistic built from Bybit public trades"
  then: "merge same-ms same-side prints into one order first; print-level sign autocorrelation and zero durations are artefacts of matching against several makers"
  scope: {assets: all, horizon: "tick", sides: both}
  effect: "1.9-4.0 prints per order; print-level sign ACF(1) 0.70-0.84 vs 0.33-0.39 for orders"
  confidence: C
  evidence: [H-MS1, H-MS3]
  review: "confirm on another venue or day"
  status: provisional
- id: R-MS-5
  kind: avoid
  rule: "WHEN modelling the tail of trade or order notional THEN avoid a pure Pareto with a universal exponent; prefer a log-normal body spliced to a heavy tail fitted per symbol, with the tail index treated as uncertain"
  when: "size distributions for simulation, impact or liquidation scenarios"
  then: "use the per-symbol size_law; expect tail index 1.3-1.8 (prints) and about 1.1-1.5 (orders) but not stable between half-days"
  scope: {assets: all, horizon: "tick", sides: both}
  effect: "CSN rejects the power law for 45/51; Vuong never prefers it to a log-normal (0/51); FIT-TEST Spearman of zeta 0.27"
  confidence: C
  evidence: [H-MS2]
  review: "multi-day tails with a stable index (Spearman > 0.7 between days)"
  status: provisional
- id: R-MS-6
  kind: expect
  rule: "WHEN traded volume in a 10 s or 1 min interval is high THEN expect the absolute return in the same interval to be large and the elevated volume to persist for several intervals"
  when: "notional volume per clock bin"
  then: "read volume bursts as volatility bursts; quote distances based on calm-period volume are too tight during them"
  scope: {assets: all, horizon: "10s-1h", sides: both}
  effect: "Spearman(volume, |return|) 0.40-0.55, positive for 51/51; lag-1 ACF of log volume 0.35-0.64; log-normal beats gamma for 49-51/51"
  confidence: C
  evidence: [H-MS4]
  review: "a week of trades_raw"
  status: provisional
- id: R-MS-7
  kind: prefer
  rule: "WHEN predicting the time to the next trade on BTC, ETH, SOL, XRP or DOGE THEN prefer the gradient-boosted AFT model (recent trade counts, last durations, prints in the last order) over ACD; for alts and TradFi perps keep the ACD"
  when: "next-arrival density, e.g. fill-timing or quote-refresh logic"
  then: "use data/models/microstructure_laws/aft_2026-11-10/<SYM>.json for majors only"
  scope: {assets: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT], horizon: "1ms-1m", sides: both}
  effect: "+0.071 nats per duration [0.063, 0.078] for majors; alts -0.012 [-0.030, 0.012]; TradFi -0.027 [-0.034, -0.019]"
  confidence: C
  evidence: [H-MS9]
  review: "refit on several days; the majors gain must survive a day-ahead test"
  status: provisional
""",
}


def _tbl(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(x if isinstance(x, str) else _f(x) for x in r) + " |")
    return "\n".join(out)


def _med(vals) -> float | None:
    v = [x for x in vals if x is not None and isinstance(x, int | float) and math.isfinite(x)]
    return float(np.median(v)) if v else None


def write_markdown(out: dict, R: dict, A: dict) -> None:
    CL = ("major", "alt", "tradfi")
    v = out["verdicts"]
    ph = out["holm"]["p_holm"]
    pv = out["holm"]["p"]
    m1, m2, m3, m4, m9 = (out[h] for h in ("H-MS1", "H-MS2", "H-MS3", "H-MS4", "H-MS9"))
    md: list[str] = []
    md.append("# Trade-level laws of trading activity: durations, sizes, sign memory, volume, ML durations (H-MS1, H-MS2, H-MS3, H-MS4, H-MS9)\n")
    md.append(f"_data: trades_raw {DATE} (every Bybit linear print, {out['n_symbols']} symbols) | FIT 00:00-12:00 UTC, TEST 12:00-24:00 UTC | "
              "segment: clean_holdout | code: src/ai_trader/sim/laws.py, src/ai_trader/sim/l2_features.py, src/ai_trader/research/microstructure_laws_study.py | "
              "tests: tests/test_laws.py | params: data/models/microstructure_laws/params_2026-11-10.json (+ .parquet, aft_2026-11-10/)_\n")
    md.append(NOTES.get("intro", "") + "\n")
    md.append("## Verdicts\n")
    rows = [
        ["H-MS1", "exponential rejected for every symbol; non-exponential family wins TEST log-lik for > 80%; ACD residual KS <= half the raw KS for > 80%",
         f"KS reject {m1['part1'][0]}/{m1['part1'][1]}; non-exp wins {m1['part2'][0]}/{m1['part2'][1]}; KS halved {m1['part3'][0]}/{m1['part3'][1]} "
         f"(median ratio {m1['median_ks_ratio']:.2f})", f"{_f(pv['H-MS1'])} / {_f(ph['H-MS1'])}", f"**{v['H-MS1']}**"],
        ["H-MS2", "tail index in [1, 2] and CSN not rejected for > 70%; power law preferred to log-normal (Vuong) for > 50%",
         f"in range and not rejected {m2['a'][0]}/{m2['a'][1]} (in range {m2['in_range'][0]}, not rejected {m2['not_rejected'][0]}); "
         f"preferred {m2['b'][0]}/{m2['b'][1]}; median zeta {m2['zeta_median']:.2f}", f"{_f(pv['H-MS2'])} / {_f(ph['H-MS2'])}", f"**{v['H-MS2']}**"],
        ["H-MS3", "power law preferred and 0 < gamma < 1 for > 80% of symbols with >= 20k prints; pooled Hurst CI above 0.5",
         f"{m3['a'][0]}/{m3['a'][1]} symbols; Hurst (1 s signed notional) {m3['hurst']['mean']:.3f} [{m3['hurst']['lo']:.3f}, {m3['hurst']['hi']:.3f}]",
         f"{_f(pv['H-MS3'])} / {_f(ph['H-MS3'])}", f"**{v['H-MS3']}**"],
        ["H-MS4", "heavy law beats gamma, power-law ACF, Spearman(volume, |r|) > 0, at 10 s and 1 m, for > 80%",
         f"{m4['share'][0]}/{m4['share'][1]} symbols pass all six", f"{_f(pv['H-MS4'])} / {_f(ph['H-MS4'])}", f"**{v['H-MS4']}**"],
        ["H-MS9", "AFT minus best ACD per-duration log-lik > 0 with CI excluding 0 in >= 2 of 3 classes",
         "; ".join(f"{c} {m9['by_class'][c]['primary']['diff']:+.3f} [{m9['by_class'][c]['primary']['lo']:+.3f}, {m9['by_class'][c]['primary']['hi']:+.3f}]"
                   for c in CL if c in m9["by_class"]), f"{_f(pv['H-MS9'])} / {_f(ph['H-MS9'])}", f"**{v['H-MS9']}**"],
    ]
    md.append(_tbl(["id", "registered success criterion", "key numbers (TEST half)", "p summary / Holm", "verdict"], rows) + "\n")
    md.append("p summary: intersection-union p of the criterion's parts (one-sided binomial tests of each share against its threshold; bootstrap p for "
              "Hurst and AFT), Holm over the five. Supports needs the literal criterion and Holm p < 0.05.\n")

    # data
    md.append("## Data and levels\n")
    drow = []
    for c in CL:
        ss = [s_ for s_ in R if asset_class(s_) == c]
        if not ss:
            continue
        drow.append([c, len(ss), _med([R[s_]["n_prints"] for s_ in ss]), _med([R[s_]["n_orders"] for s_ in ss]), _med([R[s_]["n_events"] for s_ in ss]),
                     _med([R[s_]["rates"]["fit"]["prints_per_order_mean"] for s_ in ss]), _med([R[s_]["rates"]["fit"]["events_per_s"] for s_ in ss]),
                     _med([R[s_]["rates"]["test"]["events_per_s"] for s_ in ss]), _med([R[s_]["rates"]["test"]["two_sided_ms_share"] for s_ in ss])])
    md.append(_tbl(["class", "symbols", "median prints", "median orders", "median events", "prints per order (FIT)", "events/s FIT", "events/s TEST",
                    "ms with both sides"], drow) + "\n")
    md.append(NOTES.get("data", "") + "\n")
    md.append("## Methods\n")
    for k, x in METHOD.items():
        md.append(f"- **{k}**: {x}")
    md.append("")

    # H-MS1
    md.append("## H-MS1: inter-arrival times\n")
    md.append(NOTES.get("ms1", "") + "\n")
    rows = []
    for c in CL:
        ss = {s_: x for s_, x in m1["per_symbol"].items() if x["class"] == c}
        if not ss:
            continue
        wins = m1["wins_by_class"][c]
        top = ", ".join(f"{f} {n}" for f, n in sorted(wins.items(), key=lambda kv: -kv[1]) if n)
        rows.append([c, len(ss), f"{sum(x['part1'] for x in ss.values())}/{len(ss)}", f"{sum(x['part2'] for x in ss.values())}/{len(ss)}",
                     f"{sum(x['part3'] for x in ss.values())}/{len(ss)}", top, _med([x["ks_raw"] for x in ss.values()]),
                     _med([x["ks_deseason"] for x in ss.values()]), _med([x["ks_resid"] for x in ss.values()]), _med([x["ks_ratio"] for x in ss.values()]),
                     _med([x["cv"] for x in ss.values()])])
    md.append(_tbl(["class", "n", "KS rejects exp", "non-exp wins", "KS halved", "TEST winners", "KS raw", "KS deseasonalised", "KS ACD resid",
                    "ratio", "CV of durations"], rows) + "\n")
    md.append("Mean held-out rank of each family over all symbols (1 = best TEST log-likelihood, event level): "
              + ", ".join(f"{f} {r:.2f}" for f, r in sorted(m1["mean_rank"].items(), key=lambda kv: kv[1])) + ".\n")
    md.append(f"Robustness: without 13:00-16:00 the non-exponential winner share is {m1['part2_ex'][0]}/{m1['part2_ex'][1]} and the KS-halved share "
              f"{m1['part3_ex'][0]}/{m1['part3_ex'][1]}. Print level (0 ms gaps set to 0.5 ms): exponential rejected {m1['print_part1'][0]}/{m1['print_part1'][1]}, "
              f"non-exponential winner {m1['print_part2'][0]}/{m1['print_part2'][1]} (print-level fits are dominated by the 0.5 ms atom and are not usable "
              "as a continuous law; simulate orders, then split them into prints with the prints-per-order pmf).\n")
    rows = []
    for s_, x in sorted(m1["per_symbol"].items(), key=lambda kv: (kv[1]["class"], kv[0])):
        acd = x["acd"].get(x["acd_best_fit"] or "", {})
        sh = acd.get("shape") or {}
        rows.append([s_, x["class"], x["n_test"], x["ks_raw"], x["winner"], ", ".join(x["ranking"][1:3]), x["ll_gain_vs_exp"], x["acd_best_fit"],
                     acd.get("alpha"), acd.get("beta"), ", ".join(f"{k} {_f(val)}" for k, val in sh.items()) or "-", x["ks_resid"], x["ks_ratio"]])
    md.append("<details><summary>Per symbol (event level, TEST half)</summary>\n")
    md.append(_tbl(["symbol", "class", "TEST durations", "KS raw vs exp", "winner", "2nd, 3rd", "log-lik gain vs exp per duration", "ACD (best FIT)",
                    "alpha", "beta", "innovation shape", "KS resid", "ratio"], rows) + "\n\n</details>\n")

    # H-MS2
    md.append("## H-MS2: trade-size tails\n")
    md.append(NOTES.get("ms2", "") + "\n")
    rows = []
    for c in CL:
        ss = {s_: x for s_, x in m2["per_symbol"].items() if x["class"] == c}
        if not ss:
            continue
        rows.append([c, len(ss), _med([x["zeta"] for x in ss.values()]), _med([x["hill"] for x in ss.values()]), _med([x["p_tail"] for x in ss.values()]),
                     f"{sum(x['in_range'] for x in ss.values())}/{len(ss)}", f"{sum(x['not_rejected'] for x in ss.values())}/{len(ss)}",
                     f"{sum(x['preferred'] for x in ss.values())}/{len(ss)}", _med([x["zeta_order"] for x in ss.values()]),
                     f"{sum(x['a_ok_order'] for x in ss.values())}/{len(ss)}", f"{sum(x['pref_order'] for x in ss.values())}/{len(ss)}"])
    md.append(_tbl(["class", "n", "zeta (print, TEST)", "Hill", "tail share", "zeta in [1, 2]", "CSN p > 0.1", "Vuong prefers PL", "zeta (order)",
                    "order: in range and not rejected", "order: PL preferred"], rows) + "\n")
    md.append(f"FIT-vs-TEST stability (print level): Spearman of zeta across symbols {_f(m2['fit_test_spearman'])}, median |zeta_FIT - zeta_TEST| "
              f"{_f(m2['fit_test_median_abs_diff'])}.\n")
    rows = [[s_, x["class"], x["zeta"], x["zeta_fit"], x["zeta_ex"], x["xmin"], x["n_tail"], x["ks"], x["p_boot"], x["R"], x["p_vuong"], x["hill"],
             x["zeta_order"], x["p_boot_order"], x["median_usd"], x["p99_usd"]]
            for s_, x in sorted(m2["per_symbol"].items(), key=lambda kv: (kv[1]["class"], kv[0]))]
    md.append("<details><summary>Per symbol</summary>\n")
    md.append(_tbl(["symbol", "class", "zeta TEST", "zeta FIT", "zeta ex-CPI", "x_min USD", "n tail", "KS", "CSN p", "Vuong R", "Vuong p", "Hill",
                    "zeta order", "CSN p order", "median USD", "p99 USD"], rows) + "\n\n</details>\n")

    # H-MS3
    md.append("## H-MS3: sign memory and signed flow\n")
    md.append(NOTES.get("ms3", "") + "\n")
    rows = []
    for c in CL:
        ss = {s_: x for s_, x in m3["per_symbol"].items() if x["class"] == c and x["eligible"]}
        if not ss:
            continue
        rows.append([c, len(ss), _med([x["acf1"] for x in ss.values()]), _med([x["acf10"] for x in ss.values()]), _med([x["acf100"] for x in ss.values()]),
                     _med([x["acf1000"] for x in ss.values()]), _med([x["gamma"] for x in ss.values()]), f"{sum(bool(x['preferred']) for x in ss.values())}/{len(ss)}",
                     f"{sum(x['ok'] for x in ss.values())}/{len(ss)}", _med([x["pacf_pos_11_50"] for x in ss.values()]), _med([x["print_acf1"] for x in ss.values()]),
                     _med([x["print_gamma"] for x in ss.values()]), _med([x["hurst"] for x in ss.values()]), _med([x["hurst_count"] for x in ss.values()]),
                     _med([x["flow1s_acf1"] for x in ss.values()]), _med([x["signed_notional_acf1"] for x in ss.values()])])
    md.append(_tbl(["class", "n (>= 20k prints)", "sign ACF 1", "ACF 10", "ACF 100", "ACF 1000", "gamma", "power preferred", "criterion",
                    "PACF + lags 11-50", "print ACF 1", "print gamma", "Hurst 1 s notional", "Hurst 1 s count", "1 s flow ACF 1", "order notional ACF 1"], rows) + "\n")
    h_ = m3["hurst"]
    md.append(f"Pooled Hurst (DFA of 1 s signed notional, TEST, {h_['n']} symbols): mean {h_['mean']:.3f}, 95% CI [{h_['lo']:.3f}, {h_['hi']:.3f}], "
              f"share above 0.5 {h_['share_above']:.2f}; signed COUNT: mean {m3['hurst_count']['mean']:.3f} [{m3['hurst_count']['lo']:.3f}, "
              f"{m3['hurst_count']['hi']:.3f}], share above 0.5 {m3['hurst_count']['share_above']:.2f}. Ex-CPI mean Hurst {m3['hurst_ex_mean']:.3f}. "
              f"Order-sign criterion without 13-16 UTC: {m3['a_ex'][0]}/{m3['a_ex'][1]}; at print level: {m3['print_a'][0]}/{m3['print_a'][1]}. "
              f"PACF with >= 4 significant positive lags in 11-50 (binomial 2% under white noise): {m3['pacf_any_pos_beyond10'][0]}/{m3['pacf_any_pos_beyond10'][1]}.\n")
    rows = [[s_, x["class"], x["n_orders_test"], x["acf1"], x["acf10"], x["acf100"], x["gamma"], x["gamma_fit"], x["mse_power"], x["mse_exp"], x["preferred"],
             x["pacf_pos_11_50"], x["print_gamma"], x["hurst"], x["hurst_count"]]
            for s_, x in sorted(m3["per_symbol"].items(), key=lambda kv: (kv[1]["class"], kv[0]))]
    md.append("<details><summary>Per symbol (order signs, TEST)</summary>\n")
    md.append(_tbl(["symbol", "class", "TEST orders", "ACF 1", "ACF 10", "ACF 100", "gamma TEST", "gamma FIT", "held-out MSE power", "MSE exp", "power preferred",
                    "PACF + 11-50", "print gamma", "Hurst notional", "Hurst count"], rows) + "\n\n</details>\n")

    # H-MS4
    md.append("## H-MS4: volume per interval\n")
    md.append(NOTES.get("ms4", "") + "\n")
    rows = []
    for c in CL:
        ss = {s_: x for s_, x in m4["per_symbol"].items() if x["class"] == c}
        if not ss:
            continue
        for b in ("10s", "60s"):
            rows.append([c, b, len(ss), _med([x[b]["zero_share"] for x in ss.values()]), _med([x[b]["d_ln_vs_gamma"] for x in ss.values()]),
                         _med([x[b]["d_gg_vs_gamma"] for x in ss.values()]), f"{sum(x[b]['dist_ok'] for x in ss.values())}/{len(ss)}",
                         _med([x[b]["acf1"] for x in ss.values()]), _med([x[b]["acf_gamma"] for x in ss.values()]), f"{sum(x[b]['acf_ok'] for x in ss.values())}/{len(ss)}",
                         _med([x[b]["spearman"] for x in ss.values()]), f"{sum(x[b]['sp_ok'] for x in ss.values())}/{len(ss)}",
                         f"{sum(x['ok'] for x in ss.values())}/{len(ss)}"])
    md.append(_tbl(["class", "bin", "n", "zero share", "log-normal minus gamma (per bin)", "gen. gamma minus gamma", "heavy beats gamma", "ACF 1 of log volume",
                    "ACF gamma", "power ACF preferred", "Spearman vol, |r|", "Spearman > 0", "all six"], rows) + "\n")
    md.append(f"Without 13-16 UTC: {m4['share_ex'][0]}/{m4['share_ex'][1]} symbols pass all six.\n")

    # H-MS9
    md.append("## H-MS9: machine-learned conditional durations (xgboost AFT) vs ACD\n")
    md.append(NOTES.get("ms9", "") + "\n")
    rows = []
    for c in CL:
        if c not in m9["by_class"]:
            continue
        b = m9["by_class"][c]
        for vn in ("primary", "no_tod", "ex_cpi", "l2_only"):
            if vn in b:
                x = b[vn]
                rows.append([c, vn, x["n_symbols"], x["diff"], f"[{x['lo']:+.4f}, {x['hi']:+.4f}]", x["p_le0"]])
    md.append(_tbl(["class", "variant", "symbols", "mean per-duration log-lik gain (nats)", "95% CI (5-min blocks)", "bootstrap p (gain <= 0)"], rows) + "\n")
    rows = [[s_, x["class"], x["n_test"], x["dist"], x["sigma"], x["acd_best"], x["ll_acd"], x["ll_aft"], x["diff"], x["diff_no_tod"], x["has_l2"], ", ".join(x["top_features"][:4])]
            for s_, x in sorted(m9["per_symbol"].items(), key=lambda kv: (kv[1]["class"], kv[0]))]
    md.append("<details><summary>Per symbol</summary>\n")
    md.append(_tbl(["symbol", "class", "TEST durations", "AFT dist", "sigma", "best ACD", "ACD ll/obs", "AFT ll/obs", "gain", "gain no tod", "L2", "top features (gain)"], rows)
              + "\n\n</details>\n")

    # generator
    md.append("## What the generator must reproduce\n")
    md.append(NOTES.get("generator", "") + "\n")
    P = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    keys = list(next(iter(P["class_medians"].values()))["fit"].keys())
    rows = []
    for k in keys:
        rows.append([k] + [P["class_medians"].get(c, {}).get(h, {}).get(k) for c in CL for h in ("fit", "test")])
    md.append("Class medians of the fitted parameters (each half fitted on itself):\n")
    md.append(_tbl(["parameter"] + [f"{c} {h}" for c in CL for h in ("fit", "test")], rows) + "\n")

    md.append("## Parameter file schema (for the population / simulator agent)\n")
    for k, x in PARAMS_SCHEMA.items():
        md.append(f"- `{k}`: {x}")
    md.append(f"\nTidy parquet: {out['params_rows']} rows. Example: `ai_trader.sim.laws.law_from_dict(P['symbols']['BTCUSDT']['halves']['fit']['durations']['event']['laws']['weibull']).sample(1000)`.\n")
    md.append("## Caveats\n")
    md.append(NOTES.get("caveats", "") + "\n")
    md.append("## Rule candidates\n")
    md.append("```yaml\n" + NOTES.get("rules", "").strip() + "\n```\n")
    REPORT.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------------------------

def _size_order(syms: list[str]) -> list[str]:
    sizes = {}
    for s in syms:
        f = list((ROOT / "data" / "lake" / "trades_raw" / "venue=bybit" / f"symbol={s}" / f"date={DATE}").glob("*.parquet"))
        sizes[s] = sum(x.stat().st_size for x in f)
    return sorted(syms, key=lambda s: -sizes[s])


def run_stage(fn, syms: list[str], suffix: str, workers: int, force: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [s for s in _size_order(syms) if force or not (CACHE / f"{s}{suffix}.json").exists()]
    print(f"{fn.__name__}: {len(todo)} symbols", flush=True)
    if workers <= 1:
        for s in todo:
            print(fn(s), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, s): s for s in todo}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # noqa: BLE001 - one symbol's failure is reported, the rest continue
                print(f"FAILED {futs[f]}: {e!r}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["laws", "refit", "aft", "report", "all"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    syms = [s for s in a.symbols.split(",") if s] or symbols()
    if a.stage in ("laws", "all"):
        run_stage(analyse_symbol, syms, "", a.workers, a.force)
    if a.stage == "refit":
        run_stage(refit_symbol, syms, ".refit_done", a.workers, a.force)
    if a.stage in ("aft", "all"):
        run_stage(aft_symbol, syms, "_aft", max(1, a.workers // 2), a.force)
    if a.stage in ("report", "all"):
        build_report(syms, record=a.record)


if __name__ == "__main__":
    sys.exit(main())


