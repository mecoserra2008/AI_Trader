"""Forward shadow of the D1 stop sigma (ai_trader.risk.sigma): an hourly log of every spec side by side, a forward scorer
against the bars' 5m realised variance, and the one-time switch gate (H-SIGMA-1..3, registered before the first line).

CLIs: scripts/sigma_shadow.py (log; --loop for the desk) and scripts/sigma_shadow_score.py (score; --gate; --record).

Log line (one per symbol per run), data/journal/sigma_shadow/<YYYY-MM-DD of the run>.jsonl, append-only:
  id, run_ts, as_of, symbol, spec_in_force, cfg {spec, shadow, switch_on, switch_gate}, version, store {last_hour, refresh},
  h: {"1h" | "4h": {bar_start, late, robust_bc, loghar_cr, loghar_cr_central, loghar_cr_range, loghar_cr_range_central,
                    nu, rbc_bar_start, fallback {spec: reason}, fit {spec: {refit_ts, n_train, coef, smear, clock_sha}}},
      "1d": {bar_start, late, robust_bc, vol_har, nu, rbc_bar_start}}
  Every sigma is the per-bar sd of the log return of the bar opening at bar_start, forecast from bars closed before it
  (no fallback: a spec without a forecast logs null). robust_bc is null when the lake's newest bar is not the one just
  before bar_start (stale updater). late = the run came more than LATE_MIN minutes after bar_start.
Scoring (registered): first line per symbol x horizon x bar_start among non-late lines; realised = the bar's 5m RV from the
  store (bar complete: 1h needs >= 10 valid 5m returns, 4h / 1d the 90% hour rule); QLIKE = RV/F - log(RV/F) - 1 with F =
  sigma^2; pooled = cross-symbol mean per bar_start over symbols with both specs; DM = Newey-West HAC with the HLN
  correction (vol_spec.dm_test). VaR: lower 1% of a unit-variance Student-t with the horizon's nu; hit = r < -q sigma with
  r the bar's close-to-close log return (store closes). Weekend = the bar touches the TradFi weekend (risk.sigma).
Gate (H-SIGMA-1 success rule): at the first --gate run on/after risk_sigma.switch_on, one line is appended to
  gate.jsonl {switch_on, decided_at, pass, spec, shadow, stats}; risk.sigma.spec_in_force reads the first line per switch_on.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.risk import sigma as S
from ai_trader.risk import sigma_store

LATE_MIN = 15
OFFSET_S = 150              # run at hh:02:30: the updater has written the 1h / 4h / 1d bars, the 5m klines are final
NEW_SPECS = ("loghar_cr", "loghar_cr_range")
GATE = {"min_1h": 168, "min_4h": 42, "min_symbols": 5, "t_1h": -2.0, "var_1h_max": 0.02}
MIN_H2 = 336


def journal_dir() -> Path:
    p = load().root / "data" / "journal" / "sigma_shadow"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _iso(t: datetime | None) -> str | None:
    return t.astimezone(UTC).isoformat() if t is not None else None


def _num(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


# ====================================================================== logging


def symbol_line(symbol: str, as_of: datetime, *, lake=None) -> dict:
    cfg = S.settings()
    out: dict = {"id": f"sig-{symbol}-{as_of:%Y%m%dT%H%M}", "run_ts": _iso(datetime.now(UTC)), "as_of": _iso(as_of), "symbol": symbol,
                 "spec_in_force": S.spec_in_force(as_of, cfg),
                 "cfg": {k: cfg.get(k) for k in ("spec", "shadow", "switch_on", "switch_gate")}, "version": S.VERSION,
                 "store": {"last_hour": _iso(sigma_store.last_hour(symbol))}, "h": {}}
    for g in ("1h", "4h", "1d"):
        bar = S._floor_bar(as_of, g)
        row: dict = {"bar_start": _iso(bar), "late": (as_of - bar) > timedelta(minutes=LATE_MIN), "nu": cfg["nu"][g]}
        try:
            rb = S.sigma_risk(symbol, g, as_of, spec="robust_bc", lake=lake)
            row["rbc_bar_start"] = _iso(rb.bar_start)
            row["robust_bc"] = _num(rb.sigma) if rb.bar_start == bar else None
        except Exception as e:  # noqa: BLE001
            row["robust_bc"], row["rbc_error"] = None, repr(e)[:200]
        fb, fits = {}, {}
        specs = ("loghar_cr",) if g == "1d" else NEW_SPECS
        for sp in specs:
            r = S.sigma_risk(symbol, g, as_of, spec=sp, lake=lake, fallback=False)
            key = "vol_har" if g == "1d" else sp
            row[key] = _num(r.sigma)
            if g != "1d":
                row[f"{sp}_central"] = _num(r.central)
                if r.fit:
                    fits[sp] = {k: r.fit[k] for k in ("refit_ts", "n_train", "coef", "smear", "clock_sha", "flags") if k in r.fit}
            if r.fallback:
                fb[key] = r.fallback
        if fb:
            row["fallback"] = fb
        if fits:
            row["fit"] = fits
        out["h"][g] = row
    return out


def run_once(as_of: datetime | None = None, symbols: list[str] | None = None, refresh: bool = True) -> dict:
    """One run for every symbol. With as_of (a replay) the lines go to journal_dir()/replay/ and carry replay=True: the
    forward journal only ever holds lines computed at their own time."""
    replay = as_of is not None
    as_of = S._utc(as_of) if as_of else datetime.now(UTC)
    symbols = symbols or list(S.settings()["symbols"])
    t0 = time.time()
    rf = sigma_store.refresh(symbols, now=as_of) if refresh and not replay else {}
    d = journal_dir() / "replay" if replay else journal_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{as_of:%Y-%m-%d}.jsonl"
    n, errs = 0, {}
    for s in symbols:
        try:
            line = symbol_line(s, as_of)
            if replay:
                line["replay"] = True
            if rf.get(s):
                line["store"]["refresh"] = rf[s]
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")
            n += 1
        except Exception as e:  # noqa: BLE001
            errs[s] = repr(e)[:300]
    return {"as_of": _iso(as_of), "lines": n, "errors": errs, "secs": round(time.time() - t0, 1), "file": str(path)}


def loop(symbols: list[str] | None = None, poll_s: int = 30, score: bool = True) -> None:
    """Hourly at hh:02:30 UTC. The next run is recomputed from the wall clock every poll_s seconds, so a sleep / suspend
    only skips the hours it covered (a run after wake-up is flagged late in the line when > LATE_MIN after the hour)."""
    done_hour = None
    while True:
        now = time.time()
        hour = int(now // 3600)
        if hour != done_hour and now >= hour * 3600 + OFFSET_S:
            done_hour = hour
            try:
                print(json.dumps(run_once(symbols=symbols)), flush=True)
                if score:
                    s = score_all(gate=True)
                    print(json.dumps({"scored": s.get("n_lines"), "gate": s.get("gate")}, default=str), flush=True)
            except Exception as e:  # noqa: BLE001 - a bad hour must not kill the loop
                print(json.dumps({"error": repr(e)[:500], "ts": _iso(datetime.now(UTC))}), flush=True)
        time.sleep(poll_s)


# ====================================================================== scoring


def read_lines() -> list[dict]:
    rows = []
    for p in sorted(journal_dir().glob("20*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


SPEC_KEYS = {"1h": ("robust_bc", "loghar_cr", "loghar_cr_central", "loghar_cr_range", "loghar_cr_range_central"),
             "4h": ("robust_bc", "loghar_cr", "loghar_cr_central", "loghar_cr_range", "loghar_cr_range_central"),
             "1d": ("robust_bc", "vol_har")}


def forecast_table(lines: list[dict]) -> pl.DataFrame:
    """First non-late line per symbol x horizon x bar_start: one row with every spec's sigma."""
    recs = []
    for ln in lines:
        if ln.get("replay"):
            continue
        for g, row in (ln.get("h") or {}).items():
            if row.get("late") or not row.get("bar_start"):
                continue
            rec = {"symbol": ln["symbol"], "h": g, "bar_start": row["bar_start"], "run_ts": ln.get("run_ts") or ln.get("as_of")}
            for k in SPEC_KEYS.get(g, ()):
                rec[k] = row.get(k)
            recs.append(rec)
    if not recs:
        return pl.DataFrame()
    cols = sorted({k for r in recs for k in r})
    df = pl.DataFrame([{c: r.get(c) for c in cols} for r in recs], schema={c: (pl.Float64 if c in SPEC_KEYS["1h"] + ("vol_har",) else pl.Utf8) for c in cols})
    df = df.with_columns(pl.col("bar_start").str.to_datetime(time_zone="UTC").cast(pl.Datetime("us", "UTC")))
    return df.sort("run_ts").unique(["symbol", "h", "bar_start"], keep="first").sort("h", "bar_start", "symbol")


def realised(symbol: str, now: datetime) -> dict[str, pl.DataFrame]:
    """Per horizon: event_ts, rv (5m RV of the bar), r (close-to-close log return), complete bars ending <= now."""
    h1 = sigma_store.load(symbol, now)
    out = {}
    for g in ("1h", "4h", "1d"):
        if h1.height == 0:
            out[g] = pl.DataFrame(schema={"event_ts": pl.Datetime("us", "UTC"), "rv": pl.Float64, "r": pl.Float64})
            continue
        G = S.grid(h1, g).with_columns((pl.col("close").log() - pl.col("close").log().shift(1)).alias("r"))
        G = G.filter((pl.col("event_ts") + pl.duration(hours=S.HORIZONS[g]) <= pl.lit(now).cast(pl.Datetime("us", "UTC"))) & pl.col("rv").is_not_null())
        out[g] = G.select("event_ts", "rv", "r")
    return out


def _qlike(rv: np.ndarray, var: np.ndarray) -> np.ndarray:
    from ai_trader.research.vol_spec import qlike

    return qlike(rv, var)


def _t_q(nu: float, p: float = 0.01) -> float:
    from scipy import stats

    return float(-stats.t.ppf(p, nu) * math.sqrt((nu - 2) / nu))


def scored_table(now: datetime | None = None, lines: list[dict] | None = None) -> pl.DataFrame:
    now = now or datetime.now(UTC)
    ft = forecast_table(read_lines() if lines is None else lines)
    if ft.height == 0:
        return ft
    parts = []
    for sym in ft["symbol"].unique().to_list():
        rl = realised(sym, now)
        for g in ("1h", "4h", "1d"):
            sub = ft.filter((pl.col("symbol") == sym) & (pl.col("h") == g))
            if sub.height == 0 or rl[g].height == 0:
                continue
            j = sub.join(rl[g].rename({"event_ts": "bar_start"}), on="bar_start", how="inner")
            if j.height:
                parts.append(j)
    if not parts:
        return pl.DataFrame()
    df = pl.concat(parts, how="diagonal_relaxed")
    wk = np.zeros(df.height, dtype=bool)
    for g in ("1h", "4h", "1d"):
        m = (df["h"] == g).to_numpy()
        if m.any():
            wk[m] = S.weekend_bars(df.filter(pl.col("h") == g)["bar_start"], g)
    return df.with_columns(pl.Series("weekend", wk))


def pooled_dm(df: pl.DataFrame, g: str, a: str, b: str, min_symbols: int = 1) -> dict:
    """Pooled QLIKE difference a - b (cross-symbol mean per bar_start over symbols with both), DM-HLN."""
    from ai_trader.research.vol_spec import dm_test

    d = df.filter((pl.col("h") == g) & pl.col(a).is_not_null() & pl.col(b).is_not_null())
    if d.height == 0:
        return {"n_bars": 0, "n_fc": 0}
    rv = d["rv"].to_numpy()
    la, lb = _qlike(rv, d[a].to_numpy() ** 2), _qlike(rv, d[b].to_numpy() ** 2)
    d = d.with_columns(pl.Series("la", la), pl.Series("lb", lb))
    p = (d.group_by("bar_start").agg(pl.col("la").mean(), pl.col("lb").mean(), pl.len().alias("k"))
         .filter(pl.col("k") >= min_symbols).sort("bar_start"))
    if p.height == 0:
        return {"n_bars": 0, "n_fc": int(d.height)}
    diff = (p["la"] - p["lb"]).to_numpy()
    t = dm_test(diff, h=1) if len(diff) >= 20 else {"t_hln": float("nan"), "p": float("nan")}
    ma, mb = float(p["la"].mean()), float(p["lb"].mean())
    return {"n_bars": int(p.height), "n_fc": int(d.height), "qlike_a": ma, "qlike_b": mb, "diff": float(diff.mean()),
            "rel": ma / mb - 1 if mb else float("nan"), "t_hln": float(t.get("t_hln", float("nan"))), "p": float(t.get("p", float("nan")))}


def var_rate(df: pl.DataFrame, g: str, key: str, nu: float, weekend: bool | None = None) -> dict:
    from scipy import stats

    d = df.filter((pl.col("h") == g) & pl.col(key).is_not_null() & pl.col("r").is_not_null())
    if weekend is not None:
        d = d.filter(pl.col("weekend") == weekend)
    n = d.height
    if n == 0:
        return {"n": 0}
    q = _t_q(nu)
    hits = int((d["r"].to_numpy() < -q * d[key].to_numpy()).sum())
    rate = hits / n
    # Kupiec POF vs 1%
    p0 = 0.01
    if 0 < hits < n:
        lr = -2 * ((n - hits) * math.log(1 - p0) + hits * math.log(p0) - (n - hits) * math.log(1 - rate) - hits * math.log(rate))
    elif hits == 0:
        lr = -2 * (n * math.log(1 - p0))
    else:
        lr = float("inf")
    return {"n": n, "hits": hits, "rate": rate, "kupiec_p": float(stats.chi2.sf(lr, 1))}


def summary(df: pl.DataFrame, cfg: dict) -> dict:
    out: dict = {"n_scored": int(df.height) if df.height else 0, "horizons": {}}
    if df.height == 0:
        return out
    pairs = {"1h": [("loghar_cr_range", "robust_bc"), ("loghar_cr", "robust_bc"), ("loghar_cr_range_central", "loghar_cr_central"),
                    ("loghar_cr_range", "loghar_cr_range_central")],
             "4h": [("loghar_cr_range", "robust_bc"), ("loghar_cr", "robust_bc"), ("loghar_cr_range_central", "loghar_cr_central"),
                    ("loghar_cr_range", "loghar_cr_range_central")],
             "1d": [("vol_har", "robust_bc")]}
    for g in ("1h", "4h", "1d"):
        sub = df.filter(pl.col("h") == g)
        if sub.height == 0:
            continue
        nu = cfg["nu"][g]
        h = {"n_fc": int(sub.height), "symbols": int(sub["symbol"].n_unique()), "first_bar": str(sub["bar_start"].min()), "last_bar": str(sub["bar_start"].max()),
             "dm": {f"{a}-{b}": pooled_dm(df, g, a, b) for a, b in pairs[g]},
             "dm_gate": {f"{a}-{b}": pooled_dm(df, g, a, b, GATE["min_symbols"]) for a, b in pairs[g][:1]},
             "var_1pct_lower": {k: {"all": var_rate(df, g, k, nu), "weekend": var_rate(df, g, k, nu, True)}
                                for k in SPEC_KEYS[g] if k in sub.columns}}
        out["horizons"][g] = h
    return out


def gate_rule(summ: dict) -> tuple[bool | None, dict]:
    """H-SIGMA-1's registered success rule. Returns (pass, stats); pass None = too few data (treated as a fail)."""
    h1, h4 = summ.get("horizons", {}).get("1h", {}), summ.get("horizons", {}).get("4h", {})
    d1 = h1.get("dm_gate", {}).get("loghar_cr_range-robust_bc", {})
    d4 = h4.get("dm_gate", {}).get("loghar_cr_range-robust_bc", {})
    v1 = h1.get("var_1pct_lower", {}).get("loghar_cr_range", {}).get("all", {})
    st = {"n_1h": d1.get("n_bars", 0), "n_4h": d4.get("n_bars", 0), "diff_1h": d1.get("diff"), "t_1h": d1.get("t_hln"),
          "rel_1h": d1.get("rel"), "diff_4h": d4.get("diff"), "t_4h": d4.get("t_hln"), "rel_4h": d4.get("rel"),
          "var_1h": v1.get("rate"), "var_1h_n": v1.get("n"), "rule": GATE}
    if st["n_1h"] < GATE["min_1h"] or st["n_4h"] < GATE["min_4h"]:
        return None, st
    ok = (st["diff_1h"] is not None and st["diff_1h"] < 0 and st["t_1h"] is not None and st["t_1h"] < GATE["t_1h"]
          and st["diff_4h"] is not None and st["diff_4h"] < 0
          and st["var_1h"] is not None and st["var_1h"] <= GATE["var_1h_max"])
    return bool(ok), st


def score_all(now: datetime | None = None, gate: bool = False, write: bool = True) -> dict:
    now = S._utc(now) if now else datetime.now(UTC)
    cfg = S.settings()
    lines = read_lines()
    df = scored_table(now, lines)
    summ = summary(df, cfg)
    summ.update({"computed_at": _iso(now), "n_lines": len(lines), "cfg": {k: cfg.get(k) for k in ("spec", "shadow", "switch_on", "switch_gate")},
                 "spec_in_force": S.spec_in_force(now, cfg)})
    if gate:
        summ["gate"] = maybe_gate(summ, now, cfg)
    if write:
        (journal_dir() / "score_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    return summ


def maybe_gate(summ: dict, now: datetime, cfg: dict) -> dict:
    sw, shadow = cfg.get("switch_on"), cfg.get("shadow")
    if not sw or not shadow or shadow == cfg["spec"]:
        return {"status": "no switch configured"}
    if now < S._utc(sw):
        return {"status": f"before switch_on {sw}"}
    have = S.gate_decision(str(sw), now)
    if have is not None:
        return {"status": "decided", **{k: have.get(k) for k in ("decided_at", "pass")}}
    ok, st = gate_rule(summ)
    line = {"switch_on": str(sw), "decided_at": _iso(now), "pass": bool(ok) if ok is not None else False,
            "insufficient": ok is None, "spec": cfg["spec"], "shadow": shadow, "hypothesis": "H-SIGMA-1", "stats": st}
    with S.gate_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return {"status": "decided now", "pass": line["pass"], "insufficient": line["insufficient"]}


def table(summ: dict) -> str:
    rows = [f"sigma shadow score {summ.get('computed_at')} | lines {summ.get('n_lines')} | scored {summ.get('n_scored')} | in force {summ.get('spec_in_force')}"]
    for g, h in summ.get("horizons", {}).items():
        rows.append(f"[{g}] forecasts {h['n_fc']} | symbols {h['symbols']} | bars {h['first_bar']} .. {h['last_bar']}")
        for k, d in h["dm"].items():
            if d.get("n_bars"):
                rows.append(f"   {k:44s} bars {d['n_bars']:5d}  QLIKE {d['qlike_a']:.4f} vs {d['qlike_b']:.4f}  rel {d['rel']:+.1%}  t_HLN {d['t_hln']:+.2f}")
        for k, v in h["var_1pct_lower"].items():
            a, w = v["all"], v["weekend"]
            if a.get("n"):
                rows.append(f"   VaR1% lower {k:28s} {a['hits']}/{a['n']} = {a['rate']:.2%} (Kupiec p {a['kupiec_p']:.2f})"
                            + (f" | weekend {w['hits']}/{w['n']} = {w['rate']:.2%}" if w.get("n") else ""))
    if "gate" in summ:
        rows.append(f"gate: {summ['gate']}")
    return "\n".join(rows)


def record(summ: dict) -> list[dict]:
    """Record H-SIGMA-1..3 verdicts (clean_holdout) once their registered minimum samples are met. Run under the
    results lock (AGENT_RULES §4). Returns the rows written."""
    from ai_trader.research.hypotheses import record as rec

    rows = []
    gl = [d for d in _all_gate_lines() if d.get("hypothesis") == "H-SIGMA-1"]
    if gl:
        d = min(gl, key=lambda x: x["decided_at"])
        v = "inconclusive" if d.get("insufficient") else ("supports" if d.get("pass") else "contradicts")
        rows.append(rec("H-SIGMA-1", "clean_holdout", v, d.get("stats", {}), note=f"switch gate for switch_on {d['switch_on']}", data="data/journal/sigma_shadow"))
    h1 = summ.get("horizons", {}).get("1h", {})
    d2 = h1.get("dm", {}).get("loghar_cr_range_central-loghar_cr_central", {})
    if d2.get("n_bars", 0) >= MIN_H2:
        v = "supports" if d2["diff"] < 0 and d2["t_hln"] < -2 else ("contradicts" if d2["diff"] > 0 and d2["t_hln"] > 2 else "inconclusive")
        rows.append(rec("H-SIGMA-2", "clean_holdout", v, d2, data="data/journal/sigma_shadow"))
    vw = h1.get("var_1pct_lower", {})
    fl, ce = vw.get("loghar_cr_range", {}).get("weekend", {}), vw.get("loghar_cr_range_central", {}).get("weekend", {})
    if fl.get("n", 0) >= 4 * 49:   # ~49 TradFi-weekend hours per week: four weekends
        ok = fl["rate"] <= 0.02 and fl["hits"] <= ce.get("hits", 0)
        bad = fl["rate"] > 0.02 and fl["kupiec_p"] < 0.05
        rows.append(rec("H-SIGMA-3", "clean_holdout", "supports" if ok else ("contradicts" if bad else "inconclusive"),
                        {"floored": fl, "central": ce}, data="data/journal/sigma_shadow"))
    return rows


def _all_gate_lines() -> list[dict]:
    p = S.gate_path()
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
