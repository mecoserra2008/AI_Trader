"""Forward scorecard: the AI book vs paired baselines on the SAME candidates (PLAN.md §5).

For every session in data/sessions/<id>/ (candidates.json, proposals.json, optional submitted.json written when
orders are sent), each candidate is resolved on the 1h bars that close after the session time, from its decision price:
  stop touched first (or both in one bar)  -> -1 R          (conservative intrabar ordering)
  target touched first                     -> +RR
  neither yet                              -> mark-to-market R at the latest close (flagged open)
minus round-trip costs in R. Own-exit candidates (trend sleeves, exit_rule "signal_flat": no target) resolve on the
stop or at the first 4h close after the decision where the sleeve signal has s x side <= 0 (the close price), with the
signal recomputed point in time (agent.trend_sleeves.symbol_signals as of the scorecard time). Books:
  ALL     every candidate, including gate-blocked ones (did the gates help?)
  RULE    every non-blocked candidate (the deterministic sleeves without Claude)
  RANDOM  random picks from RULE of the same size as AI each session (seeded; averaged over 200 draws)
  AI      what Claude actually took (submitted.json `ids`; the risk engine's accept / reject is in `accepted`)
Sessions marked `pipeline_test` (submitted.json or session.json) and candidates listed in submitted.json `tests` are
excluded from every book.

H-AI1 / H-AI2 running statistics (`ai_stats`) use only decision-harness sessions (session.json from
scripts/decision_session.py) that were decided, and only resolved candidates (open ones are marked to market for the
book table but excluded from the verdict until closed):
  H-AI1  per session: mean R of the AI takes minus mean R of RULE (and minus the mean of RANDOM picks of the same size,
         which equals the RULE mean in expectation); session differences are averaged per UTC day and tested with a
         Newey-West t on the day series. Verdict needs >= 60 closed AI trades.
  H-AI2  taken minus skipped mean R among non-blocked candidates, day-clustered bootstrap 95% CI. Needs >= 100 resolved.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.agent.scorecard
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now, utc
from ai_trader.storage.lake import Lake

COST_BPS = 2 * (5.5 + 2.0)  # taker + slippage, both legs (market entry); other entry types: cost_bps()


def cost_bps(entry_type: str | None) -> float:
    """Round trip (bp) for the candidate's entry type, as the engine sizes it (agent.session.entry_cost_bps); the exit is
    always taker + slippage. Unknown / missing entry types keep the old conservative market round trip."""
    from ai_trader.agent.session import entry_cost_bps

    if entry_type not in ("market", "limit", "pegged_limit"):
        return COST_BPS
    k = load().risk["costs"]
    return entry_cost_bps(entry_type) + k["taker_bps"] + k["slippage_bps"]
H_AI1_MIN_CLOSED = 60
H_AI2_MIN_RESOLVED = 100


def resolve_signal_exit(c: dict, t0: datetime, bars: pl.DataFrame, close_ms: np.ndarray, s: np.ndarray) -> dict:
    """Own-exit candidate from its decision price: on each 1h bar closing after t0 the stop first; then, at a 4h close,
    exit at the close when s x side <= 0 (agent.trend_sleeves trade layer). Costs as resolve()."""
    d = 1 if c["side"] == "long" else -1
    entry, stop = c["entry"], c["stop"]
    risk = abs(entry - stop)
    cost_r = entry * cost_bps(c.get("entry_type")) / 1e4 / risk
    sig = {int(m): float(v) for m, v in zip(close_ms, s, strict=True)}
    b = bars.filter(pl.col("close_ts") > t0) if "close_ts" in bars.columns else bars.filter(pl.col("event_ts") >= t0)
    for r in b.iter_rows(named=True):
        if (r["low"] <= stop) if d > 0 else (r["high"] >= stop):
            return {"R": -1 - cost_r, "status": "stop", "closed": str(r["event_ts"])}
        ms = int(r["close_ts"].timestamp() * 1000) if "close_ts" in r and r["close_ts"] is not None else None
        if ms is not None and ms in sig and sig[ms] * d <= 0:
            return {"R": d * (r["close"] - entry) / risk - cost_r, "status": "signal", "closed": str(r["close_ts"])}
    last = b["close"][-1] if b.height else entry
    return {"R": d * (last - entry) / risk - cost_r, "status": "open", "closed": None}


def resolve(c: dict, t0: datetime, bars: pl.DataFrame) -> dict:
    d = 1 if c["side"] == "long" else -1
    entry, stop, tp = c["entry"], c["stop"], c["take_profit"]
    risk = abs(entry - stop)
    cost_r = entry * cost_bps(c.get("entry_type")) / 1e4 / risk
    # bars that close after the decision time (a bar still forming at t0 counts; completed ones were known)
    b = bars.filter(pl.col("close_ts") > t0) if "close_ts" in bars.columns else bars.filter(pl.col("event_ts") >= t0)
    for r in b.iter_rows(named=True):
        hit_sl = (r["low"] <= stop) if d > 0 else (r["high"] >= stop)
        hit_tp = (r["high"] >= tp) if d > 0 else (r["low"] <= tp)
        if hit_sl:
            return {"R": -1 - cost_r, "status": "stop", "closed": str(r["event_ts"])}
        if hit_tp:
            return {"R": abs(tp - entry) / risk - cost_r, "status": "target", "closed": str(r["event_ts"])}
    last = b["close"][-1] if b.height else entry
    return {"R": d * (last - entry) / risk - cost_r, "status": "open", "closed": None}


def _session_meta(sdir: Path) -> dict:
    meta = {}
    for name in ("session.json", "submitted.json"):
        p = sdir / name
        if p.is_file():
            try:
                meta[name] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta[name] = {}
    return meta


def session_time(sdir: Path, meta: dict) -> datetime:
    s = meta.get("session.json") or {}
    if s.get("as_of"):
        return utc(s["as_of"])
    return datetime.strptime(sdir.name[1:14], "%Y%m%d-%H%M").replace(tzinfo=UTC)


def run(root: Path | None = None, t: datetime | None = None, lake: Lake | None = None) -> pl.DataFrame:
    root = root or load().root / "data" / "sessions"
    lake = lake or Lake()
    t = t or now()
    rows = []
    bars_cache: dict[str, pl.DataFrame] = {}
    sig_cache: dict[str, object] = {}

    def signals(sym: str):
        from ai_trader.agent.trend_sleeves import symbol_signals

        if sym not in sig_cache:
            try:
                sig_cache[sym] = symbol_signals(sym, t)
            except Exception:  # no signal: the candidate stays open (marked to market)
                sig_cache[sym] = None
        return sig_cache[sym]

    for sdir in sorted(p for p in root.glob("S*") if p.is_dir()):
        cand_f = sdir / "candidates.json"
        if not cand_f.exists():
            continue
        meta = _session_meta(sdir)
        sess, sub = meta.get("session.json") or {}, meta.get("submitted.json")
        t0 = session_time(sdir, meta)
        if t0 > t:
            continue
        cands = json.loads(cand_f.read_text(encoding="utf-8"))
        submitted = sub or {"ids": [], "tests": []}
        whole_test = bool(sess.get("pipeline_test") or submitted.get("pipeline_test"))
        for c in cands:
            sym = c["symbol"]
            if sym not in bars_cache:
                bars_cache[sym] = lake.bars(sym, "1h", t, start=t0).select("event_ts", "close_ts", "high", "low", "close").sort("event_ts")
            if c.get("exit_rule") == "signal_flat" or c.get("take_profit") is None:
                from ai_trader.agent.trend_sleeves import SLEEVES

                sg, spec = signals(sym), SLEEVES.get(c["sleeve"])
                if sg is not None and spec is not None:
                    res = resolve_signal_exit(c, t0, bars_cache[sym], sg.close_ms, sg.s[spec.signal])
                else:
                    res = resolve_signal_exit(c, t0, bars_cache[sym], np.zeros(0, np.int64), np.zeros(0))
            else:
                res = resolve(c, t0, bars_cache[sym])
            rows.append({"session": sdir.name, "day": t0.date().isoformat(), **{k: c[k] for k in ("id", "symbol", "sleeve", "side", "rr")},
                         "blocked": c["blocked"] is not None, "ai": c["id"] in submitted.get("ids", []),
                         "test": whole_test or c["id"] in submitted.get("tests", []), "phase4": bool(sess.get("harness")), "decided": sub is not None,
                         "p_quant": c.get("p_quant"), "p_final": (submitted.get("p_final") or {}).get(c["id"]), **res})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def summary(df: pl.DataFrame, draws: int = 200, seed: int = 0) -> list[dict]:
    if df.height == 0:
        return []
    d = df.filter(~pl.col("test"))
    out = []

    def book(name: str, x: pl.DataFrame) -> None:
        r = x["R"].to_numpy() if x.height else np.array([])
        out.append({"book": name, "n": len(r), "resolved": int((x["status"] != "open").sum()) if x.height else 0,
                    "mean_R": float(r.mean()) if len(r) else float("nan"), "total_R": float(r.sum()) if len(r) else 0.0,
                    "win_rate": float((r > 0).mean()) if len(r) else float("nan")})

    book("ALL (incl. blocked)", d)
    rule = d.filter(~pl.col("blocked"))
    book("RULE (sleeves, no Claude)", rule)
    ai = d.filter(pl.col("ai"))
    book("AI (submitted)", ai)
    # RANDOM: per session, sample |AI| picks from RULE; average over draws
    rng = np.random.default_rng(seed)
    tot = []
    for _ in range(draws):
        picks = []
        for s, g in rule.group_by("session"):
            k = int(ai.filter(pl.col("session") == s[0]).height)
            if k and g.height:
                picks += list(rng.choice(g["R"].to_numpy(), size=min(k, g.height), replace=False))
        tot.append(np.mean(picks) if picks else np.nan)
    out.append({"book": "RANDOM (same size as AI)", "n": int(ai.height), "resolved": None,
                "mean_R": float(np.nanmean(tot)) if tot and not all(np.isnan(tot)) else float("nan"), "total_R": None, "win_rate": None})
    return out


# ====================================================================== H-AI1 / H-AI2

def nw_t(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """(mean, Newey-West t, n) of a series (Bartlett kernel, lag floor(4 (n/100)^(2/9)) by default)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan")), float("nan"), n
    m = float(x.mean())
    e = x - m
    L = int(math.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    v = float(e @ e) / n
    for k in range(1, min(L, n - 1) + 1):
        v += 2 * (1 - k / (L + 1)) * float(e[k:] @ e[:-k]) / n
    se = math.sqrt(v / n) if v > 0 else float("nan")
    return m, (m / se if se and se > 0 else float("nan")), n


def ai_stats(df: pl.DataFrame, draws: int = 200, n_boot: int = 2000, seed: int = 0) -> dict:
    """Running H-AI1 / H-AI2 statistics on decided decision-harness sessions, resolved candidates only."""
    empty = {"H-AI1": {"closed_ai": 0, "min": H_AI1_MIN_CLOSED, "ready": False}, "H-AI2": {"resolved": 0, "min": H_AI2_MIN_RESOLVED, "ready": False}}
    if df.height == 0:
        return empty
    d = df.filter(~pl.col("test") & pl.col("phase4") & pl.col("decided") & ~pl.col("blocked") & (pl.col("status") != "open"))
    if d.height == 0:
        return empty
    rng = np.random.default_rng(seed)
    per = []
    for (s,), g in d.group_by("session", maintain_order=True):
        a = g.filter(pl.col("ai"))
        if a.height == 0:
            continue
        r_all = g["R"].to_numpy()
        k = min(a.height, len(r_all))
        rnd = float(np.mean([rng.choice(r_all, size=k, replace=False).mean() for _ in range(draws)]))
        per.append({"session": s, "day": g["day"][0], "ai": float(a["R"].mean()), "rule": float(r_all.mean()), "random": rnd})
    out: dict = {}
    n_ai = int(d["ai"].sum())
    if per:
        p = pl.DataFrame(per).with_columns((pl.col("ai") - pl.col("rule")).alias("d_rule"), (pl.col("ai") - pl.col("random")).alias("d_random"))
        days = p.group_by("day").agg(pl.col("d_rule").mean(), pl.col("d_random").mean()).sort("day")
        m1, t1, nd = nw_t(days["d_rule"].to_numpy())
        m2, t2, _ = nw_t(days["d_random"].to_numpy())
        ok = n_ai >= H_AI1_MIN_CLOSED
        verdict = None
        if ok:
            verdict = "supports" if (m1 > 0 and t1 > 2 and m2 > 0 and t2 > 2) else ("contradicts" if (m1 < 0 and t1 < -2) else "inconclusive")
        out["H-AI1"] = {"closed_ai": n_ai, "sessions": p.height, "days": nd, "ai_minus_rule": m1, "t_rule": t1, "ai_minus_random": m2, "t_random": t2,
                        "mean_R_ai": float(p["ai"].mean()), "mean_R_rule": float(p["rule"].mean()), "min": H_AI1_MIN_CLOSED, "ready": ok, "verdict": verdict}
    else:
        out["H-AI1"] = {**empty["H-AI1"], "closed_ai": n_ai}
    tk, sk = d.filter(pl.col("ai")), d.filter(~pl.col("ai"))
    n_res = d.height
    h2 = {"resolved": n_res, "taken": tk.height, "skipped": sk.height, "min": H_AI2_MIN_RESOLVED, "ready": n_res >= H_AI2_MIN_RESOLVED, "verdict": None}
    if tk.height and sk.height:
        diff = float(tk["R"].mean() - sk["R"].mean())
        days = d["day"].unique().to_list()
        by = {dd: d.filter(pl.col("day") == dd) for dd in days}
        bs = []
        for _ in range(n_boot):
            pick = pl.concat([by[x] for x in rng.choice(days, size=len(days), replace=True)])
            a, b = pick.filter(pl.col("ai"))["R"], pick.filter(~pl.col("ai"))["R"]
            if a.len() and b.len():
                bs.append(float(a.mean() - b.mean()))
        ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else [float("nan")] * 2
        h2.update({"taken_minus_skipped": diff, "ci95": ci, "days": len(days)})
        if h2["ready"]:
            h2["verdict"] = "supports" if ci[0] > 0 else ("contradicts" if ci[1] < 0 else "inconclusive")
    out["H-AI2"] = h2
    return out


if __name__ == "__main__":
    df = run()
    if df.height == 0:
        print("no sessions yet")
    else:
        print(df.select("session", "id", "sleeve", "side", "blocked", "ai", "test", "status", pl.col("R").round(2)).tail(30))
        for r in summary(df):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        print(ai_stats(df))
