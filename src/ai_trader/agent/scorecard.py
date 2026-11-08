"""Forward scorecard: the AI book vs paired baselines on the SAME candidates (PLAN.md §5).

For every session in data/sessions/<id>/ (candidates.json, proposals.json, optional submitted.json written when
orders are sent), each candidate is resolved on subsequent 1h bars from its decision price:
  stop touched first (or both in one bar)  -> -1 R          (conservative intrabar ordering)
  target touched first                     -> +RR
  neither yet                              -> mark-to-market R at the latest close (flagged open)
minus round-trip costs in R. Books:
  ALL     every candidate, including gate-blocked ones (did the gates help?)
  RULE    every non-blocked candidate (the deterministic sleeves without Claude)
  RANDOM  random picks from RULE of the same size as AI each session (seeded; averaged over 200 draws)
  AI      what Claude actually submitted
`pipeline_test` submissions are excluded from every book.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.agent.scorecard
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.storage.lake import Lake

COST_BPS = 2 * (5.5 + 2.0)  # taker + slippage, both legs


def resolve(c: dict, t0: datetime, bars: pl.DataFrame) -> dict:
    d = 1 if c["side"] == "long" else -1
    entry, stop, tp = c["entry"], c["stop"], c["take_profit"]
    risk = abs(entry - stop)
    cost_r = entry * COST_BPS / 1e4 / risk
    b = bars.filter(pl.col("event_ts") >= t0)
    for r in b.iter_rows(named=True):
        hit_sl = (r["low"] <= stop) if d > 0 else (r["high"] >= stop)
        hit_tp = (r["high"] >= tp) if d > 0 else (r["low"] <= tp)
        if hit_sl:
            return {"R": -1 - cost_r, "status": "stop", "closed": str(r["event_ts"])}
        if hit_tp:
            return {"R": abs(tp - entry) / risk - cost_r, "status": "target", "closed": str(r["event_ts"])}
    last = b["close"][-1] if b.height else entry
    return {"R": d * (last - entry) / risk - cost_r, "status": "open", "closed": None}


def run() -> pl.DataFrame:
    root = load().root / "data" / "sessions"
    lake = Lake()
    t = now()
    rows = []
    bars_cache: dict[str, pl.DataFrame] = {}
    for sdir in sorted(p for p in root.glob("S*") if p.is_dir()):
        cand_f = sdir / "candidates.json"
        if not cand_f.exists():
            continue
        t0 = datetime.strptime(sdir.name[1:], "%Y%m%d-%H%M").replace(tzinfo=t.tzinfo)
        cands = json.loads(cand_f.read_text(encoding="utf-8"))
        sub_f = sdir / "submitted.json"
        submitted = json.loads(sub_f.read_text(encoding="utf-8")) if sub_f.exists() else {"ids": [], "tests": []}
        for c in cands:
            sym = c["symbol"]
            if sym not in bars_cache:
                bars_cache[sym] = lake.bars(sym, "1h", t, start=t0).select("event_ts", "high", "low", "close").sort("event_ts")
            res = resolve(c, t0, bars_cache[sym])
            rows.append({"session": sdir.name, **{k: c[k] for k in ("id", "symbol", "sleeve", "side", "rr")}, "blocked": c["blocked"] is not None,
                         "ai": c["id"] in submitted.get("ids", []), "test": c["id"] in submitted.get("tests", []), **res})
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
    out.append({"book": "RANDOM (same size as AI)", "n": int(ai.height), "resolved": None, "mean_R": float(np.nanmean(tot)) if tot else float("nan"),
                "total_R": None, "win_rate": None})
    return out


if __name__ == "__main__":
    df = run()
    if df.height == 0:
        print("no sessions yet")
    else:
        print(df.select("session", "id", "sleeve", "side", "blocked", "ai", "test", "status", pl.col("R").round(2)).tail(30))
        for r in summary(df):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
