"""Per-sleeve edge priors become posteriors (user decision 2026-11-14 ~13:30 UTC), the same way as the stop cap.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.risk.edge_posterior [--as-of ISO] [--dry-run]

The decision sleeves set p_quant = p_null + edge, with p_null the driftless first-passage null and `edge` a small
documented prior (agent.session.PRIOR_EDGE: shock_regime +0.05, trend_4h +0.03 probability points). That edge is now
a posterior, updated from the AI paper book's closed trades of the sleeve. Pinned rule:

  data      every CLOSED trade of the sleeve in the AI paper book (data/exec/paper_book/paper/ai/trades.jsonl, strategy =
            sleeve) with closed_ts before the update time; kill-switch exits excluded (operator actions). Each trade's
            net R (costs and slippage included) is put on the edge scale with the trade's own geometry from its intent:
                y_i = (R_i - E0[R_i]) / (1 + g_i),   E0[R_i] = p_null_i g_i - (1 - p_null_i),
            with g_i the net R at the target. For a bracket trade that ends at stop or target y_i is exactly
            1{target} - p_null_i, so E[y_i] = edge and the model is a conjugate normal model on the trade's mean net R.
  model     y_i ~ N(edge, s^2) with s^2 = 0.25 pinned (the largest variance of a binary bracket outcome, so the
            weights are conservative), prior edge ~ N(edge0, s^2 / n0) with n0 = PRIOR_PSEUDO_TRADES = 30.
            Posterior: mean = (n0 edge0 + sum y) / (n0 + n), sd = s / sqrt(n0 + n).
  why 30    the prior sd is 0.5 / sqrt(30) = 0.09 probability points, wide enough for the edge to vanish or reverse;
            the data gets half the weight after 30 closed trades of the sleeve (one to two months at a few trades a
            week), so a week of noise cannot flip a sleeve on its own, and by H-AI1's 60 closed trades the data carries
            two thirds of the weight.
  schedule  weekly only, Monday 00:00 UTC (the latest Monday at or before the run); a week that already has an
            update is a no-op (no mid-week drift).
  output    data/models/risk/edge_posterior.json (version, n, prior, posterior, history). agent.session uses the
            posterior mean edge in force at as_of for p_quant and net E[R]. A posterior that pushes a sleeve's net E[R]
            to <= 0 blocks its candidates again (intended). The LLM never sets it.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

from ai_trader.core.config import load
from ai_trader.core.time import now, utc
from ai_trader.risk.stop_posterior import BOOK_RUN, EXCLUDED_EXITS, _jsonl, book_root, week_start

VERSION = "edge_posterior.v1"
SCHEMA = "edge_posterior.v1"
PRIOR_PSEUDO_TRADES = 30
OBS_SD = 0.5                 # sd of y_i (binary bracket outcome, upper bound)
MAX_ABS_EDGE = 0.5           # sanity bound on a stored edge (probability points); outside it the file is refused


def prior_edges() -> dict[str, float]:
    from ai_trader.agent.session import PRIOR_EDGE

    return dict(PRIOR_EDGE)


def model_path() -> Path:
    return load().root / "data" / "models" / "risk" / "edge_posterior.json"


def posterior(edge0: float, ys: list[float], n0: int = PRIOR_PSEUDO_TRADES, s: float = OBS_SD) -> dict:
    """Conjugate normal update of the mean edge with pinned observation sd s and n0 pseudo-trades at edge0."""
    ys = [float(y) for y in ys if y is not None and math.isfinite(y)]
    n = len(ys)
    mean = (n0 * edge0 + sum(ys)) / (n0 + n)
    sd = s / math.sqrt(n0 + n)
    return {"mean": mean, "sd": sd, "n": n, "n_eff": n0 + n, "ci90": [mean - 1.6449 * sd, mean + 1.6449 * sd],
            "data_mean": (sum(ys) / n) if n else None}


def trade_y(t: dict, intent: dict | None) -> float | None:
    """Edge-scale observation of one closed trade (see module doc); None without the intent's geometry."""
    ctx = (intent or {}).get("context") or {}
    p0, g = ctx.get("p_null"), ctx.get("g_net")
    if p0 is None or g is None or not (0 < p0 < 1) or not math.isfinite(g) or g <= -1:
        return None
    e0 = p0 * g - (1 - p0)
    return (float(t["r_multiple"]) - e0) / (1 + g)


def closed_trades(before, *, root: Path | None = None, run: str = BOOK_RUN) -> tuple[dict[str, list[dict]], list[str]]:
    root = root or book_root()
    cut = int(utc(before).timestamp() * 1000)
    intents = {}
    if (root / "intents").is_dir():
        for f in sorted((root / "intents").glob("*.jsonl")):
            for d in _jsonl(f):
                intents[d["intent_id"]] = d
    out: dict[str, list[dict]] = {}
    notes = []
    for t in _jsonl(root / "paper" / run / "trades.jsonl"):
        if t.get("closed_ts", 0) >= cut or t.get("exit_reason") in EXCLUDED_EXITS:
            continue
        y = trade_y(t, intents.get(t["id"]))
        if y is None:
            notes.append(f"{t['id']}: no p_null / g_net in its intent, skipped")
            continue
        out.setdefault(t["strategy"], []).append({"id": t["id"], "symbol": t["symbol"], "closed_ts": t["closed_ts"], "r": t["r_multiple"],
                                                  "exit_reason": t.get("exit_reason"), "y": y})
    return out, notes


def compute(trades: dict[str, list[dict]], boundary) -> dict:
    sleeves = {}
    for s, e0 in prior_edges().items():
        ts = trades.get(s, [])
        post = posterior(e0, [x["y"] for x in ts])
        sleeves[s] = {"n": post["n"], "prior": {"edge": e0, "pseudo_trades": PRIOR_PSEUDO_TRADES, "obs_sd": OBS_SD, "sd": OBS_SD / math.sqrt(PRIOR_PSEUDO_TRADES),
                                                "source": "agent.session.PRIOR_EDGE (scoreboard, deliberately small)"},
                      "posterior": post, "trades": [{k: x[k] for k in ("id", "symbol", "r", "exit_reason", "y")} for x in ts]}
    return {"schema": SCHEMA, "version": VERSION, "effective_from": boundary.isoformat(), "next_update": (boundary + timedelta(days=7)).isoformat(),
            "n": sum(v["n"] for v in sleeves.values()), "edges": {s: v["posterior"]["mean"] for s, v in sleeves.items()}, "sleeves": sleeves,
            "unit": "probability points added to the driftless first-passage null (p_quant = p_null + edge)",
            "set_by": "deterministic weekly update (never the LLM)"}


def read(path: Path | None = None) -> dict | None:
    p = path or model_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def run_update(as_of=None, *, path: Path | None = None, root: Path | None = None, trades: dict | None = None, write: bool = True) -> dict:
    as_of = utc(as_of or now())
    boundary = week_start(as_of)
    p = path or model_path()
    old = read(p)
    if old is not None and utc(old["effective_from"]) >= boundary:
        return old
    notes: list[str] = []
    if trades is None:
        trades, notes = closed_trades(boundary, root=root)
    doc = compute(trades, boundary)
    doc["computed_at"] = now().isoformat()
    doc["notes"] = notes
    hist = list((old or {}).get("history", []))
    hist.append({"effective_from": doc["effective_from"], "computed_at": doc["computed_at"],
                 "edges": doc["edges"], "n": {s: v["n"] for s, v in doc["sleeves"].items()}, "sd": {s: v["posterior"]["sd"] for s, v in doc["sleeves"].items()}})
    doc["history"] = hist
    if write:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
        tmp.replace(p)
    return doc


def applied_edges(as_of=None, path: Path | None = None) -> tuple[dict[str, float], str]:
    """(sleeve -> edge in force at as_of, source). No file / no update in force yet: the documented priors. A stored
    edge that is not finite or beyond MAX_ABS_EDGE raises (fail closed)."""
    base = prior_edges()
    doc = read(path)
    if doc is None:
        return base, "prior (agent.session.PRIOR_EDGE; no edge_posterior.json)"
    hist = doc.get("history", [])
    if as_of is not None:
        hist = [h for h in hist if utc(h["effective_from"]) <= utc(as_of)]
    if not hist:
        return base, "prior (no edge update in force at as_of)"
    h = hist[-1]
    out = dict(base)
    for s, e in h["edges"].items():
        if not isinstance(e, (int, float)) or not math.isfinite(e) or abs(e) > MAX_ABS_EDGE:
            raise ValueError(f"edge_posterior.json: edge {s}={e!r} invalid (fail closed)")
        out[s] = float(e)
    return out, f"edge_posterior.json {doc.get('version')} effective {h['effective_from'][:10]} n {h['n']}"


def describe(doc: dict | None = None) -> str:
    doc = doc if doc is not None else read()
    if doc is None:
        return "edge priors " + ", ".join(f"{s} {e:+.3f}" for s, e in prior_edges().items()) + f" (no posterior file yet; {PRIOR_PSEUDO_TRADES} pseudo-trades)"
    parts = [f"{s} {v['posterior']['mean']:+.3f} (prior {v['prior']['edge']:+.3f}, n {v['n']}, sd {v['posterior']['sd']:.3f})" for s, v in doc["sleeves"].items()]
    return (f"edge posterior ({doc['version']}, effective {doc['effective_from'][:10]}, {PRIOR_PSEUDO_TRADES} pseudo-trades, next {doc['next_update'][:10]}): "
            + "; ".join(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="weekly per-sleeve edge posterior")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(describe(run_update(utc(a.as_of) if a.as_of else None, write=not a.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
