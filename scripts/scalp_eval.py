"""H-SCALP1 evaluator: reads data/journal/scalp/*.jsonl (atx scalp shadow runs) and applies the pre-registered metric.

    uv run python scripts/scalp_eval.py [--n 30] [--cfg-hash HASH]

Metric (research/hypotheses.yaml H-SCALP1): mean net R per FILLED trade on the first N trades (fixed N = 30, evaluated
once), 95 % circular block-bootstrap CI (block 5, 10 000 resamples, seed 0); TP-hit rate vs the ex-ante break-even
p* = 1 / (1 + W), W = mean net R of the TP exits. Secondary: fill rate per entry order, mean net R per order (unfilled = 0).
Only trades of one config hash are pooled (a config change is a new hypothesis id).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def block_bootstrap_ci(x: np.ndarray, block: int = 5, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95 % CI of the mean, circular block bootstrap."""
    n = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def load(cfg_hash: str | None) -> list[dict]:
    rows = []
    for p in sorted((ROOT / "data" / "journal" / "scalp").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("mode") == "shadow"]
    if cfg_hash:
        rows = [r for r in rows if r.get("cfg_hash") == cfg_hash]
    return rows


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--cfg-hash", default=None)
    args = a.parse_args()
    rows = load(args.cfg_hash)
    hashes = Counter(r.get("cfg_hash") for r in rows if r["type"] == "exit")
    if len(hashes) > 1 and not args.cfg_hash:
        raise SystemExit(f"several config hashes {dict(hashes)}: pass --cfg-hash")
    exits = sorted((r for r in rows if r["type"] == "exit"), key=lambda r: r["ms"])
    orders = [r for r in rows if r["type"] == "order" and r.get("leg") == "entry"]
    first = exits[: args.n]
    out: dict = {"orders": len(orders), "fills": sum(r["type"] == "fill" and r.get("leg") == "entry" for r in rows), "trades": len(exits)}
    out["fill_rate"] = out["fills"] / len(orders) if orders else None
    out["exits_by_kind"] = dict(Counter(r["kind"] for r in exits))
    if exits:
        allr = np.array([r["net_r"] for r in exits])
        out["all_mean_net_r"] = float(allr.mean())
        out["all_net_usd"] = float(sum(r["net_usd"] for r in exits))
        out["mean_net_r_per_order"] = float(allr.sum() / len(orders)) if orders else None
    if len(first) < args.n:
        out["verdict"] = f"inconclusive: {len(first)} < {args.n} filled trades"
    else:
        r = np.array([x["net_r"] for x in first])
        lo, hi = block_bootstrap_ci(r)
        tp = [x["net_r"] for x in first if x["kind"] == "tp"]
        w = float(np.mean(tp)) if tp else float("nan")
        p_star = 1.0 / (1.0 + w) if tp else float("nan")
        hit = len(tp) / len(first)
        ok = r.mean() > 0 and lo > -0.05 and tp and hit > p_star
        out |= {"n": len(first), "mean_net_r": float(r.mean()), "ci95": [lo, hi], "tp_hit_rate": hit, "tp_mean_net_r": w, "break_even_p": p_star, "verdict": "supported" if ok else "rejected"}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
