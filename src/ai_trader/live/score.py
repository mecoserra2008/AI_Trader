"""Scoring of Claude's live forecasts against the frozen network (H-LIVE1, H-LIVE2; wording in research/hypotheses.yaml).

Pairing ("the same hours"): a Claude forecast for (horizon, window_start) is paired with the network line whose forecast
bar S equals window_start (that line's 1h / 24h forecast covers exactly the same window). Only the FIRST Claude forecast
per (horizon, window_start) counts, and only if logged_at < window_start (re-checked here). Network lines flagged
'stale' are not used. Realised return: 100 * (ln close(window_end) - ln close(window_start)), 1h bar closes, as known at
the scoring time.

H-LIVE1  per pair d = pinball_claude - pinball_net over taus (0.1, 0.5, 0.9) in % log-return units; d averaged per UTC
         day of window_start, Newey-West t over days (research.evaluate.newey_west_t). Per horizon: supports iff n >= 300,
         mean d < 0 and t < -2; contradicts iff n >= 300, mean d > 0 and t > 2; else inconclusive. Hypothesis verdict:
         supports iff both horizons support; contradicts iff either contradicts; inconclusive otherwise. Decidable
         (recordable) only once both horizons have >= 300 pairs.
H-LIVE2  (amended 2026-11-13 23:25 UTC, before any Claude forecast) deviations: pairs with |sigma_claude / sigma_net - 1|
         > 0.10 (both horizons pooled). Per deviation d = ln N(r; q50_claude, sigma_claude) - ln N(r; q50_net, sigma_net)
         (Gaussian log score; network median = its q50, sigmas as stored, % units). d averaged per UTC day of
         window_start, Newey-West t over days. supports iff n >= 100, mean > 0 and t > 2; contradicts iff n >= 100,
         mean < 0 and t < -2; else inconclusive. Recordable once n >= 100.
         Descriptive only (not the test): the sign-hit rate, hit = sign(ln sigma_claude - ln sigma_net) == sign(ln |r| -
         ln sigma_net) (|r| = 0 dropped), its binomial p and day-clustered bootstrap CI, and the hit rate a
         direction-blind Claude would get given its up / down mix, from the network's empirical P(|r| > sigma_net)
         (about 0.32 under a Gaussian, so downward deviations hit ~0.68 by construction).
Segment: clean_holdout (every forward forecast is after the knowledge cutoff); recorded once.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np

TAU3 = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
MIN_LIVE1 = 300
MIN_LIVE2 = 100
DEV = 0.10


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    return t.astimezone(UTC)


def pinball3(y: float, q: dict) -> float:
    return float(np.mean([max(t * (y - q[k]), (t - 1) * (y - q[k])) for k, t in TAU3.items()]))


def pairs(claude: list[dict], net_lines: list[dict], realised) -> list[dict]:
    """realised(window_start: datetime, horizon: str) -> % log return or None (window not closed / no data)."""
    net = {}
    for ln in sorted(net_lines, key=lambda r: r["logged_at"]):
        if "stale" in ln.get("flags", []):
            continue
        net.setdefault(ln["forecast_hour"], ln)
    net = {_utc(k): v for k, v in net.items()}
    seen, out = set(), []
    for c in sorted(claude, key=lambda r: r["logged_at"]):
        key = (c["horizon"], c["window_start"])
        if key in seen:
            continue
        seen.add(key)
        ws = _utc(c["window_start"])
        if _utc(c["logged_at"]) >= ws:
            continue
        ln = net.get(ws)
        if ln is None or c["horizon"] not in ln.get("net", {}):
            continue
        nh = ln["net"][c["horizon"]]
        qn = nh["quantiles_pct"]
        if any(qn.get(k) is None for k in TAU3) or nh.get("sigma_pct") is None:
            continue
        r = realised(ws, c["horizon"])
        if r is None or not math.isfinite(r):
            continue
        qc = {"q10": c["q10"], "q50": c["q50"], "q90": c["q90"]}
        out.append({"horizon": c["horizon"], "window_start": c["window_start"], "day": c["window_start"][:10], "r": r,
                    "loss_claude": pinball3(r, qc), "loss_net": pinball3(r, {k: qn[k] for k in TAU3}),
                    "sigma_claude": c["sigma"], "sigma_net": nh["sigma_pct"], "q50_claude": c["q50"], "q50_net": qn["q50"], "claude_id": c["id"], "net_id": ln["id"]})
    for p in out:
        p["d"] = p["loss_claude"] - p["loss_net"]
    return out


def live1(P: list[dict]) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    res = {}
    for h in ("1h", "24h"):
        rows = [p for p in P if p["horizon"] == h]
        byday = defaultdict(list)
        for p in rows:
            byday[p["day"]].append(p["d"])
        daily = np.array([np.mean(byday[k]) for k in sorted(byday)])
        m, t, nd = newey_west_t(daily) if len(daily) else (float("nan"), float("nan"), 0)
        n = len(rows)
        v = "pending"
        if n >= MIN_LIVE1:
            v = "supports" if (m < 0 and t < -2) else "contradicts" if (m > 0 and t > 2) else "inconclusive"
        res[h] = {"n": n, "days": nd, "mean_d": float(np.mean([p["d"] for p in rows])) if rows else None, "daily_mean_d": m, "t": t,
                  "loss_claude": float(np.mean([p["loss_claude"] for p in rows])) if rows else None,
                  "loss_net": float(np.mean([p["loss_net"] for p in rows])) if rows else None, "verdict": v}
    vs = [res[h]["verdict"] for h in ("1h", "24h")]
    if "pending" in vs:
        res["verdict"] = "pending"
    else:
        res["verdict"] = "supports" if all(v == "supports" for v in vs) else "contradicts" if "contradicts" in vs else "inconclusive"
    return res


def gauss_logpdf(y: float, mu: float, sd: float) -> float:
    return -0.5 * math.log(2 * math.pi) - math.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def live2(P: list[dict], *, n_boot: int = 2000, seed: int = 0, p_above: float | None = None) -> dict:
    from scipy import stats

    from ai_trader.research.evaluate import newey_west_t

    rows = []
    for p in P:
        sc, sn, r = p["sigma_claude"], p["sigma_net"], p["r"]
        if sn <= 0 or sc <= 0 or abs(sc / sn - 1) <= DEV:
            continue
        d = gauss_logpdf(r, p["q50_claude"], sc) - gauss_logpdf(r, p["q50_net"], sn)
        up = sc > sn
        hit = None if r == 0 else float(up == (math.log(abs(r)) > math.log(sn)))
        rows.append({"day": p["day"], "d": d, "hit": hit, "up": up, "horizon": p["horizon"]})
    n = len(rows)
    out: dict = {"n": n, "by_horizon": {h: sum(1 for r in rows if r["horizon"] == h) for h in ("1h", "24h")}}
    if n == 0:
        return {**out, "mean_d": None, "rate": None, "verdict": "pending"}
    byday = defaultdict(list)
    for r in rows:
        byday[r["day"]].append(r["d"])
    m, t, nd = newey_west_t(np.array([np.mean(byday[k]) for k in sorted(byday)]))
    out.update({"mean_d": float(np.mean([r["d"] for r in rows])), "daily_mean_d": m, "t": t, "days": nd})
    v = "pending"
    if n >= MIN_LIVE2:
        v = "supports" if (m > 0 and t > 2) else "contradicts" if (m < 0 and t < -2) else "inconclusive"
    out["verdict"] = v
    # descriptive only: sign-hit rate and the direction-blind null
    hr = [r for r in rows if r["hit"] is not None]
    if hr:
        k = int(sum(r["hit"] for r in hr))
        days = sorted({r["day"] for r in hr})
        by = defaultdict(list)
        for r in hr:
            by[r["day"]].append(r["hit"])
        rng = np.random.default_rng(seed)
        bs = [np.mean([h for i in rng.integers(0, len(days), len(days)) for h in by[days[i]]]) for _ in range(n_boot)]
        share_up = float(np.mean([r["up"] for r in hr]))
        out["descriptive"] = {"n_hit": len(hr), "hits": k, "rate": k / len(hr),
                              "p_greater": float(stats.binomtest(k, len(hr), 0.5, alternative="greater").pvalue),
                              "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "share_up": share_up}
        if p_above is not None:
            out["descriptive"]["null_rate_direction_blind"] = share_up * p_above + (1 - share_up) * (1 - p_above)
    out["rate"] = out.get("descriptive", {}).get("rate")
    return out


def p_abs_above_sigma(scores: list[dict]) -> float | None:
    """Empirical P(|r| > sigma_net) over the network's own scored forecasts (live_nn scores.jsonl)."""
    v = [s["abs_r_pct"] > s["sigma_pct"] for s in scores if s.get("sigma_pct")]
    return float(np.mean(v)) if v else None


__all__ = ["MIN_LIVE1", "MIN_LIVE2", "gauss_logpdf", "live1", "live2", "p_abs_above_sigma", "pairs", "pinball3"]
