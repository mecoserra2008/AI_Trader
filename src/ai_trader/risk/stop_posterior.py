"""H-STOP1: the stop-distance cap as a Bayesian posterior (user decision 2026-11-14).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.risk.stop_posterior [--as-of ISO] [--dry-run]

The cap on |ln(entry / stop)| / sigma.per_bar (the risk engine's placement unit, crates/at-risk) starts as a PRIOR of
4.0 sigma and is updated to a posterior from the paper book's closed trades. Pinned exactly as H-STOP1 states:

  data      x_i = ln(MAE_i / sigma_i) for every CLOSED WINNING paper trade of the AI book (net R > 0; kill-switch exits
            excluded because they are operator actions, not market outcomes). MAE_i = maximum adverse excursion from the
            fill price over the 1h bars the trade was open (bar lows for longs, highs for shorts; the exit bar counts only
            for stop / target exits, which happen inside it), in units of the per-bar risk sigma the intent was placed
            with (sigma.per_bar of its own frequency). MAE below MAE_FLOOR_SIGMA is floored there so the log exists.
  model     x_i ~ N(mu, s^2) (MAE / sigma lognormal) with a normal-inverse-gamma prior
            mu | s^2 ~ N(m0, s^2 / kappa0), s^2 ~ InvGamma(a0, b0), worth PRIOR_PSEUDO_TRADES = 20 pseudo-trades:
            kappa0 = 20 and 2 a0 = 20 (a0 = 10), b0 = a0 * PRIOR_LOG_SD^2 (so the prior's scale is PRIOR_LOG_SD).
            m0 is solved so that the posterior-predictive 90% quantile of MAE / sigma equals 4.0 exactly:
            the predictive of x is Student-t(2 a, m, sqrt(b (kappa + 1) / (a kappa))), so
            m0 = ln 4 - t_{0.9, 2 a0} * sqrt(b0 (kappa0 + 1) / (a0 kappa0)).
  update    standard conjugate NIG update on all winners closed before the update time; weekly only (Monday 00:00 UTC,
            the latest Monday at or before the run): no mid-week drift, and a week that already has an update is a
            no-op. The applied cap is the posterior-predictive 90% quantile clipped to the hard bounds
            placement.stop_cap_bounds_sigma in config/risk.yaml ([2.5, 6.0]).
  output    data/models/risk/stop_cap.json (version, n, prior, posterior, applied_cap_sigma, history). crates/at-risk
            reads it (validated against the bounds, fail closed) and agent.session placement uses applied_cap().
            The LLM never sets it.
  report    H-STOP1's statistic: the posterior of the population 90% quantile exp(mu + z_0.9 s) by seeded Monte Carlo
            from the NIG posterior: median, 80% interval and P(q90 outside [3.5, 4.5]).

PRIOR_LOG_SD = 0.5 (the documented fallback): there are no replay trades in the llm_in_sample segment (before
research.knowledge_cutoff 2026-07-01); the only replay trades on record (data/exec/paper/replay-*, Oct-Nov 2026) are
clean_holdout and were deliberately not used to set the prior.

Known limitation (reported, not corrected, because the model is pinned): a winner never touched its stop, so its MAE is
below its stop distance (<= the cap in force). Winners' MAE is therefore truncated at the cap and the posterior can
drift down more easily than up; the share of winners whose MAE came within 10% of their stop is reported
(`near_stop_share`) so the user can judge it.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import stats

from ai_trader.core.config import load
from ai_trader.core.time import now, utc

VERSION = "stop_posterior.v1"
SCHEMA = "stop_cap.v1"
PRIOR_CAP_SIGMA = 4.0
PRIOR_QUANTILE = 0.90
PRIOR_PSEUDO_TRADES = 20
PRIOR_LOG_SD = 0.5
PRIOR_LOG_SD_SOURCE = ("fallback 0.5: no replay trades in the llm_in_sample segment (before research.knowledge_cutoff); the replay "
                       "trades on record (data/exec/paper/replay-*, 2026-10/11) are clean_holdout and were not used")
MAE_FLOOR_SIGMA = 0.05
H_STOP1_BAND = (3.5, 4.5)
N_DRAWS = 20_000
BOOK_RUN = "ai"
EXCLUDED_EXITS = ("kill",)


def model_path() -> Path:
    return load().root / "data" / "models" / "risk" / "stop_cap.json"


def book_root() -> Path:
    return load().root / "data" / "exec" / "paper_book"


# ====================================================================== the conjugate model

@dataclass(frozen=True)
class NIG:
    """Normal-inverse-gamma on (mu, s^2) of x = ln(MAE / sigma)."""

    m: float
    kappa: float
    a: float
    b: float

    @property
    def df(self) -> float:
        return 2.0 * self.a

    @property
    def scale(self) -> float:
        """Scale of the Student-t posterior predictive of x."""
        return math.sqrt(self.b * (self.kappa + 1.0) / (self.a * self.kappa))

    def quantile(self, q: float = PRIOR_QUANTILE) -> float:
        """Posterior-predictive q-quantile of MAE / sigma."""
        return math.exp(self.m + float(stats.t.ppf(q, self.df)) * self.scale)

    def to_dict(self) -> dict:
        return {**asdict(self), "df": self.df, "predictive_scale": self.scale, "q90_predictive": self.quantile(PRIOR_QUANTILE)}


def prior(log_sd: float = PRIOR_LOG_SD, cap: float = PRIOR_CAP_SIGMA, n0: int = PRIOR_PSEUDO_TRADES, q: float = PRIOR_QUANTILE) -> NIG:
    """The pinned prior: n0 pseudo-trades (kappa0 = n0, a0 = n0 / 2), b0 = a0 log_sd^2, predictive q-quantile = cap."""
    if not (log_sd > 0 and cap > 0 and n0 > 0 and 0 < q < 1):
        raise ValueError("prior needs log_sd > 0, cap > 0, n0 > 0, 0 < q < 1")
    kappa, a = float(n0), n0 / 2.0
    b = a * log_sd**2
    scale = math.sqrt(b * (kappa + 1.0) / (a * kappa))
    m = math.log(cap) - float(stats.t.ppf(q, 2.0 * a)) * scale
    return NIG(m, kappa, a, b)


def update(p: NIG, x: np.ndarray | list[float]) -> NIG:
    """Conjugate NIG update with observations x (log MAE / sigma)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return p
    xbar = float(x.mean())
    ss = float(((x - xbar) ** 2).sum())
    kn = p.kappa + n
    mn = (p.kappa * p.m + n * xbar) / kn
    an = p.a + n / 2.0
    bn = p.b + 0.5 * ss + p.kappa * n * (xbar - p.m) ** 2 / (2.0 * kn)
    return NIG(mn, kn, an, bn)


def q90_summary(post: NIG, *, draws: int = N_DRAWS, seed: int = 0, band: tuple[float, float] = H_STOP1_BAND) -> dict:
    """Posterior of the population 90% quantile exp(mu + z_0.9 s) (H-STOP1's statistic), seeded Monte Carlo."""
    rng = np.random.default_rng(seed)
    s2 = post.b / rng.gamma(post.a, 1.0, draws)
    mu = rng.normal(post.m, np.sqrt(s2 / post.kappa))
    q = np.exp(mu + float(stats.norm.ppf(PRIOR_QUANTILE)) * np.sqrt(s2))
    lo, hi = band
    return {"median": float(np.median(q)), "mean": float(q.mean()), "interval_80": [float(np.quantile(q, 0.10)), float(np.quantile(q, 0.90))],
            "p_outside_band": float(((q < lo) | (q > hi)).mean()), "p_below_band": float((q < lo).mean()), "p_above_band": float((q > hi).mean()),
            "band": list(band), "draws": draws, "seed": seed}


# ====================================================================== bounds, schedule, cap lookup

def bounds() -> tuple[float, float]:
    b = load().risk["placement"].get("stop_cap_bounds_sigma")
    if not b or len(b) != 2:
        raise ValueError("config/risk.yaml: placement.stop_cap_bounds_sigma [lo, hi] is required")
    lo, hi = float(b[0]), float(b[1])
    if not (0 < lo < hi):
        raise ValueError(f"placement.stop_cap_bounds_sigma {b}: need 0 < lo < hi")
    return lo, hi


def yaml_cap() -> float:
    return float(load().risk["placement"]["max_stop_distance_sigma"])


def week_start(t: datetime) -> datetime:
    """The latest Monday 00:00 UTC at or before t (the weekly update time)."""
    t = utc(t)
    d = datetime(t.year, t.month, t.day, tzinfo=UTC)
    return d - timedelta(days=d.weekday())


def read(path: Path | None = None) -> dict | None:
    p = path or model_path()
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def applied_cap(as_of: datetime | None = None, path: Path | None = None) -> tuple[float, str]:
    """(cap in per-bar risk sigmas, source). The same rule as crates/at-risk stop_cap.rs: no file -> the yaml prior;
    a file -> its cap after validation against the hard bounds (an invalid file raises: fail closed). With as_of, the
    history entry in force at as_of is used (point in time), else the prior."""
    lo, hi = bounds()
    doc = read(path)
    if doc is None:
        return yaml_cap(), "risk.yaml placement.max_stop_distance_sigma (prior; no stop_cap.json)"
    cap = doc.get("applied_cap_sigma")
    src = f"stop_cap.json {doc.get('version')} n {doc.get('n')} effective {doc.get('effective_from')}"
    if as_of is not None:
        hist = [h for h in doc.get("history", []) if utc(h["effective_from"]) <= utc(as_of)]
        if not hist:
            return yaml_cap(), "risk.yaml prior (no stop-cap update in force at as_of)"
        cap, src = hist[-1]["applied_cap_sigma"], f"stop_cap.json history n {hist[-1]['n']} effective {hist[-1]['effective_from']}"
    if not isinstance(cap, (int, float)) or not math.isfinite(cap) or not lo <= cap <= hi:
        raise ValueError(f"stop_cap.json: applied_cap_sigma {cap!r} invalid or outside the hard bounds [{lo}, {hi}] (fail closed)")
    return float(cap), src


# ====================================================================== data: closed winners and their MAE

def mae_sigma(side: str, entry: float, lows: np.ndarray, highs: np.ndarray, sigma_per_bar: float) -> float:
    """Maximum adverse excursion in per-bar risk sigmas (log distance from the fill)."""
    if not (sigma_per_bar > 0 and entry > 0) or len(lows) == 0:
        return float("nan")
    if side == "long":
        adv = math.log(entry / float(np.min(lows)))
    else:
        adv = math.log(float(np.max(highs)) / entry)
    return max(0.0, adv) / sigma_per_bar


def _jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def closed_winners(before: datetime, *, root: Path | None = None, run: str = BOOK_RUN, lake=None) -> tuple[list[dict], list[str]]:
    """Closed winning trades of the AI paper book with closed_ts < before, with their MAE / sigma. Point in time: the
    trade list, the intents and the bars are all read as known at `before`."""
    from ai_trader.storage.lake import Lake

    root = root or book_root()
    before = utc(before)
    cut = int(before.timestamp() * 1000)
    trades = [t for t in _jsonl(root / "paper" / run / "trades.jsonl") if t.get("closed_ts", 0) < cut]
    intents = {}
    for f in sorted((root / "intents").glob("*.jsonl")) if (root / "intents").is_dir() else []:
        for d in _jsonl(f):
            intents[d["intent_id"]] = d
    lake = lake or Lake()
    out, notes = [], []
    for t in trades:
        if not t.get("r_multiple", 0) > 0 or t.get("exit_reason") in EXCLUDED_EXITS:
            continue
        it = intents.get(t["id"])
        sig = ((it or {}).get("sigma") or {}).get("per_bar")
        freq = ((it or {}).get("sigma") or {}).get("freq", "1h")
        if not sig:
            notes.append(f"{t['id']}: no intent sigma.per_bar, skipped")
            continue
        t0 = datetime.fromtimestamp(t["opened_ts"] / 1000, UTC)
        t1 = datetime.fromtimestamp(t["closed_ts"] / 1000, UTC)
        b = lake.bars(t["symbol"], "1h", before, start=t0)
        if b.height == 0:
            notes.append(f"{t['id']}: no 1h bars, skipped")
            continue
        import polars as pl

        inside = (pl.col("event_ts") < t1) | ((pl.col("event_ts") == t1) & pl.lit(t.get("exit_reason") in ("stop", "target")))
        b = b.filter((pl.col("event_ts") >= t0) & inside)
        stop = (it or {}).get("stop")
        m = mae_sigma(str(t["side"]).lower(), float(t["entry_price"]), b["low"].to_numpy(), b["high"].to_numpy(), float(sig))
        stop_sig = abs(math.log(float(t["entry_price"]) / float(stop))) / float(sig) if stop else None
        out.append({"id": t["id"], "symbol": t["symbol"], "side": t["side"], "opened": t0.isoformat(), "closed": t1.isoformat(), "r": t["r_multiple"],
                    "exit_reason": t.get("exit_reason"), "sigma_per_bar": sig, "freq": freq, "mae_sigma": m, "stop_sigma": stop_sig})
    return out, notes


# ====================================================================== the weekly update

def compute(winners: list[dict], boundary: datetime, log_sd: float = PRIOR_LOG_SD) -> dict:
    """Pure: prior + winners -> the stop-cap document for the update at `boundary` (no history, no I/O)."""
    lo, hi = bounds()
    p0 = prior(log_sd)
    w = [x for x in winners if x.get("mae_sigma") is not None and np.isfinite(x["mae_sigma"])]
    xs = np.log(np.maximum([x["mae_sigma"] for x in w], MAE_FLOOR_SIGMA)) if w else np.array([])
    post = update(p0, xs)
    q = post.quantile(PRIOR_QUANTILE)
    cap = min(max(q, lo), hi)
    near = [x for x in w if x.get("stop_sigma") and x["mae_sigma"] >= 0.9 * x["stop_sigma"]]
    return {
        "schema": SCHEMA, "version": VERSION, "effective_from": boundary.isoformat(), "next_update": (boundary + timedelta(days=7)).isoformat(),
        "n": len(w), "q90_predictive": q, "applied_cap_sigma": cap, "clipped": cap != q, "bounds_sigma": [lo, hi],
        "unit": "|ln(entry/stop)| / sigma.per_bar (per-bar risk sigma of the intent's own frequency; crates/at-risk placement)",
        "prior": {**p0.to_dict(), "pseudo_trades": PRIOR_PSEUDO_TRADES, "log_sd": log_sd, "log_sd_source": PRIOR_LOG_SD_SOURCE,
                  "cap_sigma": PRIOR_CAP_SIGMA, "quantile": PRIOR_QUANTILE},
        "posterior": {**post.to_dict(), "q90_population": q90_summary(post)},
        "near_stop_share": (len(near) / len(w)) if w else None,
        "data": {"source": f"data/exec/paper_book/paper/{BOOK_RUN}/trades.jsonl (closed winners, kill exits excluded) + intents sigma.per_bar",
                 "mae_floor_sigma": MAE_FLOOR_SIGMA, "winners": [{k: x[k] for k in ("id", "symbol", "closed", "mae_sigma", "stop_sigma")} for x in w]},
        "hypothesis": "H-STOP1", "set_by": "deterministic weekly update (never the LLM)",
    }


def run_update(as_of: datetime | None = None, *, path: Path | None = None, root: Path | None = None, winners: list[dict] | None = None,
               write: bool = True, lake=None) -> dict:
    """The weekly update. Idempotent within a week: if the file already holds the update for the latest Monday
    00:00 UTC at or before as_of (or a later one), it is returned unchanged."""
    as_of = utc(as_of or now())
    boundary = week_start(as_of)
    p = path or model_path()
    old = read(p)
    if old is not None and utc(old["effective_from"]) >= boundary:
        return old
    notes: list[str] = []
    if winners is None:
        winners, notes = closed_winners(boundary, root=root, lake=lake)
    doc = compute(winners, boundary)
    doc["computed_at"] = now().isoformat()
    doc["notes"] = notes
    hist = list((old or {}).get("history", []))
    hist.append({"effective_from": doc["effective_from"], "n": doc["n"], "q90_predictive": doc["q90_predictive"],
                 "applied_cap_sigma": doc["applied_cap_sigma"], "clipped": doc["clipped"], "computed_at": doc["computed_at"],
                 "p_q90_outside_band": doc["posterior"]["q90_population"]["p_outside_band"]})
    doc["history"] = hist
    if write:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
        tmp.replace(p)
    return doc


def describe(doc: dict | None = None) -> str:
    """One compact line for the decision packet / status."""
    doc = doc if doc is not None else read()
    if doc is None:
        return f"stop cap {yaml_cap():.2f} sigma (risk.yaml prior; no posterior file yet)"
    q = doc["posterior"]["q90_population"]
    return (f"stop cap {doc['applied_cap_sigma']:.2f} sigma (H-STOP1, {doc['version']}, effective {doc['effective_from'][:10]}, n {doc['n']} winners; "
            f"predictive q90 {doc['q90_predictive']:.2f}, population q90 median {q['median']:.2f} 80% [{q['interval_80'][0]:.2f}, {q['interval_80'][1]:.2f}], "
            f"P(outside [3.5, 4.5]) {q['p_outside_band']:.2f}; bounds {doc['bounds_sigma']}; next update {doc['next_update'][:10]})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H-STOP1 weekly stop-cap update")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    doc = run_update(utc(a.as_of) if a.as_of else None, write=not a.dry_run)
    print(describe(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
