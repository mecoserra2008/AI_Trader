"""Running scores for H-LIVE1 / H-LIVE2 from data/journal/live_claude and data/journal/live_nn (rules: ai_trader.live.score).

    PYTHONIOENCODING=utf-8 uv run python scripts/live_score.py [--record] [--as-of ISO]

Prints the running stats. With --record, a verdict is appended to research/hypotheses_results.jsonl only when the
registered minimum sample is reached (H-LIVE1: >= 300 pairs per horizon; H-LIVE2: >= 100 deviations) and only once per
hypothesis (segment clean_holdout). Otherwise it refuses and says why. H-LIVE1 / H-LIVE2 compare Claude with v1
(live_nn.v1), as registered.

Also prints a descriptive v2 vs v1 comparison (ai_trader.live.nn_v2.compare: same forecast bars, coverage of v1's ACI
bands vs v2's issued bands, band pinball), once over all v2 lines since 2026-11-13 20:00 (mostly replay) and once over
live v2 lines only. It is never recorded and decides nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.live import claude_log, nn, nn_v2, score  # noqa: E402


def lake_realised(as_of: datetime, start: datetime):
    from ai_trader.storage.lake import Lake

    b = Lake().bars(nn.SYMBOL, "1h", as_of, start=start - timedelta(days=2))
    cl = {nn._utc(t): float(c) for t, c in zip(b["close_ts"].to_list(), b["close"].to_list(), strict=True)} if b.height else {}
    return lambda ws, h: nn.realised(cl, ws, h)


def run(as_of: datetime, record: bool = False) -> dict:
    paths = nn.Paths.default()
    C = claude_log.read_all()
    N = nn.read_lines(paths)
    out: dict = {"as_of": as_of.isoformat(), "claude_forecasts": len(C), "net_lines": len(N)}
    # Principle 8: a new prompt version starts a new track record. The verdicts use the current version only;
    # earlier versions are reported alongside.
    prior = {}
    for pv in sorted({c.get("prompt_version", "") for c in C} - {claude_log.PROMPT_VERSION}):
        Cv = [c for c in C if c.get("prompt_version", "") == pv]
        Pv = score.pairs(Cv, N, lake_realised(as_of, min(nn._utc(c["window_start"]) for c in Cv)))
        prior[pv] = {"pairs": len(Pv), "H-LIVE1": score.live1(Pv)}
    out["earlier_prompt_versions"] = prior
    C = [c for c in C if c.get("prompt_version", "") == claude_log.PROMPT_VERSION]
    out["prompt_version"] = claude_log.PROMPT_VERSION
    if C:
        start = min(nn._utc(c["window_start"]) for c in C)
        P = score.pairs(C, N, lake_realised(as_of, start))
    else:
        P = []
    out["pairs"] = len(P)
    out["H-LIVE1"] = score.live1(P)
    out["H-LIVE2"] = score.live2(P, p_above=score.p_abs_above_sigma(nn.read_scores(paths)))
    V2 = nn_v2.read_lines(nn_v2.PathsV2(paths.root))
    if V2:
        since = datetime(2026, 11, 13, 20, tzinfo=UTC)
        rf = lake_realised(as_of, since)
        out["v2_vs_v1"] = {"all": nn_v2.compare(N, V2, rf, since=since),
                           "live_only": nn_v2.compare(N, [x for x in V2 if "replay" not in x.get("flags", [])], rf, since=since)}
    if record:
        from ai_trader.research.hypotheses import record as rec
        from ai_trader.research.hypotheses import results

        notes = {}
        for hid, res, need in (("H-LIVE1", out["H-LIVE1"], f">= {score.MIN_LIVE1} pairs per horizon"),
                               ("H-LIVE2", out["H-LIVE2"], f">= {score.MIN_LIVE2} deviations")):
            if res["verdict"] == "pending":
                notes[hid] = f"refused: minimum sample not reached ({need})"
            elif any(r["segment"] == "clean_holdout" for r in results(hid)):
                notes[hid] = "skipped: already recorded"
            else:
                rec(hid, "clean_holdout", res["verdict"], res, note="live forward journals; rules in src/ai_trader/live/score.py",
                    data="data/journal/live_claude/*.jsonl; data/journal/live_nn/*.jsonl")
                notes[hid] = "recorded"
        out["record"] = notes
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--as-of", default="")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC) if a.as_of else datetime.now(UTC)
    out = run(as_of, a.record)
    l1, l2 = out["H-LIVE1"], out["H-LIVE2"]
    print(f"as_of {out['as_of'][:19]}Z: {out['claude_forecasts']} Claude forecasts, {out['net_lines']} network lines, {out['pairs']} scored pairs")
    for h in ("1h", "24h"):
        r = l1[h]
        if r["n"]:
            print(f"H-LIVE1 {h}: n {r['n']} ({r['days']} days) pinball Claude {r['loss_claude']:.4f} vs net {r['loss_net']:.4f}, "
                  f"d {r['mean_d']:+.4f} (daily-mean NW t {r['t']:.2f}) -> {r['verdict']} (needs {score.MIN_LIVE1})")
        else:
            print(f"H-LIVE1 {h}: no scored pairs yet (needs {score.MIN_LIVE1})")
    print(f"H-LIVE1 overall: {l1['verdict']}")
    if l2.get("mean_d") is not None:
        ds = l2.get("descriptive", {})
        print(f"H-LIVE2: {l2['n']} deviations {l2['by_horizon']}, log-score diff {l2['mean_d']:+.4f} (daily-mean NW t {l2['t']:.2f}) "
              f"-> {l2['verdict']} (needs {score.MIN_LIVE2}); descriptive sign-hit {ds.get('rate', float('nan')):.3f}, "
              f"direction-blind null {ds.get('null_rate_direction_blind') or float('nan'):.3f}")
    else:
        print(f"H-LIVE2: no deviations yet (needs {score.MIN_LIVE2})")
    for seg, cmp in out.get("v2_vs_v1", {}).items():
        for h in ("1h", "24h"):
            c = cmp.get(h, {})
            if not c.get("n"):
                print(f"v2 vs v1 ({seg}) {h}: no common scored bars yet")
                continue
            cov = ", ".join(f"{L}% {c[f'cov_v1_{L}']:.3f} vs {c[f'cov_v2_{L}']:.3f}" for L in nn_v2.LEVELS)
            print(f"v2 vs v1 ({seg}, descriptive) {h}: n {c['n']} ({c['n_replay_v2']} v2 replay) coverage v1 vs v2 {cov}; band pinball "
                  f"{c['pinball_bands_v1']:.4f} vs {c['pinball_bands_v2']:.4f} (d {c['mean_d']:+.4f}, NW t {c['nw_t']:.2f}); "
                  f"80% width {c['width80_v1']:.2f} vs {c['width80_v2']:.2f}")
    if "record" in out:
        print("record:", json.dumps(out["record"]))
    return out


if __name__ == "__main__":
    main()
