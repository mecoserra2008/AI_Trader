"""D7F (2026-11-28): follow-up to decision D7 (user-approved 2026-11-27).

1. ``stress``: re-run the stored STRESS analysis (data/research/stress, read only) with the D7b-fixed harness
   (closed trades, bootstrap block max(5 d, p90 holding), funding in net R) and write every output under
   data/research/d7f/stress/ (results.json, the regenerated scorecard, and an old-vs-new comparison with an
   ablation of the three fixes). Nothing in data/research/stress or research/reports/stress_test_2026-11-26.md is
   touched; nothing is recorded unless ``--record`` is passed.
2. ``tost``: the H-D7F equivalence re-tests (registered in research/hypotheses_pending/D7F.yaml before running).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.d7f stress
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.d7f tost [--record]
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research import stress as S

OUT = S.ROOT / "data" / "research" / "d7f"
REDIRECT = ("results.json", "plain_words.md", "caveats.md")


class _OutProxy:
    """Stands in for stress.OUT during the re-run: inputs are read from data/research/stress, the files analyse()
    writes (results.json; the writeup's hand-written plain_words / caveats are read from here, so the regenerated
    scorecard carries none of the old prose) resolve under data/research/d7f/stress."""

    def __init__(self, src: Path, dst: Path) -> None:
        self.src, self.dst = src, dst

    def __truediv__(self, name: str) -> Path:
        return (self.dst if name in REDIRECT else self.src) / name

    def mkdir(self, *a: Any, **k: Any) -> None:
        self.dst.mkdir(parents=True, exist_ok=True)

    def __fspath__(self) -> str:
        return str(self.src)


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


# ====================================================================== 1. stress re-run
def _verdict_inputs(T: pl.DataFrame, P: pl.DataFrame, name: str, closed: bool, block: float | None, funding: bool,
                    crisis_min: float | None) -> dict:
    """Verdict inputs for one unit under one harness configuration (cheap: no engine)."""
    from ai_trader.research import stress_report as SR

    tb = SR._sel(T, name, "base")
    if tb.height == 0:
        return {"verdict": "n/a", "fired": ["no trades"], "base": {"n": 0}}

    def m(t: pl.DataFrame, c: S.Costs | None = None, ci: bool = False) -> dict:
        if t.height == 0:
            return {"n": 0}
        return S.metrics(t, S.r_net(t, c, funding=funding), ci=ci, closed_only=closed, block=block)

    base = m(tb, ci=True)
    ho = m(tb.filter(pl.col("seg") == "clean_holdout"))
    ins = {"base": base, "fees15": m(tb, S.Costs(fee_mult=1.5)), "slip5": m(tb, S.Costs(extra_bps=5.0)),
           "lat5": m(SR._sel(T, name, "lat5")), "holdout": ho, "crisis_engine_min": crisis_min}
    Pf = P if funding or P.height == 0 else P.drop("funding", strict=False)
    if closed and Pf.height and "why" in Pf.columns:
        Pf = Pf.filter(pl.col("why") != "end")
    ins["cliff"] = SR.perturb_summary(Pf, name).get("cliff")
    v, fired = S.verdict(ins)
    return {"verdict": v, "fired": fired, "base": {k: base.get(k) for k in ("n", "mean", "lo", "hi", "p_le0", "block_days", "total_ex_top5")},
            "holdout": {k: ho.get(k) for k in ("n", "mean")}, "fees15": ins["fees15"].get("mean"), "slip5": ins["slip5"].get("mean"),
            "lat5": ins["lat5"].get("mean"), "cliff": ins["cliff"], "open_end": base.get("open_end")}


def stress(workers_note: str = "") -> dict:
    from ai_trader.research import stress_report as SR
    from ai_trader.research import stress_writeup as SW

    src = S.OUT
    dst = OUT / "stress"
    dst.mkdir(parents=True, exist_ok=True)
    old = json.loads((src / "results.json").read_text(encoding="utf-8"))
    t0 = datetime.now()
    orig_out, orig_rep = S.OUT, SR.REPORT
    try:
        S.OUT = _OutProxy(src, dst)                     # type: ignore[assignment]
        SR.REPORT = dst / "stress_rerun_scorecard.md"
        new = SR.analyse(record=False)
    finally:
        S.OUT, SR.REPORT = orig_out, orig_rep
    t_analyse = (datetime.now() - t0).total_seconds()

    # ablation of the three D7b fixes on the verdict inputs (engine crisis minimum taken from the matching full run)
    design = json.loads((src / "design.json").read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(design["as_of"])
    T = S.with_funding(SR._load("trades"), as_of)
    P = S.with_funding(SR._load("perturb"), as_of)
    configs = {"A_old_all_b5_nofund": (False, S.BLOCK_DAYS, False), "B_closed_b5_nofund": (True, S.BLOCK_DAYS, False),
               "C_closed_p90_nofund": (True, None, False), "D_closed_p90_fund": (True, None, True)}
    abl: dict[str, dict] = {}
    for name in S.units():
        o, n = old["units"].get(name, {}), new["units"].get(name, {})
        row = {}
        for k, (closed, block, fund) in configs.items():
            cmin = o.get("crisis_engine_min") if k.startswith("A") else n.get("crisis_engine_min")
            row[k] = _verdict_inputs(T, P, name, closed, block, fund, cmin)
        abl[name] = row
    fund_share = {}
    for name in S.units():
        tb = SR._sel(T, name, "base")
        if tb.height:
            cm = S.closed_mask(tb)
            d = (S.r_net(tb) - S.r_net(tb, funding=False))[cm]
            fund_share[name] = {"funding_R_mean": float(np.mean(d)) if len(d) else float("nan"),
                                "funding_known_share": float(np.isfinite(tb["funding"].to_numpy()[cm]).mean()) if cm.any() else float("nan")}
    comp = {"design_as_of": design["as_of"], "seconds_analyse": t_analyse, "ablation": abl, "funding": fund_share,
            "old_verdicts": {u: {"verdict": r.get("verdict"), "fired": r.get("fired"), "closed_only_verdict": (r.get("closed_only") or {}).get("verdict"),
                                 "base": {k: (r.get("base") or {}).get(k) for k in ("n", "mean", "lo", "hi")},
                                 "holdout": {k: ((r.get("segments") or {}).get("clean_holdout") or {}).get(k) for k in ("n", "mean")},
                                 "crisis_engine_min": r.get("crisis_engine_min")} for u, r in old["units"].items()},
            "new_verdicts": {u: {"verdict": r.get("verdict"), "fired": r.get("fired"),
                                 "base": {k: (r.get("base") or {}).get(k) for k in ("n", "mean", "lo", "hi", "p_le0", "block_days")},
                                 "open_end": (r.get("base") or {}).get("open_end"),
                                 "holdout": {k: ((r.get("segments") or {}).get("clean_holdout") or {}).get(k) for k in ("n", "mean")},
                                 "crisis_engine_min": r.get("crisis_engine_min")} for u, r in new["units"].items()},
            "old_hypotheses": {h: v.get("verdict") for h, v in old["hypotheses"].items()},
            "new_hypotheses": {h: v.get("verdict") for h, v in new["hypotheses"].items()},
            "new_hypotheses_detail": new["hypotheses"]}
    (dst / "compare.json").write_text(json.dumps(comp, indent=1, default=SR._json), encoding="utf-8")
    (dst / "compare.md").write_text(compare_md(comp), encoding="utf-8")
    print(dst / "compare.md")
    return comp


def compare_md(c: dict) -> str:
    L = ["| unit | old verdict (stored) | old closed-only sensitivity | new verdict (D7b) | new rules fired | old mean [CI] | new mean [CI] (closed) | "
         "one-sided P(mean<=0) | block d | funding R/trade | holdout old -> new (n) | open at end n, sum R |", "|" + "---|" * 12]
    for u, o in c["old_verdicts"].items():
        n = c["new_verdicts"].get(u, {})
        ob, nb = o["base"], n.get("base") or {}
        oe = n.get("open_end") or {}
        fr = c["funding"].get(u, {})
        ch = "" if o["verdict"] == n.get("verdict") else " **(changes)**"
        L.append(f"| {u} | {o['verdict']} | {o.get('closed_only_verdict')} | {n.get('verdict')}{ch} | {'; '.join(n.get('fired') or []) or '-'} | "
                 f"{_f(ob.get('mean'))} [{_f(ob.get('lo'))}, {_f(ob.get('hi'))}] | {_f(nb.get('mean'))} [{_f(nb.get('lo'))}, {_f(nb.get('hi'))}] | "
                 f"{_f(nb.get('p_le0'))} | {_f(nb.get('block_days'), 1)} | {_f(fr.get('funding_R_mean'), 4)} | "
                 f"{_f(o['holdout'].get('mean'))} -> {_f((n.get('holdout') or {}).get('mean'))} ({(n.get('holdout') or {}).get('n')}) | "
                 f"{oe.get('n', 0)}, {_f(oe.get('sum_R'), 1)} |")
    L += ["", "Ablation (verdict under each harness configuration: A = stored (all trades, 5 d block, no funding), B = closed trades, "
          "C = + p90 block, D = + funding = D7b)", "", "| unit | A | B | C | D | D: base mean [CI] | D: P(mean<=0) |", "|---|---|---|---|---|---|---|"]
    for u, row in c["ablation"].items():
        d = row["D_closed_p90_fund"]["base"]
        L.append(f"| {u} | " + " | ".join(row[k]["verdict"] for k in ("A_old_all_b5_nofund", "B_closed_b5_nofund", "C_closed_p90_nofund", "D_closed_p90_fund"))
                 + f" | {_f(d.get('mean'))} [{_f(d.get('lo'))}, {_f(d.get('hi'))}] | {_f(d.get('p_le0'))} |")
    L += ["", "| hypothesis | old | new |", "|---|---|---|"]
    for h, v in c["old_hypotheses"].items():
        nv = c["new_hypotheses"].get(h)
        L.append(f"| {h} | {v} | {nv}{' **(changes)**' if nv != v else ''} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stress")
    t = sub.add_parser("tost")
    t.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "stress":
        stress()
    elif args.cmd == "tost":
        from ai_trader.research.d7f_tost import run

        run(record=args.record)


if __name__ == "__main__":
    main()
