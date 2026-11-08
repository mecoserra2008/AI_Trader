"""Run the registered studies, store priors, render a report and record mechanical verdicts.

    PYTHONIOENCODING=utf-8 uv run python scripts/run_research.py --freq 1h [--record]

Outputs
  data/priors/{strategy_by_cell,path_priors,shock_study}_<freq>_v0.parquet   (the regime playbook priors)
  data/priors/weekends_v0.parquet
  research/reports/playbook_<freq>_<date>.md
  research/hypotheses_results.jsonl   (only with --record)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.research import hypotheses, verdicts
from ai_trader.research.regimes import CELL_LABELS
from ai_trader.research.studies import BAR_HOURS, playbook, weekend_study

UNIVERSE = ("BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
            "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT").split()
CELLS = [f"R{i}" for i in range(1, 10)]


def md_table(df: pl.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--record", action="store_true", help="append mechanical verdicts to hypotheses_results.jsonl")
    args = ap.parse_args()
    cfg = load()
    pri = cfg.root / "data" / "priors"
    rep = cfg.root / "research" / "reports"
    pri.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    as_of = now()

    pb = playbook(UNIVERSE, args.freq, as_of)
    for k, v in pb.items():
        v.write_parquet(pri / f"{k}_{args.freq}_v0.parquet")
    wk = weekend_study(UNIVERSE, as_of)
    wk.write_parquet(pri / "weekends_v0.parquet")

    h24 = max(1, round(24 / BAR_HOURS[args.freq]))
    results = {
        "H-W1": verdicts.h_w1(wk), "H-W2": verdicts.h_w2(wk), "H-W3": verdicts.h_w3(wk),
        "H-W4": verdicts.h_w4(wk), "H-W5": verdicts.h_w5(wk), "H-W5b": verdicts.h_w5b(wk),
        "H-R1": verdicts.h_r1(pb["strategy_by_cell"]), "H-R2": verdicts.h_r2(pb["strategy_by_cell"]),
        "H-S1": verdicts.h_s1(pb["shock_study"], h24),
    }

    sb = pb["strategy_by_cell"]
    out = [f"# Regime playbook v0 ({args.freq}), generated {as_of:%Y-%m-%d %H:%M} UTC", "",
           f"Universe: {len(UNIVERSE)} Bybit USDT perps (today's survivors). Regime labeller v0 (rule-based). "
           f"Split at knowledge cutoff {cfg.settings['research']['knowledge_cutoff']}. P&L in risk units net of costs; "
           "cell stats are cell-conditional equal-weight portfolios (clustered by time), NW t-stats, shrunk with κ = "
           f"{cfg.settings['research']['shrinkage_kappa']} episodes.", "",
           "Cells: " + ", ".join(f"{k} {v}" for k, v in CELL_LABELS.items()), ""]
    out += ["## Hypothesis verdicts (mechanical, from pre-registered criteria)", "",
            "| id | title | in-sample | holdout |", "|---|---|---|---|"]
    reg = hypotheses.registry()
    for hid, res in results.items():
        out.append(f"| {hid} | {reg[hid]['title']} | {res['llm_in_sample'][0]} | {res['clean_holdout'][0]} |")
    for seg in ("llm_in_sample", "clean_holdout"):
        d = sb.filter(pl.col("segment") == seg)
        for metric, title in (("t_nw", "NW t-stat"), ("mean_shrunk", "shrunk mean P&L per active bar (risk units ×100)")):
            vals = d.with_columns((pl.col(metric) * (100 if metric == "mean_shrunk" else 1)).alias(metric))
            piv = vals.pivot(on="cell", index="strategy", values=metric)
            piv = piv.select("strategy", *[c for c in [*CELLS, "ALL"] if c in piv.columns]).sort("strategy")
            out += ["", f"## {title}: {seg}", "", md_table(piv)]
    pp = pb["path_priors"].filter((pl.col("segment") == "llm_in_sample") & (pl.col("h_bars") == h24))
    out += ["", f"## Forward-path priors at {h24} bars (~24 h), in-sample", "",
            md_table(pp.select("cell", "n", "mean_z", "p_up", "q10", "q50", "q90", "mfe_med", "mae_med"))]
    ss = pb["shock_study"].filter((pl.col("segment") == "llm_in_sample") & (pl.col("h_bars") == h24))
    out += ["", f"## Post-shock continuation (|z| >= {cfg.settings['research']['shock_z']}) at ~24 h, in-sample", "",
            md_table(ss.select("cell", "n_events", "mean_signed_z", "t", "p_continue"))]
    out += ["", "## Verdict metrics", ""]
    for hid, res in results.items():
        for seg, (v, m) in res.items():
            out.append(f"- **{hid} / {seg}: {v}**: `{m}`")
    path = rep / f"playbook_{args.freq}_{as_of:%Y-%m-%d}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"report: {path}")
    for hid, res in results.items():
        print(hid, {seg: v for seg, (v, _) in res.items()})
        if args.record:
            for seg, (v, m) in res.items():
                hypotheses.record(hid, seg, v, m, data=f"{args.freq} {len(UNIVERSE)} symbols as_of {as_of:%Y-%m-%dT%H:%MZ}", note="v0 regime labeller; mechanical verdict")


if __name__ == "__main__":
    main()
