"""H-SIGMA-4 (registered 2026-11-27 before this ran): historical check of the D1 joint spec on the 5m RV store.

For each universe-L perp in the store (data/state/risk_sigma/rv1h, as of AS_OF) and each grid (1h, 4h): monthly
walk-forward forecasts from ai_trader.risk.sigma.forecast (loghar_cr, loghar_cr_range; stop and central versions) and
robust_bc on the same grid's close-to-close returns (features.vol_models, bias-corrected, use_weekend=False; the
forecast for bar i made at the close of i-1, as vol_spec.build). Target = the bar's 5m RV. Loss QLIKE. Pooled = cross-
symbol mean per bar over symbols with both specs of a pair (>= 5 symbols). DM-HLN (vol_spec.dm_test). Segments split at
research.knowledge_cutoff. Disclosure: WS04 already scored loghar_cr vs robust_bc on BTC ETH SOL XRP DOGE over this period.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.sigma_check            # compute, save, print
    ... --record   (under governor.py lock results) record the saved verdicts
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from ai_trader.core.config import load

AS_OF = datetime(2026, 11, 27, 22, 0, tzinfo=UTC)
PAIRS = (("loghar_cr_range", "robust_bc"), ("loghar_cr", "robust_bc"), ("loghar_cr_range", "loghar_cr"),
         ("loghar_cr_range_central", "loghar_cr_central"), ("loghar_cr_range", "loghar_cr_range_central"))
MIN_SYMBOLS = 5


def out_dir():
    p = load().root / "data" / "research" / "sigma"
    p.mkdir(parents=True, exist_ok=True)
    return p


def symbol_losses(symbol: str, g: str) -> pl.DataFrame:
    from ai_trader.features.vol_models import VolModel, run
    from ai_trader.research.calendar import add_calendar
    from ai_trader.research.vol_spec import qlike
    from ai_trader.risk import sigma as S
    from ai_trader.risk import sigma_store

    h1 = sigma_store.load(symbol, AS_OF)
    G = S.grid(h1, g).filter(pl.col("event_ts") + pl.duration(hours=S.HORIZONS[g]) <= pl.lit(AS_OF).cast(pl.Datetime("us", "UTC")))
    r = (G["close"].log() - G["close"].log().shift(1)).fill_null(np.nan).to_numpy()
    wk = add_calendar(G.select("event_ts"))["tradfi_weekend"].to_numpy()
    v = run(np.nan_to_num(r), wk, VolModel(bias_correct=True, use_weekend=False))["v_next"]
    cols = {"robust_bc": np.r_[np.nan, v[:-1]]}
    for sp in ("loghar_cr", "loghar_cr_range"):
        fc = S.forecast(G, g, sp)
        cols[sp], cols[f"{sp}_central"] = fc["stop"], fc["central"]
    y = G["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    ok = np.isfinite(y) & np.isfinite(cols["loghar_cr"]) & np.isfinite(cols["loghar_cr_range"]) & np.isfinite(cols["robust_bc"])
    df = pl.DataFrame({"event_ts": G["event_ts"], "symbol": symbol, **{k: np.where(ok, qlike(y, c), np.nan) for k, c in cols.items()}})
    return df.filter(pl.Series(ok))


def evaluate(symbols: list[str] | None = None) -> dict:
    from ai_trader.research.evaluate import knowledge_cutoff
    from ai_trader.research.vol_spec import dm_test
    from ai_trader.risk.sigma import settings

    symbols = symbols or list(settings()["symbols"])
    cut = knowledge_cutoff()
    out: dict = {"as_of": AS_OF.isoformat(), "symbols": symbols, "horizons": {}}
    for g in ("1h", "4h"):
        L = pl.concat([symbol_losses(s, g) for s in symbols])
        res: dict = {"n_symbol_bars": L.height, "per_symbol": {}}
        for sg, flt in (("llm_in_sample", pl.col("event_ts") < pl.lit(cut).cast(pl.Datetime("us", "UTC"))),
                        ("clean_holdout", pl.col("event_ts") >= pl.lit(cut).cast(pl.Datetime("us", "UTC")))):
            Ls = L.filter(flt)
            seg = {}
            for a, b in PAIRS:
                p = (Ls.group_by("event_ts").agg(pl.col(a).mean(), pl.col(b).mean(), pl.len().alias("k"))
                     .filter(pl.col("k") >= MIN_SYMBOLS).sort("event_ts"))
                d = (p[a] - p[b]).to_numpy()
                t = dm_test(d, h=1)
                seg[f"{a}-{b}"] = {"n_bars": p.height, "qlike_a": float(p[a].mean()), "qlike_b": float(p[b].mean()),
                                   "rel": float(p[a].mean() / p[b].mean() - 1), "diff": float(d.mean()), "t_hln": t["t_hln"], "p": t["p"]}
            per = Ls.group_by("symbol").agg(*[(pl.col(a).mean() / pl.col(b).mean() - 1).alias(f"{a}-{b}") for a, b in PAIRS[:3]])
            seg["share_symbols_better"] = {f"{a}-{b}": float((per[f"{a}-{b}"] < 0).mean()) for a, b in PAIRS[:3]}
            res[sg] = seg
            res["per_symbol"][sg] = {r["symbol"]: {k: round(v, 4) for k, v in r.items() if k != "symbol"} for r in per.sort("symbol").to_dicts()}
        out["horizons"][g] = res
    (out_dir() / "h_sigma_4.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def verdicts(out: dict) -> dict:
    """The registered H-SIGMA-4 rule, per segment."""
    v = {}
    for sg in ("llm_in_sample", "clean_holdout"):
        rb = [out["horizons"][g][sg]["loghar_cr_range-robust_bc"] for g in ("1h", "4h")]
        rg = out["horizons"]["1h"][sg]["loghar_cr_range-loghar_cr"]
        if sg == "llm_in_sample":
            ok = all(x["diff"] < 0 and x["t_hln"] < -2 for x in rb) and rg["diff"] < 0 and rg["t_hln"] < -2
            bad = any(x["diff"] > 0 and x["t_hln"] > 2 for x in rb + [rg])
        else:
            ok = all(x["diff"] < 0 for x in rb) and rg["diff"] < 0
            bad = any(x["diff"] > 0 and x["t_hln"] > 2 for x in rb + [rg])
        v[sg] = "supports" if ok else ("contradicts" if bad else "inconclusive")
    return v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="record the saved h_sigma_4.json verdicts (run under the results lock)")
    a = ap.parse_args(argv)
    out = json.loads((out_dir() / "h_sigma_4.json").read_text(encoding="utf-8")) if a.record else evaluate()
    v = verdicts(out)
    for g, res in out["horizons"].items():
        for sg in ("llm_in_sample", "clean_holdout"):
            for k, d in res[sg].items():
                if k != "share_symbols_better":
                    print(f"{g} {sg:14s} {k:46s} bars {d['n_bars']:6d} rel {d['rel']:+.2%} t {d['t_hln']:+.2f}")
            print(f"{g} {sg:14s} share of symbols better: {res[sg]['share_symbols_better']}")
    print("verdicts:", v)
    if a.record:
        from ai_trader.research.hypotheses import record

        for sg, verdict in v.items():
            m = {g: {k: out["horizons"][g][sg][k] for k in ("loghar_cr_range-robust_bc", "loghar_cr-robust_bc", "loghar_cr_range-loghar_cr",
                                                            "loghar_cr_range-loghar_cr_range_central", "share_symbols_better")} for g in ("1h", "4h")}
            record("H-SIGMA-4", sg, verdict, m, note="historical check of the D1 joint spec; WS04 re-measurement for the 5 majors",
                   data="data/state/risk_sigma/rv1h (23 universe-L perps, 2022-01..2026-11-27); data/research/sigma/h_sigma_4.json")
        print("recorded")


if __name__ == "__main__":
    main()
