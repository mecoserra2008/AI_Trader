"""Week-path frequencies (the user's 'chop Monday, fall Tuesday, rise Wednesday-Thursday') and H-EV2 (bounce after a down CPI day).

    PYTHONIOENCODING=utf-8 uv run python scripts/path_study.py [--record]

BTC daily UTC returns from the lake (1h -> days); Nasdaq-100 and S&P 500 from Yahoo daily. z = r / EWMA(0.94) sigma of the
prior days. CPI release dates = ALFRED vintage dates of CPIAUCSL (FRED key from .env, never printed). Paths:
  exact: |z_Mon| < 0.5, z_Tue < -0.5, (z_Wed + z_Thu)/sqrt(2) > 0.5
  loose: |z_Mon| < 1, z_Tue < 0, z_Wed + z_Thu > 0
Frequencies over all weeks, CPI-on-Tuesday weeks, CPI weeks aligned on the CPI day (day-1 / CPI day / +1..+2), and weeks
starting in a high-vol state; vs the product of marginals (independence) and a driftless Student-t Monte Carlo with the
CPI-day vol multiplier. Descriptive except H-EV2, which follows its registered rule.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from dow_regime_study import ewma_sigma, yahoo  # noqa: E402
from regime_context import daily_closes  # noqa: E402

from ai_trader.core.config import load  # noqa: E402
from ai_trader.core.time import now  # noqa: E402
from ai_trader.storage.lake import Lake  # noqa: E402

CUTOFF = date(2026, 7, 1)


def cpi_dates() -> list[date]:
    load_dotenv(load().root / ".env")
    key = os.environ["FRED_API_KEY"]
    u = f"https://api.stlouisfed.org/fred/series/vintagedates?series_id=CPIAUCSL&api_key={key}&file_type=json&realtime_start=2005-01-01&limit=10000"
    return [date.fromisoformat(x) for x in json.load(urllib.request.urlopen(u, timeout=30))["vintage_dates"]]


def zframe(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("d")
    c = df["close"].to_numpy()
    r = np.r_[np.nan, np.diff(np.log(c))]
    s = ewma_sigma(r)
    hi = pl.Series(s).rolling_quantile(0.67, window_size=180, min_samples=90).to_numpy()
    return df.with_columns(pl.Series("r", r), pl.Series("sig", s), pl.Series("z", r / s), pl.Series("highvol", s > hi)).drop_nulls("z").filter(pl.col("z").is_finite())


def weeks(zf: pl.DataFrame) -> pl.DataFrame:
    """One row per ISO week with z for Mon..Thu (requires all four days)."""
    w = zf.with_columns(pl.col("d").dt.weekday().alias("wd"), (pl.col("d") - pl.duration(days=pl.col("d").dt.weekday() - 1)).alias("monday"))
    p = w.filter(pl.col("wd") <= 4).pivot(on="wd", index="monday", values="z", aggregate_function="first").drop_nulls()
    p = p.rename({"1": "mon", "2": "tue", "3": "wed", "4": "thu"})
    hv = w.filter(pl.col("wd") == 1).select("monday", pl.col("highvol").alias("hv_mon"))
    return p.join(hv, on="monday", how="left")


def path_flags(w: pl.DataFrame) -> pl.DataFrame:
    return w.with_columns(
        ((pl.col("mon").abs() < 0.5) & (pl.col("tue") < -0.5) & ((pl.col("wed") + pl.col("thu")) / math.sqrt(2) > 0.5)).alias("exact"),
        ((pl.col("mon").abs() < 1) & (pl.col("tue") < 0) & ((pl.col("wed") + pl.col("thu")) > 0)).alias("loose"),
    )


def freq(w: pl.DataFrame) -> dict:
    n = w.height
    if n == 0:
        return {"n": 0}
    pm = float((w["mon"].abs() < 0.5).mean())
    pt = float((w["tue"] < -0.5).mean())
    pr = float(((w["wed"] + w["thu"]) / math.sqrt(2) > 0.5).mean())
    pml, ptl, prl = float((w["mon"].abs() < 1).mean()), float((w["tue"] < 0).mean()), float(((w["wed"] + w["thu"]) > 0).mean())
    return {"n": n, "exact": float(w["exact"].mean()), "exact_if_independent": pm * pt * pr, "loose": float(w["loose"].mean()), "loose_if_independent": pml * ptl * prl,
            "p_mon_chop": pm, "p_tue_fall": pt, "p_wedthu_rise": pr, "p_rise_given_tue_fall": float((((w["wed"] + w["thu"]) / math.sqrt(2)) > 0.5).filter(w["tue"] < -0.5).mean()) if (w["tue"] < -0.5).any() else None}


def monte_carlo(n=200_000, nu=4.0, tue_mult=1.4, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    t = rng.standard_t(nu, size=(n, 4)) / math.sqrt(nu / (nu - 2))
    t[:, 1] *= tue_mult
    exact = (np.abs(t[:, 0]) < 0.5) & (t[:, 1] < -0.5) & ((t[:, 2] + t[:, 3]) / math.sqrt(2) > 0.5)
    loose = (np.abs(t[:, 0]) < 1) & (t[:, 1] < 0) & ((t[:, 2] + t[:, 3]) > 0)
    return {"exact": float(exact.mean()), "loose": float(loose.mean()), "nu": nu, "tue_vol_mult": tue_mult}


def cpi_aligned(zf: pl.DataFrame, cpi: list[date]) -> pl.DataFrame:
    d = zf["d"].to_list()
    z = zf["z"].to_numpy()
    idx = {x: i for i, x in enumerate(d)}
    rows = []
    for c in cpi:
        i = idx.get(c)
        if i is None or i < 1 or i + 2 >= len(z):
            continue
        rows.append({"cpi": c, "pre": z[i - 1], "day": z[i], "next2": z[i + 1] + z[i + 2], "dow": c.isoweekday()})
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    t = now()
    cpi = cpi_dates()
    btc = zframe(daily_closes(Lake(), "BTCUSDT", t, datetime(2019, 1, 1, tzinfo=UTC)).filter(pl.col("n") >= 20).select("d", "close"))
    assets = {"BTC": btc, "NDX": zframe(yahoo("^NDX").select("d", "close")), "SPX": zframe(yahoo("^GSPC").select("d", "close"))}
    out = {"cpi_dates_n": len(cpi), "paths": {}, "cpi_aligned": {}, "ev2": {}}
    cpi_tue_mondays = {c - timedelta(days=1) for c in cpi if c.isoweekday() == 2}
    for name, zf in assets.items():
        w = path_flags(weeks(zf))
        res = {"all_weeks": freq(w), "cpi_on_tuesday_weeks": freq(w.filter(pl.col("monday").is_in(list(cpi_tue_mondays)))),
               "high_vol_start": freq(w.filter(pl.col("hv_mon").fill_null(False))),
               "cpi_tuesday_and_high_vol": freq(w.filter(pl.col("monday").is_in(list(cpi_tue_mondays)) & pl.col("hv_mon").fill_null(False)))}
        out["paths"][name] = res
        ca = cpi_aligned(zf, cpi)
        if ca.height:
            pre_chop, down, up = ca["pre"].abs() < 0.5, ca["day"] < -0.5, ca["next2"] / math.sqrt(2) > 0.5
            out["cpi_aligned"][name] = {"n": ca.height, "p_cpi_day_down_05": float(down.mean()), "p_cpi_day_up_05": float((ca["day"] > 0.5).mean()),
                                        "mean_abs_z_cpi_day": float(ca["day"].abs().mean()), "p_path_aligned": float((pre_chop & down & up).mean()),
                                        "p_bounce_given_down": float(up.filter(down).mean()) if down.any() else None,
                                        "p_bounce_given_up_day": float(up.filter(ca["day"] > 0.5).mean()) if (ca["day"] > 0.5).any() else None}
            for seg, part in [("llm_in_sample", ca.filter(pl.col("cpi") < CUTOFF)), ("clean_holdout", ca.filter(pl.col("cpi") >= CUTOFF))]:
                dn = part.filter(pl.col("day") < 0)
                if dn.height >= 2:
                    y = dn["next2"].to_numpy()
                    m = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
                    out["ev2"].setdefault(seg, {})[name] = {"n_down_cpi_days": dn.height, "mean_next2_z": float(y.mean()), "t": float(m.tvalues[0]),
                                                            "mean_next2_z_after_up": float(part.filter(pl.col("day") >= 0)["next2"].mean()) if part.filter(pl.col("day") >= 0).height else None}
        # cross-asset on CPI days: BTC vs NDX sign agreement
    b = cpi_aligned(assets["BTC"], cpi).select("cpi", pl.col("day").alias("btc"))
    nq = cpi_aligned(assets["NDX"], cpi).select("cpi", pl.col("day").alias("ndx"), pl.col("next2").alias("ndx_next2"))
    j = b.join(nq, on="cpi").join(cpi_aligned(assets["BTC"], cpi).select("cpi", pl.col("next2").alias("btc_next2")), on="cpi")
    if j.height > 10:
        out["cpi_cross"] = {"n": j.height, "corr_btc_ndx_cpi_day": float(np.corrcoef(j["btc"], j["ndx"])[0, 1]),
                            "sign_agreement": float((np.sign(j["btc"]) == np.sign(j["ndx"])).mean()),
                            "btc_next2_when_ndx_cpi_day_down": float(j.filter(pl.col("ndx") < -0.5)["btc_next2"].mean()) if j.filter(pl.col("ndx") < -0.5).height else None,
                            "n_ndx_down": j.filter(pl.col("ndx") < -0.5).height}
    out["monte_carlo_driftless"] = {"tue_x1.0": monte_carlo(tue_mult=1.0), "tue_x1.4": monte_carlo(tue_mult=1.4), "tue_x1.7": monte_carlo(tue_mult=1.7)}
    print(json.dumps(out, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else str(x)))
    p = load().root / "research" / "reports" / "path_study_2026-11-08.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    if a.record:
        from ai_trader.research.hypotheses import record
        ins = out["ev2"].get("llm_in_sample", {}).get("BTC")
        ho = out["ev2"].get("clean_holdout", {}).get("BTC")
        if ins:
            record("H-EV2", "llm_in_sample", "supports" if ins["mean_next2_z"] > 0 and ins["t"] > 2 else "contradicts", out["ev2"]["llm_in_sample"], data="BTC/NDX/SPX daily; ALFRED CPI vintages; scripts/path_study.py")
        if ho:
            record("H-EV2", "clean_holdout", "supports" if ho["mean_next2_z"] > 0 else "contradicts", out["ev2"]["clean_holdout"], note=f"n={ho['n_down_cpi_days']} only", data="scripts/path_study.py")
        else:
            record("H-EV2", "clean_holdout", "inconclusive", {"note": "fewer than 2 down CPI days after the cutoff"}, data="scripts/path_study.py")
        print("recorded H-EV2")


if __name__ == "__main__":
    main()
