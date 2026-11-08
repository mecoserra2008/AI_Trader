"""Week-ahead probabilistic forecast for one symbol from the validated beliefs (research/BELIEFS.md).

Monte Carlo of hourly log returns for the next 168 hours:
  sigma: HAR next-day vol (V5), converging linearly to the 30-day mean daily vol by day 5;
         hours inside the TradFi weekend scaled by the learned multiplier m (V1; central estimate);
  tails: Student-t innovations, nu = 4, unit variance;
  drift: half of the in-sample 24h forward-path mean of the current regime cell (shrunk prior, P3),
         decaying linearly to 0 over 120 h (regimes don't persist forever);
compared with the options market's cone (flat DVOL, lognormal). Writes JSON for the page.

    PYTHONIOENCODING=utf-8 uv run python scripts/eth_week_forecast.py ETHUSDT out.json
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta

import numpy as np
import polars as pl

from ai_trader.core.time import now
from ai_trader.features.vol_models import har_forecast
from ai_trader.research.calendar import add_calendar
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import CELL_LABELS, label
from ai_trader.storage.lake import Lake
from ai_trader.strategies import REGISTRY

H, N, NU = 168, 20000, 4.0


def main(sym: str, out: str) -> None:
    t = now()
    f = label(market_frame(sym, "1h", t, funding=False, oi=False), 1.0)
    last = f.tail(1).row(0, named=True)
    p0, t0 = last["close"], last["close_ts"]
    daily = (f.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
             .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n"))
             .filter(pl.col("n") == 24).sort("d"))
    har_d = float(np.sqrt(har_forecast(daily["rv"].to_numpy())[-1]))
    lr_d = float(np.sqrt(daily["rv"].tail(30).mean()))
    m = float(last["wk_m"])
    # future calendar
    fut = pl.DataFrame({"event_ts": [t0 + timedelta(hours=i) for i in range(H)]}).with_columns(pl.col("event_ts").dt.cast_time_unit("ms"))
    cal = add_calendar(fut)
    wk = cal["tradfi_weekend"].to_numpy()
    nyse = cal["nyse_open"].to_numpy()
    day = np.arange(H) / 24.0
    sig_day = har_d + (lr_d - har_d) * np.clip((day - 1) / 4, 0, 1)
    sig_h_risk = sig_day / np.sqrt(24)
    sig_h = sig_h_risk * np.where(wk, m, 1.0)
    # drift from the regime prior (in-sample path priors, 1h, h = 24)
    pp = pl.read_parquet("data/priors/path_priors_1h_v0.parquet")
    row = pp.filter((pl.col("segment") == "llm_in_sample") & (pl.col("cell") == last["cell"]) & (pl.col("h_bars") == 24))
    mz24 = float(row["mean_z"][0]) if row.height else 0.0
    mu_h = 0.5 * mz24 * sig_h_risk * np.sqrt(24) / 24 * np.clip(1 - np.arange(H) / 120, 0, 1)
    rng = np.random.default_rng(7)
    z = rng.standard_t(NU, (N, H)) / np.sqrt(NU / (NU - 2))
    paths = p0 * np.exp(np.cumsum(mu_h + sig_h * z, axis=1))
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q = np.quantile(paths, qs, axis=0)
    # implied (DVOL) cone
    lake = Lake()
    ent = sym.replace("USDT", "")
    dv = lake.read_asof("alt_series", t, start=t - timedelta(days=3), filters={"source": "deribit", "series_id": "dvol_close", "entity": ent})
    iv = float(dv.sort("event_ts")["value"][-1]) if dv.height else None
    from scipy.stats import norm

    imp = None
    if iv:
        s_h = iv / 100 / np.sqrt(24 * 365)
        cum = s_h * np.sqrt(np.arange(1, H + 1))
        imp = {str(k): (p0 * np.exp(-0.5 * cum**2 + norm.ppf(k) * cum)).round(2).tolist() for k in (0.05, 0.25, 0.75, 0.95)}
    # open shock-fade trade (strategy rules, research/strategies/shock.py): target = shock open; invalidation below shock low - 0.5 sigma
    sig = REGISTRY["shock_regime"]()(f).to_numpy()
    trade = None
    if sig[-1] != 0:
        sh = f.tail(120).filter(pl.col("z").abs() >= 4).tail(1).row(0, named=True)
        direction = 1 if sig[-1] > 0 else -1
        target = sh["open"]
        inval = (sh["low"] - 0.5 * sh["sigma_next"] * sh["close"]) if direction > 0 else (sh["high"] + 0.5 * sh["sigma_next"] * sh["close"])
        hit_t = (paths >= target) if direction > 0 else (paths <= target)
        hit_i = (paths <= inval) if direction > 0 else (paths >= inval)
        ft = np.where(hit_t.any(1), hit_t.argmax(1), H + 1)
        fi = np.where(hit_i.any(1), hit_i.argmax(1), H + 1)
        trade = {"direction": "long" if direction > 0 else "short", "shock_ts": str(sh["event_ts"]), "shock_z": round(sh["z"], 2),
                 "target": round(target, 2), "invalidation": round(inval, 2),
                 "p_target_first": float(np.mean(ft < fi)), "p_invalid_first": float(np.mean(fi < ft)), "p_neither_7d": float(np.mean((ft > H) & (fi > H)))}
    end = paths[:, -1]
    def horizon(hh: int) -> dict:
        x = paths[:, hh - 1]
        edges = np.quantile(paths[:, -1], [0.002, 0.998])
        bins = np.linspace(edges[0], edges[1], 41)
        hist, _ = np.histogram(np.clip(x, bins[0], bins[-1]), bins)
        return {"h": hh, "ts": str(t0 + timedelta(hours=hh)), "bins": bins.round(2).tolist(), "density": (hist / N).round(5).tolist(),
                "q": {str(k): round(float(np.quantile(x, k)), 2) for k in (0.05, 0.25, 0.5, 0.75, 0.95)},
                "p_up": round(float((x > p0).mean()), 3)}
    data = {
        "symbol": sym, "asof": str(t0), "price": p0, "cell": last["cell"], "cell_label": CELL_LABELS.get(last["cell"], ""),
        "vol_pct": None, "har_daily": har_d, "lr_daily": lr_d, "weekend_m": m, "dvol": iv, "regime_mean_z24": mz24,
        "hours": [str(x) for x in fut["event_ts"].to_list()], "weekend": wk.astype(int).tolist(), "nyse": nyse.astype(int).tolist(),
        "q": {str(k): q[i].round(2).tolist() for i, k in enumerate(qs)}, "implied": imp,
        "sigma_h_central": (sig_h * 100).round(4).tolist(), "sigma_h_risk": (sig_h_risk * 100).round(4).tolist(),
        "sigma_h_implied": round(iv / 100 / np.sqrt(24 * 365) * 100, 4) if iv else None,
        "horizons": [horizon(24), horizon(48), horizon(96), horizon(168)],
        "probs": {
            "p_up_7d": float((end > p0).mean()),
            "p_below_2100_7d_touch": float((paths.min(1) <= 2100).mean()),
            "p_above_2400_7d_touch": float((paths.max(1) >= 2400).mean()),
            "p_move_gt_5pct_7d": float((np.abs(end / p0 - 1) > 0.05).mean()),
            "p_weekend_move_gt_2pct": float((np.abs(paths[:, int(np.argmin(wk == 1) if wk[0] else 0) - 1] / p0 - 1) > 0.02).mean()) if wk[0] else None,
        },
        "trade": trade,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print(json.dumps({k: v for k, v in data.items() if k in ("price", "cell", "har_daily", "lr_daily", "weekend_m", "dvol", "regime_mean_z24", "probs", "trade")}, indent=1, default=str))
    print("weekend hours ahead:", int(wk.sum()), "first non-weekend hour index:", int(np.argmin(wk == 1)) if wk[0] else 0)
    for hz in data["horizons"]:
        print(hz["h"], hz["q"], "p_up", hz["p_up"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT", sys.argv[2] if len(sys.argv) > 2 else "eth_forecast.json")
