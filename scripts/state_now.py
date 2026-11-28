"""Print compact state cards for the current moment: the inputs the decision layer reads.

    PYTHONIOENCODING=utf-8 uv run python scripts/state_now.py [SYMBOL ...]

Since 2026-11-27 (user decisions D4 / D5): the SIGNALS line shows the paper trend sleeves (agent.trend_sleeves: spec A
zoo_ew, spec B tsmom_30d) instead of the retired 4h ema_stack; a |z| >= 4 shock is context only (the shock fade H-S2 /
shock_regime sleeve is retired, no continuation / reversal playbook).
"""

from __future__ import annotations

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


def card(sym: str) -> str:
    t = now()
    f = label(market_frame(sym, "1h", t), 1.0)
    f4 = market_frame(sym, "4h", t, funding=False, oi=False)
    cal = add_calendar(f.tail(1))
    last = f.tail(1).row(0, named=True)
    # |z| >= 4 hourly shocks in the last 72 bars: context only (the H-S2 shock fade is retired, D5 2026-11-27)
    recent = f.tail(72).filter(pl.col("z").abs() >= 4)
    shock = "none in last 72h"
    if recent.height:
        r = recent.tail(1).row(0, named=True)
        hrs = int((last["event_ts"] - r["event_ts"]).total_seconds() // 3600)
        shock = f"{r['z']:+.1f}σ {hrs}h ago in {r['cell']} ({r['vol_state']} vol); context only (shock fade H-S2 retired)"
    from ai_trader.agent.trend_sleeves import signals_from_frame

    sg = signals_from_frame(sym, f4, t)
    daily = (f.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
             .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n")).filter(pl.col("n") == 24).sort("d"))
    har = float(np.sqrt(har_forecast(daily["rv"].to_numpy())[-1]))
    vol_pct = float((f["sigma_risk_next"].tail(24 * 365) < last["sigma_risk_next"]).mean())
    fund7 = f["funding_rate"].tail(24 * 7).drop_nulls().unique().mean()
    lake = Lake()
    dv = lake.read_asof("alt_series", t, start=t - timedelta(days=3), filters={"source": "deribit", "series_id": "dvol_close", "entity": sym[:3]}) if sym[:3] in ("BTC", "ETH") else pl.DataFrame()
    vrp = ""
    if dv.height:
        iv = dv.sort("event_ts")["value"][-1]
        vrp = f"DVOL {iv:.1f} vs HAR {har * np.sqrt(365) * 100:.1f} ann. -> implied/realised-forecast {iv / (har * np.sqrt(365) * 100):.2f}x"
    lines = [
        f"{sym} @ {last['close']:,.2f}  {last['event_ts']:%a %H:%M} UTC  session={cal['session'][0]}{'  [TRADFI WEEKEND]' if cal['tradfi_weekend'][0] else ''}",
        f"  REGIME  {last['cell']} {CELL_LABELS.get(last['cell'], '')}  (dir score {last['dir_score']:+.2f}, vol {last['vol_state']})",
        f"  VOL     risk σ/h {last['sigma_risk_next']:.2%} (1y pct {vol_pct:.0%}) | central σ/h {last['sigma_rb_next']:.2%} (weekend m {last['wk_m']:.2f}) | HAR next day {har:.2%}",
        f"  SHOCK   {shock}",
        f"  SIGNALS trend sleeves (4h): zoo_ew spec A {sg.last('zoo_ew'):+.2f} | tsmom_30d spec B {sg.last('tsmom_30d'):+.2f} (entry |s| >= 0.30)",
        f"  CARRY   funding 7d mean {fund7 * 100:+.4f}%/8h ({fund7 * 3 * 365 * 100:+.1f}%/yr)" if fund7 is not None else "  CARRY   n/a",
    ]
    if vrp:
        lines.append(f"  OPTIONS {vrp}")
    return "\n".join(lines)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(card(s) + "\n")
