"""Forward log for H-NN10: one line per universe-L symbol with the frozen hourly ML forecast and the baseline.

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/vol_ml_log.py [--as-of 2026-11-12T03:01Z] [--symbols BTCUSDT ETHUSDT]

Appends to data/research/vol_ml_forward/<YYYY-MM-DD of as_of>.jsonl (append-only; the study keeps the first line per
symbol and forecast bar). Default as_of = now. Run it shortly after each hour's close (e.g. hh:01 UTC), so the
forecast bar is the one just opened. A line flagged 'stale' is excluded from H-NN10.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.core.config import load  # noqa: E402
from ai_trader.core.time import now  # noqa: E402
from ai_trader.research.trend_zoo_study import UNIVERSE  # noqa: E402
from ai_trader.storage.lake import Lake  # noqa: E402
from ai_trader.tools.vol_ml import hourly_vol_forecast  # noqa: E402

KEEP = ("model_id", "symbol", "as_of", "forecast_for", "data_max_ts", "sigma_next_hour_ml", "sigma_next_hour_baseline", "ratio",
        "s2", "hour_of_week_factor", "top_shap_groups", "inputs_hash", "flags")


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC) if a.as_of else now()
    out_dir = Path(a.out_dir) if a.out_dir else load().root / "data" / "research" / "vol_ml_forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    lake = Lake()
    rows = []
    logged_at = now().isoformat()
    for s in a.symbols or UNIVERSE:
        try:
            v = hourly_vol_forecast(s, as_of, lake=lake)
        except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the log
            v = {"symbol": s, "as_of": as_of.isoformat(), "flags": [f"error: {type(e).__name__}: {e}"[:200]]}
        rows.append({"logged_at": logged_at, **{k: v.get(k) for k in KEEP}})
    with (out_dir / f"{as_of:%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    for r in rows:
        sys.stdout.write(json.dumps(r, default=str) + "\n")
    return rows


if __name__ == "__main__":
    main()
