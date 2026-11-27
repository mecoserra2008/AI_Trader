"""Hourly forward shadow of the D1 stop sigma (ai_trader.risk.sigma / risk.sigma_shadow): robust_bc, loghar_cr and
loghar_cr_range (stop + central) at 1h / 4h, robust_bc vs vol.har at 1d, for the risk_sigma.symbols universe.

    PYTHONIOENCODING=utf-8 uv run python scripts/sigma_shadow.py --loop          # desk loop (scripts/ops/desk_start.sh)
    PYTHONIOENCODING=utf-8 uv run python scripts/sigma_shadow.py --once [--symbols BTCUSDT ...] [--no-refresh]
    PYTHONIOENCODING=utf-8 uv run python scripts/sigma_shadow.py --seed          # 5m RV store from the research caches

--loop runs at hh:02:30 UTC; it recomputes the next run from the wall clock every 30 s (suspend-safe: a run after a
wake-up covers the current hour and is flagged late), appends data/journal/sigma_shadow/<YYYY-MM-DD>.jsonl, refreshes
the 5m store from public Bybit klines, and then runs the scorer with the switch gate (scripts/sigma_shadow_score.py).
Research / risk input only: it never places, amends or cancels anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")

from ai_trader.risk import sigma_shadow as sh  # noqa: E402
from ai_trader.risk import sigma_store  # noqa: E402
from ai_trader.risk.sigma import settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--loop", action="store_true")
    m.add_argument("--once", action="store_true")
    m.add_argument("--seed", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--as-of", default=None, help="--once only: replay time (ISO, UTC); lines go to data/journal/sigma_shadow/replay/, never scored")
    ap.add_argument("--no-refresh", action="store_true", help="--once: do not fetch 5m klines first")
    ap.add_argument("--no-score", action="store_true", help="--loop: do not run the scorer / gate after each hour")
    a = ap.parse_args(argv)
    syms = a.symbols or list(settings()["symbols"])
    if a.seed:
        print(json.dumps(sigma_store.seed(syms), indent=1, default=str))
        print(json.dumps(sigma_store.refresh(syms), indent=1, default=str))
    elif a.once:
        print(json.dumps(sh.run_once(as_of=a.as_of, symbols=syms, refresh=not a.no_refresh), indent=1, default=str))
    else:
        sh.loop(symbols=syms, score=not a.no_score)
    return 0


if __name__ == "__main__":
    sys.exit(main())
