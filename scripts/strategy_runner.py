"""Strategy runner CLI (agent K): strategies -> bias gate -> OrderIntents -> `atx exec-intents` (agent L).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/strategy_runner.py --mode replay --start 2026-10-13 --end 2026-11-12 \
        [--strategies tsmom zscore_reversion shock_regime] [--symbols BTCUSDT ETHUSDT SOLUSDT] [--workers 3] [--no-jump] [--no-atx] [--equity 100 10000]
    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/strategy_runner.py --mode paper [--strategies ...] [--symbols ...]

Research / paper only: any other mode, a mainnet environment or AI_TRADER_ALLOW_MAINNET set in this shell is refused.
See src/ai_trader/execution/runner.py and docs/EXECUTION_PIPELINE.md.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from ai_trader.execution.runner import (
    MODES,
    SafetyError,
    print_summary,
    run_paper,
    run_replay,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, help=f"one of {MODES}")
    ap.add_argument("--start", type=date.fromisoformat, help="replay: first UTC date")
    ap.add_argument("--end", type=date.fromisoformat, help="replay: last UTC date (inclusive)")
    ap.add_argument("--strategies", nargs="*", default=None, help="default: all registered strategies + zoo_ensemble")
    ap.add_argument("--symbols", nargs="*", default=None, help="default: BTCUSDT ETHUSDT SOLUSDT")
    ap.add_argument("--workers", type=int, default=3, help="processes, at most 4")
    ap.add_argument("--no-jump", action="store_true", help="driftless Gaussian p_tp instead of the jump-aware Monte Carlo")
    ap.add_argument("--no-atx", action="store_true", help="write intents only; do not call atx exec-intents")
    ap.add_argument("--equity", type=float, nargs="*", default=None,
                    help="starting equity (USDT) of the paper book; several values = one atx replay per value (e.g. 100 10000)")
    a = ap.parse_args(argv)
    try:
        if a.mode == "replay":
            if not (a.start and a.end) or a.end < a.start:
                ap.error("--mode replay needs --start <= --end")
            s = run_replay(a.start, a.end, a.strategies, a.symbols, workers=a.workers, jump=not a.no_jump, call_atx=not a.no_atx,
                           equities=a.equity)
        elif a.mode == "paper":
            s = run_paper(a.strategies, a.symbols, workers=a.workers, jump=not a.no_jump, call_atx=not a.no_atx, equities=a.equity)
        else:
            raise SafetyError(f"mode {a.mode!r} refused: only {MODES} (research / paper only)")
    except SafetyError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print_summary(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
