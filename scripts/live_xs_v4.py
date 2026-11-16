"""Hourly shadow loop of xs.v4 (H-XS4). Context only; never trades. Separate from scripts/live_nn.py (v1 / v2.1 / xs.v3 untouched).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_xs_v4.py --once
    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_xs_v4.py --loop      (runs at HH:02:30 UTC every hour)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def main() -> None:
    from ai_trader.live import xs_v4

    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    if a.once or not a.loop:
        ln = xs_v4.safe_run()
        sys.stdout.write(f"{datetime.now(UTC).isoformat()} {'ok ' + ln['id'] + ' n=' + str(ln['n_present']) if ln else 'FAILED (see errors.log)'}\n")
        return
    while True:
        t = xs_v4.next_run(datetime.now(UTC))
        time.sleep(max(1.0, (t - datetime.now(UTC)).total_seconds()))
        ln = xs_v4.safe_run()
        sys.stdout.write(f"{datetime.now(UTC).isoformat()} {'ok ' + ln['id'] + ' n=' + str(ln['n_present']) if ln else 'FAILED (see errors.log)'}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
