"""Desk collectors: RSS headlines (titles kept) and Reddit (official OAuth API) every 15 minutes, point in time.

    PYTHONIOENCODING=utf-8 uv run python scripts/desk_collect.py            # loop forever, every 15 min (the main session starts it)
    PYTHONIOENCODING=utf-8 uv run python scripts/desk_collect.py --once     # one pass
        [--interval-min 15] [--rss-counts]  (--rss-counts also runs news.rss.poll, the forward-only count log of news.rss)

Stores under data/research/sentiment/ (<= 200 MB, pruned): news/<date>.jsonl, reddit/<date>.jsonl, status.json.
Reddit needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT in .env (a script app); the values are never
printed, only present / missing. Without them the Reddit part is skipped and flagged "reddit: pending keys"; the
sentiment panel is then built from the RSS headlines only. Never scrapes X or Facebook. Stops cleanly on Ctrl+C;
writes its PID to data/research/sentiment/collector.pid so it can be stopped by PID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.core.config import load  # noqa: E402
from ai_trader.desk import news as N  # noqa: E402
from ai_trader.desk import sentiment as S  # noqa: E402
from ai_trader.desk.public import Bybit  # noqa: E402
from ai_trader.desk.text import ALIASES, TickerIndex  # noqa: E402
from ai_trader.desk.xs import base_of, select_universe  # noqa: E402

BASES_FILE = "universe_bases.json"


def universe_bases(store: Path, log=print) -> list[str]:
    """Top-60 Bybit perp bases (refreshed each pass; the cached list is used if Bybit is unreachable)."""
    p = store / BASES_FILE
    try:
        by = Bybit()
        tk, _ = by.tickers()
        bases = [base_of(u["symbol"]) for u in select_universe(tk, by.instruments(), 60)]
        p.write_text(json.dumps({"ts": datetime.now(UTC).isoformat(), "bases": bases}), encoding="utf-8")
        return bases
    except Exception as e:  # noqa: BLE001
        log(f"  universe: {type(e).__name__}; using cached list")
        return json.loads(p.read_text(encoding="utf-8"))["bases"] if p.is_file() else sorted(set(ALIASES.values()))


def one_pass(root: Path, *, rss_counts: bool = False, log=print) -> dict:
    t = datetime.now(UTC)
    store = S.store_root(root)
    bases = universe_bases(store, log)
    index = TickerIndex(set(bases) | {"BTC", "ETH", "SOL"})
    stat = {"ts": t.isoformat(timespec="seconds"), "keys": S.key_status()}
    stat["news"] = N.collect(store, index, now=t, log=log)
    client = S.Reddit.from_env()
    stat["reddit"] = S.collect_reddit(store, client, index, now=t, log=log)
    if rss_counts:
        from ai_trader.news.rss import poll
        from ai_trader.news.store import Manifest

        try:
            stat["rss_counts_new"] = poll(Manifest(), log=lambda *_: None)
        except Exception as e:  # noqa: BLE001
            stat["rss_counts_new"] = f"error: {type(e).__name__}"
    stat["prune"] = S.prune(store, t)
    sp = store / "status.json"
    prev = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
    status = {"collected_since": prev.get("collected_since") or t.isoformat(timespec="seconds"), "last_collect": stat["ts"],
              "reddit": "pending keys" if client is None else stat["reddit"].get("reddit"), "last": stat}
    sp.write_text(json.dumps(status, indent=1), encoding="utf-8")
    log(f"{stat['ts']} news +{stat['news']['new']} reddit {status['reddit']} +{stat['reddit'].get('new', 0)} store {stat['prune']['mb']} MB")
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-min", type=float, default=15.0)
    ap.add_argument("--rss-counts", action="store_true")
    a = ap.parse_args()
    root = load().root
    store = S.store_root(root)
    pid = store / "collector.pid"
    if not a.once:
        pid.write_text(str(os.getpid()), encoding="utf-8")
    try:
        while True:
            t0 = time.monotonic()
            try:
                one_pass(root, rss_counts=a.rss_counts)
            except Exception as e:  # noqa: BLE001 - a failed pass never kills the loop
                print(f"{datetime.now(UTC):%H:%M:%S} pass failed: {type(e).__name__}: {e}")
            if a.once:
                break
            time.sleep(max(5.0, a.interval_min * 60 - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if not a.once and pid.is_file() and pid.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid.unlink()


if __name__ == "__main__":
    main()
