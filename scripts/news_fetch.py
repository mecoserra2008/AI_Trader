"""Download news / events / on-chain data for the BTC field program (agent F3). Idempotent: the manifest in
data/research/news/manifest.json lists every chunk, and listed chunks are never fetched again.

    PYTHONIOENCODING=utf-8 uv run python scripts/news_fetch.py events onchain gdelt-daily gdelt-intraday rss
    ... [--until 2026-11-13] [--intraday-days 84] [--pilot]
    PYTHONIOENCODING=utf-8 uv run python scripts/news_fetch.py alt-wiki alt-edits alt-hn alt-fng alt-guardian
    ... [--hn-start 2023-01-01 --hn-stop 2023-12-31]     (non-GDELT attention sources, agent W; news/alt.py, own manifest)

`--until` is a UTC midnight (default: today 00:00 UTC), so only complete days are stored.
`--pilot` limits GDELT to the 'bitcoin' query (daily, both modes) to check the pipeline before the full run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime

from ai_trader.news import alt, events, gdelt, onchain, rss
from ai_trader.news.store import Manifest


def log(msg: str) -> None:
    print(f"{datetime.now(UTC):%H:%M:%S} {msg}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["events", "onchain", "gdelt-daily", "gdelt-intraday", "rss",
                                                   "alt-wiki", "alt-edits", "alt-hn", "alt-fng", "alt-guardian"])
    ap.add_argument("--until", default=None)
    ap.add_argument("--intraday-days", type=int, default=84)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--queries", default=None, help="comma list of GDELT query ids (default: all daily / CORE_INTRADAY intraday)")
    ap.add_argument("--modes", default=None, help="comma list: timelinevolraw,timelinetone")
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--hn-start", default=None, help="alt-hn: first day (default 2017-01-01)")
    ap.add_argument("--hn-stop", default=None, help="alt-hn: last day (pilot); default until - 1 day")
    a = ap.parse_args(argv)
    until = datetime.fromisoformat(a.until).replace(tzinfo=UTC) if a.until else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    man = Manifest()
    for w in a.what:
        log(f"== {w} until {until:%Y-%m-%d}")
        if w == "events":
            df = events.load_or_build(man, refresh=True)
            log(f"  events: {df.height} rows {df['kind'].value_counts().sort('kind').rows()}")
        elif w == "onchain":
            onchain.fetch(man, until, log=log)
        elif w == "gdelt-daily":
            qids = ["bitcoin"] if a.pilot else (a.queries.split(",") if a.queries else None)
            done = gdelt.fetch_daily(man, until, qids=qids, modes=tuple(a.modes.split(",")) if a.modes else gdelt.MODES, log=log)
            log(f"  done: {done}")
        elif w == "gdelt-intraday":
            qids = ["bitcoin"] if a.pilot else (a.queries.split(",") if a.queries else gdelt.CORE_INTRADAY)
            done = gdelt.fetch_intraday(man, until, days=a.intraday_days, window_days=a.window_days, qids=qids,
                                        modes=tuple(a.modes.split(",")) if a.modes else ("timelinevolraw",), log=log)
            log(f"  done: {len(done)} chunks")
        elif w.startswith("alt-"):
            src = w[4:]
            out = alt.update(datetime.now(UTC) if src == "fng" else until, sources=(src,), log=log,
                             hn_start=date.fromisoformat(a.hn_start) if a.hn_start else alt.HN_START,
                             hn_stop=date.fromisoformat(a.hn_stop) if a.hn_stop else None)
            log(f"  {w}: {out}; alt dir {alt.alt_manifest().size_bytes() / 1e6:.1f} MB")
        elif w == "rss":
            n = rss.poll(man, log=log)
            log(f"  rss: {n} new items")
    log(f"news dir size {man.size_bytes() / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
