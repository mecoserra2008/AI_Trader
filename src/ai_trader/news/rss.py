"""Forward-only crypto headline counts from public RSS feeds (no history: the log starts the first time `poll` runs).

Only counts are kept: per item the feed, a SHA-1 of its guid/link (for de-duplication), its pubDate and the time we
first saw it. No titles or bodies are stored. known_ts of an item = seen_ts (when our poll first saw it), so the
counts are point-in-time by construction; pub_ts is kept to measure the feed's publication lag.

    PYTHONIOENCODING=utf-8 uv run python scripts/news_fetch.py rss     # run on a schedule (e.g. every 15 min)
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import polars as pl

from ai_trader.news.store import Manifest

FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "theblock": "https://www.theblock.co/rss.xml",
    "cointelegraph": "https://cointelegraph.com/rss",
}
UA = "AI_Trader research (RSS headline counts)"


def parse_items(xml_text: str) -> list[tuple[str, datetime | None]]:
    """(sha1 of guid or link, pub_ts) per item; tolerant of RSS 2.0 and Atom."""
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter():
        tag = it.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        ident, pub = None, None
        for ch in it:
            t = ch.tag.rsplit("}", 1)[-1]
            if t in ("guid", "id") and (ch.text or "").strip():
                ident = ch.text.strip()
            elif t == "link" and ident is None:
                ident = (ch.text or ch.get("href") or "").strip() or None
            elif t in ("pubDate", "published", "updated") and pub is None and ch.text:
                try:
                    pub = parsedate_to_datetime(ch.text.strip()).astimezone(UTC)
                except (TypeError, ValueError):
                    try:
                        pub = datetime.fromisoformat(ch.text.strip().replace("Z", "+00:00")).astimezone(UTC)
                    except ValueError:
                        pub = None
        if ident:
            out.append((hashlib.sha1(ident.encode("utf-8")).hexdigest(), pub))
    return out


def poll(man: Manifest, *, log=print, get=None, now: datetime | None = None) -> int:
    path = man.root / "rss" / "items.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.is_file():
        seen = {json.loads(x)["h"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}

    def _get(u):
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    get = get or _get
    t = now or datetime.now(UTC)
    new = 0
    with path.open("a", encoding="utf-8") as f:
        for feed, url in FEEDS.items():
            try:
                items = parse_items(get(url))
            except Exception as e:  # noqa: BLE001
                log(f"  rss {feed}: {e}")
                continue
            for h, pub in items:
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps({"feed": feed, "h": h, "pub_ts": pub.isoformat() if pub else None, "seen_ts": t.isoformat(timespec="seconds")}) + "\n")
                new += 1
            log(f"  rss {feed}: {len(items)} items")
    man.put("rss/last_poll", path="rss/items.jsonl", rows=len(seen), polled_at=t.isoformat(timespec="seconds"))
    return new


def hourly_counts(man: Manifest) -> pl.DataFrame:
    """Items first seen per UTC hour and feed (known_ts = hour end). First poll's backlog is flagged (first_poll)."""
    path = man.root / "rss" / "items.jsonl"
    if not path.is_file():
        return pl.DataFrame()
    df = pl.read_ndjson(path).with_columns(pl.col("seen_ts").str.to_datetime(time_zone="UTC"))
    first = df["seen_ts"].min()
    return (df.with_columns(pl.col("seen_ts").dt.truncate("1h").alias("hour"), (pl.col("seen_ts") == first).alias("first_poll"))
            .group_by("hour", "feed").agg(pl.len().alias("n"), pl.col("first_poll").any())
            .with_columns((pl.col("hour") + pl.duration(hours=1)).alias("known_ts")).sort("hour"))


__all__ = ["FEEDS", "hourly_counts", "parse_items", "poll"]
