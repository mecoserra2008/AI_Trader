"""Storage layout and fetch manifest for news / events / on-chain data.

    data/research/news/
        manifest.json                 {key: {path, rows, fetched_at, status, meta}} - one entry per downloaded chunk
        gdelt/<mode>/<qid>/<start>_<end>.parquet
        onchain/<chart>/<start>_<end>.parquet
        events/events.parquet
        rss/counts.jsonl
        features/daily.parquet, features/hourly.parquet

A chunk listed in the manifest with status "ok" (or "empty") is never fetched again; incremental updates only fetch
the window after the last covered timestamp, so the first-seen vintage of every value is kept (GDELT backfills late
articles, so first-seen values are the closest thing to a point-in-time record).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trader.core.config import find_root


def news_root(root: Path | None = None) -> Path:
    env = os.environ.get("AI_TRADER_NEWS_DIR")
    if root is not None:
        p = Path(root)
    elif env:
        p = Path(env)
    else:
        p = find_root() / "data" / "research" / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class Manifest:
    """JSON manifest of fetched chunks. `has(key)` short-circuits any download."""

    def __init__(self, root: Path | None = None):
        self.root = news_root(root)
        self.path = self.root / "manifest.json"
        self.data: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    def has(self, key: str) -> bool:
        e = self.data.get(key)
        return bool(e) and e.get("status") in ("ok", "empty")

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, *, path: str | None, rows: int, status: str = "ok", **meta: Any) -> None:
        self.data[key] = {"path": path, "rows": int(rows), "status": status, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), **meta}
        _atomic_write(self.path, json.dumps(self.data, indent=1, sort_keys=True, default=str))

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self.data if k.startswith(prefix))

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


__all__ = ["Manifest", "news_root"]
