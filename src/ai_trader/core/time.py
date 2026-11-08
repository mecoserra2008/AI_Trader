"""UTC timestamps and the look-ahead guard (mirrors crates/at-core/src/time.rs)."""

from __future__ import annotations

from datetime import UTC, datetime

FREQ_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


class LookAheadError(RuntimeError):
    """A computation used data that was not knowable at its as_of time."""


def utc(ts: datetime | str | int) -> datetime:
    """Normalise to an aware UTC datetime. Ints are epoch milliseconds; naive datetimes are rejected."""
    if isinstance(ts, int):
        return datetime.fromtimestamp(ts / 1000, tz=UTC)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    if ts.tzinfo is None:
        raise ValueError("naive datetime: always pass timezone-aware UTC timestamps")
    return ts.astimezone(UTC)


def now() -> datetime:
    return datetime.now(UTC)


def to_ms(ts: datetime) -> int:
    return int(utc(ts).timestamp() * 1000)


def assert_no_lookahead(data_max_ts: datetime, as_of: datetime) -> None:
    if utc(data_max_ts) > utc(as_of):
        raise LookAheadError(f"data_max_ts {data_max_ts} > as_of {as_of}")
