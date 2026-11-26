"""scripts/xvenue_tape.py feed recovery: a venue that rejects a request (400 after a PC suspend: stale paging cursor) must
be probed again, not switched off for good; a long gap since the last good poll drops the cursor."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("xvenue_tape", ROOT / "scripts" / "xvenue_tape.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def _venues(monkeypatch, clock):
    monkeypatch.setattr(T.time, "time", lambda: clock[0])
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    return T.Venues(["BTC"])


def test_rejected_feed_is_retried_and_recovers(monkeypatch):
    clock = [1_000_000.0]
    V = _venues(monkeypatch, clock)
    b = V.buf[("bn", "BTC")]
    calls = {"n": 0, "fail": False}

    def fake(h, s, buf):
        calls["n"] += 1
        if calls["fail"]:
            raise T.BadSymbol("400 fromId too old")
        buf.last_id = 123

    monkeypatch.setattr(V, "binance_perp", fake)
    V.fetch(None, "bn", "BTC")
    assert b.ok and b.last_id == 123 and b.last_ok_ts == clock[0]

    calls["fail"] = True
    clock[0] += 15
    V.fetch(None, "bn", "BTC")
    assert not b.ok and b.last_id is None                        # cursor dropped, feed parked
    n = calls["n"]
    clock[0] += 60
    V.fetch(None, "bn", "BTC")
    assert calls["n"] == n                                        # still inside the retry delay: not called

    calls["fail"] = False
    clock[0] += T.RETRY_DISABLED_S
    V.fetch(None, "bn", "BTC")
    assert b.ok and b.err == "" and calls["n"] == n + 1           # probed again and recovered


def test_stale_cursor_is_dropped_after_a_long_gap(monkeypatch):
    clock = [2_000_000.0]
    V = _venues(monkeypatch, clock)
    b = V.buf[("bn", "BTC")]
    seen = []

    def fake(h, s, buf):
        seen.append(buf.last_id)
        buf.last_id = 999

    monkeypatch.setattr(V, "binance_perp", fake)
    V.fetch(None, "bn", "BTC")
    clock[0] += 20
    V.fetch(None, "bn", "BTC")
    assert seen == [None, 999]                                    # normal paging continues from the cursor
    clock[0] += T.STALE_RESET_S + 1                               # e.g. the PC was suspended
    V.fetch(None, "bn", "BTC")
    assert seen[-1] is None                                       # restarted from the latest trades


def test_never_listed_venue_stays_off(monkeypatch):
    clock = [3_000_000.0]
    V = _venues(monkeypatch, clock)
    b = V.buf[("db", "BTC")]
    b.ok, b.err = False, "not listed"
    called = []
    monkeypatch.setattr(V, "deribit", lambda h, s, buf: called.append(1))
    clock[0] += 10 * T.RETRY_DISABLED_S
    V.fetch(None, "db", "BTC")
    assert called == []
