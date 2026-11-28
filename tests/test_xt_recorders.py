"""XT recorders and helpers (ai_trader.research.xt, scripts/xt_bybit_l1.py, scripts/xt_ctrader_quotes.py, scripts/xt_study.py):
websocket framing, top-of-book state, hourly chunking with crash recovery, session checks, the change-only cTrader poller with
its unsubscribed / app-closed handling, the ob200 top-of-book extractor and the Hayashi-Yoshida lead-lag estimator."""

from __future__ import annotations

import importlib.util
import json
import struct
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.research import xt as X

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ms(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp() * 1000)


# ------------------------------------------------------------------------------------------------ rules and sessions


def test_spy_is_refused_everywhere():
    with pytest.raises(PermissionError):
        X.refuse_spy(["BTCUSDT", "SPYUSDT"])
    with pytest.raises(PermissionError):
        X.refuse_spy(["spy"])
    assert "SPYUSDT" not in X.bybit_symbols()
    assert not any(s.upper().startswith("SPY") for s in X.ctrader_symbols())
    assert X.refuse_spy(["SPXLUSDT", " SPX500 "]) == ["SPXLUSDT", "SPX500"]


XAU_SESS = {"timeZone": "Eastern Standard Time", "marketIsAlwaysOpen": False, "sessions": [
    {"startSecond": 64800, "endSecond": 147599}, {"startSecond": 151200, "endSecond": 233999},
    {"startSecond": 237600, "endSecond": 320399}, {"startSecond": 324000, "endSecond": 406799},
    {"startSecond": 410400, "endSecond": 492899}]}


def test_session_open_follows_the_weekly_schedule_in_eastern_time():
    assert not X.session_open(ms("2026-11-28T12:00:00"), XAU_SESS)      # Saturday
    assert not X.session_open(ms("2026-11-29T22:30:00"), XAU_SESS)      # Sunday before the 23:00 UTC reopen
    assert X.session_open(ms("2026-11-29T23:00:00"), XAU_SESS)          # reopen (18:00 ET)
    assert X.session_open(ms("2026-11-30T12:00:00"), XAU_SESS)          # Monday
    assert not X.session_open(ms("2026-11-30T22:30:00"), XAU_SESS)      # daily break 17:00-18:00 ET
    assert X.session_open(ms("2026-12-04T21:50:00"), XAU_SESS)          # Friday before the 21:55 close
    assert not X.session_open(ms("2026-12-04T22:00:00"), XAU_SESS)
    assert X.session_open(0, {"marketIsAlwaysOpen": True})


# ------------------------------------------------------------------------------------------------ chunk sink


def test_chunk_sink_writes_one_file_per_hour_and_recovers_parts(tmp_path):
    sch = X.pl_schema({"recv_ts": "Int64", "v": "Float64"})
    s = X.ChunkSink(tmp_path, "st", sch, flush_rows=2)
    t0 = ms("2026-11-28T10:59:58")
    for i in range(5):
        s.add((t0 + i * 1000, float(i)))               # 10:59:58, :59, 11:00:00, :01, :02
    s.close()
    a = pl.read_parquet(tmp_path / "st" / "date=2026-11-28" / "10.parquet")
    b = pl.read_parquet(tmp_path / "st" / "date=2026-11-28" / "11.parquet")
    assert a["v"].to_list() == [0.0, 1.0] and b["v"].to_list() == [2.0, 3.0, 4.0]
    assert not list((tmp_path / "st").glob("date=*/*.part*"))
    # a crash leaves part files of a past hour: the next sink compacts them at start-up
    d = tmp_path / "st" / "date=2026-11-27"
    d.mkdir(parents=True)
    pl.DataFrame({"recv_ts": [ms("2026-11-27T05:00:01")], "v": [9.0]}, schema=sch).write_parquet(d / "05.partX_0001.parquet")
    X.ChunkSink(tmp_path, "st", sch)
    assert pl.read_parquet(d / "05.parquet")["v"].to_list() == [9.0]
    assert not list(d.glob("*.part*"))


# ------------------------------------------------------------------------------------------------ websocket framing


class FakeSock:
    def __init__(self, chunks):
        self.chunks, self.sent = list(chunks), []

    def recv(self, n):
        if not self.chunks:
            raise TimeoutError
        c = self.chunks.pop(0)
        if c is None:
            raise TimeoutError
        return c

    def sendall(self, b):
        self.sent.append(b)

    def close(self):
        pass


def frame(op: int, payload: bytes, fin: bool = True) -> bytes:
    n = len(payload)
    h = bytes([(0x80 if fin else 0) | op])
    h += bytes([n]) if n < 126 else bytes([126]) + struct.pack(">H", n)
    return h + payload


def _ws(chunks):
    ws = X.WebSocket.__new__(X.WebSocket)
    ws.sock, ws.buf, ws.frag, ws.frag_op = FakeSock(chunks), b"", [], 0
    return ws


def test_websocket_reassembles_frames_answers_pings_and_never_eats_partial_frames():
    big = json.dumps({"x": "y" * 300}).encode()
    f = frame(0x1, big)
    ws = _ws([f[:3], None, f[3:] + frame(0x9, b"hi") + frame(0x1, b'{"a":', fin=False) + frame(0x0, b"1}")])
    assert ws.recv() is None                       # partial frame + timeout: nothing consumed
    assert ws.recv() == big.decode()
    assert ws.recv() == '{"a":1}'                  # fragmented message, after an answered ping
    pong = ws.sock.sent[0]
    assert pong[0] == 0x8A and pong[1] & 0x80       # masked pong
    mask = pong[2:6]
    assert bytes(b ^ mask[i % 4] for i, b in enumerate(pong[6:])) == b"hi"
    ws2 = _ws([frame(0x8, struct.pack(">H", 1000))])
    with pytest.raises(X.WSClosed):
        ws2.recv()


# ------------------------------------------------------------------------------------------------ bybit parsing


def ob1(kind, b, a, ts=1, cts=1, u=1):
    return {"topic": "orderbook.1.BTCUSDT", "type": kind, "ts": ts, "cts": cts,
            "data": {"s": "BTCUSDT", "b": b, "a": a, "u": u, "seq": 7}}


def test_l1_state_emits_rows_only_on_changes():
    st = X.L1State()
    r = st.apply(10, ob1("snapshot", [["100.0", "1"]], [["100.5", "2"]]))
    assert r[3:8] == ("BTCUSDT", 100.0, 1.0, 100.5, 2.0) and r[-1] == "s"
    assert st.apply(11, ob1("delta", [["100.0", "1"]], [])) is None          # same top
    r = st.apply(12, ob1("delta", [], [["100.4", "3"]]))
    assert r[6:8] == (100.4, 3.0)
    r = st.apply(13, ob1("delta", [["100.0", "0"]], [], u=2))                # bid removed
    assert np.isnan(r[4])
    r = st.apply(14, ob1("snapshot", [["99.0", "1"]], [["100.4", "3"]], u=3))
    assert r[-1] == "s"
    assert st.apply(15, ob1("snapshot", [["99.0", "1"]], [["100.4", "3"]], u=3))[-1] == "r"   # keep-alive re-send


def test_trades_parse_and_recorder_routes_messages(tmp_path):
    B = _load("xt_bybit_l1")
    rec = B.Recorder(["BTCUSDT", "ETHUSDT"], tmp_path, None, frozenset({"ETHUSDT"}))
    rec.handle(json.dumps(ob1("snapshot", [["100.0", "1"]], [["100.5", "2"]])))
    rec.handle(json.dumps({"topic": "publicTrade.BTCUSDT", "type": "snapshot", "ts": 5,
                           "data": [{"T": 4, "s": "BTCUSDT", "S": "Buy", "v": "0.01", "p": "100.5", "i": "abc", "BT": False}]}))
    rec.handle(json.dumps({"topic": "publicTrade.ETHUSDT", "type": "snapshot", "ts": 5,
                           "data": [{"T": 4, "s": "ETHUSDT", "S": "Sell", "v": "1", "p": "10", "i": "x", "BT": False}]}))
    rec.handle("not json")
    assert len(rec.trades.rows) == 1 and rec.n_trade["ETHUSDT"] == 1            # counted, not stored
    assert len(rec.book.rows) == 1 and rec.trades.rows[0][3:7] == ("BTCUSDT", "B", 100.5, 0.01)
    rec.book.close()
    rec.trades.close()
    assert pl.read_parquet(next((tmp_path / "bybit_trades").glob("date=*/*.parquet"))).height == 1


# ------------------------------------------------------------------------------------------------ ctrader poller


class FakeClient:
    def __init__(self, quotes, fail_after=None, **_):
        self.quotes, self.n, self.fail_after = quotes, 0, fail_after

    def call(self, tool, **a):
        assert tool in ("get_spot_prices", "get_server_time")
        if tool == "get_server_time":
            return {"unixMs": 1_795_000_000_000}
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise ConnectionRefusedError("app closed")
        q = self.quotes[a["symbolName"]]
        if q is None:
            raise RuntimeError("get_spot_prices: No live quote for X - symbol is unsubscribed")
        return q.pop(0) if isinstance(q, list) and len(q) > 1 else (q[0] if isinstance(q, list) else q)


def test_ctrader_poller_records_changes_only_and_parks_unsubscribed_symbols(tmp_path, monkeypatch):
    Q = _load("xt_ctrader_quotes")
    monkeypatch.setattr(Q.time, "sleep", lambda s: None)
    monkeypatch.setattr(Q, "load_sessions", lambda: {})
    quotes = {"BTCUSD": [{"bid": 1.0, "ask": 2.0}, {"bid": 1.0, "ask": 2.0}, {"bid": 1.5, "ask": 2.0}], "ETHUSD": None}
    p = Q.Poller(["BTCUSD", "ETHUSD"], 1000.0, tmp_path, None, client_factory=lambda **k: FakeClient(quotes))
    assert p.connect()
    for _ in range(3):
        p.poll("BTCUSD")
    p.poll("ETHUSD")
    rows = p.quotes.rows
    assert [(r[2], r[3], r[4]) for r in rows] == [("BTCUSD", 1.0, 2.0), ("BTCUSD", 1.5, 2.0)]
    assert rows[1][6] == 2                                    # polls since the previous change
    assert p.state["ETHUSD"] == "unsub" and not p.due("ETHUSD", Q.time.time())
    assert p.clock.written == 5 and (tmp_path / "ctrader_quotes_SAMPLED" / "README.txt").read_text().startswith("SAMPLED QUOTES, NOT TICKS")


def test_ctrader_poller_survives_the_app_closing(tmp_path, monkeypatch):
    Q = _load("xt_ctrader_quotes")
    clock = [1_000_000.0]
    monkeypatch.setattr(Q.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(Q, "load_sessions", lambda: {})
    calls = {"n": 0}

    def factory(**k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionRefusedError("closed")             # first reconnect fails, second works
        return FakeClient({"BTCUSD": [{"bid": 1.0, "ask": 2.0}]}, fail_after=2)

    monkeypatch.setattr(Q.time, "time", lambda: clock[0])
    p = Q.Poller(["BTCUSD"], 1000.0, tmp_path, clock[0] + 60, client_factory=factory)
    orig = p.poll

    def poll(sym):
        clock[0] += 0.5
        orig(sym)

    p.poll = poll
    p.run()
    assert calls["n"] >= 3 and p.quotes.written >= 1


# ------------------------------------------------------------------------------------------------ study helpers


def test_ob200_top_tracks_the_best_levels_through_deltas(tmp_path):
    S = _load("xt_study")
    d0 = ms("2026-11-20T00:00:00")
    msgs = [
        {"topic": "orderbook.200.X", "type": "snapshot", "ts": d0 + 1, "cts": d0, "data": {"s": "X", "b": [["10", "1"], ["9", "2"]], "a": [["11", "1"], ["12", "5"]], "u": 1}},
        {"topic": "orderbook.200.X", "type": "delta", "ts": d0 + 2, "cts": d0 + 1, "data": {"s": "X", "b": [["10", "0"]], "a": [], "u": 2}},
        {"topic": "orderbook.200.X", "type": "delta", "ts": d0 + 3, "cts": d0 + 2, "data": {"s": "X", "b": [["5", "1"]], "a": [], "u": 3}},
        {"topic": "orderbook.200.X", "type": "delta", "ts": d0 + 4, "cts": d0 + 3, "data": {"s": "X", "b": [], "a": [["10.5", "2"]], "u": 5}},
    ]
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("x.data", "\n".join(json.dumps(m) for m in msgs) + "\n")
    df, qa = S.ob200_top(z, "2026-11-20")
    assert df.select("bid", "ask").rows() == [(10.0, 11.0), (9.0, 11.0), (9.0, 10.5)]     # delta 3 did not change the top
    assert qa["u_gaps"] == 1 and qa["crossed"] == 0


def test_hayashi_yoshida_finds_the_lead():
    rng = np.random.default_rng(1)
    t = np.sort(rng.choice(np.arange(0, 600_000, 10), 3000, replace=False))
    p = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(t))))
    tb = np.sort(rng.choice(np.arange(0, 600_000, 10), 2500, replace=False))
    idx = np.searchsorted(t, tb - 500, side="right") - 1
    keep = idx >= 0
    tb, pb = tb[keep], p[idx[keep]]                             # B = A delayed by 500 ms, previous-tick at B's own times
    lags = list(range(-2000, 2001, 100))
    r = X.hy_xcorr(t, p, tb, pb, lags)
    assert lags[int(np.nanargmax(r))] == 500 and r[lags.index(500)] > 0.5


def test_gap_and_episodes():
    assert np.allclose(X.gap_bp([101.0], [100.0]), [1e4 * np.log(1.01)])
    m = np.array([False, True, True, False, True])
    t = np.array([0, 1000, 2000, 3000, 4000])
    assert X.episodes(m, t) == [(1000, 3000), (4000, 4000)]
