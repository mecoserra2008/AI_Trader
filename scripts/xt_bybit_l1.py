"""XT recorder A: Bybit public linear websocket, top of book (orderbook.1) and trades (publicTrade). Data only, no keys.

    .venv/Scripts/python.exe scripts/xt_bybit_l1.py [--symbols BTCUSDT ETHUSDT ...] [--until 2026-12-12T00:00:00Z]

One process, one thread (polars writer pool capped at 1). Writes compact hourly parquet chunks:
    data/research/xt/live/bybit_book/date=YYYY-MM-DD/HH.parquet    one row per top-of-book change (or re-sent snapshot)
        recv_ts (local ms), ts (Bybit publication ms), cts (matching-engine ms), symbol, bid, bid_sz, ask, ask_sz, u,
        kind (s = snapshot, d = delta, r = snapshot re-sent with the same u: keep-alive, old cts)
    data/research/xt/live/bybit_trades/date=.../HH.parquet         one row per trade
        recv_ts, ts (message ms), T (trade ms), symbol, side (B/S taker), price, size, block, rpi
        (--trades-skip SYM ...: no live trade rows for those symbols; the public archive has them with exchange times)
Rows are flushed to part files every 20k rows or 60 s and compacted when the hour rolls (crash-safe). Heartbeat line
every 60 s in logs/xt_bybit_l1.log (rates, reconnects, lag recv_ts - cts). One helper thread samples the Bybit server
clock (public REST /v5/market/time) at start and hourly -> data/research/xt/live/bybit_clock/ (offset = server - local). App-level ping every 20 s; a socket silent for
30 s is reconnected with exponential backoff (1 s .. 60 s). Ctrl-C / SIGTERM / SIGBREAK / --until: clean shutdown.
SPY symbols are refused.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.research import xt as X  # noqa: E402

LOG = ROOT / "logs" / "xt_bybit_l1.log"
STOP = {"flag": False}
TIME_URL = "https://api.bybit.com/v5/market/time"
CLOCK_SCHEMA = {"recv_ts": "Int64", "send_ts": "Int64", "server_ms": "Int64", "offset_ms": "Float64", "rtt_ms": "Float32"}


def clock_samples(n: int = 5) -> list[tuple]:
    """Bybit server time vs the local clock (public REST /v5/market/time): offset = server - midpoint of send / receive."""
    out = []
    for _ in range(n):
        try:
            s = time.time() * 1000
            with urllib.request.urlopen(urllib.request.Request(TIME_URL, headers={"User-Agent": "AI_Trader-research"}), timeout=10) as r:
                d = json.loads(r.read())
            e = time.time() * 1000
            srv = int(d["result"]["timeNano"]) / 1e6
            out.append((int(e), int(s), int(srv), srv - 0.5 * (s + e), e - s))
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.5)
    return out


class ClockThread(threading.Thread):
    """The one helper thread: Bybit clock samples at start and hourly, handed to the main thread through a list."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="xt-clock")
        self.rows: list[tuple] = []
        self.lock = threading.Lock()

    def run(self) -> None:
        while not STOP["flag"]:
            rows = clock_samples()
            with self.lock:
                self.rows += rows
            t_end = time.time() + 3600
            while time.time() < t_end and not STOP["flag"]:
                time.sleep(1.0)

    def drain(self) -> list[tuple]:
        with self.lock:
            rows, self.rows = self.rows, []
        return rows


def log(msg: str) -> None:
    line = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def parse_until(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC).timestamp()


def _stop(*_: object) -> None:
    STOP["flag"] = True


class Recorder:
    def __init__(self, symbols: list[str], out: Path, until: float | None, trades_skip: frozenset[str] = frozenset()) -> None:
        self.symbols, self.until, self.trades_skip = symbols, until, trades_skip
        self.book = X.ChunkSink(out, "bybit_book", X.pl_schema(X.BOOK_SCHEMA))
        self.trades = X.ChunkSink(out, "bybit_trades", X.pl_schema(X.TRADE_SCHEMA))
        self.clock_sink = X.ChunkSink(out, "bybit_clock", X.pl_schema(CLOCK_SCHEMA), flush_rows=100)
        self.clock = ClockThread()
        self.l1 = X.L1State()
        self.n_book: dict[str, int] = defaultdict(int)
        self.n_trade: dict[str, int] = defaultdict(int)
        self.lag: list[int] = []
        self.reconnects = 0
        self.last_hb = time.time()

    def subscribe(self, ws: X.WebSocket) -> None:
        args = [f"orderbook.1.{s}" for s in self.symbols] + [f"publicTrade.{s}" for s in self.symbols]
        for i in range(0, len(args), 10):
            ws.send_text(json.dumps({"op": "subscribe", "req_id": f"xt{i}", "args": args[i:i + 10]}))

    def handle(self, raw: str) -> None:
        recv = int(time.time() * 1000)
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        topic = msg.get("topic", "")
        if topic.startswith("orderbook."):
            row = self.l1.apply(recv, msg)
            if row is not None:
                self.book.add(row)
                self.n_book[row[3]] += 1
                if row[-1] != "r" and len(self.lag) < 20000:
                    self.lag.append(recv - row[2])
        elif topic.startswith("publicTrade."):
            for row in X.parse_trades(recv, msg):
                self.n_trade[row[3]] += 1
                if row[3] not in self.trades_skip:
                    self.trades.add(row)
        elif msg.get("op") == "subscribe" and not msg.get("success", True):
            log(f"subscribe refused: {msg.get('ret_msg')}")

    def heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_hb < 60:
            return
        dt = now - self.last_hb
        rates = " ".join(f"{s}:{self.n_book[s] / dt:.1f}/{self.n_trade[s] / dt:.1f}" for s in self.symbols)
        lag = f"lag recv-cts p50 {statistics.median(self.lag):.0f} ms" if self.lag else "lag n/a"
        log(f"hb book/trade per s {rates} | {lag} | reconnects {self.reconnects} | rows {self.book.written}+{self.trades.written}")
        self.n_book.clear()
        self.n_trade.clear()
        self.lag = []
        self.last_hb = now
        clk = self.clock.drain()
        for row in clk:
            self.clock_sink.add(row)
        if clk:
            best = min(clk, key=lambda r: r[4])
            log(f"bybit clock offset (server - local) {best[3]:+.0f} ms at rtt {best[4]:.0f} ms (best of {len(clk)})")
            self.clock_sink.flush()
        self.book.tick()
        self.trades.tick()

    def run(self) -> None:
        backoff = 1.0
        self.clock.start()
        while not STOP["flag"] and (self.until is None or time.time() < self.until):
            ws = None
            try:
                ws = X.WebSocket(X.BYBIT_WS, timeout=10, read_timeout=1.0)
                self.subscribe(ws)
                log(f"connected; {len(self.symbols)} symbols: {' '.join(self.symbols)}")
                self.l1 = X.L1State()          # a new subscription starts with snapshots
                last_msg, last_ping = time.time(), time.time()
                while not STOP["flag"] and (self.until is None or time.time() < self.until):
                    raw = ws.recv()
                    now = time.time()
                    if raw is not None:
                        last_msg = now
                        backoff = 1.0
                        self.handle(raw)
                    if now - last_ping >= 20:
                        ws.send_text('{"op":"ping"}')
                        last_ping = now
                    if now - last_msg > 30:
                        raise X.WSClosed("silent for 30 s")
                    self.heartbeat()
            except (OSError, X.WSClosed) as e:
                self.reconnects += 1
                log(f"disconnected ({type(e).__name__}: {str(e)[:160]}); reconnect in {backoff:.0f} s")
                t_end = time.time() + backoff
                while time.time() < t_end and not STOP["flag"]:
                    time.sleep(0.2)
                backoff = min(60.0, backoff * 2)
            finally:
                if ws is not None:
                    ws.close()
        self.heartbeat(force=True)
        self.book.close()
        self.trades.close()
        self.clock_sink.close()
        log(f"stopped cleanly; rows written book {self.book.written} trades {self.trades.written}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="XT Bybit top-of-book + trades recorder (public websocket)")
    ap.add_argument("--symbols", nargs="+", default=X.bybit_symbols())
    ap.add_argument("--until", default=None, help="UTC end time, e.g. 2026-12-12T00:00:00Z")
    ap.add_argument("--out", default=str(X.XT_DIR / "live"))
    ap.add_argument("--trades-skip", nargs="*", default=[], help="symbols whose trades are counted but not stored")
    a = ap.parse_args(argv)
    syms = X.refuse_spy([s.upper() for s in a.symbols])
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _stop)
    log(f"start pid {os.getpid()} until {a.until or 'forever'}")
    Recorder(syms, Path(a.out), parse_until(a.until), frozenset(s.upper() for s in a.trades_skip)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
