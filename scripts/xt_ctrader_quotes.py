"""XT recorder B: cTrader Desktop quotes, polled through the read-only MCP client. SAMPLED QUOTES, NOT TICKS.

    .venv/Scripts/python.exe scripts/xt_ctrader_quotes.py [--symbols BTCUSD ETHUSD ...] [--rate 10] [--until 2026-12-12T00:00:00Z]

Calls only get_spot_prices and get_server_time (ai_trader.ctrader.mcp_ro.READ_ONLY_TOOLS); nothing else on that server.
The MCP returns the app's latest quote with no timestamp, so each row carries the local send and receive times of the
request; the quote's true time lies between them. A tick that is replaced before the next poll of its symbol is never
seen: these series understate the number of quote changes and any crossing shorter than the per-symbol poll interval
(= n_active_symbols / rate). True ticks need the cBot (research/xt/TickRecorder.cs) or the cTrader Open API.

Requests are paced at --rate per second (global), round-robin over the symbols with a live quote; symbols outside their
session (data/research/xt/ctrader_map.json) are polled every 30 s and symbols without a live quote ("unsubscribed": not
open in any chart / watchlist of the app) are re-tried every 60 s, so they are picked up once the user opens them. If the
app is closed the recorder backs off (5 s .. 5 min), logs and keeps going.

Outputs (hourly parquet, same crash-safe chunking as the Bybit recorder):
    data/research/xt/live/ctrader_quotes_SAMPLED/date=D/HH.parquet  one row per CHANGED quote (bid or ask)
        recv_ts, send_ts (local ms), symbol, bid, ask, rtt_ms, polls_since_change
    data/research/xt/live/ctrader_clock/...       get_server_time samples at start, hourly and after reconnects
        recv_ts, send_ts, server_ms, offset_ms (= server - midpoint of send/recv), rtt_ms
    data/research/xt/live/ctrader_polls/...       one row per symbol per minute: polls, changes, errors, rtt p50 / p90
Heartbeat every 60 s in logs/xt_ctrader_quotes.log. SPY symbols are refused (also by mcp_ro).
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.ctrader.mcp_ro import CTraderRO  # noqa: E402
from ai_trader.research import xt as X  # noqa: E402

LOG = ROOT / "logs" / "xt_ctrader_quotes.log"
STOP = {"flag": False}
QUOTE_SCHEMA = {"recv_ts": "Int64", "send_ts": "Int64", "symbol": "String", "bid": "Float64", "ask": "Float64",
                "rtt_ms": "Float32", "polls_since_change": "Int32"}
CLOCK_SCHEMA = {"recv_ts": "Int64", "send_ts": "Int64", "server_ms": "Int64", "offset_ms": "Float64", "rtt_ms": "Float32"}
POLL_SCHEMA = {"recv_ts": "Int64", "symbol": "String", "n_polls": "Int32", "n_changes": "Int32", "n_errors": "Int32",
               "rtt_p50": "Float32", "rtt_p90": "Float32", "state": "String"}
README = ("SAMPLED QUOTES, NOT TICKS. Polled from the cTrader Desktop app through the read-only MCP (get_spot_prices). "
          "Rows only when bid or ask changed between two polls of the symbol. Changes that happen and revert between polls "
          "are invisible; crossings shorter than the per-symbol poll interval are understated. Times: local clock "
          "(send_ts / recv_ts); cTrader server offset in ../ctrader_clock.\n")
CLOSED_EVERY_S = 30.0
UNSUB_EVERY_S = 60.0


def log(msg: str) -> None:
    line = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _stop(*_: object) -> None:
    STOP["flag"] = True


def now_ms() -> int:
    return int(time.time() * 1000)


def load_sessions() -> dict[str, dict]:
    if not X.MAP_PATH.exists():
        return {}
    m = json.loads(X.MAP_PATH.read_text(encoding="utf-8"))
    return {s["name"]: s.get("sessions") or {} for s in m.get("symbols", [])}


class Poller:
    def __init__(self, symbols: list[str], rate: float, out: Path, until: float | None, client_factory=CTraderRO) -> None:
        self.symbols, self.rate, self.until = symbols, rate, until
        self.factory = client_factory
        self.client = None
        self.sessions = load_sessions()
        out.mkdir(parents=True, exist_ok=True)
        self.quotes = X.ChunkSink(out, "ctrader_quotes_SAMPLED", X.pl_schema(QUOTE_SCHEMA))
        (out / "ctrader_quotes_SAMPLED").mkdir(parents=True, exist_ok=True)
        (out / "ctrader_quotes_SAMPLED" / "README.txt").write_text(README, encoding="utf-8")
        self.clock = X.ChunkSink(out, "ctrader_clock", X.pl_schema(CLOCK_SCHEMA), flush_rows=100)
        self.polls = X.ChunkSink(out, "ctrader_polls", X.pl_schema(POLL_SCHEMA), flush_rows=2000)
        self.last: dict[str, tuple[float, float]] = {}
        self.since: dict[str, int] = defaultdict(int)
        self.state: dict[str, str] = {s: "live" for s in symbols}      # live | closed | unsub | error
        self.next_try: dict[str, float] = {s: 0.0 for s in symbols}
        self.stat: dict[str, dict] = defaultdict(lambda: {"n": 0, "chg": 0, "err": 0, "rtt": []})
        self.backoff = 5.0
        self.last_clock = 0.0
        self.last_min = int(time.time() // 60)
        self.last_hb = time.time()
        self.t_next = time.perf_counter()

    # ---- pacing and requests
    def pace(self) -> None:
        self.t_next += 1.0 / self.rate
        d = self.t_next - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            self.t_next = time.perf_counter()

    def connect(self) -> bool:
        try:
            self.client = self.factory(timeout=5.0)
            self.backoff = 5.0
            log("connected to the cTrader MCP")
            self.sample_clock()
            return True
        except OSError as e:
            self.client = None
            log(f"cTrader app not reachable ({type(e).__name__}: {str(e)[:120]}); retry in {self.backoff:.0f} s")
            t_end = time.time() + self.backoff
            while time.time() < t_end and not STOP["flag"]:
                time.sleep(0.5)
            self.backoff = min(300.0, self.backoff * 2)
            return False

    def sample_clock(self, n: int = 5) -> None:
        for _ in range(n):
            s = time.time() * 1000
            r = self.client.call("get_server_time")
            e = time.time() * 1000
            srv = int(r["unixMs"]) if isinstance(r, dict) and "unixMs" in r else None
            if srv is not None:
                self.clock.add((int(e), int(s), srv, srv - 0.5 * (s + e), e - s))
            self.pace()
        self.clock.flush()
        self.last_clock = time.time()

    def due(self, sym: str, t: float) -> bool:
        st = self.state[sym]
        if st == "live":
            return True
        return t >= self.next_try[sym]

    def poll(self, sym: str) -> None:
        s = time.time() * 1000
        t0 = time.perf_counter()
        st = self.stat[sym]
        try:
            q = self.client.call("get_spot_prices", symbolName=sym)
        except RuntimeError as e:
            msg = str(e)
            st["err"] += 1
            new = "unsub" if "No live quote" in msg else "error"
            if self.state[sym] != new:
                log(f"{sym}: {new} ({msg[:120]}); re-try every {UNSUB_EVERY_S:.0f} s")
            self.state[sym] = new
            self.next_try[sym] = time.time() + UNSUB_EVERY_S
            return
        rtt = (time.perf_counter() - t0) * 1000
        e = s + rtt
        st["n"] += 1
        if len(st["rtt"]) < 5000:
            st["rtt"].append(rtt)
        bid, ask = float(q.get("bid") or "nan"), float(q.get("ask") or "nan")
        self.since[sym] += 1
        prev = self.last.get(sym)
        if prev is None or prev != (bid, ask):
            self.quotes.add((int(e), int(s), sym, bid, ask, rtt, self.since[sym]))
            self.since[sym] = 0
            st["chg"] += 1
            self.last[sym] = (bid, ask)
        sess = self.sessions.get(sym)
        open_ = X.session_open(int(e), sess) if sess else True
        new = "live" if open_ else "closed"
        if self.state[sym] != new:
            log(f"{sym}: {new}")
        self.state[sym] = new
        if new == "closed":
            self.next_try[sym] = time.time() + CLOSED_EVERY_S

    # ---- bookkeeping
    def minute(self) -> None:
        m = int(time.time() // 60)
        if m == self.last_min:
            return
        t = m * 60_000
        for sym in self.symbols:
            st = self.stat[sym]
            if not (st["n"] or st["err"]):
                continue
            r = np.array(st["rtt"]) if st["rtt"] else None
            p50, p90 = (float(np.percentile(r, 50)), float(np.percentile(r, 90))) if r is not None else (float("nan"), float("nan"))
            self.polls.add((t, sym, st["n"], st["chg"], st["err"], p50, p90, self.state[sym]))
        if time.time() - self.last_hb >= 60:
            live = [s for s in self.symbols if self.state[s] == "live"]
            per = len(live) / self.rate if live else float("nan")
            parts = " ".join(f"{s}:{self.stat[s]['n']}/{self.stat[s]['chg']}" for s in self.symbols if self.stat[s]["n"] or self.stat[s]["err"])
            allr = [x for s in self.symbols for x in self.stat[s]["rtt"]]
            p50 = f"{np.percentile(allr, 50):.0f}/{np.percentile(allr, 90):.0f}" if allr else "n/a"
            log(f"hb polls/changes {parts} | live {len(live)} -> per-symbol interval {per:.2f} s | rtt p50/p90 {p50} ms | "
                f"unsub {[s for s in self.symbols if self.state[s] == 'unsub']}")
            self.last_hb = time.time()
        self.stat.clear()
        self.last_min = m
        self.quotes.tick()
        self.polls.tick()

    def run(self) -> None:
        i = 0
        while not STOP["flag"] and (self.until is None or time.time() < self.until):
            if self.client is None and not self.connect():
                continue
            try:
                if time.time() - self.last_clock >= 3600:
                    self.sample_clock()
                t = time.time()
                due = [s for s in self.symbols if self.due(s, t)]
                if not due:
                    time.sleep(0.2)
                    self.minute()
                    continue
                sym = due[i % len(due)]
                i += 1
                self.poll(sym)
                self.pace()
                self.minute()
            except (OSError, ValueError, KeyError) as e:      # app closed / restarted, session expired, broken reply
                log(f"request failed ({type(e).__name__}: {str(e)[:120]}); reconnecting")
                self.client = None
        self.quotes.close()
        self.clock.close()
        self.polls.close()
        log(f"stopped cleanly; quote rows {self.quotes.written}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="XT cTrader sampled-quote recorder (read-only MCP; SAMPLED QUOTES, NOT TICKS)")
    ap.add_argument("--symbols", nargs="+", default=X.ctrader_symbols())
    ap.add_argument("--rate", type=float, default=10.0, help="requests per second, all symbols together (measured: see report)")
    ap.add_argument("--until", default=None, help="UTC end time, e.g. 2026-12-12T00:00:00Z")
    ap.add_argument("--out", default=str(X.XT_DIR / "live"))
    a = ap.parse_args(argv)
    syms = X.refuse_spy(a.symbols)
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _stop)
    until = datetime.fromisoformat(a.until.replace("Z", "+00:00")).timestamp() if a.until else None
    log(f"start pid {os.getpid()} rate {a.rate}/s symbols {' '.join(syms)} until {a.until or 'forever'} (SAMPLED QUOTES, NOT TICKS)")
    Poller(syms, a.rate, Path(a.out), until).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
