"""XT: cross-venue tick study, Bybit linear perps vs BlackBull CFDs on cTrader (agent XT, 2026-11-28). Data only.

Nothing here places, amends or cancels orders. cTrader is read only through ai_trader.ctrader.mcp_ro (its READ_ONLY_TOOLS);
Bybit only through public endpoints. SPY symbols are refused on both venues (the user's standing rule, config/xvenue.yaml).

Pieces shared by the two recorders (scripts/xt_bybit_l1.py, scripts/xt_ctrader_quotes.py) and the study (scripts/xt_study.py):
    PAIRS            the instrument map (cTrader names resolved 2026-11-28 with get_symbols / get_symbol_details)
    session_open     is a cTrader symbol inside its weekly session at a UTC instant (get_symbol_sessions format)
    ChunkSink        buffered rows -> part parquet files -> one compact parquet per UTC hour (crash-safe)
    WebSocket        a minimal RFC 6455 client on the standard library (no package in the main env has one)
    parse_bybit      orderbook.1 / publicTrade messages -> rows
    gap_bp, hy_xcorr gap = 1e4 * ln(mid_bybit / mid_ctrader); Hayashi-Yoshida cross-correlation of asynchronous returns

Units: every timestamp is epoch ms UTC (int64). recv_ts is the local clock of this machine; exchange times are kept as given.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import socket
import ssl
import struct
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
XT_DIR = ROOT / "data" / "research" / "xt"
RAW_DIR = ROOT / "data" / "raw" / "XT"
MAP_PATH = XT_DIR / "ctrader_map.json"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"

# Instrument map. role: "arb" = same underlying (a gap can be traded back), "proxy" = related underlying with a drifting
# ratio (lead-lag only), "solo" = no Bybit leg. usdt: the Bybit price is in USDT (crypto and token perps), so the gap
# vs a USD CFD includes the USDT/USD basis (config/xvenue.yaml: subtract ln(USDCUSDT)).
PAIRS: list[dict[str, Any]] = [
    {"key": "gold", "class": "metal", "ctrader": ["XAUUSD"], "bybit": ["XAUUSDT", "XAUTUSDT"], "role": "arb",
     "note": "XAUTUSDT is Tether Gold (a token, USDT-priced, own premium vs spot gold): second leg, usdt-corrected"},
    {"key": "silver", "class": "metal", "ctrader": ["XAGUSD"], "bybit": ["XAGUSDT"], "role": "arb", "note": ""},
    {"key": "copper", "class": "metal", "ctrader": ["Copper"], "bybit": [], "role": "solo",
     "note": "Copper (Cash) CFD; Bybit lists no copper perp: cTrader-only series"},
    {"key": "nasdaq", "class": "index", "ctrader": ["QQQ", "QQQ-24", "NAS100", "USTEC.F"], "bybit": ["QQQUSDT"], "role": "arb",
     "note": "QQQ / QQQ-24 (24 h equities) are the same ETF as QQQUSDT (arb); NAS100 cash and USTEC.F futures are the index "
             "(proxy: ratio ~41 drifts with dividends and fees)"},
    {"key": "brent", "class": "energy", "ctrader": ["BRENT", "UKBRENT.F"], "bybit": ["BZUSDT"], "role": "arb",
     "note": "BRENT = cash CFD (rolled), UKBRENT.F = futures CFD; BZUSDT tracks the front-month future: roll basis"},
    {"key": "wti", "class": "energy", "ctrader": ["WTI", "USOIL.F"], "bybit": ["CLUSDT"], "role": "arb",
     "note": "WTI = cash CFD (rolled), USOIL.F = futures CFD; CLUSDT tracks the front-month future: roll basis"},
    {"key": "sp500", "class": "index", "ctrader": ["US500", "SPX500", "US500.F"], "bybit": ["SPXLUSDT"], "role": "proxy",
     "note": "no S&P arbitrage leg (SPYUSDT excluded by the user's rule); SPXLUSDT is a 3x daily-reset ETF: lead-lag only"},
    {"key": "btc", "class": "crypto", "ctrader": ["BTCUSD"], "bybit": ["BTCUSDT"], "role": "arb", "note": "perp premium + USDT basis"},
    {"key": "eth", "class": "crypto", "ctrader": ["ETHUSD"], "bybit": ["ETHUSDT"], "role": "arb", "note": "perp premium + USDT basis"},
]
USDT_PRICED = frozenset({"BTCUSDT", "ETHUSDT", "XAUTUSDT"})
# Pairs compared in the gap / crossing hypotheses (bybit, ctrader). Proxy legs are lead-lag only.
ARB_LEGS: list[tuple[str, str]] = [("XAUUSDT", "XAUUSD"), ("XAUTUSDT", "XAUUSD"), ("XAGUSDT", "XAGUSD"), ("QQQUSDT", "QQQ"),
                                   ("QQQUSDT", "QQQ-24"), ("BZUSDT", "BRENT"), ("BZUSDT", "UKBRENT.F"), ("CLUSDT", "WTI"),
                                   ("CLUSDT", "USOIL.F"), ("BTCUSDT", "BTCUSD"), ("ETHUSDT", "ETHUSD")]
PROXY_LEGS: list[tuple[str, str]] = [("QQQUSDT", "NAS100"), ("QQQUSDT", "USTEC.F"), ("SPXLUSDT", "US500"), ("SPXLUSDT", "SPX500"),
                                     ("SPXLUSDT", "US500.F")]


def bybit_symbols() -> list[str]:
    return sorted({s for p in PAIRS for s in p["bybit"]})


def ctrader_symbols() -> list[str]:
    return [s for p in PAIRS for s in p["ctrader"]]


def refuse_spy(symbols: Iterable[str]) -> list[str]:
    out = [s.strip() for s in symbols if s.strip()]
    bad = [s for s in out if s.upper().startswith("SPY")]
    if bad:
        raise PermissionError(f"SPY symbols are excluded (the user's own position): {bad}")
    return out


# ---------------------------------------------------------------------------------------------------------- sessions

WINDOWS_TZ = {"Eastern Standard Time": "America/New_York", "UTC": "UTC", "GMT Standard Time": "Europe/London",
              "Central Standard Time": "America/Chicago", "E. Europe Standard Time": "Europe/Chisinau",
              "FLE Standard Time": "Europe/Kiev", "W. Europe Standard Time": "Europe/Berlin"}


def tz_of(name: str) -> ZoneInfo:
    return ZoneInfo(WINDOWS_TZ.get(name, name))


def session_open(ts_ms: int, sessions: dict[str, Any]) -> bool:
    """True when ts_ms (UTC) falls inside one of the symbol's weekly sessions. `sessions` is the get_symbol_sessions
    payload: timeZone (Windows name), marketIsAlwaysOpen, sessions[{startSecond, endSecond}] counted from Sunday 00:00
    local time (endSecond inclusive). Holidays are not in this payload: a closed holiday shows as open with a stale quote."""
    if sessions.get("marketIsAlwaysOpen"):
        return True
    loc = datetime.fromtimestamp(ts_ms / 1000, UTC).astimezone(tz_of(sessions.get("timeZone", "UTC")))
    sec = ((loc.weekday() + 1) % 7) * 86400 + loc.hour * 3600 + loc.minute * 60 + loc.second   # Sunday = 0
    for s in sessions.get("sessions", []):
        a, b = int(s["startSecond"]), int(s["endSecond"])
        if a <= sec <= b or (b >= 604800 and sec + 604800 <= b):
            return True
    return False


# ---------------------------------------------------------------------------------------------------------- sink


def hour_key(ts_ms: int) -> tuple[str, str]:
    d = datetime.fromtimestamp(ts_ms / 1000, UTC)
    return d.strftime("%Y-%m-%d"), d.strftime("%H")


class ChunkSink:
    """Rows -> part files (every `flush_rows` rows or `flush_s` seconds; small buffers keep the recorders near 60-100 MB) -> one `<root>/<stream>/date=D/HH.parquet` per UTC
    hour of the bucketing column (recv_ts). Parts are compacted when the hour rolls, at close, and at start-up for any
    hour left behind by a crash (a part file is always a complete parquet file, so nothing is lost on a hard kill)."""

    def __init__(self, root: Path, stream: str, schema: dict[str, Any], bucket_col: str = "recv_ts",
                 flush_rows: int = 20_000, flush_s: float = 60.0, sort_by: Sequence[str] | None = None) -> None:
        import polars as pl

        self.pl = pl
        self.dir = Path(root) / stream
        self.schema = schema
        self.cols = list(schema)
        self.bi = self.cols.index(bucket_col)
        # hourly files sorted by (symbol, time) compress ~25% better than arrival order (measured on the 11-28 smoke test)
        self.sort_by = list(sort_by) if sort_by else (["symbol", bucket_col] if "symbol" in schema else [bucket_col])
        self.flush_rows, self.flush_s = flush_rows, flush_s
        self.rows: list[tuple] = []
        self.key: tuple[str, str] | None = None
        self.last_flush = time.time()
        self.n_part = 0
        self.tag = f"{os.getpid()}_{int(time.time())}"
        self.written = 0
        self.recover()

    def _hour_dir(self, key: tuple[str, str]) -> Path:
        return self.dir / f"date={key[0]}"

    def add(self, row: tuple) -> None:
        k = hour_key(int(row[self.bi]))
        if self.key is not None and k != self.key:
            old = self.key
            self.flush()
            self.compact(old)
        self.key = k
        self.rows.append(row)
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def tick(self) -> None:
        if self.rows and time.time() - self.last_flush >= self.flush_s:
            self.flush()

    def flush(self) -> Path | None:
        self.last_flush = time.time()
        if not self.rows or self.key is None:
            return None
        d = self._hour_dir(self.key)
        d.mkdir(parents=True, exist_ok=True)
        self.n_part += 1
        p = d / f"{self.key[1]}.part{self.tag}_{self.n_part:04d}.parquet"
        df = self.pl.DataFrame(self.rows, schema=self.schema, orient="row")
        tmp = p.with_suffix(".tmp")
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(p)
        self.written += len(self.rows)
        self.rows = []
        return p

    def compact(self, key: tuple[str, str]) -> Path | None:
        d = self._hour_dir(key)
        parts = sorted(d.glob(f"{key[1]}.part*.parquet"))
        if not parts:
            return None
        out = d / f"{key[1]}.parquet"
        srcs = ([out] if out.exists() else []) + parts
        df = self.pl.concat([self.pl.read_parquet(s) for s in srcs], how="vertical_relaxed").sort(self.sort_by, maintain_order=True)
        tmp = d / f"{key[1]}.compact.tmp"
        df.write_parquet(tmp, compression="zstd", compression_level=9, statistics=True)
        tmp.replace(out)
        for s in parts:
            s.unlink(missing_ok=True)
        return out

    def recover(self) -> None:
        """Compact part files of past hours left by an earlier (crashed) run; the current hour is left to this run."""
        if not self.dir.exists():
            return
        now = hour_key(int(time.time() * 1000))
        for p in sorted(self.dir.glob("date=*/*.part*.parquet")):
            key = (p.parent.name[5:], p.name[:2])
            if key != now:
                self.compact(key)

    def close(self) -> None:
        k = self.key
        self.flush()
        if k is not None:
            self.compact(k)


# ---------------------------------------------------------------------------------------------------------- websocket

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WSClosed(ConnectionError):
    pass


class WebSocket:
    """Minimal RFC 6455 client: TLS, text frames, fragmentation, ping -> pong, close. No extensions are requested, so
    frames are never compressed. recv() returns None on a read timeout and never consumes a partial frame."""

    def __init__(self, url: str, timeout: float = 10.0, read_timeout: float = 1.0) -> None:
        u = urlsplit(url)
        host, port = u.hostname or "", u.port or (443 if u.scheme == "wss" else 80)
        raw = socket.create_connection((host, port), timeout=timeout)
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if u.scheme == "wss" else raw
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nUser-Agent: AI_Trader-research\r\n\r\n")
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WSClosed("closed during handshake")
            resp += chunk
            if len(resp) > 65536:
                raise WSClosed("handshake too long")
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        if " 101 " not in lines[0] + " ":
            raise WSClosed(f"handshake refused: {lines[0]}")
        hdr = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip() for ln in lines[1:] if ":" in ln}
        want = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if hdr.get("sec-websocket-accept") != want:
            raise WSClosed("bad Sec-WebSocket-Accept")
        self.sock.settimeout(read_timeout)
        self.frag: list[bytes] = []
        self.frag_op = 0

    def _send(self, op: int, payload: bytes) -> None:
        n = len(payload)
        head = bytes([0x80 | op])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(head + mask + body)

    def send_text(self, text: str) -> None:
        self._send(0x1, text.encode())

    def _frame(self) -> tuple[int, bool, bytes] | None:
        """Parse one complete frame from the buffer, or None if incomplete."""
        b = self.buf
        if len(b) < 2:
            return None
        fin, op = bool(b[0] & 0x80), b[0] & 0x0F
        masked, n, i = bool(b[1] & 0x80), b[1] & 0x7F, 2
        if n == 126:
            if len(b) < 4:
                return None
            n, i = struct.unpack(">H", b[2:4])[0], 4
        elif n == 127:
            if len(b) < 10:
                return None
            n, i = struct.unpack(">Q", b[2:10])[0], 10
        mk = b""
        if masked:
            if len(b) < i + 4:
                return None
            mk, i = b[i:i + 4], i + 4
        if len(b) < i + n:
            return None
        payload = b[i:i + n]
        if masked:
            payload = bytes(x ^ mk[j % 4] for j, x in enumerate(payload))
        self.buf = b[i + n:]
        return op, fin, payload

    def recv(self) -> str | None:
        """Next complete text message, or None after one read timeout. Raises WSClosed on close / EOF."""
        while True:
            fr = self._frame()
            if fr is None:
                try:
                    chunk = self.sock.recv(1 << 16)
                except (TimeoutError, socket.timeout):
                    return None
                except ssl.SSLWantReadError:
                    return None
                if not chunk:
                    raise WSClosed("eof")
                self.buf += chunk
                continue
            op, fin, payload = fr
            if op == 0x9:
                self._send(0xA, payload)
                continue
            if op == 0xA:
                continue
            if op == 0x8:
                raise WSClosed(f"close frame {payload[:2].hex()}")
            if op in (0x1, 0x2):
                self.frag, self.frag_op = [payload], op
            elif op == 0x0:
                self.frag.append(payload)
            if fin:
                msg = b"".join(self.frag)
                self.frag = []
                return msg.decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self._send(0x8, struct.pack(">H", 1000))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------- bybit parsing

# Compact on purpose (storage: ~9 B per book row, ~6 B per trade row after sorting + zstd 9): no seq (u identifies the
# update), no trade id (a UUID string costs ~20 B per row; the public archive has every trade with its exchange time).
BOOK_SCHEMA = {"recv_ts": "Int64", "ts": "Int64", "cts": "Int64", "symbol": "String", "bid": "Float64", "bid_sz": "Float64",
               "ask": "Float64", "ask_sz": "Float64", "u": "Int64", "kind": "String"}
TRADE_SCHEMA = {"recv_ts": "Int64", "ts": "Int64", "T": "Int64", "symbol": "String", "side": "String", "price": "Float64",
                "size": "Float64", "block": "Boolean", "rpi": "Boolean"}


def pl_schema(s: dict[str, str]) -> dict[str, Any]:
    import polars as pl

    return {k: getattr(pl, v) for k, v in s.items()}


class L1State:
    """Top of book per symbol from orderbook.1 messages. On the linear feed every level-1 message is a snapshot (checked
    2026-11-28); a delta replaces a side's level and a size of 0 removes it. Returns a row when the top changed or a snapshot
    arrived; a snapshot re-sent with the previous update id u (Bybit's keep-alive after a quiet spell, its cts is old) is
    kept as kind 'r' so staleness can be measured, and must be left out of latency statistics."""

    def __init__(self) -> None:
        self.top: dict[str, list[float]] = {}
        self.last_u: dict[str, int] = {}

    def apply(self, recv_ts: int, msg: dict) -> tuple | None:
        d = msg.get("data") or {}
        sym = d.get("s") or msg.get("topic", "").split(".")[-1]
        kind = msg.get("type", "")
        cur = self.top.get(sym) or [math.nan] * 4
        new = [math.nan] * 4 if kind == "snapshot" else list(cur)
        for side, (ip, iz) in (("b", (0, 1)), ("a", (2, 3))):
            for p, s in d.get(side, ()):
                px, sz = float(p), float(s)
                if sz == 0.0:
                    if new[ip] == px:
                        new[ip], new[iz] = math.nan, math.nan
                else:
                    new[ip], new[iz] = px, sz
        changed = kind == "snapshot" or any(not (a == b or (a != a and b != b)) for a, b in zip(new, cur, strict=True))
        self.top[sym] = new
        if not changed:
            return None
        u = int(d.get("u", -1))
        k = "r" if kind == "snapshot" and self.last_u.get(sym) == u else kind[:1]
        self.last_u[sym] = u
        return (recv_ts, int(msg.get("ts", 0)), int(msg.get("cts", msg.get("ts", 0))), sym, new[0], new[1], new[2], new[3], u, k)


def parse_trades(recv_ts: int, msg: dict) -> list[tuple]:
    ts = int(msg.get("ts", 0))
    out = []
    for t in msg.get("data", ()):
        out.append((recv_ts, ts, int(t["T"]), t["s"], t["S"][:1], float(t["p"]), float(t["v"]), bool(t.get("BT", False)),
                    bool(t.get("RPI", False))))
    return out


# ---------------------------------------------------------------------------------------------------------- analysis


def gap_bp(mid_bybit: np.ndarray, mid_ctrader: np.ndarray) -> np.ndarray:
    """1e4 * ln(mid_bybit / mid_ctrader) (log ratio in bp, the house convention for spreads)."""
    return 1e4 * np.log(np.asarray(mid_bybit, float) / np.asarray(mid_ctrader, float))


def hy_xcorr(ta: np.ndarray, pa: np.ndarray, tb: np.ndarray, pb: np.ndarray, lags_ms: Sequence[int]) -> np.ndarray:
    """Hayashi-Yoshida correlation of log-price changes of two asynchronously observed series, B shifted by each lag:
    rho(l) = sum_{i,j} dA_i dB_j 1{(a_{i-1}, a_i] overlaps (b_{j-1}+l, b_j+l]} / sqrt(sum dA^2 sum dB^2).
    A positive lag l shifts B later: a peak at l > 0 means A moves first (A leads B by l ms). O(n + m) per lag."""
    ta, tb = np.asarray(ta, np.int64), np.asarray(tb, np.int64)
    ra, rb = np.diff(np.log(np.asarray(pa, float))), np.diff(np.log(np.asarray(pb, float)))
    a0, a1 = ta[:-1], ta[1:]
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    out = np.full(len(lags_ms), np.nan)
    if den == 0 or len(ra) == 0 or len(rb) == 0:
        return out
    cb = np.concatenate([[0.0], np.cumsum(rb)])
    for k, lag in enumerate(lags_ms):
        b0, b1 = tb[:-1] - lag, tb[1:] - lag          # B's interval j shifted by -lag (B later by lag = compare B_t+lag with A_t)
        # intervals of B overlapping (a0_i, a1_i]: j with b1_j > a0_i and b0_j < a1_i
        lo = np.searchsorted(b1, a0, side="right")
        hi = np.searchsorted(b0, a1, side="left")
        s = np.where(hi > lo, cb[np.maximum(hi, lo)] - cb[lo], 0.0)
        out[k] = float((ra * s).sum()) / den
    return out


def episodes(mask: np.ndarray, ts_ms: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a boolean series on timestamps -> [(start_ms, end_ms)], end = first False timestamp after the run
    (or the last timestamp)."""
    m = np.asarray(mask, bool)
    t = np.asarray(ts_ms, np.int64)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(t[s]), int(t[min(e, len(t) - 1)])) for s, e in zip(starts, ends, strict=True)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
