"""The dashboard's HTTP server: 127.0.0.1 only, GET only, read only.

Routes
  /                      research/live/dashboard/index.html (plus app.js, pct.js, xs.js, style.css; nothing else is served)
  /api/state             Store.state(): latest v1 (and v2) line, 72-line history, realised outcomes, Claude, H-LIVE, alarms,
                         regime and players
  /api/activations?id=   the PNN forward pass of that line, per layer and edge (activations.activations), cached per id,
                         plus the 'what it's thinking' sentences (recomputed per request: a v2 line may arrive later)
  /api/edges?h=1h|24h    the static edge catalogue of a net (exact forms, cubic fits, sampled curves)
  /api/kline?hours=36    proxy to Bybit's public REST /v5/market/kline (linear BTCUSDT, 1m), paged, cached 15 s
  /api/xs                the newest cross-sectional v3 line (data/journal/live_xs_v3; context only, not a signal) and its
                         recent scores
  /api/health            liveness
Single-threaded (one request at a time) and torch on 1 thread, so the process stays within 2 compute threads.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8765
STATIC = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
          "pct.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8", "xs.js": "text/javascript; charset=utf-8"}
BYBIT_KLINE = "https://api.bybit.com/v5/market/kline"
KLINE_TTL = 15.0
MAX_HOURS = 48
ACT_CACHE = 64


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def bybit_fetch(params: dict, timeout: float = 8.0) -> dict:
    url = BYBIT_KLINE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-trader-livedash/1 (read-only)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def klines(hours: float, fetch=bybit_fetch, now_ms: int | None = None) -> list[list]:
    """1m BTCUSDT linear klines for the last `hours` (<= 48), ascending: [start_ms, open, high, low, close, volume]."""
    hours = max(1.0, min(float(hours), MAX_HOURS))
    now_ms = now_ms or int(time.time() * 1000)
    start = now_ms - int(hours * 3600_000)
    rows: dict[int, list] = {}
    end = now_ms
    for _ in range(4):
        d = fetch({"category": "linear", "symbol": "BTCUSDT", "interval": "1", "limit": 1000, "end": end})
        if d.get("retCode") not in (0, None):
            raise RuntimeError(f"bybit retCode {d.get('retCode')}: {d.get('retMsg')}")
        lst = d.get("result", {}).get("list", [])
        if not lst:
            break
        for k in lst:
            t = int(k[0])
            if t >= start:
                rows[t] = [t, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
        oldest = min(int(k[0]) for k in lst)
        if oldest <= start:
            break
        end = oldest - 1
    return [rows[t] for t in sorted(rows)]


class App:
    """State shared by the handler: the store, the activation cache and the kline cache."""

    def __init__(self, root: Path | str, models_root: Path | str | None = None, static_dir: Path | str | None = None, fetch=bybit_fetch):
        from ai_trader.livedash.store import Store

        self.store = Store(root, models_root)
        self.static = Path(static_dir) if static_dir else Path(root) / "research" / "live" / "dashboard"
        self.fetch = fetch
        self._act: OrderedDict[str, dict] = OrderedDict()
        self._kl: tuple[float, float, list] | None = None

    def state(self) -> dict:
        return self.store.state()

    def xs(self) -> dict:
        return self.store.xs()

    def activations(self, line_id: str) -> dict:
        from ai_trader.livedash import activations as A
        from ai_trader.livedash import thinking as T

        line = self.store.find(line_id)
        if line is None:
            raise KeyError(line_id)
        if line_id not in self._act:
            self._act[line_id] = A.activations(line, self.store.pnn_dir, self.store.week_lines(line))
            while len(self._act) > ACT_CACHE:
                self._act.popitem(last=False)
        act = self._act[line_id]
        v1 = line if line["id"].startswith("nn-") else (self.store.find("nn-" + line["id"][3:]) or line)
        hist = [r for r in self.store.first_per_hour(self.store.v1()) if r["forecast_hour"] < v1["forecast_hour"]]
        v2 = next((r for r in reversed(self.store.v2()) if r["forecast_hour"] == v1["forecast_hour"]), None)
        alarms = [{"source": "v1", "text": a} for a in v1.get("alarms", [])] + ([{"source": "v2", "text": a} for a in v2.get("alarms", [])] if v2 else [])
        return {**act, "thinking": T.thinking(act, v1, hist[-1] if hist else None, v2, alarms)}

    def edges(self, h: str) -> dict:
        from ai_trader.livedash import activations as A

        if h not in ("1h", "24h"):
            raise KeyError(h)
        return A.edge_catalogue(h, str(self.store.pnn_dir))

    def kline(self, hours: float) -> dict:
        now = time.time()
        if self._kl and now - self._kl[0] < KLINE_TTL and self._kl[1] >= hours:
            rows = self._kl[2]
        else:
            rows = klines(hours, self.fetch)
            self._kl = (now, hours, rows)
        cut = int((now - hours * 3600) * 1000)
        return {"symbol": "BTCUSDT", "interval": "1", "source": "bybit public /v5/market/kline (linear)", "rows": [r for r in rows if r[0] >= cut]}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ai-trader-livedash/1"

        def log_message(self, fmt, *args):  # quiet
            return

        def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(_clean(obj), default=_json_default, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            path = u.path.lstrip("/") or "index.html"
            try:
                if (_b := __import__("ai_trader.livedash.brain", fromlist=["route"])).handles(path):  # brain page (livedash/brain.py)
                    return _b.route(app, self, path, q)
                if path in STATIC:
                    f = app.static / path
                    if not f.is_file():
                        return self._json(404, {"error": f"{path} not found"})
                    return self._send(200, f.read_bytes(), STATIC[path])
                if path == "api/health":
                    return self._json(200, {"ok": True})
                if path == "api/state":
                    return self._json(200, app.state())
                if path == "api/xs":
                    return self._json(200, app.xs())
                if path == "api/activations":
                    lid = (q.get("id") or [""])[0]
                    if not lid:
                        st = app.store.v1()
                        lid = st[-1]["id"] if st else ""
                    try:
                        return self._json(200, app.activations(lid))
                    except KeyError:
                        return self._json(404, {"error": f"no line {lid!r}"})
                if path == "api/edges":
                    try:
                        return self._json(200, app.edges((q.get("h") or ["1h"])[0]))
                    except KeyError:
                        return self._json(404, {"error": "h must be 1h or 24h"})
                if path == "api/kline":
                    try:
                        hours = float((q.get("hours") or ["36"])[0])
                    except ValueError:
                        hours = 36.0
                    try:
                        return self._json(200, app.kline(max(1.0, min(hours, MAX_HOURS))))
                    except Exception as e:  # noqa: BLE001 - upstream down: the page falls back to the websocket
                        return self._json(502, {"error": f"{type(e).__name__}: {e}"[:200], "rows": []})
                return self._json(404, {"error": "not found"})
            except BrokenPipeError:
                return None
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": f"{type(e).__name__}: {e}"[:300]})

        def _refuse(self):
            try:  # drain a (small) body first: closing with unread data resets the socket on Windows
                n = min(int(self.headers.get("Content-Length") or 0), 1 << 16)
                if n:
                    self.rfile.read(n)
            except (ValueError, OSError):
                pass
            self._json(405, {"error": "read-only server: GET only"})

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse

    return Handler


def make_server(app: App, host: str = HOST, port: int = PORT) -> HTTPServer:
    """Bound to 127.0.0.1 only; any other host is refused."""
    if host != HOST:
        raise ValueError(f"the dashboard binds to {HOST} only (got {host!r})")
    return HTTPServer((HOST, int(port)), make_handler(app))


__all__ = ["HOST", "PORT", "App", "klines", "make_handler", "make_server"]
