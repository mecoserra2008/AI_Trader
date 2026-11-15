"""Read-only client for the local cTrader Desktop MCP server (http://127.0.0.1:9876/mcp/, added 2026-11-15).

The server also exposes order, position, account and app-control tools. This client can only call the market-data tools
in READ_ONLY_TOOLS: anything else raises before a request is sent. Account tools (balances) are excluded too, so nothing
here can leak account figures into the public repo. SPY symbols are refused (the user's own position; config/xvenue.yaml).
Use for data only: cross-venue gaps (H-ARB1/2), Friday-close references for Bybit TradFi perps, vecopen G2.

    uv run python -m ai_trader.ctrader.mcp_ro spot NAS100
    uv run python -m ai_trader.ctrader.mcp_ro bars NAS100 h1 2026-11-13T00:00:00Z 2026-11-15T21:00:00Z
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any

URL = "http://127.0.0.1:9876/mcp/"
PROTOCOL = "2025-11-25"
READ_ONLY_TOOLS = frozenset({
    "get_symbols", "get_symbol_details", "get_spot_prices", "get_symbol_sessions", "get_trendbars", "get_news",
    "get_server_time",
})
TIMEFRAMES = ("m1", "m5", "m15", "m30", "h1", "h4", "d1")


class CTraderRO:
    def __init__(self, url: str = URL, timeout: float = 30.0) -> None:
        self.url, self.timeout, self.sid, self._id = url, timeout, None, 0
        self._post({"jsonrpc": "2.0", "id": self._next(), "method": "initialize",
                    "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                               "clientInfo": {"name": "ai_trader-readonly", "version": "1"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _next(self) -> int:
        self._id += 1
        return self._id

    def _post(self, body: dict) -> dict | None:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": PROTOCOL}
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(self.url, json.dumps(body).encode(), h)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self.sid = r.headers.get("Mcp-Session-Id") or self.sid
            raw = r.read().decode()
        if not raw.strip():
            return None
        if raw.lstrip().startswith("event:") or raw.lstrip().startswith("data:"):
            raw = next(ln[5:] for ln in raw.splitlines() if ln.startswith("data:"))
        return json.loads(raw)

    def call(self, tool: str, **args: Any) -> Any:
        if tool not in READ_ONLY_TOOLS:
            raise PermissionError(f"{tool} is not a read-only market-data tool; this client never calls it")
        sym = str(args.get("symbolName", "")).upper()
        if sym.startswith("SPY"):
            raise PermissionError("SPY symbols are excluded (the user's own position)")
        resp = self._post({"jsonrpc": "2.0", "id": self._next(), "method": "tools/call",
                           "params": {"name": tool, "arguments": args}})
        if resp is None or "error" in resp:
            raise RuntimeError(f"{tool}: {resp and resp.get('error')}")
        res = resp["result"]
        text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if res.get("isError"):
            raise RuntimeError(f"{tool}: {text[:300]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def spot(self, symbol: str) -> dict:
        return self.call("get_spot_prices", symbolName=symbol)

    def bars(self, symbol: str, timeframe: str, start: str, end: str, limit: int = 1000) -> list[dict]:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
        d = self.call("get_trendbars", symbolName=symbol, timeframe=timeframe, **{"from": start, "to": end}, limit=limit)
        return d.get("bars", []) if isinstance(d, dict) else []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Read-only cTrader market data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spot"); s.add_argument("symbol")
    f = sub.add_parser("symbols"); f.add_argument("filter")
    b = sub.add_parser("bars"); b.add_argument("symbol"); b.add_argument("tf"); b.add_argument("start"); b.add_argument("end")
    a = ap.parse_args(argv)
    c = CTraderRO()
    if a.cmd == "spot":
        out = c.spot(a.symbol)
    elif a.cmd == "symbols":
        out = c.call("get_symbols", filter=a.filter)
    else:
        out = c.bars(a.symbol, a.tf, a.start, a.end)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
