"""Connectivity probe for the quoter: REST round-trip latency, clock offset, error rate, and (on demo only) order
place / cancel round trips. Mainnet calls are read-only. Keys come from .env and are never printed.

    PYTHONIOENCODING=utf-8 uv run python scripts/conn_probe.py [--n 30] [--demo-orders 15] [--symbol XRPUSDT]
"""
from __future__ import annotations

import argparse, hashlib, hmac, json, os, statistics as st, time
import http.client
from dotenv import load_dotenv

HOST = {"mainnet": "https://api.bybit.com", "demo": "https://api-demo.bybit.com"}


class Client:
    def __init__(self, env: str):
        self.env, self.h = env, HOST[env]
        self.key, self.sec = os.environ.get(f"BYBIT_{env.upper()}_API_KEY"), os.environ.get(f"BYBIT_{env.upper()}_API_SECRET")
        self.conn = None  # one persistent HTTPS connection (keep-alive), like the Rust client

    def call(self, method, path, params=None, signed=False):
        params = params or {}
        hd = {"Content-Type": "application/json"}
        if method == "GET":
            payload = "&".join(f"{k}={v}" for k, v in params.items())
            url, body = path + (f"?{payload}" if payload else ""), None
        else:
            payload = json.dumps(params); url, body = path, payload
        if signed:
            ts, rw = str(int(time.time() * 1000)), "5000"
            hd.update({"X-BAPI-API-KEY": self.key, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": rw,
                       "X-BAPI-SIGN": hmac.new(self.sec.encode(), (ts + self.key + rw + payload).encode(), hashlib.sha256).hexdigest()})
        t0 = time.perf_counter(); w0 = time.time()
        try:
            if self.conn is None:
                self.conn = http.client.HTTPSConnection(self.h.split("//")[1], timeout=10)
            self.conn.request(method, url, body=body, headers=hd)
            d = json.loads(self.conn.getresponse().read()); ok = d.get("retCode") == 0
        except Exception as e:  # noqa: BLE001
            d, ok, self.conn = {"err": str(e)[:120]}, False, None
        return (time.perf_counter() - t0) * 1000, w0, ok, d


def summ(name, ms, oks):
    ms = sorted(ms)
    q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
    print(f"{name:<34} n={len(ms):>3}  p50 {q(.5):6.0f} ms  p90 {q(.9):6.0f}  p99/max {ms[-1]:6.0f}  errors {oks.count(False)}")
    return {"p50": q(.5), "p90": q(.9), "max": ms[-1], "errors": oks.count(False), "n": len(ms)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--demo-orders", type=int, default=15); ap.add_argument("--symbol", default="XRPUSDT")
    a = ap.parse_args(); load_dotenv(".env"); res = {}
    m = Client("mainnet")
    lat, oks, offs = [], [], []
    for _ in range(a.n):
        ms, w0, ok, d = m.call("GET", "/v5/market/time"); lat.append(ms); oks.append(ok)
        if ok: offs.append(int(d["time"]) - (w0 * 1000 + ms / 2))
        time.sleep(0.1)
    res["time"] = summ("mainnet public /market/time", lat, oks)
    print(f"{'clock offset (server - local)':<34} median {st.median(offs):+.0f} ms  range {min(offs):+.0f}..{max(offs):+.0f}  (recv_window 5000)")
    lat, oks = [], []
    for _ in range(max(5, a.n // 3)):
        ms, _, ok, d = m.call("GET", "/v5/market/tickers", {"category": "linear"}); lat.append(ms); oks.append(ok); time.sleep(0.2)
    res["tickers"] = summ("mainnet public tickers (all linear)", lat, oks)
    for path, p in (("/v5/position/list", {"category": "linear", "settleCoin": "USDT", "limit": 200}), ("/v5/order/realtime", {"category": "linear", "settleCoin": "USDT", "limit": 50}),
                    ("/v5/execution/list", {"category": "linear", "limit": 50})):
        lat, oks = [], []
        for _ in range(max(5, a.n // 2)):
            ms, _, ok, d = m.call("GET", path, p, signed=True); lat.append(ms); oks.append(ok); time.sleep(0.15)
        res[path] = summ(f"mainnet private {path.split('/')[-2]}/{path.split('/')[-1]}", lat, oks)
    lat, oks = [], []
    for _ in range(max(5, a.n // 3)):
        ms, _, ok, d = m.call("GET", "/v5/market/orderbook", {"category": "linear", "symbol": a.symbol, "limit": 50}); lat.append(ms); oks.append(ok); time.sleep(0.15)
    res["orderbook"] = summ("mainnet public orderbook", lat, oks)

    dm = Client("demo")
    if a.demo_orders and dm.key:
        _, _, ok, d = m.call("GET", "/v5/market/tickers", {"category": "linear", "symbol": a.symbol})  # public data from mainnet
        _, _, _, ins = m.call("GET", "/v5/market/instruments-info", {"category": "linear", "symbol": a.symbol})
        if not (ok and ins.get("retCode") == 0):
            print("demo order test skipped: market data failed", d.get("retMsg"), ins.get("retMsg"))
        else:
            t = d["result"]["list"][0]; i = ins["result"]["list"][0]
            tick = float(i["priceFilter"]["tickSize"]); step = float(i["lotSizeFilter"]["qtyStep"])
            minq = max(float(i["lotSizeFilter"]["minOrderQty"]), float(i["lotSizeFilter"].get("minNotionalValue", 5)) * 1.05 / float(t["bid1Price"]))
            qty = round(-(-minq // step) * step, 8); px = round(round(float(t["bid1Price"]) * 0.95 / tick) * tick, 8)
            place, cancel, ack, oks = [], [], [], []
            for k in range(a.demo_orders):
                ms, w0, ok, d = dm.call("POST", "/v5/order/create", {"category": "linear", "symbol": a.symbol, "side": "Buy", "orderType": "Limit", "qty": f"{qty}", "price": f"{px}",
                                                                   "timeInForce": "PostOnly", "orderLinkId": f"probe-{int(time.time()*1000)}-{k}"}, signed=True)
                place.append(ms); oks.append(ok)
                if not ok: print("  demo place failed:", d.get("retMsg", d)); continue
                oid = d["result"]["orderId"]
                ms2, _, ok2, d2 = dm.call("POST", "/v5/order/cancel", {"category": "linear", "symbol": a.symbol, "orderId": oid}, signed=True)
                cancel.append(ms2); oks.append(ok2)
                time.sleep(0.2)
            res["demo_place"] = summ(f"demo order/create ({a.symbol} post-only)", place, oks[::2] if place else [])
            if cancel: res["demo_cancel"] = summ("demo order/cancel", cancel, [True] * len(cancel))
            _, _, _, oo = dm.call("GET", "/v5/order/realtime", {"category": "linear", "symbol": a.symbol}, signed=True)
            print("demo leftovers (should be 0):", len([o for o in oo.get("result", {}).get("list", []) if o.get("orderLinkId", "").startswith("probe-")]))
    elif a.demo_orders:
        print("demo order test skipped: no demo key")
    print(json.dumps({k: {kk: round(vv) for kk, vv in v.items()} for k, v in res.items()}))


if __name__ == "__main__":
    main()
