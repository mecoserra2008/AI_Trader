"""Hourly alt shock scanner (watch W5, desk alts report 2026-11-15). Public Bybit data only; no orders.

    PYTHONIOENCODING=utf-8 uv run python scripts/alt_shock_scan.py [--z 4] [--top 80]

For the `--top` USDT linear perps by 24h turnover (innovation zone, TradFi and SPY excluded) it takes the last CLOSED
1h bar and computes z = r_1h / sd(previous 168 hourly log returns). Candidates have |z| >= --z, 1h ATR(14)/price <= 3%,
spread <= 3 bp and depth >= $100k within 25 bp of mid on both sides. Each candidate gets the direction the tested
shock_regime sleeve would take (fade in normal/high vol, follow in low vol; vol tercile = sd168 vs its own 90d history
approximated by the 30d hourly sd distribution). Three or more candidates in the same bar are one bet (R-MM-2):
the output flags it. Writes one JSON line per run to logs/alt_shock_scan.jsonl and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "alt_shock_scan.jsonl"
API = "https://api.bybit.com/v5/market"
UA = {"User-Agent": "AI_Trader alt shock scan (public market data)"}
TRADFI_HINT = ("XAU", "XAG", "CL", "BZ", "NVDA", "QQQ", "SOX", "MU", "SNDK", "MSTR", "COIN", "TSLA", "CRWV", "CRCL",
               "KORU", "SKHYNIX", "SPY", "AAPL", "AMZN", "GOOG", "META", "MSFT", "HOOD", "PLTR", "AMD", "NG")


def get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as r:
        return json.loads(r.read().decode())["result"]


def universe(top: int) -> list[dict]:
    inst, cursor = {}, ""
    while True:
        res = get(f"{API}/instruments-info?category=linear&limit=1000" + (f"&cursor={cursor}" if cursor else ""))
        for i in res["list"]:
            inst[i["symbol"]] = i
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    tick = get(f"{API}/tickers?category=linear")["list"]
    ok = []
    for t in tick:
        s = t["symbol"]; i = inst.get(s, {})
        if not s.endswith("USDT") or i.get("status") != "Trading" or i.get("contractType") != "LinearPerpetual":
            continue
        if i.get("symbolType") == "innovation" or any(s.startswith(h) for h in TRADFI_HINT):
            continue
        ok.append(t)
    ok.sort(key=lambda t: -float(t["turnover24h"]))
    return ok[:top]


def closed_1h(sym: str, n: int = 721) -> list[list[float]]:
    rows = get(f"{API}/kline?category=linear&symbol={sym}&interval=60&limit={min(n + 1, 1000)}")["list"]
    rows = [[float(x) for x in r[:5]] for r in reversed(rows)]  # oldest first: t, o, h, l, c
    now_ms = datetime.now(UTC).timestamp() * 1000
    return [r for r in rows if r[0] + 3_600_000 <= now_ms]


def book_ok(sym: str) -> dict:
    b = get(f"{API}/orderbook?category=linear&symbol={sym}&limit=200")
    bids = [(float(p), float(q)) for p, q in b["b"]]; asks = [(float(p), float(q)) for p, q in b["a"]]
    mid = (bids[0][0] + asks[0][0]) / 2
    spread_bp = 1e4 * math.log(asks[0][0] / bids[0][0])
    dbid = sum(p * q for p, q in bids if p >= mid * (1 - 0.0025)); dask = sum(p * q for p, q in asks if p <= mid * (1 + 0.0025))
    return {"spread_bp": round(spread_bp, 2), "depth_bid_usd": round(dbid), "depth_ask_usd": round(dask)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=4.0)
    ap.add_argument("--top", type=int, default=80)
    a = ap.parse_args()
    out, scanned, errors = [], 0, []
    bar = None
    for t in universe(a.top):
        s = t["symbol"]
        try:
            k = closed_1h(s)
            if len(k) < 200:
                continue
            scanned += 1
            r = [math.log(k[i][4] / k[i - 1][4]) for i in range(1, len(k))]
            sd = pstdev(r[-169:-1])
            if sd <= 0:
                continue
            z = r[-1] / sd
            bar = datetime.fromtimestamp(k[-1][0] / 1000, UTC).strftime("%Y-%m-%dT%H:%MZ")
            if abs(z) < a.z:
                continue
            tr = [max(k[i][2], k[i - 1][4]) - min(k[i][3], k[i - 1][4]) for i in range(len(k) - 14, len(k))]
            atr_pct = 100 * (sum(tr) / 14) / k[-1][4]
            sds = sorted(pstdev(r[j - 168:j]) for j in range(168, len(r), 24))
            pct = sum(x <= sd for x in sds) / len(sds)
            vol = "low" if pct < 1 / 3 else ("high" if pct > 2 / 3 else "normal")
            row = {"symbol": s, "bar": bar, "z": round(z, 2), "r1h_pct": round(100 * r[-1], 2), "atr1h_pct": round(atr_pct, 2),
                   "vol_tercile": vol, "sleeve_side": ("Buy" if z > 0 else "Sell") if vol == "low" else ("Sell" if z > 0 else "Buy"),
                   "close": k[-1][4], "turnover24h": round(float(t["turnover24h"]))}
            row.update(book_ok(s))
            row["tradable"] = (atr_pct <= 3 and row["spread_bp"] <= 3 and min(row["depth_bid_usd"], row["depth_ask_usd"]) >= 100_000)
            out.append(row)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: {type(e).__name__}")
    tradable = [o for o in out if o["tradable"]]
    rec = {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "bar": bar, "scanned": scanned, "z_min": a.z,
           "shocks": out, "tradable": [o["symbol"] for o in tradable], "one_bet_cluster": len(tradable) >= 3, "errors": errors[:10]}
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"bar {bar}: scanned {scanned}, shocks |z|>={a.z}: {len(out)}, tradable {len(tradable)}"
          + (" (CLUSTER: one bet)" if rec["one_bet_cluster"] else "") + (f", errors {len(errors)}" if errors else ""))
    for o in sorted(out, key=lambda o: -abs(o["z"])):
        print(f"  {o['symbol']:<14} z {o['z']:+6.2f} r {o['r1h_pct']:+6.2f}% atr {o['atr1h_pct']:.2f}% vol {o['vol_tercile']:<6}"
              f" sleeve {o['sleeve_side']:<4} spread {o['spread_bp']}bp depth {o['depth_bid_usd']:,}/{o['depth_ask_usd']:,}"
              f" {'TRADABLE' if o['tradable'] else 'no'}")


if __name__ == "__main__":
    main()
