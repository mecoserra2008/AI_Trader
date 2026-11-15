"""Live level + spot-buyer watch for BTC (public data only; no keys, no orders).

    PYTHONIOENCODING=utf-8 uv run python scripts/level_watch.py [--levels 67376.52 65904.12 ...] [--every 30]

Every `--every` seconds it reads:
  - Bybit BTCUSDT linear ticker (last, mark, index, funding, OI),
  - Binance spot BTCUSDT and Binance USD-M BTCUSDT 1m klines (taker-buy volume -> signed taker delta per minute),
  - Coinbase BTC-USD ticker (US spot; "Coinbase premium" = 1e4 * ln(coinbase / binance spot), in bp of log).
It appends one line per event to logs/level_watch.log:
  ALERT CROSS   price crossed a watched level (15s granularity; reports direction)
  ALERT CLOSE15 a 15m candle closed beyond a level it opened on the other side of
  ALERT SPOT    spot absorption: spot taker delta over 15m > +threshold while price fell, or spot leading perp
  ALERT PREM    Coinbase premium moved > 8 bp in 15m
  STATUS        every 15 min: price, levels around it, 15m/60m spot vs perp delta, premium, funding, OI
Spreads and premiums are log ratios (user convention).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "level_watch.log"
DEFAULT_LEVELS = [67376.52, 65904.12, 65270.68, 64494.49, 63836.28, 63000.0, 62067.96, 61899.2, 57756.0]
UA = {"User-Agent": "AI_Trader level watch (public market data)"}


def get(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def bybit():
    d = get("https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT")["result"]["list"][0]
    return {k: float(d[k]) for k in ("lastPrice", "markPrice", "indexPrice", "fundingRate", "openInterest")}


def binance_klines(spot: bool, limit: int = 61):
    base = "https://api.binance.com/api/v3/klines" if spot else "https://fapi.binance.com/fapi/v1/klines"
    rows = get(f"{base}?symbol=BTCUSDT&interval=1m&limit={limit}")
    out = []
    for r in rows[:-1]:  # closed minutes only
        vol, buy = float(r[5]), float(r[9])
        out.append({"t": int(r[6]), "o": float(r[1]), "c": float(r[4]), "delta": 2 * buy - vol})
    return out


def coinbase() -> float:
    return float(get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")["price"])


def log(line: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {line}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="*", default=DEFAULT_LEVELS)
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--spot-delta-btc", type=float, default=150.0, help="15m spot taker delta threshold for SPOT alerts")
    a = ap.parse_args()
    levels = sorted(a.levels)
    LOG.parent.mkdir(exist_ok=True)
    log(f"START levels={levels} every={a.every}s (public data only)")
    last_px = None
    prem_hist: deque = deque(maxlen=int(900 / a.every) + 1)
    last_status = 0.0
    last_spot_alert = 0.0
    last_15m_close_seen = None
    while True:
        try:
            b = bybit()
            px = b["lastPrice"]
            sp = binance_klines(True)
            pp = binance_klines(False)
            cb = coinbase()
            prem = 1e4 * math.log(cb / sp[-1]["c"]) if sp else float("nan")
            prem_hist.append((time.time(), prem))
            # level crosses
            if last_px is not None:
                for L in levels:
                    if (last_px - L) * (px - L) < 0:
                        log(f"ALERT CROSS {'UP' if px > L else 'DOWN'} {L:,.2f} (bybit last {px:,.1f}; from {last_px:,.1f})")
            last_px = px
            # 15m close beyond a level
            if len(sp) >= 15:
                end = sp[-1]["t"]
                if (end + 1) % 900000 == 0 and end != last_15m_close_seen:
                    last_15m_close_seen = end
                    o15, c15 = pp[-15]["o"], pp[-1]["c"]
                    for L in levels:
                        if (o15 - L) * (c15 - L) < 0:
                            log(f"ALERT CLOSE15 {'ABOVE' if c15 > L else 'BELOW'} {L:,.2f} (15m perp o {o15:,.1f} c {c15:,.1f})")
            # spot vs perp delta
            s15 = sum(k["delta"] for k in sp[-15:]); p15 = sum(k["delta"] for k in pp[-15:])
            s60 = sum(k["delta"] for k in sp[-60:]); p60 = sum(k["delta"] for k in pp[-60:])
            r15 = 100 * math.log(sp[-1]["c"] / sp[-15]["o"]) if len(sp) >= 15 else 0.0
            now = time.time()
            if now - last_spot_alert > 900:
                if s15 > a.spot_delta_btc and r15 < 0:
                    log(f"ALERT SPOT absorption: spot taker delta 15m +{s15:,.0f} BTC while price {r15:+.2f}% (perp delta {p15:+,.0f})")
                    last_spot_alert = now
                elif s15 > a.spot_delta_btc and s15 > 2 * max(p15, 1):
                    log(f"ALERT SPOT leading: spot delta 15m +{s15:,.0f} BTC vs perp {p15:+,.0f} (price {r15:+.2f}%)")
                    last_spot_alert = now
            # premium jump
            old = [p for t, p in prem_hist if now - t >= 840]
            if old and abs(prem - old[0]) > 8:
                log(f"ALERT PREM Coinbase premium {old[0]:+.1f} -> {prem:+.1f} bp (log) in 15m")
                prem_hist.clear()
            if now - last_status > 900:
                below = [L for L in levels if L < px][-2:]; above = [L for L in levels if L > px][:2]
                log(f"STATUS px {px:,.1f} | below {', '.join(f'{x:,.0f}' for x in below)} | above {', '.join(f'{x:,.0f}' for x in above)}"
                    f" | spot delta 15m {s15:+,.0f} 60m {s60:+,.0f} BTC | perp delta 15m {p15:+,.0f} 60m {p60:+,.0f} BTC"
                    f" | CB premium {prem:+.1f} bp | funding {100 * b['fundingRate']:+.4f}% | OI {b['openInterest']:,.0f}")
                last_status = now
        except Exception as e:  # noqa: BLE001 - keep watching through transient errors
            log(f"ERROR {type(e).__name__}: {str(e)[:160]}")
        time.sleep(a.every)


if __name__ == "__main__":
    main()
