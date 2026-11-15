"""Structural zones per contract from 4h bars (Bybit REST): fractal swing pivots clustered into ATR-scaled zones,
ranked by pivots, rejections and recency. Output feeds the market maker (`levels_file`), which only uses zones to
push quotes BEYOND them (fill on the sweep) - levels carry no directional evidence in our tests (P4).

    PYTHONIOENCODING=utf-8 uv run python scripts/levels.py [--config config/mm_multi.yaml] [--days 120] [--show CLUSDT BZUSDT XAUUSDT]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import UTC, datetime

import numpy as np
import yaml

from ai_trader.core.config import load


def bars4h(sym: str, n: int = 1000):
    u = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={sym}&interval=240&limit={n}"
    r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30))["result"]["list"][::-1]
    a = np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in r])
    ts = [datetime.fromtimestamp(int(x[0]) / 1000, UTC) for x in r]
    return ts, a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def zones(ts, o, h, lo, c, k: int = 3, top: int = 8, half_life_days: float = 30.0):
    n = len(c)
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(lo[1:], c[:-1])
    atr = float(np.median(tr[-180:])) if len(tr) else 0.0
    piv = []
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]):
            piv.append((h[i], i, "H"))
        if lo[i] == min(lo[i - k:i + k + 1]):
            piv.append((lo[i], i, "L"))
    piv.sort()
    clusters, cur = [], []
    for p in piv:
        if cur and p[0] - cur[0][0] > 0.6 * atr:
            clusters.append(cur)
            cur = []
        cur.append(p)
    if cur:
        clusters.append(cur)
    out = []
    now_i = n - 1
    for cl in clusters:
        prices = [p[0] for p in cl]
        zlo, zhi = min(prices), max(prices)
        if zhi - zlo < 0.25 * atr:
            m = 0.5 * (zlo + zhi)
            zlo, zhi = m - 0.125 * atr, m + 0.125 * atr
        rej = 0
        for i in range(1, n):
            touched = lo[i] <= zhi and h[i] >= zlo
            if touched and (c[i] > zhi or c[i] < zlo):
                rej += 1
        last = max(p[1] for p in cl)
        age_days = (now_i - last) * 4 / 24
        score = (2 * len(cl) + rej) * math.exp(-math.log(2) * age_days / half_life_days)
        out.append({"lo": round(zlo, 6), "hi": round(zhi, 6), "pivots": len(cl), "rejections": rej, "last_pivot": ts[last].strftime("%Y-%m-%d %H:%M"),
                    "score": round(score, 2), "kinds": "".join(sorted({p[2] for p in cl}))})
    out.sort(key=lambda z: -z["score"])
    return out[:top], atr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mm_multi.yaml")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--show", nargs="*", default=["CLUSDT", "BZUSDT", "XAUUSDT", "XAGUSDT"])
    a = ap.parse_args()
    root = load().root
    syms = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))["symbols"]
    res, meta = {}, {}
    for s in sorted(set(syms) | set(a.show)):
        try:
            ts, o, h, lo, c = bars4h(s, min(1000, a.days * 6))
        except Exception as e:  # noqa: BLE001
            print("skip", s, e)
            continue
        z, atr = zones(ts, o, h, lo, c)
        res[s] = z
        meta[s] = {"last": float(c[-1]), "atr_4h": atr, "bars": len(c), "from": ts[0].strftime("%Y-%m-%d")}
        time.sleep(0.1)
    out = root / "data" / "mm" / "levels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    (root / "data" / "mm" / "levels_meta.json").write_text(json.dumps({"as_of": datetime.now(UTC).isoformat(), "meta": meta}, indent=1), encoding="utf-8")
    print(f"zones for {len(res)} contracts -> {out}")
    for s in a.show:
        if s not in res:
            continue
        m = meta[s]
        print(f"\n{s}  last {m['last']}  4h ATR {m['atr_4h']:.4g}  ({m['bars']} bars from {m['from']})")
        for z in sorted(res[s], key=lambda z: -z["hi"]):
            tag = "  <- price" if z["lo"] <= m["last"] <= z["hi"] else ("  above" if z["lo"] > m["last"] else "  below")
            print(f"  {z['lo']:>10.4g} - {z['hi']:<10.4g} score {z['score']:5.1f}  pivots {z['pivots']}  rejections {z['rejections']:3d}  last pivot {z['last_pivot']}{tag}")


if __name__ == "__main__":
    main()
