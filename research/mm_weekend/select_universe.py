"""H-MMW universe (pre-declared 2026-11-21): Bybit USDT linear perps, Trading, not innovation zone, not TradFi, not
SPY/BTC/ETH; 24h turnover 5-50 M USD; quoted spread >= 4 bp (median of 5 snapshots); depth within 25 bp >= 50 k USD per
side (min); median |1m log return| over the last 24 h <= 3 x the spread. Rank by spread / median|r1m| (more spread per unit
of 1-minute risk first), keep the top 10. Writes research/mm_weekend/universe.json."""
import json, math, statistics, time, urllib.request
A = "https://api.bybit.com/v5/market"
def g(path):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(A + path, headers={"User-Agent": "AI_Trader"}), timeout=15) as r:
                d = json.loads(r.read())
            if d.get("retCode") == 0: return d["result"]
        except Exception: pass
        time.sleep(1 + k)
    raise RuntimeError(path)
inst = []; cur = ""
while True:
    r = g(f"/instruments-info?category=linear&limit=1000" + (f"&cursor={cur}" if cur else ""))
    inst += r["list"]; cur = r.get("nextPageCursor") or ""
    if not cur: break
TRADFI_HINT = ("XAU", "XAG", "CL", "BZ", "NG", "SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "COIN", "MSTR",
               "CRWD", "ORCL", "CRWV", "HOOD", "SOXL", "PLTR", "AMD", "NFLX", "INTC", "ANT", "XPT", "XPD", "COPPER", "EUR", "GBP", "JPY")
ok = {}
for i in inst:
    s = i["symbol"]
    if i.get("status") != "Trading" or i.get("quoteCoin") != "USDT" or i.get("contractType") != "LinearPerpetual": continue
    if i.get("symbolType", "") not in ("", "crypto") or i.get("isPreListing"): continue
    base = s[:-4]
    if base in ("BTC", "ETH", "USDC", "USDE", "FDUSD") or base.startswith(TRADFI_HINT) or "innovation" in json.dumps(i).lower(): continue
    ok[s] = i
tk = {t["symbol"]: t for t in g("/tickers?category=linear")["list"] if t["symbol"] in ok}
pre = [s for s, t in tk.items() if 5e6 <= float(t["turnover24h"]) <= 50e6]
rows = []
for s in pre:
    try:
        sp = []
        for _ in range(5):
            ob = g(f"/orderbook?category=linear&symbol={s}&limit=200")
            b, a = float(ob["b"][0][0]), float(ob["a"][0][0]); m = (a + b) / 2; sp.append(1e4 * (a - b) / m)
            time.sleep(0.15)
        bid25 = sum(float(p) * float(q) for p, q in ob["b"] if float(p) >= m * (1 - 25e-4))
        ask25 = sum(float(p) * float(q) for p, q in ob["a"] if float(p) <= m * (1 + 25e-4))
        k = g(f"/kline?category=linear&symbol={s}&interval=1&limit=1000")["list"][::-1]
        c = [float(x[4]) for x in k]; r1 = statistics.median(abs(math.log(c[j] / c[j - 1])) for j in range(1, len(c))) * 1e4
        spm = statistics.median(sp)
        rows.append({"symbol": s, "turnover24h": float(tk[s]["turnover24h"]), "spread_bp": round(spm, 2), "depth25_min_usd": round(min(bid25, ask25)),
                     "med_abs_r1m_bp": round(r1, 2), "spread_per_r1m": round(spm / max(r1, 1e-9), 3), "symbolType": ok[s].get("symbolType", "")})
    except Exception as e:
        rows.append({"symbol": s, "error": str(e)})
elig = [r for r in rows if "error" not in r and r["spread_bp"] >= 4 and r["depth25_min_usd"] >= 50_000 and r["med_abs_r1m_bp"] <= 3 * r["spread_bp"]]
elig.sort(key=lambda r: -r["spread_per_r1m"])
out = {"as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "criteria": __doc__, "n_instruments": len(ok), "n_turnover_band": len(pre),
       "eligible": elig, "universe": [r["symbol"] for r in elig[:10]], "all": rows}
json.dump(out, open("research/mm_weekend/universe.json", "w"), indent=1)
print(len(ok), "instruments;", len(pre), "in turnover band;", len(elig), "eligible")
for r in elig[:15]: print(r)
