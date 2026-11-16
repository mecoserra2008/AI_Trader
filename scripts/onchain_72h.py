"""On-chain flows over the last 72 h from free, keyless public endpoints (exploratory; no edge is claimed).

    .venv/Scripts/python.exe scripts/onchain_72h.py stables   # DefiLlama stablecoins: supply by coin and chain, daily
    .venv/Scripts/python.exe scripts/onchain_72h.py cex       # DefiLlama CEX wallets: 72 h balance change by token
    .venv/Scripts/python.exe scripts/onchain_72h.py btc       # mempool.space blocks + blockchain.info raw blocks (>= 500 BTC)
    .venv/Scripts/python.exe scripts/onchain_72h.py eth       # publicnode eth_getLogs: ERC-20 Transfers, exchange-labelled
    .venv/Scripts/python.exe scripts/onchain_72h.py match     # Bybit price / OI / funding / taker flow around the events

Outputs: data/research/onchain/*.json. Labels come only from documented public lists:
  Bybit PoR CSV (api2.bybit.com/common-static/por/bybit_por_202610.csv, the file DefiLlama's cex/bybit.js reads),
  Binance PoR addresses (www.binance.com/bapi/apex/v1/public/apex/market/por/address, read by DefiLlama's binance adapter),
  OKX BTC addresses (DefiLlama-Adapters projects/helper/bitcoin-book/okex.js). Everything else is "unknown".
PoR lists hold reserve / hot wallets, not per-user deposit addresses, so deposits that have not been swept yet are invisible.
Public, keyless data only; sleeps between calls and backs off on 429.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "research" / "onchain"
UA = "Mozilla/5.0 (AI_Trader research; public data)"
AS_OF = datetime.now(UTC).replace(second=0, microsecond=0)
START = AS_OF - timedelta(hours=72)


def get(url: str, body: dict | None = None, tries: int = 5, timeout: int = 60):
    for k in range(tries):
        try:
            hdr = {"User-Agent": UA, "Accept-Encoding": "gzip"}
            data = None
            if body is not None:
                hdr["Content-Type"] = "application/json"
                data = json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=hdr), timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (429, 418, 503):
                time.sleep(min(60, 5 * 2**k))
                continue
            raise
        except Exception:  # noqa: BLE001 - retried
            time.sleep(2 * (k + 1))
    raise OSError(f"failed: {url}")


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT / name)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%MZ")


# ------------------------------------------------------------------ labels

def labels() -> dict[str, str]:
    lab: dict[str, str] = {}
    csv = OUT / "bybit_por_202610.csv"
    if not csv.exists():
        txt = urllib.request.urlopen(urllib.request.Request(
            "https://api2.bybit.com/common-static/por/bybit_por_202610.csv", headers={"User-Agent": UA}), timeout=60).read()
        csv.write_bytes(txt)
    for line in csv.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",", 3)
        if len(parts) == 4 and parts[3].strip():
            for a in parts[3].split():
                lab[a.lower() if a.startswith("0x") else a] = "Bybit"
    bn = OUT / "binance_por_addresses.json"
    if not bn.exists():
        save("binance_por_addresses.json", get("https://www.binance.com/bapi/apex/v1/public/apex/market/por/address"))
    for x in json.loads(bn.read_text(encoding="utf-8"))["data"]:
        a = x["address"]
        lab[a.lower() if a.startswith("0x") else a] = "Binance"
    ok = OUT / "okx_btc_book.js"
    if not ok.exists():
        ok.write_bytes(urllib.request.urlopen(urllib.request.Request(
            "https://raw.githubusercontent.com/DefiLlama/DefiLlama-Adapters/main/projects/helper/bitcoin-book/okex.js",
            headers={"User-Agent": UA}), timeout=60).read())
    import re
    for a in re.findall(r'"([13bc][a-zA-Z0-9]{25,90})"', ok.read_text(encoding="utf-8")):
        lab.setdefault(a, "OKX")
    return lab


# ------------------------------------------------------------------ 1. stablecoins

def stables() -> None:
    allp = get("https://stablecoins.llama.fi/stablecoins?includePrices=true")["peggedAssets"]
    top = sorted(allp, key=lambda x: -(x.get("circulating", {}).get("peggedUSD") or 0))[:12]
    coins = []
    for c in top:
        cur = c["circulating"].get("peggedUSD") or 0
        coins.append({"id": c["id"], "symbol": c["symbol"], "circ_usd": cur,
                      "d1_usd": cur - (c.get("circulatingPrevDay", {}).get("peggedUSD") or 0),
                      "d7_usd": cur - (c.get("circulatingPrevWeek", {}).get("peggedUSD") or 0),
                      "chains_d1": sorted([(ch, (v.get("current", {}).get("peggedUSD") or 0)
                                            - (v.get("circulatingPrevDay", {}).get("peggedUSD") or 0))
                                           for ch, v in (c.get("chainCirculating") or {}).items()],
                                          key=lambda kv: -abs(kv[1]))[:6]})
    # daily per-chain history for USDT (id 1), USDC (id 2) and the top others
    hist = {}
    for c in top[:5]:
        d = get(f"https://stablecoins.llama.fi/stablecoin/{c['id']}")
        rows = {}
        for ch, v in (d.get("chainBalances") or {}).items():
            pts = [(int(p["date"]), (p.get("circulating") or {}).get("peggedUSD") or 0) for p in v.get("tokens", [])]
            pts = [p for p in pts if p[0] >= START.timestamp() - 2 * 86400]
            if len(pts) >= 2 and max(abs(b[1] - a[1]) for a, b in zip(pts, pts[1:], strict=False)) > 2e7:
                rows[ch] = [(iso(t), round(x)) for t, x in pts]
        hist[c["symbol"]] = rows
        time.sleep(1)
    total = [(iso(int(p["date"])), p["totalCirculatingUSD"].get("peggedUSD"))
             for p in get("https://stablecoins.llama.fi/stablecoincharts/all")[-6:]]
    save("stablecoins_72h.json", {"as_of": AS_OF.isoformat(), "granularity": "daily (00:00 UTC); no hourly series",
                                  "coins": coins, "per_chain_daily_moves_gt_20m": hist, "total_daily": total})


# ------------------------------------------------------------------ 2. CEX wallets (DefiLlama)

CEX = ["Bybit", "Binance", "OKX", "Bitfinex", "Robinhood", "Gemini", "Crypto.com", "HTX", "KuCoin", "Deribit", "Gate",
       "Bitget", "MEXC", "Coinbase", "Upbit"]   # Coinbase / Upbit: listed by DefiLlama without wallet data (no slug)
FOCUS = ["BTC", "WBTC", "ETH", "WETH", "SOL", "SUI", "ZEC", "XRP", "DOGE", "HYPE", "USDT", "USDC", "USDE", "KAVA",
         "SOPH", "RARE", "LINK", "ENA", "ONDO", "PEPE", "AAVE", "UNI", "TRX", "BNB"]


def cex() -> None:
    lst = get("https://api.llama.fi/cexs")["cexs"]
    slugs = {x["name"]: x.get("slug") for x in lst}
    out = {"as_of": AS_OF.isoformat(), "source": "api.llama.fi/protocol/<slug> (CEX wallets, daily + one intraday point)",
           "cexs": {}}
    for name in CEX:
        slug = slugs.get(name)
        if slug is None:
            out["cexs"][name] = {"error": "no wallet data on DefiLlama (no slug in api.llama.fi/cexs)"}
            continue
        try:
            d = get(f"https://api.llama.fi/protocol/{slug}", timeout=120)
        except Exception as e:  # noqa: BLE001
            out["cexs"][name] = {"error": str(e)[:120]}
            continue
        # sum token units across real chains (skip *-OwnTokens, borrowed, etc.)
        units: dict[int, dict[str, float]] = {}
        usd: dict[int, dict[str, float]] = {}
        for ch, v in d.get("chainTvls", {}).items():
            if "-" in ch:
                continue
            for key, tgt in (("tokens", units), ("tokensInUsd", usd)):
                for p in v.get(key) or []:
                    if p["date"] < START.timestamp() - 86400 * 1.5:
                        continue
                    t = tgt.setdefault(p["date"], {})
                    for tok, x in p["tokens"].items():
                        t[tok] = t.get(tok, 0.0) + (x or 0.0)
        ts = sorted(units)
        if len(ts) < 2:
            out["cexs"][name] = {"error": "no recent points"}
            continue
        t0 = next((t for t in ts if t >= START.timestamp() - 3600 * 12), ts[0])
        t1 = ts[-1]
        rows = []
        for tok in set(units[t1]) | set(units[t0]):
            u0, u1 = units[t0].get(tok, 0.0), units[t1].get(tok, 0.0)
            px = (usd[t1].get(tok, 0.0) / u1) if u1 else ((usd[t0].get(tok, 0.0) / u0) if u0 else 0.0)
            path = [(iso(t), round(units[t].get(tok, 0.0), 4)) for t in ts if t >= t0]
            rows.append({"token": tok, "units_t0": u0, "units_t1": u1, "d_units": u1 - u0, "px": px,
                         "d_usd_at_px1": (u1 - u0) * px, "d_pct": 100 * (u1 / u0 - 1) if u0 else None, "path": path})
        rows.sort(key=lambda r: -abs(r["d_usd_at_px1"]))
        out["cexs"][name] = {"slug": slug, "t0": iso(t0), "t1": iso(t1), "points": [iso(t) for t in ts],
                             "tvl_t0": sum(usd[t0].values()), "tvl_t1": sum(usd[t1].values()),
                             "top_moves": rows[:25], "focus": [r for r in rows if r["token"] in FOCUS]}
        print(name, slug, iso(t0), iso(t1), [(r["token"], round(r["d_usd_at_px1"] / 1e6, 1)) for r in rows[:6]])
        del d
        time.sleep(1.5)
    save("cex_wallets_72h.json", out)


# ------------------------------------------------------------------ 3. BTC

def btc(cap_s: float = 20 * 60) -> None:
    lab = labels()
    tip = int(urllib.request.urlopen("https://mempool.space/api/blocks/tip/height", timeout=30).read())
    blocks = []
    h = tip
    while True:
        bl = get(f"https://mempool.space/api/v1/blocks/{h}")
        for b in bl:
            ex = b.get("extras", {})
            blocks.append({"height": b["height"], "id": b["id"], "time": b["timestamp"], "iso": iso(b["timestamp"]),
                           "tx_count": b["tx_count"], "median_fee": ex.get("medianFee"), "fee_range": ex.get("feeRange"),
                           "total_fees_btc": (ex.get("totalFees") or 0) / 1e8, "pool": (ex.get("pool") or {}).get("name")})
        h = bl[-1]["height"] - 1
        if bl[-1]["timestamp"] < START.timestamp():
            break
        time.sleep(0.4)
    blocks = [b for b in blocks if b["time"] >= START.timestamp()]
    save("btc_blocks_72h.json", {"as_of": AS_OF.isoformat(), "tip": tip, "blocks": blocks})
    # raw blocks: interleaved order so a time cap still samples the whole window evenly
    order = blocks[::2] + blocks[1::2]
    large, exch, done = [], [], []
    t_begin = time.time()
    for b in order:
        if time.time() - t_begin > cap_s:
            break
        try:
            rb = get(f"https://blockchain.info/rawblock/{b['id']}", timeout=90)
        except Exception as e:  # noqa: BLE001
            print("skip", b["height"], e)
            continue
        for tx in rb.get("tx", []):
            outs = tx.get("out", [])
            ins = [i.get("prev_out") or {} for i in tx.get("inputs", [])]
            tot_out = sum(o.get("value", 0) for o in outs) / 1e8
            in_lab = {lab.get(i.get("addr", ""), "unknown") for i in ins if i.get("addr")}
            out_lab = {}
            for o in outs:
                lb = lab.get(o.get("addr", ""))
                if lb:
                    out_lab[lb] = out_lab.get(lb, 0) + o.get("value", 0) / 1e8
            in_val = {}
            for i in ins:
                lb = lab.get(i.get("addr", ""))
                if lb:
                    in_val[lb] = in_val.get(lb, 0) + i.get("value", 0) / 1e8
            big = [o for o in outs if o.get("value", 0) >= 500 * 1e8]
            if big:
                large.append({"txid": tx["hash"], "height": b["height"], "time": b["iso"], "total_out_btc": round(tot_out, 4),
                              "n_in": len(ins), "n_out": len(outs), "in_labels": sorted(in_lab),
                              "big_outs": [{"btc": o["value"] / 1e8, "addr": o.get("addr"),
                                            "label": lab.get(o.get("addr", ""), "unknown")} for o in big]})
            for lb in set(in_val) | set(out_lab):
                net = out_lab.get(lb, 0) - in_val.get(lb, 0)       # + = into the labelled wallets
                if abs(net) >= 10:
                    exch.append({"txid": tx["hash"], "time": b["iso"], "height": b["height"], "entity": lb,
                                 "net_in_btc": round(net, 4), "internal": lb in in_val and lb in out_lab})
        done.append(b["height"])
        del rb
        time.sleep(2.5)
    save("btc_large_72h.json", {"as_of": AS_OF.isoformat(), "blocks_in_window": len(blocks), "blocks_scanned": len(done),
                                "scanned_heights": sorted(done), "threshold": ">= 500 BTC single output",
                                "labels": "Bybit PoR 2026-10, Binance PoR, OKX (DefiLlama bitcoin-book); else unknown",
                                "large": large, "labelled_flows_ge_10btc": exch})


# ------------------------------------------------------------------ 4. ETH / ERC-20

RPC = "https://ethereum-rpc.publicnode.com"
TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
TOKENS = {  # symbol: (address, decimals, all_transfers?)
    "USDT": ("0xdac17f958d2ee523a2206206994597c13d831ec7", 6, False),
    "USDC": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6, False),
    "WETH": ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 18, False),
    "LINK": ("0x514910771af9ca656af840dff83e8264ecf986ca", 18, True),
    "UNI": ("0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", 18, True),
    "AAVE": ("0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", 18, True),
    "ENA": ("0x57e114b691db790c35207b2e685d4a43181e6061", 18, True),
    "ONDO": ("0xfaba6f8e4a5e8ab82f62fe7c39859fa577269be3", 18, True),
    "PEPE": ("0x6982508145454ce325ddbe47a25d4ec3d2311933", 18, True),
    "SHIB": ("0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", 18, True),
}
BYBIT_PX = {"USDT": "USDCUSDT", "USDC": "USDCUSDT", "WETH": "ETHUSDT", "LINK": "LINKUSDT", "UNI": "UNIUSDT",
            "AAVE": "AAVEUSDT", "ENA": "ENAUSDT", "ONDO": "ONDOUSDT", "PEPE": "1000PEPEUSDT", "SHIB": "SHIB1000USDT"}


def rpc(method: str, params: list):
    for k in range(6):
        try:
            j = get(RPC, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, tries=3)
        except urllib.error.HTTPError as e:
            if e.code == 403 and k < 4:                          # publicnode throttles bursts with 403: wait it out
                time.sleep(20 * (k + 1))
                continue
            if 400 <= e.code < 500 and e.code != 429:
                raise ValueError(f"http {e.code}") from e      # too many results / range: the caller shrinks the range
            time.sleep(3 * (k + 1))
            continue
        except Exception:  # noqa: BLE001
            time.sleep(3 * (k + 1))
            continue
        if "error" in j:
            raise ValueError(j["error"])
        return j["result"]
    raise OSError(method)


def logs(addr, topics, b0: int, b1: int, step: int) -> list:
    out, b = [], b0
    while b <= b1:
        e = min(b1, b + step - 1)
        try:
            out += rpc("eth_getLogs", [{"address": addr, "topics": topics, "fromBlock": hex(b), "toBlock": hex(e)}])
            b = e + 1
            time.sleep(0.6)
        except ValueError as err:
            if step <= 50:
                raise
            step //= 2
            print("  shrink step", step, str(err)[:80])
    return out


def eth() -> None:
    lab = labels()
    ex_addrs = sorted({a for a, v in lab.items() if a.startswith("0x") and len(a) == 42})
    px = {}
    tick = {t["symbol"]: float(t["lastPrice"]) for t in get("https://api.bybit.com/v5/market/tickers?category=linear")["result"]["list"]}
    for s, k in BYBIT_PX.items():
        px[s] = tick.get(k, 0.0) / (1000 if k.startswith("1000") or k.startswith("SHIB1000") else 1)
    px["USDT"] = 1.0
    head = int(rpc("eth_blockNumber", []), 16)
    t_head = int(rpc("eth_getBlockByNumber", [hex(head), False])["timestamp"], 16)
    b0 = head - int((t_head - START.timestamp()) / 12) - 10
    t_b0 = int(rpc("eth_getBlockByNumber", [hex(b0), False])["timestamp"], 16)
    print("blocks", b0, head, iso(t_b0), iso(t_head), len(ex_addrs), "labelled eth addrs")
    pad = ["0x" + "0" * 24 + a[2:] for a in ex_addrs]
    zero = "0x" + "0" * 64
    rows = []

    def add(lgs, sym, kind):
        a, dec, _ = TOKENS[sym]
        for x in lgs:
            frm, to = "0x" + x["topics"][1][26:], "0x" + x["topics"][2][26:]
            amt = int(x["data"], 16) / 10**dec if x["data"] not in ("0x", "") else 0.0
            rows.append({"sym": sym, "kind": kind, "block": int(x["blockNumber"], 16), "tx": x["transactionHash"],
                         "from": frm, "to": to, "from_label": lab.get(frm, "mint" if frm == "0x" + "0" * 40 else "unknown"),
                         "to_label": lab.get(to, "burn" if to == "0x" + "0" * 40 else "unknown"),
                         "amount": amt, "usd": amt * px.get(sym, 0.0)})

    failed = []
    for sym, (a, dec, all_tr) in TOKENS.items():
      t = time.time()
      try:
        # Exchange-filtered queries (topic OR-lists of the 509 labelled addresses) were refused by publicnode with
        # persistent HTTP 403 after the first calls (2026-11-16 00:00-00:25 UTC), so USDT / USDC / WETH exchange
        # flows are NOT covered; the all-transfer scan below labels both ends of every large transfer instead.
        if sym == "USDC":
            add(logs(a, [TRANSFER, zero], b0, head, 10000), sym, "mint")
            add(logs(a, [TRANSFER, None, zero], b0, head, 10000), sym, "burn")
        if all_tr:
            lg = logs(a, [TRANSFER], b0, head, 2000)
            thr = 1e6 / px[sym] if px.get(sym) else float("inf")
            add([x for x in lg if x["data"] not in ("0x", "") and int(x["data"], 16) / 10**dec >= thr], sym, "large_any")
      except Exception as e:  # noqa: BLE001 - keep the other tokens
        failed.append(f"{sym}: {str(e)[:100]}")
      print(sym, "done", round(time.time() - t), "s rows", len(rows), flush=True)
    # USDT Issue / Redeem events (Tether mints / burns on Ethereum do not emit Transfer from 0x0)
    issue = "0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a"
    redeem = "0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44"
    mint_burn = []
    for topic, kind in ((issue, "USDT_issue"), (redeem, "USDT_redeem")):
        for x in logs(TOKENS["USDT"][0], [topic], b0, head, 20000):
            mint_burn.append({"kind": kind, "block": int(x["blockNumber"], 16), "tx": x["transactionHash"],
                              "usd": int(x["data"], 16) / 1e6})
    # block -> time (linear interpolation between b0 and head is within ~1 block of truth at 12 s slots)
    def btime(n):
        return iso(t_b0 + (n - b0) * (t_head - t_b0) / (head - b0))
    seen = set()
    uniq = []
    for r in rows:
        k = (r["tx"], r["from"], r["to"], r["amount"], r["sym"])
        if k in seen:
            continue
        seen.add(k)
        r["time"] = btime(r["block"])
        uniq.append(r)
    for r in mint_burn:
        r["time"] = btime(r["block"])
    # hourly net flow into each labelled entity by token (exchange-internal transfers cancel out)
    agg: dict = {}
    for r in uniq:
        if r["kind"] not in ("from_exchange", "to_exchange", "large_any"):
            continue
        h = r["time"][:13] + ":00Z"
        for ent, sgn in ((r["to_label"], 1), (r["from_label"], -1)):
            if ent in ("Bybit", "Binance"):
                k = f"{ent}|{r['sym']}|{h}"
                agg[k] = agg.get(k, 0.0) + sgn * r["usd"]
    uniq = [r for r in uniq if r["usd"] >= 250_000 or r["kind"] in ("mint", "burn")]
    save("eth_erc20_72h.json", {"as_of": AS_OF.isoformat(), "rpc": RPC, "blocks": [b0, head], "window": [iso(t_b0), iso(t_head)],
                                "prices_bybit": px, "n_labelled_addrs": len(ex_addrs), "hourly_net_in_usd": agg,
                                "transfers_ge_250k_usd": uniq,
                                "usdt_issue_redeem": mint_burn, "failed": failed})


# ------------------------------------------------------------------ 5. match

def match() -> None:
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "ZECUSDT", "XRPUSDT", "DOGEUSDT", "HYPEUSDT", "KAVAUSDT",
            "SOPHUSDT", "RAREUSDT", "LINKUSDT", "ENAUSDT", "ONDOUSDT", "AAVEUSDT", "UNIUSDT", "1000PEPEUSDT"]
    s_ms, e_ms = int((START - timedelta(hours=12)).timestamp() * 1000), int(AS_OF.timestamp() * 1000)
    out = {"as_of": AS_OF.isoformat(), "symbols": {}}
    import duckdb
    q = (f"read_parquet('{(ROOT / 'data/lake/flow_3s/venue=bybit/category=linear').as_posix()}/date=*/*.parquet', "
         "hive_partitioning=1, union_by_name=1)")
    flow = duckdb.sql(f"""select symbol, epoch(date_trunc('hour', event_ts)) h, sum(buy_notional) b, sum(sell_notional) s
                         from {q} where date >= DATE '{(START - timedelta(days=1)).date()}' and
                         event_ts >= TIMESTAMPTZ '{(START - timedelta(hours=12)).isoformat()}' and
                         known_ts <= TIMESTAMPTZ '{AS_OF.isoformat()}' and symbol in {tuple(syms)}
                         group by 1, 2 order by 1, 2""").fetchall()
    fl: dict = {}
    for sym, h, b, s in flow:
        fl.setdefault(sym, {})[iso(h)] = (round(b or 0), round(s or 0))
    for sym in syms:
        k = get(f"https://api.bybit.com/v5/market/kline?category=linear&symbol={sym}&interval=60&start={s_ms}&end={e_ms}&limit=200")
        oi = get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={sym}&intervalTime=1h&startTime={s_ms}&endTime={e_ms}&limit=200")
        fu = get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={sym}&startTime={s_ms}&endTime={e_ms}&limit=200")
        bt = {}
        try:
            bsym = sym
            bt = {iso(x["timestamp"] / 1000): (float(x["buyVol"]), float(x["sellVol"])) for x in
                  get(f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={bsym}&period=1h&limit=90")}
        except Exception:  # noqa: BLE001
            pass
        bars = sorted(((iso(int(r[0]) / 1000), float(r[1]), float(r[4]), float(r[6])) for r in (k.get("result") or {}).get("list", [])))
        ois = {iso(int(r["timestamp"]) / 1000): float(r["openInterest"]) for r in (oi.get("result") or {}).get("list", [])}
        fus = sorted((iso(int(r["fundingRateTimestamp"]) / 1000), float(r["fundingRate"])) for r in (fu.get("result") or {}).get("list", []))
        out["symbols"][sym] = {"bars_1h": [{"t": t, "o": o, "c": c, "turnover": v, "oi": ois.get(t),
                                            "by_taker_buy": (fl.get(sym, {}).get(t) or (None, None))[0],
                                            "by_taker_sell": (fl.get(sym, {}).get(t) or (None, None))[1],
                                            "bn_taker_buy_base": (bt.get(t) or (None, None))[0],
                                            "bn_taker_sell_base": (bt.get(t) or (None, None))[1]} for t, o, c, v in bars],
                               "funding": fus}
        time.sleep(0.3)
    save("bybit_context_72h.json", out)


if __name__ == "__main__":
    {"stables": stables, "cex": cex, "btc": btc, "eth": eth, "match": match}[sys.argv[1]]()
