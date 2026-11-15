"""Analyse a market-making session from Bybit's own records (private API, read-only; keys from .env, never printed).

    PYTHONIOENCODING=utf-8 uv run python scripts/mm_analyze.py --since 2026-11-08T22:59:00Z [--until NOW] [--env mainnet]

Pulls executions (/v5/execution/list), order history (/v5/order/history) and closed P&L (/v5/position/closed-pnl),
saves them to data/mm/analysis/<env>-<stamp>/ and reports:
  - totals: fills, maker/taker, turnover, fees, realised P&L (Bybit closed-pnl) and fill-ledger P&L
  - order funnel: placed / filled / partially filled / cancelled / rejected (post-only), fill rate, requote count per fill
  - spread capture: fill price vs the mid at fill time (flow_3s mids from the lake), in bp
  - markouts: signed mid move 10 s / 60 s / 300 s after each maker fill, in bp (negative = adverse selection)
  - round trips per symbol: FIFO lots, holding time, P&L, win rate
  - per symbol and per side attribution
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from dotenv import load_dotenv

from ai_trader.core.config import load
from ai_trader.storage.lake import Lake

ROOT = load().root
HOSTS = {"mainnet": "https://api.bybit.com", "demo": "https://api-demo.bybit.com"}


def client(env: str):
    load_dotenv(ROOT / ".env")
    key, sec = os.environ[f"BYBIT_{env.upper()}_API_KEY"], os.environ[f"BYBIT_{env.upper()}_API_SECRET"]

    def get(path: str, params: dict, tries: int = 4) -> dict:
        for k in range(tries):  # slow links: retry recv_window rejections and timeouts with a fresh signature
            try:
                return _get(path, params)
            except Exception:  # noqa: BLE001
                if k == tries - 1:
                    raise
                time.sleep(1 + k)

    def _get(path: str, params: dict) -> dict:
        ts, rw = str(int(time.time() * 1000)), "5000"
        q = urllib.parse.urlencode(params)
        sig = hmac.new(sec.encode(), (ts + key + rw + q).encode(), hashlib.sha256).hexdigest()
        req = urllib.request.Request(f"{HOSTS[env]}{path}?{q}", headers={"X-BAPI-API-KEY": key, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": rw, "X-BAPI-SIGN": sig})
        d = json.load(urllib.request.urlopen(req, timeout=30))
        if d["retCode"] != 0:
            raise RuntimeError(f"{path}: {d['retMsg']}")
        return d["result"]

    return get


def paged(get, path: str, base: dict, start: int, end: int, window_ms: int = 7 * 86400_000) -> list[dict]:
    out = []
    s = start
    while s < end:
        e = min(end, s + window_ms)
        cur = ""
        while True:
            p = {**base, "startTime": s, "endTime": e, "limit": 100 if "execution" in path or "closed" in path else 50}
            if cur:
                p["cursor"] = cur
            r = get(path, p)
            out += r.get("list", [])
            cur = r.get("nextPageCursor") or ""
            if not cur:
                break
        s = e
    return out


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default=None)
    ap.add_argument("--env", default="mainnet")
    ap.add_argument("--exclude", nargs="*", default=["SPYUSDT"], help="symbols that are not the quoter's (default: the user's own SPYUSDT)")
    ap.add_argument("--leaderboard", default=None, help="quoter leaderboard.json for the mr_score vs P&L check")
    a = ap.parse_args()
    t0 = datetime.fromisoformat(a.since.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(a.until.replace("Z", "+00:00")) if a.until else datetime.now(UTC)
    s_ms, e_ms = int(t0.timestamp() * 1000), int(t1.timestamp() * 1000)
    get = client(a.env)
    tlog = paged(get, "/v5/account/transaction-log", {"accountType": "UNIFIED", "category": "linear"}, s_ms, e_ms)
    ex = paged(get, "/v5/execution/list", {"category": "linear"}, s_ms, e_ms)
    orders = paged(get, "/v5/order/history", {"category": "linear"}, s_ms, e_ms)
    cpnl = paged(get, "/v5/position/closed-pnl", {"category": "linear"}, s_ms, e_ms)
    pos = get("/v5/position/list", {"category": "linear", "settleCoin": "USDT", "limit": 200}).get("list", [])
    ex, orders, cpnl, tlog, pos = ([x for x in L if x.get("symbol") not in a.exclude] for L in (ex, orders, cpnl, tlog, pos))
    out = ROOT / "data" / "mm" / "analysis" / f"{a.env}-{t1:%Y%m%dT%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("executions", ex), ("orders", orders), ("closed_pnl", cpnl), ("transaction_log", tlog), ("positions", pos)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")

    E = pl.DataFrame([{"sym": x["symbol"], "side": x["side"], "qty": f(x["execQty"]), "px": f(x["execPrice"]), "val": f(x["execValue"]), "fee": f(x["execFee"]),
                       "maker": bool(x["isMaker"]), "type": x["execType"], "ts": int(x["execTime"]), "link": x.get("orderLinkId") or "", "oid": x["orderId"]} for x in ex]).filter(pl.col("type") == "Trade").sort("ts")
    E = E.with_columns(pl.col("link").str.starts_with("mm").alias("mm_order"), pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC").alias("t"))
    O = pl.DataFrame([{"sym": x["symbol"], "side": x["side"], "status": x["orderStatus"], "type": x["orderType"], "tif": x.get("timeInForce"), "reduce": x.get("reduceOnly"),
                       "qty": f(x["qty"]), "cum": f(x.get("cumExecQty")), "px": f(x.get("price")), "created": int(x["createdTime"]), "updated": int(x["updatedTime"]),
                       "link": x.get("orderLinkId") or "", "reject": x.get("rejectReason") or ""} for x in orders]) if orders else pl.DataFrame()
    C = pl.DataFrame([{"sym": x["symbol"], "side": x["side"], "qty": f(x["closedSize"]), "pnl": f(x["closedPnl"]), "entry": f(x["avgEntryPrice"]), "exit": f(x["avgExitPrice"]),
                       "created": int(x["createdTime"]), "updated": int(x["updatedTime"])} for x in cpnl]) if cpnl else pl.DataFrame()

    print(f"== {a.env} session {t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M} UTC  (raw saved to {out})")
    mk, tk = E.filter(pl.col("maker")), E.filter(~pl.col("maker"))
    sign = pl.when(pl.col("side") == "Sell").then(1.0).otherwise(-1.0)
    cash = float(E.select((sign * pl.col("val")).sum()).item() - E["fee"].sum())
    print(f"fills {E.height} (maker {mk.height}, taker {tk.height}) | turnover ${E['val'].sum():,.2f} (maker ${mk['val'].sum():,.2f}) | fees ${E['fee'].sum():.4f} "
          f"(maker ${mk['fee'].sum():.4f}, taker ${tk['fee'].sum():.4f})")
    if C.height:
        print(f"Bybit closed P&L: ${C['pnl'].sum():+.4f} over {C.height} closes; wins {(C['pnl'] > 0).sum()} / losses {(C['pnl'] < 0).sum()} "
              f"(win rate {(C['pnl'] > 0).mean():.0%}); avg win ${C.filter(pl.col('pnl') > 0)['pnl'].mean() or 0:+.4f}, avg loss ${C.filter(pl.col('pnl') < 0)['pnl'].mean() or 0:+.4f}")
    if tlog:
        tl = pl.DataFrame([{"type": x["type"], "cash": f(x.get("cashFlow")), "fee": f(x.get("fee")), "change": f(x.get("change")), "funding": f(x.get("funding"))} for x in tlog])
        print("transaction log (wallet change by type):", {k: round(v, 4) for k, v in tl.group_by("type").agg(pl.col("change").sum()).iter_rows()},
              f"| total change ${tl['change'].sum():+.4f}")
    upnl = sum(f(p.get("unrealisedPnl")) for p in pos if f(p.get("size")) > 0)
    print(f"open now: {sum(1 for p in pos if f(p.get('size')) > 0)} positions, ${sum(f(p.get('positionValue')) for p in pos if f(p.get('size')) > 0):.2f} notional, "
          f"unrealised ${upnl:+.4f} -> closed + open = ${(C['pnl'].sum() if C.height else 0) + upnl:+.4f}  (excluded: {', '.join(a.exclude)})")

    # ---- order funnel (mm orders only)
    if O.height:
        M = O.filter(pl.col("link").str.starts_with("mm"))
        st = M.group_by("status").len().sort("len", descending=True)
        print("\norder funnel (mm quotes):", dict(st.iter_rows()), f"| total {M.height}")
        filled = M.filter(pl.col("cum") > 0)
        print(f"  fill rate {filled.height / max(1, M.height):.1%} of quotes; quotes per fill {M.height / max(1, filled.height):.1f}")
        rej = M.filter(pl.col("reject").is_not_null() & (pl.col("reject") != "") & (pl.col("reject") != "EC_NoError"))
        if rej.height:
            print("  rejects:", dict(rej.group_by("reject").len().iter_rows()))
        if filled.height:
            life = (filled["updated"] - filled["created"]) / 1000
            print(f"  time from quote to (last) fill: median {np.median(life.to_numpy()):.0f}s, p25 {np.percentile(life.to_numpy(), 25):.0f}s, p75 {np.percentile(life.to_numpy(), 75):.0f}s")
        red = M.filter(pl.col("reduce") == True)  # noqa: E712
        print(f"  reduce-only quotes {red.height} (filled {red.filter(pl.col('cum') > 0).height}); adding quotes {M.height - red.height} (filled {filled.height - red.filter(pl.col('cum') > 0).height})")

    # ---- spread capture and markouts from flow_3s mids
    lake = Lake()
    syms = E["sym"].unique().to_list()
    fl = lake.read_asof("flow_3s", t1 + timedelta(minutes=10), symbols=syms, start=t0 - timedelta(minutes=2), end=t1 + timedelta(minutes=6))
    fl = fl.filter(pl.col("category") == "linear").select("symbol", "event_ts", "mid").drop_nulls().sort(["symbol", "event_ts"])
    mids = {s: (g["event_ts"].dt.epoch("ms").to_numpy(), g["mid"].to_numpy()) for (s,), g in fl.group_by("symbol")}

    def mid_at(sym, ts_ms):
        if sym not in mids:
            return float("nan")
        tt, mm = mids[sym]
        i = np.searchsorted(tt, ts_ms, side="right") - 1
        return float(mm[i]) if i >= 0 and ts_ms - tt[i] < 15_000 else float("nan")

    rows = []
    for r in mk.iter_rows(named=True):
        sg = 1.0 if r["side"] == "Buy" else -1.0  # buy: good if price rises afterwards
        m0 = mid_at(r["sym"], r["ts"])
        rec = {"sym": r["sym"], "side": r["side"], "val": r["val"], "capture_bp": sg * (m0 - r["px"]) / r["px"] * 1e4 if math.isfinite(m0) else np.nan}
        for h in (10, 60, 300):
            mh = mid_at(r["sym"], r["ts"] + h * 1000)
            rec[f"mo{h}_bp"] = sg * (mh - r["px"]) / r["px"] * 1e4 if math.isfinite(mh) else np.nan
        rows.append(rec)
    Q = pl.DataFrame(rows)
    if Q.height:
        def w(col):
            q = Q.drop_nans(col).drop_nulls(col)
            return (float((q[col] * q["val"]).sum() / q["val"].sum()) if q.height else float("nan")), q.height
        cap, n0 = w("capture_bp")
        print(f"\nmaker fills: spread capture vs mid at fill {cap:+.2f} bp (n={n0}); markouts (value-weighted, + = favourable):", end="")
        for h in (10, 60, 300):
            v, n = w(f"mo{h}_bp")
            print(f"  {h}s {v:+.2f} bp (n={n})", end="")
        print(f"\n  maker fee 2.0 bp/side -> net edge per fill at 60s = capture + markout - fee  (see per-side table)")
        Q = Q.with_columns([pl.col(c).fill_nan(None) for c in ("capture_bp", "mo10_bp", "mo60_bp", "mo300_bp")])
        print(Q.group_by("side").agg(pl.len().alias("fills"), pl.col("val").sum().round(1).alias("usd"), pl.col("capture_bp").median().round(2).alias("cap_med"),
                                      *[pl.col(f"mo{h}_bp").median().round(2).alias(f"mo{h}_med") for h in (10, 60, 300)],
                                      *[(pl.col(f"mo{h}_bp") > 0).mean().round(2).alias(f"mo{h}_pos") for h in (10, 60, 300)]))

    # ---- FIFO round trips per symbol
    trips = []
    book: dict[str, deque] = defaultdict(deque)
    for r in E.iter_rows(named=True):
        q, sgn = r["qty"], (1 if r["side"] == "Buy" else -1)
        dq = book[r["sym"]]
        fee_per = r["fee"] / q if q else 0.0
        while q > 1e-12 and dq and np.sign(dq[0][0]) != sgn:
            lq, lpx, lts, lfee = dq[0]
            m = min(q, abs(lq))
            pnl = (r["px"] - lpx) * m * (-sgn) * -1  # long lot (lq>0) closed by sell: (exit - entry)*m ; short lot closed by buy: (entry - exit)*m
            pnl = (r["px"] - lpx) * m if lq > 0 else (lpx - r["px"]) * m
            trips.append({"sym": r["sym"], "dir": "long" if lq > 0 else "short", "qty": m, "pnl_gross": pnl, "fees": (lfee + fee_per) * m, "hold_s": (r["ts"] - lts) / 1000, "exit_maker": r["maker"]})
            q -= m
            if m >= abs(lq) - 1e-12:
                dq.popleft()
            else:
                dq[0] = (lq - np.sign(lq) * m, lpx, lts, lfee)
        if q > 1e-12:
            dq.append((sgn * q, r["px"], r["ts"], fee_per))
    T = pl.DataFrame(trips)
    if T.height:
        T = T.with_columns((pl.col("pnl_gross") - pl.col("fees")).alias("pnl_net"))
        print(f"\nround trips (FIFO): {T.height}; net ${T['pnl_net'].sum():+.4f} (gross ${T['pnl_gross'].sum():+.4f}, fees ${T['fees'].sum():.4f}); "
              f"win rate {(T['pnl_net'] > 0).mean():.0%}; median hold {np.median(T['hold_s'].to_numpy()) / 60:.1f} min; exits as maker {T['exit_maker'].mean():.0%}")
        print(T.group_by("exit_maker").agg(pl.len().alias("trips"), pl.col("pnl_net").sum().round(4).alias("net_usd"), (pl.col("pnl_net") > 0).mean().round(2).alias("win_rate"),
                                            pl.col("pnl_net").mean().round(4).alias("avg_usd"), (pl.col("hold_s").median() / 60).round(1).alias("median_hold_min")))
        print(T.group_by("dir").agg(pl.len().alias("trips"), pl.col("pnl_net").sum().round(4).alias("net_usd"), (pl.col("pnl_net") > 0).mean().round(2).alias("win_rate"),
                                     (pl.col("hold_s").median() / 60).round(1).alias("median_hold_min")))
        by = T.group_by("sym").agg(pl.len().alias("trips"), pl.col("pnl_net").sum().round(4).alias("net_usd"), (pl.col("pnl_net") > 0).mean().round(2).alias("win_rate"),
                                   (pl.col("hold_s").median() / 60).round(1).alias("hold_min")).sort("net_usd", descending=True)
        pl.Config.set_tbl_rows(14)
        print("best symbols:"); print(by.head(7))
        print("worst symbols:"); print(by.tail(7))
        if a.leaderboard:
            lb = json.loads((ROOT / a.leaderboard).read_text(encoding="utf-8"))
            rows_lb = lb if isinstance(lb, list) else lb.get("rows", lb.get("symbols", []))
            L = pl.DataFrame([{"sym": r.get("symbol") or r.get("sym"), "mr_score": f(r.get("mr_score"))} for r in rows_lb]).join(by, on="sym", how="inner").drop_nans("mr_score")
            if L.height > 3:
                from scipy.stats import spearmanr
                rho, pv = spearmanr(L["mr_score"], L["net_usd"])
                print(f"mr_score (quoter, at run start) vs round-trip net per symbol: Spearman {rho:+.2f} (p={pv:.2f}, n={L.height})")
        open_inv = {s: sum(l[0] for l in dq) for s, dq in book.items() if abs(sum(l[0] for l in dq)) > 1e-12}
        print("open lots (not yet round-tripped):", {k: round(v, 6) for k, v in open_inv.items()})
    # forced flatten window (taker fills) vs quoting
    if tk.height:
        print(f"\ntaker fills (forced flatten / closes): {tk.height}, ${tk['val'].sum():.2f} notional, fees ${tk['fee'].sum():.4f}, first {tk['t'].min():%H:%M:%S} last {tk['t'].max():%H:%M:%S}")
    (out / "summary.json").write_text(json.dumps({"fills": E.height, "maker": mk.height, "taker": tk.height, "turnover": float(E["val"].sum()), "fees": float(E["fee"].sum()),
                                                  "closed_pnl": float(C["pnl"].sum()) if C.height else None, "ledger_cash": cash,
                                                  "round_trip_net": float(T["pnl_net"].sum()) if T.height else None}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
