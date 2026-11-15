"""Desk snapshot orchestration: fetch (public REST + lake + local stores), cut as_of, build every domain, write files.

    research/desk/<UTC date>/<HHMM>/{xs,geometry,news,sentiment,tape,account}.json + <domain>.md digests + index.json
    research/desk/LATEST.json -> the newest snapshot directory

Point in time: all network fetches finish first; as_of = the time the last response was received (the data cut), so
every fetched value has known_ts <= as_of; the domain builders then drop anything after as_of (bars must have closed,
lake rows need known_ts <= as_of, stores need seen_ts <= as_of). Resource limits: <= 2 worker threads, POLARS and
DuckDB at 2 threads, DuckDB memory 512 MB.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import polars as pl  # noqa: E402

from ai_trader.desk import DESK_VERSION  # noqa: E402
from ai_trader.desk import geometry as G  # noqa: E402
from ai_trader.desk import news as N  # noqa: E402
from ai_trader.desk import sentiment as S  # noqa: E402
from ai_trader.desk import tape as T  # noqa: E402
from ai_trader.desk import xs as X  # noqa: E402
from ai_trader.desk.public import Bybit, Deribit, Http, pmap  # noqa: E402

N_XS, N_GEO, N_TAPE, MAX_GEO, MAX_TAPE = 60, 20, 10, 30, 25
DIGEST_MAX = 60
ACCOUNT_FILE = Path("data") / "state" / "live_status.json"


def _lake():
    from ai_trader.storage.lake import Lake

    lake = Lake()
    lake.con.execute("SET threads = 2")
    lake.con.execute("SET memory_limit = '512MB'")
    return lake


def lake_bars_1h(lake, symbols: list[str], as_of: datetime, days: int = 200) -> dict[str, pl.DataFrame]:
    d = lake.read_asof("bars", as_of, symbols=symbols, start=as_of - timedelta(days=days),
                       filters={"venue": "bybit", "price_type": "last", "freq": "1h"},
                       columns=["symbol", "event_ts", "close_ts", "open", "high", "low", "close", "volume", "known_ts"])
    if d.height == 0:
        return {}
    return {s: g.drop("symbol").sort("event_ts") for (s,), g in d.group_by(["symbol"])}


def read_flow_3s(root: Path, symbols: list[str], start: datetime, as_of: datetime, con) -> dict[str, pl.DataFrame]:
    """flow_3s rows of the given symbols with event_ts >= start and known_ts <= as_of (date partitions read directly)."""
    days = sorted({(start + timedelta(days=k)).date() for k in range((as_of.date() - start.date()).days + 1)})
    globs = [(root / "flow_3s" / "venue=bybit" / "category=linear" / f"date={d.isoformat()}").as_posix() + "/*.parquet" for d in days]
    globs = [g for g in globs if Path(g).parent.is_dir() and any(Path(g).parent.glob("*.parquet"))]
    if not globs:
        return {}
    cols = "symbol, event_ts, buy_notional, sell_notional, n_buy, n_sell, spread_bps, depth_bid_25, depth_ask_25, obi_5, known_ts"
    sql = f"""SELECT {cols} FROM (SELECT *, row_number() OVER (PARTITION BY symbol, event_ts ORDER BY known_ts DESC) AS _rn
              FROM read_parquet({globs!r}, union_by_name = true) WHERE known_ts <= ? AND event_ts >= ? AND symbol IN (SELECT unnest(?)))
              WHERE _rn = 1 ORDER BY symbol, event_ts"""
    d = pl.from_arrow(con.sql(sql, params=[as_of, start, list(symbols)]).to_arrow_table())
    return {s: g.drop("symbol") for (s,), g in d.group_by(["symbol"])} if d.height else {}


def read_account(root: Path, as_of: datetime, path: Path | None = None) -> dict:
    """Read-only hook for agent LIVE: `atx live status` writes the JSON file; this never calls an exchange endpoint."""
    p = path or (root / ACCOUNT_FILE)
    base = {"domain": "account", "as_of": as_of.isoformat(), "source": "atx live status (agent LIVE)", "file": str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
            "expected_fields": ["as_of", "environment", "equity_usd", "day_pnl_usd", "trades_opened_today", "open_positions[symbol, side, entry, stop, take_profit, upl_usd]",
                                "open_orders", "limits{max_loss_per_trade_usd, daily_loss_stop_usd, max_entries_per_day, max_open, max_same_side, max_gross_usd}", "remaining{...}", "halted"]}
    if not p.is_file():
        return {**base, "status": "pending agent LIVE (no status file yet)"}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {**base, "status": "unreadable status file"}
    ts = d.get("as_of")
    if not ts or datetime.fromisoformat(ts.replace("Z", "+00:00")) > as_of:
        return {**base, "status": "status file newer than as_of or unstamped; not used"}
    return {**base, "status": "ok", "account": d}


# ---------------------------------------------------------------- digests

def _p(x, nd=1, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _px(x):
    if x is None:
        return "-"
    return f"{x:,.0f}" if x >= 1000 else (f"{x:,.2f}" if x >= 10 else f"{x:.4g}")


def _usd(x):
    if x is None:
        return "-"
    x = float(x)
    return f"{x / 1e6:.1f}M" if abs(x) >= 1e6 else (f"{x / 1e3:.0f}k" if abs(x) >= 1e3 else f"{x:.0f}")


def _hm(ts: str | None) -> str:
    return ts[11:16] if ts else "-"


def digest_xs(x: dict) -> list[str]:
    b = x["basket"]
    toks = x["tokens"]
    L = [f"# XS {x['as_of'][:16]}Z (1h bars to {_hm(x['bar_close'])}) - {x['note']}",
         f"Crypto EW basket n={b['crypto_members']}: " + " | ".join(f"{k} {_p(v, 0)}bp" for k, v in b["crypto_ret_bp"].items())
         + (f"; TradFi EW n={b['tradfi_members']}: " + " | ".join(f"{k} {_p(v, 0)}" for k, v in b["tradfi_ret_bp"].items()) if b["tradfi_members"] else ""),
         "rs = Delta ln(P / EW basket) in bp (sd = vol-standardised); funding ann % (90d pct); OI log %; basis 1e4 ln(mark/index)"]
    cells: dict[str, int] = {}
    for t in toks:
        c = t["regime"].get("cell")
        if c:
            cells[c] = cells.get(c, 0) + 1
    L.append("v0 cells: " + (", ".join(f"{k} x{v}" for k, v in sorted(cells.items(), key=lambda kv: -kv[1])) or "none") + f" (of {len(toks)}; others not in lake / short)")
    hdr = "| sym | sec | cell | rs1h | rs4h | rs24h (sd) | rs7d | fund ann% (pct) | OI24h | basis |"
    crypto = [t for t in toks if t["asset_class"] == "crypto" and t["rs_bp"].get("24h") is not None]
    crypto.sort(key=lambda t: -t["rs_bp"]["24h"])

    def row(t):
        f = t["funding"]
        pct = f"p{100 * f['pct']:.0f}" if f.get("pct") is not None else "-"
        return (f"| {t['symbol'][:-4]} | {t['sector'][:8]} | {t['regime'].get('cell') or '-'} | {_p(t['rs_bp'].get('1h'), 0)} | {_p(t['rs_bp'].get('4h'), 0)} | "
                f"{_p(t['rs_bp'].get('24h'), 0)} ({_p(t['rs_sigma'].get('24h'))}) | {_p(t['rs_bp'].get('7d'), 0)} | {_p(f.get('ann_pct_last'))} ({pct}) | "
                f"{_p(t['oi'].get('chg_pct', {}).get('24h'))} | {_p(t['basis_bp'].get('mark_index'))} |")
    L += ["## Strongest 24h (crypto)", hdr, "|" + "---|" * 10] + [row(t) for t in crypto[:7]]
    L += ["## Weakest 24h (crypto)", hdr, "|" + "---|" * 10] + [row(t) for t in crypto[-7:][::-1]]
    tf = [t for t in toks if t["asset_class"] == "tradfi" and t["rs_bp"].get("24h") is not None]
    if tf:
        tf.sort(key=lambda t: -t["rs_bp"]["24h"])
        L.append("TradFi rs24h vs TradFi basket: " + ", ".join(f"{t['symbol'][:-4]} {_p(t['rs_bp']['24h'], 0)}" for t in tf[:10]))
    fu = [t for t in toks if t["funding"].get("pct") is not None]
    fu.sort(key=lambda t: t["funding"]["last"])
    L.append("## Funding (last settled, ann %; pct of 90d window)")
    L.append("Most negative: " + ", ".join(f"{t['symbol'][:-4]} {_p(t['funding']['ann_pct_last'])}% (p{100 * t['funding']['pct']:.0f})" for t in fu[:6]))
    L.append("Most positive: " + ", ".join(f"{t['symbol'][:-4]} {_p(t['funding']['ann_pct_last'])}% (p{100 * t['funding']['pct']:.0f})" for t in fu[::-1][:6]))
    oi = [t for t in toks if t["oi"].get("chg_pct", {}).get("24h") is not None]
    oi.sort(key=lambda t: -abs(t["oi"]["chg_pct"]["24h"]))
    L.append("OI 24h movers: " + ", ".join(f"{t['symbol'][:-4]} {_p(t['oi']['chg_pct']['24h'])}% (px {_p((t['ret_bp'].get('24h') or 0) / 100)}%)" for t in oi[:8]))
    ba = [t for t in toks if t["basis_bp"].get("mark_index") is not None]
    ba.sort(key=lambda t: -abs(t["basis_bp"]["mark_index"]))
    L.append("Basis extremes (mark vs index bp): " + ", ".join(f"{t['symbol'][:-4]} {_p(t['basis_bp']['mark_index'])}" for t in ba[:6]))
    L.append("## Sectors (mean rs bp 4h / 24h / 7d; funding ann % median; OI24h mean)")
    for s in x["sectors"][:12]:
        r = s["rs_bp_mean"]
        L.append(f"- {s['sector']} (n {s['n']}): {_p(r.get('4h'), 0)} / {_p(r.get('24h'), 0)} / {_p(r.get('7d'), 0)}; fund {_p(s['funding_ann_pct_median'])}; OI {_p(s['oi_24h_pct_mean'])}%")
    v3 = x.get("v3")
    if v3:
        r4 = sorted((t for t in toks if t.get("v3") and t["v3"].get("rank_4h")), key=lambda t: t["v3"]["rank_4h"])
        L.append(f"## v3 context ({v3['id']}, {v3['age_min']:.0f} min old; low-vol ranking, not a signal, R-XS-2/3)")
        L.append("4h rel median top: " + ", ".join(f"{t['symbol'][:-4]} {_p(t['v3']['rel_4h_q50_bp'], 0)}" for t in r4[:5])
                 + " | bottom: " + ", ".join(f"{t['symbol'][:-4]} {_p(t['v3']['rel_4h_q50_bp'], 0)}" for t in r4[-5:]))
    else:
        L.append("## v3 context: no line known at as_of")
    fl = [t for t in toks if t["flags"]]
    L.append(f"## Flags ({len(fl)}; rules {x['flag_rules']}): geometry looks at these too")
    L += [f"- {t['symbol']}: " + "; ".join(t["flags"]) for t in fl[:12]]
    return L[:DIGEST_MAX]


def _lvl(r: dict) -> str:
    return f"{_px(r['price'])} {r['kind']} ({_p(r['dist_pct'], 2)}%, {_p(r['dist_atr_1h'])}A)"


def digest_geometry(g: dict) -> list[str]:
    L = [f"# Geometry {g['as_of'][:16]}Z - {g['note']}",
         "Per token: price | ATR14 1h (and % of price) | 24h POC [VAL-VAH] | 7d POC | est liq clusters (model) ; then nearest levels above / below "
         "(dist log %, A = 1h ATRs)"]
    for t in g["tokens"]:
        if "error" in t:
            L.append(f"- {t['symbol']}: error {t['error'][:80]}")
            continue
        a = t["atr14"]["1h"]
        p24, p7 = (t["profile"].get("24h") or {}), (t["profile"].get("7d") or {})
        lm = t.get("liq_map", {})
        lq = ""
        if lm.get("status") == "ok":
            lb, sa = lm["long_liq_below"][:1], lm["short_liq_above"][:1]
            lq = " | est liq: " + (f"L {_px(lb[0]['price'])} ${_usd(lb[0]['est_usd'])}" if lb else "L -") + " / " + (f"S {_px(sa[0]['price'])} ${_usd(sa[0]['est_usd'])}" if sa else "S -")
        why = f" [{'; '.join(t['why'])}]" if t.get("why") else ""
        L.append(f"**{t['symbol'][:-4]}** {_px(t['price'])} | ATR1h {_px(a)} ({_p(100 * a / t['price'] if a else None, 2, False)}%) | "
                 f"24h POC {_px(p24.get('poc'))} [{_px(p24.get('val'))}-{_px(p24.get('vah'))}] | 7d POC {_px(p7.get('poc'))}{lq}{why}")
        n = t["nearest"]
        L.append("  up: " + "; ".join(_lvl(r) for r in n["above"][:3]) + " || dn: " + "; ".join(_lvl(r) for r in n["below"][:3]))
    f = g.get("btc_field")
    if f:
        h1, h4 = f["horizons"].get("1h", {}), f["horizons"].get("4h", {})
        L.append(f"BTC field ({f['line']}, cell {f['cell']}): touch 1h +-1% {h1.get('p_touch', {}).get('+1%', 0):.0%}/{h1.get('p_touch', {}).get('-1%', 0):.0%}, "
                 f"4h +-1% {h4.get('p_touch', {}).get('+1%', 0):.0%}/{h4.get('p_touch', {}).get('-1%', 0):.0%}, +-2% {h4.get('p_touch', {}).get('+2%', 0):.0%}/"
                 f"{h4.get('p_touch', {}).get('-2%', 0):.0%} (R-FLD-3: not direction)")
    return L[:DIGEST_MAX]


def digest_tape(t: dict) -> list[str]:
    L = [f"# Tape {t['as_of'][:16]}Z - {t['note']}",
         "| sym | last | 1m | 5m | 15m | 1h | sprd bp | depth25 bid/ask | liq 1h L/S | liq 24h L/S | flow 15m imb (cov) | players |",
         "|" + "---|" * 12]
    for r in t["tokens"]:
        m = r["moves"].get("pct", {})
        b = r["book"]
        d25 = (b.get("depth_usd") or {}).get("25bp", {})
        lq = r["liq"]
        fw = (r["flow"].get("windows") or {}).get("15m", {})
        cl = " CLUSTER" if (lq.get("cluster_60m") or {}).get("fired") else ""
        pl_ = (r.get("players") or {}).get("pattern", "-")
        L.append(f"| {r['symbol'][:-4]} | {_px(r['price']['last'])} | {_p(m.get('1m'), 2)} | {_p(m.get('5m'), 2)} | {_p(m.get('15m'), 2)} | {_p(m.get('60m'), 2)} | "
                 f"{_p(b.get('spread_bp'), 1, False)} | {_usd(d25.get('bid'))}/{_usd(d25.get('ask'))}{'*' if d25.get('truncated') else ''} | "
                 f"{_usd(lq.get('1h', {}).get('long_usd'))}/{_usd(lq.get('1h', {}).get('short_usd'))}{cl} | {_usd(lq.get('24h', {}).get('long_usd'))}/{_usd(lq.get('24h', {}).get('short_usd'))} | "
                 f"{_p(fw.get('imbalance'), 2)} ({_p(fw.get('coverage'), 2, False)}) | {pl_[:70]} |")
    L.append("moves = 100 ln(last / close m minutes ago); * = book received did not reach 25 bp; imb = (buy - sell)/(buy + sell) taker notional (flow_3s);"
             " liq L/S = liquidated longs / shorts USD; R-FLOW-2: never fade a CLUSTER within 60 min")
    return L[:DIGEST_MAX]


def digest_news(n: dict) -> list[str]:
    c = n["collector"]
    L = [f"# News {n['as_of'][:16]}Z - {n['note']}",
         f"Collector: last item seen {c['last_item_seen'] or 'never'}{' (STALE: start scripts/desk_collect.py)' if c['stale'] else ''}; "
         f"24h items {n['counts_24h']['items']} by feed {n['counts_24h']['by_feed']}",
         "Tickers in 24h headlines: " + ", ".join(f"{k} {v}" for k, v in list(n["counts_24h"]["by_ticker"].items())[:12])]
    L.append("## Latest headlines (seen UTC, feed, tone, tickers)")
    for r in n["items"][:15]:
        L.append(f"- {_hm(r['seen_ts'])} {r['feed']} {_p(r.get('tone'), 2)} {','.join(r.get('tickers') or []) or '-'}: {(r.get('title') or '')[:110]}")
    L.append("## Scheduled next 24h (calendar / events table)")
    L += [f"- {e['ts'][5:16]} [{e.get('importance', e.get('kind', ''))}] {e['name']}" for e in n["scheduled_24h"][:8]] or ["- none known (calendar files: " + ", ".join(n["calendar_coverage"]) + ")"]
    L.append("## Recurring next 24h")
    L += [f"- {e['ts'][5:16]} {e['name']}" for e in n["recurring_24h"][:9]]
    o = n.get("options", {})
    L.append("## Deribit options (OI by expiry; max pain; put/call OI)")
    for cur, v in (o.items() if isinstance(o, dict) and o.get("status") is None else []):
        for e in (v.get("expiries") or [])[:2]:
            L.append(f"- {cur} {e['expiry'][:10]} ({e['hours_to']:.0f}h): OI ${_usd(e['oi_usd'])}, P/C {e['put_call_oi']}, max pain {_px(e['max_pain'])}, "
                     f"top C {[x['strike'] for x in e['top_call_strikes']]} P {[x['strike'] for x in e['top_put_strikes']]}")
    fg, al = n["fear_greed"], n["alt_attention"]
    L.append(f"Fear & Greed: {fg.get('value', '-')} ({fg.get('day', '-')}; prev {fg.get('prev_7', [])}) | attention: "
             + (", ".join(f"{k} {al.get(k)}" for k in ("wp_btc", "wp_ind", "hn_story", "hn_comment")) if al.get("status") == "ok" else al.get("status", "-")))
    return L[:DIGEST_MAX]


def digest_sentiment(s: dict) -> list[str]:
    L = [f"# Sentiment {s['as_of'][:16]}Z - {s['reddit']} - {s['note']}",
         f"Keys: {s['keys']}; collected since {s['collected_since'] or 'never'}; last collect {s['last_collect'] or 'never'}; items {s['items_known']} (reddit {s['reddit_items_known']})",
         f"Method: {s['method']['z']}; tone {s['method']['tone']}",
         f"Rising: {', '.join(s['rising']) or '-'} | Falling: {', '.join(s['falling']) or '-'}",
         "| ticker | mentions 24h | reddit | news | tone 24h | /h recent | z mentions | z tone | trend | baseline h |", "|" + "---|" * 10]
    for r in s["tickers"][:20]:
        if r["mentions_24h"] == 0:
            continue
        L.append(f"| {r['ticker']} | {r['mentions_24h']} | {r['reddit_24h']} | {r['news_24h']} | {_p(r['tone_24h'], 2)} | {r['mentions_recent_per_h'] if r['mentions_recent_per_h'] is not None else '-'} | "
                 f"{_p(r['z_mentions'])} | {_p(r['z_tone'])} | {r['trend']} | {r['baseline_hours']} |")
    return L[:DIGEST_MAX]


def digest_account(a: dict) -> list[str]:
    L = [f"# Account {a['as_of'][:16]}Z - {a['status']}", f"Source: {a['source']} via {a['file']} (read-only; no exchange call from the desk layer)"]
    acc = a.get("account")
    if acc:
        L.append(f"equity {acc.get('equity_usd')} | day pnl {acc.get('day_pnl_usd')} | trades today {acc.get('trades_opened_today')} | halted {acc.get('halted')}")
        for p in (acc.get("open_positions") or [])[:5]:
            L.append(f"- {p.get('symbol')} {p.get('side')} entry {p.get('entry')} stop {p.get('stop')} tp {p.get('take_profit')} upl {p.get('upl_usd')}")
        L.append(f"remaining: {acc.get('remaining')}")
    return L


# ---------------------------------------------------------------- run

def _vec(results: list, items: list[str], what: str, errors: list) -> dict:
    out = {}
    for s, r in zip(items, results, strict=True):
        if isinstance(r, Exception):
            errors.append(f"{what} {s}: {type(r).__name__}: {str(r)[:100]}")
        else:
            out[s] = r
    return out


def run(root: Path | None = None, *, out_root: Path | None = None, watch: list[str] | None = None, n_xs: int = N_XS, n_geo: int = N_GEO,
        n_tape: int = N_TAPE, http: Http | None = None, write: bool = True, log=print) -> dict:
    from ai_trader.core.config import load

    root = root or load().root
    t0 = time.monotonic()
    started = datetime.now(UTC)
    timings, errors = {}, []
    http = http or Http()
    by = Bybit(http)
    lake = _lake()

    def lap(k, t=[t0]):  # noqa: B006
        timings[k] = round(time.monotonic() - t[0], 2)
        t[0] = time.monotonic()

    tickers, tk_known = by.tickers()
    instruments = by.instruments()
    universe = X.select_universe(tickers, instruments, n_xs)
    syms = [u["symbol"] for u in universe]
    lap("universe")
    b1h = _vec(pmap(lambda s: by.kline(s, "1h", 400), syms), syms, "kline1h", errors)
    fund = _vec(pmap(lambda s: by.funding_history(s, 200), syms), syms, "funding", errors)
    oi = _vec(pmap(lambda s: by.open_interest(s, "1h", 200), syms), syms, "oi", errors)
    lap("xs_rest")
    lb = lake_bars_1h(lake, syms, datetime.now(UTC))
    regimes = X.regime_cells(lb, datetime.now(UTC))
    lap("regime")
    v3 = X.v3_context(X.read_v3_lines(root / "data" / "journal" / "live_xs_v3", datetime.now(UTC)), datetime.now(UTC))
    ysec = X.load_sectors(root)
    pre = X.build_xs(datetime.now(UTC), universe, b1h, fund, oi, ticker_known=tk_known, regimes=regimes, v3=v3, yaml_sectors=ysec)
    flagged = [t["symbol"] for t in pre["tokens"] if t["flags"]]
    watch = [w.upper() for w in (watch or []) if w.upper() in syms]
    reasons: dict[str, list[str]] = {}
    for i, s in enumerate(syms):
        if i < n_geo:
            reasons.setdefault(s, []).append(f"top{n_geo} turnover")
    for t in pre["tokens"]:
        if t["flags"]:
            reasons.setdefault(t["symbol"], []).extend(t["flags"])
    for w in watch:
        reasons.setdefault(w, []).append("watch")
    acc_pre = read_account(root, datetime.now(UTC))
    pos = [p.get("symbol") for p in ((acc_pre.get("account") or {}).get("open_positions") or []) if p.get("symbol") in syms]
    for p in pos:
        reasons.setdefault(p, []).append("open position")
    geo_syms = list(dict.fromkeys(syms[:n_geo] + flagged + watch + pos))[:MAX_GEO]
    tape_syms = list(dict.fromkeys(syms[:n_tape] + watch + pos + flagged))[:MAX_TAPE]
    lap("select")

    def geo_fetch(s):
        return {"1m": by.kline_n(s, "1m", 1500), "15m": by.kline(s, "15m", 1000), "4h": by.kline(s, "4h", 500), "1d": by.kline(s, "1d", 400)}
    gb = _vec(pmap(geo_fetch, geo_syms), geo_syms, "geometry bars", errors)
    lap("geo_rest")
    books = _vec(pmap(lambda s: by.orderbook(s, 200), tape_syms), tape_syms, "orderbook", errors)
    need1m = [s for s in tape_syms if s not in gb]
    k1m = {s: gb[s]["1m"] for s in tape_syms if s in gb}
    k1m.update(_vec(pmap(lambda s: by.kline(s, "1m", 70), need1m), need1m, "kline1m", errors))
    der = Deribit(http)
    opts = {}
    for cur in ("BTC", "ETH"):
        try:
            opts[cur] = der.option_book(cur)
        except Exception as e:  # noqa: BLE001
            errors.append(f"deribit {cur}: {e}"[:160])
    lap("tape_rest")

    as_of = max(http.last_known or started, datetime.now(UTC))      # the data cut: every fetch above is <= as_of
    liq = {}
    try:
        L = lake.read_asof("liquidations", as_of, symbols=tape_syms, start=as_of - timedelta(hours=24))
        liq = {s: g for (s,), g in L.group_by(["symbol"])} if L.height else {}
    except Exception as e:  # noqa: BLE001
        errors.append(f"lake liquidations: {e}"[:160])
    try:
        flow = read_flow_3s(lake.root, tape_syms, as_of - timedelta(minutes=61), as_of, lake.con)
    except Exception as e:  # noqa: BLE001
        flow = {}
        errors.append(f"lake flow_3s: {e}"[:160])
    players = {}
    for s in tape_syms:
        if s in lb:
            try:
                players[s] = T.player_read(s, as_of, lake, regimes.get(s))
            except Exception as e:  # noqa: BLE001
                errors.append(f"players {s}: {e}"[:160])
    lap("lake_tape")

    tk = {u["symbol"]: u["ticker"] for u in universe}
    xs = X.build_xs(as_of, universe, b1h, fund, oi, ticker_known=tk_known, regimes=regimes, v3=v3, yaml_sectors=ysec)
    lap("xs")
    gtoks = {}
    for s in geo_syms:
        if s in gb:
            last = float(tk[s]["lastPrice"])
            gtoks[s] = {"price": last, "bars": {**gb[s], "1h": b1h.get(s)}, "oi": oi.get(s)}
    field = G.field_card(root / "data" / "journal" / "live_nn", as_of)
    geo = G.build_geometry(as_of, gtoks, field=field, reasons=reasons)
    lap("geometry")
    tape = T.build_tape(as_of, {s: {"ticker": tk[s], "ticker_known": tk_known, "k1m": k1m.get(s), "book": books.get(s), "liq": liq.get(s),
                                    "flow": flow.get(s), "players": players.get(s)} for s in tape_syms}, reasons=reasons)
    lap("tape")
    # funding settlements in the window: next + k * interval per perp
    ft = []
    for t in xs["tokens"]:
        if t.get("next_funding_ts") and t["funding"].get("interval_h"):
            nx = datetime.fromisoformat(t["next_funding_ts"])
            for k in range(0, 25):
                tt = nx + timedelta(hours=k * t["funding"]["interval_h"])
                if tt > as_of + timedelta(hours=24):
                    break
                ft.append(tt.isoformat())
    fcount: dict[str, int] = {}
    for x in ft:
        fcount[x] = fcount.get(x, 0) + 1
    options = {cur: N.option_expiries(b, as_of, kt) for cur, (b, kt) in opts.items()}
    store = S.store_root(root)
    news = N.build_news(as_of, store, root=root, lake=lake, funding_times=sorted(fcount.items()), options=options)
    bases = {X.base_of(s): s for s in syms}
    sent = S.build_sentiment(as_of, store, bases)
    acct = read_account(root, as_of)
    lap("news_sentiment")
    domains = {"xs": xs, "geometry": geo, "news": news, "sentiment": sent, "tape": tape, "account": acct}
    digests = {"xs": digest_xs(xs), "geometry": digest_geometry(geo), "news": digest_news(news), "sentiment": digest_sentiment(sent),
               "tape": digest_tape(tape), "account": digest_account(acct)}
    out_root = out_root or (root / "research" / "desk")
    snap_dir = out_root / as_of.strftime("%Y-%m-%d") / as_of.strftime("%H%M")
    index = {"desk_version": DESK_VERSION, "as_of": as_of.isoformat(), "started": started.isoformat(), "snapshot": snap_dir.relative_to(root).as_posix() if snap_dir.is_relative_to(root) else str(snap_dir),
             "universe": len(syms), "geometry_tokens": geo_syms, "tape_tokens": tape_syms, "http_calls": http.calls,
             "runtime_s": round(time.monotonic() - t0, 1), "timings_s": timings, "errors": errors,
             "point_in_time": "every domain built at as_of = the data cut after the last network response"}
    if write:
        snap_dir.mkdir(parents=True, exist_ok=True)
        for k, v in domains.items():
            (snap_dir / f"{k}.json").write_text(json.dumps(v, default=str, indent=1), encoding="utf-8")
            (snap_dir / f"{k}.md").write_text("\n".join(digests[k]) + "\n", encoding="utf-8")
        index["runtime_s"] = round(time.monotonic() - t0, 1)
        (snap_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        (out_root / "LATEST.json").write_text(json.dumps({"snapshot": index["snapshot"], "as_of": index["as_of"]}, indent=1), encoding="utf-8")
    log(f"desk snapshot {index['snapshot']} as_of {as_of:%H:%M:%S}Z in {index['runtime_s']}s, {http.calls} calls, {len(errors)} errors")
    return {"index": index, "domains": domains, "digests": digests, "dir": snap_dir}


__all__ = ["digest_geometry", "digest_news", "digest_sentiment", "digest_tape", "digest_xs", "read_account", "read_flow_3s", "run"]
