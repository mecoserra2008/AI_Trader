"""Desk data layer: no lookahead per domain, log-ratio relative strength, level maths, the Reddit collector on canned
JSON, and the report / proposal contract validator. No network: every client is fed canned responses."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.desk import contract as C
from ai_trader.desk import geometry as G
from ai_trader.desk import news as N
from ai_trader.desk import sentiment as S
from ai_trader.desk import tape as T
from ai_trader.desk import xs as X
from ai_trader.desk.public import (
    Bybit,
    Http,
    PublicError,
    closed,
    forming_open,
    kline_frame,
    resample,
)
from ai_trader.desk.text import TickerIndex, tone

AS_OF = datetime(2026, 11, 15, 16, 30, 5, tzinfo=UTC)
US = pl.Datetime("us", "UTC")


def bars(n: int, *, tf_h: float = 1.0, end: datetime = AS_OF, seed: int = 0, start_px: float = 100.0, vol: float = 0.01, forming: bool = True) -> pl.DataFrame:
    """n bars of tf_h hours whose last CLOSED bar closes at the last tf boundary <= end; plus one forming bar if asked."""
    rng = np.random.default_rng(seed)
    step = timedelta(hours=tf_h)
    last_close = datetime.fromtimestamp((int(end.timestamp()) // int(step.total_seconds())) * step.total_seconds(), UTC)
    ev = [last_close - step * (n - i) for i in range(n + (1 if forming else 0))]
    c = start_px * np.exp(np.cumsum(rng.standard_t(5, len(ev)) * vol))
    o = np.r_[start_px, c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, vol, len(ev)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, vol, len(ev)))
    f = pl.DataFrame({"event_ts": ev, "open": o, "high": h, "low": lo, "close": c, "volume": rng.uniform(10, 100, len(ev)),
                      "turnover": c * 50}).with_columns(pl.col("event_ts").cast(US))
    return f.with_columns((pl.col("event_ts") + pl.duration(seconds=int(step.total_seconds()))).alias("close_ts"),
                          (pl.col("event_ts") + pl.duration(seconds=int(step.total_seconds()) + 2)).alias("known_ts"))


def future_rows(f: pl.DataFrame, k: int = 5, factor: float = 3.0) -> pl.DataFrame:
    """Append k bars entirely after the frame's end with absurd prices (they must never change anything)."""
    step = f["close_ts"][0] - f["event_ts"][0]
    last = f.tail(1)
    rows = []
    for i in range(1, k + 1):
        ev = last["event_ts"][0] + step * i
        rows.append({**last.row(0, named=True), "event_ts": ev, "close_ts": ev + step, "known_ts": ev + step,
                     "open": last["close"][0] * factor, "high": last["close"][0] * factor * 1.1, "low": last["close"][0] * factor * 0.9,
                     "close": last["close"][0] * factor})
    return pl.concat([f, pl.DataFrame(rows, schema=f.schema)])


def same(a, b) -> bool:
    return json.dumps(a, default=str, sort_keys=True) == json.dumps(b, default=str, sort_keys=True)


# ---------------------------------------------------------------- public helpers

def test_whitelist_refuses_non_market_paths():
    by = Bybit(Http(get_raw=lambda u: b'{"retCode":0,"result":{}}'))
    for path in ("/v5/order/create", "/v5/position/list", "/v5/account/wallet-balance"):
        with pytest.raises(PublicError):
            by.get(path, category="linear")


def test_kline_frame_and_closed_split():
    t0 = int(datetime(2026, 11, 15, 15, tzinfo=UTC).timestamp() * 1000)
    rows = [[str(t0 + 3_600_000), "2", "3", "1", "2.5", "10", "25"], [str(t0), "1", "2", "0.5", "2", "10", "20"]]   # newest first
    f = kline_frame(rows, "1h", AS_OF)
    assert f["event_ts"].is_sorted() and f["close_ts"][0] == datetime(2026, 11, 15, 16, tzinfo=UTC)
    c = closed(f, AS_OF)
    assert c.height == 1 and c["close"][0] == 2.0               # the 16:00-17:00 bar is still forming at 16:30
    assert forming_open(f, AS_OF) == 2.0


def test_resample_matches_aggregation_and_keeps_forming_out():
    t = AS_OF + timedelta(hours=1)                            # 1h bars closed through 17:00
    b = bars(100, end=t, forming=False)
    r = resample(b, "4h")
    c = closed(r, t)
    g = c.row(5, named=True)
    sub = b.filter((pl.col("event_ts") >= g["event_ts"]) & (pl.col("event_ts") < g["close_ts"]))
    assert sub.height == 4 and g["open"] == sub["open"][0] and g["close"] == sub["close"][-1]
    assert g["high"] == sub["high"].max() and g["low"] == sub["low"].min()
    assert (c["close_ts"] <= t).all() and r.height == c.height + 1      # the 16:00-20:00 group (one 1h bar so far) is forming
    assert forming_open(r, t) == b.filter(pl.col("event_ts") == datetime(2026, 11, 15, 16, tzinfo=UTC))["open"][0]


# ---------------------------------------------------------------- relative strength (log price ratios)

def test_relative_strength_is_log_ratio_vs_geometric_basket():
    rng = np.random.default_rng(3)
    n = 400
    closes = {s: 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for i, s in enumerate(("AUSDT", "BUSDT", "CUSDT", "DUSDT"))}
    out = X.relative_strength(closes, set(closes))
    for name, h in X.HORIZONS.items():
        r = {s: math.log(c[-1] / c[-1 - h]) for s, c in closes.items()}
        b = np.mean(list(r.values()))
        # geometric EW basket index B_t = exp(mean_j ln(C_j,t / C_j,T-h)); rs = Delta ln(P_i / B)
        B_T = math.exp(np.mean([math.log(c[-1] / c[-1 - h]) for c in closes.values()]))
        for s, c in closes.items():
            assert out["tokens"][s]["rs"][name] == pytest.approx(r[s] - b, abs=1e-12)
            assert out["tokens"][s]["rs"][name] == pytest.approx(math.log((c[-1] / c[-1 - h]) / B_T), abs=1e-12)
        assert sum(out["tokens"][s]["rs"][name] for s in closes) == pytest.approx(0.0, abs=1e-12)
        assert out["basket_ret"][name] == pytest.approx(b, abs=1e-12)
    # scale invariance: a token quoted 1000x (1000PEPE style) has the same rs
    scaled = {**closes, "AUSDT": closes["AUSDT"] * 1000}
    o2 = X.relative_strength(scaled, set(scaled))
    assert o2["tokens"]["AUSDT"]["rs"]["24h"] == pytest.approx(out["tokens"]["AUSDT"]["rs"]["24h"], abs=1e-12)
    # vol standardisation: rs_sigma = rs / (sd of hourly relative log returns * sqrt(h))
    s = "BUSDT"
    L = np.log(np.vstack(list(closes.values())))
    dr = np.diff(L[:, -169:], axis=1)
    rel = dr[1] - dr.mean(axis=0)
    assert out["tokens"][s]["rs_sigma"]["4h"] == pytest.approx(out["tokens"][s]["rs"]["4h"] / (np.std(rel, ddof=1) * 2), rel=1e-9)


def test_non_members_are_ranked_against_the_basket_without_moving_it():
    rng = np.random.default_rng(4)
    c = {s: 10 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))) for s in ("AUSDT", "BUSDT", "XUSDT")}
    o = X.relative_strength(c, {"AUSDT", "BUSDT"})
    o2 = X.relative_strength({k: v for k, v in c.items() if k != "XUSDT"}, {"AUSDT", "BUSDT"})
    assert o["basket_ret"]["24h"] == pytest.approx(o2["basket_ret"]["24h"])


def test_funding_oi_basis_definitions():
    ev = [AS_OF - timedelta(hours=8 * k) for k in range(30, 0, -1)]
    rates = np.linspace(-1e-4, 2e-4, 30)
    f = pl.DataFrame({"event_ts": ev, "rate": rates, "known_ts": ev}).with_columns(pl.col("event_ts", "known_ts").cast(US))
    st = X.funding_stats(f, AS_OF, predicted=1e-4, interval_h=8)
    assert st["last"] == pytest.approx(2e-4) and st["pct"] == pytest.approx((29 + 0.5) / 30, abs=1e-3)
    assert st["ann_pct_last"] == pytest.approx(100 * 2e-4 * 3 * 365, abs=0.1)
    oi = pl.DataFrame({"event_ts": [AS_OF - timedelta(hours=k) for k in range(30, -1, -1)], "oi": np.r_[np.full(30, 100.0), 110.0],
                       "known_ts": [AS_OF - timedelta(hours=k) for k in range(30, -1, -1)]}).with_columns(pl.col("event_ts", "known_ts").cast(US))
    o = X.oi_change(oi, AS_OF)
    assert o["chg_pct"]["24h"] == pytest.approx(100 * math.log(1.1), abs=0.006)
    b = X.basis_bp(100.0, 100.1, 100.0)
    assert b["mark_index"] == pytest.approx(1e4 * math.log(100.1 / 100), abs=0.05)


def _universe(syms):
    out = []
    for i, s in enumerate(syms):
        out.append({"symbol": s, "turnover_24h_usd": 1e9 / (i + 1),
                    "ticker": {"symbol": s, "lastPrice": "100", "markPrice": "100.05", "indexPrice": "100", "fundingRate": "0.0001",
                               "fundingIntervalHour": "8", "nextFundingTime": str(int((AS_OF + timedelta(hours=1)).timestamp() * 1000)), "turnover24h": "1"},
                    "instrument": {"symbol": s, "symbolType": "", "contractType": "LinearPerpetual", "status": "Trading", "quoteCoin": "USDT"}})
    return out


def test_xs_no_lookahead():
    syms = ["AUSDT", "BUSDT", "CUSDT"]
    b1h = {s: bars(400, seed=i) for i, s in enumerate(syms)}
    ev = [AS_OF - timedelta(hours=8 * k) for k in range(60, 0, -1)]
    fund = {s: pl.DataFrame({"event_ts": ev, "rate": np.full(60, 1e-4 * (i + 1)), "known_ts": ev}).with_columns(pl.col("event_ts", "known_ts").cast(US))
            for i, s in enumerate(syms)}
    oe = [AS_OF - timedelta(hours=k) for k in range(199, -1, -1)]
    oi = {s: pl.DataFrame({"event_ts": oe, "oi": np.linspace(100, 120, 200), "known_ts": oe}).with_columns(pl.col("event_ts", "known_ts").cast(US)) for s in syms}
    u = _universe(syms)
    a = X.build_xs(AS_OF, u, b1h, fund, oi, ticker_known=AS_OF - timedelta(seconds=1))
    fut = AS_OF + timedelta(hours=1)
    b1h_f = {s: future_rows(b) for s, b in b1h.items()}
    fund_f = {s: pl.concat([f, pl.DataFrame({"event_ts": [fut], "rate": [0.05], "known_ts": [fut]}).with_columns(pl.col("event_ts", "known_ts").cast(US))]) for s, f in fund.items()}
    oi_f = {s: pl.concat([o, pl.DataFrame({"event_ts": [fut], "oi": [1e9], "known_ts": [fut]}).with_columns(pl.col("event_ts", "known_ts").cast(US))]) for s, o in oi.items()}
    late = {s: o.with_columns(pl.when(pl.col("event_ts") == pl.col("event_ts").max()).then(AS_OF + timedelta(seconds=30)).otherwise(pl.col("known_ts")).alias("known_ts"))
            for s, o in oi.items()}   # a snapshot stamped before as_of but RECEIVED after it must also be ignored
    b = X.build_xs(AS_OF, u, b1h_f, fund_f, oi_f, ticker_known=AS_OF - timedelta(seconds=1))
    assert same(a, b)
    c = X.build_xs(AS_OF, u, b1h, fund, late, ticker_known=AS_OF - timedelta(seconds=1))
    assert c["tokens"][0]["oi"]["ts"] < a["tokens"][0]["oi"]["ts"]
    d = X.build_xs(AS_OF, u, b1h, fund, oi, ticker_known=AS_OF + timedelta(seconds=1))      # ticker fetched after the cut
    assert d["tokens"][0]["last"] is None and d["tokens"][0]["basis_bp"]["mark_index"] is None


def test_enrich_min_matches_research_frame():
    from ai_trader.research.frame import enrich_bars

    b = bars(600, seed=9)
    a = X.enrich_min(b)
    r = enrich_bars(b)
    for c in ("ret", "sigma_prev", "sigma_next"):
        x, y = a[c].to_numpy(), r[c].to_numpy()
        assert np.allclose(np.nan_to_num(x, nan=-1), np.nan_to_num(y, nan=-1), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------- levels on synthetic bars

def _frame(o, h, lo, c, tf_h=1.0, start=datetime(2026, 11, 1, tzinfo=UTC), v=None):
    n = len(c)
    ev = [start + timedelta(hours=tf_h * i) for i in range(n)]
    return pl.DataFrame({"event_ts": ev, "open": o, "high": h, "low": lo, "close": c, "volume": v if v is not None else np.ones(n)}).with_columns(
        pl.col("event_ts").cast(US)).with_columns((pl.col("event_ts") + pl.duration(seconds=int(tf_h * 3600))).alias("close_ts"),
                                                  (pl.col("event_ts") + pl.duration(seconds=int(tf_h * 3600))).alias("known_ts"))


def test_swings_confirmation_and_sweep():
    hi = np.array([10, 11, 12, 15, 12, 11, 10, 11, 12, 13, 16, 13, 12, 11], dtype=float)
    lo = hi - 1
    f = _frame(hi - 0.5, hi, lo, hi - 0.5)
    end_of = f["close_ts"].to_list()
    s = G.swings(f, end_of[5], k=3)                         # pivot at i=3 needs bars 4..6: not yet known at bar 5's close
    assert not s["highs"]
    s = G.swings(f, end_of[6], k=3)
    assert [p["price"] for p in s["highs"]] == [15.0] and not s["highs"][0]["swept"]
    s = G.swings(f, end_of[13], k=3)
    assert [p["price"] for p in s["highs"]] == [15.0, 16.0]
    assert s["highs"][0]["swept"] and not s["highs"][1]["swept"]
    assert s["highs"][0]["known_ts"] == end_of[6].isoformat()
    lows = s["lows"]
    assert [p["price"] for p in lows] == [9.0]               # the low at i=6 (10 - 1)


def test_period_levels_prior_day_week_month():
    as_of = datetime(2026, 11, 18, 10, tzinfo=UTC)          # Wednesday
    days = [datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(49)]   # to 2026-11-18 (forming)
    o = np.arange(49, dtype=float) + 100
    f = pl.DataFrame({"event_ts": days, "open": o, "high": o + 5, "low": o - 5, "close": o + 1, "volume": 1.0}).with_columns(
        pl.col("event_ts").cast(US)).with_columns((pl.col("event_ts") + pl.duration(days=1)).alias("close_ts"), pl.col("event_ts").alias("known_ts"))
    p = G.period_levels(f, as_of)
    assert p["day"]["open"] == 148.0 and p["day"]["prior"]["high"] == 152.0          # 11-17: open 147, high 152
    assert p["week"]["open"] == 146.0                                                 # Monday 11-16
    pw = p["week"]["prior"]                                                           # 11-09 .. 11-15
    assert pw["open"] == 139.0 and pw["high"] == 150.0 and pw["low"] == 134.0 and pw["close"] == 146.0 and pw["bars"] == 7
    pm = p["month"]["prior"]
    assert pm["open"] == 100.0 and pm["high"] == 135.0 and pm["bars"] == 31 and p["month"]["open"] == 131.0
    # a forming bar's high / low never leak into the prior periods
    f2 = f.with_columns(pl.when(pl.col("event_ts") == pl.col("event_ts").max()).then(1e6).otherwise(pl.col("high")).alias("high"))
    assert same(G.period_levels(f2, as_of), p)


def test_volume_profile_poc_and_value_area():
    n = 200
    c = np.r_[np.full(150, 100.0), np.linspace(90, 110, 50)]
    f = _frame(c, c * 1.0002, c * 0.9998, c, tf_h=1 / 60, start=AS_OF - timedelta(minutes=n + 5))
    p = G.volume_profile(f, AS_OF, timedelta(hours=24))
    assert abs(p["poc"] - 100.0) / 100 < 0.0006 and p["val"] < 100 < p["vah"] and p["va_share"] >= 0.7


def test_leg_levels_from_vecopen():
    pytest.importorskip("vecopen")
    rng = np.random.default_rng(1)
    n = 60
    c = 100 + np.cumsum(rng.normal(0, 0.1, n))
    o = np.r_[100.0, c[:-1]]
    h, lo = np.maximum(o, c) + 0.05, np.minimum(o, c) - 0.05
    i = 40                                                    # a clean bullish candle opening at its low, then a run up
    o[i], lo[i], c[i], h[i] = c[i - 1], c[i - 1], c[i - 1] + 3.0, c[i - 1] + 3.05
    for k in range(i + 1, n):
        o[k] = c[k - 1]
        c[k] = o[k] + 0.05
        h[k], lo[k] = c[k] + 0.02, o[k] - 0.02
    f = _frame(o, h, lo, c)
    out = G.leg_levels(f, "1h", f["close_ts"][-1])
    assert out["status"] == "ok"
    prices = [x["price"] for x in out["levels"]]
    assert any(abs(p - o[i]) < 1e-9 for p in prices)
    lv = next(x for x in out["levels"] if abs(x["price"] - o[i]) < 1e-9)
    assert lv["direction"].startswith("up-leg") and lv["untouched"]
    early = G.leg_levels(f, "1h", f["close_ts"][i - 1])       # before the origin candle closed: the level cannot exist
    assert not any(abs(x["price"] - o[i]) < 1e-9 for x in early.get("levels", []))


def test_round_levels_and_distance():
    r = G.round_levels(64676.0)
    assert r["major"]["below"] == 60000 and r["major"]["above"] == 65000 and r["minor"]["below"] == 64000 and r["minor"]["above"] == 65000
    d = G.dist(65000, 64676.0, 400.0, 2000.0)
    assert d["dist_pct"] == pytest.approx(100 * math.log(65000 / 64676), abs=1e-3) and d["dist_atr_1h"] == pytest.approx(0.81, abs=0.01)


def test_liq_map_removes_traded_through_positions():
    n = 48
    c = np.full(n, 100.0)
    f = _frame(c, c * 1.001, c * 0.999, c, start=AS_OF - timedelta(hours=n + 1))
    oe = f["event_ts"].to_list() + [f["event_ts"][-1] + timedelta(hours=1)]
    oi = pl.DataFrame({"event_ts": oe, "oi": np.r_[1000.0, np.full(n, 2000.0)], "known_ts": oe}).with_columns(pl.col("event_ts", "known_ts").cast(US))
    m = G.liq_map(f, oi, AS_OF, 100.0)
    assert m["status"] == "ok" and m["long_liq_below"] and m["short_liq_above"]
    assert all(x["price"] < 100 for x in m["long_liq_below"]) and all(x["price"] > 100 for x in m["short_liq_above"])
    total = m["total_est_usd"]["long_below"] + m["total_est_usd"]["short_above"]
    assert total == pytest.approx(1000 * 100.0, rel=0.01)    # one OI increase of 1000 coins at ~100
    crash = f.with_columns(pl.when(pl.col("event_ts") == pl.col("event_ts").max()).then(50.0).otherwise(pl.col("low")).alias("low"))
    m2 = G.liq_map(crash, oi, AS_OF, 100.0)
    assert not m2["long_liq_below"]                          # every long was liquidated by the wick to 50


def test_geometry_no_lookahead():
    tfb = {"1m": bars(1500, tf_h=1 / 60, seed=1), "15m": bars(1000, tf_h=0.25, seed=2), "1h": bars(400, seed=3), "4h": bars(300, tf_h=4, seed=4),
           "1d": bars(200, tf_h=24, seed=5)}
    a = G.token_geometry("AUSDT", 100.0, tfb, AS_OF)
    b = G.token_geometry("AUSDT", 100.0, {k: future_rows(v, 10) for k, v in tfb.items()}, AS_OF)
    assert same(a, b)


# ---------------------------------------------------------------- tape

def test_tape_no_lookahead():
    k1m = bars(120, tf_h=1 / 60, seed=7)
    book = {"bids": [(99.9, 10.0), (99.5, 20.0)], "asks": [(100.1, 10.0), (100.6, 5.0)], "ts": AS_OF - timedelta(seconds=2), "known_ts": AS_OF - timedelta(seconds=1)}
    le = [AS_OF - timedelta(minutes=k) for k in (300, 50, 30, 5)]
    liq = pl.DataFrame({"event_ts": le, "side": ["long", "long", "short", "long"], "price": [100.0] * 4, "size": [1.0, 2.0, 3.0, 4.0], "known_ts": le}).with_columns(
        pl.col("event_ts", "known_ts").cast(US))
    fe = [AS_OF - timedelta(seconds=3 * k) for k in range(400, 100, -1)]
    flow = pl.DataFrame({"event_ts": fe, "buy_notional": 2.0, "sell_notional": 1.0, "n_buy": 1, "n_sell": 1, "spread_bps": 1.0, "depth_bid_25": 1e5,
                         "depth_ask_25": 1e5, "obi_5": 0.0, "known_ts": [t + timedelta(seconds=3) for t in fe]}).with_columns(pl.col("event_ts", "known_ts").cast(US))
    tk = {"lastPrice": "100", "markPrice": "100", "indexPrice": "100", "price24hPcnt": "0.01"}
    tok = {"ticker": tk, "ticker_known": AS_OF - timedelta(seconds=1), "k1m": k1m, "book": book, "liq": liq, "flow": flow}
    a = T.build_tape(AS_OF, {"AUSDT": tok})
    fut = AS_OF + timedelta(minutes=1)
    liq_f = pl.concat([liq, pl.DataFrame({"event_ts": [fut], "side": ["short"], "price": [100.0], "size": [1e6], "known_ts": [fut]}).with_columns(
        pl.col("event_ts", "known_ts").cast(US))])
    flow_f = pl.concat([flow, flow.tail(1).with_columns(pl.lit(fut).cast(US).alias("event_ts"), pl.lit(fut).cast(US).alias("known_ts"), pl.lit(1e9).alias("buy_notional"))])
    b = T.build_tape(AS_OF, {"AUSDT": {**tok, "k1m": future_rows(k1m), "liq": liq_f, "flow": flow_f}})
    assert same(a, b)
    r = a["tokens"][0]
    assert r["liq"]["1h"] == {"long_usd": 600, "short_usd": 300, "n": 3} and r["liq"]["15m"] == {"long_usd": 400, "short_usd": 0, "n": 1}
    assert r["flow"]["windows"]["5m"]["imbalance"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["book"]["spread_bp"] == pytest.approx(1e4 * math.log(100.1 / 99.9), abs=0.01)
    late = T.build_tape(AS_OF, {"AUSDT": {**tok, "book": {**book, "known_ts": AS_OF + timedelta(seconds=1)}}})
    assert late["tokens"][0]["book"]["status"] == "after as_of"


def test_player_read_uses_players_pattern():
    ev = [AS_OF - timedelta(hours=8 * k) for k in range(40, 0, -1)]
    fund = pl.DataFrame({"event_ts": ev, "rate": np.r_[np.full(39, 1e-4), -1e-4], "known_ts": ev}).with_columns(pl.col("event_ts", "known_ts").cast(US))
    le = [AS_OF - timedelta(hours=1)] * 2
    liq = pl.DataFrame({"event_ts": le, "side": ["long", "long"], "price": [100.0, 100.0], "size": [10.0, 5.0], "known_ts": le}).with_columns(pl.col("event_ts", "known_ts").cast(US))
    p = T.player_read(AS_OF, {"status": "ok", "cell": "R9", "ret_24h_pct": -3.0}, funding=fund, oi=None, liq=liq)
    assert p["pattern"].startswith("long flush, funding negative") and "long-liquidation burst (4h)" in p["pattern"]


# ---------------------------------------------------------------- news

RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Bitcoin plunges as ETF outflows rise</title><link>https://x/1</link><guid>g1</guid><pubDate>Sun, 15 Nov 2026 15:00:00 GMT</pubDate></item>
<item><title>Solana rallies on upgrade</title><link>https://x/2</link><guid>g2</guid><pubDate>Sun, 15 Nov 2026 15:10:00 GMT</pubDate></item>
</channel></rss>"""
RSS2 = RSS.replace("</channel>", "<item><title>Ethereum surges</title><guid>g3</guid><pubDate>Sun, 15 Nov 2026 16:40:00 GMT</pubDate></item></channel>")


def test_news_collect_and_point_in_time(tmp_path):
    idx = TickerIndex({"BTC", "ETH", "SOL"})
    t1, t2 = AS_OF - timedelta(minutes=20), AS_OF + timedelta(minutes=15)
    s1 = N.collect(tmp_path, idx, get=lambda u: RSS, now=t1, feeds={"f": "u"})
    s2 = N.collect(tmp_path, idx, get=lambda u: RSS2, now=t2, feeds={"f": "u"})
    assert s1["new"] == 2 and s2["new"] == 1                  # de-duplicated by guid hash
    items = N.news_items(tmp_path, AS_OF)
    assert {r["title"] for r in items} == {"Bitcoin plunges as ETF outflows rise", "Solana rallies on upgrade"}
    b = next(r for r in items if r["tickers"] == ["BTC"])
    assert b["tone"] < 0
    assert next(r for r in items if r["tickers"] == ["SOL"])["tone"] > 0
    cal = tmp_path / "cal"
    cal.mkdir()
    (cal / "a.yaml").write_text("written_at: 2026-11-15T10:00:00Z\nevents:\n  - {ts_utc: 2026-11-16T13:30Z, name: CPI, importance: 3}\n", encoding="utf-8")
    (cal / "b.yaml").write_text("written_at: 2026-11-15T17:00:00Z\nevents:\n  - {ts_utc: 2026-11-16T14:00Z, name: LATE, importance: 3}\n", encoding="utf-8")
    ev = N.calendar_events(cal, AS_OF)
    assert [e["name"] for e in ev] == ["CPI"]                  # a calendar written after as_of is not known yet


def test_recurring_events_cover_cme_nyse_deribit():
    sun = datetime(2026, 11, 15, 12, tzinfo=UTC)
    ev = N.recurring_events(sun, 48, [("2026-11-15T16:00:00+00:00", 3)])
    names = [e["name"] for e in ev]
    assert any("CME Globex reopen" in n for n in names) and any("NYSE cash open" in n for n in names)
    cme = next(e for e in ev if "CME Globex reopen" in e["name"])
    assert cme["ts"].startswith("2026-11-15T23:00")            # 17:00 CT (CST, UTC-6)
    nyse = next(e for e in ev if "NYSE cash open" in e["name"])
    assert nyse["ts"].startswith("2026-11-16T14:30")
    assert all(datetime.fromisoformat(e["ts"]) > sun for e in ev)


def test_option_expiries_max_pain():
    book = [{"instrument_name": "BTC-20NOV26-60000-P", "open_interest": 100, "underlying_price": 64000},
            {"instrument_name": "BTC-20NOV26-70000-C", "open_interest": 100, "underlying_price": 64000},
            {"instrument_name": "BTC-20NOV26-65000-C", "open_interest": 10, "underlying_price": 64000},
            {"instrument_name": "BTC-14NOV26-65000-C", "open_interest": 999, "underlying_price": 64000}]   # expired
    o = N.option_expiries(book, AS_OF)
    e = o["expiries"][0]
    assert len(o["expiries"]) == 1 and e["expiry"].startswith("2026-11-20T08:00") and e["put_call_oi"] == pytest.approx(100 / 110, abs=0.01)
    assert e["max_pain"] in (60000.0, 65000.0)


# ---------------------------------------------------------------- sentiment / reddit

def _listing(kind, items, after=None):
    return {"data": {"after": after, "children": [{"kind": kind, "data": d} for d in items]}}


def test_reddit_collector_with_canned_json(tmp_path):
    now = AS_OF
    t = now.timestamp()
    posts = _listing("t3", [{"name": "t3_a", "title": "BTC to the moon, very bullish", "selftext": "", "created_utc": t - 600, "score": 10, "num_comments": 4},
                            {"name": "t3_b", "title": "Solana got hacked? panic", "selftext": "", "created_utc": t - 300, "score": 2, "num_comments": 9}])
    comments = _listing("t1", [{"name": "t1_c", "body": "selling my $ETH, this is not good", "created_utc": t - 100, "score": 1}])
    calls = []

    def http(method, url, headers, data):
        calls.append((method, url))
        assert "User-Agent" in headers
        if url.endswith("access_token"):
            assert method == "POST" and data == b"grant_type=client_credentials" and headers["Authorization"].startswith("Basic ")
            return {"access_token": "tok", "expires_in": 3600}
        assert method == "GET" and headers["Authorization"] == "bearer tok" and url.startswith(S.REDDIT_API)
        return posts if "/new?" in url else comments

    cl = S.Reddit("id", "secret", "ua", http=http)
    idx = TickerIndex({"BTC", "ETH", "SOL"})
    st = S.collect_reddit(tmp_path, cl, idx, now=now, subs={"CryptoCurrency": None})
    assert st["new"] == 3 and st["subs"]["CryptoCurrency"] == 3
    rows = [json.loads(x) for x in (tmp_path / "reddit" / f"{now.date()}.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {r["id"]: r for r in rows}
    assert by["t3_a"]["tickers"] == ["BTC"] and by["t3_a"]["tone"] > 0.5
    assert by["t3_b"]["tickers"] == ["SOL"] and by["t3_b"]["tone"] < 0
    assert by["t1_c"]["tickers"] == ["ETH"] and by["t1_c"]["tone"] < 0 and "body" not in by["t1_c"]
    st2 = S.collect_reddit(tmp_path, cl, idx, now=now + timedelta(minutes=15), subs={"CryptoCurrency": None})
    assert st2["new"] == 0                                      # already stored ids are skipped
    assert sum(1 for m, _ in calls if m == "POST") == 1         # the token is reused
    assert S.collect_reddit(tmp_path, None, idx, now=now) == {"reddit": "pending keys", "new": 0}
    with pytest.raises(ValueError):
        cl.listing("CryptoCurrency", "submit")


def test_sentiment_panel_trend_and_no_lookahead():
    since = AS_OF - timedelta(days=3)
    items = []
    rng = np.random.default_rng(0)
    for h in range(72):
        hour = since + timedelta(hours=h)
        n = 3 if hour < AS_OF - timedelta(hours=7) else 15     # mentions jump in the last 6 complete hours
        for k in range(int(rng.poisson(n))):
            ts = (hour + timedelta(minutes=k % 60)).isoformat()
            items.append({"id": f"{h}-{k}", "kind": "comment", "source": "reddit", "created_ts": ts, "seen_ts": ts, "tickers": ["BTC"], "tone": 0.1})
    p = S.hourly_panel(items, AS_OF)
    tr = S.trend(p["BTC"], AS_OF, since)
    assert tr["trend"] == "rising" and tr["z_mentions"] > 1
    fut = [{"id": f"f{k}", "kind": "comment", "source": "reddit", "created_ts": (AS_OF - timedelta(hours=1)).isoformat(),
            "seen_ts": (AS_OF + timedelta(minutes=5)).isoformat(), "tickers": ["BTC"], "tone": -1.0} for k in range(500)]
    p2 = S.hourly_panel(items + fut, AS_OF)
    assert same(S.trend(p2["BTC"], AS_OF, since), tr)          # seen after as_of: not known yet
    short = S.trend(p["BTC"], AS_OF, AS_OF - timedelta(hours=10))
    assert short["trend"] == "insufficient"


def test_tone_and_tickers():
    assert tone("very bullish breakout") > tone("bullish breakout") > 0
    assert tone("not bullish") < 0 and tone("crash and panic") < -0.5 and tone("the weather") == 0
    idx = TickerIndex({"BTC", "BCH", "ETH", "NEAR", "OP"})
    assert idx.find("Bitcoin Cash forks again") == ["BCH"]
    assert idx.find("bitcoin and ethereum") == ["BTC", "ETH"]
    assert idx.find("we are near the top") == [] and idx.find("$OP pumps") == ["OP"] and idx.find("NEAR rallies") == []
    assert idx.find("anything", default="eth") == ["ETH"]


# ---------------------------------------------------------------- contract validator

SNAP = {"xs": {"tokens": [{"symbol": "SOLUSDT", "last": 100.0, "rs_sigma": {"24h": -2.3}, "regime": {"cell": "R9"}}]},
        "geometry": {"tokens": [{"symbol": "SOLUSDT", "atr14": {"1h": 1.0, "1d": 4.0}}]},
        "tape": {"tokens": [{"symbol": "SOLUSDT", "price": {"last": 100.0}, "liq": {"cluster_60m": {"fired": False}}}]}}
RULES = {"R-XS-3", "R-RB-1", "R-EXE-1", "R-BUDGET-1", "R-FLOW-2", "R-PX-4"}


def report(**kw):
    d = {"contract": "desk_report_v1", "agent": "xs", "cycle_id": "20261115T1630", "as_of": AS_OF.isoformat(), "snapshot": "research/desk/2026-11-15/1630",
         "model": "claude-opus-5-5", "prompt_version": "desk-xs-v1",
         "regime": {"summary": "R9 liquidation across majors"},
         "ideas": [{"rank": 1, "id": "xs-1", "symbols": ["SOLUSDT"], "kind": "relative_value", "direction": "none", "thesis": "SOL lagging the basket",
                    "evidence": [{"metric": "rs_sigma.24h", "symbol": "SOLUSDT", "value": -2.3, "source": "xs.json"}],
                    "invalidation": {"price": 105.0, "condition": "1h close above"}, "conviction": 0.4, "reasoning": "largest |rs| among majors", "rules": ["R-XS-3"]},
                   {"rank": 2, "id": "xs-2", "symbols": ["SOLUSDT"], "kind": "watch", "direction": "none", "thesis": "funding",
                    "evidence": [{"metric": "rs_sigma.24h", "symbol": "SOLUSDT", "value": -2.3, "source": "xs.json"}],
                    "invalidation": {"condition": "funding back above p50"}, "conviction": 0.2, "reasoning": "weak", "rules": []}],
         "honesty": {"direction_claim": False, "what_would_change_my_mind": "rs reverting"}}
    d.update(kw)
    return d


def test_report_validator():
    assert C.validate_report(report(), SNAP, RULES)["ok"]
    bad = report()
    bad["ideas"][1]["conviction"] = 0.9
    assert any("ranked by conviction" in e for e in C.validate_report(bad, SNAP, RULES)["errors"])
    bad = report()
    bad["ideas"][0]["evidence"][0]["value"] = -1.0
    assert any("report says -1.0" in e for e in C.validate_report(bad, SNAP, RULES)["errors"])
    bad = report()
    del bad["ideas"][0]["invalidation"]
    assert not C.validate_report(bad, SNAP, RULES)["ok"]
    d = report()
    reordered = {k: d[k] for k in d if k != "regime"} | {"regime": d["regime"]}
    assert any("before ideas" in e for e in C.validate_report(reordered, SNAP, RULES)["errors"])
    bad = report()
    bad["ideas"][0]["rules"] = ["R-NOPE-9"]
    assert any("unknown rule" in e for e in C.validate_report(bad, SNAP, RULES)["errors"])


def test_markdown_validator():
    ok = "\n".join(["# XS report", "## Regime", "R9", "## Ideas", "### 1. SOLUSDT relative_value none (conviction 0.40)", "- Thesis: x", "- Evidence: y",
                    "- Invalidation: z", "## No trade / watch", "-", "## Data gaps", "-", "## Honesty", "-"])
    assert C.validate_markdown(ok)["ok"]
    assert not C.validate_markdown(ok.replace("## Regime\nR9\n", "") + "\n## Regime")["ok"]
    assert not C.validate_markdown(ok.replace("(conviction 0.40)", ""))["ok"]


def proposal(**kw):
    p = {"id": "p1", "symbol": "SOLUSDT", "side": "short", "entry_type": "limit", "entry_mode": "pegged_limit", "peg": "touch",
         "entry": 100.0, "stop": 102.0, "take_profit": 97.0, "targets": [98.0, 97.0], "p_tp": 0.40, "cluster": "crypto_beta", "channel": "desk",
         "horizon": "4h", "thesis": "post-flush rebound fading under the 24h VAH", "invalidation": "1h close above 102",
         "equilibrium": {"positioning": "funding p0, OI flat", "trapped": "late longs at 101", "stops": "above 101.8 swing high", "forced_flows": "00:00 funding",
                         "payoff_asymmetry": "1.5R", "next_moves": {"longs": "defend 99", "shorts": "add above 101"}, "pain_thresholds": {"longs": 98.5, "shorts": 102.0}},
         "supports": [{"agent": "geometry", "idea": "geo-1"}], "rules": ["R-RB-5", "R-EXE-1"],
         "evidence": [{"metric": "rs_sigma.24h", "symbol": "SOLUSDT", "value": -2.3, "source": "xs.json"}]}
    p.update(kw)
    return p


def pfile(*ps, **kw):
    d = {"session_id": "desk-20261115T1630", "proposals": list(ps),
         "meta": {"model": "claude-opus-5-5", "prompt_version": "desk-trader-v1", "as_of": AS_OF.isoformat(), "snapshot": "x", "cycle_id": "20261115T1630"}}
    d.update(kw)
    return d


def test_p_tp_driftless_gamblers_ruin():
    assert C.p_tp_driftless("long", 100, 99, 102) == pytest.approx(math.log(100 / 99) / math.log(102 / 99))
    assert C.p_tp_driftless("short", 100, 102, 97) == pytest.approx(math.log(102 / 100) / math.log(102 / 97))
    assert C.p_tp_driftless("long", 100, 101, 102) is None
    assert C.p_tp_driftless("long", 100, 98, 102) == pytest.approx(0.5, abs=0.01)   # symmetric in log space ~ 1/2
    assert C.reward_risk("long", 100, 99, 102) == pytest.approx(math.log(1.02) / math.log(100 / 99))


def test_proposal_validator():
    rules = RULES | {"R-RB-5"}
    r = C.validate_proposals(pfile(proposal()), SNAP, rules)
    assert r["ok"], r["errors"]
    assert r["derived"][0]["p_tp_driftless"] == pytest.approx(C.p_tp_driftless("short", 100, 102, 97), abs=1e-4)
    for bad, msg in ((proposal(qty=1.0), "size"), (proposal(stop=99.0), "short needs"), (proposal(entry_type="market"), "pegged_limit"),
                     (proposal(take_profit=96.0), "one of targets"), (proposal(symbol="SPYUSDT"), "SPYUSDT"), (proposal(horizon="2w"), "horizon"),
                     (proposal(equilibrium={"positioning": "x"}), "equilibrium"), (proposal(supports=[]), "supports"),
                     (proposal(rules=["R-FAKE-1"]), "unknown rule"), (proposal(symbol="DOGEUSDT"), "not in the snapshot universe")):
        out = C.validate_proposals(pfile(bad), SNAP, rules)
        assert not out["ok"] and any(msg in e for e in out["errors"]), (msg, out["errors"])
    assert not C.validate_proposals(pfile(proposal(), extra=1), SNAP, rules)["ok"]                      # atx propose denies unknown top-level keys
    assert not C.validate_proposals(pfile(*[proposal(id=f"p{i}") for i in range(4)]), SNAP, rules)["ok"]
    w = C.validate_proposals(pfile(proposal(side="long", stop=98.0, take_profit=103.0, targets=[103.0], p_tp=0.9)), SNAP, rules)
    assert any("driftless" in x for x in w["warnings"]) and any("R-RB-1" in x for x in w["warnings"])   # R9 long without R-RB-1


def test_proposal_is_valid_atx_propose_input():
    """The core fields are exactly docs/EXECUTION.md's TradeProposal; extras ride inside each proposal."""
    p = proposal()
    core = {"id", "symbol", "side", "entry_type", "entry", "stop", "take_profit", "p_tp", "cluster", "channel"}
    assert core <= set(p) and p["side"] in ("long", "short") and p["entry_type"] in ("market", "limit")
    assert set(pfile(p)) <= {"session_id", "proposals", "clusters", "meta"}
