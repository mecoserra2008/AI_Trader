"""CFTC Commitments of Traders positioning snapshot (public Socrata API, keyless).

    PYTHONIOENCODING=utf-8 uv run python scripts/cftc_positioning.py [--weeks 156]

Financial futures from the Traders in Financial Futures report (TFF, dataset gpe5-46if): leveraged funds (hedge funds,
CTAs) and asset managers. Commodities from the Disaggregated report (72hh-3qpy): managed money. For each market: net
position (long - short), net as % of open interest, 1-week and 4-week change, and its z-score / percentile over the
lookback. Reports are as of Tuesday and released Friday 15:30 ET, so `known` = report date + 3 days.
Snapshot saved to data/cftc/<report_date>.parquet.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import load

API = "https://publicreporting.cftc.gov/resource/{ds}.json"
TFF, DISAGG = "gpe5-46if", "72hh-3qpy"
MARKETS = [
    # (label, dataset, exact market_and_exchange_names, group)
    ("BTC (CME)", TFF, "BITCOIN - CHICAGO MERCANTILE EXCHANGE", "lev"),
    ("BTC (CME)", TFF, "BITCOIN - CHICAGO MERCANTILE EXCHANGE", "asset"),
    ("ETH (CME)", TFF, "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE", "lev"),
    ("S&P 500 e-mini", TFF, "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "lev"),
    ("S&P 500 e-mini", TFF, "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "asset"),
    ("Nasdaq-100 mini", TFF, "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE", "lev"),
    ("Nasdaq-100 mini", TFF, "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE", "asset"),
    ("VIX futures", TFF, "VIX FUTURES - CBOE FUTURES EXCHANGE", "lev"),
    ("UST 2Y", TFF, "UST 2Y NOTE - CHICAGO BOARD OF TRADE", "lev"),
    ("UST 10Y", TFF, "UST 10Y NOTE - CHICAGO BOARD OF TRADE", "lev"),
    ("UST 10Y", TFF, "UST 10Y NOTE - CHICAGO BOARD OF TRADE", "asset"),
    ("Japanese yen", TFF, "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "lev"),
    ("Japanese yen", TFF, "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "asset"),
    ("WTI crude", DISAGG, "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE", "mm"),
    ("WTI crude (NYMEX)", DISAGG, "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE", "mm"),
    ("Brent crude", DISAGG, "BRENT LAST DAY - NEW YORK MERCANTILE EXCHANGE", "mm"),
    ("Gold", DISAGG, "GOLD - COMMODITY EXCHANGE INC.", "mm"),
    ("Silver", DISAGG, "SILVER - COMMODITY EXCHANGE INC.", "mm"),
]
COLS = {
    "lev": ("lev_money_positions_long", "lev_money_positions_short", "leveraged funds"),
    "asset": ("asset_mgr_positions_long", "asset_mgr_positions_short", "asset managers"),
    "mm": ("m_money_positions_long_all", "m_money_positions_short_all", "managed money"),
}


def get(ds: str, where: str, limit: int) -> list[dict]:
    q = {"$where": where, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": str(limit)}
    with urllib.request.urlopen(API.format(ds=ds) + "?" + urllib.parse.urlencode(q), timeout=60) as r:
        return json.load(r)


def names_like(ds: str, pattern: str) -> list[str]:
    q = {"$select": "distinct market_and_exchange_names", "$where": f"upper(market_and_exchange_names) like '{pattern}'", "$limit": "50"}
    with urllib.request.urlopen(API.format(ds=ds) + "?" + urllib.parse.urlencode(q), timeout=60) as r:
        return [x["market_and_exchange_names"] for x in json.load(r)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--list", default=None, help="print market names matching a LIKE pattern, e.g. %%CRUDE%%")
    a = ap.parse_args()
    if a.list:
        for ds in (TFF, DISAGG):
            print(ds, names_like(ds, a.list.upper()))
        return
    rows, cache = [], {}
    for label, ds, name, grp in MARKETS:
        key = (ds, name)
        if key not in cache:
            cache[key] = get(ds, f"market_and_exchange_names = '{name}'", a.weeks + 5)
        data = cache[key]
        if not data:
            print(f"  no data: {label} ({name})")
            continue
        lc, sc, who = COLS[grp]
        df = pl.DataFrame([{"d": x["report_date_as_yyyy_mm_dd"][:10], "oi": float(x.get("open_interest_all") or 0),
                            "long": float(x.get(lc) or 0), "short": float(x.get(sc) or 0)} for x in data]).sort("d")
        df = df.with_columns((pl.col("long") - pl.col("short")).alias("net")).with_columns((pl.col("net") / pl.col("oi") * 100).alias("net_pct_oi"))
        n = df["net_pct_oi"].to_numpy()
        net = df["net"].to_numpy()
        rows.append({"market": label, "group": who, "report": df["d"][-1], "net": net[-1], "net_pct_oi": n[-1],
                     "chg_1w": net[-1] - net[-2] if len(net) > 1 else np.nan, "chg_4w": net[-1] - net[-5] if len(net) > 4 else np.nan,
                     "z_3y": (n[-1] - n.mean()) / (n.std() or 1), "pct_3y": float((n < n[-1]).mean() * 100), "oi_chg_4w_%": (df["oi"][-1] / df["oi"][-5] - 1) * 100 if df.height > 4 else np.nan,
                     "weeks": df.height})
    out = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(40), pl.Config.set_tbl_cols(20), pl.Config.set_tbl_width_chars(220), pl.Config.set_float_precision(1)
    print(out)
    d = load().root / "data" / "cftc"
    d.mkdir(parents=True, exist_ok=True)
    rep = out["report"].max()
    out.with_columns(pl.lit(str(pl.Series([rep]).str.to_date()[0] + timedelta(days=3))).alias("known")).write_parquet(d / f"{rep}.parquet")
    print("saved", d / f"{rep}.parquet")


if __name__ == "__main__":
    main()
