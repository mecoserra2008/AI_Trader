"""Deribit implied-vol surface (alt_series source=deribit_svi), one causal fit per day at 08:00 UTC (D8, 2026-11-27).

Until 2026-11-27 the series came only from a static import of the user's earlier project (BACKTEST_TREND_FOLLOWING,
`btf_svi_v2`, ending 2026-10-09). This job keeps it running with the SAME method, so the series stays one series:
  - trades: the most recent Deribit option trades strictly before 08:00 UTC of day d (history.deribit.com, newest
    first, 04:00 -> 08:00; widened to 00:00, then 16:00 of d-1 while fewer than 300 trades; at most 3000), exactly as
    research/btc_options/fetch_data.py::fetch_day;
  - fit: the project's own harness (research/btc_options/harness/surface.py: fit_day(.., "svi_v2") + summarize), which
    uses only trades with timestamp < 08:00 again;
  - rows: 12 series iv_{atm,rr25,bf25}_{7,30,90}, iv_ts_slope_30_7, iv_ts_slope_90_30, iv_basis_ann_30;
    event_ts = 08:00 of d, known_ts = 08:01 (WS02's open question, answered: every input is timestamped before 08:00,
    so the 60 s fit latency is the only lag; verified in the source 2026-11-27).
The harness reads OPT_CCY at import, so each currency runs in its own process.

    uv run python -m ai_trader.storage.deribit_svi --ccy BTC [--since 2026-10-10] [--until 2026-11-26] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from ai_trader.storage.lake import Lake

BTF = Path(os.environ.get("BTF_ROOT", "C:/Users/Asus/Documents/BACKTEST_TREND_FOLLOWING"))
HIST = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
H, DAY = 3_600_000, 86_400_000
MIN_TRADES, MAX_TRADES = 300, 3000
KEEP = ["timestamp", "trade_id", "instrument_name", "price", "mark_price", "iv", "index_price", "direction", "amount", "block_trade_id", "combo_id"]
SERIES = [f"{m}_{t}" for t in (7, 30, 90) for m in ("atm", "rr25", "bf25")] + ["ts_slope_30_7", "ts_slope_90_30", "basis_ann_30"]
LATENCY_MS = 60_000


def _get(params: dict, tries: int = 8) -> dict:
    url = HIST + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-trader/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                js = json.loads(r.read())
            if "error" in js:
                raise RuntimeError(js["error"])
            return js
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(60, 1.5 * 2 ** i))
    raise RuntimeError("unreachable")


def _page_back(ccy: str, start_ms: int, end_ms: int, have: dict, cap: int) -> int:
    cursor, oldest = end_ms, end_ms
    while len(have) < cap:
        js = _get({"currency": ccy, "kind": "option", "start_timestamp": start_ms, "end_timestamp": cursor, "count": 1000, "sorting": "desc"})
        tr = js["result"]["trades"]
        time.sleep(0.15)
        if not tr:
            return start_ms
        new = 0
        for t in tr:
            if t["trade_id"] not in have:
                have[t["trade_id"]] = {k: t.get(k) for k in KEEP}
                new += 1
        oldest = tr[-1]["timestamp"]
        if not js["result"].get("has_more", False) or len(tr) < 1000:
            return start_ms
        cursor = oldest if new > 0 and oldest < cursor else oldest - 1
    return oldest


def fetch_day(ccy: str, d: date):
    """Option trades strictly before 08:00 UTC of `d` (fetch_data.py::fetch_day, unchanged)."""
    import pandas as pd

    T = int(datetime(d.year, d.month, d.day, 8, tzinfo=UTC).timestamp() * 1000)
    have: dict = {}
    oldest, w_used = T, None
    for w0 in (T - 4 * H, T - 8 * H, T - 16 * H):
        end = T - 1 if not have else min(r["timestamp"] for r in have.values())
        oldest = _page_back(ccy, w0, end, have, MAX_TRADES)
        w_used = w0
        if len(have) >= MIN_TRADES:
            break
    df = pd.DataFrame(list(have.values()), columns=KEEP)
    df = df[df["timestamp"] < T].sort_values(["timestamp", "trade_id"]) if len(df) else df
    df["window_start_ms"] = max(w_used, oldest) if len(df) else w_used
    return T, df


def _harness():
    if str(BTF) not in sys.path:
        sys.path.insert(0, str(BTF))
    from research.btc_options.harness.surface import DaySurface, fit_day, summarize  # noqa: PLC0415

    return DaySurface, fit_day, summarize


def surface_row(ccy: str, T: int, trades) -> dict | None:
    """_surface_one_day of research/btc_options/build_dataset.py for smile_model svi_v2. None = no surface that day."""
    DaySurface, fit_day, summarize = _harness()
    if len(trades) == 0:
        return None
    tab, diag = fit_day(trades, T, "svi_v2")
    if len(tab) < 2:
        return None
    return summarize(DaySurface(T, diag["index"], tab))


def rows_for(ccy: str, T: int, summary: dict) -> pl.DataFrame:
    ev = datetime.fromtimestamp(T / 1000, UTC)
    recs = [{"source": "deribit_svi", "series_id": f"iv_{s}", "entity": ccy, "event_ts": ev, "value": float(summary[s]), "vintage": "btf_svi_v2",
             "pit_quality": "native", "known_ts": ev + timedelta(milliseconds=LATENCY_MS)}
            for s in SERIES if s in summary and summary[s] is not None and summary[s] == summary[s] and abs(float(summary[s])) != float("inf")]
    return pl.DataFrame(recs).with_columns(pl.col("event_ts", "known_ts").dt.cast_time_unit("ms")) if recs else pl.DataFrame()


def newest_day(lake: Lake, ccy: str) -> date | None:
    files = list((lake.root / "alt_series" / "source=deribit_svi").rglob("part-*.parquet"))
    if not files:
        return None
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    flist = "[" + ", ".join("'" + f.as_posix() + "'" for f in files) + "]"
    ms = con.sql(f"SELECT max(epoch_ms(event_ts)) FROM read_parquet({flist}) WHERE entity = ?", params=[ccy]).fetchall()[0][0]
    con.close()
    return None if ms is None else datetime.fromtimestamp(ms / 1000, UTC).date()


def run(ccy: str, since: date | None, until: date | None, dry_run: bool = False, lake: Lake | None = None) -> list[str]:
    lake = lake or Lake()
    now = datetime.now(UTC)
    last = newest_day(lake, ccy)
    start = since or ((last + timedelta(days=1)) if last else now.date())
    # A day's surface exists once 08:00 has passed (plus the 60 s latency); never before.
    end = min(until or now.date(), (now - timedelta(hours=8, minutes=2)).date())
    log = [f"{ccy}: newest in lake {last}, fitting {start} .. {end}"]
    d = start
    out = []
    while d <= end:
        T, tr = fetch_day(ccy, d)
        s = surface_row(ccy, T, tr)
        if s is None:
            log.append(f"{d}: {len(tr)} trades, no surface")
        else:
            r = rows_for(ccy, T, s)
            out.append(r)
            log.append(f"{d}: {len(tr)} trades, atm_30 {s.get('atm_30', float('nan')):.4f}, {r.height} rows")
        d += timedelta(days=1)
    if out and not dry_run:
        df = pl.concat(out)
        lake.write("alt_series", df)
        log.append(f"wrote {df.height} rows")
    return log


def _cli(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="deribit_svi")
    ap.add_argument("--ccy", required=True, choices=["BTC", "ETH"])
    ap.add_argument("--since", type=date.fromisoformat)
    ap.add_argument("--until", type=date.fromisoformat)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("OPT_CCY", "BTC").upper() != a.ccy:
        raise SystemExit(f"set OPT_CCY={a.ccy} before running (the fitting harness reads it at import)")
    for line in run(a.ccy, a.since, a.until, a.dry_run):
        sys.stdout.write(line + "\n")


if __name__ == "__main__":
    _cli()
