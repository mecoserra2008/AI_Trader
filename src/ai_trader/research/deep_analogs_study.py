"""Deep analogs (agent DEEP, 2026-11-15): public-exchange microstructure around the top-10 combined BTC analogs of
research/reports/analogs_2026-11-15.md and the current episode. Research only, descriptive (chosen after seeing today;
H-ANA1 stands: analogs do not forecast).

Stage 1 `fetch`: one file at a time into data/raw/deep/, reduced to small parquet aggregates under
data/research/deep_analogs/<EP>/, raw deleted right after. A manifest (data/research/deep_analogs/manifest.json) records
every URL (ok / missing / error, bytes) so nothing is downloaded twice. Polite: 1 connection, 429/418 backoff.

Sources per episode (T = analog hour, window T-3d..T+7d):
  Binance UM   1m klines T-33d..T+14d (taker-buy volume -> delta), metrics T-10d..T+7d (OI, top-trader account and position
               L/S, global account L/S, taker L/S volume ratio), fundingRate monthly, bookDepth T-3d..T+7d (2023+),
               spot 1m klines T-3d..T+7d (basis). UM liquidationSnapshot does not exist in the archive; the coin-margined
               BTCUSD_PERP liquidationSnapshot is used where present.
  Bybit        public trade archive for the T date +-1 (1m aggregates, large prints), ob200 for the T date where present
               (1m samples: top-of-book depth and span; BTC ob200 only spans a few bp), REST 1h open interest.
  OKX          REST 1H candles (cross-check), funding where the REST history still has it.
  NOW          archives up to 2026-11-14; 2026-11-15 from Binance public REST and the lake (Bybit flow_3s, liquidations).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m ai_trader.research.deep_analogs_study fetch [--only EP] [--skip-ob]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "deep"
OUT = ROOT / "data" / "research" / "deep_analogs"
MANIFEST = OUT / "manifest.json"
UA = "Mozilla/5.0 (AI_Trader research; public archives)"
BV = "https://data.binance.vision/data"
TODAY = datetime(2026, 11, 15, tzinfo=UTC)

EPISODES: list[tuple[str, str, str]] = [
    ("A01", "2025-11-22 06:00", "2025 H2 post-Oct-10 deleveraging"),
    ("A02", "2026-02-01 16:00", "2026 H1 bear (Feb)"),
    ("A03", "2026-06-07 15:00", "2026 H1 bear (June rout)"),
    ("A04", "2022-01-22 17:00", "2022 Q1 Fed tightening"),
    ("A05", "2023-03-11 10:00", "2023 SVB"),
    ("A06", "2024-07-06 04:00", "2024 Mt Gox / German govt sales"),
    ("A07", "2022-06-14 20:00", "2022 June Celsius / 3AC"),
    ("A08", "2025-11-15 00:00", "2025 H2 post-Oct-10 deleveraging"),
    ("A09", "2025-02-27 16:00", "2025 H1 Bybit hack / tariffs"),
    ("A10", "2022-05-12 21:00", "2022 May LUNA"),
    ("NOW", "2026-11-15 17:00", "current episode"),
]


def ep_T(ep: str) -> datetime:
    s = {e[0]: e[1] for e in EPISODES}[ep]
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)


def days(a: datetime, b: datetime) -> list[str]:
    d0 = a.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    while d0 <= b:
        out.append(d0.strftime("%Y-%m-%d"))
        d0 += timedelta(days=1)
    return out


def months(a: datetime, b: datetime) -> list[str]:
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# ---------------------------------------------------------------------------------------------------------------- io


class Manifest:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.d: dict[str, Any] = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    def done(self, key: str) -> bool:
        return self.d.get(key, {}).get("status") in ("ok", "missing")

    def set(self, key: str, **kw) -> None:
        self.d[key] = {**self.d.get(key, {}), **kw, "ts": datetime.now(UTC).isoformat(timespec="seconds")}
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(MANIFEST)


def http_get(url: str, dest: Path | None = None, tries: int = 6) -> tuple[int, bytes | None, int]:
    """GET with 429/418/5xx backoff. Returns (status, body or None if written to dest, bytes)."""
    wait = 2.0
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                if dest is None:
                    b = r.read()
                    return 200, b, len(b)
                dest.parent.mkdir(parents=True, exist_ok=True)
                n = 0
                with dest.open("wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        n += len(chunk)
                return 200, None, n
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, None, 0
            if e.code in (418, 429) or e.code >= 500:
                ra = e.headers.get("Retry-After") if e.headers else None
                time.sleep(float(ra) if ra and ra.isdigit() else wait)
                wait = min(wait * 2, 120)
                continue
            return e.code, None, 0
        except Exception:  # noqa: BLE001 - network hiccup: back off and retry
            time.sleep(wait)
            wait = min(wait * 2, 120)
    return -1, None, 0


def zip_csv(b: bytes, header_hint: str | None = None) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0].decode()
    has_header = not first[:1].isdigit() and not first.startswith("-")
    return pl.read_csv(raw, has_header=has_header, infer_schema_length=0)


KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
           "taker_buy_quote_volume", "ignore"]


def kl_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.columns[0] != "open_time":
        df.columns = KL_COLS[: len(df.columns)]
    t = df["open_time"].cast(pl.Int64)
    t = pl.Series(np.where(t.to_numpy() > 1e14, t.to_numpy() // 1000, t.to_numpy()))  # spot 2025+ is in us
    return pl.DataFrame({
        "ts": pl.from_epoch(t, time_unit="ms").dt.replace_time_zone("UTC"),
        **{c: df[c].cast(pl.Float64) for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume",
                                                "taker_buy_quote_volume")},
        "count": df["count"].cast(pl.Int64),
    })


# ------------------------------------------------------------------------------------------------------------ binance


def fetch_binance(ep: str, man: Manifest, log=print) -> None:
    T = ep_T(ep)
    d = OUT / ep
    d.mkdir(parents=True, exist_ok=True)
    last_arch = TODAY - timedelta(days=1)

    def daily(kind: str, url_fn, rng: tuple[datetime, datetime], parse) -> pl.DataFrame | None:
        frames = []
        for day in days(*rng):
            dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
            if dt > last_arch:
                continue
            url = url_fn(day)
            cache = d / "cache" / kind / f"{day}.parquet"
            if man.done(url):
                if cache.exists():
                    frames.append(pl.read_parquet(cache))
                continue
            st, b, n = http_get(url)
            if st == 200 and b is not None:
                try:
                    f = parse(zip_csv(b))
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    f.write_parquet(cache)
                    frames.append(f)
                    man.set(url, status="ok", bytes=n, out=str(cache.relative_to(ROOT)))
                except Exception as e:  # noqa: BLE001
                    man.set(url, status="error", err=repr(e)[:200])
            elif st == 404:
                man.set(url, status="missing")
            else:
                man.set(url, status="error", http=st)
            time.sleep(0.15)
        return pl.concat(frames, how="diagonal_relaxed").sort(frames[0].columns[0]) if frames else None

    um = "futures/um/daily"
    k = daily("um_1m", lambda day: f"{BV}/{um}/klines/BTCUSDT/1m/BTCUSDT-1m-{day}.zip", (T - timedelta(days=33), T + timedelta(days=14)), kl_frame)
    if k is not None:
        k.unique("ts").sort("ts").write_parquet(d / "bn_perp_1m.parquet")
    s = daily("spot_1m", lambda day: f"{BV}/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{day}.zip", (T - timedelta(days=3), T + timedelta(days=7)), kl_frame)
    if s is not None:
        s.unique("ts").sort("ts").write_parquet(d / "bn_spot_1m.parquet")

    def p_metrics(df: pl.DataFrame) -> pl.DataFrame:
        num = [c for c in df.columns if c not in ("create_time", "symbol")]
        return df.select(pl.col("create_time").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.replace_time_zone("UTC").alias("ts"),
                         *[pl.col(c).replace("", None).cast(pl.Float64, strict=False) for c in num])

    m = daily("metrics", lambda day: f"{BV}/{um}/metrics/BTCUSDT/BTCUSDT-metrics-{day}.zip", (T - timedelta(days=10), T + timedelta(days=7)), p_metrics)
    if m is not None:
        m.unique("ts").sort("ts").write_parquet(d / "bn_metrics.parquet")

    def p_depth(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.replace_time_zone("UTC").alias("ts"),
                         pl.col("percentage").cast(pl.Float64).alias("pct"), pl.col("notional").cast(pl.Float64))

    if T >= datetime(2023, 1, 4, tzinfo=UTC):
        bd = daily("bookdepth", lambda day: f"{BV}/{um}/bookDepth/BTCUSDT/BTCUSDT-bookDepth-{day}.zip", (T - timedelta(days=3), T + timedelta(days=7)), p_depth)
        if bd is not None:
            # 1-minute mean notional per percentage band (raw is ~30 s)
            bd = bd.with_columns(pl.col("ts").dt.truncate("1m")).group_by("ts", "pct").agg(pl.col("notional").mean()).sort("ts", "pct")
            bd.write_parquet(d / "bn_bookdepth_1m.parquet")

    def p_liq(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.from_epoch(pl.col("time").cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC").alias("ts"),
                         pl.col("side"), pl.col("original_quantity").cast(pl.Float64).alias("contracts"),
                         pl.col("average_price").cast(pl.Float64).alias("avg_price"),
                         pl.col("accumulated_fill_quantity").cast(pl.Float64).alias("filled")).unique()

    if T >= datetime(2023, 6, 25, tzinfo=UTC):
        lq = daily("cm_liq", lambda day: f"{BV}/futures/cm/daily/liquidationSnapshot/BTCUSD_PERP/BTCUSD_PERP-liquidationSnapshot-{day}.zip",
                   (T - timedelta(days=3), T + timedelta(days=7)), p_liq)
        if lq is not None:
            lq.sort("ts").write_parquet(d / "bn_cm_liq.parquet")

    fr = []
    for mo in months(T - timedelta(days=95), min(T + timedelta(days=14), TODAY)):
        url = f"{BV}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{mo}.zip"
        cache = d / "cache" / "funding" / f"{mo}.parquet"
        if man.done(url):
            if cache.exists():
                fr.append(pl.read_parquet(cache))
            continue
        st, b, n = http_get(url)
        if st == 200 and b is not None:
            df = zip_csv(b)
            f = df.select(pl.from_epoch(pl.col(df.columns[0]).cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC").alias("ts"),
                          pl.col(df.columns[2]).cast(pl.Float64).alias("rate"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            f.write_parquet(cache)
            fr.append(f)
            man.set(url, status="ok", bytes=n)
        else:
            man.set(url, status="missing" if st == 404 else "error", http=st)
    if ep == "NOW":  # current month: REST
        st, b, _ = http_get("https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=200")
        if st == 200 and b:
            js = json.loads(b)
            fr.append(pl.DataFrame({"ts": [datetime.fromtimestamp(x["fundingTime"] / 1000, UTC) for x in js],
                                    "rate": [float(x["fundingRate"]) for x in js]}))
    if fr:
        f = pl.concat(fr).with_columns(pl.col("ts").dt.truncate("1m")).unique("ts").sort("ts")
        f.write_parquet(d / "bn_funding.parquet")
    log(f"[{ep}] binance done: 1m {0 if k is None else k.height} spot {0 if s is None else s.height} metrics {0 if m is None else m.height}")


def fetch_binance_today(man: Manifest, log=print) -> None:
    """2026-11-15 (not yet in the archive) from Binance public REST: 1m perp + spot klines and 5m metrics."""
    d = OUT / "NOW"
    t0 = int(TODAY.timestamp() * 1000)

    def klines(base: str) -> pl.DataFrame:
        rows, start = [], t0
        while True:
            st, b, _ = http_get(f"{base}?symbol=BTCUSDT&interval=1m&startTime={start}&limit=1000")
            if st != 200 or not b:
                break
            js = json.loads(b)
            if not js:
                break
            rows += js
            start = js[-1][0] + 60_000
            if len(js) < 1000:
                break
            time.sleep(0.3)
        df = pl.DataFrame([r[:11] for r in rows], schema=KL_COLS[:11], orient="row")
        return kl_frame(df.with_columns(pl.all().cast(pl.Utf8)))

    for base, name in (("https://fapi.binance.com/fapi/v1/klines", "bn_perp_1m"), ("https://api.binance.com/api/v3/klines", "bn_spot_1m")):
        k = klines(base)
        p = d / f"{name}.parquet"
        old = pl.read_parquet(p) if p.exists() else None
        k = pl.concat([old, k], how="diagonal_relaxed") if old is not None else k
        k = k.unique("ts", keep="last").sort("ts")
        k.write_parquet(p)
    ep_map = {"oi": ("openInterestHist", "sumOpenInterest"), "top_acc": ("topLongShortAccountRatio", "longShortRatio"),
              "top_pos": ("topLongShortPositionRatio", "longShortRatio"), "glob_acc": ("globalLongShortAccountRatio", "longShortRatio"),
              "taker": ("takerlongshortRatio", "buySellRatio")}
    cols = {}
    for key, (path, field) in ep_map.items():
        st, b, _ = http_get(f"https://fapi.binance.com/futures/data/{path}?symbol=BTCUSDT&period=5m&limit=500")
        if st == 200 and b:
            js = json.loads(b)
            cols[key] = pl.DataFrame({"ts": [datetime.fromtimestamp(x["timestamp"] / 1000, UTC) for x in js],
                                      key: [float(x[field]) for x in js]})
        time.sleep(0.3)
    if cols:
        rest = None
        for v in cols.values():
            rest = v if rest is None else rest.join(v, on="ts", how="full", coalesce=True)
        rest = rest.rename({"oi": "sum_open_interest", "top_acc": "count_toptrader_long_short_ratio",
                            "top_pos": "sum_toptrader_long_short_ratio", "glob_acc": "count_long_short_ratio",
                            "taker": "sum_taker_long_short_vol_ratio"})
        p = d / "bn_metrics.parquet"
        old = pl.read_parquet(p) if p.exists() else None
        rest = rest.with_columns(pl.col("ts").dt.cast_time_unit("us"))
        m = pl.concat([old, rest], how="diagonal_relaxed") if old is not None else rest
        m.unique("ts", keep="first").sort("ts").write_parquet(p)
    man.set("binance_rest_today", status="ok", note="REST 1m klines perp+spot, 5m metrics, as of fetch time")
    log("[NOW] binance REST today done")


# -------------------------------------------------------------------------------------------------------------- bybit


def trades_1m(path: Path) -> pl.DataFrame:
    """One Bybit public trade day -> 1m aggregates: taker buy/sell base and USD, prints, large prints (>= $250k, >= $1M)."""
    tmp = Path(str(path) + ".csv")
    with gzip.open(path, "rb") as fi, tmp.open("wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    try:
        df = (pl.scan_csv(tmp, schema_overrides={"timestamp": pl.Float64, "side": pl.Utf8, "size": pl.Float64, "price": pl.Float64})
              .select("timestamp", "side", "size", "price")
              .with_columns(pl.from_epoch((pl.col("timestamp") * 1000).round().cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC")
                            .dt.truncate("1m").alias("ts"),
                            (pl.col("size") * pl.col("price")).alias("usd"), (pl.col("side") == "Buy").alias("buy"))
              .group_by("ts")
              .agg(pl.col("size").filter(pl.col("buy")).sum().alias("buy_qty"), pl.col("size").filter(~pl.col("buy")).sum().alias("sell_qty"),
                   pl.col("usd").filter(pl.col("buy")).sum().alias("buy_usd"), pl.col("usd").filter(~pl.col("buy")).sum().alias("sell_usd"),
                   pl.len().alias("n"),
                   pl.col("usd").filter(pl.col("buy") & (pl.col("usd") >= 250e3)).sum().alias("lg_buy_usd"),
                   pl.col("usd").filter(~pl.col("buy") & (pl.col("usd") >= 250e3)).sum().alias("lg_sell_usd"),
                   (pl.col("buy") & (pl.col("usd") >= 1e6)).sum().alias("n_1m_buy"), (~pl.col("buy") & (pl.col("usd") >= 1e6)).sum().alias("n_1m_sell"),
                   pl.col("usd").max().alias("max_usd"), pl.col("price").last().alias("close"), pl.col("price").min().alias("low"),
                   pl.col("price").max().alias("high"))
              .sort("ts").collect())
    finally:
        tmp.unlink(missing_ok=True)
    return df


def ob200_1m(path: Path) -> pl.DataFrame:
    from ai_trader.sim.book import Book, book_metrics, iter_messages

    book, rows, nxt = Book(), [], None
    for msg in iter_messages(path):
        d = msg.get("data", {})
        if not book.apply(msg.get("type", "delta"), d.get("b", ()), d.get("a", ())):
            continue
        ts = int(msg["ts"])
        if nxt is None:
            nxt = (ts // 60_000 + 1) * 60_000
        if ts >= nxt:
            bp, bs, ap, as_ = book.arrays()
            mtr = book_metrics(bp, bs, ap, as_, bands=(5, 10, 25), imb=(5,))
            rows.append((nxt - 60_000, *mtr, float((bp * bs).sum()), float((ap * as_).sum())))
            nxt = (ts // 60_000 + 1) * 60_000
    cols = ["ms", "bid1", "ask1", "bid1_sz", "ask1_sz", "mid", "spread_bp", "dep_b5", "dep_a5", "dep_b10", "dep_a10", "dep_b25", "dep_a25",
            "imb5", "span_bid_bp", "span_ask_bp", "dep_b_all", "dep_a_all"]
    df = pl.DataFrame(rows, schema=cols, orient="row")
    return df.with_columns(pl.from_epoch("ms", time_unit="ms").dt.replace_time_zone("UTC").alias("ts")).drop("ms")


def fetch_bybit(ep: str, man: Manifest, skip_ob: bool = False, log=print) -> None:
    T = ep_T(ep)
    d = OUT / ep
    d.mkdir(parents=True, exist_ok=True)
    tdays = days(T - timedelta(days=2 if ep == "NOW" else 1), T + timedelta(days=1))
    frames = []
    for day in tdays:
        dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
        if dt > TODAY - timedelta(days=1):
            continue
        url = f"https://public.bybit.com/trading/BTCUSDT/BTCUSDT{day}.csv.gz"
        cache = d / "cache" / "bybit_trades" / f"{day}.parquet"
        if man.done(url):
            if cache.exists():
                frames.append(pl.read_parquet(cache))
            continue
        raw = RAW / f"BTCUSDT{day}.csv.gz"
        t0 = time.time()
        st, _, n = http_get(url, dest=raw)
        if st == 200:
            try:
                f = trades_1m(raw)
                cache.parent.mkdir(parents=True, exist_ok=True)
                f.write_parquet(cache)
                frames.append(f)
                man.set(url, status="ok", bytes=n, out=str(cache.relative_to(ROOT)), secs=round(time.time() - t0))
            except Exception as e:  # noqa: BLE001
                man.set(url, status="error", err=repr(e)[:200])
        else:
            man.set(url, status="missing" if st == 404 else "error", http=st)
        raw.unlink(missing_ok=True)
        log(f"[{ep}] bybit trades {day}: {st} {n / 1e6:.0f} MB {time.time() - t0:.0f}s")
    if frames:
        pl.concat(frames).sort("ts").write_parquet(d / "bybit_trades_1m.parquet")

    if not skip_ob and T >= datetime(2023, 1, 1, tzinfo=UTC):
        day = T.strftime("%Y-%m-%d")
        if ep == "NOW":
            day = "2026-11-14"
        url = f"https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/{day}_BTCUSDT_ob200.data.zip"
        cache = d / "cache" / "ob200" / f"{day}.parquet"
        if not man.done(url):
            raw = RAW / f"{day}_BTCUSDT_ob200.data.zip"
            t0 = time.time()
            st, _, n = http_get(url, dest=raw)
            if st == 200:
                try:
                    f = ob200_1m(raw)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    f.write_parquet(cache)
                    man.set(url, status="ok", bytes=n, out=str(cache.relative_to(ROOT)), secs=round(time.time() - t0))
                except Exception as e:  # noqa: BLE001
                    man.set(url, status="error", err=repr(e)[:200])
            else:
                man.set(url, status="missing" if st == 404 else "error", http=st)
            raw.unlink(missing_ok=True)
            log(f"[{ep}] ob200 {day}: {st} {n / 1e6:.0f} MB {time.time() - t0:.0f}s")
        if cache.exists():
            shutil.copy(cache, d / "bybit_ob200_1m.parquet")

    # REST 1h open interest T-10d..T+7d (Bybit v5, public)
    key = f"bybit_oi_1h:{ep}"
    if not man.done(key):
        rows, end = [], int(min(T + timedelta(days=7), datetime.now(UTC)).timestamp() * 1000)
        start = int((T - timedelta(days=10)).timestamp() * 1000)
        while end > start:
            st, b, _ = http_get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1h"
                                f"&startTime={start}&endTime={end}&limit=200")
            if st != 200 or not b:
                break
            lst = json.loads(b).get("result", {}).get("list", [])
            if not lst:
                break
            rows += lst
            end = min(int(x["timestamp"]) for x in lst) - 1
            time.sleep(0.3)
        if rows:
            pl.DataFrame({"ts": [datetime.fromtimestamp(int(x["timestamp"]) / 1000, UTC) for x in rows],
                          "oi": [float(x["openInterest"]) for x in rows]}).unique("ts").sort("ts").write_parquet(d / "bybit_oi_1h.parquet")
        man.set(key, status="ok" if rows else "missing", n=len(rows))


def fetch_okx(ep: str, man: Manifest, log=print) -> None:
    T = ep_T(ep)
    key = f"okx_1h:{ep}"
    if man.done(key):
        return
    rows, after = [], int(min(T + timedelta(days=7, hours=1), datetime.now(UTC)).timestamp() * 1000)
    start = int((T - timedelta(days=3)).timestamp() * 1000)
    while after > start:
        st, b, _ = http_get(f"https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar=1H&after={after}&limit=100")
        if st != 200 or not b:
            break
        js = json.loads(b).get("data", [])
        if not js:
            break
        rows += js
        after = min(int(x[0]) for x in js)
        time.sleep(0.25)
    if rows:
        df = pl.DataFrame({"ts": [datetime.fromtimestamp(int(x[0]) / 1000, UTC) for x in rows], "open": [float(x[1]) for x in rows],
                           "high": [float(x[2]) for x in rows], "low": [float(x[3]) for x in rows], "close": [float(x[4]) for x in rows],
                           "vol_btc": [float(x[6]) for x in rows]})
        df.filter(pl.col("ts") >= T - timedelta(days=3)).unique("ts").sort("ts").write_parquet(OUT / ep / "okx_1h.parquet")
    fr = []
    if ep == "NOW":
        st, b, _ = http_get("https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100")
        if st == 200 and b:
            fr = json.loads(b).get("data", [])
            if fr:
                pl.DataFrame({"ts": [datetime.fromtimestamp(int(x["fundingTime"]) / 1000, UTC) for x in fr],
                              "rate": [float(x["realizedRate"] or x["fundingRate"]) for x in fr]}).sort("ts").write_parquet(OUT / ep / "okx_funding.parquet")
    man.set(key, status="ok" if rows else "missing", n=len(rows), funding=len(fr))
    log(f"[{ep}] okx 1h {len(rows)} funding {len(fr)}")


# --------------------------------------------------------------------------------------------------------------- lake


def lake_extract(ep: str, log=print) -> None:
    """Bybit funding / 4h OI / 1h bars (T-35d..T+30d), Deribit DVOL, Fear & Greed; NOW also flow_3s (1m) and liquidations."""
    from ai_trader.storage.lake import Lake

    lk = Lake()
    T = ep_T(ep)
    d = OUT / ep
    d.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(UTC)
    a, b = T - timedelta(days=95), T + timedelta(days=31)
    f = lk.read_asof("funding", as_of, symbols=["BTCUSDT"], filters={"venue": "bybit"}, start=a, end=b)
    f.select("event_ts", "rate", "known_ts").sort("event_ts").write_parquet(d / "bybit_funding.parquet")
    o = lk.read_asof("open_interest", as_of, symbols=["BTCUSDT"], filters={"venue": "bybit", "interval": "4h"}, start=a, end=b)
    o.select("event_ts", "open_interest", "known_ts").sort("event_ts").write_parquet(d / "bybit_oi_4h.parquet")
    bars = lk.read_asof("bars", as_of, symbols=["BTCUSDT"], filters={"venue": "bybit", "freq": "1h"}, start=T - timedelta(days=35), end=b)
    if "price_type" in bars.columns:
        bars = bars.filter(pl.col("price_type") == "last")
    bars.select("event_ts", "open", "high", "low", "close", "volume").sort("event_ts").unique("event_ts").write_parquet(d / "bybit_1h.parquet")
    dv = lk.read_asof("alt_series", as_of, filters={"source": "deribit", "series_id": "dvol_close", "entity": "BTC"},
                      start=T - timedelta(days=95), end=b)
    dv.select("event_ts", "value").sort("event_ts").write_parquet(d / "dvol_1h.parquet")
    fg = lk.read_asof("alt_series", as_of, filters={"source": "alternative_me", "series_id": "fear_greed"}, start=T - timedelta(days=40), end=b)
    fg.select("event_ts", "value").sort("event_ts").write_parquet(d / "fng.parquet")
    if ep == "NOW":
        lq = lk.read_asof("liquidations", as_of, symbols=["BTCUSDT"], filters={"venue": "bybit"}, start=T - timedelta(days=7))
        lq.select("event_ts", "side", "price", "size").sort("event_ts").write_parquet(d / "bybit_liq.parquet")
        import duckdb

        g = (ROOT / "data" / "lake" / "flow_3s" / "venue=bybit" / "category=linear").as_posix()
        con = duckdb.connect()
        con.execute("SET threads TO 2")
        con.execute("SET memory_limit='1GB'")
        q = f"""SELECT time_bucket(INTERVAL 1 minute, event_ts) AS ts, sum(buy_vol) buy_qty, sum(sell_vol) sell_qty, sum(buy_notional) buy_usd,
                 sum(sell_notional) sell_usd, sum(n_buy + n_sell) n, max(max_trade) max_trade, last(close ORDER BY event_ts) close,
                 min(low) low, max(high) high, avg(depth_bid_25) dep_b25, avg(depth_ask_25) dep_a25, avg(depth_bid_100) dep_b100,
                 avg(depth_ask_100) dep_a100, avg(book_span_bps) span_bp, avg(spread_bps) spread_bp, avg(obi_25) obi25, avg(book_cov) cov
               FROM read_parquet('{g}/date=*/*.parquet', hive_partitioning=true, union_by_name=true)
               WHERE symbol='BTCUSDT' AND event_ts >= TIMESTAMP '2026-11-10' GROUP BY 1 ORDER BY 1"""
        con.sql(q).pl().with_columns(pl.col("ts").dt.replace_time_zone("UTC") if True else pl.col("ts")).write_parquet(d / "bybit_flow_1m.parquet")
    log(f"[{ep}] lake done: funding {f.height} oi4h {o.height} bars {bars.height} dvol {dv.height} fng {fg.height}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "lake"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-ob", action="store_true")
    ap.add_argument("--skip-bybit", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    man = Manifest()
    eps = a.only or [e[0] for e in EPISODES]

    def log(s: str) -> None:
        print(f"{datetime.now(UTC):%H:%M:%S} {s}", flush=True)

    for ep in eps:
        if a.cmd == "lake":
            lake_extract(ep, log)
            continue
        fetch_binance(ep, man, log)
        if ep == "NOW":
            fetch_binance_today(man, log)
        fetch_okx(ep, man, log)
        if not a.skip_bybit:
            fetch_bybit(ep, man, skip_ob=a.skip_ob, log=log)
        lake_extract(ep, log)
    shutil.rmtree(RAW, ignore_errors=True)


if __name__ == "__main__":
    main()
