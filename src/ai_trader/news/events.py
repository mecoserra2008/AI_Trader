"""Scheduled macro and crypto events with point-in-time timestamps.

Each event row: kind, name, scheduled_ts (UTC; when the information is released), schedule_known_ts (when the date was
public; a "this event is tomorrow" flag may be used only at or after it) and source. The "event happened" flag is known
at scheduled_ts. Unscheduled events (emergency FOMC meetings, court rulings) get schedule_known_ts = scheduled_ts, so
they can never appear as an advance flag.

Sources
  FOMC   federalreserve.gov/monetarypolicy/fomccalendars.htm (2021-2027) and fomchistorical<year>.htm (2017-2020),
         fetched and parsed live; statement at 14:00 ET on the last meeting day. Notation votes and cancelled meetings
         dropped; unscheduled meetings (2020-03-03 announcement, 2020-03-15) kept as unscheduled.
  CPI    BLS release schedule, 08:30 ET. 2026 dates: fedratecalc.com/cpi-release-date (BLS schedule, fetched
         2026-11-13). 2020-2025: BLS archived release dates as recorded here (FRED/ALFRED were unreachable from this host
         on 2026-11-13 and the BLS site refuses scripted access); spot checks against the repo: 2025-12-18 (delayed
         September CPI after the 2025 shutdown), 2026-11-10. The October 2025 CPI was cancelled (shutdown).
  NFP    BLS Employment Situation, 08:30 ET. 2026: fedratecalc.com/nonfarm-payrolls-release-date (fetched 2026-11-13).
         2020-2025 as recorded here; 2025-09-05 matches the repo's notes; the September 2025 report came out 2025-11-20
         and October + November 2025 together on 2025-12-16 (shutdown).
  Halving  block heights 630,000 (2020-05-11 19:23 UTC) and 840,000 (2024-04-20 00:09 UTC); 420,000 (2016-07-09
         16:46 UTC). Advance flag allowed 7 days before (block-time estimates are good to about a day by then).
  ETF / regulatory  SEC final deadlines and launch dates, from SEC press releases and issuer announcements: 2021-10-19
         first US bitcoin futures ETF (BITO) launch; 2024-01-10 SEC approves spot bitcoin ETP listings (statement ~21:00
         UTC, final deadline for the ARK 21Shares 19b-4); 2024-01-11 spot bitcoin ETFs begin trading; 2024-05-23 SEC
         approves spot ether ETF 19b-4s (VanEck final deadline); 2024-07-23 spot ether ETFs begin trading. No scheduled
         crypto-regulatory decision is recorded for 2025-2026 (none known with a reliable date): the holdout has only
         macro events.
Advance windows (schedule_known_ts = scheduled_ts - lead): CPI / NFP / FOMC 30 days (annual schedules are published
months ahead), ETF deadlines 14 days, halving 7 days, trading start after approval = the approval time.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import UTC, date, datetime, timedelta

import polars as pl

from ai_trader.news.store import Manifest

CPI = """
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14 2026-11-10 2026-12-10
"""
NFP = """
2020-01-10 2020-02-07 2020-03-06 2020-04-03 2020-05-08 2020-06-05 2020-07-02 2020-08-07 2020-09-04 2020-10-02 2020-11-06 2020-12-04
2021-01-08 2021-02-05 2021-03-05 2021-04-02 2021-05-07 2021-06-04 2021-07-02 2021-08-06 2021-09-03 2021-10-08 2021-11-05 2021-12-03
2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03 2022-07-08 2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02
2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02 2026-11-06 2026-12-04
"""
# Delayed releases (government shutdowns): the date became public only shortly before; lead 7 days instead of 30.
LATE_SCHEDULED = {date(2025, 10, 24), date(2025, 11, 20), date(2025, 12, 16), date(2025, 12, 18), date(2026, 2, 11), date(2026, 2, 13)}
HALVINGS = [("halving 420000", datetime(2016, 7, 9, 16, 46, tzinfo=UTC)), ("halving 630000", datetime(2020, 5, 11, 19, 23, tzinfo=UTC)),
            ("halving 840000", datetime(2024, 4, 20, 0, 9, tzinfo=UTC))]
ETF = [  # (name, scheduled_ts, schedule_known_ts or lead)
    ("BITO bitcoin futures ETF launch", datetime(2021, 10, 19, 13, 30, tzinfo=UTC), datetime(2021, 10, 18, 12, 0, tzinfo=UTC)),
    ("SEC spot bitcoin ETP approval deadline", datetime(2024, 1, 10, 21, 0, tzinfo=UTC), timedelta(days=14)),
    ("spot bitcoin ETFs begin trading", datetime(2024, 1, 11, 14, 30, tzinfo=UTC), datetime(2024, 1, 10, 21, 0, tzinfo=UTC)),
    ("SEC spot ether ETF 19b-4 deadline", datetime(2024, 5, 23, 20, 0, tzinfo=UTC), timedelta(days=14)),
    ("spot ether ETFs begin trading", datetime(2024, 7, 23, 13, 30, tzinfo=UTC), datetime(2024, 7, 22, 22, 0, tzinfo=UTC)),
]
FED_PAGES = ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"] + [
    f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm" for y in (2017, 2018, 2019, 2020)]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September",
                                      "October", "November", "December"], 1)}


def _nth_sunday(y: int, m: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def et_to_utc(d: date, hh: int, mm: int) -> datetime:
    """US Eastern wall time to UTC (DST from the 2nd Sunday of March to the 1st Sunday of November, rules since 2007)."""
    dst = _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=UTC) + timedelta(hours=4 if dst else 5)


def _dates(block: str) -> list[date]:
    return [date.fromisoformat(x) for x in block.split()]


def parse_fomc_current(html: str) -> list[tuple[date, str]]:
    """(decision day, note) from fomccalendars.htm. Note: '' scheduled, 'notation' etc."""
    out = []
    for m in re.finditer(r"(\d{4}) FOMC Meetings", html):
        y = int(m.group(1))
        seg = html[m.end():m.end() + 60000]
        nx = re.search(r"\d{4} FOMC Meetings", seg)
        seg = seg[:nx.start()] if nx else seg
        mons = re.findall(r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>", seg)
        days = re.findall(r"fomc-meeting__date[^>]*>([^<]+)<", seg)
        for mo, dd in zip(mons, days, strict=False):
            note = "notation" if "notation" in dd.lower() else "cancelled" if "cancel" in dd.lower() else "unscheduled" if "unscheduled" in dd.lower() else ""
            last_mon = mo.split("/")[-1].strip()
            mnum = MONTHS.get(last_mon) or next((v for k, v in MONTHS.items() if k.startswith(last_mon[:3])), None)
            dm = re.findall(r"\d+", dd)
            if mnum is None or not dm:
                continue
            out.append((date(y, mnum, int(dm[-1])), note))
    return out


def parse_fomc_historical(html: str) -> list[tuple[date, str]]:
    out = []
    for h in re.findall(r"<h5[^>]*>([^<]+)</h5>", html):
        m = re.match(r"\s*([A-Za-z]+)(?:/([A-Za-z]+))?\s+([\d\-]+)(?:\s*\(([^)]+)\))?\s*(?:Meeting)?\s*-\s*(\d{4})", h)
        if not m:
            continue
        mon = m.group(2) or m.group(1)
        mnum = MONTHS.get(mon) or next((v for k, v in MONTHS.items() if k.startswith(mon[:3])), None)
        if mnum is None:
            continue
        note = (m.group(4) or "").lower()
        note = "notation" if "notation" in note else "cancelled" if "cancel" in note else "unscheduled" if "unscheduled" in note else ""
        out.append((date(int(m.group(5)), mnum, int(m.group(3).split("-")[-1])), note))
    return out


def fetch_fomc(fetch=None) -> list[tuple[date, str]]:
    def _get(u):
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "AI_Trader research"}), timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    get = fetch or _get
    rows: dict[date, str] = {}
    for u in FED_PAGES:
        html = get(u)
        for d, note in (parse_fomc_current(html) if u.endswith("fomccalendars.htm") else parse_fomc_historical(html)):
            rows.setdefault(d, note)
    return sorted(rows.items())


def build(fomc: list[tuple[date, str]]) -> pl.DataFrame:
    rows = []
    for d, note in fomc:
        if note in ("notation", "cancelled"):
            continue
        ts = et_to_utc(d, 14, 0)
        if note == "unscheduled":
            # 2020-03-03 statement came 10:00 ET on 2020-03-03 (meeting listed as March 2); 2020-03-15 Sunday 17:00 ET.
            ts = et_to_utc(d + timedelta(days=1), 10, 0) if d == date(2020, 3, 2) else et_to_utc(d, 17, 0)
            rows.append(("fomc_unscheduled", "FOMC unscheduled", ts, ts, "federalreserve.gov"))
        else:
            rows.append(("fomc", "FOMC decision", ts, ts - timedelta(days=30), "federalreserve.gov"))
    for kind, block, src in (("cpi", CPI, "BLS CPI schedule"), ("nfp", NFP, "BLS Employment Situation schedule")):
        for d in _dates(block):
            ts = et_to_utc(d, 8, 30)
            rows.append((kind, kind.upper(), ts, ts - timedelta(days=7 if d in LATE_SCHEDULED else 30), src))
    for name, ts in HALVINGS:
        rows.append(("halving", name, ts, ts - timedelta(days=7), "block height"))
    for name, ts, k in ETF:
        known = ts - k if isinstance(k, timedelta) else k
        rows.append(("etf_reg", name, ts, known, "SEC / issuer announcements"))
    df = pl.DataFrame(rows, schema={"kind": pl.Utf8, "name": pl.Utf8, "scheduled_ts": pl.Datetime("us", "UTC"),
                                    "schedule_known_ts": pl.Datetime("us", "UTC"), "source": pl.Utf8}, orient="row")
    return df.sort("scheduled_ts")


def load_or_build(man: Manifest, *, fetch=None, refresh: bool = False) -> pl.DataFrame:
    """Events table, cached under events/events.parquet (built once; `refresh` re-parses the Fed pages)."""
    rel = "events/events.parquet"
    if man.has("events/table") and not refresh and (man.root / rel).is_file():
        return pl.read_parquet(man.root / rel)
    df = build(fetch_fomc(fetch))
    (man.root / rel).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(man.root / rel)
    man.put("events/table", path=rel, rows=df.height, kinds=df["kind"].value_counts().to_dicts())
    return df


__all__ = ["build", "et_to_utc", "fetch_fomc", "load_or_build", "parse_fomc_current", "parse_fomc_historical"]
