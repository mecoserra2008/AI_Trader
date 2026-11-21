"""Weekend alt market making on replayed real Bybit order books (agent MMW, 2026-11-21; H-MMW1 .. H-MMW6).

Research only: no live orders, no Bybit trading or private calls (public archives and the public kline endpoint only).
The quoting code is the live `atx mm` code plus the weekend rules in crates/atx/src/mm.rs, run by `atx mm-replay`
(crates/atx/src/mm_replay.rs) on the real ob200 book (quote-saver.bycsi.com) and the real public trade tape
(public.bybit.com) of each symbol-day; BTCUSDT's public tape of the same day is a second input (H-MMW2 pull).

Pre-registered design: research/hypotheses.yaml block "Weekend alt market making" (H-MMW1-6); universe
research/mm_weekend/universe.json (a pre-declared rule, run once 2026-11-21 00:55 UTC); split, grid and every rule that
is not in the registration are frozen in data/research/mm_weekend/split.json before any P&L is computed.

Pipeline (day-major, checkpointed in data/research/mm_weekend/manifest.json):
  1. per day: BTCUSDT trades (and, for a Saturday, Friday's for the BTC sigma warm-up) -> BTC 1m bars (`--bars-only`);
  2. per symbol-day: storage check, warm-up bars of the two previous days (replays of earlier sample days, else
     `--bars-only` on a downloaded trade archive that is deleted afterwards), the day's trades and ob200 archive, the
     day's own 1m bars (for the point-in-time H-MMW5 gate), structural zones as of 00:00 UTC (4h klines, scripts/levels.py
     logic), the liquidation-proxy floor (10 x the previous day's median print notional), the gate CSV;
  3. `atx mm-replay` with the frozen grid (every variant x {pro-rata, FIFO-pessimistic} x {150, 300} ms) -> stats.json,
     quotes.json, fills.jsonl.gz, bars_1m.csv in data/research/mm_weekend/days/<symbol>/<date>/;
  4. verify (engine rejects 0, replayer checks equal, ledger identity, quoter count), delete the raw files.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.mm_weekend_study freeze      # split.json + diary (no P&L)
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.mm_weekend_study run [SYM:DAY,..]   # resumable
    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.mm_weekend_report
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.sim.ob_gen import Manifest, budget_gb, du_bytes, http_get, storage_gb, write_json

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "mm_weekend"
OUT = ROOT / "data" / "research" / "mm_weekend"
MANIFEST = OUT / "manifest.json"
SPLIT = OUT / "split.json"
LOG = OUT / "run.log"
# A private release build of atx (copied once after the last mm_replay change; target/debug/atx.exe is rebuilt by others).
ATX = OUT / "bin" / "atx-mmw.exe"
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
KLINE_URL = "https://api.bybit.com/v5/market/kline?category=linear&symbol={s}&interval=240&start={a}&end={b}&limit=1000"
UA = {"User-Agent": "Mozilla/5.0 (AI_Trader research; public archives)"}
GB = 1024**3
RAW_CAP_GB = 2.0

UNIVERSE = json.loads((ROOT / "research" / "mm_weekend" / "universe.json").read_text(encoding="utf-8"))["universe"]
WEEKENDS = ["2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07", "2026-11-14"]
FIT_WEEKENDS, HELD_WEEKENDS = WEEKENDS[:4], WEEKENDS[4:]


def _weekend_days(sat: str) -> list[str]:
    d = date.fromisoformat(sat)
    return [d.isoformat(), (d + timedelta(days=1)).isoformat()]


DAYS = [x for w in WEEKENDS for x in _weekend_days(w)]
FIT_DAYS = [x for w in FIT_WEEKENDS for x in _weekend_days(w)]
HELD_DAYS = [x for w in HELD_WEEKENDS for x in _weekend_days(w)]
WEEKEND_OF = {x: w for w in WEEKENDS for x in _weekend_days(w)}
QUEUES = ["prorata", "pessimistic"]
LATENCIES = [150, 300]
CELLS = [(q, lat) for q in QUEUES for lat in LATENCIES]
PRIMARY = ("prorata", 150)
BASE_CFG = "config/mm_illiq20.yaml"
B0 = (0.45, 0.90)  # mm_illiq20 ask / bid offsets (sigma units): the live width (R-MMR-2)
LIQ_PROXY = {"window_s": 30, "min_prints": 5, "min_levels": 3, "notional_x_median": 10.0, "pull_min": 60}


def ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


def day_ms(d: str) -> int:
    return ms(d + "T00:00:00Z")


def prev_day(d: str, k: int = 1) -> str:
    return (date.fromisoformat(d) - timedelta(days=k)).isoformat()


def log(msg: str) -> None:
    line = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# the frozen grid
# ----------------------------------------------------------------------------------------------------------------------


def _v(name: str, family: str, **kw: Any) -> dict[str, Any]:
    return {"name": name, "family": family, "config": BASE_CFG, "overrides": kw.pop("overrides", {}), "sigma_mode": "config",
            "queues": QUEUES, "latencies_ms": LATENCIES, **kw}


def width(k: float) -> dict[str, float]:
    return {"ask_offset_sigma": B0[0] * k, "bid_offset_sigma": B0[1] * k}


def base_grid() -> list[dict[str, Any]]:
    """Ungated variants. B = baselines (B0 = live mm_illiq20 width, B0h = 0.5 x B0); W1 = level ladder (A1, A2);
    W2 = BTC-move pull (width x k x cooldown); W3 = side pull (signal x threshold x action); W4 = short-queue touch."""
    out = [_v("B0", "B", overrides=width(1.0)), _v("B0h", "B", overrides=width(0.5))]
    lad = {"mode": "ladder", "ladder_band": [1.5, 3.5], "ladder_stop_sigma": 1.5, "ladder_max_per_side": 3}
    out.append(_v("W1_A1", "W1", **lad))
    out.append(_v("W1_A2", "W1", **lad, liq_pull_min=float(LIQ_PROXY["pull_min"])))
    for wn, k in (("B0", 1.0), ("B0h", 0.5)):
        for kk in (2, 3, 4):
            for c in (10, 30, 60):
                out.append(_v(f"W2_{wn}_k{kk}_c{c}", "W2", overrides=width(k), btc_k=float(kk), btc_cooldown_s=float(c)))
    for sig in ("imb5", "flow30"):
        for th in (0.3, 0.5, 0.7):
            for act, wd in (("pull", 0.0), ("widen", 1.0)):
                out.append(_v(f"W3_{sig}_t{th:g}_{act}", "W3", overrides=width(1.0), side_signal=sig, side_threshold=th, side_widen_sigma=wd))
    for q in (0.25, 0.5, 1.0):
        out.append(_v(f"W4_q{q:g}", "W4", overrides=width(1.0), mode="touch", touch_q=q, touch_leave_frac=0.5, touch_leave_ms=2000))
    return out


def grid() -> list[dict[str, Any]]:
    """Every ungated variant and its H-MMW5 gated copy (`<name>__G`, gate CSV per symbol-day, exit-only while off)."""
    out = []
    for v in base_grid():
        out.append(v)
        out.append({**v, "name": v["name"] + "__G", "family": v["family"], "gated": True})
    return out


def variants_json(sym: str, d: str) -> list[dict[str, Any]]:
    gate = str((OUT / "gates" / sym / f"{d}.csv").relative_to(ROOT)).replace("\\", "/")
    out = []
    for v in grid():
        x = {k: val for k, val in v.items() if k not in ("family", "gated")}
        if v.get("gated"):
            x["gate"] = gate
            x["gate_flatten"] = False
        out.append(x)
    return out


DESIGN = {
    "engine": "atx mm-replay (crates/atx/src/mm_replay.rs, private release build data/research/mm_weekend/bin/atx-mmw.exe): ob200 "
              "snapshot + deltas -> at_match::Ob200Replayer -> at_match::OrderBook; the symbol's public tape merged by time; BTCUSDT's "
              "public tape of the same day merged as a third stream (H-MMW2); the live atx mm decision code plus the weekend rules in "
              "crates/atx/src/mm.rs (ladder_levels, level_stop, ladder_tp, BtcTape / btc_move_z, top_imbalance, threatened_side, "
              "side_pull, short_queue, behind_thinned, LiqProxy); virtual minimum-lot orders (no impact)",
    "queue": "as H-MMR (agent M): join the back; prints at our price consume from the front; trades through or a crossing book fill "
             "us; unexplained decreases are cancels split pro-rata or FIFO-pessimistic (behind us first); RPI and off-grid prints do not fill",
    "cells": "every variant runs under queue {pro-rata, FIFO-pessimistic} x latency {150, 300} ms (order and cancel take effect one "
             "latency after they are sent; the stop and pull reactions too)",
    "fees": "config/fees.yaml crypto class (maker 2.0 bp, taker 5.5 bp)",
    "config": "config/mm_illiq20.yaml (live mm_illiq20) for every variant: caps (long 15 / short 25 USD after a fill, one lot from flat), "
              "per-symbol loss stop 1.5 USD, global loss limit 10 USD, shock pause, toxicity, refresh 10 s; levels_file zones are not "
              "used by the spread quoter (as in H-MMR); single-symbol replays (portfolio caps only in H-MMW6)",
    "B0": "the live mm_illiq20 width (ask 0.45, bid 0.90 x the 1h sigma = max(240 min, 30 min) robust sigma, floor 20 bp); B0h = 0.5 x B0",
    "W1_ladder": "structural zones of scripts/levels.py zones() (fractal 4h pivots k = 3 clustered at 0.6 x ATR(4h, median of 180 TRs), "
                 "min width 0.25 ATR, top 8 by score, half-life 30 d) from the last 720 Bybit 4h klines CLOSED by the day's 00:00 UTC "
                 "(public kline endpoint, cached in klines4h/); zones are fixed for the day. Quote prices: a support zone below the mid "
                 "at its top edge (bid), a resistance zone above at its bottom edge (ask), when that edge lies 1.5-3.5 x the 1h sigma "
                 "(log distance) from the mid; up to 3 levels per side, minimum lots, within the caps counting resting adds as filled "
                 "(one lot from flat always allowed). While holding a lot, the other side's adding orders are cancelled. After a fill: "
                 "a passive reduce-only take-profit at the mid at fill time on the tick grid, rounded away from the entry and never "
                 "less than one tick beyond the entry (a take-profit is never a loss; matters only for fills through our price), sent "
                 "at the fill; a stop 1.5 x sigma_risk beyond the fill level (sigma_risk = the quoter's 1h sigma, no weekend shrink, "
                 "R-VOL-2): the first print at or beyond it closes the lot at market (taker) one latency later",
    "W1_liquidation_proxy": "the lake's liquidation stream starts 2026-11-08 (it does not cover the FIT days), so arm A2 uses a "
                            "pre-declared public-tape proxy on the symbol's own tape: >= 5 same-side (same aggressor) prints within "
                            "30 s, spanning >= 3 distinct price levels, with summed notional >= 10 x the PREVIOUS UTC day's median "
                            "print notional (point in time); a trigger empties that side's window. A2 cancels its adding quotes at "
                            "the trigger print and adds nothing for 60 min after the latest trigger; take-profits and stops stay "
                            "(they only reduce)",
    "W2_btc_pull": "z = max(|r_1s| / (sigma_btc sqrt(1/3600)), |r_5s| / (sigma_btc sqrt(5/3600))), r_h = log return of the BTCUSDT last "
                   "price from the last print at or before t - h to the print at t, sigma_btc = robust 1h sigma (mm::sigma_1h) of the "
                   "240 closed BTC 1m bars before the minute (refreshed each minute; previous days' BTC bars as warm-up); z > k cancels "
                   "every quote (adding and reducing) at t and nothing is quoted until the cooldown c after the latest trigger",
    "W3_side_pull": "signal at each decision tick (10 s): imb5 = (bid qty - ask qty) / total over the top 5 levels; flow30 = signed taker "
                    "notional imbalance of the last 30 s; signal > +threshold threatens the ask, < -threshold the bid; 'pull' "
                    "withdraws that side's quote, 'widen' moves it 1 x the quote sigma further (both adding and reducing quotes)",
    "W4_touch": "the B0 rule's sides, sizes and reduce-only flags, priced at the touch (best bid / best ask); a resting quote stays "
                "while it is at the touch with the right size and kind; a new one is placed only when the others' size at the touch "
                "< q x the median touch size (samples of both touch sizes every 10 s over the previous 60 min); a resting touch "
                "quote is cancelled when the size behind it falls by > 50% within 2 s (queue-model dependent)",
    "W5_gate": "per minute t (closed bars only): J = BTC field P(jump in the next hour) (frozen pre-cutoff fit "
               "data/models/field/btc_field_cutoff_2026-07-01.json, row valid from its known_ts; the field is BTC-only, so it is the "
               "market-wide jump state for every alt); R = |ln(close_t / close_t-240m)| < sqrt(48) x sigma_5m (RMS of the 288 clock-5m "
               "returns of the 24 h before t, the symbol's own bars); S = no closed clock hour in (t - 6 h, t] with |r_1h| >= 4 x "
               "sqrt(12) x sigma_5m (as of that hour's start). The gate turns ON when J <= q33(J) and R and S, stays on while J <= "
               "median(J) and R and S, and is OFF otherwise (q33, median over the FIT-day hours). While OFF the quoter is exit-only "
               "(no new inventory, the live wind-down path). Applied to every variant (`__G` copies) so the gated best H-MMW1-4 "
               "variant needs no second run",
    "day": "each symbol-day starts flat at 00:00 UTC and is marked to the mid at 24:00 UTC; warm-up 1m bars of the two previous days",
    "pnl": "net = spread + adverse 60 s + inventory - fees (USD per symbol-day; the unit of every test); adverse 1 / 10 s reported",
}

SELECTION = {
    "family_candidates": {"H-MMW1": "W1_A1, W1_A2", "H-MMW2": "the 18 W2 variants", "H-MMW3": "the 12 W3 variants", "H-MMW4": "the 3 W4 variants"},
    "score": "FIT days only. Combinatorial purged CV over the 4 FIT weekends (blocks = weekends, Sat + Sun together): every choice of 2 "
             "test weekends (6 splits), the other 2 train (weekends are 5 days apart, so no extra purge). A variant's split statistic = "
             "the MINIMUM over the 4 cells (pro-rata / FIFO-pessimistic x 150 / 300 ms) of its test-weekend mean net per symbol-day "
             "(the pass rule needs all four cells); its score = the median over the 6 splits. Choice = the highest score; ties broken "
             "by grid order. One choice per family (H-MMW1: the arm)",
    "H-MMW5": "pairs (gated, ungated) for B0 and for the best H-MMW1-4 choice (the family choice with the highest CV score)",
    "held": "each family's choice is evaluated ONCE on the 8 HELD days (4 weekends, 7 symbols = 56 symbol-days)",
    "overfitting": "per family: PBO by CSCV over the 8 FIT days (one-day blocks, C(8, 4) = 70 splits, primary cell pro-rata 150 ms); "
                   "deflated Sharpe of the choice's daily net (mean over symbols per day) on FIT days, N = 74 trials (the whole grid)",
}

TESTS = {
    "unit": "net USD per symbol-day; mean over the held symbol-days; day-clustered bootstrap 95% CI (days resampled with all their "
            "symbol-days, B = 10000, seed 20261121)",
    "pass": "held mean > 0 with the CI above 0 in ALL FOUR cells (pro-rata and FIFO-pessimistic, 150 and 300 ms)",
    "H-MMW1": "supports iff the chosen arm passes; contradicts iff its held mean <= 0 with the CI below 0 in the primary cell (pro-rata, "
              "150 ms); else inconclusive. Also A2 - A1 paired per symbol-day (R-FLOW-2 for passive fills)",
    "H-MMW2": "as H-MMW1 for the chosen W2 variant; also paired difference vs the same width without the pull (B0 or B0h)",
    "H-MMW3": "as H-MMW1 for the chosen W3 variant; also paired difference vs B0",
    "H-MMW4": "as H-MMW1 for the chosen W4 variant (the FIFO-pessimistic cells are decisive)",
    "H-MMW5": "for each pair (B0, best): gated - ungated paired per symbol-day; supports iff for at least one pair gated - ungated > 0 "
              "with the CI above 0 AND the gated arm's held mean > 0, in all four cells; contradicts iff for both pairs the gated arm's "
              "held mean <= 0 with the CI below 0 in the primary cell; else inconclusive",
    "H-MMW6": "only if a variant passes H-MMW1-5: its fills on all 7 symbols merged on one clock with net 40 / gross 80 USD caps; an "
              "adding fill that would breach a cap (counting one resting minimum-lot adding order per quoting symbol and side as filled) "
              "is dropped with its later round trip; held mean net per day > 0 with a day-clustered CI above 0 supports; else not "
              "evaluated (recorded inconclusive)",
    "segment": "every day is after research.knowledge_cutoff (2026-07-01): clean_holdout",
}


def write_split(force: bool = False) -> dict[str, Any]:
    obj = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "MMW", "hypotheses": [f"H-MMW{i}" for i in range(1, 7)],
           "symbols": UNIVERSE, "weekends": WEEKENDS, "fit_weekends": FIT_WEEKENDS, "held_weekends": HELD_WEEKENDS, "days": DAYS,
           "fit_days": FIT_DAYS, "held_days": HELD_DAYS, "split_rule": "registered: FIT = the 4 older weekends, HELD = the 4 newer",
           "queues": QUEUES, "latencies_ms": LATENCIES, "primary_cell": list(PRIMARY), "base_config": BASE_CFG,
           "liquidation_proxy": LIQ_PROXY, "grid": grid(), "n_variants": len(grid()), "n_quoters_per_symbol_day": len(grid()) * len(CELLS),
           "design": DESIGN, "selection": SELECTION, "tests": TESTS, "amendments": []}
    if SPLIT.exists() and not force:
        old = json.loads(SPLIT.read_text(encoding="utf-8"))
        keep = ("written", "amendments")
        same = {k: v for k, v in old.items() if k not in keep} == json.loads(json.dumps({k: v for k, v in obj.items() if k not in keep}))
        if not same:
            raise RuntimeError("split.json exists and differs: the design is frozen (add an amendment instead)")
        return old
    write_json(SPLIT, obj)
    return obj


def freeze() -> None:
    """split.json and the diary entry, before any P&L."""
    if SPLIT.exists():
        write_split()
        print("split.json already frozen")
        return
    s = write_split()
    from ai_trader.research.diary import entry

    names = [v["name"] for v in base_grid()]
    body = (
        f"Agent MMW froze the H-MMW1-6 design in data/research/mm_weekend/split.json ({s['written']}) before downloading any "
        f"replay data or computing any P&L. Universe (pre-declared rule): {', '.join(UNIVERSE)}. FIT weekends {', '.join(FIT_WEEKENDS)}; "
        f"HELD {', '.join(HELD_WEEKENDS)} (Sat + Sun each; all clean_holdout). Cells: queue pro-rata / FIFO-pessimistic x latency "
        f"150 / 300 ms. Grid ({len(names)} ungated variants, each also gated for H-MMW5 = {len(grid())}): {', '.join(names)}.\n\n"
        "Liquidation-cluster proxy (H-MMW1 A2), declared before any result because the lake's liquidation stream starts 2026-11-08 "
        "and Bybit public trades carry no liquidation flag: >= 5 same-side prints within 30 s spanning >= 3 distinct price levels "
        "with summed notional >= 10 x the previous UTC day's median print notional; A2 adds nothing for 60 min after the latest "
        "trigger (take-profits and stops stay).\n\n"
        "Clarifications of the registration (not changes): ladder quotes sit at the zone edge nearest the mid; the take-profit is the "
        "mid at fill time rounded away from the entry and never less than one tick beyond it; sigma_risk = the quoter's unshrunk 1h "
        "sigma; the H-MMW5 'field P(jump)' is the BTC field (BTC-only) used as the market jump state for the alts, with on at the "
        "bottom tercile and off above the median (hysteresis), plus each alt's own 4h-range and 6 h shock conditions. Selection: "
        "purged CV over the FIT weekends, score = median over splits of the worst of the four cells. Pass: held mean > 0 with the "
        "day-clustered 95% CI (B = 10000) above 0 in all four cells."
    )
    entry("decision", "MMW design frozen before results", body, tags=["mm_weekend", "replay", "design"], hypotheses=[f"H-MMW{i}" for i in range(1, 7)],
          links=["data/research/mm_weekend/split.json", "research/hypotheses.yaml", "research/mm_weekend/universe.json"])
    print(f"froze split.json ({len(grid())} variants, {len(grid()) * len(CELLS)} quoters per symbol-day) and wrote the diary entry")


# ----------------------------------------------------------------------------------------------------------------------
# storage and downloads
# ----------------------------------------------------------------------------------------------------------------------

_storage_cache: dict[str, Any] = {"t": 0.0, "v": None}


def storage(refresh: bool = False) -> dict[str, float]:
    if refresh or _storage_cache["v"] is None or time.time() - _storage_cache["t"] > 300:
        _storage_cache["v"], _storage_cache["t"] = storage_gb(), time.time()
    return _storage_cache["v"]


def head_bytes(url: str) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001 - size unknown: the download itself reports
        return 0


def fetch(man: Manifest, key: str, url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    rec = man.get("downloads", key)
    if rec and rec.get("deleted"):
        log(f"{key}: downloaded and deleted before; downloading again (logged)")
        man.put("redownloads", f"{key}:{int(time.time())}", {"status": "done", "first": rec})
    need = head_bytes(url) / GB
    st = storage(refresh=True)
    raw_now = du_bytes(RAW) / GB
    if st["total_gb"] + need >= budget_gb() - 1.0:
        raise RuntimeError(f"storage {st['total_gb']:.2f} GB + {need:.2f} GB >= budget {budget_gb()} - 1 GB: no download")
    if raw_now + need > RAW_CAP_GB:
        raise RuntimeError(f"transient raw {raw_now:.2f} GB + {need:.2f} GB > {RAW_CAP_GB} GB")
    r = http_get(url, dest)
    man.put("downloads", key, {**r, "url": url, "path": str(dest.relative_to(ROOT)), "storage_before_gb": st["total_gb"], "raw_before_gb": raw_now})
    if r["status"] != "ok":
        raise RuntimeError(f"download {key}: {r}")
    return dest


def mark_deleted(man: Manifest, key: str) -> None:
    r = man.get("downloads", key) or {}
    man.put("downloads", key, {**r, "deleted": True})


def atx(args: list[str], timeout: int = 6 * 3600) -> str:
    r = subprocess.run([str(ATX), *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"atx {' '.join(args[:3])} failed: {r.stderr[-2000:]}")
    return r.stdout.strip()


def atx_grid(args: list[str], extra: list[str] | None = None) -> tuple[str, list[str]]:
    """Run atx; when a historical day's prices are off the lake's (latest) tick grid, infer the day's tick from the offending
    price's decimals and retry with --tick (logged). Returns (stdout, extra args used)."""
    extra = list(extra or [])
    for _ in range(3):
        try:
            return atx(args + extra), extra
        except RuntimeError as e:
            m = re.search(r"price ([0-9.]+) off the tick grid", str(e))
            if not m or "--tick" in extra:
                raise
            dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
            extra = ["--tick", f"{10.0 ** -dec:.{dec}f}"]
            log(f"tick grid differs from the lake rules on this day: retrying with {extra}")
    raise RuntimeError("tick inference failed")


# ----------------------------------------------------------------------------------------------------------------------
# warm-up bars, tape stats, BTC
# ----------------------------------------------------------------------------------------------------------------------


def bars_path(sym: str, d: str) -> Path:
    return OUT / "bars" / sym / f"{d}.csv"


def tape_path(sym: str, d: str) -> Path:
    return OUT / "tape" / sym / f"{d}.json"


def bars_only(man: Manifest, sym: str, d: str, tr: Path, grid_extra: list[str] | None = None) -> None:
    tmp = OUT / "tmp" / f"bars_{sym}_{d}"
    atx_grid(["mm-replay", "--symbol", sym, "--trades", str(tr), "--out", str(tmp), "--bars-only"], grid_extra)
    for src, dst in (("bars_1m.csv", bars_path(sym, d)), ("tape_stats.json", tape_path(sym, d))):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / src), dst)
    shutil.rmtree(tmp, ignore_errors=True)


def ensure_prev(man: Manifest, sym: str, d: str) -> Path | None:
    """Bars and tape stats of day d (a warm-up day): from the replay of d, else (days outside the sample) from its trade
    archive (--bars-only, raw deleted after). A sample day that failed gives None."""
    p = bars_path(sym, d)
    if p.is_file() and tape_path(sym, d).is_file():
        return p
    if d in DAYS:
        log(f"{sym}:{d}: no bars (sample day not processed): warm-up without it")
        return p if p.is_file() else None
    raw = RAW / sym / d
    tr = fetch(man, f"{sym}:{d}:trades", TRADES_URL.format(s=sym, d=d), raw / f"{sym}{d}.csv.gz")
    bars_only(man, sym, d, tr)
    nb = du_bytes(raw)
    shutil.rmtree(raw, ignore_errors=True)
    mark_deleted(man, f"{sym}:{d}:trades")
    man.put("bars", f"{sym}:{d}", {"status": "done", "source": "trades --bars-only", "raw_deleted_bytes": nb})
    return p


def ensure_btc(man: Manifest, d: str) -> Path:
    """BTCUSDT trades of day d (kept until the day's symbols are done) plus BTC 1m bars of d and of the previous day."""
    pd = prev_day(d)
    if not bars_path("BTCUSDT", pd).is_file():
        raw = RAW / "BTCUSDT" / pd
        tr = fetch(man, f"BTCUSDT:{pd}:trades", TRADES_URL.format(s="BTCUSDT", d=pd), raw / f"BTCUSDT{pd}.csv.gz")
        bars_only(man, "BTCUSDT", pd, tr)
        shutil.rmtree(raw, ignore_errors=True)
        mark_deleted(man, f"BTCUSDT:{pd}:trades")
    raw = RAW / "BTCUSDT" / d
    tr = fetch(man, f"BTCUSDT:{d}:trades", TRADES_URL.format(s="BTCUSDT", d=d), raw / f"BTCUSDT{d}.csv.gz")
    if not bars_path("BTCUSDT", d).is_file():
        bars_only(man, "BTCUSDT", d, tr)
    return tr


def drop_btc(man: Manifest, d: str) -> None:
    raw = RAW / "BTCUSDT" / d
    if raw.exists():
        shutil.rmtree(raw, ignore_errors=True)
        mark_deleted(man, f"BTCUSDT:{d}:trades")


# ----------------------------------------------------------------------------------------------------------------------
# structural zones (H-MMW1)
# ----------------------------------------------------------------------------------------------------------------------


def _levels_module():
    spec = importlib.util.spec_from_file_location("levels_script", ROOT / "scripts" / "levels.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def klines4h(sym: str) -> list[tuple[int, float, float, float, float]]:
    """Bybit public 4h klines 2026-05-01 .. 2026-11-17 (cached). Rows (start_ms, o, h, l, c), ascending."""
    p = OUT / "klines4h" / f"{sym}.json"
    if p.is_file():
        return [tuple(x) for x in json.loads(p.read_text(encoding="utf-8"))]
    a, end = ms("2026-05-01T00:00:00Z"), ms("2026-11-17T00:00:00Z")
    rows: dict[int, tuple] = {}
    while a < end:
        b = min(end, a + 999 * 4 * 3_600_000)
        u = KLINE_URL.format(s=sym, a=a, b=b)
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30))
        for x in r["result"]["list"]:
            rows[int(x[0])] = (int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
        a = b + 4 * 3_600_000
        time.sleep(0.2)
    out = sorted(rows.values())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


def zones_file(sym: str, d: str) -> Path:
    """Zones as of d 00:00 UTC from the last 720 4h klines closed by then (scripts/levels.py zones())."""
    p = OUT / "zones" / sym / f"{d}.json"
    if p.is_file():
        return p
    t0 = day_ms(d)
    k = [x for x in klines4h(sym) if x[0] + 4 * 3_600_000 <= t0][-720:]
    lv = _levels_module()
    ts = [datetime.fromtimestamp(x[0] / 1000, UTC) for x in k]
    a = np.array([x[1:] for x in k], dtype=float)
    z, atr = lv.zones(ts, a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, {sym: z, "_meta": {"as_of": f"{d}T00:00:00Z", "bars": len(k), "first": ts[0].isoformat() if ts else None,
                                     "last_close": float(a[-1, 3]) if len(a) else None, "atr_4h": atr}})
    return p


# ----------------------------------------------------------------------------------------------------------------------
# H-MMW5 gate
# ----------------------------------------------------------------------------------------------------------------------

_PJ: dict[str, Any] = {}


def field_pjump():
    """BTC P(jump in the next hour) from the frozen pre-cutoff field fit, per 1h row valid from its known_ts; FIT-hour q33 / median."""
    if "pj" in _PJ:
        return _PJ["pj"], _PJ["q33"], _PJ["med"]
    import polars as pl

    from ai_trader.field.model import Field, coords_matrix
    from ai_trader.field.state import build_state, load_frame

    cache = OUT / "gates" / "pjump.csv"
    if cache.is_file():
        pj = pl.read_csv(cache)
    else:
        fld = Field.load(ROOT / "data" / "models" / "field" / "btc_field_cutoff_2026-07-01.json")
        fr = load_frame(datetime(2026, 11, 17, tzinfo=UTC), start=datetime(2026, 4, 1, tzinfo=UTC))
        st = build_state(fr, fld.profile, fld.cfg)
        X = coords_matrix(st)
        good = np.all(np.isfinite(X), axis=1) & np.isfinite(st["vf"].to_numpy())
        st, X = st.filter(pl.Series(good)), X[good]
        _, _, p = fld.params(X, np.log(st["vf"].to_numpy()))
        pj = pl.DataFrame({"valid_from_ms": st["known_ts"].dt.epoch("ms").cast(pl.Int64), "p_jump": p}).sort("valid_from_ms")
        cache.parent.mkdir(parents=True, exist_ok=True)
        pj.write_csv(cache)
    fit_ms = [(day_ms(d), day_ms(d) + 86_400_000) for d in FIT_DAYS]
    fv = pj.filter(pl.any_horizontal([(pl.col("valid_from_ms") >= a) & (pl.col("valid_from_ms") < b) for a, b in fit_ms]))["p_jump"]
    q33, med = float(fv.quantile(1 / 3, "nearest")), float(fv.median())
    _PJ.update(pj=pj, q33=q33, med=med, n_fit_hours=fv.len())
    return pj, q33, med


def gate_minutes(sym: str, d: str) -> dict[str, Any]:
    """Per-minute gate of symbol-day (sym, d) from the 1m bars of d-2 .. d (closed minutes before t only) and the BTC field."""
    import polars as pl

    paths = [bars_path(sym, x) for x in (prev_day(d, 2), prev_day(d), d)]
    dfs = [pl.read_csv(p, schema_overrides={"minute_ms": pl.Int64}) for p in paths if p.is_file()]
    df = pl.concat(dfs).unique("minute_ms").sort("minute_ms")
    lo, hi = int(df["minute_ms"].min()), day_ms(d) + 86_400_000 - 60_000
    g = pl.DataFrame({"minute_ms": np.arange(lo, hi + 60_000, 60_000, dtype=np.int64)})
    g = g.join(df.select("minute_ms", "close"), on="minute_ms", how="left").with_columns(pl.col("close").forward_fill())
    t, c = g["minute_ms"].to_numpy(), g["close"].to_numpy().astype(float)
    n = len(t)
    idx = np.arange(n)
    lc = np.log(np.concatenate([[c[0]], c[:-1]]))  # the close known at t[i] is minute i-1's close
    bidx = np.where(t % 300_000 == 0)[0]
    r5 = np.zeros(len(bidx))
    ok = bidx >= 5
    r5[ok] = lc[bidx[ok]] - lc[bidx[ok] - 5]
    c5 = np.concatenate([[0.0], np.cumsum(r5**2)])
    nb = np.searchsorted(bidx, idx, side="right")
    cnt = np.minimum(nb, 288)
    s5 = np.sqrt((c5[nb] - c5[nb - cnt]) / np.maximum(cnt, 1))
    s5[cnt < 288] = np.nan
    s4h = np.sqrt(48.0) * s5
    r4h = np.full(n, np.nan)
    r4h[240:] = lc[240:] - lc[:-240]
    shocks = []
    for k in np.where(t % 3_600_000 == 0)[0]:
        if k < 60:
            continue
        sig = np.sqrt(12.0) * s5[k - 60]
        if np.isfinite(sig) and sig > 0 and abs(lc[k] - lc[k - 60]) >= 4.0 * sig:
            shocks.append(int(t[k]))
    sc = np.array(shocks, dtype=np.int64)
    last_shock = np.full(n, -(10**15), dtype=np.int64)
    if len(sc):
        j = np.searchsorted(sc, t, side="right") - 1
        last_shock = np.where(j >= 0, sc[np.maximum(j, 0)], -(10**15))
    S = (t - last_shock) >= 6 * 3_600_000
    with np.errstate(invalid="ignore"):
        R = np.abs(r4h) < s4h
    warm = np.isfinite(s4h) & np.isfinite(r4h)
    pj, q33, med = field_pjump()
    pjv = pj["valid_from_ms"].to_numpy()
    k = np.searchsorted(pjv, t, side="right") - 1
    P = np.where(k >= 0, pj["p_jump"].to_numpy()[np.maximum(k, 0)], np.nan)
    on = np.zeros(n, bool)
    state = False
    for i in range(n):
        base = bool(warm[i] and R[i] and S[i] and np.isfinite(P[i]))
        state = base and (P[i] <= med if state else P[i] <= q33)
        on[i] = state
    sel = (t >= day_ms(d)) & (t < day_ms(d) + 86_400_000)
    assert sel.sum() == 1440, (sym, d, int(sel.sum()))
    return {"minute_ms": t[sel], "on": on[sel], "R": R[sel], "S": S[sel], "P": P[sel], "warm": warm[sel], "q33": q33, "median": med, "shocks": [int(x) for x in sc]}


def write_gate(sym: str, d: str) -> dict[str, Any]:
    gm = gate_minutes(sym, d)
    p = OUT / "gates" / sym / f"{d}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("valid_from_ms,on\n" + "".join(f"{a},{int(b)}\n" for a, b in zip(gm["minute_ms"], gm["on"], strict=True)), encoding="utf-8")
    info = {"on_share": float(np.mean(gm["on"])), "R_share": float(np.mean(gm["R"])), "S_share": float(np.mean(gm["S"])),
            "P_le_q33_share": float(np.nanmean(gm["P"] <= gm["q33"])), "warm_share": float(np.mean(gm["warm"])), "q33": gm["q33"], "median": gm["median"]}
    write_json(OUT / "gates" / sym / f"{d}.json", info)
    return info


# ----------------------------------------------------------------------------------------------------------------------
# replay runs
# ----------------------------------------------------------------------------------------------------------------------


def verify(out: Path, n_quoters: int) -> list[str]:
    errs = []
    st = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    mk = st["market"]
    if mk["book_messages"] <= 0 or mk["trades"] <= 0:
        errs.append("empty market")
    if mk["engine_rejects"] or mk["engine_trades"]:
        errs.append(f"engine rejects {mk['engine_rejects']} trades {mk['engine_trades']}")
    if mk["replayer_check_mismatches"]:
        errs.append(f"replayer check mismatches {mk['replayer_check_mismatches']}")
    if len(st["quoters"]) != n_quoters:
        errs.append(f"{len(st['quoters'])} quoters, expected {n_quoters}")
    bad = [q["name"] for q in st["quoters"] if abs(q["pnl"]["ledger_check"]) > 1e-6]
    if bad:
        errs.append(f"ledger identity broken: {bad[:3]}")
    if mk.get("btc_prints", 0) <= 0:
        errs.append("no BTC prints")
    if not (out / "bars_1m.csv").is_file() or sum(1 for _ in (out / "bars_1m.csv").open()) < 100:
        errs.append("bars_1m.csv missing or short")
    return errs


def gzip_fills(out: Path) -> None:
    f = out / "fills.jsonl"
    if f.is_file():
        with f.open("rb") as a, gzip.open(out / "fills.jsonl.gz", "wb") as b:
            shutil.copyfileobj(a, b)
        f.unlink()


def process(man: Manifest, sym: str, d: str, btc_tr: Path) -> None:
    key = f"{sym}:{d}"
    if man.done("days", key):
        return
    t0 = time.time()
    raw = RAW / sym / d
    prevs = [prev_day(d, 2), prev_day(d)]
    warm = [w for w in (ensure_prev(man, sym, x) for x in prevs) if w is not None]
    tr = fetch(man, f"{key}:trades", TRADES_URL.format(s=sym, d=d), raw / f"{sym}{d}.csv.gz")
    ob = fetch(man, f"{key}:ob200", OB_URL.format(s=sym, d=d), raw / f"{d}_{sym}_ob200.data.zip")
    raw_bytes = du_bytes(raw)
    # the day's own 1m bars (the gate uses closed minutes before t only)
    tmp_bars = OUT / "tmp" / f"day_{sym}_{d}"
    _, grid_extra = atx_grid(["mm-replay", "--symbol", sym, "--trades", str(tr), "--out", str(tmp_bars), "--bars-only"])
    bp = bars_path(sym, d)
    bp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_bars / "bars_1m.csv", bp)
    shutil.rmtree(tmp_bars, ignore_errors=True)
    gate = write_gate(sym, d)
    zf = zones_file(sym, d)
    pt = tape_path(sym, prevs[1])
    med_prev = json.loads(pt.read_text(encoding="utf-8"))["median_print_notional"] if pt.is_file() else float("nan")
    liq_floor = LIQ_PROXY["notional_x_median"] * med_prev if np.isfinite(med_prev) and med_prev > 0 else 0.0
    vj = OUT / "variants" / sym / f"{d}.json"
    write_json(vj, {"variants": variants_json(sym, d)})
    btc_warm = [bars_path("BTCUSDT", x) for x in (prev_day(d, 2), prev_day(d)) if bars_path("BTCUSDT", x).is_file()]
    out = OUT / "days" / sym / d
    args = ["mm-replay", "--symbol", sym, "--date", d, "--ob200", str(ob), "--trades", str(tr), "--variants", str(vj), "--out", str(out),
            "--latency-ms", "150", "--zones", str(zf), "--btc-trades", str(btc_tr), "--liq-min-notional", f"{liq_floor:.6f}"]
    if warm:
        args += ["--warmup-bars", ",".join(str(w) for w in warm)]
    if btc_warm:
        args += ["--btc-warmup-bars", ",".join(str(w) for w in btc_warm)]
    msg, grid_extra = atx_grid(args, grid_extra)
    log(f"{key}: {msg}")
    errs = verify(out, len(grid()) * len(CELLS))
    if errs:
        man.put("days", key, {"status": "failed", "errors": errs, "raw_kept": str(raw.relative_to(ROOT))})
        raise RuntimeError(f"{key}: verification failed: {errs}")
    gzip_fills(out)
    st = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    tp = tape_path(sym, d)
    tp.parent.mkdir(parents=True, exist_ok=True)
    write_json(tp, {"symbol": sym, "prints": st["market"]["trades"], "median_print_notional": st["market"]["median_print_notional"]})
    shutil.rmtree(raw, ignore_errors=True)
    for k in ("ob200", "trades"):
        mark_deleted(man, f"{key}:{k}")
    man.put("days", key, {"status": "done", "seconds": round(time.time() - t0, 1), "raw_bytes": raw_bytes, "raw_deleted": True, "atx": msg,
                          "out_bytes": du_bytes(out), "grid_override": grid_extra, "gate": gate, "liq_floor_usd": liq_floor,
                          "median_print_prev": med_prev, "zones": str(zf.relative_to(ROOT))})


def run_all(only: list[str] | None = None) -> None:
    write_split()
    if not ATX.is_file():
        raise SystemExit(f"{ATX} missing: copy the release build there first")
    man = Manifest(MANIFEST)
    st0 = storage(refresh=True)
    log(f"storage before: {json.dumps({k: round(v, 3) for k, v in st0.items()})} budget {budget_gb()} GB")
    man.put("storage", "first_run" if not man.get("storage", "first_run") else f"run_{int(time.time())}", {"status": "done", **st0})
    want = [tuple(x.split(":")) for x in only] if only else [(s, d) for d in DAYS for s in UNIVERSE]
    days = [d for d in DAYS if any(w[1] == d for w in want)]
    for d in days:
        todo = [s for s in UNIVERSE if (s, d) in want and not man.done("days", f"{s}:{d}")]
        if not todo:
            continue
        try:
            btc_tr = ensure_btc(man, d)
        except Exception as e:  # noqa: BLE001
            log(f"BTCUSDT:{d}: FAILED {e!r}"[:3000])
            if "storage" in str(e) or "transient raw" in str(e):
                raise
            continue
        for s in todo:
            try:
                process(man, s, d, btc_tr)
            except Exception as e:  # noqa: BLE001 - logged in the manifest and the run log, next symbol continues
                log(f"{s}:{d}: FAILED {e!r}"[:3000])
                if "storage" in str(e) or "transient raw" in str(e):
                    raise
        drop_btc(man, d)
    st1 = storage(refresh=True)
    man.put("storage", f"after_{int(time.time())}", {"status": "done", **st1})
    log(f"storage after: {json.dumps({k: round(v, 3) for k, v in st1.items()})}")


def main(argv: list[str] | None = None) -> None:
    a = (argv if argv is not None else sys.argv[1:]) or ["run"]
    if a[0] == "freeze":
        freeze()
    elif a[0] == "split":
        print(json.dumps({k: v for k, v in write_split().items() if k in ("fit_days", "held_days", "written", "n_variants")}, indent=1))
    elif a[0] == "run":
        run_all(a[1].split(",") if len(a) > 1 else None)
    elif a[0] == "report":
        from ai_trader.research.mm_weekend_report import main as rep

        rep()
    else:
        raise SystemExit(f"unknown command {a[0]}")


if __name__ == "__main__":
    main()
