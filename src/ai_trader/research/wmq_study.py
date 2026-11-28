"""WMQ QUOTE (program WMM2, 2026-11-27; H-WMQ-1 ..): how to quote a small weekend book of illiquid Bybit alt perps.

Research only: no orders, no private calls (public archives, the public kline and funding-history endpoints only). The
market is the real ob200 book and public trade tape of each symbol-day, replayed by `atx mm-replay`
(crates/atx/src/mm_replay.rs, private release copy data/research/wmq/bin/atx-wmq.exe); the quoters are the live B0 rule
(config/mm_illiq20.yaml) and the replay-only WMQ modes: the intensity probe ("probe") and the quote rules ("model":
symmetric depth in sigma, Gueant-Lehalle-Fernandez-Tapia and Avellaneda-Stoikov skews, size skew, inventory unwinds,
trend pull, BTC pull).

Phases (weekends = Sat + Sun UTC; frozen in data/research/wmq/design.json before any P&L):
  CAL  6 weekends 2026-03-07 .. 04-11 (llm_in_sample): probe only -> fill-intensity and adverse-selection curves, the
       per-symbol GLFT / AS parameters, the adverse-selection add-on and the probe-optimal depth (calibration.json);
  SEL  11 weekends 2026-04-18 .. 06-27 (llm_in_sample): the whole grid -> selection by purged CV, PBO, deflated Sharpe;
  HOLD 21 weekends 2026-07-04 .. 11-21 (clean_holdout): the whole grid, tests of the registered hypotheses;
  FWD  2026-11-28 onwards (forward, as the archive publishes them).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.wmq_study freeze
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.wmq_study run cal [--worker i --of n]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.wmq_study calibrate
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.wmq_study run sel|hold|fwd [--worker i --of n]
"""

from __future__ import annotations

import gzip
import json
import math
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
import polars as pl

from ai_trader.sim.ob_gen import Manifest, budget_gb, du_bytes, http_get, storage_gb, write_json

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "WMQ"
OUT = ROOT / "data" / "research" / "wmq"
MMW = ROOT / "data" / "research" / "mm_weekend"
ATX = OUT / "bin" / "atx-wmq.exe"
MANIFEST = OUT / "manifest.json"
DESIGN = OUT / "design.json"
CALIB = OUT / "calibration.json"
LOG = OUT / "run.log"
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
KLINE_URL = "https://api.bybit.com/v5/market/kline?category=linear&symbol={s}&interval=1&start={a}&end={b}&limit=1000"
FUNDING_URL = "https://api.bybit.com/v5/market/funding/history?category=linear&symbol={s}&startTime={a}&endTime={b}&limit=200"
UA = {"User-Agent": "Mozilla/5.0 (AI_Trader research; public archives)"}
GB = 1024**3
QUOTA_GB = 3.0  # data/research/wmq + data/raw/WMQ
RAW_CAP_GB = 1.0
DAY = 86_400_000

# ----------------------------------------------------------------------------------------------------------------------
# universe and weekends (frozen before any P&L)
# ----------------------------------------------------------------------------------------------------------------------

_UNI = json.loads((ROOT / "research" / "mm_weekend" / "universe.json").read_text(encoding="utf-8"))
CORE = list(_UNI["universe"])  # the H-MMW universe (select_universe.py rule, snapshot 2026-11-21)
EXTRA_RULE = ("widened band on the same 2026-11-21 snapshot (research/mm_weekend/universe.json 'all', select_universe.py filters: "
              "USDT linear perps, Trading, not innovation zone, not TradFi, not SPY / BTC / ETH; 24h turnover 5-50 M USD): quoted "
              "spread >= 3 bp (was 4), depth within 25 bp >= 25 k USD per side (was 50 k), median |1m return| <= 3 x spread; ranked "
              "by spread / median |r1m|; the top 5 names not already in the core universe")


def extra_names() -> list[str]:
    rows = [r for r in _UNI["all"] if "error" not in r]
    el = [r for r in rows if r["spread_bp"] >= 3 and r["depth25_min_usd"] >= 25_000 and r["med_abs_r1m_bp"] <= 3 * r["spread_bp"]
          and 5e6 <= r["turnover24h"] <= 50e6 and r["symbol"] not in CORE]
    el.sort(key=lambda r: -r["spread_per_r1m"])
    return [r["symbol"] for r in el[:5]]


EXTRA = ["TRXUSDT", "GALAUSDT", "CRVUSDT", "1000BONKUSDT", "GRASSUSDT"]
SYMBOLS = CORE + EXTRA


def saturdays(first: str, n: int) -> list[str]:
    d = date.fromisoformat(first)
    assert d.weekday() == 5, first
    return [(d + timedelta(days=7 * i)).isoformat() for i in range(n)]


CAL_W = saturdays("2026-03-07", 6)
SEL_W = saturdays("2026-04-18", 11)
HOLD_W = saturdays("2026-07-04", 21)
FWD_FIRST = "2026-11-28"
KNOWLEDGE_CUTOFF = "2026-07-01"


def wdays(sat: str) -> list[str]:
    d = date.fromisoformat(sat)
    return [d.isoformat(), (d + timedelta(days=1)).isoformat()]


def phase_days(phase: str) -> list[str]:
    if phase == "fwd":
        out, d = [], date.fromisoformat(FWD_FIRST)
        while d <= datetime.now(UTC).date() - timedelta(days=1):
            out += wdays(d.isoformat())
            d += timedelta(days=7)
        return out
    return [x for w in {"cal": CAL_W, "sel": SEL_W, "hold": HOLD_W}[phase] for x in wdays(w)]


WEEKEND_OF = {x: w for w in CAL_W + SEL_W + HOLD_W for x in wdays(w)}
QUEUES = ["prorata", "pessimistic"]
LATENCIES = [150, 300]
CELLS = [(q, lat) for q in QUEUES for lat in LATENCIES]
PRIMARY = ("prorata", 150)
BASE_CFG = "config/mm_illiq20.yaml"
B0 = (0.45, 0.90)
PROBE_X = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
DEPTHS = [0.5, 1.0, 1.5, 2.0, 3.0]
S_CAP = [0.25, 0.5, 1.0]
Q_NOM = 5.0  # base orders at the cap (25 USD / ~5 USD minimum lot; 75 USD / 3 lots)
FEE_BP = 2.0
# Compact books kept for later re-runs (capacity, conditional gates): HOLD days of the core names, last 8 weekends.
KEEP_COMPACT = {"symbols": CORE, "weekends": HOLD_W[-8:]}


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
# the grid
# ----------------------------------------------------------------------------------------------------------------------


def glft_c2(g: float, a: float, k: float) -> float:
    return math.sqrt(g / (2 * k * a) / 3600.0 * (1 + g / k) ** (1 + k / g))


def glft_hs(g: float, a: float, k: float, q: float) -> tuple[float, float]:
    c1, c2 = math.log(1 + g / k) / g, glft_c2(g, a, k)
    return c1 + (2 * q + 1) / 2 * c2, c1 - (2 * q - 1) / 2 * c2


def as_hs(g: float, k: float, q: float, tau_s: float = 3600.0) -> tuple[float, float]:
    t = tau_s / 3600.0
    h = math.log(1 + g / k) / g + 0.5 * g * t
    return h + q * g * t, h - q * g * t


def solve_glft_gamma(skew_per_order: float, a: float, k: float) -> float:
    """gamma with c2(gamma) = skew per order (c2 is increasing in gamma); bisection on log gamma."""
    lo, hi = math.log(1e-9), math.log(1e6)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if glft_c2(math.exp(mid), a, k) < skew_per_order:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def _base(name: str, family: str, **kw: Any) -> dict[str, Any]:
    return {"name": name, "family": family, "config": BASE_CFG, "overrides": kw.pop("overrides", {}), "sigma_mode": "config",
            "queues": QUEUES, "latencies_ms": LATENCIES, **kw}


def _model(name: str, family: str, lots: int = 1, **kw: Any) -> dict[str, Any]:
    ov = {} if lots == 1 else {"symbol_loss_limit_usd": 1.5 * lots, "loss_limit_usd": 10.0 * lots}
    return _base(name, family, mode="model", lots=lots, cap_usd=25.0 * lots, overrides=ov, **kw)


def symbol_params(cal: dict[str, Any] | None, sym: str) -> dict[str, Any]:
    if cal is None:
        return {}
    return cal["symbols"][sym]["params"]


def grid(cal: dict[str, Any] | None, sym: str, phase: str) -> list[dict[str, Any]]:
    """Every variant of the phase for symbol `sym`. CAL: the probe only. SEL / HOLD / FWD: the whole grid, which needs the
    CAL calibration (A, k, the adverse-selection add-on, the probe-optimal depth) for the GLFT / AS / adj / xstar members."""
    probe = _base("PROBE", "P", mode="probe", probe_x=PROBE_X)
    if phase == "cal":
        return [probe]
    p = symbol_params(cal, sym)
    a, k = p["A"], p["kappa"]
    out = [_base("B0", "B0", overrides={"ask_offset_sigma": B0[0], "bid_offset_sigma": B0[1]}), probe]
    for m in DEPTHS:
        out.append(_model(f"D_m{m:g}", "D", price_rule="depth", depth_sigma=m))
    out.append(_model("D_xstar", "D", price_rule="depth", depth_sigma=p["xstar"]))
    for rule in ("glft", "as"):
        for s in S_CAP:
            per = s / Q_NOM
            g = solve_glft_gamma(per, a, k) if rule == "glft" else per
            for adj in ("p", "adj"):
                add = {"add_sigma": p["eps"][f"{rule}_s{s:g}"], "add_bps": FEE_BP} if adj == "adj" else {}
                out.append(_model(f"A_{rule}_s{s:g}_{adj}", "A", price_rule=rule, glft_gamma=g, glft_a=a, glft_k=k, as_tau_s=3600.0, **add))
    # A2 (amendment 1, an extra labelled trial): GLFT / AS with A, k fitted on the deep legs (nominal x >= 0.75)
    a2, k2 = p["A_deep"], p["kappa_deep"]
    for rule in ("glft", "as"):
        g = solve_glft_gamma(0.5 / Q_NOM, a2, k2) if rule == "glft" else 0.5 / Q_NOM
        for adj in ("p", "adj"):
            add = {"add_sigma": p["eps_deep"][rule], "add_bps": FEE_BP} if adj == "adj" else {}
            out.append(_model(f"A2_{rule}_s0.5_{adj}", "A2", price_rule=rule, glft_gamma=g, glft_a=a2, glft_k=k2, as_tau_s=3600.0, **add))
    for m in (1.0, 2.0):
        for sz in ("flat", "linear"):
            out.append(_model(f"B_m{m:g}_{sz[:3]}", "B", lots=3, price_rule="depth", depth_sigma=m, size_rule=sz))
    for s in (0.5, 1.0):
        g = solve_glft_gamma(s / Q_NOM, a, k)
        for sz in ("flat", "linear"):
            out.append(_model(f"C_glft_s{s:g}_{sz[:3]}", "C", lots=3, price_rule="glft", glft_gamma=g, glft_a=a, glft_k=k, size_rule=sz))
    bases = {"D_m1": dict(price_rule="depth", depth_sigma=1.0),
             "A_glft_s0.5_p": dict(price_rule="glft", glft_gamma=solve_glft_gamma(0.5 / Q_NOM, a, k), glft_a=a, glft_k=k)}
    for bn, bkw in bases.items():
        out.append(_model(f"E_{bn}_pass10", "E", unwind_passive_s=600.0, **bkw))
        out.append(_model(f"E_{bn}_pass10_t60", "E", unwind_passive_s=600.0, unwind_aggr_s=3600.0, **bkw))
        out.append(_model(f"E_{bn}_stop3_t60", "E", unwind_stop_sigma=3.0, unwind_aggr_s=3600.0, **bkw))
        out.append(_model(f"F_{bn}_btc", "F", btc_k=2.0, btc_cooldown_s=10.0, **bkw))
        out.append(_model(f"T_{bn}_trend", "T", trend_k=1.5, trend_window_min=60, **bkw))
    return out


GRID_NAMES = None  # filled lazily from a calibration


def variants_json(cal: dict[str, Any] | None, sym: str, phase: str) -> list[dict[str, Any]]:
    return [{k: v for k, v in x.items() if k != "family"} for x in grid(cal, sym, phase)]


DESIGN_TEXT = {
    "engine": "atx mm-replay (private release build data/research/wmq/bin/atx-wmq.exe): ob200 snapshot + deltas -> at_match "
              "Ob200Replayer -> OrderBook; the symbol's public tape merged by time; BTCUSDT's tape of the day for the BTC pull; "
              "virtual orders (no impact on the book); queue: join the back, prints at our price consume from the front, prints "
              "through our price or a crossing book fill us, unexplained decreases are cancels split pro-rata or FIFO-pessimistic",
    "cells": "every variant under queue {pro-rata, FIFO-pessimistic} x latency {150, 300} ms; primary = pro-rata 150 ms",
    "fees": "config/fees.yaml crypto class: maker 2.0 bp, taker 5.5 bp",
    "day": "each symbol-day starts flat at 00:00 UTC and is marked to the mid at 24:00 UTC; warm-up = 1m bars of the two previous "
           "days (the replay's own tape bars where they exist, else Bybit public 1m klines: identical closes, highs / lows can "
           "differ because a kline opens at the previous close)",
    "sigma": "the quoter's 1h sigma = max(240 min, 30 min) robust sigma of 1m closes, floor 20 bp (mm_illiq20); loghar_cr exists "
             "only for BTC ETH SOL XRP DOGE XAU XAG (WS04), none of which is in the universe, so every depth is in this sigma",
    "B0": "the live mm_illiq20 rule, ask 0.45 / bid 0.90 sigma, live caps (long 15 / short 25 USD), flow, toxicity, shock, level snap",
    "PROBE": "mode probe: per side one minimum-lot order at each x in PROBE_X sigma from the mid (x = 0: the touch, moved as soon "
             "as the touch moves); x > 0 orders follow the live replace rule (requote when the target drifted > 0.3 sigma after "
             "45 s, or > 1 sigma); a filled probe is re-sent at the next 10 s tick; no inventory logic, no stops, no pauses; per "
             "(x, side): live time, sends, fills, exposure-weighted distance (bp and sigma units at sending)",
    "model": "mode model (L = lots x the minimum lot per order, cap_usd = 25 x L symmetric, one base order from flat always "
             "allowed, per-symbol loss stop 1.5 x L USD and global 10 x L USD as mm_illiq20, the shock pause of mm_illiq20; no "
             "flow, toxicity or level layers): bid = mid (1 - (delta_b + add) sigma), ask = mid (1 + (delta_a + add) sigma), "
             "rounded away from the mid, never crossing; the live replace rule; the reducing side is reduce-only",
    "D": "price_rule depth: delta_b = delta_a = m, m in {0.5, 1, 1.5, 2, 3}, plus D_xstar: m = the probe-optimal depth x* of the "
         "symbol on CAL (calibration)",
    "A": "GLFT: delta_b = c1 + (2q + 1)/2 c2, delta_a = c1 - (2q - 1)/2 c2, c1 = ln(1 + g/k)/g, c2 = sqrt(g/(2 k A)/3600 "
         "(1 + g/k)^(1 + k/g)) with A, k of lambda(x) = A exp(-k x) (x in sigma, per second, per order-side) fitted on CAL per "
         "symbol; AS: reservation shift q g t, total spread g t + 2 ln(1 + g/k)/g, t = 1 h (tau 3600 s); q = position / base "
         "order. gamma from the skew at the cap s_cap in {0.25, 0.5, 1} sigma with Q_NOM = 5 orders at the cap: GLFT c2(g) = "
         "s_cap / 5, AS g = s_cap / 5. Each plain (p) and adjusted (adj): add_sigma = the CAL adverse-selection cost at 60 s "
         "(sigma units) at the model's flat half-spread, add_bps = the 2 bp maker fee (the Cartea-Jaimungal-Penalva depth "
         "delta* = 1/k + adverse move + fee)",
    "B": "size skew at L = 3 (cap 75 USD): symmetric depth m in {1, 2}, size flat (3 lots each side) or linear (adding round(3 (1 - "
         "|q|/cap)), reducing round(3 (1 + |q|/cap)) <= |q|)",
    "C": "GLFT price skew and size skew at L = 3: s_cap in {0.5, 1}, size flat or linear",
    "E": "inventory management on two fixed bases (D_m1, A_glft_s0.5_p): pass10 = the reducing quote joins the touch after 10 "
         "min of inventory; pass10_t60 = and a 60 min time limit (taker close); stop3_t60 = a taker close on a 3 sigma adverse "
         "move from the average entry and the 60 min time limit",
    "F": "the H-MMW2 BTC pull (k = 2, cooldown 10 s: cancel every quote when |BTC 1 s or 5 s return| > 2 x the BTC 1h sigma "
         "scaled to the horizon) on the two bases",
    "T": "trend pull on the two bases: no adding bid when the 60 min log return < -1.5 x sigma, no adding ask when > +1.5 x sigma",
    "calibration": "CAL days, primary cell: Poisson GLM of fill counts per (symbol-day, probe leg) with offset log(live seconds, "
                   "both sides) on x_eff (the leg's exposure-weighted distance in sigma): log lambda = log A - k x; also on the "
                   "bp distance and a power law log lambda = log A - alpha log(x + 0.05) (H-WMQ-2). Per leg: per-fill net edge "
                   "e_h = s (mid_t+h - price) / price - fee (bp) and the adverse cost at 60 s in sigma units -s (mid_t+60 - "
                   "mid_fill) / mid_fill / sigma. x* = argmax over legs of lambda_j e_60,j (bp per second of exposure; ties: "
                   "the nearer leg). eps (adj) = the adverse cost interpolated (linear in x_eff) at the model's flat half-spread, "
                   "shrunk to the pooled curve with w = n / (n + 200), n = fills of the two bracketing legs. Symbols with "
                   "fewer than 200 CAL probe fills at x > 0 use the pooled A, k",
    "pnl": "net = spread + adverse 60 s + inventory - fees (USD per symbol-day); adverse 1 / 10 / 300 s reported; funding: "
           "positions held across a funding time pay -position x mark x rate (Bybit public funding history), reported as "
           "net_f = net + funding; 'closed' = net - |end inventory| x (taker 5.5 bp + half spread) as a conservative mark",
}


def write_design(force: bool = False) -> dict[str, Any]:
    obj = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "WMQ", "program": "WMM2",
           "core": CORE, "extra_rule": EXTRA_RULE, "extra": EXTRA, "symbols": SYMBOLS,
           "cal_weekends": CAL_W, "sel_weekends": SEL_W, "hold_weekends": HOLD_W, "fwd_first": FWD_FIRST,
           "segments": {"cal": "llm_in_sample", "sel": "llm_in_sample", "hold": "clean_holdout", "fwd": "clean_holdout (forward)"},
           "cells": CELLS, "primary": PRIMARY, "probe_x": PROBE_X, "depths": DEPTHS, "s_cap": S_CAP, "q_nom": Q_NOM,
           "keep_compact": KEEP_COMPACT, "design": DESIGN_TEXT, "amendments": []}
    if DESIGN.exists() and not force:
        old = json.loads(DESIGN.read_text(encoding="utf-8"))
        keep = ("written", "amendments")
        same = {k: v for k, v in old.items() if k not in keep} == json.loads(json.dumps({k: v for k, v in obj.items() if k not in keep}))
        if not same:
            raise RuntimeError("design.json exists and differs: the design is frozen (add an amendment instead)")
        return old
    write_json(DESIGN, obj)
    return obj


def amend(text: str) -> None:
    d = json.loads(DESIGN.read_text(encoding="utf-8"))
    d["amendments"].append({"ts": datetime.now(UTC).isoformat(timespec="seconds"), "text": text})
    write_json(DESIGN, d)


# ----------------------------------------------------------------------------------------------------------------------
# storage, downloads, warm-up bars, funding
# ----------------------------------------------------------------------------------------------------------------------


def my_gb() -> float:
    return (du_bytes(OUT) + du_bytes(RAW)) / GB


def head_bytes(url: str) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return -404 if e.code == 404 else 0
    except Exception:  # noqa: BLE001
        return 0


def fetch(man: Manifest, key: str, url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    need = max(head_bytes(url), 0) / GB
    for _ in range(360):  # wait up to 30 min for the transient raw cap (other workers finishing)
        raw_now = du_bytes(RAW) / GB
        if raw_now + need <= RAW_CAP_GB:
            break
        time.sleep(5)
    else:
        raise RuntimeError(f"transient raw {raw_now:.2f} GB + {need:.2f} GB > {RAW_CAP_GB} GB")
    st = storage_gb()
    if st["total_gb"] + need >= budget_gb() - 1.0:
        raise RuntimeError(f"storage {st['total_gb']:.2f} GB + {need:.2f} GB >= budget {budget_gb()} - 1 GB: no download")
    if my_gb() + need > QUOTA_GB:
        raise RuntimeError(f"WMQ quota: {my_gb():.2f} GB + {need:.2f} GB > {QUOTA_GB} GB")
    r = http_get(url, dest)
    rec = man.get("downloads", key)
    if rec and rec.get("deleted"):
        man.put("redownloads", f"{key}:{int(time.time())}", {"status": "done", "first": rec})
    man.put("downloads", key, {**r, "url": url, "path": str(dest.relative_to(ROOT)), "storage_before_gb": st["total_gb"]})
    if r["status"] != "ok":
        raise RuntimeError(f"download {key}: {r}")
    return dest


def mark_deleted(man: Manifest, key: str) -> None:
    r = man.get("downloads", key) or {}
    man.put("downloads", key, {**r, "deleted": True})


def bars_path(sym: str, d: str) -> Path:
    return OUT / "bars" / sym / f"{d}.csv"


def _get_json(url: str) -> dict[str, Any]:
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                d = json.loads(r.read())
            if d.get("retCode") == 0:
                return d["result"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1 + 2 * k)
    raise RuntimeError(f"public endpoint failed: {url}")


def warm_bars(sym: str, d: str) -> Path:
    """1m bars of day d for the warm-up: this study's replay bars, else MMW's tape bars, else Bybit public 1m klines."""
    p = bars_path(sym, d)
    if p.is_file():
        return p
    q = MMW / "bars" / sym / f"{d}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if q.is_file():
        shutil.copyfile(q, p)
        return p
    t0 = day_ms(d)
    rows: dict[int, tuple] = {}
    for a in range(t0, t0 + DAY, 1000 * 60_000):
        b = min(a + 999 * 60_000, t0 + DAY - 60_000)
        for x in _get_json(KLINE_URL.format(s=sym, a=a, b=b))["list"]:
            rows[int(x[0])] = (float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))
        time.sleep(0.12)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("minute_ms,open,high,low,close,volume,n\n")
        for m in sorted(rows):
            o, h, lo, c, v = rows[m]
            f.write(f"{m},{o},{h},{lo},{c},{v},0\n")
    tmp.replace(p)
    return p


def funding(sym: str) -> list[tuple[int, float]]:
    """(funding time ms, rate) from the public funding-history endpoint, 2026-03-01 .. now (cached)."""
    p = OUT / "funding" / f"{sym}.json"
    if p.is_file() and json.loads(p.read_text())["until"] >= ms("2026-11-24T00:00:00Z"):
        return [tuple(x) for x in json.loads(p.read_text())["rows"]]
    a, end = ms("2026-03-01T00:00:00Z"), int(time.time() * 1000)
    rows: dict[int, float] = {}
    b = end
    while b > a:
        lst = _get_json(FUNDING_URL.format(s=sym, a=a, b=b))["list"]
        if not lst:
            break
        for x in lst:
            rows[int(x["fundingRateTimestamp"])] = float(x["fundingRate"])
        b = min(int(x["fundingRateTimestamp"]) for x in lst) - 1
        time.sleep(0.12)
    out = sorted(rows.items())
    write_json(p, {"symbol": sym, "until": end, "rows": out})
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the replay runs
# ----------------------------------------------------------------------------------------------------------------------


def atx(args: list[str], timeout: int = 6 * 3600) -> str:
    r = subprocess.run([str(ATX), *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"atx {' '.join(args[:3])} failed: {r.stderr[-2000:]}")
    return r.stdout.strip()


def _set_arg(extra: list[str], flag: str, val: str) -> list[str]:
    out = list(extra)
    if flag in out:
        out[out.index(flag) + 1] = val
    else:
        out += [flag, val]
    return out


def atx_grid(args: list[str], extra: list[str] | None = None) -> tuple[str, list[str]]:
    """Run atx; when a historical day's prices or sizes are off the given grid (finer than today's), infer the day's step
    from the offending number's decimals and retry with that --tick / --qty-step (logged)."""
    extra = list(extra or [])
    tried: set[str] = set()
    for _ in range(4):
        try:
            return atx(args + extra), extra
        except RuntimeError as e:
            m = re.search(r"price ([0-9.]+) off the tick grid", str(e))
            n = re.search(r"size ([0-9.e+-]+) off the qty grid", str(e))
            if m and "tick" not in tried:
                dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
                extra = _set_arg(extra, "--tick", f"{10.0 ** -dec:.{dec}f}")
                tried.add("tick")
            elif n and "qty" not in tried:
                v = n.group(1)
                dec = len(v.split(".")[1]) if "." in v else 0
                extra = _set_arg(extra, "--qty-step", f"{10.0 ** -dec:.{dec}f}")
                tried.add("qty")
            else:
                raise
            log(f"grid differs from the given rules on this day: retrying with {extra}")
    raise RuntimeError("grid inference failed")


def tick_audit(trades_gz: Path, tick: float) -> int:
    """k > 1 when (almost) every on-grid print of the day sits on a multiple of k ticks (a coarser historical grid)."""
    import csv

    ticks = []
    with gzip.open(trades_gz, "rt", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ticks.append(round(float(r["price"]) / tick))
            if i >= 200_000:
                break
    t = np.asarray(ticks, dtype=np.int64)
    if len(t) < 100:
        return 1
    for k in (100, 50, 25, 20, 10, 5, 4, 2):
        if np.mean(t % k == 0) >= 0.999:
            return k
    return 1


def rules_file(sym: str) -> Path:
    """Today's instrument rules from the public instruments endpoint (cached), in the replay's --rules format."""
    p = OUT / "rules" / f"{sym}.json"
    if not p.is_file():
        d = _get_json(f"https://api.bybit.com/v5/market/instruments-info?category=linear&symbol={sym}")["list"][0]
        lf, pf = d["lotSizeFilter"], d["priceFilter"]
        write_json(p, {"symbol": sym, "tick_size": float(pf["tickSize"]), "qty_step": float(lf["qtyStep"]), "min_qty": float(lf["minOrderQty"]),
                       "max_qty": float(lf["maxOrderQty"]), "max_mkt_qty": float(lf.get("maxMktOrderQty") or lf["maxOrderQty"]),
                       "min_notional": float(lf.get("minNotionalValue") or 5.0), "max_leverage": float(d["leverageFilter"]["maxLeverage"]),
                       "_fetched": datetime.now(UTC).isoformat(timespec="seconds"), "_funding_interval_min": int(d.get("fundingInterval") or 480)})
    return p


def rules(sym: str) -> dict[str, Any]:
    return json.loads(rules_file(sym).read_text(encoding="utf-8"))


def _fmt_step(x: float) -> str:
    dec = max(0, -int(math.floor(math.log10(x) + 1e-9)))
    return f"{x:.{dec}f}"


def verify(out: Path, n_quoters: int, btc: bool) -> list[str]:
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
    bad = [q["name"] for q in st["quoters"] if q.get("mode") != "probe" and abs(q["pnl"]["ledger_check"]) > 1e-6]
    if bad:
        errs.append(f"ledger identity broken: {bad[:3]}")
    if btc and mk.get("btc_prints", 0) <= 0:
        errs.append("no BTC prints")
    if not (out / "bars_1m.csv").is_file() or sum(1 for _ in (out / "bars_1m.csv").open()) < 100:
        errs.append("bars_1m.csv missing or short")
    return errs


QROW_KEYS = ["net", "gross", "spread", "adverse_1s", "adverse_10s", "adverse_60s", "adverse_300s", "inventory", "fees", "ledger_check"]


def convert_outputs(out: Path, sym: str, d: str) -> dict[str, Any]:
    """stats.json -> quoters.parquet (+ stats.json.gz), fills.jsonl -> fills.parquet; returns the market block."""
    st = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    rows, probe = [], []
    for q in st["quoters"]:
        r = {"symbol": sym, "date": d, "variant": q["name"], "queue": q["queue"], "latency": int(q["latency_ms"]), "mode": q.get("mode", "spread"),
             "fills": int(q["fills"]), "volume_usd": float(q["volume_usd"]), "end_inventory_usd": float(q["end_inventory_usd"]),
             "max_inventory_usd": float(q["max_inventory_usd"]), "stopped": q["stopped"] is not None,
             "fills_queue": int(q["funnel"]["fills_queue"]), "fills_through": int(q["funnel"]["fills_through"]),
             "fills_cross": int(q["funnel"]["fills_cross"]), "fills_stop": int(q["funnel"]["fills_stop"]),
             "placed": int(q["funnel"]["placed"]), "live_share_bid": float(q["funnel"]["live_share_bid"]), "live_share_ask": float(q["funnel"]["live_share_ask"]),
             "dist_bps_mean": float(q["funnel"]["dist_bps_mean"]) if q["funnel"]["dist_bps_mean"] is not None else float("nan"),
             "btc_pulls": int(q["funnel"].get("btc_pulls", 0))}
        for k in QROW_KEYS:
            r[k] = float(q["pnl"].get(k, float("nan")))
        for k, v in (q.get("model") or {}).items():
            r[f"model_{k}"] = int(v)
        rows.append(r)
        for leg in q.get("probe") or []:
            probe.append({"symbol": sym, "date": d, "queue": q["queue"], "latency": int(q["latency_ms"]), **{k: leg[k] for k in ("j", "x", "side", "live_ms", "placed", "fills")},
                          "dist_bps_mean": leg["dist_bps_mean"] if leg["dist_bps_mean"] is not None else float("nan"),
                          "x_eff_mean": leg["x_eff_mean"] if leg["x_eff_mean"] is not None else float("nan")})
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out / "quoters.parquet")
    if probe:
        pl.DataFrame(probe).write_parquet(out / "probe.parquet")
    fj = out / "fills.jsonl"
    if fj.is_file():
        recs = []
        with fj.open(encoding="utf-8") as f:
            for line in f:
                v = json.loads(line)
                pr = v.get("probe") or {}
                recs.append((v["variant"], v["queue"], int(v["latency_ms"]), int(v["ts"]), 1 if v["side"] == "Buy" else -1, float(v["price"]), float(v["qty"]),
                             bool(v["maker"]), v["kind"], bool(v["reduce_only"]), float(v["fee"]), float(v["mid"]), float(v["pos_after"]),
                             v.get("mid_1s"), v.get("mid_10s"), v.get("mid_60s"), v.get("mid_300s"), pr.get("j", -1), pr.get("place_mid"), pr.get("place_sigma"),
                             float(v["ahead0_lots"]) if v.get("ahead0_lots") is not None else float("nan")))
        cols = ["variant", "queue", "latency", "ts", "s", "price", "qty", "maker", "kind", "reduce_only", "fee", "mid", "pos_after",
                "mid_1s", "mid_10s", "mid_60s", "mid_300s", "probe_j", "place_mid", "place_sigma", "ahead0"]
        df = pl.DataFrame(recs, schema=cols, orient="row", infer_schema_length=None) if recs else pl.DataFrame(schema={c: pl.Float64 for c in cols})
        df.write_parquet(out / "fills.parquet", compression="zstd")
        fj.unlink()
    with (out / "stats.json").open("rb") as a, gzip.open(out / "stats.json.gz", "wb") as b:
        shutil.copyfileobj(a, b)
    (out / "stats.json").unlink()
    qj = out / "quotes.json"
    if qj.is_file():
        qj.unlink()  # the funnel is in stats.json.gz
    return st["market"]


def keep_compact(sym: str, d: str) -> bool:
    return sym in KEEP_COMPACT["symbols"] and WEEKEND_OF.get(d) in KEEP_COMPACT["weekends"]


def day_dir(sym: str, d: str) -> Path:
    return OUT / "days" / sym / d


def process(man: Manifest, phase: str, cal: dict[str, Any] | None, sym: str, d: str, btc: dict[str, Any] | None) -> None:
    out = day_dir(sym, d)
    if (out / "done.json").is_file():
        return
    t0 = time.time()
    key = f"{sym}:{d}"
    raw = RAW / sym / d
    warm = [warm_bars(sym, x) for x in (prev_day(d, 2), prev_day(d))]
    tr = fetch(man, f"{key}:trades", TRADES_URL.format(s=sym, d=d), raw / f"{sym}{d}.csv.gz")
    ob = fetch(man, f"{key}:ob200", OB_URL.format(s=sym, d=d), raw / f"{d}_{sym}_ob200.data.zip")
    raw_bytes = du_bytes(raw)
    ru = rules(sym)
    k = tick_audit(tr, ru["tick_size"])
    extra = ["--rules", str(rules_file(sym)), "--fee-class", "crypto", "--tick", _fmt_step(ru["tick_size"] * k), "--qty-step", _fmt_step(ru["qty_step"])]
    if k > 1:
        log(f"{key}: every print on a multiple of {k} ticks: replaying with tick {_fmt_step(ru['tick_size'] * k)}")
    vj = OUT / "variants" / phase / f"{sym}_{d}.json"
    vs = variants_json(cal, sym, phase)
    write_json(vj, {"variants": vs})
    n_q = sum(len(v["queues"]) * len(v["latencies_ms"]) for v in vs)
    tmp = OUT / "tmp" / f"{sym}_{d}"
    shutil.rmtree(tmp, ignore_errors=True)
    args = ["mm-replay", "--symbol", sym, "--date", d, "--ob200", str(ob), "--trades", str(tr), "--variants", str(vj), "--out", str(tmp),
            "--latency-ms", "150", "--warmup-bars", ",".join(str(w) for w in warm)]
    if btc is not None:
        args += ["--btc-trades", str(btc["trades"]), "--btc-warmup-bars", ",".join(str(w) for w in btc["warm"])]
    comp = keep_compact(sym, d) and phase in ("hold", "fwd")
    if comp:
        args += ["--write-compact", str(OUT / "compact" / sym / d)]
        bpx = OUT / "compact" / "BTCUSDT" / d / "px.bin.gz"
        if btc is not None and not bpx.is_file():
            args += ["--write-btc-compact", str(bpx)]
            bpx.parent.mkdir(parents=True, exist_ok=True)
    msg, extra = atx_grid(args, extra)
    errs = verify(tmp, n_q, btc is not None)
    if errs:
        man.put("days", key, {"status": "failed", "errors": errs, "phase": phase})
        raise RuntimeError(f"{key}: verification failed: {errs}")
    market = convert_outputs(tmp, sym, d)
    bp = bars_path(sym, d)
    bp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp / "bars_1m.csv", bp)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    shutil.move(str(tmp), str(out))
    shutil.rmtree(raw, ignore_errors=True)
    for kk in ("ob200", "trades"):
        mark_deleted(man, f"{key}:{kk}")
    rec = {"status": "done", "phase": phase, "seconds": round(time.time() - t0, 1), "raw_bytes": raw_bytes, "atx": msg, "grid_override": extra,
           "tick_multiple": k, "out_bytes": du_bytes(out), "compact": comp, "n_quoters": n_q,
           "market": {x: market.get(x) for x in ("book_messages", "trades", "rpi_trades", "trades_outside_touch", "mid_changes", "mid_start", "mid_end", "btc_prints")}}
    write_json(out / "done.json", rec)
    man.put("days", key, rec)
    log(f"{key} [{phase}] {msg} ({rec['seconds']} s)")


def ensure_btc(man: Manifest, d: str, keep: bool) -> dict[str, Any]:
    raw = RAW / "BTCUSDT" / d
    tr = fetch(man, f"BTCUSDT:{d}:trades", TRADES_URL.format(s="BTCUSDT", d=d), raw / f"BTCUSDT{d}.csv.gz")
    warm = [warm_bars("BTCUSDT", x) for x in (prev_day(d, 2), prev_day(d))]
    return {"trades": tr, "warm": warm, "raw": raw}


def drop_btc(man: Manifest, d: str) -> None:
    raw = RAW / "BTCUSDT" / d
    if raw.exists():
        shutil.rmtree(raw, ignore_errors=True)
        mark_deleted(man, f"BTCUSDT:{d}:trades")


def run_phase(phase: str, worker: int = 0, of: int = 1, only: list[str] | None = None) -> None:
    write_design()
    if not ATX.is_file():
        raise SystemExit(f"{ATX} missing")
    cal = None
    if phase != "cal":
        if not CALIB.is_file():
            raise SystemExit("calibration.json missing: run cal and calibrate first")
        cal = json.loads(CALIB.read_text(encoding="utf-8"))
    man = Manifest(MANIFEST)
    days = phase_days(phase)
    sats = sorted({WEEKEND_OF.get(d, d if date.fromisoformat(d).weekday() == 5 else prev_day(d)) for d in days})
    # by weekend, Saturday first: Sunday's warm-up uses Saturday's own tape bars whatever the number of workers
    mine = [d for i, w in enumerate(sats) if i % of == worker for d in wdays(w) if d in days]
    log(f"[{phase} w{worker}/{of}] {len(mine)} days; storage {storage_gb()['total_gb']:.2f} GB, WMQ {my_gb():.2f} GB")
    for d in mine:
        syms = [s for s in (only or SYMBOLS) if not (day_dir(s, d) / "done.json").is_file()]
        if not syms:
            continue
        if phase == "fwd" and head_bytes(OB_URL.format(s=syms[0], d=d)) <= 0:
            log(f"{d}: not in the archive yet")
            continue
        btc = None
        if phase != "cal":
            try:
                btc = ensure_btc(man, d, keep=False)
            except Exception as e:  # noqa: BLE001
                log(f"BTCUSDT:{d}: FAILED {e!r}"[:2000])
                raise
        for s in syms:
            try:
                process(man, phase, cal, s, d, btc)
            except Exception as e:  # noqa: BLE001 - logged, next symbol continues
                log(f"{s}:{d}: FAILED {e!r}"[:3000])
                if "quota" in str(e) or "storage" in str(e) or "transient raw" in str(e):
                    raise
        if btc is not None:
            drop_btc(man, d)
    log(f"[{phase} w{worker}/{of}] finished; storage {storage_gb()['total_gb']:.2f} GB, WMQ {my_gb():.2f} GB")


def rerun_compact(tag: str, variants: list[dict[str, Any]], symbols: list[str] | None = None, days: list[str] | None = None) -> None:
    """Replays of the stored compact HOLD subset with other variants (capacity, conditional gates): outputs in
    data/research/wmq/reruns/<tag>/<symbol>/<date>/ (quoters.parquet, fills.parquet), the grid of the original run."""
    symbols = symbols or KEEP_COMPACT["symbols"]
    days = days or [x for w in KEEP_COMPACT["weekends"] for x in wdays(w)]
    for d in days:
        for sym in symbols:
            comp = OUT / "compact" / sym / d
            out = OUT / "reruns" / tag / sym / d
            if (out / "done.json").is_file() or not (comp / "book.bin.gz").is_file():
                continue
            t0 = time.time()
            done = json.loads((day_dir(sym, d) / "done.json").read_text(encoding="utf-8"))
            vs = [{k: v for k, v in x.items() if k != "family"} for x in variants]
            for v in vs:  # per-symbol-day files (gates) are templated
                for k in ("gate", "widen_file"):
                    if v.get(k):
                        v[k] = v[k].format(sym=sym, d=d)
            vs = [v for v in vs if all(not v.get(k) or (ROOT / v[k]).is_file() for k in ("gate", "widen_file"))]
            vj = OUT / "variants" / f"rerun_{tag}" / f"{sym}_{d}.json"
            write_json(vj, {"variants": vs})
            tmp = OUT / "tmp" / f"rerun_{tag}_{sym}_{d}"
            shutil.rmtree(tmp, ignore_errors=True)
            warm = [warm_bars(sym, x) for x in (prev_day(d, 2), prev_day(d))]
            args = ["mm-replay", "--symbol", sym, "--date", d, "--compact", str(comp), "--variants", str(vj), "--out", str(tmp), "--latency-ms", "150",
                    "--warmup-bars", ",".join(str(w) for w in warm)]
            bpx = OUT / "compact" / "BTCUSDT" / d / "px.bin.gz"
            if bpx.is_file():
                args += ["--btc-compact", str(bpx), "--btc-warmup-bars", ",".join(str(warm_bars("BTCUSDT", x)) for x in (prev_day(d, 2), prev_day(d)))]
            msg = atx(args + list(done.get("grid_override", [])))
            n_q = sum(len(v["queues"]) * len(v["latencies_ms"]) for v in vs)
            errs = verify(tmp, n_q, False)
            if errs:
                raise RuntimeError(f"rerun {tag} {sym}:{d}: {errs}")
            convert_outputs(tmp, sym, d)
            (tmp / "fills.jsonl").unlink(missing_ok=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(out, ignore_errors=True)
            shutil.move(str(tmp), str(out))
            write_json(out / "done.json", {"status": "done", "seconds": round(time.time() - t0, 1), "atx": msg, "n_quoters": n_q})
            log(f"rerun {tag} {sym}:{d} {msg}")


# ----------------------------------------------------------------------------------------------------------------------
# calibration (CAL days, primary cell)
# ----------------------------------------------------------------------------------------------------------------------


def load_probe_legs(days: list[str], cell: tuple[str, int] | None = PRIMARY) -> pl.DataFrame:
    dfs = []
    for s in SYMBOLS:
        for d in days:
            p = day_dir(s, d) / "probe.parquet"
            if p.is_file():
                dfs.append(pl.read_parquet(p))
    df = pl.concat(dfs, how="diagonal_relaxed") if dfs else pl.DataFrame()
    if cell is not None and df.height:
        df = df.filter((pl.col("queue") == cell[0]) & (pl.col("latency") == cell[1]))
    return df


def load_probe_fills(days: list[str], cell: tuple[str, int] | None = PRIMARY, variant: str = "PROBE") -> pl.DataFrame:
    dfs = []
    for s in SYMBOLS:
        for d in days:
            p = day_dir(s, d) / "fills.parquet"
            if p.is_file():
                f = pl.read_parquet(p).filter(pl.col("variant") == variant)
                if cell is not None:
                    f = f.filter((pl.col("queue") == cell[0]) & (pl.col("latency") == cell[1]))
                dfs.append(f.with_columns(pl.lit(s).alias("symbol"), pl.lit(d).alias("date")))
    if not dfs:
        return pl.DataFrame()
    f = pl.concat(dfs, how="diagonal_relaxed")
    return add_edges(f)


def add_edges(f: pl.DataFrame) -> pl.DataFrame:
    """Per-fill edges in bp: spread captured, adverse markouts (1/10/60/300 s), net edge e_h = spread + adverse_h - fee."""
    s = pl.col("s").cast(pl.Float64)
    out = f.with_columns(((s * (pl.col("mid") - pl.col("price")) / pl.col("price")) * 1e4).alias("spread_bp"),
                         (pl.col("fee") / (pl.col("price") * pl.col("qty")) * 1e4).alias("fee_bp"))
    for h in ("1s", "10s", "60s", "300s"):
        out = out.with_columns((s * (pl.col(f"mid_{h}").cast(pl.Float64) - pl.col("mid")) / pl.col("price") * 1e4).alias(f"adv_{h}_bp"))
        out = out.with_columns((pl.col("spread_bp") + pl.col(f"adv_{h}_bp") - pl.col("fee_bp")).alias(f"e_{h}_bp"))
    return out.with_columns((-pl.col("adv_60s_bp") / (pl.col("place_sigma").cast(pl.Float64) * 1e4)).alias("adv_cost_60_sigma"))


def poisson_fit(n: np.ndarray, e: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """MLE of log lambda = b0 + b1 x with exposure e (Newton-Raphson); returns b0, b1, se, loglik."""
    ok = (e > 0) & np.isfinite(x)
    n, e, x = n[ok].astype(float), e[ok].astype(float), x[ok].astype(float)
    X = np.column_stack([np.ones_like(x), x])
    rate = max(n.sum(), 0.5) / e.sum()
    b = np.array([math.log(rate), 0.0])
    for _ in range(100):
        mu = e * np.exp(np.clip(X @ b, -50, 50))
        g = X.T @ (n - mu)
        H = X.T @ (X * mu[:, None])
        step = np.linalg.solve(H + 1e-12 * np.eye(2), g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = e * np.exp(X @ b)
    cov = np.linalg.inv(X.T @ (X * mu[:, None]))
    ll = float(np.sum(n * np.log(np.maximum(mu, 1e-300)) - mu - np.array([math.lgamma(k + 1) for k in n])))
    return {"b0": float(b[0]), "b1": float(b[1]), "se0": float(math.sqrt(cov[0, 0])), "se1": float(math.sqrt(cov[1, 1])), "loglik": ll, "n_obs": int(len(n)),
            "fills": float(n.sum())}


def leg_table(legs: pl.DataFrame) -> pl.DataFrame:
    """Per (symbol, date, j): live seconds and fills over both sides, exposure-weighted x_eff and bp distance."""
    return (legs.with_columns((pl.col("live_ms") / 1000.0).alias("live_s"))
            .group_by("symbol", "date", "j").agg(pl.col("x").first(), pl.col("live_s").sum(), pl.col("fills").sum(), pl.col("placed").sum(),
                                                  ((pl.col("x_eff_mean").fill_nan(None) * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("x_eff"),
                                                  ((pl.col("dist_bps_mean").fill_nan(None) * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("d_bps"))
            .sort("symbol", "date", "j"))


def calibrate() -> dict[str, Any]:
    days = phase_days("cal")
    legs = load_probe_legs(days)
    fills = load_probe_fills(days)
    lt = leg_table(legs)
    out: dict[str, Any] = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "days": days, "cell": PRIMARY, "symbols": {}, "pooled": {}}

    def curves(f: pl.DataFrame, t: pl.DataFrame) -> list[dict[str, Any]]:
        agg_t = t.group_by("j").agg(pl.col("x").first(), pl.col("live_s").sum(), pl.col("fills").sum(),
                                    ((pl.col("x_eff") * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("x_eff"),
                                    ((pl.col("d_bps") * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("d_bps")).sort("j")
        agg_f = f.group_by("probe_j").agg(pl.len().alias("n"), pl.col("e_60s_bp").mean().alias("e60"), pl.col("e_300s_bp").mean().alias("e300"),
                                          pl.col("e_1s_bp").mean().alias("e1"), pl.col("adv_cost_60_sigma").mean().alias("adv60_sigma"),
                                          pl.col("spread_bp").mean().alias("spread_bp")) if f.height else pl.DataFrame()
        rows = []
        for r in agg_t.iter_rows(named=True):
            g = agg_f.filter(pl.col("probe_j") == r["j"]).to_dicts() if agg_f.height else []
            g = g[0] if g else {"n": 0, "e60": float("nan"), "e300": float("nan"), "e1": float("nan"), "adv60_sigma": float("nan"), "spread_bp": float("nan")}
            lam = r["fills"] / r["live_s"] if r["live_s"] > 0 else float("nan")
            rows.append({**r, **{k: g[k] for k in ("n", "e60", "e300", "e1", "adv60_sigma", "spread_bp")}, "lambda": lam,
                         "rate_bp_s": lam * g["e60"] if np.isfinite(g["e60"]) else float("nan")})
        return rows

    def fit(t: pl.DataFrame) -> dict[str, Any]:
        n, e = t["fills"].to_numpy(), t["live_s"].to_numpy()
        fx = poisson_fit(n, e, t["x_eff"].to_numpy())
        fb = poisson_fit(n, e, t["d_bps"].to_numpy())
        fp = poisson_fit(n, e, np.log(t["x_eff"].to_numpy() + 0.05))
        return {"exp_sigma": {"A": math.exp(fx["b0"]), "kappa": -fx["b1"], **fx}, "exp_bps": {"A": math.exp(fb["b0"]), "k_bps": -fb["b1"], **fb},
                "power": {"A": math.exp(fp["b0"]), "alpha": -fp["b1"], **fp}}

    pooled_curve = curves(fills, lt)
    pooled_fit = fit(lt.filter(pl.col("x_eff").is_not_null()))
    ld = lt.filter(pl.col("x_eff").is_not_null() & (pl.col("x") >= 0.75))
    pooled_deep = poisson_fit(ld["fills"].to_numpy(), ld["live_s"].to_numpy(), ld["x_eff"].to_numpy())
    out["pooled"] = {"curve": pooled_curve, "fit": pooled_fit}

    def interp_eps(curve: list[dict], x0: float) -> tuple[float, int]:
        c = sorted([r for r in curve if np.isfinite(r.get("adv60_sigma", float("nan"))) and r["x_eff"] is not None], key=lambda r: r["x_eff"])
        if not c:
            return float("nan"), 0
        xs = np.array([r["x_eff"] for r in c])
        ys = np.array([r["adv60_sigma"] for r in c])
        ns = np.array([r["n"] for r in c])
        i = int(np.searchsorted(xs, x0))
        lo, hi = max(i - 1, 0), min(i, len(xs) - 1)
        return float(np.interp(x0, xs, ys)), int(ns[lo] + (ns[hi] if hi != lo else 0))

    for s in SYMBOLS:
        ts_ = lt.filter((pl.col("symbol") == s) & pl.col("x_eff").is_not_null())
        fs = fills.filter(pl.col("symbol") == s) if fills.height else fills
        curve = curves(fs, ts_) if ts_.height else []
        n_far = int(ts_.filter(pl.col("x") > 0)["fills"].sum()) if ts_.height else 0
        f = fit(ts_) if ts_.height and n_far > 0 else None
        use = f["exp_sigma"] if (f is not None and n_far >= 200 and f["exp_sigma"]["kappa"] > 0) else pooled_fit["exp_sigma"]
        a, k = float(use["A"]), float(use["kappa"])
        # probe-optimal depth: argmax lambda_j e60_j (ties: nearer)
        cand = [r for r in curve if np.isfinite(r["rate_bp_s"])]
        xstar = float(max(cand, key=lambda r: (r["rate_bp_s"], -r["j"]))["x"]) if cand else 1.0
        eps = {}
        for rule in ("glft", "as"):
            for sc in S_CAP:
                g = solve_glft_gamma(sc / Q_NOM, a, k) if rule == "glft" else sc / Q_NOM
                h = glft_hs(g, a, k, 0.0)[0] if rule == "glft" else as_hs(g, k, 0.0)[0]
                e_s, n_s = interp_eps(curve, h)
                e_p, _ = interp_eps(pooled_curve, h)
                w = n_s / (n_s + 200.0) if np.isfinite(e_s) else 0.0
                eps[f"{rule}_s{sc:g}"] = float(w * (e_s if np.isfinite(e_s) else 0.0) + (1 - w) * e_p)
        # amendment 1 (A2): A, k on the deep legs only (nominal x >= 0.75), own fit with >= 50 such fills, else pooled
        deep = ts_.filter(pl.col("x") >= 0.75) if ts_.height else ts_
        n_deep = int(deep["fills"].sum()) if deep.height else 0
        fd = poisson_fit(deep["fills"].to_numpy(), deep["live_s"].to_numpy(), deep["x_eff"].to_numpy()) if n_deep >= 50 else pooled_deep
        a2, k2 = math.exp(fd["b0"]), -fd["b1"]
        eps_deep = {}
        for rule in ("glft", "as"):
            g = solve_glft_gamma(0.5 / Q_NOM, a2, k2) if rule == "glft" else 0.5 / Q_NOM
            h = glft_hs(g, a2, k2, 0.0)[0] if rule == "glft" else as_hs(g, k2, 0.0)[0]
            e_s, n_s = interp_eps(curve, h)
            e_p, _ = interp_eps(pooled_curve, h)
            w = n_s / (n_s + 200.0) if np.isfinite(e_s) else 0.0
            eps_deep[rule] = float(w * (e_s if np.isfinite(e_s) else 0.0) + (1 - w) * e_p)
        out["symbols"][s] = {"curve": curve, "fit": f, "n_far_fills": n_far, "used": "own" if use is not pooled_fit["exp_sigma"] else "pooled",
                             "deep_fit": {**fd, "n_deep_fills": n_deep, "used": "own" if n_deep >= 50 else "pooled"},
                             "params": {"A": a, "kappa": k, "xstar": xstar, "eps": eps, "A_deep": a2, "kappa_deep": k2, "eps_deep": eps_deep}}
    write_json(CALIB, out)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------------------------------------


def freeze() -> None:
    if DESIGN.exists():
        write_design()
        print("design.json already frozen")
        return
    assert extra_names() == EXTRA, (extra_names(), EXTRA)
    d = write_design()
    from ai_trader.research.diary import entry

    body = (f"Agent WMQ froze its design in data/research/wmq/design.json ({d['written']}) before any replay P&L. Universe: the H-MMW "
            f"core ({', '.join(CORE)}) plus 5 names by a stated rule ({EXTRA_RULE}): {', '.join(EXTRA)}. Weekends (Sat + Sun UTC): CAL "
            f"{CAL_W[0]} .. {CAL_W[-1]} (6, probe only, calibration), SEL {SEL_W[0]} .. {SEL_W[-1]} (11, selection), HOLD {HOLD_W[0]} .. "
            f"{HOLD_W[-1]} (21, clean holdout), forward from {FWD_FIRST}. Cells: pro-rata / FIFO-pessimistic x 150 / 300 ms. Families: "
            "probe; D symmetric depth {0.5, 1, 1.5, 2, 3} sigma + the probe-optimal depth; A GLFT and AS price skew, s_cap {0.25, 0.5, 1} "
            "sigma, plain or adjusted by the CAL adverse-selection cost + fee; B size skew at 3 lots; C GLFT + size skew; E passive / "
            "time-limit / stop unwinds; F BTC pull; T trend pull. Hypotheses H-WMQ-1..11 registered (research/hypotheses_pending/WMQ.yaml) "
            "before this entry; conditional tests on the WMS selection and the WMN news gate will be registered as their own ids once "
            "those rules are frozen.")
    entry("decision", "WMQ design frozen before results", body, tags=["wmq", "mm_weekend", "replay", "design"],
          hypotheses=[f"H-WMQ-{i}" for i in range(1, 12)], links=["data/research/wmq/design.json", "research/hypotheses_pending/WMQ.yaml"])
    print("froze design.json and wrote the diary entry")


def main(argv: list[str] | None = None) -> None:
    a = (argv if argv is not None else sys.argv[1:]) or ["run", "cal"]
    if a[0] == "freeze":
        freeze()
    elif a[0] == "run":
        phase = a[1]
        w, n, only = 0, 1, None
        if "--worker" in a:
            w = int(a[a.index("--worker") + 1])
        if "--of" in a:
            n = int(a[a.index("--of") + 1])
        if "--only" in a:
            only = a[a.index("--only") + 1].split(",")
        run_phase(phase, w, n, only)
    elif a[0] == "calibrate":
        c = calibrate()
        for s, v in c["symbols"].items():
            print(s, v["used"], {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v["params"].items() if k != "eps"}, v["n_far_fills"])
    else:
        raise SystemExit(f"unknown command {a[0]}")


if __name__ == "__main__":
    main()
