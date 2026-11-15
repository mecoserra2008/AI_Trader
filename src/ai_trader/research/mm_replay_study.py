"""Market-making backtest on replayed real Bybit order books (agent M, 2026-11-14; H-MMR0, H-MMR1, H-MMR2).

Research only: no live orders, no Bybit trading calls. The quoting code is the live `atx mm` code path, run by the Rust
replay `atx mm-replay` (crates/atx/src/mm_replay.rs) on the real ob200 book (quote-saver.bycsi.com) and the real public
trade tape (public.bybit.com) of each symbol-day.

Pipeline (one symbol-day at a time, checkpointed in data/research/mm_replay/manifest.json):
  1. storage check (data/lake + data/research + data/raw < budget - 1 GB, transient raw < 2 GB), download the day's ob200
     zip and trade archive into data/raw/mm_replay/<symbol>/<date>/;
  2. warm-up bars: the 1m bars of the two previous days (from earlier replays, or `atx mm-replay --bars-only` on a
     downloaded previous-day trade archive, which is then deleted);
  3. `atx mm-replay` with the pre-declared grid (every variant x 3 queue models) -> stats.json, quotes.json, fills.jsonl
     (gzipped), bars_1m.csv in data/research/mm_replay/days/<symbol>/<date>/; on live days also the H-MMR0 run with the
     journaled live configuration over the live session window (data/research/mm_replay/live/...);
  4. verify (engine rejects 0, replayer checks equal, ledger identity, bars present), then delete the raw files.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.mm_replay_study split      # split + grid (before any P&L)
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.mm_replay_study run        # all symbol-days (resumable)
    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.mm_replay_study report
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_trader.sim.ob_gen import Manifest, budget_gb, du_bytes, http_get, storage_gb, write_json

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "mm_replay"
OUT = ROOT / "data" / "research" / "mm_replay"
MANIFEST = OUT / "manifest.json"
SPLIT = OUT / "split.json"
LOG = OUT / "run.log"
# A private copy of target/debug/atx.exe (copied after the last mm_replay change): other agents rebuild target/debug/atx.exe
# while the study runs, which briefly removes it (two failed days on 2026-11-14).
ATX = ROOT / "data" / "research" / "mm_replay" / "bin" / "atx-mmreplay.exe"
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
UA = {"User-Agent": "Mozilla/5.0 (AI_Trader research; public archives)"}
GB = 1024**3
RAW_CAP_GB = 2.0

GROUPS = {"btc": ["BTCUSDT"], "illiquid": ["TRXUSDT", "ETCUSDT", "XLMUSDT"]}
SYMBOLS = [s for g in GROUPS.values() for s in g]
GROUP_OF = {s: g for g, ss in GROUPS.items() for s in ss}
BASE_CFG = {"btc": "config/mm_multi.yaml", "illiquid": "config/mm_illiq20.yaml"}
DAYS = [(date(2026, 10, 25) + timedelta(days=k)).isoformat() for k in range(20)]   # 2026-10-25 .. 2026-11-13
FIT_DAYS, HELD_DAYS = DAYS[:10], DAYS[10:]
LIVE_DAYS = ["2026-11-09", "2026-11-10"]
LATENCY_MS = 150

# Live sessions replayed for H-MMR0 (journal data/journal_mainnet/*/session_start.jsonl; tick period from the
# leaderboard lines of logs/mm_mainnet_*.log: one line per 6 ticks).
LIVE_SESSIONS = {
    "2026-11-09": {"session": "mm11082326", "journal_day": "2026-11-08", "start": "2026-11-09T00:00:00Z", "end": "2026-11-09T08:13:42Z",
                   "until": "2026-11-09T21:00:00Z", "tick_ms": 16_667, "fills": "data/mm/analysis/mainnet-20261109T0817/executions.json"},
    "2026-11-10": {"session": "mm11101949", "journal_day": "2026-11-10", "start": "2026-11-10T19:49:20Z", "end": "2026-11-10T22:20:04Z",
                   "until": "2026-11-11T20:00:00Z", "tick_ms": 22_167, "fills": "data/journal_mainnet/2026-11-10/fill.jsonl"},
}

SKEWS = {"live": (0.45, 0.90), "sym": (0.675, 0.675)}   # (ask_offset_sigma, bid_offset_sigma) per unit width
MICRO = {"micro_interval_s": 15, "micro_lookback": 40, "micro_floor_bps": 3.0, "flow_trades_limit": 1000, "requote_sigma": 0.5, "min_rest_s": 10}


def ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


def day_ms(d: str) -> int:
    return ms(d + "T00:00:00Z")


def log(msg: str) -> None:
    line = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# the pre-declared grid
# ----------------------------------------------------------------------------------------------------------------------


def grid(group: str) -> list[dict[str, Any]]:
    """Variants per group (frozen in split.json before any P&L). Families: L = live 1h-sigma rule x width k; M = micro
    sigma (mm_nvda_micro settings) x horizon; F = fixed quote sigma (bp); N = BTC 1h forecast (field / network) x k.
    Each in the live skew (ask 0.45, bid 0.90 per unit) and a symmetric one (0.675 / 0.675)."""
    base = BASE_CFG[group]
    out: list[dict[str, Any]] = []
    # Amendment 2026-11-14 11:50 UTC (before any BTC fill): mm_multi's portfolio net cap (60 USD) is below one BTC lot
    # (~80 USD) and has no from-flat exemption, so the live code never quotes BTC; a single-symbol replay has no
    # portfolio, so the net cap is switched off for BTC (per-symbol caps, one lot from flat, still apply).
    extra = {"max_net_usd": 0.0} if group == "btc" else {}

    def add(name: str, fam: str, over: dict, **kw):
        out.append({"name": name, "family": fam, "config": base, "overrides": {**over, **extra}, "queues": ["pessimistic", "prorata", "optimistic"], **kw})

    for sk, (a, b) in SKEWS.items():
        for k in (0.5, 1.0, 2.0):
            add(f"L_k{k:g}_{sk}", "L", {"ask_offset_sigma": a * k, "bid_offset_sigma": b * k}, sigma_mode="config")
        for h in (30, 120, 480):
            add(f"M_h{h}_{sk}", "M", {**MICRO, "micro_horizon_s": float(h), "ask_offset_sigma": a, "bid_offset_sigma": b}, sigma_mode="config")
        for w in (3, 6, 12, 24, 48):
            add(f"F_w{w}_{sk}", "F", {"ask_offset_sigma": a, "bid_offset_sigma": b}, sigma_mode="fixed", fixed_bps=float(w))
        if group == "btc":
            for k in (0.5, 1.0, 2.0):
                add(f"N_k{k:g}_{sk}", "N", {"ask_offset_sigma": a, "bid_offset_sigma": b}, sigma_mode="forecast", forecast_scale=k)
    return out


def variants_json(group: str) -> list[dict[str, Any]]:
    return [{k: v for k, v in x.items() if k != "family"} for x in grid(group)]


DESIGN = {
    "engine": "atx mm-replay (crates/atx/src/mm_replay.rs): ob200 snapshot + deltas -> at_match::Ob200Replayer -> at_match::OrderBook; public trade "
              "tape merged by time (trade ts <= message cts first); the live atx mm decision code (mm::symbol_inputs, mm::target_quotes, "
              "mm::needs_replace, sigma / flow / micro / toxicity / shock / trend-guard steps); our orders virtual (minimum lots, no impact)",
    "queue": "join the back at activation (ahead = level size); trades at our price consume from the front; trade through our price or a real "
             "resting order crossing our price fills us; unexplained level decreases are cancels split FIFO-pessimistic (behind first), pro-rata "
             "(primary), LIFO-optimistic (ahead first); levels outside the 200-level window count fully ahead when they appear; RPI prints do not fill",
    "latency_ms": LATENCY_MS,
    "fees": "config/fees.yaml by symbol class (crypto: maker 2.0 bp, taker 5.5 bp)",
    "risk": "inventory caps, per-symbol and global loss stops from the config (a stop cancels and closes at market, taker); one-lot-from-flat rule",
    "day": "each symbol-day starts flat at 00:00 UTC and is marked to the mid at 24:00 UTC; warm-up 1m bars of the two previous days",
    "pnl": "net = spread captured + adverse selection (60 s) + inventory P&L - fees; spread = sum s q (mid_fill - price); adverse_h = sum s q "
           "(mid_t+h - mid_fill) (1 / 10 / 60 s); inventory = sum s q (mid_end - mid_t+60s)",
    "selection": "per group, on FIT days only, pro-rata queue: combinatorial purged CV over days (10 fit days, 2 test days per split = 45 "
                 "splits, training days within 1 calendar day of a test day purged); a variant's CV score = median over splits of its "
                 "test-day mean net P&L per symbol-day; choice = the highest CV score (H-MMR1); best fixed width = the F variant chosen "
                 "the same way among F only, best micro among M, best forecast among N (H-MMR2)",
    "overfitting": "PBO by CSCV over the fit days (S = 10 one-day blocks, C(10,5) = 252 splits, metric = mean net P&L per symbol-day); "
                   "deflated Sharpe of the chosen variant's daily net P&L on fit days with N = grid size trials",
    "tests": "H-MMR1: mean net P&L per symbol-day on HELD days, day-clustered bootstrap (B = 10000) 95% CI, per group, pro-rata primary; "
             "H-MMR2: paired per symbol-day difference (forecast - best fixed) on HELD days, day-clustered bootstrap; H-MMR0: live days with "
             "the journaled live config over the live window, |replay - live| / max(live, 1) fill-count error and P&L sign agreement",
    "segment": "every day is after research.knowledge_cutoff (2026-07-01): clean_holdout",
    "forecast": "N family: the BTC live network's 1h band (data/journal/live_nn net.1h.sigma_pct) where it exists (2026-11-13 20:00-24:00 UTC); "
                "every other hour the field's hour-of-week x log-vol sigma sqrt(vf_t * f_next) (ai_trader.field.state; hour-of-week profile "
                "known at the day's 00:00 UTC; row valid from its known_ts)",
}


AMENDMENTS = [
    {"ts": "2026-11-14T11:50Z", "what": "btc grid: max_net_usd = 0 (portfolio net cap off) in every BTC variant",
     "why": "mm_multi's 60 USD net cap is below one BTC lot (~80 USD) and has no from-flat exemption, so the live code placed no BTC "
            "quote on the first BTC day (0 orders, 0 fills: no P&L information); a single-symbol replay has no portfolio",
     "effect": "BTCUSDT 2026-10-25 re-run (its archives downloaded a second time, logged in the manifest); split and other variants unchanged"},
]


def write_split(force: bool = False) -> dict[str, Any]:
    """The fit / held-out split and the frozen grid, written before any P&L is computed."""
    obj = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "M", "hypotheses": ["H-MMR0", "H-MMR1", "H-MMR2"],
           "symbols": GROUPS, "days": DAYS, "fit_days": FIT_DAYS, "held_days": HELD_DAYS, "live_days": LIVE_DAYS,
           "split_rule": "chronological: the first 10 days fit, the last 10 held out (a forward test)",
           "base_configs": BASE_CFG, "grid": {g: grid(g) for g in GROUPS}, "design": DESIGN, "live_sessions": LIVE_SESSIONS,
           "amendments": AMENDMENTS}
    if SPLIT.exists() and not force:
        old = json.loads(SPLIT.read_text(encoding="utf-8"))
        same = {k: v for k, v in old.items() if k != "written"} == json.loads(json.dumps({k: v for k, v in obj.items() if k != "written"}))
        if not same:
            raise RuntimeError("split.json exists and differs: the split is frozen")
        return old
    write_json(SPLIT, obj)
    return obj


# ----------------------------------------------------------------------------------------------------------------------
# storage, downloads, warm-up, forecast
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
    """Download unless the manifest already holds a verified, deleted copy (then refuse: never twice)."""
    rec = man.get("downloads", key)
    re = man.get("redownload", key)
    if rec and rec.get("deleted") and re and re.get("status") != "done":
        man.put("redownload", key, {**re, "status": "done"})
        man.put("downloads", key, {**rec, "deleted": False, "first_download": rec})
    elif rec and rec.get("deleted"):
        raise RuntimeError(f"{key} was downloaded and deleted before: not downloading twice")
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
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


def bars_path(sym: str, d: str) -> Path:
    return OUT / "bars" / sym / f"{d}.csv"


def ensure_bars(man: Manifest, sym: str, d: str) -> Path | None:
    """1m bars of day d (warm-up): from a replay of d, else (days outside the sample only) from its trade archive
    (--bars-only, raw deleted after). A sample day that failed gives no warm-up (None) rather than a second download."""
    p = bars_path(sym, d)
    if p.is_file():
        return p
    dd = OUT / "days" / sym / d / "bars_1m.csv"
    if dd.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dd, p)
        return p
    if d in DAYS:
        log(f"{sym}:{d}: no bars (sample day not processed): warm-up without it")
        return None
    raw = RAW / sym / d
    tr = fetch(man, f"{sym}:{d}:trades", TRADES_URL.format(s=sym, d=d), raw / f"{sym}{d}.csv.gz")
    tmp = OUT / "tmp" / f"bars_{sym}_{d}"
    atx_grid(["mm-replay", "--symbol", sym, "--trades", str(tr), "--out", str(tmp), "--bars-only"])
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp / "bars_1m.csv"), p)
    shutil.rmtree(tmp, ignore_errors=True)
    nb = du_bytes(raw)
    shutil.rmtree(raw, ignore_errors=True)
    r = man.get("downloads", f"{sym}:{d}:trades") or {}
    man.put("downloads", f"{sym}:{d}:trades", {**r, "deleted": True})
    man.put("bars", f"{sym}:{d}", {"status": "done", "source": "trades --bars-only", "raw_deleted_bytes": nb})
    return p


def forecast_csv(d: str) -> Path:
    """BTC 1h sigma forecast valid during day d: the live network band where logged, else the field's sigma."""
    p = OUT / "forecast" / f"BTCUSDT_{d}.csv"
    if p.is_file():
        return p
    import numpy as np
    import polars as pl

    from ai_trader.field.state import build_state, hour_profile, load_frame

    d0 = datetime.fromisoformat(d).replace(tzinfo=UTC)
    fr = load_frame(d0 + timedelta(days=1))
    prof = hour_profile(fr, d0)
    st = build_state(fr, prof).filter((pl.col("known_ts") >= d0 - timedelta(hours=3)) & (pl.col("known_ts") < d0 + timedelta(days=1)))
    rows = [(int(k.timestamp() * 1000), float(np.sqrt(v * f)), "field") for k, v, f in zip(st["known_ts"], st["vf"], st["f_next"], strict=True)]
    nn = ROOT / "data" / "journal" / "live_nn" / f"{d}.jsonl"
    if nn.is_file():
        for line in nn.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            s = r.get("net", {}).get("1h", {}).get("sigma_pct")
            if s:
                t = int(datetime.fromisoformat(r["as_of"]).timestamp() * 1000)
                end = int(datetime.fromisoformat(r["net"]["1h"]["window_end"]).timestamp() * 1000)
                rows = [x for x in rows if not (t <= x[0] < end)] + [(t, float(s) / 100.0, "nn")]
                # the field row of the next hour takes over again at the window end
    rows.sort()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("valid_from_ms,sigma_1h,source\n")
        for t, s, src in rows:
            f.write(f"{t},{s},{src}\n")
    return p


# ----------------------------------------------------------------------------------------------------------------------
# replay runs
# ----------------------------------------------------------------------------------------------------------------------


def atx(args: list[str], timeout: int = 6 * 3600) -> str:
    r = subprocess.run([str(ATX), *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"atx {' '.join(args[:3])} failed: {r.stderr[-2000:]}")
    return r.stdout.strip()


def atx_grid(args: list[str]) -> tuple[str, list[str]]:
    """Run atx; when a historical day's prices are off the lake's (latest) tick grid, infer the day's tick from the
    offending price's decimals and retry with --tick (logged). Returns (stdout, extra args used)."""
    extra: list[str] = []
    for _ in range(3):
        try:
            return atx(args + extra), extra
        except RuntimeError as e:
            import re

            m = re.search(r"price ([0-9.]+) off the tick grid", str(e))
            if not m or "--tick" in extra:
                raise
            dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
            extra = ["--tick", f"{10.0 ** -dec:.{dec}f}"]
            log(f"tick grid differs from the lake rules on this day: retrying with {extra}")
    raise RuntimeError("tick inference failed")


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
    if not (out / "bars_1m.csv").is_file() or sum(1 for _ in (out / "bars_1m.csv").open()) < 100:
        errs.append("bars_1m.csv missing or short")
    return errs


def gzip_fills(out: Path) -> None:
    f = out / "fills.jsonl"
    if f.is_file():
        with f.open("rb") as a, gzip.open(out / "fills.jsonl.gz", "wb") as b:
            shutil.copyfileobj(a, b)
        f.unlink()


def live_config(day: str) -> Path:
    """The journaled live configuration of the H-MMR0 session (session_start record) as YAML."""
    import yaml

    s = LIVE_SESSIONS[day]
    p = OUT / "live_cfg" / f"{s['session']}.yaml"
    if p.is_file():
        return p
    for line in (ROOT / "data" / "journal_mainnet" / s["journal_day"] / "session_start.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("session_id") == s["session"]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# journaled live config of session " + s["session"] + " (session_start record)\n" + yaml.safe_dump(r["config"], sort_keys=True), encoding="utf-8")
            return p
    raise RuntimeError(f"no session_start for {s['session']}")


def process(man: Manifest, sym: str, d: str) -> None:
    key = f"{sym}:{d}"
    if man.done("days", key):
        return
    t0 = time.time()
    g = GROUP_OF[sym]
    raw = RAW / sym / d
    prev = [(datetime.fromisoformat(d) - timedelta(days=k)).date().isoformat() for k in (2, 1)]
    warm = [w for w in (ensure_bars(man, sym, x) for x in prev) if w is not None]
    ob = fetch(man, f"{sym}:{d}:ob200", OB_URL.format(s=sym, d=d), raw / f"{d}_{sym}_ob200.data.zip")
    tr = fetch(man, f"{sym}:{d}:trades", TRADES_URL.format(s=sym, d=d), raw / f"{sym}{d}.csv.gz")
    raw_bytes = du_bytes(raw)
    out = OUT / "days" / sym / d
    vj = OUT / "variants" / f"{g}.json"
    if not vj.is_file():
        write_json(vj, {"variants": variants_json(g)})
    args = ["mm-replay", "--symbol", sym, "--date", d, "--ob200", str(ob), "--trades", str(tr), "--variants", str(vj), "--out", str(out),
            "--latency-ms", str(LATENCY_MS)] + (["--warmup-bars", ",".join(str(w) for w in warm)] if warm else [])
    if g == "btc":
        args += ["--forecast", str(forecast_csv(d))]
    msg, grid_extra = atx_grid(args)
    log(f"{key}: {msg}")
    errs = verify(out, 3 * len(grid(g)))
    live_msg = None
    if d in LIVE_DAYS:
        s = LIVE_SESSIONS[d]
        lv = OUT / "live" / sym / d
        cfg = live_config(d)
        lvj = OUT / "variants" / f"live_{d}.json"
        write_json(lvj, {"variants": [{"name": f"live_{s['session']}", "config": str(cfg.relative_to(ROOT)).replace("\\", "/"), "overrides": {},
                                       "sigma_mode": "config", "tick_ms": s["tick_ms"], "flow_rotation": True, "deadline_ms": ms(s["until"]),
                                       "queues": ["pessimistic", "prorata", "optimistic"]}]})
        live_msg, _ = atx_grid(grid_extra[:0] + ["mm-replay", "--symbol", sym, "--date", d, "--start-ms", str(ms(s["start"])), "--end-ms", str(ms(s["end"])), "--ob200", str(ob),
                        "--trades", str(tr), "--variants", str(lvj), "--out", str(lv), "--latency-ms", str(LATENCY_MS)] + (["--warmup-bars", ",".join(str(w) for w in warm)] if warm else []) + grid_extra)
        log(f"{key} live: {live_msg}")
        lerr = verify(lv, 3)
        errs += [f"live: {e}" for e in lerr if "bars" not in e]
        gzip_fills(lv)
    if errs:
        man.put("days", key, {"status": "failed", "errors": errs, "raw_kept": str(raw.relative_to(ROOT))})
        raise RuntimeError(f"{key}: verification failed: {errs}")
    gzip_fills(out)
    bp = bars_path(sym, d)
    bp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "bars_1m.csv", bp)
    shutil.rmtree(raw, ignore_errors=True)
    for k in ("ob200", "trades"):
        r = man.get("downloads", f"{key}:{k}") or {}
        man.put("downloads", f"{key}:{k}", {**r, "deleted": True})
    man.put("days", key, {"status": "done", "seconds": round(time.time() - t0, 1), "raw_bytes": raw_bytes, "raw_deleted": True,
                          "atx": msg, "live": live_msg, "out_bytes": du_bytes(out), "grid_override": grid_extra})


def run_all(symbols: list[str] | None = None) -> None:
    write_split()
    man = Manifest(MANIFEST)
    st0 = storage(refresh=True)
    log(f"storage before: {json.dumps({k: round(v, 3) for k, v in st0.items()})} budget {budget_gb()} GB")
    man.put("storage", "first_run" if not man.get("storage", "first_run") else "latest_run", {"status": "done", **st0})
    syms = symbols or SYMBOLS
    # chronological (a day's bars warm up the next), illiquid before BTC within a day
    for d in DAYS:
        for s in sorted(syms, key=lambda x: (x == "BTCUSDT", x)):
            try:
                process(man, s, d)
            except Exception as e:  # noqa: BLE001 - logged in the manifest and the run log, next day continues
                log(f"{s}:{d}: FAILED {e!r}"[:3000])
                if "storage" in str(e) or "transient raw" in str(e):
                    raise
    st1 = storage(refresh=True)
    man.put("storage", "after_run", {"status": "done", **st1})
    log(f"storage after: {json.dumps({k: round(v, 3) for k, v in st1.items()})}")


def main(argv: list[str] | None = None) -> None:
    a = (argv if argv is not None else sys.argv[1:]) or ["run"]
    if a[0] == "split":
        print(json.dumps({k: v for k, v in write_split().items() if k in ("fit_days", "held_days", "written")}, indent=1))
    elif a[0] == "run":
        run_all(a[1].split(",") if len(a) > 1 else None)
    elif a[0] == "report":
        from ai_trader.research.mm_replay_report import main as rep

        rep()
    else:
        raise SystemExit(f"unknown command {a[0]}")


if __name__ == "__main__":
    main()
