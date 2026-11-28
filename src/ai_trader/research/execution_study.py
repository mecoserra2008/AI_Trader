"""Execution studies on replayed real Bybit order books (agent EXE, 2026-11-15; H-EXE1, H-MMR3).

Research only: no orders, no Bybit trading calls. Market = the real ob200 book (quote-saver.bycsi.com) and the real
public trade tape (public.bybit.com) of each symbol-day, replayed through at-match by the Rust CLI:

  H-EXE1  `atx exec-replay` (crates/atx/src/exec_replay.rs): each historical signal of the paper book's two sleeves
          (shock_regime, trend_4h; agent.session.analyse, point in time) entered (a) with a market order through the
          real book and (b) with a PostOnly limit joining the touch, re-pegged, cancelled after N in {1, 5, 15} min
          (three queue models); both arms exit on the sleeve's own stop / target.
  H-MMR3  `atx mm-replay` (agent M's replay, crates/atx/src/mm_replay.rs) with a regime gate per variant: quote only
          in quiet sideways phases, stop when jump intensity rises; gated vs ungated at the same width.

Pipeline (one symbol-day at a time, checkpointed in data/research/execution/manifest.json; nothing is downloaded twice):
  storage check -> download ob200 + trades into data/raw/execution/<sym>/<date>/ -> run the replays the day needs ->
  verify -> delete the raw files.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.execution_study signals   # H-EXE1 signal set (no P&L)
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.execution_study gates     # H-MMR3 gate files (no P&L)
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.execution_study design    # freeze design.json
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.execution_study run       # all symbol-days (resumable)
    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.execution_study report
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "execution"
RAW = ROOT / "data" / "raw" / "execution"
MANIFEST = OUT / "manifest.json"
DESIGN = OUT / "design.json"
SIGNALS = OUT / "signals.json"
LOG = OUT / "run.log"
# Private copy of the atx binary (target/debug/atx.exe is rebuilt by other agents); built from a private target dir.
ATX = OUT / "bin" / "atx-exe.exe"
MMR = ROOT / "data" / "research" / "mm_replay"          # agent M's study: split.json, variants, warm-up bars
TRADES_URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob200.data.zip"
UA = {"User-Agent": "Mozilla/5.0 (AI_Trader research; public archives)"}
GB = 1024**3
RAW_CAP_GB = 2.0
LATENCY_MS = 150
WORKERS = 2
CUTOFF = date(2026, 7, 1)

# ---- H-EXE1 design constants (frozen in design.json before any P&L)
EXE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SIGNAL_START, SIGNAL_END = date(2026, 1, 5), date(2026, 11, 13)      # slots in [start 00:00, end 24:00) UTC
SLOT_HOURS = (0, 4, 8, 12, 16, 20)
AS_OF_DELAY = timedelta(minutes=5)       # a session is prepared ~5 min after its slot (real sessions: 5-9 min)
WINDOWS_MIN = [1, 5, 15]
BOOK_EQUITY = 10_000.0                   # agent.decision.BOOK_EQUITY
STOP_CAP = 4.0                           # H-STOP1 prior (the applied cap before the first posterior update)


def log(msg: str) -> None:
    line = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


# ======================================================================================================================
# H-EXE1 signals: the paper book's candidate generator (agent.session.analyse) at every 4h slot, point in time
# ======================================================================================================================


def risk_cfg() -> dict:
    from ai_trader.core.config import load

    return load().risk


def _fast_session():
    """agent.session with market_frame served from one frame per (symbol, freq) truncated at known_ts <= as_of.

    Equivalent to the slow path (each call re-reads the lake as of t): checked on 36 random slots x symbols in 2026
    (identical candidates, sigmas, cells, trend scores; scratch check 2026-11-14). The frames are causal recursions of
    bars with known_ts <= t, so truncation reproduces the as-of read. Flow / liquidation context is display-only."""
    import ai_trader.agent.session as S

    orig = S.market_frame
    end = datetime.combine(SIGNAL_END + timedelta(days=1), datetime.min.time(), tzinfo=UTC) + timedelta(hours=1)
    cache: dict = {}

    def mf(sym, freq, t, **kw):
        k = (sym, freq, tuple(sorted(kw.items())))
        if k not in cache:
            cache[k] = orig(sym, freq, end, **kw)
        return cache[k].filter(pl.col("known_ts") <= t)

    S.market_frame = mf
    S.flow_context = lambda sym, t, minutes=60: ("", {})
    S.liq_context = lambda sym, t: ("", {})
    return S


def generate_signals() -> list[dict]:
    """Every session candidate of the two sleeves for BTC / ETH / SOL at each 4h slot (as_of = slot + 5 min), kept when
    (1) the session did not block it (weekend rule S4, stop inside the minimum distance, no sigma), (2) its net E[R]
    after costs is > 0 (the paper gate, risk.yaml min_net_expected_r_paper), and (3) for trend_4h, it is an ONSET: the
    same symbol x side had no kept trend_4h candidate at the previous slot (the tilt is a state, re-proposed every slot
    while it lasts; an onset is when the paper book would first see it). shock_regime candidates are new by
    construction (lookback 4 bars = one slot). Stop cap = the 4-sigma prior, edges = the documented priors (the
    posteriors start 2026-11-14). Outcome-free: no price after as_of is read."""
    S = _fast_session()
    k = risk_cfg()
    gate = float(k.get("min_net_expected_r_paper", 0.0))
    out: list[dict] = []
    prev_trend: dict[str, str | None] = {s: None for s in EXE_SYMBOLS}
    d = SIGNAL_START
    n_slots = 0
    t0 = time.time()
    while d <= SIGNAL_END:
        for h in SLOT_HOURS:
            slot = datetime(d.year, d.month, d.day, h, tzinfo=UTC)
            as_of = slot + AS_OF_DELAY
            n_slots += 1
            for sym in EXE_SYMBOLS:
                try:
                    _, cands, st = S.analyse(sym, as_of, lookback_bars=4, cap=STOP_CAP, edges=dict(S.PRIOR_EDGE))
                except Exception as e:  # noqa: BLE001 - logged, slot skipped for this symbol
                    log(f"signals {sym} {as_of}: analyse failed {e!r}")
                    prev_trend[sym] = None
                    continue
                cur_trend = None
                for c in cands:
                    er = S.net_er(c)
                    keep = c.blocked is None and er > gate
                    if c.sleeve == "trend_4h" and keep:
                        cur_trend = c.side
                    if not keep:
                        continue
                    if c.sleeve == "trend_4h" and prev_trend[sym] == c.side:
                        continue
                    row = asdict(c)
                    row.update({"slot": slot.isoformat(), "as_of": as_of.isoformat(), "ts_ms": ms(as_of), "net_er": er, "bar_ts": st["bar_ts"],
                                "data_max_ts": st["data_max_ts"], "cell": st["cell"], "har_day": st["har_day"], "date": as_of.date().isoformat()})
                    row["id"] = f"{c.sleeve[:5]}-{sym[:3]}-{as_of:%Y%m%dT%H%M}"
                    if datetime.fromisoformat(st["data_max_ts"]) > as_of:
                        raise RuntimeError(f"look-ahead: {sym} data_max_ts {st['data_max_ts']} > as_of {as_of}")
                    out.append(row)
                prev_trend[sym] = cur_trend
        if d.day == 1:
            log(f"signals: {d} {len(out)} so far ({time.time() - t0:.0f}s)")
        d += timedelta(days=1)
    log(f"signals: {n_slots} slots, {len(out)} signals")
    return out


def sized_qty(c: dict) -> float:
    """Fixed-risk size as the risk engine computes it for a MARKET entry (at-risk: qty = risk / (|entry - stop| +
    entry * round-trip cost)); the same qty is used by both arms so the comparison is per signal at equal size."""
    k = risk_cfg()
    risk = BOOK_EQUITY * float(k["risk_per_trade_pct"]) / 100.0
    cost = (k["costs"]["taker_bps"] + k["costs"]["slippage_bps"]) * 2 / 1e4
    return risk / (abs(c["entry"] - c["stop"]) + c["entry"] * cost)


def write_signals() -> dict:
    if SIGNALS.is_file():
        return json.loads(SIGNALS.read_text(encoding="utf-8"))
    sig = generate_signals()
    for c in sig:
        c["qty"] = sized_qty(c)
    obj = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "EXE", "hypothesis": "H-EXE1", "rule": generate_signals.__doc__,
           "symbols": EXE_SYMBOLS, "start": SIGNAL_START.isoformat(), "end": SIGNAL_END.isoformat(), "signals": sig}
    write_json(SIGNALS, obj)
    return obj


# ======================================================================================================================
# H-MMR3 gates: quiet sideways phases, point in time, from the 1m tape bars of agent M's replays
# ======================================================================================================================

MMR_GROUPS = {"btc": ["BTCUSDT"], "illiquid": ["TRXUSDT", "ETCUSDT", "XLMUSDT"]}
MMR_SYMBOLS = [s for g in MMR_GROUPS.values() for s in g]
MMR_GROUP_OF = {s: g for g, ss in MMR_GROUPS.items() for s in ss}
GATE_DIR = OUT / "gates"
GATE_RULES = {
    "V": "short-horizon vol in its bottom tercile: RV60(t) = sqrt(sum of squared 1m log returns over the 60 closed minutes before t) "
         "<= the 1/3 quantile of RV60 over the symbol's FIT-day minutes",
    "R": "narrow 4h range: |ln(close(t) / close(t - 240 min))| < sigma_4h(t) = sqrt(48) x sigma_5m(t), sigma_5m = RMS of the 288 "
         "clock-5m log returns of the 24 h before t",
    "S": "no shock in the last 6 h: no closed clock hour with |ln(close / previous hour's close)| >= 4 x sigma_1h, sigma_1h = "
         "sqrt(12) x sigma_5m as of that hour's start, closing in (t - 6 h, t] (mirrors the shock sleeve's |z| >= 4)",
    # J_field is DEPRECATED for live use (D3, 2026-11-27; H-WS05-6): see field_pjump. Kept to reproduce the study.
    "J_field": "BTCUSDT: the probability field's P(jump in the next hour) from the frozen pre-cutoff fit "
               "(data/models/field/btc_field_cutoff_2026-07-01.json; state from the 1h lake frame, row valid from its known_ts) "
               "<= its median over the FIT-day hours",
    "J_hawkes": "illiquid (and BTC as a secondary): Hawkes-type jump excitation lambda(t) = sum over past jump minutes j of "
                "2^(-(t - t_j) / 60 min), jump minute = |r_1m| > 4 x sigma_1m (RMS of the 1440 1m returns before it); "
                "<= its median over the FIT-day minutes",
    "warm": "a component needs its full look-back (24 h of 1m bars, 288 5m returns, 240 min); before that the gate is off",
}


def _minute_closes(sym: str) -> tuple[np.ndarray, np.ndarray]:
    """(minute_ms, close) on a full 1m grid over the warm-up + sample days (venue-style: minutes without trades repeat the
    previous close), from data/research/mm_replay/bars/<sym>/<day>.csv (agent M's tape bars)."""
    days = sorted((MMR / "bars" / sym).glob("*.csv"))
    df = pl.concat([pl.read_csv(p, schema_overrides={"minute_ms": pl.Int64}) for p in days]).unique("minute_ms").sort("minute_ms")
    lo, hi = int(df["minute_ms"].min()), int(df["minute_ms"].max())
    grid = pl.DataFrame({"minute_ms": np.arange(lo, hi + 60_000, 60_000, dtype=np.int64)})
    g = grid.join(df.select("minute_ms", "close"), on="minute_ms", how="left").with_columns(pl.col("close").forward_fill())
    return g["minute_ms"].to_numpy(), g["close"].to_numpy().astype(float)


def gate_components(sym: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Per decision time t = the start of minute i (uses minutes < i only, i.e. bars closed by t): the gate components."""
    t, c = _minute_closes(sym)
    n = len(t)
    idx = np.arange(n)
    lr = np.zeros(n)
    lr[1:] = np.log(c[1:] / c[:-1])               # return of minute i (known at t[i] + 1 min)
    cs = np.concatenate([[0.0], np.cumsum(lr**2)])  # cs[i] = sum of squared returns of minutes < i
    rv60 = np.sqrt(np.maximum(cs[idx] - cs[np.maximum(idx - 60, 0)], 0.0))
    s1m = np.sqrt(np.maximum(cs[idx] - cs[np.maximum(idx - 1440, 0)], 0.0) / 1440.0)
    # the close known at t[i] is minute i-1's close
    lc = np.log(np.concatenate([[c[0]], c[:-1]]))
    # clock 5m returns ending at block boundaries (t % 5 min == 0): lc[b] - lc[b - 5]
    bidx = np.where(t % 300_000 == 0)[0]
    r5 = np.zeros(len(bidx))
    ok = bidx >= 5
    r5[ok] = lc[bidx[ok]] - lc[bidx[ok] - 5]
    c5 = np.concatenate([[0.0], np.cumsum(r5**2)])
    nb = np.searchsorted(bidx, idx, side="right")   # boundaries at or before t[i]
    cnt = np.minimum(nb, 288)
    s5 = np.sqrt((c5[nb] - c5[nb - cnt]) / np.maximum(cnt, 1))
    s5[cnt < 288] = np.nan
    s4h = np.sqrt(48.0) * s5
    r4h = np.full(n, np.nan)
    r4h[240:] = lc[240:] - lc[:-240]
    # shocks: clock hours closing at t[k] (t[k] % 1 h == 0): lc[k] - lc[k - 60] vs 4 sigma_1h at the hour's start
    shock_close = []
    for k in np.where(t % 3_600_000 == 0)[0]:
        if k < 60:
            continue
        sig = np.sqrt(12.0) * s5[k - 60]
        if np.isfinite(sig) and sig > 0 and abs(lc[k] - lc[k - 60]) >= 4.0 * sig:
            shock_close.append(int(t[k]))
    sc = np.array(shock_close, dtype=np.int64)
    last_shock = np.full(n, -(10**15), dtype=np.int64)
    if len(sc):
        j = np.searchsorted(sc, t, side="right") - 1
        last_shock = np.where(j >= 0, sc[np.maximum(j, 0)], -(10**15))
    no_shock = (t - last_shock) >= 6 * 3_600_000
    # Hawkes-type excitation: jump minute i-1 (|r| > 4 sigma_1m before it) counts from t[i]; half-life 60 min
    jump = np.zeros(n, bool)
    jump[1440:] = np.abs(lr[1440:]) > 4.0 * s1m[1440:]
    lam = np.zeros(n)
    dec = 2.0 ** (-1.0 / 60.0)
    x = 0.0
    for i in range(1, n):
        x = x * dec + (1.0 if jump[i - 1] else 0.0)
        lam[i] = x
    out = pl.DataFrame({"minute_ms": t, "rv60": rv60, "s5": s5, "r4h": r4h, "s4h": s4h, "no_shock": no_shock, "lam": lam,
                        "warm": (idx >= 1441) & np.isfinite(s4h) & np.isfinite(r4h)})
    out = out.with_columns(pl.from_epoch("minute_ms", time_unit="ms").dt.date().cast(pl.Utf8).alias("day"))
    return out, sc


def field_pjump() -> pl.DataFrame:
    """BTC P(jump in the next hour) from the frozen pre-cutoff field fit, per 1h row valid from its known_ts.

    DEPRECATED for live use (user decision D3, 2026-11-27): the field's p_jump is a latent mixture weight, not the chance
    of a visible jump (H-WS05-6, R-WS05-1). Kept only to reproduce this study; live jump gating is the Hawkes gate in
    data/live/gates.json (src/ai_trader/live/gates.py)."""
    from ai_trader.field.model import Field, coords_matrix
    from ai_trader.field.state import build_state, load_frame

    fld = Field.load(ROOT / "data" / "models" / "field" / "btc_field_cutoff_2026-07-01.json")
    fr = load_frame(datetime(2026, 11, 14, 1, tzinfo=UTC), start=datetime(2026, 5, 1, tzinfo=UTC))
    st = build_state(fr, fld.profile, fld.cfg)
    X = coords_matrix(st)
    good = np.all(np.isfinite(X), axis=1) & np.isfinite(st["vf"].to_numpy())
    st, X = st.filter(pl.Series(good)), X[good]
    _, _, p = fld.params(X, np.log(st["vf"].to_numpy()))
    return pl.DataFrame({"valid_from_ms": st["known_ts"].dt.epoch("ms").cast(pl.Int64), "p_jump": p}).sort("valid_from_ms")


def write_gates() -> dict:
    """Gate CSVs per symbol-day and gate (all, onlyV/R/S/J; BTC also allHawkes). Thresholds from FIT days only."""
    split = json.loads((MMR / "split.json").read_text(encoding="utf-8"))
    fit, days = split["fit_days"], split["days"]
    summary: dict[str, Any] = {"rules": GATE_RULES, "fit_days": fit, "symbols": {}}
    pj = field_pjump()
    for sym in MMR_SYMBOLS:
        comp, shocks = gate_components(sym)
        fitm = comp.filter(pl.col("day").is_in(fit) & pl.col("warm"))
        q33 = float(fitm["rv60"].quantile(1 / 3, "nearest"))
        lam_med = float(fitm["lam"].median())
        comp = comp.with_columns((pl.col("rv60") <= q33).alias("V"), (pl.col("r4h").abs() < pl.col("s4h")).fill_null(False).alias("R"),
                                 pl.col("no_shock").alias("S"), (pl.col("lam") <= lam_med).alias("J_hawkes"))
        info: dict[str, Any] = {"rv60_q33_fit": q33, "lam_median_fit": lam_med,
                                "shock_hours_close": [datetime.fromtimestamp(x / 1000, UTC).isoformat() for x in shocks]}
        if sym == "BTCUSDT":
            j = comp.select("minute_ms").join_asof(pj, left_on="minute_ms", right_on="valid_from_ms", strategy="backward")
            comp = comp.with_columns(j["p_jump"].alias("p_jump"))
            f0, f1 = ms(datetime.fromisoformat(fit[0]).replace(tzinfo=UTC)), ms(datetime.fromisoformat(fit[-1]).replace(tzinfo=UTC) + timedelta(days=1))
            p_med = float(pj.filter((pl.col("valid_from_ms") >= f0) & (pl.col("valid_from_ms") < f1))["p_jump"].median())
            info["p_jump_median_fit"] = p_med
            comp = comp.with_columns((pl.col("p_jump") <= p_med).fill_null(False).alias("J_field"))
            jcol = "J_field"
        else:
            jcol = "J_hawkes"
        w = pl.col("warm")
        comp = comp.with_columns((pl.col("V") & pl.col("R") & pl.col("S") & pl.col(jcol) & w).alias("all"), (pl.col("V") & w).alias("onlyV"),
                                 (pl.col("R") & w).alias("onlyR"), (pl.col("S") & w).alias("onlyS"), (pl.col(jcol) & w).alias("onlyJ"))
        gates = ["all", "onlyV", "onlyR", "onlyS", "onlyJ"]
        if sym == "BTCUSDT":
            comp = comp.with_columns((pl.col("V") & pl.col("R") & pl.col("S") & pl.col("J_hawkes") & w).alias("allHawkes"))
            gates.append("allHawkes")
        info["on_share"] = {}
        for d in days:
            dd = comp.filter(pl.col("day") == d)
            assert dd.height == 1440, (sym, d, dd.height)
            for gname in gates:
                p = GATE_DIR / sym / d / f"{gname}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("valid_from_ms,on\n" + "".join(f"{a},{int(b)}\n" for a, b in zip(dd["minute_ms"], dd[gname], strict=True)), encoding="utf-8")
            info["on_share"][d] = {g: float(dd[g].mean()) for g in gates + ["V", "R", "S", "J_hawkes"] + (["J_field"] if sym == "BTCUSDT" else [])}
        info["jump_measure"] = jcol
        summary["symbols"][sym] = info
        fs = np.mean([info["on_share"][d]["all"] for d in fit])
        hs = np.mean([info["on_share"][d]["all"] for d in days if d not in fit])
        log(f"gates {sym}: rv60 q33 {q33:.6f}, lambda median {lam_med:.4f}, {len(shocks)} shock hours; gate on share fit {fs:.2f} held {hs:.2f}")
    write_json(GATE_DIR / "summary.json", summary)
    return summary


# ======================================================================================================================
# design (frozen before any P&L)
# ======================================================================================================================

PRIMARY_WIDTH = {"btc": "L_k1_live", "illiquid": "L_k1_sym"}     # agent M's fit-day choices (H-MMR1)
SECONDARY_WIDTHS = ["F_w12_sym", "F_w24_sym"]
ABLATIONS = ["onlyV", "onlyR", "onlyS", "onlyJ"]


def mmr_variants(sym: str, d: str) -> list[dict]:
    """H-MMR3 variants of one symbol-day: agent M's variant specs (split.json) x gate. Names `<width>__<gate>`."""
    split = json.loads((MMR / "split.json").read_text(encoding="utf-8"))
    g = MMR_GROUP_OF[sym]
    base = {v["name"]: {k: x for k, x in v.items() if k != "family"} for v in split["grid"][g]}
    out = []

    def add(width: str, gate: str | None, flatten: bool = False):
        v = json.loads(json.dumps(base[width]))
        v["name"] = f"{width}__{gate or 'ungated'}{'_flat' if flatten else ''}"
        if gate:
            v["gate"] = str((GATE_DIR / sym / d / f"{gate}.csv").relative_to(ROOT)).replace("\\", "/")
            v["gate_flatten"] = flatten
        out.append(v)

    w0 = PRIMARY_WIDTH[g]
    add(w0, None)
    add(w0, "all")
    add(w0, "all", True)
    for a in ABLATIONS:
        add(w0, a)
    if sym == "BTCUSDT":
        add(w0, "allHawkes")
    for w in SECONDARY_WIDTHS:
        add(w, None)
        add(w, "all")
        add(w, "all", True)
    return out


def exe_days(sig: list[dict]) -> dict[tuple[str, str], list[dict]]:
    by: dict[tuple[str, str], list[dict]] = {}
    for c in sig:
        by.setdefault((c["symbol"], c["date"]), []).append(c)
    return by


def write_design(force: bool = False) -> dict:
    sig = json.loads(SIGNALS.read_text(encoding="utf-8"))["signals"]
    split = json.loads((MMR / "split.json").read_text(encoding="utf-8"))
    gs = json.loads((GATE_DIR / "summary.json").read_text(encoding="utf-8"))
    k = risk_cfg()
    days = sorted({c["date"] for c in sig})
    obj = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "EXE", "hypotheses": ["H-EXE1", "H-MMR3"],
        "H-EXE1": {
            "signals": {"file": str(SIGNALS.relative_to(ROOT)), "n": len(sig), "rule": generate_signals.__doc__,
                        "by_sleeve": {s: sum(c["sleeve"] == s for c in sig) for s in ("shock_regime", "trend_4h")},
                        "symbols": EXE_SYMBOLS, "slots": f"{SIGNAL_START} .. {SIGNAL_END}, 00/04/08/12/16/20 UTC, as_of = slot + 5 min"},
            "symbol_days": sorted(f"{s}:{d}" for s, d in exe_days(sig)), "n_symbol_days": len(exe_days(sig)), "days": days,
            "t0": "the order is sent at as_of (slot + 5 min); the candidate (entry = last 1h close, stop, target) uses data known at as_of only",
            "arms": "market: taker order executed latency_ms after t0 against the replayed ob200 book (walks the levels); limit_N: PostOnly at "
                    "the touch, re-pegged to the touch (amend, back of the new queue), cancel sent at t0 + N min, partial fills kept, "
                    "unfilled = 0 R (crates/atx/src/exec_replay.rs)",
            "windows_min": WINDOWS_MIN, "queue_primary": "prorata", "queues_reported": ["pessimistic", "prorata", "optimistic"], "latency_ms": LATENCY_MS,
            "size": "qty = 0.5% x 10,000 USDT / (|entry - stop| + entry x 15 bp) (risk engine, market-entry costs), floored to the lot step; the "
                    "same qty for every arm of a signal",
            "costs": {"limit_entry": f"maker {k['costs']['maker_bps']} bp (config/fees.yaml crypto)", "market_entry": f"taker {k['costs']['taker_bps']} bp + "
                      "the real book's slippage (VWAP vs touch)", "exit": "taker 5.5 + slippage 2.0 bp (fees.yaml exit leg), identical rule in every arm"},
            "exit": "from the last entry fill: the first print at/through the stop or target in the day's tape (exit at that print's price); "
                    "later days: lake 1h bybit last-price bars, stop first when both are touched in a bar (at-exec's conservative rule), "
                    "a gap through the level fills at the bar open; still open at the last bar = marked at its close (exit cost charged)",
            "R": "net R = [side x filled_qty x (exit - vwap) - entry fee - exit fee] / (qty x |entry - stop|); entry = the candidate's price",
            "statistic": "paired per-signal difference limit_N - market in net R, both sleeves pooled; mean with a day-clustered bootstrap "
                         "(clusters = UTC days of t0, B = 10000, percentile 95% CI); pro-rata primary, other queues reported",
            "segments": "llm_in_sample (t0 < 2026-07-01) and clean_holdout (t0 >= 2026-07-01) recorded separately (plus pooled)",
            "verdict": "supports: for at least one N, mean > 0 with the CI above 0 (pro-rata, pooled sleeves); contradicts: for every N the CI "
                       "is below 0; else inconclusive",
            "missed_winners": "per N (pro-rata): share of signals unfilled / partly filled; their market-arm net R (mean, share > 0, share "
                              "reaching the target); decomposition of the difference into the filled part (price + fee) and the missed part",
        },
        "H-MMR3": {
            "days": split["days"], "fit_days": split["fit_days"], "held_days": split["held_days"], "symbols": MMR_GROUPS,
            "split": "agent M's split (data/research/mm_replay/split.json, chronological, written before any P&L)",
            "variants": {s: [v["name"] for v in mmr_variants(s, split["days"][0])] for s in MMR_SYMBOLS},
            "widths": {"primary": PRIMARY_WIDTH, "secondary": SECONDARY_WIDTHS, "specs": "agent M's variant specs (split.json grid), BTC max_net_usd 0"},
            "gate": {**gs["rules"], "all": "V and R and S and J (J = J_field for BTCUSDT, J_hawkes for TRX/ETC/XLM); a decision tick with the "
                     "gate off quotes exit-only (reduce-only, the live wind-down path, no new inventory); `_flat` variants instead cancel and "
                     "close at market (taker) when the gate turns off with inventory",
                     "thresholds_fit": {s: {kk: v for kk, v in i.items() if kk.endswith("_fit")} for s, i in gs["symbols"].items()},
                     "jump_measure": {s: i["jump_measure"] for s, i in gs["symbols"].items()}},
            "engine": "atx mm-replay (live atx mm quoting code; ob200 + trade tape; queue models; 150 ms latency; fees crypto maker 2.0 / taker 5.5 bp; "
                      "each symbol-day flat at 00:00, marked at 24:00; warm-up bars of the two previous days)",
            "statistic": "paired per symbol-day difference gated(all, exit-only) - ungated at the primary width on HELD days, per group, pro-rata; "
                         "day-clustered bootstrap (B = 10000) 95% CI; decomposition (spread, adverse 1/10/60 s, inventory, fees, fills) and "
                         "single-component gates reported; secondary widths and the _flat variants reported, not part of the verdict",
            "verdict": "supports: difference > 0 with the CI above 0 in at least one group; contradicts: CI below 0 in both groups; else inconclusive",
            "segment": "all days after 2026-07-01: clean_holdout",
        },
        "data": {"ob200": OB_URL, "trades": TRADES_URL, "raw_dir": str(RAW.relative_to(ROOT)), "raw_cap_gb": RAW_CAP_GB,
                 "manifest": str(MANIFEST.relative_to(ROOT)), "one_at_a_time": "download a symbol-day, run every replay it needs, verify, delete"},
    }
    if DESIGN.is_file() and not force:
        old = json.loads(DESIGN.read_text(encoding="utf-8"))
        if {kk: v for kk, v in old.items() if kk != "written"} != json.loads(json.dumps({kk: v for kk, v in obj.items() if kk != "written"}, default=float)):
            raise RuntimeError("design.json exists and differs: the design is frozen")
        return old
    write_json(DESIGN, obj)
    return obj


# ======================================================================================================================
# downloads and replays (one symbol-day at a time)
# ======================================================================================================================


LOCK = threading.Lock()   # two workers share the manifest and the storage checks


class Manifest:
    def __init__(self, path: Path = MANIFEST):
        self.path = path
        self.d = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def get(self, sec: str, key: str) -> dict | None:
        return self.d.get(sec, {}).get(key)

    def put(self, sec: str, key: str, rec: dict) -> None:
        with LOCK:
            self.d.setdefault(sec, {})[key] = {**rec, "updated": datetime.now(UTC).isoformat(timespec="seconds")}
            write_json(self.path, self.d)


def _storage() -> dict[str, float]:
    from ai_trader.sim.ob_gen import storage_gb

    return storage_gb()


def fetch(man: Manifest, key: str, url: str, dest: Path) -> Path:
    from ai_trader.sim.ob_gen import budget_gb, du_bytes, http_get

    rec = man.get("downloads", key)
    if rec and rec.get("deleted"):
        raise RuntimeError(f"{key} was downloaded and deleted before: not downloading twice")
    if dest.is_file() and dest.stat().st_size > 0 and rec and rec.get("status") == "ok":
        return dest
    with LOCK:
        st = _storage()
        raw_now = du_bytes(RAW) / GB
    if st["total_gb"] + 0.5 >= budget_gb() - 1.0:
        raise RuntimeError(f"storage {st['total_gb']:.2f} GB near budget {budget_gb()} GB: no download")
    if raw_now + 0.5 > RAW_CAP_GB:
        raise RuntimeError(f"transient raw {raw_now:.2f} GB + 0.5 GB > {RAW_CAP_GB} GB")
    t = time.time()
    r = http_get(url, dest)
    man.put("downloads", key, {**r, "url": url, "path": str(dest.relative_to(ROOT)), "seconds": round(time.time() - t, 1),
                               "storage_before_gb": st["total_gb"], "raw_before_gb": raw_now, "deleted": False})
    if r["status"] != "ok":
        raise RuntimeError(f"download {key}: {r}")
    return dest


def atx(args: list[str], timeout: int = 3 * 3600) -> str:
    r = subprocess.run([str(ATX), *args], check=False, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"atx {' '.join(args[:3])} failed: {r.stderr[-2000:]}")
    return r.stdout.strip()


def atx_grid(args: list[str], extra: list[str]) -> tuple[str, list[str]]:
    """Run atx; when the day's prices are off the lake's (latest) tick grid, infer the tick from the offending price and
    retry with --tick (as agent M's study)."""
    import re

    for _ in range(3):
        try:
            return atx(args + extra), extra
        except RuntimeError as e:
            m = re.search(r"price ([0-9.]+) off the tick grid", str(e))
            if not m or "--tick" in extra:
                raise
            dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
            extra = extra + ["--tick", f"{10.0 ** -dec:.{dec}f}"]
            log(f"tick grid differs from the lake rules on this day: retrying with {extra}")
    raise RuntimeError("tick inference failed")


def mm_grid_override(sym: str, d: str) -> list[str]:
    mm = json.loads((MMR / "manifest.json").read_text(encoding="utf-8"))
    return list(mm.get("days", {}).get(f"{sym}:{d}", {}).get("grid_override") or [])


def run_mmr3_day(sym: str, d: str, ob: Path, tr: Path) -> dict:
    out = OUT / "mmr3" / "days" / sym / d
    vj = OUT / "mmr3" / "variants" / f"{sym}_{d}.json"
    variants = mmr_variants(sym, d)
    write_json(vj, {"variants": variants})
    prev = [(date.fromisoformat(d) - timedelta(days=k)).isoformat() for k in (2, 1)]
    warm = [MMR / "bars" / sym / f"{x}.csv" for x in prev]
    warm = [w for w in warm if w.is_file()]
    args = ["mm-replay", "--symbol", sym, "--date", d, "--ob200", str(ob), "--trades", str(tr), "--variants", str(vj), "--out", str(out),
            "--latency-ms", str(LATENCY_MS)] + (["--warmup-bars", ",".join(str(w) for w in warm)] if warm else [])
    msg, extra = atx_grid(args, mm_grid_override(sym, d))
    st = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    mk = st["market"]
    errs = []
    if mk["book_messages"] <= 0 or mk["trades"] <= 0:
        errs.append("empty market")
    if mk["engine_rejects"] or mk["engine_trades"] or mk["replayer_check_mismatches"]:
        errs.append(f"engine rejects {mk['engine_rejects']} trades {mk['engine_trades']} mismatches {mk['replayer_check_mismatches']}")
    if len(st["quoters"]) != 3 * len(variants):
        errs.append(f"{len(st['quoters'])} quoters, expected {3 * len(variants)}")
    bad = [q["name"] for q in st["quoters"] if abs(q["pnl"]["ledger_check"]) > 1e-6]
    if bad:
        errs.append(f"ledger identity broken: {bad[:3]}")
    f = out / "fills.jsonl"
    if f.is_file():
        with f.open("rb") as a, gzip.open(out / "fills.jsonl.gz", "wb") as b:
            shutil.copyfileobj(a, b)
        f.unlink()
    (out / "bars_1m.csv").unlink(missing_ok=True)   # agent M kept the day's bars already
    return {"atx": msg, "errors": errs, "grid_override": extra}


def run_exe1_day(sym: str, d: str, sigs: list[dict], ob: Path, tr: Path) -> dict:
    out = OUT / "exe1" / "days" / sym / d
    sj = OUT / "exe1" / "signals" / f"{sym}_{d}.json"
    write_json(sj, {"windows_min": WINDOWS_MIN, "queues": ["pessimistic", "prorata", "optimistic"],
                    "signals": [{"id": c["id"], "ts_ms": c["ts_ms"], "side": "Buy" if c["side"] == "long" else "Sell", "qty": c["qty"],
                                 "stop": c["stop"], "take_profit": c["take_profit"]} for c in sigs]})
    args = ["exec-replay", "--symbol", sym, "--date", d, "--ob200", str(ob), "--trades", str(tr), "--signals", str(sj), "--out", str(out),
            "--latency-ms", str(LATENCY_MS)]
    msg, extra = atx_grid(args, [])
    st = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    mk = st["market"]
    errs = []
    if mk["book_messages"] <= 0 or mk["trades"] <= 0:
        errs.append("empty market")
    if mk["engine_rejects"] or mk["engine_trades"] or mk["replayer_check_mismatches"]:
        errs.append(f"engine rejects {mk['engine_rejects']} trades {mk['engine_trades']} mismatches {mk['replayer_check_mismatches']}")
    n = sum(1 for _ in (out / "results.jsonl").open(encoding="utf-8"))
    if n != len(sigs) * (1 + 3 * len(WINDOWS_MIN)):
        errs.append(f"{n} result rows, expected {len(sigs) * (1 + 3 * len(WINDOWS_MIN))}")
    return {"atx": msg, "errors": errs, "grid_override": extra}


def process(man: Manifest, sym: str, d: str, mmr3: bool, sigs: list[dict]) -> None:
    from ai_trader.sim.ob_gen import du_bytes

    key = f"{sym}:{d}"
    rec = man.get("days", key) or {}
    if rec.get("status") == "done":
        return
    t0 = time.time()
    raw = RAW / sym / d
    ob = fetch(man, f"{key}:ob200", OB_URL.format(s=sym, d=d), raw / f"{d}_{sym}_ob200.data.zip")
    tr = fetch(man, f"{key}:trades", TRADES_URL.format(s=sym, d=d), raw / f"{sym}{d}.csv.gz")
    raw_bytes = du_bytes(raw)
    res: dict[str, Any] = {}
    if mmr3 and not rec.get("mmr3_done"):
        res["mmr3"] = run_mmr3_day(sym, d, ob, tr)
        log(f"{key} mmr3: {res['mmr3']['atx']}")
    if sigs and not rec.get("exe1_done"):
        res["exe1"] = run_exe1_day(sym, d, sigs, ob, tr)
        log(f"{key} exe1: {res['exe1']['atx']}")
    errs = [f"{k}: {e}" for k, v in res.items() for e in v["errors"]]
    if errs:
        man.put("days", key, {"status": "failed", "errors": errs, "raw_kept": str(raw.relative_to(ROOT))})
        raise RuntimeError(f"{key}: verification failed: {errs}")
    shutil.rmtree(raw, ignore_errors=True)
    for k in ("ob200", "trades"):
        r = man.get("downloads", f"{key}:{k}") or {}
        man.put("downloads", f"{key}:{k}", {**r, "deleted": True})
    man.put("days", key, {"status": "done", "seconds": round(time.time() - t0, 1), "raw_bytes": raw_bytes, "raw_deleted": True,
                          "mmr3": res.get("mmr3"), "exe1": res.get("exe1"), "n_signals": len(sigs), "mmr3_done": mmr3, "exe1_done": bool(sigs)})


def run_all(only: str | None = None) -> None:
    design = write_design()
    sig = json.loads(SIGNALS.read_text(encoding="utf-8"))["signals"]
    by = exe_days(sig)
    mmr_days = {(s, d) for s in MMR_SYMBOLS for d in design["H-MMR3"]["days"]}
    # order: H-MMR3 days, then H-EXE1 clean_holdout days (newest first), then H-EXE1 llm_in_sample days (newest first)
    def rank(x: tuple[str, str]) -> tuple:
        if x in mmr_days:
            return (0, x[1], x[0])
        return (1 if date.fromisoformat(x[1]) >= CUTOFF else 2, -date.fromisoformat(x[1]).toordinal(), x[0])

    jobs = sorted(set(by) | mmr_days, key=rank)
    if only == "mmr3":
        jobs = [j for j in jobs if j in mmr_days]
    elif only == "exe1":
        jobs = [j for j in jobs if j in by]
    man = Manifest()
    st0 = _storage()
    log(f"run: {len(jobs)} symbol-days; storage before {json.dumps({k: round(v, 3) for k, v in st0.items()})}")
    if not man.get("storage", "first_run"):
        man.put("storage", "first_run", st0)
    # two workers (one single-threaded replay each); each downloads, replays and deletes its own symbol-day
    queue = list(jobs)
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            with LOCK:
                if not queue:
                    return
                s, d = queue.pop(0)
            try:
                process(man, s, d, (s, d) in mmr_days, by.get((s, d), []))
            except Exception as e:  # noqa: BLE001 - logged in the manifest and the run log, next day continues
                log(f"{s}:{d}: FAILED {e!r}"[:3000])
                if "storage" in str(e) or "transient raw" in str(e):
                    stop.set()

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    st1 = _storage()
    man.put("storage", "after_run", st1)
    log(f"run done; storage after {json.dumps({k: round(v, 3) for k, v in st1.items()})}")


# ======================================================================================================================
# analysis
# ======================================================================================================================

B_BOOT = 10_000


def boot(values: np.ndarray, clusters: np.ndarray, b: int = B_BOOT, seed: int = 0) -> dict:
    """Mean with a cluster bootstrap percentile 95% CI (clusters resampled with replacement, all their rows kept)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"n": 0, "clusters": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    cl = np.asarray(clusters)
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(b, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": len(v), "clusters": len(u), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def _bars_1h(sym: str) -> pl.DataFrame:
    from ai_trader.storage.lake import Lake

    d = Lake().read_asof("bars", datetime.now(UTC), symbols=[sym], filters={"freq": "1h", "venue": "bybit", "price_type": "last"},
                         start=datetime(2026, 1, 1, tzinfo=UTC))
    return d.select("event_ts", "open", "high", "low", "close").unique("event_ts").sort("event_ts")


def continue_exit(bars: pl.DataFrame, start: datetime, buy: bool, stop: float, tp: float) -> tuple[str, float, str]:
    """Exit on 1h bars from `start` (at-exec's conservative rule: stop first when both are touched; a gap through a level
    fills at the open); still open at the last bar = marked at its close."""
    b = bars.filter(pl.col("event_ts") >= start)
    o, h, lo, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
    ts = b["event_ts"]
    for i in range(len(o)):
        if buy:
            if o[i] <= stop:
                return "stop_gap", float(o[i]), str(ts[i])
            if o[i] >= tp:
                return "tp_gap", float(o[i]), str(ts[i])
            if lo[i] <= stop:
                return "stop", float(stop), str(ts[i])
            if h[i] >= tp:
                return "tp", float(tp), str(ts[i])
        else:
            if o[i] >= stop:
                return "stop_gap", float(o[i]), str(ts[i])
            if o[i] <= tp:
                return "tp_gap", float(o[i]), str(ts[i])
            if h[i] >= stop:
                return "stop", float(stop), str(ts[i])
            if lo[i] <= tp:
                return "tp", float(tp), str(ts[i])
    return ("open_end", float(c[-1]), str(ts[len(ts) - 1])) if len(c) else ("open_end", float("nan"), "")


def exe1_table() -> pl.DataFrame:
    """One row per signal x arm x queue with net R (the arms' P&L)."""
    sig = {c["id"]: c for c in json.loads(SIGNALS.read_text(encoding="utf-8"))["signals"]}
    k = risk_cfg()
    exit_bps = k["costs"]["taker_bps"] + k["costs"]["slippage_bps"]
    rows = []
    bars: dict[str, pl.DataFrame] = {}
    cont_cache: dict[str, tuple] = {}
    for f in sorted((OUT / "exe1" / "days").glob("*/*/results.jsonl")):
        sym = f.parent.parent.name
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            c = sig[r["signal"]]
            buy = r["side"] == "Buy"
            dsgn = 1.0 if buy else -1.0
            R0 = abs(c["entry"] - c["stop"])
            Q, F = r["qty"], r["filled_qty"]
            out = {"signal": r["signal"], "sleeve": c["sleeve"], "symbol": sym, "date": c["date"], "side": c["side"], "arm": r["arm"], "window_min": r["window_min"],
                   "queue": r["queue"], "fill_share": F / Q if Q else 0.0, "segment": "clean_holdout" if date.fromisoformat(c["date"]) >= CUTOFF else "llm_in_sample",
                   "vwap": r["vwap"], "mid0": r["mid0"], "bid0": r["bid0"], "ask0": r["ask0"], "amends": r["amends"], "postonly_rejects": r["postonly_rejects"],
                   "first_fill_ms": r["first_fill_ms"], "t0": r["t0"], "lots_through": r["lots_through"], "lots_queue": r["lots_queue"], "lots_cross": r["lots_cross"],
                   "mid_fill": r["mid_fill"], "mid_fill_60s": r["mid_fill_60s"], "mid_window_end": r["mid_window_end"], "entry": c["entry"], "stop": c["stop"], "tp": c["take_profit"]}
            if F <= 0:
                out.update({"net_R": 0.0, "gross_R": 0.0, "fee_R": 0.0, "exit_kind": None, "exit_px": None})
            else:
                if r["exit_kind"] in ("stop", "tp"):
                    kind, px = r["exit_kind"], r["exit_px"]
                else:
                    if r["signal"] not in cont_cache:
                        if sym not in bars:
                            bars[sym] = _bars_1h(sym)
                        start = datetime.fromisoformat(c["date"]).replace(tzinfo=UTC) + timedelta(days=1)
                        cont_cache[r["signal"]] = continue_exit(bars[sym], start, buy, c["stop"], c["take_profit"])
                    kind, px, _ = cont_cache[r["signal"]]
                p_in = r["vwap"]
                fee_in = F * p_in * r["entry_fee_bps"] / 1e4
                fee_out = F * px * exit_bps / 1e4
                gross = dsgn * F * (px - p_in)
                out.update({"net_R": (gross - fee_in - fee_out) / (Q * R0), "gross_R": gross / (Q * R0), "fee_R": -(fee_in + fee_out) / (Q * R0),
                            "exit_kind": kind, "exit_px": px})
            rows.append(out)
    return pl.DataFrame(rows, infer_schema_length=None)


def exe1_analysis(t: pl.DataFrame) -> dict:
    mk = t.filter(pl.col("arm") == "market").select("signal", pl.col("net_R").alias("R_mkt"), pl.col("exit_kind").alias("exit_mkt"),
                                                   pl.col("vwap").alias("vwap_mkt"), pl.col("fill_share").alias("fill_mkt"))
    lim = t.filter(pl.col("arm") == "limit").join(mk, on="signal")
    lim = lim.with_columns((pl.col("net_R") - pl.col("R_mkt")).alias("diff"),
                           (pl.col("net_R") - pl.col("fill_share") * pl.col("R_mkt")).alias("diff_filled_part"),
                           (-(1 - pl.col("fill_share")) * pl.col("R_mkt")).alias("diff_missed_part"),
                           # entry price improvement on the filled part, in bp of the price (positive = limit better)
                           pl.when(pl.col("fill_share") > 0).then(pl.when(pl.col("side") == "long").then(1.0).otherwise(-1.0) * (pl.col("vwap_mkt") - pl.col("vwap")) / pl.col("vwap_mkt") * 1e4).alias("improve_bp"))
    res: dict[str, Any] = {"n_signals": int(mk.height), "by_sleeve": {s: int(n) for s, n in t.filter(pl.col("arm") == "market").group_by("sleeve").len().iter_rows()}}
    seg_of = {r[0]: r[1] for r in t.filter(pl.col("arm") == "market").select("signal", "segment").iter_rows()}
    res["market"] = {}
    m_all = t.filter(pl.col("arm") == "market")
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        mm = m_all if seg == "pooled" else m_all.filter(pl.col("segment") == seg)
        sg = np.where(mm["side"].to_numpy() == "long", 1.0, -1.0)
        mid0 = mm["mid0"].to_numpy().astype(float)
        res["market"][seg] = {"R": boot(mm["net_R"].to_numpy(), mm["date"].to_numpy()), "n": mm.height,
                              "spread_bp": float(np.nanmean((mm["ask0"].to_numpy().astype(float) - mm["bid0"].to_numpy().astype(float)) / mid0 * 1e4)) if mm.height else float("nan"),
                              "slip_vs_mid_bp": float(np.nanmean(sg * (mm["vwap"].to_numpy().astype(float) - mid0) / mid0 * 1e4)) if mm.height else float("nan"),
                              "share_tp": float(mm["exit_kind"].is_in(["tp", "tp_gap"]).mean()) if mm.height else float("nan")}
    res["diff"] = {}
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        for q in ("pessimistic", "prorata", "optimistic"):
            for n in WINDOWS_MIN:
                for sl in ("pooled", "shock_regime", "trend_4h"):
                    x = lim.filter((pl.col("queue") == q) & (pl.col("window_min") == n))
                    if seg != "pooled":
                        x = x.filter(pl.col("segment") == seg)
                    if sl != "pooled":
                        x = x.filter(pl.col("sleeve") == sl)
                    key = f"{seg}|{q}|{n}|{sl}"
                    if x.height == 0:
                        res["diff"][key] = {"n": 0}
                        continue
                    filled = x.filter(pl.col("fill_share") > 0)
                    unf = x.filter(pl.col("fill_share") < 1)
                    res["diff"][key] = {
                        **boot(x["diff"].to_numpy(), x["date"].to_numpy(), seed=n),
                        "filled_part": float(x["diff_filled_part"].mean()), "missed_part": float(x["diff_missed_part"].mean()),
                        "R_limit": float(x["net_R"].mean()), "R_market": float(x["R_mkt"].mean()),
                        "fill_rate_any": float((x["fill_share"] > 0).mean()), "fill_share_mean": float(x["fill_share"].mean()),
                        "unfilled_share": float((x["fill_share"] == 0).mean()), "partial_share": float(((x["fill_share"] > 0) & (x["fill_share"] < 1)).mean()),
                        "improve_bp_filled": float(filled["improve_bp"].mean()) if filled.height else float("nan"),
                        "R_market_if_filled": float(filled["R_mkt"].mean()) if filled.height else float("nan"),
                        "R_market_if_unfilled": float(x.filter(pl.col("fill_share") == 0)["R_mkt"].mean()) if x.filter(pl.col("fill_share") == 0).height else float("nan"),
                        "missed": {"n": int(unf.height), "R_market_mean": float(unf["R_mkt"].mean()) if unf.height else float("nan"),
                                   "share_R_market_pos": float((unf["R_mkt"] > 0).mean()) if unf.height else float("nan"),
                                   "share_market_tp": float(unf["exit_mkt"].is_in(["tp", "tp_gap"]).mean()) if unf.height else float("nan"),
                                   "missed_R_total": float(((1 - unf["fill_share"]) * unf["R_mkt"]).sum()) if unf.height else 0.0},
                    }
    # verdicts per segment (pro-rata, pooled sleeves)
    res["verdict"] = {}
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        ds = [res["diff"][f"{seg}|prorata|{n}|pooled"] for n in WINDOWS_MIN]
        if any(d.get("n", 0) and d["mean"] > 0 and d["lo"] > 0 for d in ds):
            v = "supports"
        elif all(d.get("n", 0) and d["hi"] < 0 for d in ds):
            v = "contradicts"
        else:
            v = "inconclusive"
        res["verdict"][seg] = v
    res["seg_counts"] = {s: sum(1 for v in seg_of.values() if v == s) for s in ("llm_in_sample", "clean_holdout")}
    return res


def mmr3_table() -> pl.DataFrame:
    split = json.loads((MMR / "split.json").read_text(encoding="utf-8"))
    held = set(split["held_days"])
    rows = []
    for f in sorted((OUT / "mmr3" / "days").glob("*/*/stats.json")):
        sym, d = f.parent.parent.name, f.parent.name
        st = json.loads(f.read_text(encoding="utf-8"))
        for q in st["quoters"]:
            width, gate = q["name"].split("__")
            p, fu = q["pnl"], q["funnel"]
            rows.append({"symbol": sym, "group": MMR_GROUP_OF[sym], "day": d, "held": d in held, "name": q["name"], "width": width, "gate": gate,
                         "queue": q["queue"], "net": p["net"], "spread": p["spread"], "adverse_1s": p["adverse_1s"], "adverse_10s": p["adverse_10s"],
                         "adverse_60s": p["adverse_60s"], "inventory": p["inventory"], "fees": p["fees"], "fills": q["fills"],
                         "stopped": q["stopped"] is not None, "gate_off_share": fu.get("gate_off_ticks", 0) / max(fu["ticks"], 1),
                         "fills_gate": fu.get("fills_gate", 0), "gate_flattens": fu.get("gate_flattens", 0), "max_inv_usd": q["max_inventory_usd"]})
    return pl.DataFrame(rows)


def mmr3_analysis(t: pl.DataFrame) -> dict:
    res: dict[str, Any] = {"symbol_days": int(t.select("symbol", "day").unique().height), "diff": {}, "levels": {}}
    comps = ["net", "spread", "adverse_1s", "adverse_10s", "adverse_60s", "inventory", "fees", "fills"]
    for g in MMR_GROUPS:
        for held in (True, False):
            for q in ("pessimistic", "prorata", "optimistic"):
                x = t.filter((pl.col("group") == g) & (pl.col("held") == held) & (pl.col("queue") == q))
                for w in [PRIMARY_WIDTH[g]] + SECONDARY_WIDTHS:
                    base = x.filter((pl.col("width") == w) & (pl.col("gate") == "ungated")).select("symbol", "day", *[pl.col(c).alias(c + "_u") for c in comps])
                    for gate in sorted(set(x.filter(pl.col("width") == w)["gate"].to_list())):
                        y = x.filter((pl.col("width") == w) & (pl.col("gate") == gate))
                        lvl = {c: float(y[c].mean()) for c in comps} | {"stopped_share": float(y["stopped"].mean()), "gate_off_share": float(y["gate_off_share"].mean()), "n": y.height}
                        res["levels"][f"{g}|{'held' if held else 'fit'}|{q}|{w}|{gate}"] = {**lvl, "net_ci": boot(y["net"].to_numpy(), y["day"].to_numpy())}
                        if gate == "ungated":
                            continue
                        j = y.join(base, on=["symbol", "day"])
                        dd = {c: float((j[c] - j[c + "_u"]).mean()) for c in comps}
                        res["diff"][f"{g}|{'held' if held else 'fit'}|{q}|{w}|{gate}"] = {**boot((j["net"] - j["net_u"]).to_numpy(), j["day"].to_numpy()), "components": dd}
    res["verdict_by_group"] = {}
    for g in MMR_GROUPS:
        d = res["diff"].get(f"{g}|held|prorata|{PRIMARY_WIDTH[g]}|all", {"n": 0})
        res["verdict_by_group"][g] = d
    ds = list(res["verdict_by_group"].values())
    if any(d.get("n", 0) and d["mean"] > 0 and d["lo"] > 0 for d in ds):
        res["verdict"] = "supports"
    elif all(d.get("n", 0) and d["hi"] < 0 for d in ds):
        res["verdict"] = "contradicts"
    else:
        res["verdict"] = "inconclusive"
    return res


def mmr3_repro_check(t: pl.DataFrame) -> dict:
    """The ungated primary-width quoters must reproduce agent M's replay of the same variant exactly (same code path, no gate)."""
    out = {"n": 0, "max_abs_net_diff": 0.0, "fill_count_mismatch": 0, "days_differing": []}
    for (sym, d), y in t.filter(pl.col("gate") == "ungated").group_by("symbol", "day"):
        p = MMR / "days" / sym / d / "stats.json"
        if not p.is_file():
            continue
        st = json.loads(p.read_text(encoding="utf-8"))
        ref = {(q["name"], q["queue"]): q for q in st["quoters"]}
        for r in y.iter_rows(named=True):
            q = ref.get((r["width"], r["queue"]))
            if q is None:
                continue
            out["n"] += 1
            if abs(q["pnl"]["net"] - r["net"]) > 1e-9 and f"{sym}:{d}" not in out["days_differing"]:
                out["days_differing"].append(f"{sym}:{d}")
            out["max_abs_net_diff"] = max(out["max_abs_net_diff"], abs(q["pnl"]["net"] - r["net"]))
            out["fill_count_mismatch"] += int(q["fills"] != r["fills"])
    return out


REPORT_JSON = ROOT / "research" / "reports" / "execution_2026-11-15.json"
TABLES_MD = OUT / "tables.md"
FIG = ROOT / "research" / "figures" / "execution_2026-11-15"
COLORS = {"pessimistic": "#2a78d6", "prorata": "#eb6834", "optimistic": "#1baf7a"}


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _ci(d: dict, nd: int = 3) -> str:
    return f"{_f(d['mean'], nd)} [{_f(d['lo'], nd)}, {_f(d['hi'], nd)}]" if d.get("n") else "n/a"


def tables(e: dict, m: dict) -> str:
    L = ["## H-EXE1 tables", ""]
    L += ["| segment | signals | market net R [95% CI] | touch spread (bp) | market slippage vs mid (bp) | market reaches target |", "|---|---|---|---|---|---|"]
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        x = e["market"][seg]
        L.append(f"| {seg} | {x['n']} | {_ci(x['R'])} | {x['spread_bp']:.2f} | {x['slip_vs_mid_bp']:+.2f} | {x['share_tp']:.2f} |")
    L += ["", "Paired difference limit_N minus market, net R per signal (both sleeves pooled):", "",
          "| segment | queue | N (min) | n | diff [95% CI] | filled part | missed part | any fill | unfilled | partial | price gain if filled (bp) | limit R | market R |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        for q in ("prorata", "pessimistic", "optimistic"):
            for n in WINDOWS_MIN:
                d = e["diff"][f"{seg}|{q}|{n}|pooled"]
                if not d.get("n"):
                    continue
                L.append(f"| {seg} | {q} | {n} | {d['n']} | {_ci(d)} | {_f(d['filled_part'])} | {_f(d['missed_part'])} | {d['fill_rate_any']:.2f} | "
                         f"{d['unfilled_share']:.2f} | {d['partial_share']:.2f} | {_f(d['improve_bp_filled'], 2)} | {_f(d['R_limit'])} | {_f(d['R_market'])} |")
    L += ["", "By sleeve (pro-rata, pooled segments):", "", "| sleeve | N | n | diff [95% CI] | any fill | market R if filled | market R if unfilled |", "|---|---|---|---|---|---|---|"]
    for sl in ("shock_regime", "trend_4h"):
        for n in WINDOWS_MIN:
            d = e["diff"][f"pooled|prorata|{n}|{sl}"]
            if d.get("n"):
                L.append(f"| {sl} | {n} | {d['n']} | {_ci(d)} | {d['fill_rate_any']:.2f} | {_f(d['R_market_if_filled'])} | {_f(d['R_market_if_unfilled'])} |")
    L += ["", "Missed signals (not fully filled; pro-rata), their outcome under the market arm:", "",
          "| segment | N | missed (n) | share of signals | market R of the missed (mean) | missed with market R > 0 | missed reaching the target | missed R total |", "|---|---|---|---|---|---|---|---|"]
    for seg in ("llm_in_sample", "clean_holdout", "pooled"):
        for n in WINDOWS_MIN:
            d = e["diff"][f"{seg}|prorata|{n}|pooled"]
            if d.get("n"):
                mm = d["missed"]
                L.append(f"| {seg} | {n} | {mm['n']} | {mm['n'] / d['n']:.2f} | {_f(mm['R_market_mean'])} | {mm['share_R_market_pos']:.2f} | {mm['share_market_tp']:.2f} | {_f(mm['missed_R_total'], 2)} |")
    L += ["", "## H-MMR3 tables", "",
          "Gated minus ungated, paired per symbol-day (USD per symbol-day; held days unless marked):", "",
          "| group | days | queue | width | gate | diff net [95% CI] | d spread | d adverse 1 s | d adverse 60 s | d inventory | d fees | d fills |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in m["diff"].items():
        g, h, q, w, gate = key.split("|")
        if q != "prorata" and not (gate == "all" and w == PRIMARY_WIDTH[g]):
            continue
        c = d["components"]
        L.append(f"| {g} | {h} | {q} | {w} | {gate} | {_ci(d, 4)} | {_f(c['spread'], 4)} | {_f(c['adverse_1s'], 4)} | {_f(c['adverse_60s'], 4)} | {_f(c['inventory'], 4)} | "
                 f"{_f(-c['fees'], 4)} | {c['fills']:+.1f} |")
    L += ["", "Levels (held days, pro-rata; USD per symbol-day):", "",
          "| group | width | gate | net [95% CI] | spread | adverse 1 s | adverse 60 s | inventory | fees | fills | gate off share | symbol-stop share |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in m["levels"].items():
        g, h, q, w, gate = key.split("|")
        if h != "held" or q != "prorata":
            continue
        L.append(f"| {g} | {w} | {gate} | {_ci(d['net_ci'], 4)} | {_f(d['spread'], 4)} | {_f(d['adverse_1s'], 4)} | {_f(d['adverse_60s'], 4)} | {_f(d['inventory'], 4)} | "
                 f"{_f(-d['fees'], 4)} | {d['fills']:.1f} | {d['gate_off_share']:.2f} | {d['stopped_share']:.2f} |")
    return "\n".join(L) + "\n"


def figures(e: dict, m: dict, t_exe: pl.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#898781", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    out = []
    # 1) paired difference by N, per queue model, per segment
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, seg in zip(axs, ("llm_in_sample", "clean_holdout", "pooled"), strict=True):
        for k, q in enumerate(("pessimistic", "prorata", "optimistic")):
            xs, ys, lo, hi = [], [], [], []
            for j, n in enumerate(WINDOWS_MIN):
                d = e["diff"][f"{seg}|{q}|{n}|pooled"]
                if d.get("n"):
                    xs.append(j + (k - 1) * 0.18)
                    ys.append(d["mean"])
                    lo.append(d["mean"] - d["lo"])
                    hi.append(d["hi"] - d["mean"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=5, lw=2, capsize=0, color=COLORS[q], label=q)
        ax.axhline(0, color="#898781", lw=1)
        ax.set_xticks(range(len(WINDOWS_MIN)), [f"N = {n} min" for n in WINDOWS_MIN])
        ax.set_title(seg.replace("_", " "), color="#0b0b0b")
    axs[0].set_ylabel("limit minus market, net R per signal")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "exe1_diff_by_window.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.name)
    # 2) decomposition: filled part vs missed part (pro-rata, pooled)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fp = [e["diff"][f"pooled|prorata|{n}|pooled"]["filled_part"] for n in WINDOWS_MIN]
    mp = [e["diff"][f"pooled|prorata|{n}|pooled"]["missed_part"] for n in WINDOWS_MIN]
    x = np.arange(len(WINDOWS_MIN))
    ax.bar(x - 0.17, fp, 0.32, color="#2a78d6", label="filled part (price + fees)")
    ax.bar(x + 0.17, mp, 0.32, color="#eb6834", label="missed part (- market R not taken)")
    ax.axhline(0, color="#898781", lw=1)
    ax.set_xticks(x, [f"N = {n} min" for n in WINDOWS_MIN])
    ax.set_ylabel("net R per signal")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "exe1_decomposition.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.name)
    # 3) MMR3 decomposition gated vs ungated, held, pro-rata, primary width
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    comps = ["spread", "adverse_60s", "inventory", "fees", "net"]
    for ax, g in zip(axs, MMR_GROUPS, strict=True):
        w = PRIMARY_WIDTH[g]
        u = m["levels"][f"{g}|held|prorata|{w}|ungated"]
        a = m["levels"][f"{g}|held|prorata|{w}|all"]
        vals_u = [u[c] if c != "fees" else -u[c] for c in comps]
        vals_a = [a[c] if c != "fees" else -a[c] for c in comps]
        xx = np.arange(len(comps))
        ax.bar(xx - 0.17, vals_u, 0.32, color="#898781", label="ungated")
        ax.bar(xx + 0.17, vals_a, 0.32, color="#2a78d6", label="gated (all)")
        ax.axhline(0, color="#898781", lw=1)
        ax.set_xticks(xx, ["spread", "adverse 60 s", "inventory", "fees", "net"])
        ax.set_title(f"{g}: {w}, held days", color="#0b0b0b")
        ax.set_ylabel("USD per symbol-day")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "mmr3_decomposition.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.name)
    # 4) MMR3 by width: ungated vs gated net (held, pro-rata) with day-clustered CIs
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, g in zip(axs, MMR_GROUPS, strict=True):
        ws = [PRIMARY_WIDTH[g]] + SECONDARY_WIDTHS
        for k, (gate, col) in enumerate((("ungated", "#898781"), ("all", "#2a78d6"))):
            ys = [m["levels"][f"{g}|held|prorata|{w}|{gate}"]["net_ci"] for w in ws]
            xx = np.arange(len(ws)) + (k - 0.5) * 0.3
            ax.bar(xx, [y["mean"] for y in ys], 0.28, color=col, label="ungated" if gate == "ungated" else "gated (all)")
            ax.errorbar(xx, [y["mean"] for y in ys], yerr=[[y["mean"] - y["lo"] for y in ys], [y["hi"] - y["mean"] for y in ys]], fmt="none", ecolor="#0b0b0b", lw=1)
        ax.axhline(0, color="#898781", lw=1)
        ax.set_xticks(range(len(ws)), ws)
        ax.set_title(f"{g}: net per symbol-day, held days", color="#0b0b0b")
        ax.set_ylabel("USD per symbol-day")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "mmr3_by_width.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.name)
    return out


def report() -> dict:
    t_exe = exe1_table()
    t_exe.write_parquet(OUT / "exe1_arms.parquet")
    e = exe1_analysis(t_exe)
    t_m = mmr3_table()
    t_m.write_parquet(OUT / "mmr3_quoters.parquet")
    m = mmr3_analysis(t_m)
    m["repro_vs_agent_M"] = mmr3_repro_check(t_m)
    man = Manifest()
    data = {"days": man.d.get("days", {}), "storage": man.d.get("storage", {}),
            "downloaded_gb": sum(v.get("bytes", 0) for v in man.d.get("downloads", {}).values()) / GB,
            "raw_left": [k for k, v in man.d.get("downloads", {}).items() if not v.get("deleted")]}
    obj = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "design": json.loads(DESIGN.read_text(encoding="utf-8")), "H-EXE1": e, "H-MMR3": m,
           "data": {k: v for k, v in data.items() if k != "days"}, "n_symbol_days_done": sum(1 for v in data["days"].values() if v.get("status") == "done"),
           "failed": {k: v for k, v in data["days"].items() if v.get("status") != "done"}}
    obj["figures"] = figures(e, m, t_exe)
    write_json(REPORT_JSON, obj)
    TABLES_MD.write_text(tables(e, m), encoding="utf-8")
    return obj


def main(argv: list[str] | None = None) -> None:
    a = (argv if argv is not None else sys.argv[1:]) or ["run"]
    if a[0] == "signals":
        s = write_signals()["signals"]
        df = pl.DataFrame([{k: r[k] for k in ("sleeve", "symbol", "side", "date")} for r in s])
        print(df.group_by("sleeve", "symbol").len().sort("sleeve", "symbol"))
        print(df.with_columns(pl.col("date").str.slice(0, 7).alias("m")).group_by("m", "sleeve").len().sort("m", "sleeve"))
    elif a[0] == "gates":
        write_gates()
    elif a[0] == "design":
        d = write_design()
        print(json.dumps({"n_signals": d["H-EXE1"]["signals"]["n"], "n_symbol_days": d["H-EXE1"]["n_symbol_days"], "mmr3_variants": d["H-MMR3"]["variants"]}, indent=1))
    elif a[0] == "run":
        run_all(a[1] if len(a) > 1 else None)
    elif a[0] == "report":
        o = report()
        print(json.dumps({"exe1": o["H-EXE1"]["verdict"], "mmr3": o["H-MMR3"]["verdict"], "repro": o["H-MMR3"]["repro_vs_agent_M"]}, indent=1))
    else:
        raise SystemExit(f"unknown command {a[0]}")


if __name__ == "__main__":
    main()
