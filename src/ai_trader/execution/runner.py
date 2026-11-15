"""Strategy runner: strategy events -> bias gate -> OrderIntents -> `atx exec-intents` (agent L) -> RiskDecisions.

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/strategy_runner.py --mode replay --start 2026-10-13 --end 2026-11-12 \
        --strategies tsmom zscore_reversion shock_regime --symbols BTCUSDT ETHUSDT SOLUSDT
    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/strategy_runner.py --mode paper [--strategies ...] [--symbols ...]

Modes (nothing else is accepted; nothing here talks to an exchange):
  replay  every strategy event with decision time in [start 00:00, end 24:00) UTC, on frames read as of the end of the
          replay (capped at now). Point in time: every series is causal and the gate reads only rows with known_ts <= the
          event's decision time, so a replayed intent equals the one a live pass at that bar would have written (checked by
          tests/test_execution_gate.py). The exit that follows a gate-blocked entry is not written (no position exists);
          the count is in the run summary. Each --equity value gives one atx replay run (<run_id>-eq<E>).
  paper   one decision pass at now: only events on the last complete bar of each strategy's frame. A file flag
          data/exec/KILL suppresses new entries (exits still go out; L flattens on it).

Safety: refuses to run when the environment is mainnet, when execution.allow_mainnet is true, or when
AI_TRADER_ALLOW_MAINNET=1 is set (research / paper only, user instruction 2026-11-12).

Outputs (data/exec/):
  intents/<run_id>.jsonl      OrderIntents (contract docs/EXECUTION_PIPELINE.md)
  decisions/<run>.jsonl       IntentDecisions (the contract's RiskDecision) written by `atx exec-intents` (agent L)
  paper/<run>/                L's paper book (positions, fills, P&L)
  runs/<run_id>.json          run summary: gate statistics, strategy states, atx status
  journal/<UTC day>/*.jsonl   session_start / decision / risk_decision / session_end events (journal.Journal format)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_trader.core.config import load
from ai_trader.core.time import now

from .gate import VERSION, BiasGate, SymbolState, exec_dir
from .intents import build_intent, intents_path, read_jsonl, write_intents
from .signals import load_frames, strategy_events, strategy_refs

MODES = ("replay", "paper")
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MAX_WORKERS = 4
PAPER_BOOK = "paper"


class SafetyError(RuntimeError):
    pass


def safety_check(mode: str) -> None:
    if mode not in MODES:
        raise SafetyError(f"mode {mode!r} refused: only {MODES} (research / paper only)")
    cfg = load()
    allow = bool((cfg.settings.get("execution") or {}).get("allow_mainnet", False))
    if cfg.environment == "mainnet" or allow or os.environ.get("AI_TRADER_ALLOW_MAINNET") == "1":
        raise SafetyError(f"refused: environment={cfg.environment!r}, execution.allow_mainnet={allow}, "
                          f"AI_TRADER_ALLOW_MAINNET={os.environ.get('AI_TRADER_ALLOW_MAINNET')!r}; the strategy runner is research / paper only")


def make_run_id(mode: str, t: datetime, key: str) -> str:
    return f"{mode}-{t:%Y%m%dT%H%M%S}-{hashlib.sha256(key.encode()).hexdigest()[:6]}"


# ====================================================================== per-symbol work (process-pool friendly)

def symbol_pass(symbol: str, strategies: list[str] | None, as_of_iso: str, start_iso: str | None, end_iso: str | None, mode: str,
                jump: bool = True, last_bar_only: bool = False, kill: bool = False) -> dict:
    as_of = datetime.fromisoformat(as_of_iso)
    start = datetime.fromisoformat(start_iso) if start_iso else None
    end = datetime.fromisoformat(end_iso) if end_iso else None
    t0 = time.time()
    refs = strategy_refs(strategies)
    sf = load_frames(symbol, as_of, refs)
    out: dict[str, Any] = {"symbol": symbol, "intents": [], "states": [], "notes": list(sf.notes), "skipped_exits": 0}
    if sf.f1 is None or sf.f1.height < 500:
        out["notes"].append(f"{symbol}: not enough 1h history")
        return out
    st = SymbolState.build(sf)
    gate = BiasGate(jump=jump)
    for ref in refs:
        evs, info = strategy_events(ref, sf, start, end)
        if last_bar_only and "last_decision_ts" in info:
            evs = [e for e in evs if e.decision_ts == info["last_decision_ts"]]
            age_h = (as_of - info["last_bar_close"]).total_seconds() / 3600
            info["last_bar_age_h"] = round(age_h, 2)
            if age_h > (8 if ref.freq == "4h" else 2):
                info["stale"] = True
        out["states"].append({k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in info.items()})
        entry_gate = "unknown"
        for ev in evs:
            if kill and ev.kind == "entry":
                out["notes"].append(f"KILL flag: entry {ev.strategy} {ev.symbol} {ev.side} at {ev.decision_ts:%Y-%m-%d %H:%M} suppressed")
                continue
            if ev.kind == "exit" and entry_gate == "block" and not last_bar_only:
                out["skipped_exits"] += 1      # replay: no position exists behind a gate-blocked entry, so no exit order
                entry_gate = "unknown"
                continue
            g, plc, ctx = gate.evaluate(ev, st)
            extra = {"entry_gate": entry_gate} if ev.kind == "exit" else None
            d = build_intent(ev, g, plc, ctx, mode, rules_hash=gate.book.hash, extra_context=extra)
            entry_gate = g.decision if ev.kind == "entry" else "unknown"
            out["intents"].append(d)
    out["secs"] = round(time.time() - t0, 1)
    return out


# ====================================================================== atx exec-intents (agent L)

def atx_exe() -> Path | None:
    """The first atx binary that has the exec-intents subcommand (release, then debug)."""
    root = load().root
    name = "atx.exe" if sys.platform.startswith("win") else "atx"
    for d in ("release", "debug"):
        exe = root / "target" / d / name
        if not exe.is_file():
            continue
        try:
            h = subprocess.run([str(exe), "exec-intents", "--help"], capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if h.returncode == 0 and "--file" in h.stdout:
            return exe
    return None


def atx_exec_intents(intents: Path, run: str, mode: str, equity: float | None = None, timeout: int = 1800) -> dict:
    """`atx exec-intents --file <intents> --mode <mode> --run <run> [--equity E]` (agent L, docs/EXECUTION_PIPELINE.md
    Changes 2026-11-13). Decisions land in data/exec/decisions/<run>.jsonl, the paper book in data/exec/paper/<run>/."""
    exe = atx_exe()
    if exe is None:
        return {"status": "skipped", "reason": "no atx binary with `exec-intents` yet (agent L): intents written only"}
    args = [str(exe), "exec-intents", "--file", str(intents), "--mode", mode, "--run", run]
    if equity is not None:
        args += ["--equity", f"{equity:g}"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "args": args, "reason": str(e)}
    return {"status": "ok" if r.returncode == 0 else "error", "returncode": r.returncode, "args": args, "run": run, "equity": equity,
            "decisions_file": str(exec_dir() / "decisions" / f"{run}.jsonl"), "stdout_tail": r.stdout[-3000:], "stderr_tail": r.stderr[-2000:]}


# ====================================================================== statistics

def gate_stats(intents: list[dict]) -> dict:
    ent = [d for d in intents if d["side"] != "flat"]
    by_strat: dict[str, Counter] = defaultdict(Counter)
    by_rule: Counter = Counter()
    by_rule_strat: dict[str, Counter] = defaultdict(Counter)
    warns: Counter = Counter()
    psrc: Counter = Counter()
    er_pass = []
    for d in ent:
        g = d["gate"]
        by_strat[d["strategy"]][g["decision"]] += 1
        for rid in g.get("blocked_by", []):
            by_rule[rid] += 1
            by_rule_strat[rid][d["strategy"]] += 1
        for r in g["reasons"]:
            if r.startswith("warn "):
                warns[r.split()[1]] += 1
        psrc[str(d["context"].get("p_source", "")).split(" ")[0]] += 1
        if g["decision"] == "pass" and d.get("net_er_r") is not None:
            er_pass.append(d["net_er_r"])
    min_er = float(load().risk["min_net_expected_r"])
    return {
        "n_intents": len(intents), "n_entries": len(ent), "n_exits": len(intents) - len(ent),
        "entries_pass": sum(c["pass"] for c in by_strat.values()), "entries_block": sum(c["block"] for c in by_strat.values()),
        "by_strategy": {k: dict(v) for k, v in sorted(by_strat.items())},
        "blocked_by_rule": dict(by_rule.most_common()),
        "blocked_by_rule_strategy": {k: dict(v) for k, v in by_rule_strat.items()},
        "warnings": dict(warns),
        "p_source": dict(psrc),
        "passed_net_er_r": {"n": len(er_pass), "mean": (sum(er_pass) / len(er_pass)) if er_pass else None,
                            "n_at_or_above_min": sum(1 for x in er_pass if x >= min_er), "min_net_expected_r": min_er},
    }


def decision_stats(decisions: list[dict]) -> dict:
    c = Counter(d.get("decision") for d in decisions)
    rej = Counter(r for d in decisions for r in (d.get("rejections") or []))
    return {"n": len(decisions), "decisions": dict(c), "rejections": dict(rej.most_common())}


# ====================================================================== runs

def _journal(run_id: str):
    from ai_trader.journal import Journal

    return Journal(run_id, root=exec_dir() / "journal")


def _finish(run_id: str, mode: str, meta: dict, results: list[dict], call_atx: bool, equities: list[float] | None = None) -> dict:
    intents = [d for r in results for d in r["intents"]]
    intents.sort(key=lambda d: (d["ts_ms"], d["symbol"], d["strategy"], d["side"] != "flat"))
    p = intents_path(run_id)
    if p.exists():
        p.unlink()
    write_intents(run_id, intents, p)
    stats = gate_stats(intents)
    j = _journal(run_id)
    j.write("decision", {"mode": mode, "intents_file": str(p), "gate_stats": stats,
                         "intents": [{"intent_id": d["intent_id"], "strategy": d["strategy"], "symbol": d["symbol"], "side": d["side"],
                                      "as_of": d["as_of"], "gate": d["gate"]["decision"], "blocked_by": d["gate"]["blocked_by"]} for d in intents]})
    atx_runs = []
    if call_atx:
        eqs = equities or [None]
        for eq in eqs:
            if mode == "paper":
                run = PAPER_BOOK          # one persistent paper book across passes (L's default run name)
            else:
                run = run_id if len(eqs) == 1 else f"{run_id}-eq{eq:g}"
            a = atx_exec_intents(p, run, mode, eq)
            ids = {d["intent_id"] for d in intents}
            decs = [x for x in read_jsonl(Path(a["decisions_file"])) if x.get("intent_id") in ids] if a.get("decisions_file") else []
            a["summary"] = decision_stats(decs) if decs else None
            if decs:
                j.write("risk_decision", {"run": run, "equity": eq, "decisions_file": a["decisions_file"], "summary": a["summary"]})
            atx_runs.append(a)
    summary = {"run_id": run_id, "mode": mode, **meta, "intents_file": str(p), "gate_stats": stats,
               "skipped_exits_after_gate_block": sum(r.get("skipped_exits", 0) for r in results),
               "atx": atx_runs or [{"status": "not_called"}],
               "states": [s for r in results for s in r["states"]], "notes": [n for r in results for n in r["notes"]],
               "secs": {r["symbol"]: r.get("secs") for r in results}}
    rp = exec_dir() / "runs" / f"{run_id}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    j.write("session_end", {"run_file": str(rp), "atx": [a.get("status") for a in summary["atx"]], "n_intents": len(intents)})
    summary["run_file"] = str(rp)
    return summary


def _map(fn, jobs: list[tuple], workers: int) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(*a) for a in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS, len(jobs))) as ex:
        return list(ex.map(fn, *zip(*jobs, strict=True)))


def run_replay(start: date, end: date, strategies: list[str] | None = None, symbols: list[str] | None = None, *, workers: int = 3,
               jump: bool = True, call_atx: bool = True, equities: list[float] | None = None) -> dict:
    safety_check("replay")
    symbols = list(symbols or DEFAULT_SYMBOLS)
    names = [r.name for r in strategy_refs(strategies)]
    t = now()
    s0 = datetime(start.year, start.month, start.day, tzinfo=UTC)
    e0 = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1) - timedelta(microseconds=1)
    as_of = min(e0, t)
    run_id = make_run_id("replay", t, f"{start}|{end}|{','.join(names)}|{','.join(symbols)}")
    meta = {"start": s0.isoformat(), "end": e0.isoformat(), "as_of": as_of.isoformat(), "strategies": names, "symbols": symbols,
            "gate_version": VERSION, "jump": jump}
    _journal(run_id).write("session_start", {**meta, "rules_hash": BiasGate(jump=False).book.hash, "runner": "execution.runner"})
    jobs = [(s, names, as_of.isoformat(), s0.isoformat(), e0.isoformat(), "replay", jump, False, False) for s in symbols]
    return _finish(run_id, "replay", meta, _map(symbol_pass, jobs, workers), call_atx, equities)


def run_paper(strategies: list[str] | None = None, symbols: list[str] | None = None, *, workers: int = 3, jump: bool = True,
              call_atx: bool = True, equities: list[float] | None = None) -> dict:
    safety_check("paper")
    symbols = list(symbols or DEFAULT_SYMBOLS)
    names = [r.name for r in strategy_refs(strategies)]
    t = now()
    kill = (exec_dir() / "KILL").exists()
    run_id = make_run_id("paper", t, f"{','.join(names)}|{','.join(symbols)}")
    meta = {"as_of": t.isoformat(), "strategies": names, "symbols": symbols, "gate_version": VERSION, "jump": jump, "kill_flag": kill}
    _journal(run_id).write("session_start", {**meta, "rules_hash": BiasGate(jump=False).book.hash, "runner": "execution.runner"})
    jobs = [(s, names, t.isoformat(), None, None, "paper", jump, True, kill) for s in symbols]
    # paper: one persistent book under L's run name; a single equity value only seeds a NEW book
    return _finish(run_id, "paper", meta, _map(symbol_pass, jobs, workers), call_atx, (equities or [None])[:1])


def print_summary(s: dict) -> None:
    g = s["gate_stats"]
    print(f"run {s['run_id']} ({s['mode']}): {g['n_intents']} intents = {g['n_entries']} entries ({g['entries_pass']} pass, "
          f"{g['entries_block']} block) + {g['n_exits']} exits")
    for k, v in g["by_strategy"].items():
        print(f"  {k:22s} {v}")
    if g["blocked_by_rule"]:
        print("  blocked by rule:")
        for rid, n in g["blocked_by_rule"].items():
            print(f"    {rid:10s} {n:4d}  {g['blocked_by_rule_strategy'].get(rid, {})}")
    if g["warnings"]:
        print(f"  warnings: {g['warnings']}")
    e = g["passed_net_er_r"]
    if e["n"]:
        print(f"  passed entries: mean net_er_r {e['mean']:+.3f}; {e['n_at_or_above_min']}/{e['n']} >= min_net_expected_r {e['min_net_expected_r']}")
    print(f"  p_tp source: {g['p_source']}")
    if s.get("skipped_exits_after_gate_block"):
        print(f"  exits not sent (their entry was gate-blocked): {s['skipped_exits_after_gate_block']}")
    for a in s["atx"]:
        print(f"  atx exec-intents [{a.get('run', '')} equity {a.get('equity')}]: {a.get('status')} {a.get('reason', '')}")
        if a.get("summary"):
            print(f"    decisions: {a['summary']['decisions']}  rejections: {a['summary']['rejections']}")
        elif a.get("status") == "error":
            print(f"    {a.get('stderr_tail', '')[-500:]}")
    for n in s["notes"][:20]:
        print(f"  note: {n}")
    print(f"  intents: {s['intents_file']}\n  run file: {s['run_file']}")
