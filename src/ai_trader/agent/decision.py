"""Phase-4 decision harness: Claude decides take / skip on a paper book every 4 hours (CLI: scripts/decision_session.py).

    prepare  [--as-of ISO]   point-in-time session under data/sessions/<id>/: advances the persistent paper book over
                             the bars completed by as_of, runs the weekly stop-cap update and the H-WS16-15 monitor,
                             sends the trend sleeves' rule exits (s x side <= 0 at a 4h close; deterministic, never
                             Claude's choice), builds candidates from the sleeve config (agent.trend_sleeves via
                             agent.session: trend_zoo_ew = WS16 spec A, trend_tsmom30 = spec B; user decisions D4 / D5
                             2026-11-27), applies the bias gate (execution.gate rules) and writes PACKET.md (<= ~120
                             lines) for Claude.
    decide   --session ID --file decisions.json
                             validates one answer per candidate (take | skip, reason, optional bounded tilt) and optional
                             early closes, refuses after the deadline (as_of + 30 min) and any attempt to set size, stops,
                             targets or limits, then sends the takes as OrderIntents through `atx exec-intents` (the Rust
                             risk engine sizes them, fixed risk per trade) into the paper book. Writes submitted.json in the
                             format agent.scorecard reads.
    status   [--record]      book, scorecard (AI vs RULE vs RANDOM vs ALL; H-AI1 / H-AI2 running stats), stop posterior.

Paper only. Nothing here talks to an exchange: the book is at-exec's PaperBook driven by `atx exec-intents --mode paper`
(which refuses live mode and a mainnet environment). This module refuses to run when its own process is set up for
mainnet: the environment is mainnet or AI_TRADER_ALLOW_MAINNET is set (execution.runner.safety_check). The global
execution.allow_mainnet flag alone does not stop it (it belongs to the live desk, `atx live`, since 2026-11-15).

Persistent book: data/exec/paper_book/ (atx --out-dir): paper/<run>/{state,positions,summary}.json and
{fills,trades,cancels,equity}.jsonl, decisions/<run>.jsonl, intents/<session>.jsonl, ids.jsonl (intent -> candidate).
Run `ai` is the real AI book; `pipeline_test` is a separate book for pipeline tests (scorecard excludes those sessions).
Exits: the resting exchange stop (conservative intrabar rule of at-exec: stop first when both are touched), the trend
sleeves' rule exit (a flat intent sent by prepare, filled at the next 1h open as taker), the legacy bracket's target
(trend_4h positions opened before D4), and Claude's early-close requests; the kill switch data/exec/KILL flattens the
book and blocks new entries.

Candidates: only the sleeves of agent.trend_sleeves (SLEEVES); shock_regime and setup C are retired (D5), trend_4h is
legacy (D4: no new candidates, open positions keep their bracket). The strategy_runner paper signals are NOT included:
the non-sleeve strategies carry the driftless null (net E[R] <= 0, R-MAP-1 / R-PX-3: no validated edge, the engine
rejects them anyway), and its zoo_ensemble events are the same bet as the trend sleeves with a vol-only placement, so
adding them would double count one bet per symbol (R-PD-1 / R-PD-2).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_trader.core.config import load
from ai_trader.core.time import LookAheadError, now, utc
from ai_trader.journal import Journal

HARNESS = "decision_session.v2"       # v2 (2026-11-27, D4 / D5): trend-sleeve rule exits sent by prepare
PACKET_VERSION = "compact-3"          # H-IC4 compact arm; changing the packet layout starts a new track record
# compact-1 -> compact-2 (2026-11-15): header shows the entry mode; entries pegged to the touch (exec pegged-limit-v1)
# compact-2 -> compact-3 (2026-11-27): WS16 trend sleeves (no target, signal exit), H-WS16-15 monitor, shock lines removed
EXEC_VERSIONS = {"pegged_limit": "pegged-limit-v1", "as_signal": "as-signal-v1"}
SLOT_HOURS = (0, 4, 8, 12, 16, 20)
DEADLINE = timedelta(minutes=30)
LOOKBACK_BARS = 4                     # a shock signal that started since the previous session is new
BOOK_RUN, TEST_RUN = "ai", "pipeline_test"
BOOK_EQUITY = 10_000.0                # seed of a NEW paper book only (USDT); BTC lot minimums need more than 100
MAX_PACKET_RULES = 8                  # R-BUDGET-1: short cards
NS = uuid.UUID("2f0c1d7e-9a4b-4b8e-8d0e-6c3a5e7f9b11")
# sleeve -> contract family (execution.intents.FAMILIES; the engine's max_positions_per_cluster counts per family) and
# -> the strategy name the bias gate's rules use. trend_4h: legacy bracket sleeve (no new candidates since D4).
SLEEVE_FAMILY = {"trend_zoo_ew": "trend", "trend_tsmom30": "trend", "trend_4h": "trend"}
GATE_STRATEGY = {"trend_zoo_ew": "zoo_ensemble", "trend_tsmom30": "zoo_ensemble", "trend_4h": "zoo_ensemble"}   # the 4h trend sleeves in the gate's rules
TOP_KEYS = {"session_id", "model_id", "prompt_version", "decisions", "close", "market_view", "notes"}
DECISION_KEYS = {"candidate_id", "action", "reason", "tilt"}
TILT_KEYS = {"view_logodds", "confidence", "reason_codes", "evidence"}
CLOSE_KEYS = {"position_id", "symbol", "reason"}
# anything Claude might use to steer size, placement or limits: refused wherever it appears
CONTROL_WORDS = ("size", "qty", "quantity", "notional", "leverage", "risk", "contracts", "lots", "stop", "target", "take_profit", "tp", "sl",
                 "entry", "price", "cap", "limit", "sigma", "max", "min", "equity", "weight", "p_tp", "net_er")


class RefusedError(RuntimeError):
    """A refused prepare / decide (deadline, validation, safety). Nothing was sent to the book."""


# ====================================================================== paths and safety

@dataclass
class Paths:
    sessions: Path
    book: Path
    kill: Path
    live_nn: Path
    stop_cap: Path
    journal: Path | None = None       # None: paths.journal from settings (data/journal)
    edge: Path | None = None          # None: data/models/risk/edge_posterior.json next to stop_cap
    trend: Path | None = None         # None: data/models/risk/trend_sleeves/ next to stop_cap (peg checks, kill latch)

    def __post_init__(self) -> None:
        if self.edge is None:
            self.edge = self.stop_cap.parent / "edge_posterior.json"
        if self.trend is None:
            self.trend = self.stop_cap.parent / "trend_sleeves"

    @classmethod
    def default(cls) -> Paths:
        r = load().root
        return cls(r / "data" / "sessions", r / "data" / "exec" / "paper_book", r / "data" / "exec" / "KILL", r / "data" / "journal" / "live_nn",
                   r / "data" / "models" / "risk" / "stop_cap.json")


def safety_check() -> None:
    from ai_trader.execution.runner import SafetyError
    from ai_trader.execution.runner import safety_check as _sc

    try:
        _sc("paper")
    except SafetyError as e:
        raise RefusedError(str(e)) from e


def entry_mode() -> dict:
    """config/risk.yaml paper_entry (H-EXE1): how the paper book's entries are sent and filled."""
    pe = dict(load().risk.get("paper_entry") or {"mode": "as_signal"})
    if pe["mode"] not in EXEC_VERSIONS:
        raise RefusedError(f"paper_entry.mode {pe['mode']!r} not in {sorted(EXEC_VERSIONS)}")
    return {**pe, "exec_version": EXEC_VERSIONS[pe["mode"]]}


def pinned() -> dict:
    llm = load().settings["llm"]
    em = entry_mode()
    return {"model_id": llm["model"], "prompt_version": llm["prompt_version"], "packet_version": PACKET_VERSION, "harness": HARNESS,
            "exec_version": em["exec_version"], "entry_mode": em["mode"]}


def slot_of(t: datetime) -> datetime:
    t = utc(t)
    return t.replace(hour=max(h for h in SLOT_HOURS if h <= t.hour), minute=0, second=0, microsecond=0)


def iso(t: datetime) -> str:
    return utc(t).isoformat().replace("+00:00", "Z")


def _jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


# ====================================================================== the paper book (atx exec-intents --mode paper)

ATX_DIRS = ("target/debug", "target/agent-d/debug", "target/release")   # never the running bin/ copies


def atx_exe() -> Path | None:
    """The newest atx build with `exec-intents --as-of`: $AI_TRADER_ATX_EXE if set, else the most recently built of
    ATX_DIRS (target/agent-d is a side build used while another agent's job keeps target/debug/atx.exe locked)."""
    root = load().root
    name = "atx.exe" if sys.platform.startswith("win") else "atx"
    env = os.environ.get("AI_TRADER_ATX_EXE")
    cands = [Path(env)] if env else sorted((root / d / name for d in ATX_DIRS if (root / d / name).is_file()), key=lambda x: -x.stat().st_mtime)
    for exe in cands:
        try:
            h = subprocess.run([str(exe), "exec-intents", "--help"], capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if h.returncode == 0 and "--as-of" in h.stdout:
            return exe
    return None


def paper_gate() -> tuple[float, bool]:
    """(min net E[R], exclusive) for the paper book: config/risk.yaml min_net_expected_r_paper (exclusive, user decision
    2026-11-14), as crates/at-risk DecideConfig::for_mode applies it in paper mode; the inclusive live value if absent."""
    r = load().risk
    if r.get("min_net_expected_r_paper") is not None:
        return float(r["min_net_expected_r_paper"]), True
    return float(r["min_net_expected_r"]), False


class Book:
    """The persistent paper book of one run (at-exec PaperBook through the Rust risk engine). Paper mode only."""

    def __init__(self, paths: Paths, run: str = BOOK_RUN, *, bars_dir: Path | None = None, journal: bool = True, exe: Path | None = None):
        self.paths, self.run, self.bars_dir, self.journal = paths, run, bars_dir, journal
        self.exe = exe or atx_exe()

    @property
    def dir(self) -> Path:
        return self.paths.book / "paper" / self.run

    def _call(self, intents: Path, as_of: datetime) -> dict:
        if self.exe is None:
            raise RefusedError("no atx binary with `exec-intents --as-of` (cargo build -p atx)")
        args = [str(self.exe), "exec-intents", "--file", str(intents), "--mode", "paper", "--run", self.run, "--out-dir", str(self.paths.book),
                "--kill-file", str(self.paths.kill), "--as-of", iso(as_of), "--equity", f"{BOOK_EQUITY:g}"]
        if self.bars_dir is not None:
            args += ["--bars-dir", str(self.bars_dir)]
        if not self.journal:
            args.append("--no-journal")
        r = subprocess.run(args, capture_output=True, text=True, timeout=1800, check=False, cwd=str(load().root))
        if r.returncode != 0:
            raise RefusedError(f"atx exec-intents failed ({r.returncode}): {r.stderr[-1500:] or r.stdout[-1500:]}")
        summ = json.loads((self.dir / "summary.json").read_text(encoding="utf-8"))
        return {"stdout": r.stdout[-2000:], "summary": summ}

    def advance(self, as_of: datetime) -> dict:
        """Advance over the bars completed by as_of (no intents). Returns the new fills / trades / cancels."""
        before = {k: len(_jsonl(self.dir / f"{k}.jsonl")) for k in ("fills", "trades", "cancels")}
        f = self.paths.book / "intents" / "_advance.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("", encoding="utf-8")
        out = self._call(f, as_of)
        for k, n in before.items():
            out[k] = _jsonl(self.dir / f"{k}.jsonl")[n:]
        return out

    def submit(self, intents: Path, as_of: datetime, ids: set[str]) -> dict:
        out = self._call(intents, as_of)
        out["decisions"] = [d for d in _jsonl(self.paths.book / "decisions" / f"{self.run}.jsonl") if d.get("intent_id") in ids]
        return out

    def positions(self) -> dict:
        p = self.dir / "positions.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"positions": [], "pending_orders": []}

    def trades(self) -> list[dict]:
        return _jsonl(self.dir / "trades.jsonl")

    def summary(self) -> dict:
        p = self.dir / "summary.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def ids(self) -> dict[str, dict]:
        return {d["intent_id"]: d for d in _jsonl(self.paths.book / "ids.jsonl")}


# ====================================================================== packet pieces

def live_line(as_of: datetime, root: Path) -> str:
    """The BTC live-network line in force at as_of (the newest data/journal/live_nn line with as_of <= session as_of)."""
    as_of = utc(as_of)
    best = None
    if root.is_dir():
        for f in sorted(root.glob("20*.jsonl")):
            if f.stem > as_of.date().isoformat():
                continue
            for d in _jsonl(f):
                try:
                    t = utc(d["as_of"])
                except (KeyError, ValueError):
                    continue
                if t <= as_of and (best is None or t > utc(best["as_of"])):
                    best = d
    if best is None:
        return "BTC live network: no line known at as_of"
    n = best.get("net", {})

    def h(k: str) -> str:
        x = n.get(k) or {}
        q = x.get("quantiles_pct", {})
        a = (x.get("aci") or {}).get("80") or {}
        return (f"{k} med {q.get('q50', float('nan')):+.2f}% ACI80 [{a.get('lo_pct', float('nan')):+.2f}, {a.get('hi_pct', float('nan')):+.2f}] "
                f"sigma {x.get('sigma_pct', float('nan')):.2f}%")
    age = (as_of - utc(best["as_of"])).total_seconds() / 60
    al = best.get("alarms") or []
    return (f"BTC live network {best.get('id')} (as_of {utc(best['as_of']):%H:%M}Z, {age:.0f} min old, close {best.get('close'):,}): {h('1h')}; {h('24h')}; "
            f"flags {','.join(best.get('flags') or []) or '-'}; alarms {','.join(map(str, al)) or 'none'}. Research only, no direction call (R-FLD-3).")


def compact_card(st: dict, gate_ctx: dict | None) -> list[str]:
    """<= 8 tested lines per symbol (H-IC4 compact arm). Untested context (flow, liquidations) stays in cards.txt only."""
    dev = (gate_ctx or {}).get("dev_1h")
    nan = float("nan")
    sc = st.get("signal_close") or ""
    lines = [
        f"- {st['symbol']} {st['price']:,.6g} | bar {st.get('bar_ts', '')[:16]} | cell {st['cell']} {st.get('label', '')}"
        + (" | TRADFI WEEKEND (R-PX-9)" if st.get("weekend") else ""),
        f"- vol: risk sigma/4h {st.get('sigma_risk_4h', nan):.2%} (trend stops, R-VOL-2) | risk sigma/h {st.get('sigma_risk_h', nan):.2%} | central sigma/h "
        f"{st.get('sigma_rb_h', nan):.2%} (weekend m {st.get('wk_m', nan):.2f}, R-VOL-1) | HAR day {st.get('har_day', nan):.2%} (R-VOL-20)",
        f"- signals at the {sc[11:16] or '?'}Z 4h close: zoo_ew (spec A) {st.get('zoo_ew', nan):+.2f} | tsmom_30d (spec B) {st.get('tsmom_30d', nan):+.2f} "
        f"(entry |s| >= 0.30; WS16: fragile edge, IC ~0.03)"
        + (f" | 1h dev from EMA48 {dev:+.2f} sigma (R-PX-2: never fade >= 2)" if dev is not None else ""),
    ]
    return lines


def gate_candidates(cands: list, states: dict[str, dict], as_of: datetime) -> tuple[dict[str, dict], dict[str, dict]]:
    """Bias-gate rule check (execution.gate.BiasGate.check, the machine-checkable playbook rules) for every candidate,
    point in time (frames read as of as_of). The sleeve's structural placement is kept; only the rules are applied."""
    from ai_trader.execution.gate import BiasGate, SymbolState
    from ai_trader.execution.signals import SignalEvent, load_frames, strategy_refs

    gate = BiasGate(jump=False)
    out, ctxs = {}, {}
    by_sym: dict[str, list] = {}
    for c in cands:
        by_sym.setdefault(c.symbol, []).append(c)
    for sym, cs in by_sym.items():
        sf = load_frames(sym, as_of, strategy_refs(sorted({GATE_STRATEGY[c.sleeve] for c in cs})))
        st = SymbolState.build(sf)
        i = st.row_at(as_of)
        for c in cs:
            strat = GATE_STRATEGY[c.sleeve]
            bar_close = st.close_ts[i] if i >= 0 else as_of
            sig = c.sleeve_signal if c.sleeve_signal is not None else (1.0 if c.side == "long" else -1.0)
            sig = max(-1.0, min(1.0, float(sig))) or (1.0 if c.side == "long" else -1.0)
            ev = SignalEvent(strat, SLEEVE_FAMILY[c.sleeve], SLEEVE_FAMILY[c.sleeve], c.freq, sym, sf.asset_class, "entry", c.side, "flat", sig, sig,
                             bar_close - timedelta(hours=1), bar_close, as_of, c.entry, i)
            ctx = gate.context(ev, st)
            g = gate.check(ev, ctx, st) if ctx["i"] >= 0 else None
            ctxs[sym] = ctx
            out[c.id] = ({"decision": g.decision, "rules_applied": g.rules_applied, "reasons": g.reasons, "blocked_by": g.blocked_by, "warnings": g.warnings,
                          "asset_class": sf.asset_class, "signal": sig, "gate_strategy": strat, "rules_hash": gate.book.hash}
                         if g else {"decision": "block", "rules_applied": [], "reasons": ["no 1h bar known at as_of"], "blocked_by": ["no_data"], "warnings": [],
                                    "asset_class": sf.asset_class, "signal": sig, "gate_strategy": strat, "rules_hash": gate.book.hash})
    return out, ctxs


def packet_rules(states: dict[str, dict], fired: list[str], k: int = MAX_PACKET_RULES) -> list[str]:
    """The rules that fired on a candidate first, then the playbook's strategy_choice rules for the symbols' states that
    the gate does not machine-check, rendered by research.rules.for_prompt."""
    from ai_trader.execution.gate import MACHINE_RULES
    from ai_trader.research.playbook import select
    from ai_trader.research.rules import for_prompt, load_rules

    rules = load_rules()
    by = {r["id"]: r for r in rules}
    ids = [x for x in dict.fromkeys(fired) if x in by]
    for st in states.values():
        stt = {"asset_class": "crypto", "weekday": st.get("weekday"), "cell": st.get("cell")}
        s = select("strategy_choice", {a: b for a, b in stt.items() if b is not None}, rules)
        ids += [r["id"] for r in s["rules"] if r["id"] not in MACHINE_RULES and r["id"] not in ids]
    keep = [by[x] for x in ids[:k]]
    return [ln if len(ln) <= 230 else ln[:227] + "..." for ln in for_prompt(keep).splitlines()]


def _fmt(x: float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "-"
    return format(x, ",.6g")


# ====================================================================== prepare

def rule_exit_intents(sid: str, as_of: datetime, exits: list[dict], pipeline_test: bool, model_id: str) -> list[dict]:
    """Flat OrderIntents for the trend sleeves' rule exits (agent.trend_sleeves.exits_due): deterministic, not Claude's."""
    from ai_trader.execution.intents import validate_intent

    as_of = utc(as_of)
    out = []
    for x in exits:
        why = f"sleeve rule exit: s {x['s']:+.2f} x side <= 0 at the {x['close_ts'][11:16]}Z 4h close" + (" (late: that session did not run)" if x["late"] else "")
        it = {"intent_id": str(uuid.uuid5(NS, f"rule_exit|{sid}|{x['position_id']}")), "ts_ms": int(as_of.timestamp() * 1000), "as_of": as_of.isoformat(),
              "mode": "paper", "strategy": x["strategy"], "family": x.get("family") or "trend", "symbol": x["symbol"], "asset_class": "crypto", "side": "flat",
              "signal": 0.0, "entry": {"type": "market", "price": float(x["mark"])}, "stop": None, "targets": [],
              "sigma": {"per_bar": None, "daily_logHAR": None, "weekday_factor": None, "source": "vol.har"}, "p_tp": None, "net_er_r": None,
              "gate": {"decision": "pass", "rules_applied": ["R-EXIT-1"], "reasons": [why], "blocked_by": [], "stop_sigma": None},
              "context": {"session_id": sid, "position_id": x["position_id"], "harness": HARNESS, "decision": "rule_exit", "reason": why, "signal_close": x["close_ts"],
                          "s": x["s"], "late": x["late"], "pipeline_test": pipeline_test, "model_id": model_id}}
        validate_intent(it)
        out.append(it)
    return out


def send_rule_exits(sid: str, as_of: datetime, exits: list[dict], paths: Paths, book: Book, pipeline_test: bool, model_id: str, j: Journal) -> dict:
    """Send the rule exits to the paper book at as_of (the position closes at the next 1h open, taker + slippage)."""
    intents = rule_exit_intents(sid, as_of, exits, pipeline_test, model_id)
    ip = paths.book / "intents" / f"{sid}-rule-exits.jsonl"
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in intents), encoding="utf-8")
    with (paths.book / "ids.jsonl").open("a", encoding="utf-8") as f:
        for x in intents:
            f.write(json.dumps({"intent_id": x["intent_id"], "session": sid, "candidate_id": None, "position_id": x["context"]["position_id"],
                                "kind": "rule_exit", "book_run": book.run}) + "\n")
    res = book.submit(ip, as_of, {x["intent_id"] for x in intents})
    by = {x["intent_id"]: x for x in intents}
    sent = [{"symbol": by[d["intent_id"]]["symbol"], "strategy": by[d["intent_id"]]["strategy"], "position_id": by[d["intent_id"]]["context"]["position_id"],
             "decision": d["decision"], "rejections": d.get("rejections", []), "reason": by[d["intent_id"]]["context"]["reason"]} for d in res["decisions"]]
    j.write("order", {"status": "paper", "kind": "rule_exit", "book_run": book.run, "intents": intents})
    j.write("risk_decision", {"book_run": book.run, "kind": "rule_exit", "decisions": res["decisions"]})
    return {"sent": sent, "error": None, "intents_file": str(ip)}


def prepare(as_of: datetime | None = None, *, paths: Paths | None = None, symbols: list[str] | None = None, pipeline_test: bool = False,
            analyse_fn: Callable | None = None, gate_fn: Callable | None = None, book: Book | None = None, wall: datetime | None = None,
            signals_fn: Callable | None = None, monitor_fn: Callable | None = None) -> Path:
    """`signals_fn(sym, t)` -> trend_sleeves.SymbolSignals (default trend_sleeves.symbol_signals); `monitor_fn(t)` -> the
    H-WS16-15 monitor document (default trend_sleeves.monitor over the unconstrained stream; real sessions write)."""
    from ai_trader.agent import trend_sleeves as ts
    from ai_trader.agent.session import CLUSTER, DEFAULT, analyse, net_er
    from ai_trader.risk import edge_posterior as ep
    from ai_trader.risk import stop_posterior as sp

    safety_check()
    paths = paths or Paths.default()
    wall = utc(wall or now())
    as_of = utc(as_of or wall).replace(microsecond=0)
    if as_of > wall + timedelta(seconds=5):
        raise RefusedError(f"as_of {iso(as_of)} lies in the future")
    if not pipeline_test and wall - as_of > DEADLINE:
        raise RefusedError(f"as_of {iso(as_of)} is more than {DEADLINE} old: a real session could not be decided in time (use --pipeline-test for replays)")
    sid = f"S{as_of:%Y%m%d-%H%M}"
    out = paths.sessions / sid
    if out.exists():
        raise RefusedError(f"session {sid} already exists")
    slot = slot_of(as_of)
    if not pipeline_test:
        for other in paths.sessions.glob(f"S{slot:%Y%m%d}-*"):
            sj = other / "session.json"
            if sj.is_file():
                o = json.loads(sj.read_text(encoding="utf-8"))
                if not o.get("pipeline_test") and o.get("slot") == iso(slot):
                    raise RefusedError(f"slot {iso(slot)} already has session {other.name}: one session per slot (no re-rolls)")
    pin = pinned()
    symbols = list(symbols or DEFAULT)
    run = TEST_RUN if pipeline_test else BOOK_RUN
    book = book or Book(paths, run, journal=not pipeline_test)
    kill = paths.kill.exists()
    j = Journal(sid, root=paths.journal)

    # 1) stop cap: weekly posterior update (real sessions write it; tests only read the cap in force at as_of)
    if not pipeline_test:
        cap_doc = sp.run_update(as_of, path=paths.stop_cap, root=paths.book)
        edge_doc = ep.run_update(as_of, path=paths.edge, root=paths.book)
    else:
        cap_doc = sp.read(paths.stop_cap)
        edge_doc = ep.read(paths.edge)
    edges, edge_src = ep.applied_edges(as_of, paths.edge)
    cap, cap_src = sp.applied_cap(as_of, paths.stop_cap)

    # 2) the persistent paper book: advance over the bars completed by as_of (fills by the conservative intrabar rule)
    adv = book.advance(as_of)
    pos = book.positions()
    trades = book.trades()
    held = {p["symbol"] for p in pos["positions"]} | {o["symbol"] for o in pos["pending_orders"]}
    fam_cap = int(load().risk["max_positions_per_cluster"])
    fam_n: dict[str, int] = {}
    for x in pos["positions"] + pos["pending_orders"]:
        fam_n[x.get("family", "?")] = fam_n.get(x.get("family", "?"), 0) + 1

    # 2b) trend sleeves (agent.trend_sleeves): the H-WS16-15 monitor on the unconstrained stream, and the rule exits due
    signals_fn = signals_fn or ts.symbol_signals
    try:
        mon = (monitor_fn or (lambda t: ts.monitor(t, root=paths.trend, write=not pipeline_test, book_trades=trades)))(as_of)
    except Exception as e:  # never blocks the session; a missing monitor is shown in the packet
        mon = {"version": ts.VERSION, "as_of": iso(as_of), "forward_start": ts.FORWARD_START.isoformat(), "sleeves": {}, "error": f"{type(e).__name__}: {e}"}
    exit_errs: list[str] = []
    exits = ts.exits_due(pos["positions"], as_of, signals_fn, exit_errs)      # a failed symbol is retried next session (late exit)
    exit_err = "; ".join(exit_errs) or None
    exiting = {x["symbol"] for x in exits}

    # 3) state cards and candidates from the validated sleeves, point in time
    analyse_fn = analyse_fn or (lambda s, t: analyse(s, t, lookback_bars=LOOKBACK_BARS, cap=cap, edges=edges))
    cards, cands, states = [], [], {}
    for sym in symbols:
        try:
            c, k, st = analyse_fn(sym, as_of)
        except LookAheadError:
            raise
        except Exception as e:  # one bad symbol must not stop the session
            c, k, st = f"{sym}: unavailable ({e})", [], {"symbol": sym, "error": str(e)}
        for key in ("data_max_ts", "data_max_ts_4h"):
            if st.get(key) and st[key] != "None" and utc(st[key]) > as_of:
                raise LookAheadError(f"{sym}: {key} {st[key]} > as_of {iso(as_of)}")
        st["weekday"] = as_of.strftime("%a")
        cards.append(c)
        cands += k
        states[sym] = st

    # 4) bias gate + book + kill + engine EV gate
    gres, gctx = (gate_fn or gate_candidates)(cands, states, as_of) if cands else ({}, {})
    min_er, excl = paper_gate()
    em = entry_mode()
    rows = []
    for c in cands:
        if em["mode"] == "pegged_limit":
            # both sleeves enter with a limit pegged to the touch (H-EXE1, R-EXE-1); `entry` stays the decision-time
            # reference price, the engine costs maker + slip and the paper book fills at the next open + slip
            c.entry_type = "pegged_limit"
        g = gres.get(c.id, {"decision": "pass", "rules_applied": [], "reasons": [], "blocked_by": [], "warnings": []})
        why = []
        if c.blocked:
            why.append(c.blocked)
        if g["decision"] == "block":
            why.append("gate: " + ("; ".join(r for r in g["reasons"] if r.startswith("block")) or ", ".join(g.get("blocked_by", [])) or "block"))
        er = net_er(c)
        if er < min_er or (excl and er <= min_er):
            why.append(f"net E[R] {er:+.3f} {'<=' if excl else '<'} {min_er} (paper gate; engine rejects)")
        if c.sleeve in ts.SLEEVES:
            k_why = ts.killed(mon, c.sleeve)
            if k_why:
                why.append(f"sleeve KILLED by the H-WS16-15 monitor ({k_why})")
            try:
                rb = ts.reentry_block(c.sleeve, c.symbol, c.side, trades, signals_fn(c.symbol, as_of))
            except Exception as e:
                rb = f"re-entry state unavailable ({type(e).__name__}: {e})"
            if rb:
                why.append(rb)
        fam = SLEEVE_FAMILY.get(c.sleeve)
        if fam is not None and fam_n.get(fam, 0) >= fam_cap:
            why.append(f"family {fam} at the engine cap ({fam_n[fam]} open or pending, risk.yaml max_positions_per_cluster {fam_cap}): the engine rejects it")
        if c.symbol in held:
            why.append("symbol already in the paper book (one position per symbol)" + ("; its rule exit is sent this pass" if c.symbol in exiting else ""))
        if kill:
            why.append("KILL switch (data/exec/KILL)")
        c.blocked = "; ".join(why) if why else None
        d = asdict(c)
        d.update({"net_er": er, "gate": g, "fired": list(dict.fromkeys(g.get("blocked_by", []) + g.get("warnings", []))), "signal": g.get("signal"),
                  "asset_class": g.get("asset_class", "crypto"), "har_day": states[c.symbol].get("har_day"), "symbol_cluster": CLUSTER.get(c.symbol, "other")})
        rows.append(d)

    # 5) files
    out.mkdir(parents=True, exist_ok=False)
    meta = {"session_id": sid, "as_of": iso(as_of), "prepared_at": iso(wall), "slot": iso(slot), "off_schedule_min": round((as_of - slot).total_seconds() / 60, 1),
            "deadline": iso(as_of + DEADLINE), "pipeline_test": pipeline_test, "book_run": run, "symbols": symbols, "kill_switch": kill,
            "stop_cap_sigma": cap, "stop_cap_source": cap_src, "edges": edges, "edge_source": edge_src, "paper_gate": {"min_net_er": min_er, "exclusive": excl},
            "lookback_bars": LOOKBACK_BARS, **pin,
            "paper_entry": em,
            "tool_versions": {"gate": _gate_version(), "stop_posterior": sp.VERSION, "edge_posterior": ep.VERSION, "exec": em["exec_version"],
                              "trend_sleeves": ts.VERSION, "rules_hash": next(iter(gres.values()))["rules_hash"] if gres else None},
            "trend_monitor": _finite({"error": mon.get("error"), "sleeves": {k: {kk: v.get(kk) for kk in ("killed", "kill_reason", "closed", "mean", "ub95", "open",
                                                                                         "carried", "peg")}
                                                                             for k, v in (mon.get("sleeves") or {}).items()}}),
            "candidates": [r["id"] for r in rows]}
    _write_json(out / "session.json", meta)
    (out / "cards.txt").write_text("\n\n".join(cards), encoding="utf-8")
    _write_json(out / "candidates.json", rows)
    rx: dict = {"sent": [], "error": exit_err}
    if exits:
        try:
            rx = send_rule_exits(sid, as_of, exits, paths, book, pipeline_test, pin["model_id"], j)
        except (RefusedError, OSError, ValueError) as e:  # nothing reached the book; the exit is due again next session
            rx = {"sent": [], "error": f"{type(e).__name__}: {e}", "due": exits}
    meta["rule_exits"] = rx
    _write_json(out / "session.json", meta)
    packet = render_packet(meta, rows, states, gctx, book, adv, cap_doc, live_line(as_of, paths.live_nn), edge_doc, mon=mon, rx=rx)
    (out / "PACKET.md").write_text(packet, encoding="utf-8")
    j.write("session_start", {**pin, "as_of": iso(as_of), "pipeline_test": pipeline_test, "book_run": run, "stop_cap_sigma": cap, "tool_versions": meta["tool_versions"]})
    j.write("snapshot", {"states": list(states.values()), "packet": packet, "cards": cards})
    j.write("position_update", {"book_run": run, "advance": {k: adv[k] for k in ("fills", "trades", "cancels")}, "positions": pos})
    for f in adv["fills"]:
        j.write("fill", {"status": "paper", "book_run": run, "fill": f})
    j.write("baseline", {"RULE": [r["id"] for r in rows if not r["blocked"]], "ALL": [r["id"] for r in rows]})
    return out


def _gate_version() -> str:
    from ai_trader.execution.gate import VERSION

    return VERSION


def _entry_line(meta: dict) -> str:
    pe = meta.get("paper_entry") or {}
    if pe.get("mode") == "pegged_limit":
        return (f"entries: limit pegged to the touch, re-pegged up to {pe.get('window_min', 15):g} min, paper fill next 1h open "
                f"{pe.get('slip_bps')} bp adverse, maker fee (exec {pe.get('exec_version')}, H-EXE1 / R-EXE-2); exits unchanged")
    return f"entries: as the sleeve signals them (exec {pe.get('exec_version', '?')})"


def gates_line(as_of: datetime) -> str | None:
    """The live gate line (D2 release windows / D3 jump intensity) of ai_trader.live.gates.packet_line, in place of the
    field card's "jump probability x%" sentence (WS05: a latent mixture weight, not the chance of a visible jump; the
    decision packet never printed it). Point in time: a gates file written after as_of is not shown."""
    try:
        from ai_trader.live.gates import load as load_gates
        from ai_trader.live.gates import packet_line
    except ImportError:
        return None
    t_ms = int(utc(as_of).timestamp() * 1000)
    try:
        g = load_gates()
        if g and g.get("written_ms") and int(g["written_ms"]) > t_ms:
            return "Live gates: no gate state known at as_of (the gates file is newer)."
        out = packet_line(gates=g, now=t_ms)
    except Exception as e:  # never block a packet on a context line
        return f"Live gates: unavailable ({type(e).__name__}: {e})"
    return str(out) if out else None


def _legacy_edges(doc: dict | None) -> dict | None:
    """The edge posterior restricted to the bracket sleeves still in agent.session.PRIOR_EDGE (shock_regime is retired)."""
    from ai_trader.agent.session import PRIOR_EDGE

    if not doc or "sleeves" not in doc:
        return doc
    return {**doc, "sleeves": {k: v for k, v in doc["sleeves"].items() if k in PRIOR_EDGE}}


def _finite(o: Any) -> Any:
    """NaN / inf -> None (strict JSON in session.json)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def _exit_rule(p: dict) -> str:
    from ai_trader.agent.trend_sleeves import LEGACY, SLEEVES

    if p.get("strategy") in SLEEVES:
        return "rule exit sent" if p.get("exit_requested") else "s x side <= 0 at a 4h close"
    if p.get("strategy") in LEGACY:
        return "legacy bracket"
    return "-"


def render_packet(meta: dict, rows: list[dict], states: dict, gctx: dict, book: Book, adv: dict, cap_doc: dict | None, live: str,
                  edge_doc: dict | None = None, *, mon: dict | None = None, rx: dict | None = None) -> str:
    from ai_trader.agent import trend_sleeves as ts
    from ai_trader.risk.edge_posterior import describe as describe_edges
    from ai_trader.risk.stop_posterior import describe

    as_of = utc(meta["as_of"])
    L = [f"# Decision packet {meta['session_id']}{'  [PIPELINE TEST]' if meta['pipeline_test'] else ''}",
         f"as_of {meta['as_of']} | slot {meta['slot'][11:16]}Z" + (f" (off schedule +{meta['off_schedule_min']:.0f} min)" if meta["off_schedule_min"] > 15 else "")
         + f" | decide by {meta['deadline']} | {meta['model_id']} prompt {meta['prompt_version']} packet {meta['packet_version']}",
         _entry_line(meta),
         "PAPER book only (execution.allow_mainnet false). Size, stops, targets and caps come from the risk engine; exits of the trend sleeves come from",
         "their rule; you answer take/skip per candidate with a reason, optionally a bounded tilt, and may ask to close an open position early.", ""]
    s = book.summary()
    pos = book.positions()
    tr = book.trades()
    ids = book.ids()
    L += [f"## Book (paper run `{book.run}`)",
          f"equity {_fmt(s.get('end_equity_marked'))} (start {_fmt(s.get('start_equity'))}) | open {len(pos['positions'])} | pending {len(pos['pending_orders'])} | "
          f"closed {len(tr)} (sum R {sum(t['r_multiple'] for t in tr):+.2f}) | KILL {'ON' if meta['kill_switch'] else 'off'}"]
    if pos["positions"] or pos["pending_orders"]:
        L += ["| position | symbol | sleeve | side | entry | stop | target | exit | R now | age h |", "|---|---|---|---|---|---|---|---|---|---|"]
        for p in pos["positions"]:
            ref = ids.get(p["id"], {})
            age = (as_of - datetime.fromtimestamp(p["opened_ts"] / 1000, UTC)).total_seconds() / 3600
            L.append(f"| {ref.get('session', '?')}/{ref.get('candidate_id', p['id'][:8])} | {p['symbol']} | {p.get('strategy', '?')} | {str(p['side']).lower()} | "
                     f"{_fmt(p['entry_price'])} | {_fmt(p['stop'])} | {_fmt(p.get('target'))} | {_exit_rule(p)} | {p.get('unrealised_r', 0):+.2f} | {age:.0f} |")
        for o in pos["pending_orders"]:
            ref = ids.get(o["id"], {})
            L.append(f"| {ref.get('session', '?')}/{ref.get('candidate_id', o['id'][:8])} | {o['symbol']} | {o.get('strategy', '?')} | {str(o['side']).lower()} | "
                     f"pending {o['entry_type']} {_fmt(o['price'])} | {_fmt(o['stop'])} | {_fmt(o.get('target'))} | {_exit_rule(o)} | - | - |")
    new = [f"{t['symbol']} {t['exit_reason']} {t['r_multiple']:+.2f}R" for t in adv.get("trades", [])]
    canc = [f"{c['symbol']} {c['reason']}" for c in adv.get("cancels", [])]
    L.append(f"since last pass: {len(adv.get('fills', []))} fills; closed: {', '.join(new) or 'none'}; cancels: {', '.join(canc) or 'none'}")
    rx = rx or {}
    sent = [f"{x['symbol']} {x['strategy']} ({x['reason'].removeprefix('sleeve rule exit: ')}) -> {x['decision']}" for x in rx.get("sent", [])]
    L.append("rule exits this pass (deterministic, not yours): " + ("; ".join(sent) if sent else "none")
             + (f" | ERROR {rx['error']} (due again next session)" if rx.get("error") else ""))
    g = meta.get("paper_gate") or {}
    k = ts.stop_k(meta.get("stop_cap_sigma") or ts.STOP_K)
    L += ["", "## Stop cap, trend sleeves and the H-WS16-15 monitor (never set by you)", describe(cap_doc),
          f"trend sleeves (D4, WS16): spec A trend_zoo_ew (primary) and spec B trend_tsmom30 (benchmark). Entry |s| >= {ts.THETA:.2f} at a 4h close, "
          f"stop {k:.2f} x risk sigma 4h (the cap), exit by rule at the first 4h close with s x side <= 0, no target / trailing / pyramiding, no "
          f"re-entry after a stop until s is flat or flips; net E[R] {ts.PRIOR_NET_R:+.2f} per closed trade (WS16 forward expectation, the engine's "
          f"check); one trend position in the book at a time (risk.yaml max_positions_per_cluster). Retired: shock_regime, setup C (D5).",
          *ts.describe(mon),
          "legacy bracket edge posterior (open trend_4h position only): " + describe_edges(_legacy_edges(edge_doc)),
          f"paper EV gate: net E[R] {'>' if g.get('exclusive') else '>='} {g.get('min_net_er')} (risk.yaml min_net_expected_r_paper; live modes 0.10)",
          "", "## Live network", live]
    gl = gates_line(as_of)
    if gl:
        L.append(gl)
    L.append("")
    with_c = sorted({r["symbol"] for r in rows})
    L.append("## Market state (symbols with candidates; tested lines only)")
    if not with_c:
        L.append("no candidates this session; no symbol cards")
    for sym in with_c:
        L += compact_card(states[sym], gctx.get(sym))
    L += ["", "## Candidates (answer EVERY id; blocked ones can only be skipped)"]
    if rows:
        L += ["| id | symbol | sleeve | side | entry | stop | target | R:R | stop sigma | p_quant | net E[R] | gate | rules fired |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            gs = "pass" if not r["blocked"] else "BLOCK"
            own = r.get("take_profit") is None
            tgt = "- (signal exit)" if own else _fmt(r["take_profit"])
            rr = "-" if r.get("rr") is None else f"{r['rr']:.2f}"
            pq = "-" if r.get("p_quant") is None else f"{r['p_quant']:.3f}"
            er = f"{r['net_er']:+.3f}" + (" (prior)" if own else "")
            L.append(f"| {r['id']} | {r['symbol']} | {r['sleeve']} | {r['side']} | {r['entry_type']} {_fmt(r['entry'])} | {_fmt(r['stop'])} | {tgt} | "
                     f"{rr} | {(r.get('stop_sigma') or float('nan')):.2f} | {pq} | {er} | {gs} | {', '.join(r['fired']) or '-'} |")
        for r in rows:
            note = "; ".join(x for x in [r["rationale"], *r.get("flags", []), (f"BLOCKED: {r['blocked']}" if r["blocked"] else "")] if x)
            L.append(f"- {r['id']}: {note}")
        if any(r.get("take_profit") is None for r in rows):
            L.append("own-exit candidates: a tilt is recorded but changes nothing (no target, no p_tp; the engine checks the documented net E[R]).")
    else:
        L.append("none: no sleeve produced a new signal (answer with an empty `decisions` list)")
    fired = [x for r in rows for x in r["fired"]]
    L += ["", "## Rules (relevant, research.rules.for_prompt; never sizing)"]
    L += packet_rules({k: v for k, v in states.items() if k in with_c} or {k: v for k, v in list(states.items())[:1]}, fired)
    L += ["", "## Not included",
          "strategy_runner paper signals: not validated (driftless null, net E[R] <= 0; R-MAP-1, R-PX-3) or duplicates of the trend sleeves (R-PD-1/2).",
          "retired 2026-11-27: shock_regime (H-S2 shock fade, D5), setup C (D5), trend_4h ema_stack bracket (D4: replaced by trend_zoo_ew).",
          "", "## Answer (decisions.json)",
          '{"session_id": "' + meta["session_id"] + '", "model_id": "' + meta["model_id"] + '", "decisions": [{"candidate_id": "<id>", "action": "take|skip", '
          '"reason": "...", "tilt": {"view_logodds": 0.0, "confidence": "low|medium|high"}}], "close": [{"position_id": "<id>", "reason": "..."}]}',
          f"tilt optional, |view_logodds| <= {load().risk['llm_view']['max_abs_logodds']}; it never loosens the engine's E[R] gate. Any size/stop/target/cap field is refused."]
    return "\n".join(L) + "\n"


# ====================================================================== decide

def _control_keys(o: Any, path: str = "") -> list[str]:
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            parts = kl.replace("-", "_").split("_")
            if any(w in parts or kl == w for w in CONTROL_WORDS):
                bad.append(f"{path}{k}")
            bad += _control_keys(v, f"{path}{k}.")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _control_keys(v, f"{path}[{i}].")
    return bad


def validate_decisions(doc: Any, meta: dict, rows: list[dict], positions: dict, ids: dict[str, dict]) -> tuple[list[str], dict]:
    """(errors, normalised). Every candidate answered exactly once; take only non-blocked candidates; reasons required;
    tilts bounded; closes only for positions in the book; no field that could steer size, placement or limits."""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["decisions file must be a JSON object"], {}
    bad = _control_keys(doc)
    if bad:
        errs.append(f"refused: {bad} - size, stops, targets, caps and limits come from the risk engine; Claude cannot set or change them")
    extra = set(doc) - TOP_KEYS
    if extra:
        errs.append(f"unknown top-level keys {sorted(extra)} (allowed {sorted(TOP_KEYS)})")
    pin = pinned()
    for k in ("model_id", "prompt_version"):
        if k in doc and doc[k] != pin[k]:
            errs.append(f"{k} {doc[k]!r} != pinned {pin[k]!r} (a different model or prompt is a new track record)")
    if doc.get("session_id") not in (None, meta["session_id"]):
        errs.append(f"session_id {doc.get('session_id')!r} != {meta['session_id']}")
    lv = load().risk["llm_view"]
    cmax, scale = float(lv["max_abs_logodds"]), dict(lv["confidence_scale"])
    by = {r["id"]: r for r in rows}
    seen: dict[str, dict] = {}
    decs = doc.get("decisions", [])
    if not isinstance(decs, list):
        errs.append("decisions must be a list")
        decs = []
    for i, d in enumerate(decs):
        if not isinstance(d, dict):
            errs.append(f"decisions[{i}] is not an object")
            continue
        ex = set(d) - DECISION_KEYS
        if ex:
            errs.append(f"decisions[{i}]: unknown keys {sorted(ex)} (allowed {sorted(DECISION_KEYS)})")
        cid = d.get("candidate_id")
        if cid not in by:
            errs.append(f"decisions[{i}]: unknown candidate_id {cid!r}")
            continue
        if cid in seen:
            errs.append(f"{cid}: answered twice")
            continue
        act = d.get("action")
        if act not in ("take", "skip"):
            errs.append(f"{cid}: action must be take or skip, got {act!r}")
        reason = d.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 3:
            errs.append(f"{cid}: a reason is required")
        if act == "take" and by[cid]["blocked"]:
            errs.append(f"{cid}: blocked ({by[cid]['blocked']}); it can only be skipped")
        tilt = d.get("tilt")
        v, conf = 0.0, None
        if tilt is not None:
            if isinstance(tilt, (int, float)) and not isinstance(tilt, bool):
                v, conf = float(tilt), "medium"
            elif isinstance(tilt, dict):
                ex = set(tilt) - TILT_KEYS
                if ex:
                    errs.append(f"{cid}: unknown tilt keys {sorted(ex)} (allowed {sorted(TILT_KEYS)})")
                v, conf = tilt.get("view_logodds", 0.0), tilt.get("confidence", "medium")
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    errs.append(f"{cid}: tilt.view_logodds must be a number")
                    v = 0.0
            else:
                errs.append(f"{cid}: tilt must be a number or {{view_logodds, confidence}}")
            if not math.isfinite(float(v)) or abs(float(v)) > cmax:
                errs.append(f"{cid}: |view_logodds| {v} exceeds the bound {cmax} (ARCHITECTURE 2.4)")
            if conf not in scale:
                errs.append(f"{cid}: confidence {conf!r} not in {sorted(scale)}")
        c = scale.get(conf, 0.0) if conf else 0.0
        if by[cid].get("p_quant") is None:
            # own-exit sleeve (no target): no p_tp to tilt; the view is recorded and changes nothing
            p = p_final = p_used = None
        else:
            p = float(by[cid]["p_quant"])
            lo = math.log(p / (1 - p)) + c * float(v if isinstance(v, (int, float)) and math.isfinite(float(v)) else 0.0)
            p_final = 1 / (1 + math.exp(-lo))
            p_used = min(p, p_final)
        seen[cid] = {"candidate_id": cid, "action": act, "reason": reason, "view_logodds": float(v) if isinstance(v, (int, float)) else 0.0,
                     "confidence": conf, "scale": c, "p_quant": p, "p_final": p_final, "p_used": p_used}
    missing = [r["id"] for r in rows if r["id"] not in seen]
    if missing:
        errs.append(f"unanswered candidates {missing}: every candidate needs take or skip")
    closes = []
    held = {p["id"]: p for p in positions["positions"]} | {o["id"]: o for o in positions["pending_orders"]}
    alias = {f"{v.get('session')}/{v.get('candidate_id')}": k for k, v in ids.items()}
    cl = doc.get("close", [])
    if not isinstance(cl, list):
        errs.append("close must be a list")
        cl = []
    for i, x in enumerate(cl):
        if not isinstance(x, dict):
            errs.append(f"close[{i}] is not an object")
            continue
        ex = set(x) - CLOSE_KEYS
        if ex:
            errs.append(f"close[{i}]: unknown keys {sorted(ex)}")
        pid = x.get("position_id")
        pid = alias.get(pid, pid)
        if pid is None and x.get("symbol"):
            pid = next((k for k, v in held.items() if v["symbol"] == str(x["symbol"]).upper()), None)
        if pid not in held:
            errs.append(f"close[{i}]: no open position or pending entry {x.get('position_id') or x.get('symbol')!r} in the book")
            continue
        if not isinstance(x.get("reason"), str) or len(x["reason"].strip()) < 3:
            errs.append(f"close[{i}]: a reason is required")
        closes.append({"position_id": pid, "symbol": held[pid]["symbol"], "strategy": held[pid]["strategy"], "family": held[pid]["family"],
                       "reason": x.get("reason"), "mark": held[pid].get("mark") or held[pid].get("entry_price") or held[pid].get("price")})
    return errs, {"decisions": [seen[r["id"]] for r in rows if r["id"] in seen], "close": closes, "market_view": doc.get("market_view"), "notes": doc.get("notes")}


def build_intents(meta: dict, rows: list[dict], norm: dict) -> list[dict]:
    """OrderIntents (docs/EXECUTION_PIPELINE.md) for the takes and early closes. Never a size: the engine sizes.
    An own-exit candidate (trend sleeves: no target) sends targets [] and net_er_r = its documented expectation (the
    engine checks that value); p_tp is the reference driftless P(+3R before the stop), as execution.gate does."""
    from ai_trader.agent.session import Candidate, net_er
    from ai_trader.agent.trend_sleeves import REF_P_TP
    from ai_trader.execution.intents import validate_intent

    as_of = utc(meta["as_of"])
    ts_ms = int(as_of.timestamp() * 1000)
    by = {r["id"]: r for r in rows}
    out = []
    fields = set(Candidate.__dataclass_fields__)
    for d in norm["decisions"]:
        if d["action"] != "take":
            continue
        r = by[d["candidate_id"]]
        c = Candidate(**{k: v for k, v in r.items() if k in fields})
        own = r.get("take_profit") is None
        if own:
            er, er_null, g_net, p_tp = float(net_er(c)), None, None, REF_P_TP
        else:
            c.p_quant = d["p_used"]
            er = float(net_er(c))
            c0 = Candidate(**{k: v for k, v in r.items() if k in fields})
            c0.p_quant = float(r["p_null"])
            er_null = float(net_er(c0))
            c0.p_quant = 1.0
            g_net = float(net_er(c0))
            p_tp = d["p_used"]
        g = r["gate"]
        out.append({
            "intent_id": str(uuid.uuid5(NS, f"take|{meta['session_id']}|{r['id']}")), "ts_ms": ts_ms, "as_of": as_of.isoformat(), "mode": "paper",
            "strategy": r["sleeve"], "family": SLEEVE_FAMILY[r["sleeve"]], "symbol": r["symbol"], "asset_class": r.get("asset_class", "crypto"), "side": r["side"],
            "signal": float(r.get("signal") or (1.0 if r["side"] == "long" else -1.0)), "entry": ({"type": "limit", "price": float(r["entry"]), "peg": "touch"} if r["entry_type"] == "pegged_limit"
                      else {"type": r["entry_type"], "price": float(r["entry"])}),
            "stop": float(r["stop"]), "targets": [] if own else [float(r["take_profit"])],
            "sigma": {"per_bar": r.get("sigma_per_bar"), "daily_logHAR": r.get("har_day"), "weekday_factor": None, "source": "vol.har", "freq": r.get("freq", "1h")},
            "p_tp": p_tp, "net_er_r": er,
            "gate": {"decision": g.get("decision", "pass"), "rules_applied": g.get("rules_applied", []), "reasons": g.get("reasons", []),
                     "blocked_by": g.get("blocked_by", []), "stop_sigma": r.get("stop_sigma")},
            "context": {"session_id": meta["session_id"], "candidate_id": r["id"], "harness": HARNESS, "decision": "take", "reason": d["reason"],
                        "view_logodds": d["view_logodds"], "confidence": d["confidence"], "p_quant_session": d["p_quant"], "p_final": d["p_final"],
                        "p_null": None if own else float(r["p_null"]), "er_null": er_null, "g_net": g_net,
                        "exit_rule": r.get("exit_rule"), "sleeve_signal": r.get("sleeve_signal"), "signal_close": r.get("signal_close"),
                        "expected_r": r.get("expected_r"), "p_tp_meaning": "reference driftless P(+3R before the stop); no target" if own else "p_used",
                        "session_cluster": r["cluster"], "flags": r.get("flags", []), "rationale": r["rationale"], "pipeline_test": meta["pipeline_test"],
                        "model_id": meta["model_id"], "prompt_version": meta["prompt_version"]},
        })
    for x in norm["close"]:
        out.append({
            "intent_id": str(uuid.uuid5(NS, f"close|{meta['session_id']}|{x['position_id']}")), "ts_ms": ts_ms, "as_of": as_of.isoformat(), "mode": "paper",
            "strategy": x["strategy"], "family": x["family"], "symbol": x["symbol"], "asset_class": "crypto", "side": "flat", "signal": 0.0,
            "entry": {"type": "market", "price": float(x["mark"])}, "stop": None, "targets": [],
            "sigma": {"per_bar": None, "daily_logHAR": None, "weekday_factor": None, "source": "vol.har"}, "p_tp": None, "net_er_r": None,
            "gate": {"decision": "pass", "rules_applied": [], "reasons": ["LLM exit view (ARCHITECTURE 3.12): early close requested"], "blocked_by": [], "stop_sigma": None},
            "context": {"session_id": meta["session_id"], "position_id": x["position_id"], "harness": HARNESS, "decision": "close", "reason": x["reason"],
                        "pipeline_test": meta["pipeline_test"], "model_id": meta["model_id"]},
        })
    for it in out:
        validate_intent(it)
    return out


def decide(session_id: str, doc: dict, *, paths: Paths | None = None, wall: datetime | None = None, simulate_at: datetime | None = None,
           book: Book | None = None, raw: str | None = None) -> dict:
    safety_check()
    paths = paths or Paths.default()
    sdir = paths.sessions / session_id
    if not (sdir / "session.json").is_file():
        raise RefusedError(f"no session {session_id} under {paths.sessions}")
    meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    rows = json.loads((sdir / "candidates.json").read_text(encoding="utf-8"))
    j = Journal(session_id, root=paths.journal)

    def refuse(msg: str) -> None:
        rec = {"ts": iso(now()), "refused": msg}
        with (sdir / "refusals.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        j.write("decision", {"status": "refused", "reason": msg, "raw": raw})
        raise RefusedError(msg)

    if (sdir / "submitted.json").exists():
        refuse(f"session {session_id} is already decided")
    wall = utc(wall or now())
    t_dec = wall
    if simulate_at is not None:
        if not meta.get("pipeline_test"):
            refuse("--simulate-at is only allowed for pipeline_test sessions")
        t_dec = utc(simulate_at)
    as_of, deadline = utc(meta["as_of"]), utc(meta["deadline"])
    if t_dec < as_of:
        refuse(f"decision time {iso(t_dec)} is before the session as_of {meta['as_of']}")
    if t_dec > deadline:
        refuse(f"deadline passed: decision at {iso(t_dec)} > {meta['deadline']} (as_of + {DEADLINE}); a session cannot be decided with hindsight")
    book = book or Book(paths, meta["book_run"], journal=not meta.get("pipeline_test"))
    errs, norm = validate_decisions(doc, meta, rows, book.positions(), book.ids())
    if errs:
        refuse("invalid decisions: " + " | ".join(errs))
    intents = build_intents(meta, rows, norm)
    ip = paths.book / "intents" / f"{session_id}.jsonl"
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in intents), encoding="utf-8")
    (sdir / "intents.jsonl").write_text(ip.read_text(encoding="utf-8"), encoding="utf-8")
    with (paths.book / "ids.jsonl").open("a", encoding="utf-8") as f:
        for x in intents:
            f.write(json.dumps({"intent_id": x["intent_id"], "session": session_id, "candidate_id": x["context"].get("candidate_id"),
                                "kind": x["context"]["decision"], "book_run": meta["book_run"]}) + "\n")
    res = book.submit(ip, t_dec, {x["intent_id"] for x in intents}) if intents else {"decisions": [], "summary": book.summary()}
    by_iid = {x["intent_id"]: x for x in intents}
    acc, rej = [], {}
    for d in res["decisions"]:
        cid = by_iid[d["intent_id"]]["context"].get("candidate_id") or by_iid[d["intent_id"]]["context"].get("position_id")
        if d["decision"] == "accept":
            acc.append(cid)
        else:
            rej[cid] = d["rejections"]
    takes = [d["candidate_id"] for d in norm["decisions"] if d["action"] == "take"]
    test = bool(meta.get("pipeline_test"))
    sub = {"session_id": session_id, "harness": HARNESS, "pipeline_test": test, "ids": takes, "tests": takes if test else [],
           "skipped": [d["candidate_id"] for d in norm["decisions"] if d["action"] == "skip"], "accepted": acc, "rejected": rej,
           "closes": [c["position_id"] for c in norm["close"]], "p_final": {d["candidate_id"]: d["p_final"] for d in norm["decisions"]},
           "decided_at": iso(t_dec), "wall_clock": iso(wall), "model_id": meta["model_id"], "prompt_version": meta["prompt_version"],
           "packet_version": meta["packet_version"], "book_run": meta["book_run"], "intents_file": str(ip)}
    _write_json(sdir / "decision.json", {"input": doc, "normalised": norm, "decided_at": iso(t_dec)})
    _write_json(sdir / "submitted.json", sub)
    j.write("decision", {"status": "accepted_by_harness", "model_id": meta["model_id"], "prompt_version": meta["prompt_version"], "packet_version": meta["packet_version"],
                         "decisions": norm["decisions"], "close": norm["close"], "market_view": norm.get("market_view"), "raw": raw, "decided_at": iso(t_dec)})
    for d in norm["decisions"]:
        if d["confidence"]:
            j.write("view", {"candidate_id": d["candidate_id"], "view_logodds": d["view_logodds"], "confidence": d["confidence"], "p_quant": d["p_quant"], "p_final": d["p_final"]})
    j.write("risk_decision", {"book_run": meta["book_run"], "decisions": res["decisions"]})
    j.write("session_end", {"submitted": sub})
    return sub


# ====================================================================== status

def status(paths: Paths | None = None, record: bool = False, t: datetime | None = None) -> str:
    from ai_trader.agent.scorecard import ai_stats, run, summary
    from ai_trader.research.hypotheses import record as rec
    from ai_trader.research.hypotheses import results
    from ai_trader.risk import stop_posterior as sp

    paths = paths or Paths.default()
    book = Book(paths, BOOK_RUN, journal=False)
    s, pos, tr = book.summary(), book.positions(), book.trades()
    L = [f"# Paper book `{BOOK_RUN}` ({s.get('as_of', 'never advanced')})",
         f"equity {_fmt(s.get('end_equity_marked'))} (start {_fmt(s.get('start_equity'))}), open {len(pos['positions'])}, pending {len(pos['pending_orders'])}, "
         f"closed {len(tr)}, sum R {sum(x['r_multiple'] for x in tr):+.2f}, wins {sum(1 for x in tr if x['r_multiple'] > 0)}"]
    for p in pos["positions"]:
        L.append(f"  {p['symbol']} {str(p['side']).lower()} entry {_fmt(p['entry_price'])} stop {_fmt(p['stop'])} target {_fmt(p.get('target'))} R {p.get('unrealised_r', 0):+.2f}")
    for x in tr[-10:]:
        L.append(f"  closed {x['symbol']} {str(x['side']).lower()} {x['exit_reason']} {x['r_multiple']:+.2f}R")
    df = run(paths.sessions, t)
    L.append("\n# Scorecard (shadow R on the same candidates; pipeline_test excluded)")
    for r in summary(df):
        L.append("  " + ", ".join(f"{k} {round(v, 3) if isinstance(v, float) else v}" for k, v in r.items()))
    st = ai_stats(df)
    L.append(f"  H-AI1: {json.dumps(st['H-AI1'], default=float)}")
    L.append(f"  H-AI2: {json.dumps(st['H-AI2'], default=float)}")
    try:
        from ai_trader.agent import trend_sleeves as ts

        mon = ts.monitor(t or now(), root=paths.trend, write=False, book_trades=tr)
        L += ["\n# Trend sleeves (D4) and the H-WS16-15 monitor (unconstrained stream; read only)"] + ["  " + x for x in ts.describe(mon)]
    except Exception as e:  # status must always print
        L.append(f"\n# H-WS16-15 monitor unavailable: {type(e).__name__}: {e}")
    doc = sp.read(paths.stop_cap)
    L += ["\n# Stop posterior", "  " + sp.describe(doc)]
    if record:
        done = []
        for hid, key in (("H-AI1", "closed_ai"), ("H-AI2", "resolved")):
            m = st[hid]
            if m.get("ready") and m.get("verdict") and not any(r["metrics"].get(key) == m[key] for r in results(hid)):
                rec(hid, "clean_holdout", m["verdict"], m, note="forward paper sessions (scripts/decision_session.py status --record)", data="data/sessions")
                done.append(hid)
        if doc and doc["n"] >= 40:
            q = doc["posterior"]["q90_population"]
            v = "supports" if q["p_outside_band"] > 0.9 else ("contradicts" if q["p_outside_band"] < 0.1 else "inconclusive")
            if not any(r["metrics"].get("n") == doc["n"] for r in results("H-STOP1")):
                rec("H-STOP1", "clean_holdout", v, {"n": doc["n"], **q, "q90_predictive": doc["q90_predictive"], "applied_cap_sigma": doc["applied_cap_sigma"]},
                    note="weekly stop posterior", data=str(paths.stop_cap))
                done.append("H-STOP1")
        L.append(f"\n--record: {'recorded ' + ', '.join(done) if done else 'nothing recorded (minimum samples not reached or already recorded)'}")
    return "\n".join(L)


def load_decisions(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw), raw
