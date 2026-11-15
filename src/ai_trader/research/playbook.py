"""Decision playbook: research/rules.yaml compiled into state-conditioned cards plus a conflict audit (agent F, 2026-11-12).

    PYTHONIOENCODING=utf-8 uv run python scripts/playbook.py        # writes research/PLAYBOOK.md, data/research/playbook.json,
                                                                     # research/reports/rules_audit_<date>.md
    from ai_trader.research.playbook import card_for
    print(card_for("next_day_vol", {"asset_class": "crypto", "weekday": "Sat", "vol_state": "high"}))

rules.yaml is read only here: the audit REPORTS conflicts and stale rules, it never edits them (rules change only through
registered tests).

Facets. Every rule is parsed into facets from its scope and text (deterministic keyword rules, see `facets`):
  assets    {crypto, tradfi}             from scope.assets (symbol lists mapped by class)
  horizon   {micro, minutes, hour, day, multiday, long}   from scope.horizon ("1h-24h" covers hour and day, ...)
  topics    {vol, direction, execution, quoting, strategy, flow}
  conds     machine-checkable WHEN conditions: weekday (of the forecast day), vol_state, dir_state (v0 trend), cell
            (R1..R9), session (nyse_open, us_open, cme_reopen, off_hours, funding, expiry), event (jump, shock,
            liquidation, drawdown, funding_crowded, flow_anomaly, squeeze)
  claims    vol: up / down / null (an explicit "no effect"); direction: continue / revert / null
  tools     tools the rule says to call (positive rules) or to skip (negative / avoid rules)

Cards. A context (CONTEXTS) is a decision type with asset / horizon / topic filters and keyword cues. `card_for` keeps
active and provisional rules only, drops a rule whose machine-checkable condition contradicts the given state (a
Monday-only rule on a Saturday), ranks the rest by relevance (topic, horizon, cue words, matched state conditions),
keeps the <= 8 most relevant (R-BUDGET-1: short cards) and prints them in grade order A, B, C. Tools to call (in
pipeline order: level -> calendar / seasonal adjustment -> refinement -> explanation) and tools to skip are read off
the relevant rules.

Audit.
  * pair conflicts: two rules whose facets overlap (asset, horizon, topic, compatible state conditions sharing at least
    one condition key, or the same decision for tool choice) and whose claims are opposite (up vs down, continue vs
    revert), one negates the other's effect (explicit "no ... multiplier" vs a sized effect), they prefer different
    tools for the same decision, or they duplicate each other (same condition, same claim); plus CURATED_PAIRS found on
    reading and REVIEWED notes on flags that are not conflicts;
  * evidence (polarity-free): a cited hypothesis whose in-sample and holdout verdicts disagree (segment flip; an A
    grade needs both segments to agree), A/B rules whose cited results are all inconclusive, and curated cross-checks
    (CROSS_CHECKS) for results that bear on a rule without being cited by it (e.g. a multi-day re-test);
  * review triggers: provisional rules whose review condition has fired (a hypothesis named in the review, other
    than the rule's own evidence, now has a result or a decidable forward test; the data the review asks for now
    exists; a date passed).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ai_trader.research.rules import GRADES, load_rules

VERSION = "0.1.0"
MAX_RULES = 8
ACTIVE = ("active", "provisional")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
HORIZONS = ("micro", "minutes", "hour", "day", "multiday", "long")

# ====================================================================== facets

TRADFI_HINTS = re.compile(r"tradfi|equity|\bTSLA|INTC|GOOGL|AAPL|MSTR|NVDA|\bMU\b|QQQ|SOXL|SPCX|XAU|XAG|\bCL|\bBZ|PAXG|XAUT|commodit|universe T", re.I)
CRYPTO_HINTS = re.compile(r"crypto|universe L|BTC|ETH|SOL|XRP|DOGE|HYPE|ATOM|CARV|KAS|alts?\b|tokens?", re.I)
TRADFI_SYMS = {"TSLAUSDT", "INTCUSDT", "GOOGLUSDT", "AAPLUSDT", "MSTRUSDT", "NVDAUSDT", "MUUSDT", "QQQUSDT", "SOXLUSDT",
               "SPCXUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "PAXGUSDT", "XAUTUSDT"}


def asset_set(scope_assets: Any) -> set[str]:
    if isinstance(scope_assets, list):
        out = {("tradfi" if s in TRADFI_SYMS else "crypto") for s in scope_assets}
        return out or {"crypto", "tradfi"}
    s = str(scope_assets)
    if s.strip().lower() in ("all", "any") or re.search(r"\ball\b", s, re.I):
        return {"crypto", "tradfi"}
    out = set()
    if CRYPTO_HINTS.search(s):
        out.add("crypto")
    if TRADFI_HINTS.search(s):
        out.add("tradfi")
    return out or {"crypto", "tradfi"}


_UNIT_MIN = {"ms": 1 / 60000, "s": 1 / 60, "sec": 1 / 60, "m": 1, "min": 1, "minute": 1, "minutes": 1, "h": 60, "hour": 60, "hours": 60,
             "d": 1440, "day": 1440, "days": 1440, "w": 10080, "wk": 10080, "week": 10080, "weeks": 10080, "month": 43200}


def _bucket(minutes: float) -> str:
    if minutes < 1:
        return "micro"
    if minutes < 60:
        return "minutes"
    if minutes < 1440:
        return "hour"
    if minutes <= 1440:
        return "day"
    if minutes <= 7 * 1440:
        return "multiday"
    return "long"


def horizon_set(h: Any) -> set[str]:
    s = str(h).lower()
    if s in ("any", "") or "any" in s:
        return set(HORIZONS)
    out: set[str] = set()
    if "tick" in s or "sub-second" in s:
        out.add("micro")
    if "weekend" in s:
        out |= {"day", "multiday"}
    if "intraday" in s or "nyse close" in s or "cash hour" in s:
        out.add("hour")
    if s.strip() == "minutes":
        out.add("minutes")
    vals = []
    # ranges such as "1h-24h", "3s-5m", "0.1-60 s", "1-1000 orders", "4h-1d bars", "12h-2d", "1 day"
    for a, ua, b, ub in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]*)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*([a-z]+)", s):
        u2 = _UNIT_MIN.get(ub.rstrip("s") if ub not in _UNIT_MIN else ub)
        u1 = _UNIT_MIN.get(ua, u2) if ua else u2
        if u2 is not None and u1 is not None:
            vals += [float(a) * u1, float(b) * u2]
    for a, u in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|sec|m|min|minutes?|h|hours?|d|days?|w|wk|weeks?|month)\b", s):
        vals.append(float(a) * _UNIT_MIN[u])
    if vals:
        lo, hi = min(vals), max(vals)
        i0, i1 = HORIZONS.index(_bucket(lo)), HORIZONS.index(_bucket(hi))
        out |= set(HORIZONS[i0:i1 + 1])
    return out or set(HORIZONS)


TOPIC_RE = {
    "vol": r"volatil|variance|\bvol\b|\bsigma|\brv\b|realised|realized|quiet|jump|half-life|\bHAR\b|log-HAR",
    "direction": r"direction|continu|revers|revert|\bfade|fading|drift|follow|momentum|trend|breakout|\blean|tilt|up-lean|keep going|\bpositive (net )?trade",
    "execution": r"\bstop|exit|bracket|take-profit|target is hit|order|execut|fill|latency|deliberation",
    "quoting": r"quot|market-making|market making|inventory|fills?\b|spread|markout|\batx mm\b|simulat|generator|replay",
    "strategy": r"strateg|sleeve|book\b|candidate|regime cell|v0 cell|parameter|champion|ranking",
    "flow": r"order flow|taker|imbalance|liquidation|flow anomal|aggressive|prints|trade arrivals|order signs|durations",
}


def topics_of(r: dict) -> set[str]:
    txt = f"{r['rule']} {r['when']} {r['then']}"
    return {t for t, p in TOPIC_RE.items() if re.search(p, txt, re.I)}


def _then_text(r: dict) -> str:
    """The THEN part of the rule sentence (claims are read from it, not from the WHEN part)."""
    s = str(r["rule"])
    parts = s.split(" THEN ", 1)
    return (parts[1] if len(parts) > 1 else s) + " " + str(r["then"])


def conditions(r: dict) -> dict[str, set[str]]:
    """Machine-checkable WHEN conditions (read from the WHEN part of the rule and the `when` field)."""
    w = str(r["rule"]).split(" THEN ", 1)[0] + " " + str(r["when"])
    wl = w.lower()
    c: dict[str, set[str]] = {}
    wd: set[str] = set()
    if re.search(r"fri(day)?\s*-\s*sun|fri-sun", wl):
        wd |= {"fri", "sat", "sun"}
    if re.search(r"\bweekend|next_is_weekend|saturday or sunday", wl) and "weekend window" not in wl and "spans a weekend" not in wl and "after a weekend" not in wl:
        wd |= {"sat", "sun"}
    if "weekend window" in wl or "calendar.weekend" in wl:
        wd |= {"sat", "sun"}      # Fri 17:00 ET - Sun 18:00 ET: the forecast days it covers are Sat and Sun
    if re.search(r"\bsaturday\b", wl):
        wd.add("sat")
    if re.search(r"\bsunday\b", wl):
        wd.add("sun")
    if re.search(r"\bmonday\b|next_is_monday|monday_us_open", wl):
        wd.add("mon")
    if re.search(r"tue(sday)?\s*(-|to)\s*fri(day)?", wl):
        wd |= {"tue", "wed", "thu", "fri"}
    if re.search(r"thu 08:00 to sat", wl):
        wd |= {"thu", "fri", "sat"}
    if wd:
        c["weekday"] = wd
    vs: set[str] = set()
    if re.search(r"high-vol|vol_state == high|volatile market|vol is already high|top third|vol is high|high vol", wl):
        vs.add("high")
    if re.search(r"calm market|low vol state|vol_state == low|low-vol|7-day vol level is low|lower half", wl):
        vs.add("low")
    if re.search(r"vol_state normal", wl):
        vs.add("normal")
    if vs:
        c["vol_state"] = vs
    ds: set[str] = set()
    if re.search(r"trend (state )?(is )?up|trend up|v0 trend up|trend state is up", wl):
        ds.add("up")
    if re.search(r"range or (trend )?down", wl):
        ds |= {"range", "down"}
    if ds:
        c["dir_state"] = ds
    cells = set(re.findall(r"\bR([1-9])\b", w))
    if cells:
        c["cell"] = {f"R{x}" for x in cells}
    ss: set[str] = set()
    if re.search(r"nyse|09:00 et|cash open|first cash hour|09:30 et", wl) and not re.search(r"outside nyse|nyse_open == false", wl):
        ss.add("nyse_open")
    if re.search(r"us open|us cash open|13:00-16:00 utc", wl):
        ss.add("us_open")
    if "cme reopen" in wl or "sunday 22:00" in wl:
        ss.add("cme_reopen")
    if "outside nyse hours" in wl or "nyse_open == false" in wl:
        ss.add("off_hours")
    if "settlement" in wl or "funding hour" in wl:
        ss.add("funding")
    if "expiry" in wl:
        ss.add("expiry")
    if ss:
        c["session"] = ss
    ev: set[str] = set()
    for k, p in {"jump": r"\bjump", "shock": r"shock|\|z(_1h)?\| ?>= ?4|spike", "liquidation": r"liquidation|\bR9\b",
                 "drawdown": r"drawdown", "funding_crowded": r"funding is crowded|crowded", "flow_anomaly": r"flow anomal|imbalance|mahalanobis",
                 "squeeze": r"squeeze"}.items():
        if re.search(p, wl):
            ev.add(k)
    if ev:
        c["event"] = ev
    return c


NEG_START = re.compile(r"^\s*(do not|don't|not\b|no\b|avoid|never|treat it as no better|treat .* as no better)", re.I)
NULL_EFFECT = re.compile(r"\bno (blanket |extra |special |textbook )?[\w-]* ?(multiplier|effect|premium|drift|regime|edge|asymmetry|information|follow-through)|"
                         r"(do not|don't) expect (extra|higher|pinned|a |the )?[\w ()-]{0,30}(vol|variance|amplification|rise)", re.I)
NULL_UP = re.compile(r"extra|higher|rise|amplif|multiplier|release|leverage-effect", re.I)
NULL_DOWN = re.compile(r"lower|pinned|discount", re.I)
VOLW = r"\bvol\b|volatil|variance|sigma|\brv\b"   # not 'volume'
SIZED = re.compile(r"[+-]?\d+(\.\d+)?\s*%|\d(\.\d+)?\s*x\b|\d(\.\d+)?x", re.I)


def polarity(r: dict) -> str:
    """'negative' if the rule says what NOT to do / expect, else 'positive'."""
    then = str(r["rule"]).split(" THEN ", 1)[-1]
    return "negative" if r["kind"] == "avoid" or NEG_START.search(then) else "positive"


def claims(r: dict) -> dict[str, set[str]]:
    t = _then_text(r)
    tl = t.lower()
    lead = str(r["rule"]).split(" THEN ", 1)[-1].lower()
    out: dict[str, set[str]] = {}
    v: set[str] = set()
    dir_words = re.search(r"continu|follow|direction|revers|fade|drift|trend", lead)
    m = NULL_EFFECT.search(lead)
    if m:   # an explicit "no effect" claim: which direction it denies, if readable
        ph = m.group(0)
        v.add("null_up" if NULL_UP.search(ph) else ("null_down" if NULL_DOWN.search(ph) else "null"))
    else:
        if polarity(r) == "negative" and re.search(VOLW, lead) and not dir_words:
            v.add("null")
        if re.search(r"\b(above|higher|raise[sd]?|rise|widen|elevated|loud|increase|amplif|more (aggressive|fills)|well above)\b", lead):
            v.add("up")
        if re.search(r"\b(below|lower|smaller|quiet|discount|decay|dampen|less)\b|0\.\d+-0\.\d+x", lead):
            v.add("down")
    if v and re.search(VOLW, tl):
        out["vol"] = v
    d: set[str] = set()
    if re.search(r"no more often than|no (night )?premium|no better than|the same on", lead):
        d.add("null")
    elif re.search(r"(do not|don't) fade|keep going|expect continuation|continue rather|same sign", tl):
        d.add("continue")
    if re.search(r"expect reversal|revert|revisit", lead) and "null" not in d:
        d.add("revert")
    if re.search(r"(do not|don't) (expect|infer|tilt|trade|use|read|call)[^.;]{0,60}(continu|follow|direction|drift|lean|trend|breakout|positive net|pay)", lead):
        d.add("null")
    if d:
        out["direction"] = d
    return out


# canonical tools: (regex, name, stage) - stage 0 level, 1 calendar / seasonal, 2 refinement, 3 explanation, 4 risk
TOOLS: list[tuple[str, str, int]] = [
    (r"per-asset log-HAR|log-HAR|vol\.har\b", "vol.har (per-asset log-HAR level)", 0),
    (r"panel (log-)?HAR|panel HAR|direct 5d HAR", "panel log-HAR (short history / multi-day)", 0),
    (r"HAR-Q", "HAR-Q (level-HAR variant)", 2),
    (r"vol-level tool|robust sigma or 7-day|7-day mean log RV", "vol level: robust sigma or 7d mean log RV", 0),
    (r"robust_bc|robust (per-bar )?sigma", "vol.robust_bc (robust per-bar sigma)", 0),
    (r"hour-of-week|seasonal_factor", "seasonality.seasonal_factor (hour-of-week)", 1),
    (r"weekday|calendar tool|next_is_weekend|weekend-scaled", "calendar: forecast weekday / weekend hours", 1),
    (r"LAT1 cell", "LAT1 cell adjustment (weekday x vol tercile x v0 trend)", 2),
    (r"gradient-boosted|XGBoost|boosted[- ]tree|tree forecast", "XGB daily vol tool (when validated)", 2),
    (r"validated hourly ML tool|hourly model|tools\.vol_ml", "tools.vol_ml hourly XGB (when validated, H-NN10)", 2),
    (r"last 1-6 hours of absolute returns", "recent |r| (1-6h) + daily HAR level adjustment", 2),
    (r"DVOL|implied vol", "DVOL (BTC/ETH 7d)", 2),
    (r"TreeSHAP", "grouped TreeSHAP attribution", 3),
    (r"jump-aware first-passage", "jump-aware first-passage p_tp", 4),
    (r"unshrunk risk sigma|sigma_risk_next", "sigma_risk_next (unshrunk, for stops / risk)", 4),
    (r"Student-t", "HAR variance through a Student-t (2-sigma bar odds)", 2),
    (r"linear model of 10 / 60 / 300 s", "linear 10/60/300 s activity model (next minute)", 2),
    (r"AFT model", "XGB AFT next-arrival model (majors)", 2),
]
SKIP_TOOLS: list[tuple[str, str]] = [
    (r"jump-variation", "jump variation as a vol input"), (r"semivariance|(?-i:RS-)|(?-i:SHAR)", "semivariance / SHAR terms"),
    (r"range-estimator|range estimators|Garman-Klass|Parkinson", "range estimators (GK / Parkinson)"),
    (r"raw-return", "raw-return inputs"), (r"sparse-autoencoder|SAE", "sparse-autoencoder features"),
    (r"direction probability|direction THEN don't call|next-day direction THEN|next-minute direction", "ML direction models"),
    (r"wider neural network", "wider networks for daily vol"),
    (r"single-lambda EWMA|plain EWMA", "plain / single-lambda EWMA"), (r"variance-managed", "variance-managed exposure"),
    (r"rolling MLE signal-to-noise", "LLT signal-to-noise 'trendiness'"), (r"imbalance has \|robust z\| > 4|imbalance z", "flow imbalance |z| > 4"),
    (r"multivariate flow score", "Mahalanobis flow score"), (r"scale the 1h robust_bc sigma flat", "flat sqrt(h) scaling of the 1h sigma"),
    (r"DVOL", "DVOL"), (r"funding-hour and jump flags", "funding-hour / jump flags (next hour)"),
    (r"BTC vol spillover|compression terms", "BTC spillover / compression terms"), (r"cell table|discrete market state", "cell tables for direction"),
]


def tools_of(r: dict) -> tuple[list[tuple[str, int]], list[str]]:
    """(tools to call with their stage, tools to skip) named by the rule."""
    txt = f"{r['rule']} {r['then']}"
    neg = polarity(r) == "negative"
    call, skip = [], []
    if not neg:
        lead = str(r["rule"]).split(" THEN ", 1)[-1]
        pos_txt = re.split(r"\bover\b", lead, maxsplit=1)[0] + " " + str(r["then"])   # "prefer X over Y": only X is called
        for p, name, st in TOOLS:
            if re.search(p, pos_txt, re.I):
                call.append((name, st))
        # "prefer X over Y": Y is a skip candidate
        m = re.search(r"\bover (the )?([^;.]+)", str(r["rule"]).split(" THEN ", 1)[-1])
        if m:
            for p, name in SKIP_TOOLS:
                if re.search(p, m.group(2), re.I):
                    skip.append(name)
    else:
        for p, name in SKIP_TOOLS:
            if re.search(p, txt, re.I):
                skip.append(name)
    if re.search(r"don't call|do not call|may be skipped|skip those", txt, re.I):
        for p, name in SKIP_TOOLS:
            if re.search(p, txt, re.I) and name not in skip:
                skip.append(name)
    return call, skip


@dataclass
class Facets:
    rid: str
    kind: str
    grade: str
    status: str
    assets: set[str]
    horizon: set[str]
    topics: set[str]
    conds: dict[str, set[str]]
    claims: dict[str, set[str]]
    polarity: str
    call: list[tuple[str, int]] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)


def facets(r: dict) -> Facets:
    call, skip = tools_of(r)
    return Facets(r["id"], r["kind"], r["confidence"], r["status"], asset_set(r["scope"].get("assets")), horizon_set(r["scope"].get("horizon")),
                  topics_of(r), conditions(r), claims(r), polarity(r), call, skip)


# ====================================================================== contexts

@dataclass(frozen=True)
class ContextDef:
    key: str
    title: str
    assets: tuple[str, ...]
    horizon: tuple[str, ...]
    topics: tuple[str, ...]
    cues: str                         # regex of words that make a rule more relevant
    state: tuple[tuple[str, str], ...] = ()   # implied state (key, value) for the context itself
    exclude: str = ""                 # regex: rules matching it are not relevant here
    require: tuple[tuple[str, tuple[str, ...]], ...] = ()   # rule must carry one of these conditions ...
    require_text: str = ""            # ... or match this regex (any-of); empty = no requirement


CONTEXTS: dict[str, ContextDef] = {c.key: c for c in (
    ContextDef("next_day_vol", "Next-day (1d) volatility for a crypto perp", ("crypto",), ("day",), ("vol",),
               r"next-day|next day|daily|1d|log-HAR|HAR|weekday|forecast day|variance|weekend|saturday|sunday|monday", exclude=r"\boptions?\b|squeeze|Moreira|variance-managed"),
    ContextDef("next_day_vol_tradfi", "Next-day (1d) volatility for a TradFi perp", ("tradfi",), ("day",), ("vol",),
               r"next-day|daily|HAR|panel|short-history|TradFi|weekend|hour-of-week", exclude=r"\boptions?\b|Moreira|variance-managed"),
    ContextDef("next_hour_vol", "Next-hour volatility (intraday expectations, quote widths)", ("crypto", "tradfi"), ("hour",), ("vol",),
               r"next-hour|hour-of-week|1h|hourly|intraday|robust|jump|half-life", exclude=r"\boptions?\b|synthetic|generated paths"),
    ContextDef("next_day_direction", "Next-day direction for a crypto perp", ("crypto",), ("day", "hour"), ("direction",),
               r"direction|next-day|next 24|drift|tilt|continu|revers|fade|trend|regime cell|cell", exclude=r"quot|fill"),
    ContextDef("weekend", "Weekend (Fri 17:00 ET to Sun 18:00 ET; forecast day Sat / Sun)", ("crypto", "tradfi"), ("day", "multiday"), ("vol", "direction"),
               r"weekend|saturday|sunday|fri-sun|friday", state=(("weekday", "sat"),),
               require=(("weekday", ("sat", "sun")),), require_text=r"weekend"),
    ContextDef("monday", "Monday (CME reopen, Monday US open, forecast day Monday)", ("crypto", "tradfi"), ("hour", "day"), ("vol", "direction"),
               r"monday|cme reopen|sunday 22|us open", state=(("weekday", "mon"),),
               require=(("weekday", ("mon",)), ("session", ("cme_reopen", "us_open"))), require_text=r"monday|after a weekend"),
    ContextDef("tradfi_open", "TradFi session open (NYSE 09:30 ET) for equity perps", ("tradfi",), ("hour", "day"), ("direction", "vol"),
               r"nyse|cash open|overnight|09:00 et|first cash hour|equity perp|us cash open|opening-range", state=(("session", "nyse_open"),),
               require=(("session", ("nyse_open", "us_open", "off_hours")),), require_text=r"nyse|overnight|cash (open|hour)|equity perp|intraday"),
    ContextDef("high_vol", "High-vol regime / just after a spike or jump", ("crypto", "tradfi"), ("hour", "day", "multiday"), ("vol", "direction"),
               r"high-vol|spike|jump|shock|volatile|vol state|half-life|whipsaw", state=(("vol_state", "high"),),
               require=(("vol_state", ("high",)), ("event", ("jump", "shock"))), require_text=r"high-vol|vol is high|volatile|spike|surprise|exceeds 1\.5x|whipsaw"),
    ContextDef("liquidation", "Liquidation regime / liquidation clusters / flow anomalies", ("crypto",), ("minutes", "hour", "day", "multiday"), ("flow", "direction", "vol"),
               r"liquidation|R9|cluster|flow anomal|imbalance|forced|whipsaw", state=(("event", "liquidation"),),
               require=(("event", ("liquidation", "flow_anomaly", "funding_crowded")), ("cell", ("R9",))), require_text=r"liquidation|flow anomal|taker imbalance|R9"),
    ContextDef("quoting", "Quoting / market making (atx mm)", ("crypto", "tradfi"), ("micro", "minutes", "hour"), ("quoting", "vol", "flow"),
               r"quot|market-making|market making|fill|inventory|spread|markout|simulat|replay|hour-of-week|next-minute"),
    ContextDef("stops_and_risk", "Stops, risk sigma and multi-day horizons", ("crypto", "tradfi"), ("day", "multiday"), ("execution", "vol"),
               r"stop|risk|unshrunk|5-day|multi-day|bracket|first-passage|jump", exclude=r"Moreira|variance-managed"),
    ContextDef("strategy_choice", "Choosing / combining deterministic strategies", ("crypto", "tradfi"), ("hour", "day", "multiday", "long"), ("strategy", "direction"),
               r"strateg|sleeve|book|trend|cell|regime|champion|exits|carry|parameter"),
)}
ALIASES = {
    "next-day vol": "next_day_vol", "next day vol": "next_day_vol", "daily vol": "next_day_vol", "next_day_vol_crypto": "next_day_vol",
    "next-hour vol": "next_hour_vol", "next hour vol": "next_hour_vol", "hourly vol": "next_hour_vol",
    "next-day direction": "next_day_direction", "direction": "next_day_direction", "tradfi session open": "tradfi_open",
    "tradfi open": "tradfi_open", "high vol": "high_vol", "high-vol regime": "high_vol", "liquidation regime": "liquidation",
    "market making": "quoting", "mm": "quoting", "stops": "stops_and_risk", "risk": "stops_and_risk",
}


def context_key(context: str) -> str:
    k = context.strip().lower()
    if k in CONTEXTS:
        return k
    if k in ALIASES:
        return ALIASES[k]
    k2 = re.sub(r"[\s-]+", "_", k)
    if k2 in CONTEXTS:
        return k2
    for a, v in ALIASES.items():
        if a in k:
            return v
    raise KeyError(f"unknown context {context!r}; one of {sorted(CONTEXTS)}")


# ====================================================================== state

_WD = {"mon": "mon", "monday": "mon", "tue": "tue", "tuesday": "tue", "wed": "wed", "wednesday": "wed", "thu": "thu", "thursday": "thu",
       "fri": "fri", "friday": "fri", "sat": "sat", "saturday": "sat", "sun": "sun", "sunday": "sun"}


def normalise_state(state: dict | None, ctx: ContextDef) -> dict[str, str]:
    """Flat machine-checkable state from a market_state card (nested) or a flat dict.

    weekday: explicit 'weekday' / 'forecast_weekday' (name or 1..7 / 0..6 with Monday = 1 / 0 as isoweekday / weekday), else
    from 'as_of': for day-ahead contexts the forecast day is the UTC day starting at or after as_of, otherwise as_of's day."""
    s: dict[str, str] = {}
    if not state:
        return s
    g = lambda *ks: next((state[k] for k in ks if state.get(k) is not None), None)  # noqa: E731
    reg = state.get("regime") or {}
    wd = g("forecast_weekday", "weekday")
    if wd is None and state.get("as_of"):
        t = datetime.fromisoformat(str(state["as_of"]).replace("Z", "+00:00"))
        if "day" in ctx.horizon and not ("hour" in ctx.horizon and ctx.key != "next_day_direction") and (t.hour, t.minute, t.second) != (0, 0, 0):
            t = t + timedelta(days=1)
        wd = WEEKDAYS[t.weekday()]
    if wd is not None:
        if isinstance(wd, int):
            s["weekday"] = WEEKDAYS[(wd - 1) % 7] if 1 <= wd <= 7 else WEEKDAYS[wd % 7]
        else:
            s["weekday"] = _WD.get(str(wd).strip().lower()[:9], _WD.get(str(wd).strip().lower()[:3], str(wd).lower()))
    vs = g("vol_state") or reg.get("vol_state")
    if vs:
        s["vol_state"] = str(vs).lower()
    ds = g("dir_state", "trend") or reg.get("dir_state")
    if ds:
        s["dir_state"] = str(ds).lower()
    cell = g("cell") or reg.get("cell")
    if cell:
        s["cell"] = str(cell)
    ac = g("asset_class")
    if ac:
        s["asset_class"] = "tradfi" if str(ac).lower() in ("tradfi", "stock", "etf", "commodity", "forex") else "crypto"
    for k in ("session", "event"):
        if state.get(k):
            s[k] = str(state[k]).lower()
    asym = state.get("asymmetry") or {}
    if asym.get("drawdown_flag") or state.get("drawdown_flag"):
        s.setdefault("event", "drawdown")
    fu = state.get("funding") or {}
    if fu.get("crowded"):
        s.setdefault("event", "funding_crowded")
    return s


def state_ok(f: Facets, st: dict[str, str]) -> tuple[bool, list[str]]:
    """False if a machine-checkable condition of the rule contradicts the state; also returns the matched keys."""
    matched = []
    if "asset_class" in st and st["asset_class"] not in f.assets:
        return False, []
    for k, vals in f.conds.items():
        if k in ("event", "session"):
            if k in st:
                if st[k] in vals:
                    matched.append(k)
            continue   # events / sessions: a rule about one is not excluded by the absence of a flag
        if k in st:
            if st[k] not in vals:
                return False, []
            matched.append(k)
    return True, matched


# ====================================================================== relevance and cards

def then_topics(r: dict) -> set[str]:
    t = _then_text(r)
    return {k for k, p in TOPIC_RE.items() if re.search(p, t, re.I)}


def _required(f: Facets, r: dict, ctx: ContextDef) -> bool:
    if not ctx.require and not ctx.require_text:
        return True
    for k, vals in ctx.require:
        if f.conds.get(k, set()) & set(vals):
            return True
    return bool(ctx.require_text and re.search(ctx.require_text, f"{r['rule']} {r['when']} {r['then']}", re.I))


def relevance(r: dict, f: Facets, ctx: ContextDef) -> float:
    """0 = not relevant. Gates: asset, horizon, topic of the claim (a vol card needs a vol claim or a vol tool, a direction
    card a direction claim), the context's required condition; then cue words, specificity and grade."""
    if not (f.assets & set(ctx.assets)) or not (f.horizon & set(ctx.horizon)):
        return 0.0
    txt = f"{r['rule']} {r['when']} {r['then']}"
    if ctx.exclude and re.search(ctx.exclude, txt, re.I):
        return 0.0
    tt = then_topics(r)
    prim = set(f.claims) | tt
    if not (set(ctx.topics) & prim):
        return 0.0
    want_vol, want_dir = "vol" in ctx.topics, "direction" in ctx.topics
    is_vol = "vol" in f.claims or bool(f.call or f.skip) or (r["kind"] == "prefer" and "vol" in tt)
    is_dir = "direction" in f.claims or ("direction" in tt and not is_vol)
    other = set(ctx.topics) - {"vol", "direction"}
    if not ((want_vol and is_vol) or (want_dir and is_dir) or (other & prim)):
        return 0.0
    if not _required(f, r, ctx):
        return 0.0
    sc = 1.0 + 1.0 * len(f.topics & set(ctx.topics))
    sc += 0.5 * min(len(re.findall(ctx.cues, txt, re.I)), 6)
    if f.horizon <= set(ctx.horizon):
        sc += 1.0      # horizon fully inside the context: specific to it
    elif len(f.horizon) < 5:
        sc += 0.5
    if len(f.horizon) >= 5:
        sc -= 1.0      # 'any' horizon: generic
    if f.assets == set(ctx.assets):
        sc += 0.5
    for k, v in ctx.state:
        if v in f.conds.get(k, set()):
            sc += 2.0
    sc += {"A": 1.5, "B": 0.75, "C": 0.0}[r["confidence"]]
    if r["id"].startswith(("R-SIM", "R-MS")) and ctx.key != "quoting":
        return 0.0     # simulator / tick-level rules belong to the quoting card only
    return max(sc, 0.0)


def select(context: str, state: dict | None = None, rules: list[dict] | None = None, k: int = MAX_RULES) -> dict:
    ctx = CONTEXTS[context_key(context)]
    rules = load_rules() if rules is None else rules
    st = {**dict(ctx.state), **normalise_state(state, ctx)} if state else dict(ctx.state)
    scored, dropped = [], []
    for r in rules:
        if r["status"] not in ACTIVE:
            continue
        f = facets(r)
        sc = relevance(r, f, ctx)
        if sc <= 0:
            continue
        ok, matched = state_ok(f, st)
        if not ok:
            dropped.append(r["id"])
            continue
        unknown = [k for k in f.conds if k in ("weekday", "vol_state", "dir_state", "cell") and k not in st]
        sc += 3.0 * len(matched) - 1.0 * len(unknown)   # a conditional rule whose condition can't be checked ranks lower
        scored.append((sc, r, f))
    scored.sort(key=lambda x: (-x[0], GRADES.index(x[1]["confidence"]), x[1]["id"]))
    top = scored[:k]
    top.sort(key=lambda x: (GRADES.index(x[1]["confidence"]), -x[0], x[1]["id"]))
    call: dict[str, tuple[int, int, str]] = {}
    skip: dict[str, str] = {}
    for _sc, r, f in scored[: max(k, 12)]:
        for name, stage in f.call:
            g = GRADES.index(r["confidence"])
            if name not in call or (stage, g) < call[name][:2]:
                call[name] = (stage, g, r["id"])
        for name in f.skip:
            skip.setdefault(name, r["id"])
    skip = {n: rid for n, rid in skip.items() if n not in call}
    return {"context": ctx.key, "title": ctx.title, "state": st, "rules": [r for _, r, _ in top], "scores": {r["id"]: round(s, 2) for s, r, _ in top},
            "n_matched": len(scored), "dropped_by_state": dropped,
            "call": [(n, v[2]) for n, v in sorted(call.items(), key=lambda kv: kv[1])], "skip": sorted(skip.items())}


def rules_hash(rules: list[dict] | None = None) -> str:
    rules = load_rules() if rules is None else rules
    return hashlib.sha256(json.dumps(rules, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _short(x: Any, n: int = 110) -> str:
    t = str(x)
    cut = t.split(";")[0]
    return cut if len(cut) <= n else cut[: n - 3].rstrip() + "..."


def card_for(context: str, state: dict | None = None, rules: list[dict] | None = None) -> str:
    """Compact text the decision session loads for one decision context: <= 8 rules (grade order A, B, C), tools to call
    in order, tools to skip. `state` = market_state card fields (or a flat dict: asset_class, weekday, vol_state,
    dir_state, cell, session, event); conditions that are machine-checkable filter the rules."""
    rules = load_rules() if rules is None else rules
    s = select(context, state, rules)
    st = ", ".join(f"{k}={v}" for k, v in s["state"].items()) or "none"
    L = [f"PLAYBOOK {s['context']} - {s['title']} | state: {st} | rules {rules_hash(rules)}"]
    if s["call"]:
        L.append("Tools in order: " + " -> ".join(f"{n} [{rid}]" for n, rid in s["call"]))
    if s["skip"]:
        L.append("Skip: " + "; ".join(f"{n} [{rid}]" for n, rid in s["skip"]))
    for r in s["rules"]:
        L.append(f"[{r['confidence']}] {r['id']} ({r['kind']}{', provisional' if r['status'] == 'provisional' else ''}): {r['rule']} ({_short(r['effect'])})")
    extra = s["n_matched"] - len(s["rules"])
    tail = []
    if extra > 0:
        tail.append(f"{extra} lower-relevance rules not shown")
    if s["dropped_by_state"]:
        tail.append("excluded by state: " + ", ".join(s["dropped_by_state"]))
    if tail:
        L.append("(" + "; ".join(tail) + ")")
    L.append("Rules never size positions; the risk engine does.")
    return "\n".join(L)


# ====================================================================== audit

def _conds_compatible(a: dict[str, set[str]], b: dict[str, set[str]]) -> tuple[bool, list[str]]:
    shared = [k for k in a if k in b]
    for k in shared:
        if not (a[k] & b[k]):
            return False, []
    return True, shared


def _assets_horizon_overlap(a: Facets, b: Facets) -> bool:
    return bool(a.assets & b.assets) and bool(a.horizon & b.horizon)


def _content_words(r: dict) -> set[str]:
    stop = set("the a an of to and or in on for is it its be by with as at from than that this then when not do don't no are over "
               "expect prefer avoid use any about".split())
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", f"{r['rule']}".lower()) if w not in stop}


def _effects(c: set[str]) -> set[str]:
    return c & {"up", "down", "continue", "revert"}


def _denies(nulls: set[str], effects: set[str]) -> bool:
    """A null claim denies an effect: null_up denies up, null_down denies down, a plain null denies any effect."""
    return any((n == "null_up" and "up" in effects) or (n == "null_down" and "down" in effects)
               or (n == "null" and bool(effects)) for n in nulls)


def _fmt_conds(c: dict[str, set[str]]) -> str:
    return "{" + ", ".join(f"{k}={'|'.join(sorted(v))}" for k, v in sorted(c.items())) + "}" if c else "{}"


def pair_conflicts(rules: list[dict]) -> list[dict]:
    """Pairwise flags among active / provisional rules (see the module docstring). Simulator / tick-level rules are only
    compared with each other."""
    live = [r for r in rules if r["status"] in ACTIVE]
    F = {r["id"]: facets(r) for r in live}

    def micro(rid: str) -> bool:
        return rid.startswith(("R-SIM", "R-MS"))

    out = []
    for i, ra in enumerate(live):
        for rb in live[i + 1:]:
            if micro(ra["id"]) != micro(rb["id"]):
                continue
            a, b = F[ra["id"]], F[rb["id"]]
            if not _assets_horizon_overlap(a, b):
                continue
            ok, shared = _conds_compatible(a.conds, b.conds)
            if not ok:
                continue
            common_topics = a.topics & b.topics
            wa, wb = _content_words(ra), _content_words(rb)
            jac = len(wa & wb) / max(len(wa | wb), 1)
            found = []
            trig_a = {k for k in a.conds if k in ("session", "event")}
            trig_b = {k for k in b.conds if k in ("session", "event")}
            same_trigger = trig_a == trig_b      # a rule about an expiry / a shock is not about the plain weekday
            for t in ("vol", "direction"):
                ca, cb = a.claims.get(t, set()), b.claims.get(t, set())
                if not (ca and cb and t in common_topics and shared and same_trigger):
                    continue
                opp = {"vol": ("up", "down"), "direction": ("continue", "revert")}[t]
                ea, eb = _effects(ca), _effects(cb)
                if (opp[0] in ea and opp[1] in eb and opp[1] not in ea) or (opp[1] in ea and opp[0] in eb and opp[0] not in ea):
                    found.append(("contradiction", f"{t}: {sorted(ca)} vs {sorted(cb)} under shared {shared}"))
                elif (_denies(ca - ea, eb) and SIZED.search(_then_text(rb))) or (_denies(cb - eb, ea) and SIZED.search(_then_text(ra))):
                    found.append(("negation", f"{t}: one rule denies an effect the other sizes ({sorted(ca)} vs {sorted(cb)}) under shared {shared}"))
                elif ca == cb and ea:
                    nested = any(a.conds[k] != b.conds[k] for k in shared) or set(a.conds) != set(b.conds)
                    found.append(("overlap (nested)" if nested else "duplicate",
                                  f"{t}: same claim {sorted(ca)}; conditions {_fmt_conds(a.conds)} / {_fmt_conds(b.conds)}"))
            ta, tb = {n for n, _ in a.call}, {n for n, _ in b.call}
            sa, sb = set(a.skip), set(b.skip)
            if (ta & sb) or (tb & sa):
                found.append(("call vs skip", f"one calls what the other skips: {sorted((ta & sb) | (tb & sa))}"))
            if (a.kind == "prefer" and b.kind == "prefer" and "vol" in common_topics and a.assets == b.assets and a.horizon == b.horizon
                    and "direction" not in then_topics(ra) | then_topics(rb)):
                la = {n for n, s in a.call if s in (0, 2)}
                lb = {n for n, s in b.call if s in (0, 2)}
                if la and lb and la != lb and not (la <= lb or lb <= la):
                    found.append(("competing tools", f"{sorted(la - lb)} vs {sorted(lb - la)} for the same {sorted(a.horizon)} {sorted(a.assets)} vol decision"))
            if not found and jac >= 0.45 and common_topics:
                found.append(("duplicate", f"near-identical wording (Jaccard {jac:.2f})"))
            for kind, why in found:
                out.append({"type": kind, "a": ra["id"], "b": rb["id"], "grades": f"{ra['confidence']}/{rb['confidence']}", "why": why,
                            "rule_a": ra["rule"], "rule_b": rb["rule"], "review": REVIEWED.get((ra["id"], rb["id"]), "")})
    ids = {r["id"]: r for r in live}
    have = {(p["a"], p["b"]) for p in out}
    for a_id, b_id, kind, why in CURATED_PAIRS:
        if a_id in ids and b_id in ids and (a_id, b_id) not in have:
            ra, rb = ids[a_id], ids[b_id]
            out.append({"type": kind, "a": a_id, "b": b_id, "grades": f"{ra['confidence']}/{rb['confidence']}", "why": why,
                        "rule_a": ra["rule"], "rule_b": rb["rule"], "review": "curated (found on reading; the keyword pass misses it)"})
    order = {"contradiction": 0, "negation": 1, "call vs skip": 2, "competing tools": 3, "duplicate": 4, "overlap (nested)": 5}
    out.sort(key=lambda x: (order[x["type"]], x["a"], x["b"]))
    return out


# Pairs read by hand (agent F, 2026-11-12): conflicts the keyword pass cannot see, and flags it raises that are not conflicts.
CURATED_PAIRS: list[tuple[str, str, str, str]] = [
    ("R-VOL-9", "R-DS-2", "negation", "R-VOL-9 applies 'no blanket Monday multiplier' (H-DOW2: BTC Monday 1.3x in-sample, not in the holdout); "
     "R-DS-2 sizes one for crypto next-day vol (+34% vs log-HAR: +45% in-sample z 9.8, +25% holdout z 2.4)"),
    ("R-VOL-9", "R-ML-1", "negation", "R-ML-1's central forecast multiplies log-HAR by a weekday factor with Monday about 1.3x; R-VOL-9 says "
     "no blanket Monday multiplier"),
    ("R-PX-1", "R-PX-2", "contradiction", "a |z| >= 4 hourly shock in a high-vol state is also a >= 2 sigma deviation from trend: R-PX-1 "
     "expects reversal (fade), R-PX-2 says never fade such moves; H-S3 (forward) is the arbiter"),
]
REVIEWED: dict[tuple[str, str], str] = {
    ("R-VOL-9", "R-VOL-27"): "not a conflict: R-VOL-27 is the hour-of-week profile (loud at the US open / Sunday reopen); R-VOL-9 denies "
                             "a blanket daily Monday factor, and both widen at the open",
    ("R-DS-4", "R-SP-3"): "real but narrow: both cover a Friday forecast day (R-DS-4 Tue-Fri up-trend +24%; R-SP-3 Fri-Sun calm discount); "
                          "on a calm up-trend Friday they point opposite ways",
}


def tool_clusters(pairs: list[dict]) -> list[list[str]]:
    """Connected groups of rules linked by 'competing tools' flags: one decision, several preferred tools."""
    adj: dict[str, set[str]] = {}
    for p in pairs:
        if p["type"] == "competing tools":
            adj.setdefault(p["a"], set()).add(p["b"])
            adj.setdefault(p["b"], set()).add(p["a"])
    seen: set[str] = set()
    groups = []
    for n in sorted(adj):
        if n in seen:
            continue
        stack, g = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            g.append(x)
            stack += sorted(adj[x] - seen)
        groups.append(sorted(g))
    return groups


def latest_verdicts(res: list[dict]) -> dict[str, dict[str, tuple[str, str]]]:
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for r in res:
        out.setdefault(r["id"], {})[r["segment"]] = (r["verdict"], r["ts"])
    return out


# Results that bear on a rule without being cited by it, or forward tests a rule waits for (curated from BELIEFS.md 6-9,
# the rules' review fields and the forward-test list): (rule, hypothesis, verdict under which the rule stands, why).
CROSS_CHECKS: list[tuple[str, str, str, str]] = [
    ("R-SIM-13", "H-MD6", "supports", "R-SIM-13 prefers the v1 generator; H-MD6 (per-day v1 calibration beats v0 on >= 70% of symbol-class-days) is its multi-day test"),
    ("R-SIM-11", "H-MD1", "contradicts", "R-SIM-11 says per-perp traits are persistent; H-MD1 tests their stability across symbol-days (interim ICC 0.23-0.31, BELIEFS MD1)"),
    ("R-SIM-10", "H-MD2", "supports", "R-SIM-10 extrapolates activity scaling; H-MD2 is its multi-day test"),
    ("R-MS-3", "H-MD3", "supports", "R-MS-3 power-law sign memory; H-MD3 is its multi-day test"),
    ("R-MS-11", "H-MD4", "supports", "free-spread spread -> vol lead; H-MD4 (random days) and H-MS5b (forward days) test it"),
    ("R-MS-13", "H-MD5", "supports", "R-MS-13 ETH/SOL transmit vol; H-MD5 is its multi-day test"),
    ("R-LEV-5", "H-LEV6", "supports", "R-LEV-5's 'mild upward tilt' clause rests on H-LEV6 (forward)"),
    ("R-VOL-22", "H-VM6", "supports", "R-VOL-22 review names H-VM6 (weekday-aware HAR) as its possible replacement"),
    ("R-ML-2", "H-NN7", "supports", "R-ML-2 (smaller weekend discount in high vol) is H-NN7 (forward)"),
    ("R-ML-7", "H-NN10", "supports", "R-ML-7 needs the frozen hourly tool to hold forward (H-NN10)"),
    ("R-DS-1", "H-DS4", "supports", "R-DS-1, R-DS-2, R-DS-4 cells are H-DS4 (forward)"),
    ("R-SP-2", "H-SP4", "supports", "R-SP-2 cluster-level stability is H-SP4 (forward refits)"),
    ("R-PD-7", "H-PD5", "supports", "R-PD-7's reversal mechanism is H-PD5 (forward)"),
    ("R-TF-1", "H-TP1b", "supports", "R-TF-1 review names H-TP1b (continuation, forward)"),
    ("R-FLOW-2", "H-OF7", "supports", "R-FLOW-2 (don't fade liquidation clusters) rests on H-OF7 (forward continuation)"),
    ("R-FLOW-1", "H-OF6", "supports", "R-FLOW-1 review names the selective detector H-OF6 (forward)"),
    ("R-MM-1", "H-MM2", "supports", "R-MM-1/2 rest on the live H-MM2 runs (10 run-days needed)"),
]
REGISTERED_FOR = {"R-LEV-5": "H-LEV6", "R-ML-2": "H-NN7", "R-DS-1": "H-DS4", "R-DS-2": "H-DS4", "R-DS-3": "H-DS4", "R-DS-4": "H-DS4",
                  "R-ML-1": "H-VM6", "R-ML-4": "H-NN10"}


def evidence_audit(rules: list[dict]) -> list[dict]:
    """Polarity-free checks of each rule's cited results, plus the curated cross-checks.
      segment flip           a cited hypothesis has 'supports' in one segment and 'contradicts' in the other (an A grade
                             needs both segments to agree, BELIEFS grades);
      no decisive evidence   every cited hypothesis is inconclusive or unrecorded in every segment, yet the grade is A/B;
      holdout only           an A rule none of whose cited hypotheses has a decisive verdict in llm_in_sample."""
    from ai_trader.research.hypotheses import results

    lv = latest_verdicts(results())
    out = []
    for r in rules:
        if r["status"] not in ACTIVE:
            continue
        hids = [e for e in r["evidence"] if isinstance(e, str) and e.startswith("H-")]
        flips, decisive, ins_dec = [], 0, 0
        for h in hids:
            v = {k: x[0] for k, x in lv.get(h, {}).items()}
            ins, ch = v.get("llm_in_sample"), v.get("clean_holdout")
            if {ins, ch} == {"supports", "contradicts"}:
                flips.append((h, f"{h} (in-sample {ins}, holdout {ch})"))
            if any(x in ("supports", "contradicts") for x in v.values()):
                decisive += 1
            if ins in ("supports", "contradicts"):
                ins_dec += 1
        if flips and r["confidence"] in ("A", "B"):
            out.append({"type": "segment flip", "rule": r["id"], "grade": r["confidence"], "hypothesis": ", ".join(h for h, _ in flips),
                        "detail": "; ".join(t for _, t in flips) + (" - an A grade needs both segments to agree" if r["confidence"] == "A" else "")})
        if hids and decisive == 0 and r["confidence"] in ("A", "B"):
            out.append({"type": "no decisive evidence", "rule": r["id"], "grade": r["confidence"], "hypothesis": ", ".join(hids),
                        "detail": "every cited result is inconclusive or missing"})
        if hids and r["confidence"] == "A" and ins_dec == 0:
            out.append({"type": "holdout only", "rule": r["id"], "grade": "A", "hypothesis": ", ".join(hids),
                        "detail": "no decisive in-sample verdict among the cited tests"})
    ids = {r["id"]: r for r in rules}
    for rid, h, stands, why in CROSS_CHECKS:
        r = ids.get(rid)
        if r is None or r["status"] not in ACTIVE:
            continue
        v = lv.get(h, {})
        last = max(v.values(), key=lambda x: x[1])[0] if v else None
        if last is None:
            out.append({"type": "pending cross-check", "rule": rid, "grade": r["confidence"], "hypothesis": h, "detail": f"no result yet; {why}"})
        elif last == "inconclusive":
            out.append({"type": "cross-check inconclusive", "rule": rid, "grade": r["confidence"], "hypothesis": h,
                        "detail": f"latest verdict 'inconclusive'; {why}"})
        elif last != stands:
            out.append({"type": "cross-check against", "rule": rid, "grade": r["confidence"], "hypothesis": h,
                        "detail": f"latest verdict '{last}' (the rule stands only if '{stands}'); {why}"})
    order = {"cross-check against": 0, "segment flip": 1, "no decisive evidence": 2, "holdout only": 3, "cross-check inconclusive": 4,
             "pending cross-check": 5}
    out.sort(key=lambda e: (order[e["type"]], GRADES.index(e["grade"]), e["rule"]))
    return out


def data_available(as_of: datetime | None = None) -> dict[str, int]:
    """Amounts of data the review triggers refer to (counted from the repo, read only)."""
    from ai_trader.core.config import load
    from ai_trader.research.evaluate import knowledge_cutoff

    root = load().root
    as_of = as_of or datetime.now(UTC)
    md = root / "data" / "research" / "micro_days"
    micro = sum(1 for p in md.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name)) if md.is_dir() else 0
    k = knowledge_cutoff()
    flow_start = datetime(2026, 11, 7, tzinfo=UTC)
    fwd0 = datetime(2026, 11, 12, tzinfo=UTC)
    return {"micro_days": micro, "flow_days": max((as_of - flow_start).days, 0), "holdout_days": (as_of - k).days,
            "holdout_months": int((as_of - k).days // 30), "holdout_weekends": int((as_of - k).days // 7),
            "holdout_sessions": int((as_of - k).days * 5 / 7 * 0.96), "forward_months": max((as_of - fwd0).days, 0) // 30}


def _review_need(text: str, rid: str = "") -> list[tuple[str, int, str]]:
    """(unit, n, phrase) data requirements found in a review text."""
    t = text.lower()
    micro_ctx = bool(re.search(r"trade|book|ob200|pilot|replay|batch|symbol-day|flow", t)) or rid.startswith(("R-MS", "R-SIM", "R-ML-11", "R-ML-12"))
    fwd_ctx = bool(re.search(r"forward test|after the tool exists|forward weekends", t))
    out = []
    pat = (r"(\d+)\s*(?:more |separate |random |pilot )?(symbol-days|run-days|days|holdout months|months|month|weeks|weekends|"
           r"holdout sessions|sessions|open days|monthly expiries|expiries)\b")
    for n, unit in re.findall(pat, t):
        if unit == "symbol-days" or (unit == "days" and micro_ctx):
            u = "micro_days"
        elif unit == "days":
            u = "holdout_days"
        elif unit in ("months", "month", "holdout months"):
            u = "forward_months" if fwd_ctx and unit != "holdout months" else "holdout_months"
        else:
            u = {"run-days": "run_days", "weeks": "weeks", "weekends": "holdout_weekends", "holdout sessions": "holdout_sessions",
                 "sessions": "holdout_sessions", "open days": "holdout_sessions", "expiries": "holdout_months",
                 "monthly expiries": "holdout_months"}[unit]
        out.append((u, int(n), f"{n} {unit}"))
    if re.search(r"a week of", t):
        out.append(("micro_days", 7, "a week of trades"))
    if re.search(r"a (forward )?month of flow", t):
        out.append(("flow_days", 30, "a month of flow_3s"))
    if re.search(r"several days|more days|another (venue or )?day", t) and not out:
        out.append(("micro_days", 3, "more days"))
    return out


def review_audit(rules: list[dict], fwd_status: dict | None = None, as_of: datetime | None = None) -> list[dict]:
    from ai_trader.research.hypotheses import registry, results

    reg = registry()
    lv = latest_verdicts(results())
    avail = data_available(as_of)
    fwd = {t["hypothesis"]: t for t in (fwd_status or {}).get("tests", [])}
    today = (as_of or datetime.now(UTC)).date()
    out = []

    def hyp_state(h: str) -> tuple[str, bool]:
        if h in lv:
            last = max(lv[h].values(), key=lambda x: x[1])
            return f"fired: {h} recorded '{last[0]}' ({last[1][:10]})", True
        if h in fwd:
            t = fwd[h]
            return f"waiting: {h} forward n {t.get('n')}/{t.get('need')} {t.get('unit')}", t.get("verdict") not in (None, "pending")
        if h in reg:
            return f"waiting: {h} registered, no result yet", False
        return f"{h} named but not registered", False

    for r in rules:
        if r["status"] != "provisional":
            continue
        rev = str(r["review"])
        cited = {e for e in r["evidence"] if isinstance(e, str)}
        for h in sorted(set(re.findall(r"H-[A-Z]+\d+[a-z]?", rev))):
            if h in cited:
                out.append({"rule": r["id"], "trigger": h, "state": f"{h} is the rule's own evidence (no new result is a trigger yet)", "fired": False})
                continue
            st, fired = hyp_state(h)
            out.append({"rule": r["id"], "trigger": h, "state": st, "fired": fired})
        for unit, n, phrase in _review_need(rev, r["id"]):
            have = avail.get(unit)
            if have is None:
                continue
            fired = have >= n
            note = " (random micro days exist; multi-day re-tests H-MD1..H-MD6 registered)" if fired and unit == "micro_days" else ""
            out.append({"rule": r["id"], "trigger": phrase, "state": f"{'fired' if fired else 'waiting'}: {have} {unit.replace('_', ' ')} vs {n}{note}",
                        "fired": fired})
        for d in re.findall(r"\d{4}-\d{2}-\d{2}", rev):
            if date.fromisoformat(d) <= today and not re.search(rf"after {d}", rev):
                out.append({"rule": r["id"], "trigger": f"date {d}", "state": "date passed", "fired": True})
        if re.search(r"\bregister\b", rev, re.I) and not re.findall(r"H-[A-Z]+\d+", rev):
            h = REGISTERED_FOR.get(r["id"])
            st = (f"done: registered as {h}; " + hyp_state(h)[0]) if h and h in reg else "asks for a registration: none found"
            out.append({"rule": r["id"], "trigger": "registration requested", "state": st, "fired": False})
    return out


def recommendations(pairs: list[dict], ev: list[dict], rev: list[dict]) -> list[str]:
    """Plain recommendations derived from the flags (the orchestrator decides; nothing is changed here)."""
    recs = []
    for e in ev:
        if e["type"] == "cross-check against":
            recs.append(f"{e['rule']} (grade {e['grade']}): {e['hypothesis']} {e['detail'].split(';')[0]} -> downgrade or retire unless re-confirmed.")
    for p in pairs:
        if p["type"] in ("contradiction", "negation", "call vs skip") and not p.get("review", "").startswith("not a conflict"):
            recs.append(f"Resolve {p['type']} {p['a']} vs {p['b']} ({p['grades']}): {p['why']}.")
    dup = sorted({x for p in pairs if p["type"] in ("duplicate", "overlap (nested)") for x in (p["a"], p["b"])})
    if dup:
        recs.append(f"Overlapping statements of one effect ({', '.join(dup)}): merge into one rule family with a single headline "
                    "(the A rule) and the C cells as refinements, so a card never shows the same effect three times.")
    a_flips = [e["rule"] for e in ev if e["type"] == "segment flip" and e["grade"] == "A"]
    if a_flips:
        recs.append(f"Grade A rules with segment-flipping evidence (A needs both segments to agree): {', '.join(a_flips)} -> re-grade to B, "
                    "or narrow each rule to the components that agree.")
    for g in tool_clusters(pairs):
        recs.append(f"One decision, several preferred tools ({', '.join(g)}): state the order once (the playbook card does) and mark the "
                    "superseded ones in their review field.")
    fired = sorted({x["rule"] for x in rev if x["fired"]})
    if fired:
        recs.append(f"Re-run or re-read the tests behind {len(fired)} provisional rules whose review trigger has fired: {', '.join(fired)}.")
    return recs


def audit(rules: list[dict] | None = None, fwd_status: dict | None = None, as_of: datetime | None = None) -> dict:
    rules = load_rules() if rules is None else rules
    pairs = pair_conflicts(rules)
    ev = evidence_audit(rules)
    rv = review_audit(rules, fwd_status, as_of)
    return {"as_of": (as_of or datetime.now(UTC)).isoformat(timespec="seconds"), "rules_hash": rules_hash(rules), "n_rules": len(rules),
            "pairs": pairs, "evidence": ev, "reviews": rv, "recommendations": recommendations(pairs, ev, rv)}


# ====================================================================== rendering

def render_audit(a: dict, notes: str = "") -> str:
    from collections import Counter

    c = Counter(p["type"] for p in a["pairs"])
    L = [f"# Rules audit ({a['as_of'][:10]})", "",
         f"_Generated by `scripts/playbook.py` (src/ai_trader/research/playbook.py) from research/rules.yaml ({a['n_rules']} rules, hash "
         f"{a['rules_hash']}). Read only: nothing in rules.yaml was changed; the recommendations are for the orchestrator. The pair "
         "checks are keyword heuristics, so every flag needs a human read (see the reviewer notes); curated pairs and cross-checks are marked._", ""]
    if notes:
        L += [notes, ""]
    L += ["## Summary", "", "- pair flags: " + ", ".join(f"{k} {v}" for k, v in c.items()),
          f"- evidence flags: {sum(1 for e in a['evidence'] if e['type'] not in ('pending cross-check',))} "
          f"(+ {sum(1 for e in a['evidence'] if e['type'] == 'pending cross-check')} pending cross-checks)",
          f"- provisional rules with a fired review trigger: {len({x['rule'] for x in a['reviews'] if x['fired']})}", "",
          "## Recommendations", ""]
    L += [f"{i}. {x}" for i, x in enumerate(a["recommendations"], 1)] or ["none"]
    L += ["", "## Pair flags", "", "| type | rules | grades | why |", "|---|---|---|---|"]
    L += [f"| {p['type']} | {p['a']} / {p['b']} | {p['grades']} | {p['why']} |" for p in a["pairs"]]
    L += ["", "## Evidence flags", "", "| type | rule | grade | hypothesis | detail |", "|---|---|---|---|---|"]
    L += [f"| {e['type']} | {e['rule']} | {e['grade']} | {e['hypothesis']} | {e['detail']} |" for e in a["evidence"]]
    L += ["", "## Review triggers (provisional rules)", "", "| rule | trigger | state |", "|---|---|---|"]
    L += [f"| {x['rule']} | {x['trigger']} | {x['state']} |" for x in sorted(a["reviews"], key=lambda x: (not x["fired"], x["rule"]))]
    return "\n".join(L) + "\n"


def playbook(rules: list[dict] | None = None) -> dict:
    rules = load_rules() if rules is None else rules
    out = {"version": VERSION, "rules_hash": rules_hash(rules), "generated": datetime.now(UTC).isoformat(timespec="seconds"), "contexts": {}}
    for k, c in CONTEXTS.items():
        s = select(k, None, rules)
        out["contexts"][k] = {"title": c.title, "implied_state": dict(c.state), "rules": [r["id"] for r in s["rules"]],
                              "grades": [r["confidence"] for r in s["rules"]], "call": s["call"], "skip": s["skip"], "n_matched": s["n_matched"],
                              "card": card_for(k, None, rules)}
    return out


def render_playbook(pb: dict, examples: dict[str, str] | None = None) -> str:
    L = ["# Decision playbook", "",
         f"_Generated by `scripts/playbook.py` from [rules.yaml](rules.yaml) (hash {pb['rules_hash']}); do not edit by hand. One card per "
         "decision context: the <= 8 most relevant active / provisional rules in grade order (R-BUDGET-1: short cards), tools to call "
         "in order and tools to skip. At decision time `ai_trader.research.playbook.card_for(context, state)` re-selects with the "
         "market-state card, dropping rules whose machine-checkable condition (weekday, vol state, v0 trend, cell, asset class) "
         "does not hold. Rules never size positions._", ""]
    for k, c in pb["contexts"].items():
        L += [f"## {c['title']} (`{k}`)", "", "```text", c["card"], "```", ""]
    if examples:
        L += ["## Example: state-conditioned cards", ""]
        for title, card in examples.items():
            L += [f"### {title}", "", "```text", card, "```", ""]
    return "\n".join(L)


__all__ = ["CONTEXTS", "CROSS_CHECKS", "MAX_RULES", "audit", "card_for", "facets", "pair_conflicts", "playbook", "render_audit",
           "render_playbook", "select"]
