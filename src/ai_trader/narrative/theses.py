"""Thesis ledger: narrative theses with observable proxies, triggers, invalidations and forward-only scoring.

Definitions live in `research/desk/theses.yaml` (human / agent authored; never rewritten by code). Each desk cycle
`scripts/thesis_check.py` evaluates every thesis point in time at the snapshot's as_of and writes
`<snapshot>/theses.json` + `theses.md` (<= 30 lines); the running state (trigger time, sticky statuses, scores) is in
`data/journal/theses/state.json` and every cycle is appended to `data/journal/theses/history.jsonl`.

Leaf condition (a proxy, a trigger or an invalidation part):
    {series: "fred:DGS2" | "curve:regime" | "lake:NVDAUSDT", vs: <series, optional>, transform: <name>, op: <op>, value: <x>}
  transforms: level, chg_<N> (difference over N observations), logret_<N> (100 ln ratio over N obs),
              rel_logret_<N> (logret_N(series) - logret_N(vs)), spread (series - vs), spread_chg_<N>,
              pct_1y (percentile of the last value among the last 252 daily / 12 monthly obs), minus_median_1y,
              dd_from_high (% below the running maximum), yoy_pct (monthly, matched by calendar month),
              yoy_falling_3m (1 if the last 3 yoy changes are all negative, else 0), regime (curve only)
  ops: > >= < <= == in
Trees: {any: [...]} / {all: [...]}. A leaf with missing data evaluates to None; `any` is True if any child is True,
`all` is False if any child is False, otherwise None propagates (unknown is never treated as met).

Status per cycle (deterministic, in this order):
  1. invalidated / expired are sticky;
  2. invalidation true -> invalidated;
  3. trigger: `on_create` starts the clock at created_ts; a condition starts it at the first cycle it is true;
  4. clock started and as_of >= start + horizon -> expired; never started and as_of >= created + max_wait -> expired;
  5. otherwise net = mean(+1 supporting / -1 opposing proxies, unknown skipped): strengthening if net >= 0.5,
     weakening if net < 0, else active.
Scoring (forward only, never backtested): the realised log return of each asset from the start to start + horizon,
signed by the implied direction (neutral: hit if |r| < sigma_1h * sqrt(hours)); scored at the horizon even if
the thesis was invalidated earlier (so an invalidation cannot hide a loss); Brier only when `probability` is set.
"""

from __future__ import annotations

import json
import math
import operator
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from ai_trader.core.config import load

AUTHORS = ("user", "news_agent", "claude")
DIRECTIONS = ("long", "short", "neutral")
STATUSES = ("active", "strengthening", "weakening", "invalidated", "expired")
OPS: dict[str, Any] = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq,
                       "in": lambda a, b: a in b}
TRANSFORM_RE = re.compile(r"^(level|pct_1y|minus_median_1y|dd_from_high|yoy_pct|yoy_falling_3m|regime|spread|"
                          r"(chg|logret|rel_logret|spread_chg)_\d+)$")
REQUIRED = ("id", "component", "author", "created_ts", "claim", "mechanism", "direction", "assets", "proxies",
            "trigger", "invalidation", "horizon_days", "status")
SIZE_KEYS = {"qty", "size", "notional", "leverage", "risk_usd", "position_size"}


def ledger_path() -> Path:
    return load().root / "research" / "desk" / "theses.yaml"


def state_dir() -> Path:
    p = load().root / "data" / "journal" / "theses"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_ledger(p: Path | None = None) -> list[dict[str, Any]]:
    return yaml.safe_load((p or ledger_path()).read_text(encoding="utf-8")).get("theses", [])


def _ts(x: Any) -> datetime:
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(UTC)


# ====================================================================== validation

def _leaves(c: Any) -> list[dict]:
    if isinstance(c, dict) and ("any" in c or "all" in c):
        return [x for k in ("any", "all") for ch in c.get(k, []) for x in _leaves(ch)]
    return [c] if isinstance(c, dict) else []


def _check_leaf(tid: str, where: str, lf: dict, errs: list[str]) -> None:
    from ai_trader.narrative.macro import SERIES

    for k in ("series", "transform", "op", "value"):
        if k not in lf:
            errs.append(f"{tid} {where}: leaf missing {k!r}")
            return
    for key in ("series", "vs"):
        s = lf.get(key)
        if s is None:
            continue
        kind, _, name = str(s).partition(":")
        if kind == "fred" and name not in SERIES:
            errs.append(f"{tid} {where}: unknown FRED series {name}")
        elif kind == "lake" and (not name.endswith("USDT") or name == "SPYUSDT"):
            errs.append(f"{tid} {where}: lake series must be a USDT perp and never SPYUSDT ({name})")
        elif kind == "curve" and name != "regime":
            errs.append(f"{tid} {where}: curve series is curve:regime")
        elif kind not in ("fred", "lake", "curve"):
            errs.append(f"{tid} {where}: series {s!r} must start with fred:, lake: or curve:")
    if not TRANSFORM_RE.match(str(lf["transform"])):
        errs.append(f"{tid} {where}: unknown transform {lf['transform']!r}")
    if lf["op"] not in OPS:
        errs.append(f"{tid} {where}: op {lf['op']!r} not in {list(OPS)}")
    if str(lf["transform"]).startswith(("rel_logret", "spread")) and "vs" not in lf:
        errs.append(f"{tid} {where}: {lf['transform']} needs 'vs'")
    if (lf["transform"] == "regime") != (str(lf["series"]) == "curve:regime"):
        errs.append(f"{tid} {where}: transform regime goes with series curve:regime only")


def validate(theses: list[dict[str, Any]]) -> list[str]:
    errs: list[str] = []
    seen: set[str] = set()
    for t in theses:
        tid = t.get("id", "?")
        miss = [k for k in REQUIRED if k not in t]
        if miss:
            errs.append(f"{tid}: missing {miss}")
            continue
        if tid in seen:
            errs.append(f"{tid}: duplicate id")
        seen.add(tid)
        if t["author"] not in AUTHORS:
            errs.append(f"{tid}: author {t['author']!r} not in {AUTHORS}")
        if t["direction"] not in DIRECTIONS:
            errs.append(f"{tid}: direction {t['direction']!r} not in {DIRECTIONS}")
        if t["status"] not in STATUSES:
            errs.append(f"{tid}: status {t['status']!r} not in {STATUSES}")
        try:
            _ts(t["created_ts"])
        except ValueError:
            errs.append(f"{tid}: bad created_ts")
        if not t["assets"] or any(a == "SPYUSDT" or not str(a).endswith("USDT") for a in t["assets"]):
            errs.append(f"{tid}: assets must be USDT perps, never SPYUSDT")
        if not isinstance(t["horizon_days"], (int, float)) or t["horizon_days"] <= 0:
            errs.append(f"{tid}: horizon_days must be > 0")
        p = t.get("probability")
        if p is not None and not (0 <= float(p) <= 1):
            errs.append(f"{tid}: probability must be in [0, 1]")
        if not t["proxies"]:
            errs.append(f"{tid}: at least one proxy")
        ids = set()
        for px in t["proxies"]:
            if "id" not in px:
                errs.append(f"{tid}: proxy without id")
                continue
            ids.add(px["id"])
            for lf in _leaves({k: v for k, v in px.items() if k in ("any", "all")} if ("any" in px or "all" in px) else px):
                _check_leaf(tid, f"proxy {px['id']}", lf, errs)
        trig = t["trigger"]
        if trig != "on_create":
            for lf in _leaves(trig):
                _check_leaf(tid, "trigger", lf, errs)
            if "max_wait_days" not in t:
                errs.append(f"{tid}: a conditional trigger needs max_wait_days")
        for lf in _leaves(t["invalidation"]):
            _check_leaf(tid, "invalidation", lf, errs)
        bad = SIZE_KEYS & set(_walk_keys(t))
        if bad:
            errs.append(f"{tid}: size keys {sorted(bad)} (the risk engine sizes)")
    return errs


def _walk_keys(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [str(k) for k in x] + [k for v in x.values() for k in _walk_keys(v)]
    if isinstance(x, list):
        return [k for v in x for k in _walk_keys(v)]
    return []


# ====================================================================== data access (point in time)

class Data:
    """Caches the point-in-time series a cycle needs. Every read is filtered to known_ts <= as_of."""

    def __init__(self, as_of: datetime, lake: Any = None):
        self.as_of = as_of
        self._lake = lake
        self._cache: dict[str, pl.DataFrame] = {}

    def lake(self) -> Any:
        if self._lake is None:
            from ai_trader.storage.lake import Lake
            self._lake = Lake()
        return self._lake

    def series(self, s: str) -> pl.DataFrame:
        """obs_date, value (or regime for curve:regime)."""
        if s in self._cache:
            return self._cache[s]
        kind, _, name = s.partition(":")
        if kind == "fred":
            from ai_trader.narrative.macro import read_asof
            df = read_asof(name, self.as_of).select("obs_date", "value")
        elif kind == "curve":
            from ai_trader.narrative.curve import regimes_asof
            df = regimes_asof(self.as_of).select("obs_date", pl.col("regime").alias("value"))
        elif kind == "lake":
            b = self.lake().bars(name, "1h", self.as_of, start=self.as_of - timedelta(days=400)).select("event_ts", "close").sort("event_ts")
            b = b.filter(pl.col("event_ts") + timedelta(hours=1) <= self.as_of)
            df = (b.with_columns(pl.col("event_ts").dt.convert_time_zone("UTC").dt.date().alias("obs_date")).group_by("obs_date")
                  .agg(pl.col("close").last().alias("value")).sort("obs_date").filter(pl.col("obs_date") < self.as_of.date()))
        else:
            raise ValueError(s)
        self._cache[s] = df
        return df

    def price_at(self, sym: str, ts: datetime) -> float | None:
        """Last 1h close of a bar that closed at or before ts (and before as_of)."""
        t = min(ts, self.as_of)
        b = self.lake().bars(sym, "1h", self.as_of, start=t - timedelta(days=3)).select("event_ts", "close").sort("event_ts")
        b = b.filter(pl.col("event_ts") + timedelta(hours=1) <= t)
        return float(b["close"][-1]) if b.height else None

    def sigma_1h(self, sym: str, ts: datetime, hours: int = 720) -> float | None:
        b = self.lake().bars(sym, "1h", self.as_of, start=ts - timedelta(hours=hours + 2)).select("event_ts", "close").sort("event_ts")
        b = b.filter(pl.col("event_ts") + timedelta(hours=1) <= ts)
        c = b["close"].to_numpy()
        if len(c) < 48:
            return None
        return float(np.std(np.diff(np.log(c)), ddof=1))


# ====================================================================== transforms and conditions

def _yoy(df: pl.DataFrame) -> pl.DataFrame:
    m = df.with_columns(pl.col("obs_date").dt.offset_by("12mo").alias("k"))
    j = df.join(m.select(pl.col("k").alias("obs_date"), pl.col("value").alias("prev")), on="obs_date", how="inner")
    return j.with_columns((100 * (pl.col("value") / pl.col("prev") - 1)).alias("value")).select("obs_date", "value").sort("obs_date")


def transform(data: Data, leaf: dict) -> tuple[Any, str | None]:
    """(value, obs_date of the last observation used) or (None, None) when there is not enough data."""
    tr = str(leaf["transform"])
    df = data.series(leaf["series"])
    if df.height == 0:
        return None, None
    last_d = str(df["obs_date"][-1])
    if tr == "regime":
        return df["value"][-1], last_d
    v = df["value"].to_numpy().astype(float)
    m = re.match(r"^(chg|logret|rel_logret|spread_chg)_(\d+)$", tr)
    if tr in ("spread",) or (m and m.group(1) in ("rel_logret", "spread_chg")):
        other = data.series(leaf["vs"])
        j = df.join(other.select("obs_date", pl.col("value").alias("o")), on="obs_date", how="inner").sort("obs_date")
        if j.height == 0:
            return None, None
        a, b = j["value"].to_numpy().astype(float), j["o"].to_numpy().astype(float)
        last_d = str(j["obs_date"][-1])
        if tr == "spread":
            return float(a[-1] - b[-1]), last_d
        n = int(m.group(2))
        if len(a) <= n:
            return None, None
        if m.group(1) == "spread_chg":
            return float((a[-1] - b[-1]) - (a[-1 - n] - b[-1 - n])), last_d
        return float(100 * (math.log(a[-1] / a[-1 - n]) - math.log(b[-1] / b[-1 - n]))), last_d
    if m:
        n = int(m.group(2))
        if len(v) <= n:
            return None, None
        return (float(v[-1] - v[-1 - n]) if m.group(1) == "chg" else float(100 * math.log(v[-1] / v[-1 - n]))), last_d
    monthly = len(v) > 1 and (df["obs_date"][-1] - df["obs_date"][-2]).days > 20
    win = v[-12:] if monthly else v[-252:]
    if tr == "level":
        return float(v[-1]), last_d
    if tr == "pct_1y":
        return float((win <= v[-1]).mean()), last_d
    if tr == "minus_median_1y":
        return float(v[-1] - np.median(win)), last_d
    if tr == "dd_from_high":
        return float(100 * (v[-1] / np.max(v) - 1)), last_d
    if tr in ("yoy_pct", "yoy_falling_3m"):
        y = _yoy(df)
        if y.height < 4:
            return None, None
        yv = y["value"].to_numpy()
        if tr == "yoy_pct":
            return float(yv[-1]), str(y["obs_date"][-1])
        return float(np.all(np.diff(yv[-4:]) < 0)), str(y["obs_date"][-1])
    raise ValueError(tr)


def _describe(leaf: dict) -> str:
    vs = f" vs {leaf['vs']}" if leaf.get("vs") else ""
    return f"{leaf['transform']}({leaf['series']}{vs}) {leaf['op']} {leaf['value']}"


def evaluate(data: Data, cond: Any) -> tuple[bool | None, list[dict]]:
    """Three-valued evaluation of a condition tree; returns (result, leaf details)."""
    if cond == "on_create":
        return True, []
    if isinstance(cond, dict) and ("any" in cond or "all" in cond):
        key = "any" if "any" in cond else "all"
        res, det = [], []
        for ch in cond[key]:
            r, d = evaluate(data, ch)
            res.append(r)
            det += d
        if key == "any":
            out = True if any(r is True for r in res) else (False if all(r is False for r in res) else None)
        else:
            out = False if any(r is False for r in res) else (True if all(r is True for r in res) else None)
        return out, det
    try:
        val, d = transform(data, cond)
    except (FileNotFoundError, ValueError, KeyError) as e:
        return None, [{"cond": _describe(cond), "value": None, "error": str(e)[:120]}]
    if val is None:
        return None, [{"cond": _describe(cond), "value": None, "obs_date": None}]
    ok = bool(OPS[cond["op"]](val, cond["value"]))
    return ok, [{"cond": _describe(cond), "value": val if not isinstance(val, float) else round(val, 4), "obs_date": d, "met": ok}]


# ====================================================================== status and scoring

def score(data: Data, t: dict, start: datetime, end: datetime) -> dict[str, Any]:
    """Signed realised log returns per asset over [start, min(end, as_of)]."""
    out: dict[str, Any] = {"start": start.isoformat(), "end": end.isoformat(), "final": data.as_of >= end, "assets": {}}
    sign = {"long": 1, "short": -1, "neutral": 0}[t["direction"]]
    hits = []
    for a in t["assets"]:
        p0, p1 = data.price_at(a, start), data.price_at(a, min(end, data.as_of))
        if not p0 or not p1:
            out["assets"][a] = {"error": "no price"}
            continue
        r = 100 * math.log(p1 / p0)
        rec: dict[str, Any] = {"p0": p0, "p1": p1, "ret_pct": round(r, 3)}
        if sign:
            rec["signed_ret_pct"] = round(sign * r, 3)
            hit = sign * r > 0
        else:
            s = data.sigma_1h(a, start)
            hrs = max((min(end, data.as_of) - start).total_seconds() / 3600, 1)
            band = 100 * s * math.sqrt(hrs) if s else None
            rec["neutral_band_pct"] = band
            hit = band is not None and abs(r) < band
        rec["hit"] = bool(hit)
        hits.append(hit)
        out["assets"][a] = rec
    if hits:
        out["hit_rate"] = float(np.mean(hits))
        if t.get("probability") is not None and out["final"]:
            p = float(t["probability"])
            out["brier"] = float(np.mean([(p - float(h)) ** 2 for h in hits]))
    return out


def step(data: Data, t: dict, prev: dict | None) -> dict[str, Any]:
    """One cycle for one thesis; returns the new state record."""
    as_of = data.as_of
    created = _ts(t["created_ts"])
    st = dict(prev or {})
    st.setdefault("status", t["status"])
    st.setdefault("start_ts", created.isoformat() if t["trigger"] == "on_create" else None)
    horizon = timedelta(days=float(t["horizon_days"]))
    rec: dict[str, Any] = {"id": t["id"], "component": t["component"], "author": t["author"], "direction": t["direction"],
                           "assets": t["assets"], "prev_status": st["status"], "as_of": as_of.isoformat()}
    if as_of < created:
        return rec | {"status": "not_created_yet"}
    px_res = []
    for px in t["proxies"]:
        cond = {k: v for k, v in px.items() if k in ("any", "all")} if ("any" in px or "all" in px) else px
        r, det = evaluate(data, cond)
        px_res.append({"id": px["id"], "supports": r, "detail": det})
    known = [1 if p["supports"] else -1 for p in px_res if p["supports"] is not None]
    net = float(np.mean(known)) if known else None
    inv, inv_det = evaluate(data, t["invalidation"])
    trig, trig_det = (True, []) if t["trigger"] == "on_create" else evaluate(data, t["trigger"])
    status = st["status"]
    if status not in ("invalidated", "expired"):
        if st.get("start_ts") is None and trig is True:
            st["start_ts"] = as_of.isoformat()
        start = _ts(st["start_ts"]) if st.get("start_ts") else None
        if inv is True and start is not None:
            status, st["invalidated_ts"] = "invalidated", as_of.isoformat()
        elif inv is True and start is None:
            # a conditional thesis whose invalidation fires before its trigger: record, keep waiting
            status = "weakening"
        elif start is not None and as_of >= start + horizon:
            status = "expired"
        elif start is None and as_of >= created + timedelta(days=float(t.get("max_wait_days", t["horizon_days"]))):
            status, st["expired_untriggered"] = "expired", True
        elif net is None:
            status = "active"
        else:
            status = "strengthening" if net >= 0.5 else "weakening" if net < 0 else "active"
    st["status"] = status
    start = _ts(st["start_ts"]) if st.get("start_ts") else None
    sc = score(data, t, start, start + horizon) if start is not None else None
    if sc is not None:
        st["score"] = sc
    return rec | {"status": status, "net_support": net, "triggered": start is not None, "start_ts": st.get("start_ts"),
                  "horizon_end": (start + horizon).isoformat() if start else None, "trigger_now": trig, "trigger_detail": trig_det,
                  "invalidation_now": inv, "invalidation_detail": inv_det, "proxies": px_res, "score": sc,
                  "_state": {k: v for k, v in st.items() if k != "score"} | ({"score": sc} if sc else {})}


def check(as_of: datetime, theses: list[dict] | None = None, state: dict | None = None, lake: Any = None) -> dict[str, Any]:
    theses = load_ledger() if theses is None else theses
    errs = validate(theses)
    if errs:
        raise ValueError("theses.yaml invalid:\n  " + "\n  ".join(errs))
    data = Data(as_of, lake)
    state = state or {}
    rows = [step(data, t, state.get(t["id"])) for t in theses]
    return {"contract": "theses_check_v1", "as_of": as_of.isoformat(), "ledger": str(ledger_path().relative_to(load().root)),
            "theses": rows}


def digest(res: dict[str, Any], max_lines: int = 30) -> str:
    L = [f"# Thesis ledger check (as_of {res['as_of'][:16]} UTC)",
         "_Narrative context, not an entry signal: no thesis has a validated edge (H-MAC1, H-MAC2 inconclusive). Numbers from FRED / lake, point in time._"]
    pending = [r["id"] for r in res["theses"] if r.get("status") == "not_created_yet"]
    if pending:
        L.append(f"- not yet in force at this as_of (created later): {', '.join(pending)}")
    for r in res["theses"]:
        if r.get("status") == "not_created_yet":
            continue
        net = "n/a" if r.get("net_support") is None else f"{r['net_support']:+.2f}"
        sup = sum(1 for p in r["proxies"] if p["supports"] is True)
        tot = sum(1 for p in r["proxies"] if p["supports"] is not None)
        trig = "triggered" if r["triggered"] else "waiting for trigger"
        L.append(f"- **{r['id']}** ({r['author']}, {r['direction']} {'/'.join(a.replace('USDT', '') for a in r['assets'])}): "
                 f"{r['prev_status']} -> **{r['status']}**; support {sup}/{tot} (net {net}); {trig}; invalidation {r['invalidation_now']}")
        keyv = []
        for p in r["proxies"][:6]:
            d = p["detail"][0] if p["detail"] else {}
            v = d.get("value")
            keyv.append(f"{p['id']}={v if not isinstance(v, float) else round(v, 3)}{'+' if p['supports'] else '-' if p['supports'] is False else '?'}")
        L.append("  " + ", ".join(keyv))
        if r.get("score"):
            s = r["score"]
            sr = ", ".join(f"{a.replace('USDT', '')} {v.get('ret_pct', '?')}%" for a, v in s["assets"].items())
            L.append(f"  score since {s['start'][:16]}: {sr} ({'final' if s['final'] else 'mark-to-market'})")
    return "\n".join(L[:max_lines]) + "\n"


def load_state() -> dict[str, Any]:
    p = state_dir() / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_state(res: dict[str, Any]) -> None:
    st = load_state()
    for r in res["theses"]:
        if "_state" in r:
            st[r["id"]] = r["_state"]
    p = state_dir() / "state.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    with (state_dir() / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"as_of": res["as_of"], "theses": [{k: r.get(k) for k in ("id", "status", "net_support", "triggered", "invalidation_now")}
                                                              for r in res["theses"]]}, default=str) + "\n")


def public(res: dict[str, Any]) -> dict[str, Any]:
    return res | {"theses": [{k: v for k, v in r.items() if k != "_state"} for r in res["theses"]]}


def as_date(x: str) -> date:
    return date.fromisoformat(x[:10])
