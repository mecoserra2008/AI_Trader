"""Read-only access to the live journals and the /api/state payload.

Sources (never written here):
  data/journal/live_nn/<date>.jsonl       v1 lines (id nn-<S>), scores.jsonl (realised outcomes of closed windows)
  data/journal/live_nn_v2/<date>.jsonl    v2 shadow lines (id v2-<S>), if the directory exists
  data/journal/live_claude/<date>.jsonl   Claude's forecasts (`view` = its interpretation)
  research/live/LATEST.md                 parsed for the 'Regime and players' table, if present
  data/journal/live_xs_v3/<date>.jsonl    cross-sectional v3 lines (id xs3-<S>; context only, not a signal), scores.jsonl
Files are re-read only when their size / mtime changes.

H-LIVE running scores reuse ai_trader.live.score (pairs / live1 / live2) with realised returns taken from the network's
own scored windows (scores.jsonl, the same lake closes the scorer uses), else from consecutive journal closes (each
v1 line stores the close of the bar that closes at its forecast hour).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

H_DELTA = {"1h": timedelta(hours=1), "24h": timedelta(hours=24)}
HISTORY = 72
REALISED_N = 48
WEEK = timedelta(hours=168)


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


@dataclass
class _Cached:
    sig: tuple = ()
    rows: list = field(default_factory=list)


class Store:
    def __init__(self, root: Path | str, models_root: Path | str | None = None):
        self.root = Path(root)
        self.models_root = Path(models_root) if models_root else self.root
        self._c: dict[str, _Cached] = {}

    # ------------------------------------------------------------------ paths
    @property
    def v1_dir(self) -> Path:
        return self.root / "data" / "journal" / "live_nn"

    @property
    def v2_dir(self) -> Path:
        return self.root / "data" / "journal" / "live_nn_v2"

    @property
    def claude_dir(self) -> Path:
        return self.root / "data" / "journal" / "live_claude"

    @property
    def xs_dir(self) -> Path:
        return self.root / "data" / "journal" / "live_xs_v3"

    @property
    def latest_md(self) -> Path:
        return self.root / "research" / "live" / "LATEST.md"

    @property
    def pnn_dir(self) -> Path:
        return self.models_root / "data" / "models" / "pnn"

    # ------------------------------------------------------------------ readers
    def _read(self, key: str, files: list[Path]) -> list[dict]:
        sig = tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.is_file())
        c = self._c.get(key)
        if c is not None and c.sig == sig:
            return c.rows
        rows = []
        for f in files:
            if not f.is_file():
                continue
            for x in f.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    try:
                        rows.append(json.loads(x))
                    except json.JSONDecodeError:   # a line being appended right now
                        continue
        self._c[key] = _Cached(sig, rows)
        return rows

    def _lines(self, key: str, d: Path) -> list[dict]:
        files = sorted(d.glob("20*.jsonl")) if d.is_dir() else []
        rows = self._read(key, files)
        return sorted(rows, key=lambda r: (r.get("forecast_hour", ""), r.get("logged_at", "")))

    def v1(self) -> list[dict]:
        return self._lines("v1", self.v1_dir)

    def v2(self) -> list[dict]:
        return self._lines("v2", self.v2_dir)

    def scores(self, version: str = "v1") -> list[dict]:
        d = self.v1_dir if version == "v1" else self.v2_dir
        return self._read(f"scores_{version}", [d / "scores.jsonl"])

    def xs_lines(self, days: int = 3) -> list[dict]:
        """v3 lines of the newest `days` journal files (the panel needs the newest line and a short history)."""
        files = sorted(self.xs_dir.glob("20*.jsonl"))[-days:] if self.xs_dir.is_dir() else []
        rows = self._read("xs", files)
        return sorted(rows, key=lambda r: (r.get("forecast_hour", ""), r.get("logged_at", "")))

    def xs(self, now: datetime | None = None) -> dict:
        """/api/xs: the newest v3 line (context only, not a signal: R-XS-2 / R-XS-3) and its recent scored history."""
        now = now or datetime.now(UTC)
        L = self.xs_lines()
        out: dict = {"now": now.isoformat(), "line": None, "n_lines": len(L), "scores": [], "available": bool(L),
                     "note": "context only, not a signal (R-XS-2 / R-XS-3); the decision sessions do not read it"}
        if not L:
            return out
        cur = L[-1]
        out["line"] = cur
        out["age_min"] = (now - _utc(cur["logged_at"])).total_seconds() / 60
        sc = self._read("xs_scores", [self.xs_dir / "scores.jsonl"])
        keep = []
        for s in sc[-400:]:
            r, v = s.get("rel"), s.get("vol")
            keep.append({"id": s.get("id"), "horizon": s.get("horizon"), "forecast_hour": s.get("forecast_hour"), "origin": s.get("origin"),
                         "ic_v3": r.get("ic_v3") if r else None, "ic_lowvol": r.get("ic_lowvol") if r else None,
                         "hit_aci80": r.get("hit_aci80") if r else None, "pinball_v3": v.get("pinball_v3") if v else None,
                         "pinball_har": v.get("pinball_har") if v else None})
        out["scores"] = keep
        return out

    def claude(self) -> list[dict]:
        files = sorted(self.claude_dir.glob("*.jsonl")) if self.claude_dir.is_dir() else []
        return sorted(self._read("claude", files), key=lambda r: r.get("logged_at", ""))

    def find(self, line_id: str) -> dict | None:
        for rows in (self.v1(), self.v2()):
            for r in reversed(rows):
                if r.get("id") == line_id:
                    return r
        return None

    @staticmethod
    def first_per_hour(rows: list[dict]) -> list[dict]:
        """One line per forecast hour (the first logged, not 'stale'), as the scorer pairs them."""
        seen, out = set(), []
        for r in sorted(rows, key=lambda r: (r["forecast_hour"], r["logged_at"])):
            if r["forecast_hour"] in seen:
                continue
            seen.add(r["forecast_hour"])
            out.append(r)
        return out

    def week_lines(self, line: dict) -> list[dict]:
        """v1 lines with forecast hour in (S - 168h, S] (the 'weekly mean' reference of a line)."""
        S = _utc(line["forecast_hour"])
        return [r for r in self.first_per_hour(self.v1()) if S - WEEK < _utc(r["forecast_hour"]) <= S and "stale" not in r.get("flags", [])]

    # ------------------------------------------------------------------ realised
    def realised_fn(self):
        sc = {}
        for s in self.scores("v1"):
            sc.setdefault((s["window_start"], s["horizon"]), s["r_pct"])
        closes = {}
        for r in self.v1():
            if r.get("close") is not None and r.get("last_bar_close"):
                closes.setdefault(_utc(r["last_bar_close"]), float(r["close"]))

        def realised(ws, h: str):
            ws = _utc(ws)
            for k in (ws.isoformat(), ws.isoformat().replace("+00:00", "Z")):
                if (k, h) in sc:
                    return sc[(k, h)]
            a, b = closes.get(ws), closes.get(ws + H_DELTA[h])
            return 100 * math.log(b / a) if a and b and a > 0 and b > 0 else None

        return realised

    # ------------------------------------------------------------------ H-LIVE
    def hlive(self) -> dict:
        from ai_trader.live import score

        C, N = self.claude(), self.v1()
        key = (len(C), C[-1]["id"] if C else None, len(N), N[-1]["id"] if N else None, len(self.scores("v1")))
        if getattr(self, "_hl", None) and self._hl[0] == key:
            return self._hl[1]
        try:
            P = score.pairs(C, N, self.realised_fn()) if C else []
            l1 = score.live1(P)
            l2 = score.live2(P, p_above=score.p_abs_above_sigma(self.scores("v1")))
        except Exception as e:  # noqa: BLE001 - a panel must not stop the page
            return {"error": f"{type(e).__name__}: {e}"[:200]}
        recent = [{"horizon": p["horizon"], "window_start": p["window_start"], "r": p["r"], "loss_claude": p["loss_claude"],
                   "loss_net": p["loss_net"], "d": p["d"]} for p in P[-48:]]
        out = {"pairs": len(P), "claude_forecasts": len(C), "live1": _finite(l1), "live2": _finite(l2),
               "min_live1": score.MIN_LIVE1, "min_live2": score.MIN_LIVE2, "recent": recent,
               "realised_source": "live_nn scores.jsonl (lake closes), else consecutive journal closes"}
        self._hl = (key, out)
        return out

    # ------------------------------------------------------------------ state
    def state(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        v1 = self.v1()
        out: dict = {"now": now.isoformat(), "v1": None, "v2": None, "history": [], "realised": [], "claude": {}, "hlive": {},
                     "alarms": [], "regime": None, "week": {}, "sources": {"v1_lines": len(v1), "v2_dir": self.v2_dir.is_dir()}}
        if not v1:
            return out
        cur = v1[-1]
        S = cur["forecast_hour"]
        out["v1"] = cur
        v2 = self.v2() if self.v2_dir.is_dir() else []
        out["v2"] = next((r for r in reversed(v2) if r["forecast_hour"] == S), None)
        out["v2_latest_id"] = v2[-1]["id"] if v2 else None
        out["sources"]["v2_lines"] = len(v2)
        v2_by = {r["forecast_hour"]: r for r in self.first_per_hour(v2)} if v2 else {}

        # ---- history (previous 72 lines + current)
        sc = {}
        for s in self.scores("v1"):
            sc.setdefault((s["id"], s["horizon"]), s)
        hist = [r for r in self.first_per_hour(v1) if r["forecast_hour"] <= S][-(HISTORY + 1):]
        for r in hist:
            n1 = r["net"]["1h"]
            n24 = r["net"].get("24h", {})
            s1, s24 = sc.get((r["id"], "1h")), sc.get((r["id"], "24h"))
            h2 = v2_by.get(r["forecast_hour"])
            out["history"].append({
                "id": r["id"], "forecast_hour": r["forecast_hour"], "close": r.get("close"), "flags": r.get("flags", []),
                "q1h": n1["quantiles_pct"], "sigma_1h": n1["sigma_pct"], "aci_1h": {k: [v["lo_pct"], v["hi_pct"]] for k, v in n1["aci"].items()},
                "sigma_24h": n24.get("sigma_pct"), "aci_24h": {k: [v["lo_pct"], v["hi_pct"]] for k, v in n24.get("aci", {}).items()},
                "z": r.get("latent", {}).get("z"), "zhat_next": r.get("latent", {}).get("zhat_next"),
                "r_1h": s1["r_pct"] if s1 else None, "z_1h": s1["z"] if s1 else None, "hit80_1h": s1["hit_aci80"] if s1 else None,
                "hit90_1h": s1["hit_aci90"] if s1 else None, "hit95_1h": s1["hit_aci95"] if s1 else None,
                "r_24h": s24["r_pct"] if s24 else None, "hit80_24h": s24["hit_aci80"] if s24 else None,
                "v2_band_1h": {k: [v["lo_pct"], v["hi_pct"]] for k, v in h2["net"]["1h"]["band"].items()} if h2 else None})
        lat = cur.get("latent", {})
        out["latent"] = {"zmu": lat.get("zmu"), "zsd": lat.get("zsd")}
        out["realised"] = [h for h in out["history"] if h["r_1h"] is not None][-REALISED_N:]

        # ---- weekly reference
        wk = self.week_lines(cur)
        sg = [r["net"]["1h"]["sigma_pct"] for r in wk if r["net"]["1h"].get("sigma_pct")]
        sg24 = [r["net"]["24h"]["sigma_pct"] for r in wk if r["net"].get("24h", {}).get("sigma_pct")]
        out["week"] = {"n": len(wk), "mean_sigma_1h": sum(sg) / len(sg) if sg else None, "mean_sigma_24h": sum(sg24) / len(sg24) if sg24 else None}

        # ---- Claude
        C = self.claude()
        out["claude"] = claude_block(C, S)
        out["hlive"] = self.hlive()

        # ---- alarms
        out["alarms"] = [{"source": "v1", "text": a} for a in cur.get("alarms", [])]
        if out["v2"]:
            out["alarms"] += [{"source": "v2", "text": a} for a in out["v2"].get("alarms", [])]

        # ---- regime and players (agent V2's LATEST.md section, if present)
        try:
            out["regime"] = parse_regime(self.latest_md.read_text(encoding="utf-8")) if self.latest_md.is_file() else None
        except OSError:
            out["regime"] = None
        if out["regime"] is not None:
            self._regime = (now.isoformat(), out["regime"])
        elif getattr(self, "_regime", None) and now - _utc(self._regime[0]) < timedelta(hours=2):
            # LATEST.md is overwritten each hour; between writes the section can be briefly absent
            out["regime"] = {**self._regime[1], "last_seen": self._regime[0]}
        out["latest_md_id"] = latest_md_id(self.latest_md)
        return out


def claude_block(C: list[dict], S: str) -> dict:
    """Claude's forecasts for the current windows (first per window, as scored), its newest view and the recent list."""
    first: dict = {}
    for c in C:
        first.setdefault((c["horizon"], c["window_start"]), c)
    St = _utc(S)

    def for_window(h):
        for (hh, ws), c in first.items():
            if hh == h and _utc(ws) == St:
                return c
        return None

    latest = {h: next((c for c in reversed(C) if c["horizon"] == h), None) for h in ("1h", "24h")}
    lv = next((c for c in reversed(C) if c.get("view")), None)
    keep = ("id", "logged_at", "horizon", "window_start", "window_end", "q10", "q50", "q90", "sigma", "view", "net_ref", "model_id")
    sl = lambda c: None if c is None else {k: c.get(k) for k in keep}
    return {"current_1h": sl(for_window("1h")), "current_24h": sl(for_window("24h")), "latest_1h": sl(latest["1h"]),
            "latest_24h": sl(latest["24h"]), "latest_view": sl(lv),
            "recent": [sl(c) for c in C[-24:]][::-1], "n": len(C)}


def _finite(o):
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, str):
        try:
            o = o.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def latest_md_id(p: Path) -> str | None:
    try:
        m = re.search(r"\(line (nn-[0-9T]+)\)", p.read_text(encoding="utf-8")[:400])
    except OSError:
        return None
    return m[1] if m else None


_NUM = r"[-+]?\d+(?:\.\d+)?"


def _money(s: str) -> float | None:
    m = re.match(rf"\s*({_NUM})\s*([kKmM]?)", s or "")
    if not m:
        return None
    return float(m[1]) * {"k": 1e3, "m": 1e6}.get(m[2].lower(), 1.0)


def parse_regime(md: str) -> dict | None:
    """The '## Regime and players' section of LATEST.md (live.players.render): its table as rows plus parsed numbers,
    and the bullet lines (pattern, long/short accounts). None if the section is absent."""
    m = re.search(r"^## Regime and players(.*)$", md, flags=re.MULTILINE)
    if not m:
        return None
    rest = md[m.end():]
    nxt = re.search(r"^## ", rest, flags=re.MULTILINE)
    body = rest[: nxt.start()] if nxt else rest
    out: dict = {"title": ("Regime and players" + m[1]).strip(), "rows": [], "bullets": [], "raw": body.strip()}
    if ": error" in m[1]:
        out["error"] = m[1].strip(" :")
    header = None
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if header is None:
                header = cells
                continue
            if all(set(c) <= set("-: ") for c in cells):
                continue
            row = dict(zip(header, cells, strict=False))
            sym = row.get("", cells[0])
            p = {"symbol": sym, "cells": row}
            cell = row.get("v0 cell", "")
            cm = re.match(r"(\S+)\s*(.*)", cell)
            p["cell"], p["cell_label"] = (cm[1], cm[2]) if cm else (cell, "")
            px = re.search(_NUM, row.get("24h px", "") or "")
            p["ret_24h_pct"] = float(px[0]) if px else None
            fu = row.get("funding (90d pct)", "")
            fm = re.match(rf"\s*({_NUM})%\s*\(p(\d+)\)", fu)
            p["funding_rate_pct"], p["funding_pct_90d"] = (float(fm[1]), int(fm[2])) if fm else (None, None)
            oi = re.findall(rf"({_NUM})%", row.get("OI 24h / 72h", ""))
            p["oi_chg_24h_pct"] = float(oi[0]) if oi else None
            p["oi_chg_72h_pct"] = float(oi[1]) if len(oi) > 1 else None
            for k, col in (("liq_4h", "liq 4h long/short"), ("liq_24h", "liq 24h long/short")):
                v = row.get(col, "")
                parts = v.replace("(partial)", "").split("/")
                p[f"{k}_long_usd"] = _money(parts[0]) if parts else None
                p[f"{k}_short_usd"] = _money(parts[1]) if len(parts) > 1 else None
                p[f"{k}_partial"] = "partial" in v
            dv = re.match(rf"\s*({_NUM})\s*\(p(\d+)\)", row.get("DVOL (90d pct)", ""))
            p["dvol"], p["dvol_pct_90d"] = (float(dv[1]), int(dv[2])) if dv else (None, None)
            out["rows"].append(p)
        elif s.startswith("- "):
            out["bullets"].append(s[2:])
    pat = next((b for b in out["bullets"] if b.startswith("Pattern:")), None)
    if pat:
        out["pattern"] = {k.strip(): v.strip() for k, v in (x.strip().split(" ", 1) for x in pat[len("Pattern:"):].split(";") if " " in x.strip())}
    asof = re.search(r"as_of ([0-9T:\-]+)Z", m[1])
    out["as_of"] = asof[1] if asof else None
    return out


__all__ = ["Store", "claude_block", "latest_md_id", "parse_regime"]
