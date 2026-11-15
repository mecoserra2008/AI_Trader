"""Claude's own live BTC forecasts (H-LIVE1 / H-LIVE2), append-only, written BEFORE the target window opens.

    uv run python -m ai_trader.live.claude_log --horizon 1h --q10 -0.45 --q50 0.02 --q90 0.48 --sigma 0.36 \
        --view "vol stays at the network's level; CPI at 13:30 widens the right tail" [--net-ref nn-20261113T22]

    from ai_trader.live.claude_log import log_forecast
    log_forecast(as_of, "1h", q10, q50, q90, sigma, view, net_ref)

Units: % log return of BTCUSDT (Bybit linear, last price), close-to-close over the target window, i.e.
100 * (ln close(window_end) - ln close(window_start)), where close(X) is the close of the 1h bar that closes at X.
sigma = the forecast's standard-deviation-equivalent in the same % units (the network's sigma in live_nn lines is
(q90 - q10) / 2.5631, the normal-equivalent of its 80% band, so the two are comparable).

Target window: it starts at the first hour boundary strictly after the wall-clock time of logging (logged_at) and lasts
the horizon (1h or 24h). A forecast whose window has already opened is refused (ValueError): an explicit window_start
must be an hour boundary strictly after logged_at. Both the wall-clock time and the window are stored, plus:
  as_of        the information time the forecast was made from (normally the live_nn line's as_of; must be <= logged_at)
  net_ref      id of the live_nn line Claude read (data/journal/live_nn/*.jsonl, e.g. "nn-20261113T22")
  model_id     "claude-opus-5-5" (pinned; a different model is a new track record)
  prompt_version  PROMPT_VERSION (changing the reading procedure starts a new track record)
Output: data/journal/live_claude/<YYYY-MM-DD of logged_at>.jsonl, one JSON line per forecast. Never edited. The scorer
uses the first forecast per (horizon, window_start).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

MODEL_ID = "claude-opus-5-5"
PROMPT_VERSION = ("live_claude.v2 (reads research/live/LATEST.md incl. v2.1, regime-and-players and the xs.v3 context block, "
                  "+ the live_nn line; states the regime first; 10/50/90 + sigma in % log return)")
# live_claude.v1 (2026-11-13 23:17 .. 2026-11-15 14:00 UTC): LATEST.md + the live_nn line, before the v2.1 / players / v3 blocks.
SCHEMA = "live_claude.v1"
HORIZONS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24)}


def journal_dir(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    from ai_trader.core.config import load

    return load().root / "data" / "journal" / "live_claude"


def _utc(ts: datetime | str) -> datetime:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("naive datetime: pass a tz-aware UTC time")
    return ts.astimezone(UTC)


def next_hour_boundary(t: datetime) -> datetime:
    """First hour boundary strictly after t."""
    t = _utc(t)
    return t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def target_window(logged_at: datetime, horizon: str, window_start: datetime | None = None) -> tuple[datetime, datetime]:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {sorted(HORIZONS)}")
    logged_at = _utc(logged_at)
    if window_start is None:
        ws = next_hour_boundary(logged_at)
    else:
        ws = _utc(window_start)
        if ws.minute or ws.second or ws.microsecond:
            raise ValueError(f"window_start {ws.isoformat()} is not an hour boundary")
        if ws <= logged_at:
            raise ValueError(f"refused: the target window {ws.isoformat()} has already opened (logged at {logged_at.isoformat()})")
    return ws, ws + HORIZONS[horizon]


def log_forecast(as_of: datetime | str, horizon: str, q10: float, q50: float, q90: float, sigma: float, view: str,
                 net_ref: str | None, *, window_start: datetime | None = None, root: Path | None = None,
                 now: datetime | None = None) -> dict:
    """Append one forecast; returns the stored row. Raises ValueError (nothing written) if the window has opened, the
    quantiles cross, sigma <= 0, or as_of is after the wall-clock time."""
    logged_at = _utc(now) if now is not None else datetime.now(UTC)
    as_of = _utc(as_of)
    if as_of > logged_at + timedelta(seconds=1):
        raise ValueError(f"as_of {as_of.isoformat()} is after the wall-clock time {logged_at.isoformat()}")
    ws, we = target_window(logged_at, horizon, window_start)
    vals = [float(q10), float(q50), float(q90), float(sigma)]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("non-finite forecast value")
    if not (vals[0] <= vals[1] <= vals[2]):
        raise ValueError(f"quantiles cross: q10 {q10} q50 {q50} q90 {q90}")
    if vals[3] <= 0:
        raise ValueError("sigma must be > 0")
    if not str(view).strip():
        raise ValueError("view must say in words why (one line)")
    body = {"schema": SCHEMA, "logged_at": logged_at.isoformat(), "as_of": as_of.isoformat(), "symbol": "BTCUSDT", "horizon": horizon,
            "window_start": ws.isoformat(), "window_end": we.isoformat(), "units": "% log return, close-to-close over the window",
            "q10": vals[0], "q50": vals[1], "q90": vals[2], "sigma": vals[3], "view": str(view).strip(), "net_ref": net_ref,
            "model_id": MODEL_ID, "prompt_version": PROMPT_VERSION, "lead_minutes": round((ws - logged_at).total_seconds() / 60, 2)}
    body["id"] = "cl-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    d = journal_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{logged_at:%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(body) + "\n")
    return body


def read_all(root: Path | None = None) -> list[dict]:
    d = journal_dir(root)
    rows = []
    for f in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r["logged_at"])


def _latest_net_ref() -> tuple[str | None, str | None]:
    """(id, as_of) of the newest live_nn line, if any."""
    from ai_trader.core.config import load

    d = load().root / "data" / "journal" / "live_nn"
    files = sorted(f for f in d.glob("20*.jsonl")) if d.is_dir() else []
    for f in reversed(files):
        lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines:
            r = json.loads(lines[-1])
            return r.get("id"), r.get("as_of")
    return None, None


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Log Claude's BTC forecast before its window opens (H-LIVE1/2).")
    ap.add_argument("--horizon", required=True, choices=sorted(HORIZONS))
    ap.add_argument("--q10", type=float, required=True)
    ap.add_argument("--q50", type=float, required=True)
    ap.add_argument("--q90", type=float, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--net-ref", default="", help="live_nn line id read (default: the newest line)")
    ap.add_argument("--as-of", default="", help="information time (default: the referenced line's as_of, else now)")
    ap.add_argument("--window-start", default="", help="explicit window start (must be an hour boundary in the future)")
    a = ap.parse_args(argv)
    ref, ref_asof = (a.net_ref, None) if a.net_ref else _latest_net_ref()
    as_of = a.as_of or ref_asof or datetime.now(UTC).isoformat()
    try:
        row = log_forecast(as_of, a.horizon, a.q10, a.q50, a.q90, a.sigma, a.view, ref or None,
                           window_start=_utc(a.window_start) if a.window_start else None)
    except ValueError as e:
        print(f"REFUSED: {e}")
        raise SystemExit(2) from e
    print(json.dumps(row, indent=1))
    return row


if __name__ == "__main__":
    main()
