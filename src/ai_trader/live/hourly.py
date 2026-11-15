"""One hourly run of the live BTC networks: v1 first (computed and written exactly as before), then the v2 shadow, the
regime-and-players block, the cross-sectional v3 context line (xs.v3, live.xs_v3) and research/live/LATEST.md. Called by
scripts/live_nn.py.

Order matters: the v1 line, its runtime, scores, ACI state and journal are produced and written before anything of v2
runs, so v1's outputs cannot depend on v2 (tests check they are byte-identical with v2 on or off); v3 runs after v1, v2.1
and players and writes only its own files (tests check v1 / v2.1 are byte-identical with v3 on or off). v2, the players
block and v3 are wrapped: a failure there is reported in LATEST.md and never stops or changes the v1 line.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trader.live import nn, nn_v2, players

COMPARE_SINCE = datetime(2026, 11, 13, 20, tzinfo=UTC)


def replay_lock(paths2: nn_v2.PathsV2) -> Path:
    return paths2.state.with_name("live_aci_state_v2.replay.lock")


def lake_realised(lake, as_of: datetime, start: datetime):
    b = lake.bars(nn.SYMBOL, "1h", as_of, start=start - timedelta(days=2))
    cl = {nn._utc(t): float(c) for t, c in zip(b["close_ts"].to_list(), b["close"].to_list(), strict=True)} if b.height else {}
    return lambda ws, h: nn.realised(cl, ws, h)


def _short_alarm(a: str) -> str:
    m = re.match(r"coverage (\w+) ACI (\d+)% over last (\d+): ([\d.]+) vs", a)
    return f"{m[1]} {m[2]}%/{m[3]} {m[4]}" if m else a


def compose_latest(v1_text: str, v2_lines: list[str], player_lines: list[str], v3_lines: list[str] | None = None) -> str:
    """v1's LATEST.md text with the alarms folded into one line, the v2 block after the v1 numbers, the players block and
    the v3 cross-section block (context only)."""
    secs = v1_text.rstrip("\n").split("\n\n## ")
    head, rest = secs[0], ["## " + s for s in secs[1:]]
    out_secs = []
    for s in rest:
        if s.startswith("## Alarms"):
            al = [x[2:] for x in s.splitlines()[1:] if x.startswith("- ")]
            if len(al) > 2 and all(x.startswith("coverage ") for x in al):
                s = f"## Alarms ({len(al)})\n- coverage (band level/window cov): " + "; ".join(_short_alarm(a) for a in al)
        out_secs.append(s)
        if s.startswith("## Next 24h"):
            out_secs += ["\n".join(v2_lines).lstrip("\n")] if v2_lines else []
            out_secs += ["\n".join(player_lines).lstrip("\n")] if player_lines else []
            out_secs += ["\n".join(v3_lines).lstrip("\n")] if v3_lines else []
    return "\n\n".join([head, *out_secs]) + "\n"


def run_hour(as_of: datetime, *, paths1: nn.Paths | None = None, paths2: nn_v2.PathsV2 | None = None, lake=None, dry_run: bool = False,
             force: bool = False, with_field: bool = True, with_news: bool = True, logged_at: datetime | None = None, v2: bool = True,
             v2_require_state: bool = False, with_players: bool = True, v3: bool = False, paths3=None, clock=time.time,
             t0: float | None = None) -> dict:
    from ai_trader.storage.lake import Lake

    t0 = clock() if t0 is None else t0
    paths1 = paths1 or nn.Paths.default()
    paths2 = paths2 or nn_v2.PathsV2(paths1.root, paths1.models_root)
    lake = lake or Lake()
    res: dict = {}

    # ---- v1, unchanged
    line, st, persist, scores = nn.run_line(as_of, paths=paths1, lake=lake, persist_state=not dry_run, with_field=with_field,
                                            with_news=with_news, logged_at=logged_at)
    line["runtime_s"] = round(clock() - t0, 1)
    if dry_run:
        prev = [r for r in nn.read_lines(paths1) if r["forecast_hour"] < line["forecast_hour"]]
        v1_text = nn.render_latest(line, prev[-1] if prev else None, scores)
    else:
        res["v1"] = nn.write_outputs(paths1, line, st, persist, scores, force=force)
        v1_text = paths1.latest_md.read_text(encoding="utf-8")
    res["line"] = line

    # ---- v2 shadow
    v2_lines: list[str] = []
    if v2:
        try:
            if replay_lock(paths2).exists():
                raise RuntimeError("v2 replay in progress (lock file); v2 skipped this hour")
            if v2_require_state and not paths2.state.is_file():
                raise RuntimeError("v2 not initialised (no state): run scripts/live_nn.py --v2-replay-from 2026-11-13T20:00Z first")
            l2, st2, p2, sc2 = nn_v2.run_line(as_of, paths=paths2, lake=lake, persist_state=not dry_run, field=line.get("field"),
                                              hnn10=line.get("hnn10"), logged_at=logged_at)
            res["line_v2"] = l2
            if not dry_run:
                res["v2"] = nn_v2.write_outputs(paths2, l2, st2, p2, sc2, force=force)
            cmp = nn_v2.compare(nn.read_lines(paths1), nn_v2.read_lines(paths2) or [l2], lake_realised(lake, nn._utc(as_of), COMPARE_SINCE),
                                since=COMPARE_SINCE)
            cmp["since"] = COMPARE_SINCE.isoformat()
            res["compare"] = cmp
            v2_lines = nn_v2.render_block(l2, cmp)
        except Exception as e:  # noqa: BLE001 - v2 must never stop v1
            res["v2_error"] = f"{type(e).__name__}: {e}"[:300]
            v2_lines = [f"\n## v2 shadow: not run ({res['v2_error'][:160]})"]

    # ---- regime and players (information only)
    pl_lines: list[str] = []
    if with_players:
        try:
            P = players.players(nn._utc(as_of), lake=lake)
            res["players"] = P
            pl_lines = players.render(P)
        except Exception as e:  # noqa: BLE001
            pl_lines = [f"\n## Regime and players: error ({type(e).__name__}: {e})"[:200]]

    # ---- cross-sectional v3 (context only, not a signal: R-XS-2 / R-XS-3); after everything above, own files only
    v3_lines: list[str] = []
    if v3:
        try:
            from ai_trader.live import xs_v3

            p3 = paths3 or xs_v3.PathsV3(paths1.root, paths1.models_root)
            l3, st3, pp3, sc3 = xs_v3.run_line(nn._utc(as_of), paths=p3, lake=lake, persist_state=not dry_run, logged_at=logged_at, v1=line)
            res["line_v3"] = l3
            if not dry_run:
                res["v3"] = xs_v3.write_outputs(p3, l3, st3, pp3, sc3, force=force)
            v3_lines = xs_v3.render_block(l3)
        except Exception as e:  # noqa: BLE001 - v3 must never stop v1 / v2.1
            res["v3_error"] = f"{type(e).__name__}: {e}"[:300]
            v3_lines = [f"\n## Cross-section v3 (context only): not run ({res['v3_error'][:160]})"]

    text = compose_latest(v1_text, v2_lines, pl_lines, v3_lines)
    res["latest"] = text
    if not dry_run:
        paths1.latest_md.write_text(text, encoding="utf-8")
    res["runtime_s"] = round(clock() - t0, 1)
    return res


def replay_v2(start: datetime, *, paths2: nn_v2.PathsV2 | None = None, lake=None, end: datetime | None = None, log=print) -> list[dict]:
    """v2 lines for every bar S from start to the newest closed hour (or end), as of S + 3 min, flagged 'replay'. Holds a
    lock so the hourly run skips v2 meanwhile; refuses if v2 already has a state or lines."""
    from ai_trader.storage.lake import Lake

    paths2 = paths2 or nn_v2.PathsV2.default()
    lake = lake or Lake()
    if paths2.state.is_file() or nn_v2.read_lines(paths2):
        raise RuntimeError("v2 already has a state or journal lines: replay refused")
    lock = replay_lock(paths2)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    out = []
    try:
        S = nn._utc(start).replace(minute=0, second=0, microsecond=0)
        while True:
            now = datetime.now(UTC)
            stop = end or (now - timedelta(minutes=3)).replace(minute=0, second=0, microsecond=0)
            if S > stop:
                break
            t = time.time()
            l2, st2, p2, sc2 = nn_v2.run_line(S + timedelta(minutes=3), paths=paths2, lake=lake, flag_replay=True)
            w = nn_v2.write_outputs(paths2, l2, st2, p2, sc2)
            out.append(l2)
            log(f"{l2['id']} {l2['flags']} floor {l2['net']['1h']['floor']['active']} scored {len(sc2)} ({time.time() - t:.1f}s) {w['journal']}")
            S += timedelta(hours=1)
    finally:
        lock.unlink(missing_ok=True)
    return out


__all__ = ["compose_latest", "replay_v2", "run_hour"]
