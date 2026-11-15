"""Live desk scorecard (agent.live_scorecard): journal parsing, fill rate, slip vs the paper model, R, track split."""

import json
from pathlib import Path

from ai_trader.agent import live_scorecard as ls

V = {"model_id": "claude-opus-5-5", "prompt_version": "v0.1", "exec_version": "live-pegged-v1", "live_profile_hash": "abc"}
DAY_MS = 86_400_000
T0 = 1_794_700_800_000  # 2026-11-15 00:00 UTC


def _w(root: Path, day: str, kind: str, rows: list[dict]) -> None:
    d = root / day
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{kind}.jsonl").open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"type": kind, **r}) + "\n")


def _trade(root: Path, tid: str, k: int, *, filled: bool = True, slip: float = 1.0, ttf_ms: int = 5_000, r: float | None = 0.5, stop_slip: float | None = None,
           tier: str = "liquid", versions: dict | None = None) -> None:
    sub = T0 + k * DAY_MS // 4
    day = "2026-11-15" if k < 4 else "2026-11-16"
    v = versions or V
    _w(root, day, "order", [{"action": "entry_done", "trade_id": tid, "symbol": "BTCUSDT", "side": "long", "tier": tier, "qty": 0.003, "filled_qty": 0.003 if filled else 0.0,
                             "reason": "filled" if filled else "window", "submitted_ms": sub, "time_to_first_fill_ms": ttf_ms if filled else None, "window_ms": 900_000,
                             "paper_slip_bps": 1.3, "slip_bps_vs_mid": slip, "versions": v}])
    if filled:
        _w(root, day, "fill", [{"leg": "entry", "trade_id": tid, "exec_qty": 0.003, "exec_ms": sub + ttf_ms, "slip_bps_vs_mid": slip, "versions": v}])
    if stop_slip is not None:
        _w(root, day, "fill", [{"leg": "exit", "trade_id": tid, "exec_qty": 0.003, "exit_kind": "StopLoss", "slip_bps_vs_stop": stop_slip, "versions": v}])
    if filled and r is not None:
        _w(root, day, "position_update", [{"action": "closed", "trade_id": tid, "realised_usd": r * 1.5, "r_net": r, "closed_ms": sub + 3_600_000, "versions": v}])


def test_parses_trades_and_scores(tmp_path: Path) -> None:
    _trade(tmp_path, "a", 0, slip=1.0, r=1.0)
    _trade(tmp_path, "b", 1, slip=2.0, r=-1.0, stop_slip=2.5)
    _trade(tmp_path, "c", 2, filled=False)
    _trade(tmp_path, "d", 5, slip=1.5, r=0.5, ttf_ms=20 * 60_000)  # filled after the 15-min window
    cards = ls.scorecard(tmp_path)
    assert len(cards) == 1
    c = cards[0]
    assert c.entries == 4 and c.filled_in_window == 2
    assert abs(c.fill_rate - 0.5) < 1e-12
    assert c.slip_n == 3 and abs(c.slip_mean_bps - 1.5) < 1e-12
    assert abs(c.slip_minus_paper_bps - 0.2) < 1e-9
    assert c.stop_exits_liquid == 1 and c.stop_slip_mean_bps == 2.5
    assert c.closed == 3 and abs(c.mean_r - (1.0 - 1.0 + 0.5) / 3) < 1e-12
    assert abs(c.total_usd - 0.75) < 1e-12
    assert c.h_lt1 == "insufficient" and c.h_lt2 == "insufficient"
    assert "no live entries" not in ls.render(cards, tmp_path)


def test_tracks_split_by_versions(tmp_path: Path) -> None:
    _trade(tmp_path, "a", 0)
    _trade(tmp_path, "b", 1, versions={**V, "prompt_version": "v0.2"})
    cards = ls.scorecard(tmp_path)
    assert len(cards) == 2, "a prompt change starts a new track record"


def test_verdicts_after_minimum_samples(tmp_path: Path) -> None:
    for i in range(ls.H_LT1_MIN):
        _trade(tmp_path, f"t{i}", i % 8, slip=1.3 + (0.2 if i % 2 else -0.2), r=0.6 if i % 3 else 0.2, stop_slip=1.0 if i % 5 == 0 else None)
    c = ls.scorecard(tmp_path)[0]
    assert c.closed == ls.H_LT1_MIN
    assert c.r_ci is not None and c.r_ci[0] > 0
    assert c.h_lt1 == "supports"
    assert c.h_lt2 == "supports", (c.fill_rate, c.slip_minus_paper_bps, c.stop_slip_mean_bps)


def test_bootstrap_and_t() -> None:
    assert ls.day_bootstrap_ci([1.0], ["d"]) is None
    lo, hi = ls.day_bootstrap_ci([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert 1.0 <= lo <= hi <= 4.0
    d, t = ls.t_stat([1.0, 2.0, 3.0], 1.0)
    assert abs(d - 1.0) < 1e-12 and t > 0
    assert ls.t_stat([1.0], 0.0) is None


def test_empty_journal(tmp_path: Path) -> None:
    assert ls.scorecard(tmp_path / "missing") == []
    assert "no live entries" in ls.render([], tmp_path)
