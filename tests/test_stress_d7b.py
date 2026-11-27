"""D7b (2026-11-27): the stress harness judges closed trades with a holding-matched bootstrap block and funding in net R."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import stress as S

T0 = datetime(2025, 1, 1, tzinfo=UTC)


def _trades(rows: list[dict]) -> pl.DataFrame:
    out = []
    for i, r in enumerate(rows):
        e = T0 + timedelta(hours=r.get("h", 24 * i))
        out.append({"unit": "u", "variant": "base", "symbol": r.get("symbol", "BTCUSDT"), "side": r.get("side", 1), "entry_ts": e,
                    "exit_ts": e + timedelta(hours=r.get("dur", 2)), "entry": 100.0, "stop": 99.0, "exit": r.get("exit", 101.0),
                    "why": r.get("why", "signal"), "gap": False, "kind": "peg", "ev": float("nan"), "cell": "R5", "seg": r.get("seg", "llm_in_sample"),
                    "weekend": False, "holiday": False, "funding_win": False, "tier": "liquid", **({"funding": r["funding"]} if "funding" in r else {})})
    return pl.DataFrame(out)


def test_block_is_at_least_the_p90_holding_period():
    short = _trades([{"dur": 2} for _ in range(50)])
    assert S.block_days(short) == S.BLOCK_DAYS                       # 2-hour holds: the 5-day floor
    long = _trades([{"dur": 24 * (40 if i % 5 == 0 else 10)} for i in range(50)])
    assert S.block_days(long) == pytest.approx(np.quantile(S.holding_days(long), 0.9))
    assert S.block_days(long) == pytest.approx(40.0)
    assert S.block_days(_trades([{"dur": 24 * 40}])) == S.BLOCK_DAYS   # too few trades: the floor


def test_metrics_drop_open_at_end_positions_by_default_and_report_them():
    rows = [{"exit": 100.5 + 0.1 * (i % 3)} for i in range(40)] + [{"exit": 150.0, "why": "end"}]   # one open position marked at +50
    t = _trades(rows)
    r = S.r_net(t)
    m = S.metrics(t, r)
    assert m["n"] == 40 and m["open_end"]["n"] == 1 and m["open_end"]["sum_R"] > 20
    allm = S.metrics(t, r, closed_only=False)
    assert allm["n"] == 41 and allm["mean"] > m["mean"] + 0.5
    assert np.isfinite(m["p_le0"]) and m["block_days"] == S.BLOCK_DAYS
    only_open = S.metrics(_trades([{"why": "end"}]), np.array([1.0]))
    assert only_open["n"] == 0 and only_open["open_end"]["n"] == 1


def test_mean_ci_keeps_its_signature_and_uses_the_matched_block():
    assert list(inspect.signature(S.mean_ci).parameters)[:4] == ["t", "r", "n_boot", "seed"]
    rng = np.random.default_rng(3)
    t = _trades([{"h": 24 * i, "dur": 24 * 20, "exit": 100 + float(x)} for i, x in enumerate(rng.normal(0.3, 2, 200))])
    r = S.r_net(t)
    out = S.mean_ci(t, r)
    assert len(out) == 3 and out[1] < out[0] < out[2]
    wide = S.boot_mean(t, r)                       # block ~20 days
    narrow = S.boot_mean(t, r, block=S.BLOCK_DAYS)
    assert wide["block_days"] > 15 and narrow["block_days"] == S.BLOCK_DAYS
    assert (out[1], out[2]) == (wide["lo"], wide["hi"])


def test_funding_paid_counts_settlements_in_the_open_closed_hold_interval():
    ts = np.array([0, 8, 16, 24], dtype=np.int64) * S.H_MS
    idx = S.funding_index(pl.DataFrame({"event_ts": [T0 + timedelta(hours=int(h)) for h in (0, 8, 16, 24)], "rate": [1e-4, 2e-4, 3e-4, 4e-4]}))
    assert np.array_equal(idx[0], ts + int(T0.timestamp() * 1000))
    base = int(T0.timestamp() * 1000)
    # entry exactly at the 08:00 settlement does not pay it; an exit at 16:00 does
    f = S.funding_paid(np.array([1, -1]), np.array([base + 8 * S.H_MS] * 2), np.array([base + 16 * S.H_MS] * 2), idx)
    assert f == pytest.approx([3e-4, -3e-4])
    assert np.isnan(S.funding_paid(np.array([1]), np.array([0]), np.array([1]), None)).all()


def test_r_net_subtracts_funding_when_the_column_is_present():
    plain = _trades([{"exit": 101.0}])
    paid = _trades([{"exit": 101.0, "funding": 0.002}])            # a long that paid 20 bp of funding
    r0, r1 = S.r_net(plain)[0], S.r_net(paid)[0]
    den = 1.0 + 100.0 * S.base_budget(np.array(["peg"]))[0]
    assert r1 == pytest.approx(r0 - 0.002 * 100.0 / den)
    assert S.r_net(paid, funding=False)[0] == pytest.approx(r0)
    assert S.r_net(plain, funding=np.array([0.002]))[0] == pytest.approx(r1)
    nan = _trades([{"exit": 101.0, "funding": float("nan")}])        # unknown funding counts as 0
    assert S.r_net(nan)[0] == pytest.approx(r0)


def test_engine_stats_report_open_positions_separately():
    t = _trades([{"h": 24 * i, "exit": 99.0 if i % 2 else 102.0} for i in range(30)] + [{"h": 24 * 31, "exit": 130.0, "why": "end"}])
    acc, info = S.live_engine(t, S.r_net(t), S.LiveLimits(ev_gate=False))
    es = S.engine_stats(acc, info)
    assert es["n"] == acc.height - 1 and es["n_open_end"] == 1 and es["usd_open_end"] > 0
    assert S.engine_stats(acc, info, closed_only=False)["n"] == acc.height


def _synthetic_ctx(start: datetime, days: int = 420, seed: int = 7) -> S.Ctx:
    rng = np.random.default_rng(seed)
    n = 24 * days
    r = rng.standard_t(4, n) * 0.006 + 0.0004 * np.sin(np.arange(n) / 700)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.002, n)))
    f1, f4 = S.synth_frames(o, h, lo, c, start)
    ctx = S.Ctx("SYNUSDT", f1, f4, S.Path.from_frame(f1, 60.0))
    settle = [start + timedelta(hours=8 * k) for k in range(3 * days)]
    ctx.funding = S.funding_index(pl.DataFrame({"event_ts": settle, "rate": [1e-4] * len(settle)}))
    return ctx


def test_stress_signal_verdict_on_closed_trades_with_funding(monkeypatch):
    ctx = _synthetic_ctx(datetime(2025, 10, 1, tzinfo=UTC))
    monkeypatch.setattr(S, "load_ctx", lambda *a, **k: ctx)

    def slow_trend(fr: pl.DataFrame) -> pl.Series:
        c = fr["close"].to_numpy()
        z = np.r_[np.zeros(240), np.log(c[240:] / c[:-240])]
        return pl.Series(np.tanh(z / 0.05))

    out = S.stress_signal(slow_trend, ["SYNUSDT"], "1h", as_of=datetime(2026, 11, 25, tzinfo=UTC))
    base = out["base"]
    assert base["n"] > 5 and "open_end" in base and "verdict" in out
    assert base["block_days"] >= S.BLOCK_DAYS
    assert np.isfinite(out["funding_R_mean"]) and out["funding_R_mean"] != 0.0      # 1 bp per settlement, both sides
    assert out["open_end"]["n"] == base["open_end"]["n"]
    for seg in ("llm_in_sample", "clean_holdout"):
        assert out[seg]["n"] + out[seg].get("open_end", {}).get("n", 0) >= 0
