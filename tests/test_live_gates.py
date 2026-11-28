"""Live gates (D2 release-hour rule, D3 jump-intensity gate): src/ai_trader/live/gates.py."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.live import gates as G

ROOT = Path(__file__).resolve().parents[1]
WS05 = ROOT / "data" / "research" / "ws05"
M, B, H, D = G.M_MS, G.B_MS, G.H_MS, G.D_MS


def ms(*a) -> int:
    return int(datetime(*a, tzinfo=UTC).timestamp() * 1000)


def events_file(tmp_path: Path, rows: list[tuple[str, str, int, int]]) -> Path:
    df = pl.DataFrame({"kind": [r[0] for r in rows], "name": [r[1] for r in rows],
                       "scheduled_ts": pl.Series([r[2] for r in rows]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
                       "schedule_known_ts": pl.Series([r[3] for r in rows]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
                       "source": ["test"] * len(rows)})
    p = tmp_path / "events.parquet"
    df.write_parquet(p)
    return p


def flat_ss(n: int = 400, mu: float = 0.004, t0: int | None = None, jumps: list[int] = ()) -> dict:
    t0 = ms(2026, 11, 20) if t0 is None else t0
    ts = t0 + np.arange(n, dtype=np.int64) * B
    y = np.zeros(n)
    y[list(jumps)] = 1.0
    e1, e2 = G._lf(y, G.KAPPA1), G._lf(y, G.KAPPA2)
    return {"ts": ts, "y": y, "mu": np.full(n, mu), "e1": e1, "e2": e2, "self_abs": G.ALPHA1 * e1 + G.ALPHA2 * e2}


# ---------------------------------------------------------------- constants are WS05's frozen fits


def test_constants_match_ws05_outputs():
    from ai_trader.research.ws05_tails import lm_threshold
    assert abs(lm_threshold(288) - G.LM_THR_5M) < 1e-3
    assert abs(G.ALPHA1 - 0.0993) < 1e-3 and abs(G.ALPHA2 - 0.01043) < 1e-4
    assert abs(5 * math.log(2) / -math.log(G.KAPPA1) - 4.91) < 0.01 and abs(5 * math.log(2) / -math.log(G.KAPPA2) - 50.86) < 0.05
    if not (WS05 / "hawkes_fits.json").is_file():
        pytest.skip("data/research/ws05 not present")
    hf = json.loads((WS05 / "hawkes_fits.json").read_text())["cutoff_fit"]
    assert tuple(hf["M4b"]["theta"]) == G.M4B_THETA
    assert tuple(hf["M5"]["theta"][3:5]) == G.M5_RELEASE
    mh = json.loads((WS05 / "liq.json").read_text())["marked_hawkes"]
    assert mh["gamma_liq"] == G.LIQ_GAMMA and mh["half_life_liq_min"] == G.LIQ_HALF_LIFE_MIN
    assert mh["excess_ratio_at_lag"]["60"] == G.THRESHOLD and mh["excess_ratio_at_lag"]["1"] == G.LIQ_RATIO_LAG1


# ---------------------------------------------------------------- D2


def test_release_windows_flag_bars_and_point_in_time(tmp_path):
    t = ms(2026, 12, 4, 13, 30)
    p = events_file(tmp_path, [("nfp", "NFP", t, t - 30 * D), ("cpi", "CPI", ms(2026, 12, 10, 13, 30), ms(2026, 12, 10, 13, 30) - 30 * D),
                               ("fomc", "FOMC decision", ms(2026, 12, 9, 19), ms(2026, 12, 9, 19) + D)])   # FOMC not yet known
    ev = G.load_events(p)
    w = G.release_windows(ev, t - 6 * M)
    assert [x["event"] for x in w] == ["NFP", "CPI"], "the FOMC row is not known yet (schedule_known_ts > as_of)"
    assert (w[0]["start_ms"], w[0]["end_ms"], w[0]["active"]) == (t - 5 * M, t + 30 * M, False)
    assert G.release_windows(ev, t - 5 * M)[0]["active"] and G.release_windows(ev, t + 30 * M)[0]["active"]
    assert not any(x["event"] == "NFP" for x in G.release_windows(ev, t + 31 * M)), "a window is dropped once it ended"
    fb = G.release_flag_bars(ev, t - 2 * H)
    one = {(b["freq"], b["bar"]): b["bar_open_ms"] for b in fb if b["event"] == "NFP"}
    assert one[("1h", "release")] == ms(2026, 12, 4, 13) and one[("1h", "next")] == ms(2026, 12, 4, 14)
    assert one[("4h", "release")] == ms(2026, 12, 4, 12) and one[("4h", "next")] == ms(2026, 12, 4, 16)
    act = [b for b in G.release_flag_bars(ev, t + 40 * M) if b["active"]]
    assert {(b["freq"], b["bar"]) for b in act} == {("1h", "next"), ("4h", "release")}


def test_calendar_must_hold_a_future_release_of_every_kind(tmp_path):
    now = ms(2026, 11, 27, 22)
    rows = [("nfp", "NFP", ms(2026, 12, 4, 13, 30), ms(2026, 11, 4)), ("cpi", "CPI", ms(2026, 12, 10, 13, 30), ms(2026, 11, 10)),
            ("fomc", "FOMC decision", ms(2027, 1, 27, 19), ms(2026, 12, 28))]
    c = G.calendar_state(G.load_events(events_file(tmp_path, rows)), now)
    assert c["ok"] and c["warn"], "ok, with a warning that CPI / NFP run out within 21 days"
    c = G.calendar_state(G.load_events(events_file(tmp_path, rows)), ms(2026, 12, 5))
    assert not c["ok"] and "nfp" in c["reason"]


# ---------------------------------------------------------------- D3 model pieces


def test_liquidation_term_reproduces_ws05_excess_table():
    ss = flat_ss()
    t0 = int(ss["ts"][200])
    L = G.liq_kernel(np.r_[1.0, np.zeros(200)])           # one cluster minute at grid minute 0
    for lag, want in ((1, 3.045217081368285), (30, 2.0656570345348664), (60, 1.5429160428195066), (120, 1.140916910381488)):
        c = G.ratio_now(ss, 200, float(L[lag]), t0, np.zeros(0, dtype=np.int64))
        assert abs(c["ratio"] - want) < 1e-9, (lag, c)


def test_self_excitation_and_release_marks():
    ss = flat_ss(jumps=[100])
    for k in (1, 12, 24):
        c = G.ratio_now(ss, 100 + k, 0.0, int(ss["ts"][100 + k]), np.zeros(0, dtype=np.int64))
        want = 1 + (G.ALPHA1 * G.KAPPA1 ** (k - 1) + G.ALPHA2 * G.KAPPA2 ** (k - 1)) / 0.004
        assert abs(c["ratio"] - want) < 1e-9
    rel = np.array([int(ss["ts"][300]) + 30 * 1000], dtype=np.int64)   # release 30 s into bin 300
    assert abs(G.ratio_now(ss, 300, 0.0, int(ss["ts"][300]) + 60_000, rel)["release"] - math.exp(G.M5_RELEASE[0])) < 1e-9
    assert abs(G.release_mult(int(ss["ts"][305]), rel) - math.exp(G.M5_RELEASE[1])) < 1e-12
    assert G.release_mult(int(ss["ts"][306]), rel) == 1.0
    r0, r1 = G.release_marks(ss["ts"], rel)
    assert r0[300] == 1 and r1[301:306].sum() == 5 and r1[306] == 0


def test_projection_closes_the_gate_one_minute_after_the_60_min_level():
    ss = flat_ss()
    b = 200
    t = int(ss["ts"][b])
    st = G.jump_state(ss, G.liq_kernel(np.r_[1.0, np.zeros(3)]), t // M - 1, t, np.zeros(0, dtype=np.int64), last_valid_min=t // M - 1)
    assert st["elevated"] and abs(st["ratio"] - G.LIQ_RATIO_LAG1) < 1e-3
    assert st["until_ms"] == t + 60 * M, "lag 1 now -> lag 61 (first minute below the lag-60 level) 60 minutes later"
    assert st["p_jump_1h"] > st["p_jump_1h_base"] > 0
    quiet = G.jump_state(ss, None, None, t, np.zeros(0, dtype=np.int64), last_valid_min=t // M - 1)
    assert not quiet["elevated"] and quiet["until_ms"] is None and abs(quiet["ratio"] - 1.0) < 1e-9


def test_fail_safe_stale_flow_and_errors(tmp_path):
    ss = flat_ss()
    t = int(ss["ts"][200])
    st = G.jump_state(ss, None, None, t, np.zeros(0, dtype=np.int64), last_valid_min=t // M - 25)
    assert st["elevated"] and "flow stale" in st["reason"]

    class Boom:
        def minutes(self, as_of):
            raise OSError("lake unreadable")

    doc = G.build(ms(2026, 11, 27, 22), flow=Boom(), events_path=tmp_path / "missing.parquet", symbols=["BTCUSDT", "ETHUSDT"])
    assert not doc["calendar"]["ok"] and "unreadable" in doc["calendar"]["reason"]
    assert all(j["elevated"] and "lake unreadable" in j["reason"] for j in doc["jump"].values())
    assert len(doc["errors"]) == 2


def synthetic_minutes(days: int, seed: int = 0, t0: int | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = ms(2026, 11, 10) if t0 is None else t0
    n = days * 1440
    lp = np.cumsum(rng.standard_normal(n) * 4e-4) + math.log(100.0)
    return pl.DataFrame({"symbol": ["BTCUSDT"] * n, "minute": t0 // M + np.arange(n, dtype=np.int64), "close": np.exp(lp), "n3": np.full(n, 20, dtype=np.int64)})


def test_series_are_point_in_time():
    m1 = synthetic_minutes(5)
    mn, cl, n3 = m1["minute"].to_numpy(), m1["close"].to_numpy(), m1["n3"].to_numpy()
    cut = int(mn[0]) + 4 * 1440 + 37            # mid-bin
    k = mn <= cut - 1
    bt_a, r_a = G.five_minute(mn[k], cl[k], n3[k], cut)
    bt_b, r_b = G.five_minute(mn, cl, n3, int(mn[-1]))
    assert bt_a[-1] == cut // 5 * 5 * M, "the grid reaches the bin in progress"
    sa, sb = G.self_series(bt_a, r_a, "BTCUSDT"), G.self_series(bt_b, r_b, "BTCUSDT")
    n = len(bt_a)
    for key in ("mu", "self_abs", "e1", "e2"):
        a, b = sa[key][:n], sb[key][:n]
        ok = np.isfinite(a)
        assert ok[-1] and np.allclose(a[ok], b[ok], rtol=0, atol=1e-15), f"{key}: later data changed an earlier value"


def test_liquidation_flags_follow_r_flow_2():
    t0 = ms(2026, 11, 20) // M
    n = 26 * 60
    rng = np.random.default_rng(1)
    ev = []
    for m in range(0, n, 7):
        ev.append((t0 + m, 1.0 + rng.random(), "long"))
    burst = t0 + n - 30
    ev += [(burst, 5000.0, "long")] * 4
    liq = pl.DataFrame({"venue": "bybit", "symbol": "ETHUSDT", "event_ts": pl.Series([e[0] * M + 1000 for e in ev]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
                        "side": [e[2] for e in ev], "price": [e[1] for e in ev], "size": [1.0] * len(ev)})
    fl = G.liquidation_flags(liq, np.arange(t0, t0 + n), t0, t0 + n)["ETHUSDT"]
    assert fl[burst - t0] == 1 and fl.sum() == 1
    fl_down = G.liquidation_flags(liq, np.arange(t0, t0 + n - 60), t0, t0 + n)["ETHUSDT"]
    assert fl_down.sum() == 0, "collector down in the burst minute: liquidations unknown, not a cluster"


def test_liquidation_cache_is_incremental_deduplicated_and_point_in_time(tmp_path):
    d = tmp_path / "year=2026" / "month=11"
    d.mkdir(parents=True)
    t = ms(2026, 11, 27, 20)

    def part(name, rows):
        pl.DataFrame({"symbol": [r[0] for r in rows], "event_ts": pl.Series([r[1] for r in rows]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
                      "side": ["long"] * len(rows), "price": [100.0] * len(rows), "size": [1.0] * len(rows),
                      "known_ts": pl.Series([r[2] for r in rows]).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")}).write_parquet(d / name)

    part("part-1.parquet", [("ETHUSDT", t, t + 1000), ("ETHUSDT", t - 40 * H, t - 40 * H + 1000)])   # second row: too old
    c = G.LiqCache(tmp_path)
    a = c.read(t + 60_000, t - G.LIQ_LOOKBACK_MS)
    assert a.height == 1 and a["symbol"][0] == "ETHUSDT"
    part("part-2.parquet", [("ETHUSDT", t, t + 2000), ("SOLUSDT", t + 30_000, t + 90_000)])      # a duplicate print and a late one
    b = c.read(t + 60_000, t - G.LIQ_LOOKBACK_MS)
    assert b.height == 1, "same print twice -> one row; SOL is not known yet at as_of"
    assert c.read(t + 120_000, t - G.LIQ_LOOKBACK_MS).height == 2
    assert len(c.seen) == 2, "each part file is read once"


# ---------------------------------------------------------------- file + packet line


def test_write_atomic_and_packet_line(tmp_path):
    p = tmp_path / "gates.json"
    t = ms(2026, 12, 4, 13, 31)
    doc = {"version": G.VERSION, "min_stop_sigma": 2.0, "calendar": {"ok": True},
           "release_windows": [{"event": "NFP", "release_ms": t - M, "start_ms": t - 6 * M, "end_ms": t + 29 * M}],
           "release_flag_bars": [{"event": "NFP", "freq": "1h", "bar": "release", "bar_open_ms": ms(2026, 12, 4, 13), "bar_close_ms": ms(2026, 12, 4, 14)}],
           "jump": {"BTCUSDT": {"ratio": 77.1, "elevated": True, "until_utc": "2026-12-04T14:01:00Z", "p_jump_1h": 0.41, "p_jump_1h_base": 0.05},
                    "ETHUSDT": {"ratio": 1.02, "elevated": False, "p_jump_1h": 0.05, "p_jump_1h_base": 0.05}},
           "jump_model": {"fallback_symbol": "BTCUSDT"}}
    G.write_atomic(doc, p)
    g = G.load(p)
    assert g["written_ms"] > 0 and not (tmp_path / "gates.json.tmp").exists()
    line = G.packet_line(gates={**g, "written_ms": t - 30_000}, now=t)
    assert "RELEASE WINDOW NFP 13:30Z" in line and "BTCUSDT jump intensity x77.10 ELEVATED until 14:01Z" in line and "ETHUSDT jump intensity x1.02 normal" in line
    assert "widen" in line and "jump probability" not in line
    assert "STALE" in G.packet_line(gates={**g, "written_ms": t - 6 * M}, now=t)
    assert "missing" in G.packet_line(path=tmp_path / "nope.json", now=t)
    assert "XMRUSDT jump intensity x77.10" in G.packet_line(["XMRUSDT"], gates={**g, "written_ms": t}, now=t), "unlisted symbols read the BTC fallback"


def test_desk_contract_r_flow_2_reads_the_intensity_gate():
    from ai_trader.desk import contract as C
    tape = {"SOLUSDT": {"liq": {"cluster_60m": {"fired": True}}}}
    # no gates state: the old 60-min tape flag is the fallback
    assert any("60-min fallback" in x for x in C._gate_warnings("p", "SOLUSDT", [], None, tape))
    g = {"jump": {"BTCUSDT": {"elevated": False, "ratio": 1.05}, "SOLUSDT": {"elevated": False, "ratio": 1.1}}, "jump_model": {"fallback_symbol": "BTCUSDT"},
         "release_windows": []}
    assert C._gate_warnings("p", "SOLUSDT", [], g, tape) == [], "the gate, not the 60-min clock, decides once a gates state exists"
    g["jump"]["SOLUSDT"] = {"elevated": True, "ratio": 2.4, "until_utc": "2026-11-27T23:40:00Z", "components": {"liq": 1.3}}
    w = C._gate_warnings("p", "SOLUSDT", [], g, {})
    assert len(w) == 1 and "R-FLOW-2" in w[0] and "x2.40 until 23:40Z" in w[0] and "liquidation cascade" in w[0]
    assert C._gate_warnings("p", "SOLUSDT", ["R-FLOW-2"], g, {}) == []
    g["jump"]["BTCUSDT"]["elevated"] = True
    assert "BTCUSDT fallback" in C._gate_warnings("p", "KAVAUSDT", [], g, {})[0]
    g["release_windows"] = [{"event": "CPI", "active": True}]
    assert any("R-WS05-2" in x for x in C._gate_warnings("p", "SOLUSDT", ["R-FLOW-2"], g, {}))
