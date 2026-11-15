"""Live BTC logs (ai_trader.live): no look-ahead, the window-open refusal, scoring maths, the ACI state round-trip."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.live import claude_log, score
from ai_trader.live import nn as live
from ai_trader.ml import pnn

REPO = Path(__file__).resolve().parents[1]
HAVE_NETS = (REPO / "data" / "models" / "pnn" / "frozen_1h.json").is_file() and (REPO / "data" / "models" / "pnn" / "latent_1h.parquet").is_file()


# ------------------------------------------------------------------ ACI


def _stream(n=600, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.standard_t(4, n)
    lo, hi = np.full(n, -1.2) + 0.1 * rng.standard_normal(n), np.full(n, 1.2) + 0.1 * rng.standard_normal(n)
    return lo, hi, y


def test_online_aci_equals_pnn_aci():
    lo, hi, y = _stream()
    ref = pnn.aci(lo, hi, y, 0.2, gamma=0.005, window=100)
    o = live.OnlineACI(80, 0.005, 100)
    for t in range(len(y)):
        assert math.isclose(o.alpha_t, ref["alpha_t"][t], abs_tol=1e-12)
        blo, bhi, Q = o.band(lo[t], hi[t])
        assert math.isclose(blo, ref["lo"][t], abs_tol=1e-12) and math.isclose(bhi, ref["hi"][t], abs_tol=1e-12)
        e = o.step(lo[t], hi[t], y[t])
        assert e == ref["err"][t]


def test_aci_state_round_trip(tmp_path):
    lo, hi, y = _stream(400, seed=3)
    a = live.OnlineACI(90, 0.005, 150)
    for t in range(200):
        a.step(lo[t], hi[t], y[t], f"t{t}")
    b = live.OnlineACI.from_dict(json.loads(json.dumps(a.to_dict())))
    for t in range(200, 400):
        assert a.step(lo[t], hi[t], y[t]) == b.step(lo[t], hi[t], y[t])
        assert math.isclose(a.alpha_t, b.alpha_t, abs_tol=1e-9)
    # full state file round trip
    paths = live.Paths(tmp_path)
    st = {"schema": "x", "1h": {"last_ts": "2026-11-13T21:00:00+00:00", "aci": {L: live.OnlineACI(L, 0.005, 50) for L in live.LEVELS}},
          "24h": {"aci": {L: live.OnlineACI(L, 0.005 / 24, 50) for L in live.LEVELS}, "pending": [], "last_resolved_end": None}}
    st["1h"]["aci"][80].step(-1.0, 1.0, 0.3, "a")
    live.save_state(paths, st)
    back = live.load_state(paths)
    assert back["1h"]["aci"][80].to_dict() == st["1h"]["aci"][80].to_dict()
    assert back["1h"]["last_ts"] == st["1h"]["last_ts"] and back["24h"]["pending"] == []


def test_coverage_alarm():
    a = live.OnlineACI(80, 0.005, 50)
    a.hist = [["t", 1.0, 1.0]] * 40 + [["t", 0.0, 0.0]] * 32   # coverage 0.44 over 72
    c = live.coverage(a, 72)
    assert c["n"] == 72 and c["alarm"] and abs(c["cov"] - 32 / 72) < 1e-3


# ------------------------------------------------------------------ Claude log


def test_claude_log_window_and_refusal(tmp_path):
    now = datetime(2026, 11, 13, 22, 10, 5, tzinfo=UTC)
    row = claude_log.log_forecast(now - timedelta(minutes=7), "1h", -0.4, 0.0, 0.45, 0.33, "flat vol", "nn-20261113T22", root=tmp_path, now=now)
    assert row["window_start"] == "2026-11-13T23:00:00+00:00" and row["window_end"] == "2026-11-14T00:00:00+00:00"
    assert row["logged_at"] == now.isoformat() and row["model_id"] == "claude-opus-5-5" and row["prompt_version"]
    assert row["net_ref"] == "nn-20261113T22"
    r24 = claude_log.log_forecast(now, "24h", -2, 0, 2, 1.5, "x", None, root=tmp_path, now=now)
    assert r24["window_end"] == "2026-11-14T23:00:00+00:00"
    # exactly on a boundary: the next boundary, never the current one
    b = claude_log.log_forecast(now, "1h", -1, 0, 1, 1, "x", None, root=tmp_path, now=datetime(2026, 11, 13, 23, 0, tzinfo=UTC))
    assert b["window_start"] == "2026-11-14T00:00:00+00:00"
    n_before = len(claude_log.read_all(tmp_path))
    for ws in (datetime(2026, 11, 13, 22, tzinfo=UTC), datetime(2026, 11, 13, 21, tzinfo=UTC)):   # opened already
        with pytest.raises(ValueError, match="already opened"):
            claude_log.log_forecast(now, "1h", -1, 0, 1, 1, "x", None, window_start=ws, root=tmp_path, now=now)
    with pytest.raises(ValueError, match="cross"):
        claude_log.log_forecast(now, "1h", 1, 0, 2, 1, "x", None, root=tmp_path, now=now)
    with pytest.raises(ValueError, match="after the wall-clock"):
        claude_log.log_forecast(now + timedelta(minutes=5), "1h", -1, 0, 1, 1, "x", None, root=tmp_path, now=now)
    assert len(claude_log.read_all(tmp_path)) == n_before   # refusals write nothing
    ok = claude_log.log_forecast(now, "1h", -1, 0, 1, 1, "x", None, window_start=datetime(2026, 11, 14, 2, tzinfo=UTC), root=tmp_path, now=now)
    assert ok["window_start"] == "2026-11-14T02:00:00+00:00"


# ------------------------------------------------------------------ scoring maths on synthetic journals


def _net_line(S: datetime, q10, q50, q90, sigma, flags=()):
    q = {"q05": q10 - 0.2, "q10": q10, "q25": q50 - 0.1, "q50": q50, "q75": q50 + 0.1, "q90": q90, "q95": q90 + 0.2}
    nh = {"quantiles_pct": q, "sigma_pct": sigma, "window_start": S.isoformat(), "aci": {str(L): {"lo_pct": q10, "hi_pct": q90} for L in live.LEVELS}}
    return {"id": f"nn-{S:%Y%m%dT%H}", "forecast_hour": S.isoformat(), "logged_at": (S + timedelta(minutes=3)).isoformat(), "flags": list(flags),
            "net": {"1h": {**nh, "window_end": (S + timedelta(hours=1)).isoformat()}, "24h": {**nh, "window_end": (S + timedelta(hours=24)).isoformat()}}}


def _claude(S: datetime, h, q10, q50, q90, sigma, lead_min=50, i=0):
    return {"id": f"cl-{i}", "horizon": h, "window_start": S.isoformat(), "logged_at": (S - timedelta(minutes=lead_min)).isoformat(),
            "q10": q10, "q50": q50, "q90": q90, "sigma": sigma}


def test_pinball_by_hand():
    q = {"q10": -1.0, "q50": 0.0, "q90": 1.0}
    # y = 2: 0.1*3 + 0.5*2 + 0.9*1 = 2.2, mean 0.7333
    assert math.isclose(score.pinball3(2.0, q), (0.3 + 1.0 + 0.9) / 3)
    # y = -0.5: (0.1*0.5) + (0.5*0.5) + (0.1*1.5) = 0.05 + 0.25 + 0.15
    assert math.isclose(score.pinball3(-0.5, q), (0.05 + 0.25 + 0.15) / 3)
    assert math.isclose(live.pinball(2.0, q, {"q10": 0.1, "q50": 0.5, "q90": 0.9}), score.pinball3(2.0, q))


def test_pairing_rules_and_live1():
    S0 = datetime(2026, 11, 20, tzinfo=UTC)
    net = [_net_line(S0 + timedelta(hours=i), -1, 0, 1, 0.8) for i in range(4)]
    net[3]["flags"] = ["stale"]
    real = {S0: 0.5, S0 + timedelta(hours=1): -2.0, S0 + timedelta(hours=2): 0.1, S0 + timedelta(hours=3): 0.0}
    rf = lambda ws, h: real.get(ws) if h == "1h" else None  # noqa: E731
    C = [_claude(S0, "1h", -0.8, 0.2, 0.9, 0.7, i=1),
         _claude(S0, "1h", -5, 0, 5, 4, lead_min=20, i=2),                       # second for the same window: ignored
         _claude(S0 + timedelta(hours=1), "1h", -3, -1, 1, 1.5, lead_min=-1, i=3),  # logged after window open: ignored
         _claude(S0 + timedelta(hours=2), "1h", -1, 0, 1, 0.8, i=4),
         _claude(S0 + timedelta(hours=3), "1h", -1, 0, 1, 0.8, i=5),              # net line stale: ignored
         _claude(S0, "24h", -2, 0, 2, 1.5, i=6)]                                   # outcome not known: ignored
    P = score.pairs(C, net, rf)
    assert [p["claude_id"] for p in P] == ["cl-1", "cl-4"]
    p = P[0]
    assert math.isclose(p["loss_claude"], score.pinball3(0.5, {"q10": -0.8, "q50": 0.2, "q90": 0.9}))
    assert math.isclose(p["d"], p["loss_claude"] - score.pinball3(0.5, {"q10": -1, "q50": 0, "q90": 1}))
    assert P[1]["d"] == 0.0
    assert score.live1(P)["verdict"] == "pending"


def test_live1_verdicts_and_live2_hits():
    rng = np.random.default_rng(0)
    S0 = datetime(2026, 12, 1, tzinfo=UTC)
    net, C, real = [], [], {}
    for i in range(24 * 15):   # 360 hours, 15 days
        S = S0 + timedelta(hours=i)
        sd = 0.5 if (i // 24) % 2 else 1.5      # the network always says 1.0; Claude knows the true sd
        net.append(_net_line(S, -1.2816, 0, 1.2816, 1.0))
        C.append(_claude(S, "1h", -1.2816 * sd, 0, 1.2816 * sd, sd, i=i))
        C.append(_claude(S, "24h", -1.2816 * sd, 0, 1.2816 * sd, sd, i=10_000 + i))
        real[S] = float(rng.normal(0, sd))
    P = score.pairs(C, net, lambda ws, h: real.get(ws))
    assert len(P) == 720
    r1 = score.live1(P)
    assert r1["1h"]["n"] == 360 and r1["1h"]["mean_d"] < 0 and r1["1h"]["verdict"] == "supports" and r1["verdict"] == "supports"
    # reversed roles: Claude is the badly calibrated one
    for c in C:
        c["q10"], c["q90"] = c["q10"] * 3, c["q90"] * 3
    assert score.live1(score.pairs(C, net, lambda ws, h: real.get(ws)))["verdict"] == "contradicts"
    # H-LIVE2 (amended): Gaussian log-score difference on deviations; Claude knows the true sd -> supports
    r2 = score.live2(P, n_boot=200)
    exp = np.mean([score.gauss_logpdf(p["r"], 0, p["sigma_claude"]) - score.gauss_logpdf(p["r"], 0, 1.0) for p in P])
    assert r2["n"] == 720 and math.isclose(r2["mean_d"], exp) and r2["mean_d"] > 0 and r2["t"] > 2 and r2["verdict"] == "supports"
    hits = np.mean([(math.log(p["sigma_claude"]) > 0) == (math.log(abs(p["r"])) > 0) for p in P])
    assert math.isclose(r2["descriptive"]["rate"], hits)                # sign-hit kept, descriptive only
    bad = [dict(p, sigma_claude=1.0 / p["sigma_claude"]) for p in P]    # deviations in the wrong direction
    assert score.live2(bad, n_boot=50)["verdict"] == "contradicts"
    assert score.live2(P[:50], n_boot=50)["verdict"] == "pending"
    small = [dict(p, sigma_claude=p["sigma_net"] * 1.05) for p in P]   # within 10%: not deviations
    assert score.live2(small)["n"] == 0


def test_gauss_logscore_by_hand():
    assert math.isclose(score.gauss_logpdf(0.0, 0.0, 1.0), -0.5 * math.log(2 * math.pi))
    assert math.isclose(score.gauss_logpdf(1.0, 0.5, 2.0), -0.5 * math.log(2 * math.pi) - math.log(2.0) - 0.5 * 0.25 ** 2)


# ------------------------------------------------------------------ no look-ahead (synthetic lake, real frozen nets)


@pytest.mark.skipif(not HAVE_NETS, reason="frozen PNN artifacts not present")
def test_run_line_point_in_time_synthetic(tmp_path):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0, make_bars

    bars = make_bars(24 * 160, "BTCUSDT", seed=11)
    full = Lake(root=tmp_path / "full")
    full.write("bars", bars)
    as_of = T0 + timedelta(days=150, hours=5, minutes=3)
    part = Lake(root=tmp_path / "part")
    part.write("bars", bars.filter(pl.col("known_ts") <= as_of))
    fixed = as_of + timedelta(minutes=1)
    outs = []
    for i, lk in enumerate((full, part)):
        paths = live.Paths(tmp_path / f"out{i}", models_root=REPO)
        line, st, persist, scores = live.run_line(as_of, paths=paths, lake=lk, with_field=False, with_news=False, logged_at=fixed)
        outs.append((line, live.dump_state(st)))
    a, b = outs
    a[1].pop("created")
    b[1].pop("created")
    assert a[0] == b[0] and a[1] == b[1]
    line = a[0]
    assert line["forecast_hour"] == (T0 + timedelta(days=150, hours=5)).isoformat()
    assert line["data_max_ts"] <= as_of.isoformat() and "stale" not in line["flags"]
    q = line["net"]["1h"]["quantiles_pct"]
    assert q["q05"] < q["q10"] < q["q50"] < q["q90"] < q["q95"] and line["net"]["1h"]["sigma_pct"] > 0
    assert line["net"]["1h"]["window_start"] == line["forecast_hour"]
    # later data must not change the line (feature of bar S uses bars <= S-1)
    later = Lake(root=tmp_path / "later")
    later.write("bars", bars.filter(pl.col("known_ts") <= as_of + timedelta(hours=3)))
    c, _, _, _ = live.run_line(as_of, paths=live.Paths(tmp_path / "out2", models_root=REPO), lake=later, with_field=False, with_news=False, logged_at=fixed)
    assert c == line
    # stale flag when the newest bar is old
    s, _, _, _ = live.run_line(as_of + timedelta(hours=2), paths=live.Paths(tmp_path / "out3", models_root=REPO), lake=part, with_field=False,
                               with_news=False, logged_at=fixed)
    assert "stale" in s["flags"]


@pytest.mark.skipif(not HAVE_NETS, reason="frozen PNN artifacts not present")
def test_rolling_day_equals_training_rows_at_midnight():
    """k = 0: the live daily construction reproduces the study's cached panel row (agent F2's inputs)."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.pnn_study import DAILY_FEATURES
    from ai_trader.storage.lake import Lake

    cache = REPO / "data" / "models" / "ml" / "cache" / "panel_2026-11-12.parquet"
    if not cache.is_file():
        pytest.skip("panel cache missing")
    lake = Lake()
    S = datetime(2026, 11, 10, tzinfo=UTC)
    as_of = S + timedelta(minutes=3)
    try:
        f1 = market_frame("BTCUSDT", "1h", as_of, funding=False, oi=False, lake=lake)
    except Exception:  # noqa: BLE001
        pytest.skip("lake not available")
    if f1.height < 1000:
        pytest.skip("lake not available")
    f4 = market_frame("BTCUSDT", "4h", as_of, funding=False, oi=False, lake=lake)
    d = live.daily_row(f1, f4, lake, S, as_of)
    p = pl.read_parquet(cache).filter((pl.col("symbol") == "BTCUSDT") & (pl.col("T") == S)).to_dicts()[0]
    for c in DAILY_FEATURES:
        assert math.isclose(d[c], p[c], rel_tol=1e-9, abs_tol=1e-12), c
    assert max(f1["known_ts"].to_list()) <= as_of


# ------------------------------------------------------------------ v2 shadow (jump-cluster floor, faster ACI) and players


from ai_trader.live import hourly, nn_v2, players  # noqa: E402


def test_v2_floor_band_and_cluster():
    zq = nn_v2.Z_Q
    assert nn_v2.floor_band(-0.5, 0.5, 80, 1.0, False) == (-0.5, 0.5, False)          # inactive: untouched
    lo, hi, f = nn_v2.floor_band(-0.2, 0.6, 80, 1.0, True)                            # narrow: widened about the midpoint 0.2
    assert f and math.isclose(lo, 0.2 - zq[80]) and math.isclose(hi, 0.2 + zq[80])
    assert nn_v2.floor_band(-3.0, 3.0, 95, 1.0, True) == (-3.0, 3.0, False)           # already wider than 1.96
    assert nn_v2.floor_band(-0.1, 0.1, 90, None, True) == (-0.1, 0.1, False)
    S = datetime(2026, 11, 20, 12, tzinfo=UTC)

    def ts(h):
        return (S - timedelta(hours=h)).isoformat()

    zh = [[ts(7), 1.0, 0.2, 5.0], [ts(6), 0.1, 0.1, 1.0], [ts(1), 0.1, 0.1, -3.0]]
    assert nn_v2.cluster(zh, S, 6, 3.0) == []                                         # S-7h is outside, |z| = 3 is not > 3
    zh[1][3] = 3.1
    assert nn_v2.cluster(zh, S, 6, 3.0) == [[ts(6), 3.1]]
    zh.append([S.isoformat(), 0.1, 0.1, 9.0])                                         # bar S itself never counts
    assert nn_v2.cluster(zh, S, 6, 3.0) == [[ts(6), 3.1]]
    # band pinball = mean over the 6 endpoints: lower endpoint tau * (y - lo), upper (1 - tau) * (hi - y) when y is inside
    b = {80: (-1.0, 1.0), 90: (-2.0, 2.0), 95: (-3.0, 3.0)}
    exp = np.mean([0.1 * 1, 0.1 * 1, 0.05 * 2, 0.05 * 2, 0.025 * 3, 0.025 * 3])
    assert math.isclose(nn_v2.band_pinball(0.0, b), exp)
    exp_out = np.mean([0.1 * 5, 0.9 * 3, 0.05 * 6, 0.95 * 2, 0.025 * 7, 0.975 * 1])   # y = 4 above every band
    assert math.isclose(nn_v2.band_pinball(4.0, b), exp_out)


def test_v2_config_pinned():
    cfg = nn_v2.load_config(REPO / "config" / "live_nn_v2.yaml")
    s = nn_v2.Settings.from_config(cfg)
    assert s.gamma == {"1h": 0.02, "24h": 0.02 / 24} and s.lookback_h == 6 and s.z_trigger == 3.0 and len(s.cfg_hash) == 16
    assert cfg["model_id"] == nn_v2.MODEL_ID == "live_nn.v2.1"
    assert s.alpha_min == pytest.approx({80: 0.05, 90: 0.025, 95: 0.0125})


def test_v21_issue_alpha_floor():
    """v2.1: the band is issued at alpha_eff = max(alpha_t, nominal / 4); the state (alpha_t, scores) is untouched."""
    o = live.OnlineACI(95, 0.02, 1000, -0.006, [float(x) for x in np.linspace(0.0, 10.0, 1000)], [], 0)
    lo, hi, Q, ae = nn_v2.issue_band(o, -1.0, 1.0, 0.0125)
    assert ae == 0.0125 and o.alpha_t == -0.006                       # alpha_t stays unclipped
    assert o.band(-1.0, 1.0)[2] == 10.0                                # unclipped: the window's largest score
    k = math.ceil(1001 * (1 - 0.0125))
    assert Q == sorted(o.scores)[k - 1] < 10.0 and (lo, hi) == (-1.0 - Q, 1.0 + Q)
    o.alpha_t = 0.08                                                   # above the minimum: the plain ACI band
    assert nn_v2.issue_band(o, -1.0, 1.0, 0.0125)[:3] == o.band(-1.0, 1.0)


def _jump_bars(n: int, jump_at: int | None, size: float = 0.2, seed: int = 11) -> pl.DataFrame:
    from tests.conftest import make_bars

    b = make_bars(n, "BTCUSDT", seed=seed)
    if jump_at is None:
        return b
    c = b["close"].to_numpy().copy()
    c[jump_at:] *= math.exp(size)
    o = np.r_[b["open"][0], c[:-1]]
    return b.with_columns(pl.Series("close", c), pl.Series("open", o), pl.Series("high", np.maximum(o, c) * 1.002),
                          pl.Series("low", np.minimum(o, c) * 0.998), pl.Series("turnover", c))


def _v2_paths(root: Path) -> nn_v2.PathsV2:
    return nn_v2.PathsV2(root, models_root=REPO)


@pytest.mark.skipif(not HAVE_NETS, reason="frozen PNN artifacts not present")
def test_v2_point_in_time_and_same_network_as_v1(tmp_path):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0

    bars = _jump_bars(24 * 160, None)
    as_of = T0 + timedelta(days=150, hours=5, minutes=3)
    fixed = as_of + timedelta(minutes=1)
    lakes = {}
    for name, cut in (("full", None), ("part", as_of), ("later", as_of + timedelta(hours=3))):
        lakes[name] = Lake(root=tmp_path / name)
        lakes[name].write("bars", bars if cut is None else bars.filter(pl.col("known_ts") <= cut))
    outs = []
    for name, lk in lakes.items():
        line, st, _, _ = nn_v2.run_line(as_of, paths=_v2_paths(tmp_path / f"o_{name}"), lake=lk, with_field=False, hnn10={"status": "missing"},
                                        logged_at=fixed)
        d = nn_v2.dump_state(st)
        d.pop("created")
        outs.append((line, d))
    assert outs[0] == outs[1] == outs[2]          # data after as_of changes nothing
    line = outs[0][0]
    assert line["model_id"] == "live_nn.v2.1" and line["version"] == "live_nn.v2.1" and len(line["config_hash"]) == 16
    assert line["data_max_ts"] <= as_of.isoformat()
    v1, _, _, _ = live.run_line(as_of, paths=live.Paths(tmp_path / "v1", models_root=REPO), lake=lakes["part"], with_field=False, with_news=False,
                                logged_at=fixed)
    for h in ("1h", "24h"):
        assert line["net"][h]["quantiles_pct"] == v1["net"][h]["quantiles_pct"]   # same frozen network, only the conformal layer differs
    assert line["inputs_hash"] == v1["inputs_hash"]


@pytest.mark.skipif(not HAVE_NETS, reason="frozen PNN artifacts not present")
def test_v2_jump_cluster_floor_in_run(tmp_path):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0

    n = 24 * 160
    j = 24 * 150 + 3                                     # the bar opening at T0 + 150d 03:00 jumps +20%
    jump_ts = (T0 + timedelta(hours=j)).isoformat()
    lk = Lake(root=tmp_path / "lake")
    lk.write("bars", _jump_bars(n, j))
    as_of = T0 + timedelta(days=150, hours=5, minutes=3)   # S = 05:00, the jump bar (03:00) is S-2h
    cfg = nn_v2.Settings.from_config(nn_v2.load_config(REPO / "config" / "live_nn_v2.yaml"))

    def warm(p):   # the real warm-start stream covers these synthetic dates: start it 12 h earlier so the catch-up sees the jump
        return nn_v2.new_state(p, as_of - timedelta(hours=12), cfg)

    pa = _v2_paths(tmp_path / "a")
    line, st, _, _ = nn_v2.run_line(as_of, paths=pa, lake=lk, state=warm(pa), with_field=False, hnn10={"status": "missing"},
                                    logged_at=as_of + timedelta(minutes=1))
    assert line["aci_updates"]["1h_caught_up"] >= 10
    fl = line["net"]["1h"]["floor"]
    assert fl["active"] and jump_ts in [t for t, _ in fl["trigger"]] and "jump_cluster" in line["flags"]
    assert fl["sigma_floor_pct"] == pytest.approx(line["net"]["1h"]["sigma_pct"])   # only the net is available here
    n1 = line["net"]["1h"]
    q, sr = n1["quantiles_pct"], n1["sigma_ref_pct"]
    for L in live.LEVELS:
        b, a = n1["band"][str(L)], n1["aci"][str(L)]
        assert b["alpha_eff"] == pytest.approx(max(a["alpha_t"], (1 - L / 100) / 4))
        qlo, qhi = (q["q10"], q["q90"]) if L == 80 else (q["q05"], q["q95"])
        ilo, ihi = qlo - b["Q"] * sr, qhi + b["Q"] * sr                    # the alpha_eff band before the jump floor
        assert (b["hi_pct"] - b["lo_pct"]) / 2 >= nn_v2.Z_Q[L] * fl["sigma_floor_pct"] - 1e-9
        assert b["lo_pct"] <= ilo + 1e-9 and b["hi_pct"] >= ihi - 1e-9   # the jump floor only widens
        if b["floored"]:
            assert math.isclose(0.5 * (b["lo_pct"] + b["hi_pct"]), 0.5 * (ilo + ihi), abs_tol=1e-9)
        else:
            assert math.isclose(b["lo_pct"], ilo, abs_tol=1e-9) and math.isclose(b["hi_pct"], ihi, abs_tol=1e-9)
    # 7 h after the jump bar it is out of the lookback
    pb = _v2_paths(tmp_path / "b")
    later, _, _, _ = nn_v2.run_line(as_of + timedelta(hours=5), paths=pb, lake=lk, state=warm(pb), with_field=False,
                                    hnn10={"status": "missing"}, logged_at=as_of + timedelta(hours=5, minutes=1))
    assert jump_ts not in [t for t, _ in later["net"]["1h"]["floor"]["trigger"]]
    # without the jump, no trigger at that bar
    lk2 = Lake(root=tmp_path / "lake2")
    lk2.write("bars", _jump_bars(n, None))
    pc = _v2_paths(tmp_path / "c")
    calm, _, _, _ = nn_v2.run_line(as_of, paths=pc, lake=lk2, state=warm(pc), with_field=False, hnn10={"status": "missing"},
                                   logged_at=as_of + timedelta(minutes=1))
    assert jump_ts not in [t for t, _ in calm["net"]["1h"]["floor"]["trigger"]]


@pytest.mark.skipif(not HAVE_NETS, reason="frozen PNN artifacts not present")
def test_v1_outputs_byte_identical_with_or_without_v2(tmp_path):
    import shutil

    from ai_trader.storage.lake import Lake
    from tests.conftest import T0

    lk = Lake(root=tmp_path / "lake")
    lk.write("bars", _jump_bars(24 * 160, 24 * 150 + 3))
    S0 = T0 + timedelta(days=150, hours=5)

    def hour(root, k, v2):   # with_news=True: v1's render_latest needs the news block (with_news=False breaks it, a v1 quirk)
        S = S0 + timedelta(hours=k)
        return hourly.run_hour(S + timedelta(minutes=3), paths1=live.Paths(root, models_root=REPO), paths2=_v2_paths(root), v2=v2,
                               with_players=v2, logged_at=S + timedelta(minutes=3, seconds=5), lake=lk, with_field=False, with_news=True,
                               clock=lambda: 0.0)

    A, B = tmp_path / "A", tmp_path / "B"
    hour(A, 0, False)
    shutil.copytree(A, B)
    ks = (1, 2, 4, 5)   # hours 06, 07, (08 missed: the synthetic lake has no 4h bars for the k = 0 mod 4 construction), 09, 10
    for k in ks:
        hour(A, k, False)
        rb = hour(B, k, True)
        assert "v2_error" not in rb, rb.get("v2_error")
    v1_files = sorted(p.relative_to(A) for p in A.rglob("*") if p.is_file() and p.name != "LATEST.md")
    assert any("live_nn" in str(p) for p in v1_files) and any(p.name == "live_aci_state.json" for p in v1_files)
    for p in v1_files:
        assert (A / p).read_bytes() == (B / p).read_bytes(), p
    assert not (A / "data" / "journal" / "live_nn_v2").exists()
    v2_lines = nn_v2.read_lines(_v2_paths(B))
    assert [x["id"] for x in v2_lines] == [f"v21-{S0 + timedelta(hours=k):%Y%m%dT%H}" for k in ks]
    assert nn_v2.read_scores(_v2_paths(B))                       # closed 1h windows of earlier v2 lines were scored
    md = (B / "research" / "live" / "LATEST.md").read_text(encoding="utf-8")
    assert "## v2.1 shadow" in md and "## Regime and players" in md and len(md.splitlines()) <= 62


def _ts_ms(xs):
    return pl.Series(xs, dtype=pl.Datetime("us", "UTC")).dt.cast_time_unit("ms")


def _players_lake(root: Path, end: datetime, ls_last: datetime):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0, make_bars

    lk = Lake(root=root)
    n = int((end - T0).total_seconds() // 3600)
    lk.write("bars", make_bars(n, "BTCUSDT", seed=5))
    rng = np.random.default_rng(2)
    fts = [T0 + timedelta(hours=8 * i) for i in range(n // 8)]
    lk.write("funding", pl.DataFrame({"venue": "bybit", "symbol": "BTCUSDT", "event_ts": _ts_ms(fts), "rate": rng.normal(1e-4, 1e-4, len(fts)),
                                      "known_ts": _ts_ms([t + timedelta(seconds=2) for t in fts])}))
    ots = [T0 + timedelta(hours=4 * i) for i in range(n // 4)]
    lk.write("open_interest", pl.DataFrame({"venue": "bybit", "interval": "4h", "symbol": "BTCUSDT", "event_ts": _ts_ms(ots),
                                            "open_interest": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ots)))),
                                            "known_ts": _ts_ms([t + timedelta(hours=4, seconds=2) for t in ots])}))
    lts = [end - timedelta(days=3) + timedelta(minutes=7 * i) for i in range(int(3 * 24 * 60 / 7))]
    lk.write("liquidations", pl.DataFrame({"venue": "bybit", "symbol": "BTCUSDT", "event_ts": _ts_ms(lts),
                                           "side": ["long" if i % 3 else "short" for i in range(len(lts))],
                                           "price": 100.0, "size": rng.uniform(1, 10, len(lts)), "known_ts": _ts_ms(lts)}))
    dts = [T0 + timedelta(hours=i) for i in range(n)]
    lk.write("alt_series", pl.DataFrame({"source": "deribit", "series_id": "dvol_close", "entity": "BTC", "event_ts": _ts_ms(dts),
                                         "value": 40 + rng.normal(0, 3, n), "vintage": pl.Series([None] * n, dtype=pl.Utf8), "pit_quality": "native",
                                         "known_ts": _ts_ms([t + timedelta(hours=1, seconds=2) for t in dts])}))
    sts = [ls_last - timedelta(hours=i) for i in range(48)][::-1]
    lk.write("ls_ratio", pl.DataFrame({"venue": "bybit", "period": "1h", "symbol": "BTCUSDT", "event_ts": _ts_ms(sts), "buy_ratio": 0.6,
                                       "sell_ratio": 0.4, "known_ts": _ts_ms([t + timedelta(hours=1, seconds=2) for t in sts])}))
    return lk


def test_players_point_in_time(tmp_path):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0

    as_of = T0 + timedelta(days=200, hours=9, minutes=3)
    full = _players_lake(tmp_path / "full", as_of + timedelta(days=2), as_of + timedelta(days=1))
    P_full = players.players(as_of, lake=full, symbols=("BTCUSDT",))
    part = Lake(root=tmp_path / "part")
    for t in ("bars", "funding", "open_interest", "liquidations", "alt_series", "ls_ratio"):
        part.write(t, full.read_asof(t, as_of))
    P_part = players.players(as_of, lake=part, symbols=("BTCUSDT",))
    assert P_full == P_part                                     # rows known after as_of change nothing
    p = P_full["symbols"]["BTCUSDT"]
    assert p["regime"]["status"] == "ok" and p["regime"]["cell"] in {f"R{i}" for i in range(1, 10)} and p["regime"]["bar"] < as_of.isoformat()
    assert p["funding"]["ts"] <= as_of.isoformat() and 0 <= p["funding"]["pct_90d"] <= 1 and p["funding"]["interval_h"] == 8
    assert p["oi"]["ts"] <= (as_of - timedelta(hours=4)).isoformat() and p["oi"]["chg_24h_pct"] is not None
    liq = full.read_asof("liquidations", as_of, symbols=["BTCUSDT"]).with_columns((pl.col("price") * pl.col("size")).alias("usd"))
    w4 = liq.filter((pl.col("event_ts") > as_of - timedelta(hours=4)) & (pl.col("side") == "long"))
    assert w4.height and math.isclose(p["liq"]["long_4h_usd"], float(w4["usd"].sum()))
    assert p["dvol"]["status"] == "ok" and p["dvol"]["ts"] <= (as_of - timedelta(hours=1)).isoformat()
    assert p["ls"]["status"] == "ok" and p["ls"]["ts"] < as_of.isoformat()     # fresh ls_ratio is shown...
    stale = _players_lake(tmp_path / "stale", as_of + timedelta(days=2), as_of - timedelta(days=2))
    assert players.players(as_of, lake=stale, symbols=("BTCUSDT",))["symbols"]["BTCUSDT"]["ls"]["status"] == "stale"   # ...stale is not
    lines = players.render(P_full)
    assert lines[1].startswith("| | v0 cell") and any(x.startswith("- Pattern: BTC ") for x in lines)
