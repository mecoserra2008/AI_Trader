"""Live cross-sectional v3 (ai_trader.live.xs_v3): the exact speed-ups, no look-ahead, the ranking / scoring maths, the
ACI warm start, v1 / v2.1 byte-identical with v3 on or off, and the dashboard's /api/xs."""

import itertools
import json
import math
import shutil
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.live import xs_v3

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "data" / "models" / "xs_v3" / "frozen" / "spec.json"
HAVE_V3 = FROZEN.is_file()
HAVE_NETS = (REPO / "data" / "models" / "pnn" / "frozen_1h.json").is_file() and (REPO / "data" / "models" / "pnn" / "latent_1h.parquet").is_file()
TOKENS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT")


# ------------------------------------------------------------------ exact speed-ups


def test_tail_profiles_equal_research_functions():
    from ai_trader.features.seasonality import hour_of_week, seasonal_factor
    from ai_trader.ml import xs_net as xs

    rng = np.random.default_rng(3)
    n = 24 * 7 * 12
    ts = pl.Series((np.datetime64("2026-01-05T00:00", "us") + np.arange(n) * np.timedelta64(3600, "s")).astype("datetime64[us]"))
    how = hour_of_week(ts)
    u = rng.standard_t(3, n) ** 2
    u[rng.random(n) < 0.05] = np.nan
    u[100] = 400.0                                    # above the cap
    full_f, full_m = seasonal_factor(how, u), xs.seasonal_multi(how, u, (4, 24))
    for tail in (1, 48, 500):
        f = xs_v3._tail_profile(how, u, None, tail, 20.0, 1, 50.0, 0.1, 8)
        m = xs_v3._tail_profile(how, u, (4, 24), tail, 20.0, 1, 50.0, 0.1, 8)
        assert np.array_equal(f[-tail:], full_f[-tail:]) and np.isnan(f[:-tail]).all()       # bitwise, not approx
        for h in (4, 24):
            assert np.array_equal(m[h][-tail:], full_m[h][-tail:])
    with xs_v3.fast_seasonal(10):
        from ai_trader.features import seasonality as sz
        assert np.array_equal(sz.seasonal_factor(how, u)[-10:], full_f[-10:]) and np.array_equal(xs.seasonal_multi(how, u)[24][-10:], full_m[24][-10:])
    assert sz.seasonal_factor is seasonal_factor and xs.seasonal_multi is not None      # restored


def _funding(sym: str, start: datetime, n_hours: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ts = [start + timedelta(hours=8 * i) for i in range(n_hours // 8)]
    return pl.DataFrame({"venue": "bybit", "symbol": sym, "event_ts": ts, "rate": rng.normal(1e-4, 1e-4, len(ts)),
                         "known_ts": [t + timedelta(seconds=2) for t in ts]}).with_columns(
        pl.col(["event_ts", "known_ts"]).dt.cast_time_unit("ms"))


def _oi(sym: str, start: datetime, n_hours: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 100)
    ts = [start + timedelta(hours=4 * i) for i in range(n_hours // 4)]
    return pl.DataFrame({"venue": "bybit", "interval": "4h", "symbol": sym, "event_ts": ts,
                         "open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ts)))),
                         "known_ts": [t + timedelta(hours=4, seconds=2) for t in ts]}).with_columns(pl.col(["event_ts", "known_ts"]).dt.cast_time_unit("ms"))


def _xs_lake(root: Path, n_hours: int = 24 * 170, tokens=TOKENS, btc_bars: pl.DataFrame | None = None):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0, make_bars

    lk = Lake(root=root)
    bars = [make_bars(n_hours, s, seed=10 + k) for k, s in enumerate(tokens) if not (s == "BTCUSDT" and btc_bars is not None)]
    if btc_bars is not None:
        bars.append(btc_bars)
    lk.write("bars", pl.concat(bars))
    lk.write("funding", pl.concat([_funding(s, T0, n_hours, k) for k, s in enumerate(tokens)]))
    lk.write("open_interest", pl.concat([_oi(s, T0, n_hours, k) for k, s in enumerate(tokens) if s != "SOLUSDT"]))   # SOL: no OI (optional input)
    return lk


def test_prefetch_lake_and_lean_frame_are_exact(tmp_path):
    from ai_trader.research.frame import market_frame
    from tests.conftest import T0

    lk = _xs_lake(tmp_path / "lake", n_hours=24 * 60, tokens=("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    as_of = T0 + timedelta(days=50, hours=7, minutes=3)
    pf = xs_v3.PrefetchLake(lk, as_of, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    for s in ("ETHUSDT", "SOLUSDT", "XRPUSDT"):
        for t, f in xs_v3.PrefetchLake.SPECS.items():
            assert pf.read_asof(t, as_of, symbols=[s], filters=f).equals(lk.read_asof(t, as_of, symbols=[s], filters=f))
        a = market_frame(s, "1h", as_of, funding=True, oi=True, oi_interval="4h", lake=lk)
        with xs_v3.lean_frame():
            b = market_frame(s, "1h", as_of, funding=True, oi=True, oi_interval="4h", lake=pf)
        if a.height:
            assert a.select(*xs_v3.FRAME_COLS).equals(b.select(*xs_v3.FRAME_COLS))       # every column v3 reads, bitwise
            assert b["sigma_rb_next"].is_nan().all() or b["sigma_rb_next"].is_null().all()
        else:
            assert b.height == 0
    assert pf.data_max() <= as_of


# ------------------------------------------------------------------ ranking / scoring maths


def test_ranks_spearman_and_relative():
    v = np.array([3.0, 1.0, 2.0, np.nan, 5.0, 2.0])
    m = np.array([True, True, True, True, False, True])
    r = xs_v3.ranks_desc(v, m)
    assert r[0] == 1 and r[1] == 4 and r[2] == r[5] == 2.5 and np.isnan(r[3]) and np.isnan(r[4])
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=9), rng.normal(size=9)
    k = np.ones(9, bool)
    k[2] = False
    assert math.isclose(xs_v3.spearman(a, b, k), spearmanr(a[k], b[k]).statistic, abs_tol=1e-12)
    assert xs_v3.spearman(a[:4], b[:4], np.ones(4, bool)) is None                        # < 5 tokens
    R = np.array([0.01, 0.02, np.nan, -0.01, 0.0, 0.03])
    rel = xs_v3.relative_realised(R, np.isfinite(R))
    assert np.isnan(rel[2]) and math.isclose(np.nansum(rel), 0, abs_tol=1e-15) and math.isclose(rel[0], 0.01 - 0.01, abs_tol=1e-15)
    q = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
    taus = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    y = 0.8
    assert math.isclose(xs_v3.pinball7(y, q), float(np.mean([max(t * (y - x), (t - 1) * (y - x)) for t, x in zip(taus, q, strict=True)])))


def test_attention_summary_by_hand():
    syms = ["A", "B", "C", "D"]
    present = np.array([True, True, False, True])
    w = np.zeros((4, 5))
    w[0, [0, 1, 3, 4]] = [0.1, 0.5, 0.2, 0.2]
    w[1, [0, 1, 3, 4]] = [0.4, 0.1, 0.1, 0.4]
    w[3, [0, 1, 3, 4]] = [0.3, 0.3, 0.1, 0.3]
    a = xs_v3.attention_summary(w, present, syms)
    assert a["keys"] == ["A", "B", "D", "MARKET"] and a["queries"] == ["A", "B", "D"] and a["n_keys"] == 4
    assert math.isclose(a["received"]["A"], (0.4 + 0.3) / 2 * 4)            # from B and D, x 4 keys
    assert math.isclose(a["received"]["MARKET"], (0.2 + 0.4 + 0.3) / 3 * 4)
    assert math.isclose(a["market_share"], 0.3) and math.isclose(a["self_share"], (0.1 + 0.1 + 0.1) / 3)
    assert a["hubs"][0][0] == "B" and math.isclose(a["hubs"][0][1], 1.6) and "C" not in a["received"]


def _hand_line(S: datetime, n: int = 6) -> tuple[dict, dict]:
    rng = np.random.default_rng(4)
    toks, closes = [], {}
    for j in range(n):
        sym = f"T{j}USDT"
        s_pct = 0.5 + 0.1 * j
        sc4 = math.sqrt(4) * s_pct / 100 * 1e4
        q = [-1.0, 0.1 * (j - 2), 1.0]
        toks.append({"symbol": sym, "present": True, "close_ref": 100.0, "s_pct": s_pct, "lsig2": math.log((s_pct / 100) ** 2),
                     "vol": {f"{h}h": {"v3": [x * math.sqrt(h) * s_pct for x in (0.06, 0.12, 0.3, 0.6, 1.1, 1.6, 2.0)],
                                       "har": [x * math.sqrt(h) * s_pct for x in (0.05, 0.1, 0.3, 0.65, 1.2, 1.7, 2.1)]} for h in (1, 4)},
                     "rel": {"4h": {"q_bp": [x * sc4 for x in q], "aci_bp": [(q[0] - 0.2) * sc4, (q[2] + 0.2) * sc4], "q_std": q,
                                    "aci_std": [q[0] - 0.2, q[2] + 0.2], "Q": 0.2, "alpha_t": 0.2, "scale_bp": sc4}}})
        closes[sym] = {xs_v3._us(S + timedelta(hours=h)): 100.0 * math.exp(rng.normal(0, 0.01 * math.sqrt(h))) for h in (1, 4)}
    toks.append({"symbol": "ABSUSDT", "present": False})
    return {"id": f"xs3-{S:%Y%m%dT%H}", "forecast_hour": S.isoformat(), "tokens": toks}, closes


def test_score_line_and_aci_update_by_hand():
    from scipy.stats import spearmanr

    S = datetime(2026, 11, 16, 4, tzinfo=UTC)
    line, closes = _hand_line(S)
    toks = [t for t in line["tokens"] if t["present"]]
    sc = xs_v3.score_line(line, 4, closes)
    R = np.array([math.log(closes[t["symbol"]][xs_v3._us(S + timedelta(hours=4))] / 100) for t in toks])
    rel = R - R.mean()
    assert sc["origin"] and sc["n"] == 6 and sc["rel"]["n"] == 6
    assert math.isclose(sc["rel"]["ic_v3"], spearmanr([t["rel"]["4h"]["q_bp"][1] for t in toks], rel).statistic, abs_tol=1e-12)
    assert math.isclose(sc["rel"]["ic_lowvol"], spearmanr([-t["lsig2"] for t in toks], rel).statistic, abs_tol=1e-12)
    rbp = 1e4 * rel
    assert sc["rel"]["hit_aci80"] == np.mean([t["rel"]["4h"]["aci_bp"][0] <= x <= t["rel"]["4h"]["aci_bp"][1] for t, x in zip(toks, rbp, strict=True)])
    a = np.abs(R) / (2 * np.array([t["s_pct"] / 100 for t in toks]))
    pv = np.mean([xs_v3.pinball7(a[j], [x / (100 * 2 * toks[j]["s_pct"] / 100) for x in toks[j]["vol"]["4h"]["v3"]]) for j in range(6)])
    assert math.isclose(sc["vol"]["pinball_v3"], pv, rel_tol=1e-12) and 0 <= sc["vol"]["cov_q90_v3"] <= 1
    s1 = xs_v3.score_line(line, 1, closes)
    assert "rel" not in s1 and s1["vol"]["n"] == 6
    # ACI: resolved once at a 4h origin, never at 24h (not 00 UTC), not twice
    from ai_trader.live.nn import OnlineACI

    st = {"as_of": S.isoformat(), "4h": {"applied_until": None, "tokens": {t["symbol"]: OnlineACI(80, 0.005, 500, 0.2, [0.1, 0.2]) for t in toks}},
          "24h": {"applied_until": None, "tokens": {}}}
    assert xs_v3.aci_update(st, line, 4, closes) == 6 and st["4h"]["applied_until"] == S.isoformat()
    for j, t in enumerate(toks):
        o, y = st["4h"]["tokens"][t["symbol"]], rel[j] / (t["rel"]["4h"]["scale_bp"] / 1e4)
        err = float(not (t["rel"]["4h"]["aci_std"][0] <= y <= t["rel"]["4h"]["aci_std"][1]))
        assert math.isclose(o.alpha_t, 0.2 + 0.005 * (0.2 - err)) and math.isclose(o.scores[-1], max(-1.0 - y, y - 1.0))
    assert xs_v3.aci_update(st, line, 4, closes) == 0 and xs_v3.aci_update(st, line, 24, closes) == 0
    line5 = {**line, "forecast_hour": (S + timedelta(hours=1)).isoformat()}
    assert xs_v3.aci_update(st, line5, 4, closes) == 0                        # 05:00 is not a 4h origin
    run = xs_v3.running([{"horizon": "4h", "origin": True, **sc}, {"horizon": "1h", **s1}])
    assert run["rel_4h"]["n_lines"] == 1 and run["vol_1h"]["n_lines"] == 1 and math.isclose(run["rel_4h"]["ic_v3"], sc["rel"]["ic_v3"])


def test_aci_warm_start_continues_pnn_aci():
    """With the whole stream in scores_tail, the warm start equals pnn.aci one step later (the last update replayed)."""
    from ai_trader.ml import pnn

    rng = np.random.default_rng(7)
    n = 300
    y = rng.standard_t(4, n)
    lo, hi = -1.1 + 0.1 * rng.standard_normal(n), 1.1 + 0.1 * rng.standard_normal(n)
    a = pnn.aci(np.r_[lo, 0.0], np.r_[hi, 0.0], np.r_[y, 0.0], 0.2, gamma=0.005, window=1000)
    spec = {"version": "t", "aci": {"level": 0.8, "gamma": 0.005, "window": 1000,
                                    "state": {h: {"X": {"alpha_last": float(a["alpha_t"][n - 1]), "scores_tail": pnn.cqr_scores(lo, hi, y).tolist()}}
                                              for h in ("4h", "24h")}}}
    st = xs_v3.new_aci_state(spec, datetime(2026, 11, 15, tzinfo=UTC))
    o = st["4h"]["tokens"]["X"]
    assert math.isclose(o.alpha_t, a["alpha_t"][n], abs_tol=1e-15) and math.isclose(o.Q(), a["Q"][n], abs_tol=1e-15)


# ------------------------------------------------------------------ the line (synthetic lake, frozen model)


def _paths(root: Path) -> xs_v3.PathsV3:
    return xs_v3.PathsV3(root, models_root=REPO)


@pytest.mark.skipif(not HAVE_V3, reason="frozen xs_v3 not present")
def test_run_line_point_in_time_synthetic(tmp_path):
    from ai_trader.storage.lake import Lake
    from tests.conftest import T0

    full = _xs_lake(tmp_path / "full")
    as_of = T0 + timedelta(days=160, hours=5, minutes=3)
    fixed = as_of + timedelta(minutes=1)
    lakes = {"full": full}
    for name, cut in (("part", as_of), ("later", as_of + timedelta(hours=5))):
        lk = Lake(root=tmp_path / name)
        for t in ("bars", "funding", "open_interest"):
            lk.write(t, full.read_asof(t, cut))
        lakes[name] = lk
    outs = []
    for name, lk in lakes.items():
        line, st, _, _ = xs_v3.run_line(as_of, paths=_paths(tmp_path / f"o_{name}"), lake=lk, logged_at=fixed, clock=lambda: 0.0)
        d = xs_v3.dump_state(st)
        d.pop("created")
        outs.append((line, d))
    assert outs[0] == outs[1] == outs[2]                       # rows known after as_of change nothing
    line = outs[0][0]
    S = T0 + timedelta(days=160, hours=5)
    assert line["id"] == f"xs3-{S:%Y%m%dT%H}" and line["forecast_hour"] == S.isoformat() and line["model_id"] == "xs.v3" and line["context_only"]
    assert line["data_max_ts"] <= as_of.isoformat() and "stale" not in line["flags"] and "model_fitted_after_as_of" in line["flags"]
    spec = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert line["models"]["version"] == spec["version"] and line["models"]["weights_sha256"]["v3_seed0.pt"] == spec["files_sha256"]["v3_seed0.pt"][:16]
    assert all(line["models"]["code_sha256_ok"].values()) and len(line["inputs_hash"]) == 20
    pres = [t for t in line["tokens"] if t["present"]]
    assert {t["symbol"] for t in pres} == set(TOKENS) and line["n_present"] == len(TOKENS) and line["n_symbols"] == 23
    assert set(line["absent"]) == set(spec["symbols"]) - set(TOKENS) and all(v == "no data" for v in line["absent"].values())
    for h in ("4h", "24h"):
        by = sorted(pres, key=lambda t: t[f"rank_{h}"])
        meds = [t["rel"][h]["q_bp"][1] for t in by]
        assert meds == sorted(meds, reverse=True) and sorted(t[f"rank_{h}"] for t in pres) == list(range(1, len(pres) + 1))
        for t in pres:
            q, a = t["rel"][h]["q_bp"], t["rel"][h]["aci_bp"]
            assert q[0] < q[1] < q[2] and a[0] <= a[1]
    low = sorted(pres, key=lambda t: t["rank_lowvol"])
    assert [t["lsig2"] for t in low] == sorted(t["lsig2"] for t in pres)
    for t in pres:
        for h in ("1h", "4h"):
            v = t["vol"][h]["v3"]
            assert all(b > a for a, b in itertools.pairwise(v)) and len(t["vol"][h]["har"]) == 7
        assert t["close_ref"] > 0 and len(t["xy"]) == 2
    att = line["attention"]
    assert att["keys"][-1] == "MARKET" and len(att["w"]) == len(pres) and all(math.isclose(sum(r), 1, abs_tol=5e-3) for r in att["w"])
    block = xs_v3.render_block(line)
    assert block[0].startswith("\n## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3") and len(block) <= 8


@pytest.mark.skipif(not (HAVE_V3 and HAVE_NETS), reason="frozen networks not present")
def test_v1_v21_outputs_byte_identical_with_or_without_v3(tmp_path):
    from ai_trader.live import hourly, nn_v2
    from ai_trader.live import nn as live
    from tests.conftest import T0
    from tests.test_live import _jump_bars

    lk = _xs_lake(tmp_path / "lake", n_hours=24 * 160, btc_bars=_jump_bars(24 * 160, 24 * 150 + 3))
    S0 = T0 + timedelta(days=150, hours=5)

    def hour(root, k, v3):
        S = S0 + timedelta(hours=k)
        return hourly.run_hour(S + timedelta(minutes=3), paths1=live.Paths(root, models_root=REPO), paths2=nn_v2.PathsV2(root, models_root=REPO),
                               paths3=xs_v3.PathsV3(root, models_root=REPO), v3=v3, logged_at=S + timedelta(minutes=3, seconds=5), lake=lk,
                               with_field=False, with_news=True, clock=lambda: 0.0)

    A, B = tmp_path / "A", tmp_path / "B"
    hour(A, 0, False)
    shutil.copytree(A, B)
    ks = (1, 2, 4, 5, 6)      # 06, 07, (08 skipped: no 4h bars for v1's k = 0 mod 4 construction), 09, 10, 11
    for k in ks:
        ra = hour(A, k, False)
        rb = hour(B, k, True)
        assert "v3_error" not in rb, rb.get("v3_error")
        assert "v2_error" not in ra and "v2_error" not in rb
    own = ("live_xs_v3", "xs_v3")
    files = sorted(p.relative_to(A) for p in A.rglob("*") if p.is_file() and p.name != "LATEST.md")
    assert any("live_nn_v2" in str(p) for p in files) and any(p.name == "live_aci_state.json" for p in files)
    for p in files:
        assert (A / p).read_bytes() == (B / p).read_bytes(), p                 # v1 and v2.1 untouched by v3
    extra = [q for q in sorted(p.relative_to(B) for p in B.rglob("*") if p.is_file() and p.name != "LATEST.md") if not (A / q).exists()]
    assert extra and all(any(o in str(p) for o in own) for p in extra)          # v3 writes only its own files
    p3 = xs_v3.PathsV3(B, models_root=REPO)
    lines = xs_v3.read_lines(p3)
    assert [x["id"] for x in lines] == [f"xs3-{S0 + timedelta(hours=k):%Y%m%dT%H}" for k in ks]
    sc = xs_v3.read_scores(p3)
    assert {s["horizon"] for s in sc} >= {"1h", "4h"} and all(s["scored_by"].startswith("xs3-") for s in sc)
    s4 = [s for s in sc if s["horizon"] == "4h"]
    assert s4 and s4[0]["id"] == lines[0]["id"] and s4[0]["rel"]["ic_v3"] is not None     # the 06 line's 4h window closed at 10:00
    assert lines[-1]["running"]["vol_1h"]["n_lines"] >= 3 and p3.state.is_file()
    assert lines[-1]["vs_v1"]["id"] == f"nn-{S0 + timedelta(hours=ks[-1]):%Y%m%dT%H}"
    md = (B / "research" / "live" / "LATEST.md").read_text(encoding="utf-8")
    assert "## Cross-section v3 (context only, not a signal" in md and "## v2.1 shadow" in md and len(md.splitlines()) <= 70
    assert "Cross-section v3" not in (A / "research" / "live" / "LATEST.md").read_text(encoding="utf-8")
    # a v3 failure changes nothing else and is reported
    C = tmp_path / "C"
    shutil.copytree(A, C)
    (C / "data" / "models" / "xs_v3" / "frozen").mkdir(parents=True)
    shutil.copy(FROZEN, C / "data" / "models" / "xs_v3" / "frozen" / "spec.json")
    S = S0 + timedelta(hours=8)     # 13:00 (12:00 is a k = 0 mod 4 hour)
    rc = hourly.run_hour(S + timedelta(minutes=3), paths1=live.Paths(C, models_root=REPO), paths2=nn_v2.PathsV2(C, models_root=REPO),
                         paths3=xs_v3.PathsV3(C), v3=True, logged_at=S + timedelta(minutes=3, seconds=5), lake=lk, with_field=False,
                         with_news=True, clock=lambda: 0.0)
    ra = hourly.run_hour(S + timedelta(minutes=3), paths1=live.Paths(A, models_root=REPO), paths2=nn_v2.PathsV2(A, models_root=REPO), v3=False,
                         logged_at=S + timedelta(minutes=3, seconds=5), lake=lk, with_field=False, with_news=True, clock=lambda: 0.0)
    assert "v3_error" in rc and "Cross-section v3 (context only): not run" in rc["latest"]
    for p in sorted(q.relative_to(A) for q in A.rglob("*") if q.is_file() and q.name != "LATEST.md"):
        assert (A / p).read_bytes() == (C / p).read_bytes(), p
    assert ra["line"] == rc["line"]


# ------------------------------------------------------------------ dashboard


def test_api_xs(tmp_path):
    import urllib.request

    from ai_trader.livedash import server as srv

    S = datetime(2026, 11, 16, 4, tzinfo=UTC)
    line, _ = _hand_line(S)
    line.update({"logged_at": (S + timedelta(minutes=3)).isoformat(), "n_present": 6})
    old = {**line, "id": "xs3-20261116T03", "forecast_hour": (S - timedelta(hours=1)).isoformat(), "logged_at": (S - timedelta(minutes=57)).isoformat()}
    d = tmp_path / "data" / "journal" / "live_xs_v3"
    d.mkdir(parents=True)
    (d / "2026-11-16.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(line) + "\n", encoding="utf-8")
    (d / "scores.jsonl").write_text(json.dumps({"id": old["id"], "horizon": "1h", "forecast_hour": old["forecast_hour"], "origin": True,
                                                "vol": {"pinball_v3": 0.14, "pinball_har": 0.15}}) + "\n", encoding="utf-8")
    dash = REPO / "research" / "live" / "dashboard"
    app = srv.App(tmp_path, models_root=REPO, static_dir=dash, fetch=lambda p: {"retCode": 0, "result": {"list": []}})
    s = srv.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{s.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/xs", timeout=30) as r:
            j = json.loads(r.read())
        assert j["available"] and j["line"]["id"] == line["id"] and j["n_lines"] == 2 and "not a signal" in j["note"]
        assert j["scores"][0]["pinball_v3"] == 0.14 and j["scores"][0]["ic_v3"] is None
        with urllib.request.urlopen(base + "/xs.js", timeout=30) as r:
            assert r.status == 200 and b"not a signal" in r.read().lower()
        with urllib.request.urlopen(base + "/", timeout=30) as r:
            html = r.read()
        assert b'id="xs-panel"' in html and b"xs.js" in html and b"Context only, not a signal" in html
    finally:
        s.shutdown()
        s.server_close()
    empty = srv.App(tmp_path / "none", models_root=REPO, static_dir=dash).xs()
    assert empty["available"] is False and empty["line"] is None
