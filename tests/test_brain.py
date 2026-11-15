"""The live brain (ai_trader.livedash.brain + research/live/dashboard/brain_core.js): the browser forward pass equals the
Python one and the stored live lines, the intra-hour recipe has no lookahead and starts from the last v1 line's inputs,
the snapshot recorder's limits, the H-BR1 checkpoint logger refusing late / early / duplicate / lookahead logs, and the
scorer's 300-hour verdict gate."""

import json
import math
import shutil
import subprocess
import threading
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ai_trader.livedash import brain as B

REPO = Path(__file__).resolve().parents[1]
PNN = REPO / "data" / "models" / "pnn"
HAVE_NET = (PNN / "frozen_1h.json").is_file() and (PNN / "wf_1h").is_dir()
JOURNAL = REPO / "data" / "journal" / "live_nn"
HAVE_JOURNAL = HAVE_NET and JOURNAL.is_dir() and any(JOURNAL.glob("20*.jsonl"))
DASH = REPO / "research" / "live" / "dashboard"
NODE = shutil.which("node")
CORE = DASH / "brain_core.js"

needs_net = pytest.mark.skipif(not HAVE_NET, reason="frozen 1h network not present")
needs_journal = pytest.mark.skipif(not HAVE_JOURNAL, reason="no live_nn journal")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


@pytest.fixture(scope="module")
def net():
    return B.export_net(PNN)


def _lines() -> list[dict]:
    rows = []
    for f in sorted(JOURNAL.glob("20*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("inputs", {}).get("1h") and all(v is not None for v in r["inputs"]["1h"].values())]
    return sorted(rows, key=lambda r: (r["forecast_hour"], r["logged_at"]))


def _node(script: str, payload) -> object:
    r = subprocess.run([NODE, "-e", script], input=json.dumps(payload), capture_output=True, text=True, timeout=120, cwd=str(REPO), check=False)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _js(body: str) -> str:
    return ("const C = require(" + json.dumps(str(CORE)) + "); const fs = require('fs'); const d = JSON.parse(fs.readFileSync(0, 'utf8'));"
            + body + "; process.stdout.write(JSON.stringify(out));")


# ------------------------------------------------------------------ export and forward pass


@needs_net
def test_export_has_every_seed_and_numpy_matches_torch(net):
    from ai_trader.live import nn

    n = nn.load_net("1h", str(PNN))
    assert len(net["seeds"]) == len(n.fits) == 2
    assert net["features"] == list(n.features) and np.allclose(net["std_mu"], n.mu) and np.allclose(net["std_sd"], n.sd)
    rng = np.random.default_rng(1)
    X = n.mu + n.sd * rng.normal(0, 1.5, size=(64, len(n.mu)))
    qt = n.quantiles(X)  # torch float32, 2-seed mean (the live line's path)
    qb = B.forward_batch(net, X)["q"]
    assert np.abs(qt - qb).max() < 2e-6
    for i in range(5):
        f1 = B.forward(net, X[i])
        assert np.allclose(f1["q"], qb[i], atol=1e-12)
        for k, s in enumerate(f1["seeds"]):  # node = bias + sum of its edges
            assert np.allclose(s["z"], np.asarray(net["seeds"][k]["enc"]["bias"]) + s["enc_edges"].sum(1))
            assert np.allclose(s["o"], np.asarray(net["seeds"][k]["head"]["bias"]) + s["head_edges"].sum(1))


@needs_journal
def test_numpy_forward_reproduces_stored_live_lines(net):
    lines = _lines()
    assert lines
    for ln in lines:
        x = [float(ln["inputs"]["1h"][c]) for c in net["features"]]
        q = B.forward(net, x)["q"]
        sc = ln["net"]["1h"]["sigma_ref_pct"]
        st = np.array([ln["net"]["1h"]["quantiles_pct"][k] for k in B.QNAMES])
        assert np.abs(q * sc - st).max() < 1e-6, ln["id"]


@needs_journal
@needs_node
def test_js_forward_equals_python_and_stored_lines(net):
    lines = _lines()[-48:]
    rng = np.random.default_rng(7)
    X = [[float(ln["inputs"]["1h"][c]) for c in net["features"]] for ln in lines]
    X += (np.asarray(net["std_mu"]) + np.asarray(net["std_sd"]) * rng.normal(0, 2.0, size=(32, 19))).tolist()
    X.append([None] * 19)  # missing inputs enter as 0 (standardised), as in the live line
    out = _node(_js("const out = d.X.map(x => { const f = C.forward(d.net, x.map(v => v === null ? NaN : v)); "
                    "return {q: f.q, z: f.seeds.map(s => Array.from(s.z)), o: f.seeds.map(s => Array.from(s.o)), "
                    "e: f.seeds.map(s => s.enc.map(r => Array.from(r)))}; })"), {"net": net, "X": X})
    for x, o in zip(X, out, strict=True):
        f = B.forward(net, [np.nan if v is None else v for v in x])
        assert np.abs(np.array(o["q"]) - f["q"]).max() < 1e-9
        for s, so in zip(f["seeds"], range(2), strict=True):
            assert np.abs(np.array(o["z"][so]) - s["z"]).max() < 1e-9
            assert np.abs(np.array(o["o"][so]) - s["o"]).max() < 1e-9
            assert np.abs(np.array(o["e"][so]) - s["enc_edges"]).max() < 1e-9
    for ln, o in zip(lines, out, strict=False):  # and the stored (torch float32) live quantiles, in %
        st = np.array([ln["net"]["1h"]["quantiles_pct"][k] for k in B.QNAMES])
        assert np.abs(np.array(o["q"]) * ln["net"]["1h"]["sigma_ref_pct"] - st).max() < 1e-6, ln["id"]


# ------------------------------------------------------------------ recipe (no lookahead)


def _base(net, line=None):
    if line is None:
        line = _lines()[-1]
    return B.base_from_line(line, net), line


def test_recipe_table_declares_every_feature():
    feats = [r["feature"] for r in B.RECIPE]
    assert len(feats) == 19 and len(set(feats)) == 19
    assert set(B.CHANGING) == {"lu_1", "lu_m6", "lu_m24", "jump24", "since_jump"}
    assert all(r["how"] and r["family"] in B.FAMILY for r in B.RECIPE)
    assert sorted(sum(B.FAMILY.values(), ())) == sorted(feats)


@needs_journal
def test_frozen_inputs_equal_the_last_v1_line(net):
    base, line = _base(net)
    assert base["line_id"] == line["id"] and base["inputs"] == [line["inputs"]["1h"][c] for c in net["features"]]
    assert base["p0"] == line["close"] and base["S_ms"] == B._ms(B._utc(line["forecast_hour"]))
    st = B.partial_state(base["p0"], base["S_ms"], base["S_ms"] + 2 * 60_000)  # no data yet, inside warm-up
    x, info = B.nowcast_inputs(base, st, base["S_ms"] + 2 * 60_000)
    assert info["warm"] and x == [float(v) for v in base["inputs"]]
    q = B.forward(net, x)["q"] * line["net"]["1h"]["sigma_ref_pct"]
    assert np.abs(q - np.array(base["frozen_quantiles_pct"])).max() < 1e-6


@needs_journal
def test_recipe_uses_only_data_up_to_now(net):
    base, _ = _base(net)
    S, p0 = base["S_ms"], base["p0"]
    rng = np.random.default_rng(3)
    ts = np.sort(rng.uniform(-600e3, 3600e3 + 600e3, 400)).astype(int) + S  # includes ticks before S and after the hour
    ticks = [(int(t), p0 * math.exp(rng.normal(0, 0.003))) for t in ts]
    candles = [[S + 60_000 * k - 60_000 * 5, 1, p0 * 1.01, p0 * 0.99, p0 * (1 + 0.001 * k), 1] for k in range(70)]
    for now in (S + 7 * 60_000 + 1234, S + 31 * 60_000, S + 59 * 60_000 + 59_000):
        past = [t for t in ticks if S <= t[0] <= now]
        st_all = B.partial_state(p0, S, now, ticks=ticks, candles=candles)
        st_past = B.partial_state(p0, S, now, ticks=past, candles=[c for c in candles if S <= c[0] and c[0] + 60_000 <= now])
        assert st_all == st_past
        assert st_all["t_last_ms"] <= now
        x_all, _ = B.nowcast_inputs(base, st_all, now)
        # changing the future (ticks after now, candles still open) changes nothing
        fut = [(t, p * 1.2) if t > now else (t, p) for t, p in ticks]
        cfut = [[c[0], 1, c[2] * 1.5, c[3] * 0.5, c[4] * 1.3, 1] if c[0] + 60_000 > now else c for c in candles]
        x_fut, _ = B.nowcast_inputs(base, B.partial_state(p0, S, now, ticks=fut, candles=cfut), now)
        assert x_all == x_fut
        # only the declared features move
        for c, a, b in zip(base["features"], x_all, base["inputs"], strict=True):
            if c not in B.CHANGING:
                assert a == b, c


@needs_journal
def test_recipe_values(net):
    base, _ = _base(net)
    S, p0, s2 = base["S_ms"], base["p0"], base["s2"]
    f = 0.5
    now = S + int(f * B.HOUR_MS)
    P = p0 * math.exp(0.004)
    st = B.partial_state(p0, S, now, ticks=[(S + 1000, p0 * math.exp(-0.001)), (now, P)])
    x, info = B.nowcast_inputs(base, st, now)
    i = {c: k for k, c in enumerate(base["features"])}
    u = 0.004 ** 2 / (f * s2)
    assert math.isclose(info["u"], u, rel_tol=1e-9) and math.isclose(x[i["lu_1"]], math.log(max(u, 1e-4)), rel_tol=1e-12)
    u_last = math.exp(base["inputs"][i["lu_1"]])
    assert math.isclose(math.exp(x[i["lu_m6"]]), math.exp(base["inputs"][i["lu_m6"]]) + (u - u_last) / 6, rel_tol=1e-9)
    big = p0 * math.exp(4.5 * math.sqrt(s2))  # an excursion beyond 4 s_pre fires the jump terms, even if it reverts
    stj = B.partial_state(p0, S, now, ticks=[(S + 5000, big), (now, p0)])
    xj, ij = B.nowcast_inputs(base, stj, now)
    assert ij["J"] == 1 and xj[i["jump24"]] == 1.0 and math.isclose(xj[i["since_jump"]], 1 / 168)


@needs_journal
@needs_node
def test_js_recipe_equals_python(net):
    base, _ = _base(net)
    S, p0 = base["S_ms"], base["p0"]
    rng = np.random.default_rng(11)
    cases = []
    for k in range(40):
        now = S + int(rng.uniform(0, B.HOUR_MS))
        sd = 0.002 if k % 4 else 0.02  # some cases cross the jump threshold
        ticks = [[S + int(t), p0 * math.exp(rng.normal(0, sd))] for t in np.sort(rng.uniform(-3e5, 3.9e6, 60))]
        candles = [[S + 60_000 * j, 1, p0 * math.exp(abs(rng.normal(0, sd))), p0 * math.exp(-abs(rng.normal(0, sd))), p0 * math.exp(rng.normal(0, sd)), 1]
                   for j in range(0, 60, 3)]
        cases.append({"now": now, "ticks": ticks, "candles": candles})
    out = _node(_js("const out = d.cases.map(c => { const st = C.partialState(d.base.p0, d.base.S_ms, c.now, c.ticks, c.candles); "
                    "const r = C.nowcastInputs(d.base, st, c.now); return {st: st, x: r[0], J: r[1].J, "
                    "rem: C.remainingPct(C.forward(d.net, r[0]).q, d.base, r[1].f)}; })"), {"net": net, "base": base, "cases": cases})
    for c, o in zip(cases, out, strict=True):
        st = B.partial_state(p0, S, c["now"], ticks=c["ticks"], candles=c["candles"])
        assert st == o["st"]
        x, info = B.nowcast_inputs(base, st, c["now"])
        assert np.allclose(x, o["x"], rtol=0, atol=1e-12) and info["J"] == o["J"]
        rem = B.remaining_pct(B.forward(net, x)["q"], base, info["f"])
        assert np.allclose(rem, o["rem"], rtol=0, atol=1e-12)


# ------------------------------------------------------------------ recorder limits


def _fake_snap(t_ms, inputs, line="nn-x", pad=0):
    return {"t_ms": t_ms, "line_id": line, "inputs": inputs, "pad": "x" * pad}


def test_snapshot_writer_rate_size_and_unchanged(tmp_path):
    w = B.SnapshotWriter(tmp_path, min_gap_s=1.0, max_day_bytes=5000)
    t = int(datetime(2026, 11, 15, 12, tzinfo=UTC).timestamp() * 1000)
    assert w.write(_fake_snap(t, [1.0])) == "ok"
    assert w.write(_fake_snap(t + 400, [2.0])) == "rate"         # < 1 s after the last snapshot
    assert w.write(_fake_snap(t + 1000, [1.0])) == "unchanged"   # inputs did not change
    assert w.write(_fake_snap(t + 1000, [2.0])) == "ok"
    k = 3
    while w.write(_fake_snap(t + 1000 * k, [float(k)], pad=900)) == "ok":
        k += 1
    assert w.refused["size"] == 1 and w.path(t).stat().st_size <= 5000
    lines = w.path(t).read_text(encoding="utf-8").splitlines()
    assert all(json.loads(x)["t_ms"] >= t for x in lines) and len(lines) == k - 1


@needs_journal
def test_snapshot_fits_the_daily_budget(net, tmp_path):
    base, _ = _base(net)
    now = base["S_ms"] + 20 * 60_000
    st = B.partial_state(base["p0"], base["S_ms"], now, ticks=[(now - 5, base["p0"] * 1.002)])
    x, info = B.nowcast_inputs(base, st, now)
    snap = B.snapshot(base, net, x, info, now, "test")
    size = len(json.dumps(snap, separators=(",", ":"))) + 1
    assert 86400 / 60 * size < 50e6 / 5            # one a minute: well under the 50 MB/day budget
    assert B.MAX_DAY_BYTES <= 50_000_000            # the hard cap even at one a second
    assert len(snap["seeds"]) == 2 and len(snap["seeds"][0]["enc_edges"]) == 4 and len(snap["seeds"][0]["enc_edges"][0]) == 19


# ------------------------------------------------------------------ H-BR1 checkpoints


@needs_journal
def test_checkpoint_logger_refuses_late_early_duplicate_and_lookahead(net, tmp_path):
    base, _ = _base(net)
    S, p0 = base["S_ms"], base["p0"]
    cp = S + 30 * 60_000
    st = B.partial_state(p0, S, cp, ticks=[(cp - 10_000, p0 * 1.003)])
    late = cp + int(B.LATE_S * 1000) + 1
    assert B.log_checkpoint(tmp_path, base, net, 30, st, cp, late) == ("late", None)
    assert B.log_checkpoint(tmp_path, base, net, 30, st, cp, cp - 1) == ("early", None)
    assert B.log_checkpoint(tmp_path, base, net, 30, st, cp + 60_000, cp + 61_000) == ("early", None)  # evaluated past the checkpoint
    st_future = B.partial_state(p0, S, cp + 30_000, ticks=[(cp + 20_000, p0 * 1.01)])
    assert B.log_checkpoint(tmp_path, base, net, 30, st_future, cp, cp + 30_000) == ("lookahead", None)
    assert B.log_checkpoint(tmp_path, base, net, 20, st, cp, cp + 1000)[0] == "bad_minute"
    assert not B.checkpoints_path(tmp_path).exists()
    res, row = B.log_checkpoint(tmp_path, base, net, 30, st, cp, cp + 5000)
    assert res == "ok" and row["minute"] == 30 and math.isclose(row["f"], 0.5) and row["lag_s"] == 5.0
    k = math.sqrt(0.5) * 100 * math.sqrt(base["s2"])
    assert np.allclose(list(row["frozen_rem_pct"].values()), np.array(base["frozen_quantiles_model"]) * k, rtol=1e-6)
    assert B.log_checkpoint(tmp_path, base, net, 30, st, cp, cp + 6000) == ("duplicate", None)
    assert len(B.read_checkpoints(tmp_path)) == 1


@needs_journal
def test_recorder_step_on_rest_candles(net, tmp_path):
    line = _lines()[-1]
    jd = tmp_path / "data" / "journal" / "live_nn"
    jd.mkdir(parents=True)
    (jd / f"{line['forecast_hour'][:10]}.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    from ai_trader.livedash.store import Store

    store = Store(tmp_path, REPO)
    S = B._ms(B._utc(line["forecast_hour"]))
    p0 = line["close"]
    candles = [[S + 60_000 * j, p0, p0 * 1.001, p0 * 0.999, p0 * (1 + 0.0002 * j), 1.0] for j in range(40)]
    rec = B.Recorder(tmp_path, store, PNN, fetch=None)
    out = rec.step(now_ms=S + 15 * 60_000 + 4000, candles=candles)  # candles up to minute 39 exist but are in the future
    assert out["status"] == "ok" and out["snapshot"] == "ok" and out["checkpoint_15"] == "ok"
    cps = B.read_checkpoints(tmp_path)
    assert len(cps) == 1 and cps[0]["price"] == candles[14][4]   # close of the minute ending at :15, nothing later
    snap = json.loads((tmp_path / "data" / "journal" / "brain" / f"{line['forecast_hour'][:10]}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert snap["t_ms"] == S + 15 * 60_000 and snap["price"] == candles[14][4]
    out2 = rec.step(now_ms=S + 31 * 60_000 + 3 * 60_000, candles=candles)  # 3 min late for :30
    assert out2["checkpoint_30"] == "late" and len(B.read_checkpoints(tmp_path)) == 1
    assert rec.step(now_ms=S + 2 * B.HOUR_MS, candles=candles)["status"] == "line_not_current"


# ------------------------------------------------------------------ scorer gate


def _cps_and_closes(n_hours: int, nowcast_better: bool):
    rng = np.random.default_rng(5)
    cps, closes = [], {}
    t0 = datetime(2026, 12, 1, tzinfo=UTC)
    q = dict(zip(B.QNAMES, [-1.2, -0.9, -0.45, 0.0, 0.45, 0.9, 1.2], strict=True))
    for h in range(n_hours):
        S = t0 + timedelta(hours=h)
        y = rng.normal(0, 0.3)
        closes[(S + timedelta(hours=1)).isoformat()] = 100.0 * math.exp(y / 100)
        for m in B.CHECKPOINT_MIN:
            tight = {k: v * 0.35 for k, v in q.items()}
            cps.append({"id": f"br-{h}-{m}", "S": S.isoformat(), "minute": m, "price": 100.0,
                        "nowcast_rem_pct": tight if nowcast_better else q, "frozen_rem_pct": q if nowcast_better else tight})
    return cps, closes


def test_scorer_gives_no_verdict_below_300_hours():
    cps, closes = _cps_and_closes(299, True)
    s = B.hbr1_summary(B.score_checkpoints(cps, closes))
    assert s["n_hours"] == 299 and s["verdict"] is None and s["mean_d"] < 0
    cps, closes = _cps_and_closes(300, True)
    s = B.hbr1_summary(B.score_checkpoints(cps, closes))
    assert s["n_hours"] == 300 and s["verdict"] == "supports"
    cps, closes = _cps_and_closes(320, False)
    assert B.hbr1_summary(B.score_checkpoints(cps, closes))["verdict"] == "contradicts"
    cps, closes = _cps_and_closes(10, True)
    del closes[max(closes)]   # an unresolved hour is not scored
    assert B.hbr1_summary(B.score_checkpoints(cps, closes))["n_hours"] == 9


def test_hour_closes_uses_only_matching_non_stale_lines():
    L = [{"forecast_hour": "2026-11-15T13:00:00+00:00", "last_bar_close": "2026-11-15T13:00:00+00:00", "close": 1.0, "flags": []},
         {"forecast_hour": "2026-11-15T14:00:00+00:00", "last_bar_close": "2026-11-15T13:00:00+00:00", "close": 2.0, "flags": ["stale"]},
         {"forecast_hour": "2026-11-15T15:00:00+00:00", "last_bar_close": "2026-11-15T14:00:00+00:00", "close": 3.0, "flags": []}]
    assert B.hour_closes(L) == {"2026-11-15T13:00:00+00:00": 1.0}


# ------------------------------------------------------------------ http


def _fake_fetch(params):
    raise AssertionError("no network in tests")


@needs_journal
def test_brain_routes_and_no_recorder_with_a_fake_fetcher(tmp_path):
    from ai_trader.livedash import server as srv

    line = _lines()[-1]
    jd = tmp_path / "data" / "journal" / "live_nn"
    jd.mkdir(parents=True)
    (jd / "2026-11-15.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    app = srv.App(tmp_path, models_root=REPO, static_dir=DASH, fetch=_fake_fetch)
    s = srv.make_server(app, "127.0.0.1", 0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{s.server_address[1]}"
    try:
        def get(p):
            with urllib.request.urlopen(base + p, timeout=20) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        for f in ("brain.html", "brain.js", "brain_core.js"):
            code, _ct, body = get("/" + f)
            assert code == 200 and len(body) > 1000
        assert b"Intra-hour nowcast, not the official hourly forecast" in get("/brain.html")[2]
        b = json.loads(get("/api/brain/base")[2])
        assert b["line_id"] == line["id"] and b["inputs"] == [line["inputs"]["1h"][c] for c in b["features"]]
        assert b["recorder"]["running"] is False and "nowcast" in b["label"]
        n = json.loads(get("/api/brain/net")[2])
        assert len(n["seeds"]) == 2 and n["features"] == b["features"]
        assert get("/")[0] == 200   # the monitor still works through the brain hook
    finally:
        s.shutdown()
        s.server_close()
    assert not (tmp_path / "data" / "journal" / "brain").exists()   # nothing recorded


def test_lock_is_exclusive_until_stale(tmp_path):
    assert B.acquire_lock(tmp_path, "a", now=1000.0)
    assert not B.acquire_lock(tmp_path, "b", now=1000.0 + B.LOCK_STALE_S - 1)
    assert B.acquire_lock(tmp_path, "a", now=1100.0)
    assert B.acquire_lock(tmp_path, "b", now=1100.0 + B.LOCK_STALE_S + 1)
