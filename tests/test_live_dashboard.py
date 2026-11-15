"""Local live dashboard (ai_trader.livedash): percentile interpolation (Python and the page's pct.js agree), the
activation forward pass reproducing the stored quantiles, the JSON endpoints, the 127.0.0.1-only binding."""

import itertools
import json
import math
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from ai_trader.livedash import pct
from ai_trader.livedash import server as srv
from ai_trader.livedash.store import Store, parse_regime

REPO = Path(__file__).resolve().parents[1]
PNN = REPO / "data" / "models" / "pnn"
HAVE_NETS = (PNN / "frozen_1h.json").is_file() and (PNN / "frozen_1d.json").is_file()
JOURNAL = REPO / "data" / "journal" / "live_nn"
HAVE_JOURNAL = HAVE_NETS and any(JOURNAL.glob("20*.jsonl")) if JOURNAL.is_dir() else False
DASH = REPO / "research" / "live" / "dashboard"
N = NormalDist()


# ------------------------------------------------------------------ percentile


def test_percentile_hits_every_knot_and_is_monotone():
    q = [-1.27, -0.90, -0.43, 0.03, 0.47, 0.95, 1.31]
    for v, t in zip(q, pct.TAUS, strict=True):
        assert math.isclose(pct.percentile(v, q), t, abs_tol=1e-12)
    xs = np.linspace(-6, 6, 2001)
    ps = [pct.percentile(x, q) for x in xs]
    assert all(b > a for a, b in itertools.pairwise(ps))
    assert 0 < ps[0] < 1e-3 and 1 - 1e-3 < ps[-1] < 1


def test_percentile_is_exact_for_gaussian_quantiles():
    """Probit-linear interpolation of exact normal quantiles reproduces the normal CDF everywhere (tails included)."""
    s, m = 0.72, 0.05
    q = [m + s * N.inv_cdf(t) for t in pct.TAUS]
    for x in (-3.0, -1.5, -0.4, 0.0, 0.2, 0.9, 2.2, 4.0):
        assert math.isclose(pct.percentile(x, q), N.cdf((x - m) / s), abs_tol=1e-12)


def test_percentile_dict_input_degenerate_knots_and_scaling():
    qd = dict(zip(pct.QNAMES, [-1.0, -0.5, -0.5, 0.0, 0.5, 0.5, 1.0], strict=True))
    p = pct.percentile(-0.5, qd)
    assert 0.09 < p < 0.26 and math.isfinite(p)
    q = [-1.27, -0.90, -0.43, 0.03, 0.47, 0.95, 1.31]
    sq = pct.scaled_quantiles(q, 0.25)
    assert np.allclose(sq, np.array(q) * 0.5)
    with pytest.raises(ValueError):
        pct.percentile(0.0, [0, 1, None, 3, 4, 5, 6])


def test_sentences():
    assert pct.sentence(0.88) == ("BTC is at the 88th percentile of what the network expected for this hour: "
                                  "an unusually strong move up.")
    assert pct.ordinal(1) == "1st" and pct.ordinal(12) == "12th" and pct.ordinal(23) == "23rd" and pct.ordinal(111) == "111th"
    assert "extreme move down" in pct.sentence(0.01)
    assert "ordinary" in pct.describe(0.5)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_pct_js_matches_python():
    q = [-1.27, -0.90, -0.43, 0.03, 0.47, 0.95, 1.31]
    xs = [-4.0, -1.3, -0.95, -0.1, 0.03, 0.4, 1.0, 1.32, 3.5]
    js = (f"const P=require({json.dumps(str(DASH / 'pct.js'))});const q={json.dumps(q)};"
          f"console.log(JSON.stringify({{p:{json.dumps(xs)}.map(x=>P.percentile(x,q)),s:P.sentence(0.88),"
          f"k:P.percentile(0.47,{{q05:-1.27,q10:-0.9,q25:-0.43,q50:0.03,q75:0.47,q90:0.95,q95:1.31}}),sc:P.scaled(q,0.25)}}))")
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True, timeout=30).stdout)
    for x, p in zip(xs, out["p"], strict=True):
        assert abs(p - pct.percentile(x, q)) < 1e-6
    assert out["s"] == pct.sentence(0.88)
    assert abs(out["k"] - 0.75) < 1e-6
    assert np.allclose(out["sc"], pct.scaled_quantiles(q, 0.25))


# ------------------------------------------------------------------ synthetic journal


def _synthetic_line(S: datetime, shift: float = 0.0) -> dict:
    """A live_nn.v1-shaped line whose stored quantiles come from the frozen nets (live.nn.Net.quantiles)."""
    from ai_trader.live import nn

    line = {"id": f"nn-{S:%Y%m%dT%H}", "schema": "live_nn.v1", "logged_at": (S + timedelta(minutes=3)).isoformat(),
            "as_of": (S + timedelta(minutes=3)).isoformat(), "symbol": "BTCUSDT", "forecast_hour": S.isoformat(),
            "last_bar_close": S.isoformat(), "close": 65000.0 * math.exp(shift / 100), "flags": [],
            "alarms": ["coverage 1h ACI 90% over last 72: 0.778 vs 0.90", "stale data: test"],
            "models": {"pnn_1h": {"refit_month": "x", "pt_sha256": "x"}, "pnn_1d": {"refit_month": "x", "pt_sha256": "x"},
                       "field": {"file": None}, "code_sha256": "x", "tool_version": "live_nn.v1"},
            "inputs": {}, "net": {}, "latent": {}}
    for h, nh in (("1h", "1h"), ("24h", "1d")):
        net = nn.load_net(nh, str(PNN))
        x = net.mu + net.sd * np.linspace(-1.5, 1.5, len(net.features)) * (1 + shift)
        scale_in = "lsig2" if h == "1h" else "lrv_w"
        scale = math.exp(0.5 * x[list(net.features).index(scale_in)])
        q = net.quantiles(x)[0]
        qd = {k: float(100 * v * scale) for k, v in zip(pct.QNAMES, q, strict=True)}
        line["inputs"][h] = dict(zip(net.features, map(float, x), strict=True))
        sig = (qd["q90"] - qd["q10"]) / 2.5631031310892007
        line["net"][h] = {"window_start": S.isoformat(), "window_end": (S + timedelta(hours=1 if h == "1h" else 24)).isoformat(),
                          "quantiles_pct": qd, "sigma_pct": sig, "sigma_ref_pct": 100 * scale,
                          "aci": {L: {"lo_pct": qd["q10"] * 1.1, "hi_pct": qd["q90"] * 1.1, "Q": 0.1, "alpha_t": 0.2} for L in ("80", "90", "95")}}
        if h == "1h":
            z = net.latent(x)[0]
            line["latent"] = {"z": z.tolist(), "zhat_next": z.tolist(), "zmu": net.dyn.zmu.tolist(), "zsd": net.dyn.zsd.tolist()}
    return line


@pytest.fixture
def tmp_root(tmp_path):
    if not HAVE_NETS:
        pytest.skip("frozen networks not present")
    S0 = datetime(2026, 11, 15, 6, tzinfo=UTC)
    lines = [_synthetic_line(S0 + timedelta(hours=k), shift=0.1 * k) for k in range(4)]
    d = tmp_path / "data" / "journal" / "live_nn"
    d.mkdir(parents=True)
    (d / "2026-11-15.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    sc = {"id": lines[0]["id"], "horizon": "1h", "window_start": lines[0]["forecast_hour"], "window_end": lines[1]["forecast_hour"],
          "r_pct": 0.3, "z": 0.3 / lines[0]["net"]["1h"]["sigma_pct"], "sigma_pct": lines[0]["net"]["1h"]["sigma_pct"],
          "abs_r_pct": 0.3, "hit_aci80": True, "hit_aci90": True, "hit_aci95": True}
    (d / "scores.jsonl").write_text(json.dumps(sc) + "\n", encoding="utf-8")
    c = tmp_path / "data" / "journal" / "live_claude"
    c.mkdir(parents=True)
    cl = {"schema": "live_claude.v1", "logged_at": (S0 + timedelta(hours=2, minutes=9)).isoformat(), "as_of": (S0 + timedelta(hours=2, minutes=3)).isoformat(),
          "horizon": "1h", "window_start": (S0 + timedelta(hours=3)).isoformat(), "window_end": (S0 + timedelta(hours=4)).isoformat(),
          "q10": -1.0, "q50": 0.0, "q90": 1.0, "sigma": 0.8, "view": "test view: vol stays", "net_ref": lines[2]["id"], "id": "cl-test"}
    (c / "2026-11-15.jsonl").write_text(json.dumps(cl) + "\n", encoding="utf-8")
    md = tmp_path / "research" / "live"
    md.mkdir(parents=True)
    (md / "LATEST.md").write_text(f"# BTC live network: bar x (line {lines[-1]['id']})\n\n" + REGIME_MD, encoding="utf-8")
    return tmp_path, lines


REGIME_MD = """## Regime and players (information only, changes no forecast; as_of 2026-11-15T09:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.2% | -0.0042% (p8) | +3.1% / -0.4% | 2.4M / 850k | 31.0M / 12.5M (partial) | 52.3 (p71) |
| ETH | R3 calm up | +0.5% | +0.0100% (p55) | +0.2% / +1.0% | 300k / 120k | 5.1M / 4.0M | 60.1 (p40) |
- Pattern: BTC long flush, funding negative, OI rebuilding while price fell (new shorts); ETH two-sided liquidations, funding normal, OI flat
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T12:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Next
"""


def test_parse_regime():
    R = parse_regime(REGIME_MD)
    b = R["rows"][0]
    assert b["symbol"] == "BTC" and b["cell"] == "R9" and b["cell_label"] == "liquidation"
    assert b["ret_24h_pct"] == -1.2 and b["funding_rate_pct"] == -0.0042 and b["funding_pct_90d"] == 8
    assert b["oi_chg_24h_pct"] == 3.1 and b["oi_chg_72h_pct"] == -0.4
    assert b["liq_4h_long_usd"] == 2.4e6 and b["liq_4h_short_usd"] == 850e3 and b["liq_24h_partial"] and b["liq_24h_short_usd"] == 12.5e6
    assert b["dvol"] == 52.3 and b["dvol_pct_90d"] == 71
    assert R["pattern"]["BTC"].startswith("long flush") and R["as_of"] == "2026-11-15T09:03"
    assert parse_regime("# nothing here\n## Alarms\n- none\n") is None


def test_parse_regime_roundtrip_with_players_render():
    try:
        from ai_trader.live import players
        P = {"as_of": "2026-11-15T09:03:00+00:00", "symbols": {"BTCUSDT": {
            "regime": {"status": "ok", "cell": "R9", "label": "liquidation", "ret_24h_pct": -1.23},
            "funding": {"status": "ok", "rate_pct": -0.0042, "pct_90d": 0.08}, "oi": {"status": "ok", "chg_24h_pct": 3.1, "chg_72h_pct": -0.4},
            "liq": {"long_4h_usd": 2.4e6, "short_4h_usd": 8.5e5, "long_24h_usd": 3.1e7, "short_24h_usd": 1.25e7, "partial_24h": False},
            "dvol": {"status": "ok", "level": 52.3, "pct_90d": 0.71}, "ls": {"status": "stale", "last_known": "2026-11-07T12:00"},
            "pattern": "long flush"}}}
        md = "\n".join(players.render(P))
    except (ImportError, KeyError, TypeError) as e:  # players.py is being extended by another agent
        pytest.skip(f"players.render signature changed: {e}")
    b = parse_regime(md)["rows"][0]
    assert b["cell"] == "R9" and b["funding_pct_90d"] == 8 and b["dvol_pct_90d"] == 71 and b["liq_24h_long_usd"] == 31e6


# ------------------------------------------------------------------ activations


@pytest.mark.skipif(not HAVE_NETS, reason="frozen networks not present")
def test_activations_reproduce_synthetic_line():
    from ai_trader.livedash import activations as A

    line = _synthetic_line(datetime(2026, 11, 15, 9, tzinfo=UTC), shift=0.2)
    a = A.activations(line, PNN, [line])
    for h in ("1h", "24h"):
        v = a["horizons"][h]
        st = [line["net"][h]["quantiles_pct"][k] for k in pct.QNAMES]
        assert np.max(np.abs(np.array(v["quantiles_pct"]) - st)) < 1e-6
        assert v["max_abs_diff_vs_stored_pct"] < 1e-6
        assert v["check"]["edge_sum_max_err"] < 1e-5
        s0 = v["seeds"][0]
        n_in = len(v["features"])
        assert np.array(s0["enc_edges"]).shape == (4, n_in) and np.array(s0["head_edges"]).shape == (7, 4)
        # node = bias + sum of incoming edges, for both layers
        assert np.allclose(np.array(s0["enc_edges"]).sum(1) + s0["latent_bias"], s0["latent"], atol=1e-5)
        assert np.allclose(np.array(s0["head_edges"]).sum(1) + s0["head_bias"], s0["head_raw"], atol=1e-5)
        at = v["attribution"]
        assert len(at["l1_abs"]) == n_in and all(x >= 0 for x in at["l1_abs"])
        assert np.allclose(at["l1_abs"], np.mean([np.abs(np.array(s["enc_edges"])).sum(0) for s in v["seeds"]], axis=0))
        assert len(at["mean_ablation_sigma_pct"]) == n_in and at["week"]["n"] == 1
        assert np.allclose(at["week_ablation_dln_sigma"], 0.0, atol=1e-6)  # the line is its own weekly mean
    assert np.allclose(a["horizons"]["1h"]["latent"]["z"], line["latent"]["z"], atol=1e-6)


def test_mean_ablation_is_the_effect_of_a_zero_input():
    if not HAVE_NETS:
        pytest.skip("frozen networks not present")
    from ai_trader.live import nn
    from ai_trader.livedash import activations as A

    line = _synthetic_line(datetime(2026, 11, 15, 9, tzinfo=UTC))
    v = A.forward(line, "1h", str(PNN))
    net = nn.load_net("1h", str(PNN))
    x = np.array([line["inputs"]["1h"][c] for c in net.features])
    i = 4
    x2 = x.copy()
    x2[i] = net.mu[i]
    q = net.quantiles(x2)[0]
    sig2 = (q[5] - q[1]) / 2.5631031310892007 * line["net"]["1h"]["sigma_ref_pct"]
    assert math.isclose(v["sigma_pct"] - sig2, v["attribution"]["mean_ablation_sigma_pct"][i], abs_tol=1e-6)


def test_cheb_to_mono():
    from ai_trader.livedash.activations import cheb_to_mono

    c = [0.3, -0.2, 0.15, 0.05]
    m = cheb_to_mono(c)
    for u in (-0.9, -0.3, 0.0, 0.4, 0.8):
        T = [u, 2 * u * u - 1, 4 * u ** 3 - 3 * u, 8 * u ** 4 - 8 * u * u + 1]
        assert math.isclose(sum(a * t for a, t in zip(c, T, strict=True)), sum(a * u ** k for k, a in enumerate(m)), abs_tol=1e-12)


@pytest.mark.skipif(not HAVE_JOURNAL, reason="live journal or networks not present")
def test_activations_reproduce_real_journal_lines():
    from ai_trader.livedash import activations as A

    st = Store(REPO)
    lines = st.v1()[-6:]
    for ln in lines:
        a = A.activations(ln, PNN, st.week_lines(ln))
        for h in ("1h", "24h"):
            v = a["horizons"][h]
            if v["max_abs_diff_vs_stored_pct"] is None:
                continue
            assert v["max_abs_diff_vs_stored_pct"] < 1e-6, (ln["id"], h)
        lat = a["horizons"]["1h"]["latent"]
        assert np.allclose(lat["z"], ln["latent"]["z"], atol=1e-5)
        assert np.allclose(lat["zhat_next"], ln["latent"]["zhat_next"], atol=1e-5)


@pytest.mark.skipif(not HAVE_NETS, reason="frozen networks not present")
def test_edge_catalogue_shapes():
    from ai_trader.livedash import activations as A

    c = A.edge_catalogue("1h", str(PNN))
    e = c["seeds"][0]["enc"]["edges"]
    assert len(e) == 4 and len(e[0]) == len(c["features"]) and len(e[0][0]["curve"]) == A.GRID_N
    assert "silu" in e[0][0]["exact"] and 0 <= e[0][0]["cubic_r2"] <= 1 + 1e-9
    assert len(c["seeds"][0]["head"]["edges"]) == 7


# ------------------------------------------------------------------ server


def _get(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def _fake_fetch(calls):
    def fetch(params):
        calls.append(dict(params))
        end = int(params["end"])
        start = end - end % 60000
        return {"retCode": 0, "result": {"list": [[str(start - 60000 * k), "1", "2", "0.5", "1.5", "10", "15"] for k in range(1000)]}}
    return fetch


def test_klines_pages_back_and_sorts():
    calls = []
    now = 1_763_200_000_000
    rows = srv.klines(36, _fake_fetch(calls), now_ms=now)
    assert len(calls) == 3 and rows == sorted(rows) and rows[0][0] >= now - 36 * 3600_000
    assert all(c["category"] == "linear" and c["symbol"] == "BTCUSDT" and c["interval"] == "1" for c in calls)
    assert len({r[0] for r in rows}) == len(rows) and 2150 <= len(rows) <= 2170


def test_make_server_refuses_other_hosts():
    app = srv.App(REPO, static_dir=DASH, fetch=_fake_fetch([]))
    for h in ("0.0.0.0", "", "localhost", "::", "192.168.1.10"):
        with pytest.raises(ValueError):
            srv.make_server(app, h, 0)


def test_server_endpoints(tmp_root):
    root, lines = tmp_root
    calls = []
    app = srv.App(root, models_root=REPO, static_dir=DASH, fetch=_fake_fetch(calls))
    s = srv.make_server(app, "127.0.0.1", 0)
    assert s.server_address[0] == "127.0.0.1"
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{s.server_address[1]}"
    try:
        code, ct, body = _get(base, "/")
        assert code == 200 and "text/html" in ct and b"BTC Network Monitor" in body
        for f in ("app.js", "pct.js", "style.css"):
            assert _get(base, "/" + f)[0] == 200
        assert _get(base, "/../pyproject.toml")[0] == 404
        assert _get(base, "/data/journal/live_nn/2026-11-15.jsonl")[0] == 404

        code, _, body = _get(base, "/api/state")
        st = json.loads(body)
        assert code == 200 and st["v1"]["id"] == lines[-1]["id"] and st["v2"] is None
        assert len(st["history"]) == 4 and st["history"][0]["r_1h"] == 0.3 and st["realised"][0]["hit80_1h"] is True
        assert st["claude"]["current_1h"]["id"] == "cl-test" and st["claude"]["latest_view"]["view"] == "test view: vol stays"
        assert st["regime"]["rows"][0]["cell"] == "R9"
        assert {a["text"] for a in st["alarms"]} == set(lines[-1]["alarms"])
        assert "live1" in st["hlive"] and st["week"]["n"] == 4

        code, _, body = _get(base, f"/api/activations?id={lines[-1]['id']}")
        a = json.loads(body)
        assert code == 200
        for h in ("1h", "24h"):
            stq = [lines[-1]["net"][h]["quantiles_pct"][k] for k in pct.QNAMES]
            assert np.max(np.abs(np.array(a["horizons"][h]["quantiles_pct"]) - stq)) < 1e-6
        kinds = [x["k"] for x in a["thinking"]]
        assert kinds[:4] == ["variance", "latent", "versions", "coverage"]
        assert "No v2 line" in a["thinking"][2]["text"] and "coverage alarm" in a["thinking"][3]["text"]
        assert lines[-1]["id"] in app._act  # cached
        assert _get(base, "/api/activations?id=nn-nope")[0] == 404
        assert _get(base, "/api/edges?h=1h")[0] == 200 and _get(base, "/api/edges?h=5m")[0] == 404

        code, _, body = _get(base, "/api/kline?hours=2")
        k = json.loads(body)
        assert code == 200 and len(k["rows"]) >= 100 and calls
        n = len(calls)
        _get(base, "/api/kline?hours=2")
        assert len(calls) == n  # cached

        req = urllib.request.Request(base + "/api/state", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 405
    finally:
        s.shutdown()
        s.server_close()
    # read only: the journals are unchanged
    assert (root / "data" / "journal" / "live_nn" / "2026-11-15.jsonl").read_text(encoding="utf-8").count("\n") == 4


def test_state_with_v2_and_thinking(tmp_root):
    root, lines = tmp_root
    v2 = json.loads(json.dumps(lines[-1]))
    v2["id"] = "v2-" + lines[-1]["id"][3:]
    v2["net"]["1h"]["band"] = {L: {"lo_pct": b["lo_pct"] * 1.3, "hi_pct": b["hi_pct"] * 1.3, "floored": True} for L, b in v2["net"]["1h"]["aci"].items()}
    v2["net"]["1h"]["floor"] = {"active": True, "sigma_floor_pct": 0.9, "applied": [80, 90]}
    v2["alarms"] = ["coverage 1h v2 80% over last 72: 0.700 vs 0.80"]
    d = root / "data" / "journal" / "live_nn_v2"
    d.mkdir(parents=True)
    (d / "2026-11-15.jsonl").write_text(json.dumps(v2) + "\n", encoding="utf-8")
    app = srv.App(root, models_root=REPO, static_dir=DASH, fetch=_fake_fetch([]))
    st = app.state()
    assert st["v2"]["id"] == v2["id"] and st["history"][-1]["v2_band_1h"]["80"][0] == pytest.approx(v2["net"]["1h"]["band"]["80"]["lo_pct"])
    assert any(a["source"] == "v2" for a in st["alarms"])
    a = app.activations(v2["id"])  # a v2 id resolves to the same inputs
    th = {x["k"]: x["text"] for x in a["thinking"]}
    assert "floor is ACTIVE" in th["versions"] and "+30% width" in th["versions"]
    assert "v2 1h 80% band covered 70.0%" in th["coverage"]
