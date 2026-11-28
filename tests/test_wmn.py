"""WMN (program WMM2): tagging, PIT detector, event metrics, decay fits, gate and markout mechanics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ai_trader.research import wmn_data as wd
from ai_trader.research import wmn_gate as wg
from ai_trader.research import wmn_study as ws

DAY = 86_400_000
BAR = 300_000


def test_parse_sched_and_tags():
    assert wd.parse_sched("delisting the STORJUSDT Perpetual Contract at Nov 27, 2026, 9:00AM UTC.") == 1795770000000
    assert wd.parse_sched("on Jan 5, 2026, 12:30PM UTC") is not None
    assert wd.parse_sched("no time here") is None
    m = {"ATOM": ["ATOMUSDT"], "PEPE": ["1000PEPEUSDT"], "STORJ": ["STORJUSDT"]}
    syms = {"ATOMUSDT", "1000PEPEUSDT", "STORJUSDT"}
    assert wd.tag_text("Delisting of STORJUSDT Perpetual; Pepe (PEPE) and ATOM/USDT", m, syms) == ["STORJUSDT", "ATOMUSDT", "1000PEPEUSDT"]
    assert wd.tag_text("Bybit WSOT: ONE NEW TRUE event", m, syms) == []
    assert wd.norm_base("1000PEPE") == "PEPE" and wd.norm_base("SHIB1000") == "SHIB"


def test_classify_bybit():
    c = wd.classify_bybit
    assert c({"title": "Delisting of STORJUSDT Perpetual Contract", "type": {"key": "delistings"}}) == "delist_perp"
    assert c({"title": "Bybit to List KOR Protocol (KOR) on Spot", "type": {"key": "new_crypto"}}) == "listing_spot"
    assert c({"title": "STABLE Token Splash - prize pool", "type": {"key": "new_crypto"}}) == "launch_promo"
    assert c({"title": "TradFi stock listing: ANT CFDs now live", "type": {"key": "new_crypto"}}) == "tradfi"
    assert c({"title": "Bybit to Support TIA v10 Network Upgrade", "type": {"key": "maintenance_updates"}}) == "maintenance"


def _bars(n: int, seed: int = 0, vol: float = 0.001, start: int = 1735689600000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.standard_t(5, n) * vol
    c = 100 * np.exp(np.cumsum(r))
    return pl.DataFrame({"ts": start + BAR * np.arange(n), "o": c.astype(np.float32), "h": c.astype(np.float32),
                         "l": c.astype(np.float32), "c": c.astype(np.float32),
                         "turnover": (1000 + rng.random(n) * 100).astype(np.float32)})


def test_detector_point_in_time():
    btc = _bars(288 * 12, 1)
    alt = _bars(288 * 12, 2, vol=0.002)
    full = ws.detect(alt, btc)
    cut = 288 * 10
    part = ws.detect(alt.head(cut), btc.head(cut))
    a = full["z"].to_numpy()[:cut]
    b = part["z"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 288 * 5
    assert np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-9)
    # size: few detections on a no-jump t(5) diffusion with constant turnover
    assert full["D"].sum() <= 0.002 * full.height


def test_detector_flags_injected_jump_and_episodes():
    btc = _bars(288 * 10, 3)
    alt = _bars(288 * 10, 4, vol=0.002).with_row_index("i")
    k = 288 * 9
    c = alt["c"].to_numpy().astype(float)
    c[k:] *= np.exp(0.05)
    c[k + 3:] *= np.exp(0.04)
    tov = alt["turnover"].to_numpy().copy()
    tov[k] *= 10
    tov[k + 3] *= 10
    alt = alt.with_columns(pl.Series("c", c.astype(np.float32)), pl.Series("turnover", tov)).drop("i")
    det = ws.detect(alt, btc)
    assert bool(det["D"][k])
    ep = ws.episodes(det, "D")
    assert ep.filter((pl.col("ts") >= alt["ts"][k]) & (pl.col("ts") <= alt["ts"][k + 3])).height == 1
    assert int(ep["known_ts"][0] - ep["ts"][0]) == BAR


def test_event_metrics_variance_shock():
    n = 2000
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.001, n)
    t0i = 1200
    r[t0i:t0i + 24] *= 4.0
    ts = 1735689600000 + BAR * np.arange(n)
    w = ws.Window(ts, r, np.zeros(n), np.ones(n) * 100.0)
    m = ws.event_metrics(w, int(ts[t0i]) + 1000, market=True)
    assert m is not None
    assert m["ARV_h0_1"] > 2.0 and abs(m["ARV_pre"]) < 1.0
    assert m["ARV_h24_48"] < 1.0
    assert np.isfinite(m["_x"][1]) and m["_x"].size == 577


def test_cluster_mean_and_bh():
    y = np.array([1.0, 1.0, 3.0, 3.0])
    mu, se, n, G = ws.cluster_mean(y, np.array([1, 1, 2, 2]))
    assert mu == 2.0 and G == 2 and se > 0
    assert ws.bh([0.001, 0.2, 0.03, 0.5], 0.10) == [True, False, True, False]


def test_decay_fit_recovers_exponential():
    rng = np.random.default_rng(7)
    tau = ws.TAU_MIN
    m = 1.0 + 3.0 * np.exp(-tau / 60.0)
    Z = rng.standard_normal((600, 577)) * np.sqrt(m)
    fe = ws.decay_fit(Z, "exp")
    B, A, T = fe["theta"]
    assert 0.9 < B < 1.1 and 2.4 < A < 3.6 and 45 < T < 80
    fp = ws.decay_fit(Z, "pow")
    d = ws.decay_scores(Z, fe) - ws.decay_scores(Z, fp)
    assert d.mean() > -0.002  # exp is at least as good on exp data
    assert ws.half_life(fe) == pytest.approx(T * np.log(2))


def test_persistence_ratio():
    X = np.ones((10, 577))
    X[:, 1:13] = 3.0
    X[:, 49:] = 1.5
    assert ws.persistence_ratio(X) == pytest.approx(0.25)


def test_gate_point_in_time_and_ranges():
    p = wg.GateParams({"x_listing": wg.Kernel("pow", 5.0, [10.0, 1.0]), "jump": wg.Kernel("exp", 4.0, [30.0]),
                       "macro": wg.Kernel("exp", 2.0, [20.0]), "btc_jump": wg.Kernel("exp", 1.0, [20.0])})
    t = 1795000000000 + 60_000 * np.arange(600)
    kt = int(t[100]) + 20_000
    lam, pull, widen, lt, lm = wg.symbol_gate(p, t, [(kt, "x_listing")], [int(t[300])], [int(t[400])], [], 0.3)
    assert np.all(lt[:101] == 0)  # nothing before known_ts
    assert lt[101] > 0
    assert np.all(lt[300:305] < lt[305] + 10) and lt[304] < lt[306]  # episode active only from bar close
    assert np.all((widen >= 1) & (widen <= 2))
    assert pull[398] and pull[399] and not pull[397 - 10]  # macro pre-window
    assert np.all(np.diff(lt[110:290]) <= 1e-12)  # decays between shocks
    lam2, _, _ = wg.gate_values(t, np.full(t.size, 10.0), np.zeros(t.size), 0.0, np.zeros(t.size, bool))
    assert np.all(lam2 >= 3)


def test_gate_params_roundtrip(tmp_path):
    p = wg.GateParams({"news_pooled": wg.Kernel("exp", 1.5, [45.0])}, {"hn_story": "news_pooled"}, meta={"a": 1})
    f = tmp_path / "g.json"
    p.dump(f)
    q = wg.GateParams.load(f)
    assert q.kernel_for("hn_story").A == 1.5 and q.kernel_for("unknown").theta == [45.0]


def test_markout_signs():
    from ai_trader.research import wmn_markout as wm
    rng = np.random.default_rng(3)
    n = 6000
    t = 1735776000.0 + np.sort(rng.random(n) * 86400)
    d = rng.choice([1, -1], n)
    P = 100 + np.cumsum(d * 0.01)  # permanent impact: the efficient price moves with each taker's sign
    p = np.r_[100.0, P[:-1]]  # the maker is filled at the pre-impact price, so it is adversely selected
    tr = pl.DataFrame({"t": t, "d": d.astype(np.int8), "size": np.ones(n), "price": p, "notional": p})
    f = wm.day_features(tr, int(1735776000000), np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                        np.array([], dtype=np.int64), np.random.default_rng(0))
    assert f is not None and f.height > 1000
    assert np.nanmean(f["mo10"].to_numpy()) < 0
    assert f["t"].min() >= 1735776000.0 + 3600


def test_hawkes_filter_is_strictly_past_and_fit_recovers_news():
    from ai_trader.research import wmn_hawkes as wh
    x = np.zeros(10)
    x[3] = 1.0
    u = wh.ewfilt(x, 2.0, True)
    assert u[3] == 0 and u[4] == pytest.approx(1.0) and u[6] == pytest.approx(0.5)
    v = wh.ewfilt(x, 2.0, False)
    assert v[3] == pytest.approx(1.0)
    rng = np.random.default_rng(11)
    n = 200_000
    news = (rng.random(n) < 0.002).astype(float)
    un = wh.ewfilt(news, 12.0, False)
    lam = 0.001 + 0.05 * un
    y = (rng.random(n) < 1 - np.exp(-lam)).astype(np.int8)
    X = np.column_stack([np.full(n, 0.001), np.zeros(n), np.zeros(n), un, np.zeros(n), np.zeros(n)]).astype(np.float32)
    th, _ = wh.fit_linear(X, y, [0, 3])
    assert 0.7 < th[0] < 1.3 and 0.03 < th[1] < 0.07
