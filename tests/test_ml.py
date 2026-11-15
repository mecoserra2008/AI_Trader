from datetime import timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")

from ai_trader.ml.features import ML_FEATURES, Standardizer, add_cross_section, symbol_features  # noqa: E402
from ai_trader.ml.interpret import atom_recovery, fit_sae, integrated_gradients, shap_interactions, shap_main  # noqa: E402
from ai_trader.ml.models import fit_mlp, fit_xgb, val_split  # noqa: E402
from ai_trader.research.frame import add_robust_vol, enrich_bars  # noqa: E402
from tests.conftest import T0, make_bars  # noqa: E402


class FakeLake:
    """read_asof for funding (8h prints) and 4h OI; rows with known_ts <= as_of only."""

    def __init__(self, days: int, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        ft = [T0 + timedelta(hours=8 * i) for i in range(days * 3)]
        self.funding = pl.DataFrame({"known_ts": ft, "rate": rng.normal(1e-4, 1e-4, len(ft))}).with_columns(
            pl.col("known_ts").dt.replace_time_zone("UTC"))
        ot = [T0 + timedelta(hours=4 * i, seconds=5) for i in range(days * 6)]
        self.oi = pl.DataFrame({"known_ts": ot, "open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ot))))}).with_columns(
            pl.col("known_ts").dt.replace_time_zone("UTC"))

    def read_asof(self, table, as_of, symbols=None, filters=None, start=None):
        df = self.funding if table == "funding" else self.oi
        return df.filter(pl.col("known_ts") <= as_of)


def _frames(n_days: int, seed: int = 0):
    b1 = make_bars(24 * n_days, freq_h=1, seed=seed)
    f1 = add_robust_vol(enrich_bars(b1))
    b4 = (b1.with_columns(pl.col("event_ts").dt.truncate("4h").alias("g")).group_by("g", maintain_order=True)
          .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
               pl.col("turnover").sum(), pl.col("close_ts").last(), pl.col("known_ts").last())
          .rename({"g": "event_ts"}))
    f4 = add_robust_vol(enrich_bars(b4))
    return f1, f4


def _cut(f: pl.DataFrame, t):
    return add_robust_vol(enrich_bars(f.filter(pl.col("event_ts") < t).select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")))


def test_feature_builder_is_causal():
    n_full, n_cut = 420, 330
    f1, f4 = _frames(n_full)
    lake = FakeLake(n_full)
    t_full = T0 + timedelta(days=n_full)
    t_cut = T0 + timedelta(days=n_cut)
    full = symbol_features("TESTUSDT", f1, f4, lake, t_full)
    part = symbol_features("TESTUSDT", _cut(f1, t_cut), _cut(f4, t_cut), lake, t_cut)
    feats = [c for c in ML_FEATURES if c in part.columns]
    assert len(feats) >= 40
    # rows whose decision time is before the cut: identical features whether or not the future is appended
    a = part.filter(pl.col("T") <= t_cut).select("date", *feats).sort("date")
    b = full.join(a.select("date"), on="date", how="semi").select("date", *feats).sort("date")
    assert a.height > 200 and a.height == b.height
    for c in feats:
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        ok = ~np.isnan(x)
        assert np.allclose(x[ok], y[ok], rtol=1e-9, atol=1e-12), c
    assert int(a["vol_pct_1y"].is_not_null().sum()) > 50 and int(a["funding_z_90d"].is_not_null().sum()) > 50
    # targets are the next day's realised variance
    rv = full["rv"].to_numpy().astype(float)
    assert np.allclose(full["rv_next"].to_numpy()[:-1].astype(float), rv[1:], equal_nan=True)


def test_cross_section_is_causal():
    f1, f4 = _frames(200)
    lake = FakeLake(200)
    t_full = T0 + timedelta(days=200)
    one = symbol_features("TESTUSDT", f1, f4, lake, t_full)
    syms = ["BTCUSDT", "A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    rows = [one.with_columns(pl.lit(s).alias("symbol"), (pl.col("lrv_d") + float(rng.normal())).alias("lrv_d")) for s in syms]
    daily = pl.concat(rows)
    cut = (T0 + timedelta(days=150)).date()
    full = add_cross_section(daily, cut)
    part = add_cross_section(daily.filter(pl.col("date") < cut), cut)
    cols = ["mkt_lrv_d", "mkt_lrv_w", "mkt_ret_1d_z", "btc_ret_1d_z", "x_ret7_volpct", "x_ret1_fundz"]
    a = part.sort("date", "symbol").select(*cols).to_numpy()
    b = full.filter(pl.col("date") < cut).sort("date", "symbol").select(*cols).to_numpy()
    assert np.allclose(a.astype(float), b.astype(float), equal_nan=True)


def test_standardizer_uses_training_stats_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    s = Standardizer().fit(X[:60])
    Z = s.transform(X)
    assert np.allclose(Z[:60].mean(0), 0, atol=1e-12) and not np.allclose(Z[60:].mean(0), 0, atol=1e-3)
    X2 = X.copy()
    X2[60:] *= 100  # changing test rows does not change the transform
    assert np.allclose(Standardizer().fit(X2[:60]).transform(X2)[:60], Z[:60])


def _toy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 0.8 * X[:, 0] + 0.5 * np.tanh(X[:, 1] * X[:, 2]) + 0.1 * rng.normal(size=n)
    return X, y


def test_mlp_training_is_deterministic_with_seed():
    X, y = _toy()
    a = fit_mlp(X[:1000], y[:1000], X[1000:1200], y[1000:1200], "vol", seed=3, cfg={"hidden": (16, 8), "max_epochs": 15, "batch": 64})
    b = fit_mlp(X[:1000], y[:1000], X[1000:1200], y[1000:1200], "vol", seed=3, cfg={"hidden": (16, 8), "max_epochs": 15, "batch": 64})
    c = fit_mlp(X[:1000], y[:1000], X[1000:1200], y[1000:1200], "vol", seed=4, cfg={"hidden": (16, 8), "max_epochs": 15, "batch": 64})
    pa, pb, pc = a.predict(X[1200:]), b.predict(X[1200:]), c.predict(X[1200:])
    assert np.array_equal(pa, pb)
    assert not np.allclose(pa, pc)
    assert np.corrcoef(pa, y[1200:])[0, 1] > 0.8
    ig = integrated_gradients([a], X[1200:1250] * 1.0, steps=64)
    # completeness: attributions sum to f(x) - f(0) (in log-RV units)
    f0 = a.predict(np.zeros((1, 6)))[0]
    assert np.allclose(ig.sum(1), pa[:50] - f0, atol=0.05 * np.abs(pa[:50] - f0).max() + 1e-3)


def test_treeshap_contributions_sum_to_prediction():
    X, y = _toy(seed=1)
    names = [f"f{i}" for i in range(6)]
    m = fit_xgb(X[:1000], y[:1000], X[1000:1200], y[1000:1200], "vol", cfg={"max_rounds": 200}, names=names)
    Z = X[1200:1300]
    pred = m.predict(Z, output_margin=True)
    c = shap_main(m, Z)
    assert c.shape == (100, 7)
    assert np.allclose(c.sum(1), pred, atol=1e-4)
    it = shap_interactions(m, Z)
    assert it.shape == (100, 7, 7)
    assert np.allclose(it.sum(2), c, atol=1e-4)
    # the planted interaction (f1 x f2) is the largest off-diagonal pair
    A = np.abs(it[:, :6, :6]).sum(0)
    np.fill_diagonal(A, 0)
    i, j = np.unravel_index(np.argmax(A + A.T), A.shape)
    assert {i, j} == {1, 2}


def test_sae_recovers_planted_sparse_dictionary():
    rng = np.random.default_rng(0)
    d, k_true, n = 16, 24, 12000
    D = rng.normal(size=(d, k_true))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    Zt = np.zeros((n, k_true))
    for i in range(n):
        idx = rng.choice(k_true, size=2, replace=False)
        Zt[i, idx] = rng.uniform(0.5, 1.5, size=2)
    X = Zt @ D.T + 0.3
    sae = fit_sae(X, m=4 * d, l1=0.3, steps=6000, batch=256, lr=2e-3, seed=0)
    rec = atom_recovery(D, sae)
    assert np.mean(rec > 0.9) >= 0.9, rec
    met = sae.metrics(X)
    assert met["fvu"] < 0.1 and met["l0"] < 12


def test_val_split_purges_label_overlap():
    t0 = np.datetime64("2024-01-01T00:00:00", "us")
    T = np.repeat(t0 + np.arange(100) * np.timedelta64(86400 * 1_000_000, "us"), 3)
    fit, va = val_split(T, 0.15)
    assert not (fit & va).any()
    assert len(np.unique(T[va])) == 15
    assert T[fit].max() + np.timedelta64(86400 * 1_000_000, "us") < T[va].min()
