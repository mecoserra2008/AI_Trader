"""Cross-sectional network v3 (ml.xs_net, research.xs_network_study): equivariance, masking, panel look-ahead,
relative-return target maths and pairwise consistency."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.ml import xs_net as xs

N_FEAT = len(xs.feature_names())


def _model(kind="kan", mix=True, seed=0):
    import torch

    torch.manual_seed(seed)
    return xs.make(N_FEAT, len(xs.MKT_FEATURES), len(xs.SECTOR_ORDER), kind=kind, mix=mix)


def _inputs(B=4, N=7, seed=1):
    import torch

    rng = np.random.default_rng(seed)
    X = torch.as_tensor(rng.normal(size=(B, N, N_FEAT)), dtype=torch.float32)
    M = torch.as_tensor(rng.normal(size=(B, len(xs.MKT_FEATURES))), dtype=torch.float32)
    P = torch.ones(B, N, dtype=torch.bool)
    S = torch.as_tensor(rng.integers(0, len(xs.SECTOR_ORDER), N), dtype=torch.long)
    return X, M, P, S


# ------------------------------------------------------------------ network


@pytest.mark.parametrize("kind,mix", [("kan", True), ("mlp", True), ("kan", False)])
def test_permutation_equivariance(kind, mix):
    pytest.importorskip("torch")
    m = _model(kind, mix)
    X, M, P, S = _inputs()
    P[0, 2] = False
    perm = np.random.default_rng(3).permutation(X.shape[1])
    with __import__("torch").no_grad():
        v, r = m(X, M, P, S)
        v2, r2 = m(X[:, perm], M, P[:, perm], S[perm])
    assert np.allclose(v[:, perm].numpy(), v2.numpy(), atol=1e-5)
    assert np.allclose(r[:, perm].numpy(), r2.numpy(), atol=1e-5)


@pytest.mark.parametrize("kind", ["kan", "mlp"])
def test_absent_tokens_do_not_leak(kind):
    """Changing an absent token's features (or adding absent padding) leaves the present tokens' outputs unchanged."""
    torch = pytest.importorskip("torch")
    m = _model(kind, True)
    X, M, P, S = _inputs()
    P[:, 1] = False
    P[2, 4] = False
    with torch.no_grad():
        v, r = m(X, M, P, S)
        X2 = X.clone()
        X2[:, 1] = 100.0 * torch.randn_like(X2[:, 1])
        X2[2, 4] = -50.0
        v2, r2 = m(X2, M, P, S)
        # padding: append 3 absent tokens
        Xp = torch.cat([X, torch.randn(X.shape[0], 3, X.shape[2])], 1)
        Pp = torch.cat([P, torch.zeros(P.shape[0], 3, dtype=torch.bool)], 1)
        Sp = torch.cat([S, torch.zeros(3, dtype=torch.long)])
        v3, _ = m(Xp, M, Pp, Sp)
    keep = P.numpy()
    assert np.allclose(v.numpy()[keep], v2.numpy()[keep], atol=1e-5)
    assert np.allclose(r.numpy()[keep], r2.numpy()[keep], atol=1e-5)
    assert np.allclose(v.numpy()[keep], v3[:, :X.shape[1]].numpy()[keep], atol=1e-5)
    assert np.isfinite(v.numpy()).all() and np.isfinite(r.numpy()).all()


def test_nomix_has_no_cross_token_information():
    torch = pytest.importorskip("torch")
    m = _model("kan", False)
    X, M, P, S = _inputs()
    with torch.no_grad():
        v, r = m(X, M, P, S)
        X2 = X.clone()
        X2[:, 1:] = torch.randn_like(X2[:, 1:])
        M2 = torch.randn_like(M)
        v2, r2 = m(X2, M2, P, S)
    assert np.allclose(v[:, 0].numpy(), v2[:, 0].numpy(), atol=1e-6)
    assert np.allclose(r[:, 0].numpy(), r2[:, 0].numpy(), atol=1e-6)


def test_mixing_layer_uses_other_tokens():
    torch = pytest.importorskip("torch")
    m = _model("kan", True)
    X, M, P, S = _inputs()
    with torch.no_grad():
        v, _ = m(X, M, P, S)
        X2 = X.clone()
        X2[:, 1:] = torch.randn_like(X2[:, 1:])
        v2, _ = m(X2, M, P, S)
    assert not np.allclose(v[:, 0].numpy(), v2[:, 0].numpy(), atol=1e-6)


def test_quantiles_ordered_and_param_budget():
    torch = pytest.importorskip("torch")
    counts = {}
    for kind, mix in (("kan", True), ("mlp", True), ("kan", False)):
        m = _model(kind, mix)
        counts[(kind, mix)] = xs.n_params(m)
        X, M, P, S = _inputs(B=3, N=5)
        with torch.no_grad():
            v, r = m(10 * X, M, P, S)
        assert (np.diff(v.numpy(), axis=-1) >= 0).all() and (v.numpy() > 0).all()
        assert (np.diff(r.numpy(), axis=-1) >= 0).all()
    assert all(c <= 5000 for c in counts.values()), counts
    assert counts[("kan", True)] == counts[("kan", False)]
    assert abs(counts[("mlp", True)] / counts[("kan", True)] - 1) < 0.03, counts


def test_masked_loss_ignores_invalid_cells():
    torch = pytest.importorskip("torch")
    m = _model()
    X, M, P, S = _inputs(B=2, N=4)
    with torch.no_grad():
        v, r = m(X, M, P, S)
    Yv = torch.rand(2, 4, len(xs.VOL_H))
    Yr = torch.randn(2, 4, len(xs.REL_H))
    Mv = torch.ones_like(Yv, dtype=torch.bool)
    Mr = torch.ones_like(Yr, dtype=torch.bool)
    Mv[0, 1] = False
    Mr[1, 2] = False
    l1 = float(xs.loss_fn(v, r, Yv, Mv, Yr, Mr))
    Yv2, Yr2 = Yv.clone(), Yr.clone()
    Yv2[0, 1] = 1e6
    Yr2[1, 2] = -1e6
    assert abs(float(xs.loss_fn(v, r, Yv2, Mv, Yr2, Mr)) - l1) < 1e-6


# ------------------------------------------------------------------ targets and pairwise


def test_relative_return_target_maths():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(50, 8))
    R[3, 2] = np.nan
    member = rng.random((50, 8)) > 0.2
    member[7, :] = False
    member[7, :4] = True  # 4 members < MIN_TOKENS
    rel = xs.relative(R, member, min_n=5)
    for t in range(50):
        U = member[t] & np.isfinite(R[t])
        if U.sum() < 5:
            assert np.isnan(rel[t]).all()
            continue
        assert np.allclose(rel[t, U], R[t, U] - R[t, U].mean())
        assert abs(np.nansum(rel[t])) < 1e-10
        assert np.isnan(rel[t, ~U]).all()
    # ranking by rel = ranking by R inside U (the market term cancels)
    t = 0
    U = member[t] & np.isfinite(R[t])
    assert (np.argsort(rel[t, U]) == np.argsort(R[t, U])).all()


def test_forward_and_back_returns_align():
    rng = np.random.default_rng(1)
    logc = np.cumsum(rng.normal(size=(200, 3)) * 0.01, axis=0)
    for h in (1, 4, 24):
        f = xs.fwd_ret(logc, h)
        b = xs.back_ret(logc, h)
        # forward over hours t .. t+h-1 is what is 'past' at hour t + h
        assert np.allclose(f[1:200 - h], b[1 + h:200], equal_nan=True)
        t = 50
        assert np.isclose(f[t, 0], logc[t + h - 1, 0] - logc[t - 1, 0])
        assert np.isclose(b[t, 0], logc[t - 1, 0] - logc[t - 1 - h, 0])


def test_pairwise_consistency():
    rng = np.random.default_rng(2)
    m = rng.normal(size=(3, 6)) * 0.01
    w = rng.uniform(0.005, 0.02, size=(3, 6))
    q = np.stack([m - w, m, m + w * 1.3], axis=-1)
    P = xs.pairwise_prob(q)
    assert np.allclose(P + np.swapaxes(P, -1, -2), 1.0)
    assert np.allclose(np.diagonal(P, axis1=-2, axis2=-1), 0.5)
    for t in range(3):
        mm = q[t, :, 1]
        assert ((P[t] > 0.5) == (mm[:, None] > mm[None, :])).all()
        # the ranking by 'number of pairwise wins' equals the ranking by the median
        wins = (P[t] > 0.5).sum(1)
        assert (np.argsort(wins) == np.argsort(mm)).all()


def test_spearman_rows_and_contrib():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(20, 9)), rng.normal(size=(20, 9))
    mask = np.ones((20, 9), bool)
    mask[0, :5] = False
    ic = xs.spearman_rows(a, b, mask)
    c = xs.rank_contrib(a, b, mask)
    from scipy.stats import spearmanr

    for t in range(1, 20):
        assert np.isclose(ic[t], spearmanr(a[t], b[t]).statistic)
        assert np.isclose(np.nanmean(c[t]), ic[t])
    assert np.isnan(ic[0])


def test_rolling_beta_matches_ols():
    rng = np.random.default_rng(4)
    x = rng.normal(size=800)
    y = (0.7 * x + rng.normal(size=800) * 0.5)[:, None]
    y[100, 0] = np.nan
    b = xs.rolling_beta(y, x, w=200, min_n=150)
    t = 500
    xx, yy = x[t - 200:t], y[t - 200:t, 0]
    k = np.isfinite(yy)
    assert np.isclose(b[t, 0], np.polyfit(xx[k], yy[k], 1)[0])
    assert np.isnan(b[100, 0])  # < 150 pairs


def test_seasonal_multi_h1_equals_seasonal_factor():
    from ai_trader.features.seasonality import seasonal_factor

    rng = np.random.default_rng(5)
    n = 24 * 7 * 10
    how = np.arange(n) % 168
    u = rng.exponential(size=n) * (1 + (how % 24 > 12))
    u[50] = np.nan
    sm = xs.seasonal_multi(how, u, (1, 4))
    ref = seasonal_factor(how, u)
    assert np.allclose(sm[1], ref, equal_nan=True)
    # freeze the profile (no new data after t0): the h-hour factor is then the mean of the next h hourly factors
    t0 = n - 40
    u2 = u.copy()
    u2[t0:] = np.nan
    sm2 = xs.seasonal_multi(how, u2, (1, 4))
    for t in (t0 + 1, t0 + 7, n - 10):
        assert np.isclose(sm2[4][t], sm2[1][t:t + 4].mean())


# ------------------------------------------------------------------ panel look-ahead


def _synthetic_frame(sym: str, start: datetime, n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.standard_t(4, size=n) * 0.006
    close = 100 * np.exp(np.cumsum(ret))
    ret[0] = np.nan
    sig = np.abs(rng.normal(0.006, 0.001, size=n))
    ts = [start + timedelta(hours=i) for i in range(n)]
    fund = np.repeat(rng.normal(1e-4, 5e-5, size=n // 8 + 1), 8)[:n]
    oi = 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    return pl.DataFrame({"event_ts": ts, "close": close, "ret": ret, "sigma_risk_next": sig, "funding_rate": fund,
                         "open_interest": oi}).with_columns(pl.col("event_ts").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")))


def _perturb_after(f: pl.DataFrame, cut: datetime, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    after = (f["event_ts"] >= cut).to_numpy()
    k = int(after.sum())
    close = f["close"].to_numpy().copy()
    jump = np.exp(np.cumsum(rng.normal(0, 0.05, size=k)))
    close[after] = close[after] * jump
    ret = np.r_[np.nan, np.diff(np.log(close))]
    sig = f["sigma_risk_next"].to_numpy().copy()
    sig[after] = rng.uniform(0.001, 0.05, size=k)
    fund = f["funding_rate"].to_numpy().copy()
    fund[after] = rng.normal(0, 1e-3, size=k)
    oi = f["open_interest"].to_numpy().copy()
    oi[after] = oi[after] * rng.uniform(0.5, 2, size=k)
    return f.with_columns(pl.Series("close", close), pl.Series("ret", ret), pl.Series("sigma_risk_next", sig),
                          pl.Series("funding_rate", fund), pl.Series("open_interest", oi))


def test_panel_has_no_lookahead():
    from ai_trader.research import xs_network_study as st

    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    n = 24 * 110
    frames = {s: _synthetic_frame(s, start + timedelta(hours=24 * (i % 2)), n, seed=i) for i, s in enumerate(syms)}
    frames["ADAUSDT"] = frames["ADAUSDT"].filter((pl.col("event_ts") < datetime(2024, 3, 20, tzinfo=UTC)) | (pl.col("event_ts") > datetime(2024, 3, 21, tzinfo=UTC)))
    P0 = st.assemble({s: st.token_rows(f) for s, f in frames.items()}, syms)
    cut = datetime(2024, 4, 5, 13, tzinfo=UTC)
    P1 = st.assemble({s: st.token_rows(_perturb_after(f, cut, 99)) for s, f in frames.items()}, syms)
    cut_us = int(cut.timestamp() * 1e6)
    assert P0["present"].sum() > 1000, "synthetic panel too short to test"
    k = P0["ts"] <= cut_us  # decisions at or before the cut use bars < cut only
    assert (P0["ts"] == P1["ts"]).all()
    assert (P0["present"][k] == P1["present"][k]).all()
    assert np.array_equal(P0["X"][k], P1["X"][k], equal_nan=True)
    assert np.array_equal(P0["M"][k], P1["M"][k], equal_nan=True)
    assert np.array_equal(P0["base_rel"][k], P1["base_rel"][k], equal_nan=True)
    # ... and the perturbation is visible right after the cut
    j = int(np.searchsorted(P0["ts"], cut_us)) + 2
    assert not np.array_equal(P0["X"][j], P1["X"][j], equal_nan=True)
    # targets whose window ends before the cut are unchanged
    for kk, h in enumerate(xs.VOL_H):
        kt = P0["ts"] + (h - 1) * 3600 * 1_000_000 < cut_us
        assert np.array_equal(P0["Av"][kt, :, kk], P1["Av"][kt, :, kk], equal_nan=True)
    # gap handling: the token with the missing day is absent around it, the others are not
    # (the 00:00 decision still has the 23:00 bar; from 01:00 the last hour is missing)
    gap = (P0["ts"] >= int(datetime(2024, 3, 20, 1, tzinfo=UTC).timestamp() * 1e6)) & (P0["ts"] <= int(datetime(2024, 3, 21, tzinfo=UTC).timestamp() * 1e6))
    assert not P0["present"][gap, syms.index("ADAUSDT")].any()
    # relative targets sum to zero over the universe at every valid hour
    rel = P0["rel_raw"][..., 0]
    ok = np.isfinite(rel).sum(1) >= xs.MIN_TOKENS
    assert np.allclose(np.nansum(rel[ok], axis=1), 0, atol=1e-5)
