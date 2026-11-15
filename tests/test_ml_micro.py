import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from ai_trader.ml import micro as mc  # noqa: E402

D0 = 1_756_598_400_000  # 2025-08-31 00:00 UTC in ms


def _day(seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.choice(mc.DAY_S, size=int(mc.DAY_S * frac), replace=False))
    p = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(secs))))
    n = len(secs)
    tr = pl.DataFrame({"ts": D0 + secs * 1000, "open": p, "high": p * 1.0001, "low": p * 0.9999, "close": p, "vwap": p,
                       "buy_vol": rng.exponential(1, n), "sell_vol": rng.exponential(1, n), "buy_notional": rng.exponential(100, n),
                       "sell_notional": rng.exponential(100, n), "n_trades": rng.integers(1, 9, n).astype(np.int32),
                       "n_buy": rng.integers(0, 5, n).astype(np.int32), "n_sell": rng.integers(0, 5, n).astype(np.int32),
                       "n_orders": rng.integers(1, 5, n).astype(np.int32), "n_buy_orders": rng.integers(0, 3, n).astype(np.int32),
                       "n_sell_orders": rng.integers(0, 3, n).astype(np.int32)})
    bs = np.arange(1, mc.DAY_S + 1)
    bk = pl.DataFrame({"ts": D0 + bs * 1000, "rel_spread_bp": rng.uniform(0.5, 2, len(bs)), "depth_bid_10": rng.uniform(1e5, 2e5, len(bs)),
                       "depth_ask_10": rng.uniform(1e5, 2e5, len(bs)), "imb_10": rng.uniform(-1, 1, len(bs))}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms", "UTC")))
    return tr, bk


def test_channels_are_causal():
    tr, bk = _day()
    full = mc.symbol_day_channels(tr, bk, D0)
    cut_s = 50_000
    part = mc.symbol_day_channels(tr.filter(pl.col("ts") < D0 + cut_s * 1000), bk.filter(pl.col("ts").dt.epoch("ms") <= D0 + cut_s * 1000), D0)
    assert np.allclose(full["X"][:cut_s], part["X"][:cut_s], atol=1e-6)
    assert np.allclose(full["b"][:cut_s], part["b"][:cut_s])
    assert np.allclose(full["logp"][:cut_s], part["logp"][:cut_s], equal_nan=True)
    assert full["X"][:, 8].min() == 1.0 and np.abs(full["X"]).max() <= 10


def test_targets_windows_and_har_alignment():
    tr, bk = _day(1)
    r = mc.symbol_day_channels(tr, None, D0)
    s = mc.decision_seconds(10)
    assert s[0] == mc.TRAIL + mc.WIN and s[-1] + mc.HORIZON < mc.DAY_S
    tg = mc.targets(r["logp"], r["b"], s)
    i = 100
    assert np.isclose(tg["r1"][i], r["logp"][s[i] + 60] - r["logp"][s[i]])
    H = mc.har_features(r["X"], s)
    assert np.allclose(H[i, : mc.C], r["X"][s[i] - 9: s[i] + 1].mean(0), atol=1e-5)
    assert np.allclose(H[i, 2 * mc.C:], r["X"][s[i] - 299: s[i] + 1].mean(0), atol=1e-5)
    store = mc.Store(r["X"].astype(np.float16), {("d", "x"): 0}, {})
    from ai_trader.research.ml_micro_study import Gather, har_from_x

    g = Gather(store)
    w = g(store.windows(("d", "x"), s[[i]]))
    assert w.shape == (1, mc.WIN, mc.C)
    assert np.allclose(w[0, -1].numpy(), r["X"][s[i]].astype(np.float16).astype(np.float32))
    assert np.allclose(har_from_x(w)[0].numpy(), H[i], atol=2e-2)  # float16 storage


def test_cnn_is_deterministic_and_lin_is_its_skip():
    x = torch.randn(4, mc.WIN, mc.C)
    h = torch.randn(4, mc.C * 3)
    torch.manual_seed(0)
    a = mc.make_cnn()
    torch.manual_seed(0)
    b = mc.make_cnn()
    assert torch.equal(a(x, h), b(x, h))
    lin = mc.make_linear()
    lin.skip.load_state_dict(a.skip.state_dict())
    assert torch.allclose(lin(x, h), a.skip(h).squeeze(-1))
    f = torch.zeros(4, requires_grad=True)
    loss = mc.qlike_loss(f, torch.ones(4), torch.ones(4))
    loss.backward()
    assert abs(float(loss.detach()) - 1.0) < 1e-6 and torch.allclose(f.grad, torch.zeros(4))  # optimum at f = log(r^2 / base)


def test_fold_checkpoint_roundtrip(tmp_path, monkeypatch):
    import ai_trader.research.ml_micro_study as ms

    monkeypatch.setattr(ms, "root", lambda: tmp_path)
    days = ["2025-08-31", "2026-07-12"]
    dh = ms.design_hash(days)
    assert dh == ms.design_hash(days) and dh != ms.design_hash(days[:1])
    assert not ms.fold_done(0, dh)

    class S:
        meta = {("2025-08-31", "AUSDT"): {"day": "2025-08-31", "symbol": "AUSDT", "book": False, "traded": 20000}}

    folds = {}
    for fk, day in enumerate(days):
        pr = [pl.DataFrame({"day": [day], "symbol": ["AUSDT"], "sec": [4000], "r1": [0.001], "base": [1e-6], "F_cnn": [1e-6], "F_lin": [1e-6],
                            "p_cnn": [0.5], "p_lin": [0.5], "book": [False]})]
        ab = [pl.DataFrame({"fold": [fk], "channel": ["z"], "kind": ["cnn"], "day": [day], "symbol": ["AUSDT"], "d": [0.1]})]
        sg = ms.seg_of(day)
        ig = {f"abs_{sg}": [np.ones((mc.WIN, mc.C))], f"sgn_{sg}": [np.zeros((mc.WIN, mc.C))], f"n_{sg}": [2]}
        S.meta = {(day, "AUSDT"): {"day": day, "symbol": "AUSDT", "book": False, "traded": 20000}}
        ms.save_fold(fk, dh, pr, ab, ig, {"fold": fk, "test_days": [day]}, S)
        assert ms.fold_done(fk, dh) and not ms.fold_done(fk, "other")
        folds[fk] = [day]
    ms.aggregate(days, folds, dh, 0.0)
    assert pl.read_parquet(tmp_path / "preds.parquet").height == 2
    import json as _j

    ig = _j.loads((tmp_path / "ig.json").read_text(encoding="utf-8"))
    assert ig["llm_in_sample"]["n"] == 2 and np.allclose(np.asarray(ig["llm_in_sample"]["abs_mean"]), 0.5)
    fi = _j.loads((tmp_path / "fit_info.json").read_text(encoding="utf-8"))
    assert fi["n_symbol_days"] == 2 and len(fi["folds"]) == 2


def test_binned_auc_bootstrap_matches_exact():
    from ai_trader.research.ml_micro_study import _auc, binned_auc_boot

    rng = np.random.default_rng(0)
    g = rng.integers(0, 50, 20000)
    p = rng.random(20000)
    y = (rng.random(20000) < 0.3 + 0.3 * p).astype(int)
    exact = _auc(y, p)
    one = binned_auc_boot(y, p, g, B=200)
    assert abs(np.median(one) - exact) < 0.01 and 0.5 < exact < 0.7
    # identity weights reproduce the exact AUC up to binning
    from unittest import mock

    with mock.patch("numpy.random.default_rng") as r:
        r.return_value.integers.side_effect = lambda lo, hi, n: np.arange(n)
        ident = binned_auc_boot(y, p, g, B=1)
    assert abs(ident[0] - exact) < 2e-3
