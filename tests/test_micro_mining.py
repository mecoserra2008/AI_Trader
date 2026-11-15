import json

import numpy as np
import polars as pl
import pytest

from ai_trader.research import micro_mining as mm


def test_bh_matches_statsmodels_and_known_case():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 1, 900), rng.uniform(0, 1e-3, 100)])
    rej = mm.bh(p, 0.10)
    ref = multipletests(p, alpha=0.10, method="fdr_bh")
    assert (rej == ref[0]).all()
    assert np.allclose(mm.bh_adjusted(p), ref[1])
    # textbook case: m = 5, q = 0.1 -> thresholds 0.02 0.04 0.06 0.08 0.10; largest k with p_(k) <= thr is 3
    p2 = np.array([0.01, 0.03, 0.05, 0.2, 0.9])
    assert mm.bh(p2, 0.10).tolist() == [True, True, True, False, False]
    assert not mm.bh(np.array([0.5, 0.6]), 0.10).any()
    # all-null uniform p-values: few or no rejections
    assert mm.bh(rng.uniform(0, 1, 2000), 0.10).sum() <= 10


def test_past_percentile_uses_only_past():
    x = np.array([np.nan, 1.0, 2.0, 3.0, 0.5, 2.0])
    p = mm.past_percentile(x, min_past=2)
    assert np.isnan(p[:3]).all()          # fewer than 2 finite past values
    assert p[3] == 1.0                    # 3 > {1, 2}
    assert p[4] == 0.0                    # 0.5 < {1, 2, 3}
    assert p[5] == pytest.approx((2 + 0.5) / 4)  # mid-rank among {1, 2, 3, 0.5}
    assert mm.to_bins(np.array([0.0, 0.199, 0.2, 0.999, 1.0, np.nan]), 5).tolist() == [0, 0, 1, 4, 4, -1]


def _series(rng, n=1440):
    lc = np.cumsum(rng.normal(0, 1e-3, n)) + 5.0
    return {"lclose": lc, "rv": rng.gamma(2.0, 1e-6, n), "buy": rng.gamma(2.0, 1e4, n), "sell": rng.gamma(2.0, 1e4, n),
            "n_trades": np.full(n, 10.0)}


def test_states_do_not_depend_on_future_minutes():
    rng = np.random.default_rng(1)
    ms = _series(rng)
    sp = rng.integers(0, 2, 1440)
    t1a, t5a = mm.symbol_tables(ms, sp)
    k = 700
    ms2 = {key: v.copy() for key, v in ms.items()}
    for key in ("lclose", "rv", "buy", "sell"):
        ms2[key][k + 1:] = rng.permutation(ms2[key][k + 1:]) * 3.0
    sp2 = sp.copy()
    sp2[k + 1:] = 1 - sp2[k + 1:]
    t1b, t5b = mm.symbol_tables(ms2, sp2)
    for col in ("flow_q", "vol_t", "spread", "hour"):
        assert (t1a[col][: k + 1] == t1b[col][: k + 1]).all(), col
    sel = t5a["minute"] <= k
    for col in ("flow_q", "vol_t"):
        assert (t5a[col][sel] == t5b[col][sel]).all(), col
    # warm-up: no state before WARMUP_MIN past values
    assert (t1a["flow_q"][: mm.WARMUP_MIN] == -1).all()
    assert (t1a["vol_t"][: mm.WARMUP_MIN + mm.VOL_WIN_T1 - 1] == -1).all()
    # targets are demeaned within the symbol-day over the analysis rows
    ok = t1a["ok"]
    for tg in ("a1", "a5", "s1", "s5"):
        assert abs(np.nanmean(t1a[tg][ok])) < 1e-9


def test_spread_minutes_assigns_state_to_minute_end():
    d0 = mm.day0_ms("2026-01-01")
    ts = d0 + np.arange(1, 121) * 1000          # states at 1..120 s -> minutes 0 (1..60 s) and 1 (61..120 s)
    bid = np.full(120, 100.0)
    ask = np.where(np.arange(120) < 60, 100.1, 100.3)
    b = pl.DataFrame({"ts": ts, "bid1": bid, "ask1": ask}).with_columns(pl.from_epoch(pl.col("ts"), time_unit="ms").dt.replace_time_zone("UTC"))
    s = mm.spread_minutes(b, d0, 0.1)
    assert s[0] == 0 and s[1] == 1 and s[2] == -1


def test_confirmation_refuses_without_frozen_file(tmp_path):
    missing = tmp_path / "discoveries_frozen.json"
    t = pl.DataFrame({"day": ["x"]})
    with pytest.raises(mm.FrozenMissing):
        mm.confirmation(t, t, [mm.CONFIRMATION_DAYS[0]], frozen=missing)
    with pytest.raises(mm.FrozenMissing):
        mm._check_wall(mm.CONFIRMATION_DAYS[0], frozen=missing)
    mm._check_wall(mm.DISCOVERY_DAYS[0], frozen=missing)   # discovery days are always allowed
    with pytest.raises(ValueError):
        mm._check_wall("2024-01-01", frozen=missing)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"discoveries": []}), encoding="utf-8")
    with pytest.raises(mm.FrozenMissing):
        mm.require_frozen(bad)


def test_freeze_refuses_overwrite(tmp_path):
    f = tmp_path / "fz.json"
    disc = {"days": list(mm.DISCOVERY_DAYS), "family": {"total": 1}, "effects": [
        {"id": "F1:a1:alt:f0:v0:h0", "family": "F1", "kind": "vol", "direction": 1, "mean": 0.1, "t": 3.0, "p": 0.01, "p_bh": 0.05,
         "n_days": 16, "describe": "x", "bh_reject": True}]}
    obj = mm.freeze(disc, frozen=f)
    assert obj["n_discoveries"] == 1 and mm.require_frozen(f)["frozen_at"]
    with pytest.raises(FileExistsError):
        mm.freeze(disc, frozen=f)


def test_family_is_declared_and_unique():
    c = mm.family_counts()
    assert c["total"] == 1884 and c["vol"] == c["direction"] == 942
    assert set(mm.DISCOVERY_DAYS).isdisjoint(mm.CONFIRMATION_DAYS)
    assert len(mm.DISCOVERY_DAYS) == 16 and len(mm.CONFIRMATION_DAYS) == 15


def test_day_estimates_fast_matches_slow():
    rng = np.random.default_rng(3)
    days = ["d1", "d2", "d3"]
    n = 3000
    t1 = pl.DataFrame({"day": rng.choice(days, n), "cls": rng.choice(["alt", "major"], n), "flow_q": rng.integers(0, 5, n),
                       "vol_t": rng.integers(0, 3, n), "hour": rng.integers(0, 4, n), "a1": rng.normal(0, 1, n)})
    es = [mm.Effect("e1", "F1", "t1", "a1", (("cls", "alt"),), ("flow_q", "vol_t"), (4, 1)),
          mm.Effect("e2", "F2", "t1", "a1", (("cls", "alt"),), ("flow_q", "vol_t"), (4, 1), (0, 1))]
    fast = mm._fast_estimates({"t1": t1}, es, days)
    for e in es:
        slow = mm.day_estimates({"t1": t1}, e, days, {})
        assert np.allclose(fast[e.id], slow, equal_nan=True)
    r = mm.day_ttest(np.array([1.0, 2.0, 3.0]), sided=1)
    assert r["p"] < 0.05 and mm.day_ttest(np.array([1.0, 2.0, 3.0]), sided=-1)["p"] > 0.9
