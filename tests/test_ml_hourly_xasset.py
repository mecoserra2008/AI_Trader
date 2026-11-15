from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.ml.hourly_xasset import (
    XA_FEATURES,
    BlockStd,
    attach_leaders,
    cash_session_flag,
    cme_session_flag,
    leader_block,
    session_flags,
)
from ai_trader.research.frame import add_robust_vol, enrich_bars
from tests.conftest import T0, make_bars


def _frame(n_hours: int, seed: int, drop=()):
    b = make_bars(n_hours, freq_h=1, seed=seed)
    if drop:
        b = b.with_row_index("i").filter(~pl.col("i").is_in(list(drop))).drop("i")
    return add_robust_vol(enrich_bars(b))


def _ts(*hours: datetime) -> pl.Series:
    return pl.Series("event_ts", list(hours)).cast(pl.Datetime("us", "UTC"))


def _np(s: pl.Series) -> np.ndarray:
    return s.cast(pl.Float64).fill_null(np.nan).to_numpy()


def test_leader_block_uses_only_bars_before_t():
    full = _frame(24 * 40, seed=5, drop=range(300, 302))
    t_cut = T0 + timedelta(hours=700)
    # the leader's history truncated at t_cut: the block at bars < t_cut must not change
    part = add_robust_vol(enrich_bars(full.filter(pl.col("event_ts") < t_cut).select(
        "event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")))
    a = leader_block(part, "btc")
    b = leader_block(full, "btc").join(a.select("event_ts"), on="event_ts", how="semi")
    assert a.height == b.height > 600
    for c in a.columns[1:]:
        x, y = _np(a[c]), _np(b[c])
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=1e-9, atol=1e-12), c
    # perturbing the leader's bar t (and later) leaves the inputs at t unchanged; bar t-1 changes them
    blk = leader_block(full, "btc")
    i = blk["event_ts"].to_list().index(T0 + timedelta(hours=500))
    t = blk["event_ts"][i]
    shocked = full.with_columns(pl.when(pl.col("event_ts") >= t).then(pl.col("ret") * 50).otherwise(pl.col("ret")).alias("ret"))
    b2 = leader_block(shocked, "btc")
    assert all(np.isclose(blk[c][i], b2[c][i]) for c in blk.columns[1:] if blk[c][i] is not None)
    # explicit timing: lu1 at t = log(r_{t-1}^2 / s2_t), s2_t = sigma_risk_next of bar t-1 squared
    g = full.with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    prev = g.filter(pl.col("event_ts") == t - timedelta(hours=1))
    r1, s1 = prev["ret"][0], prev["sigma_risk_next"][0]
    assert np.isclose(blk["xbtc_lu1"][i], np.log(max(r1 * r1 / (s1 * s1), 1e-4)))
    assert np.isclose(blk["xbtc_lsig2"][i], np.log(s1 * s1))
    # the bar after the gap: the return across the missing bars is not used
    j = blk["event_ts"].to_list().index(T0 + timedelta(hours=303))
    assert blk["xbtc_lu1"][j] is None and blk["xbtc_ok"][j] == 0.0


def test_attach_excludes_own_series_and_aligns_on_event_ts():
    blocks = {s: leader_block(_frame(24 * 20, seed=k), tag) for k, (s, tag) in enumerate((("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("QQQUSDT", "qqq")))}
    # QQQ only exists from hour 200 on
    blocks["QQQUSDT"] = blocks["QQQUSDT"].filter(pl.col("event_ts") >= T0 + timedelta(hours=200))
    ts = [T0 + timedelta(hours=h) for h in (100, 300)]
    panel = pl.DataFrame({"event_ts": ts * 3, "symbol": ["BTCUSDT"] * 2 + ["SOLUSDT"] * 2 + ["QQQUSDT"] * 2}).with_columns(
        pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    p = attach_leaders(panel, blocks)
    assert set(XA_FEATURES) <= set(p.columns)
    btc = p.filter(pl.col("symbol") == "BTCUSDT")
    assert btc["xbtc_lu1"].null_count() == 2 and btc["xbtc_ok"].to_list() == [0.0, 0.0]
    assert btc["xeth_lu1"].null_count() == 0
    qqq = p.filter(pl.col("symbol") == "QQQUSDT")
    assert qqq["xqqq_lsig2"].null_count() == 2 and qqq["xqqq_ok"].to_list() == [0.0, 0.0]
    sol = p.filter(pl.col("symbol") == "SOLUSDT").sort("event_ts")
    ref = blocks["BTCUSDT"].filter(pl.col("event_ts").is_in(sol["event_ts"].implode())).sort("event_ts")
    assert np.allclose(_np(sol["xbtc_lum6"]), _np(ref["xbtc_lum6"]))
    assert sol["xqqq_ok"].to_list() == [0.0, 1.0]  # QQQ absent at hour 100 -> missing, not filled forward


def test_cash_session_flag_across_dst():
    d = lambda *a: datetime(*a, tzinfo=UTC)  # noqa: E731
    # Friday 2026-03-06 under EST: cash bars 09:00-15:00 ET = 14:00-20:00 UTC
    fri = cash_session_flag(_ts(*[d(2026, 3, 6, h) for h in range(24)]))
    assert np.flatnonzero(fri).tolist() == list(range(14, 21))
    # Monday 2026-03-09 after the US switch to EDT (2026-03-08): 13:00-19:00 UTC
    mon = cash_session_flag(_ts(*[d(2026, 3, 9, h) for h in range(24)]))
    assert np.flatnonzero(mon).tolist() == list(range(13, 20))
    # back to EST on 2026-11-01: Friday 2026-10-30 13-19 UTC, Monday 2026-11-02 14-20 UTC
    assert np.flatnonzero(cash_session_flag(_ts(*[d(2026, 10, 30, h) for h in range(24)]))).tolist() == list(range(13, 20))
    assert np.flatnonzero(cash_session_flag(_ts(*[d(2026, 11, 2, h) for h in range(24)]))).tolist() == list(range(14, 21))
    # Thanksgiving closed; the day after closes 13:00 ET = 18:00 UTC -> bars 14-17 UTC
    assert cash_session_flag(_ts(*[d(2026, 11, 26, h) for h in range(24)])).sum() == 0
    assert np.flatnonzero(cash_session_flag(_ts(*[d(2026, 11, 27, h) for h in range(24)]))).tolist() == [14, 15, 16, 17]
    # weekends off
    assert cash_session_flag(_ts(*[d(2026, 3, 7, h) for h in range(24)])).sum() == 0


def test_cme_session_flag_across_dst():
    d = lambda *a: datetime(*a, tzinfo=UTC)  # noqa: E731
    # Sunday reopen 18:00 ET: 23:00 UTC under EST (2026-03-01), 22:00 UTC under EDT (2026-03-08)
    s1 = cme_session_flag(_ts(*[d(2026, 3, 1, h) for h in range(24)]))
    s2 = cme_session_flag(_ts(*[d(2026, 3, 8, h) for h in range(24)]))
    assert np.flatnonzero(s1).tolist() == [23] and np.flatnonzero(s2).tolist() == [22, 23]
    # daily break 17:00-18:00 ET: 22 UTC (EST, Tue 2026-03-03), 21 UTC (EDT, Tue 2026-03-10)
    t1 = cme_session_flag(_ts(*[d(2026, 3, 3, h) for h in range(24)]))
    t2 = cme_session_flag(_ts(*[d(2026, 3, 10, h) for h in range(24)]))
    assert np.flatnonzero(t1 == 0).tolist() == [22] and np.flatnonzero(t2 == 0).tolist() == [21]
    # Friday close 17:00 ET: last bar 21 UTC (EST, 2026-03-06), 20 UTC (EDT, 2026-03-13); Saturday off
    f1 = cme_session_flag(_ts(*[d(2026, 3, 6, h) for h in range(24)]))
    f2 = cme_session_flag(_ts(*[d(2026, 3, 13, h) for h in range(24)]))
    assert np.flatnonzero(f1 == 0).tolist() == [22, 23] and np.flatnonzero(f2 == 0).tolist() == [21, 22, 23]
    assert cme_session_flag(_ts(*[d(2026, 3, 7, h) for h in range(24)])).sum() == 0
    # class routing: equities get only the cash flag, commodities only the CME flag
    ts = _ts(*[d(2026, 3, 10, h) for h in range(24)])
    eq, cm = session_flags(ts, "NVDAUSDT"), session_flags(ts, "XAUUSDT")
    assert eq["cme_sess"].sum() == 0 and eq["cash_sess"].sum() == 7
    assert cm["cash_sess"].sum() == 0 and cm["cme_sess"].sum() == 23


def test_block_std_missing_to_mean():
    X = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    st = BlockStd().fit(X)
    Z = st.transform(X)
    assert np.allclose(Z[:, 0], [-1.0, 1.0, 0.0]) and np.all(Z[:, 1] == 0)
