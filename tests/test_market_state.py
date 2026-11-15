"""Market state card (tools/market_state.py): point-in-time invariance and the causal flow detector."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.storage.lake import Lake
from ai_trader.tools import market_state as ms

from .conftest import T0, make_bars

SYM = "TESTUSDT"
AS_OF = T0 + timedelta(days=290, minutes=30)
UTC_MS = pl.Datetime("ms", "UTC")


def _ts(xs):
    return pl.Series(xs).dt.replace_time_zone("UTC").dt.cast_time_unit("ms") if xs and xs[0].tzinfo is None else pl.Series(xs).dt.cast_time_unit("ms")


def _funding(end):
    ev = [T0 + timedelta(hours=8 * i) for i in range(int((end - T0) / timedelta(hours=8)))]
    rng = np.random.default_rng(1)
    return pl.DataFrame({"venue": "bybit", "symbol": SYM, "event_ts": _ts(ev), "rate": rng.normal(1e-4, 5e-5, len(ev)),
                         "known_ts": _ts([t + timedelta(seconds=2) for t in ev])})


def _oi(end):
    ev = [T0 + timedelta(hours=4 * i) for i in range(int((end - T0) / timedelta(hours=4)))]
    rng = np.random.default_rng(2)
    return pl.DataFrame({"venue": "bybit", "interval": "4h", "symbol": SYM, "event_ts": _ts(ev),
                         "open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ev)))),
                         "known_ts": _ts([t + timedelta(hours=4, seconds=2) for t in ev])})


def _liq(start, end):
    rng = np.random.default_rng(3)
    ev = sorted(start + timedelta(seconds=float(s)) for s in rng.uniform(0, (end - start).total_seconds(), 300))
    return pl.DataFrame({"venue": "bybit", "symbol": SYM, "event_ts": _ts(ev), "side": rng.choice(["long", "short"], len(ev)).tolist(),
                         "price": rng.uniform(90, 110, len(ev)), "size": rng.exponential(10, len(ev)),
                         "known_ts": _ts([t + timedelta(milliseconds=300) for t in ev])})


def _flow(start, end, spike_at=None):
    n = int((end - start) / timedelta(minutes=1))
    ev = [start + timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(4)
    buy, sell = 1e4 + rng.normal(0, 1e3, n), 1e4 + rng.normal(0, 1e3, n)   # Gaussian delta: |z| > 4 by chance ~0.1 per day
    if spike_at is not None:
        k = int((spike_at - start) / timedelta(minutes=1))
        buy[k] = 1e7
    return pl.DataFrame({"venue": "bybit", "category": "linear", "symbol": SYM, "event_ts": _ts(ev),
                         "close": 100 + np.cumsum(rng.normal(0, 0.01, n)), "buy_vol": buy / 100, "sell_vol": sell / 100,
                         "buy_notional": buy, "sell_notional": sell, "n_buy": 5, "n_sell": 5, "max_trade": 1.0,
                         "book_cov": 1.0, "book_updates": 10, "known_ts": _ts([t + timedelta(minutes=1, seconds=2) for t in ev])})


@pytest.fixture(scope="module")
def lakes(tmp_path_factory):
    root = tmp_path_factory.mktemp("ms")
    end = AS_OF + timedelta(days=3)
    n1 = int((end - T0) / timedelta(hours=1))
    tables = {
        "bars": pl.concat([make_bars(n1, SYM), make_bars(n1 // 4, SYM, freq_h=4, seed=5)]),
        "funding": _funding(end),
        "open_interest": _oi(end),
        "liquidations": _liq(AS_OF - timedelta(days=2), end),
        "flow_3s": _flow(AS_OF - timedelta(hours=50), AS_OF + timedelta(hours=6), spike_at=AS_OF - timedelta(hours=3)),
    }
    full, trunc = Lake(root=root / "full"), Lake(root=root / "trunc")
    for t, df in tables.items():
        full.write(t, df)
        trunc.write(t, df.filter(pl.col("known_ts") <= AS_OF))
    # a late revision of an old funding print: known only after as_of, so it must not change anything
    rev = tables["funding"].filter(pl.col("event_ts") < AS_OF).tail(1).with_columns(pl.lit(0.05).alias("rate"),
                                                                                   (pl.lit(AS_OF) + pl.duration(hours=1)).cast(UTC_MS).alias("known_ts"))
    full.write("funding", rev)
    return full, trunc


def test_card_is_invariant_to_data_after_as_of(lakes):
    full, trunc = lakes
    a = ms.state_card(SYM, AS_OF, lake=full)
    b = ms.state_card(SYM, AS_OF, lake=trunc)
    assert a == b
    assert a["price"] is not None and a["data_max_ts"] is not None
    assert ms.utc(a["data_max_ts"]) <= AS_OF
    for k in ("returns", "regime", "vol", "asymmetry", "funding", "oi", "liquidations_24h", "flow", "zones_4h"):
        assert k in a
    assert a["vol"]["har_sigma_day"] is not None and a["vol"]["vol_surprise"] is not None
    assert a["funding"]["z_90d"] is not None and a["oi"]["chg_24h_log"] is not None
    assert a["liquidations_24h"]["long_n"] + a["liquidations_24h"]["short_n"] > 0


def test_card_reads_nothing_after_as_of(lakes):
    full, _ = lakes
    c = ms.state_card(SYM, AS_OF, lake=full)
    assert ms.utc(c["last_bar"]) + timedelta(hours=1) <= AS_OF
    assert ms.utc(c["funding"]["ts"]) <= AS_OF and abs(c["funding"]["rate"]) < 0.01  # not the late revision
    assert ms.utc(c["oi"]["ts"]) + timedelta(hours=4) <= AS_OF
    assert ms.utc(c["flow"]["last_ts"]) <= AS_OF


def test_flow_spike_is_flagged_causally(lakes):
    full, _ = lakes
    f = ms.flow_anomalies([SYM], AS_OF, lake=full)[SYM]
    assert f["n_anom_24h"] >= 1
    assert f["latest"]["side"] == "buy" and f["latest"]["ts"].startswith((AS_OF - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M"))
    # an as_of before the spike cannot see it
    g = ms.flow_anomalies([SYM], AS_OF - timedelta(hours=4), lake=full)[SYM]
    assert g["latest"] is None or ms.utc(g["latest"]["ts"]) < AS_OF - timedelta(hours=4)


def test_robust_z_uses_only_the_past():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    z = ms._robust_z(x, 500)
    y = x.copy()
    y[-100:] = 1e6                                    # change the last 100 values
    z2 = ms._robust_z(y, 500)
    np.testing.assert_array_equal(z[:400], z2[:400])  # scores before the change are untouched
    assert np.all(z2[-100:] > 4)


def test_missing_symbol_does_not_crash(lakes):
    full, _ = lakes
    cards = ms.market_state([SYM, "NOPEUSDT"], AS_OF, lake=full)
    assert cards[0]["price"] is not None
    assert cards[1]["price"] is None and "no_1h_bars" in cards[1]["flags"]


def test_registered_tool(lakes):
    from ai_trader.core.registry import registered

    full, _ = lakes
    assert "state.market_card" in registered()
    r = ms.market_card(SYM, as_of=AS_OF, lake=full)
    assert r.data_max_ts <= AS_OF and r.value["symbol"] == SYM
