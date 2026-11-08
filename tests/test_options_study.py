import math

import numpy as np
import polars as pl
import pytest

from ai_trader.research.options_study import (
    black76, black76_delta, entry_quote, option_fee, simulate, straddle,
)


def test_black76_known_values():
    # Hull, Options Futures & Other Derivatives, Example 18.x: F=20, K=20, r=9%, T=4/12, sigma=25% -> put 1.12
    assert black76(20, 20, 4 / 12, 0.25, "P", r=0.09) == pytest.approx(1.1166, abs=1e-3)
    # ATM r=0: C = F (2N(sigma sqrt T / 2) - 1)
    F, T, s = 100.0, 0.5, 0.2
    expected = F * (2 * 0.5 * (1 + math.erf(s * math.sqrt(T) / 2 / math.sqrt(2))) - 1)
    assert black76(F, F, T, s, "C") == pytest.approx(expected, rel=1e-12)
    assert black76(F, F, T, s, "C") == pytest.approx(5.6372, abs=1e-4)
    # deep ITM / OTM limits and expiry
    assert black76(100, 50, 0.01, 0.5, "C") == pytest.approx(50, abs=1e-6)
    assert black76(100, 50, 0.01, 0.5, "P") == pytest.approx(0, abs=1e-6)
    assert black76(100, 90, 0.0, 0.5, "C") == 10.0 and black76(100, 90, 0.0, 0.5, "P") == 0.0


def test_put_call_parity():
    for F, K, T, s, r in [(100, 90, 0.1, 0.6, 0.0), (2500, 2600, 3 / 365, 0.45, 0.0), (50, 55, 1.0, 0.3, 0.05)]:
        c, p = black76(F, K, T, s, "C", r), black76(F, K, T, s, "P", r)
        assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-9)


def test_delta_matches_finite_difference_and_parity():
    F, K, T, s = 2500.0, 2550.0, 3 / 365, 0.5
    h = 1e-3
    for cp in ("C", "P"):
        fd = (black76(F + h, K, T, s, cp) - black76(F - h, K, T, s, cp)) / (2 * h)
        assert black76_delta(F, K, T, s, cp) == pytest.approx(fd, abs=1e-6)
    assert black76_delta(F, K, T, s, "C") - black76_delta(F, K, T, s, "P") == pytest.approx(1.0, abs=1e-12)
    assert black76_delta(100, 100, 1.0, 0.2, "C") == pytest.approx(0.5398, abs=1e-4)  # N(0.1)
    c, p, d = straddle(F, F, T, s)
    assert c == pytest.approx(p, rel=1e-12) and d == pytest.approx(2 * black76_delta(F, F, T, s, "C") - 1)
    assert 0 < d < 0.03  # ATM-forward straddle delta = 2N(sigma sqrt(T)/2) - 1


def test_fee_cap():
    assert option_fee(60000, 1000) == pytest.approx(18.0)       # 0.03% of 60k
    assert option_fee(60000, 100) == pytest.approx(12.5)        # capped at 12.5% of a cheap option
    assert option_fee(60000, 144) == pytest.approx(18.0)        # boundary: 12.5% * 144 = 18


def _trades(rows):
    return pl.DataFrame(rows, schema=["timestamp", "instrument_name", "price", "mark_price", "iv", "index_price", "amount"],
                        orient="row")


def test_entry_quote_atm_and_fallback():
    ex = "ETH-12OCT26-"
    rows = [(i, f"{ex}{k}-{cp}", 0.01, 0.0098, iv, 2512.0, 1.0)
            for i, (k, cp, iv) in enumerate([(2450, "P", 50), (2475, "P", 48), (2500, "C", 46), (2525, "C", 47), (2600, "C", 60)])]
    rows.append((99, "ETH-30OCT26-2500-C", 0.05, 0.05, 55, 2511.0, 1.0))
    from datetime import date
    q = entry_quote(_trades(rows), date(2026, 10, 12))
    assert q["K"] == 2500 and q["step"] == 25 and q["iv_method"] == "atm_median"
    assert q["iv"] == pytest.approx(0.475)  # median of 2450..2550 trades: 50, 48, 46, 47
    assert q["half_spread"] == pytest.approx(0.0002 * 2512.0)
    # too few ATM trades -> amount-weighted near-money fallback
    rows2 = [(0, f"{ex}2500-C", 0.01, 0.01, 40, 2500.0, 3.0), (1, f"{ex}2525-C", 0.01, 0.01, 60, 2500.0, 1.0),
             (2, f"{ex}2700-C", 0.01, 0.01, 80, 2500.0, 5.0)]
    q2 = entry_quote(_trades(rows2), date(2026, 10, 12))
    assert q2["iv_method"] == "near_money_weighted" and q2["iv"] == pytest.approx(0.45)
    assert entry_quote(_trades(rows), date(2026, 10, 13))["skip"] == "no_trades_in_expiry"


def test_simulate_flat_path_keeps_premium_less_costs():
    q = {"F0": 100.0, "K": 100.0, "iv": 0.5, "half_spread": 0.01}
    path = np.full(73, 100.0)
    r = simulate(q, path)
    assert r["payoff"] == 0 and r["hedge_pnl"] == 0
    assert r["net"] == pytest.approx(r["premium"] - 0.02 - r["fees"] - r["hedge_cost"])
    assert r["rv"] == 0
    # a big one-way move loses more unhedged than hedged
    up = 100 * np.exp(np.linspace(0, 0.15, 73))
    r2 = simulate(q, up)
    assert r2["unhedged"] < 0 < r2["net"]  # smooth trend: little realised variance, the hedge keeps the premium
    jump = np.r_[np.full(36, 100.0), np.full(37, 112.0)]
    r3 = simulate(q, jump)
    assert r3["net"] < 0 and r3["net"] == pytest.approx(r3["unhedged"] + r3["hedge_pnl"] - r3["hedge_cost"])
