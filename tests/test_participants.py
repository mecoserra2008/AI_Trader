import numpy as np
import pytest

from ai_trader.research import participants as P


def test_cks_ofi_signs():
    # bid price up: + new bid size; ask unchanged: + old ask - new ask = 0
    assert P.cks_ofi(101, 2, 102, 5, 100, 3, 102, 5) == pytest.approx(2.0)
    # ask price down: - new ask size
    assert P.cks_ofi(100, 3, 101, 4, 100, 3, 102, 5) == pytest.approx(-4.0)
    # bid size shrinks at the same price: + qb - qb0
    assert P.cks_ofi(100, 1, 102, 5, 100, 3, 102, 5) == pytest.approx(-2.0)


def test_flowbook_adds_removals_refill_and_best():
    fb = P.FlowBook(near_bp=50)
    fb.apply("snapshot", [["100", "1"], ["99.9", "2"]], [["100.1", "1"], ["100.2", "3"]])
    assert fb.bb == 100 and fb.ba == 100.1
    fb.reset_bin()
    fb.apply("delta", [["100", "0.4"]], [])            # removal at best bid
    fb.apply("delta", [["100", "1.0"]], [])            # refill at the same best price
    fb.apply("delta", [["100.05", "2"]], [])           # new best bid
    fb.apply("delta", [], [["100.1", "0"]])            # best ask removed -> next level
    acc = fb.acc
    assert acc["rem_bid"] == pytest.approx(0.6 * 100)
    assert acc["refill_bid"] == pytest.approx(0.6 * 100)
    assert acc["add_bid"] == pytest.approx(0.6 * 100 + 2 * 100.05)
    assert acc["rem_ask"] == pytest.approx(100.1)
    assert fb.bb == 100.05 and fb.ba == 100.2
    assert acc["n_updates"] == 4 and acc["n_best_moves"] == 2


def test_flowbook_ignores_far_levels_and_deltas_before_snapshot():
    fb = P.FlowBook(near_bp=5)
    assert fb.apply("delta", [["1", "1"]], []) is False
    fb.apply("snapshot", [["100", "1"]], [["100.1", "1"]])
    fb.apply("delta", [["90", "5"]], [])
    assert fb.acc["add_bid"] == 0.0 and fb.acc["n_add"] == 0


def test_reduce_book_grid_and_bins():
    d0 = P.day0_ms("2026-10-06")
    msgs = [{"ts": d0 + 10, "type": "snapshot", "data": {"b": [["100", "1"]], "a": [["100.1", "1"]], "u": 1}},
            {"ts": d0 + 1500, "type": "delta", "data": {"b": [["100", "3"]], "a": [], "u": 2}}]
    df, qa = P.reduce_book(iter(msgs), d0)
    assert df.height == 86_400
    assert df["ts"][0] == d0 + 1000
    assert df["add_bid"][1] == pytest.approx(200.0) and df["add_bid"][0] == 0.0
    assert df["bid1_size"][5] == pytest.approx(3.0)   # carried forward
    assert qa["snapshots"] == 1


def test_sig_digits():
    assert P.sig_digits(np.array([0.5, 0.25, 12, 1.234, 0.001, 100])).tolist() == [1, 2, 2, 4, 1, 1]


def test_trailing_count_gap_clip():
    ms = np.array([0, 500, 900, 2000, 2001])
    assert P.trailing_count(ms, 1000).tolist() == [0, 1, 2, 0, 1]
    g = P.gap_prev(ms, cap_ms=60_000)
    assert g.tolist() == [60_000, 500, 400, 1100, 1]
    assert P.clip_count(np.array([1.0, 2.0, 1.0, 1.0, 3.0]), lags=20).tolist() == [0, 0, 1, 2, 0]


def test_by_side_scatter():
    side = np.array([1, -1, 1, -1])
    ms = np.array([0, 10, 30, 100])
    g = P.by_side(lambda m: P.gap_prev(m, 999.0), side, ms)
    assert g.tolist() == [999.0, 999.0, 30.0, 90.0]


def test_asof_index_is_point_in_time():
    bts = np.array([1000, 2000, 3000])
    assert P.asof_index(bts, np.array([999, 1000, 2999])).tolist() == [-1, 0, 1]


def test_choose_k_rule():
    st = {2: {"ari": 0.95, "bic": 100.0}, 3: {"ari": 0.85, "bic": 80.0}, 4: {"ari": 0.6, "bic": 50.0}}
    assert P.choose_k(st) == 3
    st2 = {2: {"ari": 0.5, "bic": 1.0}, 3: {"ari": 0.7, "bic": 2.0}}
    assert P.choose_k(st2) == 3


def test_forward_filter_matches_manual_two_state():
    start = np.array([0.5, 0.5])
    trans = np.array([[0.9, 0.1], [0.2, 0.8]])
    lb = np.log(np.array([[0.8, 0.1], [0.3, 0.6], [0.5, 0.5]]))
    out = P.forward_filter(lb, start, trans)
    a = start * np.exp(lb[0])
    a /= a.sum()
    ref = [a]
    for t in (1, 2):
        a = (a @ trans) * np.exp(lb[t])
        a /= a.sum()
        ref.append(a)
    assert np.allclose(out, np.array(ref))
    # point in time: appending a future observation does not change earlier filtered probabilities
    out2 = P.forward_filter(np.vstack([lb, np.log([[0.99, 0.01]])]), start, trans)
    assert np.allclose(out2[:3], out)


def test_day_boot_point_and_holm():
    pdict = {"a": (10.0, 9.0, 100.0), "b": (20.0, 19.0, 100.0)}
    r = P.day_boot(pdict, n_boot=200)
    assert r["dr2"] == pytest.approx(0.01)
    assert r["p"] == 0.0
    assert P.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_minute_mapping():
    d0 = 0
    assert P.minute_of_ms(np.array([0, 59_999, 60_000]), d0).tolist() == [0, 0, 1]
    # book bin ending at 60 000 lies in minute 0; the bin ending at 61 000 in minute 1
    assert P.minute_of_book_ts(np.array([1000, 60_000, 61_000]), d0).tolist() == [0, 0, 1]


def test_trailing_and_forward_windows():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert np.allclose(P.trailing_mean(x, 2), [1.0, 1.5, 2.0, 4.0])
    assert np.allclose(P.trailing_sum(x, 2), [1.0, 3.0, 2.0, 4.0])
    f = P.forward_sum(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert np.allclose(f[:2], [5.0, 7.0]) and np.isnan(f[2:]).all()


def test_ewm_var_point_in_time():
    r = np.array([np.nan, 0.01, -0.02, 0.03])
    v = P.ewm_var(r)
    assert np.isnan(v[0]) and v[1] == pytest.approx(1e-4)
    v2 = P.ewm_var(np.r_[r, 5.0])
    assert np.allclose(v2[:4], v, equal_nan=True)


def test_cv_blocks_embargo():
    days = ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06"]
    bl = P.cv_blocks(days, n_blocks=3, embargo_days=1)
    tr, te = bl[1]
    assert te == ["2026-10-03", "2026-10-04"]
    assert tr == ["2026-10-01", "2026-10-06"]


def test_minute_as_needs_min_orders():
    ms = np.array([0, 1, 2, 3, 4, 60_000])
    a = P.minute_as(ms, np.ones(6), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 9.0]), 0, min_orders=5)
    assert a[0] == pytest.approx(3.0) and np.isnan(a[1])


def test_split_and_segments():
    assert P.split_of("2026-10-29") == "FIT" and P.split_of("2026-10-30") == "HELD"
    assert P.segment_of("2026-06-21") == "llm_in_sample" and P.segment_of("2026-10-06") == "clean_holdout"
    days = P.all_days()
    assert len(days) == 62 and len([d for d in days if P.split_of(d) == "FIT"]) == 36
