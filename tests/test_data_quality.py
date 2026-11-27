import numpy as np
import polars as pl

from ai_trader.research import data_quality as dq


def test_missing_runs_interior_leading_trailing():
    ts = np.array([3, 4, 7, 8]) * 3000
    got = dq.missing_runs(ts, 0, 12 * 3000, 3000)
    assert got == [(0, 9000, 3), (15000, 21000, 2), (27000, 36000, 3)]
    assert dq.missing_runs(np.arange(0, 30000, 3000), 0, 30000, 3000) == []


def test_true_runs_needs_contiguity_and_length():
    ts = np.arange(10) * 3000
    flag = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=bool)
    assert dq.true_runs(ts, flag, 3000, 3) == [(0, 9000, 3), (21000, 30000, 3)]
    ts2 = np.array([0, 3000, 9000]) * 1
    assert dq.true_runs(ts2, np.ones(3, bool), 3000, 3) == []  # not contiguous on the grid


def test_merge_overlap_and_cause():
    iv = dq.merge([(10, 20), (15, 30), (40, 50)])
    assert iv == [(10, 30), (40, 50)]
    assert dq.overlap_ms(0, 100, iv) == 30
    assert dq.cause_of(12, 28, iv) == "machine_down"
    assert dq.cause_of(30, 40, iv) == "unknown"


def test_dup_stats_exact_conflicting_revision():
    df = pl.DataFrame({"symbol": ["A", "A", "A", "A", "B", "B"], "ev": [1, 1, 2, 2, 1, 1],
                       "kn": [5, 5, 6, 6, 5, 9], "v": [1.0, 1.0, 1.0, 2.0, 1.0, 3.0]})
    s = dq.dup_stats(df, ["symbol", "ev"], ["v"])
    assert s == {"dup_keys": 3, "dup_rows_extra": 3, "dup_conflicting": 1, "revised_keys": 1}


def test_merge_gap_rows_joins_touching_holes():
    g = pl.DataFrame([
        {"dataset": "flow_3s", "stream": "s", "symbol": "X", "gap_start": 0, "gap_end": 10, "n_missing": 1, "kind": "missing_rows", "cause": "c", "fillable_from": "", "filled": ""},
        {"dataset": "flow_3s", "stream": "s", "symbol": "X", "gap_start": 10, "gap_end": 20, "n_missing": 1, "kind": "missing_rows", "cause": "c", "fillable_from": "", "filled": ""},
        {"dataset": "flow_3s", "stream": "s", "symbol": "X", "gap_start": 30, "gap_end": 40, "n_missing": 1, "kind": "missing_rows", "cause": "c", "fillable_from": "", "filled": ""},
    ], schema=dq.GAP_SCHEMA)
    m = dq.merge_gap_rows(g)
    assert m.select("gap_start", "gap_end", "n_missing").rows() == [(0, 20, 2), (30, 40, 1)]


def test_funding_gaps_allow_interval_change_but_flag_holes():
    h = 3_600_000
    ts = [i * 8 * h for i in range(10)] + [9 * 8 * h + 4 * h * k for k in range(1, 10)]   # 8h regime, then 4h regime
    ts.remove(3 * 8 * h)                                                               # one missing 8h settlement
    df = pl.DataFrame({"symbol": ["A"] * len(ts), "ev": ts})
    st, gaps = dq.funding_gaps(df, [])
    assert st["misaligned_hour"] == 0
    assert len(gaps) == 1 and gaps[0]["gap_start"] == 3 * 8 * h and gaps[0]["n_missing"] == 1
