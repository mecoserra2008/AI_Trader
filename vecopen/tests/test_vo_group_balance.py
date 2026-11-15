"""Phase 3 amendment (2026-11-14): covariate balance per timeframe group (``phase3-balance``).

Synthetic tables in tmp dirs only; covariates only, no outcome column exists in any frame built here.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import polars as pl
import pytest
from vecopen import phase3
from vecopen.config import PROJECT_ROOT
from vecopen.placebo import MATCH_COVARIATES, P1, smd

COVS = list(MATCH_COVARIATES)


def test_groups_partition_the_primary_timeframes_as_in_spec_section_9(cfg):
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    grouped = [tf for g in phase3.TF_GROUPS.values() for tf in g]
    assert sorted(grouped) == sorted(tfs) and len(set(grouped)) == len(grouped)
    assert phase3.TF_GROUPS == {"intraday": ("15m", "30m", "1h", "2h", "4h"), "daily_plus": ("1d", "1w", "1M")}
    spec = (PROJECT_ROOT / "SPEC.md").read_text(encoding="utf-8")
    assert "**intraday:** 15m, 30m, 1h, 2h and 4h" in spec and "**daily-and-higher:** 1d, 1w and 1M" in spec
    assert phase3.BALANCE_SMD_MAX == 0.1


def _pairs(rng: np.random.Generator, cells: dict[tuple[str, str], int], shift: float = 0.0) -> pl.DataFrame:
    parts = []
    for (a, tf), n in cells.items():
        d = {"asset": [a] * n, "tf": [tf] * n}
        for c in COVS:
            d[f"t_{c}"] = rng.normal(1.0, 1.0, n) + shift
            d[f"p_{c}"] = rng.normal(1.0, 1.3, n)
        parts.append(pl.DataFrame(d))
    return pl.concat(parts)


def test_pooled_balance_equals_smd_of_the_pooled_pairs():
    rng = np.random.default_rng(7)
    pairs = _pairs(rng, {("A", "15m"): 300, ("B", "15m"): 200, ("A", "4h"): 50, ("B", "1d"): 40, ("A", "1M"): 9})
    bal = phase3.pooled_balance(pairs)
    for g, tfs in phase3.TF_GROUPS.items():
        sub = pairs.filter(pl.col("tf").is_in(list(tfs)))
        e = bal["groups"][g]
        assert e["n_pairs"] == sub.height and e["timeframes"] == list(tfs)
        for c in COVS:
            want = smd(sub[f"t_{c}"].to_numpy(), sub[f"p_{c}"].to_numpy())
            assert e["covariates"][c]["smd"] == pytest.approx(want, rel=1e-9, abs=1e-12)
            assert e["covariates"][c]["mean_t"] == pytest.approx(sub[f"t_{c}"].mean())
        assert e["max_abs_smd"] == pytest.approx(max(abs(e["covariates"][c]["smd"]) for c in COVS))
        assert e["pass"] == (e["max_abs_smd"] < 0.1)
    # breakdown per timeframe (only timeframes with pairs) and the overall row
    assert list(bal["by_tf"]) == ["15m", "4h", "1d", "1M"]
    x = pairs.filter(pl.col("tf") == "15m")
    assert bal["by_tf"]["15m"]["n_pairs"] == 500 and bal["by_tf"]["15m"]["group"] == "intraday"
    assert bal["by_tf"]["15m"]["covariates"]["regime"]["smd"] == pytest.approx(
        smd(x["t_regime"].to_numpy(), x["p_regime"].to_numpy()))
    assert bal["all"]["n_pairs"] == pairs.height
    assert bal["met"] == all(e["pass"] for e in bal["groups"].values())


def test_pooled_balance_flags_an_imbalanced_group_and_unknown_timeframes():
    rng = np.random.default_rng(3)
    good = _pairs(rng, {("A", "1h"): 2000})
    bad = _pairs(rng, {("A", "1w"): 2000}, shift=0.5)
    bal = phase3.pooled_balance(pl.concat([good, bad]))
    assert not bal["groups"]["daily_plus"]["pass"] and not bal["met"]
    md = phase3.group_balance_md(bal)
    assert "**no**" in md and "not met" in md
    # a group without pairs is not a pass
    only = phase3.pooled_balance(good)
    assert only["groups"]["daily_plus"]["n_pairs"] == 0 and not only["groups"]["daily_plus"]["pass"]
    with pytest.raises(ValueError, match="no group"):
        phase3.pooled_balance(_pairs(rng, {("A", "5m"): 10}))


def _write_tables(d, events: dict[str, pl.DataFrame], matches: dict[str, pl.DataFrame]) -> None:
    for a in events:
        (d / a).mkdir(parents=True, exist_ok=True)
        events[a].write_parquet(d / a / "events.parquet")
        matches[a].write_parquet(d / a / "matches.parquet")


def _ev(asset: str, rows: list[tuple[str, str, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame({"asset": [asset] * len(rows), "family": [r[0] for r in rows],
                         "event_key": [r[1] for r in rows], "age_log": [r[2] for r in rows],
                         "departure_atr": [r[3] for r in rows], "regime": [r[4] for r in rows]})


def _mt(asset: str, rows: list[tuple[str, str, str, str | None, str | None]]) -> pl.DataFrame:
    return pl.DataFrame({"asset": [asset] * len(rows), "tf": [r[0] for r in rows], "year": [2021] * len(rows),
                         "treated_key": [r[1] for r in rows], "placebo_asset": [r[2] for r in rows],
                         "placebo_key": [r[3] for r in rows], "distance": [0.1] * len(rows), "cell": ["asset"] * len(rows)},
                        schema_overrides={"placebo_asset": pl.String, "placebo_key": pl.String})


def test_load_matched_covariates_reads_covariates_only_and_follows_cross_asset_matches(tmp_path):
    # the event tables hold keys and covariates only, and no labels/levels table exists: nothing else can be read
    ev = {"A": _ev("A", [("treated", "treated:1", 1.0, 2.0, 0.1), ("treated", "treated:2", 2.0, 3.0, 0.2),
                         ("treated", "treated:3", 0.5, 2.5, 0.0), (P1, f"{P1}:1", 1.1, 2.1, 0.1),
                         ("p2_displaced", "p2_displaced:1", 9.0, 9.0, 9.0)]),
          "B": _ev("B", [(P1, f"{P1}:1", 2.2, 3.3, -0.4)])}
    mt = {"A": _mt("A", [("15m", "treated:1", "A", f"{P1}:1"), ("1d", "treated:2", "B", f"{P1}:1"),
                         ("15m", "treated:3", None, None)]),                       # unmatched: dropped
          "B": _mt("B", [])}
    _write_tables(tmp_path, ev, mt)
    assert not list(tmp_path.rglob("labels.parquet")) and not list(tmp_path.rglob("levels.parquet"))
    out = phase3.load_matched_covariates(tmp_path, ["A", "B"]).sort("tf")
    assert out.columns == ["asset", "tf", *[f"t_{c}" for c in COVS], *[f"p_{c}" for c in COVS]]
    assert out.height == 2
    r = out.filter(pl.col("tf") == "1d").row(0, named=True)
    assert (r["t_age_log"], r["p_age_log"], r["p_departure_atr"], r["p_regime"]) == (2.0, 2.2, 3.3, -0.4)
    r = out.filter(pl.col("tf") == "15m").row(0, named=True)
    assert (r["t_departure_atr"], r["p_departure_atr"]) == (2.0, 2.1)


def test_load_matched_covariates_refuses_a_pair_without_covariates(tmp_path):
    ev = {"A": _ev("A", [("treated", "treated:1", 1.0, 2.0, 0.1), (P1, f"{P1}:1", 1.0, 2.0, float("nan"))])}
    _write_tables(tmp_path, ev, {"A": _mt("A", [("15m", "treated:1", "A", f"{P1}:1")])})
    with pytest.raises(ValueError, match="finite covariate"):
        phase3.load_matched_covariates(tmp_path, ["A"])
    ev = {"A": _ev("A", [("treated", "treated:1", 1.0, 2.0, 0.1)])}             # placebo key not in the table
    _write_tables(tmp_path, ev, {"A": _mt("A", [("15m", "treated:1", "A", f"{P1}:9")])})
    with pytest.raises(ValueError, match="finite covariate"):
        phase3.load_matched_covariates(tmp_path, ["A"])


def test_verify_phase3_inputs_checks_the_manifest_and_every_file(tmp_path):
    ev = {"A": _ev("A", [("treated", "treated:1", 1.0, 2.0, 0.1)])}
    _write_tables(tmp_path, ev, {"A": _mt("A", [("15m", "treated:1", None, None)])})
    sha = {n: hashlib.sha256((tmp_path / "A" / f"{n}.parquet").read_bytes()).hexdigest() for n in ("events", "matches")}
    man = {"assets": {"A": {n: {"path": f"A/{n}.parquet", "sha256": s} for n, s in sha.items()}}}
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    msha = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    src = {"line_sha256": "x" * 64, "metrics": {"manifest_sha256": msha, "assets": ["A"]}}
    assert phase3.verify_phase3_inputs(tmp_path, src) == (["A"], msha)
    with pytest.raises(ValueError, match="manifest.json"):
        phase3.verify_phase3_inputs(tmp_path, {**src, "metrics": {"manifest_sha256": "0" * 64, "assets": ["A"]}})
    _ev("A", [("treated", "treated:1", 1.0, 2.0, 0.2)]).write_parquet(tmp_path / "A" / "events.parquet")
    with pytest.raises(ValueError, match="events.parquet differs"):
        phase3.verify_phase3_inputs(tmp_path, src)


def test_group_balance_path_never_reads_outcomes():
    # the readers name only the events and matches tables, and only key and covariate columns of events
    for fn in (phase3.load_matched_covariates, phase3.verify_phase3_inputs, phase3.cmd_phase3_balance):
        src = inspect.getsource(fn)
        for banned in ("labels.parquet", "levels.parquet", '"labels"', "_label", "touch", "fill_ts"):
            assert banned not in src, (fn.__name__, banned)
    assert set(phase3._EVENT_COV_COLS) == {"asset", "event_key", "family"}
    assert set(phase3._MATCH_COLS) == {"asset", "tf", "treated_key", "placebo_asset", "placebo_key"}
