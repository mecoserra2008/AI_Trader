"""D7F: non-superiority cells, the intersection-union verdict and the stress output redirect."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_trader.research import d7f
from ai_trader.research import d7f_tost as T


def test_cell_upper_claim_and_opposite_p_sum_to_one():
    c = T.cell(-0.2, 0.1, 0.1)                       # claim est < 0.10: z = -3
    assert c["z"] == pytest.approx(-3.0)
    assert c["p_claim"] == pytest.approx(T.phi(-3.0)) and c["p_claim"] < 0.002
    assert c["p_claim"] + c["p_opp"] == pytest.approx(1.0)
    lower = T.cell(3.0, 1.0, 0.0, upper=False)        # claim est > 0 from a t of 3
    assert lower["p_claim"] == pytest.approx(1 - T.phi(3.0))
    assert math.isnan(T.cell(0.1, float("nan"), 0.0)["p_claim"])


def test_judge_is_intersection_union_with_holm_on_the_opposite_side():
    cells = {"a": T.cell(-0.5, 0.1, 0.0), "b": T.cell(-0.05, 0.1, 0.0)}
    j = T.judge(cells)
    assert j["p_iut"] == pytest.approx(cells["b"]["p_claim"]) and j["verdict"] == "inconclusive" and j["worst_cell"] == "b"
    assert T.judge({"a": T.cell(-0.5, 0.1, 0.0), "b": T.cell(-0.4, 0.1, 0.0)})["verdict"] == "supports"
    assert T.judge({"a": T.cell(0.5, 0.1, 0.0), "b": T.cell(-0.4, 0.1, 0.0)})["verdict"] == "contradicts"
    assert T.holm_min([0.01, 0.04, 0.5]) == pytest.approx(0.03)


def test_se_from_t_and_ci():
    assert T.se_t(0.2, 2.0) == pytest.approx(0.1)
    assert math.isnan(T.se_t(0.2, 0.0))
    assert T._ci_se({"lo": -0.196, "hi": 0.196}) == pytest.approx(0.1)


def test_out_proxy_reads_inputs_from_stress_and_writes_outputs_to_d7f(tmp_path: Path):
    src, dst = tmp_path / "stress", tmp_path / "d7f"
    p = d7f._OutProxy(src, dst)
    assert p / "trades" == src / "trades" and p / "design.json" == src / "design.json"
    assert p / "results.json" == dst / "results.json" and p / "plain_words.md" == dst / "plain_words.md"
