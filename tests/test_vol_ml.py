import json
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("xgboost")

from ai_trader.ml.frozen import FREEZE_AS_OF, freeze, load_frozen, panel_asof  # noqa: E402
from ai_trader.ml.hourly import HOURLY_FEATURES, symbol_hourly  # noqa: E402
from ai_trader.research import vol_ml_forward_study as fw  # noqa: E402
from ai_trader.research.frame import market_frame  # noqa: E402
from ai_trader.storage.lake import Lake  # noqa: E402
from ai_trader.tools.vol_ml import hourly_vol_forecast  # noqa: E402
from tests.conftest import T0, make_bars  # noqa: E402

N_H = 24 * 120
CFG = {"max_depth": 3, "min_child_weight": 20, "max_rounds": 60, "early_stop": 20}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("volml")
    bars = make_bars(N_H, "TESTUSDT", seed=7)
    full = Lake(root=tmp / "full")
    full.write("bars", bars)
    t_fit = T0 + timedelta(hours=24 * 100)
    df = panel_asof(["TESTUSDT"], t_fit, lake=full)
    card = freeze(df, tmp / "model_v_test", CFG, as_of=t_fit, nthread=1)
    return {"tmp": tmp, "full": full, "bars": bars, "t_fit": t_fit, "model": tmp / "model_v_test", "card": card}


def test_frozen_card_and_reload(setup):
    card = json.loads((setup["model"] / "card.json").read_text(encoding="utf-8"))
    assert card["features"] == list(HOURLY_FEATURES) and card["best_iteration"] >= 0
    assert card["training"]["last_bar"] < str(setup["t_fit"])
    m1, m2 = load_frozen(setup["model"]), load_frozen(setup["model"])
    X = np.zeros((3, len(HOURLY_FEATURES)))
    assert np.array_equal(m1.margin(X), m2.margin(X))
    assert np.allclose(m1.contribs(X).sum(1), m1.margin(X), atol=1e-5)


def test_tool_is_point_in_time_and_matches_training_rows(setup):
    as_of = T0 + timedelta(hours=24 * 110, minutes=1)
    # a lake that only holds bars known by as_of gives the same output as the full lake queried at as_of
    part = Lake(root=setup["tmp"] / "part")
    part.write("bars", setup["bars"].filter(pl.col("known_ts") <= as_of))
    a = hourly_vol_forecast("TESTUSDT", as_of, lake=setup["full"], model_path=setup["model"])
    b = hourly_vol_forecast("TESTUSDT", as_of, lake=part, model_path=setup["model"])
    assert a == b
    assert a["flags"] == [] and a["sigma_next_hour_ml"] > 0 and a["sigma_next_hour_baseline"] > 0
    assert a["data_max_ts"] <= as_of.isoformat()
    # the next bar's features equal the row the model sees for that bar once it exists (training-time computation)
    later = T0 + timedelta(hours=N_H + 1)
    f = market_frame("TESTUSDT", "1h", later, funding=False, oi=False, lake=setup["full"])
    h = symbol_hourly(f.select("event_ts", "ret", "sigma_risk_next"))
    row = h.filter(pl.col("event_ts").cast(pl.Datetime("us", "UTC")) == pl.lit(a["forecast_for"]).str.to_datetime(time_zone="UTC"))
    assert row.height == 1
    for c in HOURLY_FEATURES:
        assert np.isclose(row[c][0], a["features"][c], rtol=1e-9, atol=1e-12), c
    assert np.isclose(row["s2"][0] * row["fac"][0], a["sigma_next_hour_baseline"] ** 2, rtol=1e-9)


def test_tool_output_is_deterministic_and_flags_stale(setup):
    as_of = T0 + timedelta(hours=24 * 105, minutes=1)
    a = hourly_vol_forecast("TESTUSDT", as_of, lake=setup["full"], model_path=setup["model"])
    b = hourly_vol_forecast("TESTUSDT", as_of, lake=setup["full"], model_path=setup["model"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    late = T0 + timedelta(hours=N_H + 5)
    c = hourly_vol_forecast("TESTUSDT", late, lake=setup["full"], model_path=setup["model"])
    assert "stale" in c["flags"]


def test_forward_study_refuses_to_record_early(tmp_path, monkeypatch):
    log = tmp_path / "log"
    log.mkdir()
    ff = fw.START + timedelta(hours=1)
    line = {"logged_at": ff.isoformat(), "symbol": "TESTUSDT", "forecast_for": ff.isoformat(), "data_max_ts": (ff + timedelta(seconds=2)).isoformat(),
            "sigma_next_hour_ml": 0.01, "sigma_next_hour_baseline": 0.011, "inputs_hash": "x", "model_id": "m", "flags": []}
    (log / "a.jsonl").write_text(json.dumps(line) + "\n" + json.dumps(line | {"logged_at": (ff + timedelta(minutes=5)).isoformat()}) + "\n", encoding="utf-8")
    lg = fw.admissible(fw.read_log(log))
    assert lg.height == 1
    called = []
    monkeypatch.setattr("ai_trader.research.hypotheses.record", lambda *a, **k: called.append(a))
    out = fw.run(as_of=fw.START + timedelta(days=2), record=True, lake=Lake(root=tmp_path / "empty"), log=log, model_path=tmp_path / "none")
    assert out["verdict"] == "pending" and out["record"].startswith("refused") and not called
    assert fw.verdict({"n_dates": 59, "diff": -1.0, "t": -9.0}) == "pending"
    assert fw.verdict({"n_dates": 60, "diff": -1.0, "t": -9.0}) == "supports"
    assert FREEZE_AS_OF < fw.START
