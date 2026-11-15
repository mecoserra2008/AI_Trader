"""load_config(): parsed baseline plus sha256 (phase 0)."""

from __future__ import annotations

import hashlib

import pytest

from vecopen import CONFIG_PATH, LoadedConfig, config_sha256, load_config


def test_returns_parsed_config_and_hash(cfg):
    assert isinstance(cfg, LoadedConfig)
    data, sha, path = cfg  # unpacks as (config, sha256, path)
    assert path == CONFIG_PATH
    assert data["version"] == "baseline-1"
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
    print(f"config sha256: {sha}")


def test_hash_is_sha256_of_lf_normalised_bytes(cfg):
    raw = CONFIG_PATH.read_bytes()
    assert cfg.sha256 == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert load_config().sha256 == cfg.sha256  # stable across loads


def test_hash_ignores_line_endings_but_not_content(tmp_path, cfg):
    raw = CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")
    crlf = tmp_path / "crlf.yaml"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert load_config(crlf).sha256 == cfg.sha256
    edited = tmp_path / "edited.yaml"
    edited.write_bytes(raw.replace(b"seed: 20261114", b"seed: 20261115"))
    other = load_config(edited)
    assert other.sha256 != cfg.sha256
    assert other.data["seed"] == 20261115
    assert config_sha256(b"a\r\nb") == config_sha256(b"a\nb")


def test_frozen_values_match_spec_section_7(cfg):
    d = cfg.data
    assert d["seed"] == 20261114
    assert d["legs"] == {"theta_L_atr": 2.0, "n_confirm": 6, "omega_max": 0.25, "atr_len": 14,
                         "volume_baseline_sessions": 20}
    assert d["levels"]["kappa_atr"] == 0.10
    assert d["levels"]["a_max_bars"] == 200
    assert d["levels"]["timeframe_weights"] == {tf: i + 1 for i, tf in enumerate(
        ["5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"])}
    assert d["levels"]["timeframes_primary"][0] == "15m"
    assert d["labels"] == {"k_pt_atr": 1.0, "k_sl_atr": 0.5, "h_tau_bars": 6, "tie_rule": "stop_first"}
    assert d["events"] == {"base_timeframe": "5m", "approach_lookback_m": 6}
    assert d["costs"]["stress_multiplier"] == 2.0
    assert d["split"] == {"research_fraction": 0.75, "holdout_fraction": 0.25}
    assert d["stats"]["bootstrap_resamples"] == 10000
    assert d["stats"]["holm_alpha"] == 0.05
    assert d["stats"]["placebo_displacement_atr"] == [1.0, 3.0]
    assert d["cv"] == {"cpcv_groups": 10, "cpcv_test_groups": 2, "m2_min_events": 1000}
    assert d["gate_g3"] == {"dsr_min": 0.95, "pbo_max": 0.20, "wf_positive_fraction": 0.667}


def test_baseline_is_centre_of_robustness_grid(cfg):
    grid, legs = cfg.data["robustness_grid"], cfg.data["legs"]
    assert set(grid) == {"theta_L_atr", "n_confirm", "omega_max"}
    for k, values in grid.items():
        assert len(values) == 3 and values == sorted(values)
        assert values[1] == legs[k]


def test_session_conventions_fixed_per_asset_class(cfg):
    a = cfg.data["assets"]
    assert a["crypto"]["session"] == "utc_24h" and a["crypto"]["aggressor_volume"] is True
    assert a["cfd"]["session"] == "exchange" and a["cfd"]["aggressor_volume"] is False


@pytest.mark.parametrize("body, msg", [
    (b"- just\n- a list\n", "mapping"),
    (b"version: x\nseed: 1\n", "missing required"),
])
def test_malformed_config_rejected(tmp_path, body, msg):
    p = tmp_path / "bad.yaml"
    p.write_bytes(body)
    with pytest.raises(ValueError, match=msg):
        load_config(p)


def test_non_integer_seed_rejected(tmp_path):
    raw = CONFIG_PATH.read_bytes().replace(b"seed: 20261114", b"seed: 1.5")
    p = tmp_path / "seed.yaml"
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="seed"):
        load_config(p)
