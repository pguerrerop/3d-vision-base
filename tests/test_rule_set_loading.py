from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_3d_acquisition.classifiers.mining_ball_rules import (
    DEFAULT_RULE_PARAMS,
    list_available_rule_sets,
    load_classifier_rule_config,
    predict_superclass_from_rules,
    resolve_classifier_rule_set,
)


def test_load_classifier_rule_config_defaults_and_unknown_params(tmp_path: Path) -> None:
    cfg = tmp_path / "rules.json"
    cfg.write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "test_v1",
                "params": {
                    "good_min_sphericity": 0.72,
                    "unknown_knob": 123,
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_classifier_rule_config(cfg)
    assert loaded["params"]["good_min_sphericity"] == 0.72
    assert loaded["params"]["good_max_eccentricity"] == DEFAULT_RULE_PARAMS["good_max_eccentricity"]
    assert "unknown_knob" in loaded["unknown_params"]


def test_load_classifier_rule_config_malformed_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.json"
    cfg.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_classifier_rule_config(cfg)


def test_predict_superclass_from_rules_deterministic() -> None:
    row = {
        "feature_sphericity_3d": 0.88,
        "feature_eccentricity": 0.2,
        "feature_flatness": 0.3,
        "feature_edge_roughness": 2.0,
        "max_height_mm": 35.0,
        "p95_height_mm": 16.0,
        "feature_volume_proxy_mm3": 15000.0,
    }
    a = predict_superclass_from_rules(row, dict(DEFAULT_RULE_PARAMS))
    b = predict_superclass_from_rules(row, dict(DEFAULT_RULE_PARAMS))
    assert a.superclass == b.superclass
    assert a.rule_path == b.rule_path


def test_list_available_rule_sets(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs" / "classifiers"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "demo_rules.json").write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "demo_v1",
                "description": "Demo rules",
                "metadata": {
                    "dataset_id": "d1",
                    "ml_set_id": "m1",
                    "created_at": "2026-05-29T00:00:00Z",
                    "optimized_metric": "macro_f1",
                    "validation_macro_f1": 0.91,
                    "source_type": "tuned",
                },
                "params": {},
            }
        ),
        encoding="utf-8",
    )
    rows = list_available_rule_sets(cfg_dir)
    assert len(rows) == 1
    assert rows[0]["rule_set_id"] == "demo_rules"
    assert rows[0]["classifier_id"] == "mining_steel_ball_classification_25d_rules"
    assert rows[0]["optimized_metric"] == "macro_f1"
    assert rows[0]["source_type"] == "tuned"


def test_resolve_classifier_rule_set_precedence_runtime_over_pipeline_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_cfg = tmp_path / "runtime.json"
    runtime_cfg.write_text(json.dumps({"version": "runtime_v1", "params": {"good_min_sphericity": 0.73}}), encoding="utf-8")
    pipeline_cfg = tmp_path / "pipeline.json"
    pipeline_cfg.write_text(json.dumps({"version": "pipeline_v1", "params": {"good_min_sphericity": 0.66}}), encoding="utf-8")
    env_cfg = tmp_path / "env.json"
    env_cfg.write_text(json.dumps({"version": "env_v1", "params": {"good_min_sphericity": 0.61}}), encoding="utf-8")
    monkeypatch.setenv("SENSOR_STUDIO_DEFAULT_RULE_SET", str(env_cfg))
    resolved = resolve_classifier_rule_set(
        runtime_override_path=runtime_cfg,
        pipeline_config_path=pipeline_cfg,
    )
    assert resolved.rule_set_source == "runtime_override"
    assert resolved.rule_set_version == "runtime_v1"


def test_resolve_classifier_rule_set_precedence_pipeline_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_cfg = tmp_path / "pipeline.json"
    pipeline_cfg.write_text(json.dumps({"version": "pipeline_v1", "params": {"good_min_sphericity": 0.66}}), encoding="utf-8")
    env_cfg = tmp_path / "env.json"
    env_cfg.write_text(json.dumps({"version": "env_v1", "params": {"good_min_sphericity": 0.61}}), encoding="utf-8")
    monkeypatch.setenv("SENSOR_STUDIO_DEFAULT_RULE_SET", str(env_cfg))
    resolved = resolve_classifier_rule_set(pipeline_config_path=pipeline_cfg)
    assert resolved.rule_set_source == "pipeline_config"
    assert resolved.rule_set_version == "pipeline_v1"


def test_resolve_classifier_rule_set_uses_env_when_no_runtime_or_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_cfg = tmp_path / "env.json"
    env_cfg.write_text(json.dumps({"version": "env_v1", "params": {"good_min_sphericity": 0.61}}), encoding="utf-8")
    monkeypatch.setenv("SENSOR_STUDIO_DEFAULT_RULE_SET", str(env_cfg))
    resolved = resolve_classifier_rule_set()
    assert resolved.rule_set_source == "env_default"
    assert resolved.rule_set_version == "env_v1"


def test_resolve_classifier_rule_set_builtin_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENSOR_STUDIO_DEFAULT_RULE_SET", raising=False)
    resolved = resolve_classifier_rule_set()
    assert resolved.rule_set_source == "builtin_default"
    assert resolved.rule_set_id == "builtin_default"
