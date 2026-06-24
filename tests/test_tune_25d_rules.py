from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tune_25d_rules import (
    DEFAULT_PARAMS,
    evaluate_rule_params,
    expected_superclass_from_row,
    make_object_safe_splits,
    predict_superclass_from_rules,
    run_tuning,
)


def _write_features_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_expected_label_mapping_to_superclass() -> None:
    assert expected_superclass_from_row({"expected_label": "buena"}) == "BALL_GOOD"
    assert expected_superclass_from_row({"expected_label": "mitad"}) == "BALL_SCRAP"
    assert expected_superclass_from_row({"expected_label": "planchuela"}) == "SCRAP_METAL"
    assert expected_superclass_from_row({"expected_superclass": "ball_good"}) is None
    assert expected_superclass_from_row({"expected_superclass": "BALL_GOOD"}) == "BALL_GOOD"


def test_hierarchical_rule_ordering_scrap_precedes_good() -> None:
    row = {
        "feature_sphericity_3d": 0.2,
        "max_height_mm": 20.0,
        "p95_height_mm": 10.0,
        "feature_volume_proxy_mm3": 9000.0,
        "feature_flatness": 0.4,
        "feature_eccentricity": 0.2,
        "feature_edge_roughness": 2.0,
    }
    pred = predict_superclass_from_rules(row, dict(DEFAULT_PARAMS))
    assert pred.superclass == "SCRAP_METAL"
    assert pred.rule_path.startswith("matched_scrap")


def test_object_safe_split_behavior() -> None:
    rows = [
        {"take_id": "t1", "physical_object_id": "obj_a"},
        {"take_id": "t2", "physical_object_id": "obj_a"},
        {"take_id": "t3", "physical_object_id": "obj_b"},
    ]
    make_object_safe_splits(
        rows,
        group_column="physical_object_id",
        seed=42,
        train_name="train",
        validation_name="validation",
        test_name="test",
    )
    split_by_take = {row["take_id"]: row["split"] for row in rows}
    assert split_by_take["t1"] == split_by_take["t2"]


def test_evaluate_metrics_and_rule_path_generation() -> None:
    rows = [
        {
            "take_id": "t1",
            "expected_superclass": "BALL_GOOD",
            "split": "train",
            "feature_sphericity_3d": 0.9,
            "feature_eccentricity": 0.2,
            "feature_flatness": 0.3,
            "feature_edge_roughness": 2.0,
            "max_height_mm": 40.0,
            "p95_height_mm": 20.0,
            "feature_volume_proxy_mm3": 20000.0,
        },
        {
            "take_id": "t2",
            "expected_superclass": "SCRAP_METAL",
            "split": "train",
            "feature_sphericity_3d": 0.1,
            "max_height_mm": 3.0,
            "p95_height_mm": 2.0,
            "feature_volume_proxy_mm3": 800.0,
            "feature_flatness": -0.6,
            "feature_eccentricity": 0.95,
            "feature_edge_roughness": 30.0,
        },
    ]
    out = evaluate_rule_params(rows, dict(DEFAULT_PARAMS), split="train")
    assert out["rows"] == 2
    assert out["accuracy"] >= 0.5
    assert all("rule_path" in row for row in out["predictions"])


def test_run_tuning_outputs_and_deterministic_seed(tmp_path: Path) -> None:
    rows = [
        {
            "take_id": "t1",
            "physical_object_id": "obj_a",
            "expected_superclass": "BALL_GOOD",
            "split": "train",
            "feature_sphericity_3d": 0.9,
            "feature_eccentricity": 0.2,
            "feature_flatness": 0.3,
            "feature_edge_roughness": 2.0,
            "max_height_mm": 40.0,
            "p95_height_mm": 20.0,
            "feature_volume_proxy_mm3": 20000.0,
        },
        {
            "take_id": "t2",
            "physical_object_id": "obj_b",
            "expected_superclass": "BALL_SCRAP",
            "split": "validation",
            "feature_sphericity_3d": 0.6,
            "feature_eccentricity": 0.9,
            "feature_flatness": 0.0,
            "feature_edge_roughness": 15.0,
            "max_height_mm": 30.0,
            "p95_height_mm": 15.0,
            "feature_volume_proxy_mm3": 12000.0,
        },
        {
            "take_id": "t3",
            "physical_object_id": "obj_c",
            "expected_superclass": "SCRAP_METAL",
            "split": "test",
            "feature_sphericity_3d": 0.1,
            "max_height_mm": 4.0,
            "p95_height_mm": 3.0,
            "feature_volume_proxy_mm3": 1000.0,
            "feature_flatness": -0.7,
            "feature_eccentricity": 0.95,
            "feature_edge_roughness": 22.0,
        },
    ]
    csv_path = tmp_path / "features.csv"
    _write_features_csv(csv_path, rows)
    out_dir = tmp_path / "tuned"
    args = SimpleNamespace(
        features_csv=csv_path,
        data_dir=tmp_path,
        dataset_id="d1",
        ml_set_id="set1",
        pipeline_id="mining_steel_ball_classification_25d",
        output_dir=out_dir,
        group_column="physical_object_id",
        label_column="expected_label",
        superclass_column="expected_superclass",
        train_split="train",
        validation_split="validation",
        test_split="test",
        seed=42,
        optimize_metric="macro_f1",
        search="random",
        max_evals=30,
        write_config=None,
    )
    first = run_tuning(args)
    second = run_tuning(args)
    assert json.loads((out_dir / "best_rules.json").read_text(encoding="utf-8"))["params"] == first["best_rules"]["params"]
    assert first["best_rules"]["params"] == second["best_rules"]["params"]
    for file_name in [
        "best_rules.json",
        "tuning_report.json",
        "confusion_matrix.csv",
        "per_class_metrics.csv",
        "predictions.csv",
        "feature_ranges.json",
    ]:
        assert (out_dir / file_name).is_file()


def test_missing_feature_handling_uses_diagnostic_prefix(tmp_path: Path) -> None:
    rows = [
        {
            "take_id": "t1",
            "physical_object_id": "obj1",
            "expected_superclass": "BALL_GOOD",
            "split": "train",
            "diag_feature_sphericity_3d": 0.85,
            "diag_feature_eccentricity": 0.3,
            "diag_feature_flatness": 0.2,
            "diag_feature_edge_roughness": 3.0,
            "diag_max_height_mm": 35.0,
            "diag_p95_height_mm": 18.0,
            "diag_feature_volume_proxy_mm3": 18000.0,
        }
    ]
    out = evaluate_rule_params(rows, dict(DEFAULT_PARAMS), split="train")
    assert out["rows"] == 1
    assert out["predictions"][0]["predicted_superclass"] in {"BALL_GOOD", "BALL_SCRAP", "SCRAP_METAL"}


def test_write_config_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    rows = [
        {
            "take_id": "t1",
            "physical_object_id": "obj_a",
            "expected_superclass": "BALL_GOOD",
            "split": "train",
            "feature_sphericity_3d": 0.9,
            "feature_eccentricity": 0.2,
            "feature_flatness": 0.3,
            "feature_edge_roughness": 2.0,
            "max_height_mm": 40.0,
            "p95_height_mm": 20.0,
            "feature_volume_proxy_mm3": 20000.0,
        }
    ]
    csv_path = tmp_path / "features.csv"
    _write_features_csv(csv_path, rows)
    cfg_path = tmp_path / "configs" / "classifiers" / "rules.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        features_csv=csv_path,
        data_dir=tmp_path,
        dataset_id="d1",
        ml_set_id="set1",
        pipeline_id="mining_steel_ball_classification_25d",
        output_dir=tmp_path / "out",
        group_column="physical_object_id",
        label_column="expected_label",
        superclass_column="expected_superclass",
        train_split="train",
        validation_split="validation",
        test_split="test",
        seed=42,
        optimize_metric="macro_f1",
        search="random",
        max_evals=5,
        write_config=cfg_path,
        overwrite=False,
    )
    with pytest.raises(ValueError, match="refusing to overwrite existing rule config"):
        run_tuning(args)
