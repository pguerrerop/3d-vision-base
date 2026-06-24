from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_25d_rules import run


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_evaluate_rules_outputs_and_compare(tmp_path: Path) -> None:
    rows = [
        {
            "take_id": "t1",
            "physical_object_id": "obj1",
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
            "physical_object_id": "obj2",
            "expected_superclass": "SCRAP_METAL",
            "split": "test",
            "feature_sphericity_3d": 0.1,
            "feature_eccentricity": 0.9,
            "feature_flatness": -0.8,
            "feature_edge_roughness": 18.0,
            "max_height_mm": 4.0,
            "p95_height_mm": 3.0,
            "feature_volume_proxy_mm3": 1500.0,
        },
    ]
    features = tmp_path / "features.csv"
    _write_csv(features, rows)

    base_cfg = tmp_path / "default_rules.json"
    base_cfg.write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "default",
                "params": {"good_min_sphericity": 0.8},
            }
        ),
        encoding="utf-8",
    )
    tuned_cfg = tmp_path / "tuned_rules.json"
    tuned_cfg.write_text(
        json.dumps(
            {
                "classifier_id": "mining_steel_ball_classification_25d_rules",
                "version": "tuned",
                "params": {"good_min_sphericity": 0.7},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "eval_out"
    args = SimpleNamespace(
        features_csv=features,
        rules_config=base_cfg,
        compare_rules_config=tuned_cfg,
        output_dir=out_dir,
        group_column="physical_object_id",
        label_column="expected_label",
        superclass_column="expected_superclass",
        train_split="train",
        validation_split="validation",
        test_split="test",
        seed=42,
    )
    report = run(args)
    assert report["rows_used"] == 2
    for file_name in [
        "evaluation_report.json",
        "confusion_matrix.csv",
        "per_class_metrics.csv",
        "predictions.csv",
        "misclassified.csv",
        "disagreements.csv",
    ]:
        assert (out_dir / file_name).is_file()
