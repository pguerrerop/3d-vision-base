from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_3d_acquisition.poc.exports import export_labeled_dataset_summary, export_object_metrics, write_rows
from vision_3d_acquisition.poc.labels import load_labels, save_labels
from vision_3d_acquisition.poc.summary import build_calibration_diagnostics, build_poc_run_summary, validate_result_payload


def _result_payload(**overrides) -> dict:
    payload = {
        "take_id": "poc_take_001",
        "processed_at": "2026-05-16T18:00:00Z",
        "processing_mode": "real",
        "processing_engine": "native",
        "algorithm_stage": "classification",
        "status": "ok",
        "summary": {"object_count": 1, "ball_count": 1, "non_ball_count": 0, "decision": "accept", "confidence": 0.72},
        "plane_model": [0.0, 0.0, 1.0, 0.0],
        "input_stats": {
            "point_count": 1200,
            "has_colors": False,
            "has_normals": False,
            "min_bound": [-10.0, -10.0, 0.0],
            "max_bound": [10.0, 10.0, 5.0],
            "extent": [20.0, 20.0, 5.0],
            "file_size_bytes": 1234,
        },
        "objects": [
            {
                "object_id": 1,
                "class_name": "ball",
                "confidence": 0.72,
                "point_count": 80,
                "center_mm": [0.0, 0.0, 3.0],
                "dimensions_mm": [6.0, 6.0, 5.8],
                "diameter_estimate_mm": 6.0,
                "bbox_min_mm": [-3.0, -3.0, 0.0],
                "bbox_max_mm": [3.0, 3.0, 6.0],
                "diameter_mm": 6.0,
                "sphericity_score": 0.91,
                "fit_rmse_mm": 0.2,
            }
        ],
        "files": {
            "point_cloud": "point_cloud.ply",
            "input_preview": "input_point_cloud_preview.png",
            "debug_clusters": "debug_clusters.png",
        },
        "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
        "profiling": {
            "total_ms": 6.0,
            "production_ms": 5.0,
            "debug_artifacts_ms": 1.0,
            "io_ms": 1.0,
            "stages": [
                {
                    "name": "PlaneFilterStage",
                    "category": "calibration_filtering",
                    "started_at": 0.0,
                    "ended_at": 1.0,
                    "duration_ms": 1.0,
                },
                {
                    "name": "BallClassificationStage",
                    "category": "classification",
                    "started_at": 1.0,
                    "ended_at": 4.0,
                    "duration_ms": 3.0,
                },
            ],
        },
        "error": None,
        "calibration_id": "cal_001",
        "calibration_file": "config/calibrations/cal_001.json",
        "plane_filtering": {
            "input_points": 1000,
            "candidate_foreground_points": 100,
            "rejected_points": 50,
            "kept_objects": 1,
            "rejected_objects": 0,
        },
        "rejected_objects": [],
    }
    payload.update(overrides)
    return payload


def test_poc_summary_generation_and_validation(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed" / "poc_take_001"
    output_dir.mkdir(parents=True)
    for filename in ("point_cloud.ply", "input_point_cloud_preview.png", "debug_clusters.png", "result.json"):
        (output_dir / filename).write_text("x", encoding="utf-8")

    summary = build_poc_run_summary(
        _result_payload(),
        metadata={"source": "test", "encoder_ticks": 10},
        output_dir=output_dir,
    )

    assert summary["take_id"] == "poc_take_001"
    assert summary["engine"] == "native"
    assert summary["calibration"]["display"] == "cal_001.json"
    assert summary["calibration"]["mode"] == "calibrated"
    assert summary["calibration"]["status"] == "calibrated"
    assert summary["status"]["demo_ready"] is True
    assert summary["objects"]["estimated_balls"] == 1
    assert summary["profiling"]["slowest_stage"]["name"] == "BallClassificationStage"
    assert summary["warnings"] == []
    assert validate_result_payload(_result_payload()).take_id == "poc_take_001"


def test_poc_warning_generation() -> None:
    payload = _result_payload(
        input_stats={**_result_payload()["input_stats"], "point_count": 10},
        plane_model=None,
        objects=[],
        calibration_id=None,
        calibration_file=None,
        plane_mode="auto",
        calibration_status="automatic plane estimation only",
    )
    payload["summary"] = {"object_count": 0, "ball_count": 0, "non_ball_count": 0, "decision": "review", "confidence": None}

    summary = build_poc_run_summary(payload)

    assert "low_point_count" in summary["warnings"]
    assert "missing_plane" in summary["warnings"]
    assert "no_objects_detected" in summary["warnings"]
    assert "automatic_plane_estimation_only" in summary["warnings"]


def test_labels_persist_independently_from_result_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    result_dir = data_dir / "processed" / "poc_take_001"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(json.dumps(_result_payload()) + "\n", encoding="utf-8")

    labels = save_labels(data_dir, "poc_take_001", ["ball", "uncertain"], notes="first pass", reviewer="qa")

    assert labels["labels"] == ["ball", "uncertain"]
    assert load_labels(data_dir, "poc_take_001")["notes"] == "first pass"
    assert json.loads((result_dir / "result.json").read_text(encoding="utf-8"))["take_id"] == "poc_take_001"


def test_label_and_object_metric_exports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    result_dir = data_dir / "processed" / "poc_take_001"
    result_dir.mkdir(parents=True)
    (result_dir / "DONE").touch()
    (result_dir / "result.json").write_text(json.dumps(_result_payload()) + "\n", encoding="utf-8")
    save_labels(data_dir, "poc_take_001", ["ball"])

    label_rows = export_labeled_dataset_summary(data_dir)
    object_rows = export_object_metrics(data_dir)
    output = tmp_path / "objects.csv"
    write_rows(output, object_rows)

    assert label_rows[0]["take_id"] == "poc_take_001"
    assert object_rows[0]["diameter_mm"] == 6.0
    assert object_rows[0]["take_labels"] == "ball"
    assert output.read_text(encoding="utf-8").startswith("bbox_max_x_mm")


def test_calibration_diagnostics_flags_tilt() -> None:
    payload = _result_payload(plane_model=[0.5, 0.0, 0.5, 0.0])
    diagnostics = build_calibration_diagnostics(payload, metadata={"encoder_ticks": 1})

    assert "excessive_plane_tilt" in diagnostics["warnings"]


def test_invalid_label_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_labels(tmp_path / "data", "take", ["not-a-label"])
