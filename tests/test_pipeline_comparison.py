from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.comparison import compare_pipeline_sources
from vision_3d_acquisition.pipelines.recipes import RecipeService
from vision_3d_acquisition.processing.status_index import append_process_run_index


def make_settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_mask(path: Path, rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rows, dtype=np.uint8) * 255, mode="L").save(path)


def test_compare_recipe_snapshots_groups_parameter_changes_by_unit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = RecipeService(settings)
    recipe_a = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="Recipe A",
        stage_params={"remove_belt_segment_objects": {"min_height_mm": 2.0}},
    )
    recipe_b = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="Recipe B",
        stage_params={"remove_belt_segment_objects": {"min_height_mm": 5.0}},
    )

    comparison = compare_pipeline_sources(
        settings,
        pipeline_id="mining_steel_ball_classification_25d",
        left={"type": "recipe", "recipe_id": recipe_a["recipe_id"], "version": recipe_a["version"]},
        right={"type": "recipe", "recipe_id": recipe_b["recipe_id"], "version": recipe_b["version"]},
    )

    assert comparison["summary"]["parameter_changes"] >= 1
    unit = comparison["units"]["remove_belt_segment_objects"]
    assert any(row["parameter_id"] == "min_height_mm" and row["status"] == "changed" for row in unit["parameter_diff"])


def test_compare_run_snapshots_reports_classification_and_artifact_changes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id = "take_compare"
    run_a_dir = settings.data_dir / "processes" / "runs" / "run_a"
    run_b_dir = settings.data_dir / "processes" / "runs" / "run_b"
    write_mask(run_a_dir / "final_object_mask.png", [[1, 1], [0, 0]])
    write_mask(run_b_dir / "final_object_mask.png", [[1, 1], [1, 0]])
    result_a = {
        "take_id": take_id,
        "processed_at": "2026-07-01T10:00:00Z",
        "status": "completed",
        "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
            {"artifact_id": "segmentation_overlay", "stage_id": "remove_belt_segment_objects", "kind": "overlay", "title": "Overlay", "path": "segmentation_overlay.png", "preview_available": True},
        ],
        "objects": [{"object_id": 1}],
        "classification": {"label": "accept", "superclass": "good", "confidence": 0.8},
    }
    result_b = {
        "take_id": take_id,
        "processed_at": "2026-07-01T10:05:00Z",
        "status": "completed",
        "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 4.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
        ],
        "objects": [{"object_id": 1}, {"object_id": 2}],
        "classification": {"label": "reject", "superclass": "bad", "confidence": 0.6},
    }
    write_result(run_a_dir / "result.json", result_a)
    write_result(run_b_dir / "result.json", result_b)
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d",
        run_id="run_a",
        pipeline_family="25d",
        status="completed",
        run_dir=run_a_dir,
        created_at="2026-07-01T10:00:00Z",
        pipeline_id="mining_steel_ball_classification_25d",
    )
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d",
        run_id="run_b",
        pipeline_family="25d",
        status="completed",
        run_dir=run_b_dir,
        created_at="2026-07-01T10:05:00Z",
        pipeline_id="mining_steel_ball_classification_25d",
    )

    comparison = compare_pipeline_sources(
        settings,
        pipeline_id="mining_steel_ball_classification_25d",
        left={"type": "run", "take_id": take_id, "run_id": "run_a"},
        right={"type": "run", "take_id": take_id, "run_id": "run_b"},
    )

    assert comparison["summary"]["classification_changed"] is True
    assert comparison["summary"]["left_object_count"] == 1
    assert comparison["summary"]["right_object_count"] == 2
    unit = comparison["units"]["remove_belt_segment_objects"]
    artifact_row = next(row for row in unit["artifact_diff"] if row["artifact_id"] == "final_object_mask")
    assert artifact_row["status"] == "changed"
    assert artifact_row["diff_available"] is True
    assert artifact_row["pixel_diff"]["added_pixels"] == 1
    assert artifact_row["pixel_diff"]["removed_pixels"] == 0
    assert artifact_row["diff_artifacts"]["overlay"]["path"].endswith("final_object_mask_diff_overlay.png")


def test_compare_run_snapshots_reports_identical_mask_iou(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id = "take_identical"
    run_a_dir = settings.data_dir / "processes" / "runs" / "same_a"
    run_b_dir = settings.data_dir / "processes" / "runs" / "same_b"
    write_mask(run_a_dir / "final_object_mask.png", [[1, 0], [0, 1]])
    write_mask(run_b_dir / "final_object_mask.png", [[1, 0], [0, 1]])
    base_payload = {
        "take_id": take_id,
        "processed_at": "2026-07-01T10:00:00Z",
        "status": "completed",
        "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
        ],
        "objects": [],
    }
    write_result(run_a_dir / "result.json", base_payload)
    write_result(run_b_dir / "result.json", {**base_payload, "processed_at": "2026-07-01T10:02:00Z"})
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="same_a", pipeline_family="25d", status="completed", run_dir=run_a_dir, created_at="2026-07-01T10:00:00Z", pipeline_id="mining_steel_ball_classification_25d")
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="same_b", pipeline_family="25d", status="completed", run_dir=run_b_dir, created_at="2026-07-01T10:02:00Z", pipeline_id="mining_steel_ball_classification_25d")
    comparison = compare_pipeline_sources(settings, pipeline_id="mining_steel_ball_classification_25d", left={"type": "run", "take_id": take_id, "run_id": "same_a"}, right={"type": "run", "take_id": take_id, "run_id": "same_b"})
    row = next(row for row in comparison["units"]["remove_belt_segment_objects"]["artifact_diff"] if row["artifact_id"] == "final_object_mask")
    assert row["pixel_diff"]["changed_pixels"] == 0
    assert row["pixel_diff"]["iou"] == 1.0


def test_compare_run_snapshots_handles_dimension_mismatch_safely(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id = "take_mismatch"
    run_a_dir = settings.data_dir / "processes" / "runs" / "mismatch_a"
    run_b_dir = settings.data_dir / "processes" / "runs" / "mismatch_b"
    write_mask(run_a_dir / "final_object_mask.png", [[1, 0], [0, 1]])
    write_mask(run_b_dir / "final_object_mask.png", [[1, 0, 1], [0, 1, 0]])
    payload = {
        "take_id": take_id,
        "processed_at": "2026-07-01T10:00:00Z",
        "status": "completed",
        "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
        ],
        "objects": [],
    }
    write_result(run_a_dir / "result.json", payload)
    write_result(run_b_dir / "result.json", {**payload, "processed_at": "2026-07-01T10:02:00Z"})
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="mismatch_a", pipeline_family="25d", status="completed", run_dir=run_a_dir, created_at="2026-07-01T10:00:00Z", pipeline_id="mining_steel_ball_classification_25d")
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="mismatch_b", pipeline_family="25d", status="completed", run_dir=run_b_dir, created_at="2026-07-01T10:02:00Z", pipeline_id="mining_steel_ball_classification_25d")
    comparison = compare_pipeline_sources(settings, pipeline_id="mining_steel_ball_classification_25d", left={"type": "run", "take_id": take_id, "run_id": "mismatch_a"}, right={"type": "run", "take_id": take_id, "run_id": "mismatch_b"})
    row = next(row for row in comparison["units"]["remove_belt_segment_objects"]["artifact_diff"] if row["artifact_id"] == "final_object_mask")
    assert row["diff_available"] is False
    assert row["diff_reason"] == "dimension_mismatch"
    assert comparison["warnings"]


def test_compare_run_snapshots_handles_unreadable_artifact_safely(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id = "take_unreadable"
    run_a_dir = settings.data_dir / "processes" / "runs" / "unread_a"
    run_b_dir = settings.data_dir / "processes" / "runs" / "unread_b"
    write_mask(run_a_dir / "final_object_mask.png", [[1, 0], [0, 1]])
    (run_b_dir / "final_object_mask.png").parent.mkdir(parents=True, exist_ok=True)
    (run_b_dir / "final_object_mask.png").write_bytes(b"not_an_image")
    payload = {
        "take_id": take_id,
        "processed_at": "2026-07-01T10:00:00Z",
        "status": "completed",
        "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
        ],
        "objects": [],
    }
    write_result(run_a_dir / "result.json", payload)
    write_result(run_b_dir / "result.json", {**payload, "processed_at": "2026-07-01T10:02:00Z"})
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="unread_a", pipeline_family="25d", status="completed", run_dir=run_a_dir, created_at="2026-07-01T10:00:00Z", pipeline_id="mining_steel_ball_classification_25d")
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="unread_b", pipeline_family="25d", status="completed", run_dir=run_b_dir, created_at="2026-07-01T10:02:00Z", pipeline_id="mining_steel_ball_classification_25d")
    comparison = compare_pipeline_sources(settings, pipeline_id="mining_steel_ball_classification_25d", left={"type": "run", "take_id": take_id, "run_id": "unread_a"}, right={"type": "run", "take_id": take_id, "run_id": "unread_b"})
    row = next(row for row in comparison["units"]["remove_belt_segment_objects"]["artifact_diff"] if row["artifact_id"] == "final_object_mask")
    assert row["diff_available"] is False
    assert str(row["diff_reason"]).startswith("unreadable:")


def test_compare_current_to_recipe_handles_current_stage_params(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = RecipeService(settings)
    recipe = service.create_recipe(
        "mining_steel_ball_classification_25d",
        name="Recipe Current",
        stage_params={"detect_belt_plane": {"background_detection_strategy": "low_gradient_surface"}},
    )

    comparison = compare_pipeline_sources(
        settings,
        pipeline_id="mining_steel_ball_classification_25d",
        left={"type": "current", "stage_params": {"detect_belt_plane": {"background_detection_strategy": "low_gradient_depth_plateaus"}}},
        right={"type": "recipe", "recipe_id": recipe["recipe_id"], "version": recipe["version"]},
    )

    unit = comparison["units"]["detect_belt_plane"]
    assert any(row["parameter_id"] == "background_detection_strategy" for row in unit["parameter_diff"])


def test_compare_run_snapshots_uses_runtime_trace_metrics_and_duration(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id = "take_runtime_trace"
    run_a_dir = settings.data_dir / "processes" / "runs" / "runtime_a"
    run_b_dir = settings.data_dir / "processes" / "runs" / "runtime_b"
    payload_base = {
        "take_id": take_id,
        "processed_at": "2026-07-01T10:00:00Z",
        "status": "completed",
        "processing_pipeline": {"id": "mining_steel_ball_classification_25d", "version": "1.0"},
        "stage_params": {"remove_belt_segment_objects": {"min_height_mm": 2.0}},
        "artifacts": [
            {"artifact_id": "final_object_mask", "stage_id": "remove_belt_segment_objects", "kind": "image", "title": "Mask", "path": "final_object_mask.png", "preview_available": True},
        ],
        "objects": [{"object_id": 1}],
    }
    write_mask(run_a_dir / "final_object_mask.png", [[1, 0], [0, 1]])
    write_mask(run_b_dir / "final_object_mask.png", [[1, 1], [0, 1]])
    payload_a = {
        **payload_base,
        "processing_unit_trace": {
            "trace_source": "runtime_unit_callbacks",
            "trace_precision": "unit_level",
            "unit_results": {
                "remove_belt_segment_objects.connected_component_preparation": {
                    "unit_id": "remove_belt_segment_objects.connected_component_preparation",
                    "stage_id": "remove_belt_segment_objects",
                    "status": "completed",
                    "duration_ms": 12,
                    "parameters_used": {"min_component_area": 120},
                    "input_artifacts": ["cleaned_object_mask"],
                    "output_artifacts": ["final_object_mask"],
                    "metrics": {"final_object_pixels": 2, "component_count": 1},
                    "diagnostics": {},
                    "warnings": [],
                    "errors": [],
                    "trace_source": "runtime_unit_callbacks",
                    "trace_precision": "unit_level",
                },
            },
        },
    }
    payload_b = {
        **payload_base,
        "processed_at": "2026-07-01T10:03:00Z",
        "processing_unit_trace": {
            "trace_source": "runtime_unit_callbacks",
            "trace_precision": "unit_level",
            "unit_results": {
                "remove_belt_segment_objects.connected_component_preparation": {
                    "unit_id": "remove_belt_segment_objects.connected_component_preparation",
                    "stage_id": "remove_belt_segment_objects",
                    "status": "warning",
                    "duration_ms": 29,
                    "parameters_used": {"min_component_area": 140},
                    "input_artifacts": ["cleaned_object_mask"],
                    "output_artifacts": ["final_object_mask"],
                    "metrics": {"final_object_pixels": 3, "component_count": 2},
                    "diagnostics": {},
                    "warnings": ["extra_component_detected"],
                    "errors": [],
                    "trace_source": "runtime_unit_callbacks",
                    "trace_precision": "unit_level",
                },
            },
        },
    }
    write_result(run_a_dir / "result.json", payload_a)
    write_result(run_b_dir / "result.json", payload_b)
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="runtime_a", pipeline_family="25d", status="completed", run_dir=run_a_dir, created_at="2026-07-01T10:00:00Z", pipeline_id="mining_steel_ball_classification_25d")
    append_process_run_index(settings.data_dir, take_id=take_id, pipeline_instance_id="instance_25d", run_id="runtime_b", pipeline_family="25d", status="completed", run_dir=run_b_dir, created_at="2026-07-01T10:03:00Z", pipeline_id="mining_steel_ball_classification_25d")
    comparison = compare_pipeline_sources(settings, pipeline_id="mining_steel_ball_classification_25d", left={"type": "run", "take_id": take_id, "run_id": "runtime_a"}, right={"type": "run", "take_id": take_id, "run_id": "runtime_b"})
    unit = comparison["units"]["remove_belt_segment_objects.connected_component_preparation"]
    assert any(row["label"] == "Duration (ms)" and row["status"] == "changed" for row in unit["metric_diff"])
    assert any(row["label"] == "Warning count" and row["status"] == "changed" for row in unit["diagnostic_diff"])
    assert any(row["parameter_id"] == "min_component_area" and row["status"] == "changed" for row in unit["parameter_diff"])
