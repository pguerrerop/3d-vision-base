from __future__ import annotations

import json
import shutil
from pathlib import Path

from vision_3d_acquisition.api.main import (
    ParentRunResultRefRequest,
    PartialRerunExecuteOptionsRequest,
    PartialRerunExecuteRequest,
    PartialRerunPlanRequest,
    pipeline_partial_rerun_execute,
    pipeline_partial_rerun_plan,
)
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.processing.status_index import (
    append_process_run_index,
    load_process_entries,
    normalize_run_entry,
)
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from tests.test_pipeline_comparison import make_settings


def _index_parent_run(data_dir: Path, take_id: str, *, run_id: str, result_dir: Path) -> Path:
    run_dir = data_dir / "processes" / "runs" / run_id
    shutil.copytree(result_dir, run_dir, dirs_exist_ok=True)
    result_path = run_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["run_id"] = run_id
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_process_run_index(
        data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d",
        run_id=run_id,
        pipeline_family="25d",
        status=str(payload.get("status") or "completed"),
        run_dir=run_dir,
        created_at=str(payload.get("processed_at") or "2026-07-03T12:00:00Z"),
        pipeline_id="mining_steel_ball_classification_25d",
    )
    return run_dir


def test_partial_rerun_execute_creates_child_run_with_reuse_and_compare(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id="partial_rerun_execute")
    full_result = run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)
    _index_parent_run(settings.data_dir, take_id, run_id="parent_segmentation_run", result_dir=full_result.output_dir)

    snapshot = {
        "parameters_by_unit": {
            "remove_belt_segment_objects.foreground_thresholding": {
                "min_height_mm": 4.0,
            },
        },
    }
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="remove_belt_segment_objects.foreground_thresholding",
            changed_parameters=snapshot["parameters_by_unit"],
            current_recipe_snapshot=snapshot,
            parent_run_result_ref=ParentRunResultRefRequest(
                take_id=take_id,
                run_id="parent_segmentation_run",
            ),
        ),
        True,
        settings,
    )

    assert plan["safe"] is True
    assert plan["plan_id"]
    assert plan["execution_boundary_unit_id"] == "remove_belt_segment_objects"

    execution = pipeline_partial_rerun_execute(
        "mining_steel_ball_classification_25d",
        PartialRerunExecuteRequest(
            plan_id=str(plan["plan_id"]),
            parent_run_ref=ParentRunResultRefRequest(
                take_id=take_id,
                run_id="parent_segmentation_run",
            ),
            effective_recipe_snapshot=snapshot,
            overrides={"parameters_by_unit": snapshot["parameters_by_unit"]},
            options=PartialRerunExecuteOptionsRequest(compare_to_parent=True),
        ),
        settings,
    )

    assert execution["ok"] is True
    assert execution["safe"] is True
    assert execution["can_execute"] is True
    assert execution["parent_run_id"] == "parent_segmentation_run"
    assert execution["child_run_id"]
    assert execution["boundary_stage_id"] == "remove_belt_segment_objects"
    assert execution["comparison_id"]

    child_result = execution["result"]
    assert isinstance(child_result, dict)
    assert child_result["parent_run_id"] == "parent_segmentation_run"
    assert child_result["execution_mode"] == "rerun_from_public_stage_boundary"
    assert child_result["partial_rerun_plan_id"] == plan["plan_id"]
    assert child_result["recipe_snapshot"]["boundary_stage_id"] == "remove_belt_segment_objects"
    assert child_result["reused_from_parent"]["strategy"] == "copied_into_child_run"

    trace_units = child_result["processing_unit_trace"]["units"]
    assert trace_units["detect_belt_plane.final_support"]["status"] == "reused"
    assert trace_units["remove_belt_segment_objects.foreground_thresholding"]["status"] in {"completed", "warning"}
    assert trace_units["measurement.height_metrics"]["status"] in {"completed", "warning"}
    assert trace_units["classification.primary_heuristic_classifier"]["status"] in {"completed", "warning"}

    child_result_path = Path(str(execution["result_path"]))
    assert child_result_path.is_file()
    assert (child_result_path.parent / "reused" / "detect_belt_plane").exists()

    index_entries = load_process_entries(settings.data_dir)
    child_entry = next(entry for entry in index_entries if entry.get("run_id") == execution["child_run_id"])
    assert child_entry["execution_mode"] == "rerun_from_public_stage_boundary"
    assert child_entry["parent_run_id"] == "parent_segmentation_run"
    assert child_entry["parent_take_id"] == take_id
    assert child_entry["boundary_stage_id"] == "remove_belt_segment_objects"
    assert child_entry["partial_rerun_plan_id"] == plan["plan_id"]
    assert child_entry["comparison_id"] == execution["comparison_id"]
    assert child_entry["summary"]["reused_units"] == len(plan.get("units_to_reuse") or execution["reused_units"])
    assert child_entry["summary"]["rerun_units"] == len(execution["rerun_units"])
    assert child_entry["summary"]["invalidated_units"] == len(execution["invalidated_units"])

    manifest_path = child_result_path.parent / "partial_rerun_artifact_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_run_id"] == "parent_segmentation_run"
    assert manifest["child_run_id"] == execution["child_run_id"]
    assert manifest["execution_mode"] == "rerun_from_public_stage_boundary"
    assert manifest["copy_strategy"] == "copy_reused_parent_artifacts"
    assert isinstance(manifest["reused_artifacts"], list) and manifest["reused_artifacts"]
    assert isinstance(manifest["rerun_artifacts_count"], int) and manifest["rerun_artifacts_count"] > 0
    assert isinstance(manifest["warnings"], list)
    for entry in manifest["reused_artifacts"]:
        assert set(entry.keys()) >= {"artifact_id", "source_path", "child_path", "source_exists", "copied", "size_bytes"}
        assert entry["artifact_id"]
        if entry["copied"]:
            assert entry["source_exists"] is True
            assert entry["child_path"]
            assert isinstance(entry["size_bytes"], int)

    perf_summary = child_result["partial_execution_summary"]
    assert perf_summary["parent_run_id"] == "parent_segmentation_run"
    assert perf_summary["boundary_stage_id"] == "remove_belt_segment_objects"
    assert perf_summary["reused_artifact_count"] == sum(1 for entry in manifest["reused_artifacts"] if entry["copied"])
    assert perf_summary["reused_artifact_bytes"] >= 0
    assert perf_summary["rerun_artifact_count"] == manifest["rerun_artifacts_count"]
    assert isinstance(perf_summary["partial_execution_duration_ms"], int)
    assert perf_summary["partial_execution_duration_ms"] >= 0


def test_partial_rerun_index_entry_backward_compatible_without_new_fields() -> None:
    legacy_entry = {
        "take_id": "take_1",
        "pipeline_instance_id": "instance_25d",
        "run_id": "legacy_run",
        "pipeline_family": "25d",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00Z",
        "path": "/tmp/legacy_run",
        "pipeline_id": "mining_steel_ball_classification_25d",
    }
    normalized = normalize_run_entry(legacy_entry)
    assert normalized["execution_mode"] == "full_run"
    assert normalized["parent_run_id"] is None
    assert normalized["parent_take_id"] is None
    assert normalized["partial_rerun_plan_id"] is None
    assert normalized["boundary_stage_id"] is None
    assert normalized["comparison_id"] is None
    assert normalized["summary"] == {}


def test_partial_rerun_execute_blocks_when_parent_run_missing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    execution = pipeline_partial_rerun_execute(
        "mining_steel_ball_classification_25d",
        PartialRerunExecuteRequest(
            plan={"selected_unit_id": "classification.primary_heuristic_classifier"},
            parent_run_ref=ParentRunResultRefRequest(
                take_id="missing_take",
                run_id="missing_run",
            ),
            effective_recipe_snapshot={
                "parameters_by_unit": {
                    "classification.primary_heuristic_classifier": {
                        "confidence_threshold": 0.8,
                    },
                },
            },
            options=PartialRerunExecuteOptionsRequest(compare_to_parent=True),
        ),
        settings,
    )

    assert execution["ok"] is False
    assert execution["safe"] is False
    assert execution["can_execute"] is False
    assert execution["fallback_mode"] == "full_run"
    assert any("could not be resolved" in reason.lower() for reason in execution["blocking_reasons"])
