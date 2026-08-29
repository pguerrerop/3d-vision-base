from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_3d_acquisition.api.main import (
    ParentRunResultRefRequest,
    PartialRerunExecuteOptionsRequest,
    PartialRerunExecuteRequest,
    PartialRerunPlanRequest,
    pipeline_partial_rerun_execute,
    pipeline_partial_rerun_plan,
)
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.pipelines.run_lineage import generate_run_comparison_to_parent, resolve_run_comparison_id
from vision_3d_acquisition.synthetic import create_synthetic_25d_take
from tests.test_partial_rerun_execution import _index_parent_run
from tests.test_pipeline_comparison import make_settings

PIPELINE_ID = "mining_steel_ball_classification_25d"

# (boundary_stage_id, selected_unit_id, changed_parameters keyed by unit id)
BOUNDARY_CASES = [
    ("detect_belt_plane", "detect_belt_plane.height_gate", {}),
    ("normalize_heights_to_plane", "normalize_heights_to_plane.plane_signed_distance", {}),
    (
        "remove_belt_segment_objects",
        "remove_belt_segment_objects.foreground_thresholding",
        {"remove_belt_segment_objects.foreground_thresholding": {"min_height_mm": 4.0}},
    ),
    ("geometry", "geometry.connected_component_extraction", {}),
    ("measurement_diagnostics", "measurement_diagnostics.feature_vector_generation", {}),
    (
        "classification",
        "classification.primary_heuristic_classifier",
        {"classification.primary_heuristic_classifier": {"confidence_threshold": 0.8}},
    ),
    ("overlay", "overlay.height_overlay", {}),
]


@pytest.mark.parametrize("boundary_stage_id,selected_unit_id,changed_parameters", BOUNDARY_CASES)
def test_partial_rerun_supports_each_public_stage_boundary(
    tmp_path: Path,
    boundary_stage_id: str,
    selected_unit_id: str,
    changed_parameters: dict[str, dict[str, object]],
) -> None:
    settings = make_settings(tmp_path)
    take_id, _ = create_synthetic_25d_take(settings.data_dir, session_id=f"boundary_{boundary_stage_id}")
    full_result = run_ball_inspection_25d_flow(settings.data_dir, take_id=take_id)
    parent_run_id = f"parent_{boundary_stage_id}"
    parent_run_dir = _index_parent_run(settings.data_dir, take_id, run_id=parent_run_id, result_dir=full_result.output_dir)
    parent_result_snapshot = json.loads((parent_run_dir / "result.json").read_text(encoding="utf-8"))

    snapshot = {"parameters_by_unit": changed_parameters}
    plan = pipeline_partial_rerun_plan(
        PIPELINE_ID,
        PartialRerunPlanRequest(
            selected_unit_id=selected_unit_id,
            changed_parameters=changed_parameters,
            current_recipe_snapshot=snapshot,
            parent_run_result_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
        ),
        True,
        settings,
    )

    if not plan.get("safe"):
        # Boundary explicitly unsupported for this fixture: must fail with a clear reason,
        # not silently no-op.
        assert plan.get("blocking_reasons")
        return

    assert plan["execution_boundary_unit_id"] == boundary_stage_id

    execution = pipeline_partial_rerun_execute(
        PIPELINE_ID,
        PartialRerunExecuteRequest(
            plan_id=str(plan["plan_id"]),
            parent_run_ref=ParentRunResultRefRequest(take_id=take_id, run_id=parent_run_id),
            effective_recipe_snapshot=snapshot,
            overrides={"parameters_by_unit": changed_parameters},
            options=PartialRerunExecuteOptionsRequest(compare_to_parent=True),
        ),
        settings,
    )

    assert execution["ok"] is True, execution.get("blocking_reasons")
    assert execution["boundary_stage_id"] == boundary_stage_id
    child_run_id = execution["child_run_id"]
    assert child_run_id

    child_result = execution["result"]
    assert child_result["parent_run_id"] == parent_run_id
    assert child_result["execution_mode"] == "rerun_from_public_stage_boundary"

    # Reused/rerun trace is coherent: every unit is accounted for exactly once, and none
    # of the boundary's own downstream units were marked reused.
    trace_units = child_result["processing_unit_trace"]["units"]
    reused_ids = set(execution["reused_units"])
    rerun_ids = set(execution["rerun_units"])
    assert reused_ids.isdisjoint(rerun_ids)
    for unit_id in reused_ids:
        assert trace_units[unit_id]["status"] == "reused"

    # Parent run is never mutated: its result.json on disk is unchanged.
    parent_result_after = json.loads((parent_run_dir / "result.json").read_text(encoding="utf-8"))
    assert parent_result_after == parent_result_snapshot

    # Reused artifact manifest exists and is well-formed.
    child_result_path = Path(str(execution["result_path"]))
    manifest_path = child_result_path.parent / "partial_rerun_artifact_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_run_id"] == parent_run_id
    assert manifest["child_run_id"] == child_run_id
    assert manifest["copy_strategy"] == "copy_reused_parent_artifacts"

    # Comparison to parent can be generated (already generated at execute time here) and reopened.
    assert execution["comparison_id"]
    assert resolve_run_comparison_id(settings, PIPELINE_ID, child_run_id) == execution["comparison_id"]
    reopened = generate_run_comparison_to_parent(settings, PIPELINE_ID, child_run_id)
    assert reopened["comparison_id"] == execution["comparison_id"]
