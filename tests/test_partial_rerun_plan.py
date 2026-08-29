from __future__ import annotations

from typing import Any

from vision_3d_acquisition.api.main import ParentRunResultRefRequest, PartialRerunPlanRequest, pipeline_partial_rerun_plan
from vision_3d_acquisition.pipelines.partial_rerun_plan import list_partial_rerun_plans, plan_partial_rerun
from vision_3d_acquisition.pipelines.processing_units import processing_unit_contract_fingerprint
from vision_3d_acquisition.pipelines.registry import list_processing_unit_definitions
from vision_3d_acquisition.processing.status_index import append_process_run_index
from tests.test_pipeline_comparison import make_settings, write_result


def _units() -> list[dict[str, Any]]:
    return list_processing_unit_definitions("mining_steel_ball_classification_25d")


def _all_output_artifact_ids(units: list[dict[str, Any]]) -> list[str]:
    artifact_ids: set[str] = set()
    for unit in units:
        for output in unit.get("outputs") or []:
            if isinstance(output, dict) and output.get("artifact_id"):
                artifact_ids.add(str(output["artifact_id"]))
    return sorted(artifact_ids)


def _parent_run_result(
    units: list[dict[str, Any]],
    *,
    drop_artifacts: set[str] | None = None,
    fingerprint: str | None = None,
    failed_unit_ids: set[str] | None = None,
) -> dict[str, Any]:
    output_artifacts = _all_output_artifact_ids(units)
    dropped = drop_artifacts or set()
    trace_units: dict[str, Any] = {}
    for unit in units:
        unit_id = str(unit.get("id") or "")
        outputs = [
            str(item.get("artifact_id"))
            for item in unit.get("outputs") or []
            if isinstance(item, dict) and item.get("artifact_id")
        ]
        trace_units[unit_id] = {
            "unit_id": unit_id,
            "stage_id": str(unit.get("stage_id") or ""),
            "parent_id": unit.get("parent_id"),
            "status": "failed" if unit_id in (failed_unit_ids or set()) else "completed",
            "output_artifacts": outputs,
            "input_artifacts": [],
            "parameters_used": {},
            "metrics": {},
            "diagnostics": {},
            "warnings": [],
            "errors": [],
            "trace_source": "runtime_unit_callbacks",
            "trace_precision": "unit_level",
        }
    return {
        "take_id": "take-1",
        "source_id": "source-1",
        "calibration_profile_id": "cal-1",
        "artifacts": [
            {"artifact_id": artifact_id, "path": f"{artifact_id}.png"}
            for artifact_id in output_artifacts
            if artifact_id not in dropped
        ],
        "recipe_snapshot": {
            "pipeline_id": "mining_steel_ball_classification_25d",
            "processing_unit_contract_fingerprint": fingerprint or processing_unit_contract_fingerprint(units),
            "calibration_snapshot_reference": {"calibration_profile_id": "cal-1"},
        },
        "processing_unit_trace": {
            "trace_source": "runtime_unit_callbacks",
            "trace_precision": "unit_level",
            "units": trace_units,
        },
    }


def test_segmentation_change_reuses_upstream_and_invalidates_downstream() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "remove_belt_segment_objects.foreground_thresholding",
        {"remove_belt_segment_objects.foreground_thresholding": {"min_height_mm": 2.0}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units),
        units,
    )
    assert plan["safe"] is True
    assert plan["stages_to_reuse"] == ["input", "detect_belt_plane", "normalize_heights_to_plane"]
    assert plan["stages_to_rerun"] == [
        "remove_belt_segment_objects",
        "geometry",
        "measurement",
        "measurement_diagnostics",
        "classification",
        "overlay",
    ]


def test_detect_reference_change_invalidates_detect_and_all_downstream() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "detect_belt_plane.height_gate",
        {"detect_belt_plane.height_gate": {"gradient_threshold_percentile": 75.0}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units),
        units,
    )
    assert plan["safe"] is True
    assert plan["stages_to_reuse"] == ["input"]
    assert plan["stages_to_rerun"][0] == "detect_belt_plane"
    assert plan["stages_to_rerun"][-1] == "overlay"


def test_classification_change_reuses_upstream_through_measurement() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "classification.primary_heuristic_classifier",
        {"classification.primary_heuristic_classifier": {"confidence_threshold": 0.8}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units),
        units,
    )
    assert plan["safe"] is True
    assert plan["stages_to_reuse"] == [
        "input",
        "detect_belt_plane",
        "normalize_heights_to_plane",
        "remove_belt_segment_objects",
        "geometry",
        "measurement",
        "measurement_diagnostics",
    ]
    assert plan["stages_to_rerun"] == ["classification", "overlay"]


def test_missing_required_artifact_blocks_partial_rerun() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "remove_belt_segment_objects.foreground_thresholding",
        {"remove_belt_segment_objects.foreground_thresholding": {"min_height_mm": 2.0}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units, drop_artifacts={"normalized_heightmap"}),
        units,
    )
    assert plan["safe"] is False
    assert "normalized_heightmap" in plan["required_missing_artifacts"]
    assert plan["suggested_mode"] == "full_run"


def test_contract_fingerprint_mismatch_blocks_reuse() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "classification.primary_heuristic_classifier",
        {"classification.primary_heuristic_classifier": {"confidence_threshold": 0.8}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units, fingerprint="deadbeef"),
        units,
    )
    assert plan["safe"] is False
    assert plan["contract_fingerprint_match"] is False
    assert any("fingerprint" in reason.lower() for reason in plan["blocking_reasons"])


def test_logical_container_stage_root_is_not_direct_rerun_target() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "remove_belt_segment_objects",
        {},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units),
        units,
    )
    assert plan["safe"] is False
    assert any("logical container" in reason.lower() for reason in plan["blocking_reasons"])


def test_optional_reflectance_unit_does_not_block_default_heightmap_run() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "detect_belt_plane.roi",
        {"detect_belt_plane.roi": {"plane_fit_roi": {"type": "rect"}}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units),
        units,
    )
    assert plan["safe"] is True
    assert "input.reflectance_rgb" in plan["units_to_reuse"]


def test_failed_upstream_trace_falls_back_to_full_rerun() -> None:
    units = _units()
    plan = plan_partial_rerun(
        "mining_steel_ball_classification_25d",
        "classification.primary_heuristic_classifier",
        {"classification.primary_heuristic_classifier": {"confidence_threshold": 0.8}},
        {"processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units)},
        _parent_run_result(units, failed_unit_ids={"measurement.height_metrics"}),
        units,
    )
    assert plan["safe"] is False
    assert plan["suggested_mode"] == "full_run"
    assert any("not trustworthy" in reason.lower() for reason in plan["blocking_reasons"])
    assert plan["plan_quality"] == "unsafe"
    assert plan["confidence"] == "low"
    assert plan["explanation"]["fallback_reason"] is not None


def test_partial_rerun_api_returns_segmentation_plan(tmp_path) -> None:
    settings = make_settings(tmp_path)
    units = _units()
    parent = _parent_run_result(units)
    current_snapshot = {
        "processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units),
        "parameters_by_unit": {
            "remove_belt_segment_objects.foreground_thresholding": {"min_height_mm": 4.0},
        },
        "calibration_snapshot_reference": {"calibration_profile_id": "cal-1"},
    }
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="remove_belt_segment_objects.foreground_thresholding",
            current_recipe_snapshot=current_snapshot,
            parent_run_result=parent,
        ),
        settings,
    )
    assert plan["safe"] is True
    assert plan["execution_boundary_unit_id"] == "remove_belt_segment_objects"
    assert plan["stages_to_reuse"] == ["input", "detect_belt_plane", "normalize_heights_to_plane"]
    assert plan["plan_quality"] in {"exact", "conservative"}
    assert plan["confidence"] in {"high", "medium"}
    assert plan["explanation"]["boundary_reason"]


def test_partial_rerun_api_falls_back_on_incompatible_fingerprint(tmp_path) -> None:
    settings = make_settings(tmp_path)
    units = _units()
    take_id = "take_plan_incompatible"
    run_dir = settings.data_dir / "processes" / "runs" / "run_plan_incompatible"
    parent = _parent_run_result(units)
    write_result(run_dir / "result.json", parent)
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_25d",
        run_id="run_plan_incompatible",
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at="2026-07-03T10:00:00Z",
        pipeline_id="mining_steel_ball_classification_25d",
    )
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="classification.primary_heuristic_classifier",
            current_recipe_snapshot={
                "processing_unit_contract_fingerprint": "deadbeef",
                "parameters_by_unit": {"classification.primary_heuristic_classifier": {"confidence_threshold": 0.8}},
                "calibration_snapshot_reference": {"calibration_profile_id": "cal-1"},
            },
            parent_run_result_ref=ParentRunResultRefRequest(take_id=take_id, run_id="run_plan_incompatible"),
        ),
        settings,
    )
    assert plan["safe"] is False
    assert plan["fallback_mode"] == "full_run"
    assert plan["plan_quality"] == "unsafe"
    assert plan["confidence"] == "low"
    assert plan["explanation"]["summary"]


def test_partial_rerun_api_missing_parent_run_returns_unsafe_plan(tmp_path) -> None:
    settings = make_settings(tmp_path)
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="classification.primary_heuristic_classifier",
            current_recipe_snapshot={"processing_unit_contract_fingerprint": "deadbeef"},
            parent_run_result_ref=ParentRunResultRefRequest(take_id="missing_take", run_id="missing_run"),
        ),
        settings,
    )
    assert plan["safe"] is False
    assert plan["fallback_mode"] == "full_run"
    assert any("could not be resolved" in reason.lower() for reason in plan["blocking_reasons"])


def test_partial_rerun_api_mismatched_pipeline_ref_returns_unsafe_plan(tmp_path) -> None:
    settings = make_settings(tmp_path)
    take_id = "take_wrong_pipeline"
    run_dir = settings.data_dir / "processes" / "runs" / "run_wrong_pipeline"
    write_result(run_dir / "result.json", {
        "take_id": take_id,
        "processing_pipeline": {"id": "wrong_pipeline"},
        "artifacts": [],
    })
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="instance_wrong",
        run_id="run_wrong_pipeline",
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at="2026-07-03T10:00:00Z",
        pipeline_id="wrong_pipeline",
    )
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="classification.primary_heuristic_classifier",
            parent_run_result_ref=ParentRunResultRefRequest(take_id=take_id, run_id="run_wrong_pipeline"),
        ),
        settings,
    )
    assert plan["safe"] is False
    assert plan["fallback_mode"] == "full_run"
    assert any("could not be resolved" in reason.lower() for reason in plan["blocking_reasons"])


def test_partial_rerun_plan_persist_and_list(tmp_path) -> None:
    settings = make_settings(tmp_path)
    units = _units()
    plan = pipeline_partial_rerun_plan(
        "mining_steel_ball_classification_25d",
        PartialRerunPlanRequest(
            selected_unit_id="classification.primary_heuristic_classifier",
            current_recipe_snapshot={
                "processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units),
                "parameters_by_unit": {"classification.primary_heuristic_classifier": {"confidence_threshold": 0.8}},
                "calibration_snapshot_reference": {"calibration_profile_id": "cal-1"},
            },
            parent_run_result=_parent_run_result(units),
        ),
        True,
        settings,
    )
    assert plan["plan_id"]
    listed = list_partial_rerun_plans(settings.data_dir, "mining_steel_ball_classification_25d")
    assert any(item["plan_id"] == plan["plan_id"] for item in listed)
