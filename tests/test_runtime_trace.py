from __future__ import annotations

import pytest

from vision_3d_acquisition.pipelines.processing_units import build_processing_unit_trace
from vision_3d_acquisition.pipelines.runtime_trace import ProcessingUnitTraceRecorder


def test_runtime_trace_helper_records_completed_unit() -> None:
    recorder = ProcessingUnitTraceRecorder()
    with recorder.trace_unit("detect_belt_plane.roi", stage_id="detect_belt_plane", parent_id="detect_belt_plane") as trace:
        trace.add_input_artifact("raw_heightmap_preview")
        trace.add_output_artifact("plane_fit_roi_mask")
        trace.add_parameter("plane_fit_roi", {"type": "rect"})
        trace.add_metric("roi_pixels", 42)
    entry = recorder.to_payload()["unit_results"]["detect_belt_plane.roi"]
    assert entry["status"] == "completed"
    assert entry["input_artifacts"] == ["raw_heightmap_preview"]
    assert entry["output_artifacts"] == ["plane_fit_roi_mask"]
    assert entry["metrics"]["roi_pixels"] == 42
    assert isinstance(entry["duration_ms"], int)


def test_runtime_trace_helper_records_failed_unit_and_reraises() -> None:
    recorder = ProcessingUnitTraceRecorder()
    with pytest.raises(RuntimeError, match="boom"):
        with recorder.trace_unit("classification.primary_heuristic_classifier", stage_id="classification", parent_id="classification"):
            raise RuntimeError("boom")
    entry = recorder.to_payload()["unit_results"]["classification.primary_heuristic_classifier"]
    assert entry["status"] == "failed"
    assert entry["errors"]


def test_runtime_trace_helper_records_skipped_unit() -> None:
    recorder = ProcessingUnitTraceRecorder()
    with recorder.trace_unit("detect_belt_plane.stripe_filter", stage_id="detect_belt_plane", parent_id="detect_belt_plane") as trace:
        trace.skip("disabled")
    entry = recorder.to_payload()["unit_results"]["detect_belt_plane.stripe_filter"]
    assert entry["status"] == "skipped"
    assert entry["diagnostics"]["skip_reason"] == "disabled"


def test_build_processing_unit_trace_merges_runtime_and_best_effort() -> None:
    units = [
        {
            "id": "remove_belt_segment_objects.morphology_cleanup",
            "stage_id": "remove_belt_segment_objects",
            "parent_id": "remove_belt_segment_objects",
            "kind": "substage",
            "parameters": [{"id": "morphology_kernel", "default": 5}],
            "artifacts": [{"artifact_id": "cleaned_object_mask", "role": "diagnostic"}],
            "inputs": [{"artifact_id": "normalized_height_threshold_mask"}],
            "outputs": [{"artifact_id": "cleaned_object_mask"}],
            "diagnostics": [],
        },
        {
            "id": "measurement_diagnostics.known_object_validation",
            "stage_id": "measurement_diagnostics",
            "parent_id": "measurement_diagnostics",
            "kind": "substage",
            "parameters": [{"id": "enabled", "default": False}],
            "artifacts": [],
            "inputs": [],
            "outputs": [],
            "diagnostics": [],
        },
    ]
    recorder = ProcessingUnitTraceRecorder()
    with recorder.trace_unit("remove_belt_segment_objects.morphology_cleanup", stage_id="remove_belt_segment_objects", parent_id="remove_belt_segment_objects") as trace:
        trace.add_parameter("morphology_kernel", 7)
        trace.add_output_artifact("cleaned_object_mask")
        trace.add_metric("cleaned_mask_pixels", 11)
    payload = build_processing_unit_trace(
        pipeline_id="mining_steel_ball_classification_25d",
        units=units,
        artifacts=[{"artifact_id": "cleaned_object_mask"}],
        stage_params={"remove_belt_segment_objects": {"morphology_kernel": 5}, "known_object_25d": {"enabled": True}},
        runtime_trace=recorder.to_payload(),
    )
    assert payload["trace_source"] == "mixed"
    runtime_entry = payload["unit_results"]["remove_belt_segment_objects.morphology_cleanup"]
    inferred_entry = payload["unit_results"]["measurement_diagnostics.known_object_validation"]
    assert runtime_entry["trace_source"] == "runtime_unit_callbacks"
    assert runtime_entry["parameters_used"]["morphology_kernel"] == 7
    assert inferred_entry["trace_source"] == "best_effort_artifact_registry"
    assert inferred_entry["status"] in {"inferred", "not_emitted"}
    summary = payload["trace_summary"]
    assert summary["total_units"] == 2
    assert summary["runtime_traced_units"] == 1
    assert summary["inferred_units"] == 1
    assert summary["trace_coverage_percent"] == 50.0
    assert summary["coverage_by_stage"]["remove_belt_segment_objects"]["runtime_traced_units"] == 1
    assert summary["coverage_by_stage"]["measurement_diagnostics"]["inferred_units"] == 1
