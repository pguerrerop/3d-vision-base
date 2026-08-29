from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from vision_3d_acquisition.api.pipeline_defaults_25d import merge_with_25d_defaults
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.apps.ball_inspection_25d.pipeline import finalize_ball_inspection_25d_result_payload
from vision_3d_acquisition.pipelines.comparison import compare_pipeline_sources, resolve_pipeline_run_source
from vision_3d_acquisition.pipelines.partial_rerun_plan import plan_partial_rerun
from vision_3d_acquisition.pipelines.processing_units import (
    PROCESSING_UNIT_REGISTRY_VERSION,
    build_processing_unit_trace,
    build_recipe_snapshot,
    recipe_parameters_by_unit,
    stage_params_from_recipe_parameters,
)
from vision_3d_acquisition.pipelines.registry import get_pipeline, list_processing_unit_definitions
from vision_3d_acquisition.processing.status_index import append_process_run_index, update_process_run_index_entry
from vision_3d_acquisition.vision_core.acquisition.sources import FileSource
from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, load_heightmap_npz
from vision_3d_acquisition.vision_core.pipelines import PipelineContext, PipelineRunner
from vision_3d_acquisition.vision_core.pipelines.stages import SerializeProcessingResultStage
from vision_3d_acquisition.vision_core.pipelines.stages_25d import (
    ApplyCalibration25DStage,
    ClassifyMiningBall25DStage,
    ComputeHeightMetricsStage,
    ComputeMeasurementDiagnosticsStage,
    DetectBeltPlaneStage,
    ExtractConnectedComponentsStage,
    FitObjectGeometryStage,
    Generate25DOverlaysStage,
    LoadHeightmapCaptureStage,
    NormalizeHeightsToPlaneStage,
    RemoveBeltAndSegmentObjectsStage,
    ValidateKnownObjectScale25DStage,
)
from vision_3d_acquisition.vision_core.serialization.results import write_processing_outputs, write_result_json


SUPPORTED_PARTIAL_BOUNDARY_STAGES = {
    "detect_belt_plane",
    "normalize_heights_to_plane",
    "remove_belt_segment_objects",
    "geometry",
    "measurement_diagnostics",
    "classification",
    "overlay",
}


def load_partial_rerun_plan(data_dir: Path, plan_id: str) -> dict[str, Any] | None:
    path = data_dir / "partial_rerun_plans" / str(plan_id) / "plan.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def execute_partial_rerun(
    settings: ApiSettings,
    *,
    pipeline_id: str,
    plan_id: str | None = None,
    plan_payload: Mapping[str, Any] | None = None,
    parent_run_ref: Mapping[str, Any] | None = None,
    effective_recipe_snapshot: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    compare_to_parent: bool = True,
) -> dict[str, Any]:
    execution_started_at = datetime.now(UTC)
    if pipeline_id != "mining_steel_ball_classification_25d":
        return _unsafe_execution_response(
            pipeline_id,
            blocking_reasons=[f"Unsupported pipeline for partial rerun execution: {pipeline_id}"],
        )

    persisted_plan = dict(plan_payload or {})
    if plan_id:
        loaded = load_partial_rerun_plan(settings.data_dir, plan_id)
        if loaded is None:
            return _unsafe_execution_response(
                pipeline_id,
                plan_id=plan_id,
                blocking_reasons=[f"Unknown partial rerun plan: {plan_id}"],
            )
        persisted_plan = loaded
    if not persisted_plan:
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=plan_id,
            blocking_reasons=["Partial rerun execution requires a persisted plan_id or direct plan payload."],
        )

    selected_unit_id = str(persisted_plan.get("selected_unit_id") or "")
    if not selected_unit_id:
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=plan_id,
            blocking_reasons=["Partial rerun plan is missing selected_unit_id."],
        )

    parent_ref = dict(parent_run_ref or {})
    if not parent_ref and isinstance(persisted_plan.get("parent_run_ref"), Mapping):
        parent_ref = dict(persisted_plan.get("parent_run_ref") or {})
    take_id = str(parent_ref.get("take_id") or "")
    requested_parent_run_id = str(parent_ref.get("run_id") or "latest")
    if not take_id:
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=plan_id,
            selected_unit_id=selected_unit_id,
            blocking_reasons=["Partial rerun execution requires parent_run_ref.take_id."],
        )

    resolved_parent = resolve_pipeline_run_source(
        settings,
        pipeline_id=pipeline_id,
        take_id=take_id,
        run_id=requested_parent_run_id,
    )
    if resolved_parent is None:
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=plan_id,
            selected_unit_id=selected_unit_id,
            blocking_reasons=[f"Parent run could not be resolved for take {take_id} run {requested_parent_run_id}."],
        )

    units = list_processing_unit_definitions(pipeline_id)
    parent_result = dict(resolved_parent.get("result_payload") or {})
    parent_artifacts = list(resolved_parent.get("artifacts") or [])
    parent_trace = dict(resolved_parent.get("processing_unit_trace") or {})
    parent_root = resolved_parent.get("artifact_root")
    if not isinstance(parent_root, Path):
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=plan_id,
            selected_unit_id=selected_unit_id,
            blocking_reasons=["Parent run artifact root is unavailable."],
        )
    parent_result["artifacts"] = parent_artifacts
    parent_result["processing_unit_trace"] = parent_trace

    merged_snapshot = _merge_effective_recipe_snapshot(
        units,
        parent_snapshot=parent_result.get("recipe_snapshot") if isinstance(parent_result.get("recipe_snapshot"), Mapping) else None,
        requested_snapshot=effective_recipe_snapshot,
        overrides=overrides,
    )
    changed_parameters = dict(persisted_plan.get("changed_parameters") or {})
    recomputed_plan = plan_partial_rerun(
        pipeline_id,
        selected_unit_id,
        changed_parameters,
        merged_snapshot,
        parent_result,
        units,
    )
    if str(recomputed_plan.get("execution_boundary_unit_id") or "") != str(persisted_plan.get("execution_boundary_unit_id") or ""):
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=str(persisted_plan.get("plan_id") or plan_id or ""),
            selected_unit_id=selected_unit_id,
            boundary_unit_id=str(recomputed_plan.get("execution_boundary_unit_id") or ""),
            blocking_reasons=["Persisted partial rerun plan is stale and no longer matches the current execution boundary."],
            plan=recomputed_plan,
        )
    if not bool(recomputed_plan.get("safe")):
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=str(persisted_plan.get("plan_id") or plan_id or ""),
            selected_unit_id=selected_unit_id,
            boundary_unit_id=str(recomputed_plan.get("execution_boundary_unit_id") or ""),
            blocking_reasons=list(recomputed_plan.get("blocking_reasons") or ["Partial rerun plan is not safe to execute."]),
            plan=recomputed_plan,
        )

    boundary_stage_id = str(recomputed_plan.get("execution_boundary_unit_id") or "")
    if boundary_stage_id not in SUPPORTED_PARTIAL_BOUNDARY_STAGES:
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=str(persisted_plan.get("plan_id") or plan_id or ""),
            selected_unit_id=selected_unit_id,
            boundary_unit_id=boundary_stage_id,
            blocking_reasons=[f"Boundary stage {boundary_stage_id} is not supported for partial rerun execution v1."],
            plan=recomputed_plan,
        )

    take_id = str(parent_result.get("take_id") or take_id)
    parent_run_id = str(parent_root.name)
    child_run_id = _new_child_run_id(boundary_stage_id)
    child_output_dir = settings.data_dir / "processes" / "runs" / child_run_id
    child_output_dir.mkdir(parents=True, exist_ok=True)

    effective_stage_params = merge_with_25d_defaults(
        settings.state_dir,
        _snapshot_stage_params(merged_snapshot, units),
    )
    context = _build_partial_context(
        data_dir=settings.data_dir,
        take_id=take_id,
        output_dir=child_output_dir,
        stage_params=effective_stage_params,
    )
    _run_prep_stages(context)
    preload_warning = _preload_boundary_context(
        context=context,
        boundary_stage_id=boundary_stage_id,
        parent_result=parent_result,
        parent_root=parent_root,
    )
    if preload_warning:
        return _unsafe_execution_response(
            pipeline_id,
            plan_id=str(persisted_plan.get("plan_id") or plan_id or ""),
            selected_unit_id=selected_unit_id,
            boundary_unit_id=boundary_stage_id,
            blocking_reasons=[preload_warning],
            plan=recomputed_plan,
        )

    stage_runner = PipelineRunner(stages=_partial_stage_sequence(boundary_stage_id, effective_stage_params))
    pipeline_result = stage_runner.run(context)
    pipeline_result.artifacts["context_runtime_trace"] = context.runtime_trace.to_payload()

    current_snapshot = dict(merged_snapshot)
    provenance = dict(current_snapshot.get("provenance") or {}) if isinstance(current_snapshot.get("provenance"), Mapping) else {}
    provenance.update(
        {
            "source": "partial_rerun_execution",
            "parent_run_id": parent_run_id,
            "execution_mode": "rerun_from_public_stage_boundary",
            "partial_rerun_plan_id": persisted_plan.get("plan_id") or plan_id,
            "boundary_stage_id": boundary_stage_id,
            "changed_unit_ids": list(recomputed_plan.get("changed_unit_ids") or []),
        }
    )
    recipe_metadata = {
        "recipe_id": current_snapshot.get("recipe_id"),
        "name": current_snapshot.get("name"),
        "version": current_snapshot.get("version"),
        "artifact_policy": current_snapshot.get("artifact_policy") if isinstance(current_snapshot.get("artifact_policy"), Mapping) else {},
        "calibration_snapshot_reference": current_snapshot.get("calibration_snapshot_reference") if isinstance(current_snapshot.get("calibration_snapshot_reference"), Mapping) else {},
        "provenance": provenance,
    }
    result_payload, _ = finalize_ball_inspection_25d_result_payload(
        pipeline_result=pipeline_result,
        data_dir=settings.data_dir,
        stage_params=effective_stage_params,
        recipe_version_id=str(current_snapshot.get("recipe_id") or "") or None,
        recipe_metadata=recipe_metadata,
        source_id=parent_result.get("source_id") if isinstance(parent_result.get("source_id"), str) else None,
        acquisition_group_id=parent_result.get("acquisition_group_id") if isinstance(parent_result.get("acquisition_group_id"), str) else None,
        calibration_profile_id=parent_result.get("calibration_profile_id") if isinstance(parent_result.get("calibration_profile_id"), str) else None,
    )

    reused_artifacts, reused_artifact_manifest_entries = _copy_reused_artifacts(
        parent_artifacts=parent_artifacts,
        parent_root=parent_root,
        child_output_dir=child_output_dir,
        reused_stage_ids=list(recomputed_plan.get("stages_to_reuse") or []),
        parent_run_id=parent_run_id,
    )
    rerun_artifact_count = len(list(result_payload.get("artifacts") or []))
    result_payload["artifacts"] = _merge_artifact_lists(reused_artifacts, list(result_payload.get("artifacts") or []))
    result_payload["run_id"] = child_run_id
    result_payload["parent_run_id"] = parent_run_id
    result_payload["execution_mode"] = "rerun_from_public_stage_boundary"
    result_payload["partial_rerun_plan_id"] = persisted_plan.get("plan_id") or plan_id
    result_payload["partial_rerun_plan"] = recomputed_plan
    result_payload["reused_from_parent"] = {
        "run_id": parent_run_id,
        "units": list(recomputed_plan.get("units_to_reuse") or []),
        "artifacts": [str(item.get("artifact_id")) for item in reused_artifacts if isinstance(item, Mapping) and item.get("artifact_id")],
        "strategy": "copied_into_child_run",
    }
    result_payload["recipe_snapshot"] = build_recipe_snapshot(
        pipeline_id=pipeline_id,
        pipeline_version=str((result_payload.get("processing_pipeline") or {}).get("version") or "1.0"),
        registry_version=PROCESSING_UNIT_REGISTRY_VERSION,
        contract_fingerprint=str((current_snapshot.get("processing_unit_contract_fingerprint") or "")),
        units=list((result_payload.get("processing_pipeline") or {}).get("processing_units") or []),
        stage_params=effective_stage_params,
        recipe_version_id=str(current_snapshot.get("recipe_id") or "") or None,
        recipe_name=str(current_snapshot.get("name") or "") or None,
        recipe_version=int(current_snapshot.get("version")) if isinstance(current_snapshot.get("version"), int) else None,
        enabled_units=list(current_snapshot.get("enabled_units") or []),
        provenance=provenance,
        artifact_policy=current_snapshot.get("artifact_policy") if isinstance(current_snapshot.get("artifact_policy"), Mapping) else {},
        calibration_snapshot_reference=current_snapshot.get("calibration_snapshot_reference") if isinstance(current_snapshot.get("calibration_snapshot_reference"), Mapping) else {},
    )
    result_payload["recipe_snapshot"]["parent_recipe_snapshot"] = deepcopy(parent_result.get("recipe_snapshot") or {})
    result_payload["recipe_snapshot"]["changed_parameters"] = deepcopy(recomputed_plan.get("changed_parameters") or {})
    result_payload["recipe_snapshot"]["boundary_stage_id"] = boundary_stage_id

    runtime_trace_payload = dict(context.runtime_trace.to_payload())
    runtime_units = dict(runtime_trace_payload.get("units") or {})
    runtime_units.update(
        _reused_runtime_trace_entries(
            units=units,
            reused_unit_ids=list(recomputed_plan.get("units_to_reuse") or []),
            parent_run_id=parent_run_id,
        )
    )
    runtime_trace_payload["units"] = runtime_units
    runtime_trace_payload["unit_results"] = runtime_units
    result_payload["processing_unit_trace"] = build_processing_unit_trace(
        pipeline_id=pipeline_id,
        units=list((result_payload.get("processing_pipeline") or {}).get("processing_units") or []),
        artifacts=list(result_payload.get("artifacts") or []),
        stage_params=effective_stage_params,
        runtime_trace=runtime_trace_payload,
    )
    trace_summary = dict(result_payload["processing_unit_trace"].get("trace_summary") or {})
    trace_summary.update(
        {
            "reused_units": int(len(list(recomputed_plan.get("units_to_reuse") or []))),
            "rerun_units": int(len(list(recomputed_plan.get("units_to_rerun") or []))),
            "invalidated_units": int(len(list(recomputed_plan.get("units_invalidated") or []))),
        }
    )
    result_payload["processing_unit_trace"]["trace_summary"] = trace_summary

    reused_artifact_bytes = sum(
        int(entry.get("size_bytes") or 0) for entry in reused_artifact_manifest_entries if entry.get("copied")
    )
    execution_duration_ms = int((datetime.now(UTC) - execution_started_at).total_seconds() * 1000)
    result_payload["partial_execution_summary"] = {
        "reused_artifact_count": sum(1 for entry in reused_artifact_manifest_entries if entry.get("copied")),
        "reused_artifact_bytes": reused_artifact_bytes,
        "rerun_artifact_count": rerun_artifact_count,
        "partial_execution_duration_ms": execution_duration_ms,
        "boundary_stage_id": boundary_stage_id,
        "parent_run_id": parent_run_id,
    }

    write_processing_outputs(
        settings.data_dir,
        child_output_dir,
        result_payload,
        runtime_message=f"25D partial rerun active from {boundary_stage_id}.",
    )
    write_result_json(child_output_dir / "result.json", result_payload)
    _write_artifact_manifest(
        child_output_dir=child_output_dir,
        parent_run_id=parent_run_id,
        child_run_id=child_run_id,
        execution_mode="rerun_from_public_stage_boundary",
        reused_artifact_manifest_entries=reused_artifact_manifest_entries,
        rerun_artifact_count=rerun_artifact_count,
        warnings=list(recomputed_plan.get("warnings") or []),
    )

    objects = list(result_payload.get("objects") or [])
    first_object = objects[0] if objects and isinstance(objects[0], Mapping) else {}
    run_summary = {
        "reused_units": int(len(list(recomputed_plan.get("units_to_reuse") or []))),
        "rerun_units": int(len(list(recomputed_plan.get("units_to_rerun") or []))),
        "invalidated_units": int(len(list(recomputed_plan.get("units_invalidated") or []))),
        "changed_parameters": int(len(dict(recomputed_plan.get("changed_parameters") or {}))),
        "classification_label": first_object.get("label") if isinstance(first_object, Mapping) else None,
        "classification_superclass": first_object.get("superclass") if isinstance(first_object, Mapping) else None,
        "object_count": len(objects),
    }
    append_process_run_index(
        settings.data_dir,
        take_id=take_id,
        pipeline_instance_id="partial_rerun_25d",
        run_id=child_run_id,
        pipeline_family="25d",
        status=str(result_payload.get("status") or "completed"),
        run_dir=child_output_dir,
        created_at=str(result_payload.get("processed_at") or datetime.now(UTC).isoformat()),
        pipeline_id=pipeline_id,
        recipe_version_id=str(current_snapshot.get("recipe_id") or "") or None,
        config_snapshot_hash=str(result_payload.get("config_snapshot_hash") or "") or None,
        source_id=str(parent_result.get("source_id") or "") or None,
        acquisition_group_id=str(parent_result.get("acquisition_group_id") or "") or None,
        calibration_profile_id=str(parent_result.get("calibration_profile_id") or "") or None,
        execution_mode="rerun_from_public_stage_boundary",
        parent_run_id=parent_run_id,
        parent_take_id=take_id,
        partial_rerun_plan_id=str(persisted_plan.get("plan_id") or plan_id or "") or None,
        boundary_stage_id=boundary_stage_id,
        boundary_unit_id=boundary_stage_id,
        selected_unit_id=selected_unit_id,
        recipe_id=str(current_snapshot.get("recipe_id") or "") or None,
        run_summary=run_summary,
    )

    comparison_id = None
    if compare_to_parent:
        comparison = compare_pipeline_sources(
            settings,
            pipeline_id=pipeline_id,
            left={"type": "run", "take_id": take_id, "run_id": parent_run_id},
            right={"type": "run", "take_id": take_id, "run_id": child_run_id},
        )
        comparison_id = str(comparison.get("comparison_id") or "") or None
        if comparison_id:
            update_process_run_index_entry(settings.data_dir, child_run_id, comparison_id=comparison_id)

    return {
        "ok": True,
        "safe": True,
        "can_execute": True,
        "pipeline_id": pipeline_id,
        "take_id": take_id,
        "child_run_id": child_run_id,
        "parent_run_id": parent_run_id,
        "execution_mode": "rerun_from_public_stage_boundary",
        "boundary_unit_id": boundary_stage_id,
        "boundary_stage_id": boundary_stage_id,
        "reused_units": list(recomputed_plan.get("units_to_reuse") or []),
        "rerun_units": list(recomputed_plan.get("units_to_rerun") or []),
        "invalidated_units": list(recomputed_plan.get("units_invalidated") or []),
        "comparison_id": comparison_id,
        "result_path": str((child_output_dir / "result.json").resolve()),
        "warnings": list(recomputed_plan.get("warnings") or []),
        "partial_rerun_plan_id": persisted_plan.get("plan_id") or plan_id,
        "partial_rerun_plan": recomputed_plan,
        "result": result_payload,
    }


def _unsafe_execution_response(
    pipeline_id: str,
    *,
    plan_id: str | None = None,
    selected_unit_id: str | None = None,
    boundary_unit_id: str | None = None,
    blocking_reasons: list[str],
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "safe": False,
        "can_execute": False,
        "pipeline_id": pipeline_id,
        "child_run_id": None,
        "parent_run_id": None,
        "execution_mode": "full_run",
        "boundary_unit_id": boundary_unit_id,
        "boundary_stage_id": boundary_unit_id,
        "warnings": list((plan or {}).get("warnings") or []),
        "blocking_reasons": list(blocking_reasons),
        "fallback_mode": "full_run",
        "partial_rerun_plan_id": plan_id,
        "partial_rerun_plan": dict(plan or {}),
        "selected_unit_id": selected_unit_id,
    }


def _snapshot_stage_params(snapshot: Mapping[str, Any], units: list[dict[str, Any]]) -> dict[str, Any]:
    stage_params = dict(snapshot.get("stage_params") or {}) if isinstance(snapshot.get("stage_params"), Mapping) else {}
    if stage_params:
        return stage_params
    params_by_unit = snapshot.get("parameters_by_unit")
    return stage_params_from_recipe_parameters(units, params_by_unit if isinstance(params_by_unit, Mapping) else {})


def _merge_stage_params(base: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = {
        str(key): deepcopy(value)
        for key, value in base.items()
    }
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(merged.get(str(key)), Mapping):
            bucket = dict(merged.get(str(key)) or {})
            bucket.update(dict(value))
            merged[str(key)] = bucket
        else:
            merged[str(key)] = deepcopy(value)
    return merged


def _merge_effective_recipe_snapshot(
    units: list[dict[str, Any]],
    *,
    parent_snapshot: Mapping[str, Any] | None,
    requested_snapshot: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(dict(parent_snapshot or {}))
    if isinstance(requested_snapshot, Mapping):
        for key, value in requested_snapshot.items():
            merged[str(key)] = deepcopy(value)
    base_stage_params = _snapshot_stage_params(merged, units)
    if isinstance(requested_snapshot, Mapping):
        requested_stage_params = _snapshot_stage_params(dict(requested_snapshot), units)
        base_stage_params = _merge_stage_params(base_stage_params, requested_stage_params)
    if isinstance(overrides, Mapping):
        override_stage_params = {}
        if isinstance(overrides.get("stage_params"), Mapping):
            override_stage_params = dict(overrides.get("stage_params") or {})
        elif isinstance(overrides.get("parameters_by_unit"), Mapping):
            override_stage_params = stage_params_from_recipe_parameters(units, overrides.get("parameters_by_unit"))
        else:
            override_stage_params = {str(key): value for key, value in overrides.items() if isinstance(value, Mapping)}
        base_stage_params = _merge_stage_params(base_stage_params, override_stage_params)
    merged["stage_params"] = base_stage_params
    merged["parameters_by_unit"] = recipe_parameters_by_unit(units, base_stage_params)
    return merged


def _build_partial_context(
    *,
    data_dir: Path,
    take_id: str,
    output_dir: Path,
    stage_params: Mapping[str, Any],
) -> PipelineContext:
    source = FileSource(data_dir)
    reference = source.by_id(take_id)
    classify_params = dict(stage_params.get("classify_25d") or {})
    pipeline_cfg = get_pipeline("mining_steel_ball_classification_25d") or {}
    classifier_cfg = pipeline_cfg.get("classifier") if isinstance(pipeline_cfg.get("classifier"), dict) else {}
    pipeline_rule_set = classifier_cfg.get("rule_set") if isinstance(classifier_cfg.get("rule_set"), dict) else {}
    pipeline_rules_path = pipeline_rule_set.get("path")
    if pipeline_rules_path and "classifier_rules_pipeline_path" not in classify_params:
        classify_params["classifier_rules_pipeline_path"] = str(pipeline_rules_path)
    context = PipelineContext()
    context.set_artifact("pipeline_name", "mining_steel_ball_classification_25d")
    context.set_artifact("pipeline_id", "mining_steel_ball_classification_25d")
    context.set_artifact("pipeline_required_modalities", ["heightmap"])
    context.set_artifact("capture_ref", reference)
    context.set_artifact("take_id", reference.take_id)
    context.set_artifact("take_dir", reference.take_dir)
    context.set_artifact("metadata", reference.metadata)
    context.set_artifact("modalities", reference.modalities)
    context.set_artifact("assets", reference.assets)
    context.set_artifact("frame_count", reference.frame_count)
    context.set_artifact("output_dir", output_dir)
    context.set_artifact("stage_params.detect_belt_plane", dict(stage_params.get("detect_belt_plane") or {}))
    context.set_artifact("stage_params.remove_belt_segment_objects", dict(stage_params.get("remove_belt_segment_objects") or {}))
    context.set_artifact("stage_params.normalize_heights_to_plane", dict(stage_params.get("normalize_heights_to_plane") or {}))
    context.set_artifact("stage_params.known_object_25d", dict(stage_params.get("known_object_25d") or {}))
    context.set_artifact("stage_params.classify_25d", classify_params)
    return context


def _run_prep_stages(context: PipelineContext) -> None:
    PipelineRunner(stages=[LoadHeightmapCaptureStage(), ApplyCalibration25DStage()]).run(context)


def _partial_stage_sequence(boundary_stage_id: str, stage_params: Mapping[str, Any]) -> list[Any]:
    detect_params = dict(stage_params.get("detect_belt_plane") or {})
    segment_params = dict(stage_params.get("remove_belt_segment_objects") or {})
    normalize_params = dict(stage_params.get("normalize_heights_to_plane") or {})
    classify_params = dict(stage_params.get("classify_25d") or {})
    stage_map = {
        "detect_belt_plane": [
            DetectBeltPlaneStage(**_stage_kwargs(DetectBeltPlaneStage, detect_params)),
            NormalizeHeightsToPlaneStage(**_stage_kwargs(NormalizeHeightsToPlaneStage, normalize_params)),
            RemoveBeltAndSegmentObjectsStage(**_stage_kwargs(RemoveBeltAndSegmentObjectsStage, segment_params)),
            ExtractConnectedComponentsStage(),
            FitObjectGeometryStage(),
            ComputeHeightMetricsStage(),
            ValidateKnownObjectScale25DStage(),
            ComputeMeasurementDiagnosticsStage(),
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
        "normalize_heights_to_plane": [
            NormalizeHeightsToPlaneStage(**_stage_kwargs(NormalizeHeightsToPlaneStage, normalize_params)),
            RemoveBeltAndSegmentObjectsStage(**_stage_kwargs(RemoveBeltAndSegmentObjectsStage, segment_params)),
            ExtractConnectedComponentsStage(),
            FitObjectGeometryStage(),
            ComputeHeightMetricsStage(),
            ValidateKnownObjectScale25DStage(),
            ComputeMeasurementDiagnosticsStage(),
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
        "remove_belt_segment_objects": [
            RemoveBeltAndSegmentObjectsStage(**_stage_kwargs(RemoveBeltAndSegmentObjectsStage, segment_params)),
            ExtractConnectedComponentsStage(),
            FitObjectGeometryStage(),
            ComputeHeightMetricsStage(),
            ValidateKnownObjectScale25DStage(),
            ComputeMeasurementDiagnosticsStage(),
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
        "geometry": [
            ExtractConnectedComponentsStage(),
            FitObjectGeometryStage(),
            ComputeHeightMetricsStage(),
            ValidateKnownObjectScale25DStage(),
            ComputeMeasurementDiagnosticsStage(),
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
        "measurement_diagnostics": [
            ComputeMeasurementDiagnosticsStage(),
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
        "classification": [
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
        "overlay": [
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ],
    }
    return stage_map[boundary_stage_id]


def _preload_boundary_context(
    *,
    context: PipelineContext,
    boundary_stage_id: str,
    parent_result: Mapping[str, Any],
    parent_root: Path,
) -> str | None:
    if boundary_stage_id == "detect_belt_plane":
        return None
    if boundary_stage_id in {"normalize_heights_to_plane", "remove_belt_segment_objects"}:
        warning = _preload_detect_outputs(context, parent_result, parent_root)
        if warning:
            return warning
    if boundary_stage_id in {"remove_belt_segment_objects", "geometry", "measurement_diagnostics", "classification", "overlay"}:
        warning = _preload_normalize_outputs(context, parent_result, parent_root)
        if warning:
            return warning
    if boundary_stage_id in {"geometry", "measurement_diagnostics", "classification", "overlay"}:
        warning = _preload_segmentation_outputs(context, parent_result, parent_root)
        if warning:
            return warning
    if boundary_stage_id in {"measurement_diagnostics", "classification", "overlay"}:
        _preload_object_outputs(context, parent_result, parent_root)
    return None


def _preload_detect_outputs(context: PipelineContext, parent_result: Mapping[str, Any], parent_root: Path) -> str | None:
    plane_model = parent_result.get("plane_model")
    if isinstance(plane_model, (list, tuple)) and len(plane_model) >= 4:
        context.set_artifact("belt_plane", tuple(float(value) for value in plane_model[:4]))
        context.set_artifact("plane_model", tuple(float(value) for value in plane_model[:4]))
    else:
        plane_json = _load_json_artifact(parent_result, parent_root, "belt_plane")
        equation = plane_json.get("equation") if isinstance(plane_json.get("equation"), Mapping) else {}
        coeffs = [equation.get("a"), equation.get("b"), equation.get("c"), equation.get("d")]
        if any(value is None for value in coeffs):
            return "Parent run is missing belt_plane coefficients required for normalization reuse."
        context.set_artifact("belt_plane", tuple(float(value) for value in coeffs))
        context.set_artifact("plane_model", tuple(float(value) for value in coeffs))
        context.set_artifact(
            "reference_surface_model",
            {
                "type": plane_json.get("reference_surface_model_type") or "plane",
                "plane_coefficients": [float(value) for value in coeffs],
                "constant_z_mm": plane_json.get("constant_z_mm"),
                "model_selection_reason": plane_json.get("model_selection_reason"),
            },
        )
    plane_inlier_mask = _load_bool_mask_artifact(parent_result, parent_root, "plane_inlier_mask")
    if plane_inlier_mask is not None:
        context.set_artifact("plane_inlier_mask", plane_inlier_mask)
    suppression_mask = _load_bool_mask_artifact(parent_result, parent_root, "reference_suppression_mask")
    if suppression_mask is not None:
        context.set_artifact("reference_suppression_mask", suppression_mask)
        context.set_artifact("surface_suppression_mask_array", suppression_mask)
    stripes_mask = _load_bool_mask_artifact(parent_result, parent_root, "belt_stripes_mask")
    if stripes_mask is not None:
        context.set_artifact("belt_stripes_mask_array", stripes_mask)
    return None


def _preload_normalize_outputs(context: PipelineContext, parent_result: Mapping[str, Any], parent_root: Path) -> str | None:
    fallback = parent_root / "normalized_heightmap.npz"
    normalized_path = fallback if fallback.is_file() else None
    if normalized_path is None:
        normalized_path = _artifact_path(parent_result, parent_root, "normalized_heightmap")
        if normalized_path is not None and normalized_path.suffix.lower() != ".npz":
            normalized_path = None
    if normalized_path is None:
        return "Parent run is missing normalized_heightmap required for downstream partial rerun."
    frame = load_heightmap_npz(normalized_path)
    context.set_artifact("normalized_heightmap_mm", np.asarray(frame.z_mm, dtype=np.float32))
    context.set_artifact("height_processing_semantic_field", "height_above_belt")
    return None


def _preload_segmentation_outputs(context: PipelineContext, parent_result: Mapping[str, Any], parent_root: Path) -> str | None:
    mask = _load_bool_mask_artifact(parent_result, parent_root, "final_object_mask")
    if mask is None:
        return "Parent run is missing final_object_mask required for geometry/classification reuse."
    context.set_artifact("segmentation_mask", mask)
    return None


def _preload_object_outputs(context: PipelineContext, parent_result: Mapping[str, Any], parent_root: Path) -> None:
    objects = deepcopy(list(parent_result.get("objects") or []))
    context.set_artifact("objects", objects)
    classification = parent_result.get("classification")
    if isinstance(classification, Mapping):
        context.set_artifact("classification_result_25d", deepcopy(dict(classification)))
    known_object = _load_json_artifact_optional(parent_result, parent_root, "known_object_scale_validation")
    if known_object is not None:
        context.set_artifact("known_object_scale_validation", known_object)


def _artifact_path(result_payload: Mapping[str, Any], artifact_root: Path, artifact_id: str) -> Path | None:
    for artifact in result_payload.get("artifacts") or []:
        if not isinstance(artifact, Mapping):
            continue
        if str(artifact.get("artifact_id") or "") != artifact_id:
            continue
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        path = Path(path_value)
        if path.is_absolute():
            return path if path.is_file() else None
        candidate = (artifact_root / path).resolve()
        return candidate if candidate.is_file() else None
    return None


def _load_json_artifact(result_payload: Mapping[str, Any], artifact_root: Path, artifact_id: str) -> dict[str, Any]:
    path = _artifact_path(result_payload, artifact_root, artifact_id)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_artifact_optional(result_payload: Mapping[str, Any], artifact_root: Path, artifact_id: str) -> dict[str, Any] | None:
    payload = _load_json_artifact(result_payload, artifact_root, artifact_id)
    return payload or None


def _load_bool_mask_artifact(result_payload: Mapping[str, Any], artifact_root: Path, artifact_id: str) -> np.ndarray | None:
    path = _artifact_path(result_payload, artifact_root, artifact_id)
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception:
        return None
    return array > 0


def _copy_reused_artifacts(
    *,
    parent_artifacts: list[dict[str, Any]],
    parent_root: Path,
    child_output_dir: Path,
    reused_stage_ids: list[str],
    parent_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    stage_filter = set(reused_stage_ids)
    for artifact in parent_artifacts:
        if not isinstance(artifact, Mapping):
            continue
        if str(artifact.get("stage_id") or "") not in stage_filter:
            continue
        cloned = deepcopy(dict(artifact))
        artifact_id = str(cloned.get("artifact_id") or "")
        path_value = cloned.get("path")
        source_path: str | None = None
        child_path: str | None = None
        source_exists = False
        was_copied = False
        size_bytes: int | None = None
        if isinstance(path_value, str) and path_value.strip():
            source = Path(path_value)
            if not source.is_absolute():
                source = (parent_root / path_value).resolve()
            source_path = str(source)
            source_exists = source.is_file()
            if source_exists:
                destination = child_output_dir / "reused" / str(cloned.get("stage_id") or "unknown") / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                cloned["path"] = str(destination.relative_to(child_output_dir))
                child_path = cloned["path"]
                was_copied = True
                try:
                    size_bytes = destination.stat().st_size
                except OSError:
                    size_bytes = None
        metadata = dict(cloned.get("metadata") or {})
        metadata["reused_from_parent_run_id"] = parent_run_id
        metadata["reused_from_parent"] = True
        metadata["artifact_copy_strategy"] = "copied_into_child_run"
        cloned["metadata"] = metadata
        copied.append(cloned)
        manifest_entries.append(
            {
                "artifact_id": artifact_id,
                "source_path": source_path,
                "child_path": child_path,
                "source_exists": source_exists,
                "copied": was_copied,
                "size_bytes": size_bytes,
            }
        )
    return copied, manifest_entries


def _write_artifact_manifest(
    *,
    child_output_dir: Path,
    parent_run_id: str,
    child_run_id: str,
    execution_mode: str,
    reused_artifact_manifest_entries: list[dict[str, Any]],
    rerun_artifact_count: int,
    warnings: list[str],
) -> None:
    manifest = {
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
        "execution_mode": execution_mode,
        "copy_strategy": "copy_reused_parent_artifacts",
        "reused_artifacts": reused_artifact_manifest_entries,
        "rerun_artifacts_count": rerun_artifact_count,
        "warnings": list(warnings),
    }
    manifest_path = child_output_dir / "partial_rerun_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _merge_artifact_lists(reused_artifacts: list[dict[str, Any]], current_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for artifact in reused_artifacts:
        artifact_id = str(artifact.get("artifact_id") or "")
        if artifact_id:
            by_id[artifact_id] = artifact
    for artifact in current_artifacts:
        artifact_id = str(artifact.get("artifact_id") or "")
        if artifact_id:
            by_id[artifact_id] = artifact
    return list(by_id.values())


def _reused_runtime_trace_entries(
    *,
    units: list[dict[str, Any]],
    reused_unit_ids: list[str],
    parent_run_id: str,
) -> dict[str, dict[str, Any]]:
    unit_by_id = {
        str(unit.get("id") or ""): unit
        for unit in units
        if isinstance(unit, Mapping) and unit.get("id")
    }
    entries: dict[str, dict[str, Any]] = {}
    for unit_id in reused_unit_ids:
        unit = unit_by_id.get(str(unit_id), {})
        entries[str(unit_id)] = {
            "unit_id": str(unit_id),
            "stage_id": str(unit.get("stage_id") or ""),
            "parent_id": unit.get("parent_id"),
            "status": "reused",
            "execution_role": "reused",
            "source_run_id": parent_run_id,
            "parameters_used": {},
            "input_artifacts": [],
            "output_artifacts": [],
            "metrics": {},
            "diagnostics": {"source_run_id": parent_run_id},
            "warnings": [],
            "errors": [],
            "trace_source": "partial_rerun_reuse",
            "trace_precision": "unit_level",
        }
    return entries


def _stage_kwargs(stage_cls: type[Any], params: dict[str, Any]) -> dict[str, Any]:
    annotations = getattr(stage_cls, "__annotations__", {}) or {}
    allowed = set(annotations.keys())
    return {key: value for key, value in params.items() if key in allowed}


def _new_child_run_id(boundary_stage_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    suffix = boundary_stage_id.replace(".", "_")
    return f"partial_{suffix}_{stamp}"
