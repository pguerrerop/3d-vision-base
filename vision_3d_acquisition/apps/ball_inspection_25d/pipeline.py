from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.contracts.result import ProcessingPipelineInfo, ProcessingProfiling
from vision_3d_acquisition.pipelines.registry import default_pipeline_info, get_pipeline
from vision_3d_acquisition.pipelines.processing_units import (
    PROCESSING_UNIT_REGISTRY_VERSION,
    build_processing_unit_trace,
    build_recipe_snapshot,
    processing_unit_contract_fingerprint,
)
from vision_3d_acquisition.poc.summary import build_calibration_diagnostics, build_poc_run_summary
from vision_3d_acquisition.vision_core.acquisition.sources import FileSource
from vision_3d_acquisition.vision_core.pipelines import PipelineContext, PipelineRunner
from vision_3d_acquisition.vision_core.pipelines.stages import (
    LoadCaptureStage,
    SerializeProcessingResultStage,
    _legacy_timing_from_profiling,
)
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
from vision_3d_acquisition.vision_core.serialization import write_processing_outputs


@dataclass
class BallInspection25DResult:
    take_id: str
    output_dir: Path
    result_payload: dict
    profiling: dict


def finalize_ball_inspection_25d_result_payload(
    *,
    pipeline_result: Any,
    data_dir: Path,
    stage_params: dict[str, Any] | None,
    recipe_version_id: str | None = None,
    recipe_metadata: dict[str, Any] | None = None,
    source_id: str | None = None,
    acquisition_group_id: str | None = None,
    calibration_profile_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    resolved_stage_params = stage_params or {}
    processing_result = pipeline_result.artifacts["processing_result"]
    processing_result.profiling = ProcessingProfiling.model_validate(pipeline_result.profiling)
    processing_result.timing_ms = _legacy_timing_from_profiling(processing_result.profiling)
    processing_result.processing_engine = "native"
    processing_result.processing_pipeline = ProcessingPipelineInfo.model_validate(
        default_pipeline_info("mining_steel_ball_classification_25d")
    )
    processing_result.output_modalities = ["heightmap", "debug_image"]

    output_dir = pipeline_result.artifacts["output_dir"]
    metadata = pipeline_result.artifacts.get("metadata") or {}
    processing_result.calibration_diagnostics = build_calibration_diagnostics(
        processing_result.model_dump(mode="json"),
        metadata=metadata,
    )
    processing_result.poc_summary = build_poc_run_summary(
        processing_result.model_dump(mode="json"),
        metadata=metadata,
        output_dir=output_dir,
        engine="native",
    )

    result_payload = processing_result.model_dump(mode="json")
    result_payload["recipe_version_id"] = recipe_version_id
    result_payload["source_id"] = source_id
    result_payload["acquisition_group_id"] = acquisition_group_id
    result_payload["calibration_profile_id"] = calibration_profile_id
    result_payload["config_snapshot_hash"] = hashlib.sha256(
        json.dumps(
            {"pipeline": "mining_steel_ball_classification_25d", "stage_params": resolved_stage_params},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result_payload["stage_params"] = resolved_stage_params
    processing_units = list((result_payload.get("processing_pipeline") or {}).get("processing_units") or [])
    result_payload["recipe_snapshot"] = build_recipe_snapshot(
        pipeline_id="mining_steel_ball_classification_25d",
        pipeline_version=str((result_payload.get("processing_pipeline") or {}).get("version") or "1.0"),
        registry_version=PROCESSING_UNIT_REGISTRY_VERSION,
        contract_fingerprint=processing_unit_contract_fingerprint(processing_units),
        units=processing_units,
        stage_params=resolved_stage_params,
        recipe_version_id=str((recipe_metadata or {}).get("recipe_id") or recipe_version_id or ""),
        recipe_name=str((recipe_metadata or {}).get("name") or "") or None,
        recipe_version=int((recipe_metadata or {}).get("version")) if isinstance((recipe_metadata or {}).get("version"), int) else None,
        enabled_units=[str(item.get("id")) for item in processing_units if isinstance(item, dict) and item.get("id")],
        provenance={
            "source": "run_ball_inspection_25d_flow",
            "selected_recipe_version_id": recipe_version_id,
            "recipe_manager_recipe_id": (recipe_metadata or {}).get("recipe_id"),
        },
        artifact_policy=(recipe_metadata or {}).get("artifact_policy") if isinstance((recipe_metadata or {}).get("artifact_policy"), dict) else None,
        calibration_snapshot_reference=(
            (recipe_metadata or {}).get("calibration_snapshot_reference")
            if isinstance((recipe_metadata or {}).get("calibration_snapshot_reference"), dict)
            else None
        ),
    )
    result_payload["processing_unit_trace"] = build_processing_unit_trace(
        pipeline_id="mining_steel_ball_classification_25d",
        units=processing_units,
        artifacts=list(result_payload.get("artifacts") or []),
        stage_params=resolved_stage_params,
        runtime_trace=pipeline_result.artifacts.get("runtime_trace_payload")
        if isinstance(pipeline_result.artifacts.get("runtime_trace_payload"), dict)
        else pipeline_result.artifacts.get("context_runtime_trace")
        if isinstance(pipeline_result.artifacts.get("context_runtime_trace"), dict)
        else None,
    )
    return result_payload, output_dir


def run_ball_inspection_25d_flow(
    data_dir: Path,
    *,
    take_id: str | None = None,
    recipe_version_id: str | None = None,
    recipe_metadata: dict[str, Any] | None = None,
    source_id: str | None = None,
    acquisition_group_id: str | None = None,
    calibration_profile_id: str | None = None,
    stage_params: dict[str, Any] | None = None,
) -> BallInspection25DResult:
    source = FileSource(data_dir)
    output_root = data_dir / "processed"
    context = PipelineContext()
    context.set_artifact("pipeline_name", "mining_steel_ball_classification_25d")
    context.set_artifact("pipeline_required_modalities", ["heightmap"])

    stage_params = stage_params or {}
    detect_params = dict(stage_params.get("detect_belt_plane") or {})
    segment_params = dict(stage_params.get("remove_belt_segment_objects") or {})
    normalize_params = dict(stage_params.get("normalize_heights_to_plane") or {})
    known_object_params = dict(stage_params.get("known_object_25d") or {})
    measurement_params = dict(stage_params.get("measurement") or {})
    diagnostics_params = dict(stage_params.get("measurement_diagnostics") or {})
    classify_params = dict(stage_params.get("classify_25d") or {})
    pipeline_cfg = get_pipeline("mining_steel_ball_classification_25d") or {}
    classifier_cfg = pipeline_cfg.get("classifier") if isinstance(pipeline_cfg.get("classifier"), dict) else {}
    pipeline_rule_set = classifier_cfg.get("rule_set") if isinstance(classifier_cfg.get("rule_set"), dict) else {}
    pipeline_rules_path = pipeline_rule_set.get("path")
    if pipeline_rules_path and "classifier_rules_pipeline_path" not in classify_params:
        classify_params["classifier_rules_pipeline_path"] = str(pipeline_rules_path)
    context.set_artifact("stage_params.detect_belt_plane", detect_params)
    context.set_artifact("stage_params.remove_belt_segment_objects", segment_params)
    context.set_artifact("stage_params.normalize_heights_to_plane", normalize_params)
    context.set_artifact("stage_params.known_object_25d", known_object_params)
    context.set_artifact("stage_params.measurement", measurement_params)
    context.set_artifact("stage_params.measurement_diagnostics", diagnostics_params)
    context.set_artifact("stage_params.classify_25d", classify_params)
    runner = PipelineRunner(
        stages=[
            LoadCaptureStage(source=source, take_id=take_id, output_root=output_root),
            LoadHeightmapCaptureStage(),
            ApplyCalibration25DStage(),
            DetectBeltPlaneStage(**_stage_kwargs(DetectBeltPlaneStage, detect_params)),
            NormalizeHeightsToPlaneStage(**_stage_kwargs(NormalizeHeightsToPlaneStage, normalize_params)),
            RemoveBeltAndSegmentObjectsStage(**_stage_kwargs(RemoveBeltAndSegmentObjectsStage, segment_params)),
            ExtractConnectedComponentsStage(),
            FitObjectGeometryStage(),
            ComputeHeightMetricsStage(**_stage_kwargs(ComputeHeightMetricsStage, measurement_params)),
            ValidateKnownObjectScale25DStage(),
            ComputeMeasurementDiagnosticsStage(**_stage_kwargs(ComputeMeasurementDiagnosticsStage, diagnostics_params)),
            ClassifyMiningBall25DStage(**_stage_kwargs(ClassifyMiningBall25DStage, classify_params)),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ]
    )
    pipeline_result = runner.run(context)
    pipeline_result.artifacts["context_runtime_trace"] = context.runtime_trace.to_payload()
    result_payload, output_dir = finalize_ball_inspection_25d_result_payload(
        pipeline_result=pipeline_result,
        data_dir=data_dir,
        stage_params=stage_params,
        recipe_version_id=recipe_version_id,
        recipe_metadata=recipe_metadata,
        source_id=source_id,
        acquisition_group_id=acquisition_group_id,
        calibration_profile_id=calibration_profile_id,
    )

    write_processing_outputs(
        data_dir,
        output_dir,
        result_payload,
        runtime_message="25D heightmap pipeline active.",
    )
    return BallInspection25DResult(
        take_id=str(result_payload.get("take_id") or take_id or ""),
        output_dir=output_dir,
        result_payload=result_payload,
        profiling=pipeline_result.profiling,
    )


def _stage_kwargs(stage_cls: type[Any], params: dict[str, Any]) -> dict[str, Any]:
    annotations = getattr(stage_cls, "__annotations__", {}) or {}
    allowed = set(annotations.keys())
    return {k: v for k, v in params.items() if k in allowed}
