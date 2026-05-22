from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from vision_3d_acquisition.contracts.result import ProcessingPipelineInfo, ProcessingProfiling
from vision_3d_acquisition.pipelines.registry import default_pipeline_info
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


def run_ball_inspection_25d_flow(
    data_dir: Path,
    *,
    take_id: str | None = None,
    recipe_version_id: str | None = None,
    source_id: str | None = None,
    acquisition_group_id: str | None = None,
    calibration_profile_id: str | None = None,
) -> BallInspection25DResult:
    source = FileSource(data_dir)
    output_root = data_dir / "processed"
    context = PipelineContext()
    context.set_artifact("pipeline_name", "mining_steel_ball_classification_25d")
    context.set_artifact("pipeline_required_modalities", ["heightmap"])

    runner = PipelineRunner(
        stages=[
            LoadCaptureStage(source=source, take_id=take_id, output_root=output_root),
            LoadHeightmapCaptureStage(),
            ApplyCalibration25DStage(),
            DetectBeltPlaneStage(),
            NormalizeHeightsToPlaneStage(),
            RemoveBeltAndSegmentObjectsStage(),
            ExtractConnectedComponentsStage(),
            FitObjectGeometryStage(),
            ComputeHeightMetricsStage(),
            ValidateKnownObjectScale25DStage(),
            ClassifyMiningBall25DStage(),
            Generate25DOverlaysStage(),
            SerializeProcessingResultStage(),
        ]
    )
    pipeline_result = runner.run(context)
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
        json.dumps({"pipeline": "mining_steel_ball_classification_25d"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    write_processing_outputs(
        data_dir,
        output_dir,
        result_payload,
        runtime_message="25D heightmap pipeline active.",
    )
    return BallInspection25DResult(
        take_id=str(processing_result.take_id),
        output_dir=output_dir,
        result_payload=result_payload,
        profiling=pipeline_result.profiling,
    )
