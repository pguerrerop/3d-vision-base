from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_3d_acquisition.apps.ball_inspection.stages import (
    BallClassificationStage,
    FitSphereOrEllipseStage,
    StatisticsStage,
)
from vision_3d_acquisition.contracts.result import ProcessingProfiling
from vision_3d_acquisition.processing.pipeline import ProcessingConfig, load_processing_config
from vision_3d_acquisition.poc.summary import build_calibration_diagnostics, build_poc_run_summary
from vision_3d_acquisition.vision_core.acquisition.sources import FileSource
from vision_3d_acquisition.vision_core.pipelines import PipelineContext, PipelineRunner
from vision_3d_acquisition.vision_core.pipelines.stages import (
    ApplyCalibrationStage,
    DecodeHeightmapStage,
    LoadCaptureStage,
    MeasureObjectsStage,
    PlaneFilterStage,
    PreprocessPointCloudStage,
    RenderDebugArtifactsStage,
    SegmentObjectsStage,
    SerializeProcessingResultStage,
    _legacy_timing_from_profiling,
)
from vision_3d_acquisition.vision_core.serialization import write_processing_outputs


@dataclass
class BallInspectionResult:
    take_id: str
    output_dir: Path
    result_payload: dict
    profiling: dict


def run_ball_inspection_flow(
    data_dir: Path,
    *,
    take_id: str | None = None,
    config_path: Path | None = None,
    calibration: Path | None = None,
    calibration_resolution_source: str | None = None,
    skip_debug_images: bool = False,
    prefer_fast_cloud: bool = True,
) -> BallInspectionResult:
    source = FileSource(data_dir)
    output_root = data_dir / "processed"
    config = load_processing_config(config_path) if config_path is not None else ProcessingConfig()
    context = PipelineContext()
    context.set_artifact("calibration_resolution_source", calibration_resolution_source or ("explicit" if calibration else "none"))
    context.set_artifact("pipeline_name", "ball_inspection")
    context.set_artifact("pipeline_required_modalities", ["point_cloud"])
    runner = PipelineRunner(
        stages=[
            LoadCaptureStage(source=source, take_id=take_id, output_root=output_root),
            DecodeHeightmapStage(prefer_fast_cloud=prefer_fast_cloud, render_preview=not skip_debug_images),
            ApplyCalibrationStage(calibration_path=calibration),
            PreprocessPointCloudStage(
                voxel_size=config.voxel_size,
                enable_outlier_removal=config.enable_outlier_removal,
                outlier_nb_neighbors=config.outlier_nb_neighbors,
                outlier_std_ratio=config.outlier_std_ratio,
                calibration_crop_margin_mm=config.calibration_crop_margin_mm,
            ),
            PlaneFilterStage(
                distance_threshold=config.plane_distance_threshold,
                ransac_n=config.plane_ransac_n,
                num_iterations=config.plane_num_iterations,
            ),
            SegmentObjectsStage(
                dbscan_eps=config.dbscan_eps,
                dbscan_min_points=config.dbscan_min_points,
                min_cluster_points=config.min_cluster_points,
            ),
            MeasureObjectsStage(),
            FitSphereOrEllipseStage(),
            BallClassificationStage(),
            StatisticsStage(),
            RenderDebugArtifactsStage(skip_debug_images=skip_debug_images),
            SerializeProcessingResultStage(),
        ]
    )
    pipeline_result = runner.run(context)
    processing_result = pipeline_result.artifacts["processing_result"]
    processing_result.profiling = ProcessingProfiling.model_validate(pipeline_result.profiling)
    processing_result.timing_ms = _legacy_timing_from_profiling(processing_result.profiling)
    processing_result.processing_engine = "native"
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
    write_processing_outputs(
        data_dir,
        output_dir,
        result_payload,
        runtime_message="Ball inspection pipeline active.",
    )
    return BallInspectionResult(
        take_id=str(processing_result.take_id),
        output_dir=output_dir,
        result_payload=result_payload,
        profiling=pipeline_result.profiling,
    )
